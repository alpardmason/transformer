"""Training utilities for the original Transformer.

Implements the exact training setup from the paper:
- Noam learning rate schedule (Section 5.3)
- Label smoothing (Section 5.4)
- Synthetic data for educational use
"""

import time
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Noam Learning Rate Scheduler (Section 5.3)
# ──────────────────────────────────────────────────────────────────────


class NoamOpt:
    """Noam learning rate schedule from Section 5.3.

        lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Two phases:
      - Linear warmup:  lrate ∝ step              (when step < warmup_steps)
      - Inverse sqrt decay: lrate ∝ 1/sqrt(step)  (when step > warmup_steps)

    This wrapped-optimizer pattern (instead of torch.optim.lr_scheduler) gives
    NoamOpt control over zero_grad() and step() so the training loop doesn't
    need to call optimizer.step() separately.

    The d_model^(-0.5) factor scales the LR based on model size — larger models
    get smaller learning rates because each parameter update has more impact.

    Adam uses beta1=0.9, beta2=0.98, eps=1e-9 (paper §5.3).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int = 512,
        factor: float = 1.0,
        warmup: int = 4000,
    ):
        self.optimizer = optimizer
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self._step = 0
        self._rate = 0.0

    def step(self) -> None:
        """Perform one optimizer step with the current learning rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def rate(self, step: int | None = None) -> float:
        if step is None:
            step = self._step
        # warmup_steps^(-1.5) precomputed to avoid repeated exponentiation
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    @property
    def current_rate(self) -> float:
        return self._rate

    def __repr__(self) -> str:
        return (
            f"NoamOpt(model_size={self.model_size}, warmup={self.warmup}, "
            f"step={self._step}, rate={self._rate:.2e})"
        )


# ──────────────────────────────────────────────────────────────────────
# Label Smoothing (Section 5.4)
# ──────────────────────────────────────────────────────────────────────


class LabelSmoothing(nn.Module):
    """Label smoothing with epsilon = 0.1 (paper §5.4).

    Instead of a hard one-hot target (all probability on the correct class),
    we use a smoothed distribution:
        q(k) = (1 - ε) * 1[k == y]  +  ε / (V - 1) * 1[k != y]

    This prevents the model from becoming overconfident (putting probability 1
    on a single token), which improves generalization and BLEU scores.

    Uses KL divergence loss: KL(q || p) = sum_k q(k) * log(q(k) / p(k)).
    Since q is fixed (no gradient through q), minimizing KL(q||p) is equivalent
    to cross-entropy with soft targets — but KL divergence makes the smoothing
    computation explicit.

    Shapes:
      x:      (B, S, V)  — raw logits from the model
      target: (B, S)     — integer class indices
      output: scalar     — average loss per non-padding token
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """x: (B, S, V), target: (B, S)"""
        vocab_size = x.size(-1)  # V

        # Start with uniform smoothed mass: ε / (V-1) for every class
        # (V-1 because the true class gets confidence instead)
        true_dist = x.new_full((x.size(-1),), self.smoothing / (vocab_size - 1))
        true_dist = true_dist.unsqueeze(0).unsqueeze(0).expand_as(x).clone()
        # true_dist: (B, S, V) — all positions filled with ε/(V-1)

        # Fill the correct class index with confidence = 1 - ε
        true_dist.scatter_(-1, target.unsqueeze(-1), self.confidence)
        # Now: correct class = 1-ε, all others = ε/(V-1) → sums to 1 ✓

        # Positions where target is padding get zero probability everywhere
        mask = target == self.ignore_index
        true_dist = true_dist.masked_fill(mask.unsqueeze(-1), 0.0)

        # KL(q || p) = q * (log q - log p), summed over vocab for each position
        kl = F.kl_div(
            F.log_softmax(x, dim=-1),  # log p(k)
            true_dist,  # q(k)
            reduction="none",
        )  # (B, S, V)
        kl = kl.sum(-1)  # (B, S) — KL per position

        # Mask out padding positions (contribute zero to total)
        kl = kl.masked_fill(mask, 0.0)

        # Average over non-padding tokens (not positions: a long sequence
        # contributes more to the loss than a short one).
        n_tokens = (1 - mask.long()).sum()
        return kl.sum() / n_tokens.clamp(min=1)


# ──────────────────────────────────────────────────────────────────────
# Mask Utilities
# ──────────────────────────────────────────────────────────────────────


def subsequent_mask(size: int) -> torch.Tensor:
    """Create a lower-triangular boolean mask that prevents attending to future tokens.

    Returns (1, size, size) where mask[:, i, j] = True iff j <= i.

    Example for size=4:
        [[T, F, F, F],
         [T, T, F, F],
         [T, T, T, F],
         [T, T, T, T]]

    Uses torch.uint8 for historical compatibility: triu with bool tensors
    behaves inconsistently across PyTorch versions, so we use uint8 ones
    and compare to zero to get a clean boolean result.
    """
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape, dtype=torch.uint8), diagonal=1)
    return mask == 0


def make_std_mask(tgt: torch.Tensor, pad: int = 0) -> torch.Tensor:
    """Create a combined padding + subsequent mask for decoder self-attention.

    Combines two masks via logical AND:
      1. Padding mask (B, 1, 1, S)   — False where tgt == pad
      2. Subsequent mask (1, S, S)   — lower-triangular, prevents looking ahead

    Broadcasting: (B, 1, 1, S) & (1, 1, S, S) → (B, 1, S, S)

    Returns: (B, 1, S_tgt, S_tgt) boolean mask, True = can attend.
    """
    tgt_mask = (tgt != pad).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S_tgt)
    seq_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)  # (1, S_tgt, S_tgt)
    return tgt_mask & seq_mask  # (B, 1, S_tgt, S_tgt)


# ──────────────────────────────────────────────────────────────────────
# Synthetic Data for Educational Training
# ──────────────────────────────────────────────────────────────────────


class SyntheticData:
    """Generate simple data for educational training.

    By default, creates a "copy task": the model must learn to copy the
    input sequence. The special tokens are:
      0 = <pad>, 1 = <bos>, 2 = <eos>
    Vocabulary tokens start from index 3.
    """

    def __init__(
        self,
        vocab_size: int = 40,
        max_len: int = 10,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def generate_batch(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of (src, tgt_in, tgt_out) for the copy task.

        Example with vocab_size=40, max_len=5, tokens=[8,12,3,4] (length 4):

          src:      [8, 12, 3, 4, EOS]              — input with EOS terminator
          tgt_in:   [BOS, 8, 12, 3, 4]              — BOS + content (no EOS)
          tgt_out:  [8, 12, 3, 4, EOS]              — content + EOS (teacher target)

        The model must learn to copy tokens from src to tgt_out. tgt_in is the
        decoder input during teacher forcing — shifted right by one from tgt_out.

        Returns:
          src:      (B, max_len+1) — padded to batch max
          tgt_in:   (B, max_len+1)
          tgt_out:  (B, max_len+1)
        """
        # Random lengths ensure variable-length sequences within the batch
        lengths = torch.randint(1, self.max_len + 1, (batch_size,))
        max_len = int(lengths.max().item())

        tokens = torch.randint(
            low=3, high=self.vocab_size, size=(batch_size, max_len)
        )  # (B, max_len) — only content tokens (3..vocab_size-1)

        # Position mask: mask[b, p] = True iff position p < length of sequence b
        mask = torch.arange(max_len) < lengths.unsqueeze(1)  # (B, max_len)

        # Source: content tokens + EOS, padded on the right
        src = torch.full((batch_size, max_len + 1), self.pad_idx, dtype=torch.long)
        src[:, :max_len] = torch.where(mask, tokens, self.pad_idx)
        src[torch.arange(batch_size), lengths] = self.eos_idx

        # Target input: BOS + content tokens (no EOS) — the decoder input
        tgt_input = torch.full(
            (batch_size, max_len + 1), self.pad_idx, dtype=torch.long
        )
        tgt_out = torch.full((batch_size, max_len + 1), self.pad_idx, dtype=torch.long)

        tgt_input[:, 0] = self.bos_idx
        tgt_input[:, 1 : max_len + 1] = torch.where(mask, tokens, self.pad_idx)

        # Target output: content tokens + EOS — what the model should predict
        tgt_out[:, :max_len] = torch.where(mask, tokens, self.pad_idx)
        tgt_out[torch.arange(batch_size), lengths] = self.eos_idx

        return src, tgt_input, tgt_out


# ──────────────────────────────────────────────────────────────────────
# Arithmetic Data — Model Learns Arithmetic from Digit Sequences
# ──────────────────────────────────────────────────────────────────────


class ArithmeticData:
    """Generate arithmetic expression batches for seq2seq learning.

    The model takes an expression like "1+1" and must decode "=2".
    Numbers are digit-by-digit so the model learns decimal place value.
    Special tokens: 0=PAD, 1=BOS, 2=EOS.

    Token layout (22 tokens total):
      0 = PAD, 1 = BOS, 2 = EOS
      3-12 = digits 0-9
      13 = '+', 14 = '-', 15 = '*', 16 = '/', 17 = '='
      18-21 = reserved

    Operations: +, -, *, /  with operands in [0, max_operand].
    Subtraction ensures a >= b (non-negative result).
    Division uses exact division only: picks b, k, sets a = b*k.
    """

    # Token constants
    NUM_OFFSET = 3       # digit d → token d + 3
    PLUS = 13
    MINUS = 14
    MUL = 15
    DIV = 16
    EQ = 17
    OPS = (PLUS, MINUS, MUL, DIV)
    VOCAB_SIZE = 22

    def __init__(
        self,
        max_len: int = 10,
        max_operand: int = 99,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        self.max_len = max_len
        self.max_operand = max_operand
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    @staticmethod
    def _to_digits(n: int) -> list[int]:
        """Convert integer to list of digit tokens (most-significant first)."""
        if n == 0:
            return [ArithmeticData.NUM_OFFSET]  # token for digit 0
        digits = []
        while n > 0:
            digits.append(n % 10 + ArithmeticData.NUM_OFFSET)
            n //= 10
        return list(reversed(digits))

    def generate_batch(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate (src, tgt_in, tgt_out) for arithmetic task.

        Example: expression 12+7, answer 19
          src:      [1→4, 2→5, +→13, 7→10, EOS→2, PAD…]
          tgt_in:   [BOS→1, =→17, 1→4, 9→12, PAD…]
          tgt_out:  [=→17, 1→4, 9→12, EOS→2, PAD…]

        Returns:
          src:      (B, max_len+1) — digit tokens + op + EOS, padded
          tgt_in:   (B, max_len+1) — BOS + = + digits, padded
          tgt_out:  (B, max_len+1) — = + digits + EOS, padded
        """
        B = batch_size
        max_op = self.max_operand

        # Pick operation per sample uniformly (25% each)
        op_indices = torch.randint(0, 4, (B,))

        # Pick operands per operation
        a_vals = torch.randint(0, max_op + 1, (B,))  # placeholder
        b_vals = torch.randint(0, max_op + 1, (B,))  # placeholder

        for i in range(B):
            op = op_indices[i].item()
            if op == 0:  # addition
                a_vals[i] = torch.randint(0, max_op + 1, (1,))
                b_vals[i] = torch.randint(0, max_op + 1, (1,))
            elif op == 1:  # subtraction: enforce a >= b
                a = torch.randint(0, max_op + 1, (1,))
                b = torch.randint(0, a.item() + 1, (1,))
                a_vals[i], b_vals[i] = a, b
            elif op == 2:  # multiplication
                a_vals[i] = torch.randint(0, max_op + 1, (1,))
                b_vals[i] = torch.randint(0, max_op + 1, (1,))
            else:  # division: exact only. b ∈ [1, max_op], k ∈ [0, max_op//b]
                b = torch.randint(1, max_op + 1, (1,))
                max_k = max_op // b.item()
                k = torch.randint(0, max_k + 1, (1,))
                a_vals[i] = b * k
                b_vals[i] = b

        # Compute answers and operator tokens
        op_tokens = torch.tensor([self.OPS[o] for o in op_indices.tolist()], dtype=torch.long)

        answers = []
        for i in range(B):
            a = a_vals[i].item()
            b = b_vals[i].item()
            op = op_indices[i].item()
            if op == 0:
                answers.append(a + b)
            elif op == 1:
                answers.append(a - b)
            elif op == 2:
                answers.append(a * b)
            else:
                answers.append(a // b if b != 0 else 0)

        # Build src, tgt token sequences as lists then pad
        src_list = []
        tgt_out_list = []
        for i in range(B):
            a_digits = self._to_digits(a_vals[i].item())
            b_digits = self._to_digits(b_vals[i].item())
            ans_digits = self._to_digits(answers[i])

            src_tokens = a_digits + [op_tokens[i].item()] + b_digits + [self.eos_idx]
            tgt_tokens = [self.EQ] + ans_digits + [self.eos_idx]

            src_list.append(src_tokens)
            tgt_out_list.append(tgt_tokens)

        # Find max lengths in batch and pad
        src_max = max(len(s) for s in src_list)
        tgt_max = max(len(t) for t in tgt_out_list)

        src = torch.full((B, src_max), self.pad_idx, dtype=torch.long)
        tgt_out = torch.full((B, tgt_max), self.pad_idx, dtype=torch.long)

        for i in range(B):
            src[i, :len(src_list[i])] = torch.tensor(src_list[i], dtype=torch.long)
            tgt_out[i, :len(tgt_out_list[i])] = torch.tensor(tgt_out_list[i], dtype=torch.long)

        # tgt_in: BOS + tgt_out[:-1]
        tgt_in = torch.full((B, tgt_max), self.pad_idx, dtype=torch.long)
        tgt_in[:, 0] = self.bos_idx
        tgt_in[:, 1:tgt_max] = tgt_out[:, :tgt_max - 1]

        return src, tgt_in, tgt_out

    @staticmethod
    def decode(tokens: list[int]) -> str:
        """Decode a token sequence to a human-readable arithmetic string."""
        chars = []
        for t in tokens:
            if t in (0, 1, 2):  # PAD, BOS, EOS
                continue
            if 3 <= t <= 12:    # digit
                chars.append(str(t - 3))
            elif t == 13:
                chars.append('+')
            elif t == 14:
                chars.append('-')
            elif t == 15:
                chars.append('*')
            elif t == 16:
                chars.append('/')
            elif t == 17:
                chars.append('=')
            else:
                chars.append('?')
        return ''.join(chars) if chars else '<empty>'


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────


def run_epoch(
    data_iter: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    model: nn.Module,
    loss_fn: nn.Module,
    opt: NoamOpt | None = None,
    desc: str = "train",
    pad_idx: int = 0,
) -> float:
    """Run one training epoch. If opt is None, runs in eval mode (no gradients).

    Uses teacher forcing: the full target sequence (tgt_in) is fed to the decoder
    in one pass, and the model predicts all output tokens in parallel. At inference
    time this isn't available, so greedy_decode/beam_search generate autoregressively.

    Loss is accumulated weighted by non-padding tokens so the reported loss is
    per-token average, not per-sequence average.
    """
    total_loss = 0.0
    n_tokens = 0
    model.train() if opt is not None else model.eval()

    for src, tgt_in, tgt_out in tqdm(data_iter, desc=desc):
        device = next(model.parameters()).device
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        if opt is not None:
            opt.zero_grad()

        # src_mask: (B, 1, 1, S_src) — False where src is padding
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        # tgt_mask: (B, 1, S_tgt, S_tgt) — combines padding + subsequent masks
        tgt_mask = make_std_mask(tgt_in, pad_idx)

        logits = model(src, tgt_in, src_mask, tgt_mask)  # (B, S_tgt, V)
        loss = loss_fn(logits, tgt_out)

        if opt is not None:
            loss.backward()
            opt.step()

        # Weight by number of non-padding tokens for correct per-token averaging
        total_loss += loss.item() * (tgt_out != pad_idx).sum().item()
        n_tokens += (tgt_out != pad_idx).sum().item()

    return total_loss / n_tokens


def run_epoch_steps(
    model: nn.Module,
    data_fn: Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    opt: NoamOpt,
    n_steps: int,
    batch_size: int,
    pad_idx: int = 0,
    device: str | torch.device = "cpu",
    print_every: int = 100,
) -> None:
    """Train a model for a fixed number of steps (used for quick educational training).

    Unlike run_epoch, this generates batches on-the-fly via data_fn instead of
    iterating over a fixed dataset. This is simpler for synthetic data where
    we can generate unlimited fresh examples (no risk of overfitting to a
    finite dataset — the model sees new tokens every step).

    The loss is unweighted at each step but the running average is computed
    over total non-padding tokens for accurate reporting.
    """
    model.train()
    total_loss = 0.0
    n_tokens = 0
    start = time.time()

    for step in range(1, n_steps + 1):
        src, tgt_in, tgt_out = data_fn(batch_size)
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        opt.zero_grad()

        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = make_std_mask(tgt_in, pad_idx)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = loss_fn(logits, tgt_out)

        loss.backward()
        opt.step()

        total_loss += loss.item()
        n_tokens += (tgt_out != pad_idx).sum().item()

        if step % print_every == 0:
            avg_loss = total_loss / n_tokens
            elapsed = time.time() - start
            print(
                f"Step {step:6d} | Loss: {avg_loss:.4f} "
                f"| LR: {opt.current_rate:.2e} "
                f"| Tokens/s: {n_tokens / elapsed:.0f}"
            )
