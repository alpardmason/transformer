"""Training utilities for the original Transformer — MLX implementation.

Implements the exact training setup from the paper:
- Noam learning rate schedule (Section 5.3)
- Label smoothing (Section 5.4)
- Synthetic data for educational use

MLX vs PyTorch — key differences annotated throughout:
  - Gradient computation: mx.value_and_grad(model, loss_fn) instead of
    loss.backward(). No zero_grad() needed — gradients are computed fresh.
  - Lazy evaluation: must call mx.eval() periodically to avoid graph growth.
  - mx.where(cond, a, b) replaces masked_fill / scatter_.
  - mx.put_along_axis replaces scatter_ for indexed assignment.
  - mx.random.randint / mx.random.normal for random tensors.
  - No torch.no_grad() context needed — model.eval() sets dropout to identity.
  - mx.argmax(axis=...) replaces .argmax(dim=...).
"""

import math
import time
from collections.abc import Callable, Iterator
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

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

    MLX difference — no wrapped optimizer:
      In PyTorch, NoamOpt wraps the optimizer and its step() calls both set
      the LR and optimizer.step(). In MLX, NoamOpt holds the MLX optimizer
      and provides step(model, grads) which sets the learning_rate before
      calling optimizer.update(model, grads).

      MLX optimizers expose .learning_rate as a settable property, so we
      update it directly before each update. This is simpler than PyTorch's
      param_groups iteration.

    Adam uses beta1=0.9, beta2=0.98, eps=1e-9 (paper §5.3).
    """

    def __init__(
        self,
        model_size: int = 512,
        factor: float = 1.0,
        warmup: int = 4000,
        optimizer: optim.Optimizer | None = None,
    ):
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0.0

    def set_optimizer(self, optimizer: optim.Optimizer) -> None:
        self.optimizer = optimizer

    def step(self, model: nn.Module, grads: dict) -> None:
        """Perform one optimizer step with the current learning rate.

        MLX: optimizer.update(model, grads) applies the gradients and updates
        parameters. learning_rate is set on the optimizer before the update.
        """
        self._step += 1
        rate = self.rate()
        # MLX: set learning_rate directly on the optimizer (simpler than PyTorch's
        # param_groups iteration)
        self.optimizer.learning_rate = rate
        self._rate = rate
        self.optimizer.update(model, grads)

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


class LabelSmoothing:
    """Label smoothing with epsilon = 0.1 (paper §5.4).

    Instead of a hard one-hot target (all probability on the correct class),
    we use a smoothed distribution:
        q(k) = (1 - ε) * 1[k == y]  +  ε / (V - 1) * 1[k != y]

    This prevents the model from becoming overconfident (putting probability 1
    on a single token), which improves generalization and BLEU scores.

    Loss — cross-entropy with soft targets (equivalent to KL when q is fixed):
      The paper describes label smoothing via KL divergence:
        KL(q || p) = sum_k q(k) * (log q(k) - log p(k))
      PyTorch implements this with F.kl_div. MLX has no built-in kl_div, but
      because q does not depend on model parameters, sum_k q(k) log q(k) is a
      constant. Minimizing KL is therefore the same as minimizing:
        -sum_k q(k) * log p(k)
      i.e. cross-entropy with the smoothed distribution q as the target.

    MLX differences:
      - Plain callable, not nn.Module (no learnable parameters).
      - mx.where(cond, a, b) replaces masked_fill (functional; no in-place ops).
      - No mx.log_softmax — compute mx.log(mx.softmax(...)) manually.

    Shapes:
      x:      (B, S, V)  — raw logits from the model
      target: (B, S)     — integer class indices
      output: scalar     — average loss per non-padding token
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def __call__(self, x: mx.array, target: mx.array) -> mx.array:
        """x: (B, S, V), target: (B, S)"""
        vocab_size = x.shape[-1]  # V

        # Build smooth target distribution via broadcast comparison.
        # MLX: no scatter_ / put_along_axis with float values. Instead, use
        # mx.where with a boolean mask identifying the correct-class positions.
        #   is_correct: (B, S, V) — True at the index matching target[b,s]
        #   q(k) = confidence if k == target else ε/(V-1)
        is_correct = mx.arange(vocab_size)[None, None, :] == target[..., None]  # (B, S, V)
        true_dist = mx.where(
            is_correct,
            mx.full(x.shape, self.confidence),
            mx.full(x.shape, self.smoothing / (vocab_size - 1)),
        )

        # Positions where target is padding get zero probability everywhere
        mask = target == self.ignore_index  # (B, S)
        # MLX: mx.where replaces masked_fill
        true_dist = mx.where(mask[..., None], mx.zeros_like(true_dist), true_dist)

        # Cross-entropy with soft targets: -sum_k q(k) * log p(k)
        # Equivalent to KL(q || p) for optimization because q is fixed — the
        # term sum_k q(k) log q(k) has zero gradient w.r.t. model parameters.
        # MLX: no mx.log_softmax — compute log after softmax manually
        log_preds = mx.log(mx.softmax(x, axis=-1))       # (B, S, V) — log p(k)
        ce = -(true_dist * log_preds).sum(axis=-1)       # (B, S) — loss per position

        # Mask out padding positions (contribute zero to total)
        ce = mx.where(mask, mx.zeros_like(ce), ce)

        # Average over non-padding tokens (not positions: a long sequence
        # contributes more to the loss than a short one).
        n_tokens = mx.maximum((~mask).astype(mx.int32).sum(), 1)
        return ce.sum() / n_tokens


# ──────────────────────────────────────────────────────────────────────
# Mask Utilities
# ──────────────────────────────────────────────────────────────────────


def subsequent_mask(size: int) -> mx.array:
    """Create a lower-triangular boolean mask that prevents attending to future tokens.

    Returns (1, size, size) where mask[:, i, j] = True iff j <= i.

    Example for size=4:
        [[T, F, F, F],
         [T, T, F, F],
         [T, T, T, F],
         [T, T, T, T]]

    MLX: mx.triu(x, k=1) replaces torch.triu(x, diagonal=1).
    No dtype gymnastics needed — mx.ones returns float32, triu with int works fine.
    """
    attn_shape = (1, size, size)
    mask = mx.triu(mx.ones(attn_shape), k=1)
    return mask == 0  # Lower triangular positions = True (can attend)


def make_std_mask(
    tgt: mx.array, pad: int = 0
) -> mx.array:
    """Create a combined padding + subsequent mask for decoder self-attention.

    Combines two masks via logical AND:
      1. Padding mask (B, 1, 1, S)   — False where tgt == pad
      2. Subsequent mask (1, S, S)   — lower-triangular, prevents looking ahead

    Broadcasting: (B, 1, 1, S) & (1, 1, S, S) → (B, 1, S, S)

    Returns: (B, 1, S_tgt, S_tgt) boolean mask, True = can attend.
    """
    tgt_mask = (tgt != pad)[:, None, None, :]  # (B, 1, 1, S_tgt)
    seq_mask = subsequent_mask(tgt.shape[-1]).astype(tgt_mask.dtype)  # (1, S_tgt, S_tgt)
    return tgt_mask & seq_mask                                          # (B, 1, S_tgt, S_tgt)


# ──────────────────────────────────────────────────────────────────────
# Synthetic Data for Educational Training
# ──────────────────────────────────────────────────────────────────────


class SyntheticData:
    """Generate simple data for educational training.

    By default, creates a "copy task": the model must learn to copy the
    input sequence. The special tokens are:
      0 = <pad>, 1 = <bos>, 2 = <eos>
    Vocabulary tokens start from index 3.

    MLX difference:
      Returns mx.array instead of torch.Tensor. Uses mx.random.randint for
      random integer generation. Otherwise identical logic to the PyTorch version.
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

    def generate_batch(self, batch_size: int) -> tuple[mx.array, mx.array, mx.array]:
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
        # MLX: mx.random.randint(low, high, shape)
        lengths = mx.random.randint(1, self.max_len + 1, (batch_size,))
        max_len = int(lengths.max().item())  # .item() triggers eval (lazy → eager)
        # Note: .item() forces evaluation of the lazy computation graph for length.max().
        # For the small synthetic data this is negligible; in production you'd batch
        # by a fixed max_len or pad to a known upper bound.

        tokens = mx.random.randint(
            low=3, high=self.vocab_size, shape=(batch_size, max_len)
        )  # (B, max_len) — only content tokens (3..vocab_size-1)

        # Position mask: mask[b, p] = True iff position p < length of sequence b
        mask = mx.arange(max_len)[None, :] < lengths[:, None]  # (B, max_len)

        # MLX: .at[...].set() doesn't exist in MLX. Instead, use mx.where with
        # position masks to build arrays, and mx.put_along_axis for single-position
        # assignments (like placing EOS at each sequence's length position).
        # Compare: PyTorch's src[:, :max_len] = ... is in-place; MLX builds a new array.

        # Build content for positions 0..max_len-1 (non-EOS positions)
        pad_value = mx.full(tokens.shape, self.pad_idx, dtype=mx.int32)
        content = mx.where(mask, tokens, pad_value)  # (B, max_len)

        # ── Source: content + EOS ──────────────────────────────────────
        # Start with content tokens; add an extra column for potential EOS
        extra_col = mx.full((batch_size, 1), self.pad_idx, dtype=mx.int32)
        src = mx.concatenate([content, extra_col], axis=1)  # (B, max_len+1)
        # Place EOS at the correct position per batch row using mx.put_along_axis
        eos_vals = mx.full((batch_size, 1), self.eos_idx, dtype=mx.int32)
        src = mx.put_along_axis(src, lengths[:, None], eos_vals, axis=1)

        # ── Target input: BOS + content (no EOS) ───────────────────────
        tgt_input = mx.concatenate(
            [mx.full((batch_size, 1), self.pad_idx, dtype=mx.int32), content], axis=1
        )
        tgt_input = mx.put_along_axis(
            tgt_input, mx.zeros((batch_size, 1), dtype=mx.int32),
            mx.full((batch_size, 1), self.bos_idx, dtype=mx.int32), axis=1
        )

        # ── Target output: content + EOS (same as source) ──────────────
        tgt_out = mx.concatenate([content, extra_col], axis=1)  # (B, max_len+1)
        tgt_out = mx.put_along_axis(tgt_out, lengths[:, None], eos_vals, axis=1)

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

    MLX difference: Uses mx.random.randint instead of torch.randint; returns
    mx.array instead of torch.Tensor. Building variable-length token lists
    then padding is done via Python lists (same approach as PyTorch).
    """

    NUM_OFFSET = 3
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
            return [ArithmeticData.NUM_OFFSET]
        digits = []
        while n > 0:
            digits.append(n % 10 + ArithmeticData.NUM_OFFSET)
            n //= 10
        return list(reversed(digits))

    def generate_batch(self, batch_size: int) -> tuple[mx.array, mx.array, mx.array]:
        B = batch_size
        max_op = self.max_operand

        op_indices = mx.random.randint(0, 4, (B,))
        op_indices_eager = op_indices.tolist()  # MLX: force evaluation to Python list

        a_vals = mx.random.randint(0, max_op + 1, (B,)).tolist()
        b_vals = mx.random.randint(0, max_op + 1, (B,)).tolist()

        for i in range(B):
            op = op_indices_eager[i]
            a = a_vals[i]
            b = b_vals[i]
            if op == 1:  # subtraction: a >= b
                if a < b:
                    a_vals[i], b_vals[i] = b, a
            elif op == 3:  # division: exact only
                b = max(b, 1)  # ensure b >= 1
                max_k = max_op // b
                import random
                k = random.randint(0, max_k)
                a_vals[i] = b * k
                b_vals[i] = b

        answers = []
        for i in range(B):
            a = a_vals[i]
            b = b_vals[i]
            op = op_indices_eager[i]
            if op == 0:
                answers.append(a + b)
            elif op == 1:
                answers.append(a - b)
            elif op == 2:
                answers.append(a * b)
            else:
                answers.append(a // b if b != 0 else 0)

        # Build token sequences as lists then pad
        src_list = []
        tgt_out_list = []
        for i in range(B):
            a_digits = self._to_digits(a_vals[i])
            b_digits = self._to_digits(b_vals[i])
            ans_digits = self._to_digits(answers[i])

            src_tokens = a_digits + [self.OPS[op_indices_eager[i]]] + b_digits + [self.eos_idx]
            tgt_tokens = [self.EQ] + ans_digits + [self.eos_idx]

            src_list.append(src_tokens)
            tgt_out_list.append(tgt_tokens)

        src_max = max(len(s) for s in src_list)
        tgt_max = max(len(t) for t in tgt_out_list)

        src = mx.full((B, src_max), self.pad_idx, dtype=mx.int32)
        tgt_out = mx.full((B, tgt_max), self.pad_idx, dtype=mx.int32)

        for i in range(B):
            src[i, :len(src_list[i])] = mx.array(src_list[i], dtype=mx.int32)
            tgt_out[i, :len(tgt_out_list[i])] = mx.array(tgt_out_list[i], dtype=mx.int32)

        # tgt_in: BOS + tgt_out[:, :-1]
        tgt_in = mx.full((B, tgt_max), self.pad_idx, dtype=mx.int32)
        tgt_in[:, 0] = self.bos_idx
        tgt_in[:, 1:tgt_max] = tgt_out[:, :tgt_max - 1]

        return src, tgt_in, tgt_out

    @staticmethod
    def decode(tokens: list[int]) -> str:
        chars = []
        for t in tokens:
            if t in (0, 1, 2):
                continue
            if 3 <= t <= 12:
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
    data_iter: Iterator[tuple[mx.array, mx.array, mx.array]],
    model: nn.Module,
    loss_fn: Callable,
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

    MLX difference — value_and_grad instead of loss.backward():
      Gradient computation is functional: mx.value_and_grad(model, loss_fn)
      returns both the loss value and gradients w.r.t. model parameters.
      No zero_grad() needed — each call computes fresh gradients.
    """
    total_loss = 0.0
    n_tokens = 0

    if opt is not None:
        model.train()
    else:
        model.eval()

    for src, tgt_in, tgt_out in tqdm(data_iter, desc=desc):
        if opt is not None:
            # Create a closure that takes model + data and returns scalar loss
            # MLX: mx.value_and_grad(model, fn) computes gradients of fn w.r.t.
            # model parameters. The fn must take (model, *args) and return scalar.
            def loss_closure(m: nn.Module, s, ti, to):
                src_mask = (s != pad_idx)[:, None, None, :]
                tgt_mask = make_std_mask(ti, pad_idx)
                logits = m(s, ti, src_mask, tgt_mask)
                return loss_fn(logits, to)

            # MLX: mx.value_and_grad(fun) takes the loss function (not model+fn).
            # By default it differentiates w.r.t. the first argument (model).
            loss_and_grad = mx.value_and_grad(loss_closure)
            loss, grads = loss_and_grad(model, src, tgt_in, tgt_out)
            opt.step(model, grads)
            # CRITICAL MLX difference — mx.eval() forces computation:
            #   MLX builds a lazy computation graph. Without mx.eval(), the graph
            #   grows unboundedly with each iteration, consuming memory. Calling
            #   mx.eval() on parameters and optimizer state flushes the graph.
            #   In training, eval once per step is the standard pattern.
            mx.eval(model.parameters(), opt.optimizer.state)
        else:
            src_mask = (src != pad_idx)[:, None, None, :]
            tgt_mask = make_std_mask(tgt_in, pad_idx)
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = loss_fn(logits, tgt_out)

        # Weight by number of non-padding tokens for correct per-token averaging
        n_tok = int((tgt_out != pad_idx).sum().item())
        total_loss += float(loss.item()) * n_tok
        n_tokens += n_tok

    return total_loss / max(n_tokens, 1)


def run_epoch_steps(
    model: nn.Module,
    data_fn: Callable[[int], tuple[mx.array, mx.array, mx.array]],
    loss_fn: Callable,
    opt: NoamOpt,
    n_steps: int,
    batch_size: int,
    pad_idx: int = 0,
    print_every: int = 100,
) -> None:
    """Train a model for a fixed number of steps (used for quick educational training).

    Unlike run_epoch, this generates batches on-the-fly via data_fn instead of
    iterating over a fixed dataset. This is simpler for synthetic data where
    we can generate unlimited fresh examples (no risk of overfitting to a
    finite dataset — the model sees new tokens every step).

    The loss is unweighted at each step but the running average is computed
    over total non-padding tokens for accurate reporting.

    MLX difference:
      - No device management (MLX handles device placement automatically on Apple Silicon)
      - mx.eval() called after each optimizer step to flush the lazy graph
      - value_and_grad for gradient computation (functional, not imperative)
    """
    model.train()
    total_loss = 0.0
    n_tokens = 0
    start = time.time()

    for step in range(1, n_steps + 1):
        src, tgt_in, tgt_out = data_fn(batch_size)

        # Define the loss computation as a closure for value_and_grad
        def loss_closure(m: nn.Module, s, ti, to):
            src_mask = (s != pad_idx)[:, None, None, :]
            tgt_mask = make_std_mask(ti, pad_idx)
            logits = m(s, ti, src_mask, tgt_mask)
            return loss_fn(logits, to)

        loss_and_grad = mx.value_and_grad(loss_closure)
        loss, grads = loss_and_grad(model, src, tgt_in, tgt_out)
        opt.step(model, grads)
        # Flush the lazy computation graph — prevents unbounded memory growth.
        # MLX lazily records operations; mx.eval() forces execution and frees
        # the graph. This is the single most important MLX pattern to remember.
        mx.eval(model.parameters(), opt.optimizer.state)

        n_tok = int((tgt_out != pad_idx).sum().item())
        total_loss += float(loss.item()) * n_tok
        n_tokens += n_tok

        if step % print_every == 0:
            avg_loss = total_loss / max(n_tokens, 1)
            elapsed = time.time() - start
            print(
                f"Step {step:6d} | Loss: {avg_loss:.4f} "
                f"| LR: {opt.current_rate:.2e} "
                f"| Tokens/s: {n_tokens / elapsed:.0f}"
            )
