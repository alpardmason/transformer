"""Training utilities for the original Transformer — JAX + Flax implementation.

Implements the exact training setup from the paper:
- Noam learning rate schedule (Section 5.3)
- Label smoothing (Section 5.4)
- Synthetic data for educational use

JAX/Flax vs PyTorch — key differences annotated throughout:
  - Explicit PRNG keys: every random op needs jax.random.PRNGKey and
    jax.random.split(). SyntheticData.generate_batch() takes/returns a key.
  - Functional training: jax.value_and_grad(loss_fn)(params) instead of
    loss.backward(). Gradients are pure values, not stored on tensors.
  - jax.jit: JIT-compile the training step for speed. First call is slow
    (trace), subsequent calls are fast.
  - flax.training.TrainState: bundles params, apply_fn, opt_state, tx.
  - optax: separate optimizer library. optax.adam, optax.chain, etc.
  - No device management: JAX arrays live on the default device.
  - Pure functions: LabelSmoothing is a function, not a class (JAX idiom).
  - jnp.where(cond, a, b) replaces masked_fill / scatter_.
"""

import time
from collections.abc import Callable
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
# Noam Learning Rate Scheduler (Section 5.3)
# ──────────────────────────────────────────────────────────────────────


def noam_schedule(model_size: int = 512, factor: float = 1.0, warmup: int = 4000):
    """Noam learning rate schedule as an optax-compatible callable.

        lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Two phases:
      - Linear warmup:  lrate ∝ step              (when step < warmup_steps)
      - Inverse sqrt decay: lrate ∝ 1/sqrt(step)  (when step > warmup_steps)

    JAX difference — functional schedule:
      In PyTorch, NoamOpt wraps the optimizer and calls .step(). In JAX +
      optax, the LR schedule is a separate function passed to the optimizer
      via optax's learning_rate parameter. optax calls the schedule with
      the current step count to get the LR for that step.

      This separation of concerns (schedule != optimizer != training loop)
      is a core JAX design principle: small, composable pure functions.

    Returns:
        A callable that maps step number (int) to learning rate (float).
    """

    def schedule(step: jnp.ndarray) -> jnp.ndarray:
        # step is a scalar jnp array from optax's internal counter
        step = step.astype(jnp.float32)
        arg1 = step ** (-0.5)
        arg2 = step * (warmup ** (-1.5))
        return factor * (model_size ** (-0.5)) * jnp.minimum(arg1, arg2)

    return schedule


class NoamOpt:
    """Thin wrapper around the Noam rate formula — for API compatibility.

    JAX difference:
      In PyTorch, NoamOpt wraps the optimizer and its step() sets the LR
      and calls optimizer.step(). In JAX, the schedule is passed directly
      to optax, so this class is just a rate() calculator for tests and
      standalone use.

    Use noam_schedule() for actual training with optax.
    """

    def __init__(
        self,
        model_size: int = 512,
        factor: float = 1.0,
        warmup: int = 4000,
    ):
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self._step = 0
        self._rate = 0.0

    def rate(self, step: int | None = None) -> float:
        if step is None:
            step = self._step
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


def label_smoothing_loss(
    logits: jnp.ndarray,
    target: jnp.ndarray,
    smoothing: float = 0.1,
    ignore_index: int = 0,
) -> jnp.ndarray:
    """Label smoothing loss with epsilon = 0.1 (paper §5.4).

    Instead of a hard one-hot target (all probability on the correct class),
    we use a smoothed distribution:
        q(k) = (1 - ε) * 1[k == y]  +  ε / (V - 1) * 1[k != y]

    JAX difference — pure function, not nn.Module:
      In PyTorch, LabelSmoothing subclasses nn.Module. JAX favors pure
      functions for stateless operations. This function takes logits and
      targets and returns a scalar loss — no class, no state.

      Uses cross-entropy with soft targets (equivalent to KL divergence
      when q is fixed). jax.nn.one_hot + manual smoothing replaces the
      scatter_/masked_fill pattern.

    Shapes:
      logits: (B, S, V)  — raw logits from the model
      target: (B, S)     — integer class indices
      output: scalar     — average loss per non-padding token
    """
    vocab_size = logits.shape[-1]  # V
    mask = target != ignore_index  # (B, S)

    # Build smoothed target distribution
    confidence = 1.0 - smoothing
    low_confidence = smoothing / (vocab_size - 1)

    # jax.nn.one_hot: (B, S) → (B, S, V)
    oh = jax.nn.one_hot(target, vocab_size)
    # Smooth: correct class = confidence, others = low_confidence
    true_dist = oh * confidence + (1 - oh) * low_confidence

    # Mask padding positions (zero probability everywhere)
    true_dist = jnp.where(mask[..., None], true_dist, 0.0)

    # Cross-entropy with soft targets: -sum(q * log p)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.sum(true_dist * log_probs, axis=-1)  # (B, S)

    # Mask padding
    ce = jnp.where(mask, ce, 0.0)

    # Average over non-padding tokens
    n_tokens = jnp.maximum(mask.sum(), 1)
    return ce.sum() / n_tokens


class LabelSmoothing:
    """Class wrapper for API compatibility with PyTorch-style code.

    JAX difference from PyTorch:
      The PyTorch version stores smoothing and ignore_index as instance state.
      This JAX wrapper delegates to the pure function label_smoothing_loss().
      In idiomatic JAX, you'd use the function directly; this class exists
      so the training script can swap between PyTorch/MLX/JAX seamlessly.
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def __call__(self, logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return label_smoothing_loss(logits, target, self.smoothing, self.ignore_index)


# ──────────────────────────────────────────────────────────────────────
# Mask Utilities
# ──────────────────────────────────────────────────────────────────────


def subsequent_mask(size: int) -> jnp.ndarray:
    """Create a lower-triangular boolean mask that prevents attending to future tokens.

    Returns (1, size, size) where mask[:, i, j] = True iff j <= i.

    JAX: jnp.triu(x, k=1) replaces torch.triu(x, diagonal=1).
    """
    attn_shape = (1, size, size)
    mask = jnp.triu(jnp.ones(attn_shape), k=1)
    return mask == 0


def make_std_mask(
    tgt: jnp.ndarray, pad: int = 0
) -> jnp.ndarray:
    """Create a combined padding + subsequent mask for decoder self-attention.

    Combines two masks via logical AND:
      1. Padding mask (B, 1, 1, S)   — False where tgt == pad
      2. Subsequent mask (1, S, S)   — lower-triangular, prevents looking ahead

    Broadcasting: (B, 1, 1, S) & (1, 1, S, S) → (B, 1, S, S)

    Returns: (B, 1, S_tgt, S_tgt) boolean mask, True = can attend.
    """
    tgt_mask = (tgt != pad)[:, jnp.newaxis, jnp.newaxis, :]  # (B, 1, 1, S_tgt)
    seq_mask = subsequent_mask(tgt.shape[-1])                  # (1, S_tgt, S_tgt)
    return tgt_mask & seq_mask                                  # (B, 1, S_tgt, S_tgt)


# ──────────────────────────────────────────────────────────────────────
# Synthetic Data for Educational Training
# ──────────────────────────────────────────────────────────────────────


class SyntheticData:
    """Generate simple data for educational training.

    By default, creates a "copy task": the model must learn to copy the
    input sequence. The special tokens are:
      0 = <pad>, 1 = <bos>, 2 = <eos>
    Vocabulary tokens start from index 3.

    JAX difference — explicit PRNG key threading:
      Every call to generate_batch() takes a jax.random.PRNGKey and returns
      a new key along with the data. The caller is responsible for threading
      the key through. This is THE fundamental difference from PyTorch random.

      Pattern:
        rng_key = jax.random.PRNGKey(42)
        src, tgt_in, tgt_out, rng_key = data.generate_batch(bs, rng_key)

      Each random operation consumes a key and returns a new one via split().
      This determinism makes JAX code reproducible by construction.
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
        self, batch_size: int, rng_key: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]:
        """Generate a batch of (src, tgt_in, tgt_out) for the copy task.

        JAX difference — PRNG key thread:
          Takes rng_key, uses jax.random.split() to derive keys for each
          random op, and returns the advanced key. Callers MUST use the
          returned key, not the original, for subsequent calls — reusing
          the same key produces identical random numbers.

        Example with vocab_size=40, max_len=5, tokens=[8,12,3,4] (length 4):

          src:      [8, 12, 3, 4, EOS]              — input with EOS terminator
          tgt_in:   [BOS, 8, 12, 3, 4]              — BOS + content (no EOS)
          tgt_out:  [8, 12, 3, 4, EOS]              — content + EOS (teacher target)

        Returns:
          src:      (B, max_len+1)
          tgt_in:   (B, max_len+1)
          tgt_out:  (B, max_len+1)
          rng_key:  advanced PRNG key for next call
        """
        rng_key, len_key, tok_key = jax.random.split(rng_key, 3)

        # Random lengths ensure variable-length sequences within the batch
        lengths = jax.random.randint(len_key, (batch_size,), 1, self.max_len + 1)
        max_len = int(lengths.max())  # jnp int → Python int for shape construction

        tokens = jax.random.randint(
            tok_key, (batch_size, max_len), 3, self.vocab_size
        )  # (B, max_len) — only content tokens (3..vocab_size-1)

        # Position mask: mask[b, p] = True iff position p < length of sequence b
        mask = jnp.arange(max_len)[None, :] < lengths[:, None]  # (B, max_len)

        # Source: content tokens + EOS, padded on the right
        src = jnp.full((batch_size, max_len + 1), self.pad_idx, dtype=jnp.int32)
        # JAX: .at[...].set(...) for indexed assignment on immutable arrays
        src = src.at[:, :max_len].set(
            jnp.where(mask, tokens, self.pad_idx)
        )
        src = src.at[jnp.arange(batch_size), lengths].set(self.eos_idx)

        # Target input: BOS + content tokens (no EOS) — the decoder input
        tgt_input = jnp.full((batch_size, max_len + 1), self.pad_idx, dtype=jnp.int32)
        tgt_out = jnp.full((batch_size, max_len + 1), self.pad_idx, dtype=jnp.int32)

        tgt_input = tgt_input.at[:, 0].set(self.bos_idx)
        tgt_input = tgt_input.at[:, 1 : max_len + 1].set(
            jnp.where(mask, tokens, self.pad_idx)
        )

        # Target output: content tokens + EOS — what the model should predict
        tgt_out = tgt_out.at[:, :max_len].set(
            jnp.where(mask, tokens, self.pad_idx)
        )
        tgt_out = tgt_out.at[jnp.arange(batch_size), lengths].set(self.eos_idx)

        return src, tgt_input, tgt_out, rng_key


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

    JAX difference: generate_batch() takes and returns a PRNG key.
    Pattern: src, tgt_in, tgt_out, rng_key = data.generate_batch(bs, rng_key)
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

    def generate_batch(
        self, batch_size: int, rng_key: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]:
        B = batch_size
        max_op = self.max_operand

        rng_key, op_key, a_key, b_key = jax.random.split(rng_key, 4)

        op_indices = jax.random.randint(op_key, (B,), 0, 4)
        a_vals = jax.random.randint(a_key, (B,), 0, max_op + 1)
        b_vals = jax.random.randint(b_key, (B,), 0, max_op + 1)

        # Convert to Python for per-sample logic
        op_list = [int(o) for o in op_indices]
        a_list = [int(a) for a in a_vals]
        b_list = [int(b) for b in b_vals]

        import random
        for i in range(B):
            op = op_list[i]
            if op == 1:  # subtraction: a >= b
                if a_list[i] < b_list[i]:
                    a_list[i], b_list[i] = b_list[i], a_list[i]
            elif op == 3:  # division: exact only
                b = max(b_list[i], 1)
                max_k = max_op // b
                k = random.randint(0, max_k)
                a_list[i] = b * k
                b_list[i] = b

        answers = []
        for i in range(B):
            a, b, op = a_list[i], b_list[i], op_list[i]
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
            a_digits = self._to_digits(a_list[i])
            b_digits = self._to_digits(b_list[i])
            ans_digits = self._to_digits(answers[i])

            src_tokens = a_digits + [self.OPS[op_list[i]]] + b_digits + [self.eos_idx]
            tgt_tokens = [self.EQ] + ans_digits + [self.eos_idx]

            src_list.append(src_tokens)
            tgt_out_list.append(tgt_tokens)

        src_max = max(len(s) for s in src_list)
        tgt_max = max(len(t) for t in tgt_out_list)

        src = jnp.full((B, src_max), self.pad_idx, dtype=jnp.int32)
        tgt_out = jnp.full((B, tgt_max), self.pad_idx, dtype=jnp.int32)

        for i in range(B):
            src = src.at[i, :len(src_list[i])].set(
                jnp.array(src_list[i], dtype=jnp.int32)
            )
            tgt_out = tgt_out.at[i, :len(tgt_out_list[i])].set(
                jnp.array(tgt_out_list[i], dtype=jnp.int32)
            )

        # tgt_in: BOS + tgt_out[:, :-1]
        tgt_in = jnp.full((B, tgt_max), self.pad_idx, dtype=jnp.int32)
        tgt_in = tgt_in.at[:, 0].set(self.bos_idx)
        tgt_in = tgt_in.at[:, 1:tgt_max].set(tgt_out[:, :tgt_max - 1])

        return src, tgt_in, tgt_out, rng_key

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
# Training Loop (JAX + Flax + optax style)
# ──────────────────────────────────────────────────────────────────────


def create_train_state(
    model: nn.Module,
    rng_key: jax.Array,
    d_model: int,
    warmup: int = 4000,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-9,
    src_vocab: int = 40,
    tgt_vocab: int = 40,
    max_len: int = 10,
) -> train_state.TrainState:
    """Initialize the Flax TrainState with model params and optimizer.

    JAX difference — TrainState pattern:
      Flax's TrainState bundles together:
        - params: the model parameters (from variables['params'])
        - apply_fn: the model's forward function (model.apply)
        - tx: the optax optimizer (combined transform)
        - opt_state: optimizer state (running means for Adam, etc.)

      This is the canonical Flax training pattern. It replaces:
        PyTorch: model.parameters() + optimizer.state_dict()
        MLX:     model.parameters() + optimizer.state

    Args:
        model: the Flax nn.Module (Transformer)
        rng_key: PRNG key for initialization
        d_model: model dimension (for Noam schedule)
        warmup: Noam warmup steps
        betas, eps: Adam hyperparameters (paper §5.3)

    Returns:
        TrainState ready for training
    """
    # Build optax optimizer chain with Noam schedule
    schedule_fn = noam_schedule(model_size=d_model, warmup=warmup)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # gradient clipping (standard practice)
        optax.adam(learning_rate=schedule_fn, b1=betas[0], b2=betas[1], eps=eps),
    )

    # Initialize model parameters
    rng_key, init_rng, dropout_rng = jax.random.split(rng_key, 3)

    # Create dummy inputs for init
    dummy_src = jnp.ones((1, max_len), dtype=jnp.int32)
    dummy_tgt = jnp.ones((1, max_len), dtype=jnp.int32)

    variables = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        dummy_src,
        dummy_tgt,
        deterministic=False,  # mode doesn't matter for init
    )

    # Apply weight tying post-init so training uses consistent tied weights
    if getattr(model, 'tie_weights', False):
        from .layers import tie_weights as _tie_weights
        variables = _tie_weights(variables)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


def run_epoch(
    model: nn.Module,
    data_iter,
    loss_fn: Callable,
    state: Optional[train_state.TrainState] = None,
    desc: str = "train",
    pad_idx: int = 0,
    rng_key: Optional[jax.Array] = None,
) -> Tuple[float, Optional[train_state.TrainState], Optional[jax.Array]]:
    """Run one training epoch with teacher forcing.

    JAX difference — state returned, not mutated:
      In PyTorch, model parameters and optimizer state are mutated in-place.
      In JAX, the TrainState is immutable — each optimizer step returns a
      new TrainState. This function returns the updated state and rng_key.

    If state is None, runs in eval mode (no gradients).

    Returns:
        avg_loss, updated_state, updated_rng_key
    """
    total_loss = 0.0
    n_tokens = 0

    for src, tgt_in, tgt_out in tqdm(data_iter, desc=desc):
        if state is not None:
            if rng_key is None:
                rng_key = jax.random.PRNGKey(0)
            rng_key, dropout_key = jax.random.split(rng_key)

            # Non-JIT training step (simpler than the JIT version in run_epoch_steps)
            def _loss_fn(params):
                src_mask = (src != pad_idx)[:, None, None, :]
                tgt_mask = make_std_mask(tgt_in, pad_idx)
                logits = state.apply_fn(
                    {"params": params},
                    src, tgt_in, src_mask, tgt_mask,
                    rngs={"dropout": dropout_key},
                )
                return loss_fn(logits, tgt_out)

            loss, grads = jax.value_and_grad(_loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            total_loss += float(loss)
            n_tokens += int((tgt_out != pad_idx).sum())
        else:
            # Eval mode — no gradient computation needed
            src_mask = (src != pad_idx)[:, None, None, :]
            tgt_mask = make_std_mask(tgt_in, pad_idx)
            if rng_key is None:
                logits = model.apply(
                    {"params": state.params} if state else {},
                    src, tgt_in, src_mask, tgt_mask,
                )
            else:
                rng_key, dropout_key = jax.random.split(rng_key)
                logits = model.apply(
                    {"params": state.params} if state else {},
                    src, tgt_in, src_mask, tgt_mask,
                    rngs={"dropout": dropout_key},
                )
            loss = loss_fn(logits, tgt_out)
            total_loss += float(loss) * int((tgt_out != pad_idx).sum())
            n_tokens += int((tgt_out != pad_idx).sum())

    return total_loss / max(n_tokens, 1), state, rng_key


def run_epoch_steps(
    model: nn.Module,
    data_fn: Callable,
    loss_fn: Callable,
    n_steps: int,
    batch_size: int,
    d_model: int = 128,
    warmup: int = 4000,
    pad_idx: int = 0,
    print_every: int = 100,
    rng_key: Optional[jax.Array] = None,
) -> None:
    """Train a model for a fixed number of steps (educational training).

    JAX difference — the full training loop pattern:
      This function demonstrates the canonical JAX training loop:
        1. Initialize model → TrainState (params + optax state)
        2. Define a jit-compiled training step
        3. Loop: generate data (thread rng_key), call train_step, log

      Key JAX concepts on display:
        - jax.jit: compile train_step for speed (first call traces)
        - jax.value_and_grad: compute loss AND gradients in one call
        - TrainState.apply_gradients: pure functional optimizer step
        - PRNG threading: key split before every random operation
        - No in-place mutations: state is replaced, not modified

    Unlike run_epoch, this generates batches on-the-fly via data_fn.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    rng_key, init_key = jax.random.split(rng_key)

    # Create TrainState
    state = create_train_state(
        model=model,
        rng_key=init_key,
        d_model=d_model,
        warmup=warmup,
    )

    total_loss = 0.0
    total_tokens = 0
    start = time.time()

    # ── JIT-compiled training step ──────────────────────────────────
    # jax.jit compiles this function to XLA. The first call traces
    # (slow), subsequent calls are fast. static_argnames marks arguments
    # whose VALUE must be known at compile time (not runtime).
    @partial(jax.jit, static_argnames=["loss_fn", "pad_idx"])
    def train_step(state, batch, dropout_key, loss_fn, pad_idx):
        src, tgt_in, tgt_out = batch

        def loss_fn_params(params):
            src_mask = (src != pad_idx)[:, None, None, :]
            tgt_mask = make_std_mask(tgt_in, pad_idx)
            logits = state.apply_fn(
                {"params": params},
                src, tgt_in, src_mask, tgt_mask,
                # Flax: dropout uses the key from rngs dict
                rngs={"dropout": dropout_key},
            )
            return loss_fn(logits, tgt_out)

        # jax.value_and_grad: compute (loss, grads) in one backward pass
        loss, grads = jax.value_and_grad(loss_fn_params)(state.params)
        # state.apply_gradients: pure functional update (returns new state)
        state = state.apply_gradients(grads=grads)
        n_tokens = (tgt_out != pad_idx).sum().astype(jnp.int32)
        return state, loss, n_tokens

    # ── Training loop ───────────────────────────────────────────────
    for step in range(1, n_steps + 1):
        src, tgt_in, tgt_out, rng_key = data_fn(batch_size, rng_key)
        rng_key, dropout_key = jax.random.split(rng_key)

        state, loss, n_tok = train_step(
            state, (src, tgt_in, tgt_out), dropout_key, loss_fn, pad_idx
        )

        total_loss += float(loss) * int(n_tok)
        total_tokens += int(n_tok)

        if step % print_every == 0:
            avg_loss = total_loss / max(total_tokens, 1)
            elapsed = time.time() - start
            print(
                f"Step {step:6d} | Loss: {avg_loss:.4f} "
                f"| Tokens/s: {total_tokens / elapsed:.0f}"
            )

    return state
