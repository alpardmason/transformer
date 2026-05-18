# CLAUDE.md — Original Transformer ("Attention is All You Need")

An educational implementation of the encoder-decoder Transformer from
Vaswani et al. (2017). Strictly follows the original paper — no modern
techniques.

**Python requirement**: >=3.12 (set in pyproject.toml). The package is
installed from `src/` via setuptools (`[tool.setuptools.packages.find]
where = ["src"]`), so `from transformer import ...` resolves to
`src/transformer/`. This is an editable install — `uv sync` installs the
package automatically.

## Project layout

```
transformer/
├── Guide.md                     # Student tutorial: reading order, best practices, modern vs. original
├── pyproject.toml               # uv-managed; optional deps: torch, mlx, jax+flax+optax
├── src/
│   ├── transformer/             # Original PyTorch implementation
│   │   ├── layers.py
│   │   ├── train.py
│   │   └── generate.py
│   ├── transformer_mlx/         # MLX (Apple Silicon) reimplementation
│   │   ├── layers.py
│   │   ├── train.py
│   │   ├── generate.py
│   │   └── __init__.py
│   └── transformer_jax/         # JAX + Flax reimplementation
│       ├── layers.py
│       ├── train.py
│       ├── generate.py
│       └── __init__.py
├── scripts/
│   ├── train.py                 # PyTorch CLI entry point
│   ├── train_mlx.py             # MLX CLI entry point
│   └── train_jax.py             # JAX CLI entry point
└── tests/
    ├── test_smoke.py            # 18 PyTorch tests
    ├── test_smoke_mlx.py        # 18 MLX tests (mirrors PyTorch)
    └── test_smoke_jax.py        # 18 JAX tests (mirrors PyTorch)
```

## How to work with this project

```bash
# First-time setup: create venv, install PyTorch deps + pytest for tests
uv sync --group dev

# Run PyTorch tests (24 tests: 18 copy-task + 6 arithmetic)
uv run pytest tests/test_smoke.py -v

# Arithmetic task (default) — max_operand=12 for quick experiments, 99 for full
uv run python scripts/train.py --steps 2000 --d-model 128 --N 2
uv run python scripts/train.py --task arithmetic --steps 2000 --max-operand 12

# Copy task (for comparison / testing architecture)
uv run python scripts/train.py --task copy --steps 2000 --d-model 128 --N 2

# MLX (Apple Silicon only)
uv sync --group mlx --group dev
uv run --group mlx --group dev pytest tests/test_smoke_mlx.py -v
uv run --group mlx python scripts/train_mlx.py --steps 1000 --max-operand 12

# JAX
uv sync --group jax --group dev
uv run --group jax --group dev pytest tests/test_smoke_jax.py -v
uv run --group jax python scripts/train_jax.py --steps 1000 --max-operand 12
```

Dependency groups are additive — `--group mlx --group dev` includes both.
**pytest is in the `dev` group** — always include `--group dev` when running tests.

## Training tasks

The project supports two tasks via `--task` flag in training scripts:

### Arithmetic (default)

Model takes an expression like `1+1` and generates `=2`. Numbers are digit-by-digit
so the model learns decimal place value. Operations: +, -, *, / with operands in
[0, max_operand] (default 99).

- Subtraction ensures a >= b (non-negative results)
- Division uses exact division only (a = b × k)
- `--max-operand` controls difficulty: 9 = single-digit, 12 = easy 2-digit, 99 = full

Token layout (22 tokens, `ArithmeticData.VOCAB_SIZE`):
| ID | Meaning | ID | Meaning |
|----|---------|----|---------|
| 0 | PAD | 1 | BOS |
| 2 | EOS | 3-12 | digits 0-9 |
| 13 | `+` | 14 | `-` |
| 15 | `*` | 16 | `/` |
| 17 | `=` | 18-21 | reserved |

### Copy

Model must reproduce the input sequence (identity mapping). Uses `SyntheticData`
with random content tokens (IDs 3..vocab_size-1). Useful for verifying the
architecture works before tackling the harder arithmetic task.

## Architecture (strict original paper)

- **Post-LN**: `LayerNorm(x + dropout(Sublayer(x)))` — NOT Pre-LN
- **ReLU** in FFN — NOT GELU
- **Sinusoidal** positional encoding — NOT learned
- **Weight tying**: encoder embedding = decoder embedding = final projection
  (done via `decoder.embedding.lut = encoder.embedding.lut` and
  `generator.weight = embedding.weight`)
- **Noam schedule**: `lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))`
- **Label smoothing**: ε=0.1, uses KL divergence

## Key conventions

### Mask shapes

Masks are boolean tensors where `True` = can attend, `False` = masked.

- `src_mask`: `(batch, 1, 1, src_len)` — padding mask, broadcasts over heads/query
- `tgt_mask`: `(batch, 1, tgt_len, tgt_len)` — `make_std_mask()` combines padding + subsequent
- `subsequent_mask(size)`: `(1, size, size)` — lower-triangular True

In `ScaledDotProductAttention`: `scores.masked_fill(~mask, float('-inf'))`

### MultiHeadAttention mask handling

Passes masks through unchanged if 4D; only adds a head dimension if the mask
is 3D. Callers should provide masks already broadcastable to
`(batch, h, seq_q, seq_k)`.

## Commenting conventions (educational focus)

This codebase is annotated for students learning the Transformer. Follow these
conventions when editing.

### Tensor shape notation

Use single-letter abbreviations in inline comments:

| Letter | Meaning |
|--------|---------|
| `B` | batch size |
| `S` | sequence length (use `S_src`, `S_tgt` when both appear) |
| `h` | number of attention heads |
| `d_k` | dimension per head (`d_model // h`) |
| `d_model` | model dimension (embedding size) |
| `d_ff` | feed-forward inner dimension |
| `V` | vocabulary size |

Shape comments go on the same line as (or directly above) the operation:

```python
x = self.embedding(x)      # (B, S_src) → (B, S_src, d_model)
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, S_q, S_k)
```

### What to comment

- **Algorithm steps** — 1–3 line explanations of non-obvious logic (beam search
  pruning, Noam warmup, label smoothing distribution construction)
- **Design choices** — why `register_buffer` vs parameter, why `copy.deepcopy`,
  why `@property` for weight tying, why Post-LN matters
- **Shapes** — at every tensor transformation, especially reshape/transpose/permute
- **Paper references** — cite section numbers (e.g. §3.4, §5.3)

### What NOT to comment

- Don't restate obvious code (`# create a tensor` next to `torch.zeros(...)`)
- Don't write multi-paragraph docstrings — keep class docstrings to 2–6 lines
- Don't add comments about "this is used by X" or "added for the Y flow"

## What to avoid

- No `torch.nn.functional.scaled_dot_product_attention` — the manual math is the point
- No Pre-LN, GELU, learned positional embeddings, or relative position biases
- No config dataclasses, registry patterns, or logging frameworks
- Keep the 3-file layout (`layers.py`, `train.py`, `generate.py`) — don't split further

## Guide.md

`Guide.md` is a tutorial for students who have just finished reading the paper.
It follows the reading sequence: `layers.py` → `train.py` → `generate.py` →
`scripts/train.py` → `test_smoke.py`. Each chapter covers:

- How the paper's equations map to this specific code
- Best practices and why they exist (`register_buffer`, `@property` for weight
  tying, `copy.deepcopy` for clones, `@torch.no_grad()`, `.contiguous()` before
  `.view()`, boolean mask conventions)
- A table comparing every aspect of this 2017 implementation to today's LLM
  stack (Pre-LN, RoPE, Flash Attention, KV-cache, SwiGLU, AdamW, cosine
  schedules, mixed precision, tokenization, decoder-only architectures)

**When to update Guide.md:** If you add or remove a file, change the reading
sequence. If you change a component's design (e.g. switch from Post-LN to
Pre-LN), update both the chapter covering that file and the modern-comparison
table.

## Gotchas

- `torch.uint8` for `triu` in `subsequent_mask` then `== 0` → boolean. Using
  `dtype=torch.bool` directly can behave differently with some PyTorch ops.
- Weight tying via `module.weight = other_module.weight` triggers
  `register_parameter` — so `Embeddings` exposes a `@property weight` that
  returns `self.lut.weight` to avoid `KeyError: "attribute 'weight' already
  exists"`.
- The `SublayerConnection` receives a `Callable`, not an `nn.Module` — the
  lambda captures mask from the enclosing scope.

## MLX-specific gotchas

### Lazy evaluation — must call `mx.eval()` every step

MLX builds a lazy computation graph. Without `mx.eval()`, the graph grows
unboundedly with each training step, consuming memory. The standard pattern:

```python
loss, grads = loss_and_grad(model, src, tgt_in, tgt_out)
opt.step(model, grads)
mx.eval(model.parameters(), opt.optimizer.state)  # flush the graph
```

### `mx.value_and_grad(fn)` — function only, NOT `(model, fn)`

PyTorch: `loss.backward()`. MLX: `mx.value_and_grad(loss_fn)(model, args...)`.
The first argument is the function, NOT `(model, loss_fn)`. Gradients are
taken w.r.t. the first argument of the function (the model).

**Wrong:** `mx.value_and_grad(model, loss_closure)`
**Right:** `mx.value_and_grad(loss_closure)` then `loss_and_grad(model, ...)`

### `ArrayAt` has NO `.set()` method

MLX's `ArrayAt` only supports `.add()`, `.subtract()`, `.multiply()`,
`.divide()`, `.maximum()`, `.minimum()`. For indexed assignment:

- Use `mx.put_along_axis(arr, indices, values, axis)` for single-position writes
- Use `mx.where(condition, a, b)` for boolean-masked selection
- Use `mx.concatenate([content, extra], axis=1)` + `mx.put_along_axis` to build
  padded sequences (see `SyntheticData.generate_batch`)

### `model.parameters()` returns nested dicts, not flat

MLX returns a tree like `{'encoder': {'layers': [...], ...}, ...}`. To count
params, recurse through dicts AND lists:

```python
def _count_params(d):
    if isinstance(d, dict):
        return sum(_count_params(v) for v in d.values())
    if isinstance(d, (list, tuple)):
        return sum(_count_params(v) for v in d)
    return d.size
```

Weight tying creates the same array under multiple paths — this recursive
counter WILL double-count tied weights, so parameter counts differ from
PyTorch's deduplicated `model.parameters()`.

### No `mx.relu` — use `nn.relu()` or `mx.maximum(x, 0)`

MLX's activation functions live in `mlx.nn`, not `mlx.core`:
- `nn.relu(x)` not `mx.relu(x)`
- `mx.softmax(x, axis=...)` for softmax
- No `mx.log_softmax` — use `mx.log(mx.softmax(x, axis=-1))`

### No `.topk()` — use `mx.argsort` + slice

```python
sorted_indices = mx.argsort(arr, axis=-1)[..., ::-1][:k]  # descending top-k
```

### Other MLX differences

- `mx.where(cond, a, b)` replaces `masked_fill` and `scatter_`
- `mx.argmax(axis=...)` replaces `.argmax(dim=...)`
- `mx.concatenate` replaces `torch.cat`
- `mx.swapaxes` replaces `transpose`
- No `register_buffer` — store constants as plain attributes
- No `.contiguous()` needed; MLX arrays are always contiguous
- `model.eval()` / `model.train()` toggle dropout (no `torch.no_grad()` needed)

## JAX + Flax specific gotchas

### Explicit PRNG keys everywhere

Every random op needs a key. `SyntheticData.generate_batch()` takes AND returns
a key. Callers MUST use the returned key, never reuse:

```python
src, tgt_in, tgt_out, rng_key = data.generate_batch(bs, rng_key)
```

### `int()` inside `@jax.jit` crashes with ConcretizationTypeError

JIT-compiled functions receive traced (abstract) arrays. Calling `int()` on a
traced value fails. Keep values as JAX arrays inside JIT, convert after return:

```python
# Inside @jax.jit — WRONG:
n_tokens = int((tgt_out != pad_idx).sum())
# Inside @jax.jit — RIGHT:
n_tokens = (tgt_out != pad_idx).sum().astype(jnp.int32)
# Outside (caller) — now concrete, int() works:
total_tokens += int(n_tok)
```

### Flax `init()` clones the module — `setup()` attrs not on original

`model.init(rngs, ...)` clones the module before calling `setup()`. Attributes
set in `setup()` (like `self.pe`) are NOT accessible on the original module.
Use `model.bind(variables)` to get a bound module with accessible attributes:

```python
# WRONG — AttributeError: "PositionalEncoding" object has no attribute "pe"
pe = PositionalEncoding(...)
pe.init({"params": key}, x)
print(pe.pe.shape)

# RIGHT — bind the variables to a new module
variables = pe.init({"params": key}, x)
bound = pe.bind(variables)
print(bound.pe.shape)
```

### Flax modules in Python lists are NOT tracked

Use `setattr(self, f'layer_{i}', layer)` in a loop for N-repeat stacks:

```python
def setup(self):
    for i in range(self.N):
        setattr(self, f'layer_{i}', EncoderLayer(...))
```

### Weight tying requires post-init variable dict copying

Flax's functional variable tree makes simple reference sharing impossible.
Initialize normally, then copy params in the variables dict:

```python
variables = model.init(rngs, src, tgt)
variables = tie_weights(variables)  # copies encoder embed to decoder + generator
```

**Critical**: When `tie_weights=True`, `Transformer.__call__` never invokes
`self.generator(dec_out)` — it uses the tied embed directly. So during `init()`,
the generator's params are NEVER created. `tie_weights()` must CREATE the
`params["generator"]` entry from scratch, not assign into it:

```python
# WRONG — KeyError: 'generator'
params["generator"]["kernel"] = enc_embed.T
# RIGHT — create the dict entry
params["generator"] = {"kernel": enc_embed.T}
```

### `TrainState` is immutable — training returns new state, not mutates

JAX training loops must capture the returned state and rng_key:

```python
state, loss, n_tok = train_step(state, batch, dropout_key, ...)
```

When `run_epoch_steps` creates its own `TrainState` internally, it must
RETURN it so evaluation can use the trained params (not the initial ones).

### Other JAX/Flax differences

- `nn.Dense(features)` replaces `nn.Linear`; `nn.Embed` replaces `nn.Embedding`
- `@nn.compact` for modules that create params lazily in `__call__`
- `setup()` for modules that need named submodules accessible after construction
- `deterministic: bool` must thread through every module's `__call__` for dropout
- `model.apply(variables, ..., method=model.encode, rngs={...})` for forward pass
- `jnp.where(cond, a, b)` replaces `masked_fill`
- `jnp.swapaxes` replaces `transpose`
- `.at[idx].set(val)` for indexed assignment on immutable arrays
- `optax` for optimizers; pass `noam_schedule()` as `learning_rate` callback
- No `.topk()` — use `jnp.argsort(arr, axis=-1)[-k:][::-1]`
- No device management — JAX handles placement automatically
