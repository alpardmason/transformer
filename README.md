# Original Transformer ("Attention is All You Need")

An educational, line-by-line implementation of the encoder-decoder Transformer
from [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762). Strictly follows
the original paper — Post-LN, ReLU activations, sinusoidal positional encodings,
Noam schedule, label smoothing.

Three identical implementations in PyTorch, MLX (Apple Silicon), and JAX+Flax.

## Quick Start

```bash
# Setup (requires Python >=3.12)
uv sync --group dev

# Run tests
uv run pytest tests/test_smoke.py -v

# Train on arithmetic (default task): model learns "1+1" → "=2"
uv run python scripts/train.py --steps 2000 --d-model 128 --N 2

# Train on copy task: model learns to reproduce input
uv run python scripts/train.py --task copy --steps 2000
```

## Arithmetic Task

The model takes an arithmetic expression and outputs the result:

```
Input:  1+1
Output: =2

Input:  99*99
Output: =9801
```

Numbers are digit-by-digit (each digit is a token), so the model must learn
decimal place value. Operations: +, -, *, / with operands in [0, 99].

`--max-operand` controls difficulty: `9` for single-digit, `12` for easy
2-digit, `99` (default) for full range.

## Project Structure

```
transformer/
├── Guide.md                  # Student tutorial (reading order, paper-to-code map)
├── src/
│   ├── transformer/          # PyTorch: layers.py, train.py, generate.py
│   ├── transformer_mlx/      # MLX (Apple Silicon): layers.py, train.py, generate.py
│   └── transformer_jax/      # JAX + Flax: layers.py, train.py, generate.py
├── scripts/
│   ├── train.py              # PyTorch CLI entry point
│   ├── train_mlx.py          # MLX CLI entry point
│   └── train_jax.py          # JAX CLI entry point
└── tests/
    ├── test_smoke.py         # 24 PyTorch tests (18 copy + 6 arithmetic)
    ├── test_smoke_mlx.py     # 23 MLX tests
    └── test_smoke_jax.py     # 24 JAX tests
```

## Other Backends

```bash
# MLX (Apple Silicon only)
uv sync --group mlx --group dev
uv run --group mlx --group dev pytest tests/test_smoke_mlx.py -v
uv run --group mlx python scripts/train_mlx.py --steps 1000

# JAX
uv sync --group jax --group dev
uv run --group jax --group dev pytest tests/test_smoke_jax.py -v
uv run --group jax python scripts/train_jax.py --steps 1000
```

## Key Design Decisions

- **Post-LN**: LayerNorm applied after the residual addition (not Pre-LN)
- **Weight tying**: Encoder embedding = decoder embedding = output projection
- **No modern techniques**: No RoPE, GELU, SwiGLU, Flash Attention, or KV-cache —
  this is the 2017 model as written

For a detailed comparison of this implementation to modern LLM architectures,
see `Guide.md`.
