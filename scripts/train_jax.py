#!/usr/bin/env python3
"""Training entry point for the JAX + Flax Transformer.

Trains on a synthetic copy task to demonstrate the model's ability to learn
a simple sequence-to-sequence mapping.

JAX difference from PyTorch script:
  - Explicit PRNG keys: jax.random.PRNGKey and jax.random.split() for all randomness
  - model.init() to create variables, then model.apply() or TrainState for training
  - No device management (JAX handles device placement automatically)
  - train_state.TrainState bundles params, apply_fn, opt_state

Usage:
    uv run python scripts/train_jax.py
    uv run python scripts/train_jax.py --steps 2000 --batch-size 32 --d-model 128
"""

import argparse

import jax
import jax.numpy as jnp
from flax import linen as nn

from transformer_jax import Transformer, LabelSmoothing, NoamOpt, tie_weights
from transformer_jax.train import SyntheticData, run_epoch_steps
from transformer_jax.generate import greedy_decode


def main():
    """Training pipeline: Init → Weight Tie → TrainState → Train → Eval.

    The default config trains on a synthetic copy task — the model learns to
    reproduce the input sequence. This demonstrates that the architecture works
    without needing real data or long training.

    JAX difference — two-phase init:
      PyTorch/MLX: model is ready after __init__()
      JAX/Flax: model.__init__() sets config, then model.init(rngs, ...)
      creates the actual parameters (variables dict). Weight tying happens
      after init by copying params in the variables dict.
    """
    parser = argparse.ArgumentParser(description="Train the original Transformer (JAX)")
    parser.add_argument("--src-vocab", type=int, default=40)
    parser.add_argument("--tgt-vocab", type=int, default=40)
    parser.add_argument("--N", type=int, default=3, help="number of encoder/decoder layers")
    parser.add_argument("--d-model", type=int, default=128, help="model dimension")
    parser.add_argument("--d-ff", type=int, default=512, help="feed-forward dimension")
    parser.add_argument("--h", type=int, default=4, help="attention heads")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000, help="number of training steps")
    parser.add_argument("--max-len", type=int, default=10, help="max sequence length for synthetic data")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--no-tie-weights", action="store_true", help="disable weight tying")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    print(f"Config: d_model={args.d_model}, d_ff={args.d_ff}, h={args.h}, N={args.N}")

    # JAX: explicit PRNG key for all randomness
    rng_key = jax.random.PRNGKey(args.seed)

    # Model
    model = Transformer(
        src_vocab=args.src_vocab,
        tgt_vocab=args.tgt_vocab,
        N=args.N,
        d_model=args.d_model,
        d_ff=args.d_ff,
        h=args.h,
        dropout=args.dropout,
        tie_weights=not args.no_tie_weights,
    )

    # JAX: Initialize variables with dummy inputs
    rng_key, init_rng, dropout_rng = jax.random.split(rng_key, 3)
    dummy_src = jnp.ones((1, args.max_len), dtype=jnp.int32)
    dummy_tgt = jnp.ones((1, args.max_len), dtype=jnp.int32)
    variables = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        dummy_src,
        dummy_tgt,
        src_mask=jnp.ones((1, 1, 1, args.max_len), dtype=jnp.bool_),
        tgt_mask=jnp.ones((1, 1, args.max_len, args.max_len), dtype=jnp.bool_),
    )

    # Apply weight tying post-init
    if not args.no_tie_weights:
        variables = tie_weights(variables)

    # Count parameters
    def count_params(params):
        return sum(p.size for p in jax.tree_util.tree_leaves(params))

    n_params = count_params(variables["params"])
    print(f"Parameters: {n_params:,}")

    # Loss
    criterion = LabelSmoothing(smoothing=0.1, ignore_index=0)

    # Data — JAX data_fn takes (batch_size, rng_key) and returns (src, tgt_in, tgt_out, rng_key)
    data = SyntheticData(vocab_size=args.src_vocab, max_len=args.max_len)

    # Train
    print(f"Training for {args.steps} steps...")
    state = run_epoch_steps(
        model=model,
        data_fn=data.generate_batch,
        loss_fn=criterion,
        n_steps=args.steps,
        batch_size=args.batch_size,
        d_model=args.d_model,
        warmup=4000,
        pad_idx=0,
        print_every=args.print_every,
        rng_key=rng_key,
    )

    # Evaluate: generate a few examples with greedy decoding.
    # Use TRAINED params from returned state, not the initial variables.
    print("\n--- Evaluation ---")
    src, tgt_in, tgt_out, rng_key = data.generate_batch(5, rng_key)

    result = greedy_decode(
        model, {"params": state.params}, src, max_len=args.max_len + 2,
        bos_idx=1, eos_idx=2, pad_idx=0, rng_key=rng_key,
    )

    for i in range(5):
        src_tokens = src[i].tolist()
        expected = tgt_out[i].tolist()
        got = result[i].tolist()

        src_str = " ".join(str(t) for t in src_tokens if t not in (0,))
        exp_str = " ".join(str(t) for t in expected if t not in (0, 1, 2))
        got_str = " ".join(str(t) for t in got if t not in (0, 1, 2))

        correct = got_str == exp_str
        print(f"  SRC: {src_str:20s} | Expected: {exp_str:12s} | Got: {got_str:12s} | {'✓' if correct else '✗'}")


if __name__ == "__main__":
    main()
