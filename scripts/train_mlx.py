#!/usr/bin/env python3
"""Training entry point for the MLX Transformer.

Supports two tasks:
  - arithmetic (default): model learns addition, subtraction, multiplication,
    division. Input like "1+1", target "=2".
  - copy: model learns to copy the input sequence (for testing architecture).

MLX difference from PyTorch script:
  - No device management (MLX runs on Apple Silicon automatically)
  - mx.eval() called periodically inside run_epoch_steps to flush lazy graph

Usage:
    uv run python scripts/train_mlx.py
    uv run python scripts/train_mlx.py --steps 1000 --d-model 128
    uv run python scripts/train_mlx.py --task copy --steps 1000
    uv run python scripts/train_mlx.py --max-operand 12
"""

import argparse

import mlx.core as mx
import mlx.optimizers as optim

from transformer_mlx import LabelSmoothing, NoamOpt, Transformer
from transformer_mlx.generate import greedy_decode
from transformer_mlx.train import ArithmeticData, SyntheticData, run_epoch_steps


def main():
    parser = argparse.ArgumentParser(description="Train the original Transformer (MLX)")
    parser.add_argument(
        "--task", type=str, default="arithmetic", choices=["copy", "arithmetic"],
        help="training task: copy or arithmetic (default: arithmetic)",
    )
    parser.add_argument("--src-vocab", type=int, default=None)
    parser.add_argument("--tgt-vocab", type=int, default=None)
    parser.add_argument("--N", type=int, default=3, help="number of encoder/decoder layers")
    parser.add_argument("--d-model", type=int, default=128, help="model dimension")
    parser.add_argument("--d-ff", type=int, default=512, help="feed-forward dimension")
    parser.add_argument("--h", type=int, default=4, help="attention heads")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000, help="number of training steps")
    parser.add_argument("--max-len", type=int, default=10, help="max sequence length for data")
    parser.add_argument("--max-operand", type=int, default=99, help="max operand for arithmetic (0-99)")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--no-tie-weights", action="store_true", help="disable weight tying")
    args = parser.parse_args()

    if args.src_vocab is None:
        args.src_vocab = ArithmeticData.VOCAB_SIZE if args.task == "arithmetic" else 40
    if args.tgt_vocab is None:
        args.tgt_vocab = ArithmeticData.VOCAB_SIZE if args.task == "arithmetic" else 40

    print(f"Task: {args.task} | max_operand={args.max_operand}")
    print(f"Config: d_model={args.d_model}, d_ff={args.d_ff}, h={args.h}, N={args.N}")

    mx.random.seed(42)

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

    def _count_params(d):
        if isinstance(d, dict):
            return sum(_count_params(v) for v in d.values())
        if isinstance(d, (list, tuple)):
            return sum(_count_params(v) for v in d)
        return d.size
    n_params = _count_params(model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer
    adam = optim.Adam(learning_rate=1.0, betas=(0.9, 0.98), eps=1e-9)
    noam = NoamOpt(model_size=args.d_model, factor=1.0, warmup=4000, optimizer=adam)

    # Loss
    criterion = LabelSmoothing(smoothing=0.1, ignore_index=0)

    # Data
    if args.task == "arithmetic":
        data = ArithmeticData(max_len=args.max_len, max_operand=args.max_operand)
    else:
        data = SyntheticData(vocab_size=args.src_vocab, max_len=args.max_len)
    data_fn = lambda bs: data.generate_batch(bs)

    # Train
    print(f"Training for {args.steps} steps...")
    run_epoch_steps(
        model=model, data_fn=data_fn, loss_fn=criterion, opt=noam,
        n_steps=args.steps, batch_size=args.batch_size,
        pad_idx=0, print_every=args.print_every,
    )

    # Evaluate
    print("\n--- Evaluation ---")
    model.eval()
    src, _, tgt_out = data.generate_batch(5)

    result = greedy_decode(model, src, max_len=args.max_len + 2, bos_idx=1, eos_idx=2, pad_idx=0)

    for i in range(5):
        src_tokens = src[i].tolist()
        expected = tgt_out[i].tolist()
        got = result[i].tolist()

        if args.task == "arithmetic":
            src_str = ArithmeticData.decode(src_tokens)
            exp_str = ArithmeticData.decode(expected)
            got_str = ArithmeticData.decode(got)
        else:
            src_str = " ".join(str(t) for t in src_tokens if t not in (0,))
            exp_str = " ".join(str(t) for t in expected if t not in (0, 1, 2))
            got_str = " ".join(str(t) for t in got if t not in (0, 1, 2))

        correct = got_str == exp_str
        print(f"  SRC: {src_str:20s} | Expected: {exp_str:12s} | Got: {got_str:12s} | {'✓' if correct else '✗'}")


if __name__ == "__main__":
    main()
