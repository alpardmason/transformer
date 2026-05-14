"""MLX reimplementation of the original Transformer from "Attention is All You Need".

Re-exports the same API surface as src/transformer/ for drop-in compatibility.

Usage:
    from transformer_mlx import Transformer, LabelSmoothing, NoamOpt
    from transformer_mlx.train import SyntheticData, run_epoch_steps
    from transformer_mlx.generate import greedy_decode
"""

from .layers import (
    Decoder,
    DecoderLayer,
    Embeddings,
    Encoder,
    EncoderLayer,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFFN,
    ScaledDotProductAttention,
    SublayerConnection,
    Transformer,
)
from .train import (
    LabelSmoothing,
    NoamOpt,
    SyntheticData,
    make_std_mask,
    run_epoch,
    run_epoch_steps,
    subsequent_mask,
)
from .generate import beam_search, greedy_decode

__all__ = [
    "Transformer",
    "Encoder",
    "Decoder",
    "EncoderLayer",
    "DecoderLayer",
    "MultiHeadAttention",
    "ScaledDotProductAttention",
    "PositionwiseFFN",
    "PositionalEncoding",
    "Embeddings",
    "SublayerConnection",
    "NoamOpt",
    "LabelSmoothing",
    "subsequent_mask",
    "make_std_mask",
    "SyntheticData",
    "run_epoch",
    "run_epoch_steps",
    "greedy_decode",
    "beam_search",
]
