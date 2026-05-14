"""JAX + Flax reimplementation of the original Transformer from "Attention is All You Need".

Re-exports the same API surface as src/transformer/ for drop-in compatibility.

Usage:
    import jax
    from transformer_jax import Transformer, LabelSmoothing, NoamOpt, tie_weights
    from transformer_jax.train import SyntheticData, run_epoch_steps
    from transformer_jax.generate import greedy_decode

    # JAX initialization pattern:
    model = Transformer(src_vocab=40, tgt_vocab=40, N=3, d_model=128, d_ff=512, h=4)
    rng_key = jax.random.PRNGKey(42)
    variables = model.init(rng_key, src, tgt, ...)
    # optionally: variables = tie_weights(variables)
    # Then use model.apply(variables, ...) or TrainState for training
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
    tie_weights,
)
from .train import (
    LabelSmoothing,
    NoamOpt,
    SyntheticData,
    create_train_state,
    label_smoothing_loss,
    make_std_mask,
    noam_schedule,
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
    "label_smoothing_loss",
    "subsequent_mask",
    "make_std_mask",
    "SyntheticData",
    "run_epoch",
    "run_epoch_steps",
    "create_train_state",
    "noam_schedule",
    "tie_weights",
    "greedy_decode",
    "beam_search",
]
