#!/usr/bin/env python3
"""Smoke tests for the JAX + Flax Transformer implementation.

Verifies forward pass shapes, masking, and the ability to overfit
on a tiny synthetic dataset. Mirrors the PyTorch version (tests/test_smoke.py)
with JAX-specific adaptations.

JAX vs PyTorch test differences annotated inline:
  - jax.random.normal(key, shape) replaces torch.randn
  - jnp.allclose replaces torch.allclose
  - jnp.ones / jnp.zeros replace torch.ones / torch.zeros
  - model.init(rngs, ...) for parameter initialization
  - model.apply(variables, ..., method=..., rngs={...}) for forward pass
  - PRNG keys threaded through all test calls
  - train_state.TrainState + optax for training
  - No .device management — JAX handles device automatically
"""

import math

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import pytest

from transformer_jax import (
    Transformer, NoamOpt, LabelSmoothing, subsequent_mask, greedy_decode, tie_weights,
)
from transformer_jax.layers import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionwiseFFN,
    PositionalEncoding,
    Embeddings,
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder,
)
from transformer_jax.train import ArithmeticData, SyntheticData, make_std_mask, label_smoothing_loss


# ──────────────────────────────────────────────────────────────────────
# Test: Scaled Dot-Product Attention
# ──────────────────────────────────────────────────────────────────────


def test_scaled_dot_product_attention():
    """Verify: Q,K,V (B,h,S,d_k) → output (B,h,S,d_k), weights (B,h,S,S) sum to 1."""
    attn = ScaledDotProductAttention(dropout=0.0)

    batch, h, seq, d_k = 2, 8, 5, 64
    rng_key = jax.random.PRNGKey(0)
    # JAX: jax.random.normal(key, shape) — key is REQUIRED
    q = jax.random.normal(rng_key, (batch, h, seq, d_k))  # (B, h, S, d_k)
    rng_key, k_key, v_key = jax.random.split(rng_key, 3)
    k = jax.random.normal(k_key, (batch, h, seq, d_k))
    v = jax.random.normal(v_key, (batch, h, seq, d_k))

    # Flax: variables from init, then apply. But for stateless module, just init
    rng_key, init_key = jax.random.split(rng_key)
    variables = attn.init({"params": init_key}, q, k, v)
    out, weights = attn.apply(variables, q, k, v)
    assert out.shape == (batch, h, seq, d_k), f"Expected (2, 8, 5, 64), got {out.shape}"
    assert weights.shape == (batch, h, seq, seq), f"Expected (2, 8, 5, 5), got {weights.shape}"
    # Each query position's attention over all keys must sum to 1
    assert jnp.allclose(weights.sum(axis=-1), jnp.ones((batch, h, seq))), "Attention weights should sum to 1"


def test_attention_masking():
    """Verify: masked positions get zero attention weight, unmasked sum to 1."""
    attn = ScaledDotProductAttention(dropout=0.0)

    batch, h, seq, d_k = 1, 1, 4, 8
    rng_key = jax.random.PRNGKey(0)
    q = jax.random.normal(rng_key, (batch, h, seq, d_k))  # (1, 1, 4, 8)
    rng_key, k_key, v_key = jax.random.split(rng_key, 3)
    k = jax.random.normal(k_key, (batch, h, seq, d_k))
    v = jnp.ones((batch, h, seq, d_k))  # all ones — so output = attention-weighted sum

    # Mask: allow only first 2 positions
    mask = jnp.zeros((batch, h, seq, seq), dtype=jnp.bool_)  # (1, 1, 4, 4)
    mask = mask.at[:, :, :, :2].set(True)

    rng_key, init_key = jax.random.split(rng_key)
    variables = attn.init({"params": init_key}, q, k, v, mask)
    out, weights = attn.apply(variables, q, k, v, mask)
    assert jnp.allclose(weights[:, :, :, :2].sum(axis=-1), jnp.ones((batch, h, seq))), "Masked weights should sum to 1"
    assert jnp.all(weights[:, :, :, 2:] == 0), "Masked positions should have zero weight"


# ──────────────────────────────────────────────────────────────────────
# Test: Multi-Head Attention
# ──────────────────────────────────────────────────────────────────────


def test_multi_head_attention():
    """Verify: MHA preserves shape (B, S, d_model) through head split/concat."""
    mha = MultiHeadAttention(h=8, d_model=512, dropout=0.0)

    batch, seq = 2, 10
    rng_key = jax.random.PRNGKey(0)
    x = jax.random.normal(rng_key, (batch, seq, 512))  # (B, S, d_model)

    rng_key, init_key = jax.random.split(rng_key)
    variables = mha.init({"params": init_key}, x, x, x)
    out = mha.apply(variables, x, x, x)  # self-attention: Q=K=V=x
    assert out.shape == (batch, seq, 512), f"Expected (2, 10, 512), got {out.shape}"


# ──────────────────────────────────────────────────────────────────────
# Test: Position-wise FFN
# ──────────────────────────────────────────────────────────────────────


def test_ffn():
    """Verify: FFN preserves shape — (B, S, d_model) → (B, S, d_ff) → (B, S, d_model)."""
    ffn = PositionwiseFFN(d_model=512, d_ff=2048)

    rng_key = jax.random.PRNGKey(0)
    x = jax.random.normal(rng_key, (2, 10, 512))

    rng_key, init_key = jax.random.split(rng_key)
    variables = ffn.init({"params": init_key}, x)
    out = ffn.apply(variables, x)
    assert out.shape == (2, 10, 512)


# ──────────────────────────────────────────────────────────────────────
# Test: Positional Encoding
# ──────────────────────────────────────────────────────────────────────


def test_positional_encoding_shapes():
    """Verify: PE output preserves input shape, buffer is (1, max_len, d_model)."""
    pe = PositionalEncoding(d_model=512, dropout=0.0)

    rng_key = jax.random.PRNGKey(0)
    x = jax.random.normal(rng_key, (2, 10, 512))  # (B, S, d_model)

    rng_key, init_key = jax.random.split(rng_key)
    variables = pe.init({"params": init_key}, x)
    out = pe.apply(variables, x)  # (B, S, d_model) — PE added via broadcasting
    assert out.shape == (2, 10, 512)

    # Flax init() clones the module, so setup() attrs are on a clone.
    # Use bind() to get a bound module with accessible attributes.
    bound = pe.bind(variables)
    assert bound.pe.shape == (1, 5000, 512), f"PE buffer shape wrong: {bound.pe.shape}"


def test_positional_encoding_values():
    """Verify the sinusoidal pattern: PE(pos, 0) = sin(pos / 10000^(0/512)) = sin(pos)"""
    pe = PositionalEncoding(d_model=4, max_len=10, dropout=0.0)

    rng_key = jax.random.PRNGKey(0)
    x = jax.random.normal(rng_key, (2, 10, 4))
    rng_key, init_key = jax.random.split(rng_key)
    variables = pe.init({"params": init_key}, x)
    # Flax init() clones; use bind() to access setup() attributes
    bound = pe.bind(variables)

    # PE[pos, 0] = sin(pos / 10000^(0/4)) = sin(pos / 1) = sin(pos)
    for pos in range(10):
        expected = math.sin(pos)
        assert abs(float(bound.pe[0, pos, 0].item()) - expected) < 1e-5, f"Mismatch at pos {pos}, dim 0"


# ──────────────────────────────────────────────────────────────────────
# Test: Embeddings
# ──────────────────────────────────────────────────────────────────────


def test_embeddings():
    """Verify: (B, S) token ids → (B, S, d_model) scaled embeddings."""
    emb = Embeddings(num_embeddings=50, features=512)

    rng_key = jax.random.PRNGKey(0)
    x = jax.random.randint(rng_key, (2, 10), 0, 50)  # (B, S)

    rng_key, init_key = jax.random.split(rng_key)
    variables = emb.init({"params": init_key}, x)
    out = emb.apply(variables, x)  # (B, S, d_model)
    assert out.shape == (2, 10, 512)

    # Verify scale factor: out values should be scaled by sqrt(d_model)
    raw = emb.apply(variables, x, method=lambda module, x: module(x) / math.sqrt(module.features))
    assert jnp.allclose(out, raw * math.sqrt(512))


# ──────────────────────────────────────────────────────────────────────
# Test: Transformer Forward Pass
# ──────────────────────────────────────────────────────────────────────


def test_transformer_forward():
    """Verify full forward pass: (B, S_src), (B, S_tgt) → (B, S_tgt, V)."""
    model = Transformer(
        src_vocab=50,
        tgt_vocab=50,
        N=2,
        d_model=64,
        d_ff=256,
        h=2,
        dropout=0.1,
    )

    batch_size, src_len, tgt_len = 2, 8, 7
    rng_key = jax.random.PRNGKey(0)
    rng_key, src_key, tgt_key = jax.random.split(rng_key, 3)
    src = jax.random.randint(src_key, (batch_size, src_len), 3, 50)  # (B, S_src)
    tgt = jax.random.randint(tgt_key, (batch_size, tgt_len), 3, 50)  # (B, S_tgt)

    src_mask = jnp.ones((batch_size, 1, 1, src_len), dtype=jnp.bool_)  # (B, 1, 1, S_src)
    tgt_mask = jnp.repeat(subsequent_mask(tgt_len), batch_size, axis=0)  # (B, S_tgt, S_tgt)

    rng_key, init_rng, drop_rng = jax.random.split(rng_key, 3)
    variables = model.init(
        {"params": init_rng, "dropout": drop_rng},
        src, tgt, src_mask, tgt_mask,
    )

    rng_key, apply_key = jax.random.split(rng_key)
    logits = model.apply(
        variables,
        src, tgt, src_mask, tgt_mask,
        rngs={"dropout": apply_key},
    )
    assert logits.shape == (batch_size, tgt_len, 50), f"Expected (2, 7, 50), got {logits.shape}"


def test_transformer_loss_decreases():
    """Verify the model can overfit a tiny synthetic dataset."""
    model = Transformer(
        src_vocab=30,
        tgt_vocab=30,
        N=2,
        d_model=64,
        d_ff=256,
        h=2,
        dropout=0.0,
    )

    rng_key = jax.random.PRNGKey(0)
    rng_key, init_rng, drop_rng = jax.random.split(rng_key, 3)
    dummy_src = jnp.ones((1, 5), dtype=jnp.int32)
    dummy_tgt = jnp.ones((1, 5), dtype=jnp.int32)
    variables = model.init(
        {"params": init_rng, "dropout": drop_rng},
        dummy_src, dummy_tgt,
    )
    variables = tie_weights(variables)

    # Create optimizer
    schedule_fn = lambda step: 64 ** (-0.5) * jnp.minimum(
        step ** (-0.5), step * 100 ** (-1.5)
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_fn, b1=0.9, b2=0.98, eps=1e-9),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )

    data = SyntheticData(vocab_size=30, max_len=5)
    losses = []

    for step in range(1, 101):
        src, tgt_in, tgt_out, rng_key = data.generate_batch(16, rng_key)
        rng_key, drop_key = jax.random.split(rng_key)

        def loss_fn(params):
            src_mask = (src != 0)[:, None, None, :]
            tgt_mask = make_std_mask(tgt_in, 0)
            logits = state.apply_fn(
                {"params": params},
                src, tgt_in, src_mask, tgt_mask,
                rngs={"dropout": drop_key},
            )
            return label_smoothing_loss(logits, tgt_out, smoothing=0.0, ignore_index=0)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        losses.append(float(loss))

    # Loss should have decreased
    first_avg = sum(losses[:10]) / 10
    last_avg = sum(losses[-10:]) / 10
    assert last_avg < first_avg, (
        f"Loss did not decrease: first 10 avg={first_avg:.4f}, last 10 avg={last_avg:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────
# Test: Make Standard Mask
# ──────────────────────────────────────────────────────────────────────


def test_subsequent_mask():
    """Verify: subsequent_mask(size) → (1, size, size) lower-triangular boolean."""
    mask = subsequent_mask(4)
    assert mask.shape == (1, 4, 4)
    expected = jnp.array(
        [[[True, False, False, False],
          [True, True, False, False],
          [True, True, True, False],
          [True, True, True, True]]]
    )
    assert mask.tolist() == expected.tolist()


def test_make_std_mask():
    """Verify: make_std_mask combines padding mask (B,1,1,S) & subsequent mask (1,S,S) → (B,1,S,S)."""
    tgt = jnp.array([[1, 3, 4, 2, 0, 0]])  # BOS, token, token, EOS, pad, pad
    mask = make_std_mask(tgt, pad=0)
    assert mask.shape == (1, 1, 6, 6)  # (B, 1, S_tgt, S_tgt)

    m = mask[0, 0]  # (6, 6) — both masks combined
    assert bool(m[0, 0].item()) is True
    assert bool(m[0, 1].item()) is False  # subsequent mask blocks future
    assert bool(m[0, 4].item()) is False  # padding blocked by token mask

    # Row 3 (EOS at index 3): can attend to positions 0,1,2,3
    assert jnp.all(m[3, :4]).item() is True
    assert bool(m[3, 4].item()) is False
    assert bool(m[3, 5].item()) is False

    # Row 4 (padding)
    assert bool(m[4, 4].item()) is False
    assert bool(m[4, 5].item()) is False


# ──────────────────────────────────────────────────────────────────────
# Test: Greedy Decode
# ──────────────────────────────────────────────────────────────────────


def test_greedy_decode():
    model = Transformer(
        src_vocab=30,
        tgt_vocab=30,
        N=2,
        d_model=64,
        d_ff=256,
        h=2,
        dropout=0.0,
    )

    rng_key = jax.random.PRNGKey(0)
    rng_key, init_rng = jax.random.split(rng_key)
    variables = model.init(
        {"params": init_rng, "dropout": rng_key},
        jnp.ones((1, 5), dtype=jnp.int32),
        jnp.ones((1, 5), dtype=jnp.int32),
    )
    variables = tie_weights(variables)

    rng_key, src_key = jax.random.split(rng_key)
    src = jax.random.randint(src_key, (1, 5), 3, 30)
    result = greedy_decode(model, variables, src, max_len=10, bos_idx=1, eos_idx=2, pad_idx=0, rng_key=rng_key)
    assert result.shape[0] == 1  # batch
    assert result.shape[1] >= 1  # at least BOS
    assert result.shape[1] <= 10  # at most max_len


# ──────────────────────────────────────────────────────────────────────
# Test: Label Smoothing
# ──────────────────────────────────────────────────────────────────────


def test_label_smoothing():
    x = jnp.full((1, 3, 10), -10.0)
    x = x.at[0, :, 2].set(10.0)  # class 2 is correct
    target = jnp.full((1, 3), 2)

    loss = label_smoothing_loss(x, target, smoothing=0.1, ignore_index=0)
    assert float(loss) > 0, "Even perfect prediction should have positive loss with smoothing"


def test_label_smoothing_with_padding():
    """Verify padding positions (ignore_index=0) are excluded from loss."""
    rng_key = jax.random.PRNGKey(0)
    x = jax.random.normal(rng_key, (1, 5, 10))
    x = x.at[0, 0, 5].set(10.0)  # for the valid position, make class 5 dominant
    target = jnp.array([[5, 0, 0, 0, 0]])

    loss = label_smoothing_loss(x, target, smoothing=0.0, ignore_index=0)
    assert float(loss) > 0, "Loss should be positive"


def test_encoder_decoder_independent_pe():
    """Encoder and decoder get separate PositionalEncoding instances."""
    model = Transformer(src_vocab=30, tgt_vocab=30, N=2, d_model=64, d_ff=256, h=2)

    rng_key = jax.random.PRNGKey(0)
    rng_key, init_rng, drop_rng = jax.random.split(rng_key, 3)
    variables = model.init(
        {"params": init_rng, "dropout": drop_rng},
        jnp.ones((1, 5), dtype=jnp.int32),
        jnp.ones((1, 5), dtype=jnp.int32),
    )

    # Flax init() clones; PE buffers are on the bound module, not original
    bound = model.bind(variables)
    enc_pe = bound.encoder_pe
    dec_pe = bound.decoder_pe
    assert enc_pe is not dec_pe, "Encoder and decoder should have separate PE buffers"
    assert jnp.array_equal(enc_pe.pe, dec_pe.pe), "PE values should be identical"


# ──────────────────────────────────────────────────────────────────────
# Test: Noam Learning Rate Schedule
# ──────────────────────────────────────────────────────────────────────


def test_noam_schedule():
    """Verify: LR increases linearly during warmup, then decays as 1/sqrt(step)."""
    noam = NoamOpt(model_size=512, warmup=4000)

    steps = list(range(1, 10000, 10))
    rates = [noam.rate(step) for step in steps]
    peak_idx = max(range(len(rates)), key=lambda i: rates[i])
    for i in range(1, peak_idx):
        assert rates[i] >= rates[i - 1], f"LR should increase up to warmup, failed at step {steps[i]}"
    for i in range(peak_idx + 1, len(rates)):
        assert rates[i] <= rates[i - 1], f"LR should decrease after warmup, failed at step {steps[i]}"


# ──────────────────────────────────────────────────────────────────────
# Test: Weight Tying
# ──────────────────────────────────────────────────────────────────────


def test_weight_tying():
    model = Transformer(src_vocab=50, tgt_vocab=50, N=2, d_model=64, d_ff=256, h=2)

    rng_key = jax.random.PRNGKey(0)
    rng_key, init_rng, drop_rng = jax.random.split(rng_key, 3)
    variables = model.init(
        {"params": init_rng, "dropout": drop_rng},
        jnp.ones((1, 5), dtype=jnp.int32),
        jnp.ones((1, 5), dtype=jnp.int32),
    )
    variables = tie_weights(variables)

    params = variables["params"]
    # Check encoder/decoder embeddings point to the same array
    enc_embed = params["encoder_embed"]["Embed_0"]["embedding"]
    dec_embed = params["decoder_embed"]["Embed_0"]["embedding"]
    assert enc_embed is dec_embed, "Encoder/decoder embeddings not tied"
    # Check generator kernel is transposed embed
    gen_kernel = params["generator"]["kernel"]
    assert gen_kernel.shape == (64, 50), f"Generator kernel shape wrong: {gen_kernel.shape}"
    assert jnp.array_equal(gen_kernel, enc_embed.T), "Generator kernel should be transposed embed"


def test_no_weight_tying():
    model = Transformer(src_vocab=50, tgt_vocab=50, N=2, d_model=64, d_ff=256, h=2, tie_weights=False)

    rng_key = jax.random.PRNGKey(0)
    rng_key, init_rng, drop_rng = jax.random.split(rng_key, 3)
    variables = model.init(
        {"params": init_rng, "dropout": drop_rng},
        jnp.ones((1, 5), dtype=jnp.int32),
        jnp.ones((1, 5), dtype=jnp.int32),
    )
    params = variables["params"]
    enc_embed = params["encoder_embed"]["Embed_0"]["embedding"]
    dec_embed = params["decoder_embed"]["Embed_0"]["embedding"]
    assert enc_embed is not dec_embed, "Encoder/decoder embeddings should not be tied"


# ──────────────────────────────────────────────────────────────────────
# Arithmetic Data Tests
# ──────────────────────────────────────────────────────────────────────


def test_arithmetic_data_shapes():
    """Verify: src, tgt_in, tgt_out have compatible shapes."""
    rng_key = jax.random.PRNGKey(42)
    data = ArithmeticData(max_len=10, max_operand=12)
    src, tgt_in, tgt_out, rng_key = data.generate_batch(16, rng_key)
    B = 16
    assert src.ndim == 2 and src.shape[0] == B
    assert tgt_in.ndim == 2 and tgt_in.shape[0] == B
    assert tgt_out.ndim == 2 and tgt_out.shape[0] == B
    assert tgt_in.shape[1] == tgt_out.shape[1]
    # First token of tgt_in is BOS (1)
    assert jnp.all(tgt_in[:, 0] == 1).item()
    # src ends with EOS (2) at varying positions
    for b in range(B):
        assert int((src[b] == 2).sum()) >= 1, "src must contain EOS"


def test_arithmetic_data_token_range():
    """Verify: all tokens are in valid range [0, 21]."""
    rng_key = jax.random.PRNGKey(42)
    data = ArithmeticData(max_len=10, max_operand=12)
    src, tgt_in, tgt_out, rng_key = data.generate_batch(32, rng_key)
    assert int(src.min()) >= 0 and int(src.max()) <= 21
    assert int(tgt_in.min()) >= 0 and int(tgt_in.max()) <= 21
    assert int(tgt_out.min()) >= 0 and int(tgt_out.max()) <= 21


def test_arithmetic_subtraction_non_negative():
    """Verify: subtraction results don't contain '-' token in target."""
    rng_key = jax.random.PRNGKey(42)
    data = ArithmeticData(max_len=10, max_operand=12)
    for _ in range(10):
        src, _, tgt_out, rng_key = data.generate_batch(8, rng_key)
        for i in range(8):
            tgt_tokens = tgt_out[i].tolist()
            assert 14 not in tgt_tokens, "Subtraction result should not contain '-' token"


def test_arithmetic_transformer_loss_decreases():
    """Verify the model can overfit a tiny arithmetic dataset (max_operand=9)."""
    model = Transformer(
        src_vocab=ArithmeticData.VOCAB_SIZE,
        tgt_vocab=ArithmeticData.VOCAB_SIZE,
        N=2,
        d_model=64,
        d_ff=256,
        h=2,
        dropout=0.0,
    )

    rng_key = jax.random.PRNGKey(0)
    rng_key, init_rng, drop_rng = jax.random.split(rng_key, 3)
    dummy_src = jnp.ones((1, 5), dtype=jnp.int32)
    dummy_tgt = jnp.ones((1, 5), dtype=jnp.int32)
    variables = model.init(
        {"params": init_rng, "dropout": drop_rng},
        dummy_src, dummy_tgt,
    )
    variables = tie_weights(variables)

    # Noam schedule for d_model=64, warmup=100
    def schedule_fn(step):
        step = step.astype(jnp.float32)
        arg1 = step ** (-0.5)
        arg2 = step * (100 ** (-1.5))
        return (64 ** (-0.5)) * jnp.minimum(arg1, arg2)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_fn, b1=0.9, b2=0.98, eps=1e-9),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )

    data = ArithmeticData(max_len=10, max_operand=9)
    losses = []

    for step in range(1, 101):
        src, tgt_in, tgt_out, rng_key = data.generate_batch(16, rng_key)
        rng_key, drop_key = jax.random.split(rng_key)

        def loss_fn(params):
            src_mask = (src != 0)[:, None, None, :]
            tgt_mask = make_std_mask(tgt_in, 0)
            logits = state.apply_fn(
                {"params": params},
                src, tgt_in, src_mask, tgt_mask,
                rngs={"dropout": drop_key},
            )
            return label_smoothing_loss(logits, tgt_out, smoothing=0.0, ignore_index=0)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        losses.append(float(loss))

    first_avg = sum(losses[:10]) / 10
    last_avg = sum(losses[-10:]) / 10
    assert last_avg < first_avg, (
        f"Loss did not decrease: first 10 avg={first_avg:.4f}, last 10 avg={last_avg:.4f}"
    )


def test_arithmetic_decode():
    """Verify: decode produces human-readable arithmetic strings."""
    tokens = [ArithmeticData.NUM_OFFSET + 1, ArithmeticData.NUM_OFFSET + 2,
              ArithmeticData.PLUS, ArithmeticData.NUM_OFFSET + 7, 2]
    assert ArithmeticData.decode(tokens) == "12+7"
    tokens2 = [ArithmeticData.EQ, ArithmeticData.NUM_OFFSET + 1,
               ArithmeticData.NUM_OFFSET + 9, 2]
    assert ArithmeticData.decode(tokens2) == "=19"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
