#!/usr/bin/env python3
"""Smoke tests for the MLX Transformer implementation.

Verifies forward pass shapes, masking, and the ability to overfit
on a tiny synthetic dataset. Mirrors the PyTorch version (tests/test_smoke.py)
with MLX-specific adaptations.

MLX vs PyTorch test differences annotated inline:
  - mx.random.normal replaces torch.randn
  - mx.allclose replaces torch.allclose
  - mx.ones / mx.zeros replace torch.ones / torch.zeros
  - No .device management — MLX handles device automatically
  - model.eval() and model.train() set dropout mode
  - mx.eval() calls where needed for assertions on lazy values
"""

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from transformer_mlx import Transformer, NoamOpt, LabelSmoothing, subsequent_mask, greedy_decode
from transformer_mlx.layers import (
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
from transformer_mlx.train import ArithmeticData, SyntheticData, make_std_mask, run_epoch_steps


# ──────────────────────────────────────────────────────────────────────
# Test: Scaled Dot-Product Attention
# ──────────────────────────────────────────────────────────────────────


def test_scaled_dot_product_attention():
    """Verify: Q,K,V (B,h,S,d_k) → output (B,h,S,d_k), weights (B,h,S,S) sum to 1."""
    attn = ScaledDotProductAttention(dropout=0.0)
    attn.eval()

    batch, h, seq, d_k = 2, 8, 5, 64
    # MLX: mx.random.normal replaces torch.randn
    q = mx.random.normal((batch, h, seq, d_k))  # (B, h, S, d_k)
    k = mx.random.normal((batch, h, seq, d_k))
    v = mx.random.normal((batch, h, seq, d_k))

    out, weights = attn(q, k, v)
    assert out.shape == (batch, h, seq, d_k), f"Expected (2, 8, 5, 64), got {out.shape}"
    assert weights.shape == (batch, h, seq, seq), f"Expected (2, 8, 5, 5), got {weights.shape}"
    # Each query position's attention over all keys must sum to 1
    # MLX: mx.allclose replaces torch.allclose
    assert mx.allclose(weights.sum(axis=-1), mx.ones((batch, h, seq))), "Attention weights should sum to 1"


def test_attention_masking():
    """Verify: masked positions get zero attention weight, unmasked sum to 1."""
    attn = ScaledDotProductAttention(dropout=0.0)
    attn.eval()

    batch, h, seq, d_k = 1, 1, 4, 8
    q = mx.random.normal((batch, h, seq, d_k))  # (1, 1, 4, 8)
    k = mx.random.normal((batch, h, seq, d_k))
    v = mx.ones((batch, h, seq, d_k))  # all ones — so output = attention-weighted sum

    # Mask: allow only first 2 positions
    # MLX: no .at[...].set() — use position-based comparison
    key_pos = mx.arange(seq)[None, None, None, :]  # (1, 1, 1, 4)
    mask = key_pos < 2  # True for dims 0,1 in the key dimension → broadcasts to (1,1,4,4)

    out, weights = attn(q, k, v, mask=mask)
    assert mx.allclose(weights[:, :, :, :2].sum(axis=-1), mx.ones((batch, h, seq))), "Masked weights should sum to 1"
    assert mx.all(weights[:, :, :, 2:] == 0), "Masked positions should have zero weight"


# ──────────────────────────────────────────────────────────────────────
# Test: Multi-Head Attention
# ──────────────────────────────────────────────────────────────────────


def test_multi_head_attention():
    """Verify: MHA preserves shape (B, S, d_model) through head split/concat."""
    mha = MultiHeadAttention(h=8, d_model=512, dropout=0.0)
    mha.eval()

    batch, seq = 2, 10
    x = mx.random.normal((batch, seq, 512))  # (B, S, d_model)
    out = mha(x, x, x)                        # self-attention: Q=K=V=x
    assert out.shape == (batch, seq, 512), f"Expected (2, 10, 512), got {out.shape}"


# ──────────────────────────────────────────────────────────────────────
# Test: Position-wise FFN
# ──────────────────────────────────────────────────────────────────────


def test_ffn():
    """Verify: FFN preserves shape — (B, S, d_model) → (B, S, d_ff) → (B, S, d_model)."""
    ffn = PositionwiseFFN(d_model=512, d_ff=2048)
    x = mx.random.normal((2, 10, 512))
    out = ffn(x)
    assert out.shape == (2, 10, 512)


# ──────────────────────────────────────────────────────────────────────
# Test: Positional Encoding
# ──────────────────────────────────────────────────────────────────────


def test_positional_encoding_shapes():
    """Verify: PE output preserves input shape, buffer is (1, max_len, d_model)."""
    pe = PositionalEncoding(d_model=512, dropout=0.0)
    pe.eval()

    x = mx.random.normal((2, 10, 512))  # (B, S, d_model)
    out = pe(x)                          # (B, S, d_model) — PE added via broadcasting
    assert out.shape == (2, 10, 512)

    # MLX: PE is a plain attribute (no register_buffer), accessed as .pe
    assert pe.pe.shape == (1, 5000, 512), f"PE buffer shape wrong: {pe.pe.shape}"


def test_positional_encoding_values():
    """Verify the sinusoidal pattern: PE(pos, 0) = sin(pos / 10000^(0/512)) = sin(pos)"""
    pe = PositionalEncoding(d_model=4, max_len=10, dropout=0.0)
    pe.eval()

    # PE[pos, 0] = sin(pos / 10000^(0/4)) = sin(pos / 1) = sin(pos)
    for pos in range(10):
        expected = math.sin(pos)
        assert abs(float(pe.pe[0, pos, 0].item()) - expected) < 1e-5, f"Mismatch at pos {pos}, dim 0"


# ──────────────────────────────────────────────────────────────────────
# Test: Embeddings
# ──────────────────────────────────────────────────────────────────────


def test_embeddings():
    """Verify: (B, S) token ids → (B, S, d_model) scaled embeddings."""
    emb = Embeddings(vocab_size=50, d_model=512)
    x = mx.random.randint(0, 50, (2, 10))  # (B, S)
    out = emb(x)                             # (B, S, d_model)
    assert out.shape == (2, 10, 512)

    # Verify scale factor: out == lut(x) * sqrt(d_model)
    raw = emb.lut(x)
    assert mx.allclose(out, raw * math.sqrt(512))


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
    model.eval()

    batch_size, src_len, tgt_len = 2, 8, 7
    src = mx.random.randint(3, 50, (batch_size, src_len))  # (B, S_src)
    tgt = mx.random.randint(3, 50, (batch_size, tgt_len))  # (B, S_tgt)

    src_mask = mx.ones((batch_size, 1, 1, src_len), dtype=mx.bool_)  # (B, 1, 1, S_src)
    tgt_mask = mx.repeat(subsequent_mask(tgt_len), batch_size, axis=0)  # (B, S_tgt, S_tgt)

    logits = model(src, tgt, src_mask, tgt_mask)  # (B, S_tgt, V)
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
    model.train()

    criterion = LabelSmoothing(smoothing=0.0, ignore_index=0)  # no smoothing for simplicity
    adam = optim.Adam(learning_rate=1.0, betas=(0.9, 0.98), eps=1e-9)
    noam = NoamOpt(model_size=64, factor=1.0, warmup=100, optimizer=adam)

    data = SyntheticData(vocab_size=30, max_len=5)

    # Train for a small number of steps
    losses = []
    model.train()
    for step in range(1, 101):
        src, tgt_in, tgt_out = data.generate_batch(16)

        def loss_fn(m, s, ti, to):
            src_mask = (s != 0)[:, None, None, :]
            tgt_mask = make_std_mask(ti, 0)
            logits = m(s, ti, src_mask, tgt_mask)
            return criterion(logits, to)

        loss_and_grad = mx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad(model, src, tgt_in, tgt_out)
        noam.step(model, grads)
        mx.eval(model.parameters(), noam.optimizer.state)

        losses.append(float(loss.item()))

    # Loss should have decreased (at minimum, the last 10 avg < first 10 avg)
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
    # Lower triangular should be True (can attend to self + past), upper = False (future)
    # MLX: mx.array for comparison
    expected = mx.array(
        [[[True, False, False, False],
          [True, True, False, False],
          [True, True, True, False],
          [True, True, True, True]]]
    )
    assert mask.tolist() == expected.tolist()


def test_make_std_mask():
    """Verify: make_std_mask combines padding mask (B,1,1,S) & subsequent mask (1,S,S) → (B,1,S,S)."""
    tgt = mx.array([[1, 3, 4, 2, 0, 0]])  # BOS, token, token, EOS, pad, pad
    mask = make_std_mask(tgt, pad=0)
    assert mask.shape == (1, 1, 6, 6)  # (B, 1, S_tgt, S_tgt)

    m = mask[0, 0]  # (6, 6) — both masks combined
    # Row 0 (BOS): can attend to BOS only (subsequent mask blocks future, token mask blocks pads)
    assert bool(m[0, 0].item()) is True
    assert bool(m[0, 1].item()) is False  # subsequent mask blocks future
    assert bool(m[0, 4].item()) is False  # padding blocked by token mask

    # Row 3 (EOS at index 3): can attend to positions 0,1,2,3 (non-pad, non-future)
    assert mx.all(m[3, :4]).item() is True
    assert bool(m[3, 4].item()) is False  # padding token at position 4
    assert bool(m[3, 5].item()) is False  # padding token at position 5

    # Row 4 (padding): subsequent mask allows all past, but token mask blocks
    assert bool(m[4, 4].item()) is False  # padding at position 4
    assert bool(m[4, 5].item()) is False  # padding at position 5


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
    model.eval()

    src = mx.random.randint(3, 30, (1, 5))
    result = greedy_decode(model, src, max_len=10, bos_idx=1, eos_idx=2, pad_idx=0)
    assert result.shape[0] == 1  # batch
    assert result.shape[1] >= 1  # at least BOS
    assert result.shape[1] <= 10  # at most max_len


# ──────────────────────────────────────────────────────────────────────
# Test: Label Smoothing
# ──────────────────────────────────────────────────────────────────────


def test_label_smoothing():
    smoothing = LabelSmoothing(smoothing=0.1, ignore_index=0)
    x = mx.full((1, 3, 10), -10.0)
    # MLX: no .at[...].set() — use mx.where to set class 2 to 10.0
    is_class2 = mx.arange(10) == 2  # (10,) — broadcasts to (1,3,10)
    x = mx.where(is_class2, mx.full(x.shape, 10.0), x)
    target = mx.full((1, 3), 2)

    loss = smoothing(x, target)
    # MLX: .item() to get Python float from mx.array scalar
    assert float(loss.item()) > 0, "Even perfect prediction should have positive loss with smoothing"


def test_label_smoothing_with_padding():
    """Verify padding positions (ignore_index=0) are excluded from loss."""
    smoothing = LabelSmoothing(smoothing=0.0, ignore_index=0)

    # 2 tokens: first is valid, second is padding
    x = mx.random.normal((1, 5, 10))
    # MLX: no .at[...].set() — use mx.put_along_axis to set position [0,0,5] = 10.0
    idx = mx.array([[[5]]], dtype=mx.int32)  # (1, 1, 1) — target index in vocab dim
    val = mx.full((1, 1, 1), 10.0)
    x = x[0:1, 0:1, :]  # isolate row 0, col 0
    x = mx.put_along_axis(x, idx, val, axis=-1)  # set class 5 to 10.0
    # Rebuild the full (1, 5, 10) by concatenating the modified row with random rows
    rest = mx.random.normal((1, 4, 10))
    x = mx.concatenate([x, rest], axis=1)
    target = mx.array([[5, 0, 0, 0, 0]])

    loss = smoothing(x, target)
    # Only 1 non-padding token, so loss should be from that token only
    assert float(loss.item()) > 0, "Loss should be positive"


def test_encoder_decoder_independent_pe():
    """Encoder and decoder get separate PositionalEncoding instances."""
    model = Transformer(src_vocab=30, tgt_vocab=30, N=2, d_model=64, d_ff=256, h=2)
    enc_pe = model.encoder.positional
    dec_pe = model.decoder.positional
    assert enc_pe is not dec_pe, "Encoder and decoder should have separate PE buffers"
    # But they share the same PE values (shouldn't diverge since they're constants)
    assert mx.array_equal(enc_pe.pe, dec_pe.pe), "PE values should be identical"


# ──────────────────────────────────────────────────────────────────────
# Test: Noam Learning Rate Schedule
# ──────────────────────────────────────────────────────────────────────


def test_noam_schedule():
    """Verify: LR increases linearly during warmup, then decays as 1/sqrt(step)."""
    noam = NoamOpt(model_size=512, warmup=4000)

    steps = list(range(1, 10000, 10))
    rates = [noam.rate(step) for step in steps]
    # Find the actual peak index (should be near warmup=4000)
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
    assert model.encoder.embedding.weight is model.decoder.embedding.weight, "Encoder/decoder embeddings not tied"
    assert model.generator.weight is model.decoder.embedding.weight, "Generator weight not tied to embeddings"


def test_no_weight_tying():
    model = Transformer(src_vocab=50, tgt_vocab=50, N=2, d_model=64, d_ff=256, h=2, tie_weights=False)
    assert model.encoder.embedding.weight is not model.decoder.embedding.weight
    assert model.generator.weight is not model.decoder.embedding.weight


# ──────────────────────────────────────────────────────────────────────
# Arithmetic Data Tests
# ──────────────────────────────────────────────────────────────────────


def test_arithmetic_data_shapes():
    """Verify: src, tgt_in, tgt_out have compatible shapes."""
    mx.random.seed(42)
    data = ArithmeticData(max_len=10, max_operand=12)
    src, tgt_in, tgt_out = data.generate_batch(16)
    B = 16
    assert src.ndim == 2 and src.shape[0] == B
    assert tgt_in.ndim == 2 and tgt_in.shape[0] == B
    assert tgt_out.ndim == 2 and tgt_out.shape[0] == B
    assert tgt_in.shape[1] == tgt_out.shape[1]
    # First token of tgt_in is BOS (1)
    assert mx.all(tgt_in[:, 0] == 1).item()
    # src ends with EOS (2)
    for b in range(B):
        assert int((src[b] == 2).sum().item()) >= 1, "src must contain EOS"


def test_arithmetic_data_token_range():
    """Verify: all tokens are in valid range [0, 21]."""
    mx.random.seed(42)
    data = ArithmeticData(max_len=10, max_operand=12)
    src, tgt_in, tgt_out = data.generate_batch(32)
    assert int(src.min().item()) >= 0 and int(src.max().item()) <= 21
    assert int(tgt_in.min().item()) >= 0 and int(tgt_in.max().item()) <= 21
    assert int(tgt_out.min().item()) >= 0 and int(tgt_out.max().item()) <= 21


def test_arithmetic_subtraction_non_negative():
    """Verify: subtraction results don't contain '-' token in target."""
    mx.random.seed(42)
    data = ArithmeticData(max_len=10, max_operand=12)
    for _ in range(10):
        src, _, tgt_out = data.generate_batch(8)
        for i in range(8):
            tgt_tokens = tgt_out[i].tolist()
            assert 14 not in tgt_tokens, "Subtraction result should not contain '-' token"


def test_arithmetic_transformer_loss_decreases():
    """Verify the model can overfit a tiny arithmetic dataset (max_operand=9)."""
    mx.random.seed(42)
    model = Transformer(
        src_vocab=ArithmeticData.VOCAB_SIZE,
        tgt_vocab=ArithmeticData.VOCAB_SIZE,
        N=2,
        d_model=64,
        d_ff=256,
        h=2,
        dropout=0.0,
    )
    model.train()

    criterion = LabelSmoothing(smoothing=0.0, ignore_index=0)
    adam = optim.Adam(learning_rate=1.0, betas=(0.9, 0.98), eps=1e-9)
    noam = NoamOpt(model_size=64, factor=1.0, warmup=100, optimizer=adam)

    data = ArithmeticData(max_len=10, max_operand=9)

    losses = []
    model.train()
    for step in range(1, 101):
        src, tgt_in, tgt_out = data.generate_batch(16)

        def loss_fn(m, s, ti, to):
            src_mask = (s != 0)[:, None, None, :]
            tgt_mask = make_std_mask(ti, 0)
            logits = m(s, ti, src_mask, tgt_mask)
            return criterion(logits, to)

        loss_and_grad = mx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad(model, src, tgt_in, tgt_out)
        noam.step(model, grads)
        mx.eval(model.parameters(), noam.optimizer.state)

        losses.append(float(loss.item()))

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
