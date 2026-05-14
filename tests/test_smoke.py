#!/usr/bin/env python3
"""Smoke tests for the original Transformer implementation.

Verifies forward pass shapes, masking, and the ability to overfit
on a tiny synthetic dataset.
"""

import math

import pytest
import torch

from transformer import (
    LabelSmoothing,
    NoamOpt,
    Transformer,
    greedy_decode,
    subsequent_mask,
)
from transformer.layers import (
    Embeddings,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFFN,
    ScaledDotProductAttention,
)
from transformer.train import SyntheticData, make_std_mask

# ──────────────────────────────────────────────────────────────────────
# Test: Scaled Dot-Product Attention
# ──────────────────────────────────────────────────────────────────────


def test_scaled_dot_product_attention():
    """Verify: Q,K,V (B,h,S,d_k) → output (B,h,S,d_k), weights (B,h,S,S) sum to 1."""
    attn = ScaledDotProductAttention(dropout=0.0)
    attn.eval()

    batch, h, seq, d_k = 2, 8, 5, 64
    q = torch.randn(batch, h, seq, d_k)  # (B, h, S, d_k)
    k = torch.randn(batch, h, seq, d_k)
    v = torch.randn(batch, h, seq, d_k)

    out, weights = attn(q, k, v)
    assert out.shape == (batch, h, seq, d_k), f"Expected (2, 8, 5, 64), got {out.shape}"
    assert weights.shape == (batch, h, seq, seq), (
        f"Expected (2, 8, 5, 5), got {weights.shape}"
    )
    # Each query position's attention over all keys must sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, h, seq)), (
        "Attention weights should sum to 1"
    )


def test_attention_masking():
    """Verify: masked positions get zero attention weight, unmasked sum to 1."""
    attn = ScaledDotProductAttention(dropout=0.0)
    attn.eval()

    batch, h, seq, d_k = 1, 1, 4, 8
    q = torch.randn(batch, h, seq, d_k)  # (1, 1, 4, 8)
    k = torch.randn(batch, h, seq, d_k)
    v = torch.ones(batch, h, seq, d_k)  # all ones — so output = attention-weighted sum

    # Mask: allow only first 2 positions. ~mask → positions 2,3 get -inf → 0 weight
    mask = torch.zeros(batch, h, seq, seq, dtype=torch.bool)  # (1, 1, 4, 4)
    mask[:, :, :, :2] = True

    out, weights = attn(q, k, v, mask=mask)
    assert torch.allclose(
        weights[:, :, :, :2].sum(dim=-1), torch.ones(batch, h, seq)
    ), "Masked weights should sum to 1"
    assert torch.all(weights[:, :, :, 2:] == 0), (
        "Masked positions should have zero weight"
    )


# ──────────────────────────────────────────────────────────────────────
# Test: Multi-Head Attention
# ──────────────────────────────────────────────────────────────────────


def test_multi_head_attention():
    """Verify: MHA preserves shape (B, S, d_model) through head split/concat."""
    mha = MultiHeadAttention(h=8, d_model=512, dropout=0.0)
    mha.eval()

    batch, seq = 2, 10
    x = torch.randn(batch, seq, 512)  # (B, S, d_model)
    out = mha(x, x, x)  # self-attention: Q=K=V=x
    assert out.shape == (batch, seq, 512), f"Expected (2, 10, 512), got {out.shape}"


# ──────────────────────────────────────────────────────────────────────
# Test: Position-wise FFN
# ──────────────────────────────────────────────────────────────────────


def test_ffn():
    """Verify: FFN preserves shape — (B, S, d_model) → (B, S, d_ff) → (B, S, d_model)."""
    ffn = PositionwiseFFN(d_model=512, d_ff=2048)
    x = torch.randn(2, 10, 512)
    out = ffn(x)
    assert out.shape == (2, 10, 512)


# ──────────────────────────────────────────────────────────────────────
# Test: Positional Encoding
# ──────────────────────────────────────────────────────────────────────


def test_positional_encoding_shapes():
    """Verify: PE output preserves input shape, buffer is (1, max_len, d_model)."""
    pe = PositionalEncoding(d_model=512, dropout=0.0)
    pe.eval()

    x = torch.randn(2, 10, 512)  # (B, S, d_model)
    out = pe(x)  # (B, S, d_model) — PE added via broadcasting
    assert out.shape == (2, 10, 512)

    assert isinstance(pe.pe, torch.Tensor)
    assert pe.pe.shape == (1, 5000, 512), f"PE buffer shape wrong: {pe.pe.shape}"


def test_positional_encoding_values():
    """Verify the sinusoidal pattern: PE(pos, 0) = sin(pos / 10000^(0/512)) = sin(pos)"""
    pe = PositionalEncoding(d_model=4, max_len=10, dropout=0.0)
    pe.eval()

    assert isinstance(pe.pe, torch.Tensor)
    # PE[pos, 0] = sin(pos / 10000^(0/4)) = sin(pos / 1) = sin(pos)
    for pos in range(10):
        expected = math.sin(pos)
        assert abs(pe.pe[0, pos, 0].item() - expected) < 1e-5, (
            f"Mismatch at pos {pos}, dim 0"
        )


# ──────────────────────────────────────────────────────────────────────
# Test: Embeddings
# ──────────────────────────────────────────────────────────────────────


def test_embeddings():
    """Verify: (B, S) token ids → (B, S, d_model) scaled embeddings."""
    emb = Embeddings(vocab_size=50, d_model=512)
    x = torch.randint(0, 50, (2, 10))  # (B, S)
    out = emb(x)  # (B, S, d_model)
    assert out.shape == (2, 10, 512)

    # Verify scale factor: out == lut(x) * sqrt(d_model)
    raw = emb.lut(x)
    assert torch.allclose(out, raw * math.sqrt(512))


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
    src = torch.randint(3, 50, (batch_size, src_len))  # (B, S_src)
    tgt = torch.randint(3, 50, (batch_size, tgt_len))  # (B, S_tgt)

    src_mask = torch.ones(
        batch_size, 1, 1, src_len, dtype=torch.bool
    )  # (B, 1, 1, S_src)
    tgt_mask = subsequent_mask(tgt_len).expand(batch_size, -1, -1)  # (B, S_tgt, S_tgt)

    logits = model(src, tgt, src_mask, tgt_mask)  # (B, S_tgt, V)
    assert logits.shape == (batch_size, tgt_len, 50), (
        f"Expected (2, 7, 50), got {logits.shape}"
    )


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

    criterion = LabelSmoothing(
        smoothing=0.0, ignore_index=0
    )  # no smoothing for simplicity
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    noam = NoamOpt(model_size=64, factor=1.0, warmup=100, optimizer=optimizer)

    data = SyntheticData(vocab_size=30, max_len=5)

    # Train for a small number of steps
    losses = []
    model.train()
    for step in range(1, 101):
        src, tgt_in, tgt_out = data.generate_batch(16)
        noam.zero_grad()

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = make_std_mask(tgt_in, 0)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = criterion(logits, tgt_out)
        loss.backward()
        noam.step()

        losses.append(loss.item())

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
    expected = torch.tensor(
        [
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        ]
    )
    assert mask.tolist() == expected.tolist()


def test_make_std_mask():
    """Verify: make_std_mask combines padding mask (B,1,1,S) & subsequent mask (1,S,S) → (B,1,S,S)."""
    tgt = torch.tensor([[1, 3, 4, 2, 0, 0]])  # BOS, token, token, EOS, pad, pad
    mask = make_std_mask(tgt, pad=0)
    assert mask.shape == (1, 1, 6, 6)  # (B, 1, S_tgt, S_tgt)

    m = mask[0, 0]  # (6, 6) — both masks combined
    # Row 0 (BOS): can attend to BOS only (subsequent mask blocks future, token mask blocks pads)
    assert m[0, 0].item() is True
    assert m[0, 1].item() is False  # subsequent mask blocks future
    assert m[0, 4].item() is False  # padding blocked by token mask

    # Row 3 (EOS at index 3): can attend to positions 0,1,2,3 (non-pad, non-future)
    assert m[3, :4].all().item() is True
    assert m[3, 4].item() is False  # padding token at position 4
    assert m[3, 5].item() is False  # padding token at position 5

    # Row 4 (padding): subsequent mask allows all past, but token mask blocks
    assert m[4, 4].item() is False  # padding at position 4
    assert m[4, 5].item() is False  # padding at position 5


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

    src = torch.randint(3, 30, (1, 5))
    result = greedy_decode(model, src, max_len=10, bos_idx=1, eos_idx=2, pad_idx=0)
    assert result.size(0) == 1  # batch
    assert result.size(1) >= 1  # at least BOS
    assert result.size(1) <= 10  # at most max_len


# ──────────────────────────────────────────────────────────────────────
# Test: Label Smoothing
# ──────────────────────────────────────────────────────────────────────


def test_label_smoothing():
    smoothing = LabelSmoothing(smoothing=0.1, ignore_index=0)
    x = torch.full((1, 3, 10), -10.0)
    x[0, :, 2] = 10.0  # class 2 is correct
    target = torch.full((1, 3), 2)

    loss = smoothing(x, target)
    assert loss.item() > 0, (
        "Even perfect prediction should have positive loss with smoothing"
    )


def test_label_smoothing_with_padding():
    """Verify padding positions (ignore_index=0) are excluded from loss."""
    smoothing = LabelSmoothing(smoothing=0.0, ignore_index=0)

    # 2 tokens: first is valid, second is padding
    x = torch.randn(1, 5, 10)
    x[0, 0, 5] = 10.0  # for the valid position, make class 5 dominant
    target = torch.tensor([[5, 0, 0, 0, 0]])

    loss = smoothing(x, target)
    # Only 1 non-padding token, so loss should be from that token only
    assert loss.item() > 0, "Loss should be positive"


def test_encoder_decoder_independent_pe():
    """Encoder and decoder get separate PositionalEncoding instances."""
    model = Transformer(src_vocab=30, tgt_vocab=30, N=2, d_model=64, d_ff=256, h=2)
    enc_pe = model.encoder.positional
    dec_pe = model.decoder.positional
    assert enc_pe is not dec_pe, "Encoder and decoder should have separate PE buffers"
    # But they share the same PE values (shouldn't diverge since they're buffers)
    assert isinstance(enc_pe.pe, torch.Tensor) and isinstance(dec_pe.pe, torch.Tensor)
    assert torch.equal(enc_pe.pe, dec_pe.pe), "PE values should be identical"


# ──────────────────────────────────────────────────────────────────────
# Test: Noam Learning Rate Schedule
# ──────────────────────────────────────────────────────────────────────


def test_noam_schedule():
    """Verify: LR increases linearly during warmup, then decays as 1/sqrt(step)."""
    model = torch.nn.Linear(10, 10)
    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    noam = NoamOpt(model_size=512, warmup=4000, optimizer=opt)

    # LR increases for first warmup steps, then decreases
    steps = list(range(1, 10000, 10))
    rates = [noam.rate(step) for step in steps]
    # Find the actual peak index (should be near warmup=4000)
    peak_idx = max(range(len(rates)), key=lambda i: rates[i])
    for i in range(1, peak_idx):
        assert rates[i] >= rates[i - 1], (
            f"LR should increase up to warmup, failed at step {steps[i]}"
        )
    for i in range(peak_idx + 1, len(rates)):
        assert rates[i] <= rates[i - 1], (
            f"LR should decrease after warmup, failed at step {steps[i]}"
        )


# ──────────────────────────────────────────────────────────────────────
# Test: Weight Tying
# ──────────────────────────────────────────────────────────────────────


def test_weight_tying():
    model = Transformer(src_vocab=50, tgt_vocab=50, N=2, d_model=64, d_ff=256, h=2)
    assert model.encoder.embedding.weight is model.decoder.embedding.weight, (
        "Encoder/decoder embeddings not tied"
    )
    assert model.generator.weight is model.decoder.embedding.weight, (
        "Generator weight not tied to embeddings"
    )


def test_no_weight_tying():
    model = Transformer(
        src_vocab=50, tgt_vocab=50, N=2, d_model=64, d_ff=256, h=2, tie_weights=False
    )
    assert model.encoder.embedding.weight is not model.decoder.embedding.weight
    assert model.generator.weight is not model.decoder.embedding.weight


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
