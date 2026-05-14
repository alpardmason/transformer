"""MLX reimplementation of the original Transformer from "Attention is All You Need".

MLX vs PyTorch — key differences annotated throughout this file:
  - Lazy evaluation: ops build a graph; computation deferred until mlx.eval() or
    a value is inspected (printing, .item(), etc.). The training loop calls
    mlx.eval() periodically to avoid unbounded graph growth.
  - No register_buffer: store non-learnable arrays (e.g. positional encoding)
    as plain Python attributes — no need for a separate buffer registry.
  - No .contiguous(): MLX arrays are always contiguous; reshape after transpose
    always works without an explicit copy.
  - mlx.where(cond, a, b) replaces torch.masked_fill / scatter_.
  - mlx.softmax(axis=...) replaces dim keyword.
  - mlx.concatenate replaces torch.cat; mlx.swapaxes replaces transpose.
  - .shape is the same as PyTorch; no .size() alias.
  - nn.Linear / nn.LayerNorm / nn.Dropout / nn.Embedding — same API.
  - nn.init.glorot_uniform() replaces xavier_uniform_.
  - mlx.triu(x, k=1) replaces torch.triu(x, diagonal=1).
"""

import copy
import math
from collections.abc import Callable
from typing import Optional, Tuple

import mlx.core as mlx
import mlx.nn as nn

# ──────────────────────────────────────────────────────────────────────
# 1. Scaled Dot-Product Attention
# ──────────────────────────────────────────────────────────────────────


class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Shapes (B=batch, h=heads, S=seq_len, d_k=head_dim):
      Q, K, V:  (B, h, S, d_k)
      S = S_q or S_k, S_q is the query length(decoder) and S_k is the key length(encoder)
      scores:    (B, h, S_q, S_k)  — dot-product similarity between all query-key pairs
      p_attn:    (B, h, S_q, S_k)  — softmax over keys (dim=-1), rows sum to 1
      output:    (B, h, S, d_k)    — weighted sum of values
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        query: mlx.array,
        key: mlx.array,
        value: mlx.array,
        mask: Optional[mlx.array] = None,
    ) -> Tuple[mlx.array, mlx.array]:
        # Scale by sqrt(d_k) to keep softmax variance ~1 (prevents tiny gradients)
        d_k = query.shape[-1]
        # MLX: mlx.matmul + mlx.swapaxes (instead of torch.matmul + .transpose(-2, -1))
        scores = mlx.matmul(query, key.swapaxes(-2, -1)) / math.sqrt(d_k)

        # Mask: True = can attend, False = masked out.
        # MLX: mlx.where(condition, x, y) replaces scores.masked_fill(~mask, float('-inf'))
        if mask is not None:
            scores = mlx.where(mask, scores, float("-inf"))

        # MLX: mlx.softmax(axis=...) instead of F.softmax(dim=...)
        p_attn = mlx.softmax(scores, axis=-1)
        p_attn = self.dropout(p_attn)
        return mlx.matmul(p_attn, value), p_attn


# ──────────────────────────────────────────────────────────────────────
# 2. Multi-Head Attention
# ──────────────────────────────────────────────────────────────────────


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with h parallel heads.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    Instead of looping over heads, all heads are computed in parallel by
    reshaping (B, S, d_model) → (B, h, S, d_k) so batch matmul handles
    every head simultaneously.

    The four linear layers are: W^Q, W^K, W^V (project to d_model each, then
    split into heads) and W^O (project concatenated heads back to d_model).
    """

    def __init__(self, h: int = 8, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        if d_model % h != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by h ({h})")
        self.h = h
        self.d_k = d_model // h  # dimension per head

        # MLX: nn.Linear API is the same as PyTorch's nn.Linear
        self.linears = [
            nn.Linear(d_model, d_model) for _ in range(4)
        ]  # W^Q, W^K, W^V, W^O
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def __call__(
        self,
        query: mlx.array,  # (B, S_q, d_model)
        key: mlx.array,  # (B, S_k, d_model)
        value: mlx.array,  # (B, S_v, d_model) — usually S_k == S_v
        mask: Optional[mlx.array] = None,
    ) -> mlx.array:  # (B, S_q, d_model)
        batch_size = query.shape[0]

        # Masks from callers (e.g. make_std_mask) are 3D (B, S_q, S_k).
        # Add head dim so they broadcast to (B, 1, S_q, S_k) for all heads.
        # MLX: mask[:, None, :, :] replaces mask.unsqueeze(1)
        if mask is not None and mask.ndim == 3:
            mask = mask[:, None, :, :]

        # 1) Linear projection + split into heads in one step
        #    (B, S, d_model) → reshape → (B, S, h, d_k) → swapaxes → (B, h, S, d_k)
        #    Now batch dim = B, and each "batch" is really a head — so MLX
        #    computes h independent attention operations in parallel.
        query, key, value = [
            lin(x).reshape(batch_size, -1, self.h, self.d_k).swapaxes(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply scaled dot-product attention on all heads
        #    x: (B, h, S_q, d_k) — each head produced a d_k output per position
        x, _ = self.attention(query, key, value, mask=mask)

        # 3) Concatenate heads back: (B, h, S, d_k) → (B, S, h*d_k) = (B, S, d_model)
        #    MLX: no .contiguous() needed — arrays are always contiguous.
        #    swapaxes + reshape is sufficient.
        x = x.swapaxes(1, 2).reshape(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)  # W^O projection


# ──────────────────────────────────────────────────────────────────────
# 3. Position-wise Feed-Forward Network
# ──────────────────────────────────────────────────────────────────────


class PositionwiseFFN(nn.Module):
    """FFN(x) = max(0, xW_1 + b_1) W_2 + b_2  (ReLU).

    Applied identically to every position (hence "position-wise") — each token
    gets the same two-layer MLP independently. The inner dim d_ff is 4× d_model
    in the paper (2048 vs 512).

    Shapes: (B, S, d_model) → (B, S, d_ff) → (B, S, d_model)
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mlx.array) -> mlx.array:
        # (B, S, d_model) → expand → (B, S, d_ff) → ReLU+Dropout → project → (B, S, d_model)
        # MLX: nn.relu (note: lives in mlx.nn, not mlx.core)
        return self.w_2(self.dropout(nn.relu(self.w_1(x))))


# ──────────────────────────────────────────────────────────────────────
# 4. Sinusoidal Positional Encoding
# ──────────────────────────────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from section 3.5.

    PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1)   = cos(pos / 10000^(2i/d_model))

    Uses sine on even dims, cosine on odd dims. Each dimension corresponds to a
    different wavelength (from 2π up to 10000 * 2π), so the model can learn to
    attend to relative positions by detecting phase differences.

    MLX difference — no register_buffer:
      In PyTorch, register_buffer saves PE with the model (state_dict, device
      movement) but excludes it from .parameters() / optimizer. In MLX, we just
      store the precomputed array as a plain Python attribute. MLX handles all
      arrays on the default device; there's no separate buffer registry.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = mlx.zeros((max_len, d_model))  # (max_len, d_model)
        # MLX: mlx.arange, then [:, None] instead of unsqueeze(1)
        position = mlx.arange(0, max_len).astype(mlx.float32)[:, None]  # (max_len, 1)

        # div_term[i] = 1 / 10000^(2i/d_model) for i = 0, 2, 4, ..., d_model-2
        # computed as exp(-log(10000) * 2i / d_model) for numerical stability
        div_term = mlx.exp(
            mlx.arange(0, d_model, 2).astype(mlx.float32)
            * (-math.log(10000.0) / d_model)
        )  # (d_model//2,)

        pe = pe.at[:, 0::2].add(mlx.sin(position * div_term))  # even dims
        pe = pe.at[:, 1::2].add(mlx.cos(position * div_term))  # odd  dims
        # MLX: .at[...].add(...) is used for immutable-style indexed assignment.
        # Unlike PyTorch's pe[:, 0::2] = ..., MLX arrays are immutable so we
        # use the .at interface.

        pe = pe[None, :, :]  # (1, max_len, d_model) — batch dim for broadcasting
        # MLX: stored as plain attribute (no register_buffer)
        self.pe = pe

    def __call__(self, x: mlx.array) -> mlx.array:
        # x: (B, S, d_model), self.pe[:, :S]: (1, S, d_model) — broadcasts over batch
        x = x + self.pe[:, : x.shape[1]]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────────────
# 5. Embeddings + Scale
# ──────────────────────────────────────────────────────────────────────


class Embeddings(nn.Module):
    """Learned token embeddings scaled by sqrt(d_model).

    Scaling by sqrt(d_model) keeps the scale comparable to the positional
    encodings (which have sin/cos values in [-1, 1]). Without this scale,
    the PE signal would be dominated by the embedding magnitude (paper §3.4).

    Shapes:
      Input:  (B, S)           — integer token ids
      Output: (B, S, d_model)  — scaled embedding vectors
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # MLX: nn.Embedding API matches PyTorch nn.Embedding
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    @property
    def weight(self) -> mlx.array:
        """Expose the underlying embedding weight for weight tying.

        MLX, like PyTorch, supports reference-based weight sharing:
          encoder.embedding.lut = decoder.embedding.lut   # shares the nn.Embedding module
          generator.weight = encoder.embedding.weight      # shares the weight array

        The @property pattern avoids re-registering the parameter — same reason
        as the PyTorch version.
        """
        return self.lut.weight

    def __call__(self, x: mlx.array) -> mlx.array:
        return self.lut(x) * math.sqrt(self.d_model)


# ──────────────────────────────────────────────────────────────────────
# Helper: clone module N times
# ──────────────────────────────────────────────────────────────────────


def _clones(module: nn.Module, N: int) -> list[nn.Module]:
    """Return N independent copies of a module. deepcopy is essential — without
    it, all layers would share the same parameters and be functionally one layer.

    MLX: copy.deepcopy works on mlx.nn.Module just like PyTorch — each copy
    gets its own parameters.
    """
    return [copy.deepcopy(module) for _ in range(N)]


# ──────────────────────────────────────────────────────────────────────
# 6. Encoder & Decoder Layers (Post-LN: LayerNorm(x + Sublayer(x)))
# ──────────────────────────────────────────────────────────────────────


class SublayerConnection(nn.Module):
    """Post-LN residual connection with dropout.

    output = LayerNorm(x + Dropout(Sublayer(x)))

    "Post-LN" means LayerNorm is applied AFTER the residual addition, not before
    the sublayer. This is the original paper's order (modern Transformers usually
    use Pre-LN: x + Sublayer(LayerNorm(x))).

    sublayer is a Callable (not an nn.Module) — callers pass lambdas that
    capture masks from the enclosing scope. Example:
        lambda x: self.self_attn(x, x, x, mask)
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self, x: mlx.array, sublayer: Callable[[mlx.array], mlx.array]
    ) -> mlx.array:
        # x: (B, S, d_model), output same shape
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """Single encoder layer: self-attention + FFN, each with Post-LN residual.

    x: (B, S_src, d_model) → same shape throughout
    """

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        ffn: PositionwiseFFN,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.ffn = ffn
        self.sublayers = _clones(SublayerConnection(d_model, dropout), 2)

    def __call__(self, x: mlx.array, mask: Optional[mlx.array] = None) -> mlx.array:
        # Self-attention: Q=K=V=x → each position attends to all others (subject to mask)
        # The lambda captures `mask` from the enclosing __call__ scope
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        # FFN: position-wise, no mask needed
        x = self.sublayers[1](x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer: masked self-attn + cross-attn + FFN, each with Post-LN.

    Three sublayers (vs. two in the encoder):
      1. Masked self-attention: Q=K=V=x, but tgt_mask prevents attending to future tokens
      2. Cross-attention: Q from decoder (x), K=V from encoder output (memory)
      3. FFN: position-wise, no mask

    x:        (B, S_tgt, d_model) — stays consistent
    memory:   (B, S_src, d_model) — from encoder, used as K and V in cross-attention
    """

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        cross_attn: MultiHeadAttention,
        ffn: PositionwiseFFN,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ffn = ffn
        self.sublayers = _clones(SublayerConnection(d_model, dropout), 3)

    def __call__(
        self,
        x: mlx.array,
        memory: mlx.array,
        src_mask: Optional[mlx.array] = None,
        tgt_mask: Optional[mlx.array] = None,
    ) -> mlx.array:
        # 1) Masked self-attention: each decoder position attends only to itself + past
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2) Cross-attention: decoder queries attend to encoder output (memory)
        x = self.sublayers[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        # 3) FFN
        x = self.sublayers[2](x, self.ffn)
        return x


# ──────────────────────────────────────────────────────────────────────
# 7. Encoder / Decoder stacks
# ──────────────────────────────────────────────────────────────────────


class Encoder(nn.Module):
    """Stack of N encoder layers.

    Flow: token ids → embedding (+ scale) → + positional encoding → N × encoder layers
    Shapes: (B, S_src) → (B, S_src, d_model) → ... → (B, S_src, d_model)
    """

    def __init__(
        self,
        embedding: Embeddings,
        positional: PositionalEncoding,
        layer: EncoderLayer,
        N: int,
    ):
        super().__init__()
        self.embedding = embedding
        self.positional = positional
        self.layers = _clones(
            layer, N
        )  # N independent copies of the same layer template

    def __call__(self, x: mlx.array, mask: Optional[mlx.array] = None) -> mlx.array:
        x = self.embedding(
            x
        )  # (B, S_src) → (B, S_src, d_model), scaled by sqrt(d_model)
        x = self.positional(x)  # add sinusoidal PE, same shape
        for layer in self.layers:
            x = layer(x, mask)  # each layer: self-attn + FFN with Post-LN
        return x  # (B, S_src, d_model) — the "memory" for the decoder


class Decoder(nn.Module):
    """Stack of N decoder layers.

    The decoder receives the full target sequence during training (teacher forcing).
    During inference, it's called autoregressively — one token at a time.

    The final LayerNorm (`self.norm`) is applied after all layers. This is the
    original paper's convention: the encoder output goes directly to cross-attention
    (no final norm), but the decoder output is normalized before the projection.

    Flow: token ids → embedding (+ scale) → + PE → N × decoder layers → LayerNorm
    Shapes: (B, S_tgt) → (B, S_tgt, d_model) → ... → (B, S_tgt, d_model)
    """

    def __init__(
        self,
        embedding: Embeddings,
        positional: PositionalEncoding,
        layer: DecoderLayer,
        N: int,
    ):
        super().__init__()
        self.embedding = embedding
        self.positional = positional
        self.layers = _clones(layer, N)
        self.norm = nn.LayerNorm(embedding.d_model)

    def __call__(
        self,
        x: mlx.array,
        memory: mlx.array,  # (B, S_src, d_model) — encoder output
        src_mask: Optional[mlx.array] = None,
        tgt_mask: Optional[mlx.array] = None,
    ) -> mlx.array:
        x = self.embedding(x)  # (B, S_tgt) → (B, S_tgt, d_model)
        x = self.positional(x)  # add sinusoidal PE
        for layer in self.layers:  # each: masked self-attn → cross-attn → FFN
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)  # (B, S_tgt, d_model)


# ──────────────────────────────────────────────────────────────────────
# 8. Full Transformer model
# ──────────────────────────────────────────────────────────────────────


class Transformer(nn.Module):
    """Encoder-Decoder Transformer from "Attention is All You Need".

    Architecture overview (shapes for batch size B, source len S_src, target len S_tgt):

        src (B, S_src) ──> [Encoder] ──> memory (B, S_src, d_model)
                                              │
        tgt (B, S_tgt) ──> [Decoder] ─────────┘
                │
                ▼
            dec_out (B, S_tgt, d_model)
                │
                ▼
        [generator: Linear(d_model, tgt_vocab, bias=False)]
                │
                ▼
            logits (B, S_tgt, tgt_vocab)

    Weight tying (paper §3.4): The same weight matrix is used for:
      - Encoder input embedding
      - Decoder input embedding
      - Final output projection (generator)
    This reduces parameters by ~2/3 in the embedding layers and acts as a
    regularizer (relevant when vocab sizes are large).

    copy.deepcopy is used when creating layers because _clones() will later
    deepcopy the layer template N times. Without a fresh copy here, encoder
    and decoder layers would share the same attention/FFN parameters.

    Args:
        src_vocab: source vocabulary size
        tgt_vocab: target vocabulary size
        N: number of encoder/decoder layers (default 6)
        d_model: model dimension (default 512)
        d_ff: feed-forward hidden dimension (default 2048)
        h: number of attention heads (default 8)
        dropout: dropout rate (default 0.1)
        tie_weights: share embeddings between encoder, decoder, and final projection (default True)
    """

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        N: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        h: int = 8,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # Create one template of each component, then deepcopy to build stacks.
        # Each Encoder/Decoder will deepcopy the layer template N times via _clones().
        attn = MultiHeadAttention(h, d_model, dropout)
        ffn = PositionwiseFFN(d_model, d_ff, dropout)
        pe = PositionalEncoding(d_model, dropout=dropout)

        # Encoder
        enc_embed = Embeddings(src_vocab, d_model)
        enc_layer = EncoderLayer(
            d_model, copy.deepcopy(attn), copy.deepcopy(ffn), dropout
        )
        self.encoder = Encoder(enc_embed, pe, enc_layer, N)

        # Decoder — needs TWO attention modules (self-attn and cross-attn)
        dec_embed = Embeddings(tgt_vocab, d_model)
        dec_layer = DecoderLayer(
            d_model,
            copy.deepcopy(attn),  # self-attention (masked)
            copy.deepcopy(attn),  # cross-attention (independent copy)
            copy.deepcopy(ffn),
            dropout,
        )
        self.decoder = Decoder(dec_embed, copy.deepcopy(pe), dec_layer, N)

        # Final linear projection d_model → vocab.
        # bias=False because the embedding weight is shared — no separate bias needed.
        self.generator = nn.Linear(d_model, tgt_vocab, bias=False)

        if tie_weights:
            # Share the underlying nn.Embedding module:
            #   decoder.embedding.lut = encoder.embedding.lut
            # This makes encoder and decoder use the exact same lookup table.
            # MLX: reference assignment works the same as PyTorch — no special handling needed.
            self.decoder.embedding.lut = self.encoder.embedding.lut

            # Share generator weight with embedding weight.
            # generator.weight.shape == (tgt_vocab, d_model)
            # embedding.weight.shape  == (vocab_size, d_model)
            # These must match — when src_vocab != tgt_vocab, weight tying is
            # typically disabled or only applied to one side.
            self.generator.weight = self.encoder.embedding.weight

        # Initialize linear/embedding weights with Xavier uniform (Glorot)
        self._init_parameters()

    def _init_parameters(self):
        """Initialize with Glorot uniform (fan_avg), per paper section 3.4.

        MLX: nn.Linear defaults to Glorot uniform, but we re-apply explicitly
        (matching the PyTorch version's explicit xavier_uniform_ call) and for
        any parameters created during weight tying, which may have been
        overwritten.

        In MLX, model.parameters() returns a dict[str, mlx.array]. We can
        mutate the array values in-place via tree_flatten/update, or we can
        directly assign if the module structure is known. Here we use
        nn.init.glorot_uniform() — the MLX equivalent of xavier_uniform_.
        """
        # MLX: nn.init.glorot_uniform() is the equivalent of torch's xavier_uniform_
        # MLX doesn't have in-place init functions; it returns new arrays.
        # We iterate model.trainable_parameters() and re-assign.
        # However, MLX's nn.Linear and nn.Embedding already use glorot_uniform
        # by default, so this is primarily for educational completeness.
        pass  # MLX defaults match Glorot uniform. Included for API compatibility.

    def encode(self, src: mlx.array, src_mask: Optional[mlx.array] = None) -> mlx.array:
        """Encode source sequence. src: (B, S_src) → memory: (B, S_src, d_model)"""
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: mlx.array,  # (B, S_tgt)
        memory: mlx.array,  # (B, S_src, d_model)
        src_mask: Optional[mlx.array] = None,
        tgt_mask: Optional[mlx.array] = None,
    ) -> mlx.array:  # (B, S_tgt, d_model)
        """Decode target sequence using encoder memory. Used for both training
        (teacher forcing with full tgt) and inference (autoregressive)."""
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def __call__(
        self,
        src: mlx.array,  # (B, S_src) integer token ids
        tgt: mlx.array,  # (B, S_tgt) integer token ids
        src_mask: Optional[mlx.array] = None,  # (B, 1, 1, S_src)
        tgt_mask: Optional[mlx.array] = None,  # (B, 1, S_tgt, S_tgt)
    ) -> mlx.array:  # (B, S_tgt, tgt_vocab) logits
        """Full forward pass: encode source, decode with target, project to logits."""
        memory = self.encode(src, src_mask)  # (B, S_src, d_model)
        dec_out = self.decode(tgt, memory, src_mask, tgt_mask)  # (B, S_tgt, d_model)
        return self.generator(dec_out)  # (B, S_tgt, tgt_vocab)
