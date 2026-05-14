"""Original Transformer from "Attention is All You Need" (Vaswani et al., 2017).

Strictly follows the original paper — no modern modifications (Post-LN, ReLU, sinusoidal PE, etc.)
"""

import copy
import math
from collections.abc import Callable
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Scale by sqrt(d_k) to keep softmax variance ~1 (prevents tiny gradients)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Mask: True = can attend, False = masked out → fill ~True positions with -inf
        # This ensures masked positions get 0 weight after softmax
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


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

        self.linears = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(4)]  # W^Q, W^K, W^V, W^O
        )
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,  # (B, S_q, d_model)
        key: torch.Tensor,  # (B, S_k, d_model)
        value: torch.Tensor,  # (B, S_v, d_model) — usually S_k == S_v
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # (B, S_q, d_model)
        batch_size = query.size(0)

        # Masks from callers (e.g. make_std_mask) are 3D (B, S_q, S_k).
        # Add head dim so they broadcast to (B, 1, S_q, S_k) for all heads.
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # 1) Linear projection + split into heads in one step
        #    (B, S, d_model) → view → (B, S, h, d_k) → transpose → (B, h, S, d_k)
        #    Now batch dim = B, and each "batch" is really a head — so PyTorch
        #    computes h independent attention operations in parallel.
        query, key, value = [
            lin(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply scaled dot-product attention on all heads
        #    x: (B, h, S_q, d_k) — each head produced a d_k output per position
        x, _ = self.attention(query, key, value, mask=mask)

        # 3) Concatenate heads back: (B, h, S, d_k) → (B, S, h*d_k) = (B, S, d_model)
        #    .contiguous() is required because transpose creates a non-contiguous view
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, d_model) → expand → (B, S, d_ff) → ReLU+Dropout → project → (B, S, d_model)
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


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

    Why register_buffer?  PE values are constants, not learnable parameters.
    register_buffer saves them with the model (so they move to GPU / are included
    in state_dict), but excludes them from .parameters() so the optimizer ignores them.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

        # div_term[i] = 1 / 10000^(2i/d_model) for i = 0, 2, 4, ..., d_model-2
        # computed as exp(-log(10000) * 2i / d_model) for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model//2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims: (max_len, d_model//2)
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  dims: (max_len, d_model//2)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) — batch dim for broadcasting
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.pe, torch.Tensor)
        # x: (B, S, d_model), self.pe[:, :S]: (1, S, d_model) — broadcasts over batch
        x = x + self.pe[:, : x.size(1)]
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
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    @property
    def weight(self) -> torch.Tensor:
        """Expose the underlying embedding weight for weight tying.

        Using @property avoids a register_parameter collision: if we set
        self.weight = self.lut.weight directly, PyTorch's __setattr__ would
        try to register it as a new parameter, clashing with the existing one
        inside self.lut. @property returns a reference without re-registering.
        """
        return self.lut.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lut(x) * math.sqrt(self.d_model)


# ──────────────────────────────────────────────────────────────────────
# Helper: clone module N times
# ──────────────────────────────────────────────────────────────────────


def _clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Return N independent copies of a module. deepcopy is essential — without
    it, all layers would share the same parameters and be functionally one layer."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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

    def forward(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention: Q=K=V=x → each position attends to all others (subject to mask)
        # The lambda captures `mask` from the enclosing forward scope
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

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,  # (B, S_src, d_model) — encoder output
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        Only initializes matrices (weights with dim > 1). Biases and other
        1D parameters are left at their default (usually zeros).
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode source sequence. src: (B, S_src) → memory: (B, S_src, d_model)"""
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,  # (B, S_tgt)
        memory: torch.Tensor,  # (B, S_src, d_model)
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # (B, S_tgt, d_model)
        """Decode target sequence using encoder memory. Used for both training
        (teacher forcing with full tgt) and inference (autoregressive)."""
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,  # (B, S_src) integer token ids
        tgt: torch.Tensor,  # (B, S_tgt) integer token ids
        src_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, S_src)
        tgt_mask: Optional[torch.Tensor] = None,  # (B, 1, S_tgt, S_tgt)
    ) -> torch.Tensor:  # (B, S_tgt, tgt_vocab) logits
        """Full forward pass: encode source, decode with target, project to logits."""
        memory = self.encode(src, src_mask)  # (B, S_src, d_model)
        dec_out = self.decode(tgt, memory, src_mask, tgt_mask)  # (B, S_tgt, d_model)
        return self.generator(dec_out)  # (B, S_tgt, tgt_vocab)
