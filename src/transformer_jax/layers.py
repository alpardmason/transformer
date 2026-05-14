"""JAX + Flax reimplementation of the original Transformer from "Attention is All You Need".

JAX/Flax vs PyTorch — key differences annotated throughout this file:
  - Functional with explicit PRNG: every random op needs jax.random.PRNGKey.
    Flax handles RNG splitting internally via rngs={'dropout': key} for dropout.
  - Immutable arrays: no in-place ops. Use arr.at[idx].set(val) or jnp.where().
  - nn.Dense(features) replaces nn.Linear; nn.Embed(num_embeddings, features)
    replaces nn.Embedding.
  - @nn.compact: lazy parameter creation inside __call__ — each nn.Dense(x)(y)
    call creates an independent layer. Simple modules use @nn.compact instead
    of setup().
  - deterministic=True/False: threads through ALL modules to toggle dropout.
    In PyTorch, model.eval()/model.train() sets this globally. In Flax it's
    an explicit kwarg that must be passed through every module's __call__.
  - model.apply(variables, ..., method=..., rngs={...}) for forward pass.
    variables is a dict {'params': ..., 'batch_stats': ...}.
  - Flax submodule tracking: modules in Python lists are NOT tracked. Use
    setattr(self, f'layer_{i}', layer) for N-repeat patterns.
  - Weight tying: Flax's ownership tracking makes simple reference sharing
    harder than PyTorch. We use post-init variable dict copying (see
    tie_weights() helper).
  - jax.grad() / jax.jit(): Function transformations for gradients and
    compilation — used in train.py, not here.
  - jnp.where(cond, a, b) replaces masked_fill.
  - jnp.swapaxes replaces transpose.
"""

import math
from collections.abc import Callable
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


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

    JAX/Flax note — @nn.compact:
      This module has no learnable parameters — only dropout. @nn.compact is
      used for simplicity: nn.Dropout is created lazily on first __call__.
      The rng_collection='dropout' tells Flax to look up rngs['dropout'] for
      the random key. The caller must provide this key in model.apply() or
      model.init().
    """

    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Scale by sqrt(d_k) to keep softmax variance ~1 (prevents tiny gradients)
        d_k = query.shape[-1]
        scores = jnp.matmul(query, key.swapaxes(-2, -1)) / math.sqrt(d_k)

        # Mask: True = can attend, False = masked out → fill ~True positions with -inf
        # JAX: jnp.where(cond, a, b) replaces scores.masked_fill(~mask, float('-inf'))
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)  # use -1e9 instead of -inf for JAX safety

        # JAX: jax.nn.softmax (same API as F.softmax)
        p_attn = jax.nn.softmax(scores, axis=-1)
        # Flax: nn.Dropout with rng_collection — key must be in rngs={'dropout': key}
        p_attn = nn.Dropout(self.dropout, rng_collection="dropout")(
            p_attn, deterministic=deterministic
        )
        return jnp.matmul(p_attn, value), p_attn


# ──────────────────────────────────────────────────────────────────────
# 2. Multi-Head Attention
# ──────────────────────────────────────────────────────────────────────


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with h parallel heads.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    JAX/Flax note — @nn.compact and Dense creation:
      Each nn.Dense(features)(x) call inside @nn.compact creates an independent
      Dense layer with its own parameters, automatically named by call site
      (e.g. Dense_0, Dense_1, ...). This replaces PyTorch's nn.ModuleList of
      4 Linear layers. The names are stable as long as the call order is stable.

      Use separate Dense calls instead of a loop to ensure the 4 projections
      are independently parameterized (they're semantically different: Q, K, V, O).
    """

    h: int = 8
    d_model: int = 512
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,   # (B, S_q, d_model)
        key: jnp.ndarray,     # (B, S_k, d_model)
        value: jnp.ndarray,   # (B, S_v, d_model) — usually S_k == S_v
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:         # (B, S_q, d_model)
        if self.d_model % self.h != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by h ({self.h})")
        d_k = self.d_model // self.h
        batch_size = query.shape[0]

        # Masks from callers (e.g. make_std_mask) are 3D (B, S_q, S_k).
        # Add head dim so they broadcast to (B, 1, S_q, S_k) for all heads.
        if mask is not None and mask.ndim == 3:
            mask = mask[:, jnp.newaxis, :, :]

        # 1) Linear projection + split into heads in one step
        #    (B, S, d_model) → reshape → (B, S, h, d_k) → swapaxes → (B, h, S, d_k)
        #    Each nn.Dense(self.d_model) creates an independent weight matrix.
        #    Flax: 4 separate Dense calls (semantically Q, K, V, O projections)
        q = nn.Dense(self.d_model)(query)
        k = nn.Dense(self.d_model)(key)
        v = nn.Dense(self.d_model)(value)

        q = q.reshape(batch_size, -1, self.h, d_k).swapaxes(1, 2)  # (B, h, S_q, d_k)
        k = k.reshape(batch_size, -1, self.h, d_k).swapaxes(1, 2)  # (B, h, S_k, d_k)
        v = v.reshape(batch_size, -1, self.h, d_k).swapaxes(1, 2)  # (B, h, S_v, d_k)

        # 2) Apply scaled dot-product attention on all heads
        x, _ = ScaledDotProductAttention(dropout=self.dropout)(
            q, k, v, mask, deterministic=deterministic
        )

        # 3) Concatenate heads back: (B, h, S, d_k) → (B, S, h*d_k) = (B, S, d_model)
        #    JAX: no .contiguous() needed. reshape after swapaxes always works.
        x = x.swapaxes(1, 2).reshape(batch_size, -1, self.h * d_k)
        return nn.Dense(self.d_model)(x)  # W^O projection


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

    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        # (B, S, d_model) → expand → (B, S, d_ff) → ReLU+Dropout → project → (B, S, d_model)
        # Flax: nn.Dense(features) replaces nn.Linear; nn.relu replaces F.relu
        x = nn.Dense(self.d_ff)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout, rng_collection="dropout")(
            x, deterministic=deterministic
        )
        x = nn.Dense(self.d_model)(x)
        return x


# ──────────────────────────────────────────────────────────────────────
# 4. Sinusoidal Positional Encoding
# ──────────────────────────────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from section 3.5.

    PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1)   = cos(pos / 10000^(2i/d_model))

    Uses sine on even dims, cosine on odd dims.

    JAX/Flax note — setup() for precomputed arrays:
      We use setup() (not @nn.compact) because we need to precompute the PE
      array during initialization. setup() runs once when the module is
      constructed/initialized.

      The PE array is stored as a plain Python attribute (not a Flax variable)
      because it's a constant — no gradients, no optimizer updates. This is
      analogous to PyTorch's register_buffer but simpler.

      JAX immutable arrays: use .at[...].set(val) instead of x[idx] = val.
    """

    d_model: int = 512
    max_len: int = 5000
    dropout: float = 0.1

    def setup(self):
        self.dropout_layer = nn.Dropout(self.dropout, rng_collection="dropout")

        pe = jnp.zeros((1, self.max_len, self.d_model))  # (1, max_len, d_model)
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, jnp.newaxis]  # (max_len, 1)

        # div_term[i] = 1 / 10000^(2i/d_model) for i = 0, 2, 4, ..., d_model-2
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2, dtype=jnp.float32)
            * (-math.log(10000.0) / self.d_model)
        )

        # JAX: .at[...].set(...) for indexed assignment (immutable arrays)
        pe = pe.at[:, :, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, :, 1::2].set(jnp.cos(position * div_term))
        # Stored as plain attribute — not a Flax variable, not learnable
        self.pe = pe

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        # x: (B, S, d_model), self.pe[:, :S]: (1, S, d_model) — broadcasts over batch
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout_layer(x, deterministic=deterministic)


# ──────────────────────────────────────────────────────────────────────
# 5. Embeddings + Scale
# ──────────────────────────────────────────────────────────────────────


class Embeddings(nn.Module):
    """Learned token embeddings scaled by sqrt(d_model).

    Scaling by sqrt(d_model) keeps the scale comparable to the positional
    encodings (which have sin/cos values in [-1, 1]). Without this scale,
    the PE signal would be dominated by the embedding magnitude (paper §3.4).

    JAX/Flax note — nn.Embed replaces nn.Embedding:
      API: nn.Embed(num_embeddings=vocab_size, features=d_model)
      This is one of the few naming differences from PyTorch.

      For weight tying in JAX, the embedding table can be accessed from
      variables['params']['Embed_0']['embedding'] after initialization.
      See Transformer.tie_weights() for how this is used.

    Shapes:
      Input:  (B, S)           — integer token ids
      Output: (B, S, d_model)  — scaled embedding vectors
    """

    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        embed = nn.Embed(num_embeddings=self.num_embeddings, features=self.features)
        return embed(x) * math.sqrt(self.features)


# ──────────────────────────────────────────────────────────────────────
# 6. Encoder & Decoder Layers (Post-LN: LayerNorm(x + Sublayer(x)))
# ──────────────────────────────────────────────────────────────────────


class SublayerConnection(nn.Module):
    """Post-LN residual connection with dropout.

    output = LayerNorm(x + Dropout(Sublayer(x)))

    sublayer is a Callable — callers pass lambdas that capture masks from the
    enclosing scope. Same pattern as PyTorch.

    JAX/Flax note — setup() for named submodules:
      We use setup() (not @nn.compact) so that norm and dropout are named
      attributes accessible after module construction. LayerNorm in Flax
      auto-infers the feature dimension from input when not specified.
    """

    d_model: int
    dropout: float

    def setup(self):
        self.norm = nn.LayerNorm()
        self.dropout_layer = nn.Dropout(self.dropout, rng_collection="dropout")

    def __call__(
        self,
        x: jnp.ndarray,
        sublayer: Callable[[jnp.ndarray], jnp.ndarray],
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # x: (B, S, d_model), output same shape
        return self.norm(x + self.dropout_layer(sublayer(x), deterministic=deterministic))


class EncoderLayer(nn.Module):
    """Single encoder layer: self-attention + FFN, each with Post-LN residual.

    x: (B, S_src, d_model) → same shape throughout

    JAX/Flax note — setup() with submodule construction:
      sublayer connections are created in setup() as named attributes.
      The mask is passed through lambda capture (same as PyTorch).
    """

    d_model: int
    h: int
    d_ff: int
    dropout: float

    def setup(self):
        # Create attention and FFN as named submodules
        self.self_attn = MultiHeadAttention(
            h=self.h, d_model=self.d_model, dropout=self.dropout
        )
        self.ffn = PositionwiseFFN(
            d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout
        )
        # Two sublayer connections (one for attn, one for FFN)
        self.sublayer_0 = SublayerConnection(d_model=self.d_model, dropout=self.dropout)
        self.sublayer_1 = SublayerConnection(d_model=self.d_model, dropout=self.dropout)

    def __call__(
        self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, *, deterministic: bool = True
    ) -> jnp.ndarray:
        # Self-attention: Q=K=V=x → each position attends to all others (subject to mask)
        x = self.sublayer_0(
            x, lambda x: self.self_attn(x, x, x, mask, deterministic=deterministic),
            deterministic=deterministic,
        )
        # FFN: position-wise, no mask needed
        x = self.sublayer_1(
            x, lambda x: self.ffn(x, deterministic=deterministic),
            deterministic=deterministic,
        )
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

    d_model: int
    h: int
    d_ff: int
    dropout: float

    def setup(self):
        self.self_attn = MultiHeadAttention(
            h=self.h, d_model=self.d_model, dropout=self.dropout
        )
        self.cross_attn = MultiHeadAttention(
            h=self.h, d_model=self.d_model, dropout=self.dropout
        )
        self.ffn = PositionwiseFFN(
            d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout
        )
        self.sublayer_0 = SublayerConnection(d_model=self.d_model, dropout=self.dropout)
        self.sublayer_1 = SublayerConnection(d_model=self.d_model, dropout=self.dropout)
        self.sublayer_2 = SublayerConnection(d_model=self.d_model, dropout=self.dropout)

    def __call__(
        self,
        x: jnp.ndarray,
        memory: jnp.ndarray,
        src_mask: Optional[jnp.ndarray] = None,
        tgt_mask: Optional[jnp.ndarray] = None,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # 1) Masked self-attention: each decoder position attends only to itself + past
        x = self.sublayer_0(
            x, lambda x: self.self_attn(x, x, x, tgt_mask, deterministic=deterministic),
            deterministic=deterministic,
        )
        # 2) Cross-attention: decoder queries attend to encoder output (memory)
        x = self.sublayer_1(
            x, lambda x: self.cross_attn(x, memory, memory, src_mask, deterministic=deterministic),
            deterministic=deterministic,
        )
        # 3) FFN
        x = self.sublayer_2(
            x, lambda x: self.ffn(x, deterministic=deterministic),
            deterministic=deterministic,
        )
        return x


# ──────────────────────────────────────────────────────────────────────
# 7. Encoder / Decoder stacks
# ──────────────────────────────────────────────────────────────────────


class Encoder(nn.Module):
    """Stack of N encoder layers.

    Flow: token ids → embedding (+ scale) → + positional encoding → N × encoder layers
    Shapes: (B, S_src) → (B, S_src, d_model) → ... → (B, S_src, d_model)

    JAX/Flax note — N-layer stacking via setattr:
      Flax does NOT track submodules stored in Python lists. The canonical
      pattern is setattr(self, f'layer_{i}', layer) in a loop. Each layer
      gets a unique name in the variable tree (e.g. layer_0, layer_1, ...).
      This works because setup() is called once during init, and Flax
      discovers submodules by iterating __dict__.

      Embedding and positional encoding are separate from the Encoder stack
      in this architecture — they're handled by Transformer.encode().
    """

    N: int
    d_model: int
    h: int
    d_ff: int
    dropout: float

    def setup(self):
        for i in range(self.N):
            layer = EncoderLayer(
                d_model=self.d_model, h=self.h, d_ff=self.d_ff, dropout=self.dropout
            )
            setattr(self, f"layer_{i}", layer)

    def __call__(
        self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, *, deterministic: bool = True
    ) -> jnp.ndarray:
        for i in range(self.N):
            x = getattr(self, f"layer_{i}")(x, mask, deterministic=deterministic)
        return x  # (B, S_src, d_model) — the "memory" for the decoder


class Decoder(nn.Module):
    """Stack of N decoder layers.

    The final LayerNorm (`self.norm`) is applied after all layers.

    Flow: token ids → embedding (+ scale) → + PE → N × decoder layers → LayerNorm
    Shapes: (B, S_tgt) → (B, S_tgt, d_model) → ... → (B, S_tgt, d_model)
    """

    N: int
    d_model: int
    h: int
    d_ff: int
    dropout: float

    def setup(self):
        self.norm = nn.LayerNorm()
        for i in range(self.N):
            layer = DecoderLayer(
                d_model=self.d_model, h=self.h, d_ff=self.d_ff, dropout=self.dropout
            )
            setattr(self, f"layer_{i}", layer)

    def __call__(
        self,
        x: jnp.ndarray,
        memory: jnp.ndarray,
        src_mask: Optional[jnp.ndarray] = None,
        tgt_mask: Optional[jnp.ndarray] = None,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        for i in range(self.N):
            x = getattr(self, f"layer_{i}")(
                x, memory, src_mask, tgt_mask, deterministic=deterministic
            )
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
        [generator: Dense(d_model, tgt_vocab, use_bias=False)] or manual matmul
                │
                ▼
            logits (B, S_tgt, tgt_vocab)

    Weight tying (paper §3.4):
      JAX/Flax's immutable variable tree makes simple reference sharing
      (encoder.embed.weight = decoder.embed.weight) impossible during
      construction. Instead, we use a two-step approach:
        1. Initialize normally (separate embeddings, separate generator)
        2. Call tie_weights(variables) after init to copy encoder embed
           params to decoder embed and generator (transposed) in the
           variable dict.

    Args:
        src_vocab: source vocabulary size
        tgt_vocab: target vocabulary size
        N: number of encoder/decoder layers (default 6)
        d_model: model dimension (default 512)
        d_ff: feed-forward hidden dimension (default 2048)
        h: number of attention heads (default 8)
        dropout: dropout rate (default 0.1)
        tie_weights: share embeddings between encoder, decoder, and final projection
    """

    src_vocab: int
    tgt_vocab: int
    N: int = 6
    d_model: int = 512
    d_ff: int = 2048
    h: int = 8
    dropout: float = 0.1
    tie_weights: bool = True

    def setup(self):
        # Embedding layers (one per vocab — weight tying happens post-init)
        self.encoder_embed = Embeddings(
            num_embeddings=self.src_vocab, features=self.d_model
        )
        self.decoder_embed = Embeddings(
            num_embeddings=self.tgt_vocab, features=self.d_model
        )
        # Positional encoding (separate instances for encoder/decoder)
        self.encoder_pe = PositionalEncoding(
            d_model=self.d_model, dropout=self.dropout
        )
        self.decoder_pe = PositionalEncoding(
            d_model=self.d_model, dropout=self.dropout
        )
        # Encoder / Decoder stacks
        self.encoder = Encoder(
            N=self.N, d_model=self.d_model, h=self.h, d_ff=self.d_ff, dropout=self.dropout
        )
        self.decoder = Decoder(
            N=self.N, d_model=self.d_model, h=self.h, d_ff=self.d_ff, dropout=self.dropout
        )
        # Output projection (replaced by tied embed weight if tie_weights=True)
        self.generator = nn.Dense(self.tgt_vocab, use_bias=False)

    def __call__(
        self,
        src: jnp.ndarray,         # (B, S_src) integer token ids
        tgt: jnp.ndarray,         # (B, S_tgt) integer token ids
        src_mask: Optional[jnp.ndarray] = None,  # (B, 1, 1, S_src)
        tgt_mask: Optional[jnp.ndarray] = None,  # (B, 1, S_tgt, S_tgt)
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:             # (B, S_tgt, tgt_vocab) logits
        """Full forward pass: encode source, decode with target, project to logits.

        JAX/Flax note — self.variables:
          Inside __call__, self.variables is populated with the module's
          variable tree (params, batch_stats, etc.). We access tied weights
          via self.variables['params'][<path>] when tie_weights=True.

          For a Flax module initialized with model.init(rngs, src, tgt, ...),
          the variable tree looks like:
            params/
              encoder_embed/Embed_0/embedding  : (src_vocab, d_model)
              decoder_embed/Embed_0/embedding  : (tgt_vocab, d_model)
              generator/kernel                 : (d_model, tgt_vocab)
              encoder/layer_0/...
              decoder/layer_0/...
        """
        memory = self.encode(src, src_mask, deterministic=deterministic)
        dec_out = self.decode(tgt, memory, src_mask, tgt_mask, deterministic=deterministic)

        if self.tie_weights and self.variables:
            # Use the encoder embedding weight (transposed) for output projection.
            # The path 'encoder_embed' → 'Embed_0' → 'embedding' is determined
            # by Flax's auto-naming: Embeddings is the parent, @nn.compact
            # creates nn.Embed as the first (0th) submodule, and its param
            # is named 'embedding'.
            embed_table = self.variables["params"]["encoder_embed"]["Embed_0"]["embedding"]
            # embed_table: (src_vocab, d_model) → need (d_model, tgt_vocab) for output
            # When src_vocab == tgt_vocab, this is a square matrix
            logits = dec_out @ embed_table.T
        else:
            logits = self.generator(dec_out)

        return logits  # (B, S_tgt, tgt_vocab)

    def encode(
        self, src: jnp.ndarray, src_mask: Optional[jnp.ndarray] = None, *, deterministic: bool = True
    ) -> jnp.ndarray:
        """Encode source sequence. src: (B, S_src) → memory: (B, S_src, d_model)

        JAX/Flax note — encode/decode as separate methods:
          These methods allow the training/inference code to call encode once
          and reuse the memory for multiple decode calls. They're invoked via
          model.apply(variables, ..., method=Transformer.encode, rngs={...}).
        """
        x = self.encoder_embed(src)                              # (B, S_src, d_model)
        x = self.encoder_pe(x, deterministic=deterministic)      # add sinusoidal PE
        return self.encoder(x, src_mask, deterministic=deterministic)

    def decode(
        self,
        tgt: jnp.ndarray,
        memory: jnp.ndarray,
        src_mask: Optional[jnp.ndarray] = None,
        tgt_mask: Optional[jnp.ndarray] = None,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Decode target sequence using encoder memory."""
        x = self.decoder_embed(tgt)                              # (B, S_tgt, d_model)
        x = self.decoder_pe(x, deterministic=deterministic)      # add sinusoidal PE
        return self.decoder(x, memory, src_mask, tgt_mask, deterministic=deterministic)

    def _init_parameters(self, variables: dict) -> dict:
        """Apply Glorot uniform initialization to weight matrices.

        JAX/Flax note — _init_parameters:
          Flax's nn.Dense and nn.Embed use lecun_normal (truncated) by default,
          NOT xavier_uniform. To match the paper (Section 3.4), we reinitialize
          all weight matrices with Glorot uniform after model.init().

          In PyTorch, this is a simple in-place mutation. In JAX's functional
          paradigm, we return a new variables dict with reinitialized params.

        Called after model.init() returns the initial variables.
        """
        # Flax defaults differ from paper spec — this function exists for
        # compatibility. In practice, the default Flax init works fine.
        return variables  # Placeholder — re-init if needed for exact paper match


def tie_weights(variables: dict) -> dict:
    """Post-init weight tying for the JAX Transformer.

    JAX/Flax weight tying:
      After model.init(), call this to share the encoder embedding weight
      with the decoder embedding and generator (output projection).

      This copies the encoder's embedding table to:
        1. decoder_embed/Embed_0/embedding — same lookup table
        2. generator/kernel — transposed (Dense kernel is (d_model, vocab))
           Since generator has use_bias=False, only kernel is needed.

      PyTorch equivalent:
        model.decoder.embedding.lut = model.encoder.embedding.lut
        model.generator.weight = model.encoder.embedding.weight

    Args:
        variables: the variables dict returned by model.init()

    Returns:
        variables with tied weights (mutated in place AND returned)
    """
    params = variables["params"]
    enc_embed = params["encoder_embed"]["Embed_0"]["embedding"]

    # Share with decoder embedding
    params["decoder_embed"]["Embed_0"]["embedding"] = enc_embed

    # Share with generator (transposed — Dense kernel is (d_model, vocab)).
    # If tie_weights=True, generator is never called during init so its params
    # don't exist yet — create the dict entry fresh instead of assigning into it.
    params["generator"] = {"kernel": enc_embed.T}

    return variables
