# Guide to the Original Transformer Codebase

This guide accompanies the educational implementation of the Transformer from
*Attention is All You Need* (Vaswani et al., 2017). It's written for students
who have just finished reading the paper and want to understand how the
equations map to code, what engineering practices make the implementation
solid, and how today's SOTA models differ from this original design.

---

## Reading Sequence

Work through the codebase in this order — each file builds on the previous one:

| Step | File | What you'll learn |
|------|------|-------------------|
| 1 | `src/transformer/layers.py` | Every architectural component: attention, FFN, positional encoding, encoder/decoder stacks, the full model |
| 2 | `src/transformer/train.py` | Noam schedule, label smoothing, mask construction, synthetic data, training loop |
| 3 | `src/transformer/generate.py` | Greedy decoding and beam search with length penalty |
| 4 | `scripts/train.py` | How the pieces are wired together into a runnable training script |
| 5 | `tests/test_smoke.py` | How to verify each component works correctly |

---

## Chapter 1: `layers.py` — The Model Architecture

Read this file first. It contains every building block from the paper, composed
bottom-up from attention through the full encoder-decoder.

### 1.1 Scaled Dot-Product Attention

`Attention(Q, K, V) = softmax(QKᵀ/√d_k) V`

**What to notice:**

```python
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
```

The scaling by `√d_k` is the only algorithmic detail that isn't obvious from
the math. Without it, the dot products grow with `d_k`, pushing the softmax
into regions of extremely small gradients. The `masked_fill(~mask, -inf)`
mechanism uses boolean masks where `True` = can attend — this is a design
choice made early in the codebase and propagated everywhere.

**Best practice — boolean mask semantics:**
The mask uses `True` = "can attend" rather than "should mask." This means callers
write `(src != pad_idx).unsqueeze(1).unsqueeze(2)` — the condition reads
naturally ("positions that are not padding"). Inside `ScaledDotProductAttention`,
the `~mask` flips it to fill masked positions with `-inf`. The convention is
documented once in `layers.py` and used consistently everywhere.

### 1.2 Multi-Head Attention

`MultiHead(Q, K, V) = Concat(head₁, ..., head_h) Wᴼ`

**What to notice — the parallel head trick:**

```python
query, key, value = [
    lin(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    for lin, x in zip(self.linears, (query, key, value))
]
```

Instead of looping over `h` heads, this reshapes `(B, S, d_model)` into
`(B, h, S, d_k)`. PyTorch's batch matmul then computes all heads in parallel.
After attention, the concatenation is done with `.transpose(1, 2).contiguous()
.view(batch_size, -1, self.h * self.d_k)`.

**Best practice — `.contiguous()` before `.view()`:**
`transpose` creates a non-contiguous view of the tensor — the underlying memory
layout doesn't match the logical layout. Calling `.view()` on a non-contiguous
tensor raises a `RuntimeError`. `.contiguous()` forces a copy into a contiguous
layout first. This is a common pattern whenever reshape follows transpose/permute.

**Mask broadcasting:**
```python
if mask is not None and mask.dim() == 3:
    mask = mask.unsqueeze(1)
```
Callers provide masks as `(B, S_q, S_k)`. This adds a head dimension `(B, 1, S_q, S_k)`
so the mask broadcasts over all heads without creating `h` copies.

### 1.3 Position-wise Feed-Forward Network

`FFN(x) = max(0, xW₁ + b₁) W₂ + b₂`

The two linear layers expand to `d_ff = 2048` then project back to `d_model = 512`.
Applied independently to every position — PyTorch's `nn.Linear` handles this
automatically since the last dimension is `d_model`.

Nothing surprising here — two linear layers with ReLU and dropout. The
simplicity is the point.

### 1.4 Sinusoidal Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**What to notice — numerical stability:**
```python
div_term = torch.exp(
    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
)
```
The paper writes `1 / 10000^(2i/d_model)`, but raising a large base to a
fractional exponent can cause precision loss. This code computes it as
`exp(-log(10000) * 2i / d_model)`, which is mathematically equivalent but
numerically stabler.

**Best practice — `register_buffer`:**
```python
self.register_buffer("pe", pe)
```
PE values are constants, not learnable parameters. `register_buffer` stores them
in the model (included in `state_dict`, moved to GPU by `.to(device)`), but
excludes them from `.parameters()` so the optimizer ignores them. This is the
correct tool for any tensor that belongs to the model state but shouldn't
receive gradients.

### 1.5 Embeddings + Scale

```python
return self.lut(x) * math.sqrt(self.d_model)
```

The paper (§3.4) multiplies embeddings by `√d_model` so their scale is
comparable to the positional encoding values (which are in `[-1, 1]`). Without
this, the PE signal would be dominated by the embedding magnitude.

**Best practice — `@property` for weight tying:**
```python
@property
def weight(self) -> torch.Tensor:
    return self.lut.weight
```
When the model ties weights (`self.generator.weight = self.encoder.embedding.weight`),
PyTorch's `__setattr__` tries to register the assigned tensor as a new parameter.
If the embedding's `weight` is a real parameter (inside `self.lut`), a direct
`self.weight = self.lut.weight` would collide. The `@property` returns a
reference to `self.lut.weight` without re-registering.

### 1.6 SublayerConnection — Post-LN

```python
return self.norm(x + self.dropout(sublayer(x)))
```

This is **Post-LN**: LayerNorm is applied **after** the residual addition.
Modern Transformers almost universally use **Pre-LN** (`x + sublayer(norm(x))`).
The distinction matters — see the modern differences section at the end of this
guide. The code makes this explicit: `LayerNorm(x + Sublayer(x))`.

The `sublayer` parameter is a `Callable`, not an `nn.Module`. Callers pass
lambdas that capture masks from the enclosing scope:
```python
x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
```
This avoids threading mask parameters through `SublayerConnection`, which would
need to know about attention-specific arguments.

### 1.7 EncoderLayer / DecoderLayer

**Encoder layer** (2 sublayers):
1. Self-attention: Q=K=V=x, attending to all source positions (subject to padding mask)
2. FFN: position-wise

**Decoder layer** (3 sublayers):
1. Masked self-attention: prevents attending to future target positions
2. Cross-attention: Q from decoder, K=V from encoder output (memory)
3. FFN: position-wise

### 1.8 Encoder / Decoder stacks

```python
self.layers = _clones(layer, N)
```

**Best practice — `copy.deepcopy` for independent copies:**
```python
def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```
Without `deepcopy`, all `N` layers would reference the same module object, sharing
parameters — functionally equivalent to a single layer. `deepcopy` creates `N`
independent copies, each with its own parameter tensors.

The `_clones` helper is used for the layer stacks, and `copy.deepcopy` is also
used when creating the layer templates themselves so that encoder and decoder
get independent attention/FFN modules.

### 1.9 Weight Tying (paper §3.4)

Three uses of the same weight matrix:
1. Encoder input embedding
2. Decoder input embedding
3. Final output projection (`generator`)

```python
self.decoder.embedding.lut = self.encoder.embedding.lut
self.generator.weight = self.encoder.embedding.weight
```

This reduces parameters by ~2/3 in the embedding layers. The `generator` linear
layer uses `bias=False` — since the embedding weight is shared, a separate bias
wouldn't be tied and would break the symmetry.

**What to notice — `_init_parameters`:**
```python
for p in self.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
```
Xavier/Glorot uniform initialization (paper §3.4). Only initializes weight
matrices (`dim > 1`) — biases and other 1D parameters are left at their
default (typically zero). This is a standard practice: zero-initialization for
biases is fine since they'll be updated by gradients from the first step, but
weight matrices need careful initialization to avoid vanishing/exploding signals.

---

## Chapter 2: `train.py` — Training Utilities

### 2.1 Noam Learning Rate Schedule (paper §5.3)

```
lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
```

Two phases:
- **Linear warmup** (`step < warmup`): LR increases linearly
- **Inverse sqrt decay** (`step > warmup`): LR decreases as `1/√step`

The `d_model^(-0.5)` factor means larger models get smaller learning rates —
each parameter update has more impact when there are more parameters.

**Best practice — wrapped optimizer pattern:**
```python
class NoamOpt:
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()
```
Rather than using `torch.optim.lr_scheduler`, `NoamOpt` wraps the optimizer
and exposes `step()` and `zero_grad()`. The training loop calls `opt.zero_grad()`,
computes loss, calls `loss.backward()`, then `opt.step()` — exactly the standard
PyTorch pattern, but the scheduler is embedded inside `step()`. This is a useful
pattern when you want a scheduler that doesn't fit the standard API (Noam needs a
custom formula that isn't in `lr_scheduler`).

Adam hyperparameters: `β₁=0.9, β₂=0.98, ε=1e-9` (paper §5.3). Note `β₂=0.98`
instead of the PyTorch default `0.999` — the paper found this worked better,
likely because it makes the second-moment estimate more responsive.

### 2.2 Label Smoothing (paper §5.4)

```
q(k) = (1-ε)·1[k==y] + ε/(V-1)·1[k≠y]
```

With `ε=0.1`: the correct class gets 0.9 probability, and the remaining 0.1 is
spread uniformly over all other classes. This prevents the model from becoming
overconfident (predicting probability 1.0 on a single token), which improves
generalization and BLEU scores.

**What to notice — KL divergence, not cross-entropy:**

The loss is `KL(q || p)`, implemented explicitly by constructing the smoothed
distribution `q` as a tensor and calling `F.kl_div`:
```python
true_dist = x.new_full((x.size(-1),), self.smoothing / (vocab_size - 1))
true_dist.scatter_(-1, target.unsqueeze(-1), self.confidence)
```

Since `q` is fixed (no gradient through `q`), minimizing `KL(q||p)` is
equivalent to cross-entropy with soft targets. Using KL divergence makes the
smoothing computation explicit — students can see exactly what distribution the
model is being trained to match.

**Best practice — averaging over tokens, not positions:**
```python
n_tokens = (1 - mask.long()).sum()
return kl.sum() / n_tokens
```
Loss is divided by the number of **non-padding tokens**, not the number of
positions. This means a sequence of length 20 contributes 20× as much as a
sequence of length 1. If you divided by the batch size instead, short sequences
would be over-weighted relative to long ones.

### 2.3 Mask Construction

**`subsequent_mask(size)`:**
```python
mask = torch.triu(torch.ones(attn_shape, dtype=torch.uint8), diagonal=1)
return mask == 0
```
Uses `torch.uint8` instead of `torch.bool` because `triu` with boolean tensors
behaves inconsistently across PyTorch versions. Creating `uint8` ones, then
comparing to zero, gives a reliable boolean lower-triangular mask.

**`make_std_mask(tgt, pad)`:**
Combines padding mask `(B, 1, 1, S)` and subsequent mask `(1, S, S)` via `&`:
```python
tgt_mask = (tgt != pad).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
seq_mask = subsequent_mask(tgt.size(-1))             # (1, S, S)
return tgt_mask & seq_mask                           # (B, 1, S, S)
```
Broadcasting handles the dimension mismatch: `(B, 1, 1, S) & (1, 1, S, S) → (B, 1, S, S)`.

### 2.4 SyntheticData — the Copy Task

The default training setup uses a **copy task**: the model must learn to reproduce
the input sequence.

```
src:     [8, 12, 3, 4, <eos>]
tgt_in:  [<bos>, 8, 12, 3, 4]
tgt_out: [8, 12, 3, 4, <eos>]
```

The source includes `<eos>` as a terminator. The target input is shifted right by
one (starting with `<bos>`). The target output is what the model should predict —
the content tokens plus `<eos>`. This is the standard sequence-to-sequence
format: the model sees `<bos>` and must output the first content token, then sees
`<bos> token₁` and must output token₂, etc.

**Best practice — on-the-fly data generation:**
```python
data_fn = lambda bs: data.generate_batch(bs)
```
For synthetic educational data, batches are generated on-the-fly rather than
loaded from disk. This means unlimited fresh examples — no risk of overfitting
to a finite dataset since the model sees new token sequences every step.

### 2.5 Training Loop — Teacher Forcing

```python
logits = model(src, tgt_in, src_mask, tgt_mask)  # (B, S_tgt, V)
loss = loss_fn(logits, tgt_out)
```

The full target sequence is fed to the decoder in one pass (**teacher forcing**).
The model predicts all output tokens in parallel, which is possible because the
subsequent mask prevents each position from seeing future tokens. During
inference, this parallelism isn't available — we generate autoregressively
(one token at a time). The parallel training is what makes Transformers
trainable: RNNs must process sequentially both at training and inference.

---

## Chapter 3: `generate.py` — Inference

### 3.1 Greedy Decoding

```python
@torch.no_grad()
def greedy_decode(model, src, max_len=100, bos_idx=1, eos_idx=2, pad_idx=0):
```

The algorithm:
1. Encode source **once** → `memory` (cached, never changes)
2. Start with `[<bos>]`
3. Each step: feed current sequence to decoder, take `argmax` of last position's logits
4. Append the chosen token
5. Repeat until `<eos>` or `max_len`

```python
memory = model.encode(src, src_mask)  # computed once
# ...
for _ in range(max_len - 1):
    out = model.decode(ys, memory, src_mask, tgt_mask)
    logits = model.generator(out)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    # Only the LAST position matters — previous positions already produced tokens
```

**Key insight:** The encoder runs only once. This is the efficiency advantage
of encoder-decoder over pure-recurrent architectures: the source context is
fixed, so we compute it once and reuse it at every decoding step.

**Best practice — `@torch.no_grad()`:**
The decorator disables gradient tracking for the entire function. This saves
memory and prevents accidentally accumulating a computation graph during
inference. Always use it for generation code.

### 3.2 Beam Search

```python
def beam_search(model, src, beam_size=4, max_len=100, alpha=0.6, ...):
```

The algorithm maintains `beam_size` active hypotheses, expanding each by the
top `beam_size × 2` tokens per step, then pruning back to `beam_size`.

**What to notice — length penalty:**
```python
lp = ((5 + seq_len) ** alpha) / (6 ** alpha)
finished_scores.append(score / lp)
```
Without a penalty, beam search favors short sequences — shorter sequences have
fewer log-prob terms summed, so the sum is less negative. With `α=0.6`, the
penalty makes the effective preference roughly length-neutral. The constants 5
and 6 are from Wu et al. (2016), the same paper the original Transformer cites
for beam search.

**Pruning optimization:**
```python
top_lp, top_tokens = log_probs.topk(beam_size * 2, dim=-1)
```
Without pruning, we'd evaluate every token in the vocabulary (e.g. 40,000 for a
typical setup) per hypothesis per step. Pruning to `beam_size × 2` keeps the
search tractable while allowing some diversity in the candidates.

**Limitation:** This implementation supports `batch_size=1` only — multiple
source sentences would need independent beams, which requires batching logic
beyond the scope of an educational implementation.

---

## Chapter 4: `scripts/train.py` — Wiring It Together

This is the entry point that connects model, optimizer, loss, data, and
training loop:

```python
model = Transformer(src_vocab=40, tgt_vocab=40, N=3, d_model=128, d_ff=512, h=4)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
noam = NoamOpt(model_size=128, warmup=4000, optimizer=optimizer)
criterion = LabelSmoothing(smoothing=0.1, ignore_index=0)
data = SyntheticData(vocab_size=40, max_len=10)
```

The defaults use a small model (`d_model=128`, `N=3`) on a synthetic copy task,
so training runs in seconds on a CPU. This is intentional — students can
experiment without a GPU.

After training, the script runs a quick evaluation comparing greedy-decoded
output against expected targets, printing ✓/✗ per example.

---

## Chapter 5: `tests/test_smoke.py` — Verification

18 tests covering every component. The testing patterns are worth studying:

**Shape tests:** Verify each component preserves expected tensor shapes through
the forward pass (e.g., `test_multi_head_attention` checks `(B,S,512) → (B,S,512)`).

**Correctness tests:**
- `test_attention_masking`: masked positions get zero weight, unmasked sum to 1
- `test_positional_encoding_values`: `PE(pos,0) = sin(pos)` verified numerically
- `test_subsequent_mask`: exact boolean pattern checked

**Overfit test:** `test_transformer_loss_decreases` trains for 100 steps and
asserts that loss decreases. This catches bugs where the model compiles but
can't learn (e.g., wrong mask polarity, broken gradient flow).

**Weight tying test:** `test_weight_tying` asserts `is` (not `==`) — the three
matrices share the exact same tensor object.

---

## Original vs. Modern: What Changed

This section contextualizes what's different between this faithful 2017
implementation and the techniques used in today's LLMs (GPT-4, LLaMA, Claude,
Gemini, etc.).

### Architectural Changes

| Component | Original (this codebase) | Modern |
|-----------|--------------------------|--------|
| **Normalization** | Post-LN: `LayerNorm(x + Sublayer(x))` | Pre-LN: `x + Sublayer(LayerNorm(x))` |
| **Activation** | ReLU in FFN | GELU (GPT), SwiGLU (LLaMA, PaLM) |
| **Position encoding** | Sinusoidal (fixed, non-learnable) | RoPE — Rotary Position Embedding (LLaMA, GPT-NeoX, Mistral) or ALiBi |
| **Weight tying** | Yes (encoder embed = decoder embed = output) | Decoder-only models don't have separate encoder/decoder embeddings; some tie input/output embeddings (GPT-2), some don't (LLaMA) |
| **Bias** | Biases in all linear layers | Many modern models remove biases from linear layers (LLaMA, PaLM) — one fewer parameter type to tune |
| **Architecture** | Encoder-decoder (two stacks) | Decoder-only (GPT, LLaMA, Claude) — the entire model is a stack of decoder layers with causal self-attention only |
| **Attention** | Explicit Q, K, V projections in separate linear layers | Often implemented as a single fused `qkv_proj` that's split after projection |

### Pre-LN vs. Post-LN: Why It Matters

Post-LN (this codebase):
```
x → Sublayer(x) → + x → LayerNorm → output
```

Pre-LN (modern):
```
x → LayerNorm → Sublayer(x) → + x → output
```

Pre-LN stabilizes training, especially for deep models (>12 layers). With
Post-LN, gradients must propagate through LayerNorm at the end of each sublayer,
which can cause gradient vanishing in early layers. Pre-LN creates a "gradient
highway" through the residual connection to earlier layers. This is why the
original Transformer was limited to 6 layers, while modern models stack 80+
layers.

### Attention Efficiency

| Technique | What it does |
|-----------|-------------|
| **Flash Attention** | Fused CUDA kernel that computes attention in tiles, avoiding materializing the full `(S,S)` attention matrix in HBM. Speedup: 2-4×, memory: O(N) instead of O(N²) |
| **KV-Cache** | Stores key/value tensors from previous decoding steps so they don't need to be recomputed. At step `t`, only the new query vector is computed; past K/V are read from cache. This makes autoregressive generation O(N) instead of O(N²) |
| **GQA / MQA** | Grouped Query Attention (LLaMA 2) and Multi-Query Attention share K/V heads across query heads, reducing KV-cache memory by 8-16× |
| **torch.nn.functional.scaled_dot_product_attention** | PyTorch's built-in fused attention (since 2.0) that automatically selects the best backend (Flash Attention, Memory-Efficient Attention, or manual math). This codebase intentionally avoids it to keep the math explicit |

### Training Infrastructure

| Original (this codebase) | Modern |
|--------------------------|--------|
| Single GPU/CPU | Multi-GPU with FSDP/tensor parallelism/pipeline parallelism |
| FP32 | Mixed precision (AMP: Automatic Mixed Precision — FP16 forward, FP32 master weights) |
| Full batch in memory | Gradient accumulation (simulate large batches on limited hardware) |
| Noam schedule | Cosine decay with linear warmup (simpler, more predictable) |
| Adam (β₁=0.9, β₂=0.98) | AdamW (decoupled weight decay) with β₁=0.9, β₂=0.95 (LLaMA), or β₂=0.999 (GPT) |
| Label smoothing (ε=0.1) | Often not used in modern LLM pre-training (cross-entropy on next-token prediction with massive data is sufficient regularization) |

### Tokenization

This codebase uses raw integer token IDs for a synthetic copy task. In practice:

- **BPE (Byte-Pair Encoding):** Used by GPT-2/3/4 — merges frequent byte pairs iteratively
- **SentencePiece / Unigram:** Used by LLaMA, T5 — treats text as a sequence of subword units
- **Tokenizers are trained separately** on a large corpus and frozen before model training
- Modern vocab sizes: 32K (GPT-2), 50K (GPT-3), 128K (GPT-4), 32K (LLaMA), 256K (Claude)

### What Stays the Same

Some things from the original paper remain essentially unchanged:

- **Multi-head attention** — still the core mechanism (just more memory-efficient implementations)
- **Residual connections** — every layer in every modern Transformer uses them
- **Layer normalization** — still present (just moved to the Pre-LN position)
- **Transformer blocks** — the fundamental self-attention + FFN structure is unchanged, though SwiGLU has replaced the two-layer ReLU FFN in many models
- **Dropout** — still widely used, though some large models omit it when data volume is sufficient
