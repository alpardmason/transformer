"""Inference utilities: greedy decoding and beam search — MLX implementation.

Beam search uses length penalty alpha=0.6 per the original paper (Wu et al. 2016).

MLX vs PyTorch — key differences annotated throughout:
  - No @torch.no_grad() decorator: MLX doesn't build autograd graphs during
    inference (gradients are only computed when you call mx.value_and_grad).
  - mx.argmax(axis=...) replaces .argmax(dim=...).
  - mx.concatenate replaces torch.cat.
  - mx.log_softmax for log-probabilities (same API).
  - No .topk() method — use mx.argsort + slice for beam search pruning.
  - model.eval() sets dropout to identity (no separate no_grad needed).
"""

import mlx.core as mx
import mlx.nn as nn

from .train import make_std_mask


def greedy_decode(
    model: nn.Module,
    src: mx.array,
    max_len: int = 100,
    bos_idx: int = 1,
    eos_idx: int = 2,
    pad_idx: int = 0,
) -> mx.array:
    """Greedy decoding: at each step, pick the most likely next token.

    Autoregressive generation loop:
      1. Encode source once → memory (cached, never changes)
      2. Start with [BOS]
      3. Each step: feed current sequence to decoder, take argmax of last position
      4. Append the chosen token, repeat until EOS or max_len

    The encoder runs only ONCE — this is the key efficiency of encoder-decoder
    architectures: the source is encoded once, then each decoding step just
    reuses the memory.

    MLX difference: No @torch.no_grad() needed. MLX is lazy and only computes
    gradients when explicitly asked via mx.value_and_grad. During inference,
    the gradient graph is never built.

    Shapes:
      src:     (B, S_src)        — input sequence
      memory:  (B, S_src, d_model)
      ys:      grows from (B, 1) to (B, up_to_max_len)
      logits:  (B, cur_len, V)   — logits[:, -1, :] gives next-token predictions
      returns: (B, up_to_max_len)

    Args:
        model: trained Transformer model
        src: (batch, src_len) source sequences
        max_len: maximum number of tokens to generate
        bos_idx, eos_idx, pad_idx: special token indices

    Returns:
        (batch, max_len) generated sequences (may be shorter if EOS was hit)
    """
    model.eval()
    batch_size = src.shape[0]

    src_mask = (src != pad_idx)[:, None, None, :]  # (B, 1, 1, S_src)
    memory = model.encode(src, src_mask)            # (B, S_src, d_model) — computed once

    ys = mx.full((batch_size, 1), bos_idx, dtype=mx.int32)  # (B, 1)

    for _ in range(max_len - 1):
        tgt_mask = make_std_mask(ys, pad_idx)              # (B, 1, cur_len, cur_len)
        out = model.decode(ys, memory, src_mask, tgt_mask) # (B, cur_len, d_model)
        logits = model.generator(out)                      # (B, cur_len, V)
        # Only the LAST position's logits matter — that's the next token
        # MLX: mx.argmax(axis=...) instead of .argmax(dim=...)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)  # (B, 1)

        # MLX: mx.concatenate replaces torch.cat
        ys = mx.concatenate([ys, next_token], axis=1)      # (B, cur_len+1)

        # Stop early if ALL batch items have produced EOS
        if (next_token == eos_idx).all().item():
            break

    return ys


def beam_search(
    model: nn.Module,
    src: mx.array,
    beam_size: int = 4,
    max_len: int = 100,
    alpha: float = 0.6,
    bos_idx: int = 1,
    eos_idx: int = 2,
    pad_idx: int = 0,
) -> mx.array:
    """Beam search decoding with length penalty alpha, per the original paper.

    Algorithm:
      1. Start with k=1 hypothesis: [BOS] with score 0
      2. Each step, expand each active hypothesis with top beam_size*2 tokens
      3. From all expansions, keep the top beam_size by cumulative log-prob
      4. When a hypothesis ends with EOS, move it to `finished` with length penalty
      5. After max_len or beam exhaustion, pick the best finished hypothesis

    Length penalty: score = sum(log P) / ((5 + |Y|)^α / 6^α)
    - Without penalty (α=0), beam search favors short sequences (fewer log-prob terms
      are added, so the sum is less negative)
    - With α=0.6, longer sequences are penalized less than they would be at α=1,
      making the effective preference roughly length-neutral
    - The 5 and 6 constants are from Wu et al. (2016); they make the penalty ~1
      for typical sequence lengths

    MLX difference — no .topk() method:
      MLX doesn't have a direct topk. Instead we use mx.argsort + slice to
      get the top-k values and indices. This is less efficient but equivalent.

      mx.argsort(arr, axis=-1) returns indices sorted ascending, so we slice
      the last beam_size*2 elements (largest) for top-k.

    batch_size=1 restriction: multiple source sentences would need independent
    beams, which requires batching logic. Here each hypothesis is a separate
    forward pass — beam_size=1 recomputes the encoder once, but then decodes
    beam_size hypotheses step by step.

    Args:
        model: trained Transformer model
        src: (1, src_len) single source batch
        beam_size: number of hypotheses to keep (default 4)
        max_len: maximum generation length
        alpha: length penalty parameter (0.6 in the paper)

    Returns:
        (1, seq_len) best hypothesis
    """
    model.eval()

    if src.shape[0] != 1:
        raise ValueError("beam_search supports batch_size=1 only")

    src_mask = (src != pad_idx)[:, None, None, :]  # (1, 1, 1, S_src)
    memory = model.encode(src, src_mask)            # (1, S_src, d_model) — computed once

    # Active hypotheses: each is a (1, cur_len) array
    sequences: list[mx.array] = [
        mx.full((1, 1), bos_idx, dtype=mx.int32)
    ]
    scores: list[float] = [0.0]  # cumulative log-prob for each active hypothesis

    # Completed (EOS-terminated) hypotheses
    finished_sequences: list[mx.array] = []
    finished_scores: list[float] = []

    for _ in range(max_len):
        if not sequences:
            break

        candidates: list[tuple[mx.array, float, int]] = []

        # Expand each active hypothesis
        for seq, cum_score in zip(sequences, scores):
            tgt_mask = make_std_mask(seq, pad_idx)                     # (1, 1, cur_len, cur_len)
            out = model.decode(seq, memory, src_mask, tgt_mask)        # (1, cur_len, d_model)
            # MLX: no mx.log_softmax — compute log after softmax manually
            log_probs = mx.log(mx.softmax(model.generator(out)[:, -1, :], axis=-1))  # (1, V)

            # Only consider top beam_size*2 extensions per hypothesis.
            # MLX: no .topk() method — use argsort descending + slice.
            # argsort returns ascending order; [..., ::-1] makes it descending.
            sorted_indices = mx.argsort(log_probs[0], axis=-1)[..., ::-1][: beam_size * 2]
            top_tokens = sorted_indices
            top_lp = log_probs[0, top_tokens]

            for j in range(top_tokens.shape[0]):
                token = int(top_tokens[j].item())
                lp = float(top_lp[j].item())
                candidate_seq = mx.concatenate(
                    [seq, mx.full((1, 1), token, dtype=mx.int32)],
                    axis=1,
                )
                candidates.append((candidate_seq, cum_score + lp, token))

        # Sort all candidates by cumulative score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Keep top beam_size active, move EOS-terminated to finished
        sequences, scores = [], []
        for seq, score, last_token in candidates:
            if last_token == eos_idx:
                # Apply length penalty to finished hypotheses
                seq_len = seq.shape[1]
                lp = ((5 + seq_len) ** alpha) / (6 ** alpha)
                finished_sequences.append(seq)
                finished_scores.append(score / lp)
            else:
                sequences.append(seq)
                scores.append(score)

            if len(sequences) == beam_size:
                break

    # Return best finished hypothesis (if any exist)
    if finished_sequences:
        best_idx = max(range(len(finished_sequences)), key=lambda i: finished_scores[i])
        return finished_sequences[best_idx]

    # If no hypothesis ended with EOS, return the best active one
    if sequences:
        return sequences[0]

    # Fallback: return EOS token
    return mx.full((1, 1), eos_idx, dtype=mx.int32)
