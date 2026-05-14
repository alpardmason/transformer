"""Inference utilities: greedy decoding and beam search — JAX + Flax implementation.

Beam search uses length penalty alpha=0.6 per the original paper (Wu et al. 2016).

JAX/Flax vs PyTorch — key differences annotated throughout:
  - model.apply(variables, ..., method=model.encode, rngs={...}) for forward pass.
    variables contains the parameter tree from model.init().
  - PRNG key threading: every call that uses dropout needs a fresh key.
  - No @torch.no_grad() — JAX doesn't build autograd graphs during inference.
  - jnp.argmax(axis=...) replaces .argmax(dim=...).
  - jnp.concatenate replaces torch.cat.
  - jax.nn.log_softmax for log-probabilities.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn

from .train import make_std_mask


def greedy_decode(
    model: nn.Module,
    variables: dict,
    src: jnp.ndarray,
    max_len: int = 100,
    bos_idx: int = 1,
    eos_idx: int = 2,
    pad_idx: int = 0,
    rng_key: jax.Array | None = None,
) -> jnp.ndarray:
    """Greedy decoding: at each step, pick the most likely next token.

    Autoregressive generation loop:
      1. Encode source once → memory (cached, never changes)
      2. Start with [BOS]
      3. Each step: feed current sequence to decoder, take argmax of last position
      4. Append the chosen token, repeat until EOS or max_len

    JAX/Flax difference — model.apply() for inference:
      Instead of model(src), we call model.apply(variables, src, ...).
      variables is the dict from model.init() containing 'params'.

      We also call specific methods (encode, decode) via method= parameter:
        model.apply(variables, src, src_mask, method=model.encode)
      This is the standard Flax pattern for models with multiple forward paths.

    Shapes:
      src:     (B, S_src)        — input sequence
      memory:  (B, S_src, d_model)
      ys:      grows from (B, 1) to (B, up_to_max_len)
      logits:  (B, cur_len, V)   — logits[:, -1, :] gives next-token predictions
      returns: (B, up_to_max_len)

    Args:
        model: trained Flax Transformer model
        variables: variables dict from model.init() (or trainer state.params)
        src: (batch, src_len) source sequences
        max_len: maximum number of tokens to generate
        bos_idx, eos_idx, pad_idx: special token indices

    Returns:
        (batch, max_len) generated sequences (may be shorter if EOS was hit)
    """
    batch_size = src.shape[0]

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Encode source ONCE with deterministic=True (no dropout)
    src_mask = (src != pad_idx)[:, jnp.newaxis, jnp.newaxis, :]  # (B, 1, 1, S_src)
    rng_key, enc_key = jax.random.split(rng_key)
    memory = model.apply(
        variables,
        src, src_mask,
        method=model.encode,
        rngs={"dropout": enc_key},
        deterministic=True,
    )  # (B, S_src, d_model)

    ys = jnp.full((batch_size, 1), bos_idx, dtype=jnp.int32)  # (B, 1)

    for _ in range(max_len - 1):
        rng_key, dec_key = jax.random.split(rng_key)
        tgt_mask = make_std_mask(ys, pad_idx)  # (B, 1, cur_len, cur_len)
        # Decode with deterministic=True (no dropout during inference)
        out = model.apply(
            variables,
            ys, memory, src_mask, tgt_mask,
            method=model.decode,
            rngs={"dropout": dec_key},
            deterministic=True,
        )  # (B, cur_len, d_model)

        # Compute logits — use the generator Dense or tied embedding
        logits = out @ variables["params"]["generator"]["kernel"]
        # Only the LAST position's logits matter — that's the next token
        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)  # (B, 1)

        ys = jnp.concatenate([ys, next_token], axis=1)  # (B, cur_len+1)

        # Stop early if ALL batch items have produced EOS
        if (next_token == eos_idx).all():
            break

    return ys


def beam_search(
    model: nn.Module,
    variables: dict,
    src: jnp.ndarray,
    beam_size: int = 4,
    max_len: int = 100,
    alpha: float = 0.6,
    bos_idx: int = 1,
    eos_idx: int = 2,
    pad_idx: int = 0,
    rng_key: jax.Array | None = None,
) -> jnp.ndarray:
    """Beam search decoding with length penalty alpha, per the original paper.

    Algorithm:
      1. Start with k=1 hypothesis: [BOS] with score 0
      2. Each step, expand each active hypothesis with top beam_size*2 tokens
      3. From all expansions, keep the top beam_size by cumulative log-prob
      4. When a hypothesis ends with EOS, move it to `finished` with length penalty
      5. After max_len or beam exhaustion, pick the best finished hypothesis

    JAX/Flax difference — model.apply() per hypothesis:
      Each hypothesis gets its own forward pass via model.apply(). Since
      beam search runs beam_size parallel forward passes, this is not
      as efficient as batched beam search. For an educational implementation,
      the clarity is worth the performance trade-off.

      JAX doesn't have .topk() either — we use jnp.argsort + slice instead.

    batch_size=1 restriction: multiple source sentences would need independent
    beams, which requires batching logic.

    Args:
        model: trained Flax Transformer model
        variables: variables dict from model.init()
        src: (1, src_len) single source batch
        beam_size: number of hypotheses to keep (default 4)
        max_len: maximum generation length
        alpha: length penalty parameter (0.6 in the paper)

    Returns:
        (1, seq_len) best hypothesis
    """
    if src.shape[0] != 1:
        raise ValueError("beam_search supports batch_size=1 only")

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Encode source ONCE
    src_mask = (src != pad_idx)[:, None, None, :]  # (1, 1, 1, S_src)
    rng_key, enc_key = jax.random.split(rng_key)
    memory = model.apply(
        variables,
        src, src_mask,
        method=model.encode,
        rngs={"dropout": enc_key},
        deterministic=True,
    )  # (1, S_src, d_model)

    # Active hypotheses: each is a (1, cur_len) array
    sequences: list[jnp.ndarray] = [
        jnp.full((1, 1), bos_idx, dtype=jnp.int32)
    ]
    scores: list[float] = [0.0]  # cumulative log-prob

    # Completed (EOS-terminated) hypotheses
    finished_sequences: list[jnp.ndarray] = []
    finished_scores: list[float] = []

    for _ in range(max_len):
        if not sequences:
            break

        candidates: list[tuple[jnp.ndarray, float, int]] = []

        # Expand each active hypothesis
        for seq, cum_score in zip(sequences, scores):
            rng_key, dec_key = jax.random.split(rng_key)
            tgt_mask = make_std_mask(seq, pad_idx)  # (1, 1, cur_len, cur_len)
            out = model.apply(
                variables,
                seq, memory, src_mask, tgt_mask,
                method=model.decode,
                rngs={"dropout": dec_key},
                deterministic=True,
            )  # (1, cur_len, d_model)

            logits = out @ variables["params"]["generator"]["kernel"]
            log_probs = jax.nn.log_softmax(logits[:, -1, :], axis=-1)  # (1, V)

            # Only consider top beam_size*2 extensions per hypothesis.
            # JAX: no .topk() — use argsort descending + slice.
            sorted_idx = jnp.argsort(log_probs[0], axis=-1)[-beam_size * 2 :]
            top_tokens = sorted_idx[::-1]  # descending
            top_lp = log_probs[0, top_tokens]

            for j in range(top_tokens.shape[0]):
                token = int(top_tokens[j])
                lp = float(top_lp[j])
                candidate_seq = jnp.concatenate(
                    [seq, jnp.full((1, 1), token, dtype=jnp.int32)],
                    axis=1,
                )
                candidates.append((candidate_seq, cum_score + lp, token))

        # Sort all candidates by cumulative score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Keep top beam_size active, move EOS-terminated to finished
        sequences, scores = [], []
        for seq, score, last_token in candidates:
            if last_token == eos_idx:
                # Apply length penalty
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
    return jnp.full((1, 1), eos_idx, dtype=jnp.int32)
