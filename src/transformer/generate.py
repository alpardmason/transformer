"""Inference utilities: greedy decoding and beam search.

Beam search uses length penalty alpha=0.6 per the original paper (Wu et al. 2016).
"""

import torch
import torch.nn.functional as F

from .layers import Transformer
from .train import make_std_mask


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_len: int = 100,
    bos_idx: int = 1,
    eos_idx: int = 2,
    pad_idx: int = 0,
) -> torch.Tensor:
    """Greedy decoding: at each step, pick the most likely next token.

    Autoregressive generation loop:
      1. Encode source once → memory (cached, never changes)
      2. Start with [BOS]
      3. Each step: feed current sequence to decoder, take argmax of last position
      4. Append the chosen token, repeat until EOS or max_len

    The encoder runs only ONCE — this is the key efficiency of encoder-decoder
    architectures: the source is encoded once, then each decoding step just
    reuses the memory.

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
    device = src.device
    batch_size = src.size(0)

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S_src)
    memory = model.encode(src, src_mask)  # (B, S_src, d_model) — computed once

    ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)  # (B, 1)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)  # (B,)

    for _ in range(max_len - 1):
        tgt_mask = make_std_mask(ys, pad_idx)  # (B, 1, cur_len, cur_len)
        out = model.decode(ys, memory, src_mask, tgt_mask)  # (B, cur_len, d_model)
        logits = model.generator(out)  # (B, cur_len, V)
        # Only the LAST position's logits matter — that's the next token
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
        # Finished sequences keep extending with PAD — no tokens after first EOS
        next_token = torch.where(
            finished.unsqueeze(1),
            torch.full_like(next_token, pad_idx),
            next_token,
        )

        ys = torch.cat([ys, next_token], dim=1)  # (B, cur_len+1)
        finished = finished | (next_token.squeeze(1) == eos_idx)

        if finished.all():
            break

    return ys


@torch.no_grad()
def beam_search(
    model: Transformer,
    src: torch.Tensor,
    beam_size: int = 4,
    max_len: int = 100,
    alpha: float = 0.6,
    bos_idx: int = 1,
    eos_idx: int = 2,
    pad_idx: int = 0,
) -> torch.Tensor:
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
    device = src.device

    if src.size(0) != 1:
        raise ValueError("beam_search supports batch_size=1 only")

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (1, 1, 1, S_src)
    memory = model.encode(src, src_mask)  # (1, S_src, d_model) — computed once

    # Active hypotheses: each is a (1, cur_len) tensor
    sequences: list[torch.Tensor] = [
        torch.full((1, 1), bos_idx, dtype=torch.long, device=device)
    ]
    scores: list[float] = [0.0]  # cumulative log-prob for each active hypothesis

    # Completed (EOS-terminated) hypotheses
    finished_sequences: list[torch.Tensor] = []
    finished_scores: list[float] = []

    for _ in range(max_len):
        if not sequences:
            break

        candidates: list[tuple[torch.Tensor, float, int]] = []

        # Expand each active hypothesis
        for seq, cum_score in zip(sequences, scores):
            tgt_mask = make_std_mask(seq, pad_idx)  # (1, 1, cur_len, cur_len)
            out = model.decode(seq, memory, src_mask, tgt_mask)  # (1, cur_len, d_model)
            log_probs = F.log_softmax(model.generator(out)[:, -1, :], dim=-1)  # (1, V)

            # Only consider top beam_size*2 extensions per hypothesis.
            # This pruning is an optimization — without it we'd evaluate
            # V candidates per hypothesis (e.g. 40,000 for a typical vocab).
            top_lp, top_tokens = log_probs.topk(beam_size * 2, dim=-1)

            for j in range(top_tokens.size(-1)):
                token = int(top_tokens[0, j].item())
                lp = float(top_lp[0, j].item())
                candidate_seq = torch.cat(
                    [seq, torch.full((1, 1), token, dtype=torch.long, device=device)],
                    dim=1,
                )
                candidates.append((candidate_seq, cum_score + lp, token))

        # Sort all candidates by cumulative score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Keep top beam_size active, move EOS-terminated to finished
        sequences, scores = [], []
        for seq, score, last_token in candidates:
            if last_token == eos_idx:
                # Apply length penalty to finished hypotheses
                seq_len = seq.size(1)
                lp = ((5 + seq_len) ** alpha) / (6**alpha)
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
    return torch.full((1, 1), eos_idx, dtype=torch.long, device=device)
