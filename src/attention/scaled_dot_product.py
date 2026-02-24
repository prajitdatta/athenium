"""
scaled_dot_product.py
─────────────────────
Athenium — Core Attention Primitive

Implements scaled dot-product attention from first principles:

    Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

This is the fundamental operation that makes transformers context-aware.
Every piece of the formula exists for a specific, measurable reason.
This module is where you understand those reasons.

Reference: Vaswani et al., "Attention Is All You Need" (2017)
           https://arxiv.org/abs/1706.03762
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """
    Scaled dot-product attention.

    Args:
        Q:          Query matrix          shape (..., seq_len_q, d_k)
        K:          Key matrix            shape (..., seq_len_k, d_k)
        V:          Value matrix          shape (..., seq_len_k, d_v)
        mask:       Boolean mask          shape (..., seq_len_q, seq_len_k)
                    Positions where mask==False receive weight ~0 (set to -inf
                    before softmax). Use for padding or causal (autoregressive) masking.
        dropout_p:  Dropout on attention weights. Set 0.0 at inference.

    Returns:
        output:     Context-aware representations    shape (..., seq_len_q, d_v)
        weights:    Attention weight matrix           shape (..., seq_len_q, seq_len_k)
                    Each row is a probability distribution summing to 1.
                    Return this for interpretability and head visualisation.

    Raises:
        ValueError: If Q and K have incompatible d_k dimensions.
    """
    if Q.size(-1) != K.size(-1):
        raise ValueError(
            f"Q and K must share the same d_k dimension. "
            f"Got Q.d_k={Q.size(-1)}, K.d_k={K.size(-1)}."
        )

    d_k = Q.size(-1)

    # ── Step 1: Alignment scores ──────────────────────────────────────────────
    #
    # scores[..., i, j] = dot product of query i with key j
    #                    = "how aligned is what token i is looking for
    #                       with what token j is advertising?"
    #
    # In financial contracts: the token "default" at position 8 will produce
    # high alignment with "event" at position 5 because the model learned
    # that "event of default" is a legal defined term requiring joint
    # interpretation. That alignment is not hard-coded — it is learned.
    scores = Q @ K.transpose(-2, -1)                         # (..., seq_q, seq_k)

    # ── Step 2: Scale by √dₖ ─────────────────────────────────────────────────
    #
    # If Q and K are unit-normal, their dot product has variance dₖ.
    # At dₖ=128 (Athenium), variance is 128 — scores are large.
    # Large inputs to softmax push outputs toward one-hot distributions:
    # one position gets weight ~1.0, all others ~0.0.
    # In this regime, softmax gradients are ~0 everywhere. Training stops.
    #
    # Dividing by √dₖ reduces variance to ~1 regardless of dimension.
    scores = scores / math.sqrt(d_k)                         # (..., seq_q, seq_k)

    # ── Step 3: Masking ───────────────────────────────────────────────────────
    #
    # Padding mask: [PAD] tokens carry no semantic content.
    #   Attending to them would dilute real signal.
    # Causal mask: in decoder self-attention, position i must not
    #   see positions i+1...n (it would be looking at the future).
    #
    # Masked positions receive -∞. exp(-∞) = 0. Zero weight.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # ── Step 4: Softmax ───────────────────────────────────────────────────────
    #
    # Normalises scores to a probability distribution along the key axis.
    # Row i now represents token i's "attention budget" across the sequence:
    # how much of its representation should come from each other token.
    weights = F.softmax(scores, dim=-1)                      # (..., seq_q, seq_k)

    # ── Step 5: Dropout on weights ────────────────────────────────────────────
    #
    # Randomly zeroes some attention connections during training.
    # Prevents the model from over-relying on a single attention path.
    if dropout_p > 0.0 and torch.is_grad_enabled():
        weights = F.dropout(weights, p=dropout_p)

    # ── Step 6: Weighted aggregation of values ────────────────────────────────
    #
    # output[i] = Σⱼ weights[i,j] · V[j]
    #
    # Token i's new representation is a blend of all value vectors,
    # weighted by how much attention it paid to each position.
    # The same token "default" will produce a completely different
    # output vector in "event of default" vs "default parameter" —
    # because it attended to completely different neighbours.
    output = weights @ V                                     # (..., seq_q, d_v)

    return output, weights


def create_causal_mask(seq_len: int, device: torch.device = None) -> Tensor:
    """
    Lower-triangular causal mask for autoregressive (decoder) attention.

    Position i may attend only to positions 0, 1, ..., i.
    The upper triangle is masked, preventing the model from seeing the future.

    Returns shape: (1, 1, seq_len, seq_len) — broadcasts over batch and heads.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def create_padding_mask(token_ids: Tensor, pad_id: int = 0) -> Tensor:
    """
    Boolean padding mask from a batch of token ID sequences.

    Real tokens → True (attend). Padding tokens → False (ignore).

    Args:
        token_ids:  (batch, seq_len)
        pad_id:     Token ID used for padding (typically 0 or eos_token_id)

    Returns:
        (batch, 1, 1, seq_len) — broadcasts over heads and query positions.
    """
    mask = (token_ids != pad_id)                             # (batch, seq_len)
    return mask.unsqueeze(1).unsqueeze(2)                    # (batch, 1, 1, seq_len)
