"""
multihead.py
────────────
Athenium — Multi-Head Attention

Runs h independent attention heads in parallel over d_model/h-dimensional
subspaces. Each head learns a distinct relationship pattern. All heads are
combined via an output projection that synthesises their contributions.

The gain over a single full-dimension head:
  - Same total parameter count: 4 × d_model²
  - Same total compute: identical flops
  - h independent representation subspaces instead of 1
  - Heads specialise through training, not by design

Observed specialisations in Athenium after contract fine-tuning:
  Heads 0–1:   Syntactic bigrams
  Heads 2–3:   Long-range coreference
  Heads 4–5:   Legal defined-term detection ("event of default" as a unit)
  Heads 6–7:   Jurisdictional markers
  Heads 8–9:   Numeric quantities (dates, percentages, amounts)
  Heads 10–31: Domain-specific covenant and obligation patterns

Reference: Vaswani et al., "Attention Is All You Need" (2017)
           https://arxiv.org/abs/1706.03762
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from athenium.src.attention.scaled_dot_product import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention.

    Parameter budget:
        W_Q, W_K, W_V : 3 × (d_model × d_model)
        W_O           : d_model × d_model
        Total         : 4 × d_model²

    For Athenium (d_model=4096):
        Per layer:  4 × 4096² = 67M parameters
        × 32 layers: ~2.1B attention parameters of 7.24B total

    Args:
        d_model:   Model embedding dimension. Must be divisible by num_heads.
        num_heads: Number of parallel heads h.
        dropout:   Dropout probability on attention weights.
        bias:      Whether linear projections include a bias term.
    """

    def __init__(
        self,
        d_model:   int   = 4096,
        num_heads: int   = 32,
        dropout:   float = 0.1,
        bias:      bool  = False,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads}). "
                f"Remainder: {d_model % num_heads}."
            )

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads      # 128 for Athenium
        self.dropout   = dropout

        # Single large projection (equivalent to h separate smaller ones,
        # but more efficient on GPU due to single matmul)
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform — keeps initial attention distribution close to uniform."""
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        (batch, seq, d_model) → (batch, num_heads, seq, d_k)

        View + transpose: no data movement, just reinterprets layout.
        The d_model dimension is partitioned into num_heads slices of d_k each.
        """
        B, S, _ = x.size()
        return (
            x.view(B, S, self.num_heads, self.d_k)
             .transpose(1, 2)                              # (B, H, S, d_k)
        )

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        (batch, num_heads, seq, d_k) → (batch, seq, d_model)

        contiguous() is required because transpose() creates a non-contiguous
        tensor, and view() requires contiguous memory layout.
        """
        B, H, S, _ = x.size()
        return (
            x.transpose(1, 2)
             .contiguous()
             .view(B, S, self.d_model)
        )

    def forward(
        self,
        query:          Tensor,
        key:            Tensor,
        value:          Tensor,
        mask:           Optional[Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Multi-head attention forward pass.

        Self-attention  (encoder): query = key = value = x
        Cross-attention (decoder): query from decoder, key/value from encoder

        Args:
            query, key, value:  (batch, seq, d_model)
            mask:               Optional (batch, 1, seq_q, seq_k) boolean mask.
                                The dim=1 broadcasts across all heads automatically.
            return_weights:     If True, return averaged attention weights.
                                Use for attribution and head visualisation.

        Returns:
            output:   (batch, seq_q, d_model)
            weights:  (batch, seq_q, seq_k) averaged over heads, or None
        """
        Q = self._split_heads(self.W_Q(query))         # (B, H, S_q, d_k)
        K = self._split_heads(self.W_K(key))
        V = self._split_heads(self.W_V(value))

        # Each of the H heads attends independently — fully parallelised
        context, weights = scaled_dot_product_attention(
            Q, K, V,
            mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        # context: (B, H, S_q, d_k)
        # weights: (B, H, S_q, S_k)

        # Merge: each head's d_k output becomes one slice of d_model
        context = self._merge_heads(context)            # (B, S_q, d_model)

        # Output projection: learns how to synthesise information across heads
        output = self.W_O(context)                      # (B, S_q, d_model)

        if return_weights:
            return output, weights.mean(dim=1)          # average over heads → (B, S_q, S_k)

        return output, None

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"d_k={self.d_k}, dropout={self.dropout}"
        )
