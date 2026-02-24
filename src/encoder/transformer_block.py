"""
transformer_block.py
────────────────────
Athenium — Full Transformer Encoder Block

One complete transformer layer:
  Pre-LayerNorm → Multi-Head Attention → Dropout → Residual
  Pre-LayerNorm → Feed-Forward Network → Dropout → Residual

This module is the repeating unit. Athenium stacks 32 of these.

Architecture choice: pre-norm (LayerNorm before each sublayer).
Used in GPT-2, LLaMA, Mistral. More stable than post-norm at depth.
See ADR-004 for the decision rationale.

References:
  Vaswani et al. (2017) — https://arxiv.org/abs/1706.03762  (original)
  Xiong et al. (2020)   — https://arxiv.org/abs/2002.04745  (pre-norm analysis)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from athenium.src.attention.multihead import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

        FFN(x) = Activation( x · W₁ ) · W₂

    Applied identically and independently to each token position.
    No cross-position interaction — that is attention's job.

    The FFN is where the model stores most of its "knowledge":
    attention routes information between tokens; FFN processes it
    per-token. Geva et al. (2021) show FFN layers act as key-value
    memories encoding factual associations.

    d_ff = 4 × d_model is the Vaswani convention. The expand-compress
    bottleneck forces the model to prioritise useful transformations.

    Activation choices:
        'gelu': smooth, used in BERT, GPT-2 — default for Athenium
        'relu': original paper, sparse activations
        'silu': used in LLaMA/Mistral (also called Swish)

    Args:
        d_model:    Input and output dimension
        d_ff:       Inner expansion dimension (default: 4 × d_model)
        activation: Activation function name
        dropout:    Applied after inner activation
    """

    _ACTIVATIONS = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
    }

    def __init__(
        self,
        d_model:    int   = 4096,
        d_ff:       Optional[int] = None,
        activation: str   = "gelu",
        dropout:    float = 0.1,
    ):
        super().__init__()

        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {list(self._ACTIVATIONS)}. "
                f"Got '{activation}'."
            )

        self.d_model = d_model
        self.d_ff    = d_ff or 4 * d_model

        self.net = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            self._ACTIVATIONS[activation],
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, self.d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)                                    # (B, S, d_model)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


class TransformerBlock(nn.Module):
    """
    One pre-norm transformer encoder layer.

    Pre-norm vs Post-norm:
        Post-norm: x = LayerNorm(x + sublayer(x))
            Residual bypasses normalisation → activations can grow unbounded
            Requires careful LR warmup. Becomes unstable above ~12 layers.

        Pre-norm:  x = x + sublayer(LayerNorm(x))   ← Athenium
            LayerNorm gates the sublayer input → controlled magnitude at all depths
            Trains stably at 32+ layers without warmup. Used in Mistral, LLaMA, GPT-2.

    Why residual connections?
        Each layer learns only the delta — what to add to the current representation.
        Gradients flow directly through residual paths to early layers.
        Without residuals, gradient magnitude decays exponentially with depth.
        With residuals, training 32 layers is as stable as training 4.

    Args:
        d_model:    Embedding dimension
        num_heads:  Attention heads (d_model must be divisible by num_heads)
        d_ff:       FFN inner dimension
        dropout:    Applied after each sublayer
        activation: FFN activation function
        eps:        LayerNorm epsilon for numerical stability
    """

    def __init__(
        self,
        d_model:    int   = 4096,
        num_heads:  int   = 32,
        d_ff:       Optional[int] = None,
        dropout:    float = 0.1,
        activation: str   = "gelu",
        eps:        float = 1e-12,
    ):
        super().__init__()

        self.norm_attn = nn.LayerNorm(d_model, eps=eps)
        self.norm_ffn  = nn.LayerNorm(d_model, eps=eps)

        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:                Tensor,
        mask:             Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x:                (batch, seq_len, d_model)
            mask:             Optional padding mask
            return_attention: Return attention weights for interpretability

        Returns:
            x:            Updated representations  (batch, seq_len, d_model)
            attn_weights: Averaged weights or None (batch, seq_len, seq_len)
        """
        # ── Sublayer 1: Multi-Head Self-Attention ─────────────────────────────
        residual = x
        attn_out, attn_weights = self.attention(
            self.norm_attn(x),
            self.norm_attn(x),
            self.norm_attn(x),
            mask=mask,
            return_weights=return_attention,
        )
        x = residual + self.dropout(attn_out)

        # ── Sublayer 2: Feed-Forward Network ──────────────────────────────────
        residual = x
        x = residual + self.dropout(self.ffn(self.norm_ffn(x)))

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Full transformer encoder: N stacked TransformerBlock layers.

    The [CLS] token's final representation (position 0) is extracted
    by the classification head for the contract risk decision.

    Args:
        n_layers:   Number of stacked blocks (32 for Athenium)
        d_model:    Embedding dimension
        num_heads:  Attention heads per layer
        d_ff:       FFN inner dimension
        dropout:    Applied throughout
        activation: FFN activation
    """

    def __init__(
        self,
        n_layers:   int   = 32,
        d_model:    int   = 4096,
        num_heads:  int   = 32,
        d_ff:       Optional[int] = None,
        dropout:    float = 0.1,
        activation: str   = "gelu",
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model, eps=1e-12)

    def forward(
        self,
        x:                      Tensor,
        mask:                   Optional[Tensor] = None,
        return_all_attentions:  bool = False,
    ) -> Tuple[Tensor, Optional[list]]:
        """
        Args:
            x:                    (batch, seq_len, d_model)
            mask:                 Optional padding mask
            return_all_attentions: Collect weights from every layer

        Returns:
            x:              Final encoder output  (batch, seq_len, d_model)
            all_attentions: List of per-layer weights, or None
        """
        all_attentions = [] if return_all_attentions else None

        for layer in self.layers:
            x, attn = layer(
                x, mask=mask,
                return_attention=return_all_attentions,
            )
            if return_all_attentions:
                all_attentions.append(attn)

        return self.norm_out(x), all_attentions
