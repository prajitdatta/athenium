"""
normalization.py
────────────────
Athenium — Normalisation Layers

Normalisation is what makes deep networks trainable. Without it, activations
drift in scale across layers — gradients explode or vanish, and training
becomes unstable. This module implements both BatchNorm and LayerNorm from
scratch, explains precisely why each was designed, and shows why transformers
use LayerNorm while convolutional networks use BatchNorm.

The key distinction:
    BatchNorm  — normalises across the BATCH dimension (needs a batch of samples)
    LayerNorm  — normalises across the FEATURE dimension (per token, per sample)

Run:
    python -m src.internals.normalization
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


# ── 1. Batch Normalisation ────────────────────────────────────────────────────

class BatchNorm1dManual(nn.Module):
    """
    Batch Normalisation (Ioffe & Szegedy, 2015) — implemented from scratch.

    Reference: https://arxiv.org/abs/1502.03167

    WHAT IT DOES
    ────────────
    For each feature dimension d, across the entire mini-batch:
        1. Compute mean μ_d and variance σ²_d across the batch
        2. Normalise: x̂ = (x − μ) / √(σ² + ε)
        3. Rescale:   y = γ · x̂ + β   (γ, β are learned per-feature)

    WHERE MEMORY LIVES
    ──────────────────
        γ (scale):     (n_features,)  — learned, initialised to 1
        β (shift):     (n_features,)  — learned, initialised to 0
        running_mean:  (n_features,)  — tracked at training time, used at inference
        running_var:   (n_features,)  — tracked at training time, used at inference

    WHY RUNNING STATS?
    ──────────────────
    At inference time, we often classify single samples (batch size = 1).
    A single sample has no meaningful batch statistics.
    Solution: maintain exponentially weighted running mean and variance during training.
    Use those at inference regardless of batch size.

    BENEFITS
    ────────
    • Allows higher learning rates (smoother loss landscape)
    • Reduces internal covariate shift
    • Acts as a weak regulariser (noise from batch statistics)

    LIMITATIONS
    ───────────
    • Statistics depend on BATCH SIZE — small batches give noisy estimates
    • Behaviour differs between train and eval mode
    • Does NOT work well for variable-length sequences (different positions
      in a padded sequence are not "the same feature")
    • This is why transformers do NOT use BatchNorm

    Args:
        n_features:  Number of features (channels / embedding dims)
        eps:         Small constant for numerical stability in division
        momentum:    For running mean/var update: stat = (1-m)·stat + m·batch_stat
        affine:      If True, learn γ and β per-feature
    """

    def __init__(
        self,
        n_features: int,
        eps:        float = 1e-5,
        momentum:   float = 0.1,
        affine:     bool  = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.eps        = eps
        self.momentum   = momentum

        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))   # scale
            self.beta  = nn.Parameter(torch.zeros(n_features))  # shift
        else:
            self.gamma = self.beta = None

        # Running stats — not parameters, updated during training
        self.register_buffer("running_mean", torch.zeros(n_features))
        self.register_buffer("running_var",  torch.ones(n_features))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, n_features)  — 1D batch of feature vectors

        Returns:
            normalised x of same shape
        """
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0)                            # (n_features,)
            batch_var  = x.var(dim=0, unbiased=False)             # (n_features,)

            # Update running statistics for inference
            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * batch_mean.detach()
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * batch_var.detach()
            )
            mean, var = batch_mean, batch_var
        else:
            # At inference: use tracked running statistics
            mean, var = self.running_mean, self.running_var

        # Normalise
        x_norm = (x - mean) / torch.sqrt(var + self.eps)         # (batch, n_features)

        # Rescale with learned γ and β
        if self.gamma is not None:
            x_norm = self.gamma * x_norm + self.beta

        return x_norm


# ── 2. Layer Normalisation ────────────────────────────────────────────────────

class LayerNormManual(nn.Module):
    """
    Layer Normalisation (Ba et al., 2016) — implemented from scratch.

    Reference: https://arxiv.org/abs/1607.06450

    WHAT IT DOES
    ────────────
    For each individual sample (and each token position in a sequence),
    normalise across the FEATURE dimension:
        1. Compute mean μ and variance σ² across features for this single token
        2. Normalise: x̂ = (x − μ) / √(σ² + ε)
        3. Rescale:   y = γ · x̂ + β   (γ, β learned per-feature)

    THE KEY DIFFERENCE FROM BATCHNORM
    ──────────────────────────────────
    BatchNorm normalises across samples for each feature dimension.
    LayerNorm normalises across features for each sample.

    Visualised for a (batch=3, seq=4, d_model=8) tensor:

        BatchNorm: │ normalise along ↓ (across batch, per feature)
                   ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
                   [s0f0 s0f1 s0f2 ...]      sample 0
                   [s1f0 s1f1 s1f2 ...]      sample 1
                   [s2f0 s2f1 s2f2 ...]      sample 2

        LayerNorm: │ normalise along → (across features, per sample per token)
                   [s0f0→s0f1→s0f2→...]     normalise each row independently
                   [s1f0→s1f1→s1f2→...]
                   [s2f0→s2f1→s2f2→...]

    WHY LAYERNORM FOR TRANSFORMERS
    ──────────────────────────────
    1. Sequence independence: each token position is normalised independently.
       Padding tokens in one sequence do not affect statistics of real tokens.

    2. Batch size independence: LayerNorm statistics depend only on the feature
       dimension of a single sample. Works identically at batch size 1 (inference).

    3. No train/eval discrepancy: LayerNorm has no running statistics.
       The same computation runs at both training and inference time.

    4. Variable sequence lengths: different clauses in Athenium have different
       lengths. BatchNorm would mix statistics across positions (meaningless).
       LayerNorm treats each token independently — correct behaviour.

    Args:
        normalised_shape:  Feature dimensions to normalise over.
                           For (batch, seq, d_model): use (d_model,)
        eps:               Numerical stability constant
        elementwise_affine: If True, learn γ and β per-feature
    """

    def __init__(
        self,
        normalised_shape: int,
        eps: float = 1e-12,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalised_shape = normalised_shape
        self.eps = eps

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalised_shape))
            self.beta  = nn.Parameter(torch.zeros(normalised_shape))
        else:
            self.gamma = self.beta = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., normalised_shape)   — any leading dimensions, last = features

        Returns:
            Normalised tensor of same shape
        """
        # Normalise over the last dimension (feature dimension)
        mean = x.mean(dim=-1, keepdim=True)                        # (..., 1)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)         # (..., 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.gamma is not None:
            x_norm = self.gamma * x_norm + self.beta

        return x_norm


# ── Side-by-Side Comparison ───────────────────────────────────────────────────

def comparison_demo():
    """
    Demonstrate the key behavioural difference with a concrete example.
    """
    print("\n" + "═" * 65)
    print("  ATHENIUM — BatchNorm vs LayerNorm")
    print("  Why transformers use LayerNorm")
    print("═" * 65)

    torch.manual_seed(42)

    # Simulate: batch=4, seq=6, features=8
    # Represents 4 contract clauses, 6 tokens each, d_model=8
    B, S, F = 4, 6, 8
    x = torch.randn(B, S, F)

    # ── LayerNorm: each token normalised independently ────────────────────────
    ln = LayerNormManual(F)
    ln_out = ln(x)

    print(f"""
  Input: {B} contract clauses, {S} tokens each, d_model={F}
  Shape: {list(x.shape)}

  ── LAYERNORM ─────────────────────────────────────────────────

  Normalises across features (dim=-1) for each token independently.

  Token 0, Sample 0 — before LayerNorm:
    mean={x[0, 0].mean().item():+.4f}, var={x[0, 0].var().item():.4f}
    values: {x[0, 0].detach().numpy().round(3)}

  Token 0, Sample 0 — after LayerNorm:
    mean={ln_out[0, 0].mean().item():+.6f} (≈ 0)
    var= {ln_out[0, 0].var().item():.6f}  (≈ 1)
    values: {ln_out[0, 0].detach().numpy().round(3)}

  Key: each of the {B*S} tokens is normalised independently.
  Batch size has ZERO effect on these statistics.
  Works identically at train time and inference time (no running stats).
""")

    # ── BatchNorm: normalises across the batch ────────────────────────────────
    # Requires reshaping: (B*S, F) — treats each (sample, position) as one element
    x_flat = x.view(B * S, F)
    bn     = BatchNorm1dManual(F)
    bn.train()
    bn_out = bn(x_flat)

    print(f"""  ── BATCHNORM ────────────────────────────────────────────────

  Normalises across the batch (dim=0) for each feature.

  Feature 0 — batch mean across {B*S} (sample, position) pairs:
    mean={x_flat[:, 0].mean().item():+.4f}, var={x_flat[:, 0].var().item():.4f}
  After BatchNorm:
    mean≈{bn_out[:, 0].mean().item():+.6f}, var≈{bn_out[:, 0].var().item():.6f}

  Problem 1: At inference (batch_size=1), there is no batch to normalise over.
             BatchNorm must use running stats from training — train/eval mismatch.

  Problem 2: Padding. If clauses have different lengths, padded positions
             contaminate the batch statistics for real tokens.

  Problem 3: Feature independence. Position 0 ("The") and position 5 ("default")
             are mixed in the same batch statistic for each feature.
             This conflates very different semantic positions.

  ── WHEN TO USE EACH ─────────────────────────────────────────

  BatchNorm  → Convolutional networks (CNNs), vision models, ResNet, EfficientNet
               Large, fixed batch sizes. Same spatial positions across samples.

  LayerNorm  → Transformers and all sequence models (BERT, GPT, Mistral, Athenium)
               Variable sequence lengths. Any batch size. Consistent train/eval behaviour.

  Athenium uses LayerNorm with eps=1e-12, placed BEFORE each sublayer (pre-norm).
  See: src/encoder/transformer_block.py — self.norm_attn and self.norm_ffn
""")


if __name__ == "__main__":
    comparison_demo()
