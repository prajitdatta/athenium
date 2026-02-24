"""
positional_encoding.py
──────────────────────
Athenium — Token Embeddings & Positional Encoding

Transformers have no built-in notion of sequence order.
The attention operation — Q·Kᵀ / √dₖ — is a set operation.
If you permute the input tokens, the output is permuted identically.
Nothing in the math distinguishes token at position 3 from token at position 7.

This module covers how order is restored:

    1. Token embeddings    — mapping integer IDs to dense vectors
    2. Sinusoidal PE       — original Vaswani et al. (2017) approach
    3. Rotary PE (RoPE)    — used in Mistral, LLaMA, GPT-NeoX

Both sinusoidal and RoPE are implemented from scratch with full annotation.

Run this file directly to see a visual comparison:
    python -m athenium.src.embeddings.positional_encoding
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


# ── 1. Token Embedding Layer ──────────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    """
    Maps integer token IDs to dense floating-point vectors.

    A lookup table of shape (vocab_size, d_model).
    Row i is the learned representation of token i.

    At initialisation, every row is a random vector.
    Through training, the rows learn to encode semantic meaning:
    tokens with similar meanings cluster in embedding space.

    Args:
        vocab_size:  Number of tokens in the vocabulary (32,000 for Mistral)
        d_model:     Embedding dimension (4,096 for Mistral-7B)
        padding_idx: Token ID used for padding — its embedding is kept at zero
    """

    def __init__(
        self,
        vocab_size:  int = 32_000,
        d_model:     int = 4_096,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        # Scale embeddings by √d_model (Vaswani et al. §3.4)
        # Keeps embedding magnitudes compatible with positional encoding magnitudes
        self.scale = math.sqrt(d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: (batch, seq_len)   integer token IDs
        Returns:
            (batch, seq_len, d_model)     dense embedding vectors
        """
        return self.embedding(token_ids) * self.scale


# ── 2. Sinusoidal Positional Encoding ─────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    MOTIVATION
    ──────────
    A transformer receives a tensor of shape (batch, seq_len, d_model).
    Without positional information, the model treats this as an *unordered set*.
    Rearranging the tokens produces the same output, rearranged identically.
    The model cannot distinguish "the dog bit the man" from "the man bit the dog".

    SOLUTION
    ────────
    Add a unique, deterministic signal to each position's embedding *before*
    the first transformer block. After addition, token i's vector encodes both
    its semantic content AND its position in the sequence.

    ENCODING FORMULA
    ────────────────
    For position pos and embedding dimension i:

        PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

    Interpretation:
      - Each dimension pair (2i, 2i+1) oscillates at a different frequency.
      - Low dimensions (small i): short wavelength → sensitive to local position
      - High dimensions (large i): long wavelength → sensitive to global position
      - The 10000 base ensures wavelengths span from 2π (dim 0) to 10000·2π (dim d_model-1)

    WHY SINUSOIDAL (not learned)?
    ──────────────────────────────
    - Deterministic: no parameters to train; the encoding exists for any position
    - Extrapolates: works for sequences longer than any seen in training
    - Relative position: PE(pos+k) can be expressed as a linear function of PE(pos),
      so the model can learn to attend based on relative distance

    Args:
        d_model:   Embedding dimension
        max_len:   Maximum sequence length to precompute (can be extended)
        dropout:   Applied after adding positional encoding
    """

    def __init__(
        self,
        d_model: int = 512,
        max_len: int = 5_000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # ── Build PE table: (max_len, d_model) ───────────────────────────────
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # Denominator term: 10000^(2i / d_model) for each dimension pair
        # Computed in log space for numerical stability
        dim_idx = torch.arange(0, d_model, 2, dtype=torch.float)        # [0, 2, 4, ...]
        div_term = torch.exp(dim_idx * (-math.log(10_000.0) / d_model)) # (d_model/2,)

        # Even dimensions: sin
        pe[:, 0::2] = torch.sin(pos * div_term)   # (max_len, d_model/2)
        # Odd  dimensions: cos
        pe[:, 1::2] = torch.cos(pos * div_term)   # (max_len, d_model/2)

        # Register as buffer: moves with the model (e.g. to GPU) but not a parameter
        # (not trained — this is a fixed encoding)
        pe = pe.unsqueeze(0)                        # (1, max_len, d_model) for broadcasting
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to token embeddings.

        Args:
            x: (batch, seq_len, d_model) — token embeddings
        Returns:
            (batch, seq_len, d_model)    — embeddings + position signal
        """
        seq_len = x.size(1)
        # self.pe[:, :seq_len] selects the PE for positions 0..seq_len-1
        x = x + self.pe[:, :seq_len].detach()
        return self.dropout(x)


# ── 3. Rotary Positional Embedding (RoPE) ─────────────────────────────────────

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — used in Mistral, LLaMA, GPT-NeoX.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
               Position Embedding" (2021) — https://arxiv.org/abs/2104.09864

    CORE IDEA
    ─────────
    Instead of adding a positional signal to the embeddings (as sinusoidal PE does),
    RoPE *rotates* the Query and Key vectors inside each attention head by an angle
    proportional to the token's position.

    The rotation is applied in pairs of dimensions:
        For dimension pair (d, d+1) at position pos:
            q_d'   =  q_d  · cos(pos·θ_d)  −  q_{d+1} · sin(pos·θ_d)
            q_{d+1}' =  q_d  · sin(pos·θ_d)  +  q_{d+1} · cos(pos·θ_d)

    where θ_d = 1 / 10000^(2d / d_model)   (same base as sinusoidal PE)

    WHY ROTARY IS BETTER THAN ADDITIVE SINUSOIDAL
    ───────────────────────────────────────────────
    Key property: the dot product Q_i · K_j depends ONLY on (i − j), not on the
    absolute positions i and j independently.

    Proof sketch:
        Rotate(q, pos_i) · Rotate(k, pos_j)
        = q · R(pos_i)ᵀ · R(pos_j) · k
        = q · R(pos_j − pos_i) · k

    This means attention scores naturally encode *relative* position.
    Models generalise better to longer sequences at inference time.

    Sinusoidal PE added to embeddings does not share this property directly.

    Args:
        d_k:     Dimension per head (attention head size)
        max_len: Maximum sequence length
    """

    def __init__(self, d_k: int = 128, max_len: int = 4_096):
        super().__init__()
        self.d_k = d_k

        # Precompute rotation angles: θ_d = 1 / 10000^(2d / d_k)
        half = d_k // 2
        theta = 1.0 / (10_000 ** (torch.arange(0, half, dtype=torch.float) / half))

        # Outer product: (max_len, half) — angles for each position and dimension
        positions = torch.arange(max_len, dtype=torch.float)
        angles = torch.outer(positions, theta)               # (max_len, half)

        # Precompute cos and sin — applied at forward time
        self.register_buffer("cos_cache", torch.cos(angles)) # (max_len, half)
        self.register_buffer("sin_cache", torch.sin(angles)) # (max_len, half)

    def _rotate_half(self, x: Tensor) -> Tensor:
        """
        Rotate dimension pairs: [x1, x2, x3, x4] → [-x2, x1, -x4, x3]
        Used to apply the rotation in a single einsum-free operation.
        """
        x1 = x[..., : self.d_k // 2]    # first half of dims
        x2 = x[..., self.d_k // 2 :]    # second half of dims
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply RoPE rotation to Query and Key tensors.

        Args:
            q: (batch, heads, seq_len, d_k)
            k: (batch, heads, seq_len, d_k)

        Returns:
            q_rot, k_rot: rotated Q and K with position encoded in their directions
        """
        seq_len = q.size(2)
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d_k//2)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)

        # Expand cos/sin to full d_k by duplicating (matches the rotate_half pattern)
        cos = torch.cat([cos, cos], dim=-1)   # (1, 1, seq, d_k)
        sin = torch.cat([sin, sin], dim=-1)

        # Apply rotation: q_rot = q·cos + rotate_half(q)·sin
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot, k_rot


# ── Comparison Demo ───────────────────────────────────────────────────────────

def demo():
    """
    Print a side-by-side comparison of sinusoidal PE and RoPE.
    Shows the encoded values for the first 4 positions, first 8 dimensions.
    """
    print("\n" + "═" * 70)
    print("  ATHENIUM — Positional Encoding Comparison")
    print("═" * 70)

    d_model = 32
    seq_len = 8

    # ── Sinusoidal PE ─────────────────────────────────────────────────────────
    spe = SinusoidalPositionalEncoding(d_model=d_model, max_len=seq_len, dropout=0.0)
    dummy = torch.zeros(1, seq_len, d_model)
    encoded = spe(dummy)[0]   # (seq_len, d_model)

    print(f"\nSinusoidal PE (first 4 positions × first 8 dims of d_model={d_model}):")
    print("  pos │ " + " ".join(f" dim{i:02d}" for i in range(8)))
    print("  ────┼" + "─" * 56)
    for pos in range(4):
        vals = "  ".join(f"{encoded[pos, d].item():+.3f}" for d in range(8))
        print(f"  {pos:>3} │ {vals}")

    print("""
  Key properties:
    • Even dims use sin(), odd dims use cos()
    • Low dims oscillate fast (position-sensitive locally)
    • High dims oscillate slow (position-sensitive globally)
    • Fixed — no parameters trained
    • Works for seq lengths not seen in training
""")

    # ── RoPE ─────────────────────────────────────────────────────────────────
    d_k = 16
    rope = RotaryPositionalEmbedding(d_k=d_k, max_len=32)
    q = torch.ones(1, 1, seq_len, d_k)  # all-ones Q for clarity
    k = torch.ones(1, 1, seq_len, d_k)
    q_rot, _ = rope(q, k)

    print(f"RoPE — rotated Q vectors (first 4 positions × first 8 dims of d_k={d_k}):")
    print("  pos │ " + " ".join(f" dim{i:02d}" for i in range(8)))
    print("  ────┼" + "─" * 56)
    for pos in range(4):
        vals = "  ".join(f"{q_rot[0, 0, pos, d].item():+.3f}" for d in range(8))
        print(f"  {pos:>3} │ {vals}")

    print("""
  Key properties:
    • Applied INSIDE attention to Q and K — not added to embeddings
    • Q_i · K_j dot product encodes RELATIVE distance (i − j) only
    • Extrapolates to longer sequences than seen in training
    • Used in: Mistral, LLaMA, GPT-NeoX, Falcon
    • Athenium (Mistral-7B backbone) uses RoPE
""")

    print("  → See src/encoder/transformer_block.py for integration with MHA.\n")


if __name__ == "__main__":
    demo()
