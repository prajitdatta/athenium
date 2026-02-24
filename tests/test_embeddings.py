"""
tests/test_embeddings.py
────────────────────────
Athenium — Positional Encoding Test Suite

Covers: TokenEmbedding, SinusoidalPositionalEncoding, RotaryPositionalEmbedding

Run: pytest tests/ -v
"""
import math
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.embeddings.positional_encoding import (
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    RotaryPositionalEmbedding,
)


class TestTokenEmbedding:
    def test_output_shape(self):
        emb = TokenEmbedding(vocab_size=1000, d_model=64)
        ids = torch.randint(0, 1000, (2, 10))
        out = emb(ids)
        assert out.shape == (2, 10, 64)

    def test_scale_applied(self):
        emb = TokenEmbedding(vocab_size=100, d_model=64)
        ids = torch.randint(0, 100, (1, 5))
        out = emb(ids)
        raw = emb.embedding(ids)
        # Output should be raw * sqrt(d_model)
        assert torch.allclose(out, raw * math.sqrt(64), atol=1e-5)


class TestSinusoidalPE:
    def test_output_shape_preserved(self):
        spe = SinusoidalPositionalEncoding(d_model=32, dropout=0.0)
        x = torch.zeros(2, 10, 32)
        out = spe(x)
        assert out.shape == x.shape

    def test_different_positions_different_encoding(self):
        """Each position must receive a unique encoding."""
        spe = SinusoidalPositionalEncoding(d_model=32, max_len=50, dropout=0.0)
        x = torch.zeros(1, 50, 32)
        out = spe(x)
        # Position 0 and position 1 should differ
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_even_dims_use_sin(self):
        """Even dimensions should be sin-encoded."""
        spe = SinusoidalPositionalEncoding(d_model=16, max_len=10, dropout=0.0)
        pe = spe.pe[0]  # (max_len, d_model)
        # At position 0: sin(0) = 0 for all even dims
        assert torch.allclose(pe[0, 0::2], torch.zeros(8), atol=1e-6)

    def test_odd_dims_use_cos(self):
        """Odd dimensions at position 0 should be cos(0) = 1."""
        spe = SinusoidalPositionalEncoding(d_model=16, max_len=10, dropout=0.0)
        pe = spe.pe[0]
        assert torch.allclose(pe[0, 1::2], torch.ones(8), atol=1e-6)

    def test_no_parameters_trained(self):
        """Sinusoidal PE has no trainable parameters."""
        spe = SinusoidalPositionalEncoding(d_model=32)
        params = list(spe.parameters())
        assert len(params) == 0, "Sinusoidal PE must have zero trainable parameters"


class TestRotaryPE:
    def test_output_shapes_preserved(self):
        rope = RotaryPositionalEmbedding(d_k=32)
        q = torch.randn(2, 4, 10, 32)
        k = torch.randn(2, 4, 10, 32)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotation_changes_vectors(self):
        """RoPE must change the Q and K vectors."""
        rope = RotaryPositionalEmbedding(d_k=32)
        q = torch.ones(1, 1, 5, 32)
        k = torch.ones(1, 1, 5, 32)
        q_rot, _ = rope(q, k)
        assert not torch.allclose(q, q_rot), "RoPE must transform the vectors"

    def test_position_zero_minimal_rotation(self):
        """At position 0: angle=0, rotation should be identity."""
        rope = RotaryPositionalEmbedding(d_k=4)
        q = torch.randn(1, 1, 1, 4)   # single token at position 0
        q_rot, _ = rope(q, q)
        # cos(0)=1, sin(0)=0 → rotation should preserve values
        assert torch.allclose(q_rot, q, atol=1e-5)

    def test_no_parameters(self):
        """RoPE is parameter-free."""
        rope = RotaryPositionalEmbedding(d_k=32)
        assert len(list(rope.parameters())) == 0
