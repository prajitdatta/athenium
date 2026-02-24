"""
tests/test_attention.py
───────────────────────
Athenium — Attention Module Test Suite

Covers: scaled_dot_product_attention, MultiHeadAttention, mask utilities.

Run:  pytest tests/ -v
      pytest tests/ -v --tb=short   (compact tracebacks)
"""

import math
import pytest
import torch
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.attention.scaled_dot_product import (
    scaled_dot_product_attention,
    create_causal_mask,
    create_padding_mask,
)
from src.attention.multihead import MultiHeadAttention


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def qkv():
    torch.manual_seed(0)
    B, S, d_k = 2, 10, 64
    Q = torch.randn(B, S, d_k)
    K = torch.randn(B, S, d_k)
    V = torch.randn(B, S, d_k)
    return Q, K, V


@pytest.fixture
def mha():
    torch.manual_seed(1)
    return MultiHeadAttention(d_model=256, num_heads=8, dropout=0.0)


@pytest.fixture
def batch():
    torch.manual_seed(2)
    return torch.randn(3, 12, 256)    # (B=3, S=12, d_model=256)


# ─────────────────────────────────────────────────────────────────────────────
# scaled_dot_product_attention
# ─────────────────────────────────────────────────────────────────────────────

class TestScaledDotProductAttention:

    def test_output_shape_matches_query(self, qkv):
        Q, K, V = qkv
        output, _ = scaled_dot_product_attention(Q, K, V)
        assert output.shape == Q.shape

    def test_weights_shape_is_seq_sq(self, qkv):
        Q, K, V = qkv
        B, S, _ = Q.shape
        _, weights = scaled_dot_product_attention(Q, K, V)
        assert weights.shape == (B, S, S)

    def test_weights_rows_sum_to_one(self, qkv):
        Q, K, V = qkv
        _, weights = scaled_dot_product_attention(Q, K, V)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            "Every attention weight row must sum to 1.0"

    def test_weights_are_non_negative(self, qkv):
        Q, K, V = qkv
        _, weights = scaled_dot_product_attention(Q, K, V)
        assert (weights >= 0).all(), "Attention weights are softmax outputs — must be ≥ 0"

    def test_scaling_reduces_score_magnitude(self, qkv):
        Q, K, V = qkv
        d_k = Q.size(-1)
        raw    = Q @ K.transpose(-2, -1)
        scaled = raw / math.sqrt(d_k)
        assert scaled.abs().mean() < raw.abs().mean(), \
            "Scaling by √d_k must reduce mean absolute score magnitude"

    def test_causal_mask_zeros_upper_triangle(self, qkv):
        Q, K, V = qkv
        B, S, _ = Q.shape
        mask = create_causal_mask(S)
        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        for i in range(S):
            for j in range(i + 1, S):
                assert weights[:, i, j].max().item() < 1e-6, \
                    f"Causal mask: position [{i}, {j}] (future) must receive ~0 weight"

    def test_padding_mask_zeros_masked_positions(self, qkv):
        Q, K, V = qkv
        B, S, _ = Q.shape
        mask = torch.ones(B, S, S, dtype=torch.bool)
        mask[:, :, -3:] = False    # mask last 3 key positions
        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        assert weights[:, :, -3:].max().item() < 1e-6, \
            "Masked positions must receive zero attention weight"

    def test_dimension_mismatch_raises_value_error(self):
        Q = torch.randn(2, 5, 64)
        K = torch.randn(2, 5, 32)    # wrong d_k
        V = torch.randn(2, 5, 64)
        with pytest.raises(ValueError, match="d_k"):
            scaled_dot_product_attention(Q, K, V)

    def test_output_no_nan_or_inf(self, qkv):
        Q, K, V = qkv
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert not torch.isnan(output).any(),   "Output contains NaN"
        assert not torch.isinf(output).any(),   "Output contains Inf"
        assert not torch.isnan(weights).any(),  "Weights contain NaN"


# ─────────────────────────────────────────────────────────────────────────────
# MultiHeadAttention
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiHeadAttention:

    def test_output_shape(self, mha, batch):
        output, _ = mha(batch, batch, batch)
        assert output.shape == batch.shape, \
            f"MHA output must match input shape. Got {output.shape}"

    def test_d_k_per_head(self, mha):
        assert mha.d_k == 256 // 8 == 32

    def test_invalid_d_model_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(d_model=100, num_heads=8)

    def test_weights_returned_when_requested(self, mha, batch):
        B, S, _ = batch.shape
        _, weights = mha(batch, batch, batch, return_weights=True)
        assert weights is not None
        assert weights.shape == (B, S, S), \
            f"Weights shape should be (B, S, S)=(3,12,12). Got {weights.shape}"

    def test_weights_none_by_default(self, mha, batch):
        _, weights = mha(batch, batch, batch)
        assert weights is None, "Weights must be None when return_weights=False"

    def test_eval_mode_is_deterministic(self, mha, batch):
        mha.eval()
        with torch.no_grad():
            out1, _ = mha(batch, batch, batch)
            out2, _ = mha(batch, batch, batch)
        assert torch.allclose(out1, out2), \
            "MHA must be deterministic in eval mode (no dropout)"

    def test_output_no_nan_or_inf(self, mha, batch):
        output, _ = mha(batch, batch, batch)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_weight_average_over_heads(self, mha, batch):
        B, S, _  = batch.shape
        H        = mha.num_heads
        _, weights = mha(batch, batch, batch, return_weights=True)
        # Averaged over H heads — should still be (B, S, S), non-negative, rows sum to 1
        assert weights.shape == (B, S, S)
        assert (weights >= 0).all()
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Mask utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskUtilities:

    def test_causal_mask_shape(self):
        mask = create_causal_mask(8)
        assert mask.shape == (1, 1, 8, 8)

    def test_causal_mask_is_lower_triangular(self):
        mask = create_causal_mask(5)[0, 0]    # (5, 5)
        for i in range(5):
            for j in range(i + 1):
                assert mask[i, j].item(), f"mask[{i},{j}] should be True (attend)"
            for j in range(i + 1, 5):
                assert not mask[i, j].item(), f"mask[{i},{j}] should be False (future)"

    def test_padding_mask_shape(self):
        ids  = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        mask = create_padding_mask(ids, pad_id=0)
        assert mask.shape == (2, 1, 1, 5)

    def test_padding_mask_real_vs_pad(self):
        ids  = torch.tensor([[1, 2, 0, 0]])
        mask = create_padding_mask(ids, pad_id=0)
        assert mask[0, 0, 0, 0].item() == True
        assert mask[0, 0, 0, 1].item() == True
        assert mask[0, 0, 0, 2].item() == False
        assert mask[0, 0, 0, 3].item() == False
