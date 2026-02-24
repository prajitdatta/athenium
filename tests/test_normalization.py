"""
tests/test_normalization.py
───────────────────────────
Athenium — Normalization Layer Test Suite

Covers: BatchNorm1dManual, LayerNormManual

Run: pytest tests/ -v
"""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.internals.normalization import BatchNorm1dManual, LayerNormManual


class TestLayerNorm:
    def test_output_shape_preserved(self):
        ln = LayerNormManual(64)
        x = torch.randn(4, 12, 64)
        assert ln(x).shape == x.shape

    def test_mean_near_zero_after_norm(self):
        """Each token's features should have near-zero mean after LayerNorm."""
        ln = LayerNormManual(64, elementwise_affine=False)
        x = torch.randn(3, 10, 64) * 5 + 3   # shifted, scaled input
        out = ln(x)
        means = out.mean(dim=-1)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)

    def test_var_near_one_after_norm(self):
        """Each token's features should have near-unit variance after LayerNorm."""
        ln = LayerNormManual(64, elementwise_affine=False)
        x = torch.randn(3, 10, 64) * 5 + 3
        out = ln(x)
        vars_ = out.var(dim=-1, unbiased=False)
        assert torch.allclose(vars_, torch.ones_like(vars_), atol=1e-4)

    def test_gamma_beta_learnable(self):
        """gamma and beta must be registered as Parameters."""
        ln = LayerNormManual(32)
        params = dict(ln.named_parameters())
        assert "gamma" in params
        assert "beta" in params

    def test_gamma_init_ones_beta_init_zeros(self):
        ln = LayerNormManual(16)
        assert torch.all(ln.gamma == 1.0)
        assert torch.all(ln.beta == 0.0)

    def test_works_at_batch_size_one(self):
        """LayerNorm must work with batch_size=1 (unlike BatchNorm)."""
        ln = LayerNormManual(32)
        x = torch.randn(1, 5, 32)
        out = ln(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_identical_train_eval(self):
        """LayerNorm has no running stats — train and eval must give same result."""
        ln = LayerNormManual(32)
        x = torch.randn(2, 5, 32)
        ln.train()
        out_train = ln(x)
        ln.eval()
        out_eval  = ln(x)
        assert torch.allclose(out_train, out_eval)


class TestBatchNorm:
    def test_output_shape_preserved(self):
        bn = BatchNorm1dManual(32)
        x = torch.randn(8, 32)
        bn.train()
        assert bn(x).shape == x.shape

    def test_mean_near_zero_train(self):
        """BatchNorm normalises feature means across batch to ~0."""
        bn = BatchNorm1dManual(32, affine=False)
        x = torch.randn(64, 32) * 4 + 2
        bn.train()
        out = bn(x)
        assert torch.allclose(out.mean(dim=0), torch.zeros(32), atol=1e-5)

    def test_var_near_one_train(self):
        bn = BatchNorm1dManual(32, affine=False)
        x = torch.randn(64, 32) * 4 + 2
        bn.train()
        out = bn(x)
        assert torch.allclose(out.var(dim=0, unbiased=False), torch.ones(32), atol=1e-4)

    def test_running_stats_updated_during_training(self):
        """Running mean/var must change after a forward pass in train mode."""
        bn = BatchNorm1dManual(8)
        x = torch.randn(16, 8) + 5   # mean≈5
        bn.train()
        bn(x)
        # Running mean should now be non-zero
        assert not torch.allclose(bn.running_mean, torch.zeros(8))

    def test_eval_uses_running_stats(self):
        """At eval time, BatchNorm uses running stats, not batch stats."""
        bn = BatchNorm1dManual(4, affine=False)
        # Warm up running stats
        bn.train()
        for _ in range(20):
            bn(torch.randn(32, 4))
        # Now eval with batch_size=1 — should not NaN
        bn.eval()
        single = torch.randn(1, 4)
        out = bn(single)
        assert not torch.isnan(out).any()
