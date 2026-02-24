"""
tests/test_gpu_memory.py
─────────────────────────
Athenium — GPU Memory Calculator Tests

Validates that the memory breakdown matches the engineering specification:
  10B fp32 model + Adam = ~149 GB total (4× rule, exact bytes)

Run: pytest tests/ -v
"""
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.internals.gpu_memory import calculate_training_memory, BYTES_PER_PARAM


class TestGPUMemoryCalculator:

    def test_10b_fp32_adam_equals_160gb(self):
        """10B fp32 model trained with Adam: weights + grads + Adam(m,v) = ~149 GB."""
        result = calculate_training_memory(n_params=10_000_000_000, param_dtype="fp32")
        assert abs(result.total_gb - 149.0) < 2.0, (
            f"Expected ~160 GB for 10B fp32 model, got {result.total_gb} GB"
        )

    def test_weights_40gb_for_10b_fp32(self):
        result = calculate_training_memory(10_000_000_000, "fp32")
        assert abs(result.params_gb - 40.0) < 0.5

    def test_gradients_40gb_for_10b_fp32(self):
        result = calculate_training_memory(10_000_000_000, "fp32")
        assert abs(result.gradients_gb - 40.0) < 0.5

    def test_adam_states_80gb_for_10b_fp32(self):
        """Adam m + v together should be ~80 GB for 10B fp32."""
        result = calculate_training_memory(10_000_000_000, "fp32")
        adam_total = result.adam_momentum_gb + result.adam_variance_gb
        assert abs(adam_total - 80.0) < 1.0

    def test_four_times_rule(self):
        """Total must be ~4× the weight memory for fp32 full training."""
        result = calculate_training_memory(7_000_000_000, "fp32")
        ratio = result.total_gb / result.params_gb
        assert abs(ratio - 4.0) < 0.05, f"4× rule violated: ratio={ratio:.3f}"

    def test_nf4_weights_half_fp32(self):
        """NF4 weights should use 0.5 bytes/param — half of int8, 1/8 of fp32."""
        fp32 = calculate_training_memory(1_000_000_000, "fp32")
        nf4  = calculate_training_memory(1_000_000_000, "nf4")
        assert abs(fp32.params_gb / nf4.params_gb - 8.0) < 0.1

    def test_lora_trainable_fraction(self):
        """LoRA adapters only: gradients and Adam states should be tiny."""
        # Mistral-7B: 8.4M trainable out of 7.24B
        result = calculate_training_memory(
            n_params=7_241_748_480,
            param_dtype="nf4",
            trainable_fraction=8_388_608 / 7_241_748_480,
        )
        # Gradients + Adam should be tiny (LoRA adapters only)
        adapter_memory = result.gradients_gb + result.adam_momentum_gb + result.adam_variance_gb
        assert adapter_memory < 0.5, f"LoRA adapter memory should be <0.5 GB, got {adapter_memory:.3f}"

    def test_bytes_per_param_constants(self):
        assert BYTES_PER_PARAM["fp32"] == 4
        assert BYTES_PER_PARAM["fp16"] == 2
        assert BYTES_PER_PARAM["bf16"] == 2
        assert BYTES_PER_PARAM["int8"] == 1
        assert BYTES_PER_PARAM["nf4"]  == 0.5
