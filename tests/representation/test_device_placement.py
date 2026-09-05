"""Device-placement regression tests for ConditionalBatchNorm reductions.

Covers the Kaggle Stage 1 crash
``RuntimeError: Expected all tensors to be on the same device ... (wrapper_CUDA_index_add)``:
with a CUDA model in training mode fed a CPU batch, ``_update_statistics`` called
``torch.index_add`` with CUDA accumulators but a CPU index. The layer must co-locate
every index, accumulator, and source used by each reduction without avoidable copies
(the ``.to()`` calls below are no-ops when already aligned, so single-device runs
pay no copies; only the mixed-device path copies the small per-batch tensors).
"""

from __future__ import annotations

import pytest
import torch

from representation.layers.normalization import ConditionalBatchNorm


def test_update_statistics_accumulates_exact_counts_and_sums() -> None:
    """Train-mode updates must record exact per-bucket counts and masked sums."""
    cbn = ConditionalBatchNorm(
        num_sensors=2,
        condition_cardinalities={"robot_idx": 2},
        min_bucket_samples=1,
    )
    cbn.train()
    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]],
            [[10.0, 20.0, 30.0, 40.0], [1.0, 1.0, 1.0, 1.0]],
        ]
    )
    inputs = {
        "input": x,
        "valid_mask": torch.tensor([[True, True, False, False], [True, True, True, True]]),
        "robot_idx": torch.tensor([0, 1]),
    }
    cbn(inputs)
    stats = cbn._get_stats(0)
    assert stats.sample_count.tolist() == [1, 1]
    # Masked sums: robot 0 sees (1+2)=3 over 2 valid steps; robot 1 sees 100 over 4.
    assert stats.value_sum[0, 0].item() == pytest.approx(3.0)
    assert stats.value_sum[0, 1].item() == pytest.approx(1.0)
    assert stats.value_sum[1, 0].item() == pytest.approx(100.0)
    assert stats.value_sum[1, 1].item() == pytest.approx(4.0)
    assert stats.value_count[0, 0].item() == pytest.approx(2.0)
    assert stats.value_count[1, 0].item() == pytest.approx(4.0)


def test_eval_forward_is_deterministic_and_freezes_statistics() -> None:
    """Eval forwards must be bit-identical and leave accumulators untouched."""
    cbn = ConditionalBatchNorm(
        num_sensors=2,
        condition_cardinalities={"robot_idx": 2},
        min_bucket_samples=1,
    )
    cbn.train()
    inputs = {
        "input": torch.randn(4, 2, 16),
        "valid_mask": torch.ones(4, 16, dtype=torch.bool),
        "robot_idx": torch.tensor([0, 0, 1, 1]),
    }
    cbn(inputs)
    cbn.eval()
    first = cbn(inputs)
    counts = cbn._get_stats(0).sample_count.clone()
    second = cbn(inputs)
    torch.testing.assert_close(first["output"], second["output"])
    assert torch.equal(cbn._get_stats(0).sample_count, counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_model_with_cpu_batch_matches_colocated_numerics() -> None:
    """Replicate the Kaggle failure mode: CUDA train-mode model, CPU batch.

    The mixed-device forward must run and match an all-CPU reference exactly.
    Pre-fix this raised the reported ``wrapper_CUDA_index_add`` RuntimeError.
    """
    torch.manual_seed(7)
    conditioned = {
        "input": torch.randn(4, 3, 24),
        "valid_mask": torch.ones(4, 24, dtype=torch.bool),
        "robot_idx": torch.tensor([0, 0, 1, 1]),
    }

    def build() -> ConditionalBatchNorm:
        module = ConditionalBatchNorm(
            num_sensors=3,
            condition_cardinalities={"robot_idx": 2},
            min_bucket_samples=1,
        )
        module.train()
        return module

    on_cuda = build().to("cuda")
    cuda_out = on_cuda({k: v.cpu() for k, v in conditioned.items()})
    assert on_cuda._get_stats(0).value_sum.is_cuda

    torch.manual_seed(7)
    on_cpu = build()
    cpu_out = on_cpu({k: v.cpu() for k, v in conditioned.items()})

    assert cuda_out["output"].device.type == "cpu"
    torch.testing.assert_close(cuda_out["output"], cpu_out["output"])
    torch.testing.assert_close(cuda_out["mean"], cpu_out["mean"])
    assert cuda_out["fallback_level"].tolist() == cpu_out["fallback_level"].tolist()
    assert on_cuda._get_stats(0).sample_count.cpu().tolist() == (
        on_cpu._get_stats(0).sample_count.tolist()
    )
