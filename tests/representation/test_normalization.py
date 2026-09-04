"""Behavioral tests for ConditionalBatchNorm and fleet fallback hierarchy."""

from __future__ import annotations

import torch

from representation.layers.normalization import ConditionalBatchNorm


def test_conditional_batch_norm_dict_contract() -> None:
    """Inputs and outputs must be dict[str, torch.Tensor]."""
    cbn = ConditionalBatchNorm(
        num_sensors=6,
        condition_cardinalities={"robot_idx": 3, "program_idx": 4},
        min_bucket_samples=4,
    )
    b, c, t = 4, 6, 50
    inputs = {
        "input": torch.randn(b, c, t),
        "robot_idx": torch.tensor([0, 1, 2, 0]),
        "program_idx": torch.tensor([0, 1, 2, 3]),
        "valid_mask": torch.ones(b, t, dtype=torch.bool),
    }
    output = cbn(inputs)
    assert isinstance(output, dict)
    expected_keys = {"output", "fallback_level", "bucket_id", "mean", "std"}
    assert expected_keys.issubset(output.keys())
    assert output["output"].shape == (b, c, t)
    assert output["mean"].shape == (b, c)
    assert output["std"].shape == (b, c)
    assert output["fallback_level"].shape == (b,)
    assert output["bucket_id"].shape == (b,)


def test_conditional_batch_norm_fallback_hierarchy() -> None:
    """Verify fallback: (robot, program) [lvl 0] -> robot [lvl 1] -> fleet [lvl 2]."""
    min_samples = 4
    cbn = ConditionalBatchNorm(
        num_sensors=3,
        condition_cardinalities={"robot_idx": 2, "program_idx": 2},
        min_bucket_samples=min_samples,
    )

    # Initial forward (unpopulated): must use fleet fallback (level 2)
    inputs_init = {
        "input": torch.randn(2, 3, 20),
        "robot_idx": torch.tensor([0, 1]),
        "program_idx": torch.tensor([0, 1]),
    }
    out_init = cbn(inputs_init)
    assert (out_init["fallback_level"] == 2).all()

    # Train robot 0 on program 0 for min_samples times
    train_input = {
        "input": torch.randn(min_samples, 3, 20),
        "robot_idx": torch.zeros(min_samples, dtype=torch.long),
        "program_idx": torch.zeros(min_samples, dtype=torch.long),
    }
    cbn.train()
    cbn(train_input)

    # Inference check
    cbn.eval()
    test_query = {
        "input": torch.randn(3, 3, 20),
        # query 0: robot 0, program 0 (has >= min_samples -> level 0)
        # query 1: robot 0, program 1 (robot 0 has >= min_samples, but program 1 has 0 -> level 1)
        # query 2: robot 1, program 0 (robot 1 has 0 samples -> level 2)
        "robot_idx": torch.tensor([0, 0, 1]),
        "program_idx": torch.tensor([0, 1, 0]),
    }
    out_query = cbn(test_query)
    levels = out_query["fallback_level"].tolist()
    assert levels[0] == 0, f"Expected level 0 for (robot 0, prog 0), got {levels[0]}"
    assert levels[1] == 1, f"Expected level 1 for (robot 0, prog 1), got {levels[1]}"
    assert levels[2] == 2, f"Expected level 2 for (robot 1, prog 0), got {levels[2]}"


def test_conditional_batch_norm_mask_awareness() -> None:
    """Padded positions must be excluded from stats and masked to 0.0 in output."""
    cbn = ConditionalBatchNorm(num_sensors=2, min_bucket_samples=1)
    b, c, t = 2, 2, 10
    x = torch.ones(b, c, t) * 5.0
    # First file valid length = 6, second file valid length = 10
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, :6] = True
    mask[1, :10] = True

    cbn.train()
    out = cbn({"input": x, "valid_mask": mask})
    # Padded positions in output must be exactly zero
    assert (out["output"][0, :, 6:] == 0.0).all()
    assert torch.isfinite(out["output"]).all()


def test_conditional_batch_norm_eval_mode_freezes_statistics() -> None:
    """In eval mode, sufficient statistics must not mutate."""
    cbn = ConditionalBatchNorm(num_sensors=2, min_bucket_samples=2)
    inputs = {
        "input": torch.randn(2, 2, 20),
        "valid_mask": torch.ones(2, 20, dtype=torch.bool),
    }
    cbn.train()
    cbn(inputs)
    stats = cbn._get_stats(len(cbn.fallback_condition_sets) - 1)
    count_before = int(stats.sample_count[0].detach().cpu().numpy())

    cbn.eval()
    cbn(inputs)
    count_after = int(stats.sample_count[0].detach().cpu().numpy())
    assert count_before == count_after, "Eval mode mutated running statistics"


def test_conditional_batch_norm_state_dict_round_trip() -> None:
    """Statistics buffers must round trip through state_dict."""
    cbn1 = ConditionalBatchNorm(num_sensors=3, condition_cardinalities={"robot_idx": 2})
    inputs = {
        "input": torch.randn(4, 3, 20),
        "robot_idx": torch.tensor([0, 0, 1, 1]),
    }
    cbn1.train()
    cbn1(inputs)
    state = cbn1.state_dict()

    cbn2 = ConditionalBatchNorm(num_sensors=3, condition_cardinalities={"robot_idx": 2})
    cbn2.load_state_dict(state)
    s1 = cbn1._get_stats(0)
    s2 = cbn2._get_stats(0)
    assert bool(torch.equal(s1.value_sum, s2.value_sum))
    assert bool(torch.equal(s1.sample_count, s2.sample_count))
