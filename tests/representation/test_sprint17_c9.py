"""Focused Task 16 guards: C9 aggregation definitions, mask/geometry semantics,
edge-case finiteness, deterministic ties, localization/duration preservation,
branch isolation, and registry parity."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from representation.sprint17_c9 import (
    ARM_ADAPTER_DESCRIPTION,
    C9_ARMS,
    TOP_K_FRACTION,
    WINDOW_DURATION_TIMESTEPS,
    arm_file_scores,
    b0_patch_energies,
    file_mean,
    topk_mean,
    window_max_mean,
)

B0_PARAMS_TOTAL = 1821698
B0_FLOPS_REFERENCE = 182016709


def _latents(seed=7):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(2, 4, 6, generator=g)
    targ = torch.randn(2, 4, 6, generator=g)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[1, 2:] = False
    return pred, targ, mask


def test_b0_oracles_pin_score_batch_formula():
    """B0 patch energies + file mean reproduce the frozen inference formula."""
    import torch.nn.functional as F
    pred, targ, mask = _latents()
    want = F.mse_loss(pred, targ, reduction="none").mean(dim=-1)
    want = want.masked_fill(~mask, 0.0)
    assert torch.equal(b0_patch_energies(pred, targ, mask), want)
    counts = mask.sum(dim=1).to(want.dtype)
    assert torch.equal(file_mean(want, mask), want.sum(dim=1) / counts.clamp_min(1.0))
    assert torch.equal(file_mean(torch.ones(2, 3), torch.zeros(2, 3, dtype=torch.bool)),
                       torch.zeros(2))


def test_frozen_numerics_are_explicit():
    assert TOP_K_FRACTION == 0.25
    assert WINDOW_DURATION_TIMESTEPS == 64
    assert "0.25" in ARM_ADAPTER_DESCRIPTION["C9-A"]
    assert "64" in ARM_ADAPTER_DESCRIPTION["C9-B"]


def test_topk_hand_values_and_ceil():
    scores = np.array([1.0, 5.0, 3.0, 2.0])
    mask = np.ones(4, dtype=bool)
    # fraction 0.25 of 4 -> k=1 -> max.
    assert topk_mean(scores, mask, 0.25) == 5.0
    # fraction 0.5 of 4 -> k=2 -> mean of top two.
    assert topk_mean(scores, mask, 0.5) == 4.0
    # ceil boundary: fraction 0.3 of 4 -> k=2.
    assert topk_mean(scores, mask, 0.3) == 4.0
    # single valid patch: k=max(1, ...)=1 regardless of fraction.
    assert topk_mean(np.array([7.0]), np.array([True]), 0.25) == 7.0


def test_topk_fraction_one_equals_file_mean():
    rng = np.random.default_rng(11)
    scores = rng.random(9)
    mask = np.array([True] * 6 + [False] * 3)
    assert topk_mean(scores, mask, 1.0) == pytest.approx(scores[mask].mean())


def test_topk_empty_mask_scores_zero_and_rejects_bad_fraction():
    assert topk_mean(np.ones(3), np.zeros(3, dtype=bool), 0.25) == 0.0
    scores, mask = np.ones(3), np.ones(3, dtype=bool)
    for bad in (0.0, -0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            topk_mean(scores, mask, bad)


def test_topk_deterministic_ties_and_padding_exclusion():
    # Tied top scores resolve to the same set on repeat call (stable order).
    scores = np.array([4.0, 4.0, 4.0, 1.0])
    mask = np.ones(4, dtype=bool)
    assert topk_mean(scores, mask, 0.5) == 4.0
    # Masked-out padding values (even huge) never enter the top-k set.
    padded = np.array([1.0, 2.0, 1e9, -1e9])
    pmask = np.array([True, True, False, False])
    assert topk_mean(padded, pmask, 1.0) == 1.5
    # Non-finite valid scores fail fast instead of silently winning top-k.
    with pytest.raises(ValueError):
        topk_mean(np.array([1.0, np.inf]), np.array([True, True]), 0.5)


def test_window_max_dominates_file_mean_and_constant_collapses():
    rng = np.random.default_rng(23)
    scores = rng.random(6) + 0.5
    # All mask-true patches carry nonempty geometry (the predictor guarantees
    # prediction_mask = requested & patch_valid, so padding slots are never
    # scored; starts == -1 slots below are all masked out).
    mask = np.array([True, True, True, False, True, False])
    starts = np.array([0, 16, 32, 48, 64, -1])
    lens = np.array([32, 32, 32, 0, 16, 0])
    # The t=0 window covers every nonempty valid patch, so the maximum window
    # mean dominates the file mean (tail readout never dilutes below B0).
    score, _ = window_max_mean(scores, mask, starts, lens, duration=10 ** 6)
    assert score >= scores[mask].mean() - 1e-12
    # Constant valid scores collapse every window to the same value.
    const, prov = window_max_mean(np.full(6, 2.5), mask, starts, lens,
                                  duration=64)
    assert const == 2.5 and prov["start"] == 0


def test_window_excludes_geometry_empty_masked_slots():
    # Defense in depth: a mask-true slot with empty geometry (unreachable on
    # real inputs per the predictor mask contract) can never join a window.
    scores = np.array([100.0, 1.0])
    mask = np.array([True, True])
    starts = np.array([-1, 0])
    lens = np.array([0, 4])
    score, prov = window_max_mean(scores, mask, starts, lens, duration=4)
    assert score == 1.0 and prov["n_covered"] == 1


def test_window_hand_values_on_regular_lattice():
    scores = np.array([1.0, 10.0, 2.0])
    mask = np.ones(3, dtype=bool)
    starts = np.array([0, 4, 8])
    lens = np.array([4, 4, 4])
    # duration 4: each window covers exactly its own patch -> max patch score.
    score, prov = window_max_mean(scores, mask, starts, lens, duration=4)
    assert score == 10.0
    assert prov["start"] == 4 and prov["n_covered"] == 1
    assert prov["covered_span"] == 4


def test_window_duration_coverage_uses_valid_lengths():
    # Irregular last patch (short valid_len): a duration-8 window from 0
    # covers patches 0..1 but must not reach the short tail patch at 12.
    scores = np.array([9.0, 9.0, 1.0])
    mask = np.ones(3, dtype=bool)
    starts = np.array([0, 4, 12])
    lens = np.array([4, 4, 2])
    score, prov = window_max_mean(scores, mask, starts, lens, duration=8)
    assert score == 9.0
    assert prov == {"start": 0, "n_covered": 2, "covered_span": 8}
    # A window starting at the tail covers only the tail patch.
    score2, prov2 = window_max_mean(
        np.array([1.0, 1.0, 50.0]), mask, starts, lens, duration=8)
    assert score2 == 50.0 and prov2["start"] == 12


def test_window_earliest_tie_break_and_empty():
    scores = np.array([5.0, 5.0])
    mask = np.ones(2, dtype=bool)
    starts = np.array([0, 8])
    lens = np.array([4, 4])
    score, prov = window_max_mean(scores, mask, starts, lens, duration=4)
    assert score == 5.0 and prov["start"] == 0
    score0, prov0 = window_max_mean(np.ones(2), np.zeros(2, dtype=bool),
                                    starts, lens, duration=4)
    assert score0 == 0.0 and prov0 == {"start": None, "n_covered": 0,
                                       "covered_span": 0}


def test_window_rejects_malformed_geometry():
    scores = np.ones(3)
    mask = np.ones(3, dtype=bool)
    starts = np.array([0, 4, 8])
    lens = np.array([4, 4, 4])
    with pytest.raises(ValueError):
        window_max_mean(scores, mask, starts, lens, duration=0)
    with pytest.raises(ValueError):
        window_max_mean(scores, mask, starts, lens, duration=-8)
    with pytest.raises(ValueError):
        window_max_mean(scores, mask, starts, np.array([4, 4]), duration=4)
    with pytest.raises(ValueError):
        window_max_mean(scores, mask, starts, np.array([4, 4, -1]), duration=4)
    with pytest.raises(ValueError):
        window_max_mean(np.array([1.0, np.nan, 1.0]), mask, starts, lens,
                        duration=4)


def test_arm_dispatcher_isolation_and_no_fusion():
    """One branch's scores per call; C9-B needs geometry; unknown arms fail."""
    scores = np.array([1.0, 8.0, 3.0])
    mask = np.ones(3, dtype=bool)
    starts = np.array([0, 16, 32])
    lens = np.array([32, 32, 32])
    s_a, prov_a = arm_file_scores("C9-A", scores, mask)
    assert s_a == 8.0 and prov_a["fraction"] == TOP_K_FRACTION
    s_b, prov_b = arm_file_scores("C9-B", scores, mask, starts, lens)
    assert np.isfinite(s_b) and prov_b["duration"] == WINDOW_DURATION_TIMESTEPS
    with pytest.raises(ValueError):
        arm_file_scores("C9-B", scores, mask)
    with pytest.raises(ValueError):
        arm_file_scores("C9-C", scores, mask, starts, lens)
    # Structural: the dispatcher accepts a single score vector — there is no
    # second-branch argument to fuse.
    params = set(inspect.signature(arm_file_scores).parameters)
    assert "other" not in params and "fuse" not in params, params
    import representation.sprint17_c9 as C9
    bodies = "\n".join(inspect.getsource(getattr(C9, n)) for n in
                       ("topk_mean", "window_max_mean", "arm_file_scores"))
    for word in ("anomaly", "severity", "label", "S_pop"):
        assert word not in bodies, word


def test_registry_and_parity_declarations():
    from representation import sprint17_ablation as A
    from representation.sprint17_ablation import TrainingParity
    for arm in C9_ARMS:
        spec = A.get_arm(arm)
        A.validate_arm(spec)
        assert spec.kind == "single" and spec.components == ("C9",)
        assert ARM_ADAPTER_DESCRIPTION[arm]
    parity = A.validate_training_parity(TrainingParity())
    assert parity.model_seeds == (171701, 171702, 171703)
    assert parity.optimizer_steps == 300 and parity.checkpoint_step == 300


def test_driver_builds_unchanged_b0_graph_with_exact_counts():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "experiments"))
    import sprint17_task16_c9 as T16
    for arm in C9_ARMS:
        model, _, _ = T16.build_model(T16.arm_config_dict(arm, 171701),
                                      torch.device("cpu"))
        assert T16.count_parameters(model) == T16.B0_PARAMS == B0_PARAMS_TOTAL
        assert T16.count_flops_reference(model) == T16.B0_FLOPS_REFERENCE == B0_FLOPS_REFERENCE
