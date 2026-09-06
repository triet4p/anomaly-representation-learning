"""Task 14: longitudinal baseline and trajectory tracker."""

from __future__ import annotations

import torch

from representation.v2_contracts import validate_trajectory
from representation.v2_trajectory import TrajectoryTracker


def _tracker(dim: int = 4, **kwargs) -> TrajectoryTracker:
    return TrajectoryTracker(dim, **kwargs)


def _commissioned(dim: int = 4, n: int = 32, seed: int = 0, **kwargs) -> TrajectoryTracker:
    torch.manual_seed(seed)
    tracker = _tracker(dim, **kwargs)
    states = torch.randn(n, dim) * 0.5
    tracker.fit_commissioning(states, torch.ones(n, dtype=torch.bool))
    return tracker


def test_slow_drift_grows_fixed_displacement_and_disagreement() -> None:
    tracker = _commissioned(trend_window=6, short_term_window=8, short_term_min_samples=3)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    disps, disagreements = [], []
    for step in range(12):
        feats = tracker.update(direction * (0.4 * step), is_suspect=False)
        validate_trajectory(feats)
        disps.append(float(feats["displacement"]))
        disagreements.append(float(feats["disagreement"]))
    # Fixed displacement grows monotonically-ish; short-term adapts, so
    # disagreement (fixed-vs-short) opens up: evidence of gradual migration.
    assert disps[-1] > disps[0] + 5.0
    assert disagreements[-1] > 0.0
    assert float(tracker.update(direction * 4.4)["trend"]) > 0.0


def test_abrupt_shift_and_benign_reversal() -> None:
    tracker = _commissioned()
    calm = [float(tracker.update(torch.zeros(4))["displacement"]) for _ in range(4)]
    shifted = tracker.update(torch.tensor([6.0, 0.0, 0.0, 0.0]))
    assert float(shifted["displacement"]) > max(calm) + 5.0
    assert float(shifted["velocity"]) > 0.0
    assert float(shifted["persistence"]) > 0.0
    # Benign reversal back to the commissioning neighborhood: displacement
    # collapses while the one-sided CUSUM retains the episode memory.
    returned = tracker.update(torch.zeros(4))
    assert float(returned["displacement"]) < float(shifted["displacement"]) / 2
    assert float(returned["persistence"]) >= 0.0


def test_maintenance_reset_starts_a_new_segment() -> None:
    tracker = _commissioned()
    for _ in range(5):
        tracker.update(torch.tensor([3.0, 0.0, 0.0, 0.0]))
    assert tracker._cusum > 0.0
    assert len(tracker._short) > 0
    fresh = tracker.update(torch.tensor([3.0, 0.0, 0.0, 0.0]), maintenance_reset=True)
    assert tracker._cusum == max(0.0, float(fresh["displacement"]) - tracker.cusum_kappa)
    assert len(tracker._short) <= 1  # only the post-reset file may seed the buffer
    assert float(fresh["velocity"]) == 0.0  # no step across the boundary


def test_suspect_files_never_update_either_reference() -> None:
    tracker = _commissioned()
    fixed_mu = tracker._fixed_mu.clone()
    fixed_cov = tracker._fixed_cov.clone()
    for _ in range(6):
        tracker.update(torch.zeros(4), is_suspect=False)
    short_len = len(tracker._short)
    updates = tracker.n_updates
    anomaly = torch.tensor([10.0, 10.0, 10.0, 10.0])
    scored = tracker.update(anomaly, is_suspect=True)
    assert float(scored["displacement"]) > 10.0  # suspect files are still scored
    torch.testing.assert_close(tracker._fixed_mu, fixed_mu)
    torch.testing.assert_close(tracker._fixed_cov, fixed_cov)
    assert len(tracker._short) == short_len  # short-term baseline untouched
    assert tracker.n_updates == updates
    # Serialization preserves the guarded state exactly.
    clone = _tracker()
    clone.load_state_dict(tracker.state_dict())
    torch.testing.assert_close(clone._fixed_mu, tracker._fixed_mu)
    assert clone._cusum == tracker._cusum
    assert clone.n_updates == tracker.n_updates
