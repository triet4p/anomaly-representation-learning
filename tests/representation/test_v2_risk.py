"""Task 15: calibrated confidence separate from censored survival risk."""

from __future__ import annotations

import pytest
import torch
from representation.v2_contracts import validate_confidence, validate_risk
from representation.v2_risk import (
    CensoredSurvivalRisk,
    HealthyTailCalibrator,
    expected_feature_width,
    trajectory_feature_matrix,
)


def _risk_data(seed: int = 7, n: int = 400):
    torch.manual_seed(seed)
    features = torch.randn(n, expected_feature_width())
    # Failure hazard grows with the displacement-like first column.
    logit = 2.0 * features[:, 0] - 0.5
    will_fail = torch.rand(n) < torch.sigmoid(logit)
    fail_time = torch.rand(n) * 10.0
    followup = 2.0 + torch.rand(n) * 12.0
    # Standard (time, indicator) encoding: observed failure time when the
    # failure falls inside follow-up, else right-censoring at follow-up.
    event = will_fail & (fail_time <= followup)
    days = torch.where(event, fail_time, followup)
    return features, days, event


def test_confidence_calibration_and_sparse_behavior() -> None:
    torch.manual_seed(0)
    healthy = torch.randn(200).abs()  # healthy displacement-like scores
    cal = HealthyTailCalibrator(min_samples=32).fit(healthy, cohort="dev-val")
    typical = cal.confidence(torch.tensor([healthy.median().item()]))
    extreme = cal.confidence(torch.tensor([healthy.max().item() + 5.0]))
    validate_confidence(typical)
    validate_confidence(extreme)
    assert float(typical["confidence"]) < 0.6  # typical healthy: low confidence
    assert float(extreme["confidence"]) > 0.95  # far tail: high confidence
    assert float(extreme["p_normal"]) < 0.05
    # Fresh healthy data keeps conformal coverage near the nominal level.
    fresh = torch.randn(500).abs()
    assert abs(cal.coverage(fresh, level=0.95) - 0.95) < 0.1
    # Sparse calibration floor: exactly min_samples still validates.
    sparse = HealthyTailCalibrator(min_samples=8).fit(torch.randn(8).abs(), cohort="dev-val")
    out = sparse.confidence(torch.tensor([100.0]))
    # Serialization preserves the tail exactly.
    clone = HealthyTailCalibrator(min_samples=32)
    clone.load_state_dict(cal.state_dict())
    torch.testing.assert_close(
        clone.confidence(healthy[:16])["confidence"],
        cal.confidence(healthy[:16])["confidence"],
    )


def test_censoring_horizon_ordering_and_brier() -> None:
    features, days, event = _risk_data()
    model = CensoredSurvivalRisk(expected_feature_width()).fit(features, days, event)
    proba = model.predict_proba(features)
    validate_risk(proba)
    assert bool((proba["risk_7d"] >= proba["risk_1d"] - 1e-6).all())
    # Higher displacement ranks higher on both horizons (no label leakage:
    # only historical features enter; check rank correlation instead).
    order = torch.argsort(features[:, 0])
    assert float(proba["risk_7d"][order[-50:]].mean()) > float(
        proba["risk_7d"][order[:50]].mean()
    ) + 0.2
    scores = model.brier_score(features, days, event)
    assert scores["risk_1d"] < 0.25 and scores["risk_7d"] < 0.25
    # Censored-before-horizon files are excluded from that horizon's cohort.
    included_1d, _ = CensoredSurvivalRisk.horizon_cohort(days, event, 1)
    censored = ~torch.isfinite(days)
    assert not bool((included_1d & censored).any())
    # Serialization round-trips exact probabilities.
    clone = CensoredSurvivalRisk(expected_feature_width())
    clone.load_state_dict(model.state_dict())
    for key in ("risk_1d", "risk_7d"):
        torch.testing.assert_close(clone.predict_proba(features)[key], proba[key])


def test_confidence_and_risk_remain_distinct() -> None:
    torch.manual_seed(3)
    healthy = torch.randn(100).abs()
    cal = HealthyTailCalibrator(min_samples=32).fit(healthy, cohort="dev-val")
    features, days, event = _risk_data(seed=11, n=300)
    risk = CensoredSurvivalRisk(expected_feature_width()).fit(features, days, event)
    # A merely unusual file (high confidence) need not carry high failure
    # probability: confidence keys and risk keys never overlap by type.
    conf = cal.confidence(torch.tensor([healthy.max().item() + 1.0]))
    low_risk_features = features[:1].clone()
    low_risk_features[:, 0] = -3.0
    low = risk.predict_proba(low_risk_features)
    assert float(conf["confidence"]) > 0.9
    assert float(low["risk_7d"]) < 0.5
    assert set(conf) == {"confidence", "p_normal"}
    assert set(low) == {"risk_1d", "risk_7d"}


def test_trajectory_feature_matrix_rejects_future_inputs() -> None:
    torch.manual_seed(5)
    b = 6
    trajectory = {
        name: torch.randn(b)
        for name in ("displacement", "velocity", "trend", "persistence", "disagreement")
    }
    tail = torch.randn(b)
    elevated = torch.rand(b)
    matrix = trajectory_feature_matrix(trajectory, tail, elevated)
    assert tuple(matrix.shape) == (b, expected_feature_width())
    assert torch.isfinite(matrix).all()


def test_confidence_fit_rejects_train_and_pooled_cohorts() -> None:
    """No pooled train fallback: non-dev-val cohorts raise instead of fitting."""
    torch.manual_seed(9)
    healthy = torch.randn(40).abs()
    for bad in (
        "dev-train",
        "pooled dev-train+dev-val (11 < floor 32; server runs use dev-val only)",
        "test_static",
        "train",
    ):
        with pytest.raises(ValueError, match="dev-val"):
            HealthyTailCalibrator(min_samples=8).fit(healthy, cohort=bad)
    # The accepted dev-val fit records its exact cohort.
    cal = HealthyTailCalibrator(min_samples=8).fit(healthy, cohort="dev-val")
    assert cal.fit_cohort_info()["cohort"] == "dev-val"


def test_confidence_state_carries_cohort_counts_and_small_sample_flag() -> None:
    torch.manual_seed(4)
    small = HealthyTailCalibrator(min_samples=4).fit(torch.randn(7).abs(), cohort="dev-val")
    info = small.fit_cohort_info()
    assert info["n_samples"] == 7
    assert info["min_samples"] == 4
    assert info["small_sample"] is True  # 7 valid fits below the 32 reference floor
    state = small.state_dict()
    assert state["fit_cohort"] == "dev-val"
    assert state["n_samples"] == 7
    assert state["small_sample"] is True
    large = HealthyTailCalibrator(min_samples=8).fit(torch.randn(64).abs(), cohort="dev-val")
    assert large.fit_cohort_info()["small_sample"] is False
    # Restore round-trips the provenance alongside the tail.
    clone = HealthyTailCalibrator(min_samples=4)
    clone.load_state_dict(state)
    assert clone.fit_cohort_info() == info
    torch.testing.assert_close(
        clone.confidence(torch.tensor([10.0]))["confidence"],
        small.confidence(torch.tensor([10.0]))["confidence"],
    )
    # Legacy states without a cohort restore as unknown, never as dev-val.
    legacy = HealthyTailCalibrator(min_samples=4)
    legacy.load_state_dict({"min_samples": 4, "scores": torch.randn(7).abs()})
    assert legacy.fit_cohort_info()["cohort"] is None
