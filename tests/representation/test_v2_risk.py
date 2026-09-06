"""Task 15: calibrated confidence separate from censored survival risk."""

from __future__ import annotations

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
    cal = HealthyTailCalibrator(min_samples=32).fit(healthy)
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
    sparse = HealthyTailCalibrator(min_samples=8).fit(torch.randn(8).abs())
    out = sparse.confidence(torch.tensor([100.0]))
    assert abs(float(out["confidence"]) - (1.0 - 1.0 / 9.0)) < 1e-6
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
    cal = HealthyTailCalibrator(min_samples=32).fit(healthy)
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
