"""Focused tests for the Sprint 15 candidate-2 frozen observable probe.

Covers the causal feature contract (shape, names, determinism, finiteness,
leakage-free signature), Fit-only standardization floor, frozen threshold
rule, and pre-outcome deterministic metric fixtures (tie-aware AUROC cases,
onset lead, bootstrap repeatability, false-alert grouping), plus an
end-to-end synthetic evaluation smoke. No data roots; synthetic inputs only.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from synth import events as E
from synth import probe15 as P

DAY = 86400.0


def _waveform(seed=0, channels=6, length=600):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(channels, length))).astype(np.float32)


def test_feature_names_match_vector():
    assert len(P.feature_names(6)) == 59
    assert len(set(P.feature_names(6))) == 59
    x = _waveform()
    assert P.extract_features(x, 600.0, 100.0).shape == (59,)


def test_features_deterministic_and_finite():
    x = _waveform(seed=3)
    first = P.extract_features(x, 600.0, 100.0)
    second = P.extract_features(x, 600.0, 100.0)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))


def test_features_finite_on_degenerate_inputs():
    const = np.full((6, 600), 2.5, dtype=np.float32)
    out = P.extract_features(const, 600.0, 0.0)
    assert out.shape == (59,) and np.all(np.isfinite(out))
    zeros = np.zeros((6, 600), dtype=np.float32)
    assert np.all(np.isfinite(P.extract_features(zeros, 0.0, -5.0)))


def test_feature_signatureadmits_no_labels_or_state():
    params = list(inspect.signature(P.extract_features).parameters)
    assert params == ["x", "duration_s", "time_since_reset_s"]


def test_standardization_floor_and_round_trip():
    rng = np.random.default_rng(11)
    healthy = rng.normal(size=(40, 59))
    healthy[:, 0] = 1.0  # constant column
    stats = P.standardize_fit(healthy)
    assert stats["std"][0] == P.STD_FLOOR
    z = P.apply_standardization(healthy, stats)
    assert np.all(np.isfinite(z))
    assert abs(z[:, 1:].mean()) < 0.05


def test_centroid_score_orders_anomaly_above_healthy():
    rng = np.random.default_rng(12)
    healthy = rng.normal(size=(200, 59))
    stats = P.standardize_fit(healthy)
    centroid = P.fit_centroid(P.apply_standardization(healthy, stats))
    assert centroid.shape == (59,)
    assert abs(centroid).max() < 0.2
    scores = P.score_files(P.apply_standardization(healthy, stats), centroid)
    assert np.all(scores >= 0.0)
    anomaly = np.full((10, 59), 5.0)
    anomaly_scores = P.score_files(P.apply_standardization(anomaly, stats),
                                   centroid)
    assert anomaly_scores.min() > np.median(scores)


def test_threshold_is_frozen_95th_percentile():
    scores = np.arange(100, dtype=np.float64)
    assert P.select_threshold(scores) == pytest.approx(94.05)
    assert P.THRESHOLD_QUANTILE == 0.95


def test_auroc_fixture_cases():
    assert E.roc_auc_tie_aware([0.5] * 6, [0.5] * 8) == 0.5
    assert E.roc_auc_tie_aware([1.0] * 6, [0.0] * 8) == 1.0
    assert E.roc_auc_tie_aware([0.0] * 6, [1.0] * 8) == 0.0
    assert E.lead_days(100.0 * DAY, [100.0 * DAY]) == 0.0


def test_bootstrap_deterministic_and_bounded():
    aucs = [0.7, 0.8, 0.6, 0.9]
    first = P.history_block_lcb(aucs)
    second = P.history_block_lcb(aucs)
    assert first == second
    assert first["lcb"] <= first["macro"] == pytest.approx(0.75)
    assert first["replicates"] == P.BOOTSTRAP_REPLICATES


def test_false_alert_grouping_fixture():
    false, far = E.false_alert_episodes({}, [], 10.0, {})
    assert (false, far) == (0, 0.0)
    flagged = {"robot-01": [0.0, 1.0 * DAY, 10.0 * DAY]}
    false, far = E.false_alert_episodes(flagged, [], 20.0, {})
    assert false == 2
    assert far == pytest.approx(0.1)


def _synthetic_history(robot, events=(("P", "P1", 15),), score_high=1.0,
                       score_low=0.0):
    rows = []
    for day in range(1, 41):
        rows.append({
            "file_id": f"{robot}-d{day}",
            "robot_id": robot,
            "program_id": "program-01",
            "start_time": (day - 0.01) * DAY,
            "end_time": day * DAY,
            "member_views": ["test-temporal"],
            "is_censored": False,
            "is_quarantined": False,
        })
    ledger = []
    for cohort, subtype, fail_day in events:
        ledger.append({
            "failure_id": f"fail-{robot}-{cohort}-{fail_day}",
            "robot_id": robot,
            "failure_time": fail_day * DAY,
            "cohort": cohort,
            "subtype": subtype,
            "degradation_onset": (fail_day - 10) * DAY,
            "duration_d": 10.0,
            "severity": 2.0,
        })
    horizons = [((fail_day - 7) * DAY, fail_day * DAY)
                for _, _, fail_day in events]
    scores = {}
    for row in rows:
        in_horizon = any(start <= row["end_time"] <= stop
                         for start, stop in horizons)
        scores[row["file_id"]] = score_high if in_horizon else score_low
    return rows, ledger, scores


def test_evaluate_histories_end_to_end_smoke():
    rows1, led1, s1 = _synthetic_history(
        "robot-01", (("P", "P1", 30), ("W", "W1", 10)))
    rows2, led2, s2 = _synthetic_history(
        "robot-02", (("P", "P2", 30), ("W", "W2", 10)))
    scores = {**s1, **s2}
    out = P.evaluate_histories(scores, [rows1, rows2], [led1, led2],
                               [{}, {}], threshold=0.5)
    assert out["probe"] == P.PROBE_ID
    assert out["macro_auc_pw"] == pytest.approx(1.0)
    assert out["directional_histories"] == 2
    assert out["macro_auc_p"] == pytest.approx(1.0)
    assert out["macro_auc_w"] == pytest.approx(1.0)
    for entry in out["per_history"]:
        assert entry["recall_p"] == pytest.approx(1.0)
        assert entry["recall_w"] == pytest.approx(1.0)
        assert entry["far"] == pytest.approx(0.0)
    assert out["per_history"][0]["lead_p_median"] == pytest.approx(7.0)
    assert out["per_history"][0]["lead_w_median"] == pytest.approx(7.0)
    again = P.evaluate_histories(scores, [rows1, rows2], [led1, led2],
                                 [{}, {}], threshold=0.5)
    assert json.dumps(out, sort_keys=True) == json.dumps(again, sort_keys=True)
