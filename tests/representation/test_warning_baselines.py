"""Focused tests for Batch F causal/censoring/leak behavior (plausible bugs)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from sprint12_task12_warning import file_score, in_maint  # noqa: E402


def test_in_maint_boundary_semantics():
    maint = {"r": [(100.0, 200.0)]}
    assert in_maint(maint, "r", 150.0, 160.0) is True  # contained
    assert in_maint(maint, "r", 50.0, 100.0) is False  # ends exactly at start
    assert in_maint(maint, "r", 200.0, 250.0) is False  # starts exactly at end
    assert in_maint(maint, "r", 199.0, 201.0) is True  # straddles
    assert in_maint({}, "r", 150.0, 160.0) is False  # unknown robot


def test_file_score_deterministic_finite():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 256)).astype(float)
    a, b = file_score(x), file_score(x)
    assert a == b and np.isfinite(a) and a >= 0.0


def _fake_history():
    files = {
        "f1": {"robot_id": "r", "start_time": 0.0, "end_time": 100.0},
        "f2": {"robot_id": "r", "start_time": 100.0, "end_time": 200.0},
        "f3": {"robot_id": "r", "start_time": 200.0, "end_time": 300.0},
    }
    samples = [SimpleNamespace(file_id=fid) for fid in ("f1", "f2", "f3")]
    sched = {"r": [{"start_time": 0.0, "end_time": 100.0, "duration": 100.0},
                   {"start_time": 100.0, "end_time": 200.0, "duration": 100.0},
                   {"start_time": 200.0, "end_time": 300.0, "duration": 100.0}]}
    return {"samples": samples, "files": files, "by_robot": {"r": samples},
            "maint": {}, "sched": sched}

def test_trailing_excludes_current_file():
    from sprint12_task12_warning import featurize

    h = _fake_history()
    scores = {"f1": 1.0, "f2": 5.0, "f3": 1.0}
    feats = {f["file_id"]: f for f in featurize(h, scores, q90=4.0)}
    assert feats["f1"]["trail_n"] == 0  # nothing strictly prior
    assert feats["f2"]["trail_n"] == 1
    assert feats["f3"]["persist"] == 1  # f2 above q90, then f1 below stops the run
    assert feats["f2"]["trail_max"] == 1.0  # only f1 visible, not self
def test_history_resets_at_maintenance():
    from sprint12_task12_warning import featurize

    h = _fake_history()
    h["maint"] = {"r": [(150.0, 160.0)]}
    scores = {"f1": 1.0, "f2": 5.0, "f3": 1.0}
    feats = {f["file_id"]: f for f in featurize(h, scores, q90=4.0)}
    # f3 starts at 200, last maintenance ends at 160: only the (200,300)
    # event counts (100 s); earlier events are cut by the reset.
    assert abs(feats["f3"]["usage_h"] - 100.0 / 3600.0) < 1e-9
    assert feats["f3"]["trail_n"] == 1  # only f2 (end 200) visible
    assert feats["f3"]["trail_max"] == 5.0

def test_features_ignore_future_targets_shuffle():
    from sprint12_task12_warning import featurize

    h = _fake_history()
    scores = {"f1": 1.0, "f2": 5.0, "f3": 1.0}
    before = featurize(h, scores, q90=4.0)
    # attach/shuffle future targets: features must not move (leak test)
    for s, flag in zip(h["samples"], (True, False, True)):
        s.future_targets = SimpleNamespace(failure_within_7d=flag, is_censored=False)
    after = featurize(h, scores, q90=4.0)
    assert len(before) == len(after) == 3
    for b, a in zip(before, after):
        assert b == a

def test_operating_points_constant_scores_yield_no_recall():
    from sprint12_task12_warning import operating_points, recall_at_fpr

    sc = np.array([0.75, 0.75, 0.75, 0.75])
    lb = np.array([1.0, 1.0, 0.0, 0.0])
    pts = operating_points(sc, lb)
    assert pts[0] == (float("inf"), 0.0, 0.0)
    assert len(pts) == 2  # tied block moves as one unit
    assert recall_at_fpr(pts, 0.05) == 0.0
    assert recall_at_fpr(pts, 0.10) == 0.0


def test_operating_points_groups_ties_deterministically():
    from sprint12_task12_warning import operating_points, recall_at_fpr

    sc = np.array([0.9, 0.9, 0.1, 0.1])
    lb = np.array([1.0, 0.0, 1.0, 0.0])
    pts = operating_points(sc, lb)
    # thresholds: inf -> (0,0); 0.9 -> flags first block (1 pos,1 neg): fpr 0.5
    assert pts[1][1] == 0.5 and pts[1][2] == 0.5
    assert recall_at_fpr(pts, 0.10) == 0.0
    assert recall_at_fpr(pts, 0.50) == 0.5


def test_threshold_for_fpr_prefers_selective_threshold():
    from sprint12_task12_warning import operating_points, threshold_for_fpr

    sc = np.array([0.9, 0.5, 0.1, 0.0])
    lb = np.array([1.0, 0.0, 1.0, 0.0])
    pts = operating_points(sc, lb)
    assert threshold_for_fpr(pts, 0.10) == 0.9


def test_group_alert_episodes_and_event_recall():
    from sprint12_task12_warning import event_recall_lead, group_alert_episodes

    eps = group_alert_episodes([1.0, 1.5, 10.0], 5.0)
    assert eps == [[1.0, 1.5], [10.0]]
    assert group_alert_episodes([], 100.0) == []
    out = event_recall_lead(
        {"r": [1.0, 1.5, 10.0]},
        [("r", 2.0 * 86400.0), ("r", 50.0 * 86400.0)], 7 * 86400.0)
    assert out["n_events"] == 2 and out["n_recalled"] == 1
    assert out["recall"] == 0.5
    assert abs(out["lead_time_d_median"] - (2.0 * 86400.0 - 10.0) / 86400.0) < 1e-9
