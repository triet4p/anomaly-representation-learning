"""Focused Sprint 13 generator and metric-contract checks (Tasks 8-10).

Exercises the real public entry point with tiny histories plus a bounded
medium history, and unit-tests the frozen event-window reference
implementation on fixtures. No protocol seeds are used here (910+ only).
"""

from __future__ import annotations

import json

import pytest

from synth import cli as synth_cli
from synth import events as events_mod
from synth.chronicle import (
    load_chronological,
    materialize_chronological,
    sprint13_history_config,
    verify_seal,
    write_seal,
)
from synth.config import CohortConfig, HealthConfig
from synth.schema import (
    DIAGNOSTIC_ONLY_FIELD_NAMES,
    MODEL_INPUT_FIELD_NAMES,
    FailureEvent,
)

DAY = 86400.0


def _row(file_id, robot, program, start, end, quarantined=False,
         censored=False, reason=None, views=("test-temporal",)):
    return {
        "file_id": file_id,
        "robot_id": robot,
        "program_id": program,
        "start_time": start,
        "end_time": end,
        "is_quarantined": quarantined,
        "quarantine_reason": reason,
        "is_censored": censored,
        "member_views": list(views),
    }


def _failure(robot, day, cohort="P"):
    return {
        "failure_id": f"fail-{robot}-{day}",
        "robot_id": robot,
        "failure_time": day * DAY,
        "cohort": cohort,
        "subtype": "A1" if cohort == "A" else None,
        "degradation_onset": None if cohort == "A" else (day - 10) * DAY,
        "duration_d": 0.0 if cohort == "A" else 10.0,
        "severity": 2.0,
    }


def test_sprint13_profile_freezes_v4_operating_point():
    cfg = sprint13_history_config(seed=301)
    assert cfg.factory.span_days == 180.0
    assert cfg.factory.dev_cutoff_days == 120.0
    assert cfg.factory.quarantine_days == 7.0
    assert cfg.fleet.n_robots == 8
    assert cfg.health.preventive_interval_s == 30.0 * DAY
    assert cfg.health.preventive_duration_s == DAY
    assert cfg.health.maintenance_duration_s == 2.0 * DAY
    cohorts = {c.cohort_id: c for c in cfg.health.cohorts}
    assert set(cohorts) == {"P", "W", "A"}
    assert [cohorts[c].share for c in "PWA"] == pytest.approx([0.45, 0.30, 0.25])
    assert (cohorts["P"].degradation_min_d, cohorts["P"].degradation_max_d) == (7.0, 14.0)
    assert (cohorts["W"].degradation_min_d, cohorts["W"].degradation_max_d) == (14.0, 28.0)
    assert (cohorts["A"].degradation_min_d, cohorts["A"].degradation_max_d) == (0.0, 0.0)
    assert cohorts["A"].subtypes == ("A1", "A2")
    assert cohorts["W"].amplitude_scale == pytest.approx(0.3)
    for stream in ("factory", "scheduler", "health", "signal", "temporal"):
        assert getattr(cfg, stream).seed == 301


def test_cohort_config_rejects_bad_mix():
    with pytest.raises(ValueError):
        HealthConfig(cohorts=(
            CohortConfig(cohort_id="P", share=0.5, base_rate=1e-6),
            CohortConfig(cohort_id="W", share=0.5, base_rate=1e-6),
        ))
    with pytest.raises(ValueError):
        CohortConfig(cohort_id="A", share=1.0, base_rate=1e-6,
                     abrupt_rate=1e-6)
    with pytest.raises(ValueError):
        CohortConfig(cohort_id="P", share=1.0, base_rate=1e-6,
                     degradation_min_d=14.0, degradation_max_d=7.0)


def test_failure_event_validation():
    good = FailureEvent(
        failure_id="fail-r1-0001", robot_id="robot-01",
        failure_time=10.0 * DAY, cohort="P", degradation_onset=2.0 * DAY,
        duration_d=8.0, severity=2.0)
    assert good.cohort == "P"
    with pytest.raises(ValueError):
        FailureEvent(
            failure_id="x", robot_id="r", failure_time=10.0, cohort="A",
            subtype="A1", degradation_onset=5.0, duration_d=1.0, severity=1.0)
    with pytest.raises(ValueError):
        FailureEvent(
            failure_id="x", robot_id="r", failure_time=10.0, cohort="A",
            subtype="A3", severity=1.0)
    with pytest.raises(ValueError):
        FailureEvent(
            failure_id="x", robot_id="r", failure_time=10.0, cohort="P",
            degradation_onset=11.0 * DAY, duration_d=1.0, severity=1.0)


def test_pos_eligibility_ignores_quarantine_but_not_maintenance():
    failure = _failure("robot-01", 50.0)
    windows: dict = {}
    inside = _row("f1", "robot-01", "program-01", 49.0 * DAY, 49.5 * DAY,
                  quarantined=True, reason="precursor_horizon")
    assert events_mod.pos_files([inside], failure, windows) == [inside]
    cen = _row("f2", "robot-01", "program-01", 49.0 * DAY, 49.5 * DAY,
               censored=True)
    assert events_mod.pos_files([cen], failure, windows) == []
    other = _row("f3", "robot-02", "program-01", 49.0 * DAY, 49.5 * DAY)
    assert events_mod.pos_files([other], failure, windows) == []
    maint = {"robot-01": [[49.0 * DAY, 50.0 * DAY]]}
    assert events_mod.pos_files([inside], failure, maint) == []


def test_anchor_eligibility_excludes_quarantine_and_maintenance():
    failure = _failure("robot-01", 50.0)
    rows = [
        _row("ok", "robot-01", "program-01", 20.0 * DAY, 20.5 * DAY),
        _row("q", "robot-01", "program-01", 21.0 * DAY, 21.5 * DAY,
             quarantined=True, reason="precursor_horizon"),
    ]
    assert [r["file_id"] for r in events_mod.anchor_rows(rows, {})] == ["ok"]
    maint = {"robot-01": [[20.0 * DAY, 21.0 * DAY]]}
    assert events_mod.anchor_rows(rows, maint) == []


def test_control_selection_is_deterministic_with_clearance():
    rows = [_row(f"f{i}", "robot-01", "program-01",
                 (i * 10) * DAY, (i * 10 + 1) * DAY) for i in range(6)]
    failures = [_failure("robot-01", 25.0)]
    first = events_mod.select_control_windows(rows, failures)
    second = events_mod.select_control_windows(list(reversed(rows)), failures)
    assert [(w["anchor_end"], w["robot_id"]) for w in first] == [
        (w["anchor_end"], w["robot_id"]) for w in second]
    for window in first:
        end = window["anchor_end"]
        assert not any(
            end - 7 * DAY <= f["failure_time"] < end + 7 * DAY
            for f in failures)
    ends = [w["anchor_end"] for w in first
            if w["robot_id"] == "robot-01"]
    assert all(b - a >= 7 * DAY for a, b in zip(ends, ends[1:]))
    for window in first:
        members = [m["end_time"] for m in window["members"]]
        assert all(window["anchor_end"] - 7 * DAY <= e <= window["anchor_end"]
                   for e in members)


def test_roc_auc_ties_and_constant_and_empty():
    assert events_mod.roc_auc_tie_aware([0.9, 0.8], [0.1, 0.2]) == pytest.approx(1.0)
    assert events_mod.roc_auc_tie_aware([0.5] * 5, [0.5] * 4) == pytest.approx(0.5)
    assert events_mod.roc_auc_tie_aware([0.7, 0.7], [0.7, 0.2]) == pytest.approx(0.75)
    with pytest.raises(events_mod.UnavailableError):
        events_mod.roc_auc_tie_aware([0.9], [])


def test_lead_zero_at_onset_and_clopper_pearson():
    assert events_mod.lead_days(50.0 * DAY, [50.0 * DAY]) == pytest.approx(0.0)
    assert events_mod.lead_days(50.0 * DAY, [47.0 * DAY, 49.0 * DAY]) == pytest.approx(3.0)
    lower, upper = events_mod.clopper_pearson(5, 10)
    assert (lower, upper) == pytest.approx((0.1871, 0.8129), abs=1e-3)
    assert events_mod.clopper_pearson(0, 10)[0] == pytest.approx(0.0)
    assert events_mod.clopper_pearson(10, 10)[1] == pytest.approx(1.0)


def test_false_alert_grouping_and_windowing():
    failures = [_failure("robot-01", 50.0)]
    flagged = {"robot-01": [10.0 * DAY, 10.5 * DAY, 49.0 * DAY]}
    false, rate = events_mod.false_alert_episodes(flagged, failures, 100.0)
    assert false == 1
    assert rate == pytest.approx(0.01)
    with pytest.raises(events_mod.UnavailableError):
        events_mod.false_alert_episodes(flagged, failures, 0.0)


def test_ledger_validation_rejects_bad_records():
    manifest = {"failure_events": [dict(_failure("robot-01", 50.0), cohort="Z")]}
    with pytest.raises(ValueError):
        events_mod.failure_ledger(manifest)
def test_cli_sprint13_tiny_round_trip_and_role(tmp_path):
    root = tmp_path / "tiny"
    assert synth_cli.main([
        "--chronological", "--profile", "sprint13", "--units", "24",
        "--seed", "911", "--role", "SMOKE-VAL", "--output", str(root),
    ]) == 0
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["role"] == "SMOKE-VAL"
    assert manifest["protocol"] == "sprint13-protocol-v4"
    assert set(manifest) >= {"failure_events", "maintenance_windows", "files",
                             "seeds", "config_hash"}
    rows = manifest["files"]
    assert all(set(r) >= {"quarantine_reason", "is_censored", "last_reset_time"}
               for r in rows)
    for row in rows:
        resets = [end for starts in manifest["maintenance_windows"].get(
            row["robot_id"], []) for _, end in [starts]]
        expected = max([0.0] + [e for e in resets if e <= row["start_time"]])
        assert row["last_reset_time"] == pytest.approx(expected)
    first, _ = load_chronological(root)
    second, _ = load_chronological(root)
    assert [s.file_id for s in first] == [s.file_id for s in second]
    assert all(a.x.tolist() == b.x.tolist()
               for a, b in zip(first, second))


def test_encoder_inputs_exclude_event_metadata(tmp_path):
    root = tmp_path / "tiny"
    synth_cli.main([
        "--chronological", "--profile", "sprint13", "--units", "24",
        "--seed", "912", "--output", str(root),
    ])
    samples, _ = load_chronological(root)
    assert samples
    for sample in samples:
        inputs = sample.encoder_inputs()
        assert set(inputs) <= MODEL_INPUT_FIELD_NAMES
        assert not (set(inputs) & DIAGNOSTIC_ONLY_FIELD_NAMES)
        assert "cohort" not in sample.__dict__
        assert "failure_events" not in sample.__dict__


def test_seal_roundtrip_and_tamper(tmp_path):
    root = tmp_path / "tiny"
    synth_cli.main([
        "--chronological", "--profile", "sprint13", "--units", "24",
        "--seed", "913", "--output", str(root),
    ])
    seal = write_seal(root, role="SMOKE-SEAL")
    assert seal["role"] == "SMOKE-SEAL"
    assert verify_seal(root)["manifest_sha256"] == seal["manifest_sha256"]
    manifest_path = root / "manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(text.replace('"role": null', '"role": "X"'),
                             encoding="utf-8")
    with pytest.raises(ValueError):
        verify_seal(root)


def test_medium_history_cohorts_bounds_and_abrupt_null(tmp_path):
    root = tmp_path / "medium"
    assert synth_cli.main([
        "--chronological", "--profile", "sprint13", "--units", "150",
        "--seed", "914", "--output", str(root),
    ]) == 0
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ledger = events_mod.failure_ledger(manifest)
    cohorts = {r["cohort"] for r in ledger}
    assert cohorts == {"P", "W", "A"}
    for record in ledger:
        if record["cohort"] == "P":
            assert 7.0 <= record["duration_d"] <= 14.0
        elif record["cohort"] == "W":
            assert 14.0 <= record["duration_d"] <= 28.0
        else:
            assert record["duration_d"] == 0.0
            assert record["degradation_onset"] is None
            assert record["subtype"] in ("A1", "A2")
    failure_ids = {e["episode_id"] for e in manifest["episodes"]
                   if e["kind"] == "failure"}
    assert {r["failure_id"] for r in ledger} == failure_ids
    samples, _ = load_chronological(root)
    by_id = {s.file_id: s for s in samples}
    for record in ledger:
        if record["cohort"] != "A":
            continue
        window = [r for r in manifest["files"]
                  if r["robot_id"] == record["robot_id"]
                  and record["failure_time"] - 7 * DAY <= r["end_time"]
                  < record["failure_time"]]
        for row in window:
            sample = by_id[row["file_id"]]
            if sample.anomaly_meta is not None:
                assert sample.anomaly_meta.extra.get("temporal_policy") != "precursor"
    pmaint = [e for e in manifest["episodes"]
              if e["episode_id"].startswith("pmaint-")]
    assert pmaint
    assert all(e["end_time"] > e["start_time"] for e in pmaint)
