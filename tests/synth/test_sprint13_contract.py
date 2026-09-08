"""Focused Sprint 13 generator and metric-contract checks (Tasks 8-10).

Exercises the real public entry point with tiny histories plus a bounded
medium history, and unit-tests the frozen event-window reference
implementation on fixtures. No protocol seeds are used here (910+ only).
"""

from __future__ import annotations

from statistics import median
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
    """v4 retired profile keeps structural shape (bounds now live in v4.1)."""
    root = tmp_path / "medium"
    assert synth_cli.main([
        "--chronological", "--profile", "sprint13", "--units", "150",
        "--seed", "914", "--output", str(root),
    ]) == 0
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ledger = events_mod.failure_ledger(manifest)
    assert {r["cohort"] for r in ledger} == {"P", "W", "A"}
    failure_ids = {e["episode_id"] for e in manifest["episodes"]
                   if e["kind"] == "failure"}
    assert {r["failure_id"] for r in ledger} == failure_ids
    pmaint = [e for e in manifest["episodes"]
              if e["episode_id"].startswith("pmaint-")]
    assert pmaint
    assert all(e["end_time"] > e["start_time"] for e in pmaint)


def _precursor_severities(manifest, samples_by_id, cohort):
    values = []
    for record in events_mod.failure_ledger(manifest):
        if record["cohort"] != cohort or record["degradation_onset"] is None:
            continue
        for row in manifest["files"]:
            if row["robot_id"] != record["robot_id"]:
                continue
            if not record["degradation_onset"] <= row["end_time"] <= record["failure_time"]:
                continue
            sample = samples_by_id[row["file_id"]]
            meta = sample.anomaly_meta
            if meta is not None and meta.extra.get("temporal_policy") == "precursor":
                values.append(float(meta.severity))
    return values


def test_v41_causal_dynamics_and_manifestation(tmp_path):
    """v4.1: causal onset linkage, P/W ordering, severity ordering from
    observable injection records, waveform-level A-null. (Day-bound
    medians are verified at full scale in v4.1 §2 evidence; medium density
    cannot test day bounds.)"""

    root = tmp_path / "v41medium"
    assert synth_cli.main([
        "--chronological", "--profile", "sprint13-v41", "--units", "150",
        "--seed", "931", "--protocol", "sprint13-protocol-v4.1",
        "--output", str(root),
    ]) == 0
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "sprint13-protocol-v4.1"
    ledger = events_mod.failure_ledger(manifest)
    assert {r["cohort"] for r in ledger} == {"P", "W", "A"}
    episodes = {e["episode_id"]: e for e in manifest["episodes"]}
    p_durs, w_durs = [], []
    for record in ledger:
        if record["cohort"] == "A":
            assert record["duration_d"] == 0.0
            assert record["degradation_onset"] is None
            assert record["subtype"] in ("A1", "A2")
            continue
        linked = episodes[record["degradation_episode_id"]]
        assert record["degradation_onset"] == linked["start_time"]
        (p_durs if record["cohort"] == "P" else w_durs).append(
            record["duration_d"])
    assert p_durs and w_durs
    assert median(p_durs) < median(w_durs)
    samples, _ = load_chronological(root)
    by_id = {s.file_id: s for s in samples}
    p_sev = _precursor_severities(manifest, by_id, "P")
    w_sev = _precursor_severities(manifest, by_id, "W")
    assert p_sev and w_sev
    assert median(p_sev) > median(w_sev)
    pw_windows = [(r["robot_id"], r["degradation_onset"], r["failure_time"])
                  for r in ledger if r["cohort"] in ("P", "W")]
    for record in ledger:
        if record["cohort"] != "A":
            continue
        for row in manifest["files"]:
            if row["robot_id"] != record["robot_id"]:
                continue
            if not record["failure_time"] - 7 * DAY <= row["end_time"] < record["failure_time"]:
                continue
            sample = by_id[row["file_id"]]
            meta = sample.anomaly_meta
            if meta is None:
                continue
            policy = meta.extra.get("temporal_policy")
            assert policy != "precursor" or any(
                rb == row["robot_id"] and o <= row["end_time"] <= t
                for rb, o, t in pw_windows), row["file_id"]
    for row in manifest["files"]:
        assert row["n_valid_patches"] >= 1
    from synth.patchify import Patchifier
    from synth.config import PatchConfig
    checker = Patchifier(PatchConfig())
    probe = by_id[manifest["files"][0]["file_id"]]
    assert manifest["files"][0]["n_valid_patches"] == int(
        checker.patchify(probe).patches.shape[0])


def test_v41_profile_freezes_amended_operating_point():
    from synth.chronicle import V41_SEEDS, sprint13_v41_history_config

    assert V41_SEEDS == (500, 501, 502, 503, 504, 505, 506, 507, 508,
                         600, 601, 602, 603)
    cfg = sprint13_v41_history_config(seed=500)
    assert cfg.factory.span_days == 180.0
    assert cfg.fleet.n_robots == 8
    cohorts = {c.cohort_id: c for c in cfg.health.cohorts}
    assert cohorts["P"].wear_rate == 2.0e-4
    assert cfg.health.upcoming_p == 0.55
    assert cohorts["W"].wear_rate == 5.0e-5
    assert cohorts["P"].failure_threshold_h == 2.0
    assert cohorts["W"].failure_threshold_h == 1.35
    assert cohorts["A"].wear_rate == 0.0


def test_eligible_operational_row_predicate():
    windows = {"robot-01": [[10.0 * DAY, 11.0 * DAY]]}
    ok = _row("a", "robot-01", "program-01", 20.0 * DAY, 20.5 * DAY)
    assert events_mod.eligible_operational_row(ok, windows)
    censored = _row("b", "robot-01", "program-01", 20.0 * DAY, 20.5 * DAY,
                    censored=True)
    assert not events_mod.eligible_operational_row(censored, windows)
    spanning = _row("c", "robot-01", "program-01", 9.5 * DAY, 10.5 * DAY)
    assert not events_mod.eligible_operational_row(spanning, windows)
    inside = _row("d", "robot-01", "program-01", 10.2 * DAY, 10.8 * DAY)
    assert not events_mod.eligible_operational_row(inside, windows)
    quarantined_ok = _row("e", "robot-01", "program-01", 20.0 * DAY, 20.5 * DAY,
                          quarantined=True, reason="precursor_horizon")
    assert events_mod.eligible_operational_row(quarantined_ok, windows)


def test_holdout_exclusion_for_fit_cal_pools():
    rows = [
        _row("a", "robot-01", "program-01", 1.0 * DAY, 1.5 * DAY),
        _row("b", "robot-01", "program-03", 2.0 * DAY, 2.5 * DAY),
        _row("c", "robot-08", "program-01", 3.0 * DAY, 3.5 * DAY),
    ]
    eligible = [r for r in rows
                if r["program_id"] != "program-03"
                and r["robot_id"] != "robot-08"]
    assert [r["file_id"] for r in eligible] == ["a"]
