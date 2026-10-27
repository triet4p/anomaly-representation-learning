"""Focused behavioral tests for the S24-7-M5 low-burden baseline audit.

These tests protect the invariants the audit depends on: the frozen audit
digest before any candidate count, fail-closed label-blind candidate
construction, low-nonzero burden eligibility at the primary ceiling, borderline
rejection inside the contrast gap, stability bounds of consecutive low runs,
nonoverlapping pair accounting that is distinct-period support (never an
independence claim), exposure/comparability gating, and the FEASIBLE / LIMITED
/ UNSUPPORTED boundaries. The fixture is small and synthetic: no accepted
data is copied.
"""

from __future__ import annotations

import pytest

from representation import sprint24_low_burden_baseline as baseline


def _structural(identity: str, start: float, **overrides: object) -> dict[str, object]:
    history, horizon_name, mode, key = identity.split("|")
    row: dict[str, object] = {
        "history_id": history,
        "horizon": horizon_name,
        "mode": mode,
        "period_identity": identity,
        "period_key": key,
        "period_start_rel_s": start,
        "period_end_rel_s": start + 7.0 * 86_400.0,
        "n_members": 40,
        "exposure_days": 0.28,
        "coverage_state": "FULL",
        "support_state": "SUPPORTED",
        "meets_support_floor": True,
        "n_reset_edges": 3,
        "member_ids_sha256": "x",
    }
    row.update(overrides)
    return row


def _posthoc(abnormal: int, members: int = 40, events: int = 5) -> dict[str, object]:
    return {
        "annotation_state": "ANNOTATED-ABNORMAL",
        "n_abnormal_members": abnormal,
        "abnormal_member_fraction": abnormal / members,
        "n_failure_events_covered": events,
        "n_degradation_onsets_covered": 1,
    }


def _pair(identity: str, previous: str, current: str, **overrides: object) -> dict[str, object]:
    history, horizon_name, mode, _ = identity.split("|", 3)
    row: dict[str, object] = {
        "history_id": history,
        "horizon": horizon_name,
        "mode": mode,
        "pair_identity": identity,
        "previous_period_identity": previous,
        "current_period_identity": current,
        "comparison_state": "COMPARABLE-MATCHED",
        "mix_drift_tv": 0.1,
        "exposure_ratio": 0.95,
        "n_reset_edges_crossed": 6,
    }
    row.update(overrides)
    return row


def test_audit_freeze_is_deterministic_and_documents_assumptions() -> None:
    first = baseline.audit_freeze()
    second = baseline.audit_freeze()
    assert first["audit_sha256"] == second["audit_sha256"] == baseline.audit_sha256()
    spec = first["spec"]
    assert isinstance(spec, dict)
    research = spec["research_assumptions"]  # type: ignore[index]
    assert research["low_burden_fraction_primary"] == 0.10
    assert research["elevated_fraction_primary"] == 0.20
    assert research["stability_range_primary"] == 0.05
    assert "not owner-approved" in research["assumption_status"]
    assert spec["grid"]["deferred_horizons"] == ["day"]  # type: ignore[index]


def test_candidate_construction_is_label_blind() -> None:
    baseline.assert_candidate_construction_label_blind()
    rows = [
        _structural("H|week|calendar|W1", 0.0),
        _structural("H|week|calendar|W2", 7.0 * 86_400.0),
    ]
    assert baseline.candidate_period_keys(rows, "H", "week", "calendar") == [
        "H|week|calendar|W1",
        "H|week|calendar|W2",
    ]


def test_low_nonzero_burden_is_eligible() -> None:
    row = baseline.classify_period(_structural("H|week|calendar|W1", 0.0), _posthoc(4))
    assert row["status"] == "LOW-BASELINE-CANDIDATE"
    assert row["abnormal_member_fraction"] == pytest.approx(0.10)
    assert row["ladder_low_0_10"] is True


def test_clean_zero_burden_period_is_also_low() -> None:
    row = baseline.classify_period(
        _structural("H|week|calendar|W1", 0.0),
        {
            "annotation_state": "ANNOTATED-CLEAN",
            "n_abnormal_members": 0,
            "abnormal_member_fraction": 0.0,
            "n_failure_events_covered": 0,
            "n_degradation_onsets_covered": 0,
        },
    )
    assert row["status"] == "LOW-BASELINE-CANDIDATE"


def test_contrast_gap_rejects_borderline() -> None:
    low = baseline.classify_period(_structural("H|week|calendar|W1", 0.0), _posthoc(4))
    high = baseline.classify_period(_structural("H|week|calendar|W2", 1.0), _posthoc(8))
    mid = baseline.classify_period(_structural("H|week|calendar|W3", 2.0), _posthoc(6))
    assert low["status"] == "LOW-BASELINE-CANDIDATE"
    assert high["status"] == "ELEVATED-CANDIDATE"
    assert mid["status"] == "BORDERLINE-BURDEN"


def test_support_and_coverage_gate_first() -> None:
    thin = baseline.classify_period(
        _structural("H|week|calendar|W1", 0.0, meets_support_floor=False), _posthoc(1)
    )
    assert thin["status"] == "REJECTED-INSUFFICIENT-SUPPORT"
    partial = baseline.classify_period(
        _structural("H|week|calendar|W2", 1.0, coverage_state="PARTIAL-PRE"), _posthoc(1)
    )
    assert partial["status"] == "REJECTED-PARTIAL-PERIOD"
    missing = baseline.classify_period(_structural("H|week|calendar|W3", 2.0), None)
    assert missing["status"] == "REJECTED-UNANNOTATED"


def test_stability_bounds_and_singleton() -> None:
    classified = [
        baseline.classify_period(_structural(f"H|week|calendar|W{i}", float(i)), _posthoc(count))
        for i, count in enumerate((3, 4, 6, 2))
    ]
    assert [row["status"] for row in classified] == [
        "LOW-BASELINE-CANDIDATE",
        "LOW-BASELINE-CANDIDATE",
        "BORDERLINE-BURDEN",
        "LOW-BASELINE-CANDIDATE",
    ]
    runs = baseline.stable_runs(classified)
    assert [run["status"] for run in runs] == ["STABLE-BASELINE-SEGMENT", "LOW-RUN-SINGLETON"]
    assert runs[0]["burden_fraction_range"] == pytest.approx(0.025)


def test_unstable_run_is_not_a_segment() -> None:
    periods = [
        baseline.classify_period(_structural(f"H|week|calendar|W{i}", float(i)), _posthoc(count))
        for i, count in enumerate((1, 4))
    ]
    runs = baseline.stable_runs(periods)
    assert [run["status"] for run in runs] == ["LOW-RUN-UNSTABLE"]


def test_pair_regions_follow_period_statuses_and_comparison() -> None:
    status = {
        "H|week|calendar|W1": "LOW-BASELINE-CANDIDATE",
        "H|week|calendar|W2": "LOW-BASELINE-CANDIDATE",
        "H|week|calendar|W3": "ELEVATED-CANDIDATE",
        "H|week|calendar|W4": "BORDERLINE-BURDEN",
    }
    posthoc = {
        "previous_annotation_state": "ANNOTATED-ABNORMAL",
        "current_annotation_state": "ANNOTATED-ABNORMAL",
        "n_abnormal_members": 6,
        "abnormal_member_fraction": 0.075,
        "n_failure_events_covered": 4,
        "n_degradation_onsets_covered": 0,
    }
    stable = baseline.classify_pair(
        _pair("H|week|calendar|PAIR|W2", "H|week|calendar|W1", "H|week|calendar|W2"),
        posthoc,
        status,
    )
    assert stable["region"] == "BASELINE-STABLE"
    assert stable["mix_flag"] == "MIX-OK"
    transition = baseline.classify_pair(
        _pair("H|week|calendar|PAIR|W3", "H|week|calendar|W2", "H|week|calendar|W3"),
        posthoc,
        status,
    )
    assert transition["region"] == "TRANSITION"
    mixed = baseline.classify_pair(
        _pair("H|week|calendar|PAIR|W4", "H|week|calendar|W2", "H|week|calendar|W4"),
        posthoc,
        status,
    )
    assert mixed["region"] == "MIXED-BORDERLINE"
    limited = baseline.classify_pair(
        _pair(
            "H|week|calendar|PAIR|W2",
            "H|week|calendar|W1",
            "H|week|calendar|W2",
            comparison_state="COMPARABLE-UNMATCHED-EXPOSURE",
        ),
        posthoc,
        status,
    )
    assert limited["region"] == "BASELINE-EXPOSURE-LIMITED"
    rejected = baseline.classify_pair(
        _pair(
            "H|week|calendar|PAIR|W2",
            "H|week|calendar|W1",
            "H|week|calendar|W2",
            comparison_state="INSUFFICIENT-PERIOD-SUPPORT",
        ),
        posthoc,
        status,
    )
    assert rejected["region"] == "REJECTED-NONCOMPARABLE"


def test_mix_flag_boundary() -> None:
    assert baseline.mix_flag(0.25) == "MIX-OK"
    assert baseline.mix_flag(0.2501) == "MIX-CONFOUNDED"
    assert baseline.mix_flag(None) == "MIX-UNKNOWN"


def test_nonoverlapping_selection_is_distinct_period_support() -> None:
    def region(previous: str, current: str) -> dict[str, object]:
        return {
            "pair_identity": f"H|week|calendar|PAIR|{current}",
            "previous_period_identity": f"H|week|calendar|{previous}",
            "current_period_identity": f"H|week|calendar|{current}",
        }

    chosen = baseline.select_nonoverlapping_pairs(
        [region("W1", "W2"), region("W2", "W3"), region("W3", "W4")]
    )
    assert chosen == ["H|week|calendar|PAIR|W2", "H|week|calendar|PAIR|W4"]


def test_disposition_boundaries() -> None:
    feasible_cells = [
        {"horizon": "week", "mode": "calendar", "disposition": "FEASIBLE", "stable_segments": [{"status": "STABLE-BASELINE-SEGMENT"}]}
        for _ in range(3)
    ] + [
        {"horizon": "week", "mode": "calendar", "disposition": "LIMITED", "stable_segments": []}
    ]
    overall = baseline.overall_disposition(feasible_cells)
    assert overall["week"]["disposition"] == "FEASIBLE"
    limited_cells = [
        {"horizon": "week", "mode": "calendar", "disposition": "LIMITED", "stable_segments": [{"status": "STABLE-BASELINE-SEGMENT"}]},
        {"horizon": "week", "mode": "calendar", "disposition": "LIMITED", "stable_segments": []},
        {"horizon": "week", "mode": "calendar", "disposition": "UNSUPPORTED", "stable_segments": []},
        {"horizon": "week", "mode": "calendar", "disposition": "UNSUPPORTED", "stable_segments": []},
    ]
    overall = baseline.overall_disposition(limited_cells)
    assert overall["week"]["disposition"] == "LIMITED"
    empty_cells = [
        {"horizon": "month", "mode": "calendar", "disposition": "UNSUPPORTED", "stable_segments": []}
        for _ in range(4)
    ]
    overall = baseline.overall_disposition(empty_cells)
    assert overall["month"]["disposition"] == "UNSUPPORTED"
