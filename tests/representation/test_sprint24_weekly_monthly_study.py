"""Behavioral tests for the weekly/monthly study contract and decision rule.

These tests protect the invariants the frozen study depends on: label-blind
manifest identities that are invariant to annotation edits, causal reference
custody and period contiguity (fail closed), a complete declared cell matrix with
explicit unsupported states, and the PROMISING / NOT-PROMISING / INCONCLUSIVE
boundaries of the decision rule -- including that a negative answer is never
declared from underpowered data, that an unverified sensitivity blocks a promise,
and that no disposition ever claims specificity.

The corpus fixture is small and synthetic: no accepted data is copied.
"""

from __future__ import annotations

import copy
import math

import pytest

from representation import sprint24_weekly_monthly_study as study

DAY = 86_400.0
ANCHOR = 1_704_067_200.0  # 2024-01-01T00:00:00Z, a Monday
C1_DAYS = 60.0
SPAN_DAYS = 240.0
HISTORY = "SYNTHETIC-WM-01"
ROBOTS = ("robot-01", "robot-02", "robot-03")
PROGRAMS = {"robot-01": "program-01", "robot-02": "program-02", "robot-03": "program-01"}
EDGES = {
    HISTORY: [
        0.0,
        24.0 * DAY,
        48.0 * DAY,
        54.0 * DAY,
        C1_DAYS * DAY,
        90.0 * DAY,
        120.0 * DAY,
        150.0 * DAY,
        180.0 * DAY,
        210.0 * DAY,
        SPAN_DAYS * DAY,
    ]
}


def _calendar() -> study.horizon.CalendarConfig:
    return study.primary_calendar()


def _manifest(offset_s: float = 0.0, salt: int = 0) -> dict[str, object]:
    files: list[dict[str, object]] = []
    step = 2.0 * 3600.0
    index = 0
    time = 1.9 * 3600.0 + offset_s
    while time < SPAN_DAYS * DAY:
        robot = ROBOTS[index % len(ROBOTS)]
        files.append(
            {
                "file_id": f"S-op-{salt}-{index:06d}-{(index + salt):08x}",
                "start_time": time,
                "end_time": time + 600.0,
                "robot_id": robot,
                "program_id": PROGRAMS[robot],
                "last_reset_time": 0.0,
                "is_quarantined": False,
                "is_censored": False,
                "n_valid_patches": 40,
                "file_label": "abnormal" if (index + salt) % 7 == 3 else "normal",
            }
        )
        index += 1
        time += step
    return {
        "files": files,
        "maintenance_windows": {},
        "failure_events": [
            {
                "failure_id": f"fail-{number}",
                "robot_id": ROBOTS[number % len(ROBOTS)],
                "failure_time": (C1_DAYS + 5.0 + 6.0 * number) * DAY,
            }
            for number in range(20)
        ],
        "episodes": [
            {
                "episode_id": f"deg-{number}",
                "kind": "degradation",
                "robot_id": "robot-02",
                "start_time": (C1_DAYS + 7.0 + 9.0 * number) * DAY,
                "end_time": (C1_DAYS + 7.5 + 9.0 * number) * DAY,
            }
            for number in range(8)
        ],
    }


def _source(manifest: dict[str, object] | None = None) -> dict[str, object]:
    return study.evaluate_source(
        {HISTORY: manifest or _manifest()}, EDGES, _calendar(), modes=study.MODES
    )


def _two_history_source() -> dict[str, object]:
    second = HISTORY + "B"
    manifests = {HISTORY: _manifest(), second: _manifest(offset_s=30.0 * DAY, salt=11)}
    edges = {HISTORY: EDGES[HISTORY], second: EDGES[HISTORY]}
    return study.evaluate_source(manifests, edges, _calendar(), modes=study.MODES)


def _study_tables(
    manifest_input: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    result = _source(manifest_input)
    manifest = study.build_label_blind_manifest(result, _calendar())
    posthoc = study.build_posthoc_burden(result)
    return result, manifest, posthoc


def _histories(
    rhos: tuple[float, ...],
    residuals: tuple[float, ...],
    pairs: tuple[int, ...] = (8, 7, 6, 5),
    distinct: int = 3,
) -> list[dict[str, object]]:
    return [
        {
            "history_id": f"H{index + 1}",
            "n_pairs": pairs[index],
            "n_distinct_burden": distinct,
            "rho_burden": rhos[index],
            "rho_residual": residuals[index],
        }
        for index in range(len(rhos))
    ]


def _analysis(
    histories: list[dict[str, object]],
    sensitivity: object,
    horizon_name: str = "week",
) -> dict[str, object]:
    return {
        "horizon": horizon_name,
        "mode": study.PRIMARY_MODE,
        "histories": histories,
        "sensitivity": sensitivity,
    }


STRONG_SENSITIVITY = {
    "rolling": {"rho_burden": 0.55},
    "capped_n": {"rho_burden": 0.50},
    "alternate_anchor": {"rho_burden": 0.60},
}


def test_exact_rank_tail_matches_the_permutation_reference() -> None:
    assert study.min_attainable_tail(2) is None
    assert study.min_attainable_tail(3) == pytest.approx(2.0 / math.factorial(3))
    assert study.min_attainable_tail(5) == pytest.approx(0.0167, abs=1e-4)
    assert study.min_attainable_tail(6) == pytest.approx(2.0 / math.factorial(6))
    # Six pairs is the first count at which even a perfect association could clear
    # the declared Holm threshold; five cannot.
    assert study.min_attainable_tail(5) > study.DECISION_ALPHA / study.DECISION_FAMILY_SIZE
    assert study.min_attainable_tail(6) < study.DECISION_ALPHA / study.DECISION_FAMILY_SIZE


def test_decision_boundary_at_the_promise_floor_promises_and_below_abstains() -> None:
    at_floor = study.evaluate_decision(
        _analysis(
            _histories((0.40, 0.50, 0.50, 0.60), (0.30, 0.30, 0.30, 0.30)),
            {
                "rolling": {"rho_burden": 0.25},  # exactly half of the 0.5 median
                "capped_n": {"rho_burden": 0.30},
                "alternate_anchor": {"rho_burden": 0.50},
            },
        )
    )
    assert at_floor["disposition"] == "PROMISING"
    assert at_floor["median_rho_burden"] == pytest.approx(0.5)
    assert at_floor["median_rho_residual"] == pytest.approx(0.3)

    below_floor = study.evaluate_decision(
        _analysis(
            _histories((0.40, 0.49, 0.50, 0.60), (0.30, 0.30, 0.30, 0.30)),
            STRONG_SENSITIVITY,
        )
    )
    assert below_floor["disposition"] == "INCONCLUSIVE"
    assert below_floor["criterion_status"]["B.promise-magnitude"] == "FAIL"


def test_decision_requires_the_composition_residual_not_only_raw_movement() -> None:
    confounded = study.evaluate_decision(
        _analysis(
            _histories((0.70, 0.65, 0.60, 0.55), (0.10, 0.05, 0.12, 0.08)),
            STRONG_SENSITIVITY,
        )
    )
    assert confounded["disposition"] == "INCONCLUSIVE"
    assert confounded["criterion_status"]["B.promise-magnitude"] == "FAIL"


def test_decision_declares_nonpromise_only_with_estimable_consistent_evidence() -> None:
    decisive = study.evaluate_decision(
        _analysis(
            _histories((-0.30, -0.10, 0.05, -0.40), (-0.25, -0.05, 0.02, -0.30)),
            {
                "rolling": {"rho_burden": -0.20},
                "capped_n": {"rho_burden": -0.15},
                "alternate_anchor": {"rho_burden": -0.25},
            },
        )
    )
    assert decisive["disposition"] == "NOT-PROMISING"

    # Identical association signs but only one history reaches the pair minimum.
    underpowered = study.evaluate_decision(
        _analysis(
            _histories(
                (-0.90, -0.40, -1.00, -0.60),
                (-0.80, -0.30, -1.00, -0.50),
                pairs=(2, 5, 2, 2),
            ),
            {
                "rolling": {"rho_burden": -0.70},
                "capped_n": {"rho_burden": -0.65},
                "alternate_anchor": {"rho_burden": -0.80},
            },
        )
    )
    assert underpowered["disposition"] == "INCONCLUSIVE"
    assert underpowered["criterion_status"]["A.estimability"] == "FAIL"


def test_decision_blocks_a_promise_on_unverified_or_flipped_sensitivity() -> None:
    flipped = study.evaluate_decision(
        _analysis(
            _histories((0.70, 0.60, 0.65, 0.55), (0.45, 0.35, 0.40, 0.33)),
            {
                "rolling": {"rho_burden": -0.30},
                "capped_n": {"rho_burden": 0.45},
                "alternate_anchor": {"rho_burden": 0.55},
            },
        )
    )
    assert flipped["disposition"] == "INCONCLUSIVE"
    assert flipped["criterion_status"]["D.promise-sensitivity"] == "FAIL"

    unverified = study.evaluate_decision(
        _analysis(
            _histories((0.70, 0.60, 0.65, 0.55), (0.45, 0.35, 0.40, 0.33)),
            {
                "rolling": {"rho_burden": 0.50},
                "capped_n": {"rho_burden": None},
                "alternate_anchor": {"rho_burden": 0.58},
            },
        )
    )
    assert unverified["disposition"] == "INCONCLUSIVE"
    assert unverified["criterion_status"]["D.promise-sensitivity"] == "UNKNOWN"


def test_no_disposition_can_claim_specificity_or_release_scaling() -> None:
    examples = study.worked_decision_examples()
    dispositions = set()
    for case in examples["cases"]:
        result = case["result"]
        dispositions.add(result["disposition"])
        assert result["specificity_claim"] is False
        assert "specificity" in result["specificity_reason"]
        assert result["blocked_claims"] == list(study.BLOCKED_CLAIMS)
        if result["disposition"] == "PROMISING":
            assert result["scaling_status"] == "PROPOSAL-ONLY-SEPARATE-AUTHORIZATION-REQUIRED"
        else:
            assert result["scaling_status"] == "NOT-RELEASED"
    assert dispositions == {"PROMISING", "NOT-PROMISING", "INCONCLUSIVE"}
    assert examples["label"].startswith("ILLUSTRATIVE")


def test_worked_examples_separate_coarse_month_resolution_from_a_promise() -> None:
    examples = {
        case["case_id"]: case["result"] for case in study.worked_decision_examples()["cases"]
    }
    coarse = examples["promise-coarse-month-resolution"]
    assert coarse["disposition"] == "PROMISING"
    assert coarse["resolution"]["coarse"] is True
    assert coarse["specificity_claim"] is False
    fine = examples["promise-boundary"]
    assert fine["disposition"] == "PROMISING"
    assert fine["resolution"]["coarse"] is False


def test_label_blind_manifest_identities_are_invariant_to_annotations() -> None:
    study.assert_study_construction_label_blind()
    baseline_result, baseline_manifest, baseline_posthoc = _study_tables()

    stripped = copy.deepcopy(_manifest())
    files = stripped["files"]
    assert isinstance(files, list)
    for row in files:
        row["file_label"] = None
    stripped["failure_events"] = []
    stripped["episodes"] = []
    _, stripped_manifest, stripped_posthoc = _study_tables(stripped)
    assert stripped_manifest["digest"] == baseline_manifest["digest"]
    assert stripped_posthoc["digest"] != baseline_posthoc["digest"]
    assert all(
        row["annotation_state"] == "UNANNOTATED" for row in stripped_posthoc["periods"]
    )
    assert all(
        row["annotation_state"] != "ANNOTATED-CLEAN" for row in baseline_posthoc["periods"]
    )
    # Absence of labels must never manufacture a control.
    contrasts, counts = study.contrast_availability(stripped_posthoc)
    assert counts["n_state_annotated_clean_pairs"] == 0
    assert counts["n_state_quiet_pairs"] == 0
    assert {row["id"]: row["state"] for row in contrasts}["N3"] == (
        "UNSUPPORTED-CONTROL-ABSENT"
    )


def test_manifest_identities_are_deterministic_and_namespaces_are_disjoint() -> None:
    _, first, _ = _study_tables()
    _, second, _ = _study_tables()
    assert first["digest"] == second["digest"]
    period_ids = {str(row["period_identity"]) for row in first["periods"]}
    pair_ids = {str(row["pair_identity"]) for row in first["pairs"]}
    reference_ids = {str(row["reference_identity"]) for row in first["references"]}
    assert not period_ids & pair_ids
    assert not period_ids & reference_ids
    assert not pair_ids & reference_ids
    assert all("|PAIR|" in identity for identity in pair_ids)
    assert len(period_ids) == len(first["periods"])
    assert len(pair_ids) == len(first["pairs"])


def test_manifest_validity_rejects_a_non_causal_reference_and_a_split_period() -> None:
    result, manifest, _ = _study_tables()
    assert manifest["validity"] == {
        "reference_causal": True,
        "periods_contiguous": True,
        "identities_unique": True,
    }
    tampered = copy.deepcopy(result)
    references = tampered["references"]
    assert isinstance(references, dict)
    reference = references[HISTORY]
    assert isinstance(reference, dict)
    reference["reference_end_max"] = float(EDGES[HISTORY][4]) + DAY
    with pytest.raises(RuntimeError, match="validity failed"):
        study.build_label_blind_manifest(tampered, _calendar())


def test_coverage_matrix_declares_every_cell_exactly_once() -> None:
    _, manifest, posthoc = _study_tables()
    coverage = study.coverage_matrix(manifest, posthoc)
    histories = sorted(manifest["histories"])
    expected = (
        len(study.HORIZONS)
        * len(study.MODES)
        * len(study.ENDPOINT_HIERARCHY)
        * len(study.REPRESENTATION_SEEDS)
        * len(histories)
    )
    assert coverage["declared_cell_count"] == expected
    assert coverage["n_cells"] == len(coverage["cells"]) == expected
    keys = {
        (
            cell["horizon"],
            cell["mode"],
            cell["endpoint_id"],
            cell["representation_seed"],
            cell["history_id"],
        )
        for cell in coverage["cells"]
    }
    assert len(keys) == expected
    assert all(cell["status"] in study.CELL_STATES for cell in coverage["cells"])
    assert {cell["endpoint_id"] for cell in coverage["cells"]} == {"E1", "E2", "E3", "E4"}
    assert all(cell["decision_relevant"] == (cell["endpoint_id"] == "E1") for cell in coverage["cells"])
    assert "day" not in {cell["horizon"] for cell in coverage["cells"]}
    # Every declared contrast is published, including the unavailable controls.
    states = {row["id"]: row["state"] for row in coverage["contrasts"]}
    assert set(states) == {"C1", "C2", "C3", "C4", "N1", "N2", "N3", "N3b"}
    assert all(state in study.CONTRAST_STATES for state in states.values())


def test_unsupported_cell_states_are_published_not_dropped() -> None:
    result = _two_history_source()
    manifest = study.build_label_blind_manifest(result, _calendar())
    posthoc = study.build_posthoc_burden(result)
    histories = sorted(manifest["histories"])
    assert len(histories) == 2
    degraded = copy.deepcopy(manifest)
    references = degraded["references"]
    assert isinstance(references, list)
    for row in references:
        if row["history_id"] == histories[0]:
            row["reference_state"] = "INSUFFICIENT-REFERENCE"
    pairs = degraded["pairs"]
    assert isinstance(pairs, list)
    for row in pairs:
        if row["history_id"] == histories[1]:
            row["comparison_state"] = "PARTIAL-PERIOD"
    coverage = study.coverage_matrix(degraded, posthoc)
    # The cell count is unchanged: an unsupported cell is published, never dropped.
    assert coverage["n_cells"] == coverage["declared_cell_count"]
    by_history: dict[str, set[str]] = {}
    for cell in coverage["cells"]:
        by_history.setdefault(str(cell["history_id"]), set()).add(str(cell["status"]))
    assert by_history[histories[0]] == {"UNSUPPORTED-INSUFFICIENT-REFERENCE"}
    assert by_history[histories[1]] == {"UNSUPPORTED-NO-COMPARABLE-PAIRS"}
    assert "UNSUPPORTED-NO-COMPARABLE-PAIRS" in coverage["status_counts"]
    assert "UNSUPPORTED-INSUFFICIENT-REFERENCE" in coverage["status_counts"]
    assert set(coverage["status_counts"]) <= set(study.CELL_STATES)
    assert sum(coverage["status_counts"].values()) == coverage["declared_cell_count"]


def test_study_spec_retains_floors_defers_daily_and_resolves_every_decision() -> None:
    spec = study.study_spec()
    assert spec["horizons"] == ["week", "month"]
    assert spec["deferred_horizons"] == ["day"]
    assert spec["membership_rule"]["floors"] == {
        "min_period_files": 30,
        "min_reference_files": 200,
    }
    tiers = study.describe_tiers()
    assert set(tiers) <= {"RETAINED", "RESEARCH-DEFAULT", "DEFERRED", "OWNER-REQUIRED"}
    ids = {str(item["id"]) for item in study.decision_resolutions()}
    assert {f"D{number}" for number in range(1, 11)} <= ids
    assert {"I1", "I2", "M3-1", "M3-2", "M3-6", "M3-7"} <= ids
    policy = spec["decision"]
    assert policy["specificity_claim_allowed"] is False
    assert policy["dispositions"] == list(study.DISPOSITIONS)
    assert study.study_freeze()["study_sha256"] == study.study_sha256()


def test_cost_bounds_fail_closed_and_report_inherited_headroom() -> None:
    ok = study.assert_cost_within_declared_bounds(5_076, 3_678)
    assert ok["rows_per_seed"] == study.DECLARED_MAX_ROWS_PER_SEED
    assert ok["fresh_span_only_exceeds_inherited_headroom"] is True
    assert ok["fresh_span_only_excess_rows"] == 15_228 - study.INHERITED_HEADROOM_FORWARDS
    assert ok["forwards_executed_by_this_study"] == 0
    with pytest.raises(RuntimeError, match="exceeds the declared envelope"):
        study.assert_cost_within_declared_bounds(5_077, 3_678)
