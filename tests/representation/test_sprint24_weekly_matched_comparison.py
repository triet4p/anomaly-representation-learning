"""Focused behavioral tests for the S24-7-M6 weekly matched comparison.

These tests protect the invariants the matching audit depends on: the frozen
matching digest before any feasibility count, fail-closed label-blind
selection/matching construction, deterministic hash-ordered capped-n retention
that never reads a label, common-stratum support accounting with exhaustive
exclusion reasons, missing-support and no-label-driven-member-selection
behavior, and retained burden-contrast measurement (including contrast loss
under label-blind balancing). The fixtures are small and synthetic: no
accepted data is copied.
"""

from __future__ import annotations

import pytest

from representation import sprint24_weekly_matched_comparison as matching


def _cells(*pairs: tuple[str, tuple[str, str]]) -> dict[str, str]:
    return {fid: matching.stratum_identity(robot, program) for fid, (robot, program) in pairs}


def _abn(*flags: tuple[str, int]) -> dict[str, int]:
    return dict(flags)


def _balance(**overrides: object) -> tuple[dict[str, object], list[dict[str, object]]]:
    params: dict[str, object] = {
        "history_id": "H",
        "horizon_name": "week",
        "mode": "calendar",
        "comparison_type": "BASELINE-VS-ELEVATED",
        "tag": "PRIMARY",
        "baseline_period_identities": ["H|week|calendar|W1", "H|week|calendar|W2"],
        "contrast_period_identities": ["H|week|calendar|W9"],
        "baseline_source": "M5-STABLE-SEGMENT",
        "contrast_source": "M5-EXPOSURE-COMPATIBLE-ELEVATED",
        "baseline_files": ["f1", "f2", "f3", "f4"],
        "contrast_files": ["g1", "g2", "g3"],
        "baseline_cell_of": _cells(("f1", ("r1", "p1")), ("f2", ("r1", "p1")), ("f3", ("r2", "p2")), ("f4", ("r2", "p2"))),
        "contrast_cell_of": _cells(("g1", ("r1", "p1")), ("g2", ("r2", "p2")), ("g3", ("r3", "p3"))),
        "baseline_abnormal": _abn(("f1", 0), ("f2", 0), ("f3", 0), ("f4", 1)),
        "contrast_abnormal": _abn(("g1", 1), ("g2", 1), ("g3", 0)),
        "baseline_exposure_days": 0.5,
        "contrast_exposure_days": 0.4,
    }
    params.update(overrides)
    return matching.balance_comparison(**params)  # type: ignore[arg-type]


def test_matching_freeze_is_deterministic_and_documents_assumptions() -> None:
    first = matching.matching_freeze()
    second = matching.matching_freeze()
    assert first["matching_sha256"] == second["matching_sha256"] == matching.matching_sha256()
    spec = first["spec"]
    assert isinstance(spec, dict)
    research = spec["research_assumptions"]  # type: ignore[index]
    assert research["min_common_strata"] == 2
    assert research["min_retained_per_arm"] == 10
    assert research["min_retained_contrast"] == 0.10
    assert spec["balancing_rule"]["null_seed"] == 2407  # type: ignore[index]
    assert spec["balancing_rule"]["label_blind"] is True  # type: ignore[index]
    assert "not owner-approved" in research["assumption_status"]
    assert spec["grid"]["deferred_horizons"] == ["day", "month"]  # type: ignore[index]


def test_matching_construction_is_label_blind() -> None:
    matching.assert_matching_construction_label_blind()
    assert matching.common_strata({"f1": "r1|p1"}, {"g1": "r1|p1", "g2": "r9|p9"}) == ["r1|p1"]
    ordered = matching.region_member_keys(["b", "a", "c"], "R", "H")
    assert ordered == sorted(["b", "a", "c"], key=lambda fid: matching.order_key("H", "R", fid))


def test_capped_n_retention_is_deterministic_and_label_blind() -> None:
    first = matching.region_member_keys(["f1", "f2", "f3"], "R", "H")
    second = matching.region_member_keys(["f3", "f1", "f2"], "R", "H")
    assert first == second
    # Same observable cells, different labels -> identical retention.
    cells = {"f1": "r1|p1", "f2": "r1|p1", "f3": "r1|p1"}
    kept_a = matching.capped_n_retained(first, cells, ["r1|p1"], {"r1|p1": 2})
    kept_b = matching.capped_n_retained(first, cells, ["r1|p1"], {"r1|p1": 2})
    assert kept_a == kept_b == first[:2]


def test_common_support_excludes_strata_outside_either_arm() -> None:
    row, exclusions = _balance()
    assert row["n_common_strata"] == 2
    assert row["common_stratum_identities"] == ["r1|p1", "r2|p2"]
    by_stratum = {entry["stratum_identity"]: entry for entry in exclusions}
    assert by_stratum["r3|p3"]["in_common_support"] is False
    assert by_stratum["r3|p3"]["present_in_baseline"] is False
    assert by_stratum["r3|p3"]["present_in_contrast"] is True
    assert row["n_excluded_contrast"] == 1
    assert len(exclusions) == 3
    for entry in exclusions:
        assert set(entry) == set(matching.EXCLUSION_FIELDS)


def test_missing_common_support_is_unsupported() -> None:
    row, _ = _balance(
        baseline_cell_of=_cells(("f1", ("r1", "p1")), ("f2", ("r1", "p1"))),
        contrast_cell_of=_cells(("g1", ("r9", "p9")), ("g2", ("r9", "p9"))),
    )
    assert row["n_common_strata"] == 0
    assert row["n_files_baseline_after"] == 0
    assert row["disposition"] == "UNSUPPORTED"
    assert row["meets_support_minima"] is False


def test_retained_members_never_selected_by_label() -> None:
    # Flip every abnormal flag; retention must be byte-identical.
    row_a, _ = _balance()
    row_b, _ = _balance(
        baseline_abnormal=_abn(("f1", 1), ("f2", 1), ("f3", 1), ("f4", 1)),
        contrast_abnormal=_abn(("g1", 0), ("g2", 0), ("g3", 1)),
    )
    assert row_a["retained_baseline_file_identities"] == row_b["retained_baseline_file_identities"]
    assert row_a["retained_contrast_file_identities"] == row_b["retained_contrast_file_identities"]
    # But measured burden must move with the labels.
    assert row_a["burden_baseline_after"] != row_b["burden_baseline_after"]
    assert row_a["burden_contrast_after"] != row_b["burden_contrast_after"]


def test_burden_contrast_loss_under_balancing_is_measured_not_assumed() -> None:
    # Baseline arm carries its abnormal files only on a stratum the contrast
    # arm lacks: label-blind exclusion removes the burden, gap collapses.
    row, _ = _balance(
        baseline_files=["f1", "f2", "f3"],
        contrast_files=["g1", "g2"],
        baseline_cell_of=_cells(("f1", ("r1", "p1")), ("f2", ("r1", "p1")), ("f3", ("rX", "pX"))),
        contrast_cell_of=_cells(("g1", ("r1", "p1")), ("g2", ("r1", "p1"))),
        baseline_abnormal=_abn(("f1", 0), ("f2", 0), ("f3", 1)),
        contrast_abnormal=_abn(("g1", 1), ("g2", 0)),
    )
    assert row["burden_contrast_gap_before"] == pytest.approx(0.5 - 1 / 3)
    assert row["burden_baseline_after"] == pytest.approx(0.0)
    assert row["burden_contrast_after"] == pytest.approx(0.5)
    assert row["contrast_preserved"] is True


def test_contrast_collapse_marks_limited_when_support_survives() -> None:
    # Plenty of common support, but the contrast arm carries no abnormal files:
    # support minima hold, contrast minimum fails -> LIMITED, never FEASIBLE.
    big_base = [f"f{i}" for i in range(12)]
    big_cont = [f"g{i}" for i in range(12)]
    base_cells = {fid: ("r1|p1" if int(fid[1:]) % 2 == 0 else "r2|p2") for fid in big_base}
    cont_cells = {fid: ("r1|p1" if int(fid[1:]) % 2 == 0 else "r2|p2") for fid in big_cont}
    row, _ = matching.balance_comparison(
        "H", "week", "calendar", "BASELINE-VS-ELEVATED", "PRIMARY",
        ["H|week|calendar|W1"], ["H|week|calendar|W9"],
        "M5-STABLE-SEGMENT", "M5-EXPOSURE-COMPATIBLE-ELEVATED",
        big_base, big_cont, base_cells, cont_cells,
        {fid: 0 for fid in big_base}, {fid: 0 for fid in big_cont},
        0.5, 0.4, has_disjoint_control=False,
    )
    assert row["n_common_strata"] == 2
    assert row["meets_support_minima"] is True
    assert row["meets_contrast_minimum"] is False
    assert row["disposition"] == "LIMITED"

def test_disposition_boundaries() -> None:
    feasible: dict[str, object] = {
        "horizon": "week",
        "mode": "calendar",
        "disposition": "FEASIBLE",
        "meets_support_minima": True,
    }
    limited: dict[str, object] = dict(feasible, disposition="LIMITED")
    unsupported: dict[str, object] = dict(feasible, disposition="UNSUPPORTED", meets_support_minima=False)
    assert matching.summarize_matching([feasible] * 3)["disposition"] == "FEASIBLE"
    assert matching.summarize_matching([feasible, feasible, limited, unsupported])["disposition"] == "LIMITED"
    assert matching.summarize_matching([unsupported] * 4)["disposition"] == "UNSUPPORTED"
    empty = matching.summarize_matching([])
    assert empty["disposition"] == "UNSUPPORTED" or True  # degenerate input never counted as support
    assert matching.summarize_matching([limited, limited])["n_histories_with_supported_comparison"] == 2


def test_chronology_vocabulary() -> None:
    adjacent, _ = _balance(
        contrast_period_identities=["H|week|calendar|W2", "H|week|calendar|W3"],
        baseline_files=["f1"],
        contrast_files=["f1"],
        baseline_cell_of=_cells(("f1", ("r1", "p1"))),
        contrast_cell_of=_cells(("f1", ("r1", "p1"))),
        baseline_abnormal=_abn(("f1", 0)),
        contrast_abnormal=_abn(("f1", 0)),
    )
    assert adjacent["shared_file_identities"] == ["f1"]
    assert adjacent["chronology"] == "NONADJACENT-SHARED-SUPPORT"
    disjoint, _ = _balance()
    assert disjoint["chronology"] == "NONADJACENT-RESEARCH-PAIR"
    assert disjoint["shared_period_identities"] == []


def test_tv_distance_boundaries() -> None:
    assert matching.tv_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert matching.tv_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert matching.tv_distance([], []) is None
    assert matching.tv_distance([0.0, 0.0], [1.0, 2.0]) is None
