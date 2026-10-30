"""Focused checks for the Sprint 24 Task 7-M14 representation-endpoint audit.

These protect consumer-visible behaviours: capped apportionment, exact
period/stratum quotas, disjoint duplicate-free arm draws, the accepted E1 and
exact-quanta E2 estimators, the bounded within-prototype diagnostic
(including tangential movement that radial distance alone would miss),
reference-derived perturbation and boundary-safe behaviour, the metadata-only
census accounting, and the frozen-specification refusal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from representation import sprint24_representation_endpoint_diagnostic as audit
from representation import sprint24_weekly_monthly_endpoints as endpoints


# --- synthetic fixtures ---------------------------------------------------------


def _cell_table(
    *,
    normal_per_stratum: tuple[int, ...] = (40, 40),
    abnormal_per_stratum: tuple[int, ...] = (8, 8),
    periods: int = 2,
    stratum_names: tuple[str, ...] = ("R01|P100", "R02|P100"),
) -> audit.CellTable:
    """Deterministic synthetic cell table with known stratum/period counts."""
    ids: list[str] = []
    entries: list[dict[str, object]] = []
    strata: list[int] = []
    abnormal: list[bool] = []
    period_codes: list[int] = []
    for index, name in enumerate(stratum_names):
        for position in range(normal_per_stratum[index]):
            fid = f"{name}|normal|{position:03d}"
            ids.append(fid)
            entries.append({"file_id": fid, "file_label": "normal", "annotation": {}})
            strata.append(index)
            abnormal.append(False)
            period_codes.append(position % periods)
        for position in range(abnormal_per_stratum[index]):
            fid = f"{name}|abnormal|{position:03d}"
            ids.append(fid)
            entries.append(
                {
                    "file_id": fid,
                    "file_label": "abnormal",
                    "annotation": {
                        "family": "alpha" if position % 2 == 0 else "beta",
                        "severity": 0.5,
                        "abnormal_step_fraction": 0.2,
                        "duration_steps": position + 1,
                    },
                }
            )
            strata.append(index)
            abnormal.append(True)
            period_codes.append(position % periods)
    return audit.CellTable(
        key="H|week",
        history_id="H",
        horizon="week",
        periods=tuple(range(1, periods + 1)),
        entries=entries,
        ids=ids,
        strata=np.asarray(strata, dtype=np.int64),
        stratum_names=list(stratum_names),
        abnormal=np.asarray(abnormal, dtype=bool),
        period_codes=np.asarray(period_codes, dtype=np.int64),
        period_numbers=list(range(1, periods + 1)),
    )


def _manifest_and_audit(
    *, normal_per_stratum: tuple[int, ...] = (100, 100), abnormal_per_stratum: tuple[int, ...] = (10, 10)
) -> tuple[dict[str, object], dict[str, object]]:
    files: list[dict[str, object]] = []
    stratum_names = ["R01|P100", "R02|P100"]
    for index, name in enumerate(stratum_names):
        robot, program = name.split("|")
        for position in range(normal_per_stratum[index]):
            files.append(
                {
                    "file_id": f"n{index}-{position:04d}",
                    "history_id": "H",
                    "arm": "OBSERVED",
                    "file_label": "normal",
                    "week_offset": 1 + position % 2,
                    "month_index": 1 + position % 2,
                    "predicted_robot_idx": int(robot[1:]) - 1,
                    "predicted_program_idx": int(program[1:]) // 10 - 10,
                    "annotation": {"family": None},
                }
            )
        for position in range(abnormal_per_stratum[index]):
            files.append(
                {
                    "file_id": f"a{index}-{position:04d}",
                    "history_id": "H",
                    "arm": "OBSERVED",
                    "file_label": "abnormal",
                    "week_offset": 1 + position % 2,
                    "month_index": 1 + position % 2,
                    "predicted_robot_idx": int(robot[1:]) - 1,
                    "predicted_program_idx": int(program[1:]) // 10 - 10,
                    "annotation": {"family": "alpha"},
                }
            )
    manifest = {"files": files}
    audit_record = {
        "pair_rows": [
            {
                "history_id": "H",
                "horizon": "week",
                "comparison_identity": "H|week|calendar|BASELINE-VS-ELEVATED|B_A_TO_E_A",
                "comparison_type": "BASELINE-VS-ELEVATED",
                "baseline_period_identities": ["H|week|calendar|W01"],
                "contrast_period_identities": ["H|week|calendar|W02"],
            }
        ]
    }
    return manifest, audit_record


# --- apportionment and quotas ---------------------------------------------------


def test_apportion_is_capped_largest_remainder() -> None:
    assert audit.apportion({"1": 10, "2": 5}, {"1": 10, "2": 5}, 8) == {"1": 5, "2": 3}
    capped = audit.apportion({"1": 10, "2": 10}, {"1": 1, "2": 10}, 5)
    assert capped == {"1": 1, "2": 4}
    assert audit.apportion({"1": 4, "2": 4}, {"1": 4, "2": 4}, 9) == {}


def test_period_quota_is_exact_and_abstains_when_infeasible() -> None:
    table = _cell_table(normal_per_stratum=(6, 6), abnormal_per_stratum=(2, 1), periods=2)
    quota, status = audit.period_quota(table, "R01|P100", 2)
    assert status == audit.COMPUTED
    assert sum(quota.values()) == 2
    for number, count in quota.items():
        code = table.period_numbers.index(number)
        positions = table.positions("R01|P100")
        assert count <= int(np.sum(table.period_codes[positions] == code))
    empty, infeasible = audit.period_quota(table, "R01|P100", 4)
    assert empty == {} and infeasible == audit.ABSTAIN_PERIOD_MATCH


def test_normal_quota_fits_two_disjoint_arms() -> None:
    table = _cell_table(normal_per_stratum=(40, 10), abnormal_per_stratum=(2, 2))
    quota, status = audit.normal_quota(table, 20, arms=2)
    assert status == audit.COMPUTED and sum(quota.values()) == 20
    for name, count in quota.items():
        positions = table.positions(name)
        assert 2 * count <= int(np.sum(~table.abnormal[positions]))


# --- estimators -----------------------------------------------------------------


def test_e1_repeats_match_the_accepted_unbiased_estimator() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(40, 8))
    positions_a = np.asarray([[0, 1, 2], [3, 4, 5]])
    positions_b = np.asarray([[6, 7, 8], [9, 10, 11]])
    values = audit.e1_repeats(matrix, 1.3, positions_a, positions_b)
    for repeat in range(2):
        expected = endpoints.mmd2_unbiased(
            matrix[positions_a[repeat]], matrix[positions_b[repeat]], 1.3
        )
        assert values[repeat] == pytest.approx(expected, rel=0, abs=1e-15)


def test_e2_quanta_equal_exact_occupancy_tv() -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2, 2, 0, 1, 2], dtype=np.int64)
    positions_a = np.asarray([[0, 1, 2], [3, 4, 5]])
    positions_b = np.asarray([[6, 7, 8], [9, 9, 9]])
    quanta = audit.m12.e2_quanta(labels, positions_a, positions_b, 3)
    size = 3
    for repeat in range(2):
        counts_a = np.bincount(labels[positions_a[repeat]], minlength=3)
        counts_b = np.bincount(labels[positions_b[repeat]], minlength=3)
        assert quanta[repeat] * 2 == int(np.abs(counts_a - counts_b).sum())
        assert (quanta[repeat] / size) == pytest.approx(
            0.5 * np.abs(counts_a / size - counts_b / size).sum()
        )


def test_within_prototype_identical_arms_are_exactly_zero() -> None:
    centroids = np.asarray([[2.0, 0.0], [-2.0, 0.0]])
    residuals = np.asarray([[0.1, 0.2], [0.3, -0.1], [-0.2, 0.05], [0.0, 0.0]])
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    positions = np.asarray([[0, 1], [2, 3]])
    diagnostic = audit.within_prototype_arrays(
        residual_a=residuals[positions],
        labels_a=labels[positions],
        residual_b=residuals[positions],
        labels_b=labels[positions],
        centroids=centroids,
        ref_scale=0.5,
        k=2,
    )
    assert diagnostic["D_LOC"][0] == 0.0
    assert diagnostic["D_SPREAD"][0] == 0.0
    assert diagnostic["coverage"][0] == 1.0


def test_within_prototype_sees_tangential_movement_that_radial_misses() -> None:
    centroids = np.asarray([[2.0, 0.0]])
    residuals_a = np.asarray([[[0.10, 0.0], [0.20, 0.0]]])
    residuals_b = np.asarray([[[0.10, 0.30], [0.20, 0.30]]])
    labels = np.asarray([[0, 0]])
    diagnostic = audit.within_prototype_arrays(
        residual_a=residuals_a,
        labels_a=labels,
        residual_b=residuals_b,
        labels_b=labels,
        centroids=centroids,
        ref_scale=1.0,
        k=1,
    )
    assert diagnostic["D_LOC"][0] > 0.0
    assert diagnostic["D_RADIAL"][0] == pytest.approx(0.0, abs=1e-15)
    assert diagnostic["D_TANGENTIAL"][0] == pytest.approx(0.09, rel=1e-12)

    radial_b = residuals_a + np.asarray([[0.20, 0.0]])
    radial = audit.within_prototype_arrays(
        residual_a=residuals_a,
        labels_a=labels,
        residual_b=radial_b,
        labels_b=labels,
        centroids=centroids,
        ref_scale=1.0,
        k=1,
    )
    assert radial["D_RADIAL"][0] == pytest.approx(0.04, rel=1e-12)
    assert radial["D_TANGENTIAL"][0] == pytest.approx(0.0, abs=1e-15)


def test_within_prototype_excludes_empty_cluster_and_abstains_below_floor() -> None:
    centroids = np.asarray([[0.0, 0.0], [5.0, 0.0]])
    residuals_a = np.asarray([[[0.1, 0.0], [0.2, 0.0]]])
    residuals_b = np.asarray([[[0.1, 0.0], [0.2, 0.0]]])
    labels_a = np.asarray([[0, 0]])
    labels_b = np.asarray([[0, 1]])
    diagnostic = audit.within_prototype_arrays(
        residual_a=residuals_a,
        labels_a=labels_a,
        residual_b=residuals_b,
        labels_b=labels_b,
        centroids=centroids,
        ref_scale=1.0,
        k=2,
    )
    assert diagnostic["coverage"][0] == pytest.approx(0.75)
    assert np.isfinite(diagnostic["D_LOC"][0])
    half = audit.within_prototype_arrays(
        residual_a=residuals_a,
        labels_a=np.asarray([[0, 0]]),
        residual_b=residuals_b,
        labels_b=np.asarray([[1, 1]]),
        centroids=centroids,
        ref_scale=1.0,
        k=2,
    )
    assert np.isnan(half["D_LOC"][0])


# --- perturbation ---------------------------------------------------------------


def test_zero_shift_perturbation_is_identity() -> None:
    rows = np.arange(12, dtype=np.float64).reshape(3, 4)
    direction = np.asarray([0.0, 1.0, 0.0, 0.0])
    shifted, mask, norm = audit.perturb(
        rows, delta=0.0, direction=direction, sigma_u=2.5, margin=np.zeros(3), mode=audit.MODE_GLOBAL
    )
    assert norm == 0.0
    assert mask.all()
    assert np.array_equal(shifted, rows)


def test_boundary_safe_perturbation_only_shifts_clear_rows() -> None:
    rows = np.zeros((4, 2))
    direction = np.asarray([1.0, 0.0])
    margins = np.asarray([0.0, 0.1, 0.5, 2.0])
    shifted, mask, norm = audit.perturb(
        rows,
        delta=1.0,
        direction=direction,
        sigma_u=0.25,
        margin=margins,
        mode=audit.MODE_SAFE,
    )
    assert norm == pytest.approx(0.25)
    assert mask.tolist() == [False, False, True, True]
    assert np.allclose(shifted[mask], np.asarray([[0.25, 0.0], [0.25, 0.0]]))
    assert np.allclose(shifted[~mask], np.zeros((2, 2)))


# --- draws ----------------------------------------------------------------------


def test_draw_pooled_arms_are_disjoint_duplicate_free_and_deterministic() -> None:
    table = _cell_table(normal_per_stratum=(40, 0), abnormal_per_stratum=(8, 0))
    first = audit.draw_pooled(table, "R01|P100", 3, 5)
    second = audit.draw_pooled(table, "R01|P100", 3, 5)
    for repeat in range(5):
        abnormal = set(first.abnormal[repeat].tolist())
        arms = [
            set(first.normal_1[repeat].tolist()),
            set(first.normal_2[repeat].tolist()),
            set(first.normal_3[repeat].tolist()),
        ]
        assert abnormal.isdisjoint(set().union(*arms))
        assert arms[0].isdisjoint(arms[1]) and arms[1].isdisjoint(arms[2])
        assert all(len(arm) == 3 for arm in arms)
        for arm in [first.abnormal[repeat], first.normal_1[repeat]]:
            assert len(set(arm.tolist())) == arm.size
    for name in ("abnormal", "normal_1", "normal_2", "normal_3"):
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_draw_controls_are_disjoint_and_follow_the_quota() -> None:
    table = _cell_table(normal_per_stratum=(40, 20), abnormal_per_stratum=(0, 0))
    quota = {"R01|P100": 2, "R02|P100": 1}
    draw = audit.draw_controls(table, 3, quota, 181801, 4)
    for repeat in range(4):
        assert set(draw.unchanged[repeat].tolist()).isdisjoint(draw.perturbed_base[repeat].tolist())
        for name, count in quota.items():
            index = table.stratum_names.index(name)
            assert int(np.sum(table.strata[draw.unchanged[repeat]] == index)) == count
            assert int(np.sum(table.strata[draw.perturbed_base[repeat]] == index)) == count


# --- census and custody ---------------------------------------------------------


def test_census_accounts_every_planned_unit_metadata_only() -> None:
    manifest, audit_record = _manifest_and_audit()
    census = audit.build_census(manifest, audit_record)
    units = census["branch_a"]["units"]
    assert len(units) == 2 * len(audit.SIZES_A["week"])
    assert census["metadata_only"] is True and census["embeddings_read"] is False
    assert census["outcomes_computed"] is False
    counted = sum(census["branch_a"]["unit_status_counts"].values())
    assert counted == len(units)
    for unit in units:
        if unit["status"] == audit.COMPUTED:
            assert unit["n_abnormal"] >= unit["size"]
            assert unit["n_normal"] >= 3 * unit["size"]
    assert census["branch_c"]["unit_status_counts"].get(audit.COMPUTED, 0) == 2


def test_census_abstains_on_a_stratum_without_abnormal_files() -> None:
    manifest, audit_record = _manifest_and_audit(
        normal_per_stratum=(100, 100), abnormal_per_stratum=(10, 0)
    )
    census = audit.build_census(manifest, audit_record)
    statuses = {unit["stratum"]: unit["status"] for unit in census["branch_a"]["units"]}
    assert statuses["R02|P100"] == audit.ABSTAIN_NO_ABNORMAL
    assert statuses["R01|P100"] == audit.COMPUTED


def test_census_is_deterministic() -> None:
    manifest, audit_record = _manifest_and_audit()
    first = audit.build_census(manifest, audit_record)
    second = audit.build_census(manifest, audit_record)
    from representation.sprint24_context_audit import canonical_json

    assert canonical_json(first) == canonical_json(second)


def test_comparison_summary_marks_null_q95_excess() -> None:
    null = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    contrast = np.asarray([5.0, 6.0, 7.0, 8.0, 9.0])
    summary = audit.comparison_summary(contrast, null, "E1", 5)
    assert summary["status"] == audit.COMPUTED
    assert summary["median_above_null_q95"] is True
    assert summary["fraction_contrast_above_null_q95"] == 1.0
    flat = audit.comparison_summary(np.asarray([0.0] * 5), null, "E1", 5)
    assert flat["median_above_null_q95"] is False


def test_frozen_specification_refuses_byte_drift(tmp_path: Path) -> None:
    loaded = audit.load_specification()
    assert loaded["artifact_id"] == audit.SPEC_ID
    source = Path(audit.REPO_ROOT) / audit.SPEC_PATH
    tampered = tmp_path / "spec.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["title"] = payload["title"] + " (tampered)"
    tampered.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="REFUSED-SPEC-DRIFT"):
        audit.load_specification(tampered)


def test_e2_detection_fraction_excludes_invalid_repeat_sentinels() -> None:
    null = np.zeros(4)
    contrast = np.asarray([1.0, 1.0, -1.0, -1.0])
    summary = audit.comparison_summary(contrast, null, "E2_K8", 4)
    assert summary["status"] == audit.COMPUTED
    assert summary["fraction_contrast_above_null_q95"] == 1.0


def _summary_values(valid_repeats: int, repeats: int = 10) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    valid = np.arange(repeats) < valid_repeats
    for k in audit.CLUSTER_COUNTS:
        values[f"D_LOC_K{k}_contrast"] = np.where(valid, 0.1, np.nan)
        values[f"D_LOC_K{k}_null"] = np.full(repeats, 0.2)
        values[f"D_SPREAD_K{k}_contrast"] = np.where(valid, 0.1, np.nan)
        values[f"D_SPREAD_K{k}_null"] = np.full(repeats, 0.2)
        coverage = np.where(valid, 1.0, 0.1)
        values[f"coverage_K{k}_contrast"] = coverage
        values[f"coverage_K{k}_null"] = coverage
        values[f"E2_K{k}_contrast"] = np.zeros(repeats)
        values[f"E2_K{k}_null"] = np.ones(repeats)
    values["D_RADIAL_contrast"] = np.where(valid, 0.1, np.nan)
    values["D_RADIAL_null"] = np.full(repeats, 0.2)
    values["D_TANGENTIAL_contrast"] = np.where(valid, 0.1, np.nan)
    values["D_TANGENTIAL_null"] = np.full(repeats, 0.2)
    values["E1_contrast"] = np.zeros(repeats)
    values["E1_null"] = np.ones(repeats)
    values["period_tv"] = np.zeros(repeats)
    values["boundary_fraction_abnormal"] = np.zeros(repeats)
    values["boundary_fraction_normal_1"] = np.zeros(repeats)
    return values


def test_coverage_record_is_descriptive_and_d_uses_valid_repeats() -> None:
    summary = audit.summarise_arm_values(_summary_values(4), 10)
    coverage = summary["coverage"]
    assert "status" not in coverage
    assert coverage["repeats_valid"] == 4 and coverage["repeats_total"] == 10
    assert coverage["reliability"] == "PARTIAL"
    assert coverage["at_least_half_valid"] is False
    assert summary["estimators"]["D_LOC"]["status"] == audit.COMPUTED
    assert summary["estimators"]["D_LOC"]["contrast"]["n"] == 4

    full = audit.summarise_arm_values(_summary_values(10), 10)
    assert full["coverage"]["reliability"] == "FULL"
    assert full["coverage"]["at_least_half_valid"] is True

    empty = audit.summarise_arm_values(_summary_values(0), 10)
    assert empty["estimators"]["D_LOC"]["status"] == audit.ABSTAIN_COVERAGE
    assert empty["coverage"]["repeats_valid"] == 0
