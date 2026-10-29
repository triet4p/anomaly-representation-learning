"""Focused checks for the Sprint 24 Task 7-M13 dose-response audit.

These protect consumer-visible behaviours: capped apportionment, the frozen
composition invariants, duplicate-free and disjoint arms, exact equal-``n``
occupancy quanta, the accepted E1 estimator, explicit abstention on
insufficient support, and the frozen-specification/custody refusals.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from representation import sprint24_controlled_dose_response as dose
from representation import sprint24_weekly_monthly_endpoints as endpoints


def _pool(
    *,
    ordinary_per_stratum: int = 40,
    abnormal_layout: tuple[tuple[int, str, int], ...] = (
        (0, "alpha", 6),
        (0, "beta", 6),
        (1, "alpha", 6),
        (1, "beta", 6),
    ),
    periods: int = 3,
) -> dose.PoolTable:
    """Build a small deterministic pool with a known stratum/family layout."""
    ids: list[str] = []
    strata: list[int] = []
    stratum_names = ["R01|P100", "R02|P100"]
    families: list[int] = []
    family_names = [dose.ORDINARY, "alpha", "beta"]
    abnormal: list[bool] = []
    period_codes: list[int] = []
    for stratum_index, name in enumerate(stratum_names):
        for position in range(ordinary_per_stratum):
            ids.append(f"{name}-ord-{position:03d}")
            strata.append(stratum_index)
            families.append(0)
            abnormal.append(False)
            period_codes.append(position % periods)
    for stratum_index, family, count in abnormal_layout:
        for position in range(count):
            ids.append(f"{stratum_names[stratum_index]}-{family}-{position:03d}")
            strata.append(stratum_index)
            families.append(family_names.index(family))
            abnormal.append(True)
            period_codes.append(position % periods)
    return dose.PoolTable(
        ids=ids,
        strata=np.asarray(strata, dtype=np.int64),
        stratum_names=stratum_names,
        families=np.asarray(families, dtype=np.int64),
        family_names=family_names,
        abnormal=np.asarray(abnormal, dtype=bool),
        periods=np.asarray(period_codes, dtype=np.int64),
        period_numbers=list(range(periods)),
        low_periods=frozenset({0}),
    )


def test_apportion_is_capped_largest_remainder() -> None:
    assert dose.apportion({1: 10, 2: 5}, {1: 10, 2: 5}, 8) == {1: 5, 2: 3}
    capped = dose.apportion({1: 10, 2: 10}, {1: 1, 2: 10}, 5)
    assert capped[1] == 1 and capped[2] == 4 and sum(capped.values()) == 5
    assert dose.apportion({1: 4, 2: 4}, {1: 4, 2: 4}, 9) == {}


def test_frame_keeps_two_disjoint_ordinary_arms_possible() -> None:
    pool = _pool()
    frame = dose.build_frame(pool, 20, 8)
    assert frame.status == dose.COMPUTED
    assert sum(frame.quota.values()) == 20
    for name, quota in frame.quota.items():
        assert 2 * quota <= frame.ordinary_by_stratum[name]


def test_abnormal_sequence_prefix_is_proportional_and_nested() -> None:
    pool = _pool()
    frame = dose.build_frame(pool, 20, 8)
    assert frame.status == dose.COMPUTED
    for count in (3, 5, 8):
        prefix = frame.sequence[:count]
        families = [pool.family_of(index) for index in prefix]
        assert abs(families.count("alpha") - families.count("beta")) <= 1
    assert set(frame.sequence[:5]) <= set(frame.sequence[:8])


def test_dose_plan_holds_context_counts_and_total() -> None:
    pool = _pool()
    frame = dose.build_frame(pool, 20, 8)
    for fraction in dose.DOSE_GRID[dose.DESIGN_POOLED]:
        plan = dose.dose_plan(frame, fraction)
        assert plan["n_abnormal"] == dose.abnormal_count(fraction, 20)
        assert plan["n_abnormal"] + plan["n_ordinary"] == 20
        context = {
            name: plan["abnormal_by_stratum"].get(name, 0)
            + plan["ordinary_by_stratum"].get(name, 0)
            for name in frame.quota
        }
        assert context == frame.quota
        assert sum(plan["abnormal_by_pair"].values()) == plan["n_abnormal"]


def test_draws_are_duplicate_free_and_disjoint_from_baseline() -> None:
    pool = _pool()
    frame = dose.build_frame(pool, 20, 8)
    baseline = dose.baseline_indices(frame, cell="cell", design="POOLED-CELL-PERIODS")
    assert np.unique(baseline).size == baseline.size == 20
    for fraction in dose.DOSE_GRID[dose.DESIGN_POOLED]:
        plan = dose.dose_plan(frame, fraction)
        draws = dose.draw_dose_arms(
            frame,
            plan,
            baseline,
            cell="cell",
            design=dose.DESIGN_POOLED,
            repeats=12,
        )
        assert draws.shape == (12, 20)
        for row in draws:
            assert np.unique(row).size == row.size
            assert not set(row.tolist()) & set(baseline.tolist())
        context = {
            name: int(np.sum([pool.stratum_of(index) == name for index in draws[0]]))
            for name in frame.quota
        }
        assert context == frame.quota


def test_abstains_when_abnormal_support_is_insufficient() -> None:
    pool = _pool(abnormal_layout=((0, "alpha", 2), (0, "beta", 2)))
    frame = dose.build_frame(pool, 20, 8)
    assert frame.status == dose.ABSTAIN_INSUFFICIENT_ABNORMAL
    assert frame.sequence == [] and frame.quota


def test_abstains_when_ordinary_capacity_is_short() -> None:
    pool = _pool(ordinary_per_stratum=8)
    frame = dose.build_frame(pool, 40, 4)
    assert frame.status == dose.ABSTAIN_ORDINARY_CAPACITY
    assert "disjoint arms" in frame.reason


def test_e1_repeats_match_the_accepted_unbiased_mmd() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(60, 8))
    group = type("Group", (), {"matrix": matrix})()
    baseline = np.arange(20)
    draws = np.stack([np.arange(20, 40), np.arange(25, 45)])
    values = dose.e1_repeats(group, baseline, draws, sigma=2.5)
    expected = [
        endpoints.mmd2_unbiased(matrix[draw], matrix[baseline], 2.5) for draw in draws
    ]
    assert values.tolist() == pytest.approx(expected, rel=0.0, abs=1e-15)


def test_e2_quanta_are_exact_integer_numerators() -> None:
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 4, size=80)
    baseline = np.arange(0, 20)
    draws = np.stack([np.arange(20, 40), np.arange(40, 60)])
    quanta = dose.e2_quanta_repeats(labels, baseline, draws, 4)
    for repeat, draw in enumerate(draws):
        left = np.bincount(labels[baseline], minlength=4) / 20.0
        right = np.bincount(labels[draw], minlength=4) / 20.0
        tv = 0.5 * float(np.abs(left - right).sum())
        assert float(quanta[repeat]) == pytest.approx(tv * 20.0, abs=1e-12)
        assert float(quanta[repeat]).is_integer()


def test_movement_summary_compares_against_the_null_band() -> None:
    null = np.arange(100, dtype=np.float64)
    above = np.full(50, 99.0)
    below = np.zeros(50)
    assert dose.movement_summary(above, null, size=50, exact_quanta=True)[
        "median_above_null_q95"
    ]
    assert (
        dose.movement_summary(below, null, size=50, exact_quanta=True)[
            "fraction_above_null_q95"
        ]
        == 0.0
    )


def test_spearman_is_tie_aware() -> None:
    assert dose.spearman([1, 2, 3], [2, 4, 9]) == pytest.approx(1.0)
    assert dose.spearman([1, 2, 3], [9, 4, 2]) == pytest.approx(-1.0)
    assert dose.spearman([1, 1, 1], [1, 2, 3]) is None
    tied = dose.spearman([0.0, 0.05, 0.1, 0.2], [0.5, 0.5, 0.6, 0.9])
    assert tied is not None and 0.0 < tied < 1.0


def test_spec_digest_refusal_on_byte_drift(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    original = (dose.REPO_ROOT / dose.SPEC_PATH).read_bytes()
    spec_path.write_bytes(original)
    assert dose.load_specification(spec_path)["sha256"] == dose.SPEC_SHA256
    spec_path.write_bytes(original.replace(b"FROZEN-BEFORE", b"THAWED-BEFORE"))
    with pytest.raises(ValueError, match="REFUSED-SPEC-DRIFT"):
        dose.load_specification(spec_path)


def test_amendment_requires_the_frozen_spec_digest(tmp_path: Path) -> None:
    foreign = tmp_path / "A01.json"
    foreign.write_text(
        json.dumps(
            {
                "amends_spec_sha256": "0" * 64,
                "created_utc": "2026-10-28T00:00:00Z",
                "reason": "foreign",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="REFUSED-AMENDMENT"):
        dose.load_amendments(tmp_path, spec_sha256=dose.SPEC_SHA256)
    assert dose.load_amendments(tmp_path / "absent", spec_sha256=dose.SPEC_SHA256) == []


def _synthetic_manifest() -> dict[str, object]:
    files: list[dict[str, object]] = []
    for index in range(40):
        for week in (1, 2, 3):
            files.append(
                {
                    "file_id": f"n-{index:02d}-{week}",
                    "history_id": "H1",
                    "week_offset": week,
                    "month_index": None,
                    "arm": "NONE",
                    "predicted_robot_idx": index % 2,
                    "predicted_program_idx": 0,
                    "file_label": "normal",
                    "annotation": {"family": None, "severity": None},
                }
            )
    for index in range(12):
        for week in (1, 2, 3):
            files.append(
                {
                    "file_id": f"a-{index:02d}-{week}",
                    "history_id": "H1",
                    "week_offset": week,
                    "month_index": None,
                    "arm": "NONE",
                    "predicted_robot_idx": index % 2,
                    "predicted_program_idx": 0,
                    "file_label": "abnormal",
                    "annotation": {
                        "family": "alpha" if index % 2 == 0 else "beta",
                        "severity": 0.5,
                        "abnormal_step_fraction": 0.1,
                        "duration_steps": 3,
                    },
                }
            )
    for index in range(20):
        files.append(
            {
                "file_id": f"r-{index:02d}",
                "history_id": "H1",
                "week_offset": None,
                "month_index": None,
                "arm": "REFERENCE",
                "predicted_robot_idx": 0,
                "predicted_program_idx": 0,
                "file_label": "normal",
                "annotation": {"family": None, "severity": None},
            }
        )
    return {"files": files}


def _synthetic_audit() -> dict[str, object]:
    return {
        "pair_rows": [
            {
                "history_id": "H1",
                "horizon": "week",
                "comparison_identity": "H1|week|BASELINE-A_TO_ELEVATED-A",
                "comparison_type": "BASELINE-VS-ELEVATED",
                "baseline_period_identities": ["H1|week|calendar|W01", "H1|week|calendar|W02"],
                "contrast_period_identities": ["H1|week|calendar|W03"],
            },
            {
                "history_id": "H1",
                "horizon": "week",
                "comparison_identity": "H1|week|BASELINE-A_TO_BASELINE-B",
                "comparison_type": "BASELINE-VS-BASELINE",
                "baseline_period_identities": ["H1|week|calendar|W01", "H1|week|calendar|W02"],
                "contrast_period_identities": ["H1|week|calendar|W02"],
            },
        ]
    }


def test_pool_excludes_reference_arm_and_uses_only_cell_periods() -> None:
    manifest = _synthetic_manifest()
    cells = dose.build_cells(_synthetic_audit())
    cell = cells["H1|week"]
    assert cell["low_periods"] == {1, 2}
    assert cell["elevated_periods"] == {3}
    pool = dose.build_pool(manifest, cell, dose.DESIGN_LOW)
    assert all(not identifier.startswith("r-") for identifier in pool.ids)
    assert set(pool.period_numbers) == {1, 2}
    assert int(pool.abnormal.sum()) == 24


def test_census_is_metadata_only_and_reports_support() -> None:
    census = dose.build_census(_synthetic_manifest(), _synthetic_audit())
    assert census["metadata_only"] is True
    assert census["embeddings_read"] is False
    assert census["outcomes_computed"] is False
    assert census["n_cells"] == 1
    assert census["n_units"] == 2 * len(dose.DESIGNS)
    assert sum(census["unit_status_counts"].values()) == census["n_units"]
    assert census["parameters"]["count_rounding"] == "floor(p * N + 0.5)"


def test_frame_is_refused_when_the_abnormal_headroom_is_exhausted() -> None:
    # Abnormal files concentrated in one stratum whose quota is smaller than the
    # requested max dose: the frame must abstain instead of overshooting context.
    pool = _pool(
        ordinary_per_stratum=40,
        abnormal_layout=((0, "alpha", 20), (0, "beta", 20)),
    )
    frame = dose.build_frame(pool, 20, 12)
    assert frame.status in {
        dose.ABSTAIN_ABNORMAL_HEADROOM,
        dose.ABSTAIN_INSUFFICIENT_ABNORMAL,
    }
    assert frame.sequence == []


def test_realized_composition_counts_the_actual_draws() -> None:
    labels = [dose._bin_label(edges) for edges in dose.SEVERITY_BINS] + [dose.ORDINARY]
    fraction_labels = [
        dose._bin_label(edges) for edges in dose.FRACTION_BINS
    ] + [dose.ORDINARY]
    codes = {
        "severity": np.asarray([0, 1, 0, 1, 4, 4, 4, 4], dtype=np.int64),
        "fraction": np.asarray([0, 0, 1, 1, 4, 4, 4, 4], dtype=np.int64),
        "duration": np.asarray([10, 20, 30, 40, -1, -1, -1, -1], dtype=np.int64),
    }
    draws = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    planned = {
        "severity_counts": {labels[0]: 1, labels[1]: 1, dose.ORDINARY: 2},
        "fraction_counts": {
            fraction_labels[0]: 1,
            fraction_labels[1]: 1,
            dose.ORDINARY: 2,
        },
        "family_counts": {"alpha": 4},
    }
    realized = dose.realized_composition(draws, codes, planned, size=4)
    assert realized["severity_counts_repeat_0"][labels[0]] == 2
    assert realized["severity_counts_repeat_0"][labels[1]] == 2
    assert realized["severity_counts_repeat_0"][dose.ORDINARY] == 0
    assert (
        realized["severity_mix_tv_vs_planned_sequence_prefix"]["median"]
        == pytest.approx(0.5)
    )
    assert realized["abnormal_duration"]["median_of_repeat_medians"] == pytest.approx(
        25.0
    )
    assert realized["abnormal_duration"]["min"] == 10
    assert realized["abnormal_duration"]["max"] == 40
    assert realized["severity_controlled"] is False
    assert realized["ordinary_count_exact"] == 0


def test_source_reuse_reports_realized_non_nesting() -> None:
    ids = [f"a{index}" for index in range(4)] + [f"o{index}" for index in range(8)]
    pool = dose.PoolTable(
        ids=ids,
        strata=np.zeros(12, dtype=np.int64),
        stratum_names=["R01|P100"],
        families=np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        family_names=[dose.ORDINARY, "alpha"],
        abnormal=np.asarray([True] * 4 + [False] * 8, dtype=bool),
        periods=np.zeros(12, dtype=np.int64),
        period_numbers=[1],
        low_periods=frozenset({1}),
    )
    draws = {
        0.0: np.asarray([[4, 5, 6, 7]], dtype=np.int64),
        0.1: np.asarray([[0, 1, 4, 5]], dtype=np.int64),
        0.2: np.asarray([[2, 3, 6, 7]], dtype=np.int64),
    }
    reuse = dose._source_reuse(pool, draws, (0.0, 0.1, 0.2), 1, 4, "cell")
    assert reuse["nesting"]["planned_prefix_nested"] is True
    assert reuse["nesting"]["realized_identity_sets_nested"] is False
    assert reuse["nesting"]["positive_transitions_checked"] == 2
    assert reuse["nesting"]["reference_repeat_failures"] == 1
    assert (
        reuse["overlap_across_repeats"]["0.20"][
            "abnormal_overlap_fraction_with_previous_dose"
        ]["median"]
        == 0.0
    )
    assert (
        reuse["overlap_across_repeats"]["0.20"][
            "ordinary_overlap_fraction_with_zero_dose"
        ]["median"]
        == 1.0
    )


def test_abnormal_only_composition_uses_the_actual_abnormal_files() -> None:
    # Abnormal-only mixtures must be normalised inside the realized abnormal
    # identities (never diluted by the ordinary part) and each adjacent-dose
    # comparison must carry its empirical TV quantum and support flag.
    pool = _pool(
        ordinary_per_stratum=160,
        abnormal_layout=(
            (0, "alpha", 16),
            (0, "beta", 16),
            (1, "alpha", 16),
            (1, "beta", 16),
        ),
    )
    frame = dose.build_frame(pool, 160, 64)
    assert frame.status == dose.COMPUTED
    grid = dose.DOSE_GRID[dose.DESIGN_POOLED]
    plans = {fraction: dose.dose_plan(frame, fraction) for fraction in grid}
    baseline = dose.baseline_indices(
        frame, cell="cell", design=dose.DESIGN_POOLED
    )
    draws = {
        fraction: dose.draw_dose_arms(
            frame,
            plans[fraction],
            baseline,
            cell="cell",
            design=dose.DESIGN_POOLED,
            repeats=4,
        )
        for fraction in grid
    }
    entries = {
        pool.ids[index]: {
            "annotation": {
                "severity": 0.5,
                "abnormal_step_fraction": 0.10,
                "duration_steps": 30,
            }
        }
        for index in np.flatnonzero(pool.abnormal).tolist()
    }
    codes = dose.composition_codes(frame, entries)
    records = dose.abnormal_only_composition(frame, draws, grid, codes, plans)

    assert records["0.00"]["n_abnormal"] == 0
    assert records["0.00"]["comparison_supported"] is False
    positive = [fraction for fraction in grid if plans[fraction]["n_abnormal"] > 0]
    # The first positive dose has no abnormal predecessor to compare against.
    assert "comparison_supported_vs_previous" not in records[
        f"{positive[0]:.2f}"
    ]
    for fraction in positive[1:]:
        record = records[f"{fraction:.2f}"]
        expected_quantum = 0.5 * (
            1.0 / plans[fraction]["n_abnormal"]
            + 1.0 / plans[positive[positive.index(fraction) - 1]]["n_abnormal"]
        )
        assert record["tv_quantum_vs_previous"] == pytest.approx(expected_quantum)
        assert record["comparison_supported_vs_previous"] is (
            expected_quantum <= dose.COMPOSITION_TV_QUANTUM_FLOOR
        )
        assert sum(record["severity_probs_repeat_0"].values()) == pytest.approx(1.0)
        assert sum(record["severity_counts_mean"].values()) == pytest.approx(
            float(record["n_abnormal"])
        )
        assert record["vs_previous_dose"]["severity"]["median"] == pytest.approx(0.0)
        assert record["vs_previous_dose"]["duration"]["median"] == pytest.approx(0.0)
        assert record["vs_planned_abnormal_only"]["severity"]["median"] == pytest.approx(
            0.0
        )

    # 8 -> 16 abnormal files is below the diagnostic resolution; 16 -> 32 and
    # the wider transitions are supported exactly as the frozen rule states.
    assert records["0.10"]["comparison_supported_vs_previous"] is False
    assert records["0.20"]["comparison_supported_vs_previous"] is True

    aggregates = dose.abnormal_composition_aggregates(
        [
            {
                "unit": "cell|month|160|POOLED-CELL-PERIODS",
                "cell": "cell",
                "horizon": "month",
                "design": dose.DESIGN_POOLED,
                "size": 160,
                "doses": records,
            }
        ]
    )
    assert len(aggregates) == 1
    assert aggregates[0]["supported_adjacent_transitions"] == 3
    assert aggregates[0]["unsupported_adjacent_transitions"] == 1
    assert aggregates[0]["median_severity_tv"] == pytest.approx(0.0)


def test_nested_prefix_arms_keep_the_context_quota_and_abnormal_nesting() -> None:
    pool = _pool()
    frame = dose.build_frame(pool, 20, 8)
    assert frame.status == dose.COMPUTED
    grid = (0.0, 0.10, 0.40)
    plans = {fraction: dose.dose_plan(frame, fraction) for fraction in grid}
    baseline = dose.baseline_indices(
        frame, cell="cell", design=dose.DESIGN_POOLED
    )
    draws = {
        fraction: dose.draw_dose_arms(
            frame,
            plans[fraction],
            baseline,
            cell="cell",
            design=dose.DESIGN_POOLED,
            repeats=6,
        )
        for fraction in grid
    }
    nested = dose.nested_prefix_arms(frame, plans, draws, grid, baseline)

    expected = np.asarray(
        [frame.quota[name] for name in pool.stratum_names], dtype=np.int64
    )
    for fraction in grid:
        arm = nested[fraction]
        assert arm.shape == (6, 20)
        assert np.all(np.diff(arm, axis=1) > 0)
        counts = np.stack(
            [
                np.bincount(pool.strata[row], minlength=len(pool.stratum_names))
                for row in arm
            ]
        )
        assert np.array_equal(counts, np.broadcast_to(expected, counts.shape))
        abnormal = pool.abnormal[arm]
        assert np.all(abnormal.sum(axis=1) == plans[fraction]["n_abnormal"])
        assert not bool(np.isin(arm, baseline).any())

    for repeat in range(6):
        for low, high in ((0.0, 0.10), (0.10, 0.40), (0.0, 0.40)):
            low_ids = set(
                nested[low][repeat][pool.abnormal[nested[low][repeat]]].tolist()
            )
            high_ids = set(
                nested[high][repeat][pool.abnormal[nested[high][repeat]]].tolist()
            )
            assert low_ids <= high_ids

    # At the anchor dose the nested arm reproduces the implemented arm exactly,
    # so the sensitivity measures only the cross-dose identity rule.
    assert np.array_equal(nested[0.40], draws[0.40])
