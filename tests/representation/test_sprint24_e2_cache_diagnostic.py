"""Focused checks for the S24-7-M12 cache-only E2 diagnostic."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from representation import sprint24_controlled_inference_readiness as readiness  # noqa: E402
from representation import sprint24_e2_cache_diagnostic as audit  # noqa: E402
from representation import sprint24_weekly_monthly_endpoints as endpoints  # noqa: E402

DAY = 86400.0


def _group(file_ids: list[str], labels: np.ndarray) -> endpoints.EmbeddingGroup:
    """Build a two-dimensional embedding group whose rows sit on the given sides."""
    matrix = np.zeros((len(file_ids), 2), dtype=np.float64)
    matrix[labels == 1, 1] = 10.0
    return endpoints.EmbeddingGroup(
        history_id="H",
        model_seed=181801,
        file_ids=list(file_ids),
        index={value: position for position, value in enumerate(file_ids)},
        matrix=matrix,
    )


def _index(entries: dict[str, tuple[int, int]]) -> dict[str, dict[str, object]]:
    """Manifest-like index with only the label-blind context fields."""
    return {
        fid: {
            "file_id": fid,
            "predicted_robot_idx": robot,
            "predicted_program_idx": program,
            "file_label": "normal",
            "annotation": {},
            "waveform_sha256": fid,
        }
        for fid, (robot, program) in entries.items()
    }


def _row(previous: list[str], current: list[str], identity: str, kind: str) -> dict[str, object]:
    return {
        "comparison_identity": identity,
        "comparison_type": kind,
        "previous_ids": previous,
        "current_ids": current,
        "n_baseline": len(previous),
        "n_contrast": len(current),
        "history_id": "H",
        "horizon": "week",
        "mode": "calendar",
        "disposition": "FEASIBLE",
        "baseline_period_identities": [],
        "contrast_period_identities": [],
        "burden_baseline_after": 0.05,
        "burden_contrast_after": 0.25,
        "burden_contrast_gap_after": 0.2,
        "exposure_ratio_after": 0.9,
    }


def test_tv_matches_hand_derived_half_l1_distance() -> None:
    first = np.array([0.5, 0.25, 0.25])
    second = np.array([0.0, 0.5, 0.5])
    assert audit.tv(first, second) == pytest.approx(0.5 * (0.5 + 0.25 + 0.25))
    assert audit.tv(first, first) == 0.0


def test_e2_equals_full_arm_value_when_a_draw_is_the_whole_arm() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    file_ids = [f"f{position}" for position in range(labels.size)]
    group = _group(file_ids, labels)
    baseline = np.array([[0, 1, 2, 3, 4]])
    contrast = np.array([[3, 4, 5, 6, 7]])
    value = audit.e2_repeats(labels, baseline, contrast, 2)
    expected = audit.full_arm_e2(
        labels, ["f0", "f1", "f2", "f3", "f4"], ["f3", "f4", "f5", "f6", "f7"], group, 2
    )
    assert value.shape == (1,)
    assert float(value[0]) == pytest.approx(expected, abs=0.0)


def test_e2_quanta_is_an_exact_integer_and_zero_for_identical_subset_counts() -> None:
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    baseline = np.array([[0, 1, 2, 3]])  # occupancy [0.5, 0.5]
    identical = np.array([[0, 4, 2, 5]])  # occupancy [0.5, 0.5] from a different row set
    quanta = audit.e2_quanta(labels, baseline, identical, 2)
    assert quanta.dtype == np.int64
    assert quanta.tolist() == [0], "identical occupancy must be exactly zero, not a float tie"
    different = np.array([[0, 1, 4, 2]])  # occupancy [0.75, 0.25]
    quanta = audit.e2_quanta(labels, baseline, different, 2)
    assert quanta.tolist() == [1]
    assert audit.e2_repeats(labels, baseline, different, 2).tolist() == [1 / 4]


def test_prototype_counts_match_a_direct_histogram() -> None:
    codes = np.array([[0, 0, 1, 2], [2, 2, 2, 0]])
    counts = audit.prototype_counts(codes, 3)
    expected = np.array([[2, 1, 1], [1, 0, 3]])
    assert np.array_equal(counts, expected)


def test_occupancy_and_tv_are_the_declared_e2_definition() -> None:
    labels = np.array([0, 1, 1, 1])
    assert audit.occupancy(labels, 2).tolist() == [0.25, 0.75]
    assert audit.occupancy(labels, 2).sum() == pytest.approx(1.0)


def test_equal_n_draws_are_exact_size_without_replacement() -> None:
    labels = np.array([0] * 10, dtype=np.int64)
    file_ids = [f"f{position}" for position in range(10)]
    index = _index({fid: (0, 0) for fid in file_ids})
    group = _group(file_ids, labels)
    row = _row(file_ids[:6], file_ids[4:], "H|week|T", "BASELINE-VS-ELEVATED")
    plan = audit.design_plan(group=group, row=row, size=4, index=index, repeats=25)
    for key in ("a_stratified_baseline", "a_stratified_contrast", "a_srs_baseline", "a_srs_contrast"):
        draws = plan[key]
        assert draws.shape == (25, 4)
        for draw in draws:
            assert len(set(draw.tolist())) == 4, "a draw repeated a row (sampling was with replacement)"
            assert set(draw.tolist()) <= set(range(10))


def test_sampling_is_reproducible_and_repeats_differ() -> None:
    labels = np.array([0] * 20, dtype=np.int64)
    file_ids = [f"f{position}" for position in range(20)]
    index = _index({fid: (0, 0) for fid in file_ids})
    group = _group(file_ids, labels)
    row = _row(file_ids[:10], file_ids[10:], "H|week|T", "BASELINE-VS-ELEVATED")
    first = audit.design_plan(group=group, row=row, size=6, index=index, repeats=5)
    second = audit.design_plan(group=group, row=row, size=6, index=index, repeats=5)
    for key in ("a_stratified_baseline", "a_stratified_contrast", "a_srs_baseline", "a_srs_contrast"):
        assert np.array_equal(first[key], second[key])
        assert not np.array_equal(first[key][0], first[key][1])


def test_stratified_draw_keeps_the_two_arms_on_an_identical_mix() -> None:
    file_ids = [f"a{position}" for position in range(6)] + [f"b{position}" for position in range(6)]
    file_ids += [f"c{position}" for position in range(6)] + [f"d{position}" for position in range(6)]
    entries: dict[str, tuple[int, int]] = {}
    for fid in file_ids:
        entries[fid] = (0, 0) if fid[0] in "ac" else (1, 0)
    index = _index(entries)
    group = _group(file_ids, np.zeros(len(file_ids), dtype=np.int64))
    baseline = [fid for fid in file_ids if fid[0] in "ab"]
    contrast = [fid for fid in file_ids if fid[0] in "cd"]
    row = _row(baseline, contrast, "H|week|T", "BASELINE-VS-ELEVATED")
    plan = audit.design_plan(group=group, row=row, size=8, index=index, repeats=3)
    assert plan["sampling_mode"] == audit.DESIGN_STRATIFIED
    assert plan["stratified_available"] is True
    allocation = plan["allocation"]
    assert sum(allocation.values()) == 8
    for repeat in range(3):
        for arm in ("a_stratified_baseline", "a_stratified_contrast"):
            for stratum_name, expected in allocation.items():
                robot = int(stratum_name.split("|")[0][1:]) - 1
                program = (int(stratum_name.split("|")[1][1:]) - 100) // 10
                drawn = [file_ids[int(position)] for position in plan[arm][repeat]]
                selected = [
                    fid
                    for fid in drawn
                    if index[fid]["predicted_robot_idx"] == robot
                    and index[fid]["predicted_program_idx"] == program
                ]
                assert len(selected) == expected


def test_allocation_respects_arm_caps_and_total() -> None:
    pooled = {0: 10, 1: 4, 2: 1}
    caps = {0: 5, 1: 2, 2: 0}
    allocation = audit.allocation_with_caps(pooled, caps, 6)
    assert sum(allocation.values()) == 6
    for key, value in allocation.items():
        assert value <= caps[key]
        assert value <= pooled[key]
    assert audit.allocation_with_caps(pooled, caps, 8) == {}


def test_infeasible_matched_allocation_abstains_without_raising() -> None:
    baseline = [f"b{position}" for position in range(4)]
    contrast = [f"c{position}" for position in range(4)]
    entries = {fid: (0, 0) for fid in baseline[:3] + contrast[:1]}
    entries.update({fid: (1, 0) for fid in baseline[3:] + contrast[1:]})
    index = _index(entries)
    group = _group(baseline + contrast, np.zeros(8, dtype=np.int64))
    row = _row(baseline, contrast, "H|week|T", "BASELINE-VS-ELEVATED")
    plan = audit.design_plan(group=group, row=row, size=4, index=index, repeats=3)
    assert plan["n_common_strata"] == 2
    assert plan["stratified_available"] is False
    assert plan["a_stratified_baseline"] is None and plan["a_stratified_contrast"] is None
    assert "cannot supply the matched size" in str(plan["unavailable_reason"])
    assert plan["a_srs_baseline"].shape == (3, 4)
    assert audit.design_summary(None, 4, audit.ABSTAIN_NO_COMMON_STRATA)["status"] == (
        audit.ABSTAIN_NO_COMMON_STRATA
    )
    assert audit.delta_summary(None, np.zeros(3, dtype=np.int64), 4, audit.ABSTAIN_NO_COMMON_STRATA)[
        "status"
    ] == audit.ABSTAIN_NO_COMMON_STRATA


def test_sample_size_effect_abstains_when_stratified_design_is_missing() -> None:
    designs = {
        audit.DESIGN_FULL_ARM: {
            "status": audit.COMPUTED,
            "per_seed": {"181801": {"4": 0.1, "8": 0.2, "16": 0.3}},
        },
        audit.DESIGN_SRS: {
            "status": audit.COMPUTED,
            "per_seed": {
                "181801": {"per_K": {"4": {"median": 0.08}, "8": {"median": 0.15}, "16": {"median": 0.25}}}
            },
        },
        audit.DESIGN_STRATIFIED: {"status": audit.ABSTAIN_NO_COMMON_STRATA, "reason": "infeasible"},
    }
    record = audit.sample_size_effect_record(designs, 181801, 8)
    assert record["status"] == audit.ABSTAIN_NO_COMMON_STRATA
    assert record["matched_stratified"] is None
    assert record["full_minus_matched_stratified"] is None
    assert record["full_minus_matched_srs"] == pytest.approx(0.2 - 0.15)
    assert record["matched_srs_minus_stratified"] is None


def test_cell_size_is_the_minimum_over_the_cell_comparisons() -> None:
    rows = [
        _row([f"f{index}" for index in range(40)], [f"g{index}" for index in range(38)], "H|week|A", "BASELINE-VS-ELEVATED"),
        _row([f"f{index}" for index in range(40)], [f"g{index}" for index in range(52)], "H|week|B", "BASELINE-VS-ELEVATED"),
        _row([f"f{index}" for index in range(60)], [f"h{index}" for index in range(60)], "H|week|C", "BASELINE-VS-BASELINE"),
    ]
    cells = audit.cell_sizes(rows)
    cell = cells["H|week"]
    assert cell["size"] == 38
    assert cell["status"] == audit.COMPUTED
    assert cell["control_identity"] == "H|week|C"
    assert cell["retained_arm_sizes"] == [38, 40, 52, 60]


def test_cell_abstains_below_the_inherited_period_floor() -> None:
    rows = [
        _row([f"f{index}" for index in range(20)], [f"g{index}" for index in range(20)], "H|week|A", "BASELINE-VS-ELEVATED"),
        _row([f"f{index}" for index in range(20)], [f"h{index}" for index in range(20)], "H|week|C", "BASELINE-VS-BASELINE"),
    ]
    cells = audit.cell_sizes(rows)
    assert cells["H|week"]["size"] == 20
    assert cells["H|week"]["status"] == audit.ABSTAIN_SHORTFALL


def test_occupancy_tv_is_not_additive_over_families() -> None:
    file_ids = [f"b{position}" for position in range(10)] + [
        f"c{position}" for position in range(10)
    ]
    labels = np.array(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] + [0] * 6 + [1] * 4, dtype=np.int64
    )
    group = _group(file_ids, labels)
    baseline = file_ids[:10]
    contrast = file_ids[10:]
    family_of = {"stuck": contrast[:6], "drift": contrast[6:]}
    result = audit.tv_nonadditivity(
        labels=labels,
        group=group,
        row=_row(baseline, contrast, "H|week|T", "BASELINE-VS-ELEVATED"),
        family_of=family_of,
        k=2,
    )
    assert result["tv_total"] == pytest.approx(0.1)
    assert result["tv_additive_sum"] == pytest.approx(0.5)
    assert result["nonadditivity_gap"] == pytest.approx(0.4)
    assert result["nonadditivity_gap"] > 0.0


def test_composition_flags_disclose_sparse_and_one_sided_strata() -> None:
    index = _index({fid: (0, 0) for fid in ["n1", "n2", "a1", "a2"]})
    index["a3"] = {
        "file_id": "a3",
        "predicted_robot_idx": 1,
        "predicted_program_idx": 2,
        "file_label": "abnormal",
        "annotation": {"family": "subtle_drift", "severity": 0.5, "abnormal_step_fraction": 0.2, "duration_steps": 10},
        "waveform_sha256": "a3",
    }
    index["a4"] = {
        "file_id": "a4",
        "predicted_robot_idx": 0,
        "predicted_program_idx": 0,
        "file_label": "abnormal",
        "annotation": {"family": "stuck", "severity": 0.8, "abnormal_step_fraction": 0.9, "duration_steps": 40},
        "waveform_sha256": "a4",
    }
    baseline = audit.arm_composition(["n1", "n2"], index)
    contrast = audit.arm_composition(["a1", "a2", "a3", "a4"], index)
    flags = audit.composition_flags(baseline, contrast)
    assert "subtle_drift" in flags["one_sided_families"]
    assert "stuck" in flags["one_sided_families"]
    assert "ordinary" not in flags["one_sided_families"]
    assert "subtle_drift" in flags["sparse_strata"]
    assert baseline["n_ordinary"] == 2 and contrast["n_abnormal"] == 2
    assert contrast["severity_bin_counts"]["[0.45,0.60)"] == 1
    assert contrast["severity_bin_counts"]["[0.75,0.90)"] == 1
    assert contrast["duration_steps_median"] == pytest.approx(25.0)
    assert any("severity" in entry for entry in flags["confounded_bins"])


def test_stratum_id_is_the_label_blind_robot_program_key() -> None:
    assert audit.stratum_id({"predicted_robot_idx": 0, "predicted_program_idx": 0}) == "R01|P100"
    assert audit.stratum_id({"predicted_robot_idx": 4, "predicted_program_idx": 7}) == "R05|P170"


def test_context_mix_tv_is_zero_for_identical_mixes_and_one_for_disjoint() -> None:
    index = _index(
        {
            "a1": (0, 0),
            "a2": (1, 0),
            "b1": (1, 0),
            "b2": (0, 0),
            "c1": (0, 0),
            "c2": (0, 0),
        }
    )
    assert audit.context_mix_tv(["a1", "a2"], ["b1", "b2"], index) == 0.0
    assert audit.context_mix_tv(["c1", "c2"], ["a2", "b1"], index) == pytest.approx(1.0)
    assert audit.mix_tv({"x": 0}, {"y": 1}) is None


def test_summary_of_quanta_counts_directions_exactly() -> None:
    summary = audit.summary_of_quanta(np.array([0, 0, 1, -1], dtype=np.int64), 4)
    assert summary["n"] == 4
    assert summary["fraction_strictly_positive"] == pytest.approx(0.25)
    assert summary["fraction_zero"] == pytest.approx(0.5)
    assert summary["fraction_strictly_negative"] == pytest.approx(0.25)
    assert summary["median"] == 0.0
    assert summary["max"] == pytest.approx(0.25)
    assert summary["max_quanta"] == 1


def test_delta_summary_of_identical_arms_is_exactly_zero() -> None:
    transition = np.array([3, 4, 5], dtype=np.int64)
    summary = audit.delta_summary(transition, transition.copy(), 8, audit.ABSTAIN_NO_COMMON_STRATA)
    assert summary["status"] == audit.COMPUTED
    assert summary["fraction_strictly_positive"] == 0.0
    assert summary["fraction_zero"] == 1.0
    assert summary["median"] == 0.0


def test_seed_median_ignores_abstentions() -> None:
    assert audit.seed_median([None, 1.0, 3.0]) == pytest.approx(2.0)
    assert audit.seed_median([None, None]) is None


def test_manifest_exposure_maps_reads_only_npy_headers(tmp_path: Path) -> None:
    np.savez(tmp_path / "sample.npz", x=np.zeros((2, 40), dtype=np.float32))
    manifest = {
        "__manifest_path__": str(tmp_path / "corpus-manifest.json"),
        "files": [
            {
                "file_id": "f1",
                "end_time_rel_s": 1000.0,
                "relpath": "sample.npz",
            },
            {"file_id": "f2", "end_time_rel_s": 2000.0, "relpath": "missing.npz"},
        ],
    }
    end_time_of, duration_of, header_reads = audit.manifest_exposure_maps(manifest)
    assert end_time_of == {"f1": 1000.0, "f2": 2000.0}
    assert duration_of == {"f1": 40.0}
    assert header_reads == 1


def test_retained_exposure_reconstruction_matches_the_accepted_definition() -> None:
    end_time_of = {"a1": 100 * DAY, "a2": 200 * DAY, "b1": 150 * DAY, "b2": 200 * DAY}
    duration_of = {"a1": 20 * DAY, "a2": 20 * DAY, "b1": 100 * DAY, "b2": 100 * DAY}
    ratio = readiness.retained_exposure(["a1", "a2"], ["b1", "b2"], end_time_of, duration_of)
    assert ratio == pytest.approx(20.0 / 50.0)
    assert readiness.retained_exposure(["a1"], ["b1"], end_time_of, {}) is None


def test_lineage_refuses_a_tampered_audit_copy(tmp_path: Path) -> None:
    corpus = REPO / "data" / "generated" / "sprint24_controlled_v2"
    preflight = (
        REPO
        / "artifacts"
        / "sprint-24"
        / "evidence"
        / "s24-7-m4-a01"
        / "g2-accepted"
        / "preflight.json"
    )
    contract = REPO / "artifacts" / "sprint-24" / "evidence" / "s24-7-m10-a02" / "prospective-contract.json"
    if not (corpus / "corpus-manifest.json").is_file() or not preflight.is_file() or not contract.is_file():
        pytest.skip("accepted corpus/contract artifacts are not present in this checkout")
    tampered = tmp_path / "corpus-audit.json"
    source = REPO / "artifacts" / "sprint-24" / "evidence" / "s24-7-m9-a01" / "corpus-audit.json"
    payload = source.read_text(encoding="utf-8")
    tampered.write_text(payload.replace('"n_pairs"', '"n_pairs_x"', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="REFUSED-DIGEST-DRIFT"):
        audit.verify_lineage(corpus, tampered, contract, preflight)


def test_quanta_are_integral_so_no_float_tie_can_be_positive() -> None:
    summary = audit.summary_of_quanta(np.zeros(10, dtype=np.int64), 55)
    assert summary["fraction_strictly_positive"] == 0.0
    assert math.isclose(summary["median"], 0.0, abs_tol=1e-15)


def test_seed_direction_distinguishes_zero_from_negative() -> None:
    zero = audit.seed_direction([0.0, 0.0, 0.0])
    assert zero["signs"] == [0]
    assert zero["agreement"] is False, "an all-zero triple is not directional agreement"
    assert zero["all_zero"] is True
    mixed = audit.seed_direction([0.0, -1.0, -2.0])
    assert mixed["signs"] == [-1, 0], "zero must not be folded into the negative sign"
    assert mixed["agreement"] is False and mixed["all_zero"] is False
    assert audit.seed_direction([-1.0, -2.0, -3.0])["agreement"] is True
    assert audit.seed_direction([3.0, 2.0, 1.0])["agreement"] is True
    assert audit.seed_direction([1.0, 2.0])["agreement"] is False


def test_matched_size_entry_abstains_when_the_stratified_plan_is_unavailable() -> None:
    baseline = [f"b{position}" for position in range(4)]
    contrast = [f"c{position}" for position in range(4)]
    entries = {fid: (0, 0) for fid in baseline[:3] + contrast[:1]}
    entries.update({fid: (1, 0) for fid in baseline[3:] + contrast[1:]})
    index = _index(entries)
    file_ids = baseline + contrast
    group = _group(file_ids, np.zeros(len(file_ids), dtype=np.int64))
    row = _row(baseline, contrast, "H|week|T", "BASELINE-VS-ELEVATED")
    plan = audit.design_plan(group=group, row=row, size=4, index=index, repeats=3)
    assert plan["a_stratified_baseline"] is None
    seeds = readiness.FROZEN_ENCODER_SEEDS
    labels = {
        f"H|{seed}": {k: np.zeros(len(file_ids), dtype=np.int64) for k in audit.CLUSTER_COUNTS}
        for seed in seeds
    }
    groups = {f"H|{seed}": group for seed in seeds}
    end_time_of = {fid: float(100 + position) for position, fid in enumerate(file_ids)}
    duration_of = {fid: 10.0 for fid in file_ids}
    stratified: dict = {}
    srs: dict = {}
    full: dict = {}
    entry = audit.matched_size_entry(
        row=row,
        cell={"status": audit.COMPUTED, "size": 4},
        key="H|week",
        size=4,
        plan=plan,
        labels=labels,
        groups=groups,
        index=index,
        end_time_of=end_time_of,
        duration_of=duration_of,
        stratified_store=stratified,
        srs_store=srs,
        full_store=full,
    )
    assert entry["designs"][audit.DESIGN_STRATIFIED]["status"] == audit.ABSTAIN_NO_COMMON_STRATA
    assert "per_seed" not in entry["designs"][audit.DESIGN_STRATIFIED]
    assert entry["designs"][audit.DESIGN_SRS]["status"] == audit.COMPUTED
    assert entry["designs"][audit.DESIGN_FULL_ARM]["status"] == audit.COMPUTED
    assert entry["sample_size_effect"][str(seeds[0])]["8"]["status"] == (
        audit.ABSTAIN_NO_COMMON_STRATA
    )
    assert not stratified, "an unavailable design must contribute no repeats"
    assert len(srs) == len(seeds) * len(audit.CLUSTER_COUNTS)
    assert len(full) == len(seeds) * len(audit.CLUSTER_COUNTS)


def test_matched_size_entry_abstains_for_a_subfloor_cell() -> None:
    baseline = [f"b{position}" for position in range(4)]
    contrast = [f"c{position}" for position in range(4)]
    index = _index({fid: (0, 0) for fid in baseline + contrast})
    file_ids = baseline + contrast
    group = _group(file_ids, np.zeros(len(file_ids), dtype=np.int64))
    row = _row(baseline, contrast, "H|week|T", "BASELINE-VS-ELEVATED")
    seeds = readiness.FROZEN_ENCODER_SEEDS
    labels = {
        f"H|{seed}": {k: np.zeros(len(file_ids), dtype=np.int64) for k in audit.CLUSTER_COUNTS}
        for seed in seeds
    }
    entry = audit.matched_size_entry(
        row=row,
        cell={"status": audit.ABSTAIN_SHORTFALL, "size": 20},
        key="H|week",
        size=20,
        plan=None,
        labels=labels,
        groups={f"H|{seed}": group for seed in seeds},
        index=index,
        end_time_of={fid: float(position) for position, fid in enumerate(file_ids)},
        duration_of={fid: 5.0 for fid in file_ids},
        stratified_store={},
        srs_store={},
        full_store={},
    )
    for design in audit.DESIGNS:
        assert entry["designs"][design]["status"] == audit.ABSTAIN_SHORTFALL
    assert entry["sample_size_effect"]["status"] == audit.ABSTAIN_SHORTFALL


def test_descriptive_associations_skip_abstained_transitions_without_raising() -> None:
    seeds = readiness.FROZEN_ENCODER_SEEDS
    diagnostics = [
        {
            "horizon": "week",
            "comparison_type": "BASELINE-VS-ELEVATED",
            "cell_status": audit.COMPUTED,
            "comparison_identity": f"H|week|T{position}",
            "mix_tv": {
                "family": 0.20 + 0.05 * position,
                "severity_bins": 0.30 + 0.02 * position,
                "anomalous_fraction_bins": 0.25 + 0.01 * position,
                "context": 0.0,
            },
            "burden_contrast_gap_after": 0.10 + 0.02 * position,
        }
        for position in range(4)
    ]
    matched: dict[str, dict[str, object]] = {
        "H|week|T0": {
            "designs": {
                audit.DESIGN_STRATIFIED: {
                    "status": audit.ABSTAIN_NO_COMMON_STRATA,
                    "reason": "infeasible allocation",
                }
            }
        }
    }
    for position in range(1, 4):
        matched[f"H|week|T{position}"] = {
            "designs": {
                audit.DESIGN_STRATIFIED: {
                    "status": audit.COMPUTED,
                    "per_seed": {
                        str(seed): {
                            "per_K": {
                                "8": {"status": audit.COMPUTED, "median": 0.05 * position + 0.001 * index}
                            }
                        }
                        for index, seed in enumerate(seeds)
                    },
                }
            }
        }
    result = audit.descriptive_associations(diagnostics, matched)
    block = result["week"]
    assert block["n_transitions"] == 4
    assert block["n_used"] == 3
    assert block["n_excluded"] == 1
    assert block["excluded_transitions"] == [
        {"comparison_identity": "H|week|T0", "reason": audit.ABSTAIN_NO_COMMON_STRATA}
    ]
    assert block["features"]["family_mix_tv"]["n"] == 3
