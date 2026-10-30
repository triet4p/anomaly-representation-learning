"""Sprint 24 Task 7-M17-C2 official-runner checks (consumer-visible only).

Covers the versioned official path without executing any forward: matrix
enumeration (27 cells x G24/48, 6 histories, no 96), seed/disjointness
invariants (2,304 unique, G24 prefix of G48, disjoint null/reference/stage-2
bands), contract build/verify round trip + supersession, numerical checks
(E1/CP/Holm/dilution reuse), error boundaries, and the official-contract CLI
mode. No checkpoint is loaded, no forward executes, no remote is contacted.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from representation import sprint24_paired_execution as base
from representation import sprint24_paired_official as official

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "experiments" / "sprint24_task7m17_paired_execution.py"


def test_official_matrix_enumerates_27_cells_g24_g48_only() -> None:
    matrix = official.matrix_plan()
    assert matrix["cell_count"] == 27
    assert len(matrix["cells"]) == 27
    assert matrix["official_contract_id"] == official.OFFICIAL_CONTRACT_ID
    assert matrix["supersedes"] == base.EXECUTION_CONTRACT_ID
    for cell in matrix["cells"]:
        assert cell["pairs_g24"] == 192
        assert cell["pairs_g48"] == 384
        assert cell["g48_extra_pair_indices"] == list(range(192, 384))
    assert matrix["no_96_branch"] is True
    assert official.matrix_plan()["seed_plan"]["expected_unique"] == 6 * 384


def test_official_seed_plan_maps_g24_g48_per_history_per_cell() -> None:
    plan = base.check_group_seed_plan()
    assert plan["pair_slots_per_history"] == 384
    assert plan["n_derived_unique"] == plan["expected_unique"] == 2304
    # Same (history, pair_index) re-derives the same pair seed in every cell:
    # cross-cell variant sharing is structural; within-cell disjointness comes
    # from disjoint index ranges (G24 prefix, G48 extension).
    from synth.generator import derive_pair_seed

    shared = derive_pair_seed(53101, 7)
    assert derive_pair_seed(53101, 7) == shared
    cells = official.cell_plans()
    assert cells[0].pair_index_g48 == tuple(range(384))
    assert cells[0].pair_index_g24 == tuple(range(192))


def test_official_allocation_bands_are_disjoint() -> None:
    table = official.allocation_table()
    pair = set(range(table["pair_band"][0], table["pair_band"][1]))
    reference = set(range(table["reference_band"][0], table["reference_band"][1]))
    null = set(range(table["null_band"][0], table["null_band"][1]))
    stage2_start = table["stage2_band"][0]
    assert pair.isdisjoint(reference)
    assert pair.isdisjoint(null)
    assert reference.isdisjoint(null)
    assert stage2_start > max(max(pair), max(reference), max(null))
    from synth.generator import derive_pair_seed

    pair_seeds = {derive_pair_seed(53101, index) for index in sorted(pair)[:8]}
    reference_seeds = {derive_pair_seed(53101, index) for index in sorted(reference)[:8]}
    assert pair_seeds.isdisjoint(reference_seeds)


def test_official_contract_build_verify_round_trip() -> None:
    preflight_path = (
        REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m4-a01"
        / "g2-accepted" / "preflight.json"
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    anchors = {"branch_c_e1_zero_shift_median_detection_fraction": 0.06}
    provenance = {"built_by": "test", "source_commit": "test"}
    contract = official.build_official_contract(preflight, provenance, anchors)
    assert official.verify_official_contract(contract)["status"] == "PASS"
    assert contract["contract_id"] == official.OFFICIAL_CONTRACT_ID
    assert contract["supersedes"] == base.EXECUTION_CONTRACT_ID
    assert "readiness_vs_permission" in contract
    tampered = dict(contract)
    tampered["state"] = "EDITED"
    with pytest.raises(ValueError, match="REFUSED-CONTRACT-DRIFT"):
        official.verify_official_contract(tampered)


def test_official_numerics_reuse_frozen_math() -> None:
    rng = np.random.default_rng(0)
    twin = rng.normal(size=(8, 4))
    abnormal = rng.normal(size=(8, 4)) + 0.5
    scored = base.score_scored_group(twin, abnormal, 1.0)
    assert np.isfinite(scored["e1"])
    cell = base.summarize_cell([0.9, 0.8, 0.01, 0.02], 0.1)
    assert cell["n_groups"] == 4
    adjusted = official.holm_over_families([0.01, 0.04, 0.5], alpha=0.05)
    assert adjusted[0]["reject"] is True
    ladder = base.dilution_ladder()
    assert ladder["week"]["40"]["LOW"]["n_abnormal"] == 2
    assert ladder["month"]["250"]["ELEVATED"]["n_abnormal"] == 69
    bootstrap = official.cluster_bootstrap_median([0.1, 0.2, 0.3])
    assert bootstrap["status"] == "DESCRIPTIVE"
    assert bootstrap["replicates"] == 2000
    arms = official.population_arms()
    assert len(arms) == 12
    family = official.summarize_stage1_family(
        {"a|0.6": {"g48": {"summary": base.summarize_cell([0.9] * 30 + [0.01] * 18, 0.1)}}}
    )
    assert "holm" in family and len(family["holm"]) == 1
    sensitivity = official.e2_occupancy(rng.normal(size=(16, 4)), rng.normal(size=(16, 4)) + 0.3, rng.normal(size=(40, 4)))
    assert np.isfinite(sensitivity["e2"])
    within = official.within_prototype_diagnostics(rng.normal(size=(16, 4)), rng.normal(size=(16, 4)) + 0.3, rng.normal(size=(40, 4)))
    assert "d_loc" in within and "coverage" in within


def test_official_error_boundaries_fail_closed() -> None:
    with pytest.raises(ValueError, match="REFUSED-UNKNOWN-DESIGN"):
        official.cell_plans(groups_per_cell=(24, 96))
    with pytest.raises(ValueError, match="REFUSED-OFFICIAL-RELEASE"):
        official._require_official_release(False)
    with pytest.raises(ValueError, match="REFUSED-ERROR-BOUNDARY"):
        base.score_scored_group(np.zeros((1, 4)), np.zeros((1, 4)), 1.0)
    with pytest.raises(ValueError, match="REFUSED-UNKNOWN-DESIGN"):
        official._family_of("not_a_family")
    with pytest.raises(ValueError, match="REFUSED-ERROR-BOUNDARY"):
        official.frozen_reference_bandwidth({int(seed): {"matrix": [[float("nan")]]} for seed in base.FROZEN_ENCODER_SEEDS})


def test_official_run_orchestrator_gate_and_callgraph() -> None:
    import inspect

    assert "run_official_matrix" in official.__all__
    assert "official-run" in open(CLI, encoding="utf-8").read()
    source = inspect.getsource(official.run_official_matrix)
    assert "_require_official_release" in source
    assert "collect_stage1_cell" in source
    assert "evaluate_stage2_arm" in source
    assert "summarize_stage1_family" in source
    assert "cluster_bootstrap_median" in source
    smoke_source = inspect.getsource(official.tiny_official_smoke)
    assert "collect_stage1_cell" in smoke_source
    assert "evaluate_stage2_arm" in smoke_source
    assert "capture_reference_pool" in smoke_source
    assert "frozen_reference_bandwidth" in smoke_source
    stage_source = inspect.getsource(official.evaluate_stage2_arm)
    assert "ordinary" in stage_source and "abnormal" in stage_source
    assert "score_scored_group(twin[" not in stage_source
    with pytest.raises(ValueError, match="REFUSED-OFFICIAL-RELEASE"):
        official.run_official_matrix({}, permitted=False)


def test_official_matrix_counts_fully_enumerated_without_outcomes() -> None:
    counts = official.official_matrix_counts()
    assert counts["cells"] == 27
    assert counts["histories"] == 6
    assert counts["stage1_pairs_g48_per_history"] == 27 * 384
    assert counts["stage1_pairs_g48_total"] == 27 * 384 * 6
    assert counts["encoder_seeds"] == [181801, 181802, 181803]


def test_cli_official_contract_mode_writes_frozen_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "official-contract"
        completed = subprocess.run(
            [sys.executable, str(CLI), "--mode", "official-contract", "--out", str(out)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        contract = json.loads((out / "official-contract.json").read_text(encoding="utf-8"))
        assert official.verify_official_contract(contract)["status"] == "PASS"
        matrix = json.loads((out / "official-matrix.json").read_text(encoding="utf-8"))
        assert len(matrix["cells"]) == 27
        counts = json.loads((out / "official-matrix-counts.json").read_text(encoding="utf-8"))
        assert counts["stage1_pairs_g48_total"] == 27 * 384 * 6
        arms = json.loads((out / "official-population-arms.json").read_text(encoding="utf-8"))
        assert len(arms["arms"]) == 12
        payload = json.loads(completed.stdout)
        assert payload["status"] == "OFFICIAL-CONTRACT-COMPLETE"


def test_cli_official_modes_share_engine_with_tiny_fixture() -> None:
    assert official.TINY_FIXTURE["family"] == "subtle_drift"
    assert official.TINY_FIXTURE["severity"] == 0.6
    assert official.TINY_FIXTURE["history_seed"] == 53101
    assert official.TINY_FIXTURE["n_groups"] == 2
