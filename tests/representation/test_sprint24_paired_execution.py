"""Sprint 24 Task 7-M17 paired-execution readiness checks (consumer-visible only).

Covers the frozen option-A execution contract and runner on isolated local
inputs: grouped seed plan, group-level operating characteristics (24/48
only, no 96), contract build/verify round trip, K=8 grouped E1 scoring
boundaries, response-fraction summaries, and the public CLI refusal/smoke
boundaries. No checkpoint is loaded, no forward executes, no remote is
contacted, no waveform signal beyond tiny synthetic smoke inputs is read.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from representation import sprint24_paired_execution as execution

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "experiments" / "sprint24_task7m17_paired_execution.py"


def test_group_grid_is_24_48_only() -> None:
    assert list(execution.STAGE1_GROUP_GRID) == [24, 48]
    assert execution.STAGE1_FUTILITY_LOOK_G == 24
    assert execution.GROUP_SIZE_K == 8
    summary = execution.execution_summary()
    assert summary["pairs_per_cell"] == {"24": 192, "48": 384}
    assert summary["no_96_branch"] is True


def test_group_seed_plan_passes_and_covers_full_span() -> None:
    plan = execution.check_group_seed_plan()
    assert plan["status"] == "PASS"
    assert plan["pair_slots_per_history"] == 384
    assert plan["n_derived_unique"] == plan["expected_unique"] == 6 * 384


def test_group_power_spot_values_are_group_level() -> None:
    grid = execution.group_power_grid()
    assert len(grid["sign_power_groups"]) == 2 * 3
    assert {entry["n_pairs"] for entry in grid["sign_power_groups"]} == {24, 48}
    assert 96 not in {entry["n_pairs"] for entry in grid["sign_power_groups"]}
    by_key = {(entry["n_pairs"], entry["p_win_assumed"]): entry for entry in grid["sign_power_groups"]}
    assert by_key[(24, 0.65)]["power"] == 0.2106
    assert by_key[(48, 0.75)]["power"] == 0.9296
    assert by_key[(48, 0.85)]["power"] == 0.9996
    assert by_key[(48, 0.75)]["critical_value"] == 32
    for entry in grid["sign_power_groups"]:
        assert "scored group" in entry["independent_unit"]


def test_sequential_plan_has_no_96_branch() -> None:
    plan = execution.sequential_multiplicity_plan()
    assert plan["interim_look"]["at_G"] == 24
    assert "NOT AUTHORIZED" in plan["expansion"]["beyond_48"]
    assert "96" not in str(plan["expansion"]["rule"])


def test_contract_build_verify_round_trip() -> None:
    import json as _json

    preflight = _json.loads(
        (REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m4-a01"
         / "g2-accepted" / "preflight.json").read_text(encoding="utf-8")
    )
    contract = execution.build_execution_contract(
        {"anchor": 0.06, "warning": "responsiveness only"},
        preflight,
        {"built_by": "test"},
    )
    assert execution.verify_execution_contract(contract)["status"] == "PASS"
    tampered = dict(contract)
    tampered["state"] = "EDITED"
    with pytest.raises(ValueError, match="REFUSED-CONTRACT-DRIFT"):
        execution.verify_execution_contract(tampered)


def test_score_scored_group_requires_k8() -> None:
    rng = np.random.default_rng(0)
    scored = execution.score_scored_group(rng.normal(size=(8, 4)), rng.normal(size=(8, 4)), 1.0)
    assert scored["k_per_arm"] == 8
    assert np.isfinite(scored["e1"])
    with pytest.raises(ValueError, match="REFUSED-ERROR-BOUNDARY"):
        execution.score_scored_group(rng.normal(size=(1, 4)), rng.normal(size=(1, 4)), 1.0)
    bad = rng.normal(size=(8, 4))
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="REFUSED-ERROR-BOUNDARY"):
        execution.score_scored_group(rng.normal(size=(8, 4)), bad, 1.0)


def test_summarize_cell_reports_inconclusive_miss() -> None:
    cell = execution.summarize_cell([0.5] * 6 + [0.01] * 18, 0.1)
    assert cell["n_groups"] == 24
    assert cell["response_fraction"] == 0.25
    assert cell["bar_met"] is False
    assert "INCONCLUSIVE" in cell["reading"]


def test_dilution_ladder_follows_m13_rounding() -> None:
    ladder = execution.dilution_ladder()
    assert ladder["week"]["40"]["ELEVATED"]["n_abnormal"] == 11
    assert ladder["month"]["250"]["ELEVATED"]["n_abnormal"] in (68, 69)
    assert ladder["week"]["40"]["LOW"]["n_abnormal"] == 2


def test_holm_wrapper_matches_declared_contrasts() -> None:
    adjusted = execution.holm_over_families([0.01, 0.04, 0.5], alpha=0.05)
    assert adjusted[0]["reject"] is True
    assert adjusted[1]["reject"] is False


def test_cli_contract_mode_writes_frozen_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "contract"
        completed = subprocess.run(
            [sys.executable, str(CLI), "--mode", "contract", "--out", str(out)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert (out / "execution-contract.json").is_file()
        assert (out / "independent-execution-check.json").is_file()
        assert (out / "seed-plan.json").is_file()
        assert (out / "run.json").is_file()
        contract = json.loads((out / "execution-contract.json").read_text(encoding="utf-8"))
        assert execution.verify_execution_contract(contract)["status"] == "PASS"
        check = json.loads((out / "independent-execution-check.json").read_text(encoding="utf-8"))
        assert check["status"] == "PASS"
        assert check["grid_is_24_48"] is True
        assert check["n_96_entries"] == 0


def test_cli_refuses_overwrite() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "contract"
        out.mkdir()
        (out / "sentinel.txt").write_text("x", encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(CLI), "--mode", "contract", "--out", str(out)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert completed.returncode != 0
        assert "REFUSED-OVERWRITE" in (completed.stdout + completed.stderr)


def test_cli_smoke_completes_with_zero_forwards() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI), "--mode", "smoke"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    payload = json.loads(completed.stdout)
    assert payload["status"] == "SMOKE-COMPLETE"
    assert payload["forwards_executed_this_run"] == 0
    assert payload["model_loads"] == 0
    assert payload["cumulative_forwards_after_run"] == 77653
    assert payload["checks"]["singleton_refused"] is True
    assert payload["checks"]["deterministic_replay"] is True
