"""Sprint 24 Task 7-M15 paired-decision experiment CLI (design task only).

Public entry points (no generation, no inference, no forwards):

1. ``--mode smoke`` runs the tiny public-path feasibility smoke: deterministic
   replay of ``generate_planned_file`` on real synthesis machinery, the paired
   base invariants inside the anomaly path, the documented twin API gap, and
   temp-output-only waveform writes. Official corpus generation is refused.
2. ``--mode contract`` verifies the seed plan, builds the proposed
   machine-readable contract with its embedded operating characteristics, and
   writes it beside a canonical digest plus the calculator check record.

A non-empty ``--out`` is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from representation import sprint24_controlled_data_execution as execution  # noqa: E402
from representation import sprint24_paired_decision_experiment as design  # noqa: E402
from representation.sprint24_context_audit import canonical_json  # noqa: E402


def write_json(path: Path, payload: object) -> str:
    text = canonical_json(payload) + "\n"
    path.write_text(text, encoding="utf-8", newline="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _refused(reason: str, detail: str) -> ValueError:
    return ValueError(f"{reason}: {detail}")


def _require_fresh_out(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise _refused("REFUSED-OVERWRITE", f"{path} is non-empty")
    path.mkdir(parents=True, exist_ok=True)


def _waveform_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def run_smoke() -> dict[str, Any]:
    """Tiny public-path feasibility smoke (temp outputs only)."""
    generator = execution.SessionGenerator(execution.SynthConfig())
    seed = 42401
    slot_abnormal = {
        "seed": seed,
        "requested_label": "abnormal",
        "requested_family": "subtle_drift",
        "requested_severity": 0.6,
        "slot_id": "s24p-smoke-abnormal",
    }
    slot_normal = {
        "seed": 42402,
        "requested_label": "normal",
        "slot_id": "s24p-smoke-normal",
    }
    abnormal_first = execution.generate_planned_file(
        dict(slot_abnormal), generator, stream="SMOKE-THROWAWAY"
    )
    abnormal_replay = execution.generate_planned_file(
        dict(slot_abnormal), generator, stream="SMOKE-THROWAWAY"
    )
    normal_first = execution.generate_planned_file(
        dict(slot_normal), generator, stream="SMOKE-THROWAWAY"
    )
    normal_replay = execution.generate_planned_file(
        dict(slot_normal), generator, stream="SMOKE-THROWAWAY"
    )
    replay_abnormal = bool(np.array_equal(abnormal_first.x, abnormal_replay.x))
    replay_normal = bool(np.array_equal(normal_first.x, normal_replay.x))
    execution.verify_encoder_label_free(abnormal_first, "s24p-smoke-abnormal")
    execution.verify_encoder_label_free(normal_first, "s24p-smoke-normal")
    annotation = execution.trace_annotation(abnormal_first, slot_abnormal)

    low = execution.generate_planned_file(
        {
            "seed": seed,
            "requested_label": "abnormal",
            "requested_family": "subtle_drift",
            "requested_severity": 0.35,
            "slot_id": "s24p-smoke-sev-lo",
        },
        generator,
        stream="SMOKE-THROWAWAY",
    )
    high = execution.generate_planned_file(
        {
            "seed": seed,
            "requested_label": "abnormal",
            "requested_family": "subtle_drift",
            "requested_severity": 0.85,
            "slot_id": "s24p-smoke-sev-hi",
        },
        generator,
        stream="SMOKE-THROWAWAY",
    )
    other_family = execution.generate_planned_file(
        {
            "seed": seed,
            "requested_label": "abnormal",
            "requested_family": "realistic_stuck",
            "requested_severity": 0.6,
            "slot_id": "s24p-smoke-fam",
        },
        generator,
        stream="SMOKE-THROWAWAY",
    )
    severity_invariant = (
        low.T == high.T
        and (low.robot_idx, low.program_idx) == (high.robot_idx, high.program_idx)
        and [
            (item.start, item.end) for item in low.regime_sequence
        ]
        == [(item.start, item.end) for item in high.regime_sequence]
    )
    family_invariant = (
        low.T == other_family.T
        and (low.robot_idx, low.program_idx)
        == (other_family.robot_idx, other_family.program_idx)
    )
    same_seed_normal = execution.generate_planned_file(
        {
            "seed": seed,
            "requested_label": "normal",
            "slot_id": "s24p-smoke-sameseed-normal",
        },
        generator,
        stream="SMOKE-THROWAWAY",
    )
    gap_same_seed_diverges = (
        same_seed_normal.T != abnormal_first.T
        or (same_seed_normal.robot_idx, same_seed_normal.program_idx)
        != (abnormal_first.robot_idx, abnormal_first.program_idx)
    )

    with tempfile.TemporaryDirectory(prefix="s24p-m15-smoke-") as tmpdir:
        first_path = Path(tmpdir) / "abnormal.npz"
        second_path = Path(tmpdir) / "normal.npz"
        np.savez_compressed(first_path, x=abnormal_first.x)
        np.savez_compressed(second_path, x=normal_first.x)
        tmp_files = sorted(Path(tmpdir).name for _ in [0])
        temp_report = {
            "dir_prefix": "s24p-m15-smoke-",
            "wrote": ["abnormal.npz", "normal.npz"],
            "abnormal_bytes": int(first_path.stat().st_size),
            "normal_bytes": int(second_path.stat().st_size),
            "note": "TemporaryDirectory cleaned on exit; no official corpus written.",
        }
        _ = tmp_files

    passed = bool(
        replay_abnormal
        and replay_normal
        and severity_invariant
        and family_invariant
        and gap_same_seed_diverges
    )
    return {
        "task": "S24-7-M15",
        "design_id": design.DESIGN_ID,
        "status": "SMOKE-COMPLETE" if passed else "SMOKE-FAILED",
        "forwards_executed_this_run": 0,
        "model_loads": 0,
        "official_corpus_written": False,
        "replay": {
            "abnormal_byte_identical": replay_abnormal,
            "normal_byte_identical": replay_normal,
            "abnormal_sha256": _waveform_sha256(abnormal_first.x),
            "normal_sha256": _waveform_sha256(normal_first.x),
        },
        "invariants": {
            "severity_sweep_same_base": bool(severity_invariant),
            "family_sweep_same_base": bool(family_invariant),
            "severity_low_T": int(low.T),
            "severity_high_T": int(high.T),
            "family_other_T": int(other_family.T),
        },
        "api_gap": {
            "id": "GAP-PAIRED-TWIN-01",
            "same_seed_normal_T": int(same_seed_normal.T),
            "same_seed_abnormal_T": int(abnormal_first.T),
            "same_seed_normal_roster": [int(same_seed_normal.robot_idx), int(same_seed_normal.program_idx)],
            "same_seed_abnormal_roster": [int(abnormal_first.robot_idx), int(abnormal_first.program_idx)],
            "same_seed_public_paths_diverge": bool(gap_same_seed_diverges),
        },
        "label_free_encoder_inputs": True,
        "realized_annotation": {
            "family": annotation["family"],
            "requested_severity": annotation["requested_severity"],
            "duration_steps": annotation["duration_steps"],
            "abnormal_step_fraction": annotation["abnormal_step_fraction"],
        },
        "temp_outputs": temp_report,
    }


def run_contract(out: Path) -> dict[str, Any]:
    """Build and verify the proposed contract; write evidence files."""
    started = time.time()
    _require_fresh_out(out)
    seed_plan = design.check_seed_plan()
    anchors = {
        "source": "artifacts/sprint-24/evidence/s24-7-m14-a01/audit/aggregates.json",
        "branch_c_e1_zero_shift_median_detection_fraction": 0.06,
        "branch_c_zero_shift_records": "GLOBAL-SHIFT delta=0 vs null q95, per unit-seed",
        "branch_c_e1_response_grid": {
            "month160": {"0.10": 0.13, "0.25": 0.55, "0.50": 1.0},
            "week40": {"0.10": 0.07, "0.25": 0.17, "0.50": 0.54},
        },
    }
    provenance = {
        "design_module": "src/representation/sprint24_paired_decision_experiment.py",
        "criterion": design.CRITERION,
        "source_commit": source_commit(),
        "cumulative_forwards_unchanged": design.CUMULATIVE_FORWARDS_BEFORE,
        "generated_utc_note": "written by run record, not part of the contract digest",
    }
    contract = design.build_contract(anchors, provenance)
    verification = design.verify_contract(contract)
    contract_digest = str(verification["contract_sha256"])
    write_json(out / "proposed-contract.json", contract)
    independent = _independent_calculator_check(contract)
    independent_digest = write_json(out / "independent-calculator-check.json", independent)
    smoke = run_smoke()
    smoke_digest = write_json(out / "smoke.json", smoke)
    seed_digest = write_json(out / "seed-plan.json", seed_plan)
    run_record = {
        "task": "S24-7-M15",
        "design_id": design.DESIGN_ID,
        "contract_id": design.CONTRACT_ID,
        "source_commit": provenance["source_commit"],
        "contract_sha256": contract_digest,
        "verification": verification,
        "independent_calculator_check": {
            "digest": independent_digest,
            "status": independent["status"],
        },
        "smoke": {"digest": smoke_digest, "status": smoke["status"]},
        "seed_plan": {"digest": seed_digest, "status": seed_plan["status"]},
        "forwards_executed_this_run": 0,
        "model_loads": 0,
        "official_corpus_written": False,
        "cumulative_forwards_after_run": design.CUMULATIVE_FORWARDS_BEFORE,
        "elapsed_s": round(time.time() - started, 2),
    }
    run_digest = write_json(out / "run.json", run_record)
    return {
        "out": str(out),
        "contract_sha256": contract_digest,
        "run_sha256": run_digest,
        "status": "CONTRACT-COMPLETE",
    }


def _independent_calculator_check(contract: dict[str, Any]) -> dict[str, Any]:
    """Recompute the operating characteristics without importing the design math."""
    import math as _math

    checked: list[dict[str, Any]] = []
    failures = 0
    for entry in contract["operating_characteristics"]["sign_power"]:
        n_pairs = int(entry["n_pairs"])
        p_win = float(entry["p_win_assumed"])
        total = 2**n_pairs
        critical = None
        for candidate in range(n_pairs + 1):
            upper = sum(_math.comb(n_pairs, i) for i in range(candidate, n_pairs + 1))
            lower = sum(_math.comb(n_pairs, i) for i in range(0, n_pairs - candidate + 1))
            if (upper + lower) / total <= float(contract["operating_characteristics"]["alpha"]):
                critical = candidate
                break
        assert critical == int(entry["critical_value"]), "critical value drift"
        upper_power = sum(
            _math.comb(n_pairs, i) * (p_win**i) * ((1.0 - p_win) ** (n_pairs - i))
            for i in range(critical, n_pairs + 1)
        )
        lower_power = sum(
            _math.comb(n_pairs, i) * (p_win**i) * ((1.0 - p_win) ** (n_pairs - i))
            for i in range(0, n_pairs - critical + 1)
        )
        recalculated = round(upper_power + lower_power, 4)
        ok = recalculated == float(entry["power"])
        failures += 0 if ok else 1
        checked.append(
            {"n_pairs": n_pairs, "p_win": p_win, "recalculated": recalculated, "match": ok}
        )
    for horizon, sizes in contract["dilution_ladder"].items():
        for size, burdens in sizes.items():
            for name, counts in burdens.items():
                expected = int(_math.floor(float(counts["burden"]) * int(size) + 0.5))
                if int(counts["n_abnormal"]) != expected:
                    failures += 1
    # Clopper-Pearson spot checks via an INDEPENDENT method: the exact
    # binomial-tail inversion (no continued fractions, no shared code with
    # the design module). Invert P(X >= k; p) = alpha/2 and
    # P(X <= k; p) = alpha/2 by bisection on the exact binomial tail sums.
    # Three rows are checked: the proposed bar (24/48), the futility
    # example (5/48) and the gray-zone example (14/48).
    spot_rows = [(24, 48), (5, 48), (14, 48)]
    spot_results = []
    for successes, trials in spot_rows:
        lower = _binomial_cp_lower(successes, trials)
        upper = _binomial_cp_upper(successes, trials)
        spot_results.append(
            {
                "successes": successes,
                "trials": trials,
                "exact_lower_bound": round(lower, 4),
                "exact_upper_bound": round(upper, 4),
            }
        )
        design_entry = design_ref_lookup(contract, successes, trials)
        if design_entry is not None:
            if (
                round(lower, 4) != round(float(design_entry["ci_lower"]), 4)
                or round(upper, 4) != round(float(design_entry["ci_upper"]), 4)
            ):
                failures += 1
                spot_results[-1]["match_design_module"] = False
            else:
                spot_results[-1]["match_design_module"] = True
    bar_lower = spot_results[0]["exact_lower_bound"]
    return {
        "status": "PASS" if failures == 0 else "FAIL",
        "n_sign_entries_recomputed": len(checked),
        "n_mismatches": failures,
        "entries": checked,
        "spot_checks": spot_results,
        "bar_spot_check": {
            "successes": spot_results[0]["successes"],
            "trials": spot_results[0]["trials"],
            "exact_lower_bound": spot_results[0]["exact_lower_bound"],
            "exact_upper_bound": spot_results[0]["exact_upper_bound"],
            "above_false_alarm_anchor_0_06": bool(bar_lower > 0.06),
        },
        "method": "exact binomial-tail inversion; no continued fractions",
        "assumptions": [
            "p_win and delta are ASSUMED sensitivity grids, not measured effects.",
            "M14 anchors calibrate responsiveness/null bands only, never power.",
            "Pairs are independent by construction (grouped shared-source seeds).",
        ],
    }


def _binomial_tail_ge(successes: int, trials: int, prob: float) -> float:
    """Exact P(X >= successes) for Binomial(trials, prob)."""
    import math as _math

    if successes <= 0:
        return 1.0
    if successes > trials:
        return 0.0
    if prob <= 0.0:
        return 0.0
    if prob >= 1.0:
        return 1.0
    return sum(
        _math.comb(trials, i) * (prob**i) * ((1.0 - prob) ** (trials - i))
        for i in range(successes, trials + 1)
    )


def _binomial_tail_le(successes: int, trials: int, prob: float) -> float:
    """Exact P(X <= successes) for Binomial(trials, prob)."""
    import math as _math

    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    if prob <= 0.0:
        return 1.0
    if prob >= 1.0:
        return 0.0
    return sum(
        _math.comb(trials, i) * (prob**i) * ((1.0 - prob) ** (trials - i))
        for i in range(0, successes + 1)
    )


def _binomial_cp_lower(successes: int, trials: int, alpha: float = 0.05) -> float:
    """Lower Clopper-Pearson bound by exact tail inversion."""
    if successes == 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _binomial_tail_ge(successes, trials, mid) > alpha / 2.0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def _binomial_cp_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    """Upper Clopper-Pearson bound by exact tail inversion."""
    if successes == trials:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _binomial_tail_le(successes, trials, mid) < alpha / 2.0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def design_ref_lookup(
    contract: dict[str, Any], successes: int, trials: int
) -> dict[str, Any] | None:
    """Look up the design module's reported interval for one spot row."""
    for row in contract.get("clopper_pearson_reference", {}).get("rows", []):
        if int(row.get("successes", -1)) == successes and int(
            row.get("trials", -1)
        ) == trials:
            return dict(row)
    return None

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S24-7-M15 paired-decision design CLI")
    parser.add_argument("--mode", choices=("smoke", "contract"), required=True)
    parser.add_argument("--out", required=False, default=None)
    args = parser.parse_args(argv)
    if args.mode == "smoke":
        print(json.dumps(run_smoke(), indent=1))
        return 0
    if not args.out:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "--mode contract requires --out")
    result = run_contract(Path(args.out))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
