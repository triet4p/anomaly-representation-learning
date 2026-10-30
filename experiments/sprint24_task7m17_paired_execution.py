"""Sprint 24 Task 7-M17 paired-execution runner CLI.

Public entry points (reuse M16 pairing + frozen-encoder capture patterns):

1. ``--mode contract --out <fresh dir>`` builds and verifies the frozen
   executable contract from the verified G2 preflight (local accepted copy),
   writes the contract + independent checks + seed plan + run record.
   No model load, no forward, no remote contact.

2. ``--mode smoke --out <fresh dir> [--device cpu]`` runs the tiny local
   CPU smoke through the PUBLIC entry surface: paired-twin generation
   (M16 ``generate_paired_twin``), grouped K=8 E1 scoring on waveform
   stand-ins (proves grouping/admission/error math, never corpus evidence),
   deterministic replay, disposition accounting and contract verification.
   No checkpoint load, no forward, no bank refit, temp outputs only.

3. ``--mode capture --out <fresh dir> --b0-root <dir> --bank-root <dir>
   --preflight <preflight.json> --device cuda`` restores the three accepted
   frozen B0 bundles and captures real embeddings for a tiny paired dataset
   (one family x one severity x G=2 scored groups = 32 files x 3 seeds = 96
   forwards) through ``capture_s24_file_features`` + ``score_scored_group``.
   Intended for the verified server runtime (main-checkout venv, CUDA).
   Refuses locally without accepted roots; never random weights.

Usage (local, isolated smoke only)::

```bash
python experiments/sprint24_task7m17_paired_execution.py --mode contract --out <fresh dir>
python experiments/sprint24_task7m17_paired_execution.py --mode smoke --out <fresh dir>
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from representation import sprint24_paired_execution as execution  # noqa: E402
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


# --- mode: contract ------------------------------------------------------------

#: Local accepted preflight copy (read-only; server digests verified equal).
LOCAL_PREFLIGHT = (
    REPO / "artifacts" / "sprint-24" / "evidence" / "s24-7-m4-a01"
    / "g2-accepted" / "preflight.json"
)


def run_contract(out: Path) -> dict[str, Any]:
    """Build and verify the frozen executable contract; write evidence."""
    started = time.time()
    _require_fresh_out(out)
    if not LOCAL_PREFLIGHT.is_file():
        raise _refused("REFUSED-MISSING-PREREQUISITE", f"accepted preflight absent: {LOCAL_PREFLIGHT}")
    preflight = json.loads(LOCAL_PREFLIGHT.read_text(encoding="utf-8"))
    if not isinstance(preflight, dict):
        raise _refused("REFUSED-IDENTITY", "accepted preflight is not a mapping")
    seed_plan = execution.check_group_seed_plan()
    anchors = {
        "source": "artifacts/sprint-24/evidence/s24-7-m14-a01/audit/aggregates.json",
        "branch_c_e1_zero_shift_median_detection_fraction": 0.06,
        "note": "responsiveness only; M14 perturbations never measure anomaly effects",
    }
    provenance = {
        "execution_module": "src/representation/sprint24_paired_execution.py",
        "criterion": execution.CRITERION,
        "review_cycle": execution.REVIEW_CYCLE,
        "source_commit": source_commit(),
        "cumulative_forwards_before": execution.CUMULATIVE_FORWARDS_BEFORE,
        "preflight_copy": str(LOCAL_PREFLIGHT),
        "generated_utc_note": "written by run record, not part of the contract digest",
    }
    contract = execution.build_execution_contract(anchors, preflight, provenance)
    verification = execution.verify_execution_contract(contract)
    contract_digest = str(verification["contract_sha256"])
    write_json(out / "execution-contract.json", contract)
    independent = _independent_execution_check(contract)
    independent_digest = write_json(out / "independent-execution-check.json", independent)
    seed_digest = write_json(out / "seed-plan.json", seed_plan)
    run_record = {
        "task": execution.TASK_ID,
        "attempt": "S24-7-M17-A04",
        "execution_contract_id": execution.EXECUTION_CONTRACT_ID,
        "design_id": execution.EXECUTION_DESIGN_ID,
        "source_commit": provenance["source_commit"],
        "contract_sha256": contract_digest,
        "verification": verification,
        "independent_execution_check": {
            "digest": independent_digest,
            "status": independent["status"],
        },
        "seed_plan": {"digest": seed_digest, "status": seed_plan["status"]},
        "forwards_executed_this_run": 0,
        "model_loads": 0,
        "official_corpus_written": False,
        "cumulative_forwards_after_run": execution.CUMULATIVE_FORWARDS_BEFORE,
        "elapsed_s": round(time.time() - started, 2),
    }
    run_digest = write_json(out / "run.json", run_record)
    return {
        "out": str(out),
        "contract_sha256": contract_digest,
        "run_sha256": run_digest,
        "status": "CONTRACT-COMPLETE",
    }


def _independent_execution_check(contract: dict[str, Any]) -> dict[str, Any]:
    """Recompute group power + CP rows without importing the execution math."""
    import math as _math

    failures = 0
    checked: list[dict[str, Any]] = []
    for entry in contract["group_operating_characteristics"]["sign_power_groups"]:
        n = int(entry["n_pairs"])
        p = float(entry["p_win_assumed"])
        total = 2**n
        critical = None
        for candidate in range(n + 1):
            upper = sum(_math.comb(n, i) for i in range(candidate, n + 1))
            lower = sum(_math.comb(n, i) for i in range(0, n - candidate + 1))
            if (upper + lower) / total <= float(contract["group_operating_characteristics"]["alpha"]):
                critical = candidate
                break
        assert critical == int(entry["critical_value"]), "group critical value drift"
        upper_power = sum(
            _math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
            for i in range(critical, n + 1)
        )
        lower_power = sum(
            _math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
            for i in range(0, n - critical + 1)
        )
        recalculated = round(upper_power + lower_power, 4)
        ok = recalculated == float(entry["power"])
        failures += 0 if ok else 1
        checked.append({"n_groups": n, "p_win": p, "recalculated": recalculated, "match": ok})
    # Grid must be exactly 24/48: any 96 entry is a failure.
    grid = list(contract["execution"]["stage1_group_grid"])
    grid_ok = grid == [24, 48]
    if not grid_ok:
        failures += 1
    n96 = sum(1 for entry in checked if int(entry["n_groups"]) == 96)
    if n96:
        failures += n96
    # CP spot rows via independent exact-tail inversion (no shared imports).
    spot_rows = [(24, 48), (5, 48), (14, 48)]
    spot_results = []
    for successes, trials in spot_rows:
        lower = _cp_lower(successes, trials)
        upper = _cp_upper(successes, trials)
        spot_results.append({
            "successes": successes,
            "trials": trials,
            "exact_lower_bound": round(lower, 4),
            "exact_upper_bound": round(upper, 4),
        })
        design_entry = _cp_lookup(contract, successes, trials)
        if design_entry is not None:
            if (
                round(lower, 4) != round(float(design_entry["ci_lower"]), 4)
                or round(upper, 4) != round(float(design_entry["ci_upper"]), 4)
            ):
                failures += 1
                spot_results[-1]["match_execution_module"] = False
            else:
                spot_results[-1]["match_execution_module"] = True
    # Dilution ladder rounding check.
    for _horizon, sizes in contract["dilution_ladder"].items():
        for size, burdens in sizes.items():
            for _name, counts in burdens.items():
                expected = int(_math.floor(float(counts["burden"]) * int(size) + 0.5))
                if int(counts["n_abnormal"]) != expected:
                    failures += 1
    return {
        "status": "PASS" if failures == 0 else "FAIL",
        "n_group_entries_recomputed": len(checked),
        "n_mismatches": failures,
        "grid_is_24_48": grid_ok,
        "n_96_entries": n96,
        "entries": checked,
        "spot_checks": spot_results,
        "method": "exact binomial-tail inversion; no shared code with the execution module",
        "assumptions": [
            "p_win is an ASSUMED sensitivity grid, not a measured effect.",
            "Independent unit is one K=8 scored group, never one pair.",
            "Response fraction 0.5 is a proposed bar, never raw E1.",
        ],
    }


def _tail_ge(successes: int, trials: int, prob: float) -> float:
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


def _tail_le(successes: int, trials: int, prob: float) -> float:
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


def _cp_lower(successes: int, trials: int, alpha: float = 0.05) -> float:
    if successes == 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _tail_ge(successes, trials, mid) > alpha / 2.0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def _cp_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    if successes == trials:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _tail_le(successes, trials, mid) < alpha / 2.0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def _cp_lookup(contract: dict[str, Any], successes: int, trials: int) -> dict[str, Any] | None:
    for row in contract.get("clopper_pearson_reference", {}).get("rows", []):
        if int(row.get("successes", -1)) == successes and int(row.get("trials", -1)) == trials:
            return dict(row)
    return None


# --- mode: smoke (local CPU, no checkpoint, no forward) ------------------------
def _fixed_summary_vector(x: np.ndarray, width: int = 24) -> np.ndarray:
    """Summarize one variable-T waveform as a fixed-width row (smoke only)."""
    array = np.asarray(x, dtype=np.float64)
    stats = np.concatenate([
        array.mean(axis=1), array.std(axis=1),
        array.min(axis=1), array.max(axis=1),
    ])
    if stats.shape[0] >= width:
        return stats[:width].copy()
    padded = np.zeros((width,), dtype=np.float64)
    padded[: stats.shape[0]] = stats
    return padded



def run_smoke(out: Path | None = None) -> dict[str, Any]:
    """Tiny public-path smoke: grouping + evaluation math on real pairs.

    Uses the real M16 ``generate_paired_twin`` on throwaway synthetic inputs
    (temp objects only, 0 forwards, 0 model loads). Waveform stand-ins prove
    the grouping/admission/error/abstention plumbing; the grouped E1 math is
    exercised on fixed-width per-file summary rows (never corpus evidence).
    """
    from synth.config import SynthConfig
    from synth.generator import SessionGenerator, derive_pair_seed
    from synth.schema import AnomalyFamily
    started = time.time()
    gen = SessionGenerator(SynthConfig())
    family = AnomalyFamily.SUBTLE_DRIFT
    severity = 0.6
    history_seed = 53101
    n_groups, k = 2, execution.GROUP_SIZE_K
    pair_seeds = [derive_pair_seed(history_seed, index) for index in range(n_groups * k)]
    pairs = [gen.generate_paired_twin(seed, family, severity) for seed in pair_seeds]
    dispositions = [pair.disposition for pair in pairs]
    n_paired = sum(1 for value in dispositions if value == "PAIRED")
    # Admission: T varies per pair by construction, so the smoke scores
    # fixed-width per-file summary vectors (mean/std/min/max per channel)
    # as KxD group rows. Real variable-T files are handled by the frozen
    # encoder in capture/full modes; no pad/truncate happens here.
    twin_rows = [_fixed_summary_vector(pair.twin.x) for pair in pairs]
    abnormal_rows = [_fixed_summary_vector(pair.abnormal.x) for pair in pairs]
    width = int(twin_rows[0].shape[0])
    twin_matrix = np.stack(twin_rows)
    abnormal_matrix = np.stack(abnormal_rows)
    pooled = np.concatenate([twin_matrix, abnormal_matrix], axis=0)
    scale = pooled.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    twin_std = (twin_matrix - pooled.mean(axis=0)) / scale
    abnormal_std = (abnormal_matrix - pooled.mean(axis=0)) / scale
    sigma = float(np.median(np.abs(pooled - np.median(pooled)))) or 1.0
    group_e1: list[float] = []
    for group in range(n_groups):
        sl = slice(group * k, (group + 1) * k)
        scored = execution.score_scored_group(twin_std[sl], abnormal_std[sl], sigma)
        group_e1.append(float(scored["e1"]))
    # Null stand-in: twin-vs-twin halves give a same-K calibration shape check.
    null_e1: list[float] = []
    halves = twin_std[: 2 * k]
    null_e1.append(float(execution.score_scored_group(halves[:k], halves[k : 2 * k], sigma)["e1"]))
    null_q95 = float(np.quantile(null_e1 + group_e1, 0.95))
    cell = execution.summarize_cell(group_e1, null_q95)
    # Deterministic replay across a fresh generator.
    gen2 = SessionGenerator(SynthConfig())
    replay = [gen2.generate_paired_twin(seed, family, severity) for seed in pair_seeds[:2]]
    replay_ok = all(
        bool((first.twin.x == second.twin.x).all())
        and bool((first.abnormal.x == second.abnormal.x).all())
        and first.source_id == second.source_id
        for first, second in zip(pairs[:2], replay)
    )
    # Error boundaries: singleton arms and non-finite inputs must raise.
    singleton_refused = False
    try:
        execution.score_scored_group(twin_std[:1], abnormal_std[:1], sigma)
    except ValueError as exc:
        singleton_refused = "REFUSED-ERROR-BOUNDARY" in str(exc)
    nonfinite_refused = False
    try:
        bad = abnormal_std.copy()
        bad[0, 0] = float("nan")
        execution.score_scored_group(twin_std[:k], bad[:k], sigma)
    except ValueError as exc:
        nonfinite_refused = "REFUSED-ERROR-BOUNDARY" in str(exc)
    checks = {
        "pair_seeds_unique": len(set(pair_seeds)) == len(pair_seeds),
        "all_paired": n_paired == len(pairs),
        "fixed_width_rows": width == 24,
        "group_e1_finite": all(float(v) == float(v) for v in group_e1),
        "cell_interval_covers": cell["clopper_pearson"]["ci_lower"] <= cell["response_fraction"] <= cell["clopper_pearson"]["ci_upper"],
        "deterministic_replay": bool(replay_ok),
        "singleton_refused": bool(singleton_refused),
        "nonfinite_refused": bool(nonfinite_refused),
        "twin_waveform_sha": _waveform_sha256(pairs[0].twin.x),
        "abnormal_waveform_sha": _waveform_sha256(pairs[0].abnormal.x),
        "group_id": pairs[0].group_id,
        "source_id": pairs[0].source_id,
    }
    result = {
        "status": "SMOKE-COMPLETE" if all(bool(checks[key]) for key in (
            "pair_seeds_unique", "all_paired", "fixed_width_rows",
            "group_e1_finite", "cell_interval_covers",
            "deterministic_replay", "singleton_refused", "nonfinite_refused",
        )) else "SMOKE-FAIL",
        "family": family.value,
        "severity": severity,
        "history_seed": history_seed,
        "n_groups": n_groups,
        "k_per_arm": k,
        "n_pairs": len(pairs),
        "width": width,
        "sigma_standin": sigma,
        "group_e1": [round(float(v), 6) for v in group_e1],
        "null_q95_standin": round(null_q95, 6),
        "cell": cell,
        "checks": checks,
        "dispositions": {str(value): int(dispositions.count(value)) for value in sorted(set(dispositions))},
        "forwards_executed_this_run": 0,
        "model_loads": 0,
        "cumulative_forwards_after_run": execution.CUMULATIVE_FORWARDS_BEFORE,
        "elapsed_s": round(time.time() - started, 2),
    }
    if out is not None:
        _require_fresh_out(out)
        write_json(out / "smoke.json", result)
        write_json(out / "run.json", {
            "task": execution.TASK_ID,
            "attempt": "S24-7-M17-A04",
            "mode": "smoke",
            "source_commit": source_commit(),
            "status": result["status"],
            "forwards_executed_this_run": 0,
            "model_loads": 0,
            "cumulative_forwards_after_run": execution.CUMULATIVE_FORWARDS_BEFORE,
        })
    return result


# --- mode: capture (server CUDA, accepted bytes only) --------------------------


def _capture_device(args: argparse.Namespace) -> str:
    """Resolve the isolated-smoke capture device (default cpu, cuda allowed)."""
    from representation.sprint24_features import resolve_s24_device

    requested = getattr(args, "device", None) or "cpu"
    return str(resolve_s24_device(requested))


def _restore_bundles(args: argparse.Namespace, preflight: dict[str, Any]) -> dict[int, Any]:
    """Restore the three accepted frozen B0 bundles on the requested device."""
    from representation.sprint18_harness import (
        Sprint18ArmConfig,
        arm_config_sha256,
        load_task6_binding,
        validate_arm_config,
    )
    from representation.sprint24_features import restore_s24_b0_bundle

    binding = load_task6_binding(
        REPO / "experiments/sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    lock_doc = json.loads(
        (REPO / "experiments/sprint18-task14-config-lock.json").read_text(encoding="utf-8")
    )
    bundles: dict[int, Any] = {}
    for seed in execution.FROZEN_ENCODER_SEEDS:
        checkpoint_path = (
            Path(args.b0_root) / f"seed-{seed}" / "checkpoints" / f"b0-seed{seed}-epoch100.pt"
        )
        bank_path = Path(args.bank_root) / f"B0-FULL-{seed}.json.fit-bank" / "bank.pt"
        expected_checkpoint = str(preflight["checkpoints"][str(seed)])
        expected_bank_file = str(preflight["bank_files"][str(seed)])
        for path, digest in ((checkpoint_path, expected_checkpoint), (bank_path, expected_bank_file)):
            if not path.is_file():
                raise _refused("REFUSED-MISSING-PREREQUISITE", f"required accepted asset absent: {path}")
            text = hashlib.sha256(path.read_bytes()).hexdigest()
            if text != digest:
                raise _refused("REFUSED-IDENTITY", f"accepted asset digest mismatch: {path}")
        arm = validate_arm_config(
            Sprint18ArmConfig(**lock_doc["locked_configs"][f"B0-FULL/seed-{seed}"]),
            require_locked=True,
        )
        if int(arm.model_seed) != int(seed):
            raise _refused("REFUSED-IDENTITY", f"accepted B0 arm seed mismatch: {seed}")
        if arm_config_sha256(arm) != str(preflight["arms"][str(seed)]):
            raise _refused("REFUSED-IDENTITY", f"accepted arm-config digest mismatch: {seed}")
        bundles[int(seed)] = restore_s24_b0_bundle(
            binding=binding,
            arm=arm,
            checkpoint_path=checkpoint_path,
            bank_path=bank_path,
            expected_checkpoint_sha256=expected_checkpoint,
            expected_bank_sha256=str(preflight["banks"][str(seed)]),
            device=_capture_device(args),
        )
    return bundles


def run_capture(out: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Capture real embeddings for a tiny paired dataset through the public entry.

    Temp outputs only; never the official corpus/cache. Defaults to CPU for the
    isolated smoke; pass --device cuda on the verified CUDA runtime.
    """
    from synth.config import SynthConfig
    from synth.generator import SessionGenerator, derive_pair_seed
    from synth.schema import AnomalyFamily
    from representation.sprint24_features import capture_s24_file_features

    started = time.time()
    _require_fresh_out(out)
    if not args.b0_root or not args.bank_root:
        raise _refused("REFUSED-PREREQUISITE", "capture needs --b0-root/--bank-root (accepted bytes)")
    if not args.preflight:
        raise _refused("REFUSED-PREREQUISITE", "capture needs --preflight (verified preflight.json)")
    preflight_path = Path(args.preflight)
    if not preflight_path.is_file():
        raise _refused("REFUSED-MISSING-PREREQUISITE", f"preflight absent: {preflight_path}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not isinstance(preflight, dict):
        raise _refused("REFUSED-IDENTITY", "preflight is not a mapping")
    bundles = _restore_bundles(args, preflight)
    from representation.sprint24_features import s24_live_state_digests

    frozen_before = {
        int(seed): s24_live_state_digests(bundles[int(seed)].model, bundles[int(seed)].bank)  # type: ignore[attr-defined]
        for seed in execution.FROZEN_ENCODER_SEEDS
    }
    gen = SessionGenerator(SynthConfig())
    family = AnomalyFamily.SUBTLE_DRIFT
    severity = 0.6
    history_seed = 53101
    n_groups, k = 2, execution.GROUP_SIZE_K
    pair_seeds = [derive_pair_seed(history_seed, index) for index in range(n_groups * k)]
    pairs = [gen.generate_paired_twin(seed, family, severity) for seed in pair_seeds]
    for pair in pairs:
        if pair.disposition != "PAIRED":
            raise _refused("REFUSED-ADMISSION", f"capture pair abstained: {pair.pair_seed} {pair.disposition}")
    per_seed_e1: dict[str, list[float]] = {}
    per_seed_null: dict[str, float] = {}
    forwards = 0
    for seed in execution.FROZEN_ENCODER_SEEDS:
        bundle = bundles[int(seed)]
        history_id = "S24P-TINY-SMOKE"
        captured = capture_s24_file_features(
            model=bundle.model,  # type: ignore[attr-defined]
            bank=bundle.bank,  # type: ignore[attr-defined]
            samples=[pair.twin for pair in pairs] + [pair.abnormal for pair in pairs],
            history_id=history_id,
            model_seed=int(seed),
            device=_capture_device(args),
        )
        forwards += len(captured)
        order = [item.file_id for item in captured]
        expected = [pair.twin.file_id for pair in pairs] + [pair.abnormal.file_id for pair in pairs]
        if order != expected:
            raise _refused("REFUSED-IDENTITY", f"capture order drifted for seed {seed}")
        vectors = np.stack([item.embedding.detach().cpu().float().numpy() for item in captured])
        if vectors.shape != (2 * n_groups * k, 128):
            raise _refused("REFUSED-IDENTITY", f"capture shape drifted for seed {seed}: {vectors.shape}")
        if not bool(np.isfinite(vectors).all()):
            raise _refused("REFUSED-ERROR-BOUNDARY", f"non-finite capture for seed {seed}")
        twin_matrix = vectors[: n_groups * k].astype(np.float64)
        abnormal_matrix = vectors[n_groups * k :].astype(np.float64)
        # Reference bandwidth: median pairwise distance over the twin pool
        # (causal-reference stand-in for the tiny smoke; the official pool
        # is the frozen 220-file causal reference per history).
        from scipy.spatial.distance import pdist

        sigma = float(np.median(pdist(twin_matrix)))
        if not np.isfinite(sigma) or sigma <= 0:
            raise _refused("REFUSED-ERROR-BOUNDARY", f"non-finite reference bandwidth for seed {seed}")
        group_e1 = []
        for group in range(n_groups):
            sl = slice(group * k, (group + 1) * k)
            scored = execution.score_scored_group(twin_matrix[sl], abnormal_matrix[sl], sigma)
            group_e1.append(float(scored["e1"]))
        per_seed_e1[str(seed)] = [round(float(v), 6) for v in group_e1]
        halves = twin_matrix[: 2 * k]
        null_value = float(execution.score_scored_group(halves[:k], halves[k : 2 * k], sigma)["e1"])
        per_seed_null[str(seed)] = round(null_value, 6)
        frozen_after = s24_live_state_digests(bundle.model, bundle.bank)  # type: ignore[attr-defined]
        if frozen_after != frozen_before[int(seed)]:
            raise _refused("REFUSED-IDENTITY", f"frozen state moved during capture for seed {seed}")
    medians = [float(np.median([per_seed_e1[str(seed)][group] for seed in execution.FROZEN_ENCODER_SEEDS])) for group in range(n_groups)]
    null_median = float(np.median([per_seed_null[str(seed)] for seed in execution.FROZEN_ENCODER_SEEDS]))
    cell = execution.summarize_cell(medians, null_median)
    result = {
        "status": "CAPTURE-COMPLETE",
        "family": family.value,
        "severity": severity,
        "history_seed": history_seed,
        "n_groups": n_groups,
        "k_per_arm": k,
        "n_pairs": len(pairs),
        "per_seed_e1": per_seed_e1,
        "per_seed_null_twin_vs_twin": per_seed_null,
        "median_group_e1": [round(float(v), 6) for v in medians],
        "median_null": round(null_median, 6),
        "cell_median_summary": cell,
        "frozen_state_before": {str(k): dict(v) for k, v in frozen_before.items()},
        "frozen_state_unchanged": True,
        "forwards_executed_this_run": int(forwards),
        "model_loads": len(bundles),
        "device": _capture_device(args),
        "cumulative_forwards_after_run": int(execution.CUMULATIVE_FORWARDS_BEFORE) + int(forwards),
        "elapsed_s": round(time.time() - started, 2),
    }
    device = _capture_device(args)
    write_json(out / "capture.json", result)
    write_json(out / "run.json", {
        "task": execution.TASK_ID,
        "attempt": "S24-7-M17-A04",
        "mode": "capture",
        "source_commit": source_commit(),
        "status": result["status"],
        "forwards_executed_this_run": int(forwards),
        "model_loads": len(bundles),
        "device": device,
        "cumulative_forwards_after_run": result["cumulative_forwards_after_run"],
    })
    return result


def run_official_contract(out: Path) -> dict[str, Any]:
    """Build and verify the versioned C2 official contract; write evidence."""
    from representation import sprint24_paired_official as official

    started = time.time()
    _require_fresh_out(out)
    if not LOCAL_PREFLIGHT.is_file():
        raise _refused("REFUSED-MISSING-PREREQUISITE", f"accepted preflight absent: {LOCAL_PREFLIGHT}")
    preflight = json.loads(LOCAL_PREFLIGHT.read_text(encoding="utf-8"))
    if not isinstance(preflight, dict):
        raise _refused("REFUSED-IDENTITY", "accepted preflight is not a mapping")
    anchors = {
        "source": "artifacts/sprint-24/evidence/s24-7-m14-a01/audit/aggregates.json",
        "branch_c_e1_zero_shift_median_detection_fraction": 0.06,
        "note": "responsiveness only; M14 perturbations never measure anomaly effects",
    }
    provenance = {
        "execution_module": "src/representation/sprint24_paired_official.py",
        "supersedes_module": "src/representation/sprint24_paired_execution.py",
        "criterion": official.CRITERION,
        "review_cycle": official.REVIEW_CYCLE,
        "source_commit": source_commit(),
        "cumulative_forwards_before": official.CUMULATIVE_FORWARDS_BEFORE,
        "preflight_copy": str(LOCAL_PREFLIGHT),
        "generated_utc_note": "written by run record, not part of the contract digest",
    }
    contract = official.build_official_contract(preflight, provenance, anchors)
    verification = official.verify_official_contract(contract)
    counts = official.official_matrix_counts()
    contract_file_digest = write_json(out / "official-contract.json", contract)
    matrix_digest = write_json(out / "official-matrix.json", official.matrix_plan())
    counts_digest = write_json(out / "official-matrix-counts.json", counts)
    allocation_digest = write_json(out / "official-allocation.json", official.allocation_table())
    arms = [
        {
            "horizon": arm.horizon,
            "size": arm.size,
            "burden": arm.burden,
            "n_abnormal": arm.n_abnormal,
            "n_ordinary": arm.n_ordinary,
            "pair_indices": list(arm.pair_indices),
        }
        for arm in official.population_arms()
    ]
    arms_digest = write_json(out / "official-population-arms.json", {"arms": arms})
    run_record = {
        "task": official_base_task(),
        "attempt": "S24-7-M17-A05",
        "official_contract_id": official.OFFICIAL_CONTRACT_ID,
        "supersedes": official.SUPERSEDED_CONTRACT_ID,
        "source_commit": provenance["source_commit"],
        "contract_sha256": str(verification["contract_sha256"]),
        "verification": verification,
        "digests": {
            "official_contract": contract_file_digest,
            "official_matrix": matrix_digest,
            "official_matrix_counts": counts_digest,
            "official_allocation": allocation_digest,
            "official_population_arms": arms_digest,
        },
        "forwards_executed_this_run": 0,
        "model_loads": 0,
        "official_corpus_written": False,
        "elapsed_s": round(time.time() - started, 2),
    }
    run_digest = write_json(out / "run.json", run_record)
    return {
        "out": str(out),
        "contract_sha256": str(verification["contract_sha256"]),
        "contract_file_sha256": contract_file_digest,
        "run_sha256": run_digest,
        "status": "OFFICIAL-CONTRACT-COMPLETE",
    }


def official_base_task() -> str:
    """Task identifier shared by the A01 and C2 execution modules."""
    return execution.TASK_ID


def run_official_smoke(out: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Run the tiny official-engine fixture smoke on real accepted bytes."""
    from representation import sprint24_paired_official as official

    started = time.time()
    _require_fresh_out(out)
    if not args.b0_root or not args.bank_root:
        raise _refused("REFUSED-PREREQUISITE", "official-smoke needs --b0-root/--bank-root (accepted bytes)")
    if not args.preflight:
        raise _refused("REFUSED-PREREQUISITE", "official-smoke needs --preflight (verified preflight.json)")
    preflight_path = Path(args.preflight)
    if not preflight_path.is_file():
        raise _refused("REFUSED-MISSING-PREREQUISITE", f"preflight absent: {preflight_path}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not isinstance(preflight, dict):
        raise _refused("REFUSED-IDENTITY", "preflight is not a mapping")
    bundles = _restore_bundles(args, preflight)
    device = _capture_device(args)
    result = official.tiny_official_smoke(bundles, device=device)
    cell = result.cell_summary
    payload = {
        "status": result.status,
        "family": result.family,
        "severity": result.severity,
        "history_seed": result.history_seed,
        "n_groups": result.n_groups,
        "k_per_arm": execution.GROUP_SIZE_K,
        "per_seed_e1": result.per_seed_e1,
        "per_seed_null_q95": result.per_seed_q95,
        "cell_summary": cell,
        "stage2_arms": result.stage2_arms,
        "forwards_executed_this_run": int(result.forwards_executed),
        "model_loads": int(result.model_loads),
        "device": device,
        "cumulative_forwards_after_run": int(official.CUMULATIVE_FORWARDS_BEFORE) + int(result.forwards_executed),
        "elapsed_s": round(time.time() - started, 2),
    }
    write_json(out / "tiny-official-smoke.json", payload)
    write_json(out / "run.json", {
        "task": official_base_task(),
        "attempt": "S24-7-M17-A05",
        "mode": "official-smoke",
        "source_commit": source_commit(),
        "status": result.status,
        "forwards_executed_this_run": int(result.forwards_executed),
        "model_loads": int(result.model_loads),
        "device": device,
        "cumulative_forwards_after_run": payload["cumulative_forwards_after_run"],
    })
    return payload


def run_official_matrix_cli(out: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Execute the official matrix through the public orchestrator (gated)."""
    from representation import sprint24_paired_official as official

    started = time.time()
    _require_fresh_out(out)
    if not args.b0_root or not args.bank_root:
        raise _refused("REFUSED-PREREQUISITE", "official-run needs --b0-root/--bank-root (accepted bytes)")
    if not args.preflight:
        raise _refused("REFUSED-PREREQUISITE", "official-run needs --preflight (verified preflight.json)")
    preflight_path = Path(args.preflight)
    if not preflight_path.is_file():
        raise _refused("REFUSED-MISSING-PREREQUISITE", f"preflight absent: {preflight_path}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not isinstance(preflight, dict):
        raise _refused("REFUSED-IDENTITY", "preflight is not a mapping")
    bundles = _restore_bundles(args, preflight)
    device = _capture_device(args)
    permitted = bool(getattr(args, "permit_official", False))
    result = official.run_official_matrix(
        bundles,
        permitted=permitted,
        histories=[str(value) for value in (getattr(args, "histories", "") or "").split(",") if value] or None,
        cells=None,
        g_level=str(getattr(args, "g_level", None) or "g48"),
        device=device,
    )
    payload = {
        "status": result["status"],
        "g_level": result["g_level"],
        "n_stage1_keys": len(result["stage1"]),
        "n_stage2_keys": len(result["stage2"]),
        "n_holm_families": len(result["holm_by_family"]),
        "forwards_executed_this_run": int(result["forwards_executed"]),
        "model_loads": int(result["model_loads"]),
        "device": device,
        "elapsed_s": round(time.time() - started, 2),
    }
    write_json(out / "official-run.json", result)
    write_json(out / "run.json", {
        "task": official_base_task(),
        "attempt": "S24-7-M17-A05",
        "mode": "official-run",
        "source_commit": source_commit(),
        **payload,
    })
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S24-7-M17 paired-execution CLI")
    parser.add_argument("--mode", choices=("contract", "smoke", "capture", "official-contract", "official-smoke", "official-run"), required=True)
    parser.add_argument("--out", required=False, default=None)
    parser.add_argument("--b0-root", required=False, default=None)
    parser.add_argument("--bank-root", required=False, default=None)
    parser.add_argument("--preflight", required=False, default=None)
    parser.add_argument("--device", required=False, default="cpu")
    parser.add_argument("--permit-official", dest="permit_official", action="store_true", default=False)
    parser.add_argument("--histories", required=False, default="")
    parser.add_argument("--g-level", dest="g_level", required=False, default="g48")
    args = parser.parse_args(argv)
    if args.mode == "contract":
        if not args.out:
            raise _refused("REFUSED-UNKNOWN-DESIGN", "--mode contract requires --out")
        print(json.dumps(run_contract(Path(args.out)), indent=1))
        return 0
    if args.mode == "smoke":
        if args.out:
            result = run_smoke(Path(args.out))
        else:
            result = run_smoke(None)
        print(json.dumps(result, indent=1))
        return 0 if result["status"] == "SMOKE-COMPLETE" else 2
    if args.mode == "official-contract":
        if not args.out:
            raise _refused("REFUSED-UNKNOWN-DESIGN", "--mode official-contract requires --out")
        result = run_official_contract(Path(args.out))
        print(json.dumps(result, indent=1))
        return 0 if result["status"] == "OFFICIAL-CONTRACT-COMPLETE" else 2
    if args.mode == "official-smoke":
        if not args.out:
            raise _refused("REFUSED-UNKNOWN-DESIGN", "--mode official-smoke requires --out")
        result = run_official_smoke(Path(args.out), args)
        print(json.dumps(result, indent=1))
        return 0 if result["status"] == "TINY-OFFICIAL-SMOKE-COMPLETE" else 2
    if args.mode == "official-run":
        if not args.out:
            raise _refused("REFUSED-UNKNOWN-DESIGN", "--mode official-run requires --out")
        result = run_official_matrix_cli(Path(args.out), args)
        print(json.dumps(result, indent=1))
        return 0 if result["status"] == "OFFICIAL-MATRIX-COMPLETE" else 2
    if not args.out:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "--mode capture requires --out")
    result = run_capture(Path(args.out), args)
    print(json.dumps(result, indent=1))
    return 0 if result["status"] == "CAPTURE-COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
