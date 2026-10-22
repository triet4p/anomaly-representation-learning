"""S18-T28 runner: locked N2-MSP experiment-scope training.

Runs the exact frozen N2-MSP arm for each frozen seed
(181801/181802/181803): 100 complete locked ``experiment``-scope epochs over
the certified healthy Fit support, the frozen diagnostic (epoch 10/30/60 plus
the declared step-300 boundary record) and primary epoch-100 coherent
checkpoints, a strict restore with model/EMA byte equality and zero-diff
forward parity, the public Fit reference bank, one q95 per branch on the
frozen Calibration support, and the four Development histories scored through
the public independent ``S_pred`` (fine patch-level, unchanged B0 path) and
``S_pop`` (unchanged reference bank) branches. Coarse-4/coarse-8 residuals are
computed through the separate diagnostic surface and are never fused into
either primary branch, threshold, or promotion metric.

Calls only committed public entry points; implements no science, no metric
redefinition, no protocol amendment:

- ``representation.sprint18_harness`` binding/loaders/sampler/config/
  bank/q95/scoring and the N2 coarse diagnostic surface
- ``representation.trainer.RepresentationTrainer.train_sprint18_msp_epoch``
  (the N2 training entry point; it requires the Fit-role digest that drives
  the frozen per-file N2 mask seeds)
- ``representation.sprint18_checkpoint`` schedules/save/strict restore
- ``representation.data.collate_variable_files``
- ``representation.model``/``representation.criterion`` public forwards

Eligibility is applied ONLY by the committed loaders
(``load_certified_healthy_role_support`` / ``load_certified_sample_subset``
with ``role_action`` and ``load_certified_development_support`` for the
label-complete Development support). This runner applies NO additional
health/stage filter and reads no simulator health anywhere.

The frozen coefficient values (alpha = 0.25, InfoNCE maximum = 0.05) are read
from the accepted Task 14 lock and validated against it before any update;
they are never selected, tuned, or reinterpreted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from representation.config import V1Config
from representation.criterion import (
    MSP_FINE_MASK_BLOCK_RANGE,
    MSP_FINE_MASK_COMPOSITION,
    MSP_FINE_MASK_STRATA,
    MSP_POSITION_SHARES,
    MSP_TOTAL_MASK_RATIO,
)
from representation.data import collate_variable_files
from representation.inference import RepresentationInference
from representation.model import V1RepresentationModel
from representation.sprint18_checkpoint import (
    SCHEDULER_ID,
    coefficient_for_epoch,
    learning_rate_for_epoch,
    restore_sprint18_checkpoint,
    save_sprint18_checkpoint,
    state_sha256,
)
from representation.sprint18_harness import (
    BATCH_SIZE,
    FIT_HEALTHY_ROWS,
    FIT_VALID_PATCHES,
    MODEL_SEEDS,
    _encoder_batch,
    arm_config_sha256,
    deterministic_mask_seed,
    deterministic_sampler_order,
    fit_calibration_q95,
    fit_fit_reference_bank,
    load_certified_development_support,
    load_certified_healthy_role_support,
    load_certified_sample_subset,
    load_task6_binding,
    make_sprint18_arm_config,
    primary_inference_parameters,
    profile_inference_flops,
    score_certified_rows,
    score_msp_coarse_rows,
    validate_score_maps,
)
from synth.config import PatchConfig
from synth.patchify import Patchifier

EPOCHS = 100
DIAGNOSTIC_EPOCHS = (10, 30, 60)
STEP_300 = 300
DIAGNOSTIC_EPOCH = 100
TASK_ID = "S18-T28"
ATTEMPT = "S18-T28-A01"
AGENT = "S18T28A01"
ARM_ID = "N2-MSP"
RUN_ID_PREFIX = "n2"
PREFLIGHT_SCHEMA = "s18-t28-n2-preflight-v1"
PAIRED_SCHEMA = "s18-t28-n2-paired-development-v1"
COARSE_DIAGNOSTIC_SCHEMA = "s18-t28-n2-coarse-diagnostics-v1"
ALPHA = 0.25
LAMBDA_INFO_MAX = 0.05
COARSE_SCALES = (4, 8)
TASK14_LOCK_RELATIVE = "experiments/sprint18-task14-config-lock.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _log_progress(handle, record: dict) -> None:
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def _fail(message: str) -> SystemExit:
    print(f"DRIVER_FATAL {message}", file=sys.stderr, flush=True)
    return SystemExit(f"DRIVER_FATAL {message}")


def _move(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}


def _encoder_input(batch: dict) -> dict:
    """Drop diagnostic-only collator fields; the frozen harness set is authoritative."""
    return _encoder_batch(batch)


def _role_assets(binding, worktree: Path, history_id: str):
    history = binding.history(history_id)
    role_dir = worktree / history.relative_path
    manifest_path = role_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_paths = {row["path"]: str(role_dir / row["path"]) for row in manifest["shards"]}
    return history, manifest, manifest_path, shard_paths


def _support_record(history, manifest, manifest_path: Path, rows) -> dict:
    ids = sorted(row.file_id for row in rows)
    return {
        "history_id": history.actual_history_id,
        "role": history.assigned_role,
        "manifest_sha256_bound": history.manifest_sha256,
        "manifest_sha256_observed": _sha256_file(manifest_path),
        "manifest_file_rows_observed": len(manifest["files"]),
        "healthy_rows": len(rows),
        "valid_patches": sum(int(row.manifest_row["n_valid_patches"]) for row in rows),
        "eligible_ids_sha256": _canonical_sha256(ids),
        "unique_ids": len(set(ids)) == len(ids),
        "all_rows_verified_healthy": all(row.is_verified_healthy for row in rows),
    }


def _task14_lock_config_hashes(worktree: Path) -> tuple[dict, str, dict]:
    lock_path = worktree / TASK14_LOCK_RELATIVE
    raw = lock_path.read_bytes()
    lock = json.loads(raw)
    hashes = lock.get("config_sha256")
    if not isinstance(hashes, dict):
        raise _fail("Task14 lock is missing its per-seed arm config hashes")
    arms = lock.get("arms")
    if not isinstance(arms, dict):
        raise _fail("Task14 lock is missing its locked arm registry")
    return hashes, hashlib.sha256(raw).hexdigest(), arms


def _assert_locked_n2(arms: dict) -> dict:
    """Fail closed unless the locked N2 entry matches the frozen arm contract."""
    entry = arms.get(ARM_ID)
    if not isinstance(entry, dict):
        raise _fail(f"Task14 lock has no {ARM_ID} entry")
    if entry.get("component_set") != [ARM_ID]:
        raise _fail(f"Task14 lock component set differs for {ARM_ID}")
    if entry.get("executable") is not True or entry.get("status") != "locked_executable":
        raise _fail(f"Task14 lock does not mark {ARM_ID} executable")
    if entry.get("representation_epochs") != EPOCHS or entry.get("scale_epochs") != 0:
        raise _fail(f"Task14 lock epoch budget differs for {ARM_ID}")
    coefficients = entry.get("coefficients", {})
    if coefficients.get("alpha") != ALPHA:
        raise _fail(f"Task14 lock N2 alpha differs from the frozen {ALPHA}")
    if coefficients.get("lambda_info_max") != LAMBDA_INFO_MAX:
        raise _fail(f"Task14 lock InfoNCE maximum differs for {ARM_ID}")
    return entry


def _n2_arm(seed: int, lock_digest: str):
    return make_sprint18_arm_config(
        ARM_ID,
        seed,
        {"lambda_info_max": LAMBDA_INFO_MAX, "lambda_hcc_max": None, "alpha": ALPHA, "beta": None},
        selection_state="locked",
        coefficient_lock_sha256=lock_digest,
    )


def run_preflight(*, binding, binding_doc, worktree: Path, out_dir: Path, environment) -> dict:
    fit_history_ids = [item.actual_history_id for item in binding.role_histories("FIT")]
    calibration_history_ids = [item.actual_history_id for item in binding.role_histories("CALIBRATION")]
    development_history_ids = [item.actual_history_id for item in binding.role_histories("DEVELOPMENT")]
    if len(fit_history_ids) != 3 or len(calibration_history_ids) != 1 or len(development_history_ids) != 4:
        raise _fail("certified roster differs from the frozen 3/1/4 Fit/Calibration/Development topology")
    started = time.time()
    fit_rows: list = []
    support_records = {"fit": [], "calibration": [], "development": []}
    for history_id in fit_history_ids:
        history, manifest, manifest_path, shard_paths = _role_assets(binding, worktree, history_id)
        expected = binding_doc["fit_support"][history_id]
        rows = load_certified_healthy_role_support(binding, history_id, manifest_path, shard_paths, action="train_fit")
        record = _support_record(history, manifest, manifest_path, rows)
        if record["manifest_sha256_observed"] != expected["manifest_sha256"]:
            raise _fail(f"fit manifest digest differs for {history_id}")
        if record["manifest_file_rows_observed"] != expected["manifest_file_rows"]:
            raise _fail(f"fit manifest rows differ for {history_id}")
        if record["healthy_rows"] != expected["healthy_files"]:
            raise _fail(f"fit healthy support differs for {history_id}")
        if record["valid_patches"] != expected["valid_patches"]:
            raise _fail(f"fit valid patches differ for {history_id}")
        if not record["unique_ids"] or not record["all_rows_verified_healthy"]:
            raise _fail(f"fit support failed identity/eligibility for {history_id}")
        support_records["fit"].append(record)
        fit_rows.extend(rows)
    fit_ids = sorted(row.file_id for row in fit_rows)
    if len(fit_ids) != binding.fit_healthy_rows or len(set(fit_ids)) != binding.fit_healthy_rows:
        raise _fail(f"certified Fit support differs from the bound {binding.fit_healthy_rows} rows")
    fit_valid_patches = sum(int(row.manifest_row["n_valid_patches"]) for row in fit_rows)
    if fit_valid_patches != FIT_VALID_PATCHES:
        raise _fail(f"certified Fit patches differ from the bound {FIT_VALID_PATCHES}")
    calibration_rows_all: list = []
    for history_id in calibration_history_ids:
        history, manifest, manifest_path, shard_paths = _role_assets(binding, worktree, history_id)
        expected = binding_doc["calibration_support"][history_id]
        normal_ids = [row["file_id"] for row in manifest["files"] if row.get("file_label") == "normal"]
        rows = load_certified_sample_subset(
            binding, history_id, manifest_path, shard_paths, normal_ids,
            role_action="calibrate_q95", verify_all_shards=True,
        )
        record = _support_record(history, manifest, manifest_path, rows)
        if record["manifest_sha256_observed"] != expected["manifest_sha256"]:
            raise _fail(f"calibration manifest digest differs for {history_id}")
        if record["healthy_rows"] != expected["healthy_files"]:
            raise _fail(f"calibration healthy support differs for {history_id}")
        support_records["calibration"].append(record)
        calibration_rows_all.extend(rows)
    if len(calibration_rows_all) != binding.calibration_healthy_rows:
        raise _fail("certified Calibration support differs from the bound healthy rows")
    for history_id in development_history_ids:
        history, manifest, manifest_path, _shard_paths = _role_assets(binding, worktree, history_id)
        expected = binding_doc["development_support"][history_id]
        if _sha256_file(manifest_path) != expected["manifest_sha256"]:
            raise _fail(f"development manifest digest differs for {history_id}")
        if len(manifest["files"]) != expected["manifest_file_rows"]:
            raise _fail(f"development manifest rows differ for {history_id}")
        support_records["development"].append({
            "history_id": history_id,
            "role": history.assigned_role,
            "manifest_sha256_bound": history.manifest_sha256,
            "manifest_sha256_observed": _sha256_file(manifest_path),
            "manifest_file_rows_observed": len(manifest["files"]),
        })
    batches_per_epoch = math.ceil(len(fit_ids) / BATCH_SIZE)
    if batches_per_epoch != math.ceil(FIT_HEALTHY_ROWS / BATCH_SIZE):
        raise _fail("Fit batch geometry differs from the frozen 128-file epoch schedule")
    task14_hashes, task14_lock_sha256, task14_arms = _task14_lock_config_hashes(worktree)
    locked = _assert_locked_n2(task14_arms)
    lock_digest = binding_doc["coefficient_lock_sha256"]
    arm_hashes: dict[str, str] = {}
    sampler_records: list[dict] = []
    for seed in MODEL_SEEDS:
        arm = _n2_arm(seed, lock_digest)
        observed_config_sha256 = arm_config_sha256(arm)
        expected_config_sha256 = task14_hashes.get(f"{ARM_ID}/seed-{seed}")
        if observed_config_sha256 != expected_config_sha256:
            raise _fail(
                f"seed {seed}: N2 arm config {observed_config_sha256} differs from the Task14 lock "
                f"{expected_config_sha256}"
            )
        arm_hashes[str(seed)] = observed_config_sha256
        for epoch in (1, 10, 30, 60, 100):
            epoch_seed, order = deterministic_sampler_order(
                binding, arm, workload="representation", scope="experiment",
                role_history_ids=fit_history_ids, epoch=epoch, eligible_source_ids=fit_ids,
            )
            if len(order) != len(fit_ids) or set(order) != set(fit_ids):
                raise _fail(f"sampler epoch {epoch} seed {seed} is not a permutation of the loaded Fit support")
            sampler_records.append({
                "model_seed": seed, "epoch": epoch, "sampler_epoch_seed": epoch_seed,
                "order_rows": len(order), "order_sha256": _canonical_sha256(list(order)),
            })
    fit_role_sha256 = _canonical_sha256(sorted(binding.history(h).manifest_sha256 for h in fit_history_ids))
    # Bounded mask-allocation observation on one real Fit batch (no optimizer step):
    # the frozen N2 v2 integer allocation, group support, and fallback counts.
    from representation.trainer import RepresentationTrainer

    config = _n2_config(MODEL_SEEDS[0])
    patchifier = _n2_patchifier()
    sample_by_id = {row.file_id: row.sample for row in fit_rows}
    first_epoch_seed, first_order = deterministic_sampler_order(
        binding, _n2_arm(MODEL_SEEDS[0], lock_digest), workload="representation",
        scope="experiment", role_history_ids=fit_history_ids, epoch=1,
        eligible_source_ids=fit_ids,
    )
    del first_epoch_seed
    probe_ids = list(first_order[:BATCH_SIZE])
    probe = collate_variable_files(
        [sample_by_id[file_id] for file_id in probe_ids], patchifier,
        masking_config=config, masking_seed=1,
    )
    masks = RepresentationTrainer.msp_batch_masks(
        probe, model_seed=MODEL_SEEDS[0], epoch=1, fit_role_sha256=fit_role_sha256,
    )
    diagnostics = masks["diagnostics"]
    union_positions = int(masks["union_mask"].sum().item())
    allocation_observation = {
        "files": len(diagnostics),
        "mask_union_positions": union_positions,
        "mask_budget_positions": int(sum(int(item["M"]) for item in diagnostics)),
        "support_coarse_g4": int(sum(int(item["q4"]) for item in diagnostics)),
        "support_coarse_g8": int(sum(int(item["q8"]) for item in diagnostics)),
        "fine_positions": int(sum(int(item["fine"]) for item in diagnostics)),
        "q8_fallback_to_4": int(sum(int(item["q8_fallback_to_4"]) for item in diagnostics)),
        "q4_fallback_to_fine": int(sum(int(item["q4_fallback_to_fine"]) for item in diagnostics)),
        "sample": [dict(item) for item in diagnostics[:4]],
    }
    if union_positions != allocation_observation["mask_budget_positions"]:
        raise _fail("N2 mask union does not equal the exact integer budget on the observation batch")
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "task_id": TASK_ID, "attempt_id": ATTEMPT, "agent_id": AGENT,
        "source_commit": environment["git_head"],
        "runner_sha256": environment["runner_sha256"],
        "binding_json_sha256": environment["binding_json_sha256"],
        "binding_executable_sha256": binding.executable_binding_sha256,
        "arm": {
            "arm_id": ARM_ID,
            "component_set": list(locked["component_set"]),
            "principal_change": locked.get("principal_change"),
            "alpha": ALPHA,
            "lambda_info_max": LAMBDA_INFO_MAX,
            "coarse_scales": list(COARSE_SCALES),
            "mask_total_ratio": MSP_TOTAL_MASK_RATIO,
            "position_shares": list(MSP_POSITION_SHARES),
            "fine_mask_composition": list(MSP_FINE_MASK_COMPOSITION),
            "fine_mask_strata": list(MSP_FINE_MASK_STRATA),
            "fine_mask_block_range": list(MSP_FINE_MASK_BLOCK_RANGE),
            "scoring_masking_policy": "b0-composition",
            "config_sha256": arm_hashes,
            "task14_lock_sha256": task14_lock_sha256,
            "locked_coefficients": dict(locked["coefficients"]),
        },
        "history_ids": {"fit": fit_history_ids, "calibration": calibration_history_ids, "development": development_history_ids},
        "fit_total": {
            "healthy_rows": len(fit_ids), "valid_patches": fit_valid_patches,
            "eligible_ids_sha256": _canonical_sha256(fit_ids), "fit_role_sha256": fit_role_sha256,
        },
        "support_ids_sha256": {
            "fit": _canonical_sha256(fit_ids),
            "calibration": _canonical_sha256(sorted(row.file_id for row in calibration_rows_all)),
        },
        "batch_geometry": {
            "batch_size": BATCH_SIZE, "batches_per_epoch": batches_per_epoch,
            "epochs": EPOCHS, "optimizer_steps_per_seed": EPOCHS * batches_per_epoch,
        },
        "support": support_records,
        "sampler_checks": sampler_records,
        "mask_allocation_observation": allocation_observation,
        "preflight_wall_s": time.time() - started,
    }
    if EPOCHS * batches_per_epoch != 4300:
        raise _fail("frozen step arithmetic broken: expected 4300")
    _write_json(out_dir / "preflight.json", payload)
    print(json.dumps({
        "PREFLIGHT_OK": True, "fit_rows": len(fit_ids), "fit_patches": fit_valid_patches,
        "calibration_rows": len(calibration_rows_all),
        "steps_per_seed": EPOCHS * batches_per_epoch,
        "coarse_g4": allocation_observation["support_coarse_g4"],
        "coarse_g8": allocation_observation["support_coarse_g8"],
    }, sort_keys=True), flush=True)
    return {
        "fit_rows": fit_rows, "fit_ids": fit_ids,
        "fit_sample_by_id": sample_by_id,
        "calibration_rows": calibration_rows_all,
        "calibration_rows_by_history": {
            h: [r for r in calibration_rows_all if r.actual_history_id == h] for h in calibration_history_ids
        },
        "fit_history_ids": fit_history_ids,
        "calibration_history_ids": calibration_history_ids,
        "development_history_ids": development_history_ids,
        "payload": payload,
    }


def _n2_patchifier() -> Patchifier:
    return Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))


def _n2_config(seed: int) -> V1Config:
    return V1Config(
        n_channels=6, patch_size=32, stride=16, n_robots=9, n_programs=8,
        seed=seed, contrastive_weight_max=LAMBDA_INFO_MAX, ema_decay=0.996,
    )


def _build_epoch_batches(*, binding, arm, epoch: int, config, patchifier, fit_ids, fit_sample_by_id, fit_history_ids):
    _, order = deterministic_sampler_order(
        binding, arm, workload="representation", scope="experiment",
        role_history_ids=fit_history_ids, epoch=epoch, eligible_source_ids=fit_ids,
    )
    batches = []
    batches_per_epoch = math.ceil(len(fit_ids) / BATCH_SIZE)
    global_step_base = (epoch - 1) * batches_per_epoch
    for batch_index in range(batches_per_epoch):
        batch_ids = order[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
        samples = [fit_sample_by_id[file_id] for file_id in batch_ids]
        # Frozen B0 lineage rule: the collated batch mask seed is the 1-based
        # global step index. The N2 trainer then derives the per-file union and
        # coarse masks from the frozen N2 v2 per-file/epoch seed rule and
        # replaces the collated mask before context encoding.
        batches.append(collate_variable_files(
            samples, patchifier, masking_config=config,
            masking_seed=global_step_base + batch_index + 1,
        ))
    return order, batches


def _sampler_state(*, binding, arm, epoch_seed: int, order, fit_history_ids, completed_epochs: int, current_epoch: int):
    ordered = sorted(order)
    return {
        "scope": "experiment",
        "workload": "representation",
        "role_history_ids": list(fit_history_ids),
        "role_manifest_sha256": [binding.history(h).manifest_sha256 for h in fit_history_ids],
        "eligible_source_ids": ordered,
        "eligible_source_ids_sha256": _canonical_sha256(ordered),
        "sampler_epoch_seed": epoch_seed,
        "epoch_order_ids": list(order),
        "epoch_order_ids_sha256": _canonical_sha256(list(order)),
        "eligible_rows": len(ordered),
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": math.ceil(len(ordered) / BATCH_SIZE),
        "completed_epochs": completed_epochs,
        "current_epoch": current_epoch,
        "batches_in_current_epoch": 0,
        "fit_role_sha256": _canonical_sha256(sorted(binding.history(h).manifest_sha256 for h in fit_history_ids)),
    }


def _coefficient_state(*, arm, epoch: int, global_step: int):
    return {
        "epoch": epoch,
        "global_step": global_step,
        "selection_state": "locked",
        "lambda_info": coefficient_for_epoch(arm.lambda_info_max, epoch),
        "lambda_hcc": None,
        "alpha": arm.alpha,
        "beta": None,
    }


def _scheduler_state(*, completed_epochs: int, active_epoch: int, global_step: int):
    return {
        "scheduler_id": SCHEDULER_ID,
        "completed_epochs": completed_epochs,
        "active_epoch": active_epoch,
        "learning_rate": learning_rate_for_epoch(active_epoch),
        "global_step": global_step,
    }


def _save_coherent(*, binding, arm, model, optimizer, out_path: Path, global_step: int,
                   completed_epoch: int, active_epoch: int, initialization_sha256: str,
                   sampler_state: dict, coefficient_state: dict, scheduler_state: dict) -> str:
    return save_sprint18_checkpoint(
        out_path, model, optimizer, arm=arm, binding=binding,
        global_step=global_step, completed_epoch=completed_epoch,
        initialization_sha256=initialization_sha256,
        scheduler_state=scheduler_state, sampler_state=sampler_state,
        coefficient_state=coefficient_state, scope="experiment",
    )


def _development_assets(*, binding, worktree: Path, history_id: str):
    history, manifest, manifest_path, shard_paths = _role_assets(binding, worktree, history_id)
    rows = load_certified_development_support(binding, history_id, manifest_path, shard_paths)
    if len(rows) != len(manifest["files"]):
        raise _fail(f"Development support differs from its manifest for {history_id}")
    return history, manifest, rows


def _coarse_descriptive(*, binding, arm, model, patchifier, checkpoint, worktree,
                        development_history_ids, device) -> dict:
    """Descriptive coarse-4/coarse-8 residual report; never fused into a primary metric."""
    from synth import events as E
    from synth import probe15 as P

    per_history: dict[str, dict] = {}
    scale_scores: dict[str, dict[str, float]] = {"group4": {}, "group8": {}, "coarse": {}}
    rows_by_history: list[list[dict]] = []
    ledgers_by_history: list[list[dict]] = []
    windows_by_history: list[dict] = []
    durations: list[int] = []
    for history_id in development_history_ids:
        _history, manifest, rows = _development_assets(binding=binding, worktree=worktree, history_id=history_id)
        scored = score_msp_coarse_rows(
            binding, arm, model, rows, patchifier, checkpoint=checkpoint,
            action="score_development", scope="experiment",
            diagnostic_epoch=DIAGNOSTIC_EPOCH, device=device,
        )
        ledger = E.failure_ledger(manifest)
        windows = manifest["maintenance_windows"]
        cohort: dict[str, str] = {}
        for failure in ledger:
            if E.positive_window_intersects_reset(failure, windows):
                continue
            for candidate in E.pos_files(manifest["files"], failure, windows):
                cohort.setdefault(candidate["file_id"], failure["cohort"])
        rows_by_id = {row.file_id: row for row in rows}
        records = []
        for record in scored.records:
            sample = rows_by_id[record.file_id].sample
            durations.append(int(sample.T))
            for scale, value in (
                ("group4", record.group4_loss),
                ("group8", record.group8_loss),
                ("coarse", record.coarse_loss),
            ):
                scale_scores[scale][record.file_id] = float(value)
            records.append({
                "file_id": record.file_id,
                "file_label": rows_by_id[record.file_id].sample.file_label.value,
                "cohort": cohort.get(record.file_id, "none"),
                "duration_timesteps": int(sample.T),
                "n_valid_patches": record.n_valid_patches,
                "mask_budget": record.mask_budget,
                "allocation": list(record.allocation),
                "groups_4": record.groups_4,
                "groups_8": record.groups_8,
                "group4_loss": record.group4_loss,
                "group8_loss": record.group8_loss,
                "coarse_loss": record.coarse_loss,
                "q8_fallback_to_4": record.q8_fallback_to_4,
                "q4_fallback_to_fine": record.q4_fallback_to_fine,
            })
        per_history[history_id] = {
            "support_sha256": scored.support.support_sha256,
            "checkpoint_sha256": scored.checkpoint_sha256,
            "diagnostic_epoch": scored.diagnostic_epoch,
            "files": len(records),
            "records": records,
        }
        rows_by_history.append(manifest["files"])
        ledgers_by_history.append(ledger)
        windows_by_history.append(windows)
    pooled = np.asarray(durations, dtype=np.float64)
    tertiles = (
        float(np.quantile(pooled, 1 / 3)) if pooled.size else 0.0,
        float(np.quantile(pooled, 2 / 3)) if pooled.size else 0.0,
    )

    def bucket(duration: int) -> str:
        if duration <= tertiles[0]:
            return "short"
        if duration <= tertiles[1]:
            return "medium"
        return "long"

    def stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.90)),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    scales: dict[str, dict] = {}
    for scale in ("group4", "group8", "coarse"):
        by_label: dict[str, list[float]] = {}
        by_cohort: dict[str, list[float]] = {}
        by_duration: dict[str, list[float]] = {}
        for history_id in development_history_ids:
            for record in per_history[history_id]["records"]:
                value = float(record[f"{scale}_loss"] if scale != "coarse" else record["coarse_loss"])
                by_label.setdefault(record["file_label"], []).append(value)
                by_cohort.setdefault(record["cohort"], []).append(value)
                by_duration.setdefault(bucket(int(record["duration_timesteps"])), []).append(value)
        auc = P.evaluate_histories(
            scale_scores[scale], rows_by_history, ledgers_by_history, windows_by_history, 0.0,
        )
        scales[scale] = {
            "residual": stats([value for values in by_label.values() for value in values]),
            "by_label": {label: stats(values) for label, values in sorted(by_label.items())},
            "by_cohort": {label: stats(values) for label, values in sorted(by_cohort.items())},
            "by_duration": {label: stats(values) for label, values in sorted(by_duration.items())},
            "descriptive_auroc": {
                "P_AUROC": [entry["auc_p"] for entry in auc["per_history"]],
                "W_AUROC": [entry["auc_w"] for entry in auc["per_history"]],
                "PW_AUROC": [entry["auc_pw"] for entry in auc["per_history"]],
                "A_AUROC": [entry["auc_a"] for entry in auc["per_history"]],
                "macro": {
                    "P_macro": auc["macro_auc_p"],
                    "W_macro": auc["macro_auc_w"],
                    "PW_macro": auc["macro_auc_pw"],
                },
                "threshold_used_for_auc_only": 0.0,
                "note": (
                    "AUROC is threshold-invariant; the 0.0 threshold was passed only to reuse the "
                    "frozen probe15 metric code. Recall/FAR from this call are not reported and no "
                    "coarse threshold, fusion, or promotion decision is implied."
                ),
            },
        }
    return {
        "schema": COARSE_DIAGNOSTIC_SCHEMA,
        "arm_id": ARM_ID,
        "scales": list(COARSE_SCALES),
        "diagnostic_epoch": DIAGNOSTIC_EPOCH,
        "fusion": "none; coarse residuals are descriptive diagnostics only",
        "duration_tertile_bounds_timesteps": {"short_max": tertiles[0], "medium_max": tertiles[1]},
        "per_history": per_history,
        "aggregate": scales,
        "support": {
            "evaluation_histories": list(development_history_ids),
            "per_history_support_sha256": {
                history_id: per_history[history_id]["support_sha256"] for history_id in development_history_ids
            },
        },
    }


def _score_development_descriptive(*, binding, arm, model, bank, patchifier, checkpoint,
                                   worktree, development_history_ids, thresholds,
                                   scope: str = "experiment") -> dict:
    """Descriptive Development scoring on frozen common support (no selection)."""
    from synth import events as E
    from synth import probe15 as P

    per_branch_file_scores: dict = {}
    per_history_rows: dict = {}
    per_history_ledgers: dict = {}
    per_history_windows: dict = {}
    per_history_supports: dict = {}
    for history_id in development_history_ids:
        history = binding.history(history_id)
        role_dir = worktree / history.relative_path
        manifest_path = role_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_paths = {row["path"]: str(role_dir / row["path"]) for row in manifest["shards"]}
        rows = load_certified_development_support(binding, history_id, manifest_path, shard_paths)
        scores = score_certified_rows(
            binding, arm, model, bank, rows, patchifier,
            checkpoint=checkpoint, action="score_development", scope=scope,
        )
        validate_score_maps(scores.support, scores.scores)
        per_branch_file_scores[history_id] = (dict(scores.scores["S_pred"]), dict(scores.scores["S_pop"]))
        per_history_rows[history_id] = manifest["files"]
        per_history_ledgers[history_id] = E.failure_ledger(manifest)
        per_history_windows[history_id] = manifest["maintenance_windows"]
        per_history_supports[history_id] = {
            "support_sha256": scores.support.support_sha256,
            "file_ids": list(scores.support.file_ids),
            "branch_provenance_sha256": dict(scores.branch_provenance_sha256),
        }
    branch_results: dict = {}
    for branch_index, branch in enumerate(("S_pred", "S_pop")):
        file_scores: dict = {}
        for history_id in development_history_ids:
            file_scores.update(per_branch_file_scores[history_id][branch_index])
        rows_by_history = [per_history_rows[h] for h in development_history_ids]
        ledgers_by_history = [per_history_ledgers[h] for h in development_history_ids]
        wins_by_history = [per_history_windows[h] for h in development_history_ids]
        canonical = P.evaluate_histories(file_scores, rows_by_history, ledgers_by_history,
                                         wins_by_history, thresholds[branch])
        per_history = []
        for index, history_id in enumerate(development_history_ids):
            entry = canonical["per_history"][index]
            per_history.append({
                "history_id": history_id,
                "schema_history_id": binding.history(history_id).schema_history_id,
                "role_manifest_sha256": binding.history(history_id).manifest_sha256,
                "support_sha256": per_history_supports[history_id]["support_sha256"],
                "P_AUROC": entry["auc_p"], "W_AUROC": entry["auc_w"],
                "PW_AUROC": entry["auc_pw"], "A_AUROC": entry["auc_a"],
                "P_recall": entry["recall_p"], "P_median_lead_days": entry["lead_p_median"],
                "W_recall": entry["recall_w"], "W_median_lead_days": entry["lead_w_median"],
                "FAR_per_robot_day": entry["far"], "robot_days": entry["robot_days"],
                "false_episodes": entry["false_episodes"],
                "n_p": entry["n_p"], "n_w": entry["n_w"],
            })
        branch_results[branch] = {
            "threshold": thresholds[branch],
            "macro": {
                "P_macro": canonical["macro_auc_p"], "W_macro": canonical["macro_auc_w"],
                "PW_macro": canonical["macro_auc_pw"], "PW_lcb95": canonical["lcb_pw"],
                "directional_histories": canonical.get("directional_histories"),
            },
            "per_history": per_history,
        }
    return {
        "branches": branch_results,
        "support": {
            "evaluation_histories": list(development_history_ids),
            "per_history": per_history_supports,
        },
    }


def _coefficient_lock(binding) -> str:
    lock_path = Path(__file__).resolve().parents[1] / TASK14_LOCK_RELATIVE
    return json.loads(lock_path.read_text(encoding="utf-8"))["coefficient_lock_sha256"]


def _run_seed(
    *, binding, worktree: Path, seed: int, fit_ids, fit_sample_by_id, fit_history_ids,
    calibration_rows, calibration_rows_by_history, development_history_ids,
    device: torch.device, progress, seed_dir: Path, fit_rows_by_id: dict,
) -> dict:
    from representation.trainer import RepresentationTrainer

    run_id = f"{RUN_ID_PREFIX}-seed{seed}"
    run_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    config = _n2_config(seed)
    patchifier = _n2_patchifier()
    model = V1RepresentationModel(config, patchifier=patchifier)
    lock_digest = _coefficient_lock(binding)
    arm = _n2_arm(seed, lock_digest)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0)
    trainer = RepresentationTrainer(
        model, optimizer=optimizer, max_grad_norm=1.0, device=device, seed=seed,
    )
    fit_role_sha256 = _canonical_sha256(sorted(binding.history(h).manifest_sha256 for h in fit_history_ids))
    initialization_sha256 = state_sha256(model.state_dict())
    batches_per_epoch = math.ceil(len(fit_ids) / BATCH_SIZE)
    seed_checkpoints: dict = {}
    exact_lambdas: dict = {}
    exact_lrs: dict = {}
    train_start = time.time()
    saved_step300: dict | None = None
    for epoch in range(1, EPOCHS + 1):
        _, batches = _build_epoch_batches(
            binding=binding, arm=arm, epoch=epoch, config=config, patchifier=patchifier,
            fit_ids=fit_ids, fit_sample_by_id=fit_sample_by_id, fit_history_ids=fit_history_ids,
        )
        metrics = trainer.train_sprint18_msp_epoch(
            batches, arm=arm, epoch=epoch, collect_balance=True, fit_role_sha256=fit_role_sha256,
        )
        expected_step = epoch * batches_per_epoch
        if trainer.step != expected_step:
            raise _fail(f"seed {seed} epoch {epoch}: trainer step {trainer.step} != deterministic position {expected_step}")
        row = trainer.objective_balance_history[-1]
        exact_lambdas[str(epoch)] = row["effective_coefficient_L_info"]
        exact_lrs[str(epoch)] = row["lr"]
        if row["lr"] != learning_rate_for_epoch(epoch):
            raise _fail(f"seed {seed} epoch {epoch}: LR drift")
        if row["effective_coefficient_L_info"] != coefficient_for_epoch(LAMBDA_INFO_MAX, epoch):
            raise _fail(f"seed {seed} epoch {epoch}: InfoNCE coefficient drift")
        if row["effective_coefficient_L_coarse"] != coefficient_for_epoch(ALPHA, epoch):
            raise _fail(f"seed {seed} epoch {epoch}: N2 alpha schedule drift")
        _log_progress(progress, {
            "run": run_id, "model_seed": seed, "epoch": epoch,
            "global_step": trainer.step, "lr": row["lr"],
            "lambda_info": row["effective_coefficient_L_info"],
            "alpha": row["effective_coefficient_L_coarse"],
            "prediction_loss": metrics.get("prediction_loss"),
            "contrastive_loss": metrics.get("contrastive_loss"),
            "coarse_loss": metrics.get("coarse_loss"),
            "weighted_coarse_loss": metrics.get("weighted_coarse_loss"),
            "joint_loss": metrics.get("joint_loss"),
            "raw_L_group4": row.get("raw_L_group4"),
            "raw_L_group8": row.get("raw_L_group8"),
            "support_coarse_g4": row.get("support_coarse_g4"),
            "support_coarse_g8": row.get("support_coarse_g8"),
            "support_prediction_patches": row.get("support_prediction_patches"),
            "elapsed_s": time.time() - train_start,
        })
        if epoch in DIAGNOSTIC_EPOCHS or epoch == EPOCHS:
            active_epoch = epoch + 1 if epoch < EPOCHS else EPOCHS
            active_lr = learning_rate_for_epoch(active_epoch)
            for group in optimizer.param_groups:
                group["lr"] = active_lr
            epoch_seed, next_order = deterministic_sampler_order(
                binding, arm, workload="representation", scope="experiment",
                role_history_ids=fit_history_ids, epoch=active_epoch, eligible_source_ids=fit_ids,
            )
            path = seed_dir / "checkpoints" / f"{run_id}-epoch{epoch:03d}.pt"
            digest = _save_coherent(
                binding=binding, arm=arm, model=model, optimizer=optimizer, out_path=path,
                global_step=trainer.step, completed_epoch=epoch, active_epoch=active_epoch,
                initialization_sha256=initialization_sha256,
                sampler_state=_sampler_state(binding=binding, arm=arm, epoch_seed=epoch_seed,
                                             order=next_order, fit_history_ids=fit_history_ids,
                                             completed_epochs=epoch, current_epoch=active_epoch),
                coefficient_state=_coefficient_state(arm=arm, epoch=active_epoch, global_step=trainer.step),
                scheduler_state=_scheduler_state(completed_epochs=epoch, active_epoch=active_epoch, global_step=trainer.step),
            )
            seed_checkpoints[str(epoch)] = {
                "path": str(path), "sha256": digest, "global_step": trainer.step,
                "completed_epoch": epoch, "active_epoch": active_epoch,
            }
        # Frozen step-300 diagnostic: 300 = 6*43 + 42, inside epoch 7. The
        # coherent schema requires sampler/global-step coherence, so record the
        # epoch-7 boundary truth instead of a padded mid-epoch checkpoint.
        if saved_step300 is None and trainer.step >= STEP_300:
            saved_step300 = {
                "global_step_300_in_epoch": epoch,
                "epoch7_complete_step": 7 * batches_per_epoch,
                "reached": True,
                "checkpoint_policy": "epoch-boundary-only",
            }
    train_wall_s = time.time() - train_start
    global_step = EPOCHS * batches_per_epoch
    if trainer.step != global_step:
        raise _fail(f"seed {seed}: final step {trainer.step} != frozen {global_step}")
    primary = seed_checkpoints[str(EPOCHS)]
    live_model_sha = state_sha256(model.state_dict())
    live_ema_sha = state_sha256(model.target_encoder.state_dict())
    restored_model = V1RepresentationModel(config, patchifier=patchifier)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=learning_rate_for_epoch(EPOCHS), weight_decay=0.0,
    )
    restored = restore_sprint18_checkpoint(
        Path(primary["path"]), restored_model, restored_optimizer,
        arm=arm, binding=binding, allowed_scope="experiment",
    )
    if restored.checkpoint_sha256 != primary["sha256"]:
        raise _fail(f"seed {seed}: strict restore digest mismatch")
    if state_sha256(restored_model.state_dict()) != live_model_sha:
        raise _fail(f"seed {seed}: strict restore model bytes mismatch")
    if state_sha256(restored_model.target_encoder.state_dict()) != live_ema_sha:
        raise _fail(f"seed {seed}: strict restore EMA bytes mismatch")
    restored_model.to(device)
    probe_ids = fit_ids[: min(4, len(fit_ids))]
    probe = collate_variable_files(
        [fit_sample_by_id[i] for i in probe_ids], patchifier, masking_config=config, masking_seed=1,
    )
    model.eval()
    restored_model.eval()
    with torch.no_grad():
        live_out = model(_move(_encoder_input(probe), device))
        restored_out = restored_model(_move(_encoder_input(probe), device))
    emb_diff = float((live_out["file_embedding"] - restored_out["file_embedding"]).abs().max().item())
    if not math.isfinite(emb_diff) or emb_diff != 0.0:
        raise _fail(f"seed {seed}: restored forward differs by {emb_diff}")
    fit_rows_ordered = [fit_rows_by_id[i] for i in fit_ids]
    bank = fit_fit_reference_bank(
        binding, arm, restored_model, fit_rows_ordered, patchifier,
        checkpoint=restored, scope="experiment",
    )
    if bank.provenance.source_rows != len(fit_ids) or bank.provenance.scope != "experiment":
        raise _fail(f"seed {seed}: Fit bank provenance mismatch")
    calibration_scores = {}
    for history_id, rows in calibration_rows_by_history.items():
        scores = score_certified_rows(
            binding, arm, restored_model, bank, rows, patchifier,
            checkpoint=restored, action="calibrate_q95", scope="experiment",
        )
        validate_score_maps(scores.support, scores.scores)
        calibration_scores[history_id] = {
            "support_sha256": scores.support.support_sha256,
            "file_ids": list(scores.support.file_ids),
            "S_pred": dict(scores.scores["S_pred"]),
            "S_pop": dict(scores.scores["S_pop"]),
            "branch_provenance_sha256": dict(scores.branch_provenance_sha256),
            "checkpoint_sha256": scores.checkpoint_sha256,
            "reference_sha256": scores.bank_sha256,
        }
    calibration_scores_obj = score_certified_rows(
        binding, arm, restored_model, bank, calibration_rows, patchifier,
        checkpoint=restored, action="calibrate_q95", scope="experiment",
    )
    calibration = fit_calibration_q95(binding, calibration_scores_obj, calibration_rows, scope="experiment")
    thresholds = {
        "S_pred": calibration.S_pred.threshold,
        "S_pop": calibration.S_pop.threshold,
    }
    development = _score_development_descriptive(
        binding=binding, arm=arm, model=restored_model, bank=bank, patchifier=patchifier,
        checkpoint=restored, worktree=worktree, development_history_ids=development_history_ids,
        thresholds=thresholds,
    )
    coarse = _coarse_descriptive(
        binding=binding, arm=arm, model=restored_model, patchifier=patchifier,
        checkpoint=restored, worktree=worktree,
        development_history_ids=development_history_ids, device=device,
    )
    probe_row_ids = [row.file_id for row in fit_rows_ordered[:1]]
    flop_batch = collate_variable_files(
        [fit_sample_by_id[i] for i in probe_row_ids], patchifier, masking_config=config,
        masking_seed=deterministic_mask_seed(seed, fit_history_ids[0], probe_row_ids[0]),
    )
    inference = RepresentationInference(restored_model, bank.bank, patchifier, masking_config=config)
    flop_input = inference.co_locate_batch(_encoder_input(flop_batch))  # type: ignore[arg-type]
    inference_flops = profile_inference_flops(
        lambda: inference.score_batch(flop_input),  # type: ignore[arg-type]
        device="cuda" if device.type == "cuda" else "cpu",
    )
    parameter_counts = {
        branch: primary_inference_parameters(restored_model, branch)  # type: ignore[arg-type]
        for branch in ("S_pred", "S_pop")
    }
    coarse_head_parameters = {
        scale: sum(parameter.numel() for parameter in module.parameters())
        for scale, module in (
            ("group4", restored_model.coarse_predictor_4),
            ("group8", restored_model.coarse_predictor_8),
        )
    }
    peak_allocated = (
        int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else None
    )
    balance_records = [dict(record) for record in trainer.objective_balance_history]
    history = [dict(record) for record in trainer.history]
    record = {
        "status": "ok", "run_id": run_id, "arm_id": ARM_ID, "model_seed": seed,
        "arm_config_sha256": arm_config_sha256(arm),
        "arm_identity": {
            "component_set": list(arm.component_set),
            "principal_change": arm.principal_change,
            "mechanism_spec_sha256": arm.mechanism_spec_sha256,
            "alpha": ALPHA,
            "lambda_info_max": LAMBDA_INFO_MAX,
            "coarse_scales": list(COARSE_SCALES),
            "mask_total_ratio": MSP_TOTAL_MASK_RATIO,
        },
        "initialization_sha256": initialization_sha256,
        "global_step": global_step,
        "representation_epochs_completed": EPOCHS,
        "batches_per_epoch": batches_per_epoch,
        "exact_schedule": {
            "alpha": ALPHA,
            "lambda_info_max": LAMBDA_INFO_MAX,
            "per_epoch_lambda": exact_lambdas,
            "per_epoch_lr": exact_lrs,
        },
        "checkpoints": seed_checkpoints,
        "step300": saved_step300,
        "primary": {
            **primary,
            "live_model_state_sha256": live_model_sha,
            "live_ema_state_sha256": live_ema_sha,
            "restored_checkpoint_sha256": restored.checkpoint_sha256,
            "restored_forward_max_abs_diff": emb_diff,
        },
        "bank": {
            "bank_sha256": bank.provenance.bank_sha256,
            "checkpoint_sha256": bank.provenance.checkpoint_sha256,
            "source_rows": bank.provenance.source_rows,
            "source_rows_sha256": bank.provenance.source_rows_sha256,
            "source_history_ids": list(bank.provenance.source_history_ids),
        },
        "calibration_scores": calibration_scores,
        "calibration": {
            "S_pred_threshold": calibration.S_pred.threshold,
            "S_pop_threshold": calibration.S_pop.threshold,
            "source_rows": calibration.S_pred.source_rows,
            "source_support_sha256": calibration.S_pred.source_support_sha256,
        },
        "development": development,
        "coarse_diagnostics": coarse,
        "compute": {
            "primary_inference_parameters": parameter_counts,
            "coarse_head_parameters": coarse_head_parameters,
            "coarse_head_ratio_to_primary_S_pred": {
                scale: value / parameter_counts["S_pred"]
                for scale, value in coarse_head_parameters.items()
            },
            "inference_flops_per_fit_file_combined": inference_flops,
            "flops_probe_file": probe_row_ids[0],
            "baseline_compute_claim": (
                "No paired B0 parameter/FLOP counters were recomputed; the accepted B0 seed records "
                "carry no compute block. The fine primary inference graph is the unchanged B0 path and "
                "coarse heads are excluded from inference, so primary parameter/FLOP parity holds by "
                "construction, not by a measured B0 ratio."
            ),
        },
        "resources": {
            "device": str(device),
            "device_name": torch.cuda.get_device_properties(device).name if device.type == "cuda" else "cpu",
            "peak_cuda_memory_allocated_bytes": peak_allocated,
            "peak_cuda_memory_reserved_bytes": peak_reserved,
            "peak_scope": (
                "torch peak since the start of this seed run, covering training, strict restore, "
                "the Fit bank, q95 calibration, Development scoring, coarse diagnostics and the "
                "FLOP profile; it is not a training-only peak"
            ),
            "train_wall_s": train_wall_s,
            "total_wall_s": time.time() - run_start,
            "files_processed": len(fit_ids) * EPOCHS,
            "steps_per_s": global_step / train_wall_s if train_wall_s > 0 else None,
        },
        "balance_records": len(balance_records),
        "history_epochs": len(history),
    }
    _write_json(seed_dir / "seed_record.json", record)
    _write_json(seed_dir / "coarse_diagnostics.json", coarse)
    del trainer, optimizer, model, restored_model, restored_optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "record": record, "balance_records": balance_records, "history": history,
        "arm": arm, "config": config,
    }


PAIRED_MACRO_METRICS = ("P_macro", "W_macro", "PW_macro", "PW_lcb95", "directional_histories")
PAIRED_HISTORY_METRICS = (
    "P_AUROC",
    "W_AUROC",
    "PW_AUROC",
    "A_AUROC",
    "P_recall",
    "W_recall",
    "P_median_lead_days",
    "W_median_lead_days",
    "FAR_per_robot_day",
)


def _paired_value(arm_value: object, b0_value: object, *, label: str) -> dict:
    """Return the paired N2/B0 value and signed difference (fail closed on gaps)."""
    for value, source in ((arm_value, "N2"), (b0_value, "B0")):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise _fail(f"paired metric {label} is missing or nonfinite in the {source} record")
    return {"arm": float(arm_value), "b0": float(b0_value), "delta": float(arm_value) - float(b0_value)}


def _paired_development(*, arm_records: dict, b0_records: dict, b0_records_sha256: str) -> dict:
    """Pair N2-MSP Development outputs with the accepted B0-FULL seed records."""
    expected_keys = [str(seed) for seed in MODEL_SEEDS]
    if sorted(arm_records) != expected_keys or sorted(b0_records) != expected_keys:
        raise _fail("paired Development requires exactly the frozen three model seeds")
    expected_steps = EPOCHS * math.ceil(FIT_HEALTHY_ROWS / BATCH_SIZE)
    calibration_support_sha256: dict[str, str] = {}
    histories: list[str] | None = None
    per_history_support: dict[str, str] = {}
    for key in expected_keys:
        arm = arm_records[key]
        b0 = b0_records[key]
        if arm.get("arm_id") != ARM_ID or arm.get("status") != "ok":
            raise _fail(f"N2 seed {key} record is not a completed {ARM_ID} run")
        if b0.get("arm_id") != "B0-FULL" or b0.get("status") != "ok":
            raise _fail(f"B0 seed {key} record is not a completed B0-FULL run")
        for record, label in ((arm, "N2"), (b0, "B0")):
            if (
                record.get("global_step") != expected_steps
                or record.get("representation_epochs_completed") != EPOCHS
                or record.get("batches_per_epoch") != math.ceil(FIT_HEALTHY_ROWS / BATCH_SIZE)
            ):
                raise _fail(f"{label} seed {key} did not complete the frozen 100-epoch schedule")
        arm_support = arm["development"]["support"]
        b0_support = b0["development"]["support"]
        if arm_support["evaluation_histories"] != b0_support["evaluation_histories"]:
            raise _fail(f"N2/B0 Development history roster differs for seed {key}")
        if histories is None:
            histories = list(arm_support["evaluation_histories"])
        elif list(arm_support["evaluation_histories"]) != histories:
            raise _fail("Development history roster differs across seeds")
        for history_id in histories:
            arm_hash = arm_support["per_history"][history_id]["support_sha256"]
            b0_hash = b0_support["per_history"][history_id]["support_sha256"]
            if arm_hash != b0_hash:
                raise _fail(f"paired Development support differs for {history_id} (seed {key})")
            per_history_support.setdefault(history_id, arm_hash)
            if per_history_support[history_id] != arm_hash:
                raise _fail(f"Development support differs across seeds for {history_id}")
        arm_cal = arm["calibration"]["source_support_sha256"]
        b0_cal = b0["calibration"]["source_support_sha256"]
        if arm_cal != b0_cal:
            raise _fail(f"N2/B0 Calibration support differs for seed {key}")
        calibration_support_sha256[key] = arm_cal
        if arm["bank"]["source_rows"] != b0["bank"]["source_rows"]:
            raise _fail(f"N2/B0 Fit source rows differ for seed {key}")
    assert histories is not None
    branches: dict[str, object] = {}
    for branch in ("S_pred", "S_pop"):
        per_seed = []
        for key in expected_keys:
            arm_branch = arm_records[key]["development"]["branches"][branch]
            b0_branch = b0_records[key]["development"]["branches"][branch]
            arm_histories = {row["history_id"]: row for row in arm_branch["per_history"]}
            b0_histories = {row["history_id"]: row for row in b0_branch["per_history"]}
            if set(arm_histories) != set(histories) or set(b0_histories) != set(histories):
                raise _fail(f"{branch} per-history rows differ from the frozen Development roster")
            macro = {
                name: _paired_value(arm_branch["macro"].get(name), b0_branch["macro"].get(name),
                                    label=f"{branch}.{name}.seed-{key}")
                for name in PAIRED_MACRO_METRICS
            }
            per_history = []
            for history_id in histories:
                metrics = {
                    name: _paired_value(arm_histories[history_id].get(name), b0_histories[history_id].get(name),
                                        label=f"{branch}.{history_id}.{name}.seed-{key}")
                    for name in PAIRED_HISTORY_METRICS
                }
                per_history.append({"history_id": history_id, "metrics": metrics})
            per_seed.append({
                "model_seed": int(key),
                "arm_threshold": arm_branch["threshold"],
                "b0_threshold": b0_branch["threshold"],
                "macro": macro,
                "per_history": per_history,
            })
        aggregate_macro = {}
        for name in PAIRED_MACRO_METRICS:
            deltas = [row["macro"][name]["delta"] for row in per_seed]
            aggregate_macro[name] = {
                "mean_delta": sum(deltas) / len(deltas),
                "positive_seeds": sum(1 for delta in deltas if delta > 0),
            }
        aggregate_history = {}
        for history_id in histories:
            aggregate_history[history_id] = {}
            for name in PAIRED_HISTORY_METRICS:
                deltas = [
                    next(row for row in seed_row["per_history"] if row["history_id"] == history_id)
                    ["metrics"][name]["delta"]
                    for seed_row in per_seed
                ]
                aggregate_history[history_id][name] = {
                    "mean_delta": sum(deltas) / len(deltas),
                    "positive_seeds": sum(1 for delta in deltas if delta > 0),
                }
        branches[branch] = {
            "per_seed": per_seed,
            "aggregate": {"macro": aggregate_macro, "per_history": aggregate_history},
        }
    return {
        "schema": PAIRED_SCHEMA,
        "task_id": TASK_ID, "attempt_id": ATTEMPT, "agent_id": AGENT,
        "arm_id": ARM_ID, "baseline_arm_id": "B0-FULL",
        "b0_seed_records_sha256": b0_records_sha256,
        "model_seeds": list(MODEL_SEEDS),
        "support": {
            "evaluation_histories": histories,
            "per_history_support_sha256": per_history_support,
            "calibration_support_sha256": calibration_support_sha256,
        },
        "branches": branches,
        "interpretation": (
            "Descriptive paired Development differences only (N2-MSP minus B0-FULL at the same model "
            "seed and frozen support). Each arm uses its own q95 thresholds; no selection, promotion, "
            "pruning, coarse fusion, or rescue is applied."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default=",".join(str(s) for s in MODEL_SEEDS))
    parser.add_argument("--b0-records", default=None)
    args = parser.parse_args()

    worktree = Path(args.worktree).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runner_sha256 = _sha256_file(Path(__file__).resolve())
    if runner_sha256 != args.runner_sha256:
        raise _fail("runner digest differs from the prebound value")
    binding_path = Path(args.binding)
    binding_doc = json.loads(binding_path.read_text(encoding="utf-8"))
    binding_json_sha256 = _sha256_file(binding_path)
    if binding_doc.get("runner_sha256") != runner_sha256:
        raise _fail("runner digest differs from the binding record")
    if binding_doc.get("source_commit") != args.source_commit:
        raise _fail("source commit differs from the binding record")
    if binding_doc.get("model_seeds") != list(MODEL_SEEDS):
        raise _fail("bound model seeds differ from the frozen seed namespace")
    if binding_doc.get("epochs") != EPOCHS or binding_doc.get("batch_size") != BATCH_SIZE:
        raise _fail("bound epochs/batch size differ from the frozen contract")
    if binding_doc.get("lambda_info_max") != LAMBDA_INFO_MAX:
        raise _fail("bound InfoNCE coefficient differs from the Task14 lock")
    if binding_doc.get("alpha") != ALPHA:
        raise _fail("bound N2 alpha differs from the accepted Task14 lock")
    if binding_doc.get("arm_id") != ARM_ID:
        raise _fail("bound arm differs from N2-MSP")
    if binding_doc.get("device") != args.device:
        raise _fail("bound device differs from the requested device")
    head = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != args.source_commit:
        raise _fail(f"source commit differs: observed {head}")
    binding = load_task6_binding(worktree / args.handoff)
    if binding.executable_binding_sha256 != binding_doc.get("expected_binding_executable_sha256"):
        raise _fail("executable binding digest differs from the prebound certified binding")
    b0_records_path = Path(args.b0_records).resolve() if args.b0_records else None
    b0_records_sha256 = _sha256_file(b0_records_path) if b0_records_path is not None else None
    if args.mode == "full":
        if b0_records_path is None:
            raise _fail("full mode requires --b0-records")
        if b0_records_sha256 != binding_doc.get("b0_seed_records_sha256"):
            raise _fail("B0 seed-records digest differs from the binding record")
    device = torch.device(args.device)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise _fail("requested CUDA device is unavailable")
        properties = torch.cuda.get_device_properties(device)
        cuda_name = properties.name
        cuda_mem = int(properties.total_memory)
    else:
        cuda_name, cuda_mem = "cpu", 0
    environment = {
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "device": str(device), "device_name": cuda_name,
        "device_total_memory_bytes": cuda_mem,
        "git_head": head, "binding_json_sha256": binding_json_sha256,
        "binding_executable_sha256": binding.executable_binding_sha256,
        "runner_sha256": runner_sha256, "command": " ".join(sys.argv),
    }
    preflight = run_preflight(
        binding=binding, binding_doc=binding_doc, worktree=worktree,
        out_dir=out_dir, environment=environment,
    )
    if args.mode == "preflight":
        _write_json(out_dir / "environment.json", {
            **environment, "mode": "preflight",
            "preflight_sha256": _canonical_sha256(preflight["payload"]),
        })
        print(json.dumps({"RUNNER_EXIT": "preflight_ok"}, sort_keys=True), flush=True)
        return 0
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if tuple(seeds) != tuple(MODEL_SEEDS):
        raise _fail("full runs must use exactly the frozen model seeds")
    fit_rows_by_id = {row.file_id: row for row in preflight["fit_rows"]}
    progress = (out_dir / "progress.jsonl").open("a", encoding="utf-8")
    seed_records = {}
    try:
        for seed in seeds:
            seed_dir = out_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            result = _run_seed(
                binding=binding, worktree=worktree, seed=seed,
                fit_ids=preflight["fit_ids"],
                fit_sample_by_id=preflight["fit_sample_by_id"],
                fit_history_ids=preflight["fit_history_ids"],
                calibration_rows=preflight["calibration_rows"],
                calibration_rows_by_history=preflight["calibration_rows_by_history"],
                development_history_ids=preflight["development_history_ids"],
                device=device, progress=progress, seed_dir=seed_dir,
                fit_rows_by_id=fit_rows_by_id,
            )
            seed_records[str(seed)] = result["record"]
            _write_json(seed_dir / "balance_history.json", result["balance_records"])
            _write_json(seed_dir / "train_history.json", result["history"])
    finally:
        progress.close()
    _write_json(out_dir / "seed_records.json", seed_records)
    assert b0_records_path is not None and b0_records_sha256 is not None
    b0_records = json.loads(b0_records_path.read_text(encoding="utf-8"))
    paired = _paired_development(
        arm_records=seed_records, b0_records=b0_records, b0_records_sha256=b0_records_sha256,
    )
    _write_json(out_dir / "paired_development.json", paired)
    _write_json(out_dir / "environment.json", {
        **environment, "mode": "full", "b0_seed_records_sha256": b0_records_sha256,
    })
    print(json.dumps({
        "RUNNER_EXIT": "ok", "seeds": list(seeds), "paired_schema": paired["schema"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
