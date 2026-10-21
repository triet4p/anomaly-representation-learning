"""S18-T25 runner: locked K7-FULL experiment-scope training.

Reuses the accepted Task 24 K3-FULL runner procedure for the exact K7-FULL
arm: for each frozen seed (181801/181802/181803), run exactly 100 complete
locked `experiment`-scope K7-FULL epochs (exact Sprint 17 C2-A + C3-A + C5-A +
C6-A members) over caller-supplied Fit rows, save the frozen diagnostic
(epoch 10/30/60 + step-300 record) and primary epoch-100 coherent checkpoints,
strict-restore the primary, refit the Fit bank, score Calibration via public
inference, and calibrate q95 — all through committed public entry points only.

Calls only committed public entry points; implements no science, no metric
redefinition, no protocol amendment:

- representation.sprint18_k_dispatch exact K7 member assembly/config/criterion
- representation.sprint18_harness binding/loader/sampler/config/bank/scoring/q95
- representation.trainer.RepresentationTrainer.train_sprint18_epoch
- representation.sprint18_checkpoint schedules/save/strict restore
- representation.data.collate_variable_files
- representation.model / representation.criterion public forwards

The C5-A channel-time block masking policy is applied to training batches,
the restored parity probe, and Fit bank/Calibration/Development scoring (the
harness derives it from the arm identity). Full mode
also validates the computed per-seed arm configuration against the Task 14
lock hashes and emits a paired Development comparison against the accepted
Task 23 B0-FULL seed records. C2-A contributes the overlap-explicit B0-parity
patchifier and C3-A the residual dilated patch encoder; the K7 model is built
with its arm-owned patchifier (never the plain B0 patchifier object).

Eligibility is applied ONLY by the committed loaders
(load_certified_healthy_role_support / load_certified_sample_subset with
role_action) and the runner-local Development full-support loader
(_load_development_rows, verified under the DEVELOPMENT score_development
permission with manifest/shard/member checks and no health filtering).
This runner applies NO additional health/stage filter and reads
no simulator health anywhere.

Global RNG discipline: the protocol requires the sampler permutation, the
per-file/view augmentation streams, and the coherent checkpoint RNG state to be
deterministic. The only non-determinism sources are (a) model parameter
initialization and (b) data-loader-independent op scheduling. This runner seeds
Python/NumPy/CPU-Torch (and CUDA when present) once per (seed) BEFORE model
construction from the frozen model seed, so shared-module initialization is
seed-matched across arms where tensor shapes permit. Per-module stable
sub-stream derivation (sprint18-init-v1 keyed per-module seeds) is NOT
implemented by the reviewed model constructor; the exact K7 builder is used
unchanged. The full-run initialization digest is recorded per seed for
provenance; seed-matched pairing across arms at identical tensor shapes holds
by construction from the shared global seed.
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

def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[1], Path.cwd()):
        if (candidate / "src" / "representation").is_dir() and (candidate / "experiments").is_dir():
            return candidate
    raise SystemExit("DRIVER_FATAL cannot locate worktree root (src/representation missing)")


REPO_ROOT = _resolve_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from representation.data import collate_variable_files
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
    arm_config_sha256,
    deterministic_sampler_order,
    fit_calibration_q95,
    fit_fit_reference_bank,
    load_certified_healthy_role_support,
    load_certified_sample_subset,
    load_task6_binding,
    make_sprint18_arm_config,
    score_certified_rows,
    validate_score_maps,
)
from representation.sprint18_k_dispatch import (
    K_LAMBDA_INFO_MAX,
    build_k_config,
    build_k_criterion,
    build_k_model,
    c5a_criterion_identity,
    masking_policy_for_arm,
    member_numerics,
    members_for_arm,
)

EPOCHS = 100
DIAGNOSTIC_EPOCHS = (10, 30, 60)
STEP_300 = 300
TASK_ID = "S18-T25"
ATTEMPT = "S18-T25-A01"
AGENT = "S18T25A01"
ARM_ID = "K7-FULL"
RUN_ID_PREFIX = "k7"
PREFLIGHT_SCHEMA = "s18-t25-k7-preflight-v1"
PAIRED_SCHEMA = "s18-t25-k7-paired-development-v1"
LAMBDA_INFO_MAX = K_LAMBDA_INFO_MAX
MASKING_POLICY_ID = "c5a-channel-time-block-v1"
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


def _task14_lock_config_hashes(worktree: Path) -> tuple[dict, str]:
    """Return the Task 14 per-seed arm hashes and the lock file digest."""
    lock_path = worktree / TASK14_LOCK_RELATIVE
    raw = lock_path.read_bytes()
    lock = json.loads(raw)
    hashes = lock.get("config_sha256")
    if not isinstance(hashes, dict):
        raise _fail("Task14 lock is missing its per-seed arm config hashes")
    return hashes, hashlib.sha256(raw).hexdigest()


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
            raise _fail(f"fit healthy support differs for {history_id}: {record['healthy_rows']} != {expected['healthy_files']}")
        if record["valid_patches"] != expected["valid_patches"]:
            raise _fail(f"fit valid patches differ for {history_id}")
        if not record["unique_ids"] or not record["all_rows_verified_healthy"]:
            raise _fail(f"fit support failed identity/eligibility for {history_id}")
        support_records["fit"].append(record)
        fit_rows.extend(rows)
    fit_ids = sorted(row.file_id for row in fit_rows)
    if len(fit_ids) != binding.fit_healthy_rows or len(set(fit_ids)) != binding.fit_healthy_rows:
        raise _fail(f"certified Fit support differs from the bound {binding.fit_healthy_rows} rows: observed {len(fit_ids)}")
    fit_valid_patches = sum(int(row.manifest_row["n_valid_patches"]) for row in fit_rows)
    if fit_valid_patches != FIT_VALID_PATCHES:
        raise _fail(f"certified Fit patches differ from the bound {FIT_VALID_PATCHES}: observed {fit_valid_patches}")
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
    development_ids: dict = {}
    for history_id in development_history_ids:
        history, manifest, manifest_path, shard_paths = _role_assets(binding, worktree, history_id)
        expected = binding_doc["development_support"][history_id]
        if _sha256_file(manifest_path) != expected["manifest_sha256"]:
            raise _fail(f"development manifest digest differs for {history_id}")
        if len(manifest["files"]) != expected["manifest_file_rows"]:
            raise _fail(f"development manifest rows differ for {history_id}")
        development_ids[history_id] = sorted(row["file_id"] for row in manifest["files"] if isinstance(row, dict) and isinstance(row.get("file_id"), str))
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
    sampler_records = []
    task14_hashes, task14_lock_sha256 = _task14_lock_config_hashes(worktree)
    arm_hashes: dict[str, str] = {}
    for seed in MODEL_SEEDS:
        arm = make_sprint18_arm_config(
            ARM_ID, seed,
            {"lambda_info_max": LAMBDA_INFO_MAX, "lambda_hcc_max": None, "alpha": None, "beta": None},
            selection_state="locked", coefficient_lock_sha256=binding_doc["coefficient_lock_sha256"],
        )
        observed_config_sha256 = arm_config_sha256(arm)
        expected_config_sha256 = task14_hashes.get(f"{ARM_ID}/seed-{seed}")
        if observed_config_sha256 != expected_config_sha256:
            raise _fail(
                f"seed {seed}: K7 arm config {observed_config_sha256} differs from the Task14 lock "
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
    development_ids_flat = sorted(i for ids in development_ids.values() for i in ids)
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "task_id": TASK_ID, "attempt_id": ATTEMPT, "agent_id": AGENT,
        "source_commit": environment["git_head"],
        "runner_sha256": environment["runner_sha256"],
        "binding_json_sha256": environment["binding_json_sha256"],
        "binding_executable_sha256": binding.executable_binding_sha256,
        "arm": {
            "arm_id": ARM_ID,
            "members": list(members_for_arm(ARM_ID)),
            "member_numerics": {member: member_numerics(member) for member in members_for_arm(ARM_ID)},
            "criterion": dict(c5a_criterion_identity()),
            "lambda_info_max": LAMBDA_INFO_MAX,
            "masking_policy": MASKING_POLICY_ID,
            "config_sha256": arm_hashes,
            "task14_lock_sha256": task14_lock_sha256,
        },
        "history_ids": {"fit": fit_history_ids, "calibration": calibration_history_ids, "development": development_history_ids},
        "fit_total": {
            "healthy_rows": len(fit_ids), "valid_patches": fit_valid_patches,
            "eligible_ids_sha256": _canonical_sha256(fit_ids), "fit_role_sha256": fit_role_sha256,
        },
        "support_ids_sha256": {
            "fit": _canonical_sha256(fit_ids),
            "calibration": _canonical_sha256(sorted(row.file_id for row in calibration_rows_all)),
            "development": _canonical_sha256(development_ids_flat),
        },
        "batch_geometry": {"batch_size": BATCH_SIZE, "batches_per_epoch": batches_per_epoch, "epochs": EPOCHS,
                           "optimizer_steps_per_seed": EPOCHS * batches_per_epoch},
        "support": support_records,
        "sampler_checks": sampler_records,
        "preflight_wall_s": time.time() - started,
    }
    expected_steps = EPOCHS * batches_per_epoch
    if expected_steps != 4300:
        raise _fail(f"frozen step arithmetic broken: expected 4300, got {expected_steps}")
    _write_json(out_dir / "preflight.json", payload)
    print(json.dumps({"PREFLIGHT_OK": True, "fit_rows": len(fit_ids), "fit_patches": fit_valid_patches,
                      "calibration_rows": len(calibration_rows_all),
                      "steps_per_seed": expected_steps}, sort_keys=True), flush=True)
    return {
        "fit_rows": fit_rows, "fit_ids": fit_ids,
        "fit_sample_by_id": {row.file_id: row.sample for row in fit_rows},
        "calibration_rows": calibration_rows_all,
        "calibration_rows_by_history": {h: [r for r in calibration_rows_all if r.actual_history_id == h] for h in calibration_history_ids},
        "fit_history_ids": fit_history_ids,
        "calibration_history_ids": calibration_history_ids,
        "development_history_ids": development_history_ids,
        "payload": payload,
    }


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
        # Frozen B0 lineage rule (sprint17_task7_b0.py, reused by the T9 pilot
        # driver): the collated batch mask seed is the 1-based global step
        # index. The deterministic per-file/view augmentation streams are
        # driven separately by train_sprint18_epoch ->
        # _forward_terms(sprint18_view_epoch=epoch). No outcome-derived seed.
        # K7-FULL uses the exact Sprint 17 C5-A channel-time block policy.
        batches.append(collate_variable_files(
            samples, patchifier, masking_config=config,
            masking_seed=global_step_base + batch_index + 1,
            masking_policy=masking_policy_for_arm(ARM_ID),
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
        "alpha": None,
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


def run_seed(*, binding, worktree: Path, seed: int, fit_ids, fit_sample_by_id, fit_history_ids,
             calibration_rows, calibration_rows_by_history,
             development_history_ids, device: torch.device,
             out_dir: Path, progress, seed_dir: Path) -> dict:
    from representation.trainer import RepresentationTrainer

    run_id = f"{RUN_ID_PREFIX}-seed{seed}"
    run_start = time.time()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    config = build_k_config(ARM_ID, seed)
    # K7 owns C2-A: build the arm's exact Sprint17 C2-A patchifier together
    # with the graph (never the plain B0 patchifier object) so the trainer's
    # frozen member-graph check and every collation share one lattice.
    model, patchifier = build_k_model(ARM_ID, config)
    criterion = build_k_criterion(ARM_ID, LAMBDA_INFO_MAX)
    _lock_digest = _coefficient_lock(binding)
    arm = make_sprint18_arm_config(
        ARM_ID, seed,
        {"lambda_info_max": LAMBDA_INFO_MAX, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="locked", coefficient_lock_sha256=_lock_digest,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0)
    trainer = RepresentationTrainer(
        model, criterion, optimizer=optimizer, max_grad_norm=1.0, device=device, seed=seed,
    )
    initialization_sha256 = state_sha256(model.state_dict())
    batches_per_epoch = math.ceil(len(fit_ids) / BATCH_SIZE)
    seed_checkpoints: dict = {}
    exact_lambdas: dict = {}
    exact_lrs: dict = {}
    train_start = time.time()
    saved_step300: dict | None = None
    for epoch in range(1, EPOCHS + 1):
        order, batches = _build_epoch_batches(
            binding=binding, arm=arm, epoch=epoch, config=config, patchifier=patchifier,
            fit_ids=fit_ids, fit_sample_by_id=fit_sample_by_id, fit_history_ids=fit_history_ids,
        )
        metrics = trainer.train_sprint18_epoch(batches, arm=arm, epoch=epoch, collect_balance=True)
        expected_step = epoch * batches_per_epoch
        if trainer.step != expected_step:
            raise _fail(f"seed {seed} epoch {epoch}: trainer step {trainer.step} != deterministic position {expected_step}")
        row = trainer.objective_balance_history[-1]
        exact_lambdas[str(epoch)] = row["effective_coefficient_L_info"]
        exact_lrs[str(epoch)] = row["lr"]
        if row["lr"] != learning_rate_for_epoch(epoch):
            raise _fail(f"seed {seed} epoch {epoch}: LR drift")
        if row["effective_coefficient_L_info"] != coefficient_for_epoch(LAMBDA_INFO_MAX, epoch):
            raise _fail(f"seed {seed} epoch {epoch}: coefficient drift")
        _log_progress(progress, {"run": run_id, "model_seed": seed, "epoch": epoch,
                                 "global_step": trainer.step, "lr": row["lr"],
                                 "lambda_info": row["effective_coefficient_L_info"],
                                 "prediction_loss": metrics.get("prediction_loss"),
                                 "contrastive_loss": metrics.get("contrastive_loss"),
                                 "joint_loss": metrics.get("joint_loss"),
                                 "elapsed_s": time.time() - train_start})
        if epoch in DIAGNOSTIC_EPOCHS or epoch == EPOCHS:
            active_epoch = epoch + 1 if epoch < EPOCHS else EPOCHS
            # Strict-checkpoint coherence: the validator requires the saved
            # optimizer LR to equal learning_rate_for_epoch(active_epoch),
            # matching the next-epoch-ready sampler/scheduler/coefficient
            # states below. The trainer pins the LR only at each trained
            # epoch start, so the live optimizer still holds LR(epoch) here;
            # advance it to LR(active_epoch) before the save. The next epoch
            # re-pins the identical value, so the trajectory is unchanged.
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
            seed_checkpoints[str(epoch)] = {"path": str(path), "sha256": digest,
                                            "global_step": trainer.step, "completed_epoch": epoch,
                                            "active_epoch": active_epoch}
        # Frozen step-300 diagnostic: 300 = 7 * 43 + 41 falls inside epoch 7
        # (batches_per_epoch = 43 on full Fit), i.e. a mid-epoch position.
        # The coherent checkpoint schema requires sampler/global-step coherence
        # (global_step == completed_epochs * batches_per_epoch +
        # batches_in_current_epoch), so the mid-epoch state cannot be saved as
        # a coherent checkpoint. Record the boundary truth instead: epoch 7 is
        # complete at step 301, and steps 300/301 are both inside the epoch-7
        # window. No padded/partial checkpoint is emitted.
        if saved_step300 is None and trainer.step >= STEP_300:
            saved_step300 = {"global_step_300_in_epoch": epoch,
                             "epoch7_complete_step": 7 * batches_per_epoch,
                             "reached": True,
                             "checkpoint_policy": "epoch-boundary-only"}
    train_wall_s = time.time() - train_start
    global_step = EPOCHS * batches_per_epoch
    if trainer.step != global_step:
        raise _fail(f"seed {seed}: final step {trainer.step} != frozen {global_step}")
    primary = seed_checkpoints[str(EPOCHS)]
    live_model_sha = state_sha256(model.state_dict())
    live_ema_sha = state_sha256(model.target_encoder.state_dict())
    restored_model, restored_patchifier = build_k_model(ARM_ID, config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=learning_rate_for_epoch(EPOCHS), weight_decay=0.0)
    restored = restore_sprint18_checkpoint(Path(primary["path"]), restored_model, restored_optimizer,
                                           arm=arm, binding=binding, allowed_scope="experiment")
    if restored.checkpoint_sha256 != primary["sha256"]:
        raise _fail(f"seed {seed}: strict restore digest mismatch")
    if state_sha256(restored_model.state_dict()) != live_model_sha:
        raise _fail(f"seed {seed}: strict restore model bytes mismatch")
    if state_sha256(restored_model.target_encoder.state_dict()) != live_ema_sha:
        raise _fail(f"seed {seed}: strict restore EMA bytes mismatch")
    # Restored-device placement: restore_sprint18_checkpoint loads with
    # map_location="cpu" and the model constructor places nothing, so the
    # verified-restored model/EMA graph joins the run device here — after
    # all strict digest/byte checks — for the parity probe and every
    # downstream consumer. Saved digests are device-invariant, the frozen
    # trajectory is untouched (no optimizer step, no RNG draw, no schedule
    # change), and public bank/scoring co-locate each CPU batch onto the
    # deployed model device. Never move the probe batch to CPU instead.
    restored_model.to(device)
    probe_ids = fit_ids[: min(4, len(fit_ids))]
    probe = collate_variable_files([fit_sample_by_id[i] for i in probe_ids], restored_patchifier,
                                   masking_config=config, masking_seed=1,
                                   masking_policy=masking_policy_for_arm(ARM_ID))
    model.eval()
    restored_model.eval()
    with torch.no_grad():
        live_out = model(_move({k: v for k, v in probe.items() if k != "file_samples"}, device))
        restored_out = restored_model(_move({k: v for k, v in probe.items() if k != "file_samples"}, device))
    emb_diff = float((live_out["file_embedding"] - restored_out["file_embedding"]).abs().max().item())
    if not math.isfinite(emb_diff) or emb_diff != 0.0:
        raise _fail(f"seed {seed}: restored forward differs by {emb_diff}")
    fit_rows_by_id = {row.file_id: row for row in _fit_rows_cache}
    fit_rows_ordered = [fit_rows_by_id[i] for i in fit_ids]
    bank = fit_fit_reference_bank(binding, arm, restored_model, fit_rows_ordered, restored_patchifier,
                                  checkpoint=restored, scope="experiment")
    if bank.provenance.source_rows != len(fit_ids) or bank.provenance.scope != "experiment":
        raise _fail(f"seed {seed}: bank provenance mismatch")
    calibration_scores = {}
    for history_id, rows in calibration_rows_by_history.items():
        scores = score_certified_rows(binding, arm, restored_model, bank, rows, restored_patchifier,
                                     checkpoint=restored, action="calibrate_q95", scope="experiment")
        validate_score_maps(scores.support, scores.scores)
        calibration_scores[history_id] = {
            "support_sha256": scores.support.support_sha256,
            "file_ids": list(scores.support.file_ids),
            "S_pred": dict(scores.scores["S_pred"]),
            "S_pop": dict(scores.scores["S_pop"]),
            "branch_provenance_sha256": dict(scores.branch_provenance_sha256),
            "checkpoint_sha256": scores.checkpoint_sha256,
            "bank_sha256": scores.bank_sha256,
        }
    cal_rows_flat = calibration_rows
    cal_scores_obj = score_certified_rows(binding, arm, restored_model, bank, cal_rows_flat, restored_patchifier,
                                          checkpoint=restored, action="calibrate_q95", scope="experiment")
    calibration = fit_calibration_q95(binding, cal_scores_obj, cal_rows_flat, scope="experiment")
    development = _score_development_descriptive(
        binding=binding, arm=arm, model=restored_model, bank=bank, patchifier=restored_patchifier,
        checkpoint=restored, worktree=worktree, development_history_ids=development_history_ids,
        thresholds={"S_pred": calibration.S_pred.threshold, "S_pop": calibration.S_pop.threshold},
    )
    prediction_cache_note = (
        "K7-FULL produces no prediction cache: the declared cache consumers are "
        "Task13B/Task30 on the final B0-FULL representations."
    )
    balance_records = [dict(r) for r in trainer.objective_balance_history]
    history = [dict(r) for r in trainer.history]
    record = {
        "status": "ok", "run_id": run_id, "arm_id": ARM_ID, "model_seed": seed,
        "arm_config_sha256": arm_config_sha256(arm),
        "arm_identity": {
            "members": list(members_for_arm(ARM_ID)),
            "member_numerics": {member: member_numerics(member) for member in members_for_arm(ARM_ID)},
            "criterion": dict(c5a_criterion_identity()),
            "lambda_info_max": LAMBDA_INFO_MAX,
            "masking_policy": MASKING_POLICY_ID,
        },
        "initialization_sha256": initialization_sha256,
        "global_step": global_step, "representation_epochs_completed": EPOCHS,
        "batches_per_epoch": batches_per_epoch,
        "exact_schedule": {"lambda_info_max": LAMBDA_INFO_MAX, "per_epoch_lambda": exact_lambdas, "per_epoch_lr": exact_lrs},
        "checkpoints": seed_checkpoints, "step300": saved_step300,
        "primary": {**primary, "live_model_state_sha256": live_model_sha, "live_ema_state_sha256": live_ema_sha,
                    "restored_checkpoint_sha256": restored.checkpoint_sha256,
                    "restored_forward_max_abs_diff": emb_diff},
        "bank": {"source_rows": bank.provenance.source_rows, "bank_sha256": bank.provenance.bank_sha256,
                 "checkpoint_sha256": bank.provenance.checkpoint_sha256,
                 "source_rows_sha256": bank.provenance.source_rows_sha256},
        "calibration_scores": calibration_scores,
        "calibration": {"S_pred_threshold": calibration.S_pred.threshold, "S_pop_threshold": calibration.S_pop.threshold,
                        "source_rows": calibration.S_pred.source_rows,
                        "source_support_sha256": calibration.S_pred.source_support_sha256},
        "development": development,
        "prediction_cache": {"produced": False, "reason": prediction_cache_note},
        "balance_records": len(balance_records),
        "history_epochs": len(history),
        "train_wall_s": train_wall_s,
        "files_processed": len(fit_ids) * EPOCHS,
        "total_wall_s": time.time() - run_start,
    }
    _write_json(seed_dir / "seed_record.json", record)
    del trainer, optimizer, model, restored_model, restored_optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"record": record, "balance_records": balance_records, "history": history,
            "arm": arm, "config": config}


def _load_development_rows(*, binding, history_id, manifest_path, shard_paths, file_ids) -> tuple:
    """Load every Development row under the frozen manifest/shard contract.

    Development histories carry normal + abnormal rows, so the healthy-only
    Fit/Calibration loaders are inapplicable: `load_certified_sample_subset`
    without `role_action` refuses non-Fit/Calibration histories, and any
    healthy `role_action` refuses non-normal-label rows. This loader therefore
    performs the same frozen manifest/shard verification inline (manifest
    digest vs the certified binding, format/role/protocol/config identity,
    file/shard membership, exact shard coverage, per-shard sha256, member
    identity/label match, channel/length/cardinality bounds) and returns
    identity-verified `CertifiedSample` rows for the full file list. It
    applies NO eligibility/label filtering, NO quarantine/censoring filter,
    and NO healthy predicate: every manifest-bound file (normal and
    abnormal) is returned. Descriptive scoring stays frozen-support: the
    manifest file list is the complete frozen Development support.
    """
    from types import MappingProxyType

    from representation.sprint18_harness import CertifiedSample, validate_role_permission
    from synth.dataset import load_sample_bytes

    validate_role_permission(binding, history_id, "score_development", healthy=False)
    history = binding.history(history_id)
    requested = tuple(file_ids)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(not isinstance(x, str) or not x for x in requested)
    ):
        raise ValueError("requested source file IDs must be a non-empty unique sequence")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != history.manifest_sha256:
        raise ValueError(f"source role manifest digest mismatch for {history_id}")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("source role manifest must be a JSON object")
    if manifest.get("format") != 1 or manifest.get("role") != history_id:
        raise ValueError("source manifest format or role identity differs from the certified handoff")
    if manifest.get("protocol") != "sprint15-benchmark-protocol-v7" or manifest.get("config_hash") != history.config_hash:
        raise ValueError("source manifest protocol/config identity differs from the certified handoff")
    files = manifest.get("files")
    shards = manifest.get("shards")
    if not isinstance(files, list) or not isinstance(shards, list):
        raise ValueError("source manifest is missing file or shard rows")
    file_by_id: dict = {}
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get("file_id"), str):
            raise ValueError("source manifest contains a malformed file row")
        file_id = row["file_id"]
        if file_id in file_by_id:
            raise ValueError("source manifest contains duplicate file IDs")
        file_by_id[file_id] = row
    shard_by_id: dict = {}
    shard_by_path: dict = {}
    for row in shards:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("file_ids"), list):
            raise ValueError("source manifest contains a malformed shard row")
        relative = row["path"]
        if relative in shard_by_path:
            raise ValueError("source manifest contains duplicate shard paths")
        shard_by_path[relative] = row
        for file_id in row["file_ids"]:
            if not isinstance(file_id, str) or file_id in shard_by_id:
                raise ValueError("source shard membership is malformed or duplicated")
            shard_by_id[file_id] = row
    if set(file_by_id) != set(shard_by_id):
        raise ValueError("source manifest file and shard membership differ")
    if any(file_id not in file_by_id for file_id in requested):
        raise ValueError("requested source rows are not bound by the certified role manifest")
    selected_shards = {shard_by_id[file_id]["path"] for file_id in requested}
    if set(shard_paths) != set(shard_by_path):
        raise ValueError("shard path inputs do not exactly cover the manifest verification scope")
    archive_by_shard: dict = {}
    shard_digest_by_path: dict = {}
    try:
        for relative in sorted(shard_by_path):
            shard_record = shard_by_path[relative]
            expected_digest = shard_record.get("sha256")
            if not isinstance(expected_digest, str) or len(expected_digest) != 64:
                raise ValueError(f"source shard digest is malformed: {relative}")
            try:
                int(expected_digest, 16)
            except ValueError:
                raise ValueError(f"source shard digest is malformed: {relative}") from None
            local_path = Path(shard_paths[relative])
            observed_digest = _sha256_file(local_path)
            if observed_digest != expected_digest:
                raise ValueError(f"source shard checksum mismatch: {relative}")
            shard_digest_by_path[relative] = observed_digest
            if relative in selected_shards:
                import zipfile as _zipfile

                archive_by_shard[relative] = _zipfile.ZipFile(local_path)
        rows: list = []
        for file_id in requested:
            shard_record = shard_by_id[file_id]
            relative = shard_record["path"]
            member_name = f"{file_id}.npz"
            archive = archive_by_shard[relative]
            if member_name not in archive.namelist():
                raise ValueError(f"source shard is missing manifest-bound member {member_name}")
            sample = load_sample_bytes(archive.read(member_name))
            sample.validate()
            manifest_row = file_by_id[file_id]
            if sample.file_id != file_id or sample.file_label.value != manifest_row.get("file_label"):
                raise ValueError(f"source member identity/label mismatch for {file_id}")
            if sample.C != 6 or not 200 <= sample.T <= 800:
                raise ValueError(f"source sample violates the certified channel/length contract: {file_id}")
            if not 0 <= sample.robot_idx < 9 or not 0 <= sample.program_idx < 8:
                raise ValueError(f"source conditioning ID is outside the certified model cardinalities: {file_id}")
            rows.append(
                CertifiedSample(
                    sample=sample,
                    actual_history_id=history_id,
                    manifest_sha256=history.manifest_sha256,
                    shard_path=relative,
                    shard_sha256=shard_digest_by_path[relative],
                    manifest_row=MappingProxyType(dict(manifest_row)),
                )
            )
    finally:
        for archive in archive_by_shard.values():
            archive.close()
    return tuple(rows)

def _score_development_descriptive(*, binding, arm, model, bank, patchifier,
                                       checkpoint, worktree, development_history_ids,
                                       thresholds, scope: str = "experiment") -> dict:
    """Descriptive Development scoring on frozen common support (no selection).

    Scores every Development file through the public inference path
    (action="score_development"), then evaluates the frozen probe15
    descriptive metrics per branch at the frozen q95 thresholds. Production
    calls use scope="experiment"; the signature accepts scope only so the
    software-only tiny proof can exercise the identical loader/scoring path
    at "task6_smoke_only" without touching production semantics. No refit,
    pruning, checkpoint reselection, recalibration, or coefficient
    change: primary comparisons remain final-checkpoint-only per protocol.
    """
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
        file_ids = [row["file_id"] for row in manifest["files"]]
        rows = _load_development_rows(
            binding=binding, history_id=history_id, manifest_path=manifest_path,
            shard_paths=shard_paths, file_ids=file_ids,
        )
        scores = score_certified_rows(binding, arm, model, bank, rows, patchifier,
                                     checkpoint=checkpoint, action="score_development",
                                     scope=scope)
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
        for h, history_id in enumerate(development_history_ids):
            entry = canonical["per_history"][h]
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
            "macro": {"P_macro": canonical["macro_auc_p"], "W_macro": canonical["macro_auc_w"],
                      "PW_macro": canonical["macro_auc_pw"], "PW_lcb95": canonical["lcb_pw"],
                      "directional_histories": canonical.get("directional_histories")},
            "per_history": per_history,
        }
    return {"branches": branch_results,
            "support": {"evaluation_histories": list(development_history_ids),
                        "per_history": per_history_supports}}


def _coefficient_lock(binding) -> str:
    import json as _json
    lock_path = REPO_ROOT / "experiments" / "sprint18-task14-config-lock.json"
    return _json.loads(lock_path.read_text(encoding="utf-8"))["coefficient_lock_sha256"]


_fit_rows_cache: list = []


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
    """Return the paired K7/B0 value and signed difference (fail closed on gaps)."""
    for value, source in ((arm_value, "K7"), (b0_value, "B0")):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise _fail(f"paired metric {label} is missing or nonfinite in the {source} record")
    return {"arm": float(arm_value), "b0": float(b0_value), "delta": float(arm_value) - float(b0_value)}


def _paired_development(*, arm_records: dict, b0_records: dict, b0_records_sha256: str) -> dict:
    """Pair K7-FULL Development outputs with the accepted B0-FULL seed records.

    Pairing requires the same model seed and the same frozen Development
    support (per-history support hashes) and Calibration support. Each branch
    keeps its own arm-specific q95 threshold; no threshold is shared, and no
    selection, promotion, pruning, or metric redefinition is applied.
    """
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
            raise _fail(f"K7 seed {key} record is not a completed {ARM_ID} run")
        if b0.get("arm_id") != "B0-FULL" or b0.get("status") != "ok":
            raise _fail(f"B0 seed {key} record is not a completed B0-FULL run")
        for record, label in ((arm, "K7"), (b0, "B0")):
            if (
                record.get("global_step") != expected_steps
                or record.get("representation_epochs_completed") != EPOCHS
                or record.get("batches_per_epoch") != math.ceil(FIT_HEALTHY_ROWS / BATCH_SIZE)
            ):
                raise _fail(f"{label} seed {key} did not complete the frozen 100-epoch schedule")
        arm_dev = arm["development"]
        b0_dev = b0["development"]
        arm_support = arm_dev["support"]
        b0_support = b0_dev["support"]
        if arm_support["evaluation_histories"] != b0_support["evaluation_histories"]:
            raise _fail(f"K7/B0 Development history roster differs for seed {key}")
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
            raise _fail(f"K7/B0 Calibration support differs for seed {key}")
        calibration_support_sha256[key] = arm_cal
        if arm["bank"]["source_rows"] != b0["bank"]["source_rows"]:
            raise _fail(f"K7/B0 Fit bank source rows differ for seed {key}")
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
            "Descriptive paired Development differences only (K7-FULL minus B0-FULL at the same "
            "model seed and frozen support). Each arm uses its own q95 thresholds; no selection, "
            "promotion, pruning, or rescue is applied."
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
        raise _fail("bound coefficient differs from the Task14 lock (0.05)")
    if binding_doc.get("arm_id") != ARM_ID:
        raise _fail("bound arm differs from K7-FULL")
    if binding_doc.get("device") != args.device:
        raise _fail("bound device differs from the requested device")
    head = subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
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
    preflight = run_preflight(binding=binding, binding_doc=binding_doc, worktree=worktree,
                              out_dir=out_dir, environment=environment)
    if args.mode == "preflight":
        _write_json(out_dir / "environment.json",
                    {**environment, "mode": "preflight",
                     "preflight_sha256": _canonical_sha256(preflight["payload"])})
        print(json.dumps({"RUNNER_EXIT": "preflight_ok"}, sort_keys=True), flush=True)
        return 0

    global _fit_rows_cache
    _fit_rows_cache = list(preflight["fit_rows"])
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if tuple(seeds) != tuple(MODEL_SEEDS):
        raise _fail("full runs must use exactly the frozen model seeds")
    progress = (out_dir / "progress.jsonl").open("a", encoding="utf-8")
    seed_records = {}
    try:
        for seed in seeds:
            seed_dir = out_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            result = run_seed(
                binding=binding, worktree=worktree, seed=seed, fit_ids=preflight["fit_ids"],
                fit_sample_by_id=preflight["fit_sample_by_id"],
                fit_history_ids=preflight["fit_history_ids"],
                calibration_rows=preflight["calibration_rows"],
                calibration_rows_by_history=preflight["calibration_rows_by_history"],
                development_history_ids=preflight["development_history_ids"],
                device=device, out_dir=out_dir,
                progress=progress, seed_dir=seed_dir,
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
    print(json.dumps({"RUNNER_EXIT": "ok", "seeds": list(seeds),
                      "paired_schema": paired["schema"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
