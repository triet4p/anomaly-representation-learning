"""Coherent Sprint18 v2 checkpoint persistence and fail-closed restore."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import cast

import numpy as np
import torch

from representation.config import V1Config
from representation.model import V1RepresentationModel
from representation.sprint18_harness import (
    EMA_DECAY,
    BATCH_SIZE,
    MODEL_LR_MAX,
    MODEL_LR_MIN,
    PROTOCOL_ID,
    TASK9_COEFFICIENT_PILOT_EPOCHS,
    Sprint18ArmConfig,
    FIT_HEALTHY_ROWS,
    FIT_VALID_PATCHES,
    Task6Binding,
    _canonical_sha256,
    _is_hash,
    arm_config_sha256,
    deterministic_sampler_order,
    validate_arm_config,
)

CHECKPOINT_SCHEMA_ID = "sprint18-coherent-checkpoint-v2"
SCHEDULER_ID = "sprint18-epoch-lr-v2"


@dataclass(frozen=True, slots=True)
class RestoredSprint18Checkpoint:
    """Validated checkpoint identity returned after a strict restore."""

    checkpoint_sha256: str
    checkpoint_kind: str
    scope: str
    arm_id: str
    model_seed: int
    global_step: int
    completed_epoch: int
    representation_step: int
    optimizer_scope: str
    paired_b0_checkpoint_sha256: str | None
    arm_specific_state: Mapping[str, object]
    data_binding_sha256: str
    arm_config_sha256: str
    model_state_sha256: str
    ema_state_sha256: str
    model_config_sha256: str


def learning_rate_for_epoch(epoch: int) -> float:
    """The exact constant-within-epoch learning rate from the common v2 protocol."""
    if not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= 100:
        raise ValueError("Sprint18 representation epoch must be in [1, 100]")
    if epoch <= 5:
        return MODEL_LR_MAX * (0.1 + 0.9 * (epoch - 1) / 4)
    return MODEL_LR_MIN + 0.5 * (MODEL_LR_MAX - MODEL_LR_MIN) * (
        1.0 + math.cos(math.pi * (epoch - 5) / 95)
    )


def coefficient_for_epoch(maximum: float, epoch: int) -> float:
    """The common 10-zero/20-ramp/70-plateau coefficient schedule."""
    if not isinstance(maximum, (int, float)) or isinstance(maximum, bool) or not math.isfinite(float(maximum)) or maximum < 0:
        raise ValueError("coefficient maximum must be finite and non-negative")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= 100:
        raise ValueError("Sprint18 representation epoch must be in [1, 100]")
    if epoch <= 10:
        return 0.0
    if epoch <= 30:
        return float(maximum) * (epoch - 10) / 20
    return float(maximum)


def capture_rng_state() -> dict[str, object]:
    """Capture Python, NumPy, CPU Torch, and all active CUDA generators."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("checkpoint RNG state is incomplete or contains unknown generators")
    python_state = state["python"]
    numpy_state = state["numpy"]
    torch_cpu = state["torch_cpu"]
    torch_cuda = state["torch_cuda"]
    if not isinstance(torch_cpu, torch.Tensor) or torch_cpu.dtype != torch.uint8 or torch_cpu.ndim != 1:
        raise ValueError("checkpoint CPU Torch RNG state is malformed")
    if not isinstance(torch_cuda, list) or any(not isinstance(item, torch.Tensor) or item.dtype != torch.uint8 for item in torch_cuda):
        raise ValueError("checkpoint CUDA RNG states are malformed")
    try:
        random.setstate(cast(tuple[object, ...], python_state))
        np.random.set_state(cast(tuple[object, ...], numpy_state))
        torch.set_rng_state(torch_cpu.cpu())
        if torch.cuda.is_available():
            if len(torch_cuda) != torch.cuda.device_count():
                raise ValueError("checkpoint CUDA RNG device count differs from the active runtime")
            torch.cuda.set_rng_state_all(torch_cuda)
        elif torch_cuda:
            raise ValueError("checkpoint has CUDA RNG state but restore runtime has no CUDA device")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("checkpoint RNG state cannot be restored coherently") from exc


def _validate_rng_state(state: object) -> Mapping[str, object]:
    if not isinstance(state, Mapping) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("checkpoint RNG state is incomplete or has unknown generators")
    python_state = state["python"]
    if (
        not isinstance(python_state, tuple) or len(python_state) != 3
        or not isinstance(python_state[0], int) or not isinstance(python_state[1], tuple)
        or len(python_state[1]) != 625
        or any(not isinstance(value, int) or isinstance(value, bool) for value in python_state[1])
        or (python_state[2] is not None and (not isinstance(python_state[2], (int, float)) or not math.isfinite(float(python_state[2]))))
    ):
        raise ValueError("checkpoint Python RNG state is malformed")
    numpy_state = state["numpy"]
    if (
        not isinstance(numpy_state, tuple) or len(numpy_state) != 5
        or not isinstance(numpy_state[0], str)
        or not isinstance(numpy_state[1], np.ndarray) or numpy_state[1].dtype != np.uint32
        or numpy_state[1].shape != (624,)
        or not isinstance(numpy_state[2], int) or isinstance(numpy_state[2], bool)
        or not isinstance(numpy_state[3], int) or isinstance(numpy_state[3], bool)
        or not isinstance(numpy_state[4], (int, float)) or not math.isfinite(float(numpy_state[4]))
    ):
        raise ValueError("checkpoint NumPy RNG state is malformed")
    torch_cpu = state["torch_cpu"]
    torch_cuda = state["torch_cuda"]
    if not isinstance(torch_cpu, torch.Tensor) or torch_cpu.dtype != torch.uint8 or torch_cpu.ndim != 1:
        raise ValueError("checkpoint CPU Torch RNG state is malformed")
    if not isinstance(torch_cuda, list) or any(
        not isinstance(item, torch.Tensor) or item.dtype != torch.uint8 or item.ndim != 1 for item in torch_cuda
    ):
        raise ValueError("checkpoint CUDA RNG states are malformed")
    return cast(Mapping[str, object], state)


def _update_digest(digest: object, value: object) -> None:
    """Hash nested checkpoint state with explicit type/shape framing."""
    hasher = cast(object, digest)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        hasher.update(b"tensor\0")  # type: ignore[attr-defined]
        hasher.update(str(tensor.dtype).encode("ascii"))  # type: ignore[attr-defined]
        hasher.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))  # type: ignore[attr-defined]
        hasher.update(tensor.numpy().tobytes())  # type: ignore[attr-defined]
    elif isinstance(value, np.ndarray):
        hasher.update(b"ndarray\0")  # type: ignore[attr-defined]
        hasher.update(str(value.dtype).encode("ascii"))  # type: ignore[attr-defined]
        hasher.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))  # type: ignore[attr-defined]
        hasher.update(np.ascontiguousarray(value).tobytes())  # type: ignore[attr-defined]
    elif isinstance(value, Mapping):
        hasher.update(b"mapping\0")  # type: ignore[attr-defined]
        for key in sorted(value, key=lambda item: str(item)):
            _update_digest(hasher, str(key))
            _update_digest(hasher, value[key])
    elif isinstance(value, (tuple, list)):
        hasher.update(b"sequence\0")  # type: ignore[attr-defined]
        _update_digest(hasher, len(value))
        for item in value:
            _update_digest(hasher, item)
    elif value is None:
        hasher.update(b"none\0")  # type: ignore[attr-defined]
    elif isinstance(value, (str, int, float, bool)):
        hasher.update(type(value).__name__.encode("ascii") + b"\0")  # type: ignore[attr-defined]
        hasher.update(repr(value).encode("utf-8"))  # type: ignore[attr-defined]
    else:
        raise TypeError(f"unsupported checkpoint hash value {type(value).__name__}")


def state_sha256(value: object) -> str:
    """Return a deterministic digest of tensor/mapping/sequence checkpoint state."""
    digest = sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _target_state(model_state: Mapping[str, object]) -> dict[str, object]:
    prefix = "target_encoder."
    target = {name[len(prefix):]: value for name, value in model_state.items() if name.startswith(prefix)}
    if not target:
        raise ValueError("checkpoint model state has no EMA target encoder")
    return target


def _optimizer_step(value: object, label: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{label} is not a scalar optimizer step")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or int(value) != value or value < 0:
        raise ValueError(f"{label} is not a non-negative integer optimizer step")
    return int(value)


def _validate_optimizer_state(
    state: object,
    *,
    global_step: int,
    active_lr: float,
    expected_weight_decay: float,
) -> Mapping[str, object]:
    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer state is missing or has an unknown structure")
    states = state["state"]
    groups = state["param_groups"]
    if not isinstance(states, Mapping) or not isinstance(groups, list) or not groups:
        raise ValueError("checkpoint optimizer state has malformed state/groups")
    if global_step and not states:
        raise ValueError("nonzero global step has an empty optimizer state")
    observed_steps: set[int] = set()
    for parameter_state in states.values():
        if not isinstance(parameter_state, Mapping) or "step" not in parameter_state:
            raise ValueError("optimizer parameter state is missing its step counter")
        observed_steps.add(_optimizer_step(parameter_state["step"], "optimizer parameter step"))
    if observed_steps and observed_steps != {global_step}:
        raise ValueError("optimizer parameter counters do not match the saved global step")
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError("optimizer parameter group is malformed")
        if group.get("lr") != active_lr or group.get("weight_decay") != expected_weight_decay:
            raise ValueError(f"optimizer parameter group {index} does not match the epoch schedule")
    return cast(Mapping[str, object], state)


def _validate_scheduler_state(
    state: object,
    *,
    expected_completed_epochs: int,
    expected_active_epoch: int,
    expected_learning_rate: float,
    expected_scheduler_id: str = SCHEDULER_ID,
) -> Mapping[str, object]:
    expected_keys = {"scheduler_id", "completed_epochs", "active_epoch", "learning_rate", "global_step"}
    if not isinstance(state, Mapping) or set(state) != expected_keys:
        raise ValueError("checkpoint scheduler state is incomplete or has unknown keys")
    if state["scheduler_id"] != expected_scheduler_id:
        raise ValueError("checkpoint scheduler is not the accepted arm-specific epoch schedule")
    for name, value in (("completed_epochs", state["completed_epochs"]), ("active_epoch", state["active_epoch"]), ("global_step", state["global_step"])):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"checkpoint scheduler {name} is invalid")
    rate = state["learning_rate"]
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(float(rate)) or rate <= 0:
        raise ValueError("checkpoint scheduler learning rate is invalid")
    if state["completed_epochs"] != expected_completed_epochs or state["active_epoch"] != expected_active_epoch:
        raise ValueError("checkpoint scheduler epoch does not match the sampler position")
    if rate != expected_learning_rate:
        raise ValueError("checkpoint scheduler learning rate does not match the frozen epoch formula")
    return cast(Mapping[str, object], state)


def _validate_sampler_state(
    state: object,
    *,
    global_step: int,
    scope: str,
    completed_epoch: int,
    arm: Sprint18ArmConfig,
    binding: Task6Binding,
) -> Mapping[str, object]:
    expected_keys = {
        "scope", "workload", "role_history_ids", "role_manifest_sha256", "eligible_source_ids",
        "eligible_source_ids_sha256", "sampler_epoch_seed", "epoch_order_ids",
        "epoch_order_ids_sha256", "eligible_rows", "batch_size", "batches_per_epoch",
        "completed_epochs", "current_epoch", "batches_in_current_epoch", "fit_role_sha256",
    }
    if not isinstance(state, Mapping) or set(state) != expected_keys:
        raise ValueError("checkpoint sampler state is incomplete or has unknown keys")
    if state["scope"] != scope or state["completed_epochs"] != completed_epoch:
        raise ValueError("checkpoint sampler scope/epoch differs from checkpoint metadata")
    history_ids = state["role_history_ids"]
    manifest_digests = state["role_manifest_sha256"]
    source_ids = state["eligible_source_ids"]
    if not isinstance(history_ids, list) or not history_ids or len(set(history_ids)) != len(history_ids):
        raise ValueError("checkpoint sampler history IDs must be non-empty and unique")
    if not isinstance(manifest_digests, list) or not manifest_digests or any(not _is_hash(value) for value in manifest_digests):
        raise ValueError("checkpoint sampler role-manifest digests are invalid")
    if not isinstance(source_ids, list) or not source_ids or any(not isinstance(item, str) or not item for item in source_ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError("checkpoint sampler eligible source IDs must be non-empty and unique")
    if source_ids != sorted(source_ids) or state["eligible_source_ids_sha256"] != _canonical_sha256(source_ids):
        raise ValueError("checkpoint sampler source identity digest/order mismatch")
    if state["fit_role_sha256"] != _canonical_sha256(
        sorted(item.manifest_sha256 for item in binding.role_histories("FIT"))
    ):
        raise ValueError("checkpoint sampler Fit-role digest differs from the certified binding")
    eligible_rows = state["eligible_rows"]
    batch_size = state["batch_size"]
    batches_per_epoch = state["batches_per_epoch"]
    current_epoch = state["current_epoch"]
    batches_in_epoch = state["batches_in_current_epoch"]
    for label, value in (("eligible_rows", eligible_rows), ("batch_size", batch_size), ("batches_per_epoch", batches_per_epoch), ("current_epoch", current_epoch)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"checkpoint sampler {label} is invalid")
    if not isinstance(batches_in_epoch, int) or isinstance(batches_in_epoch, bool) or batches_in_epoch < 0:
        raise ValueError("checkpoint sampler batches_in_current_epoch is invalid")
    expected_histories = [item.actual_history_id for item in binding.role_histories("FIT")]
    workload = state["workload"]
    if scope == "task6_smoke_only":
        if arm.arm_id == "N4-CRN" or workload != "representation":
            raise ValueError("Task6 smoke sampler only supports the bounded representation proof fixture")
        if len(history_ids) != 1 or eligible_rows != len(source_ids) or not 1 <= eligible_rows <= 128:
            raise ValueError("Task6 smoke sampler must bind a small single-history Fit subset")
        if batch_size < 1 or batches_per_epoch != math.ceil(eligible_rows / batch_size):
            raise ValueError("Task6 smoke sampler batch geometry is incoherent")
        maximum_epochs = 100
    elif scope == "experiment":
        if history_ids != expected_histories:
            raise ValueError("experiment sampler must contain all three canonical Fit histories")
        if arm.arm_id == "N4-CRN":
            if workload != "n4_scale" or eligible_rows != FIT_VALID_PATCHES or len(source_ids) != eligible_rows:
                raise ValueError("N4 sampler must bind every eligible Fit patch record")
            if batch_size != 512 or batches_per_epoch != math.ceil(FIT_VALID_PATCHES / 512):
                raise ValueError("N4 sampler differs from the frozen 512-record scale schedule")
            maximum_epochs = 50
        else:
            if workload != "representation" or eligible_rows != binding.fit_healthy_rows or len(source_ids) != eligible_rows:
                raise ValueError("representation sampler must bind the complete Fit-healthy file support")
            if batch_size != BATCH_SIZE or batches_per_epoch != math.ceil(binding.fit_healthy_rows / BATCH_SIZE):
                raise ValueError("representation sampler differs from the frozen file-batch epoch schedule")
            maximum_epochs = 100
    elif scope == "coefficient_pilot":
        if (
            arm.arm_id not in ("B0-FULL", "N1-HCC", "N2-MSP")
            or arm.selection_state != "coefficient_pilot"
            or workload != "representation"
        ):
            raise ValueError("Task9 checkpoint scope is restricted to B0 coefficient pilots")
        if history_ids != expected_histories:
            raise ValueError("Task9 coefficient-pilot checkpoint must bind all three canonical Fit histories")
        if eligible_rows != binding.fit_healthy_rows or len(source_ids) != eligible_rows:
            raise ValueError("Task9 coefficient-pilot checkpoint must bind the complete Fit-healthy support")
        if batch_size != BATCH_SIZE or batches_per_epoch != math.ceil(binding.fit_healthy_rows / BATCH_SIZE):
            raise ValueError("Task9 coefficient-pilot checkpoint differs from the frozen file-batch epoch schedule")
        maximum_epochs = TASK9_COEFFICIENT_PILOT_EPOCHS + 1
    else:
        raise ValueError("checkpoint scope is unsupported")
    if batches_in_epoch > batches_per_epoch:
        raise ValueError("checkpoint sampler position exceeds its epoch")
    if (
        scope == "coefficient_pilot"
        and current_epoch > TASK9_COEFFICIENT_PILOT_EPOCHS
        and batches_in_epoch != 0
    ):
        raise ValueError("Task9 coefficient-pilot checkpoint cannot contain updates beyond epoch 30")
    expected_step = completed_epoch * batches_per_epoch + batches_in_epoch
    if global_step != expected_step:
        raise ValueError("saved global step does not match deterministic sampler position")
    if current_epoch != min(completed_epoch + 1, maximum_epochs):
        raise ValueError("checkpoint current sampler epoch is inconsistent")
    actual_bound_histories = {item.actual_history_id for item in binding.role_histories("FIT")}
    if not set(history_ids).issubset(actual_bound_histories):
        raise ValueError("checkpoint sampler includes a history outside certified Fit")
    expected_digests = {item.actual_history_id: item.manifest_sha256 for item in binding.role_histories("FIT")}
    if manifest_digests != [expected_digests[history_id] for history_id in history_ids]:
        raise ValueError("checkpoint sampler manifest digests do not match its role identities")
    expected_seed, expected_order = deterministic_sampler_order(
        binding, arm, workload=cast(str, workload), scope=cast(str, scope),
        role_history_ids=history_ids, epoch=current_epoch, eligible_source_ids=source_ids,
    )
    if state["sampler_epoch_seed"] != expected_seed or state["epoch_order_ids"] != list(expected_order):
        raise ValueError("checkpoint sampler order/seed differs from the deterministic Fit permutation")
    if state["epoch_order_ids_sha256"] != _canonical_sha256(list(expected_order)):
        raise ValueError("checkpoint sampler epoch-order digest mismatch")
    return cast(Mapping[str, object], state)




def _validate_coefficient_state(
    value: object,
    *,
    arm: Sprint18ArmConfig,
    global_step: int,
    current_epoch: int,
    scope: str,
) -> Mapping[str, object]:
    expected_keys = {"epoch", "global_step", "selection_state", "lambda_info", "lambda_hcc", "alpha", "beta"}
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("checkpoint coefficient schedule state is incomplete or has unknown keys")
    if not isinstance(value["epoch"], int) or isinstance(value["epoch"], bool) or value["epoch"] != current_epoch:
        raise ValueError("checkpoint coefficient epoch does not match the sampler position")
    if not isinstance(value["global_step"], int) or isinstance(value["global_step"], bool) or value["global_step"] != global_step:
        raise ValueError("checkpoint coefficient step does not match global state")
    expected_selection = {
        "experiment": "locked",
        "task6_smoke_only": "task6_smoke_only",
        "coefficient_pilot": "coefficient_pilot",
    }.get(scope)
    if expected_selection is None or value["selection_state"] != expected_selection or arm.selection_state != expected_selection:
        raise ValueError("checkpoint selection state does not match its experiment, smoke, or Task9 pilot scope")
    for name, configured in (
        ("lambda_info", arm.lambda_info_max), ("lambda_hcc", arm.lambda_hcc_max),
        ("alpha", arm.alpha), ("beta", arm.beta),
    ):
        if configured is None:
            expected = None
        elif name in ("lambda_info", "lambda_hcc"):
            expected = coefficient_for_epoch(configured, current_epoch)
        else:
            expected = configured
        actual = value[name]
        if expected is None:
            if actual is not None:
                raise ValueError(f"checkpoint activates undeclared {name} coefficient")
        elif not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isfinite(float(actual)) or actual != expected:
            raise ValueError(f"checkpoint effective {name} coefficient is inconsistent with its epoch/config")

    return cast(Mapping[str, object], value)

def _validate_n4_arm_state(
    state: object,
    *,
    model_state_sha256: str,
    ema_state_sha256: str,
    paired_b0_checkpoint_sha256: str,
    global_step: int,
    ema_updates: int,
    binding: Task6Binding,
) -> Mapping[str, object]:
    expected_keys = {
        "paired_b0_checkpoint_sha256", "paired_b0_representation_step",
        "paired_b0_model_state_sha256", "paired_b0_ema_state_sha256",
        "scale_model_state", "scale_model_state_sha256", "diagnostic_folds",
        "final_fit_record_count", "final_scale_steps", "scale_bounds",
    }
    if not isinstance(state, Mapping) or set(state) != expected_keys:
        raise ValueError("N4 checkpoint must bind its paired B0 state, scale model, folds, and Fit bounds")
    if not _is_hash(state["paired_b0_checkpoint_sha256"]) or state["paired_b0_checkpoint_sha256"] != paired_b0_checkpoint_sha256:
        raise ValueError("N4 checkpoint must bind the same exact paired B0 checkpoint at both scopes")
    paired_step = 100 * math.ceil(FIT_HEALTHY_ROWS / 128)
    if state["paired_b0_representation_step"] != paired_step or ema_updates != paired_step:
        raise ValueError("N4 EMA state/update count must remain bound to the final B0 representation checkpoint")
    if state["paired_b0_model_state_sha256"] != model_state_sha256 or state["paired_b0_ema_state_sha256"] != ema_state_sha256:
        raise ValueError("N4 frozen representation/EMA bytes differ from the paired B0 checkpoint")
    scale_state = state["scale_model_state"]
    if not isinstance(scale_state, Mapping) or not scale_state:
        raise ValueError("N4 scale-model state is missing")
    if state["scale_model_state_sha256"] != state_sha256(scale_state):
        raise ValueError("N4 scale-model state digest mismatch")
    fit_history_ids = [item.actual_history_id for item in binding.role_histories("FIT")]
    folds = state["diagnostic_folds"]
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("N4 must preserve exactly three Fit-history cross-fit records")
    fold_record_counts: list[int] = []
    for index, fold in enumerate(folds):
        expected_fold_keys = {"heldout_history_id", "training_history_ids", "record_count", "optimizer_steps", "scale_state_sha256"}
        if not isinstance(fold, Mapping) or set(fold) != expected_fold_keys:
            raise ValueError(f"N4 diagnostic fold {index} is incomplete")
        heldout = fit_history_ids[index]
        training = [history_id for history_id in fit_history_ids if history_id != heldout]
        if fold["heldout_history_id"] != heldout or fold["training_history_ids"] != training:
            raise ValueError("N4 fold histories do not match the three disjoint canonical Fit folds")
        record_count = fold["record_count"]
        if not isinstance(record_count, int) or isinstance(record_count, bool) or record_count <= 0:
            raise ValueError("N4 fold Fit-record count is invalid")
        expected_steps = 50 * math.ceil(record_count / 512)
        fold_steps = fold["optimizer_steps"]
        if not isinstance(fold_steps, int) or isinstance(fold_steps, bool) or fold_steps != expected_steps or not _is_hash(fold["scale_state_sha256"]):
            raise ValueError("N4 fold optimizer-step or scale-state provenance is invalid")
        fold_record_counts.append(record_count)
    if sum(fold_record_counts) != 2 * FIT_VALID_PATCHES:
        raise ValueError("N4 two-history fold record counts do not cover each Fit record exactly twice")
    final_records = state["final_fit_record_count"]
    final_steps = state["final_scale_steps"]
    expected_final_steps = 50 * math.ceil(FIT_VALID_PATCHES / 512)
    if not isinstance(final_records, int) or isinstance(final_records, bool) or final_records != FIT_VALID_PATCHES:
        raise ValueError("N4 final scale fit must bind every eligible Fit patch record")
    if not isinstance(final_steps, int) or isinstance(final_steps, bool) or final_steps != expected_final_steps or global_step != expected_final_steps:
        raise ValueError("N4 final scale optimizer steps do not match the frozen Fit scope")
    bounds = state["scale_bounds"]
    if not isinstance(bounds, Mapping) or set(bounds) != {
        "q05", "q99", "v_floor", "v_cap", "fit_role_sha256", "residual_records_sha256"
    }:
        raise ValueError("N4 Fit-only residual-bound provenance is incomplete")
    q05 = bounds["q05"]
    q99 = bounds["q99"]
    floor = bounds["v_floor"]
    cap = bounds["v_cap"]
    for name, value in (("q05", q05), ("q99", q99), ("v_floor", floor), ("v_cap", cap)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"N4 {name} residual bound is invalid")
    expected_floor = max(float(q05), 1e-6)
    expected_cap = max(float(q99), 10 * expected_floor)
    if floor != expected_floor or cap != expected_cap:
        raise ValueError("N4 v_floor/v_cap do not match the frozen Fit residual quantile formulas")
    fit_role_digest = _canonical_sha256(sorted(item.manifest_sha256 for item in binding.role_histories("FIT")))
    if bounds["fit_role_sha256"] != fit_role_digest or not _is_hash(bounds["residual_records_sha256"]):
        raise ValueError("N4 residual bounds are not bound to all and only certified Fit histories")
    return cast(Mapping[str, object], state)


def validate_sprint18_checkpoint(
    payload: Mapping[str, object],
    *,
    model_config: V1Config,
    arm: Sprint18ArmConfig,
    binding: Task6Binding,
    allowed_scope: str,
) -> Mapping[str, object]:
    """Reject partial, stale, mismatched, or incoherent model/EMA/optimizer/scheduler/RNG state."""
    expected_keys = {
        "schema_id", "protocol_id", "checkpoint_kind", "scope", "arm_id", "arm_configuration",
        "arm_config_sha256", "model_config", "model_config_sha256", "model_state", "model_state_sha256",
        "optimizer_id", "optimizer_state", "optimizer_state_sha256", "scheduler_state", "scheduler_state_sha256",
        "global_step", "completed_epoch", "representation_step", "optimizer_scope",
        "paired_b0_checkpoint_sha256", "arm_specific_state", "arm_specific_state_sha256",
        "ema_state_sha256", "ema_updates", "ema_decay", "sampler_state", "coefficient_state",
        "rng_state", "rng_state_sha256", "data_binding_sha256", "initialization_sha256",
        "coherent_state_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("checkpoint payload is incomplete or contains unknown top-level state")
    if payload["schema_id"] != CHECKPOINT_SCHEMA_ID or payload["protocol_id"] != PROTOCOL_ID:
        raise ValueError("checkpoint schema/protocol identity is not Sprint18 v2")
    if payload["optimizer_id"] != arm.optimizer:
        raise ValueError("checkpoint optimizer identity differs from the frozen arm contract")
    if payload["scope"] != allowed_scope or allowed_scope not in ("experiment", "task6_smoke_only", "coefficient_pilot"):
        raise ValueError("checkpoint scope is not allowed for this operation")
    if payload["arm_id"] != arm.arm_id or payload["arm_configuration"] != arm.model_dump(mode="json"):
        raise ValueError("checkpoint arm/configuration differs from the requested arm")
    validate_arm_config(arm, require_locked=(allowed_scope == "experiment"))
    if allowed_scope == "task6_smoke_only" and arm.arm_id != "B0-FULL":
        raise ValueError("the only authorized Task6 optimizer-update fixture is B0-FULL")
    if allowed_scope == "coefficient_pilot" and (
        arm.arm_id not in ("B0-FULL", "N1-HCC", "N2-MSP") or arm.selection_state != "coefficient_pilot"
    ):
        raise ValueError("the only authorized Task9 checkpoint is an unselected B0 coefficient pilot")
    arm_sha = arm_config_sha256(arm)
    if payload["arm_config_sha256"] != arm_sha:
        raise ValueError("checkpoint arm configuration digest mismatch")
    saved_config = payload["model_config"]
    if not isinstance(saved_config, Mapping) or dict(saved_config) != model_config.to_dict():
        raise ValueError("checkpoint model config differs from the restored model")
    model_config_sha = _canonical_sha256(dict(saved_config))
    if payload["model_config_sha256"] != model_config_sha:
        raise ValueError("checkpoint model config digest mismatch")
    if payload["data_binding_sha256"] != binding.executable_binding_sha256:
        raise ValueError("checkpoint uses a different certified data/permission binding")
    model_state = payload["model_state"]
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint model state is missing")
    model_state_sha = state_sha256(model_state)
    if payload["model_state_sha256"] != model_state_sha:
        raise ValueError("checkpoint model state digest mismatch")
    ema_sha = state_sha256(_target_state(model_state))
    if payload["ema_state_sha256"] != ema_sha:
        raise ValueError("checkpoint EMA state digest does not match the online payload")
    step = payload["global_step"]
    completed_epoch = payload["completed_epoch"]
    max_epochs = (
        TASK9_COEFFICIENT_PILOT_EPOCHS
        if allowed_scope == "coefficient_pilot"
        else 50 if arm.arm_id == "N4-CRN" else 100
    )
    if not isinstance(step, int) or isinstance(step, bool) or step < 0 or not isinstance(completed_epoch, int) or isinstance(completed_epoch, bool) or not 0 <= completed_epoch <= max_epochs:
        raise ValueError("checkpoint step/epoch counters are invalid")
    representation_step = payload["representation_step"]
    ema_updates = payload["ema_updates"]
    if not isinstance(representation_step, int) or isinstance(representation_step, bool) or representation_step < 0:
        raise ValueError("checkpoint representation step is invalid")
    arm_state = payload["arm_specific_state"]
    if not isinstance(arm_state, Mapping):
        raise ValueError("checkpoint arm-specific state is missing")
    if payload["arm_specific_state_sha256"] != state_sha256(arm_state):
        raise ValueError("checkpoint arm-specific module state digest mismatch")
    if arm.arm_id == "N4-CRN":
        if allowed_scope != "experiment" or payload["optimizer_scope"] != "n4_scale" or representation_step != 0:
            raise ValueError("N4 checkpoint must have zero representation updates and a separate scale optimizer scope")
        paired_sha = payload["paired_b0_checkpoint_sha256"]
        if not _is_hash(paired_sha):
            raise ValueError("N4 checkpoint must bind its paired B0 primary checkpoint")
        _validate_n4_arm_state(
            arm_state, model_state_sha256=model_state_sha, ema_state_sha256=ema_sha,
            paired_b0_checkpoint_sha256=cast(str, paired_sha),
            global_step=step, ema_updates=ema_updates, binding=binding,
        )
    else:
        if payload["optimizer_scope"] != "representation" or representation_step != step:
            raise ValueError("representation optimizer/global step counters are incoherent")
        if payload["paired_b0_checkpoint_sha256"] is not None:
            raise ValueError("only N4 may bind a separate paired B0 checkpoint")
        if not isinstance(ema_updates, int) or isinstance(ema_updates, bool) or ema_updates != representation_step:
            raise ValueError("EMA update count does not match the representation optimizer step")
    if payload["ema_decay"] != EMA_DECAY:
        raise ValueError("checkpoint EMA decay differs from the frozen v2 contract")
    sampler = _validate_sampler_state(
        payload["sampler_state"], global_step=step, scope=allowed_scope,
        completed_epoch=completed_epoch, arm=arm, binding=binding,
    )
    active_epoch = cast(int, sampler["current_epoch"])
    scheduler_id = "sprint18-n4-scale-epoch-v2" if arm.arm_id == "N4-CRN" else SCHEDULER_ID
    active_lr = 1e-3 if arm.arm_id == "N4-CRN" else learning_rate_for_epoch(active_epoch)
    scheduler = _validate_scheduler_state(
        payload["scheduler_state"], expected_completed_epochs=completed_epoch,
        expected_active_epoch=active_epoch, expected_learning_rate=active_lr,
        expected_scheduler_id=scheduler_id,
    )
    if scheduler["global_step"] != step:
        raise ValueError("checkpoint scheduler step differs from the active optimizer/global step")
    optimizer_state = _validate_optimizer_state(
        payload["optimizer_state"], global_step=step, active_lr=active_lr,
        expected_weight_decay=1e-4 if arm.arm_id == "N4-CRN" else 0.0,
    )
    if payload["optimizer_state_sha256"] != state_sha256(optimizer_state):
        raise ValueError("checkpoint optimizer digest mismatch")
    if payload["scheduler_state_sha256"] != _canonical_sha256(dict(scheduler)):
        raise ValueError("checkpoint scheduler digest mismatch")
    coefficient_state = _validate_coefficient_state(
        payload["coefficient_state"], arm=arm, global_step=step,
        current_epoch=active_epoch, scope=allowed_scope,
    )
    rng_state = _validate_rng_state(payload["rng_state"])
    if payload["rng_state_sha256"] != state_sha256(rng_state):
        raise ValueError("checkpoint RNG state digest mismatch")
    initialization_sha = payload["initialization_sha256"]
    if not _is_hash(initialization_sha):
        raise ValueError("checkpoint must bind the actual pre-update initialization digest")
    if arm.arm_id == "N4-CRN":
        if completed_epoch == 50 and step != 50 * math.ceil(FIT_VALID_PATCHES / 512):
            raise ValueError("N4 primary scale checkpoint step differs from the frozen final Fit scope")
    elif arm.representation_epochs == 100 and completed_epoch == 100:
        if step != 100 * math.ceil(binding.fit_healthy_rows / 128):
            raise ValueError("primary representation checkpoint step differs from full Fit support")
    coherent_digest = _canonical_sha256(
        {
            "model_state_sha256": model_state_sha,
            "ema_state_sha256": ema_sha,
            "optimizer_state_sha256": payload["optimizer_state_sha256"],
            "scheduler_state_sha256": payload["scheduler_state_sha256"],
            "optimizer_id": payload["optimizer_id"],
            "rng_state_sha256": payload["rng_state_sha256"],
            "initialization_sha256": initialization_sha,
            "arm_specific_state_sha256": payload["arm_specific_state_sha256"],
            "global_step": step,
            "representation_step": representation_step,
            "ema_updates": ema_updates,
            "completed_epoch": completed_epoch,
            "optimizer_scope": payload["optimizer_scope"],
            "paired_b0_checkpoint_sha256": payload["paired_b0_checkpoint_sha256"],
            "sampler_state": dict(sampler),
            "coefficient_state": dict(coefficient_state),
            "arm_config_sha256": arm_sha,
            "data_binding_sha256": binding.executable_binding_sha256,
        }
    )
    if payload["coherent_state_sha256"] != coherent_digest:
        raise ValueError("checkpoint's joined model/EMA/optimizer/scheduler/RNG state digest mismatch")
    expected_kind = {
        "task6_smoke_only": "task6_smoke_fixture",
        "coefficient_pilot": "coefficient_pilot",
        "experiment": "primary_or_diagnostic",
    }[allowed_scope]
    if payload["checkpoint_kind"] != expected_kind:
        raise ValueError("checkpoint kind is incompatible with its declared execution scope")
    return payload





def save_sprint18_checkpoint(
    path: str | Path,
    model: V1RepresentationModel,
    optimizer: torch.optim.Optimizer,
    *,
    arm: Sprint18ArmConfig,
    binding: Task6Binding,
    global_step: int,
    completed_epoch: int,
    initialization_sha256: str,
    scheduler_state: Mapping[str, object],
    sampler_state: Mapping[str, object],
    coefficient_state: Mapping[str, object],
    scope: str,
    representation_step: int | None = None,
    optimizer_scope: str | None = None,
    ema_updates: int | None = None,
    paired_b0_checkpoint_sha256: str | None = None,
    arm_specific_state: Mapping[str, object] | None = None,
) -> str:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Sprint18 checkpoints require the frozen AdamW optimizer")
    if not _is_hash(initialization_sha256):
        raise ValueError("checkpoint must bind the actual pre-update initialization digest")
    representation_step = global_step if representation_step is None else representation_step
    optimizer_scope = ("n4_scale" if arm.arm_id == "N4-CRN" else "representation") if optimizer_scope is None else optimizer_scope
    ema_updates = representation_step if ema_updates is None else ema_updates
    arm_state = {} if arm_specific_state is None else dict(arm_specific_state)
    model_state = model.state_dict()
    model_state_sha = state_sha256(model_state)
    model_config = model.config.to_dict()
    model_config_sha = _canonical_sha256(model_config)
    optimizer_state = optimizer.state_dict()
    optimizer_sha = state_sha256(optimizer_state)
    scheduler_dict = dict(scheduler_state)
    scheduler_sha = _canonical_sha256(scheduler_dict)
    rng_state = capture_rng_state()
    rng_sha = state_sha256(rng_state)
    arm_sha = arm_config_sha256(arm)
    ema_sha = state_sha256(model.target_encoder.state_dict())
    arm_state_sha = state_sha256(arm_state)
    payload: dict[str, object] = {
        "schema_id": CHECKPOINT_SCHEMA_ID,
        "protocol_id": PROTOCOL_ID,
        "checkpoint_kind": (
            "task6_smoke_fixture" if scope == "task6_smoke_only"
            else "coefficient_pilot" if scope == "coefficient_pilot"
            else "primary_or_diagnostic"
        ),
        "scope": scope,
        "arm_id": arm.arm_id,
        "arm_configuration": arm.model_dump(mode="json"),
        "arm_config_sha256": arm_sha,
        "model_config": model_config,
        "model_config_sha256": model_config_sha,
        "model_state": model_state,
        "optimizer_id": arm.optimizer,
        "model_state_sha256": model_state_sha,
        "optimizer_state": optimizer_state,
        "optimizer_state_sha256": optimizer_sha,
        "scheduler_state": scheduler_dict,
        "scheduler_state_sha256": scheduler_sha,
        "global_step": global_step,
        "completed_epoch": completed_epoch,
        "representation_step": representation_step,
        "optimizer_scope": optimizer_scope,
        "paired_b0_checkpoint_sha256": paired_b0_checkpoint_sha256,
        "arm_specific_state": arm_state,
        "arm_specific_state_sha256": arm_state_sha,
        "ema_state_sha256": ema_sha,
        "ema_updates": ema_updates,
        "ema_decay": model.target_encoder.decay,
        "sampler_state": dict(sampler_state),
        "coefficient_state": dict(coefficient_state),
        "rng_state": rng_state,
        "rng_state_sha256": rng_sha,
        "data_binding_sha256": binding.executable_binding_sha256,
        "initialization_sha256": initialization_sha256,
    }
    target_state = _target_state(cast(Mapping[str, object], model_state))
    if ema_sha != state_sha256(target_state):
        raise ValueError("live model EMA state is not the state serialized with the model")
    payload["coherent_state_sha256"] = _canonical_sha256(
        {
            "model_state_sha256": model_state_sha,
            "ema_state_sha256": ema_sha,
            "optimizer_id": arm.optimizer,
            "optimizer_state_sha256": optimizer_sha,
            "scheduler_state_sha256": scheduler_sha,
            "rng_state_sha256": rng_sha,
            "initialization_sha256": initialization_sha256,
            "arm_specific_state_sha256": arm_state_sha,
            "global_step": global_step,
            "representation_step": representation_step,
            "ema_updates": ema_updates,
            "completed_epoch": completed_epoch,
            "optimizer_scope": optimizer_scope,
            "paired_b0_checkpoint_sha256": paired_b0_checkpoint_sha256,
            "sampler_state": dict(sampler_state),
            "coefficient_state": dict(coefficient_state),
            "arm_config_sha256": arm_sha,
            "data_binding_sha256": binding.executable_binding_sha256,
        }
    )
    validate_sprint18_checkpoint(
        payload, model_config=model.config, arm=arm, binding=binding, allowed_scope=scope
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary = handle.name
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    digest = sha256(destination.read_bytes()).hexdigest()
    if not _is_hash(digest):
        raise ValueError("saved checkpoint digest is invalid")
    return digest


def restore_sprint18_checkpoint(
    path: str | Path,
    model: V1RepresentationModel,
    optimizer: torch.optim.Optimizer | None,
    *,
    arm: Sprint18ArmConfig,
    binding: Task6Binding,
    allowed_scope: str,
    arm_modules: Mapping[str, torch.nn.Module] | None = None,
    restore_rng: bool = True,
) -> RestoredSprint18Checkpoint:
    """Validate every state binding before strict model/optimizer/module/RNG restoration."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Sprint18 checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Sprint18 checkpoint payload is not a mapping")
    validate_sprint18_checkpoint(
        payload, model_config=model.config, arm=arm, binding=binding, allowed_scope=allowed_scope
    )
    modules = {} if arm_modules is None else dict(arm_modules)
    arm_state = cast(Mapping[str, object], payload["arm_specific_state"])
    if arm.arm_id == "N4-CRN":
        if set(modules) != {"scale_model"}:
            raise ValueError("N4 restore requires exactly the paired scale_model module")
        scale_state = cast(Mapping[str, object], arm_state["scale_model_state"])
        modules["scale_model"].load_state_dict(scale_state, strict=True)
    elif modules:
        raise ValueError("non-N4 checkpoint cannot restore arm-specific modules")
    model.load_state_dict(cast(Mapping[str, object], payload["model_state"]), strict=True)
    if optimizer is not None:
        if not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("Sprint18 checkpoint restore requires the frozen AdamW optimizer")
        optimizer.load_state_dict(cast(dict[str, object], payload["optimizer_state"]))
        active_lr = cast(float, cast(Mapping[str, object], payload["scheduler_state"])["learning_rate"])
        _validate_optimizer_state(
            optimizer.state_dict(),
            global_step=cast(int, payload["global_step"]),
            active_lr=active_lr,
            expected_weight_decay=1e-4 if arm.arm_id == "N4-CRN" else 0.0,
        )
    if restore_rng:
        _restore_rng_state(cast(Mapping[str, object], payload["rng_state"]))
    return RestoredSprint18Checkpoint(
        checkpoint_sha256=sha256(checkpoint_path.read_bytes()).hexdigest(),
        checkpoint_kind=cast(str, payload["checkpoint_kind"]),
        scope=cast(str, payload["scope"]),
        arm_id=arm.arm_id,
        model_seed=arm.model_seed,
        global_step=cast(int, payload["global_step"]),
        completed_epoch=cast(int, payload["completed_epoch"]),
        representation_step=cast(int, payload["representation_step"]),
        optimizer_scope=cast(str, payload["optimizer_scope"]),
        paired_b0_checkpoint_sha256=cast(str | None, payload["paired_b0_checkpoint_sha256"]),
        arm_specific_state=arm_state,
        data_binding_sha256=binding.executable_binding_sha256,
        arm_config_sha256=cast(str, payload["arm_config_sha256"]),
        model_state_sha256=cast(str, payload["model_state_sha256"]),
        ema_state_sha256=cast(str, payload["ema_state_sha256"]),
        model_config_sha256=cast(str, payload["model_config_sha256"]),
    )
