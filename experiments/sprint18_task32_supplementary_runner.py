#!/usr/bin/env python
"""S18-T32 supplementary Development metric runner.

``scores`` strictly restores the accepted checkpoint and frozen reference state,
then scores the fixed Fit-healthy, Calibration-healthy, and Development supports.
K9 restores its persisted downstream artifact as-is. Every other arm materializes
the one accepted Fit-bank recipe from the frozen checkpoint and complete qualified
Fit roster; it accepts that byte state only if both the historical bank digest and
source-row digest match exactly. A mismatch aborts before scoring output or bank
artifact publication. No alternate order, recipe, threshold, or fallback exists.

N4 binds its accepted v9 scientific configuration separately from the paired B0
v1 representation configuration, then restores the accepted final scale envelope
strictly and combines public N4 normalized prediction scoring with the unchanged
paired-B0 population bank path. Both lock authorities and the accepted paired-B0
record are immutable inputs.

``metrics`` consumes three validated seed documents (and, for non-B0, the three
same-seed B0 documents), computes the frozen §8.2 families, and emits one
validated supplementary Development metric document.

Development only. No optimizer step, Confirmation path, threshold selection, or
support change. Calibration rescoring verifies the accepted q95 values only.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import sys
from types import MappingProxyType
from typing import cast

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from representation import sprint18_supplementary as S  # noqa: E402
from representation.config import V1Config  # noqa: E402
from representation.crn import (  # noqa: E402
    ContextResidualScaleModel,
    init_n4_scale_model,
    n4_calibration_q95,
    load_n4_scale_checkpoint,
    n4_score_files,
)
from representation.data import collate_variable_files  # noqa: E402
from representation.inference import NormalReferenceBank, RepresentationInference  # noqa: E402
from representation.model import V1RepresentationModel  # noqa: E402
from representation.sprint18_checkpoint import restore_sprint18_checkpoint, state_sha256  # noqa: E402
from representation.sprint18_harness import (  # noqa: E402
    HarnessScores,
    MODEL_SEEDS,
    ProvenancedBank,
    Sprint18ArmConfig,
    _arm_scoring_masking_policy,
    _canonical_sha256,
    _encoder_batch,
    _tensor_digest,
    arm_config_sha256,
    deterministic_mask_seed,
    fit_calibration_q95,
    fit_fit_reference_bank,
    load_certified_development_support,
    load_certified_healthy_role_support,
    load_certified_sample_subset,
    load_task6_binding,
    make_common_support,
    score_certified_rows,
    validate_arm_config,
    validate_role_permission,
    validate_score_maps,
)
from representation.sprint18_k9_downstream import (  # noqa: E402
    K9_ARM_ID,
    ProvenancedK9Downstream,
    _reference_state_digest as _k9_reference_state_digest,
    _standardizer_state_digest as _k9_standardizer_state_digest,
    load_k9_downstream,
    score_k9_certified_rows,
)
from representation.sprint18_k_dispatch import (  # noqa: E402
    _file_seed as k9_file_seed,
    build_k_config,
    build_k_model,
    k9_forward_latents,
    score_k9_files,
)
from synth import events as E  # noqa: E402
from synth import probe15 as P  # noqa: E402
from synth.config import PatchConfig  # noqa: E402
from synth.patchify import Patchifier  # noqa: E402

TASK_ID = "S18-T32"
DEFAULT_ATTEMPT = "S18T32C05"
DEFAULT_AGENT = "S18T32C05"
B0_ARM_ID = "B0-FULL"
K_LEGACY_ARMS = ("K3-FULL", "K7-FULL", "K9-FULL")
N4_TASK14_LOCK_SHA256 = S.N4_SCIENTIFIC_TASK14_LOCK_V9_SHA256
B0_TASK14_LOCK_SHA256 = S.TASK14_LOCK_V1_SHA256


@dataclass(frozen=True, slots=True)
class _N4Reference:
    bank: ProvenancedBank
    scale_model: ContextResidualScaleModel
    bounds: Mapping[str, float]


def _reference_state_sha256(arm_id: str, reference) -> str:
    """Digest the live immutable reference components before and after scoring."""
    if arm_id == K9_ARM_ID:
        if not isinstance(reference, ProvenancedK9Downstream):
            raise _fail("K9 scoring requires its persisted downstream reference")
        state = {
            "standardizer": _k9_standardizer_state_digest(reference.standardizer),
            "C7_reference": _k9_reference_state_digest(reference.reference),
            "window": _canonical_sha256(dict(reference.window)),
            "fit_ids": list(reference.fit_ids),
        }
        return _canonical_sha256(state)
    bank_reference = reference.bank if arm_id == "N4-CRN" else reference
    if not isinstance(bank_reference, ProvenancedBank) or bank_reference.bank.embeddings is None:
        raise _fail("supplementary scoring requires its restored Fit-bank tensor")
    state = {
        "bank_sha256": _tensor_digest(bank_reference.bank.embeddings),
        "bank_k": bank_reference.bank.k,
    }
    if arm_id == "N4-CRN":
        if not isinstance(reference, _N4Reference):
            raise _fail("N4 scoring requires its restored final scale envelope")
        state["scale_state_sha256"] = state_sha256(reference.scale_model.state_dict())
        state["scale_bounds"] = dict(reference.bounds)
    return _canonical_sha256(state)


def _fail(message: str) -> SystemExit:
    return SystemExit(f"S18-T32 supplementary runner: {message}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n")


def _assert_new_path(path: str | Path, label: str) -> None:
    if Path(path).exists():
        raise _fail(f"{label} output already exists; supplementary outputs are append-only: {path}")

def _load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mapping(value: object, label: str = "JSON object") -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail(f"{label} must be a JSON object")
    return value

def _accepted_record_state_digests(
    record: Mapping[str, object], arm_id: str
) -> tuple[str, str, str]:
    """Read the accepted model/EMA/checkpoint digests from the arm record."""
    if arm_id == "N4-CRN":
        state = _mapping(
            record.get("coherent_checkpoint"), "accepted N4 coherent checkpoint"
        )
        fields = (
            state.get("paired_b0_checkpoint_sha256"),
            state.get("restored_model_state_sha256"),
            state.get("restored_ema_state_sha256"),
        )
    else:
        state = _mapping(record.get("primary"), "accepted primary checkpoint")
        fields = (
            state.get("restored_checkpoint_sha256"),
            state.get("live_model_state_sha256"),
            state.get("live_ema_state_sha256"),
        )
    if any(not isinstance(value, str) or len(value) != 64 for value in fields):
        raise _fail("accepted record lacks checkpoint/model/EMA SHA-256 digests")
    return cast(tuple[str, str, str], fields)


def _accepted_record_thresholds(
    record: Mapping[str, object], arm_id: str
) -> dict[str, float]:
    calibration = _mapping(record.get("calibration"), "accepted calibration record")
    if arm_id == "N4-CRN":
        thresholds = _mapping(
            calibration.get("n4_thresholds"), "accepted N4 calibration thresholds"
        )
        raw = {
            "S_pred": thresholds.get("S_pred"),
            "S_pop": thresholds.get("S_pop"),
        }
    else:
        raw = {
            "S_pred": calibration.get("S_pred_threshold"),
            "S_pop": calibration.get("S_pop_threshold"),
        }
    result: dict[str, float] = {}
    for name in S.SUPPLEMENTARY_BRANCHES:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _fail(f"accepted calibration record lacks finite {name} q95 threshold")
        threshold = float(value)
        if not math.isfinite(threshold):
            raise _fail(f"accepted calibration record has non-finite {name} q95 threshold")
        result[name] = threshold
    return result

def _accepted_development_support_digests(
    record: Mapping[str, object],
) -> dict[str, str]:
    development = _mapping(record.get("development"), "accepted Development record")
    support = _mapping(development.get("support"), "accepted Development support")
    histories = support.get("evaluation_histories")
    if (
        not isinstance(histories, list)
        or len(histories) != 4
        or any(not isinstance(history, str) or not history for history in histories)
        or len(set(histories)) != 4
    ):
        raise _fail("accepted Development support must name the four unique histories")
    per_history = support.get("per_history")
    if isinstance(per_history, Mapping):
        if set(per_history) != set(histories):
            raise _fail("accepted Development per-history support roster is incomplete")
        digests = {
            history: _mapping(per_history[history]).get("support_sha256")
            for history in histories
        }
    else:
        per_history_digests = support.get("per_history_support_sha256")
        if not isinstance(per_history_digests, Mapping) or set(per_history_digests) != set(histories):
            raise _fail("accepted Development support lacks its four per-history digests")
        digests = dict(per_history_digests)
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in digests.values()
    ):
        raise _fail("accepted Development support contains an invalid SHA-256 digest")
    return cast(dict[str, str], digests)





# ---------------------------------------------------------------------------
# Frozen identity construction
# ---------------------------------------------------------------------------


def _task14_lock(inputs: S.FrozenSupplementaryInputs) -> Mapping[str, object]:
    raw = Path(inputs.task14_lock_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != inputs.task14_lock_sha256:
        raise _fail("Task14 lock bytes changed after manifest validation")
    if inputs.arm_id == "N4-CRN" and digest != N4_TASK14_LOCK_SHA256:
        raise _fail("N4 scientific Task14 lock bytes differ from the accepted v9 authority")
    return _mapping(json.loads(raw), "Task14 lock")


def _paired_baseline_task14_lock(
    inputs: S.FrozenSupplementaryInputs,
) -> Mapping[str, object]:
    if (
        inputs.arm_id != "N4-CRN"
        or inputs.paired_baseline_task14_lock_path is None
        or inputs.paired_baseline_task14_lock_sha256 is None
    ):
        raise _fail("N4 manifest lacks an explicit paired-B0 Task14 lock authority")
    raw = Path(inputs.paired_baseline_task14_lock_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != inputs.paired_baseline_task14_lock_sha256 or digest != B0_TASK14_LOCK_SHA256:
        raise _fail("paired-B0 Task14 lock bytes differ from the accepted v1 authority")
    return _mapping(json.loads(raw), "paired-B0 Task14 lock")


def _validate_n4_paired_baseline_record(
    n4_record: Mapping[str, object],
    b0_record: Mapping[str, object],
    b0_arm: Sprint18ArmConfig,
    model_seed: int,
) -> str:
    if (
        b0_record.get("status") != "ok"
        or b0_record.get("arm_id") != B0_ARM_ID
        or b0_record.get("model_seed") != model_seed
        or n4_record.get("status") != "ok"
        or n4_record.get("arm_id") != "N4-CRN"
        or n4_record.get("model_seed") != model_seed
    ):
        raise _fail("N4 and paired B0 accepted record identities do not match")
    b0_config_sha = b0_record.get("arm_config_sha256")
    if (
        not isinstance(b0_config_sha, str)
        or b0_config_sha != arm_config_sha256(b0_arm)
    ):
        raise _fail("paired B0 lock config differs from its accepted B0 record")
    b0_state = _accepted_record_state_digests(b0_record, B0_ARM_ID)
    n4_state = _accepted_record_state_digests(n4_record, "N4-CRN")
    b0_primary = _mapping(b0_record.get("primary"), "accepted paired B0 primary")
    n4_baseline = _mapping(
        n4_record.get("representation_baseline"), "accepted N4 paired B0 baseline"
    )
    if (
        b0_state != n4_state
        or b0_primary.get("sha256") != b0_state[0]
        or b0_primary.get("path") != n4_baseline.get("checkpoint_path")
        or n4_baseline.get("checkpoint_sha256") != b0_state[0]
        or n4_baseline.get("live_model_state_sha256") != b0_state[1]
        or n4_baseline.get("live_ema_state_sha256") != b0_state[2]
    ):
        raise _fail("paired B0 checkpoint/model/EMA differs from the accepted N4 representation baseline")
    return b0_config_sha


def _arm_config(lock: Mapping[str, object], arm_id: str, model_seed: int) -> Sprint18ArmConfig:
    locked = lock.get("locked_configs")
    if not isinstance(locked, Mapping):
        raise _fail("Task14 lock is missing locked_configs")
    entry = locked.get(f"{arm_id}/seed-{model_seed}")
    if not isinstance(entry, Mapping):
        raise _fail(f"Task14 lock has no {arm_id}/seed-{model_seed} config")
    config = Sprint18ArmConfig(**dict(entry))
    return validate_arm_config(config, require_locked=True)


def _restore_arm_id(arm_id: str) -> str:
    """N4-CRN shares the frozen B0 representation; its checkpoint is the paired B0."""
    return B0_ARM_ID if arm_id == "N4-CRN" else arm_id


def _build_runtime(arm_id: str, model_seed: int) -> tuple[V1RepresentationModel, Patchifier]:
    if arm_id in K_LEGACY_ARMS:
        config = build_k_config(arm_id, model_seed)
        model, patchifier = build_k_model(arm_id, config)
        return model, patchifier
    config = V1Config(
        n_channels=6,
        patch_size=32,
        stride=16,
        n_robots=9,
        n_programs=8,
        seed=model_seed,
        contrastive_weight_max=0.05,
        ema_decay=0.996,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    return V1RepresentationModel(config, patchifier=patchifier), patchifier


# ---------------------------------------------------------------------------
# Certified support loading
# ---------------------------------------------------------------------------


def _role_assets(
    data_root: str, history
) -> tuple[Path, dict[str, str], Mapping[str, object]]:
    role_dir = Path(data_root) / history.assigned_role / history.actual_history_id
    manifest_path = role_dir / "manifest.json"
    if not manifest_path.is_file():
        raise _fail(f"certified role manifest is missing: {manifest_path}")
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "role manifest")
    shard_paths = {str(row["path"]): str(role_dir / str(row["path"])) for row in manifest["shards"]}
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != history.manifest_sha256:
        raise _fail(f"role manifest digest differs from the certified binding: {manifest_path}")
    return manifest_path, shard_paths, manifest


def _load_fit_rows_from_root(data_root: str, binding) -> tuple:
    rows: list = []
    for history in binding.role_histories("FIT"):
        manifest_path, shard_paths, _ = _role_assets(data_root, history)
        rows.extend(
            load_certified_healthy_role_support(
                binding,
                history.actual_history_id,
                manifest_path,
                shard_paths,
                action="train_fit",
            )
        )
    # The accepted B0/N4 runners pass the full Fit roster in global file_id
    # ascending order. This exact order is part of the frozen bank recipe.
    rows.sort(key=lambda row: row.file_id)
    file_ids = [row.file_id for row in rows]
    if len(file_ids) != len(set(file_ids)) or any(
        left >= right for left, right in zip(file_ids, file_ids[1:])
    ):
        raise _fail("qualified Fit rows do not have unique ascending global file_id order")
    if len(rows) != binding.fit_healthy_rows:
        raise _fail("Fit-healthy support differs from the certified eligible row count")
    return tuple(rows)


def _load_fit_rows(inputs: S.FrozenSupplementaryInputs, binding) -> tuple:
    return _load_fit_rows_from_root(inputs.data_root, binding)


def _fit_rows_source_sha256(rows: Sequence) -> str:
    ordered_rows = sorted(rows, key=lambda row: (row.actual_history_id, row.file_id))
    return _canonical_sha256(
        [
            {
                "history_id": row.actual_history_id,
                "manifest_sha256": row.manifest_sha256,
                "file_id": row.file_id,
                "shard_sha256": row.shard_sha256,
            }
            for row in ordered_rows
        ]
    )


def _load_calibration_rows(inputs: S.FrozenSupplementaryInputs, binding) -> tuple:
    histories = binding.role_histories("CALIBRATION")
    if len(histories) != 1:
        raise _fail("the certified roster must carry exactly one Calibration history")
    history = histories[0]
    manifest_path, shard_paths, manifest = _role_assets(inputs.data_root, history)
    normal_ids = [
        str(row["file_id"]) for row in manifest["files"] if row.get("file_label") == "normal"
    ]
    rows = load_certified_sample_subset(
        binding,
        history.actual_history_id,
        manifest_path,
        shard_paths,
        normal_ids,
        role_action="calibrate_q95",
        verify_all_shards=True,
    )
    if len(rows) != binding.calibration_healthy_rows:
        raise _fail("Calibration-healthy support differs from the certified eligible row count")
    return tuple(rows)


def _development_manifests(inputs: S.FrozenSupplementaryInputs, binding) -> dict[str, Mapping[str, object]]:
    manifests: dict[str, Mapping[str, object]] = {}
    for history in binding.role_histories("DEVELOPMENT"):
        _, _, manifest = _role_assets(inputs.data_root, history)
        manifests[history.actual_history_id] = manifest
    if len(manifests) != 4:
        raise _fail("the certified roster must carry exactly four Development histories")
    return manifests


def _load_development_rows(
    inputs: S.FrozenSupplementaryInputs,
    binding,
    manifests: Mapping[str, Mapping[str, object]],
) -> dict[str, tuple]:
    rows_by_history: dict[str, tuple] = {}
    for history in binding.role_histories("DEVELOPMENT"):
        manifest_path, shard_paths, _ = _role_assets(inputs.data_root, history)
        rows_by_history[history.actual_history_id] = load_certified_development_support(
            binding, history.actual_history_id, manifest_path, shard_paths
        )
    if len(rows_by_history) != len(manifests):
        raise _fail("Development row support differs from the loaded manifests")
    return rows_by_history

def _development_expected_digests(data_root: str, binding) -> tuple[dict[str, str], dict[str, str]]:
    support_digests: dict[str, str] = {}
    manifest_digests: dict[str, str] = {}
    histories = binding.role_histories("DEVELOPMENT")
    for history in histories:
        manifest_path, shard_paths, _manifest = _role_assets(data_root, history)
        rows = load_certified_development_support(
            binding, history.actual_history_id, manifest_path, shard_paths
        )
        common_support = make_common_support(
            binding, history.actual_history_id, [row.file_id for row in rows]
        )
        support_digests[history.actual_history_id] = common_support.support_sha256
        manifest_digests[history.actual_history_id] = history.manifest_sha256
    if len(support_digests) != 4:
        raise _fail("the certified roster must carry exactly four Development histories")
    return support_digests, manifest_digests



# ---------------------------------------------------------------------------
# Scoring paths
# ---------------------------------------------------------------------------


def _n4_harness_scores(
    binding,
    arm: Sprint18ArmConfig,
    underlying_arm: Sprint18ArmConfig,
    model: V1RepresentationModel,
    rows: Sequence,
    patchifier: Patchifier,
    *,
    checkpoint,
    action: str,
    reference: _N4Reference,
    device: torch.device,
) -> tuple[HarnessScores, HarnessScores]:
    if arm.arm_id != "N4-CRN" or underlying_arm.arm_id != B0_ARM_ID:
        raise _fail("N4 scoring requires its strict scale restore and paired B0 checkpoint")
    b0_scores = score_certified_rows(
        binding,
        underlying_arm,
        model,
        reference.bank,
        rows,
        patchifier,
        checkpoint=checkpoint,
        action=action,  # type: ignore[arg-type]
        scope="experiment",
    )
    n4_pred: dict[str, float] = {}
    for row in rows:
        mask_seed = deterministic_mask_seed(arm.model_seed, row.actual_history_id, row.file_id)
        triples = n4_score_files(
            model=model,
            samples=[row.sample],
            scale_model=reference.scale_model,
            v_floor=float(reference.bounds["v_floor"]),
            v_cap=float(reference.bounds["v_cap"]),
            patchifier=patchifier,
            masking_seed=mask_seed,
        )
        if set(triples) != {row.file_id}:
            raise _fail(f"N4 public scorer returned the wrong file map for {row.file_id}")
        score = float(triples[row.file_id]["normalized"])
        if not math.isfinite(score) or score < 0.0:
            raise _fail(f"N4 public scorer returned an invalid normalized score for {row.file_id}")
        n4_pred[row.file_id] = score
    branches = MappingProxyType(
        {
            "S_pred": MappingProxyType(n4_pred),
            "S_pop": MappingProxyType(
                {file_id: float(value) for file_id, value in b0_scores.scores["S_pop"].items()}
            ),
        }
    )
    validate_score_maps(b0_scores.support, branches)
    scale_sha256 = state_sha256(reference.scale_model.state_dict())
    branch_provenance = MappingProxyType(
        {
            "S_pred": _canonical_sha256(
                {
                    "branch": "S_pred",
                    "score_map": dict(branches["S_pred"]),
                    "support_sha256": b0_scores.support.support_sha256,
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "scale_state_sha256": scale_sha256,
                    "role_manifest_sha256": b0_scores.role_manifest_sha256,
                }
            ),
            "S_pop": _canonical_sha256(
                {
                    "branch": "S_pop",
                    "score_map": dict(branches["S_pop"]),
                    "support_sha256": b0_scores.support.support_sha256,
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "bank_sha256": reference.bank.provenance.bank_sha256,
                    "role_manifest_sha256": b0_scores.role_manifest_sha256,
                }
            ),
        }
    )
    if branch_provenance["S_pred"] == branch_provenance["S_pop"]:
        raise _fail("N4 branch provenance digests unexpectedly alias")
    n4_scores = HarnessScores(
        arm_id=arm.arm_id,
        model_seed=arm.model_seed,
        actual_history_id=b0_scores.actual_history_id,
        schema_history_id=b0_scores.schema_history_id,
        role_manifest_sha256=b0_scores.role_manifest_sha256,
        support=b0_scores.support,
        scores=branches,
        checkpoint_sha256=b0_scores.checkpoint_sha256,
        bank_sha256=reference.bank.provenance.bank_sha256,
        data_binding_sha256=b0_scores.data_binding_sha256,
        branch_provenance_sha256=branch_provenance,
        scope="experiment",
    )
    return n4_scores, b0_scores


def _harness_scores(
    binding,
    arm: Sprint18ArmConfig,
    model: V1RepresentationModel,
    rows: Sequence,
    patchifier: Patchifier,
    *,
    checkpoint,
    action: str,
    reference,
    device: torch.device,
    underlying_arm: Sprint18ArmConfig | None = None,
) -> HarnessScores:
    scoring_arm = underlying_arm or arm
    if arm.arm_id == K9_ARM_ID:
        if scoring_arm.arm_id != K9_ARM_ID:
            raise _fail("K9 scoring arm differs from its restored checkpoint")
        return score_k9_certified_rows(
            binding,
            arm,
            model,
            reference,
            rows,
            patchifier,
            checkpoint=checkpoint,
            action=action,  # type: ignore[arg-type]
            scope="experiment",
            device=device,
        )
    if arm.arm_id == "N4-CRN":
        if not isinstance(reference, _N4Reference):
            raise _fail("N4 scoring requires its strictly restored scale reference")
        n4_scores, _b0_scores = _n4_harness_scores(
            binding,
            arm,
            scoring_arm,
            model,
            rows,
            patchifier,
            checkpoint=checkpoint,
            action=action,
            reference=reference,
            device=device,
        )
        return n4_scores
    if scoring_arm.arm_id != arm.arm_id:
        raise _fail("scoring arm differs from the restored checkpoint")
    return score_certified_rows(
        binding,
        arm,
        model,
        reference,
        rows,
        patchifier,
        checkpoint=checkpoint,
        action=action,  # type: ignore[arg-type]
        scope="experiment",
    )


@torch.no_grad()
def _timestep_pass(
    model: V1RepresentationModel,
    patchifier: Patchifier,
    rows: Sequence,
    *,
    arm: Sprint18ArmConfig,
    history_id: str,
    reference,
    device: torch.device,
) -> dict[str, tuple[float, ...]]:
    """Map each arm's frozen S_pred patch evidence onto its timesteps."""
    model.eval()
    result: dict[str, tuple[float, ...]] = {}
    for row in rows:
        if arm.arm_id == K9_ARM_ID:
            latents = k9_forward_latents(
                model, patchifier, model.config, [(row.file_id, row.sample)], device=device
            )
            _, _, details = score_k9_files(
                latents,
                {
                    "standardizer": reference.standardizer,
                    "reference": reference.reference,
                    "window": dict(reference.window),
                },
            )
            patch = _mapping(_mapping(details["patch"])[row.file_id])
            patch_values = np.asarray(patch["energies"], dtype=np.float64)
            starts = np.asarray(latents[0]["starts"])
            valid_len = np.asarray(latents[0]["valid_len"])
            length = int(row.sample.T)
        else:
            mask_seed = deterministic_mask_seed(
                arm.model_seed, history_id, row.file_id
            )
            batch = collate_variable_files(
                [row.sample],
                patchifier,
                masking_config=model.config,
                masking_seed=mask_seed,
                masking_policy=_arm_scoring_masking_policy(arm),
            )
            clean = _encoder_batch(batch)
            moved = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in clean.items()
            }
            output = model(moved)
            predicted = output["predicted_latents"]
            target = output["target_latents"]
            mask = output["prediction_mask"]
            if arm.arm_id == "N4-CRN":
                if not isinstance(reference, _N4Reference):
                    raise _fail("N4 localization requires the strictly restored scale envelope")
                context = output["context_latents"]
                valid = output["patch_valid_mask"]
                width = int(context.shape[-1])
                residual = (
                    torch.nn.functional.layer_norm(target, (width,))
                    - torch.nn.functional.layer_norm(predicted, (width,))
                ).square().mean(dim=-1)
                scales = reference.scale_model(
                    context.detach().reshape(-1, width),
                    v_floor=float(reference.bounds["v_floor"]),
                    v_cap=float(reference.bounds["v_cap"]),
                )["v"].reshape(context.shape[:2])
                patch_scores = residual / (scales + 1e-6)
                eligible = mask & valid
                patch_scores = patch_scores.masked_fill(~eligible, 0.0)
            else:
                patch_scores = torch.nn.functional.mse_loss(
                    predicted, target, reduction="none"
                ).mean(dim=-1)
                patch_scores = patch_scores.masked_fill(~mask, 0.0)
            patch_values = patch_scores.detach().cpu().numpy()[0]
            starts = batch["starts"].detach().cpu().numpy()[0]
            valid_len = batch["valid_len"].detach().cpu().numpy()[0]
            length = int(batch["file_valid_mask"][0].sum().item())
        vector = patchifier.patch_to_timestep_scores(
            patch_values, starts, valid_len, length
        )
        result[row.file_id] = tuple(float(value) for value in np.asarray(vector).reshape(-1))
    return result


@torch.no_grad()
def _fit_healthy_branch_scores(
    arm: Sprint18ArmConfig,
    model: V1RepresentationModel,
    reference,
    rows: Sequence,
    patchifier: Patchifier,
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Score Fit-healthy rows as the frozen stability-ratio denominator.

    No public Sprint 18 scoring entry point accepts the FIT role (its action
    vocabulary is Calibration/Development/Confirmation only), and the protocol
    §8.2 stability ratio explicitly requires the Fit-healthy score distribution.
    This bounded path therefore reuses the identical frozen forward/score math on
    rows already loaded under the FIT ``train_fit`` permission: it never selects,
    fits, or publishes a Fit score as an evaluation result, and it never mutates
    the model or the reference state.
    """
    model.eval()
    policy = _arm_scoring_masking_policy(arm)
    pred: dict[str, float] = {}
    pop: dict[str, float] = {}
    if arm.arm_id == K9_ARM_ID:
        for row in rows:
            latents = k9_forward_latents(
                model, patchifier, model.config, [(row.file_id, row.sample)], device=device
            )
            predicted, population, _ = score_k9_files(
                latents,
                {
                    "standardizer": reference.standardizer,
                    "reference": reference.reference,
                    "window": dict(reference.window),
                },
            )
            pred[row.file_id] = float(predicted[row.file_id])
            pop[row.file_id] = float(population[row.file_id])
        return {"S_pred": pred, "S_pop": pop}
    bank_reference = reference.bank if arm.arm_id == "N4-CRN" else reference
    if not isinstance(bank_reference, ProvenancedBank):
        raise _fail("Fit-healthy scoring requires the digest-verified Fit bank")
    inference = RepresentationInference(
        model, bank_reference.bank, patchifier, masking_config=model.config
    )
    for row in rows:
        mask_seed = deterministic_mask_seed(arm.model_seed, row.actual_history_id, row.file_id)
        batch = collate_variable_files(
            [row.sample],
            patchifier,
            masking_config=model.config,
            masking_seed=mask_seed,
            masking_policy=policy,
        )
        clean = _encoder_batch(batch)
        if arm.arm_id == "N4-CRN":
            moved = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in clean.items()
            }
            scored = inference.score_batch(cast(object, moved))  # type: ignore[arg-type]
            population = scored.get("S_pop")
            if not isinstance(population, torch.Tensor) or population.shape != (1,):
                raise _fail("N4 Fit-healthy S_pop path did not return one finite score")
            pop[row.file_id] = float(population[0].item())
            triples = n4_score_files(
                model=model,
                samples=[row.sample],
                scale_model=reference.scale_model,
                v_floor=float(reference.bounds["v_floor"]),
                v_cap=float(reference.bounds["v_cap"]),
                patchifier=patchifier,
                masking_seed=mask_seed,
            )
            pred[row.file_id] = float(triples[row.file_id]["normalized"])
            continue
        scored = inference.score_batch(cast(object, clean))  # type: ignore[arg-type]
        predicted = scored.get("S_pred")
        population = scored.get("S_pop")
        if (
            not isinstance(predicted, torch.Tensor)
            or not isinstance(population, torch.Tensor)
            or predicted.shape != (1,)
            or population.shape != (1,)
        ):
            raise _fail("Fit-healthy scoring did not return one finite score per branch")
        pred[row.file_id] = float(predicted[0].item())
        pop[row.file_id] = float(population[0].item())
    return {"S_pred": pred, "S_pop": pop}


# ---------------------------------------------------------------------------
# Reference-state preparation
# ---------------------------------------------------------------------------


def _prepare_reference(
    inputs: S.FrozenSupplementaryInputs,
    binding,
    arm: Sprint18ArmConfig,
    restore_arm: Sprint18ArmConfig,
    model: V1RepresentationModel,
    patchifier: Patchifier,
    checkpoint,
    fit_rows: Sequence,
    device: torch.device,
) -> tuple[object, str, str, dict[str, str]]:
    """Restore or materialize the single byte-identical accepted reference state."""
    artifact = inputs.reference_artifact
    kind = artifact["kind"]
    expected = inputs.expected
    expected_bank = str(expected["bank_sha256"])
    expected_rows = int(expected["bank_source_rows"])
    expected_rows_sha256 = str(expected["bank_source_rows_sha256"])
    accepted_record = _mapping(_load_json(inputs.accepted_record_path), "accepted arm record")
    accepted_bank = _mapping(
        accepted_record.get("downstream")
        if arm.arm_id == K9_ARM_ID
        else accepted_record.get("bank"),
        "accepted Fit-bank custody",
    )
    accepted_bank_sha256 = accepted_bank.get(
        "reference_sha256" if arm.arm_id == K9_ARM_ID else "bank_sha256"
    )
    if (
        accepted_bank_sha256 != expected_bank
        or accepted_bank.get("source_rows") != expected_rows
        or accepted_bank.get("source_rows_sha256") != expected_rows_sha256
        or accepted_bank.get("checkpoint_sha256") != checkpoint.checkpoint_sha256
    ):
        raise _fail("input manifest bank expectations differ from the accepted arm custody record")
    if arm.arm_id == K9_ARM_ID and accepted_bank.get("artifact_sha256") != artifact["sha256"]:
        raise _fail("K9 persisted downstream artifact digest differs from the accepted record")
    if expected_rows != binding.fit_healthy_rows or expected_rows != len(fit_rows):
        raise _fail("accepted bank source-row count differs from the complete certified Fit support")
    fit_histories = tuple(
        sorted(history.actual_history_id for history in binding.role_histories("FIT"))
    )
    fit_manifests = tuple(
        sorted(binding.history(history_id).manifest_sha256 for history_id in fit_histories)
    )
    expected_modes = {
        "S_pred": "restored_from_custody",
        "S_pop": "reconstructed_digest_verified",
    }
    if arm.arm_id == K9_ARM_ID:
        if kind != "k9_downstream":
            raise _fail("K9 supplementary scoring requires the persisted downstream artifact")
        downstream = load_k9_downstream(str(artifact["path"]))
        provenance = downstream.provenance
        if (
            provenance.arm_id != arm.arm_id
            or provenance.model_seed != arm.model_seed
            or provenance.source_role != "FIT"
            or provenance.source_history_ids != fit_histories
            or provenance.source_manifest_sha256 != fit_manifests
            or provenance.source_rows != expected_rows
            or provenance.source_rows_sha256 != expected_rows_sha256
            or tuple(downstream.fit_ids) != tuple(row.file_id for row in fit_rows)
            or provenance.checkpoint_sha256 != checkpoint.checkpoint_sha256
            or provenance.data_binding_sha256 != binding.executable_binding_sha256
            or provenance.scope != "experiment"
            or _k9_standardizer_state_digest(downstream.standardizer)
            != provenance.standardizer_sha256
            or _k9_reference_state_digest(downstream.reference) != provenance.reference_sha256
            or _canonical_sha256(dict(downstream.window)) != provenance.window_sha256
        ):
            raise _fail("K9 downstream provenance or qualified Fit membership differs from custody")
        if provenance.reference_sha256 != expected_bank:
            raise _fail("K9 downstream reference digest differs from the accepted bank digest")
        return (
            downstream,
            provenance.reference_sha256,
            "restored_from_custody",
            {name: "restored_from_custody" for name in S.SUPPLEMENTARY_BRANCHES},
        )
    if restore_arm.arm_id != _restore_arm_id(arm.arm_id):
        raise _fail("bank restoration arm differs from the accepted representation checkpoint")
    if kind != ("n4_scale" if arm.arm_id == "N4-CRN" else "none"):
        raise _fail("supplementary reference artifact kind differs from the frozen arm contract")
    restored_bank = fit_fit_reference_bank(
        binding,
        restore_arm,
        model,
        fit_rows,
        patchifier,
        checkpoint=checkpoint,
        scope="experiment",
    )
    provenance = restored_bank.provenance
    if (
        provenance.arm_id != restore_arm.arm_id
        or provenance.model_seed != arm.model_seed
        or provenance.source_role != "FIT"
        or provenance.source_history_ids != fit_histories
        or provenance.source_manifest_sha256 != fit_manifests
        or provenance.source_rows != expected_rows
        or provenance.source_rows_sha256 != expected_rows_sha256
        or provenance.data_binding_sha256 != binding.executable_binding_sha256
        or provenance.checkpoint_sha256 != checkpoint.checkpoint_sha256
        or provenance.scope != "experiment"
    ):
        raise _fail("restored bank provenance differs from accepted checkpoint/Fit custody")
    if provenance.bank_sha256 != expected_bank:
        raise _fail(
            "the single frozen Fit-bank recipe did not reproduce the accepted bank digest; "
            "aborting without alternate order, recipe, or bank artifact"
        )
    if arm.arm_id != "N4-CRN":
        return restored_bank, expected_bank, "reconstructed_digest_verified", expected_modes

    bounds = _mapping(artifact["bounds"], "N4 accepted scale bounds")
    scale_fit = _mapping(accepted_record.get("scale_fit"), "accepted N4 scale fit")
    accepted_bounds = _mapping(scale_fit.get("bounds"), "accepted N4 scale-fit bounds")
    if any(
        float(bounds[name]) != float(accepted_bounds[name])
        for name in ("v_floor", "v_cap")
    ):
        raise _fail("N4 scale bounds differ from the accepted scale-fit record")
    accepted_scale = _mapping(accepted_record.get("scale_checkpoint"), "accepted N4 scale record")
    if accepted_scale.get("training_history_ids") != list(fit_histories):
        raise _fail("N4 final scale history roster differs from accepted Fit custody")
    scale_model = init_n4_scale_model(int(model.config.d_model), arm.model_seed + 999999)
    try:
        scale_metadata = load_n4_scale_checkpoint(
            str(artifact["path"]),
            scale_model,
            expected_fit_scope="final",
            expected_training_history_ids=list(fit_histories),
            expected_bounds={
                "v_floor": float(bounds["v_floor"]),
                "v_cap": float(bounds["v_cap"]),
            },
        )
    except (OSError, ValueError) as exc:
        raise _fail(f"N4 final scale artifact failed strict restoration: {exc}") from exc
    if scale_metadata.get("state_dict_sha256") is None:
        raise _fail("N4 scale restore did not return its validated state digest")
    if (
        accepted_scale.get("fit_scope") != "final"
        or accepted_scale.get("sha256") != artifact["sha256"]
        or accepted_scale.get("state_dict_sha256") != scale_metadata["state_dict_sha256"]
    ):
        raise _fail("restored N4 scale bytes/state differ from the accepted N4 scale record")
    scale_model.to(device)
    scale_model.eval()
    reference = _N4Reference(
        bank=restored_bank,
        scale_model=scale_model,
        bounds=MappingProxyType(
            {
                "v_floor": float(bounds["v_floor"]),
                "v_cap": float(bounds["v_cap"]),
            }
        ),
    )
    return (
        reference,
        expected_bank,
        "reconstructed_digest_verified",
        {
            "S_pred": "restored_from_custody",
            "S_pop": "reconstructed_digest_verified",
        },
    )


def _bank_artifact_directory(score_path: str | Path) -> Path:
    return Path(f"{score_path}.fit-bank")


def _build_bank_artifact_records(
    inputs: S.FrozenSupplementaryInputs,
    reference,
    fit_rows: Sequence,
    checkpoint,
    score_path: str | Path,
    input_digests: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object], Path | None, bytes | None, bytes | None]:
    """Build immutable bank/provenance records; do not publish before score validation."""
    if inputs.arm_id == K9_ARM_ID:
        artifact = inputs.reference_artifact
        path = str(artifact["path"])
        digest = str(artifact["sha256"])
        return (
            {
                "kind": "s18-k9-downstream-state-v1",
                "path": path,
                "sha256": digest,
                "bank_sha256": str(inputs.expected["bank_sha256"]),
            },
            {
                "schema_id": "s18-k9-downstream-state-v1",
                "path": path,
                "sha256": digest,
            },
            None,
            None,
            None,
        )

    bank_reference = reference.bank if inputs.arm_id == "N4-CRN" else reference
    if not isinstance(bank_reference, ProvenancedBank):
        raise _fail("bank artifact publication requires a validated Fit-bank provenance record")
    bank_provenance = bank_reference.provenance
    expected = inputs.expected
    if (
        bank_reference.bank.embeddings is None
        or bank_provenance.bank_sha256 != expected["bank_sha256"]
        or bank_provenance.source_rows != expected["bank_source_rows"]
        or bank_provenance.source_rows_sha256 != expected["bank_source_rows_sha256"]
        or bank_provenance.checkpoint_sha256 != checkpoint.checkpoint_sha256
    ):
        raise _fail("refusing to persist a bank without exact accepted bytes and row provenance")
    source_file_ids = [str(row.file_id) for row in fit_rows]
    if any(left >= right for left, right in zip(source_file_ids, source_file_ids[1:])):
        raise _fail("refusing to persist a Fit bank whose input order is not global file_id ascending")
    scale_binding = None
    if inputs.arm_id == "N4-CRN":
        scale_binding = {
            "path": str(inputs.reference_artifact["path"]),
            "sha256": str(inputs.reference_artifact["sha256"]),
            "state_dict_sha256": state_sha256(reference.scale_model.state_dict()),
            "training_history_ids": list(bank_provenance.source_history_ids),
            "bounds": dict(reference.bounds),
        }
    reference_artifact = inputs.reference_artifact
    provenance_document = {
        "schema_id": S.BANK_PROVENANCE_SCHEMA_ID,
        "bank_recipe_id": S.BANK_RECIPE_ID,
        "bank_order_id": S.BANK_ORDER_ID,
        "requested_arm_id": inputs.arm_id,
        "bank_arm_id": bank_provenance.arm_id,
        "model_seed": bank_provenance.model_seed,
        "source_role": bank_provenance.source_role,
        "source_history_ids": list(bank_provenance.source_history_ids),
        "source_manifest_sha256": list(bank_provenance.source_manifest_sha256),
        "source_rows": bank_provenance.source_rows,
        "source_rows_sha256": bank_provenance.source_rows_sha256,
        "ordered_file_ids_sha256": _canonical_sha256(source_file_ids),
        "data_binding_sha256": bank_provenance.data_binding_sha256,
        "checkpoint_sha256": bank_provenance.checkpoint_sha256,
        "model_state_sha256": checkpoint.model_state_sha256,
        "ema_state_sha256": checkpoint.ema_state_sha256,
        "bank_sha256": bank_provenance.bank_sha256,
        "bank_k": bank_reference.bank.k,
        "tensor_digest_id": "representation.sprint18_harness._tensor_digest-v1",
        "scope": bank_provenance.scope,
        "n4_scale_artifact": scale_binding,
        "source_commit": inputs.source_commit,
        "task_id": inputs.task_id,
        "protocol_sha256": inputs.protocol_sha256,
        "protocol_path": str(S.PROTOCOL_PATH.resolve()),
        "metric_code_path": str(Path(S.__file__).resolve()),
        "supplementary_definition_path": str(S.SUPPLEMENTARY_DEFINITION_PATH.resolve()),
        "metric_code_sha256": inputs.metric_code_sha256,
        "supplementary_definition_sha256": inputs.supplementary_definition_sha256,
        "input_manifest_path": inputs.manifest_path,
        "input_manifest_sha256": inputs.manifest_sha256,
        "task14_lock_path": inputs.task14_lock_path,
        "arm_config_sha256": inputs.expected["arm_config_sha256"],
        "paired_baseline_arm_config_sha256": inputs.expected[
            "paired_baseline_arm_config_sha256"
        ],
        "paired_baseline_task14_lock_path": inputs.paired_baseline_task14_lock_path,
        "paired_baseline_task14_lock_sha256": inputs.paired_baseline_task14_lock_sha256,
        "paired_baseline_accepted_record_path": inputs.paired_baseline_accepted_record_path,
        "paired_baseline_accepted_record_sha256": inputs.paired_baseline_accepted_record_sha256,
        "task14_lock_sha256": inputs.task14_lock_sha256,
        "handoff_path": inputs.handoff_path,
        "handoff_sha256": inputs.handoff_sha256,
        "accepted_record_path": inputs.accepted_record_path,
        "accepted_record_sha256": inputs.accepted_record_sha256,
        "checkpoint_path": inputs.checkpoint_path,
        "reference_artifact_kind": reference_artifact["kind"],
        "reference_artifact_path": reference_artifact["path"],
        "reference_artifact_sha256": reference_artifact["sha256"],
        "data_root": inputs.data_root,
        "input_sha256": dict(input_digests),
    }
    provenance_bytes = (
        json.dumps(
            provenance_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    buffer = io.BytesIO()
    torch.save(
        {
            "schema_id": S.BANK_ARTIFACT_SCHEMA_ID,
            "bank_sha256": bank_provenance.bank_sha256,
            "k": bank_reference.bank.k,
            "embeddings": bank_reference.bank.embeddings,
            "provenance": provenance_document,
        },
        buffer,
    )
    artifact_bytes = buffer.getvalue()
    directory = _bank_artifact_directory(score_path)
    artifact_path = directory / "bank.pt"
    provenance_path = directory / "provenance.json"
    return (
        {
            "kind": S.BANK_ARTIFACT_SCHEMA_ID,
            "path": str(artifact_path.resolve()),
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "bank_sha256": bank_provenance.bank_sha256,
        },
        {
            "schema_id": S.BANK_PROVENANCE_SCHEMA_ID,
            "path": str(provenance_path.resolve()),
            "sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        },
        directory,
        artifact_bytes,
        provenance_bytes,
    )


def _publish_bank_artifact_directory(
    directory: Path | None,
    artifact_bytes: bytes | None,
    provenance_bytes: bytes | None,
) -> None:
    if directory is None or artifact_bytes is None or provenance_bytes is None:
        return
    _assert_new_path(directory, "Fit-bank artifact directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{directory.name}.", dir=str(directory.parent)
        ) as temporary:
            staging = Path(temporary)
            (staging / "bank.pt").write_bytes(artifact_bytes)
            (staging / "provenance.json").write_bytes(provenance_bytes)
            staging.rename(directory)
    except OSError as exc:
        raise _fail(f"could not publish immutable Fit-bank artifact directory: {exc}") from exc
def _calibration_thresholds(
    binding,
    arm: Sprint18ArmConfig,
    scores: HarnessScores,
    rows: Sequence,
) -> dict[str, float]:
    """Recompute the frozen q95 values for verification only (never operative)."""
    calibration = fit_calibration_q95(binding, scores, rows, scope="experiment")
    return {
        "S_pred": float(calibration.S_pred.threshold),
        "S_pop": float(calibration.S_pop.threshold),
    }


def _robot_days(rows: Sequence[Mapping[str, object]], windows) -> float:
    days = {
        (str(row["robot_id"]), int(float(cast(float, row["end_time"])) // 86400.0))
        for row in rows
        if E.eligible_operational_row(dict(row), cast(dict, windows))
    }
    return float(len(days))

def prepare_input_manifest(args: argparse.Namespace) -> dict[str, object]:
    """Derive one closed manifest only from the explicitly named accepted inputs."""
    output_path = Path(args.out).resolve()
    _assert_new_path(output_path, "supplementary input manifest")
    source_commit = str(args.source_commit)
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise _fail("--source-commit must be a lowercase 40-character commit digest")

    arm_id = str(args.arm)
    model_seed = int(args.model_seed)
    if model_seed not in MODEL_SEEDS:
        raise _fail("manifest model seed is outside the frozen Sprint 18 seed namespace")
    lock_path = Path(args.task14_lock).resolve()
    handoff_path = Path(args.handoff).resolve()
    accepted_path = Path(args.accepted_record).resolve()
    data_root = Path(args.data_root).resolve()
    lock_bytes = lock_path.read_bytes()
    lock = _mapping(json.loads(lock_bytes), "Task14 lock")
    accepted_record = _mapping(_load_json(accepted_path), "accepted arm record")
    if (
        accepted_record.get("status") != "ok"
        or accepted_record.get("arm_id") != arm_id
        or accepted_record.get("model_seed") != model_seed
    ):
        raise _fail("accepted record status/arm/seed does not match the manifest request")
    paired_lock_path: Path | None = None
    paired_record_path: Path | None = None
    paired_record: Mapping[str, object] | None = None
    paired_config_sha: str | None = None
    if arm_id == "N4-CRN":
        paired_lock_argument = getattr(args, "paired_baseline_task14_lock", None)
        paired_record_argument = getattr(args, "paired_baseline_record", None)
        if not paired_lock_argument or not paired_record_argument:
            raise _fail(
                "N4 requires explicit --paired-baseline-task14-lock and --paired-baseline-record"
            )
        if hashlib.sha256(lock_bytes).hexdigest() != N4_TASK14_LOCK_SHA256:
            raise _fail("N4 scientific Task14 lock must be the accepted v9 authority")
        paired_lock_path = Path(paired_lock_argument).resolve()
        paired_record_path = Path(paired_record_argument).resolve()
        paired_lock_bytes = paired_lock_path.read_bytes()
        if hashlib.sha256(paired_lock_bytes).hexdigest() != B0_TASK14_LOCK_SHA256:
            raise _fail("paired-B0 Task14 lock must be the accepted v1 authority")
        paired_lock = _mapping(json.loads(paired_lock_bytes), "paired-B0 Task14 lock")
        paired_record = _mapping(_load_json(paired_record_path), "accepted paired B0 record")
        arm = _arm_config(lock, arm_id, model_seed)
        restore_arm = _arm_config(paired_lock, B0_ARM_ID, model_seed)
        paired_config_sha = _validate_n4_paired_baseline_record(
            accepted_record, paired_record, restore_arm, model_seed
        )
    else:
        if getattr(args, "paired_baseline_task14_lock", None) or getattr(
            args, "paired_baseline_record", None
        ):
            raise _fail("paired-baseline authorities are only valid for N4-CRN")
        arm = _arm_config(lock, arm_id, model_seed)
        restore_arm = arm
    accepted_arm_sha = accepted_record.get("arm_config_sha256")
    if accepted_arm_sha != arm_config_sha256(arm):
        raise _fail("accepted record arm config differs from its explicit Task14 lock")

    record_checkpoint_sha, model_state_sha, ema_state_sha = (
        _accepted_record_state_digests(accepted_record, arm_id)
    )
    if arm_id == "N4-CRN":
        paired = _mapping(
            accepted_record.get("representation_baseline"),
            "accepted N4 paired B0 representation baseline",
        )
        record_checkpoint_path = paired.get("checkpoint_path")
        if (
            paired.get("checkpoint_sha256") != record_checkpoint_sha
            or paired.get("live_model_state_sha256") != model_state_sha
            or paired.get("live_ema_state_sha256") != ema_state_sha
        ):
            raise _fail("N4 paired B0 checkpoint/model/EMA digests differ across accepted record fields")
    else:
        primary = _mapping(accepted_record.get("primary"), "accepted primary checkpoint")
        record_checkpoint_path = primary.get("path")
        if primary.get("sha256") != record_checkpoint_sha:
            raise _fail("accepted primary checkpoint digest fields disagree")
    if not isinstance(record_checkpoint_path, str) or not record_checkpoint_path:
        raise _fail("accepted record does not carry an exact checkpoint path")
    checkpoint_path = (
        Path(args.checkpoint_path).resolve()
        if args.checkpoint_path
        else Path(record_checkpoint_path).resolve()
    )
    if _sha256_file(checkpoint_path) != record_checkpoint_sha:
        raise _fail("selected checkpoint bytes do not match the accepted checkpoint digest")

    if arm_id == K9_ARM_ID:
        bank_record = _mapping(accepted_record.get("downstream"), "accepted K9 downstream record")
        expected_bank_sha = bank_record.get("reference_sha256")
        reference_kind = "k9_downstream"
        record_reference_path = bank_record.get("artifact_path")
        reference_sha = bank_record.get("artifact_sha256")
        reference_fit_scope = None
        reference_bounds = None
        recipe_id, order_id = S.K9_BANK_RECIPE_ID, S.K9_BANK_ORDER_ID
    else:
        bank_record = _mapping(accepted_record.get("bank"), "accepted Fit-bank record")
        expected_bank_sha = bank_record.get("bank_sha256")
        recipe_id, order_id = S.BANK_RECIPE_ID, S.BANK_ORDER_ID
        if arm_id == "N4-CRN":
            reference_kind = "n4_scale"
            scale = _mapping(accepted_record.get("scale_checkpoint"), "accepted N4 scale checkpoint")
            scale_fit = _mapping(accepted_record.get("scale_fit"), "accepted N4 scale fit")
            reference_bounds = _mapping(scale_fit.get("bounds"), "accepted N4 scale-fit bounds")
            record_reference_path = scale.get("path")
            reference_sha = scale.get("sha256")
            reference_fit_scope = scale.get("fit_scope")
            if reference_fit_scope != "final":
                raise _fail("N4 accepted scale checkpoint is not the final Fit scope")
            reference_bounds = {
                name: reference_bounds[name] for name in ("v_floor", "v_cap")
            }
        else:
            reference_kind = "none"
            record_reference_path = None
            reference_sha = None
            reference_fit_scope = None
            reference_bounds = None
    source_rows = bank_record.get("source_rows")
    source_rows_sha = bank_record.get("source_rows_sha256")
    if bank_record.get("checkpoint_sha256") != record_checkpoint_sha:
        raise _fail("accepted Fit-bank and representation checkpoint digests disagree")
    if bank_record.get("source_role") not in (None, "FIT"):
        raise _fail("accepted bank provenance is not sourced exclusively from Fit")
    if (
        not isinstance(expected_bank_sha, str)
        or len(expected_bank_sha) != 64
        or isinstance(source_rows, bool)
        or not isinstance(source_rows, int)
        or source_rows < 1
        or not isinstance(source_rows_sha, str)
        or len(source_rows_sha) != 64
    ):
        raise _fail("accepted record lacks the complete historical Fit-bank identity")

    reference_path: Path | None = None
    if reference_kind == "none":
        if args.reference_artifact_path:
            raise _fail("a non-K9/N4 arm cannot name a persisted reference artifact")
    else:
        if not isinstance(record_reference_path, str) or not record_reference_path:
            raise _fail("accepted record does not carry an exact reference-artifact path")
        reference_path = (
            Path(args.reference_artifact_path).resolve()
            if args.reference_artifact_path
            else Path(record_reference_path).resolve()
        )
        if not isinstance(reference_sha, str) or _sha256_file(reference_path) != reference_sha:
            raise _fail("selected reference artifact bytes differ from accepted custody")

    binding = load_task6_binding(str(handoff_path))
    if not data_root.is_dir():
        raise _fail(f"certified DATA root is not a directory: {data_root}")
    accepted_thresholds = _accepted_record_thresholds(accepted_record, arm_id)

    fit_rows = _load_fit_rows_from_root(str(data_root), binding)
    fit_history_ids = tuple(
        sorted({history.actual_history_id for history in binding.role_histories("FIT")})
    )
    fit_source_sha = _fit_rows_source_sha256(fit_rows)
    if len(fit_rows) != source_rows or fit_source_sha != source_rows_sha:
        raise _fail("qualified Fit membership/digests differ from the accepted bank record")
    accepted_history_ids = bank_record.get("source_history_ids")
    if accepted_history_ids is not None and (
        not isinstance(accepted_history_ids, list)
        or tuple(accepted_history_ids) != fit_history_ids
    ):
        raise _fail("accepted Fit history roster/order differs from the certified binding")
    fit_file_ids_sha = _canonical_sha256([row.file_id for row in fit_rows])
    if arm_id == "N4-CRN":
        accepted_scale = _mapping(
            accepted_record.get("scale_checkpoint"), "accepted N4 scale checkpoint"
        )
        scale_fit = _mapping(accepted_record.get("scale_fit"), "accepted N4 scale fit")
        if (
            accepted_scale.get("training_history_ids") != list(fit_history_ids)
            or scale_fit.get("training_history_ids") != list(fit_history_ids)
        ):
            raise _fail("N4 final scale history roster differs from the accepted Fit binding")
        n4_calibration = _mapping(
            accepted_record.get("calibration"), "accepted N4 calibration"
        )
        b0_parity = _mapping(
            n4_calibration.get("b0_path_parity"), "accepted N4 B0 calibration parity"
        )
        if float(b0_parity["S_pop_threshold"]) != accepted_thresholds["S_pop"]:
            raise _fail("accepted N4 S_pop threshold differs from paired B0 custody")

    development_support, development_manifests = _development_expected_digests(
        str(data_root), binding
    )
    accepted_development_support = _accepted_development_support_digests(accepted_record)
    if development_support != accepted_development_support:
        raise _fail("certified Development support differs from the accepted arm record")
    manifest = {
        "schema_id": S.SUPPLEMENTARY_INPUT_SCHEMA_ID,
        "protocol_id": "sprint18-full-training-protocol-v2",
        "protocol_sha256": S.PROTOCOL_SHA256,
        "supplementary_definition_sha256": _sha256_file(S.SUPPLEMENTARY_DEFINITION_PATH),
        "metric_code_sha256": S.metric_code_sha256(),
        "source_commit": source_commit,
        "task_id": TASK_ID,
        "phase": "development",
        "scope": "experiment",
        "data_root": str(data_root),
        "arm_id": arm_id,
        "model_seed": model_seed,
        "task14_lock": {
            "path": str(lock_path),
            "sha256": _sha256_file(lock_path),
        },
        "paired_baseline": (
            {
                "arm_id": B0_ARM_ID,
                "accepted_record": {
                    "path": str(paired_record_path),
                    "sha256": _sha256_file(paired_record_path),
                },
                "task14_lock": {
                    "path": str(paired_lock_path),
                    "sha256": _sha256_file(paired_lock_path),
                },
                "arm_config_sha256": paired_config_sha,
            }
            if paired_record_path is not None and paired_lock_path is not None
            else None
        ),
        "handoff": {
            "path": str(handoff_path),
            "sha256": _sha256_file(handoff_path),
        },
        "accepted_record": {
            "path": str(accepted_path),
            "sha256": _sha256_file(accepted_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": record_checkpoint_sha,
        },
        "reference_artifact": {
            "kind": reference_kind,
            "path": str(reference_path) if reference_path is not None else None,
            "sha256": reference_sha,
            "fit_scope": reference_fit_scope,
            "bounds": reference_bounds,
        },
        "expected": {
            "arm_config_sha256": str(accepted_arm_sha),
            "checkpoint_arm_config_sha256": arm_config_sha256(restore_arm),
            "data_binding_sha256": binding.executable_binding_sha256,
            "model_state_sha256": model_state_sha,
            "ema_state_sha256": ema_state_sha,
            "bank_sha256": expected_bank_sha,
            "bank_source_rows": source_rows,
            "bank_source_rows_sha256": source_rows_sha,
            "bank_recipe_id": recipe_id,
            "bank_order_id": order_id,
            "calibration_thresholds": accepted_thresholds,
            "development_support_sha256": development_support,
            "paired_baseline_arm_config_sha256": paired_config_sha,
            "development_manifest_sha256": development_manifests,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.TemporaryDirectory(prefix=".s18-t32-manifest.", dir=str(output_path.parent)) as temporary:
        temporary_path = Path(temporary) / "input-manifest.json"
        temporary_path.write_text(serialized, encoding="utf-8")
        validated = S.load_frozen_input_manifest(temporary_path)
        input_digests = S.compute_input_digests(
            validated, binding=binding, definition_path=S.SUPPLEMENTARY_DEFINITION_PATH
        )
    _write_json(output_path, manifest)
    return {
        "mode": "prepare-manifest",
        "arm_id": arm_id,
        "model_seed": model_seed,
        "manifest_path": str(output_path),
        "manifest_sha256": _sha256_file(output_path),
        "checkpoint_sha256": record_checkpoint_sha,
        "bank_sha256": expected_bank_sha,
        "bank_source_rows": source_rows,
        "bank_source_rows_sha256": source_rows_sha,
        "qualified_fit_history_ids": list(fit_history_ids),
        "qualified_fit_rows": len(fit_rows),
        "qualified_fit_source_sha256": fit_source_sha,
        "global_file_id_order_sha256": fit_file_ids_sha,
        "calibration_thresholds": accepted_thresholds,
        "input_sha256": input_digests,
    }


# ---------------------------------------------------------------------------
# scores mode
# ---------------------------------------------------------------------------


def run_scores(args: argparse.Namespace) -> dict[str, object]:
    _assert_new_path(args.out, "supplementary score document")
    inputs = S.load_frozen_input_manifest(args.manifest)
    if inputs.arm_id != K9_ARM_ID:
        _assert_new_path(_bank_artifact_directory(args.out), "Fit-bank artifact directory")
    if _sha256_file(args.definition) != inputs.supplementary_definition_sha256:
        raise _fail("prospective supplement bytes differ from the frozen manifest digest")
    binding = load_task6_binding(inputs.handoff_path)
    digests_before = S.compute_input_digests(
        inputs, binding=binding, definition_path=args.definition
    )
    if binding.executable_binding_sha256 != inputs.expected["data_binding_sha256"]:
        raise _fail("certified DATA binding differs from the frozen manifest expectation")
    lock = _task14_lock(inputs)
    if inputs.arm_id == "N4-CRN":
        paired_lock = _paired_baseline_task14_lock(inputs)
        restore_arm = _arm_config(paired_lock, B0_ARM_ID, inputs.model_seed)
        arm = _arm_config(lock, inputs.arm_id, inputs.model_seed)
    else:
        restore_arm = _arm_config(lock, _restore_arm_id(inputs.arm_id), inputs.model_seed)
        arm = restore_arm
    expected = inputs.expected
    accepted_record = _mapping(_load_json(inputs.accepted_record_path), "accepted arm record")
    if (
        accepted_record.get("arm_id") != inputs.arm_id
        or accepted_record.get("model_seed") != inputs.model_seed
        or accepted_record.get("arm_config_sha256") != expected["arm_config_sha256"]
    ):
        raise _fail("input manifest arm/seed/config differs from the accepted arm record")
    if inputs.arm_id == "N4-CRN":
        if inputs.paired_baseline_accepted_record_path is None:
            raise _fail("N4 manifest lacks its accepted paired B0 record")
        paired_record = _mapping(
            _load_json(inputs.paired_baseline_accepted_record_path),
            "accepted paired B0 record",
        )
        paired_config_sha = _validate_n4_paired_baseline_record(
            accepted_record, paired_record, restore_arm, inputs.model_seed
        )
        if paired_config_sha != expected["paired_baseline_arm_config_sha256"]:
            raise _fail("N4 paired B0 record config differs from its frozen manifest digest")
    record_checkpoint_sha, record_model_sha, record_ema_sha = (
        _accepted_record_state_digests(accepted_record, inputs.arm_id)
    )
    if (
        record_checkpoint_sha != inputs.checkpoint_sha256
        or record_model_sha != expected["model_state_sha256"]
        or record_ema_sha != expected["ema_state_sha256"]
    ):
        raise _fail("input manifest checkpoint/model/EMA digests differ from accepted arm custody")
    accepted_thresholds = _mapping(expected["calibration_thresholds"], "accepted q95 thresholds")
    record_thresholds = _accepted_record_thresholds(accepted_record, inputs.arm_id)
    if dict(accepted_thresholds) != record_thresholds:
        raise _fail("input manifest q95 thresholds differ from the accepted arm record")
    if _accepted_development_support_digests(accepted_record) != dict(
        _mapping(expected["development_support_sha256"])
    ):
        raise _fail("input manifest Development support differs from accepted arm custody")
    if arm_config_sha256(arm) != expected["arm_config_sha256"]:
        raise _fail("Task14 arm config differs from the accepted supplementary arm digest")
    if arm_config_sha256(restore_arm) != expected["checkpoint_arm_config_sha256"]:
        raise _fail("Task14 checkpoint arm config differs from the paired accepted checkpoint digest")
    model, patchifier = _build_runtime(_restore_arm_id(inputs.arm_id), inputs.model_seed)
    device = torch.device(args.device)
    model.to(device)
    checkpoint = restore_sprint18_checkpoint(
        inputs.checkpoint_path,
        model,
        None,
        arm=restore_arm,
        binding=binding,
        allowed_scope="experiment",
        arm_modules=None,
        restore_rng=False,
    )
    if checkpoint.checkpoint_sha256 != inputs.checkpoint_sha256:
        raise _fail("restored checkpoint digest differs from the frozen manifest")
    if checkpoint.model_state_sha256 != expected["model_state_sha256"]:
        raise _fail("restored model state differs from the accepted model-state digest")
    if checkpoint.ema_state_sha256 != expected["ema_state_sha256"]:
        raise _fail("restored EMA state differs from the accepted EMA-state digest")
    if checkpoint.arm_config_sha256 != expected["checkpoint_arm_config_sha256"]:
        raise _fail("restored checkpoint arm config differs from its accepted digest")
    initial_model_state = state_sha256(model.state_dict())
    initial_ema_state = state_sha256(model.target_encoder.state_dict())

    fit_rows = _load_fit_rows(inputs, binding)
    calibration_rows = _load_calibration_rows(inputs, binding)
    manifests = _development_manifests(inputs, binding)
    development_rows = _load_development_rows(inputs, binding, manifests)

    reference, bank_sha256, bank_source, reference_state_modes = _prepare_reference(
        inputs,
        binding,
        arm,
        restore_arm,
        model,
        patchifier,
        checkpoint,
        fit_rows,
        device,
    )
    initial_reference_state = _reference_state_sha256(inputs.arm_id, reference)

    if arm.arm_id == "N4-CRN":
        if not isinstance(reference, _N4Reference):
            raise _fail("N4 calibration requires its strictly restored scale envelope")
        calibration_scores, b0_calibration_scores = _n4_harness_scores(
            binding,
            arm,
            restore_arm,
            model,
            calibration_rows,
            patchifier,
            checkpoint=checkpoint,
            action="calibrate_q95",
            reference=reference,
            device=device,
        )
        b0_thresholds = _calibration_thresholds(
            binding, restore_arm, b0_calibration_scores, calibration_rows
        )
        recomputed_thresholds = {
            "S_pred": n4_calibration_q95(calibration_scores.scores["S_pred"]),
            "S_pop": b0_thresholds["S_pop"],
        }
    else:
        calibration_scores = _harness_scores(
            binding,
            arm,
            model,
            calibration_rows,
            patchifier,
            checkpoint=checkpoint,
            action="calibrate_q95",
            reference=reference,
            device=device,
            underlying_arm=restore_arm,
        )
        recomputed_thresholds = _calibration_thresholds(
            binding, arm, calibration_scores, calibration_rows
        )
    if not all(
        math.isclose(recomputed_thresholds[name], accepted_thresholds[name], rel_tol=0.0, abs_tol=1e-12)
        for name in S.SUPPLEMENTARY_BRANCHES
    ):
        raise _fail("Calibration re-scoring does not reproduce the accepted q95 thresholds")
    thresholds = accepted_thresholds

    fit_scores = _fit_healthy_branch_scores(
        arm, model, reference, fit_rows, patchifier, device=device
    )

    development_support: list[dict[str, object]] = []
    for history in binding.role_histories("DEVELOPMENT"):
        history_id = history.actual_history_id
        rows = development_rows[history_id]
        manifest = manifests[history_id]
        scored = _harness_scores(
            binding,
            arm,
            model,
            rows,
            patchifier,
            checkpoint=checkpoint,
            action="score_development",
            reference=reference,
            device=device,
            underlying_arm=restore_arm,
        )
        validate_score_maps(scored.support, scored.scores)
        timesteps = _timestep_pass(
            model,
            patchifier,
            rows,
            arm=arm,
            history_id=history_id,
            reference=reference,
            device=device,
        )
        diagnostics = S.localization_diagnostics(timesteps)
        timesteps_digest = _canonical_sha256(
            {
                "history_id": history_id,
                "vectors": {
                    file_id: np.asarray(values, dtype=np.float64).tobytes().hex()
                    for file_id, values in sorted(timesteps.items())
                },
            }
        )
        expected_manifest = str(inputs.expected["development_manifest_sha256"][history_id])
        expected_support = str(inputs.expected["development_support_sha256"][history_id])
        if history.manifest_sha256 != expected_manifest:
            raise _fail(f"Development manifest digest differs for {history_id}")
        if scored.support.support_sha256 != expected_support:
            raise _fail(f"Development support digest differs for {history_id}")
        events = S.evaluable_support(
            [dict(row) for row in manifest["files"]],
            [dict(failure) for failure in manifest.get("failure_events", [])],
            manifest["maintenance_windows"],
        )
        development_support.append(
            {
                "history_id": history_id,
                "schema_history_id": history.schema_history_id,
                "manifest_sha256": history.manifest_sha256,
                "support_sha256": scored.support.support_sha256,
                "rows": len(rows),
                "excluded_rows": len(manifest["files"]) - len(rows),
                "scores": {
                    name: {
                        file_id: float(value) for file_id, value in scored.scores[name].items()
                    }
                    for name in S.SUPPLEMENTARY_BRANCHES
                },
                "localization": diagnostics,
                "timesteps_sha256": timesteps_digest,
                "p": len([event for event in events.events if event.cohort == "P"]),
                "w": len([event for event in events.events if event.cohort == "W"]),
                "a": len([event for event in events.events if event.cohort == "A"]),
                "controls": len(events.controls),
                "robot_days": _robot_days(manifest["files"], manifest["maintenance_windows"]),
            }
        )

    if state_sha256(model.state_dict()) != initial_model_state:
        raise _fail("representation model state changed during supplementary scoring")
    if state_sha256(model.target_encoder.state_dict()) != initial_ema_state:
        raise _fail("EMA state changed during supplementary scoring")
    if state_sha256(model.state_dict()) != checkpoint.model_state_sha256:
        raise _fail("representation state does not equal the restored checkpoint state")
    if _reference_state_sha256(inputs.arm_id, reference) != initial_reference_state:
        raise _fail("restored Fit-bank/downstream reference state changed during scoring")

    digests_after = S.compute_input_digests(
        inputs, binding=binding, definition_path=args.definition
    )
    S.assert_inputs_unchanged(digests_before, digests_after)
    bank_reference = reference.bank if inputs.arm_id == "N4-CRN" else reference
    bank_provenance = (
        reference.provenance
        if inputs.arm_id == K9_ARM_ID
        else bank_reference.provenance
    )
    (
        bank_artifact_record,
        bank_provenance_record,
        bank_output_directory,
        bank_artifact_bytes,
        bank_provenance_bytes,
    ) = _build_bank_artifact_records(
        inputs, reference, fit_rows, checkpoint, args.out, digests_before
    )


    document = {
        "schema_id": S.SUPPLEMENTARY_SCORES_SCHEMA_ID,
        "schema_version": 3,
        "protocol_id": "sprint18-full-training-protocol-v2",
        "protocol_sha256": inputs.protocol_sha256,
        "phase": "development",
        "arm_id": inputs.arm_id,
        "model_seed": inputs.model_seed,
        "execution": {
            "task_id": TASK_ID,
            "attempt_id": args.attempt_id,
            "agent_id": args.agent_id,
            "source_commit": inputs.source_commit,
            "metric_code_sha256": S.metric_code_sha256(),
            "protocol_path": str(S.PROTOCOL_PATH.resolve()),
            "metric_code_path": str(Path(S.__file__).resolve()),
            "supplementary_definition_path": str(Path(args.definition).resolve()),
            "supplementary_definition_sha256": inputs.supplementary_definition_sha256,
            "scope": "experiment",
            "phase": "development",
            "device": str(device),
            "started_utc": args.started_utc,
            "finished_utc": _utc_now(),
        },
        "inputs": {
            "manifest_path": inputs.manifest_path,
            "manifest_sha256": inputs.manifest_sha256,
            "data_root": inputs.data_root,
            "checkpoint_path": inputs.checkpoint_path,
            "checkpoint_sha256": inputs.checkpoint_sha256,
            "reference_artifact_path": inputs.reference_artifact["path"],
            "reference_artifact_sha256": inputs.reference_artifact["sha256"],
            "accepted_record_path": inputs.accepted_record_path,
            "accepted_record_sha256": inputs.accepted_record_sha256,
            "task14_lock_path": inputs.task14_lock_path,
            "task14_lock_sha256": inputs.task14_lock_sha256,
            "paired_baseline_task14_lock_path": inputs.paired_baseline_task14_lock_path,
            "paired_baseline_task14_lock_sha256": inputs.paired_baseline_task14_lock_sha256,
            "paired_baseline_accepted_record_path": inputs.paired_baseline_accepted_record_path,
            "paired_baseline_accepted_record_sha256": inputs.paired_baseline_accepted_record_sha256,
            "paired_baseline_arm_config_sha256": inputs.expected[
                "paired_baseline_arm_config_sha256"
            ],
            "handoff_path": inputs.handoff_path,
            "handoff_sha256": inputs.handoff_sha256,
            "seed_score_documents": [],
            "baseline_score_documents": [],
            "sha256_before": dict(digests_before),
            "sha256_after": dict(digests_after),
            "unchanged": True,
        },
        "reproduction": {
            "arm_config_sha256": arm_config_sha256(arm),
            "paired_baseline_arm_config_sha256": inputs.expected[
                "paired_baseline_arm_config_sha256"
            ],
            "data_binding_sha256": binding.executable_binding_sha256,
            "model_state_sha256": checkpoint.model_state_sha256,
            "ema_state_sha256": checkpoint.ema_state_sha256,
            "bank_sha256": bank_sha256,
            "accepted_bank_sha256": str(inputs.expected["bank_sha256"]),
            "bank_reproduced": True,
            "bank_source": bank_source,
            "bank_source_rows": bank_provenance.source_rows,
            "bank_source_rows_sha256": bank_provenance.source_rows_sha256,
            "bank_recipe_id": (
                "s18-k9-downstream-state-v1"
                if inputs.arm_id == K9_ARM_ID
                else S.BANK_RECIPE_ID
            ),
            "bank_order_id": (
                "persisted-downstream-order-v1"
                if inputs.arm_id == K9_ARM_ID
                else S.BANK_ORDER_ID
            ),
            "bank_artifact": bank_artifact_record,
            "bank_provenance": bank_provenance_record,
            "reference_state_modes": reference_state_modes,
            "calibration_thresholds": accepted_thresholds,
            "accepted_calibration_thresholds": accepted_thresholds,
            "recomputed_calibration_thresholds": recomputed_thresholds,
            "calibration_reproduced": True,
            "representation_state_unchanged": True,
            "reference_state_unchanged": True,
            "encoder_updates": 0,
        },
        "support": {
            "fit_healthy_rows": len(fit_rows),
            "fit_healthy_rows_sha256": bank_provenance.source_rows_sha256,
            "fit_healthy_scores": fit_scores,
            "fit_healthy_scores_sha256": {
                name: S.score_map_sha256(fit_scores[name]) for name in S.SUPPLEMENTARY_BRANCHES
            },
            "calibration_healthy_rows": len(calibration_rows),
            "calibration_support_sha256": calibration_scores.support.support_sha256,
            "calibration_scores": {
                name: {
                    file_id: float(value)
                    for file_id, value in calibration_scores.scores[name].items()
                }
                for name in S.SUPPLEMENTARY_BRANCHES
            },
            "calibration_scores_sha256": {
                name: S.score_map_sha256(calibration_scores.scores[name])
                for name in S.SUPPLEMENTARY_BRANCHES
            },
            "thresholds": thresholds,
            "development": development_support,
            "excluded_rows": sum(int(entry["excluded_rows"]) for entry in development_support),
        },
        "status": "VALID",
        "failure_reasons": [],
    }
    S.validate_supplementary_scores(document)
    _publish_bank_artifact_directory(
        bank_output_directory, bank_artifact_bytes, bank_provenance_bytes
    )
    _write_json(args.out, document)
    return {
        "mode": "scores",
        "arm_id": inputs.arm_id,
        "model_seed": inputs.model_seed,
        "out": str(args.out),
        "out_sha256": _sha256_file(args.out),
        "thresholds": thresholds,
        "bank_sha256": bank_sha256,
        "fit_healthy_rows": len(fit_rows),
    }


# ---------------------------------------------------------------------------
# metrics mode
# ---------------------------------------------------------------------------


def _verify_score_manifest_custody(
    document: Mapping[str, object],
    *,
    expected_data_root: str,
    expected_data_digests: Mapping[str, str],
    definition_path: str | Path,
) -> None:
    inputs = _mapping(document["inputs"])
    manifest = S.load_frozen_input_manifest(str(inputs["manifest_path"]))
    expected = manifest.expected
    reproduction = _mapping(document["reproduction"])
    support = _mapping(document["support"])
    execution = _mapping(document["execution"])
    if (
        manifest.arm_id != document["arm_id"]
        or manifest.model_seed != document["model_seed"]
        or manifest.manifest_sha256 != inputs["manifest_sha256"]
        or manifest.data_root != expected_data_root
        or inputs["data_root"] != manifest.data_root
        or inputs["checkpoint_path"] != manifest.checkpoint_path
        or inputs["checkpoint_sha256"] != manifest.checkpoint_sha256
        or inputs["accepted_record_path"] != manifest.accepted_record_path
        or inputs["accepted_record_sha256"] != manifest.accepted_record_sha256
        or inputs["task14_lock_path"] != manifest.task14_lock_path
        or inputs["task14_lock_sha256"] != manifest.task14_lock_sha256
        or inputs["paired_baseline_task14_lock_path"]
        != manifest.paired_baseline_task14_lock_path
        or inputs["paired_baseline_task14_lock_sha256"]
        != manifest.paired_baseline_task14_lock_sha256
        or inputs["paired_baseline_accepted_record_path"]
        != manifest.paired_baseline_accepted_record_path
        or inputs["paired_baseline_accepted_record_sha256"]
        != manifest.paired_baseline_accepted_record_sha256
        or inputs["paired_baseline_arm_config_sha256"]
        != expected["paired_baseline_arm_config_sha256"]
        or inputs["handoff_sha256"] != manifest.handoff_sha256
        or inputs["reference_artifact_path"] != manifest.reference_artifact["path"]
        or inputs["reference_artifact_sha256"] != manifest.reference_artifact["sha256"]
        or execution["source_commit"] != manifest.source_commit
        or execution["metric_code_sha256"] != manifest.metric_code_sha256
        or execution["supplementary_definition_sha256"]
        != manifest.supplementary_definition_sha256
        or reproduction["arm_config_sha256"] != expected["arm_config_sha256"]
        or reproduction["paired_baseline_arm_config_sha256"]
        != expected["paired_baseline_arm_config_sha256"]
        or reproduction["data_binding_sha256"] != expected["data_binding_sha256"]
        or reproduction["model_state_sha256"] != expected["model_state_sha256"]
        or reproduction["ema_state_sha256"] != expected["ema_state_sha256"]
        or reproduction["bank_sha256"] != expected["bank_sha256"]
        or reproduction["accepted_bank_sha256"] != expected["bank_sha256"]
        or reproduction["bank_source_rows"] != expected["bank_source_rows"]
        or reproduction["bank_source_rows_sha256"] != expected["bank_source_rows_sha256"]
        or reproduction["bank_recipe_id"] != expected["bank_recipe_id"]
        or reproduction["bank_order_id"] != expected["bank_order_id"]
        or dict(_mapping(reproduction["calibration_thresholds"]))
        != dict(_mapping(expected["calibration_thresholds"]))
        or support["fit_healthy_rows"] != expected["bank_source_rows"]
        or support["fit_healthy_rows_sha256"] != expected["bank_source_rows_sha256"]
    ):
        raise _fail("score document does not match its exact frozen input-manifest custody")
    accepted = _mapping(_load_json(manifest.accepted_record_path), "accepted arm record")
    primary_lock = _task14_lock(manifest)
    arm = _arm_config(primary_lock, manifest.arm_id, manifest.model_seed)
    if arm_config_sha256(arm) != expected["arm_config_sha256"]:
        raise _fail("score document scientific config differs from its bound Task14 authority")
    if manifest.arm_id == "N4-CRN":
        paired_lock = _paired_baseline_task14_lock(manifest)
        restore_arm = _arm_config(paired_lock, B0_ARM_ID, manifest.model_seed)
        if manifest.paired_baseline_accepted_record_path is None:
            raise _fail("N4 score manifest lacks its accepted paired B0 record")
        paired_record = _mapping(
            _load_json(manifest.paired_baseline_accepted_record_path),
            "accepted paired B0 record",
        )
        paired_config_sha = _validate_n4_paired_baseline_record(
            accepted, paired_record, restore_arm, manifest.model_seed
        )
        if (
            paired_config_sha != expected["paired_baseline_arm_config_sha256"]
            or arm_config_sha256(restore_arm) != expected["checkpoint_arm_config_sha256"]
        ):
            raise _fail("score document paired-B0 config differs from its bound v1 authority")
    elif arm_config_sha256(arm) != expected["checkpoint_arm_config_sha256"]:
        raise _fail("score document checkpoint config differs from its Task14 authority")
    accepted_bank = _mapping(
        accepted.get("downstream")
        if manifest.arm_id == K9_ARM_ID
        else accepted.get("bank"),
        "accepted Fit-bank custody",
    )
    accepted_bank_sha = accepted_bank.get(
        "reference_sha256" if manifest.arm_id == K9_ARM_ID else "bank_sha256"
    )
    accepted_checkpoint_sha, accepted_model_sha, accepted_ema_sha = (
        _accepted_record_state_digests(accepted, manifest.arm_id)
    )
    accepted_thresholds = _accepted_record_thresholds(accepted, manifest.arm_id)
    accepted_development_support = _accepted_development_support_digests(accepted)
    if (
        accepted.get("status") != "ok"
        or accepted.get("arm_id") != manifest.arm_id
        or accepted.get("model_seed") != manifest.model_seed
        or accepted.get("arm_config_sha256") != expected["arm_config_sha256"]
        or accepted_checkpoint_sha != manifest.checkpoint_sha256
        or accepted_model_sha != expected["model_state_sha256"]
        or accepted_ema_sha != expected["ema_state_sha256"]
        or accepted_thresholds != dict(_mapping(expected["calibration_thresholds"]))
        or accepted_development_support
        != dict(_mapping(expected["development_support_sha256"]))
        or accepted_bank_sha != expected["bank_sha256"]
        or accepted_bank.get("source_rows") != expected["bank_source_rows"]
        or accepted_bank.get("source_rows_sha256") != expected["bank_source_rows_sha256"]
        or accepted_bank.get("checkpoint_sha256") != manifest.checkpoint_sha256
    ):
        raise _fail("score document manifest differs from its accepted arm/bank record")
    if manifest.arm_id == K9_ARM_ID and accepted_bank.get("artifact_sha256") != manifest.reference_artifact["sha256"]:
        raise _fail("K9 score document does not point to its accepted downstream artifact")
    if manifest.arm_id == "N4-CRN":
        accepted_scale = _mapping(accepted.get("scale_checkpoint"), "accepted N4 scale record")
        provenance_ref = _mapping(reproduction["bank_provenance"])
        bank_provenance = _mapping(
            _load_json(str(provenance_ref["path"])), "N4 bank provenance"
        )
        scale_binding = _mapping(
            bank_provenance["n4_scale_artifact"], "N4 scale provenance"
        )
        if (
            accepted_scale.get("fit_scope") != "final"
            or accepted_scale.get("sha256") != manifest.reference_artifact["sha256"]
            or accepted_scale.get("state_dict_sha256")
            != scale_binding.get("state_dict_sha256")
        ):
            raise _fail("N4 score document does not point to its accepted final scale state")

    direct_digests = S.compute_input_digests(
        manifest, definition_path=definition_path
    )
    recorded_digests = _mapping(inputs["sha256_before"])
    if any(recorded_digests.get(key) != value for key, value in direct_digests.items()):
        raise _fail("score document direct custody digest differs from its manifest inputs")
    data_keys = lambda key: key == "data_binding" or key.startswith("data:")
    expected_data = {key: value for key, value in expected_data_digests.items() if data_keys(key)}
    recorded_data = {key: value for key, value in recorded_digests.items() if data_keys(key)}
    if recorded_data != expected_data:
        raise _fail("score document certified DATA digest set differs from the paired metrics manifest")
    development = cast(list[object], support["development"])
    for entry_value in development:
        entry = _mapping(entry_value)
        history_id = str(entry["history_id"])
        if (
            entry["manifest_sha256"] != expected["development_manifest_sha256"][history_id]
            or entry["support_sha256"] != expected["development_support_sha256"][history_id]
        ):
            raise _fail(f"score document Development support differs from accepted custody: {history_id}")


def _load_score_documents(
    paths: Sequence[str],
    *,
    expected_data_root: str,
    expected_data_digests: Mapping[str, str],
    definition_path: str | Path,
) -> list[Mapping[str, object]]:
    documents = []
    for path in paths:
        document = _mapping(_load_json(path), f"score document {path}")
        S.validate_supplementary_scores(document)
        _verify_score_manifest_custody(
            document,
            expected_data_root=expected_data_root,
            expected_data_digests=expected_data_digests,
            definition_path=definition_path,
        )
        _verify_score_bank_artifacts(document, expected_data_digests)
        documents.append(document)
    return documents


def _score_asset_digests(
    documents: Sequence[Mapping[str, object]],
    paths: Sequence[str],
    label: str,
) -> dict[str, str]:
    if len(documents) != len(paths):
        raise _fail(f"{label} score document count differs from the supplied paths")
    digests: dict[str, str] = {}
    for index, (document, path) in enumerate(zip(documents, paths)):
        prefix = f"score:{label}:{index}"
        digests[f"{prefix}:document"] = _sha256_file(path)
        reproduction = _mapping(document["reproduction"])
        for name in ("bank_artifact", "bank_provenance"):
            pointer = _mapping(reproduction[name])
            pointer_path = Path(str(pointer["path"]))
            pointer_sha = _sha256_file(pointer_path)
            if pointer_sha != pointer["sha256"]:
                raise _fail(f"{label} score {name} changed during metrics: {pointer_path}")
            digests[f"{prefix}:{name}"] = pointer_sha
    return digests

def _verify_score_bank_artifacts(
    document: Mapping[str, object],
    expected_data_digests: Mapping[str, str],
) -> None:
    reproduction = _mapping(document["reproduction"])
    inputs = _mapping(document["inputs"])
    artifact = _mapping(reproduction["bank_artifact"])
    provenance_ref = _mapping(reproduction["bank_provenance"])
    artifact_path = Path(str(artifact["path"]))
    provenance_path = Path(str(provenance_ref["path"]))
    if not artifact_path.is_file() or _sha256_file(artifact_path) != artifact["sha256"]:
        raise _fail(f"score document Fit-bank artifact is missing or changed: {artifact_path}")
    if not provenance_path.is_file() or _sha256_file(provenance_path) != provenance_ref["sha256"]:
        raise _fail(f"score document Fit-bank provenance is missing or changed: {provenance_path}")
    fit_manifest_entries = [
        (key.split(":")[2], value)
        for key, value in expected_data_digests.items()
        if len(key.split(":")) == 4
        and key.split(":")[0] == "data"
        and key.split(":")[1] == "FIT"
        and key.split(":")[3] == "manifest"
    ]
    if len(fit_manifest_entries) != 3:
        raise _fail("metrics DATA binding does not carry the three accepted Fit histories")
    fit_history_ids = sorted(history_id for history_id, _digest in fit_manifest_entries)
    fit_manifest_sha256 = sorted(digest for _history_id, digest in fit_manifest_entries)
    data_binding_sha256 = expected_data_digests.get("data_binding")
    if not isinstance(data_binding_sha256, str):
        raise _fail("metrics DATA binding digest is absent")
    if document["arm_id"] == K9_ARM_ID:
        if (
            artifact_path != provenance_path
            or artifact["sha256"] != provenance_ref["sha256"]
            or artifact_path != Path(str(inputs["reference_artifact_path"]))
            or artifact["sha256"] != inputs["reference_artifact_sha256"]
        ):
            raise _fail("K9 bank/provenance must remain in the accepted downstream artifact")
        downstream = load_k9_downstream(str(artifact_path))
        k9_provenance = downstream.provenance
        if (
            k9_provenance.arm_id != document["arm_id"]
            or k9_provenance.model_seed != document["model_seed"]
            or k9_provenance.source_role != "FIT"
            or k9_provenance.source_history_ids != tuple(fit_history_ids)
            or k9_provenance.source_manifest_sha256 != tuple(fit_manifest_sha256)
            or k9_provenance.source_rows != reproduction["bank_source_rows"]
            or k9_provenance.source_rows_sha256 != reproduction["bank_source_rows_sha256"]
            or len(downstream.fit_ids) != reproduction["bank_source_rows"]
            or tuple(downstream.fit_ids) != tuple(sorted(downstream.fit_ids))
            or k9_provenance.checkpoint_sha256 != inputs["checkpoint_sha256"]
            or k9_provenance.data_binding_sha256 != data_binding_sha256
            or k9_provenance.reference_sha256 != reproduction["bank_sha256"]
            or _k9_standardizer_state_digest(downstream.standardizer)
            != k9_provenance.standardizer_sha256
            or _k9_reference_state_digest(downstream.reference)
            != k9_provenance.reference_sha256
            or _canonical_sha256(dict(downstream.window)) != k9_provenance.window_sha256
            or k9_provenance.scope != "experiment"
        ):
            raise _fail("K9 persisted bank provenance differs from its score document")
        return

    if artifact_path.parent != provenance_path.parent:
        raise _fail("regenerated Fit-bank bytes and immutable provenance must publish together")
    try:
        bank_payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - normalize malformed persisted state to refusal
        raise _fail(f"persisted Fit-bank bytes are not a safe bank artifact: {exc}") from exc
    bank_record = _mapping(bank_payload, "Fit-bank artifact")
    if set(bank_record) != {"schema_id", "bank_sha256", "k", "embeddings", "provenance"}:
        raise _fail("persisted Fit-bank artifact has an unknown or incomplete schema")
    embeddings = bank_record["embeddings"]
    if (
        bank_record["schema_id"] != S.BANK_ARTIFACT_SCHEMA_ID
        or not isinstance(embeddings, torch.Tensor)
        or embeddings.dtype != torch.float32
        or embeddings.ndim != 2
        or embeddings.shape[0] != reproduction["bank_source_rows"]
        or not bool(torch.isfinite(embeddings).all())
        or _tensor_digest(embeddings) != reproduction["bank_sha256"]
        or bank_record["bank_sha256"] != reproduction["bank_sha256"]
        or bank_record["k"] != _mapping(bank_record["provenance"])["bank_k"]
    ):
        raise _fail("persisted Fit-bank tensor differs from its accepted digest/provenance")
    provenance = _mapping(_load_json(provenance_path), "Fit-bank provenance")
    expected_keys = {
        "schema_id",
        "bank_recipe_id",
        "bank_order_id",
        "requested_arm_id",
        "bank_arm_id",
        "model_seed",
        "source_role",
        "source_history_ids",
        "source_manifest_sha256",
        "source_rows",
        "source_rows_sha256",
        "ordered_file_ids_sha256",
        "data_binding_sha256",
        "checkpoint_sha256",
        "model_state_sha256",
        "ema_state_sha256",
        "bank_sha256",
        "bank_k",
        "tensor_digest_id",
        "scope",
        "n4_scale_artifact",
        "source_commit",
        "task_id",
        "protocol_sha256",
        "metric_code_sha256",
        "protocol_path",
        "metric_code_path",
        "supplementary_definition_path",
        "supplementary_definition_sha256",
        "input_manifest_path",
        "input_manifest_sha256",
        "task14_lock_path",
        "task14_lock_sha256",
        "arm_config_sha256",
        "paired_baseline_arm_config_sha256",
        "paired_baseline_task14_lock_path",
        "paired_baseline_task14_lock_sha256",
        "paired_baseline_accepted_record_path",
        "paired_baseline_accepted_record_sha256",
        "handoff_path",
        "handoff_sha256",
        "accepted_record_path",
        "accepted_record_sha256",
        "checkpoint_path",
        "reference_artifact_kind",
        "reference_artifact_path",
        "reference_artifact_sha256",
        "data_root",
        "input_sha256",
    }
    if set(provenance) != expected_keys or dict(_mapping(bank_record["provenance"])) != dict(provenance):
        raise _fail("Fit-bank sidecar provenance differs from the immutable byte artifact")
    execution = _mapping(document["execution"])
    expected_reference_kind = (
        "k9_downstream"
        if document["arm_id"] == K9_ARM_ID
        else "n4_scale"
        if document["arm_id"] == "N4-CRN"
        else "none"
    )
    if (
        provenance["schema_id"] != S.BANK_PROVENANCE_SCHEMA_ID
        or provenance["requested_arm_id"] != document["arm_id"]
        or provenance["model_seed"] != document["model_seed"]
        or provenance["source_role"] != "FIT"
        or provenance["source_rows"] != reproduction["bank_source_rows"]
        or provenance["source_history_ids"] != fit_history_ids
        or provenance["source_manifest_sha256"] != fit_manifest_sha256
        or provenance["source_rows_sha256"] != reproduction["bank_source_rows_sha256"]
        or provenance["bank_sha256"] != reproduction["bank_sha256"]
        or provenance["bank_recipe_id"] != reproduction["bank_recipe_id"]
        or provenance["bank_order_id"] != reproduction["bank_order_id"]
        or provenance["checkpoint_sha256"] != inputs["checkpoint_sha256"]
        or provenance["model_state_sha256"] != reproduction["model_state_sha256"]
        or provenance["ema_state_sha256"] != reproduction["ema_state_sha256"]
        or provenance["data_binding_sha256"] != reproduction["data_binding_sha256"]
        or provenance["source_commit"] != execution["source_commit"]
        or provenance["task_id"] != execution["task_id"]
        or provenance["protocol_sha256"] != document["protocol_sha256"]
        or provenance["protocol_path"] != execution["protocol_path"]
        or provenance["metric_code_path"] != execution["metric_code_path"]
        or provenance["supplementary_definition_path"]
        != execution["supplementary_definition_path"]
        or provenance["metric_code_sha256"] != execution["metric_code_sha256"]
        or provenance["supplementary_definition_sha256"]
        != execution["supplementary_definition_sha256"]
        or provenance["input_manifest_path"] != inputs["manifest_path"]
        or provenance["input_manifest_sha256"] != inputs["manifest_sha256"]
        or provenance["task14_lock_path"] != inputs["task14_lock_path"]
        or provenance["task14_lock_sha256"] != inputs["task14_lock_sha256"]
        or provenance["arm_config_sha256"] != reproduction["arm_config_sha256"]
        or provenance["paired_baseline_arm_config_sha256"]
        != reproduction["paired_baseline_arm_config_sha256"]
        or provenance["paired_baseline_task14_lock_path"]
        != inputs["paired_baseline_task14_lock_path"]
        or provenance["paired_baseline_task14_lock_sha256"]
        != inputs["paired_baseline_task14_lock_sha256"]
        or provenance["paired_baseline_accepted_record_path"]
        != inputs["paired_baseline_accepted_record_path"]
        or provenance["paired_baseline_accepted_record_sha256"]
        != inputs["paired_baseline_accepted_record_sha256"]
        or provenance["handoff_path"] != inputs["handoff_path"]
        or provenance["handoff_sha256"] != inputs["handoff_sha256"]
        or provenance["accepted_record_path"] != inputs["accepted_record_path"]
        or provenance["accepted_record_sha256"] != inputs["accepted_record_sha256"]
        or provenance["checkpoint_path"] != inputs["checkpoint_path"]
        or provenance["reference_artifact_kind"] != expected_reference_kind
        or provenance["reference_artifact_path"] != inputs["reference_artifact_path"]
        or provenance["reference_artifact_sha256"] != inputs["reference_artifact_sha256"]
        or provenance["data_root"] != inputs["data_root"]
        or dict(_mapping(provenance["input_sha256"]))
        != dict(_mapping(inputs["sha256_before"]))
        or provenance["scope"] != "experiment"
    ):
        raise _fail("Fit-bank provenance does not match its score document inputs")
    scale = provenance["n4_scale_artifact"]
    if document["arm_id"] == "N4-CRN":
        if not isinstance(scale, Mapping):
            raise _fail("N4 bank provenance omits the strictly restored scale artifact")
        if set(scale) != {
            "path",
            "sha256",
            "state_dict_sha256",
            "training_history_ids",
            "bounds",
        }:
            raise _fail("N4 scale provenance is incomplete or has unknown fields")
        scale_path = Path(str(scale["path"]))
        bounds = _mapping(scale["bounds"], "N4 scale provenance bounds")
        training_history_ids = cast(list[str], scale["training_history_ids"])
        if (
            scale_path != Path(str(inputs["reference_artifact_path"]))
            or scale["sha256"] != inputs["reference_artifact_sha256"]
            or not scale_path.is_file()
            or _sha256_file(scale_path) != scale["sha256"]
            or set(bounds) != {"v_floor", "v_cap"}
            or not training_history_ids
            or sorted(training_history_ids) != training_history_ids
            or training_history_ids != list(_mapping(provenance)["source_history_ids"])
        ):
            raise _fail("N4 scale artifact identity differs from the score input manifest")
        try:
            restored_scale = init_n4_scale_model(
                int(V1Config().d_model), int(document["model_seed"]) + 999999
            )
            scale_metadata = load_n4_scale_checkpoint(
                scale_path,
                restored_scale,
                expected_fit_scope="final",
                expected_training_history_ids=training_history_ids,
                expected_bounds={
                    "v_floor": float(bounds["v_floor"]),
                    "v_cap": float(bounds["v_cap"]),
                },
            )
        except (OSError, ValueError) as exc:
            raise _fail(f"N4 persisted scale envelope failed strict verification: {exc}") from exc
        if scale_metadata["state_dict_sha256"] != scale["state_dict_sha256"]:
            raise _fail("N4 scale state digest differs from immutable bank provenance")
    elif scale is not None:
        raise _fail("non-N4 Fit-bank provenance unexpectedly names a scale artifact")


def _assert_seed_documents(
    documents: Sequence[Mapping[str, object]],
    *,
    expected_arm: str,
    expected_seeds: Sequence[int],
    metric_code: str,
    definition: str,
) -> None:
    if len(documents) != len(expected_seeds):
        raise _fail("supplementary metrics require exactly three seed documents")
    for index, document in enumerate(documents):
        if document["arm_id"] != expected_arm:
            raise _fail("score document arm differs from the requested arm")
        if document["model_seed"] != expected_seeds[index]:
            raise _fail("score documents are not in the canonical model-seed order")
        execution = _mapping(document["execution"])
        if execution["metric_code_sha256"] != metric_code:
            raise _fail("score document was produced by different metric-code bytes")
        if execution["supplementary_definition_sha256"] != definition:
            raise _fail("score document binds a different prospective supplement")



def _assert_n4_b0_parity(
    n4_documents: Sequence[Mapping[str, object]],
    b0_documents: Sequence[Mapping[str, object]],
) -> None:
    """Require paired N4/B0 checkpoint, bank, and exact S_pop score parity."""
    if len(n4_documents) != len(MODEL_SEEDS) or len(b0_documents) != len(MODEL_SEEDS):
        raise _fail("N4 population parity requires all three paired B0 seed documents")
    for n4, b0 in zip(n4_documents, b0_documents):
        n4_inputs = _mapping(n4["inputs"])
        b0_inputs = _mapping(b0["inputs"])
        n4_reproduction = _mapping(n4["reproduction"])
        b0_reproduction = _mapping(b0["reproduction"])
        if (
            n4_inputs["paired_baseline_task14_lock_path"] != b0_inputs["task14_lock_path"]
            or n4_inputs["paired_baseline_task14_lock_sha256"] != b0_inputs["task14_lock_sha256"]
            or n4_inputs["paired_baseline_accepted_record_path"] != b0_inputs["accepted_record_path"]
            or n4_inputs["paired_baseline_accepted_record_sha256"]
            != b0_inputs["accepted_record_sha256"]
            or n4_inputs["paired_baseline_arm_config_sha256"]
            != b0_reproduction["arm_config_sha256"]
            or n4_inputs["checkpoint_sha256"] != b0_inputs["checkpoint_sha256"]
            or n4_reproduction["model_state_sha256"] != b0_reproduction["model_state_sha256"]
            or n4_reproduction["ema_state_sha256"] != b0_reproduction["ema_state_sha256"]
            or n4_reproduction["bank_sha256"] != b0_reproduction["bank_sha256"]
            or n4_reproduction["bank_source_rows"] != b0_reproduction["bank_source_rows"]
            or n4_reproduction["bank_source_rows_sha256"]
            != b0_reproduction["bank_source_rows_sha256"]
        ):
            raise _fail("N4 is not bound to the same accepted B0 lock, record, checkpoint, and Fit bank")
        n4_support = _mapping(n4["support"])
        b0_support = _mapping(b0["support"])
        for field in (
            "fit_healthy_rows",
            "fit_healthy_rows_sha256",
            "calibration_healthy_rows",
            "calibration_support_sha256",
        ):
            if n4_support[field] != b0_support[field]:
                raise _fail(f"N4/B0 support differs for {field}")
        for field in ("fit_healthy_scores", "calibration_scores"):
            n4_scores = _mapping(n4_support[field])
            b0_scores = _mapping(b0_support[field])
            if dict(_mapping(n4_scores["S_pop"])) != dict(_mapping(b0_scores["S_pop"])):
                raise _fail(f"N4 S_pop differs from paired B0 on {field}")
        n4_thresholds = _mapping(n4_reproduction["calibration_thresholds"])
        b0_thresholds = _mapping(b0_reproduction["calibration_thresholds"])
        if n4_thresholds["S_pop"] != b0_thresholds["S_pop"]:
            raise _fail("N4 S_pop q95 differs from paired B0 Calibration")
        n4_development = [
            _mapping(item) for item in cast(list[object], n4_support["development"])
        ]
        b0_development = [
            _mapping(item) for item in cast(list[object], b0_support["development"])
        ]
        if len(n4_development) != len(b0_development):
            raise _fail("N4/B0 Development history counts differ")
        for n4_history, b0_history in zip(n4_development, b0_development):
            if (
                n4_history["history_id"] != b0_history["history_id"]
                or n4_history["manifest_sha256"] != b0_history["manifest_sha256"]
                or n4_history["support_sha256"] != b0_history["support_sha256"]
                or n4_history["scores"]["S_pop"] != b0_history["scores"]["S_pop"]
            ):
                raise _fail("N4 S_pop differs from paired B0 on Development support")

def _canonical_history_values(canonical: Mapping[str, object]) -> dict[str, float]:
    return {
        "P_AUROC": float(canonical["auc_p"]),
        "W_AUROC": float(canonical["auc_w"]),
        "PW_AUROC": float(canonical["auc_pw"]),
        "FAR_per_robot_day": float(canonical["far"]),
    }


def _canonical_by_seed(
    documents: Sequence[Mapping[str, object]],
    thresholds_by_seed: Mapping[str, Sequence[float]],
    *,
    history_order: Sequence[str],
    manifests: Mapping[str, Mapping[str, object]],
) -> list[dict[str, dict[str, Mapping[str, object]]]]:
    """Recompute frozen canonical per-history values from the frozen score maps."""
    rows_by_history = [manifests[history_id]["files"] for history_id in history_order]
    ledgers = [
        manifests[history_id].get("failure_events", []) for history_id in history_order
    ]
    windows = [manifests[history_id]["maintenance_windows"] for history_id in history_order]
    result: list[dict[str, dict[str, Mapping[str, object]]]] = []
    for seed_index, document in enumerate(documents):
        support = _mapping(document["support"])
        entries = {
            str(_mapping(entry)["history_id"]): _mapping(entry)
            for entry in cast(list[object], support["development"])
        }
        ordered = [entries[history_id] for history_id in history_order]
        per_branch: dict[str, dict[str, Mapping[str, object]]] = {}
        for name in S.SUPPLEMENTARY_BRANCHES:
            file_scores: dict[str, float] = {}
            for entry in ordered:
                scores = _mapping(_mapping(entry["scores"])[name])
                file_scores.update(
                    {file_id: float(value) for file_id, value in scores.items()}
                )
            canonical = P.evaluate_histories(
                file_scores,
                rows_by_history,
                ledgers,
                windows,
                float(thresholds_by_seed[name][seed_index]),
            )
            per_branch[name] = {
                history_id: canonical["per_history"][index]
                for index, history_id in enumerate(history_order)
            }
        result.append(per_branch)
    return result


def run_metrics(args: argparse.Namespace) -> dict[str, object]:
    _assert_new_path(args.out, "supplementary metric document")
    inputs = S.load_frozen_input_manifest(args.manifest)
    if inputs.arm_id != args.arm:
        raise _fail("metrics manifest arm differs from --arm")
    binding = load_task6_binding(inputs.handoff_path)
    digests_before = S.compute_input_digests(
        inputs, binding=binding, definition_path=args.definition
    )
    metric_code = S.metric_code_sha256()
    if args.arm == B0_ARM_ID and args.baseline_docs:
        raise _fail("B0 is its own baseline and must not receive B0 baseline documents")
    baseline_paths = list(args.baseline_docs or [])
    score_document_hashes_before = {
        **{
            f"score:seed:{index}:document": _sha256_file(path)
            for index, path in enumerate(args.seed_docs)
        },
        **{
            f"score:baseline:{index}:document": _sha256_file(path)
            for index, path in enumerate(baseline_paths)
        },
    }
    seed_documents = _load_score_documents(
        args.seed_docs,
        expected_data_root=inputs.data_root,
        expected_data_digests=digests_before,
        definition_path=args.definition,
    )
    _assert_seed_documents(
        seed_documents,
        expected_arm=args.arm,
        expected_seeds=MODEL_SEEDS,
        metric_code=metric_code,
        definition=inputs.supplementary_definition_sha256,
    )
    baseline_documents: list[Mapping[str, object]] = []
    if args.arm != B0_ARM_ID and baseline_paths:
        baseline_documents = _load_score_documents(
            baseline_paths,
            expected_data_root=inputs.data_root,
            expected_data_digests=digests_before,
            definition_path=args.definition,
        )
        _assert_seed_documents(
            baseline_documents,
            expected_arm=B0_ARM_ID,
            expected_seeds=MODEL_SEEDS,
            metric_code=metric_code,
            definition=inputs.supplementary_definition_sha256,
        )
    if args.arm == "N4-CRN" and baseline_documents:
        _assert_n4_b0_parity(seed_documents, baseline_documents)
    score_asset_digests_before = _score_asset_digests(
        seed_documents, args.seed_docs, "seed"
    )
    score_asset_digests_before.update(
        _score_asset_digests(baseline_documents, baseline_paths, "baseline")
    )
    if {
        key: value
        for key, value in score_asset_digests_before.items()
        if key.endswith(":document")
    } != score_document_hashes_before:
        raise _fail("score documents changed while their validated inputs were being loaded")
    digests_before.update(score_asset_digests_before)
    manifests = _development_manifests(inputs, binding)
    history_order = [
        history.actual_history_id for history in binding.role_histories("DEVELOPMENT")
    ]

    reference_identity: tuple | None = None
    for group in (seed_documents, baseline_documents):
        for document in group:
            support = _mapping(document["support"])
            entries = cast(list[object], support["development"])
            identity = tuple(
                (
                    str(_mapping(entry)["history_id"]),
                    str(_mapping(entry)["manifest_sha256"]),
                    str(_mapping(entry)["support_sha256"]),
                )
                for entry in entries
            )
            if reference_identity is None:
                reference_identity = identity
            elif identity != reference_identity:
                raise _fail("score documents differ in Development support identity")
            if [str(_mapping(entry)["history_id"]) for entry in entries] != history_order:
                raise _fail("score document histories are not in the binding order")
            expected_manifests = inputs.expected["development_manifest_sha256"]
            expected_supports = inputs.expected["development_support_sha256"]
            for entry in entries:
                history_entry = _mapping(entry)
                history_id = str(history_entry["history_id"])
                if str(history_entry["manifest_sha256"]) != str(expected_manifests[history_id]):
                    raise _fail(f"score document manifest differs for {history_id}")
                if str(history_entry["support_sha256"]) != str(expected_supports[history_id]):
                    raise _fail(f"score document support differs for {history_id}")
    assert reference_identity is not None

    thresholds_by_seed = {
        name: [
            float(_mapping(document["reproduction"])["calibration_thresholds"][name])
            for document in seed_documents
        ]
        for name in S.SUPPLEMENTARY_BRANCHES
    }
    baseline_thresholds_by_seed = (
        {
            name: [
                float(_mapping(document["reproduction"])["calibration_thresholds"][name])
                for document in baseline_documents
            ]
            for name in S.SUPPLEMENTARY_BRANCHES
        }
        if baseline_documents
        else thresholds_by_seed
    )
    arm_canonical = _canonical_by_seed(
        seed_documents, thresholds_by_seed, history_order=history_order, manifests=manifests
    )
    baseline_canonical = (
        _canonical_by_seed(
            baseline_documents,
            baseline_thresholds_by_seed,
            history_order=history_order,
            manifests=manifests,
        )
        if baseline_documents
        else (arm_canonical if args.arm == B0_ARM_ID else None)
    )

    branches: dict[str, object] = {}
    evaluation_counts: dict[str, tuple[int, int]] = {}
    for name in S.SUPPLEMENTARY_BRANCHES:
        per_seed_rows: list[dict[str, Mapping[str, object]]] = []
        for seed_index, document in enumerate(seed_documents):
            support = _mapping(document["support"])
            entries = {
                str(_mapping(entry)["history_id"]): _mapping(entry)
                for entry in cast(list[object], support["development"])
            }
            fit_scores = _mapping(support["fit_healthy_scores"])
            fit_branch_scores = tuple(
                float(value) for value in _mapping(fit_scores[name]).values()
            )
            seed_rows: dict[str, Mapping[str, object]] = {}
            for history_id in history_order:
                entry = entries[history_id]
                manifest = manifests[history_id]
                history_rows = manifest["files"]
                history_scores = _mapping(_mapping(entry["scores"])[name])
                history_input = S.HistoryEvaluationInput(
                    history_id=history_id,
                    schema_history_id=str(entry["schema_history_id"]),
                    role_manifest_sha256=str(entry["manifest_sha256"]),
                    support_sha256=str(entry["support_sha256"]),
                    rows=tuple(dict(row) for row in history_rows),
                    ledger=tuple(
                        dict(failure) for failure in manifest.get("failure_events", [])
                    ),
                    windows=manifest["maintenance_windows"],
                    scores={
                        file_id: float(value) for file_id, value in history_scores.items()
                    },
                    localization={
                        file_id: dict(_mapping(record))
                        for file_id, record in _mapping(entry["localization"]).items()
                    },
                    fit_healthy_scores=fit_branch_scores,
                    excluded_rows=int(entry["excluded_rows"]),
                )
                seed_rows[history_id] = S.build_history_metric_row(
                    history_input,
                    threshold=float(thresholds_by_seed[name][seed_index]),
                    canonical_entry=arm_canonical[seed_index][name][history_id],
                    baseline=(
                        None
                        if baseline_canonical is None
                        else _canonical_history_values(
                            baseline_canonical[seed_index][name][history_id]
                        )
                    ),
                )
                if seed_index == 0 and name == "S_pred":
                    events = S.evaluable_support(
                        [dict(row) for row in history_rows],
                        [dict(failure) for failure in manifest.get("failure_events", [])],
                        manifest["maintenance_windows"],
                    )
                    _rate, _hits, evaluated, excluded = S.localization_rate(
                        events.events,
                        history_input.scores,
                        history_input.localization,
                        S.row_times(history_rows),
                    )
                    evaluation_counts[history_id] = (evaluated, excluded)
            per_seed_rows.append(seed_rows)
        branches[name] = S.build_branch_metrics(
            thresholds_by_seed=thresholds_by_seed[name],
            per_seed_rows=per_seed_rows,
            baseline_per_seed_rows=None,
        )

    localization_exclusions: dict[str, int] = {history_id: 0 for history_id in history_order}
    for document in seed_documents:
        entries_by_history = {
            str(_mapping(entry)["history_id"]): _mapping(entry)
            for entry in cast(list[object], _mapping(document["support"])["development"])
        }
        for history_id in history_order:
            manifest = manifests[history_id]
            entry = entries_by_history[history_id]
            events = S.evaluable_support(
                [dict(row) for row in manifest["files"]],
                [dict(failure) for failure in manifest.get("failure_events", [])],
                manifest["maintenance_windows"],
            )
            _rate, _hits, _evaluated, excluded = S.localization_rate(
                events.events,
                {
                    file_id: float(value)
                    for file_id, value in _mapping(_mapping(entry["scores"])["S_pred"]).items()
                },
                {
                    file_id: dict(_mapping(record))
                    for file_id, record in _mapping(entry["localization"]).items()
                },
                S.row_times(manifest["files"]),
            )
            localization_exclusions[history_id] += excluded

    development_support = []
    for history in binding.role_histories("DEVELOPMENT"):
        history_id = history.actual_history_id
        entries = [
            _mapping(
                next(
                    entry
                    for entry in cast(list[object], _mapping(document["support"])["development"])
                    if str(_mapping(entry)["history_id"]) == history_id
                )
            )
            for document in seed_documents
        ]
        primary = entries[0]
        evaluated, _excluded = evaluation_counts[history_id]
        development_support.append(
            {
                "history_id": history_id,
                "schema_history_id": history.schema_history_id,
                "manifest_sha256": str(primary["manifest_sha256"]),
                "support_sha256": str(primary["support_sha256"]),
                "scores_sha256": _canonical_sha256(
                    [
                        {
                            "model_seed": MODEL_SEEDS[index],
                            "S_pred": S.score_map_sha256(
                                _mapping(_mapping(entry["scores"])["S_pred"])
                            ),
                            "S_pop": S.score_map_sha256(
                                _mapping(_mapping(entry["scores"])["S_pop"])
                            ),
                            "timesteps_sha256": str(entry["timesteps_sha256"]),
                        }
                        for index, entry in enumerate(entries)
                    ]
                ),
                "rows": int(primary["rows"]),
                "excluded_rows": int(primary["excluded_rows"]),
                "p": int(primary["p"]),
                "w": int(primary["w"]),
                "a": int(primary["a"]),
                "controls": int(primary["controls"]),
                "robot_days": float(primary["robot_days"]),
                "evaluated_pw_events": int(evaluated),
                "localization_excluded_events": int(localization_exclusions[history_id]),
            }
        )

    missing: set[str] = set()
    if args.arm != B0_ARM_ID and not baseline_documents:
        missing.add("paired B0 baseline score documents are required for paired deltas")
    for name in S.SUPPLEMENTARY_BRANCHES:
        branch = _mapping(branches[name])
        for seed_row in cast(list[object], branch["per_seed"]):
            for metric in cast(list[object], _mapping(seed_row)["per_history"]):
                row = _mapping(metric)
                for field in S.SUPPLEMENTARY_HISTORY_KEYS - {"history_id"}:
                    value = row[field]
                    if field in ("severity_ordering", "subtype_slices"):
                        if any(item is None for item in _mapping(value).values()):
                            missing.add(f"{name}.{row['history_id']}.{field}")
                    elif value is None:
                        missing.add(f"{name}.{row['history_id']}.{field}")
                duration = _mapping(row["duration_accounting"])
                for field in (
                    "median_declared_duration_days",
                    "median_detected_alert_span_days",
                    "median_first_alert_lead_days",
                ):
                    if duration[field] is None:
                        missing.add(f"{name}.{row['history_id']}.duration_accounting.{field}")
        aggregate = _mapping(branch["aggregate"])
        for field in S._AGGREGATE_KEYS:
            value = aggregate[field]
            if value is None or (
                field == "background_uncertainty"
                and any(item is None for item in cast(list[object], value))
            ):
                missing.add(f"{name}.aggregate.{field}")
    status = "VALID" if not missing else "INVALID"
    failure_reasons = sorted(missing)

    digests_after = S.compute_input_digests(
        inputs, binding=binding, definition_path=args.definition
    )
    digests_after.update(
        _score_asset_digests(seed_documents, args.seed_docs, "seed")
    )
    digests_after.update(
        _score_asset_digests(baseline_documents, baseline_paths, "baseline")
    )
    S.assert_inputs_unchanged(digests_before, digests_after)

    reference_artifact = inputs.reference_artifact
    document = {
        "schema_id": S.SUPPLEMENTARY_SCHEMA_ID,
        "schema_version": 2,
        "protocol_id": "sprint18-full-training-protocol-v2",
        "protocol_sha256": inputs.protocol_sha256,
        "phase": "development",
        "arm_id": inputs.arm_id,
        "model_seed": MODEL_SEEDS[0],
        "execution": {
            "task_id": TASK_ID,
            "attempt_id": args.attempt_id,
            "agent_id": args.agent_id,
            "source_commit": inputs.source_commit,
            "metric_code_sha256": metric_code,
            "protocol_path": str(S.PROTOCOL_PATH.resolve()),
            "metric_code_path": str(Path(S.__file__).resolve()),
            "supplementary_definition_path": str(Path(args.definition).resolve()),
            "supplementary_definition_sha256": inputs.supplementary_definition_sha256,
            "scope": "experiment",
            "phase": "development",
            "device": args.device,
            "started_utc": args.started_utc,
            "finished_utc": _utc_now(),
        },
        "inputs": {
            "manifest_path": inputs.manifest_path,
            "manifest_sha256": inputs.manifest_sha256,
            "data_root": inputs.data_root,
            "checkpoint_path": inputs.checkpoint_path,
            "checkpoint_sha256": inputs.checkpoint_sha256,
            "reference_artifact_path": reference_artifact["path"],
            "reference_artifact_sha256": reference_artifact["sha256"],
            "accepted_record_path": inputs.accepted_record_path,
            "accepted_record_sha256": inputs.accepted_record_sha256,
            "task14_lock_path": inputs.task14_lock_path,
            "task14_lock_sha256": inputs.task14_lock_sha256,
            "paired_baseline_task14_lock_path": inputs.paired_baseline_task14_lock_path,
            "paired_baseline_task14_lock_sha256": inputs.paired_baseline_task14_lock_sha256,
            "paired_baseline_accepted_record_path": inputs.paired_baseline_accepted_record_path,
            "paired_baseline_accepted_record_sha256": inputs.paired_baseline_accepted_record_sha256,
            "paired_baseline_arm_config_sha256": inputs.expected[
                "paired_baseline_arm_config_sha256"
            ],
            "handoff_path": inputs.handoff_path,
            "handoff_sha256": inputs.handoff_sha256,
            "seed_score_documents": [
                {
                    "path": str(path),
                    "sha256": digests_after[f"score:seed:{index}:document"],
                }
                for index, path in enumerate(args.seed_docs)
            ],
            "baseline_score_documents": [
                {
                    "path": str(path),
                    "sha256": digests_after[f"score:baseline:{index}:document"],
                }
                for index, path in enumerate(baseline_paths)
            ],
            "sha256_before": dict(digests_before),
            "sha256_after": dict(digests_after),
            "unchanged": True,
        },
        "reproduction": {
            "arm_config_sha256_by_seed": [
                str(_mapping(seed["reproduction"])["arm_config_sha256"])
                for seed in seed_documents
            ],
            "paired_baseline_arm_config_sha256_by_seed": [
                _mapping(seed["reproduction"]).get("paired_baseline_arm_config_sha256")
                for seed in seed_documents
            ],
            "data_binding_sha256": str(
                _mapping(seed_documents[0]["reproduction"])["data_binding_sha256"]
            ),
            "model_state_sha256_by_seed": [
                str(_mapping(seed["reproduction"])["model_state_sha256"])
                for seed in seed_documents
            ],
            "ema_state_sha256_by_seed": [
                str(_mapping(seed["reproduction"])["ema_state_sha256"])
                for seed in seed_documents
            ],
            "bank_sha256_by_seed": [
                str(_mapping(seed["reproduction"])["bank_sha256"])
                for seed in seed_documents
            ],
            "accepted_bank_sha256_by_seed": [
                str(_mapping(seed["reproduction"])["accepted_bank_sha256"])
                for seed in seed_documents
            ],
            "bank_reproduced": all(
                _mapping(seed["reproduction"])["bank_reproduced"] is True
                for seed in seed_documents
            ),
            "bank_source": (
                "restored_from_custody"
                if all(
                    _mapping(seed["reproduction"])["bank_source"] == "restored_from_custody"
                    for seed in seed_documents
                )
                else "reconstructed_digest_verified"
            ),
            "reference_state_modes": {
                name: (
                    "restored_from_custody"
                    if all(
                        _mapping(_mapping(seed["reproduction"])["reference_state_modes"])[name]
                        == "restored_from_custody"
                        for seed in seed_documents
                    )
                    else "reconstructed_digest_verified"
                )
                for name in S.SUPPLEMENTARY_BRANCHES
            },
            "calibration_thresholds_by_seed": thresholds_by_seed,
            "accepted_calibration_thresholds_by_seed": thresholds_by_seed,
            "recomputed_calibration_thresholds_by_seed": {
                name: [
                    float(
                        _mapping(seed["reproduction"])["recomputed_calibration_thresholds"][name]
                    )
                    for seed in seed_documents
                ]
                for name in S.SUPPLEMENTARY_BRANCHES
            },
            "calibration_reproduced": True,
            "representation_state_unchanged": all(
                _mapping(seed["reproduction"])["representation_state_unchanged"] is True
                for seed in seed_documents
            ),
            "reference_state_unchanged": all(
                _mapping(seed["reproduction"])["reference_state_unchanged"] is True
                for seed in seed_documents
            ),
            "encoder_updates": 0,
        },
        "support": {
            "fit_healthy_rows": int(
                _mapping(seed_documents[0]["support"])["fit_healthy_rows"]
            ),
            "fit_healthy_rows_sha256": str(
                _mapping(seed_documents[0]["support"])["fit_healthy_rows_sha256"]
            ),
            "fit_healthy_scores_sha256": _canonical_sha256(
                [
                    dict(_mapping(_mapping(seed["support"])["fit_healthy_scores_sha256"]))
                    for seed in seed_documents
                ]
            ),
            "calibration_healthy_rows": int(
                _mapping(seed_documents[0]["support"])["calibration_healthy_rows"]
            ),
            "calibration_support_sha256": str(
                _mapping(seed_documents[0]["support"])["calibration_support_sha256"]
            ),
            "calibration_scores_sha256": _canonical_sha256(
                [
                    dict(_mapping(_mapping(seed["support"])["calibration_scores_sha256"]))
                    for seed in seed_documents
                ]
            ),
            "development": development_support,
            "excluded_rows": sum(int(entry["excluded_rows"]) for entry in development_support),
        },
        "metric_definitions": {
            "localization": S.LOCALIZATION_ID,
            "duration_accounting": S.DURATION_ACCOUNTING_ID,
            "severity_ordering": S.SEVERITY_ID,
            "subtype_slices": S.SUBTYPE_SLICE_ID,
            "background_stability": S.STABILITY_ID,
            "background_uncertainty": S.UNCERTAINTY_ID,
            "event_metrics": P.PROBE_ID,
        },
        "branches": branches,
        "status": status,
        "failure_reasons": failure_reasons,
    }
    S.validate_supplementary_result(document)
    _write_json(args.out, document)
    return {
        "mode": "metrics",
        "status": status,
        "failure_reasons": failure_reasons,
        "arm_id": inputs.arm_id,
        "out": str(args.out),
        "out_sha256": _sha256_file(args.out),
        "aggregate": {
            name: _mapping(_mapping(branches[name])["aggregate"])
            for name in S.SUPPLEMENTARY_BRANCHES
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    manifest = sub.add_parser(
        "prepare-manifest",
        help="derive one frozen input manifest from explicit accepted custody paths",
    )
    manifest.add_argument("--arm", required=True, choices=list(S.SUPPLEMENTARY_VALID_ARMS))
    manifest.add_argument("--model-seed", required=True, type=int)
    manifest.add_argument("--accepted-record", required=True)
    manifest.add_argument("--task14-lock", required=True)
    manifest.add_argument(
        "--paired-baseline-task14-lock",
        default=None,
        help="explicit accepted B0 Task14 lock; required with --arm N4-CRN",
    )
    manifest.add_argument(
        "--paired-baseline-record",
        default=None,
        help="explicit accepted same-seed B0 record; required with --arm N4-CRN",
    )
    manifest.add_argument("--handoff", required=True)
    manifest.add_argument("--data-root", required=True)
    manifest.add_argument("--source-commit", required=True)
    manifest.add_argument("--checkpoint-path", default=None)
    manifest.add_argument("--reference-artifact-path", default=None)
    manifest.add_argument("--out", required=True)
    scores = sub.add_parser("scores", help="produce one arm/seed supplementary score document")
    scores.add_argument("--manifest", required=True)
    scores.add_argument("--definition", required=True)
    scores.add_argument("--out", required=True)
    scores.add_argument("--device", default="cpu")
    scores.add_argument("--attempt-id", default=DEFAULT_ATTEMPT)
    scores.add_argument("--agent-id", default=DEFAULT_AGENT)
    scores.add_argument("--started-utc", default=None)
    metrics = sub.add_parser("metrics", help="aggregate the three seed score documents")
    metrics.add_argument("--manifest", required=True)
    metrics.add_argument("--definition", required=True)
    metrics.add_argument("--arm", required=True, choices=list(S.SUPPLEMENTARY_VALID_ARMS))
    metrics.add_argument("--seed-docs", nargs=3, required=True)
    metrics.add_argument("--baseline-docs", nargs=3, default=None)
    metrics.add_argument("--out", required=True)
    metrics.add_argument("--device", default="cpu")
    metrics.add_argument("--attempt-id", default=DEFAULT_ATTEMPT)
    metrics.add_argument("--agent-id", default=DEFAULT_AGENT)
    metrics.add_argument("--started-utc", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "prepare-manifest":
        result = prepare_input_manifest(args)
    else:
        if args.started_utc is None:
            args.started_utc = _utc_now()
        result = run_scores(args) if args.mode == "scores" else run_metrics(args)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
