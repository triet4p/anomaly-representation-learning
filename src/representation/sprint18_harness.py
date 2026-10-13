"""Executable common-contract harness for the accepted Sprint 18 v2 protocol.

This module validates arm identity, certified role bindings, Fit/Calibration
provenance, paired support, independent scores, compute observations, and the
closed v2 result shape. It deliberately does not implement or start experiment
training or any N1-N4 mechanism.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast
import zipfile

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

if TYPE_CHECKING:
    from representation.sprint18_checkpoint import RestoredSprint18Checkpoint
from representation.config import V1Config
from representation.data import collate_variable_files
from representation.inference import NormalReferenceBank, RepresentationInference
from representation.model import V1RepresentationModel
from synth.config import PatchConfig
from synth.dataset import load_sample_bytes
from synth.patchify import Patchifier
from synth.schema import DegradationStage, FileSample, SampleLabel

PROTOCOL_ID = "sprint18-full-training-protocol-v2"
PROTOCOL_SHA256 = "e1dcd17914e920520ad0d8f50e185874b95414e6a4fe907903619daf145622d2"
RESULT_SCHEMA_ID = "sprint18-full-training-result-v2"
VERDICT_SCHEMA_ID = "sprint18-verdict-v2"
HANDOFF_SCHEMA_ID = "s18-t71-certified-data-handoff-v1"
HANDOFF_SHA256 = "5c28434b6f5b3342a3203a73cd672782e08857299c05b709adbd8bce75d45155"
DATA_BINDING_RAW_SHA256 = "b8448292b023c9268f64886e5ca68a393c36ff5f2c41573ff160d50ce8ee6c58"
DATA_BINDING_CANONICAL_SHA256 = "8e2705140d07d9c603989ec709de5ebb766e580a8619b37c7462f6dd2751dd3b"
DATA_ROSTER_ID = "S18-ITER-0005"
DATA_ROOT_RELATIVE = (
    "data/generated/sprint18-iterative-v1/"
    "S18-ITER-0005-ASSESS-P-ALLOWANCE-V1"
)
WHOLE16_CERTIFICATION_SHA256 = "1345d1c8e04c35f3fbb18bddec2b5d474f91ec24c1f1c5081040b342debe2897"
QUALIFICATION_RECORD_SHA256 = "c5a8e8ccbc6be35a58fe7363ee4e8af489d3fafb93aaf6411c3936e5d349dc4b"
MANIFEST_AGGREGATE_SHA256 = "378baf290aa8c17d3d8125942e65def735fefd556ee8c8c2cddad289cb71a309"
MODEL_SEEDS = (181801, 181802, 181803)
BATCH_SIZE = 128
MODEL_LR_MAX = 1.5e-3
MODEL_LR_MIN = 1e-5
EMA_DECAY = 0.996
FIT_HEALTHY_ROWS = 5446
FIT_VALID_PATCHES = 167419
CALIBRATION_HEALTHY_ROWS = 1877
MODEL_CARDINALITIES = MappingProxyType({"n_robots": 9, "n_programs": 8})
PATCH_GEOMETRY = MappingProxyType({"n_channels": 6, "patch_size": 32, "stride": 16})

ArmId = Literal[
    "B0-FULL",
    "K3-FULL",
    "K7-FULL",
    "K9-FULL",
    "N1-HCC",
    "N2-MSP",
    "N3-CCRP",
    "N4-CRN",
]
RoleName = Literal["DESIGN", "FIT", "CALIBRATION", "DEVELOPMENT", "CONFIRMATION"]
Phase = Literal["development", "confirmation"]
ScoreBranchName = Literal["S_pred", "S_pop"]

MECHANISM_SPEC_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "N1-HCC": "0f5ede73d65441014caf22a22058c17d8ed34be735790d20ca7f36873a34d144",
        "N2-MSP": "a2deeb5479957ef8777ec4cfd66898e3ecd5e0aef80e2043f11882529ae3a9f2",
        "N3-CCRP": "eee44cebd29de2b5cbfdf0263fcca21d82fee92e0773922c0bc3209dee50443c",
        "N4-CRN": "4d1c8097c9cc27d3fbeb2e2a6e70a29a7bc63d7899646a9c712a315b39d9ffe3",
    }
)
CONTRACT_FILE_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "experiments/sprint18-full-training-protocol-v2.md": PROTOCOL_SHA256,
        "experiments/sprint18-hierarchical-conditional-contrastive-spec-v2.md": MECHANISM_SPEC_SHA256["N1-HCC"],
        "experiments/sprint18-multiscale-latent-prediction-spec-v2.md": MECHANISM_SPEC_SHA256["N2-MSP"],
        "experiments/sprint18-cross-channel-relational-prediction-spec-v2.md": MECHANISM_SPEC_SHA256["N3-CCRP"],
        "experiments/sprint18-context-residual-normalization-spec-v2.md": MECHANISM_SPEC_SHA256["N4-CRN"],
    }
)

DATA_PERMISSIONS: Mapping[RoleName, str] = MappingProxyType(
    {
        "DESIGN": "Integrity, causal, construction, and unchanged structural qualification; diagnostics only. No Fit/Calibration use or history selection.",
        "FIT": "Verified-healthy eligibility; fit only the fixed probe's feature mean/std and healthy centroid. No representation training, bank, alternate scorer, or non-probe fitting.",
        "CALIBRATION": "Verified-healthy eligibility; score with frozen Fit probe values and choose only the fixed probe q95. No fitting or retuning.",
        "DEVELOPMENT": "Structural qualification and one fixed-probe evaluation using frozen Fit/Calibration values. No selection, promotion, pruning, or tuning.",
        "CONFIRMATION": "After Task69 preceding-role PASS and Main release: structural certification and one fixed-probe evaluation using frozen Fit/Calibration values. No fitting, recalibration, retuning, rescue, or model score.",
    }
)
MODEL_PERMISSIONS: Mapping[RoleName, str] = MappingProxyType(
    {
        "DESIGN": "Healthy-only fixed coefficient-selection protocol; never final fitting or bank fitting.",
        "FIT": "Verified-healthy-only representation fitting, EMA targets, arm-specific reference banks/standardizers, and declared Fit-only auxiliary fitting/cross-fitting.",
        "CALIBRATION": "Verified-healthy-only scoring for each independent branch's q95 threshold; no model, bank, scale, or standardizer fitting.",
        "DEVELOPMENT": "Locked descriptive scoring on fixed common support only; no refit, pruning, promotion, checkpoint reselection, recalibration, or coefficient change.",
        "CONFIRMATION": "One-shot locked scoring only after Task33 lock and Main release; no refit, recalibration, retuning, rescue, or rerun.",
    }
)
ROLE_COUNTS: Mapping[RoleName, int] = MappingProxyType(
    {"DESIGN": 4, "FIT": 3, "CALIBRATION": 1, "DEVELOPMENT": 4, "CONFIRMATION": 4}
)
_SCHEMA_ROLE_PREFIX: Mapping[RoleName, str] = MappingProxyType(
    {"DESIGN": "DES", "FIT": "FIT", "CALIBRATION": "CAL", "DEVELOPMENT": "DEV", "CONFIRMATION": "CONF"}
)
_ROLE_ACTIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "train_fit": frozenset({"FIT"}),
        "fit_bank": frozenset({"FIT"}),
        "fit_standardizer": frozenset({"FIT"}),
        "fit_auxiliary": frozenset({"FIT"}),
        "cross_fit": frozenset({"FIT"}),
        "select_coefficients": frozenset({"DESIGN"}),
        "calibrate_q95": frozenset({"CALIBRATION"}),
        "score_development": frozenset({"DEVELOPMENT"}),
        "score_confirmation": frozenset({"CONFIRMATION"}),
    }
)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_BATCH_KEYS = frozenset(
    {
        "signals",
        "file_valid_mask",
        "patches",
        "patch_valid_mask",
        "patch_pad_mask",
        "starts",
        "valid_len",
        "mask",
        "file_ids",
        "robot_idx",
        "program_idx",
    }
)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """Frozen scientific identity and principal-change semantics for one arm."""

    arm_id: ArmId
    family: str
    component_set: tuple[str, ...]
    principal_change: str
    mechanism_spec_sha256: str | None
    representation_epochs: int
    scale_epochs: int


ARM_REGISTRY: Mapping[ArmId, ArmSpec] = MappingProxyType(
    {
        "B0-FULL": ArmSpec(
            "B0-FULL", "baseline", (), "unchanged B0 under the common Sprint18 training/evaluation contract", None, 100, 0
        ),
        "K3-FULL": ArmSpec(
            "K3-FULL", "legacy_stack", ("C5-A", "C6-A"), "exact Sprint17 K3 membership C5-A + C6-A; no member substitution", None, 100, 0
        ),
        "K7-FULL": ArmSpec(
            "K7-FULL", "legacy_stack", ("C2-A", "C3-A", "C5-A", "C6-A"), "exact Sprint17 K7 membership C2-A + C3-A + C5-A + C6-A; no member substitution", None, 100, 0
        ),
        "K9-FULL": ArmSpec(
            "K9-FULL", "legacy_stack", ("C2-A", "C3-A", "C5-A", "C6-A", "C7-A", "C8-A", "C9-B"), "exact Sprint17 K9 membership C2-A + C3-A + C5-A + C6-A + C7-A + C8-A + C9-B; no member substitution", None, 100, 0
        ),
        "N1-HCC": ArmSpec(
            "N1-HCC", "new_mechanism", ("N1-HCC",), "replace B0 InfoNCE with the one hierarchical relation-target contrastive objective; preserve B0 prediction/EMA/scoring", MECHANISM_SPEC_SHA256["N1-HCC"], 100, 0
        ),
        "N2-MSP": ArmSpec(
            "N2-MSP", "new_mechanism", ("N2-MSP",), "add only fine/4-patch/8-patch latent prediction; retain the common InfoNCE objective and fine-only primary S_pred", MECHANISM_SPEC_SHA256["N2-MSP"], 100, 0
        ),
        "N3-CCRP": ArmSpec(
            "N3-CCRP", "new_mechanism", ("N3-CCRP",), "add only held-out-channel relational latent prediction; retain common InfoNCE and uncorrupted primary inference", MECHANISM_SPEC_SHA256["N3-CCRP"], 100, 0
        ),
        "N4-CRN": ArmSpec(
            "N4-CRN", "new_mechanism", ("N4-CRN",), "normalize only S_pred with the Fit-only context-dependent scale model over frozen B0; zero representation updates and unchanged S_pop", MECHANISM_SPEC_SHA256["N4-CRN"], 0, 50
        ),
    }
)


class Sprint18ArmConfig(BaseModel):
    """Closed arm execution identity/configuration; coefficients stay unselected until locked."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: ArmId
    model_seed: int
    component_set: tuple[str, ...]
    principal_change: str
    protocol_id: Literal["sprint18-full-training-protocol-v2"] = PROTOCOL_ID
    protocol_sha256: str = PROTOCOL_SHA256
    mechanism_spec_sha256: str | None = None
    n_channels: int = 6
    patch_size: int = 32
    stride: int = 16
    n_robots: int = 9
    n_programs: int = 8
    optimizer: Literal["AdamW"] = "AdamW"
    batch_size: int
    batch_size_unit: Literal["files", "patch_records"]
    learning_rate: float
    weight_decay: float
    clip_norm: Literal[1.0] = 1.0
    precision: Literal["float32"] = "float32"
    representation_epochs: Literal[0, 100]
    scale_epochs: Literal[0, 50]
    ema_decay: Literal[0.996] = EMA_DECAY
    lambda_info_max: float | None = None
    lambda_hcc_max: float | None = None
    alpha: float | None = None
    beta: float | None = None
    primary_checkpoint_epoch: Literal[50, 100]
    selection_state: Literal["locked", "task6_smoke_only"] = "locked"
    coefficient_lock_sha256: str | None = None


def arm_spec(arm_id: str) -> ArmSpec:
    """Return the exact immutable registered arm or fail closed."""
    try:
        spec = ARM_REGISTRY[cast(ArmId, arm_id)]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown Sprint18 arm identity {arm_id!r}") from exc
    validate_arm_spec(spec)
    return spec


def validate_arm_spec(spec: ArmSpec) -> ArmSpec:
    """Reject any registry entry or supplied identity whose semantics drift."""
    if not isinstance(spec, ArmSpec) or spec.arm_id not in ARM_REGISTRY:
        raise ValueError("arm identity is not an exact Sprint18 registry entry")
    if ARM_REGISTRY[spec.arm_id] != spec:
        raise ValueError(f"arm metadata mismatch for {spec.arm_id}")
    if spec.arm_id.startswith("N") and len(spec.component_set) != 1:
        raise ValueError("each N arm must declare exactly one principal mechanism")
    if spec.arm_id == "B0-FULL" and spec.component_set:
        raise ValueError("B0-FULL must not declare a mechanism component")
    if spec.arm_id == "K3-FULL" and spec.component_set != ("C5-A", "C6-A"):
        raise ValueError("K3-FULL membership differs from the frozen registry")
    if spec.arm_id == "K7-FULL" and spec.component_set != ("C2-A", "C3-A", "C5-A", "C6-A"):
        raise ValueError("K7-FULL membership differs from the frozen registry")
    if spec.arm_id == "K9-FULL" and spec.component_set != ("C2-A", "C3-A", "C5-A", "C6-A", "C7-A", "C8-A", "C9-B"):
        raise ValueError("K9-FULL membership differs from the frozen registry")
    return spec


def validate_arm_config(config: Sprint18ArmConfig, *, require_locked: bool = True) -> Sprint18ArmConfig:
    """Validate common settings, exact arm deltas, coefficients, and v2 hashes."""
    spec = arm_spec(config.arm_id)
    if config.model_seed not in MODEL_SEEDS:
        raise ValueError("model_seed is outside the frozen Sprint18 seed namespace")
    if config.component_set != spec.component_set or config.principal_change != spec.principal_change:
        raise ValueError("configuration changes the registered arm identity or principal-change semantics")
    if config.protocol_id != PROTOCOL_ID or config.protocol_sha256 != PROTOCOL_SHA256:
        raise ValueError("configuration is not bound to the accepted v2 protocol bytes")
    if config.mechanism_spec_sha256 != spec.mechanism_spec_sha256:
        raise ValueError("configuration mechanism-spec hash does not match its registered arm")
    if (config.n_channels, config.patch_size, config.stride) != (6, 32, 16):
        raise ValueError("model patch/channel geometry differs from the certified handoff")
    if (config.n_robots, config.n_programs) != (9, 8):
        raise ValueError("model conditioning cardinalities differ from certified Fit support")
    expected = (
        (config.representation_epochs, config.scale_epochs, config.primary_checkpoint_epoch,
         config.batch_size, config.batch_size_unit, config.learning_rate, config.weight_decay)
    )
    if config.arm_id == "N4-CRN":
        frozen = (0, 50, 50, 512, "patch_records", 1e-3, 1e-4)
        applicable = (config.lambda_info_max, config.lambda_hcc_max, config.alpha, config.beta)
        if applicable != (None, None, None, None):
            raise ValueError("N4-CRN has no representation-loss coefficients")
    else:
        frozen = (100, 0, 100, 128, "files", 1.5e-3, 0.0)
    if expected != frozen:
        raise ValueError("configuration changes the frozen common optimizer/epoch/checkpoint settings")
    info_grid = (0.025, 0.05, 0.1)
    aux_grid = (0.1, 0.25, 0.5)
    if config.arm_id in ("B0-FULL", "K3-FULL", "K7-FULL", "K9-FULL"):
        required = (config.lambda_info_max, None, None, None)
    elif config.arm_id == "N1-HCC":
        required = (None, config.lambda_hcc_max, None, None)
    elif config.arm_id == "N2-MSP":
        required = (config.lambda_info_max, None, config.alpha, None)
    elif config.arm_id == "N3-CCRP":
        required = (config.lambda_info_max, None, None, config.beta)
    else:
        required = (None, None, None, None)
    actual = (config.lambda_info_max, config.lambda_hcc_max, config.alpha, config.beta)
    if actual != required:
        raise ValueError("configuration activates a coefficient outside the arm's declared principal change")
    if config.lambda_info_max is not None and config.lambda_info_max not in info_grid:
        raise ValueError("InfoNCE maximum is outside the frozen healthy-only grid")
    if config.lambda_hcc_max is not None and config.lambda_hcc_max not in info_grid:
        raise ValueError("HCC maximum is outside the frozen healthy-only grid")
    if config.alpha is not None and config.alpha not in aux_grid:
        raise ValueError("N2 alpha is outside the frozen healthy-only grid")
    if config.beta is not None and config.beta not in aux_grid:
        raise ValueError("N3 beta is outside the frozen healthy-only grid")
    if config.selection_state == "locked":
        if not _is_hash(config.coefficient_lock_sha256):
            raise ValueError("locked coefficient configuration requires its frozen coefficient-lock digest")
    elif config.coefficient_lock_sha256 is not None:
        raise ValueError("Task6 smoke-only coefficients must not claim a selection lock")
    if require_locked and config.selection_state != "locked":
        raise ValueError("smoke-only/unselected coefficients cannot start an experiment")
    return config


def make_sprint18_arm_config(
    arm_id: str,
    model_seed: int,
    coefficients: Mapping[str, float | None],
    *,
    selection_state: Literal["locked", "task6_smoke_only"] = "locked",
    coefficient_lock_sha256: str | None = None,
) -> Sprint18ArmConfig:
    """Bind caller-supplied coefficients to one exact registered arm; select none."""
    spec = arm_spec(arm_id)
    coefficient_names = {"lambda_info_max", "lambda_hcc_max", "alpha", "beta"}
    if set(coefficients) != coefficient_names:
        raise ValueError("coefficients must explicitly name all four frozen coefficient slots")
    n4 = arm_id == "N4-CRN"
    config = Sprint18ArmConfig(
        arm_id=spec.arm_id,
        model_seed=model_seed,
        component_set=spec.component_set,
        principal_change=spec.principal_change,
        mechanism_spec_sha256=spec.mechanism_spec_sha256,
        batch_size=512 if n4 else 128,
        batch_size_unit="patch_records" if n4 else "files",
        learning_rate=1e-3 if n4 else MODEL_LR_MAX,
        weight_decay=1e-4 if n4 else 0.0,
        representation_epochs=spec.representation_epochs,
        scale_epochs=spec.scale_epochs,
        primary_checkpoint_epoch=50 if n4 else 100,
        lambda_info_max=coefficients["lambda_info_max"],
        lambda_hcc_max=coefficients["lambda_hcc_max"],
        alpha=coefficients["alpha"],
        beta=coefficients["beta"],
        selection_state=selection_state,
        coefficient_lock_sha256=coefficient_lock_sha256,
    )
    return validate_arm_config(config, require_locked=(selection_state == "locked"))


def arm_config_sha256(config: Sprint18ArmConfig) -> str:
    """Hash the canonical closed arm configuration."""
    encoded = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class BoundHistory:
    """One handoff history plus the schema-only alias required by the v2 result shape."""

    assigned_role: RoleName
    actual_history_id: str
    schema_history_id: str
    manifest_sha256: str
    data_seed: int
    sample_count: int
    config_hash: str
    relative_path: str
    data_permitted_use: str
    model_permitted_use: str


@dataclass(frozen=True, slots=True)
class Task6Binding:
    """Hash-bound executable mapping of the certified DATA roster to model roles."""

    candidate_id: str
    data_root_relative: str
    handoff_sha256: str
    data_binding_raw_sha256: str
    data_binding_canonical_sha256: str
    whole16_certification_sha256: str
    qualification_record_sha256: str
    manifest_aggregate_sha256: str
    protocol_sha256: str
    mechanism_spec_sha256: Mapping[str, str]
    histories: tuple[BoundHistory, ...]
    fit_healthy_rows: int
    calibration_healthy_rows: int
    executable_binding_sha256: str

    def history(self, actual_history_id: str) -> BoundHistory:
        for item in self.histories:
            if item.actual_history_id == actual_history_id:
                return item
        raise ValueError(f"history {actual_history_id!r} is not in the certified DATA handoff")

    def role_histories(self, role: RoleName) -> tuple[BoundHistory, ...]:
        return tuple(item for item in self.histories if item.assigned_role == role)


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and _HEX_SHA256.fullmatch(value) is not None


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def verify_contract_files(repository_root: str | Path) -> Mapping[str, str]:
    """Require the complete accepted v2 protocol/spec byte set, never a v1 mix."""
    root = Path(repository_root)
    observed: dict[str, str] = {}
    for relative, expected in CONTRACT_FILE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"accepted v2 contract file is missing: {relative}")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"accepted v2 contract digest mismatch for {relative}")
        observed[relative] = digest
    return MappingProxyType(observed)


def load_task6_binding(handoff_path: str | Path) -> Task6Binding:
    """Verify the certified 16-history handoff and derive its executable role binding."""
    path = Path(handoff_path)
    raw = path.read_bytes()
    handoff_sha = sha256(raw).hexdigest()
    if handoff_sha != HANDOFF_SHA256:
        raise ValueError("certified DATA handoff bytes differ from the reviewed Task71 artifact")
    root = path.parent.parent
    verify_contract_files(root)
    doc = json.loads(raw)
    if not isinstance(doc, dict) or doc.get("schema") != HANDOFF_SCHEMA_ID:
        raise ValueError("certified DATA handoff schema identity is invalid")
    data_root = doc.get("data_root")
    identity = doc.get("identity")
    fit_cal = doc.get("fit_calibration_status")
    roster = doc.get("roster")
    if not isinstance(data_root, dict) or not isinstance(identity, dict) or not isinstance(fit_cal, dict):
        raise ValueError("certified DATA handoff is missing its binding/qualification sections")
    if not isinstance(roster, list) or len(roster) != sum(ROLE_COUNTS.values()):
        raise ValueError("certified DATA roster must contain exactly 16 histories")
    if data_root.get("candidate_id") != DATA_ROSTER_ID or data_root.get("relative_to_isolated_worktree") != DATA_ROOT_RELATIVE:
        raise ValueError("DATA candidate/root differs from the certified S18-ITER-0005 handoff")
    if identity.get("binding_file_sha256_raw") != DATA_BINDING_RAW_SHA256 or identity.get("binding_canonical_sha256") != DATA_BINDING_CANONICAL_SHA256:
        raise ValueError("certified DATA binding digest differs from the reviewed identity")
    if identity.get("whole16_certification_sha256") != WHOLE16_CERTIFICATION_SHA256:
        raise ValueError("certified whole-16 qualification digest mismatch")
    if identity.get("qualification_record_sha256") != QUALIFICATION_RECORD_SHA256:
        raise ValueError("certified Fit/Calibration qualification digest mismatch")
    if doc.get("custody", {}).get("manifest_digest_aggregate_sha256") != MANIFEST_AGGREGATE_SHA256:
        raise ValueError("certified role-manifest aggregate digest mismatch")
    if fit_cal.get("fit_healthy_files") != FIT_HEALTHY_ROWS or fit_cal.get("calibration_healthy_files") != CALIBRATION_HEALTHY_ROWS:
        raise ValueError("certified eligible healthy Fit/Calibration row counts changed")
    histories: list[BoundHistory] = []
    seen: set[str] = set()
    role_seen: dict[str, int] = {role: 0 for role in ROLE_COUNTS}
    for row in roster:
        if not isinstance(row, dict):
            raise ValueError("certified DATA roster contains a non-object row")
        role = row.get("assigned_role")
        if role not in ROLE_COUNTS:
            raise ValueError(f"unsupported certified DATA role {role!r}")
        typed_role = cast(RoleName, role)
        role_seen[role] += 1
        ordinal = role_seen[role]
        actual_id = f"S18I-ITER-0005-{role}-{ordinal:02d}"
        schema_id = f"H-S18-{_SCHEMA_ROLE_PREFIX[typed_role]}-{ordinal:02d}"
        if row.get("history_id") != actual_id or row.get("manifest_role_value") != actual_id:
            raise ValueError("certified DATA history order/role identity is not the frozen roster")
        if actual_id in seen:
            raise ValueError("certified DATA history IDs are not unique")
        seen.add(actual_id)
        if row.get("permitted_use") != DATA_PERMISSIONS[typed_role]:
            raise ValueError(f"DATA-stage permission drift for {actual_id}")
        if row.get("role_specific_checks_pass") is not True or row.get("structural_core_integrity_pass") is not True:
            raise ValueError(f"certified DATA history did not pass its recorded role/integrity checks: {actual_id}")
        manifest_sha = row.get("manifest_sha256")
        if not _is_hash(manifest_sha):
            raise ValueError(f"manifest digest is invalid for {actual_id}")
        sample_count = row.get("sample_count")
        data_seed = row.get("data_seed")
        config_hash = row.get("config_hash")
        relative_path = row.get("path_relative")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
            raise ValueError(f"sample count is invalid for {actual_id}")
        if not isinstance(data_seed, int) or isinstance(data_seed, bool) or data_seed < 0:
            raise ValueError(f"data seed is invalid for {actual_id}")
        if not isinstance(config_hash, str) or not config_hash or not isinstance(relative_path, str):
            raise ValueError(f"role path/config binding is invalid for {actual_id}")
        expected_path = f"{DATA_ROOT_RELATIVE}/{role}/{actual_id}"
        if relative_path != expected_path:
            raise ValueError(f"role path does not match the certified DATA root for {actual_id}")
        histories.append(
            BoundHistory(
                assigned_role=typed_role,
                actual_history_id=actual_id,
                schema_history_id=schema_id,
                manifest_sha256=manifest_sha,
                data_seed=data_seed,
                sample_count=sample_count,
                config_hash=config_hash,
                relative_path=relative_path,
                data_permitted_use=DATA_PERMISSIONS[typed_role],
                model_permitted_use=MODEL_PERMISSIONS[typed_role],
            )
        )
    if role_seen != dict(ROLE_COUNTS):
        raise ValueError("certified role counts do not match the accepted v2 topology")
    bound_identity: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "mechanism_spec_sha256": dict(MECHANISM_SPEC_SHA256),
        "handoff_sha256": handoff_sha,
        "data_binding_raw_sha256": DATA_BINDING_RAW_SHA256,
        "data_binding_canonical_sha256": DATA_BINDING_CANONICAL_SHA256,
        "candidate_id": DATA_ROSTER_ID,
        "data_root_relative": DATA_ROOT_RELATIVE,
        "histories": [
            {
                "assigned_role": item.assigned_role,
                "actual_history_id": item.actual_history_id,
                "schema_history_id": item.schema_history_id,
                "manifest_sha256": item.manifest_sha256,
                "data_seed": item.data_seed,
                "sample_count": item.sample_count,
                "config_hash": item.config_hash,
                "relative_path": item.relative_path,
                "data_permission": item.data_permitted_use,
                "model_permission": item.model_permitted_use,
            }
            for item in histories
        ],
        "model_cardinalities": dict(MODEL_CARDINALITIES),
        "patch_geometry": dict(PATCH_GEOMETRY),
    }
    return Task6Binding(
        candidate_id=DATA_ROSTER_ID,
        data_root_relative=DATA_ROOT_RELATIVE,
        handoff_sha256=handoff_sha,
        data_binding_raw_sha256=DATA_BINDING_RAW_SHA256,
        data_binding_canonical_sha256=DATA_BINDING_CANONICAL_SHA256,
        whole16_certification_sha256=WHOLE16_CERTIFICATION_SHA256,
        qualification_record_sha256=QUALIFICATION_RECORD_SHA256,
        manifest_aggregate_sha256=MANIFEST_AGGREGATE_SHA256,
        protocol_sha256=PROTOCOL_SHA256,
        mechanism_spec_sha256=MECHANISM_SPEC_SHA256,
        histories=tuple(histories),
        fit_healthy_rows=FIT_HEALTHY_ROWS,
        calibration_healthy_rows=CALIBRATION_HEALTHY_ROWS,
        executable_binding_sha256=_canonical_sha256(bound_identity),
    )


@dataclass(frozen=True, slots=True)
class CertifiedSample:
    """A single NPZ row verified against a certified role manifest and shard digest."""

    sample: FileSample
    actual_history_id: str
    manifest_sha256: str
    shard_path: str
    shard_sha256: str
    manifest_row: Mapping[str, object]

    @property
    def file_id(self) -> str:
        return self.sample.file_id

    @property
    def is_verified_healthy(self) -> bool:
        health = self.sample.health
        return (
            self.sample.file_label == SampleLabel.NORMAL
            and health is not None
            and health.degradation_stage == DegradationStage.HEALTHY
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_certified_sample_subset(
    binding: Task6Binding,
    actual_history_id: str,
    manifest_path: str | Path,
    shard_paths: Mapping[str, str | Path],
    file_ids: Sequence[str],
) -> tuple[CertifiedSample, ...]:
    """Load a bounded sample subset after verifying its full manifest and selected shard bytes."""
    history = binding.history(actual_history_id)
    if history.assigned_role not in ("FIT", "CALIBRATION"):
        raise ValueError("Task6's model smoke loader is restricted to Fit and Calibration histories")
    requested = tuple(file_ids)
    if not requested or len(set(requested)) != len(requested) or any(not isinstance(x, str) or not x for x in requested):
        raise ValueError("requested source file IDs must be a non-empty unique sequence")
    manifest_file = Path(manifest_path)
    raw = manifest_file.read_bytes()
    if sha256(raw).hexdigest() != history.manifest_sha256:
        raise ValueError(f"source role manifest digest mismatch for {actual_history_id}")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("source role manifest must be a JSON object")
    if manifest.get("format") != 1 or manifest.get("role") != actual_history_id:
        raise ValueError("source manifest format or role identity differs from the certified handoff")
    if manifest.get("protocol") != "sprint15-benchmark-protocol-v7" or manifest.get("config_hash") != history.config_hash:
        raise ValueError("source manifest protocol/config identity differs from the certified handoff")
    files = manifest.get("files")
    shards = manifest.get("shards")
    if not isinstance(files, list) or not isinstance(shards, list):
        raise ValueError("source manifest is missing file or shard rows")
    file_by_id: dict[str, Mapping[str, object]] = {}
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get("file_id"), str):
            raise ValueError("source manifest contains a malformed file row")
        file_by_id[cast(str, row["file_id"])] = row
    shard_by_id: dict[str, Mapping[str, object]] = {}
    for row in shards:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("file_ids"), list):
            raise ValueError("source manifest contains a malformed shard row")
        for file_id in row["file_ids"]:
            if not isinstance(file_id, str) or file_id in shard_by_id:
                raise ValueError("source shard membership is malformed or duplicated")
            shard_by_id[file_id] = row
    if any(file_id not in file_by_id or file_id not in shard_by_id for file_id in requested):
        raise ValueError("requested source rows are not bound by the certified role manifest")
    selected_shards = {cast(str, shard_by_id[file_id]["path"]) for file_id in requested}
    if set(shard_paths) != selected_shards:
        raise ValueError("shard path inputs must exactly cover the requested rows' source shards")
    archive_by_shard: dict[str, zipfile.ZipFile] = {}
    shard_digest_by_path: dict[str, str] = {}
    try:
        for relative in sorted(selected_shards):
            shard_record = next(row for row in shards if row.get("path") == relative)
            expected_digest = shard_record.get("sha256")
            if not _is_hash(expected_digest):
                raise ValueError(f"source shard digest is malformed: {relative}")
            local_path = Path(shard_paths[relative])
            observed_digest = _sha256_file(local_path)
            if observed_digest != expected_digest:
                raise ValueError(f"source shard checksum mismatch for {relative}")
            archive_by_shard[relative] = zipfile.ZipFile(local_path)
            shard_digest_by_path[relative] = observed_digest
        rows: list[CertifiedSample] = []
        for file_id in requested:
            shard_record = shard_by_id[file_id]
            relative = cast(str, shard_record["path"])
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
            if not 0 <= sample.robot_idx < MODEL_CARDINALITIES["n_robots"] or not 0 <= sample.program_idx < MODEL_CARDINALITIES["n_programs"]:
                raise ValueError(f"source conditioning ID is outside the certified model cardinalities: {file_id}")
            rows.append(
                CertifiedSample(
                    sample=sample,
                    actual_history_id=actual_history_id,
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


def validate_role_permission(
    binding: Task6Binding,
    actual_history_id: str,
    action: str,
    *,
    healthy: bool = False,
    confirmation_lock_sha256: str | None = None,
    confirmation_review_sha256: str | None = None,
    main_confirmation_release: bool = False,
) -> BoundHistory:
    """Enforce role-scoped use; Confirmation requires all independent lock/release bindings."""
    history = binding.history(actual_history_id)
    allowed = _ROLE_ACTIONS.get(action)
    if allowed is None or history.assigned_role not in allowed:
        raise ValueError(f"role {history.assigned_role} is not authorized for action {action!r}")
    if action in {"train_fit", "fit_bank", "fit_standardizer", "fit_auxiliary", "cross_fit", "select_coefficients", "calibrate_q95"} and not healthy:
        raise ValueError(f"action {action!r} requires verified-healthy support")
    if action == "score_confirmation":
        if not main_confirmation_release or not _is_hash(confirmation_lock_sha256) or not _is_hash(confirmation_review_sha256):
            raise ValueError("Confirmation is closed until Task33 lock, independent review, and Main release are all bound")
    return history


def is_verified_healthy(row: CertifiedSample) -> bool:
    """Use only the frozen normal-label plus latent-healthy eligibility predicate."""
    return row.is_verified_healthy


@dataclass(frozen=True, slots=True)
class CommonSupport:
    """Canonical fixed support used without per-arm/branch intersections."""

    actual_history_id: str
    schema_history_id: str
    role_manifest_sha256: str
    file_ids: tuple[str, ...]
    support_sha256: str


def make_common_support(
    binding: Task6Binding,
    actual_history_id: str,
    file_ids: Sequence[str],
) -> CommonSupport:
    history = binding.history(actual_history_id)
    ids = tuple(file_ids)
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("common support must be a non-empty unique sequence of file IDs")
    ordered = tuple(sorted(ids))
    support_digest = _canonical_sha256(
        {
            "binding_sha256": binding.executable_binding_sha256,
            "actual_history_id": history.actual_history_id,
            "schema_history_id": history.schema_history_id,
            "role_manifest_sha256": history.manifest_sha256,
            "file_ids": ordered,
        }
    )
    return CommonSupport(
        actual_history_id=history.actual_history_id,
        schema_history_id=history.schema_history_id,
        role_manifest_sha256=history.manifest_sha256,
        file_ids=ordered,
        support_sha256=support_digest,
    )


def validate_score_maps(
    support: CommonSupport,
    scores: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    """Require exactly two distinct finite score branches on the exact declared support."""
    if set(scores) != {"S_pred", "S_pop"}:
        raise ValueError("score output must contain exactly independent S_pred and S_pop branches")
    predicted = scores["S_pred"]
    population = scores["S_pop"]
    if predicted is population:
        raise ValueError("S_pred and S_pop score maps must not alias")
    for branch, values in (("S_pred", predicted), ("S_pop", population)):
        if not isinstance(values, Mapping) or set(values) != set(support.file_ids):
            raise ValueError(f"{branch} support differs from the fixed common support")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"{branch} contains missing/nonfinite score values")
    return scores


def validate_paired_common_support(
    support: CommonSupport,
    arm_scores: Mapping[str, Mapping[str, float]],
    baseline_scores: Mapping[str, Mapping[str, float]],
    *,
    branch: ScoreBranchName,
) -> tuple[float, ...]:
    """Return ordered arm-minus-B0 deltas only after exact branch/support identity checks."""
    validate_score_maps(support, arm_scores)
    validate_score_maps(support, baseline_scores)
    if arm_scores[branch] is baseline_scores[branch]:
        raise ValueError("paired arm and B0 scores must be separate output maps")
    return tuple(float(arm_scores[branch][key]) - float(baseline_scores[branch][key]) for key in support.file_ids)


def deterministic_mask_seed(model_seed: int, actual_history_id: str, file_id: str) -> int:
    """Derive a reproducible per-file inference mask seed without outcome inputs."""
    if model_seed not in MODEL_SEEDS or not actual_history_id or not file_id:
        raise ValueError("mask seed inputs must be bound model/history/file identities")
    key = f"sprint18-score-mask-v2|{model_seed}|{actual_history_id}|{file_id}".encode("utf-8")
    return int.from_bytes(sha256(key).digest()[:8], "big") % (2**63 - 1)



def deterministic_sampler_order(
    binding: Task6Binding,
    arm: Sprint18ArmConfig,
    *,
    workload: Literal["representation", "n4_scale"],
    scope: Literal["experiment", "task6_smoke_only"],
    role_history_ids: Sequence[str],
    epoch: int,
    eligible_source_ids: Sequence[str],
) -> tuple[int, tuple[str, ...]]:
    """Derive one reproducible epoch permutation from the bound Fit source identity."""
    validate_arm_config(arm, require_locked=(scope == "experiment"))
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError("sampler epoch must be a positive integer")
    source_ids = tuple(eligible_source_ids)
    if not source_ids or any(not isinstance(item, str) or not item for item in source_ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError("sampler source IDs must be non-empty and unique")
    canonical_ids = tuple(sorted(source_ids))
    history_ids = tuple(role_history_ids)
    if not history_ids or len(set(history_ids)) != len(history_ids):
        raise ValueError("sampler Fit history IDs must be non-empty and unique")
    fit_histories = binding.role_histories("FIT")
    expected_fit_ids = tuple(item.actual_history_id for item in fit_histories)
    if scope == "task6_smoke_only":
        if workload != "representation" or arm.arm_id == "N4-CRN" or len(history_ids) != 1:
            raise ValueError("Task6 smoke sampler is restricted to one Fit history and representation workload")
    elif scope != "experiment":
        raise ValueError("sampler scope is unsupported")
    n4_fit_scope: str | None = None
    if workload == "n4_scale":
        if scope != "experiment" or arm.arm_id != "N4-CRN":
            raise ValueError("N4 sampler must use its experiment scale workload")
        if history_ids == expected_fit_ids:
            n4_fit_scope = "final"
        elif (
            len(history_ids) == 2
            and history_ids == tuple(sorted(history_ids))
            and set(history_ids).issubset(expected_fit_ids)
        ):
            heldout_history_id = next(
                history_id for history_id in expected_fit_ids if history_id not in history_ids
            )
            n4_fit_scope = f"fold:{heldout_history_id}"
        else:
            raise ValueError("N4 sampler must bind one canonical two-history fold or the final all-Fit scope")
    elif workload == "representation":
        if arm.arm_id == "N4-CRN":
            raise ValueError("N4 cannot use the representation sampler")
        if scope == "experiment" and history_ids != expected_fit_ids:
            raise ValueError("experiment representation sampler must bind all three Fit histories")
        if scope == "task6_smoke_only" and (
            len(history_ids) != 1 or history_ids[0] not in expected_fit_ids
        ):
            raise ValueError("Task6 smoke sampler history is outside certified Fit")
    else:
        raise ValueError("sampler workload is unsupported")
    history_map = {item.actual_history_id: item for item in fit_histories}
    if any(history_id not in history_map for history_id in history_ids):
        raise ValueError("sampler source includes a history outside certified Fit")
    if scope == "task6_smoke_only":
        role_digests = [history_map[history_id].manifest_sha256 for history_id in history_ids]
        seed_digest = _canonical_sha256(
            {
                "sampler_id": "sprint18-canonical-fit-permutation-v2",
                "binding_sha256": binding.executable_binding_sha256,
                "arm_seed": arm.model_seed,
                "workload": workload,
                "history_ids": list(history_ids),
                "role_manifest_sha256": role_digests,
                "epoch": epoch,
                "eligible_source_ids_sha256": _canonical_sha256(list(canonical_ids)),
            }
        )
        seed = int(seed_digest[:16], 16) % (2**63 - 1)
    elif workload == "representation":
        fit_role_sha256 = _canonical_sha256(
            sorted(item.manifest_sha256 for item in fit_histories)
        )
        seed_text = (
            f"sprint18-fit-order-v1|{arm.model_seed}|{epoch}|{fit_role_sha256}"
        )
        seed = int.from_bytes(sha256(seed_text.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)
    else:
        if n4_fit_scope is None:
            raise ValueError("N4 sampler scope is missing")
        seed_text = (
            f"sprint18-n4-scale-order-v2|{arm.model_seed}|{n4_fit_scope}|"
            f"{','.join(sorted(history_ids))}|{epoch}"
        )
        seed = int.from_bytes(sha256(seed_text.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = tuple(canonical_ids[index] for index in torch.randperm(len(canonical_ids), generator=generator).tolist())
    return seed, order

def _encoder_batch(batch: Mapping[str, object]) -> dict[str, object]:
    """Drop all diagnostic-only collator fields before any model call."""
    output = {key: value for key, value in batch.items() if key in _MODEL_BATCH_KEYS}
    if set(output) != _MODEL_BATCH_KEYS:
        raise ValueError("collated batch is missing required encoder-visible fields")
    if any(key in output for key in ("file_labels", "anomaly_meta", "anomaly_masks", "file_samples")):
        raise ValueError("diagnostic labels/metadata cannot enter the model batch")
    return output


@dataclass(frozen=True, slots=True)
class BankProvenance:
    """Fit-only S_pop reference bank plus exact checkpoint/support identity."""

    arm_id: ArmId
    model_seed: int
    source_role: str
    source_history_ids: tuple[str, ...]
    source_manifest_sha256: tuple[str, ...]
    source_rows: int
    source_rows_sha256: str
    data_binding_sha256: str
    checkpoint_sha256: str
    bank_sha256: str
    scope: Literal["experiment", "task6_smoke_only"]


@dataclass(frozen=True, slots=True)
class ProvenancedBank:
    bank: NormalReferenceBank
    provenance: BankProvenance


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def _assert_restored_checkpoint(
    model: V1RepresentationModel,
    arm: Sprint18ArmConfig,
    binding: Task6Binding,
    checkpoint: RestoredSprint18Checkpoint,
    *,
    scope: Literal["experiment", "task6_smoke_only"],
) -> str:
    from representation.sprint18_checkpoint import RestoredSprint18Checkpoint as CheckpointIdentity
    from representation.sprint18_checkpoint import state_sha256

    if not isinstance(checkpoint, CheckpointIdentity):
        raise ValueError("inference and bank fitting require a validated restored checkpoint identity")
    if (
        checkpoint.scope != scope
        or checkpoint.arm_id != arm.arm_id
        or checkpoint.model_seed != arm.model_seed
        or checkpoint.data_binding_sha256 != binding.executable_binding_sha256
        or checkpoint.arm_config_sha256 != arm_config_sha256(arm)
        or checkpoint.model_config_sha256 != _canonical_sha256(model.config.to_dict())
        or checkpoint.model_state_sha256 != state_sha256(model.state_dict())
        or checkpoint.ema_state_sha256 != state_sha256(model.target_encoder.state_dict())
        or not _is_hash(checkpoint.checkpoint_sha256)
    ):
        raise ValueError("restored checkpoint identity/model bytes do not match this harness operation")
    expected_kind = "task6_smoke_fixture" if scope == "task6_smoke_only" else "primary_or_diagnostic"
    if checkpoint.checkpoint_kind != expected_kind:
        raise ValueError("restored checkpoint kind differs from the requested scope")
    return checkpoint.checkpoint_sha256



@torch.no_grad()
def fit_fit_reference_bank(
    binding: Task6Binding,
    arm: Sprint18ArmConfig,
    model: V1RepresentationModel,
    rows: Sequence[CertifiedSample],
    patchifier: Patchifier,
    *,
    checkpoint: RestoredSprint18Checkpoint,
    scope: Literal["experiment", "task6_smoke_only"],
) -> ProvenancedBank:
    """Fit an arm/seed bank from eligible healthy Fit embeddings only."""
    validate_arm_config(arm, require_locked=(scope == "experiment"))
    if arm.arm_id != "B0-FULL":
        raise ValueError("only B0-FULL inference is implemented in the Task6 common harness")
    checkpoint_sha256 = _assert_restored_checkpoint(model, arm, binding, checkpoint, scope=scope)
    if not rows:
        raise ValueError("Fit bank source cannot be empty")
    histories = {row.actual_history_id for row in rows}
    if not histories.issubset({item.actual_history_id for item in binding.role_histories("FIT")}):
        raise ValueError("reference bank rows include a non-Fit history")
    for row in rows:
        validate_role_permission(binding, row.actual_history_id, "fit_bank", healthy=row.is_verified_healthy)
        if not row.is_verified_healthy or row.manifest_sha256 != binding.history(row.actual_history_id).manifest_sha256:
            raise ValueError("Fit bank source contains an ineligible or unbound row")
    file_ids = [row.file_id for row in rows]
    if len(set(file_ids)) != len(file_ids):
        raise ValueError("Fit bank source contains duplicate file IDs")
    if scope == "experiment":
        expected_histories = {item.actual_history_id for item in binding.role_histories("FIT")}
        if histories != expected_histories or len(rows) != binding.fit_healthy_rows:
            raise ValueError("experiment bank requires the complete frozen three-history Fit-healthy support")
    model_config = model.config
    if (model_config.n_robots, model_config.n_programs) != (9, 8):
        raise ValueError("bank model conditioning cardinalities differ from the certified roster")
    model.eval()
    probe_inference = RepresentationInference(model, NormalReferenceBank(k=model_config.knn_k), patchifier)
    embeddings: list[torch.Tensor] = []
    for row in rows:
        seed = deterministic_mask_seed(arm.model_seed, row.actual_history_id, row.file_id)
        batch = collate_variable_files(
            [row.sample], patchifier, masking_config=model_config, masking_seed=seed
        )
        device_batch = cast(dict[str, object], probe_inference.co_locate_batch(cast(dict[str, object], _encoder_batch(batch))))
        output = model(device_batch)
        vector = output["file_embedding"]
        if vector.shape[0] != 1 or not torch.isfinite(vector).all():
            raise ValueError("Fit embedding is missing or nonfinite")
        embeddings.append(vector.detach().cpu())
    matrix = torch.cat(embeddings, dim=0)
    bank = NormalReferenceBank(k=model_config.knn_k).fit(matrix, [SampleLabel.NORMAL] * len(rows))
    ordered_rows = sorted(rows, key=lambda row: (row.actual_history_id, row.file_id))
    rows_sha = _canonical_sha256(
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
    source_manifests = tuple(sorted({row.manifest_sha256 for row in rows}))
    provenance = BankProvenance(
        arm_id=arm.arm_id,
        model_seed=arm.model_seed,
        source_role="FIT",
        source_history_ids=tuple(sorted(histories)),
        source_manifest_sha256=source_manifests,
        source_rows=len(rows),
        source_rows_sha256=rows_sha,
        data_binding_sha256=binding.executable_binding_sha256,
        checkpoint_sha256=checkpoint_sha256,
        bank_sha256=_tensor_digest(cast(torch.Tensor, bank.embeddings)),
        scope=scope,
    )
    return ProvenancedBank(bank=bank, provenance=provenance)


@dataclass(frozen=True, slots=True)
class HarnessScores:
    """Two distinct branch maps plus phase support and provenance."""

    arm_id: ArmId
    model_seed: int
    actual_history_id: str
    schema_history_id: str
    role_manifest_sha256: str
    support: CommonSupport
    scores: Mapping[str, Mapping[str, float]]
    checkpoint_sha256: str
    bank_sha256: str
    data_binding_sha256: str
    branch_provenance_sha256: Mapping[str, str]
    scope: Literal["experiment", "task6_smoke_only"]


def score_certified_rows(
    binding: Task6Binding,
    arm: Sprint18ArmConfig,
    model: V1RepresentationModel,
    provenanced_bank: ProvenancedBank,
    rows: Sequence[CertifiedSample],
    patchifier: Patchifier,
    *,
    checkpoint: RestoredSprint18Checkpoint,
    action: Literal["calibrate_q95", "score_development", "score_confirmation"],
    scope: Literal["experiment", "task6_smoke_only"],
    confirmation_lock_sha256: str | None = None,
    confirmation_review_sha256: str | None = None,
    main_confirmation_release: bool = False,
) -> HarnessScores:
    """Score every declared row through the public inference pattern without branch fusion."""
    validate_arm_config(arm, require_locked=(scope == "experiment"))
    if arm.arm_id != "B0-FULL":
        raise ValueError("only B0-FULL inference is implemented in the Task6 common harness")
    checkpoint_sha256 = _assert_restored_checkpoint(model, arm, binding, checkpoint, scope=scope)
    if scope == "task6_smoke_only" and action != "calibrate_q95":
        raise ValueError("Task6 smoke scoring is restricted to Calibration-only proof output")
    if not rows:
        raise ValueError("scoring support cannot be empty")
    if provenanced_bank.provenance.arm_id != arm.arm_id or provenanced_bank.provenance.model_seed != arm.model_seed:
        raise ValueError("bank arm/seed does not match the scoring configuration")
    if provenanced_bank.provenance.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("bank was fitted from a different checkpoint")
    if provenanced_bank.provenance.scope != scope or provenanced_bank.provenance.source_role != "FIT":
        raise ValueError("bank role/scope provenance differs from the requested scoring operation")
    bank_embeddings = provenanced_bank.bank.embeddings
    source_histories = provenanced_bank.provenance.source_history_ids
    fit_histories = {item.actual_history_id for item in binding.role_histories("FIT")}
    expected_manifests = tuple(sorted(binding.history(history_id).manifest_sha256 for history_id in source_histories if history_id in fit_histories))
    if (
        not isinstance(bank_embeddings, torch.Tensor)
        or bank_embeddings.ndim != 2
        or bank_embeddings.shape[0] != provenanced_bank.provenance.source_rows
        or provenanced_bank.bank.k != model.config.knn_k
        or not torch.isfinite(bank_embeddings).all()
        or not source_histories
        or tuple(sorted(source_histories)) != source_histories
        or not set(source_histories).issubset(fit_histories)
        or provenanced_bank.provenance.source_manifest_sha256 != expected_manifests
        or provenanced_bank.provenance.source_rows < 1
        or not _is_hash(provenanced_bank.provenance.source_rows_sha256)
        or not _is_hash(provenanced_bank.provenance.bank_sha256)
        or provenanced_bank.provenance.bank_sha256 != _tensor_digest(bank_embeddings)
    ):
        raise ValueError("Fit bank bytes/row provenance is incomplete or inconsistent")
    if provenanced_bank.provenance.data_binding_sha256 != binding.executable_binding_sha256:
        raise ValueError("bank uses a different executable DATA binding")
    history_ids = {row.actual_history_id for row in rows}
    if len(history_ids) != 1:
        raise ValueError("one scoring support cannot pool multiple histories")
    actual_history_id = next(iter(history_ids))
    history = binding.history(actual_history_id)
    healthy_only = action == "calibrate_q95"
    validate_role_permission(
        binding,
        actual_history_id,
        action,
        healthy=all(row.is_verified_healthy for row in rows) if healthy_only else False,
        confirmation_lock_sha256=confirmation_lock_sha256,
        confirmation_review_sha256=confirmation_review_sha256,
        main_confirmation_release=main_confirmation_release,
    )
    if any(row.manifest_sha256 != history.manifest_sha256 for row in rows):
        raise ValueError("scoring support contains rows from a different manifest")
    if healthy_only and any(not row.is_verified_healthy for row in rows):
        raise ValueError("Calibration support must be verified healthy")
    if healthy_only and scope == "experiment" and len(rows) != binding.calibration_healthy_rows:
        raise ValueError("experiment q95 scoring requires the complete frozen Calibration-healthy support")
    ids = [row.file_id for row in rows]
    support = make_common_support(binding, actual_history_id, ids)
    model_config = model.config
    if (model_config.n_robots, model_config.n_programs, model_config.n_channels, model_config.patch_size, model_config.stride) != (9, 8, 6, 32, 16):
        raise ValueError("inference model config is incompatible with certified DATA conditioning/geometry")
    model.eval()
    inference = RepresentationInference(
        model,
        provenanced_bank.bank,
        patchifier,
        masking_config=model_config,
    )
    pred_scores: dict[str, float] = {}
    pop_scores: dict[str, float] = {}
    for row in rows:
        seed = deterministic_mask_seed(arm.model_seed, actual_history_id, row.file_id)
        batch = collate_variable_files(
            [row.sample], patchifier, masking_config=model_config, masking_seed=seed
        )
        clean_batch = cast(dict[str, object], inference.co_locate_batch(cast(dict[str, object], _encoder_batch(batch))))
        scored = inference.score_batch(clean_batch)  # type: ignore[arg-type]
        pred = scored.get("S_pred")
        pop = scored.get("S_pop")
        if not isinstance(pred, torch.Tensor) or not isinstance(pop, torch.Tensor):
            raise ValueError("public inference did not return both independent score tensors")
        if pred.shape != (1,) or pop.shape != (1,) or not torch.isfinite(pred).all() or not torch.isfinite(pop).all():
            raise ValueError("public inference returned missing/nonfinite branch scores")
        pred_scores[row.file_id] = float(pred[0].item())
        pop_scores[row.file_id] = float(pop[0].item())
    branches: dict[str, Mapping[str, float]] = {
        "S_pred": MappingProxyType(pred_scores),
        "S_pop": MappingProxyType(pop_scores),
    }
    validate_score_maps(support, branches)
    branch_prov = MappingProxyType(
        {
            name: _canonical_sha256(
                {
                    "branch": name,
                    "score_map": dict(branches[name]),
                    "support_sha256": support.support_sha256,
                    "checkpoint_sha256": checkpoint_sha256,
                    "bank_sha256": provenanced_bank.provenance.bank_sha256,
                    "role_manifest_sha256": history.manifest_sha256,
                }
            )
            for name in ("S_pred", "S_pop")
        }
    )
    if branch_prov["S_pred"] == branch_prov["S_pop"]:
        raise ValueError("branch provenance digests unexpectedly alias")
    return HarnessScores(
        arm_id=arm.arm_id,
        model_seed=arm.model_seed,
        actual_history_id=actual_history_id,
        schema_history_id=history.schema_history_id,
        role_manifest_sha256=history.manifest_sha256,
        support=support,
        scores=MappingProxyType(branches),
        checkpoint_sha256=checkpoint_sha256,
        bank_sha256=provenanced_bank.provenance.bank_sha256,
        data_binding_sha256=binding.executable_binding_sha256,
        branch_provenance_sha256=branch_prov,
        scope=scope,
    )


@dataclass(frozen=True, slots=True)
class BranchCalibration:
    branch: ScoreBranchName
    quantile: float
    threshold: float
    source_history_id: str
    source_role_manifest_sha256: str
    source_support_sha256: str
    source_rows: int
    source_rows_sha256: str
    score_provenance_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class ProvenancedCalibration:
    arm_id: ArmId
    model_seed: int
    source_role: str
    data_binding_sha256: str
    scope: Literal["experiment", "task6_smoke_only"]
    S_pred: BranchCalibration
    S_pop: BranchCalibration


def fit_calibration_q95(
    binding: Task6Binding,
    scores: HarnessScores,
    rows: Sequence[CertifiedSample],
    *,
    scope: Literal["experiment", "task6_smoke_only"],
) -> ProvenancedCalibration:
    """Calibrate separate q95 thresholds using only verified-healthy Calibration scores."""
    if scores.actual_history_id not in {item.actual_history_id for item in binding.role_histories("CALIBRATION")}:
        raise ValueError("calibration scores do not come from the certified Calibration role")
    if scores.scope != scope or scores.data_binding_sha256 != binding.executable_binding_sha256:
        raise ValueError("calibration scope/data binding differs from scored support")
    history = binding.history(scores.actual_history_id)
    if (
        scores.arm_id != "B0-FULL"
        or scores.model_seed not in MODEL_SEEDS
        or (scope == "task6_smoke_only" and scores.arm_id != "B0-FULL")
        or scores.support.actual_history_id != history.actual_history_id
        or scores.support.schema_history_id != history.schema_history_id
        or scores.support.role_manifest_sha256 != history.manifest_sha256
        or scores.role_manifest_sha256 != history.manifest_sha256
        or not _is_hash(scores.support.support_sha256)
        or not _is_hash(scores.checkpoint_sha256)
        or not _is_hash(scores.bank_sha256)
        or scores.branch_provenance_sha256.keys() != {"S_pred", "S_pop"}
        or any(not _is_hash(value) for value in scores.branch_provenance_sha256.values())
        or scores.branch_provenance_sha256["S_pred"] == scores.branch_provenance_sha256["S_pop"]
    ):
        raise ValueError("calibration score/checkpoint/branch provenance is incomplete or inconsistent")
    if any(
        row.actual_history_id != history.actual_history_id
        or row.manifest_sha256 != history.manifest_sha256
        or not _is_hash(row.shard_sha256)
        for row in rows
    ):
        raise ValueError("Calibration rows are not bound to their exact certified history/shards")
    if len(rows) != len(scores.support.file_ids) or {row.file_id for row in rows} != set(scores.support.file_ids):
        raise ValueError("calibration rows differ from the exact scored support")
    if any(row.actual_history_id != scores.actual_history_id or not row.is_verified_healthy for row in rows):
        raise ValueError("Calibration may use only verified-healthy rows from its bound history")
    if scope == "experiment" and len(rows) != binding.calibration_healthy_rows:
        raise ValueError("experiment q95 requires the complete frozen Calibration-healthy support")
    validate_score_maps(scores.support, scores.scores)
    ordered_rows_sha = _canonical_sha256(
        [
            {"history_id": row.actual_history_id, "manifest_sha256": row.manifest_sha256,
             "file_id": row.file_id, "shard_sha256": row.shard_sha256}
            for row in sorted(rows, key=lambda item: item.file_id)
        ]
    )
    branches: dict[str, BranchCalibration] = {}
    for name in ("S_pred", "S_pop"):
        values = np.asarray([scores.scores[name][file_id] for file_id in scores.support.file_ids], dtype=np.float64)
        threshold = float(np.quantile(values, 0.95, method="linear"))
        if not math.isfinite(threshold):
            raise ValueError(f"{name} calibration quantile is nonfinite")
        branches[name] = BranchCalibration(
            branch=cast(ScoreBranchName, name),
            quantile=0.95,
            threshold=threshold,
            source_history_id=history.actual_history_id,
            source_role_manifest_sha256=history.manifest_sha256,
            source_support_sha256=scores.support.support_sha256,
            source_rows=len(rows),
            source_rows_sha256=ordered_rows_sha,
            score_provenance_sha256=scores.branch_provenance_sha256[name],
            checkpoint_sha256=scores.checkpoint_sha256,
        )
    return ProvenancedCalibration(
        arm_id=scores.arm_id,
        model_seed=scores.model_seed,
        source_role="CALIBRATION",
        data_binding_sha256=binding.executable_binding_sha256,
        scope=scope,
        S_pred=branches["S_pred"],
        S_pop=branches["S_pop"],
    )


def count_unique_parameters(modules: Sequence[nn.Module]) -> int:
    """Count each distinct deployed parameter object once, including frozen EMA parameters."""
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                total += parameter.numel()
    if total <= 0:
        raise ValueError("deployed inference path has no countable parameters")
    return total


def primary_inference_parameters(
    model: V1RepresentationModel,
    branch: ScoreBranchName,
    *,
    added_scorers: Sequence[nn.Module] = (),
) -> int:
    """Count the exact existing B0 path for one primary score branch."""
    common: list[nn.Module] = [model.patch_encoder, model.context_encoder]
    if model.conditional_norm is not None:
        common.append(model.conditional_norm)
    if branch == "S_pred":
        common.extend((model.target_encoder, model.predictor))
    elif branch != "S_pop":
        raise ValueError(f"unknown primary branch {branch!r}")
    common.extend(added_scorers)
    return count_unique_parameters(common)


def profile_inference_flops(operation: Callable[[], object], *, device: str = "cpu") -> int:
    """Measure one public inference operation with the protocol's PyTorch FLOP counter."""
    if device not in ("cpu", "cuda"):
        raise ValueError("FLOP profiling device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA FLOP profiling requested but CUDA is unavailable")
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, with_flops=True, record_shapes=True) as profiler:
        operation()
    count = sum(int(event.flops or 0) for event in profiler.key_averages())
    if count <= 0:
        raise ValueError("PyTorch profiler produced no countable FLOPs; result cannot be declared VALID")
    return count


def _ceil_ratio(value: int, numerator: int, denominator: int) -> int:
    """Compute an exact positive rational ceiling without float-boundary drift."""
    return (value * numerator + denominator - 1) // denominator
def validate_compute_envelope(
    arm_id: str,
    *,
    parameters_by_branch: Mapping[str, int],
    flops_by_branch: Mapping[str, int],
    b0_parameters_by_branch: Mapping[str, int] | None,
    b0_flops_by_branch: Mapping[str, int] | None,
    training_flops_ratio: float | None,
    n4_scale_parameters: int | None = None,
    n4_s_pop_bitwise_equal_to_b0: bool | None = None,
) -> Mapping[str, object]:
    """Return a VALID compute observation only when all arm-specific frozen counters exist and pass."""
    spec = arm_spec(arm_id)
    required_branches = {"S_pred", "S_pop"}
    if set(parameters_by_branch) != required_branches or set(flops_by_branch) != required_branches:
        raise ValueError("parameter and FLOP observations must separately cover S_pred and S_pop")
    for name, values in (("parameters", parameters_by_branch), ("FLOPs", flops_by_branch)):
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError(f"{name} observations must be positive integers")
    if (
        not isinstance(training_flops_ratio, (int, float))
        or isinstance(training_flops_ratio, bool)
        or not math.isfinite(float(training_flops_ratio))
        or training_flops_ratio <= 0
    ):
        raise ValueError("training FLOPs ratio is missing/nonfinite; arm cannot be VALID")
    if arm_id == "B0-FULL":
        return MappingProxyType({
            "parameters_inference": dict(parameters_by_branch),
            "inference_flops_per_fit_file": dict(flops_by_branch),
            "parameter_ratio_vs_B0": {name: 1.0 for name in required_branches},
            "inference_flop_ratio_vs_B0": {name: 1.0 for name in required_branches},
            "training_flops_ratio_vs_B0": training_flops_ratio,
            "combined_inference_flops_ratio_vs_B0": 1.0,
            "envelope_pass": True,
        })
    if b0_parameters_by_branch is None or b0_flops_by_branch is None:
        raise ValueError("non-B0 compute validation requires actual paired B0 counters")
    if set(b0_parameters_by_branch) != required_branches or set(b0_flops_by_branch) != required_branches:
        raise ValueError("B0 reference counters must separately cover both score branches")
    for name, values in (("paired B0 parameters", b0_parameters_by_branch), ("paired B0 FLOPs", b0_flops_by_branch)):
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError(f"{name} must contain positive integer measurements")
    parameter_ratios = {name: parameters_by_branch[name] / b0_parameters_by_branch[name] for name in required_branches}
    flop_ratios = {name: flops_by_branch[name] / b0_flops_by_branch[name] for name in required_branches}
    combined_flops_ratio: float | None = None
    passed = True
    if spec.family == "legacy_stack":
        limit = 1.25
        passed = all(parameters_by_branch[name] <= _ceil_ratio(b0_parameters_by_branch[name], 110, 100) for name in required_branches)
        passed = passed and all(flops_by_branch[name] <= _ceil_ratio(b0_flops_by_branch[name], 110, 100) for name in required_branches)
    elif arm_id == "N1-HCC":
        limit = 1.25
        passed = parameters_by_branch == dict(b0_parameters_by_branch) and flops_by_branch == dict(b0_flops_by_branch)
    elif arm_id == "N2-MSP":
        limit = 1.25
        passed = all(parameters_by_branch[name] <= _ceil_ratio(b0_parameters_by_branch[name], 110, 100) for name in required_branches)
        passed = passed and all(flops_by_branch[name] <= _ceil_ratio(b0_flops_by_branch[name], 110, 100) for name in required_branches)
    elif arm_id == "N3-CCRP":
        limit = 2.25
        passed = all(parameters_by_branch[name] <= _ceil_ratio(b0_parameters_by_branch[name], 110, 100) for name in required_branches)
        passed = passed and all(flops_by_branch[name] <= _ceil_ratio(b0_flops_by_branch[name], 110, 100) for name in required_branches)
    else:
        limit = 0.25
        if not isinstance(n4_scale_parameters, int) or isinstance(n4_scale_parameters, bool) or n4_scale_parameters < 0:
            raise ValueError("N4 scale parameter count is required")
        if n4_s_pop_bitwise_equal_to_b0 is not True:
            raise ValueError("N4 S_pop must be bitwise identical to B0 on the same cached inputs")
        passed = n4_scale_parameters <= 100_000
        passed = passed and parameters_by_branch["S_pop"] == b0_parameters_by_branch["S_pop"]
        passed = passed and parameters_by_branch["S_pred"] <= b0_parameters_by_branch["S_pred"] + n4_scale_parameters
        combined_flops = flops_by_branch["S_pred"] + flops_by_branch["S_pop"]
        b0_combined_flops = b0_flops_by_branch["S_pred"] + b0_flops_by_branch["S_pop"]
        combined_flops_ratio = combined_flops / b0_combined_flops
        passed = passed and combined_flops <= _ceil_ratio(b0_combined_flops, 110, 100)
    passed = passed and training_flops_ratio <= limit
    if not passed:
        raise ValueError(f"{arm_id} violates its frozen Sprint18 parameter/FLOP envelope")
    return MappingProxyType(
        {
            "parameters_inference": dict(parameters_by_branch),
            "inference_flops_per_fit_file": dict(flops_by_branch),
            "parameter_ratio_vs_B0": parameter_ratios,
            "inference_flop_ratio_vs_B0": flop_ratios,
            "training_flops_ratio_vs_B0": training_flops_ratio,
            "training_flops_limit_vs_B0": limit,
            "combined_inference_flops_ratio_vs_B0": combined_flops_ratio,
            "envelope_pass": True,
        }
    )


# The v2 result JSON Schema is the exact v1 embedded schema with only the four
# identity substitutions specified in the accepted v2 amendment. This manual
# validator enforces its closed object/key/type/enum/array constraints without
# adding a runtime dependency or accepting unknown fields.
_TOP_KEYS = {
    "schema_id", "protocol_id", "phase", "arm_id", "status", "failure_reasons",
    "provenance", "configuration", "seed_runs", "support", "scores", "metrics", "branch_gates",
}
_PROVENANCE_KEYS = {
    "protocol_sha256", "role_binding_sha256", "fit_role_sha256", "calibration_role_sha256",
    "evaluation_role_sha256", "config_sha256", "coefficient_lock_sha256", "metric_code_sha256",
    "compute_counter_sha256", "mechanism_spec_sha256",
}
_CONFIG_KEYS = {
    "model_seeds", "optimizer", "batch_size", "batch_size_unit", "learning_rate", "weight_decay",
    "clip_norm", "precision", "representation_epochs", "scale_epochs", "ema_decay",
    "lambda_info_max", "lambda_hcc_max", "alpha", "beta", "primary_checkpoint_epoch",
}
_SEED_KEYS = {
    "model_seed", "representation_epochs_completed", "scale_epochs_completed", "optimizer_steps",
    "primary_checkpoint_sha256", "ema_state_sha256", "optimizer_scheduler_rng_sha256",
    "fit_bank_sha256", "calibration_sha256", "coefficient_trace_sha256", "compute",
}
_COMPUTE_KEYS = {
    "parameters_inference", "inference_flops_per_fit_file", "training_flops_per_fit_file",
    "parameter_ratio_vs_B0", "inference_flop_ratio_vs_B0", "training_flop_ratio_vs_B0", "envelope_pass",
}
_SUPPORT_KEYS = {"common_support_sha256", "evaluation_histories", "per_history", "excluded_rows"}
_HISTORY_SUPPORT_KEYS = {"history_id", "p", "w", "a", "controls", "robot_days"}
_BRANCH_KEYS = {
    "calibration_quantile", "thresholds_by_seed", "calibration_role_sha256",
    "score_provenance_sha256", "independent",
}
_METRIC_KEYS = {
    "history_id", "P_AUROC", "W_AUROC", "PW_AUROC", "A_AUROC", "delta_P", "delta_W", "delta_PW",
    "P_recall", "P_median_lead_days", "W_recall", "W_median_lead_days", "FAR_per_robot_day", "delta_FAR",
    "background_stability_ratio", "localization", "severity_ordering", "subtype_slices",
}
_AGGREGATE_KEYS = {
    "P_macro", "W_macro", "PW_macro", "PW_lcb95", "delta_P_macro", "delta_W_macro", "delta_PW_macro",
    "directional_count", "positive_delta_history_count", "background_uncertainty",
}
_GATE_KEYS = {
    "recovery_applied", "minimum_effect", "positive_history_replication", "lcb", "directional",
    "P_noninferiority", "W_noninferiority", "P_recall_lead", "W_recall_lead", "FAR",
    "background_stability", "support_and_provenance", "compute", "score_separation", "recovery",
}
_RECOVERY_GATE_FIELDS = (
    "minimum_effect", "positive_history_replication", "lcb", "directional", "P_noninferiority",
    "W_noninferiority", "P_recall_lead", "W_recall_lead", "FAR", "background_stability", "recovery",
)


def _keys(value: object, expected: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly the closed v2 schema keys")
    return cast(Mapping[str, object], value)


def _int(value: object, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{label} is outside its integer schema")
    return value


def _number(value: object, label: str, *, minimum: float | None = None, exclusive_minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} is below its schema minimum")
    if exclusive_minimum is not None and result <= exclusive_minimum:
        raise ValueError(f"{label} is below its schema exclusive minimum")
    return result


def _nullable_number(value: object, label: str) -> None:
    if value is not None:
        _number(value, label)


def _nullable_hash(value: object, label: str) -> None:
    if value is not None and not _is_hash(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest or null")


def _validate_compute_record(value: object, label: str) -> None:
    row = _keys(value, _COMPUTE_KEYS, label)
    params = row["parameters_inference"]
    if params is not None:
        _int(params, f"{label}.parameters_inference")
    for name in (
        "inference_flops_per_fit_file", "training_flops_per_fit_file", "parameter_ratio_vs_B0",
        "inference_flop_ratio_vs_B0", "training_flop_ratio_vs_B0",
    ):
        _nullable_number(row[name], f"{label}.{name}")
    if row["envelope_pass"] is not None and not isinstance(row["envelope_pass"], bool):
        raise ValueError(f"{label}.envelope_pass must be boolean or null")


def _validate_history_metric(value: object, label: str, phase: str) -> str:
    row = _keys(value, _METRIC_KEYS, label)
    history_id = row["history_id"]
    expected_prefix = "DEV" if phase == "development" else "CONF"
    if not isinstance(history_id, str) or re.fullmatch(rf"H-S18-{expected_prefix}-0[1-4]", history_id) is None:
        raise ValueError(f"{label}.history_id does not match the phase's canonical result-schema alias")
    for name in _METRIC_KEYS - {"history_id", "severity_ordering", "subtype_slices"}:
        _nullable_number(row[name], f"{label}.{name}")
    severity = _keys(row["severity_ordering"], {"P", "W"}, f"{label}.severity_ordering")
    _nullable_number(severity["P"], f"{label}.severity_ordering.P")
    _nullable_number(severity["W"], f"{label}.severity_ordering.W")
    slices = _keys(row["subtype_slices"], {"P1", "P2", "W1", "W2", "A1", "A2"}, f"{label}.subtype_slices")
    for name, metric in slices.items():
        _nullable_number(metric, f"{label}.subtype_slices.{name}")
    return history_id


def _validate_branch_metrics(value: object, label: str, phase: str) -> None:
    row = _keys(value, {"per_seed", "aggregate"}, label)
    per_seed = row["per_seed"]
    if not isinstance(per_seed, list) or len(per_seed) != 3:
        raise ValueError(f"{label}.per_seed must have exactly three model seeds")
    for index, item in enumerate(per_seed):
        seed_row = _keys(item, {"model_seed", "per_history"}, f"{label}.per_seed[{index}]")
        if seed_row["model_seed"] != MODEL_SEEDS[index]:
            raise ValueError(f"{label}.per_seed seed order is not canonical")
        per_history = seed_row["per_history"]
        if not isinstance(per_history, list) or len(per_history) != 4:
            raise ValueError(f"{label}.per_seed[{index}].per_history must have four histories")
        ids = tuple(_validate_history_metric(metric, f"{label}.per_seed[{index}].per_history[{j}]", phase) for j, metric in enumerate(per_history))
        expected_ids = tuple(
            f"H-S18-{'DEV' if phase == 'development' else 'CONF'}-{ordinal:02d}"
            for ordinal in range(1, 5)
        )
        if ids != expected_ids:
            raise ValueError(f"{label}.per_seed history order differs from the canonical four-history order")
        if len(set(ids)) != 4:
            raise ValueError(f"{label}.per_seed[{index}] contains duplicate history metrics")
    aggregate = _keys(row["aggregate"], _AGGREGATE_KEYS, f"{label}.aggregate")
    for name in _AGGREGATE_KEYS - {"directional_count", "positive_delta_history_count", "background_uncertainty"}:
        _nullable_number(aggregate[name], f"{label}.aggregate.{name}")
    for name in ("directional_count", "positive_delta_history_count"):
        value = aggregate[name]
        if value is not None:
            _int(value, f"{label}.aggregate.{name}", maximum=4)
    uncertainty = aggregate["background_uncertainty"]
    if not isinstance(uncertainty, list) or len(uncertainty) != 2:
        raise ValueError(f"{label}.aggregate.background_uncertainty must have two entries")
    for index, item in enumerate(uncertainty):
        _nullable_number(item, f"{label}.aggregate.background_uncertainty[{index}]")


def validate_sprint18_result(result: Mapping[str, object]) -> Mapping[str, object]:
    """Validate the closed Sprint18 v2 result schema and all arm-specific gates."""
    row = _keys(result, _TOP_KEYS, "result")
    if row["schema_id"] != RESULT_SCHEMA_ID or row["protocol_id"] != PROTOCOL_ID:
        raise ValueError("result schema/protocol identity must be exactly Sprint18 v2")
    phase = row["phase"]
    if phase not in ("development", "confirmation"):
        raise ValueError("result phase must be development or confirmation")
    arm = arm_spec(cast(str, row["arm_id"]))
    status = row["status"]
    if status not in ("VALID", "FAILED", "INVALID"):
        raise ValueError("result status is outside the frozen three-state enum")
    failures = row["failure_reasons"]
    if not isinstance(failures, list) or any(not isinstance(reason, str) or not reason for reason in failures) or len(set(failures)) != len(failures):
        raise ValueError("failure_reasons must be a unique array of non-empty strings")
    if status == "VALID" and failures:
        raise ValueError("VALID result must have no failure reasons")
    if status in ("FAILED", "INVALID") and not failures:
        raise ValueError("FAILED/INVALID result must state at least one reason")
    provenance = _keys(row["provenance"], _PROVENANCE_KEYS, "result.provenance")
    for name in _PROVENANCE_KEYS - {"mechanism_spec_sha256"}:
        if not _is_hash(provenance[name]):
            raise ValueError(f"result.provenance.{name} must be a lowercase SHA-256 digest")
    if provenance["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("result provenance must bind the accepted v2 protocol bytes")
    mechanism = provenance["mechanism_spec_sha256"]
    if mechanism != arm.mechanism_spec_sha256:
        raise ValueError("result mechanism-spec provenance does not match its exact arm")
    configuration = _keys(row["configuration"], _CONFIG_KEYS, "result.configuration")
    if configuration["model_seeds"] != list(MODEL_SEEDS):
        raise ValueError("result model seed order differs from the frozen v2 namespace")
    if configuration["optimizer"] != "AdamW" or configuration["clip_norm"] != 1.0 or configuration["precision"] != "float32" or configuration["ema_decay"] != EMA_DECAY:
        raise ValueError("result changes a frozen common optimizer setting")
    expected_rep = 0 if arm.arm_id == "N4-CRN" else 100
    expected_scale = 50 if arm.arm_id == "N4-CRN" else 0
    expected_epoch = 50 if arm.arm_id == "N4-CRN" else 100
    expected_batch = 512 if arm.arm_id == "N4-CRN" else 128
    expected_unit = "patch_records" if arm.arm_id == "N4-CRN" else "files"
    expected_lr = 1e-3 if arm.arm_id == "N4-CRN" else 1.5e-3
    expected_decay = 1e-4 if arm.arm_id == "N4-CRN" else 0.0
    for name, expected_value in (
        ("representation_epochs", expected_rep), ("scale_epochs", expected_scale),
        ("primary_checkpoint_epoch", expected_epoch), ("batch_size", expected_batch),
        ("batch_size_unit", expected_unit), ("learning_rate", expected_lr), ("weight_decay", expected_decay),
    ):
        if configuration[name] != expected_value:
            raise ValueError(f"result configuration.{name} differs from the frozen {arm.arm_id} contract")
    _number(configuration["learning_rate"], "configuration.learning_rate", exclusive_minimum=0.0)
    _number(configuration["weight_decay"], "configuration.weight_decay", minimum=0.0)
    coefficient_grids: Mapping[str, tuple[float, ...]] = {
        "lambda_info_max": (0.025, 0.05, 0.1),
        "lambda_hcc_max": (0.025, 0.05, 0.1),
        "alpha": (0.1, 0.25, 0.5),
        "beta": (0.1, 0.25, 0.5),
    }
    for name, grid in coefficient_grids.items():
        value = configuration[name]
        if value is not None and value not in grid:
            raise ValueError(f"configuration.{name} is outside its frozen coefficient grid")
    coefficient_rules: Mapping[str, tuple[str | None, ...]] = MappingProxyType(
        {
            "B0-FULL": ("lambda_info_max", None, None, None),
            "K3-FULL": ("lambda_info_max", None, None, None),
            "K7-FULL": ("lambda_info_max", None, None, None),
            "K9-FULL": ("lambda_info_max", None, None, None),
            "N1-HCC": (None, "lambda_hcc_max", None, None),
            "N2-MSP": ("lambda_info_max", None, "alpha", None),
            "N3-CCRP": ("lambda_info_max", None, None, "beta"),
            "N4-CRN": (None, None, None, None),
        }
    )
    coefficient_fields = ("lambda_info_max", "lambda_hcc_max", "alpha", "beta")
    active_fields = tuple(name for name in coefficient_rules[arm.arm_id] if name is not None)
    for name in coefficient_fields:
        if (configuration[name] is None) != (name not in active_fields):
            raise ValueError(f"configuration coefficient applicability is invalid for {arm.arm_id}")
    if arm.arm_id == "N4-CRN":
        if configuration["lambda_info_max"] is not None or configuration["lambda_hcc_max"] is not None or configuration["alpha"] is not None or configuration["beta"] is not None:
            raise ValueError("N4-CRN cannot declare representation coefficients")
    seed_runs = row["seed_runs"]
    if not isinstance(seed_runs, list) or len(seed_runs) != 3:
        raise ValueError("seed_runs must contain exactly three canonical model seeds")
    for index, item in enumerate(seed_runs):
        seed_row = _keys(item, _SEED_KEYS, f"seed_runs[{index}]")
        if seed_row["model_seed"] != MODEL_SEEDS[index]:
            raise ValueError("seed_runs are not in the canonical model seed order")
        _int(seed_row["representation_epochs_completed"], "representation_epochs_completed", maximum=100)
        _int(seed_row["scale_epochs_completed"], "scale_epochs_completed", maximum=50)
        _int(seed_row["optimizer_steps"], "optimizer_steps")
        for name in _SEED_KEYS - {"model_seed", "representation_epochs_completed", "scale_epochs_completed", "optimizer_steps", "compute"}:
            _nullable_hash(seed_row[name], f"seed_runs[{index}].{name}")
        _validate_compute_record(seed_row["compute"], f"seed_runs[{index}].compute")
        if status == "VALID":
            expected_steps = 50 * math.ceil(FIT_VALID_PATCHES / 512) if arm.arm_id == "N4-CRN" else 100 * math.ceil(FIT_HEALTHY_ROWS / 128)
            if (
                seed_row["representation_epochs_completed"] != expected_rep
                or seed_row["scale_epochs_completed"] != expected_scale
                or seed_row["optimizer_steps"] != expected_steps
            ):
                raise ValueError("VALID seed run is missing a complete frozen primary checkpoint schedule")
            for name in (
                "primary_checkpoint_sha256", "ema_state_sha256", "optimizer_scheduler_rng_sha256",
                "fit_bank_sha256", "calibration_sha256", "coefficient_trace_sha256",
            ):
                if not _is_hash(seed_row[name]):
                    raise ValueError(f"VALID seed run is missing {name} provenance")
            compute = _keys(seed_row["compute"], _COMPUTE_KEYS, f"seed_runs[{index}].compute")
            _int(compute["parameters_inference"], f"seed_runs[{index}].compute.parameters_inference", minimum=1)
            for name in ("inference_flops_per_fit_file", "training_flops_per_fit_file"):
                _number(compute[name], f"seed_runs[{index}].compute.{name}", exclusive_minimum=0.0)
            for name in ("parameter_ratio_vs_B0", "inference_flop_ratio_vs_B0", "training_flop_ratio_vs_B0"):
                _number(compute[name], f"seed_runs[{index}].compute.{name}", minimum=0.0)
            if compute["envelope_pass"] is not True:
                raise ValueError("VALID seed run must pass the frozen parameter/FLOP envelope")
    support = _keys(row["support"], _SUPPORT_KEYS, "result.support")
    if not _is_hash(support["common_support_sha256"]):
        raise ValueError("common support digest is invalid")
    histories = support["evaluation_histories"]
    prefix = "DEV" if phase == "development" else "CONF"
    expected_histories = [f"H-S18-{prefix}-{index:02d}" for index in range(1, 5)]
    if histories != expected_histories:
        raise ValueError("result evaluation histories are not the exact canonical four-history order")
    per_history = support["per_history"]
    if not isinstance(per_history, list) or len(per_history) != 4:
        raise ValueError("support.per_history must contain exactly four histories")
    for index, item in enumerate(per_history):
        history_row = _keys(item, _HISTORY_SUPPORT_KEYS, f"support.per_history[{index}]")
        if history_row["history_id"] != expected_histories[index]:
            raise ValueError("support history order differs from the declared evaluation histories")
        for name in ("p", "w", "a", "controls"):
            _int(history_row[name], f"support.per_history[{index}].{name}")
        _number(history_row["robot_days"], f"support.per_history[{index}].robot_days", minimum=0.0)
    _int(support["excluded_rows"], "support.excluded_rows")
    scores = _keys(row["scores"], {"S_pred", "S_pop"}, "result.scores")
    for name in ("S_pred", "S_pop"):
        raw_branch = scores[name]
        if not isinstance(raw_branch, Mapping):
            raise ValueError(f"scores.{name} must be a closed score-provenance object")
        optional = {"S_pop_bitwise_equal_to_B0"} if "S_pop_bitwise_equal_to_B0" in raw_branch else set()
        branch = _keys(raw_branch, _BRANCH_KEYS | optional, f"scores.{name}")
        if name == "S_pred" and "S_pop_bitwise_equal_to_B0" in branch:
            raise ValueError("the S_pop parity field cannot be attached to S_pred")
        parity = branch.get("S_pop_bitwise_equal_to_B0")
        if parity is not None and not isinstance(parity, bool):
            raise ValueError("S_pop bitwise-parity flag must be boolean or null")
        if branch["calibration_role_sha256"] != provenance["calibration_role_sha256"]:
            raise ValueError("score branch Calibration provenance differs from the result role binding")
        if branch["calibration_quantile"] != 0.95 or branch["independent"] is not True:
            raise ValueError(f"scores.{name} must use its independent q95 branch")
        thresholds = branch["thresholds_by_seed"]
        if not isinstance(thresholds, list) or len(thresholds) != 3:
            raise ValueError(f"scores.{name}.thresholds_by_seed must have three entries")
        for index, threshold in enumerate(thresholds):
            _nullable_number(threshold, f"scores.{name}.thresholds_by_seed[{index}]")
        for hash_name in ("calibration_role_sha256", "score_provenance_sha256"):
            if not _is_hash(branch[hash_name]):
                raise ValueError(f"scores.{name}.{hash_name} is invalid")
        if status == "VALID" and any(threshold is None for threshold in thresholds):
            raise ValueError("VALID score branch requires one finite Calibration threshold per seed")
    if arm.arm_id == "N4-CRN":
        if scores["S_pop"].get("S_pop_bitwise_equal_to_B0") is not True:
            raise ValueError("N4-CRN result must prove bitwise S_pop parity with B0")
    metrics = _keys(row["metrics"], {"S_pred", "S_pop"}, "result.metrics")
    _validate_branch_metrics(metrics["S_pred"], "metrics.S_pred", cast(str, phase))
    _validate_branch_metrics(metrics["S_pop"], "metrics.S_pop", cast(str, phase))
    if status == "VALID":
        for branch_name in ("S_pred", "S_pop"):
            branch_metrics = cast(Mapping[str, object], metrics[branch_name])
            for seed_entry in cast(list[object], branch_metrics["per_seed"]):
                per_history_metrics = cast(list[object], cast(Mapping[str, object], seed_entry)["per_history"])
                for metric_entry in per_history_metrics:
                    metric = cast(Mapping[str, object], metric_entry)
                    for field in _METRIC_KEYS - {"history_id"}:
                        value = metric[field]
                        if field == "severity_ordering":
                            if any(item is None for item in cast(Mapping[str, object], value).values()):
                                raise ValueError("VALID result is missing a severity-ordering metric")
                        elif field == "subtype_slices":
                            if any(item is None for item in cast(Mapping[str, object], value).values()):
                                raise ValueError("VALID result is missing a subtype-slice metric")
                        elif value is None:
                            raise ValueError(f"VALID result is missing {field} on its fixed support")
            aggregate = cast(Mapping[str, object], branch_metrics["aggregate"])
            for field in _AGGREGATE_KEYS:
                value = aggregate[field]
                if value is None or (field == "background_uncertainty" and any(item is None for item in cast(list[object], value))):
                    raise ValueError(f"VALID result is missing aggregate {field}")
    gates = _keys(row["branch_gates"], {"S_pred", "S_pop"}, "result.branch_gates")
    for name in ("S_pred", "S_pop"):
        branch = _keys(gates[name], _GATE_KEYS, f"branch_gates.{name}")
        if not isinstance(branch["recovery_applied"], bool):
            raise ValueError(f"branch_gates.{name}.recovery_applied must be boolean")
        for gate_name in _GATE_KEYS - {"recovery_applied"}:
            value = branch[gate_name]
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"branch_gates.{name}.{gate_name} must be boolean or null")
        if status == "VALID" and any(
            branch[field] is not True for field in ("support_and_provenance", "compute", "score_separation")
        ):
            raise ValueError("VALID result must pass support, compute, and score-separation validity gates")
        if phase == "development" or status != "VALID" or (phase == "confirmation" and arm.arm_id == "B0-FULL"):
            if branch["recovery_applied"] or any(branch[field] is not None for field in _RECOVERY_GATE_FIELDS):
                raise ValueError("Development/B0/non-valid rows cannot claim recovery gates")
    return row


def serialize_sprint18_result(result: Mapping[str, object]) -> str:
    """Emit canonical JSON only after the exact closed v2 result validates."""
    validate_sprint18_result(result)
    return json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
