"""Common, execution-free contract harness for Sprint 17 ablations.

This module owns the *contract* shared by the future training/evaluation runner;
it intentionally performs no data loading, model construction, checkpoint I/O, or
score computation.  It makes the protocol-v4 registry, parity, provenance,
qualification, and paired-support guards executable before any arm can run.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence

__all__ = [
    "ARM_REGISTRY",
    "MODEL_SEEDS",
    "OPTIMIZER_STEPS",
    "CHECKPOINT_STEP",
    "PARAM_DELTA_LIMIT",
    "FLOP_DELTA_LIMIT",
    "QUALIFICATION",
    "WAIVER_ID",
    "WAIVER_SCOPE",
    "ArmSpec",
    "TrainingParity",
    "ComputeEnvelope",
    "PairedEvaluation",
    "get_arm",
    "validate_arm",
    "validate_training_parity",
    "validate_compute_envelope",
    "validate_eligibility",
    "validate_provenance",
    "evaluate_paired",
    "dry_contract",
    "validate_dry_contract",
]

PROTOCOL_ID = "sprint17-ablation-v4"
SCHEMA_ID = "sprint17-ablation-result-v4"
MODEL_SEEDS = (171701, 171702, 171703)
OPTIMIZER_STEPS = 300
CHECKPOINT_STEP = 300
PARAM_DELTA_LIMIT = 0.05
FLOP_DELTA_LIMIT = 0.10
QUALIFICATION = "MEASURABLE_WITH_USER_WAIVER"
WAIVER_ID = "S17-V4-WAIVER-H-S17-V3-DESIGN-03-COHORT_MIX_15_60-P-91-149"
WAIVER_SCOPE = {
    "history_id": "H-S17-V3-DESIGN-03",
    "role": "DESIGN",
    "gate": "cohort_mix_15_60",
    "cohort": "P",
    "observed_numerator": 91,
    "observed_denominator": 149,
    "observed_share": 91 / 149,
    "original_upper_bound": 0.60,
}
_COMPONENTS = ("C2", "C3", "C5", "C6", "C7", "C8", "C9")
_SINGLE_IDS = tuple(f"{component}-{suffix}" for component in _COMPONENTS for suffix in ("A", "B"))
_PARENT_ARM_IDS = ("B0",) + _SINGLE_IDS
_COMBINATION_COMPONENTS = {
    "K1": ("C2", "C3"),
    "K2": ("C3", "C5"),
    "K3": ("C5", "C6"),
    "K4": ("C6", "C7"),
    "K5": ("C7", "C8"),
    "K6": ("C8", "C9"),
    "K7": ("C2", "C3", "C5", "C6"),
    "K8": ("C7", "C8", "C9"),
    "K9": ("C2", "C3", "C5", "C6", "C7", "C8", "C9"),
}
_SINGLE_DESCRIPTIONS = {
    "C2-A": "fixed-width windows at 50% overlap; preserve valid length, padding, starts, and file identity",
    "C2-B": "two fixed-width offset grids with unit-mass support weights; duplicated timesteps do not gain pooling weight",
    "C3-A": "parameter-matched residual depthwise-separable dilated 1-D convolutional patch encoder",
    "C3-B": "parameter-matched lightweight within-patch Transformer with explicit position and padding masks",
    "C5-A": "fixed total mask ratio with channel-time blocks and multi-horizon EMA latent prediction",
    "C5-B": "retain EMA masked prediction; replace file-level InfoNCE with VICReg-style variance/covariance regularization",
    "C6-A": "valid-patch mean plus standard deviation, projected to the existing file-embedding dimension",
    "C6-B": "gated attention over valid patches with no anomaly-derived input or target",
    "C7-A": "per-condition robust location plus Ledoit-Wolf-style covariance shrinkage with the frozen hierarchy/fallback",
    "C7-B": "condition-aware kNN/local-density score with robust scale fitted only from Fit healthy-reference rows",
    "C8-A": "non-query-conditioned standardized Huber prediction-residual energy for S_pred",
    "C8-B": "non-query-conditioned cosine distance between predicted and target latents for S_pred",
    "C9-A": "fixed-fraction top-k mean over valid patch scores",
    "C9-B": "maximum fixed-duration contiguous-window mean using patch starts and valid lengths",
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_RE = re.compile(r"^H-S17-V3-(DESIGN|FIT|CAL|DEV|CONF)-[0-9]{2}$")


@dataclass(frozen=True)
class ArmSpec:
    """Immutable registry metadata for one protocol-v4 arm."""

    arm_id: str
    kind: str
    components: tuple[str, ...]
    one_principal_change: str
    description: str
    parent_arm_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingParity:
    """Training settings that every trainable arm must match exactly."""

    model_seeds: tuple[int, ...] = MODEL_SEEDS
    optimizer_steps: int = OPTIMIZER_STEPS
    checkpoint_step: int = CHECKPOINT_STEP
    optimizer: str = "B0-frozen-optimizer"
    schedule: str = "B0-frozen-schedule"
    batch_construction: str = "B0-frozen-batch-construction"
    gradient_clipping: float = 1.0
    weight_decay: float = 0.0
    precision: str = "B0-frozen-precision"
    device_class: str = "B0-frozen-device-class"
    initialization_policy: str = "B0-frozen-initialization"
    warm_start: bool = False
    extra_runs: int = 0


@dataclass(frozen=True)
class ComputeEnvelope:
    """Measured parameter/FLOP counts against the frozen B0 reference."""

    b0_params: int
    params_total: int
    b0_flops_per_reference_sample: int
    flops_per_reference_sample: int

    @property
    def params_delta_fraction(self) -> float:
        return (self.params_total - self.b0_params) / self.b0_params

    @property
    def flops_delta_fraction(self) -> float:
        return (self.flops_per_reference_sample - self.b0_flops_per_reference_sample) / self.b0_flops_per_reference_sample


@dataclass(frozen=True)
class PairedEvaluation:
    """Deterministic paired deltas on one explicitly declared common support."""

    arm_id: str
    history_id: str
    branch: str
    support_ids: tuple[str, ...]
    deltas: tuple[float, ...]

    @property
    def delta_mean(self) -> float:
        return sum(self.deltas) / len(self.deltas)


def _make_registry() -> dict[str, ArmSpec]:
    registry: dict[str, ArmSpec] = {
        "B0": ArmSpec("B0", "baseline", (), "none", "unchanged accepted representation-to-score architecture"),
    }
    for arm_id in _SINGLE_IDS:
        component = arm_id.split("-", 1)[0]
        registry[arm_id] = ArmSpec(arm_id, "single", (component,), component, _SINGLE_DESCRIPTIONS[arm_id])
    for arm_id, components in _COMBINATION_COMPONENTS.items():
        registry[arm_id] = ArmSpec(arm_id, "combination", components, "none", "selected single-arm implementations copied exactly")
    return registry


ARM_REGISTRY = _make_registry()


def get_arm(arm_id: str) -> ArmSpec:
    """Return an immutable registry entry or fail closed for unknown arms."""
    try:
        spec = ARM_REGISTRY[arm_id]
    except KeyError as exc:
        raise ValueError(f"unknown Sprint 17 arm: {arm_id!r}") from exc
    validate_arm(spec)
    return spec


def validate_arm(spec: ArmSpec) -> ArmSpec:
    """Validate registry identity, kind, and exact component membership."""
    if not isinstance(spec, ArmSpec) or spec.arm_id not in ARM_REGISTRY:
        raise ValueError("arm is not an exact Sprint 17 registry entry")
    registered = ARM_REGISTRY[spec.arm_id]
    if spec != registered:
        raise ValueError(f"arm metadata mismatch for {spec.arm_id}")
    if spec.kind == "baseline":
        if spec.components or spec.one_principal_change != "none":
            raise ValueError("B0 must have no components and no principal change")
    elif spec.kind == "single":
        if len(spec.components) != 1 or spec.components[0] not in _COMPONENTS:
            raise ValueError("single arm must name exactly one registered component")
        if spec.one_principal_change != spec.components[0]:
            raise ValueError("single arm principal-change field must match its component")
    elif spec.kind == "combination":
        if tuple(spec.components) != _COMBINATION_COMPONENTS[spec.arm_id]:
            raise ValueError("combination component membership is not the frozen K membership")
        if spec.parent_arm_ids:
            raise ValueError("combination parent arms are selected at Development, not registry metadata")
    else:
        raise ValueError(f"unsupported arm kind: {spec.kind!r}")
    return spec


def validate_training_parity(parity: TrainingParity) -> TrainingParity:
    """Require exact seeds, budget, checkpoint, and B0 execution settings."""
    if parity.model_seeds != MODEL_SEEDS:
        raise ValueError("model seeds must equal [171701, 171702, 171703]")
    if parity.optimizer_steps != OPTIMIZER_STEPS or parity.checkpoint_step != CHECKPOINT_STEP:
        raise ValueError("optimizer and checkpoint steps must both equal 300")
    expected = TrainingParity()
    for field in (
        "optimizer",
        "schedule",
        "batch_construction",
        "precision",
        "device_class",
        "initialization_policy",
        "warm_start",
        "extra_runs",
    ):
        if getattr(parity, field) != getattr(expected, field):
            raise ValueError(f"training parity mismatch: {field}")
    for field in ("gradient_clipping", "weight_decay"):
        value = getattr(parity, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"training parity {field} must be finite and non-negative")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"training parity {field} must be finite and non-negative")
        if value != getattr(expected, field):
            raise ValueError(f"training parity mismatch: {field}")
    return parity


def validate_compute_envelope(envelope: ComputeEnvelope) -> ComputeEnvelope:
    """Require finite positive references and the frozen ±5%/±10% envelopes."""
    values = (envelope.b0_params, envelope.params_total, envelope.b0_flops_per_reference_sample, envelope.flops_per_reference_sample)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("parameter and FLOP counts must be positive integers")
    p_delta, f_delta = envelope.params_delta_fraction, envelope.flops_delta_fraction
    if not math.isfinite(p_delta) or not math.isfinite(f_delta):
        raise ValueError("parameter and FLOP deltas must be finite")
    if abs(p_delta) > PARAM_DELTA_LIMIT:
        raise ValueError("parameter delta exceeds the frozen ±5% envelope")
    if abs(f_delta) > FLOP_DELTA_LIMIT:
        raise ValueError("FLOP delta exceeds the frozen ±10% envelope")
    return envelope


def validate_eligibility(eligibility: Mapping[str, object]) -> Mapping[str, object]:
    """Require the exact protocol-v4 user-waiver qualification tuple."""
    if not isinstance(eligibility, Mapping):
        raise ValueError("eligibility must be a mapping")
    expected = {"qualification": QUALIFICATION, "waiver_id": WAIVER_ID, "waiver_scope": WAIVER_SCOPE}
    if set(eligibility) != set(expected):
        raise ValueError("eligibility must contain exactly qualification, waiver_id, and waiver_scope")
    if eligibility["qualification"] != QUALIFICATION:
        raise ValueError("only MEASURABLE_WITH_USER_WAIVER is eligible")
    if eligibility["waiver_id"] != WAIVER_ID or eligibility["waiver_scope"] != WAIVER_SCOPE:
        raise ValueError("eligibility waiver identity does not equal the exact authorized tuple")
    return eligibility


def validate_provenance(provenance: Mapping[str, object]) -> Mapping[str, object]:
    """Validate v4 role/binding/root/metric/checkpoint/bank-cache provenance."""
    required = {
        "data_protocol", "role_binding_sha256", "data_roles", "data_seeds", "model_seeds",
        "fit_root_sha256", "calibration_root_sha256", "evaluation_root_sha256",
        "metric_code_sha256", "checkpoint_sha256", "cache_sha256", "parent_arm_ids",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != required:
        raise ValueError("incomplete or extra v4 provenance: exactly the required role/binding/root/metric/checkpoint/bank-cache fields are required")
    if provenance["data_protocol"] != "sprint15-benchmark-protocol-v7":
        raise ValueError("data protocol must be sprint15-benchmark-protocol-v7")
    roles = provenance["data_roles"]
    if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)) or not roles or any(not isinstance(role, str) or not _ROLE_RE.fullmatch(role) for role in roles):
        raise ValueError("data_roles must contain only bound H-S17-V3 role IDs")
    if provenance["model_seeds"] != list(MODEL_SEEDS) and provenance["model_seeds"] != MODEL_SEEDS:
        raise ValueError("provenance model seeds must equal the frozen model seed set")
    seeds = provenance["data_seeds"]
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("data_seeds must be non-empty non-negative integers")
    parents = provenance["parent_arm_ids"]
    if not isinstance(parents, Sequence) or isinstance(parents, (str, bytes)):
        raise ValueError("parent_arm_ids must be a non-string sequence")
    if any(not isinstance(parent, str) or parent not in _PARENT_ARM_IDS for parent in parents):
        raise ValueError("parent_arm_ids contains an invalid or non-single arm ID")
    if len(set(parents)) != len(parents):
        raise ValueError("parent_arm_ids contains duplicate arm IDs")
    for name in ("role_binding_sha256", "fit_root_sha256", "calibration_root_sha256", "evaluation_root_sha256", "metric_code_sha256", "cache_sha256"):
        value = provenance[name]
        if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    checkpoint = provenance["checkpoint_sha256"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not _HASH_RE.fullmatch(checkpoint)):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest or null")
    return provenance


def _branch_values(values: Mapping[str, Mapping[str, float]], branch: str, label: str) -> Mapping[str, float]:
    if set(values) != {"S_pred", "S_pop"}:
        raise ValueError(f"{label} must declare both independent score branches")
    if branch not in values:
        raise ValueError(f"unsupported score branch: {branch!r}")
    selected = values[branch]
    if not isinstance(selected, Mapping) or not selected:
        raise ValueError(f"{label}[{branch}] must have non-empty keyed scores")
    if any(not isinstance(key, str) or not key for key in selected):
        raise ValueError(f"{label}[{branch}] contains an invalid support key")
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in selected.values()):
        raise ValueError(f"{label}[{branch}] scores must be finite")
    return selected


def evaluate_paired(
    arm_id: str,
    history_id: str,
    arm_scores: Mapping[str, Mapping[str, float]],
    b0_scores: Mapping[str, Mapping[str, float]],
    support_ids: Sequence[str],
    *,
    branch: str,
) -> PairedEvaluation:
    """Compute deterministic arm-minus-B0 deltas on declared identical support.

    Support is checked for exact equality; this function never silently intersects,
    drops rows, pools histories, or substitutes one score branch for another.
    """
    get_arm(arm_id)
    if not _ROLE_RE.fullmatch(history_id):
        raise ValueError("history_id is not a bound Sprint 17 role")
    if branch not in ("S_pred", "S_pop"):
        raise ValueError("branch must be S_pred or S_pop")
    if arm_scores.get("S_pred") is arm_scores.get("S_pop") or b0_scores.get("S_pred") is b0_scores.get("S_pop"):
        raise ValueError("S_pred and S_pop score branches must not alias")
    arm = _branch_values(arm_scores, branch, "arm_scores")
    baseline = _branch_values(b0_scores, branch, "b0_scores")
    declared = tuple(support_ids)
    if not declared or len(set(declared)) != len(declared) or any(not isinstance(item, str) or not item for item in declared):
        raise ValueError("common support must be a non-empty unique sequence of IDs")
    if set(arm) != set(baseline) or set(arm) != set(declared):
        raise ValueError("arm, B0, and declared common support must match exactly")
    ordered = tuple(sorted(declared))
    deltas = tuple(float(arm[key]) - float(baseline[key]) for key in ordered)
    if not all(math.isfinite(delta) for delta in deltas):
        raise ValueError("paired deltas must be finite")
    return PairedEvaluation(arm_id, history_id, branch, ordered, deltas)


def dry_contract(arm_id: str) -> dict[str, object]:
    """Build a consumer-observable, execution-free contract for one arm."""
    spec = get_arm(arm_id)
    result = {
        "schema_id": SCHEMA_ID,
        "protocol_id": PROTOCOL_ID,
        "arm_id": spec.arm_id,
        "arm_kind": spec.kind,
        "component_set": list(spec.components),
        "config": {
            "one_principal_change": spec.one_principal_change,
            "optimizer_steps": OPTIMIZER_STEPS,
            "checkpoint_step": CHECKPOINT_STEP,
            "optimizer": "B0-frozen-optimizer",
            "schedule": "B0-frozen-schedule",
            "batch_construction": "B0-frozen-batch-construction",
            "gradient_clipping": 1.0,
            "weight_decay": 0.0,
            "precision": "B0-frozen-precision",
            "device_class": "B0-frozen-device-class",
            "initialization_policy": "B0-frozen-initialization",
            "warm_start": False,
            "extra_runs": 0,
            "score_branches": ["S_pred", "S_pop"],
        },
        "eligibility": {
            "qualification": QUALIFICATION,
            "waiver_id": WAIVER_ID,
            "waiver_scope": dict(WAIVER_SCOPE),
        },
    }
    validate_dry_contract(result)
    return result


def validate_dry_contract(contract: Mapping[str, object]) -> Mapping[str, object]:
    """Validate the public dry contract without training or touching data."""
    if not isinstance(contract, Mapping) or contract.get("schema_id") != SCHEMA_ID or contract.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("dry contract must identify protocol-v4 and schema-v4")
    spec = get_arm(str(contract.get("arm_id", "")))
    if contract.get("arm_kind") != spec.kind or tuple(contract.get("component_set", ())) != spec.components:
        raise ValueError("dry contract arm metadata does not match the registry")
    config = contract.get("config")
    expected_config = {
        "one_principal_change": spec.one_principal_change,
        "optimizer_steps": OPTIMIZER_STEPS,
        "checkpoint_step": CHECKPOINT_STEP,
        "optimizer": "B0-frozen-optimizer",
        "schedule": "B0-frozen-schedule",
        "batch_construction": "B0-frozen-batch-construction",
        "gradient_clipping": 1.0,
        "weight_decay": 0.0,
        "precision": "B0-frozen-precision",
        "device_class": "B0-frozen-device-class",
        "initialization_policy": "B0-frozen-initialization",
        "warm_start": False,
        "extra_runs": 0,
        "score_branches": ["S_pred", "S_pop"],
    }
    if not isinstance(config, Mapping) or set(config) != set(expected_config) or any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("dry contract config does not match frozen parity")
    validate_eligibility(contract.get("eligibility", {}))
    return contract
