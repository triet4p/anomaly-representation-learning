"""Sprint 17 Task 19 — frozen K1–K9 combination configurations.

Execution-free composition surface for the nine predeclared multi-component
combinations. Each K arm copies the exact Task 17 selected single-arm
implementations with zero combination-specific tuning: this module exposes no
hyperparameter, seed, threshold, or adapter argument of any kind, so a
per-combination override is structurally inexpressible. Any override attempt
must therefore edit these frozen bytes, which the committed freeze JSON and
focused tests detect via provenance hashes and canonical digests.

Composition inputs (Task 19 only; no execution):
- member IDs come solely from the frozen Task 17 ``selected`` map;
- implementation bytes are bound by SHA-256 provenance, never re-read here;
- outcome metrics, checkpoints, data roots, Confirmation histories, and
  Sprint 15 Sealed histories are not inputs and cannot influence output.

K execution begins in Task 20; this module performs no training, scoring,
threshold computation, checkpoint I/O, or data access of any kind.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

__all__ = [
    "PROTOCOL_ID",
    "SCHEMA_ID",
    "MODEL_SEEDS",
    "OPTIMIZER_STEPS",
    "CHECKPOINT_STEP",
    "CALIBRATION_QUANTILE",
    "QUALIFICATION",
    "WAIVER_ID",
    "WAIVER_SCOPE",
    "DATA_PROTOCOL",
    "BINDING_SHA256",
    "SUPPORT_SHA256",
    "B0_PARAMS",
    "B0_FLOPS_REFERENCE",
    "SELECTED_ARMS",
    "K_MEMBERS",
    "MEMBER_IMPLEMENTATION",
    "GLOBAL_SETTINGS",
    "REQUIRED_PROVENANCE_FILES",
    "members_for",
    "selected_arm_for_component",
    "materialize_combination",
    "materialize_all",
    "validate_frozen_combination",
    "canonical_hash",
]

PROTOCOL_ID = "sprint17-ablation-v4"
SCHEMA_ID = "sprint17-ablation-result-v4"
MODEL_SEEDS = (171701, 171702, 171703)
OPTIMIZER_STEPS = 300
CHECKPOINT_STEP = 300
CALIBRATION_QUANTILE = 0.95
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
    "original_upper_bound": 0.6,
}
DATA_PROTOCOL = "sprint15-benchmark-protocol-v7"
BINDING_SHA256 = "075868b22c028a981cc63ffe37b8d29720cd324c3e2c7254ce688678758230ba"
SUPPORT_SHA256 = "800fc825ac948c3661c2dc727c83083095a796df8950c94922dafabae318f59a"
B0_PARAMS = 1821698
B0_FLOPS_REFERENCE = 182016709

#: Exact Task 17 frozen selection (task-17.md section 5); the sole member source.
SELECTED_ARMS = {
    "C2": "C2-A",
    "C3": "C3-A",
    "C5": "C5-A",
    "C6": "C6-A",
    "C7": "C7-A",
    "C8": "C8-A",
    "C9": "C9-B",
}

#: Batch contract membership (protocol v4 section 2.1) with selected IDs bound.
K_MEMBERS = {
    "K1": ("C2-A", "C3-A"),
    "K2": ("C3-A", "C5-A"),
    "K3": ("C5-A", "C6-A"),
    "K4": ("C6-A", "C7-A"),
    "K5": ("C7-A", "C8-A"),
    "K6": ("C8-A", "C9-B"),
    "K7": ("C2-A", "C3-A", "C5-A", "C6-A"),
    "K8": ("C7-A", "C8-A", "C9-B"),
    "K9": ("C2-A", "C3-A", "C5-A", "C6-A", "C7-A", "C8-A", "C9-B"),
}

#: Exact selected single-arm implementation identities. ``frozen_numerics``
#: repeats the Task 10–16 frozen constants that define each implementation;
#: any per-combination change would have to alter these bytes.
MEMBER_IMPLEMENTATION = {
    "C2-A": {
        "component": "C2",
        "impl_module": "src/representation/sprint17_c2.py",
        "builder": "arm_patchifier",
        "adapter_description": (
            "overlap-explicit single-grid patchifier (W=32, stride 16, "
            "pad_end); no geometry adapter, no support weights; downstream "
            "pooling, masking, scoring, and metric code unchanged"
        ),
        "frozen_numerics": {"patch_size": 32, "stride": 16, "grids": [[0, 16]], "pad_end": True},
        "driver": "experiments/sprint17_task10_c2.py",
        "summary": "experiments/sprint17-task10-c2-summary.json",
    },
    "C3-A": {
        "component": "C3",
        "impl_module": "src/representation/sprint17_c3.py",
        "builder": "ResidualDilatedPatchEncoder",
        "adapter_description": (
            "residual depthwise-separable dilated-1D-conv patch encoder "
            "(in-proj k1, depthwise k3/dilation-2, pointwise k1, residual); "
            "unchanged output head and masked pooling; no other adapter"
        ),
        "frozen_numerics": {
            "local_encoder_params_at_6_128": 34688,
            "b0_local_encoder_params": 35712,
        },
        "driver": "experiments/sprint17_task11_c3.py",
        "summary": "experiments/sprint17-task11-c3-summary.json",
    },
    "C5-A": {
        "component": "C5",
        "impl_module": "src/representation/sprint17_c5.py",
        "builder": "channel_time_block_mask",
        "adapter_description": (
            "seeded contiguous channel-time block masking at the frozen "
            "0.40 ratio plus shared-head multi-horizon (0/+1/+2) EMA latent "
            "prediction; no new parameters; scoring/inference code unchanged"
        ),
        "frozen_numerics": {"total_mask_ratio": 0.40, "n_blocks": 2, "horizons": [0, 1, 2]},
        "driver": "experiments/sprint17_task12_c5.py",
        "summary": "experiments/sprint17-task12-c5-summary.json",
    },
    "C6-A": {
        "component": "C6",
        "impl_module": "src/representation/sprint17_c6.py",
        "builder": "MeanStdProjectionPooling",
        "adapter_description": (
            "valid-patch mean-plus-population-std concatenated to 2D and "
            "linearly projected to D (one Linear(256,128) + bias, 32,896 "
            "params); invalid patches masked to zero before statistics; "
            "no other adapter"
        ),
        "frozen_numerics": {"pooling_params_at_128": 32896},
        "driver": "experiments/sprint17_task13_c6.py",
        "summary": "experiments/sprint17-task13-c6-summary.json",
    },
    "C7-A": {
        "component": "C7",
        "impl_module": "src/representation/sprint17_c7.py",
        "builder": "RobustShrinkageReference",
        "adapter_description": (
            "per-condition robust location (coordinate-wise median) plus "
            "Ledoit-Wolf-style covariance shrinkage to scaled identity "
            "(floor 0.05), squared-Mahalanobis energy over the frozen "
            "(robot,program)->robot->program->global hierarchy with "
            "fallback below 8 rows; fitted statistics only, no trainable "
            "parameters; no other adapter"
        ),
        "frozen_numerics": {"min_cell_rows": 8, "shrinkage_floor": 0.05, "knn_k": 5},
        "driver": "experiments/sprint17_task14_c7.py",
        "summary": "experiments/sprint17-task14-c7-summary.json",
    },
    "C8-A": {
        "component": "C8",
        "impl_module": "src/representation/sprint17_c8.py",
        "builder": "HuberStandardizer",
        "adapter_description": (
            "non-query-conditioned standardized Huber prediction-residual "
            "energy (Fit-fitted per-dimension median/MAD, floor 1e-6, Huber "
            "delta 1.0, file-mean over valid masked patches); fitted "
            "statistics only (256 floats), no trainable parameters; "
            "no other adapter"
        ),
        "frozen_numerics": {"huber_delta": 1.0, "mad_floor": 1e-06},
        "driver": "experiments/sprint17_task15_c8.py",
        "summary": "experiments/sprint17-task15-c8-summary.json",
    },
    "C9-B": {
        "component": "C9",
        "impl_module": "src/representation/sprint17_c9.py",
        "builder": "window_max_mean",
        "adapter_description": (
            "maximum fixed-duration contiguous-window mean using patch "
            "starts/valid lengths (duration=64 timesteps, earliest-window "
            "tie-break)"
        ),
        "frozen_numerics": {"window_duration_timesteps": 64},
        "driver": "experiments/sprint17_task16_c9.py",
        "summary": "experiments/sprint17-task16-c9-summary.json",
    },
}

#: Fixed global settings shared by every K arm; validated identical per arm.
GLOBAL_SETTINGS = {
    "protocol_id": PROTOCOL_ID,
    "schema_id": SCHEMA_ID,
    "data_protocol": DATA_PROTOCOL,
    "role_binding_sha256": BINDING_SHA256,
    "model_seeds": [171701, 171702, 171703],
    "optimizer_steps": OPTIMIZER_STEPS,
    "checkpoint_step": CHECKPOINT_STEP,
    "calibration_quantile": CALIBRATION_QUANTILE,
    "b0_params": B0_PARAMS,
    "b0_flops_per_reference_sample": B0_FLOPS_REFERENCE,
    "support_sha256": SUPPORT_SHA256,
    "score_branches": ["S_pred", "S_pop"],
    "no_combination_tuning": True,
}

#: Files whose SHA-256 binds a materialized combination. Summaries enter only
#: as opaque bytes (composition never parses outcome fields).
REQUIRED_PROVENANCE_FILES = (
    "experiments/sprint17-ablation-protocol-v4.md",
    "experiments/sprint17-role-binding-v3.json",
    "experiments/sprint17-task17-selection.json",
    "experiments/sprint17-task7-b0-summary.json",
    "experiments/sprint17-task10-c2-summary.json",
    "experiments/sprint17-task11-c3-summary.json",
    "experiments/sprint17-task12-c5-summary.json",
    "experiments/sprint17-task13-c6-summary.json",
    "experiments/sprint17-task14-c7-summary.json",
    "experiments/sprint17-task15-c8-summary.json",
    "experiments/sprint17-task16-c9-summary.json",
    "src/representation/sprint17_ablation.py",
    "src/representation/sprint17_c2.py",
    "src/representation/sprint17_c3.py",
    "src/representation/sprint17_c5.py",
    "src/representation/sprint17_c6.py",
    "src/representation/sprint17_c7.py",
    "src/representation/sprint17_c8.py",
    "src/representation/sprint17_c9.py",
    "experiments/sprint17_task10_c2.py",
    "experiments/sprint17_task11_c3.py",
    "experiments/sprint17_task12_c5.py",
    "experiments/sprint17_task13_c6.py",
    "experiments/sprint17_task14_c7.py",
    "experiments/sprint17_task15_c8.py",
    "experiments/sprint17_task16_c9.py",
    "src/synth/probe15.py",
    "src/synth/events.py",
    "src/representation/attribution_metrics.py",
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_id",
        "protocol_id",
        "task",
        "phase",
        "arm_id",
        "arm_kind",
        "component_set",
        "member_arm_ids",
        "members",
        "global",
        "eligibility",
        "provenance_inputs",
        "combination_sha256",
        "no_execution",
        "confirmation_sealed_access",
    }
)
_MEMBER_KEYS = frozenset(
    {
        "component",
        "impl_module",
        "builder",
        "adapter_description",
        "frozen_numerics",
        "driver",
        "summary",
    }
)


def canonical_hash(obj: object) -> str:
    """SHA-256 over canonical JSON (sorted keys, compact separators, UTF-8)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def selected_arm_for_component(component: str) -> str:
    """Return the frozen Task 17 selected arm for one suspect component."""
    try:
        return SELECTED_ARMS[component]
    except KeyError:
        raise ValueError(f"unknown Sprint 17 suspect component: {component!r}") from None


def members_for(k_id: str) -> tuple[str, ...]:
    """Return the exact frozen member arm IDs for one combination."""
    try:
        members = K_MEMBERS[k_id]
    except KeyError:
        raise ValueError(f"unknown Sprint 17 combination: {k_id!r}") from None
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError(f"combination {k_id} membership is not duplicate-free with >=2 arms")
    for arm_id in members:
        if arm_id not in MEMBER_IMPLEMENTATION:
            raise ValueError(f"combination {k_id} member {arm_id} is not a frozen selected arm")
    return members


def materialize_combination(k_id: str, provenance_inputs: Mapping[str, str]) -> dict:
    """Materialize one frozen K config from the exact selected implementations.

    Takes no hyperparameters by design: member composition, numerics, and
    global settings are all frozen constants above. ``provenance_inputs`` must
    map exactly the :data:`REQUIRED_PROVENANCE_FILES` paths to lowercase
    SHA-256 digests of their frozen bytes.
    """
    members = members_for(k_id)
    if not isinstance(provenance_inputs, Mapping) or set(provenance_inputs) != set(
        REQUIRED_PROVENANCE_FILES
    ):
        raise ValueError("provenance_inputs must bind exactly the required frozen files")
    for path, digest in provenance_inputs.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise ValueError(f"provenance_inputs[{path}] is not a lowercase SHA-256 digest")
    components = [MEMBER_IMPLEMENTATION[arm_id]["component"] for arm_id in members]
    if len(set(components)) != len(components):
        raise ValueError(f"combination {k_id} repeats a suspect component")
    doc = {
        "schema_id": SCHEMA_ID,
        "protocol_id": PROTOCOL_ID,
        "task": "Task 19 — frozen selected combination configs",
        "phase": "development",
        "arm_id": k_id,
        "arm_kind": "combination",
        "component_set": components,
        "member_arm_ids": list(members),
        "members": {
            arm_id: {
                key: MEMBER_IMPLEMENTATION[arm_id][key] for key in sorted(_MEMBER_KEYS)
            }
            for arm_id in members
        },
        "global": json.loads(json.dumps(GLOBAL_SETTINGS, sort_keys=True)),
        "eligibility": {
            "qualification": QUALIFICATION,
            "waiver_id": WAIVER_ID,
            "waiver_scope": json.loads(json.dumps(WAIVER_SCOPE, sort_keys=True)),
        },
        "provenance_inputs": dict(sorted(provenance_inputs.items())),
        "no_execution": True,
        "confirmation_sealed_access": False,
    }
    unsigned = {k: v for k, v in doc.items() if k != "combination_sha256"}
    doc["combination_sha256"] = canonical_hash(unsigned)
    return validate_frozen_combination(doc)


def materialize_all(provenance_inputs: Mapping[str, str]) -> dict[str, dict]:
    """Materialize all nine frozen K configs in batch-contract order."""
    return {k_id: materialize_combination(k_id, provenance_inputs) for k_id in sorted(K_MEMBERS)}


def validate_frozen_combination(doc: Mapping) -> dict:
    """Fail closed unless a materialized K doc equals the frozen contract."""
    if not isinstance(doc, Mapping) or set(doc) != set(_TOP_LEVEL_KEYS):
        raise ValueError("combination doc must carry exactly the frozen top-level fields")
    if doc["schema_id"] != SCHEMA_ID or doc["protocol_id"] != PROTOCOL_ID:
        raise ValueError("combination doc must identify protocol-v4 and schema-v4")
    k_id = doc["arm_id"]
    expected_members = members_for(k_id)
    if doc["arm_kind"] != "combination":
        raise ValueError("K arm_kind must be combination")
    expected_components = [MEMBER_IMPLEMENTATION[a]["component"] for a in expected_members]
    if list(doc["component_set"]) != expected_components:
        raise ValueError(f"{k_id} component_set is not the frozen K membership")
    if list(doc["member_arm_ids"]) != list(expected_members):
        raise ValueError(f"{k_id} member_arm_ids omit, duplicate, or substitute selected arms")
    if set(doc["members"]) != set(expected_members):
        raise ValueError(f"{k_id} members omit, duplicate, or substitute selected arms")
    for arm_id in expected_members:
        entry = doc["members"][arm_id]
        if not isinstance(entry, Mapping) or set(entry) != set(_MEMBER_KEYS):
            raise ValueError(f"{k_id} member {arm_id} carries non-frozen fields")
        frozen = MEMBER_IMPLEMENTATION[arm_id]
        for key in _MEMBER_KEYS:
            if entry[key] != frozen[key]:
                raise ValueError(f"{k_id} member {arm_id} field {key} overrides the frozen implementation")
        if selected_arm_for_component(entry["component"]) != arm_id:
            raise ValueError(f"{k_id} member {arm_id} is not the frozen selected arm")
    if doc["global"] != GLOBAL_SETTINGS:
        raise ValueError(f"{k_id} global settings differ: per-combination tuning is forbidden")
    eligibility = doc["eligibility"]
    if (
        eligibility["qualification"] != QUALIFICATION
        or eligibility["waiver_id"] != WAIVER_ID
        or eligibility["waiver_scope"] != WAIVER_SCOPE
    ):
        raise ValueError(f"{k_id} eligibility is not the exact protocol-v4 waiver tuple")
    if doc["no_execution"] is not True or doc["confirmation_sealed_access"] is not False:
        raise ValueError(f"{k_id} must record no execution and no Confirmation/Sealed access")
    unsigned = {k: v for k, v in doc.items() if k != "combination_sha256"}
    if doc["combination_sha256"] != canonical_hash(unsigned):
        raise ValueError(f"{k_id} combination_sha256 does not match canonical recomputation")
    return doc
