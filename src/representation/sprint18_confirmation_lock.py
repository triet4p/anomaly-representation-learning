"""Sprint 18 Task 33 - immutable pre-Confirmation matrix lock.

Execution-free composition surface for the one-shot Sprint 18 Confirmation
comparison.  It binds the complete eight-arm matrix exactly once, before any
Confirmation score exists:

* the eight registered arm identities, including ``N1-HCC`` as an explicit
  ``STOPPED`` / non-executable arm with no invented config or checkpoint;
* the seven executable final configurations for model seeds
  ``181801``/``181802``/``181803`` from the accepted Task 14 v1 lock and, for
  ``N4-CRN`` only, the accepted scientific v9 lock plus the same-seed paired B0
  authority;
* every primary checkpoint byte reference, restored model/EMA state digest and
  (for ``N4-CRN``) the coherent artifact and final scale envelope;
* the eligible Fit support, the per-arm/per-seed historical bank digest and the
  append-only custody-restored ``bank.pt``/``provenance.json`` bytes;
* the frozen per-branch Calibration q95 thresholds;
* the exact protocol/spec/role-handoff/metric/compute-counter identities;
* the frozen scoring metric and uncertainty semantics;
* all 16 role-manifest identities and the four untouched Confirmation support
  manifests; and
* the local and remote artifact custody maps.

Non-touch guarantees (fail-closed, exercised by focused tests):

* No Confirmation feature, label, score, metric or derived outcome is read,
  parsed or stored.  Confirmation enters only as metadata already published in
  the certified Task 71 handoff (history ID, role, data seed, manifest and
  waveform digests, sample/shard counts, config hash, path).  The lock carries
  no Confirmation (or Development) outcome value.
* Confirmation support content is never opened to precompute a support digest;
  the manifest SHA-256 plus the frozen common-support rule is the sole support
  authority until the first locked scoring.
* No config, checkpoint, bank, threshold, standardizer, normalization or
  support is invented, refit, reselected or silently omitted.  Historical arm
  source identities stay distinct from the supplementary implementation source
  and from this lock's publisher.
* The lock digest is canonical JSON SHA-256 over every field except the digest
  itself, so any edit - including one that also repairs the record - changes
  the digest and is rejected by the validator.

The public consumer path is:

``python -m representation.sprint18_confirmation_lock build``

and

``python -m representation.sprint18_confirmation_lock validate --lock <path>``

Downstream Confirmation workers must call :func:`load_lock`/:func:`arm_state`
and pass ``lock["lock_sha256"]`` to the unchanged harness
``validate_role_permission(..., confirmation_lock_sha256=...)``; they must not
re-derive any of the frozen state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from representation.sprint18_harness import (
    MECHANISM_SPEC_SHA256,
    MODEL_SEEDS,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    Sprint18ArmConfig,
    Task6Binding,
    arm_config_sha256,
    arm_spec,
    load_task6_binding,
)

__all__ = [
    "SCHEMA_ID",
    "LOCK_PATH",
    "ARM_IDS",
    "EXECUTABLE_ARM_IDS",
    "N1_ARM_ID",
    "N4_ARM_ID",
    "B0_ARM_ID",
    "CONFIRMATION_HISTORY_IDS",
    "ConfirmationLockError",
    "canonical_hash",
    "lock_sha256",
    "materialize_lock",
    "validate_lock",
    "load_lock",
    "arm_state",
    "confirmation_support",
    "main",
]

TASK_ID = "S18-T33"
ATTEMPT_ID = "S18T33A01"
AGENT_ID = "S18T33A01"
ROLE = "silver-task"
SCHEMA_ID = "sprint18-confirmation-lock-v1"
LOCK_PATH = "experiments/sprint18-task33-confirmation-lock.json"
LOCK_STATE = "FROZEN_PENDING_INDEPENDENT_REVIEW"

ARM_IDS = (
    "B0-FULL",
    "K3-FULL",
    "K7-FULL",
    "K9-FULL",
    "N1-HCC",
    "N2-MSP",
    "N3-CCRP",
    "N4-CRN",
)
B0_ARM_ID = "B0-FULL"
K9_ARM_ID = "K9-FULL"
N1_ARM_ID = "N1-HCC"
N4_ARM_ID = "N4-CRN"
EXECUTABLE_ARM_IDS = tuple(arm for arm in ARM_IDS if arm != N1_ARM_ID)

FIT_HISTORY_IDS = (
    "S18I-ITER-0005-FIT-01",
    "S18I-ITER-0005-FIT-02",
    "S18I-ITER-0005-FIT-03",
)
CALIBRATION_HISTORY_ID = "S18I-ITER-0005-CALIBRATION-01"
DEVELOPMENT_HISTORY_IDS = (
    "S18I-ITER-0005-DEVELOPMENT-01",
    "S18I-ITER-0005-DEVELOPMENT-02",
    "S18I-ITER-0005-DEVELOPMENT-03",
    "S18I-ITER-0005-DEVELOPMENT-04",
)
CONFIRMATION_HISTORY_IDS = (
    "S18I-ITER-0005-CONFIRMATION-01",
    "S18I-ITER-0005-CONFIRMATION-02",
    "S18I-ITER-0005-CONFIRMATION-03",
    "S18I-ITER-0005-CONFIRMATION-04",
)
_SCHEMA_HISTORY_PREFIX = {
    "FIT": "H-S18-FIT",
    "CALIBRATION": "H-S18-CAL",
    "DEVELOPMENT": "H-S18-DEV",
    "CONFIRMATION": "H-S18-CONF",
}

FIT_ROWS = 5446
FIT_VALID_PATCHES = 167419
FIT_ROWS_SHA256 = "447b219e0c77f6d76ed54a4de7a0bee7c2b3a9a8142f61cbb434694ed73acb94"
CALIBRATION_ROWS = 1877
CALIBRATION_VALID_PATCHES = 56889
CALIBRATION_SUPPORT_SHA256 = "ad5310425f2d5737576a5fe34efc7389e2c07714d68cc188d96199ca4e4f5568"
CALIBRATION_QUANTILE = 0.95

PROTOCOL_PATH = "experiments/sprint18-full-training-protocol-v2.md"
PROTOCOL_V1_BASE_PATH = "experiments/sprint18-full-training-protocol-v1.md"
PROTOCOL_V1_BASE_SHA256 = "9ae3d43f94a6194bae602d68067958bdf7d11cf47e801dd328e87b22e7907cf6"
MECHANISM_SPEC_PATHS = {
    "N1-HCC": "experiments/sprint18-hierarchical-conditional-contrastive-spec-v2.md",
    "N2-MSP": "experiments/sprint18-multiscale-latent-prediction-spec-v2.md",
    "N3-CCRP": "experiments/sprint18-cross-channel-relational-prediction-spec-v2.md",
    "N4-CRN": "experiments/sprint18-context-residual-normalization-spec-v2.md",
}
CONTRACT_FILES: dict[str, str] = {
    PROTOCOL_PATH: PROTOCOL_SHA256,
    PROTOCOL_V1_BASE_PATH: PROTOCOL_V1_BASE_SHA256,
    **{path: MECHANISM_SPEC_SHA256[arm] for arm, path in MECHANISM_SPEC_PATHS.items()},
    "experiments/sprint18-supplementary-development-metrics-v1.md": "44916f4bd1bd5e56b0b04873936de5fa1e6e1541812200f8d2dc6739bda41067",
    "experiments/sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json": "5c28434b6f5b3342a3203a73cd672782e08857299c05b709adbd8bce75d45155",
}

TASK14_LOCK_PATH = "experiments/sprint18-task14-config-lock.json"
TASK14_LOCK_SHA256 = "c4c1a8bde14310cf4436c10762e3e98475f7c84eb74fcb8a74b3e267885edec3"
N4_SCIENTIFIC_LOCK_PATH = "experiments/sprint18-task14-config-lock-v9.json"
N4_SCIENTIFIC_LOCK_SHA256 = "a9990bd189706a0fdadfeaee422ef307599e3f4d9c5a72102dfa1f015d5cd57a"
N4_V11_REPORT_ONLY_LOCK_SHA256 = "ba93edda451fcf66607a49b00a88c5f9ad54e9af0f581e8ae568cf482b9bfed7"
N4_SCIENTIFIC_SOURCE_COMMIT = "6ffcfc9f4ab339604ff201abb75f9d18fb12466d"
N4_V9_LOCK_SOURCE_COMMIT = "1d9a646e9f57051fdc582e950919905f1dd70326"
N4_REPORT_ONLY_SOURCE_COMMIT = "1cc11165852fa90a6c4284725dadd9e234e7381f"

HANDOFF_PATH = "experiments/sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
HANDOFF_SHA256 = CONTRACT_FILES[HANDOFF_PATH]
DEVELOPMENT_MATRIX_PATH = "artifacts/sprint-18/s18-t31-a01/development-matrix.json"
DEVELOPMENT_MATRIX_SHA256 = "849f2a1dfc8f35e6a6b88b7edf2ab7e7711a314b4860b26865d7a9d0442fd627"
N1_SELECTION_PATH = "experiments/sprint18-task10-n1-hcc-selection-v1.json"
N1_SELECTION_SHA256 = "5184419956ccfa5a2b4667a2eb95f3429e6bb72be1dedd582ddb2c8ca381a967"
N1_STOPPED_ROW_PATH = "artifacts/sprint-18/s18-t27-a02/n1-hcc-stopped-row.json"
N1_STOPPED_ROW_SHA256 = "1756d7b85fdde915c17e9a9253dc4b5bf3207b8f11d2f835f0c2fc2da670009a"
N1_EVIDENCE_PATH = "artifacts/sprint-18/task-27-a02.md"
N1_EVIDENCE_SHA256 = "1a0ded932ee7e03a57993b24a97f3e1fc2fe645611ee7a3d57d45b4617f96c18"
DATA_BINDING_SHA256 = "383c62114ee6120d5fd19f866dfd585083d1d5899c9ffe91804761bd8a10a2d5"
PRIMARY_REPRESENTATION_EPOCH = 100
PRIMARY_N4_SCALE_EPOCH = 50

SUPPLEMENTARY_RUNNER_PATH = "experiments/sprint18_task32_supplementary_runner.py"
SUPPLEMENTARY_RUNNER_SHA256 = "6e2a9fc58ca5bdc45b86481af391368710abf45d05d85864bbb19eae86aba242"
SUPPLEMENTARY_SOURCE_COMMIT = "c1996ea17e639b4d9df7a1f08c2407919bd2bfc6"
SUPPLEMENTARY_SOURCE_TREE = "5960b5204f1dd8c7477b2aa631319a2a7201d050"
SUPPLEMENTARY_SOURCE_BRANCH = "sprint18-task68-a02-runtime"
SUPPLEMENTARY_SOURCE_WORKTREE = "/home/trietlm/anomaly-representation-learning-s18t68-a02-worktree"
METRIC_CODE_PATH = "src/representation/sprint18_supplementary.py"
METRIC_CODE_SHA256 = "dd7f9ac6b934a1036bfba94f85cfd8e3778fd95549f965ee88da55ce5737b4b6"

MANIFEST_CATALOG_DIR = "artifacts/sprint-18/s18-t32-runtime-a02/manifestcatalog"
MANIFEST_AUDIT_PATH = "artifacts/sprint-18/s18-t32-runtime-a02/auditsha/manifest-sha256.txt"
BANK_AUDIT_PATH = "artifacts/sprint-18/s18-t32-runtime-a02/auditsha/bank-artifact-sha256.txt"
SCORE_AUDIT_PATH = "artifacts/sprint-18/s18-t32-runtime-a02/auditsha/score-document-sha256.txt"
METRIC_AUDIT_PATH = "artifacts/sprint-18/s18-t32-runtime-a02/auditsha/metric-document-sha256.txt"

CODE_FILES: dict[str, str] = {
    "src/representation/sprint18_harness.py": "261d1479159f51de11cd2e1b81a6b07358f0d36045b495de2003379b2b77f515",
    "src/representation/sprint18_supplementary.py": METRIC_CODE_SHA256,
    "src/representation/sprint18_k9_downstream.py": "54997d0d6174c97841dfc549502f4da64dcc45b69a1f3fd9d97b4b8f3cf3cceb",
    "src/representation/sprint18_checkpoint.py": "fce263df093bc622ba0a69bfac0824941fd6a0d1c4c9ab02bcbcff62432aaef2",
    "src/representation/crn.py": "7902dd9b7209b2ad9f7ef0f0e6f7316eb13c7db20237bc04974336b863bb9251",
}
PUBLISHER_CODE_PATH = "src/representation/sprint18_confirmation_lock.py"

PRIMARY_METRICS = (
    "P_AUROC",
    "W_AUROC",
    "PW_AUROC",
    "A_AUROC",
    "P_recall",
    "P_median_lead_days",
    "W_recall",
    "W_median_lead_days",
    "FAR_per_robot_day",
    "false_episodes",
)
METRIC_FAMILIES = (
    "background_stability",
    "localization",
    "duration_accounting",
    "severity_ordering",
    "subtype_slices",
)

CONFIRMATION_GATES: dict[str, Any] = {
    "minimum_paired_delta_pw": 0.05,
    "minimum_positive_histories": 3,
    "histories_total": 4,
    "post_arm_lcb95_min": 0.55,
    "directional_count_min": 3,
    "noninferiority_delta_p": -0.02,
    "noninferiority_delta_w": -0.02,
    "p_recall_min": 0.5,
    "p_median_lead_days_min": 1.0,
    "w_recall_min": 0.25,
    "w_median_lead_days_min": 0.5,
    "false_alert_max_per_robot_day": 0.05,
    "paired_delta_far_max": 0.02,
    "background_stability_max": 0.10,
    "rounding": "none before comparison",
}

UNCERTAINTY: dict[str, Any] = {
    "history_block_bootstrap": {
        "draws": 2000,
        "seed": 20260202,
        "quantile_method": "numpy linear",
        "units": "four evaluation histories",
        "lcb_percentile": 2.5,
        "descriptive_interval_percentiles": [2.5, 97.5],
        "pairing": (
            "pair arm-minus-B0 within each seed/history; average the three seed "
            "values within each history; macro is the unweighted four-history mean"
        ),
    },
    "paired_delta_semantics": "point estimates only; no paired confidence interval is claimed",
    "lcb_semantics": (
        "post-arm history-block LCB95 of P+W AUROC; an within-arm PW_lcb95 value "
        "or a difference between such values is not a paired confidence interval"
    ),
    "background_uncertainty_semantics": (
        "[2.5, 97.5] percentile interval of the four seed-averaged per-history "
        "stability ratios; it does not relax the per-history <=0.10 gate"
    ),
    "no_significance_claim": True,
}

FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "P_AUROC",
        "W_AUROC",
        "PW_AUROC",
        "A_AUROC",
        "P_recall",
        "W_recall",
        "P_median_lead_days",
        "W_median_lead_days",
        "FAR_per_robot_day",
        "false_episodes",
        "delta_P",
        "delta_W",
        "delta_PW",
        "delta_FAR",
        "background_stability_ratio",
        "localization",
        "severity_ordering",
        "subtype_slices",
        "failure_events",
        "score_map_sha256",
        "score_document_sha256",
    }
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_id",
        "task_id",
        "attempt_id",
        "agent_id",
        "role",
        "phase",
        "lock_state",
        "confirmation_open",
        "publisher",
        "protocol",
        "supplementary_source",
        "historical_sources",
        "common_contract",
        "scoring",
        "support",
        "arms",
        "n4_authority",
        "status_matrix",
        "artifact_custody",
        "validation_rules",
        "lock_sha256",
    }
)
_TASK14_DIGEST_EXCLUDED = frozenset(
    {"coefficient_lock_sha256", "locked_configs", "config_sha256", "file_sha256"}
)


class ConfirmationLockError(ValueError):
    """Raised when the lock or its custody is incomplete, inconsistent or tampered."""


def _expect(condition: object, message: str) -> None:
    if not condition:
        raise ConfirmationLockError(message)


def canonical_hash(value: object) -> str:
    """SHA-256 of canonical JSON (sorted keys, compact separators, UTF-8)."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def lock_sha256(lock: Mapping[str, object]) -> str:
    """Canonical digest of the lock payload, excluding the digest field itself."""
    payload = {key: value for key, value in lock.items() if key != "lock_sha256"}
    return canonical_hash(payload)


def _resolve_root(repository_root: str | Path | None) -> Path:
    if repository_root is None:
        return Path(__file__).resolve().parents[2]
    root = Path(repository_root).resolve()
    _expect((root / PROTOCOL_PATH).is_file(), f"repository root has no protocol: {root}")
    return root


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pinned_json(root: Path, relative: str, expected_sha256: str) -> dict[str, Any]:
    path = root / relative
    _expect(path.is_file(), f"required evidence file is missing: {relative}")
    observed = _sha_file(path)
    _expect(
        observed == expected_sha256,
        f"{relative} bytes differ from the accepted identity: {observed} != {expected_sha256}",
    )
    document = json.loads(path.read_bytes().decode("utf-8"))
    _expect(isinstance(document, dict), f"{relative} is not a JSON object")
    return cast(dict[str, Any], document)


def _sha_catalog(path: Path) -> dict[str, str]:
    """Parse a ``sha256sum`` catalog into ``{path: digest}``."""
    _expect(path.is_file(), f"required custody catalog is missing: {path}")
    catalog: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        _expect(len(parts) == 2, f"malformed custody catalog line {number}: {raw_line}")
        digest, target = parts[0], parts[1].strip()
        _expect(len(digest) == 64 and target, f"malformed custody catalog line {number}")
        catalog[target] = digest
    return catalog


def _task14_identity(
    root: Path,
    relative: str,
    expected_raw_sha256: str,
    label: str,
) -> dict[str, Any]:
    document = _pinned_json(root, relative, expected_raw_sha256)
    core = {
        key: value
        for key, value in document.items()
        if key not in _TASK14_DIGEST_EXCLUDED
    }
    core_digest = canonical_hash(core)
    _expect(
        core_digest == document.get("coefficient_lock_sha256"),
        f"{label} config-lock core digest does not reproduce",
    )
    configs = document.get("locked_configs")
    digests = document.get("config_sha256")
    _expect(isinstance(configs, dict) and isinstance(digests, dict), f"{label} lock is malformed")
    _expect(set(configs) == set(digests), f"{label} lock config/digest keys differ")
    for key, dump in configs.items():
        config = Sprint18ArmConfig.model_validate(dump)
        _expect(config.selection_state == "locked", f"{label} config {key} is not locked")
        _expect(config.protocol_sha256 == PROTOCOL_SHA256, f"{label} config {key} protocol differs")
        _expect(
            arm_config_sha256(config) == digests[key],
            f"{label} config {key} does not reproduce its digest",
        )
    return {
        "path": relative,
        "sha256": expected_raw_sha256,
        "core_sha256": core_digest,
        "locked_config_count": len(configs),
    }


def _contract_identity(root: Path) -> dict[str, Any]:
    for relative, expected in CONTRACT_FILES.items():
        _expect(
            _sha_file(root / relative) == expected,
            f"contract file bytes differ from the accepted identity: {relative}",
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_path": PROTOCOL_PATH,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_v1_base_path": PROTOCOL_V1_BASE_PATH,
        "protocol_v1_base_sha256": PROTOCOL_V1_BASE_SHA256,
        "mechanism_specs": {
            arm: {"path": MECHANISM_SPEC_PATHS[arm], "sha256": MECHANISM_SPEC_SHA256[arm]}
            for arm in MECHANISM_SPEC_PATHS
        },
        "data_handoff": {"path": HANDOFF_PATH, "sha256": HANDOFF_SHA256},
        "supplementary_definition": {
            "path": "experiments/sprint18-supplementary-development-metrics-v1.md",
            "sha256": CONTRACT_FILES["experiments/sprint18-supplementary-development-metrics-v1.md"],
        },
    }


def _manifest_catalog(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    audit = _sha_catalog(root / MANIFEST_AUDIT_PATH)
    audit_by_name = {Path(path).name: (path, digest) for path, digest in audit.items()}
    files = sorted((root / MANIFEST_CATALOG_DIR).glob("*.json"))
    _expect(len(files) == 21, f"expected 21 accepted input manifests, found {len(files)}")
    catalog: dict[tuple[str, int], dict[str, Any]] = {}
    for path in files:
        _expect(path.name in audit_by_name, f"manifest is absent from the custody catalog: {path.name}")
        remote_path, expected_sha256 = audit_by_name[path.name]
        relative = path.relative_to(root).as_posix()
        observed = _sha_file(path)
        _expect(observed == expected_sha256, f"local manifest copy differs from remote custody: {path.name}")
        stem = path.stem
        _expect("-" in stem, f"manifest name has no arm/seed separator: {path.name}")
        arm_id, seed_text = stem.rsplit("-", 1)
        _expect(arm_id in EXECUTABLE_ARM_IDS, f"manifest names a non-executable arm: {arm_id}")
        seed = int(seed_text)
        _expect(seed in MODEL_SEEDS, f"manifest names an unregistered model seed: {seed}")
        _expect((arm_id, seed) not in catalog, f"duplicate manifest for {arm_id}/{seed}")
        document = json.loads(path.read_bytes().decode("utf-8"))
        _expect(document.get("schema_id") == "sprint18-supplementary-input-manifest-v3", f"manifest schema differs: {path.name}")
        _expect(document.get("task_id") == "S18-T32", f"manifest task differs: {path.name}")
        _expect(document.get("phase") == "development", f"manifest phase differs: {path.name}")
        _expect(document.get("scope") == "experiment", f"manifest scope differs: {path.name}")
        _expect(document.get("source_commit") == SUPPLEMENTARY_SOURCE_COMMIT, f"manifest source commit differs: {path.name}")
        _expect(document.get("protocol_sha256") == PROTOCOL_SHA256, f"manifest protocol differs: {path.name}")
        _expect(document.get("metric_code_sha256") == METRIC_CODE_SHA256, f"manifest metric code differs: {path.name}")
        _expect(
            cast(Mapping[str, Any], document.get("expected", {})).get("data_binding_sha256")
            == DATA_BINDING_SHA256,
            f"manifest data binding differs: {path.name}",
        )
        expected_lock = N4_SCIENTIFIC_LOCK_SHA256 if arm_id == N4_ARM_ID else TASK14_LOCK_SHA256
        _expect(
            isinstance(document.get("task14_lock"), dict)
            and document["task14_lock"].get("sha256") == expected_lock,
            f"manifest Task 14 authority differs: {path.name}",
        )
        catalog[(arm_id, seed)] = {
            "path": relative,
            "sha256": observed,
            "remote_path": remote_path,
            "document": document,
        }
    expected_pairs = {
        (arm_id, seed) for arm_id in EXECUTABLE_ARM_IDS for seed in MODEL_SEEDS
    }
    _expect(set(catalog) == expected_pairs, "accepted manifest coverage is incomplete")
    return catalog


def _dev_support_identity(
    matrix: Mapping[str, Any],
    manifests: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    b0_support = cast(Mapping[str, Any], matrix["arms"][B0_ARM_ID]["support"])
    per_history = cast(Mapping[str, Any], b0_support["per_history"])
    expected_manifests: dict[str, str] | None = None
    expected_supports: dict[str, str] | None = None
    result: dict[str, Any] = {}
    for history_id in DEVELOPMENT_HISTORY_IDS:
        support_sha = per_history[history_id]["support_sha256"]
        role = "DEVELOPMENT"
        ordinal = DEVELOPMENT_HISTORY_IDS.index(history_id) + 1
        schema_history_id = f"{_SCHEMA_HISTORY_PREFIX[role]}-{ordinal:02d}"
        result[history_id] = {
            "schema_history_id": schema_history_id,
            "support_sha256": support_sha,
        }
        for arm_id in EXECUTABLE_ARM_IDS:
            support = cast(Mapping[str, Any], matrix["arms"][arm_id]["support"])
            if arm_id == B0_ARM_ID:
                observed = per_history[history_id]["support_sha256"]
            else:
                observed = cast(Mapping[str, str], support["per_history_support_sha256"])[history_id]
            _expect(
                observed == support_sha,
                f"Development support differs across arms for {history_id}",
            )
    for arm_id in EXECUTABLE_ARM_IDS:
        for seed in MODEL_SEEDS:
            expected = cast(Mapping[str, Any], manifests[(arm_id, seed)]["document"]["expected"])
            support_map = cast(Mapping[str, str], expected["development_support_sha256"])
            manifest_map = cast(Mapping[str, str], expected["development_manifest_sha256"])
            if expected_supports is None:
                expected_supports = dict(support_map)
            if expected_manifests is None:
                expected_manifests = dict(manifest_map)
            _expect(dict(support_map) == expected_supports, "Development support map is not uniform")
            _expect(dict(manifest_map) == expected_manifests, "Development manifest map is not uniform")
    assert expected_manifests is not None and expected_supports is not None
    for history_id in DEVELOPMENT_HISTORY_IDS:
        _expect(
            result[history_id]["support_sha256"] == expected_supports[history_id],
            f"Development support differs between matrix and manifests: {history_id}",
        )
        result[history_id]["manifest_sha256"] = expected_manifests[history_id]
    return result


def _confirmation_support_identity(
    handoff: Mapping[str, Any], binding: Task6Binding
) -> dict[str, Any]:
    roster = handoff.get("roster")
    _expect(isinstance(roster, list) and len(roster) == 16, "certified roster is not the frozen 16 histories")
    by_history: dict[str, Mapping[str, Any]] = {}
    for row in cast(Sequence[Mapping[str, Any]], roster):
        history_id = row.get("history_id")
        _expect(isinstance(history_id, str), "certified roster row has no history id")
        by_history[cast(str, history_id)] = row
    result: dict[str, Any] = {}
    for history_id in CONFIRMATION_HISTORY_IDS:
        _expect(history_id in by_history, f"Confirmation history is absent from the certified roster: {history_id}")
        row = by_history[history_id]
        _expect(row.get("assigned_role") == "CONFIRMATION", f"role differs for {history_id}")
        ordinal = CONFIRMATION_HISTORY_IDS.index(history_id) + 1
        result[history_id] = {
            "schema_history_id": f"{_SCHEMA_HISTORY_PREFIX['CONFIRMATION']}-{ordinal:02d}",
            "manifest_sha256": row["manifest_sha256"],
            "waveform_stream_sha256": row["waveform_stream_sha256"],
            "sample_count": row["sample_count"],
            "shard_count": row["shard_count"],
            "data_seed": row["data_seed"],
            "config_hash": row["config_hash"],
            "path_relative": row["path_relative"],
            "data_permitted_use": row["permitted_use"],
            "model_permitted_use": binding.history(history_id).model_permitted_use,
        }
    return result


def _role_manifests(handoff: Mapping[str, Any], binding: Task6Binding) -> dict[str, Any]:
    roster = cast(Sequence[Mapping[str, Any]], handoff["roster"])
    return {
        cast(str, row["history_id"]): {
            "assigned_role": row["assigned_role"],
            "manifest_sha256": row["manifest_sha256"],
            "waveform_stream_sha256": row["waveform_stream_sha256"],
            "sample_count": row["sample_count"],
            "shard_count": row["shard_count"],
            "data_seed": row["data_seed"],
            "config_hash": row["config_hash"],
            "path_relative": row["path_relative"],
            "data_permitted_use": row["permitted_use"],
            "model_permitted_use": binding.history(cast(str, row["history_id"])).model_permitted_use,
        }
        for row in roster
    }


def _seed_run(arm_document: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    seed_runs = arm_document.get("seed_runs")
    _expect(isinstance(seed_runs, list), "accepted arm record has no seed runs")
    matches = [row for row in cast(Sequence[Mapping[str, Any]], seed_runs) if row.get("model_seed") == seed]
    _expect(len(matches) == 1, f"accepted arm record does not contain exactly one seed-{seed} run")
    return matches[0]


def _bank_files(audit: Mapping[str, str], arm_id: str, seed: int) -> dict[str, str]:
    suffix = f"/{arm_id}-{seed}.json.fit-bank/"
    bank = [path for path in audit if path.endswith(suffix + "bank.pt")]
    provenance = [path for path in audit if path.endswith(suffix + "provenance.json")]
    _expect(len(bank) == 1, f"bank custody is missing or ambiguous for {arm_id}/{seed}")
    _expect(len(provenance) == 1, f"bank provenance custody is missing or ambiguous for {arm_id}/{seed}")
    return {
        "artifact_path": bank[0],
        "artifact_sha256": audit[bank[0]],
        "provenance_path": provenance[0],
        "provenance_sha256": audit[provenance[0]],
    }


def _scientific_source(arm_document: Mapping[str, Any]) -> dict[str, Any]:
    source = arm_document.get("scientific_source")
    _expect(isinstance(source, dict), "executable arm is missing its historical scientific source")
    source = cast(Mapping[str, Any], source)
    for key in ("source_commit", "source_tree", "runner_sha256", "binding_sha256", "coefficient_lock_sha256"):
        _expect(isinstance(source.get(key), str), f"historical scientific source lacks {key}")
    return dict(source)


def _seed_entry(
    arm_id: str,
    seed: int,
    arm_document: Mapping[str, Any],
    manifest_entry: Mapping[str, Any],
    task14_configs: Mapping[str, str],
    bank_audit: Mapping[str, str],
    calibration_manifest_sha256: str,
) -> dict[str, Any]:
    document = cast(Mapping[str, Any], manifest_entry["document"])
    expected = cast(Mapping[str, Any], document["expected"])
    run = _seed_run(arm_document, seed)
    _expect(run.get("record_status") == "VALID", f"{arm_id}/{seed} is not an accepted VALID record")
    _expect(list(cast(Sequence[str], run.get("failure_reasons", []))) == [], f"{arm_id}/{seed} carries failure reasons")

    config_key = f"{arm_id}/seed-{seed}"
    b0_config_key = f"{B0_ARM_ID}/seed-{seed}"
    _expect(config_key in task14_configs, f"no accepted config for {config_key}")
    _expect(
        expected.get("arm_config_sha256") == task14_configs[config_key],
        f"{arm_id}/{seed} manifest config digest differs from the accepted lock",
    )

    reference_artifact: dict[str, Any] | None
    coherent_artifact: dict[str, Any] | None = None
    if arm_id == N4_ARM_ID:
        primary = cast(Mapping[str, Any], run["primary"])
        representation = cast(Mapping[str, Any], primary["representation_primary"])
        scale = cast(Mapping[str, Any], primary["scale_primary"])
        _expect(
            representation.get("paired_b0_checkpoint_sha256") == document["checkpoint"]["sha256"],
            f"N4/{seed} paired B0 checkpoint differs from the accepted manifest",
        )
        _expect(
            representation.get("restored_model_state_sha256") == expected.get("model_state_sha256"),
            f"N4/{seed} restored model state differs from the accepted digest",
        )
        _expect(
            representation.get("restored_ema_state_sha256") == expected.get("ema_state_sha256"),
            f"N4/{seed} restored EMA state differs from the accepted digest",
        )
        _expect(scale.get("path") == document["reference_artifact"]["path"], f"N4/{seed} scale path differs")
        _expect(scale.get("sha256") == document["reference_artifact"]["sha256"], f"N4/{seed} scale artifact differs")
        thresholds = {
            "S_pop": cast(Mapping[str, Any], run["calibration"])["n4_thresholds"]["S_pop"],
            "S_pred": cast(Mapping[str, Any], run["calibration"])["n4_thresholds"]["S_pred"],
        }
        coherent_artifact = {
            "path": representation.get("path"),
            "sha256": representation.get("sha256"),
            "restored_checkpoint_sha256": representation.get("restored_checkpoint_sha256"),
            "scale_model_state_sha256": representation.get("scale_model_state_sha256"),
            "state_dict_sha256": scale.get("state_dict_sha256"),
            "scale_epochs_completed": representation.get("completed_epoch"),
            "scale_global_step": representation.get("global_step"),
            "representation_step": representation.get("representation_step"),
            "paired_b0_representation_step": representation.get("paired_b0_representation_step"),
            "strict_restore_b0_model_ema_scale_equal": representation.get(
                "strict_restore_b0_model_ema_scale_equal"
            ),
        }
        reference_artifact = {
            "kind": document["reference_artifact"]["kind"],
            "path": document["reference_artifact"]["path"],
            "sha256": document["reference_artifact"]["sha256"],
            "fit_scope": document["reference_artifact"]["fit_scope"],
            "bounds": document["reference_artifact"]["bounds"],
            "state_dict_sha256": scale.get("state_dict_sha256"),
            "schema": scale.get("schema"),
            "parameterization": scale.get("parameterization"),
            "guard_contract": scale.get("guard_contract"),
        }
        active_epoch = PRIMARY_REPRESENTATION_EPOCH
        global_step = representation.get("paired_b0_representation_step")
    else:
        primary = cast(Mapping[str, Any], run["primary"])
        _expect(primary.get("path") == document["checkpoint"]["path"], f"{arm_id}/{seed} checkpoint path differs")
        _expect(
            primary.get("restored_checkpoint_sha256") == document["checkpoint"]["sha256"],
            f"{arm_id}/{seed} restored checkpoint differs from the accepted digest",
        )
        _expect(
            primary.get("live_model_state_sha256") == expected.get("model_state_sha256"),
            f"{arm_id}/{seed} live model state differs from the accepted digest",
        )
        _expect(
            primary.get("live_ema_state_sha256") == expected.get("ema_state_sha256"),
            f"{arm_id}/{seed} live EMA state differs from the accepted digest",
        )
        calibration = cast(Mapping[str, Any], run["calibration"])
        thresholds = {
            "S_pop": calibration["S_pop_threshold"],
            "S_pred": calibration["S_pred_threshold"],
        }
        reference = document.get("reference_artifact")
        _expect(isinstance(reference, dict), f"{arm_id}/{seed} reference artifact section is malformed")
        reference_artifact = (
            None
            if reference["kind"] == "none"
            else {
                "kind": reference["kind"],
                "path": reference["path"],
                "sha256": reference["sha256"],
                "fit_scope": reference["fit_scope"],
                "bounds": reference["bounds"],
            }
        )
        active_epoch = primary.get("active_epoch")
        global_step = primary.get("global_step")

    manifest_thresholds = cast(Mapping[str, Any], expected["calibration_thresholds"])
    _expect(
        float(thresholds["S_pop"]) == float(manifest_thresholds["S_pop"]),
        f"{arm_id}/{seed} S_pop threshold differs from the accepted manifest",
    )
    _expect(
        float(thresholds["S_pred"]) == float(manifest_thresholds["S_pred"]),
        f"{arm_id}/{seed} S_pred threshold differs from the accepted manifest",
    )

    _expect(
        int(cast(int, expected["bank_source_rows"])) == FIT_ROWS,
        f"{arm_id}/{seed} bank row count differs from the frozen Fit support",
    )
    _expect(
        expected["bank_source_rows_sha256"] == FIT_ROWS_SHA256,
        f"{arm_id}/{seed} bank row digest differs from the frozen Fit support",
    )
    matrix_bank = run.get("fit_bank")
    if arm_id == K9_ARM_ID:
        _expect(matrix_bank is None, "K9 accepted record unexpectedly carries a reconstructed bank")
    else:
        _expect(isinstance(matrix_bank, dict), f"{arm_id}/{seed} accepted record has no bank custody")
        matrix_bank = cast(Mapping[str, Any], matrix_bank)
        _expect(
            matrix_bank.get("bank_sha256") == expected["bank_sha256"],
            f"{arm_id}/{seed} bank digest differs from the accepted manifest",
        )
        _expect(
            matrix_bank.get("source_rows") == expected["bank_source_rows"],
            f"{arm_id}/{seed} bank source rows differ",
        )
        _expect(
            matrix_bank.get("source_rows_sha256") == expected["bank_source_rows_sha256"],
            f"{arm_id}/{seed} bank source row digest differs",
        )
        _expect(
            matrix_bank.get("checkpoint_sha256") == document["checkpoint"]["sha256"],
            f"{arm_id}/{seed} bank checkpoint differs",
        )

    if arm_id == K9_ARM_ID:
        _expect(isinstance(reference_artifact, dict), "K9 downstream reference artifact is missing")
        reference_artifact = cast(dict[str, Any], reference_artifact)
        bank_custody = {
            "artifact_path": reference_artifact["path"],
            "artifact_sha256": reference_artifact["sha256"],
            "provenance_path": None,
            "provenance_sha256": None,
        }
    else:
        bank_custody = _bank_files(bank_audit, arm_id, seed)
    _expect(
        expected.get("checkpoint_arm_config_sha256")
        == (task14_configs[b0_config_key] if arm_id == N4_ARM_ID else task14_configs[config_key]),
        f"{arm_id}/{seed} checkpoint config authority differs",
    )

    _expect(document["accepted_record"]["path"], f"{arm_id}/{seed} accepted record path is empty")
    _expect(
        len(cast(str, document["accepted_record"]["sha256"])) == 64,
        f"{arm_id}/{seed} accepted record digest is malformed",
    )

    return {
        "arm_config_sha256": expected["arm_config_sha256"],
        "accepted_record": {
            "path": document["accepted_record"]["path"],
            "sha256": document["accepted_record"]["sha256"],
        },
        "input_manifest": {
            "path": manifest_entry["remote_path"],
            "sha256": manifest_entry["sha256"],
        },
        "checkpoint": {
            "kind": "paired_b0_representation_checkpoint" if arm_id == N4_ARM_ID else "representation_checkpoint",
            "path": document["checkpoint"]["path"],
            "sha256": document["checkpoint"]["sha256"],
            "primary_epoch": PRIMARY_REPRESENTATION_EPOCH,
            "active_epoch": active_epoch,
            "global_step": global_step,
            "model_state_sha256": expected["model_state_sha256"],
            "ema_state_sha256": expected["ema_state_sha256"],
            "model_ema_coherence": "coherent checkpoint: online model and EMA target restored together",
        },
        "coherent_artifact": coherent_artifact,
        "reference_artifact": reference_artifact,
        "normalization": {
            "kind": (
                "c8a_standardizer_in_persisted_k9_downstream_state"
                if arm_id == K9_ARM_ID
                else "b0_conditional_norm_plus_n4_scale_model"
                if arm_id == N4_ARM_ID
                else "conditional_norm_in_restored_model_checkpoint"
            ),
            "model_state_sha256": expected["model_state_sha256"],
            "reference_artifact_sha256": (
                reference_artifact["sha256"] if isinstance(reference_artifact, dict) else None
            ),
            "fit_support_rows": FIT_ROWS,
            "fit_support_sha256": FIT_ROWS_SHA256,
            "fit_histories": list(FIT_HISTORY_IDS),
            "note": (
                "the eligible Fit normalization/standardizer state is restored from the bound "
                "checkpoint/reference artifact; no normalization is refitted or replaced for Confirmation"
            ),
        },
        "fit_bank": {
            "digest_sha256": expected["bank_sha256"],
            "artifact_path": bank_custody["artifact_path"],
            "artifact_sha256": bank_custody["artifact_sha256"],
            "provenance_path": bank_custody["provenance_path"],
            "provenance_sha256": bank_custody["provenance_sha256"],
            "bank_recipe_id": expected["bank_recipe_id"],
            "bank_order_id": expected["bank_order_id"],
            "source_rows": expected["bank_source_rows"],
            "source_rows_sha256": expected["bank_source_rows_sha256"],
            "checkpoint_sha256": document["checkpoint"]["sha256"],
            "custody_basis": (
                "persisted accepted downstream state restored as-is"
                if arm_id == K9_ARM_ID
                else "exact accepted-digest restoration from the frozen checkpoint/EMA/normalization and qualified Fit support"
            ),
        },
        "calibration": {
            "source_history": CALIBRATION_HISTORY_ID,
            "source_rows": CALIBRATION_ROWS,
            "source_support_sha256": CALIBRATION_SUPPORT_SHA256,
            "role_manifest_sha256": calibration_manifest_sha256,
            "quantile": CALIBRATION_QUANTILE,
            "S_pop_threshold": float(thresholds["S_pop"]),
            "S_pred_threshold": float(thresholds["S_pred"]),
            "operator": "frozen accepted value; Calibration re-score verifies within 1e-12 and never reselects",
        },
    }


def _arm_entry(
    arm_id: str,
    matrix: Mapping[str, Any],
    manifests: Mapping[tuple[str, int], Mapping[str, Any]],
    task14_configs: Mapping[str, str],
    bank_audit: Mapping[str, str],
    calibration_manifest_sha256: str,
) -> dict[str, Any]:
    arms = cast(Mapping[str, Any], matrix["arms"])
    arm_document = cast(Mapping[str, Any], arms[arm_id])
    _expect(arm_document.get("status") == "VALID", f"{arm_id} is not an accepted VALID arm")

    seeds = {
        str(seed): _seed_entry(
            arm_id,
            seed,
            arm_document,
            manifests[(arm_id, seed)],
            task14_configs,
            bank_audit,
            calibration_manifest_sha256,
        )
        for seed in MODEL_SEEDS
    }

    if arm_id == K9_ARM_ID:
        reference_state_mode = {
            "S_pred": "restored_from_custody",
            "S_pop": "restored_from_custody",
        }
    elif arm_id == N4_ARM_ID:
        reference_state_mode = {
            "S_pred": "restored_from_custody",
            "S_pop": "reconstructed_digest_verified",
        }
    else:
        reference_state_mode = {
            "S_pred": "restored_from_custody",
            "S_pop": "reconstructed_digest_verified",
        }

    spec = arm_spec(arm_id)
    entry: dict[str, Any] = {
        "arm_id": arm_id,
        "status": "VALID",
        "executable": True,
        "component_set": list(spec.component_set),
        "principal_change": spec.principal_change,
        "mechanism_spec_sha256": spec.mechanism_spec_sha256,
        "representation_epochs": spec.representation_epochs,
        "scale_epochs": spec.scale_epochs,
        "snapshot_rule": "epoch-100 representation checkpoint is the sole primary; diagnostic checkpoints are never eligible",
        "historical_source": _scientific_source(arm_document),
        "scientific_authority": {
            "task14_lock_sha256": N4_SCIENTIFIC_LOCK_SHA256 if arm_id == N4_ARM_ID else TASK14_LOCK_SHA256,
            "arm_config_sha256_by_seed": {
                str(seed): cast(Mapping[str, Any], seeds[str(seed)])["arm_config_sha256"]
                for seed in MODEL_SEEDS
            },
        },
        "reference_state_mode": reference_state_mode,
        "seeds": seeds,
    }
    return entry


def _n1_entry(
    matrix: Mapping[str, Any],
    task14_arms: Mapping[str, Any],
    n1_selection: Mapping[str, Any],
    n1_stopped_row: Mapping[str, Any],
) -> dict[str, Any]:
    arms = cast(Mapping[str, Any], matrix["arms"])
    accepted = cast(Mapping[str, Any], arms[N1_ARM_ID])
    _expect(accepted.get("status") == "STOPPED", "N1-HCC accepted record is not STOPPED")
    _expect(accepted.get("selected_coefficient") is None, "N1-HCC has an invented coefficient")
    _expect(accepted.get("training_authorized") is False, "N1-HCC training is unexpectedly authorized")
    _expect(list(cast(Sequence[Any], accepted.get("checkpoints", []))) == [], "N1-HCC carries checkpoints")
    _expect(list(cast(Sequence[Any], accepted.get("seed_runs", []))) == [], "N1-HCC carries seed runs")
    _expect(accepted.get("scientific_config") is None, "N1-HCC carries an invented scientific config")
    _expect(list(cast(Sequence[Any], accepted.get("per_history_rows", []))) == [], "N1-HCC carries invented rows")

    locked = task14_arms.get(N1_ARM_ID)
    _expect(isinstance(locked, dict), "Task 14 lock has no explicit N1-HCC state")
    locked = cast(Mapping[str, Any], locked)
    _expect(locked.get("executable") is False, "Task 14 lock marks N1-HCC executable")
    _expect(locked.get("lambda_hcc_max") is None, "Task 14 lock invents an N1 coefficient")

    selection = n1_selection.get("selection")
    _expect(isinstance(selection, dict), "N1 selection carrier has no selection record")
    selection = cast(Mapping[str, Any], selection)
    _expect(selection.get("selection_state") == "no_valid_candidate", "N1 selection state differs")
    _expect(selection.get("selected_coefficient") is None, "N1 selection carries a coefficient")
    _expect(
        n1_stopped_row.get("status") in {"FAILED", "INVALID"},
        "N1 stopped row does not preserve a failed/invalid non-execution state",
    )
    failure_reasons = cast(Sequence[str], n1_stopped_row.get("failure_reasons", []))
    _expect(
        any("no executable configuration" in reason for reason in failure_reasons),
        "N1 stopped row does not state the missing executable configuration",
    )

    return {
        "arm_id": N1_ARM_ID,
        "status": "STOPPED",
        "executable": False,
        "selection_state": "no_valid_candidate",
        "selected_coefficient": None,
        "training_authorized": False,
        "accepted_lock_state": "stopped_null_no_executable_config",
        "task14": {
            "status": locked.get("status"),
            "executable": False,
            "lambda_hcc_max": None,
            "config_sha256": None,
        },
        "selection_carrier": {
            "path": N1_SELECTION_PATH,
            "sha256": N1_SELECTION_SHA256,
        },
        "stopped_row": {
            "path": N1_STOPPED_ROW_PATH,
            "sha256": N1_STOPPED_ROW_SHA256,
            "status": n1_stopped_row.get("status"),
            "failure_reasons": list(failure_reasons),
            "note": "the FAILED non-execution row preserves the missing-config cause; its schema-required configuration block is not an executable config",
        },
        "source_records": [
            TASK14_LOCK_PATH,
            N1_SELECTION_PATH,
            N1_EVIDENCE_PATH,
            N1_STOPPED_ROW_PATH,
        ],
        "seed_runs": {},
        "checkpoints": [],
        "fit_bank": None,
        "calibration": None,
        "reference_artifact": None,
        "reason": (
            "Task 10 mechanical selection returned no_valid_candidate on the frozen grid "
            "[0.025, 0.05, 0.10]; Task 14 freezes executable=false and lambda_hcc_max=null. "
            "No same-robot/different-program Fit support, coefficient, checkpoint or metric row exists."
        ),
        "assertion": "explicit STOPPED/non-executable arm; no invented config or checkpoint and no silent omission",
    }


def _n4_authority(
    matrix: Mapping[str, Any],
    manifests: Mapping[tuple[str, int], Mapping[str, Any]],
    task14_configs: Mapping[str, str],
    v1_configs: Mapping[str, str],
) -> dict[str, Any]:
    arms = cast(Mapping[str, Any], matrix["arms"])
    n4_arm = cast(Mapping[str, Any], arms[N4_ARM_ID])
    per_seed: dict[str, Any] = {}
    for seed in MODEL_SEEDS:
        document = cast(Mapping[str, Any], manifests[(N4_ARM_ID, seed)]["document"])
        paired = cast(Mapping[str, Any], document["paired_baseline"])
        _expect(paired.get("arm_id") == B0_ARM_ID, f"N4/{seed} paired baseline is not B0-FULL")
        _expect(
            paired.get("arm_config_sha256") == task14_configs[f"{B0_ARM_ID}/seed-{seed}"],
            f"N4/{seed} paired B0 config differs from the accepted B0 lock config",
        )
        _expect(
            cast(Mapping[str, Any], paired.get("task14_lock")).get("sha256") == TASK14_LOCK_SHA256,
            f"N4/{seed} paired B0 lock is not the accepted v1 lock",
        )
        run = _seed_run(n4_arm, seed)
        representation = cast(Mapping[str, Any], cast(Mapping[str, Any], run["primary"])["representation_primary"])
        calibration = cast(Mapping[str, Any], run["calibration"])
        parity = cast(Mapping[str, Any], calibration["b0_path_parity"])
        _expect(parity.get("per_file_maps_equal_to_accepted_B0") is True, f"N4/{seed} Development B0 map parity is not true")
        _expect(parity.get("thresholds_equal_to_accepted_B0") is True, f"N4/{seed} Development thresholds do not equal B0")
        per_seed[str(seed)] = {
            "paired_b0_arm_config_sha256": paired["arm_config_sha256"],
            "paired_b0_accepted_record": paired["accepted_record"],
            "paired_b0_task14_lock": paired["task14_lock"],
            "paired_b0_checkpoint_sha256": document["checkpoint"]["sha256"],
            "S_pop_threshold": float(calibration["n4_thresholds"]["S_pop"]),
            "S_pred_threshold": float(calibration["n4_thresholds"]["S_pred"]),
            "b0_path_parity": {
                "per_file_maps_equal_to_accepted_B0": True,
                "thresholds_equal_to_accepted_B0": True,
            },
            "paired_b0_representation_step": representation.get("paired_b0_representation_step"),
        }
    return {
        "scientific_lock": {
            "path": N4_SCIENTIFIC_LOCK_PATH,
            "sha256": N4_SCIENTIFIC_LOCK_SHA256,
            "scope": "N4-CRN scientific config only",
        },
        "paired_b0_lock": {
            "path": TASK14_LOCK_PATH,
            "sha256": TASK14_LOCK_SHA256,
            "scope": "paired B0-FULL config and exact S_pop/q95 authority",
        },
        "scientific_source_commit": N4_SCIENTIFIC_SOURCE_COMMIT,
        "v9_lock_source_commit": N4_V9_LOCK_SOURCE_COMMIT,
        "report_only_source_commit": N4_REPORT_ONLY_SOURCE_COMMIT,
        "excluded_authority": {
            "v11_report_only_lock_sha256": N4_V11_REPORT_ONLY_LOCK_SHA256,
            "reason": "report-only N4 v11 lock is not an accepted scientific N4 authority",
            "v1_n4_arm_config_sha256_by_seed": {
                str(seed): v1_configs[f"{N4_ARM_ID}/seed-{seed}"] for seed in MODEL_SEEDS
            },
            "v1_n4_exclusion_reason": (
                "v1 N4 config entries do not match the accepted N4 records; only the v9 "
                "scientific lock is authoritative for N4-CRN"
            ),
        },
        "per_seed": per_seed,
        "parity_rule": (
            "N4-CRN S_pop scores and q95 must equal the same-seed paired B0 exactly; "
            "N4-CRN S_pred remains its accepted independent branch"
        ),
    }


def _local_artifacts(
    root: Path,
    manifests: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for relative in CONTRACT_FILES:
        artifacts[relative] = _sha_file(root / relative)
    artifacts[TASK14_LOCK_PATH] = _sha_file(root / TASK14_LOCK_PATH)
    artifacts[N4_SCIENTIFIC_LOCK_PATH] = _sha_file(root / N4_SCIENTIFIC_LOCK_PATH)
    artifacts[DEVELOPMENT_MATRIX_PATH] = _sha_file(root / DEVELOPMENT_MATRIX_PATH)
    artifacts[SUPPLEMENTARY_RUNNER_PATH] = _sha_file(root / SUPPLEMENTARY_RUNNER_PATH)
    artifacts[N1_SELECTION_PATH] = _sha_file(root / N1_SELECTION_PATH)
    artifacts[N1_STOPPED_ROW_PATH] = _sha_file(root / N1_STOPPED_ROW_PATH)
    artifacts[N1_EVIDENCE_PATH] = _sha_file(root / N1_EVIDENCE_PATH)
    for relative in (MANIFEST_AUDIT_PATH, BANK_AUDIT_PATH, SCORE_AUDIT_PATH, METRIC_AUDIT_PATH):
        artifacts[relative] = _sha_file(root / relative)
    for entry in manifests.values():
        artifacts[cast(str, entry["path"])] = cast(str, entry["sha256"])
    for relative in CODE_FILES:
        artifacts[relative] = _sha_file(root / relative)
    artifacts[PUBLISHER_CODE_PATH] = _sha_file(root / PUBLISHER_CODE_PATH)
    return dict(sorted(artifacts.items()))


def _remote_artifacts(
    arms: Mapping[str, Any],
    manifest_catalog: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, str]:
    remote: dict[str, str] = {}
    for arm_id in EXECUTABLE_ARM_IDS:
        entry = cast(Mapping[str, Any], arms[arm_id])
        for seed in MODEL_SEEDS:
            seed_entry = cast(Mapping[str, Any], cast(Mapping[str, Any], entry["seeds"])[str(seed)])
            remote[seed_entry["checkpoint"]["path"]] = seed_entry["checkpoint"]["sha256"]
            remote[seed_entry["accepted_record"]["path"]] = seed_entry["accepted_record"]["sha256"]
            remote[cast(str, manifest_catalog[(arm_id, seed)]["remote_path"])] = cast(
                str, manifest_catalog[(arm_id, seed)]["sha256"]
            )
            bank = cast(Mapping[str, Any], seed_entry["fit_bank"])
            remote[bank["artifact_path"]] = bank["artifact_sha256"]
            if bank.get("provenance_path"):
                remote[cast(str, bank["provenance_path"])] = cast(str, bank["provenance_sha256"])
            coherent = seed_entry.get("coherent_artifact")
            if isinstance(coherent, dict):
                remote[coherent["path"]] = coherent["sha256"]
            reference = seed_entry.get("reference_artifact")
            if isinstance(reference, dict):
                remote[reference["path"]] = reference["sha256"]
    for relative, sha256 in CONTRACT_FILES.items():
        if relative == PROTOCOL_V1_BASE_PATH:
            continue
        remote[f"{SUPPLEMENTARY_SOURCE_WORKTREE}/{relative}"] = sha256
    for relative, sha256 in CODE_FILES.items():
        remote[f"{SUPPLEMENTARY_SOURCE_WORKTREE}/{relative}"] = sha256
    remote[f"{SUPPLEMENTARY_SOURCE_WORKTREE}/{SUPPLEMENTARY_RUNNER_PATH}"] = SUPPLEMENTARY_RUNNER_SHA256
    return dict(sorted(remote.items()))


def materialize_lock(repository_root: str | Path | None = None) -> dict[str, Any]:
    """Build the deterministic pre-Confirmation lock from accepted evidence.

    Every input is pinned by its accepted SHA-256; a mismatch aborts before a
    lock is produced.  No Confirmation content is read.
    """
    root = _resolve_root(repository_root)
    contract = _contract_identity(root)
    task14 = _task14_identity(root, TASK14_LOCK_PATH, TASK14_LOCK_SHA256, "Task 14 v1")
    task14_v9 = _task14_identity(
        root, N4_SCIENTIFIC_LOCK_PATH, N4_SCIENTIFIC_LOCK_SHA256, "Task 14 v9 N4"
    )
    handoff = _pinned_json(root, HANDOFF_PATH, HANDOFF_SHA256)
    binding = load_task6_binding(root / HANDOFF_PATH)
    matrix = _pinned_json(root, DEVELOPMENT_MATRIX_PATH, DEVELOPMENT_MATRIX_SHA256)
    manifests = _manifest_catalog(root)
    bank_audit = _sha_catalog(root / BANK_AUDIT_PATH)
    _sha_catalog(root / SCORE_AUDIT_PATH)
    _sha_catalog(root / METRIC_AUDIT_PATH)

    task14_document = _pinned_json(root, TASK14_LOCK_PATH, TASK14_LOCK_SHA256)
    task14_v9_document = _pinned_json(
        root, N4_SCIENTIFIC_LOCK_PATH, N4_SCIENTIFIC_LOCK_SHA256
    )
    task14_configs = cast(Mapping[str, str], task14_document["config_sha256"])
    task14_v9_configs = cast(Mapping[str, str], task14_v9_document["config_sha256"])
    combined_configs: dict[str, str] = {
        key: digest
        for key, digest in task14_configs.items()
        if not key.startswith(f"{N4_ARM_ID}/")
    }
    for key, digest in task14_v9_configs.items():
        _expect(key.startswith(f"{N4_ARM_ID}/"), f"v9 lock carries a non-N4 config: {key}")
        _expect(key not in combined_configs, f"duplicate config authority for {key}")
        combined_configs[key] = digest
    _expect(
        len(combined_configs) == len(EXECUTABLE_ARM_IDS) * len(MODEL_SEEDS),
        "accepted config coverage is incomplete",
    )
    for key in task14_v9_configs:
        _expect(
            task14_configs.get(key) != task14_v9_configs[key],
            "v1 and v9 N4 configs unexpectedly coincide; the distinct authority is not preserved",
        )

    role_manifests = _role_manifests(handoff, binding)
    calibration_manifest_sha256 = cast(str, role_manifests[CALIBRATION_HISTORY_ID]["manifest_sha256"])
    _expect(
        cast(str, role_manifests[FIT_HISTORY_IDS[0]]["manifest_sha256"]).startswith("fb0d533c"),
        "certified Fit roster identity differs",
    )
    for history_id in FIT_HISTORY_IDS + (CALIBRATION_HISTORY_ID,) + DEVELOPMENT_HISTORY_IDS + CONFIRMATION_HISTORY_IDS:
        _expect(history_id in role_manifests, f"certified roster is missing {history_id}")

    matrix_arms = cast(Mapping[str, Any], matrix["arms"])
    arms: dict[str, Any] = {}
    for arm_id in EXECUTABLE_ARM_IDS:
        arms[arm_id] = _arm_entry(
            arm_id,
            matrix,
            manifests,
            combined_configs,
            bank_audit,
            calibration_manifest_sha256,
        )
    n1_selection = _pinned_json(root, N1_SELECTION_PATH, N1_SELECTION_SHA256)
    n1_stopped_row = _pinned_json(root, N1_STOPPED_ROW_PATH, N1_STOPPED_ROW_SHA256)
    _expect(
        _sha_file(root / N1_EVIDENCE_PATH) == N1_EVIDENCE_SHA256,
        "N1 Task 27 evidence bytes differ from the accepted identity",
    )
    arms[N1_ARM_ID] = _n1_entry(
        matrix,
        cast(Mapping[str, Any], task14_document["arms"]),
        n1_selection,
        n1_stopped_row,
    )
    ordered_arms = {arm_id: arms[arm_id] for arm_id in ARM_IDS}

    dev_support = _dev_support_identity(matrix, manifests)
    confirmation_support = _confirmation_support_identity(handoff, binding)

    common_contract = cast(Mapping[str, Any], task14_document["common_contract"])
    status_matrix = {
        arm_id: {
            "status": cast(Mapping[str, Any], ordered_arms[arm_id])["status"],
            "seeds": {
                str(seed): {
                    "status": "VALID",
                    "failure_reasons": [],
                }
                for seed in MODEL_SEEDS
            }
            if cast(Mapping[str, Any], ordered_arms[arm_id])["status"] == "VALID"
            else {},
        }
        for arm_id in ARM_IDS
    }

    historical_sources = {
        arm_id: (
            {"status": "STOPPED", "source_records": cast(Mapping[str, Any], ordered_arms[arm_id])["source_records"]}
            if arm_id == N1_ARM_ID
            else _scientific_source(cast(Mapping[str, Any], matrix_arms[arm_id]))
        )
        for arm_id in ARM_IDS
    }

    lock: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "task_id": TASK_ID,
        "attempt_id": ATTEMPT_ID,
        "agent_id": AGENT_ID,
        "role": ROLE,
        "phase": "pre_confirmation",
        "lock_state": LOCK_STATE,
        "confirmation_open": False,
        "publisher": {
            "task_id": TASK_ID,
            "attempt_id": ATTEMPT_ID,
            "agent_id": AGENT_ID,
            "role": ROLE,
        },
        "protocol": {
            **contract,
            "role_binding": {
                "handoff_path": HANDOFF_PATH,
                "handoff_sha256": binding.handoff_sha256,
                "data_binding_raw_sha256": binding.data_binding_raw_sha256,
                "data_binding_canonical_sha256": binding.data_binding_canonical_sha256,
                "executable_binding_sha256": binding.executable_binding_sha256,
                "whole16_certification_sha256": binding.whole16_certification_sha256,
                "qualification_record_sha256": binding.qualification_record_sha256,
                "manifest_aggregate_sha256": binding.manifest_aggregate_sha256,
            },
            "task14_lock": {
                "path": TASK14_LOCK_PATH,
                "sha256": TASK14_LOCK_SHA256,
                "core_sha256": task14["core_sha256"],
                "locked_config_count": task14["locked_config_count"],
            },
            "n4_scientific_lock": {
                "path": N4_SCIENTIFIC_LOCK_PATH,
                "sha256": N4_SCIENTIFIC_LOCK_SHA256,
                "core_sha256": task14_v9["core_sha256"],
                "locked_config_count": task14_v9["locked_config_count"],
            },
            "role_manifests": role_manifests,
        },
        "supplementary_source": {
            "source_commit": SUPPLEMENTARY_SOURCE_COMMIT,
            "source_tree": SUPPLEMENTARY_SOURCE_TREE,
            "source_branch": SUPPLEMENTARY_SOURCE_BRANCH,
            "isolated_worktree": SUPPLEMENTARY_SOURCE_WORKTREE,
            "runner": {"path": SUPPLEMENTARY_RUNNER_PATH, "sha256": SUPPLEMENTARY_RUNNER_SHA256},
            "metric_code": {"path": METRIC_CODE_PATH, "sha256": METRIC_CODE_SHA256},
            "supplementary_definition": contract["supplementary_definition"],
            "role": (
                "supplementary Development metric and scoring implementation; distinct from "
                "each arm's historical training source and from this lock's publisher"
            ),
            "historical_source_policy": (
                "per-arm historical source_commit/source_tree/runner_sha256 are preserved "
                "verbatim and are never replaced by the supplementary source"
            ),
        },
        "historical_sources": historical_sources,
        "common_contract": {
            "model_seeds": list(MODEL_SEEDS),
            "representation": common_contract.get("representation"),
            "checkpoint_cadence": common_contract.get("checkpoint_cadence"),
            "model_geometry": common_contract.get("model_geometry"),
            "primary_checkpoint_rule": "epoch-100 is the sole primary representation checkpoint; N4-CRN uses the final scale epoch-50 within the accepted coherent state",
            "inference_rule": "restore the exact locked model/EMA/reference state; no optimizer step (encoder_updates = 0)",
        },
        "scoring": {
            "score_branches": ["S_pred", "S_pop"],
            "branch_rule": "S_pred and S_pop are independent; no branch fusion, pooling, score switch or branch borrowing",
            "primary_metrics": list(PRIMARY_METRICS),
            "metric_families": list(METRIC_FAMILIES),
            "metric_code": {"path": METRIC_CODE_PATH, "sha256": METRIC_CODE_SHA256},
            "compute_counter": {
                "path": "src/representation/sprint18_harness.py",
                "sha256": CODE_FILES["src/representation/sprint18_harness.py"],
                "functions": [
                    "count_unique_parameters",
                    "primary_inference_parameters",
                    "profile_inference_flops",
                    "validate_compute_envelope",
                ],
            },
            "uncertainty": UNCERTAINTY,
            "confirmation_recovery_gates": CONFIRMATION_GATES,
            "status_semantics": {
                "VALID": "metrics reproduced under the frozen contract; a negative is a valid observation",
                "FAILED": "declared execution could not complete; preserved with reasons",
                "INVALID": "a contract/provenance/support/finite/score-separation/checkpoint/coefficient/compute invariant failed",
            },
            "no_outcome_adaptive_choice": "no refit, retune, reselection, rescore, rerun, threshold/support change or pooling rescue after Confirmation opens",
        },
        "support": {
            "fit": {
                "histories": list(FIT_HISTORY_IDS),
                "healthy_rows": FIT_ROWS,
                "valid_patches": FIT_VALID_PATCHES,
                "source_rows_sha256": FIT_ROWS_SHA256,
                "manifests": {
                    history_id: role_manifests[history_id]["manifest_sha256"]
                    for history_id in FIT_HISTORY_IDS
                },
                "data_permitted_use": role_manifests[FIT_HISTORY_IDS[0]]["data_permitted_use"],
                "model_permitted_use": role_manifests[FIT_HISTORY_IDS[0]]["model_permitted_use"],
                "permitted_use": (
                    "verified-healthy Fit only: representation fitting, EMA targets, arm-specific "
                    "banks/standardizers and declared Fit-only auxiliary fitting"
                ),
            },
            "calibration": {
                "history": CALIBRATION_HISTORY_ID,
                "healthy_rows": CALIBRATION_ROWS,
                "valid_patches": CALIBRATION_VALID_PATCHES,
                "support_sha256": CALIBRATION_SUPPORT_SHA256,
                "manifest_sha256": calibration_manifest_sha256,
                "data_permitted_use": role_manifests[CALIBRATION_HISTORY_ID]["data_permitted_use"],
                "model_permitted_use": role_manifests[CALIBRATION_HISTORY_ID]["model_permitted_use"],
                "permitted_use": "verified-healthy Calibration only: q95 calibration/verification, never refit",
            },
            "development": {
                "histories": list(DEVELOPMENT_HISTORY_IDS),
                "per_history": dev_support,
                "data_permitted_use": role_manifests[DEVELOPMENT_HISTORY_IDS[0]]["data_permitted_use"],
                "model_permitted_use": role_manifests[DEVELOPMENT_HISTORY_IDS[0]]["model_permitted_use"],
                "permitted_use": "descriptive metric reporting only; no arm pruning or final-checkpoint reselection",
            },
            "confirmation": {
                "histories": list(CONFIRMATION_HISTORY_IDS),
                "per_history": confirmation_support,
                "support_rule": "complete manifest file list without label filtering; the locked manifest SHA-256 is the sole support authority",
                "support_sha256": None,
                "support_sha256_policy": (
                    "not precomputed: deriving it requires reading Confirmation support content, "
                    "which stays unopened until the first locked scoring; the frozen harness "
                    "derives it from the locked manifest bytes"
                ),
                "data_permitted_use": role_manifests[CONFIRMATION_HISTORY_IDS[0]]["data_permitted_use"],
                "model_permitted_use": role_manifests[CONFIRMATION_HISTORY_IDS[0]]["model_permitted_use"],
                "permitted_use": "one-shot Confirmation scoring only after independent review and release; never tuning/refit",
                "contact_state": "NOT_ACCESSED",
            },
        },
        "arms": ordered_arms,
        "n4_authority": _n4_authority(
            matrix, manifests, combined_configs, task14_configs
        ),
        "status_matrix": status_matrix,
        "artifact_custody": {
            "local_artifacts": _local_artifacts(root, manifests),
            "remote_artifacts": _remote_artifacts(ordered_arms, manifests),
            "remote_worktree": {
                "path": SUPPLEMENTARY_SOURCE_WORKTREE,
                "branch": SUPPLEMENTARY_SOURCE_BRANCH,
                "commit": SUPPLEMENTARY_SOURCE_COMMIT,
                "tree": SUPPLEMENTARY_SOURCE_TREE,
            },
            "custody_rule": (
                "every local and remote byte used by the lock is bound by SHA-256; the "
                "isolated worktree commit/tree and every contract/code byte are re-verified "
                "before Confirmation opens"
            ),
        },
        "validation_rules": {
            "lock_digest": (
                "SHA-256 over canonical JSON (sorted keys, compact separators, UTF-8) of "
                "every field except lock_sha256"
            ),
            "closed_keys": "unknown top-level fields are rejected",
            "outcome_separation": (
                "no Development or Confirmation outcome value may appear; Confirmation "
                "enters only as published metadata/hashes"
            ),
            "downstream_consumption": (
                "Confirmation workers must load this lock through representation."
                "sprint18_confirmation_lock, use arm_state/confirmation_support, and pass "
                "lock_sha256 to the unchanged harness validate_role_permission; they must "
                "not re-derive frozen state"
            ),
            "fail_closed": "any byte, digest, support, config, seed, arm or custody mismatch aborts",
        },
    }
    lock["lock_sha256"] = lock_sha256(lock)
    return lock


def _assert_no_outcome_content(value: object, path: str = "lock") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_OUTCOME_KEYS:
                raise ConfirmationLockError(f"outcome field is forbidden in the lock: {path}.{key}")
            _assert_no_outcome_content(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_outcome_content(item, f"{path}[{index}]")


def _verify_custody(report: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, int]:
    _expect(
        report.get("schema_id") == "sprint18-confirmation-lock-custody-v1",
        "custody report schema differs",
    )
    worktree = report.get("remote_worktree")
    _expect(isinstance(worktree, dict), "custody report has no remote worktree identity")
    worktree = cast(Mapping[str, Any], worktree)
    _expect(
        worktree.get("commit") == SUPPLEMENTARY_SOURCE_COMMIT,
        "isolated worktree commit differs from the locked source commit",
    )
    _expect(
        worktree.get("tree") == SUPPLEMENTARY_SOURCE_TREE,
        "isolated worktree tree differs from the locked source tree",
    )
    artifacts = report.get("artifacts")
    _expect(isinstance(artifacts, dict), "custody report has no artifact digest map")
    artifacts = cast(Mapping[str, str], artifacts)
    expected = cast(
        Mapping[str, str], cast(Mapping[str, Any], lock["artifact_custody"])["remote_artifacts"]
    )
    missing = sorted(path for path in expected if path not in artifacts)
    _expect(not missing, f"remote custody is incomplete; missing {len(missing)} artifact(s), first: {missing[:1]}")
    mismatched = sorted(path for path in expected if artifacts[path] != expected[path])
    _expect(
        not mismatched,
        f"remote custody digest mismatch for {len(mismatched)} artifact(s), first: {mismatched[:1]}",
    )
    return {"verified_remote_artifacts": len(expected)}


def load_lock(path: str | Path) -> dict[str, Any]:
    """Read a lock document from disk without validating it."""
    candidate = Path(path)
    _expect(candidate.is_file(), f"lock file is missing: {candidate}")
    document = json.loads(candidate.read_bytes().decode("utf-8"))
    _expect(isinstance(document, dict), "lock file is not a JSON object")
    return cast(dict[str, Any], document)


def validate_lock(
    lock: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
    custody_report: str | Path | None = None,
    require_remote_custody: bool = False,
) -> dict[str, Any]:
    """Fail-closed validation of a lock against the accepted evidence and custody."""
    _expect(isinstance(lock, Mapping), "lock must be a JSON object")
    _expect(lock.get("schema_id") == SCHEMA_ID, "lock schema id differs")
    _expect(lock.get("task_id") == TASK_ID, "lock task id differs")
    _expect(lock.get("attempt_id") == ATTEMPT_ID, "lock attempt id differs")
    _expect(lock.get("agent_id") == AGENT_ID, "lock agent id differs")
    _expect(lock.get("role") == ROLE, "lock role differs")
    _expect(lock.get("phase") == "pre_confirmation", "lock phase differs")
    _expect(lock.get("confirmation_open") is False, "lock must not open Confirmation")
    _expect(set(lock) == _TOP_LEVEL_KEYS, f"lock top-level keys differ: {sorted(set(lock) ^ _TOP_LEVEL_KEYS)}")
    _assert_no_outcome_content(lock)

    submitted_digest = lock.get("lock_sha256")
    _expect(
        isinstance(submitted_digest, str) and len(submitted_digest) == 64,
        "lock has no canonical digest",
    )
    _expect(lock_sha256(lock) == submitted_digest, "lock digest does not reproduce its payload")

    expected = materialize_lock(repository_root)
    _expect(
        lock == expected,
        "submitted lock differs from the evidence-derived lock (tamper or stale evidence)",
    )

    custody = {"verified_remote_artifacts": 0}
    if custody_report is not None:
        report = json.loads(Path(custody_report).read_bytes().decode("utf-8"))
        _expect(isinstance(report, dict), "custody report is not a JSON object")
        custody = _verify_custody(cast(Mapping[str, Any], report), lock)
    elif require_remote_custody:
        raise ConfirmationLockError("a remote custody report is required but was not supplied")

    return {
        "status": "VALID",
        "lock_sha256": submitted_digest,
        "lock_state": lock["lock_state"],
        "confirmation_open": lock["confirmation_open"],
        "arms": len(cast(Mapping[str, Any], lock["arms"])),
        "executable_arms": len(EXECUTABLE_ARM_IDS),
        "seeds_per_executable_arm": len(MODEL_SEEDS),
        "n1_state": cast(Mapping[str, Any], cast(Mapping[str, Any], lock["arms"])[N1_ARM_ID])["status"],
        "local_artifacts": len(
            cast(Mapping[str, Any], cast(Mapping[str, Any], lock["artifact_custody"])["local_artifacts"])
        ),
        "remote_artifacts": len(
            cast(Mapping[str, Any], cast(Mapping[str, Any], lock["artifact_custody"])["remote_artifacts"])
        ),
        "verified_remote_artifacts": custody["verified_remote_artifacts"],
    }


def arm_state(lock: Mapping[str, Any], arm_id: str, seed: int | str) -> dict[str, Any]:
    """Return one frozen Confirmation arm/seed state; fail closed on misuse."""
    _expect(lock.get("schema_id") == SCHEMA_ID, "lock schema id differs")
    arms = lock.get("arms")
    _expect(isinstance(arms, Mapping), "lock has no arms")
    arms = cast(Mapping[str, Any], arms)
    _expect(arm_id in arms, f"unknown arm: {arm_id}")
    entry = cast(Mapping[str, Any], arms[arm_id])
    _expect(entry.get("status") == "VALID", f"{arm_id} is not an executable Confirmation arm")
    seed_key = str(int(seed))
    _expect(int(seed) in MODEL_SEEDS, f"unregistered model seed: {seed}")
    seeds = cast(Mapping[str, Any], entry["seeds"])
    _expect(seed_key in seeds, f"{arm_id} has no frozen state for seed {seed_key}")
    return cast(dict[str, Any], seeds[seed_key])


def confirmation_support(lock: Mapping[str, Any], history_id: str) -> dict[str, Any]:
    """Return one locked Confirmation support identity; fail closed on misuse."""
    _expect(lock.get("schema_id") == SCHEMA_ID, "lock schema id differs")
    support = cast(Mapping[str, Any], cast(Mapping[str, Any], lock["support"])["confirmation"])
    per_history = cast(Mapping[str, Any], support["per_history"])
    _expect(history_id in per_history, f"unknown Confirmation history: {history_id}")
    return cast(dict[str, Any], per_history[history_id])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    build = sub.add_parser("build", help="materialize the lock from accepted evidence")
    build.add_argument("--repository-root", default=None)
    build.add_argument("--output", default=None)

    validate = sub.add_parser("validate", help="validate a lock and its optional remote custody")
    validate.add_argument("--repository-root", default=None)
    validate.add_argument("--lock", default=None)
    validate.add_argument("--custody-report", default=None)
    validate.add_argument("--require-remote-custody", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = _resolve_root(args.repository_root)
    if args.mode == "build":
        lock = materialize_lock(root)
        output = Path(args.output) if args.output else root / LOCK_PATH
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(
            (json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        )
        print(
            json.dumps(
                {
                    "mode": "build",
                    "output": str(output),
                    "file_sha256": _sha_file(output),
                    "lock_sha256": lock["lock_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0

    lock_path = Path(args.lock) if args.lock else root / LOCK_PATH
    lock = load_lock(lock_path)
    summary = validate_lock(
        lock,
        repository_root=root,
        custody_report=args.custody_report,
        require_remote_custody=args.require_remote_custody,
    )
    summary = {**summary, "mode": "validate", "lock_path": str(lock_path)}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
