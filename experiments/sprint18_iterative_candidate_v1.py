"""Fail-closed runner for the prospective Sprint 18 iterative data candidate.

Supported Linux invocations run from the bound root with
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=<boundroot>/src:<boundroot>/experiments:<boundroot>`,
and `<boundroot>/.venv/bin/python`. Bare or differently rooted invocations fail closed.
Contact stages require Main's exact release receipt and run only the bound
preflight or the public chronological CLI.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import datetime
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import tempfile
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BINDING_DEFAULT = ROOT / "experiments" / "sprint18-iterative-binding-v1.json"
BASE_COMMIT = "2b1bf310d15bb682e0278ae59ae64f62bb67589c"
TASK68_A02_CHECKPOINT_COMMIT = "1b708542aeaba7a69c883893deb90ff591ce9d5f"
ACCEPTED_METHOD_LINEAGE_COMMIT = "cd04a0a018c73ae91593ea4041742f05829041a6"
TASK68_A02_SOURCE_CLOSURE_SHA256 = "e9a50ec7764d8a9cf869a522fbd00ca462beec16bd53ba6308dcb84b1863dd99"
SOURCE_IDENTITY_SCHEME = "git-tracked-utf8-text-lf-sha256-v1"
UNKNOWN_SOURCE_IDENTITY_POLICY = "reject"
NON_TEXT_SOURCE_POLICY = "reject; require a separately versioned identity scheme"
MAIN_RELEASE_SCHEMA_ID = "sprint18-main-candidate-release-v2"
PROFILE = "sprint18-iterative-v3"
EXECUTION_ROOT = "/home/trietlm/anomaly-representation-learning-s18t68-a02-worktree"
GIT_STORAGE_ROOT = "/home/trietlm/anomaly-representation-learning"
APPROVED_BRANCH = "sprint18-task68-a02-runtime"
PYTHON_PROVIDER = "/home/trietlm/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
LOCKED_PACKAGE_VERSIONS = {"numpy": "2.5.2", "scipy": "1.18.1", "torch": "2.14.0+cu130"}

TASK67_EVIDENCE = {
    "path": "artifacts/sprint-18/task-67.md",
    "sha256": "e576debdd048c780ca6c25d91923014705bc8b3723e70566e8dd99c947139867",
    "reference": "accepted Task67 evidence; provenance only, not a deployable source",
}
TASK67_EVIDENCE_PATH = TASK67_EVIDENCE["path"]
TASK67_EVIDENCE_SHA256 = TASK67_EVIDENCE["sha256"]
TASK67_EVIDENCE_REFERENCE = TASK67_EVIDENCE["reference"]
PROTOCOL = "sprint15-benchmark-protocol-v7"
ROLE_ORDER = (
    ("DESIGN", 4), ("FIT", 3), ("CALIBRATION", 1),
    ("DEVELOPMENT", 4), ("CONFIRMATION", 4),
)
ROLE_IDS = tuple(role for role, count in ROLE_ORDER for _ in range(count))
ROLE_SUFFIX = {
    "DESIGN": "DESIGN", "FIT": "FIT", "CALIBRATION": "CALIBRATION",
    "DEVELOPMENT": "DEVELOPMENT", "CONFIRMATION": "CONFIRMATION",
}
PERMISSIONS = {
    "DESIGN": "Integrity, causal, construction, and unchanged structural qualification; diagnostics only. No Fit/Calibration use or history selection.",
    "FIT": "Verified-healthy eligibility; fit only the fixed probe's feature mean/std and healthy centroid. No representation training, bank, alternate scorer, or non-probe fitting.",
    "CALIBRATION": "Verified-healthy eligibility; score with frozen Fit probe values and choose only the fixed probe q95. No fitting or retuning.",
    "DEVELOPMENT": "Structural qualification and one fixed-probe evaluation using frozen Fit/Calibration values. No selection, promotion, pruning, or tuning.",
    "CONFIRMATION": "After Task69 preceding-role PASS and Main release: structural certification and one fixed-probe evaluation using frozen Fit/Calibration values. No fitting, recalibration, retuning, rescue, or model score.",
}
SEED_FIELDS = ("factory", "scheduler", "health", "signal", "temporal")
EXPECTED_SOURCE_PATHS = frozenset({
    "docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md",
    "experiments/sprint15-benchmark-protocol-v7.md",
    "experiments/sprint15-observable-probe-v7.md",
    "experiments/sprint18-candidate4-operational-retirement-v1.json",
    "experiments/sprint18-data-method-c2-v1.md",
    "experiments/sprint18-iterative-data-contract-v1.md",
    "experiments/sprint18-iterative-data-contract-v3.md",
    "experiments/sprint18-task67-attempt-S18-T67-A05.json",
    "experiments/sprint18-task67-attempt-S18-T67-A05.md",
    "experiments/sprint18_iterative_candidate_v1.py",
    "experiments/sprint18_task5_measurability.py",
    "experiments/sprint22-pilot-binding-v1.json",
    "pyproject.toml",
    "src/synth/__init__.py",
    "src/synth/anomalies/__init__.py",
    "src/synth/anomalies/base.py",
    "src/synth/anomalies/contextual.py",
    "src/synth/anomalies/cross_channel.py",
    "src/synth/anomalies/drift.py",
    "src/synth/anomalies/duration.py",
    "src/synth/anomalies/easy_sanity.py",
    "src/synth/anomalies/freq_phase.py",
    "src/synth/anomalies/missing_event.py",
    "src/synth/anomalies/registry.py",
    "src/synth/anomalies/regularity.py",
    "src/synth/anomalies/stuck.py",
    "src/synth/anomalies/transition.py",
    "src/synth/balanced.py",
    "src/synth/chronicle.py",
    "src/synth/cli.py",
    "src/synth/config.py",
    "src/synth/contrastive.py",
    "src/synth/dataset.py",
    "src/synth/diagnostics.py",
    "src/synth/events.py",
    "src/synth/generator.py",
    "src/synth/health.py",
    "src/synth/masking.py",
    "src/synth/normal.py",
    "src/synth/patchify.py",
    "src/synth/physics/__init__.py",
    "src/synth/physics/causal.py",
    "src/synth/physics/noise.py",
    "src/synth/preflight15.py",
    "src/synth/probe15.py",
    "src/synth/regimes.py",
    "src/synth/scheduled.py",
    "src/synth/scheduler.py",
    "src/synth/schema.py",
    "src/synth/splits.py",
    "src/synth/strength.py",
    "src/synth/temporal.py",
    "uv.lock",
})
HISTORICAL_SOURCE_PATHS = frozenset({
    "docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md",
    "experiments/sprint15-benchmark-protocol-v7.md",
    "experiments/sprint15-observable-probe-v7.md",
    "experiments/sprint18-data-method-c2-v1.md",
    "experiments/sprint18-iterative-data-contract-v1.md",
    "experiments/sprint18_iterative_candidate_v1.py",
    "experiments/sprint18_task5_measurability.py",
    "experiments/sprint22-pilot-binding-v1.json",
    "pyproject.toml",
    "src/synth/__init__.py",
    "src/synth/anomalies/__init__.py",
    "src/synth/anomalies/base.py",
    "src/synth/anomalies/contextual.py",
    "src/synth/anomalies/cross_channel.py",
    "src/synth/anomalies/drift.py",
    "src/synth/anomalies/duration.py",
    "src/synth/anomalies/easy_sanity.py",
    "src/synth/anomalies/freq_phase.py",
    "src/synth/anomalies/missing_event.py",
    "src/synth/anomalies/registry.py",
    "src/synth/anomalies/regularity.py",
    "src/synth/anomalies/stuck.py",
    "src/synth/anomalies/transition.py",
    "src/synth/balanced.py",
    "src/synth/chronicle.py",
    "src/synth/cli.py",
    "src/synth/config.py",
    "src/synth/contrastive.py",
    "src/synth/dataset.py",
    "src/synth/diagnostics.py",
    "src/synth/events.py",
    "src/synth/generator.py",
    "src/synth/health.py",
    "src/synth/masking.py",
    "src/synth/normal.py",
    "src/synth/patchify.py",
    "src/synth/physics/__init__.py",
    "src/synth/physics/causal.py",
    "src/synth/physics/noise.py",
    "src/synth/preflight15.py",
    "src/synth/probe15.py",
    "src/synth/regimes.py",
    "src/synth/scheduled.py",
    "src/synth/scheduler.py",
    "src/synth/schema.py",
    "src/synth/splits.py",
    "src/synth/strength.py",
    "src/synth/temporal.py",
    "uv.lock",
})


class GuardError(RuntimeError):
    """A frozen binding or stage precondition did not match."""

def _entrypoint_pythonpath(root: Path) -> str:
    return os.pathsep.join(
        str(path.resolve()) for path in (
            root / "src", root / "experiments", root,
        )
    )


def _entrypoint_child_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = _entrypoint_pythonpath(root)
    return env


def _validate_entrypoint_environment(
    binding: dict[str, Any], root: Path, *, disposable: bool,
) -> None:
    expected_pythonpath = _entrypoint_pythonpath(root)
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise GuardError("entrypoint requires PYTHONDONTWRITEBYTECODE=1")
    if os.environ.get("PYTHONPATH") != expected_pythonpath:
        raise GuardError(
            "entrypoint requires PYTHONPATH to contain exactly the bound src, experiments, and repository root paths"
        )
    if not disposable:
        runtime = binding["runtime"]
        if Path(sys.executable).resolve() != Path(runtime["python_executable"]).resolve():
            raise GuardError("entrypoint must use the bound locked Python provider")
        if Path(sys.prefix).resolve() != Path(runtime["python_prefix"]).resolve():
            raise GuardError("entrypoint must use the bound locked virtual environment")


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_source_bytes(relative: str, raw: bytes) -> bytes:
    """Return the strict UTF-8 text identity with LF canonical newlines."""
    if b"\0" in raw:
        raise GuardError(f"frozen source is not text: {relative}")
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GuardError(f"frozen source is not UTF-8 text: {relative}") from exc
    crlf_count = raw.count(b"\r\n")
    if raw.count(b"\r") != crlf_count:
        raise GuardError(f"frozen source has bare CR bytes: {relative}")
    if crlf_count and raw.count(b"\n") != crlf_count:
        raise GuardError(f"frozen source mixes LF and CRLF newlines: {relative}")
    return raw.replace(b"\r\n", b"\n")


def _source_closure_digest(closure: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json({
        "identity_scheme": closure.get("identity_scheme"),
        "unknown_scheme_policy": closure.get("unknown_scheme_policy"),
        "non_text_policy": closure.get("non_text_policy"),
        "sha256_by_path": closure.get("sha256_by_path"),
        "git_blob_oid_by_path": closure.get("git_blob_oid_by_path"),
    }))

def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GuardError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GuardError(f"JSON input must be an object: {path}")
    return value


def _candidate_digest(binding: dict[str, Any]) -> str:
    payload = copy.deepcopy(binding)
    payload.pop("binding_sha256", None)
    return sha256_bytes(canonical_json(payload))


def _source_path(root: Path, relative: str) -> Path:
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise GuardError(f"required frozen source is absent: {relative}") from exc
    if not resolved.is_file() or root.resolve() not in resolved.parents:
        raise GuardError(f"frozen source is not a regular in-repository file: {relative}")
    return resolved

def _assert_safe_candidate_path(root: Path, path: Path) -> None:
    root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise GuardError("candidate path is outside the bound project root") from exc
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise GuardError(f"candidate path contains a symlink: {current}")

def _fsync_directory_chain(path: Path, root: Path) -> None:
    if os.name == "nt":
        return
    root = root.resolve()
    current = path.resolve(strict=True)
    if current != root and root not in current.parents:
        raise GuardError("durable candidate path is outside the project root")
    while True:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        descriptor = os.open(current, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if current == root:
            return
        current = current.parent


def verify_source_closure(binding: dict[str, Any], root: Path) -> None:
    closure = binding.get("source_closure")
    if not isinstance(closure, dict):
        raise GuardError("binding has no source closure")
    if set(closure) != {
        "identity_scheme", "unknown_scheme_policy", "non_text_policy",
        "closure_sha256", "sha256_by_path", "git_blob_oid_by_path",
    }:
        raise GuardError("source closure has unsupported identity fields")
    if closure.get("identity_scheme") != SOURCE_IDENTITY_SCHEME:
        raise GuardError("unknown source identity scheme")
    if closure.get("unknown_scheme_policy") != UNKNOWN_SOURCE_IDENTITY_POLICY:
        raise GuardError("unknown source identity policy must reject")
    if closure.get("non_text_policy") != NON_TEXT_SOURCE_POLICY:
        raise GuardError("non-text source identity policy mismatch")
    sources = closure.get("sha256_by_path")
    blob_oids = closure.get("git_blob_oid_by_path")
    if not isinstance(sources, dict) or not isinstance(blob_oids, dict):
        raise GuardError("source closure must bind text hashes and Git blob identities")
    if set(sources) != EXPECTED_SOURCE_PATHS or set(blob_oids) != EXPECTED_SOURCE_PATHS:
        raise GuardError("source closure path set differs from the exact 53-file catalog")
    if TASK67_EVIDENCE_PATH in sources:
        raise GuardError("Task67 evidence is provenance, not a runtime source")
    if closure.get("closure_sha256") != _source_closure_digest(closure):
        raise GuardError("source-closure manifest digest mismatch")
    for relative, expected in sources.items():
        blob_oid = blob_oids[relative]
        if (
            not isinstance(expected, str) or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
        ):
            raise GuardError("source closure entries must be lowercase SHA-256 identities")
        if (
            not isinstance(blob_oid, str) or len(blob_oid) != 40
            or any(char not in "0123456789abcdef" for char in blob_oid)
        ):
            raise GuardError("source closure entries must bind lowercase Git blob IDs")
        actual = sha256_bytes(canonical_source_bytes(
            relative, _source_path(root, relative).read_bytes(),
        ))
        if actual != expected:
            raise GuardError(
                f"frozen source mismatch: {relative}: expected {expected}, got {actual}"
            )


def verify_git_source_blobs(binding: dict[str, Any], root: Path) -> None:
    expected = binding["source_closure"]["git_blob_oid_by_path"]
    output = subprocess.run(
        [
            "git", "-C", str(root), "ls-tree", "-r",
            "--format=%(objectname) %(path)", "HEAD", "--",
            *sorted(expected),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    actual = {
        path: oid for oid, path in (
            line.split(" ", 1) for line in output.splitlines()
        )
    }
    if actual != expected:
        raise GuardError("tracked Git source blob identities differ from the frozen closure")


def _seed_exclusions(catalog: dict[str, Any]) -> set[int]:
    excluded: set[int] = set()
    for item in catalog.get("prior_data_seed_ranges", []):
        first, last = item.get("first"), item.get("last")
        if (not isinstance(first, int) or isinstance(first, bool)
                or not isinstance(last, int) or isinstance(last, bool)
                or first > last):
            raise GuardError("invalid prior data-seed range in collision catalog")
        excluded.update(range(first, last + 1))
    for item in catalog.get("prior_model_seed_ranges", []):
        first, last = item.get("first"), item.get("last")
        if (not isinstance(first, int) or isinstance(first, bool)
                or not isinstance(last, int) or isinstance(last, bool)
                or first > last):
            raise GuardError("invalid prior model-seed range in collision catalog")
        excluded.update(range(first, last + 1))
    singles = catalog.get("prior_single_seeds")
    if not isinstance(singles, list) or any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in singles
    ):
        raise GuardError("invalid singleton seed exclusions")
    excluded.update(singles)
    if catalog.get("catalog_id") != "S22-DISTINCT-H32-STATIC-CATALOG-v1":
        raise GuardError("unknown static collision catalog identity")
    if catalog.get("exclusions_sha256") != sha256_bytes(canonical_json(sorted(excluded))):
        raise GuardError("static collision exclusion digest mismatch")
    return excluded


def _role_entries(binding: dict[str, Any]) -> list[dict[str, Any]]:
    entries = binding.get("role_binding")
    if not isinstance(entries, list) or len(entries) != 16:
        raise GuardError("binding must contain exactly 16 role entries")
    return entries


def validate_binding(
    binding: dict[str, Any], root: Path, *, disposable: bool = False,
    check_sources: bool = True,
) -> None:
    if binding.get("schema_id") != "sprint18-iterative-binding-v1":
        raise GuardError("unsupported candidate binding schema")
    if binding.get("candidate_id") != "S18-ITER-0005" and not disposable:
        raise GuardError("candidate identity differs from the frozen fifth block")
    if binding.get("profile_id") != PROFILE or binding.get("generator_protocol_id") != PROTOCOL:
        raise GuardError("profile or protocol identity mismatch")
    if binding.get("contract_sha256") != "544de84bc201a07140559058d100c991cd148b698f93058dbd37d7fe4dc3c929":
        raise GuardError("accepted Task66 contract digest mismatch")
    if binding.get("binding_sha256") != _candidate_digest(binding):
        raise GuardError("canonical binding SHA-256 mismatch")
    if binding.get("provenance", {}).get("task67_evidence") != TASK67_EVIDENCE:
        raise GuardError("accepted Task67 evidence provenance identity mismatch")
    provenance = binding["provenance"]
    lineage = provenance.get("accepted_task68_lineage")
    if lineage != {
        "task68_a02_checkpoint_commit": TASK68_A02_CHECKPOINT_COMMIT,
        "task68_a02_checkpoint_parent": ACCEPTED_METHOD_LINEAGE_COMMIT,
        "task66_contract_sha256": binding["contract_sha256"],
        "task67_checkpoint_commit": "a17bfad5d5c625e2c8745c0d7b6feff4ee928eb4",
        "task67_checkpoint_parent": "820d724893341cee60f2a911b65881be2761b3e1",
        "task67_review_snapshot_sha256": "cd1d26b1c5e6d3556823f943b632ddcff4e36ab6b87299780b9b3469df540842",
    }:
        raise GuardError("accepted Task68/Task66 checkpoint lineage mismatch")
    historical = provenance.get("task68_a02_raw_sha256_by_path")
    if (
        provenance.get("task68_a02_raw_source_identity_scheme")
        != "raw-byte-sha256-v1"
        or provenance.get("task68_a02_raw_source_closure_sha256")
        != TASK68_A02_SOURCE_CLOSURE_SHA256
        or not isinstance(historical, dict)
        or set(historical) != HISTORICAL_SOURCE_PATHS
        or sha256_bytes(canonical_json(historical))
        != TASK68_A02_SOURCE_CLOSURE_SHA256
    ):
        raise GuardError("accepted Task68 A02 raw source provenance mismatch")
    release_policy = binding.get("release_policy", {})
    if (
        release_policy.get("candidate_release_receipt_schema_id")
        != MAIN_RELEASE_SCHEMA_ID
        or release_policy.get("required_checkpoint_parent") != BASE_COMMIT
    ):
        raise GuardError("candidate release policy does not bind the corrected checkpoint")
    validate_runtime_binding(binding)
    _validate_entrypoint_environment(binding, root, disposable=disposable)
    entries = _role_entries(binding)
    expected_roles = [role for role in ROLE_IDS]
    actual_roles = [entry.get("role") for entry in entries]
    if actual_roles != expected_roles:
        raise GuardError("role count or immutable role order mismatch")
    ids = [entry.get("history_id") for entry in entries]
    seeds = [entry.get("data_seed") for entry in entries]
    paths = [entry.get("directory") for entry in entries]
    if (len(set(ids)) != 16 or len(set(seeds)) != 16 or len(set(paths)) != 16
            or any(not isinstance(value, str) or not value for value in ids + paths)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in seeds)):
        raise GuardError("role IDs, seeds, and paths must each be unique and well formed")

    candidate = binding["candidate_id"]
    first_seed = binding.get("seed_block", {}).get("first_seed")
    if not isinstance(first_seed, int) or isinstance(first_seed, bool):
        raise GuardError("invalid first seed")
    if not disposable:
        number = binding.get("candidate_number")
        if number != 5 or first_seed != 32064 or seeds != list(range(32064, 32080)):
            raise GuardError("candidate number or fixed 32064–32079 seed block mismatch")
    elif seeds != list(range(first_seed, first_seed + 16)):
        raise GuardError("disposable smoke seed block must remain contiguous and ordered")

    root_relative = binding.get("candidate_root_relative")
    if not isinstance(root_relative, str) or root_relative != (
        f"data/generated/sprint18-iterative-v1/{candidate}"
    ):
        raise GuardError("candidate output root mismatch")
    for index, entry in enumerate(entries):
        role = expected_roles[index]
        ordinal = sum(1 for prior in expected_roles[:index] if prior == role) + 1
        if not disposable:
            wanted_id = f"S18I-ITER-0005-{ROLE_SUFFIX[role]}-{ordinal:02d}"
            if entry.get("history_id") != wanted_id:
                raise GuardError(f"history identity/order mismatch at roster position {index}")
        expected_path = f"{root_relative}/{role}/{entry['history_id']}"
        if entry.get("directory") != expected_path:
            raise GuardError(f"role path mismatch at roster position {index}")
        if entry.get("permitted_use") != PERMISSIONS[role]:
            raise GuardError(f"role permission mismatch for {entry.get('history_id')}")

    catalog = binding.get("collision_catalog")
    if not isinstance(catalog, dict):
        raise GuardError("binding has no static collision catalog")
    catalog_source = binding.get("source_closure", {}).get("sha256_by_path", {}).get(
        "experiments/sprint22-pilot-binding-v1.json"
    )
    if catalog.get("source_catalog_sha256") != catalog_source:
        raise GuardError("collision catalog is not bound to its exact static source")
    excluded = _seed_exclusions(catalog)
    collision = sorted(set(seeds) & excluded)
    if collision:
        raise GuardError(f"candidate seeds collide with the bound static catalog: {collision}")
    if catalog.get("candidate_intersection") != []:
        raise GuardError("recorded static-catalog intersection is not empty")
    if catalog.get("candidate_seed_count") != 16:
        raise GuardError("static collision proof is not for all 16 roles")

    expected_config_keys = {str(entry["data_seed"]): entry for entry in entries}
    if set(binding.get("configs", {})) != set(expected_config_keys):
        raise GuardError("resolved configuration table does not match the roster seeds")
    if check_sources:
        verify_source_closure(binding, root)
        spec = importlib.util.find_spec("synth")
        expected_package = (root / "src" / "synth" / "__init__.py").resolve()
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected_package:
            raise GuardError("synth import does not originate from the bound project src tree")
        for item in sys.path:
            if "sprint22" in item.lower() and ("source" in item.lower() or "hist" in item.lower()):
                raise GuardError("historical Sprint22 source fallback is on sys.path")
        from dataclasses import asdict
        from synth.chronicle import sprint18_iterative_v3_history_config

        semantic_hashes: set[str] = set()
        for entry in entries:
            seed = entry["data_seed"]
            cfg = sprint18_iterative_v3_history_config(seed=seed)
            resolved = asdict(cfg)
            actual_short = cfg.hash()
            actual_full = sha256_bytes(canonical_json(resolved))
            seed_fields = {field: resolved[field]["seed"] for field in SEED_FIELDS}
            normalized = copy.deepcopy(resolved)
            for field in SEED_FIELDS:
                normalized[field]["seed"] = 0
            semantic_hash = sha256_bytes(canonical_json(normalized))
            semantic_hashes.add(semantic_hash)
            expected = binding["configs"][str(seed)]
            if (actual_short != expected.get("config_hash")
                    or actual_full != expected.get("config_sha256")
                    or seed_fields != expected.get("seed_fields")
                    or semantic_hash != expected.get("seed_independent_sha256")
                    or entry.get("config_hash") != expected.get("config_hash")):
                raise GuardError(f"resolved configuration mismatch for seed {seed}")
        if len(semantic_hashes) != 1:
            raise GuardError("the fixed 16 configs are not common-method equivalents")


def validate_runtime_binding(binding: dict[str, Any]) -> None:
    runtime = binding.get("runtime", {})
    worktree = PurePosixPath(runtime.get("execution_worktree_root", ""))
    storage = PurePosixPath(runtime.get("git_storage_root", ""))
    prefix = worktree / ".venv"
    expected_origins = {
        "numpy": prefix / "lib/python3.12/site-packages/numpy/__init__.py",
        "scipy": prefix / "lib/python3.12/site-packages/scipy/__init__.py",
        "torch": prefix / "lib/python3.12/site-packages/torch/__init__.py",
        "synth": worktree / "src/synth/__init__.py",
    }
    output_root = worktree / "data/generated"
    candidate_root = output_root / "sprint18-iterative-v1/S18-ITER-0005"
    expected_entrypoint_environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": [
            str(worktree / "src"), str(worktree / "experiments"), str(worktree),
        ],
        "interpreter": ".venv/bin/python",
    }
    if (
        str(worktree) != EXECUTION_ROOT
        or str(storage) != GIT_STORAGE_ROOT
        or worktree == storage
        or runtime.get("working_directory") != str(worktree)
        or runtime.get("git_common_dir") != str(storage / ".git")
        or runtime.get("approved_branch") != APPROVED_BRANCH
        or runtime.get("origin_url") != "https://github.com/triet4p/anomaly-representation-learning.git"
        or runtime.get("hostname") != "di-server"
        or runtime.get("python_version") != [3, 12, 13]
        or runtime.get("python_executable") != PYTHON_PROVIDER
        or runtime.get("python_prefix") != str(prefix)
        or runtime.get("output_root") != str(output_root)
        or runtime.get("candidate_root") != str(candidate_root)
        or runtime.get("entrypoint_environment") != expected_entrypoint_environment
        or runtime.get("package_versions") != LOCKED_PACKAGE_VERSIONS
        or runtime.get("dependency_origins") != {
            name: str(path) for name, path in expected_origins.items()
        }
        or runtime.get("environment_status_at_binding") != "EXISTING_LOCKED_ENVIRONMENT_VERIFIED"
    ):
        raise GuardError("prospective worktree or locked runtime binding is inconsistent")
    sources = binding.get("source_closure", {}).get("sha256_by_path", {})
    if runtime.get("uv_lock_sha256") != sources.get("uv.lock"):
        raise GuardError("runtime environment is not bound to the frozen uv.lock")
    if (runtime.get("uv_version") != HOST_UV_VERSION
            or runtime.get("uv_build") != HOST_UV_BUILD):
        raise GuardError("runtime environment requires the observed locked uv provider")


HOST_UV_VERSION = "0.12.21"
HOST_UV_BUILD = "7af826859"
HOST_UV_PLATFORM = "x86_64-unknown-linux-gnu"
HOST_UV_DATE = "2026-09-29"


def _host_uv_callsign() -> dict[str, str]:
    try:
        completed = subprocess.run(
            ["uv", "--version"], check=False, capture_output=True, text=True,
        )
    except OSError as exc:
        raise GuardError(f"host uv toolchain is unavailable: {exc}") from exc
    if completed.returncode != 0:
        raise GuardError("host uv toolchain refused its version probe")
    tokens = completed.stdout.strip().split()
    if len(tokens) not in (4, 5) or tokens[0] != "uv" or not tokens[2].startswith("(") or not tokens[-1].endswith(")"):
        raise GuardError("host uv version probe has an unexpected shape")
    version, build = tokens[1], tokens[2].lstrip("(")
    parts = version.strip().split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise GuardError(f"host uv version is not a pinned triple: {version!r}")
    if not build or any(char not in "0123456789abcdef" for char in build):
        raise GuardError(f"host uv build is not a pinned hash: {build!r}")
    return {"uv_version": version, "uv_build": build}


def _validate_host_uv_toolchain(binding: dict[str, Any]) -> dict[str, str]:
    runtime = binding.get("runtime", {})
    if runtime.get("uv_version") != HOST_UV_VERSION or runtime.get("uv_build") != HOST_UV_BUILD:
        raise GuardError("binding does not carry the observed host uv provider pin")
    observed = _host_uv_callsign()
    if observed.get("uv_version") != HOST_UV_VERSION or observed.get("uv_build") != HOST_UV_BUILD:
        raise GuardError("host uv toolchain differs from the bound provider; refusing before contact")
    return observed


def _validate_runtime_dependencies(
    binding: dict[str, Any], observed: dict[str, Any],
) -> None:
    runtime = binding["runtime"]
    if observed.get("package_versions") != runtime.get("package_versions"):
        raise GuardError("runtime NumPy/SciPy/PyTorch version mismatch")
    for module, expected in runtime.get("dependency_origins", {}).items():
        actual = observed.get("dependency_origins", {}).get(module)
        if actual is None or Path(actual).resolve() != Path(expected).resolve():
            raise GuardError(f"runtime dependency origin mismatch for {module}")

def _validate_runtime_observation(
    binding: dict[str, Any], root: Path, observed: dict[str, Any],
) -> None:
    runtime = binding["runtime"]
    expected_root = Path(runtime["execution_worktree_root"]).resolve()
    if (root.resolve() != expected_root
            or Path(observed["cwd"]).resolve() != expected_root):
        raise GuardError(f"runtime execution worktree mismatch; expected {expected_root}")
    if observed.get("hostname") != runtime.get("hostname"):
        raise GuardError("runtime host identity mismatch")
    if Path(observed["python_executable"]).resolve() != Path(
        runtime["python_executable"]
    ).resolve():
        raise GuardError("runtime Python interpreter mismatch")
    if Path(observed["python_prefix"]).resolve() != Path(runtime["python_prefix"]).resolve():
        raise GuardError("runtime virtual-environment prefix mismatch")
    if observed.get("python_version") != runtime.get("python_version"):
        raise GuardError("runtime Python version mismatch")
    _validate_runtime_dependencies(binding, observed)
    if observed.get("cuda_available") is not True:
        raise GuardError("bound CUDA runtime is unavailable")
    if (observed.get("cuda_device") != runtime.get("cuda_device")
            or observed.get("cuda_device_total_mib") != runtime.get("cuda_device_total_mib")):
        raise GuardError("runtime CUDA device identity mismatch")


def _validate_public_cli_entrypoint(binding: dict[str, Any], root: Path) -> dict[str, Any]:
    """Exercise the REAL SAME-INTERPRETER public CLI parser in a subprocess.

    Runs the actual bound entrypoint environment (``sys.executable`` with the
    exact three-prefix ``PYTHONPATH``) as ``python -m synth.cli --help`` in a
    child process. ``--help`` performs no generation and requires no installed
    package metadata, so it must exit 0 on every conforming runtime; this
    fails closed BEFORE candidate contact when the child cannot even build
    its argument parser (the exact A06 boundary: an eager ``--version``
    metadata probe raised ``PackageNotFoundError`` for every CLI invocation).
    The returned mapping records the exact child origin/interpreter identity
    for durable evidence. No mocks: a tautological in-process import is not
    sufficient — the real module entrypoint must parse in the child.
    """
    command = [sys.executable, "-m", "synth.cli", "--help"]
    completed = subprocess.run(
        command, cwd=root, env=_entrypoint_child_environment(root),
        check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise GuardError(
            "public CLI entrypoint parser refused before contact: "
            f"--help exited {completed.returncode}: {completed.stderr[-2000:]}"
        )
    if "sprint18-iterative-v3" not in completed.stdout:
        raise GuardError(
            "public CLI entrypoint does not expose the bound iterative profile"
        )
    # The child resolves `synth` through the entrypoint PYTHONPATH. In the
    # bound deployment layout the first hit is <root>/src/synth/cli.py;
    # in a bare disposable root (no src tree) Python falls through to the
    # ambient project source, which the caller must pass as `root`. The
    # probe therefore asserts the child origin equals THIS runner's live
    # checkout file (the same bytecode that owns this guard), plus the
    # same-interpreter identity and a successful parser build.
    probe = subprocess.run(
        [
            sys.executable, "-c",
            ("import sys; "
             "import synth.cli as cli; "
             "print(cli.__file__); "
             "print(sys.executable); "
             "print(cli.build_parser() is not None)"),
        ],
        cwd=root, env=_entrypoint_child_environment(root),
        check=False, capture_output=True, text=True,
    )
    lines = probe.stdout.splitlines()
    expected_cli = (ROOT / "src" / "synth" / "cli.py").resolve()
    if (
        probe.returncode != 0
        or len(lines) != 3
        or Path(lines[0]).resolve() != expected_cli
        or Path(lines[1]).resolve() != Path(sys.executable).resolve()
        or lines[2] != "True"
    ):
        raise GuardError(
            "public CLI origin/interpreter probe refused before contact: "
            f"exit {probe.returncode}: {probe.stderr[-2000:]}"
        )
    return {
        "cli_module_origin": str(expected_cli),
        "cli_child_interpreter": str(Path(sys.executable).resolve()),
        "cli_help_returncode": completed.returncode,
        "cli_profile_exposed": PROFILE,
    }


def validate_runtime(binding: dict[str, Any], root: Path) -> None:
    import numpy
    import scipy
    import torch
    import synth

    observed = {
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd().resolve()),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_version": list(sys.version_info[:3]),
        "package_versions": {
            "numpy": numpy.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
        },
        "dependency_origins": {
            "numpy": str(Path(numpy.__file__).resolve()),
            "scipy": str(Path(scipy.__file__).resolve()),
            "torch": str(Path(torch.__file__).resolve()),
            "synth": str(Path(synth.__file__).resolve()),
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_device_total_mib": (
            torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            if torch.cuda.is_available() else None
        ),
    }
    _validate_runtime_observation(binding, root, observed)
    cli_probe = _validate_public_cli_entrypoint(binding, root)
    observed["public_cli_entrypoint"] = cli_probe


def validate_release_receipt(
    binding: dict[str, Any], binding_raw_sha256: str,
    release: dict[str, Any],
) -> str:
    expected = {
        "schema_id": MAIN_RELEASE_SCHEMA_ID,
        "candidate_id": binding["candidate_id"],
        "binding_sha256": binding["binding_sha256"],
        "binding_file_sha256": binding_raw_sha256,
        "source_closure_sha256": binding["source_closure"]["closure_sha256"],
        "source_identity_scheme": SOURCE_IDENTITY_SCHEME,
        "task67_evidence_path": TASK67_EVIDENCE_PATH,
        "task67_evidence_sha256": TASK67_EVIDENCE_SHA256,
        "task67_evidence_reference": TASK67_EVIDENCE_REFERENCE,
        "base_commit": BASE_COMMIT,
        "checkpoint_parent": BASE_COMMIT,
        "release_scope": "Task69-preflight-and-nonconfirmation",
        "status": "RELEASED_BY_MAIN",
        "evidence_review_verdict": "PASS",
        "actionable_findings": 0,
    }
    for key, value in expected.items():
        if release.get(key) != value:
            raise GuardError(f"Main release identity mismatch: {key}")
    if not isinstance(release.get("evidence_review_ref"), str) or not release["evidence_review_ref"]:
        raise GuardError("Main release lacks the Task68 evidence-review reference")
    checkpoint = release.get("checkpoint_commit")
    if (not isinstance(checkpoint, str) or len(checkpoint) != 40
            or any(char not in "0123456789abcdef" for char in checkpoint)):
        raise GuardError("Main release has no recorded full checkpoint SHA")
    return checkpoint


def validate_main_release(
    binding: dict[str, Any], binding_raw_sha256: str, release_path: Path,
    root: Path, *, stage: str, task69_release: dict[str, Any] | None = None,
) -> dict[str, Any]:
    release = read_json(release_path)
    checkpoint = validate_release_receipt(binding, binding_raw_sha256, release)
    parent = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"{checkpoint}^"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if parent != BASE_COMMIT:
        raise GuardError("Main checkpoint does not directly follow the frozen Task68 correction base")
    if stage == "materialize-confirmation":
        if task69_release is None:
            raise GuardError("Confirmation requires Main's separate Task69 release")
        if task69_release.get("task68_checkpoint_commit") != checkpoint:
            raise GuardError("Task69 release is not bound to this Task68 checkpoint")
        expected_head = task69_release["task69_checkpoint_commit"]
        current_parent = subprocess.run(
            ["git", "-C", str(root), "rev-parse", f"{expected_head}^"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        expected_parent = checkpoint
    else:
        expected_head = checkpoint
        expected_parent = BASE_COMMIT
        current_parent = parent
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "--short", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if branch != binding["runtime"]["approved_branch"]:
        raise GuardError("worktree is not on the approved new Sprint18 execution branch")
    if head != expected_head or current_parent != expected_parent:
        raise GuardError("worktree HEAD/checkpoint parent differs from Main release")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise GuardError("remote execution worktree is dirty; use no source mutation or candidate contact")
    origin = subprocess.run(
        ["git", "-C", str(root), "config", "--get", "remote.origin.url"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if origin != binding["runtime"]["origin_url"]:
        raise GuardError("genuine repository origin mismatch")
    common_dir = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if Path(common_dir).resolve() != Path(binding["runtime"]["git_common_dir"]).resolve():
        raise GuardError("worktree does not use the bound canonical Git storage")
    verify_git_source_blobs(binding, root)
    return release


def _canonical_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(record)
    payload.pop("event_sha256", None)
    return payload


def read_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    if ledger_path.is_symlink():
        raise GuardError("attempt ledger must not be a symlink")
    try:
        raw = ledger_path.read_bytes()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise GuardError(f"cannot read attempt ledger: {exc}") from exc
    if raw and not raw.endswith(b"\n"):
        raise GuardError("attempt ledger ends in an incomplete record")
    events: list[dict[str, Any]] = []
    previous = "0" * 64
    for index, line in enumerate(raw.splitlines()):
        try:
            event = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise GuardError(f"attempt ledger is partial/corrupt at event {index}") from exc
        if (event.get("sequence") != index + 1
                or event.get("previous_event_sha256") != previous
                or event.get("event_sha256") != sha256_bytes(canonical_json(_canonical_record(event)))):
            raise GuardError(f"attempt ledger hash chain mismatch at event {index}")
        previous = event["event_sha256"]
        events.append(event)
    return events


def append_event(ledger_path: Path, event_type: str, fields: dict[str, Any]) -> dict[str, Any]:
    events = read_ledger(ledger_path)
    record: dict[str, Any] = {
        "sequence": len(events) + 1,
        "event_type": event_type,
        "previous_event_sha256": events[-1]["event_sha256"] if events else "0" * 64,
        **fields,
    }
    record["event_sha256"] = sha256_bytes(canonical_json(record))
    encoded = canonical_json(record) + b"\n"
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if ledger_path.is_symlink():
        raise GuardError("append-only attempt ledger must not be a symlink")
    descriptor = os.open(ledger_path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "ab", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    if os.name != "nt":
        directory_fd = os.open(ledger_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return record


def _candidate_paths(binding: dict[str, Any], root: Path) -> tuple[Path, Path, Path]:
    root = root.resolve()
    candidate_root = root / binding["candidate_root_relative"]
    attempt_root = candidate_root / binding["attempt_policy"]["attempt_directory"]
    ledger_path = attempt_root / binding["attempt_policy"]["ledger_filename"]
    return candidate_root, attempt_root, ledger_path


def _expected_config_fields(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": binding["candidate_id"],
        "candidate_number": binding["candidate_number"],
        "binding_sha256": binding["binding_sha256"],
        "source_closure_sha256": binding["source_closure"]["closure_sha256"],
        "role_order": [
            {"role": entry["role"], "history_id": entry["history_id"],
             "data_seed": entry["data_seed"], "directory": entry["directory"]}
            for entry in binding["role_binding"]
        ],
        "attempt_number": 1,
    }






def _load_stage_state(binding: dict[str, Any], root: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    candidate_root, attempt_root, ledger_path = _candidate_paths(binding, root)
    _assert_safe_candidate_path(root, candidate_root)
    _assert_safe_candidate_path(root, attempt_root)
    _assert_safe_candidate_path(root, ledger_path)
    marker_path = attempt_root / "attempt.json"
    _assert_safe_candidate_path(root, marker_path)
    if not candidate_root.is_dir():
        raise GuardError("candidate root is absent; no stage may continue")
    if not attempt_root.is_dir():
        raise GuardError("candidate attempt marker is absent; no stage may continue")
    events = read_ledger(ledger_path)
    if not events or events[0].get("event_type") != "attempt_started":
        raise GuardError("candidate ledger has no durable pre-attempt marker")
    marker = read_json(marker_path)
    frozen = _expected_config_fields(binding)
    if any(marker.get(key) != value for key, value in frozen.items()):
        raise GuardError("pre-attempt marker differs from the released binding")
    if any(events[0].get(key) != value for key, value in marker.items()):
        raise GuardError("ledger does not preserve the exact durable pre-attempt marker")
    if any(events[0].get(key) != value for key, value in frozen.items()):
        raise GuardError("ledger pre-attempt marker differs from the released binding")
    if any(event.get("event_type") in {"candidate_rejected", "candidate_retired"} for event in events):
        raise GuardError("candidate has already failed or been retired")
    return candidate_root, attempt_root, events


def _fail_candidate(ledger_path: Path, binding: dict[str, Any], reason: str) -> None:
    append_event(ledger_path, "candidate_rejected", {
        "candidate_id": binding["candidate_id"],
        "binding_sha256": binding["binding_sha256"],
        "reason": reason,
    })


def run_preflight(binding: dict[str, Any], root: Path) -> int:
    candidate_root, attempt_root, preflight_ledger = _candidate_paths(binding, root)
    _assert_safe_candidate_path(root, candidate_root)
    _assert_safe_candidate_path(root, attempt_root)
    candidate_root.parent.mkdir(parents=True, exist_ok=True)
    candidate_root.mkdir(exist_ok=False)
    attempt_root.mkdir(exist_ok=False)
    marker = _expected_config_fields(binding)
    marker.update({
        "schema_id": "sprint18-iterative-pre-attempt-v1",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    marker_path = attempt_root / "attempt.json"
    with marker_path.open("xb") as handle:
        handle.write(canonical_json(marker) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory_chain(attempt_root, root)
    append_event(preflight_ledger, "attempt_started", marker)
    append_event(preflight_ledger, "preflight_started", {
        "candidate_id": binding["candidate_id"], "profile_id": PROFILE,
        "protocol_id": PROTOCOL,
        "in_memory_waveforms_expected": True,
        "persisted_role_roots_expected": False,
        "seed_order": [entry["data_seed"] for entry in binding["role_binding"]],
        "role_order": [entry["role"] for entry in binding["role_binding"]],
    })
    try:
        from synth.preflight15 import run_preflight_iterative_v3
        result = run_preflight_iterative_v3(
            [entry["data_seed"] for entry in binding["role_binding"]],
            [entry["role"] for entry in binding["role_binding"]],
        )
        output = attempt_root / "preflight-raw.json"
        with output.open("xb") as handle:
            raw = canonical_json(result) + b"\n"
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_bytes(raw)
        append_event(preflight_ledger, "preflight_recorded", {
            "result_path": str(output), "result_sha256": digest,
            "verdict": result.get("verdict"),
            "feasible_count": result.get("feasible_count"),
            "in_memory_waveforms_generated": True,
            "persisted_candidate_role_roots": False,
            "persisted_candidate_shards": False,
            "persisted_candidate_manifests": False,
        })
        if result.get("verdict") != "PREFLIGHT-PASS":
            _fail_candidate(preflight_ledger, binding, "fixed_16_history_preflight_rejected")
            return 2
        return 0
    except BaseException as exc:
        _fail_candidate(preflight_ledger, binding, f"preflight_exception:{type(exc).__name__}:{exc}")
        raise

def _release_stage_precondition(release: dict[str, Any], stage: str) -> None:
    if stage not in {
        "preflight", "materialize-nonconfirmation",
        "record-nonconfirmation-pass", "materialize-confirmation",
    }:
        raise GuardError("unsupported candidate execution stage")
    if release.get("release_scope") != "Task69-preflight-and-nonconfirmation":
        raise GuardError("Task68 release does not authorize this stage")

def _verify_preflight_pass(
    binding: dict[str, Any], attempt_root: Path, events: list[dict[str, Any]],
) -> None:
    recorded_events = [
        event for event in events if event.get("event_type") == "preflight_recorded"
    ]
    if len(recorded_events) != 1 or recorded_events[0].get("verdict") != "PREFLIGHT-PASS":
        raise GuardError("fixed-order preflight has not passed exactly once")
    raw_path = attempt_root / "preflight-raw.json"
    _assert_safe_candidate_path(attempt_root, raw_path)
    recorded = recorded_events[0]
    if sha256_file(raw_path) != recorded.get("result_sha256"):
        raise GuardError("raw preflight record digest mismatch")
    result = read_json(raw_path)
    entries = binding["role_binding"]
    expected_seeds = [entry["data_seed"] for entry in entries]
    expected_roles = [entry["role"] for entry in entries]
    expected_count = f"{len(entries)}/{len(entries)}"
    results = result.get("results")
    if (result.get("protocol") != PROTOCOL
            or result.get("seeds") != expected_seeds
            or result.get("feasible_count") != expected_count
            or result.get("verdict") != "PREFLIGHT-PASS"
            or not isinstance(results, list)
            or len(results) != len(entries)):
        raise GuardError("raw preflight record is not the exact fixed roster PASS")
    for index, (item, seed, role) in enumerate(zip(results, expected_seeds, expected_roles, strict=True)):
        qualification = item.get("qualification")
        if (item.get("history_seed") != seed or item.get("role") != role
                or item.get("feasible") is not True
                or item.get("in_memory_waveforms_generated") is not True
                or item.get("persisted_candidate_root") is not False
                or item.get("persisted_candidate_shards") is not False
                or item.get("persisted_candidate_manifest") is not False
                or item.get("no_write") is not True
                or not isinstance(qualification, dict)
                or qualification.get("passed") is not True):
            raise GuardError(f"preflight result {index} fails its fixed role/no-write invariant")


def _materialize_entries(
    binding: dict[str, Any], root: Path, attempt_root: Path, ledger_path: Path,
    events: list[dict[str, Any]], entries: list[dict[str, Any]],
) -> int:
    completed = [event for event in events if event.get("event_type") == "role_materialized"]
    started = [event for event in events if event.get("event_type") == "role_materialization_started"]
    completed_ids = [event.get("history_id") for event in completed]
    started_ids = [event.get("history_id") for event in started]
    if started_ids != completed_ids:
        _fail_candidate(ledger_path, binding, "interrupted_materialization_no_resume")
        raise GuardError("interrupted materialization retires candidate; no resume")
    expected_ids = [entry["history_id"] for entry in binding["role_binding"]]
    if completed_ids != expected_ids[:len(completed_ids)]:
        raise GuardError("materialization ledger is not an exact ordered prefix")
    next_index = len(completed)
    for entry in entries:
        index = expected_ids.index(entry["history_id"])
        if index != next_index:
            raise GuardError("requested role is not the next immutable roster coordinate")
        target = root / entry["directory"]
        _assert_safe_candidate_path(root, target)
        if target.is_symlink() or target.exists():
            _fail_candidate(ledger_path, binding, f"role_output_already_exists:{entry['history_id']}")
            raise GuardError("bound role output exists; no overwrite or replacement")
        append_event(ledger_path, "role_materialization_started", {
            "candidate_id": binding["candidate_id"],
            "history_id": entry["history_id"], "role": entry["role"],
            "data_seed": entry["data_seed"], "directory": str(target),
        })
        command = [
            sys.executable, "-m", "synth.cli", "--chronological",
            "--profile", PROFILE, "--seed", str(entry["data_seed"]),
            "--role", entry["history_id"], "--protocol", PROTOCOL,
            "--channels", "6", "--shard-size", "64", "--output", str(target),
        ]
        env = _entrypoint_child_environment(root)
        completed_process = subprocess.run(command, cwd=root, env=env, check=False)
        if completed_process.returncode != 0:
            _fail_candidate(ledger_path, binding, f"public_cli_failed:{entry['history_id']}:{completed_process.returncode}")
            return completed_process.returncode
        try:
            manifest = target / "manifest.json"
            manifest_hash = sha256_file(manifest)
            from synth.chronicle import load_chronological
            samples, loaded_manifest = load_chronological(target)
            if loaded_manifest.get("role") != entry["history_id"]:
                raise GuardError("public loader returned a different role identity")
            if loaded_manifest.get("protocol") != PROTOCOL:
                raise GuardError("public loader returned a different manifest protocol")
            if loaded_manifest.get("config_hash") != entry["config_hash"]:
                raise GuardError("public loader returned a different config hash")
        except BaseException as exc:
            _fail_candidate(ledger_path, binding, f"public_loader_failed:{entry['history_id']}:{type(exc).__name__}:{exc}")
            raise
        append_event(ledger_path, "role_materialized", {
            "candidate_id": binding["candidate_id"],
            "history_id": entry["history_id"], "role": entry["role"],
            "data_seed": entry["data_seed"], "directory": str(target),
            "manifest_sha256": manifest_hash, "sample_count": len(samples),
            "loader_manifest_role": loaded_manifest["role"],
        })
        next_index += 1
        events = read_ledger(ledger_path)
    return 0



def _verify_qualification_record(
    binding: dict[str, Any], record_path: Path, expected_sha256: str,
) -> dict[str, Any]:
    actual = sha256_file(record_path)
    if actual != expected_sha256:
        raise GuardError("Task69 qualification record SHA-256 mismatch")
    record = read_json(record_path)
    expected_ids = [entry["history_id"] for entry in binding["role_binding"][:12]]
    if (record.get("schema_id") != "sprint18-iterative-nonconfirmation-pass-v1"
            or record.get("candidate_id") != binding["candidate_id"]
            or record.get("binding_sha256") != binding["binding_sha256"]
            or record.get("role_ids") != expected_ids
            or record.get("verdict") != "PASS"
            or record.get("all_nonconfirmation_gates_pass") is not True):
        raise GuardError("Task69 record does not certify the exact 12-role preceding block")
    if record.get("fit_probe_fit_history_ids") != [
        entry["history_id"] for entry in binding["role_binding"] if entry["role"] == "FIT"
    ]:
        raise GuardError("fixed probe was not fitted on exactly the bound Fit histories")
    if record.get("probe_calibration_history_id") != next(
        entry["history_id"] for entry in binding["role_binding"] if entry["role"] == "CALIBRATION"
    ):
        raise GuardError("fixed probe calibration did not use the bound Calibration history")
    if record.get("confirmation_contacted") is not False:
        raise GuardError("Task69 record must certify that Confirmation was not contacted")
    checks = record.get("checks_by_history")
    if not isinstance(checks, dict) or set(checks) != set(expected_ids):
        raise GuardError("Task69 record has missing or extra per-history gate results")
    for history_id, values in checks.items():
        if not isinstance(values, dict) or not values or not all(value is True for value in values.values()):
            raise GuardError(f"Task69 gates do not all pass for {history_id}")
    return record


def record_nonconfirmation_result(
    binding: dict[str, Any], root: Path, record_path: Path,
    expected_sha256: str,
) -> int:
    _, attempt_root, pass_ledger_path = _candidate_paths(binding, root)
    _, _, events = _load_stage_state(binding, root)
    _verify_preflight_pass(binding, attempt_root, events)
    materialized = [event for event in events if event.get("event_type") == "role_materialized"]
    if [event.get("history_id") for event in materialized] != [
        entry["history_id"] for entry in binding["role_binding"][:12]
    ]:
        raise GuardError("all 12 Design/Fit/Calibration/Development roots must reload before Task69 qualification")
    if any(event.get("event_type") == "nonconfirmation_qualification_pass" for event in events):
        raise GuardError("Task69 qualification PASS is already recorded; do not repeat it")
    try:
        record = _verify_qualification_record(binding, record_path, expected_sha256)
    except GuardError as exc:
        _fail_candidate(pass_ledger_path, binding, f"nonconfirmation_qualification_rejected:{exc}")
        raise
    append_event(pass_ledger_path, "nonconfirmation_qualification_pass", {
        "candidate_id": binding["candidate_id"],
        "qualification_path": str(record_path),
        "qualification_sha256": expected_sha256,
        "fit_probe_summary_sha256": record.get("fit_probe_summary_sha256"),
        "calibration_summary_sha256": record.get("calibration_summary_sha256"),
        "development_summary_sha256": record.get("development_summary_sha256"),
    })
    return 0


def _validate_task69_release_identity(
    binding: dict[str, Any], qualification_sha256: str, release: dict[str, Any],
    task68_checkpoint: str,
) -> str:
    expected = {
        "schema_id": "sprint18-task69-main-release-v1",
        "candidate_id": binding["candidate_id"],
        "binding_sha256": binding["binding_sha256"],
        "qualification_sha256": qualification_sha256,
        "task68_checkpoint_commit": task68_checkpoint,
        "task67_evidence_path": TASK67_EVIDENCE_PATH,
        "task67_evidence_sha256": TASK67_EVIDENCE_SHA256,
        "task67_evidence_reference": TASK67_EVIDENCE_REFERENCE,
        "evidence_review_verdict": "PASS",
        "actionable_findings": 0,
        "verdict": "PASS",
        "release_scope": "Task70-confirmation-after-Task69-PASS",
        "status": "RELEASED_BY_MAIN",
    }
    if any(release.get(key) != value for key, value in expected.items()):
        raise GuardError("Task69 Main release does not authorize this candidate's Confirmation stage")
    if not isinstance(release.get("evidence_review_ref"), str) or not release["evidence_review_ref"]:
        raise GuardError("Task69 release lacks its evidence-review reference")
    checkpoint = release.get("task69_checkpoint_commit")
    if (not isinstance(checkpoint, str) or len(checkpoint) != 40
            or any(char not in "0123456789abcdef" for char in checkpoint)):
        raise GuardError("Task69 release lacks its recorded project checkpoint")
    return checkpoint


def _validate_task69_checkpoint_parent(parent: str, task68_checkpoint: str) -> None:
    if parent != task68_checkpoint:
        raise GuardError("Task69 checkpoint does not follow the Task68 checkpoint")


def verify_task69_release(
    binding: dict[str, Any], qualification_sha256: str, path: Path,
    task68_checkpoint: str, root: Path,
) -> dict[str, Any]:
    release = read_json(path)
    checkpoint = _validate_task69_release_identity(
        binding, qualification_sha256, release, task68_checkpoint,
    )
    parent = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"{checkpoint}^"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    _validate_task69_checkpoint_parent(parent, task68_checkpoint)
    return release


def run_confirmation(
    binding: dict[str, Any], root: Path, record_path: Path,
    record_sha256: str, task69_release_path: Path, task68_checkpoint: str,
) -> int:
    record = _verify_qualification_record(binding, record_path, record_sha256)
    task69_release = verify_task69_release(
        binding, record_sha256, task69_release_path, task68_checkpoint, root,
    )
    _, attempt_root, confirmation_ledger_path = _candidate_paths(binding, root)
    _, _, events = _load_stage_state(binding, root)
    if not any(event.get("event_type") == "nonconfirmation_qualification_pass"
               and event.get("qualification_sha256") == record_sha256 for event in events):
        raise GuardError("Task69 preceding-role PASS is not durably recorded")
    if record.get("confirmation_contacted") is not False:
        raise GuardError("Task69 record must certify that Confirmation was not contacted")
    completed = [event for event in events if event.get("event_type") == "role_materialized"]
    completed_ids = [event.get("history_id") for event in completed]
    expected_prefix = [entry["history_id"] for entry in binding["role_binding"][:12]]
    if completed_ids[:12] != expected_prefix:
        raise GuardError("all 12 preceding roles must be materialized before Confirmation")
    confirmation_ids = [entry["history_id"] for entry in binding["role_binding"][12:]]
    confirmation_tail = completed_ids[12:]
    if (len(completed_ids) > 16
            or confirmation_tail != confirmation_ids[:len(confirmation_tail)]):
        raise GuardError("Confirmation materialization is not an ordered roster prefix")
    confirmation_started = [
        event for event in events
        if event.get("event_type") == "role_materialization_started"
        and event.get("history_id") in confirmation_ids
    ]
    confirmation_completed = [
        event for event in completed if event.get("history_id") in confirmation_ids
    ]
    authorized = any(
        event.get("event_type") == "confirmation_materialization_authorized"
        for event in events
    )
    if confirmation_started and not authorized:
        _fail_candidate(confirmation_ledger_path, binding, "confirmation_materialization_without_release")
        raise GuardError("Confirmation started without its separate Main authorization")
    if authorized:
        if (len(confirmation_completed) == 4 and len(confirmation_started) == 4
                and [event.get("history_id") for event in confirmation_completed] == confirmation_ids):
            raise GuardError("Confirmation is already materialized; do not replay it")
        _fail_candidate(confirmation_ledger_path, binding, "interrupted_confirmation_no_resume")
        raise GuardError("interrupted Confirmation attempt retires candidate; no resume")
    append_event(confirmation_ledger_path, "confirmation_materialization_authorized", {
        "candidate_id": binding["candidate_id"],
        "qualification_sha256": record_sha256,
        "task69_release_path": str(task69_release_path),
        "task69_checkpoint_commit": task69_release["task69_checkpoint_commit"],
    })
    return _materialize_entries(
        binding, root, attempt_root, confirmation_ledger_path, read_ledger(confirmation_ledger_path),
        binding["role_binding"][12:],
    )

def _runtime_observation() -> dict[str, Any]:
    return {"hostname": socket.gethostname(), "python": sys.version.split()[0],
            "executable": str(Path(sys.executable).resolve()),
            "cwd": str(Path.cwd().resolve())}


def _smoke_binding(binding: dict[str, Any], root: Path) -> dict[str, Any]:
    """Build a disposable validation-only binding without writing any paths."""
    from dataclasses import asdict
    from synth.chronicle import sprint18_iterative_v3_history_config

    fixture = copy.deepcopy(binding)
    fixture["candidate_id"] = "S18-ITER-SMOKE-0001"
    fixture["candidate_number"] = 1
    fixture["candidate_root_relative"] = f"data/generated/sprint18-iterative-v1/{fixture['candidate_id']}"
    fixture["seed_block"]["first_seed"] = 93000
    roles = [entry["role"] for entry in fixture["role_binding"]]
    fixture["role_binding"] = []
    fixture["configs"] = {}
    for index, role in enumerate(roles):
        seed = 93000 + index
        ordinal = sum(1 for prior in roles[:index] if prior == role) + 1
        history_id = f"S18I-ITER-SMOKE-0001-{ROLE_SUFFIX[role]}-{ordinal:02d}"
        directory = f"{fixture['candidate_root_relative']}/{role}/{history_id}"
        cfg = sprint18_iterative_v3_history_config(seed=seed)
        resolved = asdict(cfg)
        fixture["configs"][str(seed)] = {
            "config_hash": cfg.hash(),
            "config_sha256": sha256_bytes(canonical_json(resolved)),
            "seed_fields": {field: resolved[field]["seed"] for field in SEED_FIELDS},
            "seed_independent_sha256": sha256_bytes(canonical_json({
                key: ({**value, "seed": 0} if key in SEED_FIELDS else value)
                for key, value in resolved.items()
            })),
        }
        fixture["role_binding"].append({
            "role": role, "history_id": history_id, "data_seed": seed,
            "directory": directory, "permitted_use": PERMISSIONS[role],
            "config_hash": cfg.hash(),
        })
    fixture["collision_catalog"]["candidate_intersection"] = []
    fixture["runtime"]["uv_version"] = HOST_UV_VERSION
    fixture["runtime"]["uv_build"] = HOST_UV_BUILD
    fixture.pop("binding_sha256", None)
    fixture["binding_sha256"] = _candidate_digest(fixture)
    return fixture

def run_no_contact_smoke(binding: dict[str, Any], root: Path) -> None:
    expected_seeds = list(range(32064, 32080))
    entries = _role_entries(binding)
    expected_history_ids: list[str] = []
    ordinals: dict[str, int] = {}
    for role in ROLE_IDS:
        ordinals[role] = ordinals.get(role, 0) + 1
        expected_history_ids.append(
            f"S18I-ITER-0005-{ROLE_SUFFIX[role]}-{ordinals[role]:02d}"
        )
    seed_block = binding.get("seed_block", {})
    if (
        binding.get("candidate_id") != "S18-ITER-0005"
        or binding.get("candidate_number") != 5
        or binding.get("contact_authorized") is not False
        or seed_block.get("first_seed") != 32064
        or seed_block.get("size") != 16
        or seed_block.get("stride") != 1
        or seed_block.get("seeds") != expected_seeds
        or [entry.get("role") for entry in entries] != list(ROLE_IDS)
        or [entry.get("data_seed") for entry in entries] != expected_seeds
        or [entry.get("history_id") for entry in entries] != expected_history_ids
    ):
        raise GuardError("no-contact smoke requires the exact unreleased candidate-5 binding")
    validate_binding(binding, root, disposable=True, check_sources=False)
    fixture = _smoke_binding(binding, root)
    validate_binding(fixture, root, disposable=True, check_sources=True)
    child_source = subprocess.run(
        [
            sys.executable, "-c",
            (
                "import importlib, synth, sys; "
                "unqualified = importlib.import_module('sprint18_task5_measurability'); "
                "qualified = importlib.import_module('experiments.sprint18_task5_measurability'); "
                "print(synth.__file__); print(sys.dont_write_bytecode); "
                "print(unqualified.__file__); print(qualified.__file__)"
            ),
        ],
        cwd=root, env=_entrypoint_child_environment(root), check=False,
        capture_output=True, text=True,
    )
    child_lines = child_source.stdout.splitlines()
    helper_path = (root / "experiments" / "sprint18_task5_measurability.py").resolve()
    if (
        child_source.returncode != 0
        or len(child_lines) != 4
        or Path(child_lines[0]).resolve() != (root / "src" / "synth" / "__init__.py").resolve()
        or child_lines[1] != "True"
        or Path(child_lines[2]).resolve() != helper_path
        or Path(child_lines[3]).resolve() != helper_path
    ):
        raise GuardError(
            "child import smoke did not resolve bound synth and both experiment import forms"
        )
    child_source_origin_smoke = "PASS_REAL_SUBPROCESS_BOUND_SYNTH_AND_EXPERIMENTS"
    cli_probe = _validate_public_cli_entrypoint(binding, root)
    if cli_probe["cli_profile_exposed"] != PROFILE:
        raise GuardError("public CLI entrypoint does not expose the bound iterative profile")
    child_cli_origin_smoke = "PASS_REAL_SUBPROCESS_PUBLIC_CLI_PARSER"
    source_identity_passes: list[str] = []
    for newline_style in ("LF", "CRLF"):
        with tempfile.TemporaryDirectory(prefix="s18-source-identity-") as temp:
            variant_root = Path(temp)
            for relative in sorted(EXPECTED_SOURCE_PATHS):
                canonical = canonical_source_bytes(
                    relative, _source_path(root, relative).read_bytes(),
                )
                variant = (
                    canonical if newline_style == "LF"
                    else canonical.replace(b"\n", b"\r\n")
                )
                target = variant_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(variant)
            verify_source_closure(fixture, variant_root)
        source_identity_passes.append(newline_style)

    source_negative_cases: list[str] = []
    with tempfile.TemporaryDirectory(prefix="s18-source-mutation-") as temp:
        variant_root = Path(temp)
        for relative in sorted(EXPECTED_SOURCE_PATHS):
            canonical = canonical_source_bytes(
                relative, _source_path(root, relative).read_bytes(),
            )
            target = variant_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(canonical)
        chronicle = variant_root / "src/synth/chronicle.py"
        chronicle.write_bytes(chronicle.read_bytes() + b"\n# content mutation\n")
        try:
            verify_source_closure(fixture, variant_root)
        except GuardError:
            source_negative_cases.append("non-eol-content-mutation")
        else:
            raise GuardError("source content mutation unexpectedly passed")
        mixed = canonical_source_bytes(
            "src/synth/chronicle.py", _source_path(root, "src/synth/chronicle.py").read_bytes(),
        )
        mixed_target = variant_root / "src/synth/chronicle.py"
        mixed_target.write_bytes(mixed.replace(b"\n", b"\r\n", 1))
        try:
            verify_source_closure(fixture, variant_root)
        except GuardError:
            source_negative_cases.append("mixed-lf-crlf")
        else:
            raise GuardError("mixed LF/CRLF source unexpectedly passed")
        mixed_target.write_bytes(mixed.replace(b"\n", b"\r", 1))
        try:
            verify_source_closure(fixture, variant_root)
        except GuardError:
            source_negative_cases.append("bare-cr")
        else:
            raise GuardError("bare-CR source unexpectedly passed")
        mixed_target.write_bytes(mixed + b"\0")
        try:
            verify_source_closure(fixture, variant_root)
        except GuardError:
            source_negative_cases.append("non-text-source")
        else:
            raise GuardError("NUL-containing source unexpectedly passed")
    runner_relative = Path(__file__).resolve().relative_to(root.resolve()).as_posix()
    retirement_relative = "experiments/sprint18-candidate4-operational-retirement-v1.json"
    cli_relative = "src/synth/cli.py"
    bound_git_blobs = binding["source_closure"]["git_blob_oid_by_path"]
    unchanged_git_blobs = {
        relative: oid for relative, oid in bound_git_blobs.items()
        if relative not in (runner_relative, retirement_relative, cli_relative)
    }
    if len(unchanged_git_blobs) != 50:
        raise GuardError("source closure does not contain exactly 50 unchanged blob members")
    verify_git_source_blobs(
        {"source_closure": {"git_blob_oid_by_path": unchanged_git_blobs}}, root,
    )
    runner_blob_oid = subprocess.run(
        ["git", "hash-object", f"--path={runner_relative}", runner_relative],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if runner_blob_oid != bound_git_blobs[runner_relative]:
        raise GuardError("expected path-filtered runner Git blob differs from its frozen identity")
    cli_blob_oid = subprocess.run(
        ["git", "hash-object", f"--path={cli_relative}", cli_relative],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if cli_blob_oid != bound_git_blobs[cli_relative]:
        raise GuardError("expected path-filtered CLI Git blob differs from its frozen identity")
    retirement_blob_oid = subprocess.run(
        ["git", "hash-object", f"--path={retirement_relative}", retirement_relative],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if retirement_blob_oid != bound_git_blobs[retirement_relative]:
        raise GuardError("expected path-filtered retirement Git blob differs from its frozen identity")
    bad_git_blobs = copy.deepcopy(unchanged_git_blobs)
    bad_git_blobs["src/synth/chronicle.py"] = "0" * 40
    try:
        verify_git_source_blobs(
            {"source_closure": {"git_blob_oid_by_path": bad_git_blobs}}, root,
        )
    except GuardError:
        pass
    else:
        raise GuardError("Git source blob mismatch unexpectedly passed")
    source_negative_cases.append("git-source-blob-mismatch")
    recorded_runtime_dependencies = {
        "package_versions": {
            "numpy": "2.5.2", "scipy": "1.18.1", "torch": "2.14.0+cu130",
        },
        "dependency_origins": {
            "numpy": f"{EXECUTION_ROOT}/.venv/lib/python3.12/site-packages/numpy/__init__.py",
            "scipy": f"{EXECUTION_ROOT}/.venv/lib/python3.12/site-packages/scipy/__init__.py",
            "torch": f"{EXECUTION_ROOT}/.venv/lib/python3.12/site-packages/torch/__init__.py",
            "synth": f"{EXECUTION_ROOT}/src/synth/__init__.py",
        },
    }
    _validate_runtime_dependencies(fixture, recorded_runtime_dependencies)
    runtime_negative_cases: list[str] = []
    for label, field, value in (
        (
            "runtime-torch-unqualified-release",
            "package_versions",
            {**recorded_runtime_dependencies["package_versions"], "torch": "2.14.0"},
        ),
        (
            "runtime-torch-foreign-build",
            "package_versions",
            {**recorded_runtime_dependencies["package_versions"], "torch": "2.14.0+cu131"},
        ),
        (
            "runtime-synth-foreign-origin",
            "dependency_origins",
            {
                **recorded_runtime_dependencies["dependency_origins"],
                "synth": "/__foreign__/src/synth/__init__.py",
            },
        ),
        (
            "runtime-synth-missing-origin",
            "dependency_origins",
            {
                name: path for name, path
                in recorded_runtime_dependencies["dependency_origins"].items()
                if name != "synth"
            },
        ),
    ):
        bad_observation = copy.deepcopy(recorded_runtime_dependencies)
        bad_observation[field] = value
        try:
            _validate_runtime_dependencies(fixture, bad_observation)
        except GuardError:
            runtime_negative_cases.append(label)
        else:
            raise GuardError(f"negative runtime dependency smoke unexpectedly passed: {label}")

    entrypoint_negative_cases: list[str] = []
    host_uv_negative_cases: list[str] = []
    stale_binding = copy.deepcopy(fixture)
    stale_binding["runtime"]["uv_version"] = "0.12.20"
    stale_binding["runtime"]["uv_build"] = "2274b80d6"
    try:
        _validate_host_uv_toolchain(stale_binding)
    except GuardError:
        host_uv_negative_cases.append("host-uv-stale-binding-pin")
    else:
        raise GuardError("stale host uv binding pin unexpectedly passed")
    wrong_version = copy.deepcopy(fixture)
    wrong_version["runtime"]["uv_version"] = "0.0.0"
    wrong_version["runtime"]["uv_build"] = HOST_UV_BUILD
    try:
        _validate_host_uv_toolchain(wrong_version)
    except GuardError:
        host_uv_negative_cases.append("host-uv-wrong-version-pin")
    else:
        raise GuardError("wrong host uv version pin unexpectedly passed")
    wrong_build = copy.deepcopy(fixture)
    wrong_build["runtime"]["uv_version"] = HOST_UV_VERSION
    wrong_build["runtime"]["uv_build"] = "0" * 9
    try:
        _validate_host_uv_toolchain(wrong_build)
    except GuardError:
        host_uv_negative_cases.append("host-uv-wrong-build-pin")
    else:
        raise GuardError("wrong host uv build pin unexpectedly passed")
    expected_pythonpath = _entrypoint_pythonpath(root)
    saved_environ = dict(os.environ)
    for label, key, invalid_value in (
        ("entrypoint-missing-pythonpath", "PYTHONPATH", None),
        (
            "entrypoint-foreign-pythonpath",
            "PYTHONPATH",
            expected_pythonpath + os.pathsep + str(root / "foreign-src"),
        ),
        ("entrypoint-bytecode-enabled", "PYTHONDONTWRITEBYTECODE", "0"),
    ):
        existed = key in os.environ
        previous = os.environ.get(key)
        if invalid_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = invalid_value
        try:
            _validate_entrypoint_environment(fixture, root, disposable=True)
        except GuardError:
            entrypoint_negative_cases.append(label)
        else:
            raise GuardError(f"negative entrypoint environment smoke unexpectedly passed: {label}")
        finally:
            if existed:
                os.environ[key] = previous if previous is not None else ""
            else:
                os.environ.pop(key, None)
    os.environ.clear()
    os.environ.update(saved_environ)

    raw_sha = sha256_bytes(canonical_json(fixture) + b"\n")
    release = {
        "schema_id": MAIN_RELEASE_SCHEMA_ID,
        "candidate_id": fixture["candidate_id"],
        "binding_sha256": fixture["binding_sha256"],
        "binding_file_sha256": raw_sha,
        "source_closure_sha256": fixture["source_closure"]["closure_sha256"],
        "source_identity_scheme": SOURCE_IDENTITY_SCHEME,
        "task67_evidence_path": TASK67_EVIDENCE_PATH,
        "task67_evidence_sha256": TASK67_EVIDENCE_SHA256,
        "task67_evidence_reference": TASK67_EVIDENCE_REFERENCE,
        "base_commit": BASE_COMMIT,
        "checkpoint_parent": BASE_COMMIT,
        "checkpoint_commit": "a" * 40,
        "release_scope": "Task69-preflight-and-nonconfirmation",
        "status": "RELEASED_BY_MAIN",
        "evidence_review_verdict": "PASS",
        "evidence_review_ref": "disposable-no-contact-smoke",
        "actionable_findings": 0,
    }
    if validate_release_receipt(fixture, raw_sha, release) != "a" * 40:
        raise GuardError("positive disposable release receipt validation failed")

    task68_checkpoint = "a" * 40
    task69_checkpoint = "b" * 40
    qualification_sha256 = "c" * 64
    task69_release = {
        "schema_id": "sprint18-task69-main-release-v1",
        "candidate_id": fixture["candidate_id"],
        "binding_sha256": fixture["binding_sha256"],
        "qualification_sha256": qualification_sha256,
        "task68_checkpoint_commit": task68_checkpoint,
        "task67_evidence_path": TASK67_EVIDENCE_PATH,
        "task67_evidence_sha256": TASK67_EVIDENCE_SHA256,
        "task67_evidence_reference": TASK67_EVIDENCE_REFERENCE,
        "evidence_review_verdict": "PASS",
        "evidence_review_ref": "disposable-task69-smoke",
        "actionable_findings": 0,
        "verdict": "PASS",
        "release_scope": "Task70-confirmation-after-Task69-PASS",
        "status": "RELEASED_BY_MAIN",
        "task69_checkpoint_commit": task69_checkpoint,
    }
    if _validate_task69_release_identity(
        fixture, qualification_sha256, task69_release, task68_checkpoint,
    ) != task69_checkpoint:
        raise GuardError("positive Task69 release identity smoke failed")
    _validate_task69_checkpoint_parent(task68_checkpoint, task68_checkpoint)

    mutations: list[tuple[str, Any]] = []
    bad_order = copy.deepcopy(fixture)
    bad_order["role_binding"][0], bad_order["role_binding"][1] = bad_order["role_binding"][1], bad_order["role_binding"][0]
    bad_order["binding_sha256"] = _candidate_digest(bad_order)
    mutations.append(("role-order", bad_order))
    bad_seed_order = copy.deepcopy(fixture)
    bad_seed_order["role_binding"][0]["data_seed"], bad_seed_order["role_binding"][1]["data_seed"] = (
        bad_seed_order["role_binding"][1]["data_seed"],
        bad_seed_order["role_binding"][0]["data_seed"],
    )
    bad_seed_order["binding_sha256"] = _candidate_digest(bad_seed_order)
    mutations.append(("seed-order", bad_seed_order))
    bad_binding_digest = copy.deepcopy(fixture)
    bad_binding_digest["binding_sha256"] = "0" * 64
    mutations.append(("binding-digest", bad_binding_digest))
    bad_source = copy.deepcopy(fixture)
    bad_source["source_closure"]["sha256_by_path"]["src/synth/chronicle.py"] = "0" * 64
    bad_source["source_closure"]["closure_sha256"] = _source_closure_digest(
        bad_source["source_closure"],
    )
    bad_source["binding_sha256"] = _candidate_digest(bad_source)
    mutations.append(("source-hash", bad_source))
    bad_source_path = copy.deepcopy(fixture)
    bad_source_path["source_closure"]["sha256_by_path"].pop("src/synth/chronicle.py")
    bad_source_path["source_closure"]["sha256_by_path"]["src/synth/not-tracked.py"] = "0" * 64
    bad_source_path["source_closure"]["git_blob_oid_by_path"].pop("src/synth/chronicle.py")
    bad_source_path["source_closure"]["git_blob_oid_by_path"]["src/synth/not-tracked.py"] = "0" * 40
    bad_source_path["source_closure"]["closure_sha256"] = _source_closure_digest(
        bad_source_path["source_closure"],
    )
    bad_source_path["binding_sha256"] = _candidate_digest(bad_source_path)
    mutations.append(("source-path-set", bad_source_path))
    bad_source_policy = copy.deepcopy(fixture)
    bad_source_policy["source_closure"]["identity_scheme"] = "unknown-source-scheme"
    bad_source_policy["source_closure"]["closure_sha256"] = _source_closure_digest(
        bad_source_policy["source_closure"],
    )
    bad_source_policy["binding_sha256"] = _candidate_digest(bad_source_policy)
    mutations.append(("source-identity-scheme", bad_source_policy))
    bad_unknown_policy = copy.deepcopy(fixture)
    bad_unknown_policy["source_closure"]["unknown_scheme_policy"] = "allow"
    bad_unknown_policy["source_closure"]["closure_sha256"] = _source_closure_digest(
        bad_unknown_policy["source_closure"],
    )
    bad_unknown_policy["binding_sha256"] = _candidate_digest(bad_unknown_policy)
    mutations.append(("unknown-scheme-policy", bad_unknown_policy))
    bad_closure = copy.deepcopy(fixture)
    bad_closure["source_closure"]["closure_sha256"] = "0" * 64
    bad_closure["binding_sha256"] = _candidate_digest(bad_closure)
    mutations.append(("source-closure-digest", bad_closure))
    bad_path = copy.deepcopy(fixture)
    bad_path["role_binding"][0]["directory"] += "/wrong"
    bad_path["binding_sha256"] = _candidate_digest(bad_path)
    mutations.append(("role-path", bad_path))
    bad_config = copy.deepcopy(fixture)
    bad_config["configs"]["93000"]["config_hash"] = "0" * 12
    bad_config["binding_sha256"] = _candidate_digest(bad_config)
    mutations.append(("config-hash", bad_config))
    task67_in_closure = copy.deepcopy(fixture)
    task67_in_closure["source_closure"]["sha256_by_path"][TASK67_EVIDENCE_PATH] = (
        TASK67_EVIDENCE_SHA256
    )
    task67_in_closure["source_closure"]["closure_sha256"] = _source_closure_digest(
        task67_in_closure["source_closure"],
    )
    task67_in_closure["binding_sha256"] = _candidate_digest(task67_in_closure)
    mutations.append(("task67-evidence-not-deployable", task67_in_closure))
    for label, mutated in mutations:
        try:
            validate_binding(mutated, root, disposable=True, check_sources=True)
        except GuardError:
            continue
        raise GuardError(f"negative no-contact smoke unexpectedly passed: {label}")
    wrong_release = dict(release, binding_sha256="0" * 64)
    try:
        validate_release_receipt(fixture, raw_sha, wrong_release)
    except GuardError:
        pass
    else:
        raise GuardError("negative Main release identity smoke unexpectedly passed")
    bad_release_parent = dict(
        release, checkpoint_parent="3a15b510ed6fd8a31ff172cb646a7bfa66996715",
    )
    try:
        validate_release_receipt(fixture, raw_sha, bad_release_parent)
    except GuardError:
        pass
    else:
        raise GuardError("Main release with historical checkpoint parent unexpectedly passed")
    old_release = dict(
        release,
        schema_id="sprint18-main-candidate-release-v1",
        base_commit=ACCEPTED_METHOD_LINEAGE_COMMIT,
        checkpoint_parent=ACCEPTED_METHOD_LINEAGE_COMMIT,
    )
    try:
        validate_release_receipt(fixture, raw_sha, old_release)
    except GuardError:
        pass
    else:
        raise GuardError("historical Main release unexpectedly passed the corrected receipt contract")
    bad_task69_evidence = dict(task69_release, task67_evidence_sha256="0" * 64)
    try:
        _validate_task69_release_identity(
            fixture, qualification_sha256, bad_task69_evidence, task68_checkpoint,
        )
    except GuardError:
        pass
    else:
        raise GuardError("negative Task69 evidence provenance smoke unexpectedly passed")
    try:
        _validate_task69_checkpoint_parent("d" * 40, task68_checkpoint)
    except GuardError:
        pass
    else:
        raise GuardError("negative Task69 parent smoke unexpectedly passed")
    print(json.dumps({
        "result": "PASS",
        "mode": "disposable-static-no-contact",
        "disposable_history_ids": [entry["history_id"] for entry in fixture["role_binding"]],
        "configured_role_count": len(fixture["configs"]),
        "positive_binding_and_release": "PASS",
        "task69_release_identity_and_direct_parent": "PASS",
        "source_identity_variants": source_identity_passes,
        "source_identity_git_blob_smoke": "PASS_50_BASE_TREE_RUNNER_CLI_RETIREMENT_FILTERED_EXPECTATION",
        "negative_cases": (
            [label for label, _ in mutations]
            + ["release-identity", "release-parent", "historical-release-schema", "task69-evidence-provenance", "task69-wrong-parent"]
            + source_negative_cases
            + runtime_negative_cases
            + host_uv_negative_cases
            + entrypoint_negative_cases
        ),
        "runtime_binding_status": "PASS_PROSPECTIVE_ONLY",
        "entrypoint_environment_status": "PASS_DISPOSABLE_SOURCE_CONTRACT",
        "child_source_origin_smoke": child_source_origin_smoke,
        "child_cli_origin_smoke": child_cli_origin_smoke,
        "public_cli_entrypoint": cli_probe,
        "host_uv_binding_status": "PASS_PIN_ONLY_WINDOWS_STATIC_SMOKE",
        "host_uv_version": HOST_UV_VERSION,
        "host_uv_build": HOST_UV_BUILD,
        "runtime_dependency_metadata_status": "PASS_RECORDED_A02_METADATA_ONLY",
        "runtime_environment_binding_status": fixture["runtime"]["environment_status_at_binding"],
        "runtime_observation": _runtime_observation(),
        "generator_called": False,
        "preflight_called": False,
        "loader_called": False,
        "probe_called": False,
        "candidate_contact_released": False,
        "candidate_paths_checked_or_written": False,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, default=BINDING_DEFAULT)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--stage", choices=(
        "validate", "preflight", "materialize-nonconfirmation",
        "record-nonconfirmation-pass", "materialize-confirmation",
    ), default="validate")
    parser.add_argument("--qualification-record", type=Path)
    parser.add_argument("--qualification-sha256")
    parser.add_argument("--task69-release", type=Path)
    parser.add_argument("--smoke-no-contact", action="store_true")
    return parser.parse_args()


def dispatch_stage(binding: dict[str, Any], root: Path, stage: str) -> int:
    """Run one public candidate stage end-to-end through the live dispatcher.

    This is the public stage dispatcher: it performs the full
    load → guard → loop → reload control flow for the requested stage using
    the current module bytecode (not a private helper slice). The CLI
    `--stage` entry delegates to this function so both paths are identical.
    """
    if stage == "materialize-nonconfirmation":
        _, attempt_root, nonconfirmation_ledger_path = _candidate_paths(binding, root)
        _, _, events = _load_stage_state(binding, root)
        _verify_preflight_pass(binding, attempt_root, events)
        if any(event.get("event_type") == "role_materialization_started" for event in events):
            started = [event for event in events if event.get("event_type") == "role_materialization_started"]
            materialized = [event for event in events if event.get("event_type") == "role_materialized"]
            if len(started) != len(materialized):
                _fail_candidate(nonconfirmation_ledger_path, binding, "interrupted_materialization_no_resume")
                raise GuardError("incomplete candidate roots are preserved; no resume")
        completed = [event["history_id"] for event in events if event.get("event_type") == "role_materialized"]
        expected = [entry["history_id"] for entry in binding["role_binding"][:12]]
        if completed and completed != expected[:len(completed)]:
            raise GuardError("non-Confirmation materialization is not an ordered prefix")
        return _materialize_entries(
            binding, root, attempt_root, nonconfirmation_ledger_path, events,
            binding["role_binding"][len(completed):12],
        )
    raise GuardError(f"unsupported stage {stage}")

def main() -> int:
    args = parse_args()
    try:
        binding_bytes = args.binding.read_bytes()
        binding = json.loads(binding_bytes.decode("utf-8"))
        if not isinstance(binding, dict):
            raise GuardError("binding JSON root must be an object")
        validate_binding(binding, ROOT, disposable=args.smoke_no_contact)
        if args.smoke_no_contact:
            if args.stage != "validate" or args.release or args.qualification_record or args.task69_release:
                raise GuardError("no-contact smoke accepts no execution stage or release path")
            run_no_contact_smoke(binding, ROOT)
            return 0
        if args.stage == "validate":
            validate_runtime(binding, ROOT)
            host_uv = _validate_host_uv_toolchain(binding)
            print(json.dumps({
                "result": "STATIC_BINDING_PASS", "candidate_id": binding["candidate_id"],
                "binding_sha256": binding["binding_sha256"],
                "binding_file_sha256": sha256_bytes(binding_bytes),
                "source_closure_sha256": binding["source_closure"]["closure_sha256"],
                "contacted": False,
                "host_uv_version": host_uv["uv_version"],
                "host_uv_build": host_uv["uv_build"],
            }, indent=2))
            return 0
        if args.release is None:
            raise GuardError("contact stage requires Main's exact release receipt")
        task69_release = None
        if args.stage == "materialize-confirmation":
            if (args.qualification_record is None or args.qualification_sha256 is None
                    or args.task69_release is None):
                raise GuardError("Confirmation stage requires the exact Task69 result and Main release")
            main_release_preview = read_json(args.release)
            task68_checkpoint = validate_release_receipt(
                binding, sha256_bytes(binding_bytes), main_release_preview,
            )
            task69_release = verify_task69_release(
                binding, args.qualification_sha256, args.task69_release,
                task68_checkpoint, ROOT,
            )
        release = validate_main_release(
            binding, sha256_bytes(binding_bytes), args.release, ROOT,
            stage=args.stage, task69_release=task69_release,
        )
        _release_stage_precondition(release, args.stage)
        _validate_host_uv_toolchain(binding)
        validate_runtime(binding, ROOT)
        if args.stage == "preflight":
            return run_preflight(binding, ROOT)
        if args.stage == "materialize-nonconfirmation":
            return dispatch_stage(binding, ROOT, "materialize-nonconfirmation")
        if args.stage == "record-nonconfirmation-pass":
            if args.qualification_record is None or args.qualification_sha256 is None:
                raise GuardError("Task69 record stage requires its exact path and SHA-256")
            return record_nonconfirmation_result(
                binding, ROOT, args.qualification_record, args.qualification_sha256,
            )
        if args.stage == "materialize-confirmation":
            return run_confirmation(
                binding, ROOT, args.qualification_record,
                args.qualification_sha256, args.task69_release,
                release["checkpoint_commit"],
            )
        raise GuardError(f"unsupported stage {args.stage}")
    except (GuardError, OSError, subprocess.CalledProcessError, ValueError,
            KeyError, TypeError, AttributeError, IndexError) as exc:
        print(f"sprint18 candidate guard refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
