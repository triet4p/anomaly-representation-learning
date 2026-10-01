"""Separately versioned fail-closed Sprint 22 whole-history pilot runner.

The executable requires a byte-bound Task 3 binding and a separate Main
release of that exact digest before generation. ``--dry-run`` does not touch
its output root, materialize histories, deserialize checkpoints, or score.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

RUNNER_ID = "sprint22-pilot-runner-v2"
PROTOCOL_ID = "sprint22-exploratory-pilot-v1"
BINDING_SCHEMA_ID = "sprint22-pilot-binding-v1"
BASE_COMMIT = "8c15f0204a3e495569b7f143dc109943e8b808de"
TASK1_DISPOSABLE_CONTRACT_SHA256 = "dcb9b667f86e9a2240a86022490aae93c948bf5ff88f08dd4dd9c52f9701f1d4"
GENERATOR_PROTOCOL_SHA256 = "a1fc09db2e246ed79d0595aec953a7788fd1b47c4981fbe1d92017d944b8d7b6"
CONFIG_TEMPLATE_SHA256 = "3dfecb64967e3e6a1bfe26f8dd72565f553dbda0130dca639d5569c2e42e7dd6"
CONFIG_SHA256 = "1ff67f95428ef29aab05d9f6394a305b75c8c2a9e12431f3cda09b6958596144"
METRIC_SHA256 = "b2d6af7505abe459a84f76f5279a6a5c4298e7c142901edd4596fcfe3ddeb88f"
MAPPING_SHA256 = "4f769ccc64aac4c6a91fdd356eaa6963afe976ab847a22df23e43f9331418897"
UV_LOCK_SHA256 = "4a7866878c81cdd8f5ba283cd8d0a772073bfc75941f73ec761ede5eeb2e239a"
MODEL_SEEDS = (171701, 171702, 171703)
CHECKPOINT_ROOT = Path("/tmp/sprint17-task7-out/checkpoints")
CHECKPOINTS = {
    171701: ("b0_seed171701_step300.pt", "45f9e151c5d3946804ea531eb2bcace26b1f62e634671f51b8741ac6b411a619"),
    171702: ("b0_seed171702_step300.pt", "64ed2cedec210e4692f60bb4dd3430a57cf94abfcd50551abd97c9265ea785f8"),
    171703: ("b0_seed171703_step300.pt", "9edb2122e357376a9ff1e065d73d5ab7704f8d2005991552332f1ced01b0e906"),
}
CHECKPOINT_BYTES = 18_239_321
BANK_SHA256 = {
    171701: "432852b7c8ba6d1845b26e3e8f4d34f2e986258d39da22b844c15def2bc00662",
    171702: "6bae50edd1aee8ac4a00a6255e85db01b89eeaef7341405ac58c93e0a15b511a",
    171703: "7dfe8c3e63b764f0a043c5c38ba63fced463ce78846d38153bc5588a2a6efff8",
}
BANK_ROWS = 5040
BANK_DIM = 128
BANK_K = 5
HISTORY_COUNT = 32
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260202
REL_TOL = 1e-6
ABS_TOL = 1e-7
MINIMUM_CONTROLS = 48

QUOTA = {
    "a": 16, "a1": 8, "a2": 8, "controls": 48, "p": 24,
    "p1": 12, "p2": 12, "program_cap": 0.6, "programs": 2,
    "robot_days": 240, "robot_neg_cap": 0.4, "robot_pos_cap": 0.35,
    "robots_neg": 6, "robots_pos": 6, "spacing_s": 1209600.0,
    "w": 24, "w1": 12, "w2": 12,
}
HARD_FLOORS = {
    "P": 10, "W": 10, "A": 8, "positive_total": 30,
    "controls": 25, "robot_days": 150, "positive_robots": 6,
    "negative_robots": 6, "programs_each_P_W": 2,
    "cohort_mix_min": 0.15, "cohort_mix_max": 0.60,
    "positive_robot_share_max": 0.35, "negative_robot_share_max": 0.40,
    "program_share_max": 0.60, "lead_support_P_W_min": 0.80,
}
DESIGN_MARGINS = {
    "P": 13, "W": 13, "A": 10, "positive_total": 38,
    "controls": 32, "robot_days": 188,
}
EXPECTED_RUNTIME = {
    "python": "3.12.13", "torch": "2.14.0+cu130", "cuda_available": True,
    "device_name": "NVIDIA GeForce RTX 4060 Ti", "device_total_mib": 16380,
}
PINNED_SOURCE_HASHES = {
    "src/representation/model.py": "7051f5f76be0bcc8e50d6d5ee2941a57c12a3285406d0ef27883c596aa6600f0",
    "src/representation/data.py": "88ea3becbc8328c5b21518b1f957fe710096cb96ed0cd2416fe6f139439d5eb4",
    "src/representation/masking.py": "c0efd33b24b8d01c067198f0bc04c0c90c1e99bad9a6a1ec006b37ce27c0465f",
    "src/representation/contracts.py": "0b23080fc49f218484a6f7939bb9288f07b607686312e91c3b14d9d6b84464ed",
    "src/representation/layers/normalization.py": "bfb5f68f5352115c2e044c27999cdf3fecab7aa3ce2cd05ce021204f1f9edd8d",
    "src/synth/chronicle.py": "b5992d6af3c4387d18df746b43be17b4bfc1fac6938e8b44be35b6fb7c65dafd",
    "src/synth/balanced.py": "d62de3422bd51f870d39d18b1a8bb942f9ca73fc0044d8f23cc2d0948bd35a43",
    "src/synth/events.py": "85100f5e0aca47dd2e8b01a08c58f39d32be4f1eab469cdc38e4a4b57768169d",
}
METRIC_FILES = (
    "src/synth/probe15.py", "src/synth/events.py",
    "src/representation/attribution_metrics.py",
    "src/representation/sprint17_ablation.py",
    "experiments/sprint17_task7_b0.py",
)


class PilotError(RuntimeError):
    """Fail-closed error carrying the pilot evidence disposition."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _expect(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PilotError(code, message)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(code, f"cannot read JSON at {path}: {exc}") from exc
    _expect(isinstance(value, dict), code, f"JSON at {path} must be an object")
    return value


def require_minimum_control_allocation(allocation: dict[str, Any]) -> dict[str, Any]:
    """Accept the real allocator result without reducing its qualifying controls."""
    _expect(isinstance(allocation, dict), "ALLOCATOR_RESULT_INVALID", "allocator result must be a mapping")
    controls = allocation.get("controls")
    _expect(isinstance(controls, list), "ALLOCATOR_RESULT_INVALID", "allocator result has no control list")
    count = len(controls)
    _expect(count >= MINIMUM_CONTROLS, "CONTROL_CONSTRUCTION_SHORTFALL", f"allocator controls {count} < {MINIMUM_CONTROLS}")
    margins = allocation.get("margins")
    _expect(isinstance(margins, dict) and margins.get("controls") == count, "ALLOCATOR_RESULT_INVALID", "allocator control margin differs from returned control list")
    return allocation


def _seed_exclusions() -> set[int]:
    # Historical Sprint 21 coordinates and the one Task 2 allocator-only
    # sentinel are forbidden in any future pilot data-seed binding.
    return {*range(921000, 921032), 220001, *MODEL_SEEDS}


def validate_roster(roster: object, freshness_exclusions: object) -> list[dict[str, Any]]:
    _expect(isinstance(roster, list) and len(roster) == HISTORY_COUNT, "BINDING_MISMATCH", "binding must declare exactly 32 fixed whole histories")
    _expect(isinstance(freshness_exclusions, list), "FRESHNESS_PROVENANCE_MISMATCH", "freshness exclusion catalog must be an explicit seed list")
    excluded = set(_seed_exclusions())
    for item in freshness_exclusions:
        _expect(type(item) is int, "FRESHNESS_PROVENANCE_MISMATCH", "freshness exclusion seed must be an integer")
        excluded.add(item)
    history_ids: list[str] = []
    seeds: list[int] = []
    roots: list[str] = []
    clean: list[dict[str, Any]] = []
    for record in roster:
        _expect(isinstance(record, dict) and set(record) == {"history_id", "data_seed", "relative_root", "config_sha256", "config_hash"}, "BINDING_MISMATCH", "each roster coordinate must have exactly the reviewed fields")
        history_id = record["history_id"]
        seed = record["data_seed"]
        relative_root = record["relative_root"]
        _expect(isinstance(history_id, str) and re.fullmatch(r"H-S22-PILOT-[A-Z0-9-]+", history_id) is not None, "BINDING_MISMATCH", "history ID is outside the Sprint 22 pilot namespace")
        _expect(type(seed) is int and seed >= 0, "BINDING_MISMATCH", "data seed must be a nonnegative integer")
        _expect(seed not in excluded, "FRESHNESS_COLLISION", f"data seed {seed} overlaps immutable or explicitly excluded provenance")
        _expect(isinstance(relative_root, str) and relative_root.startswith("histories/") and ".." not in Path(relative_root).parts, "BINDING_MISMATCH", "relative history root must stay below histories/")
        _expect(isinstance(record["config_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["config_sha256"]) is not None, "BINDING_MISMATCH", "full per-seed configuration SHA-256 is required")
        _expect(isinstance(record["config_hash"], str) and record["config_hash"], "BINDING_MISMATCH", "generator config hash is required")
        history_ids.append(history_id)
        seeds.append(seed)
        roots.append(relative_root)
        clean.append(dict(record))
    _expect(len(set(history_ids)) == len(history_ids) and len(set(seeds)) == len(seeds) and len(set(roots)) == len(roots), "BINDING_MISMATCH", "history IDs, data seeds, and roots must all be unique")
    return clean


def _expected_scorer() -> dict[str, Any]:
    return {
        "model_seeds": list(MODEL_SEEDS),
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "checkpoint_step": 300,
        "checkpoints": {str(seed): {"filename": filename, "sha256": digest} for seed, (filename, digest) in CHECKPOINTS.items()},
        "bank": {"rows": BANK_ROWS, "dimensions": BANK_DIM, "k": BANK_K, "sha256_by_model_seed": {str(seed): digest for seed, digest in BANK_SHA256.items()}, "refit": False},
        "config_sha256": CONFIG_SHA256,
        "metric_code_sha256": METRIC_SHA256,
        "conditioning_mapping_sha256": MAPPING_SHA256,
        "masking": "sha256(UTF-8(history_id::file_id)); unsigned big-endian first four bytes mod 2**31",
        "branches": ["S_pred", "S_pop"],
        "inference_only": True,
    }


def _expected_analysis() -> dict[str, Any]:
    return {
        "primary": "within-history tie-aware AUROC for P+W event windows versus same-history eligible control windows; max file score per window; S_pred and S_pop separate",
        "secondary": ["P AUROC", "W AUROC", "A AUROC", "support counts and reasons", "per-history score-independent support", "history variance and support-margin strata"],
        "support_selection_score_independent": True,
        "full_roster_required": HISTORY_COUNT,
        "history_macro": "unweighted mean across all 32 predeclared histories only",
        "bootstrap": {"resampling_unit": "whole history", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "lcb_percentile": 2.5, "same_draws_for_branches": True, "interpretation": "descriptive uncertainty summary only"},
        "hard_fail_or_missing_or_nonfinite_or_branch_support_mismatch": "per-history endpoint uncomputable; any such history makes full-roster macro/bootstrap uncomputable; never average a subset",
        "margin_misses": "retain the history if every hard floor passes; report as descriptive support strata only; no tolerance or miss allowance",
        "sample_variance": "unbiased sample variance across complete independent history endpoint values; scorer seeds remain separate repeated scorer variants",
        "thresholds_or_calibration": False,
        "training_or_refitting": False,
    }


def validate_binding(binding: dict[str, Any], expected_digest: str | None) -> str:
    declared = binding.get("binding_sha256")
    payload = dict(binding)
    payload.pop("binding_sha256", None)
    digest = canonical_sha256(payload)
    _expect(declared == digest, "BINDING_MISMATCH", "binding content digest differs from its declaration")
    if expected_digest is not None:
        _expect(digest == expected_digest, "BINDING_MISMATCH", "binding digest differs from the reviewed command-line digest")
    _expect(binding.get("schema_id") == BINDING_SCHEMA_ID and binding.get("protocol_id") == PROTOCOL_ID, "BINDING_MISMATCH", "binding schema or protocol identifier differs")
    _expect(binding.get("status") == "FROZEN_PENDING_PRECONTACT_REVIEW" and binding.get("contact_authorized") is False, "BINDING_MISMATCH", "binding must remain pending review and grant no contact authority")
    _expect(binding.get("history_count") == HISTORY_COUNT and binding.get("history_unit") == "independent whole history; scorer seeds are repeated scorer variants, not additional histories", "BINDING_MISMATCH", "history count or independent sampling unit differs")
    roster = validate_roster(binding.get("roster"), binding.get("freshness_exclusions"))
    _expect(binding.get("construction") == {
        "profile_id": "sprint15-v7",
        "generator_profile_protocol_id": "sprint15-benchmark-protocol-v7",
        "manifest_protocol_tag": PROTOCOL_ID,
        "protocol_sha256": GENERATOR_PROTOCOL_SHA256,
        "generator_version": "2.0.0",
        "config_factory": "synth.chronicle.sprint15_v7_history_config(data_seed)",
        "materializer": "synth.chronicle.materialize_chronological(cfg, root, shard_size=64, overwrite=False, role=history_id, protocol=manifest_protocol_tag, sprint15=None)",
        "allocation": "synth.balanced.allocate_quotas(rows, ledger, maintenance_windows, data_seed, QuotaConfig(), method='exact')",
        "quota": QUOTA,
        "quota_sha256": canonical_sha256(QUOTA),
        "config_template_sha256": CONFIG_TEMPLATE_SHA256,
        "new_construction_quota": False,
    }, "BINDING_MISMATCH", "generator, profile, allocation, or construction reserve changed")
    _expect(binding.get("scorer") == _expected_scorer(), "BINDING_MISMATCH", "scorer, checkpoint, Fit-bank, metric, or mask contract differs")
    _expect(binding.get("hard_floors") == HARD_FLOORS and binding.get("design_margins_diagnostic_only") == DESIGN_MARGINS, "BINDING_MISMATCH", "hard/support/mix gates or diagnostic margins differ")
    _expect(binding.get("analysis") == _expected_analysis(), "BINDING_MISMATCH", "whole-history endpoint and uncertainty contract differs")
    _expect(binding.get("runtime") == EXPECTED_RUNTIME, "BINDING_MISMATCH", "runtime binding differs from the exact reviewed environment")
    source = binding.get("source")
    _expect(isinstance(source, dict) and source.get("historical_base_commit") == BASE_COMMIT, "BINDING_MISMATCH", "source binding does not name the historical base")
    _expect(isinstance(source.get("runner_commit"), str) and re.fullmatch(r"[0-9a-f]{40}", source["runner_commit"]) is not None, "BINDING_MISMATCH", "full runner source commit is required")
    _expect(isinstance(source.get("runner_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", source["runner_sha256"]) is not None, "BINDING_MISMATCH", "full runner SHA-256 is required")
    _expect(isinstance(source.get("protocol_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", source["protocol_sha256"]) is not None, "BINDING_MISMATCH", "full Task 3 protocol SHA-256 is required")
    _expect(source.get("uv_lock_sha256") == UV_LOCK_SHA256 and source.get("metric_code_sha256") == METRIC_SHA256 and source.get("source_sha256") == PINNED_SOURCE_HASHES, "BINDING_MISMATCH", "historical source or lock digest differs")
    _expect(binding.get("output") == {
        "run_root": binding.get("output", {}).get("run_root"),
        "relative_history_roots": [item["relative_root"] for item in roster],
        "one_shot": True, "resume": False, "retry": False, "replacement": False,
        "omission": False, "seed_search": False, "post_failure_continue": False,
        "preflight": False, "confirmation_or_sealed_contact": False,
    } and isinstance(binding.get("output", {}).get("run_root"), str), "BINDING_MISMATCH", "one-shot output contract differs or is incomplete")
    return digest


def read_binding(path: Path, expected_digest: str | None) -> tuple[dict[str, Any], str]:
    binding = _read_json(path, "BINDING_MISMATCH")
    return binding, validate_binding(binding, expected_digest)


def verify_contract() -> str:
    contract_path = Path(__file__).resolve().parents[1] / "experiments/sprint22-disposable-boundary-proof-v1.md"
    try:
        actual = sha256_file(contract_path)
    except OSError as exc:
        raise PilotError("CONTRACT_MISMATCH", f"cannot read Task 1 reviewed disposable contract: {exc}") from exc
    _expect(actual == TASK1_DISPOSABLE_CONTRACT_SHA256, "CONTRACT_MISMATCH", "Task 1 reviewed disposable contract digest differs")
    return actual


def _git_value(root: Path, args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:
        raise PilotError("PROVENANCE_MISMATCH", f"Git provenance query failed: {args}: {exc}") from exc


def verify_runtime() -> dict[str, Any]:
    import torch

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True, text=True, check=True,
        )
        device_name, total_mib = (part.strip() for part in result.stdout.strip().split(",", 1))
        physical_total_mib = int(total_mib)
    except Exception as exc:
        raise PilotError("PROVENANCE_MISMATCH", f"cannot measure physical GPU identity/resources: {exc}") from exc
    cuda_available = bool(torch.cuda.is_available())
    observed = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "device_total_mib": physical_total_mib,
        "torch_visible_total_mib": int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)) if cuda_available else None,
    }
    _expect(not cuda_available or torch.cuda.get_device_name(0) == device_name, "PROVENANCE_MISMATCH", "torch and nvidia-smi device names differ")
    mismatches = {name: (observed.get(name), expected) for name, expected in EXPECTED_RUNTIME.items() if observed.get(name) != expected}
    _expect(not mismatches, "PROVENANCE_MISMATCH", f"runtime differs from frozen environment: {mismatches!r}")
    return observed



def _verify_pinned_source(root: Path) -> dict[str, Any]:
    _expect(_git_value(root, ["rev-parse", "HEAD"]) == BASE_COMMIT, "PROVENANCE_MISMATCH", "source worktree is not at the historical base")
    status = _git_value(root, ["status", "--porcelain", "--untracked-files=all"]).splitlines()
    allowed = "?? experiments/sprint22_disposable_boundary_proof.py"
    _expect(status in ([], [allowed]), "PROVENANCE_MISMATCH", f"historical source worktree has unauthorized changes: {status!r}")
    observed: dict[str, str] = {}
    for relative_path, expected in PINNED_SOURCE_HASHES.items():
        actual = sha256_file(root / relative_path)
        _expect(actual == expected, "PROVENANCE_MISMATCH", f"{relative_path} digest differs from the pinned historical source")
        observed[relative_path] = actual
    lock = sha256_file(root / "uv.lock")
    _expect(lock == UV_LOCK_SHA256, "PROVENANCE_MISMATCH", "uv.lock digest differs from the accepted scorer lock")
    metric_rows: list[list[str]] = []
    metric_members: dict[str, str] = {}
    for relative_path in METRIC_FILES:
        digest = sha256_file(root / relative_path)
        metric_rows.append([relative_path, digest])
        metric_members[relative_path] = digest
    metric_digest = canonical_sha256(metric_rows)
    _expect(metric_digest == METRIC_SHA256, "PROVENANCE_MISMATCH", "metric source closure digest differs")
    return {"base_commit": BASE_COMMIT, "git_status": status, "source_sha256": observed, "uv_lock_sha256": lock, "metric_code_sha256": metric_digest, "metric_member_sha256": metric_members}

def pin_historical_source(root: Path) -> None:
    source_root = (root / "src").resolve()
    source_path = str(source_root)
    if not sys.path or str(Path(sys.path[0]).resolve()) != source_path:
        sys.path.insert(0, source_path)
    assert_project_modules_pinned(root)


def assert_project_modules_pinned(root: Path) -> None:
    source_root = (root / "src").resolve()
    for name, module in sys.modules.items():
        if not (name == "synth" or name.startswith("synth.") or name == "representation" or name.startswith("representation.")):
            continue
        location = getattr(module, "__file__", None)
        if location is None:
            continue
        resolved = Path(location).resolve()
        _expect(resolved.is_relative_to(source_root), "PROVENANCE_MISMATCH", f"module {name} imported outside pinned historical source: {resolved}")




def _verify_checkpoints(checkpoint_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for seed in MODEL_SEEDS:
        filename, expected = CHECKPOINTS[seed]
        path = checkpoint_root / filename
        _expect(path.is_file(), "PROVENANCE_MISMATCH", f"checkpoint missing: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        _expect(size == CHECKPOINT_BYTES and digest == expected, "PROVENANCE_MISMATCH", f"checkpoint identity mismatch for seed {seed}")
        result[str(seed)] = {"path": str(path), "bytes": size, "sha256": digest}
    return result


def verify_release(
    binding: dict[str, Any], binding_digest: str, worktree_root: Path,
    checkpoint_root: Path, protocol_path: Path,
) -> dict[str, Any]:
    contract_sha = verify_contract()
    source = _verify_pinned_source(worktree_root)
    pin_historical_source(worktree_root)
    runtime = verify_runtime()
    _expect(checkpoint_root.resolve() == CHECKPOINT_ROOT.resolve(), "PROVENANCE_MISMATCH", "checkpoint root differs from the original read-only B0 location")
    actual_protocol_sha = sha256_file(protocol_path)
    _expect(actual_protocol_sha == binding["source"]["protocol_sha256"], "PROTOCOL_MISMATCH", "Task 3 protocol source hash differs from its binding")
    runner_root = Path(__file__).resolve().parents[1]
    runner_head = _git_value(runner_root, ["rev-parse", "HEAD"])
    runner_hash = sha256_file(Path(__file__).resolve())
    _expect(runner_head == binding["source"]["runner_commit"] and runner_hash == binding["source"]["runner_sha256"], "PROVENANCE_MISMATCH", "live runner commit or bytes differ from the reviewed binding")
    _expect(source["source_sha256"] == binding["source"]["source_sha256"], "PROVENANCE_MISMATCH", "computed pinned source hashes differ from the binding")
    _expect(source["uv_lock_sha256"] == binding["source"]["uv_lock_sha256"] and source["metric_code_sha256"] == binding["source"]["metric_code_sha256"], "PROVENANCE_MISMATCH", "computed lock or metric digest differs from the binding")
    checkpoints = _verify_checkpoints(checkpoint_root)
    return {
        "contract_sha256": contract_sha,
        "protocol_sha256": actual_protocol_sha,
        "runner_commit": runner_head,
        "runner_sha256": runner_hash,
        "binding_sha256": binding_digest,
        "source": source,
        "runtime": runtime,
        "checkpoints": checkpoints,
    }


def _load_b0_configs(root: Path) -> dict[str, dict[str, object]]:
    path = root / "experiments/sprint17_task7_b0.py"
    name = "_s22_runtime_task7_b0"
    _expect(name not in sys.modules, "PROVENANCE_MISMATCH", "pinned B0 module was already imported")
    spec = importlib.util.spec_from_file_location(name, path)
    _expect(spec is not None and spec.loader is not None, "PROVENANCE_MISMATCH", "cannot load historical B0 configuration source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        configs = {str(seed): module.b0_config_dict(seed) for seed in MODEL_SEEDS}
    finally:
        sys.modules.pop(name, None)
    _expect(canonical_sha256(configs) == CONFIG_SHA256, "PROVENANCE_MISMATCH", "full restored B0 config digest differs")
    return configs


def _conditioning_mapping(root: Path) -> dict[str, dict[str, int]]:
    src = str((root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    chronicle = importlib.import_module("synth.chronicle")
    _expect(Path(chronicle.__file__).resolve().is_relative_to(Path(src)), "PROVENANCE_MISMATCH", "chronicle imported outside the pinned historical source")
    cfg = chronicle.sprint15_v7_history_config(seed=0)
    stages = [stage for route in cfg.scheduler.routes for stage in route.stages]
    robots = sorted({stage.robot_id for stage in stages})
    programs = sorted({stage.program_id for stage in stages})
    _expect(len(robots) == cfg.fleet.n_robots and len(programs) == cfg.fleet.n_programs, "CONDITIONING_MISMATCH", "static profile cardinalities differ from source route vocabulary")
    mapping = {"robot_id_to_idx": {name: i for i, name in enumerate(robots)}, "program_id_to_idx": {name: i for i, name in enumerate(programs)}}
    _expect(canonical_sha256(mapping) == MAPPING_SHA256, "CONDITIONING_MISMATCH", "source-derived conditioning map digest differs")
    return mapping


def _history_config(coordinate: dict[str, Any], chronicle: Any) -> Any:
    cfg = chronicle.sprint15_v7_history_config(seed=coordinate["data_seed"])
    resolved = dataclasses.asdict(cfg)
    full_digest = hashlib.sha256(json.dumps(resolved, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    _expect(cfg.hash() == coordinate["config_hash"] and full_digest == coordinate["config_sha256"], "CONFIG_MISMATCH", f"data seed {coordinate['data_seed']} configuration differs from frozen binding")
    return cfg


def _read_serialized_indices(history_root: Path, manifest: dict[str, Any]) -> dict[str, tuple[int, int]]:
    import io
    import zipfile
    import numpy as np

    indices: dict[str, tuple[int, int]] = {}
    for shard in manifest.get("shards", []):
        with zipfile.ZipFile(history_root / str(shard["path"]), "r") as archive:
            for file_id in shard["file_ids"]:
                with np.load(io.BytesIO(archive.read(f"{file_id}.npz")), allow_pickle=False) as payload:
                    values: list[int] = []
                    for field_name in ("robot_idx", "program_idx"):
                        _expect(field_name in payload.files, "CONDITIONING_MISMATCH", f"{file_id}: serialized NPZ omits {field_name}")
                        raw = np.asarray(payload[field_name])
                        _expect(raw.shape == () and raw.dtype.kind in "iu", "CONDITIONING_MISMATCH", f"{file_id}: serialized {field_name} is not an integer scalar")
                        values.append(int(raw.item()))
                    _expect(file_id not in indices, "MANIFEST_MISMATCH", f"duplicate serialized member {file_id}")
                    indices[file_id] = (values[0], values[1])
    expected = [row["file_id"] for row in manifest["files"]]
    _expect(list(indices) == expected, "MANIFEST_MISMATCH", "serialized NPZ conditioning roster/order differs from manifest")
    return indices


def _materialize_one(
    output_root: Path, coordinate: dict[str, Any], history_evidence_dir: Path,
    worktree_root: Path,
) -> dict[str, Any]:
    chronicle = importlib.import_module("synth.chronicle")
    balanced = importlib.import_module("synth.balanced")
    events = importlib.import_module("synth.events")
    assert_project_modules_pinned(worktree_root)
    relative_root = Path(coordinate["relative_root"])
    root = output_root / relative_root
    _expect(root.resolve().is_relative_to(output_root.resolve()), "OUTPUT_PATH_MISMATCH", "history root escapes bound output root")
    _expect(not root.exists() and not root.is_symlink(), "OUTPUT_ALREADY_EXISTS", f"history output already exists: {root}")
    cfg = _history_config(coordinate, chronicle)
    manifest = chronicle.materialize_chronological(
        cfg, root, shard_size=64, overwrite=False,
        role=coordinate["history_id"], protocol=PROTOCOL_ID, sprint15=None,
    )
    samples, persisted = chronicle.load_chronological(root)
    _expect(manifest == persisted, "MANIFEST_MISMATCH", "materializer return differs from persisted manifest")
    _expect(persisted.get("format") == 1 and persisted.get("protocol") == PROTOCOL_ID and persisted.get("role") == coordinate["history_id"], "MANIFEST_MISMATCH", "materialized format, protocol, or whole-history role differs")
    _expect(persisted.get("generator_version") == "2.0.0" and persisted.get("config_hash") == coordinate["config_hash"], "CONFIG_MISMATCH", "materialized generator version or profile config hash differs")
    expected_seeds = {name: coordinate["data_seed"] for name in ("factory", "scheduler", "health", "signal", "temporal")}
    _expect(persisted.get("seeds") == expected_seeds, "CONFIG_MISMATCH", "manifest does not bind every generator sub-seed to its declared data seed")
    full_config_digest = hashlib.sha256(json.dumps(persisted["resolved_config"], sort_keys=True, default=str).encode()).hexdigest()
    _expect(full_config_digest == coordinate["config_sha256"], "CONFIG_MISMATCH", "materialized resolved configuration differs from its frozen full digest")
    rows = list(persisted["files"])
    sample_ids = [sample.file_id for sample in samples]
    row_ids = [row["file_id"] for row in rows]
    _expect(row_ids == sample_ids == list(dict.fromkeys(row_ids)) and len(rows) == int(persisted.get("counts", {}).get("total", -1)), "MANIFEST_MISMATCH", "manifest, sample, and count rosters differ or contain duplicate file IDs")
    indices = _read_serialized_indices(root, persisted)
    _expect(len(indices) == len(rows), "CONDITIONING_MISMATCH", "explicit NPZ conditioning roster differs from all manifest files")
    del samples, sample_ids
    ledger = events.failure_ledger(persisted)
    wins = persisted["maintenance_windows"]
    allocation: dict[str, Any] | None = None
    allocation_failure: dict[str, Any] | None = None
    try:
        allocation = balanced.allocate_quotas(rows, ledger, wins, coordinate["data_seed"], balanced.QuotaConfig(), method="exact")
        allocation = require_minimum_control_allocation(allocation)
    except PilotError as exc:
        if exc.code != "CONTROL_CONSTRUCTION_SHORTFALL":
            raise
        allocation_failure = {
            "type": type(exc).__name__, "message": str(exc),
            "record": {"reason": "control-shortfall", "missing": str(exc)}, "nonretryable": True,
        }
    except Exception as exc:
        expected_types = tuple(value for value in (getattr(balanced, "InfeasibleCandidate", None), getattr(balanced, "AllocationUnavailable", None)) if isinstance(value, type))
        if not expected_types or not isinstance(exc, expected_types):
            raise PilotError("ALLOCATION_RUNTIME_FAILURE", f"exact allocator raised unexpected {type(exc).__name__}: {exc}") from exc
        allocation_failure = {"type": type(exc).__name__, "message": str(exc), "record": getattr(exc, "record", {}), "nonretryable": True}
    report, positive_windows, control_windows = _support_and_structure(persisted, coordinate, allocation, allocation_failure, balanced, events)
    report["serialized_conditioning_entries"] = len(indices)
    report["manifest_sha256"] = sha256_file(root / "manifest.json")
    report["manifest_bytes"] = (root / "manifest.json").stat().st_size
    report["resolved_config_sha256"] = full_config_digest
    report["calendar_sha256"] = balanced.calendar_digest(persisted["schedule"])
    report["shards"] = persisted["shards"]
    evidence_path = history_evidence_dir / f"{coordinate['history_id']}.json"
    _write_json_new(evidence_path, report)
    return {
        "coordinate": coordinate, "root": root, "report": report,
        "positive_windows": positive_windows, "control_windows": control_windows,
    }


def _support_and_structure(
    manifest: dict[str, Any], coordinate: dict[str, Any], allocation: dict[str, Any] | None,
    allocation_failure: dict[str, Any] | None, balanced: Any, events: Any,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows = list(manifest["files"])
    ledger = events.failure_ledger(manifest)
    wins = manifest["maintenance_windows"]
    eligible, rejection, lead = balanced.eligible_anchors(rows, ledger, wins)
    selected_ids = set(allocation.get("selected_ids", [])) if allocation else set()
    positive_windows: dict[str, list[dict[str, Any]]] = {cohort: [] for cohort in ("P", "W", "A")}
    for failure in eligible:
        files = events.pos_files(rows, failure, wins)
        _expect(bool(files), "SUPPORT_MISMATCH", f"eligible event {failure['failure_id']} has no positive files")
        positive_windows[failure["cohort"]].append({
            "key": f"{coordinate['history_id']}::{failure['failure_id']}",
            "robot_id": failure["robot_id"], "failure_time": failure["failure_time"],
            "members": [f"{coordinate['history_id']}::{row['file_id']}" for row in files],
        })

    source_to_global = {row["file_id"]: f"{coordinate['history_id']}::{row['file_id']}" for row in rows}
    selector_controls = events.select_control_windows(events.anchor_rows(rows, wins), ledger, wins)
    if allocation:
        allocator_controls = [
            (window["robot_id"], float(window["anchor_end"]), tuple(window["members"]))
            for window in allocation.get("controls", [])
        ]
        canonical_controls = [
            (window["robot_id"], float(window["anchor_end"]), tuple(member["file_id"] for member in window["members"]))
            for window in selector_controls
        ]
        _expect(allocator_controls == canonical_controls, "SUPPORT_MISMATCH", "allocator did not retain every canonical eligible control window")
    control_windows = [{
        "key": f"{coordinate['history_id']}::CTRL::{window['robot_id']}::{window['anchor_end']:.6f}",
        "robot_id": window["robot_id"], "anchor_end": float(window["anchor_end"]),
        "members": [source_to_global[member["file_id"]] for member in window["members"]],
    } for window in selector_controls]
    support_keys = [window["key"] for cohort in ("P", "W", "A") for window in positive_windows[cohort]]
    control_keys = [window["key"] for window in control_windows]
    _expect(len(support_keys) == len(set(support_keys)) and len(control_keys) == len(set(control_keys)), "SUPPORT_MISMATCH", "positive or control support keys are duplicated")

    cohorts = {cohort: len(positive_windows[cohort]) for cohort in ("P", "W", "A")}
    positive_total = sum(cohorts.values())
    control_count = len(control_windows)
    eligible_rows = [row for row in rows if events.eligible_operational_row(row, wins)]
    robot_days = len({(row["robot_id"], int(row["end_time"] // 86400.0)) for row in eligible_rows})
    positive_by_robot = Counter(window["robot_id"] for cohort in ("P", "W", "A") for window in positive_windows[cohort])
    negative_by_robot = Counter(window["robot_id"] for window in control_windows)
    max_positive_share = max(positive_by_robot.values(), default=0) / positive_total if positive_total else 1.0
    max_negative_share = max(negative_by_robot.values(), default=0) / control_count if control_count else 1.0
    program_by_endpoint = {(row["robot_id"], row["end_time"]): row["program_id"] for row in rows}
    program_counts: dict[str, Counter[str]] = {"P": Counter(), "W": Counter()}
    for cohort in ("P", "W"):
        for window in positive_windows[cohort]:
            program = program_by_endpoint.get((window["robot_id"], window["failure_time"]))
            _expect(program is not None, "SUPPORT_MISMATCH", f"{cohort} event has no exact-endpoint program")
            program_counts[cohort][program] += 1
    program_shares = {
        cohort: max(counts.values(), default=0) / cohorts[cohort] if cohorts[cohort] else 1.0
        for cohort, counts in program_counts.items()
    }
    mix = {cohort: cohorts[cohort] / positive_total if positive_total else 0.0 for cohort in "PWA"}
    lead_support: dict[str, dict[str, Any]] = {}
    for cohort in ("P", "W"):
        details = []
        for window in positive_windows[cohort]:
            failure_id = window["key"].split("::", 1)[1]
            detail = lead.get(failure_id, {})
            details.append({
                "failure_id": failure_id, "endpoints": detail.get("endpoints", 0),
                "early_endpoint": bool(detail.get("early_endpoint")),
                "clean_baseline": bool(detail.get("clean_baseline")),
                "pass": bool(detail.get("endpoints", 0) >= 3 and detail.get("early_endpoint") and detail.get("clean_baseline")),
            })
        passed = sum(item["pass"] for item in details)
        lead_support[cohort] = {
            "passed": passed, "total": len(details),
            "fraction": passed / len(details) if details else None, "events": details,
        }

    allowed_rows = getattr(balanced, "ALLOWED_ROW_KEYS")
    by_robot: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_robot.setdefault(row["robot_id"], []).append(row)
    serialized = all(set(row) <= allowed_rows and "cohort" not in row and "subtype" not in row for row in rows)
    serial_calendar = all(
        all(left["end_time"] <= right["start_time"] + 1e-6 for left, right in zip(ordered, ordered[1:]))
        for ordered in ([*sorted(group, key=lambda item: (item["start_time"], item["end_time"]))] for group in by_robot.values())
    )
    maintenance_bounded = all(float(end) > float(start) for intervals in wins.values() for start, end in intervals)
    subtype_valid = all(
        failure["subtype"] in ({"P1", "P2"} if failure["cohort"] == "P" else {"W1", "W2"} if failure["cohort"] == "W" else {"A1", "A2"})
        for failure in ledger
    )
    p_durations = [float(failure["duration_d"]) for failure in ledger if failure["cohort"] == "P"]
    w_durations = [float(failure["duration_d"]) for failure in ledger if failure["cohort"] == "W"]
    p_shape = bool(p_durations) and min(p_durations) >= 2 and max(p_durations) <= 15 and 5 <= statistics.median(p_durations) <= 10
    w_shape = bool(w_durations) and min(w_durations) >= 6 and max(w_durations) <= 28 and 12 <= statistics.median(w_durations) <= 24
    expected_selected = {"P": {"P1": 12, "P2": 12}, "W": {"W1": 12, "W2": 12}, "A": {"A1": 8, "A2": 8}}
    actual_selected = {
        cohort: {subtype: len(ids) for subtype, ids in values.items()}
        for cohort, values in allocation.get("selected", {}).items()
    } if allocation else {}
    exact_positive = bool(allocation) and actual_selected == expected_selected and len(selected_ids) == 64 and len(selected_ids) == len(allocation.get("selected_ids", []))
    alloc_seed_match = bool(allocation) and allocation.get("alloc_seed") == balanced.alloc_seed_for(coordinate["data_seed"])
    construction_controls_pass = bool(allocation) and len(allocation.get("controls", [])) >= MINIMUM_CONTROLS and allocation.get("margins", {}).get("controls") == len(allocation.get("controls", []))
    structural = {
        "file_row_allowlist_without_hidden_event_labels": serialized,
        "nine_robot_vocabulary_and_reserve_robot": sorted(by_robot) == [f"robot-{i:02d}" for i in range(1, 10)] and "robot-08" in by_robot,
        "reserved_program_present": "program-03" in {row["program_id"] for row in rows},
        "robot_chronology_serialized": serial_calendar,
        "maintenance_windows_positive": maintenance_bounded,
        "all_P_W_A_cohorts": {failure["cohort"] for failure in ledger} == {"P", "W", "A"},
        "cohort_subtypes_valid": subtype_valid,
        "P_degradation_duration_shape": p_shape,
        "W_degradation_duration_shape": w_shape,
        "exact_positive_quota": exact_positive,
        "allocator_seed_matches_history": alloc_seed_match,
    }
    hard_flags = dict(structural)
    hard_flags.update({
        "P_ge_10": cohorts["P"] >= HARD_FLOORS["P"],
        "W_ge_10": cohorts["W"] >= HARD_FLOORS["W"],
        "A_ge_8": cohorts["A"] >= HARD_FLOORS["A"],
        "positive_total_ge_30": positive_total >= HARD_FLOORS["positive_total"],
        "controls_ge_25": control_count >= HARD_FLOORS["controls"],
        "robot_days_ge_150": robot_days >= HARD_FLOORS["robot_days"],
        "positive_robots_ge_6": len(positive_by_robot) >= HARD_FLOORS["positive_robots"],
        "negative_robots_ge_6": len(negative_by_robot) >= HARD_FLOORS["negative_robots"],
        "programs_ge_2_each_P_W": all(len(program_counts[cohort]) >= HARD_FLOORS["programs_each_P_W"] for cohort in ("P", "W")),
        "cohort_mix_15_to_60": all(HARD_FLOORS["cohort_mix_min"] <= mix[cohort] <= HARD_FLOORS["cohort_mix_max"] for cohort in "PWA"),
        "positive_robot_share_le_35": max_positive_share <= HARD_FLOORS["positive_robot_share_max"],
        "negative_robot_share_le_40": max_negative_share <= HARD_FLOORS["negative_robot_share_max"],
        "program_share_le_60_each_P_W": all(value <= HARD_FLOORS["program_share_max"] for value in program_shares.values()),
        "lead_P_ge_80": lead_support["P"]["fraction"] is not None and lead_support["P"]["fraction"] >= HARD_FLOORS["lead_support_P_W_min"],
        "lead_W_ge_80": lead_support["W"]["fraction"] is not None and lead_support["W"]["fraction"] >= HARD_FLOORS["lead_support_P_W_min"],
    })
    margins = {
        "P_ge_13": cohorts["P"] >= DESIGN_MARGINS["P"], "W_ge_13": cohorts["W"] >= DESIGN_MARGINS["W"],
        "A_ge_10": cohorts["A"] >= DESIGN_MARGINS["A"], "positive_total_ge_38": positive_total >= DESIGN_MARGINS["positive_total"],
        "controls_ge_32": control_count >= DESIGN_MARGINS["controls"], "robot_days_ge_188": robot_days >= DESIGN_MARGINS["robot_days"],
    }
    allocator_failure_reason = allocation_failure["record"].get("reason", "unclassified") if allocation_failure else None
    hard_status = all(hard_flags.values()) and construction_controls_pass and allocation_failure is None
    hard_failures = [name for name, passed in hard_flags.items() if not passed]
    if not construction_controls_pass:
        hard_failures.append("allocator_result_minimum_48_controls_failed")
    if allocation_failure:
        hard_failures.append(f"allocator_failure:{allocator_failure_reason}")
    report = {
        "history_id": coordinate["history_id"], "data_seed": coordinate["data_seed"],
        "file_count": len(rows), "raw_failure_count": len(ledger),
        "eligible_positive_windows": cohorts, "eligible_positive_total": positive_total,
        "eligible_control_windows": control_count,
        "allocator_returned_controls": len(allocation.get("controls", [])) if allocation else 0,
        "eligible_robot_days": robot_days, "positive_robots": len(positive_by_robot), "negative_robots": len(negative_by_robot),
        "mix": mix, "program_counts_P_W": {cohort: dict(counts) for cohort, counts in program_counts.items()},
        "program_share_max_P_W": program_shares, "positive_robot_max_share": max_positive_share,
        "negative_robot_max_share": max_negative_share, "lead_support": lead_support,
        "hard_flags": hard_flags, "hard_status": "PASS" if hard_status else "FAIL",
        "hard_failures": hard_failures,
        "design_margin_flags": margins, "design_margin_misses_are_diagnostic_only": True,
        "allocation": allocation, "allocation_failure": allocation_failure,
        "eligible_anchor_rejections": dict(rejection),
        "score_independent_support_keys": {
            cohort: [window["key"] for window in positive_windows[cohort]] for cohort in ("P", "W", "A")
        } | {"controls": control_keys},
        "construction": {
            "minimum_control_reserve": MINIMUM_CONTROLS,
            "allocator_result_minimum_48_controls_passes": construction_controls_pass,
            "eligible_control_pool_at_least_48": control_count >= MINIMUM_CONTROLS,
        },
        "hard_control_support_floor": HARD_FLOORS["controls"],
        "design_control_margin_diagnostic": DESIGN_MARGINS["controls"],
    }
    return report, positive_windows, control_windows


def _write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=1, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.write("\n")


def _write_json_replace(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=1, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_ledger(handle: Any, event: dict[str, Any]) -> None:
    handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def validate_runtime_roster(actual_ids: list[str], expected_ids: list[str]) -> None:
    _expect(len(expected_ids) == len(set(expected_ids)), "ROSTER_OR_SCORE_INCOMPLETE", "expected score roster contains duplicates")
    _expect(len(actual_ids) == len(expected_ids) and len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == set(expected_ids), "ROSTER_OR_SCORE_INCOMPLETE", "materialized sample roster differs from its declared full roster")


def _prepare_scoring_input(
    sample: Any, row: dict[str, Any], mapping: dict[str, dict[str, int]],
    serialized_indices: tuple[int, int], history_id: str, manifest: dict[str, Any],
    patchifier: Any, cfg: Any, masking_seed: int,
) -> tuple[dict[str, Any], str, list[int]]:
    import numpy as np
    from representation.data import collate_variable_files
    from synth.schema import FileSample, SampleLabel

    file_id = sample.file_id
    robot_id, program_id = row.get("robot_id"), row.get("program_id")
    _expect(isinstance(robot_id, str) and robot_id in mapping["robot_id_to_idx"], "CONDITIONING_MISMATCH", f"{file_id}: unknown robot string")
    _expect(isinstance(program_id, str) and program_id in mapping["program_id_to_idx"], "CONDITIONING_MISMATCH", f"{file_id}: unknown program string")
    robot_idx, program_idx = mapping["robot_id_to_idx"][robot_id], mapping["program_id_to_idx"][program_id]
    _expect(type(serialized_indices) is tuple and len(serialized_indices) == 2 and all(type(value) is int and value >= 0 for value in serialized_indices), "CONDITIONING_MISMATCH", f"{file_id}: missing explicit serialized conditioning indices")
    _expect(type(getattr(sample, "robot_idx", None)) is int and type(getattr(sample, "program_idx", None)) is int and (sample.robot_idx, sample.program_idx) == serialized_indices == (robot_idx, program_idx), "CONDITIONING_MISMATCH", f"{file_id}: manifest strings, loaded sample and serialized conditioning disagree")
    x = sample.x
    _expect(isinstance(x, np.ndarray) and x.dtype == np.float32 and x.ndim == 2 and x.shape[0] == 6 and x.shape[1] > 0 and bool(np.isfinite(x).all()), "INPUT_CONTRACT_FAILURE", f"{file_id}: expected finite float32 [6,T] model signal")
    global_id = f"{history_id}::{file_id}"
    expected_seed = int.from_bytes(hashlib.sha256(global_id.encode("utf-8")).digest()[:4], "big") % (2**31)
    _expect(masking_seed == expected_seed, "MASK_CONTRACT_FAILURE", f"{global_id}: mask seed differs from its file identity")
    neutral = FileSample(
        x=x, file_id=global_id, file_label=SampleLabel.NORMAL, seed=0,
        generator_version=manifest["generator_version"], config_hash=manifest["config_hash"],
        regime_sequence=[], robot_idx=robot_idx, program_idx=program_idx,
    )
    batch = collate_variable_files([neutral], patchifier, masking_config=cfg, masking_seed=masking_seed)
    valid = batch["patch_valid_mask"][0].detach().cpu().numpy().astype(bool)
    mask = batch["mask"][0].detach().cpu().numpy().astype(bool)
    _expect(bool(valid.any()) and int(mask.sum()) > 0 and not bool((mask & ~valid).any()), "MASK_CONTRACT_FAILURE", f"{global_id}: invalid or empty prediction mask")
    model_batch = {key: value for key, value in batch.items() if key not in {"file_labels", "anomaly_meta", "anomaly_masks", "file_samples"}}
    return model_batch, global_id, [int(value) for value in mask.tolist()]


def _score_history(
    history: dict[str, Any], model: Any, bank: Any, patchifier: Any, cfg: Any,
    mapping: dict[str, dict[str, int]], device: str, score_path: Path,
) -> dict[str, Any]:
    import torch
    from representation.inference import RepresentationInference

    manifest = history["manifest"]
    samples = history["samples"]
    coordinate = history["coordinate"]
    history_id = coordinate["history_id"]
    rows = {row["file_id"]: row for row in manifest["files"]}
    expected_ids = [f"{history_id}::{row['file_id']}" for row in manifest["files"]]
    actual_ids = [f"{history_id}::{sample.file_id}" for sample in samples]
    validate_runtime_roster(actual_ids, expected_ids)
    inference = RepresentationInference(model, bank, patchifier, masking_config=cfg)
    scores: dict[str, dict[str, float]] = {}
    support_keys = [window["key"] for cohort in ("P", "W", "A") for window in history["positive_windows"][cohort]] + [window["key"] for window in history["control_windows"]]
    _expect(len(support_keys) == len(set(support_keys)), "ROSTER_OR_SCORE_INCOMPLETE", "score-independent support contains duplicate keys")
    temporary_path = score_path.with_suffix(score_path.suffix + ".partial")
    _expect(not score_path.exists() and not temporary_path.exists(), "OUTPUT_ALREADY_EXISTS", f"score output or temporary file already exists: {score_path}")
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as output:
            for position, sample in enumerate(samples):
                source_id = sample.file_id
                row = rows[source_id]
                indices = history["serialized_indices"].get(source_id)
                _expect(indices is not None, "CONDITIONING_MISMATCH", f"{source_id}: no exact NPZ conditioning record")
                global_id = f"{history_id}::{source_id}"
                _expect(position < len(expected_ids) and expected_ids[position] == global_id, "ROSTER_OR_SCORE_INCOMPLETE", "scoring input order differs from the declared full roster")
                seed = int.from_bytes(hashlib.sha256(global_id.encode("utf-8")).digest()[:4], "big") % (2**31)
                model_batch, checked_id, mask_bits = _prepare_scoring_input(sample, row, mapping, indices, history_id, manifest, patchifier, cfg, seed)
                _expect(checked_id == global_id, "ROSTER_OR_SCORE_INCOMPLETE", "scoring helper returned an unexpected global ID")
                moved = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in model_batch.items()}
                with torch.inference_mode():
                    result = inference.score_batch(moved)
                pred = float(torch.as_tensor(result["S_pred"]).reshape(-1)[0].item())
                pop = float(torch.as_tensor(result["S_pop"]).reshape(-1)[0].item())
                patch_scores = torch.as_tensor(result["patch_scores"])
                prediction_mask = torch.as_tensor(result["prediction_mask"])
                _expect(tuple(patch_scores.shape) == tuple(prediction_mask.shape), "SCORE_MISMATCH", f"{global_id}: patch score/mask shapes differ")
                expected_pred = float((patch_scores.sum(dim=1) / prediction_mask.sum(dim=1).clamp_min(1).to(patch_scores.dtype)).reshape(-1)[0].item())
                embedding = torch.as_tensor(result["file_embedding"])
                distances = torch.cdist(embedding.float().cpu(), bank.embeddings.float().cpu())
                expected_pop = float(distances.topk(BANK_K, largest=False, dim=1).values.mean(dim=1).reshape(-1)[0].item())
                for branch, got, expected in (("S_pred", pred, expected_pred), ("S_pop", pop, expected_pop)):
                    _expect(math.isfinite(got) and math.isfinite(expected) and got >= 0 and expected >= 0, "ROSTER_OR_SCORE_INCOMPLETE", f"{global_id}/{branch}: invalid score")
                    _expect(math.isclose(got, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL), "SCORE_MISMATCH", f"{global_id}/{branch}: production score differs from independent recomputation")
                item = {
                    "history_id": history_id, "data_seed": coordinate["data_seed"],
                    "source_file_id": source_id, "global_file_id": global_id,
                    "robot_id": row["robot_id"], "robot_idx": int(indices[0]),
                    "program_id": row["program_id"], "program_idx": int(indices[1]),
                    "mask_seed": seed, "mask_bits": mask_bits, "S_pred": pred, "S_pop": pop,
                }
                output.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
                scores[global_id] = {"S_pred": pred, "S_pop": pop}
            output.flush()
            os.fsync(output.fileno())
        _expect(list(scores) == expected_ids, "ROSTER_OR_SCORE_INCOMPLETE", "score output roster differs from manifest file order")
        os.replace(temporary_path, score_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    metrics: dict[str, Any] | None = None
    metric_status = "UNCOMPUTABLE_HARD_SUPPORT_FAIL"
    metric_failure: dict[str, str] | None = None
    if history["report"]["hard_status"] == "PASS":
        metric_module = importlib.import_module("synth.events")
        try:
            metrics = _history_metrics(scores, history["positive_windows"], history["control_windows"], metric_module)
        except metric_module.UnavailableError as exc:
            metric_status = "UNCOMPUTABLE_METRIC"
            metric_failure = {"type": type(exc).__name__, "message": str(exc)}
        else:
            metric_status = "COMPUTABLE"
    return {
        "history_id": history_id,
        "hard_status": history["report"]["hard_status"],
        "hard_failures": history["report"]["hard_failures"],
        "metric_status": metric_status, "metric_failure": metric_failure,
        "primary_auc": {branch: metrics[branch]["auc"]["P+W"] if metrics else None for branch in ("S_pred", "S_pop")},
        "secondary_auc": {branch: {cohort: metrics[branch]["auc"][cohort] if metrics else None for cohort in ("P", "W", "A")} for branch in ("S_pred", "S_pop")},
        "design_margin_flags": history["report"]["design_margin_flags"],
        "support_keys": metrics["support_keys"] if metrics else {"S_pred": support_keys, "S_pop": support_keys},
        "score_file_rows": len(scores),
        "score_file_sha256": sha256_file(score_path),
        "score_file_bytes": score_path.stat().st_size,
    }


def _history_metrics(
    scores: dict[str, dict[str, float]], positive_windows: dict[str, list[dict[str, Any]]],
    control_windows: list[dict[str, Any]], events: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    support_by_branch: dict[str, list[str]] = {}
    for branch in ("S_pred", "S_pop"):
        branch_scores = {file_id: values[branch] for file_id, values in scores.items()}
        pos_values = {
            cohort: [float(events.window_score(window["members"], branch_scores)) for window in positive_windows[cohort]]
            for cohort in ("P", "W", "A")
        }
        control_values = [float(events.window_score(window["members"], branch_scores)) for window in control_windows]
        auc: dict[str, float] = {}
        for cohort in ("P+W", "P", "W", "A"):
            positives = pos_values["P"] + pos_values["W"] if cohort == "P+W" else pos_values[cohort]
            auc[cohort] = float(events.roc_auc_tie_aware(positives, control_values))
        _expect(all(math.isfinite(value) and 0 <= value <= 1 for value in auc.values()), "METRIC_UNCOMPUTABLE", "tie-aware AUROC is outside [0,1] or non-finite")
        keys = [window["key"] for cohort in ("P", "W", "A") for window in positive_windows[cohort]] + [window["key"] for window in control_windows]
        support_by_branch[branch] = sorted(keys)
        output[branch] = {"auc": auc, "positive_window_scores": pos_values, "control_window_scores": control_values}
    _expect(support_by_branch["S_pred"] == support_by_branch["S_pop"], "ROSTER_OR_SCORE_INCOMPLETE", "score branch support differs")
    output["support_keys"] = support_by_branch
    return output


def _restore_seed(root: Path, checkpoint_root: Path, seed: int, config_map: dict[str, dict[str, object]], expected_bank: str) -> tuple[Any, Any, Any, Any, str]:
    import torch
    from representation.checkpoint import load_checkpoint
    from representation.config import V1Config
    from representation.inference import NormalReferenceBank
    from representation.model import V1RepresentationModel
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    cfg_dict = config_map[str(seed)]
    cfg = V1Config(
        n_channels=cfg_dict["n_channels"], patch_size=cfg_dict["patch_size"],
        stride=cfg_dict["stride"], d_model=cfg_dict["d_model"],
        sequence_layers=cfg_dict["sequence_layers"], attention_heads=cfg_dict["attention_heads"],
        dropout=cfg_dict["dropout"], n_robots=cfg_dict["n_robots"],
        n_programs=cfg_dict["n_programs"], use_conditional_norm=cfg_dict["use_conditional_norm"],
        min_bucket_samples=cfg_dict["min_bucket_samples"], total_mask_ratio=cfg_dict["total_mask_ratio"],
        random_fraction=cfg_dict["random_fraction"], info_fraction=cfg_dict["info_fraction"],
        block_fraction=cfg_dict["block_fraction"], ema_decay=cfg_dict["ema_decay"],
        prediction_weight=cfg_dict["prediction_weight"], contrastive_weight_max=cfg_dict["contrastive_weight_max"],
        contrastive_warmup_steps=cfg_dict["contrastive_warmup_steps"], contrastive_ramp_steps=cfg_dict["contrastive_ramp_steps"],
        contrastive_temperature=cfg_dict["contrastive_temperature"], contrastive_gain_std=cfg_dict["contrastive_gain_std"],
        contrastive_offset_std=cfg_dict["contrastive_offset_std"], contrastive_noise_std=cfg_dict["contrastive_noise_std"],
        contrastive_max_shift=cfg_dict["contrastive_max_shift"], knn_k=cfg_dict["knn_k"], seed=seed,
    )
    patchifier = Patchifier(PatchConfig(patch_size=cfg.patch_size, stride=cfg.stride, pad_end=cfg_dict["pad_end"]))
    model = V1RepresentationModel(cfg, patchifier=patchifier)
    bank = NormalReferenceBank(k=BANK_K)
    checkpoint = checkpoint_root / CHECKPOINTS[seed][0]
    metadata = load_checkpoint(checkpoint, model, reference_bank=bank)
    _expect(metadata.get("step") == 300 and metadata.get("has_reference_bank"), "PROVENANCE_MISMATCH", f"seed {seed} checkpoint step or embedded bank differs")
    restored = metadata.get("config")
    _expect(isinstance(restored, dict) and canonical_sha256(restored) == canonical_sha256(cfg.to_dict()) and restored.get("seed") == seed, "PROVENANCE_MISMATCH", f"seed {seed} checkpoint config differs")
    _expect(bank.embeddings is not None and bank.k == BANK_K and tuple(bank.embeddings.shape) == (BANK_ROWS, BANK_DIM) and bool(torch.isfinite(bank.embeddings).all()), "PROVENANCE_MISMATCH", f"seed {seed} Fit bank dimensions/values differ")
    bank_digest = hashlib.sha256(bank.embeddings.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    _expect(bank_digest == expected_bank == BANK_SHA256[seed], "PROVENANCE_MISMATCH", f"seed {seed} Fit-bank digest differs")
    model.to("cuda")
    model.eval()
    return model, bank, patchifier, cfg, bank_digest


def _snapshot_model(model: Any, bank: Any) -> dict[str, str]:
    import torch

    values: dict[str, str] = {}
    for name, param in sorted(model.named_parameters()):
        values[f"param:{name}"] = hashlib.sha256(param.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    for name, buf in sorted(model.named_buffers()):
        payload = buf.detach().cpu().contiguous().numpy().tobytes() if isinstance(buf, torch.Tensor) else repr(buf).encode("utf-8")
        values[f"buffer:{name}"] = hashlib.sha256(payload).hexdigest()
    values["bank:k"] = str(int(bank.k))
    values["bank:embedding_sha256"] = hashlib.sha256(bank.embeddings.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return values


def _score_all(
    histories: list[dict[str, Any]], verified: dict[str, Any],
    output_root: Path, worktree_root: Path, checkpoint_root: Path,
) -> dict[str, Any]:
    import torch

    config_map = _load_b0_configs(worktree_root)
    mapping = _conditioning_mapping(worktree_root)
    assert_project_modules_pinned(worktree_root)
    _expect(canonical_sha256(mapping) == MAPPING_SHA256, "CONDITIONING_MISMATCH", "condition map changed before scoring")
    history_ids = [entry["coordinate"]["history_id"] for entry in histories]
    endpoint_records: dict[int, list[dict[str, Any]]] = {seed: [] for seed in MODEL_SEEDS}
    per_seed: dict[str, Any] = {}
    for seed in MODEL_SEEDS:
        checkpoint = checkpoint_root / CHECKPOINTS[seed][0]
        _expect(sha256_file(checkpoint) == CHECKPOINTS[seed][1] and checkpoint.stat().st_size == CHECKPOINT_BYTES, "PROVENANCE_MISMATCH", f"seed {seed} checkpoint changed before restore")
        model, bank, patchifier, cfg, bank_digest = _restore_seed(worktree_root, checkpoint_root, seed, config_map, BANK_SHA256[seed])
        assert_project_modules_pinned(worktree_root)
        before = _snapshot_model(model, bank)
        seed_results = []
        for history in histories:
            history_id = history["coordinate"]["history_id"]
            score_path = output_root / "scores" / f"model-{seed}" / f"{history_id}.jsonl"
            score_path.parent.mkdir(parents=True, exist_ok=True)
            samples, manifest = importlib.import_module("synth.chronicle").load_chronological(history["root"])
            actual_manifest_sha = sha256_file(history["root"] / "manifest.json")
            _expect(actual_manifest_sha == history["report"]["manifest_sha256"], "MANIFEST_MISMATCH", f"{history_id}: manifest changed between construction and scoring")
            _expect(manifest.get("role") == history_id and manifest.get("seeds") == {name: history["coordinate"]["data_seed"] for name in ("factory", "scheduler", "health", "signal", "temporal")}, "MANIFEST_MISMATCH", f"{history_id}: loaded manifest role or generator seeds changed")
            serialized_indices = _read_serialized_indices(history["root"], manifest)
            scoring_history = {
                **history, "manifest": manifest, "samples": samples,
                "serialized_indices": serialized_indices,
            }
            endpoint = _score_history(scoring_history, model, bank, patchifier, cfg, mapping, "cuda", score_path)
            del scoring_history, serialized_indices, samples, manifest
            seed_results.append(endpoint)
            endpoint_records[seed].append(endpoint)
            _write_json_new(output_root / "history-evidence" / f"{history_id}-model-{seed}.json", endpoint)
        after = _snapshot_model(model, bank)
        _expect(before == after, "RESTORE_ONLY_OR_ISOLATION_FAILURE", f"seed {seed} model/bank state changed during scoring")
        _expect(sha256_file(checkpoint) == CHECKPOINTS[seed][1], "PROVENANCE_MISMATCH", f"seed {seed} checkpoint changed during scoring")
        per_seed[str(seed)] = {
            "checkpoint_sha256_before": verified["checkpoints"][str(seed)]["sha256"],
            "checkpoint_sha256_after": sha256_file(checkpoint),
            "bank_sha256": bank_digest,
            "bank_sha256_after": after["bank:embedding_sha256"],
            "bank_k": bank.k,
            "bank_shape": list(bank.embeddings.shape),
            "state_before": before, "state_after": after,
            "state_before_sha256": canonical_sha256(before), "state_after_sha256": canonical_sha256(after),
            "history_endpoints": seed_results,
        }
        del model, bank, patchifier, cfg
        torch.cuda.empty_cache()
    final = {str(seed): _aggregate_full_roster(history_ids, endpoint_records[seed]) for seed in MODEL_SEEDS}
    return final


def _aggregate_full_roster(history_ids: list[str], records: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    _expect(len(records) == len(history_ids) and [record.get("history_id") for record in records] == history_ids, "ROSTER_OR_SCORE_INCOMPLETE", "history endpoint roster differs from full declared roster")
    computable = all(
        record.get("hard_status") == "PASS"
        and record.get("metric_status") == "COMPUTABLE"
        and isinstance(record.get("support_keys"), dict)
        and set(record["support_keys"]) == {"S_pred", "S_pop"}
        and record["support_keys"]["S_pred"] == record["support_keys"]["S_pop"]
        and all(
            isinstance(record.get("primary_auc", {}).get(branch), (int, float))
            and not isinstance(record["primary_auc"][branch], bool)
            and math.isfinite(float(record["primary_auc"][branch]))
            and all(
                isinstance(record.get("secondary_auc", {}).get(branch, {}).get(cohort), (int, float))
                and not isinstance(record["secondary_auc"][branch][cohort], bool)
                and math.isfinite(float(record["secondary_auc"][branch][cohort]))
                for cohort in ("P", "W", "A")
            )
            for branch in ("S_pred", "S_pop")
        )
        for record in records
    )
    result: dict[str, Any] = {
        "full_roster_status": "COMPUTABLE_FULL_ROSTER" if computable else "UNCOMPUTABLE_FULL_ROSTER",
        "declared_history_ids": history_ids,
        "hard_failure_history_ids": [record["history_id"] for record in records if record.get("hard_status") != "PASS"],
        "metric_uncomputable_history_ids": [record["history_id"] for record in records if record.get("hard_status") == "PASS" and record.get("metric_status") != "COMPUTABLE"],
        "primary": {}, "secondary_cohort_macros": {},
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "resampling_unit": "whole history", "percentile": 2.5, "interpretation": "descriptive uncertainty only"},
    }
    if not computable:
        result["primary"] = {branch: {"macro": None, "sample_variance": None, "bootstrap_lcb95": None, "per_history": [{"history_id": record["history_id"], "value": None, "status": "UNCOMPUTABLE_FULL_ROSTER"} for record in records]} for branch in ("S_pred", "S_pop")}
        result["secondary_cohort_macros"] = {cohort: {branch: None for branch in ("S_pred", "S_pop")} for cohort in ("P", "W", "A")}
        result["bootstrap"]["computed"] = False
        return result
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(records), size=(BOOTSTRAP_REPLICATES, len(records)))
    for branch in ("S_pred", "S_pop"):
        values = np.asarray([record["primary_auc"][branch] for record in records], dtype=np.float64)
        macros = values[draws].mean(axis=1)
        result["primary"][branch] = {
            "macro": float(values.mean()), "sample_variance": float(values.var(ddof=1)),
            "bootstrap_lcb95": float(np.quantile(macros, 0.025)),
            "per_history": [{"history_id": record["history_id"], "value": float(record["primary_auc"][branch]), "status": "PASS"} for record in records],
        }
    for cohort in ("P", "W", "A"):
        result["secondary_cohort_macros"][cohort] = {
            branch: float(np.mean([record["secondary_auc"][branch][cohort] for record in records]))
            for branch in ("S_pred", "S_pop")
        }
    result["bootstrap"]["computed"] = True
    return result


def _null_full_roster(history_ids: list[str], attempted: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    records = {record.get("history_id"): record for record in attempted}
    return {
        "full_roster_status": "UNCOMPUTABLE_FULL_ROSTER",
        "declared_history_ids": history_ids,
        "attempted_history_ids": [record.get("history_id") for record in attempted],
        "reason": reason,
        "primary": {branch: {"macro": None, "sample_variance": None, "bootstrap_lcb95": None, "per_history": [{"history_id": history_id, "value": None, "status": "UNCOMPUTABLE_FULL_ROSTER", "hard_status": records.get(history_id, {}).get("hard_status", "NOT_ATTEMPTED")} for history_id in history_ids]} for branch in ("S_pred", "S_pop")},
        "secondary_cohort_macros": {cohort: {branch: None for branch in ("S_pred", "S_pop")} for cohort in ("P", "W", "A")},
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "computed": False, "resampling_unit": "whole history"},
    }


def _run_one_shot(
    binding: dict[str, Any], binding_digest: str, verified: dict[str, Any],
    output_root: Path, worktree_root: Path, checkpoint_root: Path,
) -> dict[str, Any]:
    history_ids = [coordinate["history_id"] for coordinate in binding["roster"]]
    history_evidence = output_root / "history-evidence"
    history_evidence.mkdir(parents=False, exist_ok=False)
    attempt = {
        "schema_id": "sprint22-pilot-attempt-v1", "protocol_id": PROTOCOL_ID,
        "binding_sha256": binding_digest, "status": "RUNNING",
        "run_started_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "bound_coordinates": binding["roster"], "attempted_history_ids": [],
        "unattempted_history_ids": history_ids, "resume": False, "retry": False,
        "replacement": False, "omission": False,
        "main_release_binding_sha256": binding_digest,
    }
    _write_json_new(output_root / "attempt.json", attempt)
    histories: list[dict[str, Any]] = []
    attempted_ids: list[str] = []
    with (output_root / "pilot-ledger.jsonl").open("x", encoding="utf-8", newline="\n") as ledger:
        _append_ledger(ledger, {"event": "run_started", "binding_sha256": binding_digest, "roster_size": HISTORY_COUNT})
        try:
            for index, coordinate in enumerate(binding["roster"]):
                history_id = coordinate["history_id"]
                attempted_ids.append(history_id)
                attempt.update({
                    "attempted_history_ids": list(attempted_ids),
                    "unattempted_history_ids": history_ids[index + 1:],
                })
                _write_json_replace(output_root / "attempt.json", attempt)
                _append_ledger(ledger, {"event": "materialization_started", "history_id": history_id, "data_seed": coordinate["data_seed"]})
                history = _materialize_one(output_root, coordinate, history_evidence, worktree_root)
                histories.append(history)
                _append_ledger(ledger, {"event": "materialization_complete", "history_id": history_id, "hard_status": history["report"]["hard_status"], "hard_failures": history["report"]["hard_failures"], "manifest_sha256": history["report"]["manifest_sha256"]})
                if history["report"]["hard_status"] != "PASS":
                    attempt.update({"status": "STOPPED_SCIENTIFIC_SUPPORT_FAIL", "attempted_history_ids": list(attempted_ids), "unattempted_history_ids": history_ids[index + 1:]})
                    _write_json_replace(output_root / "attempt.json", attempt)
                    support_records = [entry["report"] for entry in histories]
                    attempted_records = [
                        {"history_id": entry["coordinate"]["history_id"], "hard_status": entry["report"]["hard_status"]}
                        for entry in histories
                    ]
                    summary = {
                        "schema_id": "sprint22-pilot-summary-v1", "protocol_id": PROTOCOL_ID,
                        "binding_sha256": binding_digest, "status": "STOPPED_SCIENTIFIC_SUPPORT_FAIL",
                        "history_count": HISTORY_COUNT, "attempted_history_ids": list(attempted_ids),
                        "unattempted_history_ids": attempt["unattempted_history_ids"],
                        "support_records": support_records,
                        "model_seed_results_separate": {str(seed): _null_full_roster(history_ids, attempted_records, "first support failure prevented scoring") for seed in MODEL_SEEDS},
                        "claim_limits": ["no model restore or score after first hard/support failure", "no retry, resume, replacement, or subset aggregate", "H=32 is a fixed sample size, not a miss allowance", "Sprint 20 NOT_READY and Sprint 18 quota/no-Cycle-4 boundaries are unchanged"],
                    }
                    _write_json_new(output_root / "summary.json", summary)
                    _append_ledger(ledger, {"event": "scientific_support_stop", "history_id": history_id, "unattempted_history_ids": attempt["unattempted_history_ids"], "restore_or_scoring_started": False})
                    return summary
            _expect(len(histories) == HISTORY_COUNT, "ROSTER_OR_SCORE_INCOMPLETE", "materialized full roster count differs")
            attempt["attempted_history_ids"] = history_ids
            attempt["unattempted_history_ids"] = []
            final = _score_all(histories, verified, output_root, worktree_root, checkpoint_root)
            summary = {
                "schema_id": "sprint22-pilot-summary-v1", "protocol_id": PROTOCOL_ID,
                "binding_sha256": binding_digest, "status": "COMPLETE_DESCRIPTIVE_RESEARCH_ONLY",
                "history_count": HISTORY_COUNT, "history_unit": "independent whole history; scorer seeds are repeated scorer variants, not additional histories",
                "model_seed_results_separate": final,
                "support_records": [entry["report"] for entry in histories],
                "source_and_runtime_verification": verified,
                "claim_limits": ["simulator-conditional descriptive pilot only", "no calibration, thresholds, training, refit, bank fitting, or score fusion", "S_pred and S_pop stay separate", "H=32 is sample size only and grants no miss allowance or tolerance", "Sprint 20 NOT_READY and Sprint 18 quota/no-Cycle-4 boundaries are unchanged"],
            }
            _write_json_new(output_root / "summary.json", summary)
            attempt["status"] = "COMPLETE"
            attempt["run_completed_utc"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
            _write_json_replace(output_root / "attempt.json", attempt)
            _append_ledger(ledger, {"event": "run_complete", "history_count": HISTORY_COUNT, "model_seeds": list(MODEL_SEEDS)})
            return summary
        except Exception as exc:
            completed_ids = {entry["coordinate"]["history_id"] for entry in histories}
            attempted_records = [
                {"history_id": entry["coordinate"]["history_id"], "hard_status": entry["report"]["hard_status"]}
                for entry in histories
            ]
            attempted_records.extend(
                {"history_id": history_id, "hard_status": "INCOMPLETE"}
                for history_id in attempted_ids if history_id not in completed_ids
            )
            attempt.update({
                "status": "ABORTED", "attempted_history_ids": list(attempted_ids),
                "unattempted_history_ids": [history_id for history_id in history_ids if history_id not in attempted_ids],
                "error_code": getattr(exc, "code", "UNEXPECTED_FAILURE"),
                "error_type": type(exc).__name__, "error_message": str(exc),
            })
            _write_json_replace(output_root / "attempt.json", attempt)
            if not (output_root / "summary.json").exists():
                _write_json_new(output_root / "summary.json", {
                    "schema_id": "sprint22-pilot-summary-v1", "protocol_id": PROTOCOL_ID,
                    "binding_sha256": binding_digest, "status": "ABORTED_OPERATIONAL",
                    "history_count": HISTORY_COUNT, "attempted_history_ids": list(attempted_ids),
                    "unattempted_history_ids": attempt["unattempted_history_ids"],
                    "support_records": [entry["report"] for entry in histories],
                    "model_seed_results_separate": {str(seed): _null_full_roster(history_ids, attempted_records, "operational abort left full roster incomplete") for seed in MODEL_SEEDS},
                    "error": {"code": attempt["error_code"], "type": attempt["error_type"], "message": attempt["error_message"]},
                    "claim_limits": ["full-roster endpoint and uncertainty are uncomputable", "no retry, resume, replacement, or continued generation after abort"],
                })
            _append_ledger(ledger, {"event": "run_aborted", "code": attempt["error_code"], "message": attempt["error_message"], "attempted_history_ids": list(attempted_ids), "unattempted_history_ids": attempt["unattempted_history_ids"]})
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="verify reviewed binding and immutable assets without generation or scoring")
    mode.add_argument("--run", action="store_true", help="one-shot execution of all exactly bound histories")
    parser.add_argument("--worktree-root", type=Path, required=True, help="clean historical source worktree at the frozen base")
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="read-only original checkpoint directory")
    parser.add_argument("--binding", type=Path, required=True, help="Task 3 prospectively frozen binding JSON")
    parser.add_argument("--expected-binding-sha256", required=True, help="reviewed full binding digest")
    parser.add_argument("--protocol-path", type=Path, required=True, help="Task 3 reviewed protocol source")
    parser.add_argument("--output-dir", type=Path, required=True, help="exact one-shot output root bound in JSON")
    parser.add_argument("--review-passed", action="store_true", help="independent Task 3 pre-contact evidence gate passed")
    parser.add_argument("--main-release-binding-sha256", help="Main's explicit release of this exact binding for the one-shot run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        worktree_root = args.worktree_root.resolve()
        binding, binding_digest = read_binding(args.binding, args.expected_binding_sha256)
        _expect(args.output_dir.resolve(strict=False) == Path(binding["output"]["run_root"]).resolve(strict=False), "OUTPUT_PATH_MISMATCH", "output directory differs from exact binding")
        verified = verify_release(binding, binding_digest, worktree_root, args.checkpoint_root.resolve(), args.protocol_path.resolve())
        if args.dry_run:
            _expect(not args.review_passed and args.main_release_binding_sha256 is None, "CLI_CONTRACT_FAILURE", "dry-run cannot take pilot release acknowledgements")
            _expect(not args.output_dir.exists() and not args.output_dir.is_symlink(), "OUTPUT_ALREADY_EXISTS", "dry-run output root must remain untouched and absent")
            print(json.dumps({"status": "DRY_RUN_PASS_NO_GENERATION_OR_SCORING", "binding_sha256": binding_digest, "contact_or_generation": False, "preflight": False, "checkpoint_deserialization": False, "output_path_touched": False, "verified": verified}, sort_keys=True))
            return 0
        _expect(args.run and args.review_passed, "REVIEW_GATE_REQUIRED", "--run requires independent pre-contact PASS")
        _expect(args.main_release_binding_sha256 == binding_digest, "MAIN_RELEASE_REQUIRED", "--run requires Main's exact-digest release")
        _expect(binding.get("contact_authorized") is False, "BINDING_MISMATCH", "binding may not itself authorize contact")
        output_root = args.output_dir.resolve(strict=False)
        _expect(not output_root.exists() and not output_root.is_symlink(), "OUTPUT_ALREADY_EXISTS", f"one-shot output root already exists: {output_root}")
        output_root.mkdir(parents=False, exist_ok=False)
        result = _run_one_shot(binding, binding_digest, verified, output_root, worktree_root, args.checkpoint_root.resolve())
        print(json.dumps({"status": result["status"], "binding_sha256": binding_digest, "history_count": HISTORY_COUNT}, sort_keys=True))
        return 0 if result["status"] != "ABORTED_OPERATIONAL" else 2
    except PilotError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "code": exc.code, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "code": "UNEXPECTED_FAILURE", "type": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
