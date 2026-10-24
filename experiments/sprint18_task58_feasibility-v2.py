#!/usr/bin/env python3
"""Versioned, no-write paired feasibility study continuation runner.

``--smoke`` exercises the persistence path with disposable fixture data only.
``--continue`` imports the verified first v1 row and evaluates only the fixed
remaining 31 profile-seed pairs after independent Task 58 review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

CONTRACT_PATH = ROOT / "experiments/sprint18-isolated-feasibility-study-v2.md"
STUDY_SEEDS = tuple(range(918000, 918016))
FIXTURE_SEED = 93058
PROFILE_ORDER = ("sprint15-v7", "sprint18-c2")
PROFILE_RATE = {"sprint15-v7": 2.2e-5, "sprint18-c2": 3.3e-5}
PROFILE_CONFIG_REFERENCE_SEED = 0
PROFILE_CONFIG_REFERENCE_HASH = {
    "sprint15-v7": "7de530a28d23",
    "sprint18-c2": "720e31e94681",
}
PROFILE_QUOTA = {
    "p": 24,
    "p1": 12,
    "p2": 12,
    "w": 24,
    "w1": 12,
    "w2": 12,
    "a": 16,
    "a1": 8,
    "a2": 8,
    "controls": 48,
    "robot_days": 240,
    "robots_pos": 6,
    "robots_neg": 6,
    "programs": 2,
    "spacing_s": 1209600.0,
    "robot_pos_cap": 0.35,
    "robot_neg_cap": 0.4,
    "program_cap": 0.6,
}
COHORTS = ("P", "W", "A")
SUBTYPE_QUOTA_FIELD = {
    ("P", "P1"): "p1",
    ("P", "P2"): "p2",
    ("W", "W1"): "w1",
    ("W", "W2"): "w2",
    ("A", "A1"): "a1",
    ("A", "A2"): "a2",
}
BINDING_PATHS = (
    "experiments/sprint18-role-binding-v1.json",
    "experiments/sprint18-role-binding-c2-v2.json",
    "experiments/sprint18-role-binding-c3-v1.json",
    "experiments/sprint17-role-binding-v3.json",
)
ORIGINAL_OUTPUT_DIR = ROOT / "artifacts/sprint-18/isolated-feasibility-v1"
ORIGINAL_ATTEMPT_PATH = ORIGINAL_OUTPUT_DIR / "attempt.json"
ORIGINAL_RESULT_PATH = ORIGINAL_OUTPUT_DIR / "result.json"
OUTPUT_DIR = ROOT / "artifacts/sprint-18/isolated-feasibility-v2"
ATTEMPT_PATH = OUTPUT_DIR / "attempt.json"
RESULT_PATH = OUTPUT_DIR / "result.json"
FORBIDDEN_ROOT = ROOT / "data/generated/sprint18-isolated-feasibility-v1"
STUDY_ID = "sprint18-isolated-feasibility-study-v1"
AMENDMENT_ID = "sprint18-isolated-feasibility-study-v2-infrastructure-continuation"
ORIGINAL_RUNNER_PATH = ROOT / "experiments/sprint18_task58_feasibility-v1.py"
ORIGINAL_CONTRACT_PATH = ROOT / "experiments/sprint18-isolated-feasibility-study-v1.md"
ORIGINAL_RUNNER_SHA256 = "d65f01ee703d2f89b46c5b0138b3690f811edce304878204d9b1cbf14fa83111"
ORIGINAL_CONTRACT_SHA256 = "1252707032aabfd95d99d1944140e57f8eff3610f7c8d35810e649c65a276bd0"
ORIGINAL_ATTEMPT_SHA256 = "e4efabee11096ab19de0aa1f37f9b5f7f2743c1aa16c02e5d02a79584e1874e8"
ORIGINAL_RESULT_SHA256 = "d9f45f6cb40040281f0b4570653f2c06e9ec5135330fc8507dc505558c2df2e6"
ORIGINAL_ROW_SHA256 = "36da034b9fa0288310811f5c49212ac7869b9d486a1720b66659ca8d38652204"
ORIGINAL_FREEZE_SHA256 = "d0707a898dee7ddf8b3aacf153cba2cea53db6e7b1f142bf081ccee469bbf037"
ORIGINAL_SOURCE_CLOSURE_SHA256 = "2678d50a328fb4d34c941c38c8d04e91d85ad3c363f4793cd02e3aa0a5b0b834"

FROZEN_INPUT_SHA256 = {
    "experiments/sprint15-benchmark-protocol-v7.md": "a1fc09db2e246ed79d0595aec953a7788fd1b47c4981fbe1d92017d944b8d7b6",
    "experiments/sprint18-data-method-c2-v1.md": "f4cc8901c2e59f615ebcaaa3509fd654773efd06f7f6fd6a55faeab6b3ba4b9b",
    "experiments/sprint18-role-binding-v1.json": "493c5d6a7a9b807f1fe915f00bb917be45a11183512581606eee34ee678bb228",
    "experiments/sprint18-role-binding-c2-v2.json": "45818410f8d483d66a5523fd4ec38169bed9dc45d3a82f16979c05f9ab91dd92",
    "experiments/sprint18-role-binding-c3-v1.json": "6d87b9bba4e7e76e25917e6b00a13a52dc886e8ea8052ef6e5c88cf01a863ddd",
    "experiments/sprint17-role-binding-v3.json": "e73fee24cf84b60d17cbe4013ecc77ddace6e9132286ca8b65155039dd47a679",
    "pyproject.toml": "30292a091a273d0d58958701f792ad30af9a9bbd7454480275dedd73d1421d01",
    "uv.lock": "4a7866878c81cdd8f5ba283cd8d0a772073bfc75941f73ec761ede5eeb2e239a",
    "src/synth/__init__.py": "6b98c860a7d39e152f57f829d43a6eff4446d5301a1e9be6454a8945067e80c6",
    "src/synth/anomalies/__init__.py": "4185ee9d09f8b3c1bcd937dac5630a33882d972b9992797a338f6088ea4db5c5",
    "src/synth/anomalies/base.py": "b98c8727168e7f4e6eb88d0474b2f51a26df0ed5d26bdeae1abe39ca1a5df206",
    "src/synth/anomalies/contextual.py": "551dad79ba4266b3524f3e261e740c43818a7eb4ba3d8462880d726e9ee050b1",
    "src/synth/anomalies/cross_channel.py": "f40ed7c447e85726ad17248f6d165e3dd38203ad4e2f4ea039935bf4fc3421ca",
    "src/synth/anomalies/drift.py": "8bece76574231c58b85e309f8c879d54d3ee87ab50f4634e6ad58c23aa6190ba",
    "src/synth/anomalies/duration.py": "f7b4e0a324554dd4379f951a1223c00b12c624695f8bd92a088d085b3fb129d9",
    "src/synth/anomalies/easy_sanity.py": "41efd72d17ee7167d8b7ef4c6fb9ebfde29e37dc6ff43de951fbfb21b97a0e01",
    "src/synth/anomalies/freq_phase.py": "83eb1454f9e8d3015647525baf8d3f2fc36a2a0531eebd0a12a865c4da84ebcb",
    "src/synth/anomalies/missing_event.py": "464fa9daed5cfdbf13186e5ab406b960a6dcb4e028c584ec0cd683937fcd5549",
    "src/synth/anomalies/registry.py": "7d34156b920bf9c79486e2c1610d28706e6073c73b7363608c9798efdf1561d1",
    "src/synth/anomalies/regularity.py": "3bb17a410df99bce933dad7cf2f0d208c23197c06b5683e4ffec499a138cce7d",
    "src/synth/anomalies/stuck.py": "9c7e14ac4a6c9cbb7af3cfa8df0b90240e4eaaf74df7badd54dc8c4988e399c8",
    "src/synth/anomalies/transition.py": "1a92b361dfeff2df80939f95de84c7bbf59ebd0ccbc3629ab326bccee87be704",
    "src/synth/balanced.py": "d62de3422bd51f870d39d18b1a8bb942f9ca73fc0044d8f23cc2d0948bd35a43",
    "src/synth/chronicle.py": "f932e1a3c3bbc4bf7c8123e0ae9d3125200fedf0ce586d6d7eaf637d335f2cb6",
    "src/synth/cli.py": "19918890c50ed121cfc6de646ed98c73fd96651bfc1d0b18f8877d05972b550d",
    "src/synth/config.py": "55fed2ef5b253d48c1b0c7f1dd5f7f966cdd03ad250cb86a988ac8de23d5c056",
    "src/synth/contrastive.py": "af1a2ded1ce5bc9cc7fef9572599d0d75443c168b58b3907660c56deb7ac9e71",
    "src/synth/dataset.py": "375029c48e3a19d2e1e30f986bde5339fb597474e9f0685405fd548c0bc0e662",
    "src/synth/diagnostics.py": "1cd87aa40893e710e3b645749eed1cb84e8a8f3f8b9a21f0e97eb006c40f8eb6",
    "src/synth/events.py": "85100f5e0aca47dd2e8b01a08c58f39d32be4f1eab469cdc38e4a4b57768169d",
    "src/synth/generator.py": "32cd227943cb00c165c43aeb75835f14433a57c60ca9adca0e5c20d21e654403",
    "src/synth/health.py": "e5854641fb85eb9b88e05ee9d8e503c69a9432b13177d8c62ad38c3494e7fc6d",
    "src/synth/masking.py": "b73843b1fd6fd7a925d3dc6e58967bdb744de89cc1ca61927958040db0513a2c",
    "src/synth/normal.py": "a8203b2b9fe9f0489c249e2694dde4c17752976c6402ddc5907d0b94c565d6ac",
    "src/synth/patchify.py": "4b7e6876d6c43c76848d7b179a02e19f1c065aaecbfbd1eb159d448c45bc8d49",
    "src/synth/physics/__init__.py": "4e0d2ca9e0bec699cbd23dde004fcbfd8a422814d7a264fc11b05aa78b885a4b",
    "src/synth/physics/causal.py": "6e97fbe655fe0a1c55433e04eb23fb1922ce28537ec061875fda2632c6c8d183",
    "src/synth/physics/noise.py": "cf3f51ad9a571a3cf77c3c40aaf2e0d69b96afda860615edf9751b9f9f063d54",
    "src/synth/preflight15.py": "da1239b3e3d1df145389083c813ce97d99b7f80c32a9452d1d7a683b61dd5df6",
    "src/synth/probe15.py": "a08b3d5fb83001e1f5c43f4c56ff536bae85e41d494db289304aeb33a339242b",
    "src/synth/regimes.py": "e0287d2dc7aadf1ca50c0bf63151f1d79b11a7d7a14c68ee7eec0a21a981cafd",
    "src/synth/scheduled.py": "feb76b9d10c500e3dbea7fda2237e7daf8d1b41abe6cf8a7e6d38a9e73c72ce7",
    "src/synth/scheduler.py": "4659b94c31f3a2628f57d0fd789ea8f7d42c5f82abfc0f7dc01ffa2e7bf3e3af",
    "src/synth/schema.py": "6d6b001c2199933c14dbd019d9eaf8ded5c3dddc3d33997420f62175ee230203",
    "src/synth/splits.py": "0abace7461b2d5da9e2b616748f539e03aac8ef0e7b65fe4325f14a719fc0316",
    "src/synth/strength.py": "6b23db6b529e4c873e41ec8f25785412ce113184b1a29847e35571cf749c786f",
    "src/synth/temporal.py": "02a8ea8d7698794e5fbc0ec499e94d7a25e4fe7a3e888e1295cc13c567f36e18",
}


class ContractError(RuntimeError):
    """A frozen input, execution, or one-shot guard failed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _verify_frozen_inputs() -> dict[str, str]:
    mismatches: dict[str, dict[str, str | None]] = {}
    actual: dict[str, str] = {}
    for relative, expected in FROZEN_INPUT_SHA256.items():
        path = ROOT / relative
        digest = _file_sha256(path) if path.is_file() else None
        if digest is not None:
            actual[relative] = digest
        if digest != expected:
            mismatches[relative] = {"expected": expected, "actual": digest}
    if mismatches:
        raise ContractError(f"frozen source/config closure mismatch: {mismatches}")
    return actual


def _load_json(relative: str) -> dict[str, object]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object in {relative}")
    return cast("dict[str, object]", value)


def _recorded_seeds(binding: Mapping[str, object], label: str) -> set[int]:
    rows = binding.get("role_binding")
    if not isinstance(rows, list):
        raise ContractError(f"{label} has no recorded role_binding list")
    seeds: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or "data_seed" not in row:
            raise ContractError(f"{label} has a malformed recorded seed row")
        seeds.add(int(row["data_seed"]))
    return seeds


def _verify_namespace_disjointness() -> dict[str, object]:
    study = set(STUDY_SEEDS)
    if tuple(sorted(study)) != STUDY_SEEDS or len(study) != 16:
        raise ContractError("the frozen study seed block is not exact and ordered")
    source_bindings = {
        relative: _load_json(relative) for relative in BINDING_PATHS
    }
    overlaps: dict[str, list[int]] = {}
    recorded_seed_sets: dict[str, set[int]] = {}
    for relative, binding in source_bindings.items():
        seeds = _recorded_seeds(binding, relative)
        recorded_seed_sets[relative] = seeds
        overlap = sorted(study & seeds)
        if overlap:
            overlaps[relative] = overlap
    c3 = source_bindings["experiments/sprint18-role-binding-c3-v1.json"]
    ranges = c3.get("disjoint_seed_ranges_checked")
    if not isinstance(ranges, list):
        raise ContractError("C3 binding has no recorded disjoint-range checks")
    checked_ranges: list[dict[str, object]] = []
    range_sources: set[str] = set()
    for item in ranges:
        if not isinstance(item, dict) or item.get("result") != "disjoint":
            raise ContractError("a recorded C3 namespace check is not disjoint")
        source = str(item.get("source", ""))
        range_sources.add(source)
        if "seeds" in item:
            occupied = {int(seed) for seed in cast("list[object]", item["seeds"])}
            overlap = sorted(study & occupied)
        elif "min" in item and "max" in item:
            low = int(item["min"])
            high = int(item["max"])
            if low > high:
                raise ContractError(f"invalid recorded seed range: {source}")
            overlap = [seed for seed in STUDY_SEEDS if low <= seed <= high]
        else:
            raise ContractError(f"unsupported recorded namespace entry: {source}")
        if overlap:
            overlaps[source] = overlap
        checked_ranges.append({"source": source, "overlap": overlap})
    required_range_sources = {
        "Sprint15 accepted Candidate 1 band",
        "Sprint15 accepted Candidate 2 band",
        "Sprint15 accepted Candidate 3 band",
        "Sprint15 accepted Candidate 4 band",
        "Sprint15 accepted Candidate 5 band",
        "Sprint15 accepted Candidate 6 band",
        "Sprint15 accepted Candidate 7 band",
        "Sprint17 fresh ablation roster (binding v3)",
        "Sprint18 Cycle 1 retired roster",
        "Sprint18 Cycle 2 retired roster",
        "Sprint18 model seeds",
        "Sprint17 model seeds",
    }
    if missing := sorted(required_range_sources - range_sources):
        raise ContractError(f"C3 record omits expected namespace ranges: {missing}")
    if overlaps:
        raise ContractError(f"study seed collision in recorded bindings: {overlaps}")
    fixture_collisions: dict[str, list[int]] = {}
    for relative, seeds in recorded_seed_sets.items():
        if FIXTURE_SEED in seeds:
            fixture_collisions[relative] = [FIXTURE_SEED]
    for item in ranges:
        if not isinstance(item, dict):
            continue
        if "seeds" in item and FIXTURE_SEED in {
            int(seed) for seed in cast("list[object]", item["seeds"])
        }:
            fixture_collisions[str(item.get("source"))] = [FIXTURE_SEED]
        elif "min" in item and "max" in item:
            if int(item["min"]) <= FIXTURE_SEED <= int(item["max"]):
                fixture_collisions[str(item.get("source"))] = [FIXTURE_SEED]
    if FIXTURE_SEED in study or fixture_collisions:
        raise ContractError(f"fixture seed is not disposable: {fixture_collisions}")
    return {
        "study_seeds": list(STUDY_SEEDS),
        "fixture_seed": FIXTURE_SEED,
        "recorded_role_binding_files": list(BINDING_PATHS),
        "c3_recorded_range_checks": checked_ranges,
        "role_binding_overlaps": {},
        "fixture_overlaps": {},
        "sealed_inputs_opened": False,
    }


def _diff_values(left: object, right: object, path: str = "") -> list[tuple[str, object, object]]:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return [(path or "$", left, right)]
        differences: list[tuple[str, object, object]] = []
        for key in left:
            child = f"{path}.{key}" if path else str(key)
            differences.extend(_diff_values(left[key], right[key], child))
        return differences
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [(path or "$", left, right)]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                _diff_values(left_item, right_item, f"{path}[{index}]")
            )
        return differences
    return [] if left == right else [(path or "$", left, right)]


def _a_rate(cfg: object) -> float:
    cohorts = getattr(getattr(cfg, "health"), "cohorts")
    return float(next(cohort.abrupt_rate for cohort in cohorts
                      if cohort.cohort_id == "A"))


def _profile_factories() -> dict[str, object]:
    from synth.chronicle import (
        sprint15_v7_history_config,
        sprint18_c2_history_config,
    )

    return {
        "sprint15-v7": sprint15_v7_history_config,
        "sprint18-c2": sprint18_c2_history_config,
    }


def _resolve_pair_configs(seed: int) -> dict[str, object]:
    factories = _profile_factories()
    configs = {
        profile: factory(seed=seed)
        for profile, factory in factories.items()
    }
    for profile in PROFILE_ORDER:
        cfg = configs[profile]
        if int(getattr(getattr(cfg, "health"), "seed")) != seed:
            raise ContractError(f"{profile} did not resolve the bound seed")
        if _a_rate(cfg) != PROFILE_RATE[profile]:
            raise ContractError(f"{profile} resolved an unbound cohort-A rate")
    differences = _diff_values(
        asdict(configs["sprint15-v7"]),
        asdict(configs["sprint18-c2"]),
    )
    if len(differences) != 1:
        raise ContractError(f"v7/C2 config delta is not exactly one field: {differences}")
    path, v7_value, c2_value = differences[0]
    a_index = next(
        index for index, cohort in enumerate(
            getattr(getattr(configs["sprint15-v7"], "health"), "cohorts")
        ) if cohort.cohort_id == "A"
    )
    if path != f"health.cohorts[{a_index}].abrupt_rate":
        raise ContractError(f"v7/C2 changed a field other than A abrupt_rate: {path}")
    if v7_value != PROFILE_RATE["sprint15-v7"] or c2_value != PROFILE_RATE["sprint18-c2"]:
        raise ContractError("v7/C2 abrupt-rate literals changed")
    return configs


def _verify_config_contract() -> dict[str, object]:
    from synth.balanced import DEFAULT_QUOTA

    if DEFAULT_QUOTA.as_dict() != PROFILE_QUOTA:
        raise ContractError("the frozen exact-allocation quota differs")
    configs = _resolve_pair_configs(PROFILE_CONFIG_REFERENCE_SEED)
    actual_hashes = {
        profile: str(getattr(config, "hash")())
        for profile, config in configs.items()
    }
    if actual_hashes != PROFILE_CONFIG_REFERENCE_HASH:
        raise ContractError(
            f"profile reference config hash mismatch: {actual_hashes}"
        )
    return {
        "reference_seed": PROFILE_CONFIG_REFERENCE_SEED,
        "profile_config_hashes": actual_hashes,
        "profile_rates": dict(PROFILE_RATE),
        "one_field_delta": "health.cohorts[A].abrupt_rate",
        "quota": DEFAULT_QUOTA.as_dict(),
    }


def _release_binding() -> dict[str, object]:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- release-binding -->\s*```json\s*(.*?)\s*```",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ContractError("v2 contract has no release-binding JSON block")
    value = json.loads(match.group(1))
    if not isinstance(value, dict):
        raise ContractError("v2 contract release binding is not an object")
    return cast("dict[str, object]", value)


def _legacy_provenance_hashes() -> dict[str, str]:
    return {
        "runner_sha256": ORIGINAL_RUNNER_SHA256,
        "contract_sha256": ORIGINAL_CONTRACT_SHA256,
        "attempt_sha256": ORIGINAL_ATTEMPT_SHA256,
        "result_sha256": ORIGINAL_RESULT_SHA256,
        "row_sha256": ORIGINAL_ROW_SHA256,
        "freeze_sha256": ORIGINAL_FREEZE_SHA256,
        "source_closure_sha256": ORIGINAL_SOURCE_CLOSURE_SHA256,
    }


def _verify_static_guards() -> dict[str, object]:
    source_hashes = _verify_frozen_inputs()
    source_closure_sha256 = _sha256(_canonical_json(source_hashes))
    if (
        len(source_hashes) != 47
        or source_closure_sha256 != ORIGINAL_SOURCE_CLOSURE_SHA256
    ):
        raise ContractError("the original 47-file source/config closure changed")
    namespace = _verify_namespace_disjointness()
    configs = _verify_config_contract()
    release = _release_binding()
    runner_sha256 = _file_sha256(Path(__file__).resolve())
    if release.get("runner_sha256") != runner_sha256:
        raise ContractError("corrected runner differs from the v2 release binding")
    if release.get("source_closure_sha256") != source_closure_sha256:
        raise ContractError("v2 release binding has a different source closure")
    if release.get("profile_config_hashes") != configs["profile_config_hashes"]:
        raise ContractError("v2 release binding has different profile config hashes")
    if release.get("historical_provenance") != _legacy_provenance_hashes():
        raise ContractError("v2 release binding has different historical provenance")
    historical_files = {
        ORIGINAL_RUNNER_PATH: ORIGINAL_RUNNER_SHA256,
        ORIGINAL_CONTRACT_PATH: ORIGINAL_CONTRACT_SHA256,
    }
    for path, expected in historical_files.items():
        actual = _file_sha256(path) if path.is_file() else None
        if actual != expected:
            raise ContractError(f"historical v1 source changed: {path}")
    return {
        "source_closure": source_hashes,
        "source_closure_sha256": source_closure_sha256,
        "namespace": namespace,
        "config_contract": configs,
        "release_binding": release,
    }


def _freeze_digest(source_hashes: Mapping[str, str]) -> str:
    if not CONTRACT_PATH.is_file():
        raise ContractError(f"frozen study contract missing: {CONTRACT_PATH}")
    contract_hash = _file_sha256(CONTRACT_PATH)
    runner_hash = _file_sha256(Path(__file__).resolve())
    material = (
        b"SPRINT18-ISOLATED-FEASIBILITY-CONTINUATION-FREEZE-v2\n"
        + bytes.fromhex(contract_hash)
        + bytes.fromhex(runner_hash)
        + _canonical_json({
            "source_closure": dict(sorted(source_hashes.items())),
            "historical_provenance": _legacy_provenance_hashes(),
        })
    )
    return _sha256(material)


def _exact_allocation_record(
    rows: list[dict[str, object]],
    ledger: list[dict[str, object]],
    maintenance: dict[str, list[list[float]]],
    seed: int,
    quota: object,
) -> tuple[dict[str, object], bool]:
    from synth import balanced as B

    quota_config = cast("B.QuotaConfig", quota)
    try:
        allocation = B.allocate_quotas(
            rows, ledger, maintenance, seed, quota_config, method="exact"
        )
    except B.InfeasibleCandidate as exc:
        failure = exc.record
        return {
            "status": "INFEASIBLE",
            "feasible": False,
            "reason": failure.get("reason"),
            "missing": failure.get("missing"),
            "details": failure,
        }, False
    except B.AllocationUnavailable as exc:
        failure = exc.record
        return {
            "status": "UNAVAILABLE",
            "feasible": None,
            "reason": failure.get("reason"),
            "missing": failure.get("missing"),
            "details": failure,
        }, True
    except RuntimeError as exc:
        return {
            "status": "WITNESS-BREACH",
            "feasible": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, True
    solver = allocation.get("solver")
    return {
        "status": "FEASIBLE",
        "feasible": True,
        "allocated": allocation["selected"],
        "controls": len(cast("list[object]", allocation["controls"])),
        "margins": allocation["margins"],
        "rejection": allocation["rejection"],
        "solver": solver,
    }, False


def _metric_record(
    seed: int,
    profile: str,
    cfg: object,
    *,
    failure_probe: bool = False,
) -> tuple[dict[str, object], bool]:
    from synth import balanced as B
    from synth import events as E
    from synth.chronicle import (
        Patchifier,
        _failure_to_dict,
        _maintenance_windows,
        _manifest_file_rows,
        build_chronological,
    )

    started = time.perf_counter()
    row: dict[str, object] = {
        "history_seed": seed,
        "profile": profile,
        "protocol": "sprint15-benchmark-protocol-v7",
        "config_hash": str(getattr(cfg, "hash")()),
        "a_abrupt_rate": _a_rate(cfg),
        "generation_writes": False,
        "no_write": True,
        "evaluation_started": True,
    }
    try:
        _, health, labeled, _ = build_chronological(cfg)
        maintenance = _maintenance_windows(health)
        patchifier = Patchifier(getattr(cfg, "patch"))
        patch_counts = {
            sample.file_id: int(patchifier.patchify(sample).patches.shape[0])
            for sample in labeled
        }
        rows = _manifest_file_rows(labeled, maintenance, patch_counts)
        ledger = [_failure_to_dict(event) for event in health.failure_events]
        inputs = B.project_allocation_inputs(rows)
        eligible, rejection, _ = B.eligible_anchors(inputs, ledger, maintenance)

        raw_subtypes = Counter(
            (str(event["cohort"]), str(event["subtype"])) for event in ledger
        )
        eligible_subtypes = Counter(
            (str(event["cohort"]), str(event["subtype"])) for event in eligible
        )
        raw_counts = {
            cohort: sum(raw_subtypes[(cohort, subtype)]
                        for pair_cohort, subtype in B.BUCKET_ORDER
                        if pair_cohort == cohort)
            for cohort in COHORTS
        }
        eligible_counts = {
            cohort: sum(eligible_subtypes[(cohort, subtype)]
                        for pair_cohort, subtype in B.BUCKET_ORDER
                        if pair_cohort == cohort)
            for cohort in COHORTS
        }
        if sum(raw_counts.values()) != len(ledger):
            raise ContractError("raw cohort counts do not cover the complete ledger")
        if sum(eligible_counts.values()) != len(eligible):
            raise ContractError("eligible cohort counts do not cover the full eligible set")
        total_raw = sum(raw_counts.values())
        total_eligible = sum(eligible_counts.values())
        reset_by_cohort = Counter(
            str(event["cohort"])
            for event in ledger
            if E.positive_window_intersects_reset(event, maintenance)
        )
        rejected_by_cohort = {
            cohort: raw_counts[cohort] - eligible_counts[cohort]
            for cohort in COHORTS
        }
        reset_counts = {cohort: reset_by_cohort[cohort] for cohort in COHORTS}
        non_reset_rejected = {
            cohort: rejected_by_cohort[cohort] - reset_counts[cohort]
            for cohort in COHORTS
        }
        if any(count < 0 for count in non_reset_rejected.values()):
            raise ContractError("reset attrition exceeds raw-to-eligible attrition")
        if total_raw - total_eligible != sum(rejection.values()):
            raise ContractError("eligible attrition disagrees with audit rejection counts")
        if sum(reset_counts.values()) != int(rejection.get("reset-horizon", 0)):
            raise ContractError("reset attrition disagrees with the frozen audit predicate")

        quota = B.DEFAULT_QUOTA
        subtype_margins = []
        for cohort, subtype in B.BUCKET_ORDER:
            field = SUBTYPE_QUOTA_FIELD[(cohort, subtype)]
            need = int(getattr(quota, field))
            count = eligible_subtypes[(cohort, subtype)]
            subtype_margins.append({
                "cohort": cohort,
                "subtype": subtype,
                "eligible": count,
                "quota": need,
                "margin": count - need,
            })
        anchors = E.anchor_rows(inputs, maintenance)
        controls = E.select_control_windows(anchors, ledger, maintenance)
        row.update({
            "raw_event_counts": raw_counts,
            "raw_event_total": total_raw,
            "eligible_cohort_counts": eligible_counts,
            "eligible_total": total_eligible,
            "eligible_mix_fraction": {
                cohort: (eligible_counts[cohort] / total_eligible
                         if total_eligible else None)
                for cohort in COHORTS
            },
            "eligible_subtype_counts": {
                f"{cohort}/{subtype}": eligible_subtypes[(cohort, subtype)]
                for cohort, subtype in B.BUCKET_ORDER
            },
            "subtype_quota_margins": subtype_margins,
            "reset_attrition": {
                "reset_horizon_rejected_by_cohort": reset_counts,
                "reset_horizon_rejected_total": sum(reset_counts.values()),
                "all_rejected_by_cohort": rejected_by_cohort,
                "non_reset_rejected_by_cohort": non_reset_rejected,
                "eligible_anchor_rejections_by_reason": dict(rejection),
                "reset_fraction_of_raw_events": (
                    sum(reset_counts.values()) / total_raw if total_raw else None
                ),
            },
            "controls": {
                "count": len(controls),
                "quota": int(quota.controls),
                "margin": len(controls) - int(quota.controls),
            },
        })
        allocation_record, allocation_blocked = _exact_allocation_record(
            rows, ledger, maintenance, seed, quota
        )
        row["exact_allocation"] = allocation_record
        if allocation_blocked:
            row["execution_status"] = "BLOCKED"
            row["block_reason"] = (
                "exact solver did not prove feasibility or infeasibility"
                if allocation_record["status"] == "UNAVAILABLE"
                else "exact allocation witness validation failed"
            )
            row["elapsed_s"] = round(time.perf_counter() - started, 1)
            return row, True
        if failure_probe:
            impossible_quota = replace(
                quota,
                p1=eligible_subtypes[("P", "P1")] + 1,
            )
            forced_failure, forced_blocked = _exact_allocation_record(
                rows, ledger, maintenance, seed, impossible_quota
            )
            required = {
                "raw_event_counts", "eligible_cohort_counts", "eligible_mix_fraction",
                "eligible_subtype_counts", "subtype_quota_margins", "reset_attrition",
                "controls",
            }
            if forced_blocked or forced_failure["status"] != "INFEASIBLE":
                raise ContractError(
                    "smoke failure probe did not produce proven exact infeasibility"
                )
            failure_row = {**row, "exact_allocation": forced_failure}
            if not required <= failure_row.keys():
                raise ContractError(
                    "observational diagnostics were unavailable on the failure row"
                )
            row["failure_diagnostic_probe"] = {
                "status": forced_failure["status"],
                "reason": forced_failure.get("reason"),
                "diagnostic_fields_preserved": sorted(required),
                "study_quota_or_result_changed": False,
            }
        row["execution_status"] = "RECORDED"
    except Exception as exc:
        row["execution_status"] = "BLOCKED"
        row["block_reason"] = "evaluation infrastructure or audit error"
        row["infrastructure_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        row["elapsed_s"] = round(time.perf_counter() - started, 1)
        return row, True
    row["elapsed_s"] = round(time.perf_counter() - started, 1)
    return row, False


def _seal_row(row: dict[str, object]) -> dict[str, object]:
    sealed = dict(row)
    sealed["record_sha256"] = _sha256(_canonical_json(row))
    return sealed


def _paired_change(v7: Mapping[str, object], c2: Mapping[str, object]) -> dict[str, object]:
    numeric_fields = (
        "raw_event_total",
        "eligible_total",
        "eligible_cohort_counts",
        "eligible_mix_fraction",
        "eligible_subtype_counts",
    )
    changes: dict[str, object] = {}
    for field in numeric_fields:
        left = v7.get(field)
        right = c2.get(field)
        if isinstance(left, dict) and isinstance(right, dict):
            changes[field] = {
                key: (right[key] - left[key]
                      if isinstance(left[key], (int, float))
                      and isinstance(right[key], (int, float))
                      else None)
                for key in left.keys() & right.keys()
            }
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            changes[field] = right - left
        else:
            changes[field] = None
    for field, source in (
        ("reset_horizon_rejected_by_cohort", "reset_horizon_rejected_by_cohort"),
        ("all_rejected_by_cohort", "all_rejected_by_cohort"),
        ("non_reset_rejected_by_cohort", "non_reset_rejected_by_cohort"),
    ):
        left_reset = v7.get("reset_attrition")
        right_reset = c2.get("reset_attrition")
        if isinstance(left_reset, dict) and isinstance(right_reset, dict):
            left_values = left_reset.get(source)
            right_values = right_reset.get(source)
            changes[field] = {
                key: right_values[key] - left_values[key]
                for key in left_values.keys() & right_values.keys()
            } if isinstance(left_values, dict) and isinstance(right_values, dict) else None
    left_controls = v7.get("controls")
    right_controls = c2.get("controls")
    changes["controls_count"] = (
        int(right_controls["count"]) - int(left_controls["count"])
        if isinstance(left_controls, dict) and isinstance(right_controls, dict)
        else None
    )
    left_margins = {
        f"{row['cohort']}/{row['subtype']}": row["margin"]
        for row in cast("list[dict[str, object]]", v7.get("subtype_quota_margins", []))
    }
    right_margins = {
        f"{row['cohort']}/{row['subtype']}": row["margin"]
        for row in cast("list[dict[str, object]]", c2.get("subtype_quota_margins", []))
    }
    changes["subtype_quota_margins"] = {
        key: int(right_margins[key]) - int(left_margins[key])
        for key in left_margins.keys() & right_margins.keys()
    }
    left_allocation = cast("dict[str, object]", v7.get("exact_allocation", {}))
    right_allocation = cast("dict[str, object]", c2.get("exact_allocation", {}))
    changes["exact_allocation_feasible"] = {
        "sprint15-v7": left_allocation.get("feasible"),
        "sprint18-c2": right_allocation.get("feasible"),
        "both_feasible": (
            left_allocation.get("feasible") is True
            and right_allocation.get("feasible") is True
        ),
    }
    return changes


def _summarize(
    results: Sequence[Mapping[str, object]],
    status: str,
    seed_order: Sequence[int] = STUDY_SEEDS,
) -> dict[str, object]:
    seeds = tuple(seed_order)
    grouped: dict[int, dict[str, Mapping[str, object]]] = {}
    for row in results:
        seed = int(row["history_seed"])
        profile = str(row["profile"])
        grouped.setdefault(seed, {})[profile] = row
    paired = []
    for seed in seeds:
        rows = grouped.get(seed, {})
        if all(profile in rows for profile in PROFILE_ORDER):
            paired.append({
                "history_seed": seed,
                "v7": rows["sprint15-v7"],
                "c2": rows["sprint18-c2"],
                "c2_minus_v7": _paired_change(
                    rows["sprint15-v7"], rows["sprint18-c2"]
                ),
            })
    counts: dict[str, dict[str, object]] = {}
    for profile in PROFILE_ORDER:
        rows = [row for row in results if row.get("profile") == profile]
        feasibility = [
            cast("dict[str, object]", row.get("exact_allocation", {})).get("feasible")
            for row in rows
        ]
        all_seed_pairs_feasible = (
            len(rows) == len(seeds)
            and all(value is True for value in feasibility)
        )
        counts[profile] = {
            "completed_rows": len(rows),
            "proven_feasible_rows": sum(value is True for value in feasibility),
            "proven_infeasible_rows": sum(value is False for value in feasibility),
            "unavailable_rows": sum(value is None for value in feasibility),
            "all_seed_pairs_feasible": (
                all_seed_pairs_feasible if len(rows) == len(seeds) else None
            ),
            "all_16_feasible": (
                all_seed_pairs_feasible
                if len(seeds) == len(STUDY_SEEDS) and len(rows) == len(seeds)
                else None
            ),
        }
    complete_pairs = len(paired) == len(seeds)
    complete_16_pairs = len(seeds) == len(STUDY_SEEDS) and complete_pairs
    return {
        "run_status": status,
        "study_evaluations_planned": len(seeds) * len(PROFILE_ORDER),
        "study_evaluations_completed": sum(
            row.get("evaluation_started") is True for row in results
        ),
        "configured_seed_count": len(seeds),
        "complete_seed_pairs": complete_pairs,
        "complete_16_pairs": complete_16_pairs,
        "profiles": counts,
        "paired_rows": paired,
        "descriptive_all_16_conjunction": {
            "v7_all_16_feasible": counts["sprint15-v7"]["all_16_feasible"],
            "c2_all_16_feasible": counts["sprint18-c2"]["all_16_feasible"],
            "both_profiles_feasible_on_all_16": (
                all(
                    cast("dict[str, object]", pair["c2_minus_v7"])
                    ["exact_allocation_feasible"]["both_feasible"]
                    for pair in paired
                ) if complete_16_pairs else None
            ),
            "interpretation": (
                "descriptive only; not a population pass-probability estimate "
                "or independent Confirmation"
            ),
        },
    }

def _write_json_atomic(path: Path, value: object, *, exclusive: bool = False) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temp = path.with_name(path.name + ".tmp")
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
        return
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
        stream.flush()
    temp.replace(path)


def _claim_attempt(output_dir: Path, metadata: Mapping[str, object]) -> Path:
    if output_dir.exists():
        raise ContractError(f"one-shot output already exists; no retry/resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    attempt = output_dir / "attempt.json"
    try:
        _write_json_atomic(attempt, metadata, exclusive=True)
    except Exception:
        # The directory itself remains as a consumed claim if marker creation fails.
        raise ContractError(f"could not persist one-shot attempt marker: {attempt}")
    return attempt


def _row_sha256(row: Mapping[str, object]) -> str:
    unsealed = dict(row)
    unsealed.pop("record_sha256", None)
    return _sha256(_canonical_json(unsealed))


def _validate_result_prefix(
    results: Sequence[Mapping[str, object]],
    expected_coordinates: Sequence[tuple[int, str]],
    *,
    require_complete: bool = False,
) -> None:
    if len(results) > len(expected_coordinates):
        raise ContractError("continuation contains extra source evaluations")
    if require_complete and len(results) != len(expected_coordinates):
        raise ContractError("continuation is missing one or more fixed evaluations")
    actual_coordinates: list[tuple[int, str]] = []
    for index, row in enumerate(results):
        coordinate = (int(row.get("history_seed", -1)), str(row.get("profile", "")))
        actual_coordinates.append(coordinate)
        if coordinate != expected_coordinates[index]:
            raise ContractError(
                f"continuation row/order mismatch at {index}: "
                f"{coordinate} != {expected_coordinates[index]}"
            )
        digest = row.get("record_sha256")
        if not isinstance(digest, str) or _row_sha256(row) != digest:
            raise ContractError(f"continuation row {index} has an invalid seal")
        if row.get("no_write") is not True or row.get("generation_writes") is not False:
            raise ContractError(f"continuation row {index} breached no-write isolation")
    if len(set(actual_coordinates)) != len(actual_coordinates):
        raise ContractError("continuation contains duplicate source evaluations")
    if require_complete and any(row.get("evaluation_started") is not True for row in results):
        raise ContractError("continuation contains a missing source evaluation")


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"could not read provenance JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"provenance JSON is not an object: {path}")
    return cast("dict[str, object]", value)


def _verify_prior_evidence(
    attempt_path: Path,
    result_path: Path,
    *,
    expected_attempt_sha256: str,
    expected_result_sha256: str,
    expected_row_sha256: str,
    expected_freeze_sha256: str,
    expected_seed_order: Sequence[int],
    source_closure_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        attempt_bytes = attempt_path.read_bytes()
        result_bytes = result_path.read_bytes()
    except OSError as exc:
        raise ContractError(f"original continuation evidence is unavailable: {exc}") from exc
    if _sha256(attempt_bytes) != expected_attempt_sha256:
        raise ContractError("original attempt marker digest does not match the amendment")
    if _sha256(result_bytes) != expected_result_sha256:
        raise ContractError("original result digest does not match the amendment")
    attempt = _read_object(attempt_path)
    result = _read_object(result_path)
    seeds = tuple(expected_seed_order)
    coordinates = [(seed, profile) for seed in seeds for profile in PROFILE_ORDER]
    budget = len(coordinates)
    if (
        attempt.get("study_id") != STUDY_ID
        or attempt.get("attempt") != 1
        or attempt.get("status") != "RUNNING"
        or attempt.get("budget") != budget
        or attempt.get("freeze_digest") != expected_freeze_sha256
        or attempt.get("seed_order") != list(seeds)
        or attempt.get("profile_order") != list(PROFILE_ORDER)
        or attempt.get("source_closure_sha256") != source_closure_sha256
        or attempt.get("generated_data_roots_written") is not False
    ):
        raise ContractError("original attempt marker state/provenance is not the frozen partial attempt")
    if (
        result.get("study_id") != STUDY_ID
        or result.get("status") != "RUNNING"
        or result.get("freeze_digest") != expected_freeze_sha256
        or result.get("seed_order") != list(seeds)
        or result.get("profile_order") != list(PROFILE_ORDER)
        or result.get("source_closure_sha256") != source_closure_sha256
        or result.get("generated_data_roots_written") is not False
    ):
        raise ContractError("original result state/provenance is not the frozen partial result")
    source_closure = result.get("source_closure")
    if (
        not isinstance(source_closure, dict)
        or _sha256(_canonical_json(source_closure)) != source_closure_sha256
    ):
        raise ContractError("original result source closure is corrupt or mismatched")
    raw_results = result.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != 1:
        raise ContractError("original result must contain exactly one sealed row")
    imported = raw_results[0]
    if not isinstance(imported, dict):
        raise ContractError("original result row is malformed")
    row = cast("dict[str, object]", imported)
    _validate_result_prefix([row], coordinates)
    if row.get("record_sha256") != expected_row_sha256:
        raise ContractError("original v7 row digest does not match the amendment")
    if row.get("evaluation_started") is not True:
        raise ContractError("original v7 row does not prove a completed source evaluation")
    summary = result.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("study_evaluations_completed") != 1
        or summary.get("complete_16_pairs") is not False
    ):
        raise ContractError("original partial result summary is inconsistent")
    return attempt, row


def _record_study_result(
    status: str,
    results: list[dict[str, object]],
    static: Mapping[str, object],
    freeze_digest: str,
    output_dir: Path,
    seed_order: Sequence[int],
    continuation_provenance: Mapping[str, object],
) -> Path:
    result_path = output_dir / "result.json"
    attempt_path = output_dir / "attempt.json"
    record: dict[str, object] = {
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "freeze_digest": freeze_digest,
        "original_freeze_digest": ORIGINAL_FREEZE_SHA256,
        "status": status,
        "source_closure_sha256": static["source_closure_sha256"],
        "source_closure": static["source_closure"],
        "namespace": static["namespace"],
        "config_contract": static["config_contract"],
        "profile_order": list(PROFILE_ORDER),
        "seed_order": list(seed_order),
        "continuation_provenance": dict(continuation_provenance),
        "results": results,
        "summary": _summarize(results, status, seed_order),
        "research_evidence_writes_only": [
            str(attempt_path).replace("\\", "/"),
            str(result_path).replace("\\", "/"),
        ],
        "generated_data_roots_written": False,
    }
    _write_json_atomic(result_path, record)
    return result_path


def _verify_historical_evidence_bytes() -> dict[str, str]:
    paths = {
        ORIGINAL_RUNNER_PATH: ORIGINAL_RUNNER_SHA256,
        ORIGINAL_CONTRACT_PATH: ORIGINAL_CONTRACT_SHA256,
        ORIGINAL_ATTEMPT_PATH: ORIGINAL_ATTEMPT_SHA256,
        ORIGINAL_RESULT_PATH: ORIGINAL_RESULT_SHA256,
    }
    observed: dict[str, str] = {}
    for path, expected in paths.items():
        actual = _file_sha256(path) if path.is_file() else None
        if actual != expected:
            raise ContractError(f"original v1 evidence changed: {path}")
        observed[str(path.relative_to(ROOT)).replace("\\", "/")] = cast("str", actual)
    return observed


def _path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _continuation_metadata(
    attempt_path: Path,
    result_path: Path,
    *,
    prior_attempt_sha256: str,
    prior_result_sha256: str,
    prior_row_sha256: str,
    prior_freeze_sha256: str,
    imported_row: Mapping[str, object],
    remaining_coordinates: Sequence[tuple[int, str]],
) -> dict[str, object]:
    return {
        "prior_attempt_path": _path_label(attempt_path),
        "prior_attempt_sha256": prior_attempt_sha256,
        "prior_result_path": _path_label(result_path),
        "prior_result_sha256": prior_result_sha256,
        "prior_row_sha256": prior_row_sha256,
        "prior_freeze_sha256": prior_freeze_sha256,
        "imported_coordinates": [{
            "history_seed": imported_row["history_seed"],
            "profile": imported_row["profile"],
        }],
        "remaining_coordinates": [
            {"history_seed": seed, "profile": profile}
            for seed, profile in remaining_coordinates
        ],
        "imported_evaluation_count": 1,
        "remaining_evaluation_count": len(remaining_coordinates),
        "total_evaluation_count": 1 + len(remaining_coordinates),
    }


def run_continuation(
    static: Mapping[str, object],
    *,
    output_dir: Path,
    prior_attempt_path: Path,
    prior_result_path: Path,
    expected_attempt_sha256: str,
    expected_result_sha256: str,
    expected_row_sha256: str,
    prior_freeze_sha256: str,
    seed_order: Sequence[int],
    freeze_digest: str,
) -> int:
    if output_dir.exists():
        raise ContractError(
            f"one-shot continuation output already exists; no retry or resume: {output_dir}"
        )
    if FORBIDDEN_ROOT.exists():
        raise ContractError(f"forbidden persistent data root exists: {FORBIDDEN_ROOT}")
    seeds = tuple(seed_order)
    if seeds not in (STUDY_SEEDS, (FIXTURE_SEED,)):
        raise ContractError("continuation seed order differs from the fixed study or smoke fixture")
    expected_coordinates = [
        (seed, profile) for seed in seeds for profile in PROFILE_ORDER
    ]
    _, imported_row = _verify_prior_evidence(
        prior_attempt_path,
        prior_result_path,
        expected_attempt_sha256=expected_attempt_sha256,
        expected_result_sha256=expected_result_sha256,
        expected_row_sha256=expected_row_sha256,
        expected_freeze_sha256=prior_freeze_sha256,
        expected_seed_order=seeds,
        source_closure_sha256=str(static["source_closure_sha256"]),
    )
    remaining_coordinates = expected_coordinates[1:]
    provenance = _continuation_metadata(
        prior_attempt_path,
        prior_result_path,
        prior_attempt_sha256=expected_attempt_sha256,
        prior_result_sha256=expected_result_sha256,
        prior_row_sha256=expected_row_sha256,
        prior_freeze_sha256=prior_freeze_sha256,
        imported_row=imported_row,
        remaining_coordinates=remaining_coordinates,
    )
    if provenance["total_evaluation_count"] != len(STUDY_SEEDS) * len(PROFILE_ORDER) and seeds == STUDY_SEEDS:
        raise ContractError("continuation budget differs from the frozen 32 evaluations")
    started = datetime.now(timezone.utc).isoformat()
    total_budget = len(expected_coordinates)
    attempt_path = _claim_attempt(output_dir, {
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "attempt": 1,
        "status": "RUNNING",
        "started_at_utc": started,
        "freeze_digest": freeze_digest,
        "original_freeze_digest": prior_freeze_sha256,
        "source_closure_sha256": static["source_closure_sha256"],
        "seed_order": list(seeds),
        "profile_order": list(PROFILE_ORDER),
        "total_evaluation_budget": total_budget,
        "imported_evaluations": 1,
        "new_evaluation_budget": len(remaining_coordinates),
        "completed_evaluations": 1,
        "new_evaluations_completed": 0,
        "remaining_evaluations": len(remaining_coordinates),
        "last_record_sha256": expected_row_sha256,
        "continuation_provenance": provenance,
        "generated_data_roots_written": False,
    })
    results = [imported_row]

    def persist(status: str, block_reason: str | None = None) -> tuple[Path, str]:
        _validate_result_prefix(results, expected_coordinates)
        result_path = _record_study_result(
            status,
            results,
            static,
            freeze_digest,
            output_dir,
            seeds,
            provenance,
        )
        result_sha256 = _file_sha256(result_path)
        completed = sum(row.get("evaluation_started") is True for row in results)
        new_completed = sum(
            row.get("evaluation_started") is True for row in results[1:]
        )
        marker: dict[str, object] = {
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "attempt": 1,
            "status": status,
            "started_at_utc": started,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_digest": freeze_digest,
            "original_freeze_digest": prior_freeze_sha256,
            "source_closure_sha256": static["source_closure_sha256"],
            "seed_order": list(seeds),
            "profile_order": list(PROFILE_ORDER),
            "total_evaluation_budget": total_budget,
            "imported_evaluations": 1,
            "new_evaluation_budget": len(remaining_coordinates),
            "completed_evaluations": completed,
            "new_evaluations_completed": new_completed,
            "remaining_evaluations": total_budget - completed,
            "last_record_sha256": results[-1]["record_sha256"],
            "result_sha256": result_sha256,
            "continuation_provenance": provenance,
            "generated_data_roots_written": False,
        }
        if block_reason is not None:
            marker["block_reason"] = block_reason
        if status == "COMPLETE":
            marker["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(attempt_path, marker)
        return result_path, result_sha256

    persist("RUNNING")
    for seed in seeds:
        pending_profiles = [
            profile
            for row_seed, profile in remaining_coordinates
            if row_seed == seed
        ]
        if not pending_profiles:
            continue
        try:
            configs = _resolve_pair_configs(seed)
        except Exception as exc:
            profile = pending_profiles[0]
            blocked_row = _seal_row({
                "history_seed": seed,
                "profile": profile,
                "execution_status": "BLOCKED",
                "evaluation_started": False,
                "generation_writes": False,
                "no_write": True,
                "block_reason": "profile config resolution failed before evaluation",
                "infrastructure_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            })
            results.append(blocked_row)
            persist("BLOCKED", str(blocked_row["block_reason"]))
            return 2
        for profile in pending_profiles:
            row, blocked = _metric_record(seed, profile, configs[profile])
            sealed_row = _seal_row(row)
            results.append(sealed_row)
            status = "BLOCKED" if blocked else "RUNNING"
            block_reason = str(row.get("block_reason")) if blocked else None
            persist(status, block_reason)
            if blocked or sealed_row.get("evaluation_started") is not True:
                return 2

    if seeds == STUDY_SEEDS:
        if output_dir != OUTPUT_DIR or len(results) != len(STUDY_SEEDS) * len(PROFILE_ORDER):
            raise ContractError("only the fixed output may complete the frozen 32-evaluation continuation")
    elif seeds == (FIXTURE_SEED,):
        if output_dir.resolve() == OUTPUT_DIR.resolve():
            raise ContractError("disposable fixture continuation cannot claim the fixed study output")
    else:
        raise ContractError("only the frozen study or disposable fixture may complete")
    result_path, result_sha256 = persist("COMPLETE")
    final_result = _read_object(result_path)
    summary = cast("dict[str, object]", final_result["summary"])
    print(json.dumps({
        "status": "COMPLETE",
        "result": _path_label(result_path),
        "result_sha256": result_sha256,
        "freeze_digest": freeze_digest,
        "completed_evaluations": summary["study_evaluations_completed"],
        "imported_evaluations": 1,
        "new_evaluations": len(remaining_coordinates),
        "summary": summary["descriptive_all_16_conjunction"],
    }, indent=2, sort_keys=True))
    return 0


def _verify_fixture_continuation(
    output_dir: Path,
    imported_row: Mapping[str, object],
    prior_attempt_sha256: str,
    prior_result_sha256: str,
) -> dict[str, object]:
    result_path = output_dir / "result.json"
    attempt_path = output_dir / "attempt.json"
    result = _read_object(result_path)
    attempt = _read_object(attempt_path)
    result_sha256 = _file_sha256(result_path)
    rows = result.get("results")
    summary = result.get("summary")
    provenance = result.get("continuation_provenance")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ContractError("fixture continuation did not persist exactly one pair")
    if not isinstance(summary, dict) or not isinstance(provenance, dict):
        raise ContractError("fixture continuation omitted its final summary/provenance")
    typed_rows = cast("list[dict[str, object]]", rows)
    _validate_result_prefix(
        typed_rows,
        [(FIXTURE_SEED, PROFILE_ORDER[0]), (FIXTURE_SEED, PROFILE_ORDER[1])],
        require_complete=True,
    )
    if typed_rows[0] != imported_row:
        raise ContractError("fixture continuation altered the sealed imported row")
    if (
        result.get("status") != "COMPLETE"
        or result.get("generated_data_roots_written") is not False
        or summary.get("study_evaluations_planned") != 2
        or summary.get("study_evaluations_completed") != 2
        or summary.get("complete_seed_pairs") is not True
        or summary.get("complete_16_pairs") is not False
    ):
        raise ContractError("fixture continuation final summary is incomplete or inconsistent")
    conjunction = summary.get("descriptive_all_16_conjunction")
    if (
        not isinstance(conjunction, dict)
        or conjunction.get("v7_all_16_feasible") is not None
        or conjunction.get("c2_all_16_feasible") is not None
        or conjunction.get("both_profiles_feasible_on_all_16") is not None
    ):
        raise ContractError("fixture continuation reported a non-study all-16 conclusion")
    if (
        attempt.get("status") != "COMPLETE"
        or attempt.get("completed_evaluations") != 2
        or attempt.get("imported_evaluations") != 1
        or attempt.get("new_evaluations_completed") != 1
        or attempt.get("remaining_evaluations") != 0
        or attempt.get("last_record_sha256") != typed_rows[-1]["record_sha256"]
        or attempt.get("result_sha256") != result_sha256
        or attempt.get("continuation_provenance") != provenance
        or provenance.get("prior_attempt_sha256") != prior_attempt_sha256
        or provenance.get("prior_result_sha256") != prior_result_sha256
    ):
        raise ContractError("fixture continuation attempt marker does not bind its final result")
    return {
        "result_sha256": result_sha256,
        "attempt_sha256": _file_sha256(attempt_path),
        "summary": summary,
        "rows": typed_rows,
    }


def run_smoke(static: Mapping[str, object]) -> dict[str, object]:
    if OUTPUT_DIR.exists() or FORBIDDEN_ROOT.exists():
        raise ContractError("continuation output/root already exists; smoke refuses to proceed")
    historical_before = _verify_historical_evidence_bytes()
    freeze_digest = _freeze_digest(
        cast("Mapping[str, str]", static["source_closure"])
    )
    configs = _resolve_pair_configs(FIXTURE_SEED)
    first_raw, blocked = _metric_record(
        FIXTURE_SEED,
        PROFILE_ORDER[0],
        configs[PROFILE_ORDER[0]],
        failure_probe=True,
    )
    if blocked:
        raise ContractError(f"smoke was blocked: {first_raw.get('block_reason')}")
    imported_row = _seal_row(first_raw)
    required = {
        "raw_event_counts", "eligible_cohort_counts", "eligible_mix_fraction",
        "eligible_subtype_counts", "subtype_quota_margins", "reset_attrition",
        "controls", "exact_allocation", "no_write", "record_sha256",
    }
    if not required <= imported_row.keys():
        raise ContractError(f"smoke metrics missing: {sorted(required - imported_row.keys())}")
    if imported_row["no_write"] is not True or imported_row["generation_writes"] is not False:
        raise ContractError("smoke did not preserve no-write generation isolation")
    failure_probe = imported_row.get("failure_diagnostic_probe")
    expected_failure_diagnostics = {
        "raw_event_counts",
        "eligible_cohort_counts",
        "eligible_mix_fraction",
        "eligible_subtype_counts",
        "subtype_quota_margins",
        "reset_attrition",
        "controls",
    }
    preserved = (
        failure_probe.get("diagnostic_fields_preserved")
        if isinstance(failure_probe, dict)
        else None
    )
    if (
        not isinstance(failure_probe, dict)
        or failure_probe.get("status") != "INFEASIBLE"
        or not isinstance(preserved, list)
        or not expected_failure_diagnostics <= set(preserved)
    ):
        raise ContractError("smoke did not prove diagnostics survive an infeasibility row")

    with tempfile.TemporaryDirectory(prefix="s18-feasibility-continuation-") as temp:
        temp_root = Path(temp)
        legacy_attempt_path = temp_root / "v1-attempt.json"
        legacy_result_path = temp_root / "v1-result.json"
        output_dir = temp_root / "v2-continuation"
        started = datetime.now(timezone.utc).isoformat()
        fixture_rows = [imported_row]
        _write_json_atomic(legacy_attempt_path, {
            "study_id": STUDY_ID,
            "attempt": 1,
            "status": "RUNNING",
            "started_at_utc": started,
            "freeze_digest": ORIGINAL_FREEZE_SHA256,
            "budget": len(PROFILE_ORDER),
            "seed_order": [FIXTURE_SEED],
            "profile_order": list(PROFILE_ORDER),
            "source_closure_sha256": static["source_closure_sha256"],
            "generated_data_roots_written": False,
        })
        _write_json_atomic(legacy_result_path, {
            "study_id": STUDY_ID,
            "freeze_digest": ORIGINAL_FREEZE_SHA256,
            "status": "RUNNING",
            "source_closure_sha256": static["source_closure_sha256"],
            "source_closure": static["source_closure"],
            "namespace": static["namespace"],
            "config_contract": static["config_contract"],
            "profile_order": list(PROFILE_ORDER),
            "seed_order": [FIXTURE_SEED],
            "results": fixture_rows,
            "summary": _summarize(fixture_rows, "RUNNING", (FIXTURE_SEED,)),
            "research_evidence_writes_only": [
                _path_label(legacy_attempt_path),
                _path_label(legacy_result_path),
            ],
            "generated_data_roots_written": False,
        })
        legacy_attempt_sha256 = _file_sha256(legacy_attempt_path)
        legacy_result_sha256 = _file_sha256(legacy_result_path)
        rejected_guards: list[str] = []

        def expect_rejected(
            label: str,
            reason: str,
            action: Callable[[], object],
        ) -> None:
            try:
                action()
            except ContractError as exc:
                if reason not in str(exc):
                    raise ContractError(
                        f"smoke guard {label!r} failed for an unexpected reason: {exc}"
                    ) from exc
                rejected_guards.append(label)
            else:
                raise ContractError(f"smoke guard accepted invalid evidence: {label}")

        valid_attempt_bytes = legacy_attempt_path.read_bytes()
        valid_result_bytes = legacy_result_path.read_bytes()

        def check_prior(
            *,
            expected_attempt_sha256: str | None = None,
            expected_result_sha256: str | None = None,
        ) -> object:
            return _verify_prior_evidence(
                legacy_attempt_path,
                legacy_result_path,
                expected_attempt_sha256=(
                    expected_attempt_sha256
                    if expected_attempt_sha256 is not None
                    else _file_sha256(legacy_attempt_path)
                ),
                expected_result_sha256=(
                    expected_result_sha256
                    if expected_result_sha256 is not None
                    else _file_sha256(legacy_result_path)
                ),
                expected_row_sha256=str(imported_row["record_sha256"]),
                expected_freeze_sha256=ORIGINAL_FREEZE_SHA256,
                expected_seed_order=(FIXTURE_SEED,),
                source_closure_sha256=str(static["source_closure_sha256"]),
            )

        expect_rejected(
            "mismatched result digest",
            "original result digest does not match",
            lambda: check_prior(expected_result_sha256="0" * 64),
        )

        invalid_attempt = _read_object(legacy_attempt_path)
        invalid_attempt["status"] = "COMPLETE"
        _write_json_atomic(legacy_attempt_path, invalid_attempt)
        expect_rejected(
            "completed original attempt",
            "original attempt marker state/provenance",
            check_prior,
        )
        legacy_attempt_path.write_bytes(valid_attempt_bytes)

        invalid_attempt = _read_object(legacy_attempt_path)
        invalid_attempt["source_closure_sha256"] = "0" * 64
        _write_json_atomic(legacy_attempt_path, invalid_attempt)
        expect_rejected(
            "mismatched attempt provenance",
            "original attempt marker state/provenance",
            check_prior,
        )
        legacy_attempt_path.write_bytes(valid_attempt_bytes)

        invalid_result = _read_object(legacy_result_path)
        invalid_result["source_closure"] = {}
        _write_json_atomic(legacy_result_path, invalid_result)
        expect_rejected(
            "corrupt result source closure",
            "original result source closure is corrupt or mismatched",
            check_prior,
        )
        legacy_result_path.write_bytes(valid_result_bytes)

        invalid_result = _read_object(legacy_result_path)
        invalid_result["results"] = []
        _write_json_atomic(legacy_result_path, invalid_result)
        expect_rejected(
            "missing imported row",
            "original result must contain exactly one sealed row",
            check_prior,
        )
        legacy_result_path.write_bytes(valid_result_bytes)

        invalid_result = _read_object(legacy_result_path)
        invalid_result["results"] = [imported_row, imported_row]
        _write_json_atomic(legacy_result_path, invalid_result)
        expect_rejected(
            "extra imported row",
            "original result must contain exactly one sealed row",
            check_prior,
        )
        legacy_result_path.write_bytes(valid_result_bytes)

        invalid_row = dict(imported_row)
        invalid_row.pop("record_sha256")
        invalid_row["profile"] = PROFILE_ORDER[1]
        invalid_result = _read_object(legacy_result_path)
        invalid_result["results"] = [_seal_row(invalid_row)]
        _write_json_atomic(legacy_result_path, invalid_result)
        expect_rejected(
            "reordered imported row",
            "continuation row/order mismatch",
            check_prior,
        )
        legacy_result_path.write_bytes(valid_result_bytes)

        invalid_row = dict(imported_row)
        invalid_row["raw_event_counts"] = {"corrupt": 1}
        invalid_result = _read_object(legacy_result_path)
        invalid_result["results"] = [invalid_row]
        _write_json_atomic(legacy_result_path, invalid_result)
        expect_rejected(
            "corrupt imported row seal",
            "invalid seal",
            check_prior,
        )
        legacy_result_path.write_bytes(valid_result_bytes)

        run_continuation(
            static,
            output_dir=output_dir,
            prior_attempt_path=legacy_attempt_path,
            prior_result_path=legacy_result_path,
            expected_attempt_sha256=legacy_attempt_sha256,
            expected_result_sha256=legacy_result_sha256,
            expected_row_sha256=str(imported_row["record_sha256"]),
            prior_freeze_sha256=ORIGINAL_FREEZE_SHA256,
            seed_order=(FIXTURE_SEED,),
            freeze_digest=freeze_digest,
        )
        fixture_result = _verify_fixture_continuation(
            output_dir,
            imported_row,
            legacy_attempt_sha256,
            legacy_result_sha256,
        )
        fixture_rows = cast("list[dict[str, object]]", fixture_result["rows"])
        fixture_coordinates = [
            (FIXTURE_SEED, profile) for profile in PROFILE_ORDER
        ]
        expect_rejected(
            "extra continuation evaluation",
            "extra source evaluations",
            lambda: _validate_result_prefix(
                [*fixture_rows, fixture_rows[-1]],
                fixture_coordinates,
            ),
        )
        expect_rejected(
            "missing continuation evaluation",
            "missing one or more fixed evaluations",
            lambda: _validate_result_prefix(
                fixture_rows[:1],
                fixture_coordinates,
                require_complete=True,
            ),
        )
        expect_rejected(
            "reordered continuation evaluations",
            "row/order mismatch",
            lambda: _validate_result_prefix(
                list(reversed(fixture_rows)),
                fixture_coordinates,
                require_complete=True,
            ),
        )
        incomplete_row = dict(fixture_rows[-1])
        incomplete_row.pop("record_sha256")
        incomplete_row["evaluation_started"] = False
        incomplete_row = _seal_row(incomplete_row)
        expect_rejected(
            "incomplete continuation evaluation",
            "continuation contains a missing source evaluation",
            lambda: _validate_result_prefix(
                [fixture_rows[0], incomplete_row],
                fixture_coordinates,
                require_complete=True,
            ),
        )
        output_result_path = output_dir / "result.json"
        output_attempt_path = output_dir / "attempt.json"
        output_before_retry = (
            _file_sha256(output_result_path),
            _file_sha256(output_attempt_path),
        )
        try:
            run_continuation(
                static,
                output_dir=output_dir,
                prior_attempt_path=legacy_attempt_path,
                prior_result_path=legacy_result_path,
                expected_attempt_sha256=legacy_attempt_sha256,
                expected_result_sha256=legacy_result_sha256,
                expected_row_sha256=str(imported_row["record_sha256"]),
                prior_freeze_sha256=ORIGINAL_FREEZE_SHA256,
                seed_order=(FIXTURE_SEED,),
                freeze_digest=freeze_digest,
            )
        except ContractError as exc:
            if "one-shot continuation output already exists" not in str(exc):
                raise
        else:
            raise ContractError("continuation output allowed a duplicate attempt")
        if output_before_retry != (
            _file_sha256(output_result_path),
            _file_sha256(output_attempt_path),
        ):
            raise ContractError("duplicate blocking changed the completed fixture output")
        if legacy_attempt_sha256 != _file_sha256(legacy_attempt_path):
            raise ContractError("fixture prior attempt changed during continuation")
        if legacy_result_sha256 != _file_sha256(legacy_result_path):
            raise ContractError("fixture prior result changed during continuation")
    historical_after = _verify_historical_evidence_bytes()
    if historical_before != historical_after:
        raise ContractError("original v1 attempt/result changed during disposable smoke")
    if OUTPUT_DIR.exists() or FORBIDDEN_ROOT.exists():
        raise ContractError("smoke created a persistent continuation output or data root")
    return {
        "status": "SMOKE-PASS",
        "fixture_seed": FIXTURE_SEED,
        "profiles": [
            {
                "history_seed": row["history_seed"],
                "profile": row["profile"],
                "config_hash": row["config_hash"],
                "eligible_counts": row["eligible_cohort_counts"],
                "eligible_mix_fraction": row["eligible_mix_fraction"],
                "subtype_margins": row["subtype_quota_margins"],
                "controls": row["controls"],
                "reset_attrition": row["reset_attrition"],
                "exact_allocation": row["exact_allocation"],
                "record_sha256": row["record_sha256"],
            }
            for row in fixture_result["rows"]
        ],
        "failure_diagnostic_probe": failure_probe,
        "persistence_flow": {
            "imported_sealed_fixture_row_sha256": imported_row["record_sha256"],
            "total_fixture_rows": len(fixture_result["rows"]),
            "summary_complete_seed_pair": cast(
                "dict[str, object]", fixture_result["summary"]
            )["complete_seed_pairs"],
            "summary_all_16_fields_null": True,
            "final_result_sha256": fixture_result["result_sha256"],
            "final_attempt_sha256": fixture_result["attempt_sha256"],
            "duplicate_attempt_blocked": True,
            "original_v1_evidence_unchanged": historical_after,
        },
        "provenance_guard_rejections": rejected_guards,
        "generated_data_roots_written": False,
        "persistent_continuation_output_created": False,
        "source_closure_sha256": static["source_closure_sha256"],
        "freeze_digest": freeze_digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--continue", dest="continue_study", action="store_true")
    parser.add_argument(
        "--review-passed",
        action="store_true",
        help="acknowledge independent Task 58 PASS before continuation",
    )
    parser.add_argument(
        "--expected-freeze",
        help="required v2 freeze digest recorded by the reviewed Task 58 evidence",
    )
    args = parser.parse_args()
    if args.continue_study and not args.review_passed:
        parser.error("--continue requires --review-passed after independent review")
    if args.continue_study and not args.expected_freeze:
        parser.error("--continue requires --expected-freeze from the reviewed Task 58 evidence")
    if args.smoke and (args.review_passed or args.expected_freeze is not None):
        parser.error("--review-passed/--expected-freeze are valid only with --continue")
    if args.review_passed and not args.continue_study:
        parser.error("--review-passed is valid only with --continue")
    static = _verify_static_guards()
    freeze_digest = _freeze_digest(
        cast("Mapping[str, str]", static["source_closure"])
    )
    if args.smoke:
        print(json.dumps(run_smoke(static), indent=2, sort_keys=True))
        return 0
    if args.continue_study:
        if args.expected_freeze != freeze_digest:
            raise ContractError(
                "expected freeze digest does not match the reviewed v2 source/doc freeze"
            )
        return run_continuation(
            static,
            output_dir=OUTPUT_DIR,
            prior_attempt_path=ORIGINAL_ATTEMPT_PATH,
            prior_result_path=ORIGINAL_RESULT_PATH,
            expected_attempt_sha256=ORIGINAL_ATTEMPT_SHA256,
            expected_result_sha256=ORIGINAL_RESULT_SHA256,
            expected_row_sha256=ORIGINAL_ROW_SHA256,
            prior_freeze_sha256=ORIGINAL_FREEZE_SHA256,
            seed_order=STUDY_SEEDS,
            freeze_digest=freeze_digest,
        )
    raise ContractError("select exactly one explicit runner mode")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
