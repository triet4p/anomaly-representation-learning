"""Fail-closed runner for the frozen Sprint 21 exploratory whole-history pilot.

The live path is intentionally one-shot and review-gated. ``--smoke`` uses
only in-memory analytical records; ``--dry-run`` checks immutable provenance
and pure configuration objects; neither calls the history generator or
preflight. Only ``--run --review-passed`` may materialize bound histories.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

PROTOCOL_ID = "sprint21-exploratory-pilot-v1"
PROTOCOL_PATH = "experiments/sprint21-exploratory-pilot-v1.md"
BINDING_SCHEMA_ID = "sprint21-pilot-binding-v1"
BASE_COMMIT = "8c15f0204a3e495569b7f143dc109943e8b808de"
TASK3_COMMIT = "f8a8fa6dd3ae1534c8f8db8d552898ffe88314ba"
TASK3_RUNNER_SHA256 = "d1d8239d34944ab54f371555f68d1431f7d4784ba0a67bd4fec49359ba8f7a32"
TASK3_RAW_ROLE_BINDING_SHA256 = "f212ca2fd5a0f60eeb206435b610fa6a8689774ea490db2aa94d44c15a01baf1"
TASK3_ROLE_BINDING_DIGEST = "075868b22c028a981cc63ffe37b8d29720cd324c3e2c7254ce688678758230ba"
TASK3_CONFIG_SHA256 = "1ff67f95428ef29aab05d9f6394a305b75c8c2a9e12431f3cda09b6958596144"
TASK3_UV_LOCK_SHA256 = "4a7866878c81cdd8f5ba283cd8d0a772073bfc75941f73ec761ede5eeb2e239a"
TASK3_METRIC_DIGEST = "b2d6af7505abe459a84f76f5279a6a5c4298e7c142901edd4596fcfe3ddeb88f"
TASK3_MAPPING_DIGEST = "4f769ccc64aac4c6a91fdd356eaa6963afe976ab847a22df23e43f9331418897"
GENERATOR_PROTOCOL_SHA256 = "a1fc09db2e246ed79d0595aec953a7788fd1b47c4981fbe1d92017d944b8d7b6"
GATE_V2_SHA256 = "d294762331ded4fd213f6870e563e32e17e12c632d7cb11a80373478ede34e1c"
HISTORY_UNIT = "independent whole history; scorer seeds are repeated scorer variants, not additional histories"

MODEL_SEEDS = (171701, 171702, 171703)
CHECKPOINT_BYTES = 18_239_321
CHECKPOINTS = {
    171701: ("b0_seed171701_step300.pt", "45f9e151c5d3946804ea531eb2bcace26b1f62e634671f51b8741ac6b411a619"),
    171702: ("b0_seed171702_step300.pt", "64ed2cedec210e4692f60bb4dd3430a57cf94abfcd50551abd97c9265ea785f8"),
    171703: ("b0_seed171703_step300.pt", "9edb2122e357376a9ff1e065d73d5ab7704f8d2005991552332f1ced01b0e906"),
}

# Direct source hashes are independently checked in addition to the immutable
# historical Git commit. This covers the Task 1/3 generator, metric, and model
# input closure; all remaining historical modules are pinned by BASE_COMMIT.
PINNED_SOURCE_HASHES = {
    "src/representation/model.py": "7051f5f76be0bcc8e50d6d5ee2941a57c12a3285406d0ef27883c596aa6600f0",
    "src/representation/data.py": "88ea3becbc8328c5b21518b1f957fe710096cb96ed0cd2416fe6f139439d5eb4",
    "src/representation/masking.py": "c0efd33b24b8d01c067198f0bc04c0c90c1e99bad9a6a1ec006b37ce27c0465f",
    "src/representation/contracts.py": "0b23080fc49f218484a6f7939bb9288f07b607686312e91c3b14d9d6b84464ed",
    "src/representation/layers/normalization.py": "bfb5f68f5352115c2e044c27999cdf3fecab7aa3ce2cd05ce021204f1f9edd8d",
    "src/synth/config.py": "55fed2ef5b253d48c1b0c7f1dd5f7f966cdd03ad250cb86a988ac8de23d5c056",
    "src/synth/health.py": "e5854641fb85eb9b88e05ee9d8e503c69a9432b13177d8c62ad38c3494e7fc6d",
    "src/synth/events.py": "85100f5e0aca47dd2e8b01a08c58f39d32be4f1eab469cdc38e4a4b57768169d",
    "src/synth/chronicle.py": "b5992d6af3c4387d18df746b43be17b4bfc1fac6938e8b44be35b6fb7c65dafd",
    "src/synth/balanced.py": "d62de3422bd51f870d39d18b1a8bb942f9ca73fc0044d8f23cc2d0948bd35a43",
    "src/synth/cli.py": "3c264a33c7e74dd2b1ab2c9cfbb7a7e2e524f1b39a1228f734460edb0d900a37",
    "src/synth/__init__.py": "5c979561a2f01581e2943f23e30061788d10dabc2ec1172cbd9f25275032b168",
    "src/synth/preflight15.py": "19f0300e97d55609c27f51403de9cc0f7a898684df7050424972fc4e4fb21dd1",
    "src/synth/probe15.py": "a08b3d5fb83001e1f5c43f4c56ff536bae85e41d494db289304aeb33a339242b",
    "docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md": GATE_V2_SHA256,
    "experiments/sprint15-benchmark-protocol-v7.md": GENERATOR_PROTOCOL_SHA256,
    "experiments/sprint17_task7_b0.py": "7969f3cacb77b4e9d0476250ba49ffe12f74a50f62157afc861179d9c141e825",
}
METRIC_MEMBER_HASHES = {
    "src/synth/probe15.py": PINNED_SOURCE_HASHES["src/synth/probe15.py"],
    "src/synth/events.py": PINNED_SOURCE_HASHES["src/synth/events.py"],
    "src/representation/attribution_metrics.py": "c14f815c19d8e5791bce09cc0f07986e3a0f4c90c5c70cbf20f1c29f5a8d0488",
    "src/representation/sprint17_ablation.py": "52162995189985a30372e75c12538cfb15f93a2d36fe460551838c1c11e994af",
    "experiments/sprint17_task7_b0.py": PINNED_SOURCE_HASHES["experiments/sprint17_task7_b0.py"],
}

CONFIG_SHA256_BY_DATA_SEED = {
    921000: "b6e2551566ad5ecc71d6574d866733be2713a25af783f2e1c34ad9badac9f42d",
    921001: "fb975aaf1c3e34ebbb6397c1d7b78d130a6c311fd96a99059f2434682843ea17",
    921002: "c24d4f398bddab2248109b001f12317d58381d0681b556e3b5785d27eeb16276",
    921003: "320afdc9d7e92de4fcc90b87f07d3cc8ddf4a4a7ce61dc4ce4d3616ff305eb07",
    921004: "bf009dad901e998914bb8284174e621598c264dc77299bc77708f955c201d44a",
    921005: "b5b7d677911b1e7846006d6fa57f0accaa45c4acbec7dae75160b50446335f5f",
    921006: "1ebb0ac324de245797d71acb1b8c41f11514b93dcc11e6dd117e6e0012e59f6c",
    921007: "52a40d129d52c623f3256ec9fb7a544a311b3517b2640cee289e82eed675d395",
    921008: "fd3e8944ae4fac44508deaa0ab54cdd32d440f31f255fb493c60406595e9349f",
    921009: "107a89f71dc5e0d87a1941e65fa8a843dfd49ea11c6d7d5dc45fd8e2217dbfb0",
    921010: "45c18fc24b99e617b222a2ee5dc773c31d1066f5978417511a8cf1fdf762ed7f",
    921011: "a693c600be5119086eadc6294991d0a3456c00bb67cd64e96c8dcb94689068f0",
    921012: "04423d07e09e057cf3d06cbcc6fbbf64d90084e115e0d70f1040dda3fddc614f",
    921013: "a9b5d8c448c2d1cc08592cc7f5cee78042f825c9cebe074fbbf61de18ebac6e3",
    921014: "5a5c3cbc57be5be1bc1ca1ac2c7d89676e91ea9d4ebbbb99a9426d672a27aeff",
    921015: "9c6f9e513aa21718ea25bd52b89384262769467f78d819477ab83ec7a6a583c5",
    921016: "a3bcac8e96c3fcc79ce39010130b1594303563f6941a2adfe60ce0ec7cc7ba81",
    921017: "42b424b0adf75391595d91b03b2d71bdc46082e13bf3fb4345c715d2b1f34b42",
    921018: "aa40735096b53e9244a96ed320a40e4b7ab9a6c01f6f6db34c32c9602b8b2860",
    921019: "053e9fbbd464ee9474c537fc621f0c43825b05aeeaae1c2024d242635cd9f8af",
    921020: "9e19547c15d4f6a7f19ec3d5ae3ffd91e66f51f9d0b359324c7c53db6cab23b2",
    921021: "1f6e4ad8613c9f4d4b81f3009927dd5a0ec66ae774038656c087e13c29a9d60c",
    921022: "e738f4a502483caac0847705634cba3626640eda57f9e0b69a4d5b879b86f04e",
    921023: "3d5b9cc7a22fa7bc6be2699b902b7afc182a5acf428e98d7242999d9cd60b715",
    921024: "e55b5d214762a0641f7271add1417ad955b58c445a80f457991481051575ebb7",
    921025: "b24e7fcf1a8d9a7ce128b6aaa3590eabb8bb30fac995648c31f596b1e230162a",
    921026: "32287246f58ec210cc253ca8f215c1bdcb06ac217510e57bcbac438af32ea3c7",
    921027: "68473ff02a228e56c5f95550fca2a2fc7dac953d1da9f60fae65a81f0bd876b5",
    921028: "0e8bea0ff843c8a1c98533e3e6c223fa12d3d657a430de87a081da8e14d0ee12",
    921029: "52ad8529a59df8d7bf897b460720c8166d8a5af5b5d00382a91d15dde7753c97",
    921030: "229693cc733b0a32745bbfc7ca8faa5a711e276f1db261616979e95f714a82a0",
    921031: "9a3c0ce241a70112a6f28cc7382e5fbc6b46c676ee3a96f320c02d45dc8ad26d",
}
CONFIG_HASH_BY_DATA_SEED = {
    seed: digest[:12] for seed, digest in CONFIG_SHA256_BY_DATA_SEED.items()
}
CONFIG_TEMPLATE_SHA256 = "3dfecb64967e3e6a1bfe26f8dd72565f553dbda0130dca639d5569c2e42e7dd6"
QUOTA = {
    "a": 16, "a1": 8, "a2": 8, "controls": 48, "p": 24,
    "p1": 12, "p2": 12, "program_cap": 0.6, "programs": 2,
    "robot_days": 240, "robot_neg_cap": 0.4, "robot_pos_cap": 0.35,
    "robots_neg": 6, "robots_pos": 6, "spacing_s": 1209600.0,
    "w": 24, "w1": 12, "w2": 12,
}
QUOTA_SHA256 = "0ebc9ec0874abcf6f0b60d1591b383b86164128db669966e7473b3c4277b359f"

EXPECTED_EXCLUDED_RANGES = (
    (742, 797), (910, 933),
    (1000, 1017), (1100, 1117), (1200, 1217), (1300, 1317),
    (1400, 1417), (1500, 1517), (1600, 1617),
    (2700, 2715), (2800, 2815), (31800, 31847),
    (918000, 918015), (918100, 918107),
)
EXPECTED_EXCLUDED_SINGLE_SEEDS = (0, 9000, 9001, 93058, 93061)
EXPECTED_NONDATA_MODEL_SEEDS = ((171701, 171703), (181801, 181803))
EXPECTED_SEED_EVIDENCE = {
    "experiments/sprint17-role-binding-v3.json": TASK3_RAW_ROLE_BINDING_SHA256,
    "experiments/sprint18-role-binding-v1.json": "493c5d6a7a9b807f1fe915f00bb917be45a11183512581606eee34ee678bb228",
    "experiments/sprint18-role-binding-c2.json": "afe28bc4453a909c5ccba2872e6f37813d0c95f40e5e8a113fd01d3c6c29f14a",
    "experiments/sprint18-role-binding-c2-v2.json": "45818410f8d483d66a5523fd4ec38169bed9dc45d3a82f16979c05f9ab91dd92",
    "experiments/sprint18-role-binding-c3-v1.json": "6d87b9bba4e7e76e25917e6b00a13a52dc886e8ea8052ef6e5c88cf01a863ddd",
    "artifacts/sprint-18/task-4.md": "5665c2fb1953bff8ddde4e987d936c2f83b84cd057c4267f359887bf1f918d4e",
    "artifacts/sprint-18/task-42.md": "e6754347ea12cae83849fc2233480cb47195d7034564f2eb5d29fae799280dc9",
    "artifacts/sprint-18/task-43.md": "d98c7d6779d97f032d105b853ed19b0a9e1c8d04e87d7803d9bf29f85153f985",
    "artifacts/sprint-18/task-51.md": "cebd49d279292c47a109e5430cfe02fb5e452d5575f572663192b58a8a92087a",
    "artifacts/sprint-18/task-58.md": "ab8406cef6f3dd074a1e08bac88720a5222c27e00f4fd7e45ff63f7966134370",
    "artifacts/sprint-18/task-60.md": "605bfebf2967245fa569da4a19e7da4fe2b6c08bbe3a95d97384d86f46e0db09",
    "artifacts/sprint-18/task-61.md": "0c8c7e026eac293fbca04ae089ab2df72568f39343b133824f5b3fc7f5293e72",
    "artifacts/sprint-18/task-63.md": "d7f705707b6a57dfa43b484ca6140a325af8ce012755746f5d0c50fbfaa358f9",
    "artifacts/sprint-18/task-65.md": "5973c3cc9e72b4c8cf81cbbbb7f0d3af88bdd501856240cd981f9bf8b31a4b7c",
    "artifacts/sprint-18/isolated-feasibility-v3/attempt.json": "30ff8f407e9fe03335eeb332abb2882297ddd05458119215f6bef0f5abd90a13",
    "artifacts/sprint-18/control-eligibility-diagnostic-v1/attempt.json": "581789103c1b7541e679c10c327b2fc6aa08960893601399c08839b6bededa5d",
    "experiments/sprint18_task58_feasibility-v3.py": "f74c1e182b7bb28c40808edf35f9505d5f2ed564d9f5f6ff9289ad14d174988c",
    "experiments/sprint18_control_eligibility_diagnostic_v1.py": "881e29c638954b35849358615676655d0d269214e3fa7a5913ad19c4fcd5b961",
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
    "python": "3.12.13",
    "torch": "2.14.0+cu130",
    "cuda_available": True,
    "device_name": "NVIDIA GeForce RTX 4060 Ti",
    "device_total_mib": 16380,
}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260202


class PilotError(RuntimeError):
    """Fail-closed pilot error with a stable evidence disposition."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PilotError(code, message)


def fixed_roster() -> list[dict[str, Any]]:
    return [
        {
            "history_id": f"H-S21-PILOT-{index:02d}",
            "data_seed": seed,
            "relative_root": f"histories/H-S21-PILOT-{index:02d}",
            "config_hash": CONFIG_HASH_BY_DATA_SEED[seed],
            "config_sha256": CONFIG_SHA256_BY_DATA_SEED[seed],
        }
        for index, seed in enumerate(range(921000, 921032), start=1)
    ]


def _evidence_rows() -> list[dict[str, str]]:
    return [
        {"path": path, "sha256": digest}
        for path, digest in EXPECTED_SEED_EVIDENCE.items()
    ]


def _expected_exclusions() -> dict[str, Any]:
    return {
        "history_data_seed_ranges": [
            {"first": first, "last": last}
            for first, last in EXPECTED_EXCLUDED_RANGES
        ],
        "single_data_seeds": list(EXPECTED_EXCLUDED_SINGLE_SEEDS),
        "nondata_model_seed_ranges": [
            {"first": first, "last": last}
            for first, last in EXPECTED_NONDATA_MODEL_SEEDS
        ],
        "source_records": _evidence_rows(),
        "scope": "Complete cited repository role-binding and preserved one-shot evidence catalog; no history/root search or Sealed-root access was performed.",
    }


def validate_roster(records: list[dict[str, Any]]) -> None:
    expected = fixed_roster()
    _expect(records == expected, "BINDING_MISMATCH", "bound 32-history roster differs from the fixed reviewed block")
    ids = [record["history_id"] for record in records]
    seeds = [record["data_seed"] for record in records]
    roots = [record["relative_root"] for record in records]
    _expect(len(set(ids)) == 32 and len(set(seeds)) == 32 and len(set(roots)) == 32,
            "BINDING_MISMATCH", "history IDs, data seeds, and roots must each be unique")
    excluded = [seed for first, last in EXPECTED_EXCLUDED_RANGES for seed in range(first, last + 1)]
    excluded.extend(EXPECTED_EXCLUDED_SINGLE_SEEDS)
    excluded.extend(seed for first, last in EXPECTED_NONDATA_MODEL_SEEDS for seed in range(first, last + 1))
    overlap = sorted(set(seeds).intersection(excluded))
    _expect(not overlap, "FRESHNESS_COLLISION", f"pilot data seeds collide with cited prior namespaces: {overlap}")


def validate_binding(binding: dict[str, Any], expected_digest: str | None = None) -> str:
    _expect(isinstance(binding, dict), "BINDING_MISMATCH", "binding JSON must be an object")
    declared = binding.get("binding_sha256")
    payload = dict(binding)
    payload.pop("binding_sha256", None)
    computed = canonical_sha256(payload)
    _expect(declared == computed, "BINDING_MISMATCH", f"binding digest {computed} != declared {declared!r}")
    if expected_digest is not None:
        _expect(expected_digest == computed, "BINDING_MISMATCH", "command-line expected binding digest does not match")
    _expect(binding.get("schema_id") == BINDING_SCHEMA_ID, "BINDING_MISMATCH", "unexpected binding schema")
    _expect(binding.get("protocol_id") == PROTOCOL_ID, "BINDING_MISMATCH", "unexpected protocol identifier")
    _expect(binding.get("status") == "FROZEN_PENDING_PRECONTACT_REVIEW", "BINDING_MISMATCH", "binding must remain pending independent pre-contact review")
    _expect(binding.get("contact_authorized") is False, "BINDING_MISMATCH", "Task 4 binding grants no contact authority")
    _expect(binding.get("pilot_claim") == "research-only, descriptive, simulator-conditional", "BINDING_MISMATCH", "pilot claim boundary changed")
    _expect(binding.get("history_unit") == HISTORY_UNIT, "BINDING_MISMATCH", "independent unit must be a whole history")
    _expect(binding.get("roster_size") == 32 and binding.get("sample_size_choice") == "H=32 is the fixed prospective history sample size only; it is not an H=32,m=2 or other miss allowance.",
            "BINDING_MISMATCH", "history count decision or no-allowance boundary changed")
    validate_roster(binding.get("roster", []))
    _expect(binding.get("freshness_exclusions") == _expected_exclusions(), "FRESHNESS_PROVENANCE_MISMATCH", "cited seed exclusion catalog or evidence digest differs")
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
        "quota_sha256": QUOTA_SHA256,
        "config_template_sha256": CONFIG_TEMPLATE_SHA256,
        "config_sha256_by_data_seed": {str(seed): digest for seed, digest in CONFIG_SHA256_BY_DATA_SEED.items()},
        "config_hash_by_data_seed": {str(seed): digest for seed, digest in CONFIG_HASH_BY_DATA_SEED.items()},
        "new_construction_quota": False,
    }, "BINDING_MISMATCH", "generator/profile/config/existing quota construction contract changed")
    _expect(binding.get("hard_floors") == HARD_FLOORS and binding.get("design_margins_diagnostic_only") == DESIGN_MARGINS,
            "BINDING_MISMATCH", "hard floors or non-gating design-margin diagnostics changed")
    _expect(binding.get("scorer") == _expected_scorer_binding(), "BINDING_MISMATCH", "scorer/checkpoint/mask contract changed")
    _expect(binding.get("runtime") == EXPECTED_RUNTIME, "BINDING_MISMATCH", "runtime contract changed")
    _expect(binding.get("source") == _expected_source_binding(binding), "BINDING_MISMATCH", "source/commit/hash binding differs")
    _expect(binding.get("output") == _expected_output_binding(), "BINDING_MISMATCH", "one-shot output path contract changed")
    _expect(binding.get("analysis") == _expected_analysis_contract(), "BINDING_MISMATCH", "primary/secondary analysis contract changed")
    return computed


def _expected_scorer_binding() -> dict[str, Any]:
    return {
        "historical_base_commit": BASE_COMMIT,
        "task3_runner_commit": TASK3_COMMIT,
        "task3_runner_sha256": TASK3_RUNNER_SHA256,
        "b0_config_sha256": TASK3_CONFIG_SHA256,
        "metric_code_sha256": TASK3_METRIC_DIGEST,
        "metric_member_sha256": METRIC_MEMBER_HASHES,
        "conditioning_mapping_sha256": TASK3_MAPPING_DIGEST,
        "uv_lock_sha256": TASK3_UV_LOCK_SHA256,
        "model_seeds": list(MODEL_SEEDS),
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "checkpoints": {
            str(seed): {"filename": filename, "sha256": digest}
            for seed, (filename, digest) in CHECKPOINTS.items()
        },
        "file_masking": "sha256(UTF-8(global_file_id)); unsigned big-endian first 4 bytes mod 2**31",
        "score_branches": ["S_pred", "S_pop"],
        "model_inputs": "signal x plus explicit source-derived robot_idx/program_idx; labels/health/events/split/future metadata are not passed; NORMAL and seed=0 are inert required FileSample sentinels",
        "bank": {"source": "checkpoint-embedded historical Fit-only bank", "rows": 5040, "dimensions": 128, "k": 5, "refit": False},
        "inference_only": True,
    }


def _expected_source_binding(binding: dict[str, Any]) -> dict[str, Any]:
    source = binding.get("source", {})
    runner_commit = source.get("task4_runner_source_commit")
    runner_sha = source.get("task4_runner_sha256")
    protocol_sha = source.get("protocol_sha256")
    _expect(isinstance(runner_commit, str) and bool(re.fullmatch(r"[0-9a-f]{40}", runner_commit)),
            "BINDING_MISMATCH", "Task4 runner source commit must be a full Git SHA")
    _expect(isinstance(runner_sha, str) and bool(re.fullmatch(r"[0-9a-f]{64}", runner_sha)),
            "BINDING_MISMATCH", "Task4 runner file hash must be a full SHA-256")
    _expect(isinstance(protocol_sha, str) and bool(re.fullmatch(r"[0-9a-f]{64}", protocol_sha)),
            "BINDING_MISMATCH", "protocol file hash must be a full SHA-256")
    return {
        "historical_base_commit": BASE_COMMIT,
        "task3_runner_commit": TASK3_COMMIT,
        "task3_runner_sha256": TASK3_RUNNER_SHA256,
        "task4_runner_source_commit": runner_commit,
        "task4_runner_sha256": runner_sha,
        "protocol_path": PROTOCOL_PATH,
        "protocol_sha256": protocol_sha,
        "pinned_source_sha256": PINNED_SOURCE_HASHES,
        "metric_member_sha256": METRIC_MEMBER_HASHES,
        "metric_code_sha256": TASK3_METRIC_DIGEST,
        "uv_lock_sha256": TASK3_UV_LOCK_SHA256,
        "task3_role_binding_raw_sha256": TASK3_RAW_ROLE_BINDING_SHA256,
        "task3_role_binding_canonical_sha256": TASK3_ROLE_BINDING_DIGEST,
    }


def _expected_output_binding() -> dict[str, Any]:
    return {
        "run_root": "/tmp/sprint21-exploratory-pilot-v1",
        "history_roots": "histories/H-S21-PILOT-01..H-S21-PILOT-32",
        "one_shot": True,
        "resume": False,
        "retry": False,
        "replacement": False,
        "omission": False,
        "seed_search": False,
        "post_failure_continue": False,
        "preflight": False,
        "confirmation_or_sealed_contact": False,
        "files": [
            "attempt.json", "pilot-ledger.jsonl", "history-evidence/H-S21-PILOT-XX.json",
            "scores/model-171701/H-S21-PILOT-XX.jsonl",
            "scores/model-171702/H-S21-PILOT-XX.jsonl",
            "scores/model-171703/H-S21-PILOT-XX.jsonl", "summary.json",
        ],
    }


def _expected_analysis_contract() -> dict[str, Any]:
    return {
        "primary": "within-history tie-aware AUROC for P+W event windows versus same-history eligible control windows; max file score per window; S_pred and S_pop separate",
        "secondary": ["P AUROC", "W AUROC", "A AUROC", "support counts and reasons", "per-history score-independent support", "history variance and support-margin strata"],
        "support_selection_score_independent": True,
        "full_roster_required": 32,
        "history_macro": "unweighted mean across all 32 predeclared histories only",
        "bootstrap": {"resampling_unit": "whole history", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "lcb_percentile": 2.5, "same_draws_for_branches": True, "interpretation": "descriptive uncertainty summary only"},
        "hard_fail_or_missing_or_nonfinite_or_branch_support_mismatch": "per-history endpoint UNCOMPUTABLE; any such history makes full-roster macro/bootstrap UNCOMPUTABLE; never average a subset",
        "margin_misses": "retain the history if every hard floor passes; report as descriptive support strata only; no tolerance or miss allowance",
        "sample_variance": "unbiased sample variance across complete independent history endpoint values; scorer seeds remain separate repeated scorer variants",
        "thresholds_or_calibration": False,
        "training_or_refitting": False,
    }


def read_binding(path: Path, expected_digest: str | None) -> tuple[dict[str, Any], str]:
    try:
        binding = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError("BINDING_MISMATCH", f"cannot read binding JSON: {exc}") from exc
    if not isinstance(binding, dict):
        raise PilotError("BINDING_MISMATCH", "binding JSON is not an object")
    return binding, validate_binding(binding, expected_digest)


def validate_runtime_roster(actual_ids: list[str], expected_ids: list[str]) -> None:
    _expect(len(expected_ids) == len(set(expected_ids)), "ROSTER_OR_SCORE_INCOMPLETE", "duplicate IDs in expected roster")
    _expect(len(actual_ids) == len(expected_ids) and len(actual_ids) == len(set(actual_ids)),
            "ROSTER_OR_SCORE_INCOMPLETE", "input file roster has missing, extra, or duplicate IDs")
    _expect(set(actual_ids) == set(expected_ids), "ROSTER_OR_SCORE_INCOMPLETE", "input file IDs differ from the predeclared roster")


def validate_score_roster(
    expected_ids: list[str], scores: dict[str, dict[str, float]],
    support_keys: dict[str, list[str] | set[str]],
) -> None:
    _expect(len(expected_ids) == len(set(expected_ids)), "ROSTER_OR_SCORE_INCOMPLETE", "duplicate declared file ID")
    _expect(set(scores) == set(expected_ids) and len(scores) == len(expected_ids),
            "ROSTER_OR_SCORE_INCOMPLETE", "scored file roster differs from expected full file roster")
    _expect(set(support_keys) == {"S_pred", "S_pop"}, "ROSTER_OR_SCORE_INCOMPLETE", "support must be supplied for both branches")
    normalized: list[list[str]] = []
    for branch in ("S_pred", "S_pop"):
        values = sorted(support_keys[branch])
        _expect(len(values) == len(set(values)), "ROSTER_OR_SCORE_INCOMPLETE", f"duplicate {branch} support key")
        normalized.append(values)
    _expect(normalized[0] == normalized[1], "ROSTER_OR_SCORE_INCOMPLETE", "score-independent support differs between branches")
    for file_id in expected_ids:
        branches = scores[file_id]
        _expect(set(branches) == {"S_pred", "S_pop"}, "ROSTER_OR_SCORE_INCOMPLETE", f"{file_id}: missing or extra score branch")
        for branch in ("S_pred", "S_pop"):
            value = branches[branch]
            _expect(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0,
                    "ROSTER_OR_SCORE_INCOMPLETE", f"{file_id}/{branch}: score must be finite and nonnegative")


def history_window_scores(
    file_scores: dict[str, dict[str, float]],
    positive_windows: dict[str, list[dict[str, Any]]],
    control_windows: list[dict[str, Any]],
    events_module: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {"branches": {}, "support_keys": {}}
    for branch in ("S_pred", "S_pop"):
        scores = {file_id: float(record[branch]) for file_id, record in file_scores.items()}
        pos_by_cohort: dict[str, list[float]] = {}
        branch_keys: list[str] = []
        for cohort in ("P", "W", "A"):
            windows = positive_windows[cohort]
            pos_by_cohort[cohort] = [
                float(events_module.window_score(window["members"], scores))
                for window in windows
            ]
            branch_keys.extend(window["key"] for window in windows)
        controls = [
            float(events_module.window_score(window["members"], scores))
            for window in control_windows
        ]
        branch_keys.extend(window["key"] for window in control_windows)
        try:
            auc = {
                "P+W": float(events_module.roc_auc_tie_aware(pos_by_cohort["P"] + pos_by_cohort["W"], controls)),
                "P": float(events_module.roc_auc_tie_aware(pos_by_cohort["P"], controls)),
                "W": float(events_module.roc_auc_tie_aware(pos_by_cohort["W"], controls)),
                "A": float(events_module.roc_auc_tie_aware(pos_by_cohort["A"], controls)),
            }
        except Exception as exc:
            raise PilotError("METRIC_UNCOMPUTABLE", f"tie-aware AUROC could not be computed: {exc}") from exc
        _expect(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in auc.values()),
                "METRIC_UNCOMPUTABLE", "AUROC output is non-finite or outside [0,1]")
        result["branches"][branch] = {
            "auc": auc,
            "positive_window_scores": pos_by_cohort,
            "control_window_scores": controls,
        }
        result["support_keys"][branch] = sorted(branch_keys)
    _expect(result["support_keys"]["S_pred"] == result["support_keys"]["S_pop"],
            "ROSTER_OR_SCORE_INCOMPLETE", "branch support keys differ")
    return result


def aggregate_full_roster(
    history_ids: list[str], history_records: list[dict[str, Any]],
    *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize only the complete H-roster; hard-fail histories never shrink H."""
    import numpy as np

    _expect(len(history_ids) == len(set(history_ids)), "ROSTER_OR_SCORE_INCOMPLETE", "duplicate history ID in declared H")
    _expect(len(history_records) == len(history_ids), "ROSTER_OR_SCORE_INCOMPLETE", "history result count differs from declared H")
    by_id = {record.get("history_id"): record for record in history_records}
    _expect(len(by_id) == len(history_records) and set(by_id) == set(history_ids),
            "ROSTER_OR_SCORE_INCOMPLETE", "history result roster is missing, extra, or duplicated")
    ordered = [by_id[history_id] for history_id in history_ids]
    full_valid = all(
        record.get("hard_status") == "PASS"
        and record.get("metric_status") == "COMPUTABLE"
        and all(
            isinstance(record.get("primary_auc", {}).get(branch), (int, float))
            and not isinstance(record.get("primary_auc", {}).get(branch), bool)
            and math.isfinite(float(record["primary_auc"][branch]))
            and all(
                isinstance(record.get("secondary_auc", {}).get(branch, {}).get(cohort), (int, float))
                and not isinstance(record.get("secondary_auc", {}).get(branch, {}).get(cohort), bool)
                and math.isfinite(float(record["secondary_auc"][branch][cohort]))
                for cohort in ("P", "W", "A")
            )
            for branch in ("S_pred", "S_pop")
        )
        for record in ordered
    )
    margins = ("P_ge_13", "W_ge_13", "A_ge_10", "positive_total_ge_38", "controls_ge_32", "robot_days_ge_188")
    support_strata: dict[str, Any] = {}
    for margin in margins:
        support_strata[margin] = {}
        for branch in ("S_pred", "S_pop"):
            support_strata[margin][branch] = {}
            for margin_met in (True, False):
                selected = [
                    record for record in ordered
                    if record.get("hard_status") == "PASS"
                    and record.get("metric_status") == "COMPUTABLE"
                    and isinstance(record.get("primary_auc", {}).get(branch), (int, float))
                    and record.get("design_margin_flags", {}).get(margin) is margin_met
                ]
                values = np.asarray([record["primary_auc"][branch] for record in selected], dtype=np.float64)
                support_strata[margin][branch]["margin_met" if margin_met else "margin_missed"] = {
                    "margin_met": margin_met,
                    "n_hard_pass_histories_with_auc": len(selected),
                    "history_ids": [record["history_id"] for record in selected],
                    "descriptive_mean_within_margin_stratum": float(values.mean()) if len(values) else None,
                    "descriptive_sample_variance_within_margin_stratum": float(values.var(ddof=1)) if len(values) > 1 else None,
                    "diagnostic_only_not_endpoint_or_selection_rule": True,
                }
    output: dict[str, Any] = {
        "declared_history_ids": list(history_ids),
        "history_unit": HISTORY_UNIT,
        "full_roster_status": "COMPUTABLE" if full_valid else "UNCOMPUTABLE_FULL_ROSTER",
        "hard_pass_count": sum(record.get("hard_status") == "PASS" for record in ordered),
        "hard_fail_history_ids": [record["history_id"] for record in ordered if record.get("hard_status") != "PASS"],
        "metric_uncomputable_history_ids": [
            record["history_id"] for record in ordered
            if record.get("hard_status") == "PASS" and record.get("metric_status") != "COMPUTABLE"
        ],
        "primary": {},
        "secondary_cohort_macros": {},
        "support_dependence_strata": support_strata,
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "percentile": 2.5,
            "resampling_unit": "whole history",
            "interpretation": "descriptive uncertainty summary only; H=32 is a fixed sample-size choice, not a precision claim",
        },
    }
    if not full_valid:
        for branch in ("S_pred", "S_pop"):
            output["primary"][branch] = {
                "status": "UNCOMPUTABLE_FULL_ROSTER",
                "macro": None,
                "sample_variance": None,
                "bootstrap_lcb95": None,
                "per_history": [
                    {"history_id": record["history_id"], "value": record.get("primary_auc", {}).get(branch),
                     "hard_status": record.get("hard_status"), "metric_status": record.get("metric_status", "UNCOMPUTABLE")}
                    for record in ordered
                ],
            }
        for cohort in ("P", "W", "A"):
            output["secondary_cohort_macros"][cohort] = {branch: None for branch in ("S_pred", "S_pop")}
        return output

    draws = np.random.default_rng(seed).integers(0, len(ordered), size=(replicates, len(ordered)))
    for branch in ("S_pred", "S_pop"):
        values = np.asarray([record["primary_auc"][branch] for record in ordered], dtype=np.float64)
        replicate_macros = values[draws].mean(axis=1)
        output["primary"][branch] = {
            "status": "COMPUTABLE_FULL_ROSTER",
            "macro": float(values.mean()),
            "sample_variance": float(values.var(ddof=1)),
            "bootstrap_lcb95": float(np.quantile(replicate_macros, 0.025)),
            "per_history": [
                {"history_id": record["history_id"], "value": float(record["primary_auc"][branch]), "status": "PASS"}
                for record in ordered
            ],
        }
    for cohort in ("P", "W", "A"):
        output["secondary_cohort_macros"][cohort] = {}
        for branch in ("S_pred", "S_pop"):
            values = [record.get("secondary_auc", {}).get(branch, {}).get(cohort) for record in ordered]
            if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
                output["secondary_cohort_macros"][cohort][branch] = None
            else:
                output["secondary_cohort_macros"][cohort][branch] = float(np.mean(np.asarray(values, dtype=np.float64)))
    return output
def _scientific_support_stop(
    history_ids: list[str], attempted_records: list[dict[str, Any]],
    unattempted_history_ids: list[str],
) -> dict[str, Any]:
    _expect(len(history_ids) == 32 and len(history_ids) == len(set(history_ids)),
            "ROSTER_OR_SCORE_INCOMPLETE", "support-stop summary requires the exact unique 32-history roster")
    records_by_id = {record.get("history_id"): record for record in attempted_records}
    _expect(len(records_by_id) == len(attempted_records), "ROSTER_OR_SCORE_INCOMPLETE", "duplicate attempted history in support-stop summary")
    attempted_ids = [record.get("history_id") for record in attempted_records]
    _expect(attempted_ids == history_ids[:len(attempted_ids)],
            "ROSTER_OR_SCORE_INCOMPLETE", "attempted histories are not a prefix of the frozen roster")
    failure_positions = [
        index for index, history_id in enumerate(attempted_ids)
        if records_by_id[history_id].get("hard_status") != "PASS"
    ]
    _expect(bool(failure_positions), "ROSTER_OR_SCORE_INCOMPLETE", "support-stop summary requires an observed hard failure")
    first_failure = failure_positions[0]
    _expect(len(attempted_ids) == first_failure + 1
            and unattempted_history_ids == history_ids[first_failure + 1:],
            "ROSTER_OR_SCORE_INCOMPLETE", "the one-shot did not stop immediately after its first hard failure")
    per_history = []
    for history_id in history_ids:
        if history_id in records_by_id:
            status = (
                "HARD_SUPPORT_FAIL" if records_by_id[history_id].get("hard_status") != "PASS"
                else "SUPPORT_PASS_NOT_SCORED_AFTER_STOP"
            )
        else:
            status = "NOT_ATTEMPTED_AFTER_STOP"
        per_history.append({"history_id": history_id, "status": status, "value": None})
    results: dict[str, Any] = {}
    for model_seed in MODEL_SEEDS:
        results[str(model_seed)] = {
            "full_roster_status": "UNCOMPUTABLE_FULL_ROSTER",
            "hard_failure_history_ids": [
                history_id for history_id in attempted_ids
                if records_by_id[history_id].get("hard_status") != "PASS"
            ],
            "unattempted_history_ids": list(unattempted_history_ids),
            "primary": {
                branch: {
                    "status": "UNCOMPUTABLE_FULL_ROSTER",
                    "macro": None, "sample_variance": None, "bootstrap_lcb95": None,
                    "per_history": per_history,
                }
                for branch in ("S_pred", "S_pop")
            },
            "secondary_cohort_macros": {
                cohort: {branch: None for branch in ("S_pred", "S_pop")}
                for cohort in ("P", "W", "A")
            },
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                "computed": False, "resampling_unit": "whole history",
                "interpretation": "not computed: one-shot stopped on the first hard support failure",
            },
        }
    return results
def _json_line(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, default=str, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(payload, target, sort_keys=True, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
def _rewrite_json(path: Path, payload: dict[str, Any]) -> None:
    _expect(path.is_file() and not path.is_symlink(), "OUTPUT_PATH_MISMATCH", f"expected evidence file is absent or not regular: {path}")
    with path.open("r+", encoding="utf-8", newline="\n") as target:
        target.seek(0)
        json.dump(payload, target, sort_keys=True, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        target.write("\n")
        target.truncate()
        target.flush()
        os.fsync(target.fileno())


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], check=True,
            capture_output=True, text=True, encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotError("SOURCE_OR_ISOLATION_MISMATCH", f"git {' '.join(args)} failed: {exc}") from exc
    return result.stdout.strip()


def _git_show_sha(root: Path, commit: str, relative_path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative_path}"],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotError("SOURCE_OR_ISOLATION_MISMATCH", f"cannot read committed {relative_path} at {commit}: {exc}") from exc
    return hashlib.sha256(result.stdout).hexdigest()


def _verify_source_commit(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    source = binding["source"]
    _expect(_git(root, "rev-parse", "HEAD") == BASE_COMMIT, "SOURCE_OR_ISOLATION_MISMATCH", "historical worktree HEAD differs from frozen B0 commit")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    changed = [line[3:] for line in status.splitlines() if line]
    allowed = sorted(["experiments/sprint21_disposable_scorer_proof.py", "experiments/sprint21_pilot_runner.py"])
    _expect(sorted(changed) == allowed, "SOURCE_OR_ISOLATION_MISMATCH", f"historical worktree has unexpected changes: {changed}")
    runner_commit = source["task4_runner_source_commit"]
    _expect(bool(re.fullmatch(r"[0-9a-f]{40}", runner_commit)), "SOURCE_OR_ISOLATION_MISMATCH", "Task4 runner source commit is not a full Git SHA")
    _expect(_git_show_sha(root, runner_commit, "experiments/sprint21_pilot_runner.py") == source["task4_runner_sha256"],
            "SOURCE_OR_ISOLATION_MISMATCH", "committed Task4 runner bytes differ from the binding")
    _expect(_git_show_sha(root, runner_commit, PROTOCOL_PATH) == source["protocol_sha256"],
            "SOURCE_OR_ISOLATION_MISMATCH", "committed protocol bytes differ from the binding")
    _expect(_git_show_sha(root, TASK3_COMMIT, "experiments/sprint21_disposable_scorer_proof.py") == TASK3_RUNNER_SHA256,
            "SOURCE_OR_ISOLATION_MISMATCH", "committed Task3 proof wrapper bytes differ from the accepted proof")
    _expect(sha256_file(root / "experiments/sprint21_disposable_scorer_proof.py") == TASK3_RUNNER_SHA256,
            "SOURCE_OR_ISOLATION_MISMATCH", "historical worktree Task3 wrapper bytes differ from its exact commit")
    _expect(sha256_file(root / "experiments/sprint21_pilot_runner.py") == source["task4_runner_sha256"],
            "SOURCE_OR_ISOLATION_MISMATCH", "historical worktree Task4 runner bytes differ from the frozen source commit")
    _expect(_git(root, "rev-parse", f"{runner_commit}^{{commit}}") == runner_commit,
            "SOURCE_OR_ISOLATION_MISMATCH", "Task4 source commit object is not available")
    actual_hashes: dict[str, str] = {}
    for relative_path, expected in PINNED_SOURCE_HASHES.items():
        path = root / relative_path
        _expect(path.is_file(), "SOURCE_OR_ISOLATION_MISMATCH", f"missing pinned source file {relative_path}")
        digest = sha256_file(path)
        actual_hashes[relative_path] = digest
        _expect(digest == expected, "SOURCE_OR_ISOLATION_MISMATCH", f"pinned source hash mismatch for {relative_path}")
    evidence_hashes: dict[str, str] = {}
    for relative_path, expected in EXPECTED_SEED_EVIDENCE.items():
        path = root / relative_path
        _expect(path.is_file(), "FRESHNESS_PROVENANCE_MISMATCH", f"missing cited seed evidence file {relative_path}")
        digest = sha256_file(path)
        _expect(digest == expected, "FRESHNESS_PROVENANCE_MISMATCH", f"cited seed evidence hash mismatch for {relative_path}")
        evidence_hashes[relative_path] = digest
    lock_digest = sha256_file(root / "uv.lock")
    _expect(lock_digest == TASK3_UV_LOCK_SHA256, "SOURCE_OR_ISOLATION_MISMATCH", "pinned uv.lock hash differs")
    return {"historical_head": BASE_COMMIT, "task4_runner_commit": runner_commit,
            "task4_runner_sha256": source["task4_runner_sha256"], "protocol_sha256": source["protocol_sha256"],
            "task3_runner_commit": TASK3_COMMIT, "task3_runner_sha256": TASK3_RUNNER_SHA256,
            "pinned_source_hashes": actual_hashes, "seed_evidence_hashes": evidence_hashes,
            "uv_lock_sha256": lock_digest, "allowed_historical_worktree_untracked_files": allowed}


def _verify_pure_profile_configs(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    src = str((root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    chronicle = importlib.import_module("synth.chronicle")
    config_module = importlib.import_module("synth.config")
    balanced = importlib.import_module("synth.balanced")
    _expect(Path(chronicle.__file__).resolve().is_relative_to(Path(src)), "SOURCE_OR_ISOLATION_MISMATCH", "history configuration imported outside pinned source")
    quota = dataclasses.asdict(balanced.DEFAULT_QUOTA)
    _expect(quota == QUOTA and hashlib.sha256(json.dumps(quota, sort_keys=True, default=str).encode()).hexdigest() == QUOTA_SHA256,
            "CONFIG_MISMATCH", "existing default construction quota changed")
    cfgs = {seed: chronicle.sprint15_v7_history_config(seed=seed) for seed in range(921000, 921032)}
    _expect(set(cfgs) == set(CONFIG_SHA256_BY_DATA_SEED), "CONFIG_MISMATCH", "static config factory did not return the fixed seed block")
    actual: dict[str, dict[str, str]] = {}
    normalized: dict[str, Any] | None = None
    for seed, cfg in cfgs.items():
        resolved = dataclasses.asdict(cfg)
        serialized = json.dumps(resolved, sort_keys=True, default=str).encode()
        full_digest = hashlib.sha256(serialized).hexdigest()
        _expect(full_digest == CONFIG_SHA256_BY_DATA_SEED[seed] and cfg.hash() == CONFIG_HASH_BY_DATA_SEED[seed],
                "CONFIG_MISMATCH", f"data seed {seed} static profile config differs from binding")
        seed_values = {
            "factory": cfg.factory.seed,
            "scheduler": cfg.scheduler.seed,
            "health": cfg.health.seed,
            "signal": cfg.signal.seed,
            "temporal": cfg.temporal.seed,
        }
        _expect(set(seed_values.values()) == {seed}, "CONFIG_MISMATCH", f"data seed {seed} is not bound into all five config sub-seeds")
        actual[str(seed)] = {"config_hash": cfg.hash(), "config_sha256": full_digest}
        if normalized is None:
            normalized = copy.deepcopy(resolved)
            for section in ("factory", "scheduler", "health", "signal", "temporal"):
                normalized[section]["seed"] = "DATA_SEED"
    _expect(hashlib.sha256(json.dumps(normalized, sort_keys=True, default=str).encode()).hexdigest() == CONFIG_TEMPLATE_SHA256,
            "CONFIG_MISMATCH", "normalized v7 history configuration template differs")
    _expect(config_module.GENERATOR_VERSION == binding["construction"]["generator_version"],
            "CONFIG_MISMATCH", "generator version differs from binding")
    return {"generator_version": config_module.GENERATOR_VERSION,
            "config_template_sha256": CONFIG_TEMPLATE_SHA256,
            "config_seed_fingerprints": actual, "quota_sha256": QUOTA_SHA256}


def _verify_runtime_and_assets(root: Path, checkpoint_root: Path) -> dict[str, Any]:
    experiments_dir = str((root / "experiments").resolve())
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
    proof = importlib.import_module("sprint21_disposable_scorer_proof")
    _expect(Path(proof.__file__).resolve() == (root / "experiments/sprint21_disposable_scorer_proof.py").resolve(),
            "SOURCE_OR_ISOLATION_MISMATCH", "Task3 proof module imported from the wrong path")
    source_gate = proof.assert_runtime()
    import torch

    name = torch.cuda.get_device_name(0)
    total_mib = int(round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)))
    runtime = {
        "python": source_gate["python"], "torch": source_gate["torch"],
        "cuda_available": torch.cuda.is_available(), "device_name": name,
        "device_total_mib": total_mib,
    }
    _expect(runtime == EXPECTED_RUNTIME, "RUNTIME_MISMATCH", f"runtime/device differs from the frozen environment: {runtime}")
    mapping, mapping_digest = proof.derive_conditioning_mapping(root)
    _expect(mapping_digest == TASK3_MAPPING_DIGEST, "CONDITIONING_MISMATCH", "source-derived conditioning map digest differs")
    config_map, config_digest = proof.load_b0_config_map(root)
    _expect(config_digest == TASK3_CONFIG_SHA256, "PROVENANCE_MISMATCH", "full three-seed B0 config digest differs")
    metric = proof.verify_metric_code(root)
    _expect(metric["digest"] == TASK3_METRIC_DIGEST and metric["files"] == METRIC_MEMBER_HASHES,
            "SOURCE_OR_ISOLATION_MISMATCH", "historical metric code digest or members differ")
    role_binding = proof.verify_role_binding(root)
    _expect(role_binding["digest"] == TASK3_ROLE_BINDING_DIGEST and role_binding["file_sha256"] == TASK3_RAW_ROLE_BINDING_SHA256,
            "SOURCE_OR_ISOLATION_MISMATCH", "historical Task3 role-binding provenance differs")
    lock_digest = proof.verify_uv_lock_digest(root)
    _expect(lock_digest == TASK3_UV_LOCK_SHA256, "SOURCE_OR_ISOLATION_MISMATCH", "historical dependency lock differs")
    for seed in MODEL_SEEDS:
        path = checkpoint_root / CHECKPOINTS[seed][0]
        proof.verify_checkpoint_file(path, seed)
    proof.assert_project_modules_pinned(root)
    return {"proof_module": proof, "mapping": mapping, "mapping_sha256": mapping_digest,
            "config_map": config_map, "runtime": runtime, "metric": metric,
            "role_binding": role_binding, "uv_lock_sha256": lock_digest,
            "checkpoints": {
                str(seed): {"path": str(checkpoint_root / CHECKPOINTS[seed][0]),
                            "bytes": CHECKPOINT_BYTES, "sha256": CHECKPOINTS[seed][1]}
                for seed in MODEL_SEEDS
            }}


def verify_release(
    worktree_root: Path, checkpoint_root: Path, binding: dict[str, Any],
) -> dict[str, Any]:
    root = worktree_root.resolve(strict=True)
    _expect(str(root) == "/tmp/sprint21-proof-hist-v1", "SOURCE_OR_ISOLATION_MISMATCH", "historical worktree path differs from the bound execution tree")
    _expect(checkpoint_root.resolve(strict=True) == Path("/tmp/sprint17-task7-out/checkpoints"),
            "PROVENANCE_MISMATCH", "checkpoint root differs from the original read-only B0 location")
    source = _verify_source_commit(root, binding)
    profile = _verify_pure_profile_configs(root, binding)
    runtime_assets = _verify_runtime_and_assets(root, checkpoint_root)
    return {"source": source, "profile": profile,
            "runtime_assets": {key: value for key, value in runtime_assets.items() if key != "proof_module"},
            "runtime_state": runtime_assets}


def _seed_lineage_record() -> dict[str, Any]:
    return {
        "history_data_seed_ranges_excluded": [list(value) for value in EXPECTED_EXCLUDED_RANGES],
        "single_data_seeds_excluded": list(EXPECTED_EXCLUDED_SINGLE_SEEDS),
        "model_seed_ranges_separate_and_avoided": [list(value) for value in EXPECTED_NONDATA_MODEL_SEEDS],
        "cited_source_records": _evidence_rows(),
        "collision_result": "No integer overlap between fixed 921000–921031 data-seed block and any enumerated prior history/data, disposable fixture, or model-seed namespace.",
        "coverage_limit": "Static repository binding/evidence catalog only; no data-root scan or Sealed-root inspection.",
    }


def _validate_manifest(
    manifest: dict[str, Any], samples: list[Any], coordinate: dict[str, Any],
    generator_version: str, expected_protocol_sha: str,
) -> dict[str, Any]:
    _expect(manifest.get("format") == 1, "MANIFEST_MISMATCH", "unexpected chronological manifest format")
    _expect(manifest.get("protocol") == PROTOCOL_ID, "MANIFEST_MISMATCH", "history protocol tag differs")
    _expect(manifest.get("role") == coordinate["history_id"], "MANIFEST_MISMATCH", "manifest whole-history ID differs")
    _expect(manifest.get("generator_version") == generator_version, "MANIFEST_MISMATCH", "generator version differs")
    _expect(manifest.get("config_hash") == coordinate["config_hash"], "CONFIG_MISMATCH", "manifest profile config hash differs")
    seeds = manifest.get("seeds", {})
    _expect(seeds == {name: coordinate["data_seed"] for name in ("factory", "scheduler", "health", "signal", "temporal")},
            "CONFIG_MISMATCH", "manifest must bind all five generator sub-seeds to the declared data seed")
    rows = manifest.get("files")
    _expect(isinstance(rows, list) and bool(rows), "MANIFEST_MISMATCH", "manifest file table is missing or empty")
    _expect(len(samples) == len(rows) == int(manifest.get("counts", {}).get("total", -1)),
            "MANIFEST_MISMATCH", "manifest file/sample/count rosters differ")
    sample_ids = [sample.file_id for sample in samples]
    row_ids = [row.get("file_id") for row in rows]
    _expect(row_ids == sample_ids, "MANIFEST_MISMATCH", "manifest and loaded sample file IDs or order differ")
    validate_runtime_roster(row_ids, sample_ids)
    _expect(len(sample_ids) == len(set(sample_ids)), "MANIFEST_MISMATCH", "duplicate file IDs in generated history")
    _expect(manifest.get("resolved_config") is not None, "MANIFEST_MISMATCH", "resolved profile configuration is absent")
    full_config_digest = hashlib.sha256(json.dumps(manifest["resolved_config"], sort_keys=True, default=str).encode()).hexdigest()
    _expect(full_config_digest == CONFIG_SHA256_BY_DATA_SEED[coordinate["data_seed"]],
            "CONFIG_MISMATCH", "materialized resolved configuration differs from the frozen full config fingerprint")
    _expect(expected_protocol_sha == GENERATOR_PROTOCOL_SHA256, "SOURCE_OR_ISOLATION_MISMATCH", "generator protocol digest differs")
    return {"file_count": len(rows), "file_ids": sample_ids,
            "resolved_config_sha256": full_config_digest, "seed_fields": seeds}


def _support_and_structure(
    manifest: dict[str, Any], coordinate: dict[str, Any], allocation: dict[str, Any] | None,
    allocation_failure: dict[str, Any] | None, task7: Any, events: Any, balanced: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(manifest["files"])
    ledger = events.failure_ledger(manifest)
    windows = manifest["maintenance_windows"]
    support = task7.evaluable_support(rows, ledger, windows)
    positive_windows: dict[str, list[dict[str, Any]]] = {"P": [], "W": [], "A": []}
    file_ids = {row["file_id"] for row in rows}
    source_to_global = {file_id: f"{coordinate['history_id']}::{file_id}" for file_id in file_ids}
    event_keys: list[str] = []
    for cohort in ("P", "W", "A"):
        for event in support["pos"][cohort]:
            key = event["key"]
            _expect(len(key) == 3 and key[0] == cohort, "SUPPORT_MISMATCH", "event support key is malformed")
            members = list(event["members"])
            _expect(bool(members) and set(members) <= file_ids, "SUPPORT_MISMATCH", "event window has missing/unknown member file IDs")
            stable_key = json.dumps(["EVENT", cohort, key[1], float(key[2])], sort_keys=True, separators=(",", ":"))
            event_keys.append(stable_key)
            positive_windows[cohort].append({
                "key": stable_key,
                "cohort": cohort,
                "robot_id": key[1],
                "failure_time": float(key[2]),
                "members": [source_to_global[file_id] for file_id in members],
            })
    _expect(len(event_keys) == len(set(event_keys)), "SUPPORT_MISMATCH", "duplicate event support keys")
    anchors = events.anchor_rows(rows, windows)
    controls_raw = events.select_control_windows(anchors, ledger, windows)
    support_controls = [list(member_ids) for member_ids in support["controls"]]
    raw_control_members = [[row["file_id"] for row in window["members"]] for window in controls_raw]
    _expect(support_controls == raw_control_members, "SUPPORT_MISMATCH", "Task7 support and canonical same-history control windows disagree")
    control_windows: list[dict[str, Any]] = []
    for window in controls_raw:
        members = [row["file_id"] for row in window["members"]]
        _expect(bool(members) and set(members) <= file_ids, "SUPPORT_MISMATCH", "control window has missing/unknown member file IDs")
        stable_key = json.dumps(["CONTROL", window["robot_id"], float(window["anchor_end"])], sort_keys=True, separators=(",", ":"))
        control_windows.append({"key": stable_key, "robot_id": window["robot_id"],
                                "anchor_end": float(window["anchor_end"]),
                                "members": [source_to_global[file_id] for file_id in members]})
    control_keys = [window["key"] for window in control_windows]
    _expect(len(control_keys) == len(set(control_keys)), "SUPPORT_MISMATCH", "duplicate control-window support keys")

    cohorts = {cohort: len(positive_windows[cohort]) for cohort in ("P", "W", "A")}
    positive_total = sum(cohorts.values())
    controls_count = len(control_windows)
    eval_rows = [row for row in rows if events.eligible_operational_row(row, windows)]
    robot_days = len({(row["robot_id"], int(row["end_time"] // 86400.0)) for row in eval_rows})
    positive_by_robot: dict[str, int] = {}
    for cohort in ("P", "W", "A"):
        for event in positive_windows[cohort]:
            positive_by_robot[event["robot_id"]] = positive_by_robot.get(event["robot_id"], 0) + 1
    negative_by_robot: dict[str, int] = {}
    for control in control_windows:
        negative_by_robot[control["robot_id"]] = negative_by_robot.get(control["robot_id"], 0) + 1
    max_pos_share = max(positive_by_robot.values(), default=0) / positive_total if positive_total else 1.0
    max_neg_share = max(negative_by_robot.values(), default=0) / controls_count if controls_count else 1.0
    program_by_op = {(row["robot_id"], row["end_time"]): row["program_id"] for row in rows}
    program_counts: dict[str, dict[str, int]] = {"P": {}, "W": {}}
    for cohort in ("P", "W"):
        for event in positive_windows[cohort]:
            program = program_by_op.get((event["robot_id"], event["failure_time"]))
            _expect(program is not None, "SUPPORT_MISMATCH", f"{cohort} event has no exact-endpoint program")
            program_counts[cohort][program] = program_counts[cohort].get(program, 0) + 1
    program_shares = {
        cohort: (max(counts.values()) / cohorts[cohort] if counts[cohort] else 1.0)
        for cohort, counts in program_counts.items()
    }
    mix = {cohort: (cohorts[cohort] / positive_total if positive_total else 0.0) for cohort in "PWA"}
    _, _, lead = balanced.eligible_anchors(rows, ledger, windows)
    failures_by_key = {
        (failure["cohort"], failure["robot_id"], float(failure["failure_time"])): failure
        for failure in ledger
    }
    lead_rates: dict[str, dict[str, Any]] = {}
    for cohort in ("P", "W"):
        event_results = []
        for event in positive_windows[cohort]:
            failure = failures_by_key.get((cohort, event["robot_id"], event["failure_time"]))
            _expect(failure is not None, "SUPPORT_MISMATCH", "eligible event window has no exact ledger event")
            detail = lead.get(failure["failure_id"], {})
            event_results.append({
                "failure_id": failure["failure_id"],
                "endpoints": int(detail.get("endpoints", 0)),
                "early_endpoint": bool(detail.get("early_endpoint", False)),
                "clean_baseline": bool(detail.get("clean_baseline", False)),
                "pass": bool(detail.get("endpoints", 0) >= 3 and detail.get("early_endpoint") and detail.get("clean_baseline")),
            })
        passed = sum(item["pass"] for item in event_results)
        lead_rates[cohort] = {
            "passed": passed, "total": len(event_results),
            "fraction": passed / len(event_results) if event_results else None,
            "events": event_results,
        }

    allowed_rows = getattr(balanced, "ALLOWED_ROW_KEYS")
    by_robot: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_robot.setdefault(row["robot_id"], []).append(row)
    expected_robots = [f"robot-{index:02d}" for index in range(1, 10)]
    row_contract = all(set(row) <= allowed_rows and "subtype" not in row and "cohort" not in row for row in rows)
    serial = all(
        all(current["start_time"] >= previous["end_time"] - 1e-6
            for previous, current in zip(sorted(robot_rows, key=lambda item: (item["start_time"], item["end_time"])),
                                         sorted(robot_rows, key=lambda item: (item["start_time"], item["end_time"]))[1:]))
        for robot_rows in by_robot.values()
    )
    maintenance_bounded = all(
        float(end) > float(start)
        for robot, intervals in windows.items() for start, end in intervals
    )
    cohort_set = {failure["cohort"] for failure in ledger}
    subtype_checks = all(
        (failure["subtype"] in ({"P1", "P2"} if failure["cohort"] == "P" else {"W1", "W2"} if failure["cohort"] == "W" else {"A1", "A2"}))
        for failure in ledger
    )
    p_durations = sorted(float(failure["duration_d"]) for failure in ledger if failure["cohort"] == "P")
    w_durations = sorted(float(failure["duration_d"]) for failure in ledger if failure["cohort"] == "W")
    p_shape = bool(p_durations) and min(p_durations) >= 2.0 and max(p_durations) <= 15.0 and 5.0 <= statistics.median(p_durations) <= 10.0
    w_shape = bool(w_durations) and min(w_durations) >= 6.0 and max(w_durations) <= 28.0 and 12.0 <= statistics.median(w_durations) <= 24.0
    expected_selected = {"P": {"P1": 12, "P2": 12}, "W": {"W1": 12, "W2": 12}, "A": {"A1": 8, "A2": 8}}
    exact_selected = bool(allocation) and {
        cohort: {subtype: len(ids) for subtype, ids in allocation.get("selected", {}).get(cohort, {}).items()}
        for cohort in ("P", "W", "A")
    } == expected_selected
    selected_ids = allocation.get("selected_ids", []) if allocation else []
    exact_selected = exact_selected and len(selected_ids) == 64 and len(set(selected_ids)) == 64
    selected_controls = len(allocation.get("controls", [])) if allocation else 0
    alloc_seed_match = bool(allocation) and allocation.get("alloc_seed") == balanced.alloc_seed_for(coordinate["data_seed"])
    structural_flags = {
        "manifest_role_protocol_and_five_seed_fields": True,
        "file_row_allowlist_no_cohort_or_subtype": row_contract,
        "nine_robot_vocabulary_and_reserve_robot": sorted(by_robot) == expected_robots and "robot-08" in by_robot,
        "reserved_program_present": "program-03" in {row["program_id"] for row in rows},
        "robot_chronology_serialized": serial,
        "maintenance_windows_positive": maintenance_bounded,
        "all_P_W_A_cohorts": cohort_set == {"P", "W", "A"},
        "cohort_subtypes_valid": subtype_checks,
        "P_degradation_duration_shape": p_shape,
        "W_degradation_duration_shape": w_shape,
        "exact_64_positive_quota": exact_selected,
        "allocator_seed_matches_history": alloc_seed_match,
        "exact_existing_48_control_construction_quota": selected_controls == QUOTA["controls"],
        "allocation_succeeded": allocation is not None,
    }
    hard_flags = dict(structural_flags)
    hard_flags.update({
        "P_ge_10": cohorts["P"] >= HARD_FLOORS["P"],
        "W_ge_10": cohorts["W"] >= HARD_FLOORS["W"],
        "A_ge_8": cohorts["A"] >= HARD_FLOORS["A"],
        "positive_total_ge_30": positive_total >= HARD_FLOORS["positive_total"],
        "controls_ge_25": controls_count >= HARD_FLOORS["controls"],
        "robot_days_ge_150": robot_days >= HARD_FLOORS["robot_days"],
        "positive_robots_ge_6": len(positive_by_robot) >= HARD_FLOORS["positive_robots"],
        "negative_robots_ge_6": len(negative_by_robot) >= HARD_FLOORS["negative_robots"],
        "programs_ge_2_each_P_W": all(len(program_counts[cohort]) >= HARD_FLOORS["programs_each_P_W"] for cohort in ("P", "W")),
        "cohort_mix_15_to_60": all(HARD_FLOORS["cohort_mix_min"] <= mix[cohort] <= HARD_FLOORS["cohort_mix_max"] for cohort in "PWA"),
        "positive_robot_share_le_35": max_pos_share <= HARD_FLOORS["positive_robot_share_max"],
        "negative_robot_share_le_40": max_neg_share <= HARD_FLOORS["negative_robot_share_max"],
        "program_share_le_60_each_P_W": all(share <= HARD_FLOORS["program_share_max"] for share in program_shares.values()),
        "lead_P_ge_80": bool(lead_rates["P"]["fraction"] is not None and lead_rates["P"]["fraction"] >= HARD_FLOORS["lead_support_P_W_min"]),
        "lead_W_ge_80": bool(lead_rates["W"]["fraction"] is not None and lead_rates["W"]["fraction"] >= HARD_FLOORS["lead_support_P_W_min"]),
    })
    design_flags = {
        "P_ge_13": cohorts["P"] >= DESIGN_MARGINS["P"],
        "W_ge_13": cohorts["W"] >= DESIGN_MARGINS["W"],
        "A_ge_10": cohorts["A"] >= DESIGN_MARGINS["A"],
        "positive_total_ge_38": positive_total >= DESIGN_MARGINS["positive_total"],
        "controls_ge_32": controls_count >= DESIGN_MARGINS["controls"],
        "robot_days_ge_188": robot_days >= DESIGN_MARGINS["robot_days"],
    }
    report = {
        "history_id": coordinate["history_id"], "data_seed": coordinate["data_seed"],
        "file_count": len(rows), "raw_failure_count": len(ledger),
        "eligible_positive_windows": cohorts, "eligible_positive_total": positive_total,
        "eligible_control_windows": controls_count,
        "selected_control_quota_count": selected_controls,
        "eligible_robot_days": robot_days,
        "positive_robots": len(positive_by_robot), "negative_robots": len(negative_by_robot),
        "mix": mix, "program_counts_P_W": program_counts, "program_share_max_P_W": program_shares,
        "positive_robot_max_share": max_pos_share, "negative_robot_max_share": max_neg_share,
        "lead_support": lead_rates, "hard_flags": hard_flags,
        "hard_status": "PASS" if all(hard_flags.values()) and allocation_failure is None else "FAIL",
        "hard_failures": [name for name, passed in hard_flags.items() if not passed] + (["exact_quota_allocator_failure"] if allocation_failure else []),
        "design_margin_flags": design_flags,
        "design_margin_misses_are_diagnostic_only": True,
        "allocation": allocation,
        "allocation_failure": allocation_failure,
        "score_independent_support_keys": {
            "P": [window["key"] for window in positive_windows["P"]],
            "W": [window["key"] for window in positive_windows["W"]],
            "A": [window["key"] for window in positive_windows["A"]],
            "controls": [window["key"] for window in control_windows],
        },
        "construction_quota": QUOTA,
    }
    return report, support, positive_windows, control_windows


def _config_record(coordinate: dict[str, Any], chronicle: Any) -> Any:
    cfg = chronicle.sprint15_v7_history_config(seed=coordinate["data_seed"])
    resolved = dataclasses.asdict(cfg)
    full_digest = hashlib.sha256(json.dumps(resolved, sort_keys=True, default=str).encode()).hexdigest()
    _expect(cfg.hash() == coordinate["config_hash"] and full_digest == coordinate["config_sha256"],
            "CONFIG_MISMATCH", f"data seed {coordinate['data_seed']} config differs at materialization time")
    return cfg
def _materialize_one(
    output_dir: Path, coordinate: dict[str, Any], verified: dict[str, Any],
    ledger_handle: Any,
) -> dict[str, Any]:
    profile_root = output_dir / coordinate["relative_root"]
    _expect(profile_root.is_relative_to(output_dir), "OUTPUT_PATH_MISMATCH", "history root escapes the bound one-shot output directory")
    _expect(not profile_root.exists(), "OUTPUT_ALREADY_EXISTS", f"history root already exists: {profile_root}")
    _json_line(ledger_handle, {"event": "history_materialization_started", "history_id": coordinate["history_id"], "data_seed": coordinate["data_seed"], "relative_root": coordinate["relative_root"]})
    chronicle = importlib.import_module("synth.chronicle")
    balanced = importlib.import_module("synth.balanced")
    events = importlib.import_module("synth.events")
    task7 = importlib.import_module("sprint17_task7_b0")
    cfg = _config_record(coordinate, chronicle)
    manifest = chronicle.materialize_chronological(
        cfg, profile_root, shard_size=64, overwrite=False,
        role=coordinate["history_id"], protocol=PROTOCOL_ID, sprint15=None,
    )
    samples, reloaded_manifest = chronicle.load_chronological(profile_root)
    _expect(manifest == reloaded_manifest, "MANIFEST_MISMATCH", "materializer result differs from verified persisted manifest")
    manifest_contract = _validate_manifest(
        reloaded_manifest, samples, coordinate,
        verified["profile"]["generator_version"], GENERATOR_PROTOCOL_SHA256,
    )
    allocation: dict[str, Any] | None = None
    allocation_failure: dict[str, Any] | None = None
    try:
        allocation = balanced.allocate_quotas(
            reloaded_manifest["files"], events.failure_ledger(reloaded_manifest),
            reloaded_manifest["maintenance_windows"], coordinate["data_seed"],
            balanced.QuotaConfig(), method="exact",
        )
    except Exception as exc:
        expected_failure_types = tuple(
            item for item in (
                getattr(balanced, "InfeasibleCandidate", None),
                getattr(balanced, "AllocationUnavailable", None),
            ) if isinstance(item, type)
        )
        if not expected_failure_types or not isinstance(exc, expected_failure_types):
            raise PilotError("ALLOCATION_RUNTIME_FAILURE", f"exact allocator raised an unclassified error: {type(exc).__name__}: {exc}") from exc
        allocation_failure = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "record": getattr(exc, "record", {}),
            "nonretryable": True,
            "replacement_or_search": False,
        }
    report, _, positive_windows, control_windows = _support_and_structure(
        reloaded_manifest, coordinate, allocation, allocation_failure, task7, events, balanced,
    )
    report["manifest_contract"] = manifest_contract
    report["manifest_sha256"] = sha256_file(profile_root / "manifest.json")
    report["manifest_bytes"] = (profile_root / "manifest.json").stat().st_size
    report["calendar_sha256"] = balanced.calendar_digest(reloaded_manifest["schedule"])
    report["shards"] = reloaded_manifest["shards"]
    report["source_root"] = str(profile_root)
    report["model_input_label_usage"] = "No labels or latent/event/split metadata are passed to the model; file labels are only retained in original manifest and diagnostic support logic."
    _write_json_new(output_dir / "history-evidence" / f"{coordinate['history_id']}.json", report)
    _json_line(ledger_handle, {"event": "history_materialization_complete", "history_id": coordinate["history_id"], "hard_status": report["hard_status"], "hard_failures": report["hard_failures"], "manifest_sha256": report["manifest_sha256"], "file_count": report["file_count"]})
    return {
        "coordinate": coordinate, "root": profile_root, "report": report,
        "positive_windows": positive_windows, "control_windows": control_windows,
    }




def _score_history(
    history: dict[str, Any], model: Any, bank: Any, patchifier: Any, cfg: Any,
    mapping: dict[str, Any], proof: Any, device: str, output_path: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from representation.data import collate_variable_files
    from representation.inference import RepresentationInference
    from synth.schema import FileSample, SampleLabel

    root_record = history["report"]
    history_id = root_record["history_id"]
    coordinate = history["coordinate"]
    manifest = history["manifest"]
    samples = history["samples"]
    row_ids = [row["file_id"] for row in manifest["files"]]
    rows = {row["file_id"]: row for row in manifest["files"]}
    ids = [f"{history_id}::{sample.file_id}" for sample in samples]
    expected_ids = [f"{history_id}::{source_id}" for source_id in row_ids]
    validate_runtime_roster(ids, expected_ids)
    by_branch: dict[str, dict[str, float]] = {"S_pred": {}, "S_pop": {}}
    file_records: list[dict[str, Any]] = []
    inference = RepresentationInference(model, bank, patchifier, masking_config=cfg)
    model.eval()
    path = output_path / f"{history_id}.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for sample in samples:
            source_id = sample.file_id
            row = rows[source_id]
            robot_id = row.get("robot_id")
            program_id = row.get("program_id")
            robot_idx = row.get("robot_idx")
            program_idx = row.get("program_idx")
            _expect(robot_id in mapping["robot_id_to_idx"] and mapping["robot_id_to_idx"][robot_id] == robot_idx,
                    "CONDITIONING_MISMATCH", f"{source_id}: robot string/index differ from pinned vocabulary")
            _expect(program_id in mapping["program_id_to_idx"] and mapping["program_id_to_idx"][program_id] == program_idx,
                    "CONDITIONING_MISMATCH", f"{source_id}: program string/index differ from pinned vocabulary")
            _expect(sample.robot_idx == robot_idx and sample.program_idx == program_idx,
                    "CONDITIONING_MISMATCH", f"{source_id}: serialized sample indices differ from manifest")
            x = sample.x
            _expect(isinstance(x, np.ndarray) and x.dtype == np.float32 and x.ndim == 2 and x.shape[0] == 6 and x.shape[1] > 0 and bool(np.isfinite(x).all()),
                    "INPUT_CONTRACT_FAILURE", f"{source_id}: model signal must be finite float32 [6,T]")
            global_id = f"{history_id}::{source_id}"
            mask_seed = int(proof.file_seed(global_id))
            # Build a fresh role-neutral model input; only x and the explicit
            # source-derived numeric conditioning IDs survive from the sample.
            neutral = FileSample(
                x=x, file_id=global_id, file_label=SampleLabel.NORMAL,
                seed=0, generator_version=manifest["generator_version"],
                config_hash=manifest["config_hash"], regime_sequence=[],
                robot_idx=robot_idx, program_idx=program_idx,
            )
            batch = collate_variable_files([neutral], patchifier, masking_config=cfg, masking_seed=mask_seed)
            valid = batch["patch_valid_mask"][0].detach().cpu().numpy().astype(bool)
            mask = batch["mask"][0].detach().cpu().numpy().astype(bool)
            _expect(bool(valid.any()) and int(mask.sum()) > 0 and not bool((mask & ~valid).any()),
                    "MASK_CONTRACT_FAILURE", f"{global_id}: invalid or empty prediction mask")
            model_batch = {
                key: value for key, value in batch.items()
                if key not in {"file_labels", "anomaly_meta", "anomaly_masks", "file_samples"}
            }
            moved = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in model_batch.items()
            }
            with torch.inference_mode():
                result = inference.score_batch(moved)
            pred = float(torch.as_tensor(result["S_pred"]).reshape(-1)[0].item())
            pop = float(torch.as_tensor(result["S_pop"]).reshape(-1)[0].item())
            patch_scores = torch.as_tensor(result["patch_scores"])
            prediction_mask = torch.as_tensor(result["prediction_mask"])
            _expect(tuple(patch_scores.shape) == tuple(prediction_mask.shape), "SCORE_MISMATCH", f"{global_id}: patch score/mask shapes differ")
            count = prediction_mask.sum(dim=1).clamp_min(1).to(patch_scores.dtype)
            expected_pred = float((patch_scores.sum(dim=1) / count).reshape(-1)[0].item())
            embedding = torch.as_tensor(result["file_embedding"])
            distances = torch.cdist(embedding.float().cpu(), bank.embeddings.float().cpu())
            expected_pop = float(distances.topk(5, largest=False, dim=1).values.mean(dim=1).reshape(-1)[0].item())
            for branch, got, expected in (("S_pred", pred, expected_pred), ("S_pop", pop, expected_pop)):
                _expect(math.isfinite(got) and math.isfinite(expected) and got >= 0.0 and expected >= 0.0,
                        "ROSTER_OR_SCORE_INCOMPLETE", f"{global_id}/{branch}: non-finite or negative score")
                tolerance = 1e-7 + 1e-6 * abs(expected)
                _expect(abs(got - expected) <= tolerance, "SCORE_MISMATCH", f"{global_id}/{branch}: inference result differs from independent recomputation")
            by_branch["S_pred"][global_id] = pred
            by_branch["S_pop"][global_id] = pop
            row_record = {
                "history_id": history_id, "data_seed": coordinate["data_seed"],
                "source_file_id": source_id, "global_file_id": global_id,
                "robot_id": robot_id, "robot_idx": robot_idx,
                "program_id": program_id, "program_idx": program_idx,
                "mask_seed": mask_seed,
                "S_pred": pred, "S_pop": pop,
            }
            output.write(json.dumps(row_record, sort_keys=True, allow_nan=False) + "\n")
            file_records.append(row_record)
        output.flush()
        os.fsync(output.fileno())
    support_keys = {
        branch: [
            *[window["key"] for cohort in ("P", "W", "A") for window in history["positive_windows"][cohort]],
            *[window["key"] for window in history["control_windows"]],
        ]
        for branch in ("S_pred", "S_pop")
    }
    combined = {
        file_id: {branch: by_branch[branch][file_id] for branch in ("S_pred", "S_pop")}
        for file_id in expected_ids
    }
    validate_score_roster(expected_ids, combined, support_keys)
    _expect(len(file_records) == len(expected_ids)
            and [record["global_file_id"] for record in file_records] == expected_ids,
            "ROSTER_OR_SCORE_INCOMPLETE", "score JSONL file roster differs from the declared history row order")
    metric_report = None
    metric_error = None
    metric_status = "UNCOMPUTABLE_HARD_SUPPORT_FAIL"
    if root_record["hard_status"] == "PASS":
        try:
            metric_report = history_window_scores(
                combined, history["positive_windows"], history["control_windows"],
                importlib.import_module("synth.events"),
            )
            metric_status = "COMPUTABLE"
        except PilotError as exc:
            if exc.code != "METRIC_UNCOMPUTABLE":
                raise
            metric_status = "UNCOMPUTABLE_METRIC"
            metric_error = str(exc)
    endpoint_record = {
        "history_id": history_id,
        "hard_status": root_record["hard_status"],
        "hard_failures": root_record["hard_failures"],
        "metric_status": metric_status,
        "metric_error": metric_error,
        "primary_auc": {
            branch: (
                metric_report["branches"][branch]["auc"]["P+W"]
                if metric_report is not None else None
            )
            for branch in ("S_pred", "S_pop")
        },
        "secondary_auc": {
            branch: (
                {cohort: metric_report["branches"][branch]["auc"][cohort] for cohort in ("P", "W", "A")}
                if metric_report is not None else {cohort: None for cohort in ("P", "W", "A")}
            )
            for branch in ("S_pred", "S_pop")
        },
        "design_margin_flags": root_record["design_margin_flags"],
        "support_keys": metric_report["support_keys"] if metric_report is not None else support_keys,
        "score_file_rows": len(file_records),
        "score_file_sha256": sha256_file(path),
        "score_file_bytes": path.stat().st_size,
    }
    return endpoint_record


def _run_pilot(
    worktree_root: Path, checkpoint_root: Path, output_dir: Path,
    binding: dict[str, Any], binding_digest: str,
) -> dict[str, Any]:
    import torch

    _expect(not output_dir.exists() and not output_dir.is_symlink(), "OUTPUT_ALREADY_EXISTS", f"one-shot output directory already exists: {output_dir}")
    verified = verify_release(worktree_root, checkpoint_root, binding)
    output_dir.mkdir(parents=False, exist_ok=False)
    ledger_path = output_dir / "pilot-ledger.jsonl"
    with ledger_path.open("x", encoding="utf-8", newline="\n") as ledger_handle:
        attempt = {
            "schema_id": "sprint21-pilot-attempt-v1",
            "protocol_id": PROTOCOL_ID,
            "binding_sha256": binding_digest,
            "status": "RUNNING",
            "run_started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "worktree": str(worktree_root), "checkpoint_root": str(checkpoint_root),
            "output_root": str(output_dir),
            "bound_coordinates": binding["roster"],
            "attempted_history_ids": [],
            "unattempted_history_ids": [coordinate["history_id"] for coordinate in binding["roster"]],
            "review_acknowledgement": "--review-passed supplied by caller after independent Task4 pre-contact PASS",
            "resume": False, "retry": False, "replacement": False,
        }
        _write_json_new(output_dir / "attempt.json", attempt)
        _json_line(ledger_handle, {"event": "run_started", "binding_sha256": binding_digest, "roster_size": 32})
        root_records: list[dict[str, Any]] = []
        endpoint_records: dict[int, list[dict[str, Any]]] = {seed: [] for seed in MODEL_SEEDS}
        try:
            history_ids = [coordinate["history_id"] for coordinate in binding["roster"]]
            for index, coordinate in enumerate(binding["roster"]):
                record = _materialize_one(output_dir, coordinate, verified, ledger_handle)
                root_records.append(record)
                if record["report"]["hard_status"] != "PASS":
                    unattempted_ids = history_ids[index + 1:]
                    stop_results = _scientific_support_stop(
                        history_ids, [item["report"] for item in root_records], unattempted_ids,
                    )
                    failed_ids = [
                        item["coordinate"]["history_id"] for item in root_records
                        if item["report"]["hard_status"] != "PASS"
                    ]
                    summary = {
                        "schema_id": "sprint21-pilot-summary-v1",
                        "protocol_id": PROTOCOL_ID,
                        "binding_sha256": binding_digest,
                        "status": "STOPPED_SCIENTIFIC_SUPPORT_FAIL",
                        "contact_authorized_by_task4": False,
                        "history_count": 32,
                        "history_unit": HISTORY_UNIT,
                        "attempted_history_ids": [item["coordinate"]["history_id"] for item in root_records],
                        "hard_failure_history_ids": failed_ids,
                        "unattempted_history_ids": unattempted_ids,
                        "model_seed_results_separate": stop_results,
                        "support_records": [item["report"] for item in root_records],
                        "source_and_runtime_verification": {
                            "source": verified["source"], "profile": verified["profile"],
                            "runtime": verified["runtime_state"]["runtime"],
                            "checkpoints": verified["runtime_state"]["checkpoints"],
                        },
                        "seed_lineage": _seed_lineage_record(),
                        "claim_limits": [
                            "no scorer restore or history score was performed after the first support-floor failure",
                            "no full-roster endpoint, history macro, variance, or bootstrap is computable",
                            "no retry, resume, replacement, omission, or continued generation after the stop",
                            "H=32 is sample size only and grants no miss allowance or tolerance rule",
                            "Sprint 20 NOT_READY, Sprint 18 quota48/no-Cycle4 boundary, and all Gate V2 decisions remain unchanged",
                        ],
                    }
                    _write_json_new(output_dir / "summary.json", summary)
                    attempt["status"] = "STOPPED_SCIENTIFIC_SUPPORT_FAIL"
                    attempt["run_stopped_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                    attempt["attempted_history_ids"] = [item["coordinate"]["history_id"] for item in root_records]
                    attempt["hard_failure_history_ids"] = failed_ids
                    attempt["unattempted_history_ids"] = unattempted_ids
                    _rewrite_json(output_dir / "attempt.json", attempt)
                    _json_line(ledger_handle, {
                        "event": "scientific_support_stop",
                        "hard_failure_history_ids": failed_ids,
                        "unattempted_history_ids": unattempted_ids,
                        "restore_or_scoring_started": False,
                        "retry": False, "resume": False, "replacement": False,
                    })
                    return summary
            _expect(len(root_records) == 32, "ROSTER_OR_SCORE_INCOMPLETE", "not all 32 bound histories materialized exactly once")
            proof = verified["runtime_state"]["proof_module"]
            config_map = verified["runtime_state"]["config_map"]
            mapping = verified["runtime_state"]["mapping"]
            device = "cuda"
            for model_seed in MODEL_SEEDS:
                checkpoint = checkpoint_root / CHECKPOINTS[model_seed][0]
                proof.verify_checkpoint_file(checkpoint, model_seed)
                model, bank, patchifier, cfg, bank_digest, saved_config_digest = proof.restore_seed(checkpoint, model_seed, device, config_map)
                state_before = proof.snapshot_state(model, bank)
                _expect(len(state_before) >= 3 and bank.k == 5 and tuple(bank.embeddings.shape) == (5040, 128),
                        "PROVENANCE_MISMATCH", f"seed {model_seed} restored bank/model dimensions differ")
                score_dir = output_dir / "scores" / f"model-{model_seed}"
                score_dir.mkdir(parents=True, exist_ok=False)
                chronicle = importlib.import_module("synth.chronicle")
                for history in root_records:
                    samples, manifest = chronicle.load_chronological(history["root"])
                    _expect(
                        sha256_file(history["root"] / "manifest.json") == history["report"]["manifest_sha256"],
                        "MANIFEST_MISMATCH", f"{history['coordinate']['history_id']}: persisted manifest changed before scoring",
                    )
                    scoring_history = {**history, "samples": samples, "manifest": manifest}
                    endpoint = _score_history(
                        scoring_history, model, bank, patchifier, cfg, mapping, proof,
                        device, score_dir,
                    )
                    endpoint_records[model_seed].append(endpoint)
                    _write_json_new(
                        output_dir / "history-evidence" / f"{history['coordinate']['history_id']}-model-{model_seed}.json",
                        endpoint,
                    )
                    _json_line(ledger_handle, {"event": "history_scored", "history_id": history["coordinate"]["history_id"], "model_seed": model_seed, "hard_status": endpoint["hard_status"], "metric_status": endpoint["metric_status"], "score_file_rows": endpoint["score_file_rows"]})
                    del scoring_history, samples, manifest
                state_after = proof.snapshot_state(model, bank)
                _expect(state_after == state_before, "RESTORE_ONLY_OR_ISOLATION_FAILURE", f"model seed {model_seed} parameters, buffers, or bank changed during scoring")
                proof.verify_checkpoint_file(checkpoint, model_seed)
                _json_line(ledger_handle, {"event": "model_seed_complete", "model_seed": model_seed, "checkpoint_sha256": CHECKPOINTS[model_seed][1], "bank_digest": bank_digest, "saved_config_sha256": saved_config_digest, "state_digest": canonical_sha256(state_after)})
                del model, bank, patchifier, cfg
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            final = {}
            history_ids = [coordinate["history_id"] for coordinate in binding["roster"]]
            for model_seed in MODEL_SEEDS:
                ordered_endpoints = endpoint_records[model_seed]
                _expect([record["history_id"] for record in ordered_endpoints] == history_ids,
                        "ROSTER_OR_SCORE_INCOMPLETE", f"model seed {model_seed} endpoint roster/order differs")
                final[str(model_seed)] = aggregate_full_roster(history_ids, ordered_endpoints)
            summary = {
                "schema_id": "sprint21-pilot-summary-v1",
                "protocol_id": PROTOCOL_ID,
                "binding_sha256": binding_digest,
                "status": "COMPLETE_DESCRIPTIVE_RESEARCH_ONLY",
                "contact_authorized_by_task4": False,
                "history_count": 32,
                "history_unit": HISTORY_UNIT,
                "model_seed_results_separate": final,
                "support_records": [record["report"] for record in root_records],
                "source_and_runtime_verification": {
                    "source": verified["source"], "profile": verified["profile"],
                    "runtime": verified["runtime_state"]["runtime"],
                    "checkpoints": verified["runtime_state"]["checkpoints"],
                },
                "seed_lineage": _seed_lineage_record(),
                "claim_limits": [
                    "simulator-conditional descriptive pilot only",
                    "no real-world prevalence, calibrated risk, fleet-generalization, or causal claim",
                    "no training, refitting, bank fitting, calibration, threshold, or score fusion",
                    "S_pred and S_pop remain separate; all three B0 scorer seeds reported independently",
                    "H=32 is sample size only and grants no miss allowance or tolerance rule",
                    "Sprint 20 NOT_READY, Sprint 18 quota48/no-Cycle4 boundary, and all Gate V2 decisions remain unchanged",
                ],
            }
            _write_json_new(output_dir / "summary.json", summary)
            attempt["attempted_history_ids"] = history_ids
            attempt["unattempted_history_ids"] = []
            attempt["status"] = "COMPLETE"
            attempt["run_completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            _rewrite_json(output_dir / "attempt.json", attempt)
            _json_line(ledger_handle, {"event": "run_complete", "status": summary["status"], "histories": 32, "model_seeds": list(MODEL_SEEDS)})
            return summary
        except Exception as exc:
            history_ids = [coordinate["history_id"] for coordinate in binding["roster"]]
            completed_by_seed = {
                str(seed): [record["history_id"] for record in endpoint_records[seed]]
                for seed in MODEL_SEEDS
            }
            completed_sets = {
                seed: set(record["history_id"] for record in endpoint_records[seed])
                for seed in MODEL_SEEDS
            }
            attempt["status"] = "ABORTED"
            attempt["run_aborted_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            attempt["materialized_completed_history_ids"] = [record["coordinate"]["history_id"] for record in root_records]
            attempt["completed_score_history_ids_by_model_seed"] = completed_by_seed
            attempt["unattempted_score_history_ids_by_model_seed"] = {
                str(seed): [history_id for history_id in history_ids if history_id not in completed_sets[seed]]
                for seed in MODEL_SEEDS
            }
            attempt["error_code"] = getattr(exc, "code", "UNEXPECTED_FAILURE")
            attempt["error_type"] = type(exc).__name__
            attempt["error_message"] = str(exc)
            _rewrite_json(output_dir / "attempt.json", attempt)
            _json_line(ledger_handle, {
                "event": "run_aborted", "error_type": type(exc).__name__,
                "code": getattr(exc, "code", "UNEXPECTED_FAILURE"), "message": str(exc),
                "materialized_completed_history_ids": [record["coordinate"]["history_id"] for record in root_records],
                "completed_score_history_ids_by_model_seed": completed_by_seed,
                "unattempted_score_history_ids_by_model_seed": {
                    str(seed): [history_id for history_id in history_ids if history_id not in completed_sets[seed]]
                    for seed in MODEL_SEEDS
                },
                "retry": False, "resume": False, "replacement": False,
            })
            raise


def _smoke() -> dict[str, Any]:
    ids = ["H-S21-TEST-01", "H-S21-TEST-02", "H-S21-TEST-03"]
    records = [
        {"history_id": ids[0], "hard_status": "PASS", "metric_status": "COMPUTABLE", "primary_auc": {"S_pred": 0.5, "S_pop": 0.4}, "secondary_auc": {"S_pred": {"P": 0.6, "W": 0.4, "A": 0.5}, "S_pop": {"P": 0.5, "W": 0.3, "A": 0.4}}, "design_margin_flags": {"P_ge_13": True, "W_ge_13": False}},
        {"history_id": ids[1], "hard_status": "PASS", "metric_status": "COMPUTABLE", "primary_auc": {"S_pred": 1.0, "S_pop": 0.8}, "secondary_auc": {"S_pred": {"P": 1.0, "W": 1.0, "A": 0.5}, "S_pop": {"P": 0.7, "W": 0.8, "A": 0.4}}, "design_margin_flags": {"P_ge_13": False, "W_ge_13": True}},
        {"history_id": ids[2], "hard_status": "PASS", "metric_status": "COMPUTABLE", "primary_auc": {"S_pred": 0.0, "S_pop": 0.6}, "secondary_auc": {"S_pred": {"P": 0.0, "W": 0.0, "A": 0.5}, "S_pop": {"P": 0.6, "W": 0.5, "A": 0.4}}, "design_margin_flags": {"P_ge_13": True, "W_ge_13": True}},
    ]
    summary = aggregate_full_roster(ids, records, replicates=100, seed=20260202)
    _expect(summary["full_roster_status"] == "COMPUTABLE" and math.isclose(summary["primary"]["S_pred"]["macro"], 0.5),
            "SMOKE_FAILURE", "full-roster macro computation failed")
    _expect(math.isclose(summary["primary"]["S_pred"]["sample_variance"], 0.25)
            and math.isclose(summary["primary"]["S_pop"]["macro"], 0.6),
            "SMOKE_FAILURE", "history variance or separate branch macro is incorrect")
    margin_strata = summary["support_dependence_strata"]["P_ge_13"]["S_pred"]
    _expect(margin_strata["margin_met"]["n_hard_pass_histories_with_auc"] == 2
            and margin_strata["margin_missed"]["n_hard_pass_histories_with_auc"] == 1,
            "SMOKE_FAILURE", "both sides of the support-margin diagnostic were not reported")
    invalid = copy.deepcopy(records)
    invalid[1]["hard_status"] = "FAIL"
    invalid[1]["primary_auc"]["S_pred"] = None
    failed = aggregate_full_roster(ids, invalid, replicates=100, seed=20260202)
    _expect(failed["full_roster_status"] == "UNCOMPUTABLE_FULL_ROSTER"
            and failed["primary"]["S_pred"]["macro"] is None
            and failed["primary"]["S_pred"]["bootstrap_lcb95"] is None,
            "SMOKE_FAILURE", "a failed history was improperly removed from the primary roster")
    metric_invalid = copy.deepcopy(records)
    metric_invalid[0]["metric_status"] = "UNCOMPUTABLE_METRIC"
    metric_failed = aggregate_full_roster(ids, metric_invalid, replicates=100, seed=20260202)
    _expect(metric_failed["full_roster_status"] == "UNCOMPUTABLE_FULL_ROSTER"
            and metric_failed["metric_uncomputable_history_ids"] == [ids[0]]
            and metric_failed["primary"]["S_pop"]["macro"] is None,
            "SMOKE_FAILURE", "an uncomputable metric history was improperly removed from the primary roster")
    stop_ids = [coordinate["history_id"] for coordinate in fixed_roster()]
    stopped = _scientific_support_stop(
        stop_ids,
        [{"history_id": stop_ids[0], "hard_status": "PASS"},
         {"history_id": stop_ids[1], "hard_status": "FAIL"}],
        stop_ids[2:],
    )
    _expect(stopped[str(MODEL_SEEDS[0])]["primary"]["S_pred"]["macro"] is None
            and stopped[str(MODEL_SEEDS[0])]["primary"]["S_pred"]["per_history"][0]["status"] == "SUPPORT_PASS_NOT_SCORED_AFTER_STOP"
            and stopped[str(MODEL_SEEDS[0])]["primary"]["S_pred"]["per_history"][2]["status"] == "NOT_ATTEMPTED_AFTER_STOP",
            "SMOKE_FAILURE", "scientific support stop continued or produced a subset endpoint")
    validate_score_roster(["a", "b"], {"a": {"S_pred": 1.0, "S_pop": 2.0}, "b": {"S_pred": 3.0, "S_pop": 4.0}}, {"S_pred": {"p1", "n1"}, "S_pop": {"p1", "n1"}})
    rejected = 0
    for malformed in (
        {"a": {"S_pred": 1.0, "S_pop": 2.0}},
        {"a": {"S_pred": 1.0, "S_pop": 2.0}, "b": {"S_pred": math.nan, "S_pop": 4.0}},
    ):
        try:
            validate_score_roster(["a", "b"], malformed, {"S_pred": {"p1"}, "S_pop": {"p1"}})
        except PilotError as exc:
            _expect(exc.code == "ROSTER_OR_SCORE_INCOMPLETE", "SMOKE_FAILURE", "wrong disposition for invalid full-roster scores")
            rejected += 1
    _expect(rejected == 2, "SMOKE_FAILURE", "invalid roster cases were not both rejected")
    try:
        validate_roster(fixed_roster()[:-1])
    except PilotError as exc:
        _expect(exc.code == "BINDING_MISMATCH", "SMOKE_FAILURE", "wrong disposition for short fixed roster")
    else:
        raise PilotError("SMOKE_FAILURE", "short fixed roster was accepted")
    return {
        "status": "SMOKE_PASS_NO_PROJECT_IMPORTS_NO_DATA_CONTACT",
        "full_roster_macro_and_history_variance": "PASS",
        "hard_failure_prevents_subset_macro_bootstrap": "PASS",
        "first_support_failure_stops_generation_and_scoring": "PASS",
        "missing_or_nonfinite_file_branch_scores_rejected": rejected,
        "fixed_roster_mutation_rejected": True,
        "generator_or_preflight_called": False,
        "model_or_checkpoint_loaded": False,
        "history_root_read_or_written": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="run in-memory helper invariants only")
    mode.add_argument("--dry-run", action="store_true", help="verify the frozen source/runtime/checkpoints without materialization or scoring")
    mode.add_argument("--run", action="store_true", help="one-shot materialize and score the complete frozen H=32 roster")
    parser.add_argument("--worktree-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--expected-binding-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--review-passed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.smoke:
        print(json.dumps(_smoke(), sort_keys=True))
        return 0
    for argument in ("worktree_root", "checkpoint_root", "binding", "expected_binding_sha256", "output_dir"):
        _expect(getattr(args, argument) is not None, "CLI_CONTRACT_FAILURE", f"--{argument.replace('_', '-')} is required")
    binding, binding_digest = read_binding(args.binding, args.expected_binding_sha256)
    expected_output = Path(binding["output"]["run_root"])
    _expect(args.output_dir.resolve(strict=False) == expected_output,
            "OUTPUT_PATH_MISMATCH", "output path differs from the exact bound run root")
    args.output_dir = expected_output
    if args.dry_run:
        _expect(not args.review_passed, "CLI_CONTRACT_FAILURE", "--dry-run does not accept review acknowledgement")
        verified = verify_release(args.worktree_root, args.checkpoint_root, binding)
        result = {
            "status": "DRY_RUN_PASS_NO_SEED_EVALUATION",
            "binding_sha256": binding_digest,
            "contact_or_generation": False,
            "preflight": False,
            "scoring_or_checkpoint_deserialization": False,
            "configuration_constructors_only": True,
            "output_path_touched": False,
            "verified": {
                "source": verified["source"],
                "profile": verified["profile"],
                "runtime": verified["runtime_state"]["runtime"],
                "checkpoints": verified["runtime_state"]["checkpoints"],
                "mapping_sha256": verified["runtime_state"]["mapping_sha256"],
                "metric_code_sha256": verified["runtime_state"]["metric"]["digest"],
            },
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    _expect(args.review_passed, "REVIEW_GATE_REQUIRED", "--run requires --review-passed after independent Task4 pre-contact PASS")
    summary = _run_pilot(args.worktree_root, args.checkpoint_root, args.output_dir, binding, binding_digest)
    print(json.dumps({"status": summary["status"], "binding_sha256": binding_digest, "history_count": 32}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "code": exc.code, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
