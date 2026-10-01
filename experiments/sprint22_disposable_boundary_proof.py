"""Sprint 22 Task 2: disposable allocator and restore-only scorer proof.

This one-shot entry point runs only the reviewed in-memory allocator fixtures
and two frozen mathematical file samples. It has no history-root, generator,
preflight, training, or network interface. See the byte-verified contract at
``experiments/sprint22-disposable-boundary-proof-v1.md``.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_SRC = (REPO_ROOT / "src").resolve()
if str(PINNED_SRC) not in sys.path:
    sys.path.insert(0, str(PINNED_SRC))

CONTRACT_SHA256 = "dcb9b667f86e9a2240a86022490aae93c948bf5ff88f08dd4dd9c52f9701f1d4"
BASE_COMMIT = "8c15f0204a3e495569b7f143dc109943e8b808de"
OUTPUT_ROOT = Path("/tmp/sprint22-disposable-boundary-proof-out-v1")
CHECKPOINT_ROOT = Path("/tmp/sprint17-task7-out/checkpoints")
MODEL_SEEDS = (171701, 171702, 171703)
CHECKPOINT_FILES = {
    171701: "b0_seed171701_step300.pt",
    171702: "b0_seed171702_step300.pt",
    171703: "b0_seed171703_step300.pt",
}
CHECKPOINT_SHA256 = {
    171701: "45f9e151c5d3946804ea531eb2bcace26b1f62e634671f51b8741ac6b411a619",
    171702: "64ed2cedec210e4692f60bb4dd3430a57cf94abfcd50551abd97c9265ea785f8",
    171703: "9edb2122e357376a9ff1e065d73d5ab7704f8d2005991552332f1ced01b0e906",
}
BANK_SHA256 = {
    171701: "432852b7c8ba6d1845b26e3e8f4d34f2e986258d39da22b844c15def2bc00662",
    171702: "6bae50edd1aee8ac4a00a6255e85db01b89eeaef7341405ac58c93e0a15b511a",
    171703: "7dfe8c3e63b764f0a043c5c38ba63fced463ce78846d38153bc5588a2a6efff8",
}
CHECKPOINT_BYTES = 18_239_321
CHECKPOINT_STEP = 300
CONFIG_SHA256 = "1ff67f95428ef29aab05d9f6394a305b75c8c2a9e12431f3cda09b6958596144"
UV_LOCK_SHA256 = "4a7866878c81cdd8f5ba283cd8d0a772073bfc75941f73ec761ede5eeb2e239a"
METRIC_SHA256 = "b2d6af7505abe459a84f76f5279a6a5c4298e7c142901edd4596fcfe3ddeb88f"
MAPPING_SHA256 = "4f769ccc64aac4c6a91fdd356eaa6963afe976ab847a22df23e43f9331418897"
METRIC_FILES = (
    "src/synth/probe15.py",
    "src/synth/events.py",
    "src/representation/attribution_metrics.py",
    "src/representation/sprint17_ablation.py",
    "experiments/sprint17_task7_b0.py",
)
SOURCE_HASHES = {
    "src/representation/model.py": "7051f5f76be0bcc8e50d6d5ee2941a57c12a3285406d0ef27883c596aa6600f0",
    "src/representation/data.py": "88ea3becbc8328c5b21518b1f957fe710096cb96ed0cd2416fe6f139439d5eb4",
    "src/representation/masking.py": "c0efd33b24b8d01c067198f0bc04c0c90c1e99bad9a6a1ec006b37ce27c0465f",
    "src/representation/contracts.py": "0b23080fc49f218484a6f7939bb9288f07b607686312e91c3b14d9d6b84464ed",
    "src/representation/layers/normalization.py": "bfb5f68f5352115c2e044c27999cdf3fecab7aa3ce2cd05ce021204f1f9edd8d",
    "src/synth/chronicle.py": "b5992d6af3c4387d18df746b43be17b4bfc1fac6938e8b44be35b6fb7c65dafd",
    "src/synth/balanced.py": "d62de3422bd51f870d39d18b1a8bb942f9ca73fc0044d8f23cc2d0948bd35a43",
    "src/synth/events.py": "85100f5e0aca47dd2e8b01a08c58f39d32be4f1eab469cdc38e4a4b57768169d",
}
CHANNELS = 6
BANK_ROWS = 5040
BANK_DIM = 128
BANK_K = 5
REL_TOL = 1e-6
ABS_TOL = 1e-7
FIXTURE_SEED = 220001
EXPECTED_PYTHON = "3.12.13"
EXPECTED_TORCH = "2.14.0+cu130"
EXPECTED_DEVICE = "NVIDIA GeForce RTX 4060 Ti"
EXPECTED_DEVICE_MIB = 16380

TOY_SPECS = (
    {
        "file_id": "S22T1-BND-TOY-A", "formula": "A",
        "robot_id": "robot-01", "robot_idx": 0,
        "program_id": "program-01", "program_idx": 0,
        "tensor_sha256": "236eed5f43d5a407fbc3598476a4e761859b658c168c90cdc0acb93ebbae7f0b",
        "mask_seed": 829471962,
        "file_id_sha256": "3170bcda6a9355d2c4065cdd80d381b28e1a9e1d0d643bb1273055d110d481b5",
    },
    {
        "file_id": "S22T1-BND-TOY-B", "formula": "B",
        "robot_id": "robot-09", "robot_idx": 8,
        "program_id": "program-08", "program_idx": 7,
        "tensor_sha256": "55afb88796385f9aaeeafe6b553b5166412c995103323bc115083981b14b9f38",
        "mask_seed": 928089417,
        "file_id_sha256": "b7518549dbc0310cda36c45f1581ec1768f33c2e12351514df61a5a727ef6ea0",
    },
)


class ProofFailure(RuntimeError):
    """Fail-closed proof error carrying an observable contract disposition."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def file_seed(file_id: str) -> int:
    digest = hashlib.sha256(file_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def verify_reviewed_contract(path: Path) -> str:
    try:
        digest = sha256_file(path)
    except OSError as exc:
        raise ProofFailure("CONTRACT_MISMATCH", f"cannot read reviewed contract: {exc}") from exc
    if digest != CONTRACT_SHA256:
        raise ProofFailure("CONTRACT_MISMATCH", f"reviewed contract SHA-256 {digest} != {CONTRACT_SHA256}")
    return digest


def verify_wrapper_identity(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"wrapper file missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"wrapper SHA-256 {actual} != supplied {expected_sha256}")
    return actual


def validate_runtime(info: dict[str, object]) -> None:
    expected = {
        "python": EXPECTED_PYTHON,
        "torch": EXPECTED_TORCH,
        "cuda_available": True,
        "device_name": EXPECTED_DEVICE,
        "device_total_mib": EXPECTED_DEVICE_MIB,
    }
    mismatches = {name: (info.get(name), value) for name, value in expected.items() if info.get(name) != value}
    if mismatches:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"runtime differs from frozen runtime: {mismatches!r}")


def runtime_info() -> dict[str, object]:
    import torch

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True, text=True, check=True,
        )
        device_name, total_mib = (part.strip() for part in result.stdout.strip().split(",", 1))
        physical_total_mib = int(total_mib)
    except Exception as exc:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"cannot measure physical GPU identity/resources: {exc}") from exc
    info: dict[str, object] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": device_name,
        "device_total_mib": physical_total_mib,
        "torch_visible_total_mib": int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)) if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available() and torch.cuda.get_device_name(0) != device_name:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"torch device name differs from nvidia-smi: {torch.cuda.get_device_name(0)!r} != {device_name!r}")
    validate_runtime(info)
    return info


def validate_source_digest(relative_path: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"{relative_path} SHA-256 {actual} != pinned {expected}")


def verify_historical_tree(worktree_root: Path, wrapper_sha256: str) -> dict[str, object]:
    if worktree_root.resolve() != REPO_ROOT.resolve():
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", "wrapper must execute from the dedicated historical worktree")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=worktree_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=worktree_root,
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
    except Exception as exc:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"cannot verify historical Git worktree: {exc}") from exc
    if head != BASE_COMMIT:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"historical HEAD {head} != {BASE_COMMIT}")
    expected_overlay = "?? experiments/sprint22_disposable_boundary_proof.py"
    if status != [expected_overlay]:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"historical worktree changes are not the single authorized wrapper: {status!r}")
    observed: dict[str, str] = {}
    for relative_path, expected in SOURCE_HASHES.items():
        path = worktree_root / relative_path
        actual = sha256_file(path)
        validate_source_digest(relative_path, actual, expected)
        observed[relative_path] = actual
    lock_hash = sha256_file(worktree_root / "uv.lock")
    validate_source_digest("uv.lock", lock_hash, UV_LOCK_SHA256)
    overlay_hash = sha256_file(worktree_root / "experiments/sprint22_disposable_boundary_proof.py")
    validate_source_digest("experiments/sprint22_disposable_boundary_proof.py", overlay_hash, wrapper_sha256)
    member_hashes: dict[str, str] = {}
    metric_rows: list[list[str]] = []
    for relative_path in METRIC_FILES:
        actual = sha256_file(worktree_root / relative_path)
        member_hashes[relative_path] = actual
        metric_rows.append([relative_path, actual])
    metric_hash = canonical_sha256(metric_rows)
    validate_source_digest("metric-code closure", metric_hash, METRIC_SHA256)
    observed.update({f"metric:{name}": digest for name, digest in member_hashes.items()})
    observed["uv.lock"] = lock_hash
    source_closure_sha256 = canonical_sha256(sorted(observed.items()))
    return {
        "base_commit": head,
        "git_status": status,
        "wrapper_overlay_sha256": overlay_hash,
        "source_hashes": observed,
        "source_closure_sha256": source_closure_sha256,
        "metric_code_sha256": metric_hash,
        "metric_member_sha256": member_hashes,
        "uv_lock_sha256": lock_hash,
    }


def assert_project_modules_pinned(worktree_root: Path) -> None:
    pinned = (worktree_root / "src").resolve()
    checked: dict[str, str] = {}
    for name, module in tuple(sys.modules.items()):
        if not (name == "synth" or name.startswith("synth.") or name == "representation" or name.startswith("representation.")):
            continue
        location = getattr(module, "__file__", None)
        if location is None:
            continue
        resolved = Path(location).resolve()
        if not resolved.is_relative_to(pinned):
            raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"module {name} imported outside pinned source: {resolved}")
        checked[name] = str(resolved)
    if not checked:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", "no project modules were imported from the pinned source")


def conditioning_mapping_from_config(config: object) -> dict[str, dict[str, int]]:
    try:
        stages = [stage for route in config.scheduler.routes for stage in route.stages]
        robot_ids = sorted({stage.robot_id for stage in stages})
        program_ids = sorted({stage.program_id for stage in stages})
        if len(robot_ids) != config.fleet.n_robots or len(program_ids) != config.fleet.n_programs:
            raise ValueError("configured fleet cardinalities differ from route identifiers")
    except Exception as exc:
        raise ProofFailure("CONDITIONING_MISMATCH", f"cannot derive identifier map from pinned static config: {exc}") from exc
    mapping = {
        "robot_id_to_idx": {name: index for index, name in enumerate(robot_ids)},
        "program_id_to_idx": {name: index for index, name in enumerate(program_ids)},
    }
    digest = canonical_sha256(mapping)
    if digest != MAPPING_SHA256:
        raise ProofFailure("CONDITIONING_MISMATCH", f"source-derived conditioning map SHA-256 {digest} != {MAPPING_SHA256}")
    return mapping


def derive_conditioning_mapping(worktree_root: Path) -> dict[str, dict[str, int]]:
    chronicle = importlib.import_module("synth.chronicle")
    module_path = Path(chronicle.__file__).resolve()
    if not module_path.is_relative_to((worktree_root / "src").resolve()):
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"chronicle imported outside pinned source: {module_path}")
    config = chronicle.sprint15_v7_history_config(seed=0)
    return conditioning_mapping_from_config(config)


def load_b0_config_map(worktree_root: Path) -> dict[str, dict[str, object]]:
    path = (worktree_root / "experiments/sprint17_task7_b0.py").resolve()
    if not path.is_file() or not path.is_relative_to((worktree_root / "experiments").resolve()):
        raise ProofFailure("PROVENANCE_MISMATCH", f"pinned B0 config source missing or outside historical tree: {path}")
    module_name = "_s22_pinned_task7_b0"
    if module_name in sys.modules:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", "pinned B0 config module name already loaded")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ProofFailure("PROVENANCE_MISMATCH", f"cannot load pinned B0 config source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        configs = {str(seed): module.b0_config_dict(seed) for seed in MODEL_SEEDS}
    except Exception as exc:
        raise ProofFailure("PROVENANCE_MISMATCH", f"cannot derive pinned B0 configs: {exc}") from exc
    finally:
        sys.modules.pop(module_name, None)
    if set(configs) != {str(seed) for seed in MODEL_SEEDS}:
        raise ProofFailure("PROVENANCE_MISMATCH", "B0 configuration map does not contain all three frozen seeds")
    for seed in MODEL_SEEDS:
        if configs[str(seed)].get("model_seed") != seed:
            raise ProofFailure("PROVENANCE_MISMATCH", f"B0 config seed mismatch for {seed}")
    digest = canonical_sha256(configs)
    if digest != CONFIG_SHA256:
        raise ProofFailure("PROVENANCE_MISMATCH", f"full B0 configuration SHA-256 {digest} != {CONFIG_SHA256}")
    return configs


def verify_checkpoint_digest(path: Path, model_seed: int, expected_size: int = CHECKPOINT_BYTES, expected_sha256: str | None = None) -> str:
    if model_seed not in CHECKPOINT_SHA256:
        raise ProofFailure("PROVENANCE_MISMATCH", f"unexpected model seed {model_seed}")
    if not path.is_file():
        raise ProofFailure("PROVENANCE_MISMATCH", f"missing checkpoint: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ProofFailure("PROVENANCE_MISMATCH", f"{path.name} size {actual_size} != {expected_size}")
    actual_sha = sha256_file(path)
    expected = CHECKPOINT_SHA256[model_seed] if expected_sha256 is None else expected_sha256
    if actual_sha != expected:
        raise ProofFailure("PROVENANCE_MISMATCH", f"{path.name} SHA-256 {actual_sha} != {expected}")
    return actual_sha


def validate_checkpoint_metadata(meta: dict[str, object], model_seed: int, expected_config: dict[str, object]) -> str:
    if meta.get("step") != CHECKPOINT_STEP:
        raise ProofFailure("PROVENANCE_MISMATCH", f"checkpoint step {meta.get('step')} != {CHECKPOINT_STEP}")
    if not meta.get("has_reference_bank"):
        raise ProofFailure("PROVENANCE_MISMATCH", "checkpoint has no embedded reference bank")
    config = meta.get("config")
    if not isinstance(config, dict) or config.get("seed") != model_seed or expected_config.get("seed") != model_seed:
        raise ProofFailure("PROVENANCE_MISMATCH", f"checkpoint config seed differs from {model_seed}")
    restored_hash = canonical_sha256(config)
    expected_hash = canonical_sha256(expected_config)
    if restored_hash != expected_hash:
        raise ProofFailure("PROVENANCE_MISMATCH", f"seed {model_seed} checkpoint config digest {restored_hash} != {expected_hash}")
    return restored_hash


def validate_bank_identity(k: int, shape: tuple[int, ...], digest: str, model_seed: int) -> None:
    if k != BANK_K or shape != (BANK_ROWS, BANK_DIM) or digest != BANK_SHA256[model_seed]:
        raise ProofFailure("PROVENANCE_MISMATCH", f"seed {model_seed} embedded bank identity differs: k={k}, shape={shape}, sha256={digest}")


def validate_module_origin(path: Path, pinned_src: Path, module_name: str) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(pinned_src.resolve()):
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"module {module_name} imported outside pinned source: {resolved}")
    return str(resolved)


def validate_conditioning(record: "ToyRecord", mapping: dict[str, dict[str, int]]) -> None:
    if not isinstance(record.robot_id, str) or record.robot_id not in mapping["robot_id_to_idx"]:
        raise ProofFailure("CONDITIONING_MISMATCH", f"{record.file_id}: unknown robot identifier {record.robot_id!r}")
    if not isinstance(record.program_id, str) or record.program_id not in mapping["program_id_to_idx"]:
        raise ProofFailure("CONDITIONING_MISMATCH", f"{record.file_id}: unknown program identifier {record.program_id!r}")
    if type(record.robot_idx) is not int or mapping["robot_id_to_idx"][record.robot_id] != record.robot_idx:
        raise ProofFailure("CONDITIONING_MISMATCH", f"{record.file_id}: robot string and serialized index disagree")
    if type(record.program_idx) is not int or mapping["program_id_to_idx"][record.program_id] != record.program_idx:
        raise ProofFailure("CONDITIONING_MISMATCH", f"{record.file_id}: program string and serialized index disagree")


def validate_input(record: "ToyRecord") -> None:
    import numpy as np

    x = record.x
    if not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.ndim != 2 or x.shape[0] != CHANNELS or x.shape[1] != 64 or not bool(np.isfinite(x).all()):
        raise ProofFailure("INPUT_CONTRACT_FAILURE", f"{record.file_id}: expected finite float32 [6,64] input")


def validate_mask(mask: Any, valid: Any, file_id: str) -> list[int]:
    import numpy as np

    actual = np.asarray(mask, dtype=bool)
    valid_bits = np.asarray(valid, dtype=bool)
    if actual.shape != valid_bits.shape or int(valid_bits.sum()) != 3 or int(actual.sum()) != 1 or bool((actual & ~valid_bits).any()):
        raise ProofFailure("MASK_CONTRACT_FAILURE", f"{file_id}: mask must select exactly one of three valid patches and no padding")
    return [int(value) for value in actual.tolist()]


def read_serialized_conditioning(payload: bytes, file_id: str) -> tuple[int, int]:
    import numpy as np

    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            result: list[int] = []
            for field_name in ("robot_idx", "program_idx"):
                if field_name not in archive.files:
                    raise ProofFailure("CONDITIONING_MISMATCH", f"{file_id}: NPZ omits {field_name}")
                array = np.asarray(archive[field_name])
                if array.shape != () or array.dtype.kind not in "iu":
                    raise ProofFailure("CONDITIONING_MISMATCH", f"{file_id}: NPZ {field_name} must be an integer scalar")
                value = int(array.item())
                if value < 0:
                    raise ProofFailure("CONDITIONING_MISMATCH", f"{file_id}: NPZ {field_name} must be nonnegative")
                result.append(value)
            return result[0], result[1]
    except ProofFailure:
        raise
    except Exception as exc:
        raise ProofFailure("CONDITIONING_MISMATCH", f"{file_id}: cannot decode serialized conditioning: {exc}") from exc


def tensor_sha256(array: Any) -> str:
    import numpy as np

    canonical = np.ascontiguousarray(array, dtype="<f4")
    return sha256_bytes(canonical.tobytes(order="C"))


@dataclass
class ToyRecord:
    file_id: str
    robot_id: str
    robot_idx: int
    program_id: str
    program_idx: int
    x: Any = field(repr=False)


def build_array(formula: str) -> Any:
    import numpy as np

    result = np.zeros((CHANNELS, 64), dtype=np.dtype("<f4"))
    for channel in range(CHANNELS):
        for timestep in range(64):
            if formula == "A":
                value = (((7 * timestep + 3 * channel) % 17) - 8) / 8
            elif formula == "B":
                value = (((5 * timestep + 11 * channel + 3) % 19) - 9) / 9
            else:
                raise ProofFailure("INPUT_CONTRACT_FAILURE", f"unknown fixture formula {formula!r}")
            result[channel, timestep] = np.float32(value)
    return np.ascontiguousarray(result, dtype="<f4")


def build_toy_roster() -> list[ToyRecord]:
    roster: list[ToyRecord] = []
    for spec in TOY_SPECS:
        x = build_array(spec["formula"])
        if tensor_sha256(x) != spec["tensor_sha256"]:
            raise ProofFailure("INPUT_CONTRACT_FAILURE", f"{spec['file_id']}: tensor digest differs from frozen contract")
        if sha256_bytes(spec["file_id"].encode("utf-8")) != spec["file_id_sha256"] or file_seed(spec["file_id"]) != spec["mask_seed"]:
            raise ProofFailure("MASK_CONTRACT_FAILURE", f"{spec['file_id']}: frozen file ID or mask seed differs")
        roster.append(ToyRecord(
            spec["file_id"], spec["robot_id"], spec["robot_idx"],
            spec["program_id"], spec["program_idx"], x,
        ))
    return roster


def prepare_toy_record(
    record: ToyRecord, mapping: dict[str, dict[str, int]], patchifier: Any,
    cfg: Any, *, masking_seed: int | None = None,
) -> tuple[dict[str, Any], int, list[int]]:
    import numpy as np
    from representation.data import collate_variable_files
    from synth.dataset import _sample_arrays, _sample_bytes
    from synth.schema import FileSample, SampleLabel

    validate_conditioning(record, mapping)
    validate_input(record)
    seed = file_seed(record.file_id) if masking_seed is None else masking_seed
    if type(seed) is not int or seed != file_seed(record.file_id):
        raise ProofFailure("MASK_CONTRACT_FAILURE", f"{record.file_id}: mask seed differs from its frozen file-ID seed")
    sample = FileSample(
        x=record.x, file_id=record.file_id, file_label=SampleLabel.NORMAL,
        seed=0, generator_version="S22-T1-BOUNDARY-MATH-FIXTURE-v1",
        config_hash="s22-t1-boundary-math-fixture-v1", regime_sequence=[],
        robot_idx=record.robot_idx, program_idx=record.program_idx,
    )
    payload = _sample_bytes(sample)
    serialized = read_serialized_conditioning(payload, record.file_id)
    if serialized != (record.robot_idx, record.program_idx):
        raise ProofFailure("CONDITIONING_MISMATCH", f"{record.file_id}: NPZ conditioning differs from fixture source fields")
    batch = collate_variable_files([sample], patchifier, masking_config=cfg, masking_seed=seed)
    valid = batch["patch_valid_mask"][0].detach().cpu().numpy()
    mask = batch["mask"][0].detach().cpu().numpy()
    bits = validate_mask(mask, valid, record.file_id)
    if len(bits) != 3:
        raise ProofFailure("MASK_CONTRACT_FAILURE", f"{record.file_id}: patchifier did not produce exactly three patch positions")
    return batch, seed, bits


def snapshot_state(model: Any, bank: Any) -> dict[str, str]:
    import torch

    values: dict[str, str] = {}
    for name, value in sorted(model.named_parameters()):
        values[f"parameter:{name}"] = sha256_bytes(value.detach().cpu().contiguous().numpy().tobytes())
    for name, value in sorted(model.named_buffers()):
        if isinstance(value, torch.Tensor):
            payload = value.detach().cpu().contiguous().numpy().tobytes()
        else:
            payload = repr(value).encode("utf-8")
        values[f"buffer:{name}"] = sha256_bytes(payload)
    values["bank:k"] = str(int(bank.k))
    if bank.embeddings is None:
        values["bank:embeddings"] = "NONE"
    else:
        values["bank:embeddings"] = sha256_bytes(bank.embeddings.detach().cpu().contiguous().numpy().tobytes())
    return values


def require_state_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    if before != after:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", "model parameter, buffer, or reference-bank state changed")


def restore_seed(checkpoint: Path, model_seed: int, config_map: dict[str, dict[str, object]], device: str) -> tuple[Any, Any, Any, Any, str, str]:
    import torch
    from representation.checkpoint import load_checkpoint
    from representation.config import V1Config
    from representation.inference import NormalReferenceBank
    from representation.model import V1RepresentationModel
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    cfg_dict = config_map[str(model_seed)]
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
        contrastive_max_shift=cfg_dict["contrastive_max_shift"], knn_k=cfg_dict["knn_k"], seed=model_seed,
    )
    patchifier = Patchifier(PatchConfig(
        patch_size=cfg.patch_size, stride=cfg.stride, pad_end=cfg_dict["pad_end"],
    ))
    model = V1RepresentationModel(cfg, patchifier=patchifier)
    bank = NormalReferenceBank(k=BANK_K)
    meta = load_checkpoint(checkpoint, model, reference_bank=bank)
    checkpoint_config_sha256 = validate_checkpoint_metadata(meta, model_seed, cfg.to_dict())
    if bank.embeddings is None or not bool(torch.isfinite(bank.embeddings).all()):
        raise ProofFailure("PROVENANCE_MISMATCH", f"seed {model_seed} embedded Fit bank is absent or non-finite")
    bank_digest = sha256_bytes(bank.embeddings.detach().cpu().contiguous().numpy().tobytes())
    validate_bank_identity(int(bank.k), tuple(bank.embeddings.shape), bank_digest, model_seed)
    model.to(device)
    model.eval()
    return model, bank, patchifier, cfg, bank_digest, checkpoint_config_sha256


def score_roster_once(
    model: Any, bank: Any, patchifier: Any, cfg: Any, roster: list[ToyRecord],
    mapping: dict[str, dict[str, int]], device: str,
) -> tuple[dict[str, dict[str, float]], dict[str, list[int]], dict[str, dict[str, float]]]:
    import torch
    from representation.inference import RepresentationInference

    inference = RepresentationInference(model, bank, patchifier, masking_config=cfg)
    scores: dict[str, dict[str, float]] = {}
    masks: dict[str, list[int]] = {}
    details: dict[str, dict[str, float]] = {}
    for record in roster:
        batch, seed, bits = prepare_toy_record(record, mapping, patchifier, cfg)
        masks[record.file_id] = bits
        model_batch = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        model.eval()
        with torch.inference_mode():
            result = inference.score_batch(model_batch)
        pred = float(torch.as_tensor(result["S_pred"]).reshape(-1)[0].item())
        pop = float(torch.as_tensor(result["S_pop"]).reshape(-1)[0].item())
        patch_scores = torch.as_tensor(result["patch_scores"])
        prediction_mask = torch.as_tensor(result["prediction_mask"])
        if patch_scores.shape != prediction_mask.shape:
            raise ProofFailure("ROSTER_OR_SCORE_INCOMPLETE", f"{record.file_id}: patch scores and prediction mask shapes differ")
        expected_pred = float((patch_scores.sum(dim=1) / prediction_mask.sum(dim=1).clamp_min(1).to(patch_scores.dtype)).reshape(-1)[0].item())
        embedding = torch.as_tensor(result["file_embedding"])
        distances = torch.cdist(embedding.float().cpu(), bank.embeddings.float().cpu())
        expected_pop = float(distances.topk(BANK_K, largest=False, dim=1).values.mean(dim=1).reshape(-1)[0].item())
        for branch, got, expected in (("S_pred", pred, expected_pred), ("S_pop", pop, expected_pop)):
            if not math.isfinite(got) or not math.isfinite(expected) or got < 0 or expected < 0:
                raise ProofFailure("ROSTER_OR_SCORE_INCOMPLETE", f"{record.file_id}/{branch}: score is not finite and nonnegative")
            if not math.isclose(got, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                raise ProofFailure("ROSTER_OR_SCORE_INCOMPLETE", f"{record.file_id}/{branch}: production score differs from independent recomputation")
        scores[record.file_id] = {"S_pred": pred, "S_pop": pop}
        details[record.file_id] = {
            "mask_seed": seed,
            "S_pred_recomputed": expected_pred,
            "S_pred_absolute_delta": abs(pred - expected_pred),
            "S_pop_recomputed": expected_pop,
            "S_pop_absolute_delta": abs(pop - expected_pop),
            "patch_scores_norm": float(patch_scores.detach().float().norm().item()),
            "file_embedding_norm": float(embedding.detach().float().norm().item()),
        }
    validate_scored_roster([record.file_id for record in roster], [
        {"file_id": file_id, **values} for file_id, values in scores.items()
    ])
    return scores, masks, details


def _roster_problem(expected_ids: list[str], rows: list[dict[str, object]], support: dict[str, list[str]] | None = None) -> str | None:
    ids = [row.get("file_id") for row in rows]
    if len(ids) != len(expected_ids) or len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        return "ROSTER_OR_SCORE_INCOMPLETE"
    for row in rows:
        if set(row) != {"file_id", "S_pred", "S_pop"}:
            return "ROSTER_OR_SCORE_INCOMPLETE"
        for branch in ("S_pred", "S_pop"):
            value = row[branch]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                return "ROSTER_OR_SCORE_INCOMPLETE"
    if support is not None:
        if set(support) != {"S_pred", "S_pop"} or sorted(support["S_pred"]) != sorted(support["S_pop"]):
            return "ROSTER_OR_SCORE_INCOMPLETE"
    return None


def full_roster_result(expected_ids: list[str], rows: list[dict[str, object]], support: dict[str, list[str]] | None = None) -> dict[str, object]:
    issue = _roster_problem(expected_ids, rows, support)
    if issue is not None:
        return {
            "status": "UNCOMPUTABLE_FULL_ROSTER",
            "disposition": issue,
            "branch_values": {"S_pred": None, "S_pop": None},
            "aggregate": None,
        }
    return {
        "status": "COMPUTABLE_FULL_ROSTER",
        "disposition": None,
        "branch_values": {
            branch: {str(row["file_id"]): float(row[branch]) for row in rows}
            for branch in ("S_pred", "S_pop")
        },
        "aggregate": None,
    }


def validate_scored_roster(expected_ids: list[str], rows: list[dict[str, object]], support: dict[str, list[str]] | None = None) -> None:
    issue = _roster_problem(expected_ids, rows, support)
    if issue is not None:
        raise ProofFailure(issue, "scored file roster, branches, values, or support do not match the declared full roster")


def verify_allocator_source_path(worktree_root: Path) -> tuple[Any, Any]:
    balanced = importlib.import_module("synth.balanced")
    events = importlib.import_module("synth.events")
    for name, module in (("synth.balanced", balanced), ("synth.events", events)):
        validate_module_origin(Path(module.__file__), (worktree_root / "src").resolve(), name)
    return balanced, events


def _allocator_row(file_id: str, operation_id: str, robot_id: str, program_id: str, end_time: float, *, quarantined: bool = False) -> dict[str, object]:
    return {
        "file_id": file_id,
        "operation_id": operation_id,
        "robot_id": robot_id,
        "program_id": program_id,
        "start_time": end_time - 3600.0,
        "end_time": end_time,
        "file_label": "normal",
        "is_quarantined": quarantined,
        "quarantine_reason": "S22 disposable robot-day sentinel" if quarantined else None,
        "is_censored": False,
        "member_views": ["test-temporal"],
        "last_reset_time": 0.0,
        "n_valid_patches": 1,
    }


def build_allocator_fixture(control_count: int) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[list[float]]], set[str], set[str]]:
    if control_count not in (47, 48, 49):
        raise ValueError("frozen allocator controls are 47, 48, or 49")
    day = 86400.0
    wins: dict[str, list[list[float]]] = {f"robot-{index:02d}": [] for index in range(1, 10)}
    rows: list[dict[str, object]] = []
    ledger: list[dict[str, object]] = []
    baseline_ids: set[str] = set()
    buckets = (("P", "P1", 12), ("P", "P2", 12), ("W", "W1", 12), ("W", "W2", 12), ("A", "A1", 8), ("A", "A2", 8))
    robot_ordinal = Counter()
    max_failure_time = 0.0
    for cohort, subtype, quota in buckets:
        for k in range(1, quota + 1):
            failure_id = f"S22-ALLOC-{cohort}-{subtype}-{k:02d}"
            robot_id = f"robot-{1 + ((k - 1) % 6):02d}"
            if cohort == "A":
                program_id = "program-01" if k <= 4 else "program-02"
            else:
                program_id = "program-01" if k <= 6 else "program-02"
            robot_ordinal[robot_id] += 1
            failure_time = float(1000 + 50 * robot_ordinal[robot_id]) * day
            max_failure_time = max(max_failure_time, failure_time)
            duration = 5.0 if cohort == "P" else 12.0 if cohort == "W" else 0.0
            ledger.append({
                "failure_id": failure_id,
                "robot_id": robot_id,
                "cohort": cohort,
                "subtype": subtype,
                "severity": 1.0,
                "failure_time": failure_time,
                "degradation_onset": failure_time - duration * day if cohort != "A" else None,
                "duration_d": duration,
            })
            endpoints = (("POS-01", -6.0), ("POS-02", -3.0), ("POS-03", 0.0)) if cohort in ("P", "W") else (("POS-01", 0.0),)
            for ordinal, offset in endpoints:
                rows.append(_allocator_row(
                    f"{failure_id}-{ordinal}", f"{failure_id}-OP-{ordinal}",
                    robot_id, program_id, failure_time + offset * day,
                ))
            if cohort in ("P", "W") and k <= 10:
                baseline_id = f"{failure_id}-BASE"
                rows.append(_allocator_row(
                    baseline_id, f"{failure_id}-OP-BASE", robot_id, program_id,
                    failure_time - 16.0 * day,
                ))
                baseline_ids.add(baseline_id)
    for j in range(1, 241):
        robot_id = f"robot-{1 + ((j - 1) % 9):02d}"
        rows.append(_allocator_row(
            f"S22-ALLOC-ROBOTDAY-{j:03d}", f"S22-ALLOC-ROBOTDAY-OP-{j:03d}",
            robot_id, "program-01", (j - 0.5) * day, quarantined=True,
        ))
    extra_ids: set[str] = set()
    for j in range(1, control_count - 39):
        extra_id = f"S22-ALLOC-EXTRA-CONTROL-{j:02d}"
        robot_id = f"robot-{1 + ((j - 1) % 9):02d}"
        program_id = "program-01" if j % 2 else "program-02"
        end_time = (max_failure_time / day + 100 + 8 * (j - 1)) * day
        rows.append(_allocator_row(extra_id, f"S22-ALLOC-EXTRA-OP-{j:02d}", robot_id, program_id, end_time))
        extra_ids.add(extra_id)
    if len(rows) != 400 + control_count or len(ledger) != 64 or len(baseline_ids) != 40:
        raise ProofFailure("ALLOCATOR_FIXTURE_FAILURE", f"fixture dimensions drifted: rows={len(rows)}, ledger={len(ledger)}")
    return rows, ledger, wins, baseline_ids, extra_ids


def consume_successful_allocation(record: dict[str, Any], consumer: Any) -> dict[str, Any]:
    try:
        consumed = consumer.require_minimum_control_allocation(record)
    except Exception as exc:
        code = getattr(exc, "code", "ALLOCATOR_CONSUMER_FAILURE")
        raise ProofFailure(str(code), f"new production consumer rejected allocation: {exc}") from exc
    if consumed is not record:
        raise ProofFailure("ALLOCATOR_CONSUMER_FAILURE", "production consumer replaced or narrowed the allocator result")
    return consumed


def prove_allocator_fixtures(balanced: Any, events: Any, consumer: Any) -> dict[str, object]:
    results: dict[str, object] = {}
    for count in (47, 48, 49):
        rows, ledger, wins, baseline_ids, extra_ids = build_allocator_fixture(count)
        expected_anchor_ids = baseline_ids | extra_ids
        selected_controls = events.select_control_windows(events.anchor_rows(rows, wins), ledger, wins)
        if len(selected_controls) != count or {member["file_id"] for window in selected_controls for member in window["members"]} != expected_anchor_ids:
            raise ProofFailure("ALLOCATOR_FIXTURE_FAILURE", f"pinned event selectors produced an unexpected {count}-control pool")
        try:
            record = balanced.allocate_quotas(
                rows, ledger, wins, FIXTURE_SEED, balanced.QuotaConfig(), method="exact",
            )
        except Exception as exc:
            failure = getattr(exc, "record", {})
            if count != 47:
                raise ProofFailure("ALLOCATOR_FIXTURE_FAILURE", f"real allocator unexpectedly failed for {count}: {type(exc).__name__}: {exc}") from exc
            failure_text = str(exc)
            if type(exc).__name__ != "InfeasibleCandidate" or failure.get("reason") != "control-shortfall" or f"{count} < 48" not in failure_text:
                raise ProofFailure("ALLOCATOR_FIXTURE_FAILURE", f"47-control allocator failure differed from frozen shortfall: {failure!r}; {failure_text}") from exc
            results[str(count)] = {
                "status": "EXPECTED_INFEASIBLE",
                "exception_type": type(exc).__name__,
                "reason": failure.get("reason"),
                "missing": failure.get("missing"),
                "eligible_control_count": count,
                "hard_control_floor_25_passes": count >= 25,
                "diagnostic_margin_32_passes": count >= 32,
                "successful_allocation": False,
            }
            continue
        if count == 47:
            raise ProofFailure("ALLOCATOR_FIXTURE_FAILURE", "real allocator returned success for 47 controls")
        record = consume_successful_allocation(record, consumer)
        selected_ids = set(record.get("selected_ids", []))
        expected_ids = {failure["failure_id"] for failure in ledger}
        selected = record.get("selected", {})
        expected_subtypes = {"P": {"P1": 12, "P2": 12}, "W": {"W1": 12, "W2": 12}, "A": {"A1": 8, "A2": 8}}
        observed_subtypes = {cohort: {subtype: len(ids) for subtype, ids in groups.items()} for cohort, groups in selected.items()}
        controls = record.get("controls", [])
        actual_control_ids = {member_id for window in controls for member_id in window["members"]}
        expected_control_ids = baseline_ids | extra_ids
        eligible, rejection, lead = balanced.eligible_anchors(rows, ledger, wins)
        eligible_by_cohort = Counter(item["cohort"] for item in eligible)
        baseline_support = {
            cohort: sum(bool(lead[failure["failure_id"]].get("clean_baseline")) for failure in ledger if failure["cohort"] == cohort)
            for cohort in ("P", "W")
        }
        margins = record.get("margins", {})
        quota = balanced.QuotaConfig()
        selected_by_id = {failure["failure_id"]: failure for failure in ledger}
        chosen = [selected_by_id[failure_id] for failure_id in selected_ids]
        positive_by_robot = Counter(failure["robot_id"] for failure in chosen)
        robot_times: dict[str, list[float]] = {}
        for failure in chosen:
            robot_times.setdefault(failure["robot_id"], []).append(float(failure["failure_time"]))
        minimum_spacing_by_robot: dict[str, float] = {}
        for robot_id, times in robot_times.items():
            ordered_times = sorted(times)
            if len(ordered_times) > 1:
                minimum_spacing_by_robot[robot_id] = min(right - left for left, right in zip(ordered_times, ordered_times[1:]))
        robot_spacing_pass = all(value >= quota.spacing_s for value in minimum_spacing_by_robot.values())
        positive_robot_share_max = max(positive_by_robot.values(), default=0) / len(chosen)
        cohort_counts = Counter(failure["cohort"] for failure in chosen)
        cohort_shares = {cohort: cohort_counts[cohort] / len(chosen) for cohort in ("P", "W", "A")}
        control_robots = Counter(window["robot_id"] for window in controls)
        negative_robot_share_max = max(control_robots.values(), default=0) / len(controls)
        cohort_mix_pass = cohort_shares == {"P": 0.375, "W": 0.375, "A": 0.25}
        program_by_endpoint = {(row["robot_id"], row["end_time"]): row["program_id"] for row in rows}
        program_counts: dict[str, Counter] = {"P": Counter(), "W": Counter()}
        for failure in chosen:
            if failure["cohort"] in program_counts:
                program = program_by_endpoint.get((failure["robot_id"], failure["failure_time"]))
                if program is None:
                    raise ProofFailure("ALLOCATOR_FIXTURE_FAILURE", f"{failure['failure_id']} has no exact-endpoint program mapping")
                program_counts[failure["cohort"]][program] += 1
        program_shares = {
            cohort: {program: count / sum(program_counts[cohort].values()) for program, count in program_counts[cohort].items()}
            for cohort in ("P", "W")
        }
        program_cap_pass = all(max(shares.values(), default=0) <= quota.program_cap for shares in program_shares.values())
        duration_days = {
            cohort: sorted({float(failure["duration_d"]) for failure in ledger if failure["cohort"] == cohort})
            for cohort in ("P", "W")
        }
        duration_shape_pass = duration_days == {"P": [5.0], "W": [12.0]}
        if (
            selected_ids != expected_ids or record.get("quota") != quota.as_dict()
            or observed_subtypes != expected_subtypes
            or len(controls) != count or margins.get("controls") != count
            or actual_control_ids != expected_control_ids or len(actual_control_ids) != count
            or record.get("alloc_seed") != balanced.alloc_seed_for(FIXTURE_SEED)
            or margins.get("robot_days", 0) < 240
            or margins.get("robots_pos", 0) < 6 or margins.get("robots_neg", 0) < 6
            or margins.get("programs_P", 0) < 2 or margins.get("programs_W", 0) < 2
            or eligible_by_cohort != Counter({"P": 24, "W": 24, "A": 16})
            or not cohort_mix_pass
            or negative_robot_share_max > quota.robot_neg_cap
            or baseline_support != {"P": 20, "W": 20}
            or not robot_spacing_pass or positive_robot_share_max > quota.robot_pos_cap
            or not program_cap_pass or not duration_shape_pass or rejection
        ):
            raise ProofFailure("ALLOCATOR_FIXTURE_FAILURE", f"{count}-control actual allocation violated selected quotas, spacing/caps, support, seed, or full-control retention")
        if not all(
            len(window["members"]) == 1 and window["members"][0] in expected_control_ids
            for window in controls
        ):
            raise ProofFailure("ALLOCATOR_FIXTURE_FAILURE", f"{count}-control result contains non-singleton or undeclared control windows")
        results[str(count)] = {
            "status": "PASS_MINIMUM_48_ALL_QUALIFYING_CONTROLS_RETAINED",
            "eligible_control_count": count,
            "retained_control_count": len(controls),
            "retained_control_member_ids_sha256": canonical_sha256(sorted(actual_control_ids)),
            "all_64_positive_failures_selected": len(selected_ids),
            "selected_subtypes": observed_subtypes,
            "alloc_seed": record["alloc_seed"],
            "margins": margins,
            "minimum_selected_spacing_s_by_robot": minimum_spacing_by_robot,
            "positive_robot_share_max": positive_robot_share_max,
            "positive_robot_share_cap": quota.robot_pos_cap,
            "program_shares_P_W": program_shares,
            "program_share_cap": quota.program_cap,
            "cohort_shares": cohort_shares,
            "negative_robot_share_max": negative_robot_share_max,
            "negative_robot_share_cap": quota.robot_neg_cap,
            "selected_duration_days_P_W": duration_days,
            "eligible_events_by_cohort": dict(eligible_by_cohort),
            "clean_baseline_lead_support": {cohort: {"passed": baseline_support[cohort], "eligible": 24} for cohort in ("P", "W")},
            "control_construction_requirement": "minimum 48; record controls returned unmodified to production consumer",
            "hard_control_floor_25_passes": True,
            "diagnostic_margin_32_passes": True,
        }
    return results


def _expect_rejection(name: str, code: str, function: Any) -> dict[str, str]:
    try:
        function()
    except ProofFailure as exc:
        if exc.code != code:
            raise ProofFailure("NEGATIVE_CASE_FAILURE", f"{name}: actual rejection {exc.code} differs from expected {code}") from exc
        return {"case": name, "disposition": exc.code, "trigger": str(exc)}
    except Exception as exc:
        raise ProofFailure("NEGATIVE_CASE_FAILURE", f"{name}: unexpected rejection type {type(exc).__name__}: {exc}") from exc
    raise ProofFailure("NEGATIVE_CASE_FAILURE", f"{name}: mutation did not trigger rejection")


def run_negative_cases(roster: list[ToyRecord], mapping: dict[str, dict[str, int]], worktree_root: Path) -> list[dict[str, str]]:
    import numpy as np
    from synth.dataset import _sample_arrays
    from synth.schema import FileSample, SampleLabel

    first = roster[0]
    second = roster[1]
    cases: list[dict[str, str]] = []
    cases.append(_expect_rejection("unknown-robot", "CONDITIONING_MISMATCH", lambda: validate_conditioning(copy.copy(first).__class__(first.file_id, "robot-99", 99, first.program_id, first.program_idx, first.x), mapping)))
    cases.append(_expect_rejection("unknown-program", "CONDITIONING_MISMATCH", lambda: validate_conditioning(ToyRecord(first.file_id, first.robot_id, first.robot_idx, "program-99", 99, first.x), mapping)))
    cases.append(_expect_rejection("robot-string-index-disagreement", "CONDITIONING_MISMATCH", lambda: validate_conditioning(ToyRecord(first.file_id, first.robot_id, 8, first.program_id, first.program_idx, first.x), mapping)))
    cases.append(_expect_rejection("program-string-index-disagreement", "CONDITIONING_MISMATCH", lambda: validate_conditioning(ToyRecord(first.file_id, first.robot_id, first.robot_idx, second.program_id, first.program_idx, first.x), mapping)))
    sample = FileSample(
        x=first.x, file_id=first.file_id, file_label=SampleLabel.NORMAL, seed=0,
        generator_version="S22-T1-BOUNDARY-MATH-FIXTURE-v1", config_hash="s22-t1-boundary-math-fixture-v1",
        regime_sequence=[], robot_idx=first.robot_idx, program_idx=first.program_idx,
    )
    arrays = _sample_arrays(sample)
    for missing in ("robot_idx", "program_idx"):
        incomplete = dict(arrays)
        incomplete.pop(missing)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **incomplete)
        cases.append(_expect_rejection(f"missing-serialized-{missing}", "CONDITIONING_MISMATCH", lambda payload=buffer.getvalue(): read_serialized_conditioning(payload, first.file_id)))
    bad_channels = copy.copy(first)
    bad_channels.x = np.zeros((5, 64), dtype=np.float32)
    cases.append(_expect_rejection("wrong-channel-count", "INPUT_CONTRACT_FAILURE", lambda: validate_input(bad_channels)))
    bad_dtype = copy.copy(first)
    bad_dtype.x = np.asarray(first.x, dtype=np.float64)
    cases.append(_expect_rejection("wrong-input-dtype", "INPUT_CONTRACT_FAILURE", lambda: validate_input(bad_dtype)))
    bad_finite = copy.copy(first)
    bad_finite.x = np.array(first.x, copy=True)
    bad_finite.x[0, 0] = np.inf
    cases.append(_expect_rejection("non-finite-input", "INPUT_CONTRACT_FAILURE", lambda: validate_input(bad_finite)))
    valid_three = np.array([True, True, True])
    cases.append(_expect_rejection("wrong-mask-cardinality", "MASK_CONTRACT_FAILURE", lambda: validate_mask(np.array([True, True, False]), valid_three, first.file_id)))
    cases.append(_expect_rejection("masked-padding", "MASK_CONTRACT_FAILURE", lambda: validate_mask(np.array([True, False, False, True]), np.array([True, True, True, False]), first.file_id)))
    # Trigger the production seed guard before the collator is called.
    try:
        prepare_toy_record(first, mapping, None, None, masking_seed=file_seed(first.file_id) + 1)
    except ProofFailure as exc:
        if exc.code != "MASK_CONTRACT_FAILURE":
            raise ProofFailure("NEGATIVE_CASE_FAILURE", f"wrong-mask-seed: actual disposition {exc.code}") from exc
        cases.append({"case": "wrong-mask-seed", "disposition": exc.code, "trigger": str(exc)})
    else:
        raise ProofFailure("NEGATIVE_CASE_FAILURE", "wrong-mask-seed did not trigger rejection")
    expected_ids = [record.file_id for record in roster]
    complete = [{"file_id": first.file_id, "S_pred": 0.1, "S_pop": 0.2}, {"file_id": second.file_id, "S_pred": 0.3, "S_pop": 0.4}]
    missing = [complete[0]]
    duplicate = [complete[0], dict(complete[0])]
    extra = [*complete, {"file_id": "S22T1-BND-EXTRA", "S_pred": 0.1, "S_pop": 0.2}]
    missing_branch = [{"file_id": first.file_id, "S_pred": 0.1}, complete[1]]
    nonfinite = [dict(complete[0], S_pred=float("nan")), complete[1]]
    negative = [dict(complete[0], S_pop=-0.1), complete[1]]
    for name, rows in (("missing-roster-member", missing), ("duplicate-roster-member", duplicate), ("extra-roster-member", extra), ("absent-score-branch", missing_branch), ("nonfinite-score", nonfinite), ("negative-score", negative)):
        result = full_roster_result(expected_ids, rows)
        if result["status"] != "UNCOMPUTABLE_FULL_ROSTER" or result["branch_values"] != {"S_pred": None, "S_pop": None} or result["aggregate"] is not None:
            raise ProofFailure("NEGATIVE_CASE_FAILURE", f"{name}: result rescued a subset or retained a partial branch")
        cases.append({"case": name, "disposition": str(result["disposition"]), "trigger": "full-roster validation returned both branches and aggregate null"})
    support_result = full_roster_result(expected_ids, complete, {"S_pred": ["event-1", "control-1"], "S_pop": ["event-1", "control-2"]})
    if support_result["status"] != "UNCOMPUTABLE_FULL_ROSTER" or support_result["branch_values"] != {"S_pred": None, "S_pop": None}:
        raise ProofFailure("NEGATIVE_CASE_FAILURE", "branch/support mismatch was accepted or partially rescued")
    cases.append({"case": "branch-support-key-disagreement", "disposition": str(support_result["disposition"]), "trigger": "score-independent support differs between branches"})
    cases.append(_expect_rejection("missing-checkpoint", "PROVENANCE_MISMATCH", lambda: verify_checkpoint_digest(worktree_root / "missing-checkpoint-sentinel.pt", MODEL_SEEDS[0])))
    cases.append(_expect_rejection("wrong-checkpoint-size", "PROVENANCE_MISMATCH", lambda: verify_checkpoint_digest(worktree_root / "experiments/sprint17_task7_b0.py", MODEL_SEEDS[0], expected_size=CHECKPOINT_BYTES)))
    cases.append(_expect_rejection("wrong-checkpoint-hash", "PROVENANCE_MISMATCH", lambda: verify_checkpoint_digest(worktree_root / "experiments/sprint17_task7_b0.py", MODEL_SEEDS[0], expected_size=(worktree_root / "experiments/sprint17_task7_b0.py").stat().st_size, expected_sha256="0" * 64)))
    cases.append(_expect_rejection("wrong-checkpoint-step", "PROVENANCE_MISMATCH", lambda: validate_checkpoint_metadata({"step": 299, "has_reference_bank": True, "config": {"seed": MODEL_SEEDS[0]}}, MODEL_SEEDS[0], {"seed": MODEL_SEEDS[0]})))
    cases.append(_expect_rejection("wrong-checkpoint-config", "PROVENANCE_MISMATCH", lambda: validate_checkpoint_metadata({"step": CHECKPOINT_STEP, "has_reference_bank": True, "config": {"seed": MODEL_SEEDS[0], "changed": True}}, MODEL_SEEDS[0], {"seed": MODEL_SEEDS[0]})))
    cases.append(_expect_rejection("wrong-checkpoint-seed", "PROVENANCE_MISMATCH", lambda: validate_checkpoint_metadata({"step": CHECKPOINT_STEP, "has_reference_bank": True, "config": {"seed": MODEL_SEEDS[1]}}, MODEL_SEEDS[0], {"seed": MODEL_SEEDS[0]})))
    cases.append(_expect_rejection("wrong-bank-k", "PROVENANCE_MISMATCH", lambda: validate_bank_identity(4, (BANK_ROWS, BANK_DIM), BANK_SHA256[MODEL_SEEDS[0]], MODEL_SEEDS[0])))
    cases.append(_expect_rejection("wrong-bank-shape", "PROVENANCE_MISMATCH", lambda: validate_bank_identity(BANK_K, (BANK_ROWS - 1, BANK_DIM), BANK_SHA256[MODEL_SEEDS[0]], MODEL_SEEDS[0])))
    cases.append(_expect_rejection("wrong-bank-digest", "PROVENANCE_MISMATCH", lambda: validate_bank_identity(BANK_K, (BANK_ROWS, BANK_DIM), "0" * 64, MODEL_SEEDS[0])))
    cases.append(_expect_rejection("source-digest-drift", "RESTORE_ONLY_OR_ISOLATION_FAILURE", lambda: validate_source_digest("src/representation/model.py", "0" * 64, SOURCE_HASHES["src/representation/model.py"])))
    cases.append(_expect_rejection("module-outside-pinned-source", "RESTORE_ONLY_OR_ISOLATION_FAILURE", lambda: validate_module_origin(Path("/tmp/outside/module.py"), (worktree_root / "src").resolve(), "synth.example")))
    cases.append(_expect_rejection("runtime-drift", "RESTORE_ONLY_OR_ISOLATION_FAILURE", lambda: validate_runtime({"python": "3.12.13", "torch": "2.14.0+cu130", "cuda_available": False, "device_name": None, "device_total_mib": None})))
    before = {"parameter:weight": "a", "buffer:norm": "b", "bank:k": "5", "bank:embeddings": "c"}
    for name, field_name in (("model-parameter-mutated", "parameter:weight"), ("normalization-buffer-mutated", "buffer:norm"), ("bank-k-mutated", "bank:k"), ("bank-embeddings-mutated", "bank:embeddings")):
        after = dict(before, **{field_name: f"changed-{field_name}"})
        cases.append(_expect_rejection(name, "RESTORE_ONLY_OR_ISOLATION_FAILURE", lambda snapshot=after: require_state_unchanged(before, snapshot)))
    return cases



def preflight(args: argparse.Namespace) -> dict[str, object]:
    contract_path = Path(args.contract_path).resolve()
    contract_hash = verify_reviewed_contract(contract_path)
    worktree_root = Path(args.worktree_root).resolve()
    wrapper_path = Path(__file__).resolve()
    wrapper_hash = verify_wrapper_identity(wrapper_path, args.expected_wrapper_sha256)
    historical = verify_historical_tree(worktree_root, wrapper_hash)
    runner_path = Path(args.runner_path).resolve()
    runner_hash = verify_wrapper_identity(runner_path, args.expected_runner_sha256)
    checkpoint_root = Path(args.checkpoint_root).resolve()
    if checkpoint_root != CHECKPOINT_ROOT.resolve():
        raise ProofFailure("PROVENANCE_MISMATCH", f"checkpoint root {checkpoint_root} is not the authorized original path {CHECKPOINT_ROOT}")
    if args.device != "cuda":
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", "the frozen proof requires CUDA; CPU fallback is forbidden")
    if Path(args.output_dir).resolve() != OUTPUT_ROOT.resolve():
        raise ProofFailure("OUTPUT_PATH_MISMATCH", f"output root must be exactly {OUTPUT_ROOT}")
    output_dir = Path(args.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ProofFailure("OUTPUT_ALREADY_EXISTS", f"one-shot output root already exists: {output_dir}")
    # This module is a pure stdlib file at import time; only this helper is
    # called from it. Its digest is compared with an independently supplied
    # hash from the pushed Task 2 source commit.
    spec = importlib.util.spec_from_file_location("_s22_pilot_runner_consumer", runner_path)
    if spec is None or spec.loader is None:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"cannot load new production runner at {runner_path}")
    consumer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consumer)
    if getattr(consumer, "RUNNER_ID", None) != "sprint22-pilot-runner-v1":
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", "runner identity differs from reviewed Sprint 22 version")
    checkpoint_preflight: dict[str, dict[str, object]] = {}
    for model_seed in MODEL_SEEDS:
        path = checkpoint_root / CHECKPOINT_FILES[model_seed]
        actual_digest = verify_checkpoint_digest(path, model_seed)
        checkpoint_preflight[str(model_seed)] = {
            "path": str(path), "bytes": path.stat().st_size, "sha256": actual_digest,
        }
    runtime = runtime_info()
    config_map = load_b0_config_map(worktree_root)
    mapping = derive_conditioning_mapping(worktree_root)
    assert_project_modules_pinned(worktree_root)
    balanced, events = verify_allocator_source_path(worktree_root)
    assert_project_modules_pinned(worktree_root)
    allocator_started = time.perf_counter()
    allocator_results = prove_allocator_fixtures(balanced, events, consumer)
    allocator_elapsed = time.perf_counter() - allocator_started
    return {
        "contract_sha256": contract_hash,
        "wrapper_sha256": wrapper_hash,
        "pilot_runner_sha256": runner_hash,
        "worktree": str(worktree_root),
        "checkpoint_root": str(checkpoint_root),
        "output_root": str(output_dir),
        "source": historical,
        "b0_config_sha256": canonical_sha256(config_map),
        "conditioning_mapping_sha256": canonical_sha256(mapping),
        "conditioning_mapping": mapping,
        "runtime": runtime,
        "checkpoint_preflight": checkpoint_preflight,
        "allocator_cases": allocator_results,
        "allocator_fixture_walltime_s": round(allocator_elapsed, 6),
        "consumer_helper": "sprint22_pilot_runner.require_minimum_control_allocation (loaded only after exact SHA-256 verification)",
    }


def _measure_gpu(torch: Any) -> dict[str, float]:
    return {
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2),
    }


def execute_scoring(args: argparse.Namespace, context: dict[str, object]) -> dict[str, object]:
    import torch

    worktree_root = Path(args.worktree_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    mapping = context["conditioning_mapping"]
    config_map = load_b0_config_map(worktree_root)
    roster = build_toy_roster()
    for record in roster:
        validate_conditioning(record, mapping)
        validate_input(record)
    negative_cases = run_negative_cases(roster, mapping, worktree_root)
    per_seed: dict[str, object] = {}
    total_started = time.perf_counter()
    for model_seed in MODEL_SEEDS:
        checkpoint = checkpoint_root / CHECKPOINT_FILES[model_seed]
        digest_before = verify_checkpoint_digest(checkpoint, model_seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        seed_started = time.perf_counter()
        model, bank, patchifier, cfg, bank_digest, config_digest = restore_seed(checkpoint, model_seed, config_map, "cuda")
        assert_project_modules_pinned(worktree_root)
        before = snapshot_state(model, bank)
        scores, masks, details = score_roster_once(model, bank, patchifier, cfg, roster, mapping, "cuda")
        repeated, repeated_masks, _ = score_roster_once(model, bank, patchifier, cfg, roster, mapping, "cuda")
        reversed_scores, reversed_masks, _ = score_roster_once(model, bank, patchifier, cfg, list(reversed(roster)), mapping, "cuda")
        for record in roster:
            file_id = record.file_id
            if masks[file_id] != repeated_masks[file_id] or masks[file_id] != reversed_masks[file_id]:
                raise ProofFailure("MASK_CONTRACT_FAILURE", f"seed {model_seed}/{file_id}: mask changed on repetition or reversed roster")
            for branch in ("S_pred", "S_pop"):
                expected = scores[file_id][branch]
                for other in (repeated, reversed_scores):
                    if not math.isclose(other[file_id][branch], expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                        raise ProofFailure("ROSTER_OR_SCORE_INCOMPLETE", f"seed {model_seed}/{file_id}/{branch}: score changed on repeat/order reversal")
        after = snapshot_state(model, bank)
        require_state_unchanged(before, after)
        if sha256_file(checkpoint) != digest_before or digest_before != CHECKPOINT_SHA256[model_seed]:
            raise ProofFailure("PROVENANCE_MISMATCH", f"seed {model_seed}: checkpoint bytes changed during scoring")
        if bank_digest != BANK_SHA256[model_seed] or bank.k != BANK_K or tuple(bank.embeddings.shape) != (BANK_ROWS, BANK_DIM):
            raise ProofFailure("PROVENANCE_MISMATCH", f"seed {model_seed}: restored Fit-only bank identity changed")
        seed_gpu = _measure_gpu(torch)
        per_seed[str(model_seed)] = {
            "checkpoint": {
                "filename": CHECKPOINT_FILES[model_seed], "bytes": checkpoint.stat().st_size,
                "sha256_before": digest_before, "sha256_after": sha256_file(checkpoint),
                "step": CHECKPOINT_STEP, "config_sha256": config_digest,
            },
            "bank": {
                "k": int(bank.k), "shape": list(bank.embeddings.shape),
                "sha256_before": bank_digest,
                "sha256_after": after["bank:embeddings"],
                "expected_sha256": BANK_SHA256[model_seed],
            },
            "model_state_before": before,
            "model_state_after": after,
            "model_state_before_sha256": canonical_sha256(before),
            "model_state_after_sha256": canonical_sha256(after),
            "scores": scores,
            "masks": masks,
            "score_recomputations": details,
            "repeat_and_reverse_stable": True,
            "gpu_resources": seed_gpu,
            "walltime_s": round(time.perf_counter() - seed_started, 6),
            "negative_cases": negative_cases if model_seed == MODEL_SEEDS[0] else [],
        }
        del model, bank, patchifier, cfg
        torch.cuda.empty_cache()
    full_roster = full_roster_result(
        [record.file_id for record in roster],
        [{"file_id": record.file_id, **per_seed["171701"]["scores"][record.file_id]} for record in roster],
        {"S_pred": [record.file_id for record in roster], "S_pop": [record.file_id for record in roster]},
    )
    if full_roster["status"] != "COMPUTABLE_FULL_ROSTER":
        raise ProofFailure("ROSTER_OR_SCORE_INCOMPLETE", "complete scorer fixture roster failed its full-roster validation")
    return {
        "schema_id": "sprint22-disposable-boundary-proof-v1",
        "status": "PASS_DISPOSABLE_ONLY",
        "contract_sha256": context["contract_sha256"],
        "pilot_runner_sha256": context["pilot_runner_sha256"],
        "historical_base_commit": BASE_COMMIT,
        "historical_worktree": str(worktree_root),
        "checkpoint_root": str(checkpoint_root),
        "source_provenance": context["source"],
        "runtime": context["runtime"],
        "config_sha256": context["b0_config_sha256"],
        "conditioning_mapping_sha256": context["conditioning_mapping_sha256"],
        "conditioning_mapping": context["conditioning_mapping"],
        "allocator_fixture": {
            "family": "S22-ALLOC-MIN48-v1",
            "history_seed_sentinel": FIXTURE_SEED,
            "history_seed_use": "allocator-only disposable sentinel; never a data or pilot seed",
            "walltime_s": context["allocator_fixture_walltime_s"],
            "cases": context["allocator_cases"],
        },
        "scoring_fixture": {
            "family": "S22-T1-SCORE-v1",
            "fixture_ids": [record.file_id for record in roster],
            "tensor_sha256": {spec["file_id"]: spec["tensor_sha256"] for spec in TOY_SPECS},
            "file_mask_seed": {spec["file_id"]: spec["mask_seed"] for spec in TOY_SPECS},
            "full_roster_result": full_roster,
            "per_seed": per_seed,
        },
        "negative_cases": negative_cases,
        "fixture_gpu_resources": {
            "device_name": context["runtime"]["device_name"],
            "device_total_mib": context["runtime"]["device_total_mib"],
            "peak_allocated_mib": max(float(item["gpu_resources"]["peak_allocated_mib"]) for item in per_seed.values()),
            "peak_reserved_mib": max(float(item["gpu_resources"]["peak_reserved_mib"]) for item in per_seed.values()),
        },
        "fixture_walltime_s": round(time.perf_counter() - total_started, 6),
        "real_history_resources": "unknown; no real history was generated or scored",
        "scope_limits": [
            "No pilot roster, fresh seed binding, history generation, or history contact was created.",
            "The allocator's synthetic rows and the two synthetic scorer samples are in-memory disposable fixtures only.",
            "This proof makes no tolerance, gate, history-level scientific, or real-history resource claim.",
            "Sprint 21 runner, protocol, binding, results, and seeds were not modified or reused.",
        ],
    }


def _load_consumer(path: Path, expected_sha256: str) -> Any:
    actual = verify_wrapper_identity(path.resolve(), expected_sha256)
    spec = importlib.util.spec_from_file_location("_s22_pilot_runner_consumer", path.resolve())
    if spec is None or spec.loader is None:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"cannot load production consumer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "RUNNER_ID", None) != "sprint22-pilot-runner-v1" or actual != expected_sha256:
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", "production consumer module identity/hash mismatch")
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-path", required=True)
    parser.add_argument("--worktree-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--runner-path", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # Contract byte identity is the first validation performed by main;
        # no model, allocator, source module, or output path is touched first.
        verify_reviewed_contract(Path(args.contract_path))
        context = preflight(args)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=False, exist_ok=False)
        stdout_path = output_dir / "command.stdout.log"
        stderr_path = output_dir / "command.stderr.log"
        started = time.perf_counter()
        payload: bytes | None = None
        with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout, stderr_path.open("x", encoding="utf-8", newline="\n") as stderr:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    record = execute_scoring(args, context)
                    record["command_walltime_s_including_preflight"] = round(time.perf_counter() - started, 6)
                    payload = json.dumps(record, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
                    if len(payload) > 2 * 1024 * 1024:
                        raise ProofFailure("OUTPUT_LIMIT_FAILURE", "proof JSON exceeds the frozen 2 MiB cap")
                    print(json.dumps({"status": record["status"], "fixture_walltime_s": record["fixture_walltime_s"], "output_bytes": len(payload)}, sort_keys=True))
                except Exception as exc:
                    print(json.dumps({"status": "FAIL_CLOSED", "code": getattr(exc, "code", "UNEXPECTED_FAILURE"), "message": str(exc)}, sort_keys=True), file=sys.stderr)
                    raise
        for path in (stdout_path, stderr_path):
            if path.stat().st_size > 1024 * 1024:
                raise ProofFailure("OUTPUT_LIMIT_FAILURE", f"command log exceeds the frozen 1 MiB cap: {path.name}")
        if payload is None:
            raise ProofFailure("OUTPUT_LIMIT_FAILURE", "proof computation produced no result record")
        proof_path = output_dir / "sprint22_disposable_boundary_proof.json"
        with proof_path.open("xb") as proof_file:
            proof_file.write(payload)
        return 0
    except ProofFailure as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "code": exc.code, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "code": "UNEXPECTED_FAILURE", "type": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
