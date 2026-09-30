"""Sprint 21 Task 3 — restore-only disposable scorer proof (toy fixture only).

Runs ONLY the exact in-memory two-sample fixture frozen in
``experiments/sprint21-disposable-scorer-proof-v1.md`` against the three
predeclared historical B0 step-300 checkpoints. Restore-only: no training,
refit, recalibration, generation, preflight, or history/root contact.

Isolation contract (enforced at import/runtime):

- every project module (``representation.*``, ``synth.*``) MUST resolve to
  files inside the pinned detached worktree root (``--worktree-root``,
  default ``/tmp/sprint21-disposable-proof-v1``);
- the wrapper MUST be executed from a source tree whose HEAD is exactly
  ``8c15f02...808de`` with a clean status;
- the wrapper MUST refuse Calibration thresholds and Task 7 ``main()``.

Usage (after Task 2 PASS; on the proof host)::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
      <historical-venv-python> experiments/sprint21_disposable_scorer_proof.py \
        --checkpoint-root /tmp/sprint17-task7-out/checkpoints \
        --output-dir /tmp/sprint21-task3-proof-out
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

WORKTREE_ENV_VAR = "S21_PROOF_WORKTREE_ROOT"
DEFAULT_WORKTREE_ROOT = "/tmp/sprint21-disposable-proof-v1"

# ---- Frozen proof constants (contract v1, sections 2-4) --------------------
BASE_COMMIT = "8c15f0204a3e495569b7f143dc109943e8b808de"
CONFIG_SHA256 = "1ff67f95428ef29aab05d9f6394a305b75c8c2a9e12431f3cda09b6958596144"
METRIC_DIGEST = "b2d6af7505abe459a84f76f5279a6a5c4298e7c142901edd4596fcfe3ddeb88f"
BINDING_DIGEST = "075868b22c028a981cc63ffe37b8d29720cd324c3e2c7254ce688678758230ba"
MAPPING_DIGEST = "4f769ccc64aac4c6a91fdd356eaa6963afe976ab847a22df23e43f9331418897"
EXPECTED_UV_LOCK_SHA256 = "4a7866878c81cdd8f5ba283cd8d0a772073bfc75941f73ec761ede5eeb2e239a"

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
CHECKPOINT_BYTES = 18_239_321
CHECKPOINT_STEP = 300

SOURCE_HASHES = {
    "src/representation/model.py": "7051f5f76be0bcc8e50d6d5ee2941a57c12a3285406d0ef27883c596aa6600f0",
    "src/representation/data.py": "88ea3becbc8328c5b21518b1f957fe710096cb96ed0cd2416fe6f139439d5eb4",
    "src/representation/masking.py": "c0efd33b24b8d01c067198f0bc04c0c90c1e99bad9a6a1ec006b37ce27c0465f",
    "src/representation/contracts.py": "0b23080fc49f218484a6f7939bb9288f07b607686312e91c3b14d9d6b84464ed",
    "src/representation/layers/normalization.py": "bfb5f68f5352115c2e044c27999cdf3fecab7aa3ce2cd05ce021204f1f9edd8d",
    "src/synth/chronicle.py": "b5992d6af3c4387d18df746b43be17b4bfc1fac6938e8b44be35b6fb7c65dafd",
}
METRIC_FILES = (
    "src/synth/probe15.py",
    "src/synth/events.py",
    "src/representation/attribution_metrics.py",
    "src/representation/sprint17_ablation.py",
    "experiments/sprint17_task7_b0.py",
)

CHANNEL_ORDER = ("feed", "current", "temperature", "arc-voltage", "arc-power", "torch-pressure")
N_CHANNELS = 6
BANK_ROWS = 5040
BANK_DIM = 128
BANK_K = 5
REL_TOL = 1e-6
ABS_TOL = 1e-7


TOY_SPECS = (
    {
        "file_id": "S21T2-TOY-A",
        "robot_id": "robot-01",
        "robot_idx": 0,
        "program_id": "program-01",
        "program_idx": 0,
        "formula": "A",
        "tensor_sha256": "236eed5f43d5a407fbc3598476a4e761859b658c168c90cdc0acb93ebbae7f0b",
        "file_seed": 1001349327,
    },
    {
        "file_id": "S21T2-TOY-B",
        "robot_id": "robot-09",
        "robot_idx": 8,
        "program_id": "program-08",
        "program_idx": 7,
        "formula": "B",
        "tensor_sha256": "55afb88796385f9aaeeafe6b553b5166412c995103323bc115083981b14b9f38",
        "file_seed": 267614413,
    },
)


class ProofFailure(ValueError):
    """Fail-closed proof error tagged with a contract disposition."""

    def __init__(self, disposition: str, message: str) -> None:
        super().__init__(f"[{disposition}] {message}")
        self.disposition = disposition


def canonical_hash(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_seed(file_id: str) -> int:
    digest = hashlib.sha256(file_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_digest(t) -> str:
    import numpy as np

    arr = np.ascontiguousarray(t, dtype=np.float32)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def verify_uv_lock_digest(worktree_root: Path) -> str:
    path = worktree_root / "uv.lock"
    if not path.is_file():
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", f"missing pinned {path.name}")
    digest = sha256_file(path)
    if digest != EXPECTED_UV_LOCK_SHA256:
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE",
            f"uv.lock digest {digest} != pinned {EXPECTED_UV_LOCK_SHA256}",
        )
    return digest


def verify_metric_code(worktree_root: Path) -> dict[str, object]:
    member_digests: dict[str, str] = {}
    digest_rows: list[list[str]] = []
    for rel in METRIC_FILES:
        path = worktree_root / rel
        if not path.is_file():
            raise ProofFailure(
                "RESTORE_ONLY_OR_ISOLATION_FAILURE", f"missing metric source {rel}"
            )
        digest = sha256_file(path)
        member_digests[rel] = digest
        digest_rows.append([rel, digest])
    digest = canonical_hash(digest_rows)
    if digest != METRIC_DIGEST:
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE",
            f"metric-code digest {digest} != pinned {METRIC_DIGEST}",
        )
    return {"digest": digest, "files": member_digests}


def verify_role_binding(worktree_root: Path) -> dict[str, str]:
    path = worktree_root / "experiments/sprint17-role-binding-v3.json"
    if not path.is_file():
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE", "missing historical role-binding JSON"
        )
    try:
        binding = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE", f"cannot read role-binding JSON: {exc}"
        ) from exc
    if not isinstance(binding, dict):
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE", "role-binding JSON is not an object"
        )
    declared_digest = binding.pop("binding_sha256", None)
    if binding.get("profile_id") != "sprint15-v7":
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE",
            f"role-binding profile {binding.get('profile_id')!r} != 'sprint15-v7'",
        )
    digest = canonical_hash(binding)
    if declared_digest != BINDING_DIGEST or digest != declared_digest:
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE",
            f"role-binding digest {digest} != declared/pinned {declared_digest!r}/{BINDING_DIGEST}",
        )
    return {
        "digest": digest,
        "file_sha256": sha256_file(path),
        "profile_id": "sprint15-v7",
    }


def validate_conditioning_mapping(mapping: dict[str, dict[str, int]]) -> str:
    if set(mapping) != {"robot_id_to_idx", "program_id_to_idx"}:
        raise ProofFailure("CONDITIONING_MISMATCH", "conditioning map has missing or extra fields")
    for key in ("robot_id_to_idx", "program_id_to_idx"):
        values = mapping[key]
        if not isinstance(values, dict) or not values:
            raise ProofFailure("CONDITIONING_MISMATCH", f"{key} is empty or not a mapping")
        if any(not isinstance(identifier, str) or not identifier.strip() for identifier in values):
            raise ProofFailure("CONDITIONING_MISMATCH", f"{key} contains an invalid identifier")
        identifiers = sorted(values)
        if any(isinstance(values[name], bool) or not isinstance(values[name], int) for name in identifiers):
            raise ProofFailure("CONDITIONING_MISMATCH", f"{key} contains a non-integer index")
        if values != {identifier: index for index, identifier in enumerate(identifiers)}:
            raise ProofFailure("CONDITIONING_MISMATCH", f"{key} is not sorted/enumerated")
    digest = canonical_hash(mapping)
    if digest != MAPPING_DIGEST:
        raise ProofFailure(
            "CONDITIONING_MISMATCH",
            f"conditioning map digest {digest} != pinned {MAPPING_DIGEST}",
        )
    return digest


def conditioning_mapping_from_config(config: object) -> tuple[dict[str, dict[str, int]], str]:
    try:
        stages = [
            stage
            for route in config.scheduler.routes
            for stage in route.stages
        ]
        robot_ids = sorted({stage.robot_id for stage in stages})
        program_ids = sorted({stage.program_id for stage in stages})
        if len(robot_ids) != config.fleet.n_robots or len(program_ids) != config.fleet.n_programs:
            raise ProofFailure(
                "CONDITIONING_MISMATCH",
                "static profile route identifiers do not match its fleet cardinalities",
            )
        mapping = {
            "robot_id_to_idx": {identifier: index for index, identifier in enumerate(robot_ids)},
            "program_id_to_idx": {
                identifier: index for index, identifier in enumerate(program_ids)
            },
        }
    except ProofFailure:
        raise
    except Exception as exc:
        raise ProofFailure(
            "CONDITIONING_MISMATCH", f"cannot derive identifiers from static history config: {exc}"
        ) from exc
    return mapping, validate_conditioning_mapping(mapping)


def derive_conditioning_mapping(worktree_root: Path) -> tuple[dict[str, dict[str, int]], str]:
    pinned_src = (worktree_root / "src").resolve()
    inserted_path = str(pinned_src)
    inserted = inserted_path not in sys.path
    if inserted:
        sys.path.insert(0, inserted_path)
    try:
        chronicle = importlib.import_module("synth.chronicle")
        module_path = Path(chronicle.__file__).resolve()
        if not module_path.is_relative_to(pinned_src):
            raise ProofFailure(
                "RESTORE_ONLY_OR_ISOLATION_FAILURE",
                f"static history config imported from {module_path}, outside {pinned_src}",
            )
        profile_config = chronicle.sprint15_v7_history_config(seed=0)
        return conditioning_mapping_from_config(profile_config)
    except ProofFailure:
        raise
    except Exception as exc:
        raise ProofFailure(
            "CONDITIONING_MISMATCH", f"cannot load pinned sprint15-v7 static config: {exc}"
        ) from exc
    finally:
        if inserted:
            sys.path.remove(inserted_path)


def build_toy_array(formula: str):
    import numpy as np

    x = np.zeros((N_CHANNELS, 64), dtype=np.float32)
    for c in range(N_CHANNELS):
        for t in range(64):
            if formula == "A":
                value = (((7 * t + 3 * c) % 17) - 8) / 8
            elif formula == "B":
                value = (((5 * t + 11 * c + 3) % 19) - 9) / 9
            else:  # pragma: no cover - contract fixes A/B only
                raise ProofFailure("INPUT_CONTRACT_FAILURE", f"unknown toy formula {formula!r}")
            x[c, t] = np.float32(value)
    return np.ascontiguousarray(x)


@dataclass
class ToyRecord:
    file_id: str
    robot_id: str
    robot_idx: int
    program_id: str
    program_idx: int
    x: object = field(repr=False)


def build_toy_roster() -> list[ToyRecord]:
    import numpy as np

    roster: list[ToyRecord] = []
    for spec in TOY_SPECS:
        x = build_toy_array(spec["formula"])
        assert isinstance(x, np.ndarray)
        if x.dtype != np.float32 or x.shape != (N_CHANNELS, 64):
            raise ProofFailure(
                "INPUT_CONTRACT_FAILURE",
                f"{spec['file_id']}: toy shape/dtype {x.shape}/{x.dtype}",
            )
        digest = tensor_digest(x)
        if digest != spec["tensor_sha256"]:
            raise ProofFailure(
                "INPUT_CONTRACT_FAILURE",
                f"{spec['file_id']}: tensor digest {digest} != {spec['tensor_sha256']}",
            )
        seed = file_seed(spec["file_id"])
        if seed != spec["file_seed"]:
            raise ProofFailure(
                "MASK_CONTRACT_FAILURE",
                f"{spec['file_id']}: file_seed {seed} != {spec['file_seed']}",
            )
        roster.append(
            ToyRecord(
                file_id=spec["file_id"],
                robot_id=spec["robot_id"],
                robot_idx=spec["robot_idx"],
                program_id=spec["program_id"],
                program_idx=spec["program_idx"],
                x=x,
            )
        )
    return roster


def check_no_history_contact(args: argparse.Namespace) -> None:
    for name in ("data_root", "history_root", "data_dir", "history_dir", "root"):
        value = getattr(args, name, None)
        if value:
            raise ProofFailure(
                "RESTORE_ONLY_OR_ISOLATION_FAILURE",
                f"history/root contact forbidden (got --{name}={value!r})",
            )


def assert_pinned_tree(worktree_root: Path, *, allow_wrapper: bool = False) -> dict[str, str]:
    """Verify the live source tree is the pinned historical tree."""
    import subprocess

    if not (worktree_root / ".git").exists() and not (worktree_root / "src").is_dir():
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE",
            f"worktree root {worktree_root} is not a pinned source tree",
        )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(worktree_root),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE", f"cannot verify worktree HEAD: {exc}"
        ) from exc
    if head != BASE_COMMIT:
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE",
            f"worktree HEAD {head} != pinned {BASE_COMMIT}",
        )
    if allow_wrapper:
        other = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(worktree_root),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        other_lines = [line for line in other.splitlines() if "sprint21_disposable_scorer_proof" not in line]
        if other_lines:
            raise ProofFailure(
                "RESTORE_ONLY_OR_ISOLATION_FAILURE",
                f"worktree is dirty at pinned commit: {other_lines[0][:200]}",
            )
    else:
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(worktree_root),
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception as exc:
            raise ProofFailure(
                "RESTORE_ONLY_OR_ISOLATION_FAILURE", f"cannot verify worktree status: {exc}"
            ) from exc
        if status:
            raise ProofFailure(
                "RESTORE_ONLY_OR_ISOLATION_FAILURE",
                f"worktree is dirty at pinned commit: {status[:200]}",
            )
    checked: dict[str, str] = {}
    for rel, expected in SOURCE_HASHES.items():
        data = (worktree_root / rel).read_bytes()
        digest = sha256_bytes(data)
        if digest != expected:
            raise ProofFailure(
                "RESTORE_ONLY_OR_ISOLATION_FAILURE",
                f"{rel} digest {digest} != pinned {expected}",
            )
        checked[rel] = digest
    checked["uv.lock"] = verify_uv_lock_digest(worktree_root)
    return checked


def assert_project_modules_pinned(worktree_root: Path) -> None:
    """Every imported project module must resolve inside the pinned tree."""
    pinned_src = (worktree_root / "src").resolve()
    for name, module in list(sys.modules.items()):
        if not (name == "representation" or name.startswith("representation.") or name == "synth" or name.startswith("synth.")):
            continue
        location = getattr(module, "__file__", None)
        if location is None:
            continue
        resolved = str(Path(location).resolve())
        if not resolved.startswith(str(pinned_src)):
            raise ProofFailure(
                "RESTORE_ONLY_OR_ISOLATION_FAILURE",
                f"module {name} loaded from {resolved}, outside pinned {pinned_src}",
            )


def assert_runtime() -> dict[str, str]:
    import torch

    info = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if info["python"] != "3.12.13":
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE",
            f"python {info['python']} != frozen 3.12.13",
        )
    if info["torch"] != "2.14.0+cu130":
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE",
            f"torch {info['torch']} != frozen 2.14.0+cu130",
        )
    if not torch.cuda.is_available():
        raise ProofFailure("RESTORE_ONLY_OR_ISOLATION_FAILURE", "CUDA is not available")
    return info


def verify_checkpoint_file(path: Path, model_seed: int) -> None:
    if not path.is_file():
        raise ProofFailure("PROVENANCE_MISMATCH", f"missing checkpoint {path}")
    size = path.stat().st_size
    if size != CHECKPOINT_BYTES:
        raise ProofFailure(
            "PROVENANCE_MISMATCH", f"{path.name}: {size} bytes != {CHECKPOINT_BYTES}"
        )
    digest = sha256_file(path)
    if digest != CHECKPOINT_SHA256[model_seed]:
        raise ProofFailure(
            "PROVENANCE_MISMATCH", f"{path.name}: sha256 {digest} does not match seed {model_seed}"
        )


def validate_conditioning(record: ToyRecord, mapping: dict[str, dict[str, int]]) -> None:
    if not isinstance(record.robot_id, str) or not record.robot_id.strip():
        raise ProofFailure("CONDITIONING_MISMATCH", f"{record.file_id}: missing robot string")
    if not isinstance(record.program_id, str) or not record.program_id.strip():
        raise ProofFailure("CONDITIONING_MISMATCH", f"{record.file_id}: missing program string")
    if (
        isinstance(record.robot_idx, bool)
        or not isinstance(record.robot_idx, int)
        or isinstance(record.program_idx, bool)
        or not isinstance(record.program_idx, int)
    ):
        raise ProofFailure("CONDITIONING_MISMATCH", f"{record.file_id}: conditioning index is not integer")
    robot_map = mapping["robot_id_to_idx"]
    program_map = mapping["program_id_to_idx"]
    if record.robot_id not in robot_map or record.program_id not in program_map:
        raise ProofFailure(
            "CONDITIONING_MISMATCH",
            f"{record.file_id}: unknown conditioning {record.robot_id}/{record.program_id}",
        )
    if robot_map[record.robot_id] != record.robot_idx:
        raise ProofFailure(
            "CONDITIONING_MISMATCH",
            f"{record.file_id}: robot string/index disagree "
            f"{record.robot_id}/{record.robot_idx}",
        )
    if program_map[record.program_id] != record.program_idx:
        raise ProofFailure(
            "CONDITIONING_MISMATCH",
            f"{record.file_id}: program string/index disagree "
            f"{record.program_id}/{record.program_idx}",
        )


def validate_input_array(record: ToyRecord) -> None:
    import numpy as np

    x = record.x
    if not isinstance(x, np.ndarray):
        raise ProofFailure("INPUT_CONTRACT_FAILURE", f"{record.file_id}: x is not ndarray")
    if x.dtype != np.float32:
        raise ProofFailure(
            "INPUT_CONTRACT_FAILURE", f"{record.file_id}: dtype {x.dtype} != float32"
        )
    if x.ndim != 2 or x.shape[0] != N_CHANNELS or x.shape[1] == 0:
        raise ProofFailure(
            "INPUT_CONTRACT_FAILURE", f"{record.file_id}: bad shape {x.shape}"
        )
    if not np.isfinite(x).all():
        raise ProofFailure("INPUT_CONTRACT_FAILURE", f"{record.file_id}: non-finite input")


def check_full_roster(
    expected_ids: list[str],
    scored: dict[str, dict[str, float]],
    support_keys: dict[str, set[str]] | None = None,
) -> None:
    """Reusable full-roster invariant for future Task 4 inputs (contract 5)."""
    if sorted(scored.keys()) != sorted(expected_ids):
        raise ProofFailure(
            "ROSTER_OR_SCORE_INCOMPLETE",
            f"scored ids {sorted(scored.keys())} != declared {sorted(expected_ids)}",
        )
    if len(set(expected_ids)) != len(expected_ids):
        raise ProofFailure("ROSTER_OR_SCORE_INCOMPLETE", "duplicate ids in expected roster")
    branch_keys: set[str] | None = None
    for file_id in expected_ids:
        branches = scored[file_id]
        if set(branches.keys()) != {"S_pred", "S_pop"}:
            raise ProofFailure(
                "ROSTER_OR_SCORE_INCOMPLETE", f"{file_id}: branch keys {sorted(branches.keys())}"
            )
        for branch, value in branches.items():
            if not isinstance(value, float) or not __import__("math").isfinite(value):
                raise ProofFailure(
                    "ROSTER_OR_SCORE_INCOMPLETE", f"{file_id}/{branch}: non-finite score"
                )
            if value < 0:
                raise ProofFailure(
                    "ROSTER_OR_SCORE_INCOMPLETE", f"{file_id}/{branch}: negative score"
                )
        if branch_keys is None:
            branch_keys = set(branches.keys())
        elif set(branches.keys()) != branch_keys:
            raise ProofFailure("ROSTER_OR_SCORE_INCOMPLETE", f"{file_id}: branch key-set drift")
    if support_keys is not None:
        keys = {json.dumps(sorted(v), sort_keys=True) for v in support_keys.values()}
        if len(keys) != 1:
            raise ProofFailure("ROSTER_OR_SCORE_INCOMPLETE", "support keys differ across branches")


def snapshot_state(model, bank) -> dict[str, str]:
    import torch

    parts: dict[str, str] = {}
    for name, param in sorted(model.named_parameters()):
        parts[f"param:{name}"] = sha256_bytes(param.detach().cpu().numpy().tobytes())
    for name, buf in sorted(model.named_buffers()):
        if isinstance(buf, torch.Tensor):
            parts[f"buffer:{name}"] = sha256_bytes(buf.detach().cpu().numpy().tobytes())
        else:
            parts[f"buffer:{name}"] = sha256_bytes(repr(buf).encode())
    parts["bank:k"] = str(int(bank.k))
    if bank.embeddings is None:
        parts["bank:embeddings"] = "NONE"
    else:
        parts["bank:embeddings"] = sha256_bytes(bank.embeddings.detach().cpu().numpy().tobytes())
    return parts


def validate_b0_config_map(configs: dict[str, dict[str, object]]) -> str:
    expected_seeds = {str(seed) for seed in MODEL_SEEDS}
    if set(configs) != expected_seeds:
        raise ProofFailure(
            "PROVENANCE_MISMATCH",
            f"B0 config seeds {sorted(configs)} != frozen {sorted(expected_seeds)}",
        )
    for seed in MODEL_SEEDS:
        config = configs[str(seed)]
        if not isinstance(config, dict) or config.get("model_seed") != seed:
            raise ProofFailure(
                "PROVENANCE_MISMATCH", f"B0 config for seed {seed} has wrong model_seed"
            )
    digest = canonical_hash(configs)
    if digest != CONFIG_SHA256:
        raise ProofFailure(
            "PROVENANCE_MISMATCH",
            f"full three-seed B0 config digest {digest} != pinned {CONFIG_SHA256}",
        )
    return digest


def load_b0_config_map(worktree_root: Path) -> tuple[dict[str, dict[str, object]], str]:
    module_path = (worktree_root / "experiments" / "sprint17_task7_b0.py").resolve()
    expected_dir = (worktree_root / "experiments").resolve()
    if not module_path.is_file() or not module_path.is_relative_to(expected_dir):
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE",
            f"pinned B0 config module missing or outside {expected_dir}: {module_path}",
        )
    module_name = "_sprint21_pinned_task7_b0"
    if module_name in sys.modules:
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE",
            f"pinned B0 config module name already loaded: {module_name}",
        )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE",
            f"cannot load pinned B0 config module at {module_path}",
        )
    task7 = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = task7
    pinned_src = str((worktree_root / "src").resolve())
    path_count = sys.path.count(pinned_src)
    try:
        spec.loader.exec_module(task7)
        configs = {
            str(seed): task7.b0_config_dict(seed)
            for seed in MODEL_SEEDS
        }
    except ProofFailure:
        raise
    except Exception as exc:
        raise ProofFailure(
            "PROVENANCE_MISMATCH", f"cannot derive pinned B0 configs: {exc}"
        ) from exc
    finally:
        sys.modules.pop(module_name, None)
        while sys.path.count(pinned_src) > path_count:
            sys.path.remove(pinned_src)
    return configs, validate_b0_config_map(configs)


def validate_restored_config(
    saved_config: object,
    expected_config: dict[str, object],
    model_seed: int,
) -> str:
    from representation.config import V1Config

    if not isinstance(saved_config, dict):
        raise ProofFailure("PROVENANCE_MISMATCH", "checkpoint config is not a mapping")
    if expected_config.get("seed") != model_seed or saved_config.get("seed") != model_seed:
        raise ProofFailure(
            "PROVENANCE_MISMATCH", f"checkpoint config seed does not match {model_seed}"
        )
    try:
        restored_config = V1Config(**saved_config).to_dict()
    except Exception as exc:
        raise ProofFailure("PROVENANCE_MISMATCH", f"checkpoint config is invalid: {exc}") from exc
    saved_digest = canonical_hash(saved_config)
    expected_digest = canonical_hash(expected_config)
    if saved_digest != expected_digest or canonical_hash(restored_config) != expected_digest:
        raise ProofFailure(
            "PROVENANCE_MISMATCH",
            f"seed {model_seed} checkpoint config digest {saved_digest} "
            f"!= expected {expected_digest}",
        )
    return saved_digest


def restore_seed(
    checkpoint_path: Path,
    model_seed: int,
    device: str,
    config_map: dict[str, dict[str, object]],
):
    validate_b0_config_map(config_map)
    if model_seed not in MODEL_SEEDS:
        raise ProofFailure("PROVENANCE_MISMATCH", f"unexpected model seed {model_seed}")
    cfg_dict = config_map[str(model_seed)]
    import torch
    from representation.config import V1Config
    from representation.inference import NormalReferenceBank
    from representation.model import V1RepresentationModel
    from synth.config import PatchConfig
    from synth.patchify import Patchifier
    from representation.checkpoint import load_checkpoint

    cfg = V1Config(
        n_channels=cfg_dict["n_channels"],
        patch_size=cfg_dict["patch_size"],
        stride=cfg_dict["stride"],
        d_model=cfg_dict["d_model"],
        sequence_layers=cfg_dict["sequence_layers"],
        attention_heads=cfg_dict["attention_heads"],
        dropout=cfg_dict["dropout"],
        n_robots=cfg_dict["n_robots"],
        n_programs=cfg_dict["n_programs"],
        use_conditional_norm=cfg_dict["use_conditional_norm"],
        min_bucket_samples=cfg_dict["min_bucket_samples"],
        total_mask_ratio=cfg_dict["total_mask_ratio"],
        random_fraction=cfg_dict["random_fraction"],
        info_fraction=cfg_dict["info_fraction"],
        block_fraction=cfg_dict["block_fraction"],
        ema_decay=cfg_dict["ema_decay"],
        prediction_weight=cfg_dict["prediction_weight"],
        contrastive_weight_max=cfg_dict["contrastive_weight_max"],
        contrastive_warmup_steps=cfg_dict["contrastive_warmup_steps"],
        contrastive_ramp_steps=cfg_dict["contrastive_ramp_steps"],
        contrastive_temperature=cfg_dict["contrastive_temperature"],
        contrastive_gain_std=cfg_dict["contrastive_gain_std"],
        contrastive_offset_std=cfg_dict["contrastive_offset_std"],
        contrastive_noise_std=cfg_dict["contrastive_noise_std"],
        contrastive_max_shift=cfg_dict["contrastive_max_shift"],
        knn_k=cfg_dict["knn_k"],
        seed=cfg_dict["model_seed"],
    )
    patchifier = Patchifier(
        PatchConfig(patch_size=cfg.patch_size, stride=cfg.stride, pad_end=cfg_dict["pad_end"])
    )
    model = V1RepresentationModel(cfg, patchifier=patchifier)
    bank: NormalReferenceBank = NormalReferenceBank(k=BANK_K)
    meta = load_checkpoint(checkpoint_path, model, reference_bank=bank)
    if meta.get("step") != CHECKPOINT_STEP:
        raise ProofFailure(
            "PROVENANCE_MISMATCH", f"checkpoint step {meta.get('step')} != {CHECKPOINT_STEP}"
        )
    if not meta.get("has_reference_bank"):
        raise ProofFailure("PROVENANCE_MISMATCH", "checkpoint has no embedded reference bank")
    saved_config_digest = validate_restored_config(
        meta.get("config"), cfg.to_dict(), model_seed
    )
    if bank.embeddings is None:
        raise ProofFailure("PROVENANCE_MISMATCH", "restored bank has no embeddings")
    if bank.k != BANK_K:
        raise ProofFailure("PROVENANCE_MISMATCH", f"bank k {bank.k} != {BANK_K}")
    if tuple(bank.embeddings.shape) != (BANK_ROWS, BANK_DIM):
        raise ProofFailure(
            "PROVENANCE_MISMATCH", f"bank shape {tuple(bank.embeddings.shape)} != {(BANK_ROWS, BANK_DIM)}"
        )
    if not bool(torch.isfinite(bank.embeddings).all()):
        raise ProofFailure("PROVENANCE_MISMATCH", "restored bank is non-finite")
    model.to(device)
    model.eval()
    bank_digest = sha256_bytes(bank.embeddings.detach().cpu().numpy().tobytes())
    return model, bank, patchifier, cfg, bank_digest, saved_config_digest


def collate_toy_record(
    record: ToyRecord,
    patchifier,
    cfg,
    mapping: dict[str, dict[str, int]],
    *,
    masking_seed: int | None = None,
) -> tuple[dict[str, object], int, list[int]]:
    from representation.data import collate_variable_files
    from synth.schema import FileSample, SampleLabel

    validate_conditioning(record, mapping)
    validate_input_array(record)
    canonical_seed = file_seed(record.file_id)
    seed = canonical_seed if masking_seed is None else masking_seed
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProofFailure("MASK_CONTRACT_FAILURE", f"{record.file_id}: mask seed is not an integer")
    sample = FileSample(
        x=record.x,
        file_id=record.file_id,
        file_label=SampleLabel.NORMAL,
        seed=0,
        generator_version="S21-T2-MATH-FIXTURE-v1",
        config_hash="s21-t2-math-fixture-v1",
        regime_sequence=[],
        robot_idx=record.robot_idx,
        program_idx=record.program_idx,
    )
    batch = collate_variable_files([sample], patchifier, masking_config=cfg, masking_seed=seed)
    if seed != canonical_seed:
        raise ProofFailure(
            "MASK_CONTRACT_FAILURE",
            f"{record.file_id}: masking seed {seed} != canonical {canonical_seed}",
        )
    valid = batch["patch_valid_mask"][0].detach().cpu().numpy()
    n_valid = int(valid.sum())
    if n_valid != 3:
        raise ProofFailure(
            "MASK_CONTRACT_FAILURE", f"{record.file_id}: {n_valid} valid patches != 3"
        )
    mask = batch["mask"][0].detach().cpu().numpy().astype(bool)
    bits = [int(value) for value in mask.tolist()]
    if sum(bits) != 1:
        raise ProofFailure(
            "MASK_CONTRACT_FAILURE", f"{record.file_id}: {sum(bits)} masked != 1"
        )
    if bool((mask & ~valid).any()):
        raise ProofFailure("MASK_CONTRACT_FAILURE", f"{record.file_id}: masked padding")
    return batch, seed, bits


def score_roster_once(
    model, bank, patchifier, cfg, roster: list[ToyRecord], device: str,
    mapping: dict[str, dict[str, int]],
):
    import torch
    import torch.nn.functional as F
    from representation.inference import RepresentationInference
    inference = RepresentationInference(model, bank, patchifier, masking_config=cfg)
    scores: dict[str, dict[str, float]] = {}
    masks: dict[str, list[int]] = {}
    details: dict[str, dict[str, object]] = {}
    for record in roster:
        batch, seed, bits = collate_toy_record(record, patchifier, cfg, mapping)
        masks[record.file_id] = bits
        moved = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
        }
        model.eval()
        with torch.inference_mode():
            res = inference.score_batch(moved)
        s_pred = float(torch.as_tensor(res["S_pred"]).reshape(-1)[0].item())
        s_pop = float(torch.as_tensor(res["S_pop"]).reshape(-1)[0].item())
        # Independent recomputation from the same inference tensors.
        errors = F.mse_loss(
            res["predicted_latents"], res["target_latents"], reduction="none"
        ).mean(dim=-1)
        pred_mask = res["prediction_mask"]
        expected_pred = float(
            (errors.masked_fill(~pred_mask, 0.0).sum(dim=1) / pred_mask.sum(dim=1).clamp_min(1).to(errors.dtype))
            .reshape(-1)[0]
            .item()
        )
        distances = torch.cdist(
            res["file_embedding"].float().cpu(), bank.embeddings.float().cpu()
        )
        expected_pop = float(
            distances.topk(BANK_K, largest=False, dim=1).values.mean(dim=1).reshape(-1)[0].item()
        )
        for name, got, want in (("S_pred", s_pred, expected_pred), ("S_pop", s_pop, expected_pop)):
            if not (__import__("math").isfinite(got) and __import__("math").isfinite(want)):
                raise ProofFailure("ROSTER_OR_SCORE_INCOMPLETE", f"{record.file_id}/{name} non-finite")
            if got < 0 or want < 0:
                raise ProofFailure("ROSTER_OR_SCORE_INCOMPLETE", f"{record.file_id}/{name} negative")
            tol = ABS_TOL + REL_TOL * abs(want)
            if abs(got - want) > tol:
                raise ProofFailure(
                    "ROSTER_OR_SCORE_INCOMPLETE",
                    f"{record.file_id}/{name}: {got} vs independent {want}",
                )
        scores[record.file_id] = {"S_pred": s_pred, "S_pop": s_pop}
        details[record.file_id] = {
            "mask_bits": bits,
            "mask_seed": seed,
            "predicted_latents_norm": float(res["predicted_latents"].detach().float().norm().item()),
            "file_embedding_norm": float(res["file_embedding"].detach().float().norm().item()),
        }
    check_full_roster([r.file_id for r in roster], scores)
    return scores, masks, details


def run_negative_cases(
    patchifier,
    cfg,
    roster: list[ToyRecord],
    mapping: dict[str, dict[str, int]],
) -> list[dict[str, str]]:
    """Exercise fail-closed cases on in-memory mutated copies only."""
    import copy

    import numpy as np

    cases: list[dict[str, str]] = []

    def expect(disposition: str, name: str, fn) -> None:
        try:
            fn()
        except ProofFailure as exc:
            if exc.disposition != disposition:
                raise ProofFailure(
                    "RESTORE_ONLY_OR_ISOLATION_FAILURE",
                    f"negative case {name}: disposition {exc.disposition} != {disposition}",
                ) from exc
            cases.append({"case": name, "disposition": exc.disposition, "observed": str(exc)})
            return
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE", f"negative case {name} did not fail closed"
        )

    base = roster[0]
    expect(
        "CONDITIONING_MISMATCH",
        "unknown-robot-string",
        lambda: validate_conditioning(
            ToyRecord(base.file_id, "robot-99", 99, base.program_id, base.program_idx, base.x),
            mapping,
        ),
    )
    expect(
        "CONDITIONING_MISMATCH",
        "string-index-disagree",
        lambda: validate_conditioning(
            ToyRecord(base.file_id, base.robot_id, 5, base.program_id, base.program_idx, base.x),
            mapping,
        ),
    )
    bad_channels = copy.copy(base)
    bad_channels.x = np.zeros((5, 64), dtype=np.float32)
    expect("INPUT_CONTRACT_FAILURE", "wrong-channel-count", lambda: validate_input_array(bad_channels))
    bad_finite = copy.copy(base)
    poisoned = np.array(base.x, dtype=np.float32, copy=True)
    poisoned[0, 0] = np.inf
    bad_finite.x = poisoned
    expect("INPUT_CONTRACT_FAILURE", "non-finite-input", lambda: validate_input_array(bad_finite))
    expect(
        "MASK_CONTRACT_FAILURE",
        "wrong-mask-seed",
        lambda: collate_toy_record(
            base,
            patchifier,
            cfg,
            mapping,
            masking_seed=file_seed(base.file_id) + 1,
        ),
    )
    expect("ROSTER_OR_SCORE_INCOMPLETE", "duplicate-roster", lambda: check_full_roster(
        [base.file_id, base.file_id],
        {base.file_id: {"S_pred": 0.1, "S_pop": 0.2}},
    ))
    expect("ROSTER_OR_SCORE_INCOMPLETE", "missing-branch", lambda: check_full_roster(
        [base.file_id], {base.file_id: {"S_pred": 0.1}}
    ))
    expect("PROVENANCE_MISMATCH", "missing-checkpoint", lambda: verify_checkpoint_file(
        Path("/nonexistent/sprint21-proof-missing.pt"), MODEL_SEEDS[0]
    ))
    return cases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 21 Task 3 restore-only toy proof")
    parser.add_argument("--checkpoint-root", required=True, help="read-only historical checkpoint dir")
    parser.add_argument("--output-dir", required=True, help="disposable proof output dir")
    parser.add_argument("--worktree-root", default=None, help="pinned detached worktree root")
    parser.add_argument("--device", default="cuda", help="torch device (proof requires CUDA)")
    parser.add_argument("--run-negative-cases", action="store_true", help="exercise fail-closed cases")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    check_no_history_contact(args)
    worktree_root = Path(
        args.worktree_root
        or __import__("os").environ.get(WORKTREE_ENV_VAR, DEFAULT_WORKTREE_ROOT)
    ).resolve()
    if worktree_root != REPO_ROOT.resolve():
        # The wrapper file itself must live in the pinned tree; a different
        # root means the caller staged an unreviewed copy.
        raise ProofFailure(
            "RESTORE_ONLY_OR_ISOLATION_FAILURE",
            f"wrapper lives in {REPO_ROOT.resolve()}, worktree root is {worktree_root}",
        )
    t0 = time.time()
    source_hashes = assert_pinned_tree(worktree_root, allow_wrapper=True)
    mapping, observed_mapping_digest = derive_conditioning_mapping(worktree_root)
    metric_provenance = verify_metric_code(worktree_root)
    binding_provenance = verify_role_binding(worktree_root)
    config_map, observed_config_digest = load_b0_config_map(worktree_root)
    assert_project_modules_pinned(worktree_root)
    roster = build_toy_roster()
    for record in roster:
        validate_conditioning(record, mapping)
    runtime = assert_runtime()

    checkpoint_root = Path(args.checkpoint_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    peak_vram_mib: float | None = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    per_seed: dict[str, dict[str, object]] = {}
    for model_seed in MODEL_SEEDS:
        checkpoint_path = checkpoint_root / CHECKPOINT_FILES[model_seed]
        verify_checkpoint_file(checkpoint_path, model_seed)
        digest_before = sha256_file(checkpoint_path)
        model, bank, patchifier, cfg, bank_digest, checkpoint_config_digest = restore_seed(
            checkpoint_path, model_seed, args.device, config_map
        )
        assert_project_modules_pinned(worktree_root)
        before = snapshot_state(model, bank)
        scores, masks, details = score_roster_once(
            model, bank, patchifier, cfg, roster, args.device, mapping
        )
        # Deterministic repeat + reversed order on the same restored state.
        scores_repeat, masks_repeat, _ = score_roster_once(
            model, bank, patchifier, cfg, roster, args.device, mapping
        )
        scores_reversed, masks_reversed, _ = score_roster_once(
            model, bank, patchifier, cfg, list(reversed(roster)), args.device, mapping
        )
        for file_id in scores:
            for branch in ("S_pred", "S_pop"):
                for other in (scores_repeat, scores_reversed):
                    want = scores[file_id][branch]
                    got = other[file_id][branch]
                    if abs(got - want) > ABS_TOL + REL_TOL * abs(want):
                        raise ProofFailure(
                            "MASK_CONTRACT_FAILURE",
                            f"seed {model_seed}/{file_id}/{branch}: non-reproducible {got} vs {want}",
                        )
            if masks_repeat[file_id] != masks[file_id] or masks_reversed[file_id] != masks[file_id]:
                raise ProofFailure(
                    "MASK_CONTRACT_FAILURE",
                    f"seed {model_seed}/{file_id}: mask not order/replay stable",
                )
        after = snapshot_state(model, bank)
        if before != after:
            raise ProofFailure(
                "RESTORE_ONLY_OR_ISOLATION_FAILURE",
                f"seed {model_seed}: model/bank state changed during scoring",
            )
        if sha256_file(checkpoint_path) != digest_before:
            raise ProofFailure("PROVENANCE_MISMATCH", f"seed {model_seed}: checkpoint bytes changed")
        if sha256_file(checkpoint_path) != CHECKPOINT_SHA256[model_seed]:
            raise ProofFailure("PROVENANCE_MISMATCH", f"seed {model_seed}: post-score hash drift")
        negative_cases: list[dict[str, str]] = []
        if args.run_negative_cases and model_seed == MODEL_SEEDS[0]:
            negative_cases = run_negative_cases(patchifier, cfg, roster, mapping)
        per_seed[str(model_seed)] = {
            "checkpoint_config_sha256": checkpoint_config_digest,
            "checkpoint": CHECKPOINT_FILES[model_seed],
            "checkpoint_sha256": CHECKPOINT_SHA256[model_seed],
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "step": CHECKPOINT_STEP,
            "bank_digest": bank_digest,
            "bank_shape": [BANK_ROWS, BANK_DIM],
            "bank_k": BANK_K,
            "scores": scores,
            "masks": masks,
            "details": details,
            "state_before": before,
            "state_after": after,
            "negative_cases": negative_cases,
        }
        del model, bank
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_s = round(time.time() - t0, 1)
    if torch.cuda.is_available():
        peak_vram_mib = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    record = {
        "contract": "experiments/sprint21-disposable-scorer-proof-v1.md",
        "base_commit": BASE_COMMIT,
        "worktree_root": str(worktree_root),
        "checkpoint_root": str(checkpoint_root),
        "config_sha256": observed_config_digest,
        "metric_code_sha256": metric_provenance["digest"],
        "metric_file_sha256": metric_provenance["files"],
        "role_binding_sha256": binding_provenance["digest"],
        "role_binding_json_sha256": binding_provenance["file_sha256"],
        "role_binding_profile_id": binding_provenance["profile_id"],
        "mapping_sha256": observed_mapping_digest,
        "uv_lock_sha256": source_hashes["uv.lock"],
        "channel_order": list(CHANNEL_ORDER),
        "source_hashes": source_hashes,
        "runtime": runtime,
        "device": args.device,
        "elapsed_s": elapsed_s,
        "peak_vram_mib": peak_vram_mib,
        "per_seed": per_seed,
    }
    (output_dir / "sprint21_task3_proof.json").write_text(
        json.dumps(record, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"elapsed_s": elapsed_s, "peak_vram_mib": peak_vram_mib,
                      "scores": {s: per_seed[s]["scores"] for s in per_seed}}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProofFailure as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
