"""Sprint 24 Task 7-M4 bounded capture runner (persistent embedding cache).

Persists the frozen weekly/monthly study's exact per-file B0 file embeddings under
the declared cache contract, at the study-required ``s24_feature_cache_identity``
(device ``cuda``, support digest over the study's full required union), so the
admission CLI can prove 26,262/26,262 reusable rows **before** any movement is
computed. It is the only component that performs a forward for this study.

Modes:

- ``fixture`` — no model, no forward. Builds the plan from the accepted metadata,
  writes bounded fixture entries through the same writer, and proves that the
  admission gate rejects them for their ``fixture-no-forward`` provenance. Local
  only; never study input.
- ``smoke`` — bounded GPU capture of 8 files per history per seed (96 forwards),
  persisted with ``gpu-forward-smoke`` provenance. Proves the public capture path,
  writer, sidecar and identity plumbing; never study input.
- ``full`` — the bounded full capture of 8,754 rows per seed (26,262 forwards),
  persisted with ``gpu-forward`` provenance at the study-required identity, and
  followed by an in-process inspection/coverage/admission proof. Requires a
  matching completed smoke record from the same execution commit.

Cost: the owner-authorized cumulative cap is 54,000 forwards with 26,431 already
spent, so at most 27,569 new forwards are available. ``smoke`` plans 96 and
``full`` plans 26,262; the runner refuses any plan that would exceed the cap and
budgets no automatic retry. No movement, contrast, uncertainty or disposition is
computed here.

Deployment boundary: this runner is new code. It must pass the readiness review
and a fresh Bronze checkpoint before it is executed remotely, and the remote
checkout must be the exact pushed commit with a clean worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from representation import sprint24_weekly_monthly_execution as execution  # noqa: E402
from representation import sprint24_weekly_monthly_study as study  # noqa: E402
from representation.sprint18_harness import (  # noqa: E402
    Sprint18ArmConfig,
    arm_config_sha256,
    load_task6_binding,
    validate_arm_config,
)
from representation.sprint24_context_audit import canonical_json, sha256_object  # noqa: E402
from representation.sprint24_features import (  # noqa: E402
    assert_s24_frozen_unchanged,
    capture_s24_file_features,
    restore_s24_b0_bundle,
    s24_device_identity,
    s24_live_state_digests,
)
from representation.sprint24_learners import enable_s24_deterministic_execution  # noqa: E402
from synth.dataset import load_sample_bytes  # noqa: E402

#: Frozen study digest (Task 7-M3 checkpoint ``0a17ffa7``).
DEFAULT_STUDY_SHA256 = "6f96c2750c15552bcdf25678a51ed7b24a1f7aecdb5914bf5985d3012d1daefd"

#: Frozen protocol document digest.
DEFAULT_PROTOCOL_SHA256 = "8da4afda89dd303099843eecda546cdc337696525ece0eed59f400d1031d8b63"

#: Owner-authorized cumulative forward cap for this amendment (Task 7-M4-A01).
AUTHORIZED_CUMULATIVE_CAP = 54_000

#: Forwards already spent before this attempt (accepted G2 accounting).
CUMULATIVE_BEFORE = study.INHERITED_CUMULATIVE_FORWARDS

#: Declared smoke size: 8 files per history x 4 histories x 3 seeds.
SMOKE_FILES_PER_HISTORY = 8
SMOKE_FORWARDS = SMOKE_FILES_PER_HISTORY * 4 * len(study.REPRESENTATION_SEEDS)

#: Accepted asset roots (server-side; overridable for a verified transfer).
DEFAULT_B0_ROOT = Path("/home/trietlm/s18-t23-a04-runs/full")
DEFAULT_BANK_ROOT = Path("/home/trietlm/s18-t32-a02-runs/scores")

HANDOFF = REPO / "experiments/sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
LOCK = REPO / "experiments/sprint18-task14-config-lock.json"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def worktree_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain=v1"], text=True
        ).strip()
    )


def write_json(path: Path, payload: object) -> str:
    text = canonical_json(payload) + "\n"
    path.write_text(text, encoding="utf-8", newline="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_expected(values: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected argument needs <key>=<sha256>: {value!r}")
        key, digest = value.split("=", 1)
        expected[key.strip()] = digest.strip()
    return expected


def verify_exact_commit(expected: str) -> None:
    if not expected or len(expected) != 40:
        raise SystemExit("--expected-execution-commit must be a 40-character SHA")
    actual = source_commit()
    if actual != expected:
        raise SystemExit(f"execution commit mismatch: {actual} != {expected}")
    if worktree_dirty():
        raise SystemExit("exact source worktree is dirty; refusing to run")


def device_provenance() -> dict[str, object]:
    deterministic = enable_s24_deterministic_execution("cuda")
    if not deterministic["deterministic_algorithms"]:
        raise RuntimeError("deterministic CUDA algorithms were not enabled")
    props = torch.cuda.get_device_properties(0)
    return {
        "requested": "cuda",
        "identity": s24_device_identity("cuda"),
        "deterministic": deterministic,
        "device_name": props.name,
        "cuda_build": torch.version.cuda,
        "parameter_dtype": "float32",
    }


def load_plan(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    edges_raw = json.loads(Path(args.edges).read_text(encoding="utf-8"))
    if not isinstance(edges_raw, dict):
        raise SystemExit("edges file must map history id to its 11 decile edges")
    edges = {str(key): [float(value) for value in values] for key, values in edges_raw.items()}
    expected_manifests = parse_expected(args.expected_manifest_sha256)
    if set(expected_manifests) != set(edges):
        raise SystemExit(
            "expected manifest digests must cover exactly the audited histories: "
            f"{sorted(set(edges) ^ set(expected_manifests))}"
        )
    manifests: dict[str, dict[str, object]] = {}
    for history_id in sorted(edges):
        path = Path(args.data_root) / "DEVELOPMENT" / history_id / "manifest.json"
        if not path.is_file():
            raise SystemExit(f"missing accepted manifest: {path}")
        digest = sha256_path(path)
        if digest != expected_manifests[history_id]:
            raise SystemExit(
                f"accepted manifest digest mismatch for {history_id}: "
                f"{digest} != {expected_manifests[history_id]}"
            )
        manifests[history_id] = json.loads(path.read_text(encoding="utf-8"))
    plan = execution.required_row_plan(manifests, edges)
    totals = plan["totals"]
    assert isinstance(totals, dict)
    if not all(
        bool(totals[key])
        for key in ("matches_declared_envelope", "matches_declared_span", "matches_declared_reference")
    ):
        raise SystemExit(f"required row plan disagrees with the frozen envelope: {totals}")
    g2_record = execution.read_accepted_g2(Path(args.g2_accepted_root))
    inputs = execution.identity_inputs(g2_record)
    identities = execution.required_cache_identities(plan, inputs)
    return plan, identities, g2_record, inputs, manifests


def plan_cost(mode: str) -> dict[str, object]:
    if mode == "fixture":
        planned = 0
    elif mode == "smoke":
        planned = SMOKE_FORWARDS
    else:
        # The full capture always follows a completed smoke from the same commit.
        planned = SMOKE_FORWARDS + int(study.DECLARED_MAX_ROWS_TOTAL)
    cumulative = CUMULATIVE_BEFORE + planned
    if cumulative > AUTHORIZED_CUMULATIVE_CAP:
        raise SystemExit(
            f"INFEASIBLE: cumulative {cumulative} exceeds the authorized cap "
            f"{AUTHORIZED_CUMULATIVE_CAP}"
        )
    return {
        "mode": mode,
        "forwards_planned": planned,
        "cumulative_before": CUMULATIVE_BEFORE,
        "authorized_cumulative_cap": AUTHORIZED_CUMULATIVE_CAP,
        "cumulative_after": cumulative,
        "remaining_after": AUTHORIZED_CUMULATIVE_CAP - cumulative,
        "automatic_retry_budget": 0,
    }


def assert_empty_store(root: Path) -> None:
    if root.exists() and any(root.glob("*.npz")):
        raise SystemExit(
            f"cache store {root} already contains entries; refusing to overwrite custody"
        )


def load_smoke_record(path: Path, expected_commit: str) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"smoke record is missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise SystemExit("smoke record is not a JSON object")
    if str(record.get("mode")) != "smoke" or str(record.get("status")) != "COMPLETE":
        raise SystemExit("smoke record is not a completed smoke run")
    if str(record.get("execution_commit")) != expected_commit:
        raise SystemExit("smoke record belongs to a different execution commit")
    if int(record.get("forwards_actual", -1)) != SMOKE_FORWARDS:
        raise SystemExit("smoke record forward count differs from the declared smoke size")
    if str(record.get("device", {}).get("identity")) != execution.REQUIRED_DEVICE_IDENTITY:
        raise SystemExit("smoke record was not produced on the required device")
    if str(record.get("study_sha256")) != study.study_sha256():
        raise SystemExit("smoke record belongs to a different study freeze")
    if str(record.get("protocol_sha256")) != DEFAULT_PROTOCOL_SHA256:
        raise SystemExit("smoke record belongs to a different protocol document")
    return {"path": str(path), "sha256": sha256_path(path), "record": record}


def materialize(directory: Path, manifest: dict[str, object], file_ids: list[str]) -> dict[str, object]:
    shard_of = {fid: shard["path"] for shard in manifest["shards"] for fid in shard["file_ids"]}
    missing = [fid for fid in file_ids if fid not in shard_of]
    if missing:
        raise RuntimeError(f"accepted manifest lost shard custody for {len(missing)} files")
    payloads: dict[str, bytes] = {}
    by_shard: dict[str, list[str]] = {}
    for fid in file_ids:
        by_shard.setdefault(str(shard_of[fid]), []).append(fid)
    for relative, ids in sorted(by_shard.items()):
        with zipfile.ZipFile(directory / relative) as archive:
            for fid in ids:
                payloads[fid] = archive.read(fid + ".npz")
    if set(payloads) != set(file_ids):
        raise RuntimeError("waveform materialization lost support IDs")
    return {fid: load_sample_bytes(payloads[fid]) for fid in file_ids}


def capture_group(bundle: object, samples: list[object], history_id: str, seed: int) -> np.ndarray:
    captured = capture_s24_file_features(
        model=bundle.model,  # type: ignore[attr-defined]
        bank=bundle.bank,  # type: ignore[attr-defined]
        samples=samples,
        history_id=history_id,
        model_seed=seed,
        device="cuda",
    )
    embeddings = np.stack(
        [feature.embedding.detach().cpu().float().numpy() for feature in captured]
    ).astype(np.float32)
    if embeddings.shape != (len(samples), study.EMBEDDING_DIM):
        raise RuntimeError(
            f"capture shape {embeddings.shape} differs from the planned rows"
        )
    if not np.isfinite(embeddings).all():
        raise RuntimeError("capture produced non-finite embeddings")
    return embeddings


def restore_bundles(args: argparse.Namespace, g2_record: dict[str, object]) -> tuple[dict[int, object], dict[str, object]]:
    preflight = g2_record["preflight"]
    assert isinstance(preflight, dict)
    device = device_provenance()
    binding = load_task6_binding(HANDOFF)
    if binding.executable_binding_sha256 != execution.ACCEPTED_BINDING_SHA256:
        raise RuntimeError("accepted Sprint18 binding mismatch")
    lock_doc = json.loads(LOCK.read_text(encoding="utf-8"))
    b0_root = Path(args.b0_root)
    bank_root = Path(args.bank_root)
    bundles: dict[int, object] = {}
    for seed in study.REPRESENTATION_SEEDS:
        checkpoint_path = b0_root / f"seed-{seed}" / "checkpoints" / f"b0-seed{seed}-epoch100.pt"
        bank_path = bank_root / f"B0-FULL-{seed}.json.fit-bank" / "bank.pt"
        expected_checkpoint = str(preflight["checkpoints"][str(seed)])
        expected_bank_file = str(preflight["bank_files"][str(seed)])
        expected_bank_tensor = str(preflight["banks"][str(seed)])
        for path, digest in (
            (checkpoint_path, expected_checkpoint),
            (bank_path, expected_bank_file),
        ):
            if not path.is_file():
                raise RuntimeError(f"required accepted asset missing: {path}")
            if sha256_path(path) != digest:
                raise RuntimeError(f"accepted asset digest mismatch: {path}")
        arm = validate_arm_config(
            Sprint18ArmConfig(**lock_doc["locked_configs"][f"B0-FULL/seed-{seed}"]),
            require_locked=True,
        )
        if arm.model_seed != seed:
            raise RuntimeError(f"accepted B0 arm seed mismatch: {seed}")
        if arm_config_sha256(arm) != str(preflight["arms"][str(seed)]):
            raise RuntimeError(f"accepted arm-config digest mismatch: {seed}")
        bundles[seed] = restore_s24_b0_bundle(
            binding=binding,
            arm=arm,
            checkpoint_path=checkpoint_path,
            bank_path=bank_path,
            expected_checkpoint_sha256=expected_checkpoint,
            expected_bank_sha256=expected_bank_tensor,
            device="cuda",
        )
    return bundles, device


def append_progress(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(canonical_json(payload) + "\n")


def run_fixture(args: argparse.Namespace, plan: dict[str, object], identities: dict[str, object], inputs: dict[str, object], cache_out: Path) -> dict[str, object]:
    records = []
    for group in identities["groups"].values():  # type: ignore[union-attr]
        file_ids = list(plan["per_history"][group["history_id"]]["required_file_ids"])[
            :SMOKE_FILES_PER_HISTORY
        ]
        subset_support = sha256_object(sorted(file_ids))
        matrix = np.zeros((len(file_ids), study.EMBEDDING_DIM), dtype=np.float32)
        record = execution.write_cache_entry(
            cache_out,
            history_id=str(group["history_id"]),
            model_seed=int(group["model_seed"]),
            file_ids=file_ids,
            embeddings=matrix,
            inputs=inputs,
            support_sha256=subset_support,
            provenance=execution.CACHE_PROVENANCE_FIXTURE,
            capture_commit=None,
        )
        expected_identity = execution.cache_identity(
            inputs, model_seed=int(group["model_seed"]), support_sha256=subset_support
        )
        if record["cache_identity"] != expected_identity:
            raise RuntimeError("fixture writer identity mismatch")
        records.append(record)
    inventory = execution.inspect_cache_stores([cache_out])
    coverage = execution.cache_coverage(plan, identities, inventory)
    refused = True
    try:
        execution.admit_execution(coverage, inventory)
        refused = False
    except execution.CachePreflightFailure:
        refused = True
    report = {
        "runner": "sprint24_task7m4_capture_runner.py",
        "mode": "fixture",
        "status": "COMPLETE",
        "fixture_no_forward": True,
        "forwards_actual": 0,
        "model_loads": 0,
        "cache_out": str(cache_out),
        "written_entries": records,
        "inspection": {
            "npz_count": inventory["npz_count"],
            "valid_entries": inventory["valid_entries"],
            "invalid_entries": inventory["invalid_entries"],
            "issues": [issue for entry in inventory["entries"] for issue in entry["issues"]],
        },
        "coverage": {
            "total_required_rows": coverage["total_required_rows"],
            "total_available_rows": coverage["total_available_rows"],
            "coverage_complete": coverage["coverage_complete"],
        },
        "admission_refused": refused,
        "study_sha256": study.study_sha256(),
    }
    if not refused:
        raise RuntimeError("fixture provenance was admitted; the gate is not strict")
    if coverage["total_available_rows"] != 0:
        raise RuntimeError("fixture rows were counted as reusable study rows")
    return report


def run_capture(args: argparse.Namespace, mode: str, plan: dict[str, object], identities: dict[str, object], g2_record: dict[str, object], inputs: dict[str, object], manifests: dict[str, dict[str, object]], out: Path, cache_out: Path) -> dict[str, object]:
    verify_exact_commit(args.expected_execution_commit)
    commit = args.expected_execution_commit
    smoke_record = load_smoke_record(Path(args.smoke_record), commit) if mode == "full" else None
    bundles, device = restore_bundles(args, g2_record)
    frozen_before = {seed: s24_live_state_digests(bundle.model, bundle.bank) for seed, bundle in bundles.items()}  # type: ignore[attr-defined]
    progress = out / "capture-progress.jsonl"
    per_history = plan["per_history"]
    assert isinstance(per_history, dict)
    entries: list[dict[str, object]] = []
    walls: dict[str, float] = {}
    forwards = 0
    for history_id, block in sorted(per_history.items()):
        required_ids = list(block["required_file_ids"])
        if mode == "smoke":
            required_ids = required_ids[:SMOKE_FILES_PER_HISTORY]
        support_sha256 = sha256_object(sorted(required_ids))
        manifest = manifests[history_id]
        directory = Path(args.data_root) / "DEVELOPMENT" / history_id
        samples = materialize(directory, manifest, required_ids)
        for seed in study.REPRESENTATION_SEEDS:
            started = time.perf_counter()
            matrix = capture_group(bundles[seed], [samples[fid] for fid in required_ids], history_id, seed)
            wall = time.perf_counter() - started
            record = execution.write_cache_entry(
                cache_out,
                history_id=history_id,
                model_seed=int(seed),
                file_ids=required_ids,
                embeddings=matrix,
                inputs=inputs,
                support_sha256=support_sha256,
                provenance=(
                    execution.CACHE_PROVENANCE_GPU if mode == "full" else execution.CACHE_PROVENANCE_SMOKE
                ),
                capture_commit=commit if mode == "full" else None,
            )
            expected_identity = (
                str(identities["groups"][f"{history_id}|{seed}"]["cache_identity"])
                if mode == "full"
                else execution.cache_identity(
                    inputs, model_seed=int(seed), support_sha256=support_sha256
                )
            )
            if record["cache_identity"] != expected_identity:
                raise RuntimeError(
                    f"captured entry identity differs from the expected identity for "
                    f"{history_id}|{seed}"
                )
            forwards += len(required_ids)
            walls[f"{history_id}|{seed}"] = wall
            entries.append(
                {
                    "history_id": history_id,
                    "model_seed": int(seed),
                    "n_rows": len(required_ids),
                    "support_sha256": support_sha256,
                    "cache_identity": record["cache_identity"],
                    "npz_sha256": record["npz_sha256"],
                    "sidecar_sha256": record["sidecar_sha256"],
                    "wall_s": wall,
                }
            )
            append_progress(
                progress,
                {
                    "history_id": history_id,
                    "model_seed": int(seed),
                    "n_rows": len(required_ids),
                    "npz_sha256": record["npz_sha256"],
                    "wall_s": wall,
                },
            )
    frozen_after = {seed: s24_live_state_digests(bundle.model, bundle.bank) for seed, bundle in bundles.items()}  # type: ignore[attr-defined]
    for seed in study.REPRESENTATION_SEEDS:
        assert_s24_frozen_unchanged(frozen_before[seed], frozen_after[seed])
    inventory = execution.inspect_cache_stores([cache_out])
    coverage = execution.cache_coverage(plan, identities, inventory)
    admitted = False
    admission: dict[str, object] = {}
    try:
        admission = execution.admit_execution(coverage, inventory)
        admitted = True
    except execution.CachePreflightFailure as failure:
        admission = dict(failure.report)
    if mode == "full" and not admitted:
        raise RuntimeError(
            "full capture did not produce 26,262 reusable rows: "
            + "; ".join(str(item) for item in admission["blockers"])
        )
    if mode == "smoke" and admitted:
        raise RuntimeError("smoke provenance was admitted; the gate is not strict")
    smoke_forwards = SMOKE_FORWARDS if mode == "full" else 0
    planned = (
        SMOKE_FORWARDS + int(study.DECLARED_MAX_ROWS_TOTAL)
        if mode == "full"
        else SMOKE_FORWARDS
    )
    return {
        "runner": "sprint24_task7m4_capture_runner.py",
        "mode": mode,
        "status": "COMPLETE",
        "execution_commit": commit,
        "study_sha256": study.study_sha256(),
        "device": device,
        "cache_out": str(cache_out),
        "forwards_actual": forwards,
        "forwards_planned": planned,
        "smoke_forwards_counted": smoke_forwards,
        "cumulative_after": CUMULATIVE_BEFORE + smoke_forwards + forwards,
        "authorized_cumulative_cap": AUTHORIZED_CUMULATIVE_CAP,
        "automatic_retry_budget": 0,
        "smoke_record": smoke_record,
        "entries": entries,
        "capture_wall_s": walls,
        "live_state": {
            "model_and_bank_unchanged": True,
            "frozen_state_sha256": {
                str(seed): frozen_after[seed] for seed in sorted(frozen_after)
            },
        },
        "inspection": {
            "npz_count": inventory["npz_count"],
            "valid_entries": inventory["valid_entries"],
            "invalid_entries": inventory["invalid_entries"],
        },
        "coverage": {
            "total_required_rows": coverage["total_required_rows"],
            "total_available_rows": coverage["total_available_rows"],
            "total_missing_rows": coverage["total_missing_rows"],
            "coverage_complete": coverage["coverage_complete"],
        },
        "admission": admission,
        "movement_computed": False,
        "outcomes_computed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["fixture", "smoke", "full"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--g2-accepted-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-out", required=True)
    parser.add_argument("--expected-manifest-sha256", action="append", default=[])
    parser.add_argument("--expected-study-sha256", default=DEFAULT_STUDY_SHA256)
    parser.add_argument("--expected-protocol-sha256", default=DEFAULT_PROTOCOL_SHA256)
    parser.add_argument("--expected-execution-commit", default=None)
    parser.add_argument("--smoke-record", default=None)
    parser.add_argument("--b0-root", default=str(DEFAULT_B0_ROOT))
    parser.add_argument("--bank-root", default=str(DEFAULT_BANK_ROOT))
    parser.add_argument("--protocol-doc", default=study.PROTOCOL_DOC)
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out)
    cache_out = Path(args.cache_out)
    out.mkdir(parents=True, exist_ok=True)
    if Path(args.protocol_doc).is_absolute():
        protocol_path = Path(args.protocol_doc)
    else:
        protocol_path = REPO / args.protocol_doc
    protocol_sha256 = sha256_path(protocol_path) if protocol_path.is_file() else None
    if protocol_sha256 != args.expected_protocol_sha256:
        raise SystemExit(
            f"frozen protocol digest mismatch: {protocol_sha256} != "
            f"{args.expected_protocol_sha256}"
        )
    if study.study_sha256() != args.expected_study_sha256:
        raise SystemExit(
            f"frozen study digest mismatch: {study.study_sha256()} != "
            f"{args.expected_study_sha256}"
        )
    plan, identities, g2_record, inputs, manifests = load_plan(args)
    cost = plan_cost(args.mode)
    assert_empty_store(cache_out)

    if args.mode == "full" and not args.smoke_record:
        raise SystemExit("full capture requires --smoke-record from the same commit")

    if args.mode == "fixture":
        report = run_fixture(args, plan, identities, inputs, cache_out)
    else:
        report = run_capture(
            args, args.mode, plan, identities, g2_record, inputs, manifests, out, cache_out
        )
    finished = time.time()
    report["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "platform": platform.platform(),
    }
    report["timing"] = {
        "started_unix": started,
        "finished_unix": finished,
        "wall_s": finished - started,
    }
    report["cost"] = cost
    report["protocol_sha256"] = protocol_sha256
    data_root = Path(args.data_root)
    report["inputs"] = {
        "source_commit": source_commit(),
        "worktree_dirty": worktree_dirty(),
        "edges": {
            "path": str(Path(args.edges).resolve()),
            "sha256": sha256_path(Path(args.edges)),
        },
        "manifests": {
            history_id: {
                "path": str(
                    (data_root / "DEVELOPMENT" / history_id / "manifest.json").resolve()
                ),
                "sha256": sha256_path(
                    data_root / "DEVELOPMENT" / history_id / "manifest.json"
                ),
                "bytes": (
                    data_root / "DEVELOPMENT" / history_id / "manifest.json"
                ).stat().st_size,
            }
            for history_id in sorted(manifests)
        },
        "accepted_g2": g2_record["digests"],
    }
    report["plan"] = {
        "totals": plan["totals"],
        "per_history": {
            history_id: {
                "n_reference_rows": block["n_reference_rows"],
                "n_span_rows": block["n_span_rows"],
                "n_rows": block["n_rows"],
                "union_ids_sha256": block["union_ids_sha256"],
            }
            for history_id, block in plan["per_history"].items()  # type: ignore[union-attr]
        },
    }
    report["identities"] = {
        key: {
            "history_id": value["history_id"],
            "model_seed": value["model_seed"],
            "support_sha256": value["support_sha256"],
            "cache_identity": value["cache_identity"],
        }
        for key, value in identities["groups"].items()  # type: ignore[union-attr]
    }
    name = {"fixture": "fixture-smoke.json", "smoke": "smoke-report.json", "full": "capture-report.json"}[args.mode]
    digest = write_json(out / name, report)
    write_json(
        out / "store-hashes.json",
        {
            "mode": args.mode,
            "cache_out": str(cache_out),
            "entries": report.get("entries", report.get("written_entries")),
        },
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": report["status"],
                "forwards_actual": report.get("forwards_actual", 0),
                "cumulative_after": report.get("cumulative_after", CUMULATIVE_BEFORE),
                "admission": report.get("admission", {}).get("status") if isinstance(report.get("admission"), dict) else "REFUSED",
                "report": str(out / name),
                "report_sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
