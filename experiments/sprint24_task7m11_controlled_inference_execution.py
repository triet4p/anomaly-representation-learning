"""Sprint 24 Task 7-M11 frozen controlled-corpus inference and movement CLI.

Real execution entry point under the accepted M10 prospective contract
(``f4f92a60ee1782ccc0ec75587e9df5d1471e4045e201c43478893d294b5757f7``):

1. verify the M9 corpus, the M10 contract and the exact encoder identities
   before any computation (canonical freeze + pinned digests + contract
   binding gate);
2. bounded real smoke (8 files x 1 history x 3 seeds = 24 forwards) through
   the accepted frozen B0 capture path, proving real admission/refusal;
3. full frozen capture (8,280 files x 3 seeds = 24,840 forwards) with resume
   per ``(history, seed)`` group and no duplicate forwards;
4. identity-strict admission of the cache, then the frozen endpoints,
   nulls, sensitivities, history-level uncertainty and primary/replication
   summaries on admitted real embeddings.

Usage:

```bash
python experiments/sprint24_task7m11_controlled_inference_execution.py \
  --corpus data/generated/sprint24_controlled_v2 \
  --audit artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json \
  --contract artifacts/sprint-24/evidence/s24-7-m10-a02/prospective-contract.json \
  --cache-root <identity-bound cache dir> \
  --out artifacts/sprint-24/evidence/s24-7-m11-a01 \
  --mode smoke
python experiments/sprint24_task7m11_controlled_inference_execution.py \
  --corpus ... --audit ... --contract ... --cache-root ... \
  --out artifacts/sprint-24/evidence/s24-7-m11-a01 \
  --mode full --b0-root <ckpt dir> --bank-root <bank dir> \
  --expected-execution-commit <40-hex HEAD>
```

``smoke`` needs bundle roots only to attempt the first group restore; with
no usable bundle it records the exact fail-closed refusal. ``full``
requires the exact execution commit and restores the three accepted frozen
B0 bundles before any forward. Non-empty ``--out`` refuses
(``REFUSED-OVERWRITE``); any digest/identity drift refuses before capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from representation import sprint24_controlled_inference_execution as m11  # noqa: E402
from representation import sprint24_controlled_inference_readiness as readiness  # noqa: E402
from representation import sprint24_weekly_monthly_execution as execution  # noqa: E402
from representation.sprint24_context_audit import canonical_json, sha256_object  # noqa: E402


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


def write_json(path: Path, payload: object) -> str:
    text = canonical_json(payload) + "\n"
    path.write_text(text, encoding="utf-8", newline="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _restore_bundles(args: argparse.Namespace, preflight: dict[str, Any]) -> dict[int, Any]:
    """Restore the three accepted frozen B0 bundles on the pinned device."""
    sys.path.insert(0, str(REPO / "src"))
    from representation.sprint18_harness import (
        Sprint18ArmConfig,
        arm_config_sha256,
        load_task6_binding,
        validate_arm_config,
    )
    from representation.sprint24_features import restore_s24_b0_bundle

    binding = load_task6_binding(
        REPO / "experiments/sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    if binding.executable_binding_sha256 != execution.ACCEPTED_BINDING_SHA256:
        raise ValueError("REFUSED-IDENTITY: accepted Sprint18 binding mismatch")
    lock_doc = json.loads(
        (REPO / "experiments/sprint18-task14-config-lock.json").read_text(encoding="utf-8")
    )
    bundles: dict[int, Any] = {}
    for seed in readiness.FROZEN_ENCODER_SEEDS:
        checkpoint_path = (
            Path(args.b0_root) / f"seed-{seed}" / "checkpoints" / f"b0-seed{seed}-epoch100.pt"
        )
        bank_path = (
            Path(args.bank_root) / f"B0-FULL-{seed}.json.fit-bank" / "bank.pt"
        )
        expected_checkpoint = str(preflight["checkpoints"][str(seed)])
        expected_bank_file = str(preflight["bank_files"][str(seed)])
        for path, digest in (
            (checkpoint_path, expected_checkpoint),
            (bank_path, expected_bank_file),
        ):
            if not path.is_file():
                raise ValueError(f"REFUSED-MISSING-PREREQUISITE: required accepted asset absent: {path}")
            if sha256_path(path) != digest:
                raise ValueError(f"REFUSED-IDENTITY: accepted asset digest mismatch: {path}")
        arm = validate_arm_config(
            Sprint18ArmConfig(**lock_doc["locked_configs"][f"B0-FULL/seed-{seed}"]),
            require_locked=True,
        )
        if arm.model_seed != seed:
            raise ValueError(f"REFUSED-IDENTITY: accepted B0 arm seed mismatch: {seed}")
        if arm_config_sha256(arm) != str(preflight["arms"][str(seed)]):
            raise ValueError(f"REFUSED-IDENTITY: accepted arm-config digest mismatch: {seed}")
        bundles[seed] = restore_s24_b0_bundle(
            binding=binding,
            arm=arm,
            checkpoint_path=checkpoint_path,
            bank_path=bank_path,
            expected_checkpoint_sha256=expected_checkpoint,
            expected_bank_sha256=str(preflight["banks"][str(seed)]),
            device="cuda",
        )
    return bundles


def _load_waveform_samples(
    corpus: Path,
    relpath_of: dict[str, str],
    file_ids: list[str],
) -> list[Any]:
    """Load waveform samples through the supported interface (label-blind)."""
    sys.path.insert(0, str(REPO / "src"))
    from synth.dataset import load_sample_bytes

    samples: list[Any] = []
    for fid in file_ids:
        rel = relpath_of.get(fid)
        if rel is None:
            raise ValueError(f"REFUSED-IDENTITY: manifest has no relpath for {fid}")
        payload = (corpus / rel).read_bytes()
        sample = load_sample_bytes(payload)
        if sample.file_id != fid:
            raise ValueError(
                f"REFUSED-IDENTITY: waveform file-ID mismatch: {sample.file_id} != {fid}"
            )
        samples.append(sample)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", required=True, choices=["smoke", "full"])
    parser.add_argument("--b0-root", default=None)
    parser.add_argument("--bank-root", default=None)
    parser.add_argument("--expected-execution-commit", default=None)
    parser.add_argument("--g2-preflight", default=None)
    args = parser.parse_args()

    corpus = Path(args.corpus)
    audit_path = Path(args.audit)
    contract_path = Path(args.contract)
    cache_root = Path(args.cache_root)
    out = Path(args.out)
    if out.is_dir() and any(out.iterdir()):
        raise SystemExit(f"REFUSED-OVERWRITE: output {out} already holds files")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if not audit_path.is_file():
        raise SystemExit(f"REFUSED-MISSING-CORPUS: observed audit absent: {audit_path}")
    if not contract_path.is_file():
        raise SystemExit(f"REFUSED-MISSING-CORPUS: prospective contract absent: {contract_path}")

    manifest, freeze, preflight = readiness.load_inputs(
        corpus, Path(args.g2_preflight) if args.g2_preflight else None
    )
    freeze_ok, freeze_detail = readiness.verify_canonical_freeze(freeze)
    if not freeze_ok:
        raise SystemExit(f"REFUSED-DIGEST-DRIFT: {freeze_detail}")
    manifest_ok, manifest_detail = readiness.verify_pinned_artifact(
        corpus / "corpus-manifest.json", "corpus manifest"
    )
    if not manifest_ok:
        raise SystemExit(f"REFUSED-DIGEST-DRIFT: {manifest_detail}")
    audit_ok, audit_detail = readiness.verify_pinned_artifact(audit_path, "corpus audit")
    if not audit_ok:
        raise SystemExit(f"REFUSED-DIGEST-DRIFT: {audit_detail}")
    preflight_path = (
        Path(args.g2_preflight)
        if args.g2_preflight
        else (
            REPO / "artifacts" / "sprint-24" / "evidence" / "s24-7-m4-a01"
            / "g2-accepted" / "preflight.json"
        )
    )
    preflight_ok, preflight_detail = readiness.verify_pinned_artifact(
        preflight_path, "G2 preflight", allowed={preflight_path}
    )
    if not preflight_ok:
        raise SystemExit(f"REFUSED-DIGEST-DRIFT: {preflight_detail}")

    custody = readiness.validate_custody_inputs(preflight)

    envelope = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict) or "contract" not in envelope:
        raise SystemExit("REFUSED-IDENTITY: prospective contract envelope is malformed")
    contract = envelope["contract"]
    if not isinstance(contract, dict):
        raise SystemExit("REFUSED-IDENTITY: prospective contract payload is malformed")
    if sha256_path(contract_path) == "":
        raise SystemExit("REFUSED-IDENTITY: contract file is empty")
    bound = m11.verify_contract_binding(
        contract,
        m11.EXPECTED_CONTRACT_SHA256,
        manifest,
        json.loads(audit_path.read_text(encoding="utf-8")),
        custody,
    )
    identity_plan = bound["plan"]
    assert isinstance(identity_plan, dict)
    plan_groups = identity_plan.get("groups")
    assert isinstance(plan_groups, dict)

    capture = m11.bound_capture_plan(manifest, custody)
    universe_raw = capture["file_ids"]
    assert isinstance(universe_raw, dict)
    universe: dict[str, list[str]] = {
        str(history_id): [str(fid) for fid in ids]  # type: ignore[union-attr]
        for history_id, ids in universe_raw.items()
    }
    relpath_raw = capture["relpath_of"]
    assert isinstance(relpath_raw, dict)
    relpaths: dict[str, str] = {str(fid): str(rel) for fid, rel in relpath_raw.items()}
    m11.set_support_universe(universe)
    inputs = m11.capture_inputs(custody)

    audit_record: dict[str, Any] = json.loads(audit_path.read_text(encoding="utf-8"))

    run_record: dict[str, Any] = {
        "task": m11.TASK_ID,
        "mode": args.mode,
        "source_commit": source_commit(),
        "expected_contract_sha256": m11.EXPECTED_CONTRACT_SHA256,
        "freeze_check": freeze_detail,
        "manifest_check": manifest_detail,
        "audit_check": audit_detail,
        "preflight_check": preflight_detail,
        "contract_binding_verified_pre_execution": True,
        "forwards_executed_this_run": 0,
        "model_loads": 0,
        "resumed_groups": [],
        "captured_groups": [],
        "refused_groups": [],
    }

    if args.mode == "smoke":
        if not args.b0_root or not args.bank_root:
            run_record.update(
                {
                    "status": "REFUSED-PREREQUISITE",
                    "reason": (
                        "smoke capture needs --b0-root/--bank-root pointing at the "
                        "accepted checkpoint/bank bytes; CUDA device is required"
                    ),
                    "forwards": 0,
                }
            )
        else:
            try:
                bundles = _restore_bundles(args, preflight)
            except (ValueError, RuntimeError, OSError) as exc:
                run_record.update(
                    {"status": "REFUSED-PREREQUISITE", "reason": str(exc), "forwards": 0}
                )
            else:
                import numpy as _np

                from representation.sprint24_features import capture_s24_file_features

                first_history = sorted(universe)[m11.SMOKE_HISTORY_INDEX]
                smoke_ids = sorted(set(universe[first_history]))[: m11.SMOKE_FILES_PER_HISTORY]
                samples = _load_waveform_samples(corpus, relpaths, smoke_ids)
                t0 = time.time()
                for seed in readiness.FROZEN_ENCODER_SEEDS:
                    bundle = bundles[seed]
                    captured = capture_s24_file_features(
                        model=bundle.model,  # type: ignore[attr-defined]
                        bank=bundle.bank,  # type: ignore[attr-defined]
                        samples=samples,
                        history_id=first_history,
                        model_seed=int(seed),
                        device="cuda",
                    )
                    vectors = _np.stack(
                        [c.embedding.detach().cpu().float().numpy() for c in captured]  # type: ignore[union-attr]
                    )
                    m11.write_group_cache(
                        cache_root,
                        history_id=first_history,
                        model_seed=int(seed),
                        file_ids=smoke_ids,
                        embeddings=vectors.astype(_np.float32),
                        inputs=inputs,
                        support_sha256=sha256_object(sorted(set(universe[first_history]))),
                        provenance=execution.CACHE_PROVENANCE_SMOKE,
                        capture_commit=None,
                    )
                    run_record["forwards_executed_this_run"] = int(run_record["forwards_executed_this_run"]) + len(smoke_ids)
                run_record.update(
                    {
                        "status": "SMOKE-COMPLETE",
                        "smoke_history": first_history,
                        "smoke_files": len(smoke_ids),
                        "smoke_wall_s": time.time() - t0,
                    }
                )
        elapsed = time.time() - started
        run_record.update(
            {
                "elapsed_s": elapsed,
                "forwards": int(run_record["forwards_executed_this_run"]),
                "cache_root": str(cache_root),
            }
        )
        run_sha = write_json(out / "run.json", run_record)
        print(f"task={m11.TASK_ID} mode=smoke status={run_record.get('status')}")
        print(f"run={run_sha} forwards={run_record['forwards']}")
        return 0 if run_record.get("status") == "SMOKE-COMPLETE" else 2

    # --- full mode ---
    if not args.expected_execution_commit or not re.fullmatch(
        r"[0-9a-f]{40}", args.expected_execution_commit
    ):
        raise SystemExit("REFUSED-IDENTITY: full mode needs --expected-execution-commit (40-hex HEAD)")
    if args.expected_execution_commit != source_commit():
        raise SystemExit(
            f"REFUSED-IDENTITY: execution commit mismatch: {args.expected_execution_commit} != {source_commit()}"
        )
    if not args.b0_root or not args.bank_root:
        raise SystemExit("REFUSED-PREREQUISITE: full mode needs --b0-root/--bank-root")
    bundles = _restore_bundles(args, preflight)
    run_record["model_loads"] = len(bundles)

    from representation.sprint24_features import capture_s24_file_features

    import numpy as _np

    resumed: list[str] = []
    captured: list[str] = []
    forwards = 0
    plan_group_map: dict[str, Any] = dict(plan_groups)  # type: ignore[arg-type]
    for history_id in sorted(universe):
        full_ids = sorted(set(universe[history_id]))
        support_sha256 = sha256_object(full_ids)
        for seed in readiness.FROZEN_ENCODER_SEEDS:
            assert isinstance(seed, int)
            key = f"{history_id}|{seed}"
            spec = plan_group_map.get(key)
            assert isinstance(spec, Mapping)
            expected_identity = str(spec.get("cache_identity"))
            hit = m11.cached_group_complete(cache_root, str(history_id), int(seed), expected_identity)
            if hit is not None:
                n_hit = hit.get("n_rows")
                assert isinstance(n_hit, int)
                if n_hit == len(full_ids):
                    resumed.append(key)
                    continue
            samples = _load_waveform_samples(corpus, dict(relpaths), full_ids)
            bundle = bundles[seed]
            captured_features = capture_s24_file_features(
                model=bundle.model,  # type: ignore[attr-defined]
                bank=bundle.bank,  # type: ignore[attr-defined]
                samples=samples,
                history_id=history_id,
                model_seed=int(seed),
                device="cuda",
            )
            order = [c.file_id for c in captured_features]  # type: ignore[union-attr]
            if order != full_ids:
                raise SystemExit(f"REFUSED-IDENTITY: capture order drifted for {key}")
            vectors = _np.stack(
                [c.embedding.detach().cpu().float().numpy() for c in captured_features]  # type: ignore[union-attr]
            ).astype(_np.float32)
            if vectors.shape != (len(full_ids), 128):
                raise SystemExit(f"REFUSED-IDENTITY: capture shape drifted for {key}")
            if not bool(_np.isfinite(vectors).all()):
                raise SystemExit(f"REFUSED-IDENTITY: non-finite capture for {key}")
            m11.write_group_cache(
                cache_root,
                history_id=history_id,
                model_seed=int(seed),
                file_ids=full_ids,
                embeddings=vectors,
                inputs=inputs,
                support_sha256=support_sha256,
                provenance=execution.CACHE_PROVENANCE_GPU,
                capture_commit=args.expected_execution_commit,
            )
            captured.append(key)
            forwards += len(full_ids)
    run_record["resumed_groups"] = resumed
    run_record["captured_groups"] = captured
    run_record["forwards_executed_this_run"] = forwards

    admission = m11.admit_cache_or_refuse(cache_root, identity_plan)
    groups = m11.load_admitted_groups(cache_root, identity_plan)
    movement = m11.run_movement_on_groups(groups, manifest, audit_record, contract)

    elapsed = time.time() - started
    run_record.update(
        {
            "status": m11.EXECUTED,
            "elapsed_s": elapsed,
            "forwards": int(run_record["forwards_executed_this_run"]),
            "cache_root": str(cache_root),
            "admission": {
                "status": str(admission.get("status")),
                "required_rows": admission.get("required_rows"),
                "available_rows": admission.get("available_rows"),
                "missing_rows": admission.get("missing_rows"),
            },
        }
    )
    movement_sha = write_json(out / "movement.json", movement)
    run_sha = write_json(out / "run.json", run_record)
    print(f"task={m11.TASK_ID} mode=full status={m11.EXECUTED}")
    print(f"movement={movement_sha} run={run_sha} forwards={run_record['forwards']}")
    summaries = movement["summaries"]
    assert isinstance(summaries, dict)
    print(
        f"captured={len(run_record['captured_groups'])} "  # type: ignore[arg-type]
        f"resumed={len(run_record['resumed_groups'])} "  # type: ignore[arg-type]
        f"records={summaries['n_records']} "
        f"computed={summaries['n_computed_endpoints']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
