"""Focused behavioral checks for S24-7-M11 frozen inference execution.

Covers the fail-closed contract gate, identity-bound cache admission and
resume, the frozen movement path on synthetic-but-admitted vectors, and
the real-CLI refusal boundaries. No checkpoint is loaded, no forward
executes, no waveform signal is read in these checks; the synthetic
vectors below prove the execution plumbing only, never corpus evidence.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from representation import sprint24_controlled_inference_execution as m11
from representation import sprint24_controlled_inference_readiness as readiness
from representation import sprint24_weekly_monthly_execution as execution

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "data" / "generated" / "sprint24_controlled_v2"
AUDIT = REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m9-a01" / "corpus-audit.json"
G2_PREFLIGHT = (
    REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m4-a01"
    / "g2-accepted" / "preflight.json"
)
CONTRACT_FILE = (
    REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m10-a02"
    / "prospective-contract.json"
)


def _inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads((CORPUS / "corpus-manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    manifest["__manifest_path__"] = str(CORPUS / "corpus-manifest.json")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert isinstance(audit, dict)
    preflight = json.loads(G2_PREFLIGHT.read_text(encoding="utf-8"))
    assert isinstance(preflight, dict)
    return manifest, audit, preflight


def _contract() -> tuple[dict[str, Any], str]:
    envelope = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    assert isinstance(envelope, dict)
    return dict(envelope["contract"]), str(envelope["contract_sha256"])


def test_m11_contract_digest_pinned() -> None:
    assert m11.EXPECTED_CONTRACT_SHA256 == "f4f92a60ee1782ccc0ec75587e9df5d1471e4045e201c43478893d294b5757f7"
    assert m11.TASK_ID == "S24-7-M11"
    contract, digest = _contract()
    assert digest == m11.EXPECTED_CONTRACT_SHA256


def test_m11_contract_binding_verified_pre_execution() -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, digest = _contract()
    bound = m11.verify_contract_binding(contract, digest, manifest, audit, custody)
    assert set(bound) == {"census", "plan"}
    assert int(bound["plan"]["n_groups"]) == 18  # type: ignore[index]


def test_m11_altered_contract_refused_pre_execution() -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, digest = _contract()
    tampered = dict(contract)
    tampered["binding_digest"] = "0" * 64
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        m11.verify_contract_binding(tampered, digest, manifest, audit, custody)


def test_m11_stale_digest_refused_pre_execution() -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, _digest = _contract()
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        m11.verify_contract_binding(contract, "0" * 64, manifest, audit, custody)


def test_m11_capture_plan_matches_frozen_groups() -> None:
    manifest, _audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, digest = _contract()
    bound = m11.verify_contract_binding(
        contract, digest, manifest, _audit, custody
    )
    capture = m11.bound_capture_plan(manifest, custody)
    live_groups = capture["plan"]["groups"]
    assert isinstance(live_groups, dict)
    frozen_groups = contract["executable_path"]["identity_plan"]["groups"]
    assert dict(live_groups) == dict(frozen_groups)
    assert len(capture["file_ids"]) == 6
    assert all(len(ids) == 1380 for ids in capture["file_ids"].values())


def test_m11_reference_pool_is_causal_and_sized() -> None:
    manifest, _audit, _preflight = _inputs()
    for history_id in sorted(m11.bound_capture_plan(manifest, readiness.validate_custody_inputs(_preflight))["file_ids"]):
        ref_ids = m11.reference_ids_for_history(manifest, history_id)
        assert len(ref_ids) == 220
    manifest_files = {str(e["file_id"]): e for e in manifest["files"] if isinstance(e, dict)}
    sample = m11.reference_ids_for_history(manifest, sorted(manifest_files)[0] and "S24C-ITER-0001-CONTROLLED-01")
    ends = [float(manifest_files[fid]["end_time_rel_s"]) for fid in sample]
    assert max(ends) <= 0.0


def _write_synthetic_group(
    root: Path,
    inputs: dict[str, Any],
    plan_groups: dict[str, Any],
    history_id: str,
    model_seed: int,
    *,
    provenance: str,
    commit: str | None,
    rng: Any,
) -> None:
    key = f"{history_id}|{model_seed}"
    spec = plan_groups[key]
    assert isinstance(spec, dict)
    support = str(spec["support_sha256"])
    n = int(spec["required_rows"])
    vectors = rng.normal(0.0, 1.0, size=(n, 128)).astype(np.float32)
    file_ids = [f"SYN-{history_id}-{model_seed}-{i:05d}" for i in range(n)]
    # Writer binds its own support digest; the caller re-derives identity via
    # the inspected sidecar, never by trusting this call's arguments.
    m11.write_group_cache(
        root,
        history_id=history_id,
        model_seed=int(model_seed),
        file_ids=file_ids,
        embeddings=vectors,
        inputs=inputs,
        support_sha256=support,
        provenance=provenance,
        capture_commit=commit,
    )


def test_m11_admission_refuses_foreign_provenance() -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, digest = _contract()
    bound = m11.verify_contract_binding(contract, digest, manifest, audit, custody)
    plan = bound["plan"]
    assert isinstance(plan, dict)
    inputs = m11.capture_inputs(custody)
    rng = np.random.default_rng(2407)
    with tempfile.TemporaryDirectory(prefix="s24-m11-neg-") as tmp:
        root = Path(tmp)
        first_history = sorted(m11.bound_capture_plan(manifest, custody)["file_ids"])[0]
        for seed in readiness.FROZEN_ENCODER_SEEDS:
            _write_synthetic_group(
                root, inputs, plan["groups"], first_history, int(seed),
                provenance=execution.CACHE_PROVENANCE_FIXTURE,
                commit=None, rng=rng,
            )
        m11.set_support_universe(m11.bound_capture_plan(manifest, custody)["file_ids"])
        with pytest.raises(execution.CachePreflightFailure):
            m11.admit_cache_or_refuse(root, plan)


def test_m11_resume_skips_complete_group_without_reforward() -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, digest = _contract()
    bound = m11.verify_contract_binding(contract, digest, manifest, audit, custody)
    plan = bound["plan"]
    assert isinstance(plan, dict)
    wanted = m11.bound_capture_plan(manifest, custody)
    m11.set_support_universe(wanted["file_ids"])
    first_history = sorted(wanted["file_ids"])[0]
    key = f"{first_history}|{readiness.FROZEN_ENCODER_SEEDS[0]}"
    expected = str(plan["groups"][key]["cache_identity"])  # type: ignore[index]
    with tempfile.TemporaryDirectory(prefix="s24-m11-resume-") as tmp:
        root = Path(tmp)
        sidecar = {
            "schema": execution.CACHE_SIDECAR_SCHEMA,
            "kind": execution.CACHE_SIDECAR_KIND,
            "history_id": first_history,
            "model_seed": int(readiness.FROZEN_ENCODER_SEEDS[0]),
            "device": "cuda",
            "checkpoint_sha256": str(custody["checkpoints"][str(readiness.FROZEN_ENCODER_SEEDS[0])]),  # type: ignore[index]
            "bank_sha256": str(custody["banks"][str(readiness.FROZEN_ENCODER_SEEDS[0])]),  # type: ignore[index]
            "support_sha256": str(plan["groups"][key]["support_sha256"]),  # type: ignore[index]
            "preprocessing_sha256": str(custody["preprocessing_sha256"]),
            "arm_config_sha256": str(custody["arms"][str(readiness.FROZEN_ENCODER_SEEDS[0])]),  # type: ignore[index]
            "data_binding_sha256": str(custody["binding_sha256"]),
            "scale_state_sha256": "none",
            "cache_identity": expected,
            "n_rows": 4,
            "capture_provenance": execution.CACHE_PROVENANCE_SMOKE,
            "capture_commit": None,
        }
        stem = f"{first_history}__seed-{readiness.FROZEN_ENCODER_SEEDS[0]}"
        (root / f"{stem}.json").write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
        vectors = np.zeros((4, 128), dtype=np.float32)
        np.savez(root / f"{stem}.npz", file_ids=np.array(["a", "b", "c", "d"]), embeddings=vectors)
        hit = m11.cached_group_complete(root, first_history, int(readiness.FROZEN_ENCODER_SEEDS[0]), expected)
        assert hit is not None and int(hit["n_rows"]) == 4
        assert m11.cached_group_complete(root, first_history, int(readiness.FROZEN_ENCODER_SEEDS[0]), "0" * 64) is None


def test_m11_cli_smoke_refuses_without_bundle_roots() -> None:
    with tempfile.TemporaryDirectory(prefix="s24-m11-cli-") as tmp:
        out = Path(tmp) / "out"
        cache = Path(tmp) / "cache"
        cache.mkdir()
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "experiments" / "sprint24_task7m11_controlled_inference_execution.py"),
                "--corpus", str(CORPUS),
                "--audit", str(AUDIT),
                "--contract", str(CONTRACT_FILE),
                "--cache-root", str(cache),
                "--out", str(out),
                "--mode", "smoke",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 2, proc.stderr[-2000:]
        run = json.loads((out / "run.json").read_text(encoding="utf-8"))
        assert run["status"] == "REFUSED-PREREQUISITE"
        assert run["forwards"] == 0
