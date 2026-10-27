"""Sprint 24 Task 7-M10 controlled-corpus inference readiness CLI (metadata only).

Public preflight entry point: reads the official M9 corpus metadata and the
accepted G2 preflight custody, validates the freeze's canonical content and
custody against INTERNAL pinned accepted values (no caller digest override can
confer authority), computes the metadata-only eligibility census, derives the
executable per-group cache-identity plan from the accepted admission
machinery, runs the controlled-corpus synthetic-fixture adapter
(write/inspect/admit/measure on real pair structure with synthetic vectors),
and freezes the versioned prospective amendment contract.  Reads NPZ NPY
headers only (shape bytes) and cache sidecars/file IDs only; no waveform
signal value, model, checkpoint, embedding value or score is read; zero
forwards execute.

Usage:

```bash
python experiments/sprint24_task7m10_controlled_inference_readiness.py \
  --corpus data/generated/sprint24_controlled_v2 \
  --audit artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json \
  --out artifacts/sprint-24/evidence/s24-7-m10-a01
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from representation import sprint24_controlled_inference_readiness as readiness  # noqa: E402
from representation.sprint24_context_audit import canonical_json  # noqa: E402


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def write_json(path: Path, payload: object) -> str:
    text = canonical_json(payload) + "\n"
    path.write_text(text, encoding="utf-8", newline="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--g2-preflight", default=None)
    parser.add_argument("--cache-root", action="append", default=[])
    args = parser.parse_args()

    corpus = Path(args.corpus)
    audit_path = Path(args.audit)
    out = Path(args.out)
    if out.is_dir() and any(out.iterdir()):
        raise SystemExit(f"REFUSED-OVERWRITE: output {out} already holds files")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if not audit_path.is_file():
        raise SystemExit(f"REFUSED-MISSING-CORPUS: observed audit absent: {audit_path}")

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
    preflight_path = Path(args.g2_preflight) if args.g2_preflight else (
        REPO / "artifacts" / "sprint-24" / "evidence" / "s24-7-m4-a01" / "g2-accepted" / "preflight.json"
    )
    preflight_ok, preflight_detail = readiness.verify_pinned_artifact(
        preflight_path, "G2 preflight", allowed={REPO / "artifacts" / "sprint-24" / "evidence" / "s24-7-m4-a01" / "g2-accepted" / "preflight.json"}
    )
    if not preflight_ok:
        raise SystemExit(f"REFUSED-DIGEST-DRIFT: {preflight_detail}")

    custody = readiness.validate_custody_inputs(preflight)

    audit_record: dict[str, Any] = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit_record, dict):
        raise SystemExit("REFUSED-IDENTITY: corpus audit is not a mapping")

    frozen = readiness.prospective_contract(manifest, audit_record, preflight)
    contract = frozen["contract"]
    assert isinstance(contract, dict)

    declared_probe = readiness.probe_declared_cache_roots([Path(root) for root in args.cache_root])
    executable_path = contract.get("executable_path") if isinstance(contract, dict) else {}
    assert isinstance(executable_path, dict)
    identity_plan = executable_path.get("identity_plan")
    assert isinstance(identity_plan, dict)
    plan_groups = identity_plan.get("groups")
    assert isinstance(plan_groups, dict) and len(plan_groups) == int(executable_path["n_identity_groups"])
    plan_probe = readiness.compare_probe_to_plan(declared_probe, identity_plan)
    if str(declared_probe.get("verdict")) in ("INVALID", "PARTIAL"):
        raise SystemExit(f"REFUSED-IDENTITY: declared cache roots are {declared_probe.get('verdict')}: {plan_probe.get('detail')}")

    elapsed = time.time() - started
    run_record: dict[str, Any] = {
        "task": "S24-7-M10",
        "readiness_id": readiness.READINESS_ID,
        "amendment_id": readiness.AMENDMENT_ID,
        "source_commit": source_commit(),
        "corpus": str(corpus),
        "audit": str(audit_path),
        "freeze_check": freeze_detail,
        "manifest_check": manifest_detail,
        "audit_check": audit_detail,
        "preflight_check": preflight_detail,
        "preflight_path": str(preflight_path),
        "custody_validated": sorted(custody.keys()),
        "contract_sha256": str(frozen["contract_sha256"]),
        "binding_digest": str(contract.get("binding_digest")),
        "n_pairs": int(contract["census"]["n_pairs"]),  # type: ignore[index]
        "eligibility_counts": {
            state: sum(1 for p in contract["census"]["pairs"] if p["eligibility"] == state)  # type: ignore[union-attr]
            for state in readiness.ELIGIBILITY_STATES
        },
        "n_identity_groups": int(contract["executable_path"]["n_identity_groups"]),  # type: ignore[index]
        "group_ids": list(contract["executable_path"]["group_ids"]),  # type: ignore[index]
        "identity_plan_sha256": str(contract["executable_path"]["identity_plan_sha256"]),  # type: ignore[index]
        "identity_plan_groups": len(plan_groups),
        "adapter_checks": bool(contract["executable_path"]["controlled_adapter"]["checks"]),  # type: ignore[index]
        "adapter_binding_ok": bool(contract["executable_path"]["controlled_adapter"]["binding_ok"]),  # type: ignore[index]
        "adapter_binding_verified_pre_execution": bool(contract["executable_path"]["controlled_adapter"]["binding_verified_pre_execution"]),  # type: ignore[index]
        "adapter_study_admission": dict(contract["executable_path"]["controlled_adapter"]["study_admission"]),  # type: ignore[index]
        "declared_cache_roots": [str(root) for root in args.cache_root],
        "declared_cache_probe": declared_probe,
        "cache_plan_comparison": plan_probe,
        "forwards": 0,
        "model_loads": 0,
        "embedding_reads": 0,
        "embedding_values_opened": 0,
        "embedding_bytes_read": 0,
        "waveform_signal_value_reads": 0,
        "npz_npy_header_reads": int(contract["reads"]["npz_npy_header_reads"]),  # type: ignore[index]
        "elapsed_s": elapsed,
    }

    census_sha = write_json(out / "eligibility-census.json", contract["census"])
    expected_sha = write_json(out / "expected-retained-census.json", contract["expected_census"])
    contract_sha = write_json(out / "prospective-contract.json", frozen)
    run_sha = write_json(out / "run.json", run_record)
    print(f"readiness={readiness.READINESS_ID} amendment={readiness.AMENDMENT_ID}")
    print(f"contract_sha256={frozen['contract_sha256']}")
    print(f"census={census_sha} expected={expected_sha} contract_file={contract_sha} run={run_sha}")
    print("forwards=0 model_loads=0 embedding_reads=0 embedding_values_opened=0 waveform_signal_value_reads=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
