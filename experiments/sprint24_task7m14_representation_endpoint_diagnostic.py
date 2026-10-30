"""Sprint 24 Task 7-M14 representation-endpoint diagnostic CLI.

Public entry point for the cache-only exploratory audit frozen in
``docs/sprint24-representation-endpoint-diagnostic-specification-a01.json``:

1. ``--stage census`` reads ONLY the accepted corpus manifest and M9 corpus
   audit and writes the metadata-only support census. It refuses to touch the
   embedding cache.
2. ``--stage audit`` verifies the frozen specification bytes, every pinned
   digest, the canonical execution freeze, the M10 prospective-contract
   canonical content and its live binding BEFORE any statistic; admits the
   retrieved S24-7-M11 cache through the accepted public admission path
   (24,840 / 24,840 rows, 18 groups, identity-strict, no recapture); then runs
   the frozen three-branch design and writes the machine-readable tables.

No model is loaded, no forward pass runs and no remote or accepted product is
modified. A non-empty ``--out`` is refused.

Usage:

```bash
python experiments/sprint24_task7m14_representation_endpoint_diagnostic.py \
  --stage census \
  --corpus data/generated/sprint24_controlled_v2 \
  --audit artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json \
  --out artifacts/sprint-24/evidence/s24-7-m14-a01/census

python experiments/sprint24_task7m14_representation_endpoint_diagnostic.py \
  --stage audit \
  --corpus data/generated/sprint24_controlled_v2 \
  --audit artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json \
  --contract artifacts/sprint-24/evidence/s24-7-m10-a02/prospective-contract.json \
  --cache-root artifacts/sprint-24/evidence/s24-7-m12-a01/cache-full \
  --census artifacts/sprint-24/evidence/s24-7-m14-a01/census/census.json \
  --out artifacts/sprint-24/evidence/s24-7-m14-a01/audit
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

import numpy as np  # noqa: E402

from representation import sprint24_controlled_inference_readiness as readiness  # noqa: E402
from representation import sprint24_representation_endpoint_diagnostic as audit  # noqa: E402
from representation.sprint24_context_audit import canonical_json  # noqa: E402


def write_json(path: Path, payload: object) -> str:
    text = canonical_json(payload) + "\n"
    path.write_text(text, encoding="utf-8", newline="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_repeats(path: Path, stores: dict[str, np.ndarray]) -> dict[str, Any]:
    """Persist every repeat vector with an explicit key index beside it."""
    keys = sorted(stores)
    arrays = {key: np.asarray(stores[key]) for key in keys}
    np.savez_compressed(path, **arrays)
    return {
        "path": path.name,
        "n_records": len(keys),
        "arrays": {
            key: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for key, array in sorted(arrays.items())
        },
        "keys": keys,
        "sha256": audit.sha256_file(path),
    }


def source_commit() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin", ".safetensors")


def _install_checkpoint_audit() -> dict[str, int]:
    """Count checkpoint-like file opens for the rest of the process."""
    counts = {"checkpoint_opens": 0}

    def hook(event: str, args: tuple[object, ...]) -> None:
        if event != "open" or not args:
            return
        try:
            path = str(args[0]).lower()
        except Exception:  # pragma: no cover - defensive, never expected
            return
        if path.endswith(CHECKPOINT_SUFFIXES):
            counts["checkpoint_opens"] += 1

    sys.addaudithook(hook)
    return counts


def _gpu_custody() -> dict[str, object]:
    """Observe whether the GPU was ever initialized (module imports aside)."""
    if "torch" not in sys.modules:
        return {"torch_module_loaded": False, "cuda_initialized": False}
    import torch  # noqa: PLC0415 - already imported transitively by the accepted modules

    return {
        "torch_module_loaded": True,
        "torch_module_role": (
            "transitive import through the accepted readiness/inference modules; "
            "no model, checkpoint, optimizer or GPU context is constructed"
        ),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated())
        if torch.cuda.is_initialized()
        else 0,
    }


def _census_stage(args: argparse.Namespace, out: Path) -> int:
    corpus = Path(args.corpus)
    audit_path = Path(args.audit)
    manifest, _freeze, _preflight = readiness.load_inputs(
        corpus, Path(args.g2_preflight) if args.g2_preflight else None
    )
    manifest_ok, manifest_detail = readiness.verify_pinned_artifact(
        corpus / "corpus-manifest.json", "corpus manifest"
    )
    audit_ok, audit_detail = readiness.verify_pinned_artifact(audit_path, "corpus audit")
    if not manifest_ok:
        raise SystemExit(f"REFUSED-DIGEST-DRIFT: {manifest_detail}")
    if not audit_ok:
        raise SystemExit(f"REFUSED-DIGEST-DRIFT: {audit_detail}")
    audit_record = json.loads(audit_path.read_text(encoding="utf-8"))
    census = audit.build_census(manifest, audit_record)
    output = {
        "task": audit.TASK_ID,
        "stage": "census",
        "metadata_only": True,
        "embeddings_read": False,
        "outcome_computed": False,
        "manifest_sha256": audit.sha256_file(corpus / "corpus-manifest.json"),
        "audit_sha256": audit.sha256_file(audit_path),
        "census": census,
    }
    path = out / "census.json"
    digest = write_json(path, output)
    print(
        "task={task} stage=census metadata_only=true outcomes=0 "
        "a_units={a} a_status={astatus} c_units={c} c_status={cstatus}".format(
            task=audit.TASK_ID,
            a=census["branch_a"]["n_units"],
            astatus=census["branch_a"]["unit_status_counts"],
            c=census["branch_c"]["n_units"],
            cstatus=census["branch_c"]["unit_status_counts"],
        )
    )
    print(f"census={path} sha256={digest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("census", "audit"), default="audit")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--contract", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--census", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--g2-preflight", default=None)
    parser.add_argument("--repeats-ab", type=int, default=audit.REPEATS_AB)
    parser.add_argument("--repeats-c", type=int, default=audit.REPEATS_C)
    args = parser.parse_args()

    out = Path(args.out)
    if out.is_dir() and any(out.iterdir()):
        raise SystemExit(f"REFUSED-OVERWRITE: output {out} already holds files")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    audit_counts = _install_checkpoint_audit()
    if args.stage == "census":
        return _census_stage(args, out)

    if not args.contract or not args.cache_root:
        raise SystemExit("REFUSED-ARGUMENTS: audit stage needs --contract and --cache-root")
    cache_root = Path(args.cache_root)
    cache_digests_before = {
        path.name: audit.sha256_file(path)
        for path in sorted(cache_root.iterdir())
        if path.is_file()
    }
    payload = audit.run_diagnostic(
        corpus=Path(args.corpus),
        audit_path=Path(args.audit),
        contract_path=Path(args.contract),
        cache_root=cache_root,
        census_path=Path(args.census) if args.census else None,
        g2_preflight_path=Path(args.g2_preflight) if args.g2_preflight else None,
        repeats_ab=int(args.repeats_ab),
        repeats_c=int(args.repeats_c),
    )
    products_after = audit.verify_accepted_products()
    cache_digests_after = {
        path.name: audit.sha256_file(path)
        for path in sorted(cache_root.iterdir())
        if path.is_file()
    }
    if cache_digests_before != cache_digests_after:
        raise SystemExit("REFUSED-CACHE-DRIFT: cache bytes changed during the audit")

    digests: dict[str, str] = {}
    digests["lineage.json"] = write_json(
        out / "lineage.json",
        {
            "task": payload["task"],
            "spec": payload["spec"],
            "amendments": payload["amendments"],
            "census_check": payload["census_check"],
            "lineage": payload["lineage"],
            "prior_products": payload["prior_products"],
            "cache": payload["cache"],
            "fit_custody": payload["fit_custody"],
            "limitations": payload["limitations"],
        },
    )
    digests["census.json"] = write_json(out / "census.json", payload["census"])
    digests["support.json"] = write_json(out / "support.json", payload["support"])
    digests["branch-a.json"] = write_json(
        out / "branch-a.json",
        {
            "task": payload["task"],
            "branch": "A",
            "note": (
                "one robot|program stratum at a time; disjoint equal-size abnormal and "
                "normal arms plus a second disjoint normal arm as the matched null; labels "
                "construct and diagnose only and never enter a fit"
            ),
            "units": payload["units_a"],
        },
    )
    digests["branch-a-period-matched.json"] = write_json(
        out / "branch-a-period-matched.json",
        {
            "task": payload["task"],
            "branch": "A",
            "variant": "PERIOD-MATCHED",
            "note": (
                "the same comparison with the per-period arm composition held exactly to "
                "the declared quota; reported only where the census supports it"
            ),
            "units": payload["units_a_matched"],
        },
    )
    digests["branch-b.json"] = write_json(
        out / "branch-b.json",
        {
            "task": payload["task"],
            "branch": "B",
            "note": (
                "endpoint matrix on the same valid arms: accepted E1, exact-quanta E2 for "
                "K in {4,8,16}, and the bounded within-prototype residual diagnostic D "
                "(location, spread, radial and tangential) with matched null bands"
            ),
            "endpoint_matrix": payload["endpoint_matrix"],
        },
    )
    digests["branch-c.json"] = write_json(
        out / "branch-c.json",
        {
            "task": payload["task"],
            "branch": "C",
            "note": (
                "reference-derived perturbation of known magnitude on disjoint normal "
                "arms; positive controls verify measurement responsiveness only"
            ),
            "records": payload["controls"],
        },
    )
    digests["aggregates.json"] = write_json(
        out / "aggregates.json",
        {
            "task": payload["task"],
            "pooled": payload["aggregates"]["pooled"],
            "period_matched": payload["aggregates"]["period_matched"],
            "branch_c": payload["aggregates"]["branch_c"],
        },
    )
    digests["arms.json"] = write_json(
        out / "arms.json",
        {
            "task": payload["task"],
            "note": (
                "repeat-0 materialized arm identities per unit and candidate pools; "
                "evidence only, no statistic depends on this table"
            ),
            "units": payload["arms"],
            "reuse": payload["reuse"],
        },
    )
    digests["accounting.json"] = write_json(
        out / "accounting.json", payload["accounting"]
    )
    repeats_record = write_repeats(out / "repeats.npz", payload["repeats"])
    digests["repeats-keys.json"] = write_json(out / "repeats-keys.json", repeats_record)
    controls_record = write_repeats(out / "controls.npz", payload["control_repeats"])
    digests["controls-keys.json"] = write_json(out / "controls-keys.json", controls_record)

    accounting = payload["accounting"]
    branch_a = accounting["branch_a"]
    branch_c = accounting["branch_c"]
    run_record: dict[str, Any] = {
        "task": payload["task"],
        "status": payload["status"],
        "source_commit": source_commit(),
        "spec_sha256": payload["spec"]["sha256"],
        "mode": "cache-only",
        "forwards_executed_this_run": 0,
        "model_loads": 0,
        "checkpoint_opens": audit_counts["checkpoint_opens"],
        **_gpu_custody(),
        "cumulative_forwards_after_run": audit.CUMULATIVE_FORWARDS,
        "cache_root": str(cache_root),
        "cache_admission": payload["cache"]["admission"],
        "cache_file_digest_digest": payload["cache"]["cache_file_digest_digest"],
        "n_cache_files": payload["cache"]["n_cache_files"],
        "cache_unchanged": cache_digests_before == cache_digests_after,
        "prior_products_unchanged_before_run": all(
            item["match"] for item in payload["prior_products"].values()
        ),
        "prior_products_unchanged_after_run": all(
            item["match"] for item in products_after.values()
        ),
        "census_check": payload["census_check"],
        "n_fits": accounting["n_fits"],
        "accounting": {
            "all_planned_cells_accounted": accounting["all_planned_cells_accounted"],
            "branch_a": branch_a,
            "branch_a_period_matched": accounting["branch_a_period_matched"],
            "branch_c": branch_c,
        },
        "table_sha256": digests,
        "elapsed_s": time.time() - started,
        "audit_elapsed_s": payload["elapsed_s"],
    }
    digests["run.json"] = write_json(out / "run.json", run_record)

    print(
        f"task={payload['task']} status={run_record['status']} forwards=0 "
        f"checkpoint_opens={run_record['checkpoint_opens']} "
        f"cuda_initialized={run_record['cuda_initialized']}"
    )
    print(
        "admission={status} rows={rows} cache={cache}".format(
            status=payload["cache"]["admission"]["status"],
            rows=payload["cache"]["n_rows"],
            cache=payload["cache"]["cache_file_digest_digest"][:12],
        )
    )
    print(
        "branch_a units={au} computed={ac} abstained={aa} estimator_cells={ae}/{ap}".format(
            au=branch_a["planned_units"],
            ac=branch_a["computed_units"],
            aa=branch_a["abstained_units"],
            ae=branch_a["computed_estimator_cells"],
            ap=branch_a["planned_estimator_cells"],
        )
    )
    print(
        "branch_a_period_matched computed={mc}/{mp} estimator_cells={me}/{mpc}".format(
            mc=accounting["branch_a_period_matched"]["computed_units"],
            mp=accounting["branch_a_period_matched"]["planned_units"],
            me=accounting["branch_a_period_matched"]["computed_estimator_cells"],
            mpc=accounting["branch_a_period_matched"]["planned_estimator_cells"],
        )
    )
    print(
        "branch_c units={cu} records={cr}/{cp} estimator_cells={ce}/{cep}".format(
            cu=branch_c["planned_units"],
            cr=branch_c["computed_records"],
            cp=branch_c["planned_records"],
            ce=branch_c["computed_estimator_cells"],
            cep=branch_c["planned_estimator_cells"],
        )
    )
    print(
        "cache_unchanged={cache} prior_products_unchanged={prior}".format(
            cache=run_record["cache_unchanged"],
            prior=run_record["prior_products_unchanged_after_run"],
        )
    )
    print("tables=" + ",".join(sorted(digests)))
    return 0 if run_record["status"] == "AUDIT-COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
