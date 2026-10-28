"""Sprint 24 Task 7-M12 E2 sample-size and abnormal-period diagnostic CLI.

Public entry point for the cache-only exploratory audit frozen in
``docs/sprint24-e2-cache-diagnostic-specification-a01.json``:

1. verify every pinned digest, the canonical execution freeze, the M10
   prospective-contract canonical content and its live census/plan/custody
   binding BEFORE any statistic;
2. admit the retrieved S24-7-M11 cache through the accepted public admission
   path (24,840 / 24,840 rows, 18 groups, identity-strict, no recapture);
3. run the matched-size E2 audit (A), the baseline-variability decomposition
   (B), the K x seed robustness sweep (C) and the post-hoc diagnostic
   decomposition (D);
4. write the machine-readable tables and a run record that accounts for every
   planned cell or explicit abstention.

No model is loaded, no forward pass runs and no remote or M11 product is
modified. A non-empty ``--out`` is refused.

Usage:

```bash
python experiments/sprint24_task7m12_e2_cache_diagnostic.py \
  --corpus data/generated/sprint24_controlled_v2 \
  --audit artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json \
  --contract artifacts/sprint-24/evidence/s24-7-m10-a02/prospective-contract.json \
  --cache-root artifacts/sprint-24/evidence/s24-7-m12-a01/cache-full \
  --out artifacts/sprint-24/evidence/s24-7-m12-a01/audit
```
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from representation import sprint24_controlled_inference_readiness as readiness  # noqa: E402
from representation import sprint24_e2_cache_diagnostic as audit  # noqa: E402
from representation.sprint24_context_audit import canonical_json  # noqa: E402


def write_json(path: Path, payload: object) -> str:
    text = canonical_json(payload) + "\n"
    path.write_text(text, encoding="utf-8", newline="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_repeats(path: Path, stores: dict[str, dict[tuple, np.ndarray]], repeats: int) -> dict[str, Any]:
    """Persist every repeat value with an explicit key index beside it.

    Arrays keep their own dtype: the matched-size stores are exact integer
    occupancy quanta (``E2 = quanta / size``), so casting them to float would
    discard the exactness the audit relies on.
    """
    keys: dict[str, list[str]] = {}
    arrays: dict[str, np.ndarray] = {}
    for name, store in sorted(stores.items()):
        ordered = sorted(store.items(), key=lambda item: tuple(str(part) for part in item[0]))
        keys[name] = ["|".join(str(part) for part in key) for key, _ in ordered]
        arrays[name] = np.stack([np.asarray(value) for _, value in ordered])
    np.savez_compressed(path, **arrays)
    return {
        "path": path.name,
        "repeats": repeats,
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in sorted(arrays.items())
        },
        "keys": keys,
        "sha256": audit.sha256_file(path),
    }


def source_commit() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--g2-preflight", default=None)
    parser.add_argument("--repeats", type=int, default=audit.REPEATS)
    args = parser.parse_args()

    corpus = Path(args.corpus)
    out = Path(args.out)
    if out.is_dir() and any(out.iterdir()):
        raise SystemExit(f"REFUSED-OVERWRITE: output {out} already holds files")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    payload = audit.run_cache_diagnostic(
        corpus=corpus,
        audit_path=Path(args.audit),
        contract_path=Path(args.contract),
        cache_root=Path(args.cache_root),
        g2_preflight_path=Path(args.g2_preflight) if args.g2_preflight else None,
        repeats=int(args.repeats),
    )

    digests: dict[str, str] = {}
    digests["lineage.json"] = write_json(
        out / "lineage.json",
        {
            "task": payload["task"],
            "spec": payload["spec"],
            "lineage": payload["lineage"],
            "cache": payload["cache"],
            "fit_custody": payload["fit_custody"],
            "limitations": payload["limitations"],
        },
    )
    digests["support.json"] = write_json(out / "support.json", payload["support"])
    digests["a-matched-size.json"] = write_json(
        out / "a-matched-size.json", payload["matched_size"]
    )
    digests["a-contrasts.json"] = write_json(out / "a-contrasts.json", payload["contrasts"])
    digests["b-baseline.json"] = write_json(out / "b-baseline.json", payload["baseline"])
    digests["c-robustness.json"] = write_json(out / "c-robustness.json", payload["robustness"])
    digests["d-diagnostics.json"] = write_json(out / "d-diagnostics.json", payload["diagnostics"])
    digests["accounting.json"] = write_json(out / "accounting.json", payload["accounting"])
    repeats_record = write_repeats(
        out / "repeats.npz", payload["repeats"], int(args.repeats)
    )
    digests["repeats-keys.json"] = write_json(out / "repeats-keys.json", repeats_record)

    accounting = payload["accounting"]
    exposure_check = payload["support"]["retained_exposure"]
    run_record: dict[str, Any] = {
        "task": payload["task"],
        "status": "AUDIT-COMPLETE" if accounting["all_planned_cells_accounted"] else "AUDIT-INCOMPLETE",
        "source_commit": source_commit(),
        "spec_sha256": payload["spec"]["sha256"],
        "mode": "cache-only",
        "forwards_executed_this_run": 0,
        "model_loads": 0,
        "cumulative_forwards_after_run": 77653,
        "cache_root": str(Path(args.cache_root)),
        "cache_admission": payload["cache"]["admission"],
        "cache_file_digest_digest": payload["cache"]["cache_file_digest_digest"],
        "n_cache_files": payload["cache"]["n_cache_files"],
        "n_fits": accounting["n_fits"],
        "retained_exposure_matches_accepted_census": exposure_check["matches_accepted_census"],
        "accounting": {
            "all_planned_cells_accounted": accounting["all_planned_cells_accounted"],
            "planned": accounting["planned"],
            "computed": accounting["computed"],
            "abstained": accounting["abstained"],
            "abstention_reasons": accounting["abstention_reasons"],
        },
        "table_sha256": digests,
        "elapsed_s": time.time() - started,
        "audit_elapsed_s": payload["elapsed_s"],
    }
    digests["run.json"] = write_json(out / "run.json", run_record)

    planned = accounting["planned"]
    computed = accounting["computed"]
    abstained = accounting["abstained"]
    deltas = payload["contrasts"]
    first_seed = str(readiness.FROZEN_ENCODER_SEEDS[0])
    positive = sum(
        1
        for item in deltas
        if (item["per_seed"][first_seed][audit.DESIGN_STRATIFIED]["8"].get("median_quanta") or 0.0)
        > 0.0
    )
    print(f"task={payload['task']} status={run_record['status']} forwards=0")
    print(f"admission={payload['cache']['admission']['status']} rows={payload['cache']['n_rows']}")
    print(
        "cells={cells} active={active} abstained_cells={inactive} comparisons={comparisons} "
        "transitions={transitions} controls={controls} repeats={repeats} fits={fits}".format(
            cells=accounting["cells"],
            active=accounting["active_cells"],
            inactive=accounting["abstained_cells"],
            comparisons=accounting["comparisons"],
            transitions=accounting["transitions"],
            controls=accounting["controls"],
            repeats=accounting["repeats"],
            fits=accounting["n_fits"],
        )
    )
    print(
        "A: full={fa}/{fp} stratified={sa}/{sp} srs={sr}/{srp} | B1={b1a}/{b1p} B2={b2a}/{b2p} | D={da}/{dp}".format(
            fa=computed["A_full_arm_cells"],
            fp=planned["A_full_arm_cells"],
            sa=computed["A_stratified_cells"],
            sp=planned["A_stratified_cells"],
            sr=computed["A_srs_cells"],
            srp=planned["A_srs_cells"],
            b1a=computed["B1_cells"],
            b1p=planned["B1_cells"],
            b2a=computed["B2_cells"],
            b2p=planned["B2_cells"],
            da=computed["D_tv_decompositions"],
            dp=planned["D_tv_decompositions"],
        )
    )
    print(
        "abstained={total} reasons={reasons}".format(
            total=sum(abstained.values()), reasons=accounting["abstention_reasons"]
        )
    )
    print(
        f"retained_exposure_census_match={exposure_check['matches_accepted_census']} "
        f"positive_transition_median_deltas_K8={positive} of {len(deltas)}"
    )
    print("tables=" + ",".join(sorted(digests)))
    return 0 if run_record["status"] == "AUDIT-COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
