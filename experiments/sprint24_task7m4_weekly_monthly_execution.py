"""Sprint 24 Task 7-M4 weekly/monthly execution admission CLI (zero new forwards).

This entry point performs the frozen study's release preflight and then stops:
it verifies the protocol, study, manifest and accepted-capture digests, compiles
the exact per-file embedding row plan, re-derives the accepted G2 capture
identity, inspects declared embedding-cache stores, and either admits execution
(exit 0) or writes a fail-closed blocker record (exit 3).

It computes no movement value, no contrast, no uncertainty interval and no
disposition. On the accepted data lineage no reusable embedding cache exists, so
the run is expected to refuse admission; the refusal *is* the recorded result and
must not be read as a scientific outcome.

Example::

    python experiments/sprint24_task7m4_weekly_monthly_execution.py \\
      --data-root F:/ai-ml/_s24m1-local-metadata \\
      --edges experiments/sprint24_task7m1_dev-edges.json \\
      --expected-manifest-sha256 <history>=<sha256> ... \\
      --g2-accepted-root artifacts/sprint-24/evidence/s24-7-m4-a01/g2-accepted \\
      --cache-root <declared cache store> ... \\
      --out artifacts/sprint-24/evidence/s24-7-m4-a01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from representation import sprint24_weekly_monthly_execution as execution  # noqa: E402
from representation import sprint24_weekly_monthly_study as study  # noqa: E402
from representation.sprint24_context_audit import canonical_json  # noqa: E402

#: Frozen study digest (Task 7-M3 checkpoint ``0a17ffa7``).
DEFAULT_STUDY_SHA256 = "6f96c2750c15552bcdf25678a51ed7b24a1f7aecdb5914bf5985d3012d1daefd"

#: Frozen protocol document digest (Task 7-M3 checkpoint ``0a17ffa7``).
DEFAULT_PROTOCOL_SHA256 = "8da4afda89dd303099843eecda546cdc337696525ece0eed59f400d1031d8b63"

#: Frozen label-blind manifest digests recorded by the accepted M3 evidence.
DEFAULT_MANIFEST_DIGESTS = {
    "references_sha256": "1057570f6723b75644b7c3207882c99970f15bd4c348ef95d94ca59cbe3da73c",
    "periods_sha256": "e86157b056d292081a1f0007a7e8ae57ce121448fd817c56d7a2dbabde9c2f3d",
    "pairs_sha256": "6b848cf6972f25a03078df94a7ae64518801db2272d0d6cac67626304726cf0e",
}

#: Exit code for a refused cache preflight (a recorded blocker, not an error).
EXIT_BLOCKED = 3


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


def parse_expected(values: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected argument needs <key>=<sha256>: {value!r}")
        key, digest = value.split("=", 1)
        expected[key.strip()] = digest.strip()
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--g2-accepted-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--cache-root",
        action="append",
        default=[],
        help="repeatable declared embedding-cache store to inspect",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        action="append",
        default=[],
        help="repeatable <history>=<sha256>; every audited history must be bound",
    )
    parser.add_argument(
        "--expected-manifest-digest",
        action="append",
        default=[],
        help="repeatable <key>=<sha256> for references_sha256/periods_sha256/pairs_sha256",
    )
    parser.add_argument("--expected-study-sha256", default=DEFAULT_STUDY_SHA256)
    parser.add_argument("--expected-protocol-sha256", default=DEFAULT_PROTOCOL_SHA256)
    parser.add_argument("--protocol-doc", default=study.PROTOCOL_DOC)
    args = parser.parse_args()

    started = time.time()
    data_root = Path(args.data_root)
    edges_path = Path(args.edges)
    g2_root = Path(args.g2_accepted_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    edges_raw = json.loads(edges_path.read_text(encoding="utf-8"))
    if not isinstance(edges_raw, dict):
        raise SystemExit("edges file must map history id to its 11 decile edges")
    edges = {str(key): [float(value) for value in values] for key, values in edges_raw.items()}
    expected_manifests = parse_expected(args.expected_manifest_sha256)
    if set(expected_manifests) != set(edges):
        raise SystemExit(
            "expected manifest digests must cover exactly the audited histories: "
            f"{sorted(set(edges) ^ set(expected_manifests))}"
        )
    expected_digests = dict(DEFAULT_MANIFEST_DIGESTS)
    explicit_digests = parse_expected(args.expected_manifest_digest)
    unknown = sorted(set(explicit_digests) - set(expected_digests))
    if unknown:
        raise SystemExit(f"unknown manifest digest keys: {unknown}")
    expected_digests.update(explicit_digests)

    # --- frozen identity verification ---------------------------------------
    freeze = study.study_freeze()
    if freeze["study_sha256"] != args.expected_study_sha256:
        raise SystemExit(
            f"frozen study digest mismatch: {freeze['study_sha256']} != "
            f"{args.expected_study_sha256}"
        )
    protocol_path = REPO / args.protocol_doc
    if not protocol_path.is_file():
        raise SystemExit(f"frozen protocol document is missing: {protocol_path}")
    protocol_sha256 = sha256_path(protocol_path)
    if protocol_sha256 != args.expected_protocol_sha256:
        raise SystemExit(
            f"frozen protocol digest mismatch: {protocol_sha256} != "
            f"{args.expected_protocol_sha256}"
        )

    manifests: dict[str, dict[str, object]] = {}
    manifest_records: dict[str, dict[str, object]] = {}
    for history_id in sorted(edges):
        path = data_root / "DEVELOPMENT" / history_id / "manifest.json"
        if not path.is_file():
            raise SystemExit(f"missing accepted manifest: {path}")
        digest = sha256_path(path)
        if digest != expected_manifests[history_id]:
            raise SystemExit(
                f"accepted manifest digest mismatch for {history_id}: "
                f"{digest} != {expected_manifests[history_id]}"
            )
        manifests[history_id] = json.loads(path.read_text(encoding="utf-8"))
        manifest_records[history_id] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }

    # --- frozen label-blind manifests (metadata only) -----------------------
    primary = study.primary_calendar()
    result = study.evaluate_source(manifests, edges, primary, modes=study.MODES)
    study.assert_study_construction_label_blind()
    label_blind = study.build_label_blind_manifest(result, primary)
    posthoc = study.build_posthoc_burden(result)
    metadata_coverage = study.coverage_matrix(label_blind, posthoc)
    observed_digests = dict(label_blind["digest"])
    assert isinstance(observed_digests, dict)
    digest_mismatches = {
        key: {"expected": expected_digests[key], "observed": str(observed_digests[key])}
        for key in sorted(expected_digests)
        if str(observed_digests.get(key)) != expected_digests[key]
    }
    if digest_mismatches:
        raise SystemExit(f"frozen label-blind manifest digest mismatch: {digest_mismatches}")

    # --- accepted G2 capture identity ---------------------------------------
    g2_record = execution.read_accepted_g2(g2_root)
    inputs = execution.identity_inputs(g2_record)
    row_plan = execution.required_row_plan(manifests, edges)
    totals = row_plan["totals"]
    assert isinstance(totals, dict)
    if not all(
        bool(totals[key])
        for key in ("matches_declared_envelope", "matches_declared_span", "matches_declared_reference")
    ):
        raise SystemExit(f"required row plan disagrees with the frozen envelope: {totals}")
    for history_id, expected_reference in execution.DECLARED_REFERENCE_ROWS.items():
        block = row_plan["per_history"][history_id]  # type: ignore[index]
        if int(block["n_reference_rows"]) != expected_reference:
            raise SystemExit(f"reference row count changed for {history_id}")
    capture = execution.g2_capture_union(row_plan, g2_record["manifests"])  # type: ignore[arg-type]
    required_identities = execution.required_cache_identities(row_plan, inputs)
    accepted_identities = execution.accepted_capture_identities(
        g2_record, row_plan, inputs, capture
    )
    if not accepted_identities["all_recorded_identities_reproduced"]:
        raise SystemExit(
            "accepted G2 cache identity could not be reproduced from its recorded inputs"
        )

    comparisons: dict[str, dict[str, object]] = {}
    required_groups = required_identities["groups"]
    accepted_groups = accepted_identities["groups"]
    assert isinstance(required_groups, dict) and isinstance(accepted_groups, dict)
    for key, required in sorted(required_groups.items()):
        accepted = accepted_groups.get(key)
        capture_block = capture["per_history"][required["history_id"]]  # type: ignore[index]
        comparisons[key] = {
            "history_id": required["history_id"],
            "model_seed": required["model_seed"],
            "required_rows": required["required_rows"],
            "required_support_sha256": required["support_sha256"],
            "required_cache_identity": required["cache_identity"],
            "accepted_identity": None if accepted is None else accepted["recorded_identity"],
            "accepted_identity_equals_required": bool(
                accepted is not None
                and accepted["recorded_identity"] == required["cache_identity"]
            ),
            "accepted_captured_rows": int(capture_block["n_captured"]),
            "accepted_uncaptured_study_rows": int(capture_block["n_uncaptured_study_rows"]),
        }

    # --- declared cache stores and admission --------------------------------
    inventory = execution.inspect_cache_stores([Path(root) for root in args.cache_root])
    coverage = execution.cache_coverage(row_plan, required_identities, inventory)
    blocked_card: dict[str, object] | None = None
    try:
        admission = execution.admit_execution(coverage, inventory)
    except execution.CachePreflightFailure as failure:
        admission = dict(failure.report)
        blocked_card = {
            "status": "REFUSED",
            "message": str(failure),
            "blockers": list(admission["blockers"]),
            "required_rows": admission["required_rows"],
            "available_rows": admission["available_rows"],
            "missing_rows": admission["missing_rows"],
            "not_a_scientific_result": (
                "a failed cache preflight is a data-availability blocker; it is not a "
                "PROMISING / NOT-PROMISING / INCONCLUSIVE disposition"
            ),
        }
    finished = time.time()

    ledger = execution.execution_ledger(
        metadata_coverage,
        admission,
        "no movement value exists: the cache preflight failed closed before outcomes"
        if not admission["complete"]
        else "admitted; endpoint computation is not part of this attempt",
    )
    cost = execution.cost_record(row_plan, coverage)
    custody = execution.custody_record(
        {
            "protocol_doc": protocol_path,
            "edges": edges_path,
            **{
                f"manifests/{history_id}/manifest.json": data_root
                / "DEVELOPMENT"
                / history_id
                / "manifest.json"
                for history_id in sorted(manifests)
            },
        },
        g2_record,
        inventory,
    )

    preflight = {
        "execution_id": execution.EXECUTION_ID,
        "study_id": study.STUDY_ID,
        "study_sha256": freeze["study_sha256"],
        "protocol_document": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
            "bytes": protocol_path.stat().st_size,
        },
        "mode": "cache-admission-preflight",
        "verified": {
            "study_digest_matches": True,
            "protocol_digest_matches": True,
            "label_blind_manifest_digests_match": True,
            "accepted_manifest_digests_match": True,
            "accepted_g2_artifact_digests_match": True,
            "accepted_g2_identity_reproduced": True,
            "required_row_plan_matches_envelope": True,
        },
        "observed_manifest_digests": observed_digests,
        "expected_manifest_digests": expected_digests,
        "row_plan_totals": totals,
        "accepted_capture": capture,
        "accepted_identities": accepted_identities,
        "required_vs_accepted_identities": comparisons,
        "cache_inventory_summary": {
            "roots": inventory["roots"],
            "npz_count": inventory["npz_count"],
            "valid_entries": inventory["valid_entries"],
            "invalid_entries": inventory["invalid_entries"],
            "total_valid_rows": inventory["total_valid_rows"],
        },
        "coverage": {
            "groups_complete": coverage["groups_complete"],
            "n_groups": coverage["n_groups"],
            "total_required_rows": coverage["total_required_rows"],
            "total_available_rows": coverage["total_available_rows"],
            "total_missing_rows": coverage["total_missing_rows"],
            "coverage_fraction": coverage["coverage_fraction"],
        },
        "admission": admission,
        "no_movement_computed": True,
        "no_disposition_computed": True,
        "forwards": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "checkpoint_loads": 0,
    }

    output_digests: dict[str, str] = {}
    output_digests["preflight.json"] = write_json(out / "preflight.json", preflight)
    output_digests["row-plan.json"] = write_json(
        out / "row-plan.json",
        {
            "study_id": study.STUDY_ID,
            "study_sha256": freeze["study_sha256"],
            "execution_id": execution.EXECUTION_ID,
            **row_plan,
        },
    )
    output_digests["required-identities.json"] = write_json(
        out / "required-identities.json",
        {
            "study_id": study.STUDY_ID,
            "study_sha256": freeze["study_sha256"],
            "execution_id": execution.EXECUTION_ID,
            **required_identities,
        },
    )
    output_digests["accepted-g2-identities.json"] = write_json(
        out / "accepted-g2-identities.json",
        {
            "study_id": study.STUDY_ID,
            "execution_id": execution.EXECUTION_ID,
            "accepted_capture": capture,
            **accepted_identities,
            "required_vs_accepted": comparisons,
        },
    )
    output_digests["cache-inventory.json"] = write_json(out / "cache-inventory.json", inventory)
    output_digests["coverage.json"] = write_json(
        out / "coverage.json",
        {
            "study_id": study.STUDY_ID,
            "study_sha256": freeze["study_sha256"],
            "execution_id": execution.EXECUTION_ID,
            **coverage,
        },
    )
    output_digests["execution-ledger.json"] = write_json(
        out / "execution-ledger.json", ledger
    )
    output_digests["cost.json"] = write_json(out / "cost.json", cost)
    output_digests["custody.json"] = write_json(out / "custody.json", custody)
    if blocked_card is not None:
        output_digests["blocked.json"] = write_json(out / "blocked.json", blocked_card)

    run_record = {
        "execution_id": execution.EXECUTION_ID,
        "study_id": study.STUDY_ID,
        "study_sha256": freeze["study_sha256"],
        "source_commit": source_commit(),
        "accepted_g2": {
            "execution_commit": execution.ACCEPTED_G2_EXECUTION_COMMIT,
            "source_commit": execution.ACCEPTED_G2_SOURCE_COMMIT,
            "runner_sha256": execution.ACCEPTED_G2_RUNNER_SHA256,
            "accepted_root": str(g2_root.resolve()),
        },
        "manifests": manifest_records,
        "histories_audited": sorted(manifests),
        "horizons": list(study.HORIZONS),
        "deferred_horizons": ["day"],
        "modes": list(study.MODES),
        "representation_seeds": list(study.REPRESENTATION_SEEDS),
        "manifest_digest": observed_digests,
        "admission": admission,
        "outcome": (
            "BLOCKED-CACHE-PREFLIGHT"
            if not admission["complete"]
            else "ADMITTED-PREFLIGHT-ONLY"
        ),
        "dispositions_computed": [],
        "movement_computed": False,
        "forwards": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "embedding_scores_read": 0,
        "checkpoint_loads": 0,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "timing": {
            "started_unix": started,
            "finished_unix": finished,
            "analysis_seconds": finished - started,
        },
        "cell_execution_status_counts": ledger["cell_execution_status_counts"],
        "outputs": output_digests,
        "limitations": [
            "cache admission only: no movement, contrast, uncertainty or disposition is computed",
            "no embedding cache with the required identity was found in the accepted lineage",
            "the accepted G2 capture was memory-only; its window statistics cannot substitute for per-period samples",
            "no new forward, cap reset, model load, fitting, corpus generation or Confirmation contact",
            "scaling and Phase C remain unreleased",
        ],
    }
    output_digests["run.json"] = write_json(out / "run.json", run_record)

    print(
        json.dumps(
            {
                "execution_id": execution.EXECUTION_ID,
                "study_sha256": freeze["study_sha256"],
                "admission": admission["status"],
                "required_rows_per_seed": totals["rows_per_seed"],
                "available_rows_total": coverage["total_available_rows"],
                "missing_rows_total": coverage["total_missing_rows"],
                "cache_roots": inventory["roots"],
                "n_cells": ledger["n_cells"],
                "cell_execution_status_counts": ledger["cell_execution_status_counts"],
                "outcome": run_record["outcome"],
                "out": str(out),
            },
            sort_keys=True,
        )
    )
    return 0 if admission["complete"] else EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
