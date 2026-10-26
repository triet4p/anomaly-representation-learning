"""Sprint 24 Task 7-M3 weekly/monthly prospective study CLI (metadata only).

Freezes the ``S24-BIZ-WM-v1`` study contract and emits the label-blind study
manifests from hash-verified accepted Development metadata:

- the study freeze record is written **before** any membership or support value is
  computed;
- the period/pair/reference manifests carry deterministic identities and exclude
  every post-hoc annotation field;
- post-hoc burden is a separate join keyed by those identities;
- the declared cell matrix, contrast availability and metadata estimability
  pre-check are published with explicit unsupported states;
- illustrative decision-rule examples are labelled as invented inputs, never as
  measurements.

Metadata-only by construction: no waveform, embedding, score, cache, checkpoint or
model is opened and no forward is executed.

Example::

    python experiments/sprint24_task7m3_weekly_monthly.py \\
      --data-root <path with DEVELOPMENT/<history>/manifest.json> \\
      --edges experiments/sprint24_task7m1_dev-edges.json \\
      --expected-manifest-sha256 <history>=<sha256> ... \\
      --out artifacts/sprint-24/evidence/s24-7-m3-a01
"""

from __future__ import annotations

import argparse
import csv
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

from representation import sprint24_business_horizon as horizon  # noqa: E402
from representation import sprint24_weekly_monthly_study as study  # noqa: E402
from representation.sprint24_context_audit import canonical_json  # noqa: E402


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def write_json(path: Path, payload: object) -> str:
    text = canonical_json(payload) + "\n"
    path.write_text(text, encoding="utf-8", newline="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return sha256_path(path)


def parse_expected(values: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--expected-manifest-sha256 needs <history>=<sha256>: {value!r}")
        history_id, digest = value.split("=", 1)
        expected[history_id.strip()] = digest.strip()
    return expected


def _horizon_estimability_rows(coverage: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    blocks = coverage["horizons"]
    assert isinstance(blocks, dict)
    for horizon_name, modes in blocks.items():
        assert isinstance(modes, dict)
        for mode, block in modes.items():
            assert isinstance(block, dict)
            counts = block["counts"]
            estimability_block = block["estimability"]
            precheck = block["expected_disposition"]
            assert isinstance(counts, dict)
            assert isinstance(estimability_block, dict)
            for history_id, entry in sorted(counts.items()):
                rows.append(
                    {
                        "horizon": horizon_name,
                        "mode": mode,
                        "mode_role": "PRIMARY" if mode == study.PRIMARY_MODE else "SENSITIVITY",
                        "history_id": history_id,
                        "n_matched_pairs": entry["matched_pairs"],
                        "n_unmatched_pairs": entry["unmatched_pairs"],
                        "n_comparable_pairs": entry["comparable_pairs"],
                        "n_distinct_matched_burden_values": entry[
                            "distinct_matched_burden_values"
                        ],
                        "n_matched_pairs_with_failure_events": entry[
                            "matched_pairs_with_failure_events"
                        ],
                        "median_matched_mix_drift_tv": entry["median_matched_mix_drift_tv"],
                        "median_matched_exposure_ratio": entry["median_matched_exposure_ratio"],
                        "estimable": estimability_block["per_history"][history_id]["estimable"],
                        "estimability_reason": estimability_block["per_history"][history_id][
                            "reason"
                        ],
                        "precheck_status": precheck["status"],
                        "precheck_disposition": precheck["disposition"],
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--expected-manifest-sha256",
        action="append",
        default=[],
        help="repeatable <history>=<sha256>; every audited history must be bound",
    )
    parser.add_argument(
        "--expected-study-sha256",
        default=None,
        help="claim-check of the frozen study digest before analysis",
    )
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument(
        "--protocol-doc",
        default=study.PROTOCOL_DOC,
        help="versioned protocol document recorded by digest in run.json",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    edges_raw = json.loads(Path(args.edges).read_text(encoding="utf-8"))
    if not isinstance(edges_raw, dict):
        raise SystemExit("edges file must map history id to its 11 decile edges")
    edges = {str(key): [float(value) for value in values] for key, values in edges_raw.items()}
    expected = parse_expected(args.expected_manifest_sha256)
    if set(expected) != set(edges):
        raise SystemExit(
            "expected manifest digests must cover exactly the audited histories: "
            f"{sorted(set(edges) ^ set(expected))}"
        )

    edge_digest = sha256_path(Path(args.edges))
    manifests: dict[str, dict[str, object]] = {}
    manifest_records: dict[str, dict[str, object]] = {}
    for history_id in sorted(edges):
        path = data_root / "DEVELOPMENT" / history_id / "manifest.json"
        if not path.is_file():
            raise SystemExit(f"missing accepted manifest: {path}")
        digest = sha256_path(path)
        if digest != expected[history_id]:
            raise SystemExit(
                f"accepted manifest digest mismatch for {history_id}: "
                f"{digest} != {expected[history_id]}"
            )
        manifests[history_id] = json.loads(path.read_text(encoding="utf-8"))
        manifest_records[history_id] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    freeze = study.study_freeze()
    if args.expected_study_sha256 and freeze["study_sha256"] != args.expected_study_sha256:
        raise SystemExit(
            f"frozen study digest mismatch: {freeze['study_sha256']} != "
            f"{args.expected_study_sha256}"
        )
    output_digests: dict[str, str] = {}
    output_digests["study-freeze.json"] = write_json(out / "study-freeze.json", freeze)
    freeze_written = time.time()

    # Construction and annotation projection happen inside the inherited metadata
    # pass; the study then derives label-blind manifests from the structural rows.
    primary = study.primary_calendar()
    result = study.evaluate_source(manifests, edges, primary, modes=study.MODES)
    if result["proposal_sha256"] != horizon.proposal_sha256():
        raise SystemExit("upstream proposal digest drifted during the run")

    study.assert_study_construction_label_blind()
    study_manifest = study.build_label_blind_manifest(result, primary)
    posthoc = study.build_posthoc_burden(result)
    coverage = study.coverage_matrix(study_manifest, posthoc)
    sensitivity = None
    if not args.skip_sensitivity:
        sensitivity = study.projection_sensitivity(
            manifests, edges, study.sensitivity_calendars(), modes=study.MODES
        )
    analysis_finished = time.time()

    output_digests["study-manifest-references.json"] = write_json(
        out / "study-manifest-references.json",
        {
            "study_id": study.STUDY_ID,
            "study_sha256": freeze["study_sha256"],
            "calendar": study_manifest["calendar"],
            "references": study_manifest["references"],
            "validity": study_manifest["validity"],
            "digest": study_manifest["digest"],
        },
    )
    output_digests["study-manifest-periods.json"] = write_json(
        out / "study-manifest-periods.json",
        {
            "study_id": study.STUDY_ID,
            "study_sha256": freeze["study_sha256"],
            "row_fields": list(study.PERIOD_STRUCTURAL_FIELDS) + ["period_identity", "row_sha256"],
            "rows": study_manifest["periods"],
        },
    )
    output_digests["study-manifest-periods.csv"] = write_csv(
        out / "study-manifest-periods.csv",
        study_manifest["periods"],  # type: ignore[arg-type]
        study.PERIOD_STRUCTURAL_FIELDS + ("period_identity", "row_sha256"),
    )
    output_digests["study-manifest-pairs.json"] = write_json(
        out / "study-manifest-pairs.json",
        {
            "study_id": study.STUDY_ID,
            "study_sha256": freeze["study_sha256"],
            "row_fields": list(study.PAIR_STRUCTURAL_FIELDS)
            + [
                "previous_period_identity",
                "current_period_identity",
                "pair_identity",
                "pair_members_sha256",
                "row_sha256",
            ],
            "rows": study_manifest["pairs"],
        },
    )
    output_digests["study-manifest-pairs.csv"] = write_csv(
        out / "study-manifest-pairs.csv",
        study_manifest["pairs"],  # type: ignore[arg-type]
        study.PAIR_STRUCTURAL_FIELDS
        + (
            "previous_period_identity",
            "current_period_identity",
            "pair_identity",
            "pair_members_sha256",
            "row_sha256",
        ),
    )
    output_digests["posthoc-burden.json"] = write_json(out / "posthoc-burden.json", posthoc)
    output_digests["coverage-matrix.json"] = write_json(
        out / "coverage-matrix.json",
        {
            "study_id": study.STUDY_ID,
            "study_sha256": freeze["study_sha256"],
            "declared_cell_count": coverage["declared_cell_count"],
            "n_cells": coverage["n_cells"],
            "status_counts": coverage["status_counts"],
            "cells": coverage["cells"],
            "contrasts": coverage["contrasts"],
            "contrast_counts": coverage["contrast_counts"],
            "notes": coverage["notes"],
        },
    )
    output_digests["coverage-cells.csv"] = write_csv(
        out / "coverage-cells.csv",
        coverage["cells"],  # type: ignore[arg-type]
        (
            "horizon",
            "mode",
            "mode_role",
            "endpoint_id",
            "endpoint_role",
            "decision_relevant",
            "representation_seed",
            "history_id",
            "n_comparable_pairs",
            "n_matched_pairs",
            "distinct_burden_values",
            "median_mix_drift_tv",
            "member_rows_per_seed",
            "status",
        ),
    )
    estimability_rows = _horizon_estimability_rows(coverage)
    output_digests["estimability.csv"] = write_csv(
        out / "estimability.csv",
        estimability_rows,
        (
            "horizon",
            "mode",
            "mode_role",
            "history_id",
            "n_matched_pairs",
            "n_unmatched_pairs",
            "n_comparable_pairs",
            "n_distinct_matched_burden_values",
            "n_matched_pairs_with_failure_events",
            "median_matched_mix_drift_tv",
            "median_matched_exposure_ratio",
            "estimable",
            "estimability_reason",
            "precheck_status",
            "precheck_disposition",
        ),
    )
    output_digests["estimability.json"] = write_json(
        out / "estimability.json",
        {
            "study_id": study.STUDY_ID,
            "study_sha256": freeze["study_sha256"],
            "row_fields": (
                "horizon",
                "mode",
                "mode_role",
                "history_id",
                "n_matched_pairs",
                "n_unmatched_pairs",
                "n_comparable_pairs",
                "n_distinct_matched_burden_values",
                "n_matched_pairs_with_failure_events",
                "median_matched_mix_drift_tv",
                "median_matched_exposure_ratio",
                "estimable",
                "estimability_reason",
                "precheck_status",
                "precheck_disposition",
            ),
            "rows": estimability_rows,
            "precheck": {
                horizon_name: {
                    mode: block["expected_disposition"]
                    for mode, block in modes.items()
                }
                for horizon_name, modes in coverage["horizons"].items()  # type: ignore[union-attr]
            },
        },
    )
    if sensitivity is not None:
        output_digests["projection-sensitivity.json"] = write_json(
            out / "projection-sensitivity.json", sensitivity
        )
    worked = study.worked_decision_examples()
    output_digests["worked-decision-examples.json"] = write_json(
        out / "worked-decision-examples.json", worked
    )

    history_totals = result["history_totals"]
    assert isinstance(history_totals, dict)
    span_rows = sum(
        int(entry["n_eligible_analysis_span"]) for entry in history_totals.values()
    )
    references = result["references"]
    assert isinstance(references, dict)
    reference_rows = sum(int(entry["reference_files"]) for entry in references.values())
    cost = study.assert_cost_within_declared_bounds(span_rows, reference_rows)
    output_digests["cost.json"] = write_json(
        out / "cost.json",
        {
            "study_id": study.STUDY_ID,
            "study_sha256": freeze["study_sha256"],
            "policy": study.cost_policy(),
            "accounting": cost,
            "histories": {
                history_id: {
                    "eligible_analysis_span": int(entry["n_eligible_analysis_span"]),
                    "reference_files": int(references[history_id]["reference_files"]),
                    "reference_state": references[history_id]["reference_state"],
                }
                for history_id, entry in sorted(history_totals.items())
            },
            "authorization": (
                "no inference is authorized by this record; a later execution task needs "
                "zero-new-forward cache reuse or an explicit bounded approval"
            ),
        },
    )

    protocol_path = REPO / args.protocol_doc
    protocol_record = (
        {
            "path": str(protocol_path),
            "sha256": sha256_path(protocol_path),
            "bytes": protocol_path.stat().st_size,
        }
        if protocol_path.is_file()
        else {"path": str(protocol_path), "sha256": None, "bytes": 0}
    )
    precheck_primary = {
        horizon_name: coverage["horizons"][horizon_name][study.PRIMARY_MODE][  # type: ignore[index]
            "expected_disposition"
        ]
        for horizon_name in study.HORIZONS
    }
    run_record = {
        "study_id": study.STUDY_ID,
        "study_sha256": freeze["study_sha256"],
        "upstream_proposal": {
            "proposal_id": horizon.PROPOSAL_ID,
            "proposal_sha256": horizon.proposal_sha256(),
        },
        "protocol_document": protocol_record,
        "mode": "metadata-only",
        "source_commit": source_commit(),
        "edges": {
            "path": str(Path(args.edges).resolve()),
            "sha256": edge_digest,
            "values": edges,
        },
        "manifests": manifest_records,
        "histories_audited": sorted(manifests),
        "horizons": list(study.HORIZONS),
        "deferred_horizons": ["day"],
        "modes": list(study.MODES),
        "representation_seeds": list(study.REPRESENTATION_SEEDS),
        "calendar": primary.describe(),
        "label_blind_construction": (
            "study manifest rows carry structural fields only; post-hoc burden is a "
            "separate identity-keyed join"
        ),
        "label_blind_guard": "assert_study_construction_label_blind passed",
        "manifest_validity": study_manifest["validity"],
        "manifest_digest": study_manifest["digest"],
        "metadata_precheck": precheck_primary,
        "dispositions_computed": [],
        "dispositions_note": (
            "no disposition is computed here: the decision rule is exercised only by the "
            "illustrative examples; real outcomes require a separately authorized run"
        ),
        "forwards": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "embedding_scores_read": 0,
        "cost": cost,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "timing": {
            "freeze_written_unix": freeze_written,
            "analysis_finished_unix": analysis_finished,
            "freeze_seconds": freeze_written - started,
            "analysis_seconds": analysis_finished - freeze_written,
        },
        "outputs": output_digests,
        "limitations": [
            "metadata only: no movement, embedding, score or model outcome is computed",
            "the calendar anchor, timezone, week start and month anchor day are declared research assumptions",
            "the accepted corpus supplies zero annotated-clean and zero quiet-period pairs at both horizons",
            "four histories at n=4 make every cross-history statement descriptive",
            "the daily horizon is deferred and no daily claim is made",
            "scaling is a separate release and is not authorized by this record",
        ],
    }
    output_digests["run.json"] = write_json(out / "run.json", run_record)

    print(
        json.dumps(
            {
                "study_sha256": freeze["study_sha256"],
                "n_period_rows": len(study_manifest["periods"]),  # type: ignore[arg-type]
                "n_pair_rows": len(study_manifest["pairs"]),  # type: ignore[arg-type]
                "n_cells": coverage["n_cells"],
                "status_counts": coverage["status_counts"],
                "metadata_precheck": {
                    horizon_name: precheck_primary[horizon_name]["status"]
                    for horizon_name in study.HORIZONS
                },
                "out": str(out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
