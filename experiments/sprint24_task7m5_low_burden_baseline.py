"""Sprint 24 Task 7-M5 low-burden, stable baseline support audit CLI (metadata only).

Runs the frozen grid from :mod:`representation.sprint24_low_burden_baseline`
against the hash-verified accepted Development manifests via the frozen
weekly/monthly study machinery. The audit freeze record is written to disk
*before* any candidate is counted, and annotations are joined only after the
complete label-blind candidate membership exists.

Metadata-only by construction: no waveform shard, embedding, score, model,
checkpoint or forward is read. The CLI reuses the accepted M3 study freeze and
fails closed on any study/audit/manifest digest mismatch.

Example::

    python experiments/sprint24_task7m5_low_burden_baseline.py \
      --data-root <dir containing DEVELOPMENT/<history>/manifest.json> \
      --edges experiments/sprint24_task7m1_dev-edges.json \
      --expected-manifest-sha256 <history>=<sha256> ... \
      --out artifacts/sprint-24/evidence/s24-7-m5-a01
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

from representation import sprint24_low_burden_baseline as baseline  # noqa: E402
from representation import sprint24_weekly_monthly_study as study  # noqa: E402
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


def write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: "" if row.get(field) is None else row.get(field) for field in fields}
            )
    return sha256_path(path)


def parse_expected(values: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for value in values:
        history_id, separator, digest = value.partition("=")
        if not separator or not history_id or not digest:
            raise SystemExit(f"malformed --expected-manifest-sha256: {value!r}")
        expected[history_id] = digest.strip().lower()
    return expected


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
        "--expected-audit-sha256",
        default=None,
        help="claim-check of the frozen audit digest before analysis",
    )
    parser.add_argument(
        "--expected-study-sha256",
        default=None,
        help="claim-check of the frozen upstream study digest before analysis",
    )
    parser.add_argument(
        "--protocol-doc",
        default="docs/sprint24-low-burden-baseline-support-audit.md",
        help="report path recorded by digest in run.json",
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
    freeze = baseline.audit_freeze()
    if args.expected_audit_sha256 and freeze["audit_sha256"] != args.expected_audit_sha256:
        raise SystemExit(
            f"frozen audit digest mismatch: {freeze['audit_sha256']} != {args.expected_audit_sha256}"
        )
    if args.expected_study_sha256 and study.study_sha256() != args.expected_study_sha256:
        raise SystemExit("frozen upstream study digest mismatch")
    output_digests: dict[str, str] = {}
    output_digests["audit-freeze.json"] = write_json(out / "audit-freeze.json", freeze)
    freeze_written = time.time()

    baseline.assert_candidate_construction_label_blind()
    study.assert_study_construction_label_blind()
    primary = study.primary_calendar()
    result = study.evaluate_source(manifests, edges, primary, modes=study.MODES)
    if result["proposal_sha256"] != baseline.horizon.proposal_sha256():
        raise SystemExit("upstream proposal digest drifted during the run")
    if study.study_sha256() != freeze["spec"]["upstream_study"]["study_sha256"]:
        raise SystemExit("upstream study digest drifted during the run")
    study_manifest = study.build_label_blind_manifest(result, primary)
    posthoc = study.build_posthoc_burden(result)
    audit = baseline.audit_manifest(study_manifest, posthoc)
    if audit["audit_sha256"] != freeze["audit_sha256"]:
        raise SystemExit("audit digest changed between freeze and analysis")
    analysis_finished = time.time()

    output_digests["period-candidates.json"] = write_json(
        out / "period-candidates.json",
        {
            "audit_id": baseline.AUDIT_ID,
            "audit_sha256": freeze["audit_sha256"],
            "row_fields": list(baseline.PERIOD_CANDIDATE_FIELDS),
            "rows": audit["period_rows"],
        },
    )
    output_digests["period-candidates.csv"] = write_csv(
        out / "period-candidates.csv",
        audit["period_rows"],  # type: ignore[arg-type]
        baseline.PERIOD_CANDIDATE_FIELDS,
    )
    output_digests["pair-regions.json"] = write_json(
        out / "pair-regions.json",
        {
            "audit_id": baseline.AUDIT_ID,
            "audit_sha256": freeze["audit_sha256"],
            "row_fields": list(baseline.PAIR_REGION_FIELDS),
            "rows": audit["pair_rows"],
        },
    )
    output_digests["pair-regions.csv"] = write_csv(
        out / "pair-regions.csv",
        audit["pair_rows"],  # type: ignore[arg-type]
        baseline.PAIR_REGION_FIELDS,
    )
    output_digests["cells.json"] = write_json(
        out / "cells.json",
        {
            "audit_id": baseline.AUDIT_ID,
            "audit_sha256": freeze["audit_sha256"],
            "cells": audit["cells"],
            "overall": audit["overall"],
        },
    )
    output_digests["study-manifest-digest.json"] = write_json(
        out / "study-manifest-digest.json",
        {
            "study_id": study.STUDY_ID,
            "study_sha256": study.study_sha256(),
            "manifest_digest": study_manifest["digest"],
            "manifest_validity": study_manifest["validity"],
            "posthoc_digest": posthoc["digest"],
        },
    )

    protocol_path = REPO / args.protocol_doc
    protocol_record: dict[str, object] = (
        {
            "path": str(protocol_path),
            "sha256": sha256_path(protocol_path),
            "bytes": protocol_path.stat().st_size,
        }
        if protocol_path.is_file()
        else {"path": str(protocol_path), "sha256": None, "bytes": 0}
    )
    run_record = {
        "audit_id": baseline.AUDIT_ID,
        "audit_sha256": freeze["audit_sha256"],
        "upstream_study": freeze["spec"]["upstream_study"],
        "report_document": protocol_record,
        "mode": "metadata-only",
        "source_commit": source_commit(),
        "edges": {
            "path": str(Path(args.edges).resolve()),
            "sha256": edge_digest,
            "values": edges,
        },
        "manifests": manifest_records,
        "histories_audited": sorted(manifests),
        "horizons": list(baseline.HORIZONS),
        "deferred_horizons": ["day"],
        "modes": list(baseline.MODES),
        "calendar": primary.describe(),
        "label_blind_construction": (
            "study manifest rows are structural only; audit candidate sets reuse "
            "label-blind identity keys; annotations classify afterwards"
        ),
        "label_blind_guard": "assert_candidate_construction_label_blind passed",
        "provenance_note": (
            "prior corpus distribution knowledge (G2/M1/M2/M3/M4 results such as "
            "zero clean controls) is exploratory prior knowledge used to motivate "
            "this audit grid; no embedding movement output was inspected to choose "
            "audit candidates, and every candidate/support count below was computed "
            "only after the audit freeze above"
        ),
        "overall": audit["overall"],
        "forwards": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "embedding_scores_read": 0,
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
            "metadata only: burden, support and comparability counts; never calibrated specificity",
            "thresholds are frozen research assumptions, not owner-approved operational cutoffs",
            "no annotated-clean baseline exists; no specificity or deployable claim follows",
            "nonoverlapping pair counts are distinct-period support, not independence proof",
            "cross-region (baseline versus elevated) robot/program mix comparability is a missing dimension",
            "within-file burden has no accepted annotation field and is unavailable",
            "four histories at n=4 make every cross-history statement descriptive",
            "the daily horizon is deferred and no daily claim is made",
        ],
    }
    output_digests["run.json"] = write_json(out / "run.json", run_record)

    print(
        json.dumps(
            {
                "audit_sha256": freeze["audit_sha256"],
                "n_period_rows": len(audit["period_rows"]),
                "n_pair_rows": len(audit["pair_rows"]),
                "overall": {
                    horizon_name: block["disposition"]
                    for horizon_name, block in audit["overall"].items()  # type: ignore[union-attr]
                },
                "out": str(out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
