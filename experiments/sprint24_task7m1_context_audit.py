"""Sprint 24 Task 7-M1 metadata-only context/time support audit CLI.

Runs the frozen candidate grid from
:mod:`representation.sprint24_context_audit` against hash-verified accepted
Development manifests. The freeze record is written to disk *before* any
support statistic is computed; annotations are joined only after the complete
candidate membership exists.

Metadata-only by construction: no waveform shard, embedding, score, model,
checkpoint or forward is read. ``--emit-windows`` adds per-window membership
digests for smoke/debug verification and is not part of the audit result.

Example::

    python experiments/sprint24_task7m1_context_audit.py \
      --data-root <path with DEVELOPMENT/<history>/manifest.json> \
      --edges experiments/sprint24_task7m1_edges.json \
      --expected-manifest-sha256 <history>=<sha256> ... \
      --out artifacts/sprint-24/evidence/s24-7-m1-a01
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

from representation import sprint24_context_audit as audit  # noqa: E402


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def write_json(path: Path, payload: object) -> str:
    text = audit.canonical_json(payload) + "\n"
    path.write_text(text, encoding="utf-8", newline="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit.SUPPORT_ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "" if row.get(field) is None else row.get(field)
                    for field in audit.SUPPORT_ROW_FIELDS
                }
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
        "--expected-grid-sha256",
        default=None,
        help="claim-check of the frozen grid digest before analysis",
    )
    parser.add_argument(
        "--emit-windows",
        action="store_true",
        help="write windows.jsonl per-window membership digests (smoke/debug only)",
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
                f"accepted manifest digest mismatch for {history_id}: {digest} != {expected[history_id]}"
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
    freeze = audit.grid_freeze()
    if args.expected_grid_sha256 and freeze["grid_sha256"] != args.expected_grid_sha256:
        raise SystemExit(
            f"frozen grid digest mismatch: {freeze['grid_sha256']} != {args.expected_grid_sha256}"
        )
    freeze_digest = write_json(out / "grid-freeze.json", freeze)
    freeze_written = time.time()

    result = audit.audit_histories(manifests, edges, include_windows=args.emit_windows)
    analysis_finished = time.time()
    if result["grid_sha256"] != freeze["grid_sha256"]:
        raise SystemExit("grid digest changed between freeze and analysis")

    support_rows = result["support_rows"]
    assert isinstance(support_rows, list)
    output_digests: dict[str, str] = {"grid-freeze.json": freeze_digest}
    output_digests["support-table.json"] = write_json(
        out / "support-table.json",
        {
            "audit_id": audit.AUDIT_ID,
            "grid_sha256": freeze["grid_sha256"],
            "row_fields": list(audit.SUPPORT_ROW_FIELDS),
            "rows": support_rows,
        },
    )
    output_digests["support-table.csv"] = write_csv(out / "support-table.csv", support_rows)
    output_digests["summary.json"] = write_json(
        out / "summary.json",
        {"audit_id": audit.AUDIT_ID, "grid_sha256": freeze["grid_sha256"], "summary": result["summary"]},
    )
    output_digests["field-audit.json"] = write_json(
        out / "field-audit.json",
        {"audit_id": audit.AUDIT_ID, "histories": result["field_audit"]},
    )
    output_digests["label-audit.json"] = write_json(
        out / "label-audit.json",
        {"audit_id": audit.AUDIT_ID, "histories": result["label_audit"]},
    )
    if args.emit_windows:
        window_path = out / "windows.jsonl"
        with window_path.open("w", encoding="utf-8") as handle:
            for row in result["window_rows"]:  # type: ignore[index]
                handle.write(audit.canonical_json(row) + "\n")
        output_digests["windows.jsonl"] = sha256_path(window_path)

    run_record = {
        "audit_id": audit.AUDIT_ID,
        "grid_sha256": freeze["grid_sha256"],
        "mode": "metadata-only",
        "source_commit": source_commit(),
        "edges": {"path": str(Path(args.edges).resolve()), "sha256": edge_digest, "values": edges},
        "manifests": manifest_records,
        "histories_audited": sorted(manifests),
        "label_blind_construction": "project_observable strips annotations before construction",
        "annotations": "joined after candidate membership; post-hoc support counts only",
        "encoder_seeds": [181801, 181802, 181803],
        "encoder_seed_multiplicity": 3,
        "encoder_seed_independent_data": False,
        "forwards": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "label_blind_guard": "assert_construction_label_blind passed",
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
            "feasibility/support counts only; never calibrated specificity",
            "overlapping windows and repeated encoder seeds are not independent support",
            "zero-support and zero-healthy-control candidates are reported, not dropped",
            "no numeric performance gate is defined or implied",
        ],
    }
    output_digests["run.json"] = write_json(out / "run.json", run_record)

    summary = result["summary"]
    assert isinstance(summary, dict)
    print(json.dumps(
        {
            "grid_sha256": freeze["grid_sha256"],
            "n_support_rows": summary["n_support_rows"],
            "any_posthoc_healthy_control": summary["any_posthoc_healthy_control"],
            "disposition_counts": summary["disposition_counts"],
            "out": str(out),
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
