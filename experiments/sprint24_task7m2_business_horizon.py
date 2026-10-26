"""Sprint 24 Task 7-M2 business-horizon measurement feasibility CLI.

Runs the frozen period/exposure machinery from
:mod:`representation.sprint24_business_horizon` against hash-verified accepted
Development manifests. The proposal freeze record is written to disk *before*
any support statistic is computed; annotations are joined only after the complete
period and pair membership exists.

Metadata-only by construction: no waveform shard, embedding, score, model,
checkpoint or forward is read. This CLI publishes feasibility, membership,
exposure and support evidence for the prospective design; it executes no
movement endpoint.

Example::

    python experiments/sprint24_task7m2_business_horizon.py \
      --data-root <path with DEVELOPMENT/<history>/manifest.json> \
      --edges experiments/sprint24_task7m1_dev-edges.json \
      --expected-manifest-sha256 <history>=<sha256> ... \
      --out artifacts/sprint-24/evidence/s24-7-m2-a01
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
from representation.sprint24_context_audit import canonical_json  # noqa: E402

#: Primary declared calendar projection for the executed feasibility pass.
PRIMARY_ANCHOR_EPOCH_S = 1_704_067_200.0  # 2024-01-01T00:00:00Z (a Monday)

#: Alternate anchors are declared assumption probes, not business facts.
ALTERNATE_ANCHOR_EPOCH_S = 1_709_769_600.0  # 2024-03-07T00:00:00Z (a Thursday)


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


def calendar_variants() -> dict[str, horizon.CalendarConfig]:
    """Declared projection variants used only to measure boundary sensitivity."""
    return {
        "primary": horizon.CalendarConfig(
            anchor_epoch_s=PRIMARY_ANCHOR_EPOCH_S,
            timezone="UTC",
            week_start="monday",
            month_anchor_day=15,
        ),
        "alternate_anchor_week_zone": horizon.CalendarConfig(
            anchor_epoch_s=ALTERNATE_ANCHOR_EPOCH_S,
            timezone="Etc/GMT-7",
            week_start="sunday",
            month_anchor_day=15,
        ),
    }


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
        "--expected-proposal-sha256",
        default=None,
        help="claim-check of the frozen proposal digest before analysis",
    )
    parser.add_argument(
        "--worked-example-history",
        default=None,
        help="history used for the real worked examples (default: first audited)",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="skip the multi-projection sensitivity pass",
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
    freeze = horizon.proposal_freeze()
    if args.expected_proposal_sha256 and freeze["proposal_sha256"] != args.expected_proposal_sha256:
        raise SystemExit(
            f"frozen proposal digest mismatch: {freeze['proposal_sha256']} != {args.expected_proposal_sha256}"
        )
    output_digests: dict[str, str] = {}
    output_digests["proposal-freeze.json"] = write_json(out / "proposal-freeze.json", freeze)
    freeze_written = time.time()

    variants = calendar_variants()
    primary = variants["primary"]
    result = horizon.evaluate(manifests, edges, primary, modes=horizon.MODES)
    if result["proposal_sha256"] != freeze["proposal_sha256"]:
        raise SystemExit("proposal digest changed between freeze and analysis")
    sensitivity = None
    if not args.skip_sensitivity:
        sensitivity = horizon.calendar_sensitivity(manifests, edges, variants, modes=horizon.MODES)
    analysis_finished = time.time()

    period_rows = result["period_rows"]
    pair_rows = result["pair_rows"]
    assert isinstance(period_rows, list) and isinstance(pair_rows, list)
    output_digests["period-table.json"] = write_json(
        out / "period-table.json",
        {
            "proposal_id": horizon.PROPOSAL_ID,
            "proposal_sha256": freeze["proposal_sha256"],
            "row_fields": list(horizon.PERIOD_ROW_FIELDS),
            "field_units": horizon.PERIOD_FIELD_UNITS,
            "rows": period_rows,
        },
    )
    output_digests["period-table.csv"] = write_csv(
        out / "period-table.csv", period_rows, horizon.PERIOD_ROW_FIELDS
    )
    output_digests["pair-table.json"] = write_json(
        out / "pair-table.json",
        {
            "proposal_id": horizon.PROPOSAL_ID,
            "proposal_sha256": freeze["proposal_sha256"],
            "row_fields": list(horizon.PAIR_ROW_FIELDS),
            "field_units": horizon.PAIR_ROW_FIELD_UNITS,
            "rows": pair_rows,
        },
    )
    output_digests["pair-table.csv"] = write_csv(
        out / "pair-table.csv", pair_rows, horizon.PAIR_ROW_FIELDS
    )
    output_digests["feasibility-matrix.json"] = write_json(
        out / "feasibility-matrix.json",
        {
            "proposal_id": horizon.PROPOSAL_ID,
            "proposal_sha256": freeze["proposal_sha256"],
            "calendar": primary.describe(),
            "row_fields": list(horizon.FEASIBILITY_ROW_FIELDS),
            "row_field_units": horizon.FEASIBILITY_FIELD_UNITS,
            "matrix_fields": list(horizon.HORIZON_MATRIX_FIELDS),
            "matrix_field_units": horizon.HORIZON_MATRIX_FIELD_UNITS,
            "rows": result["feasibility_rows"],
            "horizon_matrix": result["horizon_matrix"],
            "references": result["references"],
        },
    )
    worked = horizon.worked_examples(
        manifests,
        edges,
        primary,
        history_id=args.worked_example_history,
    )
    output_digests["worked-examples.json"] = write_json(out / "worked-examples.json", worked)
    output_digests["illustrative-examples.json"] = write_json(
        out / "illustrative-examples.json",
        horizon.illustrative_calendar_examples(primary),
    )
    if sensitivity is not None:
        output_digests["sensitivity.json"] = write_json(out / "sensitivity.json", sensitivity)

    history_totals = result["history_totals"]
    assert isinstance(history_totals, dict)
    eligible_span = sum(
        int(entry["n_eligible_analysis_span"]) for entry in history_totals.values()
    )
    references = result["references"]
    assert isinstance(references, dict)
    reference_rows = sum(int(entry["reference_files"]) for entry in references.values())
    run_record = {
        "proposal_id": horizon.PROPOSAL_ID,
        "proposal_sha256": freeze["proposal_sha256"],
        "mode": "metadata-only",
        "source_commit": source_commit(),
        "edges": {"path": str(Path(args.edges).resolve()), "sha256": edge_digest, "values": edges},
        "manifests": manifest_records,
        "histories_audited": sorted(manifests),
        "history_totals": history_totals,
        "eligible_files_analysis_span": eligible_span,
        "eligible_files_reference_pool": reference_rows,
        "eligible_files_reference_plus_span": eligible_span + reference_rows,
        "embedding_row_cap_span_only_three_seeds": 3 * eligible_span,
        "embedding_row_cap_reference_plus_span_three_seeds": 3 * (eligible_span + reference_rows),
        "calendar": primary.describe(),
        "boundary_modes": list(horizon.MODES),
        "label_blind_construction": "project_observable strips annotations before construction",
        "annotations": "joined after period/pair membership; post-hoc counts only",
        "encoder_seeds": [181801, 181802, 181803],
        "encoder_seed_multiplicity": 3,
        "encoder_seed_independent_data": False,
        "forwards": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "embedding_scores_read": 0,
        "label_blind_guard": "assert_period_construction_label_blind passed",
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
            "feasibility/support/membership counts only; no embedding movement is computed",
            "calendar anchor, timezone, week start and month anchor day are declared assumptions",
            "no annotated-clean or quiet-period control exists on the audited Development span",
            "overlapping conventions and repeated encoder seeds are not independent support",
            "zero-support and zero-control outcomes are reported, never relaxed to fit a floor",
        ],
    }
    output_digests["run.json"] = write_json(out / "run.json", run_record)

    matrix = result["horizon_matrix"]
    assert isinstance(matrix, list)
    print(
        json.dumps(
            {
                "proposal_sha256": freeze["proposal_sha256"],
                "n_period_rows": len(period_rows),
                "n_pair_rows": len(pair_rows),
                "horizon_states": {
                    f"{row['horizon']}|{row['mode']}": row["horizon_state"] for row in matrix
                },
                "control_states": {
                    f"{row['horizon']}|{row['mode']}": row["control_state"] for row in matrix
                },
                "out": str(out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
