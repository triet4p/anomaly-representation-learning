"""Sprint 24 Task 7-M6 weekly matched-comparison feasibility CLI (metadata only).

Runs the frozen weekly matching audit (``S24-WEEKLY-MATCHED-v1``) over the
accepted Development metadata: label-blind study manifests, M5 stable-segment
and exposure-compatible elevated region identities, deterministic capped-n
balancing over observable (robot_id, program_id) strata, and post-hoc retained
burden contrast. Zero forwards, zero model loads, zero waveform reads, zero
embedding/score/cache reads.
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
from representation import sprint24_low_burden_baseline as baseline  # noqa: E402
from representation import sprint24_weekly_matched_comparison as matching  # noqa: E402
from representation import sprint24_weekly_monthly_study as study  # noqa: E402
from representation.sprint24_context_audit import canonical_json, project_observable  # noqa: E402
from representation.sprint24_context_audit import prepare_history  # noqa: E402


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
            writer.writerow({key: row.get(key) for key in fields})
    return sha256_path(path)


def parse_expected(values: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for item in values:
        key, _, digest = item.partition("=")
        expected[key.strip()] = digest.strip()
    return expected


def abnormal_flag(label: object) -> int:
    if label is None:
        return 0
    return 0 if str(label) == "normal" else 1


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
    parser.add_argument("--expected-matching-sha256", default=None)
    parser.add_argument("--expected-baseline-audit-sha256", default=None)
    parser.add_argument("--expected-study-sha256", default=None)
    parser.add_argument(
        "--protocol-doc",
        default="docs/sprint24-weekly-matched-comparison-audit.md",
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
            raise SystemExit(f"accepted manifest digest mismatch for {history_id}: {digest} != {expected[history_id]}")
        manifests[history_id] = json.loads(path.read_text(encoding="utf-8"))
        manifest_records[history_id] = {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    freeze = matching.matching_freeze()
    if args.expected_matching_sha256 and freeze["matching_sha256"] != args.expected_matching_sha256:
        raise SystemExit(
            f"frozen matching digest mismatch: {freeze['matching_sha256']} != {args.expected_matching_sha256}"
        )
    if args.expected_baseline_audit_sha256 and baseline.audit_sha256() != args.expected_baseline_audit_sha256:
        raise SystemExit("frozen upstream baseline-audit digest mismatch")
    if args.expected_study_sha256 and study.study_sha256() != args.expected_study_sha256:
        raise SystemExit("frozen upstream study digest mismatch")
    output_digests: dict[str, str] = {}
    output_digests["matching-freeze.json"] = write_json(out / "matching-freeze.json", freeze)
    freeze_written = time.time()

    baseline.assert_candidate_construction_label_blind()
    study.assert_study_construction_label_blind()
    matching.assert_matching_construction_label_blind()
    primary = study.primary_calendar()
    result = study.evaluate_source(manifests, edges, primary, modes=list(matching.MODES))
    if result["proposal_sha256"] != baseline.horizon.proposal_sha256():
        raise SystemExit("upstream proposal digest drifted during the run")
    freeze_spec = freeze["spec"]
    assert isinstance(freeze_spec, dict)
    if study.study_sha256() != freeze_spec["upstream_study"]["study_sha256"]:  # type: ignore[index]
        raise SystemExit("upstream study digest drifted during the run")
    study_manifest = study.build_label_blind_manifest(result, primary)
    posthoc = study.build_posthoc_burden(result)
    audit = baseline.audit_manifest(study_manifest, posthoc)
    if audit["audit_sha256"] != baseline.audit_sha256():
        raise SystemExit("baseline audit digest changed between freeze and analysis")

    # --- M6 region assembly (label-blind membership; post-hoc burden after) ---
    cells = {(str(c["history_id"]), str(c["horizon"]), str(c["mode"])): c for c in audit["cells"]}  # type: ignore[union-attr]
    period_burden = {str(row["period_identity"]): row for row in audit["period_rows"]}  # type: ignore[union-attr]
    comparisons: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    members: dict[str, object] = {}
    control_inventory: list[dict[str, object]] = []
    rejected_elevated: list[dict[str, object]] = []

    for history_id in sorted(edges):
        key = (history_id, "week", "calendar")
        cell = cells.get(key)
        if cell is None:
            raise SystemExit(f"missing M5 week/calendar cell for {history_id}")
        manifest = manifests[history_id]
        meta = prepare_history(history_id, project_observable(manifest), edges[history_id][4])
        grid = horizon.PeriodGrid(horizon="week", mode="calendar", calendar=primary, analysis_start=meta.c1, analysis_end=meta.tmax)
        records = horizon.build_period_records(meta, grid)
        by_key = {record.period.key: record for record in records}
        file_label = {str(row["file_id"]): row.get("file_label") for row in manifest["files"]}  # type: ignore[union-attr]
        cell_of_file = {
            str(row["file_id"]): matching.stratum_identity(str(row["robot_id"]), str(row["program_id"]))
            for row in manifest["files"]  # type: ignore[union-attr]
        }
        singletons = int(cell.get("run_status_counts", {}).get("LOW-RUN-SINGLETON", 0))  # type: ignore[union-attr]
        segments: list[list[str]] = [
            [str(pid).split("|")[-1] for pid in run["period_identities"]]  # type: ignore[union-attr]
            for run in cell["stable_segments"]  # type: ignore[union-attr]
        ]
        compatible: list[str] = sorted(  # elevated period keys, union over segments
            {
                str(pid).split("|")[-1]
                for entry in cell["exposure_compatible_elevated_periods"]  # type: ignore[union-attr]
                for pid in entry["exposure_compatible_elevated_period_identities"]  # type: ignore[union-attr]
            }
        )
        elevated_all = sorted(
            str(row["period_identity"]).split("|")[-1]
            for row in period_burden.values()
            if str(row["history_id"]) == history_id
            and str(row["horizon"]) == "week"
            and str(row["mode"]) == "calendar"
            and str(row["status"]) == "ELEVATED-CANDIDATE"
        )
        for period_key in elevated_all:
            if period_key not in compatible:
                rejected_elevated.append(
                    {
                        "history_id": history_id,
                        "period_identity": f"{history_id}|week|calendar|{period_key}",
                        "reason": "elevated but exposure-incompatible with every stable segment (inherited 0.8 floor); not a contrast region",
                    }
                )

        def arm_files(period_keys: list[str]) -> list[str]:
            out_ids: list[str] = []
            for period_key in period_keys:
                out_ids.extend(list(by_key[period_key].member_ids))
            return out_ids

        def exposure(period_keys: list[str]) -> float:
            return float(sum(by_key[period_key].exposure_days for period_key in period_keys))


        if not segments:
            control_inventory.append(
                {
                    "history_id": history_id,
                    "n_stable_segments": 0,
                    "reserve_control_identities": [],
                    "control_status": "UNAVAILABLE-NO-STABLE-SEGMENT",
                    "control_reason": "history holds no stable baseline segment, so no baseline region and no disjoint control exist",
                    "n_singleton_low_periods": singletons,
                    "singleton_note": "singleton low periods are points, never segments; rejected as controls (not demonstrably stable)",
                }
            )
            continue

        # PRIMARY baseline: latest segment (frozen rule); earlier segments are reserves.
        latest = max(range(len(segments)), key=lambda i: segments[i][-1])
        primary_baseline = segments[latest]
        reserves = [segments[i] for i in range(len(segments)) if i != latest]
        disjoint_control = len(reserves) > 0
        control_inventory.append(
            {
                "history_id": history_id,
                "n_stable_segments": len(segments),
                "primary_baseline_period_keys": primary_baseline,
                "reserve_segment_period_keys": reserves,
                "reserve_control_identities": [
                    f"{history_id}|week|calendar|BASELINE-VS-BASELINE|RESERVE-CONTROL"
                ] if reserves else [],
                "control_status": "AVAILABLE-DISJOINT-RESERVE" if reserves else "UNAVAILABLE-SINGLE-SEGMENT",
                "control_reason": (
                    "reserve segment shares no file and no period with the primary baseline or the contrast region"
                    if reserves
                    else "only one stable segment exists; any control candidate shares periods with the baseline region"
                ),
                "n_singleton_low_periods": singletons,
                "singleton_note": "singleton low periods are points, never segments; rejected as controls (not demonstrably stable)",
            }
        )

        primary_base_ids = arm_files(primary_baseline)
        contrast_ids = arm_files(compatible)
        base_cells = {fid: cell_of_file[fid] for fid in primary_base_ids}
        cont_cells = {fid: cell_of_file[fid] for fid in contrast_ids}
        base_abn = {fid: abnormal_flag(file_label.get(fid)) for fid in primary_base_ids}
        cont_abn = {fid: abnormal_flag(file_label.get(fid)) for fid in contrast_ids}
        row, excl = matching.balance_comparison(
            history_id, "week", "calendar", "BASELINE-VS-ELEVATED", "PRIMARY",
            [f"{history_id}|week|calendar|{k}" for k in primary_baseline],
            [f"{history_id}|week|calendar|{k}" for k in compatible],
            "M5-STABLE-SEGMENT", "M5-EXPOSURE-COMPATIBLE-ELEVATED",
            primary_base_ids, contrast_ids, base_cells, cont_cells, base_abn, cont_abn,
            exposure(primary_baseline), exposure(compatible),
            has_disjoint_control=disjoint_control,
        )
        comparisons.append(row)
        exclusions.extend(excl)
        members[str(row["comparison_identity"])] = {
            "baseline_before": sorted(primary_base_ids),
            "contrast_before": sorted(contrast_ids),
            "retained_baseline": row["retained_baseline_file_identities"],
            "retained_contrast": row["retained_contrast_file_identities"],
        }

        for period_key in compatible:  # sampling sensitivity: one elevated period at a time
            single_ids = arm_files([period_key])
            single_cells = {fid: cell_of_file[fid] for fid in single_ids}
            single_abn = {fid: abnormal_flag(file_label.get(fid)) for fid in single_ids}
            srow, sexcl = matching.balance_comparison(
                history_id, "week", "calendar", "BASELINE-VS-ELEVATED", f"SINGLE-{period_key}",
                [f"{history_id}|week|calendar|{k}" for k in primary_baseline],
                [f"{history_id}|week|calendar|{period_key}"],
                "M5-STABLE-SEGMENT", "M5-EXPOSURE-COMPATIBLE-ELEVATED",
                primary_base_ids, single_ids, base_cells, single_cells, base_abn, single_abn,
                exposure(primary_baseline), exposure([period_key]),
                has_disjoint_control=disjoint_control,
            )
            comparisons.append(srow)
            exclusions.extend(sexcl)
            members[str(srow["comparison_identity"])] = {
                "baseline_before": sorted(primary_base_ids),
                "contrast_before": sorted(single_ids),
                "retained_baseline": srow["retained_baseline_file_identities"],
                "retained_contrast": srow["retained_contrast_file_identities"],
            }

        for reserve in reserves:  # baseline-versus-baseline reserve control
            reserve_ids = arm_files(reserve)
            reserve_cells = {fid: cell_of_file[fid] for fid in reserve_ids}
            reserve_abn = {fid: abnormal_flag(file_label.get(fid)) for fid in reserve_ids}
            crow, cexcl = matching.balance_comparison(
                history_id, "week", "calendar", "BASELINE-VS-BASELINE", "RESERVE-CONTROL",
                [f"{history_id}|week|calendar|{k}" for k in primary_baseline],
                [f"{history_id}|week|calendar|{k}" for k in reserve],
                "M5-STABLE-SEGMENT", "M5-RESERVE-SEGMENT",
                primary_base_ids, reserve_ids, base_cells, reserve_cells, base_abn, reserve_abn,
                exposure(primary_baseline), exposure(reserve),
                has_disjoint_control=True,
            )
            comparisons.append(crow)
            exclusions.extend(cexcl)
            members[str(crow["comparison_identity"])] = {
                "baseline_before": sorted(primary_base_ids),
                "contrast_before": sorted(reserve_ids),
                "retained_baseline": crow["retained_baseline_file_identities"],
                "retained_contrast": crow["retained_contrast_file_identities"],
            }

    analysis_finished = time.time()
    primary_rows = [row for row in comparisons if str(row["comparison_identity"]).endswith("|PRIMARY")]
    overall = matching.summarize_matching(primary_rows)

    output_digests["comparisons.json"] = write_json(
        out / "comparisons.json",
        {
            "matching_id": matching.MATCHING_ID,
            "matching_sha256": freeze["matching_sha256"],
            "row_fields": list(matching.COMPARISON_FIELDS),
            "rows": comparisons,
        },
    )
    output_digests["comparisons.csv"] = write_csv(out / "comparisons.csv", comparisons, matching.COMPARISON_FIELDS)
    output_digests["members.json"] = write_json(
        out / "members.json",
        {
            "matching_id": matching.MATCHING_ID,
            "matching_sha256": freeze["matching_sha256"],
            "comparisons": members,
        },
    )
    output_digests["exclusions.json"] = write_json(
        out / "exclusions.json",
        {
            "matching_id": matching.MATCHING_ID,
            "matching_sha256": freeze["matching_sha256"],
            "row_fields": list(matching.EXCLUSION_FIELDS),
            "rows": exclusions,
        },
    )
    output_digests["exclusions.csv"] = write_csv(out / "exclusions.csv", exclusions, matching.EXCLUSION_FIELDS)
    output_digests["controls.json"] = write_json(
        out / "controls.json",
        {
            "matching_id": matching.MATCHING_ID,
            "matching_sha256": freeze["matching_sha256"],
            "histories": control_inventory,
            "rejected_elevated_periods": rejected_elevated,
        },
    )
    output_digests["summary.json"] = write_json(
        out / "summary.json",
        {
            "matching_id": matching.MATCHING_ID,
            "matching_sha256": freeze["matching_sha256"],
            "n_comparisons": len(comparisons),
            "n_primary": len(primary_rows),
            "overall": overall,
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
            "baseline_audit_sha256": baseline.audit_sha256(),
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
        "matching_id": matching.MATCHING_ID,
        "matching_sha256": freeze["matching_sha256"],
        "upstream_baseline_audit": freeze_spec["upstream_baseline_audit"],
        "upstream_study": freeze_spec["upstream_study"],
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
        "horizons": list(matching.HORIZONS),
        "deferred_horizons": ["day", "month"],
        "modes": list(matching.MODES),
        "calendar": primary.describe(),
        "label_blind_construction": (
            "study manifest rows are structural only; matching regions reuse "
            "M5 period identities; common strata and capped-n retention are "
            "hash-ordered over observable robot/program cells; labels measure "
            "retained burden afterwards, never select retained files"
        ),
        "label_blind_guard": "assert_matching_construction_label_blind passed",
        "provenance_note": (
            "prior corpus distribution knowledge (M1/M2/M3/M4/M5 results such as "
            "zero clean controls and the M5 LIMITED weekly verdict) is exploratory "
            "prior knowledge used to motivate this matching audit; no embedding "
            "movement output was inspected to choose regions, and every matching "
            "count below was computed only after the matching freeze above"
        ),
        "overall": overall,
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
            "metadata only: common-support, attrition and burden-contrast counts; never calibrated specificity",
            "thresholds are frozen research assumptions, not owner-approved operational cutoffs",
            "no annotated-clean baseline exists; no specificity or deployable claim follows",
            "disjoint rows mean no shared file and no shared period, never statistical independence",
            "within-file burden has no accepted annotation field and is unavailable",
            "four histories at n=4 make every cross-history statement descriptive",
            "the daily horizon is deferred and the monthly horizon stays UNSUPPORTED per M5; no claim is made for either",
        ],
    }
    output_digests["run.json"] = write_json(out / "run.json", run_record)

    print(
        json.dumps(
            {
                "matching_sha256": freeze["matching_sha256"],
                "n_comparisons": len(comparisons),
                "n_primary": len(primary_rows),
                "overall": overall["disposition"],
                "out": str(out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
