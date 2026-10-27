"""Sprint 24 Task 7-M9 official corpus validation/audit CLI.

Reloads every persisted waveform and annotation in
``data/generated/sprint24_controlled_v2`` through the supported
``synth.dataset.load_sample`` interface, then accounts planned versus
realized files/refusals and validates finite six-channel arrays, checksums,
mask-exact duration/severity/channel provenance, calendar nesting,
baseline/elevated burden and stable baseline controls, observable
context/exposure matching, causal reference separation, disjoint identities
and realized support for all 60 planned pairs.

Observed corpus support is reported separately from planned support (and from
statistical independence, which pair counts never imply). Unsupported cells
are reported honestly without changing scientific criteria. This audit reads
persisted waveforms but executes zero forwards, zero model/embedding reads,
zero refit and zero training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, cast

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as _np  # noqa: E402

from representation import sprint24_controlled_data_execution as execution  # noqa: E402
from representation import sprint24_controlled_data_protocol as v1  # noqa: E402
from representation.sprint24_context_audit import canonical_json  # noqa: E402
from representation.sprint24_weekly_matched_comparison import balance_comparison  # noqa: E402
from synth.dataset import load_sample  # noqa: E402

EXPECTED_EXECUTION_SHA256 = "91d4ce264c73f0dada20b4a5f1a7c460e5809bdb1daf67cc75823b9a1d98a3bd"

WEEK_ARM_WEEKS: dict[str, list[int]] = {
    "BASELINE-A": [5, 6, 7],
    "ELEVATED-A": [13, 14],
    "BASELINE-B": [18, 19, 20],
    "ELEVATED-B": [26, 27],
}

WEEK_COMPARISONS: list[tuple[str, str, str]] = [
    ("BASELINE-A", "ELEVATED-A", "TRANSITION"),
    ("BASELINE-B", "ELEVATED-A", "TRANSITION"),
    ("BASELINE-A", "ELEVATED-B", "TRANSITION"),
    ("BASELINE-B", "ELEVATED-B", "TRANSITION"),
    ("BASELINE-A", "BASELINE-B", "CONTROL"),
]

MONTH_RUN_MONTHS: dict[str, list[int]] = {"RUN-A": [2, 3], "RUN-B": [5, 6]}

MONTH_COMPARISONS: list[tuple[str, str, str]] = [
    ("RUN-A", "ELEVATED-M4", "TRANSITION"),
    ("RUN-B", "ELEVATED-M4", "TRANSITION"),
    ("RUN-A", "ELEVATED-M7", "TRANSITION"),
    ("RUN-B", "ELEVATED-M7", "TRANSITION"),
    ("RUN-A", "RUN-B", "CONTROL"),
]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> str:
    text = canonical_json(payload) + "\n"
    path.write_text(text, encoding="utf-8", newline="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def month_end_span(months: list[dict[str, object]], indices: list[int]) -> tuple[float, float]:
    rows = {int(cast("int", row["index"])): row for row in months}
    return (
        min(float(cast("float", rows[i]["start_rel_s"])) for i in indices),
        max(float(cast("float", rows[i]["end_rel_s"])) for i in indices),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-execution-sha256", default=EXPECTED_EXECUTION_SHA256)
    args = parser.parse_args()

    corpus = Path(args.corpus)
    out = Path(args.out)
    if out.is_dir() and any(out.iterdir()):
        raise SystemExit(f"REFUSED-OVERWRITE: output {out} already holds files")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    execution.assert_execution_construction_label_blind()
    freeze = execution.execution_freeze()
    if str(freeze["protocol_sha256"]) != str(args.expected_execution_sha256):
        raise SystemExit(
            "frozen execution digest mismatch: "
            f"{freeze['protocol_sha256']} != {args.expected_execution_sha256}"
        )
    persisted = json.loads((corpus / "execution-freeze.json").read_text(encoding="utf-8"))
    if persisted.get("protocol_sha256") != str(freeze["protocol_sha256"]):
        raise SystemExit(
            f"REFUSED-IDENTITY: persisted freeze {persisted.get('protocol_sha256')} != "
            f"frozen {freeze['protocol_sha256']}"
        )
    manifest = json.loads((corpus / "corpus-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("execution_sha256") != str(freeze["protocol_sha256"]):
        raise SystemExit("REFUSED-IDENTITY: corpus manifest execution digest drifted")
    planned_slots_total = int(manifest.get("planned_slots", 0))
    refused_planned = list(cast("list[dict[str, object]]", manifest.get("refused_slots", [])))
    rows = list(cast("list[dict[str, object]]", manifest.get("files", [])))

    used_seeds: list[int] = []
    slot_by_id: dict[str, dict[str, object]] = {}
    history_slots: dict[str, list[dict[str, object]]] = {}
    for history_id, seed in zip(v1.NEW_HISTORY_IDS, v1.PROPOSED_DATA_SEEDS):
        plan = execution.plan_history(str(history_id), int(seed), list(used_seeds))
        used_seeds.append(int(seed))
        slots = cast("list[dict[str, object]]", plan["slots"])
        history_slots[str(history_id)] = slots
        for slot in slots:
            slot_by_id[str(slot["slot_id"])] = slot

    npz_paths = sorted((corpus / "waves").rglob("*.npz"))
    issues: list[str] = []
    seen_file_ids: set[str] = set()
    seen_slot_ids: set[str] = set()
    realized_by_slot: dict[str, dict[str, object]] = {}
    reloaded_file_ids: list[str] = []
    channel_counts: set[int] = set()
    total_bytes = 0
    t_values: list[int] = []
    realized_abnormal = 0
    for path in npz_paths:
        sample = load_sample(path)
        sample.validate()
        rel = path.relative_to(corpus).as_posix()
        row = next((r for r in rows if str(r["relpath"]) == rel), None)
        if row is None:
            issues.append(f"unmanifested waveform: {rel}")
            continue
        if sample.file_id != str(row["file_id"]):
            issues.append(f"{rel}: reloaded file_id {sample.file_id} != manifest {row['file_id']}")
        slot = slot_by_id.get(str(row["slot_id"]))
        if slot is None:
            issues.append(f"{rel}: slot {row['slot_id']} not in frozen plan")
            continue
        if int(cast("int", slot["seed"])) != int(sample.seed):
            issues.append(f"{rel}: reloaded seed {sample.seed} != planned {slot['seed']}")
        declared = str(sha256_path(path))
        if declared != str(row["waveform_sha256"]):
            issues.append(f"{rel}: checksum drift {declared} != {row['waveform_sha256']}")
        if not bool(_np.isfinite(sample.x).all()):
            issues.append(f"{rel}: non-finite waveform array")
        if sample.x.dtype != _np.float32:
            issues.append(f"{rel}: waveform dtype {sample.x.dtype} != float32")
        if sample.x.ndim != 2 or int(sample.x.shape[0]) != int(v1.N_CHANNELS):
            issues.append(f"{rel}: waveform shape {sample.x.shape} is not [{v1.N_CHANNELS}, T]")
        channel_counts.add(int(sample.x.shape[0]))
        t_values.append(int(sample.x.shape[1]))
        total_bytes += int(path.stat().st_size)
        annotation = execution.trace_annotation(sample, slot)
        stored = cast("dict[str, object]", row["annotation"])
        for key in ("family", "severity", "start_step", "end_step", "affected_channels",
                    "duration_steps", "abnormal_steps", "file_label", "requested_label",
                    "requested_family", "requested_severity"):
            if annotation.get(key) != stored.get(key):
                issues.append(f"{rel}: annotation field {key} drifted: {annotation.get(key)} != {stored.get(key)}")
        if str(annotation.get("file_label")) != str(row["requested_label"]):
            issues.append(f"{rel}: reloaded label {annotation.get('file_label')} != planned {row['requested_label']}")
        if sample.file_id in seen_file_ids:
            issues.append(f"duplicate file_id: {sample.file_id}")
        seen_file_ids.add(sample.file_id)
        if str(row["slot_id"]) in seen_slot_ids:
            issues.append(f"duplicate slot realization: {row['slot_id']}")
        seen_slot_ids.add(str(row["slot_id"]))
        reloaded_file_ids.append(sample.file_id)
        if str(annotation.get("file_label")) == "abnormal":
            realized_abnormal += 1
        realized_by_slot[str(row["slot_id"])] = {
            "file_id": sample.file_id,
            "seed": int(sample.seed),
            "file_label": str(annotation.get("file_label")),
            "family": annotation.get("family"),
            "severity": annotation.get("severity"),
            "start_step": annotation.get("start_step"),
            "end_step": annotation.get("end_step"),
            "affected_channels": annotation.get("affected_channels"),
            "abnormal_steps": annotation.get("abnormal_steps"),
            "duration_steps": annotation.get("duration_steps"),
            "abnormal_step_fraction": annotation.get("abnormal_step_fraction"),
            "waveform_steps": int(sample.x.shape[1]),
            "end_time_rel_s": slot["end_time_rel_s"],
            "week_offset": slot["week_offset"],
            "month_index": slot["month_index"],
            "arm": slot["arm"],
            "robot_idx": int(sample.robot_idx),
            "program_idx": int(sample.program_idx),
            "cell": f"{sample.robot_code}|{sample.program_number}",
        }

    planned_slot_ids = set(slot_by_id)
    missing_realizations = sorted(planned_slot_ids - seen_slot_ids)
    refused_ids = {str(r.get("slot_id")) for r in refused_planned}
    unexpected_missing = [s for s in missing_realizations if s not in refused_ids]
    if unexpected_missing:
        issues.append(f"{len(unexpected_missing)} planned slots missing without refusal, e.g. {unexpected_missing[0]}")

    audits: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    pair_exclusions: list[list[dict[str, object]]] = []
    for history_id in v1.NEW_HISTORY_IDS:
        hid = str(history_id)
        realized = {sid: r for sid, r in realized_by_slot.items() if sid.startswith(hid + "|")}
        planned = history_slots[hid]
        abnormal_ids = {sid for sid, r in realized.items() if str(r["file_label"]) == "abnormal"}
        week_member_ids: dict[int, list[str]] = {}
        month_member_ids: dict[int, list[str]] = {}
        for sid, r in realized.items():
            if r["week_offset"] is not None:
                week_member_ids.setdefault(int(cast("int", r["week_offset"])), []).append(str(r["file_id"]))
            if r["month_index"] is not None:
                month_member_ids.setdefault(int(cast("int", r["month_index"])), []).append(str(r["file_id"]))
        arm_weeks: dict[str, list[str]] = {}
        for arm, weeks in WEEK_ARM_WEEKS.items():
            arm_weeks[arm] = [fid for w in weeks for fid in week_member_ids.get(w, [])]
        run_months: dict[str, list[str]] = {}
        for run, months in MONTH_RUN_MONTHS.items():
            run_months[run] = [fid for m in months for fid in month_member_ids.get(m, [])]
        elevated_m4 = [fid for fid in month_member_ids.get(4, [])]
        elevated_m7 = [fid for fid in month_member_ids.get(7, [])]
        file_to_slot = {str(realized[sid]["file_id"]): sid for sid in realized}
        payload_cell_of = {str(realized[sid]["file_id"]): str(realized[sid]["cell"]) for sid in realized}
        payload_abnormal = {str(realized[sid]["file_id"]): (1 if sid in abnormal_ids else 0) for sid in realized}
        t_by_file = {str(realized[sid]["file_id"]): int(cast("int", realized[sid]["waveform_steps"])) for sid in realized}

        def exposure(ids: list[str]) -> float:
            # Observable member file-day exposure: summed overlap of each file's
            # [end_time - T, end_time] span with its owning arm span, in days.
            # File starts are not persisted, so the span is the realized
            # waveform duration (T steps at 1 step/second on the frozen sample
            # clock); the arm span is [min end_time, max end_time].
            # Label-blind: uses only end_time_rel_s and T, never labels.
            from synth.schema import SECONDS_PER_DAY as _SPD
            ends = sorted(float(cast("float", realized[file_to_slot[fid]]["end_time_rel_s"])) for fid in ids)
            if not ends:
                return 0.0
            lo, hi = ends[0], ends[-1]
            total_s = 0.0
            for fid in ids:
                end = float(cast("float", realized[file_to_slot[fid]]["end_time_rel_s"]))
                start = end - float(t_by_file[fid])
                total_s += max(0.0, min(end, hi) - max(start, lo))
            return float(total_s / float(_SPD))

        calendar = execution.build_history_calendar(hid)
        weeks_grid = list(cast("list[dict[str, object]]", calendar["weeks"]))
        months_grid = list(cast("list[dict[str, object]]", calendar["months"]))
        nesting = execution.validate_week_month_nesting(hid, planned, weeks_grid, months_grid, execution.DEFAULT_WEEK_BANDS)
        roster = execution.validate_roster_support(hid, planned)
        audits.append({"history_id": hid, "planned_nesting": nesting, "planned_roster": roster})

        week_bands: dict[int, dict[str, float | int]] = {}
        for offset in (5, 6, 7, 13, 14, 18, 19, 20, 26, 27):
            members = week_member_ids.get(offset, [])
            abn = sum(1 for fid in members if payload_abnormal[fid] == 1)
            week_bands[offset] = {"n": len(members), "abnormal": abn,
                                  "fraction": (abn / len(members)) if members else 0.0}
        month_bands: dict[int, dict[str, float | int]] = {}
        for index in (2, 3, 4, 5, 6, 7):
            members = month_member_ids.get(index, [])
            abn = sum(1 for fid in members if payload_abnormal[fid] == 1)
            month_bands[index] = {"n": len(members), "abnormal": abn,
                                  "fraction": (abn / len(members)) if members else 0.0}

        week_pair_inputs: dict[str, list[str]] = {
            "BASELINE-A": arm_weeks["BASELINE-A"], "BASELINE-B": arm_weeks["BASELINE-B"],
            "ELEVATED-A": arm_weeks["ELEVATED-A"], "ELEVATED-B": arm_weeks["ELEVATED-B"],
        }
        for prev, cur, kind in WEEK_COMPARISONS:
            row, exclusions = balance_comparison(
                hid, "week", "calendar",
                "BASELINE-VS-ELEVATED" if kind == "TRANSITION" else "BASELINE-VS-BASELINE",
                f"{prev}_TO_{cur}",
                [f"{hid}|week|calendar|W{w:02d}" for w in WEEK_ARM_WEEKS[prev]] if prev in WEEK_ARM_WEEKS else [],
                [f"{hid}|week|calendar|W{w:02d}" for w in WEEK_ARM_WEEKS[cur]] if cur in WEEK_ARM_WEEKS else [],
                "OBSERVED-WEEK-ARM", "OBSERVED-WEEK-ARM",
                week_pair_inputs[prev], week_pair_inputs[cur],
                payload_cell_of, payload_cell_of, payload_abnormal, payload_abnormal,
                exposure(week_pair_inputs[prev]), exposure(week_pair_inputs[cur]),
                has_disjoint_control=True,
            )
            pair_rows.append(row)
            pair_exclusions.append(exclusions)
        month_sources: dict[str, list[str]] = {
            "RUN-A": run_months["RUN-A"], "RUN-B": run_months["RUN-B"],
            "ELEVATED-M4": elevated_m4, "ELEVATED-M7": elevated_m7,
        }
        for prev, cur, kind in MONTH_COMPARISONS:
            prev_periods = [f"{hid}|month|calendar|M{m:02d}" for m in MONTH_RUN_MONTHS[prev]] if prev in MONTH_RUN_MONTHS else (
                ["M04"] if cur == "ELEVATED-M4" or prev == "ELEVATED-M4" else ["M07"])
            cur_periods = [f"{hid}|month|calendar|M{m:02d}" for m in ([4] if cur == "ELEVATED-M4" else ([7] if cur == "ELEVATED-M7" else MONTH_RUN_MONTHS[cur]))]
            row, exclusions = balance_comparison(
                hid, "month", "calendar",
                "BASELINE-VS-ELEVATED" if kind == "TRANSITION" else "BASELINE-VS-BASELINE",
                f"{prev}_TO_{cur}",
                prev_periods if prev in MONTH_RUN_MONTHS else [f"{hid}|month|calendar|M04" if prev == "ELEVATED-M4" else f"{hid}|month|calendar|M07"],
                cur_periods,
                "OBSERVED-MONTH-ARM", "OBSERVED-MONTH-ARM",
                month_sources[prev] if prev in month_sources else [],
                month_sources[cur],
                payload_cell_of, payload_cell_of, payload_abnormal, payload_abnormal,
                exposure(month_sources[prev] if prev in month_sources else []), exposure(month_sources[cur]),
                has_disjoint_control=True,
            )
            pair_rows.append(row)
            pair_exclusions.append(exclusions)

    supported = [r for r in pair_rows if str(r.get("disposition")) in ("FEASIBLE", "LIMITED")]
    unsupported = [r for r in pair_rows if str(r.get("disposition")) not in ("FEASIBLE", "LIMITED")]
    audit_record: dict[str, Any] = {
        "protocol_id": execution.PROTOCOL_ID,
        "execution_sha256": str(freeze["protocol_sha256"]),
        "corpus": str(corpus),
        "planned_slots": planned_slots_total,
        "realized_files": len(realized_by_slot),
        "n_refused_planned": len(refused_planned),
        "n_unexpected_missing": len(unexpected_missing),
        "reloaded_files": len(reloaded_file_ids),
        "audit_issues": issues,
        "n_audit_issues": len(issues),
        "realized_abnormal": realized_abnormal,
        "channel_counts": sorted(channel_counts),
        "total_waveform_bytes": total_bytes,
        "n_pairs": len(pair_rows),
        "n_supported_pairs": len(supported),
        "n_unsupported_pairs": len(unsupported),
        "supported_pairs": [
            {"history_id": r["history_id"], "horizon": r["horizon"], "comparison_identity": r["comparison_identity"],
             "disposition": r["disposition"], "n_common_strata": r["n_common_strata"],
             "gap_after": r["burden_contrast_gap_after"]} for r in pair_rows
        ],
        "planned_audits": audits,
        "pair_rows": pair_rows,
        "pair_exclusion_trails": pair_exclusions,
        "t_min": min(t_values) if t_values else 0,
        "t_max": max(t_values) if t_values else 0,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "timing": {"audit_seconds": time.time() - started},
        "limitations": [
            "pair counts are support accounting, not independent samples; pairs sharing an arm inside a history are dependent (NONADJACENT-SHARED-SUPPORT class)",
            "data validation only: no embedding, no forward, no inference outcome exists",
            "no model-performance claim follows from data validation",
        ],
    }
    audit_digest = write_json(out / "corpus-audit.json", audit_record)

    print(
        json.dumps(
            {
                "execution_sha256": str(freeze["protocol_sha256"]),
                "realized_files": len(realized_by_slot),
                "n_refused_planned": len(refused_planned),
                "n_audit_issues": len(issues),
                "n_pairs": len(pair_rows),
                "n_supported": len(supported),
                "audit_digest": audit_digest,
            },
            sort_keys=True,
        )
    )
    return 0 if not issues and not unexpected_missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
