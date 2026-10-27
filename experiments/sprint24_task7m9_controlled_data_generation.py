"""Sprint 24 Task 7-M9 official controlled-corpus generation CLI.

Executes the frozen M8 v2 plan (``S24-CONTROLLED-DATA-v2``) into a fresh,
separate supplemental corpus: six histories, 8,280 files, 60 planned
weekly/monthly pairs. Real ``generate_planned_file`` synthesis on every
planned slot/seed with zero runtime fallback/reseed/threshold rescue;
``REFUSED-GENERATION`` fails fast per-slot. The exact execution freeze is
persisted BEFORE any waveform synthesis and re-checked for digest/identity
drift before audit writes the corpus manifest.
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
from typing import Any, cast

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as _np  # noqa: E402

from representation import sprint24_controlled_data_execution as execution  # noqa: E402
from representation import sprint24_controlled_data_protocol as v1  # noqa: E402
from representation.sprint24_context_audit import canonical_json  # noqa: E402
from synth.dataset import _save_sample, load_sample  # noqa: E402


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


def freeze_plans(expected_execution_sha256: str | None) -> dict[str, object]:
    """Plan all six histories (metadata only) and pin the execution freeze."""
    if expected_execution_sha256 and execution.execution_freeze()["protocol_sha256"] != expected_execution_sha256:
        raise SystemExit(
            "frozen execution digest mismatch: "
            f"{execution.execution_freeze()['protocol_sha256']} != {expected_execution_sha256}"
        )
    execution.assert_execution_construction_label_blind()
    freeze = execution.execution_freeze()
    if expected_execution_sha256 is None and str(freeze["protocol_sha256"]) != (
        "91d4ce264c73f0dada20b4a5f1a7c460e5809bdb1daf67cc75823b9a1d98a3bd"
    ):
        raise SystemExit(
            "frozen execution digest drift: "
            f"{freeze['protocol_sha256']} != 91d4ce264c73f0dada20b4a5f1a7c460e5809bdb1daf67cc75823b9a1d98a3bd"
        )
    used_seeds: list[int] = []
    plans: list[dict[str, object]] = []
    for history_id, seed in zip(v1.NEW_HISTORY_IDS, v1.PROPOSED_DATA_SEEDS):
        plan = execution.plan_history(str(history_id), int(seed), list(used_seeds))
        used_seeds.append(int(seed))
        plans.append(plan)
    return {"freeze": freeze, "plans": plans}


def generate_corpus(out: Path, dry: dict[str, object]) -> dict[str, object]:
    """Synthesize every planned slot through the real M8 generation path."""
    execution.require_fresh_dir(out, purpose="official controlled corpus")
    freeze = cast("dict[str, object]", dry["freeze"])
    plans = cast("list[dict[str, object]]", dry["plans"])
    if str(freeze["protocol_sha256"]) != (
        "91d4ce264c73f0dada20b4a5f1a7c460e5809bdb1daf67cc75823b9a1d98a3bd"
    ):
        raise SystemExit(f"REFUSED-IDENTITY: execution freeze drifted: {freeze['protocol_sha256']}")
    write_json(out / "execution-freeze.json", freeze)
    generator = execution.fresh_generator()
    waves_dir = out / "waves"
    waves_dir.mkdir(parents=True, exist_ok=False)
    manifest_rows: list[dict[str, Any]] = []
    refused_rows: list[dict[str, str]] = []
    realized_slots: list[dict[str, object]] = []
    t0 = time.time()
    slot_total = 0
    for plan in plans:
        history_id = str(plan["history_id"])
        slots = cast("list[dict[str, object]]", plan["slots"])
        history_dir = waves_dir / history_id
        history_dir.mkdir(parents=True, exist_ok=False)
        for slot in slots:
            slot_total += 1
            try:
                sample = execution.generate_planned_file(slot, generator)
            except ValueError as exc:
                refused_rows.append({"slot_id": str(slot["slot_id"]), "reason": str(exc)})
                continue
            annotation = execution.trace_annotation(sample, slot)
            target = history_dir / f"{sample.file_id}.npz"
            if target.exists():
                raise SystemExit(f"REFUSED-OVERWRITE: corpus target {target} exists")
            _save_sample(sample, target)
            manifest_rows.append(
                {
                    "slot_id": str(slot["slot_id"]),
                    "history_id": history_id,
                    "week_offset": slot["week_offset"],
                    "week_band": slot["week_band"],
                    "month_index": slot["month_index"],
                    "month_band": slot["month_band"],
                    "arm": slot["arm"],
                    "file_id": sample.file_id,
                    "file_label": sample.file_label.value,
                    "requested_label": slot["requested_label"],
                    "requested_family": slot["requested_family"],
                    "requested_family_original": slot.get("requested_family_original"),
                    "reassigned": bool(slot.get("reassigned", False)),
                    "reassignment_reason": slot.get("reassignment_reason"),
                    "requested_severity": slot["requested_severity"],
                    "seed": int(sample.seed),
                    "end_time_rel_s": slot["end_time_rel_s"],
                    "predicted_robot_idx": slot["predicted_robot_idx"],
                    "predicted_program_idx": slot["predicted_program_idx"],
                    "anomaly_mask_present": sample.anomaly_mask is not None,
                    "annotation": annotation,
                    "relpath": f"waves/{history_id}/{target.name}",
                    "waveform_sha256": sha256_path(target),
                    "n_bytes": int(target.stat().st_size),
                }
            )
            realized_slots.append(slot)
            check = load_sample(target)
            check.validate()
    generate_seconds = time.time() - t0
    manifest: dict[str, Any] = {
        "protocol_id": execution.PROTOCOL_ID,
        "base_protocol_id": execution.BASE_PROTOCOL_ID,
        "base_protocol_sha256": execution.BASE_PROTOCOL_SHA256,
        "execution_sha256": str(freeze["protocol_sha256"]),
        "planned_slots": slot_total,
        "realized_files": len(manifest_rows),
        "refused_slots": refused_rows,
        "n_refused_slots": len(refused_rows),
        "planned_pairs": execution.default_planned_pairs(),
        "files": manifest_rows,
    }
    manifest_digest = write_json(out / "corpus-manifest.json", manifest)
    realized_payload = write_json(out / "realized-slots.json", realized_rows_payload(realized_slots))
    return {
        "manifest_digest": manifest_digest,
        "realized_digest": realized_payload,
        "n_planned": slot_total,
        "n_realized": len(manifest_rows),
        "n_refused": len(refused_rows),
        "generate_seconds": generate_seconds,
    }


def realized_rows_payload(slots: list[dict[str, object]]) -> dict[str, object]:
    rows = [
        {
            "slot_id": str(slot["slot_id"]),
            "history_id": str(slot["history_id"]),
            "week_offset": slot["week_offset"],
            "month_index": slot["month_index"],
            "arm": slot["arm"],
            "requested_label": slot["requested_label"],
            "requested_family": slot["requested_family"],
            "seed": slot["seed"],
            "end_time_rel_s": slot["end_time_rel_s"],
        }
        for slot in slots
    ]
    return {"n_slots": len(rows), "slots": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--expected-execution-sha256",
        default="91d4ce264c73f0dada20b4a5f1a7c460e5809bdb1daf67cc75823b9a1d98a3bd",
    )
    args = parser.parse_args()

    out = Path(args.out)
    if out.is_dir() and any(out.iterdir()):
        raise SystemExit(f"REFUSED-OVERWRITE: output {out} already holds files")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    dry = freeze_plans(args.expected_execution_sha256)
    freeze = cast("dict[str, object]", dry["freeze"])
    gen = generate_corpus(out, dry)
    run_record = {
        "protocol_id": execution.PROTOCOL_ID,
        "base_protocol_id": execution.BASE_PROTOCOL_ID,
        "base_protocol_sha256": execution.BASE_PROTOCOL_SHA256,
        "execution_sha256": str(freeze["protocol_sha256"]),
        "execution_freeze_before_synthesis": True,
        "source_commit": source_commit(),
        "planned_histories": len(cast("list[object]", dry["plans"])),
        "planned_slots": int(cast("int", gen["n_planned"])),
        "realized_files": int(cast("int", gen["n_realized"])),
        "refused_slots": int(cast("int", gen["n_refused"])),
        "planned_pairs_per_history_per_horizon": 5,
        "forward_gate": "OFFICIAL-CORPUS-NO-FORWARDS",
        "forwards": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "embedding_scores_read": 0,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "timing": {"run_seconds": time.time() - started, "generate_seconds": gen["generate_seconds"]},
        "limitations": [
            "data validation only: no embedding, no forward, no inference outcome exists",
            "no model-performance claim follows from data validation",
        ],
    }
    write_json(out / "run.json", run_record)

    print(
        json.dumps(
            {
                "execution_sha256": str(freeze["protocol_sha256"]),
                "n_planned": int(cast("int", gen["n_planned"])),
                "n_realized": int(cast("int", gen["n_realized"])),
                "n_refused": int(cast("int", gen["n_refused"])),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
