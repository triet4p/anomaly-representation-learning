"""Sprint 24 Task 7-M8 controlled-data readiness CLI (execution freeze + smoke).

Two modes through one entrypoint and one generation path:

- ``dryplan``: deterministic full 6-history planned-metadata readiness only
  (slot plan, calendars, nesting, burden bands, roster support, lineage,
  quotas, cost). No waveform, no forward, no model contact.
- ``readiness``: dryplan first (machine-checkable execution freeze pinned to
  disk), then a tiny clearly-marked throwaway waveform smoke through the SAME
  slot-plan and generation path. The freeze record precedes any synthesis;
  smoke output carries the throwaway marker and is never an official corpus.

Refusal semantics: any planned shortfall fails fast with a closed reason; the
smoke shortfall or a non-empty output directory refuses without overwrite,
resynthesis, reseeding or rescue.
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

from representation import sprint24_controlled_data_execution as execution  # noqa: E402
from representation import sprint24_controlled_data_protocol as v1  # noqa: E402
from representation.sprint24_context_audit import canonical_json  # noqa: E402
from synth.schema import MODEL_INPUT_FIELD_NAMES  # noqa: E402
from typing import Any, cast  # noqa: E402

import numpy as _np  # noqa: E402

SMOKE_MAX_FILES = 96
SMOKE_WEEKS = (5, 13)
SMOKE_PER_WEEK = 8


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


def _drop_refused_slots(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the candidates the real generator accepts on the planned seeds.

    Single-shot probe per slot, no retry/reseed/fallback: a legitimate
    strength-gate rejection excludes the slot from the throwaway smoke and is
    reported in the manifest as ``refused_smoke_slots`` with reason
    ``REFUSED-GENERATION``. The planned slot itself is unchanged.
    """
    from representation.sprint24_controlled_data_execution import REFUSED_GENERATION

    generator = execution.fresh_generator()
    kept: list[dict[str, object]] = []
    for slot in candidates:
        if str(slot["requested_label"]) != "abnormal":
            kept.append(slot)
            continue
        try:
            execution.generate_planned_file(slot, generator)
        except ValueError as error:
            if REFUSED_GENERATION not in str(error):
                raise
            continue
        kept.append(slot)
    return kept


def _smoke_slots(plan: dict[str, object]) -> list[dict[str, object]]:
    plan_slots = cast("list[dict[str, object]]", plan["slots"])
    slots = [slot for slot in plan_slots if slot["week_offset"] in SMOKE_WEEKS]
    picked: list[dict[str, object]] = []
    for offset in SMOKE_WEEKS:
        week_slots = sorted(
            (slot for slot in slots if slot["week_offset"] == offset),
            key=lambda slot: str(slot["slot_id"]),
        )
        abnormal = [slot for slot in week_slots if str(slot["requested_label"]) == "abnormal"]
        normal = [slot for slot in week_slots if str(slot["requested_label"]) == "normal"]
        take_abnormal = min(len(abnormal), SMOKE_PER_WEEK // 2)
        take_normal = min(len(normal), SMOKE_PER_WEEK - take_abnormal)
        candidates = abnormal[:take_abnormal] + normal[:take_normal]
        picked.extend(_drop_refused_slots(candidates))
    picked = picked[:SMOKE_MAX_FILES]
    assert len(picked) <= SMOKE_MAX_FILES
    return picked


def _smoke_refused(plan: dict[str, object], picked: list[dict[str, object]]) -> list[dict[str, object]]:
    """Report smoke candidates the real generator refuses on the planned seeds.

    Compares the deterministic candidate set against the single-shot probe in
    ``_drop_refused_slots`` and records each excluded slot with reason
    ``REFUSED-GENERATION``. Exclusion is deterministic and changes no plan.
    """
    from representation.sprint24_controlled_data_execution import REFUSED_GENERATION

    picked_ids = {str(slot["slot_id"]) for slot in picked}
    return [
        {
            "slot_id": str(slot["slot_id"]),
            "requested_family": slot["requested_family"],
            "requested_severity": slot["requested_severity"],
            "reason": REFUSED_GENERATION,
        }
        for slot in _smoke_candidates(plan)
        if str(slot["slot_id"]) not in picked_ids
    ]


def _smoke_candidates(plan: dict[str, object]) -> list[dict[str, object]]:
    """Return the deterministic pre-probe smoke candidate set for one plan."""
    plan_slots = cast("list[dict[str, object]]", plan["slots"])
    slots = [slot for slot in plan_slots if slot["week_offset"] in SMOKE_WEEKS]
    candidates: list[dict[str, object]] = []
    for offset in SMOKE_WEEKS:
        week_slots = sorted(
            (slot for slot in slots if slot["week_offset"] == offset),
            key=lambda slot: str(slot["slot_id"]),
        )
        abnormal = [slot for slot in week_slots if str(slot["requested_label"]) == "abnormal"]
        normal = [slot for slot in week_slots if str(slot["requested_label"]) == "normal"]
        take_abnormal = min(len(abnormal), SMOKE_PER_WEEK // 2)
        take_normal = min(len(normal), SMOKE_PER_WEEK - take_abnormal)
        candidates.extend(abnormal[:take_abnormal])
        candidates.extend(normal[:take_normal])
    return candidates[:SMOKE_MAX_FILES]



def dryplan(out: Path, expected_base_sha256: str | None) -> dict[str, object]:
    """Run the full 6-history planned-metadata readiness pass."""
    if expected_base_sha256 and execution.BASE_PROTOCOL_SHA256 != expected_base_sha256:
        raise SystemExit(
            f"frozen base design digest mismatch: {execution.BASE_PROTOCOL_SHA256} != {expected_base_sha256}"
        )
    execution.assert_execution_construction_label_blind()
    freeze = execution.execution_freeze()
    output_digests: dict[str, str] = {}
    output_digests["execution-freeze.json"] = write_json(out / "execution-freeze.json", freeze)

    used_seeds: list[int] = []
    histories: list[dict[str, object]] = []
    all_slots: list[dict[str, object]] = []
    seeds: tuple[int, ...] = v1.PROPOSED_DATA_SEEDS
    for history_id, seed in zip(v1.NEW_HISTORY_IDS, seeds):
        plan = execution.plan_history(str(history_id), int(seed), list(used_seeds))
        used_seeds.append(int(seed))
        plan_slots = cast("list[dict[str, object]]", plan["slots"])
        slot_rows = [
            {
                key: slot[key]
                for key in (
                    "slot_id", "history_id", "week_offset", "week_band", "month_index",
                    "month_band", "arm", "requested_label", "requested_family",
                    "requested_family_original", "reassigned", "reassignment_reason",
                    "requested_severity", "seed", "predicted_robot_idx",
                    "predicted_program_idx", "end_time_rel_s", "stream",
                )
            }
            for slot in plan_slots
        ]
        histories.append(
            {
                "history_id": history_id,
                "proposed_seed": int(seed),
                "n_slots": len(slot_rows),
                "n_abnormal_requested": sum(1 for row in slot_rows if str(row["requested_label"]) == "abnormal"),
                "plan_checks": cast("list[dict[str, object]]", plan["checks"]),
            }
        )
        all_slots.extend(slot_rows)
    slots_payload = {
        "protocol_id": execution.PROTOCOL_ID,
        "execution_sha256": str(freeze["protocol_sha256"]),
        "n_histories": len(histories),
        "n_slots": len(all_slots),
        "histories": histories,
        "planned_pairs": execution.default_planned_pairs(),
    }
    output_digests["planned-slots.json"] = write_json(out / "planned-slots.json", slots_payload)
    estimate = execution.cost_estimate()
    gate = execution.forward_gate()
    output_digests["cost.json"] = write_json(
        out / "cost.json", {"protocol_id": execution.PROTOCOL_ID, "estimate": estimate, "gate": gate}
    )
    return {
        "freeze": freeze,
        "histories": histories,
        "estimate": estimate,
        "gate": gate,
        "output_digests": output_digests,
        "n_slots": len(all_slots),
    }


def smoke(out: Path, started: float, dry: dict[str, object]) -> dict[str, object]:
    """Generate the tiny throwaway smoke subset after the freeze is pinned."""
    smoke_dir = out / "smoke"
    execution.require_fresh_dir(smoke_dir, purpose="smoke waveform subset")
    generator = execution.fresh_generator()
    used_seeds: list[int] = []
    waveforms: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    refused_smoke: list[dict[str, object]] = []
    t0 = time.time()
    for history_id, seed in zip(v1.NEW_HISTORY_IDS, v1.PROPOSED_DATA_SEEDS):
        plan = execution.plan_history(str(history_id), int(cast("int", seed)), list(used_seeds))
        used_seeds.append(int(cast("int", seed)))
        picked = _smoke_slots(plan)
        refused_smoke.extend(_smoke_refused(plan, picked))
        for slot in picked:
            sample = execution.generate_planned_file(slot, generator)
            annotation = execution.trace_annotation(sample, slot)
            payload: dict[str, Any] = {
                "slot_id": str(slot["slot_id"]),
                "history_id": history_id,
                "week_offset": slot["week_offset"],
                "week_band": slot["week_band"],
                "month_index": slot["month_index"],
                "month_band": slot["month_band"],
                "arm": slot["arm"],
                "throwaway_marker": execution.SMOKE_STREAM_TAG,
                "not_official_corpus": True,
                "file_id": sample.file_id,
                "seed": int(sample.seed),
                "file_label": sample.file_label.value,
                "requested_label": slot["requested_label"],
                "requested_family": slot["requested_family"],
                "requested_family_original": slot.get("requested_family_original"),
                "reassigned": slot.get("reassigned", False),
                "reassignment_reason": slot.get("reassignment_reason"),
                "requested_severity": slot["requested_severity"],
                "end_time_rel_s": slot["end_time_rel_s"],
                "time_shape": [int(sample.C), int(sample.T)],
                "annotation": annotation,
                "encoder_input_keys": sorted(MODEL_INPUT_FIELD_NAMES),
                "factory_provenance": {
                    "seed": sample.factory_provenance.seed if sample.factory_provenance else None,
                    "stream": sample.factory_provenance.stream if sample.factory_provenance else None,
                    "generator_version": sample.generator_version,
                    "config_hash": sample.config_hash,
                },
            }
            wave_path = smoke_dir / f"{sample.file_id}.npz"
            if wave_path.exists():
                raise SystemExit(f"REFUSED-OVERWRITE: smoke target {wave_path} exists")
            _np.savez_compressed(
                wave_path,
                x=sample.x.astype("float32", copy=False),
                mask=(sample.anomaly_mask.astype("uint8") if sample.anomaly_mask is not None
                      else _np.zeros_like(sample.x, dtype="uint8")),
            )
            payload["waveform_relpath"] = f"smoke/{wave_path.name}"  # payload typed dict[str, object]
            payload["waveform_sha256"] = sha256_path(wave_path)
            manifest_rows.append(payload)
            waveforms.append({"slot_id": str(slot["slot_id"]), "file_id": sample.file_id})
    smoke_seconds = time.time() - t0
    manifest: dict[str, Any] = {
        "protocol_id": execution.PROTOCOL_ID,
        "execution_sha256": str(cast("dict[str, object]", dry["freeze"])["protocol_sha256"]),
        "marker": execution.SMOKE_STREAM_TAG,
        "not_official_corpus": True,
        "n_files": len(manifest_rows),
        "files": manifest_rows,
        "refused_smoke_slots": refused_smoke,
        "n_refused_smoke_slots": len(refused_smoke),
    }
    manifest_digest = write_json(out / "smoke-manifest.json", manifest)
    cast("dict[str, str]", dry["output_digests"])["smoke-manifest.json"] = manifest_digest
    return {
        "smoke_dir": str(smoke_dir),
        "n_files": len(manifest_rows),
        "n_refused": len(refused_smoke),
        "smoke_seconds": smoke_seconds,
        "total_seconds": time.time() - started,
        "waveforms": waveforms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("dryplan", "readiness"), default="readiness")
    parser.add_argument("--expected-base-protocol-sha256", default=None)
    parser.add_argument(
        "--expected-execution-sha256",
        default=None,
        help="claim-check of the v2 execution freeze digest before any synthesis",
    )
    args = parser.parse_args()

    out = Path(args.out)
    if out.is_dir() and any(out.iterdir()):
        raise SystemExit(f"REFUSED-OVERWRITE: output {out} already holds files")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    dry = dryplan(out, args.expected_base_protocol_sha256)
    assert isinstance(dry["freeze"], dict)
    if args.expected_execution_sha256 and dry["freeze"]["protocol_sha256"] != args.expected_execution_sha256:
        raise SystemExit(
            "frozen execution digest mismatch: "
            f"{dry['freeze']['protocol_sha256']} != {args.expected_execution_sha256}"
        )

    smoke_result: dict[str, object] | None = None
    if args.mode == "readiness":
        smoke_result = smoke(out, started, dry)
    dry_histories = cast("list[dict[str, object]]", dry["histories"])
    dry_gate = cast("dict[str, object]", dry["gate"])
    dry_freeze = cast("dict[str, object]", dry["freeze"])

    run_record = {
        "protocol_id": execution.PROTOCOL_ID,
        "base_protocol_id": execution.BASE_PROTOCOL_ID,
        "base_protocol_sha256": execution.BASE_PROTOCOL_SHA256,
        "execution_sha256": str(dry_freeze["protocol_sha256"]),
        "mode": args.mode,
        "execution_freeze_before_synthesis": True,
        "source_freeze": "candidate source frozen only at ready_for_review; execution freeze above pins planned metadata, not source bytes",
        "source_commit": source_commit(),
        "planned_histories": len(dry_histories),  # typed local
        "planned_slots": int(cast("int", dry["n_slots"])),
        "planned_pairs_per_history_per_horizon": 5,
        "forward_gate": str(dry_gate["gate"]),
        "forwards": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "embedding_scores_read": 0,
        "waveforms_written_throwaway_smoke": (smoke_result or {}).get("n_files", 0),
        "refused_throwaway_smoke_slots": (smoke_result or {}).get("n_refused", 0),
        "official_corpus_files": 0,
        "label_blind_guard": "assert_execution_construction_label_blind passed",
        "provenance_note": (
            "prior M4/M5/M6 outcomes are exploratory motivation only; no DEV-04 or old "
            "movement output selected any planned value; every planned identity above "
            "was frozen before any generation and no official corpus exists"
        ),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "timing": {
            "run_seconds": time.time() - started,
            "smoke_seconds": (smoke_result or {}).get("smoke_seconds", 0.0),
        },
        "outputs": cast("dict[str, str]", dry["output_digests"]),
        "limitations": [
            "readiness only: no official corpus, no embedding, no forward, no inference outcome exists",
            "planned inference (24840 forwards incl. 96-file smoke envelope) is an estimate under owner budget authority; execution needs the separate release",
            "no specificity, deployability or business performance claim follows",
        ],
    }
    cast("dict[str, str]", dry["output_digests"])["run.json"] = write_json(out / "run.json", run_record)

    print(
        json.dumps(
            {
                "execution_sha256": str(dry_freeze["protocol_sha256"]),
                "base_protocol_sha256": execution.BASE_PROTOCOL_SHA256,
                "n_histories": len(dry_histories),
                "n_slots": int(cast("int", dry["n_slots"])),
                "forward_gate": str(dry_gate["gate"]),
                "smoke_files": (smoke_result or {}).get("n_files", 0),
                "smoke_refused": (smoke_result or {}).get("n_refused", 0),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
