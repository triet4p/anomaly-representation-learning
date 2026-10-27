"""Sprint 24 Task 7-M7 controlled-data protocol design CLI (metadata only).

Freezes the versioned prospective design (``S24-CONTROLLED-DATA-v1``) and
validates the six planned histories, planned identities, quotas, lineage and
forward/storage cost — without synthesizing any waveform, materializing any
manifest, reading any embedding, or executing any forward.

The design-freeze record is written to disk *before* any history is
validated. Boundary probes exercise the discriminating refusal gates on
synthetic in-memory schedules (overlap, shortfall, seed collision, lineage,
identity, cost); they never touch real data.

Example::

    python experiments/sprint24_task7m7_controlled_data_protocol.py \
      --expected-protocol-sha256 <digest> \
      --out artifacts/sprint-24/evidence/s24-7-m7-a01
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

from representation import sprint24_controlled_data_protocol as protocol  # noqa: E402
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


def run_boundary_probes() -> dict[str, dict[str, object]]:
    """Exercise each refusal gate in-memory; every probe must discriminate."""
    probes: dict[str, dict[str, object]] = {}

    negative = protocol.validate_arm_disjointness({1, 2, 3}, {3, 4, 5})
    positive = protocol.validate_arm_disjointness({1, 2, 3}, {5, 6})
    probes["schedule-disjointness"] = {
        "negative_arms": "baseline {1,2,3} vs elevated {3,4,5} share week 3",
        "negative_status": negative["status"],
        "negative_reason": negative["reason"],
        "positive_status": positive["status"],
        "discriminates": negative["status"] == "REFUSED"
        and negative["reason"] == "REFUSED-OVERLAP"
        and positive["status"] == "PASS",
    }

    schedule = protocol.validate_disjoint_support(protocol.DEFAULT_WEEK_BANDS)
    probes["schedule-structure"] = {
        "positive_status": schedule["status"],
        "discriminates": schedule["status"] == "PASS",
    }

    support_checks = protocol.validate_quotas(220, 5, 4, 12)
    support_ok = protocol.validate_quotas(220, 40, 4, 12)
    probes["support-shortfall"] = {
        "negative_plan": "5 weekly files against floor 30",
        "negative_refused": [c["reason"] for c in support_checks if c["status"] == "REFUSED"],
        "positive_all_pass": all(c["status"] == "PASS" for c in support_ok),
        "discriminates": any(c["status"] == "REFUSED" for c in support_checks)
        and all(c["status"] == "PASS" for c in support_ok),
    }

    lineage_negative = protocol.validate_lineage("S18I-ITER-0005-DEVELOPMENT-01")
    lineage_positive = protocol.validate_lineage("S24C-ITER-0001-CONTROLLED-01")
    seed_negative = protocol.validate_seeds(32064, [42401])
    seed_positive = protocol.validate_seeds(42401, [])
    probes["identity-lineage"] = {
        "old_history_status": lineage_negative["status"],
        "old_history_reason": lineage_negative["reason"],
        "old_seed_status": seed_negative["status"],
        "old_seed_reason": seed_negative["reason"],
        "planned_history_status": lineage_positive["status"],
        "planned_seed_status": seed_positive["status"],
        "discriminates": lineage_negative["status"] == "REFUSED"
        and seed_negative["status"] == "REFUSED"
        and lineage_positive["status"] == "PASS"
        and seed_positive["status"] == "PASS",
    }

    gate = protocol.forward_gate()
    estimate = protocol.cost_estimate()
    assert isinstance(estimate["forwards"], dict)
    probes["cost-refusal"] = {
        "gate": gate["gate"],
        "required_with_smoke": estimate["forwards"]["required_with_smoke"],
        "headroom": estimate["forwards"]["headroom"],
        "shortfall": estimate["forwards"]["shortfall_vs_headroom"],
        "discriminates": gate["gate"] == "INFERENCE-NOT-AUTHORIZED"
        and int(estimate["forwards"]["shortfall_vs_headroom"]) > 0,
    }

    bad_months = {1: "WARMUP-PARTIAL", 2: "LOW", 3: "ELEVATED", 4: "LOW", 5: "LOW", 6: "LOW"}
    monthly_negative = protocol.validate_monthly_bands(bad_months)
    monthly_positive = protocol.validate_monthly_bands(protocol.DEFAULT_MONTH_BANDS)
    probes["monthly-support"] = {
        "negative_status": monthly_negative["status"],
        "negative_reason": monthly_negative["reason"],
        "positive_status": monthly_positive["status"],
        "discriminates": monthly_negative["status"] == "REFUSED"
        and monthly_positive["status"] == "PASS",
    }
    return probes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--expected-protocol-sha256",
        default=None,
        help="claim-check of the frozen design digest before validation",
    )
    parser.add_argument(
        "--protocol-doc",
        default="docs/sprint24-controlled-data-protocol-v1.md",
        help="report path recorded by digest in run.json",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    freeze = protocol.protocol_freeze()
    if args.expected_protocol_sha256 and freeze["protocol_sha256"] != args.expected_protocol_sha256:
        raise SystemExit(
            f"frozen design digest mismatch: {freeze['protocol_sha256']} != {args.expected_protocol_sha256}"
        )
    output_digests: dict[str, str] = {}
    output_digests["design-freeze.json"] = write_json(out / "design-freeze.json", freeze)
    freeze_written = time.time()

    protocol.assert_planning_construction_label_blind()

    used_seeds: list[int] = []
    histories: list[dict[str, object]] = []
    for history_id, seed in zip(protocol.NEW_HISTORY_IDS, protocol.PROPOSED_DATA_SEEDS):
        verdict = protocol.validate_planned_history(
            history_id,
            seed,
            list(used_seeds),
            protocol.DEFAULT_WEEK_BANDS,
            protocol.DEFAULT_MONTH_BANDS,
        )
        used_seeds.append(seed)
        histories.append(
            {
                "history_id": history_id,
                "proposed_seed": seed,
                "week_identities": protocol.planned_week_identities(history_id),
                "month_identities": protocol.planned_month_identities(history_id),
                "reference_identity": protocol.planned_reference_identity(history_id),
                "verdict": verdict["verdict"],
                "refusals": verdict["refusals"],
            }
        )
    refused = [row for row in histories if row["verdict"] == "PLANNED-REFUSED"]
    if refused:
        raise SystemExit(f"planned histories refused: {refused}")
    analysis_finished = time.time()

    probes = run_boundary_probes()
    if not all(probe["discriminates"] for probe in probes.values()):
        failed = [name for name, probe in probes.items() if not probe["discriminates"]]
        raise SystemExit(f"boundary probes failed to discriminate: {failed}")

    output_digests["planned-histories.json"] = write_json(
        out / "planned-histories.json",
        {
            "protocol_id": protocol.PROTOCOL_ID,
            "protocol_sha256": freeze["protocol_sha256"],
            "histories": histories,
            "planned_pairs": protocol.default_planned_pairs(),
        },
    )
    output_digests["boundary-probes.json"] = write_json(
        out / "boundary-probes.json",
        {"protocol_id": protocol.PROTOCOL_ID, "probes": probes},
    )
    estimate = protocol.cost_estimate()
    gate = protocol.forward_gate()
    output_digests["cost.json"] = write_json(
        out / "cost.json", {"protocol_id": protocol.PROTOCOL_ID, "estimate": estimate, "gate": gate}
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
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_sha256": freeze["protocol_sha256"],
        "mode": "metadata-only",
        "source_commit": source_commit(),
        "report_document": protocol_record,
        "planned_histories": len(histories),
        "planned_pairs_per_history_per_horizon": 5,
        "boundary_probes": {name: bool(probe["discriminates"]) for name, probe in probes.items()},
        "forward_gate": gate["gate"],
        "forwards": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "embedding_scores_read": 0,
        "label_blind_guard": "assert_planning_construction_label_blind passed",
        "provenance_note": (
            "prior M4/M5/M6 outcomes are exploratory motivation only; no DEV-04 or old "
            "movement output selected any planned value; every planned identity above "
            "was frozen before any generation and no generation exists"
        ),
        "environment": {
            "python": platform.python_version(),
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
            "design only: no waveform, manifest, embedding, forward or outcome exists",
            "burden-schedule-driven planner and within-file severity/duration metadata are unimplemented prerequisites",
            "weekly/monthly nesting and 40-file weekly density are assumed pending the pre-generation freeze",
            "planned inference exceeds the 1211-forward headroom; explicit new budget required",
            "no specificity, deployability or business performance claim follows",
        ],
    }
    output_digests["run.json"] = write_json(out / "run.json", run_record)

    print(
        json.dumps(
            {
                "protocol_sha256": freeze["protocol_sha256"],
                "n_histories": len(histories),
                "forward_gate": gate["gate"],
                "probes": {name: probe["discriminates"] for name, probe in probes.items()},
                "out": str(out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
