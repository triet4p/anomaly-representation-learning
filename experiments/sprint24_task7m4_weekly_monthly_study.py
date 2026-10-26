"""Sprint 24 Task 7-M4 weekly/monthly study execution (frozen endpoints and decision).

Runs the frozen ``S24-BIZ-WM-v1`` estimands on the admitted persistent embedding
cache: E1-E4, C1-C4/N1-N3 contrasts, declared sensitivities, history-cluster
bootstrap and the PROMISING / NOT-PROMISING / INCONCLUSIVE decision. It loads no
model, checkpoint, waveform or score and performs no forward; all work is
statistical over the accepted captured rows.

The pre-outcome conventions artifact digest is verified and bound in ``run.json``
before any endpoint is computed.
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

from representation import sprint24_weekly_monthly_endpoints as endpoints  # noqa: E402
from representation import sprint24_weekly_monthly_execution as execution  # noqa: E402
from representation import sprint24_weekly_monthly_study as study  # noqa: E402
from representation.sprint24_context_audit import canonical_json  # noqa: E402

DEFAULT_STUDY_SHA256 = "6f96c2750c15552bcdf25678a51ed7b24a1f7aecdb5914bf5985d3012d1daefd"
DEFAULT_PROTOCOL_SHA256 = "8da4afda89dd303099843eecda546cdc337696525ece0eed59f400d1031d8b63"


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


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    text = "".join(canonical_json(row) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_expected(values: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for value in values:
        key, digest = value.split("=", 1)
        expected[key.strip()] = digest.strip()
    return expected


def load_inputs(args: argparse.Namespace):
    edges_raw = json.loads(Path(args.edges).read_text(encoding="utf-8"))
    edges = {str(key): [float(value) for value in values] for key, values in edges_raw.items()}
    expected = parse_expected(args.expected_manifest_sha256)
    if set(expected) != set(edges):
        raise SystemExit("expected manifest digests must cover exactly the audited histories")
    manifests: dict[str, dict[str, object]] = {}
    for history_id in sorted(edges):
        path = Path(args.data_root) / "DEVELOPMENT" / history_id / "manifest.json"
        digest = sha256_path(path)
        if digest != expected[history_id]:
            raise SystemExit(f"accepted manifest digest mismatch for {history_id}")
        manifests[history_id] = json.loads(path.read_text(encoding="utf-8"))
    plan = execution.required_row_plan(manifests, edges)
    g2_record = execution.read_accepted_g2(Path(args.g2_accepted_root))
    inputs = execution.identity_inputs(g2_record)
    identities = execution.required_cache_identities(plan, inputs)
    return manifests, edges, plan, identities


def median(values: list[float]) -> float | None:
    return None if not values else float(np.median(values))


def sensitivity_value(
    associations: dict[str, object], estimable: list[str]
) -> float | None:
    values = []
    per_history = associations["per_history"]
    for history_id in estimable:
        entry = per_history[history_id]  # type: ignore[index]
        value = entry["rho_burden"]
        if value is not None:
            values.append(float(value))
    return median(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--g2-accepted-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-manifest-sha256", action="append", default=[])
    parser.add_argument("--expected-study-sha256", default=DEFAULT_STUDY_SHA256)
    parser.add_argument("--expected-protocol-sha256", default=DEFAULT_PROTOCOL_SHA256)
    parser.add_argument("--protocol-doc", default=study.PROTOCOL_DOC)
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conventions = endpoints.assert_conventions_frozen(repo=REPO)
    if study.study_sha256() != args.expected_study_sha256:
        raise SystemExit("frozen study digest mismatch")
    protocol_path = REPO / args.protocol_doc
    protocol_sha256 = sha256_path(protocol_path)
    if protocol_sha256 != args.expected_protocol_sha256:
        raise SystemExit("frozen protocol digest mismatch")
    manifests, edges, plan, identities = load_inputs(args)
    groups = endpoints.load_embedding_store(Path(args.cache_root), plan, identities)

    primary_calendar = study.primary_calendar()
    alternate_calendar = study.sensitivity_calendars()["alternate_anchor_week_zone"]
    primary = endpoints.build_endpoint_records(manifests, edges, primary_calendar, groups)
    alternate = endpoints.build_endpoint_records(manifests, edges, alternate_calendar, groups)

    history_ids = sorted(manifests)
    primary_records = primary["records"]
    alternate_records = alternate["records"]
    associations: dict[str, dict[str, object]] = {}
    decisions: dict[str, object] = {}
    uncertainties: dict[str, object] = {}
    sensitivity_blocks: dict[str, object] = {}
    for horizon_name in study.HORIZONS:
        primary_assoc = endpoints.horizon_associations(
            primary_records, horizon_name, study.PRIMARY_MODE, endpoints.e1_matched, history_ids
        )
        rolling_assoc = endpoints.horizon_associations(
            primary_records, horizon_name, "rolling", endpoints.e1_matched, history_ids
        )
        capped_assoc = endpoints.horizon_associations(
            primary_records, horizon_name, study.PRIMARY_MODE, endpoints.e1_capped, history_ids
        )
        anchor_assoc = endpoints.horizon_associations(
            alternate_records, horizon_name, study.PRIMARY_MODE, endpoints.e1_matched, history_ids
        )
        estimable = [
            history_id
            for history_id in history_ids
            if primary_assoc["per_history"][history_id]["rho_burden"] is not None  # type: ignore[index]
        ]
        sensitivity_values = {
            "rolling": sensitivity_value(rolling_assoc, estimable),
            "capped_n": sensitivity_value(capped_assoc, estimable),
            "alternate_anchor": sensitivity_value(anchor_assoc, estimable),
        }
        analysis = {"horizon": horizon_name, "mode": study.PRIMARY_MODE}
        analysis.update(endpoints.decision_inputs(primary_assoc, sensitivity_values, history_ids))
        decisions[horizon_name] = study.evaluate_decision(analysis)
        history_rhos = [
            float(primary_assoc["per_history"][history_id]["rho_burden"])  # type: ignore[index]
            for history_id in estimable
        ]
        uncertainties[horizon_name] = endpoints.bootstrap_uncertainty(history_rhos)
        sensitivity_blocks[horizon_name] = {
            "rolling": rolling_assoc,
            "capped_n": capped_assoc,
            "alternate_anchor": anchor_assoc,
            "values": sensitivity_values,
            "estimable_histories": estimable,
        }
        associations[horizon_name] = {
            "primary": primary_assoc,
            "rolling": rolling_assoc,
            "capped_n": capped_assoc,
            "alternate_anchor": anchor_assoc,
        }

    result = study.evaluate_source(manifests, edges, primary_calendar)
    label_blind = study.build_label_blind_manifest(result, primary_calendar)
    posthoc = study.build_posthoc_burden(result)
    coverage = study.coverage_matrix(label_blind, posthoc)
    cells = endpoints.endpoint_cell_summaries(primary_records, coverage)
    contrasts, contrast_counts = study.contrast_availability(posthoc)
    contrast_report = []
    for contrast in contrasts:
        contrast_report.append(
            {
                **contrast,
                "outcome": (
                    associations
                    if contrast["id"] in ("C1", "C2")
                    else None
                ),
            }
        )

    output_digests: dict[str, str] = {}
    output_digests["pre-outcome-conventions.json"] = write_json(
        out / "pre-outcome-conventions.json",
        {
            "path": conventions["path"],
            "canonical_sha256": conventions["canonical_sha256"],
            "bound_before_outcomes": True,
        },
    )
    output_digests["pair-endpoints-primary.jsonl"] = write_jsonl(
        out / "pair-endpoints-primary.jsonl", primary_records
    )
    output_digests["pair-endpoints-alternate.jsonl"] = write_jsonl(
        out / "pair-endpoints-alternate.jsonl", alternate_records
    )
    output_digests["reference-fits.json"] = write_json(
        out / "reference-fits.json", {"primary": primary["fits"], "alternate": alternate["fits"]}
    )
    output_digests["associations.json"] = write_json(out / "associations.json", associations)
    output_digests["sensitivities.json"] = write_json(
        out / "sensitivities.json", sensitivity_blocks
    )
    output_digests["uncertainty.json"] = write_json(out / "uncertainty.json", uncertainties)
    output_digests["decisions.json"] = write_json(out / "decisions.json", decisions)
    output_digests["contrasts.json"] = write_json(
        out / "contrasts.json",
        {"contrasts": contrast_report, "counts": contrast_counts},
    )
    output_digests["execution-cells.json"] = write_json(
        out / "execution-cells.json",
        {
            "study_id": study.STUDY_ID,
            "study_sha256": study.study_sha256(),
            "declared_cell_count": coverage["declared_cell_count"],
            "n_cells": len(cells),
            "cells": cells,
            "status_counts": {
                status: sum(1 for cell in cells if cell["execution_status"] == status)
                for status in sorted({cell["execution_status"] for cell in cells})
            },
        },
    )
    output_digests["execution-ledger.json"] = write_json(
        out / "execution-ledger.json",
        {
            "study_id": study.STUDY_ID,
            "study_sha256": study.study_sha256(),
            "pre_outcome_conventions": conventions,
            "declared_cell_count": coverage["declared_cell_count"],
            "n_cells": len(cells),
            "cell_status_counts": {
                status: sum(1 for cell in cells if cell["execution_status"] == status)
                for status in sorted({cell["execution_status"] for cell in cells})
            },
            "cells": cells,
            "contrasts": contrast_report,
            "decisions": decisions,
            "uncertainty": uncertainties,
            "specificity_claim": False,
            "blocked_claims": list(study.BLOCKED_CLAIMS),
            "scaling_status": {
                horizon_name: decisions[horizon_name]["scaling_status"]  # type: ignore[index]
                for horizon_name in study.HORIZONS
            },
        },
    )

    total_pairs = len(primary_records)
    cost = {
        "study_id": study.STUDY_ID,
        "statistical_work_only": True,
        "forwards_executed_by_this_analysis": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "embedding_rows_read": sum(group.matrix.shape[0] for group in groups.values()),
        "pair_records_primary": total_pairs,
        "pair_records_alternate": len(alternate_records),
        "inherited_cumulative_forwards": study.INHERITED_CUMULATIVE_FORWARDS,
        "authorized_cumulative_cap": 54_000,
        "capture_cumulative_forwards": 52_789,
        "remaining_authorized_after_capture": 1_211,
    }
    output_digests["cost.json"] = write_json(out / "cost.json", cost)
    output_digests["custody.json"] = write_json(
        out / "custody.json",
        {
            "cache_root": str(Path(args.cache_root).resolve()),
            "cache_tree_sha256": sha256_path(
                REPO / "artifacts/sprint-24/evidence/s24-7-m4-a01/runs/tree-sha256-local.txt"
            )
            if (REPO / "artifacts/sprint-24/evidence/s24-7-m4-a01/runs/tree-sha256-local.txt").is_file()
            else None,
            "conventions": conventions,
            "study_sha256": study.study_sha256(),
            "protocol_sha256": protocol_sha256,
        },
    )
    finished = time.time()
    run = {
        "study_id": study.STUDY_ID,
        "study_sha256": study.study_sha256(),
        "protocol_sha256": protocol_sha256,
        "pre_outcome_conventions": conventions,
        "source_commit": source_commit(),
        "cache_root": str(Path(args.cache_root).resolve()),
        "horizons": list(study.HORIZONS),
        "representation_seeds": list(study.REPRESENTATION_SEEDS),
        "decision_summary": {
            horizon_name: {
                "disposition": decisions[horizon_name]["disposition"],
                "median_rho_burden": decisions[horizon_name]["median_rho_burden"],
                "median_rho_residual": decisions[horizon_name]["median_rho_residual"],
                "sensitivity_values": decisions[horizon_name]["sensitivity_values"],
                "estimable_histories": decisions[horizon_name]["estimable_histories"],
                "specificity_claim": False,
                "scaling_status": decisions[horizon_name]["scaling_status"],
            }
            for horizon_name in study.HORIZONS
        },
        "forwards": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "timing": {"started_unix": started, "finished_unix": finished, "wall_s": finished - started},
        "outputs": output_digests,
    }
    output_digests["run.json"] = write_json(out / "run.json", run)
    print(
        json.dumps(
            {
                "study_sha256": study.study_sha256(),
                "conventions": conventions["canonical_sha256"],
                "decisions": {
                    horizon_name: decisions[horizon_name]["disposition"]
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
