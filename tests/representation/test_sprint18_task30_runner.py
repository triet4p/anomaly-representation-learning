"""Focused Task 30 N4-CRN runner contract tests.

Covers consumer-visible behavior only: the final all-Fit residual-bound
computation (Fit-only quantiles, frozen floor/cap edges, canonical record
digest), and the paired Development aggregation (same-seed deltas, positive
seed counts, and the exactly-zero ``S_pop`` delta that the unchanged-B0 branch
guarantees). The full certified-DATA path is exercised by the runner itself;
these tests use tiny synthetic record/metric shapes, never certified evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from representation.crn import N4ScaleRecord  # noqa: E402
from representation.config import V1Config  # noqa: E402
from representation.model import V1RepresentationModel  # noqa: E402
from representation.sprint18_harness import CertifiedSample  # noqa: E402
from synth.schema import FileSample, SampleLabel  # noqa: E402

import sprint18_task30_n4_runner as runner
import sprint18_task13b_n4_capacity as capacity  # noqa: E402

HISTORIES = ("S18I-ITER-0005-FIT-01", "S18I-ITER-0005-FIT-02", "S18I-ITER-0005-FIT-03")


def _records(history_id: str, residuals: list[float]) -> list[N4ScaleRecord]:
    return [
        N4ScaleRecord(
            history_id=history_id,
            file_id=f"file-{index:03d}",
            patch_index=index,
            context=torch.zeros(128, dtype=torch.float32),
            residual=float(value),
        )
        for index, value in enumerate(residuals)
    ]


def test_final_bounds_use_only_fit_residuals_and_frozen_edges() -> None:
    pools = {
        HISTORIES[0]: _records(HISTORIES[0], [0.01, 0.02, 0.03, 0.04]),
        HISTORIES[1]: _records(HISTORIES[1], [0.05, 0.06, 0.07, 0.08]),
        HISTORIES[2]: _records(HISTORIES[2], [0.09, 0.10, 0.11, 0.12]),
    }
    bounds = runner._final_bounds(pools, list(HISTORIES))
    residuals = np.arange(1, 13, dtype=np.float64) / 100.0
    assert bounds["record_count"] == 12
    assert bounds["q05"] == float(np.quantile(residuals, 0.05, method="linear"))
    assert bounds["q95"] == float(np.quantile(residuals, 0.95, method="linear"))
    assert bounds["v_floor"] == max(bounds["q05"], 1e-6)
    assert bounds["v_cap"] == max(bounds["q95"], 10.0 * bounds["v_floor"])
    # Canonical residual digest covers every in-scope record exactly once.
    expected = hashlib.sha256(
        runner.json.dumps(
            [
                [history_id, record.file_id, int(record.patch_index), float(record.residual)]
                for history_id in HISTORIES
                for record in pools[history_id]
            ],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert bounds["residual_records_sha256"] == expected


def test_final_bounds_reject_non_fit_pool_geometry() -> None:
    with pytest.raises((ValueError, SystemExit)):
        runner._final_bounds({"H-X": []}, ["H-X"])


DEVELOPMENT_HISTORIES = (
    "S18I-ITER-0005-DEVELOPMENT-01",
    "S18I-ITER-0005-DEVELOPMENT-02",
    "S18I-ITER-0005-DEVELOPMENT-03",
    "S18I-ITER-0005-DEVELOPMENT-04",
)
#: The accepted per-history identity provenance (real A14/B0 values).
DEVELOPMENT_PROVENANCE = {
    "S18I-ITER-0005-DEVELOPMENT-01": (
        "ce7c004b3ba7c6bca4187065df3299691c4d6247500eda480747e37b48fe1e08", "H-S18-DEV-01",
    ),
    "S18I-ITER-0005-DEVELOPMENT-02": (
        "cee368b966874289e8b1b81980ebdf2db566501fe8e2864326651d73f6179310", "H-S18-DEV-02",
    ),
    "S18I-ITER-0005-DEVELOPMENT-03": (
        "b1da957137936e01fda27b2af4194ea8f4be2cf611ce96315dba677145ae8f9e", "H-S18-DEV-03",
    ),
    "S18I-ITER-0005-DEVELOPMENT-04": (
        "3af0811329031fc6f1ffc56ffa837d46a3b9b9095287128d08c46706b187f9af", "H-S18-DEV-04",
    ),
}


def _identity(history_id: str, support: str) -> dict:
    manifest, schema = DEVELOPMENT_PROVENANCE[history_id]
    return {
        "history_id": history_id,
        "role_manifest_sha256": manifest,
        "schema_history_id": schema,
        "support_sha256": support,
    }


def _b0_support_record(history_id: str, support: str) -> dict:
    """The accepted B0 ``development.support.per_history`` record, exactly.

    The accepted ledger carries three keys here — never the history identity,
    which lives on the B0 branch rows. The two branch provenance digests are
    distinct, as the public harness requires.
    """
    return {
        "branch_provenance_sha256": {
            branch: hashlib.sha256(
                f"{history_id}:{branch}".encode("utf-8")
            ).hexdigest()
            for branch in runner.SUPPORT_BRANCHES
        },
        "file_ids": [f"{history_id}-file-0", f"{history_id}-file-1"],
        "support_sha256": support,
    }


def _row_metrics(value: float) -> dict:
    metrics = {name: value for name in runner.PAIRED_HISTORY_METRICS}
    metrics.update({"robot_days": 1692.0, "false_episodes": 0, "n_p": 80, "n_w": 47})
    return metrics


def _branch(*, macro: dict, value: float, threshold: float, support: str) -> dict:
    return {
        "threshold": threshold,
        "macro": dict(macro),
        "per_history": [
            {**_row_metrics(value), **_identity(history_id, support)}
            for history_id in DEVELOPMENT_HISTORIES
        ],
    }


def _seed_record(seed: int, *, arm: bool, support: str, bank_sha: str, macro_shift: float) -> dict:
    histories = list(DEVELOPMENT_HISTORIES)
    macro = {name: 0.5 + (macro_shift if arm else 0.0) for name in runner.PAIRED_MACRO_METRICS}
    branches = {
        "S_pred": _branch(
            macro=macro,
            value=0.5 + (macro_shift if arm else 0.0),
            threshold=0.1 + (0.01 if arm else 0.0),
            support=support,
        ),
        # The N4 S_pop branch is the unchanged B0 path: identical values.
        "S_pop": _branch(
            macro={name: 0.5 for name in runner.PAIRED_MACRO_METRICS},
            value=0.5,
            threshold=0.2,
            support=support,
        ),
    }
    record = {
        "model_seed": seed,
        "arm_id": "N4-CRN" if arm else "B0-FULL",
        "status": "ok",
        "normative": True,
        "failure_reasons": [],
        "bank": {"source_rows": 5446, "bank_sha256": bank_sha},
        "calibration": {"source_support_sha256": support},
        "development": {
            "support": {
                "evaluation_histories": histories,
                "per_history": {
                    history_id: _b0_support_record(history_id, support)
                    for history_id in histories
                }
                if not arm
                else {},
            },
            "branches": branches,
            "saturation": {"cap_gate_pass": True},
            "s_pop_parity": {"S_pop_bitwise_equal_to_B0": True},
        },
    }
    if arm:
        record["calibration"] = {"support": {"support_sha256": support}}
        record["development"]["support"] = {
            "evaluation_histories": histories,
            "per_history_support_sha256": {history_id: support for history_id in histories},
        }
    return record


def _paired_case(macro_shift: float = 0.25) -> tuple[dict, dict]:
    support = "ab" * 32
    bank_sha = "cd" * 32
    arm_records = {
        str(seed): _seed_record(seed, arm=True, support=support, bank_sha=bank_sha, macro_shift=macro_shift)
        for seed in (181801, 181802, 181803)
    }
    b0_records = {
        str(seed): _seed_record(seed, arm=False, support=support, bank_sha=bank_sha, macro_shift=0.0)
        for seed in (181801, 181802, 181803)
    }
    return arm_records, b0_records


def test_paired_development_aggregates_deltas_and_zero_s_pop() -> None:
    support = "ab" * 32
    bank_sha = "cd" * 32
    arm_records = {
        str(seed): _seed_record(seed, arm=True, support=support, bank_sha=bank_sha, macro_shift=0.25)
        for seed in (181801, 181802, 181803)
    }
    b0_records = {
        str(seed): _seed_record(seed, arm=False, support=support, bank_sha=bank_sha, macro_shift=0.0)
        for seed in (181801, 181802, 181803)
    }
    paired = runner._paired_development(
        arm_records=arm_records, b0_records=b0_records, b0_records_sha256="ef" * 32,
    )
    assert paired["schema"] == runner.PAIRED_SCHEMA
    assert paired["b0_seed_records_sha256"] == "ef" * 32
    for branch in ("S_pred", "S_pop"):
        aggregate = paired["branches"][branch]["aggregate"]["macro"]
        for name in runner.PAIRED_MACRO_METRICS:
            assert set(aggregate[name]) == {"mean_delta", "positive_seeds"}
    pred = paired["branches"]["S_pred"]["aggregate"]["macro"]
    assert pred["PW_macro"]["mean_delta"] == pytest.approx(0.25)
    assert pred["PW_macro"]["positive_seeds"] == 3
    # S_pop is the unchanged B0 path: deltas are exactly zero, by construction.
    pop = paired["branches"]["S_pop"]["aggregate"]["macro"]
    assert pop["PW_macro"]["mean_delta"] == 0.0
    assert pop["PW_macro"]["positive_seeds"] == 0
    pop_history = paired["branches"]["S_pop"]["aggregate"]["per_history"]
    for history_id, metrics in pop_history.items():
        for name, value in metrics.items():
            assert value["mean_delta"] == 0.0
            assert value["positive_seeds"] == 0


def test_paired_development_rejects_support_or_bank_mismatch() -> None:
    support = "ab" * 32
    bank_sha = "cd" * 32
    arm_records = {
        str(seed): _seed_record(seed, arm=True, support=support, bank_sha=bank_sha, macro_shift=0.1)
        for seed in (181801, 181802, 181803)
    }
    b0_records = {
        str(seed): _seed_record(seed, arm=False, support=support, bank_sha=bank_sha, macro_shift=0.0)
        for seed in (181801, 181802, 181803)
    }
    drifted = {key: dict(value) for key, value in b0_records.items()}
    value_802 = b0_records["181802"]
    drifted["181802"] = {**value_802, "bank": {**value_802["bank"], "bank_sha256": "00" * 32}}
    with pytest.raises(SystemExit):
        runner._paired_development(
            arm_records=arm_records, b0_records=drifted, b0_records_sha256="ef" * 32,
        )
    drifted_support = {key: dict(value) for key, value in b0_records.items()}
    value_803 = b0_records["181803"]
    drifted_support["181803"] = {
        **value_803,
        "calibration": {**value_803["calibration"], "source_support_sha256": "11" * 32},
    }
    with pytest.raises(SystemExit):
        runner._paired_development(
            arm_records=arm_records, b0_records=drifted_support, b0_records_sha256="ef" * 32,
        )


def test_paired_development_accepts_any_per_history_row_order() -> None:
    """The paired report matches rows by authenticated identity, not positional order."""
    arm_records, b0_records = _paired_case()
    for records in (arm_records, b0_records):
        for record in records.values():
            for branch in ("S_pred", "S_pop"):
                record["development"]["branches"][branch]["per_history"].reverse()
    paired = runner._paired_development(
        arm_records=arm_records, b0_records=b0_records, b0_records_sha256="ef" * 32,
    )
    for branch in ("S_pred", "S_pop"):
        rows = paired["branches"][branch]["per_seed"][0]["per_history"]
        assert [row["history_id"] for row in rows] == list(DEVELOPMENT_HISTORIES)
        assert set(paired["branches"][branch]["aggregate"]["per_history"]) == set(
            DEVELOPMENT_HISTORIES
        )
    assert set(paired["support"]["per_history_identity"]) == set(DEVELOPMENT_HISTORIES)
    assert set(paired["record_states"]) == {"181801", "181802", "181803"}
    assert paired["aggregation"]["confidence_intervals"].startswith("none")


def test_paired_development_refuses_missing_duplicate_provenance_and_legacy_rows() -> None:
    arm_records, b0_records = _paired_case()
    extra = {"b0_records_sha256": "ef" * 32}

    missing = copy.deepcopy(arm_records)
    rows = missing["181801"]["development"]["branches"]["S_pred"]["per_history"]
    missing["181801"]["development"]["branches"]["S_pred"]["per_history"] = rows[:-1]
    with pytest.raises(SystemExit, match="omit a frozen history"):
        runner._paired_development(arm_records=missing, b0_records=b0_records, **extra)

    duplicated = copy.deepcopy(arm_records)
    rows = duplicated["181802"]["development"]["branches"]["S_pop"]["per_history"]
    duplicated["181802"]["development"]["branches"]["S_pop"]["per_history"] = [
        rows[0], rows[0], rows[1], rows[2],
    ]
    with pytest.raises(SystemExit, match="duplicate"):
        runner._paired_development(arm_records=duplicated, b0_records=b0_records, **extra)

    drifted = copy.deepcopy(arm_records)
    drifted["181803"]["development"]["branches"]["S_pred"]["per_history"][0][
        "role_manifest_sha256"
    ] = "00" * 32
    with pytest.raises(SystemExit, match="provenance differs"):
        runner._paired_development(arm_records=drifted, b0_records=b0_records, **extra)

    # The pre-fix row shape (no identity) is refused rather than joined by position.
    legacy = copy.deepcopy(b0_records)
    for row in legacy["181801"]["development"]["branches"]["S_pred"]["per_history"]:
        for field in runner.DEVELOPMENT_IDENTITY_FIELDS:
            row.pop(field)
    with pytest.raises(SystemExit, match="frozen row schema"):
        runner._paired_development(arm_records=arm_records, b0_records=legacy, **extra)

    unsupported = copy.deepcopy(arm_records)
    for row in unsupported["181801"]["development"]["branches"]["S_pred"]["per_history"]:
        row["support_sha256"] = "11" * 32
    with pytest.raises(SystemExit, match="provenance differs"):
        runner._paired_development(arm_records=unsupported, b0_records=b0_records, **extra)


def test_paired_development_joins_accepted_b0_support_to_branch_identity() -> None:
    """The accepted B0 support record is three keys; identity lives on the branch rows.

    This is the A14/A16 defect: pairing used to demand ``role_manifest_sha256``
    inside ``development.support.per_history``, which the accepted ledger never
    carries. The roster must be the branch-row provenance joined to the support
    record, with exact cross-branch agreement and support-digest equality.
    """
    arm_records, b0_records = _paired_case()
    extra = {"b0_records_sha256": "ef" * 32}
    support = "ab" * 32
    for record in b0_records.values():
        for history_id in DEVELOPMENT_HISTORIES:
            entry = record["development"]["support"]["per_history"][history_id]
            assert set(entry) == {"branch_provenance_sha256", "file_ids", "support_sha256"}
            assert set(entry) == set(runner.DEVELOPMENT_SUPPORT_FIELDS)
        for branch in runner.SUPPORT_BRANCHES:
            for row in record["development"]["branches"][branch]["per_history"]:
                assert set(row) == set(runner.DEVELOPMENT_IDENTITY_FIELDS) | set(
                    runner.DEVELOPMENT_ROW_METRICS
                )
                assert len(row) == 17

    paired = runner._paired_development(
        arm_records=arm_records, b0_records=b0_records, **extra,
    )
    for history_id, (manifest, schema) in DEVELOPMENT_PROVENANCE.items():
        assert paired["support"]["per_history_identity"][history_id] == {
            "role_manifest_sha256": manifest,
            "schema_history_id": schema,
            "support_sha256": support,
        }

    # The pre-A16 fixture shape (identity dict inside the support record) is drift.
    enriched = copy.deepcopy(b0_records)
    first_history = DEVELOPMENT_HISTORIES[0]
    enriched["181801"]["development"]["support"]["per_history"][first_history] = _identity(
        first_history, support
    )
    with pytest.raises(SystemExit, match="frozen shape"):
        runner._paired_development(arm_records=arm_records, b0_records=enriched, **extra)

    # A missing support record cannot be repaired from the branch rows.
    absent = copy.deepcopy(b0_records)
    absent["181802"]["development"]["support"]["per_history"].pop(DEVELOPMENT_HISTORIES[1])
    with pytest.raises(SystemExit, match="per-history support differs from the roster"):
        runner._paired_development(arm_records=arm_records, b0_records=absent, **extra)

    # A missing support field fails closed.
    partial = copy.deepcopy(b0_records)
    partial["181803"]["development"]["support"]["per_history"][first_history].pop("file_ids")
    with pytest.raises(SystemExit, match="frozen shape"):
        runner._paired_development(arm_records=arm_records, b0_records=partial, **extra)

    # Aliased branch provenance is refused (the public two-branch contract).
    aliased = copy.deepcopy(b0_records)
    provenance = aliased["181801"]["development"]["support"]["per_history"][first_history][
        "branch_provenance_sha256"
    ]
    provenance["S_pop"] = provenance["S_pred"]
    with pytest.raises(SystemExit, match="branch provenance"):
        runner._paired_development(arm_records=arm_records, b0_records=aliased, **extra)

    # A drifted support digest is refused against the arm support map ...
    drifted_support = copy.deepcopy(b0_records)
    drifted_support["181801"]["development"]["support"]["per_history"][first_history][
        "support_sha256"
    ] = "11" * 32
    with pytest.raises(SystemExit, match="paired Development support differs"):
        runner._paired_development(arm_records=arm_records, b0_records=drifted_support, **extra)

    # ... and against the branch rows it has to agree with.
    drifted_rows = copy.deepcopy(b0_records)
    for branch in runner.SUPPORT_BRANCHES:
        for row in drifted_rows["181802"]["development"]["branches"][branch]["per_history"]:
            if row["history_id"] == first_history:
                row["support_sha256"] = "22" * 32
    with pytest.raises(SystemExit, match="differs from its branch rows"):
        runner._paired_development(arm_records=arm_records, b0_records=drifted_rows, **extra)

    # Cross-branch identity disagreement fails closed.
    cross = copy.deepcopy(b0_records)
    for row in cross["181803"]["development"]["branches"]["S_pop"]["per_history"]:
        if row["history_id"] == first_history:
            row["role_manifest_sha256"] = "00" * 32
    with pytest.raises(SystemExit, match="branches disagree"):
        runner._paired_development(arm_records=arm_records, b0_records=cross, **extra)

    # A duplicated branch row is refused instead of silently collapsed.
    duplicated = copy.deepcopy(b0_records)
    rows = duplicated["181801"]["development"]["branches"]["S_pred"]["per_history"]
    duplicated["181801"]["development"]["branches"]["S_pred"]["per_history"] = [
        rows[0], rows[0], rows[1], rows[2],
    ]
    with pytest.raises(SystemExit, match="duplicate"):
        runner._paired_development(arm_records=arm_records, b0_records=duplicated, **extra)


DAY = 86400.0


def _synthetic_development_history(robot: str) -> tuple[list[dict], list[dict], dict[str, float]]:
    """One history with a P and a W failure, matching the frozen metric contract shape."""
    rows = [
        {
            "file_id": f"{robot}-d{day:02d}",
            "robot_id": robot,
            "program_id": "program-01",
            "start_time": (day - 0.01) * DAY,
            "end_time": day * DAY,
            "member_views": ["test-temporal"],
            "is_censored": False,
            "is_quarantined": False,
            "file_label": "normal",
        }
        for day in range(1, 41)
    ]
    ledger = [
        {
            "failure_id": f"{robot}-p", "robot_id": robot, "failure_time": 30 * DAY,
            "cohort": "P", "subtype": "P1", "degradation_onset": 20 * DAY,
            "duration_d": 10.0, "severity": 2.0,
        },
        {
            "failure_id": f"{robot}-w", "robot_id": robot, "failure_time": 10 * DAY,
            "cohort": "W", "subtype": "W1", "degradation_onset": 0.0,
            "duration_d": 10.0, "severity": 2.0,
        },
    ]
    horizons = [(23 * DAY, 30 * DAY), (3 * DAY, 10 * DAY)]
    scores = {
        row["file_id"]: (1.0 if any(start <= row["end_time"] <= stop for start, stop in horizons) else 0.0)
        for row in rows
    }
    return rows, ledger, scores


def _score_metrics_case() -> tuple[dict, list[list[dict]], list[list[dict]], list[dict], list[dict]]:
    rows_by_history, ledgers, windows, roster = [], [], [], []
    scores: dict[str, float] = {}
    for index, history_id in enumerate(DEVELOPMENT_HISTORIES, start=1):
        rows, ledger, history_scores = _synthetic_development_history(f"robot-{index:02d}")
        rows_by_history.append(rows)
        ledgers.append(ledger)
        windows.append({})
        scores.update(history_scores)
        roster.append(
            {
                **_identity(history_id, f"{index:02d}" * 32),
                "file_ids_sha256": runner._canonical_sha256([row["file_id"] for row in rows]),
            }
        )
    return scores, rows_by_history, ledgers, windows, roster


def test_score_metrics_attaches_authenticated_per_history_identity() -> None:
    from synth import probe15 as P

    scores, rows_by_history, ledgers, windows, roster = _score_metrics_case()
    result = runner._score_metrics(
        file_scores=scores, rows_by_history=rows_by_history, ledgers=ledgers,
        windows=windows, threshold=0.5, roster=roster,
    )
    canonical = P.evaluate_histories(scores, rows_by_history, ledgers, windows, 0.5)
    assert [row["history_id"] for row in result["per_history"]] == list(DEVELOPMENT_HISTORIES)
    expected_keys = set(runner.DEVELOPMENT_IDENTITY_FIELDS) | set(runner.DEVELOPMENT_ROW_METRICS)
    for row, identity, expected in zip(result["per_history"], roster, canonical["per_history"]):
        assert set(row) == expected_keys
        for field in runner.DEVELOPMENT_IDENTITY_FIELDS:
            assert row[field] == identity[field]
        assert row["P_AUROC"] == pytest.approx(expected["auc_p"])
        assert row["PW_AUROC"] == pytest.approx(expected["auc_pw"])
        assert row["n_p"] == expected["n_p"]
        assert row["robot_days"] == pytest.approx(expected["robot_days"])
    assert result["macro"]["PW_macro"] == pytest.approx(canonical["macro_auc_pw"])
    # The exact A14 defect: the previous row shape could not satisfy the paired key.
    assert all("history_id" in row for row in result["per_history"])


def test_score_metrics_refuses_roster_that_does_not_match_evaluated_rows() -> None:
    _, rows_by_history, ledgers, windows, roster = _score_metrics_case()
    empty_windows = [{} for _ in rows_by_history]
    with pytest.raises(SystemExit, match="does not match its evaluated rows"):
        runner._score_metrics(
            file_scores={}, rows_by_history=rows_by_history, ledgers=ledgers,
            windows=empty_windows, threshold=0.5, roster=list(reversed(roster)),
        )
    duplicate = [roster[0], roster[0], roster[2], roster[3]]
    with pytest.raises(SystemExit, match="repeats history"):
        runner._score_metrics(
            file_scores={}, rows_by_history=rows_by_history, ledgers=ledgers,
            windows=empty_windows, threshold=0.5, roster=duplicate,
        )


def _tiny_real_checkpoint(directory: Path, seed: int) -> Path:
    """Fit and save a real (tiny) N4 scale envelope through the public path."""
    pools = _tiny_n4_pools()
    records = [record for history_id in HISTORIES for record in pools[history_id]]
    bounds = runner._final_bounds(pools, list(HISTORIES))
    scale = runner.init_n4_scale_model(128, seed)
    result = runner.fit_n4_scale_scope(
        records=records, training_history_ids=list(HISTORIES), scope="final",
        arm=runner._n4_arm(seed, "ab" * 32), binding=_fit_binding_stub(),
        scale_model=scale, v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
    )
    block = runner._save_and_restore_scale(
        scale=scale, out_dir=directory, seed=seed, fit_history_ids=list(HISTORIES),
        bounds=bounds, fit_result={**result, "probe_records": records},
    )
    return Path(block["path"])


TINY_SUPPORT = {
    history_id: f"{index:02d}" * 32
    for index, history_id in enumerate(DEVELOPMENT_HISTORIES, start=1)
}
TINY_BANK = "cd" * 32
TINY_CALIBRATION_SUPPORT = "ef" * 32
TINY_RECORD_TOTAL = 3
#: The superseded prospective lock the A14 scientific execution ran under. The
#: report execution runs on the live v10 lock, so this also proves the two
#: identities are independent.
ORIGINAL_LOCK_PATH = "experiments/sprint18-task14-config-lock-v9.json"
ORIGINAL_LOCK_SCHEMA = "s18-task14-config-lock-v9"
ORIGINAL_LOCK_VERSION = 9
ORIGINAL_LOCK_FILE = "a9990bd189706a0fdadfeaee422ef307599e3f4d9c5a72102dfa1f015d5cd57a"
ORIGINAL_LOCK_CORE = "b171e50014a9bfad0edc865fc42a52a49d073b8b0a2e47788d5020b741266bbf"


def _tiny_metric_rows(value: float) -> list[dict]:
    return [_row_metrics(value) for _ in DEVELOPMENT_HISTORIES]


def _tiny_identified_rows(value: float) -> list[dict]:
    return [
        {**_row_metrics(value), **_identity(history_id, TINY_SUPPORT[history_id])}
        for history_id in DEVELOPMENT_HISTORIES
    ]


def _tiny_arm_record(seed: int, coherent: Path, envelope: Path, scores: Path) -> dict:
    return {
        "schema": runner.SEED_SCHEMA,
        "task_id": "S18-T30",
        "attempt_id": "S18-T30-A02",
        "agent_id": "S18T30A02",
        "arm_id": "N4-CRN",
        "baseline_arm_id": "B0-FULL",
        "model_seed": seed,
        "status": "ok",
        "normative": True,
        "failure_reasons": [],
        "bank": {"source_rows": 5446, "bank_sha256": TINY_BANK, "matches_accepted_B0_bank": True},
        "calibration": {"support": {"support_sha256": TINY_CALIBRATION_SUPPORT}},
        "scale_fit": {
            "record_total": TINY_RECORD_TOTAL,
            "optimizer_steps": 1,
            "record_counts": {history_id: 1 for history_id in HISTORIES},
        },
        "coherent_checkpoint": {
            "schema": runner.COHERENT_CHECKPOINT_SCHEMA,
            "strict_restore_b0_model_ema_scale_equal": True,
            "representation_step": 0,
            "path": str(coherent),
            "sha256": hashlib.sha256(coherent.read_bytes()).hexdigest(),
        },
        "scale_checkpoint": {
            "path": str(envelope),
            "sha256": hashlib.sha256(envelope.read_bytes()).hexdigest(),
        },
        "next_stage": {
            "development_scores_path": str(scores),
            "development_scores_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
        },
        "development": {
            "support": {
                "evaluation_histories": list(DEVELOPMENT_HISTORIES),
                "per_history_support_sha256": dict(TINY_SUPPORT),
            },
            # The A14 shape: metric rows with no identity fields.
            "branches": {
                "S_pred": {
                    "threshold": 0.1,
                    "macro": {name: 0.6 for name in runner.PAIRED_MACRO_METRICS},
                    "per_history": _tiny_metric_rows(0.6),
                },
                "S_pop": {
                    "threshold": 0.2,
                    "macro": {name: 0.5 for name in runner.PAIRED_MACRO_METRICS},
                    "per_history": _tiny_metric_rows(0.5),
                },
            },
            # The accepted B0 reference rows, which have always carried identity.
            "b0_reference": {
                branch: {
                    "threshold": 0.2,
                    "macro": {name: 0.5 for name in runner.PAIRED_MACRO_METRICS},
                    "per_history": _tiny_identified_rows(0.5),
                }
                for branch in ("S_pred", "S_pop")
            },
            "saturation": {"cap_gate_pass": True},
            "s_pop_parity": {"S_pop_bitwise_equal_to_B0": True},
        },
    }


def _tiny_b0_record(seed: int) -> dict:
    return {
        "schema": runner.SEED_SCHEMA,
        "task_id": "S18-T30",
        "arm_id": "B0-FULL",
        "model_seed": seed,
        "status": "ok",
        "normative": True,
        "failure_reasons": [],
        "bank": {"source_rows": 5446, "bank_sha256": TINY_BANK},
        "calibration": {"source_support_sha256": TINY_CALIBRATION_SUPPORT},
        "development": {
            "support": {
                "evaluation_histories": list(DEVELOPMENT_HISTORIES),
                "per_history": {
                    history_id: _b0_support_record(history_id, TINY_SUPPORT[history_id])
                    for history_id in DEVELOPMENT_HISTORIES
                },
            },
            "branches": {
                branch: {
                    "threshold": 0.2,
                    "macro": {name: 0.5 for name in runner.PAIRED_MACRO_METRICS},
                    "per_history": _tiny_identified_rows(0.5),
                }
                for branch in ("S_pred", "S_pop")
            },
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _tiny_report_fixture(tmp_path: Path) -> dict:
    """Build a tiny but real executed-run tree plus the report auth document."""
    import subprocess

    run_root = tmp_path / "run"
    run_root.mkdir()
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD^{tree}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    handoff = REPO_ROOT / "experiments" / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    handoff_sha = hashlib.sha256(handoff.read_bytes()).hexdigest()
    original_runner = "3af994ba002b5eb79599858ba90ceff18e099accdd79b28c0f1d63a31caaff80"
    live_runner = hashlib.sha256(
        (REPO_ROOT / "experiments" / "sprint18_task30_n4_runner.py").read_bytes()
    ).hexdigest()

    b0_path = run_root / "b0_seed_records.json"
    b0_records = {str(seed): _tiny_b0_record(seed) for seed in runner.MODEL_SEEDS}
    _write_json(b0_path, b0_records)

    records: dict[str, dict] = {}
    artifacts: dict[str, dict] = {}
    for seed in runner.MODEL_SEEDS:
        key = str(seed)
        seed_dir = run_root / f"seed-{key}"
        seed_dir.mkdir(parents=True)
        coherent = _tiny_real_checkpoint(seed_dir / "coherent", seed)
        envelope = _tiny_real_checkpoint(seed_dir / "scale", seed)
        scores = seed_dir / "development_scores.json"
        _write_json(scores, {"S_pred_normalized": {f"dev-{key}": 0.25}, "S_pred_b0_raw": {}, "S_pop_b0_unchanged": {}})
        record = _tiny_arm_record(seed, coherent, envelope, scores)
        records[key] = record
        _write_json(seed_dir / "seed_record.json", record)
        _write_json(seed_dir / "next_stage.json", record["next_stage"])
        artifacts[key] = {
            "coherent_checkpoint_sha256": record["coherent_checkpoint"]["sha256"],
            "scale_envelope_sha256": record["scale_checkpoint"]["sha256"],
            "development_scores_sha256": record["next_stage"]["development_scores_sha256"],
        }
    seed_records = run_root / "seed_records.json"
    _write_json(seed_records, records)

    lock = capacity.load_task14_lock(REPO_ROOT)
    lock_record = capacity.validate_task14_lock(
        lock, worktree=REPO_ROOT,
        observed_runner_sha256=hashlib.sha256(
            (REPO_ROOT / "experiments" / "sprint18_task13b_n4_capacity.py").read_bytes()
        ).hexdigest(),
    )
    custody = [
        {"model_seed": int(seed), "container_sha256": digest}
        for seed, digests in lock_record["capacity_certification"]["fold_checkpoint_sha256"].items()
        for digest in digests
    ]
    original_binding_sha = None
    original_binding = run_root / "original-binding.json"
    _write_json(
        original_binding,
        {
            "source_commit": head,
            "source_tree": tree,
            "runner_sha256": original_runner,
            "handoff_sha256": handoff_sha,
            "b0_seed_records_sha256": hashlib.sha256(b0_path.read_bytes()).hexdigest(),
        },
    )
    original_binding_sha = hashlib.sha256(original_binding.read_bytes()).hexdigest()
    preflight = {
        "schema": runner.PREFLIGHT_SCHEMA,
        "PREFLIGHT_OK": True,
        "source_commit": head,
        "runner_sha256": original_runner,
        "binding_sha256": original_binding_sha,
        "handoff_sha256": handoff_sha,
        "b0_seed_records_sha256": hashlib.sha256(b0_path.read_bytes()).hexdigest(),
        "task14_lock_binding": {
            "coefficient_lock_sha256": ORIGINAL_LOCK_CORE,
            "file_sha256": ORIGINAL_LOCK_FILE,
            "schema": ORIGINAL_LOCK_SCHEMA,
            "lock_version": ORIGINAL_LOCK_VERSION,
            "path": ORIGINAL_LOCK_PATH,
        },
        "capacity_custody_authentication": {
            "authenticated": True,
            "folds_authenticated": 9,
            "files": custody,
        },
        "development_support": [
            {
                "history_id": history_id,
                "manifest_sha256_bound": DEVELOPMENT_PROVENANCE[history_id][0],
            }
            for history_id in DEVELOPMENT_HISTORIES
        ],
        "seeds": {
            str(seed): {
                "record_total": TINY_RECORD_TOTAL,
                "final_scale_steps": 1,
                "record_counts": {history_id: 1 for history_id in HISTORIES},
            }
            for seed in runner.MODEL_SEEDS
        },
    }
    _write_json(run_root / "preflight.json", preflight)
    _write_json(run_root / "environment.json", {"mode": "full", "device": "cuda:0"})
    progress_lines = []
    for seed in runner.MODEL_SEEDS:
        progress_lines.append(json.dumps({"event": "restore_b0", "model_seed": seed}, sort_keys=True))
        progress_lines.append(
            json.dumps({"event": "seed_complete", "model_seed": seed, "status": "ok"}, sort_keys=True)
        )
    (run_root / "progress.jsonl").write_text("\n".join(progress_lines) + "\n", encoding="utf-8")

    report_binding = run_root / "report-binding.json"
    _write_json(report_binding, {"source_commit": head, "runner_sha256": live_runner})
    auth = {
        "schema": runner.REPORT_AUTH_SCHEMA,
        "task_id": "S18-T30",
        "original_execution": {"task_id": "S18-T30", "attempt_id": "S18-T30-A02", "agent_id": "S18T30A02"},
        "report_execution": {"task_id": "S18-T30", "attempt_id": "S18-T30-A15", "agent_id": "S18T30A15"},
        "source_commit": head,
        "source_tree": tree,
        "runner_sha256": original_runner,
        "binding_sha256": original_binding_sha,
        "handoff_sha256": handoff_sha,
        "lock": {
            "path": ORIGINAL_LOCK_PATH,
            "schema": ORIGINAL_LOCK_SCHEMA,
            "version": ORIGINAL_LOCK_VERSION,
            "file_sha256": ORIGINAL_LOCK_FILE,
            "core_sha256": ORIGINAL_LOCK_CORE,
        },
        "b0_seed_records": {
            "path": str(b0_path),
            "sha256": hashlib.sha256(b0_path.read_bytes()).hexdigest(),
        },
        "run": {
            "seed_records_sha256": hashlib.sha256(seed_records.read_bytes()).hexdigest(),
            "preflight_sha256": hashlib.sha256((run_root / "preflight.json").read_bytes()).hexdigest(),
            "environment_sha256": hashlib.sha256((run_root / "environment.json").read_bytes()).hexdigest(),
            "progress_sha256": hashlib.sha256((run_root / "progress.jsonl").read_bytes()).hexdigest(),
        },
        "development_history_ids": list(DEVELOPMENT_HISTORIES),
        "development_support_sha256": dict(TINY_SUPPORT),
        "artifacts": artifacts,
        "certified_data": {
            "authenticated": True,
            "folds_authenticated": 9,
            "evidence": "pinned_preflight_task14_envelope_custody",
        },
    }
    auth_path = run_root / "report-auth.json"
    _write_json(auth_path, auth)
    return {
        "run_root": run_root,
        "seed_records": seed_records,
        "auth": auth_path,
        "auth_sha256": hashlib.sha256(auth_path.read_bytes()).hexdigest(),
        "report_binding": report_binding,
        "original_binding": original_binding,
        "handoff": handoff,
        "head": head,
        "live_runner": live_runner,
    }


def test_report_entrypoint_emits_paired_development_from_legacy_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real `--mode report` entrypoint emits the report the A14 run could not.

    The consumed seed records use exactly the A14 row shape (no identity), so the
    report must join the authenticated roster/support and then pass through the
    previous ``KeyError`` line in ``_paired_development``. The immutable inputs
    are re-hashed unchanged, and no training/optimizer/scoring path is entered.
    """
    fixture = _tiny_report_fixture(tmp_path)
    seed_before = hashlib.sha256(fixture["seed_records"].read_bytes()).hexdigest()
    record_files_before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(fixture["run_root"].glob("seed-*/seed_record.json"))
    }

    def _training_path_entered(*args, **kwargs):
        raise AssertionError("report mode entered a fitting/optimizer/profiling path")

    # The report consumer must not call any scientific execution path.
    monkeypatch.setattr(runner, "fit_n4_scale_scope", _training_path_entered)
    monkeypatch.setattr(runner, "RepresentationTrainer", _training_path_entered)
    monkeypatch.setattr(runner, "profile_inference_flops", _training_path_entered)
    monkeypatch.setattr(runner, "n4_score_files", _training_path_entered)

    out_dir = tmp_path / "report-out"
    monkeypatch.setattr(
        sys, "argv",
        [
            "sprint18_task30_n4_runner.py",
            "--worktree", str(REPO_ROOT),
            "--handoff", str(fixture["handoff"]),
            "--binding", str(fixture["report_binding"]),
            "--out-dir", str(out_dir),
            "--source-commit", fixture["head"],
            "--driver-sha256", fixture["live_runner"],
            "--mode", "report",
            "--device", "cpu",
            "--seed-records", str(fixture["seed_records"]),
            "--report-auth", str(fixture["auth"]),
            "--report-auth-sha256", fixture["auth_sha256"],
            "--original-binding", str(fixture["original_binding"]),
        ],
    )
    assert runner.main() == 0

    paired_path = out_dir / "paired_development.json"
    report_path = out_dir / "report_record.json"
    manifest_path = out_dir / "manifest.json"
    assert paired_path.is_file() and report_path.is_file() and manifest_path.is_file()
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    assert paired["schema"] == runner.PAIRED_SCHEMA
    for branch in ("S_pred", "S_pop"):
        for seed_row in paired["branches"][branch]["per_seed"]:
            assert [row["history_id"] for row in seed_row["per_history"]] == list(
                DEVELOPMENT_HISTORIES
            )
    pred = paired["branches"]["S_pred"]["aggregate"]["macro"]["PW_macro"]
    assert pred["mean_delta"] == pytest.approx(0.1)
    assert pred["positive_seeds"] == 3
    pop = paired["branches"]["S_pop"]["aggregate"]["macro"]["PW_macro"]
    assert pop["mean_delta"] == 0.0
    assert paired["record_states"]["181801"]["status"] == "ok"
    assert paired["aggregation"]["confidence_intervals"].startswith("none")
    # Every per-history aggregate row is keyed by the authenticated roster.
    assert set(paired["branches"]["S_pred"]["aggregate"]["per_history"]) == set(
        DEVELOPMENT_HISTORIES
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == runner.REPORT_SCHEMA
    assert report["status"] == "complete"
    assert report["original_execution"]["attempt_id"] == "S18-T30-A02"
    assert report["report_execution"]["attempt_id"] == "S18-T30-A15"
    assert report["report_execution"]["training_calls"] == 0
    assert report["report_execution"]["optimizer_steps"] == 0
    assert report["report_execution"]["model_scoring_calls"] == 0
    assert report["paired_development_sha256"] == hashlib.sha256(
        paired_path.read_bytes()
    ).hexdigest()
    assert report["authentication"]["recorded_preflight"]["folds_authenticated"] == 9

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == runner.REPORT_MANIFEST_SCHEMA
    assert manifest["original_source"]["seed_records"]["sha256"] == seed_before
    assert manifest["report_source"]["paired_development"]["sha256"] == report[
        "paired_development_sha256"
    ]
    assert manifest["report_source"]["runner_sha256"] == fixture["live_runner"]

    # Immutable inputs were not rewritten by the report.
    assert hashlib.sha256(fixture["seed_records"].read_bytes()).hexdigest() == seed_before
    for path, digest in record_files_before.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_report_entrypoint_refuses_a_binding_or_provenance_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tampered report auth pin is refused before any report is written."""
    fixture = _tiny_report_fixture(tmp_path)
    auth = json.loads(fixture["auth"].read_text(encoding="utf-8"))
    auth["development_support_sha256"]["S18I-ITER-0005-DEVELOPMENT-02"] = "77" * 32
    tampered = tmp_path / "tampered-auth.json"
    _write_json(tampered, auth)
    out_dir = tmp_path / "report-refused"
    monkeypatch.setattr(
        sys, "argv",
        [
            "sprint18_task30_n4_runner.py",
            "--worktree", str(REPO_ROOT),
            "--handoff", str(fixture["handoff"]),
            "--binding", str(fixture["report_binding"]),
            "--out-dir", str(out_dir),
            "--source-commit", fixture["head"],
            "--driver-sha256", fixture["live_runner"],
            "--mode", "report",
            "--device", "cpu",
            "--seed-records", str(fixture["seed_records"]),
            "--report-auth", str(tampered),
            "--report-auth-sha256", hashlib.sha256(tampered.read_bytes()).hexdigest(),
            "--original-binding", str(fixture["original_binding"]),
        ],
    )
    with pytest.raises(SystemExit, match="Development support map differs"):
        runner.main()
    assert not (out_dir / "paired_development.json").exists()


def test_n4_score_row_declares_consumer_triple_and_patch_diagnostics() -> None:
    from representation.config import V1Config
    from representation.crn import init_n4_scale_model
    from representation.model import V1RepresentationModel
    from representation.sprint18_harness import deterministic_mask_seed
    from representation import n4_score_files
    from synth.config import PatchConfig
    from synth.patchify import Patchifier
    from synth.schema import FileSample, SampleLabel

    config = V1Config(
        n_channels=6, patch_size=32, stride=16, d_model=16, attention_heads=4,
        sequence_layers=1, dropout=0.0, n_robots=9, n_programs=8, seed=181801,
        contrastive_weight_max=0.05, ema_decay=0.996,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    scale = init_n4_scale_model(config.d_model, 181801)
    with torch.no_grad():
        scale.out.bias.fill_(math.log(1e-6))  # every patch clamps at v_floor
    rng = np.random.default_rng(91813)
    sample = FileSample(
        x=rng.normal(size=(6, 640)).astype(np.float32), file_id="t30-diag-00",
        file_label=SampleLabel.NORMAL, seed=181801, generator_version="t30-test",
        config_hash="t30-test", regime_sequence=[], robot_idx=0, program_idx=0,
    )
    sample.validate()

    from types import SimpleNamespace

    row = SimpleNamespace(file_id=sample.file_id, sample=sample)
    bounds = {"v_floor": 0.02, "v_cap": 1.4}
    mask_seed = deterministic_mask_seed(181801, "S18I-ITER-0005-FIT-01", sample.file_id)
    triples = n4_score_files(
        model=model, samples=[sample], scale_model=scale,
        v_floor=0.02, v_cap=1.4, patchifier=patchifier, masking_seed=mask_seed,
    )
    entry = runner._n4_score_row(
        model=model, scale=scale, row=row, history_id="S18I-ITER-0005-FIT-01",
        seed=181801, patchifier=patchifier, bounds=bounds, diagnostics=True,
    )
    triple = triples[sample.file_id]
    assert entry["file_id"] == sample.file_id
    assert entry["raw"] == triple["raw"]
    assert entry["scale"] == triple["scale"]
    assert entry["normalized"] == triple["normalized"]
    assert entry["eligible_patches"] >= 1
    assert entry["frac_at_floor"] == 1.0
    assert entry["frac_at_cap"] == 0.0
    assert entry["scale"] == pytest.approx(0.02)


def _write_prediction_cache(path: Path, records: list[dict]) -> tuple[str, str]:
    """Serialize a real Task23-schema cache and measure both of its identities."""
    inner_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": "s18-t23-c01-prediction-cache-v1",
        "cache_sha256": inner_sha256,
        "records": records,
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest(), inner_sha256


def _cache_binding_fixture(tmp_path: Path) -> tuple[dict, dict, Path]:
    """Build the exact corrected binding shape around a real serialized cache."""
    cache_path = tmp_path / "prediction_cache.json"
    file_sha256, inner_sha256 = _write_prediction_cache(
        cache_path,
        [
            {
                "file_id": "tiny-file",
                "actual_history_id": HISTORIES[0],
                "role": "FIT",
                "patch_mapping": {"eligible_patch_index": [0]},
            }
        ],
    )
    checkpoint_path = tmp_path / "b0-epoch100.pt"
    checkpoint_path.write_bytes(b"bound B0 checkpoint bytes")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    seeds = [str(seed) for seed in runner.MODEL_SEEDS]
    live_model = {seed: "a" * 64 for seed in seeds}
    live_ema = {seed: "b" * 64 for seed in seeds}
    fixture = {
        "checkpoints": {
            seed: {"path": str(checkpoint_path), "sha256": checkpoint_sha256} for seed in seeds
        },
        "ema_state_sha256": dict(live_ema),
        "live_model_state_sha256": dict(live_model),
        "prediction_cache_paths": {seed: str(cache_path) for seed in seeds},
        runner.PREDICTION_CACHE_FILE_SHA256_KEY: {seed: file_sha256 for seed in seeds},
        runner.PREDICTION_CACHE_INNER_SHA256_KEY: {seed: inner_sha256 for seed in seeds},
    }
    seed = runner.MODEL_SEEDS[0]
    key = str(seed)
    b0_record = {
        "arm_id": runner.BASELINE_ARM_ID,
        "status": "ok",
        "representation_epochs_completed": 100,
        "primary": {
            "restored_checkpoint_sha256": checkpoint_sha256,
            "live_model_state_sha256": live_model[key],
            "live_ema_state_sha256": live_ema[key],
        },
        "prediction_cache": {
            "path": str(cache_path),
            "cache_sha256": inner_sha256,
            "records": 1,
        },
        "bank": {
            "source_rows": runner.FIT_HEALTHY_ROWS,
            "bank_sha256": "c" * 64,
            "source_rows_sha256": "d" * 64,
        },
        "calibration": {
            "source_rows": runner.CALIBRATION_HEALTHY_ROWS,
            "source_support_sha256": "e" * 64,
            "S_pred_threshold": 0.1,
            "S_pop_threshold": 0.2,
        },
    }
    return fixture, b0_record, cache_path


def test_b0_cache_identity_accepts_distinct_file_and_head_digests(tmp_path: Path) -> None:
    """The accepted Task23 shape: the raw file digest and the head digest differ."""
    fixture, b0_record, _ = _cache_binding_fixture(tmp_path)
    seed = runner.MODEL_SEEDS[0]
    assert runner._validate_fixture(fixture) is fixture
    assert (
        fixture[runner.PREDICTION_CACHE_FILE_SHA256_KEY][str(seed)]
        != fixture[runner.PREDICTION_CACHE_INNER_SHA256_KEY][str(seed)]
    )
    restored_identity = runner._b0_restore_identity(
        fixture=fixture, seed=seed, b0_record=b0_record,
    )
    identity = restored_identity["identity"]
    assert identity["cache_file_sha256"] == fixture[runner.PREDICTION_CACHE_FILE_SHA256_KEY][str(seed)]
    assert identity["cache_inner_sha256"] == fixture[runner.PREDICTION_CACHE_INNER_SHA256_KEY][str(seed)]
    assert identity["cache_file_sha256"] != identity["cache_inner_sha256"]
    assert identity["cache_records"] == 1
    assert identity["checkpoint_sha256"] == fixture["checkpoints"][str(seed)]["sha256"]
    assert restored_identity["checkpoint_path"] == fixture["checkpoints"][str(seed)]["path"]


def test_b0_cache_identity_refuses_each_tampered_identity_separately(tmp_path: Path) -> None:
    """A swapped, missing or corrupted either identity fails closed on its own gate."""
    seed = runner.MODEL_SEEDS[0]
    key = str(seed)
    fixture, b0_record, cache_path = _cache_binding_fixture(tmp_path)
    file_sha256 = fixture[runner.PREDICTION_CACHE_FILE_SHA256_KEY][key]
    inner_sha256 = fixture[runner.PREDICTION_CACHE_INNER_SHA256_KEY][key]

    file_as_inner = json.loads(json.dumps(fixture))
    file_as_inner[runner.PREDICTION_CACHE_INNER_SHA256_KEY][key] = file_sha256
    with pytest.raises(SystemExit, match="head digest differs from the bound inner digest"):
        runner._b0_restore_identity(fixture=file_as_inner, seed=seed, b0_record=b0_record)

    inner_as_file = json.loads(json.dumps(fixture))
    inner_as_file[runner.PREDICTION_CACHE_FILE_SHA256_KEY][key] = inner_sha256
    with pytest.raises(SystemExit, match="cache bytes differ from the bound file digest"):
        runner._b0_restore_identity(fixture=inner_as_file, seed=seed, b0_record=b0_record)

    record_mismatch = json.loads(json.dumps(b0_record))
    record_mismatch["prediction_cache"]["cache_sha256"] = file_sha256
    with pytest.raises(SystemExit, match="inner digest differs from the accepted B0 record"):
        runner._b0_restore_identity(fixture=fixture, seed=seed, b0_record=record_mismatch)

    cache_path.write_bytes(cache_path.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="cache bytes differ from the bound file digest"):
        runner._b0_restore_identity(fixture=fixture, seed=seed, b0_record=b0_record)


def test_b0_cache_identity_refuses_legacy_single_key_and_checkpoint_drift(tmp_path: Path) -> None:
    seed = runner.MODEL_SEEDS[0]
    key = str(seed)
    fixture, b0_record, _ = _cache_binding_fixture(tmp_path)

    legacy = json.loads(json.dumps(fixture))
    legacy[runner.AMBIGUOUS_PREDICTION_CACHE_KEY] = legacy.pop(
        runner.PREDICTION_CACHE_INNER_SHA256_KEY
    )
    with pytest.raises(SystemExit, match="ambiguous legacy key"):
        runner._validate_fixture(legacy)

    duplicated = json.loads(json.dumps(fixture))
    duplicated[runner.AMBIGUOUS_PREDICTION_CACHE_KEY] = duplicated[
        runner.PREDICTION_CACHE_INNER_SHA256_KEY
    ]
    with pytest.raises(SystemExit, match="ambiguous legacy key"):
        runner._validate_fixture(duplicated)

    missing = json.loads(json.dumps(fixture))
    missing[runner.PREDICTION_CACHE_FILE_SHA256_KEY].pop(key)
    with pytest.raises(SystemExit, match="does not cover exactly the frozen model seeds"):
        runner._validate_fixture(missing)

    malformed = json.loads(json.dumps(fixture))
    malformed[runner.PREDICTION_CACHE_FILE_SHA256_KEY][key] = "not-a-digest"
    with pytest.raises(SystemExit, match="is not a SHA-256 hex digest"):
        runner._validate_fixture(malformed)

    checkpoint = Path(fixture["checkpoints"][key]["path"])
    checkpoint.write_bytes(b"different checkpoint bytes")
    with pytest.raises(SystemExit, match="B0 checkpoint bytes differ from the bound digest"):
        runner._b0_restore_identity(fixture=fixture, seed=seed, b0_record=b0_record)


def _tiny_n4_pools() -> dict[str, list[N4ScaleRecord]]:
    """Tiny deterministic Fit pools whose residual bounds pass the frozen guards."""
    pools: dict[str, list[N4ScaleRecord]] = {}
    for ordinal, history_id in enumerate(HISTORIES):
        generator = torch.Generator().manual_seed(181801 + ordinal)
        pools[history_id] = [
            N4ScaleRecord(
                history_id=history_id,
                file_id=f"probe-{ordinal}-{index:03d}",
                patch_index=0,
                context=torch.randn(128, generator=generator) * 0.01,
                residual=0.08 + 0.04 * index,
            )
            for index in range(6)
        ]
    return pools


def _fit_binding_stub() -> SimpleNamespace:
    """Minimal Fit-role binding for the sampler identity only.

    The device-alignment regression exercises the frozen fit/probe mechanics,
    not the certified DATA binding: ``deterministic_sampler_order`` reads only
    the Fit history roster for the N4 scale workload.
    """
    histories = tuple(
        SimpleNamespace(actual_history_id=history_id, assigned_role="FIT")
        for history_id in HISTORIES
    )
    return SimpleNamespace(
        role_histories=lambda role: tuple(
            item for item in histories if item.assigned_role == role
        )
    )


#: The probe restoration must run on the head's real device. CUDA is the case
#: the CPU-only suites could never reach, so it is exercised when available and
#: honestly skipped otherwise.
DEVICE_CASES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="device-alignment regression needs a real CUDA device",
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICE_CASES)
def test_final_scale_restore_probe_forwards_on_the_live_head_device(
    tmp_path: Path, device: str
) -> None:
    """The strict final-envelope probe must not forward CPU rows into a CUDA head."""
    seed = runner.MODEL_SEEDS[0]
    pools = _tiny_n4_pools()
    records = [record for history_id in HISTORIES for record in pools[history_id]]
    bounds = runner._final_bounds(pools, list(HISTORIES))
    scale = runner.init_n4_scale_model(128, seed)
    result = runner.fit_n4_scale_scope(
        records=records, training_history_ids=list(HISTORIES), scope="final",
        arm=runner._n4_arm(seed, "ab" * 32), binding=_fit_binding_stub(),
        scale_model=scale, v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
        device=device,
    )
    block = runner._save_and_restore_scale(
        scale=scale, out_dir=tmp_path, seed=seed, fit_history_ids=list(HISTORIES),
        bounds=bounds, fit_result={**result, "probe_records": records},
    )
    assert block["restored_forward_max_abs_diff"] == 0.0
    assert block["probe_records"] == len(records)
    assert math.isfinite(block["restored_probe_nll"])
    assert block["restored_state_dict_sha256"] == block["state_dict_sha256"]


@pytest.mark.parametrize("device", DEVICE_CASES)
def test_capacity_fold_restore_probe_forwards_on_the_live_head_device(
    tmp_path: Path, device: str
) -> None:
    """Each fold strict-restore probe must align CPU rows with the CUDA head."""
    seed = runner.MODEL_SEEDS[0]
    block = runner._run_capacity_folds(
        pools=_tiny_n4_pools(), fit_history_ids=list(HISTORIES),
        arm=runner._n4_arm(seed, "ab" * 32), binding=_fit_binding_stub(),
        seed=seed, device=torch.device(device), out_dir=tmp_path,
        certified_folds=[], normative=False,
    )
    assert block["folds_evaluated"] == len(HISTORIES)
    assert block["all_fold_cap_gates_pass"] is True
    assert block["all_fold_payloads_match_certified_task13b"] is None
    assert block["all_fold_custody_authenticated"] is None
    for fold in block["folds"]:
        assert fold["restored_forward_max_abs_diff"] == 0.0
        assert fold["restored_state_dict_sha256"] == fold["scale_state_sha256"]
        assert fold["certified_comparison"] == "not_compared_nonnormative"
        assert fold["certified_payload_matches"] is None
        assert fold["certified_custody_authenticated"] is False
        assert Path(fold["scale_checkpoint_path"]).is_file()


def _compute_probe_pools() -> dict[str, list[N4ScaleRecord]]:
    """Tiny Fit pools of real N4 scale records (600 rows; the probe prefix is 512)."""
    pools: dict[str, list[N4ScaleRecord]] = {}
    for ordinal, history_id in enumerate(HISTORIES):
        generator = torch.Generator().manual_seed(181801)
        pools[history_id] = [
            N4ScaleRecord(
                history_id=history_id,
                file_id=f"compute-{ordinal}-{index:04d}",
                patch_index=index % 7,
                context=torch.randn(128, generator=generator) * 0.01,
                residual=0.05 + 0.0009 * index,
            )
            for index in range(300)
        ]
    return pools


def _compute_config(seed: int) -> V1Config:
    """A real, deliberately small V1 config that matches the frozen B0 arm geometry."""
    return V1Config(
        n_channels=6, patch_size=32, stride=16, d_model=128, sequence_layers=1,
        attention_heads=4, dropout=0.1, n_robots=9, n_programs=8, ema_decay=0.996,
        contrastive_weight_max=0.05, seed=seed,
    )


def _compute_fit_files(seed: int, count: int = 128) -> list[CertifiedSample]:
    """Real tiny normal ``FileSample`` rows in the certified Fit row shape."""
    generator = np.random.default_rng(seed)
    rows: list[CertifiedSample] = []
    for index in range(count):
        sample = FileSample(
            x=(generator.standard_normal((6, 256)) * 0.1).astype(np.float32),
            file_id=f"fit-file-{index:04d}",
            file_label=SampleLabel.NORMAL,
            seed=index,
            generator_version="s18-t30-a13-tiny",
            config_hash="0" * 8,
            regime_sequence=[],
            robot_idx=index % 9,
            program_idx=index % 8,
        )
        rows.append(
            CertifiedSample(
                sample=sample,
                actual_history_id=HISTORIES[index % len(HISTORIES)],
                manifest_sha256="0" * 64,
                shard_path=f"tiny-shard-{index // 64}.npz",
                shard_sha256="1" * 64,
                manifest_row={},
            )
        )
    return rows


def _compute_block_inputs(device: str) -> dict:
    """Complete non-normative ``_compute_block`` input set on a real tiny model.

    The certified DATA/B0 caches are server-only, so this proof uses real
    ``FileSample``/``CertifiedSample`` rows and real ``N4ScaleRecord`` pools
    through the same public constructors, collator, profiler and gates.
    """
    seed = runner.MODEL_SEEDS[0]
    config = _compute_config(seed)
    patchifier = runner._patchifier()
    files = _compute_fit_files(seed)
    pools = _compute_probe_pools()
    return {
        "config": config,
        "model": V1RepresentationModel(config, patchifier=patchifier).to(device).eval(),
        "scale": runner.init_n4_scale_model(128, seed).to(device),
        "arm_n4": runner._n4_arm(seed, "ab" * 32),
        "arm_b0": runner._b0_arm(seed, "cd" * 32),
        "binding": _fit_binding_stub(),
        "fit_rows": files,
        "scale_records": runner._probe_records(pools, list(HISTORIES)),
        "fit_history_ids": list(HISTORIES),
        "probe_row": files[0],
        "probe_history_id": files[0].actual_history_id,
        "bounds": runner._final_bounds(pools, list(HISTORIES)),
        "device": torch.device(device),
        "patchifier": patchifier,
    }


@pytest.mark.parametrize("device", DEVICE_CASES)
def test_compute_block_probes_certified_scale_records_end_to_end(device: str) -> None:
    """The real compute block must fit the certified N4ScaleRecord probe and return its envelope.

    A12 smoke defect: ``_compute_block`` fed Fit *file* rows to
    ``fit_n4_scale_scope`` and failed closed after the bank/calibration/
    development gates. The corrected call executes the real 512-record probe,
    the real paired B0 step, the real profiler and the real envelope gate.
    """
    inputs = _compute_block_inputs(device)
    assert len(inputs["scale_records"]) == 512
    assert all(isinstance(record, N4ScaleRecord) for record in inputs["scale_records"])
    rng_state = torch.get_rng_state()
    try:
        compute = runner._compute_block(**inputs, final_steps=6550)
    finally:
        torch.set_rng_state(rng_state)
    assert compute["scope"] == runner.COMPUTE_SCOPE
    assert compute["envelope_pass"] is True
    assert compute["n4_scale_parameters"] == 4417
    assert compute["n4_scale_fit_steps"] == 6550
    assert compute["n4_scale_fit_flops_total"] > 0
    assert compute["n4_scale_fit_flops_total"] == pytest.approx(
        compute["n4_training_flops_per_fit_file"] * runner.FIT_HEALTHY_ROWS, rel=1e-9
    )
    assert compute["n4_training_flops_per_fit_file"] > 0
    assert compute["b0_training_flops_per_fit_file"] > 0
    assert compute["training_flops_ratio_vs_B0"] == pytest.approx(
        compute["n4_training_flops_per_fit_file"] / compute["b0_training_flops_per_fit_file"]
    )
    assert 0 < compute["training_flops_ratio_vs_B0"] <= 0.25
    assert (
        compute["parameters_inference"]["S_pop"]
        == compute["paired_b0_parameters_inference"]["S_pop"]
    )
    assert (
        compute["parameters_inference"]["S_pred"]
        - compute["paired_b0_parameters_inference"]["S_pred"]
    ) == compute["n4_scale_parameters"]
    for branch in ("S_pred", "S_pop"):
        assert compute["inference_flops_per_fit_file"][branch] > 0
        assert compute["paired_b0_inference_flops_per_fit_file"][branch] > 0
    assert compute["combined_inference_flops_ratio_vs_B0"] <= 1.10


def test_compute_block_refuses_fit_file_rows_in_the_scale_probe() -> None:
    """File rows in the N4 probe must still fail closed (the A12 smoke defect)."""
    inputs = _compute_block_inputs("cpu")
    wrong = dict(inputs)
    wrong["scale_records"] = inputs["fit_rows"]
    rng_state = torch.get_rng_state()
    try:
        with pytest.raises(ValueError, match="N4 fit records must be N4ScaleRecord entries"):
            runner._compute_block(**wrong, final_steps=6550)
    finally:
        torch.set_rng_state(rng_state)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_certified_folds(root: Path) -> list[dict]:
    """Fit one real tiny fold per Fit history and build its certified record.

    Each v3 envelope is written at its provenance path and then re-serialized
    with the identical semantic payload into a different ``torch.save``
    container (legacy format), so the raw file digest changes while the
    canonical payload digest does not. That is the acceptance half of the
    certified-capacity contract: custody by raw digest, identity by payload.
    """
    seed = runner.MODEL_SEEDS[0]
    pools = _tiny_n4_pools()
    records: list[dict] = []
    for index, heldout in enumerate(HISTORIES):
        training_ids = [history_id for history_id in HISTORIES if history_id != heldout]
        train_pool = [record for history_id in training_ids for record in pools[history_id]]
        heldout_pool = list(pools[heldout])
        residuals = np.array([record.residual for record in train_pool], dtype=np.float64)
        bounds = runner.n4_residual_bounds(residuals)
        q05 = float(np.quantile(residuals, 0.05, method="linear"))
        q95 = float(np.quantile(residuals, 0.95, method="linear"))
        scale = runner.init_n4_scale_model(128, seed)
        result = runner.fit_n4_scale_scope(
            records=train_pool, training_history_ids=training_ids,
            scope=f"fold:{heldout}", arm=runner._n4_arm(seed, "ab" * 32),
            binding=_fit_binding_stub(), scale_model=scale,
            v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
            heldout_records=heldout_pool, device="cpu",
        )
        provenance_path = root / "provenance" / f"seed{seed}-fold{index}.pt"
        runner.save_n4_scale_checkpoint(
            provenance_path, scale, fit_scope=f"fold:{heldout}",
            training_history_ids=training_ids,
            v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
            initialization={
                "init_anchor": dict(result["init_anchor"]),
                "init_interiority": dict(result["init_interiority"]),
                "first_step_gradients": dict(result["first_step_gradients"]),
            },
        )
        payload = torch.load(provenance_path, map_location="cpu", weights_only=True)
        historical_path = root / "historical" / provenance_path.name
        historical_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, historical_path, _use_new_zipfile_serialization=False)
        container_sha = _sha256_file(historical_path)
        assert container_sha != _sha256_file(provenance_path)
        records.append(
            {
                "model_seed": seed,
                "fold_index": index,
                "scope": f"fold:{heldout}",
                "heldout_history_id": heldout,
                "training_history_ids": training_ids,
                "checkpoint_path": f"checkpoints/{historical_path.name}",
                "envelope_path": str(historical_path),
                "checkpoint_sha256": container_sha,
                "scale_state_sha256": payload["state_dict_sha256"],
                "checkpoint_schema": payload["schema"],
                "checkpoint_parameterization": payload["parameterization"],
                "scale_parameterization": runner.N4_SCALE_PARAMETERIZATION,
                "guard_contract": payload["guard_contract"],
                "scale_parameters": int(sum(parameter.numel() for parameter in scale.parameters())),
                "record_count": int(result["record_count"]),
                "heldout_count": int(result["heldout_count"]),
                "optimizer_steps": int(result["optimizer_steps"]),
                "per_epoch_steps": int(result["per_epoch_steps"]),
                "bounds": {
                    "v_floor": float(bounds["v_floor"]),
                    "v_cap": float(bounds["v_cap"]),
                },
                "quantiles": {"q05": q05, "q95": q95},
                "init_anchor": dict(result["init_anchor"]),
                "init_interiority": dict(result["init_interiority"]),
                "first_step_gradients": dict(result["first_step_gradients"]),
                "epoch_order_seeds": list(result["epoch_order_seeds"]),
                "order_seed_rule": str(result["order_seed_rule"]),
                "cap_gate_pass_train": bool(result["cap_gate_pass_train"]),
            }
        )
    return records


def _run_certified_tiny_folds(tmp_path: Path, certified_folds: list[dict]) -> dict:
    seed = runner.MODEL_SEEDS[0]
    return runner._run_capacity_folds(
        pools=_tiny_n4_pools(), fit_history_ids=list(HISTORIES),
        arm=runner._n4_arm(seed, "ab" * 32), binding=_fit_binding_stub(),
        seed=seed, device=torch.device("cpu"), out_dir=tmp_path / "recertified",
        certified_folds=certified_folds, normative=True,
    )


def test_capacity_fold_certification_authenticates_payload_not_container_bytes(
    tmp_path: Path,
) -> None:
    """An independently serialized identical payload must be accepted.

    The historical envelopes carry the same canonical scale state and recipe
    in different ``torch.save`` container bytes, so the raw digests differ.
    Treating container bytes as model identity would wrongly refuse this; the
    corrected contract authenticates the file and compares the payload, and it
    records the container difference as custody evidence only.
    """
    certified = _tiny_certified_folds(tmp_path)
    block = _run_certified_tiny_folds(tmp_path, certified)
    assert block["all_fold_custody_authenticated"] is True
    assert block["all_fold_payloads_match_certified_task13b"] is True
    assert block["recertified_containers_equal_certified"] is False
    for fold, record in zip(block["folds"], certified):
        assert fold["certified_container_matches"] is False
        assert fold["certified_payload_matches"] is True
        assert fold["certified_container_sha256"] == record["checkpoint_sha256"]
        assert fold["certified_payload_sha256"] == fold["scale_state_sha256"]
        assert fold["certified_comparison"] == (
            "authenticated_historical_container_and_payload_with_exact_identity"
        )


def test_capacity_fold_certification_refuses_scientific_tamper(tmp_path: Path) -> None:
    """Changed weights, bounds, support or custody are refused, not tolerated."""
    # A coherently re-digested envelope whose scale weights differ: custody and
    # the envelope's own digest authenticate, yet the recertified payload is a
    # different scientific result.
    certified = _tiny_certified_folds(tmp_path / "weights")
    record = certified[0]
    payload = torch.load(record["envelope_path"], map_location="cpu", weights_only=True)
    state = dict(payload["state_dict"])
    name = sorted(state)[0]
    state[name] = state[name] + 1e-3
    payload["state_dict"] = state
    payload["state_dict_sha256"] = runner.state_sha256(state)
    torch.save(payload, record["envelope_path"])
    record["checkpoint_sha256"] = _sha256_file(Path(record["envelope_path"]))
    record["scale_state_sha256"] = payload["state_dict_sha256"]
    with pytest.raises(SystemExit, match="recertified scale state .* differs"):
        _run_certified_tiny_folds(tmp_path / "weights-out", certified)

    # Changed bound (quantile) identity in the certified fold record.
    certified = _tiny_certified_folds(tmp_path / "bounds")
    certified[0]["quantiles"]["q05"] = certified[0]["quantiles"]["q05"] * 1.01
    with pytest.raises(SystemExit, match="recertified quantiles differs"):
        _run_certified_tiny_folds(tmp_path / "bounds-out", certified)

    # Changed qualified-support identity in the certified fold record.
    certified = _tiny_certified_folds(tmp_path / "support")
    certified[0]["record_count"] = certified[0]["record_count"] + 1
    with pytest.raises(SystemExit, match="recertified record_count differs"):
        _run_certified_tiny_folds(tmp_path / "support-out", certified)

    # Substituted historical bytes that no longer authenticate against custody.
    certified = _tiny_certified_folds(tmp_path / "custody")
    Path(certified[0]["envelope_path"]).write_bytes(b"not the certified envelope")
    with pytest.raises(SystemExit, match="do not authenticate against the certified container digest"):
        _run_certified_tiny_folds(tmp_path / "custody-out", certified)

    # Unreachable historical custody root is a stated prerequisite, not a pass.
    certified = _tiny_certified_folds(tmp_path / "missing")
    Path(certified[0]["envelope_path"]).unlink()
    with pytest.raises(SystemExit, match="certified historical envelope is unreachable"):
        _run_certified_tiny_folds(tmp_path / "missing-out", certified)


def _posix_absolute(path: Path) -> str:
    """Render a temp path as the native absolute path used by the test host.

    The certified record holds POSIX server paths. The contract validates an
    absolute directory in the host's native form, so a Windows temp path is
    exercised here while the pinned runtime record remains POSIX.
    """
    return path.resolve().as_posix()


def _nine_certified_envelopes_in_output_dir(tmp_path: Path) -> dict:
    """Lay out nine real tiny envelopes in the certified ``$RUN/full`` layout.

    The recorded run root contains the launch output directory ``full``; the
    nine envelopes live in ``full/checkpoints`` exactly like the A02 server
    custody root. Records are projected the way the lock consumer projects
    them (relative ``checkpoints/...`` paths plus the derived envelope root).
    """
    run_root = tmp_path / "runs" / "cert"
    output_dir = run_root / "full"
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    base = _tiny_certified_folds(tmp_path / "base")
    records: list[dict] = []
    size = 0
    for seed in runner.MODEL_SEEDS:
        for record in base:
            source = Path(record["envelope_path"])
            target = checkpoints / f"seed{seed}-{source.name}"
            target.write_bytes(source.read_bytes())
            size = len(target.read_bytes())
            records.append(
                {
                    **record,
                    "model_seed": seed,
                    "checkpoint_path": f"checkpoints/{target.name}",
                    "envelope_path": _posix_absolute(target),
                    "checkpoint_sha256": _sha256_file(target),
                }
            )
    certification = {
        "evidence": {
            "remote_run_root": f"{_posix_absolute(run_root)}/",
            "remote_full": f"{_posix_absolute(output_dir)}/",
            "remote_retained_server_side": [f"full/checkpoints/*.pt (9 x {size} B)"],
        }
    }
    return {
        "certified": certification,
        "capacity_certification": {
            "envelope_root": capacity.certified_envelope_root(certification),
            "folds": records,
        },
        "run_root": run_root,
        "output_dir": output_dir,
    }


def test_certified_envelope_root_uses_the_recorded_output_directory(
    tmp_path: Path,
) -> None:
    """The real nested ``full/checkpoints`` layout resolves; the run root does not.

    Deriving the envelope directory from ``remote_run_root`` looked one level
    too high, so every certified envelope was reported unreachable even though
    the nine files were present under ``full/checkpoints``. The root must come
    from the recorded launch output directory ``remote_full``.
    """
    layout = _nine_certified_envelopes_in_output_dir(tmp_path)
    derived = layout["capacity_certification"]["envelope_root"]
    assert derived == f"{_posix_absolute(layout['output_dir'])}/"
    assert derived != f"{_posix_absolute(layout['run_root'])}/"
    nested = Path(derived) / "checkpoints"
    assert len(list(nested.glob("*.pt"))) == 9
    # The pre-fix root points at a directory that does not contain the files.
    assert not (layout["run_root"] / "checkpoints").exists()
    with pytest.raises(SystemExit, match="is unreachable at"):
        runner._authenticate_certified_folds(
            capacity_certification={
                "envelope_root": f"{_posix_absolute(layout['run_root'])}/",
                "folds": layout["capacity_certification"]["folds"],
            },
            fit_history_ids=list(HISTORIES),
        )


def test_preflight_authenticates_all_nine_certified_envelopes(tmp_path: Path) -> None:
    """All nine real envelopes are authenticated before any fitting.

    A missing, substituted or wrongly rooted envelope in any seed must fail
    here, ahead of the long Fit campaign, and no fallback path is tried.
    """
    layout = _nine_certified_envelopes_in_output_dir(tmp_path / "accept")
    capacity_certification = layout["capacity_certification"]
    custody = runner._authenticate_certified_folds(
        capacity_certification=capacity_certification,
        fit_history_ids=list(HISTORIES),
    )
    assert custody["authenticated"] is True
    assert custody["folds_authenticated"] == 9
    assert [row["fold_index"] for row in custody["files"]] == [0, 1, 2] * 3
    by_fold = {
        (row["model_seed"], row["fold_index"]): row
        for row in capacity_certification["folds"]
    }
    for row in custody["files"]:
        record = by_fold[(row["model_seed"], row["fold_index"])]
        assert row["envelope_path"] == record["envelope_path"]
        assert row["container_sha256"] == record["checkpoint_sha256"]
        assert row["payload_sha256"] == record["scale_state_sha256"]

    # A substituted file at the exact certified path is refused by raw digest.
    layout = _nine_certified_envelopes_in_output_dir(tmp_path / "substituted")
    substituted = layout["capacity_certification"]
    Path(substituted["folds"][0]["envelope_path"]).write_bytes(
        b"not the certified envelope"
    )
    with pytest.raises(
        SystemExit, match="do not authenticate against the certified container digest"
    ):
        runner._authenticate_certified_folds(
            capacity_certification=substituted,
            fit_history_ids=list(HISTORIES),
        )

    # A content swap that keeps a coherent hash but not the certified payload
    # digest is refused, not accepted as custody.
    layout = _nine_certified_envelopes_in_output_dir(tmp_path / "swapped")
    swapped = layout["capacity_certification"]
    other = Path(swapped["folds"][1]["envelope_path"])
    destination = Path(swapped["folds"][0]["envelope_path"])
    destination.write_bytes(other.read_bytes())
    swapped["folds"][0]["checkpoint_sha256"] = _sha256_file(destination)
    with pytest.raises(SystemExit, match="differs from the certified payload digest"):
        runner._authenticate_certified_folds(
            capacity_certification=swapped,
            fit_history_ids=list(HISTORIES),
        )

    # A missing envelope under any seed fails the whole nine-file preflight.
    layout = _nine_certified_envelopes_in_output_dir(tmp_path / "missing")
    missing = layout["capacity_certification"]
    Path(missing["folds"][-1]["envelope_path"]).unlink()
    with pytest.raises(SystemExit, match="is unreachable at"):
        runner._authenticate_certified_folds(
            capacity_certification=missing,
            fit_history_ids=list(HISTORIES),
        )


def test_saved_scale_envelope_restores_and_scores_through_public_consumer(
    tmp_path: Path,
) -> None:
    """A real saved envelope restores into a fresh head that drives public scoring."""
    from representation import n4_score_files
    from representation.config import V1Config
    from representation.model import V1RepresentationModel
    from representation.sprint18_harness import deterministic_mask_seed
    from synth.config import PatchConfig
    from synth.patchify import Patchifier
    from synth.schema import FileSample, SampleLabel

    seed = runner.MODEL_SEEDS[0]
    pools = _tiny_n4_pools()
    records = [record for history_id in HISTORIES for record in pools[history_id]]
    bounds = runner._final_bounds(pools, list(HISTORIES))
    scale = runner.init_n4_scale_model(128, seed)
    result = runner.fit_n4_scale_scope(
        records=records, training_history_ids=list(HISTORIES), scope="final",
        arm=runner._n4_arm(seed, "ab" * 32), binding=_fit_binding_stub(),
        scale_model=scale, v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
    )
    block = runner._save_and_restore_scale(
        scale=scale, out_dir=tmp_path, seed=seed, fit_history_ids=list(HISTORIES),
        bounds=bounds, fit_result={**result, "probe_records": records},
    )
    checkpoint_path = Path(block["path"])
    assert block["sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    restored = runner.init_n4_scale_model(128, seed + 999999)
    meta = runner.load_n4_scale_checkpoint(
        checkpoint_path, restored, expected_fit_scope="final",
        expected_training_history_ids=list(HISTORIES),
        expected_bounds={"v_floor": bounds["v_floor"], "v_cap": bounds["v_cap"]},
    )
    assert meta["state_dict_sha256"] == block["state_dict_sha256"]

    config = V1Config(
        n_channels=6, patch_size=32, stride=16, d_model=128, attention_heads=4,
        sequence_layers=1, dropout=0.0, n_robots=9, n_programs=8, seed=seed,
        contrastive_weight_max=0.05, ema_decay=0.996,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    sample = FileSample(
        x=np.random.default_rng(181801).normal(size=(6, 640)).astype(np.float32),
        file_id="t30-envelope-00", file_label=SampleLabel.NORMAL, seed=seed,
        generator_version="t30-test", config_hash="t30-test", regime_sequence=[],
        robot_idx=0, program_idx=0,
    )
    sample.validate()
    triples = n4_score_files(
        model=model, samples=[sample], scale_model=restored,
        v_floor=bounds["v_floor"], v_cap=bounds["v_cap"], patchifier=patchifier,
        masking_seed=deterministic_mask_seed(seed, HISTORIES[0], sample.file_id),
    )
    triple = triples[sample.file_id]
    assert set(triple) == {"raw", "scale", "normalized"}
    assert all(math.isfinite(float(value)) for value in triple.values())
    assert float(triple["scale"]) > 0.0


def _tiny_bank_binding(root: Path):
    """Bind three tiny Fit histories and one Calibration history locally.

    File IDs are interleaved across the Fit histories so the certified loader's
    per-history row order differs from the canonical global ``file_id`` order.
    """
    import zipfile

    from representation.sprint18_harness import BoundHistory, Task6Binding
    from synth.dataset import _sample_bytes
    from synth.schema import FileSample, SampleLabel

    generator = "s18-t30-bank-fixture"
    config_hash = "s18-t30-bank-config"
    layout = {
        "S18I-T30-FIT-01": ("FIT", ["t30-bank-02", "t30-bank-00"]),
        "S18I-T30-FIT-02": ("FIT", ["t30-bank-05", "t30-bank-01", "t30-bank-04"]),
        "S18I-T30-FIT-03": ("FIT", ["t30-bank-03"]),
        "S18I-T30-CAL-01": ("CALIBRATION", ["t30-bank-cal-01", "t30-bank-cal-00"]),
    }
    rng = np.random.default_rng(918081)
    signals = {
        file_id: rng.normal(size=(6, 256)).astype(np.float32)
        for _, file_ids in layout.values()
        for file_id in file_ids
    }

    def build(history_id: str, role: str, file_ids: list[str]):
        relative = f"{history_id}/shard-00000.zip"
        shard_path = root / relative
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        items = []
        for offset, file_id in enumerate(file_ids):
            sample = FileSample(
                x=signals[file_id], file_id=file_id, file_label=SampleLabel.NORMAL,
                seed=runner.MODEL_SEEDS[0] + offset, generator_version=generator,
                config_hash=config_hash, regime_sequence=[], robot_idx=offset % 3,
                program_idx=offset % 2,
            )
            sample.validate()
            items.append(sample)
        with zipfile.ZipFile(shard_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in items:
                archive.writestr(f"{item.file_id}.npz", _sample_bytes(item))
        manifest = {
            "format": 1,
            "role": history_id,
            "protocol": "sprint15-benchmark-protocol-v7",
            "config_hash": config_hash,
            "files": [
                {
                    "file_id": item.file_id,
                    "file_label": item.file_label.value,
                    "robot_id": f"robot-{item.robot_idx + 1:02d}",
                    "program_id": f"program-{item.program_idx + 1:02d}",
                    "is_quarantined": False,
                    "is_censored": False,
                    "start_time": 0.0,
                    "end_time": 3600.0,
                    "n_valid_patches": 13,
                }
                for item in items
            ],
            "shards": [
                {
                    "path": relative,
                    "file_ids": [item.file_id for item in items],
                    "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                }
            ],
            "maintenance_windows": {},
        }
        manifest_path = root / f"{history_id}.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return SimpleNamespace(
            history=BoundHistory(
                assigned_role=role,
                actual_history_id=history_id,
                schema_history_id=f"H-TINY-{history_id}",
                manifest_sha256=digest,
                data_seed=32064,
                sample_count=len(items),
                config_hash=config_hash,
                relative_path=f"tiny-fixture/{role}/{history_id}",
                data_permitted_use=role,
                model_permitted_use=role,
            ),
            manifest_path=manifest_path,
            shard_paths={relative: shard_path},
            eligible_ids=tuple(file_ids),
        )

    entries = {
        history_id: build(history_id, role, file_ids)
        for history_id, (role, file_ids) in layout.items()
    }
    fit_entries = [entries[history_id] for history_id in (
        "S18I-T30-FIT-01", "S18I-T30-FIT-02", "S18I-T30-FIT-03",
    )]
    calibration_entry = entries["S18I-T30-CAL-01"]
    manifest_digests = "".join(
        [entry.history.manifest_sha256 for entry in fit_entries]
        + [calibration_entry.history.manifest_sha256]
    )
    binding = Task6Binding(
        candidate_id="S18-T30-A08-ORDERING-TINY",
        data_root_relative="tiny-fixture",
        handoff_sha256=hashlib.sha256(b"s18-t30-a08-bank-handoff").hexdigest(),
        data_binding_raw_sha256=hashlib.sha256(b"s18-t30-a08-bank-raw").hexdigest(),
        data_binding_canonical_sha256=hashlib.sha256(b"s18-t30-a08-bank-canonical").hexdigest(),
        whole16_certification_sha256=hashlib.sha256(b"s18-t30-a08-bank-whole16").hexdigest(),
        qualification_record_sha256=hashlib.sha256(b"s18-t30-a08-bank-qual").hexdigest(),
        manifest_aggregate_sha256=hashlib.sha256(manifest_digests.encode("ascii")).hexdigest(),
        protocol_sha256=hashlib.sha256(b"s18-t30-a08-bank-protocol").hexdigest(),
        mechanism_spec_sha256={},
        histories=tuple(entry.history for entry in entries.values()),
        fit_healthy_rows=sum(len(entry.eligible_ids) for entry in fit_entries),
        calibration_healthy_rows=len(calibration_entry.eligible_ids),
        executable_binding_sha256=hashlib.sha256(manifest_digests.encode("ascii")).hexdigest(),
    )
    return binding, fit_entries, calibration_entry


def test_fit_bank_rows_canonicalize_global_file_id_order(tmp_path: Path) -> None:
    """Per-history or shuffled Fit input must build the accepted canonical bank.

    ``fit_fit_reference_bank`` hashes the embedding matrix in row order and the
    accepted B0 bank was built from globally ``file_id``-sorted rows. This
    regression proves the runner's canonicalization helper yields that exact
    order, identical bank values/digest and identical branch scores and q95
    thresholds for arbitrary input permutations, while the raw per-history
    order (the A07 defect) differs.
    """
    from representation.config import V1Config
    from representation.data import collate_variable_files
    from representation.inference import RepresentationInference
    from representation.model import V1RepresentationModel
    from representation.sprint18_checkpoint import (
        SCHEDULER_ID,
        coefficient_for_epoch,
        deterministic_sampler_order,
        learning_rate_for_epoch,
        restore_sprint18_checkpoint,
        save_sprint18_checkpoint,
        state_sha256,
    )
    from representation.sprint18_harness import (
        BATCH_SIZE,
        _canonical_sha256,
        _encoder_batch,
        deterministic_mask_seed,
        fit_calibration_q95,
        fit_fit_reference_bank,
        load_certified_healthy_role_support,
        load_certified_sample_subset,
        make_sprint18_arm_config,
        score_certified_rows,
    )
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    seed = runner.MODEL_SEEDS[0]
    binding, fit_entries, calibration_entry = _tiny_bank_binding(tmp_path)
    per_history_rows = []
    for entry in fit_entries:
        per_history_rows.extend(load_certified_healthy_role_support(
            binding, entry.history.actual_history_id, entry.manifest_path,
            entry.shard_paths, action="train_fit",
        ))
    calibration_rows = load_certified_sample_subset(
        binding, calibration_entry.history.actual_history_id,
        calibration_entry.manifest_path, calibration_entry.shard_paths,
        list(calibration_entry.eligible_ids),
        role_action="calibrate_q95", verify_all_shards=True,
    )
    assert len(per_history_rows) == binding.fit_healthy_rows
    assert len(calibration_rows) == binding.calibration_healthy_rows

    canonical_ids = sorted(row.file_id for row in per_history_rows)
    assert [row.file_id for row in per_history_rows] != canonical_ids
    canonical = runner._canonical_fit_rows(per_history_rows)
    assert [row.file_id for row in canonical] == canonical_ids
    shuffled = list(per_history_rows)
    random.Random(202608).shuffle(shuffled)
    assert runner._canonical_fit_rows(shuffled) == canonical
    with pytest.raises(SystemExit, match="duplicate file IDs"):
        runner._canonical_fit_rows(per_history_rows + [per_history_rows[0]])

    arm = make_sprint18_arm_config(
        "B0-FULL", seed,
        {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="locked",
        coefficient_lock_sha256="6eb108b2073181d427f1aedfc81fb3bae4c29595108d15de486082800bdd9d6e",
    )
    config = V1Config(
        n_channels=6, patch_size=32, stride=16, d_model=32, attention_heads=4,
        sequence_layers=1, dropout=0.0, n_robots=9, n_programs=8, seed=seed,
        contrastive_weight_max=0.05, ema_decay=0.996,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0
    )
    fit_history_ids = [entry.history.actual_history_id for entry in fit_entries]
    source_ids = sorted(row.file_id for row in per_history_rows)
    sampler_seed, epoch_order = deterministic_sampler_order(
        binding, arm, workload="representation", scope="experiment",
        role_history_ids=fit_history_ids, epoch=1, eligible_source_ids=source_ids,
    )
    checkpoint_path = tmp_path / "tiny-b0-primary.pt"
    digest = save_sprint18_checkpoint(
        checkpoint_path, model, optimizer, arm=arm, binding=binding,
        global_step=0, completed_epoch=0,
        initialization_sha256=state_sha256(model.state_dict()),
        scheduler_state={
            "scheduler_id": SCHEDULER_ID, "completed_epochs": 0, "active_epoch": 1,
            "learning_rate": learning_rate_for_epoch(1), "global_step": 0,
        },
        sampler_state={
            "scope": "experiment", "workload": "representation",
            "role_history_ids": fit_history_ids,
            "role_manifest_sha256": [
                binding.history(history_id).manifest_sha256 for history_id in fit_history_ids
            ],
            "eligible_source_ids": source_ids,
            "eligible_source_ids_sha256": _canonical_sha256(source_ids),
            "sampler_epoch_seed": sampler_seed,
            "epoch_order_ids": list(epoch_order),
            "epoch_order_ids_sha256": _canonical_sha256(list(epoch_order)),
            "eligible_rows": len(source_ids),
            "batch_size": BATCH_SIZE,
            "batches_per_epoch": math.ceil(len(source_ids) / BATCH_SIZE),
            "completed_epochs": 0, "current_epoch": 1,
            "batches_in_current_epoch": 0,
            "fit_role_sha256": _canonical_sha256(
                sorted(item.manifest_sha256 for item in binding.role_histories("FIT"))
            ),
        },
        coefficient_state={
            "epoch": 1, "global_step": 0, "selection_state": "locked",
            "lambda_info": coefficient_for_epoch(0.05, 1),
            "lambda_hcc": None, "alpha": None, "beta": None,
        },
        scope="experiment",
    )
    restored_model = V1RepresentationModel(config, patchifier=patchifier)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0
    )
    restored = restore_sprint18_checkpoint(
        checkpoint_path, restored_model, restored_optimizer, arm=arm,
        binding=binding, allowed_scope="experiment",
    )
    assert restored.checkpoint_sha256 == digest

    bank = fit_fit_reference_bank(
        binding, arm, restored_model, canonical, patchifier,
        checkpoint=restored, scope="experiment",
    )
    bank_replayed = fit_fit_reference_bank(
        binding, arm, restored_model, runner._canonical_fit_rows(shuffled), patchifier,
        checkpoint=restored, scope="experiment",
    )
    bank_raw = fit_fit_reference_bank(
        binding, arm, restored_model, list(per_history_rows), patchifier,
        checkpoint=restored, scope="experiment",
    )

    inference = RepresentationInference(restored_model, bank.bank, patchifier)
    direct = []
    with torch.no_grad():
        for row in canonical:
            mask_seed = deterministic_mask_seed(seed, row.actual_history_id, row.file_id)
            batch = collate_variable_files(
                [row.sample], patchifier, masking_config=config, masking_seed=mask_seed,
            )
            clean = inference.co_locate_batch(_encoder_batch(batch))
            direct.append(restored_model(clean)["file_embedding"].detach().cpu())
    assert torch.equal(bank.bank.embeddings, torch.cat(direct, dim=0))
    assert torch.equal(bank.bank.embeddings, bank_replayed.bank.embeddings)
    assert bank.provenance.bank_sha256 == bank_replayed.provenance.bank_sha256
    assert bank.provenance.source_rows_sha256 == bank_raw.provenance.source_rows_sha256
    assert bank.provenance.bank_sha256 != bank_raw.provenance.bank_sha256

    scores = score_certified_rows(
        binding, arm, restored_model, bank, calibration_rows, patchifier,
        checkpoint=restored, action="calibrate_q95", scope="experiment",
    )
    scores_replayed = score_certified_rows(
        binding, arm, restored_model, bank_replayed, calibration_rows, patchifier,
        checkpoint=restored, action="calibrate_q95", scope="experiment",
    )
    for branch in ("S_pred", "S_pop"):
        assert dict(scores.scores[branch]) == dict(scores_replayed.scores[branch])
    calibration = fit_calibration_q95(binding, scores, calibration_rows, scope="experiment")
    calibration_replayed = fit_calibration_q95(
        binding, scores_replayed, calibration_rows, scope="experiment",
    )
    assert calibration.S_pred.threshold == calibration_replayed.S_pred.threshold
    assert calibration.S_pop.threshold == calibration_replayed.S_pop.threshold
    assert math.isfinite(calibration.S_pred.threshold)
    assert math.isfinite(calibration.S_pop.threshold)
