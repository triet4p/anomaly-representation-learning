"""Focused Task 24 K3-FULL runner tests: paired Development report and lock binding.

Covers the consumer-visible reporting contract only: paired K3/B0 differences
are computed solely on identical frozen model-seed and Development-support
identities, mismatches fail closed, and the K3 arm configuration stays bound
to the Task 14 lock. Synthetic records only; software behavior, not scientific
results.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from representation.sprint18_harness import (
    MODEL_SEEDS,
    arm_config_sha256,
    make_sprint18_arm_config,
)

import sprint18_task24_k3_runner as runner

COEFFS = {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None}
LOCK = "6eb108b2073181d427f1aedfc81fb3bae4c29595108d15de486082800bdd9d6e"
TASK14_LOCK_SHA256 = "c4c1a8bde14310cf4436c10762e3e98475f7c84eb74fcb8a74b3e267885edec3"
HISTORIES = [f"S18I-ITER-0005-DEVELOPMENT-0{index}" for index in range(1, 5)]


def _record(arm_id: str, seed: int, *, offset: float) -> dict:
    """Build one synthetic completed seed record at the frozen schedule shape."""
    per_history = [
        {
            "history_id": history_id,
            "P_AUROC": 0.50 + offset,
            "W_AUROC": 0.52 + offset,
            "PW_AUROC": 0.51 + offset,
            "A_AUROC": 0.40 + offset,
            "P_recall": 0.30 + offset,
            "W_recall": 0.32 + offset,
            "P_median_lead_days": 2.0 + offset,
            "W_median_lead_days": 3.0 + offset,
            "FAR_per_robot_day": 0.02 + offset,
        }
        for history_id in HISTORIES
    ]
    branch = {
        "threshold": 0.10 + offset,
        "macro": {
            "P_macro": 0.60 + offset,
            "W_macro": 0.65 + offset,
            "PW_macro": 0.62 + offset,
            "PW_lcb95": 0.58 + offset,
            "directional_histories": 4,
        },
        "per_history": per_history,
    }
    return {
        "arm_id": arm_id,
        "status": "ok",
        "model_seed": seed,
        "global_step": 4300,
        "representation_epochs_completed": 100,
        "batches_per_epoch": 43,
        "bank": {"source_rows": 5446},
        "calibration": {"source_support_sha256": "cal-support"},
        "development": {
            "branches": {"S_pred": copy.deepcopy(branch), "S_pop": copy.deepcopy(branch)},
            "support": {
                "evaluation_histories": list(HISTORIES),
                "per_history": {history_id: {"support_sha256": "dev-support"} for history_id in HISTORIES},
            },
        },
    }


def _records(arm_id: str, *, offset: float) -> dict[str, dict]:
    return {str(seed): _record(arm_id, seed, offset=offset) for seed in MODEL_SEEDS}


def test_paired_development_reports_signed_deltas_and_means() -> None:
    paired = runner._paired_development(
        k3_records=_records("K3-FULL", offset=0.10),
        b0_records=_records("B0-FULL", offset=0.0),
        b0_records_sha256="b0-records-sha",
    )
    assert paired["schema"] == "s18-t24-k3-paired-development-v1"
    assert paired["arm_id"] == "K3-FULL" and paired["baseline_arm_id"] == "B0-FULL"
    assert paired["b0_seed_records_sha256"] == "b0-records-sha"
    assert paired["support"]["evaluation_histories"] == HISTORIES
    for branch in ("S_pred", "S_pop"):
        block = paired["branches"][branch]
        assert [row["model_seed"] for row in block["per_seed"]] == list(MODEL_SEEDS)
        row = block["per_seed"][0]
        assert row["k3_threshold"] == pytest.approx(0.20)
        assert row["b0_threshold"] == pytest.approx(0.10)
        assert row["macro"]["PW_macro"]["delta"] == pytest.approx(0.10)
        assert row["per_history"][2]["metrics"]["P_AUROC"]["delta"] == pytest.approx(0.10)
        aggregate = block["aggregate"]
        assert aggregate["macro"]["PW_macro"]["mean_delta"] == pytest.approx(0.10)
        assert aggregate["macro"]["PW_macro"]["positive_seeds"] == 3
        assert aggregate["per_history"][HISTORIES[1]]["W_AUROC"]["mean_delta"] == pytest.approx(0.10)
        assert aggregate["per_history"][HISTORIES[1]]["W_AUROC"]["positive_seeds"] == 3


def test_paired_development_rejects_support_mismatch() -> None:
    b0 = _records("B0-FULL", offset=0.0)
    b0[str(MODEL_SEEDS[1])]["development"]["support"]["per_history"][HISTORIES[2]][
        "support_sha256"
    ] = "different-support"
    with pytest.raises(SystemExit, match="support differs"):
        runner._paired_development(
            k3_records=_records("K3-FULL", offset=0.10),
            b0_records=b0,
            b0_records_sha256="b0-records-sha",
        )


def test_paired_development_rejects_wrong_arm_or_incomplete_run() -> None:
    wrong_arm = _records("K3-FULL", offset=0.10)
    with pytest.raises(SystemExit, match="B0-FULL"):
        runner._paired_development(
            k3_records=wrong_arm,
            b0_records=wrong_arm,
            b0_records_sha256="b0-records-sha",
        )
    incomplete = _records("K3-FULL", offset=0.10)
    incomplete[str(MODEL_SEEDS[0])]["representation_epochs_completed"] = 60
    with pytest.raises(SystemExit, match="frozen 100-epoch schedule"):
        runner._paired_development(
            k3_records=incomplete,
            b0_records=_records("B0-FULL", offset=0.0),
            b0_records_sha256="b0-records-sha",
        )


def test_paired_development_rejects_missing_metric() -> None:
    k3 = _records("K3-FULL", offset=0.10)
    k3[str(MODEL_SEEDS[0])]["development"]["branches"]["S_pred"]["per_history"][0][
        "P_AUROC"
    ] = None
    with pytest.raises(SystemExit, match="missing or nonfinite"):
        runner._paired_development(
            k3_records=k3,
            b0_records=_records("B0-FULL", offset=0.0),
            b0_records_sha256="b0-records-sha",
        )


def test_task14_lock_hashes_match_k3_arm_config() -> None:
    hashes, lock_sha256 = runner._task14_lock_config_hashes(REPO_ROOT)
    assert lock_sha256 == TASK14_LOCK_SHA256
    for seed in MODEL_SEEDS:
        arm = make_sprint18_arm_config(
            "K3-FULL", seed, dict(COEFFS), selection_state="locked",
            coefficient_lock_sha256=LOCK,
        )
        assert arm_config_sha256(arm) == hashes[f"K3-FULL/seed-{seed}"]
