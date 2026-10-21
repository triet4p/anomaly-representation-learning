"""Focused Task 25 K7-FULL runner tests: paired Development report, lock binding, batches.

Covers the consumer-visible reporting and training-input contract only:
paired K7/B0 differences are computed solely on identical frozen model-seed
and Development-support identities, mismatches fail closed, the K7 arm
configuration stays bound to the Task 14 lock, and K7 epoch batches use the
arm-owned C2-A patchifier with the frozen C5-A channel-time block mask.
Synthetic records and the Task 20 tiny real-schema binding only; software
behavior, not scientific results.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "representation"))

from representation.data import collate_variable_files
from representation.sprint17_c2 import Sprint17C2Patchifier
from representation.sprint18_harness import (
    BATCH_SIZE,
    MODEL_SEEDS,
    arm_config_sha256,
    make_sprint18_arm_config,
)
from representation.sprint18_k_dispatch import (
    build_k_config,
    masking_policy_for_arm,
    members_for_arm,
)

import sprint18_task25_k7_runner as runner

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
        arm_records=_records("K7-FULL", offset=0.10),
        b0_records=_records("B0-FULL", offset=0.0),
        b0_records_sha256="b0-records-sha",
    )
    assert paired["schema"] == "s18-t25-k7-paired-development-v1"
    assert paired["task_id"] == "S18-T25" and paired["attempt_id"] == "S18-T25-A01"
    assert paired["arm_id"] == "K7-FULL" and paired["baseline_arm_id"] == "B0-FULL"
    assert paired["b0_seed_records_sha256"] == "b0-records-sha"
    assert paired["support"]["evaluation_histories"] == HISTORIES
    for branch in ("S_pred", "S_pop"):
        block = paired["branches"][branch]
        assert [row["model_seed"] for row in block["per_seed"]] == list(MODEL_SEEDS)
        row = block["per_seed"][0]
        assert row["arm_threshold"] == pytest.approx(0.20)
        assert row["b0_threshold"] == pytest.approx(0.10)
        assert row["macro"]["PW_macro"]["arm"] == pytest.approx(0.72)
        assert row["macro"]["PW_macro"]["b0"] == pytest.approx(0.62)
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
            arm_records=_records("K7-FULL", offset=0.10),
            b0_records=b0,
            b0_records_sha256="b0-records-sha",
        )


def test_paired_development_rejects_wrong_arm_or_incomplete_run() -> None:
    # A K3 record must never be reported as the K7 arm.
    with pytest.raises(SystemExit, match="K7 seed"):
        runner._paired_development(
            arm_records=_records("K3-FULL", offset=0.10),
            b0_records=_records("B0-FULL", offset=0.0),
            b0_records_sha256="b0-records-sha",
        )
    with pytest.raises(SystemExit, match="B0-FULL"):
        runner._paired_development(
            arm_records=_records("K7-FULL", offset=0.10),
            b0_records=_records("K7-FULL", offset=0.0),
            b0_records_sha256="b0-records-sha",
        )
    incomplete = _records("K7-FULL", offset=0.10)
    incomplete[str(MODEL_SEEDS[0])]["representation_epochs_completed"] = 60
    with pytest.raises(SystemExit, match="frozen 100-epoch schedule"):
        runner._paired_development(
            arm_records=incomplete,
            b0_records=_records("B0-FULL", offset=0.0),
            b0_records_sha256="b0-records-sha",
        )


def test_paired_development_rejects_missing_metric() -> None:
    arm = _records("K7-FULL", offset=0.10)
    arm[str(MODEL_SEEDS[0])]["development"]["branches"]["S_pred"]["per_history"][0][
        "P_AUROC"
    ] = None
    with pytest.raises(SystemExit, match="missing or nonfinite"):
        runner._paired_development(
            arm_records=arm,
            b0_records=_records("B0-FULL", offset=0.0),
            b0_records_sha256="b0-records-sha",
        )


def test_task14_lock_hashes_match_k7_arm_config() -> None:
    hashes, lock_sha256 = runner._task14_lock_config_hashes(REPO_ROOT)
    assert lock_sha256 == TASK14_LOCK_SHA256
    assert runner.ARM_ID == "K7-FULL"
    assert members_for_arm(runner.ARM_ID) == ("C2-A", "C3-A", "C5-A", "C6-A")
    for seed in MODEL_SEEDS:
        arm = make_sprint18_arm_config(
            "K7-FULL", seed, dict(COEFFS), selection_state="locked",
            coefficient_lock_sha256=LOCK,
        )
        assert arm_config_sha256(arm) == hashes[f"K7-FULL/seed-{seed}"]
        # The lock entry is arm-specific: a K3 config cannot satisfy a K7 entry.
        k3_arm = make_sprint18_arm_config(
            "K3-FULL", seed, dict(COEFFS), selection_state="locked",
            coefficient_lock_sha256=LOCK,
        )
        assert arm_config_sha256(k3_arm) != hashes[f"K7-FULL/seed-{seed}"]


def test_k7_runner_batches_use_arm_patchifier_and_frozen_c5a_policy(tmp_path) -> None:
    import test_sprint18_k_dispatch as tiny

    binding, histories = tiny._tiny_certified_binding(tmp_path)
    fit_rows = tiny._tiny_certified_fit_rows(binding, histories)
    fit_ids = sorted(row.file_id for row in fit_rows)
    sample_by_id = {row.file_id: row.sample for row in fit_rows}
    fit_history_ids = [
        item.history.actual_history_id
        for item in histories.values()
        if item.history.assigned_role == "FIT"
    ]
    cfg = build_k_config("K7-FULL", MODEL_SEEDS[0], d_model=16)
    arm = make_sprint18_arm_config(
        "K7-FULL", MODEL_SEEDS[0], dict(COEFFS), selection_state="locked",
        coefficient_lock_sha256=LOCK,
    )
    from representation.sprint18_k_dispatch import build_k_model

    _, patchifier = build_k_model("K7-FULL", cfg)
    assert isinstance(patchifier, Sprint17C2Patchifier)
    order, batches = runner._build_epoch_batches(
        binding=binding, arm=arm, epoch=1, config=cfg, patchifier=patchifier,
        fit_ids=fit_ids, fit_sample_by_id=sample_by_id, fit_history_ids=fit_history_ids,
    )
    assert sorted(order) == fit_ids
    assert len(batches) == math.ceil(len(fit_ids) / BATCH_SIZE) == 1
    policy = masking_policy_for_arm("K7-FULL")
    ordered = [sample_by_id[file_id] for file_id in order]
    expected = collate_variable_files(
        ordered, patchifier, masking_config=cfg, masking_seed=1, masking_policy=policy,
    )
    default = collate_variable_files(
        ordered, patchifier, masking_config=cfg, masking_seed=1,
    )
    assert torch.equal(batches[0]["mask"], expected["mask"])
    assert not torch.equal(batches[0]["mask"], default["mask"])
