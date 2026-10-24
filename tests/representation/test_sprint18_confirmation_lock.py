"""Fail-closed tests for the Sprint 18 Task 33 Confirmation lock."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from representation import sprint18_confirmation_lock as L


@pytest.fixture(scope="module")
def lock() -> dict[str, object]:
    return L.materialize_lock()


@pytest.fixture()
def mutable_lock(lock: dict[str, object]) -> dict[str, object]:
    return copy.deepcopy(lock)


def _rebind_digest(lock: dict[str, object]) -> None:
    lock["lock_sha256"] = L.lock_sha256(lock)


def _custody_report(lock: dict[str, object]) -> dict[str, object]:
    custody = lock["artifact_custody"]
    assert isinstance(custody, dict)
    return {
        "schema_id": "sprint18-confirmation-lock-custody-v1",
        "remote_worktree": {
            "path": L.SUPPLEMENTARY_SOURCE_WORKTREE,
            "branch": L.SUPPLEMENTARY_SOURCE_BRANCH,
            "commit": L.SUPPLEMENTARY_SOURCE_COMMIT,
            "tree": L.SUPPLEMENTARY_SOURCE_TREE,
        },
        "artifacts": dict(custody["remote_artifacts"]),
    }


def test_lock_materializes_and_validates(lock: dict[str, object]) -> None:
    summary = L.validate_lock(lock)
    assert summary["status"] == "VALID"
    assert summary["arms"] == 8
    assert summary["executable_arms"] == 7
    assert summary["seeds_per_executable_arm"] == 3
    assert summary["n1_state"] == "STOPPED"
    assert summary["confirmation_open"] is False
    assert summary["lock_sha256"] == lock["lock_sha256"]


def test_lock_digest_excludes_only_itself(lock: dict[str, object]) -> None:
    payload = {key: value for key, value in lock.items() if key != "lock_sha256"}
    assert L.canonical_hash(payload) == lock["lock_sha256"]
    assert L.lock_sha256(lock) == lock["lock_sha256"]


def test_eight_arms_and_explicit_n1_stopped_state(lock: dict[str, object]) -> None:
    arms = lock["arms"]
    assert isinstance(arms, dict)
    assert list(arms) == list(L.ARM_IDS)
    n1 = arms["N1-HCC"]
    assert n1["status"] == "STOPPED"
    assert n1["executable"] is False
    assert n1["selected_coefficient"] is None
    assert n1["training_authorized"] is False
    assert n1["seed_runs"] == {}
    assert n1["checkpoints"] == []
    assert n1["fit_bank"] is None
    assert n1["calibration"] is None
    for arm_id in L.EXECUTABLE_ARM_IDS:
        entry = arms[arm_id]
        assert entry["status"] == "VALID"
        assert list(entry["seeds"]) == [str(seed) for seed in L.MODEL_SEEDS]


def test_n4_uses_distinct_v9_authority_and_paired_b0(lock: dict[str, object]) -> None:
    arms = lock["arms"]
    assert isinstance(arms, dict)
    n4 = arms["N4-CRN"]
    assert n4["scientific_authority"]["task14_lock_sha256"] == L.N4_SCIENTIFIC_LOCK_SHA256
    assert n4["scientific_authority"]["task14_lock_sha256"] != L.TASK14_LOCK_SHA256
    authority = lock["n4_authority"]
    assert isinstance(authority, dict)
    assert authority["scientific_source_commit"] == L.N4_SCIENTIFIC_SOURCE_COMMIT
    assert authority["excluded_authority"]["v11_report_only_lock_sha256"] == L.N4_V11_REPORT_ONLY_LOCK_SHA256
    b0 = arms["B0-FULL"]["seeds"]
    for seed in L.MODEL_SEEDS:
        seed_key = str(seed)
        paired = authority["per_seed"][seed_key]
        assert paired["paired_b0_arm_config_sha256"] == b0[seed_key]["arm_config_sha256"]
        assert paired["paired_b0_checkpoint_sha256"] == b0[seed_key]["checkpoint"]["sha256"]
        assert paired["S_pop_threshold"] == b0[seed_key]["calibration"]["S_pop_threshold"]
        assert paired["S_pred_threshold"] == n4["seeds"][seed_key]["calibration"]["S_pred_threshold"]


def test_confirmation_support_is_metadata_only(lock: dict[str, object]) -> None:
    support = lock["support"]
    assert isinstance(support, dict)
    confirmation = support["confirmation"]
    assert confirmation["contact_state"] == "NOT_ACCESSED"
    assert confirmation["support_sha256"] is None
    assert "not precomputed" in confirmation["support_sha256_policy"]
    assert list(confirmation["per_history"]) == list(L.CONFIRMATION_HISTORY_IDS)
    for record in confirmation["per_history"].values():
        assert len(record["manifest_sha256"]) == 64
        assert len(record["waveform_stream_sha256"]) == 64
        assert "model_permitted_use" in record
        assert "data_permitted_use" in record


def test_arm_state_consumer_fails_closed(lock: dict[str, object]) -> None:
    state = L.arm_state(lock, "B0-FULL", 181801)
    assert state["checkpoint"]["sha256"] == "9292d7c59037cba7b7fd337dbb896ed492777abff4713307c98f86a3b5226a8f"
    with pytest.raises(L.ConfirmationLockError):
        L.arm_state(lock, "N1-HCC", 181801)
    with pytest.raises(L.ConfirmationLockError):
        L.arm_state(lock, "B0-FULL", 181899)
    support = L.confirmation_support(lock, L.CONFIRMATION_HISTORY_IDS[0])
    assert support["schema_history_id"] == "H-S18-CONF-01"
    with pytest.raises(L.ConfirmationLockError):
        L.confirmation_support(lock, "H-S18-CONF-99")


def test_tampered_checkpoint_hash_fails_closed(
    mutable_lock: dict[str, object],
) -> None:
    seeded = mutable_lock["arms"]["B0-FULL"]["seeds"]["181801"]
    seeded["checkpoint"]["sha256"] = "0" * 64
    _rebind_digest(mutable_lock)
    with pytest.raises(L.ConfirmationLockError):
        L.validate_lock(mutable_lock)


def test_missing_seed_fails_closed(mutable_lock: dict[str, object]) -> None:
    del mutable_lock["arms"]["K3-FULL"]["seeds"]["181803"]
    _rebind_digest(mutable_lock)
    with pytest.raises(L.ConfirmationLockError):
        L.validate_lock(mutable_lock)


def test_n1_invented_executable_config_fails_closed(
    mutable_lock: dict[str, object],
) -> None:
    n1 = mutable_lock["arms"]["N1-HCC"]
    n1["executable"] = True
    n1["status"] = "VALID"
    n1["seeds"] = {"181801": {"checkpoint": {"sha256": "0" * 64}}}
    _rebind_digest(mutable_lock)
    with pytest.raises(L.ConfirmationLockError):
        L.validate_lock(mutable_lock)


def test_n4_b0_parity_mismatch_fails_closed(
    mutable_lock: dict[str, object],
) -> None:
    mutable_lock["n4_authority"]["per_seed"]["181801"]["S_pop_threshold"] = 1.0
    _rebind_digest(mutable_lock)
    with pytest.raises(L.ConfirmationLockError):
        L.validate_lock(mutable_lock)


def test_outcome_field_injection_fails_closed(
    mutable_lock: dict[str, object],
) -> None:
    mutable_lock["arms"]["B0-FULL"]["seeds"]["181801"]["P_AUROC"] = 0.91
    _rebind_digest(mutable_lock)
    with pytest.raises(L.ConfirmationLockError, match="outcome field is forbidden"):
        L.validate_lock(mutable_lock)


def test_unknown_top_level_key_fails_closed(
    mutable_lock: dict[str, object],
) -> None:
    mutable_lock["extra"] = True
    _rebind_digest(mutable_lock)
    with pytest.raises(L.ConfirmationLockError, match="top-level keys differ"):
        L.validate_lock(mutable_lock)


def test_custody_report_missing_artifact_fails_closed(
    lock: dict[str, object], tmp_path: Path
) -> None:
    report = _custody_report(lock)
    report["artifacts"].pop(next(iter(report["artifacts"])))
    path = tmp_path / "custody.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(L.ConfirmationLockError, match="remote custody is incomplete"):
        L.validate_lock(lock, custody_report=path)


def test_custody_report_mismatch_fails_closed(
    lock: dict[str, object], tmp_path: Path
) -> None:
    report = _custody_report(lock)
    first = next(iter(report["artifacts"]))
    report["artifacts"][first] = "0" * 64
    path = tmp_path / "custody.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(L.ConfirmationLockError, match="remote custody digest mismatch"):
        L.validate_lock(lock, custody_report=path)


def test_custody_report_worktree_mismatch_fails_closed(
    lock: dict[str, object], tmp_path: Path
) -> None:
    report = _custody_report(lock)
    report["remote_worktree"]["commit"] = "0" * 40
    path = tmp_path / "custody.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(L.ConfirmationLockError, match="isolated worktree commit"):
        L.validate_lock(lock, custody_report=path)


def test_complete_custody_report_passes(
    lock: dict[str, object], tmp_path: Path
) -> None:
    report = _custody_report(lock)
    path = tmp_path / "custody.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    summary = L.validate_lock(lock, custody_report=path)
    assert summary["verified_remote_artifacts"] == summary["remote_artifacts"]


def test_remote_custody_required_without_report_fails_closed(
    lock: dict[str, object],
) -> None:
    with pytest.raises(L.ConfirmationLockError, match="custody report is required"):
        L.validate_lock(lock, require_remote_custody=True)


def test_local_evidence_drift_fails_closed(
    mutable_lock: dict[str, object],
) -> None:
    mutable_lock["artifact_custody"]["local_artifacts"][L.DEVELOPMENT_MATRIX_PATH] = "0" * 64
    _rebind_digest(mutable_lock)
    with pytest.raises(L.ConfirmationLockError):
        L.validate_lock(mutable_lock)
