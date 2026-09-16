"""Focused Task 23 contract tests: Confirmation matrix lock integrity.

Execution-free: no training, scoring, threshold computation, checkpoint I/O,
or data-root outcome access. Frozen summaries enter as opaque hashed bytes
plus substring-presence checks on config/provenance literals; outcome fields
are never parsed. Confirmation manifests enter only as raw-byte SHA-256
re-hashes; no manifest outcome content is read. Sealed histories are never
touched.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from representation.sprint17_confirmation_lock import (
    ALL_MANIFEST_SHA256,
    ARM_IDS,
    B0_RECORD,
    BINDING_SHA256,
    CALIBRATION_QUANTILE,
    CHECKPOINT_STEP,
    CONFIRMATION_ROOT_SHA256,
    CONFIRMATION_ROOTS,
    EFFECT_GATES,
    K_IDS,
    K_MEMBERS,
    OPTIMIZER_STEPS,
    PROVENANCE_INPUTS,
    QUALIFICATION,
    SELECTED_ARMS,
    SHARED_METRIC_DIGESTS,
    SINGLE_IDS,
    SUPPORT_SHA256,
    TRAIN_FREE_IDS,
    WAIVER_ID,
    WAIVER_SCOPE,
    canonical_hash,
    check_confirmation_path_allowed,
    check_no_sealed_contact,
    lock_sha256,
    materialize_lock,
    validate_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FREEZE_JSON = REPO_ROOT / "experiments" / "sprint17-task23-confirmation-lock.json"

EXPECTED_ARMS = (
    "B0",
    "C2-A", "C2-B", "C3-A", "C3-B", "C5-A", "C5-B", "C6-A",
    "C6-B", "C7-A", "C7-B", "C8-A", "C8-B", "C9-A", "C9-B",
    "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9",
)
EXPECTED_K_MEMBERS = {
    "K1": ("C2-A", "C3-A"),
    "K2": ("C3-A", "C5-A"),
    "K3": ("C5-A", "C6-A"),
    "K4": ("C6-A", "C7-A"),
    "K5": ("C7-A", "C8-A"),
    "K6": ("C8-A", "C9-B"),
    "K7": ("C2-A", "C3-A", "C5-A", "C6-A"),
    "K8": ("C7-A", "C8-A", "C9-B"),
    "K9": ("C2-A", "C3-A", "C5-A", "C6-A", "C7-A", "C8-A", "C9-B"),
}
EXPECTED_CONF_MANIFESTS = {
    "H-S17-V3-CONF-01": "141d79d3e30636fc7b04123f32a74111f1118ab879f2ac913cb7f5182db5b919",
    "H-S17-V3-CONF-02": "b8cc1f74dbf30b052849f1dad47a1a75c2506e070565ed000a3a18bd5d13da85",
    "H-S17-V3-CONF-03": "8826fb4d4b9a8b9d25bed56c4f8d818cc9e82edcb2ca52a910c24437d0bfa7e7",
    "H-S17-V3-CONF-04": "f5042b59c34997aa4effc9974eb92b30c79f4d9a0a5a8c9a84403ff437bfe452",
}


def test_arm_matrix_is_exactly_24_canonical_arms() -> None:
    assert ARM_IDS == EXPECTED_ARMS
    assert len(set(ARM_IDS)) == 24
    lock = materialize_lock()
    assert set(lock["arms"]) == set(EXPECTED_ARMS)
    assert len(lock["arms"]) == 24
    assert SINGLE_IDS == EXPECTED_ARMS[1:15]
    assert K_IDS == EXPECTED_ARMS[15:]
    assert validate_lock(lock) is True


def test_k_members_match_task17_selection_unions() -> None:
    assert K_MEMBERS == EXPECTED_K_MEMBERS
    assert SELECTED_ARMS == {
        "C2": "C2-A", "C3": "C3-A", "C5": "C5-A", "C6": "C6-A",
        "C7": "C7-A", "C8": "C8-A", "C9": "C9-B",
    }
    lock = materialize_lock()
    assert {k: tuple(v) for k, v in lock["k_members"].items()} == EXPECTED_K_MEMBERS
    for k, members in EXPECTED_K_MEMBERS.items():
        assert tuple(lock["arms"][k]["parent_arm_ids"]) == members
        assert set(members) <= set(SELECTED_ARMS.values())


def test_train_free_identity_and_trainable_freshness() -> None:
    lock = materialize_lock()
    b0_ckpts = B0_RECORD["checkpoints"]
    for aid in ("C7-A", "C7-B", "C8-A", "C8-B", "K5", "K6", "K8"):
        assert aid in TRAIN_FREE_IDS
        for seed in ("171701", "171702", "171703"):
            assert lock["arms"][aid]["seeds"][seed]["ckpt_sha256"] == b0_ckpts[seed]
    assert lock["arms"]["C9-A"]["seeds"]["reuses_b0_checkpoints"] is True
    assert lock["arms"]["C9-B"]["seeds"]["reuses_b0_checkpoints"] is True
    for aid in EXPECTED_ARMS:
        if aid == "B0" or aid in TRAIN_FREE_IDS or aid in ("C9-A", "C9-B"):
            continue
        seeds = lock["arms"][aid]["seeds"]
        assert set(seeds) == {"171701", "171702", "171703"}
        for seed, rec in seeds.items():
            assert rec["ckpt_sha256"] != b0_ckpts[seed]
            assert len(rec["ckpt_sha256"]) == 64
        assert len({rec["ckpt_sha256"] for rec in seeds.values()}) == 3


def test_steps_thresholds_and_checkpoint_rule() -> None:
    assert OPTIMIZER_STEPS == 300
    assert CHECKPOINT_STEP == 300
    assert CALIBRATION_QUANTILE == 0.95
    lock = materialize_lock()
    for aid in EXPECTED_ARMS:
        if aid == "B0":
            for seed, th in lock["arms"]["B0"]["thresholds"].items():
                assert th["pred"] > 0 and th["pop"] > 0
            continue
        rec = lock["arms"][aid]
        if "reuses_b0_checkpoints" in rec["seeds"]:
            for seed, th in rec["thresholds_pred"].items():
                assert isinstance(th, float) and th > 0
            continue
        for seed, entry in rec["seeds"].items():
            assert entry["steps"] == 300
            assert entry["thr_pred"] > 0 and entry["thr_pop"] > 0


def test_support_waiver_and_gates_uniform() -> None:
    assert QUALIFICATION == "MEASURABLE_WITH_USER_WAIVER"
    assert WAIVER_ID == "S17-V4-WAIVER-H-S17-V3-DESIGN-03-COHORT_MIX_15_60-P-91-149"
    assert WAIVER_SCOPE["observed_share"] == 91 / 149
    assert EFFECT_GATES == {"min_delta_pw": 0.05, "noninferiority_delta_p": -0.02, "noninferiority_delta_w": -0.02}
    lock = materialize_lock()
    assert lock["qualification"] == QUALIFICATION
    assert lock["support_sha256"] == SUPPORT_SHA256 == "800fc825ac948c3661c2dc727c83083095a796df8950c94922dafabae318f59a"
    assert lock["binding_sha256"] == BINDING_SHA256
    for aid in EXPECTED_ARMS:
        rec = lock["arms"][aid]
        assert rec["status"] == "VALID_NEGATIVE"
        assert rec["support_sha256"] == SUPPORT_SHA256
    assert lock["no_confirmation_scoring"] is True
    assert lock["sealed_contact"] is False
    assert lock["confirmation_scoring_started"] is False
    assert lock["task24_started"] is False


def test_confirmation_roots_manifest_bytes_only() -> None:
    assert set(CONFIRMATION_ROOTS) == set(EXPECTED_CONF_MANIFESTS)
    for hid, expected in EXPECTED_CONF_MANIFESTS.items():
        rec = CONFIRMATION_ROOTS[hid]
        assert rec["manifest_sha256"] == expected
        assert rec["manifest_sha256"] == ALL_MANIFEST_SHA256[hid]
        raw = (REPO_ROOT / rec["directory"] / "manifest.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        assert len(raw) == rec["manifest_bytes"]
        assert sorted(p.name for p in (REPO_ROOT / rec["directory"]).iterdir()) == ["files", "manifest.json"]
        check_confirmation_path_allowed(rec["directory"] + "/manifest.json")
    assert canonical_hash(sorted(EXPECTED_CONF_MANIFESTS.values())) == CONFIRMATION_ROOT_SHA256
    lock = materialize_lock()
    assert lock["confirmation_root_sha256"] == CONFIRMATION_ROOT_SHA256
    assert lock["confirmation_access"] == "manifest_bytes_only"


def test_no_sealed_contact_anywhere() -> None:
    lock = materialize_lock()
    raw = json.dumps(lock, sort_keys=True, separators=(",", ":"))
    assert "H-SEAL" not in raw
    assert "SEALED" not in raw
    module_text = (REPO_ROOT / "src" / "representation" / "sprint17_confirmation_lock.py").read_text(encoding="utf-8")
    # Guard-code literals ("H-SEAL"/"SEALED" inside check_no_sealed_contact) are
    # code, not history contact: assert no sealed history ID or sealed path.
    for sealed_path in ("H-SEAL-37", "H-SEAL-38", "H-SEAL-39", "H-SEAL-40", "SEALED/", "sprint15-v7/SEALED"):
        assert sealed_path not in module_text
        assert sealed_path not in raw
    freeze_text = FREEZE_JSON.read_text(encoding="utf-8")
    assert "H-SEAL" not in freeze_text
    check_no_sealed_contact(list(lock["provenance_inputs"]))
    with pytest.raises(ValueError):
        check_no_sealed_contact(["data/generated/sprint15-v7/SEALED/H-SEAL-37"])
    with pytest.raises(ValueError):
        check_confirmation_path_allowed("data/generated/sprint17-ablation-v3/CONFIRMATION/H-S17-V3-CONF-01/scores.json")
    with pytest.raises(ValueError):
        check_confirmation_path_allowed("data/generated/sprint17-ablation-v3/DEVELOPMENT/H-S17-V3-DEV-01/manifest.json")


def test_no_outcome_content_in_lock() -> None:
    lock = materialize_lock()
    raw = json.dumps(lock, sort_keys=True, separators=(",", ":"))
    for bad in ("AUROC", "macro_pw", "delta_vs_B0", "per_history", "per_seed", "recall", "failure_events", "episodes"):
        assert bad not in raw


def test_provenance_inputs_match_working_tree_opaque_bytes() -> None:
    assert len(PROVENANCE_INPUTS) == 14
    for rel, expected in PROVENANCE_INPUTS.items():
        raw = (REPO_ROOT / rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        assert "CONFIRMATION" not in rel and "SEAL" not in rel
    for rel, expected in SHARED_METRIC_DIGESTS.items():
        raw = (REPO_ROOT / rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected


def test_locked_config_literals_appear_in_frozen_summaries() -> None:
    lock = materialize_lock()
    b0_text = (REPO_ROOT / "experiments" / "sprint17-task7-b0-summary.json").read_text(encoding="utf-8")
    for seed, ckpt in B0_RECORD["checkpoints"].items():
        assert ckpt in b0_text
    checked = 0
    for aid in SINGLE_IDS + K_IDS:
        rec = lock["arms"][aid]
        text = (REPO_ROOT / rec["summary_file"]).read_text(encoding="utf-8")
        assert rec["metric_code_sha256"] in text
        assert rec["cache_sha256"] in text
        seeds = rec["seeds"]
        if "reuses_b0_checkpoints" in seeds:
            for th in rec["thresholds_pred"].values():
                assert repr(th) in text
                checked += 1
            continue
        for seed, entry in seeds.items():
            assert entry["ckpt_sha256"] in text
            assert repr(entry["thr_pred"]) in text
            assert repr(entry["thr_pop"]) in text
            checked += 3
    assert checked > 100


def test_freeze_bytes_are_deterministic_rematerialization() -> None:
    lock = materialize_lock()
    assert validate_lock(lock) is True
    digest = lock_sha256(lock)
    lock["lock_sha256"] = digest
    assert validate_lock(lock) is True
    canonical = (json.dumps(lock, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    assert FREEZE_JSON.read_bytes() == canonical
    reparsed = json.loads(FREEZE_JSON.read_text(encoding="utf-8"))
    assert reparsed["lock_sha256"] == digest
    assert validate_lock(reparsed) is True


def test_no_unregistered_confirmation_scores_and_task25_not_started() -> None:
    # Post-Task-25 revision: the Task 24 and Task 25 confirmation summaries
    # are registered reviewed freeze artifacts alongside the Task 23 lock
    # (all allowlisted); the Task 24/25 records assert the one-shot/no-rerun/
    # no-Sealed ledgers. Task 26 remains not started.
    for evdir in ["task7-evidence", "task10-evidence", "task11-evidence", "task12-evidence", "task13-evidence", "task14-evidence", "task15-evidence", "task16-evidence", "task20-evidence", "task21-evidence"]:
        d = REPO_ROOT / "artifacts" / "sprint-17" / evdir
        if not d.exists():
            continue
        for p in d.rglob("*"):
            assert "conf" not in p.name.lower(), p
    for p in (REPO_ROOT / "experiments").glob("*"):
        if "confirmation" in p.name.lower():
            assert p.name in ("sprint17-task23-confirmation-lock.json",
                             "sprint17-task24-confirmation-summary.json",
                             "sprint17-task25-confirmation-summary.json"), p
    for name in ("task-26.md", "review-batch-e.md"):
        assert not (REPO_ROOT / "artifacts" / "sprint-17" / name).exists(), name


def test_validate_lock_fails_closed_on_mutations() -> None:
    lock = materialize_lock()
    tampered = copy.deepcopy(lock)
    tampered_arms = dict(tampered["arms"])
    del tampered_arms["K9"]
    tampered["arms"] = tampered_arms
    with pytest.raises(ValueError):
        validate_lock(tampered)
    tampered = copy.deepcopy(lock)
    tampered["arms"]["K9"]["parent_arm_ids"] = ["C2-A", "C3-A", "C5-A", "C6-A", "C7-A", "C8-A", "C9-A"]
    with pytest.raises(ValueError):
        validate_lock(tampered)
    tampered = copy.deepcopy(lock)
    tampered["confirmation_roots"]["H-S17-V3-CONF-01"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_lock(tampered)
    tampered = copy.deepcopy(lock)
    tampered["arms"]["C5-A"]["seeds"]["171701"]["thr_pred"] = 0.0
    with pytest.raises(ValueError):
        validate_lock(tampered)
    tampered = copy.deepcopy(lock)
    tampered["arms"]["K7"]["seeds"]["171701"]["ckpt_sha256"] = B0_RECORD["checkpoints"]["171701"]
    with pytest.raises(ValueError):
        validate_lock(tampered)
    tampered = copy.deepcopy(lock)
    tampered["confirmation_scoring_started"] = True
    with pytest.raises(ValueError):
        validate_lock(tampered)
    tampered = copy.deepcopy(lock)
    tampered["lock_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        validate_lock(tampered)
