"""Focused Task 19 contract tests: K1–K9 freeze against substitution drift.

Execution-free: no data, checkpoint, GPU, Confirmation, or Sealed access.
Summaries enter only as opaque hashed bytes; outcome fields are never parsed.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from representation.sprint17_ablation import (
    QUALIFICATION as HARNESS_QUALIFICATION,
    WAIVER_ID as HARNESS_WAIVER_ID,
    WAIVER_SCOPE as HARNESS_WAIVER_SCOPE,
    get_arm,
)
from representation.sprint17_combinations import (
    B0_FLOPS_REFERENCE,
    B0_PARAMS,
    BINDING_SHA256,
    CALIBRATION_QUANTILE,
    CHECKPOINT_STEP,
    GLOBAL_SETTINGS,
    K_MEMBERS,
    MEMBER_IMPLEMENTATION,
    MODEL_SEEDS,
    OPTIMIZER_STEPS,
    QUALIFICATION,
    REQUIRED_PROVENANCE_FILES,
    SELECTED_ARMS,
    SUPPORT_SHA256,
    WAIVER_ID,
    WAIVER_SCOPE,
    canonical_hash,
    materialize_all,
    materialize_combination,
    members_for,
    selected_arm_for_component,
    validate_frozen_combination,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FREEZE_JSON = REPO_ROOT / "experiments" / "sprint17-task19-combinations.json"

EXPECTED_MEMBERS = {
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
UNSELECTED_ARMS = ("C2-B", "C3-B", "C5-B", "C6-B", "C7-B", "C8-B", "C9-A")
EXPECTED_NUMERICS = {
    "C2-A": {"patch_size": 32, "stride": 16, "grids": [[0, 16]], "pad_end": True},
    "C3-A": {"local_encoder_params_at_6_128": 34688, "b0_local_encoder_params": 35712},
    "C5-A": {"total_mask_ratio": 0.40, "n_blocks": 2, "horizons": [0, 1, 2]},
    "C6-A": {"pooling_params_at_128": 32896},
    "C7-A": {"min_cell_rows": 8, "shrinkage_floor": 0.05, "knn_k": 5},
    "C8-A": {"huber_delta": 1.0, "mad_floor": 1e-06},
    "C9-B": {"window_duration_timesteps": 64},
}


def _dummy_provenance() -> dict[str, str]:
    return {path: "0" * 64 for path in REQUIRED_PROVENANCE_FILES}


def test_k_membership_matches_batch_contract_and_registry() -> None:
    assert K_MEMBERS == EXPECTED_MEMBERS
    assert set(materialize_all(_dummy_provenance())) == set(EXPECTED_MEMBERS)
    for k_id, members in EXPECTED_MEMBERS.items():
        assert members_for(k_id) == members
        assert tuple(get_arm(k_id).components) == tuple(
            MEMBER_IMPLEMENTATION[a]["component"] for a in members
        )


def test_members_are_exactly_the_task17_selected_arms() -> None:
    assert SELECTED_ARMS == {
        "C2": "C2-A",
        "C3": "C3-A",
        "C5": "C5-A",
        "C6": "C6-A",
        "C7": "C7-A",
        "C8": "C8-A",
        "C9": "C9-B",
    }
    used = {a for members in K_MEMBERS.values() for a in members}
    assert used == set(SELECTED_ARMS.values())
    for arm in UNSELECTED_ARMS:
        assert all(arm not in members for members in K_MEMBERS.values())
    for component, arm in SELECTED_ARMS.items():
        assert selected_arm_for_component(component) == arm
    with pytest.raises(ValueError, match="unknown Sprint 17 combination"):
        members_for("K10")
    with pytest.raises(ValueError, match="unknown Sprint 17 suspect component"):
        selected_arm_for_component("C1")


def test_no_duplicate_omitted_or_singleton_members() -> None:
    for k_id, members in K_MEMBERS.items():
        assert len(members) >= 2
        assert len(set(members)) == len(members)
        components = [MEMBER_IMPLEMENTATION[a]["component"] for a in members]
        assert len(set(components)) == len(components)
    doc = materialize_combination("K1", _dummy_provenance())
    tampered = copy.deepcopy(doc)
    tampered["member_arm_ids"] = ["C2-A", "C2-A"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_frozen_combination(tampered)
    tampered = copy.deepcopy(doc)
    tampered["member_arm_ids"] = ["C2-A", "C3-B"]
    with pytest.raises(ValueError, match="substitute"):
        validate_frozen_combination(tampered)


def test_no_combination_specific_override_surface() -> None:
    assert list(inspect.signature(materialize_combination).parameters) == [
        "k_id",
        "provenance_inputs",
    ]
    doc = materialize_combination("K9", _dummy_provenance())
    tampered = copy.deepcopy(doc)
    tampered["global"] = copy.deepcopy(GLOBAL_SETTINGS)
    tampered["global"]["optimizer_steps"] = 301
    with pytest.raises(ValueError, match="tuning is forbidden"):
        validate_frozen_combination(tampered)
    tampered = copy.deepcopy(doc)
    tampered["extra_tuning"] = {"lr": 0.01}
    with pytest.raises(ValueError, match="exactly the frozen top-level fields"):
        validate_frozen_combination(tampered)
    tampered = copy.deepcopy(doc)
    tampered["members"]["C2-A"] = copy.deepcopy(tampered["members"]["C2-A"])
    tampered["members"]["C2-A"]["frozen_numerics"] = {"patch_size": 64}
    with pytest.raises(ValueError, match="overrides the frozen implementation"):
        validate_frozen_combination(tampered)


def test_global_settings_identical_across_all_k() -> None:
    docs = materialize_all(_dummy_provenance())
    assert len(docs) == 9
    for doc in docs.values():
        assert doc["global"] == GLOBAL_SETTINGS
    assert MODEL_SEEDS == (171701, 171702, 171703)
    assert OPTIMIZER_STEPS == CHECKPOINT_STEP == 300
    assert CALIBRATION_QUANTILE == 0.95
    assert B0_PARAMS == 1821698
    assert B0_FLOPS_REFERENCE == 182016709


def test_frozen_numerics_pin_task10_16_values() -> None:
    for arm_id, expected in EXPECTED_NUMERICS.items():
        assert MEMBER_IMPLEMENTATION[arm_id]["frozen_numerics"] == expected


def test_waiver_identity_matches_harness_and_protocol() -> None:
    assert QUALIFICATION == HARNESS_QUALIFICATION == "MEASURABLE_WITH_USER_WAIVER"
    assert QUALIFICATION != "MEASURABLE"
    assert WAIVER_ID == HARNESS_WAIVER_ID
    assert dict(WAIVER_SCOPE) == dict(HARNESS_WAIVER_SCOPE)
    doc = materialize_combination("K5", _dummy_provenance())
    assert doc["eligibility"]["qualification"] == "MEASURABLE_WITH_USER_WAIVER"
    tampered = copy.deepcopy(doc)
    tampered["eligibility"] = copy.deepcopy(tampered["eligibility"])
    tampered["eligibility"]["qualification"] = "MEASURABLE"
    with pytest.raises(ValueError, match="waiver tuple"):
        validate_frozen_combination(tampered)


def test_freeze_json_hashes_match_working_tree_bytes() -> None:
    freeze = json.loads(FREEZE_JSON.read_text(encoding="utf-8"))
    assert set(freeze["combinations"]) == set(EXPECTED_MEMBERS)
    for path, digest in freeze["provenance_inputs"].items():
        assert hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() == digest


def test_freeze_json_digests_recompute_deterministically() -> None:
    freeze = json.loads(FREEZE_JSON.read_text(encoding="utf-8"))
    for k_id, doc in freeze["combinations"].items():
        assert validate_frozen_combination(doc) == doc
        unsigned = {k: v for k, v in doc.items() if k != "combination_sha256"}
        assert doc["combination_sha256"] == canonical_hash(unsigned)
    combos = freeze["combinations"]
    assert freeze["freeze_sha256"] == canonical_hash(combos)
    rematerialized = materialize_all(freeze["provenance_inputs"])
    assert canonical_hash(rematerialized) == canonical_hash(combos)


def test_freeze_carries_no_outcome_confirmation_or_sealed_content() -> None:
    text = FREEZE_JSON.read_text(encoding="utf-8")
    for forbidden in ("CONFIRMATION", "H-SEAL", "H-S17-V3-CONF", "AUROC", "macro_pw", "delta_vs_B0"):
        assert forbidden not in text
    freeze = json.loads(text)
    assert freeze["qualification"] == "MEASURABLE_WITH_USER_WAIVER"
    assert freeze["no_k_execution"] is True
    assert freeze["confirmation_sealed_access"] is False
    assert all(doc["no_execution"] is True for doc in freeze["combinations"].values())
