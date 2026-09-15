"""Focused Task 20 contract tests: K1–K6 composition surface (execution-free).

No data, checkpoint, GPU, Confirmation, or Sealed access. The Task 19 freeze
JSON carries no outcome fields, so freeze validation parses IDs only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import sprint17_task20_k as K


def test_k_registry_matches_batch_contract_and_freeze() -> None:
    assert K.K_MEMBERS == {
        "K1": ("C2-A", "C3-A"),
        "K2": ("C3-A", "C5-A"),
        "K3": ("C5-A", "C6-A"),
        "K4": ("C6-A", "C7-A"),
        "K5": ("C7-A", "C8-A"),
        "K6": ("C8-A", "C9-B"),
    }
    assert set(K.K_ARMS) == set(K.K_MEMBERS)
    assert set(K.TRAINABLE_K) == {"K1", "K2", "K3", "K4"}
    assert set(K.TRAIN_FREE_K) == {"K5", "K6"}
    assert not (set(K.TRAINABLE_K) & set(K.TRAIN_FREE_K))
    for k_id in ("K7", "K8", "K9"):
        assert k_id not in K.K_ARMS
    freeze = K.check_task19_freeze()
    for k_id, members in K.K_MEMBERS.items():
        assert tuple(freeze["combinations"][k_id]["member_arm_ids"]) == members


def test_b0_reload_constants_pin_task7_record() -> None:
    assert K.B0_CKPT_SHA256 == {
        171701: "45f9e151c5d3946804ea531eb2bcace26b1f62e634671f51b8741ac6b411a619",
        171702: "64ed2cedec210e4692f60bb4dd3430a57cf94abfcd50551abd97c9265ea785f8",
        171703: "9edb2122e357376a9ff1e065d73d5ab7704f8d2005991552332f1ced01b0e906",
    }
    assert K.FIDELITY_ATOL == 1e-3
    assert set(K.B0_CACHE_SHA256) == {171701, 171702, 171703}
    assert K.B0_CACHE_SHA256[171701]["H-S17-V3-DEV-01"] == (
        "eb73ffe26e7b64349bbf6cea06236518225ccc2247c5fa849a8a65f29bfa3438"
    )


def test_masking_policy_follows_c5_membership() -> None:
    assert K.masking_policy_for("K2") is not None
    assert K.masking_policy_for("K3") is not None
    for k_id in ("K1", "K4", "K5", "K6"):
        assert K.masking_policy_for(k_id) is None
    with pytest.raises(KeyError):
        K.masking_policy_for("K7")


def test_k_config_is_b0_plus_member_union_only() -> None:
    base_keys = set(K.b0_config_dict(171701))
    k1 = K.k_config_dict("K1", 171701)
    k2 = K.k_config_dict("K2", 171703)
    k4 = K.k_config_dict("K4", 171702)
    k6 = K.k_config_dict("K6", 171701)
    assert k1["arm_id"] == "K1"
    assert k1["member_arm_ids"] == ["C2-A", "C3-A"]
    assert set(k1) - base_keys == {
        "member_arm_ids", "c2_grids", "support_weights",
        "local_encoder", "local_encoder_params", "adapter_description",
    } | ({"arm_id"} if "arm_id" not in base_keys else set())
    assert "masking_policy" not in k1 and "pooling" not in k1
    assert k4["member_arm_ids"] == ["C6-A", "C7-A"]
    assert "pooling" in k4 and "masking_policy" not in k4
    assert "local_encoder" not in k4 and "c2_grids" not in k4
    assert k2["masking_policy"] == "channel_time_block"
    assert k2["criterion"] == "multi-horizon-ema"
    assert k2["prediction_horizons"] == [0, 1, 2]
    assert k6["member_arm_ids"] == ["C8-A", "C9-B"]
    for cfg in (k1, k2, k4, k6):
        assert cfg["optimizer_steps"] == cfg["checkpoint_step"] == 300
        assert cfg["calibration_quantile"] == 0.95
    assert "C2-A" in k1["adapter_description"] and "C3-A" in k1["adapter_description"]
    assert "C7-A" in K.k_config_dict("K5", 171701)["adapter_description"]
    with pytest.raises(ValueError, match="Task 20 runs only"):
        K.k_config_dict("K7", 171701)


def test_k_config_frozen_numerics_match_task19() -> None:
    assert K.k_config_dict("K1", 171701)["c2_grids"] == [[0, 16]]
    assert K.k_config_dict("K1", 171701)["local_encoder_params"] == 34688
    assert K.k_config_dict("K3", 171701)["pooling_params"] == 32896
    assert K.k_config_dict("K2", 171701)["local_encoder"] == (
        K.k_config_dict("K1", 171701)["local_encoder"]
    )
