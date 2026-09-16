"""Focused Task 24 contract tests: Confirmation scoring surface.

Execution-free: no data, checkpoint, GPU, Confirmation scoring, or Sealed
access. The driver module is imported (import-safe: constants + defs only)
and its frozen registry/config/guard surface is asserted. Summaries and the
Task 23 lock enter as opaque bytes or literal comparisons; outcome fields
are never parsed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = REPO_ROOT / "experiments" / "sprint17_task24_confirm.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("sprint17_task24_confirm", DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRIVER = _load_driver()


def test_arm_registry_is_exactly_b0_plus_14_singles() -> None:
    assert DRIVER.CONFIRM_ARMS == (
        "B0", "C2-A", "C2-B", "C3-A", "C3-B", "C5-A", "C5-B",
        "C6-A", "C6-B", "C7-A", "C7-B", "C8-A", "C8-B", "C9-A", "C9-B",
    )
    assert set(DRIVER.TRAINABLE_ARMS) | set(DRIVER.TRAIN_FREE_ARMS) == set(DRIVER.CONFIRM_ARMS)
    assert not set(DRIVER.TRAINABLE_ARMS) & set(DRIVER.TRAIN_FREE_ARMS)
    assert not {a for a in DRIVER.CONFIRM_ARMS if a.startswith("K")}
    assert len(set(DRIVER.ARM_FILE_PREFIX.values())) == 15
    assert DRIVER.ARM_COMPONENTS["B0"] == ()
    assert DRIVER.ARM_COMPONENTS["C9-B"] == ("C9",)


def test_task23_lock_constant_matches_lock_module() -> None:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from representation.sprint17_confirmation_lock import materialize_lock, lock_sha256, validate_lock
    lock = materialize_lock()
    assert validate_lock(lock) is True
    assert DRIVER.TASK23_LOCK_SHA256 == lock_sha256(lock) == "14fab303d78725af061aaa639a558f84b262c76d5b918042618a109d61afd039"
    freeze = json.loads((REPO_ROOT / DRIVER.TASK23_LOCK_PATH).read_text(encoding="utf-8"))
    assert freeze["lock_sha256"] == DRIVER.TASK23_LOCK_SHA256
    assert validate_lock(freeze) is True


def test_arm_configs_are_b0_plus_declared_delta_only() -> None:
    base = DRIVER.b0_config_dict(171701)
    for arm_id in DRIVER.CONFIRM_ARMS:
        cfg = DRIVER.arm_config_dict(arm_id, 171701)
        assert cfg["arm_id"] == arm_id
        delta = {k for k in cfg if cfg[k] != base.get(k)}
        if arm_id == "B0":
            assert delta <= {"arm_id", "adapter_description"}
        elif arm_id.startswith("C2-"):
            assert {"arm_id", "stride", "c2_grids", "support_weights", "adapter_description"} >= delta > {"arm_id"}
        elif arm_id.startswith("C3-"):
            assert {"arm_id", "local_encoder", "local_encoder_params", "adapter_description"} >= delta > {"arm_id"}
        elif arm_id.startswith("C5-"):
            assert {"arm_id", "masking_policy", "criterion", "prediction_horizons", "adapter_description"} >= delta > {"arm_id"}
        elif arm_id.startswith("C6-"):
            assert {"arm_id", "pooling", "pooling_params", "adapter_description"} >= delta > {"arm_id"}
        elif arm_id.startswith("C7-"):
            assert {"arm_id", "reference", "adapter_description"} >= delta > {"arm_id"}
        elif arm_id.startswith("C8-"):
            assert {"arm_id", "scorer", "adapter_description"} >= delta > {"arm_id"}
        elif arm_id.startswith("C9-"):
            assert {"arm_id", "aggregation", "adapter_description"} >= delta > {"arm_id"}
    assert DRIVER.arm_config_dict("C5-B", 171701)["criterion"] == "vicreg-file-level"
    assert DRIVER.arm_config_dict("C8-B", 171701)["scorer"] == "cosine-distance-mean"
    assert DRIVER.arm_config_dict("C9-A", 171701)["aggregation"] == "top-k-mean-fraction-0.25"
    with pytest.raises(ValueError):
        DRIVER.arm_config_dict("K1", 171701)


def test_masking_policy_only_c5a() -> None:
    from representation.sprint17_c5 import c5a_block_policy
    assert DRIVER.masking_policy_for("C5-A") is c5a_block_policy
    for arm_id in DRIVER.CONFIRM_ARMS:
        if arm_id != "C5-A":
            assert DRIVER.masking_policy_for(arm_id) is None


def test_no_training_surface_in_driver() -> None:
    text = DRIVER_PATH.read_text(encoding="utf-8")
    # "CosineAnnealingLR" occurs only inside the frozen B0 schedule
    # description string (config record, not executed training code).
    for banned in ("def train_seed", "RepresentationTrainer", "optimizer.step(",
                   "save_checkpoint", "from representation.trainer import",
                   "--b0-metrics"):
        assert banned not in text
    assert "scoring-only" in text
    assert "no recalibration" in text


def test_group_refusals_fail_closed() -> None:
    with pytest.raises(ValueError):
        DRIVER.load_verified_root(Path("/nonexistent"), "DEVELOPMENT", "H-S17-V3-DEV-01", 2808)
    with pytest.raises(ValueError):
        DRIVER.load_verified_root(Path("/nonexistent"), "CALIBRATION", "H-S17-V3-CAL-01", 2807)
    with pytest.raises(ValueError):
        DRIVER.load_verified_root(Path("/nonexistent"), "SEALED", "H-SEAL-37", 0)
    with pytest.raises((ValueError, FileNotFoundError)):
        DRIVER.load_verified_root(
            Path("/nonexistent"), "CONFIRMATION", "H-S17-V3-DEV-01", 2808)


def test_manifest_expectations_match_task23_lock() -> None:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from representation.sprint17_confirmation_lock import ALL_MANIFEST_SHA256
    for role in ("H-S17-V3-FIT-01", "H-S17-V3-FIT-02", "H-S17-V3-FIT-03",
                 "H-S17-V3-CONF-01", "H-S17-V3-CONF-02",
                 "H-S17-V3-CONF-03", "H-S17-V3-CONF-04"):
        assert DRIVER.EXPECTED_MANIFEST_SHA256[role] == ALL_MANIFEST_SHA256[role]
    assert set(DRIVER.EXPECTED_MANIFEST_SHA256) == {
        "H-S17-V3-FIT-01", "H-S17-V3-FIT-02", "H-S17-V3-FIT-03",
        "H-S17-V3-CONF-01", "H-S17-V3-CONF-02",
        "H-S17-V3-CONF-03", "H-S17-V3-CONF-04",
    }


def test_lock_thresholds_cover_all_arms_and_seeds() -> None:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from representation.sprint17_confirmation_lock import materialize_lock
    lock = materialize_lock()
    for arm_id in DRIVER.CONFIRM_ARMS:
        th = DRIVER.lock_thresholds(lock, arm_id)
        assert set(th) == {"171701", "171702", "171703"}
        for seed, vals in th.items():
            assert vals["pred"] > 0 and vals["pop"] > 0
    b0pop = lock["arms"]["B0"]["thresholds"]
    for seed in ("171701", "171702", "171703"):
        assert DRIVER.lock_thresholds(lock, "C9-B")[seed]["pop"] == b0pop[seed]["pop"]
        assert DRIVER.lock_thresholds(lock, "C9-B")[seed]["pred"] == lock["arms"]["C9-B"]["thresholds_pred"][seed]


def test_confirmation_roles_and_support_contract() -> None:
    assert [r for r, _ in DRIVER.CONF_ROLES] == [
        "H-S17-V3-CONF-01", "H-S17-V3-CONF-02",
        "H-S17-V3-CONF-03", "H-S17-V3-CONF-04",
    ]
    assert [s for _, s in DRIVER.CONF_ROLES] == [2812, 2813, 2814, 2815]
    assert DRIVER.OPTIMIZER_STEPS == 300 and DRIVER.CHECKPOINT_STEP == 300
    assert DRIVER.CALIBRATION_QUANTILE == 0.95
    assert DRIVER.FIT_ROWS == 5040 and DRIVER.FIT_PATCHES == 154129
    assert DRIVER.FIDELITY_ATOL == 1e-3


def test_far_delta_gate_semantics() -> None:
    assert DRIVER._far_delta_ok({"per_history_far": [0.01, 0.0, -0.01, 0.02]}) is True
    assert DRIVER._far_delta_ok({"per_history_far": [0.01, 0.0, None, 0.02]}) is False
    assert DRIVER._far_delta_ok({"per_history_far": [0.01, 0.021, 0.0, 0.0]}) is False
    assert DRIVER._far_delta_ok({"macro_pw": None}) is False


def test_development_consistency_signs_use_exact_raw_deltas() -> None:
    """Regression test for the Task 24 re-review correction (LOW).

    The committed summary's `development_consistency` block must carry the
    exact raw frozen S_pred paired Delta_PW values (never rounded inputs:
    C2-A's Development delta is -1.03e-07, which a rounded -0.0 input
    misclassifies) with sign flags under (dev >= 0) == (conf >= 0).
    Execution-free: frozen summaries enter as opaque parsed config values;
    no outcome is recomputed and no data root is touched.
    """
    expected_dev = {
        "C2-A": -1.0312510312360246e-07,
        "C2-B": -0.0044133930934630765,
        "C3-A": -0.01007096548106882,
        "C3-B": -0.015301104035036225,
        "C5-A": 0.032334926203523996,
        "C5-B": 1.1446886446867902e-05,
        "C6-A": 0.008968441594149187,
        "C6-B": -0.015497267681295224,
        "C7-A": 0.0,
        "C7-B": 0.0,
        "C8-A": 0.004356175642764877,
        "C8-B": -0.0003840809543401713,
        "C9-A": -0.008666080208195074,
        "C9-B": -0.007333795257505475,
    }
    expected_flags = {
        "C2-A": True, "C2-B": False, "C3-A": True, "C3-B": False,
        "C5-A": True, "C5-B": True, "C6-A": True, "C6-B": True,
        "C7-A": True, "C7-B": True, "C8-A": False, "C8-B": True,
        "C9-A": False, "C9-B": False,
    }
    summary = json.loads(
        (REPO_ROOT / "experiments" / "sprint17-task24-confirmation-summary.json")
        .read_text(encoding="utf-8"))
    block = summary["development_consistency"]
    assert sum(1 for aid in expected_flags if block[aid]["sign_agreement"]) == 9
    assert {aid for aid in expected_flags if block[aid]["sign_agreement"]} == {
        "C2-A", "C3-A", "C5-A", "C5-B", "C6-A", "C6-B",
        "C7-A", "C7-B", "C8-B",
    }
    for aid, dev in expected_dev.items():
        assert block[aid]["development_delta_pw"] == dev
        assert block[aid]["sign_agreement"] == expected_flags[aid]
        assert block[aid]["sign_agreement"] == (
            (block[aid]["development_delta_pw"] >= 0)
            == (block[aid]["confirmation_delta_pw"] >= 0))
