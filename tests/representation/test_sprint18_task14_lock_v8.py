"""Versioned prospective Task14 N4-CRN v8 lock regression test.

Protects the lock that carries the reviewed N4 v3 initialization/guard
contract, the corrected Task30 prediction-cache binding contract, the
corrected strict-restore probe device alignment, the corrected Task30
Fit-bank row ordering, the certified-capacity identity comparison
(historical container custody plus canonical payload/recipe identity instead
of a ``torch.save`` container-equality test), and the corrected historical
envelope directory derivation (the certified launch output directory
``evidence.remote_full``, not the run grandparent). The historical v1 lock and
the superseded v2-v7 prospective locks must stay byte-identical; the v8 lock
must recompute its canonical core digest, re-materialize the three N4-CRN seed
configs through the real arm-config consumer without reclaiming the other
arms, and the runner's lock consumer must accept it while refusing stale or
tampered variants.

No DATA access, no training, no outcome, no Git mutation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import sprint18_task13b_n4_capacity as runner  # noqa: E402

from representation.crn import (  # noqa: E402
    N4_GUARD_CONTRACT,
    N4_SATURATION_CAP_GATE,
    N4_SCALE_BATCH_SIZE,
    N4_SCALE_CHECKPOINT_SCHEMA,
    N4_SCALE_CLIP_NORM,
    N4_SCALE_EPOCHS,
    N4_SCALE_HIDDEN_WIDTH,
    N4_SCALE_LR,
    N4_SCALE_PARAM_LIMIT,
    N4_SCALE_PARAMETERIZATION,
    N4_SCALE_WEIGHT_DECAY,
)
from representation.sprint18_harness import (  # noqa: E402
    MODEL_SEEDS,
    arm_config_sha256,
    make_sprint18_arm_config,
    validate_arm_config,
)

LOCK_PATH = ROOT / "experiments" / "sprint18-task14-config-lock-v8.json"
SUPERSEDED_LOCK_PATH = ROOT / "experiments" / "sprint18-task14-config-lock-v7.json"
V6_LOCK_PATH = ROOT / "experiments" / "sprint18-task14-config-lock-v6.json"
V5_LOCK_PATH = ROOT / "experiments" / "sprint18-task14-config-lock-v5.json"
V4_LOCK_PATH = ROOT / "experiments" / "sprint18-task14-config-lock-v4.json"
V3_LOCK_PATH = ROOT / "experiments" / "sprint18-task14-config-lock-v3.json"
V2_LOCK_PATH = ROOT / "experiments" / "sprint18-task14-config-lock-v2.json"
LEGACY_LOCK_PATH = ROOT / "experiments" / "sprint18-task14-config-lock.json"
SPEC_V1_PATH = ROOT / "experiments" / "sprint18-context-residual-normalization-spec.md"
SPEC_V2_PATH = ROOT / "experiments" / "sprint18-context-residual-normalization-spec-v2.md"
SPEC_V3_PATH = ROOT / "experiments" / "sprint18-context-residual-normalization-spec-v3.md"

LEGACY_LOCK_SHA256 = "c4c1a8bde14310cf4436c10762e3e98475f7c84eb74fcb8a74b3e267885edec3"
LEGACY_COEFFICIENT_LOCK_SHA256 = (
    "6eb108b2073181d427f1aedfc81fb3bae4c29595108d15de486082800bdd9d6e"
)
SUPERSEDED_LOCK_SHA256 = "69ac1218fda6adf1364193371cf3ef8fcd8afbd9cbc5b9e8c4433dbefc61e8e7"
SUPERSEDED_COEFFICIENT_LOCK_SHA256 = (
    "c4d331577008d2e28762e85fd424cea2f7ae5052a0bd6a34cb6e08b76b71a7ff"
)
V6_LOCK_SHA256 = "b4d1f4352d88d35a024db3f195e48420448d7cd690ee98b26ebf123a0a55b81f"
V6_COEFFICIENT_LOCK_SHA256 = (
    "5f65f907aadc99c37dbdb10c82070b145348031faaa3d47309dc8a29e788421f"
)
V5_LOCK_SHA256 = "feeff6abe1a869d5dc1c910087a02db802421ba09d3851e3da294d660c2625be"
V5_COEFFICIENT_LOCK_SHA256 = (
    "e48455b0b85976c776c4413be759d839117a8d573490fae227dd8c89c48b37ed"
)
V4_LOCK_SHA256 = "edbd65f35474375b0472c0752cea83c0029282a1f955522e462bfb85f66d1829"
V4_COEFFICIENT_LOCK_SHA256 = (
    "3151a4ea795ea5dfa881ee33b2887a1a0b3958b91006cd13e7cb7efb0f05b490"
)
V3_LOCK_SHA256 = "5ce105b495b9f89569054b5bd512a2f80d31d14c89545b7463a22c8a4bd59a4e"
V3_COEFFICIENT_LOCK_SHA256 = (
    "c6c3fe84dece2a76b21cafbb87ceccfdb028b123edb138c084594db90170210b"
)
V2_LOCK_SHA256 = "6dd815c5c58fd2abe6305f730f712b7f295695f284e90b9d646b2e5482a4d087"
V2_COEFFICIENT_LOCK_SHA256 = (
    "4b399874a260bcf42904911c7635be3b844d859313b491a1d2cef7f75ea1d8db"
)
SPEC_V1_SHA256 = "fd0e1722ea4f248a354086bbdf44bc486d741a9cfb018f68ef08277827d7954d"
SPEC_V2_SHA256 = "4d1c8097c9cc27d3fbeb2e2a6e70a29a7bc63d7899646a9c712a315b39d9ffe3"
SPEC_V3_SHA256 = "e7fe42e41832ae283614ee6a76c0623ee567068dcc540a222f5b66a9d4262da2"
REVIEWED_RUNNER_SHA256 = "b439b508a5c8d28e9159180f67af06bac07a41d1e386f5a934f5316ab1d42970"
SUPERSEDED_RUNNER_PIN = "ad7fd8616aebd70390d32baeb7ad604d218d1ce07a45b98441349062609d92ba"
V6_RUNNER_PIN = "4bb560e179cf385a95a0eca4a603827cc642e9109063687c82d04ea98ae85322"
V5_RUNNER_PIN = "19331f57e0d7b307085bd8e69f67740f693448096c82adede89b207cff39c36f"
V4_RUNNER_PIN = "f24e825ed242942045f7c69c0daac08a8a8374f897914acafddbca0902bac3f9"
V3_RUNNER_PIN = "e923732778c1479f2bcac1c107c792e4a6c5730b717854b87b064dc86f0b78db"
BASE_COMMIT = "39b51d5475f14794caa98842162fb587491d7289"
BASE_PARENT = "7735669a8f603271e1b65a3caa68e09429eb2625"
FIT_HISTORIES = [
    "S18I-ITER-0005-FIT-01",
    "S18I-ITER-0005-FIT-02",
    "S18I-ITER-0005-FIT-03",
]
COEFFICIENTS = {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None}
SUPERSEDES_SCOPE = (
    "supersedes the v7 prospective lock for the certified Task13B historical "
    "envelope directory derivation (the recorded launch --out-dir "
    "evidence.remote_full instead of the run grandparent) and the refreshed "
    "reviewed implementation pins only; the v7, v6, v5, v4, v3, v2 and v1 lock "
    "bytes, every other arm, the frozen scientific recipe, the final-B0/DATA "
    "identities and all historical evidence remain unchanged"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_lock() -> dict:
    return json.loads(LOCK_PATH.read_bytes().decode("utf-8"))


def _load_legacy_lock() -> dict:
    return json.loads(LEGACY_LOCK_PATH.read_bytes().decode("utf-8"))


def _load_superseded_lock() -> dict:
    return json.loads(SUPERSEDED_LOCK_PATH.read_bytes().decode("utf-8"))


def _load_v6_lock() -> dict:
    return json.loads(V6_LOCK_PATH.read_bytes().decode("utf-8"))


def _load_v5_lock() -> dict:
    return json.loads(V5_LOCK_PATH.read_bytes().decode("utf-8"))


def _load_v4_lock() -> dict:
    return json.loads(V4_LOCK_PATH.read_bytes().decode("utf-8"))


def _load_v3_lock() -> dict:
    return json.loads(V3_LOCK_PATH.read_bytes().decode("utf-8"))


def _load_v2_lock() -> dict:
    return json.loads(V2_LOCK_PATH.read_bytes().decode("utf-8"))


def _resign(lock: dict) -> dict:
    """Recompute the derived core digest after an intentional tamper."""
    lock["coefficient_lock_sha256"] = runner.task14_lock_core_digest(lock)
    return lock


def _rebuild_locked_configs(lock: dict) -> None:
    """Rebuild the derived configs/digests a coordinated tamper would refresh."""
    digest = lock["coefficient_lock_sha256"]
    lock["locked_configs"] = {}
    lock["config_sha256"] = {}
    for seed in MODEL_SEEDS:
        key = f"N4-CRN/seed-{seed}"
        config = make_sprint18_arm_config(
            "N4-CRN",
            seed,
            COEFFICIENTS,
            selection_state="locked",
            coefficient_lock_sha256=digest,
        )
        lock["locked_configs"][key] = config.model_dump(mode="json")
        lock["config_sha256"][key] = arm_config_sha256(config)


def test_lock_v8_digest_supersedes_and_preserved_history() -> None:
    lock = _load_lock()
    assert lock["schema"] == "s18-task14-config-lock-v8"
    assert lock["lock_version"] == 8
    assert lock["attempt_id"] == "S18-T30-A11"
    assert lock["task_id"] == "S18-T14"
    assert lock["source_commit"] == BASE_COMMIT
    assert lock["source_parent"] == BASE_PARENT
    assert lock["target_ref"] == "refs/heads/master"
    assert lock["created_by"] == {
        "task_id": "S18-T30",
        "attempt_id": "S18-T30-A11",
        "role": "gold-task",
        "route": "full",
        "review_status": "pending_independent_source_review",
    }
    core = {
        key: value
        for key, value in lock.items()
        if key not in ("coefficient_lock_sha256", "locked_configs", "config_sha256", "file_sha256")
    }
    recomputed = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert recomputed == lock["coefficient_lock_sha256"]
    assert lock["coefficient_lock_sha256"] != SUPERSEDED_COEFFICIENT_LOCK_SHA256
    assert lock["coefficient_lock_sha256"] != V6_COEFFICIENT_LOCK_SHA256
    assert lock["coefficient_lock_sha256"] != V5_COEFFICIENT_LOCK_SHA256
    assert lock["coefficient_lock_sha256"] != V4_COEFFICIENT_LOCK_SHA256
    assert lock["coefficient_lock_sha256"] != V3_COEFFICIENT_LOCK_SHA256
    assert lock["coefficient_lock_sha256"] != V2_COEFFICIENT_LOCK_SHA256
    assert lock["coefficient_lock_sha256"] != LEGACY_COEFFICIENT_LOCK_SHA256
    assert lock["coefficient_selections"] == _load_legacy_lock()["coefficient_selections"]
    assert lock["supersedes"] == {
        "path": "experiments/sprint18-task14-config-lock-v7.json",
        "schema": "s18-task14-config-lock-v7",
        "file_sha256": SUPERSEDED_LOCK_SHA256,
        "coefficient_lock_sha256": SUPERSEDED_COEFFICIENT_LOCK_SHA256,
        "scope": SUPERSEDES_SCOPE,
    }
    # Historical bytes are preserved exactly through the v7 -> v6 -> v5 -> v4 -> v3 -> v2 -> v1 chain.
    assert _sha256(SUPERSEDED_LOCK_PATH) == SUPERSEDED_LOCK_SHA256
    superseded = _load_superseded_lock()
    assert superseded["schema"] == "s18-task14-config-lock-v7"
    assert superseded["lock_version"] == 7
    assert superseded["coefficient_lock_sha256"] == SUPERSEDED_COEFFICIENT_LOCK_SHA256
    assert superseded["source_commit"] == "7735669a8f603271e1b65a3caa68e09429eb2625"
    assert superseded["supersedes"]["path"] == "experiments/sprint18-task14-config-lock-v6.json"
    assert superseded["supersedes"]["file_sha256"] == V6_LOCK_SHA256
    assert superseded["supersedes"]["coefficient_lock_sha256"] == V6_COEFFICIENT_LOCK_SHA256
    assert _sha256(V6_LOCK_PATH) == V6_LOCK_SHA256
    v6 = _load_v6_lock()
    assert v6["schema"] == "s18-task14-config-lock-v6"
    assert v6["lock_version"] == 6
    assert v6["coefficient_lock_sha256"] == V6_COEFFICIENT_LOCK_SHA256
    assert v6["source_commit"] == "7735669a8f603271e1b65a3caa68e09429eb2625"
    assert v6["supersedes"]["path"] == "experiments/sprint18-task14-config-lock-v5.json"
    assert v6["supersedes"]["file_sha256"] == V5_LOCK_SHA256
    assert v6["supersedes"]["coefficient_lock_sha256"] == V5_COEFFICIENT_LOCK_SHA256
    assert _sha256(V5_LOCK_PATH) == V5_LOCK_SHA256
    v5 = _load_v5_lock()
    assert v5["schema"] == "s18-task14-config-lock-v5"
    assert v5["lock_version"] == 5
    assert v5["coefficient_lock_sha256"] == V5_COEFFICIENT_LOCK_SHA256
    assert v5["source_commit"] == "430cbbd4d5ced2f8bdc7926ff00f48bc0b124590"
    assert v5["supersedes"]["path"] == "experiments/sprint18-task14-config-lock-v4.json"
    assert v5["supersedes"]["file_sha256"] == V4_LOCK_SHA256
    assert v5["supersedes"]["coefficient_lock_sha256"] == V4_COEFFICIENT_LOCK_SHA256
    assert _sha256(V4_LOCK_PATH) == V4_LOCK_SHA256
    v4 = _load_v4_lock()
    assert v4["schema"] == "s18-task14-config-lock-v4"
    assert v4["lock_version"] == 4
    assert v4["coefficient_lock_sha256"] == V4_COEFFICIENT_LOCK_SHA256
    assert v4["source_commit"] == "74ed62b0c0c4b3ef027bb1d473b76033d79b30d9"
    assert v4["supersedes"]["path"] == "experiments/sprint18-task14-config-lock-v3.json"
    assert v4["supersedes"]["file_sha256"] == V3_LOCK_SHA256
    assert v4["supersedes"]["coefficient_lock_sha256"] == V3_COEFFICIENT_LOCK_SHA256
    assert _sha256(V3_LOCK_PATH) == V3_LOCK_SHA256
    v3 = _load_v3_lock()
    assert v3["schema"] == "s18-task14-config-lock-v3"
    assert v3["lock_version"] == 3
    assert v3["coefficient_lock_sha256"] == V3_COEFFICIENT_LOCK_SHA256
    assert v3["source_commit"] == "83710aa65d201cf3f37d7f9961368589932f6ec9"
    assert v3["supersedes"]["path"] == "experiments/sprint18-task14-config-lock-v2.json"
    assert v3["supersedes"]["file_sha256"] == V2_LOCK_SHA256
    assert v3["supersedes"]["coefficient_lock_sha256"] == V2_COEFFICIENT_LOCK_SHA256
    assert _sha256(V2_LOCK_PATH) == V2_LOCK_SHA256
    v2 = _load_v2_lock()
    assert v2["schema"] == "s18-task14-config-lock-v2"
    assert v2["lock_version"] == 2
    assert v2["coefficient_lock_sha256"] == V2_COEFFICIENT_LOCK_SHA256
    assert v2["supersedes"]["path"] == "experiments/sprint18-task14-config-lock.json"
    assert v2["supersedes"]["file_sha256"] == LEGACY_LOCK_SHA256
    assert v2["supersedes"]["coefficient_lock_sha256"] == LEGACY_COEFFICIENT_LOCK_SHA256
    assert _sha256(LEGACY_LOCK_PATH) == LEGACY_LOCK_SHA256
    legacy = _load_legacy_lock()
    assert legacy["coefficient_lock_sha256"] == LEGACY_COEFFICIENT_LOCK_SHA256
    assert legacy["schema"] == "s18-task14-config-lock-v1"
    assert _sha256(SPEC_V1_PATH) == SPEC_V1_SHA256
    assert _sha256(SPEC_V2_PATH) == SPEC_V2_SHA256
    assert _sha256(SPEC_V3_PATH) == SPEC_V3_SHA256
    # Prospective N4-only scope: the other arms stay bound by the v1 lock.
    assert set(lock["locked_configs"]) == {f"N4-CRN/seed-{seed}" for seed in MODEL_SEEDS}
    assert "historical v1 lock" in lock["locked_configs_scope"]
    assert lock["arms"]["N1-HCC"]["executable"] is False


def test_lock_v8_n4_provenance_matches_live_contract() -> None:
    lock = _load_lock()
    legacy = _load_legacy_lock()
    activation = lock["n4_v3_activation"]
    n4lock = lock["arms"]["N4-CRN"]
    for holder in (n4lock, activation):
        assert holder["mechanism_spec_amendment_sha256"] == SPEC_V3_SHA256
        assert holder["scale_parameterization"] == N4_SCALE_PARAMETERIZATION
        assert holder["guard_contract"] == N4_GUARD_CONTRACT
        assert holder["scale_checkpoint_schema"] == N4_SCALE_CHECKPOINT_SCHEMA
    assert n4lock["mechanism_spec_sha256"] == SPEC_V2_SHA256
    assert activation["mechanism_spec_base_sha256"] == SPEC_V2_SHA256
    assert activation["spec"]["path"] == "experiments/sprint18-context-residual-normalization-spec-v3.md"
    assert activation["spec"]["sha256"] == SPEC_V3_SHA256
    assert lock["normative"]["mechanism_spec_sha256"]["N4-CRN"] == SPEC_V2_SHA256
    assert lock["normative"]["mechanism_spec_amendment_sha256"]["N4-CRN"] == SPEC_V3_SHA256
    assert lock["normative"]["contract_files"][activation["spec"]["path"]] == SPEC_V3_SHA256

    # Shared contract fields are carried forward unchanged.
    assert lock["support"] == legacy["support"]
    assert lock["common_contract"] == legacy["common_contract"]
    assert lock["compute_envelopes"] == legacy["compute_envelopes"]
    for key, value in legacy["n4_recipe"].items():
        assert lock["n4_recipe"][key] == value
    recipe = lock["n4_recipe"]
    amendment = recipe["v3_amendment"]
    assert amendment["parameter_count"] == 4417
    assert amendment["parameter_limit"] == N4_SCALE_PARAM_LIMIT == 100000
    assert amendment["saturation_cap_gate"] == N4_SATURATION_CAP_GATE == 0.20
    assert amendment["scale_parameterization"] == N4_SCALE_PARAMETERIZATION
    assert amendment["guard_contract"] == N4_GUARD_CONTRACT
    assert amendment["scale_checkpoint_schema"] == N4_SCALE_CHECKPOINT_SCHEMA
    assert recipe["architecture"]["hidden_width"] == N4_SCALE_HIDDEN_WIDTH == 32
    assert recipe["fit_recipe"]["epochs"] == N4_SCALE_EPOCHS == 50
    assert recipe["fit_recipe"]["batch_size"] == N4_SCALE_BATCH_SIZE == 512
    assert recipe["fit_recipe"]["learning_rate"] == N4_SCALE_LR == 1e-3
    assert recipe["fit_recipe"]["weight_decay"] == N4_SCALE_WEIGHT_DECAY == 1e-4
    assert recipe["fit_recipe"]["gradient_clip_norm"] == N4_SCALE_CLIP_NORM == 1.0
    assert recipe["fit_recipe"]["optimizer"] == "AdamW"
    assert recipe["fit_recipe"]["representation_optimizer_steps"] == 0
    assert recipe["fit_recipe"]["no_early_stopping_or_development_selection"] is True
    assert recipe["capacity"]["scale_parameter_limit"] == N4_SCALE_PARAM_LIMIT
    assert recipe["capacity"]["observed_reference_params_d_model_128"] == 4417
    assert recipe["saturation_gate"]["rule"].startswith("frac_at_cap <= 0.20")
    assert recipe["quantile_bound_rules"]["quantile_convention"] == "numpy linear interpolation"

    data = activation["final_b0_binding"]
    assert data["model_seeds"] == list(MODEL_SEEDS) == [181801, 181802, 181803]
    assert data["fit_histories"] == FIT_HISTORIES
    assert data["fit_eligible_rows"] == {
        "S18I-ITER-0005-FIT-01": 1809,
        "S18I-ITER-0005-FIT-02": 1882,
        "S18I-ITER-0005-FIT-03": 1755,
    }
    assert data["fit_healthy_rows"] == 5446
    assert data["checkpoints_sha256"] == runner.EXPECTED_CONTRACT["checkpoints"]
    assert data["prediction_caches_sha256"] == runner.EXPECTED_CONTRACT["cache_files"]
    assert data["crossfold_roles"] == [
        {"heldout": "S18I-ITER-0005-FIT-01", "training": ["S18I-ITER-0005-FIT-02", "S18I-ITER-0005-FIT-03"]},
        {"heldout": "S18I-ITER-0005-FIT-02", "training": ["S18I-ITER-0005-FIT-01", "S18I-ITER-0005-FIT-03"]},
        {"heldout": "S18I-ITER-0005-FIT-03", "training": ["S18I-ITER-0005-FIT-01", "S18I-ITER-0005-FIT-02"]},
        {"heldout": "final", "training": FIT_HISTORIES},
    ]
    assert "no Confirmation" in data["role_permissions"]
    assert activation["legacy_a03_disposition"]["label"] == "rejected_failed_dead_zone_v1"
    assert activation["legacy_a03_disposition"]["consumed_by_v3"] is False
    assert any("does NOT imply" in boundary for boundary in activation["claim_boundaries"])
    assert activation["status"] == "prospective_locked_pending_independent_source_review_bronze_checkpoint_and_sync"

    implementation = activation["reviewed_implementation"]
    assert implementation["source_commit"] == lock["source_commit"] == BASE_COMMIT
    assert implementation["source_parent"] == BASE_PARENT
    required_paths = {
        "experiments/sprint18-context-residual-normalization-spec-v3.md",
        "experiments/sprint18_task30_n4_runner.py",
        "src/representation/__init__.py",
        "src/representation/crn.py",
        "src/representation/sprint18_checkpoint.py",
        "tests/representation/test_sprint18_crn.py",
    }
    assert set(implementation["paths"]) == required_paths
    assert implementation["paths"]["experiments/sprint18-context-residual-normalization-spec-v3.md"] == SPEC_V3_SHA256
    for relative, expected in implementation["paths"].items():
        assert _sha256(ROOT / relative) == expected
    assert implementation["review"]["verdict"] == "pending"
    assert implementation["review"]["status"] == "pending_independent_source_review"
    assert implementation["review"]["no_self_approval"] is True
    neutrality = implementation["review"]["neutrality_review"]
    assert neutrality["drift_commit"] == "b31c91bbe993be6dc4637db5cfc5917ee67469d1"
    assert neutrality["reviewed_commit_ancestor_of_source"] == "44f56b195597c1c3a46564d63c7269bb29d8c05d"
    assert neutrality["insertions"] == 18 and neutrality["deletions"] == 0
    assert neutrality["change_kind"] == "additive K9 downstream exports in imports and __all__; no N4 symbol or numeric change"
    assert neutrality["n4_semantic_neutral"] is True
    superseded_pins = implementation["review"]["superseded_v2_implementation_pins"]
    assert superseded_pins["src/representation/__init__.py"] == "815e2e7104fda9ac594aab5c1f6dfb19e34a1b94b1aaeb294ae5c85dea3b7046"
    assert set(superseded_pins) == {
        "experiments/sprint18-context-residual-normalization-spec-v3.md",
        "src/representation/__init__.py",
        "src/representation/crn.py",
        "src/representation/sprint18_checkpoint.py",
        "tests/representation/test_sprint18_crn.py",
    }
    superseded_v3_pins = implementation["review"]["superseded_v3_implementation_pins"]
    assert superseded_v3_pins["experiments/sprint18_task30_n4_runner.py"] == V3_RUNNER_PIN
    assert set(superseded_v3_pins) == set(implementation["paths"])
    superseded_v4_pins = implementation["review"]["superseded_v4_implementation_pins"]
    assert superseded_v4_pins["experiments/sprint18_task30_n4_runner.py"] == V4_RUNNER_PIN
    assert set(superseded_v4_pins) == set(implementation["paths"])
    superseded_v5_pins = implementation["review"]["superseded_v5_implementation_pins"]
    assert superseded_v5_pins["experiments/sprint18_task30_n4_runner.py"] == V5_RUNNER_PIN
    assert superseded_v5_pins["src/representation/crn.py"] == implementation["paths"]["src/representation/crn.py"]
    assert set(superseded_v5_pins) == set(implementation["paths"])
    superseded_v6_pins = implementation["review"]["superseded_v6_implementation_pins"]
    assert superseded_v6_pins["experiments/sprint18_task30_n4_runner.py"] == V6_RUNNER_PIN
    assert superseded_v6_pins["src/representation/crn.py"] == implementation["paths"]["src/representation/crn.py"]
    assert set(superseded_v6_pins) == set(implementation["paths"])
    superseded_v7_pins = implementation["review"]["superseded_v7_implementation_pins"]
    assert superseded_v7_pins["experiments/sprint18_task30_n4_runner.py"] == SUPERSEDED_RUNNER_PIN
    assert superseded_v7_pins["src/representation/crn.py"] == implementation["paths"]["src/representation/crn.py"]
    assert set(superseded_v7_pins) == set(implementation["paths"])
    correction = implementation["review"]["correction_review"]
    assert correction["runner_sha256_before"] == SUPERSEDED_RUNNER_PIN
    assert correction["runner_sha256_after"] == implementation["paths"]["experiments/sprint18_task30_n4_runner.py"]
    assert correction["recipe_numeric_semantics_changed"] is False
    assert correction["b0_data_identities_changed"] is False
    assert correction["no_self_approval"] is True
    assert "remote_run_root" in correction["defect"]
    assert "remote_full" in correction["correction"]
    assert lock["runner_identity"]["sha256"] == _sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py")
    assert lock["runner_identity"]["reviewed_source_commit_sha256"] == REVIEWED_RUNNER_SHA256
    assert lock["runner_identity"]["future_activation_commit_frozen"] is False

    scope = lock["n4_scale_scope"]
    assert scope["eligible_masked_fit_patch_records"] == 66989
    assert scope["per_history_records"] == {
        "S18I-ITER-0005-FIT-01": 22328,
        "S18I-ITER-0005-FIT-02": 23098,
        "S18I-ITER-0005-FIT-03": 21563,
    }
    assert scope["fold_record_counts"] == [44661, 43891, 45426]
    assert scope["fold_optimizer_steps"] == [4400, 4300, 4450]
    assert scope["final_optimizer_steps"] == 6550
    assert scope["valid_fit_patches_out_of_scope"] == 167419
    assert scope["epochs"] == 50 and scope["batch_size"] == 512
    certification = activation["capacity_certification"]
    assert certification["path"] == "experiments/sprint18-task13-n4-capacity-v3-a02.json"
    assert certification["file_sha256"] == _sha256(ROOT / certification["path"])
    assert certification["status"] == "historical_certification_at_original_snapshot"
    assert certification["runtime_revalidation"] == "pending_source_gate_then_in_run_fold_recertification"
    assert certification["verdict"] == "PASS"
    assert certification["folds_evaluated"] == 9
    assert certification["total_optimizer_updates"] == 39450
    assert certification["fold_record_counts"] == [44661, 43891, 45426]
    assert set(certification["fold_checkpoint_sha256"]) == {"181801", "181802", "181803"}
    assert all(len(values) == 3 for values in certification["fold_checkpoint_sha256"].values())
    assert certification["source_commit"] == "f40667a6a14e1eaadd544e506c2673058d2f7bbf"
    assert "executed-snapshot" in certification["applicability"].lower()
    assert "remote_full" in certification["applicability"]


def test_lock_v8_certified_envelope_root_uses_actual_output_directory() -> None:
    """The certified envelope dir is the recorded ``--out-dir``, not the run root.

    The A02 launch wrote the nine envelopes to ``$RUN/full`` (the retained
    ``--out-dir``). The certification record names that directory
    ``evidence.remote_full``; deriving from ``remote_run_root`` looked one
    directory too high and made every envelope unreachable.
    """
    certification = json.loads(
        (ROOT / "experiments" / "sprint18-task13-n4-capacity-v3-a02.json").read_text(
            encoding="utf-8"
        )
    )
    evidence = certification["evidence"]
    derived = runner.certified_envelope_root(certification)
    assert derived == evidence["remote_full"] == "/home/trietlm/s18-t13b-v3-certa02-runs/full/"
    assert derived != evidence["remote_run_root"]
    assert all(
        entry.startswith(derived[len(evidence["remote_run_root"]):])
        for entry in evidence["remote_retained_server_side"]
    )
    for fold in certification["folds"]:
        assert derived + fold["checkpoint_path"] != (
            evidence["remote_run_root"] + fold["checkpoint_path"]
        )
    tampered = copy.deepcopy(certification)
    tampered["evidence"].pop("remote_full")
    with pytest.raises(ValueError, match="output directory"):
        runner.certified_envelope_root(tampered)


def test_lock_v8_n4_configs_rebuild_through_real_consumer() -> None:
    lock = _load_lock()
    digest = lock["coefficient_lock_sha256"]
    for seed in MODEL_SEEDS:
        key = f"N4-CRN/seed-{seed}"
        embedded = lock["locked_configs"][key]
        config = make_sprint18_arm_config(
            "N4-CRN",
            seed,
            COEFFICIENTS,
            selection_state="locked",
            coefficient_lock_sha256=digest,
        )
        validate_arm_config(config, require_locked=True)
        assert config.model_dump(mode="json") == embedded
        assert arm_config_sha256(config) == lock["config_sha256"][key]
        assert embedded["selection_state"] == "locked"
        assert embedded["coefficient_lock_sha256"] == digest
        assert embedded["mechanism_spec_sha256"] == SPEC_V2_SHA256
        assert (embedded["batch_size"], embedded["batch_size_unit"]) == (512, "patch_records")
        assert (embedded["learning_rate"], embedded["weight_decay"]) == (1e-3, 1e-4)
        assert (embedded["representation_epochs"], embedded["scale_epochs"]) == (0, 50)


def test_v8_lock_runner_accepts_reviewed_lock() -> None:
    lock = _load_lock()
    record = runner.validate_task14_lock(
        lock,
        worktree=ROOT,
        observed_runner_sha256=_sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py"),
    )
    assert record["schema"] == "s18-task14-config-lock-v8"
    assert record["lock_version"] == 8
    assert record["coefficient_lock_sha256"] == lock["coefficient_lock_sha256"]
    assert record["file_sha256"] == _sha256(LOCK_PATH)
    assert record["mechanism_spec_sha256"] == SPEC_V3_SHA256
    assert record["mechanism_spec_base_sha256"] == SPEC_V2_SHA256
    assert record["source_commit"] == BASE_COMMIT
    assert record["supersedes"]["file_sha256"] == SUPERSEDED_LOCK_SHA256
    assert record["supersedes"]["schema"] == "s18-task14-config-lock-v7"
    assert record["historical_v6_lock"]["file_sha256"] == V6_LOCK_SHA256
    assert record["historical_v6_lock"]["schema"] == "s18-task14-config-lock-v6"
    assert record["historical_v5_lock"]["file_sha256"] == V5_LOCK_SHA256
    assert record["historical_v5_lock"]["schema"] == "s18-task14-config-lock-v5"
    assert record["historical_v4_lock"]["file_sha256"] == V4_LOCK_SHA256
    assert record["historical_v4_lock"]["schema"] == "s18-task14-config-lock-v4"
    assert record["historical_v3_lock"]["file_sha256"] == V3_LOCK_SHA256
    assert record["historical_v3_lock"]["schema"] == "s18-task14-config-lock-v3"
    assert record["historical_v2_lock"]["file_sha256"] == V2_LOCK_SHA256
    assert record["historical_v2_lock"]["schema"] == "s18-task14-config-lock-v2"
    assert record["historical_v1_lock"]["file_sha256"] == LEGACY_LOCK_SHA256
    assert record["runner_sha256"] == _sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py")
    assert record["n4_scale_scope"]["eligible_masked_fit_patch_records"] == 66989
    assert record["capacity_certification"]["folds_evaluated"] == 9
    assert record["capacity_certification"]["status"] == "historical_certification_at_original_snapshot"
    assert record["capacity_certification"]["runtime_revalidation"].startswith("pending_source_gate")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda lock: lock.__setitem__("schema", "s18-task14-config-lock-v6"),
            "schema is not",
        ),
        (
            lambda lock: lock["n4_v3_activation"].__setitem__("scale_parameterization", "legacy-v1"),
            "parameterization differs",
        ),
        (
            lambda lock: lock["n4_v3_activation"]["spec"].__setitem__("sha256", SPEC_V2_SHA256),
            "prospective spec digest",
        ),
        (
            lambda lock: lock["n4_v3_activation"]["reviewed_implementation"]["paths"].__setitem__(
                "src/representation/crn.py", "0" * 64
            ),
            "reviewed implementation bytes changed",
        ),
        (
            lambda lock: lock["supersedes"].__setitem__("file_sha256", "0" * 64),
            "supersedes the wrong prospective lock bytes",
        ),
        (
            lambda lock: lock["supersedes"].__setitem__(
                "coefficient_lock_sha256", V4_COEFFICIENT_LOCK_SHA256
            ),
            "supersedes the wrong prospective coefficient digest",
        ),
    ],
)
def test_v8_lock_runner_refuses_tampered_lock(mutate, message) -> None:
    lock = copy.deepcopy(_load_lock())
    mutate(lock)
    _resign(lock)  # a coordinated edit must still fail semantic binding
    with pytest.raises(ValueError, match=message):
        runner.validate_task14_lock(
            lock,
            worktree=ROOT,
            observed_runner_sha256=_sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py"),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda lock: lock["n4_recipe"]["fit_recipe"].__setitem__("epochs", N4_SCALE_EPOCHS + 1),
            "fit_recipe.epochs contradicts the frozen execution recipe",
        ),
        (
            lambda lock: lock["n4_recipe"]["fit_recipe"].__setitem__("batch_size", N4_SCALE_BATCH_SIZE * 2),
            "fit_recipe.batch_size contradicts the frozen execution recipe",
        ),
        (
            lambda lock: lock["n4_recipe"]["fit_recipe"].__setitem__("learning_rate", N4_SCALE_LR * 2),
            "fit_recipe.learning_rate contradicts the frozen execution recipe",
        ),
        (
            lambda lock: lock["n4_recipe"]["fit_recipe"].__setitem__(
                "gradient_clip_norm", N4_SCALE_CLIP_NORM * 0.5
            ),
            "fit_recipe.gradient_clip_norm contradicts the frozen execution recipe",
        ),
        (
            lambda lock: lock["n4_recipe"]["architecture"].__setitem__("hidden_width", 64),
            "architecture.hidden_width contradicts the frozen execution recipe",
        ),
        (
            lambda lock: lock["n4_recipe"]["v3_amendment"].__setitem__(
                "saturation_cap_gate", N4_SATURATION_CAP_GATE + 0.01
            ),
            "v3_amendment.saturation_cap_gate contradicts the frozen execution recipe",
        ),
        (
            lambda lock: lock["n4_recipe"]["v3_amendment"].__setitem__(
                "parameter_limit", N4_SCALE_PARAM_LIMIT + 1
            ),
            "v3_amendment.parameter_limit contradicts the frozen execution recipe",
        ),
        (
            lambda lock: lock["arms"]["N4-CRN"].__setitem__("scale_epochs", N4_SCALE_EPOCHS + 1),
            "arms.N4-CRN.scale_epochs contradicts the frozen execution recipe",
        ),
    ],
)
def test_v8_lock_runner_refuses_coordinated_recipe_tamper(mutate, message) -> None:
    """A re-digested lock whose recipe contradicts actual execution is refused."""
    lock = copy.deepcopy(_load_lock())
    mutate(lock)
    _resign(lock)
    _rebuild_locked_configs(lock)  # the full coordinated edit reported by review
    with pytest.raises(ValueError, match=message):
        runner.validate_task14_lock(
            lock,
            worktree=ROOT,
            observed_runner_sha256=_sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py"),
        )


def test_v8_lock_runner_refuses_carried_recipe_contradiction() -> None:
    """Recipe prose outside the live constants still must equal the pinned recipe."""
    lock = copy.deepcopy(_load_lock())
    lock["n4_recipe"]["fit_recipe"]["optimizer_step_formula"] = "per scope 51*ceil(N_scope/512)"
    _resign(lock)
    _rebuild_locked_configs(lock)
    with pytest.raises(ValueError, match="differs from the pinned historical recipe"):
        runner.validate_task14_lock(
            lock,
            worktree=ROOT,
            observed_runner_sha256=_sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py"),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda lock: lock["n4_scale_scope"].__setitem__(
                "eligible_masked_fit_patch_records", 167419
            ),
            "in-scope record total differs",
        ),
        (
            lambda lock: lock["n4_scale_scope"]["fold_record_counts"].__setitem__(0, 44662),
            "fold record counts differ",
        ),
        (
            lambda lock: lock["n4_scale_scope"].__setitem__("final_optimizer_steps", 16350),
            "final optimizer steps differ",
        ),
        (
            lambda lock: lock["n4_v3_activation"]["capacity_certification"][
                "fold_checkpoint_sha256"
            ]["181801"].__setitem__(0, "0" * 64),
            "declared fold checkpoint digests differ",
        ),
        (
            lambda lock: lock["n4_v3_activation"]["capacity_certification"].__setitem__(
                "total_optimizer_updates", 39451
            ),
            "declared capacity-certification summary differs",
        ),
    ],
)
def test_v8_lock_runner_refuses_scope_and_certification_tamper(mutate, message) -> None:
    """The in-scope record contract and certified capacity evidence are load-bearing."""
    lock = copy.deepcopy(_load_lock())
    mutate(lock)
    _resign(lock)
    _rebuild_locked_configs(lock)
    with pytest.raises(ValueError, match=message):
        runner.validate_task14_lock(
            lock,
            worktree=ROOT,
            observed_runner_sha256=_sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py"),
        )


def test_v8_lock_runner_refuses_dropped_implementation_pin() -> None:
    lock = copy.deepcopy(_load_lock())
    lock["n4_v3_activation"]["reviewed_implementation"]["paths"].pop(
        "experiments/sprint18_task30_n4_runner.py"
    )
    _resign(lock)
    _rebuild_locked_configs(lock)
    with pytest.raises(ValueError, match="reviewed implementation scope differs"):
        runner.validate_task14_lock(
            lock,
            worktree=ROOT,
            observed_runner_sha256=_sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py"),
        )


def test_v8_lock_runner_refuses_inconsistent_digest_and_wrong_runner() -> None:
    lock = copy.deepcopy(_load_lock())
    lock["n4_v3_activation"]["guard_contract"] = "legacy-guards"
    with pytest.raises(ValueError, match="core digest is inconsistent"):
        runner.validate_task14_lock(
            lock,
            worktree=ROOT,
            observed_runner_sha256=_sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py"),
        )
    lock = _load_lock()
    with pytest.raises(ValueError, match="runner digest differs"):
        runner.validate_task14_lock(lock, worktree=ROOT, observed_runner_sha256="0" * 64)


def test_v8_lock_runner_refuses_stale_historical_and_superseded_locks() -> None:
    runner_sha = _sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py")
    for stale in (
        _load_legacy_lock(),
        _load_superseded_lock(),
        _load_v6_lock(),
        _load_v5_lock(),
        _load_v4_lock(),
        _load_v3_lock(),
        _load_v2_lock(),
    ):
        with pytest.raises(ValueError, match="schema is not"):
            runner.validate_task14_lock(stale, worktree=ROOT, observed_runner_sha256=runner_sha)


def test_v8_lock_b0_restore_arms_bind_historical_v1_lock() -> None:
    """The B0-FULL strict restores keep the v1 boundary; N4 keeps the current lock.

    All three final epoch-100 B0 checkpoints were produced under the v1 lock.
    The prospective v8 lock supersedes it for N4-CRN only, so a B0 restore arm
    built with the v8 core digest is a wrong-lock binding, not a weight change.
    """
    lock = _load_lock()
    legacy = _load_legacy_lock()
    record = runner.load_legacy_b0_lock(ROOT)
    assert record["schema"] == "s18-task14-config-lock-v1"
    assert record["file_sha256"] == LEGACY_LOCK_SHA256
    assert record["coefficient_lock_sha256"] == LEGACY_COEFFICIENT_LOCK_SHA256
    assert record["coefficient_lock_sha256"] != lock["coefficient_lock_sha256"]
    b0_coefficients = {
        "lambda_info_max": 0.05,
        "lambda_hcc_max": None,
        "alpha": None,
        "beta": None,
    }
    for seed in MODEL_SEEDS:
        key = f"B0-FULL/seed-{seed}"
        arm = make_sprint18_arm_config(
            "B0-FULL",
            seed,
            b0_coefficients,
            selection_state="locked",
            coefficient_lock_sha256=record["coefficient_lock_sha256"],
        )
        validate_arm_config(arm, require_locked=True)
        assert arm.model_dump(mode="json") == legacy["locked_configs"][key]
        assert arm_config_sha256(arm) == legacy["config_sha256"][key]
        assert record["config_sha256"][str(seed)] == legacy["config_sha256"][key]
        wrong = make_sprint18_arm_config(
            "B0-FULL",
            seed,
            b0_coefficients,
            selection_state="locked",
            coefficient_lock_sha256=lock["coefficient_lock_sha256"],
        )
        differing = {
            name
            for name, value in arm.model_dump(mode="json").items()
            if value != wrong.model_dump(mode="json")[name]
        }
        assert differing == {"coefficient_lock_sha256"}
        assert wrong.model_dump(mode="json") != legacy["locked_configs"][key]
        # The N4-CRN arm on the same seed keeps the current prospective digest.
        n4_arm = make_sprint18_arm_config(
            "N4-CRN",
            seed,
            COEFFICIENTS,
            selection_state="locked",
            coefficient_lock_sha256=lock["coefficient_lock_sha256"],
        )
        assert n4_arm.model_dump(mode="json") == lock["locked_configs"][f"N4-CRN/seed-{seed}"]


def test_v8_lock_b0_restore_refuses_unpinned_legacy_lock(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    with pytest.raises(ValueError, match="historical Task14 v1 lock is missing"):
        runner.load_legacy_b0_lock(tmp_path)
    tampered = _load_legacy_lock()
    tampered["coefficient_lock_sha256"] = _load_lock()["coefficient_lock_sha256"]
    (experiments / LEGACY_LOCK_PATH.name).write_text(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="historical Task14 lock bytes changed"):
        runner.load_legacy_b0_lock(tmp_path)
