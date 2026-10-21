"""Versioned prospective Task14 N4-CRN v3 lock regression test.

Protects the lock that activates the reviewed N4 v3 initialization/guard
contract. The historical v1 lock and the v1/v2 specifications must stay
byte-identical; the v3 lock must recompute its canonical core digest,
re-materialize the three N4-CRN seed configs through the real arm-config
consumer without reclaiming the other arms, and the runner's lock consumer
must accept it while refusing stale or tampered variants.

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

LOCK_PATH = ROOT / "experiments" / "sprint18-task14-config-lock-v2.json"
LEGACY_LOCK_PATH = ROOT / "experiments" / "sprint18-task14-config-lock.json"
SPEC_V1_PATH = ROOT / "experiments" / "sprint18-context-residual-normalization-spec.md"
SPEC_V2_PATH = ROOT / "experiments" / "sprint18-context-residual-normalization-spec-v2.md"
SPEC_V3_PATH = ROOT / "experiments" / "sprint18-context-residual-normalization-spec-v3.md"

LEGACY_LOCK_SHA256 = "c4c1a8bde14310cf4436c10762e3e98475f7c84eb74fcb8a74b3e267885edec3"
LEGACY_COEFFICIENT_LOCK_SHA256 = (
    "6eb108b2073181d427f1aedfc81fb3bae4c29595108d15de486082800bdd9d6e"
)
SPEC_V1_SHA256 = "fd0e1722ea4f248a354086bbdf44bc486d741a9cfb018f68ef08277827d7954d"
SPEC_V2_SHA256 = "4d1c8097c9cc27d3fbeb2e2a6e70a29a7bc63d7899646a9c712a315b39d9ffe3"
SPEC_V3_SHA256 = "e7fe42e41832ae283614ee6a76c0623ee567068dcc540a222f5b66a9d4262da2"
REVIEWED_RUNNER_SHA256 = "09f7beaa9c8b77217d0bd9cdd5b142e5467d02d8bdc6a77c6e22a4a1344fa784"
FIT_HISTORIES = [
    "S18I-ITER-0005-FIT-01",
    "S18I-ITER-0005-FIT-02",
    "S18I-ITER-0005-FIT-03",
]
COEFFICIENTS = {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_lock() -> dict:
    return json.loads(LOCK_PATH.read_bytes().decode("utf-8"))


def _load_legacy_lock() -> dict:
    return json.loads(LEGACY_LOCK_PATH.read_bytes().decode("utf-8"))


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


def test_lock_v3_digest_supersedes_and_preserved_history() -> None:
    lock = _load_lock()
    assert lock["schema"] == "s18-task14-config-lock-v2"
    assert lock["lock_version"] == 2
    assert lock["task_id"] == "S18-T14"
    assert lock["source_commit"] == "44f56b195597c1c3a46564d63c7269bb29d8c05d"
    assert lock["source_parent"] == "fe17b9ed9bb76d353ed45db9344198461d6f8e78"
    assert lock["target_ref"] == "refs/heads/master"
    assert lock["created_by"] == {
        "task_id": "S18-T13B-LOCK-V3",
        "attempt_id": "S18T13BLOCKV3C01",
        "role": "gold-task",
        "route": "full",
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
    assert lock["coefficient_lock_sha256"] != LEGACY_COEFFICIENT_LOCK_SHA256
    assert lock["coefficient_selections"] == _load_legacy_lock()["coefficient_selections"]
    assert lock["supersedes"] == {
        "path": "experiments/sprint18-task14-config-lock.json",
        "schema": "s18-task14-config-lock-v1",
        "file_sha256": LEGACY_LOCK_SHA256,
        "coefficient_lock_sha256": LEGACY_COEFFICIENT_LOCK_SHA256,
        "scope": (
            "supersedes the v1 lock for N4-CRN only; the v1 lock bytes, every "
            "other arm and all historical evidence remain unchanged"
        ),
    }
    # Historical bytes are preserved exactly.
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


def test_lock_v3_n4_provenance_matches_live_contract() -> None:
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
    assert activation["status"] == "prospective_locked_pending_independent_review_and_bronze_checkpoint"

    implementation = activation["reviewed_implementation"]
    assert implementation["source_commit"] == lock["source_commit"]
    assert implementation["paths"]["experiments/sprint18-context-residual-normalization-spec-v3.md"] == SPEC_V3_SHA256
    for relative, expected in implementation["paths"].items():
        assert _sha256(ROOT / relative) == expected
    assert lock["runner_identity"]["sha256"] == _sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py")
    assert lock["runner_identity"]["reviewed_source_commit_sha256"] == REVIEWED_RUNNER_SHA256
    assert lock["runner_identity"]["future_activation_commit_frozen"] is False
    assert implementation["review"]["verdict"] == "PASS"


def test_lock_v3_n4_configs_rebuild_through_real_consumer() -> None:
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


def test_v3_lock_runner_accepts_reviewed_lock() -> None:
    lock = _load_lock()
    record = runner.validate_task14_v3_lock(
        lock,
        worktree=ROOT,
        observed_runner_sha256=_sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py"),
    )
    assert record["schema"] == "s18-task14-config-lock-v2"
    assert record["lock_version"] == 2
    assert record["coefficient_lock_sha256"] == lock["coefficient_lock_sha256"]
    assert record["file_sha256"] == _sha256(LOCK_PATH)
    assert record["mechanism_spec_sha256"] == SPEC_V3_SHA256
    assert record["mechanism_spec_base_sha256"] == SPEC_V2_SHA256
    assert record["source_commit"] == "44f56b195597c1c3a46564d63c7269bb29d8c05d"
    assert record["supersedes"]["file_sha256"] == LEGACY_LOCK_SHA256


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda lock: lock.__setitem__("schema", "s18-task14-config-lock-v1"),
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
            "supersedes the wrong historical lock bytes",
        ),
    ],
)
def test_v3_lock_runner_refuses_tampered_lock(mutate, message) -> None:
    lock = copy.deepcopy(_load_lock())
    mutate(lock)
    _resign(lock)  # a coordinated edit must still fail semantic binding
    with pytest.raises(ValueError, match=message):
        runner.validate_task14_v3_lock(
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
def test_v3_lock_runner_refuses_coordinated_recipe_tamper(mutate, message) -> None:
    """A re-digested lock whose recipe contradicts actual execution is refused."""
    lock = copy.deepcopy(_load_lock())
    mutate(lock)
    _resign(lock)
    _rebuild_locked_configs(lock)  # the full coordinated edit reported by review
    with pytest.raises(ValueError, match=message):
        runner.validate_task14_v3_lock(
            lock,
            worktree=ROOT,
            observed_runner_sha256=_sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py"),
        )


def test_v3_lock_runner_refuses_carried_recipe_contradiction() -> None:
    """Recipe prose outside the live constants still must equal the pinned recipe."""
    lock = copy.deepcopy(_load_lock())
    lock["n4_recipe"]["fit_recipe"]["optimizer_step_formula"] = "per scope 51*ceil(N_scope/512)"
    _resign(lock)
    _rebuild_locked_configs(lock)
    with pytest.raises(ValueError, match="differs from the pinned historical recipe"):
        runner.validate_task14_v3_lock(
            lock,
            worktree=ROOT,
            observed_runner_sha256=_sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py"),
        )


def test_v3_lock_runner_refuses_inconsistent_digest_and_wrong_runner() -> None:
    lock = copy.deepcopy(_load_lock())
    lock["n4_v3_activation"]["guard_contract"] = "legacy-guards"
    with pytest.raises(ValueError, match="core digest is inconsistent"):
        runner.validate_task14_v3_lock(
            lock,
            worktree=ROOT,
            observed_runner_sha256=_sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py"),
        )
    lock = _load_lock()
    with pytest.raises(ValueError, match="runner digest differs"):
        runner.validate_task14_v3_lock(lock, worktree=ROOT, observed_runner_sha256="0" * 64)


def test_v3_lock_runner_refuses_stale_historical_lock() -> None:
    with pytest.raises(ValueError, match="schema is not"):
        runner.validate_task14_v3_lock(
            _load_legacy_lock(),
            worktree=ROOT,
            observed_runner_sha256=_sha256(ROOT / "experiments" / "sprint18_task13b_n4_capacity.py"),
        )
