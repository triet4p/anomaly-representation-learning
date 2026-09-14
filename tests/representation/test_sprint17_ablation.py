"""Focused contract tests for the execution-free Sprint 17 Task 5 harness."""
from __future__ import annotations

from dataclasses import replace

import pytest

from representation.sprint17_ablation import (
    ARM_REGISTRY,
    CHECKPOINT_STEP,
    FLOP_DELTA_LIMIT,
    MODEL_SEEDS,
    OPTIMIZER_STEPS,
    PARAM_DELTA_LIMIT,
    QUALIFICATION,
    WAIVER_ID,
    WAIVER_SCOPE,
    ArmSpec,
    ComputeEnvelope,
    TrainingParity,
    dry_contract,
    evaluate_paired,
    get_arm,
    validate_compute_envelope,
    validate_dry_contract,
    validate_eligibility,
    validate_provenance,
    validate_training_parity,
)


EXPECTED_ARMS = {
    "B0",
    *(f"C{component}-{suffix}" for component in (2, 3, 5, 6, 7, 8, 9) for suffix in ("A", "B")),
    *(f"K{i}" for i in range(1, 10)),
}


def test_registry_is_complete_and_exact() -> None:
    assert set(ARM_REGISTRY) == EXPECTED_ARMS
    assert len(ARM_REGISTRY) == 24
    assert get_arm("B0").components == ()
    assert get_arm("C2-A").components == ("C2",)
    assert get_arm("K1").components == ("C2", "C3")
    assert get_arm("K9").components == ("C2", "C3", "C5", "C6", "C7", "C8", "C9")


def test_malformed_arm_and_combination_membership_rejected() -> None:
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_arm = get_arm("C2-A")
        validate_arm = replace(validate_arm, components=("C3",))
        from representation.sprint17_ablation import validate_arm as check_arm
        check_arm(validate_arm)
    with pytest.raises(ValueError, match="metadata mismatch"):
        malformed = replace(get_arm("K1"), components=("C2", "C5"))
        from representation.sprint17_ablation import validate_arm as check_arm
        check_arm(malformed)
    with pytest.raises(ValueError, match="unknown Sprint 17 arm"):
        get_arm("K10")


def test_training_seed_budget_and_parity_guards() -> None:
    assert validate_training_parity(TrainingParity()).model_seeds == MODEL_SEEDS
    assert OPTIMIZER_STEPS == CHECKPOINT_STEP == 300
    with pytest.raises(ValueError, match="model seeds"):
        validate_training_parity(replace(TrainingParity(), model_seeds=(1, 2, 3)))
    with pytest.raises(ValueError, match="steps"):
        validate_training_parity(replace(TrainingParity(), optimizer_steps=301))
    with pytest.raises(ValueError, match="warm_start"):
        validate_training_parity(replace(TrainingParity(), warm_start=True))
    with pytest.raises(ValueError, match="extra_runs"):
        validate_training_parity(replace(TrainingParity(), extra_runs=1))

    for field, value in (
        ("optimizer", "other-optimizer"),
        ("schedule", "other-schedule"),
        ("batch_construction", "other-batches"),
        ("precision", "float32"),
        ("device_class", "cpu"),
        ("initialization_policy", "other-init"),
    ):
        with pytest.raises(ValueError, match=field):
            validate_training_parity(replace(TrainingParity(), **{field: value}))
    for field, value in (("gradient_clipping", -1.0), ("weight_decay", -1.0), ("gradient_clipping", float("nan")), ("weight_decay", True)):
        with pytest.raises(ValueError, match=field):
            validate_training_parity(replace(TrainingParity(), **{field: value}))


def test_parameter_and_flop_envelopes_are_bounded() -> None:
    baseline = ComputeEnvelope(100, 100, 1000, 1000)
    assert validate_compute_envelope(baseline).params_delta_fraction == 0.0
    assert validate_compute_envelope(ComputeEnvelope(100, 105, 1000, 1100)) == ComputeEnvelope(100, 105, 1000, 1100)
    assert PARAM_DELTA_LIMIT == 0.05 and FLOP_DELTA_LIMIT == 0.10
    with pytest.raises(ValueError, match="±5%"):
        validate_compute_envelope(ComputeEnvelope(100, 106, 1000, 1000))
    with pytest.raises(ValueError, match="±10%"):
        validate_compute_envelope(ComputeEnvelope(100, 100, 1000, 1101))
    for field in ("b0_params", "params_total", "b0_flops_per_reference_sample", "flops_per_reference_sample"):
        with pytest.raises(ValueError, match="positive integers"):
            validate_compute_envelope(replace(baseline, **{field: True}))


def test_exact_waiver_is_required() -> None:
    eligibility = {"qualification": QUALIFICATION, "waiver_id": WAIVER_ID, "waiver_scope": WAIVER_SCOPE}
    assert validate_eligibility(eligibility) == eligibility
    for bad in (
        {},
        {**eligibility, "qualification": "MEASURABLE"},
        {**eligibility, "waiver_id": "other"},
        {**eligibility, "waiver_scope": {**WAIVER_SCOPE, "cohort": "W"}},
    ):
        with pytest.raises(ValueError):
            validate_eligibility(bad)


def _provenance() -> dict[str, object]:
    return {
        "data_protocol": "sprint15-benchmark-protocol-v7",
        "role_binding_sha256": "a" * 64,
        "data_roles": ["H-S17-V3-DEV-01"],
        "data_seeds": [2808],
        "model_seeds": list(MODEL_SEEDS),
        "fit_root_sha256": "b" * 64,
        "calibration_root_sha256": "c" * 64,
        "evaluation_root_sha256": "d" * 64,
        "metric_code_sha256": "e" * 64,
        "checkpoint_sha256": "f" * 64,
        "cache_sha256": "0" * 64,
        "parent_arm_ids": [],
    }


def test_provenance_rejects_missing_or_mismatched_cache() -> None:
    assert validate_provenance(_provenance())["cache_sha256"] == "0" * 64
    missing = _provenance()
    missing.pop("cache_sha256")
    with pytest.raises(ValueError, match="incomplete"):
        validate_provenance(missing)
    wrong_cache = _provenance()
    wrong_cache["cache_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="cache_sha256"):
        validate_provenance(wrong_cache)
    wrong_role = _provenance()
    wrong_role["data_roles"] = ["H-S17-V2-DEV-01"]
    with pytest.raises(ValueError, match="data_roles"):
        validate_provenance(wrong_role)
    extra = _provenance()
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="extra"):
        validate_provenance(extra)

    for parents in (["K1"], ["not-an-arm"], [1], "B0", ["B0", "B0"]):
        bad_parents = _provenance()
        bad_parents["parent_arm_ids"] = parents
        with pytest.raises(ValueError, match="parent_arm_ids"):
            validate_provenance(bad_parents)


def test_paired_evaluation_is_deterministic_and_requires_identical_support() -> None:
    arm = {"S_pred": {"z": 3.0, "a": 2.0}, "S_pop": {"z": 4.0, "a": 5.0}}
    b0 = {"S_pred": {"z": 1.0, "a": 1.0}, "S_pop": {"z": 2.0, "a": 3.0}}
    first = evaluate_paired("C2-A", "H-S17-V3-DEV-01", arm, b0, ["z", "a"], branch="S_pred")
    second = evaluate_paired("C2-A", "H-S17-V3-DEV-01", arm, b0, ["a", "z"], branch="S_pred")
    assert first == second
    assert first.support_ids == ("a", "z") and first.deltas == (1.0, 2.0)
    mismatch = {"S_pred": {"z": 1.0}, "S_pop": {"z": 2.0}}
    with pytest.raises(ValueError, match="match exactly"):
        evaluate_paired("C2-A", "H-S17-V3-DEV-01", arm, mismatch, ["a", "z"], branch="S_pred")


def test_paired_evaluation_rejects_branch_collision_nonfinite_and_missing_branch() -> None:
    same = {"z": 1.0}
    with pytest.raises(ValueError, match="alias"):
        evaluate_paired("B0", "H-S17-V3-DEV-01", {"S_pred": same, "S_pop": same}, {"S_pred": {"z": 0.0}, "S_pop": {"z": 0.0}}, ["z"], branch="S_pred")
    with pytest.raises(ValueError, match="finite"):
        evaluate_paired("B0", "H-S17-V3-DEV-01", {"S_pred": {"z": float("nan")}, "S_pop": {"z": 0.0}}, {"S_pred": {"z": 0.0}, "S_pop": {"z": 0.0}}, ["z"], branch="S_pred")
    with pytest.raises(ValueError, match="both independent"):
        evaluate_paired("B0", "H-S17-V3-DEV-01", {"S_pred": {"z": 1.0}}, {"S_pred": {"z": 0.0}, "S_pop": {"z": 0.0}}, ["z"], branch="S_pred")


def test_public_dry_contract_smoke() -> None:
    for arm_id in ("B0", "C2-A", "K1"):
        contract = dry_contract(arm_id)
        assert validate_dry_contract(contract) == contract
    contract = dry_contract("C2-A")
    expected = {
        "batch_construction": "B0-frozen-batch-construction",
        "gradient_clipping": 1.0,
        "weight_decay": 0.0,
        "initialization_policy": "B0-frozen-initialization",
    }
    assert all(contract["config"][key] == value for key, value in expected.items())
    for key, value in expected.items():
        bad = {**contract, "config": {**contract["config"], key: "changed"}}
        with pytest.raises(ValueError, match="config"):
            validate_dry_contract(bad)
