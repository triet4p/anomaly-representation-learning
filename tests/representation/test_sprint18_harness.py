from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from representation.config import V1Config
from representation.model import V1RepresentationModel
from representation.sprint18_checkpoint import (
    CHECKPOINT_SCHEMA_ID,
    coefficient_for_epoch,
    learning_rate_for_epoch,
    restore_sprint18_checkpoint,
    save_sprint18_checkpoint,
    state_sha256,
    validate_sprint18_checkpoint,
    _validate_coefficient_state,
    _validate_sampler_state,
)
from representation.sprint18_harness import (
    BATCH_SIZE,
    COMMON_INFO_COEFFICIENT_GRID,
    ARM_REGISTRY,
    DATA_ROSTER_ID,
    MECHANISM_SPEC_SHA256,
    MODEL_SEEDS,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    RESULT_SCHEMA_ID,
    Sprint18ArmConfig,
    Task6Binding,
    arm_spec,
    count_unique_parameters,
    deterministic_sampler_order,
    load_task6_binding,
    make_common_support,
    load_certified_healthy_role_support,
    make_sprint18_arm_config,
    profile_inference_flops,
    TASK9_COEFFICIENT_PILOT_EPOCHS,
    validate_arm_config,
    select_common_info_coefficient,
    validate_arm_spec,
    validate_compute_envelope,
    validate_paired_common_support,
    validate_role_permission,
    validate_score_maps,
    validate_sprint18_result,
)
ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "experiments" / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"


def _coefficients(arm_id: str) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "lambda_info_max": None,
        "lambda_hcc_max": None,
        "alpha": None,
        "beta": None,
    }
    if arm_id in {"B0-FULL", "K3-FULL", "K7-FULL", "K9-FULL", "N2-MSP", "N3-CCRP"}:
        values["lambda_info_max"] = 0.025
    if arm_id == "N1-HCC":
        values["lambda_hcc_max"] = 0.025
    if arm_id == "N2-MSP":
        values["alpha"] = 0.1
    if arm_id == "N3-CCRP":
        values["beta"] = 0.1
    return values


def _smoke_config(arm_id: str = "B0-FULL") -> Sprint18ArmConfig:
    return make_sprint18_arm_config(
        arm_id,
        MODEL_SEEDS[0],
        _coefficients(arm_id),
        selection_state="task6_smoke_only",
    )


def test_registry_binds_exact_eight_identities_and_principal_changes() -> None:
    assert set(ARM_REGISTRY) == {
        "B0-FULL", "K3-FULL", "K7-FULL", "K9-FULL",
        "N1-HCC", "N2-MSP", "N3-CCRP", "N4-CRN",
    }
    assert arm_spec("K3-FULL").component_set == ("C5-A", "C6-A")
    assert arm_spec("K7-FULL").component_set == ("C2-A", "C3-A", "C5-A", "C6-A")
    assert arm_spec("K9-FULL").component_set == (
        "C2-A", "C3-A", "C5-A", "C6-A", "C7-A", "C8-A", "C9-B"
    )
    for arm_id in ARM_REGISTRY:
        assert validate_arm_spec(arm_spec(arm_id)) is ARM_REGISTRY[arm_id]
    with pytest.raises(ValueError, match="unknown Sprint18 arm"):
        arm_spec("N5-UNREGISTERED")
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_arm_spec(replace(arm_spec("N1-HCC"), component_set=("N1-HCC", "N2-MSP")))


def test_arm_config_rejects_semantic_drift_and_v1_identity() -> None:
    for arm_id in ARM_REGISTRY:
        assert validate_arm_config(_smoke_config(arm_id), require_locked=False).arm_id == arm_id
    invalid = _smoke_config("N2-MSP").model_copy(update={"principal_change": "replace InfoNCE"})
    with pytest.raises(ValueError, match="principal-change semantics"):
        validate_arm_config(invalid, require_locked=False)
    invalid_cardinality = _smoke_config().model_copy(update={"n_robots": 5})
    with pytest.raises(ValueError, match="conditioning cardinalities"):
        validate_arm_config(invalid_cardinality, require_locked=False)
    invalid_protocol = _smoke_config().model_copy(update={"protocol_sha256": "0" * 64})
    with pytest.raises(ValueError, match="accepted v2 protocol"):
        validate_arm_config(invalid_protocol, require_locked=False)
    with pytest.raises(ValueError, match="coefficient-lock digest"):
        validate_arm_config(make_sprint18_arm_config("B0-FULL", MODEL_SEEDS[0], _coefficients("B0-FULL")))


def test_sampler_uses_frozen_representation_and_n4_seed_text() -> None:
    binding = load_task6_binding(HANDOFF)
    fit_histories = binding.role_histories("FIT")
    fit_ids = tuple(item.actual_history_id for item in fit_histories)
    source_ids = ("file-b", "file-a")
    epoch = 7
    fit_role_sha256 = hashlib.sha256(
        json.dumps(
            sorted(item.manifest_sha256 for item in fit_histories),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    b0 = make_sprint18_arm_config(
        "B0-FULL",
        MODEL_SEEDS[0],
        _coefficients("B0-FULL"),
        coefficient_lock_sha256=_hash("coefficient-lock"),
    )
    seed, order = deterministic_sampler_order(
        binding,
        b0,
        workload="representation",
        scope="experiment",
        role_history_ids=fit_ids,
        epoch=epoch,
        eligible_source_ids=source_ids,
    )
    expected_seed = int.from_bytes(
        hashlib.sha256(
            f"sprint18-fit-order-v1|{b0.model_seed}|{epoch}|{fit_role_sha256}".encode("utf-8")
        ).digest()[:8],
        "big",
    ) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(expected_seed)
    expected_order = tuple(
        sorted(source_ids)[index]
        for index in torch.randperm(len(source_ids), generator=generator).tolist()
    )
    assert seed == expected_seed
    assert order == expected_order

    n4 = make_sprint18_arm_config(
        "N4-CRN",
        MODEL_SEEDS[0],
        _coefficients("N4-CRN"),
        coefficient_lock_sha256=_hash("coefficient-lock"),
    )
    for role_ids, fit_scope in (
        (fit_ids, "final"),
        (tuple(history_id for history_id in fit_ids if history_id != fit_ids[1]), f"fold:{fit_ids[1]}"),
    ):
        seed, order = deterministic_sampler_order(
            binding,
            n4,
            workload="n4_scale",
            scope="experiment",
            role_history_ids=role_ids,
            epoch=epoch,
            eligible_source_ids=source_ids,
        )
        seed_text = (
            f"sprint18-n4-scale-order-v2|{n4.model_seed}|{fit_scope}|"
            f"{','.join(sorted(role_ids))}|{epoch}"
        )
        expected_seed = int.from_bytes(
            hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big"
        ) % (2**63 - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(expected_seed)
        expected_order = tuple(
            sorted(source_ids)[index]
            for index in torch.randperm(len(source_ids), generator=generator).tolist()
        )
        assert seed == expected_seed
        assert order == expected_order

    with pytest.raises(ValueError, match="history IDs must be non-empty and unique"):
        deterministic_sampler_order(
            binding,
            n4,
            workload="n4_scale",
            scope="experiment",
            role_history_ids=(fit_ids[0], fit_ids[0]),
            epoch=epoch,
            eligible_source_ids=source_ids,
        )

def test_binding_reconciles_task71_history_ids_without_reusing_task3_roster() -> None:
    binding = load_task6_binding(HANDOFF)
    assert binding.candidate_id == DATA_ROSTER_ID
    assert binding.protocol_sha256 == PROTOCOL_SHA256
    assert binding.handoff_sha256 == hashlib.sha256(HANDOFF.read_bytes()).hexdigest()
    assert len(binding.histories) == 16
    fit = binding.role_histories("FIT")
    assert tuple(row.actual_history_id for row in fit) == (
        "S18I-ITER-0005-FIT-01", "S18I-ITER-0005-FIT-02", "S18I-ITER-0005-FIT-03"
    )
    assert tuple(row.schema_history_id for row in fit) == (
        "H-S18-FIT-01", "H-S18-FIT-02", "H-S18-FIT-03"
    )
    assert all(row.relative_path.startswith(binding.data_root_relative + "/FIT/") for row in fit)
    assert fit[0].manifest_sha256 == "fb0d533c10f1a650647b95d06c87559f244f09a31973c75a35329fd8a04d19dc"
    assert binding.executable_binding_sha256 != binding.data_binding_raw_sha256


def test_role_permissions_fail_closed_by_stage_and_confirmation_gate() -> None:
    binding = load_task6_binding(HANDOFF)
    fit_id = binding.role_histories("FIT")[0].actual_history_id
    cal_id = binding.role_histories("CALIBRATION")[0].actual_history_id
    conf_id = binding.role_histories("CONFIRMATION")[0].actual_history_id
    assert validate_role_permission(binding, fit_id, "fit_bank", healthy=True).assigned_role == "FIT"
    assert validate_role_permission(binding, cal_id, "calibrate_q95", healthy=True).assigned_role == "CALIBRATION"
    with pytest.raises(ValueError, match="not authorized"):
        validate_role_permission(binding, cal_id, "fit_bank", healthy=True)
    with pytest.raises(ValueError, match="verified-healthy"):
        validate_role_permission(binding, fit_id, "fit_bank", healthy=False)
    with pytest.raises(ValueError, match="Confirmation is closed"):
        validate_role_permission(binding, conf_id, "score_confirmation")
    assert validate_role_permission(
        binding, conf_id, "score_confirmation", confirmation_lock_sha256="1" * 64,
        confirmation_review_sha256="2" * 64, main_confirmation_release=True,
    ).assigned_role == "CONFIRMATION"


def test_common_support_requires_both_distinct_branches_and_exact_keys() -> None:
    binding = load_task6_binding(HANDOFF)
    fit_id = binding.role_histories("FIT")[0].actual_history_id
    support = make_common_support(binding, fit_id, ["file-b", "file-a"])
    assert support.file_ids == ("file-a", "file-b")
    scores = {"S_pred": {"file-a": 1.0, "file-b": 2.0}, "S_pop": {"file-a": 3.0, "file-b": 4.0}}
    assert validate_score_maps(support, scores) is scores
    assert validate_paired_common_support(support, scores, {"S_pred": {"file-a": 0.5, "file-b": 1.0}, "S_pop": {"file-a": 1.0, "file-b": 1.5}}, branch="S_pred") == (0.5, 1.0)
    aliased = {"S_pred": scores["S_pred"], "S_pop": scores["S_pred"]}
    with pytest.raises(ValueError, match="must not alias"):
        validate_score_maps(support, aliased)
    with pytest.raises(ValueError, match="exactly independent"):
        validate_score_maps(support, {"S_pred": scores["S_pred"]})
    with pytest.raises(ValueError, match="fixed common support"):
        validate_score_maps(support, {"S_pred": {"file-a": 1.0}, "S_pop": scores["S_pop"]})
    with pytest.raises(ValueError, match="nonfinite"):
        validate_score_maps(support, {"S_pred": {"file-a": float("nan"), "file-b": 2.0}, "S_pop": scores["S_pop"]})


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _history_metric(history_id: str) -> dict[str, object]:
    metrics = {
        "P_AUROC": None, "W_AUROC": None, "PW_AUROC": None, "A_AUROC": None,
        "delta_P": None, "delta_W": None, "delta_PW": None,
        "P_recall": None, "P_median_lead_days": None, "W_recall": None,
        "W_median_lead_days": None, "FAR_per_robot_day": None, "delta_FAR": None,
        "background_stability_ratio": None, "localization": None,
        "severity_ordering": {"P": None, "W": None},
        "subtype_slices": {"P1": None, "P2": None, "W1": None, "W2": None, "A1": None, "A2": None},
    }
    return {"history_id": history_id, **metrics}


def _failed_result(arm_id: str = "B0-FULL", phase: str = "development") -> dict[str, object]:
    spec = arm_spec(arm_id)
    prefix = "DEV" if phase == "development" else "CONF"
    history_ids = [f"H-S18-{prefix}-{index:02d}" for index in range(1, 5)]
    coeffs = _coefficients(arm_id)
    config = {
        "model_seeds": list(MODEL_SEEDS), "optimizer": "AdamW",
        "batch_size": 512 if arm_id == "N4-CRN" else 128,
        "batch_size_unit": "patch_records" if arm_id == "N4-CRN" else "files",
        "learning_rate": 1e-3 if arm_id == "N4-CRN" else 1.5e-3,
        "weight_decay": 1e-4 if arm_id == "N4-CRN" else 0.0,
        "clip_norm": 1.0, "precision": "float32",
        "representation_epochs": spec.representation_epochs,
        "scale_epochs": spec.scale_epochs, "ema_decay": 0.996,
        "lambda_info_max": coeffs["lambda_info_max"],
        "lambda_hcc_max": coeffs["lambda_hcc_max"], "alpha": coeffs["alpha"],
        "beta": coeffs["beta"],
        "primary_checkpoint_epoch": 50 if arm_id == "N4-CRN" else 100,
    }
    provenance = {
        "protocol_sha256": PROTOCOL_SHA256,
        "role_binding_sha256": _hash("roles"), "fit_role_sha256": _hash("fit"),
        "calibration_role_sha256": _hash("cal"), "evaluation_role_sha256": _hash("eval"),
        "config_sha256": _hash("config"), "coefficient_lock_sha256": _hash("coeff"),
        "metric_code_sha256": _hash("metrics"), "compute_counter_sha256": _hash("compute"),
        "mechanism_spec_sha256": spec.mechanism_spec_sha256,
    }
    compute = {
        "parameters_inference": None, "inference_flops_per_fit_file": None,
        "training_flops_per_fit_file": None, "parameter_ratio_vs_B0": None,
        "inference_flop_ratio_vs_B0": None, "training_flop_ratio_vs_B0": None,
        "envelope_pass": None,
    }
    seeds = [
        {
            "model_seed": seed, "representation_epochs_completed": 0,
            "scale_epochs_completed": 0, "optimizer_steps": 0,
            "primary_checkpoint_sha256": None, "ema_state_sha256": None,
            "optimizer_scheduler_rng_sha256": None, "fit_bank_sha256": None,
            "calibration_sha256": None, "coefficient_trace_sha256": None,
            "compute": dict(compute),
        }
        for seed in MODEL_SEEDS
    ]
    score_branch = {
        "calibration_quantile": 0.95, "thresholds_by_seed": [None, None, None],
        "calibration_role_sha256": _hash("cal"), "score_provenance_sha256": _hash("score"),
        "independent": True,
    }
    if arm_id == "N4-CRN":
        score_pop = {**score_branch, "S_pop_bitwise_equal_to_B0": True}
    else:
        score_pop = dict(score_branch)
    metric_branch = {
        "per_seed": [
            {"model_seed": seed, "per_history": [_history_metric(history_id) for history_id in history_ids]}
            for seed in MODEL_SEEDS
        ],
        "aggregate": {
            "P_macro": None, "W_macro": None, "PW_macro": None, "PW_lcb95": None,
            "delta_P_macro": None, "delta_W_macro": None, "delta_PW_macro": None,
            "directional_count": None, "positive_delta_history_count": None,
            "background_uncertainty": [None, None],
        },
    }
    gate_branch = {
        "recovery_applied": False, "minimum_effect": None,
        "positive_history_replication": None, "lcb": None, "directional": None,
        "P_noninferiority": None, "W_noninferiority": None,
        "P_recall_lead": None, "W_recall_lead": None, "FAR": None,
        "background_stability": None, "support_and_provenance": None,
        "compute": None, "score_separation": None, "recovery": None,
    }
    return {
        "schema_id": RESULT_SCHEMA_ID, "protocol_id": PROTOCOL_ID,
        "phase": phase, "arm_id": arm_id, "status": "FAILED",
        "failure_reasons": ["schema test fixture is deliberately not an experiment result"],
        "provenance": provenance, "configuration": config, "seed_runs": seeds,
        "support": {
            "common_support_sha256": _hash("support"), "evaluation_histories": history_ids,
            "per_history": [
                {"history_id": history_id, "p": 0, "w": 0, "a": 0, "controls": 0, "robot_days": 0.0}
                for history_id in history_ids
            ],
            "excluded_rows": 0,
        },
        "scores": {"S_pred": dict(score_branch), "S_pop": score_pop},
        "metrics": {"S_pred": metric_branch, "S_pop": metric_branch},
        "branch_gates": {"S_pred": dict(gate_branch), "S_pop": dict(gate_branch)},
    }


def test_closed_v2_result_schema_accepts_failed_matrix_row_and_rejects_drift() -> None:
    result = _failed_result("N2-MSP")
    assert validate_sprint18_result(result) is result
    assert result["schema_id"] == RESULT_SCHEMA_ID
    assert result["protocol_id"] == PROTOCOL_ID
    extra = {**result, "unexpected": True}
    with pytest.raises(ValueError, match="closed v2 schema keys"):
        validate_sprint18_result(extra)
    wrong_schema = {**result, "protocol_id": "sprint18-full-training-protocol-v1"}
    with pytest.raises(ValueError, match="exactly Sprint18 v2"):
        validate_sprint18_result(wrong_schema)
    wrong_members = {**result, "arm_id": "N1-HCC"}
    with pytest.raises(ValueError, match="mechanism-spec provenance"):
        validate_sprint18_result(wrong_members)

    incomplete_valid = {**result, "status": "VALID", "failure_reasons": []}
    with pytest.raises(ValueError, match="complete frozen primary checkpoint"):
        validate_sprint18_result(incomplete_valid)

    metrics = dict(result["metrics"])
    pred_metrics = dict(metrics["S_pred"])
    per_seed = list(pred_metrics["per_seed"])
    first_seed = dict(per_seed[0])
    history_metrics = list(first_seed["per_history"])
    history_metrics[0], history_metrics[1] = history_metrics[1], history_metrics[0]
    first_seed["per_history"] = history_metrics
    per_seed[0] = first_seed
    pred_metrics["per_seed"] = per_seed
    metrics["S_pred"] = pred_metrics
    with pytest.raises(ValueError, match="canonical four-history order"):
        validate_sprint18_result({**result, "metrics": metrics})


def test_n4_result_requires_exact_s_pop_parity_flag() -> None:
    result = _failed_result("N4-CRN")
    assert validate_sprint18_result(result) is result
    scores = dict(result["scores"])
    scores["S_pop"] = {key: value for key, value in scores["S_pop"].items() if key != "S_pop_bitwise_equal_to_B0"}
    invalid = {**result, "scores": scores}
    with pytest.raises(ValueError, match="bitwise S_pop parity"):
        validate_sprint18_result(invalid)


def test_parameter_and_flop_accounting_deduplicates_parameters_and_fails_closed() -> None:
    layer = nn.Linear(2, 3)
    assert count_unique_parameters([layer, layer]) == sum(parameter.numel() for parameter in layer.parameters())
    count = profile_inference_flops(lambda: layer(torch.ones(1, 2)), device="cpu")
    assert count > 0
    baseline = {"S_pred": 100, "S_pop": 80}
    flops = {"S_pred": 1000, "S_pop": 500}
    with pytest.raises(ValueError, match="requires actual paired B0"):
        validate_compute_envelope(
            "N2-MSP", parameters_by_branch=baseline, flops_by_branch=flops,
            b0_parameters_by_branch=None, b0_flops_by_branch=None, training_flops_ratio=1.0,
        )
    result = validate_compute_envelope(
        "N2-MSP", parameters_by_branch={"S_pred": 110, "S_pop": 88},
        flops_by_branch={"S_pred": 1100, "S_pop": 550},
        b0_parameters_by_branch=baseline, b0_flops_by_branch=flops,
        training_flops_ratio=1.25,
    )
    assert result["envelope_pass"] is True

    n4 = validate_compute_envelope(
        "N4-CRN",
        parameters_by_branch={"S_pred": 100_100, "S_pop": 80},
        flops_by_branch={"S_pred": 1_100, "S_pop": 550},
        b0_parameters_by_branch=baseline,
        b0_flops_by_branch=flops,
        training_flops_ratio=0.25,
        n4_scale_parameters=100_000,
        n4_s_pop_bitwise_equal_to_b0=True,
    )
    assert n4["envelope_pass"] is True
    assert n4["combined_inference_flops_ratio_vs_B0"] == pytest.approx(1.10)
    with pytest.raises(ValueError, match="violates"):
        validate_compute_envelope(
            "N4-CRN",
            parameters_by_branch={"S_pred": 100_100, "S_pop": 80},
            flops_by_branch={"S_pred": 1_101, "S_pop": 550},
            b0_parameters_by_branch=baseline,
            b0_flops_by_branch=flops,
            training_flops_ratio=0.25,
            n4_scale_parameters=100_000,
            n4_s_pop_bitwise_equal_to_b0=True,
        )


def _checkpoint_states(binding, config: Sprint18ArmConfig, file_ids: list[str]) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    fit_history = binding.role_histories("FIT")[0]
    all_fit_digests = sorted(item.manifest_sha256 for item in binding.role_histories("FIT"))
    canonical_ids = sorted(file_ids)
    sampler_seed, epoch_order = deterministic_sampler_order(
        binding, config, workload="representation", scope="task6_smoke_only",
        role_history_ids=[fit_history.actual_history_id], epoch=1,
        eligible_source_ids=canonical_ids,
    )
    sampler = {
        "scope": "task6_smoke_only", "workload": "representation",
        "role_history_ids": [fit_history.actual_history_id],
        "role_manifest_sha256": [fit_history.manifest_sha256],
        "eligible_source_ids": canonical_ids,
        "eligible_source_ids_sha256": hashlib.sha256(
            __import__("json").dumps(canonical_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "sampler_epoch_seed": sampler_seed,
        "epoch_order_ids": list(epoch_order),
        "epoch_order_ids_sha256": hashlib.sha256(
            __import__("json").dumps(list(epoch_order), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "eligible_rows": len(canonical_ids), "batch_size": len(canonical_ids) // 2,
        "batches_per_epoch": 2, "completed_epochs": 0, "current_epoch": 1,
        "batches_in_current_epoch": 0,
        "fit_role_sha256": hashlib.sha256(
            __import__("json").dumps(all_fit_digests, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    lr = learning_rate_for_epoch(1)
    scheduler = {
        "scheduler_id": "sprint18-epoch-lr-v2", "completed_epochs": 0,
        "active_epoch": 1, "learning_rate": lr, "global_step": 0,
    }
    coefficient = {
        "epoch": 1, "global_step": 0, "selection_state": "task6_smoke_only",
        "lambda_info": 0.0, "lambda_hcc": None, "alpha": None, "beta": None,
    }
    return sampler, scheduler, coefficient


def test_checkpoint_round_trip_binds_optimizer_scheduler_ema_step_rng_and_support(tmp_path) -> None:
    binding = load_task6_binding(HANDOFF)
    config = _smoke_config()
    model_config = V1Config(
        n_robots=9, n_programs=8, seed=MODEL_SEEDS[0],
        contrastive_weight_max=0.025, contrastive_warmup_steps=430,
        contrastive_ramp_steps=860,
    )
    model = V1RepresentationModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0)
    fit_history = binding.role_histories("FIT")[0]
    sampler, scheduler, coefficient = _checkpoint_states(binding, config, ["S-op-000000-a", "S-op-000001-b"])
    initialization_sha = state_sha256(model.state_dict())
    path = tmp_path / "task6-smoke.pt"
    digest = save_sprint18_checkpoint(
        path, model, optimizer, arm=config, binding=binding,
        global_step=0, completed_epoch=0, initialization_sha256=initialization_sha,
        scheduler_state=scheduler, sampler_state=sampler,
        coefficient_state=coefficient, scope="task6_smoke_only",
    )
    restored_model = V1RepresentationModel(model_config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0)
    result = restore_sprint18_checkpoint(
        path, restored_model, restored_optimizer, arm=config, binding=binding,
        allowed_scope="task6_smoke_only",
    )
    assert result.checkpoint_sha256 == digest
    assert result.global_step == 0
    assert result.checkpoint_kind == "task6_smoke_fixture"
    assert result.representation_step == 0
    assert result.optimizer_scope == "representation"
    assert result.data_binding_sha256 == binding.executable_binding_sha256
    for name, tensor in model.state_dict().items():
        other = restored_model.state_dict()[name]
        if isinstance(tensor, torch.Tensor):
            assert torch.equal(tensor, other)
        else:
            assert tensor == other
    payload = torch.load(path, map_location="cpu", weights_only=False)
    broken = dict(payload)
    broken.pop("rng_state")
    with pytest.raises(ValueError, match="incomplete or contains unknown"):
        validate_sprint18_checkpoint(
            broken, model_config=model_config, arm=config, binding=binding,
            allowed_scope="task6_smoke_only",
        )
    bad_order = dict(payload)
    bad_sampler = dict(payload["sampler_state"])
    bad_sampler["epoch_order_ids"] = list(reversed(bad_sampler["epoch_order_ids"]))
    bad_order["sampler_state"] = bad_sampler
    with pytest.raises(ValueError, match="deterministic Fit permutation"):
        validate_sprint18_checkpoint(
            bad_order, model_config=model_config, arm=config, binding=binding,
            allowed_scope="task6_smoke_only",
        )
    assert fit_history.actual_history_id == sampler["role_history_ids"][0]


def test_frozen_schedule_values_match_protocol() -> None:
    assert learning_rate_for_epoch(1) == pytest.approx(1.5e-4)
    assert learning_rate_for_epoch(5) == pytest.approx(1.5e-3)
    assert learning_rate_for_epoch(100) == pytest.approx(1e-5)
    assert coefficient_for_epoch(0.05, 1) == 0.0
    assert coefficient_for_epoch(0.05, 10) == 0.0
    assert coefficient_for_epoch(0.05, 11) == pytest.approx(0.0025)
    assert coefficient_for_epoch(0.05, 30) == pytest.approx(0.05)
    assert coefficient_for_epoch(0.05, 31) == 0.05


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task9_balance_records(
    binding: Task6Binding,
    coefficient: float,
    observation_ratios: tuple[float, ...],
) -> list[dict[str, object]]:
    batches_per_epoch = (binding.fit_healthy_rows + BATCH_SIZE - 1) // BATCH_SIZE
    observations = [
        (epoch, step)
        for epoch in range(26, TASK9_COEFFICIENT_PILOT_EPOCHS + 1)
        for step in range(
            (epoch - 1) * batches_per_epoch + 1,
            epoch * batches_per_epoch + 1,
        )
        if step % 40 == 0
    ]
    assert len(observation_ratios) == len(observations)
    observation_ratio_by_step = dict(zip(observations, observation_ratios))
    records: list[dict[str, object]] = []
    for epoch in range(1, TASK9_COEFFICIENT_PILOT_EPOCHS + 1):
        first_step = (epoch - 1) * batches_per_epoch + 1
        steps = {first_step}
        steps.update(step for recorded_epoch, step in observations if recorded_epoch == epoch)
        for step in sorted(steps):
            observation = (epoch, step) in observation_ratio_by_step
            ratio = observation_ratio_by_step.get((epoch, step), 0.2)
            effective_coefficient = coefficient_for_epoch(coefficient, epoch)
            records.append(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "lr": learning_rate_for_epoch(epoch),
                    "raw_L_pred": 1.0,
                    "effective_coefficient_L_pred": 1.0,
                    "weighted_L_pred": 1.0,
                    "raw_L_info": 2.0,
                    "effective_coefficient_L_info": effective_coefficient,
                    "weighted_L_info": 2.0 * effective_coefficient,
                    "gradient_norm_L_pred_shared": 1.0,
                    "gradient_norm_weighted_L_info_shared": ratio,
                    "r_aux": ratio,
                    "embedding_support_files": BATCH_SIZE,
                    "support_valid_files": BATCH_SIZE,
                    "support_valid_patches": BATCH_SIZE * 12,
                    "support_prediction_files": BATCH_SIZE,
                    "support_prediction_patches": BATCH_SIZE * 4,
                    "support_info_files": BATCH_SIZE,
                    "contrastive_input_files": BATCH_SIZE,
                    "embedding_variance_mean": 0.1,
                    "embedding_std_median": 0.1,
                    "embedding_effective_rank": 3.0,
                    "protocol_gradient_observation": observation,
                }
            )
    return records


def _task9_design_history(
    actual_history_id: str,
    *,
    loss: float,
    spread: float = 0.05,
) -> dict[str, object]:
    return {
        "support_sha256": _hash(f"Task9-healthy-support-{actual_history_id}"),
        "support_files": 13,
        "healthy_embedding_std_median": spread,
        "mean_healthy_fine_prediction_loss": loss,
    }


def _task9_prediction_only_runs(
    binding: Task6Binding, *, loss: float = 1.0
) -> dict[int, dict[str, object]]:
    design_ids = tuple(item.actual_history_id for item in binding.role_histories("DESIGN"))
    return {
        seed: {
            "design_histories": {
                history_id: _task9_design_history(history_id, loss=loss)
                for history_id in design_ids
            }
        }
        for seed in MODEL_SEEDS
    }


def _task9_candidate_seed_run(
    binding: Task6Binding,
    coefficient: float,
    ratios: tuple[float, ...],
    *,
    loss: float = 1.0,
    spread: float = 0.05,
) -> dict[str, object]:
    design_ids = tuple(item.actual_history_id for item in binding.role_histories("DESIGN"))
    return {
        "balance_records": _task9_balance_records(binding, coefficient, ratios),
        "design_histories": {
            history_id: _task9_design_history(
                history_id, loss=loss, spread=spread
            )
            for history_id in design_ids
        },
    }


def test_task9_pilot_state_sampler_and_full_ramp_checkpoint_are_nonfinal() -> None:
    binding = load_task6_binding(HANDOFF)
    pilot = make_sprint18_arm_config(
        "B0-FULL",
        MODEL_SEEDS[0],
        _coefficients("B0-FULL"),
        selection_state="coefficient_pilot",
    )
    assert validate_arm_config(pilot, require_locked=False) is pilot
    with pytest.raises(ValueError, match="cannot start a locked experiment"):
        validate_arm_config(pilot)
    with pytest.raises(ValueError, match="must not claim a selection lock"):
        make_sprint18_arm_config(
            "B0-FULL",
            MODEL_SEEDS[0],
            _coefficients("B0-FULL"),
            selection_state="coefficient_pilot",
            coefficient_lock_sha256=_hash("not-selected"),
        )
    with pytest.raises(ValueError, match="restricted to B0-FULL"):
        make_sprint18_arm_config(
            "K3-FULL",
            MODEL_SEEDS[0],
            _coefficients("K3-FULL"),
            selection_state="coefficient_pilot",
        )
    prediction_only_reference = make_sprint18_arm_config(
        "B0-FULL",
        MODEL_SEEDS[0],
        {
            "lambda_info_max": 0.0,
            "lambda_hcc_max": None,
            "alpha": None,
            "beta": None,
        },
        selection_state="coefficient_pilot",
    )
    assert prediction_only_reference.lambda_info_max == 0.0
    with pytest.raises(ValueError, match="must not be boolean"):
        make_sprint18_arm_config(
            "B0-FULL",
            MODEL_SEEDS[0],
            {**_coefficients("B0-FULL"), "lambda_info_max": False},
            selection_state="coefficient_pilot",
        )
    with pytest.raises(ValueError, match="outside the frozen grid"):
        make_sprint18_arm_config(
            "B0-FULL",
            MODEL_SEEDS[0],
            {
                "lambda_info_max": 0.0,
                "lambda_hcc_max": None,
                "alpha": None,
                "beta": None,
            },
        )

    fit_histories = binding.role_histories("FIT")
    fit_history_ids = tuple(item.actual_history_id for item in fit_histories)
    fit_ids = tuple(
        f"fit-healthy-{index:05d}" for index in range(binding.fit_healthy_rows)
    )
    batches_per_epoch = (binding.fit_healthy_rows + BATCH_SIZE - 1) // BATCH_SIZE
    pilot_seed, pilot_order = deterministic_sampler_order(
        binding,
        pilot,
        workload="representation",
        scope="coefficient_pilot",
        role_history_ids=fit_history_ids,
        epoch=TASK9_COEFFICIENT_PILOT_EPOCHS + 1,
        eligible_source_ids=fit_ids,
    )
    final_config = make_sprint18_arm_config(
        "B0-FULL",
        MODEL_SEEDS[0],
        _coefficients("B0-FULL"),
        coefficient_lock_sha256=_hash("coefficient-lock"),
    )
    final_seed, final_order = deterministic_sampler_order(
        binding,
        final_config,
        workload="representation",
        scope="experiment",
        role_history_ids=fit_history_ids,
        epoch=TASK9_COEFFICIENT_PILOT_EPOCHS + 1,
        eligible_source_ids=fit_ids,
    )
    assert (pilot_seed, pilot_order) == (final_seed, final_order)
    matched_pilots = [prediction_only_reference] + [
        make_sprint18_arm_config(
            "B0-FULL",
            MODEL_SEEDS[0],
            {
                "lambda_info_max": coefficient,
                "lambda_hcc_max": None,
                "alpha": None,
                "beta": None,
            },
            selection_state="coefficient_pilot",
        )
        for coefficient in COMMON_INFO_COEFFICIENT_GRID
    ]
    for matched_pilot in matched_pilots:
        matched_seed, matched_order = deterministic_sampler_order(
            binding,
            matched_pilot,
            workload="representation",
            scope="coefficient_pilot",
            role_history_ids=fit_history_ids,
            epoch=TASK9_COEFFICIENT_PILOT_EPOCHS + 1,
            eligible_source_ids=fit_ids,
        )
        assert (matched_seed, matched_order) == (pilot_seed, pilot_order)

    role_digests = [item.manifest_sha256 for item in fit_histories]
    epoch_seed, epoch_order = deterministic_sampler_order(
        binding,
        pilot,
        workload="representation",
        scope="coefficient_pilot",
        role_history_ids=fit_history_ids,
        epoch=TASK9_COEFFICIENT_PILOT_EPOCHS + 1,
        eligible_source_ids=fit_ids,
    )
    sampler_state = {
        "scope": "coefficient_pilot",
        "workload": "representation",
        "role_history_ids": list(fit_history_ids),
        "role_manifest_sha256": role_digests,
        "eligible_source_ids": list(sorted(fit_ids)),
        "eligible_source_ids_sha256": _canonical_digest(list(sorted(fit_ids))),
        "sampler_epoch_seed": epoch_seed,
        "epoch_order_ids": list(epoch_order),
        "epoch_order_ids_sha256": _canonical_digest(list(epoch_order)),
        "eligible_rows": binding.fit_healthy_rows,
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": batches_per_epoch,
        "completed_epochs": TASK9_COEFFICIENT_PILOT_EPOCHS,
        "current_epoch": TASK9_COEFFICIENT_PILOT_EPOCHS + 1,
        "batches_in_current_epoch": 0,
        "fit_role_sha256": _canonical_digest(
            sorted(item.manifest_sha256 for item in fit_histories)
        ),
    }
    global_step = TASK9_COEFFICIENT_PILOT_EPOCHS * batches_per_epoch
    assert _validate_sampler_state(
        sampler_state,
        global_step=global_step,
        scope="coefficient_pilot",
        completed_epoch=TASK9_COEFFICIENT_PILOT_EPOCHS,
        arm=pilot,
        binding=binding,
    )["current_epoch"] == TASK9_COEFFICIENT_PILOT_EPOCHS + 1
    late_epoch_sampler = dict(sampler_state)
    late_epoch_sampler["batches_in_current_epoch"] = 1
    with pytest.raises(ValueError, match="cannot contain updates beyond epoch 30"):
        _validate_sampler_state(
            late_epoch_sampler,
            global_step=global_step + 1,
            scope="coefficient_pilot",
            completed_epoch=TASK9_COEFFICIENT_PILOT_EPOCHS,
            arm=pilot,
            binding=binding,
        )
    coefficient_state = {
        "epoch": TASK9_COEFFICIENT_PILOT_EPOCHS + 1,
        "global_step": global_step,
        "selection_state": "coefficient_pilot",
        "lambda_info": pilot.lambda_info_max,
        "lambda_hcc": None,
        "alpha": None,
        "beta": None,
    }
    assert _validate_coefficient_state(
        coefficient_state,
        arm=pilot,
        global_step=global_step,
        current_epoch=TASK9_COEFFICIENT_PILOT_EPOCHS + 1,
        scope="coefficient_pilot",
    )["lambda_info"] == pilot.lambda_info_max
    with pytest.raises(ValueError, match="selection state"):
        _validate_coefficient_state(
            {**coefficient_state, "selection_state": "locked"},
            arm=pilot,
            global_step=global_step,
            current_epoch=TASK9_COEFFICIENT_PILOT_EPOCHS + 1,
            scope="coefficient_pilot",
        )
    incomplete_sampler = dict(sampler_state)
    incomplete_sampler["eligible_source_ids"] = list(sorted(fit_ids[:-1]))
    incomplete_sampler["eligible_source_ids_sha256"] = _canonical_digest(
        list(sorted(fit_ids[:-1]))
    )
    with pytest.raises(ValueError, match="complete Fit-healthy support"):
        _validate_sampler_state(
            incomplete_sampler,
            global_step=global_step,
            scope="coefficient_pilot",
            completed_epoch=TASK9_COEFFICIENT_PILOT_EPOCHS,
            arm=pilot,
            binding=binding,
        )


def test_task9_healthy_support_loader_enforces_fit_and_design_permissions() -> None:
    binding = load_task6_binding(HANDOFF)
    fit_id = binding.role_histories("FIT")[0].actual_history_id
    design_id = binding.role_histories("DESIGN")[0].actual_history_id
    calibration_id = binding.role_histories("CALIBRATION")[0].actual_history_id
    development_id = binding.role_histories("DEVELOPMENT")[0].actual_history_id
    confirmation_id = binding.role_histories("CONFIRMATION")[0].actual_history_id
    missing_manifest = "unopened-role-manifest.json"

    with pytest.raises(ValueError, match="not authorized"):
        load_certified_healthy_role_support(
            binding, fit_id, missing_manifest, {}, action="select_coefficients"
        )
    with pytest.raises(ValueError, match="not authorized"):
        load_certified_healthy_role_support(
            binding, design_id, missing_manifest, {}, action="train_fit"
        )
    for forbidden_history in (calibration_id, development_id, confirmation_id):
        with pytest.raises(ValueError, match="not authorized"):
            load_certified_healthy_role_support(
                binding,
                forbidden_history,
                missing_manifest,
                {},
                action="select_coefficients",
            )


def test_task9_selection_uses_smallest_valid_candidate_and_inclusive_gates() -> None:
    binding = load_task6_binding(HANDOFF)
    prediction_only = _task9_prediction_only_runs(binding)
    ratios = {
        0.025: (0.14, 0.14, 0.14, 0.14, 0.14, 0.14),
        0.05: (0.15, 0.15, 0.15, 0.15, 0.15, 0.15),
        0.10: (0.20, 0.20, 0.20, 0.20, 0.20, 0.20),
    }
    candidate_runs = {
        coefficient: {
            seed: _task9_candidate_seed_run(
                binding,
                coefficient,
                ratios[coefficient],
                loss=1.10 if coefficient == 0.05 else 1.0,
                spread=0.05,
            )
            for seed in MODEL_SEEDS
        }
        for coefficient in COMMON_INFO_COEFFICIENT_GRID
    }
    selection = select_common_info_coefficient(
        binding, candidate_runs, prediction_only
    )
    assert selection["selected_coefficient"] == 0.05
    assert selection["selection_state"] == "selected"
    verdicts = selection["candidate_verdicts"]
    assert isinstance(verdicts, list)
    assert [row["coefficient"] for row in verdicts] == list(
        COMMON_INFO_COEFFICIENT_GRID
    )
    assert verdicts[0]["valid"] is False
    assert verdicts[1]["valid"] is True
    assert verdicts[2]["valid"] is True


def test_task9_selection_fails_closed_on_missing_or_invalid_candidates() -> None:
    binding = load_task6_binding(HANDOFF)
    prediction_only = _task9_prediction_only_runs(binding)
    selection = select_common_info_coefficient(binding, {}, prediction_only)
    assert selection["selected_coefficient"] is None
    assert selection["selection_state"] == "no_valid_candidate"
    with pytest.raises(ValueError, match="outside the frozen common InfoNCE grid"):
        select_common_info_coefficient(binding, {0.0: {}}, prediction_only)

    high_p90 = (0.2, 0.2, 0.2, 0.2, 1.1, 1.2)
    candidate_runs = {
        0.025: {
            seed: _task9_candidate_seed_run(binding, 0.025, high_p90)
            for seed in MODEL_SEEDS
        },
        0.05: {
            seed: _task9_candidate_seed_run(
                binding, 0.05, (0.2,) * 6, spread=0.049
            )
            for seed in MODEL_SEEDS
        },
        0.10: {
            seed: _task9_candidate_seed_run(
                binding,
                0.10,
                (0.2,) * 6,
                loss=1.1000000000000003,
            )
            for seed in MODEL_SEEDS
        },
    }
    selection = select_common_info_coefficient(
        binding, candidate_runs, prediction_only
    )
    assert selection["selected_coefficient"] is None
    per_seed_failures = [
        {
            check
            for seed_result in candidate["per_seed"]
            for check in seed_result["failed_checks"]
        }
        for candidate in selection["candidate_verdicts"]
    ]
    assert "p90_gradient_ratio_exceeds_one" in per_seed_failures[0]
    assert any(check.endswith("_embedding_collapse") for check in per_seed_failures[1])
    assert any(
        check.endswith("_healthy_prediction_retention_failed")
        for check in per_seed_failures[2]
    )

def test_task9_selector_rejects_bad_gradient_observations() -> None:
    binding = load_task6_binding(HANDOFF)
    prediction_only = _task9_prediction_only_runs(binding)
    candidate_runs = {
        coefficient: {
            seed: _task9_candidate_seed_run(binding, coefficient, (0.2,) * 6)
            for seed in MODEL_SEEDS
        }
        for coefficient in COMMON_INFO_COEFFICIENT_GRID
    }

    inconsistent_records = candidate_runs[0.025][MODEL_SEEDS[0]]["balance_records"]
    assert isinstance(inconsistent_records, list)
    inconsistent_observation = next(
        row
        for row in inconsistent_records
        if isinstance(row, dict) and row["protocol_gradient_observation"]
    )
    inconsistent_observation["r_aux"] = 0.3

    duplicate_records = candidate_runs[0.05][MODEL_SEEDS[0]]["balance_records"]
    assert isinstance(duplicate_records, list)
    duplicate_observation = next(
        row
        for row in duplicate_records
        if isinstance(row, dict) and row["protocol_gradient_observation"]
    )
    duplicate_records.append(dict(duplicate_observation))
    duplicate_records[0]["lr"] = 0.001
    invalid_support_records = candidate_runs[0.10][MODEL_SEEDS[0]]["balance_records"]
    assert isinstance(invalid_support_records, list)
    invalid_support_records[0]["support_valid_files"] = BATCH_SIZE - 1

    selection = select_common_info_coefficient(
        binding, candidate_runs, prediction_only
    )
    verdicts = selection["candidate_verdicts"]
    assert isinstance(verdicts, list)
    assert selection["selected_coefficient"] is None
    first_failures = {
        check for seed in verdicts[0]["per_seed"] for check in seed["failed_checks"]
    }
    second_failures = {
        check for seed in verdicts[1]["per_seed"] for check in seed["failed_checks"]
    }
    third_failures = {
        check for seed in verdicts[2]["per_seed"] for check in seed["failed_checks"]
    }
    assert "full_batch_file_support_mismatch" in third_failures
    assert "gradient_ratio_inconsistent" in first_failures
    assert "protocol_gradient_observations_duplicated" in second_failures
    assert "learning_rate_schedule_mismatch" in second_failures
