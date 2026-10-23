"""Focused consumer-visible tests for the N4-CRN scale-model capacity path.

Proves, on tiny REAL fixtures through the public module plus real encoder
caches (no mocks): finite Gaussian NLL, context-only construction (no target /
residual / label input surface), the fixed Linear-32-GELU-Linear-1 capacity with
frozen bounds, quantile-bound formulas with edge/saturation handling, held-out
fold isolation, exact 50-epoch / ceil(N/512) optimizer accounting, frozen
representation/optimizer invariance, and strict checkpoint save/restore of the
scale module through the committed N4 checkpoint path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from representation import (
    N4_GUARD_CONTRACT,
    N4_SCALE_BATCH_SIZE,
    N4_SCALE_CHECKPOINT_SCHEMA,
    N4_SCALE_EPOCHS,
    N4_SCALE_FIT_RECORDS,
    N4_SCALE_HISTORY_RECORDS,
    N4_SCALE_PARAMETERIZATION,
    N4ScaleGuardError,
    N4ScaleRecord,
    V1Config,
    V1RepresentationModel,
    build_n4_scale_records,
    fit_n4_scale_scope,
    init_n4_scale_model,
    load_n4_scale_checkpoint,
    make_sprint18_arm_config,
    n4_anchor_scale_model,
    n4_canonical_fold_splits,
    n4_guard_first_step_gradients,
    n4_guard_initialized_interiority,
    n4_patch_residual,
    n4_residual_bounds,
    n4_saturation,
    n4_scale_nll,
    n4_scale_scope_steps,
    n4_spearman,
    save_n4_scale_checkpoint,
    validate_n4_capacity,
)
from representation.crn import ContextResidualScaleModel
from representation.data import collate_variable_files
from representation.sprint18_checkpoint import (
    _canonical_sha256,
    restore_sprint18_checkpoint,
    save_sprint18_checkpoint,
    state_sha256,
)
from representation.sprint18_harness import MODEL_SEEDS, load_task6_binding
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import FileSample, SampleLabel

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "experiments" / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"


def _tiny_config(seed: int = 181801) -> V1Config:
    return V1Config(
        n_channels=6, patch_size=32, stride=16, n_robots=9, n_programs=8,
        seed=seed, contrastive_weight_max=0.05, ema_decay=0.996, dropout=0.0,
    )


def _tiny_samples(count: int = 6, prefix: str = "n4-behavior", seed: int = 181801) -> list[FileSample]:
    rng = np.random.default_rng(91813)
    samples: list[FileSample] = []
    for index in range(count):
        length = 640 + 64 * index
        signal = rng.normal(size=(6, length)).astype(np.float32)
        sample = FileSample(
            x=signal, file_id=f"{prefix}-{index:02d}", file_label=SampleLabel.NORMAL,
            seed=seed + index, generator_version="task13-test-fixture",
            config_hash="task13-test-fixture", regime_sequence=[],
            robot_idx=index % 3, program_idx=index % 2,
        )
        sample.validate()
        samples.append(sample)
    return samples


def _tiny_records(n: int = 40, d: int = 128, seed: int = 181801) -> list[N4ScaleRecord]:
    rng = torch.Generator().manual_seed(seed)
    records: list[N4ScaleRecord] = []
    for index in range(n):
        records.append(
            N4ScaleRecord(
                history_id=f"HIST-{index % 2:02d}",
                file_id=f"file-{index:04d}",
                patch_index=index,
                context=torch.randn(d, generator=rng),
                residual=float(abs(torch.randn((), generator=rng).item()) + 0.05),
            )
        )
    return sorted(records, key=lambda r: (r.history_id, r.file_id, r.patch_index))


def test_n4_scale_capacity_is_fixed_32_hidden_scalar_output() -> None:
    model = ContextResidualScaleModel(8)
    assert model.hidden.out_features == 32
    assert model.out.out_features == 1
    params = sum(p.numel() for p in model.parameters())
    assert 0 < params <= 100_000
    with pytest.raises(ValueError, match="positive integer"):
        ContextResidualScaleModel(0)


def test_n4_scale_forward_clamps_to_frozen_bounds() -> None:
    torch.manual_seed(181801)
    model = ContextResidualScaleModel(8)
    context = torch.randn(12, 8)
    bounds = n4_residual_bounds(np.abs(np.random.default_rng(7).normal(size=200)) + 0.05)
    out = model(context, v_floor=bounds["v_floor"], v_cap=bounds["v_cap"])
    assert torch.isfinite(out["log_v"]).all()
    assert bool((out["v"] >= bounds["v_floor"] - 1e-9).all())
    assert bool((out["v"] <= bounds["v_cap"] + 1e-9).all())
    assert torch.isfinite(n4_scale_nll(torch.rand(12) + 0.05, out["log_v"]))


def test_n4_scale_rejects_target_residual_label_inputs_by_construction() -> None:
    import inspect

    params = inspect.signature(ContextResidualScaleModel.forward).parameters
    assert set(params) == {"self", "context", "v_floor", "v_cap"}
    model = ContextResidualScaleModel(4)
    with pytest.raises(ValueError, match="2D"):
        model(torch.randn(5, 4, 2), v_floor=0.01, v_cap=1.0)
    with pytest.raises(ValueError, match="positive"):
        model(torch.randn(5, 4), v_floor=-0.01, v_cap=1.0)


def test_n4_bounds_match_linear_quantiles_with_edge_floor() -> None:
    rng = np.random.default_rng(11)
    residuals = np.abs(rng.normal(size=500)) + 0.02
    bounds = n4_residual_bounds(residuals)
    assert bounds["v_floor"] == pytest.approx(max(float(np.quantile(residuals, 0.05, method="linear")), 1e-6))
    assert bounds["v_cap"] == pytest.approx(max(float(np.quantile(residuals, 0.95, method="linear")), 10.0 * bounds["v_floor"]))
    tiny = np.full(50, 1e-9)
    edge = n4_residual_bounds(tiny)
    assert edge["v_floor"] == pytest.approx(1e-6)
    assert edge["v_cap"] == pytest.approx(10e-6)
    sat = n4_saturation(torch.full((100,), math.log(edge["v_cap"])), v_floor=edge["v_floor"], v_cap=edge["v_cap"])
    assert sat["frac_at_cap"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-empty"):
        n4_residual_bounds([])


def test_n4_patch_residual_matches_normalized_mse_and_detaches() -> None:
    torch.manual_seed(0)
    target = torch.randn(6, 8, requires_grad=True)
    predicted = torch.randn(6, 8, requires_grad=True)
    residuals = n4_patch_residual(target, predicted)
    assert not residuals.requires_grad
    expected = (
        (torch.nn.functional.layer_norm(target.detach(), (8,))
         - torch.nn.functional.layer_norm(predicted.detach(), (8,))).square().mean(dim=-1)
    )
    assert torch.allclose(residuals, expected)
    assert bool((residuals >= 0.0).all())
    rho = n4_spearman(np.arange(10, dtype=float), np.arange(10, dtype=float))
    assert rho == pytest.approx(1.0)
    assert n4_spearman(np.ones(5), np.arange(5, dtype=float)) == pytest.approx(0.0)


def test_n4_records_come_from_real_encoder_caches_with_zero_rep_updates() -> None:
    config = _tiny_config()
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    before = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
    samples = _tiny_samples()
    records = build_n4_scale_records(
        model=model, samples=samples, history_id="HIST-LOCAL",
        patchifier=patchifier, masking_seed=91813,
    )
    assert records, "expected eligible masked records from the real encoder"
    assert all(r.history_id == "HIST-LOCAL" for r in records)
    assert len({(r.file_id, r.patch_index) for r in records}) == len(records)
    assert all(math.isfinite(r.residual) and r.residual >= 0.0 for r in records)
    after = model.state_dict()
    assert set(after) == set(before)
    for key in before:
        left, right = after[key], before[key]
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            assert torch.equal(left.detach().cpu(), right)
        else:
            assert left == right
    batch = collate_variable_files(samples, patchifier, masking_config=config, masking_seed=91813)
    clean = {k: v for k, v in batch.items() if k != "file_samples"}
    with torch.no_grad():
        out = model(clean)
    direct = n4_patch_residual(out["target_latents"][0], out["predicted_latents"][0])
    assert torch.isfinite(direct).all()


def test_n4_fold_scope_fit_proves_finite_nll_bounds_saturation_and_step_counts() -> None:
    binding = load_task6_binding(HANDOFF)
    arm = make_sprint18_arm_config(
        "N4-CRN", MODEL_SEEDS[0],
        {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="locked", coefficient_lock_sha256="ab" * 32,
    )
    splits = n4_canonical_fold_splits(binding)
    assert len(splits) == 3
    fit_ids = tuple(item.actual_history_id for item in binding.role_histories("FIT"))
    assert [held for _, held in splits] == list(fit_ids)
    torch.manual_seed(MODEL_SEEDS[0])
    scale = init_n4_scale_model(128, MODEL_SEEDS[0])
    before_params = [p.detach().cpu().clone() for p in scale.parameters()]
    raw_train = _tiny_records(40)
    heldout_id = fit_ids[0]
    train_histories = sorted(h for h in fit_ids if h != heldout_id)
    train = [
        N4ScaleRecord(history_id=train_histories[i % 2], file_id=r.file_id, patch_index=r.patch_index, context=r.context, residual=r.residual)
        for i, r in enumerate(raw_train)
    ]
    heldout = [
        N4ScaleRecord(history_id=heldout_id, file_id=r.file_id, patch_index=r.patch_index, context=r.context, residual=r.residual)
        for r in _tiny_records(17, seed=77)
    ]
    residuals = np.array([r.residual for r in train])
    bounds = n4_residual_bounds(residuals)
    result = fit_n4_scale_scope(
        records=train, training_history_ids=train_histories,
        scope=f"fold:{heldout_id}", arm=arm, binding=binding, scale_model=scale,
        v_floor=bounds["v_floor"], v_cap=bounds["v_cap"], heldout_records=heldout,
    )
    assert result["record_count"] == 40
    assert result["per_epoch_steps"] == math.ceil(40 / N4_SCALE_BATCH_SIZE) == 1
    assert result["optimizer_steps"] == N4_SCALE_EPOCHS * 1
    assert len(result["per_epoch_nll"]) == N4_SCALE_EPOCHS
    assert math.isfinite(result["final_train_nll"]) and math.isfinite(result["heldout_nll"])
    assert result["heldout_spearman"] is not None and -1.0 <= float(result["heldout_spearman"]) <= 1.0
    assert 0.0 <= result["saturation_heldout"]["frac_at_cap"] <= 1.0
    changed = any(not torch.equal(a, b.detach().cpu()) for a, b in zip(before_params, scale.parameters()))
    assert changed, "scale-only optimizer must update the scale head"
    verdict = validate_n4_capacity(
        [result, result, result], record_counts=[40, 40, 40],
        per_seed_updates=3 * N4_SCALE_EPOCHS * 1,
    )
    assert verdict["verdict"] == "PASS", verdict["reasons"]
    bad = dict(result)
    bad["heldout_nll"] = float("nan")
    assert validate_n4_capacity([bad, result, result], record_counts=[40, 40, 40], per_seed_updates=150)["verdict"] == "FAILED"


def test_n4_certified_scope_constants_and_step_math() -> None:
    """The N4 in-scope record contract is the certified masked Fit support."""
    assert dict(N4_SCALE_HISTORY_RECORDS) == {
        "S18I-ITER-0005-FIT-01": 22328,
        "S18I-ITER-0005-FIT-02": 23098,
        "S18I-ITER-0005-FIT-03": 21563,
    }
    assert N4_SCALE_FIT_RECORDS == 66989
    fold_counts = [
        N4_SCALE_FIT_RECORDS - count for count in N4_SCALE_HISTORY_RECORDS.values()
    ]
    assert fold_counts == [44661, 43891, 45426]
    assert [n4_scale_scope_steps(count) for count in fold_counts] == [4400, 4300, 4450]
    assert n4_scale_scope_steps(N4_SCALE_FIT_RECORDS) == 6550
    assert sum(fold_counts) == 2 * N4_SCALE_FIT_RECORDS
    assert N4_SCALE_FIT_RECORDS != 167419  # every valid Fit patch is a different scope


def test_n4_fit_rejects_foreign_or_retuned_optimizer() -> None:
    binding = load_task6_binding(HANDOFF)
    arm = make_sprint18_arm_config(
        "N4-CRN", MODEL_SEEDS[0],
        {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="locked", coefficient_lock_sha256="ab" * 32,
    )
    scale = init_n4_scale_model(128, MODEL_SEEDS[0])
    splits = n4_canonical_fold_splits(binding)
    training_ids, heldout_id = splits[0]
    raw = _tiny_records(40)
    train = [
        N4ScaleRecord(
            history_id=training_ids[i % 2], file_id=record.file_id,
            patch_index=record.patch_index, context=record.context, residual=record.residual,
        )
        for i, record in enumerate(raw)
    ]
    bounds = n4_residual_bounds(np.array([record.residual for record in train]))
    foreign = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1e-3, weight_decay=1e-4)
    with pytest.raises(ValueError, match="bound to exactly the scale-model parameters"):
        fit_n4_scale_scope(
            records=train, training_history_ids=list(training_ids),
            scope=f"fold:{heldout_id}", arm=arm, binding=binding, scale_model=scale,
            v_floor=bounds["v_floor"], v_cap=bounds["v_cap"], optimizer=foreign,
        )
    retuned_scale = init_n4_scale_model(128, MODEL_SEEDS[0])
    retuned = torch.optim.AdamW(retuned_scale.parameters(), lr=2e-3, weight_decay=1e-4)
    with pytest.raises(ValueError, match="frozen lr/weight decay"):
        fit_n4_scale_scope(
            records=train, training_history_ids=list(training_ids),
            scope=f"fold:{heldout_id}", arm=arm, binding=binding, scale_model=retuned_scale,
            v_floor=bounds["v_floor"], v_cap=bounds["v_cap"], optimizer=retuned,
        )


def test_n4_fold_rejects_heldout_contamination_and_wrong_counts() -> None:
    binding = load_task6_binding(HANDOFF)
    arm = make_sprint18_arm_config(
        "N4-CRN", MODEL_SEEDS[0],
        {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="locked", coefficient_lock_sha256="ab" * 32,
    )
    scale = init_n4_scale_model(128, MODEL_SEEDS[0])
    train = _tiny_records(10)
    outsider = N4ScaleRecord(history_id="HIST-99", file_id="file-0000", patch_index=0, context=train[0].context.clone(), residual=0.5)
    with pytest.raises(ValueError, match="outside its in-fold histories"):
        fit_n4_scale_scope(
            records=[*train, outsider], training_history_ids=["HIST-00", "HIST-01"],
            scope="fold:HIST-99", arm=arm, binding=binding, scale_model=scale,
            v_floor=0.01, v_cap=1.0,
        )
    with pytest.raises(ValueError, match="non-empty and unique"):
        fit_n4_scale_scope(
            records=train, training_history_ids=["HIST-00", "HIST-00"],
            scope="fold:HIST-99", arm=arm, binding=binding, scale_model=scale,
            v_floor=0.01, v_cap=1.0,
        )


def test_n4_scale_checkpoint_save_restore_keeps_rep_frozen(tmp_path) -> None:
    binding = load_task6_binding(HANDOFF)
    arm = make_sprint18_arm_config(
        "N4-CRN", MODEL_SEEDS[0],
        {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="locked", coefficient_lock_sha256="ab" * 32,
    )
    config = _tiny_config(seed=MODEL_SEEDS[0])
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    scale = init_n4_scale_model(config.d_model, MODEL_SEEDS[0])
    optimizer = torch.optim.AdamW(scale.parameters(), lr=1e-3, weight_decay=1e-4)
    _probe_context = torch.randn(4, config.d_model)
    _probe_residual = torch.rand(4) + 0.05
    optimizer.zero_grad(set_to_none=True)
    _probe_loss = n4_scale_nll(_probe_residual, scale(_probe_context, v_floor=0.02, v_cap=1.4)["log_v"])
    _probe_loss.backward()
    probe_gradients = n4_guard_first_step_gradients(scale)
    optimizer.step()
    for param_state in optimizer.state_dict()["state"].values():
        param_state["step"] = torch.tensor(n4_scale_scope_steps(N4_SCALE_FIT_RECORDS))
    initialization_sha256 = state_sha256(model.state_dict())
    model_before = {k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v) for k, v in model.state_dict().items()}
    ema_before = {k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v) for k, v in model.target_encoder.state_dict().items()}
    from representation.sprint18_harness import deterministic_sampler_order
    per_patch = N4_SCALE_FIT_RECORDS
    fit_ids = tuple(f"n4-patch-{i:07d}" for i in range(per_patch))
    fit_history_ids = tuple(item.actual_history_id for item in binding.role_histories("FIT"))
    sampler_seed, epoch_order = deterministic_sampler_order(
        binding, arm, workload="n4_scale", scope="experiment",
        role_history_ids=fit_history_ids, epoch=50, eligible_source_ids=fit_ids,
    )
    ordered_ids = sorted(fit_ids)
    fit_role = _canonical_sha256(sorted(item.manifest_sha256 for item in binding.role_histories("FIT")))
    from math import ceil

    batches_per_epoch = ceil(len(ordered_ids) / 512)
    sampler_state = {
        "scope": "experiment",
        "workload": "n4_scale",
        "role_history_ids": list(fit_history_ids),
        "role_manifest_sha256": [binding.history(h).manifest_sha256 for h in fit_history_ids],
        "eligible_source_ids": ordered_ids,
        "eligible_source_ids_sha256": _canonical_sha256(ordered_ids),
        "sampler_epoch_seed": sampler_seed,
        "epoch_order_ids": list(epoch_order),
        "epoch_order_ids_sha256": _canonical_sha256(list(epoch_order)),
        "eligible_rows": len(ordered_ids),
        "batch_size": 512,
        "batches_per_epoch": batches_per_epoch,
        "completed_epochs": 50,
        "current_epoch": 50,
        "batches_in_current_epoch": 0,
        "fit_role_sha256": fit_role,
    }
    coefficient_state = {
        "epoch": 50, "global_step": 50 * batches_per_epoch, "selection_state": "locked",
        "lambda_info": None, "lambda_hcc": None, "alpha": None, "beta": None,
    }
    fold_counts = [
        per_patch - N4_SCALE_HISTORY_RECORDS[history_id] for history_id in fit_history_ids
    ]
    log_floor = math.log(0.02)
    log_cap = math.log(1.4)
    log_v_min = -3.2
    log_v_max = -0.5
    coherent_initialization = {
        "init_anchor": {
            "v_ref": 0.02, "log_v_ref": math.log(0.02), "record_count": per_patch,
        },
        "init_interiority": {
            "record_count": per_patch, "log_v_min": log_v_min, "log_v_max": log_v_max,
            "log_floor": log_floor, "log_cap": log_cap,
            "floor_margin": log_v_min - log_floor,
            "cap_margin": log_cap - log_v_max,
            "frac_at_floor": 0.0, "frac_at_cap": 0.0, "tolerance": 1e-6,
        },
        "first_step_gradients": probe_gradients,
    }
    arm_state = {
        "paired_b0_checkpoint_sha256": "cd" * 32,
        "paired_b0_representation_step": 100 * ceil(5446 / 128),
        "paired_b0_model_state_sha256": state_sha256(model.state_dict()),
        "paired_b0_ema_state_sha256": state_sha256(model.target_encoder.state_dict()),
        "scale_model_state": scale.state_dict(),
        "scale_model_state_sha256": state_sha256(scale.state_dict()),
        "scale_parameterization": N4_SCALE_PARAMETERIZATION,
        "guard_contract": N4_GUARD_CONTRACT,
        "scale_initialization": coherent_initialization,
        "diagnostic_folds": [
            {
                "heldout_history_id": fit_history_ids[i],
                "training_history_ids": [h for h in fit_history_ids if h != fit_history_ids[i]],
                "record_count": fold_counts[i],
                "optimizer_steps": 50 * ceil(fold_counts[i] / 512),
                "scale_state_sha256": state_sha256(scale.state_dict()),
            }
            for i in range(3)
        ],
        "final_fit_record_count": per_patch,
        "final_scale_steps": 50 * ceil(per_patch / 512),
        "scale_bounds": {
            "q05": 0.02, "q95": 1.4, "v_floor": 0.02, "v_cap": 1.4,
            "fit_role_sha256": fit_role, "residual_records_sha256": "ef" * 32,
        },
    }
    path = tmp_path / "n4-scale-coherent.pt"
    checkpoint_inputs = {
        "arm": arm, "binding": binding,
        "global_step": 50 * batches_per_epoch, "completed_epoch": 50,
        "initialization_sha256": initialization_sha256,
        "scheduler_state": {
            "scheduler_id": "sprint18-n4-scale-epoch-v2", "completed_epochs": 50,
            "active_epoch": 50, "learning_rate": 1e-3, "global_step": 50 * batches_per_epoch,
        },
        "sampler_state": sampler_state, "coefficient_state": coefficient_state,
        "scope": "experiment", "representation_step": 0, "optimizer_scope": "n4_scale",
        "ema_updates": 100 * ceil(5446 / 128),
        "paired_b0_checkpoint_sha256": "cd" * 32,
    }
    digest = save_sprint18_checkpoint(
        path, model, optimizer, arm_specific_state=arm_state, **checkpoint_inputs
    )
    restored_model = V1RepresentationModel(config, patchifier=patchifier)
    restored_scale = init_n4_scale_model(config.d_model, MODEL_SEEDS[0] + 999)
    restored_optimizer = torch.optim.AdamW(restored_scale.parameters(), lr=1e-3, weight_decay=1e-4)
    restored = restore_sprint18_checkpoint(
        path, restored_model, restored_optimizer, arm=arm, binding=binding,
        allowed_scope="experiment", arm_modules={"scale_model": restored_scale},
    )
    assert restored.checkpoint_sha256 == digest
    assert restored.representation_step == 0
    assert restored.optimizer_scope == "n4_scale"
    assert state_sha256(restored_model.state_dict()) == state_sha256(model.state_dict())
    for key in model_before:
        left, right = restored_model.state_dict()[key], model_before[key]
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            assert torch.equal(left.detach().cpu(), right)
        else:
            assert left == right
    for key in ema_before:
        left, right = restored_model.target_encoder.state_dict()[key], ema_before[key]
        assert torch.equal(left.detach().cpu(), right)
    for key, value in scale.state_dict().items():
        assert torch.equal(restored_scale.state_dict()[key].detach().cpu(), value.detach().cpu())

    def _contradict_aggregate(state: dict) -> None:
        gradients = state["scale_initialization"]["first_step_gradients"]
        gradients["total_nonzero"] = gradients["total_nonzero"] + 1

    def _contradict_count(state: dict) -> None:
        initialization = state["scale_initialization"]
        initialization["init_anchor"]["record_count"] -= 1
        initialization["init_interiority"]["record_count"] -= 1

    def _contradict_margin(state: dict) -> None:
        interiority = state["scale_initialization"]["init_interiority"]
        interiority["floor_margin"] = interiority["floor_margin"] + 1e-9

    for label, mutate, message in (
        ("aggregate", _contradict_aggregate, "initialization/guard provenance is invalid"),
        ("count", _contradict_count, "record count differs from the final Fit scope"),
        ("margin", _contradict_margin, "initialization/guard provenance is invalid"),
    ):
        contradictory = copy.deepcopy(arm_state)
        mutate(contradictory)
        # ``save_sprint18_checkpoint`` runs the same ``validate_sprint18_checkpoint``
        # that guards the coherent restore path, so a contradictory envelope is
        # refused before any bytes are written and can never be loaded.
        with pytest.raises(ValueError, match=message):
            save_sprint18_checkpoint(
                tmp_path / f"n4-scale-coherent-{label}.pt", model, optimizer,
                arm_specific_state=contradictory, **checkpoint_inputs,
            )


def test_n4_normalized_score_matches_frozen_formula_and_monotonic_rescale() -> None:
    from representation import n4_normalized_patch_scores

    torch.manual_seed(181801)
    residuals = torch.rand(24) + 0.05
    log_v = torch.randn(24) * 0.5
    scores = n4_normalized_patch_scores(residuals, log_v)
    assert torch.allclose(scores, residuals / (torch.exp(log_v) + 1e-6))
    assert torch.isfinite(scores).all() and bool((scores >= 0.0).all())
    constant = n4_normalized_patch_scores(torch.tensor([0.2, 0.8]), torch.tensor([0.0, 0.0]))
    assert constant[1] / constant[0] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="share length"):
        n4_normalized_patch_scores(torch.rand(3), torch.rand(4))
    with pytest.raises(ValueError, match="non-negative"):
        n4_normalized_patch_scores(torch.tensor([-0.1, 0.2]), torch.zeros(2))


def test_n4_cache_binding_proves_checkpoint_provenance_and_refuses_substitution() -> None:
    from representation import n4_cache_binding_digest, verify_n4_cache_binding

    bounds = {"q05": 0.02, "q95": 1.4, "v_floor": 0.02, "v_cap": 1.4}
    digest = n4_cache_binding_digest(
        checkpoint_sha256="ab" * 32, residual_records_sha256="cd" * 32,
        fit_role_sha256="ef" * 32, scale_state_sha256="12" * 32,
        scale_parameterization=N4_SCALE_PARAMETERIZATION, bounds=bounds,
    )
    assert verify_n4_cache_binding(
        expected_digest=digest, checkpoint_sha256="ab" * 32,
        residual_records_sha256="cd" * 32, fit_role_sha256="ef" * 32,
        scale_state_sha256="12" * 32,
        scale_parameterization=N4_SCALE_PARAMETERIZATION, bounds=bounds,
    ) == digest
    with pytest.raises(ValueError, match="refusing substituted"):
        verify_n4_cache_binding(
            expected_digest=digest, checkpoint_sha256="00" + "ab" * 31,
            residual_records_sha256="cd" * 32, fit_role_sha256="ef" * 32,
            scale_state_sha256="12" * 32,
            scale_parameterization=N4_SCALE_PARAMETERIZATION, bounds=bounds,
        )
    with pytest.raises(ValueError, match="frozen v3 scale parameterization"):
        n4_cache_binding_digest(
            checkpoint_sha256="ab" * 32, residual_records_sha256="cd" * 32,
            fit_role_sha256="ef" * 32, scale_state_sha256="12" * 32,
            scale_parameterization="legacy-dead-zone-v1", bounds=bounds,
        )
    with pytest.raises(ValueError, match="exact q05/q95"):
        n4_cache_binding_digest(
            checkpoint_sha256="ab" * 32, residual_records_sha256="cd" * 32,
            fit_role_sha256="ef" * 32, scale_state_sha256="12" * 32,
            scale_parameterization=N4_SCALE_PARAMETERIZATION,
            bounds={"q05": 0.02, "q99": 1.4, "v_floor": 0.02, "v_cap": 1.4},
        )


def test_n4_calibration_q95_uses_linear_quantile_and_rejects_empty_pool() -> None:
    from representation import n4_calibration_q95

    pool = {f"file-{i:02d}": 0.5 + 0.1 * i for i in range(9)}
    threshold = n4_calibration_q95(pool)
    assert threshold == pytest.approx(
        float(np.quantile(np.array(sorted(pool.values())), 0.95, method="linear")))
    assert math.isfinite(threshold) and threshold >= 0.0
    with pytest.raises(ValueError, match="non-empty"):
        n4_calibration_q95({})


def test_n4_score_files_keeps_b0_aggregation_and_reports_raw_scale_normalized() -> None:
    from representation import n4_score_files

    config = _tiny_config()
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    scale = init_n4_scale_model(config.d_model, MODEL_SEEDS[0])
    samples = _tiny_samples()
    scored = n4_score_files(
        model=model, samples=samples, scale_model=scale,
        v_floor=0.02, v_cap=1.4, patchifier=patchifier, masking_seed=91813,
    )
    assert set(scored) == {sample.file_id for sample in samples}
    for file_id, entry in scored.items():
        assert set(entry) == {"raw", "scale", "normalized"}
        # File score aggregates eligible masked valid patches only (same B0
        # filtering/aggregation); per-patch formula e_t/(v_t+eps) is pinned by
        # test_n4_normalized_score_matches_frozen_formula_and_monotonic_rescale.
        # Here: finite/non-negative triple, scale inside the frozen cache bounds.
        assert all(math.isfinite(value) and value >= 0.0 for value in entry.values())
    with pytest.raises(ValueError, match="non-empty file batch"):
        n4_score_files(
            model=model, samples=[], scale_model=scale,
            v_floor=0.02, v_cap=1.4, patchifier=patchifier, masking_seed=91813,
        )
    with pytest.raises(ValueError, match="scale head device differs"):
        other = init_n4_scale_model(config.d_model, MODEL_SEEDS[0]).to("meta")
        n4_score_files(
            model=model, samples=samples, scale_model=other,
            v_floor=0.02, v_cap=1.4, patchifier=patchifier, masking_seed=91813,
        )


def test_n4_consumer_scale_containment_covers_float32_rounding() -> None:
    """File-mean consumer scales stay in bounds modulo float32 rounding steps.

    Guards the exact T13B-A01 public-consumer abort: an absolute tolerance
    (e.g. 1e-9) is arithmetically unsound because real float32 file-mean
    rounding near unit-scale values is ~1e-8, strictly above it, so such a
    check aborts on sound scales. The permanent runner's per-side 8*eps32
    allowance (each side scaled by its own bound, so the frozen
    v_cap/v_floor <= 10 edge rule cannot widen one side's margin through the
    other bound) covers only the two float32 derivation steps (clamp
    log/exp, then mean accumulation; retained C04 characterization
    ``artifacts/sprint-18/s18-t13b-c04/characterize_float32.py`` +
    ``measurements.json``: 472 trials, worst bound excursion ~1.1e-7 with
    ~8.6x margin on this CPU; backend limits documented there); frozen
    bounds, per-patch clamp, and cap gate are unchanged. Beyond-allowance
    excursions -- including the old 1e-5-relative hole at either bound --
    are rejected.
    """
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    from sprint18_task13b_n4_capacity import consumer_scale_within_bounds

    # Genuine escapes are still rejected -- including excursions inside the
    # old 1e-5-relative hole (1e-6 relative at either bound exceeds the
    # ~9.5e-7 new allowance and must fail).
    assert not consumer_scale_within_bounds(0.20, 0.46, 4.64)
    assert not consumer_scale_within_bounds(9.99, 0.46, 4.64)
    assert not consumer_scale_within_bounds(4.64 * (1 + 1e-6), 0.46, 4.64)
    assert not consumer_scale_within_bounds(0.46 * (1 - 1e-6), 0.46, 4.64)
    assert not consumer_scale_within_bounds(float("nan"), 0.46, 4.64)
    assert not consumer_scale_within_bounds(-0.01, 0.46, 4.64)
    # Exact containment passes.
    assert consumer_scale_within_bounds(0.46, 0.46, 4.64)
    assert consumer_scale_within_bounds(4.64, 0.46, 4.64)
    assert consumer_scale_within_bounds(0.81, 0.46, 4.64)
    # Worst-case float32 rounding magnitude (~1.1e-7 retained measurement) passes.
    assert consumer_scale_within_bounds(0.46 - 2e-8, 0.46, 4.64)
    assert consumer_scale_within_bounds(4.64 + 2e-7, 0.46, 4.64)
    assert consumer_scale_within_bounds(4.64 * (1 + 2.5e-7), 0.46, 4.64)
    # Real public-consumer triple through the committed entrypoint obeys it.
    from representation import n4_score_files

    config = _tiny_config()
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    scale = init_n4_scale_model(config.d_model, MODEL_SEEDS[0])
    samples = _tiny_samples()
    scored = n4_score_files(
        model=model, samples=samples, scale_model=scale,
        v_floor=0.02, v_cap=1.4, patchifier=patchifier, masking_seed=91813,
    )
    for file_id, entry in scored.items():
        assert consumer_scale_within_bounds(entry["scale"], 0.02, 1.4), file_id


def _synthetic_scale_pool(n: int = 700, d: int = 128, seed: int = 424242) -> tuple[list[torch.Tensor], list[float]]:
    """Deterministic 700-record pool mirroring the reviewed guard-contract fixture."""
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.nn.functional.layer_norm(torch.randn(n, d, generator=generator), (d,))
    residual_generator = torch.Generator().manual_seed(seed + 1)
    residuals = 0.07 * torch.exp(0.35 * torch.randn(n, generator=residual_generator))
    return [row.detach().clone() for row in matrix], [float(value) for value in residuals]


def _fit_fixture() -> tuple[object, object, list[str], str]:
    binding = load_task6_binding(HANDOFF)
    arm = make_sprint18_arm_config(
        "N4-CRN", MODEL_SEEDS[0],
        {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="locked", coefficient_lock_sha256="ab" * 32,
    )
    fit_ids = [item.actual_history_id for item in binding.role_histories("FIT")]
    return binding, arm, fit_ids[:2], fit_ids[2]


def _synthetic_records(history_ids: list[str], contexts: list[torch.Tensor], residuals: list[float]) -> list[N4ScaleRecord]:
    return [
        N4ScaleRecord(
            history_id=history_ids[index % len(history_ids)],
            file_id=f"file-{index:04d}",
            patch_index=index,
            context=context,
            residual=residual,
        )
        for index, (context, residual) in enumerate(zip(contexts, residuals))
    ]


def _initialize_and_probe(
    scale, contexts: list[torch.Tensor], residuals: list[float], bounds: dict[str, float]
) -> dict[str, object]:
    """Build the exact v3 initialization/guard provenance for a fitted probe."""
    anchor = n4_anchor_scale_model(scale, residuals)
    interiority = n4_guard_initialized_interiority(
        scale, contexts, v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"])
    )
    scale.train()
    scale.zero_grad(set_to_none=True)
    batch = torch.stack(contexts[:8], dim=0)
    predicted = scale(batch, v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]))
    loss = n4_scale_nll(torch.tensor(residuals[:8], dtype=torch.float32), predicted["log_v"])
    loss.backward()
    gradients = n4_guard_first_step_gradients(scale)
    return {
        "init_anchor": anchor,
        "init_interiority": interiority,
        "first_step_gradients": gradients,
    }


def test_n4_scale_anchors_only_output_bias_to_fit_mean_and_fit_records_provenance() -> None:
    scale = init_n4_scale_model(128, MODEL_SEEDS[0])
    untouched = {
        name: value.detach().cpu().clone()
        for name, value in scale.state_dict().items()
        if name != "out.bias"
    }
    residuals = [0.04, 0.09, 0.065]
    anchor = n4_anchor_scale_model(scale, residuals)
    expected_v_ref = float(np.asarray(residuals, dtype=np.float64).mean())
    assert anchor == {
        "v_ref": expected_v_ref,
        "log_v_ref": math.log(expected_v_ref),
        "record_count": 3,
    }
    assert float(scale.out.bias.detach()) == pytest.approx(math.log(expected_v_ref), rel=0.0, abs=1e-7)
    for name, value in untouched.items():
        assert torch.equal(scale.state_dict()[name].detach().cpu(), value), name
    assert scale.out.bias.requires_grad

    contexts, pool_residuals = _synthetic_scale_pool()
    bounds = n4_residual_bounds(pool_residuals)
    binding, arm, train_histories, heldout_id = _fit_fixture()
    records = _synthetic_records(train_histories, contexts, pool_residuals)
    fitted = init_n4_scale_model(128, MODEL_SEEDS[0])
    result = fit_n4_scale_scope(
        records=records, training_history_ids=train_histories,
        scope=f"fold:{heldout_id}", arm=arm, binding=binding, scale_model=fitted,
        v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
    )
    assert result["scale_parameterization"] == N4_SCALE_PARAMETERIZATION
    assert result["guard_contract"] == N4_GUARD_CONTRACT
    assert result["init_anchor"]["v_ref"] == pytest.approx(
        float(np.asarray(pool_residuals, dtype=np.float64).mean()), rel=1e-12
    )
    assert result["init_anchor"]["record_count"] == len(records)
    assert result["init_interiority"]["frac_at_floor"] == 0.0
    assert result["init_interiority"]["frac_at_cap"] == 0.0
    assert result["first_step_gradients"]["total_nonzero"] >= 1
    assert result["first_step_gradients"]["total_nonfinite"] == 0
    assert result["first_step_gradients"]["missing_grad_parameters"] == []


def test_n4_scale_rejects_nonpositive_or_nonfinite_anchor_reference() -> None:
    scale = init_n4_scale_model(128, MODEL_SEEDS[0])
    with pytest.raises(ValueError, match="finite and positive"):
        n4_anchor_scale_model(scale, [0.0, 0.0])
    with pytest.raises(ValueError, match="finite and non-negative"):
        n4_anchor_scale_model(scale, [0.07, float("nan")])
    with pytest.raises(ValueError, match="non-empty"):
        n4_anchor_scale_model(scale, [])
    binding, arm, _train_histories, _heldout_id = _fit_fixture()
    with pytest.raises(ValueError, match="finite and positive"):
        fit_n4_scale_scope(
            records=[N4ScaleRecord(
                history_id="HIST-00", file_id="file-0000", patch_index=0,
                context=torch.zeros(128), residual=0.0,
            )],
            training_history_ids=["HIST-00"], scope="fold:HIST-01",
            arm=arm, binding=binding, scale_model=scale, v_floor=0.02, v_cap=0.4,
        )


def test_n4_scale_checkpoint_envelope_strict_restore_and_legacy_refusal(tmp_path) -> None:
    contexts, residuals = _synthetic_scale_pool()
    bounds = n4_residual_bounds(residuals)
    scale = init_n4_scale_model(128, MODEL_SEEDS[0])
    initialization = _initialize_and_probe(scale, contexts, residuals, bounds)
    path = tmp_path / "n4-scale-v3.pt"
    meta = save_n4_scale_checkpoint(
        path, scale, fit_scope="fold:S18I-ITER-0005-FIT-01",
        training_history_ids=["S18I-ITER-0005-FIT-02", "S18I-ITER-0005-FIT-03"],
        v_floor=bounds["v_floor"], v_cap=bounds["v_cap"], initialization=initialization,
    )
    assert meta["schema"] == N4_SCALE_CHECKPOINT_SCHEMA
    assert meta["parameterization"] == N4_SCALE_PARAMETERIZATION
    restored = init_n4_scale_model(128, MODEL_SEEDS[0] + 999)
    resolved = load_n4_scale_checkpoint(
        path, restored, expected_fit_scope="fold:S18I-ITER-0005-FIT-01",
        expected_training_history_ids=["S18I-ITER-0005-FIT-02", "S18I-ITER-0005-FIT-03"],
        expected_bounds={"v_floor": bounds["v_floor"], "v_cap": bounds["v_cap"]},
    )
    assert resolved["state_dict_sha256"] == meta["state_dict_sha256"]
    for name, value in scale.state_dict().items():
        assert torch.equal(restored.state_dict()[name].detach().cpu(), value.detach().cpu()), name

    legacy = tmp_path / "legacy-a03-scale.pt"
    torch.save(scale.state_dict(), legacy)
    with pytest.raises(ValueError, match="legacy or incompatible N4 scale checkpoint"):
        load_n4_scale_checkpoint(legacy, init_n4_scale_model(128, MODEL_SEEDS[0]))

    tampered = torch.load(path, map_location="cpu", weights_only=True)
    tampered["parameterization"] = "dead-zone-v1"
    bad_parameterization = tmp_path / "bad-parameterization.pt"
    torch.save(tampered, bad_parameterization)
    with pytest.raises(ValueError, match="parameterization is unknown"):
        load_n4_scale_checkpoint(bad_parameterization, init_n4_scale_model(128, MODEL_SEEDS[0]))

    tampered = torch.load(path, map_location="cpu", weights_only=True)
    tampered["guard_contract"] = "v1-no-guards"
    bad_guard = tmp_path / "bad-guard.pt"
    torch.save(tampered, bad_guard)
    with pytest.raises(ValueError, match="guard contract is unknown"):
        load_n4_scale_checkpoint(bad_guard, init_n4_scale_model(128, MODEL_SEEDS[0]))

    with pytest.raises(ValueError, match="fit scope differs"):
        load_n4_scale_checkpoint(
            path, init_n4_scale_model(128, MODEL_SEEDS[0]),
            expected_fit_scope="final",
        )
    with pytest.raises(ValueError, match="bounds differ"):
        load_n4_scale_checkpoint(
            path, init_n4_scale_model(128, MODEL_SEEDS[0]),
            expected_bounds={"v_floor": 0.01, "v_cap": bounds["v_cap"]},
        )

    def _contradict_anchor_log(payload: dict) -> None:
        payload["init_anchor"]["log_v_ref"] += 1e-9

    def _contradict_floor_margin(payload: dict) -> None:
        payload["init_interiority"]["floor_margin"] += 1e-9

    def _contradict_tolerance(payload: dict) -> None:
        payload["init_interiority"]["tolerance"] = 1e-5

    def _contradict_aggregate(payload: dict) -> None:
        payload["first_step_gradients"]["total_nonzero"] += 1

    def _contradict_roster_count(payload: dict) -> None:
        entry = payload["first_step_gradients"]["tensors"][-1]
        entry["nonzero"] += 1
        entry["zero"] -= 1

    def _contradict_log_bounds(payload: dict) -> None:
        interiority = payload["init_interiority"]
        interiority["log_floor"] += 1e-9
        interiority["floor_margin"] = interiority["log_v_min"] - interiority["log_floor"]

    for label, mutate, message in (
        ("anchor-log", _contradict_anchor_log, "does not follow"),
        ("floor-margin", _contradict_floor_margin, "margin does not follow"),
        ("tolerance", _contradict_tolerance, "frozen tolerance"),
        ("aggregate", _contradict_aggregate, "totals do not reconcile"),
        ("roster-count", _contradict_roster_count, "counts are inconsistent"),
        ("log-bounds", _contradict_log_bounds, "checkpoint bounds"),
    ):
        contradictory = copy.deepcopy(initialization)
        mutate(contradictory)
        with pytest.raises(ValueError, match=message):
            save_n4_scale_checkpoint(
                tmp_path / f"contradictory-{label}.pt", scale,
                fit_scope="fold:S18I-ITER-0005-FIT-01",
                training_history_ids=["S18I-ITER-0005-FIT-02", "S18I-ITER-0005-FIT-03"],
                v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
                initialization=contradictory,
            )
        tampered_payload = torch.load(path, map_location="cpu", weights_only=True)
        tampered_payload["initialization"] = contradictory
        bad_initialization = tmp_path / f"bad-initialization-{label}.pt"
        torch.save(tampered_payload, bad_initialization)
        with pytest.raises(ValueError, match=message):
            load_n4_scale_checkpoint(bad_initialization, init_n4_scale_model(128, MODEL_SEEDS[0]))


def test_n4_scale_guard_failures_apply_zero_updates_and_leave_state_unchanged() -> None:
    contexts, _pool_residuals = _synthetic_scale_pool()
    deadzone = init_n4_scale_model(128, MODEL_SEEDS[0])
    before = {name: value.detach().cpu().clone() for name, value in deadzone.state_dict().items()}
    with pytest.raises(N4ScaleGuardError) as guard_a:
        n4_guard_initialized_interiority(
            deadzone, contexts, v_floor=0.02, v_cap=0.4
        )
    assert guard_a.value.guard_id == "n4-scale-init-interiority"
    assert guard_a.value.diagnostics["frac_at_cap"] == 1.0
    for name, value in before.items():
        assert torch.equal(deadzone.state_dict()[name].detach().cpu(), value), name
    assert all(parameter.grad is None for parameter in deadzone.parameters())

    anchored = init_n4_scale_model(128, MODEL_SEEDS[0])
    n4_anchor_scale_model(anchored, [0.07] * len(contexts))
    anchored_before = {
        name: value.detach().cpu().clone() for name, value in anchored.state_dict().items()
    }
    poisoned = [row.clone() for row in contexts]
    with torch.no_grad():
        poisoned[0][0] = float("nan")
    # A non-finite context is an invalid Guard A forward input, so it must
    # surface as the typed guard failure (the public forward's single chunk
    # finiteness check is classified, not duplicated by the guard).
    with pytest.raises(N4ScaleGuardError) as guard_invalid:
        n4_guard_initialized_interiority(anchored, poisoned, v_floor=0.02, v_cap=0.4)
    assert guard_invalid.value.guard_id == "n4-scale-init-interiority"
    assert guard_invalid.value.diagnostics["record_count"] == 0
    assert not isinstance(guard_invalid.value, ValueError)
    # Wrong-shape contexts are structural precondition violations and keep
    # their ValueError classification before any forward runs.
    with pytest.raises(ValueError, match="1D latent row"):
        n4_guard_initialized_interiority(
            anchored, [contexts[0].reshape(1, -1)], v_floor=0.02, v_cap=0.4
        )
    for name, value in anchored_before.items():
        assert torch.equal(anchored.state_dict()[name].detach().cpu(), value), name
    assert all(parameter.grad is None for parameter in anchored.parameters())
    assert anchored.training  # Guard A preserves the fit-set training mode

    scaled = init_n4_scale_model(128, MODEL_SEEDS[0])
    n4_anchor_scale_model(scaled, [0.07] * len(contexts))
    scaled.train()
    scaled.zero_grad(set_to_none=True)
    predicted = scaled(torch.stack(contexts[:8]), v_floor=0.02, v_cap=0.4)
    n4_scale_nll(torch.full((8,), 0.07), predicted["log_v"]).backward()
    with torch.no_grad():
        scaled.hidden.weight.grad[0, 0] = float("nan")
    before = {name: value.detach().cpu().clone() for name, value in scaled.state_dict().items()}
    with pytest.raises(N4ScaleGuardError) as guard_b:
        n4_guard_first_step_gradients(scaled)
    assert guard_b.value.guard_id == "n4-scale-first-step-gradients"
    assert guard_b.value.diagnostics["total_nonfinite"] == 1
    for name, value in before.items():
        assert torch.equal(scaled.state_dict()[name].detach().cpu(), value), name


def test_n4_guard_calls_do_not_change_the_training_trajectory(monkeypatch) -> None:
    import representation.crn as crn

    contexts, residuals = _synthetic_scale_pool()
    bounds = n4_residual_bounds(residuals)
    binding, arm, train_histories, heldout_id = _fit_fixture()
    records = _synthetic_records(train_histories, contexts, residuals)
    monkeypatch.setattr(crn, "N4_SCALE_EPOCHS", 2)

    def _fit(scale) -> dict[str, object]:
        return fit_n4_scale_scope(
            records=records, training_history_ids=train_histories,
            scope=f"fold:{heldout_id}", arm=arm, binding=binding, scale_model=scale,
            v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
        )

    guarded_scale = init_n4_scale_model(128, MODEL_SEEDS[0])
    guarded = _fit(guarded_scale)
    monkeypatch.setattr(crn, "n4_guard_initialized_interiority", lambda *args, **kwargs: {})
    monkeypatch.setattr(crn, "n4_guard_first_step_gradients", lambda *args, **kwargs: {})
    control_scale = init_n4_scale_model(128, MODEL_SEEDS[0])
    control = _fit(control_scale)
    assert guarded["optimizer_steps"] == control["optimizer_steps"] == 2 * math.ceil(len(records) / N4_SCALE_BATCH_SIZE)
    assert guarded["epoch_order_seeds"] == control["epoch_order_seeds"]
    assert guarded["per_epoch_nll"] == control["per_epoch_nll"]
    for name, value in guarded_scale.state_dict().items():
        assert torch.equal(value.detach().cpu(), control_scale.state_dict()[name].detach().cpu()), name


def test_n4_restored_scale_checkpoint_scores_identically_through_public_consumer(tmp_path) -> None:
    from representation import n4_score_files

    config = _tiny_config()
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    before = {
        name: (value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value)
        for name, value in model.state_dict().items()
    }
    binding, arm, train_histories, heldout_id = _fit_fixture()
    samples = _tiny_samples()
    contexts, residuals = _synthetic_scale_pool()
    train = _synthetic_records(train_histories, contexts, residuals)
    bounds = n4_residual_bounds(residuals)
    scale = init_n4_scale_model(128, MODEL_SEEDS[0])
    result = fit_n4_scale_scope(
        records=train, training_history_ids=train_histories,
        scope=f"fold:{heldout_id}", arm=arm, binding=binding, scale_model=scale,
        v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
    )
    path = tmp_path / "consumer-scale.pt"
    save_n4_scale_checkpoint(
        path, scale, fit_scope=f"fold:{heldout_id}",
        training_history_ids=train_histories,
        v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
        initialization={
            "init_anchor": dict(result["init_anchor"]),
            "init_interiority": dict(result["init_interiority"]),
            "first_step_gradients": dict(result["first_step_gradients"]),
        },
    )
    restored = init_n4_scale_model(128, MODEL_SEEDS[0] + 999)
    load_n4_scale_checkpoint(
        path, restored, expected_fit_scope=f"fold:{heldout_id}",
        expected_training_history_ids=train_histories,
        expected_bounds={"v_floor": bounds["v_floor"], "v_cap": bounds["v_cap"]},
    )
    consumer_samples = samples[:2]
    live_scores = n4_score_files(
        model=model, samples=consumer_samples, scale_model=scale,
        v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
        patchifier=patchifier, masking_seed=91813,
    )
    restored_scores = n4_score_files(
        model=model, samples=consumer_samples, scale_model=restored,
        v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
        patchifier=patchifier, masking_seed=91813,
    )
    assert live_scores == restored_scores
    for entry in live_scores.values():
        assert set(entry) == {"raw", "scale", "normalized"}
        assert all(math.isfinite(value) and value >= 0.0 for value in entry.values())
        assert bounds["v_floor"] - 1e-5 <= entry["scale"] <= bounds["v_cap"] + 1e-5
    for name, value in before.items():
        observed = model.state_dict()[name]
        if isinstance(observed, torch.Tensor) and isinstance(value, torch.Tensor):
            assert torch.equal(observed.detach().cpu(), value), name
        else:
            assert observed == value, name


def test_n4_runner_records_typed_guard_invalid_context_failure_and_stops(
    monkeypatch, tmp_path
) -> None:
    """The permanent runner persists, emits, and terminates on an invalid context.

    Drives the real ``run_pilot`` consumer branch for an invalid Guard A
    context. Labelled adapters replace only fixture/identity I/O
    (``_load_cache_records``, ``records_from_cache``, ``run_preflight``, and the
    B0 restore/digest probes); the production record validation, public scale
    forward, Guard A, and the runner's failure record/event/re-raise path run
    unchanged. No optimizer may be constructed for the 4,417-parameter scale
    head, and no fold result or checkpoint may be written.
    """
    sys.path.insert(0, str(ROOT / "experiments"))
    import representation.sprint18_checkpoint as sprint18_checkpoint
    import sprint18_task13b_n4_capacity as runner

    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    driver_sha = hashlib.sha256(
        (ROOT / "experiments" / "sprint18_task13b_n4_capacity.py").read_bytes()
    ).hexdigest()
    model_config = {
        "n_channels": 6, "patch_size": 32, "stride": 16, "d_model": 128,
        "sequence_layers": 4, "attention_heads": 4, "dropout": 0.0,
        "n_robots": 9, "n_programs": 8, "ema_decay": 0.996,
        "contrastive_weight_max": 0.05, "seed": MODEL_SEEDS[0],
    }
    binding_path = tmp_path / "binding.json"
    binding_path.write_text("{}", encoding="utf-8")
    fixture: dict[str, dict] = {
        "prediction_cache_paths": {}, "checkpoints": {},
        "live_model_state_sha256": {}, "ema_state_sha256": {},
    }
    for seed in runner.MODEL_SEEDS:
        cache_path = tmp_path / f"cache-{seed}.json"
        checkpoint_path = tmp_path / f"ckpt-{seed}.pt"
        cache_path.write_text("{}", encoding="utf-8")
        checkpoint_path.write_bytes(b"controlled")
        fixture["prediction_cache_paths"][str(seed)] = str(cache_path)
        fixture["checkpoints"][str(seed)] = {"path": str(checkpoint_path)}
        fixture["live_model_state_sha256"][str(seed)] = "controlled-state"
        fixture["ema_state_sha256"][str(seed)] = "controlled-state"
    bdoc = {"source_commit": head, "runner_sha256": driver_sha, "fixture": fixture}
    out_dir = tmp_path / "run"

    def _restore(ckpt_path, *args, **kwargs):
        seed = int(Path(ckpt_path).stem.split("-")[1])
        return SimpleNamespace(
            checkpoint_sha256=runner.EXPECTED_CONTRACT["checkpoints"][str(seed)]
        )

    nan_context = torch.full((128,), float("nan"))

    def _pools(cache_records, history_ids):
        return {
            hid: [
                N4ScaleRecord(
                    history_id=hid, file_id=f"{hid}-file-0000", patch_index=0,
                    context=nan_context.clone(), residual=0.07,
                )
            ]
            for hid in history_ids
        }

    created_optimizers: list[int] = []
    real_adamw = torch.optim.AdamW

    def _recording_adamw(params, *args, **kwargs):
        parameters = list(params)
        created_optimizers.append(sum(p.numel() for p in parameters))
        return real_adamw(parameters, *args, **kwargs)

    monkeypatch.setattr(runner, "run_preflight", lambda *args, **kwargs: {"PREFLIGHT_OK": True})
    monkeypatch.setattr(
        runner, "_load_cache_records",
        lambda path: ({"model_config": model_config, "cache_sha256": "controlled-inner"}, []),
    )
    monkeypatch.setattr(runner, "records_from_cache", _pools)
    monkeypatch.setattr(sprint18_checkpoint, "restore_sprint18_checkpoint", _restore)
    monkeypatch.setattr(sprint18_checkpoint, "state_sha256", lambda state: "controlled-state")
    monkeypatch.setattr(torch.optim, "AdamW", _recording_adamw)

    args = SimpleNamespace(
        worktree=str(ROOT), handoff=str(HANDOFF), binding=str(binding_path),
        out_dir=str(out_dir), source_commit=head, driver_sha256=driver_sha,
        mode="pilot", device="cpu",
    )
    with pytest.raises(N4ScaleGuardError) as raised:
        runner.run_pilot(args, bdoc)
    assert raised.value.guard_id == "n4-scale-init-interiority"

    failure = json.loads((out_dir / "guard_failure.json").read_text(encoding="utf-8"))
    assert failure["schema"] == "s18-task13b-v3-guard-failure-v1"
    assert failure["model_seed"] == MODEL_SEEDS[0]
    assert failure["fold_index"] == 0
    assert failure["guard_id"] == "n4-scale-init-interiority"
    assert failure["guard_contract"] == N4_GUARD_CONTRACT
    assert failure["optimizer_steps_applied"] == 0
    assert failure["checkpoint_written"] is False
    assert failure["run_status"] == "FAILED"
    assert failure["scope"] == f"fold:{failure['heldout_history_id']}"
    assert failure["heldout_history_id"] in {
        "S18I-ITER-0005-FIT-01", "S18I-ITER-0005-FIT-02", "S18I-ITER-0005-FIT-03",
    }
    assert len(failure["training_history_ids"]) == 2
    assert failure["heldout_history_id"] not in failure["training_history_ids"]
    assert "finite" in failure["message"]
    assert set(failure["diagnostics"]) == {
        "record_count", "log_v_min", "log_v_max", "log_floor", "log_cap",
        "floor_margin", "cap_margin", "frac_at_floor", "frac_at_cap", "tolerance",
    }

    events = [
        json.loads(line)
        for line in (out_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events == [{
        "event": "scale_guard_failure", "seed": MODEL_SEEDS[0], "fold": 0,
        "scope": failure["scope"], "guard_id": "n4-scale-init-interiority",
    }]
    assert created_optimizers and 4417 not in created_optimizers
    assert list((out_dir / "runs").iterdir()) == []
    assert list((out_dir / "checkpoints").iterdir()) == []
