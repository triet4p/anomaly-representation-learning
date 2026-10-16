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

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from representation import (
    N4_SCALE_BATCH_SIZE,
    N4_SCALE_EPOCHS,
    N4ScaleRecord,
    V1Config,
    V1RepresentationModel,
    build_n4_scale_records,
    fit_n4_scale_scope,
    init_n4_scale_model,
    make_sprint18_arm_config,
    n4_canonical_fold_splits,
    n4_patch_residual,
    n4_residual_bounds,
    n4_saturation,
    n4_scale_nll,
    n4_spearman,
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


def _tiny_records(n: int = 40, d: int = 8, seed: int = 181801) -> list[N4ScaleRecord]:
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
    scale = init_n4_scale_model(8, MODEL_SEEDS[0])
    before_params = [p.detach().cpu().clone() for p in scale.parameters()]
    raw_train = _tiny_records(40, 8)
    heldout_id = fit_ids[0]
    train_histories = sorted(h for h in fit_ids if h != heldout_id)
    train = [
        N4ScaleRecord(history_id=train_histories[i % 2], file_id=r.file_id, patch_index=r.patch_index, context=r.context, residual=r.residual)
        for i, r in enumerate(raw_train)
    ]
    heldout = [
        N4ScaleRecord(history_id=heldout_id, file_id=r.file_id, patch_index=r.patch_index, context=r.context, residual=r.residual)
        for r in _tiny_records(17, 8, seed=77)
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


def test_n4_fold_rejects_heldout_contamination_and_wrong_counts() -> None:
    binding = load_task6_binding(HANDOFF)
    arm = make_sprint18_arm_config(
        "N4-CRN", MODEL_SEEDS[0],
        {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="locked", coefficient_lock_sha256="ab" * 32,
    )
    scale = init_n4_scale_model(8, MODEL_SEEDS[0])
    train = _tiny_records(10, 8)
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
    optimizer.step()
    for param_state in optimizer.state_dict()["state"].values():
        param_state["step"] = torch.tensor(16350)
    initialization_sha256 = state_sha256(model.state_dict())
    model_before = {k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v) for k, v in model.state_dict().items()}
    ema_before = {k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v) for k, v in model.target_encoder.state_dict().items()}
    from representation.sprint18_harness import FIT_VALID_PATCHES, deterministic_sampler_order
    per_patch = FIT_VALID_PATCHES
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
    fold_counts = [per_patch - 55797, per_patch - 57716, per_patch - 53906]
    arm_state = {
        "paired_b0_checkpoint_sha256": "cd" * 32,
        "paired_b0_representation_step": 100 * ceil(5446 / 128),
        "paired_b0_model_state_sha256": state_sha256(model.state_dict()),
        "paired_b0_ema_state_sha256": state_sha256(model.target_encoder.state_dict()),
        "scale_model_state": scale.state_dict(),
        "scale_model_state_sha256": state_sha256(scale.state_dict()),
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
            "q05": 0.02, "q99": 1.4, "v_floor": 0.02, "v_cap": 1.4,
            "fit_role_sha256": fit_role, "residual_records_sha256": "ef" * 32,
        },
    }
    path = tmp_path / "n4-scale-coherent.pt"
    digest = save_sprint18_checkpoint(
        path, model, optimizer, arm=arm, binding=binding,
        global_step=50 * batches_per_epoch, completed_epoch=50,
        initialization_sha256=initialization_sha256,
        scheduler_state={
            "scheduler_id": "sprint18-n4-scale-epoch-v2", "completed_epochs": 50,
            "active_epoch": 50, "learning_rate": 1e-3, "global_step": 50 * batches_per_epoch,
        },
        sampler_state=sampler_state, coefficient_state=coefficient_state,
        scope="experiment", representation_step=0, optimizer_scope="n4_scale",
        ema_updates=100 * ceil(5446 / 128),
        paired_b0_checkpoint_sha256="cd" * 32, arm_specific_state=arm_state,
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
