from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from representation import NormalReferenceBank, RepresentationInference
from representation.criterion import (
    MSP_AUX_COEFFICIENT_GRID,
    MultiscaleLatentPredictionCriterion,
    msp_allocate_budget,
    msp_fine_mask,
    msp_mask_budget,
    msp_mask_seed,
    msp_place_groups,
)
from representation.config import V1Config
from representation.data import collate_variable_files
from representation.model import V1RepresentationModel
from representation.sprint18_checkpoint import (
    _canonical_sha256,
    coefficient_for_epoch,
    learning_rate_for_epoch,
    restore_sprint18_checkpoint,
    save_sprint18_checkpoint,
    state_sha256,
)
from representation.sprint18_harness import (
    BATCH_SIZE,
    MODEL_SEEDS,
    N2_AUX_COEFFICIENT_GRID,
    TASK9_COEFFICIENT_PILOT_EPOCHS,
    deterministic_sampler_order,
    load_task6_binding,
    make_sprint18_arm_config,
    select_msp_coefficient,
)
from representation.trainer import RepresentationTrainer
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import FileSample, SampleLabel


def _brute_allocate(M: int) -> tuple[int, int, int]:
    best: tuple[int, int, int] | None = None
    best_score: int | None = None
    for g8 in range(M // 8 + 1):
        for g4 in range((M - 8 * g8) // 4 + 1):
            fine = M - 8 * g8 - 4 * g4
            score = 4 * (2 * fine - M) ** 2 + (16 * g4 - M) ** 2 + (32 * g8 - M) ** 2
            candidate = (fine, g4, g8)
            if best_score is None or score < best_score or (
                score == best_score and best is not None and candidate > best
            ):
                best = candidate
                best_score = score
    assert best is not None
    return best


def test_msp_budget_and_allocation_match_exact_integer_program() -> None:
    for n in (0, 1, 2, 5, 8, 12, 31, 43, 49, 100):
        assert msp_mask_budget(n) == min(n, int(round(0.40 * n)))
    for M in (0, 1, 2, 5, 8, 12, 16, 17, 20, 24, 32, 40):
        assert msp_allocate_budget(M) == _brute_allocate(M)
        fine, g4, g8 = msp_allocate_budget(M)
        assert fine + 4 * g4 + 8 * g8 == M
    # Lexicographic tie-break is exercised: every tied optimum resolves to the
    # greatest (F, g4, g8); allocation is deterministic on repeat calls.
    assert msp_allocate_budget(20) == msp_allocate_budget(20)
    with pytest.raises(ValueError, match="non-negative integer"):
        msp_mask_budget(-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        msp_allocate_budget(-1)


def test_msp_placement_is_seeded_disjoint_and_records_fallbacks() -> None:
    valid = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27]
    M = 12
    fine, g4, g8 = msp_allocate_budget(M)
    seed = msp_mask_seed(181801, 1, "a" * 64, "msp-file-001")
    first = msp_place_groups(valid, g8, g4, np.random.default_rng(seed))
    second = msp_place_groups(valid, g8, g4, np.random.default_rng(seed))
    assert first["groups_8"] == second["groups_8"]
    assert first["groups_4"] == second["groups_4"]
    members = [m for g in first["groups_8"] + first["groups_4"] for m in g]  # type: ignore[union-attr]
    assert len(set(members)) == len(members)
    assert all(m in valid for m in members)
    for group in list(first["groups_8"]) + list(first["groups_4"]):  # type: ignore[union-attr]
        span = len(group)
        assert span in (4, 8)
        assert max(group) - min(group) == span - 1
        assert all(x in valid for x in range(min(group), max(group) + 1))
    # 8->4->fine fallback counts are recorded and consistent.
    assert first["q8_fallback_to_4"] == g8 - first["q8"]
    assert first["q4_fallback_to_fine"] == (g4 + 2 * (g8 - first["q8"])) - first["q4"]  # type: ignore[operator]
    patch_data = np.random.default_rng(0).normal(size=(max(valid) + 1, 6, 32)).astype(np.float32)
    fine_count = M - 8 * first["q8"] - 4 * first["q4"]  # type: ignore[operator]
    fine_sel = msp_fine_mask(first["remaining_free"], fine_count, patch_data, np.random.default_rng(seed))  # type: ignore[arg-type]
    assert len(fine_sel["selected"]) == fine_count
    assert not (set(fine_sel["selected"]) & set(members))  # type: ignore[union-attr]
    assert len(set(members) | set(fine_sel["selected"])) == M  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="seed"):
        msp_mask_seed(181801, 1, "not-a-hash", "msp-file-001")


def test_msp_fine_override_hits_exact_target_with_b0_accounting() -> None:
    free = list(range(24))
    patch_data = np.random.default_rng(11).normal(size=(24, 6, 32)).astype(np.float32)
    for target in (0, 1, 5, 10):
        result = msp_fine_mask(free, target, patch_data, np.random.default_rng(91810 + target))
        assert len(result["selected"]) == target
        assert sum(result["composition"].values()) == target  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="exceeds the remaining"):
        msp_fine_mask(free, 25, patch_data, np.random.default_rng(1))


def test_msp_coarse_members_hidden_and_teacher_stop_gradient() -> None:
    config = V1Config(
        n_channels=6, patch_size=32, stride=16, n_robots=9, n_programs=8,
        seed=181801, contrastive_weight_max=0.05, ema_decay=0.996, dropout=0.0,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    model.train()
    batch_size, patches, width = 2, 12, config.d_model
    torch.manual_seed(0)
    context = torch.randn(batch_size, patches, width)
    target = torch.randn(batch_size, patches, width, requires_grad=True)
    valid = torch.ones(batch_size, patches, dtype=torch.bool)
    weights = torch.ones(batch_size, patches)
    out = model.msp_coarse_terms(
        context, target, valid, weights,
        [[(0, 1, 2, 3)], []], [[(4, 5, 6, 7, 8, 9, 10, 11)], []],
    )
    assert math.isfinite(float(out["coarse_loss"].detach())) and float(out["coarse_loss"].detach()) > 0.0
    out["coarse_loss"].backward()
    assert target.grad is None
    # Online encoder hides every coarse member before context encoding.
    raw = torch.randn(batch_size, patches, 6, 32)
    pad = torch.zeros(batch_size, patches, 32, dtype=torch.bool)
    union = torch.zeros(batch_size, patches, dtype=torch.bool)
    union[:, [0, 1, 2, 3, 4, 5, 6, 7]] = True
    hidden, _, _, _ = model._encode_patches(raw, pad, valid, union)
    shown, _, _, _ = model._encode_patches(raw, pad, valid, torch.zeros_like(union))
    assert (hidden - shown).abs().sum().item() > 0.0
    # Missing scale support is excluded from the denominator, never zero-filled.
    only4 = model.msp_coarse_terms(context, target, valid, weights, [[(0, 1, 2, 3)], []], [[], []])
    assert only4["coarse_loss"].equal(only4["group4_loss"])
    none = model.msp_coarse_terms(context, target, valid, weights, [[], []], [[], []])
    assert float(none["coarse_loss"]) == 0.0
    criterion = MultiscaleLatentPredictionCriterion()
    grouped = criterion(context.detach(), target.detach(), valid, weights, [[(0, 1, 2, 3)], []], [[], []])
    assert math.isfinite(float(grouped["coarse_loss"]))


def test_msp_coarse_groups_use_only_their_own_file_and_valid_weight_members() -> None:
    """A group is evaluated against its own file only, weighted by valid timesteps."""
    config = V1Config(
        n_channels=6, patch_size=32, stride=16, n_robots=9, n_programs=8,
        seed=181801, contrastive_weight_max=0.05, ema_decay=0.996, dropout=0.0,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    model.train()
    batch_size, patches, width = 2, 12, config.d_model
    torch.manual_seed(3)
    context = torch.randn(batch_size, patches, width)
    target = torch.randn(batch_size, patches, width)
    valid = torch.ones(batch_size, patches, dtype=torch.bool)
    weights = torch.ones(batch_size, patches)
    groups_4 = [[(0, 1, 2, 3)], []]
    baseline = model.msp_coarse_terms(context, target, valid, weights, groups_4, [[], []])
    changed_context = context.clone()
    changed_context[1] += 5.0
    changed_target = target.clone()
    changed_target[1] -= 5.0
    after = model.msp_coarse_terms(changed_context, changed_target, valid, weights, groups_4, [[], []])
    assert torch.equal(baseline["group4_loss"], after["group4_loss"])
    # N2 v2 valid_weight_t is the valid-timestep fraction, not a uniform mean.
    weighted = weights.clone()
    weighted[0, 0] = 0.25
    weighted_loss = model.msp_coarse_terms(context, target, valid, weighted, groups_4, [[], []])
    assert not torch.equal(baseline["group4_loss"], weighted_loss["group4_loss"])
    all_zero = weights.clone()
    all_zero[0, 0:4] = 0.0
    with pytest.raises(ValueError, match="positive total valid weight"):
        model.msp_coarse_terms(context, target, valid, all_zero, groups_4, [[], []])
    with pytest.raises(ValueError, match="one entry per batch row"):
        model.msp_coarse_terms(context, target, valid, weights, [(0, 1, 2, 3)], [[], []])


def _canonical_sha256_for_test() -> str:
    from representation.sprint18_checkpoint import _canonical_sha256

    binding = load_task6_binding(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "experiments"
        / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    return _canonical_sha256(sorted(item.manifest_sha256 for item in binding.role_histories("FIT")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for N2 union device-placement proof")
def test_msp_epoch_with_cpu_inputs_runs_on_cuda_batch_device() -> None:
    """Consumer device contract: CPU loader inputs train end to end on a CUDA N2 trainer."""
    trainer, arm, batch = _msp_fixture()
    assert batch["mask"].device.type == "cpu"
    trainer.device = torch.device("cuda")
    trainer.model.to(trainer.device)
    assert trainer.device.type == "cuda" and next(trainer.model.parameters()).is_cuda
    fit_role = _canonical_sha256_for_test()
    step_before = trainer.step
    params_before = state_sha256(trainer.model.state_dict())
    result = trainer.train_sprint18_msp_epoch([batch], arm=arm, epoch=11, fit_role_sha256=fit_role)
    assert trainer.step == step_before + 1
    assert state_sha256(trainer.model.state_dict()) != params_before
    assert math.isfinite(result["joint_loss"]) and math.isfinite(result["coarse_loss"])
    assert result["coarse_loss"] > 0.0
    balance = trainer.objective_balance_history[-1]
    assert balance["epoch"] == 11
    assert balance["support_coarse_g4"] > 0 and balance["support_coarse_g8"] > 0
    assert math.isfinite(balance["r_aux"])


def _msp_fixture(coefficient: float = 0.25, seed: int = 181801, lengths: tuple[int, ...] = (800, 1000, 1200, 1600, 900, 1100, 1300, 1500)):
    config = V1Config(
        n_channels=6, patch_size=32, stride=16, n_robots=9, n_programs=8,
        seed=seed, contrastive_weight_max=0.05, ema_decay=0.996,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    rng = np.random.default_rng(91810)
    samples = []
    for index, length in enumerate(lengths):
        signal = rng.normal(size=(6, length)).astype(np.float32)
        sample = FileSample(
            x=signal, file_id=f"msp-behavior-{index:02d}", file_label=SampleLabel.NORMAL,
            seed=seed + index, generator_version="task11-test-fixture",
            config_hash="task11-test-fixture", regime_sequence=[],
            robot_idx=index % 3, program_idx=index % 2,
        )
        sample.validate()
        samples.append(sample)
    batch = collate_variable_files(samples, patchifier, masking_config=config, masking_seed=91810)
    arm = make_sprint18_arm_config(
        "N2-MSP", seed,
        {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": coefficient, "beta": None},
        selection_state="coefficient_pilot",
    )
    model = V1RepresentationModel(config, patchifier=patchifier)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.0)
    trainer = RepresentationTrainer(model, optimizer=optimizer, max_grad_norm=1.0, seed=seed)
    return trainer, arm, batch


def test_n2_pilot_epoch_coarse_activates_after_warmup_with_infonce() -> None:
    from representation.sprint18_checkpoint import _canonical_sha256 as canonical

    binding = load_task6_binding(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "experiments"
        / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    fit_role = canonical(sorted(item.manifest_sha256 for item in binding.role_histories("FIT")))
    trainer, arm, batch = _msp_fixture()
    warmup = trainer.train_sprint18_msp_epoch([batch], arm=arm, epoch=1, fit_role_sha256=fit_role)
    first = trainer.objective_balance_history[-1]
    assert first["effective_coefficient_L_coarse"] == 0.0
    assert first["raw_L_coarse"] > 0.0
    assert first["weighted_L_coarse"] == pytest.approx(0.0)
    assert first["support_coarse_g4"] > 0 and first["support_coarse_g8"] > 0
    assert first["effective_coefficient_L_info"] == 0.0
    assert first["raw_L_info"] > 0.0
    active = trainer.train_sprint18_msp_epoch([batch], arm=arm, epoch=11, fit_role_sha256=fit_role)
    second = trainer.objective_balance_history[-1]
    assert second["effective_coefficient_L_coarse"] == pytest.approx(0.25 / 20.0)
    assert second["raw_L_coarse"] > 0.0
    assert second["weighted_L_coarse"] == pytest.approx(
        second["effective_coefficient_L_coarse"] * second["raw_L_coarse"]
    )
    assert second["effective_coefficient_L_info"] == pytest.approx(0.05 / 20.0)
    assert second["r_aux"] == pytest.approx(
        second["gradient_norm_weighted_L_coarse_shared"]
        / (second["gradient_norm_L_pred_shared"] + 1e-12)
    )
    assert second["lr"] == pytest.approx(learning_rate_for_epoch(11))
    assert active["epoch"] == 11.0
    with pytest.raises(ValueError, match="epoch-30"):
        trainer.train_sprint18_msp_epoch([batch], arm=arm, epoch=31, fit_role_sha256=fit_role)


def test_n2_batches_match_b0_sampler_scope() -> None:
    binding = load_task6_binding(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "experiments"
        / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    fit_history_ids = tuple(item.actual_history_id for item in binding.role_histories("FIT"))
    fit_ids = tuple(f"fit-healthy-{index:05d}" for index in range(binding.fit_healthy_rows))
    b0 = make_sprint18_arm_config(
        "B0-FULL", MODEL_SEEDS[0],
        {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="coefficient_pilot",
    )
    n2 = make_sprint18_arm_config(
        "N2-MSP", MODEL_SEEDS[0],
        {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": 0.25, "beta": None},
        selection_state="coefficient_pilot",
    )
    for epoch in (1, 15, 30):
        assert deterministic_sampler_order(
            binding, b0, workload="representation", scope="coefficient_pilot",
            role_history_ids=fit_history_ids, epoch=epoch, eligible_source_ids=fit_ids,
        ) == deterministic_sampler_order(
            binding, n2, workload="representation", scope="coefficient_pilot",
            role_history_ids=fit_history_ids, epoch=epoch, eligible_source_ids=fit_ids,
        )


def test_msp_selector_applies_frozen_grid_and_balance_gates(tmp_path=None) -> None:
    del tmp_path
    binding = load_task6_binding(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "experiments"
        / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    design_ids = tuple(item.actual_history_id for item in binding.role_histories("DESIGN"))

    def design(loss: float, spread: float = 0.1):
        return {
            history_id: {
                "support_sha256": "a" * 64,
                "support_files": 13,
                "healthy_embedding_std_median": spread,
                "mean_healthy_fine_prediction_loss": loss,
            }
            for history_id in design_ids
        }

    prediction_only = {seed: {"design_histories": design(seed, 1.0)} for seed in MODEL_SEEDS}

    def records(coefficient: float, ratios):
        batches_per_epoch = (binding.fit_healthy_rows + 128 - 1) // 128
        observations = [
            (epoch, step)
            for epoch in range(26, 31)
            for step in range((epoch - 1) * batches_per_epoch + 1, epoch * batches_per_epoch + 1)
            if step % 40 == 0
        ]
        assert len(ratios) == len(observations)
        by_step = dict(zip(observations, ratios))
        rows = []
        for epoch in range(1, TASK9_COEFFICIENT_PILOT_EPOCHS + 1):
            first_step = (epoch - 1) * batches_per_epoch + 1
            steps = {first_step} | {step for recorded_epoch, step in observations if recorded_epoch == epoch}
            for step in sorted(steps):
                observed = (epoch, step) in by_step
                ratio = by_step.get((epoch, step), 0.2)
                effective = coefficient_for_epoch(coefficient, epoch)
                rows.append(
                    {
                        "epoch": epoch,
                        "global_step": step,
                        "lr": learning_rate_for_epoch(epoch),
                        "raw_L_pred": 1.0,
                        "effective_coefficient_L_pred": 1.0,
                        "weighted_L_pred": 1.0,
                        "raw_L_coarse": 2.0,
                        "effective_coefficient_L_coarse": effective,
                        "weighted_L_coarse": 2.0 * effective,
                        "raw_L_info": 1.5,
                        "effective_coefficient_L_info": coefficient_for_epoch(0.05, epoch),
                        "weighted_L_info": 1.5 * coefficient_for_epoch(0.05, epoch),
                        "gradient_norm_L_pred_shared": 1.0,
                        "gradient_norm_weighted_L_coarse_shared": ratio,
                        "r_aux": ratio,
                        "embedding_support_files": 128,
                        "support_valid_files": 128,
                        "support_valid_patches": 128 * 12,
                        "support_prediction_files": 128,
                        "support_prediction_patches": 128 * 4,
                        "support_coarse_g4": 6,
                        "support_coarse_g8": 3,
                        "contrastive_input_files": 128,
                        "embedding_variance_mean": 0.1,
                        "embedding_std_median": 0.1,
                        "embedding_effective_rank": 3.0,
                        "protocol_gradient_observation": observed,
                    }
                )
        return rows

    candidate_runs = {
        0.10: {seed: {"balance_records": records(0.10, (0.05,) * 6), "design_histories": design(1.0)} for seed in MODEL_SEEDS},
        0.25: {seed: {"balance_records": records(0.25, (0.2,) * 6), "design_histories": design(1.0)} for seed in MODEL_SEEDS},
        0.50: {seed: {"balance_records": records(0.50, (0.9,) * 6), "design_histories": design(1.0)} for seed in MODEL_SEEDS},
    }
    selection = select_msp_coefficient(binding, candidate_runs, prediction_only)
    assert selection["grid"] == list(N2_AUX_COEFFICIENT_GRID) == list(MSP_AUX_COEFFICIENT_GRID)
    assert selection["selected_coefficient"] == 0.25
    verdicts = {row["coefficient"]: row for row in selection["candidate_verdicts"]}
    assert verdicts[0.10]["valid"] is False
    assert verdicts[0.25]["valid"] is True
    assert verdicts[0.50]["valid"] is False
    with pytest.raises(ValueError, match="frozen MSP grid"):
        select_msp_coefficient(binding, {0.0: {}}, prediction_only)


def test_n2_coherent_checkpoint_restore_preserves_trained_inference(tmp_path) -> None:
    """Prove save -> strict restore -> public inference on the real updated N2 path."""
    binding = load_task6_binding(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "experiments"
        / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    coefficient = 0.25
    seed = MODEL_SEEDS[0]
    trainer, arm, batch = _msp_fixture(coefficient=coefficient, seed=seed)
    model = trainer.model
    optimizer = trainer.optimizer
    for group in optimizer.param_groups:
        group["lr"] = learning_rate_for_epoch(1)
    initialization_sha256 = state_sha256(model.state_dict())
    batches_per_epoch = (binding.fit_healthy_rows + BATCH_SIZE - 1) // BATCH_SIZE
    assert batches_per_epoch == 43
    fit_role = _canonical_sha256(sorted(item.manifest_sha256 for item in binding.role_histories("FIT")))
    for _ in range(batches_per_epoch):
        trainer.train_sprint18_msp_epoch([batch], arm=arm, epoch=1, fit_role_sha256=fit_role)
    assert trainer.step == batches_per_epoch
    assert state_sha256(model.state_dict()) != initialization_sha256
    balance = trainer.objective_balance_history[-1]
    assert balance["epoch"] == 1
    assert balance["support_coarse_g4"] > 0 and balance["support_coarse_g8"] > 0
    assert math.isfinite(balance["r_aux"])
    for group in optimizer.param_groups:
        group["lr"] = learning_rate_for_epoch(2)
    fit_history_ids = tuple(item.actual_history_id for item in binding.role_histories("FIT"))
    fit_ids = tuple(f"fit-healthy-{index:05d}" for index in range(binding.fit_healthy_rows))
    sampler_seed, epoch_order = deterministic_sampler_order(
        binding, arm, workload="representation", scope="coefficient_pilot",
        role_history_ids=fit_history_ids, epoch=2, eligible_source_ids=fit_ids,
    )
    ordered_ids = sorted(fit_ids)
    sampler_state = {
        "scope": "coefficient_pilot",
        "workload": "representation",
        "role_history_ids": list(fit_history_ids),
        "role_manifest_sha256": [binding.history(history_id).manifest_sha256 for history_id in fit_history_ids],
        "eligible_source_ids": ordered_ids,
        "eligible_source_ids_sha256": _canonical_sha256(ordered_ids),
        "sampler_epoch_seed": sampler_seed,
        "epoch_order_ids": list(epoch_order),
        "epoch_order_ids_sha256": _canonical_sha256(list(epoch_order)),
        "eligible_rows": len(ordered_ids),
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": batches_per_epoch,
        "completed_epochs": 1,
        "current_epoch": 2,
        "batches_in_current_epoch": 0,
        "fit_role_sha256": _canonical_sha256(
            sorted(item.manifest_sha256 for item in binding.role_histories("FIT"))
        ),
    }
    coefficient_state = {
        "epoch": 2,
        "global_step": trainer.step,
        "selection_state": "coefficient_pilot",
        "lambda_info": coefficient_for_epoch(0.05, 2),
        "lambda_hcc": None,
        "alpha": coefficient,
        "beta": None,
    }
    checkpoint_path = tmp_path / "n2-msp-coherent-pilot.pt"
    digest = save_sprint18_checkpoint(
        checkpoint_path, model, optimizer, arm=arm, binding=binding,
        global_step=trainer.step, completed_epoch=1,
        initialization_sha256=initialization_sha256,
        scheduler_state={
            "scheduler_id": "sprint18-epoch-lr-v2",
            "completed_epochs": 1,
            "active_epoch": 2,
            "learning_rate": learning_rate_for_epoch(2),
            "global_step": trainer.step,
        },
        sampler_state=sampler_state, coefficient_state=coefficient_state,
        scope="coefficient_pilot",
    )
    saved_sha256 = state_sha256(model.state_dict())
    saved_ema_sha256 = state_sha256(model.target_encoder.state_dict())
    assert saved_sha256 != initialization_sha256
    model.eval()
    clean = {key: value for key, value in batch.items() if key != "file_samples"}
    with torch.no_grad():
        trained_embeddings = model(trainer._move_batch(clean))["file_embedding"]
    assert torch.isfinite(trained_embeddings).all()
    trained_bank = NormalReferenceBank(k=model.config.knn_k).fit(
        trained_embeddings.detach().cpu(), [SampleLabel.NORMAL] * trained_embeddings.shape[0]
    )
    trained_scores = RepresentationInference(
        model, trained_bank, model.patchifier, masking_config=model.config
    ).score_batch(clean)
    assert torch.isfinite(trained_scores["S_pred"]).all()
    assert torch.isfinite(trained_scores["S_pop"]).all()
    restored_model = V1RepresentationModel(model.config, patchifier=model.patchifier)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=learning_rate_for_epoch(2), weight_decay=0.0
    )
    restored = restore_sprint18_checkpoint(
        checkpoint_path, restored_model, restored_optimizer,
        arm=arm, binding=binding, allowed_scope="coefficient_pilot",
    )
    assert restored.checkpoint_sha256 == digest
    assert restored.checkpoint_kind == "coefficient_pilot" and restored.scope == "coefficient_pilot"
    assert restored.arm_id == "N2-MSP" and restored.model_seed == seed
    assert restored.global_step == trainer.step == batches_per_epoch
    assert restored.completed_epoch == 1 and restored.representation_step == trainer.step
    assert restored.model_state_sha256 == saved_sha256
    assert restored.ema_state_sha256 == saved_ema_sha256
    assert state_sha256(restored_model.state_dict()) == saved_sha256
    assert state_sha256(restored_model.target_encoder.state_dict()) == saved_ema_sha256
    restored_model.eval()
    with torch.no_grad():
        restored_embeddings = restored_model(trainer._move_batch(clean))["file_embedding"]
    restored_bank = NormalReferenceBank(k=restored_model.config.knn_k).fit(
        restored_embeddings.detach().cpu(), [SampleLabel.NORMAL] * restored_embeddings.shape[0]
    )
    restored_scores = RepresentationInference(
        restored_model, restored_bank, restored_model.patchifier,
        masking_config=restored_model.config,
    ).score_batch(clean)
    assert torch.isfinite(restored_scores["S_pred"]).all()
    assert torch.isfinite(restored_scores["S_pop"]).all()
    assert torch.equal(restored_scores["S_pred"], trained_scores["S_pred"])
    assert torch.allclose(restored_scores["S_pop"], trained_scores["S_pop"], atol=1e-6, rtol=0.0)
    assert float(restored_scores["S_pred"].max() - restored_scores["S_pred"].min()) > 0.0
    assert float(restored_scores["S_pop"].max() - restored_scores["S_pop"].min()) >= 0.0
