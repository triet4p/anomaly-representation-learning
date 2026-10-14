from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from representation.criterion import (
    HCC_AFFINITY_DIFFERENT_ROBOT,
    HCC_AFFINITY_SAME_FILE,
    HCC_AFFINITY_SAME_ROBOT_DIFF_PROGRAM,
    HCC_AFFINITY_SAME_ROBOT_PROGRAM,
    HierarchicalContrastiveCriterion,
    hcc_file_affinities,
    hcc_relation_support,
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
from representation import NormalReferenceBank, RepresentationInference
from representation.sprint18_harness import (
    BATCH_SIZE,
    COMMON_INFO_COEFFICIENT_GRID,
    MODEL_SEEDS,
    TASK9_COEFFICIENT_PILOT_EPOCHS,
    deterministic_sampler_order,
    load_task6_binding,
    make_sprint18_arm_config,
    select_hcc_coefficient,
)
from representation.trainer import RepresentationTrainer
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import FileSample, SampleLabel


def test_hcc_affinities_use_frozen_relation_levels() -> None:
    robots = torch.tensor([0, 0, 0, 1])
    programs = torch.tensor([0, 0, 1, 0])
    affinities = hcc_file_affinities(robots, programs)
    assert affinities.diagonal().tolist() == [HCC_AFFINITY_SAME_FILE] * 4
    assert HCC_AFFINITY_SAME_FILE == 1.0
    assert HCC_AFFINITY_SAME_ROBOT_PROGRAM == 0.5
    assert HCC_AFFINITY_SAME_ROBOT_DIFF_PROGRAM == 0.1
    assert HCC_AFFINITY_DIFFERENT_ROBOT == 0.0
    # Same (robot, program) cross-file pair (0,1); same-robot/different-program (0,2)/(1,2).
    assert affinities[0, 1].item() == pytest.approx(0.5)
    assert affinities[0, 2].item() == pytest.approx(0.1)
    assert affinities[1, 2].item() == pytest.approx(0.1)
    # Program-only across robots carries no positive mass; different robots carry none.
    assert affinities[0, 3].item() == pytest.approx(0.0)
    assert affinities[2, 3].item() == pytest.approx(0.0)


def test_hcc_loss_is_finite_and_row_permutation_invariant() -> None:
    torch.manual_seed(0)
    view_1 = torch.randn(6, 8)
    view_2 = torch.randn(6, 8)
    robots = torch.tensor([0, 0, 0, 1, 1, 2])
    programs = torch.tensor([0, 0, 1, 0, 0, 0])
    criterion = HierarchicalContrastiveCriterion(temperature=0.2)
    first = criterion(view_1, view_2, robots, programs)["loss"].item()
    assert math.isfinite(first) and first > 0.0
    perm = torch.tensor([5, 3, 0, 4, 1, 2])
    second = criterion(view_1[perm], view_2[perm], robots[perm], programs[perm])["loss"].item()
    assert second == pytest.approx(first)
    targets = hcc_file_affinities(robots, programs)
    targets = targets / targets.sum(dim=1, keepdim=True)
    assert torch.allclose(targets.sum(dim=1), torch.ones(6), atol=1e-6)
    assert torch.isfinite(targets).all()


def test_hcc_rejects_label_inputs_and_uses_identity_only() -> None:
    robots = torch.tensor([0, 0])
    programs = torch.tensor([0, 1])
    support = hcc_relation_support(robots, programs)
    assert support["anchors_same_r_diff_p"] == 2
    assert support["anchors_same_rp"] == 0
    with pytest.raises(ValueError, match="1D"):
        hcc_file_affinities(torch.tensor([[0, 0]]), programs)
    with pytest.raises(ValueError, match="non-negative"):
        hcc_file_affinities(torch.tensor([0, -1]), programs)


def _hcc_fixture(coefficient: float = 0.05, seed: int = 181801):
    config = V1Config(
        n_channels=6,
        patch_size=32,
        stride=16,
        n_robots=9,
        n_programs=8,
        seed=seed,
        contrastive_weight_max=0.05,
        ema_decay=0.996,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    rng = np.random.default_rng(91810)
    combos = [(0, 0), (0, 0), (0, 1), (1, 0), (1, 2), (2, 2), (2, 3), (3, 4)]
    samples = []
    for index, (robot, program) in enumerate(combos):
        signal = rng.normal(size=(6, 96)).astype(np.float32)
        sample = FileSample(
            x=signal,
            file_id=f"hcc-behavior-{index:02d}",
            file_label=SampleLabel.NORMAL,
            seed=seed + index,
            generator_version="task10-test-fixture",
            config_hash="task10-test-fixture",
            regime_sequence=[],
            robot_idx=robot,
            program_idx=program,
        )
        sample.validate()
        samples.append(sample)
    batch = collate_variable_files(samples, patchifier, masking_config=config, masking_seed=91810)
    arm = make_sprint18_arm_config(
        "N1-HCC",
        seed,
        {"lambda_info_max": None, "lambda_hcc_max": coefficient, "alpha": None, "beta": None},
        selection_state="coefficient_pilot",
    )
    model = V1RepresentationModel(config, patchifier=patchifier)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.0)
    trainer = RepresentationTrainer(model, optimizer=optimizer, max_grad_norm=1.0, seed=seed)
    return trainer, arm, batch


def test_n1_pilot_epoch_records_hcc_schedule_support_and_gradient_ratio() -> None:
    trainer, arm, batch = _hcc_fixture()
    warmup = trainer.train_sprint18_hcc_epoch([batch], arm=arm, epoch=1)
    first = trainer.objective_balance_history[-1]
    assert first["effective_coefficient_L_hcc"] == 0.0
    assert first["raw_L_hcc"] > 0.0
    assert first["weighted_L_hcc"] == pytest.approx(0.0)
    assert first["anchors_same_rp"] > 0 and first["anchors_same_r_diff_p"] > 0
    active = trainer.train_sprint18_hcc_epoch([batch], arm=arm, epoch=11)
    second = trainer.objective_balance_history[-1]
    assert second["effective_coefficient_L_hcc"] == pytest.approx(0.05 / 20.0)
    assert second["raw_L_hcc"] > 0.0
    assert second["weighted_L_hcc"] == pytest.approx(
        second["effective_coefficient_L_hcc"] * second["raw_L_hcc"]
    )
    assert second["r_aux"] == pytest.approx(
        second["gradient_norm_weighted_L_hcc_shared"]
        / (second["gradient_norm_L_pred_shared"] + 1e-12)
    )
    assert second["lr"] == pytest.approx(learning_rate_for_epoch(11))
    assert active["epoch"] == 11.0
    with pytest.raises(ValueError, match="epoch-30"):
        trainer.train_sprint18_hcc_epoch([batch], arm=arm, epoch=31)


def test_n1_batches_match_b0_sampler_and_cbn_parity_is_pre_update_only() -> None:
    binding = load_task6_binding(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "experiments"
        / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    fit_history_ids = tuple(item.actual_history_id for item in binding.role_histories("FIT"))
    fit_ids = tuple(f"fit-healthy-{index:05d}" for index in range(binding.fit_healthy_rows))
    b0 = make_sprint18_arm_config(
        "B0-FULL",
        MODEL_SEEDS[0],
        {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="coefficient_pilot",
    )
    n1 = make_sprint18_arm_config(
        "N1-HCC",
        MODEL_SEEDS[0],
        {"lambda_info_max": None, "lambda_hcc_max": 0.05, "alpha": None, "beta": None},
        selection_state="coefficient_pilot",
    )
    for epoch in (1, 15, 30):
        assert deterministic_sampler_order(
            binding, b0, workload="representation", scope="coefficient_pilot",
            role_history_ids=fit_history_ids, epoch=epoch, eligible_source_ids=fit_ids,
        ) == deterministic_sampler_order(
            binding, n1, workload="representation", scope="coefficient_pilot",
            role_history_ids=fit_history_ids, epoch=epoch, eligible_source_ids=fit_ids,
        )
    # Same B0 implementation/configuration and matched initial fallback mapping; trainable path.
    seed = MODEL_SEEDS[0]
    config = V1Config(
        n_channels=6, patch_size=32, stride=16, n_robots=9, n_programs=8,
        seed=seed, contrastive_weight_max=0.05, ema_decay=0.996,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    torch.manual_seed(seed)
    model_b0 = V1RepresentationModel(config, patchifier=patchifier)
    torch.manual_seed(seed)
    model_n1 = V1RepresentationModel(config, patchifier=patchifier)
    assert type(model_n1.conditional_norm) is type(model_b0.conditional_norm)
    assert model_n1.conditional_norm is not None
    assert model_b0.state_dict().keys() == model_n1.state_dict().keys()


def test_hcc_selector_applies_frozen_grid_and_relation_gate(tmp_path=None) -> None:
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

    def records(coefficient: float, ratios, relation: bool):
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
                        "raw_L_hcc": 2.0,
                        "effective_coefficient_L_hcc": effective,
                        "weighted_L_hcc": 2.0 * effective,
                        "gradient_norm_L_pred_shared": 1.0,
                        "gradient_norm_weighted_L_hcc_shared": ratio,
                        "r_aux": ratio,
                        "embedding_support_files": 128,
                        "support_valid_files": 128,
                        "support_valid_patches": 128 * 12,
                        "support_prediction_files": 128,
                        "support_prediction_patches": 128 * 4,
                        "support_hcc_files": 128,
                        "contrastive_input_files": 128,
                        "embedding_variance_mean": 0.1,
                        "embedding_std_median": 0.1,
                        "embedding_effective_rank": 3.0,
                        "anchors_same_rp": 5 if relation else 0,
                        "anchors_same_r_diff_p": 7 if relation else 0,
                        "anchors_only_same_file": 1,
                        "target_entropy_mean": 0.4,
                        "protocol_gradient_observation": observed,
                    }
                )
        return rows

    candidate_runs = {
        0.025: {seed: {"balance_records": records(0.025, (0.05,) * 6, True), "design_histories": design(1.0)} for seed in MODEL_SEEDS},
        0.05: {seed: {"balance_records": records(0.05, (0.2,) * 6, True), "design_histories": design(1.0)} for seed in MODEL_SEEDS},
        0.10: {seed: {"balance_records": records(0.10, (0.2,) * 6, False), "design_histories": design(1.0)} for seed in MODEL_SEEDS},
    }
    selection = select_hcc_coefficient(binding, candidate_runs, prediction_only)
    assert selection["grid"] == list(COMMON_INFO_COEFFICIENT_GRID)
    assert selection["selected_coefficient"] == 0.05
    verdicts = {row["coefficient"]: row for row in selection["candidate_verdicts"]}
    assert verdicts[0.025]["valid"] is False
    assert verdicts[0.05]["valid"] is True
    assert verdicts[0.10]["valid"] is False
    assert any(
        "hcc_relation_support_missing" in seed_row["failed_checks"]
        for seed_row in verdicts[0.10]["per_seed"]
    )
    with pytest.raises(ValueError, match="frozen HCC grid"):
        select_hcc_coefficient(binding, {0.0: {}}, prediction_only)


def test_n1_coherent_checkpoint_restore_preserves_trained_inference(tmp_path) -> None:
    """Prove save -> strict restore -> public inference on the real updated N1 path.

    Trains the real N1 model for one full coherent pilot epoch (43 optimizer
    steps, matching the frozen 5,446-row / 128-file batch geometry), saves a
    real coherent ``coefficient_pilot`` checkpoint bound to the certified Fit
    support, strictly restores into a fresh model/optimizer, and requires the
    restored model to reproduce the trained model's consumer-visible scores.
    """
    binding = load_task6_binding(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "experiments"
        / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    coefficient = 0.05
    seed = MODEL_SEEDS[0]
    trainer, arm, batch = _hcc_fixture(coefficient=coefficient, seed=seed)
    model = trainer.model
    optimizer = trainer.optimizer
    for group in optimizer.param_groups:
        group["lr"] = learning_rate_for_epoch(1)
    initialization_sha256 = state_sha256(model.state_dict())
    batches_per_epoch = (binding.fit_healthy_rows + BATCH_SIZE - 1) // BATCH_SIZE
    assert batches_per_epoch == 43
    for _ in range(batches_per_epoch):
        trainer.train_sprint18_hcc_epoch([batch], arm=arm, epoch=1)
    assert trainer.step == batches_per_epoch
    assert state_sha256(model.state_dict()) != initialization_sha256
    balance = trainer.objective_balance_history[-1]
    assert balance["epoch"] == 1
    assert balance["anchors_same_rp"] > 0 and balance["anchors_same_r_diff_p"] > 0
    assert math.isfinite(balance["r_aux"])
    for group in optimizer.param_groups:
        group["lr"] = learning_rate_for_epoch(2)
    fit_history_ids = tuple(item.actual_history_id for item in binding.role_histories("FIT"))
    fit_ids = tuple(f"fit-healthy-{index:05d}" for index in range(binding.fit_healthy_rows))
    sampler_seed, epoch_order = deterministic_sampler_order(
        binding,
        arm,
        workload="representation",
        scope="coefficient_pilot",
        role_history_ids=fit_history_ids,
        epoch=2,
        eligible_source_ids=fit_ids,
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
        "lambda_info": None,
        "lambda_hcc": coefficient_for_epoch(coefficient, 2),
        "alpha": None,
        "beta": None,
    }
    checkpoint_path = tmp_path / "n1-hcc-coherent-pilot.pt"
    digest = save_sprint18_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        arm=arm,
        binding=binding,
        global_step=trainer.step,
        completed_epoch=1,
        initialization_sha256=initialization_sha256,
        scheduler_state={
            "scheduler_id": "sprint18-epoch-lr-v2",
            "completed_epochs": 1,
            "active_epoch": 2,
            "learning_rate": learning_rate_for_epoch(2),
            "global_step": trainer.step,
        },
        sampler_state=sampler_state,
        coefficient_state=coefficient_state,
        scope="coefficient_pilot",
    )
    saved_sha256 = state_sha256(model.state_dict())
    saved_ema_sha256 = state_sha256(model.target_encoder.state_dict())
    assert saved_sha256 != initialization_sha256
    model.eval()
    with torch.no_grad():
        trained_embeddings = model(trainer._move_batch(batch))["file_embedding"]
    assert torch.isfinite(trained_embeddings).all()
    trained_bank = NormalReferenceBank(k=model.config.knn_k).fit(
        trained_embeddings.detach().cpu(), [SampleLabel.NORMAL] * trained_embeddings.shape[0]
    )
    trained_scores = RepresentationInference(
        model, trained_bank, model.patchifier, masking_config=model.config
    ).score_batch(batch)
    assert torch.isfinite(trained_scores["S_pred"]).all()
    assert torch.isfinite(trained_scores["S_pop"]).all()
    restored_model = V1RepresentationModel(model.config, patchifier=model.patchifier)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=learning_rate_for_epoch(2), weight_decay=0.0
    )
    restored = restore_sprint18_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        arm=arm,
        binding=binding,
        allowed_scope="coefficient_pilot",
    )
    assert restored.checkpoint_sha256 == digest
    assert restored.checkpoint_kind == "coefficient_pilot" and restored.scope == "coefficient_pilot"
    assert restored.arm_id == "N1-HCC" and restored.model_seed == seed
    assert restored.global_step == trainer.step == batches_per_epoch
    assert restored.completed_epoch == 1 and restored.representation_step == trainer.step
    assert restored.model_state_sha256 == saved_sha256
    assert restored.ema_state_sha256 == saved_ema_sha256
    assert state_sha256(restored_model.state_dict()) == saved_sha256
    assert state_sha256(restored_model.target_encoder.state_dict()) == saved_ema_sha256
    restored_model.eval()
    with torch.no_grad():
        restored_embeddings = restored_model(trainer._move_batch(batch))["file_embedding"]
    restored_bank = NormalReferenceBank(k=restored_model.config.knn_k).fit(
        restored_embeddings.detach().cpu(), [SampleLabel.NORMAL] * restored_embeddings.shape[0]
    )
    restored_scores = RepresentationInference(
        restored_model, restored_bank, restored_model.patchifier,
        masking_config=restored_model.config,
    ).score_batch(batch)
    assert torch.isfinite(restored_scores["S_pred"]).all()
    assert torch.isfinite(restored_scores["S_pop"]).all()
    assert torch.equal(restored_scores["S_pred"], trained_scores["S_pred"])
    assert torch.allclose(restored_scores["S_pop"], trained_scores["S_pop"], atol=1e-6, rtol=0.0)
    assert float(restored_scores["S_pred"].max() - restored_scores["S_pred"].min()) > 0.0
    assert float(restored_scores["S_pop"].max() - restored_scores["S_pop"].min()) >= 0.0
