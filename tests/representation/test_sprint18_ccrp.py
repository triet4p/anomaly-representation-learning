from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from representation import NormalReferenceBank, RepresentationInference
from representation.criterion import (
    CCRP_AUX_COEFFICIENT_GRID,
    CrossChannelRelationalCriterion,
    ccrp_assign_channels,
    ccrp_patch_count,
    ccrp_patch_seed,
    ccrp_sample_targets,
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
    N3_AUX_COEFFICIENT_GRID,
    TASK9_COEFFICIENT_PILOT_EPOCHS,
    deterministic_sampler_order,
    load_task6_binding,
    make_sprint18_arm_config,
    select_ccrp_coefficient,
)
from representation.trainer import RepresentationTrainer
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import FileSample, SampleLabel


def test_ccrp_patch_count_matches_quarter_rounding() -> None:
    for n in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 31, 43, 49, 100):
        expected = 0 if n == 0 else min(n, max(1, int(round(0.25 * n))))
        assert ccrp_patch_count(n) == expected
    assert ccrp_patch_count(0) == 0
    # Ties-to-even rounding: round(0.25*6)=round(1.5)=2, round(0.25*14)=round(3.5)=4.
    assert ccrp_patch_count(6) == 2
    assert ccrp_patch_count(14) == 4
    with pytest.raises(ValueError, match="non-negative integer"):
        ccrp_patch_count(-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        ccrp_patch_count(True)


def test_ccrp_sampling_is_seeded_and_resamples_masked_in_order() -> None:
    valid = list(range(16))
    seed = ccrp_patch_seed(181801, 3, "b" * 64, "ccrp-file-007")
    first = ccrp_sample_targets(valid, 4, np.random.default_rng(seed), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15][:0])
    second = ccrp_sample_targets(valid, 4, np.random.default_rng(seed), [])
    assert first["selected"] == second["selected"]
    assert len(first["selected"]) == 4
    # Masked initials are rejected and replaced in initial-sample order.
    masked = list(first["initial"][:2])
    resampled = ccrp_sample_targets(valid, 4, np.random.default_rng(seed), masked)
    assert resampled["rejected"] == list(first["initial"][:2])
    assert len(resampled["selected"]) == 4
    assert not (set(resampled["selected"]) & set(masked))
    # Final count equals min(Q, unmasked valid count) when replacements run out.
    starved = ccrp_sample_targets(valid, 4, np.random.default_rng(seed), valid[:15])
    assert len(starved["selected"]) == 1
    with pytest.raises(ValueError, match="seed"):
        ccrp_patch_seed(181801, 3, "not-a-hash", "ccrp-file-007")


def test_ccrp_channel_assignment_is_epoch_global_round_robin() -> None:
    keys = [(f"ccrp-file-{i:02d}", p) for i in range(4) for p in (0, 3)]
    assignment = ccrp_assign_channels(keys, 181801, 5)
    assert assignment["total"] == 8
    counts = assignment["counts"]
    assert max(counts) - min(counts) <= 1
    assert len(assignment["channels"]) == 8
    expected = [(ordinal + 181801 + 5) % 6 for ordinal in range(8)]
    assert list(assignment["channels"]) == expected
    with pytest.raises(ValueError, match="epoch-global sorted order"):
        ccrp_assign_channels(list(reversed(keys)), 181801, 5)


def test_ccrp_heldout_channel_isolated_from_student_input() -> None:
    config = V1Config(
        n_channels=6, patch_size=32, stride=16, n_robots=9, n_programs=8,
        seed=181801, contrastive_weight_max=0.05, ema_decay=0.996, dropout=0.0,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    model.train()
    torch.manual_seed(0)
    context = torch.randn(3, 128)
    full = torch.randn(3, 128)
    without = torch.randn(3, 128)
    out = model.ccrp_channel_terms(context, full, without, torch.tensor([0, 1, 5]))
    assert math.isfinite(float(out["channel_loss"].detach())) and float(out["channel_loss"].detach()) > 0.0
    assert int(out["support_channel_targets"].item()) == 3
    # Teacher path is stop-gradient: no gradient reaches the EMA local targets,
    # while the student prediction path carries real gradients.
    full_g = torch.randn(3, 128, requires_grad=True)
    without_g = torch.randn(3, 128, requires_grad=True)
    student_c = torch.randn(3, 128, requires_grad=True)
    out_g = model.ccrp_channel_terms(student_c, full_g, without_g, [0, 1, 2])
    assert math.isfinite(float(out_g["channel_loss"].detach()))
    out_g["channel_loss"].backward()
    assert full_g.grad is None
    assert without_g.grad is None
    assert student_c.grad is not None
    assert torch.isfinite(student_c.grad).all()
    same = torch.randn(2, 128)
    criterion = CrossChannelRelationalCriterion()
    with pytest.raises(ValueError, match="no supported targets"):
        criterion(torch.randn(2, 128), same, same.clone())
    probe_full = torch.randn(3, 128)
    probe_without = probe_full.clone()
    probe_without[1] += 1.0
    probe_without[2] -= 1.0
    probe = criterion(torch.randn(3, 128), probe_full, probe_without)
    assert int(probe["support_channel_targets"].item()) == 2


def _ccrp_fixture(coefficient: float = 0.25, seed: int = 181801, lengths: tuple[int, ...] = (800, 1000, 1200, 1600, 900, 1100, 1300, 1500), prefix: str = "ccrp-behavior") -> tuple:
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
            x=signal, file_id=f"{prefix}-{index:02d}", file_label=SampleLabel.NORMAL,
            seed=seed + index, generator_version="task12-test-fixture",
            config_hash="task12-test-fixture", regime_sequence=[],
            robot_idx=index % 3, program_idx=index % 2,
        )
        sample.validate()
        samples.append(sample)
    batch = collate_variable_files(samples, patchifier, masking_config=config, masking_seed=91810)
    arm = make_sprint18_arm_config(
        "N3-CCRP", seed,
        {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": coefficient},
        selection_state="coefficient_pilot",
    )
    model = V1RepresentationModel(config, patchifier=patchifier)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.0)
    trainer = RepresentationTrainer(model, optimizer=optimizer, max_grad_norm=1.0, seed=seed)
    return trainer, arm, batch


def _fit_role_sha256() -> str:
    binding = load_task6_binding(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "experiments"
        / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    return _canonical_sha256(sorted(item.manifest_sha256 for item in binding.role_histories("FIT")))


def test_n3_pilot_epoch_channel_activates_after_warmup_with_infonce() -> None:
    fit_role = _fit_role_sha256()
    trainer, arm, batch = _ccrp_fixture()
    trainer.train_sprint18_ccrp_epoch([batch], arm=arm, epoch=1, fit_role_sha256=fit_role)
    first = trainer.objective_balance_history[-1]
    assert first["effective_coefficient_L_channel"] == 0.0
    assert first["raw_L_channel"] > 0.0
    assert first["weighted_L_channel"] == pytest.approx(0.0)
    assert first["support_channel_targets"] > 0
    assert first["effective_coefficient_L_info"] == 0.0
    assert first["raw_L_info"] > 0.0
    active = trainer.train_sprint18_ccrp_epoch([batch], arm=arm, epoch=11, fit_role_sha256=fit_role)
    second = trainer.objective_balance_history[-1]
    assert second["effective_coefficient_L_channel"] == pytest.approx(0.25 / 20.0)
    assert second["raw_L_channel"] > 0.0
    assert second["weighted_L_channel"] == pytest.approx(
        second["effective_coefficient_L_channel"] * second["raw_L_channel"]
    )
    assert second["effective_coefficient_L_info"] == pytest.approx(0.05 / 20.0)
    assert second["r_aux"] == pytest.approx(
        second["gradient_norm_weighted_L_channel_shared"]
        / (second["gradient_norm_L_pred_shared"] + 1e-12)
    )
    assert second["lr"] == pytest.approx(learning_rate_for_epoch(11))
    assert active["epoch"] == 11.0
    # Epoch-scope support totals are exact on every observation row: the
    # epoch-global round-robin totals reproduce and cohere with the row total.
    per_channel = [second[f"support_channel_c{index}"] for index in range(6)]
    epoch_total = int(second["support_channel_targets"])
    assert sum(per_channel) == epoch_total
    assert max(per_channel) - min(per_channel) <= 1
    assert epoch_total > 0
    with pytest.raises(ValueError, match="epoch-30"):
        trainer.train_sprint18_ccrp_epoch([batch], arm=arm, epoch=31, fit_role_sha256=fit_role)


def test_n3_multibatch_epoch_records_coherent_selector_support() -> None:
    """Run the public trainer epoch over two batches and feed real rows to the selector."""
    fit_role = _fit_role_sha256()
    trainer, arm, batch = _ccrp_fixture()
    _, _, batch_b = _ccrp_fixture(prefix="ccrp-behavior-b")
    trainer.train_sprint18_ccrp_epoch(
        [batch, batch_b], arm=arm, epoch=27, fit_role_sha256=fit_role
    )
    rows = list(trainer.objective_balance_history)
    assert rows, "expected balance rows from the public two-batch epoch"
    for row in rows:
        total = int(row["support_channel_targets"])
        per_channel = [int(row[f"support_channel_c{index}"]) for index in range(6)]
        assert sum(per_channel) == total
        assert max(per_channel) - min(per_channel) <= 1
        assert total > 0
    full_records = list(rows)
    binding = load_task6_binding(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "experiments"
        / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    candidate_runs = {
        0.25: {seed: {"balance_records": full_records, "design_histories": {}} for seed in MODEL_SEEDS},
    }
    prediction_only_runs = {seed: {"design_histories": {}} for seed in MODEL_SEEDS}
    selection = select_ccrp_coefficient(binding, candidate_runs, prediction_only_runs)
    verdict = next(item for item in selection["candidate_verdicts"] if item["coefficient"] == 0.25)
    for seed_verdict in verdict["per_seed"]:
        assert "channel_support_counts_incoherent" not in seed_verdict["failed_checks"]

def test_n3_beta_zero_removes_auxiliary_but_keeps_infonce_schedule() -> None:
    fit_role = _fit_role_sha256()
    trainer, arm, batch = _ccrp_fixture(coefficient=0.25)
    trainer.train_sprint18_ccrp_epoch([batch], arm=arm, epoch=11, fit_role_sha256=fit_role)
    ramped = trainer.objective_balance_history[-1]
    assert ramped["weighted_L_channel"] == pytest.approx(ramped["effective_coefficient_L_channel"] * ramped["raw_L_channel"])
    # beta=0 semantics: a zero effective coefficient zeroes the weighted
    # channel term while raw channel support stays observable.
    assert 0.0 * ramped["raw_L_channel"] == pytest.approx(0.0)
    assert ramped["effective_coefficient_L_info"] == pytest.approx(0.05 / 20.0)
    assert ramped["raw_L_info"] > 0.0


def test_n3_batches_match_b0_sampler_scope() -> None:
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
    n3 = make_sprint18_arm_config(
        "N3-CCRP", MODEL_SEEDS[0],
        {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": 0.25},
        selection_state="coefficient_pilot",
    )
    for epoch in (1, 15, 30):
        assert deterministic_sampler_order(
            binding, b0, workload="representation", scope="coefficient_pilot",
            role_history_ids=fit_history_ids, epoch=epoch, eligible_source_ids=fit_ids,
        ) == deterministic_sampler_order(
            binding, n3, workload="representation", scope="coefficient_pilot",
            role_history_ids=fit_history_ids, epoch=epoch, eligible_source_ids=fit_ids,
        )


def test_ccrp_selector_applies_frozen_grid_and_balance_gates(tmp_path=None) -> None:
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
                channel_total = 12
                per = [2, 2, 2, 2, 2, 2]
                rows.append(
                    {
                        "epoch": epoch,
                        "global_step": step,
                        "lr": learning_rate_for_epoch(epoch),
                        "raw_L_pred": 1.0,
                        "effective_coefficient_L_pred": 1.0,
                        "weighted_L_pred": 1.0,
                        "raw_L_channel": 2.0,
                        "effective_coefficient_L_channel": effective,
                        "weighted_L_channel": 2.0 * effective,
                        "raw_L_info": 1.5,
                        "effective_coefficient_L_info": coefficient_for_epoch(0.05, epoch),
                        "weighted_L_info": 1.5 * coefficient_for_epoch(0.05, epoch),
                        "gradient_norm_L_pred_shared": 1.0,
                        "gradient_norm_weighted_L_channel_shared": ratio,
                        "r_aux": ratio,
                        "embedding_support_files": 128,
                        "support_valid_files": 128,
                        "support_valid_patches": 128 * 12,
                        "support_prediction_files": 128,
                        "support_prediction_patches": 128 * 4,
                        "support_channel_targets": channel_total,
                        "support_channel_excluded_near_zero": 0,
                        "support_channel_c0": per[0],
                        "support_channel_c1": per[1],
                        "support_channel_c2": per[2],
                        "support_channel_c3": per[3],
                        "support_channel_c4": per[4],
                        "support_channel_c5": per[5],
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
    selection = select_ccrp_coefficient(binding, candidate_runs, prediction_only)
    assert selection["grid"] == list(N3_AUX_COEFFICIENT_GRID) == list(CCRP_AUX_COEFFICIENT_GRID)
    assert selection["selected_coefficient"] == 0.25
    verdicts = {row["coefficient"]: row for row in selection["candidate_verdicts"]}
    assert verdicts[0.10]["valid"] is False
    assert verdicts[0.25]["valid"] is True
    assert verdicts[0.50]["valid"] is False
    with pytest.raises(ValueError, match="frozen CCRP grid"):
        select_ccrp_coefficient(binding, {0.0: {}}, prediction_only)


def test_n3_coherent_checkpoint_restore_preserves_trained_inference(tmp_path) -> None:
    """Prove save -> strict restore -> public inference on the real updated N3 path."""
    binding = load_task6_binding(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "experiments"
        / "sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    coefficient = 0.25
    seed = MODEL_SEEDS[0]
    trainer, arm, batch = _ccrp_fixture(coefficient=coefficient, seed=seed)
    model = trainer.model
    optimizer = trainer.optimizer
    for group in optimizer.param_groups:
        group["lr"] = learning_rate_for_epoch(1)
    initialization_sha256 = state_sha256(model.state_dict())
    batches_per_epoch = (binding.fit_healthy_rows + BATCH_SIZE - 1) // BATCH_SIZE
    assert batches_per_epoch == 43
    fit_role = _canonical_sha256(sorted(item.manifest_sha256 for item in binding.role_histories("FIT")))
    for _ in range(batches_per_epoch):
        trainer.train_sprint18_ccrp_epoch([batch], arm=arm, epoch=1, fit_role_sha256=fit_role)
    assert trainer.step == batches_per_epoch
    assert state_sha256(model.state_dict()) != initialization_sha256
    balance = trainer.objective_balance_history[-1]
    assert balance["epoch"] == 1
    assert balance["support_channel_targets"] > 0
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
        "alpha": None,
        "beta": coefficient,
    }
    checkpoint_path = tmp_path / "n3-ccrp-coherent-pilot.pt"
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
    assert restored.arm_id == "N3-CCRP" and restored.model_seed == seed
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
    # Resume training from the restored checkpoint/optimizer state with the
    # correct next-epoch LR and coherent protocol, then verify a real update.
    resumed_trainer = RepresentationTrainer(
        restored_model, optimizer=restored_optimizer, max_grad_norm=1.0, seed=seed
    )
    resumed_trainer.step = restored.global_step
    resumed_trainer.train_sprint18_ccrp_epoch(
        [batch], arm=arm, epoch=2, fit_role_sha256=fit_role
    )
    assert resumed_trainer.step == restored.global_step + 1
    assert state_sha256(restored_model.state_dict()) != saved_sha256
    resumed_balance = resumed_trainer.objective_balance_history[-1]
    assert resumed_balance["epoch"] == 2
    assert resumed_balance["lr"] == learning_rate_for_epoch(2)
    assert resumed_balance["effective_coefficient_L_channel"] == coefficient_for_epoch(
        coefficient, 2
    )
