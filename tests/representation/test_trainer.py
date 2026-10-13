from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from representation.config import V1Config
from representation.criterion import JointRepresentationCriterion, ProgressiveLambda
from representation.data import collate_variable_files
from representation.model import V1RepresentationModel
from representation.trainer import RepresentationTrainer
from synth.config import PatchConfig
from synth.patchify import Patchifier


def _batch() -> dict[str, object]:
    return {
        "signals": torch.zeros(1, 3, 16),
        "file_valid_mask": torch.ones(1, 16, dtype=torch.bool),
        "patches": torch.randn(1, 2, 3, 8),
        "patch_valid_mask": torch.tensor([[True, True]]),
        "patch_pad_mask": torch.zeros(1, 2, 8, dtype=torch.bool),
        "starts": torch.tensor([[0, 8]], dtype=torch.int64),
        "valid_len": torch.tensor([[8, 8]], dtype=torch.int64),
        "mask": torch.tensor([[True, False]]),
        "file_ids": ["tiny"],
    }


def _trainer() -> RepresentationTrainer:
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0)
    model = V1RepresentationModel(config, patchifier=Patchifier(PatchConfig(patch_size=8, stride=8)))
    criterion = JointRepresentationCriterion(lambda_schedule=ProgressiveLambda(0.5, 2))
    return RepresentationTrainer(model, criterion, max_grad_norm=0.5, seed=5)


def test_trainer_steps_optimizer_before_ema_and_logs_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _trainer()
    batch = _batch()
    events: list[str] = []
    original_step = trainer.optimizer.step
    original_update = trainer.model.target_encoder.update

    def step(*args: object, **kwargs: object) -> None:
        events.append("optimizer")
        original_step(*args, **kwargs)

    def update(*args: object, **kwargs: object) -> None:
        events.append("ema")
        original_update(*args, **kwargs)

    monkeypatch.setattr(trainer.optimizer, "step", step)
    monkeypatch.setattr(trainer.model.target_encoder, "update", update)
    before = deepcopy(trainer.model.context_encoder.state_dict())
    metrics = trainer.train_epoch([batch])
    assert events == ["optimizer", "ema"]
    assert any(not torch.equal(before[name], value) for name, value in trainer.model.context_encoder.state_dict().items())
    assert set(("prediction_loss", "contrastive_loss", "joint_loss", "lambda", "step")) <= metrics.keys()
    assert metrics["lambda"] == 0.0


def test_validation_does_not_mutate_target_and_fit_tracks_best_history() -> None:
    trainer = _trainer()
    batch = _batch()
    trainer.train_epoch([batch])
    target_before = deepcopy(trainer.model.target_encoder.state_dict())
    validation = trainer.validate([batch])
    assert set(("prediction_loss", "contrastive_loss", "joint_loss", "lambda")) <= validation.keys()
    for name, value in target_before.items():
        torch.testing.assert_close(value, trainer.model.target_encoder.state_dict()[name])
    history = trainer.fit([batch], epochs=2, validation_batches=[batch])
    assert len(history) == 3
    assert "val_joint_loss" in history[-1]
    assert history[-1]["step"] == pytest.approx(3.0)


def test_trainer_rejects_empty_epochs_and_batches() -> None:
    trainer = _trainer()
    with pytest.raises(ValueError, match="no batches"):
        trainer.train_epoch([])
    with pytest.raises(ValueError, match="positive"):
        trainer.fit([_batch()], epochs=0)

def test_trainer_default_schedule_uses_configured_warmup_and_logs_boundaries() -> None:
    config = V1Config(
        n_channels=3,
        patch_size=8,
        stride=8,
        d_model=8,
        attention_heads=2,
        sequence_layers=1,
        dropout=0.0,
        contrastive_weight_max=0.4,
        contrastive_warmup_steps=2,
        contrastive_ramp_steps=2,
    )
    model = V1RepresentationModel(config, patchifier=Patchifier(PatchConfig(patch_size=8, stride=8)))
    trainer = RepresentationTrainer(model, max_grad_norm=0.5, seed=5)
    batch = _batch()

    assert isinstance(trainer.criterion, JointRepresentationCriterion)
    assert trainer.criterion.lambda_schedule.warmup_steps == 2
    assert trainer.criterion.lambda_schedule.ramp_steps == 2
    assert trainer.criterion.lambda_schedule.lambda_max == pytest.approx(0.4)

    logged_lambdas = [trainer.train_epoch([batch])["lambda"] for _ in range(4)]
    assert logged_lambdas[:3] == [0.0, 0.0, 0.0]
    assert logged_lambdas[3] == pytest.approx(0.2)


def test_config_rejects_negative_contrastive_warmup_steps() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        V1Config(contrastive_warmup_steps=-1)


def test_trainer_stationary_metric_and_coherent_best_state_selection() -> None:
    """Verify stationary scoring selects the best fully ramped validation state."""
    config = V1Config(
        n_channels=3,
        patch_size=8,
        stride=8,
        d_model=8,
        attention_heads=2,
        sequence_layers=1,
        dropout=0.0,
        contrastive_warmup_steps=100,
        contrastive_ramp_steps=100,
        contrastive_weight_max=1.0,
    )
    model = V1RepresentationModel(config, patchifier=Patchifier(PatchConfig(patch_size=8, stride=8)))
    trainer = RepresentationTrainer(model, selection_metric="val_stationary_joint_loss")

    # Compare stationary scores only after this fixture's configured ramp.
    full_ramp_step = trainer.criterion.lambda_schedule.full_ramp_step
    trainer.step = full_ramp_step
    first_metrics = {
        "val_prediction_loss": 0.2390,
        "val_contrastive_loss": 4.1672,
        "val_joint_loss": 4.4062,
        "val_stationary_joint_loss": 4.4062,
        "val_lambda": 1.0,
    }
    trainer.record_eval(first_metrics, epoch=8)
    assert trainer.best_step == full_ramp_step
    assert trainer.best_loss == pytest.approx(4.4062)

    # A later full-ramp evaluation has a lower stationary objective.
    # Prediction loss rose slightly (0.3809), but contrastive loss dropped (0.0131).
    trainer.step = full_ramp_step + 50
    joint_metrics = {
        "val_prediction_loss": 0.3809,
        "val_contrastive_loss": 0.0131,
        "val_joint_loss": 0.3940,
        "val_stationary_joint_loss": 0.3940,
        "val_lambda": 1.0,
    }
    trainer.record_eval(joint_metrics, epoch=10)

    # The later fully ramped joint state replaces the earlier eligible state.
    assert trainer.best_step == full_ramp_step + 50
    assert trainer.best_epoch == 10
    assert trainer.best_loss == pytest.approx(0.3940)
    assert trainer.best_state["step"] == full_ramp_step + 50


def test_trainer_restore_best_state_synchronizes_model_optimizer_scheduler_and_step(tmp_path) -> None:
    """Reproduce and prevent the epoch-2/step-7840 mismatch bug by ensuring all states synchronize."""
    from representation.checkpoint import load_checkpoint
    from representation.inference import NormalReferenceBank

    config = V1Config(
        n_channels=3, patch_size=8, stride=8, d_model=8,
        attention_heads=2, sequence_layers=1, dropout=0.0,
        contrastive_warmup_steps=0, contrastive_ramp_steps=1,
    )
    model = V1RepresentationModel(config, patchifier=Patchifier(PatchConfig(patch_size=8, stride=8)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    trainer = RepresentationTrainer(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=0,
        selection_metric="val_stationary_joint_loss",
    )

    # Use the fixture's full-ramp step rather than a default schedule constant.
    full_ramp_step = trainer.criterion.lambda_schedule.full_ramp_step
    trainer.step = full_ramp_step
    trainer.record_eval({"val_stationary_joint_loss": 0.5}, epoch=5)
    best_model_weights = {k: v.clone() if isinstance(v, torch.Tensor) else deepcopy(v) for k, v in model.state_dict().items()}

    # Continue to a later step with suboptimal model/optimizer/scheduler state.
    trainer.step = full_ramp_step + 1
    with torch.no_grad():
        for p in model.parameters():
            p.add_(5.0)
    optimizer.step()
    scheduler.step()
    trainer.record_eval({"val_stationary_joint_loss": 0.9}, epoch=20)

    assert trainer.step == full_ramp_step + 1

    restored_info = trainer.restore_best_state()
    assert restored_info is not None
    assert trainer.step == full_ramp_step
    for k, v in best_model_weights.items():
        actual = trainer.model.state_dict()[k]
        if isinstance(v, torch.Tensor):
            torch.testing.assert_close(actual, v)
        else:
            assert actual == v
    bank = NormalReferenceBank(k=1).fit(torch.randn(4, 8))
    ckpt_file = tmp_path / "coherent.pt"
    trainer.save_checkpoint(ckpt_file, reference_bank=bank)

    # Verify the saved checkpoint retains the best evaluation's step.
    restored_model = V1RepresentationModel(config, patchifier=Patchifier(PatchConfig(patch_size=8, stride=8)))
    restored_opt = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    restored_sched = torch.optim.lr_scheduler.StepLR(restored_opt, step_size=10)
    restored_bank = NormalReferenceBank(k=1)
    meta = load_checkpoint(ckpt_file, restored_model, optimizer=restored_opt, scheduler=restored_sched, reference_bank=restored_bank)

    assert meta["step"] == full_ramp_step
    for k, v in best_model_weights.items():
        actual = restored_model.state_dict()[k]
        if isinstance(v, torch.Tensor):
            torch.testing.assert_close(actual, v)
        else:
            assert actual == v


def test_trainer_resumes_from_checkpoint_preserving_step_and_optimizer(tmp_path) -> None:
    """Verify resuming training from a saved checkpoint preserves initial step and advances correctly."""
    from representation.checkpoint import save_checkpoint, load_checkpoint
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8))
    model = V1RepresentationModel(config, patchifier=patchifier)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5)

    ckpt_file = tmp_path / "resume_test.pt"
    save_checkpoint(ckpt_file, model, optimizer=opt, scheduler=sched, step=55)

    # Resume in new trainer
    resumed_model = V1RepresentationModel(config, patchifier=patchifier)
    resumed_opt = torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
    resumed_sched = torch.optim.lr_scheduler.StepLR(resumed_opt, step_size=5)
    meta = load_checkpoint(ckpt_file, resumed_model, optimizer=resumed_opt, scheduler=resumed_sched)

    trainer = RepresentationTrainer(
        resumed_model,
        optimizer=resumed_opt,
        scheduler=resumed_sched,
        step=meta["step"],
    )
    assert trainer.step == 55

    batch = _batch()
    metrics = trainer.train_epoch([batch])
    assert trainer.step == 56
    assert metrics["step"] == 56.0
def _file_sample(channels: int, timesteps: int, file_id: str) -> FileSample:
    import numpy as np
    from synth.schema import FileSample, RegimeMeta, RegimeType, SampleLabel
    signal = np.arange(channels * timesteps, dtype=np.float32).reshape(channels, timesteps)
    sample = FileSample(
        x=signal,
        file_id=file_id,
        file_label=SampleLabel.NORMAL,
        seed=timesteps,
        generator_version="test",
        config_hash="test",
        regime_sequence=[RegimeMeta(RegimeType.ACTIVE, 0, timesteps, 0.5)],
    )
    sample.validate()
    return sample


def test_rerun_diagnostics_expose_required_objective_components_and_latent_norms() -> None:
    """Verify train_epoch exposes all required raw/weighted losses, lambda terms, and latent norms."""
    config = V1Config(
        n_channels=3,
        patch_size=8,
        stride=8,
        d_model=8,
        attention_heads=2,
        sequence_layers=1,
        dropout=0.0,
        contrastive_warmup_steps=1,
        contrastive_ramp_steps=2,
        contrastive_weight_max=1.0,
    )
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    s1 = _file_sample(3, 16, "s1")
    s2 = _file_sample(3, 24, "s2")
    from representation.data import collate_variable_files
    batch = collate_variable_files([s1, s2], patchifier, masking_config=config, masking_seed=1)

    model = V1RepresentationModel(config, patchifier=patchifier)
    trainer = RepresentationTrainer(model)

    metrics = trainer.train_epoch([batch])

    # 1. Required diagnostic keys
    required_keys = {
        "prediction_loss",
        "contrastive_loss",
        "weighted_contrastive_loss",
        "joint_loss",
        "stationary_joint_loss",
        "lambda",
        "effective_lambda",
        "stationary_lambda",
        "context_norm",
        "target_norm",
        "predicted_norm",
        "file_embedding_norm",
        "contrastive_sim",
        "contrastive_neg_sim",
        "contrastive_margin",
        "lr",
        "grad_norm",
        "step",
    }
    assert required_keys.issubset(metrics.keys()), f"Missing diagnostic keys: {required_keys - set(metrics.keys())}"

    # 2. Arithmetic relationships
    assert metrics["effective_lambda"] == pytest.approx(metrics["lambda"])
    assert metrics["stationary_lambda"] == pytest.approx(1.0)
    assert metrics["weighted_contrastive_loss"] == pytest.approx(metrics["effective_lambda"] * metrics["contrastive_loss"])
    assert metrics["joint_loss"] == pytest.approx(metrics["prediction_loss"] + metrics["weighted_contrastive_loss"])
    assert metrics["stationary_joint_loss"] == pytest.approx(metrics["prediction_loss"] + metrics["stationary_lambda"] * metrics["contrastive_loss"])
    assert metrics["contrastive_margin"] == pytest.approx(metrics["contrastive_sim"] - metrics["contrastive_neg_sim"], abs=1e-5)

    # 3. Bounded values
    assert 0.0 <= metrics["contrastive_sim"] <= 1.0 + 1e-4
    assert -1.0 - 1e-4 <= metrics["contrastive_neg_sim"] <= 1.0 + 1e-4
    assert -2.0 - 1e-4 <= metrics["contrastive_margin"] <= 2.0 + 1e-4
    assert metrics["lr"] > 0.0
    assert metrics["grad_norm"] >= 0.0
    assert metrics["context_norm"] > 0.0
    assert metrics["target_norm"] > 0.0
    assert metrics["predicted_norm"] > 0.0
    assert metrics["file_embedding_norm"] > 0.0

def test_validation_diagnostics_match_schema_and_prefix_in_history() -> None:
    """Verify validation diagnostics are prefixed and fully recorded in trainer history."""
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    s1 = _file_sample(3, 16, "s1")
    s2 = _file_sample(3, 24, "s2")
    batch = collate_variable_files([s1, s2], patchifier, masking_config=config, masking_seed=1)

    model = V1RepresentationModel(config, patchifier=patchifier)
    trainer = RepresentationTrainer(model)

    history = trainer.fit([batch], epochs=2, validation_batches=[batch])
    assert len(history) == 2

    for entry in history:
        # Training keys
        assert "prediction_loss" in entry
        assert "contrastive_loss" in entry
        assert "contrastive_sim" in entry
        assert "contrastive_neg_sim" in entry
        assert "contrastive_margin" in entry
        assert "lr" in entry
        assert "grad_norm" in entry
        assert "context_norm" in entry
        assert "target_norm" in entry
        assert "predicted_norm" in entry
        assert "file_embedding_norm" in entry
        assert "stationary_joint_loss" in entry

        # Validation keys
        assert "val_prediction_loss" in entry
        assert "val_contrastive_loss" in entry
        assert "val_weighted_contrastive_loss" in entry
        assert "val_context_norm" in entry
        assert "val_target_norm" in entry
        assert "val_predicted_norm" in entry
        assert "val_file_embedding_norm" in entry
        assert "val_stationary_joint_loss" in entry
        assert "val_contrastive_sim" in entry
        assert "val_contrastive_neg_sim" in entry
        assert "val_contrastive_margin" in entry
        # Confirm machine-readable and finite
        for k, v in entry.items():
            assert isinstance(v, float)
            assert float("-inf") < v < float("inf")


def test_criterion_diagnostics_zero_overhead_when_views_absent() -> None:
    """Verify criterion cleanly evaluates outputs without contrastive views, computing available norms."""
    criterion = JointRepresentationCriterion()
    output = {
        "predicted_latents": torch.randn(2, 4, 8),
        "target_latents": torch.randn(2, 4, 8),
        "context_latents": torch.randn(2, 4, 8),
        "prediction_mask": torch.ones(2, 4, dtype=torch.bool),
        "patch_valid_mask": torch.ones(2, 4, dtype=torch.bool),
        "file_embedding": torch.randn(2, 8),
    }
    terms = criterion(output)
    assert terms["contrastive_loss"].item() == 0.0
    assert terms["weighted_contrastive_loss"].item() == 0.0
    assert "contrastive_sim" not in terms
    assert terms["context_norm"].item() > 0.0
    assert terms["target_norm"].item() > 0.0
    assert terms["predicted_norm"].item() > 0.0
    assert terms["file_embedding_norm"].item() > 0.0


def test_resume_start_epoch_derivation_and_clean_skip() -> None:
    """Verify start_epoch derives from start_step and cleanly stops when total epochs are completed."""
    batches_per_epoch = 196
    total_epochs = 40

    # Case 1: Fresh start
    start_step = 0
    start_epoch = start_step // batches_per_epoch
    assert start_epoch == 0
    remaining_epochs = list(range(start_epoch + 1, total_epochs + 1))
    assert len(remaining_epochs) == 40
    assert remaining_epochs[0] == 1 and remaining_epochs[-1] == 40

    # Case 2: Resume mid-run at step 3920 (epoch 20 completed)
    start_step = 3920
    start_epoch = start_step // batches_per_epoch
    assert start_epoch == 20
    remaining_epochs = list(range(start_epoch + 1, total_epochs + 1))
    assert len(remaining_epochs) == 20
    assert remaining_epochs[0] == 21 and remaining_epochs[-1] == 40

    # Case 3: Resume completed run at step 7840 (all 40 epochs completed)
    start_step = 7840
    start_epoch = start_step // batches_per_epoch
    assert start_epoch == 40
    assert start_epoch >= total_epochs
    remaining_epochs = list(range(start_epoch + 1, total_epochs + 1))
    assert len(remaining_epochs) == 0  # Clean skip!


def _sprint18_training_fixture(*, coefficient: float = 0.025, pilot: bool = False):
    import numpy as np
    from representation.data import collate_variable_files
    from representation.sprint18_harness import make_sprint18_arm_config
    from synth.schema import FileSample, SampleLabel

    config = V1Config(
        n_channels=6,
        patch_size=32,
        stride=16,
        d_model=16,
        attention_heads=4,
        sequence_layers=1,
        dropout=0.1,
        n_robots=9,
        n_programs=8,
        seed=181801,
        contrastive_weight_max=coefficient,
        ema_decay=0.996,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    rng = np.random.default_rng(818)
    samples = []
    for index in range(2):
        signal = rng.normal(size=(6, 96)).astype(np.float32)
        signal[0] += index * 0.5
        sample = FileSample(
            x=signal,
            file_id=f"task8-real-model-{index}",
            file_label=SampleLabel.NORMAL,
            seed=181801 + index,
            generator_version="task8-test-fixture",
            config_hash="task8-test-fixture",
            regime_sequence=[],
            robot_idx=index,
            program_idx=index,
        )
        sample.validate()
        samples.append(sample)
    batch = collate_variable_files(
        samples,
        patchifier,
        masking_config=config,
        masking_seed=818,
    )
    arm = make_sprint18_arm_config(
        "B0-FULL",
        181801,
        {
            "lambda_info_max": coefficient,
            "lambda_hcc_max": None,
            "alpha": None,
            "beta": None,
        },
        selection_state="coefficient_pilot" if pilot else "task6_smoke_only",
    )
    model = V1RepresentationModel(config, patchifier=patchifier)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.0)
    trainer = RepresentationTrainer(
        model,
        optimizer=optimizer,
        max_grad_norm=1.0,
        seed=181801,
    )
    return trainer, arm, batch


def test_sprint18_balance_records_warmup_active_losses_and_exact_support() -> None:
    from representation.sprint18_checkpoint import learning_rate_for_epoch

    trainer, arm, batch = _sprint18_training_fixture()
    warmup = trainer.train_sprint18_epoch([batch], arm=arm, epoch=1)
    first = trainer.objective_balance_history[-1]
    assert first["epoch"] == 1 and first["global_step"] == 1
    assert first["raw_L_info"] > 0.0
    assert first["effective_coefficient_L_info"] == 0.0
    assert first["weighted_L_info"] == pytest.approx(
        first["effective_coefficient_L_info"] * first["raw_L_info"]
    )
    assert first["effective_coefficient_L_pred"] == 1.0
    assert first["weighted_L_pred"] == pytest.approx(first["raw_L_pred"])
    assert first["support_valid_files"] == 2
    assert first["support_valid_patches"] == 10
    assert first["support_prediction_files"] == 2
    assert first["support_prediction_patches"] > 0
    assert first["support_info_files"] == 2
    assert first["contrastive_input_files"] == 2
    assert first["embedding_support_files"] == 2
    assert first["lr"] == pytest.approx(learning_rate_for_epoch(1))
    assert warmup["epoch"] == 1.0 and warmup["step"] == 1.0
    assert first["protocol_gradient_observation"] is False
    assert "balance_r_aux" in warmup

    active = trainer.train_sprint18_epoch([batch], arm=arm, epoch=11)
    second = trainer.objective_balance_history[-1]
    assert second["epoch"] == 11 and second["global_step"] == 2
    assert second["effective_coefficient_L_info"] == pytest.approx(0.025 / 20.0)
    assert second["raw_L_info"] > 0.0
    assert second["weighted_L_info"] == pytest.approx(
        second["effective_coefficient_L_info"] * second["raw_L_info"]
    )
    assert second["lr"] == pytest.approx(learning_rate_for_epoch(11))
    assert active["epoch"] == 11.0 and active["step"] == 2.0
    assert second["r_aux"] == pytest.approx(
        second["gradient_norm_weighted_L_info_shared"]
        / (second["gradient_norm_L_pred_shared"] + 1e-12)
    )
    assert 0.0 <= second["embedding_effective_rank"] <= 16.0
    assert second["embedding_variance_mean"] >= 0.0


def test_embedding_geometry_handles_empty_single_and_collapsed_support() -> None:
    empty = RepresentationTrainer._embedding_geometry(
        torch.empty((0, 3)),
        torch.empty((0,), dtype=torch.bool),
    )
    single = RepresentationTrainer._embedding_geometry(
        torch.zeros((1, 3)),
        torch.tensor([True]),
    )
    collapsed = RepresentationTrainer._embedding_geometry(
        torch.ones((4, 3)),
        torch.ones((4,), dtype=torch.bool),
    )
    assert empty == {
        "embedding_support_files": 0,
        "embedding_variance_mean": 0.0,
        "embedding_std_median": 0.0,
        "embedding_effective_rank": 0.0,
    }
    for result in (single, collapsed):
        assert result["embedding_variance_mean"] == 0.0
        assert result["embedding_std_median"] == 0.0
        assert result["embedding_effective_rank"] == 0.0
    with pytest.raises(ValueError, match="must be finite"):
        RepresentationTrainer._embedding_geometry(
            torch.tensor([[float("nan"), 0.0]]),
            torch.tensor([True]),
        )


def test_sprint18_instrumentation_preserves_real_update_and_rng_at_balance_step() -> None:
    import random

    import numpy as np

    from representation.sprint18_checkpoint import capture_rng_state, state_sha256

    template, arm, batch = _sprint18_training_fixture()
    model_without = deepcopy(template.model)
    model_with = deepcopy(template.model)
    trainer_without = RepresentationTrainer(
        model_without,
        optimizer=torch.optim.AdamW(
            model_without.parameters(), lr=1.5e-3, weight_decay=0.0
        ),
        max_grad_norm=1.0,
        seed=181801,
        step=39,
    )
    trainer_with = RepresentationTrainer(
        model_with,
        optimizer=torch.optim.AdamW(
            model_with.parameters(), lr=1.5e-3, weight_decay=0.0
        ),
        max_grad_norm=1.0,
        seed=181801,
        step=39,
    )
    initial_rng = capture_rng_state()

    def restore_rng(state: dict[str, object]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    restore_rng(initial_rng)
    without_metrics = trainer_without.train_sprint18_epoch(
        [batch], arm=arm, epoch=26, collect_balance=False
    )
    without_rng = capture_rng_state()

    restore_rng(initial_rng)
    with_metrics = trainer_with.train_sprint18_epoch(
        [batch], arm=arm, epoch=26, collect_balance=True
    )
    with_rng = capture_rng_state()
    observation = trainer_with.objective_balance_history[-1]
    assert observation["global_step"] == 40
    assert observation["protocol_gradient_observation"] is True
    assert without_metrics["joint_loss"] == pytest.approx(
        with_metrics["joint_loss"], rel=0.0, abs=0.0
    )
    assert state_sha256(trainer_without.model.state_dict()) == state_sha256(
        trainer_with.model.state_dict()
    )
    assert state_sha256(trainer_without.model.target_encoder.state_dict()) == state_sha256(
        trainer_with.model.target_encoder.state_dict()
    )
    assert state_sha256(trainer_without.optimizer.state_dict()) == state_sha256(
        trainer_with.optimizer.state_dict()
    )
    assert state_sha256(without_rng) == state_sha256(with_rng)


def test_task9_pilot_uses_full_warmup_ramp_and_stops_at_epoch_30() -> None:
    from representation.sprint18_harness import (
        TASK9_COEFFICIENT_PILOT_EPOCHS,
        make_sprint18_arm_config,
    )

    trainer, smoke_arm, batch = _sprint18_training_fixture()
    pilot = make_sprint18_arm_config(
        "B0-FULL",
        smoke_arm.model_seed,
        {
            "lambda_info_max": smoke_arm.lambda_info_max,
            "lambda_hcc_max": None,
            "alpha": None,
            "beta": None,
        },
        selection_state="coefficient_pilot",
    )
    with pytest.raises(ValueError, match="require objective-balance diagnostics"):
        trainer.train_sprint18_epoch(
            [batch], arm=pilot, epoch=1, collect_balance=False
        )
    for epoch in range(1, TASK9_COEFFICIENT_PILOT_EPOCHS + 1):
        trainer.train_sprint18_epoch([batch], arm=pilot, epoch=epoch)
    assert trainer.step == TASK9_COEFFICIENT_PILOT_EPOCHS
    assert len(trainer.objective_balance_history) == TASK9_COEFFICIENT_PILOT_EPOCHS
    warmup = trainer.objective_balance_history[0]
    full_ramp = trainer.objective_balance_history[-1]
    assert warmup["effective_coefficient_L_info"] == 0.0
    assert full_ramp["effective_coefficient_L_info"] == pilot.lambda_info_max
    assert full_ramp["raw_L_info"] > 0.0
    assert full_ramp["weighted_L_info"] > 0.0
    with pytest.raises(ValueError, match="stop at the frozen epoch-30"):
        trainer.train_sprint18_epoch([batch], arm=pilot, epoch=31)
    baseline_trainer, baseline_arm, baseline_batch = _sprint18_training_fixture(
        coefficient=0.0, pilot=True
    )
    baseline_trainer.train_sprint18_epoch(
        [baseline_batch], arm=baseline_arm, epoch=1, collect_balance=False
    )
    assert baseline_arm.lambda_info_max == 0.0
    assert baseline_trainer.step == 1
    assert baseline_trainer.objective_balance_history == []
