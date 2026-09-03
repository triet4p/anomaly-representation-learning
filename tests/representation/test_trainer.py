from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from representation.config import V1Config
from representation.criterion import JointRepresentationCriterion, ProgressiveLambda
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
