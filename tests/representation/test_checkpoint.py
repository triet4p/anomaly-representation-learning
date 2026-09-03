from __future__ import annotations

from copy import deepcopy
import pytest
import torch

from representation.checkpoint import load_checkpoint, save_checkpoint
from representation.config import V1Config
from representation.inference import NormalReferenceBank
from representation.model import V1RepresentationModel
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
        "file_ids": ["checkpoint"],
    }


def _model() -> V1RepresentationModel:
    config = V1Config(
        n_channels=3,
        patch_size=8,
        stride=8,
        d_model=8,
        attention_heads=2,
        sequence_layers=1,
        dropout=0.0,
        contrastive_warmup_steps=2,
    )
    return V1RepresentationModel(config, patchifier=Patchifier(PatchConfig(patch_size=8, stride=8))).eval()


def test_checkpoint_round_trip_restores_model_ema_optimizer_step_and_bank(tmp_path) -> None:
    model = _model()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    batch = _batch()
    output = model(batch)
    output["predicted_latents"].sum().backward()
    optimizer.step()
    scheduler.step()
    bank = NormalReferenceBank(k=3).fit(torch.tensor([[0.0] * 8, [1.0] * 8]))
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer=optimizer, scheduler=scheduler, step=9, reference_bank=bank)
    expected = {
        name: value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
        for name, value in model.state_dict().items()
    }
    expected_output = model(batch)

    restored = _model()
    restored_optimizer = torch.optim.Adam([p for p in restored.parameters() if p.requires_grad], lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=2)
    restored_bank = NormalReferenceBank(k=1)
    metadata = load_checkpoint(path, restored, optimizer=restored_optimizer, scheduler=restored_scheduler, reference_bank=restored_bank)
    assert metadata["step"] == 9
    assert metadata["config"]["contrastive_warmup_steps"] == 2
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, restored.state_dict()[name])
        else:
            assert value == restored.state_dict()[name]
    torch.testing.assert_close(expected_output["file_embedding"], restored(batch)["file_embedding"])
    assert restored_optimizer.state_dict()["state"]
    assert restored_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_checkpoint_rejects_incompatible_or_incomplete_payload(tmp_path) -> None:
    model = _model()
    path = tmp_path / "bad.pt"
    torch.save({"schema_version": 999}, path)
    with pytest.raises(ValueError, match="required keys"):
        load_checkpoint(path, model)
    torch.save({"schema_version": 1, "config": model.config.to_dict(), "model_state": model.state_dict(), "step": 0}, path)
    other = V1RepresentationModel(
        V1Config(n_channels=6, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0),
        patchifier=Patchifier(PatchConfig(patch_size=8, stride=8)),
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(path, other)
    with pytest.raises(ValueError, match="optimizer_state"):
        load_checkpoint(path, model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-3))


def test_checkpoint_atomic_save_keeps_existing_file_when_serialization_fails(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    path = tmp_path / "atomic.pt"
    save_checkpoint(path, model, step=1)
    before = path.read_bytes()

    def fail_save(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="simulated"):
        save_checkpoint(path, model, step=2)
    assert path.read_bytes() == before
