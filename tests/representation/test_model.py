from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from representation.config import V1Config
from representation.data import collate_variable_files
from representation.model import V1RepresentationModel
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import FileSample, RegimeMeta, RegimeType, SampleLabel


def _sample(channels: int, timesteps: int, file_id: str) -> FileSample:
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


@pytest.mark.parametrize("channels", [3, 6])
def test_model_forwards_homogeneous_channel_modes_and_variable_lengths(channels: int) -> None:
    config = V1Config(
        n_channels=channels,
        patch_size=16,
        stride=8,
        d_model=8,
        sequence_layers=1,
        attention_heads=2,
        dropout=0.0,
        total_mask_ratio=0.4,
    )
    samples = [_sample(channels, 21, "short"), _sample(channels, 43, "long")]
    patchifier = Patchifier(PatchConfig(patch_size=16, stride=8, pad_end=True))
    batch = collate_variable_files(samples, patchifier, masking_config=config, masking_seed=7)
    model = V1RepresentationModel(config, patchifier=patchifier).eval()

    with torch.no_grad():
        output = model(batch)
    assert tuple(output["context_latents"].shape) == (2, 5, 8)
    assert tuple(output["target_latents"].shape) == (2, 5, 8)
    assert tuple(output["predicted_latents"].shape) == (2, 5, 8)
    assert tuple(output["file_embedding"].shape) == (2, 8)
    assert tuple(output["view_embedding_1"].shape) == (2, 8)
    assert tuple(output["view_embedding_2"].shape) == (2, 8)
    assert torch.equal(output["prediction_mask"], batch["mask"] & batch["patch_valid_mask"])
    assert not output["target_latents"].requires_grad
    assert torch.isfinite(output["file_embedding"]).all()
    assert "reconstruction" not in output

    valid = batch["patch_valid_mask"].to(output["context_latents"].dtype).unsqueeze(-1)
    expected = (output["context_latents"] * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    torch.testing.assert_close(output["file_embedding"], expected)


def test_model_forward_does_not_consume_file_labels() -> None:
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1)
    sample = _sample(3, 17, "file")
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    batch = collate_variable_files([sample], patchifier)
    model = V1RepresentationModel(config, patchifier=patchifier).eval()
    with torch.no_grad():
        baseline = model(batch)
    batch["file_labels"] = [SampleLabel.ABNORMAL]
    with torch.no_grad():
        changed = model(batch)
    torch.testing.assert_close(baseline["file_embedding"], changed["file_embedding"])


def test_view_rng_state_round_trips_with_model_state_dict() -> None:
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    batch = collate_variable_files([_sample(3, 17, "state")], patchifier)
    torch.manual_seed(29)
    model = V1RepresentationModel(config, patchifier=patchifier).eval()
    model(batch)
    state = deepcopy(model.state_dict())
    expected = model(batch)
    restored = V1RepresentationModel(config, patchifier=patchifier).eval()
    restored.load_state_dict(state)
    actual = restored(batch)
    torch.testing.assert_close(expected["view_embedding_1"], actual["view_embedding_1"])
    torch.testing.assert_close(expected["view_embedding_2"], actual["view_embedding_2"])
def test_training_view_rng_advances_and_fresh_seeded_models_reproduce() -> None:
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0)
    samples = [_sample(3, 17, "rng")]
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    batch = collate_variable_files(samples, patchifier)
    torch.manual_seed(13)
    first_model = V1RepresentationModel(config, patchifier=patchifier).train()
    first = first_model(batch)
    second = first_model(batch)
    assert not torch.allclose(first["view_embedding_1"], second["view_embedding_1"])
    torch.manual_seed(13)
    second_model = V1RepresentationModel(config, patchifier=patchifier).train()
    replay = second_model(batch)
    torch.testing.assert_close(first["view_embedding_1"], replay["view_embedding_1"])
    torch.testing.assert_close(first["view_embedding_2"], replay["view_embedding_2"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_view_embeddings_follow_cuda_input_device() -> None:
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0)
    sample = _sample(3, 17, "cuda")
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    batch = collate_variable_files([sample], patchifier)
    batch = {name: value.cuda() if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
    model = V1RepresentationModel(config, patchifier=patchifier).cuda().eval()
    output = model(batch)
    assert output["view_embedding_1"].device.type == "cuda"
    assert output["view_embedding_1"].dtype == batch["patches"].dtype
