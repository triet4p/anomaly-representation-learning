from __future__ import annotations

import numpy as np
import pytest
import torch

from representation.config import V1Config
from representation.contracts import validate_batch, validate_output
from synth.schema import FileSample, RegimeMeta, RegimeType, SampleLabel


def _batch() -> dict[str, object]:
    b, c, t, n, w = 2, 6, 40, 3, 16
    return {
        "signals": torch.zeros((b, c, t)),
        "file_valid_mask": torch.ones((b, t), dtype=torch.bool),
        "patches": torch.zeros((b, n, c, w)),
        "patch_valid_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "patch_pad_mask": torch.tensor(
            [
                [[False] * w, [False] * 8 + [True] * 8, [True] * w],
                [[False] * w, [False] * w, [False] * 4 + [True] * 12],
            ],
            dtype=torch.bool,
        ),
        "starts": torch.zeros((b, n), dtype=torch.int64),
        "valid_len": torch.tensor([[16, 8, 0], [16, 16, 4]], dtype=torch.int64),
        "mask": torch.tensor([[True, False, False], [False, True, False]]),
        "file_ids": ["file-a", "file-b"],
    }


def test_v1_config_accepts_supported_file_sample_channel_counts() -> None:
    for channels in (3, 6):
        sample = FileSample(
            x=np.zeros((channels, 8), dtype=np.float32),
            file_id=f"file-{channels}",
            file_label=SampleLabel.NORMAL,
            seed=0,
            generator_version="test",
            config_hash="test",
            regime_sequence=[RegimeMeta(RegimeType.IDLE, 0, 8, 0.0)],
        )
        sample.validate()
        assert V1Config(n_channels=channels).n_channels == channels


def test_v1_config_has_runtime_length_not_fixed_file_length() -> None:
    config = V1Config()
    assert config.n_channels == 6
    assert config.patch_size == 32
    assert not hasattr(config, "seq_len")
    assert config.random_fraction + config.info_fraction + config.block_fraction == pytest.approx(1.0)


def test_v1_config_rejects_invalid_cross_field_values() -> None:
    with pytest.raises(ValueError, match="n_channels"):
        V1Config(n_channels=4)
    with pytest.raises(ValueError, match="stride"):
        V1Config(stride=33)
    with pytest.raises(ValueError, match="divisible"):
        V1Config(d_model=127)
    with pytest.raises(ValueError, match="sum to 1"):
        V1Config(random_fraction=0.5, info_fraction=0.5, block_fraction=0.5)
    with pytest.raises(ValueError, match="Extra inputs"):
        V1Config(unknown_setting=1)


def test_batch_contract_accepts_valid_padded_and_masked_batch() -> None:
    validate_batch(_batch())


def test_batch_contract_rejects_masked_padding_and_bad_pad_lengths() -> None:
    batch = _batch()
    batch["mask"] = torch.tensor([[True, False, True], [False, True, False]])
    with pytest.raises(ValueError, match="masked patches"):
        validate_batch(batch)

    batch = _batch()
    batch["valid_len"] = torch.tensor([[15, 8, 0], [16, 16, 4]], dtype=torch.int64)
    with pytest.raises(ValueError, match="pad_mask"):
        validate_batch(batch)


def test_output_contract_requires_stop_gradient_targets() -> None:
    context = torch.zeros((2, 3, 8), requires_grad=True)
    valid_output = {
        "context_latents": context,
        "target_latents": torch.zeros_like(context),
        "predicted_latents": context,
        "prediction_mask": torch.tensor([[True, False, False], [False, True, False]]),
        "file_embedding": torch.zeros((2, 8)),
    }
    validate_output(valid_output)

    invalid_output = dict(valid_output)
    invalid_output["target_latents"] = torch.zeros((2, 3, 8), requires_grad=True)
    with pytest.raises(ValueError, match="stop-gradient"):
        validate_output(invalid_output)
