from __future__ import annotations

import pytest
import torch

from representation.layers.patch_encoder import LocalPatchEncoder


def _inputs(channels: int, width: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    patches = torch.randn(2, 3, channels, width)
    pad_mask = torch.zeros(2, 3, width, dtype=torch.bool)
    pad_mask[0, 1, 5:] = True
    pad_mask[0, 2] = True
    pad_mask[1, 2, 6:] = True
    return patches, pad_mask


def test_local_patch_encoder_supports_both_channel_layouts_and_runtime_counts() -> None:
    for channels in (3, 6):
        patches, pad_mask = _inputs(channels)
        encoder = LocalPatchEncoder(channels, d_model=12).eval()
        with torch.no_grad():
            output = encoder(patches, pad_mask)
        assert tuple(output.shape) == (2, 3, 12)
        assert torch.isfinite(output).all()
        assert torch.all(output[0, 2] == 0)

        shorter = LocalPatchEncoder(channels, d_model=12).eval()
        with torch.no_grad():
            runtime_output = shorter(patches[:, :2], pad_mask[:, :2])
        assert tuple(runtime_output.shape) == (2, 2, 12)


def test_padded_values_cannot_change_valid_patch_embeddings() -> None:
    patches, pad_mask = _inputs(3)
    encoder = LocalPatchEncoder(3, d_model=10).eval()
    changed = patches.clone()
    changed.masked_fill_(pad_mask.unsqueeze(2), 10_000.0)

    with torch.no_grad():
        first = encoder(patches, pad_mask)
        second = encoder(changed, pad_mask)
    torch.testing.assert_close(first, second)


def test_fully_padded_patch_has_no_usable_embedding() -> None:
    patches, pad_mask = _inputs(6)
    encoder = LocalPatchEncoder(6, d_model=8).eval()
    with torch.no_grad():
        output = encoder(patches, pad_mask)
    assert torch.equal(output[0, 2], torch.zeros(8))


def test_local_patch_encoder_rejects_invalid_shapes_and_configuration() -> None:
    with pytest.raises(ValueError, match="n_channels"):
        LocalPatchEncoder(0, d_model=8)
    encoder = LocalPatchEncoder(3, d_model=8)
    patches, pad_mask = _inputs(3)
    with pytest.raises(ValueError, match="patches must"):
        encoder(patches[0], pad_mask)
    with pytest.raises(ValueError, match="match patches"):
        encoder(patches, pad_mask[:, :, :-1])
