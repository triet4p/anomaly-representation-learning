from __future__ import annotations

import pytest
import torch

from representation.layers.sequence_encoder import SequenceContextEncoder


def test_sequence_encoder_supports_runtime_lengths_and_ignores_padding() -> None:
    encoder = SequenceContextEncoder(12, attention_heads=3, layers=1).eval()
    tokens = torch.randn(2, 5, 12)
    valid = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])
    changed = tokens.clone()
    changed[0, 3:] = 10_000.0
    changed[1, 2:] = -10_000.0

    with torch.no_grad():
        output = encoder(tokens, valid)
        changed_output = encoder(changed, valid)
        shorter = encoder(tokens[:, :3], valid[:, :3])
    assert tuple(output.shape) == (2, 5, 12)
    assert tuple(shorter.shape) == (2, 3, 12)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[valid], changed_output[valid])
    assert torch.equal(output[~valid], torch.zeros_like(output[~valid]))


def test_sequence_encoder_handles_fully_padded_files_without_nan() -> None:
    encoder = SequenceContextEncoder(8, attention_heads=2, layers=2).eval()
    tokens = torch.randn(2, 4, 8)
    valid = torch.tensor([[False, False, False, False], [True, False, False, False]])
    with torch.no_grad():
        output = encoder(tokens, valid)
    assert torch.isfinite(output).all()
    assert torch.equal(output[0], torch.zeros_like(output[0]))


def test_sequence_encoder_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SequenceContextEncoder(10, attention_heads=3)
    encoder = SequenceContextEncoder(8, attention_heads=2)
    with pytest.raises(ValueError, match="tokens must"):
        encoder(torch.randn(2, 8), torch.ones(2, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match="match tokens"):
        encoder(torch.randn(2, 3, 8), torch.ones(2, 2, dtype=torch.bool))


def test_sequence_encoder_output_is_normalized_and_bounded() -> None:
    encoder = SequenceContextEncoder(16, attention_heads=4, layers=2).eval()
    assert hasattr(encoder, "norm")
    assert isinstance(encoder.norm, torch.nn.LayerNorm)

    tokens = torch.randn(2, 6, 16) * 50.0 + 100.0
    valid = torch.tensor([[True, True, True, True, False, False], [True, True, False, False, False, False]])
    with torch.no_grad():
        output = encoder(tokens, valid)

    # Valid tokens must have zero mean, unit variance, bounded norm sqrt(D)
    for b in range(2):
        for t in range(6):
            if valid[b, t]:
                assert output[b, t].mean().abs() < 1e-4
                assert (output[b, t].std(unbiased=False) - 1.0).abs() < 1e-3
                assert (output[b, t].norm() - 4.0).abs() < 0.1
            else:
                assert torch.equal(output[b, t], torch.zeros(16))
