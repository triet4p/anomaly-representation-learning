from __future__ import annotations

import pytest
import torch

from representation.baselines.rvq import RVQBaseline
from representation.model import V1RepresentationModel


def test_rvq_baseline_is_explicit_opt_in_and_has_documented_interface() -> None:
    for channels in (3, 6):
        baseline = RVQBaseline(channels, codebook_size=4, num_levels=2, embedding_dim=8).eval()
        inputs = torch.randn(2, channels, 8)
        quantized, commitment, indices = baseline(inputs)
        assert quantized.shape == inputs.shape
        assert commitment.ndim == 0 and torch.isfinite(commitment)
        assert indices.shape == (2, 2, channels)
        assert torch.isfinite(quantized).all()
        assert not any("decoder" in name or "reconstruction" in name for name in baseline.state_dict())

    assert not any(isinstance(module, RVQBaseline) for module in V1RepresentationModel.__subclasses__())


def test_rvq_baseline_rejects_invalid_shape_and_is_deterministic_after_initialization() -> None:
    baseline = RVQBaseline(3, codebook_size=4, num_levels=1, embedding_dim=4).eval()
    inputs = torch.randn(2, 3, 4)
    first = baseline(inputs)
    second = baseline(inputs)
    torch.testing.assert_close(first[0], second[0])
    assert torch.equal(first[2], second[2])
    with pytest.raises(ValueError, match="shape"):
        baseline(torch.randn(2, 3, 4, 1))
    with pytest.raises(ValueError, match="dimensions"):
        baseline(torch.randn(2, 6, 4))
