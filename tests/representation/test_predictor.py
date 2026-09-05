from __future__ import annotations

import pytest
import torch

from representation.layers.predictor import MaskedLatentPredictor


def test_predictor_preserves_patch_alignment_and_target_dimensions() -> None:
    predictor = MaskedLatentPredictor(8, target_dim=5, hidden_dim=16).eval()
    context = torch.randn(2, 4, 8)
    requested = torch.tensor([[False, True, True, False], [True, False, True, False]])
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    predictions, prediction_mask = predictor(context, requested, valid)

    assert tuple(predictions.shape) == (2, 4, 5)
    assert torch.equal(prediction_mask, requested & valid)
    assert torch.isfinite(predictions).all()
    assert torch.equal(predictions[~prediction_mask], torch.zeros_like(predictions[~prediction_mask]))

def test_masked_query_attends_to_nearby_visible_context_not_invalid_tokens() -> None:
    predictor = MaskedLatentPredictor(8, target_dim=8).eval()
    context = torch.randn(1, 4, 8)
    requested = torch.tensor([[False, True, False, False]])
    valid = torch.tensor([[True, True, True, False]])
    with torch.no_grad():
        baseline, prediction_mask = predictor(context, requested, valid)
        nearby = context.clone()
        nearby[0, 0] += 5.0
        nearby_output, _ = predictor(nearby, requested, valid)
        invalid = context.clone()
        invalid[0, 3] += 5_000.0
        invalid_output, _ = predictor(invalid, requested, valid)
    assert prediction_mask.tolist() == [[False, True, False, False]]
    assert not torch.allclose(baseline[0, 1], nearby_output[0, 1])
    torch.testing.assert_close(baseline, invalid_output)


def test_predictor_uses_visible_context_and_routes_gradients_only_through_context() -> None:
    predictor = MaskedLatentPredictor(6, target_dim=6).eval()
    context = torch.randn(1, 4, 6, requires_grad=True)
    requested = torch.tensor([[False, True, False, True]])
    valid = torch.tensor([[True, True, True, False]])
    predictions, prediction_mask = predictor(context, requested, valid)
    loss = predictions[prediction_mask].square().mean()
    loss.backward()
    assert prediction_mask.tolist() == [[False, True, False, False]]
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()
    assert context.grad[0, 0].abs().sum() > 0
    assert context.grad[0, 2].abs().sum() > 0
    assert torch.equal(context.grad[0, 1], torch.zeros_like(context.grad[0, 1]))
    assert torch.equal(context.grad[0, 3], torch.zeros_like(context.grad[0, 3]))


def test_predictor_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="d_model"):
        MaskedLatentPredictor(0)
    predictor = MaskedLatentPredictor(8)
    with pytest.raises(ValueError, match="context_latents"):
        predictor(torch.randn(2, 8), torch.ones(2, 8, dtype=torch.bool), torch.ones(2, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match="match context_latents"):
        predictor(
            torch.randn(2, 3, 8),
            torch.ones(2, 2, dtype=torch.bool),
            torch.ones(2, 3, dtype=torch.bool),
        )


def test_predictor_output_latents_are_normalized_and_bounded() -> None:
    predictor = MaskedLatentPredictor(16, target_dim=16, hidden_dim=32).eval()
    context = torch.randn(2, 5, 16) * 30.0 - 50.0
    requested = torch.tensor([[True, False, True, False, False], [False, True, True, False, False]])
    valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])

    with torch.no_grad():
        predictions, pred_mask = predictor(context, requested, valid)

    assert tuple(predictions.shape) == (2, 5, 16)
    # Masked valid positions must have zero mean, unit variance, norm ~ sqrt(16) = 4.0
    for b in range(2):
        for t in range(5):
            if pred_mask[b, t]:
                assert predictions[b, t].mean().abs() < 1e-4
                assert (predictions[b, t].std(unbiased=False) - 1.0).abs() < 1e-3
                assert (predictions[b, t].norm() - 4.0).abs() < 0.1
            else:
                assert torch.equal(predictions[b, t], torch.zeros(16))
