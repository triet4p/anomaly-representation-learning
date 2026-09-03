from __future__ import annotations

import pytest
import torch

from representation.criterion import FileContrastiveCriterion, ProgressiveLambda


def test_file_contrastive_criterion_normalizes_views_and_uses_same_file_diagonal() -> None:
    first = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    second = torch.tensor([[2.0, 0.0], [0.0, 3.0]], requires_grad=True)
    result = FileContrastiveCriterion(temperature=0.2)(
        {
            "view_embedding_1": first,
            "view_embedding_2": second,
            "file_labels": ["normal", "abnormal"],
        }
    )
    assert torch.allclose(result["normalized_view_1"].norm(dim=-1), torch.ones(2))
    assert torch.allclose(result["normalized_view_2"].norm(dim=-1), torch.ones(2))
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert first.grad is not None and second.grad is not None


def test_file_contrastive_criterion_is_finite_for_batch_size_one() -> None:
    result = FileContrastiveCriterion()(
        {"view_embedding_1": torch.tensor([[1.0, 2.0]]), "view_embedding_2": torch.tensor([[1.0, 2.0]])}
    )
    assert result["batch_size"].item() == 1
    assert result["loss"].item() == pytest.approx(0.0)
    assert torch.isfinite(result["loss"])


def test_contrastive_temperature_and_progressive_lambda_validation_and_endpoints() -> None:
    with pytest.raises(ValueError, match="temperature"):
        FileContrastiveCriterion(temperature=0.0)
    with pytest.raises(ValueError, match="lambda_max"):
        ProgressiveLambda(lambda_max=-1.0)
    with pytest.raises(ValueError, match="ramp_steps"):
        ProgressiveLambda(ramp_steps=-1)
    schedule = ProgressiveLambda(lambda_max=0.8, ramp_steps=4)
    values = [schedule.lambda_at(step) for step in range(8)]
    assert values[0] == 0.0
    assert values[4] == pytest.approx(0.8)
    assert values[-1] == pytest.approx(0.8)
    assert values == sorted(values)
    assert ProgressiveLambda(lambda_max=0.8, ramp_steps=0).lambda_at(0) == 0.0
    assert ProgressiveLambda(lambda_max=0.8, ramp_steps=0).lambda_at(1) == pytest.approx(0.8)

def test_progressive_lambda_holds_warmup_then_ramps_monotonically() -> None:
    schedule = ProgressiveLambda(lambda_max=0.8, ramp_steps=4, warmup_steps=2)
    values = [schedule.lambda_at(step) for step in range(8)]

    assert values[1] == 0.0
    assert values[2] == 0.0
    assert values[3] == pytest.approx(0.2)
    assert values[6] == pytest.approx(0.8)
    assert values[7] == pytest.approx(0.8)
    assert values == sorted(values)

    with pytest.raises(ValueError, match="warmup_steps"):
        ProgressiveLambda(warmup_steps=-1)
