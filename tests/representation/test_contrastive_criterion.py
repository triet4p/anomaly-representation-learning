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


def test_file_contrastive_criterion_computes_negative_similarity_and_margin() -> None:
    criterion = FileContrastiveCriterion(temperature=0.2)

    # Batch size 1 boundary
    res_b1 = criterion({
        "view_embedding_1": torch.tensor([[1.0, 2.0]]),
        "view_embedding_2": torch.tensor([[1.0, 2.0]]),
    })
    assert "negative_similarity" in res_b1
    assert "margin" in res_b1
    assert res_b1["negative_similarity"].item() == pytest.approx(0.0)
    assert res_b1["margin"].item() == pytest.approx(res_b1["similarity"].item())

    # Batch size 2 with known geometry:
    # v1 = [[1, 0], [0, 1]], v2 = [[0.8, 0.6], [0.6, 0.8]]
    # normalized v1: [[1, 0], [0, 1]], normalized v2: [[0.8, 0.6], [0.6, 0.8]]
    # pos sim: (0.8 + 0.8) / 2 = 0.8
    # neg sim: (0.6 + 0.6) / 2 = 0.6
    # margin: 0.8 - 0.6 = 0.2
    v1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    v2 = torch.tensor([[0.8, 0.6], [0.6, 0.8]], requires_grad=True)
    res_b2 = criterion({"view_embedding_1": v1, "view_embedding_2": v2})
    assert res_b2["similarity"].item() == pytest.approx(0.8, abs=1e-5)
    assert res_b2["negative_similarity"].item() == pytest.approx(0.6, abs=1e-5)
    assert res_b2["margin"].item() == pytest.approx(0.2, abs=1e-5)
    assert res_b2["margin"].item() == pytest.approx(res_b2["similarity"].item() - res_b2["negative_similarity"].item())

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
