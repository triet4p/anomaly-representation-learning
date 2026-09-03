from __future__ import annotations

import pytest
import torch

from representation.criterion import LatentPredictionCriterion


def _output() -> dict[str, torch.Tensor]:
    predicted = torch.tensor(
        [[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0], [4.0, 4.0]],
         [[2.0, 2.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]]],
        requires_grad=True,
    )
    target = torch.zeros_like(predicted, requires_grad=True)
    requested = torch.tensor([[True, True, True, False], [False, True, False, True]])
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    return {
        "predicted_latents": predicted,
        "target_latents": target,
        "prediction_mask": requested,
        "patch_valid_mask": valid,
    }


def test_prediction_loss_reduces_only_valid_masked_patches() -> None:
    output = _output()
    result = LatentPredictionCriterion()(output)
    # Valid requested entries are [0,0] and [0,1] and [1,1]: squared means 1, 9, 25.
    assert result["masked_count"].item() == 3
    assert result["masked_counts"].tolist() == [2, 1]
    assert result["loss"].item() == pytest.approx(35.0 / 3.0)
    assert result["patch_prediction_error"].tolist() == [[1.0, 9.0, 0.0, 0.0], [0.0, 25.0, 0.0, 0.0]]


def test_prediction_loss_detaches_targets_and_excludes_visible_or_padded_targets() -> None:
    output = _output()
    criterion = LatentPredictionCriterion()
    baseline = criterion(output)["loss"]
    output["target_latents"].data[0, 2] = 999_999.0
    output["target_latents"].data[0, 3] = -999_999.0
    output["target_latents"].data[1, 0] = 999_999.0
    changed = criterion(output)["loss"]
    assert changed.item() == pytest.approx(baseline.item())
    changed.backward()
    assert output["target_latents"].grad is None
    assert output["predicted_latents"].grad is not None


def test_prediction_loss_has_explicit_empty_mask_policies() -> None:
    output = _output()
    output["prediction_mask"] = torch.zeros_like(output["prediction_mask"])
    result = LatentPredictionCriterion(empty_policy="zero")(output)
    assert result["masked_count"].item() == 0
    assert result["loss"].item() == 0.0
    with pytest.raises(ValueError, match="no valid masked patches"):
        LatentPredictionCriterion(empty_policy="error")(output)
