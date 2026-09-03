from __future__ import annotations

import pytest
import torch
from torch import nn

from representation.inference import NormalReferenceBank, RepresentationInference, mad_threshold
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import SampleLabel


class _StaticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predicted = torch.tensor([[[1.0, 1.0], [4.0, 4.0]]])

    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        context = torch.zeros(1, 2, 2)
        target = torch.zeros_like(context)
        return {
            "context_latents": context,
            "target_latents": target,
            "predicted_latents": self.predicted,
            "prediction_mask": torch.tensor([[True, False]]),
            "file_embedding": torch.tensor([[1.0, 0.0]]),
        }


def _batch() -> dict[str, object]:
    return {
        "starts": torch.tensor([[0, 2]], dtype=torch.int64),
        "valid_len": torch.tensor([[3, 2]], dtype=torch.int64),
        "file_valid_mask": torch.ones(1, 4, dtype=torch.bool),
    }


def test_inference_keeps_context_and_population_scores_independent() -> None:
    bank = NormalReferenceBank(k=5).fit(torch.tensor([[0.0, 0.0], [2.0, 0.0]]), [SampleLabel.NORMAL, SampleLabel.NORMAL])
    model = _StaticModel()
    inference = RepresentationInference(model, bank, Patchifier(PatchConfig(patch_size=3, stride=2)))
    first = inference.score_batch(_batch())
    model.predicted = torch.tensor([[[10.0, 10.0], [4.0, 4.0]]])
    changed_prediction = inference.score_batch(_batch())
    assert changed_prediction["S_pred"].item() != first["S_pred"].item()
    assert changed_prediction["S_pop"].item() == pytest.approx(first["S_pop"].item())
    bank.fit(torch.tensor([[10.0, 0.0], [12.0, 0.0]]), [SampleLabel.NORMAL, SampleLabel.NORMAL])
    changed_bank = inference.score_batch(_batch())
    assert changed_bank["S_pop"].item() != first["S_pop"].item()
    assert changed_bank["S_pred"].item() == pytest.approx(changed_prediction["S_pred"].item())


def test_reference_bank_is_normal_only_and_handles_k_boundaries() -> None:
    bank = NormalReferenceBank(k=10)
    with pytest.raises(ValueError, match="abnormal"):
        bank.fit(torch.ones(2, 3), [SampleLabel.NORMAL, SampleLabel.ABNORMAL])
    bank.fit(torch.zeros(2, 3), [SampleLabel.NORMAL, SampleLabel.NORMAL])
    scores = bank.score(torch.ones(1, 3))
    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()
    with pytest.raises(ValueError, match="fitted"):
        NormalReferenceBank().score(torch.ones(1, 3))


def test_inference_localization_and_independent_thresholds() -> None:
    bank = NormalReferenceBank().fit(torch.zeros(2, 2))
    inference = RepresentationInference(_StaticModel(), bank, Patchifier(PatchConfig(patch_size=3, stride=2)))
    result = inference.score_batch(_batch())
    assert result["timestep_scores"][0].tolist() == [1.0, 1.0, 0.5, 0.0]
    thresholds = inference.thresholds({"S_pred": torch.tensor([1.0, 2.0, 10.0]), "S_pop": torch.tensor([2.0, 2.0, 4.0])})
    assert set(thresholds) == {"S_pred_threshold", "S_pop_threshold"}
    assert mad_threshold(torch.tensor([1.0, 1.0])) == 1.0
