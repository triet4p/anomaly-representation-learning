from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from representation.data import collate_variable_files
from representation.inference import NormalReferenceBank, RepresentationInference, mad_threshold
from representation.model import V1RepresentationModel
from representation.config import V1Config
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import FileSample, RegimeMeta, RegimeType, SampleLabel


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


def test_prepare_reference_bank_prefers_restored_over_refit() -> None:
    """A restored bank must be kept by default and refit only on opt-in."""
    from representation.inference import prepare_reference_bank

    restored = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    replacement = torch.tensor([[10.0, 0.0], [11.0, 0.0]])
    bank = NormalReferenceBank(k=2).fit(restored)
    assert prepare_reference_bank(bank, replacement) == "restored"
    assert torch.equal(bank.embeddings, restored)
    assert prepare_reference_bank(bank, replacement, refit=True) == "refit"
    assert torch.equal(bank.embeddings, replacement)
    fresh = NormalReferenceBank(k=2)
    assert prepare_reference_bank(fresh, replacement) == "refit"
    assert torch.equal(fresh.embeddings, replacement)


def test_score_batch_never_mutates_the_scoring_bank() -> None:
    """Restore-then-score must score against the restored embeddings."""
    bank = NormalReferenceBank(k=2).fit(torch.tensor([[0.0, 0.0], [2.0, 0.0]]))
    before = bank.embeddings.clone()
    inference = RepresentationInference(_StaticModel(), bank, Patchifier(PatchConfig(patch_size=3, stride=2)))
    result = inference.score_batch(_batch())
    assert torch.equal(bank.embeddings, before)
    # Query [1,0] against restored rows [0,0],[2,0]: mean distance to both is 1.0.
    assert result["S_pop"].item() == pytest.approx(1.0)


def test_timestep_scores_localize_synthetic_spike_deterministically() -> None:
    """A spike anomaly must produce a non-degenerate trace peaking in its window."""
    torch.manual_seed(0)
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    model.eval()
    signal = np.zeros((3, 40), dtype=np.float32)
    signal[:, 16:24] = 5.0
    sample = FileSample(
        x=signal, file_id="spike", file_label=SampleLabel.NORMAL, seed=1,
        generator_version="test", config_hash="test",
        regime_sequence=[RegimeMeta(RegimeType.ACTIVE, 0, 40, 0.5)],
    )
    sample.validate()
    bank = NormalReferenceBank(k=1).fit(torch.zeros(1, config.d_model))
    inference = RepresentationInference(model, bank, patchifier, masking_config=config)
    batch = collate_variable_files([sample], patchifier, masking_config=config, masking_seed=0)
    batch.pop("file_samples", None)
    first = inference.score_batch(batch)["timestep_scores"][0]
    second = inference.score_batch(batch)["timestep_scores"][0]
    assert len(first) == 40
    assert np.array_equal(np.asarray(first), np.asarray(second))
    peak = int(np.argmax(np.asarray(first, dtype=float)))
    assert 16 <= peak < 24, f"trace peak {peak} outside spike window"
    background = np.asarray(first, dtype=float)[:16]
    assert bool((background == 0.0).all())
    assert float(np.asarray(first, dtype=float)[16:24].mean()) > 0.0
