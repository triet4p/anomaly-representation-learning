from __future__ import annotations

import numpy as np
import pytest
import torch

from representation.config import V1Config
from representation.data import collate_variable_files
from representation.masking import apply_batched_masking
from synth.config import MaskingConfig, PatchConfig
from synth.patchify import Patchifier
from synth.schema import FileSample, RegimeMeta, RegimeType, SampleLabel


def _sample(timesteps: int, file_id: str) -> FileSample:
    x = np.arange(3 * timesteps, dtype=np.float32).reshape(3, timesteps)
    return FileSample(
        x=x,
        file_id=file_id,
        file_label=SampleLabel.NORMAL,
        seed=timesteps,
        generator_version="test",
        config_hash="test",
        regime_sequence=[RegimeMeta(RegimeType.ACTIVE, 0, timesteps, 0.5)],
    )


def _batch(timesteps: tuple[int, ...] = (65, 81)) -> dict[str, object]:
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    samples = [_sample(t, str(t)) for t in timesteps]
    return collate_variable_files(samples, patchifier)


def test_batched_masking_keeps_exact_ratio_and_composition() -> None:
    config = V1Config(
        patch_size=8,
        stride=8,
        total_mask_ratio=0.4,
        random_fraction=0.34,
        info_fraction=0.33,
        block_fraction=0.33,
    )
    batch = _batch()
    result = apply_batched_masking(batch, config, seed=5)

    for row in range(result["mask"].shape[0]):
        valid = result["patch_valid_mask"][row]
        expected = round(config.total_mask_ratio * int(valid.sum()))
        assert int(result["mask"][row].sum()) == expected
        assert not (result["mask"][row] & ~valid).any()
        assert sum(result["mask_composition"][row].values()) == expected
        assert float(result["mask_ratio"][row]) == pytest.approx(
            expected / int(valid.sum())
        )


def test_batched_masking_is_seed_reproducible() -> None:
    batch = _batch()
    config = V1Config(patch_size=8, stride=8)
    first = apply_batched_masking(batch, config, seed=11)
    second = apply_batched_masking(batch, config, seed=11)
    third = apply_batched_masking(batch, config, seed=12)

    torch.testing.assert_close(first["mask"], second["mask"])
    assert not torch.equal(first["mask"], third["mask"])
def test_masking_ablation_keeps_total_count_fixed() -> None:
    batch = _batch()
    ratio = 0.4
    policies = (
        V1Config(patch_size=8, stride=8, total_mask_ratio=ratio,
                 random_fraction=1.0, info_fraction=0.0, block_fraction=0.0),
        V1Config(patch_size=8, stride=8, total_mask_ratio=ratio,
                 random_fraction=0.0, info_fraction=1.0, block_fraction=0.0),
        V1Config(patch_size=8, stride=8, total_mask_ratio=ratio,
                 random_fraction=0.0, info_fraction=0.0, block_fraction=1.0),
    )
    counts = [
        apply_batched_masking(batch, policy, seed=9)["mask"].sum(dim=1)
        for policy in policies
    ]
    for count in counts[1:]:
        torch.testing.assert_close(count, counts[0])




def test_block_only_masking_contains_contiguous_patch_run() -> None:
    batch = _batch((97, 97))
    config = V1Config(
        patch_size=8,
        stride=8,
        total_mask_ratio=0.5,
        random_fraction=0.0,
        info_fraction=0.0,
        block_fraction=1.0,
    )
    result = apply_batched_masking(batch, config, seed=2)
    indices = torch.where(result["mask"][0])[0].tolist()
    runs: list[int] = []
    run = 1
    for left, right in zip(indices, indices[1:]):
        if right == left + 1:
            run += 1
        else:
            runs.append(run)
            run = 1
    if indices:
        runs.append(run)
    assert max(runs, default=0) >= 2
    assert result["mask_composition"][0] == {"random": 0, "info": 0, "block": len(indices)}


def test_collator_hook_applies_configured_masking() -> None:
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    config = MaskingConfig(
        total_mask_ratio=0.5,
        random_fraction=1.0,
        info_fraction=0.0,
        block_fraction=0.0,
    )
    batch = collate_variable_files(
        [_sample(10, "short"), _sample(40, "long")],
        patchifier,
        masking_config=config,
        masking_seed=3,
    )
    assert batch["mask"].any()
    assert not (batch["mask"] & ~batch["patch_valid_mask"]).any()
    assert batch["mask_composition"][0]["random"] >= 0


def test_masking_requires_file_sample_metadata() -> None:
    batch = _batch()
    batch.pop("file_samples")
    with pytest.raises(ValueError, match="file_samples"):
        apply_batched_masking(batch, V1Config(patch_size=8, stride=8), seed=0)
