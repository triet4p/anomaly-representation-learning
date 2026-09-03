from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from representation.data import FileDataset, collate_variable_files
from synth.config import PatchConfig, SynthConfig
from synth.dataset import DatasetBuilder
from synth.patchify import Patchifier
from synth.schema import (
    AnomalyFamily,
    AnomalyMeta,
    FileSample,
    RegimeMeta,
    RegimeType,
    SampleLabel,
)


def _sample(channels: int, timesteps: int, file_id: str) -> FileSample:
    signal = np.arange(channels * timesteps, dtype=np.float32).reshape(channels, timesteps)
    sample = FileSample(
        x=signal,
        file_id=file_id,
        file_label=SampleLabel.NORMAL,
        seed=timesteps,
        generator_version="test",
        config_hash="test",
        regime_sequence=[RegimeMeta(RegimeType.ACTIVE, 0, timesteps, 0.5)],
    )
    sample.validate()
    return sample


def _anomaly_sample(channels: int = 3, timesteps: int = 25) -> FileSample:
    sample = _sample(channels, timesteps, "abnormal")
    anomaly_mask = np.zeros_like(sample.x, dtype=bool)
    anomaly_mask[:, 8:12] = True
    return FileSample(
        x=sample.x,
        file_id=sample.file_id,
        file_label=SampleLabel.ABNORMAL,
        seed=sample.seed,
        generator_version=sample.generator_version,
        config_hash=sample.config_hash,
        regime_sequence=sample.regime_sequence,
        anomaly_meta=AnomalyMeta(AnomalyFamily.REALISTIC_STUCK, 8, 12, 0.7, list(range(channels))),
        anomaly_mask=anomaly_mask,
    )


def test_collate_preserves_variable_lengths_and_patch_metadata() -> None:
    samples = [_sample(3, 21, "short"), _sample(3, 43, "long")]
    patchifier = Patchifier(PatchConfig(patch_size=16, stride=8, pad_end=True))
    batch = collate_variable_files(samples, patchifier)

    assert tuple(batch["signals"].shape) == (2, 3, 43)
    assert batch["file_valid_mask"].tolist() == [
        [True] * 21 + [False] * 22,
        [True] * 43,
    ]
    assert batch["file_ids"] == ["short", "long"]
    for index, sample in enumerate(samples):
        np.testing.assert_array_equal(
            batch["signals"][index, :, : sample.T].numpy(), sample.x
        )
        assert int(batch["valid_len"][index].max()) <= 16
        assert not (batch["mask"][index] & ~batch["patch_valid_mask"][index]).any()


def test_collate_supports_three_and_six_channel_files() -> None:
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    for channels in (3, 6):
        batch = collate_variable_files([_sample(channels, 17, str(channels))], patchifier)
        assert tuple(batch["signals"].shape) == (1, channels, 17)
        assert tuple(batch["patches"].shape[2:]) == (channels, 8)


def test_collate_preserves_evaluation_metadata_without_using_labels() -> None:
    sample = _anomaly_sample()
    batch = collate_variable_files([sample], Patchifier(PatchConfig(patch_size=8, stride=8)))

    assert batch["file_labels"] == [SampleLabel.ABNORMAL]
    assert batch["anomaly_meta"][0] is not None
    assert batch["anomaly_masks"][0] is not None
    np.testing.assert_array_equal(batch["anomaly_masks"][0], sample.anomaly_mask)
    assert not batch["mask"].any()


def test_collate_rejects_mixed_channel_count_batches() -> None:
    with pytest.raises(ValueError, match="same channel count"):
        collate_variable_files(
            [_sample(3, 20, "three"), _sample(6, 20, "six")],
            Patchifier(PatchConfig()),
        )


def test_file_dataset_is_lazy_and_reiterable_from_callable() -> None:
    samples = [_sample(3, 20, "a"), _sample(3, 22, "b")]
    calls = 0

    def source():
        nonlocal calls
        calls += 1
        yield from samples

    dataset = FileDataset(source)
    assert [sample.file_id for sample in dataset] == ["a", "b"]
    assert [sample.file_id for sample in dataset] == ["a", "b"]
    assert calls == 2


def test_materialized_sprint_one_round_trip_preserves_files(tmp_path: Path) -> None:
    config = SynthConfig()
    builder = DatasetBuilder(config)
    builder.materialize_sharded(
        tmp_path,
        n_train=2,
        n_val=0,
        n_test=0,
        shard_size=2,
    )
    original = list(builder.iter_split("train", 2))
    loaded = list(FileDataset(tmp_path, split="train"))

    assert [sample.file_id for sample in loaded] == [sample.file_id for sample in original]
    for expected, actual in zip(original, loaded):
        np.testing.assert_array_equal(actual.x, expected.x)
