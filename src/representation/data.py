"""Lazy full-file loading and minibatch collation for V1."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from representation.config import V1Config
from representation.contracts import RepresentationBatch, validate_batch
from synth.config import MaskingConfig
from synth.dataset import iter_materialized
from synth.patchify import Patchifier
from synth.schema import FileSample

SampleSource = (
    Iterable[FileSample]
    | Callable[[], Iterable[FileSample]]
    | str
    | Path
)


class FileDataset(IterableDataset):
    """Lazily yield complete ``FileSample`` objects.

    A path points at a Sprint 1 materialized dataset and is read through
    ``iter_materialized``. A callable is preferred for one-shot generators so
    each epoch can obtain a fresh iterator. No shuffling or label filtering is
    performed here; deterministic ordering belongs to the source.
    """

    def __init__(self, source: SampleSource, split: str | None = None) -> None:
        super().__init__()
        self.source = source
        self.split = split

    def _iter_source(self) -> Iterator[FileSample]:
        if isinstance(self.source, (str, Path)):
            yield from iter_materialized(self.source, self.split)
            return
        source = self.source() if callable(self.source) else self.source
        yield from source

    def __iter__(self) -> Iterator[FileSample]:
        worker = get_worker_info()
        for index, sample in enumerate(self._iter_source()):
            if worker is not None and index % worker.num_workers != worker.id:
                continue
            yield sample


def collate_variable_files(
    samples: Sequence[FileSample],
    patchifier: Patchifier,
    *,
    masking_config: V1Config | MaskingConfig | None = None,
    masking_seed: int | None = None,
) -> RepresentationBatch:
    """Pad complete files only within a minibatch and attach patch metadata.

    The returned tensors use ``B`` files, ``C`` channels, ``T`` batch-maximum
    timesteps, ``N`` batch-maximum patches, and the patchifier's ``W``. Files
    with different channel counts cannot share a tensor batch and are rejected;
    C=3 and C=6 are each supported in separate batches.
    """
    if not samples:
        raise ValueError("cannot collate an empty sample sequence")

    for sample in samples:
        sample.validate()
    channels = {sample.C for sample in samples}
    if len(channels) != 1:
        raise ValueError("all files in a batch must have the same channel count")
    channel_count = channels.pop()

    patch_batches = [patchifier.patchify(sample) for sample in samples]
    width = patchifier.cfg.patch_size
    max_timesteps = max(sample.T for sample in samples)
    max_patches = max(1, max(batch.N for batch in patch_batches))
    batch_size = len(samples)

    signals = torch.zeros((batch_size, channel_count, max_timesteps), dtype=torch.float32)
    file_valid_mask = torch.zeros((batch_size, max_timesteps), dtype=torch.bool)
    patches = torch.zeros(
        (batch_size, max_patches, channel_count, width), dtype=torch.float32
    )
    patch_valid_mask = torch.zeros((batch_size, max_patches), dtype=torch.bool)
    patch_pad_mask = torch.ones((batch_size, max_patches, width), dtype=torch.bool)
    starts = torch.full((batch_size, max_patches), -1, dtype=torch.int64)
    valid_len = torch.zeros((batch_size, max_patches), dtype=torch.int64)

    for batch_index, (sample, patch_batch) in enumerate(zip(samples, patch_batches)):
        length = sample.T
        signals[batch_index, :, :length] = torch.from_numpy(sample.x.copy())
        file_valid_mask[batch_index, :length] = True
        count = patch_batch.N
        if count == 0:
            continue
        patches[batch_index, :count] = torch.from_numpy(patch_batch.patches.copy())
        patch_pad_mask[batch_index, :count] = torch.from_numpy(patch_batch.pad_mask.copy())
        starts[batch_index, :count] = torch.from_numpy(patch_batch.starts.copy())
        valid_len[batch_index, :count] = torch.from_numpy(patch_batch.valid_len.copy())
        patch_valid_mask[batch_index, :count] = torch.from_numpy(
            ~patch_batch.pad_mask.all(axis=1)
        )

    # Task 3 applies the configured policy when a masking config is supplied.
    # The default remains an explicit all-visible placeholder for data-only use.
    mask = torch.zeros((batch_size, max_patches), dtype=torch.bool)
    batch: RepresentationBatch = {
        "signals": signals,
        "file_valid_mask": file_valid_mask,
        "patches": patches,
        "patch_valid_mask": patch_valid_mask,
        "patch_pad_mask": patch_pad_mask,
        "starts": starts,
        "valid_len": valid_len,
        "mask": mask,
        "file_ids": [sample.file_id for sample in samples],
    }
    validate_batch(batch)
    batch["file_labels"] = [sample.file_label for sample in samples]
    batch["anomaly_meta"] = [sample.anomaly_meta for sample in samples]
    batch["anomaly_masks"] = [sample.anomaly_mask for sample in samples]
    batch["file_samples"] = list(samples)
    if masking_config is not None:
        from representation.masking import apply_batched_masking

        batch = apply_batched_masking(
            batch,
            masking_config,
            seed=masking_seed,
        )
    return batch
