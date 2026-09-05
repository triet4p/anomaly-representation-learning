"""End-to-end Stage 1 finalization regression.

Covers the Kaggle failure ``NameError: name '_artifact_info' is not defined``:
``BoundedEmbeddingExtractor.extract`` built every artifact, then crashed while
assembling the ``files`` mapping because the artifact-metadata helper was never
defined. This test drives the public ``extract_embeddings(config)`` path on a
tiny coherent CPU fixture (real generated dataset + real checkpoint) through
successful final publication, asserting the complete output inventory, manifest
linkage/checksums/record count, and returned paths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from representation.checkpoint import save_checkpoint
from representation.config import V1Config
from representation.data import FileDataset, collate_variable_files
from representation.embedding_extraction import extract_embeddings
from representation.geometry import DiagnosticConfig
from representation.inference import NormalReferenceBank
from representation.model import V1RepresentationModel
from synth.config import PatchConfig, SynthConfig
from synth.dataset import DatasetBuilder
from synth.patchify import Patchifier


def _tiny_model() -> tuple[V1Config, Patchifier, V1RepresentationModel]:
    config = V1Config(
        n_channels=6,
        patch_size=8,
        stride=8,
        d_model=8,
        attention_heads=2,
        sequence_layers=1,
        dropout=0.0,
        n_robots=5,
        n_programs=8,
        min_bucket_samples=1,
    )
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    return config, patchifier, V1RepresentationModel(config, patchifier=patchifier)


def test_extract_embeddings_publishes_complete_cache(tmp_path: Path) -> None:
    """Public extraction must publish the full inventory with linked manifest."""
    torch.manual_seed(0)
    data_root = tmp_path / "data"
    DatasetBuilder(SynthConfig()).materialize_sharded(
        data_root, n_train=4, n_val=2, n_test=4, shard_size=4
    )

    config, patchifier, model = _tiny_model()
    model.eval()
    train_samples = list(FileDataset(data_root, split="train"))[:2]
    assert len(train_samples) == 2
    batch = collate_variable_files(train_samples, patchifier)
    batch.pop("file_samples", None)
    with torch.inference_mode():
        reference = model(batch)["file_embedding"]
    bank = NormalReferenceBank(k=2).fit(reference)

    checkpoint_path = tmp_path / "tiny_representation.pt"
    save_checkpoint(checkpoint_path, model, step=3, reference_bank=bank)

    output_dir = tmp_path / "geometry-cache"
    result = extract_embeddings(
        DiagnosticConfig(
            dataset_root=data_root,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            splits=("test",),
            reference_split="train",
            max_samples=4,
            max_reference_samples=8,
            batch_size=2,
            seed=0,
            sampling="head",
            device="cpu",
        )
    )

    records = result["records"]
    assert len(records) == 4
    assert all(row.split == "test" for row in records)

    paths = result["paths"]
    assert paths.root == output_dir
    for name in ("embeddings.npz", "records.csv", "metrics.json", "neighbors.csv", "geometry-manifest.json"):
        assert (output_dir / name).is_file(), name
    assert (output_dir / "figures").is_dir()

    payload = np.load(output_dir / "embeddings.npz", allow_pickle=False)
    assert payload["embeddings"].shape == (4, config.d_model)
    assert np.array_equal(payload["row_index"], np.arange(4, dtype=np.int64))
    assert payload["S_pred"].shape == (4,) and payload["S_pop"].shape == (4,)
    assert np.isfinite(payload["embeddings"]).all()

    manifest_raw = json.loads((output_dir / "geometry-manifest.json").read_text(encoding="utf-8"))
    assert manifest_raw["records_count"] == 4
    assert manifest_raw["splits"] == {"test": 4}
    assert manifest_raw["feature_dim"] == config.d_model
    assert set(manifest_raw["files"]) == {"embeddings.npz", "records.csv", "metrics.json", "neighbors.csv"}
    for name, entry in manifest_raw["files"].items():
        data = (output_dir / name).read_bytes()
        assert entry["bytes"] == len(data)
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()

    manifest = result["manifest"]
    assert manifest.records_count == 4
    assert paths.manifest.is_file()
