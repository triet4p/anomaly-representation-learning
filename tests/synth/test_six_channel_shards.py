from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from synth.config import SynthConfig
from synth.dataset import DatasetBuilder, DatasetGenerationError, iter_materialized
from synth.generator import SessionGenerator
from synth.schema import (
    AnomalyFamily, AnomalyMeta, FileSample, RegimeMeta, RegimeType, SampleLabel,
)


HARD_FAMILIES = [f for f in AnomalyFamily if not f.value.startswith("easy_")]


def test_default_physics_is_coherent_six_channels():
    cfg = SynthConfig()
    assert cfg.n_channels == 6
    sample = SessionGenerator(cfg).generate_normal(11)
    sample.validate()
    assert sample.x.shape[0] == 6
    corr = np.corrcoef(sample.x.astype(float))
    assert np.mean(np.abs(corr[0, 1:])) > 0.15
    assert np.mean(np.abs(corr[1, 3:])) > 0.15


@pytest.mark.parametrize("family", HARD_FAMILIES)
def test_every_hard_family_validates_at_c6(family: AnomalyFamily):
    sample = SessionGenerator(SynthConfig()).generate_anomaly(
        200 + HARD_FAMILIES.index(family), family=family, severity=0.7
    )
    sample.validate()
    assert sample.file_label == SampleLabel.ABNORMAL
    assert sample.anomaly_mask is not None and sample.anomaly_mask.shape == sample.x.shape
    assert sample.anomaly_meta is not None and sample.anomaly_meta.family == family

def test_shards_exact_counts_and_clean_resume_match(tmp_path: Path):
    builder = DatasetBuilder(SynthConfig())
    clean, resumed = tmp_path / "clean", tmp_path / "resumed"
    clean_manifest = builder.materialize_sharded(
        clean, n_train=4, n_val=4, n_test=4, shard_size=2
    )
    builder.materialize_sharded(resumed, n_train=4, n_val=4, n_test=4, shard_size=2)
    builder.materialize_sharded(
        resumed, n_train=4, n_val=4, n_test=4, shard_size=2, resume=True
    )
    assert [s.file_id for s in iter_materialized(clean)] == [s.file_id for s in iter_materialized(resumed)]
    assert len(list(iter_materialized(clean))) == 12
    assert json.loads((resumed / "manifest.json").read_text())["format"] == 2
    for split, entry in clean_manifest["splits"].items():
        assert len(list((clean / split).glob("*.zip"))) == 2
        assert entry["accepted_normal"] == (4 if split == "train" else 2)
        assert entry["accepted_abnormal"] == (0 if split == "train" else 2)
    for split in ("train", "val", "test"):
        clean_archives = sorted((clean / split).glob("*.zip"))
        resumed_archives = sorted((resumed / split).glob("*.zip"))
        assert [p.read_bytes() for p in clean_archives] == [p.read_bytes() for p in resumed_archives]


def test_archive_corruption_is_detected(tmp_path: Path):
    builder = DatasetBuilder(SynthConfig())
    builder.materialize_sharded(tmp_path, n_train=2, n_val=2, n_test=2, shard_size=1)
    archive_path = next((tmp_path / "train").glob("*.zip"))
    archive_path.write_bytes(archive_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        list(iter_materialized(tmp_path, "train"))


def test_resume_rejects_configuration_change(tmp_path: Path):
    DatasetBuilder(SynthConfig()).materialize_sharded(
        tmp_path, n_train=2, n_val=2, n_test=2, shard_size=1
    )
    with pytest.raises(ValueError, match="incompatible"):
        DatasetBuilder(SynthConfig(n_channels=3)).materialize_sharded(
            tmp_path, n_train=2, n_val=2, n_test=2, shard_size=1, resume=True
        )


def _stub_sample(label: SampleLabel, family: AnomalyFamily, attempts: int, seed: int) -> FileSample:
    x = np.ones((6, 4), dtype=np.float32) if label == SampleLabel.ABNORMAL else np.zeros((6, 4), dtype=np.float32)
    mask = np.ones_like(x, dtype=bool) if label == SampleLabel.ABNORMAL else None
    meta = AnomalyMeta(family, 1, 3, 0.7, list(range(6)), extra={"n_attempts": attempts})
    return FileSample(
        x=x, file_id=f"{label.value}-{seed}", file_label=label, seed=seed,
        generator_version="stub", config_hash="stub", regime_sequence=[
            RegimeMeta(RegimeType.ACTIVE, 0, 4, 0.5)
        ], anomaly_meta=meta if label == SampleLabel.ABNORMAL else None,
        anomaly_mask=mask, rejection_meta=meta if label == SampleLabel.NORMAL else None,
    )


def test_outer_retry_provenance_is_canonical():
    builder = DatasetBuilder(SynthConfig())
    calls = iter([
        _stub_sample(SampleLabel.NORMAL, AnomalyFamily.MISSING_EVENT, 2, 1),
        _stub_sample(SampleLabel.ABNORMAL, AnomalyFamily.MISSING_EVENT, 3, 2),
    ])
    builder.gen.generate_anomaly = lambda seed, split, family: next(calls)
    sample = builder._sample_at("test", 1, 1.0, 0, [AnomalyFamily.MISSING_EVENT])
    assert sample.anomaly_meta is not None
    assert sample.anomaly_meta.extra["dataset_rejected_attempts"] == 4
    assert len(sample.anomaly_meta.extra["dataset_retry_rejections"]) == 1


def test_manifest_uses_canonical_retry_count(tmp_path: Path):
    builder = DatasetBuilder(SynthConfig())
    calls = iter([
        _stub_sample(SampleLabel.NORMAL, AnomalyFamily.MISSING_EVENT, 2, 1),
        _stub_sample(SampleLabel.ABNORMAL, AnomalyFamily.MISSING_EVENT, 3, 2),
    ])
    builder.gen.generate_anomaly = lambda seed, split, family: next(calls)
    manifest = builder.materialize_sharded(
        tmp_path, n_train=1, n_val=0, n_test=0, shard_size=1, contamination=1.0
    )
    entry = manifest["splits"]["train"]
    assert entry["rejection_attempt_total"] == 4
    assert entry["rejection_attempts_by_family"]["missing_event"] == 4


def test_retry_exhaustion_records_failure_manifest(tmp_path: Path):
    cfg = SynthConfig()
    cfg.anomaly.max_rejection_attempts = 2
    builder = DatasetBuilder(cfg)
    builder.gen.generate_anomaly = lambda seed, split, family: _stub_sample(
        SampleLabel.NORMAL, family, 2, seed
    )
    with pytest.raises(DatasetGenerationError) as raised:
        builder.materialize_sharded(
            tmp_path, n_train=1, n_val=0, n_test=0, shard_size=1, contamination=1.0
        )
    assert raised.value.provenance and len(raised.value.provenance) == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    failure = manifest["splits"]["train"]["failure"]
    assert failure["index"] == 0 and len(failure["provenance"]) == 2



def test_interrupted_later_shard_resumes_prior_archive(tmp_path: Path):
    builder = DatasetBuilder(SynthConfig())
    original = builder._sample_at
    state = {"fail": True}
    calls: list[int] = []

    def maybe_fail(split, n, ratio, index, fams):
        calls.append(index)
        if state["fail"] and index >= 2:
            raise DatasetGenerationError(split, index, AnomalyFamily.MISSING_EVENT, [])
        return original(split, n, ratio, index, fams)

    builder._sample_at = maybe_fail
    with pytest.raises(DatasetGenerationError):
        builder.materialize_sharded(tmp_path, n_train=4, n_val=0, n_test=0, shard_size=2)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["splits"]["train"]["shards"]) == 1
    first_archive = next((tmp_path / "train").glob("*.zip"))
    first_bytes = first_archive.read_bytes()
    state["fail"] = False
    calls.clear()
    builder.materialize_sharded(
        tmp_path, n_train=4, n_val=0, n_test=0, shard_size=2, resume=True
    )
    assert first_archive.read_bytes() == first_bytes
    assert calls == [2, 3]
    assert len(json.loads((tmp_path / "manifest.json").read_text())["splits"]["train"]["shards"]) == 2