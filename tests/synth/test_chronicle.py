"""Focused test for chronological materialization and manifest (Task 7)."""

from __future__ import annotations

import json

from synth import cli as synth_cli

import numpy as np
import pytest

from synth.chronicle import (
    CHRONICLE_FORMAT,
    client_config,
    load_chronological,
    materialize_chronological,
    server_config,
)
from synth.config import FactoryCalendarConfig, SchedulerConfig, SynthConfig
from synth.schema import SampleLabel


def _tiny_config() -> SynthConfig:
    cfg = SynthConfig()
    cfg.factory = FactoryCalendarConfig(
        span_days=90.0, dev_cutoff_days=45.0, quarantine_days=7.0, seed=11
    )
    cfg.scheduler = SchedulerConfig(
        n_units=8, arrival_interval_s=86400.0, seed=11
    )
    return cfg


def _decoded_equal(first, second) -> bool:
    if first.file_id != second.file_id:
        return False
    if first.file_label is not second.file_label:
        return False
    if not np.array_equal(first.x, second.x):
        return False
    if (first.anomaly_mask is None) != (second.anomaly_mask is None):
        return False
    if first.anomaly_mask is not None and not np.array_equal(
        first.anomaly_mask, second.anomaly_mask
    ):
        return False
    return (
        first.anomaly_labels == second.anomaly_labels
        and first.future_targets == second.future_targets
        and first.split_provenance == second.split_provenance
        and first.health == second.health
    )


def test_tiny_materialization_round_trip(tmp_path):
    root = tmp_path / "chrono"
    manifest = materialize_chronological(
        _tiny_config(), root, shard_size=8
    )
    assert manifest["format"] == CHRONICLE_FORMAT
    assert manifest["counts"]["total"] == 16
    assert manifest["seeds"]["scheduler"] == 11
    assert manifest["calendar"]["cutoff_time"] == pytest.approx(
        45.0 * 86400.0
    )
    assert manifest["schedule"] and manifest["splits"]
    assert manifest["shards"]
    assert (root / "manifest.json").is_file()

    samples, reloaded_manifest = load_chronological(root)
    assert reloaded_manifest["config_hash"] == manifest["config_hash"]
    assert len(samples) == manifest["counts"]["total"]
    assert [s.file_id for s in samples] == [
        row["file_id"] for row in manifest["files"]
    ]
    for sample in samples:
        sample.validate()
        sample.encoder_inputs()
    assert sum(
        1 for s in samples if s.file_label is SampleLabel.ABNORMAL
    ) == manifest["counts"]["abnormal"]


def test_materialization_is_deterministic_decoded(tmp_path):
    first_root = tmp_path / "a"
    second_root = tmp_path / "b"
    first_manifest = materialize_chronological(
        _tiny_config(), first_root, shard_size=8
    )
    second_manifest = materialize_chronological(
        _tiny_config(), second_root, shard_size=8
    )
    for key in (
        "format", "generator_version", "config_hash", "resolved_config",
        "seeds", "counts", "calendar", "schedule", "episodes", "splits",
        "files",
    ):
        assert second_manifest[key] == first_manifest[key]
    first, _ = load_chronological(first_root)
    second, _ = load_chronological(second_root)
    assert len(first) == len(second)
    assert all(_decoded_equal(a, b) for a, b in zip(first, second))


def test_materialization_fails_fast(tmp_path):
    root = tmp_path / "chrono"
    materialize_chronological(_tiny_config(), root, shard_size=8)
    with pytest.raises(FileExistsError):
        materialize_chronological(_tiny_config(), root, shard_size=8)
    with pytest.raises((ValueError, FileNotFoundError)):
        materialize_chronological(_tiny_config(), "", shard_size=8)
    with pytest.raises(ValueError):
        materialize_chronological(
            _tiny_config(), tmp_path / "other", shard_size=0
        )
    with pytest.raises(FileNotFoundError):
        load_chronological(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        load_chronological(tmp_path)


def test_load_detects_corrupt_shard(tmp_path):
    root = tmp_path / "chrono"
    materialize_chronological(_tiny_config(), root, shard_size=8)
    manifest = json.loads((root / "manifest.json").read_text())
    shard_path = root / manifest["shards"][0]["path"]
    shard_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        load_chronological(root)


def test_client_and_server_profiles_are_multi_robot():
    client = client_config(seed=0)
    robots = {
        stage.robot_id
        for route in client.scheduler.routes
        for stage in route.stages
    }
    assert len(robots) >= 3
    assert {r.route_id for r in client.scheduler.routes} >= {
        "route-A", "route-B",
    }
    server = server_config(seed=0)
    assert server.scheduler.n_units > client.scheduler.n_units
    assert server.hash() != client.hash()


def test_cli_chronological_client_profile(tmp_path):
    """The public CLI materializes the routed client dataset end to end."""
    root = tmp_path / "client"
    rc = synth_cli.main([
        "--chronological", "--profile", "client",
        "--output", str(root), "--shard-size", "32", "--overwrite",
    ])
    assert rc == 0
    samples, manifest = load_chronological(root)
    counts = manifest["counts"]
    assert counts["total"] == 120 == len(samples)
    assert counts["normal"] + counts["abnormal"] == 120
    assert manifest["splits"]["dev_train"] and manifest["splits"]["dev_val"]
    assert manifest["splits"]["test_static"] and manifest["splits"]["test_temporal"]
    assert {e["robot_id"] for e in manifest["schedule"]} == {
        "robot-01", "robot-02", "robot-03",
    }
