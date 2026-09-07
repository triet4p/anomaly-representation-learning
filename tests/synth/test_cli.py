from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from synth import cli


ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTION_NAME = "anomaly-representation-learning"


def _version_environment() -> dict[str, str]:
    env = os.environ.copy()
    source_path = str(ROOT / "src")
    env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_module_version_matches_package_metadata_without_side_effects(tmp_path: Path):
    output_dir = tmp_path / "should-not-be-created"
    result = subprocess.run(
        [sys.executable, "-m", "synth.cli", "--version", "--output", str(output_dir)],
        cwd=ROOT,
        env=_version_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == importlib.metadata.version(DISTRIBUTION_NAME)
    assert result.stderr == ""
    assert not output_dir.exists()


def test_installed_entry_point_version_matches_module_and_has_no_side_effects(tmp_path: Path):
    entry_point = shutil.which("synth-generate")
    if entry_point is None:
        pytest.skip("synth-generate is available only when the project is installed")

    output_dir = tmp_path / "should-not-be-created"
    result = subprocess.run(
        [entry_point, "--version", "--output", str(output_dir)],
        cwd=ROOT,
        env=_version_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == importlib.metadata.version(DISTRIBUTION_NAME)
    assert result.stderr == ""
    assert not output_dir.exists()


def test_normal_generation_argument_parsing_and_setup_are_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    split = SimpleNamespace(train_seed=1, val_seed=2, test_seed=3, n_train=10, n_val=5, n_test=5)
    config_calls: list[int] = []
    materialize_calls: list[dict[str, object]] = []

    class FakeConfig:
        def __init__(self, *, n_channels: int):
            config_calls.append(n_channels)
            self.split = split

    class FakeBuilder:
        def __init__(self, config: FakeConfig):
            assert config.split is split

        def materialize_sharded(self, output: Path, **kwargs: object):
            materialize_calls.append({"output": output, **kwargs})
            return {"counts": {"train": 4, "val": 4, "test": 4}}

    monkeypatch.setattr(cli, "_load_generation_dependencies", lambda: (FakeConfig, FakeBuilder))

    output_dir = tmp_path / "dataset"
    assert cli.main(
        [
            "--output",
            str(output_dir),
            "--channels",
            "3",
            "--seed",
            "7",
            "--count",
            "4",
            "--shard-size",
            "2",
            "--resume",
            "--overwrite",
            "--small",
        ]
    ) == 0

    assert config_calls == [3]
    assert split.train_seed == 7
    assert split.val_seed == 1_000_007
    assert split.test_seed == 2_000_007
    assert (split.n_train, split.n_val, split.n_test) == (4, 4, 4)
    assert materialize_calls == [
        {
            "output": output_dir,
            "shard_size": 2,
            "resume": True,
            "overwrite": True,
            "profile": "small",
        }
    ]

def test_chronological_units_preserves_arrival_span():
    """--units rescales cadence inversely, keeping the calendar span fixed."""
    from synth.chronicle import server_config

    args = cli.build_parser().parse_args([
        "--chronological", "--profile", "server", "--units", "1200",
        "--output", "dummy",
    ])
    assert args.units == 1200
    cfg = server_config(seed=100)
    base_span = cfg.scheduler.n_units * cfg.scheduler.arrival_interval_s
    cfg.scheduler.n_units = args.units
    scaled = base_span / args.units
    assert scaled == 5400.0
    assert base_span == 300 * 21600.0


def test_chronological_units_rejects_nonpositive():
    with __import__("pytest").raises(SystemExit):
        cli.main([
            "--chronological", "--profile", "server", "--units", "0",
            "--output", "dummy",
        ])
