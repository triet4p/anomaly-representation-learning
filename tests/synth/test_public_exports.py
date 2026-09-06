"""Focused test for the public `synth` export table (Batch A2 re-gate)."""

from __future__ import annotations

import ast
from pathlib import Path

import synth


def test_every_all_entry_resolves_by_name():
    for name in synth.__all__:
        assert getattr(synth, name) is not None, name


def test_every_all_entry_has_module_mapping():
    for name in synth.__all__:
        assert name in synth._EXPORT_MODULES, name


def test_export_map_has_no_duplicate_keys():
    source = (
        Path(__file__).resolve().parents[2]
        / "src" / "synth" / "__init__.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == (
            "_EXPORT_MODULES"
        ):
            keys = [
                key.value
                for key in node.value.keys  # type: ignore[attr-defined]
            ]
            assert len(keys) == len(set(keys)), "duplicate export mapping"
            assert set(synth.__all__) <= set(keys)


def test_wildcard_import_covers_all():
    namespace: dict[str, object] = {}
    exec("from synth import *", namespace)
    for name in synth.__all__:
        assert name in namespace, name


def test_batch_a2_names_importable():
    assert synth.TemporalAnomalyProcess is not None
    assert synth.SUPPORTED_CHANNEL_COUNTS == (3, 6)
    assert synth.ChronologicalSplitter is not None
    assert synth.ChronologicalSplits is not None
