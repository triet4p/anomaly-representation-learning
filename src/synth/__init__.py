"""Synthetic data-generation subsystem for anomaly representation learning.

Generation modules are loaded lazily so metadata-only CLI operations do not
initialize the generation stack.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "FileSample",
    "AnomalyMeta",
    "RegimeMeta",
    "SampleLabel",
    "SessionGenerator",
    "SynthConfig",
    "SUPPORTED_CHANNEL_COUNTS",
    "DatasetBuilder",
    "iter_materialized",
]

_EXPORT_MODULES = {
    "FileSample": "synth.schema",
    "AnomalyMeta": "synth.schema",
    "RegimeMeta": "synth.schema",
    "SampleLabel": "synth.schema",
    "SessionGenerator": "synth.generator",
    "SynthConfig": "synth.config",
    "SUPPORTED_CHANNEL_COUNTS": "synth.config",
    "DatasetBuilder": "synth.dataset",
    "iter_materialized": "synth.dataset",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from synth.config import SUPPORTED_CHANNEL_COUNTS, SynthConfig
    from synth.dataset import DatasetBuilder, iter_materialized
    from synth.generator import SessionGenerator
    from synth.schema import AnomalyMeta, FileSample, RegimeMeta, SampleLabel
