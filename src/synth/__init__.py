"""Synthetic data-generation subsystem for anomaly representation learning."""
from synth.schema import FileSample, AnomalyMeta, RegimeMeta, SampleLabel
from synth.generator import SessionGenerator
from synth.config import SynthConfig, SUPPORTED_CHANNEL_COUNTS
from synth.dataset import DatasetBuilder, iter_materialized
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
