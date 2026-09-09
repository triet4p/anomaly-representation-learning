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
    "FactoryScheduler",
    "ScheduledSignalGenerator",
    "SignalConfig",
    "FactoryHealth",
    "SchedulerConfig",
    "RouteConfig",
    "RouteStageConfig",
    "HealthConfig",
    "FactoryCalendarConfig",
    "SUPPORTED_CHANNEL_COUNTS",
    "DatasetBuilder",
    "iter_materialized",
    "OperationEvent",
    "OperatingContext",
    "RobotHealthState",
    "HealthEpisode",
    "DegradationStage",
    "EpisodeKind",
    "ObservableAnomalyLabels",
    "FutureFailureTargets",
    "SplitProvenance",
    "FactoryProvenance",
    "FailureEvent",
    "MODEL_INPUT_FIELD_NAMES",
    "DIAGNOSTIC_ONLY_FIELD_NAMES",
    "ALLOWED_SPLIT_VIEWS",
    "SECONDS_PER_DAY",
    "SECONDS_PER_WEEK",
    "TemporalAnomalyProcess",
    "TemporalAnomalies",
    "TemporalAnomalyConfig",
    "ChronologicalSplitter",
    "ChronologicalSplits",
    "materialize_chronological",
    "load_chronological",
    "client_config",
    "server_config",
    "sprint13_history_config",
    "sprint13_v41_history_config",
    "sprint14_v3_history_config",
    "write_seal",
    "verify_seal",
]

_EXPORT_MODULES = {
    "FileSample": "synth.schema",
    "AnomalyMeta": "synth.schema",
    "RegimeMeta": "synth.schema",
    "SampleLabel": "synth.schema",
    "OperationEvent": "synth.schema",
    "OperatingContext": "synth.schema",
    "RobotHealthState": "synth.schema",
    "HealthEpisode": "synth.schema",
    "DegradationStage": "synth.schema",
    "EpisodeKind": "synth.schema",
    "ObservableAnomalyLabels": "synth.schema",
    "FutureFailureTargets": "synth.schema",
    "SplitProvenance": "synth.schema",
    "FactoryProvenance": "synth.schema",
    "FailureEvent": "synth.schema",
    "MODEL_INPUT_FIELD_NAMES": "synth.schema",
    "DIAGNOSTIC_ONLY_FIELD_NAMES": "synth.schema",
    "ALLOWED_SPLIT_VIEWS": "synth.schema",
    "SECONDS_PER_DAY": "synth.schema",
    "SECONDS_PER_WEEK": "synth.schema",
    "SessionGenerator": "synth.generator",
    "ScheduledSignalGenerator": "synth.scheduled",
    "FactoryScheduler": "synth.scheduler",
    "FactorySchedule": "synth.scheduler",
    "RobotHealthProcess": "synth.health",
    "FactoryHealth": "synth.health",
    "SchedulerConfig": "synth.config",
    "RouteConfig": "synth.config",
    "RouteStageConfig": "synth.config",
    "HealthConfig": "synth.config",
    "SignalConfig": "synth.config",
    "CohortConfig": "synth.config",
    "SynthConfig": "synth.config",
    "FactoryCalendarConfig": "synth.config",
    "SUPPORTED_CHANNEL_COUNTS": "synth.config",
    "TemporalAnomalyProcess": "synth.temporal",
    "TemporalAnomalies": "synth.temporal",
    "TemporalAnomalyConfig": "synth.config",
    "ChronologicalSplitter": "synth.splits",
    "ChronologicalSplits": "synth.splits",
    "materialize_chronological": "synth.chronicle",
    "load_chronological": "synth.chronicle",
    "client_config": "synth.chronicle",
    "server_config": "synth.chronicle",
    "sprint13_history_config": "synth.chronicle",
    "sprint13_v41_history_config": "synth.chronicle",
    "sprint14_v3_history_config": "synth.chronicle",
    "write_seal": "synth.chronicle",
    "verify_seal": "synth.chronicle",
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
    from synth.config import SUPPORTED_CHANNEL_COUNTS, FactoryCalendarConfig, SynthConfig
    from synth.config import CohortConfig, RouteConfig, RouteStageConfig, SchedulerConfig, HealthConfig
    from synth.config import SignalConfig, TemporalAnomalyConfig
    from synth.temporal import TemporalAnomalies, TemporalAnomalyProcess
    from synth.scheduler import FactorySchedule, FactoryScheduler
    from synth.health import FactoryHealth, RobotHealthProcess
    from synth.scheduled import ScheduledSignalGenerator
    from synth.splits import ChronologicalSplitter, ChronologicalSplits
    from synth.chronicle import client_config, load_chronological
    from synth.chronicle import materialize_chronological, server_config
    from synth.chronicle import sprint13_history_config, verify_seal, write_seal
    from synth.chronicle import sprint13_v41_history_config
    from synth.chronicle import sprint14_v3_history_config
    from synth.generator import SessionGenerator
    from synth.schema import AnomalyMeta, FileSample, RegimeMeta, SampleLabel
    from synth.schema import DegradationStage, EpisodeKind, FactoryProvenance
    from synth.schema import ALLOWED_SPLIT_VIEWS, DIAGNOSTIC_ONLY_FIELD_NAMES
    from synth.schema import FailureEvent
    from synth.schema import FutureFailureTargets, HealthEpisode, ObservableAnomalyLabels
    from synth.schema import OperatingContext, OperationEvent, RobotHealthState, SplitProvenance
    from synth.schema import MODEL_INPUT_FIELD_NAMES, SECONDS_PER_DAY, SECONDS_PER_WEEK
