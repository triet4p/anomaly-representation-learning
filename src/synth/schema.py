"""
Core data contracts for the synthetic data subsystem.

Every generated file/session is a FileSample.  Anomaly samples carry
AnomalyMeta describing the intervention exactly.  Patch operations produce
PatchBatch with explicit start indices and a timestep↔patch mask map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from synth.config import SUPPORTED_CHANNEL_COUNTS


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SampleLabel(str, Enum):
    NORMAL = "normal"
    ABNORMAL = "abnormal"


class AnomalyFamily(str, Enum):
    """Hard anomaly families required by DATA.md §4."""
    CONTEXTUAL_REPLACEMENT = "contextual_replacement"
    WRONG_TRANSITION = "wrong_transition"
    REALISTIC_STUCK = "realistic_stuck"
    OVER_REGULARITY = "over_regularity"
    SUBTLE_DRIFT = "subtle_drift"
    FREQ_PHASE_MISMATCH = "freq_phase_mismatch"
    CROSS_CHANNEL_INCONSISTENCY = "cross_channel_inconsistency"
    DURATION_ANOMALY = "duration_anomaly"
    MISSING_EVENT = "missing_event"
    # Easy sanity (not hard-benchmark defaults)
    EASY_SPIKE = "easy_spike"
    EASY_FLATLINE = "easy_flatline"


class RegimeType(str, Enum):
    IDLE = "idle"
    RAMP_UP = "ramp_up"
    ACTIVE = "active"
    RAMP_DOWN = "ramp_down"
    PERIODIC = "periodic"
    RECOVERY = "recovery"
    TRANSITION = "transition"


# ---------------------------------------------------------------------------
# Regime metadata
# ---------------------------------------------------------------------------

@dataclass
class RegimeMeta:
    """One regime segment within a session."""
    regime: RegimeType
    start: int      # inclusive timestep index
    end: int        # exclusive timestep index
    target_level: float  # normalised feed level [0, 1]
    frequency: float | None = None   # for periodic regime
    phase: float | None = None

    @property
    def duration(self) -> int:
        return self.end - self.start


# ---------------------------------------------------------------------------
# Anomaly metadata
# ---------------------------------------------------------------------------

@dataclass
class AnomalyMeta:
    """Complete provenance for one anomaly injection."""
    family: AnomalyFamily
    start: int           # inclusive timestep
    end: int             # exclusive timestep
    severity: float      # continuous [0, 1]
    affected_channels: list[int] = field(default_factory=list)
    # Per-channel intervention parameters (populated as needed)
    lag: int | None = None
    phase_shift: float | None = None
    gain_change: float | None = None
    drift_rate: float | None = None
    stuck_sigma_ratio: float | None = None
    transition_speed: float | None = None
    donor_file_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chronological factory contracts (Sprint 11 Task 1)
# ---------------------------------------------------------------------------
#
# Typed persisted contracts for the deterministic 3–6 month shared-unit
# factory calendar (methodology §20). All timestamps are float seconds
# since the factory calendar origin. These types establish validated
# configuration and per-file provenance only; scheduling algorithms,
# health simulation, signal generation, and split materialization remain
# later tasks and MUST NOT live here.
#
# Model-input separation: the encoder may condition on signal, regime,
# robot/program identity, scheduling, and operating context. Latent
# health state, episodes, observable anomaly labels, anomaly masks, and
# future-failure targets are diagnostics/supervision only and MUST NOT
# enter encoder inputs (methodology §20.9).

SECONDS_PER_DAY = 86400.0
SECONDS_PER_WEEK = 604800.0

# Split views known to the provenance contract. Split assignment itself
# is a later task; this tuple only bounds the vocabulary.
ALLOWED_SPLIT_VIEWS = ("dev-train", "dev-val", "test-static", "test-temporal")

# Absolute tolerance (seconds) for derived-timing consistency checks.
_TIME_ABS_TOL = 1e-6


class DegradationStage(str, Enum):
    """Causal health progression of one robot (methodology §20.7)."""
    HEALTHY = "healthy"
    LATENT_DRIFT = "latent_drift"
    INTERMITTENT = "intermittent"
    PERSISTENT = "persistent"
    OBVIOUS = "obvious"
    FAILED = "failed"
    IN_MAINTENANCE = "in_maintenance"
    RECOMMISSIONED = "recommissioned"


class EpisodeKind(str, Enum):
    """Kind of one robot health episode (methodology §20.7–20.8)."""
    DEGRADATION = "degradation"
    FAILURE = "failure"
    MAINTENANCE = "maintenance"


def _require_identity(value: str, name: str) -> str:
    """Return ``value`` if it is a usable identity string, else fail clearly."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_finite(value: float, name: str) -> float:
    """Return ``value`` as a finite float, else fail clearly."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _require_nonnegative_time(value: float, name: str) -> float:
    """Return ``value`` as a non-negative finite timestamp/duration."""
    result = _require_finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


@dataclass
class OperationEvent:
    """One scheduled robot operation on one physical unit (methodology §20.2).

    Times are float seconds since the factory calendar origin with
    ``arrival_time <= start_time <= end_time``, ``duration`` equal to
    ``end_time - start_time``, and ``queue_delay`` equal to
    ``start_time - arrival_time``. ``travel_time`` covers unit transport to
    the next route stage; ``idle_before`` is the robot idle gap preceding
    this operation. Identity fields live in distinct namespaces and MUST
    NOT be conflated (methodology §20.4).
    """
    operation_id: str
    unit_id: str
    product_type: str
    route_id: str
    route_position: int
    robot_id: str
    program_id: str
    arrival_time: float
    start_time: float
    end_time: float
    duration: float
    queue_delay: float
    travel_time: float
    idle_before: float

    def __post_init__(self) -> None:
        for name in ("operation_id", "unit_id", "product_type",
                     "route_id", "robot_id", "program_id"):
            _require_identity(getattr(self, name), name)
        if (isinstance(self.route_position, bool)
                or not isinstance(self.route_position, int)
                or self.route_position < 0):
            raise ValueError(
                f"route_position must be a non-negative integer, got {self.route_position!r}"
            )
        self.arrival_time = _require_nonnegative_time(self.arrival_time, "arrival_time")
        self.start_time = _require_nonnegative_time(self.start_time, "start_time")
        self.end_time = _require_nonnegative_time(self.end_time, "end_time")
        for name in ("duration", "queue_delay", "travel_time", "idle_before"):
            setattr(self, name, _require_nonnegative_time(getattr(self, name), name))
        if not (self.arrival_time <= self.start_time <= self.end_time):
            raise ValueError(
                "operation timestamps must satisfy arrival_time <= start_time <= end_time, "
                f"got {self.arrival_time}, {self.start_time}, {self.end_time}"
            )
        if not math.isclose(self.duration, self.end_time - self.start_time,
                             rel_tol=1e-9, abs_tol=_TIME_ABS_TOL):
            raise ValueError(
                f"duration {self.duration} != end_time - start_time "
                f"{self.end_time - self.start_time}"
            )
        if not math.isclose(self.queue_delay, self.start_time - self.arrival_time,
                             rel_tol=1e-9, abs_tol=_TIME_ABS_TOL):
            raise ValueError(
                f"queue_delay {self.queue_delay} != start_time - arrival_time "
                f"{self.start_time - self.arrival_time}"
            )
        identities = (self.operation_id, self.unit_id, self.robot_id, self.program_id)
        if len(set(identities)) != len(identities):
            raise ValueError(
                "operation_id, unit_id, robot_id, and program_id must be distinct "
                f"identities, got {identities!r}"
            )


@dataclass
class OperatingContext:
    """Operating conditions at the scheduled operation (methodology §20.6 ``C_t``).

    Encoder-visible conditioning context, NOT a diagnostic: roster shift,
    workload relative to nominal, and ambient temperature.
    """
    shift: str = "day"
    load: float = 1.0
    ambient_temp_c: float = 25.0

    def __post_init__(self) -> None:
        _require_identity(self.shift, "shift")
        self.load = _require_finite(self.load, "load")
        if self.load <= 0.0:
            raise ValueError(f"load must be positive, got {self.load}")
        self.ambient_temp_c = _require_finite(self.ambient_temp_c, "ambient_temp_c")


@dataclass
class RobotHealthState:
    """Latent robot health diagnostics (methodology §20.5 and §20.9).

    Diagnostic ground truth only — MUST NOT enter encoder inputs.
    ``health_value`` is the robot-wide latent state ``H_r(t)``;
    ``manifested_value`` is the program-specific observable ``G_{r,p}(t)``
    with ``program_sensitivity`` γ.
    """
    health_value: float
    program_sensitivity: float
    manifested_value: float
    degradation_stage: DegradationStage
    degradation_severity: float
    degradation_episode_id: str | None = None

    def __post_init__(self) -> None:
        self.health_value = _require_finite(self.health_value, "health_value")
        self.program_sensitivity = _require_finite(
            self.program_sensitivity, "program_sensitivity")
        if self.program_sensitivity < 0.0:
            raise ValueError(
                f"program_sensitivity must be non-negative, got {self.program_sensitivity}"
            )
        self.manifested_value = _require_finite(self.manifested_value, "manifested_value")
        if isinstance(self.degradation_stage, str):
            try:
                self.degradation_stage = DegradationStage(self.degradation_stage)
            except ValueError:
                raise ValueError(
                    f"unknown degradation_stage {self.degradation_stage!r}") from None
        elif not isinstance(self.degradation_stage, DegradationStage):
            raise ValueError(
                f"degradation_stage must be a DegradationStage, got {self.degradation_stage!r}"
            )
        self.degradation_severity = _require_finite(
            self.degradation_severity, "degradation_severity")
        if not 0.0 <= self.degradation_severity <= 1.0:
            raise ValueError(
                f"degradation_severity must be in [0, 1], got {self.degradation_severity}"
            )
        if self.degradation_episode_id is not None:
            _require_identity(self.degradation_episode_id, "degradation_episode_id")


@dataclass
class HealthEpisode:
    """One degradation, failure, or maintenance episode on a robot (§20.7–20.8).

    ``end_time`` is None while the episode is still open at materialization
    time. Maintenance episodes are explicit recommissioning boundaries:
    trajectories MUST NOT be connected across them.
    """
    episode_id: str
    kind: EpisodeKind
    robot_id: str
    start_time: float
    end_time: float | None = None

    def __post_init__(self) -> None:
        _require_identity(self.episode_id, "episode_id")
        _require_identity(self.robot_id, "robot_id")
        if isinstance(self.kind, str):
            try:
                self.kind = EpisodeKind(self.kind)
            except ValueError:
                raise ValueError(f"unknown episode kind {self.kind!r}") from None
        elif not isinstance(self.kind, EpisodeKind):
            raise ValueError(f"kind must be an EpisodeKind, got {self.kind!r}")
        self.start_time = _require_nonnegative_time(self.start_time, "start_time")
        if self.end_time is not None:
            self.end_time = _require_nonnegative_time(self.end_time, "end_time")
            if self.end_time < self.start_time:
                raise ValueError(
                    f"episode end_time {self.end_time} precedes start_time {self.start_time}"
                )

@dataclass
class FailureEvent:
    """One benchmark failure event with its post-hoc category labels.

    Post-hoc diagnostic/target metadata only (methodology §20.9) — it MUST
    NOT enter encoder inputs. ``failure_time`` is the failure-onset timestamp
    ``T``; ``cohort`` is one of ``P`` (progressive), ``W`` (weak-precursor),
    or ``A`` (abrupt/no-precursor). Abrupt events carry no degradation onset
    and zero duration by construction. ``degradation_onset`` is the drawn
    manifest onset (``failure_time - duration_d``) for P/W. ``severity`` is
    the ordered support level in {1.0, 2.0, 4.0}. ``subtype`` names the
    abrupt subtype (A1/A2) and is None otherwise.
    """
    failure_id: str
    robot_id: str
    failure_time: float
    cohort: str
    subtype: str | None = None
    degradation_onset: float | None = None
    duration_d: float = 0.0
    severity: float = 1.0
    degradation_episode_id: str | None = None
    maintenance_episode_id: str | None = None

    def __post_init__(self) -> None:
        _require_identity(self.failure_id, "failure_id")
        _require_identity(self.robot_id, "robot_id")
        self.failure_time = _require_nonnegative_time(
            self.failure_time, "failure_time")
        if self.cohort not in ("P", "W", "A"):
            raise ValueError(
                f"cohort must be one of 'P', 'W', 'A', got {self.cohort!r}")
        self.duration_d = _require_nonnegative_time(
            self.duration_d, "duration_d")
        if self.cohort == "A":
            if self.degradation_onset is not None:
                raise ValueError("abrupt failures must not carry a "
                                 "degradation onset")
            if self.duration_d != 0.0:
                raise ValueError("abrupt failures must have zero duration")
            if self.subtype not in ("A1", "A2"):
                raise ValueError(
                    f"abrupt failures need subtype A1/A2, got {self.subtype!r}")
        else:
            if self.subtype is not None:
                raise ValueError("non-abrupt failures must not carry a subtype")
            if self.degradation_onset is None:
                raise ValueError(
                    f"cohort {self.cohort} failures need a degradation onset")
            self.degradation_onset = _require_nonnegative_time(
                self.degradation_onset, "degradation_onset")
            if not self.degradation_onset < self.failure_time:
                raise ValueError("degradation onset must precede failure time")
            if self.duration_d <= 0.0:
                raise ValueError("non-abrupt failures need positive duration")
        if self.severity not in (1.0, 2.0, 4.0):
            raise ValueError(
                f"severity must be one of 1.0, 2.0, 4.0, got {self.severity!r}")
        if self.degradation_episode_id is not None:
            _require_identity(self.degradation_episode_id,
                              "degradation_episode_id")
        if self.maintenance_episode_id is not None:
            _require_identity(self.maintenance_episode_id,
                              "maintenance_episode_id")

@dataclass
class ObservableAnomalyLabels:
    """Observable file-level anomaly labels for static detection (§20.9).

    Distinct from latent health and future-failure targets: a pre-failure
    file may carry ``is_file_anomalous=False`` while ``failure_within_7d``
    is True. Supervision/evaluation only — MUST NOT enter encoder inputs.
    """
    is_file_anomalous: bool
    anomaly_family: str | None = None
    anomaly_severity: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.is_file_anomalous, bool):
            raise ValueError(
                f"is_file_anomalous must be a bool, got {self.is_file_anomalous!r}"
            )
        if self.is_file_anomalous:
            _require_identity(self.anomaly_family or "", "anomaly_family")
            if self.anomaly_severity is None:
                raise ValueError("anomaly_severity is required for anomalous files")
            self.anomaly_severity = _require_finite(
                self.anomaly_severity, "anomaly_severity")
            if not 0.0 <= self.anomaly_severity <= 1.0:
                raise ValueError(
                    f"anomaly_severity must be in [0, 1], got {self.anomaly_severity}"
                )
        elif self.anomaly_family is not None or self.anomaly_severity is not None:
            raise ValueError(
                "normal files must not carry anomaly_family or anomaly_severity"
            )


@dataclass
class FutureFailureTargets:
    """Future-event supervision/evaluation targets only (methodology §20.9).

    MUST NOT enter encoder inputs. Consistency is structural: a one-day
    warning implies a seven-day warning, and censored files carry no
    failure time and no positive horizon flag.
    """
    time_to_next_failure: float | None
    failure_within_1d: bool
    failure_within_7d: bool
    is_censored: bool

    def __post_init__(self) -> None:
        for name in ("failure_within_1d", "failure_within_7d", "is_censored"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
        if self.time_to_next_failure is not None:
            self.time_to_next_failure = _require_nonnegative_time(
                self.time_to_next_failure, "time_to_next_failure")
        if self.is_censored:
            if self.time_to_next_failure is not None:
                raise ValueError("censored files must not carry time_to_next_failure")
            if self.failure_within_1d or self.failure_within_7d:
                raise ValueError("censored files must not carry positive horizon flags")
        elif self.time_to_next_failure is None:
            if self.failure_within_1d or self.failure_within_7d:
                raise ValueError(
                    "positive horizon flags require time_to_next_failure")
        else:
            expected_1d = self.time_to_next_failure <= SECONDS_PER_DAY
            expected_7d = self.time_to_next_failure <= SECONDS_PER_WEEK
            if self.failure_within_1d != expected_1d:
                raise ValueError(
                    f"failure_within_1d={self.failure_within_1d} contradicts "
                    f"time_to_next_failure={self.time_to_next_failure}"
                )
            if self.failure_within_7d != expected_7d:
                raise ValueError(
                    f"failure_within_7d={self.failure_within_7d} contradicts "
                    f"time_to_next_failure={self.time_to_next_failure}"
                )


@dataclass
class SplitProvenance:
    """Split and precursor-quarantine provenance (§20.10–20.12).

    Records the development cutoff and quarantine state that later split
    construction MUST respect: quarantined files MUST NOT appear in
    healthy development views. Split assignment itself is a later task.
    """
    cutoff_time: float
    is_quarantined: bool
    quarantine_reason: str | None = None
    member_views: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.cutoff_time = _require_nonnegative_time(self.cutoff_time, "cutoff_time")
        if not isinstance(self.is_quarantined, bool):
            raise ValueError(
                f"is_quarantined must be a bool, got {self.is_quarantined!r}"
            )
        if self.is_quarantined:
            _require_identity(self.quarantine_reason or "", "quarantine_reason")
        elif self.quarantine_reason is not None:
            raise ValueError(
                "non-quarantined files must not carry quarantine_reason"
            )
        views = tuple(self.member_views)
        if len(set(views)) != len(views):
            raise ValueError(f"member_views must not contain duplicates, got {views!r}")
        for view in views:
            if view not in ALLOWED_SPLIT_VIEWS:
                raise ValueError(
                    f"unknown split view {view!r}; expected one of {ALLOWED_SPLIT_VIEWS}"
                )
        if self.is_quarantined and ("dev-train" in views or "dev-val" in views):
            raise ValueError(
                "quarantined files MUST NOT appear in healthy development views "
                f"(got {views!r})"
            )
        self.member_views = views


@dataclass
class FactoryProvenance:
    """Deterministic RNG and manifest provenance (methodology §20.14).

    ``stream`` names the independent generator stream (for example
    "calendar", "health", "signal", or "noise") so that resampling one
    concern does not silently reorder another.
    """
    seed: int
    stream: str
    generator_version: str
    config_hash: str

    def __post_init__(self) -> None:
        if (isinstance(self.seed, bool) or not isinstance(self.seed, int)
                or self.seed < 0):
            raise ValueError(f"seed must be a non-negative integer, got {self.seed!r}")
        _require_identity(self.stream, "stream")
        _require_identity(self.generator_version, "generator_version")
        _require_identity(self.config_hash, "config_hash")


# ---------------------------------------------------------------------------
# Model-input separation contract
# ---------------------------------------------------------------------------
#
# The encoder may condition on signal, regime, robot/program identity,
# scheduling, and operating context. Every diagnostic, label, mask, and
# future-target field is enumerated below and MUST NOT enter encoder
# inputs (methodology §20.9, §20.14).
MODEL_INPUT_FIELD_NAMES = frozenset({
    "x",
    "regime_sequence",
    "robot_idx",
    "program_idx",
    "robot_code",
    "program_number",
    "operation",
    "operating_context",
    "file_id",
    "seed",
    "generator_version",
    "config_hash",
    "factory_provenance",
})

DIAGNOSTIC_ONLY_FIELD_NAMES = frozenset({
    "file_label",
    "health",
    "episode",
    "anomaly_labels",
    "future_targets",
    "anomaly_meta",
    "anomaly_mask",
    "rejection_meta",
    "split_provenance",
})


# ---------------------------------------------------------------------------
# Primary sample contract
# ---------------------------------------------------------------------------

@dataclass
class FileSample:
    """
    One complete synthetic session (semantic unit = full file).

    x:              float32 array [C, T] — the signal
    file_id:        stable deterministic string identifier
    file_label:     NORMAL or ABNORMAL
    seed:           integer seed that produced this sample exactly
    generator_version: string tag for reproducibility auditing
    config_hash:    short hash of the SynthConfig used

    regime_sequence: ordered list of RegimeMeta covering all of [0, T)
    anomaly_meta: None for normal; AnomalyMeta for accepted abnormal
    anomaly_mask: None for normal; bool array [C, T] for accepted anomalies
    rejection_meta: optional rejected injection provenance for audit reports
    operation: optional scheduled factory operation event (methodology §20.2)
    operating_context: optional encoder-visible operating conditions (§20.6)
    health: optional latent robot health diagnostics — NEVER a model input (§20.9)
    episode: optional degradation/failure/maintenance episode — NEVER a model input
    anomaly_labels: optional observable file labels — NEVER a model input (§20.9)
    future_targets: optional one-day/seven-day targets — NEVER a model input (§20.9)
    split_provenance: optional split/quarantine provenance (§20.10–20.12)
    factory_provenance: optional deterministic RNG/manifest provenance (§20.14)
    """
    x: npt.NDArray[np.float32]           # [C, T]
    file_id: str
    file_label: SampleLabel
    seed: int
    generator_version: str
    config_hash: str
    regime_sequence: list[RegimeMeta]
    anomaly_meta: AnomalyMeta | None = None
    anomaly_mask: npt.NDArray[np.bool_] | None = None  # [C, T]
    rejection_meta: AnomalyMeta | None = None
    robot_idx: int = 0
    program_idx: int = 0
    robot_code: str = "R01"
    program_number: str = "P100"
    operation: OperationEvent | None = None
    operating_context: OperatingContext | None = None
    health: RobotHealthState | None = None
    episode: HealthEpisode | None = None
    anomaly_labels: ObservableAnomalyLabels | None = None
    future_targets: FutureFailureTargets | None = None
    split_provenance: SplitProvenance | None = None
    factory_provenance: FactoryProvenance | None = None

    @property
    def C(self) -> int:
        return self.x.shape[0]

    @property
    def T(self) -> int:
        return self.x.shape[1]

    def validate(self) -> None:
        """Assert internal consistency."""
        assert self.x.dtype == np.float32, "x must be float32"
        assert self.x.ndim == 2, "x must be [C, T]"
        assert self.C in SUPPORTED_CHANNEL_COUNTS, (
            f"unsupported channel count {self.C}; expected {SUPPORTED_CHANNEL_COUNTS}"
        )
        assert self.T > 0, "session must contain at least one timestep"
        assert np.isfinite(self.x).all(), "x must be finite"
        assert self.robot_idx >= 0, "robot_idx must be non-negative"
        assert self.program_idx >= 0, "program_idx must be non-negative"

        if self.anomaly_mask is not None:
            assert self.anomaly_mask.shape == self.x.shape, (
                f"anomaly_mask shape {self.anomaly_mask.shape} != x shape {self.x.shape}"
            )
        if self.file_label == SampleLabel.ABNORMAL:
            assert self.anomaly_meta is not None, "abnormal sample must have anomaly_meta"
            assert self.anomaly_mask is not None, "abnormal sample must have anomaly_mask"
            assert self.anomaly_mask.any(), "anomaly_mask must have at least one True"

        # Regime sequence must cover [0, T) without gaps or overlaps
        if self.regime_sequence:
            starts = [r.start for r in self.regime_sequence]
            ends = [r.end for r in self.regime_sequence]
            assert starts[0] == 0, "regime sequence must start at 0"
            assert ends[-1] == self.T, f"regime sequence must end at T={self.T}"
            for i in range(1, len(self.regime_sequence)):
                assert starts[i] == ends[i - 1], "regime sequence must be contiguous"

        # Factory contracts self-validate on construction; here only
        # cross-field agreement with the file-level label/provenance is
        # checked. Cross-event scheduler invariants belong to a later task.
        if self.anomaly_labels is not None:
            assert self.anomaly_labels.is_file_anomalous == (
                self.file_label == SampleLabel.ABNORMAL
            ), "anomaly_labels must agree with file_label"
            if self.file_label == SampleLabel.ABNORMAL:
                assert self.anomaly_meta is not None
                assert self.anomaly_labels.anomaly_family == (
                    self.anomaly_meta.family.value
                ), "observable anomaly family must match injected anomaly_meta family"
        if self.factory_provenance is not None:
            prov = self.factory_provenance
            assert prov.seed == self.seed, "factory_provenance.seed must match seed"
            assert prov.generator_version == self.generator_version, (
                "factory_provenance.generator_version must match generator_version"
            )
            assert prov.config_hash == self.config_hash, (
                "factory_provenance.config_hash must match config_hash"
            )

    def encoder_inputs(self) -> dict[str, Any]:
        """Return exactly the encoder-visible conditioning for this file.

        Latent health, episodes, observable labels, masks, and future
        targets are diagnostics/supervision and are NEVER included
        (methodology §20.9). Every key returned here must belong to
        ``MODEL_INPUT_FIELD_NAMES``; every field in
        ``DIAGNOSTIC_ONLY_FIELD_NAMES`` must stay out.
        """
        inputs = {
            "x": self.x,
            "regime_sequence": self.regime_sequence,
            "robot_idx": self.robot_idx,
            "program_idx": self.program_idx,
            "robot_code": self.robot_code,
            "program_number": self.program_number,
            "operation": self.operation,
            "operating_context": self.operating_context,
            "file_id": self.file_id,
            "seed": self.seed,
            "generator_version": self.generator_version,
            "config_hash": self.config_hash,
            "factory_provenance": self.factory_provenance,
        }
        assert set(inputs) <= MODEL_INPUT_FIELD_NAMES, (
            "encoder_inputs leaked a non-input field"
        )
        assert not (set(inputs) & DIAGNOSTIC_ONLY_FIELD_NAMES), (
            "encoder_inputs leaked a diagnostic-only field"
        )
        return inputs


# ---------------------------------------------------------------------------
# Patch batch
# ---------------------------------------------------------------------------

@dataclass
class PatchBatch:
    """
    Patched view of one FileSample.

    patches:     float32 [N, C, W]
    starts:      int64   [N]           — timestep start of each patch
    valid_len:   int64   [N]           — number of real (non-padded) timesteps per patch
    pad_mask:    bool    [N, W]        — True where padding was inserted (not real signal)
    file_sample: the source FileSample
    """
    patches: npt.NDArray[np.float32]   # [N, C, W]
    starts: npt.NDArray[np.int64]      # [N]
    valid_len: npt.NDArray[np.int64]   # [N]
    pad_mask: npt.NDArray[np.bool_]    # [N, W]  True = padded position
    file_sample: FileSample

    @property
    def N(self) -> int:
        return self.patches.shape[0]

    @property
    def W(self) -> int:
        return self.patches.shape[2]

    def timestep_to_patch_mask(self) -> npt.NDArray[np.bool_] | None:
        """
        Map the file-level anomaly_mask (if present) to patch-level.

        Returns bool array [N] where True means the patch contains ≥1
        anomalous timestep in any affected channel.
        Returns None if no anomaly_mask.
        """
        if self.file_sample.anomaly_mask is None:
            return None
        C, T = self.file_sample.anomaly_mask.shape
        result = np.zeros(self.N, dtype=bool)
        for i, (start, vlen) in enumerate(zip(self.starts, self.valid_len)):
            end = int(start) + int(vlen)
            if end > T:
                end = T
            if start >= T:
                continue
            window = self.file_sample.anomaly_mask[:, int(start):end]
            result[i] = window.any()
        return result


# ---------------------------------------------------------------------------
# Masking result
# ---------------------------------------------------------------------------

@dataclass
class MaskResult:
    """Output of a masking policy applied to a PatchBatch."""
    mask: npt.NDArray[np.bool_]   # [N] True = patch is masked
    composition: dict[str, int]   # how many patches from each strategy
    total_ratio: float
