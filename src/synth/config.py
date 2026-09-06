"""
SynthConfig — the single typed configuration for all generation parameters.

All defaults produce hard-benchmark data.  Easy-sanity anomalies are
opt-in via ``include_easy_sanity=True``.
"""

from __future__ import annotations

import numpy as np
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import date

GENERATOR_VERSION = "2.0.0"

# The physics model has two deliberately supported layouts.  Three channels
# retain the original feed/current/temperature contract; six channels add
# voltage, power, and torch-pressure sensors with causal links to the base
# process.  Rejecting other counts is preferable to silently fabricating
# independent channels.
SUPPORTED_CHANNEL_COUNTS = (3, 6)
CHANNEL_NAMES = (
    "wire_feed_speed", "welding_current", "temperature",
    "arc_voltage", "arc_power", "torch_pressure",
)


def channel_offset_scales(n_channels: int, config: "PhysicsConfig") -> np.ndarray:
    """Return physical offset scales for a supported channel layout."""
    if n_channels not in SUPPORTED_CHANNEL_COUNTS:
        raise ValueError(
            f"n_channels must be one of {SUPPORTED_CHANNEL_COUNTS}, got {n_channels}"
        )
    base = np.array([
        config.lti_max_feed * 0.05,
        config.lti_max_feed * config.lti_target_current_ratio * 0.05,
        5.0,
        2.0,
        config.lti_max_feed * config.lti_target_current_ratio * 0.10,
        0.15,
    ], dtype=np.float64)
    return base[:n_channels]


@dataclass
class PhysicsConfig:
    """Parameters for the causal signal physics."""
    # LTI current dynamics
    lti_natural_freq: float = 8.0       # ωₙ rad/s
    lti_damping_ratio: float = 0.55     # ζ — ~12% overshoot
    lti_dc_gain: float = 1.0
    lti_delay_steps: int = 2
    lti_max_feed: float = 300.0         # mm/min normalisation
    lti_target_current_ratio: float = 0.04

    # Thermal dynamics
    thermal_alpha_heat: float = 0.018
    thermal_alpha_cool: float = 0.006
    thermal_ambient: float = 25.0
    thermal_current_to_temp_gain: float = 0.65

    # Channel gain/offset natural variation (±fraction of nominal)
    channel_gain_variation: float = 0.10
    channel_offset_variation: float = 0.05

    # Shared latent factor — adds cross-channel correlation
    shared_factor_amplitude: float = 0.04   # fraction of signal amplitude
    shared_factor_ar1_phi: float = 0.90


@dataclass
class NoiseConfig:
    """Natural sensor noise configuration."""
    # S0 wire-feed vibration
    s0_vibration_amplitude: float = 0.008   # fraction of signal
    s0_vibration_freq: float = 0.025        # Hz (relative to sample rate)
    s0_gaussian_std: float = 0.005
    # S1 arc noise
    s1_arc_noise_std: float = 0.05          # fraction of local std
    # S2 thermal random walk
    s2_rw_step_std: float = 0.10            # °C per step
    s2_glitch_prob: float = 0.0001
    s2_glitch_magnitude: float = 5.0
    # Shared AR(1)
    ar1_phi: float = 0.85
    local_window: int = 40


@dataclass
class RegimeConfig:
    """Parameters for regime sequence and transition generation."""
    # Session length
    min_total_steps: int = 200
    max_total_steps: int = 800
    # Number of regimes per session
    min_regimes: int = 2
    max_regimes: int = 5
    # Duration ranges per regime type (steps)
    idle_range: tuple[int, int] = (15, 60)
    active_range: tuple[int, int] = (40, 200)
    ramp_range: tuple[int, int] = (10, 40)
    periodic_range: tuple[int, int] = (30, 120)
    recovery_range: tuple[int, int] = (20, 80)
    # Transition smoothing (cosine blend steps)
    min_transition_steps: int = 5
    max_transition_steps: int = 20
    # Target feed speed range
    min_feed_level: float = 0.5     # fraction of lti_max_feed
    max_feed_level: float = 1.0
    # Periodic regime frequency range (cycles / total period)
    min_periodic_freq: float = 0.05
    max_periodic_freq: float = 0.30
    # Low-frequency drift per channel (amplitude as fraction of signal range)
    drift_amplitude: float = 0.03


@dataclass
class AnomalyConfig:
    """Parameters for anomaly generation and intervention acceptance."""
    # Composite strength gate (per-channel MAD/std-change), not raw amplitude.
    strength_weak_min: float = 0.05
    strength_strong_max: float = 1.5
    max_rejection_attempts: int = 20

    # ── Per-family severity defaults ──
    # Contextual replacement
    ctx_donor_regime_match_required: bool = True
    ctx_boundary_blend_steps: int = 8
    # Wrong transition
    wt_speed_range: tuple[float, float] = (0.3, 0.7)  # fraction of normal speed
    wt_timing_shift_range: tuple[float, float] = (0.2, 0.5)  # fraction of regime duration
    # Realistic stuck
    stuck_sigma_ratio_range: tuple[float, float] = (0.3, 0.6)  # σ_abn / σ_expected
    stuck_min_duration: int = 15
    stuck_max_duration: int = 80
    # Over-regularity
    reg_jitter_suppress_range: tuple[float, float] = (0.05, 0.25)  # residual jitter fraction
    reg_min_duration: int = 20
    # Subtle drift
    drift_rate_range: tuple[float, float] = (0.0002, 0.002)  # units/step
    drift_min_duration: int = 30
    freq_delta_range: tuple[float, float] = (0.1, 0.4)   # fraction of base freq
    phase_delta_range: tuple[float, float] = (0.2, 0.8)  # fraction of π
    fp_min_duration: int = 15
    # Cross-channel inconsistency
    cc_lag_range: tuple[int, int] = (3, 12)
    cc_gain_delta_range: tuple[float, float] = (0.15, 0.40)
    cc_phase_delta_range: tuple[float, float] = (0.2, 0.7)
    cc_dynamics_threshold: float = 0.15  # min local std to consider dynamic
    # Duration anomaly
    dur_stretch_range: tuple[float, float] = (1.5, 3.0)
    dur_compress_range: tuple[float, float] = (0.2, 0.6)
    dur_min_regime_steps: int = 20
    # Missing event
    me_replacement_mode: str = "wrong_regime"  # or "continuation"
    # Easy sanity
    easy_spike_amplitude: float = 4.0    # × local amplitude
    easy_spike_duration: int = 5
    easy_flatline_duration_range: tuple[int, int] = (40, 60)

    include_easy_sanity: bool = False


@dataclass
class SplitConfig:
    """Dataset materialization parameters."""
    # Production: 100k normal train; balanced 20k validation; 50k/50k test.
    n_train: int = 100_000
    n_val: int = 20_000
    n_test: int = 100_000
    anomaly_families: list[str] = field(default_factory=lambda: [
        "contextual_replacement", "wrong_transition", "realistic_stuck",
        "over_regularity", "subtle_drift", "freq_phase_mismatch",
        "cross_channel_inconsistency", "duration_anomaly", "missing_event",
    ])
    contamination_ratios: list[float] = field(default_factory=lambda: [
        0.0, 0.01, 0.05, 0.10, 0.20
    ])
    unseen_seed_fraction: float = 0.3
    train_seed: int = 0
    val_seed: int = 1_000_000
    test_seed: int = 2_000_000


@dataclass
class PatchConfig:
    """Patchification parameters."""
    patch_size: int = 32
    stride: int = 16
    pad_end: bool = True      # pad last partial patch to full size
    pad_value: float = 0.0


@dataclass
class MaskingConfig:
    """Masking composition parameters (total ratio fixed for ablations)."""
    total_mask_ratio: float = 0.40
    # Composition fractions (must sum to 1.0)
    random_fraction: float = 0.34
    info_fraction: float = 0.33
    block_fraction: float = 0.33
    # Info-aware stratification
    info_stable_fraction: float = 0.20
    info_transition_fraction: float = 0.30
    info_dynamic_fraction: float = 0.30
    info_extreme_fraction: float = 0.20
    # Block masking
    min_block_size: int = 2
    max_block_size: int = 6


@dataclass
class ContrastiveConfig:
    """Same-file contrastive view augmentation."""
    gain_std: float = 0.05      # per-channel multiplicative gain jitter
    offset_std: float = 0.03    # per-channel additive offset jitter
    noise_std: float = 0.015    # additional white noise (fraction of signal std)
    max_shift: int = 4          # max timestep shift (edge-preserving)

    def __post_init__(self) -> None:
        if self.gain_std < 0.0:
            raise ValueError("gain_std must be non-negative")
        if self.offset_std < 0.0:
            raise ValueError("offset_std must be non-negative")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if self.max_shift < 0:
            raise ValueError("max_shift must be non-negative")

@dataclass
class FleetConfig:
    """Robot and program fleet configuration for conditional normalization."""
    n_robots: int = 5
    n_programs: int = 8
    robot_gain_std: float = 0.05
    robot_temp_offset_range: tuple[float, float] = (-3.0, 8.0)
    program_speed_scales: list[float] = field(default_factory=lambda: [
        0.7, 0.85, 1.0, 1.15, 1.3, 0.9, 1.2, 1.4
    ])


@dataclass
class FactoryCalendarConfig:
    """3–6 month shared-unit factory calendar (methodology §20.1, §20.10).

    ``calendar_origin`` is the ISO YYYY-MM-DD date of t=0; scheduled event
    timestamps are float seconds after that origin. ``dev_cutoff_days``
    defaults to two months; ``quarantine_days`` MUST cover the largest
    warning horizon (at least 7 days). Calendar scheduling itself is a
    later task; this contract only bounds the calendar structurally.
    """
    calendar_origin: str = "2024-01-01"
    span_days: float = 120.0
    dev_cutoff_days: float = 60.0
    quarantine_days: float = 7.0
    seed: int = 0

    def __post_init__(self) -> None:
        try:
            date.fromisoformat(self.calendar_origin)
        except (TypeError, ValueError):
            raise ValueError(
                f"calendar_origin must be an ISO YYYY-MM-DD date, "
                f"got {self.calendar_origin!r}"
            ) from None
        for name in ("span_days", "dev_cutoff_days", "quarantine_days"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")
            setattr(self, name, value)
        if not 90.0 <= self.span_days <= 183.0:
            raise ValueError(
                f"span_days must cover a 3–6 month calendar (≈90–183 days), "
                f"got {self.span_days}"
            )
        if not 0.0 < self.dev_cutoff_days < self.span_days:
            raise ValueError(
                f"dev_cutoff_days must lie inside (0, span_days), got "
                f"{self.dev_cutoff_days} with span_days={self.span_days}"
            )
        if not 7.0 <= self.quarantine_days < self.span_days:
            raise ValueError(
                f"quarantine_days must cover the 7-day warning horizon and lie "
                f"inside [7, span_days), got {self.quarantine_days}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


@dataclass
class RouteStageConfig:
    """One operation position on a production route (methodology §20.2).

    ``duration_s`` is the fixed operation duration; ``travel_after_s`` is
    the unit travel time to the next stage. The final stage of a route
    MUST set ``travel_after_s`` to zero: line-exit travel is not
    represented in any operation event.
    """
    robot_id: str
    program_id: str
    duration_s: float
    travel_after_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("robot_id", "program_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
        for name in ("duration_s", "travel_after_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")
            setattr(self, name, value)
        if self.duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if self.travel_after_s < 0.0:
            raise ValueError(
                f"travel_after_s must be non-negative, got {self.travel_after_s}")


@dataclass
class RouteConfig:
    """One defined production route traversed by physical units (§20.2)."""
    route_id: str
    product_type: str
    stages: list[RouteStageConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("route_id", "product_type"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
        if not self.stages:
            raise ValueError(f"route {self.route_id!r} must define at least one stage")
        if self.stages[-1].travel_after_s != 0.0:
            raise ValueError(
                f"route {self.route_id!r} final stage must have travel_after_s=0, "
                f"got {self.stages[-1].travel_after_s}")


def _default_routes() -> list[RouteConfig]:
    """Two-stage default line used when no routes are configured."""
    return [RouteConfig(
        route_id="route-A",
        product_type="sedan",
        stages=[
            RouteStageConfig(robot_id="robot-01", program_id="program-01",
                             duration_s=600.0, travel_after_s=60.0),
            RouteStageConfig(robot_id="robot-02", program_id="program-02",
                             duration_s=600.0, travel_after_s=0.0),
        ],
    )]


@dataclass
class SchedulerConfig:
    """Causal shared-unit factory scheduler parameters (methodology §20.3).

    Units arrive at a mean rate of one per ``arrival_interval_s`` seconds
    starting at t=0 and are assigned a route deterministically from
    ``seed``. Robots serve one operation at a time; scheduling itself is a
    later-task-free causal construction — health, signals, and anomalies
    are NOT modeled here. ``arrival_jitter_s`` adds a deterministic
    seed-drawn uniform offset in ``[-arrival_jitter_s, +arrival_jitter_s]``
    (clamped at zero) so sparse server-scale calendars keep asynchronous
    cross-robot utilization instead of a perfectly periodic grid;
    0.0 preserves exact periodicity.
    """
    n_units: int = 20
    arrival_interval_s: float = 1800.0
    arrival_jitter_s: float = 0.0
    routes: list[RouteConfig] = field(default_factory=_default_routes)
    seed: int = 0

    def __post_init__(self) -> None:
        if (isinstance(self.n_units, bool) or not isinstance(self.n_units, int)
                or self.n_units < 1):
            raise ValueError(f"n_units must be a positive integer, got {self.n_units!r}")
        self.arrival_interval_s = float(self.arrival_interval_s)
        if not np.isfinite(self.arrival_interval_s) or self.arrival_interval_s <= 0.0:
            raise ValueError(
                f"arrival_interval_s must be positive, got {self.arrival_interval_s}")
        self.arrival_jitter_s = float(self.arrival_jitter_s)
        if not np.isfinite(self.arrival_jitter_s) or self.arrival_jitter_s < 0.0:
            raise ValueError(
                f"arrival_jitter_s must be non-negative, got {self.arrival_jitter_s}")
        if not self.routes:
            raise ValueError("scheduler requires at least one route")
        route_ids = [route.route_id for route in self.routes]
        if len(set(route_ids)) != len(route_ids):
            raise ValueError(f"route_ids must be unique, got {route_ids!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

@dataclass
class HealthConfig:
    """Robot-wide health, failure, and maintenance parameters (§20.5–20.8).

    One latent trajectory ``H_r(t)`` per robot evolves through calendar
    aging, operation wear, and small stochastic variation; programs reveal
    it with deterministic per-pair sensitivity ``γ``. Failures arise from
    a hazard coupled to health and accumulated usage, never from
    independent timestamps. The health RNG stream is seeded from ``seed``
    alone and never shared with scheduling or signal streams, so noise
    resampling cannot reorder the factory calendar (§20.14). Health
    simulation itself lives in ``synth.health``; this contract only bounds
    the process parameters.
    """
    seed: int = 0
    aging_rate: float = 1e-7
    wear_rate: float = 1e-4
    noise_scale: float = 1e-3
    sensitivity_min: float = 0.5
    sensitivity_max: float = 1.5
    degradation_onset: float = 0.3
    severity_scale: float = 2.0
    base_rate: float = 1e-9
    abrupt_rate: float = 0.0
    alpha: float = 2.0
    beta: float = 0.0
    maintenance_duration_s: float = 86400.0
    recommission_mean: float = 0.05
    recommission_scale: float = 0.02

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        for name in ("aging_rate", "wear_rate", "noise_scale", "base_rate",
                     "abrupt_rate", "maintenance_duration_s",
                     "recommission_mean", "recommission_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be a non-negative finite value, "
                    f"got {getattr(self, name)!r}")
            setattr(self, name, value)
        for name in ("sensitivity_min", "sensitivity_max", "alpha", "beta"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite, got {getattr(self, name)!r}")
            setattr(self, name, value)
        if self.sensitivity_min < 0.0 or self.sensitivity_max < self.sensitivity_min:
            raise ValueError(
                "program sensitivities require 0 <= sensitivity_min <= "
                f"sensitivity_max, got {self.sensitivity_min}, {self.sensitivity_max}")
        self.degradation_onset = float(self.degradation_onset)
        if not np.isfinite(self.degradation_onset) or self.degradation_onset < 0.0:
            raise ValueError(
                "degradation_onset must be a non-negative finite value, "
                f"got {self.degradation_onset!r}")
        self.severity_scale = float(self.severity_scale)
        if not np.isfinite(self.severity_scale) or self.severity_scale <= 0.0:
            raise ValueError(
                f"severity_scale must be positive, got {self.severity_scale!r}")

@dataclass
class SignalConfig:
    """Scheduled-operation signal composition parameters (methodology §20.6).

    Carries the independent ``seed`` for the signal-composition stream:
    per-event regime/healthy-baseline draws and per-event sensor-noise
    draws are sibling ``numpy.random.Generator`` instances derived from
    this seed plus the operation identity, never shared with the
    scheduler (``SchedulerConfig.seed``) or health (``HealthConfig.seed``)
    streams, so resampling signal noise cannot reorder the factory
    calendar or move failure episodes (§20.14). Signal synthesis itself
    lives in ``synth.scheduled``; this contract only names the stream.
    """
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

@dataclass
class TemporalAnomalyConfig:
    """Temporal anomaly and degradation manifestation policy (section 20.7).
    Decides per scheduled operation whether a localized symptom is injected,
    using only causal information: manifested health G and health stage.
    """
    seed: int = 0
    isolated_rate: float = 0.02
    precursor_slope: float = 3.0
    precursor_intercept: float = -3.0
    severity_floor: float = 0.15
    severity_scale: float = 2.0
    failure_severity: float = 0.9
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        self.isolated_rate = float(self.isolated_rate)
        if not np.isfinite(self.isolated_rate) or not 0.0 <= self.isolated_rate <= 1.0:
            raise ValueError(f"isolated_rate must lie in [0, 1], got {self.isolated_rate!r}")
        for name in ("precursor_slope", "precursor_intercept"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")
            setattr(self, name, value)
        if self.precursor_slope < 0.0:
            raise ValueError(f"precursor_slope must be non-negative, got {self.precursor_slope}")
        self.severity_floor = float(self.severity_floor)
        if not np.isfinite(self.severity_floor) or not 0.0 <= self.severity_floor <= 1.0:
            raise ValueError(f"severity_floor must lie in [0, 1], got {self.severity_floor!r}")
        self.severity_scale = float(self.severity_scale)
        if not np.isfinite(self.severity_scale) or self.severity_scale <= 0.0:
            raise ValueError(f"severity_scale must be positive, got {self.severity_scale!r}")
        self.failure_severity = float(self.failure_severity)
        if not np.isfinite(self.failure_severity) or not 0.0 <= self.failure_severity <= 1.0:
            raise ValueError(f"failure_severity must lie in [0, 1], got {self.failure_severity!r}")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise ValueError(f"max_attempts must be a positive integer, got {self.max_attempts!r}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be a positive integer, got {self.max_attempts!r}")

@dataclass
class SynthConfig:
    """Master configuration — single source of truth for all generation."""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    factory: FactoryCalendarConfig = field(default_factory=FactoryCalendarConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    temporal: TemporalAnomalyConfig = field(default_factory=TemporalAnomalyConfig)

    # Six channels are the production layout; the legacy three-channel
    # layout remains supported for old experiments and fixtures.
    n_channels: int = 6

    def __post_init__(self) -> None:
        if self.n_channels not in SUPPORTED_CHANNEL_COUNTS:
            raise ValueError(
                f"n_channels must be one of {SUPPORTED_CHANNEL_COUNTS}, "
                f"got {self.n_channels}"
            )
        if self.fleet.n_robots <= 0:
            raise ValueError("fleet n_robots must be positive")
        if self.fleet.n_programs <= 0:
            raise ValueError("fleet n_programs must be positive")
        r = self.regime
        if r.min_total_steps <= 0 or r.max_total_steps < r.min_total_steps:
            raise ValueError("regime total-step bounds are invalid")
        if r.min_regimes <= 0 or r.max_regimes < r.min_regimes:
            raise ValueError("regime-count bounds are invalid")
        for name in ("idle_range", "active_range", "ramp_range",
                     "periodic_range", "recovery_range"):
            lo, hi = getattr(r, name)
            if lo <= 0 or hi < lo:
                raise ValueError(f"{name} must be an ordered positive range")
        p = self.patch
        if p.patch_size <= 0 or p.stride <= 0:
            raise ValueError("patch_size and stride must be positive")
        m = self.masking
        if not 0 <= m.total_mask_ratio <= 1:
            raise ValueError("total_mask_ratio must be in [0, 1]")
        comp = (m.random_fraction, m.info_fraction, m.block_fraction)
        if any(v < 0 for v in comp) or not np.isclose(sum(comp), 1.0):
            raise ValueError("masking composition fractions must sum to 1")
        strata = (m.info_stable_fraction, m.info_transition_fraction,
                  m.info_dynamic_fraction, m.info_extreme_fraction)
        if any(v < 0 for v in strata) or not np.isclose(sum(strata), 1.0):
            raise ValueError("information strata fractions must sum to 1")
        if m.min_block_size <= 0 or m.max_block_size < m.min_block_size:
            raise ValueError("block-size bounds are invalid")
        a = self.anomaly
        if a.strength_weak_min < 0 or a.strength_strong_max < a.strength_weak_min:
            raise ValueError("strength gate bounds are invalid")
        if a.max_rejection_attempts <= 0:
            raise ValueError("max_rejection_attempts must be positive")
        def ordered(name: str, *, lower: float = 0.0, upper: float | None = None) -> None:
            lo, hi = getattr(a, name)
            if (not np.isfinite(lo) or not np.isfinite(hi) or lo < lower
                    or hi < lo or (upper is not None and hi > upper)):
                suffix = f" in [{lower}, {upper}]" if upper is not None else f" >= {lower}"
                raise ValueError(f"{name} must be ordered with finite values{suffix}")
        ordered("wt_speed_range", upper=1.0)
        ordered("wt_timing_shift_range", upper=1.0)
        ordered("stuck_sigma_ratio_range", upper=1.0)
        ordered("reg_jitter_suppress_range", upper=1.0)
        ordered("drift_rate_range")
        ordered("freq_delta_range")
        ordered("phase_delta_range")
        ordered("cc_gain_delta_range")
        ordered("cc_phase_delta_range")
        ordered("dur_stretch_range", lower=1.0)
        ordered("dur_compress_range", upper=1.0)
        if any(getattr(a, n) <= 0 for n in (
            "stuck_min_duration", "stuck_max_duration", "reg_min_duration",
            "drift_min_duration", "fp_min_duration", "cc_dynamics_threshold",
            "dur_min_regime_steps",
        )):
            raise ValueError("anomaly duration and dynamics thresholds must be positive")
        if a.stuck_max_duration < a.stuck_min_duration:
            raise ValueError("stuck duration bounds are invalid")
        if any(int(v) != v or v <= 0 for v in a.cc_lag_range) or a.cc_lag_range[1] < a.cc_lag_range[0]:
            raise ValueError("cc_lag_range must be ordered positive integers")
        if a.me_replacement_mode not in {"wrong_regime", "continuation"}:
            raise ValueError("me_replacement_mode must be 'wrong_regime' or 'continuation'")
        lo, hi = a.easy_flatline_duration_range
        if lo <= 0 or hi < lo:
            raise ValueError("easy_flatline_duration_range must be ordered positive")
        s = self.split
        if min(s.n_train, s.n_val, s.n_test) < 0:
            raise ValueError("split sizes must be non-negative")
        if not 0 <= s.unseen_seed_fraction <= 1:
            raise ValueError("unseen_seed_fraction must be in [0, 1]")
        if any(not 0 <= float(v) <= 1 for v in s.contamination_ratios):
            raise ValueError("contamination ratios must be in [0, 1]")

    def hash(self) -> str:
        """Short SHA-256 hex digest of this config for provenance tagging."""
        d = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(d.encode()).hexdigest()[:12]
