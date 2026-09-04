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
    gain_std: float = 0.02      # per-channel multiplicative gain jitter
    offset_std: float = 0.01    # per-channel additive offset jitter
    noise_std: float = 0.005    # additional white noise (fraction of signal std)
    max_shift: int = 2          # max timestep shift (edge-preserving)

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
