"""
Regime-sequence generator for multi-regime variable-length sessions.

Regime graph (valid successor edges):
  idle       → ramp_up, periodic
  ramp_up    → active
  active     → ramp_down, periodic
  ramp_down  → idle, recovery
  periodic   → ramp_up, active, recovery
  recovery   → idle, (end)
  (end)      — any regime may end the session

Valid sequence examples (DATA.md §3.1):
  idle → ramp_up → active → ramp_down → recovery
  idle → periodic → ramp_up → active → ramp_down
  periodic → active → ramp_down → recovery
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.config import RegimeConfig, PhysicsConfig
from synth.schema import RegimeMeta, RegimeType


# ---------------------------------------------------------------------------
# Regime graph
# ---------------------------------------------------------------------------

# Maps each regime to its valid successors.
_REGIME_GRAPH: dict[RegimeType, list[RegimeType]] = {
    RegimeType.IDLE:     [RegimeType.RAMP_UP, RegimeType.PERIODIC],
    RegimeType.RAMP_UP:  [RegimeType.ACTIVE],
    RegimeType.ACTIVE:   [RegimeType.RAMP_DOWN, RegimeType.PERIODIC],
    RegimeType.RAMP_DOWN:[RegimeType.IDLE, RegimeType.RECOVERY],
    RegimeType.PERIODIC: [RegimeType.RAMP_UP, RegimeType.ACTIVE, RegimeType.RECOVERY],
    RegimeType.RECOVERY: [RegimeType.IDLE],
}

# Regimes that can start a session
_START_REGIMES = [RegimeType.IDLE, RegimeType.PERIODIC]


def sample_regime_sequence(
    rng: np.random.Generator,
    cfg: RegimeConfig,
) -> list[tuple[RegimeType, int]]:
    """
    Sample a valid regime sequence with durations.

    Returns: list of (RegimeType, duration_steps) pairs.
    Total duration is sampled uniformly in [min_total, max_total].

    Guarantees:
    - At least cfg.min_regimes segments.
    - No fixed ordering (idle→ramp_up→active is just one option).
    - Variable durations drawn from per-regime ranges.
    - Session ends with RECOVERY or RAMP_DOWN (natural close).
    """
    target_T = int(rng.integers(cfg.min_total_steps, cfg.max_total_steps + 1))
    n_regimes = int(rng.integers(cfg.min_regimes, cfg.max_regimes + 1))

    # Build regime sequence via random walk on the graph.
    # Use int indexing (not rng.choice on enum list) to preserve RegimeType enum type.
    seq: list[RegimeType] = [_START_REGIMES[int(rng.integers(0, len(_START_REGIMES)))]]
    for _ in range(n_regimes - 1):
        successors = _REGIME_GRAPH.get(seq[-1], [])
        if not successors:
            break
        seq.append(successors[int(rng.integers(0, len(successors)))])

    # Ensure a clean ending
    if seq[-1] not in (RegimeType.RECOVERY, RegimeType.RAMP_DOWN, RegimeType.IDLE):
        seq.append(RegimeType.RAMP_DOWN)
        if rng.random() < 0.5:
            seq.append(RegimeType.RECOVERY)

    # Sample durations from per-regime ranges, then rescale to hit target_T
    raw_durations = [_sample_duration(rng, cfg, r) for r in seq]
    total_raw = sum(raw_durations)
    scale = target_T / total_raw
    durations = [max(5, round(d * scale)) for d in raw_durations]
    # Fix rounding error by adjusting the longest segment
    diff = target_T - sum(durations)
    if diff != 0:
        idx = int(np.argmax(durations))
        durations[idx] += diff

    return list(zip(seq, durations))


def _sample_duration(rng: np.random.Generator, cfg: RegimeConfig, r: RegimeType) -> int:
    ranges = {
        RegimeType.IDLE:      cfg.idle_range,
        RegimeType.RAMP_UP:   cfg.ramp_range,
        RegimeType.ACTIVE:    cfg.active_range,
        RegimeType.RAMP_DOWN: cfg.ramp_range,
        RegimeType.PERIODIC:  cfg.periodic_range,
        RegimeType.RECOVERY:  cfg.recovery_range,
    }
    lo, hi = ranges.get(r, (20, 60))
    return int(rng.integers(lo, hi + 1))


# ---------------------------------------------------------------------------
# Feed-speed envelope builder
# ---------------------------------------------------------------------------

def build_feed_envelope(
    regime_sequence: list[tuple[RegimeType, int]],
    cfg_regime: RegimeConfig,
    cfg_physics: PhysicsConfig,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], list[RegimeMeta]]:
    """
    Build the wire_feed_speed envelope from a regime sequence.

    Returns:
        envelope: float64 [T] feed-speed values
        meta:     list[RegimeMeta] with exact start/end/target_level
    """
    T = sum(dur for _, dur in regime_sequence)
    envelope = np.zeros(T, dtype=np.float64)
    meta: list[RegimeMeta] = []

    max_feed = cfg_physics.lti_max_feed
    min_level = cfg_regime.min_feed_level
    max_level = cfg_regime.max_feed_level

    # Sample a target feed level for this session
    feed_level = rng.uniform(min_level, max_level)
    target_feed = feed_level * max_feed

    cursor = 0
    prev_level = 0.0  # last level of previous regime

    for regime, dur in regime_sequence:
        start = cursor
        end = cursor + dur
        blend_steps = int(rng.integers(cfg_regime.min_transition_steps, cfg_regime.max_transition_steps + 1))
        blend_steps = min(blend_steps, dur // 3)

        if regime == RegimeType.IDLE:
            # Near-zero feed with small idle noise
            idle_level = rng.uniform(0.0, 0.02) * max_feed
            segment = np.full(dur, idle_level)
            _smooth_blend(segment, prev_level, idle_level, blend_steps)
            this_level = idle_level

        elif regime == RegimeType.RAMP_UP:
            # Smooth ramp from prev_level to target_feed
            t_norm = np.linspace(0, 1, dur)
            # Cosine ease-in
            segment = prev_level + (target_feed - prev_level) * _cosine_ease(t_norm)
            this_level = target_feed

        elif regime == RegimeType.RAMP_DOWN:
            # Smooth ramp from prev_level to near-zero
            low = rng.uniform(0.0, 0.02) * max_feed
            t_norm = np.linspace(0, 1, dur)
            segment = prev_level + (low - prev_level) * _cosine_ease(t_norm)
            this_level = low

        elif regime == RegimeType.ACTIVE:
            # Active at target level with slight variation
            variation = rng.normal(0, 0.01 * target_feed, dur)
            segment = np.clip(target_feed + variation, 0.3 * target_feed, 1.1 * max_feed)
            _smooth_blend(segment, prev_level, float(segment[0]), blend_steps)
            this_level = float(segment[-1])

        elif regime == RegimeType.PERIODIC:
            # Oscillating between low and high feed
            freq = rng.uniform(cfg_regime.min_periodic_freq, cfg_regime.max_periodic_freq)
            phase = rng.uniform(0, 2 * np.pi)
            t_arr = np.arange(dur) / dur
            lo_feed = rng.uniform(0.1, 0.3) * target_feed
            hi_feed = rng.uniform(0.7, 1.0) * target_feed
            mid = (lo_feed + hi_feed) / 2
            amp = (hi_feed - lo_feed) / 2
            segment = mid + amp * np.sin(2 * np.pi * freq * dur * t_arr + phase)
            _smooth_blend(segment, prev_level, float(segment[0]), blend_steps)
            this_level = float(segment[-1])

        elif regime == RegimeType.RECOVERY:
            # Exponential decay to near-zero
            decay_rate = rng.uniform(0.03, 0.12)
            t_arr = np.arange(dur)
            low = rng.uniform(0.0, 0.02) * max_feed
            segment = prev_level * np.exp(-decay_rate * t_arr) + low
            this_level = float(segment[-1])

        else:
            segment = np.full(dur, prev_level)
            this_level = prev_level

        envelope[start:end] = segment
        meta.append(RegimeMeta(
            regime=regime,
            start=start,
            end=end,
            target_level=float(np.mean(np.abs(segment))) / max_feed,
            frequency=None if regime != RegimeType.PERIODIC else freq,
            phase=None if regime != RegimeType.PERIODIC else phase,
        ))
        prev_level = this_level
        cursor = end

    # Add slow low-frequency drift across the whole session
    drift = _low_freq_drift(T, cfg_regime.drift_amplitude * max_feed, rng)
    envelope += drift
    envelope = np.clip(envelope, 0.0, None)  # feed speed is non-negative
    return envelope, meta

def smooth_regime_boundaries(
    x: npt.NDArray[np.float64],
    regime_sequence: list[RegimeMeta],
) -> npt.NDArray[np.float64]:
    """Remove one-sample joins left by sensor noise at regime boundaries.

    The causal envelope is continuous, but independent sensor noise can make
    the first sample after a regime boundary look like a step.  Interpolating
    that sample from its immediate neighbors preserves the regime waveform
    while keeping transitions smooth.
    """
    result = np.asarray(x, dtype=np.float64).copy()
    T = result.shape[1]
    for regime in regime_sequence[:-1]:
        boundary = int(regime.end)
        if 0 < boundary < T - 1:
            result[:, boundary] = 0.5 * (
                result[:, boundary - 1] + result[:, boundary + 1]
            )
    return result


def _cosine_ease(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Cosine ease-in-out: smooth S-curve from 0 to 1."""
    return (1 - np.cos(np.pi * t)) / 2


def _smooth_blend(
    segment: npt.NDArray[np.float64],
    from_level: float,
    to_level: float,
    blend_steps: int,
) -> None:
    """In-place cosine blend at the start of a segment."""
    if blend_steps <= 0 or blend_steps > len(segment):
        return
    t = np.linspace(0, 1, blend_steps)
    blend = from_level + (to_level - from_level) * _cosine_ease(t)
    segment[:blend_steps] = blend


def _low_freq_drift(T: int, amplitude: float, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Sinusoidal low-frequency drift with random phase/frequency."""
    freq = rng.uniform(0.001, 0.01)  # very low freq relative to T
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(T)
    return amplitude * np.sin(2 * np.pi * freq * t + phase) * rng.uniform(0.3, 1.0)
