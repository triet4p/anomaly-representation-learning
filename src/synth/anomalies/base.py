"""
Base types for the anomaly injection API.

Every injector receives an InjectionContext and returns an AnomalyResult.
All injectors are pure functions of (context, severity, rng) — no global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from synth.schema import RegimeMeta, AnomalyMeta, AnomalyFamily


@dataclass
class InjectionContext:
    """
    Input to every anomaly injector.

    x:            float64 [C, T] clean causal signal (before final float32 cast)
    regimes:      ordered regime metadata for this session
    rng:          numpy Generator (caller provides; injector advances it)
    dt:           sampling interval (seconds)
    robot_temp_offset: thermal offset for causal recomputation
    """
    x: npt.NDArray[np.float64]    # [C, T]
    regimes: list[RegimeMeta]
    rng: np.random.Generator
    dt: float = 0.125
    robot_temp_offset: float = 0.0


@dataclass
class AnomalyResult:
    """
    Output of every anomaly injector.

    x_modified: float64 [C, T] signal after injection (T may change for a
    duration intervention)
    mask:       bool [C, T] True where anomaly is active
    meta:       AnomalyMeta with exact provenance
    accepted:   False if strength gate rejected this attempt
    regimes_modified: updated metadata when an intervention changes T
    """
    x_modified: npt.NDArray[np.float64]
    mask: npt.NDArray[np.bool_]
    meta: AnomalyMeta
    accepted: bool = True
    regimes_modified: list[RegimeMeta] | None = None


# ---------------------------------------------------------------------------
# Shared utilities used by multiple injectors
# ---------------------------------------------------------------------------

def cosine_blend(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    n_blend: int,
) -> npt.NDArray[np.float64]:
    """
    Cosine-blend the first n_blend samples of 'b' from 'a[-1]' to 'b[n_blend-1]'.
    Returns a modified copy of 'b'.
    """
    result = b.copy()
    if n_blend <= 0 or len(b) == 0:
        return result
    n = min(n_blend, len(b))
    t = np.linspace(0, 1, n)
    alpha = (1 - np.cos(np.pi * t)) / 2   # 0→1 cosine
    start_val = a[-1] if len(a) > 0 else b[0]
    result[:n] = start_val * (1 - alpha) + result[:n] * alpha
    return result


def cosine_blend_end(
    b: npt.NDArray[np.float64],
    c_first: float,
    n_blend: int,
) -> npt.NDArray[np.float64]:
    """
    Cosine-blend the last n_blend samples of 'b' towards 'c_first'.
    """
    result = b.copy()
    if n_blend <= 0 or len(b) == 0:
        return result
    n = min(n_blend, len(b))
    t = np.linspace(0, 1, n)
    alpha = (1 - np.cos(np.pi * t)) / 2   # 0→1 cosine
    end_val = result[-n - 1] if len(result) > n else result[0]
    result[-n:] = end_val * (1 - alpha) + c_first * alpha
    return result


def local_stats(
    x: npt.NDArray[np.float64],
    start: int,
    end: int,
    channel: int,
) -> tuple[float, float]:
    """Return (mean, std) of x[channel, start:end]."""
    seg = x[channel, start:end]
    return float(np.mean(seg)), float(np.std(seg) + 1e-8)


def measure_intervention_strength(
    x_orig: npt.NDArray[np.float64],
    x_mod: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> float:
    """
    Normalised mean absolute deviation in the anomaly region.

    strength = mean|x_mod - x_orig| / (mean|x_orig| in region + 1e-6)
    """
    if not mask.any():
        return 0.0
    diff = np.abs(x_mod[mask] - x_orig[mask])
    baseline = np.abs(x_orig[mask])
    return float(np.mean(diff) / (np.mean(baseline) + 1e-6))


def find_dynamic_region(
    x: npt.NDArray[np.float64],
    channel: int,
    min_local_std: float,
    min_duration: int,
) -> list[tuple[int, int]]:
    """
    Find contiguous regions where local std > min_local_std.
    Returns list of (start, end) tuples (exclusive end).
    """
    from scipy.ndimage import uniform_filter1d
    sig = x[channel]
    T = len(sig)
    window = 20
    pad = window // 2
    padded = np.pad(sig, pad, mode="reflect")
    m = uniform_filter1d(padded.astype(float), size=window, mode="reflect")
    m2 = uniform_filter1d((padded**2).astype(float), size=window, mode="reflect")
    var = np.maximum(m2 - m**2, 0.0)
    local_std = np.sqrt(var)[pad:pad + T]

    is_dynamic = local_std > min_local_std
    regions = []
    in_region = False
    start = 0
    for i in range(T):
        if is_dynamic[i] and not in_region:
            start = i
            in_region = True
        elif not is_dynamic[i] and in_region:
            if i - start >= min_duration:
                regions.append((start, i))
            in_region = False
    if in_region and T - start >= min_duration:
        regions.append((start, T))
    return regions
