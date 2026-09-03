"""
Anomaly registry: maps AnomalyFamily → injector function.

Provides ``inject_anomaly(family, ctx, severity, cfg)`` as the single
dispatch point used by SessionGenerator.
"""

from __future__ import annotations

from synth.anomalies.base import InjectionContext, AnomalyResult
from synth.anomalies.contextual import inject_contextual_replacement
from synth.anomalies.transition import inject_wrong_transition
from synth.anomalies.stuck import inject_realistic_stuck
from synth.anomalies.regularity import inject_over_regularity
from synth.anomalies.drift import inject_subtle_drift
from synth.anomalies.freq_phase import inject_freq_phase_mismatch
from synth.anomalies.cross_channel import inject_cross_channel_inconsistency
from synth.anomalies.duration import inject_duration_anomaly
from synth.anomalies.missing_event import inject_missing_event
from synth.anomalies.easy_sanity import inject_easy_spike, inject_easy_flatline
from synth.config import AnomalyConfig
from synth.schema import AnomalyFamily


# Maps family → injector callable
ANOMALY_REGISTRY: dict[AnomalyFamily, object] = {
    AnomalyFamily.CONTEXTUAL_REPLACEMENT:    inject_contextual_replacement,
    AnomalyFamily.WRONG_TRANSITION:          inject_wrong_transition,
    AnomalyFamily.REALISTIC_STUCK:           inject_realistic_stuck,
    AnomalyFamily.OVER_REGULARITY:           inject_over_regularity,
    AnomalyFamily.SUBTLE_DRIFT:              inject_subtle_drift,
    AnomalyFamily.FREQ_PHASE_MISMATCH:       inject_freq_phase_mismatch,
    AnomalyFamily.CROSS_CHANNEL_INCONSISTENCY: inject_cross_channel_inconsistency,
    AnomalyFamily.DURATION_ANOMALY:          inject_duration_anomaly,
    AnomalyFamily.MISSING_EVENT:             inject_missing_event,
    AnomalyFamily.EASY_SPIKE:               inject_easy_spike,
    AnomalyFamily.EASY_FLATLINE:            inject_easy_flatline,
}

HARD_FAMILIES: list[AnomalyFamily] = [
    AnomalyFamily.CONTEXTUAL_REPLACEMENT,
    AnomalyFamily.WRONG_TRANSITION,
    AnomalyFamily.REALISTIC_STUCK,
    AnomalyFamily.OVER_REGULARITY,
    AnomalyFamily.SUBTLE_DRIFT,
    AnomalyFamily.FREQ_PHASE_MISMATCH,
    AnomalyFamily.CROSS_CHANNEL_INCONSISTENCY,
    AnomalyFamily.DURATION_ANOMALY,
    AnomalyFamily.MISSING_EVENT,
]

EASY_FAMILIES: list[AnomalyFamily] = [
    AnomalyFamily.EASY_SPIKE,
    AnomalyFamily.EASY_FLATLINE,
]


def inject_anomaly(
    family: AnomalyFamily | str,
    ctx: InjectionContext,
    severity: float,
    cfg: AnomalyConfig,
) -> AnomalyResult:
    """
    Dispatch to the correct injector.

    Args:
        family:   AnomalyFamily enum or string name
        ctx:      InjectionContext carrying the clean signal and RNG
        severity: continuous [0, 1]
        cfg:      AnomalyConfig

    Returns:
        AnomalyResult with x_modified, mask, meta, accepted flag.
    """
    if isinstance(family, str):
        family = AnomalyFamily(family)
    fn = ANOMALY_REGISTRY.get(family)
    if fn is None:
        raise ValueError(f"Unknown anomaly family: {family!r}")
    return fn(ctx, severity, cfg)  # type: ignore[call-arg]
