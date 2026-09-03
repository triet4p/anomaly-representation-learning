"""Anomaly injection package."""
from synth.anomalies.registry import ANOMALY_REGISTRY, inject_anomaly
from synth.anomalies.base import AnomalyResult, InjectionContext

__all__ = ["ANOMALY_REGISTRY", "inject_anomaly", "AnomalyResult", "InjectionContext"]
