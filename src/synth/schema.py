"""
Core data contracts for the synthetic data subsystem.

Every generated file/session is a FileSample.  Anomaly samples carry
AnomalyMeta describing the intervention exactly.  Patch operations produce
PatchBatch with explicit start indices and a timestep↔patch mask map.
"""

from __future__ import annotations

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
