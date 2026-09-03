"""
Normal session generator.

Produces variable-length FileSamples with:
- Multiple valid regime sequences
- Smooth transitions
- Cross-channel causal structure (S0→S1→S2)
- Natural per-channel gain/offset/noise variation
- Shared latent drift factor
- Deterministic from (seed, config)
"""

from __future__ import annotations

import hashlib
import numpy as np

from synth.config import (
    SynthConfig, GENERATOR_VERSION, channel_offset_scales,
)
from synth.physics.causal import CausalSignalGenerator
from synth.physics.noise import SensorNoiseGenerator
from synth.regimes import sample_regime_sequence, build_feed_envelope, smooth_regime_boundaries
from synth.schema import FileSample, SampleLabel


def _file_id(seed: int, split: str, config_hash: str) -> str:
    tag = hashlib.sha256(
        f"normal:{split}:{seed}:{config_hash}".encode()
    ).hexdigest()[:10]
    return f"N-{split}-{tag}"


class NormalGenerator:
    """
    Generates a single normal FileSample from a seed.

    All randomness is derived from the seed; the generator is fully
    deterministic for the same (seed, SynthConfig).
    """

    def __init__(self, config: SynthConfig) -> None:
        self.cfg = config
        self._config_hash = config.hash()

    def generate(self, seed: int, split: str = "train") -> FileSample:
        """Generate one normal session."""
        rng = np.random.default_rng(seed)

        # ── Natural per-channel variation ──────────────────────────────
        pcfg = self.cfg.physics
        ncfg = self.cfg.noise
        rcfg = self.cfg.regime
        C = self.cfg.n_channels

        gain = 1.0 + rng.uniform(-pcfg.channel_gain_variation, pcfg.channel_gain_variation, C)
        off_frac = rng.uniform(-pcfg.channel_offset_variation, pcfg.channel_offset_variation, C)

        # Robot temperature offset (natural variation)
        robot_temp_offset = rng.uniform(-3.0, 8.0)

        # ── Regime sequence + feed envelope ────────────────────────────
        seq_pairs = sample_regime_sequence(rng, rcfg)
        T = sum(d for _, d in seq_pairs)
        feed_envelope, regime_meta = build_feed_envelope(seq_pairs, rcfg, pcfg, rng)

        # Channel offsets in physical units, preserving semantics for C=3/6.
        off_abs = off_frac * channel_offset_scales(C, pcfg)

        # ── Causal signal generation ───────────────────────────────────
        causal_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        causal = CausalSignalGenerator(pcfg, causal_rng, channel_gain=gain, channel_off=off_abs)
        x = causal.generate(feed_envelope, robot_temp_offset=robot_temp_offset)  # [C, T], C=3 or 6

        # ── Natural sensor noise ───────────────────────────────────────
        noise_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        noiser = SensorNoiseGenerator(ncfg, noise_rng)
        x = noiser.add_all(x)
        x = smooth_regime_boundaries(x, regime_meta)
        x = x.astype(np.float32)

        return FileSample(
            x=x,
            file_id=_file_id(seed, split, self._config_hash),
            file_label=SampleLabel.NORMAL,
            seed=seed,
            generator_version=GENERATOR_VERSION,
            config_hash=self._config_hash,
            regime_sequence=regime_meta,
        )
