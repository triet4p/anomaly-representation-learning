"""
Top-level session generator: orchestrates normal + anomaly session creation
with strength gating, metadata, and reproducibility.

One SessionGenerator instance serves all splits.  Callers specify split,
seed, and optionally anomaly family; the generator handles:
  - Normal generation (fully deterministic from seed)
  - Anomaly injection with strength-gate retry loop
  - FileSample assembly with complete metadata/mask
  - float32 cast at the very end
"""

from __future__ import annotations

import hashlib
import numpy as np

from synth.config import SynthConfig, GENERATOR_VERSION, AnomalyConfig, channel_offset_scales
from synth.normal import NormalGenerator
from synth.anomalies.base import InjectionContext
from synth.anomalies.registry import (
    inject_anomaly, HARD_FAMILIES, EASY_FAMILIES, ANOMALY_REGISTRY
)
from synth.physics.causal import CausalSignalGenerator
from synth.physics.noise import SensorNoiseGenerator
from synth.regimes import sample_regime_sequence, build_feed_envelope, smooth_regime_boundaries
from synth.schema import (
    FileSample, SampleLabel, AnomalyMeta, AnomalyFamily
)
from synth.strength import generate_with_strength_gate


def _anomaly_file_id(
    seed: int, family: str, split: str, severity: float, config_hash: str
) -> str:
    canonical = f"{severity:.9g}"
    tag = hashlib.sha256(
        f"anomaly:{split}:{family}:{seed}:{canonical}:{config_hash}".encode()
    ).hexdigest()[:10]
    return f"A-{split}-{family[:4]}-{tag}"


class SessionGenerator:
    """
    Generates FileSample objects (normal and anomalous) deterministically.

    Usage::

        gen = SessionGenerator(SynthConfig())
        normal  = gen.generate_normal(seed=42, split="train")
        anomaly = gen.generate_anomaly(
            seed=42, split="test",
            family=AnomalyFamily.REALISTIC_STUCK,
            severity=0.7,
        )

    The same (seed, family, severity, config) always produces byte-identical
    output.  Changing any argument changes the output.
    """

    def __init__(self, config: SynthConfig | None = None) -> None:
        self.cfg = config or SynthConfig()
        self._config_hash = self.cfg.hash()
        self._normal_gen = NormalGenerator(self.cfg)

    # ------------------------------------------------------------------ #
    # Normal generation                                                   #
    # ------------------------------------------------------------------ #

    def generate_normal(self, seed: int, split: str = "train") -> FileSample:
        """Generate one normal session.  Fully deterministic from seed."""
        return self._normal_gen.generate(seed=seed, split=split)

    # ------------------------------------------------------------------ #
    # Anomaly generation                                                  #
    # ------------------------------------------------------------------ #

    def generate_anomaly(
        self,
        seed: int,
        split: str = "test",
        family: AnomalyFamily | str | None = None,
        severity: float | None = None,
    ) -> FileSample:
        """
        Generate one anomalous session.

        The process:
          1. Generate a clean normal session (normal_seed derived from seed)
          2. Build the injection context from the clean signal
          3. Run the anomaly injector with strength-gate retry loop
          4. Assemble and return the FileSample

        If family is None, pick uniformly from HARD_FAMILIES.
        If severity is None, sample uniformly from [0.3, 0.9].

        The RNG for injection is seeded independently from the normal
        generation RNG, ensuring anomaly variations don't perturb normal data.
        """
        rng_base = np.random.default_rng(seed)
        acfg = self.cfg.anomaly

        if family is None:
            family = AnomalyFamily(
                str(rng_base.choice([f.value for f in HARD_FAMILIES]))
            )
        elif isinstance(family, str):
            family = AnomalyFamily(family)
        if severity is None:
            severity = float(rng_base.uniform(0.3, 0.9))
        if family in EASY_FAMILIES and not acfg.include_easy_sanity:
            raise ValueError(
                f"{family.value} is opt-in; set anomaly.include_easy_sanity=True"
            )
        if not 0.0 <= float(severity) <= 1.0:
            raise ValueError("severity must be in [0, 1]")

        # ── Build clean causal signal (same as NormalGenerator) ───────
        pcfg = self.cfg.physics
        ncfg = self.cfg.noise
        rcfg = self.cfg.regime
        acfg = self.cfg.anomaly
        C = self.cfg.n_channels

        # Normal seed is deterministically derived — does not depend on
        # anomaly family/severity so normal base is stable.
        normal_seed = int(rng_base.integers(0, 2**31))
        rng_normal = np.random.default_rng(normal_seed)
        fcfg = self.cfg.fleet
        robot_idx = int(rng_normal.integers(0, fcfg.n_robots))
        program_idx = int(rng_normal.integers(0, fcfg.n_programs))
        robot_code = f"R{robot_idx+1:02d}"
        program_number = f"P{100 + program_idx * 10}"

        robot_rng = np.random.default_rng(100_000 + robot_idx)
        robot_gain_bias = robot_rng.normal(1.0, fcfg.robot_gain_std, C)
        robot_temp_offset = float(robot_rng.uniform(fcfg.robot_temp_offset_range[0], fcfg.robot_temp_offset_range[1]))

        gain = (1.0 + rng_normal.uniform(-pcfg.channel_gain_variation, pcfg.channel_gain_variation, C)) * robot_gain_bias
        off_frac = rng_normal.uniform(-pcfg.channel_offset_variation, pcfg.channel_offset_variation, C)
        seq_pairs = sample_regime_sequence(rng_normal, rcfg)
        off_abs = off_frac * channel_offset_scales(C, pcfg)
        feed_envelope, regime_meta = build_feed_envelope(seq_pairs, rcfg, pcfg, rng_normal)
        speed_scale = fcfg.program_speed_scales[program_idx % len(fcfg.program_speed_scales)]
        feed_envelope = feed_envelope * speed_scale

        causal_rng = np.random.default_rng(int(rng_normal.integers(0, 2**31)))
        causal = CausalSignalGenerator(pcfg, causal_rng, channel_gain=gain, channel_off=off_abs)
        x_clean = causal.generate(feed_envelope, robot_temp_offset=robot_temp_offset)
        x_clean = smooth_regime_boundaries(x_clean, regime_meta)
        # ── Anomaly injection seed (independent from normal seed) ──────
        anom_seed = int(rng_base.integers(0, 2**31))

        def make_attempt() -> object:
            attempt_rng = np.random.default_rng(anom_seed + make_attempt.counter)
            make_attempt.counter += 1
            ctx = InjectionContext(
                x=x_clean.copy(),
                regimes=regime_meta,
                rng=attempt_rng,
                robot_temp_offset=robot_temp_offset,
            )
            return inject_anomaly(family, ctx, severity, acfg)
        make_attempt.counter = 0

        result, strength, n_attempts = generate_with_strength_gate(
            make_attempt, x_clean, acfg
        )
        if result.regimes_modified is not None:
            regime_meta = result.regimes_modified


        # ── Add noise to the modified signal ──────────────────────────
        noise_seed = int(rng_normal.integers(0, 2**31))
        noise_rng = np.random.default_rng(noise_seed)
        noiser = SensorNoiseGenerator(ncfg, noise_rng)
        x_noisy = noiser.add_all(result.x_modified)  # float64
        x_noisy = smooth_regime_boundaries(x_noisy, regime_meta)

        x_out = x_noisy.astype(np.float32)
        file_id = _anomaly_file_id(
            seed, family.value, split, float(severity), self._config_hash
        )
        rejected_meta = None
        if not result.accepted:
            result.meta.extra["strength"] = float(strength)
            result.meta.extra["n_attempts"] = n_attempts
            result.meta.extra["accepted"] = False
            rejected_meta = result.meta
        sample = FileSample(
            x=x_out,
            file_id=file_id,
            file_label=SampleLabel.ABNORMAL if result.accepted else SampleLabel.NORMAL,
            seed=seed,
            generator_version=GENERATOR_VERSION,
            config_hash=self._config_hash,
            regime_sequence=regime_meta,
            anomaly_meta=result.meta if result.accepted else None,
            anomaly_mask=result.mask.astype(bool) if result.accepted else None,
            rejection_meta=rejected_meta,
            robot_idx=robot_idx,
            program_idx=program_idx,
            robot_code=robot_code,
            program_number=program_number,
        )
        # Attach generation diagnostics to meta.extra
        if result.accepted and sample.anomaly_meta is not None:
            sample.anomaly_meta.extra["strength"] = float(strength)
            sample.anomaly_meta.extra["n_attempts"] = n_attempts
            sample.anomaly_meta.extra["accepted"] = True

        return sample

    # ------------------------------------------------------------------ #
    # Convenience: generate a batch                                       #
    # ------------------------------------------------------------------ #

    def generate_normal_batch(
        self, seeds: list[int], split: str = "train"
    ) -> list[FileSample]:
        return [self.generate_normal(s, split) for s in seeds]

    def generate_anomaly_batch(
        self,
        seeds: list[int],
        split: str = "test",
        families: list[AnomalyFamily | str] | None = None,
        severity: float | None = None,
    ) -> list[FileSample]:
        if families is None:
            families = [None] * len(seeds)  # type: ignore[list-item]
        return [
            self.generate_anomaly(s, split, fam, severity)
            for s, fam in zip(seeds, families)
        ]
