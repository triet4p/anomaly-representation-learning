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

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import numpy as np
import numpy.typing as npt

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
    FileSample, SampleLabel, AnomalyMeta, AnomalyFamily, RegimeMeta
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


PAIRED_TWIN_DOMAIN = "s24p-paired-twin-v1"


def derive_pair_seed(history_seed: int, pair_index: int) -> int:
    """Derive a deterministic pair seed under the paired-twin domain tag."""
    digest = hashlib.sha256(
        f"s24p-paired-v1:{int(history_seed)}:{int(pair_index):05d}".encode("utf-8")
    ).hexdigest()
    return int(digest[:15], 16) % (2**31)


def _require_pair_seed(pair_seed: int) -> int:
    if isinstance(pair_seed, bool) or not isinstance(pair_seed, int):
        raise ValueError(f"pair_seed must be an integer, got {pair_seed!r}")
    if pair_seed < 0:
        raise ValueError(f"pair_seed must be non-negative, got {pair_seed}")
    return pair_seed


def _require_twin_severity(severity: float) -> float:
    value = float(severity)
    if not 0.0 <= value <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    return value


@dataclass(frozen=True)
class PairedTwin:
    """One public paired-twin draw: an unmodified twin plus one variant."""
    pair_seed: int
    twin: FileSample
    abnormal: FileSample
    disposition: str
    family: AnomalyFamily
    severity: float
    group_id: str
    source_id: str
    n_attempts: int
    strength: float
    normal_path_diverges: bool


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

    # ------------------------------------------------------------------ #
    # Paired-twin constructor (Sprint 24 Task 7-M16)                       #
    # ------------------------------------------------------------------ #

    def _anomaly_source(self, seed: int) -> dict[str, object]:
        """Rebuild the exact common pre-injection source of generate_anomaly."""
        rng_base = np.random.default_rng(seed)
        pcfg = self.cfg.physics
        rcfg = self.cfg.regime
        C = self.cfg.n_channels
        normal_seed = int(rng_base.integers(0, 2**31))
        anom_seed = int(rng_base.integers(0, 2**31))
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
        noise_seed = int(rng_normal.integers(0, 2**31))
        return {
            "normal_seed": normal_seed,
            "anom_seed": anom_seed,
            "noise_seed": noise_seed,
            "x_clean": x_clean,
            "regime_meta": regime_meta,
            "robot_idx": robot_idx,
            "program_idx": program_idx,
            "robot_code": robot_code,
            "program_number": program_number,
            "robot_temp_offset": robot_temp_offset,
        }

    @staticmethod
    def _noise_copy(x: npt.NDArray[np.float64], seed: int, cfg: SynthConfig) -> npt.NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        return SensorNoiseGenerator(cfg.noise, rng).add_all(np.asarray(x, dtype=np.float64))

    def generate_paired_twin(
        self,
        pair_seed: int,
        family: AnomalyFamily | str,
        severity: float,
        split: str = "test",
    ) -> PairedTwin:
        """Build the unmodified twin and requested anomaly from one source."""
        seed = _require_pair_seed(pair_seed)
        if isinstance(family, str):
            try:
                family = AnomalyFamily(family)
            except ValueError:
                raise ValueError(f"Unknown anomaly family: {family!r}")
        if family is None or not isinstance(family, AnomalyFamily):
            raise ValueError(f"family must name a supported anomaly family, got {family!r}")
        if family in EASY_FAMILIES and not self.cfg.anomaly.include_easy_sanity:
            raise ValueError(
                f"{family.value} is opt-in; set anomaly.include_easy_sanity=True"
            )
        value = _require_twin_severity(severity)
        acfg = self.cfg.anomaly
        source = self._anomaly_source(seed)
        x_clean = source["x_clean"]
        assert isinstance(x_clean, np.ndarray)
        regime_meta = source["regime_meta"]
        assert isinstance(regime_meta, list)
        robot_temp_offset = float(source["robot_temp_offset"])  # type: ignore[arg-type]
        anom_seed = int(source["anom_seed"])  # type: ignore[arg-type]
        noise_seed = int(source["noise_seed"])  # type: ignore[arg-type]
        counter = 0

        def make_attempt() -> object:
            nonlocal counter
            attempt_rng = np.random.default_rng(anom_seed + counter)
            counter += 1
            ctx = InjectionContext(
                x=x_clean.copy(),
                regimes=regime_meta,
                rng=attempt_rng,
                robot_temp_offset=robot_temp_offset,
            )
            return inject_anomaly(family, ctx, value, acfg)

        result, strength, n_attempts = generate_with_strength_gate(
            make_attempt, x_clean, acfg
        )
        twin_regimes: list[RegimeMeta] = list(regime_meta)
        abnormal_regimes = result.regimes_modified if result.regimes_modified is not None else list(regime_meta)
        twin_noisy = self._noise_copy(x_clean, noise_seed, self.cfg)
        twin_noisy = smooth_regime_boundaries(twin_noisy, twin_regimes)
        twin_sample = FileSample(
            x=twin_noisy.astype(np.float32),
            file_id=f"T-{split}-{PAIRED_TWIN_DOMAIN}-{seed:010d}",
            file_label=SampleLabel.NORMAL,
            seed=seed,
            generator_version=GENERATOR_VERSION,
            config_hash=self._config_hash,
            regime_sequence=[RegimeMeta(
                regime=item.regime, start=item.start, end=item.end,
                target_level=item.target_level, frequency=item.frequency, phase=item.phase,
            ) for item in twin_regimes],
            robot_idx=int(source["robot_idx"]),  # type: ignore[arg-type]
            program_idx=int(source["program_idx"]),  # type: ignore[arg-type]
            robot_code=str(source["robot_code"]),
            program_number=str(source["program_number"]),
        )
        twin_sample.validate()
        abnormal_noisy = self._noise_copy(result.x_modified, noise_seed, self.cfg)
        abnormal_noisy = smooth_regime_boundaries(abnormal_noisy, abnormal_regimes)
        source_id = hashlib.sha256(
            f"{PAIRED_TWIN_DOMAIN}:{seed}:{self._config_hash}:"
            f"{twin_sample.T}:{(twin_sample.robot_idx, twin_sample.program_idx)}".encode()
        ).hexdigest()[:16]
        group_id = f"{PAIRED_TWIN_DOMAIN}:{seed:010d}"
        if not result.accepted:
            meta = result.meta
            meta.extra["strength"] = float(strength)
            meta.extra["n_attempts"] = n_attempts
            meta.extra["accepted"] = False
            meta.extra["pair_seed"] = seed
            meta.extra["group_id"] = group_id
            meta.extra["source_id"] = source_id
            meta.extra["disposition"] = "NORMAL_RETURN"
            declined = FileSample(
                x=abnormal_noisy.astype(np.float32),
                file_id=f"R-{split}-{PAIRED_TWIN_DOMAIN}-{seed:010d}",
                file_label=SampleLabel.NORMAL,
                seed=seed,
                generator_version=GENERATOR_VERSION,
                config_hash=self._config_hash,
                regime_sequence=[RegimeMeta(
                    regime=item.regime, start=item.start, end=item.end,
                    target_level=item.target_level, frequency=item.frequency, phase=item.phase,
                ) for item in abnormal_regimes],
                rejection_meta=meta,
                robot_idx=twin_sample.robot_idx,
                program_idx=twin_sample.program_idx,
                robot_code=twin_sample.robot_code,
                program_number=twin_sample.program_number,
            )
            declined.validate()
            return PairedTwin(
                pair_seed=seed, twin=twin_sample, abnormal=declined,
                disposition="NORMAL_RETURN", family=family, severity=value,
                group_id=group_id, source_id=source_id, n_attempts=n_attempts,
                strength=float(strength),
                normal_path_diverges=self._normal_path_diverges(
                    self.generate_normal(seed=seed, split=split), source
                ),
            )
        duration_note = ""
        if family == AnomalyFamily.DURATION_ANOMALY:
            duration_note = (
                "duration_anomaly changes T/regimes; this pair is within-family only, "
                "not a length-matched cross-family comparison."
            )
        meta = result.meta
        meta.extra["strength"] = float(strength)
        meta.extra["n_attempts"] = n_attempts
        meta.extra["accepted"] = True
        meta.extra["pair_seed"] = seed
        meta.extra["group_id"] = group_id
        meta.extra["source_id"] = source_id
        meta.extra["disposition"] = "PAIRED"
        if duration_note:
            meta.extra["duration_pairing_note"] = duration_note
        abnormal_sample = FileSample(
            x=abnormal_noisy.astype(np.float32),
            file_id=_anomaly_file_id(seed, family.value, split, value, self._config_hash),
            file_label=SampleLabel.ABNORMAL,
            seed=seed,
            generator_version=GENERATOR_VERSION,
            config_hash=self._config_hash,
            regime_sequence=abnormal_regimes,
            anomaly_meta=meta,
            anomaly_mask=result.mask.astype(bool),
            robot_idx=twin_sample.robot_idx,
            program_idx=twin_sample.program_idx,
            robot_code=twin_sample.robot_code,
            program_number=twin_sample.program_number,
        )
        abnormal_sample.validate()
        return PairedTwin(
            pair_seed=seed, twin=twin_sample, abnormal=abnormal_sample,
            disposition="PAIRED", family=family, severity=value,
            group_id=group_id, source_id=source_id, n_attempts=n_attempts,
            strength=float(strength),
            normal_path_diverges=self._normal_path_diverges(
                self.generate_normal(seed=seed, split=split), source
            ),
        )

    def generate_paired_family_sweep(
        self,
        pair_seed: int,
        families: Sequence[AnomalyFamily | str],
        severity: float,
        split: str = "test",
    ) -> list[PairedTwin]:
        """Build one twin plus one variant per requested family at one severity."""
        seed = _require_pair_seed(pair_seed)
        names = list(families)
        if not names:
            raise ValueError("families must list at least one anomaly family")
        return [self.generate_paired_twin(seed, item, severity, split) for item in names]

    def generate_paired_severity_sweep(
        self,
        pair_seed: int,
        family: AnomalyFamily | str,
        severities: Sequence[float],
        split: str = "test",
    ) -> list[PairedTwin]:
        """Build one twin per severity level of one family from one source."""
        seed = _require_pair_seed(pair_seed)
        levels = list(severities)
        if not levels:
            raise ValueError("severities must list at least one severity in [0, 1]")
        return [self.generate_paired_twin(seed, family, item, split) for item in levels]

    def _normal_path_diverges(
        self, probe: FileSample, source: dict[str, object]
    ) -> bool:
        width = int(source["x_clean"].shape[1])  # type: ignore[union-attr]
        return bool(
            probe.T != width
            or (probe.robot_idx, probe.program_idx)
            != (int(source["robot_idx"]), int(source["program_idx"]))  # type: ignore[arg-type]
        )
