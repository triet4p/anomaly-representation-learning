"""
Natural sensor noise for each channel.

Noise is kept strictly separate from anomaly generation:
  S0: sinusoidal mechanical vibration + AR(1) momentum noise
  S1: AR(1) with local-variance scaling
  S2: leaky random walk + rare glitch spikes

An additional cross-channel shared AR(1) factor is applied in
CausalSignalGenerator to give correlated baseline drift.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d

from synth.config import NoiseConfig


class SensorNoiseGenerator:
    """Generates per-channel natural noise.  One instance per session."""

    def __init__(self, config: NoiseConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _local_std(signal: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
        n = len(signal)
        if n <= window:
            return np.full(n, np.std(signal))
        pad = window // 2
        padded = np.pad(signal, pad, mode="reflect")
        m = uniform_filter1d(padded.astype(float), size=window, mode="reflect")
        m2 = uniform_filter1d((padded ** 2).astype(float), size=window, mode="reflect")
        var = np.maximum(m2 - m ** 2, 0.0)
        return np.sqrt(var)[pad: pad + n]

    def _ar1(
        self,
        n: int,
        phi: float,
        step_std: float | npt.NDArray[np.float64],
        init: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        if n == 0:
            return np.array([], dtype=np.float64)
        innov = self.rng.standard_normal(n)
        if isinstance(step_std, np.ndarray):
            innov *= step_std
        else:
            innov *= float(step_std)
        noise = np.empty(n, dtype=np.float64)
        noise[0] = init + innov[0]
        for i in range(1, n):
            noise[i] = phi * noise[i - 1] + innov[i]
        return noise

    # ------------------------------------------------------------------ #
    # Per-channel noise                                                    #
    # ------------------------------------------------------------------ #

    def add_s0_noise(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Wire-feed speed: sinusoidal vibration + AR(1) gaussian noise."""
        n = len(signal)
        if n == 0:
            return signal.copy()
        cfg = self.cfg
        t = np.arange(n)
        mask = np.abs(signal) > 1
        sig_amp = np.mean(np.abs(signal[mask])) if mask.any() else 3000.0
        phase = self.rng.uniform(0, 2 * np.pi)
        vibration = (
            cfg.s0_vibration_amplitude
            * sig_amp
            * np.sin(2.0 * np.pi * cfg.s0_vibration_freq * t + phase)
        )
        local_std = self._local_std(signal, cfg.local_window)
        gauss_scale = cfg.s0_gaussian_std * np.where(local_std > 0, local_std, sig_amp)
        gaussian = self._ar1(n, phi=cfg.ar1_phi, step_std=gauss_scale)
        return signal + vibration + gaussian

    def add_s1_noise(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Welding current: AR(1) arc noise with local-variance scaling."""
        n = len(signal)
        if n == 0:
            return signal.copy()
        cfg = self.cfg
        sig_amp = max(np.mean(np.abs(signal)), 1.0)
        local_std = self._local_std(signal, cfg.local_window)
        blended = np.maximum(local_std, sig_amp * 0.01)
        noise_scale = cfg.s1_arc_noise_std * blended
        noise = self._ar1(n, phi=cfg.ar1_phi, step_std=noise_scale)
        return signal + noise

    def add_s2_noise(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Temperature: leaky random walk + rare glitch spikes."""
        n = len(signal)
        if n == 0:
            return signal.copy()
        cfg = self.cfg
        local_std = self._local_std(signal, cfg.local_window)
        step_scale = np.maximum(local_std / (np.mean(local_std) + 1e-8), 0.1)
        rw_std = cfg.s2_rw_step_std * step_scale
        rw_steps = self.rng.normal(0, rw_std, n)
        rw = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            rw[i] = 0.999 * rw[i - 1] + rw_steps[i]
        glitch_mask = self.rng.random(n) < cfg.s2_glitch_prob
        glitches = glitch_mask * self.rng.normal(cfg.s2_glitch_magnitude, cfg.s2_glitch_magnitude * 0.3, n)
        return signal + rw + glitches

    def add_all(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply natural noise to a ``[C, T]`` signal (C is 3 or 6)."""
        if x.ndim != 2 or x.shape[0] not in (3, 6):
            raise ValueError("noise expects a [C, T] array with C equal to 3 or 6")
        result = x.copy()
        result[0] = self.add_s0_noise(x[0])
        result[1] = self.add_s1_noise(x[1])
        result[2] = self.add_s2_noise(x[2])
        for c in range(3, x.shape[0]):
            local_std = self._local_std(x[c], self.cfg.local_window)
            scale = np.maximum(local_std, float(np.std(x[c])) * 0.01 + 1e-6)
            # Added sensors use the same correlated sensor-noise family but
            # retain their own physical scale through local variance.
            result[c] = x[c] + self._ar1(
                x.shape[1], phi=self.cfg.ar1_phi,
                step_std=0.02 * scale,
            )
        return result
