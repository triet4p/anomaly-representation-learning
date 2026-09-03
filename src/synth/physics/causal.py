"""Causal process physics for the supported three- and six-channel layouts.

The first three channels are the original feed/current/temperature chain.  The
production six-channel layout adds arc voltage, arc power, and torch pressure;
those channels are derived from the same process variables rather than being
independent random signals.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import TransferFunction, lsim

from synth.config import PhysicsConfig, SUPPORTED_CHANNEL_COUNTS


class CausalSignalGenerator:
    """Generate a coherent causal signal with exactly 3 or 6 channels."""

    def __init__(
        self,
        config: PhysicsConfig,
        rng: np.random.Generator,
        channel_gain: npt.NDArray[np.float64] | None = None,
        channel_off: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.cfg = config
        self.rng = rng
        if channel_gain is None and channel_off is None:
            self.n_channels = 6
            self._gain = np.ones(self.n_channels, dtype=np.float64)
            self._off = np.zeros(self.n_channels, dtype=np.float64)
        else:
            candidate = channel_gain if channel_gain is not None else channel_off
            assert candidate is not None
            self.n_channels = int(len(candidate))
            if self.n_channels not in SUPPORTED_CHANNEL_COUNTS:
                raise ValueError(f"channel arrays must have 3 or 6 entries, got {self.n_channels}")
            self._gain = (np.ones(self.n_channels) if channel_gain is None
                          else np.asarray(channel_gain, dtype=np.float64))
            self._off = (np.zeros(self.n_channels) if channel_off is None
                         else np.asarray(channel_off, dtype=np.float64))
        if len(self._gain) != self.n_channels or len(self._off) != self.n_channels:
            raise ValueError("channel_gain and channel_off must have the same supported length")
        self._build_lti()

    def _build_lti(self) -> None:
        wn, z, K = self.cfg.lti_natural_freq, self.cfg.lti_damping_ratio, self.cfg.lti_dc_gain
        self._tf = TransferFunction([K * wn**2], [1.0, 2.0 * z * wn, wn**2])

    def _compute_current(self, feed_speed: npt.NDArray[np.float64], dt: float,
                         damping_ratio: float | None = None) -> npt.NDArray[np.float64]:
        n = len(feed_speed)
        if n == 0:
            return np.array([], dtype=np.float64)
        if damping_ratio is not None and damping_ratio != self.cfg.lti_damping_ratio:
            wn, K = self.cfg.lti_natural_freq, self.cfg.lti_dc_gain
            tf = TransferFunction([K * wn**2], [1.0, 2.0 * damping_ratio * wn, wn**2])
        else:
            tf = self._tf
        delay = self.cfg.lti_delay_steps
        norm = feed_speed / self.cfg.lti_max_feed
        delayed = np.zeros_like(norm)
        if delay > 0:
            delayed[delay:] = norm[:-delay]
        else:
            delayed = norm
        _, yout, _ = lsim(tf, delayed, T=np.arange(n) * dt)
        amplitude = self.cfg.lti_max_feed * self.cfg.lti_target_current_ratio
        return np.asarray(yout).reshape(-1) * amplitude + 5.0

    def _compute_temperature(self, welding_current: npt.NDArray[np.float64],
                             robot_temp_offset: float = 0.0,
                             initial_temp: float | None = None) -> npt.NDArray[np.float64]:
        n = len(welding_current)
        if n == 0:
            return np.array([], dtype=np.float64)
        cfg = self.cfg
        equilibrium = (cfg.thermal_ambient + cfg.thermal_current_to_temp_gain * welding_current
                       + robot_temp_offset)
        temp = np.empty(n, dtype=np.float64)
        temp[0] = cfg.thermal_ambient + robot_temp_offset if initial_temp is None else initial_temp
        for i in range(1, n):
            alpha = cfg.thermal_alpha_heat if equilibrium[i] > temp[i - 1] else cfg.thermal_alpha_cool
            temp[i] = temp[i - 1] + alpha * (equilibrium[i] - temp[i - 1])
        return temp

    def _shared_factor(self, n: int) -> npt.NDArray[np.float64]:
        if n == 0:
            return np.array([], dtype=np.float64)
        phi, amp = self.cfg.shared_factor_ar1_phi, self.cfg.shared_factor_amplitude
        noise = self.rng.standard_normal(n)
        fac = np.empty(n, dtype=np.float64)
        fac[0] = noise[0] * amp
        scale = amp * np.sqrt(max(0.0, 1.0 - phi**2))
        for i in range(1, n):
            fac[i] = phi * fac[i - 1] + noise[i] * scale
        return fac

    def _base_channels(self, feed: npt.NDArray[np.float64], dt: float,
                       robot_temp_offset: float, damping_ratio: float | None) -> list[npt.NDArray[np.float64]]:
        current = self._compute_current(feed, dt, damping_ratio)
        temp = self._compute_temperature(current, robot_temp_offset)
        if self.n_channels == 3:
            return [feed.copy(), current, temp]
        # Sensor noise is applied separately; these are deterministic process
        # observables with causal relationships to feed/current.
        voltage = 19.0 + 0.012 * feed + 0.09 * current
        power = np.maximum(current, 0.0) * np.maximum(voltage, 1.0) / 20.0
        pressure = 1.0 + 0.0025 * feed + 0.012 * current
        pressure += 0.02 * np.tanh((current - np.mean(current)) / (np.std(current) + 1e-6))
        return [feed.copy(), current, temp, voltage, power, pressure]

    def generate(self, feed_envelope: npt.NDArray[np.float64], dt: float = 0.125,
                 robot_temp_offset: float = 0.0,
                 damping_ratio: float | None = None) -> npt.NDArray[np.float64]:
        channels = self._base_channels(np.asarray(feed_envelope, dtype=np.float64), dt,
                                        robot_temp_offset, damping_ratio)
        shared = self._shared_factor(len(feed_envelope))
        x = np.stack(channels, axis=0)
        for c in range(self.n_channels):
            x[c] = x[c] * self._gain[c] + self._off[c]
            x[c] += shared * max(float(np.std(x[c])), 1e-3)
        return x

    def recompute_all_from_s0(self, s0: npt.NDArray[np.float64], dt: float = 0.125,
                              robot_temp_offset: float = 0.0,
                              damping_ratio: float | None = None) -> npt.NDArray[np.float64]:
        """Recompute every downstream channel from a modified feed trace."""
        return np.stack(self._base_channels(np.asarray(s0, dtype=np.float64), dt,
                                             robot_temp_offset, damping_ratio), axis=0)

    def recompute_from_s0(self, s0: npt.NDArray[np.float64], dt: float = 0.125,
                          robot_temp_offset: float = 0.0,
                          damping_ratio: float | None = None) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Backward-compatible return of current and temperature downstream traces."""
        channels = self._base_channels(np.asarray(s0, dtype=np.float64), dt,
                                        robot_temp_offset, damping_ratio)
        return channels[1], channels[2]

    @classmethod
    def from_rng_seed(cls, seed: int, config: PhysicsConfig) -> "CausalSignalGenerator":
        return cls(config, np.random.default_rng(seed))
