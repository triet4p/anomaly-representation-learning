"""Longitudinal baseline and trajectory tracker (Sprint 11 Task 14).

Maintains a frozen fixed commissioning baseline (verified-healthy
history, reset only at documented maintenance boundaries) alongside a
guarded short-term operational baseline that never updates from
suspect/warning files and never replaces the fixed baseline. Reports
regularized-Mahalanobis absolute displacement and velocity, rolling
trend, one-sided CUSUM persistence, and fixed-versus-short-term
disagreement. Suspect files are scored but update neither reference.
"""

from __future__ import annotations

from collections import deque

import torch

from representation.v2_contracts import validate_trajectory


def _regularized_covariance(
    block: torch.Tensor, parent: torch.Tensor, shrinkage: float, eps: float
) -> torch.Tensor:
    dim = block.shape[1]
    if block.shape[0] < 2:
        return parent + eps * torch.eye(dim, dtype=torch.float32)
    centered = block - block.mean(dim=0, keepdim=True)
    sample = (centered.T @ centered) / max(1, block.shape[0] - 1)
    sample = torch.where(torch.isfinite(sample), sample, torch.zeros_like(sample))
    cov = (1.0 - shrinkage) * sample + shrinkage * parent + eps * torch.eye(
        dim, dtype=torch.float32
    )
    return cov


def _mahalanobis_squared(
    rows: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor
) -> torch.Tensor:
    diff = rows - mu
    try:
        solved = torch.linalg.solve(cov, diff.T).T
    except (RuntimeError, torch._C._LinAlgError):  # type: ignore[attr-defined]
        solved = diff / torch.diagonal(cov).clamp_min(1e-6).unsqueeze(0)
    out = (diff * solved).sum(dim=-1)
    return torch.where(torch.isfinite(out), out, torch.full_like(out, 1e6))


class TrajectoryTracker:
    """Per-episode longitudinal monitoring for one (robot, program) stream.

    One tracker instance serves one chronological file stream; call
    :meth:`reset_episode` at every maintenance/re-commissioning boundary
    (trajectories never bridge maintenance). Episodes are selected by the
    caller — typically one tracker per robot, reset on maintenance.
    """

    def __init__(
        self,
        d_model: int,
        *,
        shrinkage: float = 0.2,
        covariance_eps: float = 1e-4,
        trend_window: int = 8,
        cusum_kappa: float = 1.0,
        short_term_window: int = 16,
        short_term_min_samples: int = 4,
    ) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if covariance_eps <= 0.0:
            raise ValueError("covariance_eps must be positive")
        if trend_window < 2 or short_term_window < 2 or short_term_min_samples < 2:
            raise ValueError("trend/short-term windows and floors must be >= 2")
        self.d_model = int(d_model)
        self.shrinkage = float(shrinkage)
        self.eps = float(covariance_eps)
        self.trend_window = int(trend_window)
        self.cusum_kappa = float(cusum_kappa)
        self.short_term_window = int(short_term_window)
        self.short_term_min_samples = int(short_term_min_samples)
        self._fixed_mu: torch.Tensor | None = None
        self._fixed_cov: torch.Tensor | None = None
        self._short: deque[torch.Tensor] = deque(maxlen=self.short_term_window)
        self._disp_history: deque[float] = deque(maxlen=self.trend_window)
        self._prev_state: torch.Tensor | None = None
        self._cusum = 0.0
        self.n_updates = 0

    # -- commissioning -------------------------------------------------
    def fit_commissioning(
        self, file_states: torch.Tensor, healthy_mask: torch.Tensor
    ) -> "TrajectoryTracker":
        """Freeze the fixed baseline from verified-healthy commissioning files."""
        if file_states.ndim != 2 or file_states.shape[1] != self.d_model:
            raise ValueError(f"file_states must be [M, {self.d_model}]")
        if tuple(healthy_mask.shape) != (file_states.shape[0],):
            raise ValueError("healthy_mask must have shape [M]")
        if healthy_mask.dtype is not torch.bool:
            raise ValueError("healthy_mask must have torch.bool dtype")
        rows = file_states.detach().to(dtype=torch.float32)
        if not bool(healthy_mask.any()):
            raise ValueError("commissioning requires at least one verified-healthy file")
        if not torch.isfinite(rows[healthy_mask]).all():
            raise ValueError("healthy commissioning states must be finite")
        healthy = rows[healthy_mask]
        self._fixed_mu = healthy.mean(dim=0)
        parent = torch.eye(self.d_model, dtype=torch.float32)
        self._fixed_cov = _regularized_covariance(
            healthy, parent, self.shrinkage, self.eps
        )
        return self

    def reset_episode(self) -> None:
        """Start a new episode after maintenance: clear adaptive state only.

        The fixed commissioning baseline is preserved across the call; the
        caller refits it only when a documented re-commissioning provides
        fresh verified-healthy history (via :meth:`fit_commissioning`).
        """
        self._short.clear()
        self._disp_history.clear()
        self._prev_state = None
        self._cusum = 0.0

    # -- monitoring ----------------------------------------------------
    def update(
        self,
        file_state: torch.Tensor,
        *,
        is_suspect: bool = False,
        maintenance_reset: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Score one chronological file state and advance guarded references.

        Suspect files (warnings/uncertain) are scored identically but update
        neither the fixed nor the short-term reference — this is the
        absorption guard. ``maintenance_reset`` applies :meth:`reset_episode`
        before scoring so the post-maintenance file starts a new segment.
        """
        if self._fixed_mu is None or self._fixed_cov is None:
            raise ValueError("tracker has no commissioning baseline; call fit_commissioning")
        if maintenance_reset:
            self.reset_episode()
        state = file_state.detach().to(dtype=torch.float32).reshape(self.d_model)
        if state.numel() != self.d_model or not torch.isfinite(state).all():
            raise ValueError(f"file_state must be a finite [{self.d_model}] vector")
        assert self._fixed_mu is not None and self._fixed_cov is not None

        disp = float(_mahalanobis_squared(state.unsqueeze(0), self._fixed_mu, self._fixed_cov).item())
        if self._prev_state is None:
            vel = 0.0
        else:
            vel = float(
                _mahalanobis_squared(
                    (state - self._prev_state).unsqueeze(0),
                    torch.zeros_like(state),
                    self._fixed_cov,
                ).item()
            )
        self._disp_history.append(disp)
        hist = list(self._disp_history)
        if len(hist) < 2:
            trend = 0.0
        else:
            xs = torch.arange(len(hist), dtype=torch.float64)
            ys = torch.tensor(hist, dtype=torch.float64)
            x_mean, y_mean = xs.mean(), ys.mean()
            denom = ((xs - x_mean) ** 2).sum()
            trend = float((((xs - x_mean) * (ys - y_mean)).sum() / denom).item()) if denom > 0 else 0.0
        self._cusum = max(0.0, self._cusum + disp - self.cusum_kappa)

        if len(self._short) >= self.short_term_min_samples:
            block = torch.stack(list(self._short))
            short_mu = block.mean(dim=0)
            short_cov = _regularized_covariance(
                block, self._fixed_cov, self.shrinkage, self.eps
            )
            short_disp = float(
                _mahalanobis_squared(state.unsqueeze(0), short_mu, short_cov).item()
            )
            disagreement = abs(disp - short_disp)
        else:
            disagreement = 0.0

        if not is_suspect:
            self._short.append(state.clone())
            self.n_updates += 1
        self._prev_state = state.clone()

        features: dict[str, torch.Tensor] = {
            "displacement": torch.tensor([disp], dtype=torch.float32),
            "velocity": torch.tensor([vel], dtype=torch.float32),
            "trend": torch.tensor([trend], dtype=torch.float32),
            "persistence": torch.tensor([self._cusum], dtype=torch.float32),
            "disagreement": torch.tensor([disagreement], dtype=torch.float32),
        }
        validate_trajectory(features)
        if not torch.isfinite(features["disagreement"]).all() or bool(
            (features["disagreement"] < 0).any()
        ):
            raise ValueError("disagreement must be finite and non-negative")
        return features

    # -- persistence ---------------------------------------------------
    def state_dict(self) -> dict[str, object]:
        """Return serializable tracker state (fixed baseline + counters)."""
        if self._fixed_mu is None or self._fixed_cov is None:
            raise ValueError("tracker has no commissioning baseline to serialize")
        return {
            "d_model": self.d_model,
            "shrinkage": self.shrinkage,
            "covariance_eps": self.eps,
            "trend_window": self.trend_window,
            "cusum_kappa": self.cusum_kappa,
            "short_term_window": self.short_term_window,
            "short_term_min_samples": self.short_term_min_samples,
            "fixed_mu": self._fixed_mu.clone(),
            "fixed_cov": self._fixed_cov.clone(),
            "short": [s.clone() for s in self._short],
            "disp_history": list(self._disp_history),
            "prev_state": None if self._prev_state is None else self._prev_state.clone(),
            "cusum": self._cusum,
            "n_updates": self.n_updates,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore tracker state saved by :meth:`state_dict`."""
        if state.get("d_model") != self.d_model:
            raise ValueError("tracker state d_model mismatch")
        mu = state.get("fixed_mu")
        cov = state.get("fixed_cov")
        if not isinstance(mu, torch.Tensor) or not isinstance(cov, torch.Tensor):
            raise ValueError("tracker state must carry fixed_mu/fixed_cov tensors")
        self._fixed_mu = mu.detach().to(dtype=torch.float32).clone()
        self._fixed_cov = cov.detach().to(dtype=torch.float32).clone()
        short = state.get("short")
        if not isinstance(short, list):
            raise ValueError("tracker state must carry a short-term list")
        self._short = deque(
            [s.detach().to(dtype=torch.float32).clone() for s in short],
            maxlen=self.short_term_window,
        )
        hist = state.get("disp_history")
        if not isinstance(hist, list):
            raise ValueError("tracker state must carry disp_history")
        self._disp_history = deque([float(v) for v in hist], maxlen=self.trend_window)
        prev = state.get("prev_state")
        self._prev_state = (
            None if prev is None else prev.detach().to(dtype=torch.float32).clone()
        )
        self._cusum = float(state.get("cusum", 0.0))
        self.n_updates = int(state.get("n_updates", 0))
