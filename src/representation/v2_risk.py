"""Calibrated anomaly confidence and failure-risk layer (Sprint 11 Task 15).

Two deliberately separate outputs:

- :class:`HealthyTailCalibrator` — empirical/conformal healthy-tail
  anomaly confidence: how unusual a state is relative to verified-healthy
  behavior for the same context. This controls false alerts; it is NOT
  failure probability.
- :class:`CensoredSurvivalRisk` — discrete-time censored survival layer
  producing one-day and seven-day failure probabilities from historical
  trajectory features. Censored files (no observed failure within the
  recorded horizon) contribute survival likelihood honestly; horizon
  ordering ``risk_7d >= risk_1d`` holds by construction.

Only historical trajectory features enter either layer — future targets
and labels never do. ``v2_risk`` imports no supervision-bearing synth
types; use the ``*_mask``/``event`` tensor arguments explicitly.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from representation.v2_contracts import validate_confidence, validate_risk


class HealthyTailCalibrator:
    """Empirical healthy-tail anomaly confidence (conformal-style).

    Fit exclusively on verified-healthy dev-validation scores. For a query
    score ``s``::

        p_normal = (1 + #{healthy_j >= s}) / (V + 1)
        confidence = 1 - p_normal

    High scores (far tail) yield confidence near one; typical healthy
    scores yield confidence near zero.

    The fit cohort is recorded in the state (cohort label, sample count,
    small-sample status) and persisted separately from the operating
    decision threshold cohort: pooling train data or mislabeling the
    cohort raises instead of silently fitting.
    """

    #: Reference floor for the small-sample disclosure flag. Fits at or
    #: above ``min_samples`` are valid, but cohorts below this conventional
    #: size (e.g. an 11-file dev-val) are flagged so downstream readers do
    #: not mistake the fit for a large-sample calibration.
    REFERENCE_FLOOR: int = 32

    def __init__(self, min_samples: int = 32) -> None:
        if min_samples < 1:
            raise ValueError("min_samples must be positive")
        self.min_samples = int(min_samples)
        self._scores: torch.Tensor | None = None
        self._fit_cohort: str | None = None

    @staticmethod
    def _require_dev_val_cohort(cohort: str) -> str:
        """Return the cohort label after enforcing dev-val-only fitting."""
        label = str(cohort)
        normalized = label.strip().lower()
        if not normalized.startswith("dev-val"):
            raise ValueError(
                "confidence-calibrator fitting requires a verified-healthy "
                f"dev-val cohort (got {cohort!r}); pooled train data is rejected"
            )
        if "train" in normalized or "pool" in normalized:
            raise ValueError(
                f"confidence-calibrator fitting must not use train or pooled "
                f"data (got {cohort!r}); dev-val only"
            )
        return label

    def fit(self, healthy_scores: torch.Tensor, *, cohort: str) -> "HealthyTailCalibrator":
        """Store sorted verified-healthy dev-validation scores for tail lookup.

        ``cohort`` names the exact fit cohort and MUST denote verified-healthy
        dev-validation data (e.g. ``"dev-val"``); train/pooled labels raise.
        """
        label = self._require_dev_val_cohort(cohort)
        flat = healthy_scores.detach().to(dtype=torch.float32).reshape(-1)
        if flat.numel() < self.min_samples:
            raise ValueError(
                f"calibration requires >= {self.min_samples} healthy scores, "
                f"got {flat.numel()}"
            )
        if not torch.isfinite(flat).all():
            raise ValueError("healthy calibration scores must be finite")
        self._scores = torch.sort(flat).values
        self._fit_cohort = label
        return self

    def confidence(self, scores: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return anomaly confidence and empirical ``p_normal`` for queries."""
        if self._scores is None:
            raise ValueError("calibrator is not fitted")
        flat = scores.detach().to(dtype=torch.float32)
        if not torch.isfinite(flat).all():
            raise ValueError("query scores must be finite")
        ref = self._scores
        v = ref.numel()
        # #{healthy_j >= s} via sorted search (right side).
        order = torch.bucketize(flat.reshape(-1), ref, right=True)
        greater_equal = v - order
        p_normal = (1.0 + greater_equal.to(dtype=torch.float32)) / (v + 1.0)
        out: dict[str, torch.Tensor] = {
            "confidence": (1.0 - p_normal).reshape(flat.shape),
            "p_normal": p_normal.reshape(flat.shape),
        }
        validate_confidence(out)
        return out

    def coverage(self, healthy_scores: torch.Tensor, *, level: float = 0.95) -> float:
        """Return the fraction of healthy queries with ``p_normal >= 1-level``.

        Conformal-style self-check: on fresh healthy data the empirical
        tail values are near-uniform, so coverage should sit near ``level``.
        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        out = self.confidence(healthy_scores)
        return float((out["p_normal"] >= (1.0 - level)).float().mean().item())

    def fit_cohort_info(self) -> dict[str, object]:
        """Return the persisted fit-cohort provenance record."""
        if self._scores is None:
            raise ValueError("calibrator is not fitted")
        n_samples = int(self._scores.numel())
        return {
            "cohort": self._fit_cohort,
            "n_samples": n_samples,
            "min_samples": int(self.min_samples),
            "small_sample": bool(n_samples < self.REFERENCE_FLOOR),
        }

    def state_dict(self) -> dict[str, object]:
        if self._scores is None:
            raise ValueError("calibrator is not fitted")
        return {
            "min_samples": self.min_samples,
            "scores": self._scores.clone(),
            "fit_cohort": self._fit_cohort,
            "n_samples": int(self._scores.numel()),
            "small_sample": bool(int(self._scores.numel()) < self.REFERENCE_FLOOR),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        scores = state.get("scores")
        if not isinstance(scores, torch.Tensor) or scores.ndim != 1:
            raise ValueError("calibrator state must carry 1-D scores")
        self.min_samples = int(state.get("min_samples", self.min_samples))
        self._scores = scores.detach().to(dtype=torch.float32).clone()
        cohort = state.get("fit_cohort")
        self._fit_cohort = None if cohort is None else str(cohort)


class _HorizonLogit(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).reshape(-1)


class CensoredSurvivalRisk:
    """Discrete-time censored survival for fixed 1-day/7-day horizons.

    One logistic head per horizon maps historical trajectory features to
    the cumulative failure probability within that horizon. Censoring is
    handled per horizon: a file censored (or failed) before horizon ``H``
    without an observed failure inside ``H`` is excluded from head ``H``;
    a file observed failure-free through ``H`` is a negative; a file
    failing within ``H`` is a positive. Ordering is enforced at predict
    time via ``risk_7d = max(risk_7d_raw, risk_1d)``.
    """

    def __init__(
        self,
        n_features: int,
        *,
        horizons_days: tuple[int, int] = (1, 7),
        l2: float = 1e-3,
    ) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if len(horizons_days) != 2 or not horizons_days[0] < horizons_days[1]:
            raise ValueError("horizons_days must be an increasing (short, long) pair")
        if l2 < 0.0:
            raise ValueError("l2 must be non-negative")
        self.n_features = int(n_features)
        self.horizons_days = (int(horizons_days[0]), int(horizons_days[1]))
        self.l2 = float(l2)
        self._heads: dict[int, _HorizonLogit] | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    # -- fitting -----------------------------------------------------
    @staticmethod
    def horizon_cohort(
        days_to_event: torch.Tensor,
        event_observed: torch.Tensor,
        horizon_days: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(included_mask, label)`` for one horizon.

        Standard discrete-time survival encoding: ``days_to_event`` is the
        observed failure time when ``event_observed`` is true, else the
        right-censoring time (last failure-free follow-up). A file
        censored before the horizon carries no information about it and
        is excluded; a file observed failure-free through the horizon is
        a negative. NaN times (e.g. ``FutureFailureTargets.is_censored``
        files with no follow-up recorded) are treated as censored before
        every horizon and excluded.
        """
        if days_to_event.shape != event_observed.shape:
            raise ValueError("days_to_event and event_observed must share shape")
        if event_observed.dtype is not torch.bool:
            raise ValueError("event_observed must have torch.bool dtype")
        horizon = float(horizon_days)
        finite_time = torch.isfinite(days_to_event)
        time = torch.where(finite_time, days_to_event, torch.full_like(days_to_event, float("-inf")))
        observed_in = event_observed & finite_time & (time <= horizon)
        survived_through = finite_time & (~event_observed) & (time >= horizon)
        # Observed failures outside the horizon are informative negatives
        # for it (no failure *within* the horizon).
        failed_later = event_observed & finite_time & (time > horizon)
        included = observed_in | survived_through | failed_later
        if bool((event_observed & ~finite_time).any()):
            raise ValueError("observed events must carry a finite days_to_event")
        label = observed_in
        return included, label

    def fit(
        self,
        features: torch.Tensor,
        days_to_event: torch.Tensor,
        event_observed: torch.Tensor,
        *,
        steps: int = 500,
        lr: float = 0.1,
        seed: int = 0,
    ) -> "CensoredSurvivalRisk":
        """Fit one calibrated logistic head per horizon (deterministic)."""
        if features.ndim != 2 or features.shape[1] != self.n_features:
            raise ValueError(f"features must be [N, {self.n_features}]")
        n = features.shape[0]
        if tuple(days_to_event.shape) != (n,) or tuple(event_observed.shape) != (n,):
            raise ValueError("days_to_event/event_observed must have shape [N]")
        if not torch.isfinite(features).all():
            raise ValueError("risk features must be finite")
        gen = torch.Generator().manual_seed(int(seed))
        x = features.detach().to(dtype=torch.float32)
        self._mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
        self._std = torch.where(torch.isfinite(std), std, torch.ones_like(std))
        xn = (x - self._mean) / self._std
        heads: dict[int, _HorizonLogit] = {}
        for horizon in self.horizons_days:
            included, label = self.horizon_cohort(days_to_event, event_observed, horizon)
            if int(included.sum().item()) < 2:
                raise ValueError(f"horizon {horizon}d has fewer than 2 usable files")
            if bool((label[included].sum() == 0).item()) or bool(
                (label[included].sum() == included.sum()).item()
            ):
                raise ValueError(
                    f"horizon {horizon}d cohort is single-class; "
                    "cannot calibrate a risk head"
                )
            head = _HorizonLogit(self.n_features)
            for p in head.parameters():
                nn.init.normal_(p, std=0.01, generator=gen)
            opt = torch.optim.LBFGS(head.parameters(), max_iter=20)
            xb = xn[included]
            yb = label[included].to(dtype=torch.float32)

            def closure() -> torch.Tensor:
                assert opt is not None
                opt.zero_grad()
                logits = head(xb)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
                loss = loss + self.l2 * sum(p.pow(2).sum() for p in head.parameters())
                loss.backward()
                return loss

            for _ in range(max(1, steps // 20)):
                opt.step(closure)
            # Deterministic polish with plain full-batch steps.
            opt2 = torch.optim.SGD(head.parameters(), lr=lr)
            for _ in range(steps % 20 + 5):
                opt2.zero_grad()
                logits = head(xb)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
                loss = loss + self.l2 * sum(p.pow(2).sum() for p in head.parameters())
                loss.backward()
                opt2.step()
            heads[horizon] = head
        self._heads = heads
        return self

    def predict_proba(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ordered ``risk_1d``/``risk_7d`` failure probabilities."""
        if self._heads is None or self._mean is None or self._std is None:
            raise ValueError("risk layer is not fitted")
        x = features.detach().to(dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"features must be [N, {self.n_features}]")
        if not torch.isfinite(x).all():
            raise ValueError("risk features must be finite")
        xn = (x - self._mean) / self._std
        short, long = self.horizons_days
        with torch.no_grad():
            r_short = torch.sigmoid(self._heads[short](xn))
            r_long = torch.sigmoid(self._heads[long](xn))
        risk_1d = r_short
        risk_7d = torch.maximum(r_long, r_short)
        out = {"risk_1d": risk_1d, "risk_7d": risk_7d}
        validate_risk(out)
        return out

    def brier_score(
        self,
        features: torch.Tensor,
        days_to_event: torch.Tensor,
        event_observed: torch.Tensor,
    ) -> dict[str, float]:
        """Return per-horizon Brier scores on the usable (uncensored) cohort."""
        proba = self.predict_proba(features)
        scores: dict[str, float] = {}
        for horizon, key in zip(self.horizons_days, ("risk_1d", "risk_7d")):
            included, label = self.horizon_cohort(days_to_event, event_observed, horizon)
            if int(included.sum().item()) == 0:
                raise ValueError(f"horizon {horizon}d has no usable files")
            y = label[included].to(dtype=torch.float32)
            p = proba[key][included]
            scores[key] = float(((p - y) ** 2).mean().item())
        return scores

    # -- persistence ---------------------------------------------------
    def state_dict(self) -> dict[str, object]:
        if self._heads is None or self._mean is None or self._std is None:
            raise ValueError("risk layer is not fitted")
        return {
            "n_features": self.n_features,
            "horizons_days": list(self.horizons_days),
            "l2": self.l2,
            "mean": self._mean.clone(),
            "std": self._std.clone(),
            "heads": {h: m.state_dict() for h, m in self._heads.items()},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("n_features") != self.n_features:
            raise ValueError("risk state n_features mismatch")
        mean = state.get("mean")
        std = state.get("std")
        heads = state.get("heads")
        if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
            raise ValueError("risk state must carry mean/std tensors")
        if not isinstance(heads, dict):
            raise ValueError("risk state must carry per-horizon heads")
        self._mean = mean.detach().to(dtype=torch.float32).clone()
        self._std = std.detach().to(dtype=torch.float32).clone()
        restored: dict[int, _HorizonLogit] = {}
        for horizon in self.horizons_days:
            payload = heads.get(horizon, heads.get(str(horizon)))
            if payload is None:
                raise ValueError(f"risk state is missing horizon {horizon}d head")
            head = _HorizonLogit(self.n_features)
            head.load_state_dict(payload)
            restored[horizon] = head
        self._heads = restored


def trajectory_feature_matrix(
    trajectory: dict[str, torch.Tensor],
    tail_energy: torch.Tensor,
    elevated_fraction: torch.Tensor,
    *,
    time_since_maintenance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assemble the historical risk feature matrix for one file batch.

    Columns are ``[displacement, velocity, trend, persistence,
    disagreement, tail_energy, elevated_fraction, (time_since_maintenance)]``.
    Every input is a same-``[B]`` historical output — future targets and
    labels are not arguments and cannot leak through this constructor.
    """
    names = ("displacement", "velocity", "trend", "persistence", "disagreement")
    cols = []
    batch: int | None = None
    for name in (*names,):
        tensor = trajectory[name]
        if tensor.ndim != 1:
            raise ValueError(f"trajectory[{name!r}] must have shape [B]")
        batch = tensor.shape[0] if batch is None else batch
        cols.append(tensor.detach().to(dtype=torch.float32).reshape(-1, 1))
    for name, tensor in (("tail_energy", tail_energy), ("elevated_fraction", elevated_fraction)):
        if tuple(tensor.shape) != (batch,):
            raise ValueError(f"{name} must have shape [B]")
        cols.append(tensor.detach().to(dtype=torch.float32).reshape(-1, 1))
    if time_since_maintenance is not None:
        if tuple(time_since_maintenance.shape) != (batch,):
            raise ValueError("time_since_maintenance must have shape [B]")
        cols.append(time_since_maintenance.detach().to(dtype=torch.float32).reshape(-1, 1))
    matrix = torch.cat(cols, dim=1)
    if not torch.isfinite(matrix).all():
        raise ValueError("risk features must be finite")
    return matrix


def expected_feature_width(*, with_maintenance_time: bool = False) -> int:
    """Return the canonical risk feature width (7, or 8 with maintenance time)."""
    return 8 if with_maintenance_time else 7
