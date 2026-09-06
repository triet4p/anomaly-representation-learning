"""Hierarchical regularized Mahalanobis geometry (Sprint 11 Task 11).

Estimates verified-healthy ``(robot, program, regime)`` references with
strict fallback ``(robot, program)`` -> ``robot`` -> explicit low-confidence
fleet. Covariance uses shrinkage toward the parent plus an epsilon floor,
diagonal fallback for sparse groups, and mixture-density energy over regime
components. References are frozen for monitoring; suspect files never update
them. There is no program-only cross-robot sharing.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch

GroupKey = tuple[int, int, int]


@dataclass
class GroupStats:
    """One fitted reference: mean, regularized covariance, and provenance."""

    mu: torch.Tensor  # [D] detached CPU float32
    cov: torch.Tensor  # [D, D] detached CPU float32, PD by construction
    n: int
    level: str
    low_confidence: bool


def _empirical_covariance(rows: torch.Tensor) -> torch.Tensor:
    centered = rows - rows.mean(dim=0, keepdim=True)
    denom = max(1, rows.shape[0] - 1)
    return (centered.T @ centered) / denom


def _regularize(
    sample_cov: torch.Tensor,
    parent_cov: torch.Tensor,
    shrinkage: float,
    eps: float,
) -> torch.Tensor:
    dim = sample_cov.shape[0]
    eye = torch.eye(dim, dtype=sample_cov.dtype)
    shrunk = (1.0 - shrinkage) * sample_cov + shrinkage * parent_cov
    return shrunk + eps * eye


def _safe_cholesky(cov: torch.Tensor) -> torch.Tensor | None:
    try:
        return torch.linalg.cholesky(cov)
    except (RuntimeError, torch._C._LinAlgError):  # type: ignore[attr-defined]
        return None


class HierarchicalMahalanobisGeometry:
    """Fit-once hierarchical references with frozen monitoring behavior."""

    def __init__(
        self,
        d_model: int,
        shrinkage: float = 0.2,
        covariance_eps: float = 1e-4,
        min_group_samples: int = 8,
        diag_min_samples: int = 32,
    ) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if covariance_eps <= 0.0:
            raise ValueError("covariance_eps must be positive")
        if min_group_samples < 2 or diag_min_samples < min_group_samples:
            raise ValueError("require 2 <= min_group_samples <= diag_min_samples")
        self.d_model = d_model
        self.shrinkage = float(shrinkage)
        self.eps = float(covariance_eps)
        self.min_group_samples = int(min_group_samples)
        self.diag_min_samples = int(diag_min_samples)
        self._groups: dict[GroupKey, GroupStats] = {}
        self._pair: dict[tuple[int, int], GroupStats] = {}
        self._robot: dict[int, GroupStats] = {}
        self._fleet: GroupStats | None = None
        self._frozen = False

    # -- fitting ---------------------------------------------------------
    def fit(
        self,
        latents: torch.Tensor,
        robot_ids: torch.Tensor,
        program_ids: torch.Tensor,
        regime_ids: torch.Tensor,
        healthy_mask: torch.Tensor,
    ) -> "HierarchicalMahalanobisGeometry":
        """Fit references from verified-healthy rows only."""
        if self._frozen:
            raise ValueError("frozen references cannot be refit")
        if latents.ndim != 2 or latents.shape[1] != self.d_model:
            raise ValueError(f"latents must be [M, {self.d_model}]")
        m = latents.shape[0]
        for name, tensor in (
            ("robot_ids", robot_ids),
            ("program_ids", program_ids),
            ("regime_ids", regime_ids),
            ("healthy_mask", healthy_mask),
        ):
            if tuple(tensor.shape) != (m,):
                raise ValueError(f"{name} must have shape [{m}]")
        if healthy_mask.dtype is not torch.bool:
            raise ValueError("healthy_mask must have torch.bool dtype")
        rows = latents.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(rows[healthy_mask]).all():
            raise ValueError("healthy reference latents must be finite")
        if not bool(healthy_mask.any()):
            raise ValueError("fit requires at least one verified-healthy row")

        healthy = rows[healthy_mask]
        fleet_mu = healthy.mean(dim=0)
        fleet_cov = self._group_covariance(healthy, None)
        self._fleet = GroupStats(fleet_mu, fleet_cov, int(healthy.shape[0]), "fleet_low_confidence", True)

        for robot in sorted({int(v) for v in robot_ids[healthy_mask].tolist()}):
            sel = healthy_mask & (robot_ids == robot)
            block = rows[sel]
            self._robot[robot] = GroupStats(
                block.mean(dim=0),
                self._group_covariance(block, fleet_cov),
                int(block.shape[0]),
                "robot",
                False,
            )
        for robot in list(self._robot):
            programs = sorted({int(v) for v in program_ids[healthy_mask & (robot_ids == robot)].tolist()})
            for program in programs:
                sel = healthy_mask & (robot_ids == robot) & (program_ids == program)
                block = rows[sel]
                parent = self._robot[robot].cov
                self._pair[(robot, program)] = GroupStats(
                    block.mean(dim=0),
                    self._group_covariance(block, parent),
                    int(block.shape[0]),
                    "robot_program",
                    False,
                )
                regimes = sorted({int(v) for v in regime_ids[sel].tolist()})
                for regime in regimes:
                    gsel = sel & (regime_ids == regime)
                    block_g = rows[gsel]
                    self._groups[(robot, program, regime)] = GroupStats(
                        block_g.mean(dim=0),
                        self._group_covariance(block_g, self._pair[(robot, program)].cov),
                        int(block_g.shape[0]),
                        "robot_program_regime",
                        False,
                    )
        return self

    def _group_covariance(self, block: torch.Tensor, parent: torch.Tensor | None) -> torch.Tensor:
        dim = self.d_model
        if parent is None:
            parent = torch.eye(dim, dtype=torch.float32)
        if block.shape[0] < self.diag_min_samples:
            var = block.var(dim=0, unbiased=False) + self.eps
            var = torch.where(torch.isfinite(var), var, torch.ones_like(var))
            diag = torch.diag(var)
            return _regularize(diag, parent, self.shrinkage, self.eps)
        sample = _empirical_covariance(block)
        sample = torch.where(torch.isfinite(sample), sample, torch.zeros_like(sample))
        cov = _regularize(sample, parent, self.shrinkage, self.eps)
        if _safe_cholesky(cov) is None:
            var = block.var(dim=0, unbiased=False).clamp_min(self.eps)
            cov = _regularize(torch.diag(var), parent, self.shrinkage, self.eps)
        return cov

    # -- resolution ------------------------------------------------------
    def resolve(self, robot: int, program: int, regime: int) -> GroupStats:
        """Resolve the finest available reference; never program-only."""
        if self._fleet is None:
            raise ValueError("geometry has not been fitted")
        key = (int(robot), int(program), int(regime))
        group = self._groups.get(key)
        if group is not None and group.n >= self.min_group_samples:
            return group
        pair = self._pair.get((int(robot), int(program)))
        if pair is not None and pair.n >= self.min_group_samples:
            return pair
        robot_stats = self._robot.get(int(robot))
        if robot_stats is not None and robot_stats.n >= self.min_group_samples:
            return robot_stats
        assert self._fleet is not None
        return self._fleet

    # -- energy ----------------------------------------------------------
    @staticmethod
    def _mahalanobis_squared(rows: torch.Tensor, stats: GroupStats) -> torch.Tensor:
        diff = rows - stats.mu.to(rows.device, rows.dtype)
        cov = stats.cov.to(rows.device, rows.dtype)
        try:
            solved = torch.linalg.solve(cov, diff.T).T
        except (RuntimeError, torch._C._LinAlgError):  # type: ignore[attr-defined]
            diag = torch.diagonal(cov).clamp_min(1e-6)
            solved = diff / diag.unsqueeze(0)
        out = (diff * solved).sum(dim=-1)
        return torch.where(torch.isfinite(out), out, torch.full_like(out, 1e6))

    def patch_energy(
        self,
        latents: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        robot_idx: torch.Tensor,
        program_idx: torch.Tensor,
        regime_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return per-patch Mahalanobis energy with fallback provenance."""
        if latents.ndim != 3 or latents.shape[2] != self.d_model:
            raise ValueError(f"latents must be [B, N, {self.d_model}]")
        b, n, _ = latents.shape
        if tuple(patch_valid_mask.shape) != (b, n):
            raise ValueError("patch_valid_mask must match [B, N]")
        confidence = torch.ones((b, n), dtype=torch.float32)
        levels: list[list[str]] = [[""] * n for _ in range(b)]
        # Query latents stay attached: reference stats carry no grad, so the
        # Mahalanobis solve differentiates w.r.t. queries only.
        flat = latents.to(dtype=torch.float32).reshape(b * n, self.d_model)
        zero = torch.zeros((), device=flat.device, dtype=torch.float32)
        values: list[torch.Tensor] = []
        for i in range(b):
            for j in range(n):
                if not bool(patch_valid_mask[i, j]):
                    levels[i][j] = "invalid"
                    values.append(zero)
                    continue
                stats = self.resolve(int(robot_idx[i]), int(program_idx[i]), int(regime_ids[i, j]))
                value = self._mahalanobis_squared(flat[i * n + j].unsqueeze(0), stats)
                values.append(value.reshape(()).to(dtype=torch.float32))
                confidence[i, j] = 0.25 if stats.low_confidence else 1.0
                levels[i][j] = stats.level
        energy = torch.stack(values).reshape(b, n).masked_fill(~patch_valid_mask, 0.0)
        return {"patch_energy": energy, "group_confidence": confidence, "fallback_level": levels}
    def mixture_energy(
        self,
        latents: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        robot_idx: torch.Tensor,
        program_idx: torch.Tensor,
        regime_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Mixture-density energy over regime components of each (robot, program)."""
        if latents.ndim != 3 or latents.shape[2] != self.d_model:
            raise ValueError(f"latents must be [B, N, {self.d_model}]")
        b, n, d = latents.shape
        const = d * math.log(2.0 * math.pi)
        flat = latents.to(dtype=torch.float32).reshape(b * n, d)
        zero = torch.zeros((), device=flat.device, dtype=torch.float32)
        big = torch.tensor(1e6, device=flat.device, dtype=torch.float32)
        values: list[torch.Tensor] = []
        for i in range(b):
            robot, program = int(robot_idx[i]), int(program_idx[i])
            components = [
                (key, stats)
                for key, stats in self._groups.items()
                if key[0] == robot and key[1] == program and stats.n >= 1
            ]
            if not components:
                stats = self.resolve(robot, program, int(regime_ids[i, 0]))
                components = [((robot, program, 0), stats)]
            total = sum(s.n for _, s in components)
            for j in range(n):
                if not bool(patch_valid_mask[i, j]):
                    values.append(zero)
                    continue
                row = flat[i * n + j].unsqueeze(0)
                log_terms = []
                for _, stats in components:
                    d2 = self._mahalanobis_squared(row, stats)
                    sign, logdet = torch.linalg.slogdet(stats.cov)
                    if sign.item() <= 0 or not math.isfinite(logdet.item()):
                        logdet = torch.tensor(0.0)
                    log_terms.append(
                        math.log(stats.n / total) - 0.5 * (d2 + logdet + const)
                    )
                stacked = torch.stack(log_terms)
                value = -(torch.logsumexp(stacked, dim=0))
                values.append(value.reshape(()) if torch.isfinite(value).all() else big)
        out = torch.stack(values).reshape(b, n).masked_fill(~patch_valid_mask, 0.0)
        return {"patch_energy": out}

    # -- freezing --------------------------------------------------------
    def frozen(self) -> "FrozenReference":
        """Return an immutable detached snapshot for monitoring."""
        if self._fleet is None:
            raise ValueError("geometry has not been fitted")
        self._frozen = True
        snapshot = copy.deepcopy(self)
        return FrozenReference(snapshot)
    # -- snapshots -----------------------------------------------------
    def snapshot(self) -> dict[str, object]:
        """Return a detached serializable copy of all fitted references."""
        if self._fleet is None:
            raise ValueError("geometry has not been fitted")
        def freeze(stats: GroupStats) -> dict[str, object]:
            return {
                "mu": stats.mu.detach().to(dtype=torch.float32).clone(),
                "cov": stats.cov.detach().to(dtype=torch.float32).clone(),
                "n": stats.n,
                "level": stats.level,
                "low_confidence": stats.low_confidence,
            }
        return {
            "d_model": self.d_model,
            "shrinkage": self.shrinkage,
            "covariance_eps": self.eps,
            "min_group_samples": self.min_group_samples,
            "diag_min_samples": self.diag_min_samples,
            "groups": {f"{r},{p},{g}": freeze(s) for (r, p, g), s in self._groups.items()},
            "pair": {f"{r},{p}": freeze(s) for (r, p), s in self._pair.items()},
            "robot": {str(r): freeze(s) for r, s in self._robot.items()},
            "fleet": freeze(self._fleet),
        }

    def restore_snapshot(self, snap: dict[str, object]) -> "HierarchicalMahalanobisGeometry":
        """Restore references saved by :meth:`snapshot`; fail fast on mismatch."""
        if snap.get("d_model") != self.d_model:
            raise ValueError("geometry snapshot d_model mismatch")
        if self._frozen:
            raise ValueError("frozen references cannot be refit")
        def thaw(payload: object) -> GroupStats:
            if not isinstance(payload, dict):
                raise ValueError("geometry snapshot holds malformed group stats")
            mu = payload.get("mu")
            cov = payload.get("cov")
            if not isinstance(mu, torch.Tensor) or not isinstance(cov, torch.Tensor):
                raise ValueError("geometry snapshot holds malformed tensors")
            return GroupStats(
                mu.detach().to(dtype=torch.float32).clone(),
                cov.detach().to(dtype=torch.float32).clone(),
                int(payload.get("n", 0)),
                str(payload.get("level", "")),
                bool(payload.get("low_confidence", False)),
            )
        groups = snap.get("groups")
        pair = snap.get("pair")
        robot = snap.get("robot")
        fleet = snap.get("fleet")
        if not isinstance(groups, dict) or not isinstance(pair, dict):
            raise ValueError("geometry snapshot is missing reference groups")
        if not isinstance(robot, dict) or fleet is None:
            raise ValueError("geometry snapshot is missing robot/fleet references")
        self._groups = {
            (int(a), int(b), int(c)): thaw(s) for key, s in groups.items()
            for a, b, c in [key.split(",")]
        }
        self._pair = {
            (int(a), int(b)): thaw(s) for key, s in pair.items()
            for a, b in [key.split(",")]
        }
        self._robot = {int(k): thaw(s) for k, s in robot.items()}
        self._fleet = thaw(fleet)
        if snap.get("shrinkage") is not None:
            self.shrinkage = float(snap["shrinkage"])  # type: ignore[arg-type]
        if snap.get("covariance_eps") is not None:
            self.eps = float(snap["covariance_eps"])  # type: ignore[arg-type]
        if snap.get("min_group_samples") is not None:
            self.min_group_samples = int(snap["min_group_samples"])  # type: ignore[arg-type]
        if snap.get("diag_min_samples") is not None:
            self.diag_min_samples = int(snap["diag_min_samples"])  # type: ignore[arg-type]
        return self



class FrozenReference:
    """Immutable monitoring view: energies only, no refit from suspect files."""

    def __init__(self, geometry: HierarchicalMahalanobisGeometry) -> None:
        self._geometry = geometry
        for stats in (
            list(geometry._groups.values())
            + list(geometry._pair.values())
            + list(geometry._robot.values())
            + ([geometry._fleet] if geometry._fleet is not None else [])
        ):
            stats.mu.requires_grad_(False)
            stats.cov.requires_grad_(False)

    def patch_energy(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        """Score patches against the frozen references."""
        return self._geometry.patch_energy(*args, **kwargs)  # type: ignore[arg-type]

    def mixture_energy(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        """Score patches with the frozen mixture-density energy."""
        return self._geometry.mixture_energy(*args, **kwargs)  # type: ignore[arg-type]

    def resolve(self, robot: int, program: int, regime: int) -> GroupStats:
        """Resolve through the frozen hierarchy."""
        return self._geometry.resolve(robot, program, regime)
