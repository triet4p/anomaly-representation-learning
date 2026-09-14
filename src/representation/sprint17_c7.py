"""Sprint 17 Task 14 — C7 geometry/reference-bank alternatives (C7-A / C7-B).

Frozen-registry implementation for the C7 (healthy geometry / reference
bank) suspect only. Both arms replace the B0 global kNN bank for ``S_pop``
scoring with a condition-aware reference fitted exclusively from Fit
healthy-reference rows. Representations, targets, ``S_pred``, and all
metric code are untouched:

- ``C7-A`` robust-shrinkage: per-condition robust location (coordinate-wise
  median) plus Ledoit-Wolf-style covariance shrinkage toward scaled
  identity, scored as squared Mahalanobis energy. The frozen
  condition hierarchy ``(robot, program) → robot → program → global``
  with fallback at cells below ``MIN_CELL_ROWS`` is preserved, never
  re-tuned.
- ``C7-B`` local-density: condition-aware kNN (k=5, the B0 neighborhood
  size) mean distance normalized by a robust within-cell scale (median
  leave-one-out 1NN distance plus floor), fitted only from Fit
  healthy-reference rows under the same frozen hierarchy/fallback.

Conditions are model-input indices (``robot_idx``/``program_idx`` from the
file samples, the B0 conditional-norm vocabulary); labels, anomaly masks,
severity, categories, and simulator hidden state never enter fitting or
scoring. All statistics use float64 CPU arithmetic; scores are finite for
finite inputs, larger means more anomalous.
"""

from __future__ import annotations

import torch

#: Registered C7 arm IDs (protocol sprint17-ablation-v4 §2).
C7_ARMS = ("C7-A", "C7-B")

#: Frozen B0 conditional vocabulary (model sizing; Task 7 binding).
N_ROBOTS = 9
N_PROGRAMS = 8

#: Frozen hierarchy levels, most specific first.
LEVELS = ("cell", "robot", "program", "global")

#: Frozen minimum reference rows to serve a level; smaller cells fall back.
MIN_CELL_ROWS = 8

#: Frozen kNN neighborhood (the B0 bank neighborhood size).
KNN_K = 5

#: Frozen shrinkage floor keeping every shrunk covariance positive-definite.
SHRINKAGE_FLOOR = 0.05

#: Frozen floors for degenerate scales.
COV_SCALE_FLOOR = 1e-12
DENSITY_SCALE_FLOOR = 1e-6

ARM_ADAPTER_DESCRIPTION: dict[str, str] = {
    "C7-A": ("per-condition robust location (coordinate-wise median) plus "
             "Ledoit-Wolf-style covariance shrinkage to scaled identity "
             "(floor 0.05), squared-Mahalanobis energy over the frozen "
             "(robot,program)->robot->program->global hierarchy with "
             "fallback below 8 rows; fitted statistics only, no trainable "
             "parameters; no other adapter"),
    "C7-B": ("condition-aware kNN (k=5) mean distance normalized by a robust "
             "within-cell scale (median leave-one-out 1NN distance, floor "
             "1e-6) over the same frozen hierarchy/fallback; reference rows "
             "and scales from Fit healthy rows only; no trainable "
             "parameters; no other adapter"),
}


def validate_condition(robot_idx: int, program_idx: int) -> tuple[int, int]:
    """Fail fast unless a condition is inside the frozen B0 vocabulary."""
    robot = int(robot_idx)
    program = int(program_idx)
    if not 0 <= robot < N_ROBOTS:
        raise ValueError(f"robot_idx {robot} outside frozen 0..{N_ROBOTS - 1}")
    if not 0 <= program < N_PROGRAMS:
        raise ValueError(f"program_idx {program} outside frozen 0..{N_PROGRAMS - 1}")
    return (robot, program)


def resolve_level(n_cell: int, n_robot: int, n_program: int, n_global: int,
                  min_rows: int = MIN_CELL_ROWS) -> str:
    """Return the most specific hierarchy level meeting the frozen minimum."""
    if n_cell >= min_rows:
        return "cell"
    if n_robot >= min_rows:
        return "robot"
    if n_program >= min_rows:
        return "program"
    if n_global >= min_rows:
        return "global"
    raise ValueError(
        f"no hierarchy level meets min_rows={min_rows} "
        f"(cell/robot/program/global = {n_cell}/{n_robot}/{n_program}/{n_global})")


def _as_conditions(robots, programs) -> list[tuple[int, int]]:
    if len(robots) != len(programs):
        raise ValueError("robot and program condition lists must share length")
    return [validate_condition(r, p) for r, p in zip(robots, programs)]


def _check_labels(labels) -> None:
    if labels is None:
        return
    for label in labels:
        value = getattr(label, "value", label)
        if str(value).lower() == "abnormal":
            raise ValueError("C7 references cannot include abnormal rows")


def _as_f64(embeddings: torch.Tensor) -> torch.Tensor:
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("embeddings must be a torch.Tensor")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty [M, D] tensor")
    if not torch.isfinite(embeddings).all():
        raise ValueError("reference embeddings must be finite")
    return embeddings.detach().to(device="cpu", dtype=torch.float64)


def ledoit_wolf_shrinkage(x: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    """Ledoit-Wolf-style shrinkage of the sample covariance to scaled identity.

    Returns ``(delta, mu, sigma)`` with ``sigma = (1-delta)*S + delta*mu*I``
    positive-definite by construction (``delta >= SHRINKAGE_FLOOR``,
    ``mu >= COV_SCALE_FLOOR``). Location is handled by the caller (robust
    median); ``S`` here uses the ML (ddof=0) second moment about the given
    center ``x`` (already centered input).
    """
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("shrinkage needs at least 2 centered rows")
    n, dim = x.shape
    s = (x.T @ x) / n
    mu = float(torch.trace(s) / dim)
    mu = max(mu, COV_SCALE_FLOOR)
    identity = torch.eye(dim, dtype=torch.float64)
    diff = s - mu * identity
    d2 = float((diff * diff).sum())
    if d2 <= 0.0:
        delta = 0.0
    else:
        # ||x_i x_i^T - S||_F^2 = (x_i^T x_i)^2 - 2 x_i^T S x_i + ||S||_F^2.
        norms_sq = (x * x).sum(dim=1)
        s_norm_sq = float((s * s).sum())
        cross = (x @ s * x).sum(dim=1)
        pi_sum = float((norms_sq * norms_sq - 2.0 * cross + s_norm_sq).sum())
        b2 = max(pi_sum / (n * n), 0.0)
        delta = b2 / d2
    delta = min(max(delta, SHRINKAGE_FLOOR), 1.0)
    sigma = (1.0 - delta) * s + delta * mu * identity
    return delta, mu, sigma


class _ConditionedReference:
    """Shared Fit-only conditioning, hierarchy, and fallback bookkeeping."""

    #: Rows served per level for the last score() call (provenance).
    level_counts: dict[str, int]

    def _level_rows(self, cond: tuple[int, int],
                    cell_members: dict[tuple[int, int], list[int]]) -> tuple[str, list[int]]:
        robot, program = cond
        n_cell = len(cell_members.get(cond, ()))
        robot_rows = [i for (r, _), members in cell_members.items()
                      if r == robot for i in members]
        program_rows = [i for (_, p), members in cell_members.items()
                        if p == program for i in members]
        level = resolve_level(n_cell, len(robot_rows), len(program_rows),
                              self._n_total)
        if level == "cell":
            return level, list(cell_members[cond])
        if level == "robot":
            return level, robot_rows
        if level == "program":
            return level, program_rows
        return level, list(range(self._n_total))

    def _check_fitted(self) -> None:
        if getattr(self, "_embeddings", None) is None:
            raise ValueError("C7 reference has not been fitted")


class RobustShrinkageReference(_ConditionedReference):
    """C7-A: robust location plus shrunk-covariance Mahalanobis energy."""

    def __init__(self) -> None:
        self._embeddings: torch.Tensor | None = None
        self._cells: dict[tuple[int, int], list[int]] = {}
        self._stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._stats_key: dict[str, str] = {}
        self.level_counts = {level: 0 for level in LEVELS}

    def fit(self, embeddings: torch.Tensor, robots, programs,
            labels=None) -> "RobustShrinkageReference":
        """Fit per-level robust location and shrunk covariance (Fit-only)."""
        _check_labels(labels)
        x = _as_f64(embeddings)
        conds = _as_conditions(list(robots), list(programs))
        if len(conds) != x.shape[0]:
            raise ValueError("conditions length must match embeddings")
        self._embeddings = x
        self._n_total = x.shape[0]
        cells: dict[tuple[int, int], list[int]] = {}
        for i, cond in enumerate(conds):
            cells.setdefault(cond, []).append(i)
        self._cells = cells
        self._stats = {}
        self._stats_key = {}
        # Fit every hierarchy node that can serve: each observed cell, each
        # observed robot/program marginal, and global.
        nodes: dict[str, list[int]] = {"global": list(range(self._n_total))}
        for (robot, program), members in cells.items():
            nodes[f"cell:{robot}:{program}"] = list(members)
        for robot in range(N_ROBOTS):
            members = [i for (r, _), ms in cells.items()
                       if r == robot for i in ms]
            if members:
                nodes[f"robot:{robot}"] = members
        for program in range(N_PROGRAMS):
            members = [i for (_, p), ms in cells.items()
                       if p == program for i in ms]
            if members:
                nodes[f"program:{program}"] = members
        for key, members in nodes.items():
            block = x[members]
            location = block.median(dim=0).values
            _, _, sigma = ledoit_wolf_shrinkage(block - location.unsqueeze(0))
            try:
                chol = torch.linalg.cholesky(sigma)
            except RuntimeError as exc:
                raise ValueError(f"shrunk covariance not PD at {key}") from exc
            self._stats[key] = (location, chol)
        return self

    def _node_for(self, cond: tuple[int, int]) -> tuple[str, str]:
        level, _ = self._level_rows(cond, self._cells)
        robot, program = cond
        if level == "cell":
            return level, f"cell:{robot}:{program}"
        if level == "robot":
            return level, f"robot:{robot}"
        if level == "program":
            return level, f"program:{program}"
        return level, "global"

    def score(self, queries: torch.Tensor, qrobots, qprograms) -> tuple[torch.Tensor, list[str]]:
        """Squared-Mahalanobis energies with per-query fallback levels."""
        self._check_fitted()
        q = _as_f64(queries)
        if q.shape[1] != self._embeddings.shape[1]:
            raise ValueError("query dimension must match reference dimension")
        qconds = _as_conditions(list(qrobots), list(qprograms))
        if len(qconds) != q.shape[0]:
            raise ValueError("query conditions length must match queries")
        energies = torch.empty(q.shape[0], dtype=torch.float64)
        levels: list[str] = []
        self.level_counts = {level: 0 for level in LEVELS}
        for i, cond in enumerate(qconds):
            level, key = self._node_for(cond)
            location, chol = self._stats[key]
            dev = q[i] - location
            solved = torch.cholesky_solve(dev.unsqueeze(1), chol).squeeze(1)
            energies[i] = float(dev @ solved)
            levels.append(level)
            self.level_counts[level] += 1
        if not torch.isfinite(energies).all():
            raise ValueError("C7-A energies must be finite")
        return energies, levels

    def provenance(self) -> dict[str, object]:
        """Row/source accounting for the fitted reference."""
        self._check_fitted()
        return {
            "source": "Fit-only healthy-eligible rows",
            "n_total": self._n_total,
            "n_cells": len(self._cells),
            "level_counts": dict(self.level_counts),
        }


class LocalDensityReference(_ConditionedReference):
    """C7-B: condition-aware kNN distance over a robust Fit-only scale."""

    def __init__(self, k: int = KNN_K) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)
        self._embeddings: torch.Tensor | None = None
        self._cells: dict[tuple[int, int], list[int]] = {}
        self._scales: dict[str, float] = {}
        self.level_counts = {level: 0 for level in LEVELS}

    def fit(self, embeddings: torch.Tensor, robots, programs,
            labels=None) -> "LocalDensityReference":
        """Fit per-level reference sets and robust scales (Fit-only)."""
        _check_labels(labels)
        x = _as_f64(embeddings)
        conds = _as_conditions(list(robots), list(programs))
        if len(conds) != x.shape[0]:
            raise ValueError("conditions length must match embeddings")
        self._embeddings = x
        self._n_total = x.shape[0]
        cells: dict[tuple[int, int], list[int]] = {}
        for i, cond in enumerate(conds):
            cells.setdefault(cond, []).append(i)
        self._cells = cells
        self._scales = {}
        nodes: dict[str, list[int]] = {"global": list(range(self._n_total))}
        for (robot, program), members in cells.items():
            nodes[f"cell:{robot}:{program}"] = list(members)
        for robot in range(N_ROBOTS):
            members = [i for (r, _), ms in cells.items()
                       if r == robot for i in ms]
            if members:
                nodes[f"robot:{robot}"] = members
        for program in range(N_PROGRAMS):
            members = [i for (_, p), ms in cells.items()
                       if p == program for i in ms]
            if members:
                nodes[f"program:{program}"] = members
        for key, members in nodes.items():
            block = x[members]
            dists = torch.cdist(block, block)
            dists.fill_diagonal_(float("inf"))
            nearest = dists.min(dim=1).values
            if not torch.isfinite(nearest).all():
                raise ValueError(f"degenerate reference set at {key}")
            self._scales[key] = float(nearest.median()) + DENSITY_SCALE_FLOOR
        return self

    def score(self, queries: torch.Tensor, qrobots, qprograms) -> tuple[torch.Tensor, list[str]]:
        """Mean-kNN distance over the robust scale, with fallback levels."""
        self._check_fitted()
        q = _as_f64(queries)
        if q.shape[1] != self._embeddings.shape[1]:
            raise ValueError("query dimension must match reference dimension")
        qconds = _as_conditions(list(qrobots), list(qprograms))
        if len(qconds) != q.shape[0]:
            raise ValueError("query conditions length must match queries")
        assert self._embeddings is not None
        scores = torch.empty(q.shape[0], dtype=torch.float64)
        levels: list[str] = []
        self.level_counts = {level: 0 for level in LEVELS}
        for i, cond in enumerate(qconds):
            level, rows = self._level_rows(cond, self._cells)
            refs = self._embeddings[rows]
            dists = torch.cdist(q[i].unsqueeze(0), refs).squeeze(0)
            nearest = dists.topk(min(self.k, refs.shape[0]),
                                 largest=False).values
            robot, program = cond
            key = {"cell": f"cell:{robot}:{program}", "robot": f"robot:{robot}",
                   "program": f"program:{program}", "global": "global"}[level]
            scores[i] = float(nearest.mean() / self._scales[key])
            levels.append(level)
            self.level_counts[level] += 1
        if not torch.isfinite(scores).all():
            raise ValueError("C7-B scores must be finite")
        return scores, levels

    def provenance(self) -> dict[str, object]:
        """Row/source accounting for the fitted reference."""
        self._check_fitted()
        return {
            "source": "Fit-only healthy-eligible rows",
            "n_total": self._n_total,
            "n_cells": len(self._cells),
            "level_counts": dict(self.level_counts),
        }


def arm_reference(arm_id: str, k: int = KNN_K):
    """Build the frozen reference for a registered C7 arm."""
    if arm_id == "C7-A":
        return RobustShrinkageReference()
    if arm_id == "C7-B":
        return LocalDensityReference(k=k)
    raise ValueError(f"unknown Sprint 17 C7 arm: {arm_id!r}")
