"""Shared stagewise diagnostic metric contract for Sprint 16 attribution.

Implements ``experiments/sprint16-attribution-protocol-v3.md`` §§4–4.5 and the
§10 decision table as executable code. Every threshold below is a frozen
protocol constant, not a tunable: Task 3 owns this implementation, and later
tasks consume it without modification.

Fail-closed design: empty/single-class/degenerate inputs raise ``ValueError``
or return protocol-specified ``nan`` (never a fabricated rank); the frozen
standardizer cannot be refit; score families are always reported separately
(no fusion path exists); support mismatches raise instead of silently
intersecting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from synth.probe15 import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, STD_FLOOR

__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "STD_FLOOR",
    "DIRECTIONAL_FLOOR",
    "GAP_FLOOR",
    "MOVEMENT_FLOOR",
    "RESTORATION_FRACTION",
    "ORACLE_FLOOR",
    "ORACLE_LCB_FLOOR",
    "FAR_CAP",
    "FAR_DELTA_CAP",
    "FROZEN_THRESHOLD",
    "STABILITY_RATIO_CAP",
    "RECALL_P",
    "LEAD_P_D",
    "RECALL_W",
    "LEAD_W_D",
    "REPLICATION_HISTORIES",
    "TOPK_TAIL",
    "TOPK_EVENT",
    "PCTL_Q",
    "MIN_WINDOW_PATCHES",
    "FrozenStandardizer",
    "DiagnosticSet",
    "assert_same_provenance",
    "intersect_ids",
    "tie_auc",
    "bootstrap_ci",
    "severity_spearman",
    "stability_ratio",
    "stability_ci",
    "exceedance_fraction",
    "recall_at_threshold",
    "median_lead",
    "aggregate_patches",
    "movement",
    "restoration_met",
    "component_verdict",
    "overall_state",
]

#: Frozen protocol floors (v3 §§4, 4.5, 10). Not tunable.
DIRECTIONAL_FLOOR = 0.55
GAP_FLOOR = 0.10
MOVEMENT_FLOOR = 0.05
RESTORATION_FRACTION = 0.5
ORACLE_FLOOR = 0.65
ORACLE_LCB_FLOOR = 0.55
FAR_CAP = 0.05
FAR_DELTA_CAP = 0.02
FROZEN_THRESHOLD = 138.03
STABILITY_RATIO_CAP = 0.10
RECALL_P = 0.50
LEAD_P_D = 1.0
RECALL_W = 0.25
LEAD_W_D = 0.5
REPLICATION_HISTORIES = 3
TOPK_TAIL = 8
TOPK_EVENT = 4
PCTL_Q = 0.9
MIN_WINDOW_PATCHES = 8


def _as_float(a: object, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite (no NaN/inf)")
    return arr


class FrozenStandardizer:
    """Global Fit-healthy standardization with frozen provenance.

    Mirrors ``synth.probe15.standardize_fit`` semantics (single global
    float64 mean/std over verified-healthy rows, std floored at
    ``STD_FLOOR``) while carrying the protocol §4.4 provenance record.
    Instances are immutable after ``fit``; ``refit`` always raises.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, source: str, n_fit: int):
        self._mean = mean
        self._std = std
        self._source = source
        self._n_fit = n_fit

    @classmethod
    def fit(cls, features_healthy: object, *, source: str) -> "FrozenStandardizer":
        """Fit once on verified-healthy rows only. Empty input raises."""
        arr = np.asarray(features_healthy, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("fit requires a 2-D feature array")
        if arr.shape[0] == 0:
            raise ValueError("fit requires at least one healthy row")
        if not np.isfinite(arr).all():
            raise ValueError("fit rows must be finite")
        if not source:
            raise ValueError("fit requires a non-empty source label")
        mean = arr.mean(axis=0)
        std = np.maximum(arr.std(axis=0), STD_FLOOR)
        return cls(mean, std, source, int(arr.shape[0]))

    def refit(self, *_: object, **__: object) -> "FrozenStandardizer":
        """Always raises: frozen transforms are never refit on eval rows."""
        raise ValueError("FrozenStandardizer is frozen; refit is forbidden")

    def apply(self, features: object) -> np.ndarray:
        """Apply the FROZEN transform (never fit on eval rows)."""
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("apply requires a 2-D feature array")
        if arr.shape[-1] != self._mean.shape[-1]:
            raise ValueError(
                f"feature dim {arr.shape[-1]} != fitted dim {self._mean.shape[-1]}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("apply rows must be finite")
        return (arr - self._mean) / self._std

    def provenance(self) -> dict[str, object]:
        """Return the §4.4 provenance record (source, n, center/scale)."""
        return {
            "source": self._source,
            "n_fit": self._n_fit,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "floor": STD_FLOOR,
        }

    def token(self) -> str:
        """Deterministic provenance token for cross-stage comparison guards.

        Digest formation (frozen): sha256 over
        ``repr(source) || n_fit as 8-byte big-endian || mean.tobytes() ||
        std.tobytes() || repr(floor)``, all float64. Any transform difference
        (source, n, center, scale, floor) changes the token.
        """
        import hashlib

        h = hashlib.sha256()
        h.update(repr(self._source).encode())
        h.update(int(self._n_fit).to_bytes(8, "big"))
        h.update(np.ascontiguousarray(self._mean, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self._std, dtype=np.float64).tobytes())
        h.update(repr(float(STD_FLOOR)).encode())
        return h.hexdigest()


def intersect_ids(*id_arrays: object) -> np.ndarray:
    """Canonical sorted intersection of id sets. Empty intersection raises."""
    sets = [set(np.asarray(a).tolist()) for a in id_arrays]
    if any(len(s) == 0 for s in sets):
        raise ValueError("support sets must be non-empty")
    common = sorted(set.intersection(*sets))
    if not common:
        raise ValueError("identical-support intersection is empty")
    return np.asarray(common)


def tie_auc(scores: object, labels: object) -> float:
    """Tie-aware AUROC (Mann-Whitney, average ranks). Single-class → nan.

    Matches the ``sklearn.metrics.roc_auc_score`` convention exactly,
    including ties; never fabricates a rank for degenerate input. Clipped to
    [0, 1] like ``synth.events.roc_auc_tie_aware`` (float-roundoff guard).
    """
    s = _as_float(scores, "scores")
    y = np.asarray(labels, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("scores and labels must share shape")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("labels must be binary 0/1")
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(s, method="average")
    value = float((ranks[y == 1.0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return min(max(value, 0.0), 1.0)


def bootstrap_ci(
    values: object,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    """Deterministic index-resampling CI of the mean. Same input → same output."""
    v = _as_float(values, "values")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, v.size, size=(replicates, v.size))
    means = v[draws].mean(axis=1)
    return {
        "point": float(v.mean()),
        "lcb": float(np.quantile(means, 0.025)),
        "ucb": float(np.quantile(means, 0.975)),
        "replicates": float(replicates),
        "seed": float(seed),
    }


def severity_spearman(scores: object, levels: object) -> float:
    """Within-cohort Spearman ρ (average-rank ties). Constant input → nan."""
    s = _as_float(scores, "scores")
    lv = _as_float(levels, "levels")
    if lv.shape != s.shape:
        raise ValueError("scores and levels must share shape")
    if np.ptp(s) == 0.0 or np.ptp(lv) == 0.0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = stats.spearmanr(s, lv)
    return float(rho)


def stability_ratio(
    background_scores: object, fit_healthy_scores: object
) -> dict[str, float]:
    """Unaffected-background stability per v3 §4.5 (exact frozen formula).

    ratio = |median(A) − median(B)| / pooled_scale with
    pooled_scale = max(sqrt((var(A) + var(B)) / 2), STD_FLOOR),
    population variances (ddof=0, float64). A = this-history background
    scores, B = frozen Fit-healthy scores.
    """
    a = _as_float(background_scores, "background_scores")
    b = _as_float(fit_healthy_scores, "fit_healthy_scores")
    pooled_scale = max(
        float(np.sqrt((a.var() + b.var()) / 2.0)), float(STD_FLOOR)
    )
    ratio = float(abs(np.median(a) - np.median(b)) / pooled_scale)
    return {"ratio": ratio, "pooled_scale": pooled_scale, "cap": STABILITY_RATIO_CAP}


def stability_ci(
    background_scores: object,
    fit_healthy_scores: object,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    """Stability ratio with bootstrap uncertainty (v3 §4.5, frozen).

    Resamples indices of the background set A ONLY with
    ``np.random.default_rng(seed).integers``; the Fit-healthy reference B is
    held fixed across draws. Returns the point ratio, the 2.5th/97.5th
    percentiles, the point pooled_scale, and the cap. Same inputs (incl.
    seed) give identical outputs; empty/non-finite inputs raise.
    """
    a = _as_float(background_scores, "background_scores")
    b = _as_float(fit_healthy_scores, "fit_healthy_scores")
    point = stability_ratio(a, b)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, a.size, size=(replicates, a.size))
    ratios = np.asarray([stability_ratio(a[row], b)["ratio"] for row in draws])
    return {
        "ratio": point["ratio"],
        "lcb": float(np.quantile(ratios, 0.025)),
        "ucb": float(np.quantile(ratios, 0.975)),
        "pooled_scale": point["pooled_scale"],
        "cap": STABILITY_RATIO_CAP,
        "replicates": float(replicates),
        "seed": float(seed),
    }


def exceedance_fraction(scores: object, threshold: float) -> float:
    """Fraction of scores strictly above threshold (FAR-style numerator)."""
    s = _as_float(scores, "scores")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return float((s > threshold).mean())


def recall_at_threshold(scores: object, labels: object, threshold: float) -> float:
    """Positive recall at a frozen threshold. No positives → nan."""
    s = _as_float(scores, "scores")
    y = np.asarray(labels, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("scores and labels must share shape")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("labels must be binary 0/1")
    pos = s[y == 1.0]
    if pos.size == 0:
        return float("nan")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return float((pos > threshold).mean())


def median_lead(leads_days: object, recalled: object) -> float:
    """Median lead (days) over recalled events. None recalled → nan."""
    leads = _as_float(leads_days, "leads_days")
    rec = np.asarray(recalled, dtype=bool)
    if rec.shape != leads.shape:
        raise ValueError("leads and recalled mask must share shape")
    if not rec.any():
        return float("nan")
    return float(np.median(leads[rec]))


def aggregate_patches(
    patch_scores: object,
    valid: object,
    method: str,
    *,
    k: int | None = None,
    q: float | None = None,
    min_valid: int = MIN_WINDOW_PATCHES,
) -> float:
    """Patch-to-event reduction with frozen semantics (v3 §8 C9).

    ``max``: per-event max (reporting). ``topk_mean`` requires ``k``.
    ``median``/``percentile`` use linear interpolation over valid patches.
    Fewer than ``min_valid`` valid patches raises (exclusion, counted).
    """
    s = np.asarray(patch_scores, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    if v.shape != s.shape:
        raise ValueError("patch_scores and valid mask must share shape")
    if not np.isfinite(s[v]).all() or v.sum() == 0:
        raise ValueError("no finite valid patches")
    if int(v.sum()) < min_valid:
        raise ValueError(
            f"only {int(v.sum())} valid patches < min_valid {min_valid}"
        )
    x = s[v]
    if method == "max":
        return float(x.max())
    if method == "mean":
        return float(x.mean())
    if method == "median":
        return float(np.percentile(x, 50.0, method="linear"))
    if method == "topk_mean":
        if k is None:
            raise ValueError("topk_mean requires explicit k (no hidden default)")
        if k <= 0 or k > x.size:
            raise ValueError(f"k={k} out of range for {x.size} patches")
        return float(np.partition(x, -k)[-k:].mean())
    if method == "percentile":
        if q is None:
            raise ValueError("percentile requires explicit q (no hidden default)")
        if not 0.0 < q < 1.0:
            raise ValueError(f"q={q} must lie in (0, 1)")
        return float(np.percentile(x, 100.0 * q, method="linear"))
    raise ValueError(f"unknown aggregation method {method!r}")


@dataclass
class DiagnosticSet:
    """One history's evaluable rows with separated score families.

    Families (e.g. ``S_pred``, ``S_pop``, ``oracle``) are stored and reported
    independently; this class offers no fusion operation. ``restrict`` is the
    only narrowing path, enforcing the §4.1 identical-support rule.
    ``provenance`` optionally carries a ``FrozenStandardizer.token()``; stage
    comparisons MUST pass their sets through ``assert_same_provenance``.
    """

    history_id: str
    ids: np.ndarray = field(repr=False)
    families: dict[str, np.ndarray] = field(repr=False)
    labels: np.ndarray = field(repr=False)
    cohorts: np.ndarray | None = field(default=None, repr=False)
    severities: np.ndarray | None = field(default=None, repr=False)
    is_background: np.ndarray | None = field(default=None, repr=False)
    provenance: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        ids = np.asarray(self.ids)
        if ids.size == 0:
            raise ValueError("DiagnosticSet requires non-empty ids")
        if len(set(ids.tolist())) != ids.size:
            raise ValueError("ids must be unique")
        n = ids.size
        for name, arr in self.families.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.shape != (n,):
                raise ValueError(f"family {name!r} length != ids length")
            if not np.isfinite(a).all():
                raise ValueError(f"family {name!r} must be finite")
        y = np.asarray(self.labels, dtype=np.float64)
        if y.shape != (n,) or not set(np.unique(y)) <= {0.0, 1.0}:
            raise ValueError("labels must be binary 0/1 with ids length")
        for attr in ("cohorts", "severities", "is_background"):
            val = getattr(self, attr)
            if val is not None and np.asarray(val).shape != (n,):
                raise ValueError(f"{attr} length != ids length")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(
            self, "families", {k: np.asarray(v, dtype=np.float64)
                               for k, v in self.families.items()})
        object.__setattr__(self, "labels", y)

    def family(self, name: str) -> np.ndarray:
        """Return one family's scores. Unknown families raise (fail closed)."""
        if name not in self.families:
            raise KeyError(f"unknown score family {name!r}")
        return self.families[name]

    def restrict(self, ids: object) -> "DiagnosticSet":
        """Narrow to an explicit id subset. Unknown/empty selection raises.

        Deterministic: ``set``/``frozenset`` selections are sorted (unordered
        collections have no stable order); lists/tuples/ndarrays keep caller
        order.
        """
        if isinstance(ids, (set, frozenset)):
            want = np.asarray(sorted(ids))
        else:
            want = np.asarray(list(ids) if not isinstance(ids, np.ndarray) else ids)
        have = {v: i for i, v in enumerate(self.ids.tolist())}
        missing = [v for v in want.tolist() if v not in have]
        if missing:
            raise ValueError(f"restrict ids not in support: {missing[:5]}")
        if want.size == 0:
            raise ValueError("restrict selection must be non-empty")
        idx = np.asarray([have[v] for v in want.tolist()])

        def _take(a: np.ndarray | None) -> np.ndarray | None:
            return None if a is None else np.asarray(a)[idx]

        return DiagnosticSet(
            history_id=self.history_id,
            ids=self.ids[idx],
            families={k: v[idx] for k, v in self.families.items()},
            labels=self.labels[idx],
            cohorts=_take(self.cohorts),
            severities=_take(self.severities),
            is_background=_take(self.is_background),
            provenance=self.provenance,
        )


def assert_same_provenance(*sets: DiagnosticSet) -> str:
    """Fail closed on standardization-provenance mismatch (§4.3).

    Every set must carry a non-null ``provenance`` token (see
    ``FrozenStandardizer.token``) and all tokens must be equal. Returns the
    shared token. Any missing or differing token raises ``ValueError``.
    """
    if not sets:
        raise ValueError("assert_same_provenance requires at least one set")
    tokens = [s.provenance for s in sets]
    if any(t is None for t in tokens):
        raise ValueError("compared stages lack standardization provenance")
    if len(set(tokens)) != 1:
        raise ValueError("standardization provenance differs across stages")
    return str(tokens[0])


def movement(deltas: object) -> dict[str, object]:
    """v3 §10 movement: max |Δ| over (variant × history) per-history deltas.

    Returns magnitude, the signed direction at the argmax, ALL tied argmax
    signs in ``directions`` (first-seen order), and count. Thresholds apply
    to the magnitude; every tied sign is reported evidence.
    """
    d = _as_float(deltas, "deltas")
    mag = float(np.abs(d).max())
    tied = [float(np.sign(v)) for v in d.tolist() if abs(v) == mag]
    return {
        "magnitude": mag,
        "direction": tied[0],
        "directions": tied,
        "n": float(d.size),
    }


def restoration_met(r_int: float, r_post: float, gap: float, post_lcb: float) -> bool:
    """Evaluate the §10(ii) restoration arithmetic (caller-computed input).

    True iff ``R_int − R_post ≥ RESTORATION_FRACTION × G`` with
    ``post_lcb > ORACLE_LCB_FLOOR``. ``component_verdict`` takes the resulting
    boolean as ``restoration``; P3 capability caps and the nuisance /
    replication / encoder booleans remain caller obligations (Task 16 owns
    §10 — this module freezes the arithmetic, not the evidence gathering).
    """
    for name, v in (("r_int", r_int), ("r_post", r_post), ("gap", gap),
                    ("post_lcb", post_lcb)):
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite")
    return bool(r_int - r_post >= RESTORATION_FRACTION * gap
                and post_lcb > ORACLE_LCB_FLOOR)


def component_verdict(
    *,
    g1_fail: bool,
    upstream_lcb: float,
    gap: float,
    move_magnitude: float,
    restoration: bool,
    nuisance_ok: bool,
    replication_ok: bool,
    encoders_ok: bool,
) -> str:
    """Apply the v3 §10 G0–R4 table (row order; exactly one row fires).

    Caller obligations (Task 16 owns §10): ``restoration`` MUST be computed
    with ``restoration_met`` (the §10(ii) arithmetic); P3 capability caps
    (C3 ≤ SUSPECT, C5 replicated-retrain requirement) and the nuisance /
    replication / encoder evidence are caller-computed booleans. Movement
    magnitude comes from ``movement()["magnitude"]``.
    """
    for v in (upstream_lcb, gap, move_magnitude):
        if not np.isfinite(v):
            raise ValueError("verdict inputs must be finite")
    if g1_fail:
        return "UNRESOLVED"
    if not upstream_lcb > ORACLE_LCB_FLOOR:
        return "UNRESOLVED"
    if gap < GAP_FLOOR and move_magnitude < MOVEMENT_FLOOR:
        return "HEALTHY"
    if gap < GAP_FLOOR:
        return "SUSPECT"
    if restoration and nuisance_ok and replication_ok and encoders_ok:
        return "BOTTLENECK"
    return "SUSPECT"


def overall_state(
    component_verdicts: dict[str, str],
    *,
    independent_groups: dict[str, str] | None = None,
) -> str:
    """Map per-component verdicts to IDENTIFIED / MULTIPLE BOTTLENECKS / UNRESOLVED.

    §10 independence rule: components sharing one restoration MUST be passed
    under one group key (e.g. ``{"C8": "tail", "C9": "tail"}``); each group
    contributes at most one restoration credit, so a shared C8/C9 win counts
    ONCE (IDENTIFIED), while wins in distinct groups count separately. With
    ``independent_groups=None`` every component is its own group — i.e. the
    caller asserts all restorations independent. Unknown components or
    verdicts raise.
    """
    if not component_verdicts:
        raise ValueError("component_verdicts must be non-empty")
    allowed = {"HEALTHY", "SUSPECT", "BOTTLENECK", "UNRESOLVED"}
    unknown = set(component_verdicts.values()) - allowed
    if unknown:
        raise ValueError(f"unknown verdicts: {sorted(unknown)}")
    groups = independent_groups or {}
    stray = set(groups) - set(component_verdicts)
    if stray:
        raise ValueError(f"groups for unknown components: {sorted(stray)}")
    credits = 0
    seen: set[str] = set()
    for comp, verdict in component_verdicts.items():
        if verdict != "BOTTLENECK":
            continue
        key = groups.get(comp, comp)
        if key not in seen:
            seen.add(key)
            credits += 1
    if credits == 1:
        return "IDENTIFIED"
    if credits >= 2:
        return "MULTIPLE BOTTLENECKS"
    return "UNRESOLVED"
