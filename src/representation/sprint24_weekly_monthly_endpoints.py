"""Endpoint, contrast, uncertainty and decision computation for ``S24-BIZ-WM-v1``.

This module implements the frozen study's declared estimands on the persisted
accepted B0 file embeddings:

- ``E1`` DIST-MOVE: unbiased squared MMD (primary), biased plug-in, exposure-
  weighted variant and capped-``n`` variant, with a Gaussian kernel whose
  bandwidth is the median pairwise distance over the full causal reference pool;
- ``E2`` CLUSTER-MOVE: total-variation distance of cluster occupancy with
  K-means prototypes fitted on the reference pool (``K = 8``, ``K in {4,16}``);
- ``E3`` LOC-MOVE and ``E4`` DISP-MOVE: descriptive decomposition;
- ``N1``/``N2``: reference split-half null and within-period half-split band;
- ``C1``/``C2``: within-history burden association and mix-adjusted residual;
- the declared sensitivities (rolling mode, capped-``n``, alternate anchor,
  K, exposure weighting, biased estimator);
- history-cluster bootstrap uncertainty and the frozen
  PROMISING / NOT-PROMISING / INCONCLUSIVE decision.

Every convention is frozen in
``artifacts/sprint-24/evidence/s24-7-m4-a01/pre-outcome-estimator-conventions.json``
before any movement value; :func:`assert_conventions_frozen` re-verifies that
digest. No model, checkpoint, waveform or score is read here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans

from representation import sprint24_business_horizon as horizon
from representation import sprint24_weekly_monthly_execution as execution
from representation import sprint24_weekly_monthly_study as study
from representation.sprint24_context_audit import sha256_object

#: Frozen pre-outcome conventions artifact and its canonical digest.
CONVENTIONS_PATH = "docs/sprint24-weekly-monthly-estimator-conventions-a01.json"
CONVENTIONS_CANONICAL_SHA256 = (
    "014ed05952c48196776e3292e096bfd56590f379fd08bf4a24aeb23dfcf06460"
)

#: Cluster counts declared for E2 (8 primary, 4/16 sensitivity).
CLUSTER_COUNTS = (4, 8, 16)
PRIMARY_K = 8

#: Comparison states that carry computable endpoints.
COMPARABLE_STATES = ("COMPARABLE-MATCHED", "COMPARABLE-UNMATCHED-EXPOSURE")
MATCHED_STATE = "COMPARABLE-MATCHED"

#: Endpoint keys published per pair and seed.
ENDPOINT_KEYS = (
    "E1",
    "E1_biased",
    "E1_weighted",
    "E1_capped",
    "E2_K4",
    "E2_K8",
    "E2_K16",
    "E3",
    "E4",
    "N2_previous",
    "N2_current",
    "N2_max",
)


class ConventionsDrift(RuntimeError):
    """Raised when the pre-outcome conventions artifact no longer matches its digest."""


def assert_conventions_frozen(path: Path | None = None, repo: Path | None = None) -> dict[str, object]:
    """Load the pre-outcome conventions and fail closed on digest drift."""
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    target = Path(path) if path is not None else root / CONVENTIONS_PATH
    if not target.is_file():
        raise ConventionsDrift(f"pre-outcome conventions artifact is missing: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    digest = sha256_object(payload)
    if digest != CONVENTIONS_CANONICAL_SHA256:
        raise ConventionsDrift(
            f"pre-outcome conventions digest changed: {digest} != {CONVENTIONS_CANONICAL_SHA256}"
        )
    if not payload.get("no_outcome_computed"):
        raise ConventionsDrift("conventions artifact is not marked outcome-free")
    return {"path": str(target), "canonical_sha256": digest}


# --- cache loading ------------------------------------------------------------


@dataclass(slots=True)
class EmbeddingGroup:
    """One (history, seed) embedding group with an ID index."""

    history_id: str
    model_seed: int
    file_ids: list[str]
    index: dict[str, int]
    matrix: np.ndarray


def load_embedding_store(
    root: Path,
    plan: Mapping[str, object],
    identities: Mapping[str, object],
) -> dict[str, EmbeddingGroup]:
    """Load and identity-verify every cached embedding group before any statistic.

    The admission gate must already admit this store: the loader re-runs the
    inspection, coverage and admission checks so a movement run can never start on
    incomplete or foreign-provenance rows.
    """
    inventory = execution.inspect_cache_stores([Path(root)])
    coverage = execution.cache_coverage(plan, identities, inventory)
    execution.admit_execution(coverage, inventory)
    groups: dict[str, EmbeddingGroup] = {}
    for entry in inventory["entries"]:  # type: ignore[union-attr]
        if not entry["valid"]:
            continue
        with np.load(Path(entry["path"]), allow_pickle=False) as payload:
            file_ids = [str(value) for value in np.asarray(payload["file_ids"]).tolist()]
            matrix = np.asarray(payload["embeddings"], dtype=np.float64)
        key = f"{entry['history_id']}|{entry['model_seed']}"
        groups[key] = EmbeddingGroup(
            history_id=str(entry["history_id"]),
            model_seed=int(entry["model_seed"]),
            file_ids=file_ids,
            index={value: position for position, value in enumerate(file_ids)},
            matrix=matrix,
        )
    if len(groups) != len(study.REPRESENTATION_SEEDS) * len(execution.DECLARED_SPAN_ROWS):
        raise RuntimeError(f"embedding store did not yield the declared groups: {len(groups)}")
    return groups


def group_rows(group: EmbeddingGroup, file_ids: Sequence[str]) -> np.ndarray:
    """Return the embedding rows for the requested file IDs, failing closed if absent."""
    missing = [value for value in file_ids if value not in group.index]
    if missing:
        raise RuntimeError(f"embedding group lost {len(missing)} required rows")
    positions = [group.index[value] for value in file_ids]
    return group.matrix[positions]


# --- reference fit ------------------------------------------------------------


@dataclass(slots=True)
class ReferenceFit:
    """Reference-only fitted quantities for one history and representation seed."""

    sigma: float
    sd: np.ndarray
    kmeans: dict[int, KMeans]
    null_q95: float
    null_values: list[float]
    n_reference: int


def _kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian kernel matrix with bandwidth ``sigma``."""
    sq = (
        np.einsum("ij,ij->i", a, a)[:, None]
        + np.einsum("ij,ij->i", b, b)[None, :]
        - 2.0 * (a @ b.T)
    )
    np.maximum(sq, 0.0, out=sq)
    return np.exp(-sq / (2.0 * sigma * sigma))


def _unbiased_from_kernels(
    kxx: np.ndarray, kyy: np.ndarray, kxy: np.ndarray
) -> float:
    n = kxx.shape[0]
    m = kyy.shape[0]
    if n < 2 or m < 2:
        raise ValueError("unbiased MMD requires at least two rows per sample")
    sx = float(kxx.sum() - np.trace(kxx))
    sy = float(kyy.sum() - np.trace(kyy))
    return sx / (n * (n - 1)) + sy / (m * (m - 1)) - 2.0 * float(kxy.sum()) / (n * m)


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Unbiased squared MMD between two samples."""
    return _unbiased_from_kernels(
        _kernel(x, x, sigma), _kernel(y, y, sigma), _kernel(x, y, sigma)
    )


def mmd2_biased(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Biased plug-in squared MMD, reported for continuity only."""
    kxx = _kernel(x, x, sigma)
    kyy = _kernel(y, y, sigma)
    kxy = _kernel(x, y, sigma)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def mmd2_weighted_biased(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
) -> float:
    """Exposure-weighted biased plug-in MMD (declared sensitivity only)."""
    wx = np.asarray(weights_x, dtype=np.float64)
    wy = np.asarray(weights_y, dtype=np.float64)
    wx = wx / wx.sum()
    wy = wy / wy.sum()
    kxx = _kernel(x, x, sigma)
    kyy = _kernel(y, y, sigma)
    kxy = _kernel(x, y, sigma)
    return float(wx @ kxx @ wx + wy @ kyy @ wy - 2.0 * (wx @ kxy @ wy))


def fit_reference(
    group: EmbeddingGroup,
    reference_ids: Sequence[str],
    *,
    resamples: int = 200,
    seed: int = study.BOOTSTRAP_SEED,
) -> ReferenceFit:
    """Fit bandwidth, standardization, cluster prototypes and the sampling null."""
    matrix = group_rows(group, reference_ids)
    n = matrix.shape[0]
    if n < study.MIN_REFERENCE_FILES:
        raise ValueError(f"reference pool below the inherited floor: {n}")
    sigma = float(np.median(pdist(matrix)))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("non-finite reference bandwidth")
    sd = matrix.std(axis=0, ddof=0)
    kmeans = {
        k: KMeans(n_clusters=k, random_state=seed, n_init=10).fit(matrix)
        for k in CLUSTER_COUNTS
    }
    kernel = _kernel(matrix, matrix, sigma)
    rng = np.random.default_rng(seed)
    null_values: list[float] = []
    half = n // 2
    for _ in range(resamples):
        permutation = rng.permutation(n)
        shuffled = kernel[np.ix_(permutation, permutation)]
        a = shuffled[:half, :half]
        b = shuffled[half : 2 * half, half : 2 * half]
        ab = shuffled[:half, half : 2 * half]
        sx = float(a.sum() - np.trace(a))
        sy = float(b.sum() - np.trace(b))
        null_values.append(
            sx / (half * (half - 1))
            + sy / (half * (half - 1))
            - 2.0 * float(ab.sum()) / (half * half)
        )
    return ReferenceFit(
        sigma=sigma,
        sd=sd,
        kmeans=kmeans,
        null_q95=float(np.quantile(null_values, 0.95)),
        null_values=null_values,
        n_reference=n,
    )


# --- endpoint computation -----------------------------------------------------


def _exposure_weights(
    starts: np.ndarray, ends: np.ndarray, period_start: float, period_end: float
) -> np.ndarray | None:
    overlap = np.clip(
        np.minimum(ends, period_end) - np.maximum(starts, period_start), 0.0, None
    )
    total = float(overlap.sum())
    if total <= 0:
        return None
    return overlap / total


def _hash_order(ids: Sequence[str], prefix: str) -> list[str]:
    return sorted(ids, key=lambda value: hashlib.sha256(f"{prefix}{value}".encode()).hexdigest())


def _occupancy(labels: np.ndarray, k: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    return counts / counts.sum()


def _tv(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())


def _cov_trace(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return float("nan")
    return float(np.trace(np.cov(matrix, rowvar=False)))


def pair_endpoint_values(
    *,
    previous_ids: Sequence[str],
    current_ids: Sequence[str],
    previous_period,
    current_period,
    group: EmbeddingGroup,
    fit: ReferenceFit,
    seed: int,
    previous_identity: str,
    current_identity: str,
    history_id: str,
) -> dict[str, object]:
    """Compute every declared endpoint for one pair and representation seed."""
    prev = group_rows(group, previous_ids)
    cur = group_rows(group, current_ids)
    n_prev = prev.shape[0]
    n_cur = cur.shape[0]
    values: dict[str, object] = {"status": "COMPUTED", "reason": "", "n_previous": n_prev, "n_current": n_cur}
    try:
        kxx = _kernel(prev, prev, fit.sigma)
        kyy = _kernel(cur, cur, fit.sigma)
        kxy = _kernel(prev, cur, fit.sigma)
        values["E1"] = _unbiased_from_kernels(kxx, kyy, kxy)
        values["E1_biased"] = float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
        weights_prev = _exposure_weights(
            previous_period.member_starts,
            previous_period.member_ends,
            previous_period.period.start,
            previous_period.period.end,
        )
        weights_cur = _exposure_weights(
            current_period.member_starts,
            current_period.member_ends,
            current_period.period.start,
            current_period.period.end,
        )
        if weights_prev is not None and weights_cur is not None:
            wx = weights_prev / weights_prev.sum()
            wy = weights_cur / weights_cur.sum()
            values["E1_weighted"] = float(
                wx @ kxx @ wx + wy @ kyy @ wy - 2.0 * (wx @ kxy @ wy)
            )
        else:
            values["E1_weighted"] = None
        k = min(n_prev, n_cur)
        capped_prev = _hash_order(
            previous_ids, f"{study.BOOTSTRAP_SEED}|{history_id}|{previous_identity}|"
        )[:k]
        capped_cur = _hash_order(
            current_ids, f"{study.BOOTSTRAP_SEED}|{history_id}|{current_identity}|"
        )[:k]
        values["E1_capped"] = mmd2_unbiased(
            group_rows(group, capped_prev), group_rows(group, capped_cur), fit.sigma
        )
        prev_occ = _occupancy(fit.kmeans[PRIMARY_K].predict(prev), PRIMARY_K)
        cur_occ = _occupancy(fit.kmeans[PRIMARY_K].predict(cur), PRIMARY_K)
        values["E2_K8"] = _tv(prev_occ, cur_occ)
        for k_value in (4, 16):
            prev_occ_k = _occupancy(fit.kmeans[k_value].predict(prev), k_value)
            cur_occ_k = _occupancy(fit.kmeans[k_value].predict(cur), k_value)
            values[f"E2_K{k_value}"] = _tv(prev_occ_k, cur_occ_k)
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(fit.sd > 0, (cur.mean(axis=0) - prev.mean(axis=0)) / fit.sd, np.nan)
        values["E3"] = float(np.nansum(shift**2))
        prev_trace = _cov_trace(prev)
        cur_trace = _cov_trace(cur)
        if (
            np.isfinite(prev_trace)
            and np.isfinite(cur_trace)
            and prev_trace > 0
            and cur_trace > 0
        ):
            values["E4"] = float(abs(np.log(cur_trace / prev_trace)))
        else:
            values["E4"] = None
        half_prev_ids = _hash_order(previous_ids, f"{seed}|{previous_identity}|")
        half_cur_ids = _hash_order(current_ids, f"{seed}|{current_identity}|")
        h_prev = len(half_prev_ids) // 2
        h_cur = len(half_cur_ids) // 2
        if h_prev >= 2 and h_cur >= 2:
            values["N2_previous"] = mmd2_unbiased(
                group_rows(group, half_prev_ids[:h_prev]),
                group_rows(group, half_prev_ids[h_prev : 2 * h_prev]),
                fit.sigma,
            )
            values["N2_current"] = mmd2_unbiased(
                group_rows(group, half_cur_ids[:h_cur]),
                group_rows(group, half_cur_ids[h_cur : 2 * h_cur]),
                fit.sigma,
            )
            values["N2_max"] = max(values["N2_previous"], values["N2_current"])  # type: ignore[type-var]
        else:
            values["N2_previous"] = None
            values["N2_current"] = None
            values["N2_max"] = None
    except (ValueError, FloatingPointError) as exc:
        values["status"] = "UNSUPPORTED"
        values["reason"] = str(exc)
        for key in ENDPOINT_KEYS:
            values.setdefault(key, None)
    return values


# --- tables -------------------------------------------------------------------


def build_endpoint_records(
    manifests: Mapping[str, Mapping[str, object]],
    edges: Mapping[str, Sequence[float]],
    calendar: horizon.CalendarConfig,
    groups: Mapping[str, EmbeddingGroup],
) -> dict[str, object]:
    """Build every period-pair record with per-seed endpoint values."""
    result = study.evaluate_source(manifests, edges, calendar)
    label_blind = study.build_label_blind_manifest(result, calendar)
    posthoc = study.build_posthoc_burden(result)
    pair_rows = {str(row["pair_identity"]): row for row in label_blind["pairs"]}  # type: ignore[union-attr]
    burden_rows = {str(row["pair_identity"]): row for row in posthoc["pairs"]}  # type: ignore[union-attr]
    metas: dict[str, horizon.HistoryMetadata] = {}
    for history_id in sorted(manifests):
        observable = horizon.project_observable(manifests[history_id])
        metas[history_id] = horizon.prepare_history(
            history_id, observable, [float(value) for value in edges[history_id]][4]
        )
    fits: dict[str, ReferenceFit] = {}
    for history_id, meta in metas.items():
        reference_ids = sorted(
            meta.file_ids[int(position)]
            for position in np.flatnonzero(meta.eligible & (meta.ends <= meta.c1))
        )
        for seed in study.REPRESENTATION_SEEDS:
            fits[f"{history_id}|{seed}"] = fit_reference(
                groups[f"{history_id}|{seed}"], reference_ids
            )
    records: list[dict[str, object]] = []
    for history_id, meta in sorted(metas.items()):
        for horizon_name in study.HORIZONS:
            for mode in study.MODES:
                built = horizon.build_history_horizon(meta, horizon_name, mode, calendar)
                for pair in built.pairs:
                    identity = study.pair_identity(
                        history_id, horizon_name, mode, pair.current.period.key
                    )
                    row = pair_rows[identity]
                    burden = burden_rows[identity]
                    record: dict[str, object] = {
                        "history_id": history_id,
                        "horizon": horizon_name,
                        "mode": mode,
                        "pair_identity": identity,
                        "previous_period_identity": study.period_identity(
                            history_id, horizon_name, mode, pair.previous.period.key
                        ),
                        "current_period_identity": study.period_identity(
                            history_id, horizon_name, mode, pair.current.period.key
                        ),
                        "comparison_state": row["comparison_state"],
                        "n_previous": int(row["n_previous"]),
                        "n_current": int(row["n_current"]),
                        "exposure_ratio": row["exposure_ratio"],
                        "mix_drift_tv": row["mix_drift_tv"],
                        "n_reset_edges_crossed": int(row["n_reset_edges_crossed"]),
                        "burden_events": int(burden["n_failure_events_covered"]),
                        "abnormal_member_fraction": burden["abnormal_member_fraction"],
                        "attribution_class": burden["attribution_class"],
                        "previous_annotation_state": burden["previous_annotation_state"],
                        "current_annotation_state": burden["current_annotation_state"],
                        "endpoints": {},
                    }
                    if str(row["comparison_state"]) in COMPARABLE_STATES:
                        for seed in study.REPRESENTATION_SEEDS:
                            record["endpoints"][str(seed)] = pair_endpoint_values(
                                previous_ids=pair.previous.member_ids,
                                current_ids=pair.current.member_ids,
                                previous_period=pair.previous,
                                current_period=pair.current,
                                group=groups[f"{history_id}|{seed}"],
                                fit=fits[f"{history_id}|{seed}"],
                                seed=seed,
                                previous_identity=record["previous_period_identity"],
                                current_identity=record["current_period_identity"],
                                history_id=history_id,
                            )
                    else:
                        for seed in study.REPRESENTATION_SEEDS:
                            record["endpoints"][str(seed)] = {
                                "status": "UNSUPPORTED",
                                "reason": str(row["comparison_state"]),
                                **{key: None for key in ENDPOINT_KEYS},
                            }
                    records.append(record)
    return {
        "calendar": calendar.describe(),
        "records": records,
        "fits": {
            key: {
                "sigma": fit.sigma,
                "null_q95": fit.null_q95,
                "null_median": float(np.median(fit.null_values)),
                "n_reference": fit.n_reference,
            }
            for key, fit in sorted(fits.items())
        },
    }


# --- associations -------------------------------------------------------------


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(np.median(values))


def _seed_value(
    records: Sequence[Mapping[str, object]],
    *,
    history_id: str,
    horizon_name: str,
    mode: str,
    seed: int,
    value_fn: Callable[[Mapping[str, object], Mapping[str, object]], float | None],
) -> dict[str, object] | None:
    rows = [
        record
        for record in records
        if record["history_id"] == history_id
        and record["horizon"] == horizon_name
        and record["mode"] == mode
    ]
    values: list[float] = []
    burdens: list[float] = []
    drifts: list[float] = []
    used: list[Mapping[str, object]] = []
    for record in rows:
        endpoint = record["endpoints"][str(seed)]  # type: ignore[index]
        if endpoint["status"] != "COMPUTED":
            continue
        value = value_fn(record, endpoint)
        if value is None or not np.isfinite(value):
            continue
        burden = float(record["burden_events"])
        values.append(float(value))
        burdens.append(burden)
        drift = record["mix_drift_tv"]
        drifts.append(float(drift) if drift is not None else float("nan"))
        used.append(record)
    if len(values) < study.DECISION_MIN_PAIRS_PER_HISTORY or len(set(burdens)) < 2:
        return None
    rho_burden = spearmanr(values, burdens).statistic
    if drifts and all(np.isfinite(drift) for drift in drifts):
        x_rank = rankdata(values, method="average")
        d_rank = rankdata(drifts, method="average")
        design = np.column_stack([np.ones_like(d_rank), d_rank])
        coefficients, *_ = np.linalg.lstsq(design, x_rank, rcond=None)
        residual = x_rank - design @ coefficients
        rho_residual = spearmanr(residual, burdens).statistic
    else:
        rho_residual = None
    return {
        "history_id": history_id,
        "n_pairs": len(values),
        "n_distinct_burden": len(set(burdens)),
        "rho_burden": float(rho_burden),
        "rho_residual": None if rho_residual is None else float(rho_residual),
        "pair_identities": [str(record["pair_identity"]) for record in used],
        "values": values,
        "burden": burdens,
    }


def horizon_associations(
    records: Sequence[Mapping[str, object]],
    horizon_name: str,
    mode: str,
    value_fn: Callable[[Mapping[str, object], Mapping[str, object]], float | None],
    history_ids: Sequence[str],
) -> dict[str, object]:
    """Within-history associations per seed and their frozen seed medians."""
    per_history: dict[str, dict[str, object]] = {}
    for history_id in history_ids:
        per_seed = {
            str(seed): _seed_value(
                records,
                history_id=history_id,
                horizon_name=horizon_name,
                mode=mode,
                seed=seed,
                value_fn=value_fn,
            )
            for seed in study.REPRESENTATION_SEEDS
        }
        available = [value for value in per_seed.values() if value is not None]
        burden_rhos = [float(value["rho_burden"]) for value in available]
        residual_rhos = [
            float(value["rho_residual"])
            for value in available
            if value["rho_residual"] is not None
        ]
        per_history[history_id] = {
            "per_seed": per_seed,
            "n_pairs": available[0]["n_pairs"] if available else 0,
            "n_distinct_burden": available[0]["n_distinct_burden"] if available else 0,
            "seed_rho_burden": burden_rhos,
            "rho_burden": _median(burden_rhos),
            "seed_rho_residual": residual_rhos,
            "rho_residual": _median(residual_rhos),
            "seed_sign_agreement": (
                len({1 if value > 0 else -1 if value < 0 else 0 for value in burden_rhos}) == 1
                if burden_rhos
                else False
            ),
            "estimable": bool(available),
        }
    return {"horizon": horizon_name, "mode": mode, "per_history": per_history}


def _e1(record: Mapping[str, object], endpoint: Mapping[str, object]) -> float | None:
    if record["comparison_state"] != MATCHED_STATE:
        return None
    value = endpoint.get("E1")
    return None if value is None else float(value)


def _e1_capped(record: Mapping[str, object], endpoint: Mapping[str, object]) -> float | None:
    if record["comparison_state"] == MATCHED_STATE:
        value = endpoint.get("E1")
    else:
        value = endpoint.get("E1_capped")
    return None if value is None else float(value)


def e1_matched(record: Mapping[str, object], endpoint: Mapping[str, object]) -> float | None:
    """Association value function for the primary/rolling/anchor matched sets."""
    return _e1(record, endpoint)


def e1_capped(record: Mapping[str, object], endpoint: Mapping[str, object]) -> float | None:
    """Association value function for the capped-``n`` sensitivity set."""
    return _e1_capped(record, endpoint)


# --- uncertainty and decision -------------------------------------------------


def bootstrap_uncertainty(
    history_rhos: Sequence[float],
    *,
    replicates: int = study.BOOTSTRAP_REPLICATES,
    seed: int = study.BOOTSTRAP_SEED,
) -> dict[str, object]:
    """History-cluster bootstrap of the median association (descriptive)."""
    if not history_rhos:
        return {"status": "UNAVAILABLE", "reason": "no estimable histories"}
    rng = np.random.default_rng(seed)
    draws = np.asarray(history_rhos, dtype=np.float64)
    medians = [
        float(np.median(draws[rng.integers(0, len(draws), size=len(draws))]))
        for _ in range(replicates)
    ]
    return {
        "status": "DESCRIPTIVE",
        "unit": "history",
        "replicates": replicates,
        "seed": seed,
        "point_median": float(np.median(draws)),
        "ci95_low": float(np.quantile(medians, 0.025)),
        "ci95_high": float(np.quantile(medians, 0.975)),
        "note": "n = 4 histories; descriptive only, never a precise standard error",
    }


def decision_inputs(
    primary: Mapping[str, object],
    sensitivity_values: Mapping[str, float | None],
    history_ids: Sequence[str],
) -> dict[str, object]:
    """Assemble the frozen decision-rule input from seed-median history values."""
    histories = []
    for history_id in history_ids:
        entry = primary["per_history"][history_id]  # type: ignore[index]
        if entry["rho_burden"] is None:
            continue
        histories.append(
            {
                "history_id": history_id,
                "n_pairs": int(entry["n_pairs"]),
                "n_distinct_burden": int(entry["n_distinct_burden"]),
                "rho_burden": entry["rho_burden"],
                "rho_residual": entry["rho_residual"],
            }
        )
    return {
        "histories": histories,
        "sensitivity": {
            name: {"rho_burden": value} for name, value in sensitivity_values.items()
        },
    }


# --- execution cells ----------------------------------------------------------


def endpoint_cell_summaries(
    records: Sequence[Mapping[str, object]],
    coverage_matrix: Mapping[str, object],
) -> list[dict[str, object]]:
    """Summarize the actual endpoint outcomes for every declared matrix cell."""
    endpoint_keys = {
        "E1": "E1",
        "E2": "E2_K8",
        "E3": "E3",
        "E4": "E4",
    }
    cells = coverage_matrix["cells"]
    assert isinstance(cells, list)
    summaries: list[dict[str, object]] = []
    for cell in cells:
        key = endpoint_keys[str(cell["endpoint_id"])]
        values: list[float] = []
        n_pairs = 0
        for record in records:
            if (
                record["history_id"] != cell["history_id"]
                or record["horizon"] != cell["horizon"]
                or record["mode"] != cell["mode"]
            ):
                continue
            if str(record["comparison_state"]) not in COMPARABLE_STATES:
                continue
            endpoint = record["endpoints"][str(cell["representation_seed"])]  # type: ignore[index]
            if endpoint["status"] != "COMPUTED":
                continue
            value = endpoint.get(key)
            if value is None or not np.isfinite(value):
                continue
            n_pairs += 1
            values.append(float(value))
        summaries.append(
            {
                "horizon": cell["horizon"],
                "mode": cell["mode"],
                "mode_role": cell["mode_role"],
                "endpoint_id": cell["endpoint_id"],
                "endpoint_role": cell["endpoint_role"],
                "decision_relevant": bool(cell["decision_relevant"]),
                "representation_seed": cell["representation_seed"],
                "history_id": cell["history_id"],
                "metadata_status": cell["status"],
                "n_pairs": n_pairs,
                "median": _median(values),
                "min": None if not values else float(min(values)),
                "max": None if not values else float(max(values)),
                "execution_status": "EXECUTED" if values else "UNSUPPORTED-NO-COMPUTABLE-PAIRS",
            }
        )
    return summaries


__all__ = [
    "CLUSTER_COUNTS",
    "COMPARABLE_STATES",
    "CONVENTIONS_CANONICAL_SHA256",
    "CONVENTIONS_PATH",
    "ConventionsDrift",
    "ENDPOINT_KEYS",
    "EmbeddingGroup",
    "MATCHED_STATE",
    "PRIMARY_K",
    "ReferenceFit",
    "assert_conventions_frozen",
    "bootstrap_uncertainty",
    "build_endpoint_records",
    "decision_inputs",
    "e1_capped",
    "e1_matched",
    "endpoint_cell_summaries",
    "fit_reference",
    "group_rows",
    "horizon_associations",
    "load_embedding_store",
    "mmd2_biased",
    "mmd2_unbiased",
    "mmd2_weighted_biased",
    "pair_endpoint_values",
]
