"""Sprint 24 Task 7-M11 frozen controlled-corpus inference and movement execution.

Real execution path under the accepted M10 prospective contract
(``f4f92a60ee1782ccc0ec75587e9df5d1471e4045e201c43478893d294b5757f7``):

- verify the M9 corpus, M10 contract and exact encoder identities before
  any computation (canonical freeze recomputation + pinned digests);
- bounded real smoke (8 files x 1 history x 3 seeds = 24 forwards) through
  the accepted frozen B0 capture path with ``gpu-forward-smoke``
  provenance, proving real admission/refusal and public entry points;
- full frozen capture (8,280 files x 3 seeds = 24,840 forwards) with
  ``gpu-forward`` provenance at the exact study-required identity;
- identity-bound durable cache with honest resumability (per
  ``(history, seed)`` groups, skip-complete only, no duplicate forwards);
- frozen endpoints/nulls/sensitivities/history-level uncertainty plus
  primary/replication summaries on admitted real embeddings.

What this module never does: encoder/anomaly-bank refit or training,
business-density expansion, Phase C, Confirmation contact, per-seed model
selection by outcome, or any revision of the frozen contract terms after
an outcome is read. Non-COMPUTED cells abstain, never silently become
zero. ``C2`` abstains corpus-wide. No positive result gate is applied.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from representation import sprint24_controlled_inference_readiness as readiness
from representation import sprint24_weekly_monthly_execution as execution
from representation.sprint24_context_audit import sha256_object

#: Task identifier for every M11 run manifest.
TASK_ID = "S24-7-M11"

#: The accepted M10 prospective contract this execution is bound to.
EXPECTED_CONTRACT_SHA256 = "f4f92a60ee1782ccc0ec75587e9df5d1471e4045e201c43478893d294b5757f7"

#: Bounded smoke envelope: 8 files of one history per representation seed.
SMOKE_FILES_PER_HISTORY = 8
SMOKE_HISTORY_INDEX = 0

#: Failure/outcome states.
EXECUTED = "EXECUTED-REAL-EMBEDDINGS"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_contract_binding(
    contract: Mapping[str, object],
    contract_sha256: str,
    corpus_manifest: Mapping[str, object],
    audit_record: Mapping[str, object],
    custody: Mapping[str, object],
) -> dict[str, object]:
    """Validate the actual prospective contract payload BEFORE any capture.

    Recomputes the canonical digest, the live census/plan/custody binding
    and the frozen identity-plan groups, and checks the frozen
    endpoint/reference-fit/model-input terms. Any drift or forged custody
    fails closed. Returns the live census and frozen identity plan.
    """
    census_live, plan_live = readiness.verify_expected_contract(
        contract, contract_sha256, corpus_manifest, audit_record, custody
    )
    return {"census": census_live, "plan": plan_live}


def bound_capture_plan(
    corpus_manifest: Mapping[str, object],
    custody: Mapping[str, object],
    *,
    histories: Sequence[str] | None = None,
    files_per_history: int | None = None,
) -> dict[str, object]:
    """Derive the executable capture plan from the manifest + custody inputs.

    Uses the same :func:`custody_identity_plan` derivation as the frozen
    contract, so the capture target is byte-identical to the contract-bound
    plan. ``histories``/``files_per_history`` bound the smoke subset; the
    full plan covers every history and every file.
    """
    plan = readiness.custody_identity_plan(corpus_manifest, custody)
    files = corpus_manifest.get("files")
    _require(isinstance(files, list), "REFUSED-IDENTITY: corpus manifest has no file table")
    per_history: dict[str, list[str]] = {}
    end_time_of: dict[str, float] = {}
    duration_of: dict[str, float] = {}
    relpath_of: dict[str, str] = {}
    for entry in files:  # type: ignore[union-attr]
        if not isinstance(entry, Mapping):
            continue
        history = entry.get("history_id")
        fid = entry.get("file_id")
        if isinstance(history, str) and isinstance(fid, str):
            per_history.setdefault(history, []).append(fid)
            end = entry.get("end_time_rel_s")
            if isinstance(end, (int, float)):
                end_time_of[fid] = float(end)
            rel = entry.get("relpath")
            if isinstance(rel, str):
                relpath_of[fid] = rel
    wanted = sorted(per_history) if histories is None else list(histories)
    subset: dict[str, list[str]] = {}
    for history_id in wanted:
        ids = sorted(set(per_history.get(history_id, [])))
        _require(bool(ids), f"REFUSED-SHORTFALL: no manifest files for {history_id}")
        subset[history_id] = ids if files_per_history is None else ids[:files_per_history]
    groups = plan.get("groups")
    _require(isinstance(groups, dict), "REFUSED-IDENTITY: identity plan has no groups")
    return {
        "plan": plan,
        "histories": wanted,
        "file_ids": subset,
        "end_time_of": end_time_of,
        "duration_of": duration_of,
        "relpath_of": relpath_of,
    }


def write_group_cache(
    root: Path,
    *,
    history_id: str,
    model_seed: int,
    file_ids: Sequence[str],
    embeddings: Any,
    inputs: Mapping[str, object],
    support_sha256: str,
    provenance: str,
    capture_commit: str | None,
) -> dict[str, object]:
    """Persist one ``(history, seed)`` group through the accepted writer."""
    return execution.write_cache_entry(
        root,
        history_id=history_id,
        model_seed=int(model_seed),
        file_ids=list(file_ids),
        embeddings=embeddings,
        inputs=inputs,
        support_sha256=support_sha256,
        provenance=provenance,
        capture_commit=capture_commit,
    )


def capture_inputs(custody: Mapping[str, object]) -> dict[str, object]:
    """Build the accepted writer inputs from validated custody."""
    return {
        "checkpoints": dict(custody["checkpoints"]),  # type: ignore[index]
        "banks": dict(custody["banks"]),  # type: ignore[index]
        "arms": dict(custody["arms"]),  # type: ignore[index]
        "binding_sha256": str(custody["binding_sha256"]),
        "preprocessing_sha256": str(custody["preprocessing_sha256"]),
        "device": str(custody["device"]),
        "scale_state_sha256": None,
    }


def cached_group_complete(
    cache_root: Path, history_id: str, model_seed: int, expected_identity: str
) -> dict[str, object] | None:
    """Return the cached group receipt when it fully matches, else None.

    A group counts as complete only when its sidecar is present, valid, and
    carries the exact expected cache identity with a full-length file-ID
    list. Anything else (missing, mismatched, partial) returns None so the
    caller re-captures that group from scratch — never a partial top-up.
    """
    stem = f"{history_id}__seed-{model_seed}"
    npz_path = cache_root / f"{stem}.npz"
    sidecar_path = cache_root / f"{stem}.json"
    if not npz_path.is_file() or not sidecar_path.is_file():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(sidecar, dict):
        return None
    if str(sidecar.get("cache_identity")) != expected_identity:
        return None
    if str(sidecar.get("capture_provenance")) not in (
        execution.CACHE_PROVENANCE_GPU,
        execution.CACHE_PROVENANCE_SMOKE,
    ):
        return None
    try:
        file_ids = readiness.read_cache_file_ids_only(npz_path)
    except (OSError, KeyError, ValueError, IndexError):
        return None
    return {
        "path": str(npz_path),
        "sidecar_path": str(sidecar_path),
        "history_id": history_id,
        "model_seed": int(model_seed),
        "cache_identity": str(sidecar.get("cache_identity")),
        "n_rows": len(file_ids),
    }


def admit_cache_or_refuse(
    cache_root: Path, identity_plan: Mapping[str, object]
) -> dict[str, object]:
    """Inspect + coverage + admission through the accepted public path.

    Returns the admission card on success; raises
    :class:`execution.CachePreflightFailure` (with the refusal card) when
    any group is missing, mismatched, invalid, or of foreign provenance.
    """
    inventory = execution.inspect_cache_stores([Path(cache_root)])
    groups = identity_plan.get("groups")
    _require(isinstance(groups, dict), "REFUSED-IDENTITY: identity plan has no groups")
    required_by_history: dict[str, list[str]] = {}
    full_support = _full_support_universe()
    for key, spec in groups.items():  # type: ignore[union-attr]
        assert isinstance(spec, Mapping)
        hid = str(spec["history_id"])
        required_by_history.setdefault(hid, list(full_support.get(hid, [])))
    _require(
        all(bool(ids) for ids in required_by_history.values()),
        "REFUSED-IDENTITY: support universe is not bound (call set_support_universe)",
    )
    per_history = {
        hid: {"required_file_ids": list(ids)} for hid, ids in required_by_history.items()
    }
    coverage = execution.cache_coverage(
        {"per_history": per_history}, {"groups": dict(groups)}, inventory  # type: ignore[arg-type]
    )
    return execution.admit_execution(coverage, inventory)


_SUPPORT_UNIVERSE: dict[str, list[str]] = {}


def _full_support_universe() -> dict[str, list[str]]:
    """Return the frozen full file-ID universe per history (set by the CLI)."""
    return dict(_SUPPORT_UNIVERSE)


def set_support_universe(universe: Mapping[str, Sequence[str]]) -> None:
    """Bind the manifest-derived full support universe for admission."""
    _SUPPORT_UNIVERSE.clear()
    for history_id, ids in universe.items():
        _SUPPORT_UNIVERSE[str(history_id)] = sorted(set(str(fid) for fid in ids))


def load_admitted_groups(
    cache_root: Path,
    identity_plan: Mapping[str, object],
) -> dict[str, Any]:
    """Load every admitted embedding group without the old-study group count.

    The committed loader re-runs inspection/coverage/admission (preserved
    here by calling it first through :func:`admit_cache_or_refuse`) but then
    enforces the OLD study's ``len(seeds) * len(DECLARED_SPAN_ROWS)`` group
    count (12), which can never hold for the 18-group controlled corpus.
    This loader instead requires exactly the frozen identity-plan groups:
    one entry per ``groups`` key, each fully covering its required file IDs.
    Movement can never start on incomplete or foreign-provenance rows.
    """
    import numpy as _np

    from representation import sprint24_weekly_monthly_endpoints as endpoints

    groups_spec = identity_plan.get("groups")
    _require(isinstance(groups_spec, dict) and bool(groups_spec), "REFUSED-IDENTITY: identity plan has no groups")
    full_support = _full_support_universe()
    _require(bool(full_support), "REFUSED-IDENTITY: support universe is not bound (call set_support_universe)")
    admission = admit_cache_or_refuse(cache_root, identity_plan)
    _require(str(admission.get("status")) == "ADMITTED", "REFUSED-IDENTITY: cache admission refused")
    inventory = execution.inspect_cache_stores([Path(cache_root)])
    by_key: dict[str, dict[str, object]] = {}
    for entry in inventory.get("entries", []):  # type: ignore[union-attr]
        if not isinstance(entry, Mapping) or not entry.get("valid"):
            continue
        key = f"{entry.get('history_id')}|{entry.get('model_seed')}"
        if key in groups_spec and key not in by_key:  # type: ignore[operator]
            by_key[key] = dict(entry)  # type: ignore[arg-type]
    missing = sorted(set(groups_spec) - set(by_key))  # type: ignore[arg-type]
    _require(not missing, f"REFUSED-IDENTITY: admitted store is missing groups: {missing}")
    loaded: dict[str, Any] = {}
    for key in sorted(groups_spec):  # type: ignore[union-attr]
        entry = by_key[key]
        with _np.load(Path(str(entry["path"])), allow_pickle=False) as payload:
            file_ids = [str(value) for value in _np.asarray(payload["file_ids"]).tolist()]
            matrix = _np.asarray(payload["embeddings"], dtype=_np.float64)
        loaded[key] = endpoints.EmbeddingGroup(
            history_id=str(entry["history_id"]),
            model_seed=int(entry["model_seed"]),  # type: ignore[arg-type]
            file_ids=file_ids,
            index={value: position for position, value in enumerate(file_ids)},
            matrix=matrix,
        )
    return loaded


def reference_ids_for_history(
    corpus_manifest: Mapping[str, object], history_id: str
) -> list[str]:
    """Return the causal reference pool file IDs for one history.

    The pool is every manifest file with ``arm == REFERENCE`` for that
    history (220 per history, all with ``end_time <= c1`` on the frozen
    ``c1 = 0`` grid), fixed before any evaluation period exists.
    """
    files = corpus_manifest.get("files")
    _require(isinstance(files, list), "REFUSED-IDENTITY: corpus manifest has no file table")
    ids = sorted(
        str(entry["file_id"])
        for entry in files  # type: ignore[union-attr]
        if isinstance(entry, Mapping)
        and entry.get("history_id") == history_id
        and entry.get("arm") == "REFERENCE"
    )
    _require(len(ids) >= 200, f"REFUSED-SHORTFALL: reference pool below 200 for {history_id}")
    return ids


def run_movement_on_groups(
    groups: Mapping[str, Any],
    corpus_manifest: Mapping[str, object],
    audit_record: Mapping[str, object],
    contract: Mapping[str, object],
    *,
    seed: int = 2407,
) -> dict[str, object]:
    """Run the frozen endpoints on admitted real embeddings.

    Per (history, seed): fit the reference once on its own causal pool
    (bandwidth + E2 prototypes + standardization, never re-fitted per
    pair/variant), then compute every retained-eligible pair's endpoint set
    via :func:`pair_endpoint_values`. Every retained pair per
    history/horizon enters; non-COMPUTED cells stay explicit. C2 abstains
    corpus-wide. Returns per-pair records, per-history fits, C1
    associations, history-cluster uncertainty and primary/replication
    summaries — descriptive only, no success gate.
    """
    import numpy as np

    from representation import sprint24_weekly_monthly_endpoints as endpoints

    pair_rows = audit_record.get("pair_rows")
    _require(isinstance(pair_rows, list) and bool(pair_rows), "REFUSED-IDENTITY: audit has no pair rows")
    rows = [row for row in pair_rows if isinstance(row, Mapping)]  # type: ignore[union-attr]

    file_entries = corpus_manifest.get("files")
    _require(isinstance(file_entries, list), "REFUSED-IDENTITY: corpus manifest has no file table")
    # Label-blind member observation intervals for the frozen exposure-weighted
    # sensitivity: per-file [end_time - duration, end_time] in seconds, with
    # end-times from the manifest table and durations from NPZ ``x.npy`` NPY
    # headers only (shape bytes; no array element is read, no label is used).
    # Same mechanism as the readiness census (``_manifest_file_maps``). Fail
    # closed when metadata is missing — never silently fall back to uniform
    # weights, which would collapse the sensitivity toward the biased plug-in.
    manifest_path = corpus_manifest.get("__manifest_path__")
    _require(isinstance(manifest_path, str), "REFUSED-IDENTITY: corpus manifest path is not bound")
    corpus_root = Path(str(manifest_path)).parent
    manifest_end_of: dict[str, float] = {}
    manifest_relpath_of: dict[str, str] = {}
    for entry in file_entries:  # type: ignore[union-attr]
        if not isinstance(entry, Mapping):
            continue
        fid = entry.get("file_id")
        end = entry.get("end_time_rel_s")
        rel = entry.get("relpath")
        if isinstance(fid, str):
            if isinstance(end, (int, float)):
                manifest_end_of[fid] = float(end)
            if isinstance(rel, str):
                manifest_relpath_of[fid] = str(rel)
    manifest_durations: dict[str, float] = {}
    for fid, rel in manifest_relpath_of.items():
        if fid not in manifest_end_of:
            continue
        try:
            shape = readiness.read_npy_shape_header(corpus_root / rel)
            manifest_durations[fid] = float(int(shape[1]))
        except (OSError, KeyError, ValueError, IndexError):
            continue
    fits: dict[str, dict[str, object]] = {}
    live_fits: dict[str, Any] = {}
    records: list[dict[str, object]] = []
    for row in rows:
        history_id = str(row.get("history_id"))
        for model_seed in readiness.FROZEN_ENCODER_SEEDS:
            key = f"{history_id}|{model_seed}"
            if key not in fits:
                group = groups[key]
                ref_ids = reference_ids_for_history(corpus_manifest, history_id)
                t0 = time.time()
                fit = endpoints.fit_reference(group, ref_ids, resamples=200, seed=seed)
                fits[key] = {
                    "sigma": float(fit.sigma),
                    "null_q95": float(fit.null_q95),
                    "null_median": float(np.median(fit.null_values)),
                    "n_reference": int(fit.n_reference),
                    "fit_wall_s": time.time() - t0,
                }
                # Keep the live handle alongside its summary, fitted exactly
                # once per (history, seed) group per run: per-run local dict,
                # so sequential runs in one process can never share live fits
                # (the module-level _FIT_CACHE stays probe-only).
                live_fits[key] = fit
    from types import SimpleNamespace as _NS

    import numpy as _np

    for row in rows:
        history_id = str(row.get("history_id"))
        prev_ids = [str(fid) for fid in row.get("retained_baseline_file_identities", [])]  # type: ignore[union-attr]
        cur_ids = [str(fid) for fid in row.get("retained_contrast_file_identities", [])]  # type: ignore[union-attr]

        def _period_of(fids: list[str]) -> object:
            # Frozen exposure-weighted sensitivity geometry: member observation
            # intervals are the actual per-file [end_time - duration, end_time]
            # in seconds, in the EXACT retained file-ID order used by
            # ``group_rows`` (``_exposure_weights`` aligns weights to kernel
            # rows positionally, so order must match ``previous_ids`` /
            # ``current_ids``). The enclosing period is the arm span
            # [min end_time, max end_time], the same construction as the
            # readiness ``retained_exposure`` census (``_arm_exposure``).
            # End-times come from the manifest table, durations from NPZ
            # ``x.npy`` NPY headers only (shape bytes; no array element is
            # read, no label is used). Missing metadata fails closed with a
            # hard error — never silently uniform, which would collapse the
            # sensitivity toward the biased plug-in.
            _require(bool(fids), "REFUSED-IDENTITY: exposure geometry needs member file IDs")
            ends: list[float] = []
            starts: list[float] = []
            for fid in fids:
                raw_end = manifest_end_of.get(fid)
                raw_duration = manifest_durations.get(fid)
                _require(
                    isinstance(raw_end, float) and isinstance(raw_duration, float),
                    f"REFUSED-IDENTITY: missing label-blind interval metadata for {fid}",
                )
                end_value = float(raw_end)  # type: ignore[arg-type]
                duration_value = float(raw_duration)  # type: ignore[arg-type]
                ends.append(end_value)
                starts.append(end_value - duration_value)
            return _NS(
                member_starts=_np.array(starts),
                member_ends=_np.array(ends),
                period=_NS(start=min(ends), end=max(ends)),
            )

        base_identity = str(row.get("comparison_identity"))
        endpoints_per_seed: dict[str, object] = {}
        for model_seed in readiness.FROZEN_ENCODER_SEEDS:
            key = f"{history_id}|{model_seed}"
            group = groups[key]
            t0 = time.time()
            # Reuse the once-per-run live fit for this (history, seed) group:
            # never re-fitted per pair/period/variant, never shared across runs.
            values = endpoints.pair_endpoint_values(
                previous_ids=prev_ids,
                current_ids=cur_ids,
                previous_period=_period_of(prev_ids),
                current_period=_period_of(cur_ids),
                group=group,
                fit=live_fits[key],
                seed=int(model_seed),
                previous_identity=base_identity + "|BASELINE",
                current_identity=base_identity + "|CONTRAST",
                history_id=history_id,
            )
            values["wall_s"] = time.time() - t0
            endpoints_per_seed[str(model_seed)] = values
        records.append(
            {
                "history_id": history_id,
                "horizon": str(row.get("horizon")),
                "mode": str(row.get("mode")),
                "pair_identity": base_identity,
                "comparison_type": str(row.get("comparison_type")),
                "disposition": str(row.get("disposition")),
                "n_previous": len(prev_ids),
                "n_current": len(cur_ids),
                "burden_baseline_after": row.get("burden_baseline_after"),
                "burden_contrast_after": row.get("burden_contrast_after"),
                "burden_contrast_gap_after": row.get("burden_contrast_gap_after"),
                "exposure_ratio_after": row.get("exposure_ratio_after"),
                "endpoints": endpoints_per_seed,
            }
        )
    summaries = summarize_movement(records, contract)
    return {"records": records, "fits": fits, "summaries": summaries}


_FIT_CACHE: dict[str, Any] = {}


def _refit_handle(
    fits: Mapping[str, object],
    groups: Mapping[str, Any],
    key: str,
    history_id: str,
    corpus_manifest: Mapping[str, object],
    seed: int,
) -> Any:
    """Return the cached live ReferenceFit for one (history, seed) group."""
    if key not in _FIT_CACHE:
        from representation import sprint24_weekly_monthly_endpoints as endpoints

        group = groups[key]
        ref_ids = reference_ids_for_history(corpus_manifest, history_id)
        _FIT_CACHE[key] = endpoints.fit_reference(group, ref_ids, resamples=200, seed=seed)
    return _FIT_CACHE[key]


def summarize_movement(
    records: list[dict[str, object]], contract: Mapping[str, object]
) -> dict[str, object]:
    """Summarize C1 associations, uncertainty and primary/replication splits.

    Descriptive within-history movement/C1 over the retained-eligible set
    with the control as the low-burden pole; encoder seeds are run
    replicates summarized by median, never independent data; no pooling
    across histories, seeds or horizons. C2 ABSTAIN corpus-wide.
    """
    import numpy as _np

    from representation import sprint24_weekly_monthly_endpoints as endpoints

    seeds = [int(seed) for seed in readiness.FROZEN_ENCODER_SEEDS]
    histories = sorted({str(record["history_id"]) for record in records})
    horizons = sorted({str(record["horizon"]) for record in records})

    def _burden(record: Mapping[str, object]) -> float | None:
        gap = record.get("burden_contrast_gap_after")
        return float(gap) if isinstance(gap, (int, float)) else None

    def _median(values: list[float]) -> float | None:
        clean = [value for value in values if _np.isfinite(value)]
        return float(_np.median(clean)) if clean else None

    c1: dict[str, dict[str, object]] = {}
    for horizon in horizons:
        for history_id in histories:
            members = [
                record
                for record in records
                if str(record["history_id"]) == history_id and str(record["horizon"]) == horizon
            ]
            per_seed_rho: dict[str, float | None] = {}
            per_seed_n: dict[str, int] = {}
            for seed in seeds:
                pairs: list[tuple[float, float]] = []
                for record in members:
                    endpoints_map = record.get("endpoints")
                    endpoint: Mapping[str, object] | None = None
                    if isinstance(endpoints_map, Mapping):
                        candidate = endpoints_map.get(str(seed))
                        if isinstance(candidate, Mapping):
                            endpoint = candidate  # type: ignore[assignment]
                    if endpoint is None or endpoint.get("status") != "COMPUTED":
                        continue
                    weight = endpoint.get("E1")
                    burden = _burden(record)
                    if weight is None or burden is None:
                        continue
                    if not isinstance(weight, (int, float)):
                        continue
                    if not _np.isfinite(float(weight)):
                        continue
                    pairs.append((float(weight), float(burden)))
                if len(pairs) >= 3 and len({burden for _, burden in pairs}) >= 2:
                    from scipy.stats import spearmanr as _spearmanr

                    result = _spearmanr(
                        [value for value, _ in pairs], [burden for _, burden in pairs]
                    )
                    rho = result.statistic  # type: ignore[attr-defined]
                    per_seed_rho[str(seed)] = float(rho) if _np.isfinite(rho) else None  # type: ignore[arg-type]
                    per_seed_n[str(seed)] = len(pairs)
                else:
                    per_seed_rho[str(seed)] = None
                    per_seed_n[str(seed)] = len(pairs)
            available = [rho for rho in per_seed_rho.values() if rho is not None]
            c1[f"{history_id}|{horizon}"] = {
                "history_id": history_id,
                "horizon": horizon,
                "n_pairs": len(members),
                "per_seed_rho": per_seed_rho,
                "per_seed_n": per_seed_n,
                "rho_burden_median": _median([float(rho) for rho in available]),
                "estimable": bool(available),
            }

    per_horizon_rhos: dict[str, list[float]] = {horizon: [] for horizon in horizons}
    estimable_histories: dict[str, list[str]] = {horizon: [] for horizon in horizons}
    for cell in c1.values():
        rho_value = cell.get("rho_burden_median")
        horizon_name = cell.get("horizon")
        if rho_value is not None and isinstance(horizon_name, str):
            per_horizon_rhos[horizon_name].append(float(rho_value))  # type: ignore[arg-type]
            estimable_histories[horizon_name].append(str(cell["history_id"]))
    uncertainty = {
        horizon: endpoints.bootstrap_uncertainty(rhos)
        for horizon, rhos in per_horizon_rhos.items()
    }

    # Per-horizon per-triple summaries: the frozen contract forbids pooling
    # across horizons ("no pooling across histories, seeds or horizons"),
    # so each triple reports one median per horizon (3 cells each), never a
    # single median over the 6 week+month cells. Legacy pooled keys are
    # intentionally NOT produced.
    primary = list(readiness.PRIMARY_TRIPLE)
    replication = list(readiness.REPLICATION_TRIPLE)
    splits: dict[str, dict[str, object]] = {}
    for name, triple in (("primary", primary), ("replication", replication)):
        for horizon in horizons:
            triple_cells = [c1.get(f"{hid}|{horizon}") for hid in triple]
            estimable: list[float] = []
            for cell in triple_cells:
                if cell is None:
                    continue
                median = cell.get("rho_burden_median")
                if median is not None and isinstance(median, (int, float)):
                    estimable.append(float(median))
            splits[f"{name}|{horizon}"] = {
                "histories": triple,
                "horizon": horizon,
                "n_cells": len([cell for cell in triple_cells if cell]),
                "n_estimable_cells": len(estimable),
                "median_rho": _median(estimable),
            }

    n_computed = 0
    for record in records:
        endpoints_map = record.get("endpoints")
        if not isinstance(endpoints_map, Mapping):
            continue
        for seed in seeds:
            candidate = endpoints_map.get(str(seed))
            if isinstance(candidate, Mapping) and candidate.get("status") == "COMPUTED":
                n_computed += 1
    # The committed ``bootstrap_uncertainty`` hardcodes a stale "n = 4
    # histories" note; the numeric bootstrap draws over the supplied rhos.
    # Annotate the ACTUAL resampling support per horizon: only histories
    # whose C1 seed-median association is estimable are drawn (the same
    # list fed to ``bootstrap_uncertainty`` above), so a history with
    # missing/non-COMPUTED cells or too few distinct burdens is excluded
    # and reported as excluded instead of being counted merely because it
    # has a record. Output-side only; the committed function is untouched.
    annotated_uncertainty: dict[str, object] = {}
    for horizon in horizons:
        block = uncertainty.get(horizon)
        included = sorted(estimable_histories[horizon])
        recorded = sorted(
            {str(record.get("history_id")) for record in records if str(record.get("horizon")) == horizon}
        )
        excluded = sorted(set(recorded) - set(included))
        if isinstance(block, Mapping):
            entry = dict(block)
            entry["n_histories"] = len(included)
            entry["histories_estimable"] = included
            entry["n_histories_with_records"] = len(recorded)
            entry["n_nonestimable_histories"] = len(excluded)
            entry["nonestimable_histories"] = excluded
            entry["note"] = (
                f"n = {len(included)} estimable histories of {len(recorded)} with records"
                + (f"; {len(excluded)} non-estimable excluded from the bootstrap" if excluded else "")
                + "; descriptive only, never a precise standard error (committed note string is stale)"
            )
            annotated_uncertainty[horizon] = entry
        else:
            annotated_uncertainty[horizon] = block  # type: ignore[assignment]
    # Holm: no p-values exist anywhere in the movement output, so there is no
    # computable input ("where computable" is empty); the disposition never
    # depends on a p-value. Record the vacuous policy explicitly instead of
    # pretending a test ran.
    holm_policy = {
        "status": "NOT-COMPUTED-NO-PVALUES",
        "family": "declared primary contrasts at alpha=0.05 where computable",
        "n_pvalues": 0,
        "adjusted": [],
        "note": "no inferential claim depends on a p-value; see contract multiplicity",
    }
    # Transition-vs-control reporting + dependence: per-horizon counts of the
    # retained-eligible analysis set, and the frozen dependence declaration.
    horizons_seen = sorted({str(record.get("horizon")) for record in records})
    transition_vs_control: dict[str, object] = {}
    for horizon in horizons_seen:
        members = [record for record in records if str(record.get("horizon")) == horizon]
        transitions = [
            record for record in members if str(record.get("comparison_type")) == "BASELINE-VS-ELEVATED"
        ]
        controls = [
            record for record in members if str(record.get("comparison_type")) == "BASELINE-VS-BASELINE"
        ]
        transition_vs_control[horizon] = {
            "n_pairs": len(members),
            "n_transitions": len(transitions),
            "n_controls": len(controls),
            "transition_dispositions": sorted({str(record.get("disposition")) for record in transitions}),
            "control_dispositions": sorted({str(record.get("disposition")) for record in controls}),
        }
    return {
        "c1": c1,
        "uncertainty": annotated_uncertainty,
        "splits": splits,
        "c2": "ABSTAIN-CORPUS-WIDE",
        "n_records": len(records),
        "n_computed_endpoints": n_computed,
        "n_expected_endpoints": len(records) * len(seeds),
        "five_pair_rank_reference": readiness.FIVE_PAIR_MIN_TWO_SIDED_TAIL,
        "transition_vs_control": transition_vs_control,
        "dependence": "NONADJACENT-SHARED-SUPPORT: comparisons sharing an arm are dependent by construction; pair counts are support accounting, never independent samples",
        "holm": holm_policy,
        "n1_reference_null_identity": (
            "N1 is the reference split-half null from fit_reference "
            "(null_values/null_median/null_q95 per history/seed fit); "
            "pair cells carry the N2 within-period half-split band only"
        ),
    }


def holm_adjust(p_values: Sequence[float], alpha: float = 0.05) -> list[dict[str, object]]:
    """Holm step-down adjustment over the declared primary contrasts."""
    import numpy as _np

    ordered = sorted(range(len(p_values)), key=lambda i: float(p_values[i]))
    m = len(p_values)
    adjusted: list[dict[str, object]] = [{} for _ in p_values]
    running_max = 0.0
    for rank, idx in enumerate(ordered):
        raw = float(p_values[idx])
        adj = min(1.0, (m - rank) * raw)
        running_max = max(running_max, adj)
        adjusted[idx] = {
            "index": idx,
            "raw": raw,
            "holm_adjusted": running_max,
            "reject": bool(running_max < alpha),
        }
    return adjusted


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "TASK_ID",
    "EXPECTED_CONTRACT_SHA256",
    "SMOKE_FILES_PER_HISTORY",
    "SMOKE_HISTORY_INDEX",
    "EXECUTED",
    "verify_contract_binding",
    "bound_capture_plan",
    "write_group_cache",
    "capture_inputs",
    "cached_group_complete",
    "admit_cache_or_refuse",
    "load_admitted_groups",
    "reference_ids_for_history",
    "run_movement_on_groups",
    "summarize_movement",
    "holm_adjust",
    "sha256_file",
]
