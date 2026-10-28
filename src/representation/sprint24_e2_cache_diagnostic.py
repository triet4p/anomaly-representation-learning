"""Sprint 24 Task 7-M12 E2 sample-size and abnormal-period movement diagnostic.

Cache-only, read-only exploratory audit of the already admitted S24-7-M11
embeddings (24,840 vectors, 18 ``(history, seed)`` groups). The module never
loads a model, never runs a forward pass, never refits an encoder or an
anomaly bank, never regenerates the corpus and never rewrites M11 products.

The frozen design is recorded in
``docs/sprint24-e2-cache-diagnostic-specification-a01.json`` (artifact
``S24-E2-CACHE-DIAGNOSTIC-SPEC-A01``), including a disclosed amendment block:

- A: matched-size E2 — for each history x horizon cell the per-arm size is
  ``s = min over the cell's five retained comparisons of min(n_baseline,
  n_contrast)``, and every comparison of the cell is evaluated at exactly
  ``s`` files per arm through repeated deterministic subsampling without
  replacement within a draw. The four transitions are published separately
  against their low-burden control, never only as an aggregate.
- B: baseline variability — the observed cross-period baseline control at
  the A size, plus a size-matched decomposition (cross-period vs
  within-period disjoint random halves) that isolates resampling
  variability. A within-period split never establishes temporal stability.
- C: robustness — the identical support/sampling design at K in {4, 8, 16}
  and all three encoder seeds; prototypes are fitted once per
  (history, seed) on the causal reference pool only.
- D: post-hoc diagnostics — realized anomaly family, severity,
  within-file anomalous fraction/duration and robot/program context joined
  by immutable file identity, with exact denominators, sparse/one-sided/
  confounded strata and the measured non-additivity of occupancy TV.

All equal-``n`` quantities are computed in EXACT OCCUPANCY QUANTA. With ``s``
rows per arm every occupancy entry is an integer divided by ``s``, so the
total-variation distance is ``m / s`` for an integer ``m`` (the half-sum of
absolute count differences, always integral because the count differences sum
to zero). The audit therefore stores integer numerators and derives every
median, quantile and direction statistic from integers, so a mathematically
equal pair is exactly zero and never a floating-point tie.

Abstention is explicit and complete: every planned cell either resolves or is
recorded with a closed reason code and stays in the denominator. No p-value,
success gate, performance claim or causal attribution is produced.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from representation import sprint24_controlled_inference_execution as m11
from representation import sprint24_controlled_inference_readiness as readiness
from representation.sprint24_context_audit import sha256_object

#: Task identifier for every M12 record.
TASK_ID = "S24-7-M12"

#: Frozen pre-outcome analysis specification.
SPEC_ID = "S24-E2-CACHE-DIAGNOSTIC-SPEC-A01"
SPEC_PATH = "docs/sprint24-e2-cache-diagnostic-specification-a01.json"

#: Byte digest of the accepted M10 prospective-contract envelope.
CONTRACT_ENVELOPE_SHA256 = "34031e8c1aca744258c2049659c54cff8417b98f1a889ebde52e48d31c9faa01"

#: Repository root, used to resolve the frozen specification path.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Monte Carlo repeats per comparison (resampling variability, never evidence).
REPEATS = 1000

#: Inherited cluster counts (K=8 primary reporting level, 4/16 sensitivity).
CLUSTER_COUNTS: tuple[int, ...] = (4, 8, 16)
INHERITED_PRIMARY_K = 8

#: Inherited period floor; also the declared B2 split floor (see the specification).
MIN_CELL_SIZE = 30
MIN_SPLIT_ARM_FILES = MIN_CELL_SIZE

#: Sampling designs evaluated for every comparison.
DESIGN_FULL_ARM = "FULL-ARM-UNSUBSAMPLED"
DESIGN_SRS = "SRS-UNSTRATIFIED"
DESIGN_STRATIFIED = "STRATIFIED-ROBOT-PROGRAM"
DESIGNS: tuple[str, ...] = (DESIGN_FULL_ARM, DESIGN_SRS, DESIGN_STRATIFIED)

#: Cell and record resolutions.
COMPUTED = "COMPUTED"
ABSTAIN_SHORTFALL = "ABSTAIN-SHORTFALL-SIZE"
ABSTAIN_NO_COMMON_STRATA = "ABSTAIN-NO-COMMON-STRATA"
ABSTAIN_EMPTY = "ABSTAIN-EMPTY"
ABSTAIN_REASONS: tuple[str, ...] = (
    ABSTAIN_SHORTFALL,
    ABSTAIN_NO_COMMON_STRATA,
    ABSTAIN_EMPTY,
)

#: Diagnostic bin edges (frozen before any M12 outcome).
SEVERITY_BINS: tuple[tuple[float, float], ...] = (
    (0.25, 0.45),
    (0.45, 0.60),
    (0.60, 0.75),
    (0.75, 0.90),
)
FRACTION_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.05),
    (0.05, 0.15),
    (0.15, 0.35),
    (0.35, 1.01),
)
ORDINARY = "ordinary"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- lineage and admission -----------------------------------------------------


def verify_lineage(
    corpus: Path,
    audit_path: Path,
    contract_path: Path,
    g2_preflight_path: Path | None = None,
) -> dict[str, Any]:
    """Verify corpus, audit, preflight, freeze, contract binding and custody.

    Every check runs before a single statistic is computed. Any digest drift,
    canonical-freeze mismatch, contract-term substitution or custody forgery
    raises ``REFUSED-*``; there is no recapture fallback.
    """
    manifest, freeze, preflight = readiness.load_inputs(corpus, g2_preflight_path)
    freeze_ok, freeze_detail = readiness.verify_canonical_freeze(freeze)
    _require(freeze_ok, f"REFUSED-DIGEST-DRIFT: {freeze_detail}")
    manifest_ok, manifest_detail = readiness.verify_pinned_artifact(
        corpus / "corpus-manifest.json", "corpus manifest"
    )
    _require(manifest_ok, f"REFUSED-DIGEST-DRIFT: {manifest_detail}")
    audit_ok, audit_detail = readiness.verify_pinned_artifact(audit_path, "corpus audit")
    _require(audit_ok, f"REFUSED-DIGEST-DRIFT: {audit_detail}")
    resolved_preflight = (
        g2_preflight_path
        if g2_preflight_path is not None
        else (
            REPO_ROOT
            / "artifacts"
            / "sprint-24"
            / "evidence"
            / "s24-7-m4-a01"
            / "g2-accepted"
            / "preflight.json"
        )
    )
    preflight_ok, preflight_detail = readiness.verify_pinned_artifact(
        resolved_preflight, "G2 preflight", allowed={resolved_preflight}
    )
    _require(preflight_ok, f"REFUSED-DIGEST-DRIFT: {preflight_detail}")

    custody = readiness.validate_custody_inputs(preflight)
    envelope_bytes = contract_path.read_bytes()
    envelope_sha = hashlib.sha256(envelope_bytes).hexdigest()
    _require(
        envelope_sha == CONTRACT_ENVELOPE_SHA256,
        f"REFUSED-IDENTITY: contract envelope digest drift: {envelope_sha}",
    )
    envelope = json.loads(envelope_bytes.decode("utf-8"))
    _require(
        isinstance(envelope, dict) and isinstance(envelope.get("contract"), dict),
        "REFUSED-IDENTITY: prospective contract envelope is malformed",
    )
    contract = envelope["contract"]
    audit_record = json.loads(audit_path.read_text(encoding="utf-8"))
    bound = m11.verify_contract_binding(
        contract, m11.EXPECTED_CONTRACT_SHA256, manifest, audit_record, custody
    )
    identity_plan = bound["plan"]
    _require(isinstance(identity_plan, dict), "REFUSED-IDENTITY: identity plan is malformed")
    groups = identity_plan.get("groups")
    return {
        "corpus": str(corpus),
        "audit_path": str(audit_path),
        "contract_path": str(contract_path),
        "g2_preflight_path": str(resolved_preflight),
        "freeze_check": freeze_detail,
        "manifest_check": manifest_detail,
        "audit_check": audit_detail,
        "preflight_check": preflight_detail,
        "contract_envelope_sha256": envelope_sha,
        "contract_canonical_sha256": m11.EXPECTED_CONTRACT_SHA256,
        "contract_binding_verified_pre_execution": True,
        "identity_plan_summary": {
            "n_groups": len(groups) if isinstance(groups, Mapping) else 0,
            "n_histories": identity_plan.get("n_histories"),
            "n_seeds": identity_plan.get("n_seeds"),
        },
        "manifest": manifest,
        "audit_record": audit_record,
        "contract": contract,
        "custody": custody,
        "identity_plan": identity_plan,
    }


def load_cache(
    cache_root: Path,
    manifest: Mapping[str, object],
    custody: Mapping[str, object],
    identity_plan: Mapping[str, object],
) -> dict[str, Any]:
    """Admit the retrieved cache through the accepted public path and load it.

    The support universe is the manifest-derived full file-ID list per
    history, so admission requires every one of the 24,840 required rows at
    the exact frozen group identity. A partial, foreign-provenance or
    identity-drifted copy refuses here, before any statistic.
    """
    _require(cache_root.is_dir(), f"REFUSED-MISSING-CACHE: cache root absent: {cache_root}")
    capture = m11.bound_capture_plan(manifest, custody)
    universe_raw = capture["file_ids"]
    _require(isinstance(universe_raw, Mapping), "REFUSED-IDENTITY: capture universe is malformed")
    universe = {
        str(history_id): [str(fid) for fid in ids]  # type: ignore[union-attr]
        for history_id, ids in universe_raw.items()
    }
    m11.set_support_universe(universe)
    admission = m11.admit_cache_or_refuse(cache_root, identity_plan)
    groups = m11.load_admitted_groups(cache_root, identity_plan)
    by_history: dict[str, list[str]] = {}
    for key in sorted(groups):
        history_id = str(groups[key].history_id)
        if history_id not in by_history:
            by_history[history_id] = groups[key].file_ids
        else:
            _require(
                groups[key].file_ids == by_history[history_id],
                f"REFUSED-IDENTITY: group {key} carries a different file-ID order within its history",
            )
    digests = {
        path.name: sha256_file(path) for path in sorted(cache_root.iterdir()) if path.is_file()
    }
    return {
        "cache_root": str(cache_root),
        "universe": universe,
        "admission": {
            "status": str(admission.get("status")),
            "required_rows": admission.get("required_rows"),
            "available_rows": admission.get("available_rows"),
            "missing_rows": admission.get("missing_rows"),
            "n_groups": len(groups),
        },
        "groups": groups,
        "n_groups": len(groups),
        "n_rows": int(sum(group.matrix.shape[0] for group in groups.values())),
        "n_cache_files": len(digests),
        "cache_file_digests": digests,
        "cache_file_digest_digest": sha256_object(digests),
    }


# --- reference fits and prototype labels ---------------------------------------


def fit_references(
    groups: Mapping[str, Any],
    manifest: Mapping[str, object],
    *,
    seed: int = readiness.NULL_SEED,
    resamples: int = 200,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit one reference per ``(history, seed)`` on its causal pool only.

    Identical construction to the accepted M11 movement fit (same pool, same
    KMeans settings, same bandwidth rule), so the full-arm recomputation can
    be compared with the published M11 cell values. Evaluation-period rows
    and every label/family/severity/context field stay outside this function.
    """
    from representation import sprint24_weekly_monthly_endpoints as endpoints

    fits: dict[str, Any] = {}
    custody: dict[str, Any] = {}
    for history_id in sorted({str(key.split("|")[0]) for key in groups}):
        reference_ids = m11.reference_ids_for_history(manifest, history_id)
        reference_digest = sha256_object(reference_ids)
        for model_seed in readiness.FROZEN_ENCODER_SEEDS:
            key = f"{history_id}|{model_seed}"
            started = time.time()
            fit = endpoints.fit_reference(groups[key], reference_ids, resamples=resamples, seed=seed)
            fits[key] = fit
            custody[key] = {
                "history_id": history_id,
                "model_seed": int(model_seed),
                "n_reference": int(fit.n_reference),
                "reference_file_ids_sha256": reference_digest,
                "sigma": float(fit.sigma),
                "null_median": float(np.median(fit.null_values)),
                "null_q95": float(fit.null_q95),
                "kmeans": {
                    str(k): {
                        "random_state": int(seed),
                        "n_init": 10,
                        "inertia": float(fit.kmeans[k].inertia_),
                        "n_iter": int(fit.kmeans[k].n_iter_),
                    }
                    for k in CLUSTER_COUNTS
                },
                "fit_wall_s": time.time() - started,
            }
    return fits, custody


def prototype_labels(
    groups: Mapping[str, Any], fits: Mapping[str, Any]
) -> dict[str, dict[int, np.ndarray]]:
    """Assign every admitted row of every group to its reference prototypes.

    Labels are prototype indices produced by the reference-only KMeans; no
    corpus label, family, severity or context field participates.
    """
    labels: dict[str, dict[int, np.ndarray]] = {}
    for key, group in groups.items():
        labels[key] = {
            k: np.asarray(fits[key].kmeans[k].predict(group.matrix), dtype=np.int64)
            for k in CLUSTER_COUNTS
        }
    return labels


# --- support and comparisons ---------------------------------------------------


def manifest_index(manifest: Mapping[str, object]) -> dict[str, dict[str, Any]]:
    """Return the manifest file table keyed by immutable ``file_id``."""
    files = manifest.get("files")
    _require(isinstance(files, list), "REFUSED-IDENTITY: corpus manifest has no file table")
    index: dict[str, dict[str, Any]] = {}
    for entry in files:  # type: ignore[union-attr]
        if isinstance(entry, Mapping) and isinstance(entry.get("file_id"), str):
            index[str(entry["file_id"])] = dict(entry)
    _require(bool(index), "REFUSED-IDENTITY: corpus manifest file table is empty")
    return index


def stratum_id(entry: Mapping[str, Any]) -> str:
    """Return the label-blind robot|program context stratum of one file."""
    robot = int(entry["predicted_robot_idx"])
    program = int(entry["predicted_program_idx"])
    return f"R{robot + 1:02d}|P{100 + program * 10}"


def manifest_exposure_maps(
    manifest: Mapping[str, object],
) -> tuple[dict[str, float], dict[str, float], int]:
    """Return the label-blind end-time/duration maps used by the accepted census.

    Reproduces the accepted M10 construction exactly: end-times come from the
    manifest ``end_time_rel_s`` field and durations from the NPZ ``x.npy`` NPY
    header only (shape bytes; no array element and no label is read). The
    accepted convention treats the header's sample count as the file duration
    in seconds, which is what :func:`readiness.retained_exposure` consumes; the
    reconstruction is cross-checked against the published census ranges.
    """
    files = manifest.get("files")
    _require(isinstance(files, list), "REFUSED-IDENTITY: corpus manifest has no file table")
    manifest_path = manifest.get("__manifest_path__")
    corpus_root = Path(str(manifest_path)).parent if isinstance(manifest_path, str) else None
    end_time_of: dict[str, float] = {}
    duration_of: dict[str, float] = {}
    header_reads = 0
    for entry in files:  # type: ignore[union-attr]
        if not isinstance(entry, Mapping):
            continue
        fid = entry.get("file_id")
        if not isinstance(fid, str):
            continue
        end = entry.get("end_time_rel_s")
        if isinstance(end, (int, float)):
            end_time_of[fid] = float(end)
        rel = entry.get("relpath")
        if corpus_root is not None and isinstance(rel, str):
            try:
                shape = readiness.read_npy_shape_header(corpus_root / rel)
            except (OSError, KeyError, ValueError, IndexError):
                continue
            duration_of[fid] = float(int(shape[1]))
            header_reads += 1
    return end_time_of, duration_of, header_reads


def context_mix_tv(
    baseline_ids: Sequence[str],
    contrast_ids: Sequence[str],
    index: Mapping[str, Mapping[str, Any]],
) -> float | None:
    """Robot/program mix total-variation distance between two retained arms."""

    def counts(ids: Sequence[str]) -> dict[str, int] | None:
        out: dict[str, int] = {}
        for fid in ids:
            entry = index.get(fid)
            if entry is None:
                return None
            name = stratum_id(entry)
            out[name] = out.get(name, 0) + 1
        return out

    baseline = counts(baseline_ids)
    contrast = counts(contrast_ids)
    if baseline is None or contrast is None:
        return None
    return mix_tv(baseline, contrast)


def comparison_rows(audit_record: Mapping[str, object]) -> list[dict[str, Any]]:
    """Return the retained comparisons in a deterministic order.

    Every row keeps its retained per-arm file identities, comparison type,
    disposition, burden and exposure facts from the M9 audit; nothing is
    re-balanced, dropped or re-selected.
    """
    pair_rows = audit_record.get("pair_rows")
    _require(isinstance(pair_rows, list) and bool(pair_rows), "REFUSED-IDENTITY: audit has no pair rows")
    rows: list[dict[str, Any]] = []
    for row in pair_rows:  # type: ignore[union-attr]
        if not isinstance(row, Mapping):
            continue
        previous = [str(fid) for fid in row.get("retained_baseline_file_identities", [])]  # type: ignore[union-attr]
        current = [str(fid) for fid in row.get("retained_contrast_file_identities", [])]  # type: ignore[union-attr]
        rows.append(
            {
                "history_id": str(row["history_id"]),
                "horizon": str(row["horizon"]),
                "mode": str(row.get("mode")),
                "comparison_identity": str(row["comparison_identity"]),
                "comparison_type": str(row["comparison_type"]),
                "disposition": str(row.get("disposition")),
                "baseline_period_identities": list(row.get("baseline_period_identities", [])),  # type: ignore[arg-type]
                "contrast_period_identities": list(row.get("contrast_period_identities", [])),  # type: ignore[arg-type]
                "n_baseline": len(previous),
                "n_contrast": len(current),
                "previous_ids": previous,
                "current_ids": current,
                "burden_baseline_after": row.get("burden_baseline_after"),
                "burden_contrast_after": row.get("burden_contrast_after"),
                "burden_contrast_gap_after": row.get("burden_contrast_gap_after"),
                "exposure_ratio_after": row.get("exposure_ratio_after"),
            }
        )
    rows.sort(key=lambda item: (item["history_id"], item["horizon"], item["comparison_identity"]))
    _require(bool(rows), "REFUSED-IDENTITY: audit retained no comparison")
    return rows


def cell_key(history_id: str, horizon: str) -> str:
    return f"{history_id}|{horizon}"


def cell_sizes(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the matched per-arm size and the control identity of every cell."""
    cells: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = cell_key(str(row["history_id"]), str(row["horizon"]))
        entry = cells.setdefault(
            key,
            {
                "history_id": str(row["history_id"]),
                "horizon": str(row["horizon"]),
                "comparisons": [],
                "control_identity": None,
            },
        )
        entry["comparisons"].append(str(row["comparison_identity"]))
        if str(row["comparison_type"]) == "BASELINE-VS-BASELINE":
            entry["control_identity"] = str(row["comparison_identity"])
    for key, entry in cells.items():
        members = [
            row for row in rows if cell_key(str(row["history_id"]), str(row["horizon"])) == key
        ]
        size = min(min(int(row["n_baseline"]), int(row["n_contrast"])) for row in members)
        entry["size"] = int(size)
        entry["status"] = COMPUTED if size >= MIN_CELL_SIZE else ABSTAIN_SHORTFALL
        entry["n_comparisons"] = len(members)
        entry["n_transitions"] = len(
            [row for row in members if str(row["comparison_type"]) == "BASELINE-VS-ELEVATED"]
        )
        entry["retained_arm_sizes"] = sorted(
            {int(row["n_baseline"]) for row in members} | {int(row["n_contrast"]) for row in members}
        )
        _require(
            entry["control_identity"] is not None,
            f"REFUSED-IDENTITY: cell {key} has no retained low-burden control",
        )
    return cells


# --- sampling design -----------------------------------------------------------


def stratum_codes(
    ids: Sequence[str], index: Mapping[str, Mapping[str, Any]], lookup: dict[str, int]
) -> np.ndarray:
    """Encode file IDs with a shared label-blind stratum code mapping."""
    codes = np.empty(len(ids), dtype=np.int64)
    for position, fid in enumerate(ids):
        entry = index.get(fid)
        _require(entry is not None, f"REFUSED-IDENTITY: retained file {fid} is not in the manifest")
        name = stratum_id(entry)
        if name not in lookup:
            lookup[name] = len(lookup)
        codes[position] = lookup[name]
    return codes


def allocation_with_caps(
    pooled: Mapping[int, int], caps: Mapping[int, int], total: int
) -> dict[int, int]:
    """Proportional allocation of ``total`` over common strata, capped per arm.

    Largest-remainder apportionment on the pooled two-arm counts, capped by
    the smaller arm's count in each stratum, then redistributed over the
    still-uncapped strata. Returns ``{}`` when the support cannot supply
    ``total`` at this composition, which makes the stratified design
    unavailable for that comparison. Deterministic and support-only.
    """
    keys = sorted(pooled)
    if not keys or total <= 0:
        return {}
    if sum(min(caps[key], pooled[key]) for key in keys) < total:
        return {}
    allocation = {key: 0 for key in keys}
    remaining = total
    active = list(keys)
    while remaining > 0 and active:
        weight = float(sum(pooled[key] for key in active))
        if weight <= 0:
            return {}
        ideal = {key: remaining * pooled[key] / weight for key in active}
        granted = False
        for key in active:
            room = caps[key] - allocation[key]
            take = min(int(math.floor(ideal[key])), room)
            if take > 0:
                allocation[key] += take
                remaining -= take
                granted = True
        if remaining <= 0:
            break
        if not granted:
            order = sorted(active, key=lambda key: (-(ideal[key] - math.floor(ideal[key])), key))
            for key in order:
                if allocation[key] < caps[key]:
                    allocation[key] += 1
                    remaining -= 1
                    granted = True
                    break
            if not granted:
                return {}
        active = [key for key in active if allocation[key] < caps[key]]
    return allocation if remaining == 0 else {}


def _draw_seed(parts: Sequence[object]) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:16], 16)


def stratified_draw(
    positions: np.ndarray,
    codes: np.ndarray,
    allocation: Mapping[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``allocation[code]`` rows per stratum, without replacement."""
    permutation = rng.permutation(positions.size)
    shuffled_codes = codes[permutation]
    order = np.argsort(shuffled_codes, kind="stable")
    ordered_positions = positions[permutation][order]
    ordered_codes = shuffled_codes[order]
    unique = np.unique(ordered_codes)
    stops = np.searchsorted(ordered_codes, unique, side="right")
    starts = np.concatenate(([0], stops[:-1]))
    picked = [
        ordered_positions[starts[index] : starts[index] + allocation[int(code)]]
        for index, code in enumerate(unique)
        if allocation.get(int(code), 0) > 0
    ]
    _require(bool(picked), "REFUSED-IDENTITY: stratified draw selected no row")
    return np.sort(np.concatenate(picked))


def design_plan(
    *,
    group: Any,
    row: Mapping[str, Any],
    size: int,
    index: Mapping[str, Mapping[str, Any]],
    repeats: int,
) -> dict[str, Any]:
    """Build both matched-size draw designs for one comparison.

    Both designs draw exactly ``size`` rows per arm without replacement within
    a draw and reuse one plan across K and seeds. The stratified design uses
    the label-blind robot|program stratum; the SRS design is the unstratified
    design that isolates the pure sample-size effect. When no feasible
    proportional allocation exists the stratified arrays are ``None`` and
    ``stratified_available`` is ``False``; the caller records that as an
    explicit abstention for the stratified design while the SRS design still
    resolves.
    """
    baseline_ids = list(row["previous_ids"])
    contrast_ids = list(row["current_ids"])
    lookup: dict[str, int] = {}
    baseline_codes = stratum_codes(baseline_ids, index, lookup)
    contrast_codes = stratum_codes(contrast_ids, index, lookup)
    baseline_positions = np.array([group.index[fid] for fid in baseline_ids], dtype=np.int64)
    contrast_positions = np.array([group.index[fid] for fid in contrast_ids], dtype=np.int64)
    baseline_counts = {
        code: int(np.count_nonzero(baseline_codes == code)) for code in np.unique(baseline_codes)
    }
    contrast_counts = {
        code: int(np.count_nonzero(contrast_codes == code)) for code in np.unique(contrast_codes)
    }
    common = sorted(set(baseline_counts) & set(contrast_counts))
    label_of = {code: name for name, code in lookup.items()}
    plan: dict[str, Any] = {
        "comparison_identity": str(row["comparison_identity"]),
        "n_baseline": len(baseline_ids),
        "n_contrast": len(contrast_ids),
        "size": int(size),
        "repeats": int(repeats),
        "n_baseline_strata": len(baseline_counts),
        "n_contrast_strata": len(contrast_counts),
        "n_common_strata": len(common),
        "a_stratified_baseline": None,
        "a_stratified_contrast": None,
        "a_srs_baseline": None,
        "a_srs_contrast": None,
        "allocation": None,
        "unavailable_reason": None,
        "stratified_available": False,
    }
    allocation: dict[int, int] = {}
    if common:
        caps = {code: min(baseline_counts[code], contrast_counts[code]) for code in common}
        pooled = {code: baseline_counts[code] + contrast_counts[code] for code in common}
        allocation = allocation_with_caps(pooled, caps, size)
    if allocation:
        plan["stratified_available"] = True
        plan["allocation"] = {label_of[code]: int(allocation[code]) for code in sorted(allocation)}
        stratified_baseline = np.empty((repeats, size), dtype=np.int32)
        stratified_contrast = np.empty((repeats, size), dtype=np.int32)
        for repeat in range(repeats):
            parts = [TASK_ID, row["history_id"], row["horizon"], row["comparison_identity"], repeat]
            stratified_baseline[repeat] = stratified_draw(
                baseline_positions,
                baseline_codes,
                allocation,
                np.random.default_rng(_draw_seed(parts + ["stratified", "baseline"])),
            )
            stratified_contrast[repeat] = stratified_draw(
                contrast_positions,
                contrast_codes,
                allocation,
                np.random.default_rng(_draw_seed(parts + ["stratified", "contrast"])),
            )
        plan["a_stratified_baseline"] = stratified_baseline
        plan["a_stratified_contrast"] = stratified_contrast
    else:
        plan["unavailable_reason"] = (
            "no common robot/program stratum in both arms"
            if not common
            else "the retained support cannot supply the matched size at any proportional allocation"
        )
    plan["sampling_mode"] = (
        DESIGN_STRATIFIED if plan["stratified_available"] else DESIGN_SRS
    )
    srs_baseline = np.empty((repeats, size), dtype=np.int32)
    srs_contrast = np.empty((repeats, size), dtype=np.int32)
    for repeat in range(repeats):
        parts = [TASK_ID, row["history_id"], row["horizon"], row["comparison_identity"], repeat]
        srs_baseline[repeat] = np.sort(
            np.random.default_rng(_draw_seed(parts + ["srs", "baseline"])).choice(
                baseline_positions, size=size, replace=False
            )
        )
        srs_contrast[repeat] = np.sort(
            np.random.default_rng(_draw_seed(parts + ["srs", "contrast"])).choice(
                contrast_positions, size=size, replace=False
            )
        )
    plan["a_srs_baseline"] = srs_baseline
    plan["a_srs_contrast"] = srs_contrast
    return plan


# --- E2 in exact occupancy quanta ----------------------------------------------


def prototype_counts(codes: np.ndarray, k: int) -> np.ndarray:
    """Prototype counts per draw for a ``(repeats, size)`` prototype-label matrix."""
    repeats, _size = codes.shape
    offsets = np.arange(repeats, dtype=np.int64)[:, None] * k
    flat = (codes + offsets).reshape(-1)
    return np.bincount(flat, minlength=repeats * k).reshape(repeats, k)


def e2_quanta(
    labels: np.ndarray, baseline: np.ndarray, contrast: np.ndarray, k: int
) -> np.ndarray:
    """Exact E2 numerator per draw for two equal-size arms.

    With ``s`` rows per arm every occupancy entry is ``count / s``, so
    ``E2 = sum_g |c_g - d_g| / (2 s)``. The count differences sum to zero, so
    their absolute sum is even and the half-sum is an integer: the returned
    ``int64`` array is the numerator ``m`` with ``E2 = m / s`` exactly. No
    floating-point comparison is ever needed to test a direction.
    """
    size = baseline.shape[1]
    _require(contrast.shape[1] == size, "REFUSED-IDENTITY: arms are not size-matched")
    counts_b = prototype_counts(labels[baseline], k)
    counts_c = prototype_counts(labels[contrast], k)
    absolute_total = np.abs(counts_b - counts_c).sum(axis=1)
    _require(
        bool(np.all(absolute_total % 2 == 0)),
        "REFUSED-INTERNAL: occupancy count differences do not sum to an even total",
    )
    return (absolute_total // 2).astype(np.int64)


def e2_repeats(labels: np.ndarray, baseline: np.ndarray, contrast: np.ndarray, k: int) -> np.ndarray:
    """E2 per draw as ``quanta / matched size`` (float view of the exact numerator)."""
    size = baseline.shape[1]
    return e2_quanta(labels, baseline, contrast, k) / float(size)


def occupancy(labels: np.ndarray, k: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    return counts / counts.sum()


def tv(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())


def full_arm_e2(
    labels: np.ndarray,
    baseline_ids: Sequence[str],
    contrast_ids: Sequence[str],
    group: Any,
    k: int,
) -> float:
    """Unsubsampled E2 on the whole retained arms (their sizes may differ)."""
    baseline_rows = labels[[group.index[fid] for fid in baseline_ids]]
    contrast_rows = labels[[group.index[fid] for fid in contrast_ids]]
    return tv(occupancy(baseline_rows, k), occupancy(contrast_rows, k))


# --- summaries -----------------------------------------------------------------


def summary_of_quanta(quanta: np.ndarray, size: int) -> dict[str, Any]:
    """Summarise an exact integer-quanta vector; directions are exact.

    ``values * size`` converts back to E2. Quantiles of the repeat vector are
    resampling variability of one finite support, never confidence limits.
    """
    values = np.asarray(quanta, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return {"n": 0, "status": ABSTAIN_EMPTY}
    return {
        "n": int(values.size),
        "status": COMPUTED,
        "size": int(size),
        "median": float(np.median(values)) / size,
        "mean": float(values.mean()) / size,
        "q05": float(np.quantile(values, 0.05)) / size,
        "q95": float(np.quantile(values, 0.95)) / size,
        "min": float(values.min()) / size,
        "max": float(values.max()) / size,
        "median_quanta": float(np.median(values)),
        "q05_quanta": float(np.quantile(values, 0.05)),
        "q95_quanta": float(np.quantile(values, 0.95)),
        "min_quanta": int(values.min()),
        "max_quanta": int(values.max()),
        "fraction_strictly_positive": float(np.mean(values > 0)),
        "fraction_zero": float(np.mean(values == 0)),
        "fraction_strictly_negative": float(np.mean(values < 0)),
    }


def abstain_summary(reason: str) -> dict[str, Any]:
    return {"n": 0, "status": reason}


def design_summary(
    quanta: np.ndarray | None, size: int, reason: str
) -> dict[str, Any]:
    """Per-design summary, or an explicit abstention when the design is absent."""
    if quanta is None:
        return abstain_summary(reason)
    return summary_of_quanta(quanta, size)


def delta_summary(
    transition: np.ndarray | None,
    control: np.ndarray | None,
    size: int,
    reason: str,
) -> dict[str, Any]:
    """Transition-minus-control summary from exact quanta, or an abstention.

    The difference of two integer quanta is an integer, so a mathematically
    equal pair contributes exactly zero and can never be counted positive.
    """
    if transition is None or control is None:
        return abstain_summary(reason)
    _require(
        transition.shape == control.shape,
        "REFUSED-IDENTITY: transition and control repeats are not aligned",
    )
    return summary_of_quanta(
        np.asarray(transition, dtype=np.int64) - np.asarray(control, dtype=np.int64), size
    )


def seed_median(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.median(clean)) if clean else None


# --- diagnostics --------------------------------------------------------------


def bin_label(edges: tuple[float, float]) -> str:
    return f"[{edges[0]:.2f},{edges[1]:.2f})"


def arm_composition(ids: Sequence[str], index: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Post-hoc composition of one retained arm, joined by immutable file identity."""
    family_counts: dict[str, int] = {}
    severity_counts = {bin_label(edges): 0 for edges in SEVERITY_BINS}
    fraction_counts = {bin_label(edges): 0 for edges in FRACTION_BINS}
    severity_families: dict[str, set[str]] = {bin_label(edges): set() for edges in SEVERITY_BINS}
    fraction_families: dict[str, set[str]] = {bin_label(edges): set() for edges in FRACTION_BINS}
    context_counts: dict[str, int] = {}
    severities: list[float] = []
    fractions: list[float] = []
    durations: list[float] = []
    family_of: dict[str, list[str]] = {}
    for fid in ids:
        entry = index[fid]
        annotation = entry.get("annotation")
        if not isinstance(annotation, Mapping):
            annotation = {}
        if str(entry.get("file_label")) == "normal":
            family = ORDINARY
            severity: float | None = None
            fraction: float | None = None
            duration: float | None = None
        else:
            family = str(annotation.get("family") or "unknown")
            raw_severity = annotation.get("severity")
            raw_fraction = annotation.get("abnormal_step_fraction")
            raw_duration = annotation.get("duration_steps")
            severity = float(raw_severity) if isinstance(raw_severity, (int, float)) else None
            fraction = float(raw_fraction) if isinstance(raw_fraction, (int, float)) else None
            duration = float(raw_duration) if isinstance(raw_duration, (int, float)) else None
        family_counts[family] = family_counts.get(family, 0) + 1
        family_of.setdefault(family, []).append(fid)
        context = stratum_id(entry)
        context_counts[context] = context_counts.get(context, 0) + 1
        if severity is not None:
            severities.append(severity)
            for edges in SEVERITY_BINS:
                if edges[0] <= severity < edges[1]:
                    key = bin_label(edges)
                    severity_counts[key] += 1
                    severity_families[key].add(family)
                    break
        if fraction is not None:
            fractions.append(fraction)
            for edges in FRACTION_BINS:
                if edges[0] <= fraction < edges[1]:
                    key = bin_label(edges)
                    fraction_counts[key] += 1
                    fraction_families[key].add(family)
                    break
        if duration is not None:
            durations.append(duration)
    n = len(ids)
    ordinary = family_counts.get(ORDINARY, 0)
    confounded: list[str] = []
    for partition, mapping in (
        ("severity", severity_families),
        ("anomalous_fraction", fraction_families),
    ):
        for name, families in sorted(mapping.items()):
            if len(families) == 1:
                confounded.append(f"{partition}:{name}:single-family:{next(iter(families))}")
    return {
        "n_files": n,
        "n_ordinary": ordinary,
        "n_abnormal": n - ordinary,
        "file_ids_sha256": sha256_object(sorted(ids)),
        "waveform_sha256_digest": sha256_object(
            sorted(str(index[fid].get("waveform_sha256")) for fid in ids)
        ),
        "family_counts": family_counts,
        "family_fractions": {name: count / n for name, count in sorted(family_counts.items())},
        "severity_bin_counts": severity_counts,
        "anomalous_fraction_bin_counts": fraction_counts,
        "severity_bin_families": {
            name: sorted(families) for name, families in sorted(severity_families.items())
        },
        "anomalous_fraction_bin_families": {
            name: sorted(families) for name, families in sorted(fraction_families.items())
        },
        "confounded_bins": sorted(confounded),
        "severity_median": float(np.median(severities)) if severities else None,
        "anomalous_fraction_median": float(np.median(fractions)) if fractions else None,
        "duration_steps_median": float(np.median(durations)) if durations else None,
        "duration_steps_min": float(min(durations)) if durations else None,
        "duration_steps_max": float(max(durations)) if durations else None,
        "context_counts": context_counts,
        "n_context_strata": len(context_counts),
        "family_of": family_of,
    }


def mix_tv(first: Mapping[str, int], second: Mapping[str, int]) -> float | None:
    """Categorical total-variation distance, ``None`` when either side is empty.

    ``None`` is the honest "undefined" marker: a partition with no member in
    one arm (for example a fully ordinary baseline arm in the severity bins)
    has no defined composition distance and must never be written as zero.
    """
    names = sorted(set(first) | set(second))
    total_first = float(sum(first.values()))
    total_second = float(sum(second.values()))
    if total_first <= 0 or total_second <= 0:
        return None
    return float(
        0.5
        * sum(abs(first.get(name, 0) / total_first - second.get(name, 0) / total_second) for name in names)
    )


def composition_flags(
    baseline: Mapping[str, Any], contrast: Mapping[str, Any]
) -> dict[str, Any]:
    """Disclose sparse, one-sided and small-family strata instead of dropping them."""
    baseline_families = baseline["family_counts"]
    contrast_families = contrast["family_counts"]
    one_sided = sorted(
        name
        for name in set(baseline_families) | set(contrast_families)
        if (name in baseline_families) != (name in contrast_families)
    )
    sparse = sorted(
        {
            name
            for name, count in list(baseline_families.items()) + list(contrast_families.items())
            if count < 5
        }
    )
    small_family_detail = []
    confounded_detail = []
    for arm_name, composition in (("baseline", baseline), ("contrast", contrast)):
        for family, count in sorted(composition["family_counts"].items()):
            if family != ORDINARY and count < 5:
                small_family_detail.append(f"{arm_name}:{family}:n={count}")
        for entry in composition["confounded_bins"]:
            confounded_detail.append(f"{arm_name}:{entry}")
    return {
        "one_sided_families": one_sided,
        "sparse_strata": sparse,
        "small_family_detail": small_family_detail,
        "confounded_bins": confounded_detail,
        "n_families_both_arms": len(set(baseline_families) & set(contrast_families)),
        "n_families_union": len(set(baseline_families) | set(contrast_families)),
    }


def tv_nonadditivity(
    *,
    labels: np.ndarray,
    group: Any,
    row: Mapping[str, Any],
    family_of: Mapping[str, Sequence[str]],
    k: int,
) -> dict[str, Any]:
    """Measure the non-additivity of occupancy TV over contrast-arm families.

    ``TV_total = TV(p, q)`` for the full contrast occupancy ``q`` and
    ``TV_add = sum_f w_f TV(p, q_f)`` over realized-family subsets ``q_f``
    with ``w_f = n_f / n``. Convexity makes ``TV_add >= TV_total``; the gap is
    the measured non-additivity, never an additive partition of the total.
    """
    baseline_ids = list(row["previous_ids"])
    contrast_ids = list(row["current_ids"])
    p = occupancy(labels[[group.index[fid] for fid in baseline_ids]], k)
    q = occupancy(labels[[group.index[fid] for fid in contrast_ids]], k)
    total = tv(p, q)
    weighted = 0.0
    per_family: dict[str, dict[str, Any]] = {}
    n_contrast = len(contrast_ids)
    for family, ids in sorted(family_of.items()):
        subset = occupancy(labels[[group.index[fid] for fid in ids]], k)
        value = tv(p, subset)
        weight = len(ids) / n_contrast
        weighted += weight * value
        per_family[family] = {
            "n_files": len(ids),
            "weight": weight,
            "tv_vs_baseline": value,
            "weighted_tv": weight * value,
        }
    return {
        "K": int(k),
        "tv_total": total,
        "tv_additive_sum": weighted,
        "nonadditivity_gap": weighted - total,
        "n_families": len(family_of),
        "per_family": per_family,
    }


def sample_size_effect_record(
    designs: Mapping[str, Any], model_seed: int, k: int
) -> dict[str, Any]:
    """Full-arm versus matched-size effect, abstaining when a design is absent.

    Kept separate from the run loop so the missing-design path is exercised
    directly: a comparison whose stratified design is unavailable must yield
    an explicit abstention record, never a lookup error.
    """
    stratified = designs[DESIGN_STRATIFIED]
    srs = designs[DESIGN_SRS]
    full = designs[DESIGN_FULL_ARM]
    stratified_median = (
        stratified["per_seed"][str(model_seed)]["per_K"][str(k)]["median"]
        if stratified.get("status") == COMPUTED
        else None
    )
    srs_median = (
        srs["per_seed"][str(model_seed)]["per_K"][str(k)]["median"]
        if srs.get("status") == COMPUTED
        else None
    )
    full_value = (
        full["per_seed"][str(model_seed)][str(k)] if full.get("status") == COMPUTED else None
    )
    return {
        "status": COMPUTED
        if stratified_median is not None and srs_median is not None and full_value is not None
        else ABSTAIN_NO_COMMON_STRATA,
        "full_arm": full_value,
        "matched_stratified": stratified_median,
        "matched_srs": srs_median,
        "full_minus_matched_stratified": (
            full_value - stratified_median
            if (full_value is not None and stratified_median is not None)
            else None
        ),
        "full_minus_matched_srs": (
            full_value - srs_median
            if (full_value is not None and srs_median is not None)
            else None
        ),
        "matched_srs_minus_stratified": (
            srs_median - stratified_median
            if (srs_median is not None and stratified_median is not None)
            else None
        ),
    }


def seed_direction(values: Sequence[float]) -> dict[str, Any]:
    """Three-way direction of the encoder-seed replicates.

    A zero is neither positive nor negative, it is its own sign. ``signs`` is
    the sorted set of observed signs (-1, 0, +1) and ``agreement`` is true
    only when every seed is present and all of them share one non-zero sign;
    ``all_zero`` records the unanimous-zero case separately so that it is
    never counted as directional agreement.
    """
    signs = sorted({int(np.sign(float(value))) for value in values})
    return {
        "n": len(values),
        "signs": signs,
        "agreement": bool(
            len(values) == len(readiness.FROZEN_ENCODER_SEEDS)
            and len(signs) == 1
            and signs[0] != 0
        ),
        "all_zero": bool(bool(values) and signs == [0]),
    }


def matched_size_entry(
    *,
    row: Mapping[str, Any],
    cell: Mapping[str, Any],
    key: str,
    size: int,
    plan: Mapping[str, Any] | None,
    labels: Mapping[str, Mapping[int, np.ndarray]],
    groups: Mapping[str, Any],
    index: Mapping[str, Mapping[str, Any]],
    end_time_of: Mapping[str, float],
    duration_of: Mapping[str, float],
    stratified_store: dict[tuple[str, int, int], np.ndarray],
    srs_store: dict[tuple[str, int, int], np.ndarray],
    full_store: dict[tuple[str, int, int], float],
) -> dict[str, Any]:
    """Assemble one comparison's matched-size records, abstaining when needed.

    The whole per-comparison path lives here so an unsupported comparison —
    a cell below the period floor, or a comparison for which no matched-size
    sampling plan exists — is reachable and testable without a real corpus:
    it returns explicit abstention records for every design and never indexes
    an absent sampling array.
    """
    identity = str(row["comparison_identity"])
    history_id = str(row["history_id"])
    retained_exposure = readiness.retained_exposure(
        row["previous_ids"], row["current_ids"], end_time_of, duration_of
    )
    retained_context = context_mix_tv(row["previous_ids"], row["current_ids"], index)
    entry: dict[str, Any] = {
        "cell": key,
        "cell_status": cell["status"],
        "history_id": history_id,
        "horizon": str(row["horizon"]),
        "comparison_identity": identity,
        "comparison_type": str(row["comparison_type"]),
        "disposition": str(row["disposition"]),
        "size": size,
        "n_baseline_full": int(row["n_baseline"]),
        "n_contrast_full": int(row["n_contrast"]),
        "burden_baseline_after": row["burden_baseline_after"],
        "burden_contrast_after": row["burden_contrast_after"],
        "burden_contrast_gap_after": row["burden_contrast_gap_after"],
        "exposure_ratio_after_inherited_raw": row["exposure_ratio_after"],
        "exposure_ratio_after_inherited_raw_note": (
            "the M9 record's exposure_ratio_after is the RAW pre-balancing arm "
            "ratio (the inherited audit sets after = before); it is NOT the "
            "retained exposure and is reported only as the inherited limitation"
        ),
        "retained_exposure": retained_exposure,
        "retained_exposure_status": (
            COMPUTED if retained_exposure is not None else "ABSTAIN-METADATA"
        ),
        "retained_exposure_floor": readiness.EXPOSURE_RATIO_FLOOR,
        "retained_exposure_at_or_above_floor": (
            retained_exposure is not None
            and retained_exposure >= readiness.EXPOSURE_RATIO_FLOOR
        ),
        "retained_context_mix_tv": retained_context,
        "sampling_mode": None,
        "stratified_available": False,
        "designs": {},
    }
    if cell["status"] != COMPUTED or plan is None:
        reason = ABSTAIN_SHORTFALL if cell["status"] != COMPUTED else ABSTAIN_NO_COMMON_STRATA
        detail = (
            "cell below the inherited 30-file period floor"
            if reason == ABSTAIN_SHORTFALL
            else "no matched-size sampling plan is available for this comparison"
        )
        for design in DESIGNS:
            entry["designs"][design] = {"status": reason, "reason": detail}
        entry["sample_size_effect"] = abstain_summary(reason)
        return entry
    entry["sampling_mode"] = plan["sampling_mode"]
    entry["stratified_available"] = bool(plan["stratified_available"])
    for design, prefix in ((DESIGN_STRATIFIED, "a_stratified"), (DESIGN_SRS, "a_srs")):
        baseline_positions = plan[f"{prefix}_baseline"]
        contrast_positions = plan[f"{prefix}_contrast"]
        if baseline_positions is None or contrast_positions is None:
            entry["designs"][design] = {
                "status": ABSTAIN_NO_COMMON_STRATA,
                "reason": plan.get("unavailable_reason"),
            }
            continue
        per_seed: dict[str, Any] = {}
        for model_seed in readiness.FROZEN_ENCODER_SEEDS:
            per_k: dict[str, Any] = {}
            for k in CLUSTER_COUNTS:
                quanta = e2_quanta(
                    labels[f"{history_id}|{model_seed}"][k],
                    baseline_positions,
                    contrast_positions,
                    k,
                )
                per_k[str(k)] = summary_of_quanta(quanta, size)
                if design == DESIGN_STRATIFIED:
                    stratified_store[(identity, int(k), int(model_seed))] = quanta
                else:
                    srs_store[(identity, int(k), int(model_seed))] = quanta
            per_seed[str(model_seed)] = {"per_K": per_k}
        entry["designs"][design] = {"status": COMPUTED, "per_seed": per_seed}
    full_arm: dict[str, dict[str, float]] = {}
    for model_seed in readiness.FROZEN_ENCODER_SEEDS:
        group_key = f"{history_id}|{model_seed}"
        full_arm[str(model_seed)] = {}
        for k in CLUSTER_COUNTS:
            value = full_arm_e2(
                labels[group_key][k],
                list(row["previous_ids"]),
                list(row["current_ids"]),
                groups[group_key],
                k,
            )
            full_arm[str(model_seed)][str(k)] = value
            full_store[(identity, int(k), int(model_seed))] = value
    entry["designs"][DESIGN_FULL_ARM] = {"status": COMPUTED, "per_seed": full_arm}
    entry["sample_size_effect"] = {
        str(model_seed): {
            str(k): sample_size_effect_record(entry["designs"], int(model_seed), int(k))
            for k in CLUSTER_COUNTS
        }
        for model_seed in readiness.FROZEN_ENCODER_SEEDS
    }
    return entry


def descriptive_associations(
    diagnostics: Sequence[Mapping[str, Any]],
    matched_by_identity: Mapping[str, Mapping[str, Any]],
    *,
    k: int = INHERITED_PRIMARY_K,
) -> dict[str, dict[str, Any]]:
    """Descriptive rank association between matched-size E2 and composition.

    A transition whose stratified design is unavailable is skipped with an
    explicit exclusion record instead of being indexed, so the integrated path
    abstains on an unsupported matching design rather than raising. Descriptive
    only: shared arms make these transitions dependent and no p-value is
    produced.
    """
    associations: dict[str, dict[str, Any]] = {}
    for horizon in sorted({str(item["horizon"]) for item in diagnostics}):
        transitions = [
            item
            for item in diagnostics
            if str(item["horizon"]) == horizon
            and str(item["comparison_type"]) == "BASELINE-VS-ELEVATED"
            and item.get("cell_status") == COMPUTED
        ]
        used: list[Mapping[str, Any]] = []
        matched: list[float] = []
        excluded: list[dict[str, str]] = []
        for item in transitions:
            identity = str(item["comparison_identity"])
            entry = matched_by_identity.get(identity)
            record = entry.get("designs", {}).get(DESIGN_STRATIFIED) if entry else None
            if not isinstance(record, Mapping):
                excluded.append(
                    {"comparison_identity": identity, "reason": "no matched-size record"}
                )
                continue
            if record.get("status") != COMPUTED:
                excluded.append(
                    {
                        "comparison_identity": identity,
                        "reason": str(record.get("status")),
                    }
                )
                continue
            per_seed = record.get("per_seed")
            if not isinstance(per_seed, Mapping):
                excluded.append(
                    {
                        "comparison_identity": identity,
                        "reason": "stratified design carries no per-seed record",
                    }
                )
                continue
            seed_values: list[float] = []
            for model_seed in readiness.FROZEN_ENCODER_SEEDS:
                block = per_seed.get(str(model_seed))
                per_k = block.get("per_K") if isinstance(block, Mapping) else None
                cell_record = per_k.get(str(k)) if isinstance(per_k, Mapping) else None
                if isinstance(cell_record, Mapping) and cell_record.get("status") == COMPUTED:
                    seed_values.append(float(cell_record["median"]))
            value = seed_median(seed_values)
            if value is None:
                excluded.append(
                    {
                        "comparison_identity": identity,
                        "reason": "no computed stratified median",
                    }
                )
                continue
            used.append(item)
            matched.append(value)
        features = {
            "family_mix_tv": [item["mix_tv"]["family"] for item in used],
            "severity_mix_tv": [item["mix_tv"]["severity_bins"] for item in used],
            "anomalous_fraction_mix_tv": [item["mix_tv"]["anomalous_fraction_bins"] for item in used],
            "context_mix_tv": [item["mix_tv"]["context"] for item in used],
            "burden_contrast_gap": [item["burden_contrast_gap_after"] for item in used],
        }
        block: dict[str, Any] = {
            "horizon": horizon,
            "K": int(k),
            "n_transitions": len(transitions),
            "n_used": len(used),
            "n_excluded": len(excluded),
            "excluded_transitions": excluded,
            "dependence": (
                "transitions within a history share arms; these are descriptive rank "
                "associations only, never independent samples and never a test"
            ),
            "features": {},
        }
        for name, series in features.items():
            clean_x: list[float] = []
            clean_y: list[float] = []
            for position, x_value in enumerate(series):
                y_value = matched[position]
                if x_value is None or not np.isfinite(float(x_value)):
                    continue
                clean_x.append(float(x_value))
                clean_y.append(float(y_value))
            if len(clean_x) >= 3 and len(set(clean_x)) >= 2 and len(set(clean_y)) >= 2:
                from scipy.stats import spearmanr

                result = spearmanr(clean_x, clean_y)
                block["features"][name] = {
                    "n": len(clean_x),
                    "spearman_rho": float(result.statistic),  # type: ignore[attr-defined]
                }
            else:
                block["features"][name] = {"n": len(clean_x), "spearman_rho": None}
        associations[horizon] = block
    return associations


# --- main audit ---------------------------------------------------------------


def run_cache_diagnostic(
    *,
    corpus: Path,
    audit_path: Path,
    contract_path: Path,
    cache_root: Path,
    g2_preflight_path: Path | None = None,
    repeats: int = REPEATS,
    seed: int = readiness.NULL_SEED,
) -> dict[str, Any]:
    """Run the complete cache-only M12 audit and return its tables."""
    started = time.time()
    lineage = verify_lineage(corpus, audit_path, contract_path, g2_preflight_path)
    manifest = lineage["manifest"]
    audit_record = lineage["audit_record"]
    identity_plan = lineage["identity_plan"]
    cache = load_cache(cache_root, manifest, lineage["custody"], identity_plan)
    groups = cache["groups"]
    index = manifest_index(manifest)
    end_time_of, duration_of, header_reads = manifest_exposure_maps(manifest)

    fits, fit_custody = fit_references(groups, manifest, seed=seed)
    labels = prototype_labels(groups, fits)

    rows = comparison_rows(audit_record)
    cells = cell_sizes(rows)
    n_seeds = len(readiness.FROZEN_ENCODER_SEEDS)
    n_k = len(CLUSTER_COUNTS)

    # --- A: matched-size E2 in both designs, exact quanta ---------------------
    a_comparisons: list[dict[str, Any]] = []
    stratified_quanta: dict[tuple[str, int, int], np.ndarray] = {}
    srs_quanta: dict[tuple[str, int, int], np.ndarray] = {}
    full_values: dict[tuple[str, int, int], float] = {}
    plan_records: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = cell_key(str(row["history_id"]), str(row["horizon"]))
        cell = cells[key]
        identity = str(row["comparison_identity"])
        size = int(cell["size"])
        plan: Mapping[str, Any] | None
        if cell["status"] != COMPUTED:
            plan = None
            plan_records[identity] = {
                "status": ABSTAIN_SHORTFALL,
                "reason": "cell below the inherited 30-file period floor",
            }
        else:
            plan = design_plan(
                group=groups[f"{row['history_id']}|{readiness.FROZEN_ENCODER_SEEDS[0]}"],
                row=row,
                size=size,
                index=index,
                repeats=repeats,
            )
            plan_records[identity] = {
                "status": plan["sampling_mode"],
                "sampling_mode": plan["sampling_mode"],
                "stratified_available": bool(plan["stratified_available"]),
                "size": plan["size"],
                "repeats": plan["repeats"],
                "n_common_strata": plan["n_common_strata"],
                "n_baseline_strata": plan["n_baseline_strata"],
                "n_contrast_strata": plan["n_contrast_strata"],
                "allocation": plan["allocation"],
                "unavailable_reason": plan["unavailable_reason"],
            }
        a_comparisons.append(
            matched_size_entry(
                row=row,
                cell=cell,
                key=key,
                size=size,
                plan=plan,
                labels=labels,
                groups=groups,
                index=index,
                end_time_of=end_time_of,
                duration_of=duration_of,
                stratified_store=stratified_quanta,
                srs_store=srs_quanta,
                full_store=full_values,
            )
        )

    # --- retained-exposure reconstruction vs the accepted M10 census ----------
    census = readiness.expected_retained_census()
    transition_retained = [
        entry["retained_exposure"]
        for entry in a_comparisons
        if entry["comparison_type"] == "BASELINE-VS-ELEVATED"
        and entry["retained_exposure"] is not None
    ]
    control_retained = [
        entry["retained_exposure"]
        for entry in a_comparisons
        if entry["comparison_type"] == "BASELINE-VS-BASELINE"
        and entry["retained_exposure"] is not None
    ]

    def _range(values: Sequence[float]) -> list[float] | None:
        return [float(min(values)), float(max(values))] if values else None

    transition_accepted = list(census["retained_exposure_transition_range"])  # type: ignore[arg-type]
    control_accepted = list(census["retained_exposure_control_range"])  # type: ignore[arg-type]
    transition_observed = _range(transition_retained)
    control_observed = _range(control_retained)
    retained_exposure_check = {
        "definition": census["exposure_definition"],
        "exposure_floor": census["exposure_floor"],
        "n_transitions_with_value": len(transition_retained),
        "n_controls_with_value": len(control_retained),
        "observed_transition_range": transition_observed,
        "observed_control_range": control_observed,
        "accepted_transition_range": transition_accepted,
        "accepted_control_range": control_accepted,
        "matches_accepted_census": bool(
            transition_observed is not None
            and control_observed is not None
            and abs(transition_observed[0] - transition_accepted[0]) <= 1e-12
            and abs(transition_observed[1] - transition_accepted[1]) <= 1e-12
            and abs(control_observed[0] - control_accepted[0]) <= 1e-12
            and abs(control_observed[1] - control_accepted[1]) <= 1e-12
        ),
        "retained_eligible_transitions": int(
            sum(
                1
                for entry in a_comparisons
                if entry["comparison_type"] == "BASELINE-VS-ELEVATED"
                and entry["retained_exposure_at_or_above_floor"]
            )
        ),
        "retained_eligible_controls": int(
            sum(
                1
                for entry in a_comparisons
                if entry["comparison_type"] == "BASELINE-VS-BASELINE"
                and entry["retained_exposure_at_or_above_floor"]
            )
        ),
        "inherited_raw_transition_range": _range(
            [
                float(entry["exposure_ratio_after_inherited_raw"])  # type: ignore[arg-type]
                for entry in a_comparisons
                if entry["comparison_type"] == "BASELINE-VS-ELEVATED"
                and isinstance(entry["exposure_ratio_after_inherited_raw"], (int, float))
            ]
        ),
        "inherited_raw_control_range": _range(
            [
                float(entry["exposure_ratio_after_inherited_raw"])  # type: ignore[arg-type]
                for entry in a_comparisons
                if entry["comparison_type"] == "BASELINE-VS-BASELINE"
                and isinstance(entry["exposure_ratio_after_inherited_raw"], (int, float))
            ]
        ),
        "inherited_raw_role": "inherited-rule limitation/sensitivity; never the retained exposure",
        "metadata_header_reads": header_reads,
    }

    # --- transition-minus-control deltas (exact quanta) -----------------------
    deltas: list[dict[str, Any]] = []
    for cell_name, cell in sorted(cells.items()):
        control_identity = str(cell["control_identity"])
        members = [
            entry
            for entry in a_comparisons
            if entry["cell"] == cell_name and entry["comparison_type"] == "BASELINE-VS-ELEVATED"
        ]
        for entry in members:
            per_seed: dict[str, Any] = {}
            size = int(entry["size"])
            default_reason = (
                ABSTAIN_SHORTFALL
                if cell["status"] != COMPUTED
                else ABSTAIN_NO_COMMON_STRATA
            )
            for model_seed in readiness.FROZEN_ENCODER_SEEDS:
                per_design: dict[str, Any] = {}
                for design, store in ((DESIGN_STRATIFIED, stratified_quanta), (DESIGN_SRS, srs_quanta)):
                    per_design[design] = {}
                    for k in CLUSTER_COUNTS:
                        per_design[design][str(k)] = delta_summary(
                            store.get((entry["comparison_identity"], int(k), int(model_seed))),
                            store.get((control_identity, int(k), int(model_seed))),
                            size,
                            default_reason,
                        )
                per_seed[str(model_seed)] = per_design
            deltas.append(
                {
                    "cell": cell_name,
                    "cell_status": cell["status"],
                    "history_id": entry["history_id"],
                    "horizon": entry["horizon"],
                    "comparison_identity": entry["comparison_identity"],
                    "control_identity": control_identity,
                    "size": size,
                    "per_seed": per_seed,
                }
            )

    cell_summaries: dict[str, dict[str, Any]] = {}
    for cell_name, cell in sorted(cells.items()):
        members = [item for item in deltas if item["cell"] == cell_name]
        block: dict[str, Any] = {
            "history_id": cell["history_id"],
            "horizon": cell["horizon"],
            "size": cell["size"],
            "status": cell["status"],
            "n_transitions": len(members),
            "control_identity": cell["control_identity"],
            "per_seed": {},
            "per_K": {},
        }
        if cell["status"] != COMPUTED:
            block["reason"] = "cell below the inherited 30-file period floor"
            cell_summaries[cell_name] = block
            continue
        for model_seed in readiness.FROZEN_ENCODER_SEEDS:
            for k in CLUSTER_COUNTS:
                records = [
                    item["per_seed"][str(model_seed)][DESIGN_STRATIFIED][str(k)]
                    for item in members
                ]
                computed = [record for record in records if record["status"] == COMPUTED]
                quanta_medians = [record["median_quanta"] for record in computed]
                block["per_seed"].setdefault(str(model_seed), {})[str(k)] = {
                    "status": COMPUTED if computed else ABSTAIN_NO_COMMON_STRATA,
                    "n_transitions": len(computed),
                    "transition_median_deltas": [record["median"] for record in computed],
                    "transition_median_deltas_quanta": quanta_medians,
                    "min": min((record["median"] for record in computed), default=None),
                    "max": max((record["median"] for record in computed), default=None),
                    "median_of_transitions": seed_median(
                        [record["median"] for record in computed]
                    ),
                    "n_positive": int(sum(1 for value in quanta_medians if value > 0)),
                    "n_zero": int(sum(1 for value in quanta_medians if value == 0)),
                    "n_negative": int(sum(1 for value in quanta_medians if value < 0)),
                    "median_direction_fraction": seed_median(
                        [record["fraction_strictly_positive"] for record in computed]
                    ),
                    "median_zero_fraction": seed_median(
                        [record["fraction_zero"] for record in computed]
                    ),
                    "transition_median_deltas_srs": [
                        record["median"]
                        for record in (
                            item["per_seed"][str(model_seed)][DESIGN_SRS][str(k)]
                            for item in members
                        )
                        if record["status"] == COMPUTED
                    ],
                }
        for k in CLUSTER_COUNTS:
            per_transition: list[float | None] = []
            sign_sets: list[list[int] | None] = []
            for item in members:
                seed_values = []
                for model_seed in readiness.FROZEN_ENCODER_SEEDS:
                    record = item["per_seed"][str(model_seed)][DESIGN_STRATIFIED][str(k)]
                    if record["status"] == COMPUTED:
                        seed_values.append(record["median_quanta"])
                per_transition.append(seed_median(seed_values))
                direction = seed_direction(seed_values)
                sign_sets.append(direction["signs"] if direction["n"] == n_seeds else None)
            values = [value for value in per_transition if value is not None]
            all_seeds_zero = sum(1 for signs in sign_sets if signs == [0])
            agreement = bool(members) and all(
                signs is not None and len(signs) == 1 and signs[0] != 0 for signs in sign_sets
            )
            block["per_K"][str(k)] = {
                "status": COMPUTED if values else ABSTAIN_NO_COMMON_STRATA,
                "transition_seed_median_deltas_quanta": per_transition,
                "transition_seed_median_deltas": [
                    None if value is None else value / int(cell["size"]) for value in per_transition
                ],
                "n_positive": int(sum(1 for value in values if value > 0)),
                "n_zero": int(sum(1 for value in values if value == 0)),
                "n_negative": int(sum(1 for value in values if value < 0)),
                "n_transitions": len(values),
                "seed_signs_by_transition": sign_sets,
                "n_all_seeds_zero": int(all_seeds_zero),
                "median_quanta": float(np.median(values)) if values else None,
                "median": (float(np.median(values)) / int(cell["size"])) if values else None,
                "min": (float(min(values)) / int(cell["size"])) if values else None,
                "max": (float(max(values)) / int(cell["size"])) if values else None,
                "direction_agreement_all_seeds": bool(agreement),
            }
        cell_summaries[cell_name] = block

    # --- B: baseline variability ----------------------------------------------
    baseline_blocks: dict[str, dict[str, Any]] = {}
    b1_values: dict[tuple[str, int, int], np.ndarray] = {}
    b2_cross: dict[tuple[str, int, int], np.ndarray] = {}
    b2_within: dict[tuple[str, int, str, int, int], np.ndarray] = {}
    for cell_name, cell in sorted(cells.items()):
        history_id = str(cell["history_id"])
        control = next(
            row
            for row in rows
            if cell_key(str(row["history_id"]), str(row["horizon"])) == cell_name
            and str(row["comparison_type"]) == "BASELINE-VS-BASELINE"
        )
        block: dict[str, Any] = {
            "history_id": history_id,
            "horizon": str(cell["horizon"]),
            "status": cell["status"],
            "control_identity": str(control["comparison_identity"]),
            "baseline_arm_sizes": [int(control["n_baseline"]), int(control["n_contrast"])],
            "size_a": int(cell["size"]),
            "b1": {},
            "b2": {},
        }
        if cell["status"] != COMPUTED:
            block["reason"] = "cell below the inherited 30-file period floor"
            block["b1_status"] = ABSTAIN_SHORTFALL
            block["b2_status"] = ABSTAIN_SHORTFALL
            block["size_b"] = None
            baseline_blocks[cell_name] = block
            continue
        size_a = int(cell["size"])
        size_b = min(int(control["n_baseline"]) // 2, int(control["n_contrast"]) // 2)
        block["size_b"] = size_b
        # B1 is the A-size cross-period control: same arms, same draws.
        b1_designs: set[str] = set()
        for model_seed in readiness.FROZEN_ENCODER_SEEDS:
            block["b1"].setdefault(str(model_seed), {})
            for k in CLUSTER_COUNTS:
                key = (str(control["comparison_identity"]), int(k), int(model_seed))
                quanta = stratified_quanta.get(key)
                design_used = DESIGN_STRATIFIED
                if quanta is None:
                    quanta = srs_quanta.get(key)
                    design_used = DESIGN_SRS
                block["b1"][str(model_seed)][str(k)] = design_summary(
                    quanta, size_a, ABSTAIN_NO_COMMON_STRATA
                )
                if quanta is not None:
                    b1_values[(cell_name, int(k), int(model_seed))] = quanta
                    b1_designs.add(design_used)
        block["b1_design"] = (
            sorted(b1_designs)[0] if len(b1_designs) == 1 else sorted(b1_designs)
        )
        block["b1_status"] = COMPUTED if b1_values else ABSTAIN_NO_COMMON_STRATA
        if size_b < MIN_SPLIT_ARM_FILES:
            block["b2_status"] = ABSTAIN_SHORTFALL
            block["b2_reason"] = f"s_B {size_b} is below the inherited {MIN_SPLIT_ARM_FILES}-file floor"
            baseline_blocks[cell_name] = block
            continue
        block["b2_status"] = COMPUTED
        first_group = groups[f"{history_id}|{readiness.FROZEN_ENCODER_SEEDS[0]}"]
        a_positions = np.array([first_group.index[fid] for fid in control["previous_ids"]], dtype=np.int64)
        b_positions = np.array([first_group.index[fid] for fid in control["current_ids"]], dtype=np.int64)
        cross_a = np.empty((repeats, size_b), dtype=np.int32)
        cross_b = np.empty((repeats, size_b), dtype=np.int32)
        within_a1 = np.empty((repeats, size_b), dtype=np.int32)
        within_a2 = np.empty((repeats, size_b), dtype=np.int32)
        within_b1 = np.empty((repeats, size_b), dtype=np.int32)
        within_b2 = np.empty((repeats, size_b), dtype=np.int32)
        for repeat in range(repeats):
            parts = [TASK_ID, cell_name, "B2", repeat]
            cross_a[repeat] = np.sort(
                np.random.default_rng(_draw_seed(parts + ["cross", "baseline-A"])).choice(
                    a_positions, size=size_b, replace=False
                )
            )
            cross_b[repeat] = np.sort(
                np.random.default_rng(_draw_seed(parts + ["cross", "baseline-B"])).choice(
                    b_positions, size=size_b, replace=False
                )
            )
            split = np.random.default_rng(_draw_seed(parts + ["within", "baseline-A"])).permutation(
                a_positions.size
            )
            within_a1[repeat] = np.sort(a_positions[split][:size_b])
            within_a2[repeat] = np.sort(a_positions[split][size_b : 2 * size_b])
            split = np.random.default_rng(_draw_seed(parts + ["within", "baseline-B"])).permutation(
                b_positions.size
            )
            within_b1[repeat] = np.sort(b_positions[split][:size_b])
            within_b2[repeat] = np.sort(b_positions[split][size_b : 2 * size_b])
        for model_seed in readiness.FROZEN_ENCODER_SEEDS:
            block["b2"].setdefault(str(model_seed), {})
            label_rows_by_k = labels[f"{history_id}|{model_seed}"]
            for k in CLUSTER_COUNTS:
                label_rows = label_rows_by_k[k]
                cross = e2_quanta(label_rows, cross_a, cross_b, k)
                within_first = e2_quanta(label_rows, within_a1, within_a2, k)
                within_second = e2_quanta(label_rows, within_b1, within_b2, k)
                b2_cross[(cell_name, int(k), int(model_seed))] = cross
                b2_within[(cell_name, int(k), "baseline-A", int(model_seed))] = within_first
                b2_within[(cell_name, int(k), "baseline-B", int(model_seed))] = within_second
                cross_summary = summary_of_quanta(cross, size_b)
                first_summary = summary_of_quanta(within_first, size_b)
                second_summary = summary_of_quanta(within_second, size_b)
                block["b2"][str(model_seed)][str(k)] = {
                    "cross_period": cross_summary,
                    "within_period_baseline_A": first_summary,
                    "within_period_baseline_B": second_summary,
                    "temporal_excess_vs_A": cross_summary["median"] - first_summary["median"],
                    "temporal_excess_vs_B": cross_summary["median"] - second_summary["median"],
                    "temporal_excess_quanta_vs_A": cross_summary["median_quanta"]
                    - first_summary["median_quanta"],
                    "temporal_excess_quanta_vs_B": cross_summary["median_quanta"]
                    - second_summary["median_quanta"],
                }
        block["b2_arms_disjoint"] = True
        block["b2_sampling_mode"] = DESIGN_SRS
        baseline_blocks[cell_name] = block

    # --- C: robustness summaries ----------------------------------------------
    history_level: list[dict[str, Any]] = []
    for cell_name, block in sorted(cell_summaries.items()):
        for k in CLUSTER_COUNTS:
            values = block["per_K"][str(k)]
            history_level.append(
                {
                    "cell": cell_name,
                    "history_id": block["history_id"],
                    "horizon": block["horizon"],
                    "K": int(k),
                    "size": block["size"],
                    "status": values["status"],
                    "transition_seed_median_deltas": values["transition_seed_median_deltas"],
                    "transition_seed_median_deltas_quanta": values[
                        "transition_seed_median_deltas_quanta"
                    ],
                    "n_positive": values["n_positive"],
                    "n_zero": values["n_zero"],
                    "n_negative": values["n_negative"],
                    "median": values["median"],
                    "min": values["min"],
                    "max": values["max"],
                    "n_all_seeds_zero": values["n_all_seeds_zero"],
                    "direction_agreement_all_seeds": values["direction_agreement_all_seeds"],
                }
            )
    horizon_counts: dict[str, dict[str, Any]] = {}
    for horizon in sorted({str(block["horizon"]) for block in cell_summaries.values()}):
        entries = [item for item in history_level if item["horizon"] == horizon]
        horizon_counts[horizon] = {
            "n_cells": len({item["cell"] for item in entries}),
            "n_cells_abstained": len(
                {item["cell"] for item in entries if item["status"] != COMPUTED}
            ),
            "K": {
                str(k): {
                    "n_transitions_positive": int(
                        sum(item["n_positive"] for item in entries if item["K"] == k)
                    ),
                    "n_transitions_zero": int(
                        sum(item["n_zero"] for item in entries if item["K"] == k)
                    ),
                    "n_transitions_negative": int(
                        sum(item["n_negative"] for item in entries if item["K"] == k)
                    ),
                    "n_transitions_total": int(
                        sum(
                            len(
                                [
                                    value
                                    for value in item["transition_seed_median_deltas_quanta"]
                                    if value is not None
                                ]
                            )
                            for item in entries
                            if item["K"] == k
                        )
                    ),
                    "cells": {item["cell"]: item["n_positive"] for item in entries if item["K"] == k},
                }
                for k in CLUSTER_COUNTS
            },
            "note": "counting summary only; no estimate is pooled across histories or horizons",
        }

    # --- D: diagnostic decomposition ------------------------------------------
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        history_id = str(row["history_id"])
        key = cell_key(history_id, str(row["horizon"]))
        cell = cells[key]
        baseline = arm_composition(list(row["previous_ids"]), index)
        contrast = arm_composition(list(row["current_ids"]), index)
        flags = composition_flags(baseline, contrast)
        family_of = contrast["family_of"]
        for composition in (baseline, contrast):
            composition.pop("family_of", None)
        decomposition: dict[str, dict[str, Any]] = {}
        for model_seed in readiness.FROZEN_ENCODER_SEEDS:
            decomposition[str(model_seed)] = {}
            for k in CLUSTER_COUNTS:
                if cell["status"] != COMPUTED:
                    decomposition[str(model_seed)][str(k)] = {
                        "status": ABSTAIN_SHORTFALL,
                        "reason": "cell below the inherited 30-file period floor",
                    }
                    continue
                decomposition[str(model_seed)][str(k)] = tv_nonadditivity(
                    labels=labels[f"{history_id}|{model_seed}"][k],
                    group=groups[f"{history_id}|{model_seed}"],
                    row=row,
                    family_of=family_of,
                    k=k,
                )
        mix_values = {
            "family": mix_tv(baseline["family_counts"], contrast["family_counts"]),
            "severity_bins": mix_tv(
                baseline["severity_bin_counts"], contrast["severity_bin_counts"]
            ),
            "anomalous_fraction_bins": mix_tv(
                baseline["anomalous_fraction_bin_counts"],
                contrast["anomalous_fraction_bin_counts"],
            ),
            "context": mix_tv(baseline["context_counts"], contrast["context_counts"]),
        }
        mix_undefined = sorted(name for name, value in mix_values.items() if value is None)
        diagnostics.append(
            {
                "cell": key,
                "cell_status": cell["status"],
                "history_id": history_id,
                "horizon": str(row["horizon"]),
                "comparison_identity": str(row["comparison_identity"]),
                "comparison_type": str(row["comparison_type"]),
                "disposition": str(row["disposition"]),
                "burden_contrast_gap_after": row["burden_contrast_gap_after"],
                "baseline": baseline,
                "contrast": contrast,
                "mix_tv": mix_values,
                "mix_tv_undefined": mix_undefined,
                "mix_tv_undefined_reason": (
                    "an arm has no member in that partition, so the composition "
                    "distance is undefined and is reported as null, never as zero"
                )
                if mix_undefined
                else None,
                "flags": flags,
                "tv_decomposition": decomposition,
            }
        )

    matched_by_identity = {entry["comparison_identity"]: entry for entry in a_comparisons}
    associations = descriptive_associations(diagnostics, matched_by_identity)

    # --- accounting -----------------------------------------------------------
    n_rows = len(rows)
    n_cells = len(cells)
    planned = {
        "cells": n_cells,
        "comparisons": n_rows,
        "transitions": sum(int(cell["n_transitions"]) for cell in cells.values()),
        "controls": n_cells,
        "A_full_arm_cells": n_rows * n_k * n_seeds,
        "A_stratified_cells": n_rows * n_k * n_seeds,
        "A_srs_cells": n_rows * n_k * n_seeds,
        "B1_cells": n_cells * n_k * n_seeds,
        "B2_cells": n_cells * 3 * n_k * n_seeds,
        "D_tv_decompositions": n_rows * n_k * n_seeds,
    }
    computed = {
        "A_full_arm_cells": len(full_values),
        "A_stratified_cells": len(stratified_quanta),
        "A_srs_cells": len(srs_quanta),
        "B1_cells": len(b1_values),
        "B2_cells": len(b2_cross) + len(b2_within),
        "D_tv_decompositions": sum(
            1
            for item in diagnostics
            for seed_block in [item["tv_decomposition"]]
            for k_block in seed_block.values()
            for record in k_block.values()
            if record.get("status") != ABSTAIN_SHORTFALL
        ),
    }
    abstained = {name: planned[name] - computed[name] for name in computed}
    abstention_reasons: dict[str, int] = {}

    def _tally(status: object) -> None:
        if isinstance(status, str) and status != COMPUTED:
            abstention_reasons[status] = abstention_reasons.get(status, 0) + 1

    for record in a_comparisons:
        for design in DESIGNS:
            _tally(record["designs"][design].get("status"))
    for item in diagnostics:
        for seed_block in item["tv_decomposition"].values():
            for record in seed_block.values():
                _tally(record.get("status"))
    accounting = {
        "cells": n_cells,
        "active_cells": len([cell for cell in cells.values() if cell["status"] == COMPUTED]),
        "abstained_cells": len([cell for cell in cells.values() if cell["status"] != COMPUTED]),
        "comparisons": n_rows,
        "transitions": planned["transitions"],
        "controls": planned["controls"],
        "n_fits": len(fits),
        "n_seeds": n_seeds,
        "cluster_counts": list(CLUSTER_COUNTS),
        "repeats": repeats,
        "planned": planned,
        "computed": computed,
        "abstained": abstained,
        "abstention_reasons": abstention_reasons,
        "all_planned_cells_accounted": bool(
            all(computed[name] + abstained[name] == planned[name] for name in computed)
        ),
        "cell_statuses": {name: cell["status"] for name, cell in sorted(cells.items())},
    }

    spec_file = REPO_ROOT / SPEC_PATH
    return {
        "task": TASK_ID,
        "spec": {
            "artifact_id": SPEC_ID,
            "path": SPEC_PATH,
            "sha256": sha256_file(spec_file) if spec_file.is_file() else None,
        },
        "lineage": {
            key: value
            for key, value in lineage.items()
            if key not in ("manifest", "audit_record", "contract", "custody", "identity_plan")
        },
        "cache": {key: value for key, value in cache.items() if key != "groups"},
        "fit_custody": fit_custody,
        "support": {
            "cells": cells,
            "sampling_plans": plan_records,
            "retained_exposure": retained_exposure_check,
        },
        "matched_size": {"comparisons": a_comparisons, "cells": cell_summaries},
        "contrasts": deltas,
        "baseline": baseline_blocks,
        "robustness": {"history_level": history_level, "horizon_counts": horizon_counts},
        "diagnostics": {"comparisons": diagnostics, "associations": associations},
        "accounting": accounting,
        "repeats": {
            "stratified_quanta": stratified_quanta,
            "srs_quanta": srs_quanta,
            "b1_quanta": b1_values,
            "b2_cross_quanta": b2_cross,
            "b2_within_quanta": b2_within,
        },
        "elapsed_s": time.time() - started,
        "forwards_executed_this_run": 0,
        "model_loads": 0,
        "limitations": [
            "cache-only exploratory audit of one fixed 24,840-vector admitted cache; no new capture, no new forward pass and no independent confirmation of S24-7-M11",
            "the author had read the published M11 movement values before freezing this design, so no M12 quantity is a blind replication",
            "equal-n quantities are exact occupancy quanta (E2 = m/s); the subsampled designs are compared at one support-derived per-arm size, so the full-arm and matched-size values differ in both sample size and conditioning and no causal attribution follows from their difference",
            "repeats are Monte Carlo integration of the sampling distribution of one finite support; their quantiles are resampling variability, never confidence limits",
            "encoder seeds are run replicates, never independent data; comparisons sharing an arm are dependent by construction",
            "diagnostic family/severity/context fields are post-hoc only and never entered a prototype fit, a sampling draw or a sample-size rule",
            "no p-value, success gate, performance, detection-accuracy, business-density or causal claim is produced",
        ],
    }


__all__ = [
    "TASK_ID",
    "SPEC_ID",
    "SPEC_PATH",
    "CONTRACT_ENVELOPE_SHA256",
    "REPEATS",
    "CLUSTER_COUNTS",
    "INHERITED_PRIMARY_K",
    "MIN_CELL_SIZE",
    "MIN_SPLIT_ARM_FILES",
    "DESIGN_FULL_ARM",
    "DESIGN_SRS",
    "DESIGN_STRATIFIED",
    "DESIGNS",
    "COMPUTED",
    "ABSTAIN_SHORTFALL",
    "ABSTAIN_NO_COMMON_STRATA",
    "ABSTAIN_EMPTY",
    "ABSTAIN_REASONS",
    "SEVERITY_BINS",
    "FRACTION_BINS",
    "ORDINARY",
    "verify_lineage",
    "load_cache",
    "sha256_file",
    "fit_references",
    "prototype_labels",
    "manifest_index",
    "stratum_id",
    "manifest_exposure_maps",
    "context_mix_tv",
    "comparison_rows",
    "cell_key",
    "cell_sizes",
    "stratum_codes",
    "allocation_with_caps",
    "stratified_draw",
    "design_plan",
    "prototype_counts",
    "e2_quanta",
    "e2_repeats",
    "occupancy",
    "tv",
    "full_arm_e2",
    "summary_of_quanta",
    "abstain_summary",
    "design_summary",
    "delta_summary",
    "seed_median",
    "arm_composition",
    "mix_tv",
    "composition_flags",
    "tv_nonadditivity",
    "sample_size_effect_record",
    "seed_direction",
    "matched_size_entry",
    "descriptive_associations",
    "run_cache_diagnostic",
]
