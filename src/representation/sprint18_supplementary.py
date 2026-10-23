"""Supplementary Development metric contract for the accepted Sprint 18 protocol.

Task 32 established that five mandatory protocol §8.2 metric families were never
produced by the Sprint 18 evaluation path (background stability + uncertainty,
localization/duration accounting, severity ordering, and the P1/P2/W1/W2/A1/A2
slices). The project user authorised a bounded supplementary Development metric
pass; this module freezes the metric definitions, the prospective duration
accounting supplement, and the strict input/aggregation contract consumed by
the supplementary runner.

Design constraints, all fail-closed:

* **Exact reuse.** Event AUROC, recall/lead, FAR/robot-day, the stability ratio
  and the history-block bootstrap reuse the accepted frozen implementations
  (``synth.probe15``, ``synth.events``, ``representation.attribution_metrics``).
  Nothing here redefines an accepted metric.
* **Additive versioning.** The frozen v2 result schema (``_METRIC_KEYS`` /
  ``_AGGREGATE_KEYS`` / ``_validate_history_metric``) is unchanged. The
  supplementary document is a *new* closed schema that adds exactly
  ``duration_accounting``; its per-history rows project back onto the v2 row
  shape and are validated by the unchanged v2 validator. Feeding a supplementary
  row to the v2 validator is still refused, so no historical row is silently
  reinterpreted.
* **No outcome-adaptive choice.** Every definition, threshold source, seed and
  bootstrap rule below is fixed before any supplementary value is observed.
* **Development only.** There is no Confirmation code path: the phase is a
  literal, the role action comes from the unchanged harness permission table,
  and Confirmation scoring stays closed behind the Task 33 lock.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import cast

import numpy as np

from representation import attribution_metrics as M
from representation.sprint18_harness import (
    _AGGREGATE_KEYS,
    _METRIC_KEYS,
    _canonical_sha256,
    _is_hash,
    _validate_history_metric,
    MODEL_SEEDS,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
)
from synth import events as E
from synth import probe15 as P

__all__ = [
    "SUPPLEMENTARY_SCHEMA_ID",
    "SUPPLEMENTARY_SCHEMA_VERSION",
    "SUPPLEMENTARY_INPUT_SCHEMA_ID",
    "SUPPLEMENTARY_SCORES_SCHEMA_ID",
    "DURATION_ACCOUNTING_ID",
    "LOCALIZATION_ID",
    "SEVERITY_ID",
    "SUBTYPE_SLICE_ID",
    "STABILITY_ID",
    "UNCERTAINTY_ID",
    "SUPPLEMENTARY_VALID_ARMS",
    "SUPPLEMENTARY_BRANCHES",
    "DURATION_ACCOUNTING_KEYS",
    "SUPPLEMENTARY_HISTORY_KEYS",
    "SUPPLEMENTARY_AVERAGED_ROW_KEYS",
    "SUPPLEMENTARY_RESULT_KEYS",
    "FrozenSupplementaryInputs",
    "EvaluableEvent",
    "EvaluableSupport",
    "HistoryEvaluationInput",
    "evaluable_support",
    "localization_diagnostics",
    "localization_rate",
    "severity_ordering",
    "subtype_slices",
    "background_stability_ratio",
    "duration_accounting",
    "history_block_interval",
    "build_history_metric_row",
    "build_branch_metrics",
    "row_times",
    "supplementary_v2_projection",
    "validate_supplementary_result",
    "validate_supplementary_scores",
    "score_document_digest",
    "score_map_sha256",
    "metric_code_sha256",
    "load_frozen_input_manifest",
    "compute_input_digests",
    "assert_inputs_unchanged",
]

#: Closed supplementary result identity. A new version is required for any change.
SUPPLEMENTARY_SCHEMA_ID = "sprint18-supplementary-development-v1"
SUPPLEMENTARY_SCHEMA_VERSION = 1

#: Frozen supplementary input-manifest identity (custody contract).
SUPPLEMENTARY_INPUT_SCHEMA_ID = "sprint18-supplementary-input-manifest-v2"

#: Frozen per-file score document identity (runner output / metric consumer input).
SUPPLEMENTARY_SCORES_SCHEMA_ID = "sprint18-supplementary-scores-v2"

BANK_RECIPE_ID = "sprint18-fit-bank-restore-v1"
BANK_ORDER_ID = "global-file-id-ascending-v1"
BANK_ARTIFACT_SCHEMA_ID = "sprint18-fit-reference-bank-v2"
BANK_PROVENANCE_SCHEMA_ID = "sprint18-fit-reference-bank-provenance-v2"
K9_BANK_RECIPE_ID = "s18-k9-downstream-state-v1"
K9_BANK_ORDER_ID = "persisted-downstream-order-v1"

#: Prospective supplement identities, fixed before any value is observed.
DURATION_ACCOUNTING_ID = "sprint18-duration-accounting-v1"
LOCALIZATION_ID = "sprint18-localization-v1"
SEVERITY_ID = "sprint18-severity-ordering-v1"
SUBTYPE_SLICE_ID = "sprint18-subtype-slices-v1"
STABILITY_ID = "sprint18-background-stability-v1"
UNCERTAINTY_ID = "sprint18-history-block-uncertainty-v1"

#: The seven executable arms. N1-HCC is STOPPED with no accepted run and is
#: barred here rather than silently skipped.
SUPPLEMENTARY_VALID_ARMS = (
    "B0-FULL",
    "K3-FULL",
    "K7-FULL",
    "K9-FULL",
    "N2-MSP",
    "N3-CCRP",
    "N4-CRN",
)

SUPPLEMENTARY_BRANCHES = ("S_pred", "S_pop")

#: The one additive v1 duration-accounting record (see the prospective supplement
#: ``experiments/sprint18-supplementary-development-metrics-v1.md`` §3).
DURATION_ACCOUNTING_KEYS = frozenset(
    {
        "schema_id",
        "evaluated_events",
        "recalled_events",
        "unrecalled_events",
        "median_declared_duration_days",
        "median_detected_alert_span_days",
        "median_first_alert_lead_days",
        "total_detected_alert_days",
        "episode_grouping_days",
        "events_without_positive_duration",
        "excluded_events",
    }
)

#: A supplementary per-history row is exactly the frozen v2 metric row plus the
#: single additive duration record. No other field may appear.
SUPPLEMENTARY_HISTORY_KEYS = frozenset(_METRIC_KEYS) | {"duration_accounting"}

SUPPLEMENTARY_EXECUTION_KEYS = frozenset(
    {
        "task_id",
        "attempt_id",
        "agent_id",
        "source_commit",
        "metric_code_sha256",
        "protocol_path",
        "metric_code_path",
        "supplementary_definition_path",
        "supplementary_definition_sha256",
        "scope",
        "phase",
        "device",
        "started_utc",
        "finished_utc",
    }
)

SUPPLEMENTARY_INPUTS_KEYS = frozenset(
    {
        "manifest_path",
        "manifest_sha256",
        "data_root",
        "checkpoint_path",
        "checkpoint_sha256",
        "reference_artifact_path",
        "reference_artifact_sha256",
        "accepted_record_path",
        "accepted_record_sha256",
        "task14_lock_path",
        "task14_lock_sha256",
        "handoff_path",
        "handoff_sha256",
        "seed_score_documents",
        "baseline_score_documents",
        "sha256_before",
        "sha256_after",
        "unchanged",
    }
)

SUPPLEMENTARY_REPRODUCTION_KEYS = frozenset(
    {
        "arm_config_sha256_by_seed",
        "data_binding_sha256",
        "model_state_sha256_by_seed",
        "ema_state_sha256_by_seed",
        "bank_sha256_by_seed",
        "accepted_bank_sha256_by_seed",
        "bank_reproduced",
        "bank_source",
        "reference_state_modes",
        "calibration_thresholds_by_seed",
        "accepted_calibration_thresholds_by_seed",
        "recomputed_calibration_thresholds_by_seed",
        "calibration_reproduced",
        "representation_state_unchanged",
        "reference_state_unchanged",
        "encoder_updates",
    }
)

SCORES_REPRODUCTION_KEYS = frozenset(
    {
        "arm_config_sha256",
        "data_binding_sha256",
        "model_state_sha256",
        "ema_state_sha256",
        "bank_sha256",
        "accepted_bank_sha256",
        "bank_reproduced",
        "bank_source",
        "bank_source_rows",
        "bank_source_rows_sha256",
        "bank_recipe_id",
        "bank_order_id",
        "bank_artifact",
        "bank_provenance",
        "reference_state_modes",
        "calibration_thresholds",
        "accepted_calibration_thresholds",
        "recomputed_calibration_thresholds",
        "calibration_reproduced",
        "representation_state_unchanged",
        "reference_state_unchanged",
        "encoder_updates",
    }
)

#: How a frozen reference/normalization state entered the supplementary run.
#: ``restored_from_custody`` means the bytes came from an accepted persisted
#: artifact; ``reconstructed_digest_verified`` means they were deterministically
#: re-derived and accepted only because they reproduce the accepted digest
#: exactly. The two are never conflated in any output.
REFERENCE_STATE_MODES = ("restored_from_custody", "reconstructed_digest_verified")

SUPPLEMENTARY_RESULT_KEYS = frozenset(
    {
        "schema_id",
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "phase",
        "arm_id",
        "model_seed",
        "execution",
        "inputs",
        "reproduction",
        "support",
        "metric_definitions",
        "branches",
        "status",
        "failure_reasons",
    }
)

SUPPLEMENTARY_BRANCH_KEYS = frozenset({"thresholds_by_seed", "per_seed", "per_history", "aggregate"})

SUPPLEMENTARY_SEED_KEYS = frozenset({"model_seed", "per_history"})

#: Seed-averaged gate cells are exactly the frozen v2 metric row, so the
#: unchanged v2 history validator consumes them without reinterpretation.
SUPPLEMENTARY_AVERAGED_ROW_KEYS = frozenset(_METRIC_KEYS)

SUPPLEMENTARY_SUPPORT_KEYS = frozenset(
    {
        "fit_healthy_rows",
        "fit_healthy_rows_sha256",
        "fit_healthy_scores_sha256",
        "calibration_healthy_rows",
        "calibration_support_sha256",
        "calibration_scores_sha256",
        "development",
        "excluded_rows",
    }
)

SUPPLEMENTARY_DEVELOPMENT_SUPPORT_KEYS = frozenset(
    {
        "history_id",
        "schema_history_id",
        "manifest_sha256",
        "support_sha256",
        "scores_sha256",
        "rows",
        "excluded_rows",
        "p",
        "w",
        "a",
        "controls",
        "robot_days",
        "evaluated_pw_events",
        "localization_excluded_events",
    }
)

_METRIC_DEFINITION_KEYS = frozenset(
    {
        "localization",
        "duration_accounting",
        "severity_ordering",
        "subtype_slices",
        "background_stability",
        "background_uncertainty",
        "event_metrics",
    }
)

_PHASE = "development"
REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "experiments" / "sprint18-full-training-protocol-v2.md"
SUPPLEMENTARY_DEFINITION_PATH = (
    REPO_ROOT / "experiments" / "sprint18-supplementary-development-metrics-v1.md"
)


def metric_code_sha256() -> str:
    """Digest of these frozen metric definitions, bound by every input manifest."""
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _require_hash(value: object, label: str) -> str:
    if not _is_hash(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return cast(str, value)


def _require_digest_map(value: object, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a string-to-SHA-256 mapping")
    digests: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} contains an empty or non-string key")
        digests[key] = _require_hash(digest, f"{label}.{key}")
    return digests


def _require_count(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _optional_finite(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite or null")
    return result


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _closed_keys(value: object, expected: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{label} must contain exactly the closed {SUPPLEMENTARY_SCHEMA_ID} keys")
    return cast(Mapping[str, object], value)


def _validate_execution_paths(execution: Mapping[str, object], label: str) -> None:
    expected_paths = {
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "metric_code_path": str(Path(__file__).resolve()),
        "supplementary_definition_path": str(SUPPLEMENTARY_DEFINITION_PATH.resolve()),
    }
    for name, expected in expected_paths.items():
        value = execution[name]
        if not isinstance(value, str) or value != expected:
            raise ValueError(f"{label}.{name} differs from the frozen source path")
    if _hash_file(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError(f"{label}.protocol_path bytes differ from the accepted protocol digest")
    if execution["metric_code_sha256"] != metric_code_sha256():
        raise ValueError(f"{label}.metric_code_sha256 differs from the executed source bytes")
    if _hash_file(SUPPLEMENTARY_DEFINITION_PATH) != execution["supplementary_definition_sha256"]:
        raise ValueError(f"{label}.supplementary_definition_sha256 differs from the source document bytes")


# ---------------------------------------------------------------------------
# Frozen event-support mirror (exact Sprint 17 sprint17_task10_c2 semantics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluableEvent:
    """One evaluable positive event: exactly the Sprint 17 support triple plus
    the ledger fields the supplementary duration/severity/slice families need."""

    key: tuple[str, str, float]
    cohort: str
    members: tuple[str, ...]
    severity: float | None
    subtype: str | None
    duration_days: float | None
    failure_time: float


@dataclass(frozen=True, slots=True)
class EvaluableSupport:
    """Evaluable positive events, control windows, and skip count for one history.

    ``events`` preserve the frozen ledger order; ``skipped`` counts failures the
    canonical metric path also excludes (reset-spanning horizon or no
    POS-ELIGIBLE file). ``controls`` are the frozen greedy non-overlapping
    window member lists.
    """

    events: tuple[EvaluableEvent, ...]
    controls: tuple[tuple[str, ...], ...]
    skipped: int


def evaluable_support(
    rows: Sequence[Mapping[str, object]],
    ledger: Sequence[Mapping[str, object]],
    windows: Mapping[str, Sequence[Sequence[float]]],
) -> EvaluableSupport:
    """Mirror ``probe15.evaluate_histories`` event/control selection exactly.

    Line-for-line semantics of the accepted Sprint 17 helper: reset-spanning
    positive horizons are skipped, failures with no POS-ELIGIBLE file are
    skipped, and control windows come from the frozen greedy selector. The only
    additions are the ledger ``severity``/``subtype``/``duration_d``/
    ``failure_time`` values carried onto each event.
    """
    row_list = [dict(row) for row in rows]
    window_map = cast(dict[str, list[list[float]]], dict(windows))
    events: list[EvaluableEvent] = []
    skipped = 0
    for failure in ledger:
        if E.positive_window_intersects_reset(failure, window_map):
            skipped += 1
            continue
        candidates = E.pos_files(row_list, failure, window_map)
        if not candidates:
            skipped += 1
            continue
        severity = failure.get("severity")
        duration = failure.get("duration_d")
        events.append(
            EvaluableEvent(
                key=(
                    str(failure["cohort"]),
                    str(failure["robot_id"]),
                    float(failure["failure_time"]),
                ),
                cohort=str(failure["cohort"]),
                members=tuple(str(item["file_id"]) for item in candidates),
                severity=None if severity is None else float(severity),
                subtype=None if failure.get("subtype") is None else str(failure["subtype"]),
                duration_days=None if duration is None else float(duration),
                failure_time=float(failure["failure_time"]),
            )
        )
    anchors = E.anchor_rows(row_list, window_map)
    controls = E.select_control_windows(anchors, [dict(failure) for failure in ledger], window_map)
    return EvaluableSupport(
        events=tuple(events),
        controls=tuple(tuple(str(member["file_id"]) for member in window["members"]) for window in controls),
        skipped=skipped,
    )


def _positive_events(support: EvaluableSupport) -> tuple[EvaluableEvent, ...]:
    return tuple(event for event in support.events if event.cohort in ("P", "W"))


def _control_member_scores(support: EvaluableSupport, scores: Mapping[str, float]) -> tuple[float, ...]:
    """Frozen background sample: the flattened control-window member scores."""
    return tuple(
        float(scores[file_id])
        for members in support.controls
        for file_id in members
    )


# ---------------------------------------------------------------------------
# Localization (exact Sprint 17 peak-timestep-in-horizon definition)
# ---------------------------------------------------------------------------


def localization_diagnostics(
    timesteps: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, object]]:
    """Reduce each file's per-timestep diagnostic to the compact frozen fields.

    Sprint 17 localized by the arg-max timestep of the per-file timestep score
    vector, skipping empty, non-finite, and constant vectors. Persisting the
    arg-max and length (instead of the whole vector) keeps the supplementary
    document bounded while remaining exactly sufficient for that definition;
    the raw vectors are additionally pinned by a digest in the score document.
    """
    reduced: dict[str, dict[str, object]] = {}
    for file_id, values in timesteps.items():
        array = np.asarray(list(values), dtype=np.float64)
        finite = bool(array.size > 0 and np.isfinite(array).all())
        constant = bool(finite and array.max() == array.min())
        reduced[str(file_id)] = {
            "n_timesteps": int(array.size),
            "peak_index": int(np.argmax(array)) if finite and not constant else None,
            "finite": finite,
            "constant": constant,
        }
    return reduced


def localization_rate(
    events: Sequence[EvaluableEvent],
    scores: Mapping[str, float],
    diagnostics: Mapping[str, Mapping[str, object]],
    row_times: Mapping[str, tuple[float, float]],
) -> tuple[float | None, int, int, int]:
    """Peak-timestep-in-horizon rate over evaluable P+W events (Sprint 17 exact).

    Returns ``(rate, hits, evaluated, excluded)``. ``rate`` is ``None`` when no
    event is evaluable (never a fabricated 0.0). ``excluded`` counts events whose
    top-scoring file has an unusable timestep diagnostic.
    """
    hits = 0
    evaluated = 0
    excluded = 0
    for event in _positive_events_of(events):
        top = max(event.members, key=lambda file_id: scores[file_id])
        diagnostic = diagnostics.get(top)
        if diagnostic is None or diagnostic.get("peak_index") is None:
            excluded += 1
            continue
        start, end = row_times[top]
        n_timesteps = int(cast(int, diagnostic["n_timesteps"]))
        peak_index = int(cast(int, diagnostic["peak_index"]))
        delta = (float(end) - float(start)) / max(1, n_timesteps)
        peak_time = float(start) + peak_index * delta
        evaluated += 1
        if event.failure_time - E.HORIZON_S <= peak_time <= event.failure_time:
            hits += 1
    rate = float(hits / evaluated) if evaluated else None
    return rate, hits, evaluated, excluded


def _positive_events_of(events: Sequence[EvaluableEvent]) -> tuple[EvaluableEvent, ...]:
    return tuple(event for event in events if event.cohort in ("P", "W"))


# ---------------------------------------------------------------------------
# Severity ordering and subtype slices (exact Sprint 17 definitions)
# ---------------------------------------------------------------------------


def severity_ordering(
    events: Sequence[EvaluableEvent],
    scores: Mapping[str, float],
) -> dict[str, float | None]:
    """Within-cohort Spearman correlation of event score vs severity.

    Sprint 17 fired only when at least three severity-carrying events existed
    for the cohort; fewer points (and a non-finite rho) report ``None``.
    """
    result: dict[str, float | None] = {}
    for cohort in ("P", "W"):
        points = [
            (float(cast(float, event.severity)), E.window_score(list(event.members), dict(scores)))
            for event in events
            if event.cohort == cohort and event.severity is not None
        ]
        if len(points) >= 3:
            rho = M.severity_spearman(
                [score for _, score in points],
                [level for level, _ in points],
            )
            result[cohort] = float(rho) if math.isfinite(rho) else None
        else:
            result[cohort] = None
    return result


def subtype_slices(
    events: Sequence[EvaluableEvent],
    scores: Mapping[str, float],
    control_scores: Sequence[float],
) -> dict[str, float | None]:
    """Tie-aware event AUROC per frozen subtype on identical control negatives."""
    slice_map = {
        "P1": ("P", "P1"),
        "P2": ("P", "P2"),
        "W1": ("W", "W1"),
        "W2": ("W", "W2"),
        "A1": ("A", "A1"),
        "A2": ("A", "A2"),
    }
    negatives = [float(value) for value in control_scores]
    result: dict[str, float | None] = {}
    for slice_id, (cohort, subtype) in slice_map.items():
        cell = [
            E.window_score(list(event.members), dict(scores))
            for event in events
            if event.cohort == cohort and event.subtype == subtype
        ]
        if not cell or not negatives:
            result[slice_id] = None
            continue
        try:
            result[slice_id] = float(E.roc_auc_tie_aware(cell, negatives))
        except E.UnavailableError:
            result[slice_id] = None
    return result


def background_stability_ratio(
    background_scores: Sequence[float],
    fit_healthy_scores: Sequence[float],
) -> float | None:
    """Frozen Sprint 17 stability ratio; ``None`` when either sample is empty.

    ``|median(background) − median(Fit healthy)| / max(pooled SD, STD_FLOOR)``
    via ``attribution_metrics.stability_ratio`` (population variances, float64).
    Non-finite input raises instead of producing a rank.
    """
    background = np.asarray(list(background_scores), dtype=np.float64)
    fit_healthy = np.asarray(list(fit_healthy_scores), dtype=np.float64)
    if background.size == 0 or fit_healthy.size == 0:
        return None
    return float(M.stability_ratio(background, fit_healthy)["ratio"])


# ---------------------------------------------------------------------------
# Prospective duration accounting (the additive v1 supplement)
# ---------------------------------------------------------------------------


def duration_accounting(
    events: Sequence[EvaluableEvent],
    scores: Mapping[str, float],
    row_times: Mapping[str, tuple[float, float]],
    threshold: float,
    *,
    excluded_events: int,
) -> dict[str, object]:
    """Bounded v1 duration accounting for evaluable P+W events at one threshold.

    Fixed before any value was observed. For every evaluable P+W event the
    frozen alert rule (``file score >= threshold``) marks recalled events; the
    declared degradation duration comes from the accepted ledger ``duration_d``
    and the detected span is the distance between the first and last flagged file
    end inside the event's positive window. Nothing is imputed: undefined
    medians are ``null`` and every exclusion is counted.

    ``episode_grouping_days`` records the inherited ``<= 2``-day alert-episode
    grouping (``synth.events.EPISODE_GAP_S``) used by the FAR path, so duration
    and FAR share one grouping definition.
    """
    _finite(threshold, "duration threshold")
    evaluated = 0
    recalled = 0
    leads: list[float] = []
    spans: list[float] = []
    declared: list[float] = []
    without_positive_duration = 0
    for event in _positive_events_of(events):
        evaluated += 1
        flagged = sorted(
            row_times[file_id][1]
            for file_id in event.members
            if float(scores[file_id]) >= float(threshold)
        )
        duration = event.duration_days
        if duration is None or not math.isfinite(duration) or duration <= 0.0:
            without_positive_duration += 1
        else:
            declared.append(float(duration))
        if flagged:
            recalled += 1
            leads.append(float(E.lead_days(event.failure_time, flagged)))
            spans.append(float((flagged[-1] - flagged[0]) / 86400.0))
    return {
        "schema_id": DURATION_ACCOUNTING_ID,
        "evaluated_events": evaluated,
        "recalled_events": recalled,
        "unrecalled_events": evaluated - recalled,
        "median_declared_duration_days": float(np.median(declared)) if declared else None,
        "median_detected_alert_span_days": float(np.median(spans)) if spans else None,
        "median_first_alert_lead_days": float(np.median(leads)) if leads else None,
        "total_detected_alert_days": float(sum(spans)),
        "episode_grouping_days": float(E.EPISODE_GAP_S / 86400.0),
        "events_without_positive_duration": without_positive_duration,
        "excluded_events": _require_count(excluded_events, "excluded_events"),
    }


# ---------------------------------------------------------------------------
# Deterministic aggregation
# ---------------------------------------------------------------------------


def history_block_interval(
    values: Sequence[float],
    *,
    seed: int = P.BOOTSTRAP_SEED,
    replicates: int = P.BOOTSTRAP_REPLICATES,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
) -> list[float]:
    """History-block percentile interval of the four-history macro.

    Protocol §8.2: resample the four history units with replacement, take the
    macro (mean), ``B=2000``, seed ``20260202``, NumPy linear quantiles. The
    draw construction matches ``probe15.history_block_lcb`` so the lower bound
    is the same frozen quantity. Requires exactly four finite history units.
    """
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size != 4:
        raise ValueError("history-block uncertainty requires exactly four history units")
    if not np.isfinite(array).all():
        raise ValueError("history-block uncertainty requires finite history values")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(replicates, array.size))
    macros = array[draws].mean(axis=1)
    return [
        float(np.quantile(macros, lower_percentile / 100.0)),
        float(np.quantile(macros, upper_percentile / 100.0)),
    ]


# ---------------------------------------------------------------------------
# Per-history / per-branch metric assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HistoryEvaluationInput:
    """One Development history's frozen inputs for supplementary metric assembly."""

    history_id: str
    schema_history_id: str
    role_manifest_sha256: str
    support_sha256: str
    rows: tuple[Mapping[str, object], ...]
    ledger: tuple[Mapping[str, object], ...]
    windows: Mapping[str, Sequence[Sequence[float]]]
    scores: Mapping[str, float]
    localization: Mapping[str, Mapping[str, object]]
    fit_healthy_scores: tuple[float, ...]
    excluded_rows: int


def row_times(rows: Sequence[Mapping[str, object]]) -> dict[str, tuple[float, float]]:
    times: dict[str, tuple[float, float]] = {}
    for row in rows:
        file_id = str(row["file_id"])
        times[file_id] = (float(cast(float, row["start_time"])), float(cast(float, row["end_time"])))
    return times


def _baseline_value(baseline: Mapping[str, object] | None, name: str, *, label: str) -> float | None:
    """Return the same-seed/same-history B0 value, or ``None`` when absent.

    A missing baseline yields explicit ``null`` deltas (and a non-VALID result);
    it never fabricates a paired comparison.
    """
    if baseline is None:
        return None
    value = baseline.get(name)
    return _finite(value, f"baseline {label}")


def build_history_metric_row(
    history: HistoryEvaluationInput,
    *,
    threshold: float,
    canonical_entry: Mapping[str, object],
    baseline: Mapping[str, object] | None,
) -> dict[str, object]:
    """Assemble one supplementary per-history row.

    The canonical event metrics come verbatim from ``probe15.evaluate_histories``
    (the unchanged frozen producer); the five supplementary families are added;
    paired deltas are arm minus the same-seed/same-history B0 canonical values.
    """
    support = evaluable_support(history.rows, history.ledger, history.windows)
    scores = dict(history.scores)
    times = row_times(history.rows)
    control_scores = _control_member_scores(support, scores)

    rate, _hits, evaluated, loc_excluded = localization_rate(
        support.events, scores, history.localization, times
    )
    severity = severity_ordering(support.events, scores)
    slices = subtype_slices(support.events, scores, control_scores)
    stability = background_stability_ratio(control_scores, history.fit_healthy_scores)
    duration = duration_accounting(
        support.events, scores, times, threshold, excluded_events=support.skipped
    )

    def canonical(name: str) -> float:
        return _finite(canonical_entry[name], name)

    base_p = _baseline_value(baseline, "P_AUROC", label="P_AUROC")
    base_w = _baseline_value(baseline, "W_AUROC", label="W_AUROC")
    base_pw = _baseline_value(baseline, "PW_AUROC", label="PW_AUROC")
    base_far = _baseline_value(baseline, "FAR_per_robot_day", label="FAR_per_robot_day")
    delta_p = None if base_p is None else canonical("auc_p") - base_p
    delta_w = None if base_w is None else canonical("auc_w") - base_w
    delta_pw = None if base_pw is None else canonical("auc_pw") - base_pw
    delta_far = None if base_far is None else canonical("far") - base_far

    row: dict[str, object] = {
        "history_id": history.schema_history_id,
        "P_AUROC": canonical("auc_p"),
        "W_AUROC": canonical("auc_w"),
        "PW_AUROC": canonical("auc_pw"),
        "A_AUROC": canonical("auc_a"),
        "delta_P": delta_p,
        "delta_W": delta_w,
        "delta_PW": delta_pw,
        "P_recall": canonical("recall_p"),
        "P_median_lead_days": canonical("lead_p_median"),
        "W_recall": canonical("recall_w"),
        "W_median_lead_days": canonical("lead_w_median"),
        "FAR_per_robot_day": canonical("far"),
        "delta_FAR": delta_far,
        "background_stability_ratio": stability,
        "localization": rate,
        "severity_ordering": severity,
        "subtype_slices": slices,
        "duration_accounting": duration,
    }
    if set(row) != set(SUPPLEMENTARY_HISTORY_KEYS):
        raise ValueError("supplementary history row does not cover the closed metric coverage")
    if evaluated and loc_excluded == evaluated:
        # Every event was unevaluable: report the exclusion, never a 0.0 rate.
        row["localization"] = None
    return row


def _seed_averaged_row(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Average one history's supplementary rows across the three model seeds.

    Scalar metrics average; the nested ordering/slice maps average per key;
    ``duration_accounting`` stays in the per-seed cells because it is a counted
    per-execution record rather than an averageable rank. The averaged cell is
    exactly the frozen v2 metric row, so the unchanged v2 validator consumes it
    without reinterpretation.
    """
    if len(rows) != len(MODEL_SEEDS):
        raise ValueError("seed averaging requires exactly three seed rows")
    averaged: dict[str, object] = {}
    for name in _METRIC_KEYS - {"history_id"}:
        if name == "severity_ordering":
            keys = ("P", "W")
        elif name == "subtype_slices":
            keys = ("P1", "P2", "W1", "W2", "A1", "A2")
        else:
            keys = None
        if keys is not None:
            cell: dict[str, float | None] = {}
            for key in keys:
                values = [cast(Mapping[str, object], row[name])[key] for row in rows]
                cell[key] = (
                    float(np.mean([_finite(value, f"{name}.{key}") for value in values]))
                    if all(value is not None for value in values)
                    else None
                )
            averaged[name] = cell
        else:
            values = [row[name] for row in rows]
            averaged[name] = (
                float(np.mean([_finite(value, name) for value in values]))
                if all(value is not None for value in values)
                else None
            )
    averaged["history_id"] = rows[0]["history_id"]
    return {name: averaged[name] for name in _METRIC_KEYS}


def build_branch_metrics(
    *,
    thresholds_by_seed: Sequence[float],
    per_seed_rows: Sequence[Mapping[str, Mapping[str, object]]],
    baseline_per_seed_rows: Sequence[Mapping[str, Mapping[str, object]]] | None,
) -> dict[str, object]:
    """Return the four closed branch sections for one score branch.

    ``per_seed`` keeps every raw 3×4 seed/history cell (each with the additive
    duration record); ``per_history`` holds the four seed-averaged gate cells in
    exactly v2 shape; ``aggregate`` applies protocol §8.2 arithmetic to those
    seed-averaged cells. The background interval is the frozen history-block
    bootstrap of the four seed-averaged stability ratios.
    """
    if len(thresholds_by_seed) != len(MODEL_SEEDS):
        raise ValueError("supplementary branch requires one frozen q95 threshold per seed")
    for index, value in enumerate(thresholds_by_seed):
        _finite(value, f"thresholds_by_seed[{index}]")
    if len(per_seed_rows) != len(MODEL_SEEDS):
        raise ValueError("supplementary aggregation requires exactly three seeds")
    if baseline_per_seed_rows is not None and len(baseline_per_seed_rows) != len(MODEL_SEEDS):
        raise ValueError("paired aggregation requires the three B0 baseline seed rows")
    history_ids: tuple[str, ...] | None = None
    for seed_rows in per_seed_rows:
        ids = tuple(seed_rows)
        if len(ids) != 4:
            raise ValueError("supplementary aggregation requires exactly four histories per seed")
        if history_ids is None:
            history_ids = ids
        elif ids != history_ids:
            raise ValueError("supplementary seed rows differ in history order")
    assert history_ids is not None
    per_seed: list[dict[str, object]] = []
    for model_seed, seed_rows in zip(MODEL_SEEDS, per_seed_rows):
        rows = [dict(seed_rows[history_id]) for history_id in history_ids]
        if any(set(row) != set(SUPPLEMENTARY_HISTORY_KEYS) for row in rows):
            raise ValueError("per-seed supplementary rows do not have the closed keys")
        per_seed.append({"model_seed": model_seed, "per_history": rows})
    averaged = [
        _seed_averaged_row([seed_rows[history_id] for seed_rows in per_seed_rows])
        for history_id in history_ids
    ]

    def averaged_value(name: str) -> list[float | None]:
        return [cast(float | None, row[name]) for row in averaged]

    pw_values = averaged_value("PW_AUROC")
    if any(value is None for value in pw_values):
        raise ValueError("PW AUROC is undefined on a declared Development history")
    stability_values = averaged_value("background_stability_ratio")
    delta_pw_values = averaged_value("delta_PW")

    def macro(name: str) -> float | None:
        values = averaged_value(name)
        if any(value is None for value in values):
            return None
        return float(np.mean([cast(float, value) for value in values]))

    aggregate = {
        "P_macro": macro("P_AUROC"),
        "W_macro": macro("W_AUROC"),
        "PW_macro": macro("PW_AUROC"),
        "PW_lcb95": float(P.history_block_lcb([cast(float, value) for value in pw_values])["lcb"]),
        "delta_P_macro": macro("delta_P"),
        "delta_W_macro": macro("delta_W"),
        "delta_PW_macro": macro("delta_PW"),
        "directional_count": sum(1 for value in pw_values if cast(float, value) > M.DIRECTIONAL_FLOOR),
        "positive_delta_history_count": (
            sum(1 for value in delta_pw_values if value is not None and cast(float, value) > 0.0)
            if all(value is not None for value in delta_pw_values)
            else None
        ),
        "background_uncertainty": (
            history_block_interval([cast(float, value) for value in stability_values])
            if all(value is not None for value in stability_values)
            else [None, None]
        ),
    }
    if set(aggregate) != set(_AGGREGATE_KEYS):
        raise ValueError("supplementary aggregate does not cover the frozen aggregate keys")
    return {
        "thresholds_by_seed": [float(value) for value in thresholds_by_seed],
        "per_seed": per_seed,
        "per_history": averaged,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Schema validation (closed, additive, v2-compatible projection)
# ---------------------------------------------------------------------------


def _validate_duration_accounting(value: object, label: str) -> None:
    row = _closed_keys(value, DURATION_ACCOUNTING_KEYS, label)
    if row["schema_id"] != DURATION_ACCOUNTING_ID:
        raise ValueError(f"{label}.schema_id must be exactly {DURATION_ACCOUNTING_ID}")
    for name in ("evaluated_events", "recalled_events", "unrecalled_events", "events_without_positive_duration", "excluded_events"):
        _require_count(row[name], f"{label}.{name}")
    if cast(int, row["recalled_events"]) + cast(int, row["unrecalled_events"]) != cast(int, row["evaluated_events"]):
        raise ValueError(f"{label} recalled/unrecalled counts do not sum to the evaluated events")
    for name in ("median_declared_duration_days", "median_detected_alert_span_days", "median_first_alert_lead_days"):
        _optional_finite(row[name], f"{label}.{name}")
    _finite(row["total_detected_alert_days"], f"{label}.total_detected_alert_days")
    if float(cast(float, row["episode_grouping_days"])) != float(E.EPISODE_GAP_S / 86400.0):
        raise ValueError(f"{label}.episode_grouping_days differs from the frozen episode grouping")


def supplementary_v2_projection(row: Mapping[str, object]) -> dict[str, object]:
    """Project one supplementary row onto the frozen v2 metric-row shape."""
    if set(row) != set(SUPPLEMENTARY_HISTORY_KEYS):
        raise ValueError("supplementary history row does not have the closed supplementary keys")
    return {name: row[name] for name in _METRIC_KEYS}


def _validate_branch(value: object, label: str) -> None:
    branch = _closed_keys(value, SUPPLEMENTARY_BRANCH_KEYS, label)
    thresholds = branch["thresholds_by_seed"]
    if not isinstance(thresholds, list) or len(thresholds) != len(MODEL_SEEDS):
        raise ValueError(f"{label}.thresholds_by_seed must have three entries")
    for index, item in enumerate(thresholds):
        _optional_finite(item, f"{label}.thresholds_by_seed[{index}]")
    expected_ids = [f"H-S18-DEV-{index:02d}" for index in range(1, 5)]
    per_seed = branch["per_seed"]
    if not isinstance(per_seed, list) or len(per_seed) != len(MODEL_SEEDS):
        raise ValueError(f"{label}.per_seed must contain exactly three model seeds")
    for index, item in enumerate(per_seed):
        seed_row = _closed_keys(item, SUPPLEMENTARY_SEED_KEYS, f"{label}.per_seed[{index}]")
        if seed_row["model_seed"] != MODEL_SEEDS[index]:
            raise ValueError(f"{label}.per_seed seed order is not canonical")
        rows = seed_row["per_history"]
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError(f"{label}.per_seed[{index}].per_history must have four histories")
        seen: list[str] = []
        for row_index, row_value in enumerate(rows):
            row = _closed_keys(
                row_value, SUPPLEMENTARY_HISTORY_KEYS, f"{label}.per_seed[{index}].per_history[{row_index}]"
            )
            _validate_history_metric(
                supplementary_v2_projection(row),
                f"{label}.per_seed[{index}].per_history[{row_index}]",
                _PHASE,
            )
            _validate_duration_accounting(
                row["duration_accounting"],
                f"{label}.per_seed[{index}].per_history[{row_index}].duration_accounting",
            )
            seen.append(cast(str, row["history_id"]))
        if seen != expected_ids:
            raise ValueError(f"{label}.per_seed[{index}] is not the canonical four-history order")
    averaged = branch["per_history"]
    if not isinstance(averaged, list) or len(averaged) != 4:
        raise ValueError(f"{label}.per_history must contain exactly four histories")
    averaged_ids: list[str] = []
    for index, item in enumerate(averaged):
        # The seed-averaged gate cell is exactly the frozen v2 metric row.
        row = _closed_keys(item, SUPPLEMENTARY_AVERAGED_ROW_KEYS, f"{label}.per_history[{index}]")
        averaged_ids.append(
            _validate_history_metric(row, f"{label}.per_history[{index}]", _PHASE)
        )
    if averaged_ids != expected_ids:
        raise ValueError(f"{label}.per_history is not the canonical four-history order")
    aggregate = _closed_keys(branch["aggregate"], _AGGREGATE_KEYS, f"{label}.aggregate")
    for name in _AGGREGATE_KEYS - {"directional_count", "positive_delta_history_count", "background_uncertainty"}:
        _optional_finite(aggregate[name], f"{label}.aggregate.{name}")
    for name in ("directional_count", "positive_delta_history_count"):
        if aggregate[name] is not None:
            _require_count(aggregate[name], f"{label}.aggregate.{name}")
            if cast(int, aggregate[name]) > 4:
                raise ValueError(f"{label}.aggregate.{name} exceeds the four histories")
    uncertainty = aggregate["background_uncertainty"]
    if not isinstance(uncertainty, list) or len(uncertainty) != 2:
        raise ValueError(f"{label}.aggregate.background_uncertainty must have two entries")
    for index, item in enumerate(uncertainty):
        _optional_finite(item, f"{label}.aggregate.background_uncertainty[{index}]")


def validate_supplementary_result(result: Mapping[str, object]) -> Mapping[str, object]:
    """Validate the closed supplementary v1 document and its v2 projection.

    Additive-versioning proof: every per-history row's v2 projection must pass
    the unchanged frozen v2 history validator, while the complete row (with
    ``duration_accounting``) must still be refused by it. Historical v2 rows are
    therefore never reinterpreted.
    """
    row = _closed_keys(result, SUPPLEMENTARY_RESULT_KEYS, "supplementary result")
    if row["schema_id"] != SUPPLEMENTARY_SCHEMA_ID or row["schema_version"] != SUPPLEMENTARY_SCHEMA_VERSION:
        raise ValueError(f"supplementary result identity must be exactly {SUPPLEMENTARY_SCHEMA_ID} v1")
    if row["protocol_id"] != PROTOCOL_ID or row["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("supplementary result must bind the accepted v2 protocol bytes")
    if row["phase"] != _PHASE:
        raise ValueError("supplementary metrics exist only for the Development phase")
    arm_id = row["arm_id"]
    if arm_id not in SUPPLEMENTARY_VALID_ARMS:
        raise ValueError("supplementary metrics exist only for the seven executable arms")
    if row["model_seed"] not in MODEL_SEEDS:
        raise ValueError("supplementary model seed is outside the frozen namespace")
    status = row["status"]
    if status not in ("VALID", "FAILED", "INVALID"):
        raise ValueError("supplementary status is outside the frozen three-state enum")
    failures = row["failure_reasons"]
    if not isinstance(failures, list) or any(not isinstance(reason, str) or not reason for reason in failures):
        raise ValueError("failure_reasons must be a list of non-empty strings")
    if len(set(failures)) != len(failures):
        raise ValueError("failure_reasons must be unique")
    if status == "VALID" and failures:
        raise ValueError("a VALID supplementary result must state no failure reason")
    if status in ("FAILED", "INVALID") and not failures:
        raise ValueError("FAILED/INVALID supplementary results must state a reason")
    execution = _closed_keys(row["execution"], SUPPLEMENTARY_EXECUTION_KEYS, "supplementary execution")
    _validate_execution_paths(execution, "supplementary execution")
    for name in ("metric_code_sha256", "supplementary_definition_sha256"):
        _require_hash(execution[name], f"supplementary execution.{name}")
    if execution["phase"] != _PHASE or execution["scope"] != "experiment":
        raise ValueError("supplementary execution must be an experiment-scope Development execution")
    inputs = _closed_keys(row["inputs"], SUPPLEMENTARY_INPUTS_KEYS, "supplementary inputs")
    for name in (
        "manifest_sha256",
        "checkpoint_sha256",
        "accepted_record_sha256",
        "task14_lock_sha256",
        "handoff_sha256",
    ):
        _require_hash(inputs[name], f"supplementary inputs.{name}")
    if not isinstance(inputs["data_root"], str) or not inputs["data_root"]:
        raise ValueError("supplementary inputs.data_root must be a non-empty path")
    for name in ("reference_artifact_sha256",):
        value = inputs[name]
        if value is not None:
            _require_hash(value, f"supplementary inputs.{name}")
    for name in ("seed_score_documents", "baseline_score_documents"):
        documents = inputs[name]
        if not isinstance(documents, list):
            raise ValueError(f"supplementary inputs.{name} must be a list")
        for index, item in enumerate(documents):
            record = _closed_keys(item, frozenset({"path", "sha256"}), f"supplementary inputs.{name}[{index}]")
            if not isinstance(record["path"], str) or not record["path"]:
                raise ValueError(f"supplementary inputs.{name}[{index}].path must be a non-empty string")
            _require_hash(record["sha256"], f"supplementary inputs.{name}[{index}].sha256")
    if arm_id == "B0-FULL" and len(inputs["baseline_score_documents"]) != 0:
        raise ValueError("the B0 supplementary result is its own baseline")
    if len(inputs["seed_score_documents"]) != len(MODEL_SEEDS):
        raise ValueError("supplementary inputs must bind the three seed score documents")
    before = _require_digest_map(inputs["sha256_before"], "supplementary inputs.sha256_before")
    after = _require_digest_map(inputs["sha256_after"], "supplementary inputs.sha256_after")
    if dict(before) != dict(after):
        raise ValueError("immutable input digest proof is missing or changed")
    if inputs["unchanged"] is not True:
        raise ValueError("immutable inputs must be unchanged for a supplementary result")
    reproduction = _closed_keys(row["reproduction"], SUPPLEMENTARY_REPRODUCTION_KEYS, "supplementary reproduction")
    _require_hash(reproduction["data_binding_sha256"], "supplementary reproduction.data_binding_sha256")
    for name in (
        "arm_config_sha256_by_seed",
        "model_state_sha256_by_seed",
        "ema_state_sha256_by_seed",
        "bank_sha256_by_seed",
        "accepted_bank_sha256_by_seed",
    ):
        values = reproduction[name]
        if not isinstance(values, list) or len(values) != len(MODEL_SEEDS):
            raise ValueError(f"supplementary reproduction.{name} must have one digest per seed")
        for index, value in enumerate(values):
            _require_hash(value, f"supplementary reproduction.{name}[{index}]")
    if list(reproduction["bank_sha256_by_seed"]) != list(reproduction["accepted_bank_sha256_by_seed"]):
        raise ValueError("supplementary bank bytes do not reproduce the accepted bank digests")
    if reproduction["bank_reproduced"] is not True:
        raise ValueError("supplementary run must reproduce the accepted reference-bank digest")
    if reproduction["bank_source"] not in REFERENCE_STATE_MODES:
        raise ValueError("supplementary bank_source is not a frozen reference-state mode")
    modes = reproduction["reference_state_modes"]
    if not isinstance(modes, Mapping) or set(modes) != set(SUPPLEMENTARY_BRANCHES):
        raise ValueError("reference_state_modes must name both score branches")
    for name, mode in modes.items():
        if mode not in REFERENCE_STATE_MODES:
            raise ValueError(f"reference_state_modes.{name} is not a frozen reference-state mode")
    if (
        reproduction["representation_state_unchanged"] is not True
        or reproduction["reference_state_unchanged"] is not True
        or reproduction["encoder_updates"] != 0
    ):
        raise ValueError("supplementary metrics require unchanged model, EMA, and reference state")
    thresholds = reproduction["calibration_thresholds_by_seed"]
    accepted_thresholds = reproduction["accepted_calibration_thresholds_by_seed"]
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(SUPPLEMENTARY_BRANCHES):
        raise ValueError("calibration thresholds must cover both score branches")
    if not isinstance(accepted_thresholds, Mapping) or dict(thresholds) != dict(accepted_thresholds):
        raise ValueError("supplementary q95 thresholds do not reproduce the accepted thresholds")
    if reproduction["calibration_reproduced"] is not True:
        raise ValueError("supplementary run must reproduce the accepted q95 thresholds")
    recomputed = reproduction["recomputed_calibration_thresholds_by_seed"]
    if not isinstance(recomputed, Mapping) or set(recomputed) != set(SUPPLEMENTARY_BRANCHES):
        raise ValueError("recomputed_calibration_thresholds_by_seed must cover both branches")
    for name in SUPPLEMENTARY_BRANCHES:
        values = thresholds[name]
        if not isinstance(values, list) or len(values) != len(MODEL_SEEDS):
            raise ValueError(f"calibration_thresholds.{name} must have one value per seed")
        for index, item in enumerate(values):
            _finite(item, f"calibration_thresholds.{name}[{index}]")
        observed = recomputed[name]
        if not isinstance(observed, list) or len(observed) != len(MODEL_SEEDS):
            raise ValueError(f"recomputed_calibration_thresholds_by_seed.{name} must have one value per seed")
        for index, item in enumerate(observed):
            _finite(item, f"recomputed_calibration_thresholds_by_seed.{name}[{index}]")
    support = _closed_keys(row["support"], SUPPLEMENTARY_SUPPORT_KEYS, "supplementary support")
    for name in ("fit_healthy_rows_sha256", "fit_healthy_scores_sha256", "calibration_support_sha256", "calibration_scores_sha256"):
        _require_hash(support[name], f"supplementary support.{name}")
    for name in ("fit_healthy_rows", "calibration_healthy_rows", "excluded_rows"):
        _require_count(support[name], f"supplementary support.{name}")
    development = support["development"]
    if not isinstance(development, list) or len(development) != 4:
        raise ValueError("supplementary support must declare the four Development histories")
    for index, item in enumerate(development):
        entry = _closed_keys(item, SUPPLEMENTARY_DEVELOPMENT_SUPPORT_KEYS, f"supplementary support.development[{index}]")
        for name in ("manifest_sha256", "support_sha256", "scores_sha256"):
            _require_hash(entry[name], f"supplementary support.development[{index}].{name}")
        for name in ("rows", "excluded_rows", "p", "w", "a", "controls", "evaluated_pw_events", "localization_excluded_events"):
            _require_count(entry[name], f"supplementary support.development[{index}].{name}")
        _finite(entry["robot_days"], f"supplementary support.development[{index}].robot_days")
    definitions = _closed_keys(row["metric_definitions"], _METRIC_DEFINITION_KEYS, "supplementary metric_definitions")
    expected_definitions = {
        "localization": LOCALIZATION_ID,
        "duration_accounting": DURATION_ACCOUNTING_ID,
        "severity_ordering": SEVERITY_ID,
        "subtype_slices": SUBTYPE_SLICE_ID,
        "background_stability": STABILITY_ID,
        "background_uncertainty": UNCERTAINTY_ID,
        "event_metrics": P.PROBE_ID,
    }
    if dict(definitions) != expected_definitions:
        raise ValueError("supplementary metric definitions differ from the frozen supplement")
    branches = _closed_keys(row["branches"], frozenset(SUPPLEMENTARY_BRANCHES), "supplementary branches")
    for name in SUPPLEMENTARY_BRANCHES:
        _validate_branch(branches[name], f"supplementary branches.{name}")
    # Support accounting and the duration record must describe the same event set.
    support_events: dict[str, int] = {}
    for index, item in enumerate(cast(list[object], support["development"])):
        entry = _closed_keys(
            item, SUPPLEMENTARY_DEVELOPMENT_SUPPORT_KEYS, f"supplementary support.development[{index}]"
        )
        support_events[str(entry["schema_history_id"])] = int(entry["evaluated_pw_events"])
    for name in SUPPLEMENTARY_BRANCHES:
        branch = cast(Mapping[str, object], branches[name])
        for seed_row in cast(list[object], branch["per_seed"]):
            for metric in cast(list[object], cast(Mapping[str, object], seed_row)["per_history"]):
                history_id = cast(str, cast(Mapping[str, object], metric)["history_id"])
                duration = cast(Mapping[str, object], cast(Mapping[str, object], metric)["duration_accounting"])
                if int(duration["evaluated_events"]) != support_events[history_id]:
                    raise ValueError(
                        f"duration accounting for {history_id} disagrees with the declared "
                        "evaluated P+W event support"
                    )
    if status == "VALID":
        # A VALID non-B0 result must bind its three authorised B0 baseline
        # documents; a non-VALID result may report null paired deltas with the
        # missing baseline stated as a failure reason.
        if arm_id != "B0-FULL" and len(inputs["baseline_score_documents"]) != len(MODEL_SEEDS):
            raise ValueError("a VALID non-B0 supplementary result requires the three B0 baseline score documents")
        for name in SUPPLEMENTARY_BRANCHES:
            branch = cast(Mapping[str, object], branches[name])
            for seed_index, seed_item in enumerate(cast(list[object], branch["per_seed"])):
                seed_row = cast(Mapping[str, object], seed_item)
                for row_index, item in enumerate(cast(list[object], seed_row["per_history"])):
                    metric = cast(Mapping[str, object], item)
                    for field in _METRIC_KEYS - {"history_id"}:
                        value = metric[field]
                        if field in ("severity_ordering", "subtype_slices"):
                            if any(item_value is None for item_value in cast(Mapping[str, object], value).values()):
                                raise ValueError(f"VALID supplementary result is missing {field}")
                        elif value is None:
                            raise ValueError(f"VALID supplementary result is missing {field} on its fixed support")
                    duration = cast(Mapping[str, object], metric["duration_accounting"])
                    if cast(int, duration["evaluated_events"]) == 0:
                        raise ValueError("VALID supplementary result has no evaluable P+W events")
                    for field in ("median_declared_duration_days", "median_detected_alert_span_days", "median_first_alert_lead_days"):
                        if duration[field] is None:
                            raise ValueError(f"VALID supplementary result is missing duration_accounting.{field}")
            for index, item in enumerate(cast(list[object], branch["per_history"])):
                metric = cast(Mapping[str, object], item)
                for field in _METRIC_KEYS - {"history_id"}:
                    value = metric[field]
                    if field in ("severity_ordering", "subtype_slices"):
                        if any(item_value is None for item_value in cast(Mapping[str, object], value).values()):
                            raise ValueError(f"VALID supplementary result is missing seed-averaged {field}")
                    elif value is None:
                        raise ValueError(f"VALID supplementary result is missing seed-averaged {field}")
            aggregate = cast(Mapping[str, object], branch["aggregate"])
            for field in _AGGREGATE_KEYS:
                value = aggregate[field]
                if value is None or (
                    field == "background_uncertainty"
                    and any(item_value is None for item_value in cast(list[object], value))
                ):
                    raise ValueError(f"VALID supplementary result is missing aggregate {field}")
    return row


def _assert_v2_refuses_supplementary_row(row: Mapping[str, object]) -> None:
    """Prove the frozen v2 validator does not silently accept the additive row."""
    try:
        _validate_history_metric(row, "supplementary row", _PHASE)
    except ValueError:
        return
    raise ValueError("the frozen v2 validator unexpectedly accepted a duration_accounting row")


# The refusal proof runs on module import for the closed key sets, so a future
# edit that quietly widens the v2 shape fails immediately.
_probe_row = {name: None for name in SUPPLEMENTARY_HISTORY_KEYS}
_probe_row["history_id"] = "H-S18-DEV-01"
_assert_v2_refuses_supplementary_row(_probe_row)
del _probe_row


# ---------------------------------------------------------------------------
# Frozen input manifest (custody) validation and immutability proof
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrozenSupplementaryInputs:
    """Validated custody manifest for one supplementary arm/seed execution."""

    manifest_path: str
    manifest_sha256: str
    arm_id: str
    model_seed: int
    source_commit: str
    protocol_sha256: str
    metric_code_sha256: str
    supplementary_definition_sha256: str
    task_id: str
    scope: str
    phase: str
    data_root: str
    task14_lock_path: str
    task14_lock_sha256: str
    handoff_path: str
    handoff_sha256: str
    accepted_record_path: str
    accepted_record_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    reference_artifact: Mapping[str, object]
    expected: Mapping[str, object]
    document: Mapping[str, object]


_INPUT_MANIFEST_KEYS = frozenset(
    {
        "schema_id",
        "protocol_id",
        "protocol_sha256",
        "supplementary_definition_sha256",
        "metric_code_sha256",
        "source_commit",
        "task_id",
        "phase",
        "scope",
        "data_root",
        "arm_id",
        "model_seed",
        "task14_lock",
        "handoff",
        "accepted_record",
        "checkpoint",
        "reference_artifact",
        "expected",
    }
)

_EXPECTED_KEYS = frozenset(
    {
        "arm_config_sha256",
        "checkpoint_arm_config_sha256",
        "data_binding_sha256",
        "model_state_sha256",
        "ema_state_sha256",
        "bank_sha256",
        "bank_source_rows",
        "bank_source_rows_sha256",
        "bank_recipe_id",
        "bank_order_id",
        "calibration_thresholds",
        "development_support_sha256",
        "development_manifest_sha256",
    }
)


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _custody_file(record: object, label: str) -> tuple[str, str]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an object with exactly path and sha256")
    path = record["path"]
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label}.path must be a non-empty string")
    digest = _require_hash(record["sha256"], f"{label}.sha256")
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    observed = _hash_file(Path(path))
    if observed != digest:
        raise ValueError(f"{label} digest mismatch: {path}")
    return path, digest


def load_frozen_input_manifest(path: str | Path) -> FrozenSupplementaryInputs:
    """Load and fully validate one supplementary custody manifest.

    Every referenced file is re-hashed; the declared metric-code digest must
    match this module's bytes; only the seven executable arms, the Development
    phase, and experiment scope are accepted. Any mismatch aborts before a single
    inference call.
    """
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    manifest_sha = sha256(raw).hexdigest()
    document = json.loads(raw)
    if not isinstance(document, Mapping):
        raise ValueError("supplementary input manifest must be a JSON object")
    row = _closed_keys(document, _INPUT_MANIFEST_KEYS, "supplementary input manifest")
    if row["schema_id"] != SUPPLEMENTARY_INPUT_SCHEMA_ID:
        raise ValueError(f"supplementary input manifest schema must be exactly {SUPPLEMENTARY_INPUT_SCHEMA_ID}")
    if row["protocol_id"] != PROTOCOL_ID or row["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("supplementary input manifest must bind the accepted v2 protocol bytes")
    if row["phase"] != _PHASE or row["scope"] != "experiment":
        raise ValueError("supplementary input manifest is Development-only and experiment-scoped")
    arm_id = row["arm_id"]
    if arm_id not in SUPPLEMENTARY_VALID_ARMS:
        raise ValueError("supplementary input manifest arm is not one of the seven executable arms")
    model_seed = row["model_seed"]
    if model_seed not in MODEL_SEEDS:
        raise ValueError("supplementary input manifest model seed is outside the frozen namespace")
    declared_code = _require_hash(row["metric_code_sha256"], "metric_code_sha256")
    if declared_code != metric_code_sha256():
        raise ValueError("supplementary input manifest was built for different metric-code bytes")
    definition = _require_hash(
        row["supplementary_definition_sha256"], "supplementary_definition_sha256"
    )
    source_commit = row["source_commit"]
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("source_commit must be a 40-character commit identity")
    task_id = row["task_id"]
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be a non-empty string")
    data_root = row["data_root"]
    if not isinstance(data_root, str) or not data_root:
        raise ValueError("data_root must be a non-empty path")
    if not Path(data_root).is_dir():
        raise FileNotFoundError(f"certified data root does not exist: {data_root}")
    task14_path, task14_sha = _custody_file(row["task14_lock"], "task14_lock")
    handoff_path, handoff_sha = _custody_file(row["handoff"], "handoff")
    accepted_path, accepted_sha = _custody_file(row["accepted_record"], "accepted_record")
    checkpoint_path, checkpoint_sha = _custody_file(row["checkpoint"], "checkpoint")
    reference = row["reference_artifact"]
    if not isinstance(reference, Mapping) or set(reference) != {"kind", "path", "sha256", "fit_scope", "bounds"}:
        raise ValueError("reference_artifact must be a closed kind/path/sha256/fit_scope/bounds object")
    kind = reference["kind"]
    if kind not in ("k9_downstream", "n4_scale", "none"):
        raise ValueError("reference_artifact.kind is not a frozen supplementary kind")
    ref_path = reference["path"]
    ref_sha = reference["sha256"]
    if kind == "none":
        if ref_path is not None or ref_sha is not None or reference["fit_scope"] is not None or reference["bounds"] is not None:
            raise ValueError("a 'none' reference artifact must not declare path/sha256/scope/bounds")
    else:
        if not isinstance(ref_path, str) or not ref_path:
            raise ValueError("reference_artifact.path must be a non-empty string")
        ref_sha = _require_hash(ref_sha, "reference_artifact.sha256")
        if not Path(ref_path).is_file():
            raise FileNotFoundError(f"reference artifact does not exist: {ref_path}")
        if _hash_file(Path(ref_path)) != ref_sha:
            raise ValueError(f"reference artifact digest mismatch: {ref_path}")
        if kind == "n4_scale":
            if reference["fit_scope"] != "final":
                raise ValueError("the N4 scale envelope must be the final Fit scope")
            bounds = reference["bounds"]
            if not isinstance(bounds, Mapping) or set(bounds) != {"v_floor", "v_cap"}:
                raise ValueError("N4 reference_artifact.bounds must be exactly v_floor/v_cap")
            _finite(bounds["v_floor"], "reference_artifact.bounds.v_floor")
            _finite(bounds["v_cap"], "reference_artifact.bounds.v_cap")
        elif reference["bounds"] is not None:
            raise ValueError("only the N4 scale envelope declares bounds")
    expected = _closed_keys(row["expected"], _EXPECTED_KEYS, "supplementary expected")
    for name in (
        "arm_config_sha256",
        "checkpoint_arm_config_sha256",
        "data_binding_sha256",
        "model_state_sha256",
        "ema_state_sha256",
        "bank_sha256",
        "bank_source_rows_sha256",
    ):
        _require_hash(expected[name], f"expected.{name}")
    bank_rows = expected["bank_source_rows"]
    if isinstance(bank_rows, bool) or not isinstance(bank_rows, int) or bank_rows < 1:
        raise ValueError("expected.bank_source_rows must be a positive integer")
    expected_recipe = K9_BANK_RECIPE_ID if arm_id == "K9-FULL" else BANK_RECIPE_ID
    expected_order = K9_BANK_ORDER_ID if arm_id == "K9-FULL" else BANK_ORDER_ID
    if expected["bank_recipe_id"] != expected_recipe or expected["bank_order_id"] != expected_order:
        raise ValueError("supplementary bank restoration recipe/order is not the frozen accepted recipe")
    thresholds = expected["calibration_thresholds"]
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(SUPPLEMENTARY_BRANCHES):
        raise ValueError("expected.calibration_thresholds must bind both frozen score branches")
    for name in SUPPLEMENTARY_BRANCHES:
        _finite(thresholds[name], f"expected.calibration_thresholds.{name}")
    if arm_id == "K9-FULL" and kind != "k9_downstream":
        raise ValueError("K9 input manifests require the persisted K9 downstream reference artifact")
    if arm_id == "N4-CRN" and kind != "n4_scale":
        raise ValueError("N4 input manifests require the persisted final N4 scale envelope")
    if arm_id not in ("K9-FULL", "N4-CRN") and kind != "none":
        raise ValueError("only K9 and N4 declare a persisted reference artifact")
    development_support = expected["development_support_sha256"]
    development_manifests = expected["development_manifest_sha256"]
    if not isinstance(development_support, Mapping) or len(development_support) != 4:
        raise ValueError("expected.development_support_sha256 must bind the four Development histories")
    if not isinstance(development_manifests, Mapping) or set(development_manifests) != set(development_support):
        raise ValueError("expected.development_manifest_sha256 must bind the same four histories")
    for name in list(development_support) + list(development_manifests):
        _require_hash(development_support[name], f"expected.development_support_sha256.{name}")
        _require_hash(development_manifests[name], f"expected.development_manifest_sha256.{name}")
    return FrozenSupplementaryInputs(
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha,
        arm_id=cast(str, arm_id),
        model_seed=cast(int, model_seed),
        source_commit=source_commit,
        protocol_sha256=PROTOCOL_SHA256,
        metric_code_sha256=declared_code,
        supplementary_definition_sha256=definition,
        task_id=task_id,
        scope="experiment",
        phase=_PHASE,
        data_root=data_root,
        task14_lock_path=task14_path,
        task14_lock_sha256=task14_sha,
        handoff_path=handoff_path,
        handoff_sha256=handoff_sha,
        accepted_record_path=accepted_path,
        accepted_record_sha256=accepted_sha,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        reference_artifact=dict(reference),
        expected=dict(expected),
        document=document,
    )


def compute_input_digests(
    inputs: FrozenSupplementaryInputs,
    *,
    binding=None,
    definition_path: str | Path | None = None,
) -> dict[str, str]:
    """Hash every custody input the supplementary run may read.

    When the certified role binding is supplied, hash its FIT, Calibration and
    Development manifests and shards as well as the direct custody files. The
    score runner calls this before and after inference; equality proves that
    source bytes were unchanged rather than merely repeating manifest claims.
    """
    declared = {
        "frozen_input_manifest": inputs.manifest_sha256,
        "protocol": inputs.protocol_sha256,
        "task14_lock": inputs.task14_lock_sha256,
        "handoff": inputs.handoff_sha256,
        "accepted_record": inputs.accepted_record_sha256,
        "checkpoint": inputs.checkpoint_sha256,
        "metric_code": inputs.metric_code_sha256,
    }
    paths = {
        "frozen_input_manifest": Path(inputs.manifest_path),
        "protocol": PROTOCOL_PATH,
        "task14_lock": Path(inputs.task14_lock_path),
        "handoff": Path(inputs.handoff_path),
        "accepted_record": Path(inputs.accepted_record_path),
        "checkpoint": Path(inputs.checkpoint_path),
    }
    digests = {key: _hash_file(path) for key, path in paths.items()}
    digests["metric_code"] = metric_code_sha256()
    for key, expected in declared.items():
        if digests[key] != expected:
            raise ValueError(f"supplementary immutable input changed: {key}")

    if definition_path is not None:
        actual_definition_path = Path(definition_path).resolve()
        if actual_definition_path != SUPPLEMENTARY_DEFINITION_PATH.resolve():
            raise ValueError("supplementary definition path differs from the frozen source contract")
        digests["supplementary_definition"] = _hash_file(actual_definition_path)
        if digests["supplementary_definition"] != inputs.supplementary_definition_sha256:
            raise ValueError("supplementary definition bytes differ from the frozen manifest")

    reference = inputs.reference_artifact
    if reference["kind"] != "none":
        reference_path = Path(cast(str, reference["path"]))
        reference_sha = _hash_file(reference_path)
        if reference_sha != reference["sha256"]:
            raise ValueError("supplementary reference artifact changed during the run")
        digests["reference_artifact"] = reference_sha

    if binding is not None:
        digests["data_binding"] = binding.executable_binding_sha256
        for role in ("FIT", "CALIBRATION", "DEVELOPMENT"):
            for history in binding.role_histories(role):
                history_id = history.actual_history_id
                role_dir = Path(inputs.data_root) / history.assigned_role / history_id
                manifest_path = role_dir / "manifest.json"
                manifest_sha = _hash_file(manifest_path)
                if manifest_sha != history.manifest_sha256:
                    raise ValueError(f"certified role manifest changed: {history_id}")
                digests[f"data:{role}:{history_id}:manifest"] = manifest_sha
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                shards = manifest.get("shards") if isinstance(manifest, Mapping) else None
                if not isinstance(shards, list):
                    raise ValueError(f"certified role manifest has no shard list: {history_id}")
                for shard in shards:
                    if not isinstance(shard, Mapping) or not isinstance(shard.get("path"), str):
                        raise ValueError(f"certified role manifest has an invalid shard row: {history_id}")
                    relative_path = str(shard["path"])
                    declared_shard_sha = _require_hash(
                        shard.get("sha256"), f"certified shard digest {history_id}/{relative_path}"
                    )
                    shard_sha = _hash_file(role_dir / relative_path)
                    if shard_sha != declared_shard_sha:
                        raise ValueError(f"certified role shard changed: {history_id}/{relative_path}")
                    digests[f"data:{role}:{history_id}:shard:{relative_path}"] = shard_sha
    return digests


def assert_inputs_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> None:
    """Fail closed unless every immutable input digest is identical."""
    if dict(before) != dict(after):
        changed = sorted(
            key for key in set(before) | set(after) if before.get(key) != after.get(key)
        )
        raise ValueError(f"supplementary immutable inputs changed during the run: {changed}")


def _canonical_score_digest(scores: Mapping[str, float]) -> str:
    """Canonical digest over one file→score map (bounded, order-independent)."""
    return _canonical_sha256({str(key): float(value) for key, value in sorted(scores.items())})


def score_map_sha256(scores: Mapping[str, float]) -> str:
    """Public wrapper for the frozen score-map digest used in provenance."""
    return _canonical_score_digest(scores)


# ---------------------------------------------------------------------------
# Per-file score document: the bounded supplementary inference output
# ---------------------------------------------------------------------------

SCORES_RESULT_KEYS = frozenset(
    {
        "schema_id",
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "phase",
        "arm_id",
        "model_seed",
        "execution",
        "inputs",
        "reproduction",
        "support",
        "status",
        "failure_reasons",
    }
)

SCORES_SUPPORT_KEYS = frozenset(
    {
        "fit_healthy_rows",
        "fit_healthy_rows_sha256",
        "fit_healthy_scores",
        "fit_healthy_scores_sha256",
        "calibration_healthy_rows",
        "calibration_support_sha256",
        "calibration_scores",
        "calibration_scores_sha256",
        "thresholds",
        "development",
        "excluded_rows",
    }
)

SCORES_DEVELOPMENT_KEYS = frozenset(
    {
        "history_id",
        "schema_history_id",
        "manifest_sha256",
        "support_sha256",
        "rows",
        "excluded_rows",
        "scores",
        "localization",
        "timesteps_sha256",
        "p",
        "w",
        "a",
        "controls",
        "robot_days",
    }
)

_LOCALIZATION_RECORD_KEYS = frozenset({"n_timesteps", "peak_index", "finite", "constant"})


def _validate_branch_score_map(value: object, label: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty file→score map")
    scores: dict[str, float] = {}
    for file_id, score in value.items():
        if not isinstance(file_id, str) or not file_id:
            raise ValueError(f"{label} contains a malformed file id")
        scores[file_id] = _finite(score, f"{label}[{file_id}]")
    return scores


def _validate_localization_record(value: object, label: str) -> None:
    record = _closed_keys(value, _LOCALIZATION_RECORD_KEYS, label)
    _require_count(record["n_timesteps"], f"{label}.n_timesteps")
    if not isinstance(record["finite"], bool) or not isinstance(record["constant"], bool):
        raise ValueError(f"{label}.finite/constant must be boolean")
    if record["finite"] is False and record["constant"] is not False:
        raise ValueError(f"{label} cannot be constant when it is non-finite")
    peak = record["peak_index"]
    usable = cast(bool, record["finite"]) and not cast(bool, record["constant"])
    if usable:
        _require_count(peak, f"{label}.peak_index")
    elif peak is not None:
        raise ValueError(f"{label}.peak_index must be null for an unusable timestep vector")


def validate_supplementary_scores(result: Mapping[str, object]) -> Mapping[str, object]:
    """Validate one closed per-file score document before any metric consumes it."""
    row = _closed_keys(result, SCORES_RESULT_KEYS, "supplementary scores")
    if row["schema_id"] != SUPPLEMENTARY_SCORES_SCHEMA_ID or row["schema_version"] != 2:
        raise ValueError(f"supplementary scores identity must be exactly {SUPPLEMENTARY_SCORES_SCHEMA_ID} v2")
    if row["protocol_id"] != PROTOCOL_ID or row["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("supplementary scores must bind the accepted v2 protocol bytes")
    if row["phase"] != _PHASE:
        raise ValueError("supplementary scores exist only for the Development phase")
    if row["arm_id"] not in SUPPLEMENTARY_VALID_ARMS:
        raise ValueError("supplementary scores exist only for the seven executable arms")
    if row["model_seed"] not in MODEL_SEEDS:
        raise ValueError("supplementary score seed is outside the frozen namespace")
    status = row["status"]
    if status not in ("VALID", "FAILED", "INVALID"):
        raise ValueError("supplementary score status is outside the frozen enum")
    failures = row["failure_reasons"]
    if not isinstance(failures, list) or len(set(failures)) != len(failures):
        raise ValueError("supplementary score failure_reasons must be a unique list")
    if status == "VALID" and failures:
        raise ValueError("a VALID supplementary score document must state no failure reason")
    if status in ("FAILED", "INVALID") and not failures:
        raise ValueError("a FAILED/INVALID supplementary score document must state a reason")
    execution = _closed_keys(row["execution"], SUPPLEMENTARY_EXECUTION_KEYS, "supplementary scores.execution")
    _validate_execution_paths(execution, "supplementary scores.execution")
    for name in ("metric_code_sha256", "supplementary_definition_sha256"):
        _require_hash(execution[name], f"supplementary scores.execution.{name}")
    if execution["phase"] != _PHASE or execution["scope"] != "experiment":
        raise ValueError("supplementary scores must be an experiment-scope Development execution")
    inputs = _closed_keys(row["inputs"], SUPPLEMENTARY_INPUTS_KEYS, "supplementary scores.inputs")
    for name in ("manifest_sha256", "checkpoint_sha256", "accepted_record_sha256", "task14_lock_sha256", "handoff_sha256"):
        _require_hash(inputs[name], f"supplementary scores.inputs.{name}")
    if not isinstance(inputs["data_root"], str) or not inputs["data_root"]:
        raise ValueError("supplementary scores.inputs.data_root must be a non-empty path")
    if inputs["unchanged"] is not True:
        raise ValueError("supplementary score inputs must be unchanged")
    before = _require_digest_map(inputs["sha256_before"], "supplementary scores.inputs.sha256_before")
    after = _require_digest_map(inputs["sha256_after"], "supplementary scores.inputs.sha256_after")
    if dict(before) != dict(after):
        raise ValueError("supplementary score input digest proof is inconsistent")
    reproduction = _closed_keys(
        row["reproduction"], SCORES_REPRODUCTION_KEYS, "supplementary scores.reproduction"
    )
    for name in (
        "arm_config_sha256",
        "data_binding_sha256",
        "model_state_sha256",
        "ema_state_sha256",
        "bank_sha256",
        "accepted_bank_sha256",
        "bank_source_rows_sha256",
    ):
        _require_hash(reproduction[name], f"supplementary scores.reproduction.{name}")
    bank_rows = reproduction["bank_source_rows"]
    if isinstance(bank_rows, bool) or not isinstance(bank_rows, int) or bank_rows < 1:
        raise ValueError("supplementary scores bank_source_rows must be a positive integer")
    expected_recipe = K9_BANK_RECIPE_ID if row["arm_id"] == "K9-FULL" else BANK_RECIPE_ID
    expected_order = K9_BANK_ORDER_ID if row["arm_id"] == "K9-FULL" else BANK_ORDER_ID
    if (
        reproduction["bank_recipe_id"] != expected_recipe
        or reproduction["bank_order_id"] != expected_order
    ):
        raise ValueError("supplementary score bank recipe/order differs from the frozen arm contract")
    if reproduction["bank_sha256"] != reproduction["accepted_bank_sha256"]:
        raise ValueError("supplementary bank bytes do not reproduce the accepted bank digest")
    if reproduction["bank_reproduced"] is not True:
        raise ValueError("supplementary scoring must reproduce the accepted reference-bank digest")
    expected_source = "restored_from_custody" if row["arm_id"] == "K9-FULL" else "reconstructed_digest_verified"
    if reproduction["bank_source"] != expected_source:
        raise ValueError("supplementary bank source does not match the arm's frozen custody path")
    artifact = _closed_keys(
        reproduction["bank_artifact"],
        {"kind", "path", "sha256", "bank_sha256"},
        "supplementary scores.reproduction.bank_artifact",
    )
    if artifact["kind"] not in (BANK_ARTIFACT_SCHEMA_ID, "s18-k9-downstream-state-v1"):
        raise ValueError("supplementary bank artifact kind is not a frozen bank artifact")
    if not isinstance(artifact["path"], str) or not artifact["path"]:
        raise ValueError("supplementary bank artifact path must be non-empty")
    _require_hash(artifact["sha256"], "supplementary scores.reproduction.bank_artifact.sha256")
    _require_hash(artifact["bank_sha256"], "supplementary scores.reproduction.bank_artifact.bank_sha256")
    if artifact["bank_sha256"] != reproduction["bank_sha256"]:
        raise ValueError("supplementary bank artifact does not bind the reproduced bank digest")
    provenance = _closed_keys(
        reproduction["bank_provenance"],
        {"schema_id", "path", "sha256"},
        "supplementary scores.reproduction.bank_provenance",
    )
    expected_provenance_schema = (
        "s18-k9-downstream-state-v1"
        if row["arm_id"] == "K9-FULL"
        else BANK_PROVENANCE_SCHEMA_ID
    )
    if provenance["schema_id"] != expected_provenance_schema:
        raise ValueError("supplementary bank provenance schema does not match the frozen custody path")
    if not isinstance(provenance["path"], str) or not provenance["path"]:
        raise ValueError("supplementary bank provenance path must be non-empty")
    _require_hash(provenance["sha256"], "supplementary scores.reproduction.bank_provenance.sha256")
    if row["arm_id"] == "K9-FULL":
        if artifact["kind"] != "s18-k9-downstream-state-v1":
            raise ValueError("K9 bank bytes must remain inside the accepted downstream artifact")
    elif artifact["kind"] != BANK_ARTIFACT_SCHEMA_ID:
        raise ValueError("regenerated bank bytes must use the frozen byte-artifact schema")
    if reproduction["bank_source"] not in REFERENCE_STATE_MODES:
        raise ValueError("supplementary scoring bank_source is not a frozen reference-state mode")
    modes = reproduction["reference_state_modes"]
    if not isinstance(modes, Mapping) or set(modes) != set(SUPPLEMENTARY_BRANCHES):
        raise ValueError("supplementary scoring reference_state_modes must name both branches")
    expected_modes = (
        {"S_pred": "restored_from_custody", "S_pop": "restored_from_custody"}
        if row["arm_id"] == "K9-FULL"
        else {"S_pred": "restored_from_custody", "S_pop": "reconstructed_digest_verified"}
    )
    if dict(modes) != expected_modes:
        raise ValueError("supplementary per-branch reference modes do not match the frozen arm contract")
    if (
        reproduction["representation_state_unchanged"] is not True
        or reproduction["reference_state_unchanged"] is not True
        or reproduction["encoder_updates"] != 0
    ):
        raise ValueError("supplementary scoring may never mutate model, EMA, or reference state")
    thresholds = reproduction["calibration_thresholds"]
    accepted = reproduction["accepted_calibration_thresholds"]
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(SUPPLEMENTARY_BRANCHES):
        raise ValueError("supplementary score thresholds must cover both branches")
    if not isinstance(accepted, Mapping) or dict(thresholds) != dict(accepted):
        raise ValueError("supplementary q95 thresholds do not reproduce the accepted thresholds")
    if reproduction["calibration_reproduced"] is not True:
        raise ValueError("supplementary scoring must reproduce the accepted q95 thresholds")
    recomputed = reproduction["recomputed_calibration_thresholds"]
    if not isinstance(recomputed, Mapping) or set(recomputed) != set(SUPPLEMENTARY_BRANCHES):
        raise ValueError("supplementary scores recomputed thresholds must cover both branches")
    for name in SUPPLEMENTARY_BRANCHES:
        _finite(thresholds[name], f"supplementary scores.reproduction.calibration_thresholds.{name}")
        _finite(
            recomputed[name],
            f"supplementary scores.reproduction.recomputed_calibration_thresholds.{name}",
        )
    support = _closed_keys(row["support"], SCORES_SUPPORT_KEYS, "supplementary scores.support")
    for name in ("fit_healthy_rows", "calibration_healthy_rows", "excluded_rows"):
        _require_count(support[name], f"supplementary scores.support.{name}")
    for name in ("fit_healthy_rows_sha256", "calibration_support_sha256"):
        _require_hash(support[name], f"supplementary scores.support.{name}")
    fit_scores = support["fit_healthy_scores"]
    if not isinstance(fit_scores, Mapping) or set(fit_scores) != set(SUPPLEMENTARY_BRANCHES):
        raise ValueError("supplementary fit_healthy_scores must cover both branches")
    fit_maps = {
        name: _validate_branch_score_map(fit_scores[name], f"fit_healthy_scores.{name}")
        for name in SUPPLEMENTARY_BRANCHES
    }
    if set(fit_maps["S_pred"]) != set(fit_maps["S_pop"]):
        raise ValueError("fit_healthy score branches cover different file sets")
    if len(fit_maps["S_pred"]) != support["fit_healthy_rows"]:
        raise ValueError("fit_healthy score rows differ from the declared Fit-healthy support")
    if support["fit_healthy_rows"] != bank_rows:
        raise ValueError("Fit score count differs from the accepted bank source-row count")
    if support["fit_healthy_rows_sha256"] != reproduction["bank_source_rows_sha256"]:
        raise ValueError("Fit score support digest differs from the accepted bank source-row digest")
    if support["fit_healthy_scores_sha256"] != {
        name: score_map_sha256(fit_maps[name]) for name in SUPPLEMENTARY_BRANCHES
    }:
        raise ValueError("fit_healthy score digest does not match the declared digest")
    calibration_scores = support["calibration_scores"]
    if not isinstance(calibration_scores, Mapping) or set(calibration_scores) != set(SUPPLEMENTARY_BRANCHES):
        raise ValueError("supplementary calibration_scores must cover both branches")
    calibration_maps = {
        name: _validate_branch_score_map(calibration_scores[name], f"calibration_scores.{name}")
        for name in SUPPLEMENTARY_BRANCHES
    }
    if set(calibration_maps["S_pred"]) != set(calibration_maps["S_pop"]):
        raise ValueError("Calibration score branches cover different file sets")
    if len(calibration_maps["S_pred"]) != support["calibration_healthy_rows"]:
        raise ValueError("Calibration score rows differ from the declared Calibration support")
    if support["calibration_scores_sha256"] != {
        name: score_map_sha256(calibration_maps[name]) for name in SUPPLEMENTARY_BRANCHES
    }:
        raise ValueError("Calibration score digest does not match the declared digest")
    recomputed_thresholds = {
        name: float(np.quantile(np.asarray(sorted(calibration_maps[name].values()), dtype=np.float64), 0.95, method="linear"))
        for name in SUPPLEMENTARY_BRANCHES
    }
    if any(not math.isclose(recomputed_thresholds[name], float(thresholds[name]), rel_tol=0.0, abs_tol=1e-12) for name in SUPPLEMENTARY_BRANCHES):
        raise ValueError("declared q95 threshold does not match the frozen Calibration scores")
    development = support["development"]
    if not isinstance(development, list) or len(development) != 4:
        raise ValueError("supplementary scores must cover the four Development histories")
    expected_ids = [f"H-S18-DEV-{index:02d}" for index in range(1, 5)]
    seen: list[str] = []
    for index, item in enumerate(development):
        entry = _closed_keys(item, SCORES_DEVELOPMENT_KEYS, f"supplementary scores.support.development[{index}]")
        for name in ("manifest_sha256", "support_sha256", "timesteps_sha256"):
            _require_hash(entry[name], f"supplementary scores.support.development[{index}].{name}")
        for name in ("rows", "excluded_rows", "p", "w", "a", "controls"):
            _require_count(entry[name], f"supplementary scores.support.development[{index}].{name}")
        _finite(entry["robot_days"], f"supplementary scores.support.development[{index}].robot_days")
        branch_scores = entry["scores"]
        if not isinstance(branch_scores, Mapping) or set(branch_scores) != set(SUPPLEMENTARY_BRANCHES):
            raise ValueError(f"supplementary scores.support.development[{index}].scores must cover both branches")
        maps = {
            name: _validate_branch_score_map(branch_scores[name], f"development[{index}].scores.{name}")
            for name in SUPPLEMENTARY_BRANCHES
        }
        if set(maps["S_pred"]) != set(maps["S_pop"]):
            raise ValueError("Development score branches cover different file sets")
        if len(maps["S_pred"]) != entry["rows"] - entry["excluded_rows"]:
            raise ValueError("Development score rows differ from the declared support")
        localization = entry["localization"]
        if not isinstance(localization, Mapping) or set(localization) != set(maps["S_pred"]):
            raise ValueError("Development localization diagnostics differ from the score support")
        for file_id, record in localization.items():
            _validate_localization_record(record, f"development[{index}].localization[{file_id}]")
        actual_id = entry["history_id"]
        if not isinstance(actual_id, str) or not actual_id:
            raise ValueError("supplementary score history identity must be a non-empty string")
        if entry["schema_history_id"] != expected_ids[index]:
            raise ValueError("supplementary score schema history order is not canonical")
        if actual_id in seen:
            raise ValueError("supplementary score histories are not unique")
        seen.append(actual_id)
    if len(seen) != 4:
        raise ValueError("supplementary score histories are not the four Development histories")
    return row


def score_document_digest(document: Mapping[str, object]) -> str:
    """Canonical digest of a validated per-file score document."""
    validate_supplementary_scores(document)
    return _canonical_sha256(document)
