"""Metadata-only context/time support audit (Sprint 24 Task 7-M1).

This module audits whether the accepted Development manifests can support
temporal windows, causal references and post-hoc healthy/ordinary-transition
controls under a **frozen, bounded candidate grid** of deployment-observable
context groupings and window shapes. It is feasibility machinery, not a
detector and not a new performance gate.

Label-blind boundary (binding):

- Window and reference membership is built from source-observable metadata
  only: ``file_id``/``start_time``/``end_time``/``robot_id``/``program_id``/
  ``last_reset_time``/``is_quarantined``/``is_censored``/``n_valid_patches``
  plus the manifest ``maintenance_windows`` table and the frozen inherited
  training-eligibility predicate. :func:`project_observable` strips every
  annotation field before construction can see a row, and
  :func:`assert_construction_label_blind` fails closed if construction code
  references an annotation/failure/episode name.
- Annotations (``file_label``, ``failure_events``, degradation/maintenance
  episodes) are projected separately by :func:`project_labels` and read only
  by :func:`summarize_posthoc`, i.e. after candidate membership already
  exists. They annotate support; they never choose files, windows or
  references, and they are not detector inputs.

Cost: zero encoder forwards. No embedding, score, waveform, model or checkpoint
is opened anywhere in this module, so the three accepted encoder seeds create
no additional computation and no additional independent data. Overlapping
windows and repeated seeds are reported as dependent support, never counted as
independent samples.

The candidate grid, support-minimum policy, disposition vocabulary and output
spec are frozen in :func:`grid_spec` (and therefore in its digest) before any
support statistic is computed; see :func:`grid_freeze`.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from representation.sprint24_views import is_eligible_for_training
from synth.balanced import PROGRAM_RESERVE, ROBOT_RESERVE
from synth.events import _overlaps_maintenance

#: Audit family identifier frozen by this module.
AUDIT_ID = "S24-CTX-TIME-AUDIT-v1"

#: Seconds per day (all window/age units in this module are days).
DAY_S = 86_400.0

#: Row fields construction may read. Anything else is dropped by projection.
OBSERVABLE_ROW_FIELDS = (
    "file_id",
    "start_time",
    "end_time",
    "robot_id",
    "program_id",
    "last_reset_time",
    "is_quarantined",
    "is_censored",
    "n_valid_patches",
)

#: Annotation fields construction must never see or reference.
FORBIDDEN_ROW_FIELDS = (
    "file_label",
    "anomaly_family",
    "anomaly_severity",
    "future_targets",
    "failure_id",
    "degradation_episode_id",
    "episode_id",
    "quarantine_reason",
)

#: Candidate context groupings (all deployment-observable columns).
GROUPINGS = ("fleet", "robot", "program", "robot_x_program")

#: Candidate window widths in days (bounded grid; see :func:`grid_spec`).
WIDTH_DAYS = (3.0, 7.0, 14.0, 28.0)

#: Candidate window strides in days; only ``stride <= width`` is admitted.
STRIDE_DAYS = (1.0, 7.0, 14.0, 28.0)

#: Inherited minimum support per window (``S24-GEO-TIME-v1`` Section 4). Reused
#: unchanged: this audit classifies support, it does not tune a threshold.
MIN_WINDOW_FILES = 30

#: Inherited causal reference-pool floor (``S24-GEO-TIME-v1`` Section 5).
MIN_REFERENCE_FILES = 200

#: Descriptive sensitivity ladder for window support counts (not gates).
SUPPORT_FLOOR_LADDER = (10, 30, 100)

#: Descriptive sensitivity ladder for causal reference depth (not gates).
REFERENCE_FLOOR_LADDER = (30, 100, 200)

#: Descriptive reference value for "freshly recommissioned" context.
RESET_RECENCY_DAYS = 7.0

#: Descriptive reference value for a materially drifted robot/program mix.
MIX_DRIFT_REFERENCE = 0.25

#: Closed disposition vocabulary (support classification, never performance).
DISPOSITIONS = (
    "NO-WINDOW-SUPPORT",
    "WINDOW-SUPPORT-NO-REFERENCE-FLOOR",
    "WINDOW-SUPPORT-REFERENCE-FLOOR",
)

#: Frozen per candidate-instance output row fields (units in
#: :data:`SUPPORT_FIELD_UNITS`).
SUPPORT_ROW_FIELDS = (
    "history_id",
    "grouping",
    "context_id",
    "candidate_id",
    "width_days",
    "stride_days",
    "n_context_rows",
    "n_context_eligible",
    "n_context_eligible_eval_span",
    "n_context_eligible_prefix",
    "excluded_quarantined",
    "excluded_censored",
    "excluded_reserve_identity",
    "excluded_maintenance_overlap",
    "n_reset_edges_eval_span",
    "n_windows",
    "n_zero_member_windows",
    "n_supported_windows",
    "n_unsupported_windows",
    "max_window_files",
    "median_window_files",
    "n_windows_ge_10",
    "n_windows_ge_30",
    "n_windows_ge_100",
    "n_windows_with_pre_c1_members",
    "window_file_assignments",
    "unique_window_files",
    "reuse_factor",
    "mean_adjacent_overlap_fraction",
    "effective_independent_windows",
    "prior_depth_min",
    "prior_depth_median",
    "prior_depth_max",
    "n_windows_prior_ge_30",
    "n_windows_prior_ge_100",
    "n_windows_prior_ge_200",
    "n_windows_spanning_reset",
    "n_supported_windows_spanning_reset",
    "median_reset_age_days_p50",
    "median_reset_age_days_p95",
    "n_supported_windows_reset_recent_7d",
    "mix_drift_tv_p50",
    "mix_drift_tv_p95",
    "n_supported_windows_zero_mix_drift",
    "n_supported_windows_mix_drift_ge_0_25",
    "max_gap_hours_p50",
    "max_gap_hours_p95",
    "n_supported_windows_zero_abnormal_files",
    "n_healthy_supported_windows",
    "healthy_window_density_supported",
    "n_supported_windows_covering_failure_events",
    "n_failure_events_context_scope",
    "n_failure_events_covered_unique",
    "failure_event_cover_assignments",
    "failure_event_reuse_factor",
    "n_degradation_onsets_context_scope",
    "n_degradation_onsets_covered_unique",
    "abnormal_member_fraction_supported",
    "n_supported_windows_unknown_label_members",
    "window_support_digest",
    "reference_support_digest",
    "disposition",
    "posthoc_healthy_control_present",
)

#: Units/definitions for every frozen output field.
SUPPORT_FIELD_UNITS: dict[str, str] = {
    "history_id": "Development history identifier",
    "grouping": "context grouping name",
    "context_id": "context instance inside the grouping",
    "candidate_id": "frozen grid cell '<grouping>|<width>d|<stride>d'",
    "width_days": "days",
    "stride_days": "days",
    "n_context_rows": "manifest file rows matching the context (all custody)",
    "n_context_eligible": "rows passing the inherited eligibility predicate",
    "n_context_eligible_eval_span": "eligible rows with c1 < end_time <= tmax",
    "n_context_eligible_prefix": "eligible rows with end_time <= c1 (fixed causal reference pool)",
    "excluded_quarantined": "eval-span rows excluded as quarantined",
    "excluded_censored": "eval-span rows excluded as censored",
    "excluded_reserve_identity": "eval-span rows excluded as reserve robot/program identity",
    "excluded_maintenance_overlap": "eval-span rows excluded for manifest maintenance overlap",
    "n_reset_edges_eval_span": "maintenance-window end times (recommissioning edges) in (c1, tmax] for context robots",
    "n_windows": "trailing windows anchored every stride from c1+stride to tmax",
    "n_zero_member_windows": "windows with zero eligible members",
    "n_supported_windows": "windows with n_eligible >= 30 (inherited floor)",
    "n_unsupported_windows": "windows with 1 <= n_eligible < 30",
    "max_window_files": "largest eligible member count over all windows",
    "median_window_files": "median eligible member count over all windows",
    "n_windows_ge_10": "windows with n_eligible >= 10 (descriptive ladder)",
    "n_windows_ge_30": "windows with n_eligible >= 30 (inherited floor)",
    "n_windows_ge_100": "windows with n_eligible >= 100 (descriptive ladder)",
    "n_windows_with_pre_c1_members": "windows whose interval reaches before c1 (reference overlap)",
    "window_file_assignments": "sum of eligible member counts over windows",
    "unique_window_files": "distinct eligible file IDs appearing in any window",
    "reuse_factor": "assignments / unique files (null when unique files = 0)",
    "mean_adjacent_overlap_fraction": "mean |W_i and W_{i+1}| / |W_i| over adjacent windows",
    "effective_independent_windows": "unique files / mean supported window size (dependent-window correction)",
    "prior_depth_min": "smallest eligible prior-observation count before a window start",
    "prior_depth_median": "median eligible prior-observation count before a window start",
    "prior_depth_max": "largest eligible prior-observation count before a window start",
    "n_windows_prior_ge_30": "windows with >= 30 eligible prior observations (descriptive ladder)",
    "n_windows_prior_ge_100": "windows with >= 100 eligible prior observations (descriptive ladder)",
    "n_windows_prior_ge_200": "windows with >= 200 eligible prior observations (inherited floor)",
    "n_windows_spanning_reset": "windows whose interval contains a context reset edge",
    "n_supported_windows_spanning_reset": "supported windows containing a context reset edge",
    "median_reset_age_days_p50": "p50 over supported windows of the window median (T - last_reset_time), days",
    "median_reset_age_days_p95": "p95 over supported windows of the window median (T - last_reset_time), days",
    "n_supported_windows_reset_recent_7d": "supported windows with median reset age <= 7 days (descriptive)",
    "mix_drift_tv_p50": "p50 over supported windows of max(robot, program) TV mix drift vs causal prefix",
    "mix_drift_tv_p95": "p95 over supported windows of max(robot, program) TV mix drift vs causal prefix",
    "n_supported_windows_zero_mix_drift": "supported windows with mix drift exactly 0",
    "n_supported_windows_mix_drift_ge_0_25": "supported windows with mix drift >= 0.25 (descriptive)",
    "max_gap_hours_p50": "p50 over supported windows of the largest internal calendar gap, hours",
    "max_gap_hours_p95": "p95 over supported windows of the largest internal calendar gap, hours",
    "n_supported_windows_zero_abnormal_files": "supported windows with zero abnormal member files (post-hoc)",
    "n_healthy_supported_windows": "supported windows with zero abnormal members and zero covered failure events (post-hoc)",
    "healthy_window_density_supported": "n_healthy_supported_windows / n_supported_windows (null when none)",
    "n_supported_windows_covering_failure_events": "supported windows covering >= 1 context-scope failure event (post-hoc)",
    "n_failure_events_context_scope": "manifest failure events whose robot is in the context scope (inherited all-manifest denominator)",
    "n_failure_events_covered_unique": "distinct context-scope failure events inside any window interval (post-hoc)",
    "failure_event_cover_assignments": "sum of covered failure events over windows (dependent duplicates)",
    "failure_event_reuse_factor": "cover assignments / unique covered events (null when none)",
    "n_degradation_onsets_context_scope": "context-robot degradation episode onsets in the manifest (post-hoc denominator)",
    "n_degradation_onsets_covered_unique": "distinct context-robot degradation onsets inside any window (post-hoc)",
    "abnormal_member_fraction_supported": "abnormal member assignments / member assignments over supported windows (post-hoc)",
    "n_supported_windows_unknown_label_members": "supported windows containing a file ID with no label row",
    "window_support_digest": "sha256 over ordered (window_end, member-ID digest) pairs",
    "reference_support_digest": "sha256 over sorted fixed-reference file IDs",
    "disposition": "support classification (see DISPOSITIONS); not a performance verdict",
    "posthoc_healthy_control_present": "true when n_healthy_supported_windows > 0",
}

#: Frozen output file specification for the CLI (descriptions only).
OUTPUT_FILES: dict[str, str] = {
    "grid-freeze.json": "frozen candidate grid, support policy, output spec and grid digest",
    "support-table.json": "one row per history x candidate grid cell x context instance",
    "support-table.csv": "same rows flattened for inspection",
    "summary.json": "per-history, per-grid-cell and grouping aggregates plus feasibility headline",
    "field-audit.json": "observable-field coverage, ranges and inherited-exclusion lineage",
    "label-audit.json": "post-hoc annotation denominator/provenance counts",
    "run.json": "provenance, source commit, manifest digests, timings and output digests",
}


def canonical_json(obj: object) -> str:
    """Return the canonical JSON encoding used for every digest."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(obj: object) -> object:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_object(obj: object) -> str:
    """Return the hex SHA-256 of the canonical JSON encoding of ``obj``."""
    return sha256_text(canonical_json(obj))


def _sha256_sequence(values: Sequence[str]) -> str:
    return sha256_text("\n".join(values))


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One frozen grid cell: a context grouping and a trailing window shape."""

    grouping: str
    width_days: float
    stride_days: float

    def candidate_id(self) -> str:
        return f"{self.grouping}|{self.width_days:g}d|{self.stride_days:g}d"

    @property
    def width_s(self) -> float:
        return self.width_days * DAY_S

    @property
    def stride_s(self) -> float:
        return self.stride_days * DAY_S


@dataclass(frozen=True, slots=True)
class Context:
    """One observable context instance inside a grouping."""

    context_id: str
    grouping: str
    robot_ids: tuple[str, ...]
    program_ids: tuple[str, ...]
    filter_robot: str | None
    filter_program: str | None


def candidate_grid() -> tuple[Candidate, ...]:
    """Return the frozen candidate grid in canonical order."""
    cells = [
        Candidate(grouping, width, stride)
        for grouping in GROUPINGS
        for width in WIDTH_DAYS
        for stride in STRIDE_DAYS
        if stride <= width
    ]
    return tuple(cells)


def support_policy() -> dict[str, object]:
    """Return the frozen support-minimum policy (inherited floors, no new gates)."""
    return {
        "min_window_files": MIN_WINDOW_FILES,
        "min_window_files_basis": (
            "inherited S24-GEO-TIME-v1 Section 4 minimum support per window; "
            "reused unchanged, not tuned"
        ),
        "min_reference_files": MIN_REFERENCE_FILES,
        "min_reference_files_basis": (
            "inherited S24-GEO-TIME-v1 Section 5 reference-pool floor; reused "
            "unchanged, not tuned"
        ),
        "support_floor_ladder": list(SUPPORT_FLOOR_LADDER),
        "reference_floor_ladder": list(REFERENCE_FLOOR_LADDER),
        "reset_recency_days": RESET_RECENCY_DAYS,
        "reset_recency_basis": (
            "descriptive reference value for freshly recommissioned context; "
            "not a gate and not a detector input"
        ),
        "mix_drift_reference": MIX_DRIFT_REFERENCE,
        "mix_drift_reference_basis": (
            "descriptive reference value for a materially drifted robot/program "
            "mix; not a gate and not a detector input"
        ),
        "dispositions": list(DISPOSITIONS),
        "note": (
            "Support classification only. No numeric performance gate is defined "
            "or implied; a zero-support or zero-healthy-control outcome is a valid "
            "result. Feasibility is not calibrated specificity."
        ),
    }


def construction_rules() -> dict[str, object]:
    """Return the frozen construction rules applied to observable metadata."""
    return {
        "analysis_span": "(c1, tmax] on each history's own clock; no cross-history concatenation",
        "c1_definition": "frozen Task 2/Task 7 decile edge at 40% of the history span (external input)",
        "reference_pool": "eligible files with end_time <= c1 within the same context",
        "window_shape": "trailing (T - width, T] with T = c1 + k * stride while T <= tmax",
        "causal_membership": "only eligible files with end_time <= T enter a window",
        "causal_reference_depth": "eligible files in the context with end_time <= window_start",
        "eligibility": (
            "frozen inherited predicate is_eligible_for_training (not quarantined, "
            "not censored, not reserve robot/program identity, no manifest "
            "maintenance-window overlap)"
        ),
        "reset_edge": "maintenance-window end time for a context robot (recommissioning)",
        "reset_age": "T - median(last_reset_time of window members), days",
        "mix_drift": "max(total-variation distance of robot mix, program mix) vs causal prefix",
        "window_independence": (
            "adjacent windows overlap whenever stride < width; reuse factor and "
            "effective independent windows are reported, never assumed"
        ),
        "seed_multiplicity": (
            "metadata support is encoder-seed independent: 3 accepted B0 seeds "
            "create 0 additional forwards and 0 additional independent data"
        ),
    }


def output_spec() -> dict[str, object]:
    """Return the frozen output specification written before any analysis."""
    return {
        "row_fields": list(SUPPORT_ROW_FIELDS),
        "units": dict(SUPPORT_FIELD_UNITS),
        "files": dict(OUTPUT_FILES),
        "null_policy": "undefined ratios are JSON null, never 0",
        "optional_debug": (
            "--emit-windows writes per-window membership digests for smoke/debug "
            "verification; it is not part of the audit result"
        ),
    }


def grid_spec() -> dict[str, object]:
    """Return the complete frozen grid/policy/spec object whose digest is the freeze."""
    return {
        "audit_id": AUDIT_ID,
        "groupings": list(GROUPINGS),
        "width_days": list(WIDTH_DAYS),
        "stride_days": list(STRIDE_DAYS),
        "candidates": [
            {
                "candidate_id": c.candidate_id(),
                "grouping": c.grouping,
                "width_days": c.width_days,
                "stride_days": c.stride_days,
            }
            for c in candidate_grid()
        ],
        "support_policy": support_policy(),
        "construction_rules": construction_rules(),
        "output_spec": output_spec(),
        "observable_inputs": list(OBSERVABLE_ROW_FIELDS),
        "annotations_excluded_from_construction": list(FORBIDDEN_ROW_FIELDS),
    }


def grid_sha256() -> str:
    """Return the freeze digest of the frozen grid/policy/output spec."""
    return sha256_object(grid_spec())


def grid_freeze() -> dict[str, object]:
    """Return the grid spec together with its freeze digest."""
    spec = grid_spec()
    return {"grid_sha256": sha256_object(spec), "spec": spec}


def assert_construction_label_blind() -> None:
    """Fail closed if window/reference construction code references annotations."""
    forbidden = (
        "file_label",
        "anomaly",
        "failure_event",
        "failure_time",
        "degradation",
        "episode",
        "future_target",
        "quarantine_reason",
    )
    checked = (
        project_observable,
        build_contexts,
        prepare_context,
        build_membership,
        _anchor_ends,
    )
    for fn in checked:
        source = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    node.body = body[1:] or [ast.Pass()]
        code = ast.unparse(tree).lower()
        for token in forbidden:
            if token in code:
                raise RuntimeError(
                    f"label-blind violation: {fn.__name__} references {token!r}"
                )


def project_observable(manifest: Mapping[str, object]) -> dict[str, object]:
    """Project a manifest onto observable construction inputs only.

    The returned mapping contains no label, event, episode or annotation key, so
    construction code physically cannot read one.
    """
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("manifest is missing its file rows")
    projected: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest file row is not a mapping")
        record: dict[str, object] = {}
        for field in OBSERVABLE_ROW_FIELDS:
            if field not in row:
                raise ValueError(f"observable field missing from file row: {field}")
            record[field] = row[field]
        projected.append(record)
    if not projected:
        raise ValueError("manifest has no file rows")
    maintenance = manifest.get("maintenance_windows")
    if not isinstance(maintenance, Mapping):
        raise ValueError("manifest is missing its maintenance-window table")
    return {
        "files": projected,
        "maintenance_windows": {
            str(robot): [[float(a), float(b)] for a, b in spans]
            for robot, spans in maintenance.items()
        },
    }


def project_labels(manifest: Mapping[str, object]) -> dict[str, object]:
    """Project a manifest onto post-hoc annotation inputs only."""
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("manifest is missing its file rows")
    file_labels: dict[str, str] = {}
    for row in rows:
        file_labels[str(row["file_id"])] = str(row.get("file_label", ""))
    events = manifest.get("failure_events") or []
    episodes = manifest.get("episodes") or []
    failure_events = [
        {
            "failure_id": str(event["failure_id"]),
            "robot_id": str(event["robot_id"]),
            "failure_time": float(event["failure_time"]),
        }
        for event in events
    ]
    degradation_onsets = [
        {
            "episode_id": str(episode["episode_id"]),
            "robot_id": str(episode["robot_id"]),
            "onset": float(episode["start_time"]),
        }
        for episode in episodes
        if str(episode.get("kind")) == "degradation"
    ]
    return {
        "file_labels": file_labels,
        "failure_events": failure_events,
        "degradation_onsets": degradation_onsets,
        "n_file_labels": len(file_labels),
        "n_failure_events": len(failure_events),
        "n_degradation_onsets": len(degradation_onsets),
    }


def normalize_maintenance(raw: Mapping[str, Sequence[Sequence[float]]]) -> dict[str, list[list[float]]]:
    """Normalize a maintenance-window table to float spans per robot."""
    return {
        str(robot): [[float(start), float(end)] for start, end in spans]
        for robot, spans in raw.items()
    }


def eligibility_exclusion(
    row: Mapping[str, object], maintenance: Mapping[str, Sequence[Sequence[float]]]
) -> str | None:
    """Return the first inherited exclusion reason, or ``None`` when eligible.

    Mirrors ``representation.sprint24_views.is_eligible_for_training`` exactly and
    only classifies *why* a row is excluded (custody lineage). The truth value is
    cross-checked against the inherited predicate by the caller.
    """
    if row.get("is_quarantined"):
        return "quarantined"
    if row.get("is_censored"):
        return "censored"
    if str(row.get("robot_id")) == ROBOT_RESERVE or str(row.get("program_id")) == PROGRAM_RESERVE:
        return "reserve_identity"
    if maintenance and _overlaps_maintenance(
        {
            "robot_id": str(row["robot_id"]),
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
        },
        {robot: [list(span) for span in spans] for robot, spans in maintenance.items()},
    ):
        return "maintenance_overlap"
    return None


@dataclass(slots=True)
class HistoryMetadata:
    """Observable, label-free arrays for one Development history."""

    history_id: str
    c1: float
    tmax: float
    file_ids: tuple[str, ...]
    starts: np.ndarray
    ends: np.ndarray
    last_reset: np.ndarray
    robot_ids: tuple[str, ...]
    program_ids: tuple[str, ...]
    robot_index: np.ndarray
    program_index: np.ndarray
    eligible: np.ndarray
    exclusions: tuple[str | None, ...]
    maintenance: dict[str, list[list[float]]]
    contexts: tuple[Context, ...]


def build_contexts(
    files: Sequence[Mapping[str, object]],
) -> tuple[Context, ...]:
    """Enumerate observable context instances from the manifest's own columns."""
    robots = tuple(sorted({str(row["robot_id"]) for row in files}))
    programs = tuple(sorted({str(row["program_id"]) for row in files}))
    pairs = sorted({(str(row["robot_id"]), str(row["program_id"])) for row in files})
    contexts: list[Context] = [
        Context(
            context_id="fleet",
            grouping="fleet",
            robot_ids=robots,
            program_ids=programs,
            filter_robot=None,
            filter_program=None,
        )
    ]
    for robot in robots:
        contexts.append(
            Context(
                context_id=f"robot:{robot}",
                grouping="robot",
                robot_ids=(robot,),
                program_ids=tuple(
                    sorted({str(r["program_id"]) for r in files if str(r["robot_id"]) == robot})
                ),
                filter_robot=robot,
                filter_program=None,
            )
        )
    for program in programs:
        contexts.append(
            Context(
                context_id=f"program:{program}",
                grouping="program",
                robot_ids=tuple(
                    sorted({str(r["robot_id"]) for r in files if str(r["program_id"]) == program})
                ),
                program_ids=(program,),
                filter_robot=None,
                filter_program=program,
            )
        )
    for robot, program in pairs:
        contexts.append(
            Context(
                context_id=f"robot_x_program:{robot}|{program}",
                grouping="robot_x_program",
                robot_ids=(robot,),
                program_ids=(program,),
                filter_robot=robot,
                filter_program=program,
            )
        )
    return tuple(contexts)


def prepare_history(
    history_id: str,
    observable: Mapping[str, object],
    c1: float,
) -> HistoryMetadata:
    """Build observable arrays, eligibility lineage and contexts for one history."""
    files = observable["files"]
    maintenance = normalize_maintenance(observable["maintenance_windows"])  # type: ignore[arg-type]
    assert isinstance(files, list)
    file_ids = tuple(str(row["file_id"]) for row in files)
    if len(set(file_ids)) != len(file_ids):
        raise ValueError(f"duplicate file IDs in accepted manifest: {history_id}")
    starts = np.asarray([float(row["start_time"]) for row in files], dtype=np.float64)
    ends = np.asarray([float(row["end_time"]) for row in files], dtype=np.float64)
    last_reset = np.asarray(
        [
            float(row["last_reset_time"]) if row.get("last_reset_time") is not None else np.nan
            for row in files
        ],
        dtype=np.float64,
    )
    robots = tuple(sorted({str(row["robot_id"]) for row in files}))
    programs = tuple(sorted({str(row["program_id"]) for row in files}))
    robot_pos = {robot: index for index, robot in enumerate(robots)}
    program_pos = {program: index for index, program in enumerate(programs)}
    robot_index = np.asarray([robot_pos[str(row["robot_id"])] for row in files], dtype=np.int64)
    program_index = np.asarray([program_pos[str(row["program_id"])] for row in files], dtype=np.int64)
    exclusions = tuple(eligibility_exclusion(row, maintenance) for row in files)
    eligible = np.asarray([reason is None for reason in exclusions], dtype=bool)
    for row, reason in zip(files, exclusions):
        if (reason is None) != is_eligible_for_training(dict(row), maintenance):
            raise RuntimeError(f"eligibility lineage disagrees with inherited predicate: {history_id}")
    return HistoryMetadata(
        history_id=history_id,
        c1=float(c1),
        tmax=float(ends.max()),
        file_ids=file_ids,
        starts=starts,
        ends=ends,
        last_reset=last_reset,
        robot_ids=robots,
        program_ids=programs,
        robot_index=robot_index,
        program_index=program_index,
        eligible=eligible,
        exclusions=exclusions,
        maintenance=maintenance,
        contexts=build_contexts(files),
    )


@dataclass(slots=True)
class ContextArrays:
    """Prepared per-context arrays, reused across every candidate window shape."""

    context: Context
    context_c1: float
    context_tmax: float
    all_count: int
    exclusion_counts: dict[str, int]
    eligible_eval_span: int
    ok_index: np.ndarray
    ok_ends: np.ndarray
    ok_starts: np.ndarray
    ok_last_reset: np.ndarray
    ok_robot: np.ndarray
    ok_program: np.ndarray
    ok_ids: tuple[str, ...]
    prefix_count: int
    prefix_digest: str
    reset_edges: np.ndarray


def prepare_context(meta: HistoryMetadata, context: Context) -> ContextArrays:
    """Select one context's eligible rows and causal-reference/reset tables."""
    mask = np.ones(len(meta.file_ids), dtype=bool)
    if context.filter_robot is not None:
        target = meta.robot_ids.index(context.filter_robot)
        mask &= meta.robot_index == target
    if context.filter_program is not None:
        target = meta.program_ids.index(context.filter_program)
        mask &= meta.program_index == target
    all_index = np.flatnonzero(mask)
    eval_index = all_index[
        (meta.ends[all_index] > meta.c1) & (meta.ends[all_index] <= meta.tmax)
    ]
    exclusion_counts: dict[str, int] = Counter(
        meta.exclusions[index] for index in eval_index if meta.exclusions[index] is not None
    )
    ok_index = all_index[meta.eligible[all_index]]
    ok_index = ok_index[np.argsort(meta.ends[ok_index], kind="stable")]
    ok_ends = meta.ends[ok_index]
    prefix_mask = ok_ends <= meta.c1
    prefix_index = ok_index[prefix_mask]
    reference_ids = sorted(meta.file_ids[int(index)] for index in prefix_index)
    edges: list[float] = []
    for robot in context.robot_ids:
        for span in meta.maintenance.get(robot, []):
            edge = float(span[1])
            if meta.c1 < edge <= meta.tmax:
                edges.append(edge)
    return ContextArrays(
        context=context,
        context_c1=meta.c1,
        context_tmax=meta.tmax,
        all_count=int(all_index.size),
        exclusion_counts=dict(exclusion_counts),
        eligible_eval_span=int(ok_index.size - prefix_index.size),
        ok_index=ok_index,
        ok_ends=ok_ends,
        ok_starts=meta.starts[ok_index],
        ok_last_reset=meta.last_reset[ok_index],
        ok_robot=meta.robot_index[ok_index],
        ok_program=meta.program_index[ok_index],
        ok_ids=tuple(meta.file_ids[int(index)] for index in ok_index),
        prefix_count=int(prefix_index.size),
        prefix_digest=_sha256_sequence(reference_ids),
        reset_edges=np.asarray(sorted(edges), dtype=np.float64),
    )


def _anchor_ends(c1: float, tmax: float, stride_s: float) -> np.ndarray:
    """Return trailing-window end anchors on the history's own clock."""
    if stride_s <= 0:
        raise ValueError("stride must be positive")
    count = int(np.floor((tmax - c1) / stride_s))
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    return c1 + stride_s * np.arange(1, count + 1, dtype=np.float64)


@dataclass(slots=True)
class WindowRecord:
    """One constructed window (label-blind) and its observable covariates."""

    window_end: float
    window_start: float
    n_eligible: int
    n_pre_c1_members: int
    n_prior: int
    member_ids: tuple[str, ...]
    support_sha256: str
    median_reset_age_days: float | None
    mix_drift_tv: float | None
    max_gap_hours: float
    spans_reset: bool


@dataclass(slots=True)
class CandidateMembership:
    """All constructed windows for one history x grid cell x context instance."""

    history_id: str
    candidate: Candidate
    context_id: str
    grouping: str
    arrays: ContextArrays
    windows: list[WindowRecord]


def _mix_count(values: np.ndarray, size: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(size, dtype=np.float64)
    return np.bincount(values, minlength=size).astype(np.float64)


def _tv_distance(first: np.ndarray, second: np.ndarray) -> float | None:
    total_first = float(first.sum())
    total_second = float(second.sum())
    if total_first == 0 or total_second == 0:
        return None
    return float(0.5 * np.abs(first / total_first - second / total_second).sum())


def _max_calendar_gap_hours(starts: np.ndarray, ends: np.ndarray) -> float:
    if starts.size < 2:
        return 0.0
    order = np.argsort(starts, kind="stable")
    ordered_starts = starts[order]
    ordered_ends = ends[order]
    running_max = np.maximum.accumulate(ordered_ends)
    previous_max = np.concatenate(([ordered_ends[0]], running_max[:-1]))
    gaps = ordered_starts[1:] - previous_max[:-1]
    return float(max(0.0, gaps.max()) / 3600.0) if gaps.size else 0.0


def build_membership(
    meta: HistoryMetadata,
    arrays: ContextArrays,
    candidate: Candidate,
) -> CandidateMembership:
    """Construct trailing windows and causal references from observable data only."""
    anchors = _anchor_ends(meta.c1, meta.tmax, candidate.stride_s)
    width_s = candidate.width_s
    robot_count = len(meta.robot_ids)
    program_count = len(meta.program_ids)
    windows: list[WindowRecord] = []
    for window_end in anchors:
        window_start = float(window_end - width_s)
        left = int(np.searchsorted(arrays.ok_ends, window_start, side="right"))
        right = int(np.searchsorted(arrays.ok_ends, window_end, side="right"))
        n_eligible = right - left
        c1_cut = min(right, max(left, int(np.searchsorted(arrays.ok_ends, meta.c1, side="right"))))
        member_ids = arrays.ok_ids[left:right]
        member_starts = arrays.ok_starts[left:right]
        member_ends = arrays.ok_ends[left:right]
        member_reset = arrays.ok_last_reset[left:right]
        member_robot = arrays.ok_robot[left:right]
        member_program = arrays.ok_program[left:right]
        robot_window = _mix_count(member_robot, robot_count)
        robot_prefix = _mix_count(arrays.ok_robot[:left], robot_count)
        program_window = _mix_count(member_program, program_count)
        program_prefix = _mix_count(arrays.ok_program[:left], program_count)
        robot_tv = _tv_distance(robot_window, robot_prefix)
        program_tv = _tv_distance(program_window, program_prefix)
        drifts = [value for value in (robot_tv, program_tv) if value is not None]
        mix_drift = max(drifts) if drifts else None
        median_reset_age = None
        finite_reset = member_reset[np.isfinite(member_reset)]
        if finite_reset.size:
            median_reset_age = float(
                (window_end - float(np.median(finite_reset))) / DAY_S
            )
        reset_count = int(
            np.searchsorted(arrays.reset_edges, window_end, side="right")
            - np.searchsorted(arrays.reset_edges, window_start, side="right")
        )
        windows.append(
            WindowRecord(
                window_end=float(window_end),
                window_start=window_start,
                n_eligible=n_eligible,
                n_pre_c1_members=int(c1_cut - left),
                n_prior=left,
                member_ids=member_ids,
                support_sha256=_sha256_sequence(member_ids),
                median_reset_age_days=median_reset_age,
                mix_drift_tv=mix_drift,
                max_gap_hours=_max_calendar_gap_hours(member_starts, member_ends),
                spans_reset=reset_count > 0,
            )
        )
    return CandidateMembership(
        history_id=meta.history_id,
        candidate=candidate,
        context_id=arrays.context.context_id,
        grouping=arrays.context.grouping,
        arrays=arrays,
        windows=windows,
    )


def summarize_structure(membership: CandidateMembership) -> dict[str, object]:
    """Aggregate label-blind support, reference, overlap, reset and gap metrics."""
    windows = membership.windows
    arrays = membership.arrays
    sizes = [window.n_eligible for window in windows]
    supported = [window for window in windows if window.n_eligible >= MIN_WINDOW_FILES]
    assignments = int(sum(sizes))
    unique_files = 0
    covered_right = 0
    for window in windows:
        left = int(np.searchsorted(arrays.ok_ends, window.window_start, side="right"))
        right = left + window.n_eligible
        unique_files += max(0, right - max(left, covered_right))
        covered_right = max(covered_right, right)
    overlaps: list[float] = []
    if len(windows) > 1 and membership.candidate.stride_s < membership.candidate.width_s:
        for first, second in zip(windows, windows[1:]):
            if first.n_eligible == 0:
                continue
            overlap = int(np.searchsorted(arrays.ok_ends, first.window_end, side="right")) - int(
                np.searchsorted(arrays.ok_ends, second.window_start, side="right")
            )
            overlaps.append(max(0.0, float(overlap) / float(first.n_eligible)))
    mean_supported_size = (
        float(np.mean([window.n_eligible for window in supported])) if supported else 0.0
    )
    prior_depths = [window.n_prior for window in windows]
    reset_ages = [
        window.median_reset_age_days
        for window in supported
        if window.median_reset_age_days is not None
    ]
    drifts = [
        window.mix_drift_tv for window in supported if window.mix_drift_tv is not None
    ]
    gaps = [window.max_gap_hours for window in supported]
    window_digests = [
        [repr(window.window_end), window.support_sha256] for window in windows
    ]
    disposition = "NO-WINDOW-SUPPORT"
    if supported:
        disposition = (
            "WINDOW-SUPPORT-REFERENCE-FLOOR"
            if arrays.prefix_count >= MIN_REFERENCE_FILES
            else "WINDOW-SUPPORT-NO-REFERENCE-FLOOR"
        )
    return {
        "history_id": membership.history_id,
        "grouping": membership.grouping,
        "context_id": membership.context_id,
        "candidate_id": membership.candidate.candidate_id(),
        "width_days": membership.candidate.width_days,
        "stride_days": membership.candidate.stride_days,
        "n_context_rows": arrays.all_count,
        "n_context_eligible": int(arrays.ok_index.size),
        "n_context_eligible_eval_span": arrays.eligible_eval_span,
        "n_context_eligible_prefix": arrays.prefix_count,
        "excluded_quarantined": arrays.exclusion_counts.get("quarantined", 0),
        "excluded_censored": arrays.exclusion_counts.get("censored", 0),
        "excluded_reserve_identity": arrays.exclusion_counts.get("reserve_identity", 0),
        "excluded_maintenance_overlap": arrays.exclusion_counts.get("maintenance_overlap", 0),
        "n_reset_edges_eval_span": int(arrays.reset_edges.size),
        "n_windows": len(windows),
        "n_zero_member_windows": sum(1 for size in sizes if size == 0),
        "n_supported_windows": len(supported),
        "n_unsupported_windows": sum(1 for size in sizes if 0 < size < MIN_WINDOW_FILES),
        "max_window_files": max(sizes) if sizes else 0,
        "median_window_files": _quantile(sizes, 50),
        "n_windows_ge_10": sum(1 for size in sizes if size >= 10),
        "n_windows_ge_30": sum(1 for size in sizes if size >= 30),
        "n_windows_ge_100": sum(1 for size in sizes if size >= 100),
        "n_windows_with_pre_c1_members": sum(
            1 for window in windows if window.n_pre_c1_members > 0
        ),
        "window_file_assignments": assignments,
        "unique_window_files": unique_files,
        "reuse_factor": _ratio(assignments, unique_files),
        "mean_adjacent_overlap_fraction": (
            float(np.mean(overlaps)) if overlaps else (0.0 if windows else None)
        ),
        "effective_independent_windows": (
            float(unique_files) / mean_supported_size if mean_supported_size > 0 else None
        ),
        "prior_depth_min": min(prior_depths) if prior_depths else None,
        "prior_depth_median": _quantile(prior_depths, 50),
        "prior_depth_max": max(prior_depths) if prior_depths else None,
        "n_windows_prior_ge_30": sum(1 for depth in prior_depths if depth >= 30),
        "n_windows_prior_ge_100": sum(1 for depth in prior_depths if depth >= 100),
        "n_windows_prior_ge_200": sum(1 for depth in prior_depths if depth >= 200),
        "n_windows_spanning_reset": sum(1 for window in windows if window.spans_reset),
        "n_supported_windows_spanning_reset": sum(
            1 for window in supported if window.spans_reset
        ),
        "median_reset_age_days_p50": _quantile(reset_ages, 50),
        "median_reset_age_days_p95": _quantile(reset_ages, 95),
        "n_supported_windows_reset_recent_7d": sum(
            1 for age in reset_ages if age <= RESET_RECENCY_DAYS
        ),
        "mix_drift_tv_p50": _quantile(drifts, 50),
        "mix_drift_tv_p95": _quantile(drifts, 95),
        "n_supported_windows_zero_mix_drift": sum(1 for value in drifts if value == 0.0),
        "n_supported_windows_mix_drift_ge_0_25": sum(
            1 for value in drifts if value >= MIX_DRIFT_REFERENCE
        ),
        "max_gap_hours_p50": _quantile(gaps, 50),
        "max_gap_hours_p95": _quantile(gaps, 95),
        "window_support_digest": sha256_object(window_digests),
        "reference_support_digest": arrays.prefix_digest,
        "disposition": disposition,
    }


def summarize_posthoc(
    membership: CandidateMembership,
    labels: Mapping[str, object],
) -> dict[str, object]:
    """Annotate support with post-hoc labels; never feeds construction or selection."""
    file_labels: Mapping[str, str] = labels["file_labels"]  # type: ignore[assignment]
    failure_events: Sequence[Mapping[str, object]] = labels["failure_events"]  # type: ignore[assignment]
    degradation_onsets: Sequence[Mapping[str, object]] = labels["degradation_onsets"]  # type: ignore[assignment]
    meta = membership.arrays
    scope_robots = set(meta.context.robot_ids)
    event_rows = [
        (float(event["failure_time"]), str(event["failure_id"]))
        for event in failure_events
        if str(event["robot_id"]) in scope_robots
    ]
    onset_rows = [
        float(onset["onset"])
        for onset in degradation_onsets
        if str(onset["robot_id"]) in scope_robots
    ]
    # Protocol Section 9 denominator: every manifest failure event in the context's
    # robot scope, regardless of the evaluation-span cut (windows may reach before c1).
    event_scope = sorted(event_rows)
    onset_scope = sorted(onset_rows)
    event_times = np.asarray([entity[0] for entity in event_scope], dtype=np.float64)
    onset_times = np.asarray(onset_scope, dtype=np.float64)
    covered_events: set[str] = set()
    covered_assignments = 0
    covered_onsets: set[int] = set()
    healthy = 0
    zero_abnormal = 0
    covering_windows = 0
    unknown_windows = 0
    abnormal_members = 0
    member_assignments = 0
    for window in membership.windows:
        if window.n_eligible < MIN_WINDOW_FILES:
            continue
        abnormal = 0
        unknown = 0
        for file_id in window.member_ids:
            label = file_labels.get(file_id)
            if label is None:
                unknown += 1
            elif label != "normal":
                abnormal += 1
        member_assignments += window.n_eligible
        abnormal_members += abnormal
        if unknown:
            unknown_windows += 1
        left = int(np.searchsorted(event_times, window.window_start, side="right"))
        right = int(np.searchsorted(event_times, window.window_end, side="right"))
        if right > left:
            covering_windows += 1
            covered_assignments += right - left
            covered_events.update(event_scope[index][1] for index in range(left, right))
        onset_left = int(np.searchsorted(onset_times, window.window_start, side="right"))
        onset_right = int(np.searchsorted(onset_times, window.window_end, side="right"))
        covered_onsets.update(range(onset_left, onset_right))
        if abnormal == 0 and unknown == 0:
            zero_abnormal += 1
            if right == left:
                healthy += 1
    supported = sum(1 for window in membership.windows if window.n_eligible >= MIN_WINDOW_FILES)
    return {
        "n_supported_windows_zero_abnormal_files": zero_abnormal,
        "n_healthy_supported_windows": healthy,
        "healthy_window_density_supported": _ratio(healthy, supported),
        "n_supported_windows_covering_failure_events": covering_windows,
        "n_failure_events_context_scope": len(event_scope),
        "n_failure_events_covered_unique": len(covered_events),
        "failure_event_cover_assignments": covered_assignments,
        "failure_event_reuse_factor": _ratio(covered_assignments, len(covered_events)),
        "n_degradation_onsets_context_scope": len(onset_scope),
        "n_degradation_onsets_covered_unique": len(covered_onsets),
        "abnormal_member_fraction_supported": _ratio(abnormal_members, member_assignments),
        "n_supported_windows_unknown_label_members": unknown_windows,
        "posthoc_healthy_control_present": healthy > 0,
    }


def field_audit(meta: HistoryMetadata, observable: Mapping[str, object]) -> dict[str, object]:
    """Report observable-field coverage, ranges and inherited-exclusion lineage."""
    files = observable["files"]
    assert isinstance(files, list)
    presence = {
        field: sum(1 for row in files if row.get(field) is not None)
        for field in OBSERVABLE_ROW_FIELDS
    }
    exclusion_counts = Counter(reason for reason in meta.exclusions if reason is not None)
    return {
        "history_id": meta.history_id,
        "n_rows": len(files),
        "n_eligible": int(meta.eligible.sum()),
        "observable_field_present": presence,
        "n_rows_zero_valid_patches": sum(
            1 for row in files if int(row.get("n_valid_patches") or 0) <= 0
        ),
        "n_rows_missing_last_reset": sum(
            1 for row in files if row.get("last_reset_time") is None
        ),
        "start_time_range": [float(meta.starts.min()), float(meta.starts.max())],
        "end_time_range": [float(meta.ends.min()), float(meta.ends.max())],
        "last_reset_range": [
            float(np.nanmin(meta.last_reset)),
            float(np.nanmax(meta.last_reset)),
        ],
        "c1_s": meta.c1,
        "tmax_s": meta.tmax,
        "eval_span_days": (meta.tmax - meta.c1) / DAY_S,
        "exclusion_counts": dict(exclusion_counts),
        "robot_ids": list(meta.robot_ids),
        "program_ids": list(meta.program_ids),
        "robot_program_pairs": sorted(
            {(str(row["robot_id"]), str(row["program_id"])) for row in files}
        ),
        "n_reset_edges_eval_span_all_robots": sum(
            1
            for spans in meta.maintenance.values()
            for span in spans
            if meta.c1 < float(span[1]) <= meta.tmax
        ),
    }


def label_audit(history_id: str, labels: Mapping[str, object], meta: HistoryMetadata) -> dict[str, object]:
    """Report post-hoc annotation denominators for one history (evaluation only)."""
    labels_by_id: Mapping[str, str] = labels["file_labels"]  # type: ignore[assignment]
    counts = Counter(labels_by_id.values())
    failure_events: Sequence[Mapping[str, object]] = labels["failure_events"]  # type: ignore[assignment]
    onsets: Sequence[Mapping[str, object]] = labels["degradation_onsets"]  # type: ignore[assignment]
    return {
        "history_id": history_id,
        "n_file_labels": len(labels_by_id),
        "label_counts": dict(counts),
        "n_failure_events": len(failure_events),
        "n_failure_events_eval_span": sum(
            1 for event in failure_events if meta.c1 < float(event["failure_time"]) <= meta.tmax
        ),
        "n_degradation_onsets": len(onsets),
        "n_degradation_onsets_eval_span": sum(
            1 for onset in onsets if meta.c1 < float(onset["onset"]) <= meta.tmax
        ),
        "labels_used": "posthoc-support-annotation-only",
    }


def summarize(
    rows: Sequence[Mapping[str, object]],
    field_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate the support table without selecting a preferred candidate."""
    grid: list[dict[str, object]] = []
    for grouping in GROUPINGS:
        for width in WIDTH_DAYS:
            for stride in STRIDE_DAYS:
                if stride > width:
                    continue
                candidate_id = Candidate(grouping, width, stride).candidate_id()
                subset = [row for row in rows if row["candidate_id"] == candidate_id]
                populated = [
                    row for row in subset if int(row["n_context_eligible"]) > 0  # type: ignore[arg-type]
                ]
                supported = [
                    row for row in populated if int(row["n_supported_windows"]) > 0  # type: ignore[arg-type]
                ]
                healthy = [
                    row for row in subset if bool(row["posthoc_healthy_control_present"])
                ]
                reference = [
                    row
                    for row in subset
                    if int(row["n_context_eligible_prefix"]) >= MIN_REFERENCE_FILES  # type: ignore[arg-type]
                ]
                both = [
                    row
                    for row in healthy
                    if int(row["n_context_eligible_prefix"]) >= MIN_REFERENCE_FILES  # type: ignore[arg-type]
                ]
                grid.append(
                    {
                        "candidate_id": candidate_id,
                        "grouping": grouping,
                        "width_days": width,
                        "stride_days": stride,
                        "n_candidate_instances": len(subset),
                        "n_instances_with_eligible_rows": len(populated),
                        "n_instances_zero_eligible_rows": len(subset) - len(populated),
                        "n_instances_with_window_support": len(supported),
                        "n_instances_zero_window_support": len(populated) - len(supported),
                        "n_instances_with_reference_floor": len(reference),
                        "n_instances_with_healthy_control": len(healthy),
                        "n_instances_with_reference_floor_and_healthy_control": len(both),
                        "total_supported_windows": int(
                            sum(int(row["n_supported_windows"]) for row in subset)  # type: ignore[arg-type]
                        ),
                        "total_healthy_supported_windows": int(
                            sum(int(row["n_healthy_supported_windows"]) for row in subset)  # type: ignore[arg-type]
                        ),
                        "max_supported_windows_per_instance": max(
                            (int(row["n_supported_windows"]) for row in subset), default=0  # type: ignore[arg-type]
                        ),
                        "max_window_files": max(
                            (int(row["max_window_files"]) for row in subset), default=0  # type: ignore[arg-type]
                        ),
                    }
                )
    by_grouping: list[dict[str, object]] = []
    for grouping in GROUPINGS:
        subset = [row for row in rows if row["grouping"] == grouping]
        by_grouping.append(
            {
                "grouping": grouping,
                "n_candidate_instances": len(subset),
                "n_contexts": len({(row["history_id"], row["context_id"]) for row in subset}),
                "n_instances_with_window_support": sum(
                    1 for row in subset if int(row["n_supported_windows"]) > 0  # type: ignore[arg-type]
                ),
                "n_instances_with_healthy_control": sum(
                    1 for row in subset if bool(row["posthoc_healthy_control_present"])
                ),
                "total_supported_windows": sum(
                    int(row["n_supported_windows"]) for row in subset  # type: ignore[arg-type]
                ),
                "total_healthy_supported_windows": sum(
                    int(row["n_healthy_supported_windows"]) for row in subset  # type: ignore[arg-type]
                ),
            }
        )
    disposition_counts = Counter(str(row["disposition"]) for row in rows)
    healthy_rows = [
        {
            "history_id": row["history_id"],
            "grouping": row["grouping"],
            "context_id": row["context_id"],
            "candidate_id": row["candidate_id"],
            "n_supported_windows": row["n_supported_windows"],
            "n_healthy_supported_windows": row["n_healthy_supported_windows"],
            "healthy_window_density_supported": row["healthy_window_density_supported"],
            "n_context_eligible_prefix": row["n_context_eligible_prefix"],
        }
        for row in rows
        if bool(row["posthoc_healthy_control_present"])
    ]
    histories: list[dict[str, object]] = []
    for history in sorted({str(row["history_id"]) for row in rows}):
        subset = [row for row in rows if row["history_id"] == history]
        audit_row = next(row for row in field_rows if row["history_id"] == history)
        histories.append(
            {
                "history_id": history,
                "n_rows": audit_row["n_rows"],
                "n_eligible": audit_row["n_eligible"],
                "n_contexts": len({row["context_id"] for row in subset}),
                "n_candidate_instances": len(subset),
                "n_instances_with_window_support": sum(
                    1 for row in subset if int(row["n_supported_windows"]) > 0  # type: ignore[arg-type]
                ),
                "n_instances_with_healthy_control": sum(
                    1 for row in subset if bool(row["posthoc_healthy_control_present"])
                ),
                "total_supported_windows": sum(
                    int(row["n_supported_windows"]) for row in subset  # type: ignore[arg-type]
                ),
                "total_healthy_supported_windows": sum(
                    int(row["n_healthy_supported_windows"]) for row in subset  # type: ignore[arg-type]
                ),
            }
        )
    with_support = [row for row in rows if int(row["n_supported_windows"]) > 0]  # type: ignore[arg-type]
    return {
        "n_support_rows": len(rows),
        "n_grid_cells": len({row["candidate_id"] for row in rows}),
        "disposition_counts": dict(disposition_counts),
        "candidate_instances_with_window_support": len(with_support),
        "candidate_instances_with_healthy_control": len(healthy_rows),
        "candidate_instances_with_reference_floor": sum(
            1
            for row in rows
            if int(row["n_context_eligible_prefix"]) >= MIN_REFERENCE_FILES  # type: ignore[arg-type]
        ),
        "candidate_instances_with_reference_floor_and_window_support": sum(
            1
            for row in with_support
            if int(row["n_context_eligible_prefix"]) >= MIN_REFERENCE_FILES  # type: ignore[arg-type]
        ),
        "candidate_instances_with_reference_floor_and_healthy_control": sum(
            1
            for row in healthy_rows
            if int(row["n_context_eligible_prefix"]) >= MIN_REFERENCE_FILES  # type: ignore[arg-type]
        ),
        "any_posthoc_healthy_control": bool(healthy_rows),
        "any_healthy_control_with_inherited_reference_floor": any(
            int(row["n_context_eligible_prefix"]) >= MIN_REFERENCE_FILES  # type: ignore[arg-type]
            for row in healthy_rows
        ),
        "healthy_control_instances": healthy_rows,
        "by_history": histories,
        "by_grouping": by_grouping,
        "by_grid_cell": grid,
        "feasibility_note": (
            "Counts and quantiles only. No candidate is preferred, ranked or "
            "selected, no performance gate is defined, and overlap/seed multiplicity "
            "are not counted as independent support."
        ),
    }


def audit_histories(
    manifests: Mapping[str, Mapping[str, object]],
    edges: Mapping[str, Sequence[float]],
    include_windows: bool = False,
) -> dict[str, object]:
    """Run the frozen audit over the supplied manifests (metadata-only).

    Per history the whole candidate grid is constructed from projected
    observable rows first; annotations are projected and joined only afterwards,
    so no label can influence membership, support counts or dispositions.
    """
    assert_construction_label_blind()
    rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    grid = candidate_grid()
    for history_id in sorted(manifests):
        manifest = manifests[history_id]
        if history_id not in edges:
            raise KeyError(f"missing frozen decile edges for {history_id}")
        edge_values = [float(value) for value in edges[history_id]]
        if len(edge_values) < 5:
            raise ValueError(f"decile edge table too short for {history_id}")
        observable = project_observable(manifest)
        meta = prepare_history(history_id, observable, edge_values[4])
        field_rows.append(field_audit(meta, observable))
        prepared = {context.context_id: prepare_context(meta, context) for context in meta.contexts}
        memberships: list[CandidateMembership] = []
        for candidate in grid:
            for context in meta.contexts:
                if context.grouping != candidate.grouping:
                    continue
                memberships.append(
                    build_membership(meta, prepared[context.context_id], candidate)
                )
        labels = project_labels(manifest)
        label_rows.append(label_audit(history_id, labels, meta))
        for membership in memberships:
            row = summarize_structure(membership)
            row.update(summarize_posthoc(membership, labels))
            rows.append(row)
            if include_windows:
                for window in membership.windows:
                    window_rows.append(
                        {
                            "history_id": history_id,
                            "grouping": membership.grouping,
                            "context_id": membership.context_id,
                            "candidate_id": membership.candidate.candidate_id(),
                            "window_end_T": window.window_end,
                            "window_start": window.window_start,
                            "n_eligible": window.n_eligible,
                            "n_prior": window.n_prior,
                            "support_sha256": window.support_sha256,
                        }
                    )
    result: dict[str, object] = {
        "audit_id": AUDIT_ID,
        "grid_sha256": grid_sha256(),
        "support_rows": rows,
        "field_audit": field_rows,
        "label_audit": label_rows,
        "summary": summarize(rows, field_rows),
    }
    if include_windows:
        result["window_rows"] = window_rows
    return result
