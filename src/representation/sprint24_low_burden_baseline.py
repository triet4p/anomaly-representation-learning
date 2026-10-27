"""Low-burden, stable baseline support audit (Sprint 24 Task 7-M5, metadata only).

This module audits whether the accepted Development metadata can support
low-burden, temporally stable baseline regions -- and comparable
elevated-burden contrast regions -- at weekly/monthly granularity for a future
collective-movement comparison. It is feasibility machinery, not a detector,
not a movement measurement, and not a new performance gate.

Owner clarification carried in (not re-derived): a normal-trending comparison
region may contain a small amount of faults; zero faults are not required.
Accordingly a period with a small *nonzero* abnormal file fraction is eligible
for baseline candidacy when every other rule holds. Labels are used here only
as post-hoc diagnostic annotations that select comparison regions; they are not
model inputs and this audit makes no label-blind detection claim.

Binding boundaries (inherited, unchanged):

- Period/pair membership, exposure, reference custody and comparability come
  exclusively from the frozen weekly/monthly study machinery
  (:mod:`representation.sprint24_weekly_monthly_study` built on
  :mod:`representation.sprint24_business_horizon`). Candidate *sets* are the
  label-blind manifest rows; annotations only *classify* them afterwards.
  :func:`assert_candidate_construction_label_blind` fails closed if candidate
  construction references an annotation name.
- Inherited floors are reused unchanged: 30 eligible files per period, 200
  reference files per history, 0.8 exposure-ratio floor, 0.25 mix-drift
  reference. The audit classifies support; it never tunes a floor.
- Research assumptions (frozen here, never owner-approved thresholds): the low
  burden ceiling, the elevated contrast floor, and the stability range below.
  They are declared before any candidate is counted and are never adjusted to
  obtain support. A descriptive ladder is published beside each primary value
  so a zero at the primary cap cannot be confused with a zero at every cap.
- Cost: zero encoder forwards. No model, checkpoint, waveform, embedding,
  score or cache is opened anywhere in this module. The three accepted encoder
  seeds are metadata-identical and add no independent data.
- Within-file burden (per-file severity/area) has no accepted annotation
  field: only the file-level ``file_label`` exists. File prevalence is
  classified; within-file burden is reported as an unavailable dimension.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Mapping, Sequence

from representation import sprint24_business_horizon as horizon
from representation import sprint24_weekly_monthly_study as study
from representation.sprint24_context_audit import sha256_object

#: Audit family identifier frozen by this module.
AUDIT_ID = "S24-LOW-BURDEN-BASELINE-v1"

#: Audited horizons. ``day`` is deferred by owner prioritization and is
#: deliberately absent: no daily candidate, count or claim exists here.
HORIZONS = ("week", "month")

#: Boundary conventions. ``calendar`` is primary; ``rolling`` is sensitivity.
PRIMARY_MODE = study.PRIMARY_MODE
MODES = study.MODES

#: Primary low-burden ceiling on the abnormal member fraction. Justification
#: (research assumption, not an owner-approved threshold): at most one file in
#: ten is abnormal, i.e. faults are a small minority of a normal-trending
#: region, which is the plain reading of the owner's "small amount of faults".
LOW_BURDEN_FRACTION_PRIMARY = 0.10

#: Descriptive burden ladder published beside the primary ceiling (not gates).
LOW_BURDEN_FRACTION_LADDER = (0.05, 0.10, 0.15)

#: Primary elevated-burden floor on the abnormal member fraction.
#: Justification: twice the low ceiling, so the baseline and elevated regions
#: are separated by a contrast gap rather than sharing one boundary value.
ELEVATED_FRACTION_PRIMARY = 0.20

#: Descriptive elevated ladder (not gates).
ELEVATED_FRACTION_LADDER = (0.15, 0.20, 0.25)

#: Primary stability bound: the maximum abnormal-fraction range inside a
#: stable baseline run. Justification: half the low ceiling; a baseline whose
#: burden swings by more than half its ceiling across consecutive periods is
#: not demonstrably stable at these small counts.
STABILITY_RANGE_PRIMARY = 0.05

#: Minimum consecutive low-burden full supported periods forming a segment.
#: Two is the smallest run in which stability (a trend, not a point) is even
#: observable; a singleton low period is reported, never a segment.
MIN_STABLE_RUN_PERIODS = 2

#: Inherited floors, re-declared so the study cannot silently change them.
MIN_PERIOD_FILES = horizon.MIN_PERIOD_FILES
MIN_REFERENCE_FILES = horizon.MIN_REFERENCE_FILES
EXPOSURE_RATIO_FLOOR = horizon.PROPOSED_EXPOSURE_RATIO_FLOOR
MIX_DRIFT_REFERENCE = 0.25

#: Minimum FEASIBLE histories (of four) for a horizon-level FEASIBLE verdict.
MIN_FEASIBLE_HISTORIES = 3

#: Closed period-candidacy vocabulary.
PERIOD_STATUSES = (
    "LOW-BASELINE-CANDIDATE",
    "BORDERLINE-BURDEN",
    "ELEVATED-CANDIDATE",
    "REJECTED-INSUFFICIENT-SUPPORT",
    "REJECTED-PARTIAL-PERIOD",
    "REJECTED-UNANNOTATED",
)

#: Closed run vocabulary.
RUN_STATUSES = (
    "STABLE-BASELINE-SEGMENT",
    "LOW-RUN-UNSTABLE",
    "LOW-RUN-SINGLETON",
)

#: Closed pair-region vocabulary.
PAIR_REGIONS = (
    "BASELINE-STABLE",
    "ELEVATED-CONSISTENT",
    "TRANSITION",
    "MIXED-BORDERLINE",
    "BASELINE-EXPOSURE-LIMITED",
    "ELEVATED-EXPOSURE-LIMITED",
    "TRANSITION-EXPOSURE-LIMITED",
    "REJECTED-NONCOMPARABLE",
    "REJECTED-UNANNOTATED",
)

#: Closed pair mix-comparability flags.
MIX_FLAGS = ("MIX-OK", "MIX-CONFOUNDED", "MIX-UNKNOWN")

#: Closed disposition vocabulary (support classification, never performance).
DISPOSITIONS = ("FEASIBLE", "LIMITED", "UNSUPPORTED")

#: Frozen per-period candidate output fields.
PERIOD_CANDIDATE_FIELDS: tuple[str, ...] = (
    "history_id",
    "horizon",
    "mode",
    "period_identity",
    "period_key",
    "period_start_rel_s",
    "period_end_rel_s",
    "n_members",
    "exposure_days",
    "annotation_state",
    "n_abnormal_members",
    "abnormal_member_fraction",
    "n_failure_events_covered",
    "n_degradation_onsets_covered",
    "ladder_low_0_05",
    "ladder_low_0_10",
    "ladder_low_0_15",
    "ladder_elevated_0_15",
    "ladder_elevated_0_20",
    "ladder_elevated_0_25",
    "status",
    "status_reason",
)

#: Frozen per-pair region output fields.
PAIR_REGION_FIELDS: tuple[str, ...] = (
    "history_id",
    "horizon",
    "mode",
    "pair_identity",
    "previous_period_identity",
    "current_period_identity",
    "comparison_state",
    "previous_status",
    "current_status",
    "region",
    "region_reason",
    "mix_drift_tv",
    "mix_flag",
    "exposure_ratio",
    "n_reset_edges_crossed",
    "pair_abnormal_member_fraction",
    "pair_n_failure_events_covered",
    "in_stable_segment",
)


def audit_spec() -> dict[str, object]:
    """Return the frozen audit identity, grid, rules and vocabularies."""
    return {
        "audit_id": AUDIT_ID,
        "upstream_study": {
            "study_id": study.STUDY_ID,
            "study_sha256": study.study_sha256(),
            "proposal_id": horizon.PROPOSAL_ID,
            "proposal_sha256": horizon.proposal_sha256(),
            "membership": "label-blind study manifest rows; annotations classify afterwards",
        },
        "grid": {
            "horizons": list(HORIZONS),
            "modes": list(MODES),
            "primary_mode": PRIMARY_MODE,
            "deferred_horizons": ["day"],
        },
        "research_assumptions": {
            "low_burden_fraction_primary": LOW_BURDEN_FRACTION_PRIMARY,
            "low_burden_fraction_ladder": list(LOW_BURDEN_FRACTION_LADDER),
            "elevated_fraction_primary": ELEVATED_FRACTION_PRIMARY,
            "elevated_fraction_ladder": list(ELEVATED_FRACTION_LADDER),
            "stability_range_primary": STABILITY_RANGE_PRIMARY,
            "min_stable_run_periods": MIN_STABLE_RUN_PERIODS,
            "assumption_status": (
                "research audit assumptions frozen before counting; "
                "not owner-approved operational thresholds"
            ),
        },
        "inherited_unchanged": {
            "min_period_files": MIN_PERIOD_FILES,
            "min_reference_files": MIN_REFERENCE_FILES,
            "exposure_ratio_floor": EXPOSURE_RATIO_FLOOR,
            "mix_drift_reference": MIX_DRIFT_REFERENCE,
            "eligibility_predicate": "representation.sprint24_views.is_eligible_for_training",
            "analysis_span": "(c1, tmax] per history, never concatenated",
        },
        "vocabularies": {
            "period_status": list(PERIOD_STATUSES),
            "run_status": list(RUN_STATUSES),
            "pair_region": list(PAIR_REGIONS),
            "mix_flag": list(MIX_FLAGS),
            "disposition": list(DISPOSITIONS),
        },
    }


def audit_sha256() -> str:
    """Return the freeze digest of the audit identity/grid/rules."""
    return sha256_object(audit_spec())


def audit_freeze() -> dict[str, object]:
    """Return the audit spec together with its freeze digest."""
    spec = audit_spec()
    return {"audit_sha256": sha256_object(spec), "spec": spec}


#: Names that must never appear in candidate-set construction code.
_FORBIDDEN_CANDIDATE_TOKENS = (
    "file_label",
    "anomaly",
    "failure_event",
    "failure_time",
    "degradation",
    "episode",
    "future_target",
    "quarantine_reason",
    "attribution",
    "annotation",
    "abnormal",
    "burden",
)


def _candidate_construction_functions() -> tuple[object, ...]:
    return (
        candidate_period_keys,
        candidate_pair_keys,
        order_period_identities,
    )


def assert_candidate_construction_label_blind() -> None:
    """Fail closed if candidate-set construction references annotations."""
    for fn in _candidate_construction_functions():
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
        for token in _FORBIDDEN_CANDIDATE_TOKENS:
            if token in code:
                raise RuntimeError(f"label-blind violation: {fn.__name__} references {token!r}")


def candidate_period_keys(
    manifest_periods: Sequence[Mapping[str, object]],
    history_id: str,
    horizon_name: str,
    mode: str,
) -> list[str]:
    """Return the label-blind candidate period identities in time order."""
    selected = [
        row
        for row in manifest_periods
        if str(row["history_id"]) == history_id
        and str(row["horizon"]) == horizon_name
        and str(row["mode"]) == mode
    ]
    selected.sort(key=lambda row: float(row["period_start_rel_s"]))
    return [str(row["period_identity"]) for row in selected]


def candidate_pair_keys(
    manifest_pairs: Sequence[Mapping[str, object]],
    history_id: str,
    horizon_name: str,
    mode: str,
) -> list[str]:
    """Return the label-blind candidate pair identities in time order."""
    selected = [
        row
        for row in manifest_pairs
        if str(row["history_id"]) == history_id
        and str(row["horizon"]) == horizon_name
        and str(row["mode"]) == mode
    ]
    selected.sort(key=lambda row: float(row["previous_start_rel_s"]))
    return [str(row["pair_identity"]) for row in selected]


def order_period_identities(
    manifest_periods: Sequence[Mapping[str, object]],
    identities: Sequence[str],
) -> list[str]:
    """Order period identities by period start (label-blind)."""
    start = {str(row["period_identity"]): float(row["period_start_rel_s"]) for row in manifest_periods}
    return sorted((str(identity) for identity in identities), key=lambda key: start[key])


def classify_period(
    structural: Mapping[str, object],
    posthoc: Mapping[str, object] | None,
) -> dict[str, object]:
    """Classify one label-blind period row with its post-hoc burden entry."""
    identity = str(structural["period_identity"])
    if str(structural["coverage_state"]) != "FULL":
        return _period_result(structural, posthoc, "REJECTED-PARTIAL-PERIOD", "period is not FULL coverage")
    if not bool(structural["meets_support_floor"]):
        return _period_result(
            structural, posthoc, "REJECTED-INSUFFICIENT-SUPPORT", "period below the inherited 30-file floor"
        )
    if posthoc is None or str(posthoc.get("annotation_state", "")).startswith("UNANNOTATED"):
        return _period_result(
            structural, posthoc, "REJECTED-UNANNOTATED", "annotation coverage is absent; absence is never health"
        )
    fraction = posthoc.get("abnormal_member_fraction")
    if fraction is None:
        return _period_result(
            structural, posthoc, "REJECTED-UNANNOTATED", "burden fraction unavailable on a supported period"
        )
    value = float(fraction)
    if value <= LOW_BURDEN_FRACTION_PRIMARY:
        return _period_result(
            structural,
            posthoc,
            "LOW-BASELINE-CANDIDATE",
            f"abnormal fraction {value:.4f} at or below the primary low ceiling "
            f"{LOW_BURDEN_FRACTION_PRIMARY}; small nonzero burden is eligible",
        )
    if value >= ELEVATED_FRACTION_PRIMARY:
        return _period_result(
            structural,
            posthoc,
            "ELEVATED-CANDIDATE",
            f"abnormal fraction {value:.4f} at or above the elevated floor "
            f"{ELEVATED_FRACTION_PRIMARY} (twice the low ceiling)",
        )
    return _period_result(
        structural,
        posthoc,
        "BORDERLINE-BURDEN",
        f"abnormal fraction {value:.4f} inside the contrast gap "
        f"({LOW_BURDEN_FRACTION_PRIMARY}, {ELEVATED_FRACTION_PRIMARY}); neither region",
    )


def _period_result(
    structural: Mapping[str, object],
    posthoc: Mapping[str, object] | None,
    status: str,
    reason: str,
) -> dict[str, object]:
    fraction = None if posthoc is None else posthoc.get("abnormal_member_fraction")
    value = None if fraction is None else float(fraction)
    return {
        "history_id": str(structural["history_id"]),
        "horizon": str(structural["horizon"]),
        "mode": str(structural["mode"]),
        "period_identity": str(structural["period_identity"]),
        "period_key": str(structural["period_key"]),
        "period_start_rel_s": float(structural["period_start_rel_s"]),
        "period_end_rel_s": float(structural["period_end_rel_s"]),
        "n_members": int(structural["n_members"]),
        "exposure_days": float(structural["exposure_days"]),
        "annotation_state": None if posthoc is None else posthoc.get("annotation_state"),
        "n_abnormal_members": None if posthoc is None else posthoc.get("n_abnormal_members"),
        "abnormal_member_fraction": value,
        "n_failure_events_covered": None if posthoc is None else posthoc.get("n_failure_events_covered"),
        "n_degradation_onsets_covered": None if posthoc is None else posthoc.get("n_degradation_onsets_covered"),
        "ladder_low_0_05": None if value is None else bool(value <= 0.05),
        "ladder_low_0_10": None if value is None else bool(value <= 0.10),
        "ladder_low_0_15": None if value is None else bool(value <= 0.15),
        "ladder_elevated_0_15": None if value is None else bool(value >= 0.15),
        "ladder_elevated_0_20": None if value is None else bool(value >= 0.20),
        "ladder_elevated_0_25": None if value is None else bool(value >= 0.25),
        "status": status,
        "status_reason": reason,
    }


def stable_runs(classified_periods: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Assemble maximal consecutive LOW runs and assess their stability."""
    ordered = sorted(classified_periods, key=lambda row: float(row["period_start_rel_s"]))
    runs: list[dict[str, object]] = []
    current: list[Mapping[str, object]] = []
    for row in ordered:
        if str(row["status"]) == "LOW-BASELINE-CANDIDATE":
            current.append(row)
        else:
            if current:
                runs.append(_assess_run(current))
                current = []
    if current:
        runs.append(_assess_run(current))
    return runs


def _assess_run(members: Sequence[Mapping[str, object]]) -> dict[str, object]:
    fractions = [float(row["abnormal_member_fraction"]) for row in members]  # type: ignore[arg-type]
    burden_range = max(fractions) - min(fractions)
    identities = [str(row["period_identity"]) for row in members]
    if len(members) < MIN_STABLE_RUN_PERIODS:
        status = "LOW-RUN-SINGLETON"
        reason = "a single low period shows a point, not a stable trend"
    elif burden_range <= STABILITY_RANGE_PRIMARY:
        status = "STABLE-BASELINE-SEGMENT"
        reason = (
            f"run of {len(members)} consecutive low periods with burden range "
            f"{burden_range:.4f} within {STABILITY_RANGE_PRIMARY}"
        )
    else:
        status = "LOW-RUN-UNSTABLE"
        reason = (
            f"run of {len(members)} consecutive low periods with burden range "
            f"{burden_range:.4f} above {STABILITY_RANGE_PRIMARY}; not demonstrably stable"
        )
    return {
        "status": status,
        "status_reason": reason,
        "n_periods": len(members),
        "period_identities": identities,
        "burden_fraction_min": min(fractions),
        "burden_fraction_max": max(fractions),
        "burden_fraction_range": burden_range,
    }


def mix_flag(drift: object) -> str:
    """Return the mix-comparability flag for one pair drift value."""
    if drift is None:
        return "MIX-UNKNOWN"
    return "MIX-OK" if float(drift) <= MIX_DRIFT_REFERENCE else "MIX-CONFOUNDED"


def classify_pair(
    structural: Mapping[str, object],
    posthoc: Mapping[str, object] | None,
    period_status: Mapping[str, str],
) -> dict[str, object]:
    """Classify one label-blind pair row with its post-hoc region entry."""
    comparison = str(structural["comparison_state"])
    previous_identity = str(structural["previous_period_identity"])
    current_identity = str(structural["current_period_identity"])
    previous_status = period_status.get(previous_identity, "REJECTED-UNANNOTATED")
    current_status = period_status.get(current_identity, "REJECTED-UNANNOTATED")
    drift = structural.get("mix_drift_tv")
    flag = mix_flag(drift)
    exposure = structural.get("exposure_ratio")
    pair_fraction = None if posthoc is None else posthoc.get("abnormal_member_fraction")
    pair_events = None if posthoc is None else posthoc.get("n_failure_events_covered")
    unannotated = (
        posthoc is None
        or "UNANNOTATED" in str(posthoc.get("previous_annotation_state", ""))
        or "UNANNOTATED" in str(posthoc.get("current_annotation_state", ""))
        or previous_status == "REJECTED-UNANNOTATED"
        or current_status == "REJECTED-UNANNOTATED"
    )
    if unannotated:
        region = "REJECTED-UNANNOTATED"
        reason = "annotation coverage is absent on at least one side; absence is never a control"
    elif comparison == "COMPARABLE-MATCHED":
        region, reason = _matched_region(previous_status, current_status)
    elif comparison == "COMPARABLE-UNMATCHED-EXPOSURE":
        region, reason = _unmatched_region(previous_status, current_status)
    else:
        region = "REJECTED-NONCOMPARABLE"
        reason = f"pair comparability is {comparison}; support or coverage gate first"
    return {
        "history_id": str(structural["history_id"]),
        "horizon": str(structural["horizon"]),
        "mode": str(structural["mode"]),
        "pair_identity": str(structural["pair_identity"]),
        "previous_period_identity": previous_identity,
        "current_period_identity": current_identity,
        "comparison_state": comparison,
        "previous_status": previous_status,
        "current_status": current_status,
        "region": region,
        "region_reason": reason,
        "mix_drift_tv": None if drift is None else float(drift),
        "mix_flag": flag,
        "exposure_ratio": None if exposure is None else float(exposure),
        "n_reset_edges_crossed": int(structural["n_reset_edges_crossed"]),
        "pair_abnormal_member_fraction": None if pair_fraction is None else float(pair_fraction),
        "pair_n_failure_events_covered": pair_events,
        "in_stable_segment": False,
    }


def _matched_region(previous_status: str, current_status: str) -> tuple[str, str]:
    sides = (previous_status, current_status)
    if all(side == "LOW-BASELINE-CANDIDATE" for side in sides):
        return "BASELINE-STABLE", "comparable matched pair with both periods low-burden"
    if all(side == "ELEVATED-CANDIDATE" for side in sides):
        return "ELEVATED-CONSISTENT", "comparable matched pair with both periods elevated"
    if set(sides) == {"LOW-BASELINE-CANDIDATE", "ELEVATED-CANDIDATE"}:
        return "TRANSITION", "comparable matched pair joining a low period to an elevated period"
    return "MIXED-BORDERLINE", "comparable matched pair involving a borderline or rejected period"


def _unmatched_region(previous_status: str, current_status: str) -> tuple[str, str]:
    sides = (previous_status, current_status)
    if all(side == "LOW-BASELINE-CANDIDATE" for side in sides):
        return (
            "BASELINE-EXPOSURE-LIMITED",
            "both periods low-burden but exposure-mismatched; capped-n sensitivity only",
        )
    if all(side == "ELEVATED-CANDIDATE" for side in sides):
        return (
            "ELEVATED-EXPOSURE-LIMITED",
            "both periods elevated but exposure-mismatched; capped-n sensitivity only",
        )
    if set(sides) == {"LOW-BASELINE-CANDIDATE", "ELEVATED-CANDIDATE"}:
        return (
            "TRANSITION-EXPOSURE-LIMITED",
            "low-to-elevated pair but exposure-mismatched; capped-n sensitivity only",
        )
    return "MIXED-BORDERLINE", "exposure-mismatched pair involving a borderline or rejected period"


def mark_segment_members(
    pair_regions: Sequence[dict[str, object]],
    runs: Sequence[Mapping[str, object]],
) -> None:
    """Flag pairs fully inside a stable segment (in place)."""
    inside: set[str] = set()
    for run in runs:
        if str(run["status"]) != "STABLE-BASELINE-SEGMENT":
            continue
        members = [str(identity) for identity in run["period_identities"]]  # type: ignore[union-attr]
        for first, second in zip(members, members[1:]):
            inside.add(f"{first}\n{second}")
    for row in pair_regions:
        key = f"{row['previous_period_identity']}\n{row['current_period_identity']}"
        row["in_stable_segment"] = key in inside




def select_nonoverlapping_pairs(
    pair_regions_in_order: Sequence[Mapping[str, object]],
) -> list[str]:
    """Greedily select pairs whose period sets are pairwise disjoint."""
    chosen: list[str] = []
    used: set[str] = set()
    for row in pair_regions_in_order:
        previous = str(row["previous_period_identity"])
        current = str(row["current_period_identity"])
        if previous in used or current in used:
            continue
        used.add(previous)
        used.add(current)
        chosen.append(str(row["pair_identity"]))
    return chosen


def summarize_history(
    history_id: str,
    horizon_name: str,
    mode: str,
    period_rows: Sequence[dict[str, object]],
    pair_rows: Sequence[dict[str, object]],
    reference_state: str,
) -> dict[str, object]:
    """Summarize one history x horizon x mode audit cell."""
    segment_periods = [row for row in period_rows if str(row["period_identity"]).startswith(f"{history_id}|{horizon_name}|{mode}|")]
    ordered_periods = sorted(segment_periods, key=lambda row: float(row["period_start_rel_s"]))
    runs = stable_runs(ordered_periods)
    start_of = {str(row["period_identity"]): float(row["period_start_rel_s"]) for row in ordered_periods}
    segment_pairs = sorted(
        (
            row for row in pair_rows
            if str(row["pair_identity"]).startswith(f"{history_id}|{horizon_name}|{mode}|")
        ),
        key=lambda row: start_of.get(str(row["previous_period_identity"]), float("inf")),
    )
    mark_segment_members(segment_pairs, runs)
    stable_segments = [run for run in runs if str(run["status"]) == "STABLE-BASELINE-SEGMENT"]
    baseline_pairs = [
        row for row in segment_pairs
        if str(row["region"]) == "BASELINE-STABLE" and bool(row["in_stable_segment"])
    ]
    baseline_mix_ok = [row for row in baseline_pairs if str(row["mix_flag"]) == "MIX-OK"]
    contrast_pairs = [
        row for row in segment_pairs
        if str(row["region"]) in ("ELEVATED-CONSISTENT", "TRANSITION")
    ]
    nonoverlap_baseline = select_nonoverlapping_pairs(baseline_pairs)
    nonoverlap_contrast = select_nonoverlapping_pairs(contrast_pairs)
    distinct_baseline_periods = sorted(
        {str(row["previous_period_identity"]) for row in baseline_pairs}
        | {str(row["current_period_identity"]) for row in baseline_pairs}
    )
    elevated_periods = sorted(
        str(row["period_identity"]) for row in ordered_periods
        if str(row["status"]) == "ELEVATED-CANDIDATE"
    )
    exposure_compatible = _exposure_compatible_elevated(ordered_periods, stable_segments, elevated_periods)
    if not stable_segments:
        disposition = "UNSUPPORTED"
        reason = "no stable baseline segment exists in this cell"
    elif baseline_mix_ok and contrast_pairs:
        disposition = "FEASIBLE"
        reason = (
            "at least one stable segment, one mix-ok stable baseline pair, "
            "and one comparable elevated/transition pair coexist"
        )
    else:
        disposition = "LIMITED"
        if not baseline_pairs:
            reason = "a stable segment exists but no comparable baseline-stable pair survives exposure matching"
        elif not baseline_mix_ok:
            reason = "stable baseline pairs exist but every one is mix-confounded or mix-unknown"
        elif not contrast_pairs:
            reason = "stable baseline support exists but no comparable elevated or transition pair does"
        else:  # pragma: no cover - defensive; reachable only under future vocabulary edits
            reason = "stable baseline support exists with an unclassified contrast gap"
    return {
        "history_id": history_id,
        "horizon": horizon_name,
        "mode": mode,
        "reference_state": reference_state,
        "n_periods": len(ordered_periods),
        "period_status_counts": _counts(row["status"] for row in ordered_periods),
        "n_runs": len(runs),
        "run_status_counts": _counts(run["status"] for run in runs),
        "stable_segments": stable_segments,
        "n_pairs": len(segment_pairs),
        "pair_region_counts": _counts(row["region"] for row in segment_pairs),
        "n_baseline_stable_pairs_overlapping": len(baseline_pairs),
        "n_baseline_stable_pairs_nonoverlapping": len(nonoverlap_baseline),
        "nonoverlapping_baseline_pair_identities": nonoverlap_baseline,
        "n_distinct_baseline_periods": len(distinct_baseline_periods),
        "distinct_baseline_period_identities": distinct_baseline_periods,
        "n_baseline_mix_ok_pairs": len(baseline_mix_ok),
        "n_contrast_pairs_overlapping": len(contrast_pairs),
        "n_contrast_pairs_nonoverlapping": len(nonoverlap_contrast),
        "nonoverlapping_contrast_pair_identities": nonoverlap_contrast,
        "exposure_compatible_elevated_periods": exposure_compatible,
        "cross_region_mix_status": "UNKNOWN-MISSING-DIMENSION",
        "disposition": disposition,
        "disposition_reason": reason,
    }


def _exposure_compatible_elevated(
    ordered_periods: Sequence[Mapping[str, object]],
    stable_segments: Sequence[Mapping[str, object]],
    elevated_identities: Sequence[str],
) -> list[dict[str, object]]:
    """List elevated periods exposure-compatible with at least one segment period.

    Exposure compatibility is the exposure-ratio floor applied to period
    exposure-days. Robot/program mix drift between nonadjacent periods is not
    persisted in the study manifests, so cross-region mix comparability is an
    explicit missing dimension, never an assumed match.
    """
    exposure = {str(row["period_identity"]): float(row["exposure_days"]) for row in ordered_periods}
    out: list[dict[str, object]] = []
    for run in stable_segments:
        members = [str(identity) for identity in run["period_identities"]]  # type: ignore[union-attr]
        compatible: list[str] = []
        for elevated in elevated_identities:
            if elevated in members:
                continue
            for member in members:
                denominator = max(exposure[member], exposure[elevated])
                ratio = 1.0 if denominator == 0 else min(exposure[member], exposure[elevated]) / denominator
                if ratio >= EXPOSURE_RATIO_FLOOR:
                    compatible.append(str(elevated))
                    break
        out.append(
            {
                "segment_period_identities": members,
                "n_exposure_compatible_elevated_periods": len(compatible),
                "exposure_compatible_elevated_period_identities": sorted(compatible),
                "cross_region_mix": "UNKNOWN-MISSING-DIMENSION",
            }
        )
    return out


def _counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:  # type: ignore[union-attr]
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def audit_manifest(
    study_manifest: Mapping[str, object],
    posthoc: Mapping[str, object],
) -> dict[str, object]:
    """Run the frozen audit over every declared history x horizon x mode cell."""
    assert_candidate_construction_label_blind()
    manifest_periods = study_manifest["periods"]
    manifest_pairs = study_manifest["pairs"]
    assert isinstance(manifest_periods, list) and isinstance(manifest_pairs, list)
    posthoc_periods = {str(row["period_identity"]): row for row in posthoc["periods"]}  # type: ignore[union-attr]
    posthoc_pairs = {str(row["pair_identity"]): row for row in posthoc["pairs"]}  # type: ignore[union-attr]
    period_index = {str(row["period_identity"]): row for row in manifest_periods}
    pair_index = {str(row["pair_identity"]): row for row in manifest_pairs}
    references = {str(row["history_id"]): str(row["reference_state"]) for row in study_manifest["references"]}  # type: ignore[union-attr]
    histories = sorted(references)
    period_rows: list[dict[str, object]] = []
    for identity in sorted(period_index):
        structural = period_index[identity]
        if str(structural["horizon"]) not in HORIZONS or str(structural["mode"]) not in MODES:
            continue
        period_rows.append(classify_period(structural, posthoc_periods.get(identity)))
    period_status = {row["period_identity"]: str(row["status"]) for row in period_rows}
    pair_rows: list[dict[str, object]] = []
    for identity in sorted(pair_index):
        structural = pair_index[identity]
        if str(structural["horizon"]) not in HORIZONS or str(structural["mode"]) not in MODES:
            continue
        pair_rows.append(classify_pair(structural, posthoc_pairs.get(identity), period_status))
    cells: list[dict[str, object]] = []
    for history_id in histories:
        for horizon_name in HORIZONS:
            for mode in MODES:
                cells.append(
                    summarize_history(
                        history_id,
                        horizon_name,
                        mode,
                        [row for row in period_rows if row["history_id"] == history_id and row["horizon"] == horizon_name and row["mode"] == mode],
                        [row for row in pair_rows if row["history_id"] == history_id and row["horizon"] == horizon_name and row["mode"] == mode],
                        references[history_id],
                    )
                )
    return {
        "audit_id": AUDIT_ID,
        "audit_sha256": audit_sha256(),
        "histories": histories,
        "period_rows": period_rows,
        "pair_rows": pair_rows,
        "cells": cells,
        "overall": overall_disposition(cells),
    }


def overall_disposition(cells: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return the per-horizon verdict over the primary calendar mode."""
    out: dict[str, object] = {}
    for horizon_name in HORIZONS:
        primary = [
            cell for cell in cells
            if str(cell["horizon"]) == horizon_name and str(cell["mode"]) == PRIMARY_MODE
        ]
        feasible = sum(1 for cell in primary if str(cell["disposition"]) == "FEASIBLE")
        segment_histories = sum(
            1 for cell in primary
            if any(str(run["status"]) == "STABLE-BASELINE-SEGMENT" for run in cell["stable_segments"])  # type: ignore[union-attr]
        )
        if feasible >= MIN_FEASIBLE_HISTORIES:
            verdict: str = "FEASIBLE"
            detail = f"{feasible} of {len(primary)} histories FEASIBLE against {MIN_FEASIBLE_HISTORIES}"
        elif segment_histories > 0:
            verdict = "LIMITED"
            detail = (
                f"{feasible} of {len(primary)} histories FEASIBLE but "
                f"{segment_histories} hold at least one stable segment"
            )
        else:
            verdict = "UNSUPPORTED"
            detail = "no stable baseline segment exists in any history at this horizon"
        out[horizon_name] = {
            "mode": PRIMARY_MODE,
            "n_histories": len(primary),
            "n_feasible": feasible,
            "n_histories_with_stable_segment": segment_histories,
            "disposition": verdict,
            "disposition_reason": detail,
        }
    return out


__all__ = [
    "AUDIT_ID",
    "HORIZONS",
    "MODES",
    "PRIMARY_MODE",
    "LOW_BURDEN_FRACTION_PRIMARY",
    "LOW_BURDEN_FRACTION_LADDER",
    "ELEVATED_FRACTION_PRIMARY",
    "ELEVATED_FRACTION_LADDER",
    "STABILITY_RANGE_PRIMARY",
    "MIN_STABLE_RUN_PERIODS",
    "MIN_PERIOD_FILES",
    "MIN_REFERENCE_FILES",
    "EXPOSURE_RATIO_FLOOR",
    "MIX_DRIFT_REFERENCE",
    "MIN_FEASIBLE_HISTORIES",
    "PERIOD_STATUSES",
    "RUN_STATUSES",
    "PAIR_REGIONS",
    "MIX_FLAGS",
    "DISPOSITIONS",
    "PERIOD_CANDIDATE_FIELDS",
    "PAIR_REGION_FIELDS",
    "audit_spec",
    "audit_sha256",
    "audit_freeze",
    "assert_candidate_construction_label_blind",
    "candidate_period_keys",
    "candidate_pair_keys",
    "order_period_identities",
    "classify_period",
    "stable_runs",
    "mix_flag",
    "classify_pair",
    "mark_segment_members",
    "select_nonoverlapping_pairs",
    "summarize_history",
    "audit_manifest",
    "overall_disposition",
]
