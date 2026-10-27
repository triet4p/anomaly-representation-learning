"""Weekly matched-comparison feasibility audit (Sprint 24 Task 7-M6, metadata only).

This module audits whether the accepted Development metadata can support a
deterministic, label-blind weekly matched comparison between low-burden stable
baseline regions and elevated-burden contrast regions (plus baseline-versus-
baseline controls). It is feasibility machinery, not a detector, not a movement
measurement, and not a new performance gate.

Owner clarification carried in (not re-derived): a normal-trending comparison
region may contain a small amount of faults; zero faults are not required.
Labels select post-hoc region *types* and quantify retained burden contrast;
they never select individual retained files. Post-balancing file fractions may
shift even under label-blind selection: contrast is measured, never assumed.

Binding boundaries (inherited, unchanged):

- Period/pair membership, exposure, reference custody and comparability come
  exclusively from the frozen weekly/monthly study machinery
  (:mod:`representation.sprint24_weekly_monthly_study` built on
  :mod:`representation.sprint24_business_horizon`). Candidate *sets* are the
  label-blind manifest rows; post-hoc burden annotations only *classify* them
  afterwards. :func:`assert_matching_construction_label_blind` fails closed if
  selection/matching construction references an annotation name.
- Inherited floors and references are reused unchanged from the M5 audit:
  30 eligible files per period, 200 reference files per history, 0.8
  exposure-ratio floor, 0.25 mix-drift reference; the M5 low-burden ceiling
  (0.10), elevated floor (0.20) and stability range (0.05) with its two-period
  minimum are reused unchanged. The audit classifies support; it never tunes a
  floor or rescues a failed cell.
- No threshold rescue: no burden, stability, exposure or balance constant is
  adjusted after counting to obtain support.
- Cost: zero encoder forwards. No model, checkpoint, waveform, embedding,
  score or cache is opened anywhere in this module. The three accepted encoder
  seeds are metadata-identical and add no independent data.
- Within-file burden (per-file severity/area) has no accepted annotation
  field: only the file-level ``file_label`` exists. File prevalence is
  classified; within-file burden is reported as an unavailable dimension.

Frozen matching design (``S24-WEEKLY-MATCHED-v1``, research assumptions frozen
before any feasibility count; never owner-approved thresholds):

- Weekly calendar mode only (primary). Month is deferred after the M5
  ``UNSUPPORTED`` verdict; no monthly claim exists here. Rolling mode is not
  enumerated: the M6 question is adjacent-week common support, and the M5
  rolling sensitivity already reproduced the weekly pattern.
- Stratum: the observable ``(robot_id, program_id)`` file cell. Every balancing
  set is a set of strata; per-file retention inside a stratum is a
  deterministic hash of an inherited null seed and never reads a label.
- Baseline region per history: the union of that history's M5 primary-calendar
  stable-segment members (exact M5 period identities; histories without a
  stable segment contribute no baseline region and no comparison).
- Contrast region per history: the M5 exposure-compatible elevated periods
  for the same history (exact M5 identities, ``ELEVATED-CANDIDATE`` periods
  whose period-exposure-days ratio against at least one segment member meets
  the inherited 0.8 floor). Elevated periods are post-hoc region *types*; the
  elevated pole is the binding constraint (only 21 elevated periods exist
  grid-wide, zero at month).
- Common support: strata present in both the baseline and the contrast region
  of the same history. Files outside common support are excluded label-blind
  (stratum membership is observable; no label is read to exclude).
- Deterministic balancing rule (``capped-n``): within each common stratum,
  keep the first ``min(n_baseline, n_contrast)`` members of each arm ordered by
  ``sha256(f'{null_seed}:{history_id}:{region_identity}:{file_id}')`` with the
  inherited null seed (2407). Equal arm sizes per stratum, exact replication
  of the M3 ``CAPPED_N_RULE`` ordering idea applied per stratum instead of per
  period. Retained members are hash-selected; labels only count the retained
  burden afterwards.
- Balance diagnostics: robot-mix total-variation drift between the two arms
  before and after balancing (same ``_tv_distance`` construction as the
  inherited machinery), exposure-days ratio of the two arms, and per-arm
  retained abnormal fractions measured post-hoc.
- Baseline-versus-baseline controls: only the second stable segment of the one
  history holding two segments (DEV-04) provides a disjoint, exposure-matched
  reserve control; every other control candidate shares a period with (or is)
  the baseline region and is rejected as non-independent support.

Feasibility rules (frozen research assumptions, not owner-approved gates; no
positive result is required):

- ``FEASIBLE``: at least one history holds a balanced baseline-versus-elevated
  comparison with common support on at least two strata, at least 10 retained
  files per arm, retained burden contrast (elevated minus baseline) of at
  least 0.10, and an available disjoint baseline control.
- ``LIMITED``: at least one history holds a balanced comparison meeting the
  support minima (two strata, ten files per arm) but no history meets the full
  ``FEASIBLE`` bar (contrast lost under balancing or no disjoint control).
- ``UNSUPPORTED``: no history meets even the support minima.

Disjoint support versus statistical independence: ``disjoint`` below means the
two regions share no file and no period; it is never a claim of statistical
independence. Adjacent chronology (consecutive-week pairs sharing a period)
is distinguished from nonadjacent research pairs (baseline segment versus a
distant elevated period sharing neither files nor periods).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from representation import sprint24_business_horizon as horizon
from representation import sprint24_low_burden_baseline as baseline
from representation import sprint24_weekly_monthly_study as study
from representation.sprint24_context_audit import sha256_object

#: Matching design family identifier frozen by this module.
MATCHING_ID = "S24-WEEKLY-MATCHED-v1"

#: Audited horizon. Month is deferred after the M5 UNSUPPORTED verdict; day was
#: already deferred by owner prioritization. No claim exists for either here.
HORIZONS = ("week",)

#: Boundary convention. Calendar is primary; rolling is not enumerated for M6
#: (the M5 rolling sensitivity already reproduced the weekly pattern).
MODES = ("calendar",)
PRIMARY_MODE = "calendar"

#: Frozen stratum key: the observable (robot_id, program_id) file cell.
STRATUM_FIELDS = ("robot_id", "program_id")

#: Inherited null seed used for the deterministic capped-n ordering.
NULL_SEED = horizon.PROPOSED_NULL_SEED

#: Minimum common strata for a balanced comparison to count as supported.
MIN_COMMON_STRATA = 2

#: Minimum retained files per arm after balancing.
MIN_RETAINED_PER_ARM = 10

#: Minimum retained burden contrast (elevated minus baseline abnormal fraction).
MIN_RETAINED_CONTRAST = 0.10

#: Inherited floors and references, re-declared so the audit cannot silently
#: change them.
MIN_PERIOD_FILES = horizon.MIN_PERIOD_FILES
MIN_REFERENCE_FILES = horizon.MIN_REFERENCE_FILES
EXPOSURE_RATIO_FLOOR = horizon.PROPOSED_EXPOSURE_RATIO_FLOOR
MIX_DRIFT_REFERENCE = baseline.MIX_DRIFT_REFERENCE

#: Reused M5 burden/stability research assumptions (never owner thresholds).
LOW_BURDEN_FRACTION_PRIMARY = baseline.LOW_BURDEN_FRACTION_PRIMARY
ELEVATED_FRACTION_PRIMARY = baseline.ELEVATED_FRACTION_PRIMARY
STABILITY_RANGE_PRIMARY = baseline.STABILITY_RANGE_PRIMARY
MIN_STABLE_RUN_PERIODS = baseline.MIN_STABLE_RUN_PERIODS

#: Minimum FEASIBLE histories (of four) for a horizon-level FEASIBLE verdict.
MIN_FEASIBLE_HISTORIES = 3

#: Closed comparison-type vocabulary.
COMPARISON_TYPES = (
    "BASELINE-VS-ELEVATED",
    "BASELINE-VS-BASELINE",
)

#: Closed member-source vocabulary (which M5 identities feed each arm).
MEMBER_SOURCES = (
    "M5-STABLE-SEGMENT",
    "M5-EXPOSURE-COMPATIBLE-ELEVATED",
    "M5-RESERVE-SEGMENT",
)

#: Closed balancing-rule vocabulary.
BALANCING_RULES = ("CAPPED-N-PER-STRATUM",)

#: Closed feasibility disposition vocabulary (support classification, never
#: performance).
DISPOSITIONS = ("FEASIBLE", "LIMITED", "UNSUPPORTED")

#: Reused M5 disposition vocabulary for the inherited cell verdicts.
M5_DISPOSITIONS = baseline.DISPOSITIONS

#: Frozen per-comparison output fields.
COMPARISON_FIELDS: tuple[str, ...] = (
    "history_id",
    "horizon",
    "mode",
    "comparison_type",
    "comparison_identity",
    "baseline_period_identities",
    "contrast_period_identities",
    "baseline_source",
    "contrast_source",
    "chronology",
    "shared_period_identities",
    "shared_file_identities",
    "retained_baseline_file_identities",
    "retained_contrast_file_identities",
    "balancing_rule",
    "n_strata_total_baseline",
    "n_strata_total_contrast",
    "n_common_strata",
    "common_stratum_identities",
    "n_files_baseline_before",
    "n_files_contrast_before",
    "n_files_baseline_after",
    "n_files_contrast_after",
    "n_excluded_baseline",
    "n_excluded_contrast",
    "attrition_fraction_baseline",
    "attrition_fraction_contrast",
    "mix_drift_tv_before",
    "mix_drift_tv_after",
    "exposure_ratio_before",
    "exposure_ratio_after",
    "burden_baseline_before",
    "burden_contrast_before",
    "burden_contrast_gap_before",
    "burden_baseline_after",
    "burden_contrast_after",
    "burden_contrast_gap_after",
    "contrast_preserved",
    "meets_support_minima",
    "meets_contrast_minimum",
    "has_disjoint_control",
    "disposition",
    "disposition_reason",
)

#: Frozen per-exclusion output fields (stratum-level audit trail).
EXCLUSION_FIELDS: tuple[str, ...] = (
    "history_id",
    "horizon",
    "mode",
    "comparison_identity",
    "stratum_identity",
    "present_in_baseline",
    "present_in_contrast",
    "n_baseline_members",
    "n_contrast_members",
    "in_common_support",
    "exclusion_reason",
)

def matching_spec() -> dict[str, object]:
    """Return the frozen matching identity, grid, rules and vocabularies."""
    return {
        "matching_id": MATCHING_ID,
        "upstream_baseline_audit": {
            "audit_id": baseline.AUDIT_ID,
            "audit_sha256": baseline.audit_sha256(),
        },
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
            "deferred_horizons": ["day", "month"],
            "deferral_reason": (
                "day deferred by owner prioritization; month deferred after "
                "the M5 UNSUPPORTED verdict (no stable monthly segment)"
            ),
        },
        "stratum": {
            "fields": list(STRATUM_FIELDS),
            "note": "observable (robot_id, program_id) file cell; stratum membership never reads a label",
        },
        "balancing_rule": {
            "rule": "CAPPED-N-PER-STRATUM",
            "order_key": "sha256(f'{null_seed}:{history_id}:{region_identity}:{file_id}')",
            "selection": "ascending order key, keep the first min(n_baseline, n_contrast) members per common stratum per arm",
            "null_seed": NULL_SEED,
            "label_blind": True,
        },
        "research_assumptions": {
            "low_burden_fraction_primary": LOW_BURDEN_FRACTION_PRIMARY,
            "elevated_fraction_primary": ELEVATED_FRACTION_PRIMARY,
            "stability_range_primary": STABILITY_RANGE_PRIMARY,
            "min_stable_run_periods": MIN_STABLE_RUN_PERIODS,
            "min_common_strata": MIN_COMMON_STRATA,
            "min_retained_per_arm": MIN_RETAINED_PER_ARM,
            "min_retained_contrast": MIN_RETAINED_CONTRAST,
            "min_feasible_histories": MIN_FEASIBLE_HISTORIES,
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
            "comparison_type": list(COMPARISON_TYPES),
            "member_source": list(MEMBER_SOURCES),
            "balancing_rule": list(BALANCING_RULES),
            "disposition": list(DISPOSITIONS),
        },
    }


def matching_sha256() -> str:
    """Return the freeze digest of the matching identity/grid/rules."""
    return sha256_object(matching_spec())


def matching_freeze() -> dict[str, object]:
    """Return the matching spec together with its freeze digest."""
    spec = matching_spec()
    return {"matching_sha256": sha256_object(spec), "spec": spec}


#: Names that must never appear in selection/matching construction code.
_FORBIDDEN_MATCHING_TOKENS = (
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


def _matching_construction_functions() -> tuple[Callable[..., object], ...]:
    return (
        region_member_keys,
        common_strata,
        capped_n_retained,
    )


def assert_matching_construction_label_blind() -> None:
    """Fail closed if selection/matching construction references annotations."""
    for fn in _matching_construction_functions():
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
        for token in _FORBIDDEN_MATCHING_TOKENS:
            if token in code:
                raise RuntimeError(f"label-blind violation: {fn.__name__} references {token!r}")


def stratum_identity(robot_id: str, program_id: str) -> str:
    """Return the deterministic stratum identity for one observable cell."""
    return f"{robot_id}|{program_id}"


def order_key(history_id: str, region_identity: str, file_id: str) -> str:
    """Return the deterministic capped-n ordering key for one file."""
    return hashlib.sha256(f"{NULL_SEED}:{history_id}:{region_identity}:{file_id}".encode("utf-8")).hexdigest()


def region_member_keys(
    member_ids: Sequence[str],
    region_identity: str,
    history_id: str,
) -> list[str]:
    """Return region member keys ordered by the deterministic hash (label-blind).

    The ordering key is a hash of the inherited null seed, the history, the
    region identity and the file ID only. No label, event or burden value
    enters the order.
    """
    return sorted((str(file_id) for file_id in member_ids), key=lambda fid: order_key(history_id, region_identity, fid))


def common_strata(
    baseline_cells: Mapping[str, str],
    contrast_cells: Mapping[str, str],
) -> list[str]:
    """Return the sorted common stratum identities (label-blind).

    Each mapping carries file ID to stratum identity built from observable
    ``robot_id``/``program_id`` columns only.
    """
    baseline_present = set(baseline_cells.values())
    contrast_present = set(contrast_cells.values())
    return sorted(baseline_present & contrast_present)


def capped_n_retained(
    ordered_member_ids: Sequence[str],
    cell_of: Mapping[str, str],
    strata: Sequence[str],
    cap_per_stratum: Mapping[str, int],
) -> list[str]:
    """Return the capped-n retained members over common strata (label-blind).

    Within each common stratum the first ``cap`` members of the pre-ordered
    (hash-ordered) list are retained. Strata outside common support contribute
    nothing; the cap never reads a label.
    """
    cap = {str(key): int(value) for key, value in cap_per_stratum.items()}
    wanted = set(str(item) for item in strata)
    seen: dict[str, int] = {}
    retained: list[str] = []
    for file_id in ordered_member_ids:
        stratum = str(cell_of.get(str(file_id), ""))
        if stratum not in wanted:
            continue
        used = seen.get(stratum, 0)
        if used < cap.get(stratum, 0):
            retained.append(str(file_id))
            seen[stratum] = used + 1
    return retained


def tv_distance(first: Sequence[float], second: Sequence[float]) -> float | None:
    """Return the total-variation distance between two count vectors."""

    array_first = np.asarray([float(value) for value in first], dtype=np.float64)
    array_second = np.asarray([float(value) for value in second], dtype=np.float64)
    total_first = float(array_first.sum())
    total_second = float(array_second.sum())
    if total_first == 0 or total_second == 0:
        return None
    return float(0.5 * float(np.abs(array_first / total_first - array_second / total_second).sum()))


def mix_counts(cells: Sequence[str], universe: Sequence[str]) -> list[float]:
    """Return per-stratum counts of cells over the frozen stratum universe."""
    order = [str(item) for item in universe]
    position = {key: index for index, key in enumerate(order)}
    counts = [0.0] * len(order)
    for cell in cells:
        index = position.get(str(cell))
        if index is not None:
            counts[index] += 1.0
    return counts


def comparison_identity(
    history_id: str,
    horizon_name: str,
    mode: str,
    comparison_type: str,
    tag: str,
) -> str:
    """Return the deterministic comparison identity."""
    return f"{history_id}|{horizon_name}|{mode}|{comparison_type}|{tag}"


def _fraction(abnormal: int, total: int) -> float | None:
    if total <= 0:
        return None
    return float(abnormal) / float(total)


def _ratio(first: float, second: float) -> float | None:
    denominator = max(first, second)
    if denominator == 0:
        return None
    return min(first, second) / denominator


def balance_comparison(
    history_id: str,
    horizon_name: str,
    mode: str,
    comparison_type: str,
    tag: str,
    baseline_period_identities: Sequence[str],
    contrast_period_identities: Sequence[str],
    baseline_source: str,
    contrast_source: str,
    baseline_files: Sequence[str],
    contrast_files: Sequence[str],
    baseline_cell_of: Mapping[str, str],
    contrast_cell_of: Mapping[str, str],
    baseline_abnormal: Mapping[str, int],
    contrast_abnormal: Mapping[str, int],
    baseline_exposure_days: float,
    contrast_exposure_days: float,
    baseline_strata_universe: Sequence[str] | None = None,
    has_disjoint_control: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Balance one baseline-versus-contrast comparison deterministically.

    Selection is label-blind: common strata come from observable cells and the
    capped-n retention is hash-ordered. Abnormal counts are joined only
    afterwards to measure retained burden contrast. Returns the comparison row
    and the per-stratum exclusion audit trail.
    """
    base_ids = [str(item) for item in baseline_files]
    cont_ids = [str(item) for item in contrast_files]
    base_cells = {fid: str(baseline_cell_of[fid]) for fid in base_ids if fid in baseline_cell_of}
    cont_cells = {fid: str(contrast_cell_of[fid]) for fid in cont_ids if fid in contrast_cell_of}
    strata = common_strata(base_cells, cont_cells)
    universe = [str(item) for item in (baseline_strata_universe or sorted(set(base_cells.values()) | set(cont_cells.values())))]

    base_by_stratum: dict[str, list[str]] = {}
    cont_by_stratum: dict[str, list[str]] = {}
    for fid in base_ids:
        cell = base_cells.get(fid)
        if cell is not None:
            base_by_stratum.setdefault(cell, []).append(fid)
    for fid in cont_ids:
        cell = cont_cells.get(fid)
        if cell is not None:
            cont_by_stratum.setdefault(cell, []).append(fid)

    region_base = f"{history_id}|{horizon_name}|{mode}|BASELINE-REGION"
    region_cont = f"{history_id}|{horizon_name}|{mode}|CONTRAST-REGION"
    ordered_base = region_member_keys(base_ids, region_base, history_id)
    ordered_cont = region_member_keys(cont_ids, region_cont, history_id)
    cap = {stratum: min(len(base_by_stratum.get(stratum, [])), len(cont_by_stratum.get(stratum, []))) for stratum in strata}
    retained_base = capped_n_retained(ordered_base, base_cells, strata, cap)
    retained_cont = capped_n_retained(ordered_cont, cont_cells, strata, cap)

    mix_before = tv_distance(
        mix_counts([base_cells[fid] for fid in base_ids if fid in base_cells], universe),
        mix_counts([cont_cells[fid] for fid in cont_ids if fid in cont_cells], universe),
    )
    mix_after = tv_distance(
        mix_counts([base_cells[fid] for fid in retained_base], universe),
        mix_counts([cont_cells[fid] for fid in retained_cont], universe),
    )

    abnormal_before_base = sum(int(baseline_abnormal.get(fid, 0)) for fid in base_ids)
    abnormal_before_cont = sum(int(contrast_abnormal.get(fid, 0)) for fid in cont_ids)
    abnormal_after_base = sum(int(baseline_abnormal.get(fid, 0)) for fid in retained_base)
    abnormal_after_cont = sum(int(contrast_abnormal.get(fid, 0)) for fid in retained_cont)
    burden_base_before = _fraction(abnormal_before_base, len(base_ids))
    burden_cont_before = _fraction(abnormal_before_cont, len(cont_ids))
    burden_base_after = _fraction(abnormal_after_base, len(retained_base))
    burden_cont_after = _fraction(abnormal_after_cont, len(retained_cont))
    gap_before = None if burden_base_before is None or burden_cont_before is None else burden_cont_before - burden_base_before
    gap_after = None if burden_base_after is None or burden_cont_after is None else burden_cont_after - burden_base_after

    n_excluded_base = len(base_ids) - len(retained_base)
    n_excluded_cont = len(cont_ids) - len(retained_cont)
    attrition_base = None if not base_ids else n_excluded_base / len(base_ids)
    attrition_cont = None if not cont_ids else n_excluded_cont / len(cont_ids)
    exposure_before = _ratio(baseline_exposure_days, contrast_exposure_days)

    meets_support = (
        len(strata) >= MIN_COMMON_STRATA
        and len(retained_base) >= MIN_RETAINED_PER_ARM
        and len(retained_cont) >= MIN_RETAINED_PER_ARM
    )
    meets_contrast = gap_after is not None and gap_after >= MIN_RETAINED_CONTRAST
    if meets_support and meets_contrast and has_disjoint_control:
        disposition = "FEASIBLE"
        reason = (
            "balanced comparison meets common-strata, retained-size and "
            "retained-contrast minima with a disjoint baseline control"
        )
    elif meets_support and meets_contrast:
        disposition = "LIMITED"
        reason = "balanced comparison meets support and contrast minima but no disjoint baseline control exists"
    elif meets_support:
        disposition = "LIMITED"
        reason = "balanced comparison meets support minima but retained burden contrast is below the minimum"
    else:
        disposition = "UNSUPPORTED"
        reason = "common support or retained size is below the support minima"

    base_periods = [str(item) for item in baseline_period_identities]
    cont_periods = [str(item) for item in contrast_period_identities]
    shared_periods = sorted(set(base_periods) & set(cont_periods))
    shared_files = sorted(set(base_ids) & set(cont_ids))
    if shared_periods or shared_files:
        chronology = "NONADJACENT-SHARED-SUPPORT"
    else:
        chronology = "NONADJACENT-RESEARCH-PAIR"
    identity = comparison_identity(history_id, horizon_name, mode, comparison_type, tag)
    row: dict[str, object] = {
        "history_id": history_id,
        "horizon": horizon_name,
        "mode": mode,
        "comparison_type": comparison_type,
        "comparison_identity": identity,
        "baseline_period_identities": sorted(base_periods),
        "contrast_period_identities": sorted(cont_periods),
        "baseline_source": baseline_source,
        "contrast_source": contrast_source,
        "chronology": chronology,
        "shared_period_identities": shared_periods,
        "shared_file_identities": shared_files,
        "retained_baseline_file_identities": sorted(retained_base),
        "retained_contrast_file_identities": sorted(retained_cont),
        "balancing_rule": "CAPPED-N-PER-STRATUM",
        "n_strata_total_baseline": len(set(base_cells.values())),
        "n_strata_total_contrast": len(set(cont_cells.values())),
        "n_common_strata": len(strata),
        "common_stratum_identities": list(strata),
        "n_files_baseline_before": len(base_ids),
        "n_files_contrast_before": len(cont_ids),
        "n_files_baseline_after": len(retained_base),
        "n_files_contrast_after": len(retained_cont),
        "n_excluded_baseline": n_excluded_base,
        "n_excluded_contrast": n_excluded_cont,
        "attrition_fraction_baseline": attrition_base,
        "attrition_fraction_contrast": attrition_cont,
        "mix_drift_tv_before": mix_before,
        "mix_drift_tv_after": mix_after,
        "exposure_ratio_before": exposure_before,
        "exposure_ratio_after": exposure_before,
        "burden_baseline_before": burden_base_before,
        "burden_contrast_before": burden_cont_before,
        "burden_contrast_gap_before": gap_before,
        "burden_baseline_after": burden_base_after,
        "burden_contrast_after": burden_cont_after,
        "burden_contrast_gap_after": gap_after,
        "contrast_preserved": meets_contrast,
        "meets_support_minima": meets_support,
        "meets_contrast_minimum": meets_contrast,
        "has_disjoint_control": bool(has_disjoint_control),
        "disposition": disposition,
        "disposition_reason": reason,
    }

    exclusions: list[dict[str, object]] = []
    for stratum in universe:
        in_base = stratum in base_by_stratum
        in_cont = stratum in cont_by_stratum
        in_common = stratum in strata
        if in_common:
            reason_text = "retained under capped-n per stratum"
        elif in_base and in_cont:
            reason_text = "unreachable: stratum present on both arms but outside common support"
        elif in_base:
            reason_text = "excluded: stratum absent from the contrast region (no common support)"
        elif in_cont:
            reason_text = "excluded: stratum absent from the baseline region (no common support)"
        else:
            reason_text = "stratum in universe but empty on both arms"
        exclusions.append(
            {
                "history_id": history_id,
                "horizon": horizon_name,
                "mode": mode,
                "comparison_identity": identity,
                "stratum_identity": stratum,
                "present_in_baseline": bool(in_base),
                "present_in_contrast": bool(in_cont),
                "n_baseline_members": len(base_by_stratum.get(stratum, [])),
                "n_contrast_members": len(cont_by_stratum.get(stratum, [])),
                "in_common_support": bool(in_common),
                "exclusion_reason": reason_text,
            }
        )
    return row, exclusions


def summarize_matching(
    comparisons: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return the per-horizon feasibility verdict over balanced comparisons."""
    primary = [row for row in comparisons if str(row["horizon"]) == "week" and str(row["mode"]) == PRIMARY_MODE]
    feasible = sum(1 for row in primary if str(row["disposition"]) == "FEASIBLE")
    supported = sum(1 for row in primary if bool(row["meets_support_minima"]))
    if feasible >= MIN_FEASIBLE_HISTORIES:
        verdict = "FEASIBLE"
        detail = f"{feasible} of 4 histories FEASIBLE against {MIN_FEASIBLE_HISTORIES}"
    elif supported > 0:
        verdict = "LIMITED"
        detail = (
            f"{feasible} of 4 histories FEASIBLE but "
            f"{supported} hold at least one balanced comparison meeting the support minima"
        )
    else:
        verdict = "UNSUPPORTED"
        detail = "no history holds a balanced comparison meeting the support minima"
    return {
        "mode": PRIMARY_MODE,
        "n_histories": 4,
        "n_feasible": feasible,
        "n_histories_with_supported_comparison": supported,
        "disposition": verdict,
        "disposition_reason": detail,
    }


__all__ = [
    "MATCHING_ID",
    "HORIZONS",
    "MODES",
    "PRIMARY_MODE",
    "STRATUM_FIELDS",
    "NULL_SEED",
    "MIN_COMMON_STRATA",
    "MIN_RETAINED_PER_ARM",
    "MIN_RETAINED_CONTRAST",
    "MIN_FEASIBLE_HISTORIES",
    "MIN_PERIOD_FILES",
    "MIN_REFERENCE_FILES",
    "EXPOSURE_RATIO_FLOOR",
    "MIX_DRIFT_REFERENCE",
    "LOW_BURDEN_FRACTION_PRIMARY",
    "ELEVATED_FRACTION_PRIMARY",
    "STABILITY_RANGE_PRIMARY",
    "MIN_STABLE_RUN_PERIODS",
    "COMPARISON_TYPES",
    "MEMBER_SOURCES",
    "BALANCING_RULES",
    "DISPOSITIONS",
    "M5_DISPOSITIONS",
    "COMPARISON_FIELDS",
    "EXCLUSION_FIELDS",
    "matching_spec",
    "matching_sha256",
    "matching_freeze",
    "assert_matching_construction_label_blind",
    "stratum_identity",
    "order_key",
    "region_member_keys",
    "common_strata",
    "capped_n_retained",
    "tv_distance",
    "mix_counts",
    "comparison_identity",
    "balance_comparison",
    "summarize_matching",
]
