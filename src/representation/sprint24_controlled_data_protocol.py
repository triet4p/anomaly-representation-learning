"""Prospective controlled-data protocol design (Sprint 24 Task 7-M7, design only).

This module freezes a versioned, machine-checkable *design* for a bounded
supplemental synthetic research corpus supporting weekly/monthly collective
distribution/cluster movement comparisons with low-nonzero-burden stable
baseline-to-baseline controls and baseline-to-elevated-burden transitions.

Design-only boundary (binding):

- No waveform is synthesized, no manifest is materialized, no embedding is
  read, no forward is executed anywhere in this module. Every count below is
  a *planned quota* or a *stated assumption*, never a measurement.
- The planned structural identities (history/period/reference/pair keys) are
  built label-blind by construction; burden bands are attached afterwards as
  a separate annotated layer with its own digest. Forbidden-token AST guards
  fail closed on the structural constructors.
- Constants inherited from frozen predecessors (``S24-BIZ-WM-v1`` calendar,
  M5 burden/stability bands, M6 balancing rule, null seed 2407, accepted
  encoder seeds) are re-declared unchanged; nothing here re-tunes them.
  No DEV-04 or old movement output selected any value; prior audits are
  exploratory motivation, not confirmation.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable, Mapping, Sequence

from representation.sprint24_context_audit import sha256_object

#: Versioned protocol identifier frozen by this module.
PROTOCOL_ID = "S24-CONTROLLED-DATA-v1"

#: Horizons covered. ``day`` is deferred by owner prioritization (unchanged).
HORIZONS = ("week", "month")

#: Boundary convention. Calendar is primary; no rolling sensitivity is
#: declared for the supplemental corpus (the M3 rolling answers already exist
#: on the old corpus; the new corpus tests the primary convention only).
MODES = ("calendar",)
PRIMARY_MODE = "calendar"

# --- inherited research bands (re-declared unchanged, never re-tuned) --------

#: Low-burden ceiling on the abnormal member fraction (M5 primary).
#: A small *nonzero* fraction is eligible; zero faults are not required.
LOW_BURDEN_FRACTION_PRIMARY = 0.10

#: Proposed low-nonzero operating band for baseline scheduling: baseline
#: regions must plan a strictly positive fraction (faults present but few).
LOW_BASELINE_OPERATING_BAND = (0.02, 0.10)

#: Elevated-burden floor (M5 primary): twice the low ceiling.
ELEVATED_FRACTION_PRIMARY = 0.20

#: Proposed elevated operating band for contrast scheduling.
ELEVATED_OPERATING_BAND = (0.20, 0.35)

#: Stability bound: maximum abnormal-fraction range inside a stable run.
STABILITY_RANGE_PRIMARY = 0.05

#: Minimum consecutive low periods forming a segment/run.
MIN_STABLE_RUN_PERIODS = 2

#: Inherited floors, re-declared so the design cannot silently change them.
MIN_PERIOD_FILES = 30
MIN_REFERENCE_FILES = 200
EXPOSURE_RATIO_FLOOR = 0.8
MIX_DRIFT_REFERENCE = 0.25

#: Inherited null seed for every hash-ordered selection (M3/M6 convention).
NULL_SEED = 2407

#: History-cluster bootstrap replicates/seed for descriptive uncertainty.
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 2407

#: Accepted frozen encoder seeds (identity-selected run replicates, never
#: independent data). The encoder stays frozen as-is; any refit needs a
#: separately approved protocol.
FROZEN_ENCODER_SEEDS = (181801, 181802, 181803)
EMBEDDING_DIM = 128
N_CHANNELS = 6

# --- lineage: old corpus immutable, new corpus separate -----------------------

#: Frozen old-corpus history identities (accepted Development corpus).
#: The new corpus must never reuse, extend or mutate these.
OLD_CORPUS_PREFIX = "S18I-ITER-0005"
OLD_HISTORY_IDS = (
    "S18I-ITER-0005-DEVELOPMENT-01",
    "S18I-ITER-0005-DEVELOPMENT-02",
    "S18I-ITER-0005-DEVELOPMENT-03",
    "S18I-ITER-0005-DEVELOPMENT-04",
)

#: Certified old-corpus data-seed block (must not be reused).
CERTIFIED_OLD_SEED_MIN = 32064
CERTIFIED_OLD_SEED_MAX = 32079

#: New supplemental corpus: six independent histories (lower-bound research
#: scale: a 3-history primary triple plus a 3-history replication triple;
#: justified by the support bar and replication, never by a power claim).
NEW_HISTORY_IDS = (
    "S24C-ITER-0001-CONTROLLED-01",
    "S24C-ITER-0001-CONTROLLED-02",
    "S24C-ITER-0001-CONTROLLED-03",
    "S24C-ITER-0001-CONTROLLED-04",
    "S24C-ITER-0001-CONTROLLED-05",
    "S24C-ITER-0001-CONTROLLED-06",
)
PRIMARY_TRIPLE = NEW_HISTORY_IDS[:3]
REPLICATION_TRIPLE = NEW_HISTORY_IDS[3:]

#: Proposed fresh data-seed block (research default; the pre-generation
#: freeze must verify non-collision against every certified range).
PROPOSED_DATA_SEEDS = (42401, 42402, 42403, 42404, 42405, 42406)

#: Proposed generator family tag for lineage (records intent only; the
#: burden-schedule-driven planner itself is a missing prerequisite).
PROPOSED_GENERATOR_VERSION = "synth generator 2.0.0 + burden-schedule planner (to be implemented)"

# --- planned layout: one 26-week span per history serves both horizons -------

#: Planned weekly span length (W00 warmup + 25 analysis weeks).
N_PLANNED_WEEKS = 26

#: Planned monthly span (M01..M06 derived from the same files by end_time).
N_PLANNED_MONTHS = 6

#: Closed weekly band vocabulary (annotation layer, never structural).
WEEK_BANDS = (
    "WARMUP",
    "BASELINE-A",
    "BASELINE-B",
    "ELEVATED-A",
    "ELEVATED-B",
    "BUFFER",
)

#: Closed monthly band vocabulary.
MONTH_BANDS = (
    "WARMUP-PARTIAL",
    "LOW",
    "ELEVATED",
)

#: Default week-resolution band plan (offsets from c1; W00 is warmup and is
#: excluded from every arm; BUFFER weeks separate arms temporally).
DEFAULT_WEEK_BANDS: dict[int, str] = {
    0: "WARMUP",
    1: "BASELINE-A", 2: "BASELINE-A", 3: "BASELINE-A",
    4: "BUFFER",
    5: "ELEVATED-A", 6: "ELEVATED-A",
    7: "BUFFER",
    8: "BASELINE-B", 9: "BASELINE-B", 10: "BASELINE-B",
    11: "BUFFER",
    12: "ELEVATED-B", 13: "ELEVATED-B",
    14: "BUFFER",
    15: "BUFFER", 16: "BUFFER", 17: "BUFFER", 18: "BUFFER",
    19: "BUFFER", 20: "BUFFER", 21: "BUFFER", 22: "BUFFER",
    23: "BUFFER", 24: "BUFFER", 25: "BUFFER",
}

#: Default month-resolution band plan. Monthly bands aggregate the same
#: planned files; the pre-generation freeze verifies on real c1 values that
#: every weekly BASELINE segment nests inside a LOW month and every weekly
#: ELEVATED block nests inside an ELEVATED month.
DEFAULT_MONTH_BANDS: dict[int, str] = {
    1: "WARMUP-PARTIAL",
    2: "LOW",
    3: "LOW",
    4: "ELEVATED",
    5: "LOW",
    6: "LOW",
}

#: Planned per-period file quotas (floors inherited; planned values above).
PLANNED_REFERENCE_FILES = 220
PLANNED_WEEKLY_FILES = 40

#: Planned common-support quota: observable (robot_id, program_id) cells
#: shared by both arms of a within-history comparison.
PLANNED_MIN_COMMON_STRATA = 4

#: Planned retained minimum per arm after capped-n balancing (M6 bar: 2
#: strata / 10 files per arm / 0.10 retained gap; planned values above bar).
PLANNED_MIN_RETAINED_PER_ARM = 12
PLANNED_MIN_RETAINED_GAP = 0.10

#: Closed planned-pair kinds.
PAIR_KINDS = ("TRANSITION", "CONTROL")

#: Closed structural verdicts for a planned history.
PLAN_VERDICTS = ("PLANNED-SUPPORTED", "PLANNED-REFUSED")

#: Closed refusal reasons (a refusal stops; never a retry, reseed or rescue).
REFUSAL_REASONS = (
    "REFUSED-OVERLAP",
    "REFUSED-SHORTFALL",
    "REFUSED-NO-MONTHLY-CONTROL",
    "REFUSED-NO-WEEKLY-CONTROL",
    "REFUSED-SEED-COLLISION",
    "REFUSED-LINEAGE",
    "REFUSED-IDENTITY",
)

#: Forward/budget accounting inherited from the executed M4 study (no reset,
#: no reuse; a future inference needs an explicit new authorization).
CUMULATIVE_FORWARDS_SPENT = 52_789
FORWARD_CAP = 54_000
FORWARD_HEADROOM = FORWARD_CAP - CUMULATIVE_FORWARDS_SPENT

#: Assumed mean file length for storage arithmetic (a stated assumption,
#: inside the RegimeConfig 200-800 step range; never presented as measured).
ASSUMED_MEAN_STEPS_PER_FILE = 400


def calendar_assumptions() -> dict[str, object]:
    """Return the frozen calendar projection reused from ``S24-BIZ-WM-v1``."""
    from representation import sprint24_weekly_monthly_study as study

    primary = study.primary_calendar()
    return {
        "timezone": "UTC",
        "week_start": "monday (ISO-8601)",
        "projection_anchor": "t = 0 maps to 2024-01-01T00:00:00Z (arithmetic projection)",
        "month_boundary": "local first-of-month midnight; never a nominal length",
        "membership_rule": "period_start < end_time <= period_end",
        "warmup_rule": "first week after c1 (W00) and first partial month excluded from every arm",
        "reference_rule": "eligible files with end_time <= c1, fixed per history before any evaluation period",
        "upstream_study_id": study.STUDY_ID,
        "upstream_study_sha256": study.study_sha256(),
        "calendar_describe": primary.describe(),
    }


def protocol_spec() -> dict[str, object]:
    """Return the frozen controlled-data protocol identity, grid and rules."""
    return {
        "protocol_id": PROTOCOL_ID,
        "design_scope": (
            "design only: no waveform synthesis, no manifest materialization, "
            "no embedding/forward, no outcome inspection"
        ),
        "preserved_outcomes": {
            "M4": "INCONCLUSIVE (both horizons; retained, never revised)",
            "M5": "weekly LIMITED, monthly UNSUPPORTED (retained)",
            "M6": "exploratory weekly LIMITED; only DEV-04 feasible (retained)",
        },
        "horizons": list(HORIZONS),
        "modes": list(MODES),
        "primary_mode": PRIMARY_MODE,
        "deferred_horizons": ["day"],
        "calendar": calendar_assumptions(),
        "research_bands": {
            "low_burden_fraction_primary": LOW_BURDEN_FRACTION_PRIMARY,
            "low_baseline_operating_band": list(LOW_BASELINE_OPERATING_BAND),
            "elevated_fraction_primary": ELEVATED_FRACTION_PRIMARY,
            "elevated_operating_band": list(ELEVATED_OPERATING_BAND),
            "stability_range_primary": STABILITY_RANGE_PRIMARY,
            "min_stable_run_periods": MIN_STABLE_RUN_PERIODS,
            "assumption_status": (
                "proposed research defaults frozen before any generation; "
                "not owner-approved operational thresholds or performance gates"
            ),
        },
        "inherited_unchanged": {
            "min_period_files": MIN_PERIOD_FILES,
            "min_reference_files": MIN_REFERENCE_FILES,
            "exposure_ratio_floor": EXPOSURE_RATIO_FLOOR,
            "mix_drift_reference": MIX_DRIFT_REFERENCE,
            "null_seed": NULL_SEED,
            "balancing_rule": "CAPPED-N-PER-STRATUM over observable (robot_id, program_id) cells",
            "eligibility_predicate": "representation.sprint24_views.is_eligible_for_training",
            "analysis_span": "(c1, tmax] per history, never concatenated",
        },
        "scale": {
            "n_histories": len(NEW_HISTORY_IDS),
            "history_ids": list(NEW_HISTORY_IDS),
            "primary_triple": list(PRIMARY_TRIPLE),
            "replication_triple": list(REPLICATION_TRIPLE),
            "scale_justification": (
                "lower-bound research scale justified by the support bar "
                "(each horizon bar needs >= 3 histories with complete "
                "weekly/monthly support, mirroring the M5/M6 three-of-four "
                "bar) plus an independent 3-history replication triple; "
                "a disposition must hold in both triples. No power "
                "calculation is claimed; with n = 6 every cross-history "
                "statement stays descriptive."
            ),
        },
        "lineage": {
            "old_corpus_prefix": OLD_CORPUS_PREFIX,
            "old_history_ids": list(OLD_HISTORY_IDS),
            "old_corpus_policy": "immutable: never reused, extended, mutated or re-read as input",
            "new_histories": list(NEW_HISTORY_IDS),
            "data_seeds_proposed": list(PROPOSED_DATA_SEEDS),
            "generator_lineage": PROPOSED_GENERATOR_VERSION,
        },
        "encoder_custody": {
            "mode": "FROZEN-AS-IS",
            "encoder_seeds": list(FROZEN_ENCODER_SEEDS),
            "embedding_dim": EMBEDDING_DIM,
            "n_channels": N_CHANNELS,
            "reference_policy": (
                "bandwidth, cluster prototypes and standardization are fitted "
                "once per NEW history on its own causal reference pool "
                "(end_time <= c1); old-history banks are never reused"
            ),
            "refit_policy": "any encoder refit needs a separately approved protocol",
        },
        "planned_quotas": {
            "reference_files_per_history": PLANNED_REFERENCE_FILES,
            "weekly_files_per_period": PLANNED_WEEKLY_FILES,
            "planned_weeks": N_PLANNED_WEEKS,
            "planned_months": N_PLANNED_MONTHS,
            "min_common_strata": PLANNED_MIN_COMMON_STRATA,
            "min_retained_per_arm": PLANNED_MIN_RETAINED_PER_ARM,
            "min_retained_gap": PLANNED_MIN_RETAINED_GAP,
            "planned_pairs_per_history_per_horizon": 5,
            "pair_plan": "4 TRANSITION (each baseline segment x each elevated block) + 1 CONTROL (baseline A x baseline B)",
        },
        "endpoints": {
            "E1": "PRIMARY: unbiased squared MMD, Gaussian kernel, bandwidth = median pairwise distance on the new causal reference pool",
            "E2": "SECONDARY: cluster-occupancy total variation, K = 8 (4/16 sensitivity), prototypes on the new reference pool",
            "E3_E4": "DESCRIPTIVE-ONLY decomposition; never decision-bearing",
            "N1": "DESCRIPTIVE-ONLY reference split-half null (sampling crossing is necessary, never sufficient)",
            "N2": "DESCRIPTIVE-ONLY within-period half-split band; control pairs must sit inside it",
            "C1_C2": "within-history rank association of E1 (and mix-adjusted residual) with burden over the 5 planned comparable pairs",
        },
        "estimands": {
            "unit": "history (pairs inside a history are dependent; the within-history statistic is computed once per history)",
            "primary_estimand": "within-history Spearman rank association (C1) and mix-adjusted residual association (C2) over planned comparable pairs; seeds are run replicates summarized by median, never independent data",
            "control_expectation": "baseline-to-baseline control-pair E1 lies inside the N2 band with near-zero burden gap",
            "no_pooling": "no pooling across histories, seeds or horizons to reach a threshold",
        },
        "uncertainty_multiplicity": {
            "uncertainty": "history-cluster bootstrap, B = 2000, seed 2407, descriptive at n = 6",
            "multiplicity": "Holm over the 4 declared primary contrasts (2 horizons x C1/C2) at alpha = 0.05, auxiliary reported inference only; the disposition does not depend on a p-value",
            "triage": "PROMISING / NOT-PROMISING / INCONCLUSIVE prospectively; PROMISING only when every promise criterion holds in BOTH triples",
        },
        "refusal_policy": {
            "on_shortfall": "PLANNED-REFUSED with a closed reason; the attempt stops",
            "forbidden": "no retries, no reseeding, no threshold/band rescue, no silent rescaling",
            "releases": "implementation, generation and inference are separate future releases, each needing explicit authorization and (for inference) an explicit forward budget",
        },
        "vocabularies": {
            "week_band": list(WEEK_BANDS),
            "month_band": list(MONTH_BANDS),
            "pair_kind": list(PAIR_KINDS),
            "plan_verdict": list(PLAN_VERDICTS),
            "refusal_reason": list(REFUSAL_REASONS),
        },
    }


def protocol_sha256() -> str:
    """Return the freeze digest of the protocol identity/grid/rules."""
    return sha256_object(protocol_spec())


# --- validation helpers --------------------------------------------------------


def _check(reason: str, ok: bool, detail: str) -> dict[str, object]:
    return {"reason": reason, "status": "PASS" if ok else "REFUSED", "detail": detail}


def protocol_freeze() -> dict[str, object]:
    """Return the protocol spec together with its freeze digest."""
    spec = protocol_spec()
    return {"protocol_sha256": sha256_object(spec), "spec": spec}


# --- label-blind structural constructors -------------------------------------
#: Names that must never appear in planned-identity construction code.
_FORBIDDEN_PLANNING_TOKENS = (
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
    "baseline",
    "elevated",
    "severity",
)


def _planning_construction_functions() -> tuple[Callable[..., object], ...]:
    return (
        planned_week_identities,
        planned_month_identities,
        planned_reference_identity,
        order_planned_identities,
    )


def assert_planning_construction_label_blind() -> None:
    """Fail closed if planned-identity construction references annotations."""
    for fn in _planning_construction_functions():
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
        for token in _FORBIDDEN_PLANNING_TOKENS:
            if token in code:
                raise RuntimeError(f"label-blind violation: {getattr(fn, '__name__', str(fn))} references {token!r}")


def planned_week_identities(history_id: str) -> list[str]:
    """Return the label-blind planned weekly period identities W00..W25."""
    return [f"{history_id}|week|calendar|W{k:02d}" for k in range(N_PLANNED_WEEKS)]


def planned_month_identities(history_id: str) -> list[str]:
    """Return the label-blind planned monthly period identities M01..M06."""
    return [f"{history_id}|month|calendar|M{k:02d}" for k in range(1, N_PLANNED_MONTHS + 1)]


def planned_reference_identity(history_id: str) -> str:
    """Return the label-blind planned causal-reference identity."""
    return f"{history_id}|REFERENCE|end_le_c1"


def order_planned_identities(identities: Sequence[str]) -> list[str]:
    """Return the planned identities in canonical sorted order."""
    return sorted(str(value) for value in identities)


def planned_pair_identity(history_id: str, horizon: str, previous_key: str, current_key: str) -> str:
    """Return the planned nonadjacent research-pair identity.

    The pair key carries both ends (``PREV_TO_CUR``) because baseline and
    elevated blocks are nonadjacent by design; the M3 single-key form would
    be ambiguous here.
    """
    return f"{history_id}|{horizon}|calendar|PAIR|{previous_key}_TO_{current_key}"


def _band_weeks(week_bands: Mapping[int, str], *bands: str) -> set[int]:
    """Return the week offsets carrying any of the given bands."""
    return {week for week, band in week_bands.items() if band in bands}


def validate_arm_disjointness(baseline_weeks: set[int], elevated_weeks: set[int]) -> dict[str, object]:
    """Refuse baseline/elevated arm week-sets that share a week."""
    overlap = sorted(set(baseline_weeks) & set(elevated_weeks))
    if overlap:
        return _check("REFUSED-OVERLAP", False, f"baseline and elevated share weeks {overlap}")
    return _check(
        "REFUSED-OVERLAP",
        True,
        f"disjoint: baseline {sorted(baseline_weeks)} vs elevated {sorted(elevated_weeks)}",
    )


def validate_schedule_structure(week_bands: Mapping[int, str]) -> dict[str, object]:
    """Validate the planned weekly schedule structure (controls and contrast)."""
    control_a = _band_weeks(week_bands, "BASELINE-A")
    control_b = _band_weeks(week_bands, "BASELINE-B")
    if not control_a or not control_b or (control_a & control_b):
        return _check("REFUSED-NO-WEEKLY-CONTROL", False, "two disjoint weekly baseline segments required")
    for run in (sorted(control_a), sorted(control_b)):
        if len(run) < MIN_STABLE_RUN_PERIODS:
            return _check("REFUSED-NO-WEEKLY-CONTROL", False, "each weekly baseline segment needs >= 2 weeks")
    elevated = _band_weeks(week_bands, "ELEVATED-A", "ELEVATED-B")
    if not elevated:
        return _check("REFUSED-SHORTFALL", False, "at least one planned elevated block required")
    return _check(
        "REFUSED-NO-WEEKLY-CONTROL",
        True,
        f"baseline segments {sorted(control_a)} and {sorted(control_b)} with elevated weeks {sorted(elevated)}",
    )


def validate_disjoint_support(week_bands: Mapping[int, str]) -> dict[str, object]:
    """Refuse any planned schedule whose arms are not disjoint or lack controls."""
    baseline = _band_weeks(week_bands, "BASELINE-A", "BASELINE-B")
    elevated = _band_weeks(week_bands, "ELEVATED-A", "ELEVATED-B")
    disjoint = validate_arm_disjointness(baseline, elevated)
    if disjoint["status"] == "REFUSED":
        return disjoint
    return validate_schedule_structure(week_bands)


def validate_seeds(seed: int, used: Sequence[int]) -> dict[str, object]:
    """Refuse seeds colliding with the certified old block or each other."""
    if CERTIFIED_OLD_SEED_MIN <= seed <= CERTIFIED_OLD_SEED_MAX:
        return _check("REFUSED-SEED-COLLISION", False, f"seed {seed} lies in the certified old block")
    if seed in used:
        return _check("REFUSED-SEED-COLLISION", False, f"seed {seed} is already assigned")
    if seed < 0:
        return _check("REFUSED-SEED-COLLISION", False, f"seed {seed} is not a valid seed")
    return _check("REFUSED-SEED-COLLISION", True, f"seed {seed} is fresh and unique")


def validate_quotas(
    n_reference: int,
    n_weekly: int,
    n_common_strata: int,
    n_retained_per_arm: int,
) -> list[dict[str, object]]:
    """Check planned quotas against inherited floors and the M6-derived bar."""
    return [
        _check(
            "REFUSED-SHORTFALL",
            n_reference >= MIN_REFERENCE_FILES,
            f"planned reference {n_reference} against floor {MIN_REFERENCE_FILES}",
        ),
        _check(
            "REFUSED-SHORTFALL",
            n_weekly >= MIN_PERIOD_FILES,
            f"planned weekly files {n_weekly} against floor {MIN_PERIOD_FILES}",
        ),
        _check(
            "REFUSED-SHORTFALL",
            n_common_strata >= PLANNED_MIN_COMMON_STRATA,
            f"planned common strata {n_common_strata} against minimum {PLANNED_MIN_COMMON_STRATA}",
        ),
        _check(
            "REFUSED-SHORTFALL",
            n_retained_per_arm >= PLANNED_MIN_RETAINED_PER_ARM,
            f"planned retained per arm {n_retained_per_arm} against minimum {PLANNED_MIN_RETAINED_PER_ARM}",
        ),
    ]


def validate_lineage(history_id: str) -> dict[str, object]:
    """Refuse any planned history that collides with the immutable old corpus."""
    if history_id in OLD_HISTORY_IDS:
        return _check("REFUSED-LINEAGE", False, f"{history_id} reuses an immutable old-corpus identity")
    if history_id.startswith(OLD_CORPUS_PREFIX):
        return _check("REFUSED-LINEAGE", False, f"{history_id} shares the old-corpus prefix {OLD_CORPUS_PREFIX}")
    if history_id not in NEW_HISTORY_IDS:
        return _check("REFUSED-IDENTITY", False, f"{history_id} is not a frozen planned identity")
    return _check("REFUSED-LINEAGE", True, f"{history_id} is lineage-separate from the old corpus")


def default_planned_pairs() -> list[dict[str, str]]:
    """Return the five planned comparable pairs per history per horizon."""
    weekly = [
        ("BASELINE-A", "ELEVATED-A", "TRANSITION"),
        ("BASELINE-B", "ELEVATED-A", "TRANSITION"),
        ("BASELINE-A", "ELEVATED-B", "TRANSITION"),
        ("BASELINE-B", "ELEVATED-B", "TRANSITION"),
        ("BASELINE-A", "BASELINE-B", "CONTROL"),
    ]
    monthly = [
        ("RUN-A", "ELEVATED-M4", "TRANSITION"),
        ("RUN-B", "ELEVATED-M4", "TRANSITION"),
        ("RUN-A", "ELEVATED-M6", "TRANSITION"),
        ("RUN-B", "ELEVATED-M6", "TRANSITION"),
        ("RUN-A", "RUN-B", "CONTROL"),
    ]
    rows: list[dict[str, str]] = []
    for horizon, plan in (("week", weekly), ("month", monthly)):
        for previous, current, kind in plan:
            rows.append(
                {"horizon": horizon, "previous": previous, "current": current, "kind": kind}
            )
    return rows


def validate_monthly_bands(month_bands: Mapping[int, str]) -> dict[str, object]:
    """Refuse a planned monthly layout without two disjoint LOW runs plus contrast."""
    low_months = sorted(month for month, band in month_bands.items() if band == "LOW")
    elevated = sorted(month for month, band in month_bands.items() if band == "ELEVATED")
    runs: list[list[int]] = []
    current: list[int] = []
    for month in low_months:
        if current and month != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(month)
    if current:
        runs.append(current)
    qualifying = [run for run in runs if len(run) >= MIN_STABLE_RUN_PERIODS]
    if len(qualifying) < 2:
        return _check(
            "REFUSED-NO-MONTHLY-CONTROL",
            False,
            f"two disjoint LOW runs of >= {MIN_STABLE_RUN_PERIODS} months required; found {qualifying}",
        )
    if not elevated:
        return _check("REFUSED-SHORTFALL", False, "at least one planned ELEVATED month required")
    shared = set(low_months) & set(elevated)
    if shared:
        return _check("REFUSED-OVERLAP", False, f"monthly LOW and ELEVATED share months {sorted(shared)}")
    return _check(
        "REFUSED-NO-MONTHLY-CONTROL",
        True,
        f"monthly runs {qualifying} with elevated months {elevated}",
    )


def validate_planned_history(
    history_id: str,
    seed: int,
    used_seeds: Sequence[int],
    week_bands: Mapping[int, str],
    month_bands: Mapping[int, str],
    n_reference: int = PLANNED_REFERENCE_FILES,
    n_weekly: int = PLANNED_WEEKLY_FILES,
    n_common_strata: int = PLANNED_MIN_COMMON_STRATA,
    n_retained_per_arm: int = PLANNED_MIN_RETAINED_PER_ARM,
) -> dict[str, object]:
    """Validate one planned history; shortfalls refuse, never rescue."""
    checks: list[dict[str, object]] = [validate_lineage(history_id)]
    checks.append(validate_seeds(seed, used_seeds))
    checks.extend(validate_quotas(n_reference, n_weekly, n_common_strata, n_retained_per_arm))
    checks.append(validate_disjoint_support(week_bands))
    checks.append(validate_monthly_bands(month_bands))
    refused = [check for check in checks if check["status"] == "REFUSED"]
    return {
        "history_id": history_id,
        "verdict": "PLANNED-REFUSED" if refused else "PLANNED-SUPPORTED",
        "checks": checks,
        "refusals": [check["reason"] for check in refused],
    }


# --- forward/storage cost ------------------------------------------------------


def cost_estimate() -> dict[str, object]:
    """Return forward/storage formulas with evaluated planned numbers.

    Every number below is arithmetic on planned quotas plus stated
    assumptions; nothing here is a measurement and no runtime is estimated.
    """
    n_histories = len(NEW_HISTORY_IDS)
    span_files = (N_PLANNED_WEEKS - 1) * PLANNED_WEEKLY_FILES
    per_history = PLANNED_REFERENCE_FILES + span_files
    total_files = n_histories * per_history
    n_seeds = len(FROZEN_ENCODER_SEEDS)
    required_forwards = total_files * n_seeds
    smoke_forwards = 96
    required_with_smoke = required_forwards + smoke_forwards
    shortfall = required_with_smoke - FORWARD_HEADROOM
    wave_bytes = total_files * ASSUMED_MEAN_STEPS_PER_FILE * N_CHANNELS * 4
    embedding_bytes = total_files * EMBEDDING_DIM * 4 * n_seeds
    manifest_bytes = total_files * 2048
    return {
        "formulas": {
            "files_per_history": "N_ref + (N_weeks - 1) * N_weekly",
            "total_files": "H * files_per_history",
            "required_forwards": "total_files * n_encoder_seeds (+ 96 bounded smoke)",
            "waveform_bytes": "total_files * mean_steps * n_channels * 4",
            "embedding_bytes": "total_files * embedding_dim * 4 * n_encoder_seeds",
            "manifest_bytes": "total_files * 2048",
        },
        "assumptions": {
            "mean_steps_per_file": ASSUMED_MEAN_STEPS_PER_FILE,
            "mean_steps_basis": "stated assumption inside the RegimeConfig 200-800 step range; not measured",
            "weekly_density_basis": "planned 40 files/week carried from observed old-corpus density; verified at the pre-generation freeze, never silently rescaled",
            "business_density_note": "owner-reported 500-1000 files/day is later scale context, not this protocol's target",
        },
        "planned_counts": {
            "n_histories": n_histories,
            "files_per_history": per_history,
            "total_files": total_files,
            "n_encoder_seeds": n_seeds,
        },
        "forwards": {
            "required_forwards": required_forwards,
            "required_with_smoke": required_with_smoke,
            "cumulative_spent": CUMULATIVE_FORWARDS_SPENT,
            "cap": FORWARD_CAP,
            "headroom": FORWARD_HEADROOM,
            "shortfall_vs_headroom": shortfall,
            "cap_policy": "no reset, no reuse; inference needs an explicit new owner-authorized budget",
        },
        "storage_bytes": {
            "waveforms": wave_bytes,
            "embeddings": embedding_bytes,
            "manifests": manifest_bytes,
        },
    }


def forward_gate() -> dict[str, object]:
    """Return the inference-authorization gate derived from the cost estimate."""
    estimate = cost_estimate()
    forwards = estimate["forwards"]
    assert isinstance(forwards, dict)
    shortfall = int(forwards["shortfall_vs_headroom"])
    authorized = shortfall <= 0
    return {
        "gate": "INFERENCE-AUTHORIZED" if authorized else "INFERENCE-NOT-AUTHORIZED",
        "reason": (
            "planned inference fits the remaining headroom"
            if authorized
            else f"planned {forwards['required_with_smoke']} forwards exceed headroom "
            f"{forwards['headroom']} by {shortfall}; explicit new budget required"
        ),
        "cost_digest": sha256_object(estimate),
    }


__all__ = [
    "ASSUMED_MEAN_STEPS_PER_FILE",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CERTIFIED_OLD_SEED_MAX",
    "CERTIFIED_OLD_SEED_MIN",
    "CUMULATIVE_FORWARDS_SPENT",
    "DEFAULT_MONTH_BANDS",
    "DEFAULT_WEEK_BANDS",
    "ELEVATED_FRACTION_PRIMARY",
    "ELEVATED_OPERATING_BAND",
    "EMBEDDING_DIM",
    "EXPOSURE_RATIO_FLOOR",
    "FORWARD_CAP",
    "FORWARD_HEADROOM",
    "FROZEN_ENCODER_SEEDS",
    "HORIZONS",
    "LOW_BASELINE_OPERATING_BAND",
    "LOW_BURDEN_FRACTION_PRIMARY",
    "MIX_DRIFT_REFERENCE",
    "MIN_PERIOD_FILES",
    "MIN_REFERENCE_FILES",
    "MIN_STABLE_RUN_PERIODS",
    "MODES",
    "NEW_HISTORY_IDS",
    "NULL_SEED",
    "N_CHANNELS",
    "N_PLANNED_MONTHS",
    "N_PLANNED_WEEKS",
    "OLD_CORPUS_PREFIX",
    "OLD_HISTORY_IDS",
    "PAIR_KINDS",
    "PLAN_VERDICTS",
    "PLANNED_MIN_COMMON_STRATA",
    "PLANNED_MIN_RETAINED_GAP",
    "PLANNED_MIN_RETAINED_PER_ARM",
    "PLANNED_REFERENCE_FILES",
    "PLANNED_WEEKLY_FILES",
    "PRIMARY_MODE",
    "PRIMARY_TRIPLE",
    "PROPOSED_DATA_SEEDS",
    "PROPOSED_GENERATOR_VERSION",
    "PROTOCOL_ID",
    "REFUSAL_REASONS",
    "REPLICATION_TRIPLE",
    "STABILITY_RANGE_PRIMARY",
    "WEEK_BANDS",
    "MONTH_BANDS",
    "validate_arm_disjointness",
    "calendar_assumptions",
    "cost_estimate",
    "default_planned_pairs",
    "forward_gate",
    "order_planned_identities",
    "planned_month_identities",
    "planned_pair_identity",
    "planned_reference_identity",
    "planned_week_identities",
    "protocol_freeze",
    "protocol_sha256",
    "protocol_spec",
    "validate_disjoint_support",
    "validate_lineage",
    "validate_monthly_bands",
    "validate_planned_history",
    "validate_quotas",
    "validate_seeds",
]
