"""Weekly/monthly prospective study amendment — metadata-only manifests and decision rule (S24-7-M3).

This module freezes the *executable study contract* for the owner-prioritized
week-over-week and month-over-month collective-movement research. It resolves the
``S24-BIZ-HORIZON-v1`` proposal into:

- a digest-bound study identity (:func:`study_spec` / :func:`study_sha256`),
  including the retained/defaulted/deferred resolution of every proposal
  decision D1--D10 plus the two implicit decisions and the new M3 decisions;
- **label-blind** period, pair and reference manifests with deterministic
  identities (:func:`build_label_blind_manifest`), frozen before any outcome;
- a separate post-hoc burden projection (:func:`build_posthoc_burden`) that is
  joined by identity only after membership exists;
- the declared horizon x mode x endpoint x representation-seed x history cell
  matrix with an explicit status for every cell and for every declared contrast
  (:func:`coverage_matrix`), including ``UNSUPPORTED`` states;
- the prospectively justified PROMISING / NOT-PROMISING / INCONCLUSIVE decision
  rule (:func:`evaluate_decision`) and its worked illustrative examples.

Binding boundaries:

- **No movement is computed here.** Nothing in this module reads an embedding, a
  score, a cache, a checkpoint or a waveform, opens a model, or performs a
  forward. The decision rule is a pure function of declared outcome summaries; it
  is exercised only with illustrative inputs explicitly labelled as such.
- **Movement is not anomaly, and absence of annotation is not health.** The
  study's only available burden contrast is a post-hoc association. No
  disposition may claim calibrated specificity, detector recall/delay/false
  alarms, or per-file classification: the accepted corpus has zero
  annotated-clean and zero quiet-period pairs at both horizons (M2 evidence).
- **Floors are inherited unchanged** (30 eligible files per period, 200
  reference files). Abstention is a declared outcome, never a reason to relax a
  floor.
- **Daily is deferred, not removed.** The owner deferred the daily horizon; it is
  absent from this study's declared horizon set and no daily claim follows.
- **Scaling is not authorized.** A PROMISING disposition only proposes a
  separately authorized scaling study; it releases no corpus generation, no
  scaled inference, no retraining and no Phase C execution.

The calendar projection (anchor epoch, timezone, week start, month anchor day)
is a declared *research default*: the accepted Development clock is relative and
carries no wall-clock epoch, so no deployment timezone or epoch is asserted.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
import textwrap
from collections.abc import Mapping, Sequence
from statistics import median as _statistics_median

from representation import sprint24_business_horizon as horizon
from representation.sprint24_context_audit import sha256_object

#: Study family identifier frozen by this module.
STUDY_ID = "S24-BIZ-WM-v1"

#: Versioned protocol document this module implements.
PROTOCOL_DOC = "docs/sprint24-weekly-monthly-protocol-v1.md"

#: Upstream proposal that this study resolves.
UPSTREAM_PROPOSAL_ID = horizon.PROPOSAL_ID

#: Study horizons. ``day`` is deferred by owner prioritization and is deliberately
#: absent: no daily comparison is declared, supported or claimed.
HORIZONS = ("week", "month")

#: Primary boundary convention. ``rolling`` is a declared sensitivity convention.
PRIMARY_MODE = "calendar"
SENSITIVITY_MODES = ("rolling",)
MODES = (PRIMARY_MODE, *SENSITIVITY_MODES)

#: Projection anchor used by the accepted M2 feasibility pass (2024-01-01Z, a
#: Monday). A declared research default, never a corpus or deployment fact.
PRIMARY_ANCHOR_EPOCH_S = 1_704_067_200.0

#: Alternate anchor used only for the declared projection sensitivity.
ALTERNATE_ANCHOR_EPOCH_S = 1_709_769_600.0

#: Accepted encoder seeds (identity-selected, not outcome-ranked). The three seeds
#: are metadata-identical: they add runs, not independent data.
REPRESENTATION_SEEDS = (181801, 181802, 181803)

#: File-embedding width of the accepted frozen representation (no projection).
EMBEDDING_DIM = 128

#: Endpoint hierarchy. Only ``E1`` feeds the decision rule; ``E2`` is a declared
#: secondary collective endpoint and ``E3``/``E4`` are descriptive components.
ENDPOINT_HIERARCHY: tuple[dict[str, object], ...] = (
    {
        "id": "E1",
        "name": "DIST-MOVE",
        "role": "PRIMARY",
        "decision_relevant": True,
        "note": "unbiased squared MMD; kernel bandwidth from the causal reference pool only",
    },
    {
        "id": "E2",
        "name": "CLUSTER-MOVE",
        "role": "SECONDARY",
        "decision_relevant": False,
        "note": "total-variation distance of cluster occupancy; prototypes fitted on the reference pool",
    },
    {
        "id": "E3",
        "name": "LOC-MOVE",
        "role": "DESCRIPTIVE-ONLY",
        "decision_relevant": False,
        "note": "location component; never a decision input",
    },
    {
        "id": "E4",
        "name": "DISP-MOVE",
        "role": "DESCRIPTIVE-ONLY",
        "decision_relevant": False,
        "note": "dispersion component; never a decision input",
    },
)

#: Declared contrast availability vocabulary.
CONTRAST_STATES = (
    "AVAILABLE",
    "DESCRIPTIVE-ONLY",
    "UNSUPPORTED-CONTROL-ABSENT",
)

#: Declared cell status vocabulary. Every declared matrix cell carries one value.
CELL_STATES = (
    "ESTIMABLE",
    "NOT-ESTIMABLE-INSUFFICIENT-PAIRS",
    "DESCRIPTIVE-SUPPORTED",
    "SENSITIVITY-SUPPORTED",
    "UNSUPPORTED-NO-COMPARABLE-PAIRS",
    "UNSUPPORTED-INSUFFICIENT-REFERENCE",
)

#: Decision dispositions. Exactly one is returned per horizon analysis.
DISPOSITIONS = ("PROMISING", "NOT-PROMISING", "INCONCLUSIVE")

#: Claims that no disposition authorizes (declared once, quoted by every result).
BLOCKED_CLAIMS = (
    "calibrated anomaly specificity",
    "detector recall, detection delay or false-alarm rate",
    "per-file classification accuracy or PR-AUC",
    "out-of-sample or fresh-generalization claim",
    "Confirmation reuse as validation",
    "Phase C release or sequential-study execution",
    "scaled inference, corpus generation or retraining",
)

#: Inherited floors, re-declared so the study cannot silently change them.
MIN_PERIOD_FILES = horizon.MIN_PERIOD_FILES
MIN_REFERENCE_FILES = horizon.MIN_REFERENCE_FILES

#: Declared maximum embedding rows per representation seed: the accepted custody
#: envelope (3,678 causal reference rows + 5,076 analysis-span rows).
DECLARED_MAX_ROWS_PER_SEED = 8_754

#: Declared maximum embedding rows across the three accepted seeds.
DECLARED_MAX_ROWS_TOTAL = DECLARED_MAX_ROWS_PER_SEED * len(REPRESENTATION_SEEDS)

#: Inherited geometry-time forward accounting (G2 record: cumulative 26,431 of a
#: strict 41,000 cap, no reset). Quoted so the study cannot silently assume a
#: fresh budget.
INHERITED_CUMULATIVE_FORWARDS = 26_431
INHERITED_FORWARD_CAP = 41_000
INHERITED_HEADROOM_FORWARDS = INHERITED_FORWARD_CAP - INHERITED_CUMULATIVE_FORWARDS

# --- decision-rule constants (prospectively declared research defaults) -------

#: Minimum comparable matched pairs in a history before a within-history rank
#: association is treated as estimable. Three is the smallest count at which a
#: rank association has a defined, non-degenerate permutation distribution
#: (perfect association: two-sided 0.333, 0.083 at four, 0.0167 at five). The floor
#: is deliberately not raised to the count that would allow a significant
#: per-history test: the accepted design supplies at most 6 comparable matched
#: month pairs per history, so a higher floor would predetermine the month
#: disposition regardless of the evidence -- an impossible gate rather than a rule.
DECISION_MIN_PAIRS_PER_HISTORY = 3

#: Minimum number of estimable histories (of the four declared) for any
#: disposition other than INCONCLUSIVE.
DECISION_MIN_ESTIMABLE_HISTORIES = 3

#: Promise-level median within-history rank association between movement and
#: post-hoc burden (C1).
DECISION_PROMISE_BURDEN_RHO = 0.5

#: Promise-level median within-history rank association between the mix-adjusted
#: movement residual and post-hoc burden (C2). A third of the rank signal must
#: survive composition adjustment, because G3 measured r(MMD2, TV drift) ~ +0.78.
DECISION_PROMISE_RESIDUAL_RHO = 0.3

#: Minimum number of estimable histories with a positive burden association.
DECISION_CONSISTENCY_MIN_HISTORIES = 3

#: A sensitivity variant must keep the primary sign and at least this fraction of
#: the primary magnitude for the disposition to remain decisive.
DECISION_SENSITIVITY_RETENTION = 0.5

#: Family-wise alpha for any reported inferential statement (Holm over the
#: declared 4 primary contrasts: 2 horizons x 2 association endpoints).
DECISION_ALPHA = 0.05
DECISION_FAMILY_SIZE = len(HORIZONS) * 2

#: Declared robustness variants that the disposition must survive.
DECISION_SENSITIVITY_VARIANTS = ("rolling", "capped_n", "alternate_anchor")

#: History-cluster bootstrap replicates and seed for every descriptive interval
#: (inherited convention from the frozen prospective protocol's uncertainty rule).
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = horizon.PROPOSED_NULL_SEED

#: Label-blind rule for the capped-n exposure-mismatch sensitivity: both periods
#: are reduced to the smaller member count by a deterministic hash of the file ID
#: under the inherited null seed. Frozen here so it cannot track outcomes.
CAPPED_N_RULE = {
    "order_key": "sha256(f'{null_seed}:{history_id}:{period_identity}:{file_id}')",
    "selection": "ascending order key, keep the first min(n_previous, n_current) members per period",
    "null_seed": horizon.PROPOSED_NULL_SEED,
}

#: Frozen label-blind structural fields of the period manifest.
PERIOD_STRUCTURAL_FIELDS: tuple[str, ...] = (
    "history_id",
    "horizon",
    "mode",
    "period_key",
    "period_start_rel_s",
    "period_end_rel_s",
    "period_length_days",
    "n_members",
    "n_straddling_members",
    "exposure_days",
    "coverage_state",
    "support_state",
    "meets_support_floor",
    "n_reset_edges",
    "member_ids_sha256",
)

#: Frozen label-blind structural fields of the pair manifest.
PAIR_STRUCTURAL_FIELDS: tuple[str, ...] = (
    "history_id",
    "horizon",
    "mode",
    "previous_key",
    "current_key",
    "previous_start_rel_s",
    "current_end_rel_s",
    "n_previous",
    "n_current",
    "exposure_previous_days",
    "exposure_current_days",
    "exposure_ratio",
    "mix_drift_tv",
    "n_reset_edges_crossed",
    "coverage_state",
    "support_state",
    "exposure_state",
    "comparison_state",
)

#: Frozen post-hoc burden fields of a period (never part of an identity digest).
POSTHOC_PERIOD_FIELDS: tuple[str, ...] = (
    "n_abnormal_members",
    "n_unknown_label_members",
    "abnormal_member_fraction",
    "n_failure_events_covered",
    "n_degradation_onsets_covered",
    "annotation_state",
)

#: Frozen post-hoc burden fields of a pair (never part of an identity digest).
POSTHOC_PAIR_FIELDS: tuple[str, ...] = (
    "previous_annotation_state",
    "current_annotation_state",
    "n_abnormal_members",
    "abnormal_member_fraction",
    "n_failure_events_covered",
    "n_degradation_onsets_covered",
    "attribution_class",
)

#: Calibrated-claim boundary restated in machine-readable form.
CLAIM_BOUNDARIES: tuple[str, ...] = (
    "movement is not an anomaly: a distribution shift is not evidence of a fault",
    "absence of annotation is not health: missing labels yield UNANNOTATED, never a control",
    "no annotated-clean or quiet-period pair exists at week or month granularity on the accepted span",
    "sampling-null crossing is necessary, not sufficient (G2 crossed 369/369)",
    "the calendar projection is a declared research assumption, not a corpus or deployment fact",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(_statistics_median(values))


# --- declared calendars ------------------------------------------------------


def primary_calendar() -> horizon.CalendarConfig:
    """Return the declared primary projection (UTC, ISO Monday, 15th month anchor)."""
    return horizon.CalendarConfig(
        anchor_epoch_s=PRIMARY_ANCHOR_EPOCH_S,
        timezone=horizon.PROPOSED_TIMEZONE,
        week_start=horizon.PROPOSED_WEEK_START,
        month_anchor_day=horizon.PROPOSED_MONTH_ANCHOR_DAY,
    )


def sensitivity_calendars() -> dict[str, horizon.CalendarConfig]:
    """Return the declared projection variants, primary first.

    The first mapping entry is the study's primary projection; every later entry is
    a declared assumption probe used only to measure sensitivity.
    """
    return {
        "primary": primary_calendar(),
        "alternate_anchor_week_zone": horizon.CalendarConfig(
            anchor_epoch_s=ALTERNATE_ANCHOR_EPOCH_S,
            timezone="Etc/GMT-7",
            week_start="sunday",
            month_anchor_day=horizon.PROPOSED_MONTH_ANCHOR_DAY,
        ),
    }


# --- study identity ----------------------------------------------------------


def decision_policy() -> dict[str, object]:
    """Return the frozen decision-rule policy, its justification and its limits."""
    return {
        "dispositions": list(DISPOSITIONS),
        "min_pairs_per_history": DECISION_MIN_PAIRS_PER_HISTORY,
        "min_pairs_justification": (
            "3 is the smallest pair count with a defined, non-degenerate rank "
            "permutation distribution (perfect association two-sided 0.333, 0.083 at "
            "4 pairs, 0.0167 at 5); a floor tied to per-history significance is "
            "deliberately rejected because the accepted month design supplies at most "
            "5 comparable matched pairs per history, so any higher floor would "
            "predetermine the month disposition"
        ),
        "min_estimable_histories": DECISION_MIN_ESTIMABLE_HISTORIES,
        "min_distinct_burden_values": 2,
        "promise_burden_rho": DECISION_PROMISE_BURDEN_RHO,
        "promise_residual_rho": DECISION_PROMISE_RESIDUAL_RHO,
        "promise_magnitude_justification": (
            "a burden rank association below 0.5 explains under a quarter of the rank "
            "variance; the 0.3 residual floor requires a third of the signal to survive "
            "composition adjustment because G3 measured r(MMD2, TV drift) ~ +0.78"
        ),
        "consistency_min_histories": DECISION_CONSISTENCY_MIN_HISTORIES,
        "sensitivity_variants": list(DECISION_SENSITIVITY_VARIANTS),
        "sensitivity_retention": DECISION_SENSITIVITY_RETENTION,
        "alpha": DECISION_ALPHA,
        "alpha_status": (
            "declared reporting convention for any auxiliary inferential statement; "
            "this is a research default, not an owner-approved scientific gate, and no "
            "disposition depends on it"
        ),
        "multiplicity": {
            "family": "2 horizons x 2 association endpoints (C1 burden, C2 residual)",
            "family_size": DECISION_FAMILY_SIZE,
            "adjustment": "Holm step-down over the declared family",
            "exact_permutation_reference": (
                "with rank ties absent, the two-sided permutation tail of a perfect rank "
                "association is 2/n!: 0.333 at 3 pairs, 0.0833 at 4, 0.0167 at 5 and "
                "0.00278 at 6"
            ),
            "consequence": (
                "at the month maximum of 5 comparable matched pairs per history even a "
                "perfect rank association gives 0.0167, above the Holm threshold of "
                "0.0125, so no month per-history significance claim is possible; week "
                "histories with 6 or more pairs can reach it. Any reported per-history "
                "inference is Holm-adjusted and explicitly auxiliary: the disposition is "
                "an effect-magnitude, consistency and sensitivity triage, and its "
                "PROMISING outcome proposes a separately authorized scaling study with "
                "more histories or density, never a significance or specificity claim"
            ),
        },
        "rule_kind": (
            "prospective triage rule for whether to propose a separately authorized "
            "scaling study; not a performance gate, not a detector operating point"
        ),
        "revision_policy": (
            "any revision is allowed only before the outcome freeze; a post-outcome "
            "revision invalidates the study rather than re-deciding it"
        ),
        "specificity_claim_allowed": False,
        "specificity_reason": (
            "no calibrated specificity is claimable: zero annotated-clean and zero "
            "quiet-period pairs exist at week or month granularity on the accepted "
            "Development span"
        ),
    }


def cost_policy() -> dict[str, object]:
    """Return the frozen cost/cap policy and its inherited-accounting consequence."""
    return {
        "mode": "metadata-only until a separately authorized execution task",
        "declared_max_rows_per_seed": DECLARED_MAX_ROWS_PER_SEED,
        "declared_max_rows_total": DECLARED_MAX_ROWS_TOTAL,
        "seed_multiplicity": len(REPRESENTATION_SEEDS),
        "seed_independent_data": False,
        "reuse_required": True,
        "reuse_basis": (
            "the accepted G2 run captured the same 8,754 files per seed (3,678 causal "
            "reference + 5,076 analysis span) under the same three accepted seeds; reuse "
            "is permitted only when every cache-identity input matches exactly"
        ),
        "inherited_cumulative_forwards": INHERITED_CUMULATIVE_FORWARDS,
        "inherited_forward_cap": INHERITED_FORWARD_CAP,
        "inherited_headroom_forwards": INHERITED_HEADROOM_FORWARDS,
        "inherited_headroom_consequence": (
            "a fresh span-only capture (5,076 x 3 = 15,228 rows) exceeds the inherited "
            "headroom of 14,569 by 659 rows, so zero-new-forward cache reuse is a release "
            "prerequisite unless the owner explicitly approves incremental headroom"
        ),
        "fail_closed": (
            "the CLI refuses any plan whose per-seed row requirement exceeds the declared "
            "envelope, and reports the actual requirement"
        ),
        "representation_coverage": "UNKNOWN until the execution preflight verifies cache identity",
    }


def decision_resolutions() -> tuple[dict[str, object], ...]:
    """Return every resolved upstream/implicit/new decision with its tier.

    Tiers: ``RETAINED`` (inherited rule kept unchanged), ``RESEARCH-DEFAULT`` (a
    declared research assumption for this study, never an owner-approved business
    fact), ``DEFERRED`` (postponed, no claim follows), ``OWNER-REQUIRED`` (cannot be
    resolved by any research default).
    """
    return (
        {
            "id": "D1",
            "topic": "Business timezone (IANA)",
            "tier": "RESEARCH-DEFAULT",
            "value": horizon.PROPOSED_TIMEZONE,
            "owner_confirmation_required": True,
            "consequence": "a deployment zone that differs shifts day/week membership and pair counts",
        },
        {
            "id": "D2",
            "topic": "Projection anchor mapping the relative clock to a calendar",
            "tier": "RESEARCH-DEFAULT",
            "value": "2024-01-01T00:00:00Z (arithmetic projection)",
            "owner_confirmation_required": True,
            "consequence": "month and week phase are arbitrary; the alternate-anchor sensitivity bounds the effect",
        },
        {
            "id": "D3",
            "topic": "Week start",
            "tier": "RESEARCH-DEFAULT",
            "value": horizon.PROPOSED_WEEK_START,
            "owner_confirmation_required": True,
            "consequence": "a Sunday start moves week membership; measured in the projection sensitivity",
        },
        {
            "id": "D4",
            "topic": "Primary boundary convention",
            "tier": "RETAINED",
            "value": "calendar-aligned primary, rolling declared as sensitivity",
            "owner_confirmation_required": False,
            "consequence": "a rolling primary would change the compared pairs and is not declared",
        },
        {
            "id": "D5",
            "topic": "Partial-period policy",
            "tier": "RETAINED",
            "value": "exclude partial periods from primary comparisons",
            "owner_confirmation_required": False,
            "consequence": "including them would confound movement with exposure",
        },
        {
            "id": "D6",
            "topic": "Daily-horizon support floor",
            "tier": "DEFERRED",
            "value": "floor unchanged at 30; the daily horizon is deferred by owner prioritization",
            "owner_confirmation_required": True,
            "consequence": "no daily comparison, claim or floor relaxation is part of this study",
        },
        {
            "id": "D7",
            "topic": "Normal-variation reference",
            "tier": "RETAINED",
            "value": "none available; event-load association only, no specificity claim",
            "owner_confirmation_required": False,
            "consequence": "no abnormal-versus-quiet contrast and no calibrated operating point",
        },
        {
            "id": "D8",
            "topic": "Operating-point policy",
            "tier": "RETAINED",
            "value": "descriptive only; no calibrated detector or threshold",
            "owner_confirmation_required": False,
            "consequence": "recall, delay and false-alarm claims stay unavailable",
        },
        {
            "id": "D9",
            "topic": "Cluster count for the composition endpoint",
            "tier": "RESEARCH-DEFAULT",
            "value": f"K = {horizon.PROPOSED_CLUSTER_COUNT} primary, 4/16 sensitivity",
            "owner_confirmation_required": False,
            "consequence": "the composition endpoint depends on K, which is fixed before outcomes",
        },
        {
            "id": "D10",
            "topic": "Exposure-mismatch policy",
            "tier": "RESEARCH-DEFAULT",
            "value": f"ratio floor {horizon.PROPOSED_EXPOSURE_RATIO_FLOOR}; excluded from primary, capped-n sensitivity",
            "owner_confirmation_required": False,
            "consequence": "exposure-mismatched pairs stay published but out of the primary aggregate",
        },
        {
            "id": "I1",
            "topic": "Business unit: fleet or permitted sub-context",
            "tier": "RETAINED",
            "value": "fleet only; per-robot and robot x program strata are unsupported",
            "owner_confirmation_required": False,
            "consequence": "composition adjustment is limited to fleet-level observable mix drift",
        },
        {
            "id": "I2",
            "topic": "Deployment source with real timestamps and labels as the healthy control",
            "tier": "DEFERRED",
            "value": "not used; requires a separate data decision and release",
            "owner_confirmation_required": True,
            "consequence": "specificity remains unverifiable in this study",
        },
        {
            "id": "M3-1",
            "topic": "Study horizon set",
            "tier": "RETAINED",
            "value": "week and month primary; daily deferred without removal from the business requirement",
            "owner_confirmation_required": False,
            "consequence": "no daily claim; the inferred scale question is not read into current density",
        },
        {
            "id": "M3-2",
            "topic": "Decision rule for 'promising enough to propose scaling'",
            "tier": "RESEARCH-DEFAULT",
            "value": "frozen magnitude/consistency/sensitivity triage over E1 with the C2 confound residual",
            "owner_confirmation_required": False,
            "consequence": "the rule cannot be revised after outcomes; revision invalidates the study",
        },
        {
            "id": "M3-3",
            "topic": "Endpoint hierarchy and estimator",
            "tier": "RETAINED",
            "value": "E1 primary, E2 secondary, E3/E4 descriptive; reference-pool-only fitting",
            "owner_confirmation_required": False,
            "consequence": "no endpoint may be promoted or re-fitted per period or per pair",
        },
        {
            "id": "M3-4",
            "topic": "Multiplicity family",
            "tier": "RESEARCH-DEFAULT",
            "value": f"Holm over {DECISION_FAMILY_SIZE} primary contrasts at alpha {DECISION_ALPHA}",
            "owner_confirmation_required": False,
            "consequence": "reported significance is family-adjusted; the disposition stays a triage",
        },
        {
            "id": "M3-5",
            "topic": "Uncertainty",
            "tier": "RETAINED",
            "value": f"history-cluster bootstrap B = {BOOTSTRAP_REPLICATES}, seed {BOOTSTRAP_SEED}",
            "owner_confirmation_required": False,
            "consequence": "n = 4 histories makes every interval descriptive",
        },
        {
            "id": "M3-6",
            "topic": "Inference cost and cap",
            "tier": "OWNER-REQUIRED",
            "value": "zero-new-forward cache reuse required; any incremental row needs explicit approval",
            "owner_confirmation_required": True,
            "consequence": "no forward is authorized by this study contract",
        },
        {
            "id": "M3-7",
            "topic": "Scaling release",
            "tier": "OWNER-REQUIRED",
            "value": "separate release; a PROMISING disposition only proposes it",
            "owner_confirmation_required": True,
            "consequence": "no corpus generation, scaled inference, retraining or Phase C execution follows",
        },
    )


def study_spec() -> dict[str, object]:
    """Return the frozen study identity: horizons, matrix, identities, rules, budget."""
    for item in ENDPOINT_HIERARCHY:
        if item["id"] not in {entry["id"] for entry in horizon.ENDPOINTS}:
            raise RuntimeError(f"unknown endpoint id: {item['id']!r}")
    return {
        "study_id": STUDY_ID,
        "protocol_doc": PROTOCOL_DOC,
        "upstream_proposal_id": UPSTREAM_PROPOSAL_ID,
        "horizons": list(HORIZONS),
        "deferred_horizons": ["day"],
        "modes": list(MODES),
        "primary_mode": PRIMARY_MODE,
        "sensitivity_modes": list(SENSITIVITY_MODES),
        "calendars": {
            name: calendar.describe() for name, calendar in sensitivity_calendars().items()
        },
        "representation": {
            "seeds": list(REPRESENTATION_SEEDS),
            "seeds_are_independent_data": False,
            "embedding_dim": EMBEDDING_DIM,
            "space": "original frozen file-embedding space; no projection",
        },
        "endpoints": [dict(item) for item in ENDPOINT_HIERARCHY],
        "estimands": {
            "C1": "within-history rank association of E1/E2 with post-hoc burden",
            "C2": "within-history rank association of the mix-drift-adjusted E1 residual with post-hoc burden",
            "burden": "covered failure events and abnormal member fraction, joined post-hoc only",
        },
        "identity": {
            "period_identity": "{history_id}|{horizon}|{mode}|{period_key}",
            "pair_identity": "{history_id}|{horizon}|{mode}|PAIR|{current_key}",
            "reference_identity": "{history_id}|REFERENCE|end_le_c1",
            "period_members_sha256": "sha256 over the sorted member file IDs",
            "pair_members_sha256": "sha256 over the concatenated previous and current period digests",
            "identity_digest_excludes": "every post-hoc annotation field",
        },
        "membership_rule": {
            "assignment": "file end_time belongs to the period with start < end_time <= end",
            "reference_pool": "eligible files with end_time <= c1, fixed before any evaluation period",
            "partial_periods": "excluded from primary comparisons",
            "floors": {"min_period_files": MIN_PERIOD_FILES, "min_reference_files": MIN_REFERENCE_FILES},
        },
        "uncertainty": {
            "cluster_unit": "history; pairs inside a history are dependent because adjacent pairs share a period",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "status": "descriptive at n = 4 histories; never a precise standard error",
            "seed_replication": "the three representation seeds are run replicates, not independent data",
        },
        "confound_handling": {
            "primary": "C2 mix-adjusted residual association is co-required for a PROMISING disposition",
            "covariate": "pair-level total-variation distance between robot-mix vectors",
            "stratification": "label-blind median split of the covariate within history x horizon",
            "sub_contexts": "unsupported; fleet-level covariates only",
        },
        "decision": decision_policy(),
        "cost": cost_policy(),
        "capped_n_rule": dict(CAPPED_N_RULE),
        "decisions": [dict(item) for item in decision_resolutions()],
        "claim_boundaries": list(CLAIM_BOUNDARIES),
        "blocked_claims": list(BLOCKED_CLAIMS),
        "cell_states": list(CELL_STATES),
        "contrast_states": list(CONTRAST_STATES),
        "upstream_freeze": {
            "proposal_id": horizon.PROPOSAL_ID,
            "proposal_sha256": horizon.proposal_sha256(),
            "inherited_unmodified": [
                "docs/sprint24-geometry-time-diagnostic-v1.md",
                "docs/sprint24-geometry-time-decision-g3.md",
                "docs/sprint24-business-horizon-measurement-proposal.md",
            ],
        },
    }


def study_sha256() -> str:
    """Return the freeze digest of the study identity."""
    return sha256_object(study_spec())


def study_freeze() -> dict[str, object]:
    """Return the study spec together with its freeze digest."""
    spec = study_spec()
    return {"study_sha256": sha256_object(spec), "spec": spec}


# --- source construction -----------------------------------------------------


def evaluate_source(
    manifests: Mapping[str, Mapping[str, object]],
    edges: Mapping[str, Sequence[float]],
    calendar: horizon.CalendarConfig,
    modes: Sequence[str] = MODES,
) -> dict[str, object]:
    """Construct only the study's declared horizons through the inherited machinery.

    This is the accepted M2 construction path restricted to ``HORIZONS``: the
    deferred daily horizon is never enumerated, so no daily row, count or claim can
    leak into the study. Construction for every history x horizon x mode happens
    before any annotation is projected, and annotations are joined only afterwards.

    Returns the same structural tables the upstream pass exposes
    (``period_rows``/``pair_rows``/``references``/``history_totals``) so that the
    label-blind manifest builder and the post-hoc join can be reused unchanged.
    """
    horizon.assert_period_construction_label_blind()
    results: list[horizon.HistoryHorizonResult] = []
    metas: dict[str, object] = {}
    for history_id in sorted(manifests):
        if history_id not in edges:
            raise KeyError(f"missing frozen decile edges for {history_id}")
        edge_values = [float(value) for value in edges[history_id]]
        if len(edge_values) < 5:
            raise ValueError(f"decile edge table too short for {history_id}")
        observable = horizon.project_observable(manifests[history_id])
        meta = horizon.prepare_history(history_id, observable, edge_values[4])
        metas[history_id] = meta
        for mode in modes:
            for horizon_name in HORIZONS:
                results.append(horizon.build_history_horizon(meta, horizon_name, mode, calendar))
    for result in results:
        horizon.annotate_history_horizon(result, manifests[result.history_id], calendar)
    history_totals = {}
    for history_id, meta in sorted(metas.items()):
        history_totals[history_id] = {
            "n_rows": len(meta.file_ids),
            "n_eligible": int(meta.eligible.sum()),
            "n_eligible_analysis_span": int(
                (meta.eligible & (meta.ends > meta.c1) & (meta.ends <= meta.tmax)).sum()
            ),
            "c1_rel_s": meta.c1,
            "tmax_rel_s": meta.tmax,
        }
    return {
        "proposal_id": horizon.PROPOSAL_ID,
        "proposal_sha256": horizon.proposal_sha256(),
        "calendar": calendar.describe(),
        "modes": list(modes),
        "horizons": list(HORIZONS),
        "period_rows": [row for result in results for row in result.period_rows],
        "pair_rows": [row for result in results for row in result.pair_rows],
        "references": {
            history_id: horizon.reference_pool(meta) for history_id, meta in sorted(metas.items())
        },
        "history_totals": history_totals,
        "feasibility_rows": [result.feasibility for result in results],
    }


def projection_sensitivity(
    manifests: Mapping[str, Mapping[str, object]],
    edges: Mapping[str, Sequence[float]],
    variants: Mapping[str, horizon.CalendarConfig],
    modes: Sequence[str] = MODES,
) -> dict[str, object]:
    """Compare declared calendar projections over the study's horizons only.

    The first mapping entry is the reference projection for the published deltas;
    every later entry is a declared assumption probe. Only the study's declared
    horizons are enumerated, so a deferred horizon never enters a study artifact.
    """
    if not variants:
        raise ValueError("at least one calendar variant is required")
    counters = (
        "n_periods",
        "n_full_periods",
        "n_supported_periods",
        "n_pairs",
        "n_pairs_comparable_matched",
        "n_pairs_comparable_unmatched",
    )
    runs: dict[str, object] = {}
    for label, calendar in variants.items():
        result = evaluate_source(manifests, edges, calendar, modes=modes)
        cells: dict[str, dict[str, int]] = {}
        for row in result["feasibility_rows"]:  # type: ignore[union-attr]
            key = f"{row['horizon']}|{row['mode']}"
            entry = cells.setdefault(key, {name: 0 for name in counters})
            for name in counters:
                entry[name] += int(row[name])
        runs[label] = {"calendar": calendar.describe(), "matrix": cells}
    baseline_label = next(iter(variants))
    baseline_matrix = runs[baseline_label]["matrix"]  # type: ignore[index]
    deltas: dict[str, object] = {}
    for label, run in runs.items():
        if label == baseline_label:
            continue
        matrix = run["matrix"]  # type: ignore[index]
        deltas[label] = {
            cell: {
                name: int(matrix[cell][name]) - int(baseline_matrix[cell][name])
                for name in counters
            }
            for cell in baseline_matrix
        }
    return {
        "study_id": STUDY_ID,
        "study_sha256": study_sha256(),
        "variants": list(variants),
        "baseline": baseline_label,
        "note": (
            "projection probes only; no timezone, week start or anchor is a verified "
            "business fact, and a variant can never be used to rescue a failed primary cell"
        ),
        "counters": list(counters),
        "runs": runs,
        "deltas_vs_baseline": deltas,
    }


# --- label-blind construction ------------------------------------------------


def period_identity(history_id: str, horizon_name: str, mode: str, period_key: str) -> str:
    """Return the deterministic label-blind period identity."""
    return f"{history_id}|{horizon_name}|{mode}|{period_key}"


def pair_identity(history_id: str, horizon_name: str, mode: str, current_key: str) -> str:
    """Return the deterministic label-blind pair identity.

    The explicit ``PAIR`` segment keeps the pair namespace disjoint from the period
    namespace, so a period identity can never collide with the pair it anchors.
    """
    return f"{history_id}|{horizon_name}|{mode}|PAIR|{current_key}"


def reference_identity(history_id: str) -> str:
    """Return the deterministic causal-reference identity."""
    return f"{history_id}|REFERENCE|end_le_c1"


def _period_row(row: Mapping[str, object]) -> dict[str, object]:
    missing = [field for field in PERIOD_STRUCTURAL_FIELDS if field not in row]
    if missing:
        raise RuntimeError(f"upstream period row lost declared fields: {missing}")
    out = {field: row[field] for field in PERIOD_STRUCTURAL_FIELDS}
    out["period_identity"] = period_identity(
        str(row["history_id"]), str(row["horizon"]), str(row["mode"]), str(row["period_key"])
    )
    out["row_sha256"] = sha256_object(out)
    return out


def _pair_row(
    row: Mapping[str, object],
    previous_member_digest: Mapping[str, str],
    current_member_digest: Mapping[str, str],
) -> dict[str, object]:
    missing = [field for field in PAIR_STRUCTURAL_FIELDS if field not in row]
    if missing:
        raise RuntimeError(f"upstream pair row lost declared fields: {missing}")
    out = {field: row[field] for field in PAIR_STRUCTURAL_FIELDS}
    identity = pair_identity(
        str(row["history_id"]), str(row["horizon"]), str(row["mode"]), str(row["current_key"])
    )
    previous_key = period_identity(
        str(row["history_id"]), str(row["horizon"]), str(row["mode"]), str(row["previous_key"])
    )
    current_key = period_identity(
        str(row["history_id"]), str(row["horizon"]), str(row["mode"]), str(row["current_key"])
    )
    out["pair_identity"] = identity
    out["previous_period_identity"] = previous_key
    out["current_period_identity"] = current_key
    out["pair_members_sha256"] = _sha256_text(
        f"{previous_member_digest[previous_key]}\n{current_member_digest[current_key]}"
    )
    out["row_sha256"] = sha256_object(out)
    return out


def _reference_row(history_id: str, reference: Mapping[str, object]) -> dict[str, object]:
    return {
        "reference_identity": reference_identity(history_id),
        "history_id": history_id,
        "reference_files": int(reference["reference_files"]),  # type: ignore[arg-type]
        "reference_digest": str(reference["reference_digest"]),
        "reference_end_max_rel_s": reference["reference_end_max"],
        "robot_mix": list(reference["robot_mix"]),  # type: ignore[arg-type]
        "reference_state": str(reference["reference_state"]),
    }


def build_label_blind_manifest(
    result: Mapping[str, object],
    calendar: horizon.CalendarConfig,
) -> dict[str, object]:
    """Build the frozen label-blind study manifest from constructed period/pair tables.

    The manifest is derived exclusively from structural fields: membership,
    exposure, support, comparability and the observable mix/reset covariates. No
    annotation field enters any row or digest. :func:`assert_study_construction_label_blind`
    guards the construction functions, and the manifest ``validity`` block re-asserts
    causal custody and period contiguity, failing closed on violation.
    """
    period_rows_upstream = result["period_rows"]
    pair_rows_upstream = result["pair_rows"]
    references_upstream = result["references"]
    history_totals = result["history_totals"]
    if not isinstance(period_rows_upstream, list) or not isinstance(pair_rows_upstream, list):
        raise RuntimeError("upstream result is missing its period/pair tables")
    if not isinstance(references_upstream, Mapping) or not isinstance(history_totals, Mapping):
        raise RuntimeError("upstream result is missing its reference/history tables")
    periods = [_period_row(row) for row in period_rows_upstream]
    member_digest = {
        str(row["period_identity"]): str(row["member_ids_sha256"]) for row in periods
    }
    pairs = [_pair_row(row, member_digest, member_digest) for row in pair_rows_upstream]
    references = [
        _reference_row(str(history_id), reference)
        for history_id, reference in sorted(references_upstream.items())
    ]
    histories = {
        str(history_id): {
            "c1_rel_s": float(totals["c1_rel_s"]),  # type: ignore[index]
            "tmax_rel_s": float(totals["tmax_rel_s"]),  # type: ignore[index]
            "n_rows": int(totals["n_rows"]),  # type: ignore[index]
            "n_eligible": int(totals["n_eligible"]),  # type: ignore[index]
            "n_eligible_analysis_span": int(totals["n_eligible_analysis_span"]),  # type: ignore[index]
        }
        for history_id, totals in sorted(history_totals.items())
    }
    manifest: dict[str, object] = {
        "study_id": STUDY_ID,
        "study_sha256": study_sha256(),
        "calendar": calendar.describe(),
        "histories": histories,
        "references": references,
        "periods": periods,
        "pairs": pairs,
    }
    manifest["validity"] = _assert_manifest_validity(manifest)
    manifest["digest"] = {
        "references_sha256": sha256_object(references),
        "periods_sha256": sha256_object(periods),
        "pairs_sha256": sha256_object(pairs),
    }
    return manifest


def _assert_manifest_validity(manifest: Mapping[str, object]) -> dict[str, object]:
    """Verify causal custody, contiguity and identity integrity; fail closed."""
    histories = manifest["histories"]
    references = manifest["references"]
    periods = manifest["periods"]
    pairs = manifest["pairs"]
    if not isinstance(histories, Mapping) or not isinstance(references, list):
        raise RuntimeError("manifest history/reference tables are malformed")
    if not isinstance(periods, list) or not isinstance(pairs, list):
        raise RuntimeError("manifest period/pair tables are malformed")
    checks = {"reference_causal": True, "periods_contiguous": True, "identities_unique": True}
    for reference in references:
        history_id = str(reference["history_id"])
        c1 = float(histories[history_id]["c1_rel_s"])  # type: ignore[index]
        end_max = reference["reference_end_max_rel_s"]
        if end_max is None or float(end_max) > c1 + 1e-9:
            checks["reference_causal"] = False
    for row in periods:
        if str(row["coverage_state"]) != "FULL":
            continue
        history_id = str(row["history_id"])
        c1 = float(histories[history_id]["c1_rel_s"])  # type: ignore[index]
        tmax = float(histories[history_id]["tmax_rel_s"])  # type: ignore[index]
        start = float(row["period_start_rel_s"])
        end = float(row["period_end_rel_s"])
        if start < c1 - 1e-9 or end > tmax + 1e-9:
            checks["periods_contiguous"] = False
    period_index = {str(row["period_identity"]): row for row in periods}
    for row in pairs:
        previous = period_index[str(row["previous_period_identity"])]
        current = period_index[str(row["current_period_identity"])]
        # Adjacent periods must touch exactly, and the pair must report that same
        # span. A gap or overlap would make the comparison non-adjacent.
        if float(previous["period_end_rel_s"]) != float(current["period_start_rel_s"]):
            checks["periods_contiguous"] = False
        if float(row["previous_start_rel_s"]) != float(previous["period_start_rel_s"]):
            checks["periods_contiguous"] = False
        if float(row["current_end_rel_s"]) != float(current["period_end_rel_s"]):
            checks["periods_contiguous"] = False
    identities = [str(row["period_identity"]) for row in periods] + [
        str(row["pair_identity"]) for row in pairs
    ] + [str(row["reference_identity"]) for row in references]
    if len(set(identities)) != len(identities):
        checks["identities_unique"] = False
    if not all(checks.values()):
        raise RuntimeError(f"study manifest validity failed: {checks}")
    return checks


#: Names that must never appear in label-blind construction code.
_FORBIDDEN_CONSTRUCTION_TOKENS = (
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
)


def assert_study_construction_label_blind() -> None:
    """Fail closed if label-blind manifest construction references annotations."""
    checked = (
        _period_row,
        _pair_row,
        _reference_row,
        period_identity,
        pair_identity,
        reference_identity,
        build_label_blind_manifest,
        _assert_manifest_validity,
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
        for token in _FORBIDDEN_CONSTRUCTION_TOKENS:
            if token in code:
                raise RuntimeError(f"label-blind violation: {fn.__name__} references {token!r}")


def build_posthoc_burden(result: Mapping[str, object]) -> dict[str, object]:
    """Project post-hoc burden rows keyed by the frozen label-blind identities.

    Grain: one row per period identity and per pair identity. Absent coverage stays
    explicit as ``UNANNOTATED``; a zero count is never read as health.
    """
    period_rows = result["period_rows"]
    pair_rows = result["pair_rows"]
    if not isinstance(period_rows, list) or not isinstance(pair_rows, list):
        raise RuntimeError("upstream result is missing its period/pair tables")
    periods = []
    for row in period_rows:
        identity = period_identity(
            str(row["history_id"]), str(row["horizon"]), str(row["mode"]), str(row["period_key"])
        )
        entry: dict[str, object] = {"period_identity": identity}
        entry.update({field: row[field] for field in POSTHOC_PERIOD_FIELDS})
        periods.append(entry)
    pairs = []
    for row in pair_rows:
        identity = pair_identity(
            str(row["history_id"]), str(row["horizon"]), str(row["mode"]), str(row["current_key"])
        )
        entry = {"pair_identity": identity, "comparison_state": str(row["comparison_state"])}
        entry.update({field: row[field] for field in POSTHOC_PAIR_FIELDS})
        pairs.append(entry)
    return {
        "study_id": STUDY_ID,
        "study_sha256": study_sha256(),
        "grain": "post-hoc annotation join by frozen identity; never a membership input",
        "periods": periods,
        "pairs": pairs,
        "digest": {
            "periods_sha256": sha256_object(periods),
            "pairs_sha256": sha256_object(pairs),
        },
    }


# --- coverage and estimability ----------------------------------------------


def horizon_counts(
    manifest: Mapping[str, object],
    posthoc: Mapping[str, object],
    horizon_name: str,
    mode: str,
) -> dict[str, dict[str, object]]:
    """Return per-history comparable-pair and burden-variation counts.

    Comparable pairs are the published ``COMPARABLE-MATCHED`` (primary) and
    ``COMPARABLE-UNMATCHED-EXPOSURE`` (sensitivity) states. Distinct burden values
    are read from the post-hoc block only.
    """
    pairs = manifest["pairs"]
    burden = posthoc["pairs"]
    if not isinstance(pairs, list) or not isinstance(burden, list):
        raise RuntimeError("manifest/post-hoc pair tables are malformed")
    burden_by_pair = {str(row["pair_identity"]): row for row in burden}
    histories = sorted(str(key) for key in manifest["histories"])  # type: ignore[union-attr]
    out: dict[str, dict[str, object]] = {
        history_id: {
            "history_id": history_id,
            "matched_pairs": 0,
            "unmatched_pairs": 0,
            "comparable_pairs": 0,
            "matched_member_rows": 0,
            "distinct_matched_burden_values": 0,
            "matched_pairs_with_failure_events": 0,
            "median_matched_mix_drift_tv": None,
            "median_matched_exposure_ratio": None,
            "states": {},
        }
        for history_id in histories
    }
    matched_burden: dict[str, list[int]] = {history_id: [] for history_id in histories}
    matched_drift: dict[str, list[float]] = {history_id: [] for history_id in histories}
    matched_ratio: dict[str, list[float]] = {history_id: [] for history_id in histories}
    for row in pairs:
        if str(row["horizon"]) != horizon_name or str(row["mode"]) != mode:
            continue
        history_id = str(row["history_id"])
        state = str(row["comparison_state"])
        entry = out[history_id]
        states = entry["states"]
        assert isinstance(states, dict)
        states[state] = int(states.get(state, 0)) + 1
        if state not in ("COMPARABLE-MATCHED", "COMPARABLE-UNMATCHED-EXPOSURE"):
            continue
        entry["comparable_pairs"] = int(entry["comparable_pairs"]) + 1
        entry["matched_member_rows"] = int(entry["matched_member_rows"]) + int(
            row["n_previous"]
        ) + int(row["n_current"])
        if state == "COMPARABLE-UNMATCHED-EXPOSURE":
            entry["unmatched_pairs"] = int(entry["unmatched_pairs"]) + 1
            continue
        entry["matched_pairs"] = int(entry["matched_pairs"]) + 1
        post = burden_by_pair[str(row["pair_identity"])]
        events = int(post["n_failure_events_covered"])
        matched_burden[history_id].append(events)
        if events > 0:
            entry["matched_pairs_with_failure_events"] = (
                int(entry["matched_pairs_with_failure_events"]) + 1
            )
        if row["mix_drift_tv"] is not None:
            matched_drift[history_id].append(float(row["mix_drift_tv"]))
        if row["exposure_ratio"] is not None:
            matched_ratio[history_id].append(float(row["exposure_ratio"]))
    for history_id in histories:
        entry = out[history_id]
        entry["distinct_matched_burden_values"] = len(set(matched_burden[history_id]))
        entry["median_matched_mix_drift_tv"] = _median(matched_drift[history_id])
        entry["median_matched_exposure_ratio"] = _median(matched_ratio[history_id])
    return out


def estimability(
    counts: Mapping[str, Mapping[str, object]],
    min_pairs: int = DECISION_MIN_PAIRS_PER_HISTORY,
) -> dict[str, object]:
    """Classify each history as estimable for the within-history burden relation."""
    per_history = {}
    estimable_ids = []
    for history_id, entry in sorted(counts.items()):
        pairs = int(entry["matched_pairs"])
        distinct = int(entry["distinct_matched_burden_values"])
        ok = pairs >= min_pairs and distinct >= 2
        per_history[history_id] = {
            "matched_pairs": pairs,
            "distinct_burden_values": distinct,
            "estimable": ok,
            "reason": (
                "ok"
                if ok
                else (
                    f"fewer than {min_pairs} comparable matched pairs"
                    if pairs < min_pairs
                    else "burden has no rank variation"
                )
            ),
        }
        if ok:
            estimable_ids.append(history_id)
    return {
        "min_pairs": min_pairs,
        "min_distinct_burden_values": 2,
        "estimable_histories": estimable_ids,
        "n_estimable_histories": len(estimable_ids),
        "n_histories": len(per_history),
        "per_history": per_history,
    }


def expected_disposition(estimable_block: Mapping[str, object]) -> dict[str, object]:
    """Return the pre-outcome metadata estimability verdict for one horizon.

    This is a structural pre-check, not a result: when too few histories are
    estimable the disposition is already determined to be ``INCONCLUSIVE``
    regardless of the eventual association values.
    """
    n_estimable = int(estimable_block["n_estimable_histories"])
    if n_estimable < DECISION_MIN_ESTIMABLE_HISTORIES:
        return {
            "status": "PREDETERMINED-INCONCLUSIVE",
            "disposition": "INCONCLUSIVE",
            "reason": (
                f"{n_estimable} of {estimable_block['n_histories']} histories reach the "
                f"predeclared {DECISION_MIN_PAIRS_PER_HISTORY}-pair minimum; "
                f"{DECISION_MIN_ESTIMABLE_HISTORIES} are required for a decisive disposition"
            ),
            "kind": "metadata estimability pre-check, not a scientific result",
        }
    return {
        "status": "UNDETERMINED-PENDING-OUTCOMES",
        "disposition": None,
        "reason": (
            f"{n_estimable} of {estimable_block['n_histories']} histories reach the "
            f"predeclared {DECISION_MIN_PAIRS_PER_HISTORY}-pair minimum"
        ),
        "kind": "metadata estimability pre-check, not a scientific result",
    }


def contrast_availability(
    posthoc: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Return the declared contrast catalogue with its metadata-derived state."""
    pairs = posthoc["pairs"]
    if not isinstance(pairs, list):
        raise RuntimeError("post-hoc pair table is malformed")
    # Controls are defined on the published comparable pairs only: an unsupported
    # or partial pair can never supply or deny a control.
    comparable = [
        row
        for row in pairs
        if str(row["pair_identity"]).split("|")[1] in HORIZONS
        and str(row["comparison_state"])
        in ("COMPARABLE-MATCHED", "COMPARABLE-UNMATCHED-EXPOSURE")
    ]
    clean = sum(
        1
        for row in comparable
        if str(row["previous_annotation_state"]) == "ANNOTATED-CLEAN"
        and str(row["current_annotation_state"]) == "ANNOTATED-CLEAN"
    )
    quiet = sum(1 for row in comparable if str(row["attribution_class"]) == "NO-EVENT-ANNOTATED")
    unspecified = sum(1 for row in comparable if str(row["attribution_class"]) == "UNANNOTATED")
    return [
        {
            "id": "C1",
            "name": "EVENT-LOAD-ASSOCIATION",
            "state": "AVAILABLE",
            "note": "primary burden association used by the decision rule",
        },
        {
            "id": "C2",
            "name": "MIX-ADJUSTED-RESIDUAL",
            "state": "AVAILABLE",
            "note": "co-required confound residual for a PROMISING disposition",
        },
        {
            "id": "C3",
            "name": "QUIET-PERIOD-CONTRAST",
            "state": "UNSUPPORTED-CONTROL-ABSENT" if quiet == 0 else "AVAILABLE",
            "note": (
                "zero quiet-period pairs at week/month granularity; abstention is the declared outcome"
                if quiet == 0
                else f"{quiet} quiet-period pairs present; re-review required before execution"
            ),
        },
        {
            "id": "C4",
            "name": "TEMPORAL-SPECIFICITY",
            "state": "UNSUPPORTED-CONTROL-ABSENT",
            "note": "no event-free neighbouring period exists at either horizon on the accepted span",
        },
        {
            "id": "N1",
            "name": "SAMPLING-NULL",
            "state": "DESCRIPTIVE-ONLY",
            "note": "G2 crossed its own sampling q95 in 369/369 windows, so crossing is necessary but never sufficient",
        },
        {
            "id": "N2",
            "name": "HALF-SPLIT-BAND",
            "state": "DESCRIPTIVE-ONLY",
            "note": "conservative within-period noise band; reported, never a decision criterion",
        },
        {
            "id": "N3",
            "name": "ANNOTATED-CLEAN-CONTROL",
            "state": "UNSUPPORTED-CONTROL-ABSENT" if clean == 0 else "AVAILABLE",
            "note": (
                "zero annotated-clean comparable pairs at either horizon"
                if clean == 0
                else f"{clean} annotated-clean pairs present; re-review required before execution"
            ),
        },
        {
            "id": "N3b",
            "name": "QUIET-PERIOD-CONTROL",
            "state": "UNSUPPORTED-CONTROL-ABSENT" if quiet == 0 else "AVAILABLE",
            "note": "identical availability to C3 by construction",
        },
    ], {
        "n_compared_pairs": len(comparable),
        "n_state_annotated_clean_pairs": clean,
        "n_state_quiet_pairs": quiet,
        "n_state_unannotated_pairs": unspecified,
    }


def coverage_matrix(
    manifest: Mapping[str, object],
    posthoc: Mapping[str, object],
) -> dict[str, object]:
    """Return every declared study cell with its status, support and cost.

    The declared cell space is ``horizon x mode x endpoint x representation seed x
    history``. Every cell appears exactly once; unsupported and non-estimable cells
    are published rather than dropped.
    """
    histories = sorted(str(key) for key in manifest["histories"])  # type: ignore[union-attr]
    reference_state = {
        str(row["history_id"]): str(row["reference_state"]) for row in manifest["references"]  # type: ignore[union-attr]
    }
    cells: list[dict[str, object]] = []
    horizon_blocks: dict[str, dict[str, object]] = {}
    for horizon_name in HORIZONS:
        block: dict[str, object] = {}
        for mode in MODES:
            counts = horizon_counts(manifest, posthoc, horizon_name, mode)
            estimable_block = estimability(counts)
            block[mode] = {
                "counts": counts,
                "estimability": estimable_block,
                "expected_disposition": expected_disposition(estimable_block),
            }
            for endpoint in ENDPOINT_HIERARCHY:
                for seed in REPRESENTATION_SEEDS:
                    for history_id in histories:
                        entry = counts[history_id]
                        reference_ok = reference_state[history_id] == "REFERENCE-OK"
                        if not reference_ok:
                            status = "UNSUPPORTED-INSUFFICIENT-REFERENCE"
                        elif int(entry["comparable_pairs"]) == 0:
                            status = "UNSUPPORTED-NO-COMPARABLE-PAIRS"
                        elif mode != PRIMARY_MODE:
                            # A sensitivity cell is computed and reported, but it can
                            # never be promoted to a decision-bearing cell.
                            status = "SENSITIVITY-SUPPORTED"
                        elif endpoint["decision_relevant"]:
                            ok = estimable_block["per_history"][history_id]["estimable"]  # type: ignore[index]
                            status = "ESTIMABLE" if ok else "NOT-ESTIMABLE-INSUFFICIENT-PAIRS"
                        else:
                            status = "DESCRIPTIVE-SUPPORTED"
                        cells.append(
                            {
                                "horizon": horizon_name,
                                "mode": mode,
                                "mode_role": "PRIMARY" if mode == PRIMARY_MODE else "SENSITIVITY",
                                "endpoint_id": endpoint["id"],
                                "endpoint_role": endpoint["role"],
                                "decision_relevant": bool(endpoint["decision_relevant"]),
                                "representation_seed": seed,
                                "history_id": history_id,
                                "n_comparable_pairs": int(entry["comparable_pairs"]),
                                "n_matched_pairs": int(entry["matched_pairs"]),
                                "distinct_burden_values": int(entry["distinct_matched_burden_values"]),
                                "median_mix_drift_tv": entry["median_matched_mix_drift_tv"],
                                "member_rows_per_seed": int(entry["matched_member_rows"]),
                                "status": status,
                            }
                        )
        horizon_blocks[horizon_name] = block
    contrasts, contrast_counts = contrast_availability(posthoc)
    status_counts: dict[str, int] = {}
    for cell in cells:
        status_counts[str(cell["status"])] = int(status_counts.get(str(cell["status"]), 0)) + 1
    return {
        "study_id": STUDY_ID,
        "study_sha256": study_sha256(),
        "declared_cell_count": len(HORIZONS)
        * len(MODES)
        * len(ENDPOINT_HIERARCHY)
        * len(REPRESENTATION_SEEDS)
        * len(histories),
        "n_cells": len(cells),
        "status_counts": status_counts,
        "cells": cells,
        "horizons": horizon_blocks,
        "contrasts": contrasts,
        "contrast_counts": contrast_counts,
        "notes": [
            "member_rows_per_seed counts member occurrences inside a history's compared periods; files recur across adjacent periods, so these counts must not be summed into a total",
            "the study total is 3,678 causal reference rows plus 5,076 analysis-span rows per seed",
            "no cell implies a specificity, detector or generalization claim",
        ],
    }


# --- decision rule -----------------------------------------------------------


def _criterion(criterion_id: str, status: str, detail: str) -> dict[str, object]:
    return {"id": criterion_id, "status": status, "detail": detail}


def min_attainable_tail(n_pairs: int) -> float | None:
    """Return the smallest two-sided rank-permutation tail attainable at ``n_pairs``.

    With no ties the tail of a perfect rank association is ``2 / n!``. The value is
    published beside every disposition so a coarse month-level association can never
    be read as if it carried per-history inferential resolution.
    """
    if n_pairs < 3:
        return None
    return 2.0 / math.factorial(n_pairs)


def resolution_block(histories: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return the disclosed rank resolution of each contributing history."""
    per_history: dict[str, dict[str, object]] = {}
    for item in histories:
        n_pairs = int(item["n_pairs"])
        per_history[str(item["history_id"])] = {
            "n_pairs": n_pairs,
            "min_attainable_two_sided_rank_tail": min_attainable_tail(n_pairs),
        }
    coarse = any(
        float(entry["min_attainable_two_sided_rank_tail"] or 0.0) > DECISION_ALPHA
        for entry in per_history.values()
    )
    return {
        "per_history": per_history,
        "coarse": coarse,
        "note": (
            "a coarse horizon has at least one history whose smallest attainable rank "
            "tail exceeds the declared alpha, so no per-history significance claim is "
            "possible there; the disposition remains a triage and a PROMISING outcome "
            "proposes more histories or density"
            if coarse
            else "every contributing history has more rank resolution than the declared alpha requires"
        ),
    }


def _sign_of(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def evaluate_decision(
    analysis: Mapping[str, object],
    policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the PROMISING / NOT-PROMISING / INCONCLUSIVE disposition for one horizon.

    Pure function of declared outcome summaries. It computes nothing empirical: the
    caller supplies per-history rank associations (``rho_burden`` for C1,
    ``rho_residual`` for C2), their pair counts and burden variation, plus the
    declared sensitivity variants. A disposition is decisive only when the same
    evidence burden is met in both directions: a negative answer is *not* declared
    from underpowered data, and a positive answer is *not* declared when a declared
    sensitivity variant is unknown or sign-flipped.
    """
    rules = policy or decision_policy()
    horizon_name = str(analysis.get("horizon", "?"))
    raw = analysis.get("histories")
    if not isinstance(raw, list) or not raw:
        raise ValueError("analysis must declare non-empty per-history observations")
    histories = [dict(item) for item in raw]
    for item in histories:
        item.setdefault("n_distinct_burden", 0)
    min_pairs = int(rules["min_pairs_per_history"])  # type: ignore[arg-type]
    min_estimable = int(rules["min_estimable_histories"])  # type: ignore[arg-type]
    estimable = [
        item
        for item in histories
        if int(item["n_pairs"]) >= min_pairs and int(item["n_distinct_burden"]) >= 2
    ]
    criteria: list[dict[str, object]] = []
    criteria.append(
        _criterion(
            "A.estimability",
            "PASS" if len(estimable) >= min_estimable else "FAIL",
            f"{len(estimable)} of {len(histories)} histories reach {min_pairs} pairs with "
            f">= 2 burden values (minimum {min_estimable})",
        )
    )
    unknown = [
        item
        for item in estimable
        if item.get("rho_burden") is None or item.get("rho_residual") is None
    ]
    burden = [float(item["rho_burden"]) for item in estimable if item.get("rho_burden") is not None]
    residual = [
        float(item["rho_residual"]) for item in estimable if item.get("rho_residual") is not None
    ]
    median_burden = _median(burden) if not unknown else None
    median_residual = _median(residual) if not unknown else None
    if unknown:
        criteria.append(
            _criterion(
                "B.promise-magnitude",
                "UNKNOWN",
                f"{len(unknown)} estimable histories have no association estimate",
            )
        )
        criteria.append(
            _criterion("C.promise-consistency", "UNKNOWN", "missing association estimates")
        )
        criteria.append(
            _criterion("N1.nonpromise-magnitude", "UNKNOWN", "missing association estimates")
        )
        criteria.append(
            _criterion("N2.nonpromise-consistency", "UNKNOWN", "missing association estimates")
        )
    else:
        promise_magnitude = (
            median_burden is not None
            and median_burden >= float(rules["promise_burden_rho"])  # type: ignore[arg-type]
            and median_residual is not None
            and median_residual >= float(rules["promise_residual_rho"])  # type: ignore[arg-type]
        )
        positive = sum(1 for value in burden if value > 0)
        promise_consistency = positive >= int(rules["consistency_min_histories"])  # type: ignore[arg-type]
        criteria.append(
            _criterion(
                "B.promise-magnitude",
                "PASS" if promise_magnitude else "FAIL",
                f"median burden rho {median_burden} "
                f"(floor {rules['promise_burden_rho']}), median residual rho {median_residual} "
                f"(floor {rules['promise_residual_rho']})",
            )
        )
        criteria.append(
            _criterion(
                "C.promise-consistency",
                "PASS" if promise_consistency else "FAIL",
                f"{positive} of {len(burden)} estimable histories have a positive burden "
                f"association (minimum {rules['consistency_min_histories']})",
            )
        )
        negative = sum(1 for value in burden if value <= 0)
        nonpromise_magnitude = (
            median_burden is not None
            and median_burden <= 0.0
            and median_residual is not None
            and median_residual <= 0.0
        )
        nonpromise_consistency = negative >= int(rules["consistency_min_histories"])  # type: ignore[arg-type]
        criteria.append(
            _criterion(
                "N1.nonpromise-magnitude",
                "PASS" if nonpromise_magnitude else "FAIL",
                f"median burden rho {median_burden} <= 0 and median residual rho "
                f"{median_residual} <= 0 required",
            )
        )
        criteria.append(
            _criterion(
                "N2.nonpromise-consistency",
                "PASS" if nonpromise_consistency else "FAIL",
                f"{negative} of {len(burden)} estimable histories have a non-positive burden "
                f"association (minimum {rules['consistency_min_histories']})",
            )
        )
    sensitivity = analysis.get("sensitivity")
    variants = [str(name) for name in rules["sensitivity_variants"]]  # type: ignore[arg-type]
    variant_values: dict[str, float | None] = {}
    if not isinstance(sensitivity, Mapping):
        variant_values = {name: None for name in variants}
    else:
        for name in variants:
            entry = sensitivity.get(name)
            value = entry.get("rho_burden") if isinstance(entry, Mapping) else None
            variant_values[name] = None if value is None else float(value)  # type: ignore[arg-type]
    missing_variants = [name for name, value in variant_values.items() if value is None]
    retention = float(rules["sensitivity_retention"])  # type: ignore[arg-type]
    if missing_variants or median_burden is None:
        criteria.append(
            _criterion(
                "D.promise-sensitivity",
                "UNKNOWN",
                f"unverifiable variants: {sorted(set(missing_variants))}"
                if missing_variants
                else "no primary median to compare against",
            )
        )
        criteria.append(
            _criterion(
                "N3.nonpromise-sensitivity",
                "UNKNOWN",
                "sensitivity variants are not verifiable",
            )
        )
    else:
        primary_sign = _sign_of(median_burden)
        promise_sensitivity = primary_sign > 0 and all(
            _sign_of(value) == primary_sign and abs(value) >= retention * abs(median_burden)
            for value in variant_values.values()
            if value is not None
        )
        nonpromise_sensitivity = median_burden <= 0.0 and all(
            value <= 0.0 for value in variant_values.values() if value is not None
        )
        criteria.append(
            _criterion(
                "D.promise-sensitivity",
                "PASS" if promise_sensitivity else "FAIL",
                "each variant keeps the primary sign and at least "
                f"{retention} of |{median_burden}|; observed {variant_values}",
            )
        )
        criteria.append(
            _criterion(
                "N3.nonpromise-sensitivity",
                "PASS" if nonpromise_sensitivity else "FAIL",
                f"every variant must be non-positive; observed {variant_values}",
            )
        )
    statuses = {str(item["id"]): str(item["status"]) for item in criteria}
    promise_ok = all(
        statuses[key] == "PASS"
        for key in ("A.estimability", "B.promise-magnitude", "C.promise-consistency", "D.promise-sensitivity")
    )
    nonpromise_ok = all(
        statuses[key] == "PASS"
        for key in ("A.estimability", "N1.nonpromise-magnitude", "N2.nonpromise-consistency", "N3.nonpromise-sensitivity")
    )
    if statuses["A.estimability"] != "PASS":
        disposition = "INCONCLUSIVE"
        rationale = (
            "insufficient per-history resolution: the predeclared pair minimum is not met in "
            "enough histories, so neither a promise nor a non-promise may be declared"
        )
    elif promise_ok:
        disposition = "PROMISING"
        rationale = (
            "all predeclared promise criteria hold: a burden association above the magnitude "
            "floors survives composition adjustment, is consistent across most estimable "
            "histories, and is retained under every declared sensitivity variant; this proposes "
            "a separately authorized scaling study and claims no specificity"
        )
    elif nonpromise_ok:
        disposition = "NOT-PROMISING"
        rationale = (
            "all predeclared non-promise criteria hold: no burden association above zero, "
            "consistent across most estimable histories, and stable under every declared "
            "sensitivity variant; the association-level question is answered negatively"
        )
    else:
        failing = sorted(key for key, status in statuses.items() if status != "PASS")
        disposition = "INCONCLUSIVE"
        rationale = (
            "criteria are in conflict, so no disposition is declared; non-passing criteria: "
            + ", ".join(failing)
        )
    return {
        "horizon": horizon_name,
        "mode": analysis.get("mode"),
        "disposition": disposition,
        "rationale": rationale,
        "criteria": criteria,
        "criterion_status": statuses,
        "estimable_histories": [str(item["history_id"]) for item in estimable],
        "median_rho_burden": median_burden,
        "median_rho_residual": median_residual,
        "sensitivity_values": variant_values,
        "resolution": resolution_block(estimable),
        "multiplicity": dict(rules["multiplicity"]),  # type: ignore[arg-type]
        "specificity_claim": False,
        "specificity_reason": str(rules["specificity_reason"]),
        "blocked_claims": list(BLOCKED_CLAIMS),
        "scaling_status": (
            "PROPOSAL-ONLY-SEPARATE-AUTHORIZATION-REQUIRED"
            if disposition == "PROMISING"
            else "NOT-RELEASED"
        ),
        "policy": {
            "min_pairs_per_history": min_pairs,
            "min_estimable_histories": min_estimable,
            "promise_burden_rho": rules["promise_burden_rho"],
            "promise_residual_rho": rules["promise_residual_rho"],
            "sensitivity_variants": variants,
            "sensitivity_retention": retention,
        },
    }


def worked_decision_examples() -> dict[str, object]:
    """Return worked decision-rule examples with **illustrative** inputs only.

    The numbers below are invented to exercise every boundary of the rule. They are
    not measurements, not simulated data, and not evidence about the corpus: no
    movement statistic has been computed for this study.
    """
    base_sensitivity = {
        "rolling": {"rho_burden": 0.48},
        "capped_n": {"rho_burden": 0.39},
        "alternate_anchor": {"rho_burden": 0.52},
    }

    def case(case_id: str, horizon_name: str, histories: list[dict[str, object]],
             sensitivity: object, expected: str, purpose: str) -> dict[str, object]:
        result = evaluate_decision(
            {
                "horizon": horizon_name,
                "mode": "calendar",
                "histories": histories,
                "sensitivity": sensitivity,
            }
        )
        if result["disposition"] != expected:
            raise RuntimeError(
                f"worked example {case_id} expected {expected}, got {result['disposition']}"
            )
        return {
            "case_id": case_id,
            "purpose": purpose,
            "inputs_are_illustrative": True,
            "expected_disposition": expected,
            "inputs": {
                "horizon": horizon_name,
                "histories": histories,
                "sensitivity": sensitivity,
            },
            "result": result,
        }

    cases = [
        case(
            "promise-boundary",
            "week",
            [
                {"history_id": "ILLUSTRATIVE-1", "n_pairs": 12, "n_distinct_burden": 7, "rho_burden": 0.71, "rho_residual": 0.44},
                {"history_id": "ILLUSTRATIVE-2", "n_pairs": 8, "n_distinct_burden": 5, "rho_burden": 0.55, "rho_residual": 0.36},
                {"history_id": "ILLUSTRATIVE-3", "n_pairs": 6, "n_distinct_burden": 4, "rho_burden": 0.62, "rho_residual": 0.41},
                {"history_id": "ILLUSTRATIVE-4", "n_pairs": 5, "n_distinct_burden": 3, "rho_burden": 0.48, "rho_residual": 0.30},
            ],
            base_sensitivity,
            "PROMISING",
            "every promise criterion holds, including the composition residual and all three sensitivity variants",
        ),
        case(
            "nonpromise-boundary",
            "week",
            [
                {"history_id": "ILLUSTRATIVE-1", "n_pairs": 10, "n_distinct_burden": 6, "rho_burden": -0.22, "rho_residual": -0.18},
                {"history_id": "ILLUSTRATIVE-2", "n_pairs": 7, "n_distinct_burden": 4, "rho_burden": -0.05, "rho_residual": -0.02},
                {"history_id": "ILLUSTRATIVE-3", "n_pairs": 6, "n_distinct_burden": 3, "rho_burden": 0.10, "rho_residual": 0.05},
                {"history_id": "ILLUSTRATIVE-4", "n_pairs": 5, "n_distinct_burden": 3, "rho_burden": -0.31, "rho_residual": -0.24},
            ],
            {
                "rolling": {"rho_burden": -0.20},
                "capped_n": {"rho_burden": -0.12},
                "alternate_anchor": {"rho_burden": -0.25},
            },
            "NOT-PROMISING",
            "a negative answer is declared only with the same evidence burden as a promise: estimable pairs, median at or below zero, three of four histories non-positive, all variants non-positive",
        ),
        case(
            "inconclusive-insufficient-pairs",
            "month",
            [
                {"history_id": "ILLUSTRATIVE-1", "n_pairs": 2, "n_distinct_burden": 2, "rho_burden": -0.90, "rho_residual": -0.80},
                {"history_id": "ILLUSTRATIVE-2", "n_pairs": 5, "n_distinct_burden": 3, "rho_burden": -0.40, "rho_residual": -0.30},
                {"history_id": "ILLUSTRATIVE-3", "n_pairs": 2, "n_distinct_burden": 2, "rho_burden": -1.00, "rho_residual": -1.00},
                {"history_id": "ILLUSTRATIVE-4", "n_pairs": 2, "n_distinct_burden": 2, "rho_burden": -0.60, "rho_residual": -0.50},
            ],
            {
                "rolling": {"rho_burden": -0.70},
                "capped_n": {"rho_burden": -0.65},
                "alternate_anchor": {"rho_burden": -0.80},
            },
            "INCONCLUSIVE",
            "strongly negative point estimates still cannot yield NOT-PROMISING when only one history reaches the three-pair estimability minimum; underpowered evidence abstains",
        ),
        case(
            "promise-coarse-month-resolution",
            "month",
            [
                {"history_id": "ILLUSTRATIVE-1", "n_pairs": 4, "n_distinct_burden": 3, "rho_burden": 0.80, "rho_residual": 0.60},
                {"history_id": "ILLUSTRATIVE-2", "n_pairs": 5, "n_distinct_burden": 3, "rho_burden": 0.70, "rho_residual": 0.55},
                {"history_id": "ILLUSTRATIVE-3", "n_pairs": 4, "n_distinct_burden": 2, "rho_burden": 1.00, "rho_residual": 0.90},
                {"history_id": "ILLUSTRATIVE-4", "n_pairs": 5, "n_distinct_burden": 3, "rho_burden": 0.90, "rho_residual": 0.70},
            ],
            {
                "rolling": {"rho_burden": 0.75},
                "capped_n": {"rho_burden": 0.68},
                "alternate_anchor": {"rho_burden": 0.82},
            },
            "PROMISING",
            "a coarse month-level association can still be PROMISING, but the resolution block marks it coarse (smallest attainable two-sided rank tail 0.083 at four pairs) so it proposes more histories or density and never a per-history significance or specificity claim",
        ),
        case(
            "inconclusive-composition-confound",
            "week",
            [
                {"history_id": "ILLUSTRATIVE-1", "n_pairs": 12, "n_distinct_burden": 7, "rho_burden": 0.66, "rho_residual": 0.12},
                {"history_id": "ILLUSTRATIVE-2", "n_pairs": 8, "n_distinct_burden": 5, "rho_burden": 0.58, "rho_residual": 0.05},
                {"history_id": "ILLUSTRATIVE-3", "n_pairs": 6, "n_distinct_burden": 4, "rho_burden": 0.61, "rho_residual": 0.19},
                {"history_id": "ILLUSTRATIVE-4", "n_pairs": 5, "n_distinct_burden": 3, "rho_burden": 0.49, "rho_residual": 0.08},
            ],
            base_sensitivity,
            "INCONCLUSIVE",
            "the raw burden association clears its floor but the mix-adjusted residual does not, which is exactly the G3 drift confound; no promise is declared",
        ),
        case(
            "inconclusive-sensitivity-flip",
            "week",
            [
                {"history_id": "ILLUSTRATIVE-1", "n_pairs": 12, "n_distinct_burden": 7, "rho_burden": 0.70, "rho_residual": 0.42},
                {"history_id": "ILLUSTRATIVE-2", "n_pairs": 8, "n_distinct_burden": 5, "rho_burden": 0.57, "rho_residual": 0.35},
                {"history_id": "ILLUSTRATIVE-3", "n_pairs": 6, "n_distinct_burden": 4, "rho_burden": 0.63, "rho_residual": 0.40},
                {"history_id": "ILLUSTRATIVE-4", "n_pairs": 5, "n_distinct_burden": 3, "rho_burden": 0.51, "rho_residual": 0.33},
            ],
            {
                "rolling": {"rho_burden": -0.30},
                "capped_n": {"rho_burden": 0.41},
                "alternate_anchor": {"rho_burden": 0.55},
            },
            "INCONCLUSIVE",
            "the rolling boundary convention reverses the sign, so the association is boundary-phase dependent and no disposition is declared",
        ),
        case(
            "inconclusive-unverified-sensitivity",
            "week",
            [
                {"history_id": "ILLUSTRATIVE-1", "n_pairs": 12, "n_distinct_burden": 7, "rho_burden": 0.72, "rho_residual": 0.45},
                {"history_id": "ILLUSTRATIVE-2", "n_pairs": 9, "n_distinct_burden": 5, "rho_burden": 0.60, "rho_residual": 0.38},
                {"history_id": "ILLUSTRATIVE-3", "n_pairs": 7, "n_distinct_burden": 4, "rho_burden": 0.64, "rho_residual": 0.42},
                {"history_id": "ILLUSTRATIVE-4", "n_pairs": 6, "n_distinct_burden": 3, "rho_burden": 0.55, "rho_residual": 0.34},
            ],
            {
                "rolling": {"rho_burden": 0.50},
                "capped_n": {"rho_burden": None},
                "alternate_anchor": {"rho_burden": 0.58},
            },
            "INCONCLUSIVE",
            "an unverifiable sensitivity variant blocks a promise; unknown is never treated as a pass",
        ),
        case(
            "promise-without-specificity",
            "month",
            [
                {"history_id": "ILLUSTRATIVE-1", "n_pairs": 9, "n_distinct_burden": 6, "rho_burden": 0.80, "rho_residual": 0.70},
                {"history_id": "ILLUSTRATIVE-2", "n_pairs": 8, "n_distinct_burden": 5, "rho_burden": 0.75, "rho_residual": 0.66},
                {"history_id": "ILLUSTRATIVE-3", "n_pairs": 7, "n_distinct_burden": 4, "rho_burden": 0.90, "rho_residual": 0.85},
                {"history_id": "ILLUSTRATIVE-4", "n_pairs": 6, "n_distinct_burden": 4, "rho_burden": 0.70, "rho_residual": 0.60},
            ],
            {
                "rolling": {"rho_burden": 0.78},
                "capped_n": {"rho_burden": 0.72},
                "alternate_anchor": {"rho_burden": 0.76},
            },
            "PROMISING",
            "even an illustrative maximum-strength association yields specificity_claim false and starts only a proposal for separate scaling authorization",
        ),
    ]
    return {
        "label": "ILLUSTRATIVE-DECISION-RULE-EXAMPLES-NOT-EMPIRICAL-EVIDENCE",
        "warning": (
            "every number in this file is invented to exercise the decision-rule boundaries; "
            "no movement, embedding, score or model outcome exists for this study"
        ),
        "policy": decision_policy(),
        "cases": cases,
    }


# --- comparable-cost accounting ---------------------------------------------


def assert_cost_within_declared_bounds(
    span_rows: int,
    reference_rows: int,
) -> dict[str, object]:
    """Fail closed when the study's per-seed row requirement exceeds the envelope."""
    per_seed = int(span_rows) + int(reference_rows)
    if per_seed > DECLARED_MAX_ROWS_PER_SEED:
        raise RuntimeError(
            f"per-seed embedding requirement {per_seed} exceeds the declared envelope "
            f"{DECLARED_MAX_ROWS_PER_SEED}"
        )
    total = per_seed * len(REPRESENTATION_SEEDS)
    if total > DECLARED_MAX_ROWS_TOTAL:
        raise RuntimeError(
            f"total embedding requirement {total} exceeds the declared envelope "
            f"{DECLARED_MAX_ROWS_TOTAL}"
        )
    span_only_total = int(span_rows) * len(REPRESENTATION_SEEDS)
    return {
        "span_rows_per_seed": int(span_rows),
        "reference_rows_per_seed": int(reference_rows),
        "rows_per_seed": per_seed,
        "rows_total_three_seeds": total,
        "span_only_rows_total_three_seeds": span_only_total,
        "declared_max_rows_per_seed": DECLARED_MAX_ROWS_PER_SEED,
        "declared_max_rows_total": DECLARED_MAX_ROWS_TOTAL,
        "inherited_headroom_forwards": INHERITED_HEADROOM_FORWARDS,
        "fresh_span_only_exceeds_inherited_headroom": (
            span_only_total > INHERITED_HEADROOM_FORWARDS
        ),
        "fresh_span_only_excess_rows": max(0, span_only_total - INHERITED_HEADROOM_FORWARDS),
        "reuse_required": True,
        "forwards_executed_by_this_study": 0,
    }


def describe_tiers() -> dict[str, list[str]]:
    """Return the decision ids grouped by resolution tier (for reports and tests)."""
    out: dict[str, list[str]] = {}
    for item in decision_resolutions():
        out.setdefault(str(item["tier"]), []).append(str(item["id"]))
    return out


__all__ = [
    "ALTERNATE_ANCHOR_EPOCH_S",
    "BLOCKED_CLAIMS",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CAPPED_N_RULE",
    "CELL_STATES",
    "CLAIM_BOUNDARIES",
    "CONTRAST_STATES",
    "DECISION_ALPHA",
    "DECISION_CONSISTENCY_MIN_HISTORIES",
    "DECISION_FAMILY_SIZE",
    "DECISION_MIN_ESTIMABLE_HISTORIES",
    "DECISION_MIN_PAIRS_PER_HISTORY",
    "DECISION_PROMISE_BURDEN_RHO",
    "DECISION_PROMISE_RESIDUAL_RHO",
    "DECISION_SENSITIVITY_RETENTION",
    "DECISION_SENSITIVITY_VARIANTS",
    "DECLARED_MAX_ROWS_PER_SEED",
    "DECLARED_MAX_ROWS_TOTAL",
    "DISPOSITIONS",
    "EMBEDDING_DIM",
    "ENDPOINT_HIERARCHY",
    "HORIZONS",
    "INHERITED_CUMULATIVE_FORWARDS",
    "INHERITED_FORWARD_CAP",
    "INHERITED_HEADROOM_FORWARDS",
    "MIN_PERIOD_FILES",
    "MIN_REFERENCE_FILES",
    "MODES",
    "PAIR_STRUCTURAL_FIELDS",
    "PERIOD_STRUCTURAL_FIELDS",
    "POSTHOC_PAIR_FIELDS",
    "POSTHOC_PERIOD_FIELDS",
    "PRIMARY_ANCHOR_EPOCH_S",
    "PRIMARY_MODE",
    "PROTOCOL_DOC",
    "REPRESENTATION_SEEDS",
    "SENSITIVITY_MODES",
    "STUDY_ID",
    "assert_cost_within_declared_bounds",
    "assert_study_construction_label_blind",
    "build_label_blind_manifest",
    "build_posthoc_burden",
    "contrast_availability",
    "coverage_matrix",
    "cost_policy",
    "decision_policy",
    "decision_resolutions",
    "describe_tiers",
    "estimability",
    "evaluate_source",
    "evaluate_decision",
    "expected_disposition",
    "horizon_counts",
    "min_attainable_tail",
    "pair_identity",
    "period_identity",
    "primary_calendar",
    "projection_sensitivity",
    "reference_identity",
    "resolution_block",
    "sensitivity_calendars",
    "study_freeze",
    "study_sha256",
    "study_spec",
    "worked_decision_examples",
]
