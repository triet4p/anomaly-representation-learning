"""Business-horizon collective movement measurement — metadata feasibility (S24-7-M2).

This module implements **only** the reproducible metadata machinery a prospective
business-horizon design needs: calendar period membership, exposure accounting,
period-pair comparability, support/abstention classification and the per-horizon
feasibility matrix. It computes no embedding statistic, opens no model,
checkpoint, score or waveform, and defines no performance gate.

Business horizons (owner requirement, not a retroactive change to the frozen
G1/G2 grid):

- ``day`` — a calendar day compared with the immediately preceding calendar day.
- ``week`` — a calendar week compared with the immediately preceding week.
- ``month`` — a calendar month compared with the immediately preceding calendar
  month at its **actual** length (28/29/30/31 days; a nominal 28- or 30-day
  month is never substituted).

Binding boundaries:

- **Label-blind construction.** Period membership, exposure and comparability are
  built from source-observable metadata only: ``file_id``/``start_time``/
  ``end_time``/``robot_id``/``program_id``/``last_reset_time``/``is_quarantined``/
  ``is_censored``/``n_valid_patches`` plus the manifest ``maintenance_windows``
  table and the inherited training-eligibility predicate. Annotations
  (``file_label``, failure events, degradation episodes) are projected
  separately and joined only after membership exists.
  :func:`assert_period_construction_label_blind` fails closed if construction
  code references an annotation name.
- **No annotation means UNKNOWN, not healthy.** A period with absent or partial
  annotation coverage is ``UNANNOTATED``/``UNANNOTATED-PARTIAL``; only complete
  coverage with zero abnormal members and zero covered events is
  ``ANNOTATED-CLEAN``.
- **Support floors are inherited unchanged** (30 eligible files per period, 200
  reference files). This module classifies support and abstains; it never lowers
  a floor to manufacture comparable periods. A descriptive ladder is published
  beside the floor.

The calendar projection (anchor epoch, timezone, week start, month anchor day)
and every numeric default are **declared assumptions**, exposed through
:func:`proposal_spec` and bound by its digest. The accepted Development clock is
relative (seconds from a per-history origin) and carries no wall-clock epoch, so
the projection anchor is an owner decision, not an established business fact.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import inspect
import math
import textwrap
import zoneinfo
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from representation.sprint24_context_audit import (
    HistoryMetadata,
    prepare_history,
    project_observable,
    sha256_object,
    sha256_text,
)

#: Proposal family identifier frozen by this module.
PROPOSAL_ID = "S24-BIZ-HORIZON-v1"

#: Seconds per day. Every period/exposure unit in this module is days.
DAY_S = 86_400.0

#: Required business comparison horizons.
HORIZONS = ("day", "week", "month")

#: Boundary conventions. ``calendar`` is the primary design; ``rolling`` is the
#: declared sensitivity convention (analysis-anchored blocks; for months a
#: same-day-of-month window whose length is the real inter-anchor length).
MODES = ("calendar", "rolling")

#: Proposed IANA timezone for calendar boundaries (owner decision).
PROPOSED_TIMEZONE = "UTC"

#: Proposed week start (owner decision); ISO-8601 Monday.
PROPOSED_WEEK_START = "monday"

#: Proposed day-of-month anchor for rolling months (owner decision).
PROPOSED_MONTH_ANCHOR_DAY = 15

#: Weekday ordinals accepted for ``PROPOSED_WEEK_START``.
WEEK_START_DAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

#: Inherited minimum eligible files per period (locked protocol Section 4).
MIN_PERIOD_FILES = 30

#: Inherited minimum fixed causal reference pool (locked protocol Section 5).
MIN_REFERENCE_FILES = 200

#: Descriptive support ladder published beside the inherited floor (not gates).
SUPPORT_LADDER = (5, 10, 15, 20, 30, 100)

#: Proposed exposure-comparability floor for a period pair (owner decision).
PROPOSED_EXPOSURE_RATIO_FLOOR = 0.8

#: Proposed resample count for the sampling null and half-split band.
PROPOSED_NULL_RESAMPLES = 200

#: Inherited split-half/null seed convention (locked protocol Sections 8 and 10).
PROPOSED_NULL_SEED = 2407

#: Proposed fixed cluster count for the composition endpoint (owner decision).
PROPOSED_CLUSTER_COUNT = 8

#: Proposed minimum matched/quiet pairs before a control contrast is attempted.
PROPOSED_MIN_CONTRAST_PAIRS = 2

#: Period coverage states. ``FULL`` requires the period to sit inside the
#: inherited analysis span ``[c1, tmax]``.
COVERAGE_STATES = ("FULL", "PARTIAL-PRE", "PARTIAL-POST", "PARTIAL-BOTH")

#: Period support states.
PERIOD_SUPPORT_STATES = ("SUPPORTED", "INSUFFICIENT-PERIOD-SUPPORT")

#: Period-pair exposure states.
EXPOSURE_STATES = ("EXPOSURE-MATCHED", "EXPOSURE-MISMATCH")

#: Period-pair comparability states, in precedence order.
PAIR_COMPARISON_STATES = (
    "INSUFFICIENT-REFERENCE",
    "PARTIAL-PERIOD",
    "INSUFFICIENT-PERIOD-SUPPORT",
    "COMPARABLE-UNMATCHED-EXPOSURE",
    "COMPARABLE-MATCHED",
)

#: Post-hoc annotation states. Absent coverage is never ``ANNOTATED-CLEAN``.
ANNOTATION_STATES = (
    "ANNOTATED-CLEAN",
    "ANNOTATED-ABNORMAL",
    "UNANNOTATED-PARTIAL",
    "UNANNOTATED",
)

#: Post-hoc attribution class of a period pair.
ATTRIBUTION_CLASSES = (
    "EVENT-CURRENT",
    "EVENT-PREVIOUS",
    "NO-EVENT-ANNOTATED",
    "UNANNOTATED",
)

#: Per-horizon feasibility states.
HORIZON_STATES = ("FEASIBLE-DESCRIPTIVE", "PARTIALLY-FEASIBLE", "INFEASIBLE")

#: Control-availability states (the business question needs a normal-variation
#: reference; these record what the metadata can actually supply).
CONTROL_STATES = (
    "ANNOTATED-CLEAN-CONTROL-AVAILABLE",
    "QUIET-PERIOD-CONTROL-AVAILABLE",
    "NO-QUIET-PERIOD-CONTROL",
    "NO-SUPPORTED-PERIODS",
)

#: Frozen per-period output fields (units in :data:`PERIOD_FIELD_UNITS`).
PERIOD_ROW_FIELDS = (
    "history_id",
    "horizon",
    "mode",
    "period_key",
    "period_start_rel_s",
    "period_end_rel_s",
    "period_length_days",
    "period_start_iso",
    "period_end_iso",
    "n_members",
    "n_straddling_members",
    "exposure_days",
    "coverage_state",
    "coverage_reason",
    "support_state",
    "meets_support_floor",
    "n_reset_edges",
    "member_ids_sha256",
    "n_abnormal_members",
    "n_unknown_label_members",
    "abnormal_member_fraction",
    "n_failure_events_covered",
    "n_degradation_onsets_covered",
    "annotation_state",
)

#: Frozen per-pair output fields (units in :data:`PAIR_ROW_FIELD_UNITS`).
PAIR_ROW_FIELDS = (
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
    "previous_annotation_state",
    "current_annotation_state",
    "n_abnormal_members",
    "abnormal_member_fraction",
    "n_failure_events_covered",
    "n_degradation_onsets_covered",
    "attribution_class",
)

#: Frozen per-history x horizon x mode feasibility fields.
FEASIBILITY_ROW_FIELDS = (
    "history_id",
    "horizon",
    "mode",
    "n_periods",
    "n_full_periods",
    "n_partial_periods",
    "n_supported_periods",
    "min_period_members",
    "median_period_members",
    "max_period_members",
    "n_full_periods_ge_5",
    "n_full_periods_ge_10",
    "n_full_periods_ge_15",
    "n_full_periods_ge_20",
    "n_full_periods_ge_30",
    "n_full_periods_ge_100",
    "period_length_days_min",
    "period_length_days_max",
    "n_pairs",
    "n_pairs_full",
    "n_pairs_supported_full",
    "n_pairs_comparable_matched",
    "n_pairs_comparable_unmatched",
    "median_exposure_ratio_supported",
    "median_mix_drift_tv_supported",
    "n_supported_pairs_zero_failure_events",
    "n_supported_pairs_annotated_clean",
    "n_periods_annotated_clean",
    "reference_files",
    "reference_state",
    "horizon_state",
    "control_state",
)

#: Frozen aggregate horizon-matrix fields.
HORIZON_MATRIX_FIELDS = (
    "horizon",
    "mode",
    "n_histories",
    "n_periods_total",
    "n_full_periods_total",
    "n_supported_periods_total",
    "n_pairs_total",
    "n_pairs_comparable_matched_total",
    "n_pairs_comparable_unmatched_total",
    "n_supported_pairs_zero_failure_events_total",
    "n_supported_pairs_annotated_clean_total",
    "median_period_members",
    "min_period_members",
    "max_period_members",
    "median_supported_pairs_per_history",
    "min_supported_pairs_per_history",
    "max_supported_pairs_per_history",
    "period_length_days_min",
    "period_length_days_max",
    "horizon_state",
    "control_state",
)

#: Units/definitions for every frozen per-period field.
PERIOD_FIELD_UNITS: dict[str, str] = {
    "history_id": "Development history identifier",
    "horizon": "business horizon (day/week/month)",
    "mode": "boundary convention (calendar/rolling)",
    "period_key": "period identifier from its projected calendar start",
    "period_start_rel_s": "seconds on the history's own relative clock (exclusive lower bound)",
    "period_end_rel_s": "seconds on the history's own relative clock (inclusive upper bound)",
    "period_length_days": "days, actual calendar length (28/29/30/31 for months)",
    "period_start_iso": "declared calendar projection of the period start (assumption, not a corpus date)",
    "period_end_iso": "declared calendar projection of the period end (assumption, not a corpus date)",
    "n_members": "eligible files assigned by end_time in (period_start, period_end]",
    "n_straddling_members": "members whose start_time is at or before the period start",
    "exposure_days": "summed file-days of member observation clipped to the period, in days",
    "coverage_state": "FULL | PARTIAL-PRE | PARTIAL-POST | PARTIAL-BOTH",
    "coverage_reason": "why a period is partial, empty when FULL",
    "support_state": "SUPPORTED at the inherited 30-file floor, else INSUFFICIENT-PERIOD-SUPPORT",
    "meets_support_floor": "boolean alias of support_state == SUPPORTED",
    "n_reset_edges": "maintenance-window end times (recommissioning) inside the period",
    "member_ids_sha256": "SHA-256 of the sorted member file IDs",
    "n_abnormal_members": "post-hoc members whose file_label is not 'normal'",
    "n_unknown_label_members": "post-hoc members with no file_label row",
    "abnormal_member_fraction": "n_abnormal_members / n_members (post-hoc)",
    "n_failure_events_covered": "post-hoc failure events with failure_time in (start, end]",
    "n_degradation_onsets_covered": "post-hoc degradation onsets in (start, end]",
    "annotation_state": "ANNOTATED-CLEAN | ANNOTATED-ABNORMAL | UNANNOTATED-PARTIAL | UNANNOTATED",
}

#: Units/definitions for every frozen per-pair field.
PAIR_ROW_FIELD_UNITS: dict[str, str] = {
    "history_id": "Development history identifier",
    "horizon": "business horizon (day/week/month)",
    "mode": "boundary convention (calendar/rolling)",
    "previous_key": "period key compared against the current period",
    "current_key": "current period key",
    "previous_start_rel_s": "current pair lower bound on the relative clock (exclusive)",
    "current_end_rel_s": "current pair upper bound on the relative clock (inclusive)",
    "n_previous": "eligible files in the previous period",
    "n_current": "eligible files in the current period",
    "exposure_previous_days": "previous-period member file-days",
    "exposure_current_days": "current-period member file-days",
    "exposure_ratio": "min(exposure) / max(exposure) over the pair, dimensionless",
    "mix_drift_tv": "total-variation distance between the pair's robot mixes, dimensionless",
    "n_reset_edges_crossed": "recommissioning edges in (previous_start, current_end]",
    "coverage_state": "PAIR-FULL when both periods are FULL, else PAIR-PARTIAL",
    "support_state": "PAIR-SUPPORTED when both periods meet the 30-file floor",
    "exposure_state": "EXPOSURE-MATCHED when exposure_ratio >= the proposed floor",
    "comparison_state": "closed comparability state (see PAIR_COMPARISON_STATES)",
    "previous_annotation_state": "post-hoc annotation state of the previous period",
    "current_annotation_state": "post-hoc annotation state of the current period",
    "n_abnormal_members": "post-hoc abnormal members across both periods",
    "abnormal_member_fraction": "n_abnormal_members / (n_previous + n_current) (post-hoc)",
    "n_failure_events_covered": "post-hoc failure events across the pair interval",
    "n_degradation_onsets_covered": "post-hoc degradation onsets across the pair interval",
    "attribution_class": "EVENT-CURRENT | EVENT-PREVIOUS | NO-EVENT-ANNOTATED | UNANNOTATED",
}

#: Units/definitions for every frozen feasibility and horizon-matrix field.
FEASIBILITY_FIELD_UNITS: dict[str, str] = {
    "history_id": "Development history identifier",
    "horizon": "business horizon (day/week/month)",
    "mode": "boundary convention (calendar/rolling)",
    "n_periods": "periods intersecting the analysis span",
    "n_full_periods": "periods fully inside [c1, tmax]",
    "n_partial_periods": "periods extending beyond an analysis-span edge",
    "n_supported_periods": "periods with >= 30 eligible member files (inherited floor)",
    "min_period_members": "eligible files, minimum over all periods",
    "median_period_members": "eligible files, median over all periods",
    "max_period_members": "eligible files, maximum over all periods",
    "n_full_periods_ge_5": "full periods with at least 5 eligible files (descriptive ladder)",
    "n_full_periods_ge_10": "full periods with at least 10 eligible files (descriptive ladder)",
    "n_full_periods_ge_15": "full periods with at least 15 eligible files (descriptive ladder)",
    "n_full_periods_ge_20": "full periods with at least 20 eligible files (descriptive ladder)",
    "n_full_periods_ge_30": "full periods with at least 30 eligible files (inherited floor)",
    "n_full_periods_ge_100": "full periods with at least 100 eligible files (descriptive ladder)",
    "period_length_days_min": "days; real calendar length of the shortest full period",
    "period_length_days_max": "days; real calendar length of the longest full period",
    "n_pairs": "consecutive period pairs enumerated",
    "n_pairs_full": "pairs with both periods FULL",
    "n_pairs_supported_full": "FULL pairs with both periods at or above the inherited floor",
    "n_pairs_comparable_matched": "pairs passing coverage, support and exposure matching",
    "n_pairs_comparable_unmatched": "pairs passing coverage and support but failing exposure matching",
    "median_exposure_ratio_supported": "median min/max member file-day ratio over comparable pairs",
    "median_mix_drift_tv_supported": "median robot-mix total-variation drift over comparable pairs",
    "n_supported_pairs_zero_failure_events": "comparable pairs covering no annotated failure event (post-hoc)",
    "n_supported_pairs_annotated_clean": "comparable pairs where both periods are ANNOTATED-CLEAN (post-hoc)",
    "n_periods_annotated_clean": "periods with complete annotation coverage and no abnormal member or event (post-hoc)",
    "reference_files": "eligible files with end_time <= c1 (inherited fixed reference pool)",
    "reference_state": "REFERENCE-OK at the inherited 200-file floor, else INSUFFICIENT-REFERENCE",
    "horizon_state": "FEASIBLE-DESCRIPTIVE | PARTIALLY-FEASIBLE | INFEASIBLE for this history",
    "control_state": "control availability for this history (see CONTROL_STATES)",
}

#: Units/definitions for the aggregate horizon matrix.
HORIZON_MATRIX_FIELD_UNITS: dict[str, str] = {
    "horizon": "business horizon (day/week/month)",
    "mode": "boundary convention (calendar/rolling)",
    "n_histories": "Development histories contributing rows",
    "n_periods_total": "periods across histories",
    "n_full_periods_total": "full periods across histories",
    "n_supported_periods_total": "periods at or above the inherited file floor across histories",
    "n_pairs_total": "consecutive pairs across histories",
    "n_pairs_comparable_matched_total": "exposure-matched comparable pairs across histories",
    "n_pairs_comparable_unmatched_total": "supported but exposure-mismatched pairs across histories",
    "n_supported_pairs_zero_failure_events_total": "comparable pairs with zero covered events (post-hoc)",
    "n_supported_pairs_annotated_clean_total": "comparable pairs with both periods clean (post-hoc)",
    "median_period_members": "median of per-history median eligible files per period",
    "min_period_members": "minimum eligible files over every period and history",
    "max_period_members": "maximum eligible files over every period and history",
    "median_supported_pairs_per_history": "median comparable pairs per history",
    "min_supported_pairs_per_history": "minimum comparable pairs per history",
    "max_supported_pairs_per_history": "maximum comparable pairs per history",
    "period_length_days_min": "days; shortest real full-period length observed",
    "period_length_days_max": "days; longest real full-period length observed",
    "horizon_state": "FEASIBLE-DESCRIPTIVE | PARTIALLY-FEASIBLE | INFEASIBLE",
    "control_state": "control availability (see CONTROL_STATES)",
}

#: Proposed movement endpoints (design catalogue; execution is separately gated).
ENDPOINTS: tuple[dict[str, object], ...] = (
    {
        "id": "E1",
        "name": "DIST-MOVE",
        "statistic": "unbiased squared MMD, fixed Gaussian kernel, bandwidth from the reference pool",
        "units": "squared distance in the frozen original 128-d embedding space",
        "role": "primary collective distribution-movement endpoint",
    },
    {
        "id": "E2",
        "name": "CLUSTER-MOVE",
        "statistic": "total-variation distance between period cluster-occupancy vectors",
        "units": "dimensionless, range [0, 1]",
        "role": "collective cluster/composition movement endpoint (clusters fixed on the reference)",
    },
    {
        "id": "E3",
        "name": "LOC-MOVE",
        "statistic": "squared Euclidean mean shift standardized by reference per-dimension SD",
        "units": "dimensionless (reference SD units, squared)",
        "role": "location component of movement",
    },
    {
        "id": "E4",
        "name": "DISP-MOVE",
        "statistic": "absolute log ratio of trace covariance",
        "units": "natural-log ratio, dimensionless",
        "role": "dispersion component of movement",
    },
)

#: Predeclared contrasts and normal-variation references (design catalogue).
CONTRASTS: tuple[dict[str, object], ...] = (
    {
        "id": "N1",
        "name": "SAMPLING-NULL",
        "rule": "paired label permutation inside the pooled pair, preserving n; fixed resamples and seed",
        "role": "sampling-noise reference only, never a specificity or operating point",
    },
    {
        "id": "N2",
        "name": "HALF-SPLIT-BAND",
        "rule": "seeded half splits of each period; movement between halves",
        "role": "conservative within-period sampling band",
    },
    {
        "id": "C1",
        "name": "EVENT-LOAD-ASSOCIATION",
        "rule": "Spearman correlation of E1/E2 with covered event count and abnormal member fraction",
        "role": "primary association contrast; reports negative results unchanged",
    },
    {
        "id": "C2",
        "name": "MIX-ADJUSTED-RESIDUAL",
        "rule": "E1 residual after regressing on observable mix drift, then associated with event load",
        "role": "addresses the G3 mix-drift confound (r(MMD2, TV drift) = +0.78)",
    },
    {
        "id": "C3",
        "name": "QUIET-PERIOD-CONTRAST",
        "rule": "event pairs vs pairs covering no annotated event, matched on mix drift and exposure",
        "role": "requires quiet periods; abstains when none exist (label-blind matching rule)",
    },
    {
        "id": "C4",
        "name": "TEMPORAL-SPECIFICITY",
        "rule": "event period vs its own preceding and following pairs",
        "role": "requires event-free neighbours; abstains when none exist",
    },
)

#: Frozen output file specification for the CLI (descriptions only).
OUTPUT_FILES: dict[str, str] = {
    "proposal-freeze.json": "frozen proposal identity, defaults, rules and endpoints with digest",
    "period-table.json": "every period instance with membership, exposure, support and post-hoc annotations",
    "period-table.csv": "flat CSV projection of the period table",
    "pair-table.json": "every consecutive period pair with comparability and post-hoc annotations",
    "pair-table.csv": "flat CSV projection of the pair table",
    "feasibility-matrix.json": "per-history x horizon x mode feasibility rows and horizon aggregates",
    "worked-examples.json": "reproducible real-metadata membership/exposure examples with units",
    "illustrative-examples.json": "synthetic calendar cases, explicitly labelled illustrative",
    "sensitivity.json": "anchor/week-start/mode sensitivity of the feasibility counts",
    "run.json": "inputs, digests, environment, cost and limitations",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


@dataclass(frozen=True, slots=True)
class CalendarConfig:
    """Declared calendar projection for business-period boundaries.

    The accepted Development clock is relative, so the anchor is a declared
    assumption. ``timezone`` is an IANA zone name resolved through ``zoneinfo``;
    ``week_start`` is a weekday name; ``month_anchor_day`` only affects the
    rolling month convention.
    """

    anchor_epoch_s: float
    timezone: str = PROPOSED_TIMEZONE
    week_start: str = PROPOSED_WEEK_START
    month_anchor_day: int = PROPOSED_MONTH_ANCHOR_DAY

    def __post_init__(self) -> None:
        if self.week_start not in WEEK_START_DAYS:
            raise ValueError(f"unknown week start: {self.week_start!r}")
        if not 1 <= int(self.month_anchor_day) <= 31:
            raise ValueError("month_anchor_day must be in 1..31")
        # A day-of-month anchor past the end of a short month clamps to that
        # month's last day, so rolling-month lengths remain real inter-anchor
        # calendar gaps (28..31 days) instead of a nominal month length.

    @property
    def zone(self) -> dt.tzinfo:
        try:
            return zoneinfo.ZoneInfo(self.timezone)
        except zoneinfo.ZoneInfoNotFoundError as error:
            raise ValueError(f"unknown IANA timezone: {self.timezone!r}") from error

    @property
    def week_start_ordinal(self) -> int:
        return WEEK_START_DAYS[self.week_start]

    def local(self, rel_s: float) -> dt.datetime:
        """Return the projected local datetime of a relative-clock instant."""
        return dt.datetime.fromtimestamp(self.anchor_epoch_s + float(rel_s), self.zone)

    def relative(self, moment: dt.datetime) -> float:
        """Return the relative-clock instant of a projected local datetime."""
        return float(moment.timestamp()) - float(self.anchor_epoch_s)

    def local_midnight(self, year: int, month: int, day: int) -> dt.datetime:
        return dt.datetime(year, month, day, tzinfo=self.zone)

    def describe(self) -> dict[str, object]:
        return {
            "anchor_epoch_s": float(self.anchor_epoch_s),
            "anchor_iso": dt.datetime.fromtimestamp(self.anchor_epoch_s, dt.timezone.utc).isoformat(),
            "timezone": self.timezone,
            "week_start": self.week_start,
            "month_anchor_day": int(self.month_anchor_day),
            "projection_basis": (
                "declared assumption: accepted Development timestamps are relative "
                "seconds with no wall-clock epoch, so this anchor is not a corpus fact"
            ),
        }


def _month_length(year: int, month: int) -> int:
    if month == 12:
        following = dt.date(year + 1, 1, 1)
    else:
        following = dt.date(year, month + 1, 1)
    return (following - dt.date(year, month, 1)).days


def _previous_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


@dataclass(frozen=True, slots=True)
class Period:
    """One business period on the history's relative clock.

    Assignment convention (inherited trailing-window convention): a file belongs
    to the period whose interval satisfies ``start < end_time <= end``, i.e. the
    interval is left-exclusive and right-inclusive in ``end_time``.
    """

    horizon: str
    mode: str
    key: str
    start: float
    end: float

    @property
    def length_days(self) -> float:
        """Real period length in days, rounded to nanosecond precision."""
        return round((self.end - self.start) / DAY_S, 9)


@dataclass(frozen=True, slots=True)
class PeriodGrid:
    """Enumerated business periods for one horizon, convention and analysis span."""

    horizon: str
    mode: str
    calendar: CalendarConfig
    analysis_start: float
    analysis_end: float

    def __post_init__(self) -> None:
        if self.horizon not in HORIZONS:
            raise ValueError(f"unknown horizon: {self.horizon!r}")
        if self.mode not in MODES:
            raise ValueError(f"unknown mode: {self.mode!r}")
        if not self.analysis_end > self.analysis_start:
            raise ValueError("analysis span must be increasing")

    # -- boundary computation (observable metadata only) -------------------
    def _containing_calendar_day(self, rel_s: float) -> Period:
        moment = self.calendar.local(rel_s)
        start_local = self.calendar.local_midnight(moment.year, moment.month, moment.day)
        following = dt.date(moment.year, moment.month, moment.day) + dt.timedelta(days=1)
        end_local = self.calendar.local_midnight(following.year, following.month, following.day)
        return Period(
            horizon="day",
            mode=self.mode,
            key=f"D{start_local.date().isoformat()}",
            start=self.calendar.relative(start_local),
            end=self.calendar.relative(end_local),
        )

    def _containing_calendar_week(self, rel_s: float) -> Period:
        moment = self.calendar.local(rel_s)
        midnight = self.calendar.local_midnight(moment.year, moment.month, moment.day)
        offset = (midnight.weekday() - self.calendar.week_start_ordinal) % 7
        start_date = midnight.date() - dt.timedelta(days=offset)
        start_local = self.calendar.local_midnight(start_date.year, start_date.month, start_date.day)
        end_date = start_date + dt.timedelta(days=7)
        end_local = self.calendar.local_midnight(end_date.year, end_date.month, end_date.day)
        return Period(
            horizon="week",
            mode=self.mode,
            key=f"W{start_date.isoformat()}",
            start=self.calendar.relative(start_local),
            end=self.calendar.relative(end_local),
        )

    def _containing_calendar_month(self, rel_s: float) -> Period:
        moment = self.calendar.local(rel_s)
        start_local = self.calendar.local_midnight(moment.year, moment.month, 1)
        year, month = _next_month(moment.year, moment.month)
        end_local = self.calendar.local_midnight(year, month, 1)
        return Period(
            horizon="month",
            mode=self.mode,
            key=f"M{moment.year:04d}-{moment.month:02d}",
            start=self.calendar.relative(start_local),
            end=self.calendar.relative(end_local),
        )

    def _rolling_month_anchor(self, year: int, month: int) -> dt.datetime:
        day = min(int(self.calendar.month_anchor_day), _month_length(year, month))
        return self.calendar.local_midnight(year, month, day)

    def _containing_rolling_month(self, rel_s: float) -> Period:
        moment = self.calendar.local(rel_s)
        anchor = self._rolling_month_anchor(moment.year, moment.month)
        if moment < anchor:
            previous_year, previous_month = _previous_month(moment.year, moment.month)
            start_local = self._rolling_month_anchor(previous_year, previous_month)
            end_local = anchor
            key_year, key_month = previous_year, previous_month
        else:
            next_year, next_month = _next_month(moment.year, moment.month)
            start_local = anchor
            end_local = self._rolling_month_anchor(next_year, next_month)
            key_year, key_month = moment.year, moment.month
        return Period(
            horizon="month",
            mode=self.mode,
            key=(
                f"M{key_year:04d}-{key_month:02d}"
                f"@{int(self.calendar.month_anchor_day):02d}"
            ),
            start=self.calendar.relative(start_local),
            end=self.calendar.relative(end_local),
        )

    def _containing_anchored_block(self, rel_s: float, length_s: float) -> Period:
        index = math.floor((float(rel_s) - self.analysis_start) / length_s)
        start = self.analysis_start + index * length_s
        end = start + length_s
        start_date = self.calendar.local(start).date()
        prefix = "D" if self.horizon == "day" else "W"
        return Period(
            horizon=self.horizon,
            mode=self.mode,
            key=f"{prefix}{start_date.isoformat()}A",
            start=start,
            end=end,
        )

    def period_containing(self, rel_s: float) -> Period:
        """Return the period whose interval ``start <= rel < end`` contains ``rel``."""
        if self.mode == "rolling" and self.horizon == "day":
            return self._containing_anchored_block(rel_s, DAY_S)
        if self.mode == "rolling" and self.horizon == "week":
            return self._containing_anchored_block(rel_s, 7.0 * DAY_S)
        if self.mode == "rolling" and self.horizon == "month":
            return self._containing_rolling_month(rel_s)
        if self.horizon == "day":
            return self._containing_calendar_day(rel_s)
        if self.horizon == "week":
            return self._containing_calendar_week(rel_s)
        return self._containing_calendar_month(rel_s)

    def periods(self) -> tuple[Period, ...]:
        """Enumerate every period intersecting the analysis span, in time order.

        The first and/or last period may extend beyond the span; that is recorded
        as a partial coverage state rather than silently dropped.
        """
        first = self.period_containing(self.analysis_start)
        while first.end <= self.analysis_start:
            first = self.period_containing(first.end + 1.0)
        out = [first]
        while out[-1].start < self.analysis_end:
            following = self.period_containing(out[-1].end + 1.0)
            if following.start < out[-1].end:
                raise RuntimeError("period enumeration is not monotone")
            if following.start >= self.analysis_end:
                break
            out.append(following)
        return tuple(out)

    def period_for_end_time(self, rel_s: float) -> Period:
        """Return the period that owns a file ending at ``rel_s``.

        Implements the inherited assignment rule ``start < end_time <= end``. An
        ``end_time`` exactly on a boundary is owned by the period that ends there;
        the comparison is exact (no floating epsilon) because every boundary is
        computed from the same declared calendar.
        """
        value = float(rel_s)
        period = self.period_containing(value)
        if value <= period.start:
            # Periods are at least one day long, so one second earlier is safely
            # inside the period that ends at this boundary.
            return self.period_containing(value - 1.0)
        return period


def proposal_spec() -> dict[str, object]:
    """Return the frozen proposal identity, defaults, rules and endpoint catalogue."""
    return {
        "proposal_id": PROPOSAL_ID,
        "horizons": list(HORIZONS),
        "modes": list(MODES),
        "proposed_defaults": {
            "timezone": PROPOSED_TIMEZONE,
            "week_start": PROPOSED_WEEK_START,
            "month_anchor_day": PROPOSED_MONTH_ANCHOR_DAY,
            "exposure_ratio_floor": PROPOSED_EXPOSURE_RATIO_FLOOR,
            "null_resamples": PROPOSED_NULL_RESAMPLES,
            "null_seed": PROPOSED_NULL_SEED,
            "cluster_count": PROPOSED_CLUSTER_COUNT,
            "min_contrast_pairs": PROPOSED_MIN_CONTRAST_PAIRS,
        },
        "inherited_unchanged": {
            "min_period_files": MIN_PERIOD_FILES,
            "min_reference_files": MIN_REFERENCE_FILES,
            "support_ladder": list(SUPPORT_LADDER),
            "eligibility_predicate": "representation.sprint24_views.is_eligible_for_training",
            "analysis_span": "(c1, tmax] per history, never concatenated",
            "reference_pool": "eligible files with end_time <= c1",
        },
        "membership_rule": {
            "assignment": "file end_time belongs to the period with start < end_time <= end",
            "boundary_convention": "inherited left-exclusive/right-inclusive trailing-window rule",
            "exposure": "summed member file-days clipped to the period interval",
            "full_coverage": "period start >= c1 and period end <= tmax",
        },
        "state_vocabularies": {
            "coverage": list(COVERAGE_STATES),
            "period_support": list(PERIOD_SUPPORT_STATES),
            "exposure": list(EXPOSURE_STATES),
            "comparison": list(PAIR_COMPARISON_STATES),
            "annotation": list(ANNOTATION_STATES),
            "attribution": list(ATTRIBUTION_CLASSES),
            "horizon": list(HORIZON_STATES),
            "control": list(CONTROL_STATES),
        },
        "endpoints": [dict(item) for item in ENDPOINTS],
        "contrasts": [dict(item) for item in CONTRASTS],
        "output_files": dict(OUTPUT_FILES),
    }


def proposal_sha256() -> str:
    """Return the freeze digest of the proposal identity/defaults/rules."""
    return sha256_object(proposal_spec())


def proposal_freeze() -> dict[str, object]:
    """Return the proposal spec together with its freeze digest."""
    spec = proposal_spec()
    return {"proposal_sha256": sha256_object(spec), "spec": spec}


def assert_period_construction_label_blind() -> None:
    """Fail closed if period/exposure construction references annotations."""
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
        PeriodGrid._containing_calendar_day,
        PeriodGrid._containing_calendar_week,
        PeriodGrid._containing_calendar_month,
        PeriodGrid._containing_rolling_month,
        PeriodGrid._containing_anchored_block,
        PeriodGrid.period_containing,
        PeriodGrid.periods,
        build_period_records,
        build_pair_records,
        _tv_distance,
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


@dataclass(slots=True)
class PeriodRecord:
    """One constructed period (label-blind) with observable covariates."""

    history_id: str
    period: Period
    n_members: int
    n_straddling_members: int
    exposure_days: float
    coverage_state: str
    coverage_reason: str
    n_reset_edges: int
    member_ids: tuple[str, ...]
    member_starts: np.ndarray
    member_ends: np.ndarray
    robot_mix: tuple[int, ...]
    program_mix: tuple[int, ...]

    @property
    def support_state(self) -> str:
        return (
            "SUPPORTED"
            if self.n_members >= MIN_PERIOD_FILES
            else "INSUFFICIENT-PERIOD-SUPPORT"
        )

    @property
    def meets_support_floor(self) -> bool:
        return self.n_members >= MIN_PERIOD_FILES

    @property
    def member_ids_sha256(self) -> str:
        return _sha256_text("\n".join(sorted(self.member_ids)))


@dataclass(slots=True)
class PeriodPair:
    """One consecutive period pair (label-blind) with comparability covariates."""

    history_id: str
    previous: PeriodRecord
    current: PeriodRecord
    exposure_ratio: float | None
    mix_drift_tv: float | None
    n_reset_edges_crossed: int
    reference_files: int

    @property
    def coverage_state(self) -> str:
        both_full = (
            self.previous.coverage_state == "FULL" and self.current.coverage_state == "FULL"
        )
        return "PAIR-FULL" if both_full else "PAIR-PARTIAL"

    @property
    def support_state(self) -> str:
        both = self.previous.meets_support_floor and self.current.meets_support_floor
        return "PAIR-SUPPORTED" if both else "PAIR-INSUFFICIENT-PERIOD-SUPPORT"

    @property
    def exposure_state(self) -> str:
        if self.exposure_ratio is None:
            return "EXPOSURE-MISMATCH"
        matched = self.exposure_ratio >= PROPOSED_EXPOSURE_RATIO_FLOOR
        return "EXPOSURE-MATCHED" if matched else "EXPOSURE-MISMATCH"

    @property
    def comparison_state(self) -> str:
        if self.reference_files < MIN_REFERENCE_FILES:
            return "INSUFFICIENT-REFERENCE"
        if self.coverage_state != "PAIR-FULL":
            return "PARTIAL-PERIOD"
        if self.support_state != "PAIR-SUPPORTED":
            return "INSUFFICIENT-PERIOD-SUPPORT"
        if self.exposure_state != "EXPOSURE-MATCHED":
            return "COMPARABLE-UNMATCHED-EXPOSURE"
        return "COMPARABLE-MATCHED"


def _tv_distance(first: Sequence[float], second: Sequence[float]) -> float | None:
    """Return the total-variation distance between two count vectors."""
    array_first = np.asarray(first, dtype=np.float64)
    array_second = np.asarray(second, dtype=np.float64)
    total_first = float(array_first.sum())
    total_second = float(array_second.sum())
    if total_first == 0 or total_second == 0:
        return None
    return float(0.5 * np.abs(array_first / total_first - array_second / total_second).sum())


def reference_pool(meta: HistoryMetadata) -> dict[str, object]:
    """Return the inherited fixed causal reference pool summary (end_time <= c1)."""
    index = np.flatnonzero(meta.eligible & (meta.ends <= meta.c1))
    ids = sorted(meta.file_ids[int(position)] for position in index)
    robot_mix = np.bincount(meta.robot_index[index], minlength=len(meta.robot_ids))
    return {
        "history_id": meta.history_id,
        "reference_files": int(index.size),
        "reference_digest": sha256_text("\n".join(ids)),
        "reference_end_max": float(meta.ends[index].max()) if index.size else None,
        "robot_mix": [int(value) for value in robot_mix],
        "reference_state": (
            "REFERENCE-OK" if index.size >= MIN_REFERENCE_FILES else "INSUFFICIENT-REFERENCE"
        ),
    }


def _coverage_state(period: Period, analysis_start: float, analysis_end: float) -> tuple[str, str]:
    pre = period.start < analysis_start - 1e-9
    post = period.end > analysis_end + 1e-9
    if pre and post:
        return "PARTIAL-BOTH", "period extends beyond both analysis-span edges"
    if pre:
        return "PARTIAL-PRE", "period starts before the inherited c1 boundary"
    if post:
        return "PARTIAL-POST", "period is truncated by the history's tmax"
    return "FULL", ""


def build_period_records(meta: HistoryMetadata, grid: PeriodGrid) -> list[PeriodRecord]:
    """Construct period membership and exposure from observable metadata only.

    Membership candidates are eligible files with ``end_time > c1`` (the inherited
    analysis span), because pre-``c1`` rows are reference custody and must never
    enter an evaluation period.
    """
    periods = grid.periods()
    if not periods:
        return []
    ends = np.asarray([period.end for period in periods], dtype=np.float64)
    starts = np.asarray([period.start for period in periods], dtype=np.float64)
    candidate_mask = meta.eligible & (meta.ends > meta.c1) & (meta.ends <= meta.tmax)
    candidate_index = np.flatnonzero(candidate_mask)
    # Assignment: start < end_time <= end -> first period with end >= end_time.
    slot = np.searchsorted(ends, meta.ends[candidate_index], side="left")
    slot = np.clip(slot, 0, len(periods) - 1)
    reset_edges: list[float] = []
    for robot in meta.robot_ids:
        for span in meta.maintenance.get(robot, []):
            edge = float(span[1])
            if meta.c1 < edge <= meta.tmax:
                reset_edges.append(edge)
    reset_edges.sort()
    records: list[PeriodRecord] = []
    for position, period in enumerate(periods):
        members = candidate_index[slot == position]
        members = members[(meta.ends[members] > period.start) & (meta.ends[members] <= period.end)]
        member_starts = meta.starts[members]
        member_ends = meta.ends[members]
        overlap = np.clip(
            np.minimum(member_ends, period.end) - np.maximum(member_starts, period.start),
            0.0,
            None,
        )
        exposure_days = float(overlap.sum() / DAY_S)
        straddling = int(np.count_nonzero(member_starts <= period.start + 1e-9))
        robot_mix = np.bincount(
            meta.robot_index[members], minlength=len(meta.robot_ids)
        ).astype(int)
        program_mix = np.bincount(
            meta.program_index[members], minlength=len(meta.program_ids)
        ).astype(int)
        coverage_state, coverage_reason = _coverage_state(period, meta.c1, meta.tmax)
        n_reset = int(
            np.searchsorted(np.asarray(reset_edges), period.end, side="right")
            - np.searchsorted(np.asarray(reset_edges), period.start, side="right")
        )
        records.append(
            PeriodRecord(
                history_id=meta.history_id,
                period=period,
                n_members=int(members.size),
                n_straddling_members=straddling,
                exposure_days=exposure_days,
                coverage_state=coverage_state,
                coverage_reason=coverage_reason,
                n_reset_edges=n_reset,
                member_ids=tuple(meta.file_ids[int(index)] for index in members),
                member_starts=member_starts,
                member_ends=member_ends,
                robot_mix=tuple(int(value) for value in robot_mix),
                program_mix=tuple(int(value) for value in program_mix),
            )
        )
    return records


def build_pair_records(
    records: Sequence[PeriodRecord],
    meta: HistoryMetadata,
    reference_files: int,
) -> list[PeriodPair]:
    """Construct consecutive period pairs with exposure and mix covariates."""
    reset_edges = sorted(
        float(span[1])
        for robot in meta.robot_ids
        for span in meta.maintenance.get(robot, [])
        if meta.c1 < float(span[1]) <= meta.tmax
    )
    edge_array = np.asarray(reset_edges, dtype=np.float64)
    pairs: list[PeriodPair] = []
    for previous, current in zip(records, records[1:]):
        maximum = max(previous.exposure_days, current.exposure_days)
        ratio = None if maximum == 0 else min(
            previous.exposure_days, current.exposure_days
        ) / maximum
        drift = _tv_distance(previous.robot_mix, current.robot_mix)
        crossed = int(
            np.searchsorted(edge_array, current.period.end, side="right")
            - np.searchsorted(edge_array, previous.period.start, side="right")
        )
        pairs.append(
            PeriodPair(
                history_id=meta.history_id,
                previous=previous,
                current=current,
                exposure_ratio=ratio,
                mix_drift_tv=drift,
                n_reset_edges_crossed=crossed,
                reference_files=reference_files,
            )
        )
    return pairs


def project_annotations(manifest: Mapping[str, object]) -> dict[str, object]:
    """Project post-hoc annotations while preserving absent labels as ``None``.

    The inherited M1 projection stringifies every label, so a missing annotation
    would look like a non-normal label. This projection keeps absence explicit so
    an unannotated period can never be classified as an event or as healthy.
    """
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("manifest is missing its file rows")
    file_labels: dict[str, str | None] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest file row is not a mapping")
        raw = row.get("file_label")
        file_labels[str(row["file_id"])] = None if raw is None else str(raw)
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
    }


def posthoc_period(record: PeriodRecord, labels: Mapping[str, object]) -> dict[str, object]:
    """Annotate one period post-hoc; never feeds membership or support."""
    file_labels: Mapping[str, str | None] = labels["file_labels"]  # type: ignore[assignment]
    events: Sequence[Mapping[str, object]] = labels["failure_events"]  # type: ignore[assignment]
    onsets: Sequence[Mapping[str, object]] = labels["degradation_onsets"]  # type: ignore[assignment]
    abnormal = 0
    unknown = 0
    for file_id in record.member_ids:
        label = file_labels.get(file_id)
        if label is None:
            unknown += 1
        elif label != "normal":
            abnormal += 1
    start, end = record.period.start, record.period.end
    covered_events = sum(
        1 for event in events if start < float(event["failure_time"]) <= end
    )
    covered_onsets = sum(1 for onset in onsets if start < float(onset["onset"]) <= end)
    if unknown == 0 and record.n_members > 0:
        state = "ANNOTATED-CLEAN" if (abnormal == 0 and covered_events == 0) else "ANNOTATED-ABNORMAL"
    elif unknown == record.n_members:
        state = "UNANNOTATED"
    else:
        state = "UNANNOTATED-PARTIAL"
    return {
        "n_abnormal_members": abnormal,
        "n_unknown_label_members": unknown,
        "abnormal_member_fraction": _ratio(abnormal, record.n_members),
        "n_failure_events_covered": covered_events,
        "n_degradation_onsets_covered": covered_onsets,
        "annotation_state": state,
    }


def posthoc_pair(
    pair: PeriodPair,
    previous_annotations: Mapping[str, object],
    current_annotations: Mapping[str, object],
) -> dict[str, object]:
    """Annotate one pair post-hoc; never feeds comparability."""
    abnormal = int(previous_annotations["n_abnormal_members"]) + int(
        current_annotations["n_abnormal_members"]
    )
    members = pair.previous.n_members + pair.current.n_members
    events = int(previous_annotations["n_failure_events_covered"]) + int(
        current_annotations["n_failure_events_covered"]
    )
    onsets = int(previous_annotations["n_degradation_onsets_covered"]) + int(
        current_annotations["n_degradation_onsets_covered"]
    )
    states = (
        str(previous_annotations["annotation_state"]),
        str(current_annotations["annotation_state"]),
    )
    if any(state.startswith("UNANNOTATED") for state in states):
        attribution = "UNANNOTATED"
    elif int(current_annotations["n_failure_events_covered"]) > 0:
        attribution = "EVENT-CURRENT"
    elif int(previous_annotations["n_failure_events_covered"]) > 0:
        attribution = "EVENT-PREVIOUS"
    else:
        attribution = "NO-EVENT-ANNOTATED"
    return {
        "n_abnormal_members": abnormal,
        "abnormal_member_fraction": _ratio(abnormal, members),
        "n_failure_events_covered": events,
        "n_degradation_onsets_covered": onsets,
        "attribution_class": attribution,
    }


def period_row(record: PeriodRecord, annotations: Mapping[str, object]) -> dict[str, object]:
    """Merge one period's structural and post-hoc fields into the frozen row."""
    period = record.period
    return {
        "history_id": record.history_id,
        "horizon": period.horizon,
        "mode": period.mode,
        "period_key": period.key,
        "period_start_rel_s": period.start,
        "period_end_rel_s": period.end,
        "period_length_days": period.length_days,
        "period_start_iso": "",
        "period_end_iso": "",
        "n_members": record.n_members,
        "n_straddling_members": record.n_straddling_members,
        "exposure_days": record.exposure_days,
        "coverage_state": record.coverage_state,
        "coverage_reason": record.coverage_reason,
        "support_state": record.support_state,
        "meets_support_floor": record.meets_support_floor,
        "n_reset_edges": record.n_reset_edges,
        "member_ids_sha256": record.member_ids_sha256,
        **dict(annotations),
    }


def pair_row(
    pair: PeriodPair,
    previous_annotations: Mapping[str, object],
    current_annotations: Mapping[str, object],
    annotations: Mapping[str, object],
) -> dict[str, object]:
    """Merge one pair's structural and post-hoc fields into the frozen row."""
    return {
        "history_id": pair.history_id,
        "horizon": pair.current.period.horizon,
        "mode": pair.current.period.mode,
        "previous_key": pair.previous.period.key,
        "current_key": pair.current.period.key,
        "previous_start_rel_s": pair.previous.period.start,
        "current_end_rel_s": pair.current.period.end,
        "n_previous": pair.previous.n_members,
        "n_current": pair.current.n_members,
        "exposure_previous_days": pair.previous.exposure_days,
        "exposure_current_days": pair.current.exposure_days,
        "exposure_ratio": pair.exposure_ratio,
        "mix_drift_tv": pair.mix_drift_tv,
        "n_reset_edges_crossed": pair.n_reset_edges_crossed,
        "coverage_state": pair.coverage_state,
        "support_state": pair.support_state,
        "exposure_state": pair.exposure_state,
        "comparison_state": pair.comparison_state,
        "previous_annotation_state": str(previous_annotations["annotation_state"]),
        "current_annotation_state": str(current_annotations["annotation_state"]),
        **dict(annotations),
    }


def _period_iso(period: Period, calendar: CalendarConfig) -> tuple[str, str]:
    start = calendar.local(period.start).isoformat()
    end = calendar.local(period.end).isoformat()
    return start, end


def attach_iso(rows: list[dict[str, object]], periods: Sequence[Period], calendar: CalendarConfig) -> None:
    """Fill the declared-projection ISO labels on period rows (illustrative mapping)."""
    by_key = {period.key: period for period in periods}
    for row in rows:
        period = by_key[str(row["period_key"])]
        start_iso, end_iso = _period_iso(period, calendar)
        row["period_start_iso"] = start_iso
        row["period_end_iso"] = end_iso


def history_feasibility(
    history_id: str,
    horizon: str,
    mode: str,
    records: Sequence[PeriodRecord],
    pairs: Sequence[PeriodPair],
    reference: Mapping[str, object],
) -> dict[str, object]:
    """Summarize support/comparability for one history x horizon x mode."""
    support_sizes = [record.n_members for record in records]
    full = [record for record in records if record.coverage_state == "FULL"]
    full_sizes = [record.n_members for record in full]
    supported = [record for record in records if record.meets_support_floor]
    supported_pairs = [
        pair for pair in pairs if pair.comparison_state in ("COMPARABLE-MATCHED", "COMPARABLE-UNMATCHED-EXPOSURE")
    ]
    comparable_matched = [pair for pair in pairs if pair.comparison_state == "COMPARABLE-MATCHED"]
    comparable_unmatched = [
        pair for pair in pairs if pair.comparison_state == "COMPARABLE-UNMATCHED-EXPOSURE"
    ]
    lengths = [record.period.length_days for record in records]
    full_lengths = [record.period.length_days for record in full] or lengths
    return {
        "history_id": history_id,
        "horizon": horizon,
        "mode": mode,
        "n_periods": len(records),
        "n_full_periods": len(full),
        "n_partial_periods": len(records) - len(full),
        "n_supported_periods": len(supported),
        "min_period_members": min(support_sizes) if support_sizes else 0,
        "median_period_members": _quantile(support_sizes, 50),
        "max_period_members": max(support_sizes) if support_sizes else 0,
        "n_full_periods_ge_5": sum(1 for size in full_sizes if size >= 5),
        "n_full_periods_ge_10": sum(1 for size in full_sizes if size >= 10),
        "n_full_periods_ge_15": sum(1 for size in full_sizes if size >= 15),
        "n_full_periods_ge_20": sum(1 for size in full_sizes if size >= 20),
        "n_full_periods_ge_30": sum(1 for size in full_sizes if size >= 30),
        "n_full_periods_ge_100": sum(1 for size in full_sizes if size >= 100),
        "period_length_days_min": min(full_lengths) if full_lengths else None,
        "period_length_days_max": max(full_lengths) if full_lengths else None,
        "n_pairs": len(pairs),
        "n_pairs_full": sum(1 for pair in pairs if pair.coverage_state == "PAIR-FULL"),
        "n_pairs_supported_full": sum(
            1 for pair in pairs if pair.coverage_state == "PAIR-FULL" and pair.support_state == "PAIR-SUPPORTED"
        ),
        "n_pairs_comparable_matched": len(comparable_matched),
        "n_pairs_comparable_unmatched": len(comparable_unmatched),
        "median_exposure_ratio_supported": _quantile(
            [pair.exposure_ratio for pair in supported_pairs if pair.exposure_ratio is not None], 50
        ),
        "median_mix_drift_tv_supported": _quantile(
            [pair.mix_drift_tv for pair in supported_pairs if pair.mix_drift_tv is not None], 50
        ),
        "n_supported_pairs_zero_failure_events": 0,
        "n_supported_pairs_annotated_clean": 0,
        "n_periods_annotated_clean": 0,
        "reference_files": int(reference["reference_files"]),  # type: ignore[arg-type]
        "reference_state": str(reference["reference_state"]),
        "horizon_state": (
            "FEASIBLE-DESCRIPTIVE" if supported_pairs else "INFEASIBLE"
        ),
        "control_state": "NO-QUIET-PERIOD-CONTROL" if supported_pairs else "NO-SUPPORTED-PERIODS",
    }


def aggregate_horizon(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-history feasibility rows into the per-horizon matrix."""
    out: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for mode in MODES:
            subset = [
                row for row in rows if row["horizon"] == horizon and row["mode"] == mode
            ]
            if not subset:
                continue
            histories_with_pairs = sum(
                1 for row in subset if int(row["n_pairs_comparable_matched"]) + int(row["n_pairs_comparable_unmatched"]) > 0
            )
            if histories_with_pairs == len(subset):
                state = "FEASIBLE-DESCRIPTIVE"
            elif histories_with_pairs > 0:
                state = "PARTIALLY-FEASIBLE"
            else:
                state = "INFEASIBLE"
            zero_event = sum(int(row["n_supported_pairs_zero_failure_events"]) for row in subset)
            clean_pairs = sum(int(row["n_supported_pairs_annotated_clean"]) for row in subset)
            clean_periods = sum(int(row["n_periods_annotated_clean"]) for row in subset)
            supported_pairs = [
                int(row["n_pairs_comparable_matched"]) + int(row["n_pairs_comparable_unmatched"])
                for row in subset
            ]
            if clean_pairs > 0:
                control = "ANNOTATED-CLEAN-CONTROL-AVAILABLE"
            elif zero_event > 0:
                control = "QUIET-PERIOD-CONTROL-AVAILABLE"
            elif sum(int(row["n_supported_periods"]) for row in subset) > 0:
                control = "NO-QUIET-PERIOD-CONTROL"
            else:
                control = "NO-SUPPORTED-PERIODS"
            lengths_min = [row["period_length_days_min"] for row in subset if row["period_length_days_min"] is not None]
            lengths_max = [row["period_length_days_max"] for row in subset if row["period_length_days_max"] is not None]
            out.append(
                {
                    "horizon": horizon,
                    "mode": mode,
                    "n_histories": len(subset),
                    "n_periods_total": sum(int(row["n_periods"]) for row in subset),
                    "n_full_periods_total": sum(int(row["n_full_periods"]) for row in subset),
                    "n_supported_periods_total": sum(int(row["n_supported_periods"]) for row in subset),
                    "n_pairs_total": sum(int(row["n_pairs"]) for row in subset),
                    "n_pairs_comparable_matched_total": sum(
                        int(row["n_pairs_comparable_matched"]) for row in subset
                    ),
                    "n_pairs_comparable_unmatched_total": sum(
                        int(row["n_pairs_comparable_unmatched"]) for row in subset
                    ),
                    "n_supported_pairs_zero_failure_events_total": zero_event,
                    "n_supported_pairs_annotated_clean_total": clean_pairs,
                    "median_period_members": _quantile(
                        [float(row["median_period_members"]) for row in subset if row["median_period_members"] is not None], 50
                    ),
                    "min_period_members": min(int(row["min_period_members"]) for row in subset),
                    "max_period_members": max(int(row["max_period_members"]) for row in subset),
                    "median_supported_pairs_per_history": _quantile(
                        [float(value) for value in supported_pairs], 50
                    ),
                    "min_supported_pairs_per_history": min(supported_pairs),
                    "max_supported_pairs_per_history": max(supported_pairs),
                    "period_length_days_min": min(float(value) for value in lengths_min) if lengths_min else None,
                    "period_length_days_max": max(float(value) for value in lengths_max) if lengths_max else None,
                    "horizon_state": state,
                    "control_state": control,
                }
            )
    return out


@dataclass(slots=True)
class HistoryHorizonResult:
    """Per-history x horizon x mode construction result, kept inspectable."""

    history_id: str
    horizon: str
    mode: str
    grid: PeriodGrid
    records: list[PeriodRecord] = field(default_factory=list)
    pairs: list[PeriodPair] = field(default_factory=list)
    period_rows: list[dict[str, object]] = field(default_factory=list)
    pair_rows: list[dict[str, object]] = field(default_factory=list)
    feasibility: dict[str, object] = field(default_factory=dict)


def build_history_horizon(
    meta: HistoryMetadata,
    horizon: str,
    mode: str,
    calendar: CalendarConfig,
) -> HistoryHorizonResult:
    """Build one history x horizon x mode period/pair table and its annotations."""
    grid = PeriodGrid(
        horizon=horizon,
        mode=mode,
        calendar=calendar,
        analysis_start=meta.c1,
        analysis_end=meta.tmax,
    )
    reference = reference_pool(meta)
    records = build_period_records(meta, grid)
    pairs = build_pair_records(records, meta, int(reference["reference_files"]))  # type: ignore[arg-type]
    return HistoryHorizonResult(
        history_id=meta.history_id,
        horizon=horizon,
        mode=mode,
        grid=grid,
        records=records,
        pairs=pairs,
        feasibility=history_feasibility(
            meta.history_id, horizon, mode, records, pairs, reference
        ),
    )


def annotate_history_horizon(
    result: HistoryHorizonResult,
    manifest: Mapping[str, object],
    calendar: CalendarConfig,
) -> None:
    """Join post-hoc annotations to an already-constructed period/pair table."""
    labels = project_annotations(manifest)
    period_annotations = {
        record.period.key: posthoc_period(record, labels) for record in result.records
    }
    period_rows = [
        period_row(record, period_annotations[record.period.key]) for record in result.records
    ]
    attach_iso(period_rows, [record.period for record in result.records], calendar)
    pair_rows = []
    for pair in result.pairs:
        previous = period_annotations[pair.previous.period.key]
        current = period_annotations[pair.current.period.key]
        pair_rows.append(
            pair_row(pair, previous, current, posthoc_pair(pair, previous, current))
        )
    result.period_rows = period_rows
    result.pair_rows = pair_rows
    result.feasibility.update(_posthoc_feasibility(period_rows, pair_rows))
    return None


def _posthoc_feasibility(
    period_rows: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    supported_pairs = [
        row
        for row in pair_rows
        if row["comparison_state"]
        in ("COMPARABLE-MATCHED", "COMPARABLE-UNMATCHED-EXPOSURE")
    ]
    clean_pairs = sum(
        1
        for row in supported_pairs
        if row["previous_annotation_state"] == "ANNOTATED-CLEAN"
        and row["current_annotation_state"] == "ANNOTATED-CLEAN"
    )
    # A "quiet" pair must be fully annotated and cover no failure event. Absent
    # annotations yield attribution_class == UNANNOTATED, which never counts as
    # quiet: absence of annotation is not known-normal exposure.
    quiet_pairs = sum(
        1 for row in supported_pairs if row["attribution_class"] == "NO-EVENT-ANNOTATED"
    )
    return {
        "n_supported_pairs_zero_failure_events": quiet_pairs,
        "n_supported_pairs_annotated_clean": clean_pairs,
        "n_periods_annotated_clean": sum(
            1 for row in period_rows if row["annotation_state"] == "ANNOTATED-CLEAN"
        ),
    }


def evaluate(
    manifests: Mapping[str, Mapping[str, object]],
    edges: Mapping[str, Sequence[float]],
    calendar: CalendarConfig,
    modes: Sequence[str] = ("calendar",),
) -> dict[str, object]:
    """Run the metadata-only business-horizon feasibility pass.

    Construction for every history x horizon x mode happens before any annotation
    is projected, so labels cannot influence membership, exposure, support or
    comparability.
    """
    assert_period_construction_label_blind()
    results: list[HistoryHorizonResult] = []
    metas: dict[str, HistoryMetadata] = {}
    for history_id in sorted(manifests):
        if history_id not in edges:
            raise KeyError(f"missing frozen decile edges for {history_id}")
        edge_values = [float(value) for value in edges[history_id]]
        if len(edge_values) < 5:
            raise ValueError(f"decile edge table too short for {history_id}")
        observable = project_observable(manifests[history_id])
        meta = prepare_history(history_id, observable, edge_values[4])
        metas[history_id] = meta
        for mode in modes:
            for horizon in HORIZONS:
                results.append(build_history_horizon(meta, horizon, mode, calendar))
    for result in results:
        annotate_history_horizon(result, manifests[result.history_id], calendar)
    period_rows = [row for result in results for row in result.period_rows]
    pair_rows = [row for result in results for row in result.pair_rows]
    feasibility_rows = [result.feasibility for result in results]
    references = {
        history_id: reference_pool(meta) for history_id, meta in metas.items()
    }
    history_totals = {
        history_id: {
            "n_rows": len(meta.file_ids),
            "n_eligible": int(meta.eligible.sum()),
            "n_eligible_analysis_span": int(
                (meta.eligible & (meta.ends > meta.c1) & (meta.ends <= meta.tmax)).sum()
            ),
            "c1_rel_s": meta.c1,
            "tmax_rel_s": meta.tmax,
        }
        for history_id, meta in metas.items()
    }
    return {
        "proposal_id": PROPOSAL_ID,
        "proposal_sha256": proposal_sha256(),
        "calendar": calendar.describe(),
        "modes": list(modes),
        "period_rows": period_rows,
        "pair_rows": pair_rows,
        "feasibility_rows": feasibility_rows,
        "horizon_matrix": aggregate_horizon(feasibility_rows),
        "references": references,
        "history_totals": history_totals,
    }


def _period_example(
    record: PeriodRecord, calendar: CalendarConfig
) -> dict[str, object]:
    period = record.period
    start_iso, end_iso = _period_iso(period, calendar)
    return {
        "period_key": period.key,
        "period_start_rel_s": period.start,
        "period_end_rel_s": period.end,
        "period_start_iso": start_iso,
        "period_end_iso": end_iso,
        "period_length_days": period.length_days,
        "n_members": record.n_members,
        "n_straddling_members": record.n_straddling_members,
        "exposure_days": record.exposure_days,
        "coverage_state": record.coverage_state,
        "coverage_reason": record.coverage_reason,
        "support_state": record.support_state,
        "n_reset_edges": record.n_reset_edges,
    }


def _straddling_example(
    record: PeriodRecord, calendar: CalendarConfig
) -> dict[str, object] | None:
    period = record.period
    for position, file_id in enumerate(record.member_ids):
        start = float(record.member_starts[position])
        end = float(record.member_ends[position])
        if start <= period.start:
            overlap = min(end, period.end) - max(start, period.start)
            return {
                "file_id": file_id,
                "file_start_rel_s": start,
                "file_end_rel_s": end,
                "file_start_iso": calendar.local(start).isoformat(),
                "file_end_iso": calendar.local(end).isoformat(),
                "assigned_period_key": period.key,
                "period_start_rel_s": period.start,
                "assignment_check": "period_start < file_end_time <= period_end",
                "assignment_holds": bool(period.start < end <= period.end),
                "exposure_inside_period_s": float(overlap),
                "exposure_inside_period_days": float(overlap / DAY_S),
                "straddles_period_start": True,
            }
    return None


def _boundary_assignment_example(
    record: PeriodRecord, calendar: CalendarConfig
) -> dict[str, object] | None:
    period = record.period
    if not record.member_ids:
        return None
    position = 0
    start = float(record.member_starts[position])
    end = float(record.member_ends[position])
    return {
        "file_id": record.member_ids[position],
        "file_start_rel_s": start,
        "file_end_rel_s": end,
        "file_end_iso": calendar.local(end).isoformat(),
        "assigned_period_key": period.key,
        "period_start_rel_s": period.start,
        "period_end_rel_s": period.end,
        "period_start_iso": calendar.local(period.start).isoformat(),
        "period_end_iso": calendar.local(period.end).isoformat(),
        "inequality": "period_start < file_end_rel_s <= period_end",
        "inequality_holds": bool(period.start < end <= period.end),
        "exposure_inside_period_days": float(
            (min(end, period.end) - max(start, period.start)) / DAY_S
        ),
    }


def worked_examples(
    manifests: Mapping[str, Mapping[str, object]],
    edges: Mapping[str, Sequence[float]],
    calendar: CalendarConfig,
    history_id: str | None = None,
) -> dict[str, object]:
    """Build reproducible real-metadata membership/exposure examples.

    Every value here comes from an accepted Development manifest and its frozen
    decile edges. The examples are structural (membership, exposure, support,
    calendar lengths); they contain no embedding, score or model outcome.
    """
    chosen = history_id or sorted(manifests)[0]
    observable = project_observable(manifests[chosen])
    meta = prepare_history(chosen, observable, [float(value) for value in edges[chosen]][4])
    reference = reference_pool(meta)
    out: dict[str, object] = {
        "label": "REAL-ACCEPTED-DEVELOPMENT-METADATA-WORKED-EXAMPLES",
        "history_id": chosen,
        "analysis_span": {
            "c1_rel_s": meta.c1,
            "tmax_rel_s": meta.tmax,
            "c1_iso": calendar.local(meta.c1).isoformat(),
            "tmax_iso": calendar.local(meta.tmax).isoformat(),
            "c1_rule": "inherited 40 percent decile edge (edges[4])",
            "tmax_rule": "history maximum file end_time",
        },
        "reference": reference,
        "horizons": {},
    }
    horizons: dict[str, object] = {}
    for horizon in HORIZONS:
        grid = PeriodGrid(horizon, "calendar", calendar, meta.c1, meta.tmax)
        records = build_period_records(meta, grid)
        pairs = build_pair_records(records, meta, int(reference["reference_files"]))  # type: ignore[arg-type]
        full = [record for record in records if record.coverage_state == "FULL"]
        supported = [record for record in records if record.meets_support_floor]
        largest = max(records, key=lambda record: record.n_members) if records else None
        straddling_record = next(
            (record for record in records if record.n_straddling_members > 0), None
        )
        matched_pairs = [pair for pair in pairs if pair.comparison_state == "COMPARABLE-MATCHED"]
        median_pair = None
        if matched_pairs:
            ratios = sorted(
                (
                    (pair, pair.exposure_ratio if pair.exposure_ratio is not None else 0.0)
                    for pair in matched_pairs
                ),
                key=lambda item: item[1],
            )
            median_pair = ratios[len(ratios) // 2][0]
        entry: dict[str, object] = {
            "n_periods": len(records),
            "n_full_periods": len(full),
            "n_supported_periods": len(supported),
            "n_pairs": len(pairs),
            "n_pairs_comparable_matched": len(matched_pairs),
            "first_period": _period_example(records[0], calendar) if records else None,
            "last_period": _period_example(records[-1], calendar) if records else None,
            "largest_period": (
                _period_example(largest, calendar) if largest is not None else None
            ),
            "largest_period_support_floor": MIN_PERIOD_FILES,
            "largest_period_meets_floor": bool(
                largest is not None and largest.meets_support_floor
            ),
            "straddling_period_key": (
                straddling_record.period.key if straddling_record is not None else None
            ),
            "straddling_example": (
                _straddling_example(straddling_record, calendar)
                if straddling_record is not None
                else None
            ),
            "boundary_assignment_example": (
                _boundary_assignment_example(full[0], calendar) if full else None
            ),
            "median_exposure_matched_pair": (
                {
                    "previous_key": median_pair.previous.period.key,
                    "current_key": median_pair.current.period.key,
                    "previous_length_days": median_pair.previous.period.length_days,
                    "current_length_days": median_pair.current.period.length_days,
                    "n_previous": median_pair.previous.n_members,
                    "n_current": median_pair.current.n_members,
                    "exposure_previous_days": median_pair.previous.exposure_days,
                    "exposure_current_days": median_pair.current.exposure_days,
                    "exposure_ratio": median_pair.exposure_ratio,
                    "exposure_ratio_floor": PROPOSED_EXPOSURE_RATIO_FLOOR,
                    "mix_drift_tv": median_pair.mix_drift_tv,
                    "n_reset_edges_crossed": median_pair.n_reset_edges_crossed,
                }
                if median_pair is not None
                else None
            ),
            "pair_comparison_state_counts": dict(
                Counter(pair.comparison_state for pair in pairs)
            ),
        }
        if horizon == "month":
            entry["calendar_month_sequence"] = [
                {
                    "period_key": record.period.key,
                    "length_days": record.period.length_days,
                    "n_members": record.n_members,
                    "exposure_days": record.exposure_days,
                    "coverage_state": record.coverage_state,
                    "support_state": record.support_state,
                }
                for record in records
            ]
            adjacent = [
                (
                    previous,
                    current,
                )
                for previous, current in zip(records, records[1:])
                if previous.coverage_state == "FULL" and current.coverage_state == "FULL"
            ]
            entry["unequal_adjacent_month_pair"] = (
                {
                    "previous_key": adjacent[0][0].period.key,
                    "previous_length_days": adjacent[0][0].period.length_days,
                    "current_key": adjacent[0][1].period.key,
                    "current_length_days": adjacent[0][1].period.length_days,
                    "note": "adjacent months keep their real calendar lengths; no nominal month substitute",
                }
                if adjacent
                else None
            )
        horizons[horizon] = entry
    out["horizons"] = horizons
    return out


def illustrative_calendar_examples(calendar: CalendarConfig) -> dict[str, object]:
    """Return synthetic boundary cases, explicitly labelled illustrative.

    These cases exercise the boundary arithmetic only. They are not accepted
    corpus evidence and no support statistic is derived from them.
    """
    examples: list[dict[str, object]] = []
    # Right-inclusive end_time boundary: an end exactly on a day boundary belongs
    # to the period that ends there.
    day_grid = PeriodGrid("day", "calendar", calendar, 0.0, 10.0 * DAY_S)
    examples.append(
        {
            "case": "end_time_exactly_on_day_boundary",
            "rule": "start < end_time <= end assigns the boundary instant to the earlier period",
            "boundary_rel_s": DAY_S,
            "assigned_period_key": day_grid.period_for_end_time(DAY_S).key,
            "period_ending_at_boundary_key": day_grid.period_containing(DAY_S - 1.0).key,
            "period_starting_at_boundary_key": day_grid.period_containing(DAY_S).key,
            "assigned_to_earlier_period": day_grid.period_for_end_time(DAY_S).key
            == day_grid.period_containing(DAY_S - 1.0).key,
        }
    )
    # Leap February and unequal calendar month lengths, pure calendar arithmetic.
    lengths = []
    for year, month in ((2024, 1), (2024, 2), (2024, 3), (2025, 2), (2024, 6), (2024, 7)):
        probe = calendar.relative(calendar.local_midnight(year, month, 15))
        period = day_grid.period_containing(probe)
        month_period = PeriodGrid("month", "calendar", calendar, 0.0, 1.0).period_containing(probe)
        lengths.append(
            {
                "month": f"{year:04d}-{month:02d}",
                "length_days": month_period.length_days,
                "period_key": month_period.key,
                "probe_day_period_key": period.key,
            }
        )
    examples.append({"case": "calendar_month_lengths", "months": lengths})
    # Rolling month preserves real inter-anchor gaps instead of a nominal month.
    rolling = PeriodGrid("month", "rolling", calendar, 0.0, 1.0)
    rolling_lengths = []
    for year, month in ((2024, 1), (2024, 2), (2024, 3), (2024, 6), (2024, 7)):
        anchor = rolling._rolling_month_anchor(year, month)  # noqa: SLF001 - documented helper
        relative_anchor = calendar.relative(anchor)
        period = rolling.period_containing(relative_anchor + 1.0)
        rolling_lengths.append(
            {
                "month": f"{year:04d}-{month:02d}",
                "period_key": period.key,
                "rolling_length_days": period.length_days,
            }
        )
    examples.append(
        {
            "case": "rolling_month_unequal_lengths",
            "note": "lengths are real inter-anchor gaps; a nominal 28- or 30-day month is never substituted",
            "months": rolling_lengths,
        }
    )
    # Week-start sensitivity: Monday vs Sunday week containing the same instant.
    monday_grid = PeriodGrid("week", "calendar", calendar, 0.0, 10.0 * DAY_S)
    sunday_calendar = CalendarConfig(
        anchor_epoch_s=calendar.anchor_epoch_s,
        timezone=calendar.timezone,
        week_start="sunday",
        month_anchor_day=calendar.month_anchor_day,
    )
    sunday_grid = PeriodGrid("week", "calendar", sunday_calendar, 0.0, 10.0 * DAY_S)
    probe = 4.5 * DAY_S
    examples.append(
        {
            "case": "week_start_sensitivity",
            "probe_rel_s": probe,
            "monday_week_key": monday_grid.period_containing(probe).key,
            "sunday_week_key": sunday_grid.period_containing(probe).key,
            "same_instant_different_key": monday_grid.period_containing(probe).key
            != sunday_grid.period_containing(probe).key,
        }
    )
    return {
        "label": "ILLUSTRATIVE-SYNTHETIC-CALENDAR-EXAMPLES-NOT-CORPUS-EVIDENCE",
        "calendar": calendar.describe(),
        "examples": examples,
    }


def calendar_sensitivity(
    manifests: Mapping[str, Mapping[str, object]],
    edges: Mapping[str, Sequence[float]],
    variants: Mapping[str, CalendarConfig],
    modes: Sequence[str] = MODES,
) -> dict[str, object]:
    """Compare feasibility counts across declared calendar projections.

    Every variant is a declared assumption probe, not a business recommendation;
    the first mapping entry is the reference variant for the reported deltas.
    """
    if not variants:
        raise ValueError("at least one calendar variant is required")
    runs: dict[str, object] = {}
    for label, calendar in variants.items():
        result = evaluate(manifests, edges, calendar, modes=modes)
        runs[label] = {
            "calendar": calendar.describe(),
            "matrix": {
                f"{row['horizon']}|{row['mode']}": {
                    key: row[key]
                    for key in (
                        "n_periods_total",
                        "n_full_periods_total",
                        "n_supported_periods_total",
                        "n_pairs_total",
                        "n_pairs_comparable_matched_total",
                        "n_pairs_comparable_unmatched_total",
                        "n_supported_pairs_zero_failure_events_total",
                        "n_supported_pairs_annotated_clean_total",
                        "horizon_state",
                        "control_state",
                    )
                }
                for row in result["horizon_matrix"]  # type: ignore[index]
            },
        }
    baseline_label = next(iter(variants))
    deltas: dict[str, object] = {}
    baseline_matrix = runs[baseline_label]["matrix"]  # type: ignore[index]
    for label, run in runs.items():
        if label == baseline_label:
            continue
        matrix = run["matrix"]  # type: ignore[index]
        deltas[label] = {
            cell: {
                field: int(matrix[cell][field]) - int(baseline_matrix[cell][field])
                for field in (
                    "n_periods_total",
                    "n_supported_periods_total",
                    "n_pairs_total",
                    "n_pairs_comparable_matched_total",
                    "n_pairs_comparable_unmatched_total",
                )
            }
            for cell in baseline_matrix
        }
    return {
        "variants": list(variants),
        "baseline": baseline_label,
        "note": "projection probes only; no timezone, week start or anchor is a verified business fact",
        "runs": runs,
        "deltas_vs_baseline": deltas,
    }
