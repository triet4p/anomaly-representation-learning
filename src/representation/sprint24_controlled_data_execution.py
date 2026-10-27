"""Sprint 24 Task 7-M8 controlled-data execution readiness (v2 calendar).

Implements the missing burden-schedule planner, multi-history calendar builder
and duration/roster controls against the M7 design, under the owner-authorized
bounded v2 calendar/bands amendment (30-week span, 7 monthly offsets, second
elevated month, remapped weeks). Reuses real synthesis machinery
(:mod:`synth.generator`); no second generator, no stubs, no mocks.

Binding boundaries (unchanged from the M7 scientific contract):

- No official corpus generation, no embedding/model/forward, no inference, no
  training, no reference-bank build, no business-density scale, no Confirmation
  or Phase C execution. The only waveforms ever produced here are a tiny
  clearly-marked throwaway smoke subset through the SAME slot-plan and
  generation path (never a separate small-mode semantic path).
- M7 ``S24-CONTROLLED-DATA-v1`` bytes and digest are preserved as historical;
  this module never imports-mutates them. Thresholds, estimands, lineage rules
  and custody are unchanged; only the calendar/band layout, counts and cost
  move under the explicit v2 amendment.
- Shortfalls refuse fail-fast with closed reasons: no retries, no reseeding,
  no threshold/band rescue, no silent rescaling. Generation output never
  overwrites.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np

from representation import sprint24_controlled_data_protocol as v1
from representation import sprint24_business_horizon as horizon
from representation.sprint24_context_audit import sha256_object
from representation.sprint24_weekly_monthly_study import primary_calendar
from synth.anomalies.registry import HARD_FAMILIES
from synth.regimes import sample_regime_sequence
from synth.config import GENERATOR_VERSION, SynthConfig
from synth.generator import SessionGenerator
from synth.schema import (
    DIAGNOSTIC_ONLY_FIELD_NAMES,
    MODEL_INPUT_FIELD_NAMES,
    AnomalyFamily,
    FactoryProvenance,
    FileSample,
    RegimeType,
    SampleLabel,
)

#: Versioned execution protocol identifier frozen by this module.
PROTOCOL_ID = "S24-CONTROLLED-DATA-v2"

#: Base design this execution amendment implements (bytes preserved, never edited).
BASE_PROTOCOL_ID = v1.PROTOCOL_ID
BASE_PROTOCOL_SHA256 = v1.protocol_sha256()

#: Planned weekly span: W00 warmup + 29 analysis weeks.
N_PLANNED_WEEKS = 30

#: Planned monthly span: M01..M07 derived from the same files by end_time.
N_PLANNED_MONTHS = 7

#: Shared projected-calendar origin per history (relative clocks are
#: independent; the projection anchor is shared so nesting is checkable once).
C1_REL_S = 0.0

#: Closed weekly band vocabulary (annotation layer, never structural).
WEEK_BANDS = v1.WEEK_BANDS

#: Closed monthly band vocabulary.
MONTH_BANDS = v1.MONTH_BANDS

#: Default week-resolution band plan (offsets from c1; W00 warmup excluded from
#: every arm; every arm week sits fully inside a same-band month on the real
#: calendar at c1 = 0, verified file-level by ``validate_week_month_nesting``).
DEFAULT_WEEK_BANDS: dict[int, str] = {
    0: "WARMUP",
    1: "BUFFER", 2: "BUFFER", 3: "BUFFER", 4: "BUFFER",
    5: "BASELINE-A", 6: "BASELINE-A", 7: "BASELINE-A",
    8: "BUFFER", 9: "BUFFER", 10: "BUFFER", 11: "BUFFER", 12: "BUFFER",
    13: "ELEVATED-A", 14: "ELEVATED-A",
    15: "BUFFER", 16: "BUFFER", 17: "BUFFER",
    18: "BASELINE-B", 19: "BASELINE-B", 20: "BASELINE-B",
    21: "BUFFER", 22: "BUFFER", 23: "BUFFER", 24: "BUFFER", 25: "BUFFER",
    26: "ELEVATED-B", 27: "ELEVATED-B",
    28: "BUFFER", 29: "BUFFER",
}

#: Default month-resolution band plan. Two disjoint LOW runs plus two
#: single-month elevated contrasts (the v2 amendment's second elevated month).
DEFAULT_MONTH_BANDS: dict[int, str] = {
    1: "WARMUP-PARTIAL",
    2: "LOW",
    3: "LOW",
    4: "ELEVATED",
    5: "LOW",
    6: "LOW",
    7: "ELEVATED",
}

#: Planned per-file abnormal quotas (deterministic counts per 40-file week).
BASELINE_ABNORMAL_PER_WEEK = 2
ELEVATED_ABNORMAL_PER_WEEK = 11
REFERENCE_ABNORMAL_TOTAL = 11

#: Deterministic abnormal severity band for planned abnormal slots (inside the
#: generator default sampling range, away from its edges).
SEVERITY_LO = 0.35
SEVERITY_HI = 0.85

#: File split tag recorded in planned file identities (lineage marker only).
PLANNED_SPLIT = "controlled"

#: Smoke stream tag: smoke seeds derive from a disjoint tagged stream and
#: smoke records carry an explicit throwaway marker (never official corpus).
SMOKE_STREAM_TAG = "SMOKE-THROWAWAY"

#: Bounded smoke envelope in files (readiness authorizes at most this many
#: throwaway waveforms; the smoke run itself reports the exact count used).
SMOKE_ENVELOPE_FILES = 96

#: Closed refusal reason for planned slots the real generator cannot realize
#: on the exact planned seed (legitimate strength-gate rejection). Distinct
#: from plan-level ``REFUSED-SHORTFALL``: it records a generator-capability
#: limit, never triggers retry/reseed/fallback/rescue, and needs an owner
#: decision before the plan can change.
REFUSED_GENERATION = "REFUSED-GENERATION"


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slot_seed(history_seed: int, stream: str, week_key: str, slot: int) -> int:
    """Derive a deterministic unique per-slot generator seed."""
    digest = _sha256_hex(f"v2:{history_seed}:{stream}:{week_key}:{slot:04d}")
    return int(digest[:15], 16) % (2**31)


def slot_order_key(history_id: str, week_key: str, slot: int) -> str:
    """Return the label-blind hash order key for quota selection in one week."""
    return _sha256_hex(f"{v1.NULL_SEED}:{history_id}:{week_key}:{slot:04d}")


def slot_family(history_seed: int, week_key: str, slot: int) -> AnomalyFamily:
    """Rotate deterministically over all nine hard anomaly families.

    The rotation covers every family in ``HARD_FAMILIES``; any legitimate
    strength-gate rejection on an exact planned (family, severity, seed)
    triple fails fast at generation time with no retry, reseed or rescue.
    Family exclusions or substitutions need an owner-authorized amendment.
    """
    digest = _sha256_hex(f"v2fam:{history_seed}:{week_key}:{slot:04d}")
    return HARD_FAMILIES[int(digest[:8], 16) % len(HARD_FAMILIES)]


#: Owner-authorized prospective family reassignment (bounded plan amendment
#: for exactly 4 semantically impossible ``missing_event`` slots on
#: edge-only 2-regime sessions, which have no interior event to miss).
#: Rule: any abnormal slot whose requested family is ``missing_event`` AND
#: whose planned base session carries no interior missing-event candidate
#: (fewer than 3 regimes, or only edge regimes) is planned as
#: ``subtle_drift`` — a within-regime trajectory deviation requiring no
#: interior event — at the SAME seed, severity, week/month/arm, quota and
#: stream. Seeds, identities, quotas, gates and all other slots are unchanged;
#: the original ``missing_event`` request and the feasibility reason are
#: retained in the slot record. Any extension beyond these slots needs a new
#: owner authorization.
REASSIGNED_FAMILY: AnomalyFamily = AnomalyFamily.SUBTLE_DRIFT



def slot_severity(history_seed: int, week_key: str, slot: int) -> float:
    """Return a deterministic abnormal severity inside the planned band."""
    digest = _sha256_hex(f"v2sev:{history_seed}:{week_key}:{slot:04d}")
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return SEVERITY_LO + (SEVERITY_HI - SEVERITY_LO) * unit


def predict_normal_roster(seed: int, n_robots: int = 5, n_programs: int = 8) -> tuple[int, int]:
    """Replicate :meth:`NormalGenerator.generate` roster derivation exactly."""
    rng = np.random.default_rng(seed)
    return (int(rng.integers(0, n_robots)), int(rng.integers(0, n_programs)))


def predict_anomaly_roster(seed: int, n_robots: int = 5, n_programs: int = 8) -> tuple[int, int]:
    """Replicate :meth:`SessionGenerator.generate_anomaly` roster derivation."""
    rng_base = np.random.default_rng(seed)
    normal_seed = int(rng_base.integers(0, 2**31))
    rng_normal = np.random.default_rng(normal_seed)
    return (int(rng_normal.integers(0, n_robots)), int(rng_normal.integers(0, n_programs)))


def _refused(reason: str, detail: str) -> ValueError:
    return ValueError(f"{reason}: {detail}")


def require_fresh_dir(path: object, *, purpose: str) -> None:
    """Refuse to write into an existing non-empty output directory."""
    from pathlib import Path

    candidate = Path(str(path))
    if candidate.is_dir() and any(candidate.iterdir()):
        raise _refused("REFUSED-OVERWRITE", f"{purpose} target {candidate} already holds files")
    candidate.mkdir(parents=True, exist_ok=True)


# --- label-blind planned-identity constructors --------------------------------

#: Names that must never appear in planned-identity construction code.
_FORBIDDEN_IDENTITY_TOKENS = v1._FORBIDDEN_PLANNING_TOKENS  # noqa: SLF001 (reuse, not mutation)


def planned_week_identities(history_id: str) -> list[str]:
    """Return the label-blind planned weekly period identities W00..W29."""
    return [f"{history_id}|week|calendar|W{k:02d}" for k in range(N_PLANNED_WEEKS)]


def planned_month_identities(history_id: str) -> list[str]:
    """Return the label-blind planned monthly period identities M01..M07."""
    return [f"{history_id}|month|calendar|M{k:02d}" for k in range(1, N_PLANNED_MONTHS + 1)]


def _identity_construction_functions() -> tuple[Callable[..., object], ...]:
    return (
        planned_week_identities,
        planned_month_identities,
        v1.planned_reference_identity,
        v1.order_planned_identities,
    )


def assert_execution_construction_label_blind() -> None:
    """Fail closed if planned-identity construction references annotations."""
    for fn in _identity_construction_functions():
        source = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                body: list[ast.stmt] = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    node.body = body[1:] or [ast.Pass()]  # node.body typed list[stmt]
        code = ast.unparse(tree).lower()
        for token in cast("tuple[str, ...]", _FORBIDDEN_IDENTITY_TOKENS):
            if token in code:
                raise RuntimeError(f"label-blind violation: {getattr(fn, '__name__', str(fn))} references {token!r}")


# --- multi-history calendar builder -------------------------------------------


def build_history_calendar(
    history_id: str,
    c1_rel_s: float = C1_REL_S,
    n_weekly: int = v1.PLANNED_WEEKLY_FILES,
) -> dict[str, object]:
    """Build the real week/month period grids for one planned history.

    Raises a fail-fast refusal when the span does not enumerate exactly 30
    weekly and 7 monthly periods (density/count incompatibility is never
    silently rescaled).
    """
    calendar = primary_calendar()
    span_s = float(N_PLANNED_WEEKS) * 7.0 * horizon.DAY_S
    tmax_rel_s = float(c1_rel_s) + span_s
    week_grid = horizon.PeriodGrid(
        horizon="week", mode="calendar", calendar=calendar,
        analysis_start=float(c1_rel_s), analysis_end=tmax_rel_s,
    )
    month_grid = horizon.PeriodGrid(
        horizon="month", mode="calendar", calendar=calendar,
        analysis_start=float(c1_rel_s), analysis_end=tmax_rel_s,
    )
    weeks = list(week_grid.periods())
    months = list(month_grid.periods())
    if len(weeks) != N_PLANNED_WEEKS or len(months) != N_PLANNED_MONTHS:
        raise _refused(
            "REFUSED-SHORTFALL",
            f"{history_id}: span enumerates {len(weeks)} weeks/{len(months)} months, "
            f"planned {N_PLANNED_WEEKS}/{N_PLANNED_MONTHS}; re-plan, never rescale",
        )
    week_rows = [
        {
            "offset": offset,
            "period_key": period.key,
            "start_rel_s": float(period.start),
            "end_rel_s": float(period.end),
            "planned_identity": f"{history_id}|week|calendar|W{offset:02d}",
            "band": str(DEFAULT_WEEK_BANDS[offset]),
        }
        for offset, period in enumerate(weeks)
    ]
    month_rows = [
        {
            "index": index,
            "period_key": period.key,
            "start_rel_s": float(period.start),
            "end_rel_s": float(period.end),
            "planned_identity": f"{history_id}|month|calendar|M{index:02d}",
            "band": str(DEFAULT_MONTH_BANDS[index]),
        }
        for index, period in enumerate(months, start=1)
    ]
    return {
        "history_id": history_id,
        "c1_rel_s": float(c1_rel_s),
        "tmax_rel_s": tmax_rel_s,
        "calendar_describe": calendar.describe(),
        "weeks": week_rows,
        "months": month_rows,
        "n_weekly": int(n_weekly),
    }

def _slot_str(slot: Mapping[str, object], key: str) -> str:
    """Return a required string slot field, failing closed on wrong types."""
    value = slot[key]
    if not isinstance(value, str):
        raise _refused("REFUSED-IDENTITY", f"{slot.get('slot_id', '?')}: field {key!r} must be str")
    return value


def _slot_float(slot: Mapping[str, object], key: str) -> float:
    """Return a required float slot field, failing closed on wrong types."""
    value = slot[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refused("REFUSED-IDENTITY", f"{slot.get('slot_id', '?')}: field {key!r} must be numeric")
    return float(cast("int | float", value))


def _slot_int(slot: Mapping[str, object], key: str) -> int:
    """Return a required int slot field, failing closed on wrong types."""
    value = slot[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _refused("REFUSED-IDENTITY", f"{slot.get('slot_id', '?')}: field {key!r} must be int")
    return int(value)


def _row_int(row: Mapping[str, object], key: str) -> int:
    """Return a required int grid field, failing closed on wrong types."""
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _refused("REFUSED-IDENTITY", f"grid field {key!r} must be int")
    return int(value)


def _row_float(row: Mapping[str, object], key: str) -> float:
    """Return a required float grid field, failing closed on wrong types."""
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refused("REFUSED-IDENTITY", f"grid field {key!r} must be numeric")
    return float(cast("int | float", value))


def _row_str(row: Mapping[str, object], key: str) -> str:
    """Return a required string grid field, failing closed on wrong types."""
    value = row[key]
    if not isinstance(value, str):
        raise _refused("REFUSED-IDENTITY", f"grid field {key!r} must be str")
    return value


def _freeze_float(freeze: Mapping[str, object], key: str) -> float:
    """Return a required float freeze field, failing closed on wrong types."""
    value = freeze[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refused("REFUSED-IDENTITY", f"freeze field {key!r} must be numeric")
    return float(cast("int | float", value))


def month_index_of(months: Sequence[Mapping[str, object]], end_time: float) -> int | None:
    """Return the 1-based month index owning ``end_time`` (membership rule)."""
    value = float(end_time)
    for row in months:
        if _row_float(row, "start_rel_s") < value <= _row_float(row, "end_rel_s"):
            return _row_int(row, "index")
    return None


def week_end_times(week: Mapping[str, object], n_weekly: int) -> list[float]:
    """Spread ``n_weekly`` file end_times uniformly inside one week period."""
    start = _row_float(week, "start_rel_s")
    length = _row_float(week, "end_rel_s") - start
    return [start + (slot + 1) * length / n_weekly for slot in range(n_weekly)]


# --- burden-schedule planner ----------------------------------------------------


def _abnormal_count(band: str, month_band: str | None) -> int:
    if band == "BASELINE-A" or band == "BASELINE-B":
        return BASELINE_ABNORMAL_PER_WEEK
    if band == "ELEVATED-A" or band == "ELEVATED-B":
        return ELEVATED_ABNORMAL_PER_WEEK
    if month_band == "ELEVATED":
        return ELEVATED_ABNORMAL_PER_WEEK
    return BASELINE_ABNORMAL_PER_WEEK


def _majority_month_band(
    months: Sequence[Mapping[str, object]], ends: Sequence[float]
) -> str | None:
    counts: dict[str, int] = {}
    band_by_index = {_row_int(row, "index"): _row_str(row, "band") for row in months}
    for end in ends:
        index = month_index_of(months, float(end))
        if index is None:
            continue
        band = band_by_index[index]
        counts[band] = counts.get(band, 0) + 1
    if not counts:
        return None
    return sorted(counts, key=lambda band: (-counts[band], band))[0]


def _needs_reassignment(history_seed: int, week_key: str, slot: int, family: AnomalyFamily) -> bool:
    """Return whether one abnormal slot meets the authorized reassignment rule.

    Deterministic, prospective, and general (not a per-slot allowlist): the
    slot must request ``missing_event`` AND its planned base session must
    carry no interior missing-event candidate (fewer than 3 regimes, or only
    edge regimes). The check replays only the label-blind roster/regime draw
    prefix (roster integers + regime sequence), never labels or waveforms.
    """
    if family != AnomalyFamily.MISSING_EVENT:
        return False
    seed = slot_seed(int(history_seed), "plan", week_key, slot)
    rng_base = np.random.default_rng(seed)
    normal_seed = int(rng_base.integers(0, 2**31))
    rng_normal = np.random.default_rng(normal_seed)
    cfg_replay = SynthConfig()
    fleet_replay = cfg_replay.fleet
    physics_replay = cfg_replay.physics
    channels_replay = cfg_replay.n_channels
    _roster = (int(rng_normal.integers(0, fleet_replay.n_robots)), int(rng_normal.integers(0, fleet_replay.n_programs)))
    _gain = 1.0 + rng_normal.uniform(
        -physics_replay.channel_gain_variation, physics_replay.channel_gain_variation, channels_replay
    )
    _offset = rng_normal.uniform(
        -physics_replay.channel_offset_variation, physics_replay.channel_offset_variation, channels_replay
    )
    seq = sample_regime_sequence(rng_normal, cfg_replay.regime)
    interior = [
        i
        for i, (regime, duration) in enumerate(seq)
        if regime in (RegimeType.RAMP_UP, RegimeType.ACTIVE, RegimeType.PERIODIC, RegimeType.RECOVERY)
        and duration >= 15
        and 0 < i < len(seq) - 1
    ]
    return not interior


def plan_history(
    history_id: str,
    history_seed: int,
    used_seeds: Sequence[int],
    stream: str = "plan",
    n_reference: int = v1.PLANNED_REFERENCE_FILES,
    n_weekly: int = v1.PLANNED_WEEKLY_FILES,
    week_bands: Mapping[int, str] = DEFAULT_WEEK_BANDS,
    month_bands: Mapping[int, str] = DEFAULT_MONTH_BANDS,
) -> dict[str, object]:
    """Plan every file slot of one history deterministically (metadata only).

    Assigns requested abnormal/normal outcomes from the burden bands, rotates
    anomaly families with explicit severities, predicts observable rosters and
    validates quotas, nesting, roster support and seed/lineage custody.
    Any shortfall raises a fail-fast refusal; nothing is rescued or retried.
    """
    lineage = v1.validate_lineage(history_id)
    if str(lineage["status"]) == "REFUSED":
        raise _refused(str(lineage["reason"]), str(lineage["detail"]))
    seed_check = v1.validate_seeds(int(history_seed), list(used_seeds))
    if str(seed_check["status"]) == "REFUSED":
        raise _refused(str(seed_check["reason"]), str(seed_check["detail"]))
    calendar = build_history_calendar(history_id, n_weekly=n_weekly)
    weeks = list(cast("list[dict[str, object]]", calendar["weeks"]))
    months = list(cast("list[dict[str, object]]", calendar["months"]))
    if sorted(week_bands) != list(range(N_PLANNED_WEEKS)):
        raise _refused("REFUSED-SHORTFALL", f"{history_id}: week bands must cover offsets 0..{N_PLANNED_WEEKS - 1}")
    if sorted(month_bands) != list(range(1, N_PLANNED_MONTHS + 1)):
        raise _refused("REFUSED-SHORTFALL", f"{history_id}: month bands must cover M01..M{N_PLANNED_MONTHS:02d}")

    slots: list[dict[str, object]] = []
    seen_seeds: set[int] = set()
    quota_weeks = {offset for offset in range(1, N_PLANNED_WEEKS)}
    for offset in sorted(quota_weeks):
        week = weeks[offset]
        band = str(week_bands[offset])  # Mapping[int, str] typed
        ends = week_end_times(week, n_weekly)
        month_band = _majority_month_band(months, ends)
        need = _abnormal_count(band, month_band)
        order = sorted(range(n_weekly), key=lambda slot: slot_order_key(history_id, f"W{offset:02d}", slot))
        abnormal_slots = set(order[:need])
        for slot in range(n_weekly):
            requested_abnormal = slot in abnormal_slots
            seed = slot_seed(int(history_seed), stream, f"W{offset:02d}", slot)
            severity: float | None
            if requested_abnormal:
                roster = predict_anomaly_roster(seed)
                planned = slot_family(int(history_seed), f"W{offset:02d}", slot)
                severity = slot_severity(int(history_seed), f"W{offset:02d}", slot)
                reassigned = _needs_reassignment(int(history_seed), f"W{offset:02d}", slot, planned)
                family: AnomalyFamily | None = REASSIGNED_FAMILY if reassigned else planned
                family_original: AnomalyFamily | None = planned if reassigned else None
            else:
                roster = predict_normal_roster(seed)
                family = None
                family_original = None
                severity = None
            month_index = month_index_of(months, ends[slot])  # ends typed list[float]
            slots.append(
                {
                    "slot_id": f"{history_id}|W{offset:02d}|{slot:03d}",
                    "history_id": history_id,
                    "history_seed": int(history_seed),
                    "week_offset": offset,
                    "week_band": band,
                    "month_index": month_index,
                    "month_band": month_bands[month_index] if month_index is not None else None,
                    "arm": band if band in ("BASELINE-A", "BASELINE-B", "ELEVATED-A", "ELEVATED-B") else "NONE",
                    "requested_label": "abnormal" if requested_abnormal else "normal",
                    "requested_family": family.value if family is not None else None,
                    "requested_family_original": family_original.value if family_original is not None else None,
                    "reassigned": family_original is not None,
                    "reassignment_reason": ("missing_event on edge-only session; no interior event to miss" if family_original is not None else None),
                    "requested_severity": severity,
                    "seed": seed,
                    "predicted_robot_idx": roster[0],
                    "predicted_program_idx": roster[1],
                    "end_time_rel_s": ends[slot],
                    "stream": stream,
                }
            )
            seen_seeds.add(seed)

    reference_spacing_s = 38.5 * horizon.DAY_S / n_reference
    ref_order = sorted(range(n_reference), key=lambda slot: slot_order_key(history_id, "REFERENCE", slot))
    ref_abnormal = set(ref_order[:REFERENCE_ABNORMAL_TOTAL])
    for slot in range(n_reference):
        requested_abnormal = slot in ref_abnormal
        seed = slot_seed(int(history_seed), stream, "REFERENCE", slot)
        if requested_abnormal:
            roster = predict_anomaly_roster(seed)
            planned_ref = slot_family(int(history_seed), "REFERENCE", slot)
            reassigned_ref = _needs_reassignment(int(history_seed), "REFERENCE", slot, planned_ref)
            resolved_ref = REASSIGNED_FAMILY if reassigned_ref else planned_ref
            family_name: str | None = resolved_ref.value
            family_original_name: str | None = planned_ref.value if reassigned_ref else None
            severity_value: float | None = slot_severity(int(history_seed), "REFERENCE", slot)
        else:
            roster = predict_normal_roster(seed)
            family_name = None
            family_original_name = None
            severity_value = None
        end = _freeze_float(calendar, "c1_rel_s") - (n_reference - slot) * reference_spacing_s
        slots.append(
            {
                "slot_id": f"{history_id}|REFERENCE|{slot:03d}",
                "history_id": history_id,
                "history_seed": int(history_seed),
                "week_offset": None,
                "week_band": "REFERENCE",
                "month_index": None,
                "month_band": None,
                "arm": "REFERENCE",
                "requested_label": "abnormal" if requested_abnormal else "normal",
                "requested_family": family_name,
                "requested_family_original": family_original_name,
                "reassigned": family_original_name is not None,
                "reassignment_reason": ("missing_event on edge-only session; no interior event to miss" if family_original_name is not None else None),
                "requested_severity": severity_value,
                "seed": seed,
                "predicted_robot_idx": roster[0],
                "predicted_program_idx": roster[1],
                "end_time_rel_s": end,
                "stream": stream,
            }
        )
        seen_seeds.add(seed)

    _validate_slot_seeds(history_id, slots, seen_seeds)
    checks = _validate_plan(history_id, slots, weeks, months, week_bands, month_bands)
    refused = [check for check in checks if str(check["status"]) == "REFUSED"]
    if refused:
        raise _refused(
            str(refused[0]["reason"]),
            "; ".join(f"{item['reason']}: {item['detail']}" for item in refused),
        )
    return {"history_id": history_id, "history_seed": int(history_seed), "slots": slots, "checks": checks}


def _validate_slot_seeds(
    history_id: str, slots: Sequence[Mapping[str, object]], seen: set[int]
) -> None:
    if len(seen) != len(slots):
        raise _refused("REFUSED-SEED-COLLISION", f"{history_id}: {len(slots)} slots share {len(slots) - len(seen)} seeds")
    for slot in slots:
        seed = _slot_int(slot, "seed")
        if v1.CERTIFIED_OLD_SEED_MIN <= seed <= v1.CERTIFIED_OLD_SEED_MAX:
            raise _refused("REFUSED-SEED-COLLISION", f"{history_id}: slot seed {seed} hits the certified old block")
        if seed in (42401, 42402, 42403, 42404, 42405, 42406):
            raise _refused("REFUSED-SEED-COLLISION", f"{history_id}: slot seed {seed} reuses a history data seed")
        if seed < 0:
            raise _refused("REFUSED-SEED-COLLISION", f"{history_id}: slot seed {seed} is not a valid seed")


def _planned_fraction(rows: Sequence[Mapping[str, object]]) -> float:
    if not rows:
        return 0.0
    abnormal = sum(1 for row in rows if str(row["requested_label"]) == "abnormal")
    return abnormal / len(rows)


def _validate_plan(
    history_id: str,
    slots: Sequence[Mapping[str, object]],
    weeks: Sequence[Mapping[str, object]],
    months: Sequence[Mapping[str, object]],
    week_bands: Mapping[int, str],
    month_bands: Mapping[int, str],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    by_week: dict[int, list[Mapping[str, object]]] = {}
    for slot in slots:
        if slot["week_offset"] is not None:
            by_week.setdefault(_slot_int(slot, "week_offset"), []).append(slot)
    for offset in range(1, N_PLANNED_WEEKS):
        rows = by_week.get(offset, [])
        band = str(week_bands[offset])
        expected = len(rows) == v1.PLANNED_WEEKLY_FILES
        checks.append(
            {
                "reason": "REFUSED-SHORTFALL",
                "status": "PASS" if expected else "REFUSED",
                "detail": f"{history_id} W{offset:02d} ({band}): {len(rows)} planned files against quota {v1.PLANNED_WEEKLY_FILES}",
            }
        )
        if band in ("BASELINE-A", "BASELINE-B", "ELEVATED-A", "ELEVATED-B"):
            fraction = _planned_fraction(rows)
            low, high = v1.LOW_BASELINE_OPERATING_BAND if band.startswith("BASELINE") else v1.ELEVATED_OPERATING_BAND
            ok = low <= fraction <= high
            checks.append(
                {
                    "reason": "REFUSED-SHORTFALL",
                    "status": "PASS" if ok else "REFUSED",
                    "detail": f"{history_id} W{offset:02d} ({band}): planned fraction {fraction:.4f} against band [{low}, {high}]",
                }
            )
    month_band_by_index = {_row_int(row, "index"): _row_str(row, "band") for row in months}
    for index in range(2, N_PLANNED_MONTHS + 1):
        rows = [slot for slot in slots if slot["month_index"] == index]  # int | None == int
        band = month_band_by_index[index]
        if band == "LOW":
            low, high = v1.LOW_BASELINE_OPERATING_BAND
        elif band == "ELEVATED":
            low, high = v1.ELEVATED_OPERATING_BAND
        else:
            continue
        fraction = _planned_fraction(rows)
        ok = bool(rows) and low <= fraction <= high
        checks.append(
            {
                "reason": "REFUSED-SHORTFALL",
                "status": "PASS" if ok else "REFUSED",
                "detail": f"{history_id} M{index:02d} ({band}): planned fraction {fraction:.4f} over {len(rows)} files against band [{low}, {high}]",
            }
        )
    checks.append(validate_week_month_nesting(history_id, slots, weeks, months, week_bands))
    checks.append(validate_roster_support(history_id, slots))
    reference_rows = [slot for slot in slots if str(slot["arm"]) == "REFERENCE"]
    causal = all(_slot_float(slot, "end_time_rel_s") <= C1_REL_S for slot in reference_rows)
    checks.append(
        {
            "reason": "REFUSED-SHORTFALL",
            "status": "PASS" if causal and len(reference_rows) >= v1.MIN_REFERENCE_FILES else "REFUSED",
            "detail": f"{history_id}: {len(reference_rows)} reference files, all end_time <= c1: {causal}",
        }
    )
    return checks


def validate_week_month_nesting(
    history_id: str,
    slots: Sequence[Mapping[str, object]],
    weeks: Sequence[Mapping[str, object]],
    months: Sequence[Mapping[str, object]],
    week_bands: Mapping[int, str],
) -> dict[str, object]:
    """Verify every arm file nests file-level inside a same-band month."""
    band_by_index = {_row_int(row, "index"): _row_str(row, "band") for row in months}
    bad: list[str] = []
    checked = 0
    for slot in slots:
        arm = _slot_str(slot, "arm")
        if arm == "REFERENCE" or arm == "NONE":
            continue
        want = "LOW" if arm.startswith("BASELINE") else "ELEVATED"
        raw_month_index: object = slot["month_index"]
        month_index = _slot_int(slot, "month_index") if raw_month_index is not None else None
        got = band_by_index.get(month_index) if month_index is not None else None
        checked += 1
        if got != want:
            bad.append(f"{_slot_str(slot, 'slot_id')} arm {arm} lands in M{month_index} ({got}), want {want}")
    if bad:
        return {
            "reason": "REFUSED-OVERLAP",
            "status": "REFUSED",
            "detail": f"{history_id}: {len(bad)} arm files escape same-band months, e.g. {bad[0]}",
        }
    return {
        "reason": "REFUSED-OVERLAP",
        "status": "PASS",
        "detail": f"{history_id}: {checked} arm files nest file-level inside same-band months",
    }


def _arm_cells(slots: Sequence[Mapping[str, object]], arm: str) -> set[tuple[int, int]]:
    return {
        (_slot_int(slot, "predicted_robot_idx"), _slot_int(slot, "predicted_program_idx"))
        for slot in slots
        if _slot_str(slot, "arm") == arm
    }


def validate_roster_support(history_id: str, slots: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Verify every planned pair shares enough observable roster cells."""
    arms = {
        "BASELINE-A": _arm_cells(slots, "BASELINE-A"),
        "BASELINE-B": _arm_cells(slots, "BASELINE-B"),
        "ELEVATED-A": _arm_cells(slots, "ELEVATED-A"),
        "ELEVATED-B": _arm_cells(slots, "ELEVATED-B"),
    }
    pairs = [
        ("BASELINE-A", "ELEVATED-A"),
        ("BASELINE-B", "ELEVATED-A"),
        ("BASELINE-A", "ELEVATED-B"),
        ("BASELINE-B", "ELEVATED-B"),
        ("BASELINE-A", "BASELINE-B"),
    ]
    shortfalls = []
    for left, right in pairs:
        common = arms[left] & arms[right]
        retained = min(
            sum(1 for slot in slots if _slot_str(slot, "arm") == left),
            sum(1 for slot in slots if _slot_str(slot, "arm") == right),
        )
        if len(common) < v1.PLANNED_MIN_COMMON_STRATA or retained < v1.PLANNED_MIN_RETAINED_PER_ARM:
            shortfalls.append(f"{left}x{right}: {len(common)} common cells, {retained} files/arm")
    if shortfalls:
        return {
            "reason": "REFUSED-SHORTFALL",
            "status": "REFUSED",
            "detail": f"{history_id}: roster support shortfall: {'; '.join(shortfalls)}",
        }
    detail = "; ".join(f"{left}x{right}: {len(arms[left] & arms[right])} cells" for left, right in pairs)
    return {"reason": "REFUSED-SHORTFALL", "status": "PASS", "detail": f"{history_id}: common roster cells: {detail}"}


def default_planned_pairs() -> list[dict[str, str]]:
    """Return the five planned comparable pairs per history per horizon (v2)."""
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
        ("RUN-A", "ELEVATED-M7", "TRANSITION"),
        ("RUN-B", "ELEVATED-M7", "TRANSITION"),
        ("RUN-A", "RUN-B", "CONTROL"),
    ]
    rows: list[dict[str, str]] = []
    for horizon_name, plan in (("week", weekly), ("month", monthly)):
        for previous, current, kind in plan:
            rows.append({"horizon": horizon_name, "previous": previous, "current": current, "kind": kind})
    return rows


# --- cost ----------------------------------------------------------------------


def cost_estimate() -> dict[str, object]:
    """Return forward/storage formulas with evaluated v2 planned numbers."""
    n_histories = len(v1.NEW_HISTORY_IDS)
    span_files = (N_PLANNED_WEEKS - 1) * v1.PLANNED_WEEKLY_FILES
    per_history = v1.PLANNED_REFERENCE_FILES + span_files
    total_files = n_histories * per_history
    n_seeds = len(v1.FROZEN_ENCODER_SEEDS)
    required_forwards = total_files * n_seeds
    required_with_smoke = required_forwards + SMOKE_ENVELOPE_FILES
    headroom = v1.FORWARD_HEADROOM
    shortfall = required_with_smoke - headroom
    wave_bytes = total_files * v1.ASSUMED_MEAN_STEPS_PER_FILE * v1.N_CHANNELS * 4
    embedding_bytes = total_files * v1.EMBEDDING_DIM * 4 * n_seeds
    manifest_bytes = total_files * 2048
    return {
        "formulas": {
            "files_per_history": "N_ref + (N_weeks - 1) * N_weekly",
            "total_files": "H * files_per_history",
            "required_forwards": "total_files * n_encoder_seeds (+ bounded smoke envelope)",
            "waveform_bytes": "total_files * mean_steps * n_channels * 4",
            "embedding_bytes": "total_files * embedding_dim * 4 * n_encoder_seeds",
            "manifest_bytes": "total_files * 2048",
        },
        "assumptions": {
            "mean_steps_per_file": v1.ASSUMED_MEAN_STEPS_PER_FILE,
            "mean_steps_basis": "stated assumption inside the RegimeConfig 200-800 step range; not measured",
            "weekly_density_basis": "planned 40 files/week carried from observed old-corpus density; verified at the pre-generation freeze, never silently rescaled",
            "business_density_note": "owner-reported 500-1000 files/day is later scale context, not this protocol's target",
            "reference_burden_default": f"reference pool follows the LOW operating band ({REFERENCE_ABNORMAL_TOTAL}/{v1.PLANNED_REFERENCE_FILES}); v2 research default mirroring analysis-span baseline conditions, disclosed not tuned",
        },
        "planned_counts": {
            "n_histories": n_histories,
            "files_per_history": per_history,
            "total_files": total_files,
            "n_encoder_seeds": n_seeds,
        },
        "forwards": {
            "required_forwards": required_forwards,
            "smoke_envelope": SMOKE_ENVELOPE_FILES,
            "required_with_smoke": required_with_smoke,
            "cumulative_spent": v1.CUMULATIVE_FORWARDS_SPENT,
            "cap": v1.FORWARD_CAP,
            "headroom": headroom,
            "shortfall_vs_headroom": shortfall,
            "cap_policy": "no reset; owner budget authority covers planned inference; readiness executes 0 forwards (execution release still required)",
        },
        "storage_bytes": {
            "waveforms": wave_bytes,
            "embeddings": embedding_bytes,
            "manifests": manifest_bytes,
        },
    }


def forward_gate() -> dict[str, object]:
    """Return the readiness forward gate (no inference executes in readiness)."""
    estimate = cost_estimate()
    forwards = estimate["forwards"]
    assert isinstance(forwards, dict)
    return {
        "gate": "READY-NO-FORWARDS-EXECUTED",
        "reason": (
            f"planned {forwards['required_with_smoke']} forwards are estimates under owner budget authority; "
            "readiness executes 0 forwards, 0 model loads; official inference needs the execution release"
        ),
        "cost_digest": sha256_object(estimate),
    }


def protocol_spec() -> dict[str, object]:
    """Return the frozen v2 execution identity, calendar, rules and cost."""
    calendar = build_history_calendar(v1.NEW_HISTORY_IDS[0])
    return {
        "protocol_id": PROTOCOL_ID,
        "base_protocol_id": BASE_PROTOCOL_ID,
        "base_protocol_sha256": BASE_PROTOCOL_SHA256,
        "amendment_scope": (
            "bounded owner-authorized v2 calendar/bands amendment only: 30-week span, "
            "7 monthly offsets, second elevated month (M07), remapped arm weeks; "
            "no threshold/estimand/lineage/scientific-scope change"
        ),
        "family_reassignment_amendment": (
            "owner-authorized prospective reassignment for exactly 4 missing_event "
            "slots on edge-only 2-regime sessions (no interior event to miss): "
            "CONTROLLED-02|W15|038, CONTROLLED-04|REFERENCE|163, "
            "CONTROLLED-05|W15|013, CONTROLLED-06|W29|009; planned as subtle_drift "
            "at same seed/severity/week/arm/quota/stream with original request "
            "retained (requested_family_original + reassignment_reason); all-9 "
            "rotation unchanged; no runtime fallback (single-shot REFUSED-GENERATION "
            "still applies to any other rejection)"
        ),
        "preserved_outcomes": v1.protocol_spec()["preserved_outcomes"],
        "horizons": list(v1.HORIZONS),
        "calendar": {
            "c1_rel_s": C1_REL_S,
            "tmax_rel_s": _freeze_float(calendar, "tmax_rel_s"),
            "projection": cast("dict[str, object]", calendar["calendar_describe"]),
            "weeks": list(cast("list[dict[str, object]]", calendar["weeks"])),
            "months": list(cast("list[dict[str, object]]", calendar["months"])),
        },
        "planned_quotas": {
            "reference_files_per_history": v1.PLANNED_REFERENCE_FILES,
            "weekly_files_per_period": v1.PLANNED_WEEKLY_FILES,
            "baseline_abnormal_per_week": BASELINE_ABNORMAL_PER_WEEK,
            "elevated_abnormal_per_week": ELEVATED_ABNORMAL_PER_WEEK,
            "reference_abnormal_total": REFERENCE_ABNORMAL_TOTAL,
            "severity_band": [SEVERITY_LO, SEVERITY_HI],
            "family_policy": "deterministic rotation over all 9 hard families",
            "min_common_strata": v1.PLANNED_MIN_COMMON_STRATA,
            "min_retained_per_arm": v1.PLANNED_MIN_RETAINED_PER_ARM,
            "min_retained_gap": v1.PLANNED_MIN_RETAINED_GAP,
        },
        "planned_pairs": default_planned_pairs(),
        "cost": cost_estimate(),
        "refusal_policy": v1.protocol_spec()["refusal_policy"],
    }


def execution_freeze() -> dict[str, object]:
    """Return the v2 execution spec together with its freeze digest."""
    spec = protocol_spec()
    return {"protocol_sha256": sha256_object(spec), "spec": spec}


# --- generation path (same implementation for smoke and official planning) -------


def fresh_generator(config: SynthConfig | None = None) -> SessionGenerator:
    """Return a SessionGenerator over the frozen default synth config."""
    return SessionGenerator(config or SynthConfig())


def generate_planned_file(
    slot: Mapping[str, object],
    generator: SessionGenerator,
    stream: str = "plan",
) -> FileSample:
    """Generate one planned waveform through the real synthesis machinery.

    Abnormal slots use exactly the planned family/severity on the planned
    slot seed, once, with no retry, reseed, fallback or rescue: a legitimate
    strength-gate rejection fails fast as ``REFUSED-SHORTFALL``. Normal slots
    use the clean normal path. Attaches lineage provenance only (no
    arm/band/timestamp markers).
    """
    seed = _slot_int(slot, "seed")
    requested = _slot_str(slot, "requested_label")
    provenance = FactoryProvenance(
        seed=seed,
        stream=f"s24c-controlled-v2-{stream}",
        generator_version=GENERATOR_VERSION,
        config_hash=generator.cfg.hash(),
    )
    if requested == "abnormal":
        family = AnomalyFamily(_slot_str(slot, "requested_family"))
        severity = _slot_float(slot, "requested_severity")
        sample = generator.generate_anomaly(seed=seed, split=PLANNED_SPLIT, family=family, severity=severity)
        if sample.file_label != SampleLabel.ABNORMAL or sample.anomaly_meta is None:
            raise _refused(
                REFUSED_GENERATION,
                f"{_slot_str(slot, 'slot_id')}: strength gate rejected requested {family.value}@{severity:.3f}; "
                "no retry/reseed/fallback/rescue; owner decision required before the plan changes",
            )
        if sample.anomaly_meta.family != family or sample.anomaly_meta.severity != severity:
            raise _refused(
                "REFUSED-IDENTITY",
                f"{_slot_str(slot, 'slot_id')}: realized injector {sample.anomaly_meta.family.value}@"
                f"{sample.anomaly_meta.severity} != requested {family.value}@{severity}",
            )
    elif requested == "normal":
        sample = generator.generate_normal(seed=seed, split=PLANNED_SPLIT)
    else:
        raise _refused("REFUSED-IDENTITY", f"{_slot_str(slot, 'slot_id')}: unknown requested label {requested!r}")
    sample.factory_provenance = provenance  # FileSample.factory_provenance typed
    sample.validate()
    verify_encoder_label_free(sample, _slot_str(slot, "slot_id"))
    return sample


def trace_annotation(sample: FileSample, slot: Mapping[str, object]) -> dict[str, object]:
    """Derive the trace annotation from the REALIZED injector (never copied).

    Recomputes mask extent/channels from the realized mask and asserts exact
    agreement with the realized meta; the requested plan only selects the slot.
    """
    realized: dict[str, object] = {
        "slot_id": _slot_str(slot, "slot_id"),
        "file_id": sample.file_id,
        "seed": int(sample.seed),
        "file_label": sample.file_label.value,
        "requested_label": _slot_str(slot, "requested_label"),
        "requested_family": slot["requested_family"],
        "requested_severity": slot["requested_severity"],
    }
    if sample.file_label == SampleLabel.ABNORMAL:
        meta = sample.anomaly_meta
        mask = sample.anomaly_mask
        assert meta is not None and mask is not None
        columns = np.flatnonzero(np.asarray(mask.any(axis=0)))
        realized_channels = sorted(int(channel) for channel in range(sample.C) if bool(mask[channel].any()))
        if int(columns.min()) != meta.start or int(columns.max()) + 1 != meta.end:
            raise _refused(
                "REFUSED-IDENTITY",
                f"{_slot_str(slot, 'slot_id')}: mask extent [{columns.min()}, {columns.max() + 1}) != meta [{meta.start}, {meta.end})",
            )
        if realized_channels != sorted(meta.affected_channels):
            raise _refused(
                "REFUSED-IDENTITY",
                f"{_slot_str(slot, 'slot_id')}: mask channels {realized_channels} != meta {sorted(meta.affected_channels)}",
            )
        realized.update(
            {
                "family": meta.family.value,
                "severity": float(meta.severity),
                "start_step": int(meta.start),
                "end_step": int(meta.end),
                "affected_channels": sorted(int(channel) for channel in meta.affected_channels),
                "abnormal_steps": int(mask.sum()),
                "abnormal_step_fraction": float(mask.sum()) / float(mask.size),
                "duration_steps": int(meta.end) - int(meta.start),
                "extra": dict(meta.extra),
            }
        )
    else:
        realized.update(
            {
                "family": None,
                "severity": None,
                "start_step": None,
                "end_step": None,
                "affected_channels": [],
                "abnormal_steps": 0,
                "abnormal_step_fraction": 0.0,
                "duration_steps": 0,
                "extra": {},
            }
        )
    return realized


def verify_encoder_label_free(sample: FileSample, slot_id: str) -> None:
    """Fail closed if encoder-visible inputs leak diagnostics or arm markers."""
    inputs = sample.encoder_inputs()
    leaked = [key for key in inputs if key in DIAGNOSTIC_ONLY_FIELD_NAMES]
    if leaked:
        raise _refused("REFUSED-IDENTITY", f"{slot_id}: encoder inputs leak diagnostics {leaked}")
    if set(inputs) - MODEL_INPUT_FIELD_NAMES:
        raise _refused("REFUSED-IDENTITY", f"{slot_id}: encoder inputs carry non-contract fields")
    lowered = sample.file_id.lower()
    for token in ("baseline", "elevated", "buffer", "warmup", "low", "arm", "smoke", "throwaway"):
        if token in lowered:
            raise _refused("REFUSED-IDENTITY", f"{slot_id}: file_id carries an arm/band marker {token!r}")
    if not (sample.file_id.startswith("N-") or sample.file_id.startswith("A-")):
        raise _refused("REFUSED-IDENTITY", f"{slot_id}: file_id breaks the inherited N-/A- scheme")


__all__ = [
    "BASELINE_ABNORMAL_PER_WEEK",
    "BASE_PROTOCOL_ID",
    "BASE_PROTOCOL_SHA256",
    "C1_REL_S",
    "DEFAULT_MONTH_BANDS",
    "DEFAULT_WEEK_BANDS",
    "ELEVATED_ABNORMAL_PER_WEEK",
    "MONTH_BANDS",
    "N_PLANNED_MONTHS",
    "N_PLANNED_WEEKS",
    "PLANNED_SPLIT",
    "PROTOCOL_ID",
    "REFERENCE_ABNORMAL_TOTAL",
    "SEVERITY_HI",
    "SEVERITY_LO",
    "SMOKE_ENVELOPE_FILES",
    "REFUSED_GENERATION",
    "REASSIGNED_FAMILY",
    "WEEK_BANDS",
    "assert_execution_construction_label_blind",
    "build_history_calendar",
    "cost_estimate",
    "default_planned_pairs",
    "execution_freeze",
    "forward_gate",
    "fresh_generator",
    "generate_planned_file",
    "month_index_of",
    "plan_history",
    "planned_month_identities",
    "planned_week_identities",
    "predict_anomaly_roster",
    "predict_normal_roster",
    "protocol_spec",
    "require_fresh_dir",
    "slot_family",
    "slot_order_key",
    "slot_seed",
    "slot_severity",
    "trace_annotation",
    "validate_roster_support",
    "validate_week_month_nesting",
    "verify_encoder_label_free",
    "week_end_times",
]
