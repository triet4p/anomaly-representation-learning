"""Sprint 24 Task 7-M17-C2 official paired-experiment runner (versioned correction).

Public official-matrix path for the frozen option-A science. This module is the
C2 correction deliverable: it generates/captures/evaluates the COMPLETE
official matrix (27 cells x G in {24, 48} x 6 histories, reference pools,
population arms) reusing the ACCEPTED frozen mathematics only:

- grouping/dependence: ``GROUP_SIZE_K = 8`` independent scored groups per cell,
  G24 interim (192 pairs/cell) then G48 release (384 pairs/cell), NO 96 branch;
- seed plan: ``check_group_seed_plan`` (6 histories x 384 slots = 2,304 unique);
- evaluation: ``score_scored_group`` / ``summarize_cell`` /
  ``holm_over_families`` / ``dilution_ladder`` / ``calibration_spec`` /
  ``evaluation_spec`` from ``sprint24_paired_execution`` (imported, not copied);
- reference fits: causal pool rule ``end_time <= c1`` is realized as disjoint
  pair-index allocation (pair seeds derive from the frozen domain tag, so a
  pair seed can never equal a reference seed drawn from the reserved
  reference-only index band); bandwidth = median pairwise distance (same rule
  as ``fit_reference``); KMeans prototypes K in {4, 8, 16} primary 8;
- sensitivity: E2_K8 occupancy (TV between K=8 prototype occupancies),
  D_LOC/D_SPREAD within-prototype diagnostics (M14 ``within_prototype_arrays``
  math with the frozen ``D_MIN_COVERAGE = 0.5`` floor), E1 primary;
- uncertainty: frozen null q95 per cell per encoder seed (matched-size
  normal-normal twin groups, K=8 vs K=8); Clopper-Pearson exact intervals;
  Holm step-down over the 9 family contrasts; history-cluster bootstrap
  (B=2000, seed 2407, descriptive) for stage-2 medians;
- leakage: group key = pair seed (all variants of one seed stay in one
  split); reference/control seeds disjoint from pair seeds; labels never enter
  encoder/reference/prototype/bandwidth/detector fits (generation labels only
  select FileSample arms; ``encoder_inputs`` exclusion is asserted by the
  schema test suite).

What this module does NOT do: train, refit, contact any remote host, release
Phase C, or edit accepted M11-M16 products. Official full inference still
requires M17 C2 evidence PASS + Bronze checkpoint + M18 release; until then
only the tiny fixture smoke (``tiny_official_smoke``) executes forwards, and
every forward/load is counted.

Contract identity: ``S24-PAIRED-EXECUTION-CONTRACT-C2`` (versioned successor of
the immutable historical A01 ``S24-PAIRED-EXECUTION-CONTRACT-A01``; A01 bytes
are preserved byte-identical and only superseded for the official path).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from representation import sprint24_paired_decision_experiment as design
from representation import sprint24_paired_execution as base
from representation.sprint24_paired_execution import holm_over_families
from representation.sprint24_paired_journal import CaptureJournal
from representation.sprint24_context_audit import canonical_json

#: Versioned C2 official contract identifier (successor of A01, never a silent edit).
OFFICIAL_CONTRACT_ID = "S24-PAIRED-EXECUTION-CONTRACT-C2"

#: Superseded historical contract (bytes preserved, read-only).
SUPERSEDED_CONTRACT_ID = base.EXECUTION_CONTRACT_ID

#: Criterion pointer (C2 correction scope).
CRITERION = "docs/sprint-plans/sprint-24.md:266-271"

#: Review cycle this correction targets.
REVIEW_CYCLE = "S24-7-M17-C2"

#: Cumulative observed forwards before this attempt (M17 checkpoint + M18 zero).
CUMULATIVE_FORWARDS_BEFORE = 77845

#: Pair-index allocation per history: [0, 384) scored pairs, [384, 384+220)
#: causal reference pool, [604, 604+K] null-calibration pool, [1000, ...) stage-2
#: population pool. Bands are disjoint by construction, so reference/control
#: seeds can never collide with pair seeds (same frozen domain derivation).
PAIR_SLOTS_PER_HISTORY = 384
REFERENCE_SLOTS_PER_HISTORY = 220
REFERENCE_INDEX_BASE = PAIR_SLOTS_PER_HISTORY
NULL_PAIRS_PER_CELL_SEED = 8
NULL_INDEX_BASE = PAIR_SLOTS_PER_HISTORY + REFERENCE_SLOTS_PER_HISTORY
STAGE2_INDEX_BASE = 10_000

#: Null-calibration groups per cell per encoder seed (matched-size K=8 vs K=8).
NULL_GROUPS_PER_CELL = 4
BANDWIDTH_INDEX_BASE = NULL_INDEX_BASE + NULL_GROUPS_PER_CELL * 2 * base.GROUP_SIZE_K
BANDWIDTH_SLOTS_PER_HISTORY = 220

#: Stage-2 cluster-bootstrap frozen settings (M14 inheritance).
CLUSTER_BOOTSTRAP_REPLICATES = 2000
CLUSTER_BOOTSTRAP_SEED = 2407

#: Closed disposition vocabulary (inherited, never extended silently).
DISPOSITIONS = base.DISPOSITIONS

#: Closed refusal reasons for the official runner.
REFUSAL_REASONS = base.REFUSAL_REASONS + (
    "REFUSED-PARTIAL-MATRIX",
    "REFUSED-OFFICIAL-RELEASE",
)


def _refused(reason: str, detail: str) -> ValueError:
    return ValueError(f"{reason}: {detail}")


def _require_official_release(permitted: bool) -> None:
    if not permitted:
        raise _refused(
            "REFUSED-OFFICIAL-RELEASE",
            "official full-matrix inference needs M17 C2 PASS + checkpoint + M18 release",
        )


# --- official matrix ----------------------------------------------------------

#: Ordered cell grid: 9 families x 3 severities = 27 cells (frozen order).
CELL_GRID: tuple[tuple[str, float], ...] = tuple(
    (str(family), float(severity))
    for family in base.PAIRED_FAMILIES
    for severity in base.PAIRED_SEVERITIES
)


@dataclass(frozen=True)
class CellPlan:
    """One (family x severity) cell: exact pair-index ranges per G level."""

    family: str
    severity: float
    pair_index_g24: tuple[int, ...]
    pair_index_g48_extra: tuple[int, ...]

    @property
    def pair_index_g48(self) -> tuple[int, ...]:
        return self.pair_index_g24 + self.pair_index_g48_extra


def cell_plans(
    groups_per_cell: Sequence[int] = base.STAGE1_GROUP_GRID,
    group_size_k: int = base.GROUP_SIZE_K,
) -> list[CellPlan]:
    """Return the 27 frozen cell plans with disjoint pair-index ranges.

    G24 (interim) owns pair indices [0, G24*K); G48 (release) appends the next
    G24*K indices, so the interim result is a strict prefix of the release.
    Ranges are IDENTICAL for every history; cross-cell sharing of a pair seed
    is explicit (same (history_seed, pair_index) re-derives the same physical
    pair seed in every cell), while within-cell disjointness across groups
    comes from disjoint index ranges.
    """
    grid = [int(value) for value in groups_per_cell]
    if grid != [24, 48]:
        raise _refused("REFUSED-UNKNOWN-DESIGN", f"group grid must be [24, 48], got {grid}")
    if int(group_size_k) != base.GROUP_SIZE_K:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "group size K must stay 8")
    g24 = grid[0] * int(group_size_k)
    g48_extra = (grid[1] - grid[0]) * int(group_size_k)
    plans: list[CellPlan] = []
    for family, severity in CELL_GRID:
        plans.append(
            CellPlan(
                family=str(family),
                severity=float(severity),
                pair_index_g24=tuple(range(0, g24)),
                pair_index_g48_extra=tuple(range(g24, g24 + g48_extra)),
            )
        )
    return plans


def matrix_plan() -> dict[str, Any]:
    """Freeze the complete official matrix as a machine-readable mapping."""
    plans = cell_plans()
    seed_plan = base.check_group_seed_plan()
    per_history_pair_slots = seed_plan["pair_slots_per_history"]
    return {
        "official_contract_id": OFFICIAL_CONTRACT_ID,
        "supersedes": SUPERSEDED_CONTRACT_ID,
        "review_cycle": REVIEW_CYCLE,
        "criterion": CRITERION,
        "cell_count": len(plans),
        "cells": [
            {
                "family": plan.family,
                "severity": plan.severity,
                "g24_pair_indices": list(plan.pair_index_g24),
                "g48_extra_pair_indices": list(plan.pair_index_g48_extra),
                "pairs_g24": len(plan.pair_index_g24),
                "pairs_g48": len(plan.pair_index_g48),
            }
            for plan in plans
        ],
        "histories": list(base.PAIRED_HISTORY_IDS),
        "history_seeds": list(base.PROPOSED_HISTORY_SEEDS),
        "seed_plan": seed_plan,
        "pairs_per_cell": {str(g): int(g) * base.GROUP_SIZE_K for g in base.STAGE1_GROUP_GRID},
        "total_pair_slots_per_history": int(per_history_pair_slots),
        "total_unique_pair_seeds": int(seed_plan["expected_unique"]),
        "reference_slots_per_history": REFERENCE_SLOTS_PER_HISTORY,
        "null_groups_per_cell": NULL_GROUPS_PER_CELL,
        "stage2": {
            "sizes": {key: list(value) for key, value in base.STAGE2_SIZES.items()},
            "burdens": dict(base.STAGE2_BURDENS),
        },
        "frozen_encoder_seeds": list(base.FROZEN_ENCODER_SEEDS),
        "no_96_branch": True,
    }


def allocation_table() -> dict[str, Any]:
    """Map every history/role to exact disjoint pair-index bands.

    The same table governs generation, capture, calibration and population
    arms, so lineage/no-leakage can be verified without re-running capture.
    """
    return {
        "pair_band": [0, PAIR_SLOTS_PER_HISTORY],
        "reference_band": [REFERENCE_INDEX_BASE, REFERENCE_INDEX_BASE + REFERENCE_SLOTS_PER_HISTORY],
        "null_band": [NULL_INDEX_BASE, NULL_INDEX_BASE + NULL_GROUPS_PER_CELL * 2 * base.GROUP_SIZE_K],
        "stage2_band": [STAGE2_INDEX_BASE, None],
        "rule": (
            "pair seeds derive from (history_seed, pair_index) under the frozen "
            "domain tag; distinct index bands guarantee disjoint reference/control "
            "seeds from pair seeds; group key = pair seed keeps every variant of "
            "one seed in one split."
        ),
    }


# --- generation ---------------------------------------------------------------

#: One generated pair plus its lineage (file IDs + disposition + attempts).
@dataclass(frozen=True)
class GeneratedPair:
    pair_seed: int
    history_seed: int
    pair_index: int
    family: str
    severity: float
    twin_file_id: str
    abnormal_file_id: str
    disposition: str
    n_attempts: int
    group_id: str
    source_id: str


def _family_of(name: str) -> Any:
    from synth.schema import AnomalyFamily

    try:
        return AnomalyFamily(str(name))
    except ValueError:
        raise _refused("REFUSED-UNKNOWN-DESIGN", f"unknown anomaly family: {name!r}")


def generate_cell_pairs(
    history_seed: int,
    pair_indices: Sequence[int],
    family: str,
    severity: float,
) -> list[GeneratedPair]:
    """Generate every planned pair of one cell/history slice (no redraw).

    Every pair index is attempted exactly once; non-PAIRED outcomes are kept
    with their closed disposition (never redrawn, never substituted).
    """
    from synth.config import SynthConfig
    from synth.generator import SessionGenerator, derive_pair_seed

    gen = SessionGenerator(SynthConfig())
    target = _family_of(family)
    records: list[GeneratedPair] = []
    for pair_index in [int(value) for value in pair_indices]:
        pair_seed = int(derive_pair_seed(int(history_seed), int(pair_index)))
        pair = gen.generate_paired_twin(pair_seed, target, float(severity))
        records.append(
            GeneratedPair(
                pair_seed=int(pair_seed),
                history_seed=int(history_seed),
                pair_index=int(pair_index),
                family=str(family),
                severity=float(severity),
                twin_file_id=str(pair.twin.file_id),
                abnormal_file_id=str(pair.abnormal.file_id),
                disposition=str(pair.disposition),
                n_attempts=int(pair.n_attempts),
                group_id=str(pair.group_id),
                source_id=str(pair.source_id),
            )
        )
    return records


def _pair_lineage(pair: Any, pair_index: int) -> dict[str, Any]:
    """Describe the actual selected draw, including rejected dispositions."""
    return {
        "pair_index": int(pair_index), "pair_seed": int(pair.pair_seed),
        "source_id": str(pair.source_id), "group_id": str(pair.group_id),
        "twin_file_id": str(pair.twin.file_id),
        "abnormal_file_id": str(pair.abnormal.file_id),
        "disposition": str(pair.disposition), "n_attempts": int(pair.n_attempts),
        "strength": float(pair.strength),
        "attempts": [{"attempt": value.attempt, "attempt_seed": value.attempt_seed,
                      "status": value.status, "strength": value.strength, "reason": value.reason}
                     for value in pair.attempts],
        "normal_path_diverges": bool(pair.normal_path_diverges),
        "config_hash": str(pair.twin.config_hash),
        "generator_version": str(pair.twin.generator_version),
    }


def _generate_with_custody(
    gen: Any, pair_seed: int, family: Any, severity: float, pair_index: int,
    history_id: str, role: str, journal: CaptureJournal | None,
) -> Any:
    """Persist each actual attempt as it happens; no redraw or reconstruction."""
    if journal is None:
        return gen.generate_paired_twin(int(pair_seed), family, float(severity))

    def record_attempt(attempt: dict[str, object]) -> None:
        journal.record("injection_attempt", {
            "history_id": history_id, "role": role, "pair_index": int(pair_index),
            "family": str(family.value), "severity": float(severity), **attempt,
        })

    identity = {"history_id": history_id, "role": role, "pair_index": int(pair_index),
                "pair_seed": int(pair_seed), "family": str(family.value), "severity": float(severity)}
    journal.record("source_started", identity)
    try:
        return gen.generate_paired_twin(int(pair_seed), family, float(severity), on_attempt=record_attempt)
    except Exception as exc:
        journal.record("source_failed", {**identity, "exception": repr(exc)})
        raise


def _pre_c1_lineage(pair: Any, pair_index: int, role: str, c1: float) -> dict[str, Any]:
    """Materialize the synthetic pre-c1 slot, separate from execution time."""
    from representation.sprint24_business_horizon import DAY_S

    spacing = 38.5 * DAY_S / REFERENCE_SLOTS_PER_HISTORY
    if role == "reference":
        slot, fraction = pair_index - REFERENCE_INDEX_BASE, 0.0
    elif role == "bandwidth":
        slot, fraction = pair_index - BANDWIDTH_INDEX_BASE, 0.5
    elif role == "null":
        slot, fraction = pair_index - NULL_INDEX_BASE, 0.25
    else:
        raise _refused("REFUSED-UNKNOWN-DESIGN", f"unknown pre-c1 role: {role}")
    if not 0 <= slot < REFERENCE_SLOTS_PER_HISTORY:
        raise _refused("REFUSED-IDENTITY", "pre-c1 schedule slot out of range")
    steps = int(pair.twin.T)
    duration_s = steps * 0.125  # synth InjectionContext and CausalSignalGenerator default dt
    end = float(c1) - (REFERENCE_SLOTS_PER_HISTORY - slot - fraction) * spacing
    start = end - duration_s
    if steps <= 0 or duration_s >= spacing / 4 or end > c1:
        raise _refused("REFUSED-ERROR-BOUNDARY", "synthetic pre-c1 waveform does not fit its slot")
    return {**_pair_lineage(pair, pair_index), "time_basis": "planned synthetic calendar",
            "role": role, "sample_interval_s": 0.125, "observed_waveform_steps": steps,
            "window_start_rel_s": float(c1) - 38.5 * DAY_S, "window_end_rel_s": float(c1),
            "planned_start_rel_s": start, "planned_end_rel_s": end, "c1_rel_s": float(c1)}


def _evaluation_lineage(
    pair: Any, pair_index: int, scenario_id: str, role: str,
    window: Mapping[str, Any], slot: int, n_slots: int, c1: float,
) -> dict[str, Any]:
    """Bind a source pair to a non-overlapping slot in a synthetic scenario."""
    window_start, window_end = float(window["start_rel_s"]), float(window["end_rel_s"])
    if not 0 <= slot < n_slots or n_slots <= 0:
        raise _refused("REFUSED-IDENTITY", "invalid scenario slot")
    spacing = (window_end - window_start) / n_slots
    end = window_start + (slot + 1) * spacing
    twin_steps, variant_steps = int(pair.twin.T), int(pair.abnormal.T)
    duration_s = max(twin_steps, variant_steps) * 0.125
    start = end - duration_s
    if (spacing <= 0 or duration_s <= 0 or duration_s >= spacing
            or start < window_start or start <= c1 or end > window_end):
        raise _refused("REFUSED-ERROR-BOUNDARY", "waveform does not fit its post-c1 scenario slot")
    return {**_pair_lineage(pair, pair_index), "time_basis": "planned synthetic calendar",
            "scenario_id": scenario_id, "role": role, "window_key": str(window["period_key"]),
            "window_start_rel_s": window_start, "window_end_rel_s": window_end,
            "planned_start_rel_s": start, "planned_end_rel_s": end, "c1_rel_s": c1,
            "sample_interval_s": 0.125, "twin_waveform_steps": twin_steps,
            "variant_waveform_steps": variant_steps, "slot": slot, "n_slots": n_slots}


def generate_reference_pool(history_seed: int) -> list[int]:
    """Derive the frozen causal-reference pair-seed pool for one history."""
    from synth.generator import derive_pair_seed

    return [
        int(derive_pair_seed(int(history_seed), REFERENCE_INDEX_BASE + offset))
        for offset in range(REFERENCE_SLOTS_PER_HISTORY)
    ]


def generate_null_pool(history_seed: int, n_files: int = NULL_GROUPS_PER_CELL * 2 * base.GROUP_SIZE_K) -> list[int]:
    """Derive matched-size null-calibration twin seeds (disjoint band)."""
    from synth.generator import derive_pair_seed

    return [
        int(derive_pair_seed(int(history_seed), NULL_INDEX_BASE + offset))
        for offset in range(int(n_files))
    ]

def generate_bandwidth_pool(history_seed: int) -> list[int]:
    """Derive a separate pre-evaluation source pool for bandwidth estimation."""
    from synth.generator import derive_pair_seed

    return [
        int(derive_pair_seed(int(history_seed), BANDWIDTH_INDEX_BASE + offset))
        for offset in range(BANDWIDTH_SLOTS_PER_HISTORY)
    ]



#: In-process capture counter (test/smoke isolation; honest forward custody).
_CAPTURE_COUNTER = {"files_captured": 0}


def capture_files_count_holder() -> dict[str, int]:
    """Return the cumulative files captured through the shared path (this process)."""
    return dict(_CAPTURE_COUNTER)


def reset_capture_counter() -> None:
    """Reset the in-process capture counter (test/smoke isolation only)."""
    _CAPTURE_COUNTER["files_captured"] = 0

#: One file embedding row (device-neutral float64) plus identity.
@dataclass
class CapturedRow:
    file_id: str
    embedding: Any
    history_id: str
    model_seed: int


def capture_files(
    bundles: Mapping[int, Any],
    samples: Sequence[Any],
    history_id: str,
    device: str | None = None,
    journal: CaptureJournal | None = None,
) -> dict[int, Any]:
    """Capture frozen embeddings for one file batch on every encoder bundle.

    Thin wrapper over the public ``capture_s24_file_features`` entry: order,
    shape and finiteness admission per seed; frozen-state digests verified
    unchanged around the capture.
    """
    from representation.sprint24_features import (
        assert_s24_frozen_unchanged,
        capture_s24_file_features,
        s24_live_state_digests,
    )

    materialized = list(samples)
    if not materialized:
        raise _refused("REFUSED-ERROR-BOUNDARY", "capture needs a non-empty file batch")
    file_ids = [str(sample.file_id) for sample in materialized]
    if len(set(file_ids)) != len(file_ids):
        raise _refused("REFUSED-IDENTITY", "capture file IDs are not unique")
    per_seed: dict[int, Any] = {}
    for seed in base.FROZEN_ENCODER_SEEDS:
        bundle = bundles[int(seed)]
        before = s24_live_state_digests(bundle.model, bundle.bank)  # type: ignore[attr-defined]
        captured = capture_s24_file_features(
            model=bundle.model,  # type: ignore[attr-defined]
            bank=bundle.bank,  # type: ignore[attr-defined]
            samples=materialized,
            history_id=str(history_id),
            model_seed=int(seed),
            device=device,
        )
        order = [item.file_id for item in captured]
        if order != file_ids:
            raise _refused("REFUSED-IDENTITY", f"capture order drifted for seed {seed}")
        after = s24_live_state_digests(bundle.model, bundle.bank)  # type: ignore[attr-defined]
        assert_s24_frozen_unchanged(before, after)
        _CAPTURE_COUNTER["files_captured"] += len(captured)
        if journal is not None:
            journal.capture(int(seed), str(history_id), list(captured))
        per_seed[int(seed)] = captured
    return per_seed


def embedding_matrices(
    per_seed_capture: Mapping[int, Any],
    twin_ids: Sequence[str],
    abnormal_ids: Sequence[str],
) -> dict[int, dict[str, Any]]:
    """Stack per-seed (twin, abnormal) float64 matrices with admission checks."""
    import numpy as np

    twin_ids = [str(value) for value in twin_ids]
    abnormal_ids = [str(value) for value in abnormal_ids]
    if len(set(twin_ids)) != len(twin_ids) or len(set(abnormal_ids)) != len(abnormal_ids):
        raise _refused("REFUSED-IDENTITY", "twin/abnormal ID lists are not unique")
    if set(twin_ids) & set(abnormal_ids):
        raise _refused("REFUSED-IDENTITY", "twin/abnormal ID lists overlap")
    matrices: dict[int, dict[str, Any]] = {}
    for seed in base.FROZEN_ENCODER_SEEDS:
        captured = list(per_seed_capture[int(seed)])
        by_id = {str(item.file_id): item for item in captured}
        missing = [value for value in twin_ids + abnormal_ids if value not in by_id]
        if missing:
            raise _refused("REFUSED-IDENTITY", f"capture missing {len(missing)} files for seed {seed}")
        twin = np.stack(
            [np.asarray(by_id[value].embedding.detach().cpu().float().numpy()) for value in twin_ids]
        ).astype(np.float64)
        abnormal = np.stack(
            [np.asarray(by_id[value].embedding.detach().cpu().float().numpy()) for value in abnormal_ids]
        ).astype(np.float64)
        if twin.ndim != 2 or abnormal.ndim != 2:
            raise _refused("REFUSED-IDENTITY", f"embedding rank drift for seed {seed}")
        if twin.shape[1] != abnormal.shape[1]:
            raise _refused("REFUSED-IDENTITY", f"embedding width drift for seed {seed}")
        if twin.shape[0] == 0 or abnormal.shape[0] == 0:
            raise _refused("REFUSED-IDENTITY", f"empty embedding arm for seed {seed}")
        if not bool(np.isfinite(twin).all()) or not bool(np.isfinite(abnormal).all()):
            raise _refused("REFUSED-ERROR-BOUNDARY", f"non-finite embeddings for seed {seed}")
        matrices[int(seed)] = {"twin": twin, "abnormal": abnormal, "width": int(twin.shape[1])}
    widths = {int(value["width"]) for value in matrices.values()}
    if len(widths) != 1:
        raise _refused("REFUSED-IDENTITY", "encoder widths disagree across seeds")
    return matrices


# --- stage-1 evaluation --------------------------------------------------------

#: One scored group outcome (binary response + raw E1 values).
@dataclass(frozen=True)
class GroupOutcome:
    group_index: int
    pair_indices: tuple[int, ...]
    e1_per_seed: tuple[float, ...]
    median_e1: float
    null_q95_median: float
    response: bool


def paired_mmd2_unbiased(twin: Any, abnormal: Any, sigma: float) -> float:
    """Estimate marginal MMD² with independent source-pairs, excluding matched cross terms."""
    import numpy as np
    from representation.sprint24_weekly_monthly_endpoints import _kernel

    x = np.asarray(twin, dtype=np.float64)
    y = np.asarray(abnormal, dtype=np.float64)
    if (x.ndim != 2 or y.shape != x.shape or x.shape[0] < 2
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not math.isfinite(sigma) or sigma <= 0):
        raise _refused("REFUSED-ERROR-BOUNDARY", "paired MMD requires finite matched matrices and positive bandwidth")
    kxx, kyy, kxy = _kernel(x, x, sigma), _kernel(y, y, sigma), _kernel(x, y, sigma)
    n = x.shape[0]
    return float(((kxx.sum() - np.trace(kxx)) + (kyy.sum() - np.trace(kyy))
                  - 2.0 * (kxy.sum() - np.trace(kxy))) / (n * (n - 1)))


def score_cell_groups(
    matrices: Mapping[int, dict[str, Any]],
    pair_indices: Sequence[int],
    null_q95_per_seed: Mapping[int, float],
    sigma_per_seed: Mapping[int, float],
    group_size_k: int = base.GROUP_SIZE_K,
) -> list[GroupOutcome]:
    """Score every K=8 group of one cell slice with frozen null responses.

    Bandwidth arrives frozen per encoder seed (median pairwise distance on the
    causal reference pool); this function never re-estimates sigma from test cells.
    """
    import numpy as np

    indices = [int(value) for value in pair_indices]
    if len(indices) % int(group_size_k) != 0:
        raise _refused("REFUSED-ERROR-BOUNDARY", "pair count is not a multiple of K=8")
    for seed in base.FROZEN_ENCODER_SEEDS:
        sigma_check = float(sigma_per_seed[int(seed)])
        if not math.isfinite(sigma_check) or sigma_check <= 0:
            raise _refused("REFUSED-ERROR-BOUNDARY", "frozen reference bandwidth is non-positive")
    null_median = float(np.median([float(null_q95_per_seed[int(seed)]) for seed in base.FROZEN_ENCODER_SEEDS]))
    outcomes: list[GroupOutcome] = []
    for group_index, start in enumerate(range(0, len(indices), int(group_size_k))):
        sl = slice(start, start + int(group_size_k))
        per_seed_e1: list[float] = []
        for seed in base.FROZEN_ENCODER_SEEDS:
            twin = np.asarray(matrices[int(seed)]["twin"])[sl]
            abnormal = np.asarray(matrices[int(seed)]["abnormal"])[sl]
            per_seed_e1.append(paired_mmd2_unbiased(twin, abnormal, float(sigma_per_seed[int(seed)])))
        median_e1 = float(np.median(per_seed_e1))
        outcomes.append(
            GroupOutcome(
                group_index=int(group_index),
                pair_indices=tuple(indices[start : start + int(group_size_k)]),
                e1_per_seed=tuple(float(value) for value in per_seed_e1),
                median_e1=float(median_e1),
                null_q95_median=float(null_median),
                response=bool(median_e1 > null_median),
            )
        )
    return outcomes

def capture_reference_pool(
    bundles: Mapping[int, Any],
    history_seed: int,
    history_id: str,
    device: str | None = None,
    journal: CaptureJournal | None = None,
    source_role: str = "reference",
) -> dict[int, dict[str, Any]]:
    """Capture the original reference or separate bandwidth source pool."""
    import numpy as np

    from synth.config import SynthConfig
    from synth.generator import SessionGenerator
    from synth.schema import AnomalyFamily

    gen = SessionGenerator(SynthConfig())
    if source_role == "reference":
        seeds, index_base = generate_reference_pool(int(history_seed)), REFERENCE_INDEX_BASE
    elif source_role == "bandwidth":
        seeds, index_base = generate_bandwidth_pool(int(history_seed)), BANDWIDTH_INDEX_BASE
    else:
        raise _refused("REFUSED-UNKNOWN-DESIGN", f"unknown source role: {source_role}")
    if len(set(seeds)) != len(seeds):
        raise _refused("REFUSED-SEED-COLLISION", f"{source_role} source seeds are not distinct")
    if set(generate_reference_pool(int(history_seed))) & set(generate_bandwidth_pool(int(history_seed))):
        raise _refused("REFUSED-SEED-COLLISION", "reference and bandwidth source seeds overlap")
    pairs = [_generate_with_custody(gen, value, AnomalyFamily.SUBTLE_DRIFT, 0.6,
                                    index_base + index, history_id, source_role, journal)
             for index, value in enumerate(seeds)]
    from representation.sprint24_controlled_data_execution import build_history_calendar
    calendar = build_history_calendar(history_id)
    planned = [{**_pre_c1_lineage(pair, index_base + index, source_role, float(calendar["c1_rel_s"])),
                "scenario_id": f"{history_id}|{source_role}"}
               for index, pair in enumerate(pairs)]
    if journal is not None:
        journal.record(f"{source_role}_pairs", {"history_id": history_id, "history_seed": history_seed,
                       "scenario_id": f"{history_id}|{source_role}", "calendar": calendar,
                       "pairs": planned})
    twins = [pair.twin for pair in pairs]
    per_seed = capture_files(bundles, twins, str(history_id), device=device, journal=journal)
    reference: dict[int, dict[str, Any]] = {}
    for seed in base.FROZEN_ENCODER_SEEDS:
        captured = list(per_seed[int(seed)])
        matrix = np.stack(
            [np.asarray(item.embedding.detach().cpu().float().numpy()) for item in captured]
        ).astype(np.float64)
        file_ids = [str(item.file_id) for item in captured]
        reference[int(seed)] = {"matrix": matrix, "file_ids": file_ids, "n_reference": int(matrix.shape[0])}
    return reference


def frozen_reference_bandwidth(reference: Mapping[int, dict[str, Any]]) -> dict[int, float]:
    """Frozen sigma per encoder seed: median pairwise distance on the reference pool."""
    from scipy.spatial.distance import pdist

    sigma: dict[int, float] = {}
    for seed in base.FROZEN_ENCODER_SEEDS:
        matrix = dict(reference[int(seed)])["matrix"]
        value = float(__import__("numpy").median(pdist(matrix)))
        if not math.isfinite(value) or value <= 0:
            raise _refused("REFUSED-ERROR-BOUNDARY", "non-finite reference bandwidth")
        sigma[int(seed)] = float(value)
    return sigma


def fit_null_q95(
    bundles: Mapping[int, Any],
    history_seed: int,
    history_id: str,
    sigma_per_seed: Mapping[int, float] | None = None,
    device: str | None = None,
    journal: CaptureJournal | None = None,
) -> dict[int, float]:
    """Fit the frozen matched-size null q95 per encoder seed (twin-vs-twin).

    Null groups are twin-vs-twin halves from the disjoint null band, scored
    with the FROZEN reference bandwidth (passed in, never re-estimated here).
    """
    import numpy as np

    from synth.config import SynthConfig
    from synth.generator import SessionGenerator
    from synth.schema import AnomalyFamily

    gen = SessionGenerator(SynthConfig())
    seeds = generate_null_pool(int(history_seed))
    reserved = set(generate_reference_pool(int(history_seed))) | set(generate_bandwidth_pool(int(history_seed)))
    if len(set(seeds)) != len(seeds) or set(seeds) & reserved:
        raise _refused("REFUSED-SEED-COLLISION", "null source seeds collide with reserved sources")
    pairs = [_generate_with_custody(gen, value, AnomalyFamily.SUBTLE_DRIFT, 0.6,
                                    NULL_INDEX_BASE + index, history_id, "null", journal)
             for index, value in enumerate(seeds)]
    from representation.sprint24_controlled_data_execution import build_history_calendar
    calendar = build_history_calendar(history_id)
    planned = [{**_pre_c1_lineage(pair, NULL_INDEX_BASE + index, "null", float(calendar["c1_rel_s"])),
                "scenario_id": f"{history_id}|null"}
               for index, pair in enumerate(pairs)]
    if journal is not None:
        journal.record("null_pairs", {"history_id": history_id, "history_seed": history_seed,
                       "scenario_id": f"{history_id}|null", "calendar": calendar, "pairs": planned})
    twins = [pair.twin for pair in pairs]
    per_seed = capture_files(bundles, twins, str(history_id), device=device, journal=journal)
    thresholds: dict[int, float] = {}
    for seed in base.FROZEN_ENCODER_SEEDS:
        captured = list(per_seed[int(seed)])
        matrix = np.stack(
            [np.asarray(item.embedding.detach().cpu().float().numpy()) for item in captured]
        ).astype(np.float64)
        sigma = float(sigma_per_seed[int(seed)]) if sigma_per_seed is not None else None
        if sigma is None:
            raise _refused("REFUSED-ERROR-BOUNDARY", "frozen reference bandwidth is required for the null fit")
        if not math.isfinite(sigma) or sigma <= 0:
            raise _refused("REFUSED-ERROR-BOUNDARY", "non-finite null bandwidth")
        values: list[float] = []
        for group in range(NULL_GROUPS_PER_CELL):
            first = slice(group * base.GROUP_SIZE_K, (group + 1) * base.GROUP_SIZE_K)
            second = slice(
                (group + NULL_GROUPS_PER_CELL) * base.GROUP_SIZE_K,
                (group + NULL_GROUPS_PER_CELL + 1) * base.GROUP_SIZE_K,
            )
            values.append(paired_mmd2_unbiased(matrix[first], matrix[second], sigma))
        thresholds[int(seed)] = float(np.quantile(values, 0.95))
        if journal is not None:
            journal.record("null_scores", {"history_id": history_id, "model_seed": int(seed),
                           "sigma": sigma, "scores": values, "q95": thresholds[int(seed)]})
    return thresholds



# --- sensitivity estimators ---------------------------------------------------

D_MIN_COVERAGE = 0.5


def e2_occupancy(
    twin: Any,
    abnormal: Any,
    reference: Any,
    k: int = base.PRIMARY_K,
    seed: int = CLUSTER_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Total-variation distance between K-prototype occupancies (frozen KMeans)."""
    import numpy as np
    from sklearn.cluster import KMeans

    twin = np.asarray(twin, dtype=np.float64)
    abnormal = np.asarray(abnormal, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    model = KMeans(n_clusters=int(k), random_state=int(seed), n_init=10).fit(reference)
    labels_twin = model.predict(twin)
    labels_abnormal = model.predict(abnormal)
    counts_twin = np.bincount(labels_twin, minlength=int(k)).astype(np.float64)
    counts_abnormal = np.bincount(labels_abnormal, minlength=int(k)).astype(np.float64)
    occupancy_twin = counts_twin / counts_twin.sum()
    occupancy_abnormal = counts_abnormal / counts_abnormal.sum()
    return {
        "e2": float(0.5 * np.abs(occupancy_twin - occupancy_abnormal).sum()),
        "k": int(k),
        "seed": int(seed),
    }


def within_prototype_diagnostics(
    twin: Any,
    abnormal: Any,
    reference: Any,
    k: int = base.PRIMARY_K,
    seed: int = CLUSTER_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Within-prototype location/spread diagnostics (M14 math, frozen floor)."""
    import numpy as np
    from sklearn.cluster import KMeans

    twin = np.asarray(twin, dtype=np.float64)
    abnormal = np.asarray(abnormal, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    model = KMeans(n_clusters=int(k), random_state=int(seed), n_init=10).fit(reference)
    centroids = np.asarray(model.cluster_centers_, dtype=np.float64)
    labels_twin = model.predict(twin)
    labels_abnormal = model.predict(abnormal)
    size = int(twin.shape[0])
    one_hot_twin = np.zeros((size, int(k)), dtype=np.float64)
    one_hot_twin[np.arange(size), labels_twin] = 1.0
    one_hot_abnormal = np.zeros((size, int(k)), dtype=np.float64)
    one_hot_abnormal[np.arange(size), labels_abnormal] = 1.0
    counts_twin = one_hot_twin.sum(axis=0)
    counts_abnormal = one_hot_abnormal.sum(axis=0)
    mean_twin = np.divide(
        one_hot_twin.T @ twin, counts_twin[:, None], out=np.zeros_like(centroids), where=counts_twin[:, None] > 0
    )
    mean_abnormal = np.divide(
        one_hot_abnormal.T @ abnormal,
        counts_abnormal[:, None],
        out=np.zeros_like(centroids),
        where=counts_abnormal[:, None] > 0,
    )
    residual_reference = reference - centroids[model.predict(reference)]
    ref_scale = float(np.mean(np.einsum("ij,ij->i", residual_reference, residual_reference)))
    if not math.isfinite(ref_scale) or ref_scale <= 0:
        raise _refused("REFUSED-ERROR-BOUNDARY", "non-positive reference residual scale")
    difference = mean_twin - mean_abnormal
    norm_sq = np.einsum("kd,kd->k", difference, difference)
    common = (counts_twin > 0) & (counts_abnormal > 0)
    mass = np.where(common, (counts_twin + counts_abnormal) / (2.0 * size), 0.0)
    coverage = float(mass.sum())
    weight = mass / coverage if coverage > 0 else mass
    valid = bool(coverage >= D_MIN_COVERAGE)
    return {
        "d_loc": float((weight * norm_sq).sum() / ref_scale) if valid else float("nan"),
        "coverage": float(coverage),
        "valid": bool(valid),
        "d_min_coverage": D_MIN_COVERAGE,
        "k": int(k),
    }


# --- stage-2 population evaluation --------------------------------------------

#: Stage-2 arm manifest (independent population arms, disjoint seeds).
@dataclass(frozen=True)
class PopulationArm:
    horizon: str
    size: int
    burden: str
    n_abnormal: int
    n_ordinary: int
    pair_indices: tuple[int, ...]


def population_arms(pair_index_base: int = STAGE2_INDEX_BASE) -> list[PopulationArm]:
    """Enumerate every weekly/monthly x burden population arm (frozen counts)."""
    ladder = base.dilution_ladder()
    arms: list[PopulationArm] = []
    cursor = int(pair_index_base)
    for horizon in ("week", "month"):
        for size_text in sorted(ladder[horizon], key=int):
            for burden in ("LOW", "MID", "ELEVATED"):
                counts = ladder[horizon][size_text][burden]
                size = int(counts["arm_size"])
                arms.append(
                    PopulationArm(
                        horizon=str(horizon),
                        size=int(size),
                        burden=str(burden),
                        n_abnormal=int(counts["n_abnormal"]),
                        n_ordinary=int(counts["n_ordinary"]),
                        pair_indices=tuple(range(cursor, cursor + size)),
                    )
                )
                cursor += size
    return arms


def cluster_bootstrap_median(
    history_values: Sequence[float],
    replicates: int = CLUSTER_BOOTSTRAP_REPLICATES,
    seed: int = CLUSTER_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """History-cluster bootstrap of the median (descriptive, unit = history)."""
    import numpy as np

    values = [float(value) for value in history_values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"status": "UNAVAILABLE", "reason": "no estimable histories"}
    rng = np.random.default_rng(int(seed))
    draws = np.asarray(values, dtype=np.float64)
    medians = [
        float(np.median(draws[rng.integers(0, len(draws), size=len(draws))])) for _ in range(int(replicates))
    ]
    return {
        "status": "DESCRIPTIVE",
        "unit": "history",
        "replicates": int(replicates),
        "seed": int(seed),
        "point_median": float(np.median(draws)),
        "ci95_low": float(np.quantile(medians, 0.025)),
        "ci95_high": float(np.quantile(medians, 0.975)),
        "n_histories": len(draws),
    }


# --- official contract (C2, versioned) -----------------------------------------

#: Tiny-fixture mode grid (explicit smoke parameterization of the same engine).
TINY_FIXTURE = {
    "family": "subtle_drift",
    "severity": 0.6,
    "history_seed": 53101,
    "n_groups": 2,
    "null_groups": 2,
    "stage2_arms": [("week", 40, "ELEVATED")],
}


def build_official_contract(
    preflight: Mapping[str, Any],
    provenance: Mapping[str, Any],
    false_alarm_anchors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the versioned C2 official contract (deterministic, fail-closed)."""
    if not isinstance(preflight, Mapping) or not preflight:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "verified preflight is required")
    if not isinstance(provenance, Mapping) or not provenance:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "provenance is required")
    matrix = matrix_plan()
    grid = base.group_power_grid()
    cp_reference = base.clopper_pearson_rows()
    contract = {
        "contract_id": OFFICIAL_CONTRACT_ID,
        "supersedes": SUPERSEDED_CONTRACT_ID,
        "supersession": (
            "A01 (S24-PAIRED-EXECUTION-CONTRACT-A01) is preserved byte-identical as "
            "the immutable historical record; C2 authorizes the complete official "
            "matrix path that A01 explicitly forbade. A01 forbids official runs; "
            "C2 permits them only after C2 evidence PASS + Bronze checkpoint + M18 release."
        ),
        "state": "FROZEN-EXECUTABLE",
        "not_an_official_release": False,
        "release_condition": "M17 C2 evidence PASS + Bronze checkpoint + M18 release",
        "readiness_vs_permission": (
            "this contract freezes WHAT the official run must execute; it does not "
            "itself authorize WHEN the official full-matrix inference may run."
        ),
        "execution": base.execution_summary() | {"official_contract_id": OFFICIAL_CONTRACT_ID},
        "official_matrix": matrix,
        "allocation": allocation_table(),
        "group_operating_characteristics": grid,
        "clopper_pearson_reference": cp_reference,
        "sequential_multiplicity": base.sequential_multiplicity_plan(),
        "dilution_ladder": base.dilution_ladder(),
        "calibration": base.calibration_spec(),
        "evaluation": base.evaluation_spec(),
        "sensitivity": {
            "occupancy": "E2_K8 total-variation on frozen KMeans K=8 prototypes",
            "location": "D_LOC within-prototype diagnostic, coverage floor 0.5",
            "cluster_bootstrap": {"replicates": CLUSTER_BOOTSTRAP_REPLICATES, "seed": CLUSTER_BOOTSTRAP_SEED},
        },
        "runtime_pins": base.runtime_pins(preflight),
        "owner_approvals": [
            "option A: unbiased E1 on independent K=8 scored groups, 24->48 groups/cell",
            "no 96 branch: any G>48 needs separate explicit owner authorization",
            "0.5 = response fraction, not raw E1; misses are inconclusive",
            "duration within-family scope only",
        ],
        "tiny_fixture": dict(TINY_FIXTURE),
        "false_alarm_anchors": dict(false_alarm_anchors or {}),
        "provenance": dict(provenance),
    }
    contract["contract_sha256"] = hashlib.sha256(
        (canonical_json(dict(contract)) + "\n").encode("utf-8")
    ).hexdigest()
    return contract


def verify_official_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the contract is the deterministic C2 build output."""
    if not isinstance(contract, Mapping):
        raise _refused("REFUSED-CONTRACT-DRIFT", "official contract is not a mapping")
    if contract.get("contract_id") != OFFICIAL_CONTRACT_ID:
        raise _refused("REFUSED-CONTRACT-DRIFT", "official contract ID drifted")
    if contract.get("supersedes") != SUPERSEDED_CONTRACT_ID:
        raise _refused("REFUSED-CONTRACT-DRIFT", "supersession pointer drifted")
    recorded = str(contract.get("contract_sha256"))
    candidate = dict(contract)
    candidate.pop("contract_sha256", None)
    recomputed = hashlib.sha256((canonical_json(candidate) + "\n").encode("utf-8")).hexdigest()
    if recorded != recomputed:
        raise _refused("REFUSED-CONTRACT-DRIFT", "official contract digest drifted")
    matrix = contract.get("official_matrix") or {}
    cells = matrix.get("cells") or []
    if len(cells) != 27:
        raise _refused("REFUSED-CONTRACT-DRIFT", "official matrix must hold 27 cells")
    grid = list((contract.get("execution") or {}).get("stage1_group_grid") or [])
    if grid != [24, 48]:
        raise _refused("REFUSED-CONTRACT-DRIFT", "official grid must be exactly [24, 48]")
    return {"status": "PASS", "contract_sha256": recorded}


# --- tiny fixture smoke (same engine, smoke parameterization) ------------------

#: Tiny-fixture smoke record (same engine, explicit smoke parameterization).
@dataclass
class TinySmokeResult:
    status: str
    family: str
    severity: float
    history_seed: int
    n_groups: int
    per_seed_e1: dict[str, list[float]]
    per_seed_q95: dict[str, float]
    cell_summary: dict[str, Any]
    disjoint_test: dict[str, Any]
    stage2_arms: list[dict[str, Any]]
    forwards_executed: int
    model_loads: int
    elapsed_s: float


def collect_stage1_cell(
    bundles: Mapping[int, Any],
    history_seed: int,
    history_id: str,
    family: str,
    severity: float,
    pair_indices: Sequence[int],
    reference: Mapping[int, dict[str, Any]],
    sigma_per_seed: Mapping[int, float],
    null_q95_per_seed: Mapping[int, float],
    device: str | None = None,
    journal: CaptureJournal | None = None,
) -> dict[str, Any]:
    """Execute one cell slice through the official stage-1 engine (shared path).

    Generation (frozen pairing), capture (public frozen entry), frozen-bandwidth
    scoring, matched null responses, sensitivity diagnostics and abstention
    denominators all run here; the tiny fixture and the full orchestrator call
    this same function.
    """
    import numpy as np

    target = _family_of(family)
    from synth.config import SynthConfig
    from synth.generator import SessionGenerator, derive_pair_seed

    gen = SessionGenerator(SynthConfig())
    indices = [int(value) for value in pair_indices]
    pair_seeds = [int(derive_pair_seed(int(history_seed), index)) for index in indices]
    if (len(set(pair_seeds)) != len(pair_seeds)
            or set(pair_seeds) & (set(generate_reference_pool(int(history_seed))) | set(generate_null_pool(int(history_seed))) | set(generate_bandwidth_pool(int(history_seed))))):
        raise _refused("REFUSED-SEED-COLLISION", "scored sources collide with each other or reference/null")
    pairs = [_generate_with_custody(gen, value, target, severity, index, history_id, "stage1", journal)
             for index, value in zip(indices, pair_seeds)]
    from representation.sprint24_controlled_data_execution import build_history_calendar
    calendar = build_history_calendar(history_id)
    weeks = calendar["weeks"]
    scenario_id = f"{history_id}|stage1|{family}|{severity:.9g}"
    lineage = [
        _evaluation_lineage(pair, index, scenario_id, "scored-pair",
                            weeks[1 + index // 40], index % 40, 40, float(calendar["c1_rel_s"]))
        for index, pair in zip(indices, pairs)
    ]
    paired_indices = [index for index, pair in zip(indices, pairs) if pair.disposition == "PAIRED"]
    if journal is not None:
        journal.record("stage1_pairs", {"history_id": history_id, "history_seed": history_seed,
                       "scenario_id": scenario_id, "family": family, "severity": severity,
                       "calendar": calendar, "pairs": lineage})
    dispositions: dict[str, int] = {}
    for pair in pairs:
        dispositions[str(pair.disposition)] = int(dispositions.get(str(pair.disposition), 0)) + 1
    paired = [pair for pair in pairs if pair.disposition == "PAIRED"]
    twin_samples = [pair.twin for pair in paired]
    abnormal_samples = [pair.abnormal for pair in paired]
    n_paired = len(paired)
    n_groups_complete = n_paired // base.GROUP_SIZE_K
    usable = n_groups_complete * base.GROUP_SIZE_K
    per_seed_capture: dict[int, Any] = {}
    matrices: dict[int, dict[str, Any]] = {}
    outcomes: list[GroupOutcome] = []
    if n_paired:
        per_seed_capture = capture_files(bundles, twin_samples + abnormal_samples, str(history_id), device=device, journal=journal)
        twin_ids = [sample.file_id for sample in twin_samples]
        abnormal_ids = [sample.file_id for sample in abnormal_samples]
        matrices = embedding_matrices(per_seed_capture, twin_ids, abnormal_ids)
        if usable >= base.GROUP_SIZE_K:
            outcomes = score_cell_groups(matrices, paired_indices[:usable], null_q95_per_seed, sigma_per_seed)
    from representation.sprint24_paired_disjoint import source_disjoint_test
    disjoint_test = source_disjoint_test(
        indices, paired_indices, matrices, sigma_per_seed, int(history_seed), family, severity,
    )
    if journal is not None:
        journal.record("stage1_disjoint_test", {
            "history_id": history_id, "history_seed": history_seed, "family": family,
            "severity": severity, "result": disjoint_test,
        })
    if journal is not None:
        journal.record("stage1_groups", {
            "history_id": history_id, "family": family, "severity": severity,
            "paired_unscored_indices": paired_indices[usable:],
            "outcomes": [{"group_index": value.group_index, "pair_indices": list(value.pair_indices),
                          "e1_per_seed": list(value.e1_per_seed), "median_e1": value.median_e1,
                          "null_q95_median": value.null_q95_median, "response": value.response}
                         for value in outcomes],
        })
    # Sensitivity diagnostics on the first complete group (frozen reference prototypes).
    sensitivity: dict[str, Any] = {"status": "UNAVAILABLE", "reason": "no complete group"}
    if usable >= base.GROUP_SIZE_K:
        reference_matrix = np.asarray(dict(reference[int(base.FROZEN_ENCODER_SEEDS[0])])["matrix"])
        twin = np.asarray(matrices[int(base.FROZEN_ENCODER_SEEDS[0])]["twin"])[: base.GROUP_SIZE_K]
        abnormal = np.asarray(matrices[int(base.FROZEN_ENCODER_SEEDS[0])]["abnormal"])[: base.GROUP_SIZE_K]
        sensitivity = {
            "e2": e2_occupancy(twin, abnormal, reference_matrix),
            "d": within_prototype_diagnostics(twin, abnormal, reference_matrix),
        }
    if journal is not None:
        clean_sensitivity = dict(sensitivity)
        if "d" in clean_sensitivity and not math.isfinite(float(clean_sensitivity["d"]["d_loc"])):
            clean_sensitivity["d"] = {**clean_sensitivity["d"], "d_loc": None}
        journal.record("stage1_sensitivity", {"history_id": history_id, "family": family,
                       "severity": severity, "sensitivity": clean_sensitivity})
    return {
        "n_planned_pairs": len(indices),
        "dispositions": dispositions,
        "n_paired": int(n_paired),
        "n_groups": int(n_groups_complete),
        "pairs": lineage,
        "disjoint_test": disjoint_test,
        "paired_unscored_indices": paired_indices[usable:],
        "outcomes": [
            {
                "group_index": value.group_index,
                "pair_indices": list(value.pair_indices),
                "e1_per_seed": dict(zip(map(str, base.FROZEN_ENCODER_SEEDS), value.e1_per_seed)),
                "null_q95_median": value.null_q95_median,
                "median_e1": value.median_e1,
                "response": value.response,
            }
            for value in outcomes
        ],
        "sensitivity": sensitivity,
    }


def evaluate_stage2_arm(
    bundles: Mapping[int, Any],
    history_seed: int,
    history_id: str,
    family: str,
    severity: float,
    arm: PopulationArm,
    arm_cursor: int,
    reference: Mapping[int, dict[str, Any]],
    sigma_per_seed: Mapping[int, float],
    device: str | None = None,
    journal: CaptureJournal | None = None,
) -> dict[str, Any]:
    """Execute one stage-2 population arm through the official engine.

    Full-N mixture estimand: P_elevated (all n_abnormal abnormal files +
    n_ordinary ordinary files from a disjoint slice) vs P_control (N ordinary
    files from a second disjoint slice), each scored as full-N unbiased E1
    against the causal-reference pool with the frozen reference sigma. Never
    truncated to [:8]; never abnormal-only or ordinary-only.
    """
    import numpy as np

    target = _family_of(family)
    from synth.config import SynthConfig
    from synth.generator import SessionGenerator, derive_pair_seed

    gen = SessionGenerator(SynthConfig())
    size = int(arm.size)
    abnormal_indices = list(range(int(arm_cursor), int(arm_cursor) + int(arm.n_abnormal)))
    ordinary_indices = list(
        range(int(arm_cursor) + int(arm.n_abnormal), int(arm_cursor) + int(arm.n_abnormal) + int(arm.n_ordinary))
    )
    control_indices = list(
        range(int(arm_cursor) + int(arm.n_abnormal) + int(arm.n_ordinary), int(arm_cursor) + int(arm.n_abnormal) + int(arm.n_ordinary) + size)
    )
    abnormal_seeds = [int(derive_pair_seed(int(history_seed), STAGE2_INDEX_BASE + index)) for index in abnormal_indices]
    ordinary_seeds = [int(derive_pair_seed(int(history_seed), STAGE2_INDEX_BASE + index)) for index in ordinary_indices]
    control_seeds = [int(derive_pair_seed(int(history_seed), STAGE2_INDEX_BASE + index)) for index in control_indices]
    population_seeds = abnormal_seeds + ordinary_seeds + control_seeds
    reserved = set(generate_reference_pool(int(history_seed))) | set(generate_null_pool(int(history_seed))) | set(generate_bandwidth_pool(int(history_seed)))
    reserved.update(int(derive_pair_seed(int(history_seed), index)) for index in range(PAIR_SLOTS_PER_HISTORY))
    if len(set(population_seeds)) != len(population_seeds) or set(population_seeds) & reserved:
        raise _refused("REFUSED-SEED-COLLISION", "population source seeds collide internally or with reserved sources")
    abnormal_pairs = [_generate_with_custody(gen, value, target, severity, STAGE2_INDEX_BASE + index,
                                              history_id, "mixture-abnormal", journal)
                      for index, value in zip(abnormal_indices, abnormal_seeds)]
    ordinary_pairs = [_generate_with_custody(gen, value, target, severity, STAGE2_INDEX_BASE + index,
                                              history_id, "mixture-ordinary", journal)
                      for index, value in zip(ordinary_indices, ordinary_seeds)]
    control_pairs = [_generate_with_custody(gen, value, target, severity, STAGE2_INDEX_BASE + index,
                                             history_id, "control", journal)
                     for index, value in zip(control_indices, control_seeds)]
    from representation.sprint24_controlled_data_execution import build_history_calendar
    calendar = build_history_calendar(history_id)
    window = calendar["weeks"][1] if arm.horizon == "week" else calendar["months"][1]
    stem = f"{history_id}|stage2|{arm.horizon}|{arm.size}|{arm.burden}"
    mixture_scenario, control_scenario = stem + "|mixture", stem + "|control"
    abnormal_lineage = [
        _evaluation_lineage(pair, STAGE2_INDEX_BASE + index, mixture_scenario, "mixture-abnormal",
                            window, slot, size, float(calendar["c1_rel_s"]))
        for slot, (pair, index) in enumerate(zip(abnormal_pairs, abnormal_indices))
    ]
    ordinary_lineage = [
        _evaluation_lineage(pair, STAGE2_INDEX_BASE + index, mixture_scenario, "mixture-ordinary",
                            window, len(abnormal_pairs) + slot, size, float(calendar["c1_rel_s"]))
        for slot, (pair, index) in enumerate(zip(ordinary_pairs, ordinary_indices))
    ]
    control_lineage = [
        _evaluation_lineage(pair, STAGE2_INDEX_BASE + index, control_scenario, "control",
                            window, slot, size, float(calendar["c1_rel_s"]))
        for slot, (pair, index) in enumerate(zip(control_pairs, control_indices))
    ]
    if journal is not None:
        journal.record("stage2_pairs", {
            "history_id": history_id, "history_seed": history_seed, "family": family,
            "severity": severity, "horizon": arm.horizon, "size": arm.size, "burden": arm.burden,
            "mixture_scenario_id": mixture_scenario, "control_scenario_id": control_scenario,
            "calendar": calendar, "abnormal": abnormal_lineage,
            "ordinary": ordinary_lineage, "control": control_lineage,
        })
    dispositions: dict[str, int] = {}
    for pair in abnormal_pairs + ordinary_pairs + control_pairs:
        dispositions[str(pair.disposition)] = int(dispositions.get(str(pair.disposition), 0)) + 1
    abnormal_samples = [pair.abnormal for pair in abnormal_pairs if pair.disposition == "PAIRED"]
    ordinary_samples = [pair.twin for pair in ordinary_pairs if pair.disposition == "PAIRED"]
    control_samples = [pair.twin for pair in control_pairs if pair.disposition == "PAIRED"]
    if (len(abnormal_samples) == 1 or len(ordinary_samples) == 1
            or len(abnormal_samples) + len(ordinary_samples) < 2 or len(control_samples) < 2):
        if journal is not None:
            journal.record("stage2_abstention", {
                "history_id": history_id, "horizon": arm.horizon, "size": arm.size,
                "burden": arm.burden, "n_mixture": len(abnormal_samples) + len(ordinary_samples),
                "n_control": len(control_samples), "reason": "singleton nonzero component or insufficient observations",
            })
        return {
            "horizon": arm.horizon, "size": arm.size, "burden": arm.burden,
            "n_abnormal": int(arm.n_abnormal), "n_ordinary": int(arm.n_ordinary),
            "n_abnormal_paired": len(abnormal_samples), "n_ordinary_paired": len(ordinary_samples),
            "n_control_paired": len(control_samples),
            "dispositions": dispositions, "status": "ABSTAIN-COVERAGE",
            "reason": "singleton nonzero component or insufficient observations",
        }
    mixture_samples = abnormal_samples + ordinary_samples
    captured = capture_files(bundles, mixture_samples + control_samples, str(history_id), device=device, journal=journal)
    mixture_ids = [sample.file_id for sample in mixture_samples]
    control_ids = [sample.file_id for sample in control_samples]
    matrices = embedding_matrices(captured, mixture_ids, control_ids)
    per_seed: dict[str, Any] = {}
    for seed in base.FROZEN_ENCODER_SEEDS:
        mixture = np.asarray(matrices[int(seed)]["twin"])
        control = np.asarray(matrices[int(seed)]["abnormal"])
        reference_matrix = np.asarray(dict(reference[int(seed)])["matrix"])
        sigma = float(sigma_per_seed[int(seed)])
        n_mix = int(mixture.shape[0])
        n_ctl = int(control.shape[0])
        n_ref = int(reference_matrix.shape[0])
        from representation.sprint24_weekly_monthly_endpoints import mmd2_unbiased as _full_n_mmd2
        if n_ref < 2:
            raise _refused("REFUSED-ERROR-BOUNDARY", "unbiased MMD requires at least two reference observations")
        reference_ids = list(dict(reference[int(seed)])["file_ids"])
        if len(reference_ids) != n_ref or len(set(reference_ids)) != n_ref:
            raise _refused("REFUSED-IDENTITY", "reference observations must have distinct file IDs")
        if set(reference_ids) & (set(mixture_ids) | set(control_ids)):
            raise _refused("REFUSED-IDENTITY", "reference and population file IDs overlap")
        from representation.sprint24_paired_estimators import stratified_mmd2_unbiased
        n_abnormal = len(abnormal_samples)
        e1_transition = stratified_mmd2_unbiased(
            mixture[:n_abnormal], mixture[n_abnormal:], reference_matrix, sigma,
        )
        e1_control = float(_full_n_mmd2(control, reference_matrix, sigma))
        per_seed[str(seed)] = {
            "e1_transition": e1_transition, "e1_control": e1_control,
            "n_mixture": n_mix, "n_control": n_ctl, "n_reference": n_ref,
        }
    if journal is not None:
        journal.record("stage2_scores", {
            "history_id": history_id, "horizon": arm.horizon, "size": arm.size,
            "burden": arm.burden, "mixture_ids": mixture_ids, "control_ids": control_ids,
            "estimator": "stratified-realized-mixture-mmd2-v1",
            "n_abnormal_paired": len(abnormal_samples), "n_ordinary_paired": len(ordinary_samples),
            "realized_abnormal_fraction": len(abnormal_samples) / len(mixture_samples),
            "per_seed": per_seed,
        })
    return {
        "horizon": arm.horizon, "size": arm.size, "burden": arm.burden,
        "n_abnormal": int(arm.n_abnormal), "n_ordinary": int(arm.n_ordinary),
        "n_abnormal_paired": len(abnormal_samples), "n_ordinary_paired": len(ordinary_samples),
        "realized_abnormal_fraction": len(abnormal_samples) / len(mixture_samples),
        "n_control_paired": len(control_samples),
        "mixture_ids": [sample.file_id for sample in mixture_samples],
        "control_ids": [sample.file_id for sample in control_samples],
        "dispositions": dispositions, "status": "COMPUTED", "per_seed": per_seed,
    }


def descriptive_cell_summary(values: Sequence[float], null_median: float) -> dict[str, Any]:
    """Describe responses without asserting unvalidated calibrated detection."""
    if not values:
        return {"n_groups": 0, "n_responses": 0, "status": "ABSTAIN-COVERAGE",
                "reason": "no complete scored group"}
    summary = base.summarize_cell(values, null_median)
    summary["bar_met"] = None
    summary["reading"] = ("DESCRIPTIVE ONLY: calibration and conditional independence "
                          "are not established; this interval is not a detection claim.")
    return summary


def tiny_official_smoke(
    bundles: Mapping[int, Any],
    device: str | None = None,
    fixture: Mapping[str, Any] | None = None,
    journal: CaptureJournal | None = None,
) -> TinySmokeResult:
    """Exercise the official engine on tiny real-checkpoint/bank inputs (shared path)."""
    import numpy as np

    started = time.time()
    options = dict(TINY_FIXTURE)
    if fixture:
        options.update(dict(fixture))
    family = str(options["family"])
    severity = float(options["severity"])
    history_seed = int(options["history_seed"])
    n_groups = int(options["n_groups"])
    history_id = "S24P-TINY-OFFICIAL"
    reference = capture_reference_pool(bundles, history_seed, history_id + "-REF", device=device, journal=journal)
    bandwidth = capture_reference_pool(bundles, history_seed, history_id + "-BANDWIDTH",
                                       device=device, journal=journal, source_role="bandwidth")
    sigma_per_seed = frozen_reference_bandwidth(bandwidth)
    null_q95 = fit_null_q95(bundles, history_seed, history_id + "-NULL", sigma_per_seed, device=device, journal=journal)
    cell = collect_stage1_cell(
        bundles, history_seed, history_id, family, severity,
        list(range(n_groups * base.GROUP_SIZE_K)), reference, sigma_per_seed, null_q95, device=device, journal=journal,
    )
    outcomes_median = [float(entry["median_e1"]) for entry in cell["outcomes"]]
    null_median = float(np.median([float(null_q95[int(seed)]) for seed in base.FROZEN_ENCODER_SEEDS]))
    cell_summary = descriptive_cell_summary(outcomes_median, null_median)
    horizon, size, burden = options["stage2_arms"][0]
    ladder = base.dilution_ladder()[str(horizon)][str(int(size))][str(burden)]
    arm = PopulationArm(horizon=str(horizon), size=int(size), burden=str(burden), n_abnormal=int(ladder["n_abnormal"]), n_ordinary=int(ladder["n_ordinary"]), pair_indices=tuple(range(int(size))))
    stage2 = evaluate_stage2_arm(bundles, history_seed, history_id + "-STAGE2", family, severity, arm, 0, reference, sigma_per_seed, device=device, journal=journal)
    stage2_arms = [
        {"model_seed": int(seed), "e1_transition": float(stage2["per_seed"][str(seed)]["e1_transition"]), "e1_control": float(stage2["per_seed"][str(seed)]["e1_control"])}
        for seed in base.FROZEN_ENCODER_SEEDS
    ] if stage2.get("status") == "COMPUTED" else []
    per_seed_e1: dict[str, list[float]] = {}
    for seed in base.FROZEN_ENCODER_SEEDS:
        per_seed_e1[str(seed)] = [round(float(entry["e1_per_seed"][str(seed)]), 6) for entry in cell["outcomes"]]
    counted = capture_files_count_holder()
    forwards = int(counted["files_captured"])
    return TinySmokeResult(
        status="TINY-OFFICIAL-SMOKE-COMPLETE",
        family=family,
        severity=severity,
        history_seed=history_seed,
        n_groups=int(cell["n_groups"]),
        per_seed_e1={key: [round(float(value), 6) for value in values] for key, values in per_seed_e1.items()},
        per_seed_q95={str(seed): round(float(value), 6) for seed, value in null_q95.items()},
        cell_summary=cell_summary,
        disjoint_test=cell["disjoint_test"],
        stage2_arms=stage2_arms,
        forwards_executed=int(forwards),
        model_loads=len(bundles),
        elapsed_s=round(time.time() - started, 2),
    )


def run_official_matrix(
    bundles: Mapping[int, Any],
    *,
    permitted: bool = False,
    histories: Sequence[str] | None = None,
    cells: Sequence[tuple[str, float]] | None = None,
    g_level: str = "g48",
    stage2_burdens: Sequence[str] | None = None,
    device: str | None = None,
    journal: CaptureJournal | None = None,
) -> dict[str, Any]:
    """Execute the official matrix through the shared engine (gated).

    Iterates every selected history x cell x G-level through
    ``collect_stage1_cell`` and every selected stage-2 arm through
    ``evaluate_stage2_arm`` — the SAME functions the tiny fixture calls —
    then summarizes families with Holm, sensitivities and cluster bootstraps.
    ``permitted=False`` (default) refuses: full-matrix inference needs M17 C2
    PASS + checkpoint + M18 release. Parameterized subsets (tiny fixture) are
    the smoke path.
    """
    _require_official_release(bool(permitted))
    if journal is None:
        raise _refused("REFUSED-PREREQUISITE", "official capture requires a durable journal")
    import numpy as np

    started = time.time()
    reset_capture_counter()
    wanted_histories = list(histories) if histories is not None else list(base.PAIRED_HISTORY_IDS)
    wanted_cells = [(str(family), float(severity)) for family, severity in (cells if cells is not None else CELL_GRID)]
    wanted_burdens = [str(value) for value in (stage2_burdens if stage2_burdens is not None else ("LOW", "MID", "ELEVATED"))]
    plans = {f"{family}|{severity}": plan for plan in cell_plans() for family, severity in [(plan.family, plan.severity)]}
    reference_cache: dict[str, tuple[dict[int, dict[str, Any]], dict[int, float]]] = {}
    stage1: dict[str, Any] = {}
    for history_id, history_seed in zip(base.PAIRED_HISTORY_IDS, base.PROPOSED_HISTORY_SEEDS):
        if history_id not in wanted_histories:
            continue
        reference = capture_reference_pool(bundles, int(history_seed), f"{history_id}-REF", device=device, journal=journal)
        bandwidth = capture_reference_pool(bundles, int(history_seed), f"{history_id}-BANDWIDTH",
                                           device=device, journal=journal, source_role="bandwidth")
        sigma_per_seed = frozen_reference_bandwidth(bandwidth)
        reference_cache[history_id] = (reference, sigma_per_seed)
        null_q95 = fit_null_q95(bundles, int(history_seed), f"{history_id}-NULL", sigma_per_seed, device=device, journal=journal)
        for family, severity in wanted_cells:
            plan = plans[f"{family}|{severity}"]
            indices = list(plan.pair_index_g48) if g_level == "g48" else list(plan.pair_index_g24)
            result = collect_stage1_cell(
                bundles, int(history_seed), history_id, family, severity,
                indices, reference, sigma_per_seed, null_q95, device=device, journal=journal,
            )
            medians = [float(entry["median_e1"]) for entry in result["outcomes"]]
            null_median = float(np.median([float(null_q95[int(seed)]) for seed in base.FROZEN_ENCODER_SEEDS]))
            summary = descriptive_cell_summary(medians, null_median)
            stage1[f"{history_id}|{family}|{severity}"] = {
                "history_id": history_id, "family": family, "severity": severity,
                **result, "summary": summary,
            }
    stage2: dict[str, Any] = {}
    cursor = 0
    cursor_step = 0
    for arm in population_arms():
        # Stride accounts for the FULL slice footprint of every arm (mixture +
        # control = n_abnormal + n_ordinary + size), so arm bands never overlap
        # even when a burden subset is executed.
        cursor_step = int(arm.n_abnormal) + int(arm.n_ordinary) + int(arm.size)
        if arm.burden not in wanted_burdens:
            cursor += cursor_step
            continue
        if histories is not None and len(wanted_histories) == 1:
            histories_for_arm = wanted_histories
        else:
            histories_for_arm = wanted_histories
        arm_results: list[dict[str, Any]] = []
        for history_id, history_seed in zip(base.PAIRED_HISTORY_IDS, base.PROPOSED_HISTORY_SEEDS):
            if history_id not in histories_for_arm:
                continue
            reference, sigma_per_seed = reference_cache[history_id]
            family0, severity0 = wanted_cells[0]
            arm_results.append(
                evaluate_stage2_arm(
                    bundles, int(history_seed), f"{history_id}-STAGE2", family0, severity0,
                    arm, cursor, reference, sigma_per_seed, device=device, journal=journal,
                )
            )
        medians = [float(np.median([float(entry["e1_transition"]) for entry in result["per_seed"].values() if np.isfinite(float(entry["e1_transition"]))])) for result in arm_results if result.get("status") == "COMPUTED"]
        stage2[f"{arm.horizon}|{arm.size}|{arm.burden}"] = {
            "arm": {"horizon": arm.horizon, "size": arm.size, "burden": arm.burden},
            "per_history": arm_results,
            "bootstrap": cluster_bootstrap_median(medians),
        }
        cursor += cursor_step
    # Global family is exactly 6 histories x 27 cells; partial runs do not
    # acquire a smaller, easier multiplicity family.
    expected_keys = {
        f"{history}|{family}|{severity}"
        for history in base.PAIRED_HISTORY_IDS for family, severity in CELL_GRID
    }
    if set(stage1) == expected_keys and g_level == "g48":
        keys = sorted(expected_keys)
        holm = {
            "status": "CONDITIONAL-ON-EXCHANGEABILITY",
            "family": "162 history-by-family-by-severity cells",
            "alpha": 0.05, "cell_order": keys,
            "adjustments": holm_over_families(
                [float(stage1[key]["disjoint_test"]["p_value"]) for key in keys], alpha=0.05,
            ),
        }
    else:
        holm = {"status": "UNSUPPORTED-SUBSET", "reason": "global 162-cell family incomplete"}
    return {
        "status": "OFFICIAL-MATRIX-COMPLETE",
        "g_level": str(g_level),
        "stage1": stage1,
        "stage2": stage2,
        "holm_by_family": holm,
        "forwards_executed": int(capture_files_count_holder()["files_captured"]),
        "model_loads": len(bundles),
        "elapsed_s": round(time.time() - started, 2),
    }


def official_matrix_counts() -> dict[str, Any]:
    """Count the full official workload (no outcomes, enumeration only)."""
    matrix = matrix_plan()
    stage1_pairs_per_history = sum(cell["pairs_g48"] for cell in matrix["cells"])
    reference = REFERENCE_SLOTS_PER_HISTORY
    arms = population_arms()
    stage2_files_per_history = sum(int(arm.size) for arm in arms)
    null_files_per_history = NULL_PAIRS_PER_CELL_SEED * 2 * len(matrix["cells"])
    return {
        "histories": len(matrix["histories"]),
        "cells": len(matrix["cells"]),
        "stage1_pairs_g48_per_history": int(stage1_pairs_per_history),
        "stage1_pairs_g48_total": int(stage1_pairs_per_history * len(matrix["histories"])),
        "reference_seeds_per_history": int(reference),
        "null_files_per_history": int(null_files_per_history),
        "stage2_arms": len(arms),
        "stage2_files_per_history": int(stage2_files_per_history),
        "stage2_files_total": int(stage2_files_per_history * len(matrix["histories"])),
        "encoder_seeds": list(base.FROZEN_ENCODER_SEEDS),
        "note": "enumeration only; no outcomes inferred",
    }


__all__ = [
    "OFFICIAL_CONTRACT_ID",
    "SUPERSEDED_CONTRACT_ID",
    "CRITERION",
    "REVIEW_CYCLE",
    "CUMULATIVE_FORWARDS_BEFORE",
    "CELL_GRID",
    "D_MIN_COVERAGE",
    "CLUSTER_BOOTSTRAP_REPLICATES",
    "CLUSTER_BOOTSTRAP_SEED",
    "TINY_FIXTURE",
    "CellPlan",
    "GeneratedPair",
    "PopulationArm",
    "TinySmokeResult",
    "allocation_table",
    "build_official_contract",
    "capture_files",
    "capture_files_count_holder",
    "capture_reference_pool",
    "cell_plans",
    "cluster_bootstrap_median",
    "collect_stage1_cell",
    "e2_occupancy",
    "embedding_matrices",
    "evaluate_stage2_arm",
    "fit_null_q95",
    "frozen_reference_bandwidth",
    "generate_cell_pairs",
    "generate_null_pool",
    "generate_reference_pool",
    "holm_over_families",
    "matrix_plan",
    "official_matrix_counts",
    "population_arms",
    "reset_capture_counter",
    "run_official_matrix",
    "score_cell_groups",
    "tiny_official_smoke",
    "verify_official_contract",
    "within_prototype_diagnostics",
]
