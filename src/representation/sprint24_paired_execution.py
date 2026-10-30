"""Sprint 24 Task 7-M17 executable paired-experiment contract and runner.

Full-route implementation of the owner-approved option A (Main relay, frozen
in ``docs/sprint-plans/sprint-24.md:266-271``): unbiased E1 scored on
independent groups of K=8 paired sources. N=24 then N=48 INDEPENDENT SCORED
GROUPS (192/384 source pairs per cell), never files or repeated variants.
NO 96 branch anywhere in the contract/runner/gray-zone logic; any G>48
needs a separate owner authorization.

Stage-1 scores each scored group with unbiased MMD^2 (E1, primary) between
its K=8 twin embeddings and its K=8 abnormal embeddings (same family x
severity, disjoint seeds), calibrated against matched-size normal-normal
twin groups (same K per arm, frozen q95 per family x severity cell per
encoder seed). The per-cell summary is the detection fraction over G
independent groups with an exact Clopper-Pearson interval; sign-test power
is recomputed at the GROUP level (never transfer M15 per-pair sign/t power).

The 0.5 bar is a response fraction (fraction of groups above the frozen
null q95), never a raw E1 value. Misses are INCONCLUSIVE; no absence claim
without a separately owner-approved relevant-effect margin (none exists).
Week 40/60 and month 160/250 burdens 0.05/0.15/0.275 preserved; 3 fixed
encoders as run-replicate medians; duration within-family scope only.

This module does NOT generate the official paired corpus, load a model, run
a forward pass, train, refit, contact any remote host, or release Phase C.
It freezes identities/splits/calibration/reference/evaluation,
seed-namespace/history/calendar/checkpoints/banks/denominators/dispositions
and decision parameters BEFORE official outcomes. No outcome-driven tuning
or silent redraw: the contract builder fails closed on drift.

Cumulative observed forward passes before this task: 77,653 (smoke forwards
are counted from there; official corpus/full inference needs M17 evidence
C1 PASS + Bronze checkpoint + M18 release).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from representation.sprint24_context_audit import canonical_json
from representation import sprint24_paired_decision_experiment as design

# --- M17 execution identity ---------------------------------------------------

#: Versioned M17 execution contract identifier (immutable once frozen).
EXECUTION_CONTRACT_ID = "S24-PAIRED-EXECUTION-CONTRACT-A01"

#: Design this execution freezes (bytes preserved; M15 design/report/contract
#: are read-only context, never edited here).
EXECUTION_DESIGN_ID = design.DESIGN_ID
EXECUTION_BASE_CONTRACT_ID = design.CONTRACT_ID

#: Criterion pointer (plan-assigned scope).
CRITERION = "docs/sprint-plans/sprint-24.md:266-271"

#: Review cycle / round this implementation targets.
REVIEW_CYCLE = "S24-7-M17-C1"

#: Cumulative observed forward passes before M17 (M14 accounting, unchanged).
CUMULATIVE_FORWARDS_BEFORE = 77653

#: Option-A scored-group size: K twin + K abnormal source pairs per group.
GROUP_SIZE_K = 8

#: Approved finite scored-group grid per (family x severity) cell. G=48 is
#: the planned release; G=24 is the interim futility look. NO 96 branch:
#: any G>48 needs a separate explicit owner authorization.
STAGE1_GROUP_GRID: tuple[int, ...] = (24, 48)
STAGE1_FUTILITY_LOOK_G = 24

#: Seed fan-out per scored-group bar: G groups x K pairs x 2 draws
#: (twin + abnormal). 24 -> 192, 48 -> 384 source pairs per cell.
GROUPS_TO_PAIRS = GROUP_SIZE_K

#: Inherited constants (re-declared unchanged, never re-tuned).
FROZEN_ENCODER_SEEDS = design.FROZEN_ENCODER_SEEDS
PAIRED_FAMILIES = design.PAIRED_FAMILIES
PAIRED_SEVERITIES = design.PAIRED_SEVERITIES
PAIRED_HISTORY_IDS = design.PAIRED_HISTORY_IDS
PRIMARY_TRIPLE = design.PRIMARY_TRIPLE
REPLICATION_TRIPLE = design.REPLICATION_TRIPLE
PROPOSED_HISTORY_SEEDS = design.PROPOSED_HISTORY_SEEDS
PAIR_SEED_DOMAIN = design.PAIR_SEED_DOMAIN
CERTIFIED_OLD_SEED_MIN = design.CERTIFIED_OLD_SEED_MIN
CERTIFIED_OLD_SEED_MAX = design.CERTIFIED_OLD_SEED_MAX
EXECUTED_M8_HISTORY_SEEDS = design.EXECUTED_M8_HISTORY_SEEDS
STAGE2_SIZES = design.STAGE2_SIZES
STAGE2_BURDENS = design.STAGE2_BURDENS
BUSINESS_BASELINE_RULE = design.BUSINESS_BASELINE_RULE
GROUPING_RULE = design.GROUPING_RULE
PRIMARY_ESTIMATOR = design.PRIMARY_ESTIMATOR
SENSITIVITY_ESTIMATOR = design.SENSITIVITY_ESTIMATOR
OCCUPANCY_ESTIMATOR = design.OCCUPANCY_ESTIMATOR
PRIMARY_K = design.PRIMARY_K
CLUSTER_COUNTS = design.CLUSTER_COUNTS
REFERENCE_FILES_PER_HISTORY = design.REFERENCE_FILES_PER_HISTORY
REFERENCE_CAUSAL_RULE = design.REFERENCE_CAUSAL_RULE
LOW_BURDEN_OPERATING_BAND = design.LOW_BURDEN_OPERATING_BAND
ELEVATED_OPERATING_BAND = design.ELEVATED_OPERATING_BAND
NULL_SEED = design.NULL_SEED
ALPHA_TWO_SIDED = design.ALPHA_TWO_SIDED

#: Approved bars (proposals with owner sign-off, frozen as of option-A
#: approval; no silent change without a new owner decision).
APPROVED_RESPONSE_BAR = 0.5
APPROVED_FUTILITY_CAP = 0.10
FALSE_ALARM_ANCHOR = 0.06

#: Per-cell count: 9 families x 3 severities; duration pairs within family only.
CELL_COUNT = len(PAIRED_FAMILIES) * len(PAIRED_SEVERITIES)

#: Closed disposition vocabulary for generated pairs.
DISPOSITIONS = (
    "PAIRED",
    "NORMAL_RETURN",
    "ABSTAIN-BOUNDARY",
    "ABSTAIN-COVERAGE",
    "ABSTAIN-REFERENCE-SHORTFALL",
    "ABSTAIN-STRENGTH-GATE",
)

#: Closed decision branches (bounded; no guaranteed result).
DECISION_BRANCHES = design.DECISION_BRANCHES


def _refused(reason: str, detail: str) -> ValueError:
    return ValueError(f"{reason}: {detail}")


# --- seed plan (grouped, lineage-separate) ------------------------------------


def check_group_seed_plan(
    history_seeds: Sequence[int] = PROPOSED_HISTORY_SEEDS,
    groups_per_cell: Sequence[int] = STAGE1_GROUP_GRID,
    group_size_k: int = GROUP_SIZE_K,
) -> dict[str, Any]:
    """Fail closed unless the grouped seed plan is collision-free and exact.

    The plan needs ``max(groups) * K`` pair slots per history per cell;
    every pair index derives under the frozen pair-seed domain, so the
    check covers the full ``max(G)*K`` span per history, not just G.
    """
    seeds = [int(value) for value in history_seeds]
    if len(set(seeds)) != len(seeds):
        raise _refused("REFUSED-SEED-COLLISION", "history seeds are not unique")
    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise _refused("REFUSED-SEED-COLLISION", f"history seed is not an integer: {seed!r}")
        if CERTIFIED_OLD_SEED_MIN <= seed <= CERTIFIED_OLD_SEED_MAX:
            raise _refused("REFUSED-SEED-COLLISION", f"history seed {seed} collides with the certified old block")
        if seed in tuple(int(value) for value in EXECUTED_M8_HISTORY_SEEDS):
            raise _refused("REFUSED-SEED-COLLISION", f"history seed {seed} collides with the executed M8 block")
    slots = max(int(value) for value in groups_per_cell) * int(group_size_k)
    derived: set[int] = set()
    for history_seed in seeds:
        for pair_index in range(slots):
            value = design.pair_seed(int(history_seed), int(pair_index))
            if not 0 <= value < 2**31:
                raise _refused("REFUSED-SEED-COLLISION", "derived pair seed out of range")
            derived.add(value)
    forbidden = set(range(CERTIFIED_OLD_SEED_MIN, CERTIFIED_OLD_SEED_MAX + 1))
    forbidden.update(int(value) for value in EXECUTED_M8_HISTORY_SEEDS)
    overlap = sorted(derived & forbidden)
    if overlap:
        raise _refused("REFUSED-SEED-COLLISION", f"derived pair seeds collide: {overlap[:8]}")
    return {
        "status": "PASS",
        "history_seeds": seeds,
        "groups_per_cell": [int(value) for value in groups_per_cell],
        "group_size_k": int(group_size_k),
        "pair_slots_per_history": slots,
        "n_derived_unique": len(derived),
        "expected_unique": len(seeds) * slots,
    }


# --- group-level operating characteristics ------------------------------------


def group_power_grid(
    groups: Sequence[int] = STAGE1_GROUP_GRID,
    p_grid: Sequence[float] = (0.65, 0.75, 0.85),
    alpha: float = ALPHA_TWO_SIDED,
) -> dict[str, Any]:
    """Recompute exact group-level sign power for scored groups (option A).

    Each scored group contributes ONE binary response (above/below the
    frozen q95), so the independent unit is the group G, never the pair.
    The M15 per-pair sign/t power values are NOT transferred.
    """
    entries = [design.sign_test_power(int(g), float(p), alpha) for g in groups for p in p_grid]
    for entry in entries:
        entry["independent_unit"] = "one scored group of K=8 paired sources"
        entry["assumption"] = (
            "p_win is ASSUMED (sensitivity grid), not a measured anomaly "
            "effect; M14 embedding perturbations measure responsiveness only. "
            "Binary response per scored group; paired cross-arm dependence is "
            "absorbed inside the group contrast, never counted as two samples."
        )
    return {
        "groups": [int(value) for value in groups],
        "p_grid": [float(value) for value in p_grid],
        "alpha": alpha,
        "sign_power_groups": entries,
    }


def clopper_pearson_rows(groups: Sequence[int] = STAGE1_GROUP_GRID) -> dict[str, Any]:
    """Exact two-sided intervals for the response-fraction bar and anchors."""
    rows = [
        design.clopper_pearson(24, 24),
        design.clopper_pearson(12, 24),
        design.clopper_pearson(24, 48),
        design.clopper_pearson(32, 48),
        design.clopper_pearson(5, 48),
        design.clopper_pearson(14, 48),
        design.clopper_pearson(5, 24),
    ]
    return {
        "method": "Clopper-Pearson exact (regularized incomplete beta, Lentz)",
        "response_fraction_note": (
            "fraction = share of scored groups above the frozen null q95; "
            "0.5 is a PROPOSED bar on that fraction, never a raw E1 value."
        ),
        "rows": rows,
    }


def sequential_multiplicity_plan(
    groups: Sequence[int] = STAGE1_GROUP_GRID,
    alpha: float = ALPHA_TWO_SIDED,
) -> dict[str, Any]:
    """Declare the sequential/expansion and multiplicity handling (no 96)."""
    n_families = len(PAIRED_FAMILIES)
    n_cells = CELL_COUNT
    return {
        "interim_look": {
            "at_G": STAGE1_FUTILITY_LOOK_G,
            "rule": (
                "at G=24, a family at or below the measured false-alarm "
                f"anchor (cap {APPROVED_FUTILITY_CAP}) stops for futility; "
                "no optional stopping beyond this rule; no outcome-based "
                "metric selection."
            ),
        },
        "expansion": {
            "from_G": 24,
            "to_G": 48,
            "planned": True,
            "rule": "the 24->48 step is the predeclared plan; both results publish.",
            "beyond_48": (
                "NOT AUTHORIZED: no 96 cap-only branch, no gray-zone "
                "expansion path; any G>48 needs a separate explicit owner "
                "authorization with recomputed operating characteristics."
            ),
        },
        "multiplicity": {
            "family_level_holm_alpha": alpha,
            "n_families": n_families,
            "n_cells": n_cells,
            "rule": (
                "Holm step-down over the declared primary contrasts "
                f"(per family, alpha={alpha}); per-cell detection fractions "
                "report exact intervals regardless of disposition."
            ),
        },
        "misses": (
            "INCONCLUSIVE: a missed 0.5 response-fraction bar is never read "
            "as absence. No REPRESENTATION negative without a separately "
            "owner-approved relevant-effect margin plus a valid "
            "upper-confidence-bound computation against it (none exists)."
        ),
    }


def dilution_ladder() -> dict[str, Any]:
    """Exact burden counts (M13 rounding), inherited unchanged."""
    return {
        horizon: {
            str(size): {
                name: design.dilution_counts(int(size), float(burden))
                for name, burden in STAGE2_BURDENS.items()
            }
            for size in sizes
        }
        for horizon, sizes in STAGE2_SIZES.items()
    }


# --- calibration / evaluation contract ----------------------------------------


def calibration_spec() -> dict[str, Any]:
    """Freeze the matched-size normal-normal group calibration rule."""
    return {
        "null_unit": "one matched-size normal-normal twin group (K=8 twin embeddings per arm)",
        "same_K": GROUP_SIZE_K,
        "statistic": "unbiased MMD^2 (E1, primary) between the two K=8 arms",
        "threshold": "frozen q95 per (family x severity) cell per encoder seed",
        "encoder_seeds": list(FROZEN_ENCODER_SEEDS),
        "encoder_role": "run replicates summarized by median, never independent data",
        "dependence": (
            "paired twin/abnormal arms share the pre-injection source, so the "
            "two MMD arms are NOT independent samples; the cross-term couples "
            "them. The independent unit across groups is the binary response "
            "(above/below the frozen q95), not the MMD value itself."
        ),
        "reference_pool": {
            "files_per_history": REFERENCE_FILES_PER_HISTORY,
            "causal_rule": REFERENCE_CAUSAL_RULE,
            "bandwidth": "median pairwise distance on the causal reference pool",
            "kmeans": list(CLUSTER_COUNTS),
            "primary_k": PRIMARY_K,
            "disjoint_from_pair_control": True,
        },
    }


def evaluation_spec() -> dict[str, Any]:
    """Freeze the grouped Stage-1 and inherited Stage-2 evaluation rules."""
    return {
        "stage1": {
            "independent_unit": "one scored group of K=8 paired sources",
            "contrast": "unbiased E1: K=8 twin embeddings vs K=8 abnormal embeddings (same family x severity, disjoint seeds)",
            "summary": "detection fraction over G independent groups with exact Clopper-Pearson interval",
            "proposed_bar": f"E1 response fraction >= {APPROVED_RESPONSE_BAR} with the exact lower bound above the measured false-alarm anchor",
            "grouping": GROUPING_RULE,
            "no_transfer": "M15 per-pair sign/t power is not transferred; group-level power above governs",
        },
        "stage2": {
            "sizes": {key: list(value) for key, value in STAGE2_SIZES.items()},
            "burdens": dict(STAGE2_BURDENS),
            "rounding": "n_abnormal = floor(b * N + 0.5), inherited from M13",
            "primary": "E1 transition median above the context-only control q95 at elevated burden, weekly and monthly",
            "dilution_ladder": "low/mid must show monotone non-increasing response (descriptive)",
            "uncertainty": "history-cluster bootstrap, B=2000, seed 2407, descriptive at small n; no pair-level bootstrap",
        },
        "duration_scope": "duration_anomaly pairs within family only, never cross-family evidence",
        "abstentions": list(DISPOSITIONS),
        "denominators": "every planned group counts or carries a closed abstention; no silent redraw",
    }


# --- contract builder / verifier ----------------------------------------------


def execution_summary() -> dict[str, Any]:
    """Return the frozen execution constants as a machine-readable mapping."""
    return {
        "execution_contract_id": EXECUTION_CONTRACT_ID,
        "design_id": EXECUTION_DESIGN_ID,
        "base_contract_id": EXECUTION_BASE_CONTRACT_ID,
        "criterion": CRITERION,
        "review_cycle": REVIEW_CYCLE,
        "cumulative_forwards_before": CUMULATIVE_FORWARDS_BEFORE,
        "group_size_k": GROUP_SIZE_K,
        "stage1_group_grid": list(STAGE1_GROUP_GRID),
        "stage1_futility_look_g": STAGE1_FUTILITY_LOOK_G,
        "pairs_per_cell": {str(g): int(g) * GROUPS_TO_PAIRS for g in STAGE1_GROUP_GRID},
        "cell_count": CELL_COUNT,
        "paired_families": list(PAIRED_FAMILIES),
        "paired_severities": list(PAIRED_SEVERITIES),
        "paired_histories": list(PAIRED_HISTORY_IDS),
        "primary_triple": list(PRIMARY_TRIPLE),
        "replication_triple": list(REPLICATION_TRIPLE),
        "proposed_history_seeds": list(PROPOSED_HISTORY_SEEDS),
        "pair_seed_domain": PAIR_SEED_DOMAIN,
        "frozen_encoder_seeds": list(FROZEN_ENCODER_SEEDS),
        "primary_estimator": PRIMARY_ESTIMATOR,
        "sensitivity_estimator": SENSITIVITY_ESTIMATOR,
        "occupancy_estimator": OCCUPANCY_ESTIMATOR,
        "cluster_counts": list(CLUSTER_COUNTS),
        "primary_k": PRIMARY_K,
        "reference_files_per_history": REFERENCE_FILES_PER_HISTORY,
        "reference_causal_rule": REFERENCE_CAUSAL_RULE,
        "business_baseline_rule": BUSINESS_BASELINE_RULE,
        "grouping_rule": GROUPING_RULE,
        "approved_response_bar": APPROVED_RESPONSE_BAR,
        "approved_futility_cap": APPROVED_FUTILITY_CAP,
        "false_alarm_anchor": FALSE_ALARM_ANCHOR,
        "alpha_two_sided": ALPHA_TWO_SIDED,
        "no_96_branch": True,
    }


def runtime_pins(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the verified server runtime (checkpoints/banks/device) by digest."""
    checkpoints = dict(preflight.get("checkpoints") or {})
    bank_files = dict(preflight.get("bank_files") or {})
    banks = dict(preflight.get("banks") or {})
    arms = dict(preflight.get("arms") or {})
    for seed in FROZEN_ENCODER_SEEDS:
        key = str(seed)
        for table, name in ((checkpoints, "checkpoint"), (bank_files, "bank file"), (banks, "bank")):
            value = table.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise _refused("REFUSED-IDENTITY", f"preflight {name} digest missing for seed {seed}")
    return {
        "server": "trietlm@192.168.99.214",
        "checkpoint_roots": "/home/trietlm/s18-t23-a04-runs/full",
        "bank_roots": "/home/trietlm/s18-t32-a02-runs/scores",
        "checkpoints": {str(seed): checkpoints[str(seed)] for seed in FROZEN_ENCODER_SEEDS},
        "bank_files": {str(seed): bank_files[str(seed)] for seed in FROZEN_ENCODER_SEEDS},
        "banks": {str(seed): banks[str(seed)] for seed in FROZEN_ENCODER_SEEDS},
        "arms": {str(seed): arms[str(seed)] for seed in FROZEN_ENCODER_SEEDS},
        "device": "cuda",
        "interpreter": "main-checkout venv python (torch 2.14.0+cu130, CUDA True)",
    }


def build_execution_contract(
    false_alarm_anchors: Mapping[str, Any],
    preflight: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the frozen executable contract (deterministic, fail-closed)."""
    if not isinstance(false_alarm_anchors, Mapping) or not false_alarm_anchors:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "false-alarm anchors are required")
    if not isinstance(preflight, Mapping) or not preflight:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "verified preflight is required")
    if not isinstance(provenance, Mapping) or not provenance:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "provenance is required")
    seed_plan = check_group_seed_plan()
    grid = group_power_grid()
    cp_reference = clopper_pearson_rows()
    contract = {
        "contract_id": EXECUTION_CONTRACT_ID,
        "design_id": EXECUTION_DESIGN_ID,
        "base_contract_id": EXECUTION_BASE_CONTRACT_ID,
        "state": "FROZEN-EXECUTABLE",
        "not_an_official_release": True,
        "execution": execution_summary(),
        "false_alarm_anchors": dict(false_alarm_anchors),
        "group_operating_characteristics": grid,
        "clopper_pearson_reference": cp_reference,
        "sequential_multiplicity": sequential_multiplicity_plan(),
        "dilution_ladder": dilution_ladder(),
        "calibration": calibration_spec(),
        "evaluation": evaluation_spec(),
        "runtime_pins": runtime_pins(preflight),
        "seed_plan": seed_plan,
        "owner_approvals": [
            "option A: unbiased E1 on independent K=8 scored groups, 24->48 groups/cell",
            "no 96 branch: any G>48 needs separate explicit owner authorization",
            "0.5 = response fraction, not raw E1; misses are inconclusive",
            "duration within-family scope only",
        ],
        "execution_boundary": {
            "this_contract_authorizes": [
                "tiny real paired-dataset capture through the public entry (isolated smoke only)",
                "frozen-encoder embedding capture through capture_s24_file_features on accepted bytes",
                "grouped E1 evaluation on real embeddings with deterministic replay",
            ],
            "this_contract_forbids": [
                "official paired-corpus generation",
                "official full inference/scale",
                "training, bank refit, Phase C release",
                "random weights, fallback checkpoints, silent redraw",
                "edits to accepted M11-M16 products",
            ],
        },
        "provenance": dict(provenance),
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def verify_execution_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the contract is the deterministic build output."""
    if not isinstance(contract, Mapping):
        raise _refused("REFUSED-CONTRACT-DRIFT", "contract is not a mapping")
    if contract.get("contract_id") != EXECUTION_CONTRACT_ID:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "unknown execution contract id")
    if contract.get("design_id") != EXECUTION_DESIGN_ID:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "unknown design id")
    if contract.get("no_96_branch") is False:
        raise _refused("REFUSED-CONTRACT-DRIFT", "96 branch is not authorized")
    execution = contract.get("execution")
    if not isinstance(execution, Mapping):
        raise _refused("REFUSED-CONTRACT-DRIFT", "execution block is not a mapping")
    if list(execution.get("stage1_group_grid") or []) != [24, 48]:
        raise _refused("REFUSED-CONTRACT-DRIFT", "group grid drifted from 24/48")
    if execution.get("no_96_branch") is not True:
        raise _refused("REFUSED-CONTRACT-DRIFT", "96 branch is not authorized")
    recorded = contract.get("contract_sha256")
    rebuilt = _canonical_sha256({key: value for key, value in contract.items() if key != "contract_sha256"})
    if recorded != rebuilt:
        raise _refused("REFUSED-CONTRACT-DRIFT", "contract bytes do not match their recorded digest")
    return {"status": "PASS", "contract_sha256": recorded}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256((canonical_json(dict(payload)) + "\n").encode("utf-8")).hexdigest()


# --- grouped evaluation on real embeddings ------------------------------------

#: Task identifier for every M17 run manifest.
TASK_ID = "S24-7-M17"

#: Closed refusal reasons for the runner.
REFUSAL_REASONS = (
    "REFUSED-OVERWRITE",
    "REFUSED-CONTRACT-DRIFT",
    "REFUSED-UNKNOWN-DESIGN",
    "REFUSED-SEED-COLLISION",
    "REFUSED-IDENTITY",
    "REFUSED-MISSING-PREREQUISITE",
    "REFUSED-ADMISSION",
    "REFUSED-ERROR-BOUNDARY",
)


def score_scored_group(
    twin_matrix: Any,
    abnormal_matrix: Any,
    sigma: float,
) -> dict[str, Any]:
    """Score one K=8 scored group with unbiased E1 (fail-closed on shape).

    Both arms must carry exactly GROUP_SIZE_K embedding rows; fewer rows
    (e.g. a singleton pair) raise instead of falling back to a biased or
    singleton substitute.
    """
    import numpy as np

    twin = np.asarray(twin_matrix, dtype=np.float64)
    abnormal = np.asarray(abnormal_matrix, dtype=np.float64)
    if twin.ndim != 2 or abnormal.ndim != 2:
        raise _refused("REFUSED-ERROR-BOUNDARY", "group arms must be 2-D embedding matrices")
    if twin.shape[0] != GROUP_SIZE_K or abnormal.shape[0] != GROUP_SIZE_K:
        raise _refused(
            "REFUSED-ERROR-BOUNDARY",
            f"scored group needs K={GROUP_SIZE_K} rows per arm, got {twin.shape[0]}/{abnormal.shape[0]}",
        )
    if twin.shape[1] != abnormal.shape[1]:
        raise _refused("REFUSED-ERROR-BOUNDARY", "group arms disagree on embedding width")
    if not bool(np.isfinite(twin).all()) or not bool(np.isfinite(abnormal).all()):
        raise _refused("REFUSED-ERROR-BOUNDARY", "group embeddings are non-finite")
    if not math.isfinite(sigma) or sigma <= 0:
        raise _refused("REFUSED-ERROR-BOUNDARY", "reference bandwidth is non-positive")
    from representation.sprint24_weekly_monthly_endpoints import mmd2_unbiased

    value = float(mmd2_unbiased(twin, abnormal, float(sigma)))
    if not math.isfinite(value):
        raise _refused("REFUSED-ERROR-BOUNDARY", "group E1 is non-finite")
    return {
        "statistic": "E1-unbiased-K8-vs-K8",
        "k_per_arm": GROUP_SIZE_K,
        "e1": value,
        "sigma": float(sigma),
    }


def summarize_cell(
    group_e1: Sequence[float],
    null_q95: float,
    alpha: float = ALPHA_TWO_SIDED,
) -> dict[str, Any]:
    """Summarize one cell as a response fraction with an exact interval."""
    values = [float(value) for value in group_e1]
    if not values:
        raise _refused("REFUSED-ERROR-BOUNDARY", "cell has no scored groups")
    if not math.isfinite(null_q95):
        raise _refused("REFUSED-ERROR-BOUNDARY", "null q95 is non-finite")
    for value in values:
        if not math.isfinite(value):
            raise _refused("REFUSED-ERROR-BOUNDARY", "group E1 is non-finite")
    responses = sum(1 for value in values if value > float(null_q95))
    interval = design.clopper_pearson(int(responses), len(values), alpha)
    return {
        "n_groups": len(values),
        "n_responses": int(responses),
        "response_fraction": round(int(responses) / len(values), 4),
        "null_q95": float(null_q95),
        "clopper_pearson": interval,
        "bar": APPROVED_RESPONSE_BAR,
        "bar_met": bool(interval["ci_lower"] > FALSE_ALARM_ANCHOR and (int(responses) / len(values)) >= APPROVED_RESPONSE_BAR),
        "reading": (
            "INCONCLUSIVE unless the bar is met with the exact lower bound "
            "above the measured false-alarm anchor; a miss is never absence."
        ),
    }


def holm_over_families(p_values: Sequence[float], alpha: float = ALPHA_TWO_SIDED) -> list[dict[str, Any]]:
    """Holm step-down over the declared primary family contrasts."""
    from representation.sprint24_controlled_inference_execution import holm_adjust

    return holm_adjust(list(p_values), alpha=alpha)


__all__ = [
    "ALPHA_TWO_SIDED",
    "APPROVED_FUTILITY_CAP",
    "APPROVED_RESPONSE_BAR",
    "BUSINESS_BASELINE_RULE",
    "CELL_COUNT",
    "CERTIFIED_OLD_SEED_MAX",
    "CERTIFIED_OLD_SEED_MIN",
    "CLUSTER_COUNTS",
    "CRITERION",
    "CUMULATIVE_FORWARDS_BEFORE",
    "DECISION_BRANCHES",
    "DISPOSITIONS",
    "ELEVATED_OPERATING_BAND",
    "EXECUTED_M8_HISTORY_SEEDS",
    "EXECUTION_BASE_CONTRACT_ID",
    "EXECUTION_CONTRACT_ID",
    "EXECUTION_DESIGN_ID",
    "FALSE_ALARM_ANCHOR",
    "FROZEN_ENCODER_SEEDS",
    "GROUPS_TO_PAIRS",
    "GROUPING_RULE",
    "GROUP_SIZE_K",
    "LOW_BURDEN_OPERATING_BAND",
    "NULL_SEED",
    "OCCUPANCY_ESTIMATOR",
    "PAIRED_FAMILIES",
    "PAIRED_HISTORY_IDS",
    "PAIRED_SEVERITIES",
    "PAIR_SEED_DOMAIN",
    "PRIMARY_ESTIMATOR",
    "PRIMARY_K",
    "PRIMARY_TRIPLE",
    "PROPOSED_HISTORY_SEEDS",
    "REFUSAL_REASONS",
    "REFERENCE_CAUSAL_RULE",
    "REFERENCE_FILES_PER_HISTORY",
    "REPLICATION_TRIPLE",
    "REVIEW_CYCLE",
    "SENSITIVITY_ESTIMATOR",
    "STAGE1_FUTILITY_LOOK_G",
    "STAGE1_GROUP_GRID",
    "STAGE2_BURDENS",
    "STAGE2_SIZES",
    "TASK_ID",
    "build_execution_contract",
    "calibration_spec",
    "check_group_seed_plan",
    "clopper_pearson_rows",
    "dilution_ladder",
    "evaluation_spec",
    "execution_summary",
    "group_power_grid",
    "holm_over_families",
    "runtime_pins",
    "score_scored_group",
    "sequential_multiplicity_plan",
    "summarize_cell",
    "verify_execution_contract",
]
