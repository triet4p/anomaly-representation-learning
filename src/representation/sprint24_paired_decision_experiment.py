"""Sprint 24 Task 7-M15 paired-source representation and weekly/monthly movement experiment.

Design-only module: frozen paired-decision design constants, a deterministic
operating-characteristic/sample-size calculator (exact binomial math plus a
closed-form normal approximation; no model, no embedding, no forward pass),
a metadata-only paired-slot seed planner, and a builder/verifier for the
proposed machine-readable contract.

What this module does NOT do: generate the official paired corpus, load any
model, run any forward pass, train, refit, or contact any remote host. The
paired unmodified twin it specifies (``GAP-PAIRED-TWIN-01``) has no public
generator API today; the gap is documented with reproducible evidence, and the
twin constructor itself belongs to the subsequent execution task.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

# --- design identity ---------------------------------------------------------

#: Versioned design identifier frozen by this module.
DESIGN_ID = "S24-PAIRED-DECISION-EXPERIMENT-A01"

#: Proposed machine-readable contract identifier built by this module.
CONTRACT_ID = "S24-PAIRED-DECISION-CONTRACT-A01"

#: Criterion pointer (plan-assigned scope).
CRITERION = "docs/sprint-plans/sprint-24.md:250-257"

# --- inherited conventions (re-declared unchanged, never re-tuned) -----------

#: Inherited null seed for every hash-ordered selection (M3/M6 convention).
NULL_SEED = 2407

#: Accepted frozen encoder seeds (identity-selected run replicates, never data).
FROZEN_ENCODER_SEEDS = (181801, 181802, 181803)

#: Inherited low-burden ceiling / operating band (M5/M8, nonzero faults).
LOW_BURDEN_CEILING = 0.10
LOW_BURDEN_OPERATING_BAND = (0.02, 0.10)

#: Inherited elevated operating band for contrast scheduling (M5/M8).
ELEVATED_OPERATING_BAND = (0.20, 0.35)

#: Inherited primary occupancy prototype count and sensitivities (M12/M14).
PRIMARY_K = 8
CLUSTER_COUNTS = (4, 8, 16)

#: Inherited causal reference depth per history (M10/M14).
REFERENCE_FILES_PER_HISTORY = 220

#: Causal reference eligibility stays end_time <= c1 (M10/M14).
REFERENCE_CAUSAL_RULE = "end_time <= c1"

#: Cumulative observed forward passes before this design task (M14 accounting).
CUMULATIVE_FORWARDS_BEFORE = 77653

# --- paired construction -----------------------------------------------------

#: New paired-corpus namespace (lineage-separate from S18 and S24C blocks).
PAIRED_HISTORY_PREFIX = "S24P-ITER-0001"
PAIRED_HISTORY_IDS = tuple(
    f"{PAIRED_HISTORY_PREFIX}-PAIRED-{index:02d}" for index in range(1, 7)
)
PRIMARY_TRIPLE = PAIRED_HISTORY_IDS[:3]
REPLICATION_TRIPLE = PAIRED_HISTORY_IDS[3:]

#: Proposed fresh history seeds (outside the certified old block and the
#: executed M8 block; collision-checked by ``check_seed_plan``).
PROPOSED_HISTORY_SEEDS = (53101, 53102, 53103, 53104, 53105, 53106)
CERTIFIED_OLD_SEED_MIN = 32064
CERTIFIED_OLD_SEED_MAX = 32079
EXECUTED_M8_HISTORY_SEEDS = (42401, 42402, 42403, 42404, 42405, 42406)

#: Domain-separation tag for paired-seed derivation (never a v2 slot stream).
PAIR_SEED_DOMAIN = "s24p-paired-v1"

#: Hard anomaly families in scope (all nine; duration_anomaly pairs within
#: family only because it changes T/regimes — see GAP notes).
PAIRED_FAMILIES = (
    "contextual_replacement",
    "wrong_transition",
    "realistic_stuck",
    "over_regularity",
    "subtle_drift",
    "freq_phase_mismatch",
    "cross_channel_inconsistency",
    "duration_anomaly",
    "missing_event",
)

#: Within-pair severity ladder (inside the planned [0.35, 0.85] band).
PAIRED_SEVERITIES = (0.35, 0.60, 0.85)

#: Stage-1 finite paired-seed grid per (family x severity) cell; 96 is the
#: predeclared gray-zone expansion cap, never a silent second look.
STAGE1_SEED_GRID = (24, 48)
STAGE1_EXPAND_MAX = 96
STAGE1_FUTILITY_LOOK_N = 24

#: Stage-2 inherited per-arm sizes (M14-supported) and burden ladder.
STAGE2_SIZES = {"week": (40, 60), "month": (160, 250)}
STAGE2_BURDENS = {"LOW": 0.05, "MID": 0.15, "ELEVATED": 0.275}

#: Business baselines carry few faults (low-nonzero band), never zero by
#: requirement; pure-normal arms are diagnostic controls only.
BUSINESS_BASELINE_RULE = "low-nonzero burden in [0.02, 0.10]; pure normal is diagnostic control only"

#: Grouping rule against pseudo-N and leakage.
GROUPING_RULE = (
    "group key = pair seed; every variant (twin + all family/severity "
    "abnormals) of one seed stays in one split. Reference pool seeds are "
    "disjoint from pair/control seeds. Anomaly labels never enter encoder, "
    "reference, prototype, bandwidth or detector fits."
)

# --- estimators (rationale from M14 branch C ordering) ------------------------

#: E1 primary (first to respond, 0.25-0.5 sigma_u); D_LOC sensitivity;
#: E2_K8 primary occupancy (least responsive, 1.0-2.0 sigma_u); K4/K16 and
#: D_SPREAD/radial/tangential descriptive.
PRIMARY_ESTIMATOR = "E1"
SENSITIVITY_ESTIMATOR = "D_LOC"
OCCUPANCY_ESTIMATOR = "E2_K8"

# --- decision thresholds (PROPOSED for owner approval, never gates) -----------

#: Operational responsiveness threshold inherited from M14 branch-C language.
PROPOSED_DETECTION_THRESHOLD = 0.5

#: Futility band at the interim look (at or below the measured false-alarm
#: anchor means stop this family for futility).
PROPOSED_FUTILITY_CAP = 0.10

#: Gray-zone band licensing the predeclared 48 -> 96 expansion.
PROPOSED_GRAY_ZONE = (0.30, 0.50)

#: Two-sided error budget for the primary tests (exact tests; the single
#: gray-zone expansion is part of the procedure and both N=48/N=96 results
#: are reported).
ALPHA_TWO_SIDED = 0.05

# --- closed vocabulary ---------------------------------------------------------

DECISION_BRANCHES = (
    "SCALE",
    "MEASURE-DILUTION",
    "REPRESENTATION",
    "CONTEXT",
)
REFUSAL_REASONS = (
    "REFUSED-CONTRACT-DRIFT",
    "REFUSED-OVERWRITE",
    "REFUSED-SEED-COLLISION",
    "REFUSED-UNKNOWN-DESIGN",
)

#: Exact-interval reference rows embedded in the proposed contract: the
#: proposed bar (24/48), the futility example (5/48) and the gray-zone
#: example (14/48).
_CP_REFERENCE_ROWS: tuple[tuple[int, int], ...] = ((24, 48), (5, 48), (14, 48))

def _refused(reason: str, detail: str) -> ValueError:
    return ValueError(f"{reason}: {detail}")


# --- paired-slot seed planner (metadata only) ----------------------------------


def pair_seed(history_seed: int, pair_index: int) -> int:
    """Derive a deterministic unique pair seed under the paired domain tag."""
    digest = hashlib.sha256(
        f"{PAIR_SEED_DOMAIN}:{history_seed}:{pair_index:05d}".encode("utf-8")
    ).hexdigest()
    return int(digest[:15], 16) % (2**31)


def check_seed_plan(
    history_seeds: Sequence[int] = PROPOSED_HISTORY_SEEDS,
    used_slot_seeds: Sequence[int] = (),
) -> dict[str, Any]:
    """Fail closed if the paired seed plan collides with forbidden blocks."""
    seeds = [int(value) for value in history_seeds]
    if len(set(seeds)) != len(seeds):
        raise _refused(
            "REFUSED-SEED-COLLISION", "paired history seeds are not unique"
        )
    for seed in seeds:
        if CERTIFIED_OLD_SEED_MIN <= seed <= CERTIFIED_OLD_SEED_MAX:
            raise _refused(
                "REFUSED-SEED-COLLISION",
                f"paired history seed {seed} hits the certified old block",
            )
        if seed in EXECUTED_M8_HISTORY_SEEDS:
            raise _refused(
                "REFUSED-SEED-COLLISION",
                f"paired history seed {seed} reuses an executed M8 history seed",
            )
    probe = {pair_seed(seed, index) for seed in seeds for index in range(4)}
    if probe & set(int(value) for value in used_slot_seeds):
        raise _refused(
            "REFUSED-SEED-COLLISION", "paired pair-seed probe hits a used slot seed"
        )
    if any(
        CERTIFIED_OLD_SEED_MIN <= value <= CERTIFIED_OLD_SEED_MAX for value in probe
    ):
        raise _refused(
            "REFUSED-SEED-COLLISION", "paired pair-seed probe hits the certified old block"
        )
    return {
        "history_seeds": seeds,
        "n_histories": len(seeds),
        "pair_seed_probe": sorted(probe),
        "status": "PASS",
    }


# --- operating-characteristic / sample-size calculator --------------------------
# Deterministic, exact where stated. No embeddings, no forwards. Assumptions
# are returned beside every number and must travel with it.


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def sign_test_power(
    n_pairs: int, p_win: float, alpha: float = ALPHA_TWO_SIDED
) -> dict[str, Any]:
    """Exact two-sided sign-test power for N independent pairs.

    H0: P(win) = 0.5. Rejects when wins >= c or wins <= n - c with the
    largest c whose exact size stays within alpha. ``p_win`` is an ASSUMED
    win probability, never a measured anomaly effect.
    """
    if n_pairs <= 0:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "n_pairs must be positive")
    if not 0.0 < p_win < 1.0:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "p_win must be in (0, 1)")
    total = 2**n_pairs
    critical: int | None = None
    actual_alpha = 0.0
    for candidate in range(n_pairs + 1):
        upper = sum(math.comb(n_pairs, i) for i in range(candidate, n_pairs + 1))
        lower = sum(math.comb(n_pairs, i) for i in range(0, n_pairs - candidate + 1))
        size = (upper + lower) / total
        if size <= alpha:
            critical = candidate
            actual_alpha = size
            break
    if critical is None:  # pragma: no cover - unreachable for sane alpha
        raise _refused("REFUSED-UNKNOWN-DESIGN", "no critical value at this n/alpha")
    upper_power = sum(
        math.comb(n_pairs, i) * (p_win**i) * ((1.0 - p_win) ** (n_pairs - i))
        for i in range(critical, n_pairs + 1)
    )
    lower_power = sum(
        math.comb(n_pairs, i) * (p_win**i) * ((1.0 - p_win) ** (n_pairs - i))
        for i in range(0, n_pairs - critical + 1)
    )
    return {
        "n_pairs": n_pairs,
        "p_win_assumed": p_win,
        "alpha_nominal": alpha,
        "critical_value": critical,
        "alpha_actual": round(actual_alpha, 4),
        "power": round(upper_power + lower_power, 4),
        "assumption": (
            "p_win is ASSUMED (sensitivity grid), not a measured anomaly "
            "effect; M14 embedding perturbations measure responsiveness only."
        ),
    }


def one_sample_t_power(
    n: int, delta: float, alpha: float = ALPHA_TWO_SIDED
) -> dict[str, Any]:
    """Normal-approximation one-sided one-sample power for standardized delta.

    Models Stage 1 (mean paired contrast vs the calibrated null center).
    ``delta`` is an ASSUMED standardized shift; no paired-effect measurement
    exists on the exposed cache.
    """
    if n <= 1:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "n must exceed 1")
    z_alpha = -_normal_cdf_inverse(alpha)
    power = 1.0 - _normal_cdf(z_alpha - delta * math.sqrt(n))
    return {
        "n": n,
        "delta_assumed": delta,
        "alpha_one_sided": alpha,
        "z_alpha": round(z_alpha, 4),
        "power_approx": round(power, 4),
        "method": "normal approximation; ASSUMED standardized shift",
    }


def _normal_cdf_inverse(probability: float) -> float:
    """Acklam-style inverse normal CDF (deterministic, no dependency)."""
    if not 0.0 < probability < 1.0:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "probability must be in (0, 1)")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    lower, upper = 0.02425, 1.0 - 0.02425
    if probability < lower:
        q = math.sqrt(-2.0 * math.log(probability))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if probability <= upper:
        q = probability - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - probability))
    num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
    den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return -num / den


def clopper_pearson(
    successes: int, trials: int, alpha: float = ALPHA_TWO_SIDED
) -> dict[str, Any]:
    """Exact two-sided Clopper-Pearson interval for a detection fraction."""
    if not 0 <= successes <= trials or trials <= 0:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "invalid successes/trials")
    lower = (
        0.0
        if successes == 0
        else _beta_quantile(alpha / 2.0, successes, trials - successes + 1)
    )
    upper = (
        1.0
        if successes == trials
        else _beta_quantile(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return {
        "successes": successes,
        "trials": trials,
        "fraction": round(successes / trials, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "method": "Clopper-Pearson exact",
    }


def _beta_quantile(probability: float, alpha_shape: float, beta_shape: float) -> float:
    """Inverse regularized incomplete beta via bisection (deterministic)."""
    if alpha_shape <= 0 or beta_shape <= 0:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "beta shapes must be positive")
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _beta_cdf(mid, alpha_shape, beta_shape) < probability:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _beta_cdf(value: float, alpha_shape: float, beta_shape: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return _beta_regularized_cf(value, alpha_shape, beta_shape)


def _beta_regularized_cf(
    value: float, alpha_shape: float, beta_shape: float
) -> float:
    """Continued-fraction evaluation of the regularized incomplete beta.

    Modified Lentz's method after Numerical Recipes ``betacf``: the
    fraction terms update the convergent through the ``c``/``d`` pair, so
    no update cancels out of the quotient.
    """
    tiny, stop, eps = 1e-300, 200, 3e-14
    if value > (alpha_shape + 1.0) / (alpha_shape + beta_shape + 2.0):
        return 1.0 - _beta_regularized_cf(
            1.0 - value, beta_shape, alpha_shape
        )
    front = (
        math.exp(
            math.lgamma(alpha_shape + beta_shape)
            - math.lgamma(alpha_shape)
            - math.lgamma(beta_shape)
            + alpha_shape * math.log(value)
            + beta_shape * math.log(1.0 - value)
        )
        / alpha_shape
    )
    qab, qap, qam = alpha_shape + beta_shape, alpha_shape + 1.0, alpha_shape - 1.0
    converging = 1.0
    first = 1.0
    second = 1.0 - qab * value / qap
    if abs(second) < tiny:
        second = tiny
    second = 1.0 / second
    converging = second
    for term in range(1, stop + 1):
        twice = 2 * term
        numerator = (
            term
            * (beta_shape - term)
            * value
            / ((qam + twice) * (alpha_shape + twice))
        )
        second = 1.0 + numerator * second
        if abs(second) < tiny:
            second = tiny
        first = 1.0 + numerator / first
        if abs(first) < tiny:
            first = tiny
        second = 1.0 / second
        converging *= second * first
        numerator = (
            -(alpha_shape + term)
            * (qab + term)
            * value
            / ((alpha_shape + twice) * (qap + twice))
        )
        second = 1.0 + numerator * second
        if abs(second) < tiny:
            second = tiny
        first = 1.0 + numerator / first
        if abs(first) < tiny:
            first = tiny
        second = 1.0 / second
        step = first * second
        converging *= step
        if abs(step - 1.0) < eps:
            break
    return max(0.0, min(1.0, front * converging))


def dilution_counts(arm_size: int, burden: float) -> dict[str, Any]:
    """Exact abnormal/ordinary file counts for one diluted population arm."""
    if arm_size <= 0 or not 0.0 <= burden <= 1.0:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "invalid arm size or burden")
    n_abnormal = int(math.floor(burden * arm_size + 0.5))
    return {
        "arm_size": arm_size,
        "burden": burden,
        "n_abnormal": n_abnormal,
        "n_ordinary": arm_size - n_abnormal,
        "realized_burden": round(n_abnormal / arm_size, 4),
        "rule": "n_abnormal = floor(b * N + 0.5), inherited from M13",
    }


def power_grid(
    n_grid: Sequence[int] = (24, 48, 96),
    p_grid: Sequence[float] = (0.65, 0.75, 0.85),
    delta_grid: Sequence[float] = (0.5, 0.8, 1.0),
    alpha: float = ALPHA_TWO_SIDED,
) -> dict[str, Any]:
    """Full operating-characteristic grid (exact sign power + t approximation)."""
    return {
        "n_grid": [int(value) for value in n_grid],
        "p_grid": [float(value) for value in p_grid],
        "delta_grid": [float(value) for value in delta_grid],
        "alpha": alpha,
        "sign_power": [
            sign_test_power(int(n), float(p), alpha)
            for n in n_grid
            for p in p_grid
        ],
        "t_power": [
            one_sample_t_power(int(n), float(delta), alpha)
            for n in n_grid
            for delta in delta_grid
        ],
    }


# --- contract builder / verifier ----------------------------------------------


def design_summary() -> dict[str, Any]:
    """Return the frozen design constants as a machine-readable mapping."""
    return {
        "design_id": DESIGN_ID,
        "contract_id": CONTRACT_ID,
        "criterion": CRITERION,
        "no_execution_in_design": True,
        "cumulative_forwards_unchanged": CUMULATIVE_FORWARDS_BEFORE,
        "null_seed": NULL_SEED,
        "frozen_encoder_seeds": list(FROZEN_ENCODER_SEEDS),
        "paired_histories": list(PAIRED_HISTORY_IDS),
        "primary_triple": list(PRIMARY_TRIPLE),
        "replication_triple": list(REPLICATION_TRIPLE),
        "proposed_history_seeds": list(PROPOSED_HISTORY_SEEDS),
        "pair_seed_domain": PAIR_SEED_DOMAIN,
        "paired_families": list(PAIRED_FAMILIES),
        "paired_severities": list(PAIRED_SEVERITIES),
        "stage1_seed_grid": list(STAGE1_SEED_GRID),
        "stage1_expand_max": STAGE1_EXPAND_MAX,
        "stage1_futility_look_n": STAGE1_FUTILITY_LOOK_N,
        "stage2_sizes": {key: list(value) for key, value in STAGE2_SIZES.items()},
        "stage2_burdens": dict(STAGE2_BURDENS),
        "business_baseline_rule": BUSINESS_BASELINE_RULE,
        "grouping_rule": GROUPING_RULE,
        "primary_estimator": PRIMARY_ESTIMATOR,
        "sensitivity_estimator": SENSITIVITY_ESTIMATOR,
        "occupancy_estimator": OCCUPANCY_ESTIMATOR,
        "cluster_counts": list(CLUSTER_COUNTS),
        "primary_k": PRIMARY_K,
        "reference_files_per_history": REFERENCE_FILES_PER_HISTORY,
        "reference_causal_rule": REFERENCE_CAUSAL_RULE,
        "low_burden_band": list(LOW_BURDEN_OPERATING_BAND),
        "elevated_band": list(ELEVATED_OPERATING_BAND),
        "decision_branches": list(DECISION_BRANCHES),
        "proposed_detection_threshold": PROPOSED_DETECTION_THRESHOLD,
        "proposed_futility_cap": PROPOSED_FUTILITY_CAP,
        "proposed_gray_zone": list(PROPOSED_GRAY_ZONE),
        "alpha_two_sided": ALPHA_TWO_SIDED,
    }


def build_contract(
    false_alarm_anchors: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the proposed machine-readable contract (deterministic)."""
    if not isinstance(false_alarm_anchors, Mapping) or not false_alarm_anchors:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "false-alarm anchors are required")
    if not isinstance(provenance, Mapping) or not provenance:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "provenance is required")
    grid = power_grid()
    dilution = {
        horizon: {
            str(size): {
                name: dilution_counts(int(size), float(burden))
                for name, burden in STAGE2_BURDENS.items()
            }
            for size in sizes
        }
        for horizon, sizes in STAGE2_SIZES.items()
    }
    cp_reference = {
        "method": "Clopper-Pearson exact (regularized incomplete beta, Lentz)",
        "rows": [clopper_pearson(successes, trials) for successes, trials in _CP_REFERENCE_ROWS],
    }
    contract = {
        "contract_id": CONTRACT_ID,
        "design_id": DESIGN_ID,
        "state": "PROPOSED-AWAITING-OWNER-APPROVAL",
        "not_an_execution_release": True,
        "design": design_summary(),
        "false_alarm_anchors": dict(false_alarm_anchors),
        "operating_characteristics": grid,
        "clopper_pearson_reference": cp_reference,
        "dilution_ladder": dilution,
        "owner_decisions_required": list(_owner_decisions()),
        "api_gaps": list(_api_gaps()),
        "decision_rule": _decision_rule(),
        "execution_boundary": _execution_boundary(),
        "provenance": dict(provenance),
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _owner_decisions() -> tuple[dict[str, Any], ...]:
    return (
        {
            "id": "D1",
            "question": "Approve the paired-twin constructor interface for the execution task.",
            "context": "GAP-PAIRED-TWIN-01: no public API returns the unmodified twin of an anomaly-path draw.",
        },
        {
            "id": "D2",
            "question": "Approve business burden levels for Stage-2 baselines and elevated arms.",
            "context": "Proposed LOW 0.05 / MID 0.15 / ELEVATED 0.275; baselines stay low-nonzero, never zero-required.",
        },
        {
            "id": "D3",
            "question": "Approve the Stage-1 seed grid (24 primary, 48 planned, 96 gray-zone cap) and the futility rule.",
            "context": "Exact sign-test and t-approximation powers are embedded in operating_characteristics.",
        },
        {
            "id": "D4",
            "question": "Approve the proposed numerical decision thresholds as release criteria.",
            "context": "Thresholds are proposals, not gates; non-significance is never read as absence.",
        },
        {
            "id": "D5",
            "question": "Release official paired-corpus generation and frozen-encoder execution (forward budget).",
            "context": "No generation, inference, training, refit, remote or scale work executes under this design.",
        },
        {
            "id": "D6",
            "question": "Rule on duration_anomaly scope given its T/regime change.",
            "context": "Pairs within family only, or exclude; owner decides before the execution freeze.",
        },
    )


def _api_gaps() -> tuple[dict[str, Any], ...]:
    return (
        {
            "id": "GAP-PAIRED-TWIN-01",
            "statement": (
                "No public generator API returns the unmodified twin of an "
                "anomaly-path draw: generate_normal(seed) and "
                "generate_anomaly(seed) at equal seeds use independent base "
                "draws (roster, regimes, T, noise position all diverge; "
                "measured seed 42401: normal T=386 roster=(4, 2) vs anomaly "
                "T=450 roster=(3, 1)). Reconstructing the twin replays "
                "non-public internals, so the twin constructor is new "
                "implementation for the execution task, silently approximated nowhere."
            ),
            "feasible_without_implementation": [
                "deterministic replay of every public path (byte-identical)",
                "same-seed severity/family sweeps within the anomaly path share the exact base",
                "label-free encoder inputs and realized-mask annotations",
                "temp-output-only tiny smoke",
            ],
        },
        {
            "id": "INV-PAIRED-BASE-01",
            "statement": (
                "Verified paired invariant: at fixed seed, the anomaly-path "
                "base (T, regime bounds, roster, noise_seed position) is "
                "identical across severities and across non-duration "
                "families; duration_anomaly changes T/regimes and pairs "
                "within family only."
            ),
        },
        {
            "id": "INV-NOISE-COUPLING-01",
            "statement": (
                "Verified coupling limit: post-injection noise is "
                "signal-dependent (S1 local-variance scaling, S2 "
                "local-std-scaled walk), so twin and abnormal differ outside "
                "the mask too. Pairing controls the pre-noise base and the "
                "noise seed; realized noise differences are part of the "
                "contrast, disclosed not fixed."
            ),
        },
    )


def _decision_rule() -> dict[str, Any]:
    return {
        "stage1_primary": (
            "Detection fraction of per-seed paired contrasts above the frozen "
            "null q95; PROPOSED bar >= 0.5 with the exact lower bound above "
            "the measured false-alarm anchor (owner approval D4)."
        ),
        "stage2_primary": (
            "E1 transition median above the context-only control q95 at "
            "elevated burden, weekly and monthly; dilution ladder low/mid must "
            "show monotone non-increasing response (descriptive)."
        ),
        "branches": {
            "SCALE": "Stage-1 bar met and Stage-2 primary met, context controls clean.",
            "MEASURE-DILUTION": "Stage-1 bar met but Stage-2 primary missed; investigate sample-size/measurement dilution.",
            "REPRESENTATION": "Stage-1 bar missed at N=96 with clean context; bounded negative, change representation/objective.",
            "CONTEXT": "Context-only controls move; resolve confounding before any scale claim.",
        },
        "non_significance": "never read as absence; bounded-negative only at the N=96 cap with clean context.",
    }


def _execution_boundary() -> dict[str, Any]:
    return {
        "this_design_authorizes": [
            "tiny public-path feasibility smoke (temp outputs only)",
            "deterministic operating-characteristic calculator",
            "proposed machine-readable contract",
        ],
        "this_design_forbids": [
            "official paired-corpus generation",
            "embedding capture, model loads, forward passes",
            "training, bank refit, remote execution, scale, Phase C",
            "edits to accepted M11-M14 products",
        ],
        "next_task_owns": [
            "paired-twin constructor implementation (GAP-PAIRED-TWIN-01)",
            "official generation freeze and execution",
            "frozen-encoder capture and two-stage evaluation",
        ],
    }


def verify_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the contract is the deterministic build output."""
    if not isinstance(contract, Mapping):
        raise _refused("REFUSED-CONTRACT-DRIFT", "contract is not a mapping")
    if contract.get("contract_id") != CONTRACT_ID:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "unknown contract id")
    if contract.get("design_id") != DESIGN_ID:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "unknown design id")
    recorded = contract.get("contract_sha256")
    rebuilt = _canonical_sha256(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    )
    if recorded != rebuilt:
        raise _refused(
            "REFUSED-CONTRACT-DRIFT",
            "contract bytes do not match their recorded digest",
        )
    return {"status": "PASS", "contract_sha256": recorded}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    from representation.sprint24_context_audit import canonical_json

    return hashlib.sha256((canonical_json(dict(payload)) + "\n").encode("utf-8")).hexdigest()


__all__ = [
    "ALPHA_TWO_SIDED",
    "BUSINESS_BASELINE_RULE",
    "CERTIFIED_OLD_SEED_MAX",
    "CERTIFIED_OLD_SEED_MIN",
    "CLUSTER_COUNTS",
    "CONTRACT_ID",
    "CRITERION",
    "CUMULATIVE_FORWARDS_BEFORE",
    "DECISION_BRANCHES",
    "DESIGN_ID",
    "ELEVATED_OPERATING_BAND",
    "EXECUTED_M8_HISTORY_SEEDS",
    "FROZEN_ENCODER_SEEDS",
    "GROUPING_RULE",
    "LOW_BURDEN_CEILING",
    "LOW_BURDEN_OPERATING_BAND",
    "NULL_SEED",
    "PAIR_SEED_DOMAIN",
    "PAIRED_FAMILIES",
    "PAIRED_HISTORY_IDS",
    "PAIRED_SEVERITIES",
    "PRIMARY_K",
    "PRIMARY_TRIPLE",
    "PROPOSED_DETECTION_THRESHOLD",
    "PROPOSED_FUTILITY_CAP",
    "PROPOSED_GRAY_ZONE",
    "PROPOSED_HISTORY_SEEDS",
    "REFUSAL_REASONS",
    "REFERENCE_CAUSAL_RULE",
    "REFERENCE_FILES_PER_HISTORY",
    "REPLICATION_TRIPLE",
    "STAGE1_EXPAND_MAX",
    "STAGE1_FUTILITY_LOOK_N",
    "STAGE1_SEED_GRID",
    "STAGE2_BURDENS",
    "STAGE2_SIZES",
    "build_contract",
    "check_seed_plan",
    "clopper_pearson",
    "design_summary",
    "dilution_counts",
    "one_sample_t_power",
    "pair_seed",
    "power_grid",
    "sign_test_power",
    "verify_contract",
]
