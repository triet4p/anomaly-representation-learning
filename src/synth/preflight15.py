"""Sprint 15 candidate-7 rejection-only structural preflight (protocol v7 §1b).

Runs the frozen structural chain — calendar construction, eligibility,
exact-CSP witness, per-subtype pools, control-window selection, and support
qualification — for fixed seeds without writes. It invokes the actual calendar,
health, split, and waveform path in memory; it persists no candidate shards,
manifests, or role roots. Any miss rejects only the supplied attempt. Passing
does not select or promote a history.

Sprint 18 profiles reuse the same chain. C2 retains its historical behavior;
iterative-v1 composes the C2 failure rate with the reviewed exposure-only
calendar change and applies the role-specific early structural predicates.
All profiles fail closed on an unknown name.

The module imports standard-library helpers, ``synth`` calendar/allocation
predicates, and the existing Task 5 structural summary. It never imports
dataset persistence, seals, scoring, or probe evaluation, and it performs no
filesystem writes.

"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict
from typing import cast

from synth import balanced as B
#: Frozen preflight profile (protocol v7 §1b — the revised method).
PREFLIGHT_PROFILE = "sprint15-v7"

#: Frozen preflight protocol tag.
PREFLIGHT_PROTOCOL = "sprint15-benchmark-protocol-v7"

#: Sprint 18 Cycle 2 preflight profile (method ``sprint18-data-method-c2-v1``
#: §5 C2-M1). Labels histories built by ``sprint18_c2_history_config``; the
#: generator protocol bytes stay frozen v7.
PREFLIGHT_C2_PROFILE = "sprint18-c2"

#: Generator protocol tag reported by C2 preflight (frozen v7 bytes).
PREFLIGHT_C2_PROTOCOL = "sprint15-benchmark-protocol-v7"

#: Sprint 18 iterative-v1 profile and inherited manifest protocol.
PREFLIGHT_ITERATIVE_PROFILE = "sprint18-iterative-v1"
PREFLIGHT_ITERATIVE_PROTOCOL = "sprint15-benchmark-protocol-v7"
_ITERATIVE_ROLES = {
    "DESIGN", "FIT", "CALIBRATION", "DEVELOPMENT", "CONFIRMATION"
}

#: Expected cohort-A ``abrupt_rate`` per preflight profile. The selected
#: factory's resolved config must carry its profile's literal; any mismatch
#: raises instead of running mislabeled preflight.
_PREFLIGHT_ABRUPT_RATE = {
    PREFLIGHT_PROFILE: 2.2e-5,
    PREFLIGHT_C2_PROFILE: 3.3e-5,
    PREFLIGHT_ITERATIVE_PROFILE: 3.3e-5,
}


def _resolve_preflight(profile: str):
    """Resolve a preflight profile to its factory, label, and protocol.

    Unknown profiles raise rather than silently falling back to v7.
    """
    from synth.chronicle import sprint15_v7_history_config
    from synth.chronicle import sprint18_c2_history_config
    from synth.chronicle import sprint18_iterative_v1_history_config

    if profile == PREFLIGHT_PROFILE:
        return sprint15_v7_history_config, PREFLIGHT_PROFILE, PREFLIGHT_PROTOCOL
    if profile == PREFLIGHT_C2_PROFILE:
        return (
            sprint18_c2_history_config,
            PREFLIGHT_C2_PROFILE,
            PREFLIGHT_C2_PROTOCOL,
        )
    if profile == PREFLIGHT_ITERATIVE_PROFILE:
        return (
            sprint18_iterative_v1_history_config,
            PREFLIGHT_ITERATIVE_PROFILE,
            PREFLIGHT_ITERATIVE_PROTOCOL,
        )
    raise ValueError(f"unknown preflight profile {profile!r}")


def _preflight_seed_for(
    history_seed: int, profile: str, role: str | None = None
) -> dict[str, object]:
    """Run the shared in-memory rejection-only chain for one seed.

    Iterative-v1 requires the fixed roster role so the applicable Design or
    Confirmation gate is explicit. The actual scheduler, health, signal,
    temporal, and split path runs in memory, but no candidate root is written.
    """
    factory, label, _ = _resolve_preflight(profile)
    if label == PREFLIGHT_ITERATIVE_PROFILE and role not in _ITERATIVE_ROLES:
        raise ValueError(
            "sprint18-iterative-v1 preflight requires one of "
            f"{sorted(_ITERATIVE_ROLES)!r} as role"
        )

    from synth.chronicle import Patchifier
    from synth.chronicle import _failure_to_dict
    from synth.chronicle import _maintenance_windows
    from synth.chronicle import _manifest_file_rows
    from synth.chronicle import build_chronological

    t0 = time.perf_counter()
    cfg = factory(seed=history_seed)
    expected = _PREFLIGHT_ABRUPT_RATE[label]
    actual = next(
        cohort.abrupt_rate
        for cohort in cfg.health.cohorts
        if cohort.cohort_id == "A"
    )
    if actual != expected:
        raise ValueError(
            f"preflight profile {label!r} resolved abrupt_rate {actual!r}, "
            f"expected {expected!r}"
        )
    schedule, health, labeled, splits = build_chronological(cfg)
    wins = _maintenance_windows(health)
    patchifier = Patchifier(cfg.patch)
    patch_counts = {
        sample.file_id: int(patchifier.patchify(sample).patches.shape[0])
        for sample in labeled
    }
    rows = _manifest_file_rows(labeled, wins, patch_counts)
    ledger = [_failure_to_dict(record) for record in health.failure_events]
    inputs = B.project_allocation_inputs(rows)
    eligible, _, _ = B.eligible_anchors(inputs, ledger, wins)
    pools = {
        f"{cohort}/{subtype}": sum(
            1 for failure in eligible
            if failure["cohort"] == cohort and failure["subtype"] == subtype
        )
        for cohort, subtype in B.BUCKET_ORDER
    }

    if label != PREFLIGHT_ITERATIVE_PROFILE:
        try:
            allocation = B.allocate_quotas(
                rows, ledger, wins, history_seed, B.DEFAULT_QUOTA, "exact"
            )
        except (B.InfeasibleCandidate, B.AllocationUnavailable) as exc:
            return {
                "history_seed": history_seed,
                "profile": label,
                "feasible": False,
                "reason": exc.record.get("reason"),
                "missing": exc.record.get("missing"),
                "eligible_pool": exc.record.get("eligible_pool"),
                "rejection": exc.record.get("rejection"),
                "solver": exc.record.get("solver"),
                "pools": pools,
                "elapsed_s": round(time.perf_counter() - t0, 1),
                "no_write": True,
            }
        except RuntimeError as exc:
            return {
                "history_seed": history_seed,
                "profile": label,
                "feasible": False,
                "reason": "witness-breach",
                "missing": str(exc),
                "pools": pools,
                "elapsed_s": round(time.perf_counter() - t0, 1),
                "no_write": True,
            }
        selected = cast(
            "dict[str, dict[str, list[str]]]", allocation["selected"]
        )
        solver = cast("dict[str, object]", allocation["solver"])
        return {
            "history_seed": history_seed,
            "profile": label,
            "feasible": True,
            "pools": pools,
            "allocated": {
                cohort: {subtype: len(ids) for subtype, ids in subtypes.items()}
                for cohort, subtypes in selected.items()
            },
            "controls": len(cast("list[object]", allocation["controls"])),
            "margins": allocation["margins"],
            "rejection": allocation["rejection"],
            "solver": {
                key: solver[key] for key in B.SOLVER_MANIFEST_KEYS
            },
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "no_write": True,
        }

    split_ids = {
        "dev_train": list(splits.dev_train),
        "dev_val": list(splits.dev_val),
        "test_static": list(splits.test_static),
        "test_temporal": list(splits.test_temporal),
        "quarantined": list(splits.quarantined),
        "failed_episode_ids": list(splits.failed_episode_ids),
    }
    labels = Counter(sample.file_label.value for sample in labeled)
    manifest = {
        "resolved_config": json.loads(
            json.dumps(asdict(cfg), sort_keys=True, default=str)
        ),
        "calendar": {"cutoff_time": splits.cutoff_time},
        "schedule": schedule.events,
        "failure_events": ledger,
        "maintenance_windows": wins,
        "files": rows,
        "splits": split_ids,
        "counts": {
            "total": len(labeled),
            "normal": labels["normal"],
            "abnormal": labels["abnormal"],
            "dev_train": len(splits.dev_train),
            "dev_val": len(splits.dev_val),
            "test_static": len(splits.test_static),
            "test_temporal": len(splits.test_temporal),
            "quarantined": len(splits.quarantined),
        },
    }
    from experiments.sprint18_task5_measurability import (
        confirmation_hard_floor_checks,
        design_promotion_checks,
        structural_summary,
    )

    entry = {
        "history_id": f"S18I-PREFLIGHT-{history_seed}",
        "role": role or "PREFLIGHT",
        "data_seed": history_seed,
    }
    structural = structural_summary(entry, manifest)
    support = structural["support"]
    quota = B.DEFAULT_QUOTA
    construction_checks = {
        f"eligible_{subtype}_pool_ge_{need}": pools[f"{cohort}/{subtype}"] >= need
        for cohort, subtype, need in (
            ("P", "P1", quota.p1), ("P", "P2", quota.p2),
            ("W", "W1", quota.w1), ("W", "W2", quota.w2),
            ("A", "A1", quota.a1), ("A", "A2", quota.a2),
        )
    }
    construction_checks.update({
        "selected_controls_ge_48": (
            support["negative_control_windows"] >= quota.controls
        ),
        "evaluable_robot_days_ge_240": (
            support["evaluable_robot_days"] >= quota.robot_days
        ),
        "positive_robots_ge_6": (
            support["positive_contributing_robots"] >= quota.robots_pos
        ),
        "negative_robots_ge_6": (
            support["negative_contributing_robots"] >= quota.robots_neg
        ),
        "at_least_2_programs_each_P_W": all(
            len(support["programs_by_cohort"][cohort]) >= quota.programs
            for cohort in ("P", "W")
        ),
    })
    design_checks = design_promotion_checks(structural)
    confirmation_checks = confirmation_hard_floor_checks(structural)
    role_checks = (
        design_checks if role == "DESIGN"
        else confirmation_checks if role == "CONFIRMATION"
        else {}
    )
    qualification_checks = {
        "structural_core_integrity": bool(structural["core_integrity_pass"]),
        **construction_checks,
        **{f"role_gate/{name}": passed for name, passed in role_checks.items()},
    }
    qualification_pass = all(qualification_checks.values())

    allocation = None
    allocation_failure: dict[str, object] | None = None
    allocation_failure_reason: str | None = None
    try:
        allocation = B.allocate_quotas(
            rows, ledger, wins, history_seed, quota, "exact"
        )
    except (B.InfeasibleCandidate, B.AllocationUnavailable) as exc:
        allocation_failure = exc.record
        allocation_failure_reason = str(
            exc.record.get("reason") or "allocator-rejected"
        )
    except RuntimeError as exc:
        allocation_failure_reason = "witness-breach"
        allocation_failure = {"missing": str(exc)}

    selected_counts = None
    controls = support["negative_control_windows"]
    margins = None
    rejection = dict(structural["support"]["eligibility_rejections"])
    solver: dict[str, object] | None = None
    if allocation is not None:
        selected = cast(
            "dict[str, dict[str, list[str]]]", allocation["selected"]
        )
        selected_counts = {
            cohort: {subtype: len(ids) for subtype, ids in subtypes.items()}
            for cohort, subtypes in selected.items()
        }
        controls = len(cast("list[object]", allocation["controls"]))
        margins = allocation["margins"]
        rejection = allocation["rejection"]
        solver = cast("dict[str, object]", allocation["solver"])
    elif allocation_failure is not None:
        failure_solver = allocation_failure.get("solver")
        if isinstance(failure_solver, dict):
            solver = failure_solver

    feasible = allocation is not None and qualification_pass
    if allocation_failure_reason is not None:
        reason = allocation_failure_reason
    else:
        reason = None if qualification_pass else "qualification-shortfall"
    missing = (
        allocation_failure.get("missing")
        if allocation_failure is not None
        else [name for name, passed in qualification_checks.items() if not passed]
    )
    return {
        "history_seed": history_seed,
        "profile": label,
        "role": role,
        "feasible": feasible,
        "reason": reason,
        "missing": missing,
        "eligible_pool": (
            allocation_failure.get("eligible_pool")
            if allocation_failure is not None else
            {cohort: sum(count for key, count in pools.items()
                         if key.startswith(f"{cohort}/"))
             for cohort in ("P", "W", "A")}
        ),
        "rejection": rejection,
        "solver": solver,
        "pools": pools,
        "allocated": selected_counts,
        "controls": controls,
        "margins": margins,
        "qualification": {
            "passed": qualification_pass,
            "checks": qualification_checks,
            "construction_checks": construction_checks,
            "design_promotion_checks": design_checks,
            "confirmation_hard_floor_checks": confirmation_checks,
            "applied_role_gate": role,
            "full_eligible_cohort_mix": support["cohort_mix"],
            "full_eligible_lead_support": support["lead_support"],
            "support": support,
            "structural_checks": structural["checks"],
        },
        "in_memory_waveforms_generated": True,
        "persisted_candidate_root": False,
        "persisted_candidate_shards": False,
        "persisted_candidate_manifest": False,
        "no_write": True,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def preflight_seed(
    history_seed: int,
    profile: str = PREFLIGHT_PROFILE,
    *,
    role: str | None = None,
) -> dict[str, object]:
    """Run the no-write source-computable structural chain for one seed.

    Iterative-v1 requires its prospective fixed role, applying the Design or
    Confirmation gate where applicable. Other profiles retain their previous
    no-role call shape.
    """
    return _preflight_seed_for(history_seed, profile, role)


def preflight_seed_c2(history_seed: int) -> dict[str, object]:
    """Run the no-write structural chain for one fixed seed under C2.

    Explicit fail-closed C2 entry: identical chain to :func:`preflight_seed`
    through ``sprint18_c2_history_config`` with profile ``sprint18-c2``.
    Writes nothing.
    """
    return _preflight_seed_for(history_seed, PREFLIGHT_C2_PROFILE)


def preflight_seed_iterative_v1(
    history_seed: int, role: str
) -> dict[str, object]:
    """Run iterative-v1 rejection-only preflight for one fixed role seed."""
    return _preflight_seed_for(
        history_seed, PREFLIGHT_ITERATIVE_PROFILE, role
    )


def _run_preflight_for(
    seeds: list[int],
    profile: str,
    roles: list[str] | None = None,
) -> dict[str, object]:
    """Run a fixed seed list in caller order, without selection or writes."""
    _, _, protocol = _resolve_preflight(profile)
    if roles is not None and len(roles) != len(seeds):
        raise ValueError("preflight roles must match the seed-list length")
    if profile == PREFLIGHT_ITERATIVE_PROFILE and roles is None:
        raise ValueError("iterative-v1 roster preflight requires fixed roles")
    results = [
        _preflight_seed_for(seed, profile, role)
        for seed, role in zip(seeds, roles or [None] * len(seeds), strict=True)
    ]
    feasible = sum(1 for r in results if r["feasible"])
    return {
        "protocol": protocol,
        "seeds": list(seeds),
        "results": results,
        "feasible_count": f"{feasible}/{len(results)}",
        "verdict": ("PREFLIGHT-PASS"
                    if feasible == len(results) and results
                    else "PREFLIGHT-FAIL"),
    }


def run_preflight(
    seeds: list[int],
    profile: str = PREFLIGHT_PROFILE,
    *,
    roles: list[str] | None = None,
) -> dict[str, object]:
    """Run the fixed seed list in caller order, with role-specific gates.

    Iterative-v1 requires one frozen role per seed; legacy profile callers
    retain their previous signatures and unanimity behavior.
    """
    return _run_preflight_for(seeds, profile, roles)


def run_preflight_c2(seeds: list[int]) -> dict[str, object]:
    """Run :func:`preflight_seed_c2` over a fixed seed list, in order.

    Explicit fail-closed C2 roster entry: identical unanimity semantics to
    :func:`run_preflight` through ``sprint18_c2_history_config`` with
    profile ``sprint18-c2``. Writes nothing.
    """
    return _run_preflight_for(seeds, PREFLIGHT_C2_PROFILE)


def run_preflight_iterative_v1(
    seeds: list[int], roles: list[str]
) -> dict[str, object]:
    """Run iterative-v1 preflight over the exact seed/role order supplied."""
    return _run_preflight_for(
        seeds, PREFLIGHT_ITERATIVE_PROFILE, roles
    )
