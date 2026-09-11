"""Sprint 15 candidate-7 rejection-only structural preflight (protocol v7 §1b).

Runs the frozen structural chain — calendar construction, frozen-calendar
bytes, eligible anchor set, exact-CSP witness, per-subtype pool tallies,
control-window selection, cap/window/support arithmetic — for fixed seeds
with zero writes: no waveforms, no shards, no manifests, no seals, and no
observable metrics. Any miss retires the candidate under protocol §8; passing
merely authorizes the normal evidence gates. Rejection-only: fixed roster in,
verdict out — no selection, no tuning.

Waveform-blindness contract: this module imports only the standard library,
``synth.chronicle`` (calendar/profile builders), ``synth.balanced`` (frozen
allocation/audit machinery), and ``synth.events`` (frozen metadata predicates).
It never imports signal synthesis, dataset persistence, seals, scoring, or
probe evaluation, and it performs no filesystem writes. A focused test pins
these import and call-site constraints.
"""

from __future__ import annotations

import time
from typing import cast

from synth import balanced as B

#: Frozen preflight profile (protocol v7 §1b — the revised method).
PREFLIGHT_PROFILE = "sprint15-v7"

#: Frozen preflight protocol tag.
PREFLIGHT_PROTOCOL = "sprint15-benchmark-protocol-v7"


def preflight_seed(history_seed: int) -> dict[str, object]:
    """Run the no-write structural chain for one fixed seed.

    Builds the frozen calendar, derives eligibility, solves the exact CSP to a
    validated witness, selects control windows, and checks support margins —
    the production allocation path up to (excluding) waveform/shard/manifest
    persistence. Returns a feasible record on success; on any miss returns a
    record with ``feasible`` False carrying the machine-readable basis
    (proven-infeasible record, unavailable record, or witness-breach detail).
    Writes nothing.
    """
    from synth.chronicle import Patchifier
    from synth.chronicle import _failure_to_dict
    from synth.chronicle import _maintenance_windows
    from synth.chronicle import _manifest_file_rows
    from synth.chronicle import build_chronological
    from synth.chronicle import sprint15_v7_history_config

    t0 = time.perf_counter()
    cfg = sprint15_v7_history_config(seed=history_seed)
    _, health, labeled, _ = build_chronological(cfg)
    wins = _maintenance_windows(health)
    patchifier = Patchifier(cfg.patch)
    patch_counts = {
        sample.file_id: int(patchifier.patchify(sample).patches.shape[0])
        for sample in labeled
    }
    rows = _manifest_file_rows(labeled, wins, patch_counts)
    ledger = [_failure_to_dict(r) for r in health.failure_events]
    inputs = B.project_allocation_inputs(rows)
    eligible, _, _ = B.eligible_anchors(inputs, ledger, wins)
    pools = {
        f"{cohort}/{subtype}": sum(
            1 for f in eligible
            if f["cohort"] == cohort and f["subtype"] == subtype)
        for cohort, subtype in B.BUCKET_ORDER
    }
    try:
        allocation = B.allocate_quotas(
            rows, ledger, wins, history_seed, B.DEFAULT_QUOTA, "exact")
    except (B.InfeasibleCandidate, B.AllocationUnavailable) as exc:
        return {
            "history_seed": history_seed,
            "profile": PREFLIGHT_PROFILE,
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
            "profile": PREFLIGHT_PROFILE,
            "feasible": False,
            "reason": "witness-breach",
            "missing": str(exc),
            "pools": pools,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "no_write": True,
        }
    selected = cast(
        "dict[str, dict[str, list[str]]]", allocation["selected"])
    solver = cast("dict[str, object]", allocation["solver"])
    return {
        "history_seed": history_seed,
        "profile": PREFLIGHT_PROFILE,
        "feasible": True,
        "pools": pools,
        "allocated": {
            cohort: {st: len(ids) for st, ids in subs.items()}
            for cohort, subs in selected.items()
        },
        "controls": len(cast("list[object]", allocation["controls"])),
        "margins": allocation["margins"],
        "rejection": allocation["rejection"],
        "solver": {
            key: solver[key] for key in B.SOLVER_MANIFEST_KEYS},
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "no_write": True,
    }


def run_preflight(seeds: list[int]) -> dict[str, object]:
    """Run :func:`preflight_seed` over a fixed seed list, in order.

    The list is executed exactly as given — no filtering, reordering,
    skimming, or early exit: every seed gets exactly one attempt so the
    record is complete. Returns the roster record with per-seed results and
    the unanimous verdict (``PREFLIGHT-PASS`` only when every seed is
    feasible). Writes nothing.
    """
    results = [preflight_seed(seed) for seed in seeds]
    feasible = sum(1 for r in results if r["feasible"])
    return {
        "protocol": PREFLIGHT_PROTOCOL,
        "seeds": list(seeds),
        "results": results,
        "feasible_count": f"{feasible}/{len(results)}",
        "verdict": ("PREFLIGHT-PASS"
                    if feasible == len(results) and results
                    else "PREFLIGHT-FAIL"),
    }
