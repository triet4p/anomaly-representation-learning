"""Sprint 15 Task 9 cycle-7: rejection-only roster preflight (candidate 7).

Verifies the frozen Candidate 7 roster binding, then runs the deterministic
no-write structural preflight over all 18 fixed seeds 1600-1617 plus replay
regression of retired failure seeds 1306 and 1509 under the v7 method. Writes
``artifacts/sprint-15/preflight-cycle-7.json`` and prints a machine-readable
summary (recorded in ``artifacts/sprint-15/task-9-cycle-7.md``).

Generates no waveforms, shards, manifests, seals, or observable metrics.
Any miss retires the candidate; passing merely authorizes the normal gates.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("artifacts/sprint-15/preflight-cycle-7.json")

EXPECTED_ROSTER = {
    1600: ("DESIGN", "H-DESIGN-37"),
    1601: ("DESIGN", "H-DESIGN-38"),
    1602: ("DESIGN", "H-DESIGN-39"),
    1603: ("DESIGN", "H-DESIGN-40"),
    1604: ("FIT", "H-FIT-28"),
    1605: ("FIT", "H-FIT-29"),
    1606: ("FIT", "H-FIT-30"),
    1607: ("CALIBRATION", "H-CAL-10"),
    1608: ("CONFIRMATION", "H-CONF-34"),
    1609: ("CONFIRMATION", "H-CONF-35"),
    1610: ("CONFIRMATION", "H-CONF-36"),
    1611: ("CONFIRMATION", "H-CONF-37"),
    1612: ("SEALED", "H-SEAL-37"),
    1613: ("SEALED", "H-SEAL-38"),
    1614: ("SEALED", "H-SEAL-39"),
    1615: ("SEALED", "H-SEAL-40"),
    1616: ("PROOF", "H-PROOF-13"),
    1617: ("PROOF", "H-PROOF-14"),
}

REPLAY_SEEDS = (1306, 1509)


def main() -> int:
    from synth import balanced as B
    from synth import preflight15 as P

    live = {s: B.S15_ROSTER[s] for s in range(1600, 1618)}
    assert live == EXPECTED_ROSTER, "roster binding mismatch"
    assert B.S15_B7 == 1600
    assert B.S15_PROFILE_V7 == "sprint15-v7"
    assert B.S15_PROTOCOL_V7 == "sprint15-benchmark-protocol-v7"

    replay = [P.preflight_seed(seed) for seed in REPLAY_SEEDS]
    roster = P.run_preflight(list(range(1600, 1618)))
    record = {
        "protocol": P.PREFLIGHT_PROTOCOL,
        "roster_binding_ok": True,
        "replay_seeds": list(REPLAY_SEEDS),
        "replay": replay,
        "replay_feasible": all(r["feasible"] for r in replay),
        "roster": roster,
        "verdict": ("PREFLIGHT-PASS"
                    if roster["verdict"] == "PREFLIGHT-PASS"
                    and all(r["feasible"] for r in replay)
                    else "PREFLIGHT-FAIL"),
    }
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True))
    from typing import cast

    replay_rows = cast("list[dict[str, object]]", replay)
    roster_rows = cast("list[dict[str, object]]", roster["results"])
    print(json.dumps({
        "roster_binding_ok": True,
        "replay": [(r["history_seed"], r["feasible"],
                     r.get("controls"),
                     (r.get("solver") or {}).get("solver_status")
                     if isinstance(r.get("solver"), dict) else None)
                    for r in replay_rows],
        "roster_feasible": roster["feasible_count"],
        "controls": [(r["history_seed"], r.get("controls"))
                     for r in roster_rows if r["feasible"]],
        "misses": [(r["history_seed"], r.get("reason"), r.get("missing"))
                   for r in roster_rows if not r["feasible"]],
        "verdict": record["verdict"],
    }, indent=1, sort_keys=True))
    return 0 if record["verdict"] == "PREFLIGHT-PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
