"""Sprint 15 balanced causal quota allocation, audit, and fixtures (Batch B).

Implements the frozen ``sprint15-benchmark-protocol-v1`` machinery on top of
the untouched chronological pipeline:

- :class:`QuotaConfig` — exact per-history quotas, subtype splits, caps, spacing.
- :func:`allocate_quotas` — dedicated-RNG quota selection over frozen-calendar
  anchors with fail-fast :class:`InfeasibleCandidate` rejection.
- :func:`audit_sprint15` — frozen structural audit (floors, promotion margins,
  mix/caps, lead support, controls, support projections, role isolation, EG1/EG3).
- EG3 deterministic fixture battery (:func:`eg3_fixtures`).
- Analytic nuisance/signal-margin gates (:func:`check_nuisance_envelopes`,
  :func:`analytic_signal_margin`).
- Sprint 15 materialization binding (:class:`Sprint15Binding`,
  :func:`prepare_sprint15_block`, :func:`calendar_digest`).

Waveform-blindness contract: this module imports only the standard library,
numpy, ``scipy`` (deferred ``scipy.optimize`` import inside the exact-path
solver only — no top-level dependency), ``synth.config`` (quota/profile
contracts), and ``synth.events`` (frozen metadata predicates over manifest
dicts). It never imports signal synthesis, temporal manifestation, dataset
persistence, or health/scheduler internals, and allocation consumes only
allowlisted file-row metadata — fixed before waveform synthesis — never
waveform amplitudes. A focused test pins these import and key-allowlist
constraints.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

from synth import events as E

#: Frozen Sprint 15 benchmark identifiers.
S15_PROFILE = "sprint15-v1"
S15_PROTOCOL = "sprint15-benchmark-protocol-v1"
S15_PROFILE_V2 = "sprint15-v2"
S15_PROTOCOL_V2 = "sprint15-benchmark-protocol-v2"
S15_PROFILE_V3 = "sprint15-v3"
S15_PROTOCOL_V3 = "sprint15-benchmark-protocol-v3"
S15_PROFILE_V4 = "sprint15-v4"
S15_PROTOCOL_V4 = "sprint15-benchmark-protocol-v4"
S15_PROFILE_V5 = "sprint15-v5"
S15_PROTOCOL_V5 = "sprint15-benchmark-protocol-v5"
S15_PROFILE_V6 = "sprint15-v6"
S15_PROTOCOL_V6 = "sprint15-benchmark-protocol-v6"
S15_PROFILE_V7 = "sprint15-v7"
S15_PROTOCOL_V7 = "sprint15-benchmark-protocol-v7"

#: Candidate-1 base seed (``B_1 = 1000``); Task 4 binding.
S15_B1 = 1000

#: Candidate-2 base seed (``B_2 = 1100``); Task 4 cycle-2 binding.
S15_B2 = 1100

#: Candidate-3 base seed (``B_3 = 1200``); Task 4 cycle-3 binding.
S15_B3 = 1200

#: Candidate-4 base seed (``B_4 = 1300``); Task 4 cycle-4 binding.
S15_B4 = 1300

#: Candidate-5 base seed (``B_5 = 1400``); Task 4 cycle-5 binding.
S15_B5 = 1400

#: Candidate-6 base seed (``B_6 = 1500``); Task 4 cycle-6 binding.
S15_B6 = 1500

#: Candidate-7 base seed (``B_7 = 1600``); Task 4 cycle-7 binding.
S15_B7 = 1600

#: Exact-CSP concentration caps (protocol v4 §3b; corrected Gate A F-01 floor
#: bounds — the frozen V2 concentration gates, evaluated against total
#: selected positives ``64`` and per-cohort ``P/W`` totals ``24``):
#: per-robot selected positives ``<= floor(0.35 * 64) = 22``;
#: per-program-within-each-predictable-cohort selected positives
#: ``<= floor(0.60 * 24) = 14``. The joint round-robin path keeps its frozen
#: ``ceil``-form loop-guard expressions for legacy candidates; only the exact
#: (v4) path uses these floor caps.
ROBOT_POS_CAP_MAX = 22
PROGRAM_CAP_MAX = 14

#: Seconds per day (mirrors the frozen event-window reference).
DAY = 86400.0

#: Minimum same-robot separation between allocated quota anchors (protocol §2).
ANCHOR_SPACING_S = 14.0 * DAY

#: Dedicated allocator RNG domain separation (protocol §3).
_ALLOC_RNG_A = 31
_ALLOC_RNG_B = 7

#: Frozen roster: seed -> (role group, root identity).
#: Candidate 1 (1000–1017) continues past every Sprint 13/14 family;
#: candidate 2 (1100–1117) continues past candidate 1;
#: candidate 3 (1200–1217) continues past candidate 2;
#: candidate 4 (1300–1317) continues past candidate 3;
#: candidate 5 (1400–1417) continues past candidate 4;
#: candidate 6 (1500–1517) continues past candidate 5;
#: candidate 7 (1600–1617) continues past candidate 6.
S15_ROSTER: dict[int, tuple[str, str]] = {
    1000: ("DESIGN", "H-DESIGN-13"),
    1001: ("DESIGN", "H-DESIGN-14"),
    1002: ("DESIGN", "H-DESIGN-15"),
    1003: ("DESIGN", "H-DESIGN-16"),
    1004: ("FIT", "H-FIT-10"),
    1005: ("FIT", "H-FIT-11"),
    1006: ("FIT", "H-FIT-12"),
    1007: ("CALIBRATION", "H-CAL-4"),
    1008: ("CONFIRMATION", "H-CONF-10"),
    1009: ("CONFIRMATION", "H-CONF-11"),
    1010: ("CONFIRMATION", "H-CONF-12"),
    1011: ("CONFIRMATION", "H-CONF-13"),
    1012: ("SEALED", "H-SEAL-13"),
    1013: ("SEALED", "H-SEAL-14"),
    1014: ("SEALED", "H-SEAL-15"),
    1015: ("SEALED", "H-SEAL-16"),
    1016: ("PROOF", "H-PROOF-1"),
    1017: ("PROOF", "H-PROOF-2"),
    1100: ("DESIGN", "H-DESIGN-17"),
    1101: ("DESIGN", "H-DESIGN-18"),
    1102: ("DESIGN", "H-DESIGN-19"),
    1103: ("DESIGN", "H-DESIGN-20"),
    1104: ("FIT", "H-FIT-13"),
    1105: ("FIT", "H-FIT-14"),
    1106: ("FIT", "H-FIT-15"),
    1107: ("CALIBRATION", "H-CAL-5"),
    1108: ("CONFIRMATION", "H-CONF-14"),
    1109: ("CONFIRMATION", "H-CONF-15"),
    1110: ("CONFIRMATION", "H-CONF-16"),
    1111: ("CONFIRMATION", "H-CONF-17"),
    1112: ("SEALED", "H-SEAL-17"),
    1113: ("SEALED", "H-SEAL-18"),
    1114: ("SEALED", "H-SEAL-19"),
    1115: ("SEALED", "H-SEAL-20"),
    1116: ("PROOF", "H-PROOF-3"),
    1117: ("PROOF", "H-PROOF-4"),
    1200: ("DESIGN", "H-DESIGN-21"),
    1201: ("DESIGN", "H-DESIGN-22"),
    1202: ("DESIGN", "H-DESIGN-23"),
    1203: ("DESIGN", "H-DESIGN-24"),
    1204: ("FIT", "H-FIT-16"),
    1205: ("FIT", "H-FIT-17"),
    1206: ("FIT", "H-FIT-18"),
    1207: ("CALIBRATION", "H-CAL-6"),
    1208: ("CONFIRMATION", "H-CONF-18"),
    1209: ("CONFIRMATION", "H-CONF-19"),
    1210: ("CONFIRMATION", "H-CONF-20"),
    1211: ("CONFIRMATION", "H-CONF-21"),
    1212: ("SEALED", "H-SEAL-21"),
    1213: ("SEALED", "H-SEAL-22"),
    1214: ("SEALED", "H-SEAL-23"),
    1215: ("SEALED", "H-SEAL-24"),
    1216: ("PROOF", "H-PROOF-5"),
    1217: ("PROOF", "H-PROOF-6"),
    1300: ("DESIGN", "H-DESIGN-25"),
    1301: ("DESIGN", "H-DESIGN-26"),
    1302: ("DESIGN", "H-DESIGN-27"),
    1303: ("DESIGN", "H-DESIGN-28"),
    1304: ("FIT", "H-FIT-19"),
    1305: ("FIT", "H-FIT-20"),
    1306: ("FIT", "H-FIT-21"),
    1307: ("CALIBRATION", "H-CAL-7"),
    1308: ("CONFIRMATION", "H-CONF-22"),
    1309: ("CONFIRMATION", "H-CONF-23"),
    1310: ("CONFIRMATION", "H-CONF-24"),
    1311: ("CONFIRMATION", "H-CONF-25"),
    1312: ("SEALED", "H-SEAL-25"),
    1313: ("SEALED", "H-SEAL-26"),
    1314: ("SEALED", "H-SEAL-27"),
    1315: ("SEALED", "H-SEAL-28"),
    1316: ("PROOF", "H-PROOF-7"),
    1317: ("PROOF", "H-PROOF-8"),
    1400: ("DESIGN", "H-DESIGN-29"),
    1401: ("DESIGN", "H-DESIGN-30"),
    1402: ("DESIGN", "H-DESIGN-31"),
    1403: ("DESIGN", "H-DESIGN-32"),
    1404: ("FIT", "H-FIT-22"),
    1405: ("FIT", "H-FIT-23"),
    1406: ("FIT", "H-FIT-24"),
    1407: ("CALIBRATION", "H-CAL-8"),
    1408: ("CONFIRMATION", "H-CONF-26"),
    1409: ("CONFIRMATION", "H-CONF-27"),
    1410: ("CONFIRMATION", "H-CONF-28"),
    1411: ("CONFIRMATION", "H-CONF-29"),
    1412: ("SEALED", "H-SEAL-29"),
    1413: ("SEALED", "H-SEAL-30"),
    1414: ("SEALED", "H-SEAL-31"),
    1415: ("SEALED", "H-SEAL-32"),
    1416: ("PROOF", "H-PROOF-9"),
    1417: ("PROOF", "H-PROOF-10"),
    1500: ("DESIGN", "H-DESIGN-33"),
    1501: ("DESIGN", "H-DESIGN-34"),
    1502: ("DESIGN", "H-DESIGN-35"),
    1503: ("DESIGN", "H-DESIGN-36"),
    1504: ("FIT", "H-FIT-25"),
    1505: ("FIT", "H-FIT-26"),
    1506: ("FIT", "H-FIT-27"),
    1507: ("CALIBRATION", "H-CAL-9"),
    1508: ("CONFIRMATION", "H-CONF-30"),
    1509: ("CONFIRMATION", "H-CONF-31"),
    1510: ("CONFIRMATION", "H-CONF-32"),
    1511: ("CONFIRMATION", "H-CONF-33"),
    1512: ("SEALED", "H-SEAL-33"),
    1513: ("SEALED", "H-SEAL-34"),
    1514: ("SEALED", "H-SEAL-35"),
    1515: ("SEALED", "H-SEAL-36"),
    1516: ("PROOF", "H-PROOF-11"),
    1517: ("PROOF", "H-PROOF-12"),
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

#: Role group -> manifest output subdirectory under ``data/generated/sprint15-v1/``.
S15_ROLE_DIRS = {
    "DESIGN": "DESIGN",
    "FIT": "FIT",
    "CALIBRATION": "CALIBRATION",
    "CONFIRMATION": "CONFIRMATION",
    "SEALED": "SEALED",
    "PROOF": "PROOF",
}

#: Reserved identities outside Fit/Calibration (protocol §6; holdout labels,
#: not roots — same reserves as Sprint 14).
ROBOT_RESERVE = "robot-08"
PROGRAM_RESERVE = "program-03"
P_SUBTYPE_RESERVE = "P2"
W_SUBTYPE_RESERVE = "W2"

#: Model-visible file-row key allowlist (identical to the Sprint 14 audit
#: contract): hidden event state must never enter these rows.
ALLOWED_ROW_KEYS = frozenset({
    "file_id", "operation_id", "robot_id", "program_id", "start_time",
    "end_time", "file_label", "is_quarantined", "quarantine_reason",
    "is_censored", "member_views", "last_reset_time", "n_valid_patches",
})

#: Bounded nuisance envelopes (protocol §4): axis -> (nominal, lower, upper).
NUISANCE_ENVELOPES = {
    "sensor_noise_scale": (1.0e-3, 0.95e-3, 1.05e-3),
    "timing_jitter_fraction": (0.0, 0.0, 0.02),
    "usage_variation_fraction": (0.0, 0.0, 0.10),
    "duration_variation_fraction": (0.0, 0.0, 0.10),
    "coupling_drift_absolute": (0.0, 0.0, 0.05),
}

#: Per-subtype manifestation gains (frozen; protocol §5).
SUBTYPE_GAINS = {"P1": 1.0, "P2": 0.55, "W1": 0.3, "W2": 0.16}

#: Minimum program sensitivity (manifestation floor input).
SENSITIVITY_MIN = 0.5

#: Minimum failure-file severity scale (s=1 -> 0.9*1/2 = 0.45; conservative
#: floor for precursor-visible signatures).
SEVERITY_MIN = 0.45

#: Conservative analytic nuisance ceiling in manifested-value units
#: (noise 0.00105 + drift 0.004 + jitter coupling 0.003, worst case).
NUISANCE_CEILING = 0.008

#: Required analytic signal-to-nuisance margin (protocol §5).
SIGNAL_MARGIN_MIN = 3.0


@dataclass(frozen=True)
class QuotaConfig:
    """Exact per-history balanced quotas and allocation constraints."""

    p: int = 24
    p1: int = 12
    p2: int = 12
    w: int = 24
    w1: int = 12
    w2: int = 12
    a: int = 16
    a1: int = 8
    a2: int = 8
    controls: int = 48
    robot_days: int = 240
    robots_pos: int = 6
    robots_neg: int = 6
    programs: int = 2
    spacing_s: float = ANCHOR_SPACING_S
    robot_pos_cap: float = 0.35
    robot_neg_cap: float = 0.40
    program_cap: float = 0.60

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable quota record."""
        return asdict(self)


#: Frozen candidate-1 quota configuration.
DEFAULT_QUOTA = QuotaConfig()


class InfeasibleCandidate(ValueError):
    """The frozen calendar cannot place quotas under frozen constraints."""

    def __init__(self, record: dict[str, object]) -> None:
        super().__init__(
            "infeasible candidate: "
            + str(record.get("missing", record.get("reason")))
        )
        self.record = record


def alloc_seed_for(history_seed: int) -> int:
    """Return the dedicated allocator RNG seed (protocol §3)."""
    return int(history_seed) * _ALLOC_RNG_A + _ALLOC_RNG_B


#: Canonical (cohort, subtype) bucket visitation order for joint fair
#: allocation (protocol v3 §3a). Fixed for all histories.
BUCKET_ORDER: tuple[tuple[str, str], ...] = (
    ("P", "P1"), ("P", "P2"),
    ("W", "W1"), ("W", "W2"),
    ("A", "A1"), ("A", "A2"),
)


def bucket_sub_seed(alloc_seed: int, cohort: str, subtype: str) -> int:
    """Derive one bucket's independent shuffle seed (protocol v3 §3a).

    Deterministic in ``(alloc_seed, cohort, subtype)`` only: a bucket's
    shuffled priority list never depends on bucket visitation order,
    acceptance outcomes, or any other bucket's pool. Draw count depends
    only on that bucket's own pool size.
    """
    digest = hashlib.sha256(
        f"{int(alloc_seed)}|{cohort}|{subtype}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def calendar_digest(schedule_events: list[dict[str, object]]) -> str:
    """Return the SHA256 of the canonical frozen-calendar serialization."""
    canonical = json.dumps(schedule_events, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def project_allocation_inputs(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Project waveform-blind allocation inputs, enforcing the row contract.

    Every row must carry exactly the allowlisted model-visible keys: hidden
    cohort/subtype/failure-time state in a row raises ``ValueError``.
    """
    projected = []
    for row in rows:
        keys = set(row)
        if keys - ALLOWED_ROW_KEYS:
            raise ValueError(
                f"leakage keys in model-visible row {row.get('file_id')}: "
                f"{sorted(keys - ALLOWED_ROW_KEYS)}"
            )
        if keys < ALLOWED_ROW_KEYS:
            raise ValueError(
                f"missing row keys in {row.get('file_id')}: "
                f"{sorted(ALLOWED_ROW_KEYS - keys)}"
            )
        projected.append({key: row[key] for key in sorted(ALLOWED_ROW_KEYS)})
    return projected


def _lead_detail(
    rows: list[dict],
    failure: dict,
    wins: dict[str, list[list[float]]],
    ledger: list[dict],
) -> dict[str, object]:
    """Return lead-support components for one failure (frozen predicates)."""
    cands = E.pos_files(rows, failure, wins)
    ends = sorted(c["end_time"] for c in cands)
    t_end = failure["failure_time"]
    robot = failure["robot_id"]
    base_cands = [
        x for x in rows
        if x["robot_id"] == robot
        and t_end - 42 * DAY <= x["end_time"] < t_end - 7 * DAY
        and not x["is_quarantined"]
        and not E._overlaps_maintenance(x, wins)
    ]
    degs = [
        (g["degradation_onset"], g["failure_time"])
        for g in ledger
        if g["robot_id"] == robot
        and g["degradation_onset"] is not None
    ]
    clean = any(
        not any(o <= x["end_time"] < t for o, t in degs)
        for x in base_cands
    )
    return {
        "endpoints": len(ends),
        "early_endpoint": any(e <= t_end - DAY for e in ends),
        "clean_baseline": clean,
    }


def eligible_anchors(
    rows: list[dict],
    ledger: list[dict],
    wins: dict[str, list[list[float]]],
) -> tuple[list[dict], Counter, dict[str, dict[str, object]]]:
    """Split the realized ledger into eligible anchors and rejections.

    P/W anchors are eligible iff their ``[T-7d, T]`` horizon is reset-free,
    has POS-ELIGIBLE files, and meets the frozen horizon-endpoint predicate
    (>=3 eligible file endpoints in-window, >=1 ending >=24 h before ``T``;
    protocol v1 §2). A anchors are eligible iff reset-free with >=1
    POS-ELIGIBLE file (= V2 A-evaluable): abrupt failures carry no precursor
    episode by definition, so clean pre-cutoff A horizons hold only the
    failure file in the temporal view, while the V2 gates impose
    lead-support exclusively on P/W (80% rule) and require of A only
    structural representation with separate reporting and no predictability
    minimum. Clean-baseline status is recorded per anchor and gated at 80%
    by the audit for P/W. Every ineligible failure gets exactly one reason.
    """
    eligible: list[dict] = []
    rejection: Counter = Counter()
    lead: dict[str, dict[str, object]] = {}
    for failure in ledger:
        fid = failure["failure_id"]
        if E.positive_window_intersects_reset(failure, wins):
            rejection["reset-horizon"] += 1
            continue
        detail = _lead_detail(rows, failure, wins, ledger)
        lead[fid] = detail
        if not E.pos_files(rows, failure, wins):
            rejection["no-endpoints"] += 1
            continue
        if failure["cohort"] in ("P", "W") and not (
            detail["endpoints"] >= 3  # type: ignore[operator]
            and detail["early_endpoint"]
        ):
            rejection["lead-shortfall"] += 1
            continue
        eligible.append(failure)
    return eligible, rejection, lead


class AllocationUnavailable(ValueError):
    """The exact allocator could not verify feasibility (protocol v4 §3b).

    Raised when the MILP solver completes without proving optimality or
    infeasibility (numerical error, iteration abort, unreturned witness).
    The candidate is UNAVAILABLE: it is never promoted on an unverified
    status and never retried within the candidate.
    """

    def __init__(self, record: dict[str, object]) -> None:
        super().__init__(
            "unavailable candidate: "
            + str(record.get("missing", record.get("reason")))
        )
        self.record = record


def map_solver_status(status: int, success: bool) -> str:
    """Map a MILP completion code to the frozen verdict class (protocol v4 §3b).

    Proven optimal (``status == 0`` with ``success``) proceeds to witness
    validation; proven infeasible (``status == 2``, HiGHS model-status
    Infeasible) raises :class:`InfeasibleCandidate`; every other completion
    is UNAVAILABLE — never promoted on an unverified status.
    """
    if int(status) == 0 and bool(success):
        return "optimal"
    if int(status) == 2:
        return "infeasible"
    return "unavailable"


def build_allocation_program(
    eligible: list[dict],
    inputs: list[dict],
    quota: QuotaConfig,
) -> dict[str, object]:
    """Build the exact binary anchor-selection program (protocol v4 §3b).

    Variables index the eligible anchors in canonical
    ``(failure_time, failure_id)`` order; duplicate failure ids are rejected.
    Constraint groups: six quota equalities (one per P1/P2/W1/W2/A1/A2
    bucket); one ``x_i + x_j <= 1`` row per same-robot anchor pair with
    ``|t_i - t_j| < spacing_s``; one per-robot concentration row
    (``<= ROBOT_POS_CAP_MAX``); one per-program row inside each predictable
    cohort (``<= PROGRAM_CAP_MAX``). Zero objective: the first proven-optimal
    witness under canonical indexing is the allocation — no selection order
    or preference enters. Waveform-blind: metadata inputs only.
    """
    order = sorted(
        eligible, key=lambda f: (f["failure_time"], f["failure_id"]))
    index = {f["failure_id"]: i for i, f in enumerate(order)}
    if len(index) != len(order):
        raise ValueError("duplicate failure_id in eligible anchors")
    needs = {
        ("P", "P1"): quota.p1, ("P", "P2"): quota.p2,
        ("W", "W1"): quota.w1, ("W", "W2"): quota.w2,
        ("A", "A1"): quota.a1, ("A", "A2"): quota.a2,
    }
    quota_rows = []
    for cohort, subtype in BUCKET_ORDER:
        members = [index[f["failure_id"]] for f in order
                   if f["cohort"] == cohort and f["subtype"] == subtype]
        quota_rows.append({
            "bucket": f"{cohort}/{subtype}",
            "need": needs[(cohort, subtype)],
            "members": members,
        })
    by_robot: dict[str, list[int]] = {}
    for i, failure in enumerate(order):
        by_robot.setdefault(failure["robot_id"], []).append(i)
    spacing_edges = []
    for members in by_robot.values():
        for a_pos in range(len(members)):
            for b_pos in range(a_pos + 1, len(members)):
                i, j = members[a_pos], members[b_pos]
                if abs(order[i]["failure_time"]
                       - order[j]["failure_time"]) < quota.spacing_s:
                    spacing_edges.append((i, j))
    robot_rows = {robot: sorted(members)
                  for robot, members in sorted(by_robot.items())}
    prog_by_op = {(r["robot_id"], r["end_time"]): r["program_id"]
                  for r in inputs}
    program_rows: dict[str, list[int]] = {}
    for i, failure in enumerate(order):
        if failure["cohort"] not in ("P", "W"):
            continue
        prog = prog_by_op.get(
            (failure["robot_id"], failure["failure_time"]))
        if prog is None:
            continue
        program_rows.setdefault(
            f"{failure['cohort']}/{prog}", []).append(i)
    program_rows = {key: sorted(members)
                    for key, members in sorted(program_rows.items())}
    return {
        "order": [f["failure_id"] for f in order],
        "quota_rows": quota_rows,
        "spacing_edges": spacing_edges,
        "robot_rows": robot_rows,
        "program_rows": program_rows,
        "robot_cap": ROBOT_POS_CAP_MAX,
        "program_cap": PROGRAM_CAP_MAX,
        "n_variables": len(order),
        "n_constraints": (len(quota_rows) + len(spacing_edges)
                          + len(robot_rows) + len(program_rows)),
    }


def solve_allocation_program(
    program: dict[str, object],
) -> tuple[str, list[str] | None, dict[str, object]]:
    """Solve the frozen binary program; return verdict, witness, provenance.

    ``verdict`` is ``"optimal"`` with the canonical witness ids,
    ``"infeasible"`` with ``None``, or ``"unavailable"`` with ``None``.
    ``provenance`` records the locked solver identity and version, raw
    completion code, message, solve time, and problem dimensions. The
    solver import is deferred so the module import contract is unchanged.
    """
    import time

    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp

    order = list(program["order"])
    n = int(program["n_variables"])
    quota_rows = list(program["quota_rows"])
    edges = list(program["spacing_edges"])
    robot_rows = dict(program["robot_rows"])
    program_rows = dict(program["program_rows"])
    robot_cap = int(program["robot_cap"])
    row_cap = int(program["program_cap"])
    robots = sorted(robot_rows)
    progs = sorted(program_rows)
    n_le = len(edges) + len(robots) + len(progs)
    a_eq = np.zeros((len(quota_rows), n))
    b_eq = np.zeros(len(quota_rows))
    for r, row in enumerate(quota_rows):
        for i in row["members"]:
            a_eq[r, int(i)] = 1.0
        b_eq[r] = float(row["need"])
    a_le = np.zeros((n_le, n))
    ub = np.zeros(n_le)
    r = 0
    for i, j in edges:
        a_le[r, int(i)] = 1.0
        a_le[r, int(j)] = 1.0
        ub[r] = 1.0
        r += 1
    for robot in robots:
        for i in robot_rows[robot]:
            a_le[r, int(i)] = 1.0
        ub[r] = float(robot_cap)
        r += 1
    for key in progs:
        for i in program_rows[key]:
            a_le[r, int(i)] = 1.0
        ub[r] = float(row_cap)
        r += 1
    t0 = time.perf_counter()
    res = milp(
        np.zeros(n),
        constraints=[
            LinearConstraint(a_eq, b_eq, b_eq),
            LinearConstraint(a_le, np.full(n_le, -np.inf), ub),
        ],
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
    )
    solve_s = time.perf_counter() - t0
    verdict = map_solver_status(res.status, res.success)
    witness: list[str] | None = None
    if verdict == "optimal" and res.x is not None:
        witness = sorted(
            order[i] for i in range(n) if float(res.x[i]) > 0.5)
    provenance = {
        "solver": "scipy.optimize.milp (HiGHS)",
        "scipy_version": str(scipy.__version__),
        "solver_status": int(res.status),
        "solver_message": str(res.message),
        "solve_s": round(float(solve_s), 2),
        "n_variables": n,
        "n_constraints": len(quota_rows) + n_le,
    }
    return verdict, witness, provenance


def validate_allocation_witness(
    witness: list[str],
    program: dict[str, object],
    eligible_by_id: dict[str, dict],
    spacing_s: float,
) -> list[str]:
    """Re-verify a solver witness using only the frozen predicates.

    Returns breach descriptions (empty = valid): every id unique and a
    member of the eligible set; exact per-subtype counts; every selected
    same-robot pair satisfies ``|dt| >= spacing_s``; per-robot counts
    ``<= ROBOT_POS_CAP_MAX``; per-program P/W counts ``<= PROGRAM_CAP_MAX``.
    """
    breaches = []
    if len(set(witness)) != len(witness):
        breaches.append("duplicate-ids")
    unknown = sorted(fid for fid in witness if fid not in eligible_by_id)
    if unknown:
        breaches.append(f"unknown-ids: {unknown}")
    chosen = set(witness)
    order = list(program["order"])
    for row in program["quota_rows"]:
        got = sum(1 for i in row["members"] if order[int(i)] in chosen)
        if got != row["need"]:
            breaches.append(
                f"{row['bucket']}: {got} != {row['need']}")
    for i, j in program["spacing_edges"]:
        if order[int(i)] in chosen and order[int(j)] in chosen:
            a, b = eligible_by_id[order[int(i)]], eligible_by_id[order[int(j)]]
            if abs(a["failure_time"] - b["failure_time"]) < spacing_s:
                breaches.append(
                    f"spacing: {order[int(i)]}/{order[int(j)]}")
    for robot, members in program["robot_rows"].items():
        got = sum(1 for i in members if order[int(i)] in chosen)
        if got > int(program["robot_cap"]):
            breaches.append(f"robot-cap: {robot}: {got}")
    for key, members in program["program_rows"].items():
        got = sum(1 for i in members if order[int(i)] in chosen)
        if got > int(program["program_cap"]):
            breaches.append(f"program-cap: {key}: {got}")
    return breaches


def _finalize_allocation(
    *,
    inputs: list[dict],
    wins: dict[str, list[list[float]]],
    ledger: list[dict],
    selected: dict[str, dict[str, list[str]]],
    per_robot: Counter,
    per_program: dict[str, Counter],
    rejection: Counter,
    lead: dict[str, dict[str, object]],
    history_seed: int,
    seed: int,
    quota: QuotaConfig,
    solver: dict[str, object] | None = None,
) -> dict[str, object]:
    """Shared post-selection tail: controls, margins, checks, return record.

    Both allocation methods finalize here so the observable contract
    (controls, support margins, record shape) is method-independent. The
    solver provenance joins the record only on the exact path, keeping the
    joint-path return shape frozen.
    """
    anchors = E.anchor_rows(inputs, wins)
    controls = E.select_control_windows(anchors, ledger, wins)
    if len(controls) < quota.controls:
        raise InfeasibleCandidate({
            "reason": "control-shortfall",
            "missing": f"controls: {len(controls)} < {quota.controls}",
            "history_seed": history_seed,
            "alloc_seed": seed,
            "rejection": dict(rejection),
        })

    eval_days = {
        (r["robot_id"], int(r["end_time"] // DAY)) for r in inputs
        if E.eligible_operational_row(r, wins)
    }
    ctrl_by_robot = Counter(w["robot_id"] for w in controls)
    margins = {
        "controls": len(controls),
        "robot_days": len(eval_days),
        "robots_pos": sum(1 for v in per_robot.values() if v > 0),
        "robots_neg": sum(1 for v in ctrl_by_robot.values() if v > 0),
        "programs_P": len(per_program["P"]),
        "programs_W": len(per_program["W"]),
    }
    checks = {
        "controls_ge": margins["controls"] >= quota.controls,
        "robot_days_ge": margins["robot_days"] >= quota.robot_days,
        "robots_pos_ge": margins["robots_pos"] >= quota.robots_pos,
        "robots_neg_ge": margins["robots_neg"] >= quota.robots_neg,
        "programs_ge": (
            margins["programs_P"] >= quota.programs
            and margins["programs_W"] >= quota.programs
        ),
    }
    if not all(checks.values()):
        failed = sorted(k for k, v in checks.items() if not v)
        raise InfeasibleCandidate({
            "reason": "support-margin-shortfall",
            "missing": f"margins: {failed}",
            "history_seed": history_seed,
            "alloc_seed": seed,
            "margins": margins,
            "rejection": dict(rejection),
        })
    selected_ids = _selected_ids(selected)
    record = {
        "history_seed": history_seed,
        "alloc_seed": seed,
        "quota": quota.as_dict(),
        "selected": {
            cohort: {st: list(ids) for st, ids in subs.items()}
            for cohort, subs in selected.items()
        },
        "selected_ids": sorted(selected_ids),
        "lead": {fid: lead[fid] for fid in selected_ids},
        "controls": [
            {
                "anchor_end": w["anchor_end"],
                "robot_id": w["robot_id"],
                "members": [m["file_id"] for m in w["members"]],
            }
            for w in controls
        ],
        "margins": margins,
        "rejection": dict(rejection),
    }
    if solver is not None:
        record["solver"] = solver
    return record


def _allocate_exact(
    inputs: list[dict],
    eligible: list[dict],
    rejection: Counter,
    lead: dict[str, dict[str, object]],
    wins: dict[str, list[list[float]]],
    ledger: list[dict],
    history_seed: int,
    seed: int,
    quota: QuotaConfig,
) -> dict[str, object]:
    """Run the exact deterministic CSP path (protocol v4 §3b).

    Builds the frozen binary program over canonical anchor indexing, solves
    it to proven optimality with the locked MILP solver, re-verifies the
    witness with the independent checker, and finalizes through the shared
    controls/margins tail. Proven infeasibility raises
    :class:`InfeasibleCandidate`; any other non-optimal completion raises
    :class:`AllocationUnavailable`; a checker breach is a hard error (no
    root is written). The ``lead`` mapping is forwarded to the
    shared finalizer.
    """
    by_cohort: dict[str, list[dict]] = {}
    for failure in eligible:
        by_cohort.setdefault(failure["cohort"], []).append(failure)
    program = build_allocation_program(eligible, inputs, quota)
    if int(program["n_variables"]) == 0:
        raise InfeasibleCandidate({
            "reason": "quota-shortfall",
            "missing": "exact CSP: empty eligible pool",
            "history_seed": history_seed,
            "alloc_seed": seed,
            "eligible_pool": {},
            "rejection": dict(rejection),
        })
    verdict, witness, provenance = solve_allocation_program(program)
    if verdict == "infeasible":
        raise InfeasibleCandidate({
            "reason": "quota-shortfall",
            "missing": ("exact CSP proven infeasible; quotas unmet: "
                        + ", ".join(
                            f"{row['bucket']}<{row['need']}"
                            for row in program["quota_rows"])),
            "history_seed": history_seed,
            "alloc_seed": seed,
            "eligible_pool": {c: len(v) for c, v in by_cohort.items()},
            "rejection": dict(rejection),
            "solver": provenance,
        })
    if verdict != "optimal" or witness is None:
        raise AllocationUnavailable({
            "reason": "solver-unavailable",
            "missing": (f"exact CSP unverified: "
                        f"{provenance['solver_status']}: "
                        f"{provenance['solver_message']}"),
            "history_seed": history_seed,
            "alloc_seed": seed,
            "eligible_pool": {c: len(v) for c, v in by_cohort.items()},
            "rejection": dict(rejection),
            "solver": provenance,
        })
    eligible_by_id = {f["failure_id"]: f for f in eligible}
    breaches = validate_allocation_witness(
        witness, program, eligible_by_id, quota.spacing_s)
    if breaches:
        raise RuntimeError(f"exact-CSP witness breach: {breaches}")
    order = list(program["order"])
    chosen = set(witness)
    selected: dict[str, dict[str, list[str]]] = {}
    per_robot: Counter = Counter()
    per_program: dict[str, Counter] = {"P": Counter(), "W": Counter()}
    for row in program["quota_rows"]:
        cohort, _, subtype = row["bucket"].partition("/")
        got = [order[int(i)] for i in sorted(row["members"])
               if order[int(i)] in chosen]
        selected.setdefault(cohort, {})[subtype] = got
        for fid in got:
            per_robot[eligible_by_id[fid]["robot_id"]] += 1
    for key, members in program["program_rows"].items():
        cohort, _, prog = key.partition("/")
        for i in members:
            if order[int(i)] in chosen:
                per_program[cohort][prog] += 1
    return _finalize_allocation(
        inputs=inputs, wins=wins, ledger=ledger, selected=selected,
        per_robot=per_robot, per_program=per_program,
        rejection=rejection, lead=lead, history_seed=history_seed,
        seed=seed, quota=quota, solver=provenance,
    )


def allocate_quotas(
    rows: list[dict],
    ledger: list[dict],
    wins: dict[str, list[list[float]]],
    history_seed: int,
    quota: QuotaConfig = DEFAULT_QUOTA,
    method: str = "joint",
) -> dict[str, object]:
    """Allocate exact quotas from eligible anchors or raise infeasible.

    ``method="joint"`` (frozen; candidates 1-3): joint fair rounds over the
    canonical buckets ``[P1, P2, W1, W2, A1, A2]`` (protocol v3 §3a); each
    bucket shuffles its eligible pool once with an independent sub-RNG, then
    rounds grant every incomplete bucket exactly one acceptance opportunity
    with global 14-day same-robot spacing and robot/program caps. Buckets
    that exhaust their pool stall terminally.
    ``method="exact"`` (protocol v4 §3b): the frozen binary program over
    canonical anchor indexing is solved to proven optimality with the locked
    MILP solver under exact floor concentration caps, and the witness is
    re-verified by the independent checker. Both methods finalize through
    the shared controls/margins tail. Deterministic per seed;
    waveform-blind (metadata inputs only).
    """
    inputs = project_allocation_inputs(rows)
    eligible, rejection, lead = eligible_anchors(inputs, ledger, wins)
    seed = alloc_seed_for(history_seed)
    if method == "exact":
        return _allocate_exact(
            inputs, eligible, rejection, lead, wins, ledger,
            history_seed, seed, quota)
    if method != "joint":
        raise ValueError(f"unknown allocation method {method!r}")
    by_cohort: dict[str, list[dict]] = {}
    for failure in eligible:
        by_cohort.setdefault(failure["cohort"], []).append(failure)

    needs = {
        ("P", "P1"): quota.p1, ("P", "P2"): quota.p2,
        ("W", "W1"): quota.w1, ("W", "W2"): quota.w2,
        ("A", "A1"): quota.a1, ("A", "A2"): quota.a2,
    }
    buckets: list[dict] = []
    for cohort, subtype in BUCKET_ORDER:
        pool = sorted(
            (f for f in by_cohort.get(cohort, [])
             if f["subtype"] == subtype),
            key=lambda f: (f["failure_time"], f["failure_id"]),
        )
        sub_rng = np.random.default_rng(
            bucket_sub_seed(seed, cohort, subtype))
        sub_rng.shuffle(pool)
        buckets.append({
            "cohort": cohort, "subtype": subtype,
            "need": needs[(cohort, subtype)],
            "pool": pool, "cursor": 0, "got": [],
            "stalled": False,
        })
    selected: dict[str, dict[str, list[str]]] = {}
    chosen: set[str] = set()
    per_robot: Counter = Counter()
    kept_moments: dict[str, list[float]] = {}
    per_program: dict[str, Counter] = {"P": Counter(), "W": Counter()}
    prog_by_op = {
        (r["robot_id"], r["end_time"]): r["program_id"] for r in inputs
    }
    total_quota = quota.p + quota.w + quota.a
    robot_cap = max(1, int(math.ceil(quota.robot_pos_cap * total_quota)))

    def _accept(bucket: dict, failure: dict) -> bool:
        """Accept one anchor under global caps/spacing, or tally and skip."""
        cohort = bucket["cohort"]
        if failure["failure_id"] in chosen:
            return True
        robot = failure["robot_id"]
        if per_robot[robot] >= robot_cap:
            rejection["cap-skip"] += 1
            return False
        moment = failure["failure_time"]
        if any(abs(moment - kept) < quota.spacing_s
               for kept in kept_moments.get(robot, [])):
            rejection["spacing-skip"] += 1
            return False
        if cohort in ("P", "W"):
            prog = prog_by_op.get((robot, moment))
            if prog is not None and (
                per_program[cohort][prog]
                >= math.ceil(quota.program_cap * (quota.p if cohort == "P" else quota.w))
            ):
                rejection["program-cap-skip"] += 1
                return False
        bucket["got"].append(failure["failure_id"])
        chosen.add(failure["failure_id"])
        per_robot[robot] += 1
        kept_moments.setdefault(robot, []).append(moment)
        if cohort in ("P", "W"):
            prog = prog_by_op.get((robot, moment))
            if prog is not None:
                per_program[cohort][prog] += 1
        return True

    while True:
        if all(len(b["got"]) >= b["need"] for b in buckets):
            break
        accepted_this_round = False
        for b in buckets:
            if len(b["got"]) >= b["need"] or b["stalled"]:
                continue
            while b["cursor"] < len(b["pool"]):
                failure = b["pool"][b["cursor"]]
                b["cursor"] += 1
                if _accept(b, failure):
                    accepted_this_round = True
                    break
            else:
                b["stalled"] = True
        if not accepted_this_round:
            missing = next(
                b for b in buckets if len(b["got"]) < b["need"])
            raise InfeasibleCandidate({
                "reason": "quota-shortfall",
                "missing": (f"{missing['cohort']}/{missing['subtype']}: "
                            f"{len(missing['got'])} < {missing['need']}"),
                "history_seed": history_seed,
                "alloc_seed": seed,
                "eligible_pool": {c: len(v) for c, v in by_cohort.items()},
                "rejection": dict(rejection),
                "stalled_buckets": sorted(
                    f"{x['cohort']}/{x['subtype']}"
                    for x in buckets if x["stalled"]),
            })
    for b in buckets:
        selected.setdefault(b["cohort"], {})[b["subtype"]] = b["got"]

    return _finalize_allocation(
        inputs=inputs, wins=wins, ledger=ledger, selected=selected,
        per_robot=per_robot, per_program=per_program,
        rejection=rejection, lead=lead, history_seed=history_seed,
        seed=seed, quota=quota,
    )


def _selected_ids(
    selected: dict[str, dict[str, list[str]]],
) -> set[str]:
    """Return every selected failure id across cohorts and subtypes."""
    return {
        fid
        for subs in selected.values()
        for ids in subs.values()
        for fid in ids
    }


def check_nuisance_envelopes(resolved_config: dict[str, object]) -> dict[str, object]:
    """Verify realized DGP knobs lie inside the frozen nuisance envelopes."""
    health = resolved_config["health"]
    scheduler = resolved_config["scheduler"]
    noise = float(health["noise_scale"])  # type: ignore[index]
    _, lo, hi = NUISANCE_ENVELOPES["sensor_noise_scale"]
    durations = [
        float(stage["duration_s"])  # type: ignore[index]
        for route in scheduler["routes"]  # type: ignore[index]
        for stage in route["stages"]
    ]
    nominal_duration = 600.0
    _, _, dur_hi = NUISANCE_ENVELOPES["duration_variation_fraction"]
    result = {
        "sensor_noise_scale": noise,
        "noise_inside": bool(lo <= noise <= hi),
        "op_durations": durations,
        "durations_inside": bool(
            all(abs(d - nominal_duration) / nominal_duration <= dur_hi
                for d in durations)
        ),
        "timing_jitter_fraction": 0.0,
        "usage_variation_fraction": 0.0,
        "coupling_drift_absolute": 0.0,
    }
    result["pass"] = bool(
        result["noise_inside"] and result["durations_inside"]
    )
    return result


def analytic_signal_margin(
    gains: dict[str, float] = SUBTYPE_GAINS,
) -> dict[str, object]:
    """Recompute the conservative analytic signal-to-nuisance margin."""
    weakest = min(gains.items(), key=lambda kv: kv[1])
    floor = SENSITIVITY_MIN * weakest[1] * SEVERITY_MIN
    margin = floor / NUISANCE_CEILING
    return {
        "weakest_subtype": weakest[0],
        "weakest_gain": weakest[1],
        "signature_floor": floor,
        "nuisance_ceiling": NUISANCE_CEILING,
        "margin": margin,
        "required": SIGNAL_MARGIN_MIN,
        "pass": bool(margin >= SIGNAL_MARGIN_MIN),
    }


def _arm_from_scores(
    rows: list[dict],
    ledger: list[dict],
    wins: dict[str, list[list[float]]],
    scores: dict[str, float],
    threshold: float = 0.3,
) -> dict[str, object]:
    """Compute one frozen E1–E5 composite arm (Sprint 14 Task 9 semantics)."""
    pos: list[float] = []
    uneval = 0
    for failure in ledger:
        if E.positive_window_intersects_reset(failure, wins):
            uneval += 1
            continue
        cands = E.pos_files(rows, failure, wins)
        if not cands:
            uneval += 1
            continue
        pos.append(E.window_score([c["file_id"] for c in cands], scores))
    anchors = E.anchor_rows(rows, wins)
    controls = E.select_control_windows(anchors, ledger, wins)
    neg = [
        E.window_score([m["file_id"] for m in w["members"]], scores)
        for w in controls
    ]
    try:
        auc = E.roc_auc_tie_aware(pos, neg)
        auc_state = "defined"
    except E.UnavailableError:
        auc, auc_state = None, "UNAVAILABLE"
    recalled, leads, persists = 0, [], []
    flagged_by_robot: dict[str, list[float]] = {}
    for row in rows:
        if not E.eligible_operational_row(row, wins):
            continue
        if scores[row["file_id"]] >= threshold:
            flagged_by_robot.setdefault(row["robot_id"], []).append(
                row["end_time"]
            )
    for failure in ledger:
        if E.positive_window_intersects_reset(failure, wins):
            continue
        cands = E.pos_files(rows, failure, wins)
        flagged = sorted(
            c["end_time"] for c in cands if scores[c["file_id"]] >= threshold
        )
        if flagged:
            recalled += 1
            leads.append(E.lead_days(failure["failure_time"], flagged))
            persists.append(len(flagged))
    eval_days = {
        (r["robot_id"], int(r["end_time"] // DAY)) for r in rows
        if E.eligible_operational_row(r, wins)
    }
    false, far = E.false_alert_episodes(
        flagged_by_robot, ledger, float(len(eval_days)), wins
    )
    arm = {
        "event_auc": auc,
        "auc_state": auc_state,
        "positives_evaluable": len(pos),
        "positives_unevaluable": uneval,
        "negatives": len(neg),
        "recall": (recalled / len(pos) if pos else None),
        "recall_cp": (list(E.clopper_pearson(recalled, len(pos)))
                      if pos else None),
        "lead": E.summarize_values(leads),
        "lead_positive_fraction": (
            sum(1 for x in leads if x > 0.0) / len(leads) if leads else None),
        "persistence": E.summarize_values(persists),
        "false_episodes": false,
        "far": far,
        "far_bound": (E.rule_of_three_bound(float(len(eval_days)))
                      if false == 0 and eval_days else None),
    }
    arm["computable"] = E.check_arm_computable(arm)
    return arm


def eg3_fixtures(
    rows: list[dict],
    ledger: list[dict],
    wins: dict[str, list[list[float]]],
) -> dict[str, object]:
    """Run the frozen EG3 metric-contract fixture battery (no scores tuned).

    Requires exact ``0.5`` / ``1.0`` / ``0.0`` AUROCs, exact ``0.0``-day
    onset lead, a positive-lead advance fixture with reportable P/W recall,
    a zero-score false-alert fixture reproducing ``(0, 0.0)``, finite
    schema-valid deterministic outputs, and byte-identical repeat output.
    """
    file_ids = [r["file_id"] for r in rows]
    constant = {fid: 0.5 for fid in file_ids}
    horizon_files: set[str] = set()
    for failure in ledger:
        start = failure["failure_time"] - E.HORIZON_S
        stop = failure["failure_time"]
        for row in rows:
            if (
                row["robot_id"] == failure["robot_id"]
                and start <= row["end_time"] <= stop
            ):
                horizon_files.add(row["file_id"])
    correct = {fid: (1.0 if fid in horizon_files else 0.0) for fid in file_ids}
    reversed_scores = {fid: 1.0 - v for fid, v in correct.items()}
    onset_files: set[str] = set()
    for failure in ledger:
        for row in rows:
            if (
                row["robot_id"] == failure["robot_id"]
                and row["end_time"] == failure["failure_time"]
            ):
                onset_files.add(row["file_id"])
    onset = {fid: (1.0 if fid in onset_files else 0.0) for fid in file_ids}
    advance_files: set[str] = set()
    for failure in ledger:
        start = failure["failure_time"] - E.HORIZON_S
        early = failure["failure_time"] - DAY
        for row in rows:
            if (
                row["robot_id"] == failure["robot_id"]
                and start <= row["end_time"] <= early
            ):
                advance_files.add(row["file_id"])
    advance = {fid: (1.0 if fid in advance_files else 0.0) for fid in file_ids}
    zeros = {fid: 0.0 for fid in file_ids}
    time_scores = {
        r["file_id"]: max(0.0, (r["end_time"] - r["last_reset_time"]) / DAY)
        / (max(0.0, (r["end_time"] - r["last_reset_time"]) / DAY) + 30.0)
        for r in rows
    }
    arms = {
        name: _arm_from_scores(rows, ledger, wins, scores)
        for name, scores in (
            ("constant", constant),
            ("correct", correct),
            ("reversed", reversed_scores),
            ("onset", onset),
            ("advance", advance),
            ("zeros", zeros),
            ("observable_time", time_scores),
        )
    }
    advance_p = _recalled_by_cohort(rows, ledger, wins, advance, "P")
    advance_w = _recalled_by_cohort(rows, ledger, wins, advance, "W")
    checks = {
        "constant_is_half": arms["constant"]["event_auc"] == 0.5,
        "correct_is_one": arms["correct"]["event_auc"] == 1.0,
        "reversed_is_zero": arms["reversed"]["event_auc"] == 0.0,
        "onset_lead_is_zero": _onset_max_lead(rows, ledger, wins, onset) == 0.0,
        "advance_lead_positive": bool(
            (arms["advance"]["lead"].get("median") or 0.0) > 0.0
        ),
        "advance_recall_sufficient": advance_p >= 10 and advance_w >= 10,
        "zeros_far_is_zero": (
            arms["zeros"]["false_episodes"] == 0
            and arms["zeros"]["far"] == 0.0
        ),
        "all_computable": all(a["computable"] for a in arms.values()),
        "repeat_identical": (
            json.dumps(arms, sort_keys=True, default=str)
            == json.dumps(
                {
                    name: _arm_from_scores(rows, ledger, wins, scores)
                    for name, scores in (
                        ("constant", constant),
                        ("correct", correct),
                        ("reversed", reversed_scores),
                        ("onset", onset),
                        ("advance", advance),
                        ("zeros", zeros),
                        ("observable_time", time_scores),
                    )
                },
                sort_keys=True,
                default=str,
            )
        ),
    }
    return {
        "arms": arms,
        "advance_recalled": {"P": advance_p, "W": advance_w},
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _recalled_by_cohort(
    rows: list[dict],
    ledger: list[dict],
    wins: dict[str, list[list[float]]],
    scores: dict[str, float],
    cohort: str,
    threshold: float = 0.3,
) -> int:
    """Count recalled failures of one cohort under fixed scores."""
    recalled = 0
    for failure in ledger:
        if failure["cohort"] != cohort:
            continue
        if E.positive_window_intersects_reset(failure, wins):
            continue
        cands = E.pos_files(rows, failure, wins)
        if any(scores[c["file_id"]] >= threshold for c in cands):
            recalled += 1
    return recalled


def _onset_max_lead(
    rows: list[dict],
    ledger: list[dict],
    wins: dict[str, list[list[float]]],
    scores: dict[str, float],
    threshold: float = 0.3,
) -> float:
    """Return the maximum onset-arm lead (must be exactly 0.0)."""
    leads = []
    for failure in ledger:
        if E.positive_window_intersects_reset(failure, wins):
            continue
        cands = E.pos_files(rows, failure, wins)
        flagged = sorted(
            c["end_time"] for c in cands if scores[c["file_id"]] >= threshold
        )
        if flagged:
            leads.append(E.lead_days(failure["failure_time"], flagged))
    return max(leads) if leads else 0.0


#: Manifest-persisted solver provenance keys (protocol v4 §3b). The
#: deterministic subset only: ``solve_s`` varies run to run and stays in
#: in-memory records and task evidence, never in manifests, so persisted
#: manifests remain byte-identical across fresh-process reruns.
SOLVER_MANIFEST_KEYS = (
    "solver", "scipy_version", "solver_status",
    "solver_message", "n_variables", "n_constraints",
)


@dataclass
class Sprint15Binding:
    """Materialization-time binding for one Sprint 15 history."""

    quota: QuotaConfig = field(default_factory=QuotaConfig)
    profile: str = S15_PROFILE
    protocol: str = S15_PROTOCOL
    method: str = "joint"


def prepare_sprint15_block(
    *,
    rows: list[dict],
    ledger: list[dict],
    wins: dict[str, list[list[float]]],
    schedule_events: list[dict[str, object]],
    resolved_config: dict[str, object],
    config_hash: str,
    history_seed: int,
    role: str,
    quota: QuotaConfig = DEFAULT_QUOTA,
    profile: str = S15_PROFILE,
    protocol: str = S15_PROTOCOL,
    method: str = "joint",
) -> dict[str, object]:
    """Allocate quotas and build the ``sprint15`` manifest block.

    Validates the ``(seed, role)`` pair against the frozen roster, runs the
    fail-fast allocator before any shard is written, and returns the
    provenance block. Raises :class:`InfeasibleCandidate` or ``ValueError``.
    """
    expected = S15_ROSTER.get(history_seed)
    if expected is None or expected[1] != role:
        raise ValueError(
            f"sprint15 roster mismatch: seed {history_seed} role {role!r} "
            f"expected {expected!r}"
        )
    allocation = allocate_quotas(
        rows, ledger, wins, history_seed, quota, method)
    block_allocation: dict[str, object] = {
        "selected": allocation["selected"],
        "selected_ids": allocation["selected_ids"],
        "controls": allocation["controls"],
        "margins": allocation["margins"],
        "rejection": allocation["rejection"],
    }
    solver = allocation.get("solver")
    if isinstance(solver, dict):
        block_allocation["solver"] = {
            key: solver[key] for key in SOLVER_MANIFEST_KEYS}
    return {
        "profile": profile,
        "protocol": protocol,
        "role_group": expected[0],
        "history_seed": history_seed,
        "alloc_seed": allocation["alloc_seed"],
        "quota": allocation["quota"],
        "calendar_sha256": calendar_digest(schedule_events),
        "config_hash": config_hash,
        "nuisance": check_nuisance_envelopes(resolved_config),
        "signal_margin": analytic_signal_margin(),
        "allocation": block_allocation,
    }


def audit_sprint15(
    manifest: dict[str, object],
    quota: QuotaConfig = DEFAULT_QUOTA,
    profile: str = S15_PROFILE,
    protocol: str = S15_PROTOCOL,
) -> dict[str, object]:
    """Audit one materialized Sprint 15 history against the frozen contract."""

    errors: list[str] = []
    root_role = manifest.get("role")
    seeds = manifest["seeds"]
    history_seed = seeds["health"]
    block = manifest.get("sprint15") or {}

    def check(cond: bool, name: str) -> None:
        if not cond:
            errors.append(name)

    check(manifest.get("protocol") == protocol, "protocol-tag")
    check(block.get("protocol") == protocol, "block-protocol-tag")
    check(block.get("profile") == profile, "block-profile")
    check(all(s == history_seed for s in seeds.values()), "seed-match")
    expected = S15_ROSTER.get(history_seed)
    check(expected is not None and expected[1] == root_role, "roster-match")
    rows = manifest["files"]
    ledger = E.failure_ledger(manifest)
    wins = manifest["maintenance_windows"]
    check(all(set(r) <= ALLOWED_ROW_KEYS for r in rows), "no-score-keys")
    check(
        all("subtype" not in r and "cohort" not in r for r in rows),
        "no-subtype-in-rows",
    )
    by_robot: dict[str, list[dict]] = {}
    for row in rows:
        by_robot.setdefault(row["robot_id"], []).append(row)
    check(
        sorted(by_robot) == [f"robot-{i:02d}" for i in range(1, 10)],
        "nine-robots",
    )
    check(ROBOT_RESERVE in by_robot, "robot-reserve-present")
    check(PROGRAM_RESERVE in {r["program_id"] for r in rows}, "program-present")
    for robot, file_rows in by_robot.items():
        ordered = sorted(
            file_rows, key=lambda r: (r["start_time"], r["end_time"])
        )
        for prev, cur in zip(ordered, ordered[1:]):
            if not cur["start_time"] >= prev["end_time"] - 1e-6:
                errors.append(f"robot-serialization:{robot}")
                break
    for robot, intervals in wins.items():
        for start, end in intervals:
            if not end > start:
                errors.append(f"maint-bounded:{robot}")
                break
    cohorts = Counter(r["cohort"] for r in ledger)
    check(set(cohorts) == {"P", "W", "A"}, "all-cohorts")
    for record in ledger:
        if record["cohort"] == "A":
            check(
                record["duration_d"] == 0.0
                and record["degradation_onset"] is None
                and record["subtype"] in ("A1", "A2"),
                "a-shape",
            )
        elif record["cohort"] == "P":
            check(record["subtype"] in ("P1", "P2"), "p-subtype")
        else:
            check(record["subtype"] in ("W1", "W2"), "w-subtype")
    p_durs = sorted(r["duration_d"] for r in ledger if r["cohort"] == "P")
    w_durs = sorted(r["duration_d"] for r in ledger if r["cohort"] == "W")
    if p_durs:
        check(min(p_durs) >= 2.0 and max(p_durs) <= 15.0, "p-dist-shape")
        from statistics import median as _med

        check(5.0 <= _med(p_durs) <= 10.0, "p-dist-median")
    if w_durs:
        check(min(w_durs) >= 6.0 and max(w_durs) <= 28.0, "w-dist-shape")
        from statistics import median as _med

        check(12.0 <= _med(w_durs) <= 24.0, "w-dist-median")

    recomputed = calendar_digest(manifest["schedule"])
    check(block.get("calendar_sha256") == recomputed, "calendar-digest")
    check(block.get("config_hash") == manifest.get("config_hash"), "block-config")
    check(
        block.get("alloc_seed") == alloc_seed_for(history_seed), "alloc-seed"
    )
    allocation = block.get("allocation") or {}
    selected_ids = set(allocation.get("selected_ids", []))
    ledger_ids = {f["failure_id"] for f in ledger}
    check(bool(selected_ids) and selected_ids <= ledger_ids, "selected-known")
    selected = allocation.get("selected", {})
    quota_ok = True
    for cohort, subs in (
        ("P", (("P1", quota.p1), ("P2", quota.p2))),
        ("W", (("W1", quota.w1), ("W2", quota.w2))),
        ("A", (("A1", quota.a1), ("A2", quota.a2))),
    ):
        for subtype, need in subs:
            have = len(selected.get(cohort, {}).get(subtype, []))
            if have != need:
                quota_ok = False
                errors.append(f"quota-mismatch:{cohort}/{subtype}={have}")
    by_id = {f["failure_id"]: f for f in ledger}
    for cohort, subs in selected.items():
        for subtype, ids in subs.items():
            for fid in ids:
                record = by_id.get(fid)
                if record is None or record["cohort"] != cohort:
                    errors.append(f"selected-cohort:{fid}")
                elif record["subtype"] != subtype:
                    errors.append(f"selected-subtype:{fid}")
    spacing_ok = True
    per_robot_selected: dict[str, list[float]] = {}
    for cohort, subs in selected.items():
        for ids in subs.values():
            for fid in ids:
                record = by_id[fid]
                per_robot_selected.setdefault(
                    record["robot_id"], []).append(record["failure_time"])
    for robot, moments in per_robot_selected.items():
        for prev, cur in zip(sorted(moments), sorted(moments)[1:]):
            if cur - prev < quota.spacing_s - 1e-6:
                spacing_ok = False
                errors.append(f"spacing:{robot}")
                break

    pos_eval, uneval = 0, 0
    pos_cat: Counter = Counter()
    pos_eval_by_robot: Counter = Counter()
    sub_eval: Counter = Counter()
    sub_total: Counter = Counter()
    lead_support: dict[str, dict[str, object]] = {}
    _, _, lead = eligible_anchors(rows, ledger, wins)
    for failure in ledger:
        if failure["subtype"]:
            sub_total[failure["subtype"]] += 1
        if E.positive_window_intersects_reset(failure, wins):
            uneval += 1
            continue
        cands = E.pos_files(rows, failure, wins)
        if cands:
            pos_eval += 1
            pos_cat[failure["cohort"]] += 1
            pos_eval_by_robot[failure["robot_id"]] += 1
            if failure["subtype"]:
                sub_eval[failure["subtype"]] += 1
            if failure["cohort"] in ("P", "W"):
                lead_support[failure["failure_id"]] = lead.get(
                    failure["failure_id"], {}
                )
        else:
            uneval += 1
    anchors = E.anchor_rows(rows, wins)
    controls = E.select_control_windows(anchors, ledger, wins)
    ctrl_by_robot = Counter(w["robot_id"] for w in controls)
    eval_days = {
        (r["robot_id"], int(r["end_time"] // DAY)) for r in rows
        if E.eligible_operational_row(r, wins)
    }
    prog_by_op = {
        (r["robot_id"], r["end_time"]): r["program_id"] for r in rows
    }
    progs: dict[str, set[str]] = {}
    for failure in ledger:
        if E.positive_window_intersects_reset(failure, wins):
            continue
        if not E.pos_files(rows, failure, wins):
            continue
        if failure["cohort"] in ("P", "W"):
            prog = prog_by_op.get(
                (failure["robot_id"], failure["failure_time"])
            )
            if prog is not None:
                progs.setdefault(failure["cohort"], set()).add(prog)
    cutoff = manifest["calendar"]["cutoff_time"]
    healthy = [
        r for r in rows
        if r["file_label"] == "normal"
        and not r["is_quarantined"]
        and r["end_time"] <= cutoff
    ]
    healthy_eligible = [
        r for r in healthy
        if r["program_id"] != PROGRAM_RESERVE
        and r["robot_id"] != ROBOT_RESERVE
    ]
    dev_val_ids = set(manifest["splits"]["dev_val"])
    cal_rows = [
        r for r in rows
        if r["file_id"] in dev_val_ids
        and r["program_id"] != PROGRAM_RESERVE
        and r["robot_id"] != ROBOT_RESERVE
    ]
    total_pos = pos_eval
    robots_pos = sum(1 for v in pos_eval_by_robot.values() if v > 0)
    robots_neg = sum(1 for v in ctrl_by_robot.values() if v > 0)
    checks = {
        "P_ge_13": pos_cat.get("P", 0) >= 13,
        "W_ge_13": pos_cat.get("W", 0) >= 13,
        "A_ge_10": pos_cat.get("A", 0) >= 10,
        "total_ge_38": total_pos >= 38,
        "negatives_ge_32": len(controls) >= 32,
        "robot_days_ge_188": len(eval_days) >= 188,
        "robots_pos_ge_6": robots_pos >= 6,
        "robots_neg_ge_6": robots_neg >= 6,
        "programs_ge_2": all(len(progs.get(c, set())) >= 2 for c in ("P", "W")),
    }
    max_pos_share = (
        max(pos_eval_by_robot.values()) / total_pos if total_pos else 1.0
    )
    max_neg_share = (
        max(ctrl_by_robot.values()) / len(controls) if controls else 1.0
    )
    prog_shares = {}
    for cohort in ("P", "W"):
        tot = pos_cat.get(cohort, 0)
        per_prog: Counter = Counter()
        for failure in ledger:
            if failure["cohort"] != cohort:
                continue
            if E.positive_window_intersects_reset(failure, wins):
                continue
            if not E.pos_files(rows, failure, wins):
                continue
            prog = prog_by_op.get(
                (failure["robot_id"], failure["failure_time"])
            )
            per_prog[prog if prog is not None else "unknown"] += 1
        prog_shares[cohort] = max(per_prog.values()) / tot if tot else 1.0
    cohort_shares = {
        c: pos_cat.get(c, 0) / total_pos if total_pos else 0.0 for c in "PWA"
    }
    checks["robot_pos_le_35"] = max_pos_share <= 0.35
    checks["robot_neg_le_40"] = max_neg_share <= 0.40
    checks["program_le_60"] = all(v <= 0.60 for v in prog_shares.values())
    checks["cohort_mix_15_60"] = all(
        0.15 <= v <= 0.60 for v in cohort_shares.values()
    )

    def lead_ok(cohort: str) -> bool:
        events = [
            v for k, v in lead_support.items() if by_id[k]["cohort"] == cohort
        ]
        if not events:
            return False
        good = sum(
            1 for v in events
            if v.get("endpoints", 0) >= 3
            and v.get("early_endpoint")
            and v.get("clean_baseline")
        )
        return good / len(events) >= 0.80

    checks["lead_P_80"] = lead_ok("P")
    checks["lead_W_80"] = lead_ok("W")
    checks["fit_support_projection"] = len(healthy_eligible) >= 200
    checks["cal_support_projection"] = len(cal_rows) >= 40
    fixtures = eg3_fixtures(rows, ledger, wins)
    checks["eg3_pass"] = fixtures["pass"]
    checks["quota_exact"] = quota_ok and spacing_ok
    checks["eg1_pass"] = not errors
    structural = all(checks.values())
    return {
        "role": root_role,
        "seed": history_seed,
        "task12_errors": errors,
        "task12_pass": not errors,
        "positives_evaluable": pos_eval,
        "positives_by_cohort": dict(pos_cat),
        "positives_unevaluable": uneval,
        "negatives": len(controls),
        "negatives_by_robot": dict(ctrl_by_robot),
        "evaluated_robot_days": len(eval_days),
        "per_robot_evaluable_positives": dict(pos_eval_by_robot),
        "per_subtype": {
            s: {
                "evaluable": sub_eval.get(s, 0),
                "total": sub_total.get(s, 0),
            }
            for s in ("P1", "P2", "W1", "W2", "A1", "A2")
        },
        "lead_support": {
            "P_events": sum(
                1 for k in lead_support if by_id[k]["cohort"] == "P"
            ),
            "W_events": sum(
                1 for k in lead_support if by_id[k]["cohort"] == "W"
            ),
            "P_ok": checks["lead_P_80"],
            "W_ok": checks["lead_W_80"],
        },
        "allocated": {
            cohort: {st: len(ids) for st, ids in subs.items()}
            for cohort, subs in selected.items()
        },
        "rejection_reasons": dict((block.get("allocation") or {}).get(
            "rejection", {})),
        "concentration": {
            "max_pos_share": max_pos_share,
            "max_neg_share": max_neg_share,
            "program_shares": prog_shares,
            "cohort_shares": cohort_shares,
        },
        "support_projection": {
            "healthy_eligible": len(healthy_eligible),
            "cal_eligible_rows": len(cal_rows),
        },
        "nuisance": block.get("nuisance"),
        "signal_margin": block.get("signal_margin"),
        "promotion_checks": checks,
        "eg3": {
            "pass": fixtures["pass"],
            "checks": fixtures["checks"],
            "advance_recalled": fixtures["advance_recalled"],
            "constant_auc": fixtures["arms"]["constant"]["event_auc"],
        },
        "design_state": "DESIGN-PASS" if structural else "DESIGN-FAIL",
    }
