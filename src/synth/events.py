"""Deterministic event-window and metric reference implementation (Task 9).

Pure functions over materialized manifests implementing the Task 2 / Protocol
v4 §4 semantics: eligibility predicates, greedy control selection, and the
E1–E5 companions. Operates on manifest file rows plus the failure ledger;
never on model internals. Scores enter as explicit ``{file_id: float}`` maps
(deterministic fixtures or causal scorers), so metric computability stays
independent of any model. Undefined metrics raise :class:`UnavailableError`
instead of falling back to file-level substitutes.
"""

from __future__ import annotations

import math
import statistics

#: Primary causal horizon in seconds (H = 7 days).
HORIZON_S = 7.0 * 86400.0

#: Alert-episode grouping gap in seconds (alerts ≤ 2 days apart join).
EPISODE_GAP_S = 2.0 * 86400.0

#: Frozen temporal evaluation view for event-level units (v4.1.1 §7).
TEMPORAL_VIEW = "test-temporal"

#: Minimum recalled events for a reportable median (Task 3 T-LEAD).
MEDIAN_MINIMUM_N = 10

_ALLOWED_COHORTS = ("P", "W", "A")
_ALLOWED_SEVERITIES = (1.0, 2.0, 4.0)


class UnavailableError(ValueError):
    """An event metric is undefined (too few eligible units)."""


def failure_ledger(manifest: dict) -> list[dict]:
    """Return the validated failure-event ledger from a manifest."""
    records = list(manifest.get("failure_events", []))
    for record in records:
        if record["cohort"] not in _ALLOWED_COHORTS:
            raise ValueError(f"unknown cohort {record['cohort']!r}")
        if record["severity"] not in _ALLOWED_SEVERITIES:
            raise ValueError(f"bad severity {record['severity']!r}")
        if record["cohort"] == "A":
            if record["degradation_onset"] is not None:
                raise ValueError("abrupt ledger records carry no onset")
            if record["duration_d"] != 0.0:
                raise ValueError("abrupt ledger records have zero duration")
        elif not (
            record["degradation_onset"] is not None
            and record["degradation_onset"] < record["failure_time"]
            and record["duration_d"] > 0.0
        ):
            raise ValueError("non-abrupt ledger records need onset and duration")
    return records


def _overlaps_maintenance(
    row: dict, windows: dict[str, list[list[float]]]
) -> bool:
    """Check whether a file interval overlaps any maintenance window."""
    for start, end in windows.get(row["robot_id"], []):
        if start < row["end_time"] and row["start_time"] < end:
            return True
    return False


def window_intersects_reset(
    start: float, end: float, robot_id: str,
    windows: dict[str, list[list[float]]],
) -> bool:
    """Check whether a continuous window span crosses a reset boundary.

    Any maintenance window intersecting the open span ``(start, end)`` —
    i.e. ``win.start < end and start < win.end`` — makes horizons
    unevaluable and splits alert episodes (Task 2 §7, no
    cross-maintenance lookback). Window endpoints touching a boundary
    exactly do not count as crossing.
    """
    for win_start, win_end in windows.get(robot_id, []):
        if win_start < end and start < win_end:
            return True
    return False


def positive_window_intersects_reset(
    failure: dict, windows: dict[str, list[list[float]]],
    horizon: float = HORIZON_S,
) -> bool:
    """Check whether a positive ``[T−H, T]`` horizon spans a reset."""
    stop = failure["failure_time"]
    return window_intersects_reset(
        stop - horizon, stop, failure["robot_id"], windows)


def eligible_operational_row(
    row: dict, windows: dict[str, list[list[float]]]
) -> bool:
    """Shared E5/robot-day eligibility (v4.1 §7, v4.1.1 conformance).

    A row counts iff it is in the frozen temporal view, NOT censored, AND
    NOT full-interval maintenance-overlapping. Used identically for the E5
    numerator (scored rows), the E5 denominator days, and robot-day floors.
    A (robot, day) counts iff at least one eligible row ends that day.
    """
    return (TEMPORAL_VIEW in row.get("member_views", [])
            and not row["is_censored"]
            and not _overlaps_maintenance(row, windows))

def pos_files(
    rows: list[dict],
    failure: dict,
    windows: dict[str, list[list[float]]],
    *,
    exclude_programs: tuple[str, ...] = (),
    exclude_robots: tuple[str, ...] = (),
) -> list[dict]:
    """POS-ELIGIBLE files for one failure (v4.1.1 §7: temporal view, no
    quarantine condition). Row-level maintenance overlap excluded here;
    callers must ALSO treat reset-spanning ``[T−H, T]`` horizons as
    unevaluable via :func:`positive_window_intersects_reset`."""
    start = failure["failure_time"] - HORIZON_S
    stop = failure["failure_time"]
    return [
        row for row in rows
        if row["robot_id"] == failure["robot_id"]
        and row["program_id"] not in exclude_programs
        and row["robot_id"] not in exclude_robots
        and TEMPORAL_VIEW in row.get("member_views", [])
        and not row["is_censored"]
        and start <= row["end_time"] <= stop
        and not _overlaps_maintenance(row, windows)
    ]


def anchor_rows(
    rows: list[dict],
    windows: dict[str, list[list[float]]],
    *,
    exclude_programs: tuple[str, ...] = (),
    exclude_robots: tuple[str, ...] = (),
) -> list[dict]:
    """ANCHOR-ELIGIBLE file rows (v4.1.1 §7: temporal view + quarantine excluded)."""
    return [
        row for row in rows
        if row["program_id"] not in exclude_programs
        and row["robot_id"] not in exclude_robots
        and TEMPORAL_VIEW in row.get("member_views", [])
        and not row["is_censored"]
        and not row["is_quarantined"]
        and not _overlaps_maintenance(row, windows)
    ]


def select_control_windows(
    anchors: list[dict],
    failures: list[dict],
    windows: dict[str, list[list[float]]] | None = None,
) -> list[dict]:
    """Greedy deterministic non-overlapping control windows (Task 2 §4).

    Candidates ``[e−H, e]`` anchored at eligible end-times in end-time order;
    a candidate is kept iff no failure of the same robot starts in
    ``[e−H, e+H)`` AND (when ``windows`` is given) the span ``[e−H, e]``
    crosses no maintenance/recommission boundary. Returns
    ``{"anchor_end", "robot_id", "members"}`` with member rows drawn from
    the eligible anchors inside the window.
    """
    starts: dict[str, list[float]] = {}
    for failure in failures:
        starts.setdefault(failure["robot_id"], []).append(
            failure["failure_time"])
    ordered = sorted(anchors, key=lambda row: row["end_time"])
    kept: list[dict] = []
    last_kept: dict[str, float] = {}
    for anchor in ordered:
        end = anchor["end_time"]
        robot = anchor["robot_id"]
        if end - last_kept.get(robot, -math.inf) < HORIZON_S:
            continue
        if any(start >= end - HORIZON_S and start < end + HORIZON_S
               for start in starts.get(robot, [])):
            continue
        if windows is not None and window_intersects_reset(
                end - HORIZON_S, end, robot, windows):
            continue
        members = [
            row for row in ordered
            if row["robot_id"] == robot
            and end - HORIZON_S <= row["end_time"] <= end
        ]
        kept.append({"anchor_end": end, "robot_id": robot, "members": members})
        last_kept[robot] = end
    return kept


def window_score(member_ids: list[str], scores: dict[str, float]) -> float:
    """Max file score over one window's member ids."""
    missing = [i for i in member_ids if i not in scores]
    if missing:
        raise KeyError(f"scores missing for {len(missing)} files")
    return max(scores[i] for i in member_ids)


def roc_auc_tie_aware(pos: list[float], neg: list[float]) -> float:
    """Tie-aware ROC AUC via mean ranks (sklearn convention).

    Tied blocks share average ranks; a constant arm yields exactly 0.5.
    Raises :class:`UnavailableError` when either class is empty.
    """
    if not pos or not neg:
        raise UnavailableError("event AUC needs both classes non-empty")
    order = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum = 0.0
    index = 0
    while index < len(order):
        tie_end = index
        while (tie_end + 1 < len(order)
               and order[tie_end + 1][0] == order[index][0]):
            tie_end += 1
        mean_rank = (index + 1 + tie_end + 1) / 2.0
        rank_sum += mean_rank * sum(
            label for _, label in order[index:tie_end + 1])
        index = tie_end + 1
    auc = (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return min(1.0, max(0.0, auc))


def false_alert_episodes(
    flagged_by_robot: dict[str, list[float]],
    failures: list[dict],
    evaluated_robot_days: float,
    windows: dict[str, list[list[float]]] | None = None,
) -> tuple[int, float]:
    """Group flagged ends into ≤2 d episodes; count false ones and the rate.

    An episode is false iff none of its alerts falls inside any failure
    window ``[T−H, T]`` of the same robot. When ``windows`` is given,
    episodes are additionally split at maintenance/recommission boundaries
    so no episode bridges a reset (Task 2 §7).
    """
    if evaluated_robot_days <= 0.0:
        raise UnavailableError("false-alert rate needs a positive denominator")
    windows_by_robot: dict[str, list[tuple[float, float]]] = {}
    for failure in failures:
        windows_by_robot.setdefault(failure["robot_id"], []).append(
            (failure["failure_time"] - HORIZON_S, failure["failure_time"]))

    def _false(episode: list[float], robot: str) -> bool:
        return not any(
            start <= alert <= stop
            for alert in episode
            for start, stop in windows_by_robot.get(robot, []))

    def _split_at_reset(prev: float, curr: float, robot: str) -> bool:
        if windows is None:
            return False
        return any(ms < curr and prev < me
                   for ms, me in windows.get(robot, []))

    false = 0
    for robot, ends in flagged_by_robot.items():
        episode: list[float] = []
        for end in sorted(ends):
            if episode and (end - episode[-1] > EPISODE_GAP_S
                            or _split_at_reset(episode[-1], end, robot)):
                if _false(episode, robot):
                    false += 1
                episode = []
            episode.append(end)
        if episode and _false(episode, robot):
            false += 1
    return false, false / evaluated_robot_days


def median_iqr(values: list[float]) -> tuple[float, float, float]:
    """Median and hinges (Q1, Q3) via exclusive quartiles; needs n ≥ 1."""
    if not values:
        raise UnavailableError("median needs at least one value")
    ordered = sorted(values)
    median = statistics.median(ordered)
    hinges = statistics.quantiles(ordered, n=4, method="exclusive")
    return median, hinges[0], hinges[2]


def summarize_values(
    values: list[float], minimum: int = MEDIAN_MINIMUM_N,
) -> dict:
    """Frozen lead/persistence report: median+IQR, or SPARSE below minimum.

    Returns ``{"n", "status": "ok", "median", "q1", "q3"}`` for n ≥ minimum,
    else ``{"n", "status": "sparse"}`` — a defined report, never a fallback.
    """
    result: dict = {"n": len(values)}
    if len(values) < minimum:
        result["status"] = "sparse"
        return result
    median, q1, q3 = median_iqr(values)
    result.update({"status": "ok", "median": median, "q1": q1, "q3": q3})
    return result


def lead_days(failure_time: float, flagged_ends: list[float]) -> float:
    """First-alert lead time in days (0.0 for onset detection)."""
    return (failure_time - min(flagged_ends)) / 86400.0


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided binomial interval by bisection on tail sums."""
    if n <= 0:
        raise UnavailableError("recall interval needs n > 0")
    if not 0 <= k <= n:
        raise ValueError("k must lie in [0, n]")

    def tail(prob: float, low: bool) -> float:
        total = 0.0
        for i in range(k if low else 0, (n + 1) if low else k + 1):
            total += math.comb(n, i) * prob**i * (1.0 - prob) ** (n - i)
        return total

    def invert(target_is_cdf: bool) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            value = tail(mid, target_is_cdf)
            if value > alpha / 2.0:
                if target_is_cdf:
                    hi = mid
                else:
                    lo = mid
            else:
                if target_is_cdf:
                    lo = mid
                else:
                    hi = mid
        return (lo + hi) / 2.0

    lower = 0.0 if k == 0 else invert(True)
    upper = 1.0 if k == n else invert(False)
    return lower, upper


def rule_of_three_bound(denominator: float) -> float:
    """95% upper bound for a zero numerator over a positive denominator."""
    if denominator <= 0.0:
        raise UnavailableError("rule-of-three needs a positive denominator")
    return 3.0 / denominator


def check_arm_computable(arm: dict) -> bool:
    """Validate a complete frozen E1–E5 composite output (Task 14 contract).

    Requires every estimand present and defined: event AUC value, recall
    value plus its Clopper–Pearson interval, lead and persistence summaries
    (``ok`` or ``sparse`` both count as defined reports), and the FAR value
    (plus the zero-count bound whenever zero false episodes were observed).
    """
    if arm.get("event_auc") is None:
        return False
    if arm.get("recall") is None or arm.get("recall_cp") is None:
        return False
    for key in ("lead", "persistence"):
        summary = arm.get(key)
        if not isinstance(summary, dict) or summary.get("n") is None:
            return False
        if summary.get("status") == "ok" and summary.get("median") is None:
            return False
        if summary.get("status") not in ("ok", "sparse"):
            return False
    if arm.get("far") is None:
        return False
    if arm.get("false_episodes") == 0 and arm.get("far_bound") is None:
        return False
    return True
