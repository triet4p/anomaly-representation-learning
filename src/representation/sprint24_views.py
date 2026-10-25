"""Immutable versioned views with a delayed label-release boundary (Sprint 24 Task 4).

This module implements the frozen ``S24-PROTO-v1`` Section 2 source-to-role
views without mutating source data. It reuses the existing loader and
eligibility conventions (``synth.events._overlaps_maintenance``,
``synth.balanced`` reserve identities, ``synth.chronicle`` manifest row
vocabulary) and adds one new convention beside them: versioned in-memory
view manifests keyed by history-namespaced IDs.

What it provides for T5/T6/T7:

- :func:`build_history_views` groups one history's manifest file rows into
  episode groups, assigns time frames (``V-ADAPT``/``V-SEQ-E*``/``V-FUTURE``),
  and draws the fixed disjoint slices (``V-VAL``/``V-TRIG``/``V-R0``) and
  nested label budgets (``B1 <= B2 <= B3``) with frozen seeds.
- :class:`LabelReleaseGate` enforces the delayed-feedback boundary: labels
  are released once per period close plus annotator latency, and any access
  before release fails closed.
- :class:`ReplayEligibility` answers the past-only replay question: a file
  is replay-eligible at period ``p`` only when its label was released in a
  strictly earlier period.

Out of scope by assignment: no model feature extraction or learners (T5/T6),
no sequential evaluator (Task 8). Views carry file metadata and label
references only; waveforms are never copied here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import random

from synth.balanced import PROGRAM_RESERVE, ROBOT_RESERVE
from synth.events import _overlaps_maintenance

#: View-manifest family frozen by ``S24-PROTO-v1`` Section 2.
VIEWS_FAMILY = "S24-VIEWS-v1"

#: Frozen protocol identifier these views implement.
PROTOCOL_ID = "S24-PROTO-v1"

#: Annotator latency added to every period close before labels release.
ANNOTATOR_LATENCY_S = 86_400.0

#: Failure/outside-attachment gap above which a file becomes its own group.
SINGLETON_GAP_S = 604_800.0

#: Frozen stratified-sampling seeds (``S24-PROTO-v1`` Sections 2-3, 6).
SEED_VAL = 2401
SEED_BUDGET = 2402
SEED_R0 = 2403
SEED_REPLAY = 2404
SEED_TRIG = 2410
SEED_RESERVE_SLICE = 2411

#: Fixed slice shapes ``(total, abnormal, normal)``.
SHAPE_VAL = (60, 12, 48)
SHAPE_TRIG = (100, 20, 80)
SHAPE_R0 = (50, 10, 40)

#: Nested label budgets ``(total, abnormal, normal)`` per history.
BUDGETS: Mapping[str, tuple[int, int, int]] = {
    "B1": (40, 8, 32),
    "B2": (120, 24, 96),
    "B3": (320, 64, 256),
}
BUDGET_ORDER = ("B1", "B2", "B3")

#: Named view/frame vocabulary. ``V-RESERVE-SLICE`` is a derived
#: identity slice of an eval frame, not a separate frame.
VIEW_NAMES = (
    "V-ADAPT",
    "V-VAL",
    "V-TRIG",
    "V-R0",
    "V-SEQ-E1",
    "V-SEQ-E2",
    "V-SEQ-E3",
    "V-FUTURE",
    "V-CONF-EVAL",
    "V-RESERVE-SLICE",
    "V-NF-E4",
    "V-NF-F",
)


def is_eligible_for_training(
    row: Mapping[str, object],
    windows: Mapping[str, Sequence[Sequence[float]]] | None = None,
) -> bool:
    """Frozen training eligibility (``S24-PROTO-v1`` Section 2).

    ``not quarantined`` AND ``not censored`` AND not a reserve identity
    AND no manifest maintenance-window overlap. Normal rows passing it
    are supervised negatives; abnormal rows passing it are supervised
    positives. Reserve identities never enter any training/validation
    frame. Extends the B0 observable healthy rule to abnormal rows, so
    the label check is intentionally absent here.
    """
    if row.get("is_quarantined") or row.get("is_censored"):
        return False
    if row.get("robot_id") == ROBOT_RESERVE:
        return False
    if row.get("program_id") == PROGRAM_RESERVE:
        return False
    if windows is not None and _overlaps_maintenance(
        dict(row),  # type: ignore[arg-type]
        dict(windows),  # type: ignore[arg-type]
    ):
        return False
    return True


def merge_intervals(intervals: Sequence[Sequence[float]]) -> list[list[float]]:
    """Merge overlapping ``[start, end)`` intervals (sorted, deterministic)."""
    merged: list[list[float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def build_episode_groups(
    history_id: str,
    file_rows: Sequence[Mapping[str, object]],
    failure_windows: Sequence[Mapping[str, object]],
    degradation_episodes: Sequence[Mapping[str, object]],
    maintenance_windows: Mapping[str, Sequence[Sequence[float]]] | None = None,
) -> tuple[dict[str, list[str]], list[dict[str, str]], dict[str, str]]:
    """Group files by ``(history_id, robot_id, merged related-event interval)``.

    Failure windows (degradation onset, or failure time when unlinked,
    through the linked maintenance end) are merged with same-robot
    overlapping degradation episodes (open episodes with missing time
    bounds are skipped: 8 per Development history in the Task 2
    inventory). Each merged span is split at manifest maintenance reset
    edges so no group ever crosses a reset boundary (edges belong to
    exactly one side; touching an edge exactly does not split).
    Eligible-healthy files inside intervals move with their episode.
    Outside-interval rule (owner decision S24-T4-D1 Q1): abnormal files
    attach to the nearest same-robot interval by ``|file end_time -
    interval end|`` (gap ``> 7`` days or no same-robot interval yields a
    singleton); NON-abnormal files outside every interval each form a
    singleton group ``(history_id, file_id)`` with no arbitrary healthy
    attachment. Every outside-interval file is reported in the returned
    audit list.

    Returns ``(groups, attached_from_outside, parent_of)``: ``group_id ->
    ordered file IDs`` plus one audit record per outside-interval file plus
    the D3 parent-custody map (split piece group -> parent merged-span id;
    singletons map to themselves). ``group_id`` is
    ``"{history_id}|{robot_id}|g{index:04d}"`` for merged intervals
    and ``"{history_id}|{file_id}"`` for singletons. Bare episode or
    file IDs are never keys: file IDs are globally unique but
    episode/failure IDs collide across histories, so every key is
    history-namespaced.
    """
    if not history_id or not isinstance(history_id, str):
        raise ValueError("history_id must be a non-empty string")
    windows_by_robot: dict[str, list[list[float]]] = {}
    if maintenance_windows:
        for robot, spans in maintenance_windows.items():
            windows_by_robot[str(robot)] = [[float(s), float(e)] for s, e in spans]
    seen_files: set[str] = set()
    for row in file_rows:
        file_id = row.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            raise ValueError("every file row must carry a non-empty file_id")
        if file_id in seen_files:
            raise ValueError(f"duplicate file_id in source rows: {file_id!r}")
        seen_files.add(file_id)

    per_robot: dict[str, list[list[float]]] = {}
    for window in failure_windows:
        robot = str(window["robot_id"])
        per_robot.setdefault(robot, []).append(
            [float(window["window_start"]), float(window["window_end"])]
        )
    for episode in degradation_episodes:
        if episode.get("start_time") is None or episode.get("end_time") is None:
            continue
        robot = str(episode["robot_id"])
        per_robot.setdefault(robot, []).append(
            [float(episode["start_time"]), float(episode["end_time"])]
        )
    merged_by_robot = {robot: merge_intervals(iv) for robot, iv in per_robot.items()}

    # D3 (owner-approved): split spans at manifest maintenance reset edges,
    # KEEP parent episode/trajectory linkage across pieces. Each split piece
    # records its parent merged-span index; ``parent_of`` maps every piece
    # group to its parent so role allocation keeps all pieces of one parent
    # in a single role (no fragments across train/validation/trigger/
    # retention/eval). Maintenance-overlap files stay excluded from
    # adaptation by the frozen eligibility predicate; no source timestamp
    # or waveform is cut or changed.
    parent_of: dict[str, str] = {}
    piece_parent: dict[str, str] = {}
    for robot in list(merged_by_robot):
        maint = windows_by_robot.get(robot, [])
        split: list[list[float]] = []
        parents: list[int] = []
        for parent_idx, (start, end) in enumerate(merged_by_robot[robot]):
            cuts = sorted({m for span in maint for m in span if start < m < end})
            bounds = [start] + cuts + [end]
            for lo, hi in zip(bounds, bounds[1:]):
                if lo < hi:
                    parents.append(parent_idx)
                    split.append([lo, hi])
        merged_by_robot[robot] = split
        for piece_idx, parent_idx in enumerate(parents):
            piece_id = f"{history_id}|{robot}|g{piece_idx:04d}"
            parent_id = f"{history_id}|{robot}|p{parent_idx:04d}"
            piece_parent[piece_id] = parent_id
    for piece_id, parent_id in piece_parent.items():
        parent_of[piece_id] = parent_id
    ordered_robots = sorted(merged_by_robot)
    group_index: dict[str, dict[int, str]] = {
        robot: {i: f"{history_id}|{robot}|g{i:04d}" for i in range(len(merged_by_robot[robot]))}
        for robot in ordered_robots
    }
    groups: dict[str, list[str]] = {}
    for robot in ordered_robots:
        for interval_index in range(len(merged_by_robot[robot])):
            groups[group_index[robot][interval_index]] = []

    attached_from_outside: list[dict[str, str]] = []
    rows_by_robot: dict[str, list[Mapping[str, object]]] = {}
    for row in file_rows:
        rows_by_robot.setdefault(str(row["robot_id"]), []).append(row)
    for robot in sorted(rows_by_robot):
        rows_by_robot[robot].sort(key=lambda r: (float(r["end_time"]), str(r["file_id"])))

    for robot in sorted(rows_by_robot):
        merged = merged_by_robot.get(robot, [])
        for row in rows_by_robot[robot]:
            file_id = str(row["file_id"])
            start = float(row["start_time"])
            end = float(row["end_time"])
            hit: int | None = None
            for i, (istart, iend) in enumerate(merged):
                # Split pieces share only endpoints (reset edges). A file
                # spanning a shared edge goes to the later piece holding
                # its end_time (istart < end <= iend); strict-interior
                # matches are unambiguous.
                holds_end = istart < end <= iend
                strict = istart < end and start < iend and start >= istart and end <= iend
                if holds_end or strict:
                    if hit is not None:
                        raise ValueError(
                            f"same-robot overlapping merged intervals for {file_id!r}; "
                            "intervals must be merged before grouping"
                        )
                    hit = i
            if hit is not None:
                groups[group_index[robot][hit]].append(file_id)
                continue
            if str(row.get("file_label")) == "abnormal":
                attached_from_outside.append({"file_id": file_id, "file_label": "abnormal"})
                best: int | None = None
                best_gap: float | None = None
                for i, (_, iend) in enumerate(merged):
                    gap = abs(end - iend)
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best = i
                if best is None or best_gap is None or best_gap > SINGLETON_GAP_S:
                    groups[f"{history_id}|{file_id}"] = [file_id]
                else:
                    groups[group_index[robot][best]].append(file_id)
            else:
                attached_from_outside.append(
                    {"file_id": file_id, "file_label": str(row.get("file_label")), "disposition": "singleton_per_S24-T4-D1-Q1"}
                )
                groups[f"{history_id}|{file_id}"] = [file_id]
    return groups, attached_from_outside, parent_of


def _stratified_take(
    candidates: Sequence[Mapping[str, object]],
    total: int,
    abnormal: int,
    normal: int,
    seed: int,
    purpose: str,
) -> list[Mapping[str, object]]:
    """Take a stratified ``(total, abnormal, normal)`` slice without replacement."""
    ab = sorted(
        (r for r in candidates if r.get("file_label") == "abnormal"),
        key=lambda r: str(r["file_id"]),
    )
    nm = sorted(
        (r for r in candidates if r.get("file_label") != "abnormal"),
        key=lambda r: str(r["file_id"]),
    )
    if len(ab) < abnormal or len(nm) < normal:
        raise ValueError(
            f"{purpose}: pool holds {len(ab)} abnormal / {len(nm)} normal, "
            f"needs {abnormal} / {normal}"
        )
    rng = random.Random(seed)
    picked = rng.sample(ab, abnormal) + rng.sample(nm, normal)
    picked_ids = [str(r["file_id"]) for r in picked]
    rng.shuffle(picked_ids)
    by_id = {str(r["file_id"]): r for r in candidates}
    return [by_id[i] for i in picked_ids]


def early_group_ids(
    member_to_group: Mapping[str, str],
    adapt_eligible: Sequence[str],
    by_id: Mapping[str, Mapping[str, object]],
) -> list[str]:
    """Return group IDs covering the earliest time half of eligible files."""
    ordered = sorted(adapt_eligible, key=lambda i: float(by_id[i]["end_time"]))
    early = ordered[: len(ordered) // 2] or ordered
    seen: list[str] = []
    for i in early:
        group_id = member_to_group.get(i)
        if group_id is not None and group_id not in seen:
            seen.append(group_id)
    return seen


def _take_groups_for_role(
    purpose: str,
    candidate_group_ids: Sequence[str],
    abnormal: int,
    normal: int,
    seed: int,
    claimed_groups: set[str],
    group_label_counts: Mapping[str, Mapping[str, int]],
    group_members: Mapping[str, Sequence[Mapping[str, object]]],
    parent_of: Mapping[str, str] | None = None,
    feasible_abnormal: bool = False,
) -> list[str]:
    """Claim whole parent episodes for one role, pick exact file quota inside.

    D3 parent custody: claiming one split piece claims its whole parent, so
    no parent episode ever spans two roles. Claimed parents join
    ``claimed_groups`` as parent ids (plus their pieces) so later roles can
    never reuse them; unused members stay unused in-role and never leak.
    Inside the claimed parents, exact ``(abnormal, normal)`` file quotas
    are picked with seeded ``random.sample`` (no replacement, no
    oversampling). Raises ``ValueError`` naming the role when the claimed
    parents cannot supply the quota.

    With ``feasible_abnormal`` (D2 per-history rule), the abnormal quota is
    the deterministic feasible count: claim parents in seed-shuffled order
    until no unclaimed parent adds abnormal, then take ALL available
    abnormal (up to the requested quota) and fill normals to the exact
    file total. The file total is always exact; the abnormal count is the
    source-derived feasible value, recorded per manifest.
    """
    parent_of = parent_of or {}

    def parent(g: str) -> str:
        return parent_of.get(g, g)

    ordered_parents: list[str] = []
    seen_parents: set[str] = set()
    for g in sorted(set(candidate_group_ids)):
        p = parent(g)
        if p not in seen_parents:
            seen_parents.add(p)
            ordered_parents.append(p)
    ordered_parents = [p for p in ordered_parents if p not in claimed_groups]
    rng = random.Random(seed)
    rng.shuffle(ordered_parents)
    total = abnormal + normal
    picked_parents: list[str] = []
    avail_ab = avail_nm = 0
    parent_pieces: dict[str, list[str]] = {}
    for g in sorted(set(candidate_group_ids)):
        parent_pieces.setdefault(parent(g), []).append(g)
    for p in ordered_parents:
        if avail_ab >= abnormal and avail_nm >= (abnormal + normal) - min(abnormal, avail_ab):
            break
        picked_parents.append(p)
        for g in parent_pieces.get(p, []):
            counts = group_label_counts.get(g, {"abnormal": 0, "normal": 0})
            avail_ab += int(counts.get("abnormal", 0))
            avail_nm += int(counts.get("normal", 0))
        if not feasible_abnormal and avail_ab >= abnormal and avail_nm >= normal:
            break
    if not feasible_abnormal:
        need_ab, need_nm = abnormal, normal
        if avail_ab < need_ab or avail_nm < need_nm:
            raise ValueError(
                f"{purpose}: claimed parents hold {avail_ab} abnormal / {avail_nm} normal, "
                f"need {need_ab} / {need_nm}"
            )
    else:
        need_ab = min(abnormal, avail_ab)
        need_nm = (abnormal + normal) - need_ab
        if avail_nm < need_nm:
            raise ValueError(
                f"{purpose}: feasible abnormal {need_ab} leaves normal shortfall "
                f"(have {avail_nm}, need {need_nm})"
            )
    for p in picked_parents:
        claimed_groups.add(p)
        for g in parent_pieces.get(p, []):
            claimed_groups.add(g)
    ab_pool: list[str] = []
    nm_pool: list[str] = []
    for p in picked_parents:
        for g in parent_pieces.get(p, []):
            for r in group_members.get(g, []):
                (ab_pool if r.get("file_label") == "abnormal" else nm_pool).append(str(r["file_id"]))
    ab_pool.sort()
    nm_pool.sort()
    return rng.sample(ab_pool, need_ab) + rng.sample(nm_pool, need_nm)

@dataclass(frozen=True)
class HistoryViews:
    """Immutable per-history view manifest (``S24-VIEWS-v1`` member).

    ``frames`` maps frame/view name -> ordered file IDs. ``groups`` maps
    group ID -> ordered file IDs. ``budgets`` maps ``B1/B2/B3`` -> ordered
    file IDs (nested: ``B1 <= B2 <= B3``). ``label_release`` maps period
    name -> release timestamp (period close + annotator latency).
    ``attached_from_outside`` audits every file attached to an interval it
    does not overlap (abnormal branch frozen; other labels auditable, see
    ``build_episode_groups``). ``source_digest`` binds the exact source rows
    this manifest derives from; ``manifest_digest`` identifies this manifest.
    """
    history_id: str
    protocol_id: str = PROTOCOL_ID
    family: str = VIEWS_FAMILY
    frames: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    budgets: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    label_release: Mapping[str, float] = field(default_factory=dict)
    attached_from_outside: tuple[Mapping[str, str], ...] = ()
    parent_of: Mapping[str, str] = field(default_factory=dict)
    source_digest: str = ""
    manifest_digest: str = ""

    def frame(self, name: str) -> tuple[str, ...]:
        """Return the ordered file IDs of one named frame/view."""
        if name not in self.frames:
            raise KeyError(f"unknown view frame: {name!r}")
        return self.frames[name]

    def group_of(self, file_id: str) -> str:
        """Return the group ID containing ``file_id``."""
        for group_id, members in self.groups.items():
            if file_id in members:
                return group_id
        raise KeyError(f"file {file_id!r} is in no group")


def _digest_manifest(payload: Mapping[str, object]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def build_history_views(
    history_id: str,
    file_rows: Sequence[Mapping[str, object]],
    failure_windows: Sequence[Mapping[str, object]],
    degradation_episodes: Sequence[Mapping[str, object]],
    maintenance_windows: Mapping[str, Sequence[Sequence[float]]] | None = None,
    *,
    cutoff_c1: float,
    cutoff_c2: float,
    bin_edges: Sequence[float] | None = None,
    withheld_family: str | None = None,
    feasible_abnormal: bool = True,
) -> HistoryViews:
    """Build the immutable per-history views for one Development history.

    Frame assignment is by ``end_time`` against the frozen Task 2 decile
    bin edges: ``V-ADAPT`` = bins 0-3 (``end_time <= c1``), ``V-SEQ-E1``
    = bin 4, ``V-SEQ-E2`` = bin 5, ``V-SEQ-E3`` = bin 6,
    ``V-FUTURE`` = bins 7-9 (``end_time > c2``). ``bin_edges`` carries
    the 11 recorded edges; when omitted, ``c1``/``c2`` alone define
    ADAPT/FUTURE and the open-closed ``c1`` to ``c2`` span splits into time thirds for
    the three experiences. Fixed slices ``V-VAL`` (seed 2401),
    ``V-TRIG`` (seed 2410) and ``V-R0`` (seed 2403) are drawn
    stratified without replacement from the training-eligible rows of
    ``V-ADAPT`` (``V-R0`` from the earliest time half of that eligible
    set); nested budgets ``B1 <= B2 <= B3`` (seed 2402) are drawn from
    the remainder. Ineligible files (e.g. quarantined observers inside
    an event window) belong to their group but never to a sampled frame.
    The withheld new-fault family, when given, is excluded from every Task
    7/``E1-E3`` training/validation frame and populates the post-release
    ``V-NF-E4`` (bins 7-8) / ``V-NF-F`` (bin 9) frames.

    Raises ``ValueError`` when two fixed/budget slices would share a
    file, when a group would split across the disjoint sampled frames
    (``V-VAL``/``V-TRIG``/``V-R0``/``B3``/``E1``/``E2``/``E3``), or when
    any pool is too small. Time frames legitimately share episode groups;
    the guard that rejects splits applies to the sampled disjoint frames,
    which is the actual leakage boundary. With ``feasible_abnormal``
    (D2 per-history rule), each role's abnormal quota is its deterministic
    feasible count at exact file totals (no ad hoc per-history numbers).
    """
    if not file_rows:
        raise ValueError("file_rows must be non-empty")
    if not (cutoff_c1 < cutoff_c2):
        raise ValueError("cutoffs must satisfy c1 < c2")
    edges = [float(e) for e in bin_edges] if bin_edges is not None else None
    if edges is not None:
        if len(edges) != 11 or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("bin_edges must be 11 strictly increasing edges")
        if edges[4] != cutoff_c1 or edges[7] != cutoff_c2:
            raise ValueError("bin_edges disagree with c1/c2 at edges[4]/edges[7]")
    groups, attached_from_outside, parent_of = build_episode_groups(
        history_id,
        file_rows,
        failure_windows,
        degradation_episodes,
        maintenance_windows,
    )
    by_id = {str(r["file_id"]): r for r in file_rows}
    if withheld_family is not None:
        withheld = {
            str(r["file_id"])
            for r in file_rows
            if r.get("anomaly_family") == withheld_family
            and r.get("file_label") == "abnormal"
        }
    else:
        withheld = set()

    member_to_group: dict[str, str] = {}
    for group_id, members in groups.items():
        for member in members:
            if member in member_to_group:
                raise ValueError(f"file {member!r} is in two groups")
            member_to_group[member] = group_id

    def in_bins(wanted: Sequence[int], file_id: str) -> bool:
        end = float(by_id[file_id]["end_time"])
        for b in wanted:
            lo, hi = edges[b], edges[b + 1]
            if b == 0:
                if end <= hi:
                    return True
            elif lo < end <= hi:
                return True
        return False

    def raw_frame_of(file_id: str) -> str:
        end = float(by_id[file_id]["end_time"])
        for name, bins in (("V-SEQ-E1", (4,)), ("V-SEQ-E2", (5,)), ("V-SEQ-E3", (6,))):
            for b in bins:
                lo, hi = edges[b], edges[b + 1]
                if lo < end <= hi:
                    return name
        return "V-SEQ-EDGE"

    # Groups spanning an E-bin edge snap to the frame holding their LATEST
    # member (matches the end-holding group assignment); the earlier frame
    # keeps only groups fully inside it. E-bin edges are measure-time bin
    # boundaries, not reset edges; no waveform moves. Snapping is at PARENT
    # level (D3 custody): all pieces of one parent go to the same frame.
    latest_frame: dict[str, str] = {}
    for gid, members in groups.items():
        got = [raw_frame_of(m) for m in members if m in by_id and raw_frame_of(m) != "V-SEQ-EDGE"]
        if got:
            latest_frame[gid] = max(got)
    latest_parent_frame: dict[str, str] = {}
    for gid, frame in latest_frame.items():
        p = parent_of.get(gid, gid)
        if p not in latest_parent_frame or frame > latest_parent_frame[p]:
            latest_parent_frame[p] = frame
    if edges is not None:
        adapt = sorted(i for i in by_id if in_bins((0, 1, 2, 3), i) and i not in withheld)
        raw_e1 = sorted(i for i in by_id if in_bins((4,), i) and i not in withheld)
        raw_e2 = sorted(i for i in by_id if in_bins((5,), i) and i not in withheld)
        raw_e3 = sorted(i for i in by_id if in_bins((6,), i) and i not in withheld)
        e1 = sorted(i for i in raw_e1 if latest_parent_frame.get(parent_of.get(member_to_group.get(i, i), member_to_group.get(i, i)), "V-SEQ-E1") == "V-SEQ-E1")
        e2 = sorted(i for i in raw_e2 if latest_parent_frame.get(parent_of.get(member_to_group.get(i, i), member_to_group.get(i, i)), "V-SEQ-E2") == "V-SEQ-E2")
        e3 = sorted(i for i in raw_e3 if latest_parent_frame.get(parent_of.get(member_to_group.get(i, i), member_to_group.get(i, i)), "V-SEQ-E3") == "V-SEQ-E3")
        future = sorted(i for i in by_id if in_bins((7, 8, 9), i))
        nf_e4 = sorted(i for i in future if in_bins((7, 8), i))
        nf_f = sorted(i for i in future if in_bins((9,), i))
    else:
        adapt = sorted(
            str(r["file_id"])
            for r in file_rows
            if float(r["end_time"]) <= cutoff_c1 and str(r["file_id"]) not in withheld
        )
        third = (cutoff_c2 - cutoff_c1) / 3.0
        e1 = sorted(
            str(r["file_id"])
            for r in file_rows
            if cutoff_c1 < float(r["end_time"]) <= cutoff_c1 + third
            and str(r["file_id"]) not in withheld
        )
        e2 = sorted(
            str(r["file_id"])
            for r in file_rows
            if cutoff_c1 + third < float(r["end_time"]) <= cutoff_c1 + 2 * third
            and str(r["file_id"]) not in withheld
        )
        e3 = sorted(
            str(r["file_id"])
            for r in file_rows
            if cutoff_c1 + 2 * third < float(r["end_time"]) <= cutoff_c2
            and str(r["file_id"]) not in withheld
        )
        future = sorted(
            str(r["file_id"]) for r in file_rows if float(r["end_time"]) > cutoff_c2
        )
        nf_e4, nf_f = [], []
    if not adapt or not e1 or not e2 or not e3 or not future:
        raise ValueError("time framing left an empty frame; check cutoffs")
    if edges is not None and (not nf_e4 or not nf_f):
        raise ValueError("withheld-family framing left V-NF-E4/V-NF-F empty")

    eligible_ids = {
        str(r["file_id"]) for r in file_rows if is_eligible_for_training(r, maintenance_windows)
    }
    adapt_eligible = [i for i in adapt if i in eligible_ids and i not in withheld]
    # Q2 GROUP-AWARE sampling (owner decision S24-T4-D1): whole episode
    # groups are allocated exclusively to slice/budget roles BEFORE file
    # selection; unused members of a claimed group stay unused in that
    # role and never leak elsewhere. Exact file quotas are then picked
    # inside the claimed groups (seeded, no replacement). Budgets are
    # nested B1 <= B2 <= B3 at file level within the B3-claimed groups.
    group_members: dict[str, list[Mapping[str, object]]] = {}
    for i in adapt_eligible:
        group_members.setdefault(member_to_group.get(i, f"{history_id}|{i}"), []).append(by_id[i])
    group_label_counts: dict[str, dict[str, int]] = {}
    for group_id, members in group_members.items():
        counts = {"abnormal": 0, "normal": 0}
        for r in members:
            counts["abnormal" if r.get("file_label") == "abnormal" else "normal"] += 1
        group_label_counts[group_id] = counts
    claimed_groups: set[str] = set()
    # E-frames are claimed at group level FIRST: their groups can never be
    # reused by V-R0/V-VAL/V-TRIG/B3, so no episode group spans the ADAPT
    # boundary, the V-TRIG boundary, or any two sampled frames.
    for i in list(e1) + list(e2) + list(e3):
        piece = member_to_group.get(i, f"{history_id}|{i}")
        claimed_groups.add(parent_of.get(piece, piece))
    parent_claimed: set[str] = set(claimed_groups)
    early_groups = early_group_ids(member_to_group, adapt_eligible, by_id)
    r0_ids = set(
        _take_groups_for_role("V-R0", early_groups, *SHAPE_R0[1:], SEED_R0, claimed_groups, group_label_counts, group_members, parent_of, feasible_abnormal)
    )
    val_ids = set(
        _take_groups_for_role("V-VAL", [g for g in sorted(group_members) if g not in claimed_groups], *SHAPE_VAL[1:], SEED_VAL, claimed_groups, group_label_counts, group_members, parent_of, feasible_abnormal)
    )
    trig_ids = set(
        _take_groups_for_role("V-TRIG", [g for g in sorted(group_members) if g not in claimed_groups], *SHAPE_TRIG[1:], SEED_TRIG, claimed_groups, group_label_counts, group_members, parent_of, feasible_abnormal)
    )
    pool_group_ids = [g for g in sorted(group_members) if g not in claimed_groups]
    b3_ids = _take_groups_for_role("B3", pool_group_ids, *BUDGETS["B3"][1:], SEED_BUDGET, claimed_groups, group_label_counts, group_members, parent_of, feasible_abnormal)
    budgets: dict[str, tuple[str, ...]] = {}
    budgets["B3"] = tuple(sorted(b3_ids))
    b3_ab = sorted(i for i in b3_ids if by_id[i].get("file_label") == "abnormal")
    b3_nm = sorted(i for i in b3_ids if by_id[i].get("file_label") != "abnormal")
    rng_b2 = random.Random(SEED_BUDGET + 1)
    b2_ab = rng_b2.sample(b3_ab, BUDGETS["B2"][1])
    b2_nm = rng_b2.sample(b3_nm, BUDGETS["B2"][2])
    budgets["B2"] = tuple(sorted(b2_ab + b2_nm))
    rng_b1 = random.Random(SEED_BUDGET + 2)
    b1_ab = rng_b1.sample(sorted(set(b2_ab)), BUDGETS["B1"][1])
    b1_nm = rng_b1.sample(sorted(set(b2_nm)), BUDGETS["B1"][2])
    budgets["B1"] = tuple(sorted(b1_ab + b1_nm))
    if not set(budgets["B1"]) <= set(budgets["B2"]) <= set(budgets["B3"]):
        raise ValueError("budgets are not nested B1 <= B2 <= B3")
    r0_rows = [by_id[i] for i in sorted(r0_ids)]
    val_rows = [by_id[i] for i in sorted(val_ids)]
    trig_rows = [by_id[i] for i in sorted(trig_ids)]
    pool_rows = [by_id[i] for i in sorted(set(adapt_eligible) - r0_ids - val_ids - trig_ids)]
    frames: dict[str, tuple[str, ...]] = {
        "V-ADAPT": tuple(adapt),
        "V-VAL": tuple(str(r["file_id"]) for r in val_rows),
        "V-TRIG": tuple(str(r["file_id"]) for r in trig_rows),
        "V-R0": tuple(str(r["file_id"]) for r in r0_rows),
        "V-SEQ-E1": tuple(e1),
        "V-SEQ-E2": tuple(e2),
        "V-SEQ-E3": tuple(e3),
        "V-FUTURE": tuple(future),
        "V-CONF-EVAL": (),
        "V-NF-E4": tuple(nf_e4),
        "V-NF-F": tuple(nf_f),
    }

    reserve_ids = tuple(
        sorted(
            i
            for name in ("V-SEQ-E1", "V-SEQ-E2", "V-SEQ-E3", "V-FUTURE")
            for i in frames[name]
            if by_id[i].get("robot_id") == ROBOT_RESERVE
            or by_id[i].get("program_id") == PROGRAM_RESERVE
        )
    )
    frames["V-RESERVE-SLICE"] = reserve_ids

    seen: set[str] = set()
    for name in ("V-VAL", "V-TRIG", "V-R0"):
        for i in frames[name]:
            if i in seen:
                raise ValueError(f"file {i!r} in two fixed slices")
            seen.add(i)
    pool_ids = pool_rows_ids(pool_rows)
    for name in BUDGET_ORDER:
        if not set(budgets[name]) <= pool_ids - seen:
            raise ValueError(f"budget {name} overlaps a fixed slice")
    disjoint_frames = {
        "V-VAL": set(frames["V-VAL"]),
        "V-TRIG": set(frames["V-TRIG"]),
        "V-R0": set(frames["V-R0"]),
        "B3": set(budgets["B3"]),
        "V-SEQ-E1": set(frames["V-SEQ-E1"]),
        "V-SEQ-E2": set(frames["V-SEQ-E2"]),
        "V-SEQ-E3": set(frames["V-SEQ-E3"]),
    }
    group_frames: dict[str, set[str]] = {}
    for frame_name, members in disjoint_frames.items():
        for i in members:
            group_frames.setdefault(member_to_group.get(i, f"{history_id}|{i}"), set()).add(frame_name)
    split_groups = {g: f for g, f in group_frames.items() if len(f) > 1}
    if split_groups:
        first = next(iter(split_groups))
        raise ValueError(
            f"group {first!r} splits across disjoint frames {sorted(split_groups[first])}"
        )

    third = (cutoff_c2 - cutoff_c1) / 3.0
    label_release = {
        "V-ADAPT": cutoff_c1 + ANNOTATOR_LATENCY_S,
        "V-SEQ-E1": cutoff_c1 + third + ANNOTATOR_LATENCY_S,
        "V-SEQ-E2": cutoff_c1 + 2 * third + ANNOTATOR_LATENCY_S,
        "V-SEQ-E3": cutoff_c2 + ANNOTATOR_LATENCY_S,
    }
    source_digest = _digest_manifest(
        {
            "history_id": history_id,
            "rows": [
                {
                    "file_id": str(r["file_id"]),
                    "robot_id": str(r["robot_id"]),
                    "program_id": str(r["program_id"]),
                    "start_time": float(r["start_time"]),
                    "end_time": float(r["end_time"]),
                    "file_label": str(r.get("file_label")),
                    "is_quarantined": bool(r.get("is_quarantined")),
                    "is_censored": bool(r.get("is_censored")),
                    "anomaly_family": str(r.get("anomaly_family", "")),
                }
                for r in sorted(file_rows, key=lambda r: str(r["file_id"]))
            ],
        }
    )
    manifest_digest = _digest_manifest(
        {
            "protocol": PROTOCOL_ID,
            "family": VIEWS_FAMILY,
            "history_id": history_id,
            "frames": {k: list(v) for k, v in frames.items()},
            "budgets": {k: list(v) for k, v in budgets.items()},
            "groups": groups,
            "cutoffs": [cutoff_c1, cutoff_c2],
            "bin_edges": list(edges) if edges is not None else None,
            "seeds": {
                "V-VAL": SEED_VAL,
                "budgets": SEED_BUDGET,
                "V-R0": SEED_R0,
                "replay": SEED_REPLAY,
                "V-TRIG": SEED_TRIG,
            },
            "withheld_family": withheld_family,
            "source_digest": source_digest,
        }
    )
    return HistoryViews(
        history_id=history_id,
        frames={k: tuple(v) for k, v in frames.items()},
        budgets={k: tuple(v) for k, v in budgets.items()},
        groups={k: tuple(v) for k, v in groups.items()},
        label_release=dict(label_release),
        attached_from_outside=tuple(attached_from_outside),
        parent_of=dict(parent_of),
        source_digest=source_digest,
        manifest_digest=manifest_digest,
    )


def tmax(
    ids: Sequence[str], by_id: Mapping[str, Mapping[str, object]]
) -> float:
    """Return the maximum ``end_time`` over ``ids``."""
    return max(float(by_id[i]["end_time"]) for i in ids)


def pool_rows_ids(pool_rows: Sequence[Mapping[str, object]]) -> set[str]:
    """Return the file IDs of one pool row sequence."""
    return {str(r["file_id"]) for r in pool_rows}


class LabelReleaseGate:
    """Delayed-feedback label boundary for one history's views.

    Labels release once per period close plus annotator latency
    (``S24-PROTO-v1`` Section 4): Task 7 releases everything at
    ``c1 + 86,400`` s; Task 9 releases experience ``Ek`` at
    ``end(Ek) + 86,400`` s. ``released_labels(now_s)`` returns only the
    labels whose release time has passed; :meth:`require_released`
    fails closed on premature access. Evaluation labels
    (``V-FUTURE``/``V-CONF-EVAL``/retention probes) are never released
    to the learner through this gate.
    """

    def __init__(
        self,
        views: HistoryViews,
        labels: Mapping[str, Mapping[str, object]],
    ) -> None:
        unknown = set(labels) - {
            i for members in views.groups.values() for i in members
        } - set(views.frames.get("V-CONF-EVAL", ()))
        if unknown:
            raise ValueError(f"labels reference {len(unknown)} unknown files")
        self._views = views
        self._labels = dict(labels)

    @property
    def views(self) -> HistoryViews:
        """Return the bound immutable views."""
        return self._views

    def release_time_of(self, file_id: str) -> float:
        """Return the release timestamp of one file's label."""
        for period in ("V-ADAPT", "V-SEQ-E1", "V-SEQ-E2", "V-SEQ-E3"):
            if file_id in self._views.frames.get(period, ()):
                release = self._views.label_release.get(period)
                if release is None:
                    raise KeyError(f"no release schedule for period {period}")
                return release
        raise KeyError(
            f"file {file_id!r} is not in a label-releasing period; "
            "evaluation labels are never released to the learner"
        )

    def released_labels(
        self, now_s: float
    ) -> Mapping[str, Mapping[str, object]]:
        """Return labels released strictly before ``now_s`` (past-only)."""
        released: dict[str, Mapping[str, object]] = {}
        for file_id, label in self._labels.items():
            try:
                release = self.release_time_of(file_id)
            except KeyError:
                continue
            if release <= float(now_s):
                released[file_id] = label
        return released

    def require_released(
        self, file_ids: Sequence[str], now_s: float
    ) -> Mapping[str, Mapping[str, object]]:
        """Return labels for ``file_ids`` or raise on any premature access."""
        premature = [
            i
            for i in file_ids
            if self.release_time_of(i) > float(now_s)
        ]
        if premature:
            raise ValueError(
                f"{len(premature)} labels not yet released at t={now_s} "
                f"(first: {premature[0]!r})"
            )
        return {i: self._labels[i] for i in file_ids}


class ReplayEligibility:
    """Past-only replay eligibility for bounded replay buffers (Task 9).

    A file is replay-eligible at period ``p`` only when its label was
    released in a strictly earlier period. The buffer cap (``K = 256``
    files, class-balanced reservoir, eviction seed 2404) is enforced by
    the Task 8 evaluator; this class answers the eligibility question
    that keeps future examples out of replay.
    """

    def __init__(self, gate: LabelReleaseGate) -> None:
        self._gate = gate

    def eligible_at(
        self, file_ids: Sequence[str], now_s: float
    ) -> tuple[str, ...]:
        """Return the past-only replay-eligible subset of ``file_ids``."""
        released = self._gate.released_labels(now_s)
        return tuple(i for i in file_ids if i in released)

    def require_past_only(
        self, file_ids: Sequence[str], now_s: float
    ) -> None:
        """Raise when any of ``file_ids`` has no released label yet."""
        ineligible = [i for i in file_ids if i not in self._gate.released_labels(now_s)]
        if ineligible:
            raise ValueError(
                f"{len(ineligible)} replay candidates draw on unreleased "
                f"(present/future) labels (first: {ineligible[0]!r})"
            )
