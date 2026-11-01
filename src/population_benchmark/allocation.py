"""Deterministic, collision-free physical-source allocation (S24-7-M20 A02).

The frozen index bands keep reference, bandwidth, calibration and held-out test
*indices* disjoint. Disjoint indices do not make disjoint *sources*:

1. the accepted pair-seed rule truncates SHA-256 to 31 bits, so two planned
   indices can derive the same pair seed and therefore the same
   ``group_id``/``source_id``/file ids and byte-identical waveforms;
2. the generator derives the actual clean source from the pair seed, so two
   *different* pair seeds can still resolve to the same physical source. The
   accepted public constructor draws ``normal_seed`` first from
   ``default_rng(pair_seed)`` and then builds the whole clean stream (fleet,
   regimes, gain/offset, causal draw, noise seed) from ``default_rng(normal_seed)``.
   The unmodified twin is a pure function of that first draw, so equal clean
   keys give byte-identical twins while the metadata ids differ.

Both layers are counted here, and the census covers the exact ACTIVE plan: one
row per logical source — ``(role, history, episode, slot)``. An intended
twin/variant pair and the week/month nesting of one episode are the SAME logical
source and appear once; independent roles, episodes and histories must not share
a physical source.

The resolver is a deterministic first-claim allocation over the full census:

* the canonical-first claimant of a physical-source key keeps its planned index
  and seed, so every uncollided source generates exactly the legacy waveform;
* each later claimant probes its own dedicated index namespace
  (``PROBE_INDEX_BASE + row * MAX_PROBES + attempt``) until the derived seed
  yields an unused key;
* the result is post-derivation verified for global key/seed uniqueness and
  enforced minimality, and it is frozen by ``digest``.

The allocator never inspects outcomes, labels, features or encoder state, and it
does not change ``synth.generator.derive_pair_seed``, so the historical
M11-M19 deterministic datasets are untouched.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synth.generator import derive_pair_seed

#: Frozen resolver identity, recorded in the design/contract.
ALLOCATION_VERSION = "S24-PB-SOURCE-ALLOCATION-V2"
IDENTITY_VERSION = "S24-PB-SOURCE-IDENTITY-V2"

#: Roles whose sources are plain normal draws (``NormalGenerator``).
NORMAL_ROLES: tuple[str, ...] = ("reference", "bandwidth")
#: Roles whose sources are paired draws (``generate_paired_twin``).
PAIRED_ROLES: tuple[str, ...] = ("calibration", "test")
ROLES: tuple[str, ...] = NORMAL_ROLES + PAIRED_ROLES

#: The generator's 31-bit seed space (``derive_pair_seed`` and ``normal_seed``).
SEED_SPACE = 2 ** 31

#: Probe indices live far outside every planned band (the largest planned test
#: index is below 1.3 million; the deepest probe index is below 1.003e9).
PROBE_INDEX_BASE = 1_000_000_000
MAX_PROBES = 8

RESOLVER = (
    "canonical order (role reference<bandwidth<calibration<test, then history, "
    "then plan episode order, then slot); the first claimant of a "
    "physical-source key keeps its planned index and seed; every later claimant "
    "probes its dedicated index namespace probe_base+row*max_probes+attempt "
    "until the derived seed yields an unused key; no outcome, label or encoder "
    "state enters the resolver"
)

#: Digest recipe, published so a reviewer can recompute it independently.
DIGEST_RULE = (
    "sha256(ALLOCATION_VERSION + '\\n' + concat(row + '\\n')) with "
    "row = role|history_index|episode_id|slot|source_index|seed|source_key|probe_tier"
)


def _refused(reason: str, detail: str) -> ValueError:
    return ValueError(f"{reason}: {detail}")


def paired_clean_key(pair_seed: int) -> int:
    """Physical-source key of a paired draw: its first derived normal seed.

    ``SessionGenerator._anomaly_source`` draws ``normal_seed`` first from
    ``np.random.default_rng(pair_seed)``; every later draw of that source (fleet,
    regimes, gain/offset, causal stream, noise seed, and therefore the unmodified
    twin) comes from ``np.random.default_rng(normal_seed)``. ``normal_seed`` is
    the identity of the physical source, not the 31-bit pair seed.
    """
    return int(np.random.default_rng(int(pair_seed)).integers(0, SEED_SPACE))


def source_key(role: str, seed: int) -> int:
    """Physical-source key of one logical source in frozen role order."""
    if role in NORMAL_ROLES:
        # ``NormalGenerator.generate`` draws its whole stream from
        # ``default_rng(seed)``, so the seed itself is the clean-source key.
        return int(seed)
    if role in PAIRED_ROLES:
        return paired_clean_key(int(seed))
    raise _refused("REFUSED-UNKNOWN-DESIGN", f"unknown source role {role!r}")


@dataclass(frozen=True)
class SourceKey:
    """One logical source row of the exact active census."""

    role: str
    history_index: int
    episode_id: str | None
    slot: int | None
    source_index: int

    def identity(self) -> tuple[str, int, int]:
        return (self.role, int(self.history_index), int(self.source_index))


@dataclass(frozen=True)
class Allocation:
    """Frozen assignment of one generator seed to every logical source."""

    version: str
    identity_version: str
    history_ids: tuple[str, ...]
    sources: tuple[SourceKey, ...]
    seeds: tuple[int, ...]
    keys: tuple[int, ...]
    tiers: tuple[int, ...]
    probe_indices: tuple[int | None, ...]
    legacy_seeds: tuple[int, ...]
    legacy_keys: tuple[int, ...]
    digest: str
    payload: dict[str, Any]
    _row_of: dict[tuple[str, int, int], int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.sources) != len(self.seeds):
            raise _refused("REFUSED-IDENTITY", "allocation row count drifted")
        row_of: dict[tuple[str, int, int], int] = {}
        for index, source in enumerate(self.sources):
            row_of[source.identity()] = index
        object.__setattr__(self, "_row_of", row_of)

    # -- lookup ---------------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.sources)

    def row(self, role: str, history_index: int, source_index: int) -> int:
        try:
            return self._row_of[(str(role), int(history_index), int(source_index))]
        except KeyError:
            raise _refused(
                "REFUSED-IDENTITY",
                f"source is not in the frozen census: {role}/{history_index}/{source_index}",
            ) from None

    def seed_for(self, role: str, history_index: int, source_index: int) -> int:
        return int(self.seeds[self.row(role, history_index, source_index)])

    def key_for(self, role: str, history_index: int, source_index: int) -> int:
        return int(self.keys[self.row(role, history_index, source_index)])

    def tier_for(self, role: str, history_index: int, source_index: int) -> int:
        return int(self.tiers[self.row(role, history_index, source_index)])


def _row_text(role: str, history_index: int, episode_id: str | None, slot: int | None,
              source_index: int, seed: int, key: int, tier: int) -> str:
    return (
        f"{role}|{int(history_index)}|{episode_id if episode_id is not None else ''}"
        f"|{int(slot) if slot is not None else -1}|{int(source_index)}"
        f"|{int(seed)}|{int(key)}|{int(tier)}"
    )


def allocation_digest(sources: Sequence[SourceKey], seeds: Sequence[int],
                      keys: Sequence[int], tiers: Sequence[int]) -> str:
    """Incremental canonical digest over the full frozen assignment."""
    digest = hashlib.sha256()
    digest.update((ALLOCATION_VERSION + "\n").encode("utf-8"))
    for source, seed, key, tier in zip(sources, seeds, keys, tiers):
        digest.update((_row_text(source.role, source.history_index, source.episode_id,
                                source.slot, source.source_index, seed, key, tier) + "\n").encode("utf-8"))
    return digest.hexdigest()


def allocate(
    *,
    history_ids: Sequence[str],
    history_seeds: Sequence[int],
    sources: Sequence[SourceKey],
    derive: Callable[[int, int], int] = derive_pair_seed,
    key_of: Callable[[str, int], int] = source_key,
) -> Allocation:
    """Resolve one unique generator seed for every logical source.

    ``derive`` and ``key_of`` are injectable so the closed-failure path can be
    exercised without touching the accepted generator derivation.
    """
    if len(history_ids) != len(history_seeds):
        raise _refused("REFUSED-IDENTITY", "history ids and seeds disagree")
    if not sources:
        raise _refused("REFUSED-MISSING-PREREQUISITE", "source census is empty")

    legacy_seeds: list[int] = []
    legacy_keys: list[int] = []
    for source in sources:
        if not 0 <= int(source.history_index) < len(history_seeds):
            raise _refused("REFUSED-IDENTITY", f"history index outside plan: {source.history_index}")
        seed = int(derive(int(history_seeds[int(source.history_index)]), int(source.source_index)))
        legacy_seeds.append(seed)
        legacy_keys.append(int(key_of(source.role, seed)))

    first_claim: dict[int, int] = {}
    for index, key in enumerate(legacy_keys):
        first_claim.setdefault(key, index)

    occupied: set[int] = set(legacy_keys)
    seeds: list[int] = []
    keys: list[int] = []
    tiers: list[int] = []
    probe_indices: list[int | None] = []
    for index, source in enumerate(sources):
        if first_claim[legacy_keys[index]] == index:
            seeds.append(legacy_seeds[index])
            keys.append(legacy_keys[index])
            tiers.append(0)
            probe_indices.append(None)
            continue
        history_seed = int(history_seeds[int(source.history_index)])
        for attempt in range(MAX_PROBES):
            probe_index = PROBE_INDEX_BASE + index * MAX_PROBES + attempt
            candidate_seed = int(derive(history_seed, probe_index))
            candidate_key = int(key_of(source.role, candidate_seed))
            if candidate_key in occupied:
                continue
            occupied.add(candidate_key)
            seeds.append(candidate_seed)
            keys.append(candidate_key)
            tiers.append(attempt + 1)
            probe_indices.append(probe_index)
            break
        else:
            raise _refused(
                "REFUSED-SEED-COLLISION",
                f"no free physical source after {MAX_PROBES} probes for "
                f"{source.role}/{source.history_index}/{source.source_index}",
            )

    # --- post-derivation verification (finite full census) -------------------
    if len(seeds) != len(sources):
        raise _refused("REFUSED-IDENTITY", "allocation did not cover the census")
    if len(set(keys)) != len(keys):
        raise _refused("REFUSED-SEED-COLLISION", "assigned physical source keys are not unique")
    # Seeds must be unique inside each generation domain. The normal and paired
    # constructors draw different streams for the same integer, so a cross-domain
    # integer match is not a shared waveform; it is only reported.
    paired_seeds = [seed for source, seed in zip(sources, seeds) if source.role in PAIRED_ROLES]
    normal_seeds = [seed for source, seed in zip(sources, seeds) if source.role in NORMAL_ROLES]
    unique_seeds_per_domain = (
        len(set(paired_seeds)) == len(paired_seeds) and len(set(normal_seeds)) == len(normal_seeds)
    )
    if not unique_seeds_per_domain:
        raise _refused("REFUSED-SEED-COLLISION", "assigned generator seeds repeat inside one domain")
    cross_domain_shared_seed_values = len(set(paired_seeds) & set(normal_seeds))
    for index, tier in enumerate(tiers):
        if tier == 0:
            if seeds[index] != legacy_seeds[index]:
                raise _refused("REFUSED-IDENTITY", "first claimant lost its planned seed")
            if keys[index] != legacy_keys[index]:
                raise _refused("REFUSED-IDENTITY", "first claimant lost its planned source")
        elif keys[index] == legacy_keys[index]:
            raise _refused("REFUSED-IDENTITY", "probe did not separate a duplicate source")
        elif seeds[index] == legacy_seeds[index]:
            raise _refused("REFUSED-IDENTITY", "probe reused the duplicate seed")

    duplicate_groups: dict[int, list[int]] = {}
    for index, key in enumerate(legacy_keys):
        duplicate_groups.setdefault(key, []).append(index)
    duplicate_rows = sum(len(group) - 1 for group in duplicate_groups.values() if len(group) > 1)
    if duplicate_rows != sum(1 for tier in tiers if tier > 0):
        raise _refused("REFUSED-IDENTITY", "reassignment count is not minimal")

    seed_groups: dict[int, int] = {}
    for seed in legacy_seeds:
        seed_groups[seed] = seed_groups.get(seed, 0) + 1
    seed_collision_groups = sum(1 for count in seed_groups.values() if count > 1)

    per_role: dict[str, int] = {}
    per_history: dict[str, int] = {str(name): 0 for name in history_ids}
    per_role_history: dict[str, dict[str, int]] = {role: {str(name): 0 for name in history_ids} for role in ROLES}
    for source in sources:
        per_role[source.role] = per_role.get(source.role, 0) + 1
        history_id = str(history_ids[int(source.history_index)])
        per_history[history_id] += 1
        per_role_history.setdefault(source.role, {}).setdefault(history_id, 0)
        per_role_history[source.role][history_id] += 1

    reassignments: list[dict[str, Any]] = []
    tier_histogram: dict[str, int] = {}
    for index, tier in enumerate(tiers):
        tier_histogram[str(tier)] = tier_histogram.get(str(tier), 0) + 1
        if tier == 0:
            continue
        source = sources[index]
        reassignments.append({
            "role": source.role,
            "history_id": str(history_ids[int(source.history_index)]),
            "episode_id": source.episode_id,
            "slot": source.slot,
            "source_index": int(source.source_index),
            "legacy_seed": int(legacy_seeds[index]),
            "legacy_key": int(legacy_keys[index]),
            "probe_index": int(probe_indices[index]),  # type: ignore[arg-type]
            "probe_tier": int(tier),
            "assigned_seed": int(seeds[index]),
            "assigned_key": int(keys[index]),
        })

    digest = allocation_digest(sources, seeds, keys, tiers)
    payload = {
        "allocation_version": ALLOCATION_VERSION,
        "identity_version": IDENTITY_VERSION,
        "resolver": RESOLVER,
        "digest_rule": DIGEST_RULE,
        "probe_index_base": PROBE_INDEX_BASE,
        "max_probes": MAX_PROBES,
        "census": {
            "total": len(sources),
            "per_role": per_role,
            "per_history": per_history,
            "per_role_history": per_role_history,
        },
        "legacy": {
            "distinct_seeds": len(seed_groups),
            "seed_collision_groups": int(seed_collision_groups),
            "distinct_keys": len(set(legacy_keys)),
            "key_collision_groups": sum(1 for group in duplicate_groups.values() if len(group) > 1),
            "duplicate_occurrences": int(duplicate_rows),
        },
        "assigned": {
            "distinct_seeds": len(set(seeds)),
            "unique_seeds_per_domain": bool(unique_seeds_per_domain),
            "cross_domain_shared_seed_values": int(cross_domain_shared_seed_values),
            "distinct_keys": len(set(keys)),
            "reassigned_sources": int(sum(1 for tier in tiers if tier > 0)),
            "tier_histogram": dict(sorted(tier_histogram.items())),
        },
        "digest": digest,
        "reassignments": reassignments,
    }
    return Allocation(
        version=ALLOCATION_VERSION,
        identity_version=IDENTITY_VERSION,
        history_ids=tuple(str(name) for name in history_ids),
        sources=tuple(sources),
        seeds=tuple(seeds),
        keys=tuple(keys),
        tiers=tuple(tiers),
        probe_indices=tuple(probe_indices),
        legacy_seeds=tuple(legacy_seeds),
        legacy_keys=tuple(legacy_keys),
        digest=digest,
        payload=payload,
    )


class SourceIdentityLedger:
    """Fail-closed run-local binding of generated sources to a frozen allocation.

    Every generated logical source must match the seed *and* physical-source key
    that the frozen allocation assigned to its census row; a repeated logical
    source, physical source, group/source id, file id or waveform content is
    refused before it can be written as independent evidence. Metadata ids alone
    never prove identity: the waveform bytes are hashed as well.
    """

    def __init__(self, allocation: Allocation) -> None:
        self._allocation = allocation
        self._rows: set[tuple[str, int, int]] = set()
        self._keys: set[int] = set()
        self._file_ids: set[str] = set()
        self._waveforms: set[str] = set()
        self._ids: dict[str, set[str]] = {}
        self.sources = 0
        self.files = 0

    def claim_source(
        self,
        *,
        role: str,
        history_index: int,
        source_index: int,
        seed: int,
        source_key: int,
        group_id: str | None = None,
        source_id: str | None = None,
    ) -> None:
        row = (str(role), int(history_index), int(source_index))
        if row in self._rows:
            raise _refused("REFUSED-IDENTITY", f"logical source generated twice: {row}")
        expected_seed = int(self._allocation.seed_for(*row))
        expected_key = int(self._allocation.key_for(*row))
        if int(seed) != expected_seed:
            raise _refused(
                "REFUSED-IDENTITY",
                f"generated seed {seed} differs from the frozen allocation {expected_seed} for {row}",
            )
        if int(source_key) != expected_key:
            raise _refused(
                "REFUSED-IDENTITY",
                f"generated source key {source_key} differs from the frozen allocation {expected_key} for {row}",
            )
        if int(source_key) in self._keys:
            raise _refused("REFUSED-IDENTITY", f"physical source repeated across rows: {source_key}")
        for label, value in (("group", group_id), ("source", source_id)):
            if value is None:
                continue
            seen = self._ids.setdefault(label, set())
            if str(value) in seen:
                raise _refused("REFUSED-IDENTITY", f"{label} id repeated: {value}")
            seen.add(str(value))
        self._rows.add(row)
        self._keys.add(int(source_key))
        self.sources += 1

    def claim_file(self, file_id: str, waveform: Any) -> None:
        """Register one captured file: unique id and unique waveform bytes."""
        values = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
        content = hashlib.sha256(values.tobytes()).hexdigest()
        self._claim_file_id(str(file_id))
        if content in self._waveforms:
            raise _refused("REFUSED-IDENTITY", f"waveform content repeated for file {file_id}")
        self._waveforms.add(content)
        self.files += 1

    def _claim_file_id(self, file_id: str) -> None:
        if file_id in self._file_ids:
            raise _refused("REFUSED-IDENTITY", f"file id repeated: {file_id}")
        self._file_ids.add(file_id)

    def summary(self) -> dict[str, int]:
        return {
            "sources_claimed": int(len(self._rows)),
            "physical_keys_claimed": int(len(self._keys)),
            "files_claimed": int(len(self._file_ids)),
            "unique_waveforms_claimed": int(len(self._waveforms)),
        }


def identity_payload(allocation: Allocation) -> dict[str, Any]:
    """Review-sized source-identity manifest embedded in the design/contract."""
    return {
        "identity_version": IDENTITY_VERSION,
        "logical_source": (
            "one row per (role, history, episode, slot); the twin and its variant are the same "
            "logical source, and the weekly and monthly windows of one episode reuse the same "
            "sources — they are counted once, never as independent occurrences"
        ),
        "normal_domain": {
            "roles": list(NORMAL_ROLES),
            "key": "the derived 31-bit seed itself: NormalGenerator.generate draws its whole stream from default_rng(seed)",
        },
        "paired_domain": {
            "roles": list(PAIRED_ROLES),
            "key": (
                "normal_seed = int(default_rng(pair_seed).integers(0, 2**31)), the first draw of "
                "SessionGenerator._anomaly_source; the fleet, regimes, gain/offset, causal stream, "
                "noise seed, twin and group/source ids all derive from it"
            ),
            "why_not_the_pair_seed": (
                "two distinct pair seeds with the same normal_seed generate byte-identical twins while "
                "their group_id/source_id/file ids differ; integer-seed uniqueness alone is therefore "
                "not proof of source disjointness"
            ),
        },
        "cross_domain": (
            "equal integers in the normal and paired domains do not share a clean stream, because the "
            "two constructors draw different streams; both domains are nevertheless keyed in one "
            "namespace through their own clean keys, so any real stream reuse is caught"
        ),
        "allocation": allocation.payload,
    }


__all__ = [
    "ALLOCATION_VERSION",
    "Allocation",
    "DIGEST_RULE",
    "IDENTITY_VERSION",
    "MAX_PROBES",
    "NORMAL_ROLES",
    "PAIRED_ROLES",
    "PROBE_INDEX_BASE",
    "RESOLVER",
    "ROLES",
    "SEED_SPACE",
    "SourceKey",
    "SourceIdentityLedger",
    "allocate",
    "allocation_digest",
    "identity_payload",
    "paired_clean_key",
    "source_key",
]
