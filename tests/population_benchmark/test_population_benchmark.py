"""Focused consumer-contract tests for the S24-7-M20 population benchmark.

These tests protect the frozen scientific/engineering contracts (namespaces,
nested membership, statistic identity, reference-only nuisance fits,
calibration, gates, decision branches, custody and replay). They do not run
an encoder; the real tiny smoke is an explicit CLI mode.
"""

from __future__ import annotations

import collections
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from population_benchmark import allocation, evaluation, runner, spec, storage

SEEDS = spec.ENCODER_SEEDS


# --- synthetic run fixture (no encoder) ---------------------------------------


def _items(file_ids: list[str], abnormal: list[bool], magnitude: float = 0.0) -> dict[str, np.ndarray]:
    n = len(file_ids)
    raw = np.zeros((n, 85), dtype=np.float32)
    length = np.zeros((n, 3), dtype=np.float32)
    pooled = {seed: np.zeros((n, 128), dtype=np.float32) for seed in SEEDS}
    prepool = {seed: np.zeros((n, 128), dtype=np.float32) for seed in SEEDS}
    for index, (file_id, is_abnormal) in enumerate(zip(file_ids, abnormal)):
        rng = np.random.default_rng(int(hashlib.sha256(file_id.encode()).hexdigest()[:12], 16))
        shift = magnitude if is_abnormal else 0.0
        raw[index] = rng.normal(0.0, 1.0, 85)
        raw[index, 0] += shift
        length[index] = [200.0 + 7.0 * index, float(10 + index), float(10 + index)]
        for seed in SEEDS:
            pooled[seed][index] = rng.normal(0.0, 1.0, 128)
            pooled[seed][index, 0] += shift
            prepool[seed][index] = rng.normal(0.0, 1.0, 128)
            prepool[seed][index, 0] += shift
    items: dict[str, np.ndarray] = {"raw": raw, "length": length}
    for seed in SEEDS:
        items[evaluation.readout_key("POOLED", seed)] = pooled[seed]
        items[evaluation.readout_key("PREPOOL", seed)] = prepool[seed]
    return items


def _normal_records(plan: spec.Plan, role: str, history_index: int = 0) -> list[dict[str, object]]:
    """Reference/bandwidth file rows carrying the frozen source identity."""
    frozen = spec.source_allocation(plan)
    base = (
        spec.REFERENCE_INDEX_BASE
        if role == "reference"
        else spec.REFERENCE_INDEX_BASE + plan.reference_files
    )
    count = plan.reference_files if role == "reference" else plan.bandwidth_files
    rows: list[dict[str, object]] = []
    for offset in range(count):
        source_index = base + offset
        rows.append({
            "file_id": f"{role}-{offset:03d}", "T": 64, "C": 6, "n_patches": 2, "n_valid_patches": 2,
            "source_index": source_index,
            "source_seed": frozen.seed_for(role, history_index, source_index),
            "source_key": frozen.key_for(role, history_index, source_index),
        })
    return rows


def _write_synthetic_run(root: Path, plan: spec.Plan | None = None, magnitude: float = 1.5) -> tuple[Path, spec.Plan]:
    plan = plan or spec.smoke_plan()
    frozen = spec.source_allocation(plan)
    store = storage.RunStore.create(root, fresh=True)
    design = spec.design_payload(plan)
    design["kind"] = "design"
    storage.write_json(root / "design.json", design)
    history_id = plan.history_ids[0]
    for kind, chunk_id in (("reference", "ref"), ("bandwidth", "bw")):
        records = _normal_records(plan, kind)
        ids = [str(row["file_id"]) for row in records]
        store.write_chunk(
            history_id=history_id, kind=kind, chunk_id=chunk_id,
            items=_items(ids, [False] * len(ids)),
            waveforms=[np.zeros((6, 64), dtype=np.float32)] * len(ids),
            meta={"files": records, "windows": [], "source_role": kind},
        )
    counts = spec.block_fault_counts(plan.block_files)

    def episode_chunk(episode: spec.Episode, max_level: int, kind: str) -> None:
        ranks: dict[int, int] = {}
        twin: dict[int, str] = {}
        abnormal: dict[int, str] = {}
        for block in range(episode.n_blocks):
            for rank, slot in enumerate(spec.rank_order(episode.history_id, episode.episode_id, block, plan.block_files)):
                ranks[slot] = rank
                twin[slot] = f"{episode.episode_id}-t-{slot:03d}"
                abnormal[slot] = f"{episode.episode_id}-a-{slot:03d}"
        windows = spec.episode_windows(
            episode=episode, plan=plan, admitted_ranks=ranks, twin_ids=twin,
            abnormal_ids=abnormal, max_level=max_level,
        )
        limit = (counts[0], counts[1], counts[2])[max_level]
        needed: list[tuple[str, bool, int]] = []
        for slot in sorted(ranks):
            if ranks[slot] >= counts[0]:
                needed.append((twin[slot], False, slot))
            if ranks[slot] < limit:
                needed.append((abnormal[slot], True, slot))
        ids = [file_id for file_id, _, _ in needed]
        records = []
        for index, (file_id, _, slot) in enumerate(needed):
            source_index = episode.source_index(slot)
            records.append({
                "file_id": file_id, "T": 200 + 7 * index, "C": 6,
                "n_patches": 2 + index % 3, "n_valid_patches": 2 + index % 3,
                "source_index": source_index,
                "source_seed": frozen.seed_for(episode.role, episode.history_index, source_index),
                "source_key": frozen.key_for(episode.role, episode.history_index, source_index),
            })
        store.write_chunk(
            history_id=episode.history_id, kind=kind, chunk_id=episode.episode_id,
            items=_items(ids, [flag for _, flag, _ in needed], magnitude),
            waveforms=[np.zeros((6, 64), dtype=np.float32)] * len(ids),
            meta={
                "episode_id": episode.episode_id, "cell_index": episode.cell_index,
                "family": episode.family, "severity": episode.severity,
                "history_seed": episode.history_seed, "windows": windows,
                "files": records,
            },
        )

    for episode in plan.calibration_episode_list():
        episode_chunk(episode, 0, "calibration")
    for episode in plan.test_episodes():
        episode_chunk(episode, len(spec.BURDENS) - 1, "test")
    return root, plan


@pytest.fixture()
def synthetic_run(tmp_path: Path) -> tuple[Path, spec.Plan]:
    return _write_synthetic_run(tmp_path / "run")


# --- frozen design -------------------------------------------------------------


def test_official_plan_matches_inherited_counts_and_bands() -> None:
    plan = spec.official_plan()
    spec.validate_plan(plan)
    assert spec.block_fault_counts(40) == (2, 6, 11)
    assert spec.block_fault_counts(40) == spec.EXPECTED_FAULT_COUNTS["week"]
    assert plan.fault_counts("week") == (2, 6, 11)
    assert plan.fault_counts("month") == (8, 24, 44)
    assert plan.window_size("week") == 40
    assert plan.window_size("month") == 160
    assert len(plan.test_episodes()) == 1350
    assert len(plan.calibration_episode_list()) == 300
    summary = spec.summary(plan)
    assert summary["test_episodes_per_history"] == {
        "S24PB-ITER-0001-BENCH-01": 243,
        "S24PB-ITER-0001-BENCH-02": 243,
        "S24PB-ITER-0001-BENCH-03": 216,
        "S24PB-ITER-0001-BENCH-04": 216,
        "S24PB-ITER-0001-BENCH-05": 216,
        "S24PB-ITER-0001-BENCH-06": 216,
    }
    bands = summary["source_bands"]
    assert bands["reference"] == [0, 400]
    assert bands["bandwidth"] == [400, 620]
    assert bands["calibration"][0] >= 10_000 and bands["test"][0] >= 1_000_000
    assert bands["calibration"][1] < bands["test"][0]


def test_required_variants_are_nested_and_exactly_counted() -> None:
    for block_files in (8, 40):
        counts = spec.block_fault_counts(block_files)
        for max_level in (0, 1, 2):
            twins = 0
            abnormals = 0
            for rank in range(block_files):
                twin, abnormal = spec.required_variants(rank, block_files, max_level)
                twins += int(twin)
                abnormals += int(abnormal)
                if rank < counts[max_level]:
                    assert abnormal
            assert twins == block_files - counts[0]
            assert abnormals == min(counts[max_level], block_files)
    assert spec.required_variants(0, 40, 2) == (False, True)
    assert spec.required_variants(39, 40, 2) == (True, False)
    assert spec.required_variants(5, 40, 2) == (True, True)


def test_membership_is_nested_and_hash_order_is_global() -> None:
    plan = spec.smoke_plan()
    episode = plan.test_episodes()[0]
    ranks: dict[int, int] = {}
    twin: dict[int, str] = {}
    abnormal: dict[int, str] = {}
    for block in range(episode.n_blocks):
        order = spec.rank_order(episode.history_id, episode.episode_id, block, plan.block_files)
        assert set(order) == set(episode.block_slots(block))
        for rank, slot in enumerate(order):
            ranks[slot] = rank
            twin[slot] = f"t{slot}"
            abnormal[slot] = f"a{slot}"
    windows = spec.episode_windows(
        episode=episode, plan=plan, admitted_ranks=ranks, twin_ids=twin,
        abnormal_ids=abnormal, max_level=2,
    )
    by_key = {(window["horizon"], window["burden"]): window for window in windows}
    slot_of = {file_id: slot for mapping in (twin, abnormal) for slot, file_id in mapping.items()}
    abnormal_ids = set(abnormal.values())
    for horizon, blocks in plan.horizons.items():
        per_block = [
            window["realized_faults"] // blocks
            for window in [by_key[(horizon, burden)] for burden in spec.BURDEN_LABELS]
        ]
        sets: list[set[int]] = []
        for burden_index, burden in enumerate(spec.BURDEN_LABELS):
            window = by_key[(horizon, burden)]
            assert window["coverage"] == "COVERED"
            assert window["realized_faults"] == plan.fault_counts(horizon)[burden_index]
            assert window["n_files"] == plan.window_size(horizon)
            assert len(set(window["file_ids"])) == len(window["file_ids"])
            faults = {
                slot_of[file_id] for file_id in window["file_ids"] if file_id in abnormal_ids
            }
            expected = {
                slot
                for block in range(blocks)
                for slot in spec.rank_order(
                    episode.history_id, episode.episode_id, block, plan.block_files
                )[: per_block[burden_index]]
            }
            assert faults == expected
            sets.append(faults)
        assert sets[0] <= sets[1] <= sets[2]
        rosters = [
            {slot_of[file_id] for file_id in by_key[(horizon, burden)]["file_ids"]}
            for burden in spec.BURDEN_LABELS
        ]
        assert rosters[0] == rosters[1] == rosters[2]


def test_degenerate_block_abstains_without_crashing() -> None:
    plan = spec.smoke_plan()
    episode = plan.test_episodes()[0]
    # Only slot 0 of block 0 is admitted: no burden level can be realized.
    ranks = {0: 0}
    twin = {0: "t0"}
    abnormal = {0: "a0"}
    windows = spec.episode_windows(
        episode=episode, plan=plan, admitted_ranks=ranks,
        twin_ids=twin, abnormal_ids=abnormal, max_level=2,
    )
    assert windows
    for window in windows:
        assert window["coverage"].startswith("ABSTAIN-COVERAGE-")
        assert window["coverage"] != "COVERED"


# --- physical source identity / allocation ------------------------------------


def _first_clean_key_collision(limit: int = 200_000) -> tuple[int, int, int]:
    """First pair of real 31-bit pair seeds that resolve to one clean source."""
    seen: dict[int, int] = {}
    for seed in range(int(limit)):
        key = allocation.paired_clean_key(seed)
        if key in seen:
            return seen[key], seed, key
        seen[key] = seed
    raise AssertionError("no physical-source collision in the seed scan; the identity model changed")


def test_paired_clean_key_is_the_physical_source_not_the_pair_seed() -> None:
    from synth.config import SynthConfig
    from synth.generator import SessionGenerator
    from synth.schema import AnomalyFamily

    first, second, key = _first_clean_key_collision()
    assert first != second
    assert allocation.paired_clean_key(first) == allocation.paired_clean_key(second) == key
    generator = SessionGenerator(SynthConfig())
    # The allocator's key is the generator's own first derived normal seed.
    assert int(generator._anomaly_source(first)["normal_seed"]) == key  # noqa: SLF001 - frozen identity binding
    left = generator.generate_paired_twin(first, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    right = generator.generate_paired_twin(second, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    # Distinct metadata ids hide a repeated waveform: identity cannot be metadata.
    assert left.group_id != right.group_id
    assert left.source_id != right.source_id
    assert left.twin.file_id != right.twin.file_id
    assert np.array_equal(left.twin.x, right.twin.x)


def test_allocator_separates_real_collision_inputs_and_is_deterministic() -> None:
    first, second, key = _first_clean_key_collision()
    sources = (
        allocation.SourceKey("test", 0, "E1", 0, 101),
        allocation.SourceKey("test", 0, "E2", 0, 202),
    )

    def derive(history_seed: int, pair_index: int) -> int:
        del history_seed
        return {101: first, 202: second}.get(int(pair_index), 7_000_000 + int(pair_index))

    allocated = allocation.allocate(history_ids=("H",), history_seeds=(424242,), sources=sources, derive=derive)
    assert allocated.legacy_keys == (key, key)
    assert allocated.tiers == (0, 1)
    assert allocated.seeds[1] != second
    assert len(set(allocated.keys)) == 2
    assert allocated.keys[0] == key
    again = allocation.allocate(history_ids=("H",), history_seeds=(424242,), sources=sources, derive=derive)
    assert again.digest == allocated.digest


def test_allocator_fails_closed_when_no_probe_can_separate_sources() -> None:
    sources = (
        allocation.SourceKey("test", 0, "E1", 0, 1),
        allocation.SourceKey("test", 0, "E2", 0, 2),
    )
    with pytest.raises(ValueError, match="REFUSED-SEED-COLLISION"):
        allocation.allocate(
            history_ids=("H",), history_seeds=(7,), sources=sources, derive=lambda history_seed, index: 5
        )


def test_smoke_allocation_is_deterministic_minimal_and_bound() -> None:
    plan = spec.smoke_plan()
    census = spec.source_census(plan)
    first = allocation.allocate(history_ids=plan.history_ids, history_seeds=plan.history_seeds, sources=census)
    second = allocation.allocate(history_ids=plan.history_ids, history_seeds=plan.history_seeds, sources=census)
    assert first.digest == second.digest == spec.source_allocation_digest(plan)
    assert len(set(first.keys)) == first.total
    assert first.payload["assigned"]["unique_seeds_per_domain"]
    reassigned = [index for index, tier in enumerate(first.tiers) if tier > 0]
    assert len(reassigned) == first.payload["legacy"]["duplicate_occurrences"]
    for index, (key, legacy) in enumerate(zip(first.keys, first.legacy_keys)):
        if index in reassigned:
            assert key != legacy
            assert first.seeds[index] != first.legacy_seeds[index]
        else:
            assert key == legacy
            assert first.seeds[index] == first.legacy_seeds[index]


def test_official_census_binds_every_active_role_once() -> None:
    plan = spec.official_plan()
    census = spec.source_census(plan)
    frozen = spec.source_allocation(plan)
    assert len(census) == 2400 + 1320 + 48000 + 216000
    assert len({(row.role, row.history_index, row.source_index) for row in census}) == len(census)
    # One row per episode slot: the twin/variant pair and the weekly/monthly
    # windows of one episode are one logical source, never independent rows.
    per_episode = collections.Counter(
        (row.role, row.history_index, row.episode_id) for row in census if row.episode_id is not None
    )
    assert set(per_episode.values()) == {plan.block_files * plan.max_blocks}
    legacy = frozen.payload["legacy"]
    assigned = frozen.payload["assigned"]
    assert legacy["seed_collision_groups"] == 15
    assert legacy["key_collision_groups"] == 25
    assert legacy["duplicate_occurrences"] == 25
    assert assigned["distinct_keys"] == frozen.total
    assert assigned["reassigned_sources"] == 25
    assert assigned["tier_histogram"] == {"0": 267_695, "1": 25}
    # The reference population and the bandwidth pool keep their planned seeds.
    assert all(
        tier == 0
        for tier, row in zip(frozen.tiers, census)
        if row.role in allocation.NORMAL_ROLES
    )


def test_identity_ledger_refuses_mismatched_and_repeated_sources() -> None:
    plan = spec.smoke_plan()
    frozen = spec.source_allocation(plan)
    ledger = allocation.SourceIdentityLedger(frozen)
    episode = plan.test_episodes()[0]
    first = episode.source_index(0)
    second = episode.source_index(1)
    ledger.claim_source(
        role=episode.role, history_index=episode.history_index, source_index=first,
        seed=frozen.seed_for(episode.role, episode.history_index, first),
        source_key=frozen.key_for(episode.role, episode.history_index, first),
        group_id="G1", source_id="S1",
    )
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        ledger.claim_source(
            role=episode.role, history_index=episode.history_index, source_index=first,
            seed=frozen.seed_for(episode.role, episode.history_index, first),
            source_key=frozen.key_for(episode.role, episode.history_index, first),
        )
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        ledger.claim_source(
            role=episode.role, history_index=episode.history_index, source_index=second,
            seed=frozen.seed_for(episode.role, episode.history_index, first),
            source_key=frozen.key_for(episode.role, episode.history_index, second),
        )
    ledger.claim_file("W1", np.zeros((6, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        ledger.claim_file("W9", np.zeros((6, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        ledger.claim_file("W1", np.ones((6, 8), dtype=np.float32))


def test_persisted_rows_must_match_the_frozen_allocation(tmp_path: Path) -> None:
    plan = spec.smoke_plan()
    frozen = spec.source_allocation(plan)
    store = storage.RunStore.create(tmp_path / "store", fresh=True)
    records = _normal_records(plan, "reference")
    ids = [str(row["file_id"]) for row in records]
    store.write_chunk(
        history_id=plan.history_ids[0], kind="reference", chunk_id="ref",
        items=_items(ids, [False] * len(ids)),
        waveforms=[np.zeros((6, 64), dtype=np.float32)] * len(ids),
        meta={"files": records, "windows": []},
    )
    assert store.verify_source_identity(frozen)["pass"]

    sidecar = tmp_path / "store/chunks" / plan.history_ids[0] / "reference" / "ref.json"
    tampered = json.loads(sidecar.read_text())
    tampered["files"][0]["source_seed"] = int(frozen.seed_for("reference", 0, spec.REFERENCE_INDEX_BASE)) + 1
    sidecar.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        store.verify_source_identity(frozen)

    anonymous = storage.RunStore.create(tmp_path / "anonymous", fresh=True)
    anonymous.write_chunk(
        history_id=plan.history_ids[0], kind="reference", chunk_id="ref",
        items={"raw": np.zeros((1, 3), dtype=np.float32)},
        waveforms=[np.zeros((6, 8), dtype=np.float32)],
        meta={"files": [{"file_id": "X"}], "windows": []},
    )
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        anonymous.verify_source_identity(frozen)


# --- statistic / calibration ---------------------------------------------------


def test_mmd_matches_direct_unbiased_formula() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 5))
    y = rng.normal(size=(20, 5))
    sigma = 1.7

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-d2 / (2 * sigma * sigma))

    n, m = x.shape[0], y.shape[0]
    direct = (
        2.0 * kernel(x, y).mean()
        - (kernel(x, x).sum() - n) / (n * (n - 1))
        - (kernel(y, y).sum() - m) / (m * (m - 1))
    )
    # Identity standardizer + the same kernel bandwidth isolates the estimator.
    identity = evaluation.ReferenceSpace(
        center=np.zeros(y.shape[1]), scale=np.ones(y.shape[1]), sigma=sigma,
        reference=y, reference_kernel_sum=float(kernel(y, y).sum()),
        reference_sq=(y * y).sum(axis=1),
    )
    assert evaluation.mmd2_unbiased(x, identity) == pytest.approx(float(direct), rel=1e-12, abs=1e-12)


def test_threshold_is_an_order_statistic_with_tail_interval() -> None:
    values = [float(value) for value in range(300)]
    threshold = evaluation.null_threshold(values, 0.975)
    assert threshold["rank_0based"] == 292
    assert threshold["threshold"] == 292.0
    assert threshold["n"] == 300
    lower, upper = threshold["exceedance_interval_95"]
    assert 0.0 < lower <= upper < 1.0


def test_nuisance_fits_use_reference_only(synthetic_run: tuple[Path, spec.Plan]) -> None:
    root, plan = synthetic_run
    store = storage.RunStore(root=root)
    chunks = store.load_chunks()
    grouped = {chunk.history_id: chunk for chunk in chunks if chunk.kind == "reference"}
    reference = grouped[plan.history_ids[0]]
    bandwidth = next(chunk for chunk in chunks if chunk.kind == "bandwidth")
    test_chunks = [chunk for chunk in chunks if chunk.kind == "test"]
    space = evaluation.build_history_space(
        plan.history_ids[0], reference, bandwidth, [chunk for chunk in chunks if chunk.kind in ("calibration", "test")]
    )
    baseline = {key: (value.sigma, value.center.copy(), value.scale.copy()) for key, value in space.spaces.items()}
    # A window item change must not move the frozen transforms.
    mutated = [chunk for chunk in test_chunks]
    for chunk in mutated:
        for key in list(chunk.items):
            chunk.items[key] = chunk.items[key] + 10.0
    space2 = evaluation.build_history_space(
        plan.history_ids[0], reference, bandwidth, [chunk for chunk in chunks if chunk.kind in ("calibration", "test")]
    )
    for key, value in baseline.items():
        assert space2.spaces[key].sigma == value[0]
        assert np.array_equal(space2.spaces[key].center, value[1])
    # A reference change does move the transform.
    reference.items["raw"] = reference.items["raw"] + 1.0
    space3 = evaluation.build_history_space(
        plan.history_ids[0], reference, bandwidth, [chunk for chunk in chunks if chunk.kind in ("calibration", "test")]
    )
    assert not np.array_equal(space3.spaces["raw"].center, baseline["raw"][1])


def test_no_pseudo_n_and_equal_file_weight(synthetic_run: tuple[Path, spec.Plan]) -> None:
    root, plan = synthetic_run
    store = storage.RunStore(root=root)
    forecast = spec.forecast(plan)
    windows = len(plan.test_episodes())
    assert forecast["intervals_per_readout"] == windows * len(plan.horizons)
    assert forecast["files"]["total"] == (
        len(plan.test_episodes()) * 20
        + len(plan.calibration_episode_list()) * 16
        + plan.reference_files * len(plan.history_ids)
        + plan.bandwidth_files * len(plan.history_ids)
    )
    for chunk in store.load_chunks(kinds=("test",)):
        if not chunk.windows:
            continue
        rows = {key: value.shape[0] for key, value in chunk.items.items()}
        assert set(rows.values()) == {len(chunk.meta["files"])}
        ids = [record["file_id"] for record in chunk.meta["files"]]
        assert len(set(ids)) == len(ids)
        for window in chunk.windows:
            assert len(set(window["file_ids"])) == len(window["file_ids"])
            assert len(window["file_ids"]) <= len(ids)


# --- gates / decision ----------------------------------------------------------


def test_decision_table_covers_every_branch() -> None:
    fixtures = {
        "keep_pooled": ({"POOLED": spec.READOUT_QUALIFIES, "PREPOOL": spec.READOUT_QUALIFIES, "RAW": spec.READOUT_FAILS}, True, spec.ACTION_KEEP_POOLED),
        "change_readout": ({"POOLED": spec.READOUT_FAILS, "PREPOOL": spec.READOUT_QUALIFIES, "RAW": spec.READOUT_FAILS}, True, spec.ACTION_CHANGE_READOUT),
        "target_representation": ({"POOLED": spec.READOUT_FAILS, "PREPOOL": spec.READOUT_FAILS, "RAW": spec.READOUT_QUALIFIES}, True, spec.ACTION_TARGET_REPRESENTATION),
        "stop_scale": ({"POOLED": spec.READOUT_FAILS, "PREPOOL": spec.READOUT_FAILS, "RAW": spec.READOUT_FAILS}, True, spec.ACTION_STOP_SCALE),
        "invalid_coverage": ({"POOLED": spec.READOUT_QUALIFIES, "PREPOOL": spec.READOUT_QUALIFIES, "RAW": spec.READOUT_QUALIFIES}, False, spec.ACTION_INVALID_IMPLEMENTATION),
        "invalid_readout": ({"POOLED": spec.READOUT_INVALID, "PREPOOL": spec.READOUT_QUALIFIES, "RAW": spec.READOUT_QUALIFIES}, True, spec.ACTION_INVALID_IMPLEMENTATION),
    }
    for name, (statuses, coverage, action) in fixtures.items():
        result = evaluation.decision_from_fixture(statuses, coverage_ok=coverage)
        assert result["action"] == action, name
    keep = evaluation.decision_from_fixture(fixtures["keep_pooled"][0])
    assert keep["learned_route_valid"] and not keep["raw_qualifies"]


_DURATION_CELLS = tuple(
    index
    for index in range(len(spec.FAMILIES) * len(spec.SEVERITIES))
    if spec.FAMILIES[index // len(spec.SEVERITIES)] == spec.DURATION_FAMILY
)


def _gate_records(
    plan: spec.Plan,
    duration_true_counts: Mapping[int, Sequence[int]] | None = None,
) -> list[dict[str, object]]:
    """Real per-window alarms at exactly the coverage floor in every cell.

    Gated burdens are unanimous except the week/15% duration family, where
    ``duration_true_counts[seed]`` gives the alarming windows in each of its
    three severity cells. Every combined alarm remains the per-window majority
    of the three encoder alarms.
    """
    durations = duration_true_counts or {}
    records: list[dict[str, object]] = []
    for horizon in plan.horizons:
        for burden in spec.BURDEN_LABELS:
            for cell in range(plan.cell_count):
                severity_index = cell % len(spec.SEVERITIES)
                for window in range(spec.MIN_REALIZED_EPISODES):
                    per_seed: dict[str, bool] = {}
                    for seed in SEEDS:
                        if burden == spec.BURDEN_LABELS[0]:
                            alarm = False
                        elif burden == spec.BURDEN_LABELS[2] or horizon != "week" or cell not in _DURATION_CELLS:
                            alarm = True
                        else:
                            counts = durations.get(seed, (45, 45, 45))
                            alarm = window < int(counts[severity_index])
                        per_seed[str(seed)] = alarm
                    combined = sum(per_seed.values()) >= 2
                    records.append({
                        "window_id": f"fixture:{horizon}:{burden}:{cell}:{window}",
                        "horizon": horizon,
                        "burden": burden,
                        "cell_index": cell,
                        "family": spec.FAMILIES[cell // len(spec.SEVERITIES)],
                        "severity": spec.SEVERITIES[severity_index],
                        "coverage": "COVERED",
                        "combined": {readout: combined for readout in spec.READOUTS},
                        "alarms": {readout: dict(per_seed) for readout in spec.READOUTS},
                        "stats": {},
                    })
    return records


def test_per_encoder_family_floor_blocks_qualification() -> None:
    """Two encoders below one family floor cannot meet 2-of-3 robustness."""
    plan = spec.official_plan()
    records = _gate_records(plan, {SEEDS[1]: (45, 0, 0), SEEDS[2]: (0, 45, 0)})
    aggregate = evaluation.aggregate_test_records("POOLED", records, plan)
    assert aggregate["coverage_ok"] is True
    assert all(item["pass"] for item in aggregate["checks"])
    duration_family = aggregate["family_rates"]["week"][spec.BURDEN_LABELS[1]][spec.DURATION_FAMILY]
    assert duration_family["rate"] == pytest.approx(2.0 / 3.0)
    assert duration_family["cells"] == 3
    assert aggregate["encoder_pass"] == {str(SEEDS[0]): True, str(SEEDS[1]): False, str(SEEDS[2]): False}
    assert aggregate["encoder_robustness_ok"] is False
    assert aggregate["qualifies"] is False
    for seed in SEEDS[1:]:
        checks = aggregate["encoder_checks"][str(seed)]
        failed = [item["check"] for item in checks if not item["pass"]]
        assert failed == [f"family:week:{spec.BURDEN_LABELS[1]}:{spec.DURATION_FAMILY}"]
        family_check = next(item for item in checks if item["check"] == failed[0])
        assert family_check["value"] == pytest.approx(1.0 / 3.0)
    aggregates = {readout: evaluation.aggregate_test_records(readout, records, plan) for readout in spec.READOUTS}
    decision = evaluation.decide(aggregates, coverage_ok=True)
    assert decision["statuses"] == {readout: spec.READOUT_FAILS for readout in spec.READOUTS}
    assert decision["action"] == spec.ACTION_STOP_SCALE


def test_two_of_three_encoders_with_family_floor_still_qualify() -> None:
    """Exactly two encoders clearing every gate still satisfy robustness."""
    plan = spec.official_plan()
    records = _gate_records(plan, {SEEDS[2]: (45, 35, 0)})
    aggregate = evaluation.aggregate_test_records("POOLED", records, plan)
    assert aggregate["encoder_pass"] == {str(SEEDS[0]): True, str(SEEDS[1]): True, str(SEEDS[2]): False}
    assert aggregate["encoder_robustness_ok"] is True
    assert aggregate["qualifies"] is True
    checks = aggregate["encoder_checks"][str(SEEDS[2])]
    failing = [item for item in checks if not item["pass"]]
    assert [item["check"] for item in failing] == [f"family:week:{spec.BURDEN_LABELS[1]}:{spec.DURATION_FAMILY}"]
    assert failing[0]["value"] == pytest.approx((1.0 + 35.0 / 45.0) / 3.0)
    core = [item for item in checks if item["check"].split(":")[0] in ("fpr", "detection", "non_duration")]
    assert core and all(item["pass"] for item in core)
    combined_cell = aggregate["cell_rates"]["week"][spec.BURDEN_LABELS[1]][str(_DURATION_CELLS[1])]
    assert combined_cell["trials"] == 45 and combined_cell["successes"] == 45


def test_encoder_family_floor_boundary_is_inclusive() -> None:
    """A family rate of exactly 0.60 passes the frozen ``>=`` floor."""
    plan = spec.official_plan()
    records = _gate_records(plan, {SEEDS[1]: (45, 36, 0)})
    aggregate = evaluation.aggregate_test_records("POOLED", records, plan)
    family_check = next(
        item for item in aggregate["encoder_checks"][str(SEEDS[1])]
        if item["check"] == f"family:week:{spec.BURDEN_LABELS[1]}:{spec.DURATION_FAMILY}"
    )
    assert family_check["value"] == pytest.approx(0.6)
    assert family_check["pass"] is True
    assert aggregate["encoder_pass"][str(SEEDS[1])] is True
    assert aggregate["qualifies"] is True


def test_abstention_is_excluded_from_the_alarm_denominator() -> None:
    plan = spec.smoke_plan()
    covered = [
        {
            "window_id": f"c{index}", "horizon": "week", "burden": spec.BURDEN_LABELS[1],
            "cell_index": 0, "family": spec.FAMILIES[0], "severity": spec.SEVERITIES[0],
            "coverage": "COVERED",
            "combined": {readout: False for readout in spec.READOUTS},
            "alarms": {readout: {str(seed): False for seed in SEEDS} for readout in spec.READOUTS},
            "stats": {},
        }
        for index in range(2)
    ]
    abstained = {
        "window_id": "a0", "horizon": "week", "burden": spec.BURDEN_LABELS[1],
        "cell_index": 0, "family": spec.FAMILIES[0], "severity": spec.SEVERITIES[0],
        "coverage": "ABSTAIN-COVERAGE-15",
        "combined": {readout: True for readout in spec.READOUTS},
        "alarms": {readout: {str(seed): True for seed in SEEDS} for readout in spec.READOUTS},
        "stats": {},
    }
    aggregate = evaluation.aggregate_test_records("POOLED", covered + [abstained], plan)
    rate = aggregate["cell_rates"]["week"][spec.BURDEN_LABELS[1]]["0"]
    assert rate["trials"] == 2
    assert rate["successes"] == 0
    assert aggregate["coverage"]["week"][spec.BURDEN_LABELS[1]]["0"] == 2


# --- custody / replay ----------------------------------------------------------


def test_chunk_roundtrip_and_hash_failure(tmp_path: Path) -> None:
    store = storage.RunStore.create(tmp_path / "store", fresh=True)
    items = {"raw": np.zeros((2, 3), dtype=np.float32)}
    record = store.write_chunk(
        history_id="H", kind="reference", chunk_id="ref",
        items=items, waveforms=[np.zeros((6, 8), dtype=np.float32)] * 2,
        meta={"files": [{"file_id": "F1"}, {"file_id": "F2"}], "windows": []},
    )
    chunk = store.read_chunk("H", "reference", "ref")
    assert chunk.items["raw"].shape == (2, 3)
    assert store.verify()["pass"]
    target = tmp_path / "store/chunks/H/reference/ref.items.npz"
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0xFF
    target.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="REFUSED-HASH-MISMATCH"):
        store.read_chunk("H", "reference", "ref")
    assert record["sha256"]["items"] != storage.sha256_file(target)


def test_replay_matches_and_detects_tamper(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_synthetic_run(root)
    result = runner.evaluate_run(root)
    assert result["decision"]["action"] in spec.ACTIONS
    replay = runner.replay_run(root, root / "replay")
    assert replay["status"] == "REPLAY-MATCH"
    assert replay["encoder_calls"] == 0
    assert not replay["torch_imported"]
    # Tamper with the persisted statistics: the recomputation must disagree.
    statistics = json.loads((root / "statistics.json").read_text())
    statistics["aggregates"]["POOLED"]["macro"]["week"][spec.BURDEN_LABELS[0]]["rate"] = 0.123456
    (root / "statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
    with pytest.raises(ValueError, match="REFUSED-REPLAY-MISMATCH"):
        runner.replay_run(root, root / "replay-tampered")


def test_contract_verifies_and_refuses_drift() -> None:
    payload = spec.contract_payload()
    payload["contract_sha256"] = spec.contract_sha256()
    assert spec.verify_contract(payload)["status"] == "CONTRACT-VERIFIED"
    payload["gates"]["fpr_max"] = 0.5
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        spec.verify_contract(payload)


def test_replay_import_guard_blocks_encoder_modules() -> None:
    import importlib.util

    path = Path(runner.CLI)
    module_spec = importlib.util.spec_from_file_location("m20_cli_guard", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    guard = module._ReplayImportGuard()
    with pytest.raises(ImportError):
        guard.find_spec("torch")
    with pytest.raises(ImportError):
        guard.find_spec("representation.sprint24_features")
    assert guard.find_spec("numpy") is None
