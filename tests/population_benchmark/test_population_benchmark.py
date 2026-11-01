"""Focused consumer-contract tests for the S24-7-M20 population benchmark.

These tests protect the frozen scientific/engineering contracts (namespaces,
nested membership, statistic identity, reference-only nuisance fits,
calibration, gates, decision branches, custody and replay). They do not run
an encoder; the real tiny smoke is an explicit CLI mode.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from population_benchmark import evaluation, runner, spec, storage

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


def _files(count: int, prefix: str) -> list[dict[str, object]]:
    return [
        {"file_id": f"{prefix}-{index:03d}", "T": 64, "C": 6, "n_patches": 2, "n_valid_patches": 2}
        for index in range(count)
    ]


def _write_synthetic_run(root: Path, plan: spec.Plan | None = None, magnitude: float = 1.5) -> tuple[Path, spec.Plan]:
    plan = plan or spec.smoke_plan()
    store = storage.RunStore.create(root, fresh=True)
    design = spec.design_payload(plan)
    design["kind"] = "design"
    storage.write_json(root / "design.json", design)
    history_id = plan.history_ids[0]
    for kind, chunk_id, count in (
        ("reference", "ref", plan.reference_files),
        ("bandwidth", "bw", plan.bandwidth_files),
    ):
        ids = [f"{kind}-{index:03d}" for index in range(count)]
        store.write_chunk(
            history_id=history_id, kind=kind, chunk_id=chunk_id,
            items=_items(ids, [False] * count),
            waveforms=[np.zeros((6, 64), dtype=np.float32)] * count,
            meta={"files": _files(count, kind), "windows": []},
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
        needed: list[tuple[str, bool]] = []
        for slot in sorted(ranks):
            if ranks[slot] >= counts[0]:
                needed.append((twin[slot], False))
            if ranks[slot] < limit:
                needed.append((abnormal[slot], True))
        ids = [file_id for file_id, _ in needed]
        records = [
            {
                "file_id": file_id, "T": 200 + 7 * index, "C": 6,
                "n_patches": 2 + index % 3, "n_valid_patches": 2 + index % 3,
            }
            for index, file_id in enumerate(ids)
        ]
        store.write_chunk(
            history_id=episode.history_id, kind=kind, chunk_id=episode.episode_id,
            items=_items(ids, [flag for _, flag in needed], magnitude),
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
