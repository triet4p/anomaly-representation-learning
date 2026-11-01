"""Reusable public runner for the S24-7-M20 population benchmark.

Modes (see ``experiments/sprint24_task7m20_population_benchmark.py``):

- design / contract: deterministic manifests, no model;
- smoke: real tiny generation + the three accepted frozen bundles through all
  readouts, durable disk custody, fresh-process replay and the decision table;
- official: the full frozen plan, gated by an explicit release manifest;
- replay: encoder-free recomputation from persisted items.

The official run and replay share ``evaluate_run``/``replay_run``; the official
run never reads its own in-memory statistics to produce the decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from population_benchmark import evaluation, spec, storage

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "experiments" / "sprint24_task7m20_population_benchmark.py"
DOCS_CONTRACT = REPO / "docs" / "sprint24-population-benchmark-contract-a01.json"

#: Provisional cumulative completed forwards before M20 (M18/PROVISIONAL record).
CUMULATIVE_FORWARDS_BEFORE = 1_013_581

#: Frozen sources whose working-tree hashes bind a run (no silent edits).
SOURCE_PATHS: tuple[str, ...] = (
    "src/population_benchmark/spec.py",
    "src/population_benchmark/evaluation.py",
    "src/population_benchmark/storage.py",
    "src/population_benchmark/capture.py",
    "src/population_benchmark/runner.py",
    "experiments/sprint24_task7m20_population_benchmark.py",
    "docs/sprint24-population-benchmark-contract-a01.json",
)


def _refused(reason: str, detail: str) -> ValueError:
    return ValueError(f"{reason}: {detail}")


def source_commit(repo: Path = REPO) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def source_hashes(repo: Path = REPO) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = Path(repo) / relative
        if path.is_file():
            output[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return output


def _plan_for(label: str) -> spec.Plan:
    if label == "official":
        return spec.official_plan()
    if label == "smoke":
        return spec.smoke_plan()
    raise _refused("REFUSED-UNKNOWN-DESIGN", f"unknown plan label {label!r}")


def _plan_from_design(design: Mapping[str, Any]) -> spec.Plan:
    plan = _plan_for(str(design["plan"]["label"]))
    recorded = spec.canonical_json(design["plan"])
    recomputed = spec.canonical_json(spec.summary(plan))
    if recorded != recomputed:
        raise _refused("REFUSED-IDENTITY", "design plan summary differs from the frozen plan")
    return plan


# --- manifests -----------------------------------------------------------------


def write_design(root: Path, plan: spec.Plan) -> dict[str, Any]:
    payload = spec.design_payload(plan)
    payload["kind"] = "design"
    payload["source_commit"] = source_commit()
    payload["source_hashes"] = source_hashes()
    path = Path(root) / "design.json"
    sha = storage.write_json(path, payload)
    return {"path": str(path), "sha256": sha, "plan": spec.summary(plan)}


def write_contract(root: Path) -> dict[str, Any]:
    body = spec.contract_payload()
    body["contract_sha256"] = spec.contract_sha256()
    path = Path(root) / "contract.json"
    sha = storage.write_json(path, body)
    storage.write_json(DOCS_CONTRACT, body)
    return {"path": str(path), "sha256": sha, "contract_sha256": body["contract_sha256"]}


def verify_contract_file(path: Path) -> dict[str, Any]:
    payload = storage.read_json(Path(path))
    return spec.verify_contract(payload)


# --- capture orchestration ------------------------------------------------------


def _merge_records(captured: Sequence[Mapping[str, Any]], planned: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(item["file_id"]): item for item in planned}
    merged: list[dict[str, Any]] = []
    for record in captured:
        extra = by_id.get(str(record["file_id"]), {})
        merged.append({
            **dict(record),
            **{key: extra[key] for key in ("slot", "block", "source_seed", "rank") if key in extra},
        })
    return merged


def _normal_samples(generator: Any, history_seed: int, indices: Sequence[int]) -> tuple[list[Any], list[dict[str, Any]]]:
    from synth.generator import derive_pair_seed
    from population_benchmark import capture as capture_module

    samples: list[Any] = []
    rejected: list[dict[str, Any]] = []
    config_hash = str(generator.cfg.hash())
    for index in indices:
        seed = int(derive_pair_seed(int(history_seed), int(index)))
        sample = generator.generate_normal(seed)
        reason = capture_module.admit_file(sample, config_hash)
        if reason != "ADMITTED":
            rejected.append({"index": int(index), "seed": int(seed), "reason": reason})
            continue
        samples.append(sample)
    return samples, rejected


def _latent_dim(bundles: Mapping[int, Any]) -> int:
    dims = {int(bundle.model.config.d_model) for bundle in bundles.values()}
    if len(dims) != 1:
        raise _refused("REFUSED-IDENTITY", "encoder latent widths disagree")
    return int(dims.pop())


def _bandwidth_maps(bundles: Mapping[int, Any], latents: Mapping[int, list[tuple[Any, Any]]]) -> dict[int, Any]:
    from population_benchmark import capture as capture_module

    dim = _latent_dim(bundles)
    maps: dict[int, Any] = {}
    for seed in spec.ENCODER_SEEDS:
        sigma = capture_module.fit_prepool_sigma(latents[int(seed)])
        maps[int(seed)] = capture_module.RFFMap.create(dim, sigma)
    return maps


def _maps_from_meta(meta: Mapping[str, Any], bundles: Mapping[int, Any]) -> dict[int, Any]:
    from population_benchmark import capture as capture_module

    recorded = meta.get("prepool", {})
    maps: dict[int, Any] = {}
    for seed in spec.ENCODER_SEEDS:
        entry = recorded[str(seed)]
        maps[int(seed)] = capture_module.RFFMap.create(
            int(entry["latent_dim"]), float(entry["sigma"]), int(entry["dim"]), int(entry["seed"])
        )
    return maps


def _maps_meta(maps: Mapping[int, Any]) -> dict[str, Any]:
    return {str(seed): value.spec() for seed, value in maps.items()}


def _capture_normals(
    *,
    store: storage.RunStore,
    bundles: Mapping[int, Any],
    generator: Any,
    plan: spec.Plan,
    history_id: str,
    history_seed: int,
    kind: str,
    chunk_id: str,
    indices: Sequence[int],
    maps: Mapping[int, Any] | None,
    device: str,
    counters: dict[str, int],
    fit_bandwidth: bool,
) -> dict[str, Any]:
    from population_benchmark import capture as capture_module

    samples, rejected = _normal_samples(generator, history_seed, indices)
    captured = capture_module.capture_samples(
        bundles=bundles,
        samples=samples,
        history_id=history_id,
        prepool_maps=maps,
        keep_latents=fit_bandwidth,
        device=device,
    )
    counters["forwards"] += len(samples) * len(spec.ENCODER_SEEDS)
    counters["files"] += len(samples)
    items = dict(captured.items)
    if fit_bandwidth:
        maps = _bandwidth_maps(bundles, captured.latents)
        for seed in spec.ENCODER_SEEDS:
            items[evaluation.readout_key("PREPOOL", seed)] = capture_module.build_prepool_items(
                captured.latents[int(seed)], {int(seed): maps[int(seed)]}
            )[int(seed)]
    meta = {
        "episode_id": None,
        "cell_index": None,
        "family": None,
        "severity": None,
        "history_seed": int(history_seed),
        "files": list(captured.file_records),
        "windows": [],
        "requested": {},
        "realized": {"admitted": len(samples), "planned": len(indices)},
        "admission": {"admitted": len(samples), "planned": len(indices), "rejected": rejected},
        "prepool": _maps_meta(maps or {}),
    }
    record = store.write_chunk(
        history_id=history_id,
        kind=kind,
        chunk_id=chunk_id,
        items=items,
        waveforms=[sample.x for sample in samples],
        meta=meta,
    )
    counters["chunks"] += 1
    store.append_event("chunk_written", {
        "kind": kind, "chunk_id": chunk_id, "history_id": history_id,
        "files": len(samples), "forwards": len(samples) * len(spec.ENCODER_SEEDS),
        "items_sha256": record["sha256"]["items"],
    })
    return record


def _capture_episode(
    *,
    store: storage.RunStore,
    bundles: Mapping[int, Any],
    generator: Any,
    plan: spec.Plan,
    episode: spec.Episode,
    max_level: int,
    maps: Mapping[int, Any],
    device: str,
    kind: str,
    counters: dict[str, int],
) -> dict[str, Any]:
    from population_benchmark import capture as capture_module

    build = capture_module.build_episode(
        generator=generator, episode=episode, plan=plan, max_level=max_level
    )
    if not build.samples:
        raise _refused("REFUSED-ERROR-BOUNDARY", f"episode {episode.episode_id} has no admitted files")
    captured = capture_module.capture_samples(
        bundles=bundles,
        samples=build.samples,
        history_id=episode.history_id,
        prepool_maps=maps,
        keep_latents=False,
        device=device,
    )
    counters["forwards"] += len(build.samples) * len(spec.ENCODER_SEEDS)
    counters["files"] += len(build.samples)
    admission_counts: dict[str, int] = {}
    for source in build.sources:
        admission_counts[source.reason] = admission_counts.get(source.reason, 0) + 1
    meta = {
        "episode_id": episode.episode_id,
        "cell_index": int(episode.cell_index),
        "family": str(episode.family),
        "severity": float(episode.severity),
        "history_seed": int(episode.history_seed),
        "files": _merge_records(captured.file_records, build.file_records),
        "sources": [
            {
                "slot": source.slot, "block": source.block, "source_index": source.source_index,
                "source_seed": source.source_seed, "disposition": source.disposition,
                "admitted": source.admitted, "reason": source.reason,
                "strength": source.strength, "n_attempts": source.n_attempts,
                "twin_file_id": source.twin_file_id, "abnormal_file_id": source.abnormal_file_id,
                "rank": source.rank, "robot_idx": source.robot_idx,
                "program_idx": source.program_idx,
                "normal_path_diverges": source.normal_path_diverges,
                "config_hash": source.config_hash, "generator_version": source.generator_version,
                "attempts": list(source.attempts),
            }
            for source in build.sources
        ],
        "windows": build.windows,
        "requested": {
            "sources": len(build.sources),
            "faults": {
                horizon: list(spec.window_fault_counts(horizon, plan.block_files))
                for horizon in plan.horizons
            },
        },
        "realized": {
            "admitted_sources": sum(1 for source in build.sources if source.admitted),
            "files": len(build.samples),
        },
        "admission": {"counts": admission_counts, "planned_sources": len(build.sources)},
        "prepool": _maps_meta(maps),
    }
    record = store.write_chunk(
        history_id=episode.history_id,
        kind=kind,
        chunk_id=episode.episode_id,
        items=captured.items,
        waveforms=[sample.x for sample in build.samples],
        meta=meta,
    )
    counters["chunks"] += 1
    store.append_event("chunk_written", {
        "kind": kind, "chunk_id": episode.episode_id, "history_id": episode.history_id,
        "files": len(build.samples), "forwards": len(build.samples) * len(spec.ENCODER_SEEDS),
        "coverage": {window["window_id"]: window["coverage"] for window in build.windows},
        "items_sha256": record["sha256"]["items"],
    })
    return record


def _verify_or_none(store: storage.RunStore, history_id: str, kind: str, chunk_id: str) -> dict[str, Any] | None:
    if not store.chunk_exists(history_id, kind, chunk_id):
        return None
    chunk = store.read_chunk(history_id, kind, chunk_id)
    return chunk.meta


def run_plan_capture(
    *,
    store: storage.RunStore,
    plan: spec.Plan,
    bundles: Mapping[int, Any],
    device: str,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute (or resume) every frozen chunk of a plan."""
    from synth.config import SynthConfig
    from synth.generator import SessionGenerator

    generator = SessionGenerator(SynthConfig())
    counters = {"forwards": 0, "model_loads": len(bundles), "files": 0, "chunks": 0, "resumed": 0}
    maps_by_history: dict[str, dict[int, Any]] = {}
    bandwidth_base = spec.REFERENCE_INDEX_BASE + plan.reference_files
    for history_id, history_seed in zip(plan.history_ids, plan.history_seeds):
        # 1) independent bandwidth pool -> RFF sigma only (no item-space bandwidth yet).
        existing = _verify_or_none(store, history_id, "bandwidth", "bw") if resume else None
        if existing is not None:
            maps = _maps_from_meta(existing, bundles)
            counters["resumed"] += 1
        else:
            _capture_normals(
                store=store, bundles=bundles, generator=generator, plan=plan,
                history_id=history_id, history_seed=history_seed, kind="bandwidth", chunk_id="bw",
                indices=range(bandwidth_base, bandwidth_base + plan.bandwidth_files),
                maps=None, device=device, counters=counters, fit_bandwidth=True,
            )
            meta = store.read_meta(history_id, "bandwidth", "bw")
            maps = _maps_from_meta(meta, bundles)
        maps_by_history[history_id] = maps
        # 2) reference population in the same frozen latent/kernel transform.
        existing = _verify_or_none(store, history_id, "reference", "ref") if resume else None
        if existing is not None:
            counters["resumed"] += 1
        else:
            _capture_normals(
                store=store, bundles=bundles, generator=generator, plan=plan,
                history_id=history_id, history_seed=history_seed, kind="reference", chunk_id="ref",
                indices=range(spec.REFERENCE_INDEX_BASE, spec.REFERENCE_INDEX_BASE + plan.reference_files),
                maps=maps, device=device, counters=counters, fit_bandwidth=False,
            )
    for episode in plan.calibration_episode_list():
        if resume and _verify_or_none(store, episode.history_id, "calibration", episode.episode_id) is not None:
            counters["resumed"] += 1
            continue
        _capture_episode(
            store=store, bundles=bundles, generator=generator, plan=plan, episode=episode,
            max_level=0, maps=maps_by_history[episode.history_id], device=device,
            kind="calibration", counters=counters,
        )
    for episode in plan.test_episodes():
        if resume and _verify_or_none(store, episode.history_id, "test", episode.episode_id) is not None:
            counters["resumed"] += 1
            continue
        _capture_episode(
            store=store, bundles=bundles, generator=generator, plan=plan, episode=episode,
            max_level=len(spec.BURDENS) - 1, maps=maps_by_history[episode.history_id],
            device=device, kind="test", counters=counters,
        )
    return counters


# --- evaluation and replay ------------------------------------------------------


def evaluate_run(root: Path) -> dict[str, Any]:
    """Recompute everything from persisted items; write statistics and decision."""
    root = Path(root)
    design = storage.read_json(root / "design.json")
    plan = _plan_from_design(design)
    store = storage.RunStore(root=root)
    chunks = store.load_chunks()
    results = evaluation.evaluate_chunks(chunks, plan)
    aggregates = results["aggregates"]
    coverage_ok = all(aggregates[readout]["coverage_ok"] for readout in spec.READOUTS)
    decision = evaluation.decide(aggregates, coverage_ok=coverage_ok)
    decision["plan_label"] = plan.label
    decision["evidence_class"] = (
        "OFFICIAL-BENCHMARK" if plan.label == "official" else "SMOKE-NOT-PERFORMANCE-EVIDENCE"
    )
    decision["plan_digest"] = spec.enumeration_digest(plan)
    statistics = {
        "plan_label": plan.label,
        "plan_digest": spec.enumeration_digest(plan),
        "coverage_ok": coverage_ok,
        **results,
    }
    storage.write_json(root / "statistics.json", statistics)
    storage.write_json(root / "decision.json", decision)
    return {"statistics": statistics, "decision": decision}


def _compare(actual: Any, expected: Any, path: str = "", tol: float = 1e-12) -> list[str]:
    problems: list[str] = []
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        keys = set(expected) | set(actual)
        for key in sorted(keys):
            if key not in expected:
                problems.append(f"{path}.{key}: unexpected")
            elif key not in actual:
                problems.append(f"{path}.{key}: missing")
            else:
                problems.extend(_compare(actual[key], expected[key], f"{path}.{key}", tol))
        return problems
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)) and isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)):
        if len(expected) != len(actual):
            return [f"{path}: length {len(actual)} != {len(expected)}"]
        for index, (left, right) in enumerate(zip(actual, expected)):
            problems.extend(_compare(left, right, f"{path}[{index}]", tol))
        return problems
    if isinstance(expected, bool) or isinstance(actual, bool):
        if bool(actual) != bool(expected):
            problems.append(f"{path}: {actual!r} != {expected!r}")
        return problems
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol):
            problems.append(f"{path}: {actual!r} != {expected!r}")
        return problems
    if actual != expected:
        problems.append(f"{path}: {actual!r} != {expected!r}")
    return problems


def replay_run(root: Path, out: Path | None = None) -> dict[str, Any]:
    """Encoder-free recomputation and comparison against the persisted results."""
    root = Path(root)
    destination = Path(out) if out is not None else root
    destination.mkdir(parents=True, exist_ok=True)
    design = storage.read_json(root / "design.json")
    plan = _plan_from_design(design)
    checks: list[dict[str, Any]] = []
    if plan.label == "official":
        contract_path = root / "contract.json"
        if not contract_path.is_file():
            raise _refused("REFUSED-MISSING-PREREQUISITE", "official replay needs contract.json")
        contract = verify_contract_file(contract_path)
        checks.append({"check": "contract", "pass": True, "sha256": contract["sha256"]})
    store = storage.RunStore(root=root)
    verification = store.verify()
    checks.append({"check": "chunk_custody", "pass": bool(verification["pass"]),
                   "chunks": verification["checks"], "failures": verification["failures"]})
    if not verification["pass"]:
        raise _refused("REFUSED-HASH-MISMATCH", "chunk custody failed before replay")
    chunks = store.load_chunks()
    results = evaluation.evaluate_chunks(chunks, plan)
    aggregates = results["aggregates"]
    coverage_ok = all(aggregates[readout]["coverage_ok"] for readout in spec.READOUTS)
    decision = evaluation.decide(aggregates, coverage_ok=coverage_ok)
    checks.append({"check": "plan_digest", "pass": results["plan_digest"] == design["plan"]["enumeration_sha256"]})

    # Window-level identity: one row per file, no duplicate file inside a window.
    duplicates: list[str] = []
    for chunk in chunks:
        for window in chunk.windows:
            ids = [str(value) for value in window["file_ids"]]
            if len(set(ids)) != len(ids):
                duplicates.append(str(window["window_id"]))
    checks.append({"check": "window_file_uniqueness", "pass": not duplicates, "duplicates": duplicates[:10]})

    statistics_path = root / "statistics.json"
    decision_path = root / "decision.json"
    if statistics_path.is_file():
        persisted = storage.read_json(statistics_path)
        problems = _compare(
            {
                "thresholds": results["thresholds"],
                "aggregates": results["aggregates"],
                "plan_digest": results["plan_digest"],
            },
            {
                "thresholds": persisted["thresholds"],
                "aggregates": persisted["aggregates"],
                "plan_digest": persisted["plan_digest"],
            },
            path="statistics",
        )
        checks.append({"check": "statistics_match", "pass": not problems, "problems": problems[:20]})
    else:
        checks.append({"check": "statistics_match", "pass": False, "problems": ["statistics.json missing"]})
    if decision_path.is_file():
        persisted_decision = storage.read_json(decision_path)
        problems = _compare(
            {"action": decision["action"], "statuses": decision["statuses"], "coverage_ok": decision["coverage_ok"]},
            {"action": persisted_decision["action"], "statuses": persisted_decision["statuses"], "coverage_ok": persisted_decision["coverage_ok"]},
            path="decision",
        )
        checks.append({"check": "decision_match", "pass": not problems, "problems": problems[:20]})
    else:
        checks.append({"check": "decision_match", "pass": False, "problems": ["decision.json missing"]})

    payload = {
        "status": "REPLAY-MATCH" if all(item["pass"] for item in checks) else "REPLAY-MISMATCH",
        "plan_label": plan.label,
        "plan_digest": results["plan_digest"],
        "decision": decision,
        "checks": checks,
        "encoder_calls": 0,
        "model_loads": 0,
        "torch_imported": "torch" in sys.modules,
    }
    storage.write_json(destination / "replay.json", payload)
    if payload["status"] != "REPLAY-MATCH":
        raise _refused("REFUSED-REPLAY-MISMATCH", json.dumps(checks, sort_keys=True)[:1000])
    return payload


# --- behavior fixtures ----------------------------------------------------------


def _fixture_records(plan: spec.Plan, coverage: str, alarms: Sequence[bool]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, alarm in enumerate(alarms):
        records.append({
            "window_id": f"fixture:{index}",
            "horizon": "week",
            "burden": spec.BURDEN_LABELS[1],
            "cell_index": 0,
            "family": spec.FAMILIES[0],
            "severity": spec.SEVERITIES[0],
            "coverage": coverage,
            "combined": {readout: bool(alarm) for readout in spec.READOUTS},
            "alarms": {
                readout: {str(seed): bool(alarm) for seed in spec.ENCODER_SEEDS}
                for readout in spec.READOUTS
            },
            "stats": {readout: {} for readout in spec.READOUTS},
        })
    return records


def behavior_fixtures(root: Path) -> dict[str, Any]:
    """Exercise abstention, hash failure, length weights and every action branch."""
    root = Path(root)
    fixtures: dict[str, Any] = {}

    branches = {
        "keep_pooled": {"POOLED": spec.READOUT_QUALIFIES, "PREPOOL": spec.READOUT_QUALIFIES, "RAW": spec.READOUT_FAILS},
        "change_readout": {"POOLED": spec.READOUT_FAILS, "PREPOOL": spec.READOUT_QUALIFIES, "RAW": spec.READOUT_FAILS},
        "target_representation": {"POOLED": spec.READOUT_FAILS, "PREPOOL": spec.READOUT_FAILS, "RAW": spec.READOUT_QUALIFIES},
        "stop_scale": {"POOLED": spec.READOUT_FAILS, "PREPOOL": spec.READOUT_FAILS, "RAW": spec.READOUT_FAILS},
        "invalid_coverage": {"POOLED": spec.READOUT_QUALIFIES, "PREPOOL": spec.READOUT_QUALIFIES, "RAW": spec.READOUT_QUALIFIES},
        "invalid_readout": {"POOLED": spec.READOUT_INVALID, "PREPOOL": spec.READOUT_QUALIFIES, "RAW": spec.READOUT_QUALIFIES},
    }
    expected = {
        "keep_pooled": spec.ACTION_KEEP_POOLED,
        "change_readout": spec.ACTION_CHANGE_READOUT,
        "target_representation": spec.ACTION_TARGET_REPRESENTATION,
        "stop_scale": spec.ACTION_STOP_SCALE,
        "invalid_coverage": spec.ACTION_INVALID_IMPLEMENTATION,
        "invalid_readout": spec.ACTION_INVALID_IMPLEMENTATION,
    }
    fixture_branch_results: dict[str, Any] = {}
    for name, statuses in branches.items():
        coverage_ok = name != "invalid_coverage"
        result = evaluation.decision_from_fixture(statuses, coverage_ok=coverage_ok)
        fixture_branch_results[name] = {
            "expected": expected[name],
            "actual": result["action"],
            "pass": result["action"] == expected[name],
        }
    fixtures["decision_branches"] = {
        "pass": all(item["pass"] for item in fixture_branch_results.values()),
        "results": fixture_branch_results,
    }

    # Abstention: a non-COVERED window never enters an alarm denominator.
    plan = spec.smoke_plan()
    covered = _fixture_records(plan, "COVERED", [False, False])
    abstained = _fixture_records(plan, "ABSTAIN-COVERAGE-15", [True])
    aggregate = evaluation.aggregate_test_records(spec.READOUTS[0], covered + abstained, plan)
    cell0 = aggregate["coverage"]["week"][spec.BURDEN_LABELS[1]]["0"]
    rate0 = aggregate["cell_rates"]["week"][spec.BURDEN_LABELS[1]]["0"]
    fixtures["abstention"] = {
        "pass": cell0 == 2 and rate0["trials"] == 2 and rate0["successes"] == 0,
        "covered_windows": cell0,
        "trials": rate0["trials"],
        "successes": rate0["successes"],
    }

    # Length weighting: each file is exactly one item row in every window meta.
    store = storage.RunStore(root=root)
    length_checks: list[dict[str, Any]] = []
    for chunk in store.load_chunks(kinds=("test",)):
        if not chunk.windows:
            continue
        rows = {key: int(value.shape[0]) for key, value in chunk.items.items()}
        lengths = [int(record["T"]) for record in chunk.meta["files"]]
        patches = [int(record["n_patches"]) for record in chunk.meta["files"]]
        ids = [str(record["file_id"]) for record in chunk.meta["files"]]
        file_weights_ok = all(count == len(ids) for count in rows.values())
        length_checks.append({
            "chunk_id": chunk.chunk_id,
            "files": len(ids),
            "rows_match_files": file_weights_ok,
            "distinct_ids": len(set(ids)) == len(ids),
            "length_range": [min(lengths), max(lengths)] if lengths else [0, 0],
            "patch_range": [min(patches), max(patches)] if patches else [0, 0],
        })
    fixtures["length_weights"] = {
        "pass": bool(length_checks) and all(
            item["rows_match_files"] and item["distinct_ids"] for item in length_checks
        ),
        "chunks": length_checks[:5],
        "interpretation": "one equal-weight item per file regardless of timestep or patch count",
    }

    # Hash failure: a corrupted item member must fail closed.
    import tempfile

    with tempfile.TemporaryDirectory(dir=root, prefix="hash-fixture-") as temporary:
        fixture_store = storage.RunStore.create(Path(temporary) / "store", fresh=True)
        chunk = fixture_store.write_chunk(
            history_id="FIXTURE",
            kind="reference",
            chunk_id="corrupt",
            items={"raw": __import__("numpy").zeros((2, 3), dtype="float32")},
            waveforms=[__import__("numpy").zeros((6, 8), dtype="float32")] * 2,
            meta={"files": [{"file_id": "F1"}, {"file_id": "F2"}], "windows": []},
        )
        target = Path(temporary) / "store/chunks/FIXTURE/reference/corrupt.items.npz"
        payload = bytearray(target.read_bytes())
        payload[-1] ^= 0xFF
        target.write_bytes(bytes(payload))
        try:
            fixture_store.read_chunk("FIXTURE", "reference", "corrupt")
            failure = {"pass": False, "error": "corruption was accepted"}
        except ValueError as exc:
            failure = {"pass": "REFUSED-HASH-MISMATCH" in str(exc), "error": str(exc)}
        failure["chunk_sha256"] = chunk["sha256"]["items"]
    fixtures["hash_failure"] = failure

    # Tokens/items: the three readouts consume the same file row set.
    chunk = next((item for item in store.load_chunks(kinds=("test",)) if item.windows), None)
    if chunk is None:
        fixtures["same_waveforms"] = {"pass": False, "error": "no test chunk"}
    else:
        keys = evaluation.item_keys()
        shapes = {key: int(chunk.items[key].shape[0]) for key in keys if key in chunk.items}
        fixtures["same_waveforms"] = {
            "pass": len(set(shapes.values())) == 1 and set(shapes.values()) == {len(chunk.meta["files"])},
            "rows": shapes,
            "files": len(chunk.meta["files"]),
        }
    fixtures["all_pass"] = all(
        item.get("pass", False)
        for key, item in fixtures.items()
        if key not in ("interpretation",)
    )
    return fixtures


# --- mode entry points ----------------------------------------------------------


def _require_bundle_args(args: argparse.Namespace) -> None:
    for name in ("b0_root", "bank_root", "preflight"):
        if not getattr(args, name, None):
            raise _refused("REFUSED-PREREQUISITE", f"smoke/official needs --{name.replace('_', '-')}")


def _fresh_root(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise _refused("REFUSED-OVERWRITE", f"{root} is non-empty")


def run_smoke(out: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Real tiny smoke: accepted bundles, all readouts, disk, replay, decision."""
    from population_benchmark import capture as capture_module

    started = time.time()
    out = Path(out)
    _fresh_root(out)
    _require_bundle_args(args)
    plan = spec.smoke_plan()
    spec.validate_plan(plan)
    store = storage.RunStore.create(out, fresh=True)
    design = write_design(out, plan)
    store.append_event("run_start", {
        "task": spec.TASK_ID, "attempt": args.attempt_id or spec.ATTEMPT_ID,
        "mode": "smoke", "plan_digest": spec.enumeration_digest(plan),
        "source_commit": source_commit(),
    })
    bundles = capture_module.restore_bundles(
        repo=REPO,
        b0_root=Path(args.b0_root),
        bank_root=Path(args.bank_root),
        preflight_path=Path(args.preflight),
        device=args.device or "cpu",
    )
    try:
        counters = run_plan_capture(
            store=store, plan=plan, bundles=bundles, device=args.device or "cpu", resume=False
        )
    except Exception as exc:  # noqa: BLE001 - durable terminal record then re-raise
        store.append_event("terminal_failure", {"exception": repr(exc), "stage": "capture"})
        raise
    evaluation_result = evaluate_run(out)
    replay_dir = out / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(CLI), "--mode", "replay", "--run", str(out), "--out", str(replay_dir)],
        cwd=str(REPO), env=env, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        store.append_event("terminal_failure", {
            "exception": "fresh replay failed", "returncode": completed.returncode,
            "stderr": completed.stderr[-2000:], "stage": "replay",
        })
        raise _refused("REFUSED-REPLAY-MISMATCH", completed.stderr[-1000:])
    replay = storage.read_json(replay_dir / "replay.json")
    fixtures = behavior_fixtures(out)
    elapsed = round(time.time() - started, 2)
    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    summary = {
        "status": "SMOKE-COMPLETE",
        "task": spec.TASK_ID,
        "attempt": args.attempt_id or spec.ATTEMPT_ID,
        "agent": getattr(args, "agent_id", None),
        "command": " ".join(sys.argv),
        "plan_label": plan.label,
        "plan_digest": spec.enumeration_digest(plan),
        "design": design,
        "contract_sha256": spec.contract_sha256(),
        "forwards_executed_this_process": int(counters["forwards"]),
        "files_captured": int(counters["files"]),
        "chunks_written": int(counters["chunks"]),
        "model_loads": int(counters["model_loads"]),
        "device": args.device or "cpu",
        "source_commit": source_commit(),
        "source_hashes": source_hashes(),
        "checkpoints": {str(seed): bundle.checkpoint.checkpoint_sha256 for seed, bundle in bundles.items()},
        "banks": {
            str(seed): storage.sha256_file(
                Path(args.bank_root) / f"B0-FULL-{seed}.json.fit-bank" / "bank.pt"
            )
            for seed in spec.ENCODER_SEEDS
        },
        "bank_envelopes": {
            str(seed): str(preflight["banks"][str(seed)]) for seed in spec.ENCODER_SEEDS
        },
        "decision": evaluation_result["decision"],
        "replay": {
            "status": replay["status"],
            "checks": replay["checks"],
            "encoder_calls": replay["encoder_calls"],
            "torch_imported": replay["torch_imported"],
        },
        "behavior": fixtures,
        "non_evidence": True,
        "elapsed_s": elapsed,
        "cumulative_forwards_before": CUMULATIVE_FORWARDS_BEFORE,
        "cumulative_forwards_after": CUMULATIVE_FORWARDS_BEFORE + int(counters["forwards"]),
    }
    storage.write_json(out / "run.json", {
        "task": spec.TASK_ID, "attempt": summary["attempt"], "agent": summary["agent"],
        "mode": "smoke", "status": summary["status"], "source_commit": summary["source_commit"],
        "source_hashes": summary["source_hashes"], "command": summary["command"],
        "device": summary["device"], "forwards_executed_this_process": summary["forwards_executed_this_process"],
        "model_loads": summary["model_loads"], "chunks_written": summary["chunks_written"],
        "plan_digest": summary["plan_digest"], "contract_sha256": summary["contract_sha256"],
        "checkpoints": summary["checkpoints"], "banks": summary["banks"],
        "bank_envelopes": summary["bank_envelopes"],
        "elapsed_s": elapsed,
    })
    storage.write_json(out / "smoke.json", summary)
    store.append_event("run_complete", {"status": summary["status"], "forwards": counters["forwards"]})
    return summary


def _release_ok(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise _refused("REFUSED-OFFICIAL-RELEASE", f"release manifest absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "task": spec.TASK_ID,
        "m20_evidence_pass": True,
        "m20_checkpoint": True,
        "m21_release": True,
        "contract_sha256": spec.contract_sha256(),
    }
    bad = {key: (payload.get(key), value) for key, value in required.items() if payload.get(key) != value}
    if bad:
        raise _refused("REFUSED-OFFICIAL-RELEASE", f"release manifest mismatch: {sorted(bad)}")
    return payload


def run_official(out: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Full frozen plan; refuses without an explicit matching release manifest."""
    from population_benchmark import capture as capture_module

    if not getattr(args, "permit_official", False):
        raise _refused("REFUSED-OFFICIAL-RELEASE", "official capture needs --permit-official")
    if not getattr(args, "release_manifest", None):
        raise _refused("REFUSED-OFFICIAL-RELEASE", "official capture needs --release-manifest")
    _release_ok(Path(args.release_manifest))
    _require_bundle_args(args)
    started = time.time()
    out = Path(out)
    resume = bool(getattr(args, "resume", False))
    plan = spec.official_plan()
    spec.validate_plan(plan)
    if resume:
        if not (out / "design.json").is_file():
            raise _refused("REFUSED-PREREQUISITE", "resume needs an existing design.json")
        design = storage.read_json(out / "design.json")
        if spec.canonical_json(design["plan"]) != spec.canonical_json(spec.summary(plan)):
            raise _refused("REFUSED-IDENTITY", "resume plan differs from the frozen plan")
        store = storage.RunStore.create(out, fresh=False)
    else:
        _fresh_root(out)
        store = storage.RunStore.create(out, fresh=True)
        write_design(out, plan)
    contract_path = out / "contract.json"
    if contract_path.is_file():
        verify_contract_file(contract_path)
    else:
        write_contract(out)
    disk = store.require_disk(spec.DISK_BUDGET_BYTES)
    if not disk["pass"]:
        store.append_event("terminal_failure", {"exception": "disk budget", "disk": disk})
        raise _refused("REFUSED-DISK-BUDGET", json.dumps(disk, sort_keys=True))
    store.append_event("run_start", {
        "task": spec.TASK_ID, "attempt": args.attempt_id or spec.ATTEMPT_ID, "mode": "official",
        "plan_digest": spec.enumeration_digest(plan), "contract_sha256": spec.contract_sha256(),
        "source_commit": source_commit(), "disk": disk, "resume": resume,
    })
    bundles = capture_module.restore_bundles(
        repo=REPO, b0_root=Path(args.b0_root), bank_root=Path(args.bank_root),
        preflight_path=Path(args.preflight), device=args.device or "cuda:0",
    )
    try:
        counters = run_plan_capture(
            store=store, plan=plan, bundles=bundles, device=args.device or "cuda:0", resume=resume
        )
        evaluation_result = evaluate_run(out)
    except Exception as exc:  # noqa: BLE001
        store.append_event("terminal_failure", {"exception": repr(exc), "stage": "official"})
        storage.write_json(out / "run.json", {
            "task": spec.TASK_ID, "attempt": args.attempt_id or spec.ATTEMPT_ID,
            "mode": "official", "status": "RUNTIME-FAIL", "exception": repr(exc),
            "forwards_executed_this_process": int(counters["forwards"]) if "counters" in locals() else 0,
            "elapsed_s": round(time.time() - started, 2),
        })
        raise
    replay_dir = out / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(CLI), "--mode", "replay", "--run", str(out), "--out", str(replay_dir)],
        cwd=str(REPO), env=env, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        store.append_event("terminal_failure", {
            "exception": "fresh replay failed", "returncode": completed.returncode,
            "stderr": completed.stderr[-2000:], "stage": "replay",
        })
        raise _refused("REFUSED-REPLAY-MISMATCH", completed.stderr[-1000:])
    replay = storage.read_json(replay_dir / "replay.json")
    elapsed = round(time.time() - started, 2)
    run = {
        "task": spec.TASK_ID,
        "attempt": args.attempt_id or spec.ATTEMPT_ID,
        "agent": getattr(args, "agent_id", None),
        "mode": "official",
        "status": "OFFICIAL-COMPLETE",
        "source_commit": source_commit(),
        "source_hashes": source_hashes(),
        "device": args.device or "cuda:0",
        "plan_digest": spec.enumeration_digest(plan),
        "contract_sha256": spec.contract_sha256(),
        "checkpoints": {str(seed): bundle.checkpoint.checkpoint_sha256 for seed, bundle in bundles.items()},
        "preprocessing": {str(seed): bundle.preprocessing_sha256 for seed, bundle in bundles.items()},
        "forwards_executed_this_process": int(counters["forwards"]),
        "files_captured": int(counters["files"]),
        "chunks_written": int(counters["chunks"]),
        "chunks_resumed": int(counters["resumed"]),
        "model_loads": int(counters["model_loads"]),
        "cumulative_forwards_before": CUMULATIVE_FORWARDS_BEFORE,
        "cumulative_forwards_after": CUMULATIVE_FORWARDS_BEFORE + int(counters["forwards"]),
        "decision": evaluation_result["decision"]["action"],
        "replay_status": replay["status"],
        "elapsed_s": elapsed,
        "disk": disk,
        "command": " ".join(sys.argv),
    }
    storage.write_json(out / "run.json", run)
    store.append_event("run_complete", {
        "status": run["status"], "forwards": counters["forwards"], "replay": replay["status"],
    })
    return run


__all__ = [
    "CUMULATIVE_FORWARDS_BEFORE",
    "DOCS_CONTRACT",
    "REPO",
    "SOURCE_PATHS",
    "behavior_fixtures",
    "evaluate_run",
    "replay_run",
    "run_official",
    "run_plan_capture",
    "run_smoke",
    "source_commit",
    "source_hashes",
    "verify_contract_file",
    "write_contract",
    "write_design",
]
