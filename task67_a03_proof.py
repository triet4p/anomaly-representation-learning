from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from synth import balanced as B
from synth import events as E
from synth import probe15 as P
from synth.chronicle import (
    _episode_to_dict,
    _failure_to_dict,
    _maintenance_windows,
    _operation_to_dict,
    load_chronological,
    sprint18_iterative_v1_history_config,
    sprint18_iterative_v2_history_config,
)
from synth.health import RobotHealthProcess
from synth.scheduler import FactoryScheduler
from synth.schema import EpisodeKind, FileSample, SampleLabel
from synth.temporal import SUBTYPE_GAINS, _cohort_windows
from experiments.sprint18_task5_measurability import (
    confirmation_hard_floor_checks,
    design_promotion_checks,
    healthy_features,
    structural_summary,
)

ROOT = Path.cwd()
TASK_ROOT = Path(
    "data/generated/.sprint18-iterative-contract-v2/task67/"
    "S18-T67-A03-C02"
)
V1_B_ROOT = Path(
    "data/generated/.sprint18-iterative-contract-v1/task67/"
    "S18-T67-A01-20261002-03/fixture-b"
)
V1_B_MIX_ROOT = Path(
    "data/generated/.sprint18-iterative-contract-v1/task67/"
    "S18-T67-A02-20261002-02/fixture-b"
)
V1_C_ROOT = Path(
    "data/generated/.sprint18-iterative-contract-v1/task67/"
    "S18-T67-A01-20261002-05/fixture-c"
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=str,
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite proof record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def strip_subtype(record: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "subtype"}


def _check_episode_and_failure_labels(
    health, seed: int, config,
) -> dict[str, object]:
    episodes = health.episodes
    failures = health.failure_events
    by_episode = {episode.episode_id: episode for episode in episodes}
    episode_ordinals: Counter[tuple[str, str]] = Counter()
    episode_counts: Counter[tuple[str, str]] = Counter()
    for episode in episodes:
        if episode.kind is not EpisodeKind.DEGRADATION:
            continue
        subtype = episode.subtype
        if subtype is None:
            continue
        cohort = "P" if subtype in ("P1", "P2") else "W"
        key = (episode.robot_id, cohort)
        episode_ordinals[key] += 1
        episode_counts[key] += 1
        digest = hashlib.sha256("|".join([
            "sprint14-subtype", str(seed), key[0], cohort,
            str(episode_ordinals[key]),
        ]).encode()).digest()
        labels = next(item.subtypes for item in config.health.cohorts
                      if item.cohort_id == cohort)
        expected = labels[(episode_ordinals[key] + digest[0]) % len(labels)]
        assert subtype == expected, (episode.episode_id, subtype, expected)

    failure_ordinals: Counter[tuple[str, str]] = Counter()
    failure_counts: dict[str, dict[str, int]] = {}
    dual_label_count = 0
    for failure in failures:
        cohort = failure.cohort
        if cohort == "A":
            continue
        key = (failure.robot_id, cohort)
        failure_ordinals[key] += 1
        digest = hashlib.sha256("|".join([
            "sprint18-failure-subtype", str(seed), key[0], cohort,
            str(failure_ordinals[key]),
        ]).encode()).digest()
        labels = next(item.subtypes for item in config.health.cohorts
                      if item.cohort_id == cohort)
        expected = labels[(failure_ordinals[key] + digest[0]) % len(labels)]
        assert failure.subtype == expected, (
            failure.failure_id, failure.subtype, expected)
        episode = by_episode[failure.degradation_episode_id]
        assert episode.subtype in labels
        if episode.subtype != failure.subtype:
            dual_label_count += 1
        bucket = failure_counts.setdefault(key[0] + "/" + cohort, {})
        bucket[failure.subtype] = bucket.get(failure.subtype, 0) + 1

    for key, counts in failure_counts.items():
        cohort = key.rsplit("/", 1)[1]
        labels = ("P1", "P2") if cohort == "P" else ("W1", "W2")
        assert abs(counts.get(labels[0], 0) - counts.get(labels[1], 0)) <= 1, (
            key, counts)

    return {
        "episode_opening_counts_by_robot_cohort": {
            f"{robot}/{cohort}": count
            for (robot, cohort), count in sorted(episode_counts.items())
        },
        "failure_counts_by_robot_cohort_subtype": {
            key: counts for key, counts in sorted(failure_counts.items())
        },
        "dual_label_failure_count": dual_label_count,
    }


def fixture_a(root: Path, seed: int, reload_number: int) -> None:
    samples, manifest = load_chronological(root)
    assert manifest["role"] == f"TASK67-FIX-REAL-{seed}"
    assert manifest["protocol"] == "sprint15-benchmark-protocol-v7"
    assert "sprint15" not in manifest
    assert len(samples) == manifest["counts"]["total"]
    assert manifest["calendar"]["span_s"] == 450.0 * 86400.0
    assert manifest["calendar"]["cutoff_time"] == 225.0 * 86400.0
    assert manifest["resolved_config"]["scheduler"]["n_units"] == 2880

    config = sprint18_iterative_v2_history_config(seed)
    legacy_config = sprint18_iterative_v1_history_config(seed)
    assert asdict(config) == asdict(legacy_config)
    assert manifest["config_hash"] == config.hash()
    assert manifest["resolved_config"] == json.loads(
        json.dumps(asdict(config), sort_keys=True, default=str)
    )

    sample_ids = [sample.file_id for sample in samples]
    assert sample_ids == [row["file_id"] for row in manifest["files"]]
    assert all(set(row) == B.ALLOWED_ROW_KEYS for row in manifest["files"])
    forbidden = {"cohort", "subtype", "failure_time", "allocation"}
    assert all(not (forbidden & set(row)) for row in manifest["files"])
    waveform_hash = hashlib.sha256()
    channel_min = np.full(6, np.inf, dtype=np.float64)
    channel_max = np.full(6, -np.inf, dtype=np.float64)
    for sample in samples:
        sample.validate()
        assert sample.x.shape[0] == 6
        assert np.isfinite(sample.x).all()
        assert sample.x.flags.c_contiguous
        waveform_hash.update(sample.file_id.encode("utf-8"))
        waveform_hash.update(memoryview(sample.x).cast("B"))
        channel_min = np.minimum(channel_min, sample.x.min(axis=1))
        channel_max = np.maximum(channel_max, sample.x.max(axis=1))

    schedule = FactoryScheduler(config).build()
    schedule_rows = [_operation_to_dict(event) for event in schedule.events]
    assert schedule_rows == manifest["schedule"]
    assert all(0.0 <= row["start_time"] <= row["end_time"]
               <= 450.0 * 86400.0 for row in schedule_rows)

    on_health = RobotHealthProcess(config).run(schedule)
    v1_health = RobotHealthProcess(legacy_config).run(schedule)
    off_config = sprint18_iterative_v2_history_config(seed)
    off_config.health = replace(
        off_config.health, stratified_subtype_emission=False)
    off_health = RobotHealthProcess(off_config).run(schedule)
    on_failures = [_failure_to_dict(item) for item in on_health.failure_events]
    off_failures = [_failure_to_dict(item) for item in off_health.failure_events]
    v1_failures = [_failure_to_dict(item) for item in v1_health.failure_events]
    on_episodes = [_episode_to_dict(item) for item in on_health.episodes]
    off_episodes = [_episode_to_dict(item) for item in off_health.episodes]
    assert on_failures == manifest["failure_events"]
    assert on_episodes == manifest["episodes"]
    assert [strip_subtype(item) for item in on_failures] == [
        strip_subtype(item) for item in v1_failures]
    assert [asdict(item) for item in on_health.states] == [
        asdict(item) for item in off_health.states]
    assert [strip_subtype(asdict(item)) for item in on_health.episodes] == [
        strip_subtype(asdict(item)) for item in off_health.episodes]
    assert on_episodes == off_episodes
    assert [strip_subtype(item) for item in on_failures] == [
        strip_subtype(item) for item in off_failures]
    off_episodes_by_id = {
        episode.episode_id: episode for episode in off_health.episodes}
    assert all(
        failure.subtype == off_episodes_by_id[
            failure.degradation_episode_id].subtype
        for failure in off_health.failure_events if failure.cohort in ("P", "W")
    )

    label_metrics = _check_episode_and_failure_labels(on_health, seed, config)
    assert label_metrics["dual_label_failure_count"] > 0
    temporal_windows = _cohort_windows(on_health, config.temporal.failure_severity)
    for failure in manifest["failure_events"]:
        if failure["cohort"] == "A":
            continue
        matches = [entry for entry in temporal_windows[failure["robot_id"]]
                   if entry[0] == failure["degradation_onset"]
                   and entry[1] == failure["failure_time"]]
        assert len(matches) == 1
        expected_mult = SUBTYPE_GAINS[failure["subtype"]]
        assert matches[0][2] == failure["cohort"]
        assert matches[0][3] == expected_mult

    rows = manifest["files"]
    ledger = manifest["failure_events"]
    windows = manifest["maintenance_windows"]
    inputs = B.project_allocation_inputs(rows)
    eligible, _, _ = B.eligible_anchors(inputs, ledger, windows)
    pools = {
        f"{cohort}/{subtype}": sum(
            event["cohort"] == cohort and event["subtype"] == subtype
            for event in eligible
        )
        for cohort, subtype in B.BUCKET_ORDER
    }
    allocation = B.allocate_quotas(
        rows, ledger, windows, seed, B.DEFAULT_QUOTA, "exact")
    program = B.build_allocation_program(
        eligible, inputs, B.DEFAULT_QUOTA)
    witness = allocation["selected_ids"]
    witness_breaches = B.validate_allocation_witness(
        witness, program,
        {event["failure_id"]: event for event in eligible},
        B.DEFAULT_QUOTA.spacing_s,
    )
    assert len(witness) == 64 and not witness_breaches
    selected_counts = {
        cohort: {subtype: len(ids) for subtype, ids in subtypes.items()}
        for cohort, subtypes in allocation["selected"].items()
    }
    assert selected_counts == {
        "P": {"P1": 12, "P2": 12},
        "W": {"W1": 12, "W2": 12},
        "A": {"A1": 8, "A2": 8},
    }
    assert allocation["margins"]["controls"] >= 49
    assert all(value >= need for value, need in zip(
        pools.values(), (12, 12, 12, 12, 8, 8), strict=True))

    role_entry = {
        "history_id": f"S18I-FIX-REAL-{seed}",
        "role": "DESIGN",
        "data_seed": seed,
    }
    structural = structural_summary(role_entry, manifest)
    design = design_promotion_checks(structural)
    hard_floor = confirmation_hard_floor_checks(structural)
    assert structural["core_integrity_pass"]
    assert all(structural["checks"].values())
    assert all(design.values())

    schema_projection = {
        "allowed_row_keys": sorted(B.ALLOWED_ROW_KEYS),
        "model_input_fields": sorted(configured for configured in
                                       __import__("synth.schema", fromlist=[
                                           "MODEL_INPUT_FIELD_NAMES"]).MODEL_INPUT_FIELD_NAMES),
    }
    manifest_bytes = (root / "manifest.json").read_bytes()
    source_results = {
        "fixture_id": f"S18I-FIX-REAL-{seed}",
        "seed": seed,
        "profile": "sprint18-iterative-v2",
        "manifest_role": manifest["role"],
        "protocol": manifest["protocol"],
        "config_hash": manifest["config_hash"],
        "resolved_config_sha256": sha256(canonical(manifest["resolved_config"])),
        "manifest_sha256": sha256(manifest_bytes),
        "manifest_counts": manifest["counts"],
        "calendar": manifest["calendar"],
        "schedule_count": len(schedule_rows),
        "schedule_sha256": sha256(canonical(schedule_rows)),
        "sample_count": len(samples),
        "shard_count": len(manifest["shards"]),
        "waveform_stream_sha256": waveform_hash.hexdigest(),
        "channel_min": channel_min.tolist(),
        "channel_max": channel_max.tolist(),
        "schema_projection_sha256": sha256(canonical(schema_projection)),
        "failure_event_count": len(ledger),
        "failure_ledger_sha256": sha256(canonical(ledger)),
        "episode_count": len(manifest["episodes"]),
        "episode_ledger_sha256": sha256(canonical(manifest["episodes"])),
        "failure_subtype_metrics": label_metrics,
        "eligible_pool_counts": pools,
        "selected_counts": selected_counts,
        "selected_ids_sha256": sha256(canonical(witness)),
        "witness_breaches": witness_breaches,
        "control_count": allocation["margins"]["controls"],
        "support_margins": allocation["margins"],
        "structural_checks": structural["checks"],
        "design_promotion_checks": design,
        "confirmation_hard_floor_checks": hard_floor,
        "legacy_flag_timing_density_and_health_state_invariant": True,
        "flag_off_episode_inheritance_preserved": True,
        "subtype_to_precursor_gain_matches_failure_record": True,
        "fresh_process_pid": os.getpid(),
        "reload_number": reload_number,
    }
    assert all(source_results["manifest_counts"][key] > 0 for key in
               ("normal", "abnormal", "dev_train", "dev_val",
                "test_static", "test_temporal"))
    assert (root / f"reload-{reload_number}.json").parent == root
    write_json(root / f"reload-{reload_number}.json", source_results)



def compare_reloads(root: Path) -> None:
    first = json.loads((root / "reload-1.json").read_text(encoding="utf-8"))
    second = json.loads((root / "reload-2.json").read_text(encoding="utf-8"))
    assert first["fresh_process_pid"] != second["fresh_process_pid"]
    fields = (
        "manifest_sha256", "manifest_counts", "calendar", "schedule_count",
        "schedule_sha256", "sample_count", "shard_count",
        "waveform_stream_sha256", "schema_projection_sha256",
        "failure_event_count", "failure_ledger_sha256", "episode_count",
        "episode_ledger_sha256", "failure_subtype_metrics",
        "eligible_pool_counts", "selected_counts", "selected_ids_sha256",
        "control_count", "support_margins", "structural_checks",
        "design_promotion_checks", "confirmation_hard_floor_checks",
    )
    for field in fields:
        assert first[field] == second[field], field
    write_json(root / "reload-comparison.json", {
        "fresh_processes": [first["fresh_process_pid"], second["fresh_process_pid"]],
        "identical_fields": list(fields),
        "identical": True,
    })


def fixture_b() -> None:
    output_root = TASK_ROOT / "fixture-b"
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse Fixture B root: {output_root}")
    output_root.mkdir(parents=True)
    source_files = {
        "joint-good-49": (V1_B_ROOT / "joint-good-input.json", "v1-a01"),
        "controls-47": (V1_B_ROOT / "controls-47-input.json", "v1-a01"),
        "controls-48": (V1_B_ROOT / "controls-48-input.json", "v1-a01"),
        "controls-49": (V1_B_ROOT / "controls-49-input.json", "v1-a01"),
        "w2-11": (V1_B_ROOT / "w2-11-input.json", "v1-a01"),
        "lead-76": (V1_B_ROOT / "lead-76-input.json", "v1-a01"),
        "mix-exact-15": (V1_B_MIX_ROOT / "mix-exact15-input.json", "v1-a02"),
        "mix-below-15": (V1_B_MIX_ROOT / "mix-below15-input.json", "v1-a02"),
        "joint-csp-conflict": (V1_B_ROOT / "joint-csp-conflict-input.json", "v1-a01"),
    }
    bundles: dict[str, dict[str, object]] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    for name, (source, source_attempt) in source_files.items():
        raw = source.read_bytes()
        bundle = json.loads(raw)
        bundles[name] = bundle
        target = output_root / f"{name}-input.json"
        target.write_bytes(raw)
        input_hashes[name] = {
            "source_attempt": source_attempt,
            "source_sha256": sha256(raw),
            "copy_sha256": sha256(target.read_bytes()),
        }
        assert input_hashes[name]["source_sha256"] == input_hashes[name]["copy_sha256"]

    w1_bundle = copy.deepcopy(bundles["w2-11"])
    w1_bundle["fixture_id"] = "S18I-FIX-w1-11-mirror"
    for failure in w1_bundle["manifest"]["failure_events"]:
        if failure["cohort"] == "W":
            failure["subtype"] = "W1" if failure["subtype"] == "W2" else "W2"
    w1_bytes = canonical(w1_bundle) + b"\n"
    (output_root / "w1-11-input.json").write_bytes(w1_bytes)
    bundles["w1-11"] = w1_bundle
    input_hashes["w1-11"] = {
        "source_attempt": "mirror-of-v1-a01-w2-11",
        "source_sha256": sha256(w1_bytes),
        "copy_sha256": sha256((output_root / "w1-11-input.json").read_bytes()),
        "transformation": "swap only W1/W2 event subtype labels; retain rows, timing, windows, support, IDs and controls",
    }

    def run_case(name: str, bundle: dict[str, object]) -> dict[str, object]:
        manifest = bundle["manifest"]
        seed = int(bundle["history_seed"])
        rows = manifest["files"]
        ledger = E.failure_ledger(manifest)
        windows = manifest["maintenance_windows"]
        inputs = B.project_allocation_inputs(rows)
        eligible, _, _ = B.eligible_anchors(inputs, ledger, windows)
        pools = {
            f"{cohort}/{subtype}": sum(
                item["cohort"] == cohort and item["subtype"] == subtype
                for item in eligible
            )
            for cohort, subtype in B.BUCKET_ORDER
        }
        support = structural_summary({
            "history_id": bundle["fixture_id"],
            "role": "DESIGN",
            "data_seed": seed,
        }, manifest)
        design = design_promotion_checks(support)
        hard_floor = confirmation_hard_floor_checks(support)
        record: dict[str, object] = {
            "case": name,
            "input_sha256": input_hashes[name]["copy_sha256"],
            "history_seed": seed,
            "file_rows": len(rows),
            "failure_events": len(ledger),
            "eligible_pools": pools,
            "selected_controls": support["support"]["negative_control_windows"],
            "lead_support": support["support"]["lead_support"],
            "structural_checks": support["checks"],
            "design_promotion_checks": design,
            "confirmation_hard_floor_checks": hard_floor,
        }
        try:
            allocation = B.allocate_quotas(
                rows, ledger, windows, seed, B.DEFAULT_QUOTA, "exact")
        except B.InfeasibleCandidate as exc:
            record["allocation_feasible"] = False
            record["allocation_failure"] = exc.record
            return record
        selected_ids = allocation["selected_ids"]
        program = B.build_allocation_program(
            eligible, inputs, B.DEFAULT_QUOTA)
        breaches = B.validate_allocation_witness(
            selected_ids, program,
            {item["failure_id"]: item for item in eligible},
            B.DEFAULT_QUOTA.spacing_s,
        )
        assert len(selected_ids) == 64 and not breaches
        record.update({
            "allocation_feasible": True,
            "selected_ids": selected_ids,
            "selected_ids_sha256": sha256(canonical(selected_ids)),
            "selected_by_subtype": {
                cohort: {subtype: len(ids) for subtype, ids in subtypes.items()}
                for cohort, subtypes in allocation["selected"].items()
            },
            "witness_breaches": breaches,
            "solver": allocation["solver"],
            "control_count": len(allocation["controls"]),
            "allocation_margins": allocation["margins"],
        })
        return record

    cases = {name: run_case(name, bundle) for name, bundle in bundles.items()}
    # lead-80 is the passing 20/25 side already present in joint-good-49.
    cases["lead-80"] = {
        "case": "lead-80",
        "input_sha256": cases["joint-good-49"]["input_sha256"],
        "lead_support": cases["joint-good-49"]["lead_support"],
        "pass_80_percent": {
            cohort: cases["joint-good-49"]["lead_support"][cohort]["pass_80_percent"]
            for cohort in ("P", "W")
        },
    }
    for name in ("joint-good-49", "controls-48", "controls-49",
                 "mix-exact-15"):
        assert cases[name]["allocation_feasible"]
        assert cases[name]["witness_breaches"] == []
    assert not cases["controls-47"]["allocation_feasible"]
    assert cases["controls-47"]["allocation_failure"]["reason"] == "control-shortfall"
    for name, subtype in (("w1-11", "W1"), ("w2-11", "W2")):
        assert cases[name]["eligible_pools"][f"W/{subtype}"] == 11
        assert not cases[name]["allocation_feasible"]
        assert cases[name]["lead_support"]["W"][
            "events_meeting_three_endpoints_early_endpoint_and_clean_baseline"] == 20
        assert cases[name]["lead_support"]["W"]["eligible_events"] == 24
        assert cases[name]["selected_controls"] == 49
    assert cases["mix-exact-15"]["design_promotion_checks"][
        "each_cohort_mix_15_to_60_percent"]
    assert not cases["mix-below-15"]["design_promotion_checks"][
        "each_cohort_mix_15_to_60_percent"]
    assert [key for key, value in cases["mix-below-15"][
        "design_promotion_checks"].items() if not value] == [
            "each_cohort_mix_15_to_60_percent"]
    assert cases["lead-80"]["pass_80_percent"] == {"P": True, "W": True}
    assert not cases["lead-76"]["design_promotion_checks"][
        "P_lead_support_ge_80_percent"]
    assert [key for key, value in cases["lead-76"][
        "design_promotion_checks"].items() if not value] == [
            "P_lead_support_ge_80_percent"]
    assert not cases["joint-csp-conflict"]["allocation_feasible"]
    assert cases["joint-csp-conflict"]["allocation_failure"]["solver"][
        "solver_status"] == 2

    write_json(output_root / "results.json", {
        "task_id": "S18-T67",
        "attempt_id": "S18-T67-A03",
        "contract_id": "sprint18-iterative-data-contract-v2",
        "fixture_id": "Fixture B — retained allocator suite plus W1 mirror",
        "input_hashes": input_hashes,
        "cases": cases,
        "allocator": "balanced.allocate_quotas + independent build_allocation_program/validate_allocation_witness",
        "all_named_acceptance_passed": True,
    })


def fixture_c() -> None:
    output_root = TASK_ROOT / "fixture-c"
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse Fixture C root: {output_root}")
    output_root.mkdir(parents=True)
    source_path = V1_C_ROOT / "complete-inputs.json"
    raw_input = source_path.read_bytes()
    (output_root / "complete-inputs.json").write_bytes(raw_input)
    spec = json.loads(raw_input)
    descriptor_sha = sha256(canonical(spec["descriptor"]))
    assert descriptor_sha == spec["descriptor_sha256"]
    assert descriptor_sha == "a9f999fda657e4a9c4ff0af12975bfb21f958dee9d9c1802e163e43d9a239e88"

    waveforms = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in spec["waveforms"].items()
    }
    waveform_sha_to_array = {
        sha256(array.tobytes(order="C")): array
        for array in waveforms.values()
    }
    assert set(waveform_sha_to_array) == set(spec["waveform_sha256"].values())

    def materialize(bundle: dict[str, object]):
        manifest = bundle["manifest"]
        assignments = {
            assignment["file_id"]: assignment["waveform_sha256"]
            for assignment in bundle["waveform_assignments"]
        }
        assert set(assignments) == {row["file_id"] for row in manifest["files"]}
        samples = []
        for row in manifest["files"]:
            waveform = waveform_sha_to_array[assignments[row["file_id"]]]
            sample = FileSample(
                x=waveform,
                file_id=row["file_id"],
                file_label=SampleLabel(row["file_label"]),
                seed=0,
                generator_version=str(manifest["generator_version"]),
                config_hash=str(manifest["config_hash"]),
                regime_sequence=[],
            )
            sample.validate()
            samples.append(sample)
        return samples, manifest

    fit_features = []
    fit_ids: list[str] = []
    for bundle in spec["fit_inputs"]:
        samples, manifest = materialize(bundle)
        features, ids, _, _ = healthy_features(samples, manifest)
        fit_features.extend(features)
        fit_ids.extend(ids)
    assert len(fit_features) == 3
    assert len(fit_ids) == len(set(fit_ids))
    fit = np.asarray(fit_features, dtype=np.float64)
    stats = P.standardize_fit(fit)
    standardized_fit = P.apply_standardization(fit, stats)
    centroid = P.fit_centroid(standardized_fit)
    assert len(P.feature_names()) == 59
    assert np.all(stats["std"] == P.STD_FLOOR)
    assert float(np.max(np.abs(centroid))) <= 1e-9

    calibration_samples, calibration_manifest = materialize(
        spec["calibration_input"])
    calibration_features, calibration_ids, _, calibration_hygiene = healthy_features(
        calibration_samples, calibration_manifest)
    assert calibration_ids == [
        row["file_id"] for row in calibration_manifest["files"]
    ]
    calibration_matrix = np.asarray(calibration_features, dtype=np.float64)
    calibration_scores = P.score_files(
        P.apply_standardization(calibration_matrix, stats), centroid)
    threshold = P.select_threshold(calibration_scores)
    assert len(calibration_scores) == 20
    assert np.all(calibration_scores[:19] == 0.0)
    assert calibration_scores[19] > 0.0
    assert np.isclose(threshold, 0.05 * calibration_scores[19],
                      rtol=1e-12, atol=1e-12)

    all_scores: dict[str, float] = {}
    rows_by_role: dict[str, list[list[dict[str, object]]]] = {
        "development": [], "confirmation": [],
    }
    ledgers_by_role: dict[str, list[list[dict[str, object]]]] = {
        "development": [], "confirmation": [],
    }
    windows_by_role: dict[str, list[dict[str, list[list[float]]]]] = {
        "development": [], "confirmation": [],
    }
    support: dict[str, dict[str, object]] = {}
    input_roles = spec["evaluation_inputs"]
    for role in ("development", "confirmation"):
        for history_id, bundle in input_roles[role].items():
            samples, manifest = materialize(bundle)
            sample_by_id = {sample.file_id: sample for sample in samples}
            rows = manifest["files"]
            matrix = np.asarray([
                P.extract_features(
                    np.asarray(sample_by_id[row["file_id"]].x, dtype=np.float64),
                    row["end_time"] - row["start_time"],
                    row["end_time"] - row["last_reset_time"],
                )
                for row in rows
            ], dtype=np.float64)
            scores = P.score_files(P.apply_standardization(matrix, stats), centroid)
            all_scores.update({row["file_id"]: float(score)
                               for row, score in zip(rows, scores, strict=True)})
            ledger = E.failure_ledger(manifest)
            windows = manifest["maintenance_windows"]
            rows_by_role[role].append(rows)
            ledgers_by_role[role].append(ledger)
            windows_by_role[role].append(windows)
            anchors = E.anchor_rows(rows, windows)
            controls = E.select_control_windows(anchors, ledger, windows)
            positives = {
                cohort: sum(
                    len(E.pos_files(rows, failure, windows))
                    for failure in ledger if failure["cohort"] == cohort
                )
                for cohort in ("P", "W", "A")
            }
            support[history_id] = {
                "role": manifest["role"],
                "files": len(rows),
                "failure_events": len(ledger),
                "positive_windows": positives,
                "controls": len(controls),
                "row_keys_allowlisted": all(
                    set(row) == B.ALLOWED_ROW_KEYS for row in rows),
            }
            assert len(controls) > 0
            assert all(support[history_id]["row_keys_allowlisted"] for _ in [0])

    metrics = {
        role: P.evaluate_histories(
            all_scores, rows_by_role[role], ledgers_by_role[role],
            windows_by_role[role], threshold,
        )
        for role in ("development", "confirmation")
    }
    for role_result in metrics.values():
        assert role_result["macro_auc_pw"] == 1.0
        assert role_result["macro_auc_p"] == 1.0
        assert role_result["macro_auc_w"] == 1.0
        assert role_result["lcb_pw"] == 1.0
        assert role_result["directional_histories"] == 4
        assert all(item["auc_pw"] == item["auc_p"] == item["auc_w"] == 1.0
                   for item in role_result["per_history"])
        assert all(item["recall_p"] == item["recall_w"] == 1.0
                   for item in role_result["per_history"])

    positive_file_ids: set[str] = set()
    for role in ("development", "confirmation"):
        for rows, ledger, windows in zip(
                rows_by_role[role], ledgers_by_role[role],
                windows_by_role[role], strict=True):
            for failure in ledger:
                if failure["cohort"] in ("P", "W"):
                    positive_file_ids.update(
                        row["file_id"] for row in E.pos_files(rows, failure, windows)
                    )
            for row in rows:
                score = all_scores[row["file_id"]]
                if row["file_id"] in positive_file_ids:
                    assert score > threshold
                else:
                    assert abs(score) <= 1e-9

    assert all(len(rows_by_role[role]) == 4
               for role in ("development", "confirmation"))
    assert len(set(fit_ids) & set(calibration_ids)) == 0
    result = {
        "task_id": "S18-T67",
        "attempt_id": "S18-T67-A03",
        "contract_id": "sprint18-iterative-data-contract-v2",
        "fixture_id": spec["fixture_id"],
        "fixture_contract_id": spec["contract_id"],
        "fixture_descriptor_sha256": descriptor_sha,
        "fixture_input_source": "accepted Task 67 v1 Fixture C complete-inputs.json, byte-identical frozen inherited descriptor/role rows",
        "fixture_input_sha256": sha256(raw_input),
        "waveform_sha256": spec["waveform_sha256"],
        "waveform_assignments_valid": True,
        "feature_count": len(P.feature_names()),
        "fit_file_ids": fit_ids,
        "fit_standard_deviation": stats["std"].tolist(),
        "fit_centroid_max_abs": float(np.max(np.abs(centroid))),
        "calibration_file_ids": calibration_ids,
        "calibration_hygiene": calibration_hygiene,
        "calibration_scores": calibration_scores.tolist(),
        "threshold_quantile": P.THRESHOLD_QUANTILE,
        "threshold": threshold,
        "development_and_confirmation_metrics": metrics,
        "role_support": support,
        "result_sha256": None,
        "limitations": [
            "Fixture C is the frozen deterministic in-memory probe input, not a newly generated real Fit/Calibration/Development/Confirmation role roster.",
            "No new role-specific generation coordinates or roots are authorized by Task 67; actual real-role fit/calibration/probe qualification remains Task 69 acceptance work.",
        ],
    }
    write_json(output_root / "results.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser_a = subparsers.add_parser("fixture-a")
    parser_a.add_argument("--root", type=Path, required=True)
    parser_a.add_argument("--seed", type=int, required=True)
    parser_a.add_argument("--reload-number", type=int, choices=(1, 2), required=True)
    parser_compare = subparsers.add_parser("compare-reloads")
    parser_compare.add_argument("--root", type=Path, required=True)
    subparsers.add_parser("fixture-b")
    subparsers.add_parser("fixture-c")
    args = parser.parse_args()
    if args.command == "fixture-a":
        fixture_a(args.root, args.seed, args.reload_number)
    elif args.command == "compare-reloads":
        compare_reloads(args.root)
    elif args.command == "fixture-b":
        fixture_b()
    else:
        fixture_c()


if __name__ == "__main__":
    main()
