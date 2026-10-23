"""Focused Task 30 N4-CRN runner contract tests.

Covers consumer-visible behavior only: the final all-Fit residual-bound
computation (Fit-only quantiles, frozen floor/cap edges, canonical record
digest), and the paired Development aggregation (same-seed deltas, positive
seed counts, and the exactly-zero ``S_pop`` delta that the unchanged-B0 branch
guarantees). The full certified-DATA path is exercised by the runner itself;
these tests use tiny synthetic record/metric shapes, never certified evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from representation.crn import N4ScaleRecord  # noqa: E402

import sprint18_task30_n4_runner as runner  # noqa: E402

HISTORIES = ("S18I-ITER-0005-FIT-01", "S18I-ITER-0005-FIT-02", "S18I-ITER-0005-FIT-03")


def _records(history_id: str, residuals: list[float]) -> list[N4ScaleRecord]:
    return [
        N4ScaleRecord(
            history_id=history_id,
            file_id=f"file-{index:03d}",
            patch_index=index,
            context=torch.zeros(128, dtype=torch.float32),
            residual=float(value),
        )
        for index, value in enumerate(residuals)
    ]


def test_final_bounds_use_only_fit_residuals_and_frozen_edges() -> None:
    pools = {
        HISTORIES[0]: _records(HISTORIES[0], [0.01, 0.02, 0.03, 0.04]),
        HISTORIES[1]: _records(HISTORIES[1], [0.05, 0.06, 0.07, 0.08]),
        HISTORIES[2]: _records(HISTORIES[2], [0.09, 0.10, 0.11, 0.12]),
    }
    bounds = runner._final_bounds(pools, list(HISTORIES))
    residuals = np.arange(1, 13, dtype=np.float64) / 100.0
    assert bounds["record_count"] == 12
    assert bounds["q05"] == float(np.quantile(residuals, 0.05, method="linear"))
    assert bounds["q95"] == float(np.quantile(residuals, 0.95, method="linear"))
    assert bounds["v_floor"] == max(bounds["q05"], 1e-6)
    assert bounds["v_cap"] == max(bounds["q95"], 10.0 * bounds["v_floor"])
    # Canonical residual digest covers every in-scope record exactly once.
    expected = hashlib.sha256(
        runner.json.dumps(
            [
                [history_id, record.file_id, int(record.patch_index), float(record.residual)]
                for history_id in HISTORIES
                for record in pools[history_id]
            ],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert bounds["residual_records_sha256"] == expected


def test_final_bounds_reject_non_fit_pool_geometry() -> None:
    with pytest.raises((ValueError, SystemExit)):
        runner._final_bounds({"H-X": []}, ["H-X"])


def _branch(macro_values: dict[str, float], history_values: dict[str, float], threshold: float) -> dict:
    per_history = []
    for ordinal in range(1, 5):
        history_id = f"S18I-ITER-0005-DEVELOPMENT-{ordinal:02d}"
        per_history.append({"history_id": history_id, **history_values})
    return {"threshold": threshold, "macro": dict(macro_values), "per_history": per_history}


def _seed_record(seed: int, *, arm: bool, support: str, bank_sha: str, macro_shift: float) -> dict:
    histories = [f"S18I-ITER-0005-DEVELOPMENT-{ordinal:02d}" for ordinal in range(1, 5)]
    macro = {name: 0.5 + (macro_shift if arm else 0.0) for name in runner.PAIRED_MACRO_METRICS}
    history_values = {
        name: 0.5 + (macro_shift if arm else 0.0) for name in runner.PAIRED_HISTORY_METRICS
    }
    branches = {
        "S_pred": _branch(macro, history_values, threshold=0.1 + (0.01 if arm else 0.0)),
        # The N4 S_pop branch is the unchanged B0 path: identical values.
        "S_pop": _branch(
            {name: 0.5 for name in runner.PAIRED_MACRO_METRICS},
            {name: 0.5 for name in runner.PAIRED_HISTORY_METRICS},
            threshold=0.2,
        ),
    }
    record = {
        "model_seed": seed,
        "arm_id": "N4-CRN" if arm else "B0-FULL",
        "status": "ok",
        "bank": {"source_rows": 5446, "bank_sha256": bank_sha},
        "calibration": {
            "source_support_sha256": support,
        },
        "development": {
            "support": {
                "evaluation_histories": histories,
                "per_history": {history_id: {"support_sha256": support} for history_id in histories}
                if not arm
                else {},
            },
            "branches": branches,
        },
    }
    if arm:
        record["calibration"] = {"support": {"support_sha256": support}}
        record["development"]["support"] = {
            "evaluation_histories": histories,
            "per_history_support_sha256": {history_id: support for history_id in histories},
        }
    return record


def test_paired_development_aggregates_deltas_and_zero_s_pop() -> None:
    support = "ab" * 32
    bank_sha = "cd" * 32
    arm_records = {
        str(seed): _seed_record(seed, arm=True, support=support, bank_sha=bank_sha, macro_shift=0.25)
        for seed in (181801, 181802, 181803)
    }
    b0_records = {
        str(seed): _seed_record(seed, arm=False, support=support, bank_sha=bank_sha, macro_shift=0.0)
        for seed in (181801, 181802, 181803)
    }
    paired = runner._paired_development(
        arm_records=arm_records, b0_records=b0_records, b0_records_sha256="ef" * 32,
    )
    assert paired["schema"] == runner.PAIRED_SCHEMA
    assert paired["b0_seed_records_sha256"] == "ef" * 32
    for branch in ("S_pred", "S_pop"):
        aggregate = paired["branches"][branch]["aggregate"]["macro"]
        for name in runner.PAIRED_MACRO_METRICS:
            assert set(aggregate[name]) == {"mean_delta", "positive_seeds"}
    pred = paired["branches"]["S_pred"]["aggregate"]["macro"]
    assert pred["PW_macro"]["mean_delta"] == pytest.approx(0.25)
    assert pred["PW_macro"]["positive_seeds"] == 3
    # S_pop is the unchanged B0 path: deltas are exactly zero, by construction.
    pop = paired["branches"]["S_pop"]["aggregate"]["macro"]
    assert pop["PW_macro"]["mean_delta"] == 0.0
    assert pop["PW_macro"]["positive_seeds"] == 0
    pop_history = paired["branches"]["S_pop"]["aggregate"]["per_history"]
    for history_id, metrics in pop_history.items():
        for name, value in metrics.items():
            assert value["mean_delta"] == 0.0
            assert value["positive_seeds"] == 0


def test_paired_development_rejects_support_or_bank_mismatch() -> None:
    support = "ab" * 32
    bank_sha = "cd" * 32
    arm_records = {
        str(seed): _seed_record(seed, arm=True, support=support, bank_sha=bank_sha, macro_shift=0.1)
        for seed in (181801, 181802, 181803)
    }
    b0_records = {
        str(seed): _seed_record(seed, arm=False, support=support, bank_sha=bank_sha, macro_shift=0.0)
        for seed in (181801, 181802, 181803)
    }
    drifted = {key: dict(value) for key, value in b0_records.items()}
    value_802 = b0_records["181802"]
    drifted["181802"] = {**value_802, "bank": {**value_802["bank"], "bank_sha256": "00" * 32}}
    with pytest.raises(SystemExit):
        runner._paired_development(
            arm_records=arm_records, b0_records=drifted, b0_records_sha256="ef" * 32,
        )
    drifted_support = {key: dict(value) for key, value in b0_records.items()}
    value_803 = b0_records["181803"]
    drifted_support["181803"] = {
        **value_803,
        "calibration": {**value_803["calibration"], "source_support_sha256": "11" * 32},
    }
    with pytest.raises(SystemExit):
        runner._paired_development(
            arm_records=arm_records, b0_records=drifted_support, b0_records_sha256="ef" * 32,
        )


def test_n4_score_row_declares_consumer_triple_and_patch_diagnostics() -> None:
    from representation.config import V1Config
    from representation.crn import init_n4_scale_model
    from representation.model import V1RepresentationModel
    from representation.sprint18_harness import deterministic_mask_seed
    from representation import n4_score_files
    from synth.config import PatchConfig
    from synth.patchify import Patchifier
    from synth.schema import FileSample, SampleLabel

    config = V1Config(
        n_channels=6, patch_size=32, stride=16, d_model=16, attention_heads=4,
        sequence_layers=1, dropout=0.0, n_robots=9, n_programs=8, seed=181801,
        contrastive_weight_max=0.05, ema_decay=0.996,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    scale = init_n4_scale_model(config.d_model, 181801)
    with torch.no_grad():
        scale.out.bias.fill_(math.log(1e-6))  # every patch clamps at v_floor
    rng = np.random.default_rng(91813)
    sample = FileSample(
        x=rng.normal(size=(6, 640)).astype(np.float32), file_id="t30-diag-00",
        file_label=SampleLabel.NORMAL, seed=181801, generator_version="t30-test",
        config_hash="t30-test", regime_sequence=[], robot_idx=0, program_idx=0,
    )
    sample.validate()

    from types import SimpleNamespace

    row = SimpleNamespace(file_id=sample.file_id, sample=sample)
    bounds = {"v_floor": 0.02, "v_cap": 1.4}
    mask_seed = deterministic_mask_seed(181801, "S18I-ITER-0005-FIT-01", sample.file_id)
    triples = n4_score_files(
        model=model, samples=[sample], scale_model=scale,
        v_floor=0.02, v_cap=1.4, patchifier=patchifier, masking_seed=mask_seed,
    )
    entry = runner._n4_score_row(
        model=model, scale=scale, row=row, history_id="S18I-ITER-0005-FIT-01",
        seed=181801, patchifier=patchifier, bounds=bounds, diagnostics=True,
    )
    triple = triples[sample.file_id]
    assert entry["file_id"] == sample.file_id
    assert entry["raw"] == triple["raw"]
    assert entry["scale"] == triple["scale"]
    assert entry["normalized"] == triple["normalized"]
    assert entry["eligible_patches"] >= 1
    assert entry["frac_at_floor"] == 1.0
    assert entry["frac_at_cap"] == 0.0
    assert entry["scale"] == pytest.approx(0.02)


def _write_prediction_cache(path: Path, records: list[dict]) -> tuple[str, str]:
    """Serialize a real Task23-schema cache and measure both of its identities."""
    inner_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": "s18-t23-c01-prediction-cache-v1",
        "cache_sha256": inner_sha256,
        "records": records,
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest(), inner_sha256


def _cache_binding_fixture(tmp_path: Path) -> tuple[dict, dict, Path]:
    """Build the exact corrected binding shape around a real serialized cache."""
    cache_path = tmp_path / "prediction_cache.json"
    file_sha256, inner_sha256 = _write_prediction_cache(
        cache_path,
        [
            {
                "file_id": "tiny-file",
                "actual_history_id": HISTORIES[0],
                "role": "FIT",
                "patch_mapping": {"eligible_patch_index": [0]},
            }
        ],
    )
    checkpoint_path = tmp_path / "b0-epoch100.pt"
    checkpoint_path.write_bytes(b"bound B0 checkpoint bytes")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    seeds = [str(seed) for seed in runner.MODEL_SEEDS]
    live_model = {seed: "a" * 64 for seed in seeds}
    live_ema = {seed: "b" * 64 for seed in seeds}
    fixture = {
        "checkpoints": {
            seed: {"path": str(checkpoint_path), "sha256": checkpoint_sha256} for seed in seeds
        },
        "ema_state_sha256": dict(live_ema),
        "live_model_state_sha256": dict(live_model),
        "prediction_cache_paths": {seed: str(cache_path) for seed in seeds},
        runner.PREDICTION_CACHE_FILE_SHA256_KEY: {seed: file_sha256 for seed in seeds},
        runner.PREDICTION_CACHE_INNER_SHA256_KEY: {seed: inner_sha256 for seed in seeds},
    }
    seed = runner.MODEL_SEEDS[0]
    key = str(seed)
    b0_record = {
        "arm_id": runner.BASELINE_ARM_ID,
        "status": "ok",
        "representation_epochs_completed": 100,
        "primary": {
            "restored_checkpoint_sha256": checkpoint_sha256,
            "live_model_state_sha256": live_model[key],
            "live_ema_state_sha256": live_ema[key],
        },
        "prediction_cache": {
            "path": str(cache_path),
            "cache_sha256": inner_sha256,
            "records": 1,
        },
        "bank": {
            "source_rows": runner.FIT_HEALTHY_ROWS,
            "bank_sha256": "c" * 64,
            "source_rows_sha256": "d" * 64,
        },
        "calibration": {
            "source_rows": runner.CALIBRATION_HEALTHY_ROWS,
            "source_support_sha256": "e" * 64,
            "S_pred_threshold": 0.1,
            "S_pop_threshold": 0.2,
        },
    }
    return fixture, b0_record, cache_path


def test_b0_cache_identity_accepts_distinct_file_and_head_digests(tmp_path: Path) -> None:
    """The accepted Task23 shape: the raw file digest and the head digest differ."""
    fixture, b0_record, _ = _cache_binding_fixture(tmp_path)
    seed = runner.MODEL_SEEDS[0]
    assert runner._validate_fixture(fixture) is fixture
    assert (
        fixture[runner.PREDICTION_CACHE_FILE_SHA256_KEY][str(seed)]
        != fixture[runner.PREDICTION_CACHE_INNER_SHA256_KEY][str(seed)]
    )
    restored_identity = runner._b0_restore_identity(
        fixture=fixture, seed=seed, b0_record=b0_record,
    )
    identity = restored_identity["identity"]
    assert identity["cache_file_sha256"] == fixture[runner.PREDICTION_CACHE_FILE_SHA256_KEY][str(seed)]
    assert identity["cache_inner_sha256"] == fixture[runner.PREDICTION_CACHE_INNER_SHA256_KEY][str(seed)]
    assert identity["cache_file_sha256"] != identity["cache_inner_sha256"]
    assert identity["cache_records"] == 1
    assert identity["checkpoint_sha256"] == fixture["checkpoints"][str(seed)]["sha256"]
    assert restored_identity["checkpoint_path"] == fixture["checkpoints"][str(seed)]["path"]


def test_b0_cache_identity_refuses_each_tampered_identity_separately(tmp_path: Path) -> None:
    """A swapped, missing or corrupted either identity fails closed on its own gate."""
    seed = runner.MODEL_SEEDS[0]
    key = str(seed)
    fixture, b0_record, cache_path = _cache_binding_fixture(tmp_path)
    file_sha256 = fixture[runner.PREDICTION_CACHE_FILE_SHA256_KEY][key]
    inner_sha256 = fixture[runner.PREDICTION_CACHE_INNER_SHA256_KEY][key]

    file_as_inner = json.loads(json.dumps(fixture))
    file_as_inner[runner.PREDICTION_CACHE_INNER_SHA256_KEY][key] = file_sha256
    with pytest.raises(SystemExit, match="head digest differs from the bound inner digest"):
        runner._b0_restore_identity(fixture=file_as_inner, seed=seed, b0_record=b0_record)

    inner_as_file = json.loads(json.dumps(fixture))
    inner_as_file[runner.PREDICTION_CACHE_FILE_SHA256_KEY][key] = inner_sha256
    with pytest.raises(SystemExit, match="cache bytes differ from the bound file digest"):
        runner._b0_restore_identity(fixture=inner_as_file, seed=seed, b0_record=b0_record)

    record_mismatch = json.loads(json.dumps(b0_record))
    record_mismatch["prediction_cache"]["cache_sha256"] = file_sha256
    with pytest.raises(SystemExit, match="inner digest differs from the accepted B0 record"):
        runner._b0_restore_identity(fixture=fixture, seed=seed, b0_record=record_mismatch)

    cache_path.write_bytes(cache_path.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="cache bytes differ from the bound file digest"):
        runner._b0_restore_identity(fixture=fixture, seed=seed, b0_record=b0_record)


def test_b0_cache_identity_refuses_legacy_single_key_and_checkpoint_drift(tmp_path: Path) -> None:
    seed = runner.MODEL_SEEDS[0]
    key = str(seed)
    fixture, b0_record, _ = _cache_binding_fixture(tmp_path)

    legacy = json.loads(json.dumps(fixture))
    legacy[runner.AMBIGUOUS_PREDICTION_CACHE_KEY] = legacy.pop(
        runner.PREDICTION_CACHE_INNER_SHA256_KEY
    )
    with pytest.raises(SystemExit, match="ambiguous legacy key"):
        runner._validate_fixture(legacy)

    duplicated = json.loads(json.dumps(fixture))
    duplicated[runner.AMBIGUOUS_PREDICTION_CACHE_KEY] = duplicated[
        runner.PREDICTION_CACHE_INNER_SHA256_KEY
    ]
    with pytest.raises(SystemExit, match="ambiguous legacy key"):
        runner._validate_fixture(duplicated)

    missing = json.loads(json.dumps(fixture))
    missing[runner.PREDICTION_CACHE_FILE_SHA256_KEY].pop(key)
    with pytest.raises(SystemExit, match="does not cover exactly the frozen model seeds"):
        runner._validate_fixture(missing)

    malformed = json.loads(json.dumps(fixture))
    malformed[runner.PREDICTION_CACHE_FILE_SHA256_KEY][key] = "not-a-digest"
    with pytest.raises(SystemExit, match="is not a SHA-256 hex digest"):
        runner._validate_fixture(malformed)

    checkpoint = Path(fixture["checkpoints"][key]["path"])
    checkpoint.write_bytes(b"different checkpoint bytes")
    with pytest.raises(SystemExit, match="B0 checkpoint bytes differ from the bound digest"):
        runner._b0_restore_identity(fixture=fixture, seed=seed, b0_record=b0_record)
