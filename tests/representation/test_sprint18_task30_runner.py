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
import random
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _tiny_n4_pools() -> dict[str, list[N4ScaleRecord]]:
    """Tiny deterministic Fit pools whose residual bounds pass the frozen guards."""
    pools: dict[str, list[N4ScaleRecord]] = {}
    for ordinal, history_id in enumerate(HISTORIES):
        generator = torch.Generator().manual_seed(181801 + ordinal)
        pools[history_id] = [
            N4ScaleRecord(
                history_id=history_id,
                file_id=f"probe-{ordinal}-{index:03d}",
                patch_index=0,
                context=torch.randn(128, generator=generator) * 0.01,
                residual=0.08 + 0.04 * index,
            )
            for index in range(6)
        ]
    return pools


def _fit_binding_stub() -> SimpleNamespace:
    """Minimal Fit-role binding for the sampler identity only.

    The device-alignment regression exercises the frozen fit/probe mechanics,
    not the certified DATA binding: ``deterministic_sampler_order`` reads only
    the Fit history roster for the N4 scale workload.
    """
    histories = tuple(
        SimpleNamespace(actual_history_id=history_id, assigned_role="FIT")
        for history_id in HISTORIES
    )
    return SimpleNamespace(
        role_histories=lambda role: tuple(
            item for item in histories if item.assigned_role == role
        )
    )


#: The probe restoration must run on the head's real device. CUDA is the case
#: the CPU-only suites could never reach, so it is exercised when available and
#: honestly skipped otherwise.
DEVICE_CASES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="device-alignment regression needs a real CUDA device",
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICE_CASES)
def test_final_scale_restore_probe_forwards_on_the_live_head_device(
    tmp_path: Path, device: str
) -> None:
    """The strict final-envelope probe must not forward CPU rows into a CUDA head."""
    seed = runner.MODEL_SEEDS[0]
    pools = _tiny_n4_pools()
    records = [record for history_id in HISTORIES for record in pools[history_id]]
    bounds = runner._final_bounds(pools, list(HISTORIES))
    scale = runner.init_n4_scale_model(128, seed)
    result = runner.fit_n4_scale_scope(
        records=records, training_history_ids=list(HISTORIES), scope="final",
        arm=runner._n4_arm(seed, "ab" * 32), binding=_fit_binding_stub(),
        scale_model=scale, v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
        device=device,
    )
    block = runner._save_and_restore_scale(
        scale=scale, out_dir=tmp_path, seed=seed, fit_history_ids=list(HISTORIES),
        bounds=bounds, fit_result={**result, "probe_records": records},
    )
    assert block["restored_forward_max_abs_diff"] == 0.0
    assert block["probe_records"] == len(records)
    assert math.isfinite(block["restored_probe_nll"])
    assert block["restored_state_dict_sha256"] == block["state_dict_sha256"]


@pytest.mark.parametrize("device", DEVICE_CASES)
def test_capacity_fold_restore_probe_forwards_on_the_live_head_device(
    tmp_path: Path, device: str
) -> None:
    """Each fold strict-restore probe must align CPU rows with the CUDA head."""
    seed = runner.MODEL_SEEDS[0]
    block = runner._run_capacity_folds(
        pools=_tiny_n4_pools(), fit_history_ids=list(HISTORIES),
        arm=runner._n4_arm(seed, "ab" * 32), binding=_fit_binding_stub(),
        seed=seed, device=torch.device(device), out_dir=tmp_path,
        certified_folds=[], normative=False,
    )
    assert block["folds_evaluated"] == len(HISTORIES)
    assert block["all_fold_cap_gates_pass"] is True
    assert block["all_fold_payloads_match_certified_task13b"] is None
    assert block["all_fold_custody_authenticated"] is None
    for fold in block["folds"]:
        assert fold["restored_forward_max_abs_diff"] == 0.0
        assert fold["restored_state_dict_sha256"] == fold["scale_state_sha256"]
        assert fold["certified_comparison"] == "not_compared_nonnormative"
        assert fold["certified_payload_matches"] is None
        assert fold["certified_custody_authenticated"] is False
        assert Path(fold["scale_checkpoint_path"]).is_file()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_certified_folds(root: Path) -> list[dict]:
    """Fit one real tiny fold per Fit history and build its certified record.

    Each v3 envelope is written at its provenance path and then re-serialized
    with the identical semantic payload into a different ``torch.save``
    container (legacy format), so the raw file digest changes while the
    canonical payload digest does not. That is the acceptance half of the
    certified-capacity contract: custody by raw digest, identity by payload.
    """
    seed = runner.MODEL_SEEDS[0]
    pools = _tiny_n4_pools()
    records: list[dict] = []
    for index, heldout in enumerate(HISTORIES):
        training_ids = [history_id for history_id in HISTORIES if history_id != heldout]
        train_pool = [record for history_id in training_ids for record in pools[history_id]]
        heldout_pool = list(pools[heldout])
        residuals = np.array([record.residual for record in train_pool], dtype=np.float64)
        bounds = runner.n4_residual_bounds(residuals)
        q05 = float(np.quantile(residuals, 0.05, method="linear"))
        q95 = float(np.quantile(residuals, 0.95, method="linear"))
        scale = runner.init_n4_scale_model(128, seed)
        result = runner.fit_n4_scale_scope(
            records=train_pool, training_history_ids=training_ids,
            scope=f"fold:{heldout}", arm=runner._n4_arm(seed, "ab" * 32),
            binding=_fit_binding_stub(), scale_model=scale,
            v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
            heldout_records=heldout_pool, device="cpu",
        )
        provenance_path = root / "provenance" / f"seed{seed}-fold{index}.pt"
        runner.save_n4_scale_checkpoint(
            provenance_path, scale, fit_scope=f"fold:{heldout}",
            training_history_ids=training_ids,
            v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
            initialization={
                "init_anchor": dict(result["init_anchor"]),
                "init_interiority": dict(result["init_interiority"]),
                "first_step_gradients": dict(result["first_step_gradients"]),
            },
        )
        payload = torch.load(provenance_path, map_location="cpu", weights_only=True)
        historical_path = root / "historical" / provenance_path.name
        historical_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, historical_path, _use_new_zipfile_serialization=False)
        container_sha = _sha256_file(historical_path)
        assert container_sha != _sha256_file(provenance_path)
        records.append(
            {
                "model_seed": seed,
                "fold_index": index,
                "scope": f"fold:{heldout}",
                "heldout_history_id": heldout,
                "training_history_ids": training_ids,
                "checkpoint_path": f"checkpoints/{historical_path.name}",
                "envelope_path": str(historical_path),
                "checkpoint_sha256": container_sha,
                "scale_state_sha256": payload["state_dict_sha256"],
                "checkpoint_schema": payload["schema"],
                "checkpoint_parameterization": payload["parameterization"],
                "scale_parameterization": runner.N4_SCALE_PARAMETERIZATION,
                "guard_contract": payload["guard_contract"],
                "scale_parameters": int(sum(parameter.numel() for parameter in scale.parameters())),
                "record_count": int(result["record_count"]),
                "heldout_count": int(result["heldout_count"]),
                "optimizer_steps": int(result["optimizer_steps"]),
                "per_epoch_steps": int(result["per_epoch_steps"]),
                "bounds": {
                    "v_floor": float(bounds["v_floor"]),
                    "v_cap": float(bounds["v_cap"]),
                },
                "quantiles": {"q05": q05, "q95": q95},
                "init_anchor": dict(result["init_anchor"]),
                "init_interiority": dict(result["init_interiority"]),
                "first_step_gradients": dict(result["first_step_gradients"]),
                "epoch_order_seeds": list(result["epoch_order_seeds"]),
                "order_seed_rule": str(result["order_seed_rule"]),
                "cap_gate_pass_train": bool(result["cap_gate_pass_train"]),
            }
        )
    return records


def _run_certified_tiny_folds(tmp_path: Path, certified_folds: list[dict]) -> dict:
    seed = runner.MODEL_SEEDS[0]
    return runner._run_capacity_folds(
        pools=_tiny_n4_pools(), fit_history_ids=list(HISTORIES),
        arm=runner._n4_arm(seed, "ab" * 32), binding=_fit_binding_stub(),
        seed=seed, device=torch.device("cpu"), out_dir=tmp_path / "recertified",
        certified_folds=certified_folds, normative=True,
    )


def test_capacity_fold_certification_authenticates_payload_not_container_bytes(
    tmp_path: Path,
) -> None:
    """An independently serialized identical payload must be accepted.

    The historical envelopes carry the same canonical scale state and recipe
    in different ``torch.save`` container bytes, so the raw digests differ.
    Treating container bytes as model identity would wrongly refuse this; the
    corrected contract authenticates the file and compares the payload, and it
    records the container difference as custody evidence only.
    """
    certified = _tiny_certified_folds(tmp_path)
    block = _run_certified_tiny_folds(tmp_path, certified)
    assert block["all_fold_custody_authenticated"] is True
    assert block["all_fold_payloads_match_certified_task13b"] is True
    assert block["recertified_containers_equal_certified"] is False
    for fold, record in zip(block["folds"], certified):
        assert fold["certified_container_matches"] is False
        assert fold["certified_payload_matches"] is True
        assert fold["certified_container_sha256"] == record["checkpoint_sha256"]
        assert fold["certified_payload_sha256"] == fold["scale_state_sha256"]
        assert fold["certified_comparison"] == (
            "authenticated_historical_container_and_payload_with_exact_identity"
        )


def test_capacity_fold_certification_refuses_scientific_tamper(tmp_path: Path) -> None:
    """Changed weights, bounds, support or custody are refused, not tolerated."""
    # A coherently re-digested envelope whose scale weights differ: custody and
    # the envelope's own digest authenticate, yet the recertified payload is a
    # different scientific result.
    certified = _tiny_certified_folds(tmp_path / "weights")
    record = certified[0]
    payload = torch.load(record["envelope_path"], map_location="cpu", weights_only=True)
    state = dict(payload["state_dict"])
    name = sorted(state)[0]
    state[name] = state[name] + 1e-3
    payload["state_dict"] = state
    payload["state_dict_sha256"] = runner.state_sha256(state)
    torch.save(payload, record["envelope_path"])
    record["checkpoint_sha256"] = _sha256_file(Path(record["envelope_path"]))
    record["scale_state_sha256"] = payload["state_dict_sha256"]
    with pytest.raises(SystemExit, match="recertified scale state .* differs"):
        _run_certified_tiny_folds(tmp_path / "weights-out", certified)

    # Changed bound (quantile) identity in the certified fold record.
    certified = _tiny_certified_folds(tmp_path / "bounds")
    certified[0]["quantiles"]["q05"] = certified[0]["quantiles"]["q05"] * 1.01
    with pytest.raises(SystemExit, match="recertified quantiles differs"):
        _run_certified_tiny_folds(tmp_path / "bounds-out", certified)

    # Changed qualified-support identity in the certified fold record.
    certified = _tiny_certified_folds(tmp_path / "support")
    certified[0]["record_count"] = certified[0]["record_count"] + 1
    with pytest.raises(SystemExit, match="recertified record_count differs"):
        _run_certified_tiny_folds(tmp_path / "support-out", certified)

    # Substituted historical bytes that no longer authenticate against custody.
    certified = _tiny_certified_folds(tmp_path / "custody")
    Path(certified[0]["envelope_path"]).write_bytes(b"not the certified envelope")
    with pytest.raises(SystemExit, match="do not authenticate against the certified container digest"):
        _run_certified_tiny_folds(tmp_path / "custody-out", certified)

    # Unreachable historical custody root is a stated prerequisite, not a pass.
    certified = _tiny_certified_folds(tmp_path / "missing")
    Path(certified[0]["envelope_path"]).unlink()
    with pytest.raises(SystemExit, match="certified historical envelope is unreachable"):
        _run_certified_tiny_folds(tmp_path / "missing-out", certified)


def test_saved_scale_envelope_restores_and_scores_through_public_consumer(
    tmp_path: Path,
) -> None:
    """A real saved envelope restores into a fresh head that drives public scoring."""
    from representation import n4_score_files
    from representation.config import V1Config
    from representation.model import V1RepresentationModel
    from representation.sprint18_harness import deterministic_mask_seed
    from synth.config import PatchConfig
    from synth.patchify import Patchifier
    from synth.schema import FileSample, SampleLabel

    seed = runner.MODEL_SEEDS[0]
    pools = _tiny_n4_pools()
    records = [record for history_id in HISTORIES for record in pools[history_id]]
    bounds = runner._final_bounds(pools, list(HISTORIES))
    scale = runner.init_n4_scale_model(128, seed)
    result = runner.fit_n4_scale_scope(
        records=records, training_history_ids=list(HISTORIES), scope="final",
        arm=runner._n4_arm(seed, "ab" * 32), binding=_fit_binding_stub(),
        scale_model=scale, v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
    )
    block = runner._save_and_restore_scale(
        scale=scale, out_dir=tmp_path, seed=seed, fit_history_ids=list(HISTORIES),
        bounds=bounds, fit_result={**result, "probe_records": records},
    )
    checkpoint_path = Path(block["path"])
    assert block["sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    restored = runner.init_n4_scale_model(128, seed + 999999)
    meta = runner.load_n4_scale_checkpoint(
        checkpoint_path, restored, expected_fit_scope="final",
        expected_training_history_ids=list(HISTORIES),
        expected_bounds={"v_floor": bounds["v_floor"], "v_cap": bounds["v_cap"]},
    )
    assert meta["state_dict_sha256"] == block["state_dict_sha256"]

    config = V1Config(
        n_channels=6, patch_size=32, stride=16, d_model=128, attention_heads=4,
        sequence_layers=1, dropout=0.0, n_robots=9, n_programs=8, seed=seed,
        contrastive_weight_max=0.05, ema_decay=0.996,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    sample = FileSample(
        x=np.random.default_rng(181801).normal(size=(6, 640)).astype(np.float32),
        file_id="t30-envelope-00", file_label=SampleLabel.NORMAL, seed=seed,
        generator_version="t30-test", config_hash="t30-test", regime_sequence=[],
        robot_idx=0, program_idx=0,
    )
    sample.validate()
    triples = n4_score_files(
        model=model, samples=[sample], scale_model=restored,
        v_floor=bounds["v_floor"], v_cap=bounds["v_cap"], patchifier=patchifier,
        masking_seed=deterministic_mask_seed(seed, HISTORIES[0], sample.file_id),
    )
    triple = triples[sample.file_id]
    assert set(triple) == {"raw", "scale", "normalized"}
    assert all(math.isfinite(float(value)) for value in triple.values())
    assert float(triple["scale"]) > 0.0


def _tiny_bank_binding(root: Path):
    """Bind three tiny Fit histories and one Calibration history locally.

    File IDs are interleaved across the Fit histories so the certified loader's
    per-history row order differs from the canonical global ``file_id`` order.
    """
    import zipfile

    from representation.sprint18_harness import BoundHistory, Task6Binding
    from synth.dataset import _sample_bytes
    from synth.schema import FileSample, SampleLabel

    generator = "s18-t30-bank-fixture"
    config_hash = "s18-t30-bank-config"
    layout = {
        "S18I-T30-FIT-01": ("FIT", ["t30-bank-02", "t30-bank-00"]),
        "S18I-T30-FIT-02": ("FIT", ["t30-bank-05", "t30-bank-01", "t30-bank-04"]),
        "S18I-T30-FIT-03": ("FIT", ["t30-bank-03"]),
        "S18I-T30-CAL-01": ("CALIBRATION", ["t30-bank-cal-01", "t30-bank-cal-00"]),
    }
    rng = np.random.default_rng(918081)
    signals = {
        file_id: rng.normal(size=(6, 256)).astype(np.float32)
        for _, file_ids in layout.values()
        for file_id in file_ids
    }

    def build(history_id: str, role: str, file_ids: list[str]):
        relative = f"{history_id}/shard-00000.zip"
        shard_path = root / relative
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        items = []
        for offset, file_id in enumerate(file_ids):
            sample = FileSample(
                x=signals[file_id], file_id=file_id, file_label=SampleLabel.NORMAL,
                seed=runner.MODEL_SEEDS[0] + offset, generator_version=generator,
                config_hash=config_hash, regime_sequence=[], robot_idx=offset % 3,
                program_idx=offset % 2,
            )
            sample.validate()
            items.append(sample)
        with zipfile.ZipFile(shard_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in items:
                archive.writestr(f"{item.file_id}.npz", _sample_bytes(item))
        manifest = {
            "format": 1,
            "role": history_id,
            "protocol": "sprint15-benchmark-protocol-v7",
            "config_hash": config_hash,
            "files": [
                {
                    "file_id": item.file_id,
                    "file_label": item.file_label.value,
                    "robot_id": f"robot-{item.robot_idx + 1:02d}",
                    "program_id": f"program-{item.program_idx + 1:02d}",
                    "is_quarantined": False,
                    "is_censored": False,
                    "start_time": 0.0,
                    "end_time": 3600.0,
                    "n_valid_patches": 13,
                }
                for item in items
            ],
            "shards": [
                {
                    "path": relative,
                    "file_ids": [item.file_id for item in items],
                    "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                }
            ],
            "maintenance_windows": {},
        }
        manifest_path = root / f"{history_id}.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return SimpleNamespace(
            history=BoundHistory(
                assigned_role=role,
                actual_history_id=history_id,
                schema_history_id=f"H-TINY-{history_id}",
                manifest_sha256=digest,
                data_seed=32064,
                sample_count=len(items),
                config_hash=config_hash,
                relative_path=f"tiny-fixture/{role}/{history_id}",
                data_permitted_use=role,
                model_permitted_use=role,
            ),
            manifest_path=manifest_path,
            shard_paths={relative: shard_path},
            eligible_ids=tuple(file_ids),
        )

    entries = {
        history_id: build(history_id, role, file_ids)
        for history_id, (role, file_ids) in layout.items()
    }
    fit_entries = [entries[history_id] for history_id in (
        "S18I-T30-FIT-01", "S18I-T30-FIT-02", "S18I-T30-FIT-03",
    )]
    calibration_entry = entries["S18I-T30-CAL-01"]
    manifest_digests = "".join(
        [entry.history.manifest_sha256 for entry in fit_entries]
        + [calibration_entry.history.manifest_sha256]
    )
    binding = Task6Binding(
        candidate_id="S18-T30-A08-ORDERING-TINY",
        data_root_relative="tiny-fixture",
        handoff_sha256=hashlib.sha256(b"s18-t30-a08-bank-handoff").hexdigest(),
        data_binding_raw_sha256=hashlib.sha256(b"s18-t30-a08-bank-raw").hexdigest(),
        data_binding_canonical_sha256=hashlib.sha256(b"s18-t30-a08-bank-canonical").hexdigest(),
        whole16_certification_sha256=hashlib.sha256(b"s18-t30-a08-bank-whole16").hexdigest(),
        qualification_record_sha256=hashlib.sha256(b"s18-t30-a08-bank-qual").hexdigest(),
        manifest_aggregate_sha256=hashlib.sha256(manifest_digests.encode("ascii")).hexdigest(),
        protocol_sha256=hashlib.sha256(b"s18-t30-a08-bank-protocol").hexdigest(),
        mechanism_spec_sha256={},
        histories=tuple(entry.history for entry in entries.values()),
        fit_healthy_rows=sum(len(entry.eligible_ids) for entry in fit_entries),
        calibration_healthy_rows=len(calibration_entry.eligible_ids),
        executable_binding_sha256=hashlib.sha256(manifest_digests.encode("ascii")).hexdigest(),
    )
    return binding, fit_entries, calibration_entry


def test_fit_bank_rows_canonicalize_global_file_id_order(tmp_path: Path) -> None:
    """Per-history or shuffled Fit input must build the accepted canonical bank.

    ``fit_fit_reference_bank`` hashes the embedding matrix in row order and the
    accepted B0 bank was built from globally ``file_id``-sorted rows. This
    regression proves the runner's canonicalization helper yields that exact
    order, identical bank values/digest and identical branch scores and q95
    thresholds for arbitrary input permutations, while the raw per-history
    order (the A07 defect) differs.
    """
    from representation.config import V1Config
    from representation.data import collate_variable_files
    from representation.inference import RepresentationInference
    from representation.model import V1RepresentationModel
    from representation.sprint18_checkpoint import (
        SCHEDULER_ID,
        coefficient_for_epoch,
        deterministic_sampler_order,
        learning_rate_for_epoch,
        restore_sprint18_checkpoint,
        save_sprint18_checkpoint,
        state_sha256,
    )
    from representation.sprint18_harness import (
        BATCH_SIZE,
        _canonical_sha256,
        _encoder_batch,
        deterministic_mask_seed,
        fit_calibration_q95,
        fit_fit_reference_bank,
        load_certified_healthy_role_support,
        load_certified_sample_subset,
        make_sprint18_arm_config,
        score_certified_rows,
    )
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    seed = runner.MODEL_SEEDS[0]
    binding, fit_entries, calibration_entry = _tiny_bank_binding(tmp_path)
    per_history_rows = []
    for entry in fit_entries:
        per_history_rows.extend(load_certified_healthy_role_support(
            binding, entry.history.actual_history_id, entry.manifest_path,
            entry.shard_paths, action="train_fit",
        ))
    calibration_rows = load_certified_sample_subset(
        binding, calibration_entry.history.actual_history_id,
        calibration_entry.manifest_path, calibration_entry.shard_paths,
        list(calibration_entry.eligible_ids),
        role_action="calibrate_q95", verify_all_shards=True,
    )
    assert len(per_history_rows) == binding.fit_healthy_rows
    assert len(calibration_rows) == binding.calibration_healthy_rows

    canonical_ids = sorted(row.file_id for row in per_history_rows)
    assert [row.file_id for row in per_history_rows] != canonical_ids
    canonical = runner._canonical_fit_rows(per_history_rows)
    assert [row.file_id for row in canonical] == canonical_ids
    shuffled = list(per_history_rows)
    random.Random(202608).shuffle(shuffled)
    assert runner._canonical_fit_rows(shuffled) == canonical
    with pytest.raises(SystemExit, match="duplicate file IDs"):
        runner._canonical_fit_rows(per_history_rows + [per_history_rows[0]])

    arm = make_sprint18_arm_config(
        "B0-FULL", seed,
        {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="locked",
        coefficient_lock_sha256="6eb108b2073181d427f1aedfc81fb3bae4c29595108d15de486082800bdd9d6e",
    )
    config = V1Config(
        n_channels=6, patch_size=32, stride=16, d_model=32, attention_heads=4,
        sequence_layers=1, dropout=0.0, n_robots=9, n_programs=8, seed=seed,
        contrastive_weight_max=0.05, ema_decay=0.996,
    )
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0
    )
    fit_history_ids = [entry.history.actual_history_id for entry in fit_entries]
    source_ids = sorted(row.file_id for row in per_history_rows)
    sampler_seed, epoch_order = deterministic_sampler_order(
        binding, arm, workload="representation", scope="experiment",
        role_history_ids=fit_history_ids, epoch=1, eligible_source_ids=source_ids,
    )
    checkpoint_path = tmp_path / "tiny-b0-primary.pt"
    digest = save_sprint18_checkpoint(
        checkpoint_path, model, optimizer, arm=arm, binding=binding,
        global_step=0, completed_epoch=0,
        initialization_sha256=state_sha256(model.state_dict()),
        scheduler_state={
            "scheduler_id": SCHEDULER_ID, "completed_epochs": 0, "active_epoch": 1,
            "learning_rate": learning_rate_for_epoch(1), "global_step": 0,
        },
        sampler_state={
            "scope": "experiment", "workload": "representation",
            "role_history_ids": fit_history_ids,
            "role_manifest_sha256": [
                binding.history(history_id).manifest_sha256 for history_id in fit_history_ids
            ],
            "eligible_source_ids": source_ids,
            "eligible_source_ids_sha256": _canonical_sha256(source_ids),
            "sampler_epoch_seed": sampler_seed,
            "epoch_order_ids": list(epoch_order),
            "epoch_order_ids_sha256": _canonical_sha256(list(epoch_order)),
            "eligible_rows": len(source_ids),
            "batch_size": BATCH_SIZE,
            "batches_per_epoch": math.ceil(len(source_ids) / BATCH_SIZE),
            "completed_epochs": 0, "current_epoch": 1,
            "batches_in_current_epoch": 0,
            "fit_role_sha256": _canonical_sha256(
                sorted(item.manifest_sha256 for item in binding.role_histories("FIT"))
            ),
        },
        coefficient_state={
            "epoch": 1, "global_step": 0, "selection_state": "locked",
            "lambda_info": coefficient_for_epoch(0.05, 1),
            "lambda_hcc": None, "alpha": None, "beta": None,
        },
        scope="experiment",
    )
    restored_model = V1RepresentationModel(config, patchifier=patchifier)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0
    )
    restored = restore_sprint18_checkpoint(
        checkpoint_path, restored_model, restored_optimizer, arm=arm,
        binding=binding, allowed_scope="experiment",
    )
    assert restored.checkpoint_sha256 == digest

    bank = fit_fit_reference_bank(
        binding, arm, restored_model, canonical, patchifier,
        checkpoint=restored, scope="experiment",
    )
    bank_replayed = fit_fit_reference_bank(
        binding, arm, restored_model, runner._canonical_fit_rows(shuffled), patchifier,
        checkpoint=restored, scope="experiment",
    )
    bank_raw = fit_fit_reference_bank(
        binding, arm, restored_model, list(per_history_rows), patchifier,
        checkpoint=restored, scope="experiment",
    )

    inference = RepresentationInference(restored_model, bank.bank, patchifier)
    direct = []
    with torch.no_grad():
        for row in canonical:
            mask_seed = deterministic_mask_seed(seed, row.actual_history_id, row.file_id)
            batch = collate_variable_files(
                [row.sample], patchifier, masking_config=config, masking_seed=mask_seed,
            )
            clean = inference.co_locate_batch(_encoder_batch(batch))
            direct.append(restored_model(clean)["file_embedding"].detach().cpu())
    assert torch.equal(bank.bank.embeddings, torch.cat(direct, dim=0))
    assert torch.equal(bank.bank.embeddings, bank_replayed.bank.embeddings)
    assert bank.provenance.bank_sha256 == bank_replayed.provenance.bank_sha256
    assert bank.provenance.source_rows_sha256 == bank_raw.provenance.source_rows_sha256
    assert bank.provenance.bank_sha256 != bank_raw.provenance.bank_sha256

    scores = score_certified_rows(
        binding, arm, restored_model, bank, calibration_rows, patchifier,
        checkpoint=restored, action="calibrate_q95", scope="experiment",
    )
    scores_replayed = score_certified_rows(
        binding, arm, restored_model, bank_replayed, calibration_rows, patchifier,
        checkpoint=restored, action="calibrate_q95", scope="experiment",
    )
    for branch in ("S_pred", "S_pop"):
        assert dict(scores.scores[branch]) == dict(scores_replayed.scores[branch])
    calibration = fit_calibration_q95(binding, scores, calibration_rows, scope="experiment")
    calibration_replayed = fit_calibration_q95(
        binding, scores_replayed, calibration_rows, scope="experiment",
    )
    assert calibration.S_pred.threshold == calibration_replayed.S_pred.threshold
    assert calibration.S_pop.threshold == calibration_replayed.S_pop.threshold
    assert math.isfinite(calibration.S_pred.threshold)
    assert math.isfinite(calibration.S_pop.threshold)
