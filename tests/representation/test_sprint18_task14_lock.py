"""Sprint 18 Task 14 config-lock regression test.

Protects the frozen final training configs against plausible regression:
an accidental edit of experiments/sprint18-task14-config-lock.json (a
coefficient, schedule value, seed, geometry, hash, or the stopped-N1
representation) must fail closed here before it can reach execution.

The test re-derives trust through the real public consumers, never by
asserting source text:
- lock digest recomputation (canonical JSON rule);
- all 21 executable locked configs re-materialized through
  make_sprint18_arm_config and re-validated with require_locked=True,
  with arm_config_sha256 equality against the embedded digests;
- frozen schedule spot values through learning_rate_for_epoch /
  coefficient_for_epoch;
- stopped-N1 consumer refusal through the real trainer gate
  (train_sprint18_hcc_epoch with a null-HCC locked config);
- N4 recipe constants against the live crn module;
- compute-envelope boundaries through validate_compute_envelope.

No DATA access, no training, no outcome.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from representation.config import V1Config
from representation.crn import (
    N4_SATURATION_CAP_GATE,
    N4_SCALE_BATCH_SIZE,
    N4_SCALE_CLIP_NORM,
    N4_SCALE_EPOCHS,
    N4_SCALE_HIDDEN_WIDTH,
    N4_SCALE_LR,
    N4_SCALE_PARAM_LIMIT,
    N4_SCALE_WEIGHT_DECAY,
)
from representation.model import V1RepresentationModel
from representation.sprint18_checkpoint import (
    coefficient_for_epoch,
    learning_rate_for_epoch,
)
from representation.sprint18_harness import (
    BATCH_SIZE,
    EMA_DECAY,
    MECHANISM_SPEC_SHA256,
    MODEL_LR_MAX,
    MODEL_LR_MIN,
    MODEL_SEEDS,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    Sprint18ArmConfig,
    arm_config_sha256,
    make_sprint18_arm_config,
    validate_arm_config,
    validate_compute_envelope,
)
from representation.trainer import RepresentationTrainer
from synth.patchify import PatchConfig, Patchifier

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "experiments" / "sprint18-task14-config-lock.json"

EXPECTED_SPECS = {
    "N1-HCC": "0f5ede73d65441014caf22a22058c17d8ed34be735790d20ca7f36873a34d144",
    "N2-MSP": "a2deeb5479957ef8777ec4cfd66898e3ecd5e0aef80e2043f11882529ae3a9f2",
    "N3-CCRP": "eee44cebd29de2b5cbfdf0263fcca21d82fee92e0773922c0bc3209dee50443c",
    "N4-CRN": "4d1c8097c9cc27d3fbeb2e2a6e70a29a7bc63d7899646a9c712a315b39d9ffe3",
}

EXPECTED_COEFFICIENTS = {
    "B0-FULL": {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None},
    "K3-FULL": {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None},
    "K7-FULL": {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None},
    "K9-FULL": {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None},
    "N2-MSP": {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": 0.25, "beta": None},
    "N3-CCRP": {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": 0.50},
    "N4-CRN": {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None},
}


def _load_lock() -> dict:
    return json.loads(LOCK_PATH.read_bytes().decode("utf-8"))


def test_task14_lock_digest_and_normative_hashes() -> None:
    lock = _load_lock()
    assert lock["schema"] == "s18-task14-config-lock-v1"
    assert lock["task_id"] == "S18-T14"
    assert lock["source_commit"] == "e5b11130ddd2f155bbdb087dfdf17ed55a962af3"
    assert PROTOCOL_ID == "sprint18-full-training-protocol-v2"
    assert PROTOCOL_SHA256 == lock["normative"]["protocol_sha256"]
    assert dict(MECHANISM_SPEC_SHA256) == EXPECTED_SPECS
    assert lock["normative"]["mechanism_spec_sha256"] == EXPECTED_SPECS
    core = {
        key: value
        for key, value in lock.items()
        if key not in ("coefficient_lock_sha256", "locked_configs", "config_sha256")
    }
    recomputed = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert recomputed == lock["coefficient_lock_sha256"]


def test_task14_all_executable_configs_validate_locked() -> None:
    lock = _load_lock()
    digest = lock["coefficient_lock_sha256"]
    assert len(lock["locked_configs"]) == 21
    assert set(lock["locked_configs"]) == {
        f"{arm}/seed-{seed}" for arm in EXPECTED_COEFFICIENTS for seed in MODEL_SEEDS
    }
    assert "N1-HCC/seed-181801" not in lock["locked_configs"]
    n1 = lock["arms"]["N1-HCC"]
    assert n1["executable"] is False
    assert n1["status"] == "stopped_null_no_executable_config"
    assert n1["lambda_hcc_max"] is None
    for key, embedded in lock["locked_configs"].items():
        arm_id, _, seed = key.partition("/seed-")
        config = Sprint18ArmConfig(**embedded)
        assert config.selection_state == "locked"
        assert config.coefficient_lock_sha256 == digest
        validate_arm_config(config, require_locked=True)
        rebuilt = make_sprint18_arm_config(
            arm_id,
            int(seed),
            EXPECTED_COEFFICIENTS[arm_id],
            selection_state="locked",
            coefficient_lock_sha256=digest,
        )
        assert rebuilt.model_dump(mode="json") == embedded
        assert arm_config_sha256(config) == lock["config_sha256"][key] == arm_config_sha256(rebuilt)
        frozen = lock["arms"][arm_id]
        assert list(config.component_set) == frozen["component_set"]
    assert lock["common_contract"]["model_seeds"] == list(MODEL_SEEDS) == [181801, 181802, 181803]
    assert (BATCH_SIZE, MODEL_LR_MAX, MODEL_LR_MIN, EMA_DECAY) == (128, 1.5e-3, 1e-5, 0.996)
    assert lock["common_contract"]["representation"]["optimizer_steps_per_arm_seed"] == 4300
    n4cfg = Sprint18ArmConfig(**lock["locked_configs"]["N4-CRN/seed-181801"])
    assert (n4cfg.batch_size, n4cfg.batch_size_unit) == (512, "patch_records")
    assert (n4cfg.learning_rate, n4cfg.weight_decay) == (1e-3, 1e-4)
    assert (n4cfg.representation_epochs, n4cfg.scale_epochs, n4cfg.primary_checkpoint_epoch) == (0, 50, 50)


def test_task14_frozen_schedule_spot_values() -> None:
    assert learning_rate_for_epoch(1) == MODEL_LR_MAX * 0.1
    assert learning_rate_for_epoch(5) == MODEL_LR_MAX
    assert learning_rate_for_epoch(100) == 1e-5
    assert learning_rate_for_epoch(6) < MODEL_LR_MAX
    for maximum in (0.05, 0.25, 0.50):
        assert coefficient_for_epoch(maximum, 10) == 0.0
        assert coefficient_for_epoch(maximum, 30) == maximum
        assert coefficient_for_epoch(maximum, 31) == maximum
        assert coefficient_for_epoch(maximum, 100) == maximum


def test_task14_stopped_n1_refuses_locked_execution() -> None:
    lock = _load_lock()
    digest = lock["coefficient_lock_sha256"]
    null = make_sprint18_arm_config(
        "N1-HCC",
        MODEL_SEEDS[0],
        {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="locked",
        coefficient_lock_sha256=digest,
    )
    assert null.lambda_hcc_max is None
    config = V1Config(
        n_channels=6,
        patch_size=32,
        stride=16,
        d_model=16,
        attention_heads=4,
        sequence_layers=1,
        dropout=0.0,
        n_robots=9,
        n_programs=8,
        seed=MODEL_SEEDS[0],
        contrastive_weight_max=0.05,
        ema_decay=0.996,
    )
    model = V1RepresentationModel(config, patchifier=Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0)
    trainer = RepresentationTrainer(model, optimizer=optimizer, max_grad_norm=1.0, seed=MODEL_SEEDS[0])
    try:
        trainer.train_sprint18_hcc_epoch([], arm=null, epoch=31)
    except ValueError as exc:
        assert "frozen HCC coefficient maximum" in str(exc)
    else:
        raise AssertionError("stopped N1-HCC null config must refuse locked execution")


def test_task14_n4_recipe_matches_verified_contract() -> None:
    lock = _load_lock()
    recipe = lock["n4_recipe"]
    assert recipe["architecture"]["hidden_width"] == N4_SCALE_HIDDEN_WIDTH == 32
    assert recipe["fit_recipe"]["epochs"] == N4_SCALE_EPOCHS == 50
    assert recipe["fit_recipe"]["batch_size"] == N4_SCALE_BATCH_SIZE == 512
    assert recipe["fit_recipe"]["learning_rate"] == N4_SCALE_LR == 1e-3
    assert recipe["fit_recipe"]["weight_decay"] == N4_SCALE_WEIGHT_DECAY == 1e-4
    assert recipe["fit_recipe"]["gradient_clip_norm"] == N4_SCALE_CLIP_NORM == 1.0
    assert recipe["saturation_gate"]["rule"].startswith("frac_at_cap <= 0.20")
    assert N4_SATURATION_CAP_GATE == 0.20
    assert recipe["capacity"]["scale_parameter_limit"] == N4_SCALE_PARAM_LIMIT == 100000
    assert recipe["quantile_bound_rules"]["fitted_values"] is None
    assert recipe["empirical_capacity_verdict"] == "deferred_to_task13b_after_task23"


def test_task14_compute_envelope_boundaries() -> None:
    b0_params = {"S_pred": 100000, "S_pop": 90000}
    b0_flops = {"S_pred": 2000000, "S_pop": 1800000}
    assert (
        validate_compute_envelope(
            "K3-FULL",
            parameters_by_branch={"S_pred": 110000, "S_pop": 99000},
            flops_by_branch={"S_pred": 2200000, "S_pop": 1980000},
            b0_parameters_by_branch=b0_params,
            b0_flops_by_branch=b0_flops,
            training_flops_ratio=1.25,
        )["envelope_pass"]
        is True
    )
    for arm_id, params, flops, ratio, extra in (
        ("K3-FULL", {"S_pred": 110001, "S_pop": 99000}, {"S_pred": 2200000, "S_pop": 1980000}, 1.25, {}),
        ("N1-HCC", {"S_pred": 100001, "S_pop": 90000}, dict(b0_flops), 1.0, {}),
        (
            "N4-CRN",
            {"S_pred": 104417, "S_pop": 90000},
            {"S_pred": 2100000, "S_pop": 1800000},
            0.2,
            {"n4_scale_parameters": 100001, "n4_s_pop_bitwise_equal_to_b0": True},
        ),
        ("N3-CCRP", {"S_pred": 110000, "S_pop": 99000}, {"S_pred": 2200000, "S_pop": 1980000}, 2.26, {}),
    ):
        try:
            validate_compute_envelope(
                arm_id,
                parameters_by_branch=params,
                flops_by_branch=flops,
                b0_parameters_by_branch=b0_params,
                b0_flops_by_branch=b0_flops,
                training_flops_ratio=ratio,
                **extra,
            )
        except ValueError as exc:
            assert "envelope" in str(exc)
        else:
            raise AssertionError(f"{arm_id} envelope breach must fail closed")
