"""Task 16: tiny real public train-save-load-infer contract test.

Uses real chronological files from the public ``build_chronological``
entry point (client profile, seed 0): dev-train normals for training,
geometry commissioning, and the tracker baseline; dev-val for the
healthy-tail calibrator. No test view is touched anywhere. The dev
cohort carries no failure positives, so the survival-risk head is
correctly absent here — risk fitting is proved in ``test_v2_risk.py``.
"""

from __future__ import annotations

import pytest
import torch

from representation.data import collate_variable_files
from representation.v2_aggregation import aggregate_file_state, calibrate_elevated_threshold
from representation.v2_checkpoint import load_v2_checkpoint
from representation.v2_config import V2Config, V2_ENCODER_FORBIDDEN_FIELDS
from representation.v2_contracts import (
    validate_confidence,
    validate_file_state,
    validate_patch_output,
    validate_trajectory,
)
from representation.v2_inference import V2InferencePipeline, patch_regime_ids
from representation.v2_risk import HealthyTailCalibrator
from representation.v2_trainer import build_v2_training_stack
from representation.v2_trajectory import TrajectoryTracker
from synth.chronicle import build_chronological, client_config
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import SampleLabel


def _config() -> V2Config:
    return V2Config(
        n_channels=6,
        d_model=8,
        n_robots=3,
        n_programs=4,
        n_regimes=7,
        n_prototypes=1,
        sequence_layers=1,
        attention_heads=2,
        dropout=0.0,
        min_group_samples=2,
        diag_min_samples=4,
        top_q_fraction=0.25,
        elevated_threshold=3.0,
        boundary_alpha_max=1.0,
        boundary_warmup_steps=0,
        boundary_ramp_steps=10,
        calibration_min_samples=8,
        seed=0,
    )


def _dev_files():
    _, _, labeled, splits = build_chronological(client_config(seed=0))
    by_id = {s.file_id: s for s in labeled}
    train = [by_id[i] for i in splits.dev_train[:8]]
    val = [by_id[i] for i in splits.dev_val]
    assert all(s.file_label is SampleLabel.NORMAL for s in train + val)
    return train, val


def _train_batch(trainer, samples, patchifier, seed: int = 0):
    batch = collate_variable_files(samples, patchifier)
    n = batch["patches"].shape[1]
    regimes = patch_regime_ids(samples, batch["starts"], n)
    gen = torch.Generator().manual_seed(seed)
    corruption = (torch.rand(batch["patch_valid_mask"].shape, generator=gen) < 0.1) & batch[
        "patch_valid_mask"
    ]
    return {
        "patches": batch["patches"],
        "patch_pad_mask": batch["patch_pad_mask"],
        "patch_valid_mask": batch["patch_valid_mask"],
        "robot_idx": batch["robot_idx"],
        "program_idx": batch["program_idx"],
        "regime_ids": regimes,
        "corruption_mask": corruption,
        "severity": 2.0,
    }, batch, samples


def test_v2_train_save_load_infer_contract(tmp_path) -> None:
    config = _config()
    train_files, val_files = _dev_files()
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16))
    model, _, trainer = build_v2_training_stack(config, device="cpu", seed=0, lr=1e-3)

    v2_batch, raw_batch, _ = _train_batch(trainer, train_files, patchifier)
    # Encoder inputs never carry supervision: the trainer allowlists signal
    # plus context ids, while corruption_mask/severity stay loss-only.
    encoder_inputs = trainer._encoder_inputs(v2_batch)
    assert not (set(encoder_inputs) & set(V2_ENCODER_FORBIDDEN_FIELDS))
    assert "corruption_mask" not in encoder_inputs and "severity" not in encoder_inputs

    terms = trainer.train_step(v2_batch)
    assert all(abs(v) != float("inf") and v == v for v in terms.values())
    assert trainer.step == 1
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)

    stationary = trainer.stationary_loss([v2_batch])
    assert stationary == stationary and abs(stationary) != float("inf")
    assert trainer.record_eval(stationary) is True
    assert trainer.record_eval(stationary + 1.0) is False
    trainer.restore_best_state()

    # Geometry commissioning from verified-healthy train latents only.
    model.eval()
    with torch.no_grad():
        encoded = model(
            v2_batch["patches"], v2_batch["patch_pad_mask"], v2_batch["patch_valid_mask"],
            v2_batch["robot_idx"], v2_batch["program_idx"], v2_batch["regime_ids"],
        )
    latents, valid = encoded["patch_latents"], v2_batch["patch_valid_mask"]
    rows = latents[valid]
    robot_rows = v2_batch["robot_idx"].unsqueeze(1).expand_as(valid)[valid]
    program_rows = v2_batch["program_idx"].unsqueeze(1).expand_as(valid)[valid]
    regime_rows = v2_batch["regime_ids"][valid]
    geometry = trainer.fit_geometry(
        rows, robot_rows, program_rows, regime_rows,
        torch.ones(rows.shape[0], dtype=torch.bool),
    )

    # File states + tracker commissioning on train; calibrator on dev-val.
    file_state = aggregate_file_state(
        latents, geometry.mixture_energy(
            latents, valid, v2_batch["robot_idx"], v2_batch["program_idx"],
            v2_batch["regime_ids"])["patch_energy"],
        valid, v2_batch["regime_ids"],
        top_q_fraction=config.top_q_fraction,
        elevated_threshold=config.elevated_threshold,
        n_regimes=config.n_regimes,
    )
    tracker = TrajectoryTracker(config.d_model)
    tracker.fit_commissioning(
        file_state["file_state"], torch.ones(len(train_files), dtype=torch.bool)
    )
    val_batch, _, _ = _train_batch(trainer, val_files, patchifier, seed=1)
    with torch.no_grad():
        val_encoded = model(
            val_batch["patches"], val_batch["patch_pad_mask"], val_batch["patch_valid_mask"],
            val_batch["robot_idx"], val_batch["program_idx"], val_batch["regime_ids"],
        )
    val_file = aggregate_file_state(
        val_encoded["patch_latents"],
        geometry.mixture_energy(
            val_encoded["patch_latents"], val_batch["patch_valid_mask"],
            val_batch["robot_idx"], val_batch["program_idx"], val_batch["regime_ids"]
        )["patch_energy"],
        val_batch["patch_valid_mask"], val_batch["regime_ids"],
        top_q_fraction=config.top_q_fraction,
        elevated_threshold=config.elevated_threshold,
        n_regimes=config.n_regimes,
    )
    probe = TrajectoryTracker(config.d_model)
    probe.fit_commissioning(
        file_state["file_state"], torch.ones(len(train_files), dtype=torch.bool)
    )
    # dev_val alone holds 7 files (below the calibration floor of 8), so the
    # smoke calibrates on the combined dev stream; the floor itself is
    # exercised in test_v2_risk.py.
    both = torch.cat([file_state["file_state"], val_file["file_state"]])
    all_disps = torch.cat([probe.update(both[i])["displacement"] for i in range(both.shape[0])])
    calibrator = HealthyTailCalibrator(
        min_samples=config.calibration_min_samples
    ).fit(all_disps)

    checkpoint = tmp_path / "v2_contract.pt"
    trainer.save(checkpoint, geometry=geometry, calibrator=calibrator, tracker=tracker)
    assert checkpoint.is_file()

    # Restored-reference-by-default inference: identical energies post-load.
    with torch.no_grad():
        before = model(
            v2_batch["patches"], v2_batch["patch_pad_mask"], v2_batch["patch_valid_mask"],
            v2_batch["robot_idx"], v2_batch["program_idx"], v2_batch["regime_ids"],
        )
    pipeline = V2InferencePipeline.load(checkpoint, device="cpu")
    restored_tracker = TrajectoryTracker(config.d_model)
    checkpoint_payload = load_v2_checkpoint(checkpoint, pipeline.model, expected_config=config)
    assert isinstance(checkpoint_payload["tracker"], dict)
    restored_tracker.load_state_dict(checkpoint_payload["tracker"])
    assert checkpoint_payload["step"] == trainer.step
    out = pipeline.score_patches(
        v2_batch["patches"], v2_batch["patch_pad_mask"], v2_batch["patch_valid_mask"],
        v2_batch["robot_idx"], v2_batch["program_idx"], v2_batch["regime_ids"],
        tracker=restored_tracker,
    )
    validate_patch_output(out["patch"])
    validate_file_state(out["file"])
    validate_trajectory(out["trajectory"])
    validate_confidence(out["confidence"])
    assert out["risk"] == {}  # no survival head fitted on all-negative dev data
    torch.testing.assert_close(
        out["patch"]["patch_energy"],
        geometry.mixture_energy(
            before["patch_latents"], valid, v2_batch["robot_idx"],
            v2_batch["program_idx"], v2_batch["regime_ids"])["patch_energy"],
    )

    # Fail-fast behavior: missing checkpoints and config mismatches raise.
    with pytest.raises(FileNotFoundError):
        V2InferencePipeline.load(tmp_path / "absent.pt")
    with pytest.raises(ValueError):
        load_v2_checkpoint(
            checkpoint, pipeline.model,
            expected_config=V2Config(**{**config.to_dict(), "d_model": 16}),
        )
