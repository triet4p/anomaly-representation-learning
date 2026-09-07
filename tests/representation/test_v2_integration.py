"""Task 16: tiny real public train-save-load-infer contract test.

Uses real chronological files from the public ``build_chronological``
entry point (client profile, seed 0): dev-train normals for training,
geometry commissioning, and the tracker baseline; dev-val for the
operating elevated threshold and the healthy-tail calibrator. No test
view is touched anywhere. The dev cohort carries no failure positives,
so the survival-risk head is correctly absent here — risk fitting is
proved in ``test_v2_risk.py``. The monitoring chain consumes encoder
``context_energy``; hierarchical ``population_energy`` is asserted as a
separately named signal.
"""

from __future__ import annotations

import pytest
import torch

from representation.data import collate_variable_files
from representation.v2_aggregation import (
    aggregate_file_state,
    calibrate_elevated_threshold_with_provenance,
)
from representation.v2_checkpoint import load_v2_checkpoint
from representation.v2_config import V2Config, V2_ENCODER_FORBIDDEN_FIELDS
from representation.v2_contracts import (
    validate_confidence,
    validate_context_patch_output,
    validate_file_state,
    validate_population_patch_output,
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
        calibration_min_samples=4,
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

    # Monitoring chain on the boundary-trained context energy; the operating
    # elevated threshold and the confidence calibrator fit on dev-val only.
    file_state = aggregate_file_state(
        latents, encoded["context_energy"],
        valid, v2_batch["regime_ids"],
        energy_source="context_energy",
        top_q_fraction=config.top_q_fraction,
        elevated_threshold=config.elevated_threshold,
        n_regimes=config.n_regimes,
    )
    assert file_state["energy_source"] == "context_energy"
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
        val_encoded["patch_latents"], val_encoded["context_energy"],
        val_batch["patch_valid_mask"], val_batch["regime_ids"],
        energy_source="context_energy",
        top_q_fraction=config.top_q_fraction,
        elevated_threshold=config.elevated_threshold,
        n_regimes=config.n_regimes,
    )
    # Dev-val-calibrated operating threshold from valid context energies.
    operating_value, operating_provenance = calibrate_elevated_threshold_with_provenance(
        val_encoded["context_energy"], val_batch["patch_valid_mask"], cohort="dev-val"
    )
    assert operating_provenance["fit_cohort"] == "dev-val"
    probe = TrajectoryTracker(config.d_model)
    probe.fit_commissioning(
        file_state["file_state"], torch.ones(len(train_files), dtype=torch.bool)
    )
    # Dev-val-only conformal calibration: the 7-file dev-val cohort clears
    # the small test floor of 4 and is flagged below the reference floor.
    val_disps = torch.cat(
        [probe.update(val_file["file_state"][i])["displacement"]
         for i in range(val_file["file_state"].shape[0])]
    )
    calibrator = HealthyTailCalibrator(
        min_samples=config.calibration_min_samples
    ).fit(val_disps, cohort="dev-val")
    assert calibrator.fit_cohort_info()["cohort"] == "dev-val"

    checkpoint = tmp_path / "v2_contract.pt"
    trainer.save(
        checkpoint, geometry=geometry, calibrator=calibrator, tracker=tracker,
        operating_threshold=dict(operating_provenance),
    )
    assert checkpoint.is_file()

    # Restored-reference-by-default inference: identical energies post-load.
    with torch.no_grad():
        before = model(
            v2_batch["patches"], v2_batch["patch_pad_mask"], v2_batch["patch_valid_mask"],
            v2_batch["robot_idx"], v2_batch["program_idx"], v2_batch["regime_ids"],
        )
    pipeline = V2InferencePipeline.load(checkpoint, device="cpu")
    # The calibrated non-default cutoff restores by default (never 3.0 here).
    assert pipeline.elevated_threshold == pytest.approx(operating_value)
    assert pipeline.elevated_threshold != pytest.approx(3.0)
    assert "dev-val" in pipeline.elevated_threshold_source
    restored_tracker = TrajectoryTracker(config.d_model)
    checkpoint_payload = load_v2_checkpoint(checkpoint, pipeline.model, expected_config=config)
    assert isinstance(checkpoint_payload["tracker"], dict)
    assert checkpoint_payload["operating_threshold"]["value"] == pytest.approx(operating_value)
    assert checkpoint_payload["confidence_calibrator_fit_cohort"]["cohort"] == "dev-val"
    restored_tracker.load_state_dict(checkpoint_payload["tracker"])
    assert checkpoint_payload["step"] == trainer.step
    out = pipeline.score_patches(
        v2_batch["patches"], v2_batch["patch_pad_mask"], v2_batch["patch_valid_mask"],
        v2_batch["robot_idx"], v2_batch["program_idx"], v2_batch["regime_ids"],
        tracker=restored_tracker,
    )
    validate_context_patch_output(out["patch"])
    validate_population_patch_output(out["population"])
    validate_file_state(out["file"])
    validate_file_state(out["file_population"])
    validate_trajectory(out["trajectory"])
    validate_confidence(out["confidence"])
    assert out["risk"] == {}  # no survival head fitted on all-negative dev data
    assert out["file"]["energy_source"] == "context_energy"
    assert out["file_population"]["energy_source"] == "population_energy"
    # Monitoring score is the exact encoder output; population is the
    # hierarchical energy — identical post-load restoration.
    torch.testing.assert_close(out["patch"]["context_energy"], before["context_energy"])
    torch.testing.assert_close(
        out["population"]["population_energy"],
        geometry.mixture_energy(
            before["patch_latents"], valid, v2_batch["robot_idx"],
            v2_batch["program_idx"], v2_batch["regime_ids"])["population_energy"],
    )
    # File outputs use the restored calibrated cutoff end to end.
    torch.testing.assert_close(
        out["file"]["elevated_fraction"],
        aggregate_file_state(
            before["patch_latents"], before["context_energy"], valid,
            v2_batch["regime_ids"], energy_source="context_energy",
            top_q_fraction=config.top_q_fraction,
            elevated_threshold=operating_value, n_regimes=config.n_regimes,
        )["elevated_fraction"],
    )

    # Fail-fast behavior: missing checkpoints and config mismatches raise.
    with pytest.raises(FileNotFoundError):
        V2InferencePipeline.load(tmp_path / "absent.pt")
    with pytest.raises(ValueError):
        load_v2_checkpoint(
            checkpoint, pipeline.model,
            expected_config=V2Config(**{**config.to_dict(), "d_model": 16}),
        )
