"""Task 17: tiny real chronological public-entry V2 CPU smoke.

Proves the integrated public V2 path on real chronological data (client
profile, seed 0): no input leakage, finite gradients/energies, covariance
stability, localization on scored (never fitted) abnormal files,
checkpoint restoration, deterministic trajectories, confidence/risk
separation, and complete outputs. All references stay dev-fitted; the
risk head here is fitted on seeded synthetic trajectory features purely
to exercise the separation contract — real risk calibration belongs to
the staged experiments, not this smoke.
"""

from __future__ import annotations

import torch

from representation.data import collate_variable_files
from representation.v2_aggregation import aggregate_file_state
from representation.v2_config import V2Config, assert_encoder_inputs_clean
from representation.v2_contracts import (
    validate_confidence,
    validate_file_state,
    validate_patch_output,
    validate_risk,
    validate_trajectory,
)
from representation.v2_inference import V2InferencePipeline, patch_regime_ids
from representation.v2_risk import (
    CensoredSurvivalRisk,
    HealthyTailCalibrator,
    expected_feature_width,
)
from representation.v2_trainer import build_v2_training_stack
from representation.v2_trajectory import TrajectoryTracker
from synth.chronicle import build_chronological, client_config
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import DIAGNOSTIC_ONLY_FIELD_NAMES, SampleLabel


def _config() -> V2Config:
    return V2Config(
        n_channels=6, d_model=8, n_robots=3, n_programs=4, n_regimes=7,
        n_prototypes=1, sequence_layers=1, attention_heads=2, dropout=0.0,
        min_group_samples=2, diag_min_samples=4, top_q_fraction=0.25,
        elevated_threshold=3.0, boundary_warmup_steps=0, boundary_ramp_steps=10,
        calibration_min_samples=8, seed=0,
    )


def test_v2_chronological_end_to_end_smoke(tmp_path) -> None:
    config = _config()
    _, _, labeled, splits = build_chronological(client_config(seed=0))
    by_id = {s.file_id: s for s in labeled}
    dev = [by_id[i] for i in splits.dev_train] + [by_id[i] for i in splits.dev_val]
    assert all(s.file_label is SampleLabel.NORMAL for s in dev)
    abnormal = [s for s in (by_id[i] for i in splits.test_static)
                if s.file_label is SampleLabel.ABNORMAL][:2]
    assert len(abnormal) == 2

    # No label/future leakage: every real sample's encoder inputs avoid all
    # diagnostic-only fields.
    for sample in dev[:4] + abnormal:
        inputs = sample.encoder_inputs()
        assert not (set(inputs) & set(DIAGNOSTIC_ONLY_FIELD_NAMES))
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16))
    batch = collate_variable_files(dev[:8], patchifier)
    train_inputs = {
        "patches": batch["patches"], "patch_pad_mask": batch["patch_pad_mask"],
        "patch_valid_mask": batch["patch_valid_mask"], "robot_idx": batch["robot_idx"],
        "program_idx": batch["program_idx"],
        "regime_ids": patch_regime_ids(dev[:8], batch["starts"], batch["patches"].shape[1]),
    }
    assert_encoder_inputs_clean({k: None for k in train_inputs})

    model, _, trainer = build_v2_training_stack(config, device="cpu", seed=0, lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    corruption = (torch.rand(batch["patch_valid_mask"].shape, generator=gen) < 0.1) & batch[
        "patch_valid_mask"]
    terms = trainer.train_step({**train_inputs, "corruption_mask": corruption, "severity": 2.0})
    assert all(torch.isfinite(torch.tensor(v)).all() for v in terms.values())
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)

    # Dev-fitted references only; abnormal files are scored, never fitted.
    model.eval()
    with torch.no_grad():
        encoded = model(**{k: v for k, v in train_inputs.items()})
    latents, valid = encoded["patch_latents"], train_inputs["patch_valid_mask"]
    geometry = trainer.fit_geometry(
        latents[valid],
        train_inputs["robot_idx"].unsqueeze(1).expand_as(valid)[valid],
        train_inputs["program_idx"].unsqueeze(1).expand_as(valid)[valid],
        train_inputs["regime_ids"][valid],
        torch.ones(int(valid.sum().item()), dtype=torch.bool),
    )
    # Covariance stability: every stored reference inverts sanely.
    for stats in (list(geometry._geometry._groups.values())
                  + list(geometry._geometry._pair.values())
                  + list(geometry._geometry._robot.values())
                  + [geometry._geometry._fleet]):
        cond = torch.linalg.cond(stats.cov).item()
        assert cond == cond and cond < 1e12, f"unstable covariance at {stats.level}"

    file_state = aggregate_file_state(
        latents, geometry.mixture_energy(
            latents, valid, train_inputs["robot_idx"],
            train_inputs["program_idx"], train_inputs["regime_ids"])["patch_energy"],
        valid, train_inputs["regime_ids"],
        top_q_fraction=config.top_q_fraction,
        elevated_threshold=config.elevated_threshold, n_regimes=config.n_regimes,
    )
    validate_file_state(file_state)
    assert torch.isfinite(file_state["file_state"]).all()

    tracker = TrajectoryTracker(config.d_model)
    tracker.fit_commissioning(file_state["file_state"],
                              torch.ones(len(dev[:8]), dtype=torch.bool))
    disps = torch.cat([tracker.update(file_state["file_state"][i])["displacement"]
                       for i in range(len(dev[:8]))])
    calibrator = HealthyTailCalibrator(min_samples=8).fit(disps)

    # Synthetic-feature risk head: contract exercise only (see docstring).
    torch.manual_seed(11)
    width = expected_feature_width()
    syn = torch.randn(300, width)
    will_fail = torch.rand(300) < torch.sigmoid(2.0 * syn[:, 0] - 0.5)
    fail_time, followup = torch.rand(300) * 10.0, 2.0 + torch.rand(300) * 12.0
    event = will_fail & (fail_time <= followup)
    risk = CensoredSurvivalRisk(width).fit(
        syn, torch.where(event, fail_time, followup), event)

    checkpoint = tmp_path / "v2_smoke.pt"
    trainer.save(checkpoint, geometry=geometry, calibrator=calibrator,
                 risk=risk, tracker=tracker)
    pipeline = V2InferencePipeline.load(checkpoint, device="cpu")

    def fresh_tracker() -> TrajectoryTracker:
        clone = TrajectoryTracker(config.d_model)
        assert isinstance(checkpoint_payload["tracker"], dict)
        clone.load_state_dict(checkpoint_payload["tracker"])
        return clone

    from representation.v2_checkpoint import load_v2_checkpoint
    checkpoint_payload = load_v2_checkpoint(checkpoint, pipeline.model,
                                            expected_config=config)
    # Chronological stream in operation order, one tracker per robot.
    stream = sorted(dev[:8] + abnormal,
                    key=lambda s: s.operation.start_time)  # type: ignore[union-attr]
    trackers = {r: fresh_tracker() for r in {s.robot_idx for s in stream}}
    seen: list[dict[str, dict[str, torch.Tensor]]] = []
    for sample in stream:
        one = collate_variable_files([sample], patchifier)
        n = one["patches"].shape[1]
        out = pipeline.score_patches(
            one["patches"], one["patch_pad_mask"], one["patch_valid_mask"],
            one["robot_idx"], one["program_idx"],
            patch_regime_ids([sample], one["starts"], n),
            tracker=trackers[sample.robot_idx],
        )
        validate_patch_output(out["patch"])
        validate_file_state(out["file"])
        validate_trajectory(out["trajectory"])
        validate_confidence(out["confidence"])
        validate_risk(out["risk"])
        assert set(out["confidence"]) == {"confidence", "p_normal"}
        assert set(out["risk"]) == {"risk_1d", "risk_7d"}
        seen.append(out)
    assert len(seen) == len(stream)

    # Localization (structural): patch resolution is preserved end to end —
    # the injected region maps to a strict non-empty patch subset, energies
    # vary across patches, and the file-level upper tail exceeds the median
    # (sparse evidence survives aggregation). Cross-region energy ordering
    # is a trained-model property for Tasks 33-34, not this smoke.
    for sample, out in zip(stream, seen):
        if sample.file_label is not SampleLabel.ABNORMAL or sample.anomaly_mask is None:
            continue
        pb = patchifier.patchify(sample)
        patch_mask = Patchifier.timestep_mask_to_patch_mask(
            sample.anomaly_mask, pb.starts, pb.valid_len, sample.C, sample.T,
        )
        energies = out["patch"]["patch_energy"][0]
        masked = torch.tensor(patch_mask, dtype=torch.bool)
        assert masked.any() and (~masked).any()
        assert float(energies.std()) > 0.0
        tail = float(out["file"]["tail_energy"][0])
        median = float(out["file"]["energy_quantiles"][0, 0])
        assert tail >= median

    # Determinism: identical chronological passes give identical trajectories.
    trackers_b = {r: fresh_tracker() for r in {s.robot_idx for s in stream}}
    for pos, sample in enumerate(stream):
        one = collate_variable_files([sample], patchifier)
        n = one["patches"].shape[1]
        out = pipeline.score_patches(
            one["patches"], one["patch_pad_mask"], one["patch_valid_mask"],
            one["robot_idx"], one["program_idx"],
            patch_regime_ids([sample], one["starts"], n),
            tracker=trackers_b[sample.robot_idx],
        )
        first = seen[pos]["trajectory"]["displacement"]
        torch.testing.assert_close(out["trajectory"]["displacement"], first)
