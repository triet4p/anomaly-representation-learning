from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from representation.config import V1Config
from representation.data import collate_variable_files
from representation.model import V1RepresentationModel
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import FileSample, RegimeMeta, RegimeType, SampleLabel


def _sample(channels: int, timesteps: int, file_id: str) -> FileSample:
    signal = np.arange(channels * timesteps, dtype=np.float32).reshape(channels, timesteps)
    sample = FileSample(
        x=signal,
        file_id=file_id,
        file_label=SampleLabel.NORMAL,
        seed=timesteps,
        generator_version="test",
        config_hash="test",
        regime_sequence=[RegimeMeta(RegimeType.ACTIVE, 0, timesteps, 0.5)],
    )
    sample.validate()
    return sample


@pytest.mark.parametrize("channels", [3, 6])
def test_model_forwards_homogeneous_channel_modes_and_variable_lengths(channels: int) -> None:
    config = V1Config(
        n_channels=channels,
        patch_size=16,
        stride=8,
        d_model=8,
        sequence_layers=1,
        attention_heads=2,
        dropout=0.0,
        total_mask_ratio=0.4,
    )
    samples = [_sample(channels, 21, "short"), _sample(channels, 43, "long")]
    patchifier = Patchifier(PatchConfig(patch_size=16, stride=8, pad_end=True))
    batch = collate_variable_files(samples, patchifier, masking_config=config, masking_seed=7)
    model = V1RepresentationModel(config, patchifier=patchifier).eval()

    with torch.no_grad():
        output = model(batch)
    assert tuple(output["context_latents"].shape) == (2, 5, 8)
    assert tuple(output["target_latents"].shape) == (2, 5, 8)
    assert tuple(output["predicted_latents"].shape) == (2, 5, 8)
    assert tuple(output["file_embedding"].shape) == (2, 8)
    assert tuple(output["view_embedding_1"].shape) == (2, 8)
    assert tuple(output["view_embedding_2"].shape) == (2, 8)
    assert torch.equal(output["prediction_mask"], batch["mask"] & batch["patch_valid_mask"])
    assert not output["target_latents"].requires_grad
    assert torch.isfinite(output["file_embedding"]).all()
    assert "reconstruction" not in output

    valid = batch["patch_valid_mask"].to(output["context_latents"].dtype).unsqueeze(-1)
    expected = (output["context_latents"] * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    torch.testing.assert_close(output["file_embedding"], expected)


def test_model_forward_does_not_consume_file_labels() -> None:
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1)
    sample = _sample(3, 17, "file")
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    batch = collate_variable_files([sample], patchifier)
    model = V1RepresentationModel(config, patchifier=patchifier).eval()
    with torch.no_grad():
        baseline = model(batch)
    batch["file_labels"] = [SampleLabel.ABNORMAL]
    with torch.no_grad():
        changed = model(batch)
    torch.testing.assert_close(baseline["file_embedding"], changed["file_embedding"])


def test_view_rng_state_round_trips_with_model_state_dict() -> None:
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    batch = collate_variable_files([_sample(3, 17, "state")], patchifier)
    torch.manual_seed(29)
    model = V1RepresentationModel(config, patchifier=patchifier).eval()
    model(batch)
    state = deepcopy(model.state_dict())
    expected = model(batch)
    restored = V1RepresentationModel(config, patchifier=patchifier).eval()
    restored.load_state_dict(state)
    actual = restored(batch)
    torch.testing.assert_close(expected["view_embedding_1"], actual["view_embedding_1"])
    torch.testing.assert_close(expected["view_embedding_2"], actual["view_embedding_2"])
def test_training_view_rng_advances_and_fresh_seeded_models_reproduce() -> None:
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0)
    samples = [_sample(3, 17, "rng")]
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    batch = collate_variable_files(samples, patchifier)
    torch.manual_seed(13)
    first_model = V1RepresentationModel(config, patchifier=patchifier).train()
    first = first_model(batch)
    second = first_model(batch)
    assert not torch.allclose(first["view_embedding_1"], second["view_embedding_1"])
    torch.manual_seed(13)
    second_model = V1RepresentationModel(config, patchifier=patchifier).train()
    replay = second_model(batch)
    torch.testing.assert_close(first["view_embedding_1"], replay["view_embedding_1"])
    torch.testing.assert_close(first["view_embedding_2"], replay["view_embedding_2"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_view_embeddings_follow_cuda_input_device() -> None:
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0)
    sample = _sample(3, 17, "cuda")
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    batch = collate_variable_files([sample], patchifier)
    batch = {name: value.cuda() if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
    model = V1RepresentationModel(config, patchifier=patchifier).cuda().eval()
    output = model(batch)
    assert output["view_embedding_1"].device.type == "cuda"
    assert output["view_embedding_1"].dtype == batch["patches"].dtype


def test_contrastive_views_enforce_encoder_input_normalization_parity() -> None:
    """Ensure contrastive views enter LocalPatchEncoder with normalized scale, not raw physical units."""
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0, min_bucket_samples=1)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))

    # Physical scale signal with high DC offset (500.0)
    signal = np.full((3, 24), 500.0, dtype=np.float32)
    for c in range(3):
        signal[c] += np.sin(np.linspace(0, 3.14, 24)).astype(np.float32) * 5.0
    sample = FileSample(
        x=signal,
        file_id="physical-1",
        file_label=SampleLabel.NORMAL,
        seed=1,
        generator_version="test",
        config_hash="test",
        regime_sequence=[RegimeMeta(RegimeType.ACTIVE, 0, 24, 0.5)],
    )
    batch = collate_variable_files([sample], patchifier)

    model = V1RepresentationModel(config, patchifier=patchifier).train()

    captured_patches: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig_forward = model.patch_encoder.forward
    def hooked_forward(patches: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        captured_patches.append((patches.clone(), pad_mask.clone()))
        return orig_forward(patches, pad_mask)
    model.patch_encoder.forward = hooked_forward  # type: ignore[method-assign]

    with torch.no_grad():
        model(batch)

    assert len(captured_patches) == 4, f"Expected 4 patch_encoder calls, got {len(captured_patches)}"
    for call_idx, (patches, pad_mask) in enumerate(captured_patches):
        mask_4d = pad_mask.unsqueeze(2).expand_as(patches)
        valid = patches[~mask_4d]
        # If raw physical values reached the encoder, valid.mean() would be ~500.0
        assert valid.mean().abs() < 0.2, f"Call {call_idx} failed normalization parity: mean={valid.mean().item()}"
        assert 0.5 < valid.std() < 1.5, f"Call {call_idx} failed unit variance: std={valid.std().item()}"
        assert valid.max() < 10.0, f"Call {call_idx} exceeded normalized bound: max={valid.max().item()}"


def test_contrastive_views_respect_fleet_conditioned_normalization() -> None:
    """Ensure contrastive views are normalized against each sample's fleet bucket baseline."""
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0, min_bucket_samples=1)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))

    s1_signal = np.full((3, 24), 300.0, dtype=np.float32)
    s2_signal = np.full((3, 24), 50.0, dtype=np.float32)
    for c in range(3):
        s1_signal[c] += np.sin(np.linspace(0, 3.14, 24)).astype(np.float32) * 4.0
        s2_signal[c] += np.cos(np.linspace(0, 3.14, 24)).astype(np.float32) * 4.0
    s1 = FileSample(
        x=s1_signal,
        file_id="robot-0",
        file_label=SampleLabel.NORMAL,
        seed=1,
        generator_version="test",
        config_hash="test",
        regime_sequence=[RegimeMeta(RegimeType.ACTIVE, 0, 24, 0.5)],
    )
    s2 = FileSample(
        x=s2_signal,
        file_id="robot-1",
        file_label=SampleLabel.NORMAL,
        seed=2,
        generator_version="test",
        config_hash="test",
        regime_sequence=[RegimeMeta(RegimeType.ACTIVE, 0, 24, 0.5)],
    )
    batch = collate_variable_files([s1, s2], patchifier)
    batch["robot_idx"] = torch.tensor([0, 1], dtype=torch.long)
    batch["program_idx"] = torch.tensor([1, 2], dtype=torch.long)

    model = V1RepresentationModel(config, patchifier=patchifier).train()

    captured_patches: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig_forward = model.patch_encoder.forward
    def hooked_forward(patches: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        captured_patches.append((patches.clone(), pad_mask.clone()))
        return orig_forward(patches, pad_mask)
    model.patch_encoder.forward = hooked_forward  # type: ignore[method-assign]

    with torch.no_grad():
        model(batch)

    # Check contrastive view calls (calls 2 and 3)
    for view_call in [2, 3]:
        patches, pad_mask = captured_patches[view_call]
        for sample_idx in range(2):
            p_sample = patches[sample_idx]
            m_sample = pad_mask[sample_idx].unsqueeze(1).expand_as(p_sample)
            valid = p_sample[~m_sample]
            assert valid.mean().abs() < 0.2, f"View call {view_call} sample {sample_idx} fleet norm mean={valid.mean().item()}"
            assert 0.5 < valid.std() < 1.5, f"View call {view_call} sample {sample_idx} fleet norm std={valid.std().item()}"


def test_contrastive_views_zero_padding_and_invariance_to_padded_leakage() -> None:
    """Ensure contrastive view padding is strictly zeroed and padded tail changes do not leak."""
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0, min_bucket_samples=1)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))

    s_short = _sample(3, 16, "short")  # 2 patches
    s_long = _sample(3, 32, "long")  # 4 patches
    batch = collate_variable_files([s_short, s_long], patchifier)

    model = V1RepresentationModel(config, patchifier=patchifier).eval()

    captured_patches: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig_forward = model.patch_encoder.forward
    def hooked_forward(patches: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        captured_patches.append((patches.clone(), pad_mask.clone()))
        return orig_forward(patches, pad_mask)
    model.patch_encoder.forward = hooked_forward  # type: ignore[method-assign]

    with torch.no_grad():
        out1 = model(batch)

    # Check contrastive calls (2 and 3) for strict zeroing of padded positions
    for call_idx in [2, 3]:
        patches, pad_mask = captured_patches[call_idx]
        mask_4d = pad_mask.unsqueeze(2).expand_as(patches)
        padded = patches[mask_4d]
        assert padded.numel() > 0, f"Expected padded elements in call {call_idx}, found none"
        torch.testing.assert_close(padded, torch.zeros_like(padded), msg=f"Call {call_idx} has non-zero padded values")


def test_model_context_target_and_predicted_latents_have_bounded_scale() -> None:
    """Verify context, target, and predicted latents are strictly bounded across variable-length padded files."""
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=16, attention_heads=4, sequence_layers=2, dropout=0.0, min_bucket_samples=1)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))

    s1 = _sample(3, 16, "short")  # 2 patches
    s2 = _sample(3, 32, "long")   # 4 patches
    batch = collate_variable_files([s1, s2], patchifier)

    model = V1RepresentationModel(config, patchifier=patchifier).eval()
    with torch.no_grad():
        output = model(batch)

    ctx = output["context_latents"]
    tgt = output["target_latents"]
    pred = output["predicted_latents"]
    pred_mask = output["prediction_mask"]
    valid = batch["patch_valid_mask"]

    # Stop-gradient invariant
    assert not tgt.requires_grad

    # Bounded scale invariant: sqrt(16) = 4.0
    for b in range(2):
        for t in range(4):
            if valid[b, t]:
                assert ctx[b, t].mean().abs() < 1e-4
                assert (ctx[b, t].std(unbiased=False) - 1.0).abs() < 1e-3
                assert (ctx[b, t].norm() - 4.0).abs() < 0.1

                assert tgt[b, t].mean().abs() < 1e-4
                assert (tgt[b, t].std(unbiased=False) - 1.0).abs() < 1e-3
                assert (tgt[b, t].norm() - 4.0).abs() < 0.1
            else:
                assert torch.equal(ctx[b, t], torch.zeros(16))
                assert torch.equal(tgt[b, t], torch.zeros(16))

            if pred_mask[b, t]:
                assert pred[b, t].mean().abs() < 1e-4
                assert (pred[b, t].std(unbiased=False) - 1.0).abs() < 1e-3
                assert (pred[b, t].norm() - 4.0).abs() < 0.1
                # Mean squared error between normalized vectors is bounded in [0, 4.0]
                mse = (pred[b, t] - tgt[b, t]).square().mean().item()
                assert 0.0 <= mse <= 4.0
            else:
                assert torch.equal(pred[b, t], torch.zeros(16))

    # File embedding is an average of normalized tokens -> norm bounded by sqrt(16) = 4.0
    for b in range(2):
        assert output["file_embedding"][b].norm().item() <= 4.0 + 1e-3


def test_contrastive_projector_isolates_gradients_from_predictor_and_target() -> None:
    """Verify contrastive gradients route through projector and encoders, isolating predictor and target."""
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0, min_bucket_samples=1)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    s1 = _sample(3, 16, "s1")
    s2 = _sample(3, 24, "s2")
    batch = collate_variable_files([s1, s2], patchifier, masking_config=config, masking_seed=1)

    model = V1RepresentationModel(config, patchifier=patchifier).train()

    # 1. Backpropagate contrastive loss
    model.zero_grad()
    out = model(batch)
    contrastive_loss = out["view_embedding_1"].sum() + out["view_embedding_2"].sum()
    contrastive_loss.backward()

    # Contrastive projector, context encoder, and patch encoder receive gradients
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.contrastive_projector.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.context_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.patch_encoder.parameters())

    # Predictor must NOT receive contrastive gradients
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.predictor.parameters())

    # Target encoder must NOT receive gradients (stop-gradient invariant)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.target_encoder.parameters())

    # 2. Backpropagate prediction loss
    model.zero_grad()
    out = model(batch)
    pred_loss = out["predicted_latents"].sum()
    pred_loss.backward()

    # Predictor, context encoder, and patch encoder receive gradients
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.context_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.patch_encoder.parameters())

    # Contrastive projector must NOT receive prediction gradients
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.contrastive_projector.parameters())

    # Target encoder must NOT receive gradients
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.target_encoder.parameters())


def test_contrastive_projector_shape_and_contract_validation() -> None:
    """Verify file_embedding and projected_file_embedding contracts and projector input validation."""
    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0, min_bucket_samples=1)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8, pad_end=True))
    s1 = _sample(3, 16, "s1")
    s2 = _sample(3, 24, "s2")
    batch = collate_variable_files([s1, s2], patchifier)

    model = V1RepresentationModel(config, patchifier=patchifier).eval()
    with torch.no_grad():
        out = model(batch)

    # file_embedding must be [B, D] and unprojected (matching average pooled context)
    valid = batch["patch_valid_mask"].to(out["context_latents"].dtype).unsqueeze(-1)
    expected_pooled = (out["context_latents"] * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    torch.testing.assert_close(out["file_embedding"], expected_pooled)

    # projected_file_embedding must equal project_contrastive(file_embedding)
    expected_projected = model.project_contrastive(out["file_embedding"])
    torch.testing.assert_close(out["projected_file_embedding"], expected_projected)

    # view_embedding_1 and view_embedding_2 must have shape [B, D]
    assert tuple(out["view_embedding_1"].shape) == (2, 8)
    assert tuple(out["view_embedding_2"].shape) == (2, 8)

    # Input dimension validation on project_contrastive
    with pytest.raises(ValueError, match="expected embeddings with shape"):
        model.project_contrastive(torch.randn(2, 5))
    with pytest.raises(ValueError, match="expected embeddings with shape"):
        model.project_contrastive(torch.randn(2, 4, 8))


def test_contrastive_projector_state_serializes_and_restores_in_checkpoint(tmp_path) -> None:
    """Verify contrastive_projector parameters round-trip through atomic checkpointing."""
    from representation.checkpoint import save_checkpoint, load_checkpoint

    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8))
    model = V1RepresentationModel(config, patchifier=patchifier)

    ckpt_file = tmp_path / "projector_test.pt"
    save_checkpoint(ckpt_file, model, step=42)

    restored = V1RepresentationModel(config, patchifier=patchifier)
    # Mutate restored weights before loading
    with torch.no_grad():
        for p in restored.contrastive_projector.parameters():
            p.add_(10.0)

    meta = load_checkpoint(ckpt_file, restored)
    assert meta["step"] == 42

    # Verify restored projector parameters match original exactly
    for name, param in model.contrastive_projector.named_parameters():
        restored_param = dict(restored.contrastive_projector.named_parameters())[name]
        torch.testing.assert_close(param, restored_param)


def test_v1_config_contrastive_calibration_defaults_and_boundaries() -> None:
    """Verify calibrated defaults and boundary rejection on contrastive configuration fields."""
    cfg = V1Config()
    assert cfg.contrastive_temperature == pytest.approx(0.2)
    assert cfg.contrastive_gain_std == pytest.approx(0.05)
    assert cfg.contrastive_offset_std == pytest.approx(0.03)
    assert cfg.contrastive_noise_std == pytest.approx(0.015)
    assert cfg.contrastive_max_shift == 4

    with pytest.raises(ValueError, match="greater than 0"):
        V1Config(contrastive_temperature=0.0)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        V1Config(contrastive_gain_std=-0.01)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        V1Config(contrastive_offset_std=-0.01)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        V1Config(contrastive_noise_std=-0.01)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        V1Config(contrastive_max_shift=-1)

    # Model inherits custom config fields
    custom_cfg = V1Config(
        n_channels=3,
        patch_size=8,
        stride=8,
        d_model=8,
        attention_heads=2,
        sequence_layers=1,
        dropout=0.0,
        contrastive_gain_std=0.08,
        contrastive_offset_std=0.04,
        contrastive_noise_std=0.02,
        contrastive_max_shift=6,
    )
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8))
    m = V1RepresentationModel(custom_cfg, patchifier=patchifier)
    assert m.contrastive_config.gain_std == pytest.approx(0.08)
    assert m.contrastive_config.offset_std == pytest.approx(0.04)
    assert m.contrastive_config.noise_std == pytest.approx(0.02)
    assert m.contrastive_config.max_shift == 6


def test_backward_compatible_checkpoint_loading_with_legacy_config(tmp_path) -> None:
    """Verify that checkpoints saved with legacy config dicts missing new contrastive fields load cleanly."""
    from representation.checkpoint import save_checkpoint, load_checkpoint

    config = V1Config(n_channels=3, patch_size=8, stride=8, d_model=8, attention_heads=2, sequence_layers=1, dropout=0.0)
    patchifier = Patchifier(PatchConfig(patch_size=8, stride=8))
    model = V1RepresentationModel(config, patchifier=patchifier)

    # Simulate legacy checkpoint config missing contrastive_gain_std, offset_std, noise_std, max_shift
    legacy_dict = config.to_dict()
    legacy_dict.pop("contrastive_gain_std", None)
    legacy_dict.pop("contrastive_offset_std", None)
    legacy_dict.pop("contrastive_noise_std", None)
    legacy_dict.pop("contrastive_max_shift", None)

    ckpt_file = tmp_path / "legacy.pt"
    save_checkpoint(ckpt_file, model, config=legacy_dict, step=15)

    restored = V1RepresentationModel(config, patchifier=patchifier)
    meta = load_checkpoint(ckpt_file, restored)
    assert meta["step"] == 15


def test_v1_config_lambda_max_alias_and_default() -> None:
    """Verify lambda_max alias, default value 0.1, and boundary validation in V1Config."""
    cfg_def = V1Config()
    assert cfg_def.contrastive_weight_max == pytest.approx(0.1)
    assert cfg_def.lambda_max == pytest.approx(0.1)

    cfg_alias = V1Config(lambda_max=0.35)
    assert cfg_alias.contrastive_weight_max == pytest.approx(0.35)
    assert cfg_alias.lambda_max == pytest.approx(0.35)

    cfg_orig = V1Config(contrastive_weight_max=0.5)
    assert cfg_orig.contrastive_weight_max == pytest.approx(0.5)
    assert cfg_orig.lambda_max == pytest.approx(0.5)

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        V1Config(lambda_max=-0.05)
