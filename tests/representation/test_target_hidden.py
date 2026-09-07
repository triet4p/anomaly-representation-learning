"""Focused tests for the target-hidden scorer (leakage/gradient/checkpoint)."""

import pytest
import torch

from representation.target_hidden import TargetHiddenScorer, exclusion_bias


def _toy(b=2, n=9, d=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    latents = torch.randn((b, n, d), generator=gen)
    valid = torch.ones((b, n), dtype=torch.bool)
    valid[0, -2:] = False
    return latents, valid


def test_exclusion_blocks_target_overlap_and_pad():
    _, valid = _toy()
    bias = exclusion_bias(9, valid)
    assert bias.shape == (2, 9, 9)
    for j in range(9):
        for k in (j - 1, j, j + 1):
            if 0 <= k < 9:
                assert bias[1, j, k] == float("-inf")
    # padded keys blocked everywhere on row 0
    assert bool((bias[0, :, 7:] == float("-inf")).all())
    # distant valid keys stay open
    assert bias[1, 0, 5] == 0.0
    with pytest.raises(ValueError):
        exclusion_bias(8, valid)


def test_no_target_path_through_context():
    latents, valid = _toy()
    model = TargetHiddenScorer(d_model=8).eval()
    with torch.no_grad():
        base = model.context_for(latents, valid)
        polluted = latents.clone()
        polluted[:, 4, :] = 1e4  # obliterate the target patch latent
        again = model.context_for(polluted, valid)
    # query 4's context is bit-identical: target hidden before mixing
    assert torch.equal(base[:, 4], again[:, 4])
    # overlap neighbors' latents also excluded from query 4
    polluted2 = latents.clone()
    polluted2[:, 5, :] = -1e4
    with torch.no_grad():
        again2 = model.context_for(polluted2, valid)
    assert torch.equal(base[:, 4], again2[:, 4])


def test_forward_finite_with_zeroed_invalid():
    latents, valid = _toy()
    model = TargetHiddenScorer(d_model=8).eval()
    with torch.no_grad():
        out = model(latents, valid)
    assert out["hidden_context_energy"].shape == (2, 9)
    assert bool(torch.isfinite(out["hidden_context_energy"]).all())
    assert bool((out["hidden_context_energy"][~valid] == 0.0).all())


def test_nonzero_finite_learning_gradients():
    torch.manual_seed(0)
    latents, valid = _toy()
    latents.requires_grad_(True)
    model = TargetHiddenScorer(d_model=8, n_layers=1)
    out = model(latents, valid)
    loss = out["hidden_context_energy"][valid].mean()
    loss.backward()
    seen = 0
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert bool(torch.isfinite(param.grad).all()), name
        if param.grad.abs().sum() > 0:
            seen += 1
    assert seen == len(list(model.parameters()))
def test_save_load_roundtrip_and_config_guard(tmp_path):
    latents, valid = _toy()
    model = TargetHiddenScorer(d_model=8, n_heads=2, n_layers=1).eval()
    path = tmp_path / "hidden.pt"
    model.save(path)
    restored = TargetHiddenScorer.load(path)
    assert restored.config() == model.config()
    with torch.no_grad():
        a = model(latents, valid)["hidden_context_energy"]
        b = restored(latents, valid)["hidden_context_energy"]
    assert torch.equal(a, b)
    other = TargetHiddenScorer(d_model=8, n_heads=4, n_layers=1)
    other.save(tmp_path / "other.pt")
    payload = torch.load(tmp_path / "other.pt", map_location="cpu", weights_only=False)
    payload["config"]["d_model"] = 16  # tampered architecture claim
    torch.save(payload, tmp_path / "tampered.pt")
    with pytest.raises((ValueError, RuntimeError)):
        TargetHiddenScorer.load(tmp_path / "tampered.pt")
