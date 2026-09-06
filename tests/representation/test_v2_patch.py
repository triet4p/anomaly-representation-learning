"""Task 10: context-conditioned patch distribution encoder."""

from __future__ import annotations

import torch

from representation.v2_patch import ContextConditionedPatchEncoder
from representation.v2_contracts import validate_patch_output


def _encoder() -> ContextConditionedPatchEncoder:
    return ContextConditionedPatchEncoder(
        n_channels=3,
        d_model=8,
        n_robots=2,
        n_programs=2,
        n_regimes=3,
        n_prototypes=2,
        sequence_layers=1,
        attention_heads=2,
        dropout=0.0,
    )


def _inputs(batch: int = 2, patches: int = 4, width: int = 8):
    torch.manual_seed(0)
    signal = torch.randn(batch, patches, 3, width)
    pad = torch.zeros(batch, patches, width, dtype=torch.bool)
    pad[0, 1, 6:] = True  # partial patch padding
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    pad[0, 3, :] = True
    pad[1, 2:, :] = True
    robot = torch.tensor([0, 1])
    program = torch.tensor([1, 0])
    regimes = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 0]])
    return signal, pad, valid, robot, program, regimes


def test_variable_length_shapes_and_localization() -> None:
    enc = _encoder().eval()
    patches, pad, valid, robot, program, regimes = _inputs()
    with torch.no_grad():
        out = enc(patches, pad, valid, robot, program, regimes)
    assert tuple(out["patch_latents"].shape) == (2, 4, 8)
    assert tuple(out["cond_mean"].shape) == (2, 4, 8)
    assert tuple(out["prototype_logits"].shape) == (2, 4, 2)
    assert torch.isfinite(out["patch_latents"]).all()
    # Invalid patches carry no signal; valid patches are localized per position.
    assert (out["patch_latents"][~valid] == 0).all()
    assert (out["patch_energy"][~valid] == 0).all()


def test_padding_values_cannot_leak_into_valid_embeddings() -> None:
    enc = _encoder().eval()
    patches, pad, valid, robot, program, regimes = _inputs()
    with torch.no_grad():
        baseline = enc(patches, pad, valid, robot, program, regimes)
    poisoned = patches.clone()
    poisoned[pad.unsqueeze(2).expand_as(poisoned)] = 1e4
    with torch.no_grad():
        actual = enc(poisoned, pad, valid, robot, program, regimes)
    torch.testing.assert_close(actual["patch_latents"], baseline["patch_latents"])
    torch.testing.assert_close(actual["patch_energy"], baseline["patch_energy"])


def test_low_variance_conditional_nll_is_negative_and_valid() -> None:
    enc = _encoder().eval()
    # A perfect prediction under a tight conditional stays finite-negative NLL.
    latents = torch.zeros(1, 2, 8)
    mean = torch.zeros(1, 2, 8)
    logvar = torch.full((1, 2, 8), -4.0)
    energy = enc._gaussian_nll(latents, mean, logvar)
    assert bool((energy < 0).all())
    assert torch.isfinite(energy).all()
    valid = torch.tensor([[True, False]])
    validate_patch_output(
        {
            "patch_latents": latents.masked_fill(~valid.unsqueeze(-1), 0.0),
            "patch_energy": energy.masked_fill(~valid, 0.0),
            "patch_valid_mask": valid,
        }
    )


def test_context_conditioning_changes_energy_and_gradients_flow() -> None:
    enc = _encoder().train()
    patches, pad, valid, robot, program, regimes = _inputs()
    out = enc(patches, pad, valid, robot, program, regimes)
    assert "reconstruction" not in out
    alt_program = torch.tensor([0, 1])
    with torch.no_grad():
        alt = enc(patches, pad, valid, robot, alt_program, regimes)
    assert not torch.equal(out["patch_energy"], alt["patch_energy"])
    loss = out["patch_energy"][valid].mean() + out["prototype_logits"][valid].pow(2).mean()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and bool((g != 0).any()) for g in grads)
