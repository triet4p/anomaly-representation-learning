"""Focused tests for Task 7 controlled mechanisms (plausible faults only)."""

import pytest
import torch

from representation.corruptions import (
    BOUNDS,
    MECHANISMS,
    NUISANCES,
    corrupt,
    severity_of,
)


def _batch(b=2, n=6, c=6, w=32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    patches = torch.randn((b, n, c, w), generator=gen)
    pad = torch.zeros((b, n, w), dtype=torch.bool)
    pad[:, -1, -4:] = True
    patches = patches.masked_fill(pad.unsqueeze(2), 0.0)  # production pads are zero
    valid = torch.ones((b, n), dtype=torch.bool)
    valid[0, -1] = False
    mask = torch.zeros((b, n), dtype=torch.bool)
    mask[:, 1:3] = True
    return patches, pad, valid, mask


def test_all_mechanisms_finite_and_shaped():
    for name in MECHANISMS + NUISANCES:
        patches, pad, valid, mask = _batch()
        out = corrupt(name, patches, pad, valid, mask, 1.0,
                      torch.Generator().manual_seed(0))
        assert out.shape == patches.shape
        assert bool(torch.isfinite(out).all())


def test_deterministic_under_fixed_seed():
    patches, pad, valid, mask = _batch()
    for name in MECHANISMS:
        a = corrupt(name, patches, pad, valid, mask, 2.0,
                    torch.Generator().manual_seed(7))
        b = corrupt(name, patches, pad, valid, mask, 2.0,
                    torch.Generator().manual_seed(7))
        assert torch.equal(a, b)


def test_severity_ordering_monotone_norm():
    patches, pad, valid, mask = _batch()
    smooth = torch.sin(torch.linspace(0, 6.28, 32)).expand_as(patches).clone()
    for name in MECHANISMS:
        # timing shifts of white noise are still noise: use a smooth signal
        src = smooth if name == "timing_shift" else patches
        norms = []
        for sev in (0.5, 1.0, 2.0, 4.0):
            out = corrupt(name, src, pad, valid, mask, sev,
                          torch.Generator().manual_seed(3))
            delta = (out - src)[mask]
            norms.append(float(delta.pow(2).mean().sqrt()))
        assert norms[0] <= norms[1] <= norms[2] <= norms[3], (name, norms)


def test_support_locality_outside_mask_untouched():
    patches, pad, valid, mask = _batch()
    for name in MECHANISMS + NUISANCES:
        out = corrupt(name, patches, pad, valid, mask, 2.0,
                      torch.Generator().manual_seed(5))
        keep = ~(mask & valid).unsqueeze(-1).unsqueeze(-1).expand_as(patches)
        assert torch.equal(out[keep], patches[keep]), name
        assert bool((out[pad.unsqueeze(2).expand_as(patches)] == 0.0).all()), name


def test_physical_bounds_respected():
    patches, pad, valid, mask = _batch()
    out = corrupt("level_drift", patches, pad, valid, mask, 1.0,
                  torch.Generator().manual_seed(9))
    assert float((out - patches).abs().max()) <= BOUNDS["level_drift"]["max_abs_offset"] + 1e-5
    out = corrupt("transient", patches, pad, valid, mask, 1.0,
                  torch.Generator().manual_seed(9))
    assert float((out - patches).abs().max()) <= BOUNDS["transient"]["max_abs_spike"] + 1e-5


def test_nuisance_probes_are_small():
    patches, pad, valid, mask = _batch()
    mech = corrupt("level_drift", patches, pad, valid, mask, 2.0,
                   torch.Generator().manual_seed(11))
    mech_delta = float((mech - patches)[mask].abs().mean())
    for name in NUISANCES:
        out = corrupt(name, patches, pad, valid, mask, 1.0,
                      torch.Generator().manual_seed(11))
        delta = float((out - patches)[mask].abs().mean())
        assert delta < mech_delta, (name, delta, mech_delta)


def test_rejects_unknown_mechanism_and_bad_shapes():
    patches, pad, valid, mask = _batch()
    with pytest.raises(ValueError):
        corrupt("swap_channels", patches, pad, valid, mask, 1.0)
    with pytest.raises(ValueError):
        corrupt("level_drift", patches, pad, valid, mask, -1.0)
    with pytest.raises(ValueError):
        corrupt("level_drift", patches, pad, valid[:, :3], mask, 1.0)
    assert severity_of(2) == 2.0
