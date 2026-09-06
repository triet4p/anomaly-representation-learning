"""Task 11: hierarchical regularized Mahalanobis geometry."""

from __future__ import annotations

import pytest
import torch

from representation.v2_geometry import HierarchicalMahalanobisGeometry


def _fit_data(seed: int = 0, per_group: int = 40, dim: int = 4):
    rng = torch.Generator().manual_seed(seed)
    groups = []
    for robot in (0, 1):
        for program in (0, 1):
            for regime in (0, 1):
                center = float(robot * 10 + program * 3 + regime)
                block = torch.randn(per_group, dim, generator=rng) + center
                groups.append((block, robot, program, regime))
    latents = torch.cat([g[0] for g in groups])
    robot_ids = torch.cat([torch.full((per_group,), g[1]) for g in groups])
    program_ids = torch.cat([torch.full((per_group,), g[2]) for g in groups])
    regime_ids = torch.cat([torch.full((per_group,), g[3]) for g in groups])
    healthy = torch.ones(latents.shape[0], dtype=torch.bool)
    return latents, robot_ids, program_ids, regime_ids, healthy


def _geometry(**kwargs) -> HierarchicalMahalanobisGeometry:
    params = {"d_model": 4, "diag_min_samples": 16, "min_group_samples": 4}
    params.update(kwargs)
    geo = HierarchicalMahalanobisGeometry(**params)
    latents, robot, program, regime, healthy = _fit_data()
    return geo.fit(latents, robot, program, regime, healthy)


def test_strict_fallback_chain_without_program_only_sharing() -> None:
    geo = _geometry()
    assert geo.resolve(0, 0, 0).level == "robot_program_regime"
    # Unknown regime falls back to (robot, program), never to another robot.
    assert geo.resolve(0, 0, 9).level == "robot_program"
    assert geo.resolve(0, 9, 9).level == "robot"
    fleet = geo.resolve(9, 9, 9)
    assert fleet.level == "fleet_low_confidence" and fleet.low_confidence
    # Same program on a different robot resolves to that robot, not a program pool.
    assert geo.resolve(1, 0, 9).level == "robot_program"
    assert torch.equal(geo.resolve(1, 0, 9).mu, geo._pair[(1, 0)].mu)


def test_singular_and_sparse_groups_stay_finite() -> None:
    geo = HierarchicalMahalanobisGeometry(d_model=4, diag_min_samples=64, min_group_samples=2)
    rows = torch.ones(6, 4)  # rank-deficient: zero empirical variance
    ids = torch.zeros(6, dtype=torch.long)
    geo.fit(rows, ids, ids, ids, torch.ones(6, dtype=torch.bool))
    stats = geo.resolve(0, 0, 0)
    assert torch.isfinite(stats.cov).all()
    out = geo.patch_energy(
        torch.ones(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([[0, 0]]),
    )
    assert torch.isfinite(out["patch_energy"]).all()
    # Sparse single-sample group still resolves through a parent, never NaN.
    geo2 = HierarchicalMahalanobisGeometry(d_model=4, diag_min_samples=64, min_group_samples=4)
    latents, robot, program, regime, healthy = _fit_data(per_group=2)
    geo2.fit(latents, robot, program, regime, healthy)
    assert geo2.resolve(0, 0, 0).level in {"robot_program", "robot", "fleet_low_confidence"}


def test_references_reject_abnormal_and_freeze_for_monitoring() -> None:
    geo = HierarchicalMahalanobisGeometry(d_model=4, diag_min_samples=16, min_group_samples=4)
    latents, robot, program, regime, _ = _fit_data()
    healthy = torch.zeros(latents.shape[0], dtype=torch.bool)
    healthy[:10] = True
    with pytest.raises(ValueError, match="finite|healthy"):
        geo.fit(torch.full_like(latents, float("nan")), robot, program, regime, torch.ones(latents.shape[0], dtype=torch.bool))
    with pytest.raises(ValueError, match="verified-healthy"):
        geo.fit(latents, robot, program, regime, torch.zeros(latents.shape[0], dtype=torch.bool))
    geo.fit(latents, robot, program, regime, healthy)
    frozen = geo.frozen()
    latents_q = torch.randn(1, 2, 4)
    out = frozen.patch_energy(
        latents_q, torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([0]), torch.tensor([0]), torch.tensor([[0, 1]]),
    )
    assert torch.isfinite(out["patch_energy"]).all()
    with pytest.raises(ValueError, match="frozen"):
        geo.fit(latents, robot, program, regime, healthy)


def test_mixture_energy_covers_modes_and_unknown_group_is_low_confidence() -> None:
    geo = _geometry()
    query = torch.tensor([[[10.0, 10.0, 10.0, 10.0]]])  # near robot-1 mode
    valid = torch.ones(1, 1, dtype=torch.bool)
    near = geo.mixture_energy(query, valid, torch.tensor([1]), torch.tensor([0]), torch.tensor([[0]]))
    far = geo.mixture_energy(-query, valid, torch.tensor([1]), torch.tensor([0]), torch.tensor([[0]]))
    assert torch.isfinite(near["patch_energy"]).all()
    assert bool((far["patch_energy"] > near["patch_energy"]).all())
    unknown = geo.patch_energy(query, valid, torch.tensor([9]), torch.tensor([9]), torch.tensor([[9]]))
    assert torch.isfinite(unknown["patch_energy"]).all()
    assert float(unknown["group_confidence"][0, 0]) == pytest.approx(0.25)


def test_scoring_preserves_query_gradients_but_not_reference_params() -> None:
    geo = _geometry()
    for scoring in (geo.patch_energy, geo.mixture_energy):
        query = torch.randn(2, 3, 4, requires_grad=True)
        valid = torch.ones(2, 3, dtype=torch.bool)
        valid[0, 2] = False
        out = scoring(
            query, valid, torch.tensor([0, 1]), torch.tensor([0, 0]),
            torch.tensor([[0, 1, 0], [0, 1, 0]]),
        )
        assert torch.isfinite(out["patch_energy"]).all()
        out["patch_energy"].sum().backward()
        assert query.grad is not None
        assert torch.isfinite(query.grad).all()
        assert bool((query.grad[valid] != 0).any())
        assert bool((query.grad[~valid] == 0).all())
    for stats in (
        list(geo._groups.values()) + list(geo._pair.values())
        + list(geo._robot.values()) + [geo._fleet]
    ):
        assert stats is not None and not stats.mu.requires_grad
        assert not stats.cov.requires_grad


def test_tight_mixture_density_scores_negative_and_validates() -> None:
    from representation.v2_contracts import validate_patch_output

    geo = HierarchicalMahalanobisGeometry(
        d_model=2, shrinkage=0.0, covariance_eps=1e-6,
        min_group_samples=2, diag_min_samples=4,
    )
    rng = torch.Generator().manual_seed(0)
    tight = torch.randn(64, 2, generator=rng) * 0.05 + 5.0
    ids = torch.zeros(64, dtype=torch.long)
    geo.fit(tight, ids, ids, ids, torch.ones(64, dtype=torch.bool))
    query = torch.full((1, 2, 2), 5.0)
    valid = torch.tensor([[True, False]])
    out = geo.mixture_energy(
        query, valid, torch.tensor([0]), torch.tensor([0]), torch.tensor([[0, 0]])
    )
    assert torch.isfinite(out["patch_energy"]).all()
    assert float(out["patch_energy"][0, 0]) < 0.0
    validate_patch_output(
        {
            "patch_latents": query.masked_fill(~valid.unsqueeze(-1), 0.0),
            "patch_energy": out["patch_energy"],
            "patch_valid_mask": valid,
        }
    )
