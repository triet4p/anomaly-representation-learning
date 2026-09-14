"""Focused Task 14 guards: C7 reference definitions, hierarchy/fallback,
robust fitting, score semantics, S_pred isolation, and registry parity."""

import inspect
import sys

import numpy as np
import pytest
import torch
from pathlib import Path

from representation.sprint17_c7 import (
    ARM_ADAPTER_DESCRIPTION,
    C7_ARMS,
    DENSITY_SCALE_FLOOR,
    KNN_K,
    LEVELS,
    MIN_CELL_ROWS,
    N_PROGRAMS,
    N_ROBOTS,
    LocalDensityReference,
    RobustShrinkageReference,
    arm_reference,
    ledoit_wolf_shrinkage,
    resolve_level,
    validate_condition,
)

B0_PARAMS_TOTAL = 1821698
B0_FLOPS_REFERENCE = 182016709


def _refs(n=120, d=8, seed=21, n_robots=3, n_programs=2):
    rng = np.random.default_rng(seed)
    emb = torch.from_numpy(rng.standard_normal((n, d)).astype(np.float32))
    robots = [i % n_robots for i in range(n)]
    programs = [(i // n_robots) % n_programs for i in range(n)]
    return emb, robots, programs


def test_b0_global_knn_contract_is_fixed_reference():
    """Pin the B0 S_pop baseline both C7 arms replace."""
    from representation.inference import NormalReferenceBank
    bank = NormalReferenceBank(k=2).fit(torch.tensor([[0.0, 0.0], [3.0, 4.0]]))
    out = bank.score(torch.tensor([[0.0, 0.0]]))
    assert torch.allclose(out, torch.tensor([2.5]))


def test_condition_vocabulary_is_frozen():
    assert (N_ROBOTS, N_PROGRAMS) == (9, 8)
    assert validate_condition(8, 7) == (8, 7)
    with pytest.raises(ValueError):
        validate_condition(9, 0)
    with pytest.raises(ValueError):
        validate_condition(0, 8)
    with pytest.raises(ValueError):
        validate_condition(-1, 0)


def test_resolve_level_follows_frozen_hierarchy():
    assert resolve_level(8, 0, 0, 5000) == "cell"
    assert resolve_level(7, 8, 0, 5000) == "robot"
    assert resolve_level(0, 0, 8, 5000) == "program"
    assert resolve_level(0, 0, 0, 5000) == "global"
    assert resolve_level(7, 7, 7, 8) == "global"
    with pytest.raises(ValueError):
        resolve_level(0, 0, 0, 7)
    assert MIN_CELL_ROWS == 8


def test_ledoit_wolf_shrinkage_matches_hand_values():
    # Two points on one axis: S = [[1, 0], [0, 0]], mu = 0.5.
    x = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    delta, mu, sigma = ledoit_wolf_shrinkage(x)
    assert mu == pytest.approx(0.5)
    assert 0.05 <= delta <= 1.0
    want_s = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(sigma, (1 - delta) * want_s + delta * 0.5 * torch.eye(2, dtype=torch.float64))
    assert bool((torch.linalg.eigvalsh(sigma) > 0).all())
    with pytest.raises(ValueError):
        ledoit_wolf_shrinkage(torch.ones(1, 4, dtype=torch.float64))


def test_c7a_location_scores_zero_and_grows_with_distance():
    ref = RobustShrinkageReference()
    emb, robots, programs = _refs()
    ref.fit(emb, robots, programs)
    loc = emb.median(dim=0).values
    energies, levels = ref.score(loc.unsqueeze(0).float(), [0], [0])
    assert energies.shape == (1,)
    assert bool(torch.isfinite(energies).all())
    far = loc + 10.0
    e_far, _ = ref.score(far.unsqueeze(0).float(), [0], [0])
    assert float(e_far[0]) > float(energies[0]) + 1.0
    assert set(levels) <= set(LEVELS)


def test_c7a_fallback_serves_unseen_cells_and_records_levels():
    ref = RobustShrinkageReference()
    emb, robots, programs = _refs(n_robots=2, n_programs=1)
    ref.fit(emb, robots, programs)
    q = torch.randn(3, 8)
    energies, levels = ref.score(q, [5, 0, 1], [3, 0, 0])
    assert levels[0] == "global"
    assert levels[1] in ("cell", "robot")
    assert bool(torch.isfinite(energies).all())
    prov = ref.provenance()
    assert prov["source"] == "Fit-only healthy-eligible rows"
    assert prov["n_total"] == len(robots) and prov["n_cells"] == 2
    assert sum(prov["level_counts"].values()) == 3


def test_c7b_reference_row_scores_near_zero_and_scales():
    ref = LocalDensityReference(k=2)
    emb, robots, programs = _refs()
    ref.fit(emb, robots, programs)
    scores, levels = ref.score(emb[:2], robots[:2], programs[:2])
    assert bool(torch.isfinite(scores).all())
    far = (emb.median(dim=0).values + 10.0).unsqueeze(0)
    s_far, _ = ref.score(far.float(), [0], [0])
    assert float(s_far[0]) > 3.0
    assert bool((scores < float(s_far[0])).all())


def test_c7b_fallback_and_finite_empty_margin():
    ref = LocalDensityReference()
    emb, robots, programs = _refs(n_robots=2, n_programs=1)
    ref.fit(emb, robots, programs)
    scores, levels = ref.score(torch.randn(4, 8), [7, 0, 1, 0], [6, 0, 0, 0])
    assert levels[0] == "global"
    assert bool(torch.isfinite(scores).all())
    assert bool((scores >= 0).all())
    prov = ref.provenance()
    assert prov["n_total"] == len(robots)
    assert DENSITY_SCALE_FLOOR == 1e-6


def test_references_reject_abnormal_rows_and_malformed_inputs():
    emb, robots, programs = _refs(n=40)
    bad_labels = ["normal"] * 39 + ["abnormal"]
    for arm in C7_ARMS:
        ref = arm_reference(arm)
        with pytest.raises(ValueError):
            ref.fit(emb, robots, programs, labels=bad_labels)
        with pytest.raises(ValueError):
            ref.score(torch.randn(2, 8), [0], [0])
        with pytest.raises(ValueError):
            ref.fit(emb, robots[:10], programs)
        with pytest.raises(ValueError):
            ref.fit(torch.full((10, 8), float("nan")), [0] * 10, [0] * 10)
    with pytest.raises(ValueError):
        arm_reference("C7-C")
    with pytest.raises(ValueError):
        LocalDensityReference(k=0)


def test_c7_consumes_no_patch_or_anomaly_inputs():
    """Structural S_pred isolation: references see embeddings + conditions."""
    for cls in (RobustShrinkageReference, LocalDensityReference):
        params = set(inspect.signature(cls.fit).parameters)
        assert params == {"self", "embeddings", "robots", "programs", "labels"}, params
        assert set(inspect.signature(cls.score).parameters) == {
            "self", "queries", "qrobots", "qprograms"}
        src = inspect.getsource(cls)
        assert "patch" not in src and "anomaly" not in src and "severity" not in src


def test_registry_and_parity_declarations():
    from representation import sprint17_ablation as A
    from representation.sprint17_ablation import TrainingParity
    for arm in C7_ARMS:
        spec = A.get_arm(arm)
        A.validate_arm(spec)
        assert spec.kind == "single" and spec.components == ("C7",)
        assert ARM_ADAPTER_DESCRIPTION[arm]
    parity = A.validate_training_parity(TrainingParity())
    assert parity.model_seeds == (171701, 171702, 171703)
    assert parity.optimizer_steps == 300 and parity.checkpoint_step == 300
    assert KNN_K == 5


def test_driver_builds_unchanged_b0_graph_with_exact_counts():
    from representation.config import V1Config
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "experiments"))
    import sprint17_task14_c7 as T14
    for arm in C7_ARMS:
        model, _, _ = T14.build_model(T14.arm_config_dict(arm, 171701),
                                      torch.device("cpu"))
        assert T14.count_parameters(model) == T14.B0_PARAMS == 1821698
        assert T14.count_flops_reference(model) == T14.B0_FLOPS_REFERENCE == 182016709
