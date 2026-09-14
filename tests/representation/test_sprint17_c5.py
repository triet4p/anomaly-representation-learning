"""Focused Task 12 guards: C5-A block masking/counts, multi-horizon
parity/hand-values, C5-B VICReg finiteness, dormant-lambda training parity,
and registry parity."""

import numpy as np
import pytest
import torch

from representation import sprint17_ablation as A
from representation.criterion import (
    JointRepresentationCriterion,
    LatentPredictionCriterion,
    ProgressiveLambda,
)
from representation.sprint17_c5 import (
    ARM_ADAPTER_DESCRIPTION,
    C5A_HORIZONS,
    FileRedundancyCriterion,
    MultiHorizonPredictionCriterion,
    arm_criterion,
    c5a_block_policy,
    channel_time_block_mask,
)


def _mask_case(n, m, target, seed=11):
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    keep = np.ones(n, dtype=bool)
    keep[rng.choice(n, size=n - m, replace=False)] = False
    valid = order[keep]
    return channel_time_block_mask(n, valid, target, rng)
def test_block_mask_exact_count_valid_only_contiguous():
    for n, m in ((31, 31), (40, 25), (10, 10), (64, 60), (5, 5)):
        target = min(m, int(round(0.40 * m)))
        rng = np.random.default_rng(11)
        order = np.arange(n)
        keep = np.ones(n, dtype=bool)
        keep[rng.choice(n, size=n - m, replace=False)] = False
        valid = order[keep]
        mask, comp = channel_time_block_mask(n, valid, target, rng)
        assert int(mask.sum()) == target
        assert comp == {"time_block": target}
        assert mask.shape == (n,)
        # Contiguous in valid-order: masked positions form at most
        # (number of blocks) runs along the valid order.
        pos = np.flatnonzero(np.isin(valid, np.flatnonzero(mask)))
        runs = 1 + int((np.diff(pos) > 1).sum()) if pos.size else 0
        assert runs <= 2, (n, m, target)


def test_block_mask_deterministic_and_edged():
    a, _ = _mask_case(40, 40, 16, seed=7)
    b, _ = _mask_case(40, 40, 16, seed=7)
    np.testing.assert_array_equal(a, b)
    c, _ = _mask_case(40, 40, 16, seed=8)
    assert not np.array_equal(a, c)
    z, cz = _mask_case(10, 0, 0)
    assert not z.any() and cz == {"time_block": 0}
    e, _ = _mask_case(10, 10, 0)
    assert not e.any()
    f, cf = _mask_case(10, 6, 99)
    assert int(f.sum()) == 6 and cf == {"time_block": 6}
    g, _ = _mask_case(10, 10, 1)
    assert int(g.sum()) == 1


def test_block_policy_adapter_matches_bridge_contract():
    rng = np.random.default_rng(3)
    mask, comp = c5a_block_policy(32, np.arange(32), 13, rng)
    assert mask.dtype == bool and mask.shape == (32,)
    assert int(mask.sum()) == 13 and comp == {"time_block": 13}


def _criterion_inputs(seed=5):
    torch.manual_seed(seed)
    b, n, d = 2, 9, 8
    predicted = torch.randn(b, n, d)
    target = torch.randn(b, n, d)
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, 1:6] = True
    valid = torch.ones(b, n, dtype=torch.bool)
    valid[0, 7:] = False
    return {"predicted_latents": predicted, "target_latents": target,
            "prediction_mask": mask, "patch_valid_mask": valid}


def test_multi_horizon_single_horizon_matches_b0_criterion():
    out = _criterion_inputs()
    b0 = LatentPredictionCriterion()(out)["loss"]
    arm = MultiHorizonPredictionCriterion(horizons=(0,))(out)["loss"]
    assert torch.equal(b0, arm)
def test_multi_horizon_matches_hand_values():
    out = _criterion_inputs()
    got = MultiHorizonPredictionCriterion(horizons=(0, 1, 2))(out)["loss"]
    pred, tgt = out["predicted_latents"], out["target_latents"]
    base = out["prediction_mask"] & out["patch_valid_mask"]
    parts = []
    for h in (0, 1, 2):
        shifted = torch.zeros_like(tgt)
        shifted[:, :9 - h] = tgt[:, h:]
        sel = base & torch.cat(
            [out["patch_valid_mask"][:, h:],
             torch.zeros_like(out["patch_valid_mask"][:, :h])], dim=1)
        err = (pred - shifted).square().mean(-1).masked_fill(~sel, 0.0)
        parts.append(err.sum() / sel.sum().to(err.dtype))
    assert torch.equal(got, torch.stack(parts).mean())
    assert bool(torch.isfinite(got))


def test_multi_horizon_rejects_bad_configs():
    with pytest.raises(ValueError):
        MultiHorizonPredictionCriterion(horizons=())
    with pytest.raises(ValueError):
        MultiHorizonPredictionCriterion(horizons=(0, -1))


def test_vicreg_finite_and_hand_checked():
    crit = FileRedundancyCriterion()
    torch.manual_seed(2)
    z1 = torch.randn(4, 6)
    z2 = torch.randn(4, 6)
    res = crit({"view_embedding_1": z1, "view_embedding_2": z2})
    assert bool(torch.isfinite(res["loss"]))
    assert set(res) == {"loss", "invariance", "variance", "covariance"}
    assert torch.equal(res["invariance"], torch.nn.functional.mse_loss(z1, z2))
    std = z1.std(dim=0, unbiased=False)
    want_var = 0.5 * (torch.relu(1 - std).mean()
                      + torch.relu(1 - z2.std(dim=0, unbiased=False)).mean())
    assert torch.equal(res["variance"], want_var)
    # Single-file batch stays finite (no NaN from degenerate statistics).
    solo = crit({"view_embedding_1": z1[:1], "view_embedding_2": z2[:1]})
    assert bool(torch.isfinite(solo["loss"]))
    with pytest.raises(ValueError):
        crit({"view_embedding_1": z1})


def test_dormant_lambda_makes_c5b_training_identical_to_prediction():
    out = _criterion_inputs()
    out["view_embedding_1"] = torch.randn(2, 8)
    out["view_embedding_2"] = torch.randn(2, 8)
    joint = JointRepresentationCriterion(
        prediction=LatentPredictionCriterion(),
        contrastive=FileRedundancyCriterion(),
        lambda_schedule=ProgressiveLambda(lambda_max=0.1, ramp_steps=980,
                                          warmup_steps=980),
        prediction_weight=1.0)
    terms = joint(out, step=300)
    assert terms["effective_lambda"] == 0.0
    assert torch.equal(terms["joint_loss"], terms["prediction_loss"])
    assert torch.equal(terms["joint_loss"],
                       LatentPredictionCriterion()(out)["loss"])


def test_arm_criterion_construction_and_schedule():
    for arm in ("C5-A", "C5-B"):
        crit = arm_criterion(arm)
        assert isinstance(crit, JointRepresentationCriterion)
        assert crit.lambda_schedule.warmup_steps == 980
        assert crit.lambda_schedule.lambda_max == 0.1
        assert crit.prediction_weight == 1.0
    assert isinstance(arm_criterion("C5-A").prediction,
                      MultiHorizonPredictionCriterion)
    assert isinstance(arm_criterion("C5-B").contrastive,
                      FileRedundancyCriterion)
    with pytest.raises(ValueError):
        arm_criterion("C5-C")


def test_registry_and_parity_declarations():
    from representation.sprint17_ablation import TrainingParity
    for arm in ("C5-A", "C5-B"):
        spec = A.get_arm(arm)
        A.validate_arm(spec)
        assert spec.kind == "single" and spec.components == ("C5",)
        assert ARM_ADAPTER_DESCRIPTION[arm]
    parity = A.validate_training_parity(TrainingParity())
    assert parity.model_seeds == (171701, 171702, 171703)
    assert parity.optimizer_steps == 300 and parity.checkpoint_step == 300
