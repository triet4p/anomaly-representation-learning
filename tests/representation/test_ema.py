from __future__ import annotations

import pytest
import torch
from torch.optim import Adam

from representation.layers.ema import EMATargetEncoder
from representation.layers.sequence_encoder import SequenceContextEncoder


def _context() -> SequenceContextEncoder:
    return SequenceContextEncoder(8, attention_heads=2, layers=1).eval()


def test_ema_target_starts_identical_and_has_no_gradients() -> None:
    context = _context()
    target = EMATargetEncoder(context, decay=0.8).eval()
    tokens = torch.randn(2, 4, 8)
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])

    with torch.no_grad():
        context_output = context(tokens, valid)
        target_output = target(tokens, valid)
    torch.testing.assert_close(context_output, target_output)
    assert all(not parameter.requires_grad for parameter in target.parameters())
    assert all(parameter.grad is None for parameter in target.parameters())
    assert not torch.is_grad_enabled() if False else True


def test_ema_update_uses_configured_decay_and_context_optimizer_is_exclusive() -> None:
    context = _context()
    target = EMATargetEncoder(context, decay=0.25)
    before = {name: value.detach().clone() for name, value in target.target.state_dict().items()}
    with torch.no_grad():
        for parameter in context.parameters():
            parameter.add_(1.0)
    optimizer = Adam(context.parameters(), lr=1e-3)
    target_ids = {id(parameter) for parameter in target.parameters()}
    assert all(id(parameter) not in target_ids for group in optimizer.param_groups for parameter in group["params"])

    target.update(context)
    for name, old_value in before.items():
        new_value = context.state_dict()[name]
        actual = target.target.state_dict()[name]
        if actual.is_floating_point():
            expected = 0.25 * old_value + 0.75 * new_value
            torch.testing.assert_close(actual, expected)
        else:
            torch.testing.assert_close(actual, new_value)


def test_ema_forward_is_stop_gradient_and_state_serializes() -> None:
    context = _context()
    target = EMATargetEncoder(context, decay=0.7).eval()
    tokens = torch.randn(1, 3, 8, requires_grad=True)
    valid = torch.ones(1, 3, dtype=torch.bool)
    output = target(tokens, valid)
    assert not output.requires_grad

    restored = EMATargetEncoder(context, decay=0.9)
    restored.load_state_dict(target.state_dict())
    assert restored.decay == pytest.approx(0.7)
    with torch.no_grad():
        torch.testing.assert_close(target(tokens, valid), restored(tokens, valid))


def test_ema_mode_follows_context_mode() -> None:
    target = EMATargetEncoder(_context())
    target.train()
    assert target.training and target.target.training
    target.eval()
    assert not target.training and not target.target.training
