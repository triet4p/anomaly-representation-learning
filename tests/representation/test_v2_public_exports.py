"""Batch C2 re-gate: public-export contracts for representation and v2_staged.

Every name in ``__all__`` must resolve on both named and wildcard
imports, with no duplicates and no stale entries (regression cover for
the dropped ``v2_inference`` import and the ``BALANCE_MATRIX``
constant/function convention).
"""

from __future__ import annotations

import importlib


def test_representation_all_resolves_named_and_wildcard() -> None:
    import representation

    names = list(representation.__all__)
    assert len(names) == len(set(names)), "representation.__all__ has duplicates"
    namespace: dict[str, object] = {}
    exec("from representation import *", namespace)
    for name in names:
        assert name in namespace, f"representation.__all__ entry {name!r} missing from wildcard import"
        assert getattr(representation, name, None) is not None or hasattr(
            representation, name
        ), f"representation.{name} does not resolve"
    assert "V2InferencePipeline" in namespace
    assert "patch_regime_ids" in namespace


def test_v2_staged_all_resolves_named_and_wildcard() -> None:
    module = importlib.import_module("representation.v2_staged")
    names = list(module.__all__)
    assert len(names) == len(set(names)), "v2_staged.__all__ has duplicates"
    namespace: dict[str, object] = {}
    exec("from representation.v2_staged import *", namespace)
    for name in names:
        assert name in namespace, f"v2_staged.__all__ entry {name!r} missing from wildcard import"


def test_balance_matrix_constant_matches_function() -> None:
    from representation.v2_staged import BALANCE_MATRIX, balance_matrix

    assert isinstance(BALANCE_MATRIX, tuple)
    assert list(BALANCE_MATRIX) == balance_matrix()
    assert [cell["cell"] for cell in BALANCE_MATRIX] == [
        "balance-a",
        "balance-b",
        "balance-c",
        "balance-control",
    ]
