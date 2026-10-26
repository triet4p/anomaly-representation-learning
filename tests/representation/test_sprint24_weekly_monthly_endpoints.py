"""Focused behavioral tests for the frozen weekly/monthly endpoint estimators.

These protect the observable properties the frozen study relies on: the biased
plug-in is zero for identical samples, the unbiased estimator is finite and
non-negative-biased for separated samples, cluster occupancy, location and
dispersion components collapse when the two periods are identical, the capped-n
subsampling reduces to the smaller period, and the primary association uses only
comparable matched pairs.

All fixtures are synthetic; no accepted embedding is copied.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from representation import sprint24_weekly_monthly_endpoints as endpoints
from representation import sprint24_weekly_monthly_study as study

DAY = 86_400.0


class _StubKMeans:
    """Deterministic one-dimensional cluster assignment for occupancy tests."""

    def __init__(self, k: int) -> None:
        self.k = k

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.abs(np.floor(matrix[:, 0]).astype(int)) % self.k


def _fit(sigma: float = 1.0) -> endpoints.ReferenceFit:
    sd = np.ones(study.EMBEDDING_DIM)
    return endpoints.ReferenceFit(
        sigma=sigma,
        sd=sd,
        kmeans={k: _StubKMeans(k) for k in endpoints.CLUSTER_COUNTS},
        null_q95=0.1,
        null_values=[0.0, 0.1],
        n_reference=study.MIN_REFERENCE_FILES,
    )


def _group(matrix: np.ndarray, prefix: str = "S-op") -> endpoints.EmbeddingGroup:
    ids = [f"{prefix}-{index:04d}" for index in range(matrix.shape[0])]
    return endpoints.EmbeddingGroup(
        history_id="SYNTHETIC-WM-01",
        model_seed=181801,
        file_ids=ids,
        index={value: position for position, value in enumerate(ids)},
        matrix=matrix,
    )


def _period(start: float, end: float, starts: np.ndarray, ends: np.ndarray):
    return SimpleNamespace(
        member_starts=starts,
        member_ends=ends,
        period=SimpleNamespace(start=start, end=end),
    )


def test_biased_mmd_is_zero_for_identical_samples() -> None:
    rng = np.random.default_rng(7)
    sample = rng.normal(size=(12, study.EMBEDDING_DIM))
    assert endpoints.mmd2_biased(sample, sample, 2.0) == pytest.approx(0.0, abs=1e-4)


def test_unbiased_mmd_is_near_zero_for_same_distribution_and_positive_for_shift() -> None:
    rng = np.random.default_rng(11)
    first = rng.normal(size=(40, study.EMBEDDING_DIM))
    second = rng.normal(size=(40, study.EMBEDDING_DIM))
    shifted = second + 6.0
    same = endpoints.mmd2_unbiased(first, second, 30.0)
    different = endpoints.mmd2_unbiased(first, shifted, 30.0)
    assert np.isfinite(same) and abs(same) < 0.02
    assert different > same + 0.05


def test_identical_periods_collapse_occupancy_location_and_dispersion() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(8, study.EMBEDDING_DIM)).astype(np.float32)
    group = _group(matrix)
    ids = list(group.file_ids)
    starts = np.zeros(len(ids))
    ends = np.full(len(ids), DAY)
    values = endpoints.pair_endpoint_values(
        previous_ids=ids,
        current_ids=ids,
        previous_period=_period(0.0, DAY, starts, ends),
        current_period=_period(0.0, DAY, starts, ends),
        group=group,
        fit=_fit(),
        seed=181801,
        previous_identity="H|week|calendar|P0",
        current_identity="H|week|calendar|P1",
        history_id="SYNTHETIC-WM-01",
    )
    assert values["status"] == "COMPUTED"
    assert values["E1_biased"] == pytest.approx(0.0, abs=1e-4)
    assert values["E2_K8"] == pytest.approx(0.0, abs=1e-12)
    assert values["E3"] == pytest.approx(0.0, abs=1e-12)
    assert values["E4"] == pytest.approx(0.0, abs=1e-12)
    assert values["E1_capped"] == pytest.approx(values["E1"], rel=1e-9)


def test_capped_variant_matches_the_smaller_period() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(14, study.EMBEDDING_DIM)).astype(np.float32)
    group = _group(matrix)
    previous = list(group.file_ids[:8])
    current = list(group.file_ids[8:14])
    starts = np.zeros(8)
    ends = np.full(8, DAY)
    cur_starts = np.zeros(6)
    cur_ends = np.full(6, DAY)
    values = endpoints.pair_endpoint_values(
        previous_ids=previous,
        current_ids=current,
        previous_period=_period(0.0, DAY, starts, ends),
        current_period=_period(0.0, DAY, cur_starts, cur_ends),
        group=group,
        fit=_fit(),
        seed=181801,
        previous_identity="H|week|calendar|P0",
        current_identity="H|week|calendar|P1",
        history_id="SYNTHETIC-WM-01",
    )
    capped_previous = endpoints._hash_order(
        previous, f"{study.BOOTSTRAP_SEED}|SYNTHETIC-WM-01|H|week|calendar|P0|"
    )[:6]
    capped_current = endpoints._hash_order(
        current, f"{study.BOOTSTRAP_SEED}|SYNTHETIC-WM-01|H|week|calendar|P1|"
    )[:6]
    expected = endpoints.mmd2_unbiased(
        endpoints.group_rows(group, capped_previous),
        endpoints.group_rows(group, capped_current),
        _fit().sigma,
    )
    assert values["E1_capped"] == pytest.approx(expected)


def test_primary_association_rejects_unmatched_pairs() -> None:
    record = {
        "comparison_state": "COMPARABLE-UNMATCHED-EXPOSURE",
        "endpoints": {"181801": {"status": "COMPUTED", "E1": 0.5, "E1_capped": 0.4}},
    }
    endpoint = record["endpoints"]["181801"]
    assert endpoints.e1_matched(record, endpoint) is None
    assert endpoints.e1_capped(record, endpoint) == pytest.approx(0.4)
    matched = {**record, "comparison_state": "COMPARABLE-MATCHED"}
    assert endpoints.e1_matched(matched, endpoint) == pytest.approx(0.5)


def test_pre_outcome_conventions_digest_is_frozen() -> None:
    conventions = endpoints.assert_conventions_frozen()
    assert conventions["canonical_sha256"] == endpoints.CONVENTIONS_CANONICAL_SHA256
