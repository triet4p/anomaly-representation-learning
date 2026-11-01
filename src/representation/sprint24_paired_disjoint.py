"""Predeclared source-disjoint, joint-encoder permutation statistic for C3."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from representation.sprint24_weekly_monthly_endpoints import _kernel


PERMUTATIONS = 9999


def _seed(domain: str, history_seed: int, family: str, severity: float) -> int:
    identity = f"{domain}|{int(history_seed)}|{family}|{float(severity):.9g}"
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")


def source_disjoint_test(
    planned_indices: Sequence[int],
    admitted_indices: Sequence[int],
    matrices: Mapping[int, Mapping[str, Any]],
    sigma_per_seed: Mapping[int, float],
    history_seed: int,
    family: str,
    severity: float,
) -> dict[str, Any]:
    """Test joint encoder features under independent-source exchangeability.

    The split is chosen over all planned sources, not conditioned on outcomes or
    membership. Each retained pair contributes one arm only. Conditional p-values
    are meaningful only under the scientific assumptions in the C03 contract.
    """
    planned = [int(i) for i in planned_indices]
    admitted = [int(i) for i in admitted_indices]
    if len(set(planned)) != len(planned) or len(set(admitted)) != len(admitted):
        raise ValueError("REFUSED-IDENTITY: repeated source index")
    if not set(admitted).issubset(planned):
        raise ValueError("REFUSED-IDENTITY: admitted source outside plan")
    split_seed = _seed("s24-c3-source-split-v1", history_seed, family, severity)
    permutation_seed = _seed("s24-c3-permutation-v1", history_seed, family, severity)
    shuffled = np.random.default_rng(split_seed).permutation(planned).tolist()
    twin_source = set(shuffled[:len(shuffled) // 2])
    twin = [i for i, index in enumerate(admitted) if index in twin_source]
    variant = [i for i, index in enumerate(admitted) if index not in twin_source]
    result: dict[str, Any] = {
        "twin_pair_indices": [admitted[i] for i in twin],
        "variant_pair_indices": [admitted[i] for i in variant],
        "split_seed": split_seed, "permutation_seed": permutation_seed,
        "permutations": PERMUTATIONS,
    }
    n, m = len(twin), len(variant)
    if n < 2 or m < 2:
        return {**result, "status": "ABSTAIN-COVERAGE", "p_value": 1.0,
                "reason": "fewer than two admitted sources in one arm"}
    combined: np.ndarray | None = None
    for seed in sorted(sigma_per_seed):
        source = matrices[int(seed)]
        x, y = np.asarray(source["twin"], dtype=np.float64), np.asarray(source["abnormal"], dtype=np.float64)
        if x.ndim != 2 or y.shape != x.shape or len(x) != len(admitted):
            raise ValueError("REFUSED-IDENTITY: source features disagree with admitted pairs")
        vectors = np.concatenate((x[twin], y[variant]), axis=0)
        sigma = float(sigma_per_seed[int(seed)])
        if not np.isfinite(vectors).all() or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("REFUSED-ERROR-BOUNDARY: non-finite source features or bandwidth")
        kernel = _kernel(vectors, vectors, sigma)
        combined = kernel if combined is None else combined + kernel
    if combined is None:
        raise ValueError("REFUSED-ERROR-BOUNDARY: missing encoder features")
    combined /= len(sigma_per_seed)
    # For binary selector z, z'Kz and z'K1 produce both within-arm sums
    # and the cross sum; diagonals are removed separately in the U-statistic.
    total = n + m
    row_sums = combined.sum(axis=1)
    all_sum = float(row_sums.sum())
    diagonal = np.diag(combined)
    diagonal_sum = float(diagonal.sum())

    def statistics(selectors: np.ndarray) -> np.ndarray:
        within_x_all = np.einsum("ij,ij->i", selectors @ combined, selectors)
        x_all = selectors @ row_sums
        diagonal_x = selectors @ diagonal
        within_x = within_x_all - diagonal_x
        cross = x_all - within_x_all
        within_y = all_sum - 2 * x_all + within_x_all - (diagonal_sum - diagonal_x)
        return within_x / (n * (n - 1)) + within_y / (m * (m - 1)) - 2 * cross / (n * m)

    observed_selector = np.zeros((1, total), dtype=np.float64)
    observed_selector[0, :n] = 1.0
    observed = float(statistics(observed_selector)[0])
    rng = np.random.default_rng(permutation_seed)
    exceedances = 0
    for start in range(0, PERMUTATIONS, 128):
        count = min(128, PERMUTATIONS - start)
        selectors = np.zeros((count, total), dtype=np.float64)
        for row in selectors:
            row[rng.permutation(total)[:n]] = 1.0
        exceedances += int(np.count_nonzero(statistics(selectors) >= observed - 1e-12))
    return {**result, "status": "COMPUTED", "statistic": observed,
            "exceedances": exceedances, "p_value": (1 + exceedances) / (PERMUTATIONS + 1),
            "null": "joint encoder feature exchangeability conditional on admission and reference"}
