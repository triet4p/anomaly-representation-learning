"""Unbiased fixed-composition mixture MMD with distinct reference observations."""

from __future__ import annotations

import math

import numpy as np

from representation.sprint24_weekly_monthly_endpoints import _kernel


def stratified_mmd2_unbiased(
    abnormal: np.ndarray, ordinary: np.ndarray, reference: np.ndarray, sigma: float,
) -> float:
    """Estimate MMD² of the realized weighted component mixture against reference.

    The components and reference must consist of independent iid source draws
    within their respective strata. A nonempty stratum requires at least two
    observations; zero-weight components are omitted. The function does not
    assume iid sampling across a fixed-count heterogeneous mixture.
    """
    a = np.asarray(abnormal, dtype=np.float64)
    o = np.asarray(ordinary, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if (a.ndim != 2 or o.ndim != 2 or r.ndim != 2 or a.shape[1] != r.shape[1]
            or o.shape[1] != r.shape[1] or len(r) < 2 or len(a) + len(o) < 2
            or len(a) == 1 or len(o) == 1 or not math.isfinite(sigma) or sigma <= 0
            or not all(np.isfinite(x).all() for x in (a, o, r))):
        raise ValueError("REFUSED-ERROR-BOUNDARY: stratified MMD needs at least two observations per nonempty stratum and reference")

    def off_diagonal_mean(matrix: np.ndarray) -> float:
        k = _kernel(matrix, matrix, sigma)
        return float((k.sum() - np.trace(k)) / (len(matrix) * (len(matrix) - 1)))

    total = len(a) + len(o)
    wa, wo = len(a) / total, len(o) / total
    value = off_diagonal_mean(r)
    if len(a):
        value += wa * wa * off_diagonal_mean(a) - 2 * wa * float(_kernel(a, r, sigma).mean())
    if len(o):
        value += wo * wo * off_diagonal_mean(o) - 2 * wo * float(_kernel(o, r, sigma).mean())
    if len(a) and len(o):
        value += 2 * wa * wo * float(_kernel(a, o, sigma).mean())
    return float(value)
