"""Sprint 24 Task 7-M15 paired-decision design checks (design task only).

Covers the deterministic calculator, the metadata-only seed plan, the
contract build/verify round trip, dilution ladder arithmetic, and the
CLI refusal boundary. No generation, no inference, no forwards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from representation import sprint24_paired_decision_experiment as design  # noqa: E402


def test_sign_test_power_exact_spot_values() -> None:
    assert design.sign_test_power(6, 0.85)["power"] == 0.3772
    assert design.sign_test_power(48, 0.75)["power"] == 0.9296
    assert design.sign_test_power(48, 0.85)["power"] == 0.9996
    assert design.sign_test_power(24, 0.65)["power"] == 0.2106


def test_sign_test_power_critical_values_are_exact_size() -> None:
    first = design.sign_test_power(48, 0.75)
    assert first["critical_value"] == 32
    assert first["alpha_actual"] == 0.0293


def test_t_power_orders_with_n_and_delta() -> None:
    small = design.one_sample_t_power(24, 0.5)["power_approx"]
    large = design.one_sample_t_power(96, 1.0)["power_approx"]
    assert large > small
    assert 0.0 < small < 1.0
    assert 0.0 < large <= 1.0


def test_clopper_pearson_bar_spot_check() -> None:
    bar = design.clopper_pearson(24, 48)
    assert bar["fraction"] == 0.5
    assert bar["ci_lower"] == 0.3523
    assert bar["ci_upper"] == 0.6477
    weak = design.clopper_pearson(5, 48)
    assert weak["ci_lower"] == 0.0347
    assert weak["ci_upper"] == 0.2266
    assert weak["ci_upper"] < 0.5
    gray = design.clopper_pearson(14, 48)
    assert gray["ci_lower"] == 0.1695
    assert gray["ci_upper"] == 0.4406

def test_dilution_counts_follow_m13_rounding() -> None:
    assert design.dilution_counts(40, 0.05)["n_abnormal"] == 2
    assert design.dilution_counts(40, 0.275)["n_abnormal"] == 11
    assert design.dilution_counts(250, 0.15)["n_abnormal"] in (37, 38)


def test_seed_plan_rejects_collisions() -> None:
    assert design.check_seed_plan()["status"] == "PASS"
    with pytest.raises(ValueError, match="REFUSED-SEED-COLLISION"):
        design.check_seed_plan(history_seeds=(32064, 53102, 53103, 53104, 53105, 53106))
    with pytest.raises(ValueError, match="REFUSED-SEED-COLLISION"):
        design.check_seed_plan(history_seeds=(42401, 53102, 53103, 53104, 53105, 53106))


def test_pair_seeds_are_deterministic_and_unique() -> None:
    first = [design.pair_seed(53101, index) for index in range(96)]
    second = [design.pair_seed(53101, index) for index in range(96)]
    assert first == second
    assert len(set(first)) == 96


def test_contract_build_verify_round_trip() -> None:
    contract = design.build_contract(
        {"anchor": 0.06, "warning": "responsiveness only"},
        {"built_by": "test"},
    )
    assert design.verify_contract(contract)["status"] == "PASS"
    tampered = dict(contract)
    tampered["state"] = "EDITED"
    with pytest.raises(ValueError, match="REFUSED-CONTRACT-DRIFT"):
        design.verify_contract(tampered)


def test_power_grid_has_expected_shape() -> None:
    grid = design.power_grid()
    assert len(grid["sign_power"]) == 3 * 3
    assert len(grid["t_power"]) == 3 * 3
    assert {entry["n_pairs"] for entry in grid["sign_power"]} == {24, 48, 96}
