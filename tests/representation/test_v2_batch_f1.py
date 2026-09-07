"""Focused tests for Sprint 11 Batch F1 aggregate helpers (Tasks 33-34)."""

import math

import numpy as np
import pytest

from representation.v2_batch_f1 import (
    anisotropy_ratio,
    confusion,
    describe,
    effective_rank,
    group_rates,
    health_bin,
    mask_overlap_fraction,
    rates_with_counts,
    severity_bin,
    summarize_deltas,
    top_tail_mass,
)


def test_effective_rank_isotropic_vs_collapsed():
    rng = np.random.RandomState(0)
    isotropic = rng.normal(size=(400, 8))
    collapsed = np.zeros((400, 8))
    collapsed[:, 0] = rng.normal(size=400)
    assert effective_rank(isotropic) > effective_rank(collapsed)
    assert effective_rank(isotropic) <= 8.0
    with pytest.raises(ValueError):
        effective_rank(np.full((10, 4), np.nan))


def test_anisotropy_spread_direction():
    rng = np.random.RandomState(1)
    tight = rng.normal(scale=1.0, size=(300, 4))
    stretched = rng.normal(size=(300, 4)) @ np.diag([10.0, 1.0, 1.0, 1.0])
    assert anisotropy_ratio(stretched) > anisotropy_ratio(tight)
    assert anisotropy_ratio(tight) >= 1.0


def test_mask_overlap_fraction_edges():
    flagged = [False] * 100
    for i in range(20, 40):
        flagged[i] = True
    assert mask_overlap_fraction(flagged, 20, 40) == pytest.approx(1.0)
    assert mask_overlap_fraction(flagged, 0, 10) == pytest.approx(0.0)
    assert mask_overlap_fraction(flagged, 30, 50) == pytest.approx(0.5)
    assert mask_overlap_fraction([], 0, 10) == pytest.approx(0.0)


def test_top_tail_mass_sparse_vs_diffuse():
    valid = np.ones(20, dtype=bool)
    sparse = np.zeros(20)
    sparse[0] = 10.0
    diffuse = np.ones(20)
    assert top_tail_mass(sparse, valid, 0.1) > top_tail_mass(diffuse, valid, 0.1)
    assert top_tail_mass(diffuse, valid, 0.1) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        top_tail_mass(np.ones(5), np.ones(4, dtype=bool), 0.1)


def test_bins_are_fixed_and_documented():
    assert severity_bin(0.1) == "low(<0.5)"
    assert severity_bin(0.7) == "mid[0.5,1.0)"
    assert severity_bin(2.0) == "high[>=1.0)"
    assert health_bin(0.05) == "healthy(<0.2)"
    assert health_bin(0.3) == "worn[0.2,0.5)"
    assert health_bin(0.9) == "degraded[>=0.5)"


def test_confusion_and_rates():
    stats = confusion([True, True, False, False], [True, False, True, False])
    assert (stats["tp"], stats["fp"], stats["fn"], stats["tn"]) == (1, 1, 1, 1)
    assert stats["f1"] == pytest.approx(0.5)
    rates = rates_with_counts([])
    assert rates == {"n": 0, "flagged": 0, "rate": 0.0}
    grouped = group_rates(["a", "a", "b"], [True, False, True])
    assert grouped["a"]["rate"] == pytest.approx(0.5)
    assert grouped["b"]["n"] == 1


def test_summarize_deltas_and_describe():
    deltas = summarize_deltas(np.array([1.0, 2.0, 3.0]))
    assert deltas["mean"] == pytest.approx(2.0)
    assert deltas["median"] == pytest.approx(2.0)
    assert deltas["n"] == 3
    desc = describe([1.0, 2.0, 3.0, 4.0])
    assert desc["n"] == 4 and desc["min"] == pytest.approx(1.0)
    assert describe([]) == {"n": 0}
    with pytest.raises(ValueError):
        summarize_deltas(np.array([]))
    with pytest.raises(ValueError):
        describe([1.0, math.inf])
import sys
from pathlib import Path

_EXPERIMENTS_DIR = str(Path(__file__).resolve().parents[2] / "experiments" / "v2_staged")
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

from batch_f1_analyze import (  # noqa: E402
    CONTEXT_ENERGY_FIELD,
    POPULATION_ENERGY_FIELD,
    _localization_scores,
    _signal_block,
    build_task34,
    continuity_probe,
    retention_probe,
    severity_energy,
)


def _fake_file(
    file_id: str,
    split: str,
    abnormal: bool,
    tail_context: float,
    tail_population: float,
    confidence: float,
    robot: str = "robot-01",
    program: str = "program-01",
) -> dict:
    seed = abs(hash(file_id)) % (2**31)
    rng = np.random.RandomState(seed)
    return {
        "file_id": file_id,
        "split": split,
        "abnormal": abnormal,
        "family": "freq_phase" if abnormal else "normal",
        "severity": 0.7 if abnormal else 0.0,
        "robot": robot,
        "program": program,
        "dominant_regime": "active",
        "has_mask": abnormal,
        "start_time": float(seed % 100),
        "tail_energy_context": tail_context,
        "mean_energy_context": tail_context - 1.0,
        "elevated_fraction_context": 0.9 if abnormal else 0.2,
        "top_tail_mass_context": 0.3,
        "tail_energy_population": tail_population,
        "mean_energy_population": tail_population - 1.0,
        "elevated_fraction_population": 0.8 if abnormal else 0.3,
        "top_tail_mass_population": 0.25,
        "confidence": confidence,
        "argmax_hit_context": abnormal,
        "top3_overlap_context": 0.5 if abnormal else 0.0,
        "argmax_hit_population": False,
        "top3_overlap_population": 0.1,
        "file_state_context": rng.normal(size=8),
        "file_state_population": rng.normal(size=8),
    }


def _fake_scored(static_rows: list[dict], dev_val_rows: list[dict]) -> dict:
    return {
        "files": static_rows + dev_val_rows,
        "restored_operating_threshold": {
            "energy_field": "context_energy",
            "fit_cohort": "dev-val",
            "method": "healthy-validation-quantile",
            "n_samples": 330,
            "quantile": 0.95,
            "value": -86.0,
        },
        "restored_confidence_cohort": {
            "cohort": "dev-val",
            "min_samples": 4,
            "n_samples": 11,
            "small_sample": True,
        },
        "restored_elevated_threshold": -86.0,
        "elevated_threshold_source": "calibration-record (dev-val)",
    }


def test_localization_scores_hit_and_empty():
    energy = np.array([0.1, 0.9, 0.2, 0.3])
    valid = np.array([True, True, True, True])
    flagged = [False] * 64
    for i in range(16, 32):
        flagged[i] = True
    hit, overlap = _localization_scores(energy, valid, [0, 16, 32, 48], 16, flagged)
    assert hit is True
    assert overlap == pytest.approx((1.0 + 0.0 + 0.0) / 3.0)
    assert _localization_scores(
        np.array([0.5]), np.array([False]), [0], 16, flagged
    ) == (None, None)


def test_signal_block_reports_decisions_and_program_slice():
    rows = [
        _fake_file(f"s{i}", "static", abnormal=(i % 2 == 0), tail_context=10.0 + i,
                   tail_population=20.0 + i, confidence=0.9 if i % 2 == 0 else 0.1)
        for i in range(6)
    ]
    for f in rows:
        f["_decision_confidence"] = f["confidence"] >= 0.5
    block = _signal_block(rows, "tail_energy_context", "_decision_confidence",
                          "context")
    assert block["energy_source"] == CONTEXT_ENERGY_FIELD
    assert (block["tp"], block["fp"], block["fn"], block["tn"]) == (3, 0, 0, 3)
    assert block["f1"] == pytest.approx(1.0)
    assert "fp_by_program" in block and "fp_by_pair" in block
    assert block["localization"]["n_masked_abnormal"] == 3
    assert block["localization"]["argmax_hit"]["rate"] == pytest.approx(1.0)


def test_build_task34_reports_both_signals_with_provenance():
    def make_rows(shift: float) -> tuple[list[dict], list[dict]]:
        static = [
            _fake_file(f"s{i}", "static", abnormal=(i < 4),
                       tail_context=10.0 + i + shift,
                       tail_population=20.0 + i + shift,
                       confidence=0.95 if i < 4 else 0.05)
            for i in range(6)
        ]
        dev_val = [
            _fake_file(f"v{i}", "dev_val", abnormal=False,
                       tail_context=9.0 + 0.1 * i, tail_population=19.0 + 0.1 * i,
                       confidence=0.1 + 0.01 * i)
            for i in range(4)
        ]
        return static, dev_val
    sc, vc = make_rows(0.0)
    sh, vh = make_rows(-1.0)
    task34 = build_task34(_fake_scored(sc, vc), _fake_scored(sh, vh))
    for variant in ("control", "hybrid"):
        assert task34[variant]["reference_source"] == "restored-checkpoint"
        assert task34[variant]["restored_operating_threshold"]["fit_cohort"] == "dev-val"
        assert task34[variant]["restored_confidence_cohort"]["small_sample"] is True
        for signal, field in (("context", CONTEXT_ENERGY_FIELD),
                              ("population", POPULATION_ENERGY_FIELD)):
            block = task34[variant][signal]
            assert block["energy_source"] == field
            assert "auroc_tail" in block and "fp_by_program" in block
            assert "recall_by_family" in block and "localization" in block
        assert task34[variant]["confidence_decision"]["threshold"] >= 0.0
    deltas = task34["deltas_hybrid_minus_control"]
    assert "context" in deltas and "population" in deltas
    assert deltas["context"]["tail_energy"]["mean"] == pytest.approx(-1.0)
    assert "threshold_note" in task34


def test_task33_probes_report_per_signal():
    static = [
        _fake_file(f"s{i}", "static", abnormal=(i % 2 == 0), tail_context=10.0 + i,
                   tail_population=20.0 + i, confidence=0.5)
        for i in range(6)
    ]
    dev = [
        _fake_file(f"d{i}", "dev_train", abnormal=False, tail_context=9.0,
                   tail_population=19.0, confidence=0.2)
        for i in range(3)
    ] + [
        _fake_file(f"v{i}", "dev_val", abnormal=False, tail_context=9.5,
                   tail_population=19.5, confidence=0.3)
        for i in range(2)
    ]
    scored = {"files": static + dev}
    retention = retention_probe(scored, scored)
    assert set(retention["control"]) == {"context", "population"}
    severity = severity_energy(scored, scored)
    assert "context" in severity["hybrid"]
    continuity = continuity_probe(scored, scored)
    assert "population" in continuity["control"]
