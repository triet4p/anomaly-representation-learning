"""Focused contract tests for the Sprint 18 supplementary metric source.

These protect observable behaviour of the new supplementary contract:

* exact Sprint 17 metric semantics (support mirror, localization, severity,
  slices, stability, duration accounting);
* the frozen §8.2 aggregation and history-block uncertainty;
* the additive closed schema, its v2-compatible projection, and the refusal of
  a supplementary row by the unchanged v2 validator;
* frozen input-manifest validation (metric-code lock, arm/phase/scope, digests).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from representation import sprint18_supplementary as S
from representation.sprint18_harness import _validate_history_metric

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASK14_V1_LOCK = _REPO_ROOT / "experiments/sprint18-task14-config-lock.json"
_TASK14_V9_LOCK = _REPO_ROOT / "experiments/sprint18-task14-config-lock-v9.json"


def _row(file_id: str, end: float, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "file_id": file_id,
        "robot_id": "robot-01",
        "program_id": "program-01",
        "file_label": "normal",
        "is_quarantined": False,
        "is_censored": False,
        "start_time": end - 600.0,
        "end_time": end,
        "member_views": ["test-temporal"],
    }
    row.update(extra)
    return row


def _ledger_row(cohort: str, failure_time: float, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "cohort": cohort,
        "robot_id": "robot-01",
        "failure_time": failure_time,
        "severity": 1.0,
        "subtype": f"{cohort}1",
        "duration_d": 1.0,
    }
    row.update(extra)
    return row


def _support() -> S.EvaluableSupport:
    rows = [_row(f"f{index}", 600.0 * index) for index in range(1, 13)]
    ledger = [
        _ledger_row("P", 7200.0, severity=2.0, subtype="P1", duration_d=1.5),
        _ledger_row("P", 7200.0, severity=4.0, subtype="P2", duration_d=2.5),
        _ledger_row("P", 7200.0, severity=4.0, subtype="P2", duration_d=3.5),
        _ledger_row("W", 7200.0, severity=1.0, subtype="W1", duration_d=0.0),
        _ledger_row("A", 7200.0, severity=4.0, subtype="A1", duration_d=0.2),
    ]
    return S.evaluable_support(rows, ledger, {"robot-01": []})


def _scores() -> dict[str, float]:
    return {f"f{index}": float(index) for index in range(1, 13)}


def test_evaluable_support_mirrors_the_frozen_selection() -> None:
    support = _support()
    assert [event.cohort for event in support.events] == ["P", "P", "P", "W", "A"]
    assert support.events[0].key == ("P", "robot-01", 7200.0)
    assert support.events[0].members == tuple(f"f{index}" for index in range(1, 13))
    assert support.skipped == 0
    # No control window can exist without anchor-eligible rows outside a failure
    # horizon; the frozen selector returns an empty list here.
    assert support.controls == ()


def _event(failure_time: float, members: tuple[str, ...] = ("top",)) -> S.EvaluableEvent:
    return S.EvaluableEvent(
        key=("P", "robot-01", failure_time),
        cohort="P",
        members=members,
        severity=1.0,
        subtype="P1",
        duration_days=1.0,
        failure_time=failure_time,
    )


def test_localization_uses_peak_timestep_inside_the_horizon() -> None:
    horizon = 7.0 * 86400.0
    failure = 10.0 * horizon
    start_of_horizon = failure - horizon
    # File ends inside the horizon; an increasing vector peaks at the file end.
    inside = {
        "top": (start_of_horizon + 100.0, start_of_horizon + 700.0),
    }
    event = _event(failure)
    rate, hits, evaluated, excluded = S.localization_rate(
        (event,),
        {"top": 5.0},
        S.localization_diagnostics({"top": np.linspace(0.0, 1.0, 40)}),
        inside,
    )
    assert (rate, hits, evaluated, excluded) == (1.0, 1, 1, 0)

    # A file straddling the horizon start whose peak sits before the horizon is a
    # genuine localization miss (never silently counted as a hit).
    straddling = {"top": (start_of_horizon - 600.0, start_of_horizon)}
    rate, hits, evaluated, _ = S.localization_rate(
        (event,),
        {"top": 5.0},
        S.localization_diagnostics({"top": np.linspace(1.0, 0.0, 40)}),
        straddling,
    )
    assert (rate, hits, evaluated) == (0.0, 0, 1)

    # Empty, non-finite, and constant vectors are exclusions, never misses.
    for unusable in (np.zeros(40), np.array([]), np.array([0.0, math.nan])):
        rate, _hits, evaluated, excluded = S.localization_rate(
            (event,),
            {"top": 5.0},
            S.localization_diagnostics({"top": unusable}),
            inside,
        )
        assert rate is None
        assert evaluated == 0
        assert excluded == 1


def test_severity_ordering_requires_three_points_and_isfinite() -> None:
    support = _support()
    ordering = S.severity_ordering(support.events, _scores())
    # Three P events with severities 2/4/4 share the same event score (all members
    # score above every file), so the cohort is constant -> None; W has one point.
    assert ordering["P"] is None
    assert ordering["W"] is None

    # Distinct event windows with distinct max scores give a rankable cohort.
    rows = [_row(f"g{index}", 600.0 * index) for index in range(1, 13)]
    ledger = [
        _ledger_row("P", 1200.0, severity=1.0, subtype="P1"),
        _ledger_row("P", 2400.0, severity=2.0, subtype="P1"),
        _ledger_row("P", 3600.0, severity=4.0, subtype="P1"),
    ]
    support = S.evaluable_support(rows, ledger, {"robot-01": []})
    scores = {f"g{index}": float(index) for index in range(1, 13)}
    ordering = S.severity_ordering(support.events, scores)
    assert ordering["P"] == pytest.approx(1.0)
    assert ordering["W"] is None


def test_subtype_slices_use_the_identical_control_negatives() -> None:
    support = _support()
    controls = [0.0, 0.5]
    slices = S.subtype_slices(support.events, _scores(), controls)
    assert set(slices) == {"P1", "P2", "W1", "W2", "A1", "A2"}
    assert slices["P1"] == 1.0
    assert slices["P2"] == 1.0
    assert slices["W2"] is None  # no W2 event exists
    assert slices["A2"] is None
    assert S.subtype_slices(support.events, _scores(), [])["P1"] is None


def test_background_stability_is_the_frozen_formula_or_none() -> None:
    background = [1.0, 2.0, 3.0]
    fit = [1.1, 2.1, 3.1]
    ratio = S.background_stability_ratio(background, fit)
    from representation.attribution_metrics import stability_ratio

    assert ratio == pytest.approx(stability_ratio(background, fit)["ratio"])
    assert S.background_stability_ratio([], fit) is None
    assert S.background_stability_ratio(background, []) is None
    with pytest.raises(ValueError):
        S.background_stability_ratio([math.nan], fit)


def test_duration_accounting_counts_recall_and_exclusions() -> None:
    support = _support()
    times = {f"f{index}": (600.0 * index - 600.0, 600.0 * index) for index in range(1, 13)}
    record = S.duration_accounting(
        support.events, _scores(), times, 5.0, excluded_events=support.skipped
    )
    assert record["schema_id"] == S.DURATION_ACCOUNTING_ID
    assert record["evaluated_events"] == 4
    assert record["recalled_events"] + record["unrecalled_events"] == 4
    assert record["recalled_events"] == 4
    assert record["events_without_positive_duration"] == 1
    assert record["median_declared_duration_days"] == pytest.approx(2.5)
    assert record["episode_grouping_days"] == pytest.approx(2.0)
    assert record["excluded_events"] == 0
    low = S.duration_accounting(support.events, _scores(), times, 10_000.0, excluded_events=2)
    assert low["recalled_events"] == 0
    assert low["median_first_alert_lead_days"] is None
    assert low["total_detected_alert_days"] == 0.0
    assert low["excluded_events"] == 2


def test_history_block_interval_is_deterministic_and_needs_four_units() -> None:
    values = [0.01, 0.02, 0.03, 0.04]
    first = S.history_block_interval(values)
    assert first == S.history_block_interval(values)
    assert len(first) == 2 and first[0] <= np.mean(values) <= first[1]
    rng = np.random.default_rng(S.P.BOOTSTRAP_SEED)
    draws = rng.integers(0, 4, size=(S.P.BOOTSTRAP_REPLICATES, 4))
    macros = np.asarray(values)[draws].mean(axis=1)
    assert first[0] == pytest.approx(float(np.quantile(macros, 0.025)))
    assert first[1] == pytest.approx(float(np.quantile(macros, 0.975)))
    for bad in ([0.1, 0.2, 0.3], [0.1] * 5, [0.1, math.nan, 0.3, 0.4]):
        with pytest.raises(ValueError):
            S.history_block_interval(bad)


def _history(history_id: str, schema_id: str, threshold: float) -> dict[str, object]:
    support = _support()
    rows = tuple(_row(f"f{index}", 600.0 * index) for index in range(1, 13))
    ledger = (_ledger_row("P", 7200.0, severity=2.0, subtype="P1"),)
    scores = _scores()
    times = S.row_times(rows)
    canonical = {
        "auc_p": 0.7,
        "auc_w": 0.6,
        "auc_pw": 0.65,
        "auc_a": 0.5,
        "recall_p": 0.5,
        "lead_p_median": 1.0,
        "recall_w": 0.3,
        "lead_w_median": 0.5,
        "far": 0.01,
    }
    history = S.HistoryEvaluationInput(
        history_id=history_id,
        schema_history_id=schema_id,
        role_manifest_sha256="a" * 64,
        support_sha256="b" * 64,
        rows=rows,
        ledger=ledger,
        windows={"robot-01": []},
        scores=scores,
        localization=S.localization_diagnostics(
            {file_id: np.linspace(0.0, 1.0, 30) for file_id in scores}
        ),
        fit_healthy_scores=tuple(float(index) for index in range(1, 13)),
        excluded_rows=0,
    )
    baseline = {"P_AUROC": 0.6, "W_AUROC": 0.5, "PW_AUROC": 0.55, "FAR_per_robot_day": 0.0}
    return S.build_history_metric_row(
        history, threshold=threshold, canonical_entry=canonical, baseline=baseline
    )


def _branch_rows(seed: int) -> dict[str, dict[str, object]]:
    return {
        f"H-S18-DEV-0{index}": _history(
            f"ACTUAL-DEV-0{index}", f"H-S18-DEV-0{index}", 0.1 * (index + seed)
        )
        for index in range(1, 5)
    }


def test_history_row_covers_the_closed_supplementary_keys() -> None:
    row = _history("ACTUAL-DEV-01", "H-S18-DEV-01", 0.5)
    assert set(row) == set(S.SUPPLEMENTARY_HISTORY_KEYS)
    projection = S.supplementary_v2_projection(row)
    assert _validate_history_metric(projection, "projection", "development") == "H-S18-DEV-01"
    with pytest.raises(ValueError):
        _validate_history_metric(row, "supplementary", "development")
    assert row["delta_P"] == pytest.approx(0.1)
    assert row["duration_accounting"]["evaluated_events"] == 1


def test_branch_aggregation_applies_frozen_seed_averaging_and_uncertainty() -> None:
    per_seed = [_branch_rows(seed) for seed in range(3)]
    branch = S.build_branch_metrics(
        thresholds_by_seed=[0.1, 0.2, 0.3],
        per_seed_rows=per_seed,
        baseline_per_seed_rows=None,
    )
    assert set(branch) == {"thresholds_by_seed", "per_seed", "per_history", "aggregate"}
    assert branch["thresholds_by_seed"] == [0.1, 0.2, 0.3]
    assert [entry["model_seed"] for entry in branch["per_seed"]] == [181801, 181802, 181803]
    averaged = branch["per_history"]
    assert [row["history_id"] for row in averaged] == [f"H-S18-DEV-0{i}" for i in range(1, 5)]
    assert averaged[0]["PW_AUROC"] == pytest.approx(0.65)
    for row in averaged:
        # Seed-averaged cells are exactly v2 rows.
        _validate_history_metric(row, "averaged", "development")
    aggregate = branch["aggregate"]
    assert set(aggregate) == {
        "P_macro",
        "W_macro",
        "PW_macro",
        "PW_lcb95",
        "delta_P_macro",
        "delta_W_macro",
        "delta_PW_macro",
        "directional_count",
        "positive_delta_history_count",
        "background_uncertainty",
    }
    assert len(aggregate["background_uncertainty"]) == 2
    assert aggregate["PW_macro"] == pytest.approx(0.65)
    assert aggregate["directional_count"] == 4
    assert aggregate["positive_delta_history_count"] == 4
    with pytest.raises(ValueError):
        S.build_branch_metrics(
            thresholds_by_seed=[0.1, 0.2], per_seed_rows=per_seed, baseline_per_seed_rows=None
        )


def _document(branch: dict[str, object]) -> dict[str, object]:
    return {
        "schema_id": S.SUPPLEMENTARY_SCHEMA_ID,
        "schema_version": 2,
        "protocol_id": "sprint18-full-training-protocol-v2",
        "protocol_sha256": "e1dcd17914e920520ad0d8f50e185874b95414e6a4fe907903619daf145622d2",
        "phase": "development",
        "arm_id": "B0-FULL",
        "model_seed": 181801,
        "execution": {
            "task_id": "S18-T32",
            "attempt_id": "S18T32C05-TEST",
            "agent_id": "S18T32C05",
            "source_commit": "1" * 40,
            "protocol_path": str(S.PROTOCOL_PATH.resolve()),
            "metric_code_path": str(Path(S.__file__).resolve()),
            "supplementary_definition_path": str(S.SUPPLEMENTARY_DEFINITION_PATH.resolve()),
            "metric_code_sha256": S.metric_code_sha256(),
            "supplementary_definition_sha256": hashlib.sha256(
                S.SUPPLEMENTARY_DEFINITION_PATH.read_bytes()
            ).hexdigest(),
            "scope": "experiment",
            "phase": "development",
            "device": "cpu",
            "started_utc": "2026-10-24T00:00:00Z",
            "finished_utc": "2026-10-24T00:00:01Z",
        },
        "inputs": {
            "manifest_path": "m.json",
            "manifest_sha256": "e" * 64,
            "data_root": "data-root",
            "checkpoint_path": "c.pt",
            "checkpoint_sha256": "f" * 64,
            "reference_artifact_path": None,
            "reference_artifact_sha256": None,
            "accepted_record_path": "a.json",
            "accepted_record_sha256": "0" * 64,
            "task14_lock_path": str(_TASK14_V1_LOCK),
            "task14_lock_sha256": S.TASK14_LOCK_V1_SHA256,
            "paired_baseline_task14_lock_path": None,
            "paired_baseline_task14_lock_sha256": None,
            "paired_baseline_accepted_record_path": None,
            "paired_baseline_accepted_record_sha256": None,
            "paired_baseline_arm_config_sha256": None,
            "handoff_path": "h.json",
            "handoff_sha256": "2" * 64,
            "seed_score_documents": [{"path": "s.json", "sha256": "3" * 64}] * 3,
            "baseline_score_documents": [],
            "sha256_before": {
                "checkpoint": "f" * 64,
                "accepted_record": "0" * 64,
                "task14_lock": S.TASK14_LOCK_V1_SHA256,
            },
            "sha256_after": {
                "checkpoint": "f" * 64,
                "accepted_record": "0" * 64,
                "task14_lock": S.TASK14_LOCK_V1_SHA256,
            },
            "unchanged": True,
        },
        "reproduction": {
            "arm_config_sha256_by_seed": ["4" * 64] * 3,
            "paired_baseline_arm_config_sha256_by_seed": [None] * 3,
            "data_binding_sha256": "5" * 64,
            "model_state_sha256_by_seed": ["6" * 64] * 3,
            "ema_state_sha256_by_seed": ["7" * 64] * 3,
            "bank_sha256_by_seed": ["8" * 64] * 3,
            "accepted_bank_sha256_by_seed": ["8" * 64] * 3,
            "bank_reproduced": True,
            "bank_source": "reconstructed_digest_verified",
            "reference_state_modes": {
                "S_pred": "reconstructed_digest_verified",
                "S_pop": "reconstructed_digest_verified",
            },
            "calibration_thresholds_by_seed": {
                name: [0.1, 0.2, 0.3] for name in ("S_pred", "S_pop")
            },
            "accepted_calibration_thresholds_by_seed": {
                name: [0.1, 0.2, 0.3] for name in ("S_pred", "S_pop")
            },
            "recomputed_calibration_thresholds_by_seed": {
                name: [0.1, 0.2, 0.3] for name in ("S_pred", "S_pop")
            },
            "calibration_reproduced": True,
            "representation_state_unchanged": True,
            "reference_state_unchanged": True,
            "encoder_updates": 0,
        },
        "support": {
            "fit_healthy_rows": 36,
            "fit_healthy_rows_sha256": "9" * 64,
            "fit_healthy_scores_sha256": "a" * 64,
            "calibration_healthy_rows": 20,
            "calibration_support_sha256": "b" * 64,
            "calibration_scores_sha256": "c" * 64,
            "development": [
                {
                    "history_id": f"ACTUAL-DEV-0{index}",
                    "schema_history_id": f"H-S18-DEV-0{index}",
                    "manifest_sha256": "d" * 64,
                    "support_sha256": "e" * 64,
                    "scores_sha256": "f" * 64,
                    "rows": 12,
                    "excluded_rows": 0,
                    "p": 1,
                    "w": 0,
                    "a": 0,
                    "controls": 1,
                    "robot_days": 12.0,
                    "evaluated_pw_events": 1,
                    "localization_excluded_events": 0,
                }
                for index in range(1, 5)
            ],
            "excluded_rows": 0,
        },
        "metric_definitions": {
            "localization": S.LOCALIZATION_ID,
            "duration_accounting": S.DURATION_ACCOUNTING_ID,
            "severity_ordering": S.SEVERITY_ID,
            "subtype_slices": S.SUBTYPE_SLICE_ID,
            "background_stability": S.STABILITY_ID,
            "background_uncertainty": S.UNCERTAINTY_ID,
            "event_metrics": S.P.PROBE_ID,
        },
        "branches": {"S_pred": branch, "S_pop": branch},
        "status": "VALID",
        "failure_reasons": [],
    }


def _branch() -> dict[str, object]:
    rows = [_branch_rows(seed) for seed in range(3)]
    # Every cell must be populated for a VALID document; fill the fixture-only
    # absences (no W2/A2 events) with finite placeholders.
    for seed_rows in rows:
        for row in seed_rows.values():
            if row["background_stability_ratio"] is None:
                row["background_stability_ratio"] = 0.05
            for key, value in row["severity_ordering"].items():
                if value is None:
                    row["severity_ordering"][key] = 0.5
            for key, value in row["subtype_slices"].items():
                if value is None:
                    row["subtype_slices"][key] = 0.5
    return S.build_branch_metrics(
        thresholds_by_seed=[0.1, 0.2, 0.3], per_seed_rows=rows, baseline_per_seed_rows=None
    )


def test_closed_supplementary_schema_accepts_valid_and_rejects_drift() -> None:
    document = _document(_branch())
    S.validate_supplementary_result(document)

    def tampered(**changes: object) -> dict[str, object]:
        clone = json.loads(json.dumps(document))
        clone.update(changes)
        return clone

    with pytest.raises(ValueError):
        S.validate_supplementary_result(tampered(phase="confirmation"))
    with pytest.raises(ValueError):
        S.validate_supplementary_result(tampered(schema_id="sprint18-supplementary-development-v1"))
    with pytest.raises(ValueError):
        S.validate_supplementary_result(tampered(arm_id="N1-HCC"))
    with pytest.raises(ValueError):
        S.validate_supplementary_result(tampered(extra_field=1))
    bad_inputs = json.loads(json.dumps(document))
    bad_inputs["inputs"]["sha256_after"] = {"checkpoint": "9" * 64}
    with pytest.raises(ValueError):
        S.validate_supplementary_result(bad_inputs)
    bad_mode = json.loads(json.dumps(document))
    bad_mode["reproduction"]["bank_source"] = "refit_from_development"
    with pytest.raises(ValueError):
        S.validate_supplementary_result(bad_mode)
    bad_support = json.loads(json.dumps(document))
    bad_support["support"]["development"][0]["evaluated_pw_events"] = 0
    with pytest.raises(ValueError):
        S.validate_supplementary_result(bad_support)
    bad_status = json.loads(json.dumps(document))
    bad_status["status"] = "FAILED"
    with pytest.raises(ValueError):
        S.validate_supplementary_result(bad_status)
    empty_duration = json.loads(json.dumps(document))
    empty_duration["branches"]["S_pred"]["per_seed"][0]["per_history"][0][
        "duration_accounting"
    ]["median_declared_duration_days"] = None
    with pytest.raises(ValueError):
        S.validate_supplementary_result(empty_duration)


def _manifest(tmp_path: Path, **changes: object) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    handoff = tmp_path / "handoff.json"
    handoff.write_bytes(b"{}")
    accepted = tmp_path / "accepted.json"
    accepted.write_bytes(b"{}")
    data_root = tmp_path / "data"
    data_root.mkdir()
    document: dict[str, object] = {
        "schema_id": S.SUPPLEMENTARY_INPUT_SCHEMA_ID,
        "protocol_id": "sprint18-full-training-protocol-v2",
        "protocol_sha256": "e1dcd17914e920520ad0d8f50e185874b95414e6a4fe907903619daf145622d2",
        "supplementary_definition_sha256": "1" * 64,
        "metric_code_sha256": S.metric_code_sha256(),
        "source_commit": "2" * 40,
        "task_id": "S18-T32",
        "phase": "development",
        "scope": "experiment",
        "data_root": str(data_root),
        "arm_id": "B0-FULL",
        "model_seed": 181801,
        "task14_lock": {
            "path": str(_TASK14_V1_LOCK),
            "sha256": S.TASK14_LOCK_V1_SHA256,
        },
        "handoff": {"path": str(handoff), "sha256": hashlib.sha256(b"{}").hexdigest()},
        "paired_baseline": None,
        "accepted_record": {"path": str(accepted), "sha256": hashlib.sha256(b"{}").hexdigest()},
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": hashlib.sha256(b"fixture-checkpoint").hexdigest(),
        },
        "reference_artifact": {
            "kind": "none",
            "path": None,
            "sha256": None,
            "fit_scope": None,
            "bounds": None,
        },
        "expected": {
            "arm_config_sha256": "3" * 64,
            "checkpoint_arm_config_sha256": "3" * 64,
            "paired_baseline_arm_config_sha256": None,
            "data_binding_sha256": "4" * 64,
            "model_state_sha256": "5" * 64,
            "ema_state_sha256": "6" * 64,
            "bank_sha256": "7" * 64,
            "bank_source_rows": 12,
            "bank_source_rows_sha256": "b" * 64,
            "bank_recipe_id": S.BANK_RECIPE_ID,
            "bank_order_id": S.BANK_ORDER_ID,
            "calibration_thresholds": {"S_pred": 0.1, "S_pop": 0.2},
            "development_support_sha256": {f"H-S18-DEV-0{i}": "8" * 64 for i in range(1, 5)},
            "development_manifest_sha256": {f"H-S18-DEV-0{i}": "9" * 64 for i in range(1, 5)},
        },
    }
    document.update(changes)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _n4_manifest(tmp_path: Path) -> Path:
    path = _manifest(tmp_path, arm_id="N4-CRN")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["task14_lock"] = {
        "path": str(_TASK14_V9_LOCK),
        "sha256": S.N4_SCIENTIFIC_TASK14_LOCK_V9_SHA256,
    }
    checkpoint = document["checkpoint"]
    checkpoint_path = checkpoint["path"]
    checkpoint_sha = checkpoint["sha256"]
    b0_config_sha = "c" * 64
    n4_config_sha = "d" * 64
    model_sha = document["expected"]["model_state_sha256"]
    ema_sha = document["expected"]["ema_state_sha256"]
    scale_path = tmp_path / "n4-scale.pt"
    scale_path.write_bytes(b"fixture-scale")
    document["reference_artifact"] = {
        "kind": "n4_scale",
        "path": str(scale_path),
        "sha256": hashlib.sha256(scale_path.read_bytes()).hexdigest(),
        "fit_scope": "final",
        "bounds": {"v_floor": 1e-5, "v_cap": 100.0},
    }

    baseline_lock_path = _TASK14_V1_LOCK
    baseline_record_path = tmp_path / "accepted-b0.json"
    baseline_record = {
        "status": "ok",
        "arm_id": "B0-FULL",
        "model_seed": 181801,
        "arm_config_sha256": b0_config_sha,
        "primary": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha,
            "restored_checkpoint_sha256": checkpoint_sha,
            "live_model_state_sha256": model_sha,
            "live_ema_state_sha256": ema_sha,
        },
    }
    baseline_record_bytes = json.dumps(baseline_record).encode("utf-8")
    baseline_record_path.write_bytes(baseline_record_bytes)
    n4_record = {
        "status": "ok",
        "arm_id": "N4-CRN",
        "model_seed": 181801,
        "arm_config_sha256": n4_config_sha,
        "coherent_checkpoint": {
            "paired_b0_checkpoint_sha256": checkpoint_sha,
            "restored_model_state_sha256": model_sha,
            "restored_ema_state_sha256": ema_sha,
        },
        "representation_baseline": {
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha,
            "live_model_state_sha256": model_sha,
            "live_ema_state_sha256": ema_sha,
        },
    }
    accepted_record_path = Path(document["accepted_record"]["path"])
    accepted_record_bytes = json.dumps(n4_record).encode("utf-8")
    accepted_record_path.write_bytes(accepted_record_bytes)
    document["accepted_record"]["sha256"] = hashlib.sha256(
        accepted_record_bytes
    ).hexdigest()
    document["paired_baseline"] = {
        "arm_id": "B0-FULL",
        "accepted_record": {
            "path": str(baseline_record_path),
            "sha256": hashlib.sha256(baseline_record_bytes).hexdigest(),
        },
        "task14_lock": {
            "path": str(baseline_lock_path),
            "sha256": S.TASK14_LOCK_V1_SHA256,
        },
        "arm_config_sha256": b0_config_sha,
    }
    document["expected"]["arm_config_sha256"] = n4_config_sha
    document["expected"]["checkpoint_arm_config_sha256"] = b0_config_sha
    document["expected"]["paired_baseline_arm_config_sha256"] = b0_config_sha
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_n4_manifest_requires_distinct_digest_bound_paired_b0_authority(
    tmp_path: Path,
) -> None:
    path = _n4_manifest(tmp_path)
    inputs = S.load_frozen_input_manifest(path)
    assert inputs.task14_lock_sha256 != inputs.paired_baseline_task14_lock_sha256
    assert inputs.task14_lock_sha256 == S.N4_SCIENTIFIC_TASK14_LOCK_V9_SHA256
    assert inputs.paired_baseline_task14_lock_sha256 == S.TASK14_LOCK_V1_SHA256
    assert inputs.expected["checkpoint_arm_config_sha256"] == "c" * 64
    assert inputs.expected["arm_config_sha256"] == "d" * 64
    assert set(S.compute_input_digests(inputs)) >= {
        "paired_baseline_task14_lock",
        "paired_baseline_accepted_record",
    }

    wrong_science = json.loads(path.read_text(encoding="utf-8"))
    wrong_science["task14_lock"] = {
        "path": str(_TASK14_V1_LOCK),
        "sha256": S.TASK14_LOCK_V1_SHA256,
    }
    wrong_science_path = tmp_path / "wrong-science-lock.json"
    wrong_science_path.write_text(json.dumps(wrong_science), encoding="utf-8")
    with pytest.raises(ValueError, match="accepted N4 scientific v9"):
        S.load_frozen_input_manifest(wrong_science_path)

    wrong_pair = json.loads(path.read_text(encoding="utf-8"))
    wrong_pair["paired_baseline"]["task14_lock"] = {
        "path": str(_TASK14_V9_LOCK),
        "sha256": S.N4_SCIENTIFIC_TASK14_LOCK_V9_SHA256,
    }
    wrong_pair_path = tmp_path / "wrong-paired-lock.json"
    wrong_pair_path.write_text(json.dumps(wrong_pair), encoding="utf-8")
    with pytest.raises(ValueError, match="accepted v1 authority"):
        S.load_frozen_input_manifest(wrong_pair_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["paired_baseline"]["arm_config_sha256"] = "e" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="paired baseline config digest"):
        S.load_frozen_input_manifest(path)


def test_closed_metric_validation_binds_distinct_n4_task14_authorities() -> None:
    document = _document(_branch())
    document["arm_id"] = "N4-CRN"
    document["status"] = "INVALID"
    document["failure_reasons"] = ["fixture-only invalid metric row"]
    inputs = document["inputs"]
    inputs["task14_lock_path"] = str(_TASK14_V9_LOCK)
    inputs["task14_lock_sha256"] = S.N4_SCIENTIFIC_TASK14_LOCK_V9_SHA256
    inputs["paired_baseline_task14_lock_path"] = str(_TASK14_V1_LOCK)
    inputs["paired_baseline_task14_lock_sha256"] = S.TASK14_LOCK_V1_SHA256
    inputs["paired_baseline_accepted_record_path"] = "accepted-b0.json"
    inputs["paired_baseline_accepted_record_sha256"] = "a" * 64
    inputs["paired_baseline_arm_config_sha256"] = "c" * 64
    for name in ("sha256_before", "sha256_after"):
        inputs[name]["task14_lock"] = S.N4_SCIENTIFIC_TASK14_LOCK_V9_SHA256
        inputs[name]["paired_baseline_task14_lock"] = S.TASK14_LOCK_V1_SHA256
        inputs[name]["paired_baseline_accepted_record"] = "a" * 64
    document["reproduction"]["paired_baseline_arm_config_sha256_by_seed"] = ["c" * 64] * 3
    S.validate_supplementary_result(document)

    for field, value in (
        ("task14_lock_sha256", S.TASK14_LOCK_V1_SHA256),
        ("paired_baseline_task14_lock_sha256", S.N4_SCIENTIFIC_TASK14_LOCK_V9_SHA256),
    ):
        tampered = json.loads(json.dumps(document))
        tampered["inputs"][field] = value
        with pytest.raises(ValueError):
            S.validate_supplementary_result(tampered)

    tampered = json.loads(json.dumps(document))
    tampered["inputs"]["paired_baseline_task14_lock_path"] = str(_TASK14_V9_LOCK)
    with pytest.raises(ValueError, match="must be distinct"):
        S.validate_supplementary_result(tampered)

    tampered = json.loads(json.dumps(document))
    del tampered["inputs"]["sha256_before"]["paired_baseline_task14_lock"]
    del tampered["inputs"]["sha256_after"]["paired_baseline_task14_lock"]
    with pytest.raises(ValueError, match="raw custody digest"):
        S.validate_supplementary_result(tampered)


def test_frozen_input_manifest_enforces_the_metric_code_and_identity(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    inputs = S.load_frozen_input_manifest(path)
    assert inputs.arm_id == "B0-FULL" and inputs.scope == "experiment"
    assert inputs.metric_code_sha256 == S.metric_code_sha256()
    assert set(S.compute_input_digests(inputs)) >= {
        "frozen_input_manifest",
        "checkpoint",
        "accepted_record",
        "task14_lock",
        "handoff",
    }
    other = S.load_frozen_input_manifest(path)
    S.assert_inputs_unchanged(S.compute_input_digests(inputs), S.compute_input_digests(other))
    changed = S.compute_input_digests(inputs)
    changed["checkpoint"] = "0" * 64
    with pytest.raises(ValueError):
        S.assert_inputs_unchanged(S.compute_input_digests(inputs), changed)

    Path(inputs.checkpoint_path).write_bytes(b"changed-after-manifest-load")
    with pytest.raises(ValueError, match="immutable input changed: checkpoint"):
        S.compute_input_digests(inputs)

    with pytest.raises(ValueError):
        S.load_frozen_input_manifest(_manifest(tmp_path / "a", metric_code_sha256="0" * 64))
    with pytest.raises(ValueError):
        S.load_frozen_input_manifest(_manifest(tmp_path / "b", arm_id="N1-HCC"))
    with pytest.raises(ValueError):
        S.load_frozen_input_manifest(_manifest(tmp_path / "c", phase="confirmation"))
    with pytest.raises(ValueError):
        S.load_frozen_input_manifest(_manifest(tmp_path / "d", scope="task6_smoke_only"))
    with pytest.raises(ValueError):
        S.load_frozen_input_manifest(_manifest(tmp_path / "e", model_seed=1))

    for field in ("bank_recipe_id", "bank_order_id"):
        bad_path = _manifest(tmp_path / f"bad-{field}")
        bad_document = json.loads(bad_path.read_text(encoding="utf-8"))
        bad_document["expected"][field] = "not-the-accepted-fit-bank-contract"
        bad_path.write_text(json.dumps(bad_document), encoding="utf-8")
        with pytest.raises(ValueError, match="recipe/order"):
            S.load_frozen_input_manifest(bad_path)
