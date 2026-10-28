"""Focused behavioral checks for S24-7-M11 frozen inference execution.

Covers the fail-closed contract gate, identity-bound cache admission and
resume, the frozen movement path on synthetic-but-admitted vectors, and
the real-CLI refusal boundaries. No checkpoint is loaded, no forward
executes, no waveform signal is read in these checks; the synthetic
vectors below prove the execution plumbing only, never corpus evidence.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from representation import sprint24_controlled_inference_execution as m11
from representation import sprint24_controlled_inference_readiness as readiness
from representation import sprint24_weekly_monthly_execution as execution

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "data" / "generated" / "sprint24_controlled_v2"
AUDIT = REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m9-a01" / "corpus-audit.json"
G2_PREFLIGHT = (
    REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m4-a01"
    / "g2-accepted" / "preflight.json"
)
CONTRACT_FILE = (
    REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m10-a02"
    / "prospective-contract.json"
)


def _inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads((CORPUS / "corpus-manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    manifest["__manifest_path__"] = str(CORPUS / "corpus-manifest.json")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert isinstance(audit, dict)
    preflight = json.loads(G2_PREFLIGHT.read_text(encoding="utf-8"))
    assert isinstance(preflight, dict)
    return manifest, audit, preflight


def _contract() -> tuple[dict[str, Any], str]:
    envelope = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    assert isinstance(envelope, dict)
    return dict(envelope["contract"]), str(envelope["contract_sha256"])


def test_m11_contract_digest_pinned() -> None:
    assert m11.EXPECTED_CONTRACT_SHA256 == "f4f92a60ee1782ccc0ec75587e9df5d1471e4045e201c43478893d294b5757f7"
    assert m11.TASK_ID == "S24-7-M11"
    contract, digest = _contract()
    assert digest == m11.EXPECTED_CONTRACT_SHA256


def test_m11_contract_binding_verified_pre_execution() -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, digest = _contract()
    bound = m11.verify_contract_binding(contract, digest, manifest, audit, custody)
    assert set(bound) == {"census", "plan"}
    assert int(bound["plan"]["n_groups"]) == 18  # type: ignore[index]


def test_m11_altered_contract_refused_pre_execution() -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, digest = _contract()
    tampered = dict(contract)
    tampered["binding_digest"] = "0" * 64
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        m11.verify_contract_binding(tampered, digest, manifest, audit, custody)


def test_m11_stale_digest_refused_pre_execution() -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, _digest = _contract()
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        m11.verify_contract_binding(contract, "0" * 64, manifest, audit, custody)


def test_m11_capture_plan_matches_frozen_groups() -> None:
    manifest, _audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, digest = _contract()
    bound = m11.verify_contract_binding(
        contract, digest, manifest, _audit, custody
    )
    capture = m11.bound_capture_plan(manifest, custody)
    live_groups = capture["plan"]["groups"]
    assert isinstance(live_groups, dict)
    frozen_groups = contract["executable_path"]["identity_plan"]["groups"]
    assert dict(live_groups) == dict(frozen_groups)
    assert len(capture["file_ids"]) == 6
    assert all(len(ids) == 1380 for ids in capture["file_ids"].values())


def test_m11_reference_pool_is_causal_and_sized() -> None:
    manifest, _audit, _preflight = _inputs()
    for history_id in sorted(m11.bound_capture_plan(manifest, readiness.validate_custody_inputs(_preflight))["file_ids"]):
        ref_ids = m11.reference_ids_for_history(manifest, history_id)
        assert len(ref_ids) == 220
    manifest_files = {str(e["file_id"]): e for e in manifest["files"] if isinstance(e, dict)}
    sample = m11.reference_ids_for_history(manifest, sorted(manifest_files)[0] and "S24C-ITER-0001-CONTROLLED-01")
    ends = [float(manifest_files[fid]["end_time_rel_s"]) for fid in sample]
    assert max(ends) <= 0.0


def _write_synthetic_group(
    root: Path,
    inputs: dict[str, Any],
    plan_groups: dict[str, Any],
    history_id: str,
    model_seed: int,
    *,
    provenance: str,
    commit: str | None,
    rng: Any,
) -> None:
    key = f"{history_id}|{model_seed}"
    spec = plan_groups[key]
    assert isinstance(spec, dict)
    support = str(spec["support_sha256"])
    n = int(spec["required_rows"])
    vectors = rng.normal(0.0, 1.0, size=(n, 128)).astype(np.float32)
    file_ids = [f"SYN-{history_id}-{model_seed}-{i:05d}" for i in range(n)]
    # Writer binds its own support digest; the caller re-derives identity via
    # the inspected sidecar, never by trusting this call's arguments.
    m11.write_group_cache(
        root,
        history_id=history_id,
        model_seed=int(model_seed),
        file_ids=file_ids,
        embeddings=vectors,
        inputs=inputs,
        support_sha256=support,
        provenance=provenance,
        capture_commit=commit,
    )


def test_m11_admission_refuses_foreign_provenance() -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, digest = _contract()
    bound = m11.verify_contract_binding(contract, digest, manifest, audit, custody)
    plan = bound["plan"]
    assert isinstance(plan, dict)
    inputs = m11.capture_inputs(custody)
    rng = np.random.default_rng(2407)
    with tempfile.TemporaryDirectory(prefix="s24-m11-neg-") as tmp:
        root = Path(tmp)
        first_history = sorted(m11.bound_capture_plan(manifest, custody)["file_ids"])[0]
        for seed in readiness.FROZEN_ENCODER_SEEDS:
            _write_synthetic_group(
                root, inputs, plan["groups"], first_history, int(seed),
                provenance=execution.CACHE_PROVENANCE_FIXTURE,
                commit=None, rng=rng,
            )
        m11.set_support_universe(m11.bound_capture_plan(manifest, custody)["file_ids"])
        with pytest.raises(execution.CachePreflightFailure):
            m11.admit_cache_or_refuse(root, plan)


def test_m11_resume_skips_complete_group_without_reforward() -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, digest = _contract()
    bound = m11.verify_contract_binding(contract, digest, manifest, audit, custody)
    plan = bound["plan"]
    assert isinstance(plan, dict)
    wanted = m11.bound_capture_plan(manifest, custody)
    m11.set_support_universe(wanted["file_ids"])
    first_history = sorted(wanted["file_ids"])[0]
    key = f"{first_history}|{readiness.FROZEN_ENCODER_SEEDS[0]}"
    expected = str(plan["groups"][key]["cache_identity"])  # type: ignore[index]
    with tempfile.TemporaryDirectory(prefix="s24-m11-resume-") as tmp:
        root = Path(tmp)
        sidecar = {
            "schema": execution.CACHE_SIDECAR_SCHEMA,
            "kind": execution.CACHE_SIDECAR_KIND,
            "history_id": first_history,
            "model_seed": int(readiness.FROZEN_ENCODER_SEEDS[0]),
            "device": "cuda",
            "checkpoint_sha256": str(custody["checkpoints"][str(readiness.FROZEN_ENCODER_SEEDS[0])]),  # type: ignore[index]
            "bank_sha256": str(custody["banks"][str(readiness.FROZEN_ENCODER_SEEDS[0])]),  # type: ignore[index]
            "support_sha256": str(plan["groups"][key]["support_sha256"]),  # type: ignore[index]
            "preprocessing_sha256": str(custody["preprocessing_sha256"]),
            "arm_config_sha256": str(custody["arms"][str(readiness.FROZEN_ENCODER_SEEDS[0])]),  # type: ignore[index]
            "data_binding_sha256": str(custody["binding_sha256"]),
            "scale_state_sha256": "none",
            "cache_identity": expected,
            "n_rows": 4,
            "capture_provenance": execution.CACHE_PROVENANCE_SMOKE,
            "capture_commit": None,
        }
        stem = f"{first_history}__seed-{readiness.FROZEN_ENCODER_SEEDS[0]}"
        (root / f"{stem}.json").write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
        vectors = np.zeros((4, 128), dtype=np.float32)
        np.savez(root / f"{stem}.npz", file_ids=np.array(["a", "b", "c", "d"]), embeddings=vectors)
        hit = m11.cached_group_complete(root, first_history, int(readiness.FROZEN_ENCODER_SEEDS[0]), expected)
        assert hit is not None and int(hit["n_rows"]) == 4
        assert m11.cached_group_complete(root, first_history, int(readiness.FROZEN_ENCODER_SEEDS[0]), "0" * 64) is None


def test_m11_cli_smoke_refuses_without_bundle_roots() -> None:
    with tempfile.TemporaryDirectory(prefix="s24-m11-cli-") as tmp:
        out = Path(tmp) / "out"
        cache = Path(tmp) / "cache"
        cache.mkdir()
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "experiments" / "sprint24_task7m11_controlled_inference_execution.py"),
                "--corpus", str(CORPUS),
                "--audit", str(AUDIT),
                "--contract", str(CONTRACT_FILE),
                "--cache-root", str(cache),
                "--out", str(out),
                "--mode", "smoke",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 2, proc.stderr[-2000:]
        run = json.loads((out / "run.json").read_text(encoding="utf-8"))
        assert run["status"] == "REFUSED-PREREQUISITE"
        assert run["forwards"] == 0


_OVERLAP_HISTORY = "S24C-ITER-0001-CONTROLLED-01"
#: (file_id, manifest end_time_rel_s, NPZ header duration in seconds)
_OVERLAP_BASELINE = (
    ("FIX-BASE-0", 1000.0, 100),
    ("FIX-BASE-1", 2000.0, 200),
    ("FIX-BASE-2", 3000.0, 300),
)
_OVERLAP_CONTRAST = (
    ("FIX-CONT-0", 10000.0, 200),
    ("FIX-CONT-1", 11000.0, 100),
    ("FIX-CONT-2", 12000.0, 300),
)
#: Expected frozen overlap weights, derived by hand from the fixture intervals
#: clipped to each arm span [min end, max end]:
#: baseline overlaps [0, 200, 300] / 500 -> [0, 0.4, 0.6];
#: contrast overlaps [0, 100, 300] / 400 -> [0, 0.25, 0.75].
_OVERLAP_WX = np.asarray([0.0, 0.4, 0.6])
_OVERLAP_WY = np.asarray([0.0, 0.25, 0.75])
#: Exact closed form: 1.145 + 0.855*exp(-1/2) - 0.3*exp(-2)
#: - 1.1*exp(-9/2) - 0.6*exp(-8) (arm positions 0/1/2 vs 3/4/5,
#: fitted bandwidth sigma = 1), evaluated to double precision.
_OVERLAP_EXPECTED_WEIGHTED = 1.6105619553145096


def _overlap_fixture(
    root: Path, *, reverse: bool = False
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a tiny real-metadata corpus: manifest end times + NPZ header durations.

    The 200 reference files satisfy the causal reference-pool floor; their
    embeddings are two points at 0 and 1, so the fitted bandwidth is exactly
    1.0 (the median pairwise distance). Pair members carry unequal NPZ header
    durations, and the frozen geometry reads only the manifest end time and
    the ``x.npy`` shape header. ``reverse`` permutes the retained member order
    while every file keeps its own embedding row and interval metadata.
    """
    from representation import sprint24_weekly_monthly_endpoints as endpoints

    reference = [f"FIX-REF-{index:03d}" for index in range(200)]
    files: list[dict[str, Any]] = []
    (root / "ref").mkdir(parents=True, exist_ok=True)
    (root / "arm").mkdir(parents=True, exist_ok=True)
    for index, fid in enumerate(reference):
        rel = f"ref/{fid}.npz"
        np.savez(root / rel, x=np.zeros((6, 1), dtype=np.float32))
        files.append(
            {
                "file_id": fid,
                "history_id": _OVERLAP_HISTORY,
                "arm": "REFERENCE",
                "end_time_rel_s": -86400.0 * (index + 1),
                "relpath": rel,
            }
        )
    for members, arm in ((_OVERLAP_BASELINE, "BASELINE"), (_OVERLAP_CONTRAST, "ELEVATED")):
        for fid, end, duration in members:
            rel = f"arm/{fid}.npz"
            np.savez(root / rel, x=np.zeros((6, duration), dtype=np.float32))
            files.append(
                {
                    "file_id": fid,
                    "history_id": _OVERLAP_HISTORY,
                    "arm": arm,
                    "end_time_rel_s": end,
                    "relpath": rel,
                }
            )
    manifest_path = root / "corpus-manifest.json"
    manifest_path.write_text(json.dumps({"files": files}), encoding="utf-8")
    manifest = {"files": files, "__manifest_path__": str(manifest_path)}
    baseline_ids = [fid for fid, _, _ in _OVERLAP_BASELINE]
    contrast_ids = [fid for fid, _, _ in _OVERLAP_CONTRAST]
    if reverse:
        baseline_ids = list(reversed(baseline_ids))
        contrast_ids = list(reversed(contrast_ids))
    audit = {
        "pair_rows": [
            {
                "history_id": _OVERLAP_HISTORY,
                "horizon": "week",
                "mode": "calendar",
                "comparison_identity": f"{_OVERLAP_HISTORY}|week|calendar|P00",
                "comparison_type": "BASELINE-VS-ELEVATED",
                "disposition": "FEASIBLE",
                "burden_baseline_after": 0.0,
                "burden_contrast_after": 0.5,
                "burden_contrast_gap_after": 0.5,
                "exposure_ratio_after": 0.9,
                "retained_baseline_file_identities": baseline_ids,
                "retained_contrast_file_identities": contrast_ids,
            }
        ]
    }
    file_ids = (
        list(reference)
        + [fid for fid, _, _ in _OVERLAP_BASELINE]
        + [fid for fid, _, _ in _OVERLAP_CONTRAST]
    )
    coords = [0.0] * 100 + [1.0] * 100 + [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    # Two-dimensional rows so every endpoint (including the covariance-trace
    # E4) is computable; only the first coordinate varies, so the kernel and
    # the fitted bandwidth sigma are unchanged by the extra zero column.
    matrix = np.zeros((len(coords), 2), dtype=np.float64)
    matrix[:, 0] = np.asarray(coords, dtype=np.float64)
    groups = {
        f"{_OVERLAP_HISTORY}|{int(seed)}": endpoints.EmbeddingGroup(
            history_id=_OVERLAP_HISTORY,
            model_seed=int(seed),
            file_ids=list(file_ids),
            index={fid: position for position, fid in enumerate(file_ids)},
            matrix=matrix.copy(),
        )
        for seed in readiness.FROZEN_ENCODER_SEEDS
    }
    return manifest, audit, groups


def _overlap_kernel(first: Any, second: Any, sigma: float = 1.0) -> Any:
    distances = (np.asarray(first)[:, None] - np.asarray(second)[None, :]) ** 2
    return np.exp(-distances / (2.0 * sigma * sigma))


def _expected_overlap_weighted() -> float:
    """Independent kernel formula over the hand-derived overlap weights."""
    baseline = np.asarray([0.0, 1.0, 2.0])
    contrast = np.asarray([3.0, 4.0, 5.0])
    kxx = _overlap_kernel(baseline, baseline)
    kyy = _overlap_kernel(contrast, contrast)
    kxy = _overlap_kernel(baseline, contrast)
    return float(
        _OVERLAP_WX @ kxx @ _OVERLAP_WX
        + _OVERLAP_WY @ kyy @ _OVERLAP_WY
        - 2.0 * (_OVERLAP_WX @ kxy @ _OVERLAP_WY)
    )


def test_m11_weighted_geometry_matches_exact_expected_overlap() -> None:
    """Public M11 geometry on real metadata must reproduce the exact frozen value."""
    expected = _expected_overlap_weighted()
    assert expected == pytest.approx(_OVERLAP_EXPECTED_WEIGHTED, rel=0.0, abs=1e-12)
    uniform_weights = np.full(3, 1.0 / 3.0)
    uniform_value = float(
        uniform_weights @ _overlap_kernel(np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 1.0, 2.0])) @ uniform_weights
        + uniform_weights @ _overlap_kernel(np.asarray([3.0, 4.0, 5.0]), np.asarray([3.0, 4.0, 5.0])) @ uniform_weights
        - 2.0
        * (
            uniform_weights
            @ _overlap_kernel(np.asarray([0.0, 1.0, 2.0]), np.asarray([3.0, 4.0, 5.0]))
            @ uniform_weights
        )
    )
    # The fixture must discriminate the frozen overlap weighting from the
    # equal-file plug-in, otherwise an exact assertion would be vacuous.
    assert abs(expected - uniform_value) > 0.1
    with tempfile.TemporaryDirectory(prefix="s24-m11-overlap-") as tmp:
        manifest, audit, groups = _overlap_fixture(Path(tmp))
        movement = m11.run_movement_on_groups(groups, manifest, audit, {})
    records = movement["records"]
    assert isinstance(records, list) and len(records) == 1
    cells = records[0]["endpoints"]
    assert isinstance(cells, dict)
    for seed in readiness.FROZEN_ENCODER_SEEDS:
        cell = cells[str(seed)]
        assert isinstance(cell, dict)
        assert cell["status"] == "COMPUTED"
        assert float(cell["E1_weighted"]) == pytest.approx(expected, rel=0.0, abs=1e-12)
    # One retained pair cannot support the rank association, so the uncertainty
    # summary must report the history as excluded rather than counted.
    summaries = movement["summaries"]
    assert isinstance(summaries, dict)
    uncertainty = summaries["uncertainty"]
    assert isinstance(uncertainty, dict)
    week = uncertainty["week"]
    assert isinstance(week, dict)
    assert week["n_histories"] == 0
    assert week["n_histories_with_records"] == 1
    assert week["n_nonestimable_histories"] == 1


def test_m11_weighted_geometry_permutation_invariant() -> None:
    """Simultaneously permuted retained IDs and interval weights preserve the statistic."""
    with tempfile.TemporaryDirectory(prefix="s24-m11-overlap-run-") as tmp:
        manifest, audit, groups = _overlap_fixture(Path(tmp))
        forward = m11.run_movement_on_groups(groups, manifest, audit, {})
    with tempfile.TemporaryDirectory(prefix="s24-m11-overlap-rev-") as tmp:
        manifest_r, audit_r, groups_r = _overlap_fixture(Path(tmp), reverse=True)
        reversed_run = m11.run_movement_on_groups(groups_r, manifest_r, audit_r, {})
    straight_audit = audit["pair_rows"][0]
    permuted_audit = audit_r["pair_rows"][0]
    assert (
        permuted_audit["retained_baseline_file_identities"]
        != straight_audit["retained_baseline_file_identities"]
    )
    assert (
        permuted_audit["retained_contrast_file_identities"]
        != straight_audit["retained_contrast_file_identities"]
    )
    expected = _expected_overlap_weighted()
    forward_records = forward["records"]
    reversed_records = reversed_run["records"]
    assert isinstance(forward_records, list) and isinstance(reversed_records, list)
    straight_cells = forward_records[0]["endpoints"]
    permuted_cells = reversed_records[0]["endpoints"]
    assert isinstance(straight_cells, dict) and isinstance(permuted_cells, dict)
    for seed in readiness.FROZEN_ENCODER_SEEDS:
        straight = straight_cells[str(seed)]
        permuted = permuted_cells[str(seed)]
        assert isinstance(straight, dict) and isinstance(permuted, dict)
        assert straight["status"] == permuted["status"] == "COMPUTED"
        assert float(straight["E1_weighted"]) == pytest.approx(
            float(permuted["E1_weighted"]), rel=0.0, abs=1e-12
        )
        assert float(permuted["E1_weighted"]) == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_m11_splits_separate_horizons_and_label_n() -> None:
    """Split summaries must be per-horizon (no week+month pooling) with explicit history counts."""
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, _digest = _contract()
    bound = m11.verify_contract_binding(contract, _digest, manifest, audit, custody)
    assert isinstance(bound["plan"], dict)
    capture = m11.bound_capture_plan(manifest, custody)
    assert isinstance(capture.get("file_ids"), dict)
    m11.set_support_universe(capture["file_ids"])  # type: ignore[arg-type]
    from representation import sprint24_weekly_monthly_endpoints as endpoints

    rng = np.random.default_rng(2407)
    groups: dict[str, Any] = {}
    for key in sorted(bound["plan"]["groups"]):  # type: ignore[index]
        history, seed = str(key).split("|")
        file_ids = sorted(capture["file_ids"][history])  # type: ignore[index]
        matrix = rng.normal(0.0, 1.0, size=(len(file_ids), 128)).astype(np.float64)
        groups[key] = endpoints.EmbeddingGroup(
            history_id=history, model_seed=int(seed), file_ids=file_ids,
            index={fid: pos for pos, fid in enumerate(file_ids)}, matrix=matrix,
        )
    movement = m11.run_movement_on_groups(groups, manifest, audit, contract)
    assert isinstance(movement["summaries"], dict)
    splits = movement["summaries"]["splits"]  # type: ignore[index,attr-defined]
    assert isinstance(splits, dict)
    assert sorted(splits) == ["primary|month", "primary|week", "replication|month", "replication|week"]
    assert "primary" not in splits and "replication" not in splits
    for key in sorted(splits):
        assert splits[key]["n_cells"] == 3  # type: ignore[index]
        assert splits[key]["horizon"] == key.split("|")[1]  # type: ignore[index]
    uncertainty = movement["summaries"]["uncertainty"]  # type: ignore[index,attr-defined]
    assert uncertainty["week"]["n_histories"] == 6  # type: ignore[index]
    assert uncertainty["month"]["n_histories"] == 6  # type: ignore[index]
    assert uncertainty["week"]["n_histories_with_records"] == 6  # type: ignore[index]
    assert uncertainty["week"]["n_nonestimable_histories"] == 0  # type: ignore[index]
    assert "n = 6 estimable histories of 6 with records" in str(uncertainty["week"]["note"])  # type: ignore[index]
    assert str(movement["summaries"]["c2"]) == "ABSTAIN-CORPUS-WIDE"  # type: ignore[index]
    assert "NONADJACENT-SHARED-SUPPORT" in str(movement["summaries"]["dependence"])  # type: ignore[index]
    assert "N1" in str(movement["summaries"]["n1_reference_null_identity"])  # type: ignore[index]
    transition = movement["summaries"]["transition_vs_control"]  # type: ignore[index,attr-defined]
    assert transition["week"] == {  # type: ignore[index]
        "n_pairs": 30, "n_transitions": 24, "n_controls": 6,
        "transition_dispositions": ["FEASIBLE"], "control_dispositions": ["LIMITED"],
    }
    holm = movement["summaries"]["holm"]  # type: ignore[index,attr-defined]
    assert holm["status"] == "NOT-COMPUTED-NO-PVALUES" and holm["n_pvalues"] == 0  # type: ignore[index]
    endpoints_by_record = movement["records"]  # type: ignore[attr-defined]
    assert isinstance(endpoints_by_record, list)
    weighted = 0
    for record in endpoints_by_record:
        assert isinstance(record, dict)
        cells = record.get("endpoints")
        assert isinstance(cells, dict)
        for seed, cell in cells.items():
            assert isinstance(cell, dict)
            if cell.get("E1_weighted") != cell.get("E1_biased"):
                weighted += 1
    assert weighted == 180


def test_m11_uncertainty_reports_only_estimable_bootstrap_support() -> None:
    """The annotated bootstrap n must equal the estimable C1 inputs, not record presence."""
    seeds = [int(seed) for seed in readiness.FROZEN_ENCODER_SEEDS]
    estimable = [f"S24C-ITER-{index:04d}-CONTROLLED-{index:02d}" for index in range(1, 6)]
    records: list[dict[str, Any]] = []
    for order, history in enumerate(estimable):
        for burden in (0.1, 0.2, 0.3):
            records.append(
                {
                    "history_id": history,
                    "horizon": "week",
                    "comparison_type": "BASELINE-VS-ELEVATED",
                    "disposition": "FEASIBLE",
                    "burden_contrast_gap_after": burden,
                    "endpoints": {
                        str(seed): {"status": "COMPUTED", "E1": burden * (order + 1)}
                        for seed in seeds
                    },
                }
            )
    incomplete = "S24C-ITER-0006-CONTROLLED-06"
    for burden in (0.1, 0.2, 0.3):
        records.append(
            {
                "history_id": incomplete,
                "horizon": "week",
                "comparison_type": "BASELINE-VS-ELEVATED",
                "disposition": "FEASIBLE",
                "burden_contrast_gap_after": burden,
                "endpoints": {str(seed): {"status": "UNSUPPORTED", "E1": None} for seed in seeds},
            }
        )
    summaries = m11.summarize_movement(records, {})
    uncertainty = summaries["uncertainty"]
    assert isinstance(uncertainty, dict)
    week = uncertainty["week"]
    assert isinstance(week, dict)
    assert week["n_histories"] == len(estimable)
    assert week["histories_estimable"] == estimable
    assert week["n_histories_with_records"] == len(estimable) + 1
    assert week["n_nonestimable_histories"] == 1
    assert week["nonestimable_histories"] == [incomplete]
    assert "n = 5 estimable histories of 6 with records" in str(week["note"])
    assert "1 non-estimable excluded from the bootstrap" in str(week["note"])
    assert float(week["point_median"]) == pytest.approx(1.0)
