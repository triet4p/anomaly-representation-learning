"""Sprint 24 Task 7-M16 paired-twin constructor smoke CLI.

One tiny public-path smoke through the real paired-twin constructor on
throwaway synthetic inputs: shared-source pairing, deterministic replay,
severity/family sweeps, duration-family within-family pairing, retry-selected
source, normal-return/missing-event and error boundaries. Temporary outputs
only; no official corpus, inference, forwards, training, refit, remote,
scale or Phase C.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from synth.config import SynthConfig  # noqa: E402
from synth.generator import SessionGenerator, derive_pair_seed  # noqa: E402
from synth.schema import AnomalyFamily, SampleLabel  # noqa: E402


def _waveform_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def run_smoke() -> dict[str, Any]:
    """Run the tiny public-path paired-twin smoke (temp outputs only)."""
    generator = SessionGenerator(SynthConfig())
    seed = 42401
    pair = generator.generate_paired_twin(seed, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    replay = generator.generate_paired_twin(seed, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    legacy = generator.generate_anomaly(
        seed=seed, split="test", family=AnomalyFamily.SUBTLE_DRIFT, severity=0.6
    )
    probe_normal = generator.generate_normal(seed=seed, split="test")
    severity_sweep = generator.generate_paired_severity_sweep(
        seed, AnomalyFamily.SUBTLE_DRIFT, [0.35, 0.60, 0.85]
    )
    family_sweep = generator.generate_paired_family_sweep(
        seed, [AnomalyFamily.SUBTLE_DRIFT, AnomalyFamily.REALISTIC_STUCK], 0.6
    )
    duration = generator.generate_paired_twin(seed, AnomalyFamily.DURATION_ANOMALY, 0.6)
    retry = generator.generate_paired_twin(42447, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    retry_legacy = generator.generate_anomaly(
        seed=42447, split="test", family=AnomalyFamily.SUBTLE_DRIFT, severity=0.6
    )
    normal_return = generator.generate_paired_twin(42471, AnomalyFamily.MISSING_EVENT, 0.6)
    errors: dict[str, str] = {}
    for label, call in (
        ("negative_seed", lambda: generator.generate_paired_twin(-1, AnomalyFamily.SUBTLE_DRIFT, 0.6)),
        ("unknown_family", lambda: generator.generate_paired_twin(7, "nope", 0.6)),
        ("bad_severity", lambda: generator.generate_paired_twin(7, AnomalyFamily.SUBTLE_DRIFT, 9.0)),
    ):
        try:
            call()
            errors[label] = "NO-ERROR"
        except ValueError as exc:
            errors[label] = f"{type(exc).__name__}: {exc}"[:120]

    severity_shared = (
        {item.T for item in (pair.twin, *[item.twin for item in severity_sweep])} == {pair.twin.T}
        and {item.source_id for item in severity_sweep} == {severity_sweep[0].source_id}
        and bool((severity_sweep[0].twin.x == severity_sweep[1].twin.x).all())
    )
    family_shared = {item.source_id for item in family_sweep} == {family_sweep[0].source_id} and bool(
        (family_sweep[0].twin.x == family_sweep[1].twin.x).all()
    )
    mask = np.asarray(pair.abnormal.anomaly_mask)
    outside = ~mask
    signal_dependent_noise = bool(
        (np.asarray(pair.twin.x)[outside] != np.asarray(pair.abnormal.x)[outside]).any()
    )
    with tempfile.TemporaryDirectory(prefix="s24p-m16-smoke-") as tmpdir:
        twin_path = Path(tmpdir) / "twin.npz"
        abnormal_path = Path(tmpdir) / "abnormal.npz"
        np.savez_compressed(twin_path, x=np.asarray(pair.twin.x))
        np.savez_compressed(abnormal_path, x=np.asarray(pair.abnormal.x))
        temp_report = {
            "dir_prefix": "s24p-m16-smoke-",
            "wrote": ["twin.npz", "abnormal.npz"],
            "twin_bytes": int(twin_path.stat().st_size),
            "abnormal_bytes": int(abnormal_path.stat().st_size),
            "note": "TemporaryDirectory cleaned on exit; no official corpus written.",
        }
    checks = {
        "twin_is_normal_label": pair.twin.file_label == SampleLabel.NORMAL,
        "abnormal_is_abnormal_label": pair.abnormal.file_label == SampleLabel.ABNORMAL,
        "shared_roster": (pair.twin.robot_idx, pair.twin.program_idx)
        == (pair.abnormal.robot_idx, pair.abnormal.program_idx),
        "shared_regimes": [(item.start, item.end) for item in pair.twin.regime_sequence]
        == [(item.start, item.end) for item in pair.abnormal.regime_sequence],
        "group_identity": pair.group_id == f"s24p-paired-twin-v1:{seed:010d}"
        and pair.abnormal.anomaly_meta.extra["group_id"] == pair.group_id
        and pair.source_id == pair.abnormal.anomaly_meta.extra["source_id"],
        "legacy_waveform_equal": bool((np.asarray(pair.abnormal.x) == np.asarray(legacy.x)).all()),
        "legacy_mask_equal": bool(
            (np.asarray(pair.abnormal.anomaly_mask) == np.asarray(legacy.anomaly_mask)).all()
        ),
        "replay_identical": bool((np.asarray(pair.twin.x) == np.asarray(replay.twin.x)).all())
        and bool((np.asarray(pair.abnormal.x) == np.asarray(replay.abnormal.x)).all()),
        "severity_sweep_shared": bool(severity_shared),
        "family_sweep_shared": bool(family_shared),
        "duration_within_family_only": duration.abnormal.T != duration.twin.T
        and "within-family only" in duration.abnormal.anomaly_meta.extra["duration_pairing_note"],
        "retry_selected_source": pair_retry_ok(retry, retry_legacy),
        "normal_return_no_fake_success": normal_return.disposition == "NORMAL_RETURN"
        and normal_return.abnormal.file_label == SampleLabel.NORMAL
        and normal_return.abnormal.anomaly_meta is None,
        "normal_path_diverges": pair.normal_path_diverges is True and probe_normal.T != pair.twin.T,
        "signal_dependent_noise_disclosed": bool(signal_dependent_noise),
        "error_boundaries": all(value.startswith("ValueError") for value in errors.values()),
    }
    passed = all(checks.values())
    return {
        "task": "S24-7-M16",
        "status": "SMOKE-COMPLETE" if passed else "SMOKE-FAILED",
        "forwards_executed_this_run": 0,
        "model_loads": 0,
        "official_corpus_written": False,
        "pair_seed": seed,
        "pair_seed_derivation": derive_pair_seed(53101, 0),
        "group_id": pair.group_id,
        "source_id": pair.source_id,
        "disposition": pair.disposition,
        "n_attempts": pair.n_attempts,
        "strength": float(pair.strength),
        "checks": checks,
        "waveforms": {
            "twin_sha256": _waveform_sha256(np.asarray(pair.twin.x)),
            "abnormal_sha256": _waveform_sha256(np.asarray(pair.abnormal.x)),
            "twin_T": int(pair.twin.T),
            "abnormal_T": int(pair.abnormal.T),
        },
        "provenance": {
            "twin_file_id": pair.twin.file_id,
            "abnormal_file_id": pair.abnormal.file_id,
            "roster": [int(pair.twin.robot_idx), int(pair.twin.program_idx)],
        },
        "errors": errors,
        "temp_outputs": temp_report,
    }


def pair_retry_ok(pair: object, legacy: object) -> bool:
    """Compare the retry-selected twin output against the legacy retry output."""
    twin_pair = pair  # SessionGenerator.PairedTwin at runtime
    legacy_sample = legacy  # FileSample at runtime
    return bool(
        twin_pair.n_attempts == legacy_sample.anomaly_meta.extra["n_attempts"]  # type: ignore[union-attr]
        and (twin_pair.abnormal.x == legacy_sample.x).all()  # type: ignore[union-attr]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S24-7-M16 paired-twin smoke CLI")
    parser.add_argument("--mode", choices=["smoke"], default="smoke")
    args = parser.parse_args(argv)
    assert args.mode == "smoke"
    report = run_smoke()
    print(f"status={report['status']} group={report['group_id']} source={report['source_id']}")
    for name, value in report["checks"].items():
        print(f"check {name}={value}")
    if report["status"] != "SMOKE-COMPLETE":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
