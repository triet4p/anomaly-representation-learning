"""Focused Task 5 behavioral tests: boundaries only, no wiring asserts."""

from __future__ import annotations

import pytest
import torch

from representation.sprint24_features import (
    S24_FEATURES_SCHEMA_ID,
    assert_s24_frozen_unchanged,
    capture_s24_file_features,
    restore_s24_provenanced_bank,
    s24_feature_cache_identity,
    s24_preprocessing_digest,
    score_s24_development_support,
    score_s24_n4_normalized,
)


class _Cfg:
    n_channels = 6
    patch_size = 32
    stride = 16
    knn_k = 5

class _Model:
    config = _Cfg()


def test_preprocessing_digest_rejects_bad_k() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        s24_preprocessing_digest(model=_Model(), bank_k=0)  # type: ignore[arg-type]


def test_preprocessing_digest_changes_with_geometry() -> None:
    base = s24_preprocessing_digest(model=_Model(), bank_k=5)  # type: ignore[arg-type]
    other = s24_preprocessing_digest(model=_Model(), bank_k=6)  # type: ignore[arg-type]
    assert base != other
    assert len(base) == 64


def test_cache_identity_separates_b0_and_n4() -> None:
    kwargs: dict[str, object] = {
        "checkpoint_sha256": "a" * 64,
        "bank_sha256": "b" * 64,
        "support_sha256": "c" * 64,
        "preprocessing_sha256": "d" * 64,
        "arm_config_sha256": "e" * 64,
        "data_binding_sha256": "f" * 64,
        "model_seed": 181801,
    }
    b0 = s24_feature_cache_identity(**kwargs)  # type: ignore[arg-type]
    n4 = s24_feature_cache_identity(scale_state_sha256="1" * 64, **kwargs)  # type: ignore[arg-type]
    assert b0 != n4
    assert S24_FEATURES_SCHEMA_ID == "s24-frozen-features-v1"


def test_cache_identity_rejects_wrong_seed_namespace() -> None:
    with pytest.raises(ValueError, match="seed namespace"):
        s24_feature_cache_identity(
            checkpoint_sha256="a" * 64,
            bank_sha256="b" * 64,
            support_sha256="c" * 64,
            preprocessing_sha256="d" * 64,
            arm_config_sha256="e" * 64,
            data_binding_sha256="f" * 64,
            model_seed=999999,
        )


def test_frozen_guard_reports_moved_keys() -> None:
    with pytest.raises(ValueError, match="frozen state moved"):
        assert_s24_frozen_unchanged({"bank_sha256": "a"}, {"bank_sha256": "b"})


def test_empty_rows_fail_fast_in_development_scoring() -> None:
    with pytest.raises(ValueError, match="non-empty row batch"):
        score_s24_development_support(
            binding=None,  # type: ignore[arg-type]
            arm=None,  # type: ignore[arg-type]
            bundle=None,  # type: ignore[arg-type]
            rows=[],
        )


def test_empty_samples_fail_fast_in_feature_capture() -> None:
    with pytest.raises(ValueError, match="non-empty file batch"):
        capture_s24_file_features(
            model=None,  # type: ignore[arg-type]
            bank=None,  # type: ignore[arg-type]
            samples=[],
            history_id="S18I-ITER-0005-DEVELOPMENT-01",
            model_seed=181801,
        )


def test_empty_samples_fail_fast_in_n4_scoring() -> None:
    with pytest.raises(ValueError, match="non-empty file batch"):
        score_s24_n4_normalized(
            model=None,  # type: ignore[arg-type]
            scale_model=None,  # type: ignore[arg-type]
            samples=[],
            history_id="S18I-ITER-0005-DEVELOPMENT-01",
            model_seed=181801,
            v_floor=1.0,
            v_cap=2.0,
        )


def test_missing_bank_file_fails_fast() -> None:
    import json

    from representation.sprint18_harness import Sprint18ArmConfig, load_task6_binding

    binding = load_task6_binding(
        "experiments/sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    lock = json.loads(open("experiments/sprint18-task14-config-lock.json").read())
    arm = Sprint18ArmConfig(**lock["locked_configs"]["B0-FULL/seed-181801"])
    with pytest.raises(ValueError, match="custody file is missing"):
        restore_s24_provenanced_bank(
            "/tmp/does-not-exist-s24-bank.pt",
            binding=binding,
            arm=arm,
            expected_bank_sha256="b" * 64,
            expected_checkpoint_sha256="a" * 64,
        )
