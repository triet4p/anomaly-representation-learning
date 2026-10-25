"""Focused Task 6 behavioral tests: leakage, restoration and real algorithms.

Every test exercises a real fail-closed path or a real training/inference
outcome of ``representation.sprint24_learners``; none asserts wiring, source
text, or a mocked response. Row-level boundary checks use the accepted DEV-02
metadata fixture through the public views builder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

# The documented deterministic-CUDA setting must exist in the process
# environment before CUDA initialization; no other test path depends on it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from representation.config import V1Config
from representation.sprint18_checkpoint import state_sha256
from representation.sprint24_learners import (
    METHOD_RANDOM_FROZEN,
    METHOD_SCRATCH,
    S24_CUBLAS_WORKSPACE_CONFIG,
    S24FeatureNormalizer,
    S24LabelConsumptionLedger,
    S24TrainSupport,
    assert_s24_support_role,
    assert_s24_trainables,
    build_s24_random_frozen_learner,
    build_s24_scratch_learner,
    build_s24_train_support,
    enable_s24_deterministic_execution,
    fit_s24_learner,
    restore_s24_learner_checkpoint,
    save_s24_learner_checkpoint,
)
from representation.sprint24_views import HistoryViews, LabelReleaseGate, build_history_views
from synth.schema import FileSample, SampleLabel

FIXTURE_DIR = Path("artifacts/sprint-24/evidence/s24-t4/s24-t4-dev02-fixture")
HISTORY_ID = "S18I-ITER-0005-DEVELOPMENT-02"


@pytest.fixture(scope="module")
def dev02_views() -> HistoryViews:
    rows = json.loads((FIXTURE_DIR / "rows.json").read_text(encoding="utf-8"))
    windows = json.loads((FIXTURE_DIR / "failure_windows.json").read_text(encoding="utf-8"))
    degradations = json.loads(
        (FIXTURE_DIR / "degradation_episodes.json").read_text(encoding="utf-8")
    )
    maintenance = json.loads(
        (FIXTURE_DIR / "maintenance_windows.json").read_text(encoding="utf-8")
    )
    edges = json.loads((FIXTURE_DIR / "bin_edges.json").read_text(encoding="utf-8"))
    return build_history_views(
        HISTORY_ID,
        rows,
        windows,
        degradations,
        maintenance,
        cutoff_c1=edges[4],
        cutoff_c2=edges[7],
        bin_edges=edges,
    )


@pytest.fixture(scope="module")
def dev02_gate(dev02_views: HistoryViews) -> LabelReleaseGate:
    rows = json.loads((FIXTURE_DIR / "rows.json").read_text(encoding="utf-8"))
    labels = {
        str(row["file_id"]): {"file_label": row["file_label"]}
        for row in rows
    }
    return LabelReleaseGate(dev02_views, labels)


def _synthetic_cell(
    *,
    d_model: int = 128,
    train_count: int = 8,
    val_count: int = 4,
    value_shift: float = 0.0,
) -> tuple[HistoryViews, LabelReleaseGate, S24TrainSupport, dict[str, FileSample], dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(11)
    train = [
        _sample(f"train-{index:02d}", seed=index) for index in range(train_count)
    ]
    val = [
        _sample(f"val-{index:02d}", seed=100 + index) for index in range(val_count)
    ]
    ids = [sample.file_id for sample in train] + [sample.file_id for sample in val]
    labels = {
        file_id: {"file_label": "abnormal" if index % 2 else "normal"}
        for index, file_id in enumerate(ids)
    }
    views = HistoryViews(
        history_id="S24-UNIT-HISTORY",
        frames={
            "V-ADAPT": tuple(ids),
            "V-VAL": tuple(sample.file_id for sample in val),
        },
        budgets={"B1": tuple(sample.file_id for sample in train)},
        label_release={
            "V-ADAPT": 0.0,
            "V-SEQ-E1": 0.0,
            "V-SEQ-E2": 0.0,
            "V-SEQ-E3": 0.0,
        },
        groups={f"group-{index}": (file_id,) for index, file_id in enumerate(ids)},
        manifest_digest="d" * 64,
    )
    gate = LabelReleaseGate(views, labels)
    support = build_s24_train_support(
        views=views,
        gate=gate,
        budget="B1",
        now_s=1.0,
        waveform_ids=ids,
    )
    embeddings = {
        file_id: torch.randn(d_model, generator=generator) + value_shift
        for file_id in ids
    }
    samples = {sample.file_id: sample for sample in train + val}
    return views, gate, support, samples, embeddings


def _sample(file_id: str, *, seed: int, timesteps: int = 48) -> FileSample:
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((6, timesteps)).astype(np.float32)
    return FileSample(
        x=signal,
        file_id=file_id,
        file_label=SampleLabel.NORMAL,
        seed=seed,
        generator_version="s24-t6-test",
        config_hash="s24-t6-test",
        regime_sequence=[],
    )


def test_support_roles_reject_untrained_and_evaluation_frames(
    dev02_views: HistoryViews,
) -> None:
    with pytest.raises(ValueError, match="outside V-VAL"):
        assert_s24_support_role(
            dev02_views, dev02_views.frame("V-TRIG"), role="validation"
        )
    with pytest.raises(ValueError, match="outside B1"):
        assert_s24_support_role(
            dev02_views, dev02_views.frame("V-R0"), role="training", budget="B1"
        )
    with pytest.raises(ValueError, match="evaluation frame"):
        assert_s24_support_role(
            dev02_views, dev02_views.frame("V-FUTURE"), role="evaluation", frame="V-VAL"
        )
    assert assert_s24_support_role(
        dev02_views, dev02_views.frame("V-FUTURE"), role="evaluation", frame="V-FUTURE"
    ) == tuple(dev02_views.frame("V-FUTURE"))


def test_support_builder_rejects_premature_and_unreleased_labels(
    dev02_views: HistoryViews, dev02_gate: LabelReleaseGate
) -> None:
    budget_ids = list(dev02_views.budgets["B1"])
    val_ids = list(dev02_views.frame("V-VAL"))
    with pytest.raises(ValueError, match="not yet released"):
        build_s24_train_support(
            views=dev02_views,
            gate=dev02_gate,
            budget="B1",
            now_s=dev02_views.label_release["V-ADAPT"] - 1.0,
            waveform_ids=budget_ids + val_ids,
        )
    with pytest.raises(ValueError, match="missing .* waveform"):
        build_s24_train_support(
            views=dev02_views,
            gate=dev02_gate,
            budget="B1",
            now_s=dev02_views.label_release["V-ADAPT"] + 1.0,
            waveform_ids=budget_ids[:-1],
        )
    with pytest.raises(KeyError):
        dev02_gate.require_released(
            list(dev02_views.frame("V-FUTURE"))[:3], now_s=10**12
        )


def test_checkpoint_restore_reproduces_predictions_and_detects_tampering(
    tmp_path: Path,
) -> None:
    views, gate, support, _, embeddings = _synthetic_cell()
    learner = build_s24_random_frozen_learner(
        model_config=V1Config(sequence_layers=1),
        support=support,
        init_seed=241801,
        mask_model_seed=181801,
        frozen_embeddings=embeddings,
    )
    fit_s24_learner(learner, epochs=4)
    matrix = torch.stack([embeddings[file_id] for file_id in support.label_ids])
    before = learner.predict_embeddings(matrix)
    path = tmp_path / "restore.pt"
    save_s24_learner_checkpoint(path, learner)
    restored = restore_s24_learner_checkpoint(path, views=views, gate=gate)
    assert torch.equal(before, restored.predict_embeddings(matrix))
    assert restored.normalizer is not None
    assert learner.normalizer is not None
    assert torch.equal(restored.normalizer.mean, learner.normalizer.mean)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["support"]["training_labels"][0] = 1 - payload["support"]["training_labels"][0]
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="digest"):
        restore_s24_learner_checkpoint(tampered, views=views, gate=gate)


def test_restore_rejects_a_different_view_manifest(tmp_path: Path) -> None:
    views, gate, support, _, embeddings = _synthetic_cell()
    learner = build_s24_random_frozen_learner(
        model_config=V1Config(sequence_layers=1),
        support=support,
        init_seed=241802,
        mask_model_seed=181801,
        frozen_embeddings=embeddings,
    )
    fit_s24_learner(learner, epochs=2)
    path = tmp_path / "manifest.pt"
    save_s24_learner_checkpoint(path, learner)
    other = HistoryViews(
        history_id=views.history_id,
        frames=views.frames,
        budgets=views.budgets,
        groups=views.groups,
        label_release=views.label_release,
        manifest_digest="e" * 64,
    )
    with pytest.raises(ValueError, match="view manifest"):
        restore_s24_learner_checkpoint(path, views=other, gate=None)


def test_resume_from_checkpoint_matches_uninterrupted_training(tmp_path: Path) -> None:
    views, gate, support, _, embeddings = _synthetic_cell()
    config = V1Config(sequence_layers=1)

    uninterrupted = build_s24_random_frozen_learner(
        model_config=config,
        support=support,
        init_seed=241803,
        mask_model_seed=181801,
        frozen_embeddings=embeddings,
    )
    fit_s24_learner(uninterrupted, epochs=6)

    partial = build_s24_random_frozen_learner(
        model_config=config,
        support=support,
        init_seed=241803,
        mask_model_seed=181801,
        frozen_embeddings=embeddings,
    )
    fit_s24_learner(partial, epochs=3)
    path = tmp_path / "resume.pt"
    save_s24_learner_checkpoint(path, partial)
    resumed = restore_s24_learner_checkpoint(path, views=views, gate=gate)
    fit_s24_learner(resumed, epochs=3)
    matrix = torch.stack([embeddings[file_id] for file_id in support.label_ids])
    assert torch.equal(
        uninterrupted.head.linear.weight, resumed.head.linear.weight
    )
    assert torch.equal(uninterrupted.head.linear.bias, resumed.head.linear.bias)
    assert torch.equal(
        uninterrupted.predict_embeddings(matrix), resumed.predict_embeddings(matrix)
    )


def test_double_consumption_of_one_label_event_is_rejected() -> None:
    views, gate, support, _, embeddings = _synthetic_cell()
    config = V1Config(sequence_layers=1)
    ledger = S24LabelConsumptionLedger()
    first = build_s24_random_frozen_learner(
        model_config=config,
        support=support,
        init_seed=241801,
        mask_model_seed=181801,
        frozen_embeddings=embeddings,
    )
    fit_s24_learner(first, epochs=2, ledger=ledger)

    second = build_s24_random_frozen_learner(
        model_config=config,
        support=support,
        init_seed=241801,
        mask_model_seed=181801,
        frozen_embeddings=embeddings,
    )
    with pytest.raises(ValueError, match="already consumed"):
        fit_s24_learner(second, epochs=1, ledger=ledger)

    later_support = build_s24_train_support(
        views=views,
        gate=gate,
        budget="B1",
        now_s=support.now_s + 10.0,
        waveform_ids=list(support.label_ids),
    )
    third = build_s24_random_frozen_learner(
        model_config=config,
        support=later_support,
        init_seed=241801,
        mask_model_seed=181801,
        frozen_embeddings=embeddings,
    )
    with pytest.raises(ValueError, match="already consumed"):
        fit_s24_learner(third, epochs=1, ledger=ledger)


def test_scratch_updates_the_encoder_while_frozen_heads_do_not() -> None:
    _, _, support, samples, embeddings = _synthetic_cell(d_model=16, value_shift=0.0)
    config = V1Config(
        d_model=16, sequence_layers=1, attention_heads=4, n_robots=9, n_programs=8
    )
    frozen = build_s24_random_frozen_learner(
        model_config=config,
        support=support,
        init_seed=241802,
        mask_model_seed=181801,
        frozen_embeddings=embeddings,
    )
    frozen_encoder = state_sha256(frozen.encoder.state_dict())
    frozen_head = state_sha256(frozen.head.state_dict())
    fit_s24_learner(frozen, epochs=3)
    assert state_sha256(frozen.encoder.state_dict()) == frozen_encoder
    assert state_sha256(frozen.head.state_dict()) != frozen_head
    frozen_scope = assert_s24_trainables(frozen)
    assert frozen_scope["declared_trainable_parameters"] == 17
    assert frozen_scope["encoder_gradient_parameters"] == 0

    scratch = build_s24_scratch_learner(
        model_config=config, support=support, mask_model_seed=181801
    )
    student_before = state_sha256(scratch.encoder.state_dict())
    target_before = state_sha256(scratch.encoder.target_encoder.state_dict())
    head_before = state_sha256(scratch.head.state_dict())
    summary = fit_s24_learner(scratch, samples, epochs=2)
    assert summary["epochs_completed"] == 2
    history = cast(list[dict[str, float]], summary["history"])
    assert history[-1]["train_loss"] > 0.0
    assert state_sha256(scratch.encoder.state_dict()) != student_before
    assert state_sha256(scratch.encoder.target_encoder.state_dict()) != target_before
    assert state_sha256(scratch.head.state_dict()) != head_before
    scratch_scope = assert_s24_trainables(scratch)
    declared = cast(int, scratch_scope["declared_trainable_parameters"])
    encoder_declared = cast(int, scratch_scope["encoder_declared_parameters"])
    gradient = cast(int, scratch_scope["gradient_trainable_parameters"])
    assert declared == encoder_declared + 17
    assert gradient < declared
    validation = scratch.score_validation(samples)
    assert np.isfinite(validation["val_loss"]) and validation["val_loss"] > 0.0
    assert frozen.method == METHOD_RANDOM_FROZEN and scratch.method == METHOD_SCRATCH


def test_normalizer_is_fitted_on_training_support_only() -> None:
    _, _, support, _, embeddings = _synthetic_cell(value_shift=0.0)
    shifted = {file_id: value + 50.0 for file_id, value in embeddings.items()}
    learner = build_s24_random_frozen_learner(
        model_config=V1Config(sequence_layers=1),
        support=support,
        init_seed=241801,
        mask_model_seed=181801,
        frozen_embeddings=shifted,
    )
    fit_s24_learner(learner, epochs=1)
    assert learner.normalizer is not None
    train_matrix = torch.stack(
        [shifted[file_id] for file_id in support.training_ids]
    )
    expected = S24FeatureNormalizer.fit(train_matrix)
    assert torch.allclose(learner.normalizer.mean, expected.mean, atol=1e-6)
    assert torch.allclose(learner.normalizer.std, expected.std, atol=1e-6)
    combined = torch.stack([shifted[file_id] for file_id in support.label_ids])
    assert not torch.allclose(learner.normalizer.mean, combined.mean(dim=0), atol=1e-3)


def _device_type(module: torch.nn.Module) -> str:
    parameter = next(module.parameters(), None)
    assert parameter is not None
    return parameter.device.type


def _nested_tensors(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        found: list[torch.Tensor] = []
        for item in value.values():
            found.extend(_nested_tensors(item))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for item in value:
            found.extend(_nested_tensors(item))
        return found
    return []


def test_cpu_default_keeps_every_training_tensor_on_cpu() -> None:
    """The default device must keep the whole cell on the CPU (compatibility contract)."""
    _, _, support, _, embeddings = _synthetic_cell()
    learner = build_s24_random_frozen_learner(
        model_config=V1Config(sequence_layers=1),
        support=support,
        init_seed=241801,
        mask_model_seed=181801,
        frozen_embeddings=embeddings,
    )
    assert learner.device == torch.device("cpu")
    assert _device_type(learner.encoder) == "cpu"
    assert _device_type(learner.head) == "cpu"
    assert learner._frozen_embeddings and all(
        vector.device.type == "cpu" for vector in learner._frozen_embeddings.values()
    )
    fit_s24_learner(learner, epochs=1)
    assert learner.normalizer is not None
    assert learner.normalizer.mean.device.type == "cpu"
    assert learner._train_features is not None
    assert learner._train_features.device.type == "cpu"
    assert learner._val_features is not None
    assert learner._val_features.device.type == "cpu"
    assert learner._optimizer is not None
    assert all(
        value.device.type == "cpu"
        for entry in learner._optimizer.state.values()
        for value in entry.values()
        if isinstance(value, torch.Tensor)
    )
    assert assert_s24_trainables(learner)["declared_trainable_parameters"] == 129


def test_checkpoint_is_device_neutral_and_binds_its_rng_state(tmp_path: Path) -> None:
    """A checkpoint is written as CPU tensors plus bound execution/RNG provenance."""
    views, gate, support, _, embeddings = _synthetic_cell()
    learner = build_s24_random_frozen_learner(
        model_config=V1Config(sequence_layers=1),
        support=support,
        init_seed=241802,
        mask_model_seed=181801,
        frozen_embeddings=embeddings,
    )
    fit_s24_learner(learner, epochs=2)
    matrix = torch.stack([embeddings[file_id] for file_id in support.label_ids])
    before = learner.predict_embeddings(matrix)

    path = tmp_path / "device-neutral.pt"
    metadata = save_s24_learner_checkpoint(path, learner)
    assert metadata["device"] == "cpu"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    execution = payload["execution"]
    assert execution["device"] == "cpu" and execution["device_type"] == "cpu"
    assert execution["deterministic_algorithms"] is False
    assert execution["cublas_workspace_config"] is None
    stored = _nested_tensors(payload["state"])
    assert stored
    assert all(tensor.device.type == "cpu" for tensor in stored)
    assert payload["state"]["torch_cuda_rng_state"] == []

    restored = restore_s24_learner_checkpoint(path, views=views, gate=gate, device="cpu")
    assert restored.device == torch.device("cpu")
    assert torch.equal(before, restored.predict_embeddings(matrix))
    assert_s24_trainables(restored)

    # Restore -> re-save -> restore must keep the pending RNG binding instead of
    # recording the ambient process RNG.
    chained = restore_s24_learner_checkpoint(path, views=views, gate=gate)
    chain_path = tmp_path / "device-neutral-chain.pt"
    save_s24_learner_checkpoint(chain_path, chained)
    chain_payload = torch.load(chain_path, map_location="cpu", weights_only=False)
    assert torch.equal(
        chain_payload["state"]["torch_rng_state"], payload["state"]["torch_rng_state"]
    )
    chained_restored = restore_s24_learner_checkpoint(
        chain_path, views=views, gate=gate
    )
    assert torch.equal(before, chained_restored.predict_embeddings(matrix))

    payload["state"]["torch_rng_state"][0] ^= 1
    tampered = tmp_path / "device-neutral-tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="RNG state does not match"):
        restore_s24_learner_checkpoint(tampered, views=views, gate=gate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_scratch_training_restore_and_resume_are_device_aligned(
    tmp_path: Path,
) -> None:
    """CUDA training, checkpoint restore and deterministic resume keep one device.

    The learner enables the documented deterministic CUDA execution setting, so
    restore, same-device resume and two independent identically seeded runs must
    all be exactly equal; the resume leg also proves the RNG binding. AdamW keeps
    its ``step`` counter as a CPU scalar by design, while the moment tensors
    follow the parameter device.
    """
    views, gate, support, samples, _ = _synthetic_cell(
        d_model=16, train_count=4, val_count=2
    )
    config = V1Config(
        d_model=16, sequence_layers=1, attention_heads=4, n_robots=9, n_programs=8
    )
    learner = build_s24_scratch_learner(
        model_config=config, support=support, mask_model_seed=181801, device="cuda"
    )
    fit_s24_learner(learner, samples, epochs=2)
    assert learner.device.type == "cuda"
    assert _device_type(learner.encoder) == "cuda"
    assert _device_type(learner.head) == "cuda"
    assert learner.normalizer is not None and learner.normalizer.mean.is_cuda
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert learner._optimizer is not None
    moments = [
        value
        for entry in learner._optimizer.state.values()
        for key, value in entry.items()
        if key != "step" and isinstance(value, torch.Tensor)
    ]
    assert moments and all(value.is_cuda for value in moments)
    steps = [
        entry["step"] for entry in learner._optimizer.state.values() if "step" in entry
    ]
    assert steps and all(
        isinstance(step, torch.Tensor)
        and step.device.type == "cpu"
        and step.numel() == 1
        for step in steps
    )
    support_samples = [samples[file_id] for file_id in support.label_ids]
    before = learner.predict(support_samples, history_id=support.history_id)

    path = tmp_path / "cuda-cell.pt"
    save_s24_learner_checkpoint(path, learner)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["execution"]["device_type"] == "cuda"
    assert payload["state"]["torch_cuda_rng_state"]
    assert all(t.device.type == "cpu" for t in _nested_tensors(payload["state"]))
    restored = restore_s24_learner_checkpoint(
        path, views=views, gate=gate, device="cuda"
    )
    assert restored.device.type == "cuda"
    assert restored.predict(support_samples, history_id=support.history_id) == before
    assert_s24_trainables(restored)

    uninterrupted = build_s24_scratch_learner(
        model_config=config, support=support, mask_model_seed=181801, device="cuda"
    )
    fit_s24_learner(uninterrupted, samples, epochs=2)
    twin = build_s24_scratch_learner(
        model_config=config, support=support, mask_model_seed=181801, device="cuda"
    )
    fit_s24_learner(twin, samples, epochs=2)
    assert state_sha256(twin.encoder.state_dict()) == state_sha256(
        uninterrupted.encoder.state_dict()
    )
    assert torch.equal(twin.head.linear.weight, uninterrupted.head.linear.weight)
    partial = build_s24_scratch_learner(
        model_config=config, support=support, mask_model_seed=181801, device="cuda"
    )
    fit_s24_learner(partial, samples, epochs=1)
    partial_path = tmp_path / "cuda-cell-partial.pt"
    save_s24_learner_checkpoint(partial_path, partial)
    resumed = restore_s24_learner_checkpoint(
        partial_path, views=views, gate=gate, device="cuda"
    )
    fit_s24_learner(resumed, samples, epochs=1)
    assert state_sha256(uninterrupted.encoder.state_dict()) == state_sha256(
        resumed.encoder.state_dict()
    )
    assert torch.equal(uninterrupted.head.linear.weight, resumed.head.linear.weight)
    assert uninterrupted.predict(
        support_samples, history_id=support.history_id
    ) == resumed.predict(support_samples, history_id=support.history_id)


def test_deterministic_execution_is_cuda_only_and_fails_closed() -> None:
    """CPU needs no CUDA environment variable; an absent CUDA target fails closed."""
    before = torch.are_deterministic_algorithms_enabled()
    settings = enable_s24_deterministic_execution("cpu")
    assert settings["device"] == "cpu"
    assert settings["deterministic_algorithms"] is False
    assert torch.are_deterministic_algorithms_enabled() is before
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA is unavailable"):
            enable_s24_deterministic_execution("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deterministic_cuda_execution_requires_the_documented_workspace_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented cuBLAS workspace setting is mandatory before CUDA init."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        enable_s24_deterministic_execution("cuda")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", S24_CUBLAS_WORKSPACE_CONFIG)
    settings = enable_s24_deterministic_execution("cuda")
    assert settings["deterministic_algorithms"] is True
    assert settings["cublas_workspace_config"] == S24_CUBLAS_WORKSPACE_CONFIG
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
