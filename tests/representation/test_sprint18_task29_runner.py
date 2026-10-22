"""Focused Task 29 N3-CCRP tests: the real runner post-train path and channel diagnostics.

Covers consumer-visible behavior only: the N3 arm reaches the frozen public
reference-bank, q95 and Development scoring path on its own restored primary
checkpoint; the frozen N3 selector over a complete real Fit epoch is
reproducible and round-robin balanced; and cross-channel contribution
diagnostics are per-file, deterministic, never fused, and cannot change
``S_pred``/``S_pop`` even when the auxiliary head is overwritten. The persisted
preflight payload dispatches each frozen seed its own selector observation, and
the actual per-seed runner rejects another seed's observation. No
wiring/source-text/incidental assertions.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from representation.config import V1Config
from representation.data import collate_variable_files
from representation.model import V1RepresentationModel
from representation.sprint18_checkpoint import (
    SCHEDULER_ID,
    _canonical_sha256,
    coefficient_for_epoch,
    learning_rate_for_epoch,
    restore_sprint18_checkpoint,
    save_sprint18_checkpoint,
    state_sha256,
)
from representation.sprint18_harness import (
    BATCH_SIZE,
    MODEL_SEEDS,
    BoundHistory,
    Task6Binding,
    deterministic_sampler_order,
    fit_calibration_q95,
    fit_fit_reference_bank,
    load_certified_development_support,
    load_certified_sample_subset,
    make_sprint18_arm_config,
    score_ccrp_channel_rows,
    score_certified_rows,
    validate_score_maps,
)
from representation.trainer import RepresentationTrainer
from synth.config import PatchConfig
from synth.dataset import _sample_bytes
from synth.patchify import Patchifier
from synth.schema import FileSample, SampleLabel

import sprint18_task29_n3_runner as runner

SEED = MODEL_SEEDS[0]
LOCK = "6eb108b2073181d427f1aedfc81fb3bae4c29595108d15de486082800bdd9d6e"
FIT_HISTORIES = tuple(f"S18I-ITER-0005-FIT-TINY-0{index}" for index in range(1, 4))
CALIBRATION_HISTORY = "S18I-ITER-0005-CAL-TINY-01"
DEVELOPMENT_HISTORIES = tuple(f"S18I-ITER-0005-DEV-TINY-0{index}" for index in range(1, 5))
# 784 timesteps -> 48 fully valid patches -> N3 Q = max(1, round(0.25*48)) = 12;
# per-file patch counts (and therefore Q) are derived by ``_patch_count``.
FILE_TIMESTEPS = 784


@dataclass(frozen=True)
class _TinyHistory:
    history: BoundHistory
    manifest_path: Path
    shard_paths: dict[str, Path]
    eligible_ids: tuple[str, ...]


def _tiny_config(seed: int = SEED) -> V1Config:
    return V1Config(
        n_channels=6,
        patch_size=32,
        stride=16,
        d_model=16,
        attention_heads=4,
        sequence_layers=1,
        dropout=0.0,
        n_robots=9,
        n_programs=8,
        seed=seed,
        contrastive_weight_max=runner.LAMBDA_INFO_MAX,
        ema_decay=0.996,
    )


def _arm(selection_state: str = "locked"):
    return make_sprint18_arm_config(
        "N3-CCRP",
        SEED,
        {
            "lambda_info_max": runner.LAMBDA_INFO_MAX,
            "lambda_hcc_max": None,
            "alpha": None,
            "beta": runner.BETA,
        },
        selection_state=selection_state,  # type: ignore[arg-type]
        coefficient_lock_sha256=LOCK if selection_state == "locked" else None,
    )


def _patch_count(timesteps: int) -> int:
    """Mirror the frozen Patchifier window geometry for a single-timestep axis."""
    starts = list(range(0, timesteps - 32 + 1, 16)) if timesteps >= 32 else []
    if not starts:
        return 1
    covered_end = starts[-1] + 32
    return len(starts) + (1 if covered_end < timesteps else 0)


def _tiny_binding(
    root: Path, *, timesteps_by_index: dict[int, int] | None = None,
) -> tuple[Task6Binding, dict[str, _TinyHistory]]:
    """Bind a tiny 3-FIT/1-CALIBRATION/4-DEVELOPMENT certified-shaped roster."""
    generator = "s18-t29-n3-fixture"
    config_hash = "s18-t29-n3-config"
    plan = (
        *[(history_id, "FIT", 4) for history_id in FIT_HISTORIES],
        (CALIBRATION_HISTORY, "CALIBRATION", 3),
        *[(history_id, "DEVELOPMENT", 2) for history_id in DEVELOPMENT_HISTORIES],
    )
    rng = np.random.default_rng(92901)
    total = sum(int(item[-1]) for item in plan)
    lengths = {index: (timesteps_by_index or {}).get(index, FILE_TIMESTEPS) for index in range(total)}
    signals = {
        f"n29-tiny-{index:02d}": rng.normal(size=(6, lengths[index])).astype(np.float32)
        for index in range(total)
    }
    histories: dict[str, _TinyHistory] = {}
    cursor = 0
    for history_id, role, count in plan:
        relative = f"{history_id}/shard-00000.zip"
        shard_path = root / relative
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        file_ids = [f"n29-tiny-{cursor + offset:02d}" for offset in range(count)]
        items = []
        for offset, file_id in enumerate(file_ids):
            sample = FileSample(
                x=signals[file_id],
                file_id=file_id,
                file_label=SampleLabel.NORMAL,
                seed=SEED + cursor + offset,
                generator_version=generator,
                config_hash=config_hash,
                regime_sequence=[],
                robot_idx=(cursor + offset) % 3,
                program_idx=(cursor + offset) % 2,
            )
            sample.validate()
            items.append(sample)
        with zipfile.ZipFile(shard_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in items:
                info = zipfile.ZipInfo(f"{item.file_id}.npz", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, _sample_bytes(item))
        manifest = {
            "format": 1,
            "role": history_id,
            "protocol": "sprint15-benchmark-protocol-v7",
            "config_hash": config_hash,
            "files": [
                {
                    "file_id": item.file_id,
                    "file_label": item.file_label.value,
                    "robot_id": f"robot-{item.robot_idx + 1:02d}",
                    "program_id": f"program-{item.program_idx + 1:02d}",
                    "is_quarantined": False,
                    "is_censored": False,
                    "start_time": 0.0,
                    "end_time": float(lengths[cursor + offset]),
                    "n_valid_patches": _patch_count(lengths[cursor + offset]),
                }
                for offset, item in enumerate(items)
            ],
            "shards": [
                {
                    "path": relative,
                    "file_ids": [item.file_id for item in items],
                    "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                }
            ],
            "maintenance_windows": {},
        }
        manifest_path = root / f"{history_id}.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        histories[history_id] = _TinyHistory(
            history=BoundHistory(
                assigned_role=role,  # type: ignore[arg-type]
                actual_history_id=history_id,
                schema_history_id=f"H-TINY-{history_id}",
                manifest_sha256=digest,
                data_seed=32901,
                sample_count=len(items),
                config_hash=config_hash,
                relative_path=f"tiny-fixture/{role}/{history_id}",
                data_permitted_use=role,
                model_permitted_use=role,
            ),
            manifest_path=manifest_path,
            shard_paths={relative: shard_path},
            eligible_ids=tuple(item.file_id for item in items),
        )
        cursor += count
    digest_source = "".join(item.history.manifest_sha256 for item in histories.values())
    binding = Task6Binding(
        candidate_id="S18-T29-N3-TINY",
        data_root_relative="tiny-fixture",
        handoff_sha256=hashlib.sha256(b"s18-t29-n3-handoff").hexdigest(),
        data_binding_raw_sha256=hashlib.sha256(b"s18-t29-n3-raw").hexdigest(),
        data_binding_canonical_sha256=hashlib.sha256(b"s18-t29-n3-canonical").hexdigest(),
        whole16_certification_sha256=hashlib.sha256(b"s18-t29-n3-whole16").hexdigest(),
        qualification_record_sha256=hashlib.sha256(b"s18-t29-n3-qual").hexdigest(),
        manifest_aggregate_sha256=hashlib.sha256(digest_source.encode("ascii")).hexdigest(),
        protocol_sha256=hashlib.sha256(b"s18-t29-n3-protocol").hexdigest(),
        mechanism_spec_sha256={},
        histories=tuple(item.history for item in histories.values()),
        fit_healthy_rows=12,
        calibration_healthy_rows=3,
        executable_binding_sha256=hashlib.sha256(digest_source.encode("ascii")).hexdigest(),
    )
    return binding, histories


def _rows(binding: Task6Binding, histories: dict[str, _TinyHistory], role: str) -> list:
    if role == "DEVELOPMENT":
        rows: list = []
        for item in histories.values():
            if item.history.assigned_role != "DEVELOPMENT":
                continue
            rows.extend(
                load_certified_development_support(
                    binding,
                    item.history.actual_history_id,
                    item.manifest_path,
                    item.shard_paths,
                )
            )
        return rows
    rows = []
    action = "train_fit" if role == "FIT" else "calibrate_q95"
    for item in histories.values():
        if item.history.assigned_role != role:
            continue
        rows.extend(
            load_certified_sample_subset(
                binding,
                item.history.actual_history_id,
                item.manifest_path,
                item.shard_paths,
                list(item.eligible_ids),
                role_action=action,
                verify_all_shards=True,
            )
        )
    return rows


def _trained_n3_checkpoint(tmp_path: Path, binding: Task6Binding, fit_rows, seed: int = SEED):
    """Run real N3 optimizer steps, save a coherent experiment checkpoint, strict-restore it."""
    config = _tiny_config(seed)
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    model = V1RepresentationModel(config, patchifier=patchifier)
    arm = _arm("locked")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0)
    trainer = RepresentationTrainer(model, optimizer=optimizer, max_grad_norm=1.0, seed=seed)
    fit_history_ids = [item for item in FIT_HISTORIES]
    fit_role_sha256 = _canonical_sha256(
        sorted(binding.history(h).manifest_sha256 for h in fit_history_ids)
    )
    batch = collate_variable_files(
        [row.sample for row in fit_rows], patchifier, masking_config=config, masking_seed=1,
    )
    before = state_sha256(model.state_dict())
    trainer.train_sprint18_ccrp_epoch(
        [batch], arm=arm, epoch=1, collect_balance=True, fit_role_sha256=fit_role_sha256,
    )
    assert trainer.step == 1
    assert state_sha256(model.state_dict()) != before
    source_ids = sorted(row.file_id for row in fit_rows)
    sampler_seed, epoch_order = deterministic_sampler_order(
        binding, arm, workload="representation", scope="experiment",
        role_history_ids=fit_history_ids, epoch=2, eligible_source_ids=source_ids,
    )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate_for_epoch(2)
    path = tmp_path / "tiny-n3.pt"
    digest = save_sprint18_checkpoint(
        path,
        model,
        optimizer,
        arm=arm,
        binding=binding,
        global_step=trainer.step,
        completed_epoch=1,
        initialization_sha256=before,
        scheduler_state={
            "scheduler_id": SCHEDULER_ID,
            "completed_epochs": 1,
            "active_epoch": 2,
            "learning_rate": learning_rate_for_epoch(2),
            "global_step": trainer.step,
        },
        sampler_state={
            "scope": "experiment",
            "workload": "representation",
            "role_history_ids": fit_history_ids,
            "role_manifest_sha256": [binding.history(h).manifest_sha256 for h in fit_history_ids],
            "eligible_source_ids": source_ids,
            "eligible_source_ids_sha256": _canonical_sha256(source_ids),
            "sampler_epoch_seed": sampler_seed,
            "epoch_order_ids": list(epoch_order),
            "epoch_order_ids_sha256": _canonical_sha256(list(epoch_order)),
            "eligible_rows": len(source_ids),
            "batch_size": BATCH_SIZE,
            "batches_per_epoch": 1,
            "completed_epochs": 1,
            "current_epoch": 2,
            "batches_in_current_epoch": 0,
            "fit_role_sha256": fit_role_sha256,
        },
        coefficient_state={
            "epoch": 2,
            "global_step": trainer.step,
            "selection_state": "locked",
            "lambda_info": coefficient_for_epoch(runner.LAMBDA_INFO_MAX, 2),
            "lambda_hcc": None,
            "alpha": None,
            "beta": runner.BETA,
        },
        scope="experiment",
    )
    restored_model = V1RepresentationModel(config, patchifier=patchifier)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=learning_rate_for_epoch(2), weight_decay=0.0
    )
    restored = restore_sprint18_checkpoint(
        path, restored_model, restored_optimizer, arm=arm, binding=binding, allowed_scope="experiment",
    )
    assert restored.checkpoint_sha256 == digest
    assert state_sha256(restored_model.state_dict()) == state_sha256(model.state_dict())
    return restored_model, patchifier, restored, arm


def test_n3_selector_observation_is_reproducible_and_round_robin(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path / "fixture")
    fit_rows = _rows(binding, histories, "FIT")
    fit_ids = sorted(row.file_id for row in fit_rows)
    fit_history_ids = list(FIT_HISTORIES)
    fit_role_sha256 = _canonical_sha256(
        sorted(binding.history(h).manifest_sha256 for h in fit_history_ids)
    )
    observation = runner._selector_observation(
        binding=binding, arm=_arm("locked"), config=_tiny_config(), patchifier=runner._n3_patchifier(),
        fit_ids=fit_ids, fit_sample_by_id={row.file_id: row.sample for row in fit_rows},
        fit_history_ids=fit_history_ids, fit_role_sha256=fit_role_sha256, epoch=1,
    )
    assert observation["files"] == len(fit_ids) == 12
    assert observation["targets"] == sum(observation["channel_counts"])
    assert observation["channel_count_span"] <= 1
    assert observation["masks"]["fully_masked_never_selected"] is True
    assert observation["masks"]["final_equals_min_requested_unmasked"] is True
    assert observation["requested_total"] == 12 * 12
    # byte-for-byte reproducibility of the complete epoch observation
    repeated = runner._selector_observation(
        binding=binding, arm=_arm("locked"), config=_tiny_config(), patchifier=runner._n3_patchifier(),
        fit_ids=fit_ids, fit_sample_by_id={row.file_id: row.sample for row in fit_rows},
        fit_history_ids=fit_history_ids, fit_role_sha256=fit_role_sha256, epoch=1,
    )
    assert repeated["target_keys_sha256"] == observation["target_keys_sha256"]
    assert repeated["channel_counts"] == observation["channel_counts"]


def test_n3_public_bank_q95_and_development_scoring_on_restored_checkpoint(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path / "fixture")
    fit_rows = _rows(binding, histories, "FIT")
    calibration_rows = _rows(binding, histories, "CALIBRATION")
    model, patchifier, restored, arm = _trained_n3_checkpoint(tmp_path, binding, fit_rows)
    before = state_sha256(model.state_dict())
    bank = fit_fit_reference_bank(
        binding, arm, model, fit_rows, patchifier, checkpoint=restored, scope="experiment",
    )
    assert bank.provenance.source_rows == len(fit_rows)
    calibration_scores = score_certified_rows(
        binding, arm, model, bank, calibration_rows, patchifier,
        checkpoint=restored, action="calibrate_q95", scope="experiment",
    )
    validate_score_maps(calibration_scores.support, calibration_scores.scores)
    calibration = fit_calibration_q95(binding, calibration_scores, calibration_rows, scope="experiment")
    assert math.isfinite(calibration.S_pred.threshold) and math.isfinite(calibration.S_pop.threshold)
    development_rows = _rows(binding, histories, "DEVELOPMENT")
    per_history = {history_id: [] for history_id in DEVELOPMENT_HISTORIES}
    for row in development_rows:
        per_history[row.actual_history_id].append(row)
    pred_scores: dict[str, float] = {}
    pop_scores: dict[str, float] = {}
    for history_id, rows in per_history.items():
        scored = score_certified_rows(
            binding, arm, model, bank, rows, patchifier,
            checkpoint=restored, action="score_development", scope="experiment",
        )
        validate_score_maps(scored.support, scored.scores)
        pred_scores.update(dict(scored.scores["S_pred"]))
        pop_scores.update(dict(scored.scores["S_pop"]))
    assert len(pred_scores) == len(development_rows) == len(pop_scores)
    assert all(math.isfinite(value) for value in pred_scores.values())
    assert all(math.isfinite(value) for value in pop_scores.values())
    # Primary scoring never mutates the restored model and never fuses branches.
    assert state_sha256(model.state_dict()) == before
    assert pred_scores != pop_scores


def test_n3_channel_diagnostics_are_unfused_and_never_touch_primary_scores(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path / "fixture")
    fit_rows = _rows(binding, histories, "FIT")
    model, patchifier, restored, arm = _trained_n3_checkpoint(tmp_path, binding, fit_rows)
    before = state_sha256(model.state_dict())
    item = histories[DEVELOPMENT_HISTORIES[0]]
    rows = load_certified_development_support(
        binding, item.history.actual_history_id, item.manifest_path, item.shard_paths,
    )
    bank = fit_fit_reference_bank(
        binding, arm, model, fit_rows, patchifier, checkpoint=restored, scope="experiment",
    )
    primary = score_certified_rows(
        binding, arm, model, bank, rows, patchifier,
        checkpoint=restored, action="score_development", scope="experiment",
    )
    diagnostics = score_ccrp_channel_rows(
        binding, arm, model, rows, patchifier, checkpoint=restored,
        action="score_development", scope="experiment", diagnostic_epoch=100,
    )
    assert diagnostics.actual_history_id == item.history.actual_history_id
    assert {record.file_id for record in diagnostics.records} == {row.file_id for row in rows}
    for record in diagnostics.records:
        assert sum(record.channel_counts) == record.selected_targets
        assert sum(record.channel_supported) == record.supported_targets
        assert record.supported_targets + record.near_zero_excluded == record.selected_targets
        if record.supported_targets:
            assert record.channel_loss is not None and math.isfinite(record.channel_loss)
            assert record.contribution_target_delta_norm_mean is not None
            assert record.predictor_effective_rank is None or record.predictor_effective_rank >= 1.0
            assert all(
                value is None or math.isfinite(value)
                for value in record.channel_mean_prediction_residual
            )
    # Channel diagnostics leave model state untouched and cannot alter the primaries.
    assert state_sha256(model.state_dict()) == before
    repeated = score_certified_rows(
        binding, arm, model, bank, rows, patchifier,
        checkpoint=restored, action="score_development", scope="experiment",
    )
    assert dict(primary.scores["S_pred"]) == dict(repeated.scores["S_pred"])
    assert dict(primary.scores["S_pop"]) == dict(repeated.scores["S_pop"])


def test_n3_primary_inference_never_consults_the_auxiliary_head(tmp_path, monkeypatch) -> None:
    """Corruption is training-only: the primary path must not touch the channel head."""
    binding, histories = _tiny_binding(tmp_path / "fixture")
    fit_rows = _rows(binding, histories, "FIT")
    model, patchifier, restored, arm = _trained_n3_checkpoint(tmp_path, binding, fit_rows)
    item = histories[DEVELOPMENT_HISTORIES[0]]
    rows = load_certified_development_support(
        binding, item.history.actual_history_id, item.manifest_path, item.shard_paths,
    )
    bank = fit_fit_reference_bank(
        binding, arm, model, fit_rows, patchifier, checkpoint=restored, scope="experiment",
    )
    before = state_sha256(model.state_dict())
    primary = score_certified_rows(
        binding, arm, model, bank, rows, patchifier,
        checkpoint=restored, action="score_development", scope="experiment",
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("primary inference must never call the training-only channel head")

    monkeypatch.setattr(model.channel_predictor, "forward", _boom)
    monkeypatch.setattr(model.channel_embedding, "forward", _boom)
    repeated = score_certified_rows(
        binding, arm, model, bank, rows, patchifier,
        checkpoint=restored, action="score_development", scope="experiment",
    )
    refit = fit_fit_reference_bank(
        binding, arm, model, fit_rows, patchifier, checkpoint=restored, scope="experiment",
    )
    assert dict(repeated.scores["S_pred"]) == dict(primary.scores["S_pred"])
    assert dict(repeated.scores["S_pop"]) == dict(primary.scores["S_pop"])
    assert refit.provenance.bank_sha256 == bank.provenance.bank_sha256
    assert state_sha256(model.state_dict()) == before


def test_n3_fit_bank_permission_refuses_development_rows(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path / "fixture")
    fit_rows = _rows(binding, histories, "FIT")
    model, patchifier, restored, arm = _trained_n3_checkpoint(tmp_path, binding, fit_rows)
    item = histories[DEVELOPMENT_HISTORIES[0]]
    development_rows = load_certified_development_support(
        binding, item.history.actual_history_id, item.manifest_path, item.shard_paths,
    )
    with pytest.raises(ValueError, match="non-Fit history"):
        fit_fit_reference_bank(
            binding, arm, model, development_rows, patchifier,
            checkpoint=restored, scope="experiment",
        )


def test_n3_task14_lock_binds_beta_and_seed_configs() -> None:
    hashes, _lock_sha256, arms = runner._task14_lock_config_hashes(REPO_ROOT)
    locked = runner._assert_locked_n3(arms)
    assert locked["coefficients"]["beta"] == runner.BETA == 0.5
    assert locked["coefficients"]["alpha"] is None
    assert locked["coefficients"]["lambda_info_max"] == runner.LAMBDA_INFO_MAX == 0.05
    arm = _arm("locked")
    from representation.sprint18_harness import arm_config_sha256

    assert arm_config_sha256(arm) == hashes["N3-CCRP/seed-181801"]


def _development_probe15_manifests(binding: Task6Binding, histories: dict[str, _TinyHistory], worktree: Path) -> dict[str, bytes]:
    """Lay the roster out for the runner and give Development its frozen probe15 fields."""
    raw_by_history: dict[str, bytes] = {}
    for history_id, item in histories.items():
        raw = item.manifest_path.read_bytes()
        manifest = json.loads(raw)
        if item.history.assigned_role == "DEVELOPMENT":
            for row in manifest["files"]:
                row["member_views"] = ["test-temporal"]
            robot = manifest["files"][0]["robot_id"]
            end = float(manifest["files"][0]["end_time"])
            manifest["failure_events"] = [
                {
                    "cohort": "P",
                    "severity": 2.0,
                    "robot_id": robot,
                    "degradation_onset": end - 1.0,
                    "failure_time": end,
                    "duration_d": 0.5,
                },
                {
                    "cohort": "W",
                    "severity": 2.0,
                    "robot_id": robot,
                    "degradation_onset": end,
                    "failure_time": end + 1.0,
                    "duration_d": 0.5,
                },
            ]
            raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        raw_by_history[history_id] = raw
        role_dir = worktree / item.history.relative_path
        role_dir.mkdir(parents=True, exist_ok=True)
        (role_dir / "manifest.json").write_bytes(raw)
        for shard in manifest["shards"]:
            target = role_dir / shard["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(item.shard_paths[shard["path"]]).read_bytes())
    return raw_by_history


def _rebind_development_manifests(binding: Task6Binding, histories: dict[str, _TinyHistory], raw_by_history: dict[str, bytes]) -> Task6Binding:
    updated = []
    for item in histories.values():
        history = item.history
        if history.assigned_role == "DEVELOPMENT":
            history = replace(
                history,
                manifest_sha256=hashlib.sha256(raw_by_history[history.actual_history_id]).hexdigest(),
            )
        updated.append(history)
    aggregate = hashlib.sha256("".join(item.manifest_sha256 for item in updated).encode("ascii")).hexdigest()
    return replace(
        binding,
        histories=tuple(updated),
        manifest_aggregate_sha256=aggregate,
        executable_binding_sha256=aggregate,
    )


def test_n3_run_seed_reaches_bank_q95_development_and_channel_reports(tmp_path) -> None:
    """The actual N3 per-seed runner must reach bank/q95/Development and write its record."""
    binding, histories = _tiny_binding(tmp_path / "fixture")
    worktree = tmp_path / "worktree"
    raw_by_history = _development_probe15_manifests(binding, histories, worktree)
    binding = _rebind_development_manifests(binding, histories, raw_by_history)
    fit_rows = _rows(binding, histories, "FIT")
    calibration_rows = _rows(binding, histories, "CALIBRATION")
    fit_ids = sorted(row.file_id for row in fit_rows)
    fit_history_ids = list(FIT_HISTORIES)
    fit_role_sha256 = _canonical_sha256(
        sorted(binding.history(h).manifest_sha256 for h in fit_history_ids)
    )
    expectation = runner._selector_observation(
        binding=binding, arm=_arm("locked"), config=_tiny_config(), patchifier=runner._n3_patchifier(),
        fit_ids=fit_ids, fit_sample_by_id={row.file_id: row.sample for row in fit_rows},
        fit_history_ids=fit_history_ids, fit_role_sha256=fit_role_sha256, epoch=1,
    )
    seed_dir = tmp_path / f"seed-{SEED}"
    seed_dir.mkdir()
    with (tmp_path / "progress.jsonl").open("w", encoding="utf-8") as progress:
        result = runner._run_seed(
            binding=binding,
            worktree=worktree,
            seed=SEED,
            fit_ids=fit_ids,
            fit_sample_by_id={row.file_id: row.sample for row in fit_rows},
            fit_history_ids=fit_history_ids,
            calibration_rows=calibration_rows,
            calibration_rows_by_history={
                CALIBRATION_HISTORY: [
                    row for row in calibration_rows if row.actual_history_id == CALIBRATION_HISTORY
                ]
            },
            development_history_ids=list(DEVELOPMENT_HISTORIES),
            device=torch.device("cpu"),
            progress=progress,
            seed_dir=seed_dir,
            fit_rows_by_id={row.file_id: row for row in fit_rows},
            selector_expectation=expectation,
        )
    record = result["record"]
    assert json.loads((seed_dir / "seed_record.json").read_text(encoding="utf-8")) == record
    assert record["status"] == "ok"
    assert record["normative"] is True
    assert record["global_step"] == runner.EPOCHS
    assert record["representation_epochs_completed"] == runner.EPOCHS
    assert sorted(record["checkpoints"], key=int) == ["10", "30", "60", "100"]
    assert record["primary"]["restored_forward_max_abs_diff"] == 0.0
    assert record["bank"]["source_rows"] == len(fit_ids)
    assert math.isfinite(record["calibration"]["S_pred_threshold"])
    assert math.isfinite(record["calibration"]["S_pop_threshold"])
    for branch in ("S_pred", "S_pop"):
        payload = record["development"]["branches"][branch]
        assert [row["history_id"] for row in payload["per_history"]] == list(DEVELOPMENT_HISTORIES)
    training = record["channel_diagnostics"]["training"]
    assert training["epochs"] == runner.EPOCHS
    assert training["per_epoch"][0]["channel_counts"] == expectation["channel_counts"]
    assert training["per_epoch"][0]["targets"] == expectation["targets"]
    assert sum(training["channel_support_total"].values()) == training["targets_total"]
    assert all(row["channel_count_span"] <= 1 for row in training["per_epoch"])
    assert all(row["effective_coefficient_L_channel"] == coefficient_for_epoch(runner.BETA, row["epoch"])
               for row in training["per_epoch"])
    channels = record["channel_diagnostics"]["development"]
    assert channels["fusion"].startswith("none")
    assert sorted(channels["per_history"]) == sorted(DEVELOPMENT_HISTORIES)
    for payload in channels["per_history"].values():
        assert payload["records"]
        for row in payload["records"]:
            assert sum(row["channel_counts"]) == row["selected_targets"]
            assert sum(row["channel_supported"]) == row["supported_targets"]
    assert record["compute"]["envelope_pass"] is True
    assert record["compute"]["parameters_by_branch"] == record["compute"]["b0_parameters_by_branch"]
    # the frozen N3 aux head is training-only and excluded from the inference path
    assert record["compute"]["channel_head_parameters_training_only"] > 0


def test_n3_preflight_binds_each_frozen_seed_to_its_own_selector_observation(tmp_path) -> None:
    """The persisted preflight payload must hand each frozen seed its own observation."""
    # Eleven default-length Fit files (48 patches -> Q=12) plus one 736-step
    # file (45 patches -> Q=11): 11*12 + 11 = 143 targets, so the frozen
    # (ordinal + model_seed + epoch) mod 6 rotation puts the shortfall on a
    # different channel for each frozen seed.
    binding, histories = _tiny_binding(tmp_path / "fixture", timesteps_by_index={11: 736})
    worktree = tmp_path / "worktree"
    raw_by_history = _development_probe15_manifests(binding, histories, worktree)
    binding = _rebind_development_manifests(binding, histories, raw_by_history)
    fit_rows = _rows(binding, histories, "FIT")
    calibration_rows = _rows(binding, histories, "CALIBRATION")
    fit_ids = sorted(row.file_id for row in fit_rows)
    fit_sample_by_id = {row.file_id: row.sample for row in fit_rows}
    fit_history_ids = list(FIT_HISTORIES)
    fit_role_sha256 = _canonical_sha256(
        sorted(binding.history(h).manifest_sha256 for h in fit_history_ids)
    )
    observations = runner._selector_observations(
        binding=binding,
        lock_digest=runner._coefficient_lock(binding),
        fit_ids=fit_ids,
        fit_sample_by_id=fit_sample_by_id,
        fit_history_ids=fit_history_ids,
        fit_role_sha256=fit_role_sha256,
    )
    assert sorted(observations) == sorted(str(seed) for seed in MODEL_SEEDS)
    assert len({tuple(item["channel_counts"]) for item in observations.values()}) == len(MODEL_SEEDS)
    assert len({item["target_keys_sha256"] for item in observations.values()}) == len(MODEL_SEEDS)
    # The persisted preflight payload is the dispatch contract.
    payload = {"schema": runner.PREFLIGHT_SCHEMA, "selector_observations": observations}
    for seed in MODEL_SEEDS:
        assert runner._selector_expectation(payload, seed) is observations[str(seed)]
    with pytest.raises(SystemExit):
        runner._selector_expectation(payload, MODEL_SEEDS[-1] + 1000)
    with pytest.raises(SystemExit):
        runner._selector_expectation({}, MODEL_SEEDS[0])

    calibration_rows_by_history = {
        CALIBRATION_HISTORY: [
            row for row in calibration_rows if row.actual_history_id == CALIBRATION_HISTORY
        ]
    }
    fit_rows_by_id = {row.file_id: row for row in fit_rows}
    # The actual runner accepts each seed's own observation over the real
    # post-train path (2 isolated epochs) and rejects another seed's observation.
    for seed in MODEL_SEEDS:
        seed_dir = tmp_path / f"seed-{seed}"
        seed_dir.mkdir()
        with (tmp_path / f"progress-{seed}.jsonl").open("w", encoding="utf-8") as progress:
            result = runner._run_seed(
                binding=binding,
                worktree=worktree,
                seed=seed,
                fit_ids=fit_ids,
                fit_sample_by_id=fit_sample_by_id,
                fit_history_ids=fit_history_ids,
                calibration_rows=calibration_rows,
                calibration_rows_by_history=calibration_rows_by_history,
                development_history_ids=list(DEVELOPMENT_HISTORIES),
                device=torch.device("cpu"),
                progress=progress,
                seed_dir=seed_dir,
                fit_rows_by_id=fit_rows_by_id,
                epochs=2,
                selector_expectation=runner._selector_expectation(payload, seed),
            )
        record = result["record"]
        per_epoch = record["channel_diagnostics"]["training"]["per_epoch"]
        assert record["status"] == "ok"
        assert record["representation_epochs_completed"] == 2
        assert per_epoch[0]["channel_counts"] == observations[str(seed)]["channel_counts"]
        assert per_epoch[0]["targets"] == observations[str(seed)]["targets"]

    wrong_seed = MODEL_SEEDS[1]
    wrong_dir = tmp_path / f"seed-{wrong_seed}-wrong"
    wrong_dir.mkdir()
    with (tmp_path / "progress-wrong.jsonl").open("w", encoding="utf-8") as progress:
        with pytest.raises(SystemExit, match="epoch-1 channel counts differ from this seed's"):
            runner._run_seed(
                binding=binding,
                worktree=worktree,
                seed=wrong_seed,
                fit_ids=fit_ids,
                fit_sample_by_id=fit_sample_by_id,
                fit_history_ids=fit_history_ids,
                calibration_rows=calibration_rows,
                calibration_rows_by_history=calibration_rows_by_history,
                development_history_ids=list(DEVELOPMENT_HISTORIES),
                device=torch.device("cpu"),
                progress=progress,
                seed_dir=wrong_dir,
                fit_rows_by_id=fit_rows_by_id,
                epochs=2,
                selector_expectation=observations[str(MODEL_SEEDS[0])],
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for N3 paired-B0 device-placement proof",
)
def test_n3_compute_block_profiles_paired_b0_on_the_requested_cuda_device(tmp_path) -> None:
    """The real compute block must forward its paired B0 counter on the N3 model's device."""
    binding, histories = _tiny_binding(tmp_path / "fixture")
    fit_rows = _rows(binding, histories, "FIT")
    patchifier = runner._n3_patchifier()
    device = torch.device("cuda:0")
    model = V1RepresentationModel(runner._n3_config(SEED), patchifier=patchifier).to(device)
    payload = runner._compute_block(
        binding=binding,
        arm=_arm("locked"),
        model=model,
        patchifier=patchifier,
        fit_ids=sorted(row.file_id for row in fit_rows),
        fit_sample_by_id={row.file_id: row.sample for row in fit_rows},
        fit_history_ids=list(FIT_HISTORIES),
        device=device,
        profile_epoch=1,
    )
    assert payload["envelope_pass"] is True
    for key in ("inference_flops_per_fit_file_by_branch", "b0_inference_flops_per_fit_file_by_branch"):
        assert set(payload[key]) == {"S_pred", "S_pop"}
        assert all(isinstance(value, int) and value > 0 for value in payload[key].values())
