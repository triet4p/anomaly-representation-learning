"""Focused Task 23 contract tests: B0-FULL runner Development loader + tiny E2E proof.

Covers consumer-visible behavior only: the runner's Development loader returns
the full frozen Development support (normal + abnormal rows) under the frozen
manifest/shard checks with the DEVELOPMENT score_development permission, and a
tiny real-schema fixture runs the actual public path end to end (2 genuine
schedule updates -> coherent save -> strict restore -> Fit bank refit ->
Calibration q95 -> runner Development scoring -> prediction-cache build with
serialization readback + consumer-compatibility + patch mapping).
Software-only fixture: tiny CPU molecules of the production contract shapes,
NOT certified DATA, pilot, capacity, or quality evidence. The normative
production contract (100 epochs, full support, 3 seeds) is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from representation.config import V1Config
from representation.crn import n4_patch_residual
from representation.data import collate_variable_files
from representation.model import V1RepresentationModel
from representation.sprint18_checkpoint import (
    coefficient_for_epoch,
    learning_rate_for_epoch,
    restore_sprint18_checkpoint,
    save_sprint18_checkpoint,
    state_sha256,
)
from representation.sprint18_harness import (
    MODEL_SEEDS,
    BoundHistory,
    Task6Binding,
    fit_calibration_q95,
    fit_fit_reference_bank,
    load_certified_sample_subset,
    make_common_support,
    make_sprint18_arm_config,
    score_certified_rows,
    validate_arm_config,
    validate_score_maps,
)
from representation.trainer import RepresentationTrainer
from synth.config import PatchConfig
from synth.dataset import _sample_bytes
from synth.patchify import Patchifier
from synth.schema import AnomalyFamily, AnomalyMeta, FileSample, SampleLabel

import sprint18_task23_b0_runner as runner

SEED = MODEL_SEEDS[0]
COEFFS = {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None}
LOCK = "6eb108b2073181d427f1aedfc81fb3bae4c29595108d15de486082800bdd9d6e"


def _tiny_config(seed: int = SEED, contrastive_max: float = 0.05) -> V1Config:
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
        contrastive_weight_max=contrastive_max,
        ema_decay=0.996,
    )


@dataclass(frozen=True)
class _TinyHistory:
    history: BoundHistory
    manifest_path: Path
    shard_paths: dict[str, Path]
    all_ids: tuple[str, ...]


def _make_sample(file_id: str, index: int, *, abnormal: bool, length: int) -> FileSample:
    rng = np.random.default_rng(94100 + index)
    signal = rng.normal(size=(6, length)).astype(np.float32)
    label = SampleLabel.ABNORMAL if abnormal else SampleLabel.NORMAL
    item = FileSample(
        x=signal,
        file_id=file_id,
        file_label=label,
        seed=94100 + index,
        generator_version="s18-t23-c01-test",
        config_hash="s18-t23-c01-test",
        regime_sequence=[],
        robot_idx=index % 6,
        program_idx=index % 2,
    )
    if abnormal:
        item.anomaly_meta = AnomalyMeta(
            family=AnomalyFamily.EASY_SPIKE, start=16, end=32,
            severity=0.5, affected_channels=[0],
        )
        mask = np.zeros((6, length), dtype=bool)
        mask[0, 16:32] = True
        item.anomaly_mask = mask
    item.validate()
    return item


def _bind_tiny(root: Path) -> tuple[Task6Binding, dict[str, _TinyHistory]]:
    """Bind software-only Fit/Calibration/Development histories (local bytes).

    Fit carries one history with one file (the full frozen support for this
    scope); Calibration carries one history with one file; Development
    carries two histories with the shared dense temporal scaffold below, so
    the frozen probe15 descriptive metrics stay computable (mixed normal and
    abnormal files, positives and controls): no silent sparse-metric gaps.
    """
    specs = (
        ("T23-TINY-FIT-01", "FIT", ("normal",)),
        ("T23-TINY-CAL-01", "CALIBRATION", ("normal",)),
        ("T23-TINY-DEV-01", "DEVELOPMENT", ("dense",)),
        ("T23-TINY-DEV-02", "DEVELOPMENT", ("dense",)),
    )
    catalog: dict[str, _TinyHistory] = {}
    counter = 0
    prebuilt: dict[str, tuple[list[FileSample], list[dict]]] = {}
    for history_id, role, kind in specs:
        members: list[FileSample] = []
        manifest_rows: list[dict] | None = None
        if kind == ("dense",):
            members, manifest_rows = _dense_development_files(history_id, counter)
            counter += len(members)
        else:
            members = [_make_sample(f"t23-{history_id.lower()}-00", counter,
                                    abnormal=False, length=256)]
            counter += 1
        prebuilt[history_id] = (members, manifest_rows)
        relative = f"{history_id}/shard-00000.zip"
        shard_path = root / relative
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(shard_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in members:
                archive.writestr(f"{item.file_id}.npz", _sample_bytes(item))
        cached = prebuilt[history_id][1]
        if cached is not None:
            rows = [dict(r) for r in cached]
        else:
            rows = []
            for pos, item in enumerate(members):
                start = 3600.0 * pos
                end = 3600.0 * (pos + 1)
                rows.append({"file_id": item.file_id,
                             "file_label": item.file_label.value,
                             "robot_id": f"robot-{item.robot_idx + 1:02d}",
                             "program_id": f"program-{item.program_idx + 1:02d}",
                             "is_quarantined": False, "is_censored": False,
                             "member_views": ["test-temporal"],
                             "start_time": start, "end_time": end,
                             "n_valid_patches": 13})
        if kind == ("dense",):
            windows: dict[str, list] = {}
            failures = [
                {"failure_id": f"fail-{history_id}-P-1", "robot_id": "robot-01",
                 "failure_time": 30 * 86400.0, "cohort": "P", "subtype": "P1",
                 "degradation_onset": 20 * 86400.0,
                 "duration_d": 10.0, "severity": 2.0},
                {"failure_id": f"fail-{history_id}-W-1", "robot_id": "robot-02",
                 "failure_time": 30 * 86400.0, "cohort": "W", "subtype": "W1",
                 "degradation_onset": 20 * 86400.0,
                 "duration_d": 10.0, "severity": 2.0},
            ]
        else:
            windows = {}
            failures = []
        manifest = {
            "format": 1, "role": history_id,
            "protocol": "sprint15-benchmark-protocol-v7",
            "config_hash": "s18-t23-c01-test",
            "files": rows,
            "shards": [{"path": relative, "file_ids": [item.file_id for item in members],
                        "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest()}],
            "maintenance_windows": windows,
            "failure_events": failures,
        }
        manifest_path = root / f"{history_id}.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                                 encoding="utf-8")
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        history = BoundHistory(
            assigned_role=role,  # type: ignore[arg-type]
            actual_history_id=history_id,
            schema_history_id=f"H-TINY-{history_id}",
            manifest_sha256=digest,
            data_seed=32064,
            sample_count=len(members),
            config_hash="s18-t23-c01-test",
            relative_path=f"tiny-fixture/{role}/{history_id}",
            data_permitted_use=role,
            model_permitted_use=role,
        )
        catalog[history_id] = _TinyHistory(
            history=history, manifest_path=manifest_path,
            shard_paths={relative: shard_path},
            all_ids=tuple(item.file_id for item in members))
    return _finish_binding(catalog)


def _finish_binding(catalog: dict[str, _TinyHistory]) -> tuple[Task6Binding, dict[str, _TinyHistory]]:
    histories = tuple(entry.history for entry in catalog.values())
    fit_rows_n = sum(len(catalog[h].all_ids) for h in catalog
                     if catalog[h].history.assigned_role == "FIT")
    cal_rows_n = sum(len(catalog[h].all_ids) for h in catalog
                     if catalog[h].history.assigned_role == "CALIBRATION")
    binding = Task6Binding(
        candidate_id="S18-T23-C01-TINY",
        data_root_relative="tiny-fixture",
        handoff_sha256=hashlib.sha256(b"s18-t23-c01-handoff").hexdigest(),
        data_binding_raw_sha256=hashlib.sha256(b"s18-t23-c01-raw").hexdigest(),
        data_binding_canonical_sha256=hashlib.sha256(b"s18-t23-c01-canonical").hexdigest(),
        whole16_certification_sha256=hashlib.sha256(b"s18-t23-c01-whole16").hexdigest(),
        qualification_record_sha256=hashlib.sha256(b"s18-t23-c01-qual").hexdigest(),
        manifest_aggregate_sha256=hashlib.sha256(
            "".join(entry.history.manifest_sha256 for entry in catalog.values()).encode("ascii")
        ).hexdigest(),
        protocol_sha256=hashlib.sha256(b"s18-t23-c01-protocol").hexdigest(),
        mechanism_spec_sha256={},
        histories=histories,
        fit_healthy_rows=fit_rows_n,
        calibration_healthy_rows=cal_rows_n,
        executable_binding_sha256=hashlib.sha256(
            "".join(entry.history.manifest_sha256 for entry in catalog.values()).encode("ascii")
        ).hexdigest(),
    )
    return binding, catalog


def _dense_development_files(history_id: str, start_index: int) -> tuple[list[FileSample], list[dict]]:
    """Forty daily files over two robots with mixed normal/abnormal labels.

    Robot-01 days 23-29 (inside the P failure horizon [23d, 30d]) are
    abnormal; robot-02 days 23-29 (inside the W horizon) are abnormal; all
    other days are normal. Both histories share the scaffold with distinct
    file ids so scores never alias across histories.
    """
    samples: list[FileSample] = []
    rows: list[dict] = []
    for day in range(1, 41):
        for robot in (0, 1):
            abnormal = 23 <= day <= 29
            file_id = f"t23-{history_id.lower()}-r{robot + 1:02d}-d{day:02d}"
            length = 224 + (day % 3) * 32
            item = _make_sample(file_id, start_index + len(samples), abnormal=abnormal,
                                length=length)
            item.robot_idx = robot
            item.program_idx = (day + robot) % 2
            item.validate()
            samples.append(item)
            rows.append({"file_id": item.file_id, "file_label": item.file_label.value,
                         "robot_id": f"robot-{robot + 1:02d}",
                         "program_id": f"program-{item.program_idx + 1:02d}",
                         "is_quarantined": False, "is_censored": False,
                         "member_views": ["test-temporal"],
                         "start_time": float((day - 1) * 86400),
                         "end_time": float(day * 86400),
                         "n_valid_patches": 13})
    return samples, rows


def _epoch_batches(binding, arm, epoch, config, patchifier, samples, batch_size: int):
    batches = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        batches.append(collate_variable_files(chunk, patchifier, masking_config=config,
                                              masking_seed=1000 + epoch * 10 + start))
    return batches


def test_runner_development_loader_returns_full_mixed_support(tmp_path) -> None:
    """Runner Development loader keeps abnormal rows the healthy loader refuses."""
    binding, catalog = _bind_tiny(tmp_path / "dev-loader")
    dev = catalog["T23-TINY-DEV-01"]
    loaded = runner._load_development_rows(
        binding=binding, history_id=dev.history.actual_history_id,
        manifest_path=dev.manifest_path, shard_paths=dev.shard_paths,
        file_ids=list(dev.all_ids),
    )
    labels = {row.sample.file_label for row in loaded}
    assert SampleLabel.ABNORMAL in labels and SampleLabel.NORMAL in labels
    assert all(row.actual_history_id == dev.history.actual_history_id for row in loaded)
    assert all(row.manifest_sha256 == dev.history.manifest_sha256 for row in loaded)
    # The healthy-role permission for a Development history refuses (wrong role).
    with pytest.raises(ValueError, match="not authorized"):
        load_certified_sample_subset(
            binding, dev.history.actual_history_id, dev.manifest_path, dev.shard_paths,
            list(dev.all_ids), role_action="calibrate_q95", verify_all_shards=True,
        )


def test_tiny_b0_chain_save_restore_bank_q95_development_cache(tmp_path) -> None:
    """Tiny E2E through public consumers plus the actual runner Development/cache helpers.

    The train/save/restore/bank/q95/cache path runs at smoke scope while the
    runner Development helper runs at experiment scope against the same tiny
    software-only binding; the production runner always uses experiment scope
    for every stage.
    """
    _run_tiny_chain(tmp_path, scope="task6_smoke_only", development_scope="experiment")


def _run_tiny_chain(root: Path, *, scope: str, development_scope: str | None = None):
    binding, catalog = _bind_tiny(root / "chain")
    fit_entry = catalog["T23-TINY-FIT-01"]
    cal_entry = catalog["T23-TINY-CAL-01"]
    dev_entry = catalog["T23-TINY-DEV-01"]
    fit_rows = list(load_certified_sample_subset(
        binding, fit_entry.history.actual_history_id, fit_entry.manifest_path,
        fit_entry.shard_paths, list(fit_entry.all_ids),
        role_action="train_fit", verify_all_shards=True))
    cal_rows = list(load_certified_sample_subset(
        binding, cal_entry.history.actual_history_id, cal_entry.manifest_path,
        cal_entry.shard_paths, list(cal_entry.all_ids),
        role_action="calibrate_q95", verify_all_shards=True))
    assert len(fit_rows) == 1 and len(cal_rows) == 1  # single-file Fit/Cal frozen support

    locked = (scope == "experiment") or (development_scope == "experiment")
    arm_scope = "locked" if locked else scope
    eff_scope = "experiment" if locked else scope
    config = _tiny_config()
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    if locked:
        arm = make_sprint18_arm_config("B0-FULL", SEED, dict(COEFFS),
                                       selection_state="locked",
                                       coefficient_lock_sha256=LOCK)
    else:
        arm = make_sprint18_arm_config("B0-FULL", SEED, dict(COEFFS),
                                       selection_state=scope)
    model = V1RepresentationModel(config, patchifier=patchifier)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate_for_epoch(1), weight_decay=0.0)
    trainer = RepresentationTrainer(model, optimizer=optimizer, max_grad_norm=1.0,
                                    device=torch.device("cpu"), seed=SEED)
    init_sha = state_sha256(model.state_dict())
    fit_samples = [row.sample for row in fit_rows]
    batch_one = collate_variable_files(fit_samples, patchifier, masking_config=config,
                                       masking_seed=1001)
    metrics_1 = trainer.train_sprint18_epoch([batch_one], arm=arm, epoch=1)
    assert trainer.step == 1
    balance_1 = trainer.objective_balance_history[-1]
    assert balance_1["effective_coefficient_L_info"] == 0.0
    assert balance_1["lr"] == learning_rate_for_epoch(1)
    batch_31 = collate_variable_files(fit_samples, patchifier, masking_config=config,
                                      masking_seed=1031)
    metrics_31 = trainer.train_sprint18_epoch([batch_31], arm=arm, epoch=31)
    assert trainer.step == 2
    balance_31 = trainer.objective_balance_history[-1]
    assert balance_31["effective_coefficient_L_info"] == coefficient_for_epoch(0.05, 31) == 0.05
    assert balance_31["lr"] == learning_rate_for_epoch(31)
    assert math.isfinite(metrics_1["joint_loss"]) and math.isfinite(metrics_31["joint_loss"])

    fit_ids = [row.file_id for row in fit_rows]
    ordered_ids = sorted(fit_ids)
    from representation.sprint18_harness import _canonical_sha256, deterministic_sampler_order
    sampler_epoch = 1
    smoke_seed, smoke_order = deterministic_sampler_order(
        binding, arm, workload="representation", scope=eff_scope,
        role_history_ids=[fit_entry.history.actual_history_id], epoch=sampler_epoch,
        eligible_source_ids=fit_ids)
    assert set(smoke_order) == set(fit_ids)
    ckpt_path = root / "t23-c01-chain.pt"
    lambda_at_1 = coefficient_for_epoch(0.05, 1)
    sampler_state = {"scope": eff_scope, "workload": "representation",
                     "role_history_ids": [fit_entry.history.actual_history_id],
                     "role_manifest_sha256": [fit_entry.history.manifest_sha256],
                     "eligible_source_ids": ordered_ids,
                     "eligible_source_ids_sha256": _canonical_sha256(ordered_ids),
                     "sampler_epoch_seed": smoke_seed, "epoch_order_ids": list(smoke_order),
                     "epoch_order_ids_sha256": _canonical_sha256(list(smoke_order)),
                     "eligible_rows": len(ordered_ids), "batch_size": 128,
                     "batches_per_epoch": 1, "completed_epochs": 0, "current_epoch": 1,
                     "batches_in_current_epoch": 0,
                     "fit_role_sha256": _canonical_sha256(
                         sorted(item.manifest_sha256 for item in binding.role_histories("FIT")))}
    ckpt_kwargs = dict(global_step=0, completed_epoch=0)
    sched = {"scheduler_id": "sprint18-epoch-lr-v2", "completed_epochs": 0,
             "active_epoch": 1, "learning_rate": learning_rate_for_epoch(1),
             "global_step": 0}
    coeff = {"epoch": 1, "global_step": 0, "selection_state": arm_scope,
             "lambda_info": lambda_at_1,
             "lambda_hcc": None, "alpha": None, "beta": None}
    fresh_model = V1RepresentationModel(_tiny_config(), patchifier=patchifier)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(),
                                  lr=learning_rate_for_epoch(1), weight_decay=0.0)
    fresh_init = state_sha256(fresh_model.state_dict())
    digest = save_sprint18_checkpoint(
        ckpt_path, fresh_model, fresh_opt, arm=arm, binding=binding,
        initialization_sha256=fresh_init,
        scheduler_state=sched,
        sampler_state=sampler_state,
        coefficient_state=coeff,
        scope=eff_scope, **ckpt_kwargs)
    live_sha = state_sha256(fresh_model.state_dict())
    live_ema = state_sha256(fresh_model.target_encoder.state_dict())
    restored_model = V1RepresentationModel(config, patchifier=patchifier)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(),
                                           lr=learning_rate_for_epoch(1), weight_decay=0.0)
    restored = restore_sprint18_checkpoint(ckpt_path, restored_model, restored_optimizer,
                                           arm=arm, binding=binding,
                                           allowed_scope=eff_scope)
    assert restored.checkpoint_sha256 == digest
    assert state_sha256(restored_model.state_dict()) == live_sha
    assert state_sha256(restored_model.target_encoder.state_dict()) == live_ema
    # Resumed step on the restored graph proves the restored bytes are trainable.
    resumed = collate_variable_files(fit_samples[:2], patchifier, masking_config=config,
                                     masking_seed=2003)
    before = state_sha256(restored_model.state_dict())
    trainer2 = RepresentationTrainer(restored_model, optimizer=restored_optimizer,
                                     max_grad_norm=1.0, device=torch.device("cpu"), seed=SEED)
    dev_fit_probe = collate_variable_files(fit_samples[:2], patchifier,
                                           masking_config=config,
                                           masking_seed=2004)
    trainer2.train_sprint18_epoch([dev_fit_probe], arm=arm, epoch=4)
    assert state_sha256(restored_model.state_dict()) != before
    bank_model = V1RepresentationModel(config, patchifier=patchifier)
    bank_opt = torch.optim.AdamW(bank_model.parameters(),
                                 lr=learning_rate_for_epoch(1), weight_decay=0.0)
    bank_restored = restore_sprint18_checkpoint(ckpt_path, bank_model, bank_opt,
                                                arm=arm, binding=binding,
                                                allowed_scope=eff_scope)
    bank = fit_fit_reference_bank(binding, arm, bank_model, fit_rows, patchifier,
                                  checkpoint=bank_restored, scope=eff_scope)
    restored_model = bank_model
    restored_optimizer = bank_opt
    restored = bank_restored
    assert bank.provenance.source_rows == len(fit_rows)
    assert bank.provenance.scope == eff_scope
    scored = score_certified_rows(binding, arm, restored_model, bank, cal_rows, patchifier,
                                  checkpoint=restored, action="calibrate_q95",
                                  scope=eff_scope)
    validate_score_maps(scored.support, scored.scores)
    calibration = fit_calibration_q95(binding, scored, cal_rows, scope=eff_scope)
    assert math.isfinite(calibration.S_pred.threshold)
    assert math.isfinite(calibration.S_pop.threshold)
    if development_scope is None:
        return

    # The actual runner Development helper on the tiny mixed fixture.
    development_ids = {h: list(catalog[h].all_ids) for h in ("T23-TINY-DEV-01", "T23-TINY-DEV-02")}
    worktree = root / "worktree-roles"
    for entry in catalog.values():
        role_dir = worktree / entry.history.relative_path
        role_dir.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(entry.manifest_path.read_text(encoding="utf-8"))
        (role_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        for relative, source in entry.shard_paths.items():
            target = role_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(source).read_bytes())
    dev_scope = development_scope or scope
    dev_arm = arm
    dev_checkpoint = restored
    dev_bank = bank
    dev_model = restored_model
    if dev_scope == "experiment":
        for group in restored_optimizer.param_groups:
            group["lr"] = learning_rate_for_epoch(1)
        dev_arm = make_sprint18_arm_config("B0-FULL", SEED, dict(COEFFS),
                                           selection_state="locked",
                                           coefficient_lock_sha256=LOCK)
        exp_seed, exp_order = deterministic_sampler_order(
            binding, dev_arm, workload="representation", scope="experiment",
            role_history_ids=[entry.history.actual_history_id
                              for entry in catalog.values()
                              if entry.history.assigned_role == "FIT"],
            epoch=1, eligible_source_ids=fit_ids)
        exp_ckpt_path = root / "t23-c01-chain-exp.pt"
        exp_model = V1RepresentationModel(config, patchifier=patchifier)
        exp_opt = torch.optim.AdamW(exp_model.parameters(),
                                    lr=learning_rate_for_epoch(1),
                                    weight_decay=0.0)
        exp_init = state_sha256(exp_model.state_dict())
        exp_digest = save_sprint18_checkpoint(
            exp_ckpt_path, exp_model, exp_opt, arm=dev_arm,
            binding=binding, global_step=0, completed_epoch=0,
            initialization_sha256=exp_init,
            scheduler_state={"scheduler_id": "sprint18-epoch-lr-v2",
                             "completed_epochs": 0, "active_epoch": 1,
                             "learning_rate": learning_rate_for_epoch(1),
                             "global_step": 0},
            sampler_state={"scope": "experiment", "workload": "representation",
                           "role_history_ids": [entry.history.actual_history_id
                                                for entry in catalog.values()
                                                if entry.history.assigned_role == "FIT"],
                           "role_manifest_sha256": [entry.history.manifest_sha256
                                                    for entry in catalog.values()
                                                    if entry.history.assigned_role == "FIT"],
                           "eligible_source_ids": ordered_ids,
                           "eligible_source_ids_sha256": _canonical_sha256(ordered_ids),
                           "sampler_epoch_seed": exp_seed,
                           "epoch_order_ids": list(exp_order),
                           "epoch_order_ids_sha256": _canonical_sha256(list(exp_order)),
                           "eligible_rows": len(ordered_ids), "batch_size": 128,
                           "batches_per_epoch": 1, "completed_epochs": 0,
                           "current_epoch": 1, "batches_in_current_epoch": 0,
                           "fit_role_sha256": _canonical_sha256(
                               sorted(item.manifest_sha256
                                      for item in binding.role_histories("FIT")))},
            coefficient_state={"epoch": 1, "global_step": 0,
                               "selection_state": "locked",
                               "lambda_info": coefficient_for_epoch(0.05, 1),
                               "lambda_hcc": None, "alpha": None, "beta": None},
            scope="experiment")
        assert exp_digest and len(exp_digest) == 64
        exp_restored = V1RepresentationModel(config, patchifier=patchifier)
        exp_restored_opt = torch.optim.AdamW(exp_restored.parameters(),
                                             lr=learning_rate_for_epoch(1),
                                             weight_decay=0.0)
        dev_checkpoint = restore_sprint18_checkpoint(
            exp_ckpt_path, exp_restored, exp_restored_opt, arm=dev_arm,
            binding=binding, allowed_scope="experiment")
        dev_model = exp_restored
        dev_bank = fit_fit_reference_bank(binding, dev_arm, exp_restored, fit_rows,
                                          patchifier, checkpoint=dev_checkpoint,
                                          scope="experiment")
    development = runner._score_development_descriptive(
        binding=binding, arm=dev_arm, model=dev_model, bank=dev_bank,
        patchifier=patchifier,
        checkpoint=dev_checkpoint, worktree=worktree,
        development_history_ids=["T23-TINY-DEV-01", "T23-TINY-DEV-02"],
        thresholds={"S_pred": calibration.S_pred.threshold,
                    "S_pop": calibration.S_pop.threshold}, scope=dev_scope)
    assert set(development["branches"]) == {"S_pred", "S_pop"}
    assert development["support"]["evaluation_histories"] == ["T23-TINY-DEV-01", "T23-TINY-DEV-02"]
    per_history = development["branches"]["S_pred"]["per_history"]
    assert len(per_history) == 2
    per_history = per_history[0]
    assert per_history["history_id"] == dev_entry.history.actual_history_id
    assert per_history["support_sha256"] == development["support"]["per_history"][
        dev_entry.history.actual_history_id]["support_sha256"]

    cache = runner._build_prediction_cache(
        binding=binding, arm=arm, model=restored_model, patchifier=patchifier, config=config,
        fit_rows_ordered=fit_rows, calibration_rows=cal_rows,
        calibration_history_ids=[cal_entry.history.actual_history_id],
        development_ids=development_ids,
        development_history_ids=["T23-TINY-DEV-01", "T23-TINY-DEV-02"])
    assert len(cache["records"]) == len(fit_rows) + len(cal_rows)
    first = cache["records"][0]
    for key in ("file_id", "actual_history_id", "manifest_sha256", "shard_sha256",
                "role", "mask_seed", "patch_mapping", "context_latents", "target_latents",
                "predicted_latents", "patch_residuals", "file_mean_raw_residual"):
        assert key in first
    mapping = first["patch_mapping"]
    assert set(mapping) == {"starts", "valid_len", "patch_valid_mask", "eligible_patch_index"}
    cache_path = root / "prediction_cache.json"
    cache_path.write_text(json.dumps(cache, sort_keys=True) + "\n", encoding="utf-8")
    reloaded = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(reloaded["records"]) == len(cache["records"])
    support = make_common_support(binding, first["actual_history_id"], [first["file_id"]])
    assert first["file_id"] in support.file_ids
    residuals = n4_patch_residual(
        torch.tensor(first["target_latents"]), torch.tensor(first["predicted_latents"]))
    assert residuals.ndim == 1 and torch.isfinite(residuals).all()
