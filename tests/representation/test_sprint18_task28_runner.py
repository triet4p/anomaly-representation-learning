"""Focused Task 28 N2-MSP tests: public bank/q95/Development path and coarse diagnostics.

Covers consumer-visible behavior only: the N2 arm reaches the frozen public
reference-bank, q95 and Development scoring path on its own restored primary
checkpoint; coarse-4/coarse-8 residuals are per-file diagnostics evaluated on
that file's own groups and are never fused into a primary score; the saturated
role fences still hold; and the runner's Task 14 lock binding matches the
accepted alpha. No wiring/source-text/incidental assertions.
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
    score_certified_rows,
    score_msp_coarse_rows,
    validate_score_maps,
)
from representation.sprint18_harness import _encoder_batch
from representation.trainer import RepresentationTrainer
from synth.config import PatchConfig
from synth.dataset import _sample_bytes
from synth.patchify import Patchifier
from synth.schema import FileSample, SampleLabel

import sprint18_task28_n2_runner as runner

SEED = MODEL_SEEDS[0]
LOCK = "6eb108b2073181d427f1aedfc81fb3bae4c29595108d15de486082800bdd9d6e"
FIT_HISTORIES = tuple(f"S18I-ITER-0005-FIT-TINY-0{index}" for index in range(1, 4))
CALIBRATION_HISTORY = "S18I-ITER-0005-CAL-TINY-01"
DEVELOPMENT_HISTORIES = tuple(f"S18I-ITER-0005-DEV-TINY-0{index}" for index in range(1, 5))
# 784 timesteps -> 48 fully valid patches -> N2 budget M = round(0.4*48) = 19,
# whose frozen allocation is (fine=7, coarse-4 groups=1, coarse-8 groups=1).
FILE_TIMESTEPS = 784
PATCHES_PER_FILE = 48


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
        "N2-MSP",
        SEED,
        {
            "lambda_info_max": runner.LAMBDA_INFO_MAX,
            "lambda_hcc_max": None,
            "alpha": runner.ALPHA,
            "beta": None,
        },
        selection_state=selection_state,  # type: ignore[arg-type]
        coefficient_lock_sha256=LOCK if selection_state == "locked" else None,
    )


def _tiny_binding(root: Path) -> tuple[Task6Binding, dict[str, _TinyHistory]]:
    """Bind a tiny 3-FIT/1-CALIBRATION/4-DEVELOPMENT certified-shaped roster."""
    generator = "s18-t28-n2-fixture"
    config_hash = "s18-t28-n2-config"
    plan = (
        *[(history_id, "FIT", 4) for history_id in FIT_HISTORIES],
        (CALIBRATION_HISTORY, "CALIBRATION", 3),
        *[(history_id, "DEVELOPMENT", 2) for history_id in DEVELOPMENT_HISTORIES],
    )
    rng = np.random.default_rng(92801)
    total = sum(int(item[-1]) for item in plan)
    signals = {
        f"n28-tiny-{index:02d}": rng.normal(size=(6, FILE_TIMESTEPS)).astype(np.float32)
        for index in range(total)
    }
    histories: dict[str, _TinyHistory] = {}
    cursor = 0
    for history_id, role, count in plan:
        relative = f"{history_id}/shard-00000.zip"
        shard_path = root / relative
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        file_ids = [f"n28-tiny-{cursor + offset:02d}" for offset in range(count)]
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
                # Fixed member timestamps keep the shard bytes (and therefore the
                # manifest digest and the derived N2 mask seeds) reproducible.
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
                    "end_time": float(FILE_TIMESTEPS),
                    "n_valid_patches": PATCHES_PER_FILE,
                }
                for item in items
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
                data_seed=32801,
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
        candidate_id="S18-T28-N2-TINY",
        data_root_relative="tiny-fixture",
        handoff_sha256=hashlib.sha256(b"s18-t28-n2-handoff").hexdigest(),
        data_binding_raw_sha256=hashlib.sha256(b"s18-t28-n2-raw").hexdigest(),
        data_binding_canonical_sha256=hashlib.sha256(b"s18-t28-n2-canonical").hexdigest(),
        whole16_certification_sha256=hashlib.sha256(b"s18-t28-n2-whole16").hexdigest(),
        qualification_record_sha256=hashlib.sha256(b"s18-t28-n2-qual").hexdigest(),
        manifest_aggregate_sha256=hashlib.sha256(digest_source.encode("ascii")).hexdigest(),
        protocol_sha256=hashlib.sha256(b"s18-t28-n2-protocol").hexdigest(),
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


def _trained_n2_checkpoint(tmp_path: Path, binding: Task6Binding, fit_rows, seed: int = SEED):
    """Run real N2 optimizer steps, save a coherent experiment checkpoint, strict-restore it."""
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
    trainer.train_sprint18_msp_epoch(
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
    path = tmp_path / "tiny-n2.pt"
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
            "alpha": runner.ALPHA,
            "beta": None,
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


def test_n2_public_bank_q95_and_development_scoring_on_restored_checkpoint(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path / "fixture")
    fit_rows = _rows(binding, histories, "FIT")
    calibration_rows = _rows(binding, histories, "CALIBRATION")
    model, patchifier, restored, arm = _trained_n2_checkpoint(tmp_path, binding, fit_rows)
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


def test_n2_coarse_diagnostics_are_per_file_and_unfused(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path / "fixture")
    fit_rows = _rows(binding, histories, "FIT")
    model, patchifier, restored, arm = _trained_n2_checkpoint(tmp_path, binding, fit_rows)
    before = state_sha256(model.state_dict())
    for history_id in DEVELOPMENT_HISTORIES:
        item = histories[history_id]
        rows = load_certified_development_support(
            binding, history_id, item.manifest_path, item.shard_paths,
        )
        scored = score_msp_coarse_rows(
            binding, arm, model, rows, patchifier, checkpoint=restored,
            action="score_development", scope="experiment", diagnostic_epoch=100,
        )
        assert scored.actual_history_id == history_id
        assert len(scored.records) == len(rows)
        assert {record.file_id for record in scored.records} == {row.file_id for row in rows}
        for record in scored.records:
            # M = min(n, round(0.40*n)) = 19 for 48 fully valid patches, and the
            # frozen allocation for M=19 is (7, 1, 1).
            assert record.n_valid_patches == PATCHES_PER_FILE
            assert record.mask_budget == 19
            assert record.allocation == (7, 1, 1)
            assert record.groups_4 >= 1 and record.groups_8 >= 1
            assert math.isfinite(record.group4_loss) and math.isfinite(record.group8_loss)
            assert record.coarse_loss == pytest.approx(
                0.5 * record.group4_loss + 0.5 * record.group8_loss
            )
    assert state_sha256(model.state_dict()) == before


def test_n2_primary_scores_ignore_coarse_diagnostics(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path / "fixture")
    fit_rows = _rows(binding, histories, "FIT")
    calibration_rows = _rows(binding, histories, "CALIBRATION")
    model, patchifier, restored, arm = _trained_n2_checkpoint(tmp_path, binding, fit_rows)
    bank = fit_fit_reference_bank(
        binding, arm, model, fit_rows, patchifier, checkpoint=restored, scope="experiment",
    )
    item = histories[DEVELOPMENT_HISTORIES[0]]
    rows = load_certified_development_support(
        binding, item.history.actual_history_id, item.manifest_path, item.shard_paths,
    )
    primary = score_certified_rows(
        binding, arm, model, bank, rows, patchifier,
        checkpoint=restored, action="score_development", scope="experiment",
    )
    score_msp_coarse_rows(
        binding, arm, model, rows, patchifier, checkpoint=restored,
        action="score_development", scope="experiment",
    )
    repeated = score_certified_rows(
        binding, arm, model, bank, rows, patchifier,
        checkpoint=restored, action="score_development", scope="experiment",
    )
    assert dict(primary.scores["S_pred"]) == dict(repeated.scores["S_pred"])
    assert dict(primary.scores["S_pop"]) == dict(repeated.scores["S_pop"])


def test_n2_fit_bank_permission_refuses_development_rows(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path / "fixture")
    fit_rows = _rows(binding, histories, "FIT")
    model, patchifier, restored, arm = _trained_n2_checkpoint(tmp_path, binding, fit_rows)
    item = histories[DEVELOPMENT_HISTORIES[0]]
    development_rows = load_certified_development_support(
        binding, item.history.actual_history_id, item.manifest_path, item.shard_paths,
    )
    with pytest.raises(ValueError, match="non-Fit history"):
        fit_fit_reference_bank(
            binding, arm, model, development_rows, patchifier,
            checkpoint=restored, scope="experiment",
        )


def test_n2_task14_lock_binds_alpha_and_seed_configs() -> None:
    hashes, _lock_sha256, arms = runner._task14_lock_config_hashes(REPO_ROOT)
    locked = runner._assert_locked_n2(arms)
    assert locked["coefficients"]["alpha"] == runner.ALPHA == 0.25
    assert locked["coefficients"]["lambda_info_max"] == runner.LAMBDA_INFO_MAX == 0.05
    arm = _arm("locked")
    from representation.sprint18_harness import arm_config_sha256

    assert arm_config_sha256(arm) == hashes["N2-MSP/seed-181801"]


def _development_probe15_manifests(binding: Task6Binding, histories: dict[str, _TinyHistory], worktree: Path) -> dict[str, bytes]:
    """Lay the roster out for the runner and give Development its frozen probe15 fields.

    The tiny Development histories need ``member_views`` on every file row and a
    P/W failure ledger for the frozen event metrics to be computable; a second,
    failure-free robot supplies the control windows.
    """
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


def test_n2_run_seed_reaches_bank_q95_and_development(tmp_path) -> None:
    """The corrected per-seed runner must reach bank/q95/Development and write its record.

    The three former ``device=`` forwardings to ``score_certified_rows`` used to
    raise ``TypeError`` after training, strict restore and the Fit bank, so no
    seed record, q95 or Development output could exist.
    """
    binding, histories = _tiny_binding(tmp_path / "fixture")
    worktree = tmp_path / "worktree"
    raw_by_history = _development_probe15_manifests(binding, histories, worktree)
    binding = _rebind_development_manifests(binding, histories, raw_by_history)
    fit_rows = _rows(binding, histories, "FIT")
    calibration_rows = _rows(binding, histories, "CALIBRATION")
    fit_ids = sorted(row.file_id for row in fit_rows)
    seed_dir = tmp_path / f"seed-{SEED}"
    seed_dir.mkdir()
    with (tmp_path / "progress.jsonl").open("w", encoding="utf-8") as progress:
        result = runner._run_seed(
            binding=binding,
            worktree=worktree,
            seed=SEED,
            fit_ids=fit_ids,
            fit_sample_by_id={row.file_id: row.sample for row in fit_rows},
            fit_history_ids=list(FIT_HISTORIES),
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
        )
    record = result["record"]
    assert json.loads((seed_dir / "seed_record.json").read_text(encoding="utf-8")) == record
    assert record["status"] == "ok"
    assert record["global_step"] == runner.EPOCHS
    assert sorted(record["checkpoints"], key=int) == ["10", "30", "60", "100"]
    assert record["primary"]["restored_forward_max_abs_diff"] == 0.0
    assert record["bank"]["source_rows"] == len(fit_ids)
    assert math.isfinite(record["calibration"]["S_pred_threshold"])
    assert math.isfinite(record["calibration"]["S_pop_threshold"])
    for branch in ("S_pred", "S_pop"):
        payload = record["development"]["branches"][branch]
        assert [row["history_id"] for row in payload["per_history"]] == list(DEVELOPMENT_HISTORIES)
    coarse = record["coarse_diagnostics"]
    assert coarse["fusion"].startswith("none")
    assert sorted(coarse["per_history"]) == sorted(DEVELOPMENT_HISTORIES)
    for payload in coarse["per_history"].values():
        assert payload["records"]
        for row in payload["records"]:
            assert row["mask_budget"] == 19
            assert row["allocation"] == [7, 1, 1]
