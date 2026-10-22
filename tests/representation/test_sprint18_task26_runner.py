"""Focused Task 26 K9-FULL tests: Fit-enforced downstream path and runner wiring.

Covers consumer-visible behavior only: the K9 downstream members are fitted
on verified-healthy Fit rows of the K9 arm's own final state, scored as two
independent finite branches on the exact declared support, refused for any
other arm/role/checkpoint, persisted without refit, and reachable through the
new K9 runner's locked membership and paired-Development fail-closed checks.
No wiring/source-text/incidental assertions.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "representation"))

from representation.config import V1Config
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
    fit_calibration_q95,
    load_certified_sample_subset,
    make_sprint18_arm_config,
    validate_score_maps,
)
from representation.sprint18_k9_downstream import (
    fit_k9_downstream_members,
    load_k9_downstream,
    save_k9_downstream,
    score_k9_certified_rows,
)
from representation.sprint18_k_dispatch import (
    build_k_config,
    build_k_criterion,
    build_k_model,
    masking_policy_for_arm,
    members_for_arm,
)
from representation.trainer import RepresentationTrainer
from synth.dataset import _sample_bytes
from synth.schema import FileSample, SampleLabel

import sprint18_task26_k9_runner as runner

SEED = MODEL_SEEDS[0]
COEFFS = {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None}
LOCK = "6eb108b2073181d427f1aedfc81fb3bae4c29595108d15de486082800bdd9d6e"
HISTORIES = [f"S18I-ITER-0005-DEVELOPMENT-0{index}" for index in range(1, 5)]


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
        contrastive_weight_max=0.05,
        ema_decay=0.996,
    )


def _tiny_binding(root: Path) -> tuple[Task6Binding, dict[str, _TinyHistory]]:
    """Bind tiny FIT/CALIBRATION/DEVELOPMENT histories for the K9 path.

    Local proxy for the certified DATA contract: the manifest/shard bytes
    carry the same observable eligibility fields as the certified roster and
    the binding digest covers every bound history. The Fit support carries
    more than the frozen C7-A minimum cell rows so the frozen reference
    hierarchy is exercised instead of a degenerate fallback.
    """
    generator = "s18-t26-k9-fixture"
    config_hash = "s18-t26-k9-config"
    plan = (("FIT", "S18-T26-TINY-FIT-01", 12), ("CALIBRATION", "S18-T26-TINY-CAL-01", 3),
            ("DEVELOPMENT", "S18-T26-TINY-DEV-01", 4))
    rng = np.random.default_rng(92601)
    total = sum(count for _, _, count in plan)
    signals = {
        f"k26-tiny-{index:02d}": rng.normal(size=(6, 256 + 32 * index)).astype(np.float32)
        for index in range(total)
    }

    def make_sample(file_id: str, index: int) -> FileSample:
        sample = FileSample(
            x=signals[file_id],
            file_id=file_id,
            file_label=SampleLabel.NORMAL,
            seed=SEED + index,
            generator_version=generator,
            config_hash=config_hash,
            regime_sequence=[],
            robot_idx=index % 3,
            program_idx=index % 2,
        )
        sample.validate()
        return sample

    histories: dict[str, _TinyHistory] = {}
    cursor = 0
    for role, history_id, count in plan:
        relative = f"{history_id}/shard-00000.zip"
        shard_path = root / relative
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        file_ids = [f"k26-tiny-{cursor + offset:02d}" for offset in range(count)]
        items = [make_sample(file_id, cursor + offset) for offset, file_id in enumerate(file_ids)]
        with zipfile.ZipFile(shard_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in items:
                archive.writestr(f"{item.file_id}.npz", _sample_bytes(item))
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
                    "end_time": 3600.0,
                    "n_valid_patches": 13,
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
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        histories[history_id] = _TinyHistory(
            history=BoundHistory(
                assigned_role=role,  # type: ignore[arg-type]
                actual_history_id=history_id,
                schema_history_id=f"H-TINY-{history_id}",
                manifest_sha256=digest,
                data_seed=32601,
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
        candidate_id="S18-T26-K9-TINY",
        data_root_relative="tiny-fixture",
        handoff_sha256=hashlib.sha256(b"s18-t26-k9-handoff").hexdigest(),
        data_binding_raw_sha256=hashlib.sha256(b"s18-t26-k9-raw").hexdigest(),
        data_binding_canonical_sha256=hashlib.sha256(b"s18-t26-k9-canonical").hexdigest(),
        whole16_certification_sha256=hashlib.sha256(b"s18-t26-k9-whole16").hexdigest(),
        qualification_record_sha256=hashlib.sha256(b"s18-t26-k9-qual").hexdigest(),
        manifest_aggregate_sha256=hashlib.sha256(digest_source.encode("ascii")).hexdigest(),
        protocol_sha256=hashlib.sha256(b"s18-t26-k9-protocol").hexdigest(),
        mechanism_spec_sha256={},
        histories=tuple(item.history for item in histories.values()),
        fit_healthy_rows=12,
        calibration_healthy_rows=3,
        executable_binding_sha256=hashlib.sha256(digest_source.encode("ascii")).hexdigest(),
    )
    return binding, histories


def _rows(binding: Task6Binding, histories: dict[str, _TinyHistory], role: str, action: str):
    item = next(value for value in histories.values() if value.history.assigned_role == role)
    return list(
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


def _trained_k9_checkpoint(tmp_path: Path, binding: Task6Binding, fit_rows, seed: int = SEED):
    """Run real K9 optimizer steps, save a coherent checkpoint and strict-restore it."""
    config = _tiny_config(seed)
    model, patchifier = build_k_model("K9-FULL", config)
    criterion = build_k_criterion("K9-FULL")
    arm = make_sprint18_arm_config(
        "K9-FULL", seed, dict(COEFFS), selection_state="task6_smoke_only"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0)
    trainer = RepresentationTrainer(
        model, criterion=criterion, optimizer=optimizer, max_grad_norm=1.0, seed=seed
    )
    from representation.data import collate_variable_files

    policy = masking_policy_for_arm("K9-FULL")
    batch = collate_variable_files(
        [row.sample for row in fit_rows],
        patchifier,
        masking_config=config,
        masking_seed=1,
        masking_policy=policy,
    )
    before = state_sha256(model.state_dict())
    trainer.train_sprint18_epoch([batch], arm=arm, epoch=1, collect_balance=True)
    after = state_sha256(model.state_dict())
    assert trainer.step == 1
    assert before != after, "the K9 update must change the trained model bytes"

    source_ids = sorted(row.file_id for row in fit_rows)
    fit_history = fit_rows[0].actual_history_id
    from representation.sprint18_harness import deterministic_sampler_order

    sampler_seed, epoch_order = deterministic_sampler_order(
        binding,
        arm,
        workload="representation",
        scope="task6_smoke_only",
        role_history_ids=[fit_history],
        epoch=2,
        eligible_source_ids=source_ids,
    )
    path = tmp_path / "tiny-k9.pt"
    for group in optimizer.param_groups:
        group["lr"] = learning_rate_for_epoch(2)
    digest = save_sprint18_checkpoint(
        path,
        model,
        optimizer,
        arm=arm,
        binding=binding,
        global_step=trainer.step,
        completed_epoch=1,
        initialization_sha256=state_sha256(model.state_dict()),
        scheduler_state={
            "scheduler_id": SCHEDULER_ID,
            "completed_epochs": 1,
            "active_epoch": 2,
            "learning_rate": learning_rate_for_epoch(2),
            "global_step": trainer.step,
        },
        sampler_state={
            "scope": "task6_smoke_only",
            "workload": "representation",
            "role_history_ids": [fit_history],
            "role_manifest_sha256": [binding.history(fit_history).manifest_sha256],
            "eligible_source_ids": source_ids,
            "eligible_source_ids_sha256": _canonical_sha256(source_ids),
            "sampler_epoch_seed": sampler_seed,
            "epoch_order_ids": list(epoch_order),
            "epoch_order_ids_sha256": _canonical_sha256(list(epoch_order)),
            "eligible_rows": len(source_ids),
            "batch_size": len(source_ids),
            "batches_per_epoch": 1,
            "completed_epochs": 1,
            "current_epoch": 2,
            "batches_in_current_epoch": 0,
            "fit_role_sha256": _canonical_sha256(
                sorted(item.manifest_sha256 for item in binding.role_histories("FIT"))
            ),
        },
        coefficient_state={
            "epoch": 2,
            "global_step": trainer.step,
            "selection_state": "task6_smoke_only",
            "lambda_info": coefficient_for_epoch(0.05, 2),
            "lambda_hcc": None,
            "alpha": None,
            "beta": None,
        },
        scope="task6_smoke_only",
    )
    restored_model, restored_patchifier = build_k_model("K9-FULL", _tiny_config(seed))
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0
    )
    restored = restore_sprint18_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        arm=arm,
        binding=binding,
        allowed_scope="task6_smoke_only",
    )
    assert restored.checkpoint_sha256 == digest
    assert state_sha256(restored_model.state_dict()) == state_sha256(model.state_dict())
    assert state_sha256(restored_model.target_encoder.state_dict()) == state_sha256(
        model.target_encoder.state_dict()
    )
    return arm, restored, restored_model, restored_patchifier


def test_k9_downstream_fit_is_fit_role_enforced(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path)
    fit_rows = _rows(binding, histories, "FIT", "train_fit")
    cal_rows = _rows(binding, histories, "CALIBRATION", "calibrate_q95")
    arm, restored, restored_model, restored_patchifier = _trained_k9_checkpoint(
        tmp_path, binding, fit_rows
    )

    downstream = fit_k9_downstream_members(
        binding,
        arm,
        restored_model,
        fit_rows,
        restored_patchifier,
        checkpoint=restored,
        scope="task6_smoke_only",
    )
    assert downstream.provenance.source_role == "FIT"
    assert downstream.provenance.source_rows == len(fit_rows)
    assert downstream.provenance.checkpoint_sha256 == restored.checkpoint_sha256
    assert downstream.provenance.standardizer_sha256 != downstream.provenance.reference_sha256

    # A Calibration history is never a valid downstream fitting source.
    with pytest.raises(ValueError, match="non-Fit history"):
        fit_k9_downstream_members(
            binding,
            arm,
            restored_model,
            cal_rows,
            restored_patchifier,
            checkpoint=restored,
            scope="task6_smoke_only",
        )

    # Another legacy-stack arm may not use the K9 downstream path.
    k7_arm = make_sprint18_arm_config(
        "K7-FULL", SEED, dict(COEFFS), selection_state="task6_smoke_only"
    )
    with pytest.raises(ValueError, match="only the K9-FULL arm"):
        fit_k9_downstream_members(
            binding,
            k7_arm,
            restored_model,
            fit_rows,
            restored_patchifier,
            checkpoint=restored,
            scope="task6_smoke_only",
        )


def test_k9_downstream_scoring_is_role_checked_and_branch_independent(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path)
    fit_rows = _rows(binding, histories, "FIT", "train_fit")
    cal_rows = _rows(binding, histories, "CALIBRATION", "calibrate_q95")
    arm, restored, restored_model, restored_patchifier = _trained_k9_checkpoint(
        tmp_path, binding, fit_rows
    )
    downstream = fit_k9_downstream_members(
        binding,
        arm,
        restored_model,
        fit_rows,
        restored_patchifier,
        checkpoint=restored,
        scope="task6_smoke_only",
    )
    scored = score_k9_certified_rows(
        binding,
        arm,
        restored_model,
        downstream,
        cal_rows,
        restored_patchifier,
        checkpoint=restored,
        action="calibrate_q95",
        scope="task6_smoke_only",
    )
    validate_score_maps(scored.support, scored.scores)
    assert set(scored.scores["S_pred"]) == {row.file_id for row in cal_rows}
    assert set(scored.scores["S_pop"]) == {row.file_id for row in cal_rows}
    assert all(
        np.isfinite(value) for value in (*scored.scores["S_pred"].values(), *scored.scores["S_pop"].values())
    )
    assert scored.scores["S_pred"] is not scored.scores["S_pop"]
    assert (
        scored.branch_provenance_sha256["S_pred"] != scored.branch_provenance_sha256["S_pop"]
    )
    calibration = fit_calibration_q95(binding, scored, cal_rows, scope="task6_smoke_only")
    assert math.isfinite(calibration.S_pred.threshold)
    assert math.isfinite(calibration.S_pop.threshold)

    # A Calibration history may not be scored under the Development permission.
    with pytest.raises(ValueError, match="not authorized"):
        score_k9_certified_rows(
            binding,
            arm,
            restored_model,
            downstream,
            cal_rows,
            restored_patchifier,
            checkpoint=restored,
            action="score_development",
            scope="task6_smoke_only",
        )


def test_k9_downstream_state_persists_without_refit(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path)
    fit_rows = _rows(binding, histories, "FIT", "train_fit")
    cal_rows = _rows(binding, histories, "CALIBRATION", "calibrate_q95")
    arm, restored, restored_model, restored_patchifier = _trained_k9_checkpoint(
        tmp_path, binding, fit_rows
    )
    downstream = fit_k9_downstream_members(
        binding,
        arm,
        restored_model,
        fit_rows,
        restored_patchifier,
        checkpoint=restored,
        scope="task6_smoke_only",
    )
    path = tmp_path / "k9-downstream.pt"
    digest = save_k9_downstream(path, downstream)
    assert len(digest) == 64
    reloaded = load_k9_downstream(path)
    assert reloaded.provenance == downstream.provenance
    assert reloaded.fit_ids == downstream.fit_ids
    scored = score_k9_certified_rows(
        binding,
        arm,
        restored_model,
        reloaded,
        cal_rows,
        restored_patchifier,
        checkpoint=restored,
        action="calibrate_q95",
        scope="task6_smoke_only",
    )
    direct = score_k9_certified_rows(
        binding,
        arm,
        restored_model,
        downstream,
        cal_rows,
        restored_patchifier,
        checkpoint=restored,
        action="calibrate_q95",
        scope="task6_smoke_only",
    )
    assert dict(scored.scores["S_pred"]) == dict(direct.scores["S_pred"])
    assert dict(scored.scores["S_pop"]) == dict(direct.scores["S_pop"])


def test_k9_downstream_scoring_refuses_foreign_checkpoint(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path)
    fit_rows = _rows(binding, histories, "FIT", "train_fit")
    cal_rows = _rows(binding, histories, "CALIBRATION", "calibrate_q95")
    arm, restored, restored_model, restored_patchifier = _trained_k9_checkpoint(
        tmp_path, binding, fit_rows
    )
    downstream = fit_k9_downstream_members(
        binding,
        arm,
        restored_model,
        fit_rows,
        restored_patchifier,
        checkpoint=restored,
        scope="task6_smoke_only",
    )
    other_arm, other_restored, other_model, other_patchifier = _trained_k9_checkpoint(
        tmp_path / "other", binding, fit_rows, seed=MODEL_SEEDS[1]
    )
    with pytest.raises(ValueError, match="restored checkpoint identity"):
        score_k9_certified_rows(
            binding,
            arm,
            restored_model,
            downstream,
            cal_rows,
            restored_patchifier,
            checkpoint=other_restored,
            action="calibrate_q95",
            scope="task6_smoke_only",
        )
    assert other_restored.checkpoint_sha256 != restored.checkpoint_sha256

    # A downstream state that does not belong to the requested arm/seed state
    # is refused even when the checkpoint itself is the requested one.
    from dataclasses import replace

    tampered = replace(
        downstream,
        provenance=replace(downstream.provenance, model_seed=MODEL_SEEDS[1]),
    )
    with pytest.raises(ValueError, match="does not match the requested"):
        score_k9_certified_rows(
            binding,
            arm,
            restored_model,
            tampered,
            cal_rows,
            restored_patchifier,
            checkpoint=restored,
            action="calibrate_q95",
            scope="task6_smoke_only",
        )


def test_k9_downstream_does_not_mutate_the_restored_model(tmp_path) -> None:
    """The downstream fit/scoring must leave the restored checkpoint identity intact."""
    binding, histories = _tiny_binding(tmp_path)
    fit_rows = _rows(binding, histories, "FIT", "train_fit")
    cal_rows = _rows(binding, histories, "CALIBRATION", "calibrate_q95")
    arm, restored, restored_model, restored_patchifier = _trained_k9_checkpoint(
        tmp_path, binding, fit_rows
    )
    before = state_sha256(restored_model.state_dict())
    downstream = fit_k9_downstream_members(
        binding,
        arm,
        restored_model,
        fit_rows,
        restored_patchifier,
        checkpoint=restored,
        scope="task6_smoke_only",
    )
    score_k9_certified_rows(
        binding,
        arm,
        restored_model,
        downstream,
        cal_rows,
        restored_patchifier,
        checkpoint=restored,
        action="calibrate_q95",
        scope="task6_smoke_only",
    )
    assert state_sha256(restored_model.state_dict()) == before
    # A second downstream fit on the same restored model is still accepted,
    # which only holds when the model state was left untouched.
    refit = fit_k9_downstream_members(
        binding,
        arm,
        restored_model,
        fit_rows,
        restored_patchifier,
        checkpoint=restored,
        scope="task6_smoke_only",
    )
    assert refit.provenance.standardizer_sha256 == downstream.provenance.standardizer_sha256
    assert refit.provenance.reference_sha256 == downstream.provenance.reference_sha256


def test_k9_downstream_latents_are_unaffected_by_the_view_generation(tmp_path) -> None:
    """Dropping the raw samples must not change any downstream latent."""
    from representation.data import collate_variable_files
    from representation.sprint18_k_dispatch import k9_forward_latents

    binding, histories = _tiny_binding(tmp_path)
    fit_rows = _rows(binding, histories, "FIT", "train_fit")
    arm, restored, restored_model, restored_patchifier = _trained_k9_checkpoint(
        tmp_path, binding, fit_rows
    )
    config = restored_model.config
    policy = masking_policy_for_arm("K9-FULL")
    items = [(row.file_id, row.sample) for row in fit_rows]
    latents = k9_forward_latents(
        restored_model, restored_patchifier, config, items
    )
    for row, latent in zip(fit_rows, latents):
        batch = collate_variable_files(
            [row.sample],
            restored_patchifier,
            masking_config=config,
            masking_seed=int.from_bytes(
                __import__("hashlib").sha256(row.file_id.encode("utf-8")).digest()[:4], "big"
            )
            % (2**31),
            masking_policy=policy,
        )
        with torch.no_grad():
            output = restored_model(
                {k: v for k, v in batch.items() if k != "file_samples"}
            )
        assert torch.equal(latent["predicted"], output["predicted_latents"][0].detach().cpu())
        assert torch.equal(latent["target"], output["target_latents"][0].detach().cpu())
        assert torch.equal(latent["file_embedding"], output["file_embedding"][0].detach().cpu())
        assert torch.equal(latent["prediction_mask"], output["prediction_mask"][0].detach().cpu())


def test_task14_lock_k9_membership_matches_frozen_dispatch() -> None:
    hashes, lock_sha256, arms = runner._task14_lock_config_hashes(REPO_ROOT)
    entry = runner._assert_locked_k9_membership(arms)
    assert entry["component_set"] == list(members_for_arm("K9-FULL"))
    for seed in MODEL_SEEDS:
        assert f"K9-FULL/seed-{seed}" in hashes
    assert lock_sha256 == "c4c1a8bde14310cf4436c10762e3e98475f7c84eb74fcb8a74b3e267885edec3"
    with pytest.raises(SystemExit, match="membership"):
        runner._assert_locked_k9_membership(
            {**arms, "K9-FULL": {**arms["K9-FULL"], "component_set": ["C2-A"]}}
        )


def test_k9_runner_batches_use_arm_patchifier_and_frozen_c5a_policy(tmp_path) -> None:
    binding, histories = _tiny_binding(tmp_path)
    fit_rows = _rows(binding, histories, "FIT", "train_fit")
    arm = make_sprint18_arm_config(
        "K9-FULL", SEED, dict(COEFFS), selection_state="locked", coefficient_lock_sha256=LOCK
    )
    config = build_k_config("K9-FULL", SEED)
    model, patchifier = build_k_model("K9-FULL", config)
    order, batches = runner._build_epoch_batches(
        binding=binding,
        arm=arm,
        epoch=1,
        config=config,
        patchifier=patchifier,
        fit_ids=sorted(row.file_id for row in fit_rows),
        fit_sample_by_id={row.file_id: row.sample for row in fit_rows},
        fit_history_ids=[fit_rows[0].actual_history_id],
    )
    assert len(order) == len(fit_rows)
    sample_by_id = {row.file_id: row.sample for row in fit_rows}
    ordered = [sample_by_id[file_id] for file_id in order]
    policy_batch = runner.collate_variable_files(
        ordered,
        patchifier,
        masking_config=config,
        masking_seed=1,
        masking_policy=masking_policy_for_arm("K9-FULL"),
    )
    default_batch = runner.collate_variable_files(
        ordered,
        patchifier,
        masking_config=config,
        masking_seed=1,
    )
    assert not torch.equal(batches[0]["mask"], default_batch["mask"])
    assert torch.equal(batches[0]["mask"], policy_batch["mask"])


def test_k9_runner_paired_development_is_fail_closed() -> None:
    def _record(arm_id: str, *, offset: float) -> dict:
        return {
            "arm_id": arm_id,
            "status": "ok",
            "global_step": 4300,
            "representation_epochs_completed": 100,
            "batches_per_epoch": 43,
            "downstream": {"source_rows": 5446},
            "bank": {"source_rows": 5446},
            "calibration": {"source_support_sha256": "a" * 64},
            "development": {
                "support": {
                    "evaluation_histories": HISTORIES,
                    "per_history": {
                        history_id: {"support_sha256": f"{index}" * 64}
                        for index, history_id in enumerate(HISTORIES)
                    },
                },
                "branches": {
                    branch: {
                        "threshold": 1.0,
                        "macro": {
                            "P_macro": 0.5 + offset,
                            "W_macro": 0.5 + offset,
                            "PW_macro": 0.5 + offset,
                            "PW_lcb95": 0.4 + offset,
                            "directional_histories": 2,
                        },
                        "per_history": [
                            {
                                "history_id": history_id,
                                "P_AUROC": 0.5 + offset,
                                "W_AUROC": 0.5 + offset,
                                "PW_AUROC": 0.5 + offset,
                                "A_AUROC": 0.5 + offset,
                                "P_recall": 0.5,
                                "W_recall": 0.5,
                                "P_median_lead_days": 1.0,
                                "W_median_lead_days": 1.0,
                                "FAR_per_robot_day": 0.1,
                            }
                            for history_id in HISTORIES
                        ],
                    }
                    for branch in ("S_pred", "S_pop")
                },
            },
        }

    arm_records = {str(seed): _record("K9-FULL", offset=0.1) for seed in MODEL_SEEDS}
    b0_records = {str(seed): _record("B0-FULL", offset=0.0) for seed in MODEL_SEEDS}
    paired = runner._paired_development(
        arm_records=arm_records,
        b0_records=b0_records,
        b0_records_sha256="b" * 64,
    )
    assert paired["arm_id"] == "K9-FULL"
    assert paired["baseline_arm_id"] == "B0-FULL"
    for branch in ("S_pred", "S_pop"):
        aggregate = paired["branches"][branch]["aggregate"]["macro"]["PW_macro"]
        assert aggregate["mean_delta"] == pytest.approx(0.1)
        assert aggregate["positive_seeds"] == 3

    # A K7 record must never be reported as the K9 arm.
    wrong = {str(seed): _record("K7-FULL", offset=0.1) for seed in MODEL_SEEDS}
    with pytest.raises(SystemExit, match="K9 seed"):
        runner._paired_development(
            arm_records=wrong, b0_records=b0_records, b0_records_sha256="b" * 64
        )

    # A different Fit source size breaks the pairing.
    mismatched = {str(seed): _record("K9-FULL", offset=0.1) for seed in MODEL_SEEDS}
    mismatched[str(MODEL_SEEDS[0])]["downstream"] = {"source_rows": 100}
    with pytest.raises(SystemExit, match="Fit source rows"):
        runner._paired_development(
            arm_records=mismatched, b0_records=b0_records, b0_records_sha256="b" * 64
        )
