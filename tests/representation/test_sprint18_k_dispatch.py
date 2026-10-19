"""Focused Task 20 contract tests: K3/K7/K9 Sprint18 harness port.

Covers consumer-visible behavior only: exact membership/numerics binding,
member-graph assembly (C2/C3/C5/C6 identities), common-schedule training
(LR + InfoNCE activation + warmup zero + gradient isolation), K9 downstream
Fit-only fitting/scoring/refusals, and harness/bank/scoring dispatch for K
arms. No wiring/source-text/incidental assertions.
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

from representation.config import V1Config
from representation.data import collate_variable_files
from representation.model import V1RepresentationModel
from representation.sprint17_c2 import arm_patchifier
from representation.sprint17_c3 import ResidualDilatedPatchEncoder, arm_local_encoder
from representation.sprint17_c5 import MultiHorizonPredictionCriterion, c5a_block_policy
from representation.sprint17_c6 import C6ArmModel, MeanStdProjectionPooling
from representation.sprint17_c7 import arm_reference
from representation.sprint17_c8 import HuberStandardizer
from representation.sprint17_combinations import K_MEMBERS, MEMBER_IMPLEMENTATION
from representation.sprint18_checkpoint import (
    coefficient_for_epoch,
    learning_rate_for_epoch,
)
from representation.sprint18_harness import (
    BATCH_SIZE,
    MODEL_SEEDS,
    BoundHistory,
    CertifiedSample,
    Task6Binding,
    fit_calibration_q95,
    fit_fit_reference_bank,
    load_certified_sample_subset,
    make_sprint18_arm_config,
    score_certified_rows,
)
from representation.sprint18_k_dispatch import (
    K_LAMBDA_INFO_MAX,
    build_k_config,
    build_k_criterion,
    build_k_model,
    fit_k9_downstream,
    k9_forward_latents,
    masking_policy_for_arm,
    members_for_arm,
    score_k9_files,
)
from representation.trainer import RepresentationTrainer
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import FileSample, SampleLabel

import sprint17_task20_k as KORACLE

SEED = MODEL_SEEDS[0]
COEFFS = {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None}
LOCK = "6eb108b2073181d427f1aedfc81fb3bae4c29595108d15de486082800bdd9d6e"


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


def _tiny_samples(n: int = 6, base_length: int = 256) -> list[FileSample]:
    rng = np.random.default_rng(93020)
    out: list[FileSample] = []
    for index in range(n):
        signal = rng.normal(size=(6, base_length + 32 * index)).astype(np.float32)
        sample = FileSample(
            x=signal,
            file_id=f"k20-test-{index:02d}",
            file_label=SampleLabel.NORMAL,
            seed=SEED + index,
            generator_version="s18-t20-test",
            config_hash="s18-t20-test",
            regime_sequence=[],
            robot_idx=index % 3,
            program_idx=index % 2,
        )
        sample.validate()
        out.append(sample)
    return out


def _k_trainer(arm_id: str, seed: int = SEED) -> tuple[RepresentationTrainer, object, object, object]:
    cfg = _tiny_config(seed)
    model, patchifier = build_k_model(arm_id, cfg)
    arm = make_sprint18_arm_config(arm_id, seed, dict(COEFFS), selection_state="locked",
                                   coefficient_lock_sha256=LOCK)
    criterion = build_k_criterion(arm_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.0)
    trainer = RepresentationTrainer(model, criterion=criterion, optimizer=optimizer,
                                    max_grad_norm=1.0, seed=seed)
    return trainer, model, patchifier, arm


@dataclass(frozen=True)
class _TinyHistory:
    history: BoundHistory
    manifest_path: Path
    shard_paths: dict[str, Path]
    eligible_ids: tuple[str, ...]


def _tiny_certified_binding(root: Path) -> tuple[Task6Binding, dict[str, _TinyHistory]]:
    """Bind one tiny Fit history + one tiny Calibration history locally.

    Local proxy for the certified DATA contract: manifest/shard bytes carry
    the same observable eligibility fields (file label, quarantine/censoring,
    reserve identities, maintenance windows) as the certified roster, and the
    binding digest covers both histories. Distinct file bytes per id keep the
    consumer proof honest (no shared-embedding shortcut).
    """
    from synth.dataset import _sample_bytes

    generator = "s18-t20-bank-fixture"
    config_hash = "s18-t20-bank-config"
    rows = {"fit": 6, "cal": 3}
    rng = np.random.default_rng(93021)
    signals = {
        f"k20-bank-{index:02d}": rng.normal(
            size=(6, 256 + 32 * index)).astype(np.float32)
        for index in range(rows["fit"] + rows["cal"])
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

    def build(history_id: str, role: str, file_ids: list[str], start: int) -> _TinyHistory:
        relative = f"{history_id}/shard-00000.zip"
        shard_path = root / relative
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        items = [make_sample(file_id, start + pos) for pos, file_id in enumerate(file_ids)]
        with zipfile.ZipFile(shard_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in items:
                archive.writestr(f"{item.file_id}.npz", _sample_bytes(item))
        manifest = {
            "format": 1,
            "role": history_id,
            "protocol": "sprint15-benchmark-protocol-v7",
            "config_hash": config_hash,
            "files": [
                {"file_id": item.file_id, "file_label": item.file_label.value,
                 "robot_id": f"robot-{item.robot_idx + 1:02d}",
                 "program_id": f"program-{item.program_idx + 1:02d}",
                 "is_quarantined": False, "is_censored": False,
                 "start_time": 0.0, "end_time": 3600.0, "n_valid_patches": 13}
                for item in items
            ],
            "shards": [{"path": relative,
                        "file_ids": [item.file_id for item in items],
                        "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest()}],
            "maintenance_windows": {},
        }
        manifest_path = root / f"{history_id}.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return _TinyHistory(
            history=BoundHistory(
                assigned_role=role,  # type: ignore[arg-type]
                actual_history_id=history_id,
                schema_history_id=f"H-TINY-{history_id}",
                manifest_sha256=digest,
                data_seed=32064,
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

    fit_ids = [f"k20-bank-{index:02d}" for index in range(rows["fit"])]
    cal_ids = [f"k20-bank-{index:02d}" for index in range(rows["fit"], rows["fit"] + rows["cal"])]
    fit = build("K20-TINY-FIT-01", "FIT", fit_ids, 0)
    cal = build("K20-TINY-CAL-01", "CALIBRATION", cal_ids, rows["fit"])
    binding = Task6Binding(
        candidate_id="S18-T20-BANK-TINY",
        data_root_relative="tiny-fixture",
        handoff_sha256=hashlib.sha256(b"s18-t20-bank-handoff").hexdigest(),
        data_binding_raw_sha256=hashlib.sha256(b"s18-t20-bank-raw").hexdigest(),
        data_binding_canonical_sha256=hashlib.sha256(b"s18-t20-bank-canonical").hexdigest(),
        whole16_certification_sha256=hashlib.sha256(b"s18-t20-bank-whole16").hexdigest(),
        qualification_record_sha256=hashlib.sha256(b"s18-t20-bank-qual").hexdigest(),
        manifest_aggregate_sha256=hashlib.sha256(
            (fit.history.manifest_sha256 + cal.history.manifest_sha256).encode("ascii")).hexdigest(),
        protocol_sha256=hashlib.sha256(b"s18-t20-bank-protocol").hexdigest(),
        mechanism_spec_sha256={},
        histories=(fit.history, cal.history),
        fit_healthy_rows=len(fit.eligible_ids),
        calibration_healthy_rows=len(cal.eligible_ids),
        executable_binding_sha256=hashlib.sha256(
            (fit.history.manifest_sha256 + cal.history.manifest_sha256).encode("ascii")).hexdigest(),
    )
    return binding, {fit.history.actual_history_id: fit, cal.history.actual_history_id: cal}


def _tiny_certified_fit_rows(binding: Task6Binding, histories: dict[str, _TinyHistory]) -> list[CertifiedSample]:
    """Load the tiny Fit support through the certified subset loader."""
    item = next(value for value in histories.values() if value.history.assigned_role == "FIT")
    return list(load_certified_sample_subset(
        binding, item.history.actual_history_id, item.manifest_path, item.shard_paths,
        list(item.eligible_ids), role_action="train_fit", verify_all_shards=True))


def _tiny_certified_cal_rows(binding: Task6Binding, histories: dict[str, _TinyHistory]) -> list[CertifiedSample]:
    """Load the tiny Calibration support through the certified subset loader."""
    item = next(value for value in histories.values() if value.history.assigned_role == "CALIBRATION")
    return list(load_certified_sample_subset(
        binding, item.history.actual_history_id, item.manifest_path, item.shard_paths,
        list(item.eligible_ids), role_action="calibrate_q95", verify_all_shards=True))


def test_k_membership_matches_sprint17_lock_and_oracle() -> None:
    assert members_for_arm("K3-FULL") == ("C5-A", "C6-A") == K_MEMBERS["K3"]
    assert members_for_arm("K7-FULL") == ("C2-A", "C3-A", "C5-A", "C6-A") == K_MEMBERS["K7"]
    assert members_for_arm("K9-FULL") == (
        "C2-A", "C3-A", "C5-A", "C6-A", "C7-A", "C8-A", "C9-B") == K_MEMBERS["K9"]
    assert dict(MEMBER_IMPLEMENTATION["C5-A"]["frozen_numerics"]) == {
        "total_mask_ratio": 0.40, "n_blocks": 2, "horizons": [0, 1, 2]}
    assert dict(MEMBER_IMPLEMENTATION["C9-B"]["frozen_numerics"]) == {
        "window_duration_timesteps": 64}
    with pytest.raises(ValueError, match="unknown Sprint18 legacy-stack arm"):
        members_for_arm("K5-FULL")


def test_k_member_graph_uses_exact_sprint17_builders() -> None:
    samples = _tiny_samples()
    for s18_arm, s17_k in (("K3-FULL", "K3"), ("K7-FULL", "K7"), ("K9-FULL", "K9")):
        cfg = _tiny_config()
        model, patchifier = build_k_model(s18_arm, cfg)
        members = members_for_arm(s18_arm)
        oracle_patchifier = arm_patchifier("C2-A") if "C2-A" in members else Patchifier(
            PatchConfig(patch_size=32, stride=16, pad_end=True))
        for sample in samples:
            assert model.patchifier.patchify(sample).N == oracle_patchifier.patchify(sample).N
        if "C3-A" in members:
            assert isinstance(model.patch_encoder, ResidualDilatedPatchEncoder)
        if "C6-A" in members:
            assert isinstance(model, C6ArmModel)
            assert isinstance(model.c6_pooling, MeanStdProjectionPooling)
            assert model.c6_arm_id == "C6-A"
        assert masking_policy_for_arm(s18_arm) is KORACLE.masking_policy_for(s17_k)
        criterion = build_k_criterion(s18_arm)
        assert isinstance(criterion.prediction, MultiHorizonPredictionCriterion)
        assert tuple(criterion.prediction.horizons) == (0, 1, 2)
        assert criterion.prediction_weight == 1.0
        assert criterion.lambda_schedule.lambda_max == K_LAMBDA_INFO_MAX == 0.05


def test_k_training_applies_common_schedule_and_isolates_gradients() -> None:
    samples = _tiny_samples()
    for arm_id in ("K3-FULL", "K7-FULL", "K9-FULL"):
        trainer, model, patchifier, arm = _k_trainer(arm_id)
        cfg = model.config
        policy = masking_policy_for_arm(arm_id)
        batch31 = collate_variable_files(samples, patchifier, masking_config=cfg,
                                         masking_seed=93131, masking_policy=policy)
        trainer.train_sprint18_epoch([batch31], arm=arm, epoch=31)
        row31 = trainer.objective_balance_history[-1]
        assert row31["effective_coefficient_L_info"] == 0.05
        assert row31["lr"] == learning_rate_for_epoch(31)
        batch01 = collate_variable_files(samples, patchifier, masking_config=cfg,
                                         masking_seed=93101, masking_policy=policy)
        trainer.train_sprint18_epoch([batch01], arm=arm, epoch=1)
        row01 = trainer.objective_balance_history[-1]
        assert row01["effective_coefficient_L_info"] == 0.0
        assert row01["gradient_norm_weighted_L_info_shared"] == 0.0
        assert row01["r_aux"] == 0.0
        assert trainer.step == 2


def test_k_training_refuses_wrong_criterion_or_coefficient() -> None:
    from representation.criterion import JointRepresentationCriterion

    trainer, model, patchifier, arm = _k_trainer("K3-FULL")
    cfg = model.config
    batch = collate_variable_files(_tiny_samples(2), patchifier, masking_config=cfg,
                                   masking_seed=93201,
                                   masking_policy=masking_policy_for_arm("K3-FULL"))
    trainer.criterion = JointRepresentationCriterion()  # B0 single-horizon prediction
    with pytest.raises(ValueError, match="multi-horizon prediction criterion"):
        trainer.train_sprint18_epoch([batch], arm=arm, epoch=31)
    trainer2, _, patchifier2, _ = _k_trainer("K3-FULL")
    bad_arm = make_sprint18_arm_config(
        "K3-FULL", SEED,
        {"lambda_info_max": 0.10, "lambda_hcc_max": None, "alpha": None, "beta": None},
        selection_state="locked", coefficient_lock_sha256=LOCK)
    batch2 = collate_variable_files(_tiny_samples(2), patchifier2, masking_config=cfg,
                                    masking_seed=93202,
                                    masking_policy=masking_policy_for_arm("K3-FULL"))
    with pytest.raises(ValueError, match="frozen Task14 lock"):
        trainer2.train_sprint18_epoch([batch2], arm=bad_arm, epoch=31)


def test_k9_downstream_fits_fit_only_and_scores_both_branches() -> None:
    samples = _tiny_samples(15)
    fit_items = [(s.file_id, s) for s in samples[:12]]
    # every observed (robot, program) cell has >= 2 rows (C7-A minimum)
    cfg = _tiny_config()
    model, patchifier = build_k_model("K9-FULL", cfg)
    downstream = fit_k9_downstream(model, patchifier, cfg, fit_items)
    assert downstream["window"]["duration"] == 64
    assert tuple(downstream["fit_ids"]) == tuple(fid for fid, _ in fit_items)
    cal_lat = k9_forward_latents(model, patchifier, cfg, [(s.file_id, s) for s in samples[12:]])
    s_pred, s_pop, _ = score_k9_files(cal_lat, downstream)
    assert set(s_pred) == {s.file_id for s in samples[12:]}
    assert set(s_pop) == {s.file_id for s in samples[12:]}
    assert all(np.isfinite(v) for v in (*s_pred.values(), *s_pop.values()))
    # determinism: identical refit gives identical scores
    downstream2 = fit_k9_downstream(model, patchifier, cfg, fit_items)
    s_pred2, s_pop2, _ = score_k9_files(
        k9_forward_latents(model, patchifier, cfg, [(s.file_id, s) for s in samples[12:]]),
        downstream2)
    assert s_pred == s_pred2 and s_pop == s_pop2
    # Fit-only refusal: abnormal rows rejected by both downstream fitters
    with pytest.raises(ValueError, match="[Aa]bnormal"):
        HuberStandardizer().fit(torch.randn(4, cfg.d_model),
                                labels=["normal", "abnormal", "normal", "normal"])
    with pytest.raises(ValueError, match="[Aa]bnormal"):
        arm_reference("C7-A").fit(torch.randn(2, cfg.d_model), [0, 1], [0, 1],
                                  labels=["normal", "abnormal"])


def test_harness_bank_scoring_dispatch_serves_k_arms(tmp_path) -> None:
    from representation.sprint18_checkpoint import (
        SCHEDULER_ID,
        _canonical_sha256,
        coefficient_for_epoch,
        deterministic_sampler_order,
        learning_rate_for_epoch,
        restore_sprint18_checkpoint,
        save_sprint18_checkpoint,
        state_sha256,
    )

    binding, histories = _tiny_certified_binding(tmp_path)
    fit_rows = _tiny_certified_fit_rows(binding, histories)
    fit_history = fit_rows[0].actual_history_id
    cal_rows = _tiny_certified_cal_rows(binding, histories)
    served: dict[str, object] = {}
    for arm_id in ("K3-FULL", "K7-FULL", "K9-FULL"):
        cfg = _tiny_config()
        model, patchifier = build_k_model(arm_id, cfg)
        arm = make_sprint18_arm_config(arm_id, SEED, dict(COEFFS), selection_state="task6_smoke_only")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0)
        init_sha = state_sha256(model.state_dict())
        source_ids = sorted(row.file_id for row in fit_rows)
        sampler_seed, epoch_order = deterministic_sampler_order(
            binding, arm, workload="representation", scope="task6_smoke_only",
            role_history_ids=[fit_history], epoch=1, eligible_source_ids=source_ids)
        checkpoint_path = tmp_path / f"tiny-{arm_id.lower()}-bank.pt"
        digest = save_sprint18_checkpoint(
            checkpoint_path, model, optimizer, arm=arm, binding=binding,
            global_step=0, completed_epoch=0, initialization_sha256=init_sha,
            scheduler_state={"scheduler_id": SCHEDULER_ID, "completed_epochs": 0,
                             "active_epoch": 1,
                             "learning_rate": learning_rate_for_epoch(1), "global_step": 0},
            sampler_state={"scope": "task6_smoke_only", "workload": "representation",
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
                           "completed_epochs": 0, "current_epoch": 1,
                           "batches_in_current_epoch": 0,
                           "fit_role_sha256": _canonical_sha256(
                               sorted(item.manifest_sha256 for item in binding.role_histories("FIT")))},
            coefficient_state={"epoch": 1, "global_step": 0,
                               "selection_state": "task6_smoke_only",
                               "lambda_info": coefficient_for_epoch(0.05, 1),
                               "lambda_hcc": None, "alpha": None, "beta": None},
            scope="task6_smoke_only")
        restored_model, _ = build_k_model(arm_id, _tiny_config())
        restored_optimizer = torch.optim.AdamW(
            restored_model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0)
        restored = restore_sprint18_checkpoint(
            checkpoint_path, restored_model, restored_optimizer, arm=arm,
            binding=binding, allowed_scope="task6_smoke_only")
        assert restored.checkpoint_sha256 == digest
        # Public bank consumer: the restored K graph serves the Fit bank path.
        provenanced = fit_fit_reference_bank(
            binding, arm, restored_model, fit_rows, patchifier,
            checkpoint=restored, scope="task6_smoke_only")
        assert provenanced.provenance.arm_id == arm_id
        assert provenanced.provenance.source_rows == len(fit_rows)
        assert torch.isfinite(provenanced.bank.embeddings).all()
        # Public scoring consumer: the same restored graph scores Calibration
        # rows through the public inference pattern (no branch fusion).
        scored = score_certified_rows(
            binding, arm, restored_model, provenanced, cal_rows, patchifier,
            checkpoint=restored, action="calibrate_q95", scope="task6_smoke_only")
        assert set(scored.scores["S_pred"]) == {row.file_id for row in cal_rows}
        assert set(scored.scores["S_pop"]) == {row.file_id for row in cal_rows}
        assert all(np.isfinite(v) for v in (*scored.scores["S_pred"].values(),
                                            *scored.scores["S_pop"].values()))
        # Calibration consumer: NumPy-linear q95 thresholds stay finite.
        calibration = fit_calibration_q95(binding, scored, cal_rows, scope="task6_smoke_only")
        assert math.isfinite(calibration.S_pred.threshold)
        assert math.isfinite(calibration.S_pop.threshold)
        served[arm_id] = {"bank_sha256": provenanced.provenance.bank_sha256,
                          "checkpoint_sha256": digest}
    # Mechanism arms stay refused through the same public entry points: K/N
    # dispatch never broadens the coefficient-pilot surface.
    n1_arm = make_sprint18_arm_config(
        "N1-HCC", SEED,
        {"lambda_info_max": None, "lambda_hcc_max": 0.025, "alpha": None, "beta": None},
        selection_state="locked", coefficient_lock_sha256=LOCK)
    with pytest.raises(ValueError, match="legacy-stack"):
        fit_fit_reference_bank(
            binding, n1_arm, restored_model, fit_rows, patchifier,
            checkpoint=restored, scope="task6_smoke_only")
    assert set(served) == {"K3-FULL", "K7-FULL", "K9-FULL"}


def test_k_build_config_matches_task14_lock() -> None:
    for arm_id in ("K3-FULL", "K7-FULL", "K9-FULL"):
        cfg = build_k_config(arm_id, SEED)
        assert (cfg.n_channels, cfg.patch_size, cfg.stride) == (6, 32, 16)
        assert (cfg.n_robots, cfg.n_programs) == (9, 8)
        assert cfg.contrastive_weight_max == 0.05
        assert cfg.ema_decay == 0.996
        assert cfg.seed == SEED
        arm = make_sprint18_arm_config(arm_id, SEED, dict(COEFFS), selection_state="locked",
                                       coefficient_lock_sha256=LOCK)
        assert arm.lambda_info_max == 0.05
        assert arm.alpha is None and arm.beta is None and arm.lambda_hcc_max is None
