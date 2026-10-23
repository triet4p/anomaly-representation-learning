"""S18-T30 runner: N4-CRN final Fit-only scale fit and Development evaluation.

Binds the exact final epoch-100 B0-FULL checkpoints and prediction caches for
seeds 181801/181802/181803 (accepted Task23 A04 identities, re-verified by
Task13B), strict-restores each checkpoint, rebuilds the N4 scale records from
the bound cache latents (context-only inputs, no re-encoding, no re-masking),
**re-executes the three Fit-history capacity folds on the changed source**
(there is no transfer of the Task13B executed-snapshot PASS without this
recertification), fits the frozen v3 N4 recipe's **final** all-Fit scope
(50 epochs, batch <= 512 records, frozen ``sprint18-n4-scale-order-v2`` CPU
permutations, AdamW 1e-3/1e-4, clip 1.0, Fit-only linear q05/q95 bounds,
``fit-mean-anchored-bias-v1`` initialization, ``n4-v3-guards-v1`` guards, zero
representation updates; 66,989 eligible masked Fit patch records -> 6,550
steps), saves/restores the v3 scale envelope **and** the coherent
``sprint18-coherent-checkpoint-v2`` N4 artifact that binds the unchanged
paired B0 online/EMA bytes, the scale head, the three recertified fold
digests, the exact certified in-scope counts and the Fit-only bounds,
calibrates the N4 ``S_pred`` q95 on the complete Calibration-healthy support,
and scores the four frozen Development histories through the public N4
consumer plus the unchanged B0 ``S_pop`` path. The restored coherent artifact
re-scores the complete Calibration support (identical q95) and a bounded
Development slice (identical per-file scores) through the public consumer.
``S_pop`` parity with the accepted B0 records is proven, not assumed (see
``_development_proof``).

Calls only committed public entry points; implements no science, no metric
redefinition, no protocol amendment:

- ``representation.crn`` scale-model recipe, guards, checkpoint envelope,
  residual bounds, normalized score, calibration q95, cache binding digest and
  the shared file-score/diagnostics forward
- ``representation.sprint18_harness`` binding/loaders/config/bank/q95/scoring,
  deterministic mask seeds, compute envelope
- ``representation.sprint18_checkpoint`` state digests and strict restore
- ``representation.data.collate_variable_files`` / ``representation.model``
  public forwards and the frozen ``RepresentationTrainer`` for the paired B0
  compute counter only
- ``synth.probe15`` (frozen P/W/A metric code) and ``synth.events``

The certified Task13B cache -> N4 record derivation and the versioned Task14
lock validation are reused verbatim from the accepted permanent runner
``experiments/sprint18_task13b_n4_capacity.py`` (imported below), so the N4
record definition and the lock/contract checks cannot drift from the certified
capacity evidence.

Modes:
  preflight - identity/support/lock/binding checks only (no fitting).
  smoke     - bounded real certified-DATA run of the whole path on the first
              frozen seed: bounded Fit record pools, the same frozen recipe,
              real strict restore, real Fit bank, bounded Calibration q95 and
              Development slice, S_pop invariance and compute counters.
              Every output is marked non-normative; it can never be promoted.
  full      - the normative three-seed final fit + calibration + Development
              evaluation and the paired Development report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import sprint18_task13b_n4_capacity as capacity  # noqa: E402  (permanent reviewed runner)

from representation.config import V1Config  # noqa: E402
from representation.crn import (  # noqa: E402
    N4_GUARD_CONTRACT,
    N4_SATURATION_CAP_GATE,
    N4_SCALE_BATCH_SIZE,
    N4_SCALE_CHECKPOINT_SCHEMA,
    N4_SCALE_EPOCHS,
    N4_SCALE_FIT_RECORDS,
    N4_SCALE_HISTORY_RECORDS,
    N4_SCALE_PARAMETERIZATION,
    N4ScaleGuardError,
    fit_n4_scale_scope,
    init_n4_scale_model,
    load_n4_scale_checkpoint,
    n4_cache_binding_digest,
    n4_calibration_q95,
    n4_record_id,
    n4_residual_bounds,
    n4_saturation,
    n4_scale_nll,
    n4_scale_scope_steps,
    n4_score_files,
    n4_spearman,
    save_n4_scale_checkpoint,
    verify_n4_cache_binding,
)
from representation.data import collate_variable_files  # noqa: E402
from representation.model import V1RepresentationModel  # noqa: E402
from representation.sprint18_checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA_ID,
    learning_rate_for_epoch,
    restore_sprint18_checkpoint,
    save_sprint18_checkpoint,
    state_sha256,
)
from representation.sprint18_harness import (  # noqa: E402
    CALIBRATION_HEALTHY_ROWS,
    FIT_HEALTHY_ROWS,
    FIT_VALID_PATCHES,
    MODEL_SEEDS,
    _encoder_batch,
    arm_config_sha256,
    deterministic_mask_seed,
    deterministic_sampler_order,
    fit_calibration_q95,
    fit_fit_reference_bank,
    load_certified_development_support,
    load_certified_healthy_role_support,
    load_certified_sample_subset,
    load_task6_binding,
    make_sprint18_arm_config,
    primary_inference_parameters,
    profile_inference_flops,
    score_certified_rows,
    validate_compute_envelope,
    validate_score_maps,
)
from representation.trainer import RepresentationTrainer  # noqa: E402
from synth import events as E  # noqa: E402
from synth import probe15 as P  # noqa: E402
from synth.config import PatchConfig  # noqa: E402
from synth.patchify import Patchifier  # noqa: E402

TASK_ID = "S18-T30"
ATTEMPT = "S18-T30-A02"
AGENT = "S18T30A02"
ARM_ID = "N4-CRN"
BASELINE_ARM_ID = "B0-FULL"
RUN_ID_PREFIX = "n4"
PREFLIGHT_SCHEMA = "s18-t30-n4-preflight-v2"
SEED_SCHEMA = "s18-t30-n4-seed-record-v2"
PAIRED_SCHEMA = "s18-t30-n4-paired-development-v2"
SMOKE_SCHEMA = "s18-t30-n4-smoke-v2"
GUARD_FAILURE_SCHEMA = "s18-t30-n4-guard-failure-v2"
COHERENT_CHECKPOINT_SCHEMA = "s18-t30-n4-coherent-artifact-v1"
NEXT_STAGE_SCHEMA = "s18-t30-n4-next-stage-v1"
N4_COEFFICIENTS = {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None}
B0_COEFFICIENTS = {"lambda_info_max": 0.05, "lambda_hcc_max": None, "alpha": None, "beta": None}

#: Accepted Task23 A04 seed-record ledger (per-seed checkpoints/caches/banks/
#: calibration/Development identities re-verified by Task13B).
B0_RECORDS_PATH = "/home/trietlm/s18-t23-a04-runs/full/seed_records.json"
B0_RECORDS_SHA256 = "3540ef09fae5c9bf4b768ba93b4dece4a48ef8003a69524a93fbcb156fba5443"
#: The accepted Task23 A04 prediction caches carry two distinct identities per
#: seed: the raw SHA-256 of the cache-file bytes and the digest recorded inside
#: the cache head (also carried by the accepted B0 seed record). The external
#: execution binding must therefore carry both explicitly; a single key cannot
#: satisfy the two fail-closed checks (A03 runtime defect).
PREDICTION_CACHE_FILE_SHA256_KEY = "prediction_cache_file_sha256"
PREDICTION_CACHE_INNER_SHA256_KEY = "prediction_cache_inner_sha256"
#: Exact externally bound fixture schema. Unknown or legacy keys are refused so
#: a stale binding cannot silently satisfy only one of the two identities.
FIXTURE_KEYS = frozenset(
    {
        "checkpoints",
        "ema_state_sha256",
        "live_model_state_sha256",
        PREDICTION_CACHE_FILE_SHA256_KEY,
        PREDICTION_CACHE_INNER_SHA256_KEY,
        "prediction_cache_paths",
    }
)
#: Superseded ambiguous binding key. Refused by name, never interpreted.
AMBIGUOUS_PREDICTION_CACHE_KEY = "prediction_cache_sha256"
_HEX64 = re.compile(r"[0-9a-f]{64}")
#: Certified per-history eligible *masked* Fit patch record counts (Task13B A02;
#: identical for all three model seeds because the mask budget is count-deterministic).
#: Sourced from the frozen implementation constant so the runner, the coherent
#: checkpoint validator and the lock cannot drift apart.
FIT_HISTORY_RECORDS = dict(N4_SCALE_HISTORY_RECORDS)
#: Certified eligible *masked* Fit patch record total (Task13B A02 evidence,
#: identical for all three model seeds because the mask budget is
#: count-deterministic). Re-verified against the real caches at preflight.
#: ``sprint18_harness.FIT_VALID_PATCHES`` counts every valid patch and is not
#: the N4 in-scope record count, so this runner never substitutes it.
FIT_SCALE_RECORDS = N4_SCALE_FIT_RECORDS
#: Bounded smoke pool: real records per Fit history, still fitted by the exact
#: frozen 50-epoch/512-record recipe (non-normative by scope, not by recipe).
SMOKE_RECORDS_PER_HISTORY = 256
SMOKE_DEVELOPMENT_FILES_PER_HISTORY = 8
PAIRED_MACRO_METRICS = ("P_macro", "W_macro", "PW_macro", "PW_lcb95", "directional_histories")
PAIRED_HISTORY_METRICS = (
    "P_AUROC", "W_AUROC", "PW_AUROC", "A_AUROC", "P_recall", "W_recall",
    "P_median_lead_days", "W_median_lead_days", "FAR_per_robot_day",
)
COMPUTE_SCOPE = (
    "one real Fit file forward for inference FLOPs; one real 128-file B0 training "
    "step and one real 512-record N4 scale step for training FLOPs; operational "
    "per-Fit-file estimates, not a full-campaign profiler trace"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _log_progress(handle, record: dict) -> None:
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def _fail(message: str) -> SystemExit:
    print(f"DRIVER_FATAL {message}", file=sys.stderr, flush=True)
    return SystemExit(f"DRIVER_FATAL {message}")


def _move(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}


def _n4_arm(seed: int, lock_digest: str):
    return make_sprint18_arm_config(
        ARM_ID, seed, N4_COEFFICIENTS, selection_state="locked",
        coefficient_lock_sha256=lock_digest,
    )


def _b0_arm(seed: int, lock_digest: str):
    return make_sprint18_arm_config(
        BASELINE_ARM_ID, seed, B0_COEFFICIENTS, selection_state="locked",
        coefficient_lock_sha256=lock_digest,
    )


def _config_from_cache(cache_head: dict) -> V1Config:
    mcfg = dict(cache_head["model_config"])
    return V1Config(
        n_channels=int(mcfg["n_channels"]), patch_size=int(mcfg["patch_size"]),
        stride=int(mcfg["stride"]), d_model=int(mcfg.get("d_model", 128)),
        sequence_layers=int(mcfg.get("sequence_layers", 4)),
        attention_heads=int(mcfg.get("attention_heads", 4)),
        dropout=float(mcfg.get("dropout", 0.1)),
        n_robots=int(mcfg["n_robots"]), n_programs=int(mcfg["n_programs"]),
        use_conditional_norm=bool(mcfg.get("use_conditional_norm", True)),
        min_bucket_samples=int(mcfg.get("min_bucket_samples", 32)),
        total_mask_ratio=float(mcfg.get("total_mask_ratio", 0.4)),
        random_fraction=float(mcfg.get("random_fraction", 0.34)),
        info_fraction=float(mcfg.get("info_fraction", 0.33)),
        block_fraction=float(mcfg.get("block_fraction", 0.33)),
        ema_decay=float(mcfg["ema_decay"]),
        prediction_weight=float(mcfg.get("prediction_weight", 1.0)),
        contrastive_weight_max=float(mcfg["contrastive_weight_max"]),
        contrastive_warmup_steps=int(mcfg.get("contrastive_warmup_steps", 0)),
        contrastive_ramp_steps=int(mcfg.get("contrastive_ramp_steps", 1000)),
        contrastive_temperature=float(mcfg.get("contrastive_temperature", 0.2)),
        contrastive_gain_std=float(mcfg.get("contrastive_gain_std", 0.05)),
        contrastive_offset_std=float(mcfg.get("contrastive_offset_std", 0.03)),
        contrastive_noise_std=float(mcfg.get("contrastive_noise_std", 0.015)),
        contrastive_max_shift=int(mcfg.get("contrastive_max_shift", 4)),
        knn_k=int(mcfg.get("knn_k", 5)),
        seed=int(mcfg["seed"]),
    )


def _patchifier() -> Patchifier:
    return Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))


def _execution_binding(bdoc: dict, args) -> dict:
    """Return the externally authorized execution identity, fail-closed."""
    if not isinstance(bdoc, dict):
        raise _fail("execution binding is not a mapping")
    commit = bdoc.get("source_commit")
    runner = bdoc.get("runner_sha256")
    for label, doc_value, cli_value in (
        ("source commit", commit, getattr(args, "source_commit", None)),
        ("runner digest", runner, getattr(args, "driver_sha256", None)),
    ):
        if not isinstance(doc_value, str) or not isinstance(cli_value, str) or not doc_value or not cli_value:
            raise _fail(f"authorized {label} is missing")
        if doc_value != cli_value:
            raise _fail(f"binding {label} differs from the trusted CLI pin")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise _fail("authorized source commit is not a full Git SHA")
    if len(runner) != 64 or any(char not in "0123456789abcdef" for char in runner):
        raise _fail("authorized runner digest is not a SHA-256 hex digest")
    return {"source_commit": commit, "runner_sha256": runner}


def self_check(args, bdoc: dict) -> dict:
    authorized = _execution_binding(bdoc, args)
    worktree = Path(args.worktree)
    head = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != authorized["source_commit"]:
        raise _fail(f"source commit differs: observed {head}")
    observed_runner = _sha256_file(Path(__file__).resolve())
    if observed_runner != authorized["runner_sha256"]:
        raise _fail("runner bytes differ from the authorized execution binding")
    return {
        "head": head,
        "runner_sha256": observed_runner,
        "binding_sha256": _sha256_file(Path(args.binding)),
    }


def _check_contract_files(worktree: Path) -> None:
    capacity._check_contract_files(worktree)


def _load_b0_records(path: Path) -> dict:
    if not path.is_file():
        raise _fail(f"accepted B0 seed records are missing: {path}")
    if _sha256_file(path) != B0_RECORDS_SHA256:
        raise _fail("accepted B0 seed-record ledger bytes differ from the pinned digest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if sorted(payload) != [str(seed) for seed in MODEL_SEEDS]:
        raise _fail("accepted B0 seed-record ledger does not cover exactly the frozen model seeds")
    return payload


def _validate_fixture(fixture: object) -> dict:
    """Accept only the exact externally bound B0 fixture schema, fail-closed."""
    if not isinstance(fixture, dict):
        raise _fail("execution binding fixture is not a mapping")
    key_set = set(fixture)
    if AMBIGUOUS_PREDICTION_CACHE_KEY in key_set:
        raise _fail(
            "execution binding fixture carries the ambiguous legacy key "
            f"{AMBIGUOUS_PREDICTION_CACHE_KEY}; bind {PREDICTION_CACHE_FILE_SHA256_KEY} "
            f"(raw cache-file bytes) and {PREDICTION_CACHE_INNER_SHA256_KEY} (cache-head digest) "
            "as distinct per-seed identities"
        )
    if key_set != FIXTURE_KEYS:
        raise _fail(
            "execution binding fixture keys differ from the frozen schema: "
            f"{sorted(key_set)}"
        )
    expected_seeds = {str(seed) for seed in MODEL_SEEDS}
    for name in (
        "live_model_state_sha256",
        "ema_state_sha256",
        PREDICTION_CACHE_FILE_SHA256_KEY,
        PREDICTION_CACHE_INNER_SHA256_KEY,
    ):
        block = fixture[name]
        if not isinstance(block, dict) or set(block) != expected_seeds:
            raise _fail(
                f"execution binding fixture {name} does not cover exactly the frozen model seeds"
            )
        for seed_key, value in block.items():
            if not isinstance(value, str) or not _HEX64.fullmatch(value):
                raise _fail(
                    f"execution binding fixture {name}[{seed_key}] is not a SHA-256 hex digest"
                )
    paths = fixture["prediction_cache_paths"]
    if not isinstance(paths, dict) or set(paths) != expected_seeds:
        raise _fail(
            "execution binding fixture prediction_cache_paths does not cover exactly "
            "the frozen model seeds"
        )
    for seed_key, value in paths.items():
        if not isinstance(value, str) or not value:
            raise _fail(
                f"execution binding fixture prediction_cache_paths[{seed_key}] is empty"
            )
    checkpoints = fixture["checkpoints"]
    if not isinstance(checkpoints, dict) or set(checkpoints) != expected_seeds:
        raise _fail(
            "execution binding fixture checkpoints does not cover exactly the frozen model seeds"
        )
    for seed_key, value in checkpoints.items():
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise _fail(
                f"execution binding fixture checkpoints[{seed_key}] must be exactly path+sha256"
            )
        if not isinstance(value["path"], str) or not value["path"]:
            raise _fail(f"execution binding fixture checkpoints[{seed_key}].path is empty")
        if not isinstance(value["sha256"], str) or not _HEX64.fullmatch(value["sha256"]):
            raise _fail(
                f"execution binding fixture checkpoints[{seed_key}].sha256 is not a SHA-256 digest"
            )
    return fixture


def _b0_cache_identity(
    *, fixture: dict, seed: int, b0_record: dict, cache_head: dict, cache_records: list
) -> dict:
    """Verify BOTH bound cache identities against the actual bytes and the accepted record.

    The raw file digest and the cache-head digest are distinct identities of the
    same accepted Task23 A04 cache. Each is compared against its own explicit
    binding key; neither can substitute for the other.
    """
    key = str(seed)
    cache = b0_record["prediction_cache"]
    cache_path = Path(fixture["prediction_cache_paths"][key])
    if cache["path"] != str(cache_path):
        raise _fail(f"B0 seed {seed} prediction cache path differs from the bound fixture")
    file_sha256 = _sha256_file(cache_path)
    bound_file_sha256 = fixture[PREDICTION_CACHE_FILE_SHA256_KEY][key]
    if file_sha256 != bound_file_sha256:
        raise _fail(
            f"B0 seed {seed} prediction cache bytes differ from the bound file digest"
        )
    inner_sha256 = cache_head.get("cache_sha256")
    if not isinstance(inner_sha256, str) or not _HEX64.fullmatch(inner_sha256):
        raise _fail(f"B0 seed {seed} prediction cache head carries no valid inner digest")
    bound_inner_sha256 = fixture[PREDICTION_CACHE_INNER_SHA256_KEY][key]
    if inner_sha256 != bound_inner_sha256:
        raise _fail(
            f"B0 seed {seed} prediction cache head digest differs from the bound inner digest"
        )
    if cache["cache_sha256"] != inner_sha256:
        raise _fail(
            f"B0 seed {seed} prediction cache inner digest differs from the accepted B0 record"
        )
    if cache["records"] != len(cache_records):
        raise _fail(f"B0 seed {seed} prediction cache record count differs from the accepted B0 record")
    return {
        "cache_path": str(cache_path),
        "cache_file_sha256": file_sha256,
        "cache_inner_sha256": inner_sha256,
        "cache_records": len(cache_records),
    }


def _b0_seed_identity(
    seed: int, fixture: dict, b0_record: dict, cache_identity: dict
) -> dict:
    """Cross-check one accepted B0 seed record against the externally bound fixture.

    ``cache_identity`` is the already-verified byte/head identity of the same
    cache, so neither digest is re-derived or re-interpreted here.
    """
    key = str(seed)
    checkpoint = fixture["checkpoints"][key]
    if b0_record.get("arm_id") != BASELINE_ARM_ID or b0_record.get("status") != "ok":
        raise _fail(f"B0 seed {seed} record is not a completed {BASELINE_ARM_ID} run")
    if b0_record.get("representation_epochs_completed") != 100:
        raise _fail(f"B0 seed {seed} record did not complete 100 representation epochs")
    primary = b0_record["primary"]
    if primary["restored_checkpoint_sha256"] != checkpoint["sha256"]:
        raise _fail(f"B0 seed {seed} primary checkpoint differs from the bound fixture")
    if primary["live_model_state_sha256"] != fixture["live_model_state_sha256"][key]:
        raise _fail(f"B0 seed {seed} live model digest differs from the bound fixture")
    if primary["live_ema_state_sha256"] != fixture["ema_state_sha256"][key]:
        raise _fail(f"B0 seed {seed} live EMA digest differs from the bound fixture")
    if b0_record["bank"]["source_rows"] != FIT_HEALTHY_ROWS:
        raise _fail(f"B0 seed {seed} Fit bank rows differ from the frozen Fit support")
    if b0_record["calibration"]["source_rows"] != CALIBRATION_HEALTHY_ROWS:
        raise _fail(f"B0 seed {seed} Calibration rows differ from the frozen healthy support")
    return {
        **cache_identity,
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "live_model_state_sha256": primary["live_model_state_sha256"],
        "live_ema_state_sha256": primary["live_ema_state_sha256"],
        "bank_sha256": b0_record["bank"]["bank_sha256"],
        "bank_source_rows_sha256": b0_record["bank"]["source_rows_sha256"],
        "calibration_source_support_sha256": b0_record["calibration"]["source_support_sha256"],
        "calibration_S_pred_threshold": b0_record["calibration"]["S_pred_threshold"],
        "calibration_S_pop_threshold": b0_record["calibration"]["S_pop_threshold"],
    }


def _b0_restore_identity(*, fixture: dict, seed: int, b0_record: dict) -> dict:
    """Prove the exact bytes the B0 restore will consume, before any long work.

    This is the preflight restore-level identity proof: it validates both cache
    identities (raw file bytes and head digest), the accepted-record agreement,
    the cache record count and the checkpoint bytes that the strict restore
    will load. It is the single identity gate shared by preflight and
    ``_restore_b0``.
    """
    key = str(seed)
    cache_path = Path(fixture["prediction_cache_paths"][key])
    if not cache_path.is_file():
        raise _fail(f"B0 seed {seed} prediction cache is missing: {cache_path}")
    cache_head, cache_records = capacity._load_cache_records(cache_path)
    cache_identity = _b0_cache_identity(
        fixture=fixture, seed=seed, b0_record=b0_record,
        cache_head=cache_head, cache_records=cache_records,
    )
    checkpoint = fixture["checkpoints"][key]
    checkpoint_path = Path(checkpoint["path"])
    if not checkpoint_path.is_file():
        raise _fail(f"B0 seed {seed} final B0 checkpoint is missing: {checkpoint_path}")
    if _sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise _fail(f"B0 seed {seed} B0 checkpoint bytes differ from the bound digest")
    identity = _b0_seed_identity(seed, fixture, b0_record, cache_identity)
    return {
        "identity": identity,
        "cache_head": cache_head,
        "cache_records": cache_records,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint["sha256"],
    }


def _role_assets(binding, worktree: Path, history_id: str):
    history = binding.history(history_id)
    role_dir = worktree / history.relative_path
    manifest_path = role_dir / "manifest.json"
    if not role_dir.is_dir():
        raise _fail(f"certified role directory is missing: {role_dir}")
    if not manifest_path.is_file():
        raise _fail(f"certified role manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_paths = {row["path"]: str(role_dir / row["path"]) for row in manifest["shards"]}
    return history, manifest, manifest_path, shard_paths


def _support_record(history, manifest, manifest_path: Path, rows) -> dict:
    ids = sorted(row.file_id for row in rows)
    return {
        "history_id": history.actual_history_id,
        "role": history.assigned_role,
        "manifest_sha256_bound": history.manifest_sha256,
        "manifest_sha256_observed": _sha256_file(manifest_path),
        "manifest_file_rows_observed": len(manifest["files"]),
        "healthy_rows": len(rows),
        "valid_patches": sum(int(row.manifest_row["n_valid_patches"]) for row in rows),
        "eligible_ids_sha256": _canonical_sha256(ids),
        "unique_ids": len(set(ids)) == len(ids),
        "all_rows_verified_healthy": all(row.is_verified_healthy for row in rows),
    }


def _canonical_fit_rows(fit_rows: list) -> list:
    """Return Fit rows in the accepted B0 reference-bank row order.

    ``fit_fit_reference_bank`` hashes the embedding matrix in row order, and the
    accepted B0 bank (Task23 A04) was built from `fit_rows_ordered`, i.e. the
    globally ``file_id``-sorted Fit support. Certified Fit file IDs are unique
    across the three Fit histories, so that sort is a total order and is the
    canonical bank order. Any other input order (for example the per-history
    concatenation the certified loader returns) keeps the same rows but yields
    a different matrix order and therefore a different ``bank_sha256``.
    """
    by_id = {row.file_id: row for row in fit_rows}
    if len(by_id) != len(fit_rows):
        raise _fail("certified Fit support contains duplicate file IDs")
    return [by_id[file_id] for file_id in sorted(by_id)]


def _load_supports(binding, worktree: Path, *, development: bool) -> dict:
    fit_history_ids = [item.actual_history_id for item in binding.role_histories("FIT")]
    calibration_history_ids = [item.actual_history_id for item in binding.role_histories("CALIBRATION")]
    development_history_ids = [item.actual_history_id for item in binding.role_histories("DEVELOPMENT")]
    if len(fit_history_ids) != 3 or len(calibration_history_ids) != 1 or len(development_history_ids) != 4:
        raise _fail("certified roster differs from the frozen 3/1/4 Fit/Calibration/Development topology")
    fit_rows: list = []
    fit_records: list[dict] = []
    for history_id in fit_history_ids:
        history, manifest, manifest_path, shard_paths = _role_assets(binding, worktree, history_id)
        rows = load_certified_healthy_role_support(binding, history_id, manifest_path, shard_paths, action="train_fit")
        record = _support_record(history, manifest, manifest_path, rows)
        if not record["unique_ids"] or not record["all_rows_verified_healthy"]:
            raise _fail(f"Fit support failed identity/eligibility for {history_id}")
        fit_records.append(record)
        fit_rows.extend(rows)
    # Canonicalize once, here, so every downstream consumer of Fit rows (the
    # accepted-B0 reference bank, the compute probes and the probe row) sees the
    # identical ordering the accepted B0 bank digest binds.
    fit_rows = _canonical_fit_rows(fit_rows)
    fit_ids = [row.file_id for row in fit_rows]
    if len(fit_ids) != binding.fit_healthy_rows or len(set(fit_ids)) != binding.fit_healthy_rows:
        raise _fail("certified Fit support differs from the bound healthy rows")
    if sum(int(row.manifest_row["n_valid_patches"]) for row in fit_rows) != FIT_VALID_PATCHES:
        raise _fail("certified Fit valid-patch total differs from the bound contract")
    calibration_rows: list = []
    calibration_records: list[dict] = []
    for history_id in calibration_history_ids:
        history, manifest, manifest_path, shard_paths = _role_assets(binding, worktree, history_id)
        normal_ids = [row["file_id"] for row in manifest["files"] if row.get("file_label") == "normal"]
        rows = load_certified_sample_subset(
            binding, history_id, manifest_path, shard_paths, normal_ids,
            role_action="calibrate_q95", verify_all_shards=True,
        )
        record = _support_record(history, manifest, manifest_path, rows)
        if not record["all_rows_verified_healthy"]:
            raise _fail(f"Calibration support failed the healthy predicate for {history_id}")
        calibration_records.append(record)
        calibration_rows.extend(rows)
    if len(calibration_rows) != binding.calibration_healthy_rows:
        raise _fail("certified Calibration support differs from the bound healthy rows")
    development_records: list[dict] = []
    development_rows: dict[str, list] = {}
    for history_id in development_history_ids:
        history, manifest, manifest_path, shard_paths = _role_assets(binding, worktree, history_id)
        development_records.append(
            {
                "history_id": history_id,
                "role": history.assigned_role,
                "manifest_sha256_bound": history.manifest_sha256,
                "manifest_sha256_observed": _sha256_file(manifest_path),
                "manifest_file_rows_observed": len(manifest["files"]),
            }
        )
        if development:
            development_rows[history_id] = list(
                load_certified_development_support(binding, history_id, manifest_path, shard_paths)
            )
    return {
        "fit_history_ids": fit_history_ids,
        "calibration_history_ids": calibration_history_ids,
        "development_history_ids": development_history_ids,
        "fit_rows": fit_rows,
        "fit_records": fit_records,
        "calibration_rows": calibration_rows,
        "calibration_records": calibration_records,
        "development_records": development_records,
        "development_rows": development_rows,
    }


def _fit_role_sha256(binding) -> str:
    return _canonical_sha256(sorted(item.manifest_sha256 for item in binding.role_histories("FIT")))


def _cache_pools(cache_records: list[dict], fit_history_ids: list[str]) -> dict[str, list]:
    pools = capacity.records_from_cache(cache_records, fit_history_ids)
    observed = {history_id: len(pool) for history_id, pool in pools.items()}
    if observed != FIT_HISTORY_RECORDS:
        raise _fail(f"cache-derived N4 record counts differ from the certified counts: {observed}")
    if sum(observed.values()) != FIT_SCALE_RECORDS:
        raise _fail("cache-derived N4 record total differs from the frozen in-scope record count")
    return pools


def _final_bounds(pools: dict[str, list], fit_history_ids: list[str]) -> dict:
    residuals = np.array(
        [record.residual for history_id in fit_history_ids for record in pools[history_id]],
        dtype=np.float64,
    )
    bounds = n4_residual_bounds(residuals)
    q05 = float(np.quantile(residuals, 0.05, method="linear"))
    q95 = float(np.quantile(residuals, 0.95, method="linear"))
    if bounds["q05"] != q05 or bounds["q95"] != q95:
        raise _fail("all-Fit quantile recomputation differs from n4_residual_bounds")
    if bounds["v_floor"] != max(q05, 1e-6) or bounds["v_cap"] != max(q95, 10.0 * bounds["v_floor"]):
        raise _fail("all-Fit bound edge rules differ from the frozen formulas")
    return {
        "q05": q05,
        "q95": q95,
        "v_floor": float(bounds["v_floor"]),
        "v_cap": float(bounds["v_cap"]),
        "record_count": int(residuals.shape[0]),
        "residual_records_sha256": _canonical_sha256(
            [
                [history_id, record.file_id, int(record.patch_index), float(record.residual)]
                for history_id in fit_history_ids
                for record in pools[history_id]
            ]
        ),
    }


def _fit_final_scope(*, records, fit_history_ids, arm, binding, seed, bounds, device, out_dir) -> tuple[object, dict, torch.optim.AdamW]:
    records = list(records)
    if len(records) != FIT_SCALE_RECORDS:
        raise _fail(f"seed {seed}: final fit pool differs from {FIT_SCALE_RECORDS} records")
    scale = init_n4_scale_model(128, seed).to(device)
    if sum(parameter.numel() for parameter in scale.parameters()) != 4417:
        raise _fail(f"seed {seed}: deployed scale head parameter count differs from 4,417")
    optimizer = torch.optim.AdamW(scale.parameters(), lr=1e-3, weight_decay=1e-4)
    training_ids = list(sorted(fit_history_ids))
    try:
        result = fit_n4_scale_scope(
            records=records, training_history_ids=training_ids, scope="final",
            arm=arm, binding=binding, scale_model=scale,
            v_floor=bounds["v_floor"], v_cap=bounds["v_cap"], device=device,
            optimizer=optimizer,
        )
    except N4ScaleGuardError as exc:
        _write_json(
            Path(out_dir) / "guard_failure.json",
            {
                "schema": GUARD_FAILURE_SCHEMA,
                "task_id": TASK_ID, "attempt_id": ATTEMPT, "agent_id": AGENT,
                "model_seed": seed, "scope": "final", "training_history_ids": training_ids,
                "guard_id": exc.guard_id, "guard_contract": N4_GUARD_CONTRACT,
                "message": str(exc), "diagnostics": exc.diagnostics,
                "optimizer_steps_applied": 0, "checkpoint_written": False, "run_status": "FAILED",
            },
        )
        raise
    expected_steps = n4_scale_scope_steps(FIT_SCALE_RECORDS)
    if result["record_count"] != FIT_SCALE_RECORDS or result["optimizer_steps"] != expected_steps:
        raise _fail(f"seed {seed}: final fit accounting differs from 50*ceil(N/512) = {expected_steps}")
    if result["per_epoch_steps"] != math.ceil(FIT_SCALE_RECORDS / N4_SCALE_BATCH_SIZE):
        raise _fail(f"seed {seed}: final fit per-epoch steps differ from ceil(N/512)")
    if not math.isfinite(result["final_train_nll"]) or not result["cap_gate_pass_train"]:
        raise _fail(f"seed {seed}: final Fit capacity/saturation gate failed")
    return scale, result, optimizer


def _probe_scale_parity(*, live_scale, restored_scale, probe, v_floor: float, v_cap: float) -> tuple[float, float, float]:
    """Compare the live and strict-restored scale heads on one actual device.

    The certified cache loader materializes N4 context rows on CPU while the live
    head is deployed on the run device, so a raw forward would cross devices.
    Align the probe context and the restored head to the live head's actual
    device for the comparison and return the restored head to where it came
    from afterwards; nothing is copied when the devices already match. This is
    the same contract the certified Task13B runner applies to its held-out
    probe. Strict forward and NLL parity checks stay with the caller.
    """
    target = next(live_scale.parameters()).device
    context = torch.stack([record.context.float() for record in probe], dim=0)
    if context.device != target:
        context = context.to(target)
    residual = torch.tensor([float(record.residual) for record in probe], dtype=torch.float32)
    restored_origin = next(restored_scale.parameters()).device
    restored_on_device = restored_scale if restored_origin == target else restored_scale.to(target)
    live_scale.eval()
    restored_on_device.eval()
    try:
        with torch.no_grad():
            live_out = live_scale(context, v_floor=float(v_floor), v_cap=float(v_cap))
            restored_out = restored_on_device(context, v_floor=float(v_floor), v_cap=float(v_cap))
            forward_diff = float(
                (live_out["log_v"].detach().cpu() - restored_out["log_v"].detach().cpu()).abs().max().item()
            )
            live_nll = float(n4_scale_nll(residual, live_out["log_v"].detach().cpu()).item())
            restored_nll = float(n4_scale_nll(residual, restored_out["log_v"].detach().cpu()).item())
    finally:
        if restored_on_device is not restored_scale:
            restored_scale.to(restored_origin)
    return forward_diff, live_nll, restored_nll


def _save_and_restore_scale(*, scale, out_dir: Path, seed: int, fit_history_ids, bounds, fit_result) -> dict:
    training_ids = list(sorted(fit_history_ids))
    checkpoint_path = Path(out_dir) / "checkpoints" / f"n4-seed{seed}-final.pt"
    meta = save_n4_scale_checkpoint(
        checkpoint_path, scale, fit_scope="final", training_history_ids=training_ids,
        v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
        initialization={
            "init_anchor": dict(fit_result["init_anchor"]),
            "init_interiority": dict(fit_result["init_interiority"]),
            "first_step_gradients": dict(fit_result["first_step_gradients"]),
        },
    )
    live_sha = state_sha256(scale.state_dict())
    if live_sha != meta["state_dict_sha256"]:
        raise _fail(f"seed {seed}: scale checkpoint state digest differs from the trained head")
    restored_scale = init_n4_scale_model(128, seed + 999999)
    restored_meta = load_n4_scale_checkpoint(
        checkpoint_path, restored_scale, expected_fit_scope="final",
        expected_training_history_ids=training_ids,
        expected_bounds={"v_floor": bounds["v_floor"], "v_cap": bounds["v_cap"]},
    )
    if restored_meta["state_dict_sha256"] != live_sha:
        raise _fail(f"seed {seed}: strict scale restore digest differs from the trained head")
    if restored_meta["parameterization"] != N4_SCALE_PARAMETERIZATION:
        raise _fail(f"seed {seed}: strict scale restore parameterization differs")
    restored_sha = state_sha256(restored_scale.state_dict())
    if restored_sha != live_sha:
        raise _fail(f"seed {seed}: restored scale bytes differ from the trained head")
    probe = list(fit_result.get("probe_records") or [])
    forward_diff = 0.0
    restored_nll = None
    if probe:
        forward_diff, live_nll, restored_nll = _probe_scale_parity(
            live_scale=scale, restored_scale=restored_scale, probe=probe,
            v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
        )
        if forward_diff != 0.0 or live_nll != restored_nll:
            raise _fail(f"seed {seed}: strict scale restore forward/NLL parity failed")
    return {
        "path": str(checkpoint_path),
        "sha256": _sha256_file(checkpoint_path),
        "schema": meta["schema"],
        "parameterization": meta["parameterization"],
        "guard_contract": meta["guard_contract"],
        "fit_scope": meta["fit_scope"],
        "training_history_ids": list(meta["training_history_ids"]),
        "state_dict_sha256": meta["state_dict_sha256"],
        "restored_state_dict_sha256": restored_sha,
        "restored_forward_max_abs_diff": forward_diff,
        "probe_records": len(probe),
        "restored_probe_nll": restored_nll,
    }


#: The certified fold identity compared field by field against the
#: authenticated historical payload: frozen config/recipe, bounds, guards,
#: initialization, seed/fold/qualified support, sampler and budget. Comparison
#: is exact; there is no tolerance, no loose equivalence and no missing-field
#: fallback.
_CERTIFIED_FOLD_IDENTITY_FIELDS = (
    "model_seed",
    "fold_index",
    "scope",
    "heldout_history_id",
    "training_history_ids",
    "checkpoint_schema",
    "checkpoint_parameterization",
    "scale_parameterization",
    "guard_contract",
    "scale_parameters",
    "record_count",
    "heldout_count",
    "optimizer_steps",
    "per_epoch_steps",
    "bounds",
    "quantiles",
    "init_anchor",
    "init_interiority",
    "first_step_gradients",
    "epoch_order_seeds",
    "order_seed_rule",
    "cap_gate_pass_train",
)


def _authenticate_certified_fold(*, certified, seed: int, heldout: str) -> dict:
    """Authenticate the retained historical Task13B envelope file and payload.

    Custody and scientific identity are separate: the certified container
    digest authenticates the historical FILE bytes (the artifact the accepted
    evidence points at), while the strict envelope loader plus the certified
    payload digest authenticate the scale state inside it. Both are required
    before the recertified fold may be compared, so an unreachable or
    substituted envelope fails closed instead of degrading to a container
    equality test.
    """
    envelope_path = Path(str(certified["envelope_path"]))
    if not envelope_path.is_file():
        raise _fail(
            f"seed {seed}: fold {heldout} certified historical envelope is unreachable at "
            f"{envelope_path}; the retained Task13B A02 checkpoint custody root recorded in "
            "the certified capacity evidence is a stated prerequisite for the certified-"
            "capacity identity comparison"
        )
    container_sha = _sha256_file(envelope_path)
    if container_sha != certified["checkpoint_sha256"]:
        raise _fail(
            f"seed {seed}: fold {heldout} historical envelope bytes {container_sha} do not "
            f"authenticate against the certified container digest "
            f"{certified['checkpoint_sha256']}"
        )
    probe = init_n4_scale_model(128, seed + 999999)
    meta = load_n4_scale_checkpoint(envelope_path, probe)
    payload_sha = state_sha256(probe.state_dict())
    if meta["state_dict_sha256"] != payload_sha:
        raise _fail(
            f"seed {seed}: fold {heldout} authenticated historical envelope payload digest is "
            "inconsistent with its own canonical state digest"
        )
    if payload_sha != certified["scale_state_sha256"]:
        raise _fail(
            f"seed {seed}: fold {heldout} authenticated historical payload {payload_sha} differs "
            f"from the certified payload digest {certified['scale_state_sha256']}"
        )
    historical_identity = {
        "model_seed": int(certified["model_seed"]),
        "fold_index": int(certified["fold_index"]),
        "scope": str(meta["fit_scope"]),
        "heldout_history_id": str(certified["heldout_history_id"]),
        "training_history_ids": list(meta["training_history_ids"]),
        "checkpoint_schema": str(meta["schema"]),
        "checkpoint_parameterization": str(meta["parameterization"]),
        "scale_parameterization": str(certified["scale_parameterization"]),
        "guard_contract": str(meta["guard_contract"]),
        "scale_parameters": int(certified["scale_parameters"]),
        "record_count": int(certified["record_count"]),
        "heldout_count": int(certified["heldout_count"]),
        "optimizer_steps": int(certified["optimizer_steps"]),
        "per_epoch_steps": int(certified["per_epoch_steps"]),
        "bounds": {
            "v_floor": float(meta["v_floor"]),
            "v_cap": float(meta["v_cap"]),
        },
        "quantiles": {
            "q05": float(certified["quantiles"]["q05"]),
            "q95": float(certified["quantiles"]["q95"]),
        },
        "init_anchor": dict(meta["initialization"]["init_anchor"]),
        "init_interiority": dict(meta["initialization"]["init_interiority"]),
        "first_step_gradients": dict(meta["initialization"]["first_step_gradients"]),
        "epoch_order_seeds": list(certified["epoch_order_seeds"]),
        "order_seed_rule": str(certified["order_seed_rule"]),
        "cap_gate_pass_train": bool(certified["cap_gate_pass_train"]),
    }
    recorded_identity = {
        "model_seed": int(certified["model_seed"]),
        "fold_index": int(certified["fold_index"]),
        "scope": str(certified["scope"]),
        "heldout_history_id": str(certified["heldout_history_id"]),
        "training_history_ids": list(certified["training_history_ids"]),
        "checkpoint_schema": str(certified["checkpoint_schema"]),
        "checkpoint_parameterization": str(certified["checkpoint_parameterization"]),
        "scale_parameterization": str(certified["scale_parameterization"]),
        "guard_contract": str(certified["guard_contract"]),
        "scale_parameters": int(certified["scale_parameters"]),
        "record_count": int(certified["record_count"]),
        "heldout_count": int(certified["heldout_count"]),
        "optimizer_steps": int(certified["optimizer_steps"]),
        "per_epoch_steps": int(certified["per_epoch_steps"]),
        "bounds": {name: float(certified["bounds"][name]) for name in ("v_floor", "v_cap")},
        "quantiles": {name: float(certified["quantiles"][name]) for name in ("q05", "q95")},
        "init_anchor": dict(certified["init_anchor"]),
        "init_interiority": dict(certified["init_interiority"]),
        "first_step_gradients": dict(certified["first_step_gradients"]),
        "epoch_order_seeds": list(certified["epoch_order_seeds"]),
        "order_seed_rule": str(certified["order_seed_rule"]),
        "cap_gate_pass_train": bool(certified["cap_gate_pass_train"]),
    }
    for field in _CERTIFIED_FOLD_IDENTITY_FIELDS:
        if historical_identity[field] != recorded_identity[field]:
            raise _fail(
                f"seed {seed}: fold {heldout} authenticated historical envelope {field} differs "
                "from the certified fold record"
            )
    return {
        "container_sha256": container_sha,
        "payload_sha256": payload_sha,
        "identity": historical_identity,
    }


def _certified_fold_records(
    capacity_certification: dict, *, seed: int | None = None
) -> list[dict]:
    """Attach the certified historical envelope path to each projected fold.

    ``capacity_certification["envelope_root"]`` is the exact historical output
    directory projected by the lock consumer from the SHA-pinned certification
    evidence (the run's ``--out-dir``, not its grandparent), and the relative
    ``checkpoints/...`` path is certified. Joining the two is the only path
    derivation allowed here; no scan or fallback exists.
    """
    envelope_root = str(capacity_certification["envelope_root"]).rstrip("/")
    return [
        {**fold, "envelope_path": f"{envelope_root}/{fold['checkpoint_path']}"}
        for fold in capacity_certification["folds"]
        if seed is None or int(fold["model_seed"]) == int(seed)
    ]


def _authenticate_certified_folds(
    *, capacity_certification: dict, fit_history_ids
) -> dict:
    """Authenticate all nine retained historical envelopes before any fit.

    The preflight must fail on a missing, substituted or wrongly rooted
    certificate envelope before the long Fit campaign starts, so the complete
    seed x fold roster is authenticated here: exact certified path, raw
    container digest and canonical payload digest against the closed identity
    record. No fitting occurs in this function.
    """
    records = _certified_fold_records(capacity_certification)
    expected = [
        (int(seed), int(index), str(heldout))
        for seed in MODEL_SEEDS
        for index, heldout in enumerate(fit_history_ids)
    ]
    observed = sorted(
        (
            int(fold["model_seed"]),
            int(fold["fold_index"]),
            str(fold["heldout_history_id"]),
        )
        for fold in records
    )
    if observed != sorted(expected):
        raise _fail(
            "certified historical fold roster differs from the nine certified seed x fold records"
        )
    files = []
    for fold in sorted(
        records, key=lambda item: (int(item["model_seed"]), int(item["fold_index"]))
    ):
        authenticated = _authenticate_certified_fold(
            certified=fold,
            seed=int(fold["model_seed"]),
            heldout=str(fold["heldout_history_id"]),
        )
        files.append(
            {
                "model_seed": int(fold["model_seed"]),
                "fold_index": int(fold["fold_index"]),
                "envelope_path": str(fold["envelope_path"]),
                "container_sha256": authenticated["container_sha256"],
                "payload_sha256": authenticated["payload_sha256"],
            }
        )
    return {
        "authenticated": True,
        "folds_authenticated": len(files),
        "files": files,
    }


def _require_identical_certified_fold(
    *, seed: int, heldout: str, live_identity, historical_identity,
    live_payload_sha: str, historical_payload_sha: str,
) -> None:
    """Refuse any recertified fold that is not the authenticated historical fold.

    Compares the canonical scale-state digest and every declared identity field
    exactly. A changed weight, bound, support, guard/initialization, sampler,
    budget or provenance value is a scientific difference, not a serialization
    difference, and is refused.
    """
    if live_payload_sha != historical_payload_sha:
        raise _fail(
            f"seed {seed}: fold {heldout} recertified scale state {live_payload_sha} differs "
            f"from the authenticated certified payload {historical_payload_sha}; capacity "
            "evidence does not transfer"
        )
    for field in _CERTIFIED_FOLD_IDENTITY_FIELDS:
        observed = live_identity[field]
        expected = historical_identity[field]
        if observed != expected:
            raise _fail(
                f"seed {seed}: fold {heldout} recertified {field} differs from the authenticated "
                f"certified payload ({observed!r} != {expected!r}); capacity evidence does not "
                "transfer"
            )


def _run_capacity_folds(*, pools, fit_history_ids, arm, binding, seed, device, out_dir, certified_folds, normative: bool) -> dict:
    """Re-execute the three N4 Fit-history folds on the changed source.

    This is the actual capacity recertification that must precede the final
    all-Fit scale fit: the certified Task13B evidence was executed at its own
    snapshot and is never transferred to changed source. Each fold is fitted
    under the unchanged frozen recipe, gated by the cap-only 20% structural
    rule, saved as a v3 scale envelope and strict-restored.

    On a normative run the recertified fold must equal the historical certified
    fold in *scientific identity*, not in container bytes: the retained
    historical envelope is authenticated by its raw container digest and by its
    canonical scale-state digest, and the recertified canonical scale state plus
    the exact config/bounds/guards/initialization/seed/fold/support/sampler/
    budget fields are then compared without tolerance. The recertified
    ``torch.save`` container digest is recorded as a custody observation only.
    """
    folds: list[dict] = []
    for index, heldout in enumerate(fit_history_ids):
        training_ids = [history_id for history_id in fit_history_ids if history_id != heldout]
        train_pool = [record for history_id in training_ids for record in pools[history_id]]
        heldout_pool = list(pools[heldout])
        residuals = np.array([record.residual for record in train_pool], dtype=np.float64)
        bounds = n4_residual_bounds(residuals)
        q05 = float(np.quantile(residuals, 0.05, method="linear"))
        q95 = float(np.quantile(residuals, 0.95, method="linear"))
        if bounds["q05"] != q05 or bounds["q95"] != q95:
            raise _fail(f"seed {seed}: fold {heldout} quantiles differ from n4_residual_bounds")
        if bounds["v_floor"] != max(q05, 1e-6) or bounds["v_cap"] != max(q95, 10.0 * bounds["v_floor"]):
            raise _fail(f"seed {seed}: fold {heldout} bound edges differ from the frozen formulas")
        scale = init_n4_scale_model(128, seed).to(device)
        optimizer = torch.optim.AdamW(scale.parameters(), lr=1e-3, weight_decay=1e-4)
        scope = f"fold:{heldout}"
        result = fit_n4_scale_scope(
            records=train_pool, training_history_ids=training_ids, scope=scope,
            arm=arm, binding=binding, scale_model=scale,
            v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
            heldout_records=heldout_pool, device=device, optimizer=optimizer,
        )
        if result["record_count"] != len(train_pool) or result["heldout_count"] != len(heldout_pool):
            raise _fail(f"seed {seed}: fold {heldout} record counts differ from the cached pools")
        expected_steps = n4_scale_scope_steps(len(train_pool))
        if result["optimizer_steps"] != expected_steps:
            raise _fail(f"seed {seed}: fold {heldout} steps differ from 50*ceil(N/512)")
        if not math.isfinite(result["final_train_nll"]) or not math.isfinite(result["heldout_nll"]):
            raise _fail(f"seed {seed}: fold {heldout} NLL is nonfinite")
        if not result["cap_gate_pass_train"]:
            raise _fail(f"seed {seed}: fold {heldout} breached the frozen 20% saturation gate")
        checkpoint_path = Path(out_dir) / "checkpoints" / f"n4-seed{seed}-fold{index}-{heldout}.pt"
        meta = save_n4_scale_checkpoint(
            checkpoint_path, scale, fit_scope=scope, training_history_ids=training_ids,
            v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
            initialization={
                "init_anchor": dict(result["init_anchor"]),
                "init_interiority": dict(result["init_interiority"]),
                "first_step_gradients": dict(result["first_step_gradients"]),
            },
        )
        envelope_sha = _sha256_file(checkpoint_path)
        live_sha = state_sha256(scale.state_dict())
        if meta["state_dict_sha256"] != live_sha:
            raise _fail(f"seed {seed}: fold {heldout} envelope state digest differs from the trained head")
        restored_scale = init_n4_scale_model(128, seed + 999999)
        restored_meta = load_n4_scale_checkpoint(
            checkpoint_path, restored_scale, expected_fit_scope=scope,
            expected_training_history_ids=training_ids,
            expected_bounds={"v_floor": float(bounds["v_floor"]), "v_cap": float(bounds["v_cap"])},
        )
        if restored_meta["state_dict_sha256"] != live_sha or state_sha256(restored_scale.state_dict()) != live_sha:
            raise _fail(f"seed {seed}: fold {heldout} strict restore differs from the trained head")
        probe = sorted(heldout_pool, key=lambda record: (record.file_id, record.patch_index))[:64]
        forward_diff = 0.0
        if probe:
            forward_diff, live_nll, restored_nll = _probe_scale_parity(
                live_scale=scale, restored_scale=restored_scale, probe=probe,
                v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
            )
            if forward_diff != 0.0:
                raise _fail(f"seed {seed}: fold {heldout} strict restore forward parity failed")
            if live_nll != restored_nll:
                raise _fail(f"seed {seed}: fold {heldout} strict restore NLL parity failed")
        certified = next(
            (
                record
                for record in certified_folds
                if int(record["model_seed"]) == int(seed)
                and str(record["heldout_history_id"]) == heldout
            ),
            None,
        )
        live_identity = {
            "model_seed": int(seed),
            "fold_index": int(index),
            "scope": str(scope),
            "heldout_history_id": str(heldout),
            "training_history_ids": list(training_ids),
            "checkpoint_schema": str(meta["schema"]),
            "checkpoint_parameterization": str(meta["parameterization"]),
            "scale_parameterization": str(result["scale_parameterization"]),
            "guard_contract": str(result["guard_contract"]),
            "scale_parameters": int(sum(parameter.numel() for parameter in scale.parameters())),
            "record_count": int(result["record_count"]),
            "heldout_count": int(result["heldout_count"]),
            "optimizer_steps": int(result["optimizer_steps"]),
            "per_epoch_steps": int(result["per_epoch_steps"]),
            "bounds": {
                "v_floor": float(bounds["v_floor"]),
                "v_cap": float(bounds["v_cap"]),
            },
            "quantiles": {"q05": q05, "q95": q95},
            "init_anchor": dict(result["init_anchor"]),
            "init_interiority": dict(result["init_interiority"]),
            "first_step_gradients": dict(result["first_step_gradients"]),
            "epoch_order_seeds": list(result["epoch_order_seeds"]),
            "order_seed_rule": str(result["order_seed_rule"]),
            "cap_gate_pass_train": bool(result["cap_gate_pass_train"]),
        }
        authenticated = None
        if normative:
            if certified is None:
                raise _fail(
                    f"seed {seed}: fold {heldout} has no certified historical fold record; "
                    "capacity evidence does not transfer"
                )
            authenticated = _authenticate_certified_fold(
                certified=certified, seed=seed, heldout=heldout
            )
            _require_identical_certified_fold(
                seed=seed,
                heldout=heldout,
                live_identity=live_identity,
                historical_identity=authenticated["identity"],
                live_payload_sha=live_sha,
                historical_payload_sha=authenticated["payload_sha256"],
            )
        folds.append(
            {
                "fold_index": index,
                "heldout_history_id": heldout,
                "training_history_ids": list(training_ids),
                "record_count": int(result["record_count"]),
                "optimizer_steps": int(result["optimizer_steps"]),
                "scale_state_sha256": live_sha,
                "scale_checkpoint_path": str(checkpoint_path),
                "scale_checkpoint_sha256": envelope_sha,
                "restored_state_dict_sha256": state_sha256(restored_scale.state_dict()),
                "restored_forward_max_abs_diff": forward_diff,
                "bounds": {key: float(bounds[key]) for key in ("q05", "q95", "v_floor", "v_cap")},
                "final_train_nll": float(result["final_train_nll"]),
                "heldout_nll": float(result["heldout_nll"]),
                "heldout_spearman": float(result["heldout_spearman"]),
                "cap_gate_pass_train": True,
                "certified_container_sha256": (
                    str(certified["checkpoint_sha256"]) if certified is not None else None
                ),
                "certified_container_matches": (
                    envelope_sha == certified["checkpoint_sha256"]
                    if certified is not None
                    else None
                ),
                "certified_payload_sha256": (
                    str(certified["scale_state_sha256"]) if certified is not None else None
                ),
                "certified_payload_matches": (
                    live_sha == certified["scale_state_sha256"]
                    if certified is not None and normative
                    else None
                ),
                "certified_custody_authenticated": bool(authenticated is not None),
                "certified_identity_fields": list(_CERTIFIED_FOLD_IDENTITY_FIELDS),
                "certified_comparison": (
                    capacity.CAPACITY_IDENTITY_CONTRACT
                    if normative
                    else "not_compared_nonnormative"
                ),
            }
        )
    return {
        "schema": "s18-t30-n4-capacity-recertification-v2",
        "task_id": TASK_ID, "attempt_id": ATTEMPT, "agent_id": AGENT,
        "model_seed": seed,
        "normative_scope": bool(normative),
        "folds": folds,
        "folds_evaluated": len(folds),
        "all_fold_cap_gates_pass": all(fold["cap_gate_pass_train"] for fold in folds),
        "all_fold_payloads_match_certified_task13b": (
            all(fold["certified_payload_matches"] for fold in folds) if normative else None
        ),
        "all_fold_custody_authenticated": (
            all(fold["certified_custody_authenticated"] for fold in folds) if normative else None
        ),
        "recertified_containers_equal_certified": (
            all(fold["certified_container_matches"] for fold in folds) if normative else None
        ),
        "container_equality_is_observation_only": True,
        "executed_source_note": (
            "capacity is recertified on the executed changed source; the Task13B A02 PASS is "
            "executed-snapshot evidence and is not transferred without this re-execution"
        ),
    }


def _save_and_restore_coherent(*, model, config, scale, optimizer, arm, binding, seed, records,
                               fit_history_ids, folds, bounds, fit_result, paired_b0_sha256,
                               paired_b0_step, device, out_dir, patchifier) -> dict:
    """Write and strict-restore the coherent final N4-CRN artifact.

    The artifact is the next-stage consumer contract for Task31/Task33: the
    unchanged paired B0 online/EMA bytes, the fitted v3 scale head, the three
    recertified fold digests, the exact certified in-scope record counts, the
    Fit-only bounds and the frozen parameterization/guard identities, all
    joined under ``sprint18-coherent-checkpoint-v2``. Representation steps are
    exactly zero and the optimizer scope is the separate N4 scale optimizer.
    """
    training_ids = list(sorted(fit_history_ids))
    canonical_ids = sorted(
        n4_record_id(record.history_id, record.file_id, int(record.patch_index))
        for record in records
    )
    if len(canonical_ids) != FIT_SCALE_RECORDS:
        raise _fail(f"seed {seed}: coherent artifact record identity differs from the certified scope")
    sampler_seed, epoch_order = deterministic_sampler_order(
        binding, arm, workload="n4_scale", scope="experiment",
        role_history_ids=training_ids, epoch=N4_SCALE_EPOCHS,
        eligible_source_ids=canonical_ids,
    )
    if len(epoch_order) != FIT_SCALE_RECORDS or set(epoch_order) != set(canonical_ids):
        raise _fail(f"seed {seed}: coherent artifact sampler order is not the full certified scope")
    fit_role = _fit_role_sha256(binding)
    sampler_state = {
        "scope": "experiment",
        "workload": "n4_scale",
        "role_history_ids": training_ids,
        "role_manifest_sha256": [binding.history(history_id).manifest_sha256 for history_id in training_ids],
        "eligible_source_ids": canonical_ids,
        "eligible_source_ids_sha256": _canonical_sha256(canonical_ids),
        "sampler_epoch_seed": sampler_seed,
        "epoch_order_ids": list(epoch_order),
        "epoch_order_ids_sha256": _canonical_sha256(list(epoch_order)),
        "eligible_rows": len(canonical_ids),
        "batch_size": N4_SCALE_BATCH_SIZE,
        "batches_per_epoch": math.ceil(len(canonical_ids) / N4_SCALE_BATCH_SIZE),
        "completed_epochs": N4_SCALE_EPOCHS,
        "current_epoch": N4_SCALE_EPOCHS,
        "batches_in_current_epoch": 0,
        "fit_role_sha256": fit_role,
    }
    global_step = int(fit_result["optimizer_steps"])
    scheduler_state = {
        "scheduler_id": "sprint18-n4-scale-epoch-v2",
        "completed_epochs": N4_SCALE_EPOCHS,
        "active_epoch": N4_SCALE_EPOCHS,
        "learning_rate": 1e-3,
        "global_step": global_step,
    }
    coefficient_state = {
        "epoch": N4_SCALE_EPOCHS,
        "global_step": global_step,
        "selection_state": "locked",
        "lambda_info": None,
        "lambda_hcc": None,
        "alpha": None,
        "beta": None,
    }
    scale_initialization = {
        "init_anchor": dict(fit_result["init_anchor"]),
        "init_interiority": dict(fit_result["init_interiority"]),
        "first_step_gradients": dict(fit_result["first_step_gradients"]),
    }
    scale_state = {
        name: value.detach().cpu().clone() for name, value in scale.state_dict().items()
    }
    arm_state = {
        "paired_b0_checkpoint_sha256": paired_b0_sha256,
        "paired_b0_representation_step": int(paired_b0_step),
        "paired_b0_model_state_sha256": state_sha256(model.state_dict()),
        "paired_b0_ema_state_sha256": state_sha256(model.target_encoder.state_dict()),
        "scale_model_state": scale_state,
        "scale_model_state_sha256": state_sha256(scale_state),
        "diagnostic_folds": [
            {
                "heldout_history_id": fold["heldout_history_id"],
                "training_history_ids": list(fold["training_history_ids"]),
                "record_count": int(fold["record_count"]),
                "optimizer_steps": int(fold["optimizer_steps"]),
                "scale_state_sha256": fold["scale_state_sha256"],
            }
            for fold in folds["folds"]
        ],
        "final_fit_record_count": int(fit_result["record_count"]),
        "final_scale_steps": global_step,
        "scale_bounds": {
            "q05": float(bounds["q05"]),
            "q95": float(bounds["q95"]),
            "v_floor": float(bounds["v_floor"]),
            "v_cap": float(bounds["v_cap"]),
            "fit_role_sha256": fit_role,
            "residual_records_sha256": bounds["residual_records_sha256"],
        },
        "scale_parameterization": N4_SCALE_PARAMETERIZATION,
        "guard_contract": N4_GUARD_CONTRACT,
        "scale_initialization": scale_initialization,
    }
    checkpoint_path = Path(out_dir) / "checkpoints" / f"n4-seed{seed}-coherent.pt"
    digest = save_sprint18_checkpoint(
        checkpoint_path, model, optimizer, arm=arm, binding=binding,
        global_step=global_step, completed_epoch=N4_SCALE_EPOCHS,
        initialization_sha256=_canonical_sha256(scale_initialization),
        scheduler_state=scheduler_state, sampler_state=sampler_state,
        coefficient_state=coefficient_state, scope="experiment",
        representation_step=0, optimizer_scope="n4_scale",
        ema_updates=int(paired_b0_step), paired_b0_checkpoint_sha256=paired_b0_sha256,
        arm_specific_state=arm_state,
    )
    restored_model = V1RepresentationModel(config, patchifier=patchifier).to(device)
    restored_scale = init_n4_scale_model(128, seed + 999999).to(device)
    restored = restore_sprint18_checkpoint(
        checkpoint_path, restored_model, None, arm=arm, binding=binding,
        allowed_scope="experiment", arm_modules={"scale_model": restored_scale},
    )
    if restored.checkpoint_sha256 != digest:
        raise _fail(f"seed {seed}: coherent artifact restore digest differs from the saved bytes")
    if restored.representation_step != 0 or restored.optimizer_scope != "n4_scale":
        raise _fail(f"seed {seed}: coherent artifact representation/optimizer scope changed")
    if restored.paired_b0_checkpoint_sha256 != paired_b0_sha256:
        raise _fail(f"seed {seed}: coherent artifact paired B0 binding changed")
    model_sha = state_sha256(model.state_dict())
    ema_sha = state_sha256(model.target_encoder.state_dict())
    if state_sha256(restored_model.state_dict()) != model_sha:
        raise _fail(f"seed {seed}: coherent artifact strict restore changed the B0 model bytes")
    if state_sha256(restored_model.target_encoder.state_dict()) != ema_sha:
        raise _fail(f"seed {seed}: coherent artifact strict restore changed the B0 EMA bytes")
    if state_sha256(restored_scale.state_dict()) != arm_state["scale_model_state_sha256"]:
        raise _fail(f"seed {seed}: coherent artifact strict restore changed the scale-model bytes")
    restored_model.eval()
    restored_scale.eval()
    return {
        "schema": COHERENT_CHECKPOINT_SCHEMA,
        "task_id": TASK_ID, "attempt_id": ATTEMPT, "agent_id": AGENT,
        "model_seed": seed,
        "path": str(checkpoint_path),
        "sha256": digest,
        "checkpoint_schema": CHECKPOINT_SCHEMA_ID,
        "arm_id": ARM_ID,
        "arm_config_sha256": arm_config_sha256(arm),
        "global_step": global_step,
        "completed_epoch": N4_SCALE_EPOCHS,
        "representation_step": 0,
        "optimizer_scope": "n4_scale",
        "ema_updates": int(paired_b0_step),
        "paired_b0_checkpoint_sha256": paired_b0_sha256,
        "paired_b0_representation_step": int(paired_b0_step),
        "model_state_sha256": model_sha,
        "ema_state_sha256": ema_sha,
        "scale_model_state_sha256": arm_state["scale_model_state_sha256"],
        "restored_model_state_sha256": state_sha256(restored_model.state_dict()),
        "restored_ema_state_sha256": state_sha256(restored_model.target_encoder.state_dict()),
        "restored_scale_model_state_sha256": state_sha256(restored_scale.state_dict()),
        "restored_checkpoint_sha256": restored.checkpoint_sha256,
        "strict_restore_b0_model_ema_scale_equal": True,
        "sampler_eligible_rows": len(canonical_ids),
        "sampler_batches_per_epoch": sampler_state["batches_per_epoch"],
        "sampler_epoch_seed": sampler_seed,
        "diagnostic_fold_count": len(arm_state["diagnostic_folds"]),
        "final_fit_record_count": int(fit_result["record_count"]),
        "final_scale_steps": global_step,
        "_restored_model": restored_model,
        "_restored_scale": restored_scale,
        "_restored_checkpoint": restored,
    }


def _restore_b0(*, fixture, seed: int, b0_record: dict, binding, legacy_b0: dict,
                device: torch.device, patchifier):
    key = str(seed)
    restore_identity = _b0_restore_identity(fixture=fixture, seed=seed, b0_record=b0_record)
    identity = restore_identity["identity"]
    cache_head = restore_identity["cache_head"]
    cache_records = restore_identity["cache_records"]
    checkpoint_path = Path(restore_identity["checkpoint_path"])
    config = _config_from_cache(cache_head)
    model = V1RepresentationModel(config, patchifier=patchifier)
    arm_b0 = _b0_arm(seed, legacy_b0["coefficient_lock_sha256"])
    if arm_config_sha256(arm_b0) != legacy_b0["config_sha256"][key]:
        raise _fail(f"seed {seed}: B0 arm config differs from the historical v1 lock")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    restored = restore_sprint18_checkpoint(
        checkpoint_path, model, optimizer, arm=arm_b0, binding=binding, allowed_scope="experiment",
    )
    if restored.checkpoint_sha256 != identity["checkpoint_sha256"]:
        raise _fail(f"seed {seed}: strict B0 restore digest mismatch")
    if state_sha256(model.state_dict()) != fixture["live_model_state_sha256"][key]:
        raise _fail(f"seed {seed}: restored online bytes differ from the Task23 live digest")
    if state_sha256(model.target_encoder.state_dict()) != fixture["ema_state_sha256"][key]:
        raise _fail(f"seed {seed}: restored EMA bytes differ from the Task23 live digest")
    model.to(device)
    model.eval()
    return {
        "model": model,
        "arm_b0": arm_b0,
        "restored": restored,
        "config": config,
        "cache_head": cache_head,
        "cache_records": cache_records,
        "identity": identity,
        "checkpoint_sha256": restored.checkpoint_sha256,
    }


def _model_state_snapshot(model) -> dict:
    return {
        key: (value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value)
        for key, value in model.state_dict().items()
    }


def _state_equal(model, snapshot: dict) -> bool:
    after = model.state_dict()
    if set(after) != set(snapshot):
        return False
    for key, value in snapshot.items():
        current = after[key]
        if isinstance(current, torch.Tensor) and isinstance(value, torch.Tensor):
            if not torch.equal(current.detach().cpu(), value):
                return False
        elif current != value:
            return False
    return True


def _cache_binding(*, fixture, seed: int, binding, bounds, scale_state_sha256: str) -> dict:
    expected = n4_cache_binding_digest(
        checkpoint_sha256=fixture["checkpoints"][str(seed)]["sha256"],
        residual_records_sha256=bounds["residual_records_sha256"],
        fit_role_sha256=_fit_role_sha256(binding),
        scale_state_sha256=scale_state_sha256,
        scale_parameterization=N4_SCALE_PARAMETERIZATION,
        bounds={key: bounds[key] for key in ("q05", "q95", "v_floor", "v_cap")},
    )
    observed = verify_n4_cache_binding(
        expected_digest=expected,
        checkpoint_sha256=fixture["checkpoints"][str(seed)]["sha256"],
        residual_records_sha256=bounds["residual_records_sha256"],
        fit_role_sha256=_fit_role_sha256(binding),
        scale_state_sha256=scale_state_sha256,
        scale_parameterization=N4_SCALE_PARAMETERIZATION,
        bounds={key: bounds[key] for key in ("q05", "q95", "v_floor", "v_cap")},
    )
    return {"digest": observed, "fit_role_sha256": _fit_role_sha256(binding)}


def _fit_bank(*, binding, arm_b0, model, fit_rows, patchifier, restored, seed: int, identity: dict) -> dict:
    previous_file_id = None
    for row in fit_rows:
        if previous_file_id is not None and row.file_id <= previous_file_id:
            raise _fail(
                f"seed {seed}: Fit bank rows are not in the accepted canonical "
                "file_id order (global ascending, unique)"
            )
        previous_file_id = row.file_id
    bank = fit_fit_reference_bank(
        binding, arm_b0, model, fit_rows, patchifier, checkpoint=restored, scope="experiment",
    )
    provenance = bank.provenance
    if provenance.source_rows != FIT_HEALTHY_ROWS or provenance.scope != "experiment":
        raise _fail(f"seed {seed}: Fit bank provenance differs from the frozen Fit support")
    if provenance.bank_sha256 != identity["bank_sha256"]:
        raise _fail(f"seed {seed}: refit Fit bank differs from the accepted B0 bank digest")
    if provenance.source_rows_sha256 != identity["bank_source_rows_sha256"]:
        raise _fail(f"seed {seed}: Fit bank source rows differ from the accepted B0 record")
    if provenance.checkpoint_sha256 != identity["checkpoint_sha256"]:
        raise _fail(f"seed {seed}: Fit bank checkpoint differs from the accepted B0 primary")
    return {
        "bank_sha256": provenance.bank_sha256,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "source_rows": provenance.source_rows,
        "source_rows_sha256": provenance.source_rows_sha256,
        "source_history_ids": list(provenance.source_history_ids),
        "matches_accepted_B0_bank": True,
        "bank": bank,
    }


def _n4_score_row(*, model, scale, row, history_id: str, seed: int, patchifier, bounds,
                  diagnostics: bool) -> dict:
    """Score one file through the public N4 consumer, optionally adding diagnostics.

    The declared ``raw``/``scale``/``normalized`` triple always comes from the
    committed public consumer ``n4_score_files``. The diagnostic pass re-runs
    the identical frozen forward to report bound-saturation fractions and the
    scale--residual rank correlation, and must reproduce the consumer triple
    exactly for that file; a mismatch fails closed, so the structural gate input
    can never drift from the declared score.
    """
    mask_seed = deterministic_mask_seed(seed, history_id, row.file_id)
    triples = n4_score_files(
        model=model, samples=[row.sample], scale_model=scale,
        v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
        patchifier=patchifier, masking_seed=mask_seed,
    )
    if set(triples) != {row.file_id}:
        raise _fail(f"seed {seed}: public N4 consumer returned {sorted(triples)} for one row")
    triple = triples[row.file_id]
    entry = {
        "file_id": row.file_id,
        "raw": float(triple["raw"]),
        "scale": float(triple["scale"]),
        "normalized": float(triple["normalized"]),
        "eligible_patches": None,
        "frac_at_floor": None,
        "frac_at_cap": None,
        "spearman_scale_residual": None,
    }
    if not diagnostics:
        return entry
    batch = collate_variable_files(
        [row.sample], patchifier, masking_config=getattr(model, "config", None),
        masking_seed=mask_seed,
    )
    clean = {key: value for key, value in batch.items() if key != "file_samples"}
    device = next(model.parameters()).device
    clean = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in clean.items()}
    model.eval()
    scale.eval()
    with torch.no_grad():
        output = model(clean)
        context = output["context_latents"].detach()
        target = output["target_latents"].detach()
        predicted = output["predicted_latents"].detach()
        requested = output["prediction_mask"]
        valid = output["patch_valid_mask"]
        eligible = requested & valid
        width = context.shape[-1]
        if width != scale.d_model:
            raise _fail(f"seed {seed}: diagnostic context width differs from the scale head")
        residuals = (
            (torch.nn.functional.layer_norm(target, (width,))
             - torch.nn.functional.layer_norm(predicted, (width,))).square().mean(dim=-1).detach()
        )
        scale_out = scale(
            context.detach().reshape(-1, width),
            v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
        )
        log_v = scale_out["log_v"].detach().reshape(context.shape[0], context.shape[1])
        scales = scale_out["v"].detach().reshape(context.shape[0], context.shape[1])
        rows = torch.nonzero(eligible[0], as_tuple=False).flatten().tolist()
        if not rows:
            raise _fail(f"seed {seed}: diagnostic file has no eligible masked valid patches")
        raw_mean = float(residuals[0, rows].mean().item())
        scale_mean = float(scales[0, rows].mean().item())
        normalized = residuals / (scales + 1e-6)
        norm_mean = float(normalized[0, rows].mean().item())
        saturation = n4_saturation(
            log_v[0, rows].detach().cpu(),
            v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
        )
        spearman = float(
            n4_spearman(
                scales[0, rows].detach().cpu().numpy(),
                residuals[0, rows].detach().cpu().numpy(),
            )
        )
    if (raw_mean, scale_mean, norm_mean) != (entry["raw"], entry["scale"], entry["normalized"]):
        raise _fail(f"seed {seed}: diagnostic pass differs from the public N4 consumer triple")
    entry.update(
        {
            "eligible_patches": len(rows),
            "frac_at_floor": float(saturation["frac_at_floor"]),
            "frac_at_cap": float(saturation["frac_at_cap"]),
            "spearman_scale_residual": spearman,
        }
    )
    return entry


def _n4_score_rows(*, model, scale, rows, history_id: str, seed: int, patchifier, bounds,
                   diagnostic_ids: set[str] | None) -> dict:
    """Score one support through the public consumer, adding diagnostics where requested."""
    file_rows: list[dict] = []
    for row in rows:
        entry = _n4_score_row(
            model=model, scale=scale, row=row, history_id=history_id, seed=seed,
            patchifier=patchifier, bounds=bounds,
            diagnostics=diagnostic_ids is None or row.file_id in diagnostic_ids,
        )
        file_rows.append(entry)
    return {"rows": file_rows}


def _score_metrics(*, file_scores: dict, rows_by_history, ledgers, windows, threshold: float) -> dict:
    canonical = P.evaluate_histories(file_scores, rows_by_history, ledgers, windows, threshold)
    per_history = []
    for index, entry in enumerate(canonical["per_history"]):
        per_history.append(
            {
                "P_AUROC": entry["auc_p"], "W_AUROC": entry["auc_w"],
                "PW_AUROC": entry["auc_pw"], "A_AUROC": entry["auc_a"],
                "P_recall": entry["recall_p"], "P_median_lead_days": entry["lead_p_median"],
                "W_recall": entry["recall_w"], "W_median_lead_days": entry["lead_w_median"],
                "FAR_per_robot_day": entry["far"], "robot_days": entry["robot_days"],
                "false_episodes": entry["false_episodes"],
                "n_p": entry["n_p"], "n_w": entry["n_w"],
            }
        )
    return {
        "threshold": float(threshold),
        "macro": {
            "P_macro": canonical["macro_auc_p"], "W_macro": canonical["macro_auc_w"],
            "PW_macro": canonical["macro_auc_pw"], "PW_lcb95": canonical["lcb_pw"],
            "directional_histories": canonical.get("directional_histories"),
        },
        "per_history": per_history,
    }


def _calibration_proof(*, binding, arm_b0, model, scale, bank, rows, patchifier, restored,
                       seed: int, calibration_history_id: str, identity: dict, b0_record: dict,
                       bounds, normative: bool) -> dict:
    scores = score_certified_rows(
        binding, arm_b0, model, bank, rows, patchifier,
        checkpoint=restored, action="calibrate_q95", scope="experiment",
    )
    validate_score_maps(scores.support, scores.scores)
    if scores.support.support_sha256 != identity["calibration_source_support_sha256"]:
        raise _fail(f"seed {seed}: recomputed Calibration support differs from the accepted B0 record")
    recorded = b0_record["calibration_scores"][calibration_history_id]
    recorded_checkpoint = recorded["checkpoint_sha256"]
    if recorded_checkpoint != identity["checkpoint_sha256"]:
        raise _fail(f"seed {seed}: accepted B0 Calibration scores bind a different checkpoint")
    for branch in ("S_pred", "S_pop"):
        observed = {file_id: float(value) for file_id, value in dict(scores.scores[branch]).items()}
        if observed != {file_id: float(value) for file_id, value in recorded[branch].items()}:
            raise _fail(f"seed {seed}: recomputed B0 Calibration {branch} map differs from the accepted record")
    calibration = fit_calibration_q95(binding, scores, rows, scope="experiment")
    if calibration.S_pred.threshold != identity["calibration_S_pred_threshold"]:
        raise _fail(f"seed {seed}: recomputed B0 S_pred q95 differs from the accepted record")
    if calibration.S_pop.threshold != identity["calibration_S_pop_threshold"]:
        raise _fail(f"seed {seed}: recomputed B0 S_pop q95 differs from the accepted record")
    diagnostics = _n4_score_rows(
        model=model, scale=scale, rows=rows, history_id=calibration_history_id,
        seed=seed, patchifier=patchifier, bounds=bounds, diagnostic_ids=None,
    )
    normalized = {row["file_id"]: float(row["normalized"]) for row in diagnostics["rows"]}
    if set(normalized) != {row.file_id for row in rows}:
        raise _fail(f"seed {seed}: N4 Calibration scoring coverage differs from the healthy support")
    n4_threshold = n4_calibration_q95(normalized)
    total_patches = sum(int(row["eligible_patches"]) for row in diagnostics["rows"])
    cap_fraction = (
        sum(int(row["eligible_patches"]) * float(row["frac_at_cap"]) for row in diagnostics["rows"])
        / total_patches
    )
    correlation_mean = sum(float(row["spearman_scale_residual"]) for row in diagnostics["rows"]) / len(
        diagnostics["rows"]
    )
    return {
        "support": {
            "history_id": calibration_history_id,
            "source_rows": len(rows),
            "support_sha256": scores.support.support_sha256,
            "matches_accepted_B0_support": True,
        },
        "b0_path_parity": {
            "S_pred_map_sha256": _canonical_sha256(dict(scores.scores["S_pred"])),
            "S_pop_map_sha256": _canonical_sha256(dict(scores.scores["S_pop"])),
            "S_pred_threshold": calibration.S_pred.threshold,
            "S_pop_threshold": calibration.S_pop.threshold,
            "per_file_maps_equal_to_accepted_B0": True,
            "thresholds_equal_to_accepted_B0": True,
        },
        "n4_thresholds": {
            "S_pred": float(n4_threshold),
            "S_pop": float(calibration.S_pop.threshold),
        },
        "n4_calibration": {
            "normative_scope": bool(normative),
            "files": len(diagnostics["rows"]),
            "eligible_patches": total_patches,
            "frac_at_cap": cap_fraction,
            "cap_gate_pass": cap_fraction <= N4_SATURATION_CAP_GATE,
            "spearman_scale_residual_mean_per_file": correlation_mean,
        },
    }


def _development_proof(*, binding, arm_b0, model, scale, bank, development_rows: dict,
                       development_history_ids, binding_worktree: Path, patchifier, restored,
                       seed: int, b0_record: dict, calibration: dict, bounds, smoke_limit: int | None) -> dict:
    rows_by_history = []
    ledgers = []
    windows = []
    b0_maps: dict[str, dict[str, dict[str, float]]] = {"S_pred": {}, "S_pop": {}}
    n4_maps: dict[str, dict[str, float]] = {}
    support_hashes: dict[str, str] = {}
    per_history_raw_scale: dict[str, dict] = {}
    saturation = {"eligible_patches": 0, "cap_patches": 0, "floor_patches": 0}
    tier_accumulator: dict[tuple[str, str], list[float]] = {}
    correlation_values: list[float] = []
    for history_id in development_history_ids:
        history, manifest, manifest_path, _shard_paths = _role_assets(binding, binding_worktree, history_id)
        rows = development_rows[history_id]
        if smoke_limit is not None:
            rows = sorted(rows, key=lambda row: row.file_id)[:smoke_limit]
        recorded_support = b0_record["development"]["support"]["per_history"][history_id]["support_sha256"]
        scores = score_certified_rows(
            binding, arm_b0, model, bank, rows, patchifier,
            checkpoint=restored, action="score_development", scope="experiment",
        )
        validate_score_maps(scores.support, scores.scores)
        if smoke_limit is None and scores.support.support_sha256 != recorded_support:
            raise _fail(f"seed {seed}: Development support differs from the accepted B0 record for {history_id}")
        support_hashes[history_id] = scores.support.support_sha256
        for branch in ("S_pred", "S_pop"):
            b0_maps[branch].update({file_id: float(value) for file_id, value in dict(scores.scores[branch]).items()})
        normal_ids = {
            row["file_id"] for row in manifest["files"] if row.get("file_label") == "normal"
        }
        robot_program = {
            row["file_id"]: (str(row.get("robot_id")), str(row.get("program_id")))
            for row in manifest["files"]
        }
        scored = _n4_score_rows(
            model=model, scale=scale, rows=rows, history_id=history_id,
            seed=seed, patchifier=patchifier, bounds=bounds, diagnostic_ids=normal_ids,
        )
        raw_values = [float(entry["raw"]) for entry in scored["rows"]]
        scale_values = [float(entry["scale"]) for entry in scored["rows"]]
        norm_values = [float(entry["normalized"]) for entry in scored["rows"]]
        n4_maps.update({entry["file_id"]: float(entry["normalized"]) for entry in scored["rows"]})
        per_history_raw_scale[history_id] = {
            "files": len(scored["rows"]),
            "raw_mean": sum(raw_values) / len(raw_values),
            "scale_mean": sum(scale_values) / len(scale_values),
            "normalized_mean": sum(norm_values) / len(norm_values),
        }
        for entry in scored["rows"]:
            if entry["eligible_patches"] is None:
                # Abnormal files are scored through the consumer but do not enter
                # the healthy-patch structural gate.
                continue
            eligible = int(entry["eligible_patches"])
            saturation["eligible_patches"] += eligible
            saturation["cap_patches"] += eligible * float(entry["frac_at_cap"])
            saturation["floor_patches"] += eligible * float(entry["frac_at_floor"])
            correlation_values.append(float(entry["spearman_scale_residual"]))
            robot_id, program_id = robot_program.get(entry["file_id"], ("unknown", "unknown"))
            tier_accumulator.setdefault((robot_id, program_id), []).append(float(entry["scale"]))
        rows_by_history.append(list(manifest["files"]))
        ledgers.append(E.failure_ledger(manifest))
        windows.append(manifest["maintenance_windows"])
    saturation_frac_cap = (
        saturation["cap_patches"] / saturation["eligible_patches"] if saturation["eligible_patches"] else 0.0
    )
    saturation_frac_floor = (
        saturation["floor_patches"] / saturation["eligible_patches"] if saturation["eligible_patches"] else 0.0
    )
    thresholds = calibration["n4_thresholds"]
    if smoke_limit is None:
        branches = {
            "S_pred": _score_metrics(
                file_scores=n4_maps, rows_by_history=rows_by_history, ledgers=ledgers,
                windows=windows, threshold=float(thresholds["S_pred"]),
            ),
            "S_pop": _score_metrics(
                file_scores=b0_maps["S_pop"], rows_by_history=rows_by_history, ledgers=ledgers,
                windows=windows, threshold=float(thresholds["S_pop"]),
            ),
        }
    else:
        # A bounded slice cannot support the frozen metric code (control windows
        # reference unscored files), so the smoke path records plumbing only.
        branches = {
            branch: {
                "threshold": float(thresholds[branch]),
                "macro": None,
                "per_history": [],
                "evaluated": False,
                "note": "bounded non-normative smoke slice: metrics are not evaluated",
            }
            for branch in ("S_pred", "S_pop")
        }
    b0_reference = {
        "S_pred": {
            "threshold": b0_record["development"]["branches"]["S_pred"]["threshold"],
            "macro": dict(b0_record["development"]["branches"]["S_pred"]["macro"]),
            "per_history": [dict(row) for row in b0_record["development"]["branches"]["S_pred"]["per_history"]],
        },
        "S_pop": {
            "threshold": b0_record["development"]["branches"]["S_pop"]["threshold"],
            "macro": dict(b0_record["development"]["branches"]["S_pop"]["macro"]),
            "per_history": [dict(row) for row in b0_record["development"]["branches"]["S_pop"]["per_history"]],
        },
    }
    b0_reproduction: dict[str, object] = {"scope": "complete_support" if smoke_limit is None else "bounded_slice"}
    if smoke_limit is None:
        # Strongest available parity proof: recomputing the unchanged B0 path on
        # this snapshot must reproduce the accepted B0 Development metrics exactly
        # for BOTH branches (metrics are deterministic functions of the score maps).
        for branch in ("S_pred", "S_pop"):
            recomputed = _score_metrics(
                file_scores=b0_maps[branch], rows_by_history=rows_by_history, ledgers=ledgers,
                windows=windows,
                threshold=float(b0_reference[branch]["threshold"]),
            )
            if recomputed["macro"] != b0_reference[branch]["macro"]:
                raise _fail(f"seed {seed}: recomputed B0 {branch} macro metrics differ from the accepted record")
            recorded_rows = {row["history_id"]: row for row in b0_reference[branch]["per_history"]}
            for index, row in enumerate(recomputed["per_history"]):
                history_id = list(development_history_ids)[index]
                expected = recorded_rows.get(history_id)
                if expected is None:
                    raise _fail(f"seed {seed}: accepted B0 record lacks {branch} metrics for {history_id}")
                for name in PAIRED_HISTORY_METRICS:
                    if float(row[name]) != float(expected[name]):
                        raise _fail(
                            f"seed {seed}: recomputed B0 {branch} {history_id}.{name} differs from the accepted record"
                        )
        b0_reproduction["macro_equal"] = True
        b0_reproduction["per_history_equal"] = True
    parity = {
        "S_pop_bitwise_equal_to_B0": True,
        "scope": "complete Development support" if smoke_limit is None else "bounded non-normative slice",
        "S_pop_map_sha256": _canonical_sha256(b0_maps["S_pop"]),
        "S_pred_raw_map_sha256": _canonical_sha256(b0_maps["S_pred"]),
        "S_pop_threshold": float(thresholds["S_pop"]),
        "B0_reference_threshold": float(b0_reference["S_pop"]["threshold"]),
        "thresholds_equal": float(thresholds["S_pop"]) == float(b0_reference["S_pop"]["threshold"]),
        "B0_reproduction": b0_reproduction,
        "proof": (
            "N4 declares no S_pop mechanism: the branch is the unchanged public B0 inference on the "
            "same restored checkpoint, refit Fit bank and per-file mask seeds. Parity is proven by "
            "exact reproduction of the accepted B0 Calibration per-file maps/thresholds, exact "
            "reproduction of the accepted B0 Development metrics for both branches, unchanged "
            "model/EMA/bank digests across the N4 fit and scoring, and a bounded post-N4 re-derivation."
        ),
    }
    return {
        "support": {
            "evaluation_histories": list(development_history_ids),
            "per_history_support_sha256": support_hashes,
        },
        "branches": branches,
        "b0_reference": b0_reference,
        "b0_reproduction": b0_reproduction,
        "s_pop_parity": parity,
        "per_history_raw_scale_normalized": per_history_raw_scale,
        "saturation": {
            "scope": (
                "Development healthy files: every eligible masked patch of every normal-label file "
                "(the structural v_cap gate input); abnormal files are scored but not gated"
            ),
            "eligible_patches": saturation["eligible_patches"],
            "frac_at_cap": saturation_frac_cap,
            "frac_at_floor": saturation_frac_floor,
            "cap_gate_pass": saturation_frac_cap <= N4_SATURATION_CAP_GATE,
            "spearman_scale_residual_mean_per_file": (
                sum(correlation_values) / len(correlation_values) if correlation_values else None
            ),
        },
        "support_tiers": {
            f"{robot_id}|{program_id}": {
                "files": len(values),
                "scale_mean": sum(values) / len(values),
            }
            for (robot_id, program_id), values in sorted(tier_accumulator.items())
        },
        "n4_maps": n4_maps,
        "b0_maps": b0_maps,
    }


def _compute_block(*, config, model, scale, arm_n4, arm_b0, binding, fit_rows, fit_history_ids,
                   probe_row, probe_history_id, bounds, device, patchifier, final_steps: int) -> dict:
    profile_device = "cuda" if device.type == "cuda" else "cpu"
    b0_model = V1RepresentationModel(config, patchifier=patchifier)
    b0_model.to(device)
    b0_model.eval()
    parameter_counts = {
        "S_pred": primary_inference_parameters(model, "S_pred", added_scorers=(scale,)),
        "S_pop": primary_inference_parameters(model, "S_pop"),
    }
    b0_parameter_counts = {
        "S_pred": primary_inference_parameters(b0_model, "S_pred"),
        "S_pop": primary_inference_parameters(b0_model, "S_pop"),
    }
    mask_seed = deterministic_mask_seed(arm_n4.model_seed, probe_history_id, probe_row.file_id)
    probe_batch = collate_variable_files(
        [probe_row.sample], patchifier, masking_config=config, masking_seed=mask_seed,
    )
    probe_input = _move(_encoder_batch(probe_batch), device)
    model.eval()
    b0_forward_flops = profile_inference_flops(lambda: b0_model(probe_input), device=profile_device)
    n4_pred_flops = profile_inference_flops(
        lambda: n4_score_files(
            model=model, samples=[probe_row.sample], scale_model=scale,
            v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
            patchifier=patchifier, masking_seed=mask_seed,
        ),
        device=profile_device,
    )
    flops_by_branch = {"S_pred": n4_pred_flops, "S_pop": b0_forward_flops}
    b0_flops_by_branch = {"S_pred": b0_forward_flops, "S_pop": b0_forward_flops}
    # N4 scale-fit training FLOPs: one real 50-epoch fit over a bounded real
    # 512-record scope on a disposable head, scaled to the frozen final scope.
    torch.manual_seed(11)
    probe_scale = init_n4_scale_model(128, arm_n4.model_seed)
    probe_records = list(fit_rows)[:512]
    probe_fit_flops = profile_inference_flops(
        lambda: fit_n4_scale_scope(
            records=probe_records, training_history_ids=list(sorted(fit_history_ids)),
            scope="final", arm=arm_n4, binding=binding, scale_model=probe_scale,
            v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]), device=device,
        ),
        device=profile_device,
    )
    probe_steps = N4_SCALE_EPOCHS * math.ceil(len(probe_records) / N4_SCALE_BATCH_SIZE)
    if probe_steps < 1:
        raise _fail("N4 compute probe produced no optimizer steps")
    n4_training_flops_per_fit_file = (probe_fit_flops / probe_steps) * final_steps / FIT_HEALTHY_ROWS
    # Paired B0 training FLOPs: one real 128-file Fit step on a disposable model.
    torch.manual_seed(11)
    prof_b0_model = V1RepresentationModel(config, patchifier=patchifier)
    prof_b0_optimizer = torch.optim.AdamW(
        prof_b0_model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=0.0,
    )
    prof_b0_trainer = RepresentationTrainer(
        prof_b0_model, optimizer=prof_b0_optimizer, max_grad_norm=1.0, device=device,
        seed=arm_b0.model_seed,
    )
    b0_batch = collate_variable_files(
        [row.sample for row in list(fit_rows)[:128]], patchifier,
        masking_config=config, masking_seed=1,
    )
    b0_step_flops = profile_inference_flops(
        lambda: prof_b0_trainer.train_sprint18_epoch(
            [b0_batch], arm=arm_b0, epoch=1, collect_balance=False,
        ),
        device=profile_device,
    )
    b0_steps = 100 * math.ceil(FIT_HEALTHY_ROWS / 128)
    b0_training_flops_per_fit_file = b0_step_flops * b0_steps / FIT_HEALTHY_ROWS
    training_ratio = n4_training_flops_per_fit_file / b0_training_flops_per_fit_file
    envelope = validate_compute_envelope(
        ARM_ID,
        parameters_by_branch=parameter_counts,
        flops_by_branch=flops_by_branch,
        b0_parameters_by_branch=b0_parameter_counts,
        b0_flops_by_branch=b0_flops_by_branch,
        training_flops_ratio=training_ratio,
        n4_scale_parameters=int(sum(parameter.numel() for parameter in scale.parameters())),
        n4_s_pop_bitwise_equal_to_b0=True,
    )
    return {
        "scope": COMPUTE_SCOPE,
        "parameters_inference": dict(parameter_counts),
        "paired_b0_parameters_inference": dict(b0_parameter_counts),
        "inference_flops_per_fit_file": dict(flops_by_branch),
        "paired_b0_inference_flops_per_fit_file": dict(b0_flops_by_branch),
        "n4_scale_parameters": int(sum(parameter.numel() for parameter in scale.parameters())),
        "n4_scale_fit_steps": int(final_steps),
        "n4_scale_fit_flops_total": float(probe_fit_flops / probe_steps * final_steps),
        "n4_training_flops_per_fit_file": float(n4_training_flops_per_fit_file),
        "b0_training_flops_per_fit_file": float(b0_training_flops_per_fit_file),
        "training_flops_ratio_vs_B0": float(training_ratio),
        "parameter_ratio_vs_B0": dict(envelope["parameter_ratio_vs_B0"]),
        "inference_flop_ratio_vs_B0": dict(envelope["inference_flop_ratio_vs_B0"]),
        "combined_inference_flops_ratio_vs_B0": float(envelope["combined_inference_flops_ratio_vs_B0"]),
        "envelope_pass": bool(envelope["envelope_pass"]),
        "limitations": (
            "one-file/one-batch operational estimates measured on the run device; the bank cdist "
            "population term is identical for both arms and is not separately profiled"
        ),
    }


def _paired_development(*, arm_records: dict, b0_records: dict, b0_records_sha256: str,
                        arm_id: str = ARM_ID) -> dict:
    expected_keys = [str(seed) for seed in MODEL_SEEDS]
    if sorted(arm_records) != expected_keys or sorted(b0_records) != expected_keys:
        raise _fail("paired Development requires exactly the frozen three model seeds")
    histories: list[str] | None = None
    per_history_support: dict[str, str] = {}
    calibration_support: dict[str, str] = {}
    for key in expected_keys:
        arm = arm_records[key]
        b0 = b0_records[key]
        if arm.get("arm_id") != arm_id or arm.get("status") != "ok":
            raise _fail(f"{arm_id} seed {key} record is not a completed run")
        if b0.get("arm_id") != BASELINE_ARM_ID or b0.get("status") != "ok":
            raise _fail(f"B0 seed {key} record is not a completed B0-FULL run")
        arm_support = arm["development"]["support"]
        b0_development_support = b0["development"]["support"]
        if arm_support["evaluation_histories"] != b0_development_support["evaluation_histories"]:
            raise _fail(f"paired Development history roster differs for seed {key}")
        if histories is None:
            histories = list(arm_support["evaluation_histories"])
        elif list(arm_support["evaluation_histories"]) != histories:
            raise _fail("Development history roster differs across seeds")
        for history_id in histories:
            arm_hash = arm_support["per_history_support_sha256"][history_id]
            b0_hash = b0_development_support["per_history"][history_id]["support_sha256"]
            if arm_hash != b0_hash:
                raise _fail(f"paired Development support differs for {history_id} (seed {key})")
            per_history_support.setdefault(history_id, arm_hash)
            if per_history_support[history_id] != arm_hash:
                raise _fail(f"Development support differs across seeds for {history_id}")
        if arm["calibration"]["support"]["support_sha256"] != b0["calibration"]["source_support_sha256"]:
            raise _fail(f"Calibration support differs from B0 for seed {key}")
        calibration_support[key] = arm["calibration"]["support"]["support_sha256"]
        if arm["bank"]["source_rows"] != b0["bank"]["source_rows"]:
            raise _fail(f"Fit bank source rows differ from B0 for seed {key}")
        if arm["bank"]["bank_sha256"] != b0["bank"]["bank_sha256"]:
            raise _fail(f"Fit bank digest differs from B0 for seed {key}")
    assert histories is not None
    branches: dict[str, object] = {}
    for branch in ("S_pred", "S_pop"):
        per_seed = []
        for key in expected_keys:
            arm_branch = arm_records[key]["development"]["branches"][branch]
            b0_branch = b0_records[key]["development"]["branches"][branch]
            arm_histories = {row["history_id"]: row for row in arm_branch["per_history"]}
            b0_histories = {row["history_id"]: row for row in b0_branch["per_history"]}
            if set(arm_histories) != set(histories) or set(b0_histories) != set(histories):
                raise _fail(f"{branch} per-history rows differ from the frozen Development roster")
            macro = {}
            for name in PAIRED_MACRO_METRICS:
                arm_value = arm_branch["macro"].get(name)
                b0_value = b0_branch["macro"].get(name)
                for value, source in ((arm_value, arm_id), (b0_value, "B0")):
                    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                        raise _fail(f"paired metric {branch}.{name} is missing or nonfinite in the {source} record")
                macro[name] = {
                    "arm": float(arm_value), "b0": float(b0_value),
                    "delta": float(arm_value) - float(b0_value),
                }
            per_history = []
            for history_id in histories:
                metrics = {}
                for name in PAIRED_HISTORY_METRICS:
                    arm_value = arm_histories[history_id].get(name)
                    b0_value = b0_histories[history_id].get(name)
                    if arm_value is None or b0_value is None:
                        raise _fail(f"paired metric {branch}.{history_id}.{name} is missing")
                    metrics[name] = {
                        "arm": float(arm_value), "b0": float(b0_value),
                        "delta": float(arm_value) - float(b0_value),
                    }
                per_history.append({"history_id": history_id, "metrics": metrics})
            per_seed.append(
                {
                    "model_seed": int(key),
                    "arm_threshold": arm_branch["threshold"],
                    "b0_threshold": b0_branch["threshold"],
                    "macro": macro,
                    "per_history": per_history,
                }
            )
        aggregate_macro = {}
        for name in PAIRED_MACRO_METRICS:
            deltas = [row["macro"][name]["delta"] for row in per_seed]
            aggregate_macro[name] = {
                "mean_delta": sum(deltas) / len(deltas),
                "positive_seeds": sum(1 for delta in deltas if delta > 0),
            }
        aggregate_history = {}
        for history_id in histories:
            aggregate_history[history_id] = {}
            for name in PAIRED_HISTORY_METRICS:
                deltas = [
                    next(row for row in seed_row["per_history"] if row["history_id"] == history_id)
                    ["metrics"][name]["delta"]
                    for seed_row in per_seed
                ]
                aggregate_history[history_id][name] = {
                    "mean_delta": sum(deltas) / len(deltas),
                    "positive_seeds": sum(1 for delta in deltas if delta > 0),
                }
        branches[branch] = {
            "per_seed": per_seed,
            "aggregate": {"macro": aggregate_macro, "per_history": aggregate_history},
        }
    return {
        "schema": PAIRED_SCHEMA,
        "task_id": TASK_ID, "attempt_id": ATTEMPT, "agent_id": AGENT,
        "arm_id": arm_id, "baseline_arm_id": BASELINE_ARM_ID,
        "b0_seed_records_sha256": b0_records_sha256,
        "model_seeds": list(MODEL_SEEDS),
        "support": {
            "evaluation_histories": histories,
            "per_history_support_sha256": per_history_support,
            "calibration_support_sha256": calibration_support,
        },
        "branches": branches,
        "interpretation": (
            "Descriptive paired Development differences only (N4-CRN minus B0-FULL at the same model "
            "seed and frozen support). Each arm uses its own q95 thresholds; no selection, promotion, "
            "pruning, fusion, or rescue is applied. N4 declares no S_pop mechanism and its S_pop branch "
            "is the unchanged B0 path, so S_pop deltas are exactly zero by construction."
        ),
    }


def _environment_block(args) -> dict:
    environment = {
        "task_id": TASK_ID, "attempt_id": ATTEMPT, "agent_id": AGENT,
        "mode": args.mode, "device": args.device,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        environment.update(
            {
                "device_name": properties.name,
                "device_total_memory_bytes": int(properties.total_memory),
                "device_count": torch.cuda.device_count(),
            }
        )
    return environment


def run_preflight(args, bdoc: dict) -> dict:
    worktree = Path(args.worktree)
    checks = self_check(args, bdoc)
    _check_contract_files(worktree)
    lock = capacity.load_task14_lock(worktree)
    legacy_b0 = capacity.load_legacy_b0_lock(worktree)
    lock_binding = capacity.validate_task14_lock(
        lock, worktree=worktree,
        observed_runner_sha256=_sha256_file(_EXPERIMENTS_DIR / "sprint18_task13b_n4_capacity.py"),
    )
    binding = load_task6_binding(args.handoff)
    if binding.executable_binding_sha256 != capacity.EXPECTED_CONTRACT["executable_binding_sha256"]:
        raise _fail("executable certified-DATA binding digest changed")
    fixture = _validate_fixture(bdoc["fixture"])
    b0_records = _load_b0_records(Path(bdoc.get("b0_seed_records", B0_RECORDS_PATH)))
    support = _load_supports(binding, worktree, development=False)
    fit_history_ids = support["fit_history_ids"]
    certified_custody = _authenticate_certified_folds(
        capacity_certification=lock_binding["capacity_certification"],
        fit_history_ids=list(fit_history_ids),
    )
    lock_digest = lock["coefficient_lock_sha256"]
    seed_blocks: dict[str, dict] = {}
    for seed in MODEL_SEEDS:
        key = str(seed)
        restore_identity = _b0_restore_identity(
            fixture=fixture, seed=seed, b0_record=b0_records[key],
        )
        identity = restore_identity["identity"]
        cache_records = restore_identity["cache_records"]
        arm = _n4_arm(seed, lock_digest)
        observed_config = arm_config_sha256(arm)
        expected_config = lock["config_sha256"].get(f"{ARM_ID}/seed-{seed}")
        if observed_config != expected_config:
            raise _fail(f"seed {seed}: N4 arm config differs from the prospective Task14 lock")
        pools = _cache_pools(cache_records, fit_history_ids)
        bounds = _final_bounds(pools, fit_history_ids)
        canonical_ids = sorted(
            f"{record.history_id}|{record.file_id}|{int(record.patch_index):05d}"
            for history_id in fit_history_ids for record in pools[history_id]
        )
        sampler_checks = []
        for epoch in (1, N4_SCALE_EPOCHS):
            epoch_seed, order = deterministic_sampler_order(
                binding, arm, workload="n4_scale", scope="experiment",
                role_history_ids=list(sorted(fit_history_ids)), epoch=epoch,
                eligible_source_ids=canonical_ids,
            )
            if len(order) != FIT_SCALE_RECORDS or set(order) != set(canonical_ids):
                raise _fail(f"seed {seed}: final-scope sampler epoch {epoch} is not a full permutation")
            sampler_checks.append(
                {"epoch": epoch, "sampler_epoch_seed": epoch_seed,
                 "order_sha256": _canonical_sha256(list(order))}
            )
        record_counts = {history_id: len(pools[history_id]) for history_id in fit_history_ids}
        fold_counts = [
            sum(record_counts[history_id] for history_id in fit_history_ids if history_id != heldout)
            for heldout in fit_history_ids
        ]
        expected_fold_counts = [
            FIT_SCALE_RECORDS - FIT_HISTORY_RECORDS[heldout] for heldout in fit_history_ids
        ]
        if fold_counts != expected_fold_counts:
            raise _fail(f"seed {seed}: live fold counts differ from the certified scope: {fold_counts}")
        seed_blocks[key] = {
            "b0_identity": identity,
            "n4_arm_config_sha256": observed_config,
            "cache_records": len(cache_records),
            "record_counts": record_counts,
            "record_total": sum(record_counts.values()),
            "fold_record_counts": fold_counts,
            "fold_optimizer_steps": [n4_scale_scope_steps(count) for count in fold_counts],
            "final_bounds": bounds,
            "final_scale_steps": n4_scale_scope_steps(FIT_SCALE_RECORDS),
            "sampler_checks": sampler_checks,
        }
    scale_model = init_n4_scale_model(128, MODEL_SEEDS[0])
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "task_id": TASK_ID, "attempt_id": ATTEMPT, "agent_id": AGENT,
        "source_commit": checks["head"],
        "runner_sha256": checks["runner_sha256"],
        "binding_sha256": checks["binding_sha256"],
        "handoff_sha256": _sha256_file(Path(args.handoff)),
        "executable_binding_sha256": binding.executable_binding_sha256,
        "protocol_sha256": capacity.EXPECTED_CONTRACT["protocol_sha256"],
        "mechanism_spec_sha256": capacity.EXPECTED_CONTRACT["mechanism_N4-CRN"],
        "mechanism_spec_base_sha256": capacity.EXPECTED_CONTRACT["mechanism_N4-CRN-v2-base"],
        "scale_parameterization": N4_SCALE_PARAMETERIZATION,
        "guard_contract": N4_GUARD_CONTRACT,
        "scale_checkpoint_schema": N4_SCALE_CHECKPOINT_SCHEMA,
        "scale_head_parameters": int(sum(parameter.numel() for parameter in scale_model.parameters())),
        "task14_lock_binding": dict(lock_binding),
        "n4_scale_scope": dict(lock_binding["n4_scale_scope"]),
        "capacity_recertification": dict(lock_binding["capacity_certification"]),
        "capacity_custody_authentication": dict(certified_custody),
        "capacity_recertification_required_before_final_fit": True,
        "b0_legacy_lock_binding": dict(legacy_b0),
        "fit_role_sha256": _fit_role_sha256(binding),
        "fit_support": support["fit_records"],
        "calibration_support": support["calibration_records"],
        "development_support": support["development_records"],
        "b0_seed_records_sha256": B0_RECORDS_SHA256,
        "model_seeds": list(MODEL_SEEDS),
        "seeds": seed_blocks,
        "PREFLIGHT_OK": True,
    }
    return payload


def _probe_records(pools: dict[str, list], fit_history_ids: list[str], limit: int = 512) -> list:
    records = [record for history_id in fit_history_ids for record in pools[history_id]]
    return records[:limit]


def _run_seed(*, seed: int, fixture: dict, b0_record: dict, binding, legacy_b0: dict, lock: dict,
              support: dict, capacity_certification: dict, worktree: Path, out_dir: Path,
              device: torch.device, progress, seed_dir: Path, smoke: bool,
              development_limit: int | None) -> dict:
    run_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    patchifier = _patchifier()
    fit_history_ids = support["fit_history_ids"]
    _log_progress(progress, {"run": f"{RUN_ID_PREFIX}-seed{seed}", "event": "restore_b0", "model_seed": seed})
    restored_b0 = _restore_b0(
        fixture=fixture, seed=seed, b0_record=b0_record, binding=binding, legacy_b0=legacy_b0,
        device=device, patchifier=patchifier,
    )
    identity = restored_b0["identity"]
    model = restored_b0["model"]
    config = restored_b0["config"]
    snapshot = _model_state_snapshot(model)
    ema_snapshot = state_sha256(model.target_encoder.state_dict())
    arm_n4 = _n4_arm(seed, lock["coefficient_lock_sha256"])
    if arm_config_sha256(arm_n4) != lock["config_sha256"].get(f"{ARM_ID}/seed-{seed}"):
        raise _fail(f"seed {seed}: N4 arm config differs from the prospective Task14 lock")
    pools = _cache_pools(restored_b0["cache_records"], fit_history_ids)
    if smoke:
        pools = {history_id: list(pool)[:SMOKE_RECORDS_PER_HISTORY] for history_id, pool in pools.items()}
    bounds = _final_bounds(pools, fit_history_ids)
    if smoke and bounds["record_count"] != SMOKE_RECORDS_PER_HISTORY * len(fit_history_ids):
        raise _fail(f"seed {seed}: bounded smoke pool differs from {SMOKE_RECORDS_PER_HISTORY}/history")
    records = [record for history_id in fit_history_ids for record in pools[history_id]]
    certified_folds = _certified_fold_records(capacity_certification, seed=seed)
    certified_folds.sort(key=lambda fold: int(fold["fold_index"]))
    if [fold["heldout_history_id"] for fold in certified_folds] != list(fit_history_ids):
        raise _fail(f"seed {seed}: certified historical fold roster differs from the Fit roster")
    fold_normative = not smoke
    _log_progress(
        progress,
        {"run": f"{RUN_ID_PREFIX}-seed{seed}", "event": "capacity_folds", "model_seed": seed,
         "fold_records": [sum(len(pools[history_id]) for history_id in fit_history_ids if history_id != heldout)
                          for heldout in fit_history_ids]},
    )
    folds = _run_capacity_folds(
        pools=pools, fit_history_ids=fit_history_ids, arm=arm_n4, binding=binding,
        seed=seed, device=device, out_dir=seed_dir, certified_folds=certified_folds,
        normative=fold_normative,
    )
    if not folds["all_fold_cap_gates_pass"]:
        raise _fail(f"seed {seed}: capacity recertification fold gate failed")
    _log_progress(
        progress,
        {"run": f"{RUN_ID_PREFIX}-seed{seed}", "event": "fit_final", "model_seed": seed,
         "records": len(records), "v_floor": bounds["v_floor"], "v_cap": bounds["v_cap"]},
    )
    fit_start = time.time()
    if smoke:
        scale = init_n4_scale_model(128, seed).to(device)
        optimizer = torch.optim.AdamW(scale.parameters(), lr=1e-3, weight_decay=1e-4)
        result = fit_n4_scale_scope(
            records=records, training_history_ids=list(sorted(fit_history_ids)), scope="final",
            arm=arm_n4, binding=binding, scale_model=scale,
            v_floor=bounds["v_floor"], v_cap=bounds["v_cap"], device=device,
            optimizer=optimizer,
        )
        expected_steps = n4_scale_scope_steps(len(records))
        if result["optimizer_steps"] != expected_steps:
            raise _fail(f"seed {seed}: bounded smoke fit step accounting differs from 50*ceil(N/512)")
    else:
        scale, result, optimizer = _fit_final_scope(
            records=records, fit_history_ids=fit_history_ids, arm=arm_n4, binding=binding,
            seed=seed, bounds=bounds, device=device, out_dir=seed_dir,
        )
    fit_wall = time.time() - fit_start
    for epoch_index, nll in enumerate(result["per_epoch_nll"], start=1):
        _log_progress(
            progress,
            {"run": f"{RUN_ID_PREFIX}-seed{seed}", "event": "epoch", "model_seed": seed,
             "epoch": epoch_index, "nll": nll, "elapsed_s": time.time() - run_start},
        )
    _log_progress(progress, {"run": f"{RUN_ID_PREFIX}-seed{seed}", "event": "checkpoint", "model_seed": seed})
    scale_block = _save_and_restore_scale(
        scale=scale, out_dir=seed_dir, seed=seed, fit_history_ids=fit_history_ids,
        bounds=bounds, fit_result={**result, "probe_records": _probe_records(pools, fit_history_ids)},
    )
    coherent_block = None
    if not smoke:
        _log_progress(progress, {"run": f"{RUN_ID_PREFIX}-seed{seed}", "event": "coherent_checkpoint", "model_seed": seed})
        coherent_block = _save_and_restore_coherent(
            model=model, config=config, scale=scale, optimizer=optimizer, arm=arm_n4,
            binding=binding, seed=seed, records=records, fit_history_ids=fit_history_ids,
            folds=folds, bounds=bounds, fit_result=result,
            paired_b0_sha256=restored_b0["checkpoint_sha256"],
            paired_b0_step=100 * math.ceil(FIT_HEALTHY_ROWS / 128),
            device=device, out_dir=seed_dir, patchifier=patchifier,
        )
    cache_binding = _cache_binding(
        fixture=fixture, seed=seed, binding=binding, bounds=bounds,
        scale_state_sha256=state_sha256(scale.state_dict()),
    )
    _log_progress(progress, {"run": f"{RUN_ID_PREFIX}-seed{seed}", "event": "bank", "model_seed": seed})
    bank_block = _fit_bank(
        binding=binding, arm_b0=restored_b0["arm_b0"], model=model,
        fit_rows=support["fit_rows"], patchifier=patchifier,
        restored=restored_b0["restored"], seed=seed, identity=identity,
    )
    bank = bank_block.pop("bank")
    calibration_rows = support["calibration_rows"]
    _log_progress(progress, {"run": f"{RUN_ID_PREFIX}-seed{seed}", "event": "calibration", "model_seed": seed})
    calibration_start = time.time()
    calibration = _calibration_proof(
        binding=binding, arm_b0=restored_b0["arm_b0"], model=model, scale=scale, bank=bank,
        rows=calibration_rows, patchifier=patchifier, restored=restored_b0["restored"],
        seed=seed, calibration_history_id=support["calibration_history_ids"][0],
        identity=identity, b0_record=b0_record, bounds=bounds, normative=not smoke,
    )
    calibration_wall = time.time() - calibration_start
    _log_progress(progress, {"run": f"{RUN_ID_PREFIX}-seed{seed}", "event": "development", "model_seed": seed})
    development_start = time.time()
    development = _development_proof(
        binding=binding, arm_b0=restored_b0["arm_b0"], model=model, scale=scale,
        bank=bank, development_rows=support["development_rows"],
        development_history_ids=support["development_history_ids"],
        binding_worktree=worktree, patchifier=patchifier, restored=restored_b0["restored"],
        seed=seed, b0_record=b0_record, calibration=calibration, bounds=bounds,
        smoke_limit=development_limit,
    )
    development_wall = time.time() - development_start
    restored_artifact_proof = None
    if coherent_block is not None:
        _log_progress(progress, {"run": f"{RUN_ID_PREFIX}-seed{seed}", "event": "restored_artifact", "model_seed": seed})
        restored_calibration = _calibration_proof(
            binding=binding, arm_b0=restored_b0["arm_b0"],
            model=coherent_block["_restored_model"], scale=coherent_block["_restored_scale"],
            bank=bank, rows=calibration_rows, patchifier=patchifier,
            restored=restored_b0["restored"], seed=seed,
            calibration_history_id=support["calibration_history_ids"][0],
            identity=identity, b0_record=b0_record, bounds=bounds, normative=True,
        )
        if restored_calibration["n4_thresholds"] != calibration["n4_thresholds"]:
            raise _fail(f"seed {seed}: restored-artifact N4 q95 differs from the live N4 q95")
        if restored_calibration["b0_path_parity"] != calibration["b0_path_parity"]:
            raise _fail(f"seed {seed}: restored-artifact B0 parity differs from the live path")
        restored_slice: dict[str, float] = {}
        slice_support: dict[str, int] = {}
        for history_id in support["development_history_ids"]:
            selected = sorted(support["development_rows"][history_id], key=lambda row: row.file_id)[:2]
            scored = _n4_score_rows(
                model=coherent_block["_restored_model"], scale=coherent_block["_restored_scale"],
                rows=selected, history_id=history_id, seed=seed, patchifier=patchifier,
                bounds=bounds, diagnostic_ids=None,
            )
            for row in scored["rows"]:
                restored_slice[row["file_id"]] = float(row["normalized"])
            slice_support[history_id] = len(selected)
        live_slice = {file_id: float(development["n4_maps"][file_id]) for file_id in restored_slice}
        if restored_slice != live_slice:
            raise _fail(f"seed {seed}: restored-artifact public N4 scoring differs from the live path")
        restored_artifact_proof = {
            "schema": "s18-t30-n4-restored-artifact-proof-v1",
            "coherent_checkpoint_sha256": coherent_block["sha256"],
            "calibration_full_support_rescored": True,
            "calibration_n4_q95_equal_to_live": True,
            "b0_path_parity_equal_to_live": True,
            "development_bounded_files_per_history": slice_support,
            "development_public_scoring_equal_to_live": True,
            "s_pop_unchanged": True,
        }
    # Bounded post-N4 re-derivation: the unchanged B0 path must be bitwise stable
    # across the whole N4 fit/scoring pipeline.
    slice_rows = []
    slice_expected = {"S_pred": {}, "S_pop": {}}
    for history_id in support["development_history_ids"]:
        rows = sorted(support["development_rows"][history_id], key=lambda row: row.file_id)
        limit = development_limit if development_limit is not None else 2
        selected = rows[:limit]
        slice_rows.extend(selected)
        for file_id in (row.file_id for row in selected):
            for branch in ("S_pred", "S_pop"):
                if file_id not in development["b0_maps"][branch]:
                    raise _fail(f"seed {seed}: post-N4 parity slice file missing from the B0 map")
                slice_expected[branch][file_id] = development["b0_maps"][branch][file_id]
    rederived = {}
    for history_id in support["development_history_ids"]:
        selected = [row for row in slice_rows if row.actual_history_id == history_id]
        if not selected:
            continue
        scores = score_certified_rows(
            binding, restored_b0["arm_b0"], model, bank, selected, patchifier,
            checkpoint=restored_b0["restored"], action="score_development", scope="experiment",
        )
        for branch in ("S_pred", "S_pop"):
            rederived.setdefault(branch, {}).update(
                {file_id: float(value) for file_id, value in dict(scores.scores[branch]).items()}
            )
    for branch in ("S_pred", "S_pop"):
        if rederived.get(branch) != slice_expected[branch]:
            raise _fail(f"seed {seed}: post-N4 {branch} re-derivation differs from the pre-N4 path")
    model_unchanged = _state_equal(model, snapshot)
    ema_unchanged = state_sha256(model.target_encoder.state_dict()) == ema_snapshot
    bank_unchanged = bank.provenance.bank_sha256 == bank_block["bank_sha256"]
    if not (model_unchanged and ema_unchanged and bank_unchanged):
        raise _fail(f"seed {seed}: N4 fitting/scoring perturbed the frozen representation or bank")
    _log_progress(progress, {"run": f"{RUN_ID_PREFIX}-seed{seed}", "event": "compute", "model_seed": seed})
    probe_row = support["fit_rows"][0]
    compute = _compute_block(
        config=config, model=model, scale=scale, arm_n4=arm_n4, arm_b0=restored_b0["arm_b0"],
        binding=binding, fit_rows=support["fit_rows"], fit_history_ids=fit_history_ids,
        probe_row=probe_row, probe_history_id=probe_row.actual_history_id,
        bounds=bounds, device=device, patchifier=patchifier,
        final_steps=int(result["optimizer_steps"]),
    )
    if not compute["envelope_pass"]:
        raise _fail(f"seed {seed}: compute envelope did not pass")
    peak_allocated = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    peak_reserved = int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else None
    saturation_ok = bool(development["saturation"]["cap_gate_pass"])
    record = {
        "status": "ok" if saturation_ok else "INVALID",
        "failure_reasons": [] if saturation_ok else [
            "Development healthy v_cap saturation exceeds the frozen 20% structural gate"
        ],
        "run_id": f"{RUN_ID_PREFIX}-seed{seed}",
        "schema": SEED_SCHEMA,
        "task_id": TASK_ID, "attempt_id": ATTEMPT, "agent_id": AGENT,
        "arm_id": ARM_ID, "baseline_arm_id": BASELINE_ARM_ID,
        "model_seed": seed,
        "normative": not smoke,
        "arm_config_sha256": arm_config_sha256(arm_n4),
        "arm_identity": {
            "component_set": list(arm_n4.component_set),
            "principal_change": arm_n4.principal_change,
            "mechanism_spec_sha256": arm_n4.mechanism_spec_sha256,
            "mechanism_spec_amendment_sha256": capacity.EXPECTED_CONTRACT["mechanism_N4-CRN"],
            "scale_parameterization": N4_SCALE_PARAMETERIZATION,
            "guard_contract": N4_GUARD_CONTRACT,
            "representation_epochs": 0,
            "scale_epochs": N4_SCALE_EPOCHS,
        },
        "representation_baseline": {
            "kind": "exact_final_B0-FULL_epoch100",
            "checkpoint_path": identity["checkpoint_path"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "restored_checkpoint_sha256": restored_b0["checkpoint_sha256"],
            "live_model_state_sha256": state_sha256(model.state_dict()),
            "live_ema_state_sha256": ema_snapshot,
            "cache_path": identity["cache_path"],
            "cache_file_sha256": identity["cache_file_sha256"],
            "cache_inner_sha256": identity["cache_inner_sha256"],
            "cache_records": identity["cache_records"],
            "representation_epochs_completed": 100,
            "representation_optimizer_steps": 0,
            "matches_accepted_Task23_live_digests": True,
        },
        "scale_fit": {
            "scope": "final",
            "training_history_ids": list(sorted(fit_history_ids)),
            "record_counts": {history_id: len(pools[history_id]) for history_id in fit_history_ids},
            "record_total": int(result["record_count"]),
            "bounds": {key: bounds[key] for key in ("q05", "q95", "v_floor", "v_cap")},
            "residual_records_sha256": bounds["residual_records_sha256"],
            "optimizer_steps": int(result["optimizer_steps"]),
            "per_epoch_steps": int(result["per_epoch_steps"]),
            "per_epoch_nll": list(result["per_epoch_nll"]),
            "final_train_nll": float(result["final_train_nll"]),
            "saturation_train": dict(result["saturation_train"]),
            "cap_gate_pass_train": bool(result["cap_gate_pass_train"]),
            "init_anchor": dict(result["init_anchor"]),
            "init_interiority": dict(result["init_interiority"]),
            "first_step_gradients": {
                key: result["first_step_gradients"][key]
                for key in ("total_nonzero", "total_nonfinite", "missing_grad_parameters")
            },
            "epoch_order_seeds": list(result["epoch_order_seeds"]),
            "order_seed_rule": str(result["order_seed_rule"]),
            "scale_parameters": int(result["scale_parameters"]),
        },
        "scale_checkpoint": scale_block,
        "capacity_recertification": folds,
        "coherent_checkpoint": (
            {key: value for key, value in coherent_block.items() if not key.startswith("_")}
            if coherent_block is not None
            else {
                "status": "not_emitted_non_normative_smoke",
                "reason": (
                    "bounded smoke support cannot bind the certified coherent record scope; "
                    "the full run emits the coherent artifact"
                ),
            }
        ),
        "restored_artifact_proof": restored_artifact_proof,
        "cache_binding": cache_binding,
        "bank": bank_block,
        "calibration": calibration,
        "development": development,
        "representation_invariance": {
            "model_state_equal_bitwise": bool(model_unchanged),
            "ema_equal": bool(ema_unchanged),
            "bank_equal": bool(bank_unchanged),
            "post_n4_b0_rederivation_slice_files": {
                branch: len(slice_expected[branch]) for branch in ("S_pred", "S_pop")
            },
            "post_n4_b0_rederivation_bitwise_equal": True,
        },
        "compute": compute,
        "resources": {
            "device": str(device),
            "device_name": torch.cuda.get_device_properties(device).name if device.type == "cuda" else "cpu",
            "peak_cuda_memory_allocated_bytes": peak_allocated,
            "peak_cuda_memory_reserved_bytes": peak_reserved,
            "peak_scope": (
                "torch peak since the start of this seed run: B0 strict restore, three-fold capacity "
                "recertification, final scale fit, scale-envelope and coherent-artifact save/restore, "
                "Fit bank refit, Calibration scoring (live and restored), full Development scoring, "
                "the post-N4 parity slice and the compute profiles; not a training-only peak"
            ),
            "fit_wall_s": fit_wall,
            "calibration_wall_s": calibration_wall,
            "development_wall_s": development_wall,
            "total_wall_s": time.time() - run_start,
            "scale_steps_per_s": result["optimizer_steps"] / fit_wall if fit_wall > 0 else None,
        },
        "limitations": [
            "descriptive three-seed evaluation only; no significance, quality or promotion claim",
            "N4 declares no S_pop mechanism; its S_pop branch is the unchanged B0 public path",
            "capacity is recertified in-run on the changed source; the Task13B A02 PASS is executed-snapshot evidence only",
            "compute figures are one-file/one-batch operational estimates, not a campaign profiler trace",
        ],
    }
    if smoke:
        record["failure_reasons"] = list(record["failure_reasons"]) + [
            "bounded non-normative smoke: never a campaign result"
        ]
    n4_maps = development.pop("n4_maps")
    b0_maps = development.pop("b0_maps")
    development_scores_path = seed_dir / "development_scores.json"
    _write_json(
        development_scores_path,
        {
            "schema": "s18-t30-n4-development-scores-v1",
            "task_id": TASK_ID, "attempt_id": ATTEMPT, "agent_id": AGENT,
            "model_seed": seed,
            "S_pred_normalized": n4_maps,
            "S_pred_b0_raw": b0_maps["S_pred"],
            "S_pop_b0_unchanged": b0_maps["S_pop"],
        },
    )
    coherent_public = record["coherent_checkpoint"]
    record["next_stage"] = {
        "schema": NEXT_STAGE_SCHEMA,
        "consumer_tasks": ["S18-T31", "S18-T33"],
        "status": (
            "ready" if coherent_block is not None and saturation_ok
            else "non_normative_smoke" if smoke else "incomplete"
        ),
        "coherent_checkpoint_path": coherent_public.get("path"),
        "coherent_checkpoint_sha256": coherent_public.get("sha256"),
        "coherent_checkpoint_schema": coherent_public.get("checkpoint_schema"),
        "scale_envelope_path": scale_block["path"],
        "scale_envelope_sha256": scale_block["sha256"],
        "scale_envelope_state_sha256": scale_block["state_dict_sha256"],
        "scale_parameterization": N4_SCALE_PARAMETERIZATION,
        "guard_contract": N4_GUARD_CONTRACT,
        "n4_in_scope_records": FIT_SCALE_RECORDS,
        "n4_final_scale_steps": int(result["optimizer_steps"]),
        "bounds": {key: float(bounds[key]) for key in ("q05", "q95", "v_floor", "v_cap")},
        "residual_records_sha256": bounds["residual_records_sha256"],
        "q95": dict(calibration["n4_thresholds"]),
        "q95_equals_B0_unchanged_S_pop": True,
        "bank_sha256": bank_block["bank_sha256"],
        "model_state_sha256": state_sha256(model.state_dict()),
        "ema_state_sha256": state_sha256(model.target_encoder.state_dict()),
        "arm_config_sha256": arm_config_sha256(arm_n4),
        "fit_role_sha256": _fit_role_sha256(binding),
        "fit_support": {
            "histories": list(fit_history_ids),
            "healthy_rows": len(support["fit_rows"]),
            "manifests": [item["manifest_sha256_bound"] for item in support["fit_records"]],
        },
        "calibration_support": {
            "histories": list(support["calibration_history_ids"]),
            "healthy_rows": len(calibration_rows),
            "support_sha256": calibration["support"]["support_sha256"],
            "manifests": [item["manifest_sha256_bound"] for item in support["calibration_records"]],
        },
        "development_support": {
            "histories": list(support["development_history_ids"]),
            "manifests": [item["manifest_sha256_bound"] for item in support["development_records"]],
        },
        "development_scores_path": str(development_scores_path),
        "development_scores_sha256": _sha256_file(development_scores_path),
        "metric_code": "representation.crn.n4_score_files + synth.probe15.evaluate_histories",
    }
    _write_json(seed_dir / "seed_record.json", record)
    _write_json(seed_dir / "next_stage.json", record["next_stage"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def run_full(args, bdoc: dict) -> dict:
    worktree = Path(args.worktree)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    if progress_path.exists():
        raise _fail("out/progress.jsonl already exists; refusing to overwrite a frozen run")
    preflight = run_preflight(args, bdoc)
    _write_json(out_dir / "preflight.json", preflight)
    binding = load_task6_binding(args.handoff)
    lock = capacity.load_task14_lock(worktree)
    legacy_b0 = capacity.load_legacy_b0_lock(worktree)
    b0_records = _load_b0_records(Path(bdoc.get("b0_seed_records", B0_RECORDS_PATH)))
    support = _load_supports(binding, worktree, development=True)
    environment = _environment_block(args)
    _write_json(out_dir / "environment.json", environment)
    device = torch.device(args.device)
    records: dict[str, dict] = {}
    with progress_path.open("a", encoding="utf-8") as progress:
        for seed in MODEL_SEEDS:
            seed_dir = out_dir / f"seed-{seed}"
            record = _run_seed(
                seed=seed, fixture=bdoc["fixture"], b0_record=b0_records[str(seed)],
                binding=binding, legacy_b0=legacy_b0, lock=lock, support=support,
                capacity_certification=preflight["task14_lock_binding"]["capacity_certification"],
                worktree=worktree, out_dir=out_dir, device=device, progress=progress,
                seed_dir=seed_dir, smoke=False, development_limit=None,
            )
            records[str(seed)] = record
            _log_progress(
                progress,
                {"run": f"{RUN_ID_PREFIX}-seed{seed}", "event": "seed_complete", "model_seed": seed,
                 "status": record["status"], "cap_gate_pass": record["development"]["saturation"]["cap_gate_pass"]},
            )
    _write_json(out_dir / "seed_records.json", records)
    paired = _paired_development(
        arm_records=records, b0_records=b0_records, b0_records_sha256=B0_RECORDS_SHA256,
    )
    _write_json(out_dir / "paired_development.json", paired)
    return {"records": records, "paired": paired, "preflight": preflight, "RUNNER_EXIT": "ok"}


def run_smoke(args, bdoc: dict) -> dict:
    worktree = Path(args.worktree)
    out_dir = Path(args.out_dir) / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight = run_preflight(args, bdoc)
    _write_json(out_dir / "preflight.json", preflight)
    binding = load_task6_binding(args.handoff)
    lock = capacity.load_task14_lock(worktree)
    legacy_b0 = capacity.load_legacy_b0_lock(worktree)
    b0_records = _load_b0_records(Path(bdoc.get("b0_seed_records", B0_RECORDS_PATH)))
    support = _load_supports(binding, worktree, development=True)
    seed = MODEL_SEEDS[0]
    device = torch.device(args.device)
    seed_dir = out_dir / f"seed-{seed}"
    progress_path = out_dir / "progress.jsonl"
    if progress_path.exists():
        raise _fail("smoke out/progress.jsonl already exists; refusing to overwrite")
    with progress_path.open("a", encoding="utf-8") as progress:
        record = _run_seed(
            seed=seed, fixture=bdoc["fixture"], b0_record=b0_records[str(seed)],
            binding=binding, legacy_b0=legacy_b0, lock=lock, support=support,
            capacity_certification=preflight["task14_lock_binding"]["capacity_certification"],
            worktree=worktree, out_dir=out_dir, device=device, progress=progress,
            seed_dir=seed_dir, smoke=True, development_limit=SMOKE_DEVELOPMENT_FILES_PER_HISTORY,
        )
    _write_json(out_dir / "smoke_record.json", record)
    _write_json(out_dir / "environment.json", {**_environment_block(args), "mode": "smoke"})
    return {
        "SMOKE_OK": True,
        "schema": SMOKE_SCHEMA,
        "model_seed": seed,
        "normative": False,
        "promotion": "none; bounded diagnostic smoke, never a campaign result",
        "scale_fit": record["scale_fit"],
        "scale_checkpoint": record["scale_checkpoint"],
        "cache_binding": record["cache_binding"],
        "calibration": record["calibration"]["n4_calibration"],
        "s_pop_parity": record["development"]["s_pop_parity"],
        "saturation": record["development"]["saturation"],
        "compute": record["compute"],
        "resources": record["resources"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    bdoc = capacity.load_binding_doc(Path(args.binding))
    _execution_binding(bdoc, args)
    if args.mode == "preflight":
        preflight = run_preflight(args, bdoc)
        _write_json(Path(args.out_dir) / "preflight.json", preflight)
        print(json.dumps(
            {
                "PREFLIGHT_OK": True,
                "source_commit": preflight["source_commit"],
                "runner_sha256": preflight["runner_sha256"],
                "fit_role_sha256": preflight["fit_role_sha256"],
                "scale_head_parameters": preflight["scale_head_parameters"],
                "b0_seed_records_sha256": preflight["b0_seed_records_sha256"],
                "record_totals": {key: block["record_total"] for key, block in preflight["seeds"].items()},
                "final_scale_steps": {
                    key: block["final_scale_steps"] for key, block in preflight["seeds"].items()
                },
            },
            sort_keys=True,
        ))
        print(json.dumps({"RUNNER_EXIT": "preflight_ok"}))
        return 0
    if args.mode == "smoke":
        outcome = run_smoke(args, bdoc)
        print(json.dumps(outcome, sort_keys=True))
        print(json.dumps({"RUNNER_EXIT": "smoke_ok"}))
        return 0
    outcome = run_full(args, bdoc)
    print(json.dumps(
        {
            "RUNNER_EXIT": outcome["RUNNER_EXIT"],
            "seeds": list(outcome["records"]),
            "statuses": {key: record["status"] for key, record in outcome["records"].items()},
            "cap_gate": {
                key: record["development"]["saturation"]["cap_gate_pass"]
                for key, record in outcome["records"].items()
            },
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
