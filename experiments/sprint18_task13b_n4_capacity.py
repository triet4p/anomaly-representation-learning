"""S18-T13B normative N4-CRN Fit-crossfold capacity runner (permanent entrypoint).

Binds the exact final epoch-100 B0-FULL online/EMA checkpoints and prediction
caches for seeds 181801/181802/181803 (Task23 A04), strict-restores each
checkpoint (experiment scope), rebuilds N4 scale records from the bound cache
latents (context-only inputs, detached, no re-encoding, no re-masking), and
runs the prospective v3 N4 recipe: 3 Fit-history cross-fold scale fits per seed
(50 epochs, batch<=512, frozen sprint18-n4-scale-order-v2 CPU permutations,
AdamW 1e-3/1e-4, clip 1.0, Fit-only linear q05/q95 bounds, cap-only 20% gate,
zero representation updates), with the accepted Fit-mean-anchored output-bias
initialization (``fit-mean-anchored-bias-v1``) and fail-closed startup guards
(``n4-v3-guards-v1``). Proves v3 scale-checkpoint envelope save/strict restore
plus the public forward/output parity through the committed ``n4_score_files``
consumer triple per fold.

The operative N4 mechanism specification is the prospective v3 amendment
(``experiments/sprint18-context-residual-normalization-spec-v3.md``); the
v1/v2 files remain immutable historical records. A normative v3 run requires
the versioned prospective Task14 lock
(``experiments/sprint18-task14-config-lock-v2.json``), which binds the v3
amendment digest, the reviewed model-implementation commit, the required
runner digest, the final-B0/cache identities and the three N4-CRN seed
configs. The historical v1 lock is preserved byte-identically and continues
to bind every other arm; this runner never edits or reinterprets it. In
particular the three B0-FULL strict restores bind the v1 coefficient-lock
digest (their checkpoints' producing lock) and are verified against the v1
locked arm configurations; only the N4-CRN arms bind the prospective v3
digest. The nine
A03 scale checkpoints are reclassified ``rejected_failed_dead_zone_v1`` and
are never loaded by this path.

Product-device contract: ``n4_score_files`` co-locates the collated batch onto
the deployed model device (CPU callers observe a no-op). This runner
co-deploys the restored B0 model and the fitted scale head on the same device
and scores the public consumer there directly -- no CPU-clone workaround.
Consumer bound-containment check: per-patch ``v`` is clamped into
``[v_floor, v_cap]`` by the frozen forward, so the file-mean scale inherits
that containment up to two float32 rounding steps: clamping through
``log``/``exp`` in float32, then the file-mean accumulation in float32.
Retained local CPU characterization (472 trials:
``artifacts/sprint-18/s18-t13b-c04/characterize_float32.py`` driving the
committed ``ContextResidualScaleModel.forward`` clamp/exp plus the exact
row-``mean()`` reduction in ``n4_score_files``; bound magnitudes
``1e-3..1e2``, cap/floor ratios ``1..10`` incl. the frozen 10x edge, file
lengths ``1..500``, uniform/mixed/random/adversarial patterns) measures a
worst-case end-to-end bound excursion of ``~1.1e-7`` relative and a worst
single-step error of ``~4.1e-7`` relative. The runner therefore allows a
per-side tolerance of ``8 * eps32`` relative to each bound (``~9.5e-7``
relative, about 8.6x the worst observed bound excursion): the below-floor
side scales with ``|v_floor|``, the above-cap side with ``|v_cap|``, so the
frozen ``v_cap/v_floor <= 10`` edge rule cannot widen one side's margin
through the other bound's magnitude. Evidence is CPU-only (local host has no
CUDA device; see the retained script header for the backend limit); any
uncharacterized backend can only abort loudly here, never pass silently.
The bounds themselves are never loosened, clamped, or re-tuned; the
tolerance covers only these two representational steps in the file-mean
derivation. An absolute tolerance (e.g. 1e-9) is arithmetically unsound:
real float32 mean rounding near unit-scale values is ~1e-8, which strictly
exceeds it, so such a check aborts on sound scales. Any excursion beyond
the allowance raises ``ValueError`` and aborts the run -- genuine
out-of-bounds scales cannot pass silently.

Modes:
  preflight - fixture/support/binding checks only (fast, CPU-light).
  pilot     - full 9-fold normative runtime on the requested device.
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
from hashlib import sha256
from pathlib import Path


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[1], Path.cwd()):
        if (candidate / "src" / "representation").is_dir() and (candidate / "experiments").is_dir():
            return candidate
    raise SystemExit("DRIVER_FATAL cannot locate worktree root (src/representation missing)")


REPO_ROOT = _resolve_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

ATTEMPT = "S18-T13B-V3"
ARM_ID = "N4-CRN"
MODEL_SEEDS = (181801, 181802, 181803)
#: Per-side file-mean containment allowance for the public-consumer bound
#: check: 8 * eps32 relative to EACH bound (below-floor side scales with
#: |v_floor|, above-cap side with |v_cap|). Covers the two float32 rounding
#: steps (clamp log/exp, then mean accumulation; retained C04 CPU
#: characterization: worst bound excursion ~1.1e-7, ~8.6x margin on each
#: side). Bounds, clamps, and recipe are unchanged.
CONSUMER_SCALE_REL_TOL = 8 * 2.0**-23
#: Bounded public-consumer proof: the first N qualified heldout files per
#: fold are scored through ``n4_score_files`` (proof, not resampling).
CONSUMER_FILES_PER_FOLD = 2

EXPECTED_CONTRACT = {
    "protocol_sha256": "e1dcd17914e920520ad0d8f50e185874b95414e6a4fe907903619daf145622d2",
    #: Operative prospective v3 N4 mechanism amendment.
    "mechanism_N4-CRN": "e7fe42e41832ae283614ee6a76c0623ee567068dcc540a222f5b66a9d4262da2",
    #: Immutable v2 base amendment retained for historical integrity + lock identity.
    "mechanism_N4-CRN-v2-base": "4d1c8097c9cc27d3fbeb2e2a6e70a29a7bc63d7899646a9c712a315b39d9ffe3",
    "executable_binding_sha256": "383c62114ee6120d5fd19f866dfd585083d1d5899c9ffe91804761bd8a10a2d5",
    "fit_rows": {"S18I-ITER-0005-FIT-01": 1809, "S18I-ITER-0005-FIT-02": 1882, "S18I-ITER-0005-FIT-03": 1755},
    "checkpoints": {
        "181801": "9292d7c59037cba7b7fd337dbb896ed492777abff4713307c98f86a3b5226a8f",
        "181802": "8ae58da90818883ffa8b7dd7b0c73fadfcd5a24a0d3f2fc713c9a1d055b6621f",
        "181803": "79edbdf37e4180368814e02b7201c7dd3f8a7b4c7bc47408bd9bbde5d059d408",
    },
    "cache_files": {
        "181801": "18345f863d04f71d16d9be7b6f58982568a4ee4317eba526922369471355bece",
        "181802": "e035b2b7cdac53b6484244239f3db1d851cb0395e2fdd788133de5278f9333b2",
        "181803": "1d4a2a74de36d8bf45a67e7ff795ce279e6aec08494d8ca27a1bca817db332a7",
    },
}
#: Frozen disposition of the failed A03 v1 (dead-zone) scale evidence. It is
#: provenance only: those checkpoints/results are never loaded or consumed by
#: this v3 path, and their bytes are never edited.
LEGACY_A03_DISPOSITION = {
    "label": "rejected_failed_dead_zone_v1",
    "runs_root": "/home/trietlm/s18-t13b-a03-runs/",
    "consumed_by_v3": False,
    "reason": (
        "initialized pre-clamp log-variance above the hard cap produced exactly zero loss "
        "gradients; AdamW decoupled decay was the only parameter motion on all nine folds"
    ),
}
#: The v3 amendment's operative mechanism spec path (verified by digest).
MECHANISM_SPEC_PATH = "experiments/sprint18-context-residual-normalization-spec-v3.md"
#: Authorized-execution binding supplied per run (never frozen in source):
#: the external binding document (``--binding``) plus the trusted CLI pins
#: (``--source-commit`` + ``--driver-sha256``) identify the reviewed commit
#: the runtime is allowed to execute. Freezing a commit SHA inside this file
#: would re-freeze every checkpoint that adds this file — the A02 defect —
#: so the authorized commit is read from the trusted external binding/args
#: instead of from this source. The scientific fixture identity above stays
#: immutable in source; the execution identity below stays external.
EXECUTION_BINDING_FIELDS = ("source_commit", "runner_sha256")

#: Versioned prospective Task14 lock activating N4-CRN v3. The historical v1
#: lock stays byte-identical and is referenced (never edited) by this version;
#: every arm other than N4-CRN remains bound by that v1 lock.
TASK14_LOCK_PATH = "experiments/sprint18-task14-config-lock-v2.json"
TASK14_LEGACY_LOCK_PATH = "experiments/sprint18-task14-config-lock.json"
TASK14_LOCK_SCHEMA = "s18-task14-config-lock-v2"
TASK14_LOCK_VERSION = 2
#: Frozen lock-digest rule: SHA-256 over canonical JSON of the core (every
#: field except these derived containers).
TASK14_LOCK_CORE_EXCLUDE = frozenset(
    {"coefficient_lock_sha256", "locked_configs", "config_sha256", "file_sha256"}
)
#: Historical v1 lock identities. The v3 lock must supersede these exact bytes
#: and never rewrite them.
LEGACY_TASK14_LOCK_SHA256 = "c4c1a8bde14310cf4436c10762e3e98475f7c84eb74fcb8a74b3e267885edec3"
LEGACY_TASK14_COEFFICIENT_LOCK_SHA256 = (
    "6eb108b2073181d427f1aedfc81fb3bae4c29595108d15de486082800bdd9d6e"
)
#: N4-CRN declares no representation-loss coefficients.
TASK14_LOCK_COEFFICIENTS = {
    "lambda_info_max": None,
    "lambda_hcc_max": None,
    "alpha": None,
    "beta": None,
}
#: Historical B0-FULL coefficient selection. The final epoch-100 B0
#: checkpoints were produced and locked under the v1 lock; the v3 lock
#: supersedes that lock for N4-CRN only, so the B0 restore arm keeps the
#: historical selection and must bind the v1 coefficient-lock digest.
B0_FULL_COEFFICIENTS = {
    "lambda_info_max": 0.05,
    "lambda_hcc_max": None,
    "alpha": None,
    "beta": None,
}
TASK14_FIT_HISTORIES = (
    "S18I-ITER-0005-FIT-01",
    "S18I-ITER-0005-FIT-02",
    "S18I-ITER-0005-FIT-03",
)


def _fail(message: str) -> ValueError:
    return ValueError(f"S18-T13B-C06 frozen-contract violation: {message}")


_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _execution_binding(bdoc: dict, args) -> dict:
    """Return the externally authorized execution identity, fail-closed.

    Scientific fixture identity (protocol/spec/binding/fit counts/fixture
    digests) stays frozen in ``EXPECTED_CONTRACT``. The authorized source
    commit and runner digest are execution authorization, not science: they
    arrive via the Main-authorized external binding document
    (``bdoc["source_commit"]``/``bdoc["runner_sha256"]``) and the trusted
    CLI pins (``args.source_commit``/``args.driver_sha256``), and all four
    must agree exactly. No untrusted self-report is accepted: live ``HEAD``
    must equal the authorized commit, and the observed runner bytes must
    equal the authorized runner digest. Never defaults, never empty-string
    coercion, never silent acceptance.
    """
    if not isinstance(bdoc, dict):
        raise _fail("execution binding is not a mapping")
    bdoc_commit = bdoc.get("source_commit")
    bdoc_runner = bdoc.get("runner_sha256")
    cli_commit = getattr(args, "source_commit", None)
    cli_runner = getattr(args, "driver_sha256", None)
    for value in (bdoc_commit, bdoc_runner, cli_commit, cli_runner):
        if not isinstance(value, str) or not value:
            raise _fail("execution binding is missing an authorized identity field")
    if not _HEX40.fullmatch(bdoc_commit) or not _HEX40.fullmatch(cli_commit):
        raise _fail("authorized source commit is not a full Git SHA")
    if not _HEX64.fullmatch(bdoc_runner) or not _HEX64.fullmatch(cli_runner):
        raise _fail("authorized runner digest is not a SHA-256 hex digest")
    if bdoc_commit != cli_commit:
        raise _fail("binding source commit differs from the trusted CLI pin")
    if bdoc_runner != cli_runner:
        raise _fail("binding runner digest differs from the trusted CLI pin")
    return {"source_commit": bdoc_commit, "runner_sha256": bdoc_runner}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_binding_doc(path: Path) -> dict:
    return json.loads(path.read_bytes())


def self_check(args, bdoc: dict) -> dict:
    worktree = Path(args.worktree)
    authorized = _execution_binding(bdoc, args)
    head = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != authorized["source_commit"]:
        raise _fail(f"source commit differs: observed {head}")
    observed_driver_sha = _sha_file(Path(__file__).resolve())
    if observed_driver_sha != authorized["runner_sha256"]:
        raise _fail("driver digest differs from the authorized execution binding")
    bdoc_sha = _sha_file(Path(args.binding))
    return {"head": head, "driver_sha256": observed_driver_sha, "binding_sha256": bdoc_sha}


def _check_contract_files(worktree: Path) -> None:
    from representation.sprint18_harness import verify_contract_files

    observed = verify_contract_files(worktree)
    if observed.get("experiments/sprint18-full-training-protocol-v2.md") != EXPECTED_CONTRACT["protocol_sha256"]:
        raise _fail("protocol v2 digest changed")
    if observed.get("experiments/sprint18-context-residual-normalization-spec-v2.md") != EXPECTED_CONTRACT["mechanism_N4-CRN-v2-base"]:
        raise _fail("historical N4 v2 base digest changed")
    v3_path = worktree / MECHANISM_SPEC_PATH
    if not v3_path.is_file():
        raise _fail("prospective N4 v3 mechanism amendment is missing")
    if _sha_file(v3_path) != EXPECTED_CONTRACT["mechanism_N4-CRN"]:
        raise _fail("prospective N4 v3 mechanism amendment digest changed")


def task14_lock_core_digest(lock: dict) -> str:
    """Recompute the Task14 lock core digest by the frozen canonical rule."""
    core = {key: value for key, value in lock.items() if key not in TASK14_LOCK_CORE_EXCLUDE}
    canonical = json.dumps(
        core, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_task14_v3_lock(worktree: Path) -> dict:
    """Load the prospective versioned Task14 lock; never fall back to v1."""
    path = Path(worktree) / TASK14_LOCK_PATH
    if not path.is_file():
        raise _fail(f"prospective Task14 v3 lock is missing: {TASK14_LOCK_PATH}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise _fail("prospective Task14 v3 lock is not a mapping")
    return payload


def validate_task14_v3_lock(
    lock: dict, *, worktree: Path, observed_runner_sha256: str
) -> dict:
    """Accept only the reviewed prospective v3 Task14 lock, fail-closed.

    The lock is not self-authenticating. The reviewed model-implementation
    commit must be an ancestor of the executed ``HEAD``, every reviewed
    implementation byte is compared against the live worktree, and the
    declared runner digest must equal the observed runner bytes (already tied
    to the external binding/CLI pin by ``self_check``). The historical v1 lock
    is never edited and all other arms remain bound by it. No future
    activation commit is frozen in source or in the lock.

    Recipe metadata is never trusted on its own: every execution-bearing
    recipe value is compared with the live frozen constants the runner and
    harness actually execute, and the carried-forward recipe must equal the
    recipe in the SHA-pinned historical lock. A coordinated edit that
    re-derives the core digest and rebuilds the config digests still fails
    when it contradicts the executed recipe.
    """
    from representation.crn import (
        N4_GUARD_CONTRACT,
        N4_SATURATION_CAP_GATE,
        N4_SCALE_BATCH_SIZE,
        N4_SCALE_CHECKPOINT_SCHEMA,
        N4_SCALE_CLIP_NORM,
        N4_SCALE_EPOCHS,
        N4_SCALE_HIDDEN_WIDTH,
        N4_SCALE_LR,
        N4_SCALE_PARAM_COUNT,
        N4_SCALE_PARAM_LIMIT,
        N4_SCALE_PARAMETERIZATION,
        N4_SCALE_WEIGHT_DECAY,
    )
    from representation.sprint18_harness import arm_config_sha256, make_sprint18_arm_config

    if not isinstance(lock, dict):
        raise _fail("Task14 v3 lock is not a mapping")
    if lock.get("schema") != TASK14_LOCK_SCHEMA:
        raise _fail(f"Task14 lock schema is not {TASK14_LOCK_SCHEMA}")
    if lock.get("lock_version") != TASK14_LOCK_VERSION:
        raise _fail("Task14 v3 lock version differs")
    if lock.get("task_id") != "S18-T14":
        raise _fail("Task14 v3 lock task identity differs")
    if lock.get("target_ref") != "refs/heads/master":
        raise _fail("Task14 v3 lock target ref differs")
    digest = lock.get("coefficient_lock_sha256")
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise _fail("Task14 v3 lock core digest is missing or malformed")
    if digest != task14_lock_core_digest(lock):
        raise _fail("Task14 v3 lock core digest is inconsistent")

    supersedes = lock.get("supersedes")
    if not isinstance(supersedes, dict) or supersedes.get("path") != TASK14_LEGACY_LOCK_PATH:
        raise _fail("Task14 v3 lock does not supersede the historical lock path")
    if supersedes.get("coefficient_lock_sha256") != LEGACY_TASK14_COEFFICIENT_LOCK_SHA256:
        raise _fail("Task14 v3 lock supersedes the wrong historical coefficient digest")
    if supersedes.get("file_sha256") != LEGACY_TASK14_LOCK_SHA256:
        raise _fail("Task14 v3 lock supersedes the wrong historical lock bytes")
    legacy_path = Path(worktree) / TASK14_LEGACY_LOCK_PATH
    if not legacy_path.is_file() or _sha_file(legacy_path) != LEGACY_TASK14_LOCK_SHA256:
        raise _fail("historical Task14 lock bytes changed")
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))

    v3 = lock.get("n4_v3_activation")
    if not isinstance(v3, dict):
        raise _fail("Task14 v3 activation block is missing")
    impl = v3.get("reviewed_implementation")
    if not isinstance(impl, dict) or impl.get("source_commit") != lock.get("source_commit"):
        raise _fail("Task14 v3 lock reviewed implementation commit is inconsistent")
    reviewed_commit = impl.get("source_commit")
    if not isinstance(reviewed_commit, str) or not _HEX40.fullmatch(reviewed_commit):
        raise _fail("Task14 v3 lock reviewed implementation commit is not a full Git SHA")
    ancestor = subprocess.run(
        ["git", "-C", str(worktree), "merge-base", "--is-ancestor", reviewed_commit, "HEAD"],
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise _fail("reviewed implementation commit is not an ancestor of the executed HEAD")
    paths = impl.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise _fail("Task14 v3 lock reviewed implementation paths are missing")
    for relative, expected in paths.items():
        path = Path(worktree) / relative
        if not path.is_file():
            raise _fail(f"reviewed implementation path is missing: {relative}")
        if _sha_file(path) != expected:
            raise _fail(f"reviewed implementation bytes changed: {relative}")

    runner = lock.get("runner_identity")
    if not isinstance(runner, dict) or runner.get("path") != "experiments/sprint18_task13b_n4_capacity.py":
        raise _fail("Task14 v3 lock runner identity differs")
    if runner.get("sha256") != observed_runner_sha256:
        raise _fail("Task14 v3 lock runner digest differs from the observed runner bytes")

    n4lock = (lock.get("arms") or {}).get("N4-CRN")
    if not isinstance(n4lock, dict) or n4lock.get("status") != "locked_executable":
        raise _fail("Task14 v3 N4-CRN lock status changed")
    if n4lock.get("mechanism_spec_sha256") != EXPECTED_CONTRACT["mechanism_N4-CRN-v2-base"]:
        raise _fail("Task14 N4-CRN historical v2 mechanism binding changed")
    for holder, label in ((n4lock, "N4-CRN arm"), (v3, "activation block")):
        if holder.get("mechanism_spec_amendment_sha256") != EXPECTED_CONTRACT["mechanism_N4-CRN"]:
            raise _fail(f"Task14 v3 {label} does not bind the prospective v3 mechanism digest")
        if holder.get("scale_parameterization") != N4_SCALE_PARAMETERIZATION:
            raise _fail(f"Task14 v3 {label} parameterization differs")
        if holder.get("guard_contract") != N4_GUARD_CONTRACT:
            raise _fail(f"Task14 v3 {label} guard contract differs")
        if holder.get("scale_checkpoint_schema") != N4_SCALE_CHECKPOINT_SCHEMA:
            raise _fail(f"Task14 v3 {label} scale-checkpoint schema differs")

    recipe = lock.get("n4_recipe")
    if not isinstance(recipe, dict):
        raise _fail("Task14 v3 lock N4 recipe is missing")
    amendment = recipe.get("v3_amendment")
    fit_recipe = recipe.get("fit_recipe")
    architecture = recipe.get("architecture")
    capacity = recipe.get("capacity")
    if (
        not isinstance(amendment, dict)
        or not isinstance(fit_recipe, dict)
        or not isinstance(architecture, dict)
        or not isinstance(capacity, dict)
    ):
        raise _fail("Task14 v3 lock N4 recipe structure is incomplete")
    # The embedded core digest is derived from these same bytes, so it cannot
    # authenticate them. Bind every execution-bearing recipe field to the live
    # frozen constants the runner and harness execute, then require the whole
    # carried-forward recipe to equal the SHA-pinned historical lock recipe.
    for label, observed, expected in (
        ("fit_recipe.epochs", fit_recipe.get("epochs"), N4_SCALE_EPOCHS),
        ("fit_recipe.batch_size", fit_recipe.get("batch_size"), N4_SCALE_BATCH_SIZE),
        ("fit_recipe.batch_size_unit", fit_recipe.get("batch_size_unit"), "patch_records"),
        ("fit_recipe.learning_rate", fit_recipe.get("learning_rate"), N4_SCALE_LR),
        ("fit_recipe.weight_decay", fit_recipe.get("weight_decay"), N4_SCALE_WEIGHT_DECAY),
        ("fit_recipe.gradient_clip_norm", fit_recipe.get("gradient_clip_norm"), N4_SCALE_CLIP_NORM),
        ("fit_recipe.optimizer", fit_recipe.get("optimizer"), "AdamW"),
        ("fit_recipe.representation_optimizer_steps", fit_recipe.get("representation_optimizer_steps"), 0),
        (
            "fit_recipe.no_early_stopping_or_development_selection",
            fit_recipe.get("no_early_stopping_or_development_selection"),
            True,
        ),
        ("architecture.hidden_width", architecture.get("hidden_width"), N4_SCALE_HIDDEN_WIDTH),
        (
            "capacity.observed_reference_params_d_model_128",
            capacity.get("observed_reference_params_d_model_128"),
            N4_SCALE_PARAM_COUNT,
        ),
        ("capacity.scale_parameter_limit", capacity.get("scale_parameter_limit"), N4_SCALE_PARAM_LIMIT),
        ("v3_amendment.parameter_count", amendment.get("parameter_count"), N4_SCALE_PARAM_COUNT),
        ("v3_amendment.parameter_limit", amendment.get("parameter_limit"), N4_SCALE_PARAM_LIMIT),
        ("v3_amendment.saturation_cap_gate", amendment.get("saturation_cap_gate"), N4_SATURATION_CAP_GATE),
        (
            "v3_amendment.scale_parameterization",
            amendment.get("scale_parameterization"),
            N4_SCALE_PARAMETERIZATION,
        ),
        ("v3_amendment.guard_contract", amendment.get("guard_contract"), N4_GUARD_CONTRACT),
        (
            "v3_amendment.scale_checkpoint_schema",
            amendment.get("scale_checkpoint_schema"),
            N4_SCALE_CHECKPOINT_SCHEMA,
        ),
        ("v3_amendment.spec_path", amendment.get("spec_path"), MECHANISM_SPEC_PATH),
        (
            "v3_amendment.spec_sha256",
            amendment.get("spec_sha256"),
            EXPECTED_CONTRACT["mechanism_N4-CRN"],
        ),
        ("arms.N4-CRN.scale_epochs", n4lock.get("scale_epochs"), N4_SCALE_EPOCHS),
        ("arms.N4-CRN.representation_epochs", n4lock.get("representation_epochs"), 0),
        ("arms.N4-CRN.initialization", n4lock.get("initialization"), N4_SCALE_PARAMETERIZATION),
    ):
        if observed != expected:
            raise _fail(f"Task14 v3 lock {label} contradicts the frozen execution recipe")
    carried_recipe = {key: value for key, value in recipe.items() if key != "v3_amendment"}
    if carried_recipe != legacy.get("n4_recipe"):
        raise _fail("Task14 v3 lock N4 recipe differs from the pinned historical recipe")

    spec = v3.get("spec")
    if not isinstance(spec, dict) or spec.get("path") != MECHANISM_SPEC_PATH:
        raise _fail("Task14 v3 activation does not bind the prospective spec path")
    if spec.get("sha256") != EXPECTED_CONTRACT["mechanism_N4-CRN"]:
        raise _fail("Task14 v3 activation does not bind the prospective spec digest")
    if _sha_file(Path(worktree) / MECHANISM_SPEC_PATH) != EXPECTED_CONTRACT["mechanism_N4-CRN"]:
        raise _fail("prospective N4 v3 mechanism amendment digest changed")

    data_binding = v3.get("final_b0_binding")
    if not isinstance(data_binding, dict):
        raise _fail("Task14 v3 final-B0/data binding block is missing")
    if list(data_binding.get("model_seeds") or []) != list(MODEL_SEEDS):
        raise _fail("Task14 v3 lock model seeds differ")
    if data_binding.get("fit_histories") != list(TASK14_FIT_HISTORIES):
        raise _fail("Task14 v3 lock Fit history roster differs")
    if data_binding.get("fit_eligible_rows") != EXPECTED_CONTRACT["fit_rows"]:
        raise _fail("Task14 v3 lock Fit eligible-row counts differ")
    if data_binding.get("checkpoints_sha256") != EXPECTED_CONTRACT["checkpoints"]:
        raise _fail("Task14 v3 lock final-B0 checkpoint identities differ")
    if data_binding.get("prediction_caches_sha256") != EXPECTED_CONTRACT["cache_files"]:
        raise _fail("Task14 v3 lock prediction-cache identities differ")
    expected_crossfold = [
        {"heldout": heldout, "training": sorted(h for h in TASK14_FIT_HISTORIES if h != heldout)}
        for heldout in TASK14_FIT_HISTORIES
    ] + [{"heldout": "final", "training": list(TASK14_FIT_HISTORIES)}]
    if data_binding.get("crossfold_roles") != expected_crossfold:
        raise _fail("Task14 v3 lock cross-fold roles differ")

    locked = lock.get("locked_configs")
    expected_keys = {f"N4-CRN/seed-{seed}" for seed in MODEL_SEEDS}
    if not isinstance(locked, dict) or set(locked) != expected_keys:
        raise _fail("Task14 v3 lock must contain exactly the three N4-CRN seed configs")
    config_sha = lock.get("config_sha256")
    if not isinstance(config_sha, dict):
        raise _fail("Task14 v3 lock config digests are missing")
    for seed in MODEL_SEEDS:
        key = f"N4-CRN/seed-{seed}"
        config = make_sprint18_arm_config(
            "N4-CRN",
            seed,
            TASK14_LOCK_COEFFICIENTS,
            selection_state="locked",
            coefficient_lock_sha256=digest,
        )
        if config.model_dump(mode="json") != locked[key]:
            raise _fail(f"Task14 v3 N4-CRN locked config differs for seed {seed}")
        if arm_config_sha256(config) != config_sha.get(key):
            raise _fail(f"Task14 v3 N4-CRN config digest differs for seed {seed}")

    return {
        "schema": lock["schema"],
        "lock_version": lock["lock_version"],
        "path": TASK14_LOCK_PATH,
        "file_sha256": _sha_file(Path(worktree) / TASK14_LOCK_PATH),
        "coefficient_lock_sha256": digest,
        "source_commit": lock["source_commit"],
        "reviewed_implementation_commit": reviewed_commit,
        "reviewed_runner_sha256_at_source_commit": runner.get("reviewed_source_commit_sha256"),
        "runner_sha256": runner["sha256"],
        "mechanism_spec_path": MECHANISM_SPEC_PATH,
        "mechanism_spec_sha256": EXPECTED_CONTRACT["mechanism_N4-CRN"],
        "mechanism_spec_base_sha256": EXPECTED_CONTRACT["mechanism_N4-CRN-v2-base"],
        "scale_parameterization": N4_SCALE_PARAMETERIZATION,
        "guard_contract": N4_GUARD_CONTRACT,
        "scale_checkpoint_schema": N4_SCALE_CHECKPOINT_SCHEMA,
        "locked_config_keys": sorted(locked),
        "supersedes": {
            "path": supersedes["path"],
            "file_sha256": supersedes["file_sha256"],
            "coefficient_lock_sha256": supersedes["coefficient_lock_sha256"],
            "scope": supersedes.get("scope"),
        },
        "activation_status": v3.get("status"),
        "legacy_a03_disposition": dict(LEGACY_A03_DISPOSITION),
    }


def load_legacy_b0_lock(worktree: Path) -> dict:
    """Bind the B0-FULL restore arms to the preserved historical v1 lock.

    All three final epoch-100 B0-FULL checkpoints were produced under the
    historical v1 coefficient lock. The prospective v3 lock supersedes that
    lock for N4-CRN only: reinstating the v3 core digest on the B0 restore
    arm is a wire/binding error, not a weight change (the strict checkpoint
    validator compares the whole closed arm configuration). The v1 bytes are
    SHA-pinned and never edited; every B0 arm rebuilt from the v1 digest must
    equal the v1 locked configuration and config digest exactly.
    """
    from representation.sprint18_harness import arm_config_sha256, make_sprint18_arm_config

    path = Path(worktree) / TASK14_LEGACY_LOCK_PATH
    if not path.is_file():
        raise _fail(f"historical Task14 v1 lock is missing: {TASK14_LEGACY_LOCK_PATH}")
    file_sha = _sha_file(path)
    if file_sha != LEGACY_TASK14_LOCK_SHA256:
        raise _fail("historical Task14 lock bytes changed")
    legacy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(legacy, dict) or legacy.get("schema") != "s18-task14-config-lock-v1":
        raise _fail("historical Task14 lock schema changed")
    digest = legacy.get("coefficient_lock_sha256")
    if digest != LEGACY_TASK14_COEFFICIENT_LOCK_SHA256:
        raise _fail("historical Task14 coefficient lock digest changed")
    locked = legacy.get("locked_configs")
    config_sha = legacy.get("config_sha256")
    if not isinstance(locked, dict) or not isinstance(config_sha, dict):
        raise _fail("historical Task14 locked configs are missing")
    config_shas: dict[str, str] = {}
    for seed in MODEL_SEEDS:
        key = f"B0-FULL/seed-{seed}"
        if key not in locked or key not in config_sha:
            raise _fail(f"historical Task14 lock lacks the B0-FULL config for seed {seed}")
        config = make_sprint18_arm_config(
            "B0-FULL",
            seed,
            B0_FULL_COEFFICIENTS,
            selection_state="locked",
            coefficient_lock_sha256=digest,
        )
        if config.model_dump(mode="json") != locked[key]:
            raise _fail(f"historical Task14 B0-FULL locked config differs for seed {seed}")
        observed = arm_config_sha256(config)
        if observed != config_sha[key]:
            raise _fail(f"historical Task14 B0-FULL config digest differs for seed {seed}")
        config_shas[str(seed)] = observed
    return {
        "path": TASK14_LEGACY_LOCK_PATH,
        "schema": legacy["schema"],
        "file_sha256": file_sha,
        "coefficient_lock_sha256": digest,
        "locked_config_keys": sorted(f"B0-FULL/seed-{seed}" for seed in MODEL_SEEDS),
        "config_sha256": config_shas,
    }


def consumer_scale_within_bounds(scale: float, v_floor: float, v_cap: float) -> bool:
    """Return True when a file-mean consumer scale is within the frozen bounds.

    Per-patch ``v`` is clamped into ``[v_floor, v_cap]`` by the frozen forward,
    so the file mean inherits containment up to two float32 rounding steps
    (clamp ``log``/``exp``, then mean accumulation; retained C04 CPU
    characterization: worst bound excursion ~1.1e-7, ~8.6x margin). The
    per-side allowance ``8 * eps32`` relative to each bound covers that
    rounding and rejects anything larger; it never loosens the bounds
    themselves. Each side scales with its own bound, so the frozen
    ``v_cap/v_floor <= 10`` edge rule cannot widen the floor-side margin
    through the cap magnitude.
    """
    if not math.isfinite(scale) or scale < 0.0:
        return False
    lo_tol = CONSUMER_SCALE_REL_TOL * abs(v_floor)
    hi_tol = CONSUMER_SCALE_REL_TOL * abs(v_cap)
    return (v_floor - lo_tol) <= scale <= (v_cap + hi_tol)


def _load_cache_records(cache_path: Path) -> tuple[dict, list[dict]]:
    with open(cache_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "s18-t23-c01-prediction-cache-v1":
        raise _fail("prediction cache schema changed")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise _fail("prediction cache records missing")
    return payload, records


def records_from_cache(cache_records: list[dict], history_ids: list[str]) -> dict[str, list]:
    """Build N4ScaleRecord pools per Fit history from bound cache latents.

    Context-only: uses cached detached context_latents plus the scalar residual
    recomputed EXACTLY as n4_patch_residual(layer_norm(target),
    layer_norm(pred)) restricted to eligible_patch_index (masked valid).
    Target/predicted latents never enter the scale input. Residuals are the
    fit target, not an input. No re-encoding, no re-masking, no label use.
    """
    import torch as _torch

    from representation.crn import N4ScaleRecord, n4_patch_residual

    pools: dict[str, list] = {hid: [] for hid in history_ids}
    for rec in cache_records:
        hid = rec.get("actual_history_id")
        if hid not in pools:
            continue  # Calibration/Development records excluded: Fit-only scope
        if rec.get("role") != "FIT":
            raise _fail(f"non-FIT role inside Fit history {hid}: {rec.get('role')}")
        elig = rec["patch_mapping"]["eligible_patch_index"]
        if not elig:
            raise _fail(f"cache record has no eligible patches: {rec.get('file_id')}")
        ctx = _torch.tensor(rec["context_latents"], dtype=_torch.float32)
        tgt = _torch.tensor(rec["target_latents"], dtype=_torch.float32)
        prd = _torch.tensor(rec["predicted_latents"], dtype=_torch.float32)
        if ctx.shape != tgt.shape or ctx.shape != prd.shape or ctx.ndim != 2 or ctx.shape[1] != 128:
            raise _fail(f"cache latent geometry changed: {rec.get('file_id')}")
        cached_res = rec.get("patch_residuals")
        recomputed = n4_patch_residual(tgt, prd)
        for pos, patch_index in enumerate([int(i) for i in elig]):
            exp_res = float(cached_res[patch_index])
            got_res = float(recomputed[patch_index].item())
            if exp_res != got_res:
                raise _fail(f"cache residual mismatch {rec.get('file_id')}:{patch_index}")
            context_row = ctx[patch_index].detach().cpu().float()
            if not bool(_torch.isfinite(context_row).all()):
                raise _fail("cache context has nonfinite entries")
            pools[hid].append(
                N4ScaleRecord(
                    history_id=hid,
                    file_id=str(rec["file_id"]),
                    patch_index=int(patch_index),
                    context=context_row,
                    residual=got_res,
                )
            )
    for hid, pool in pools.items():
        if not pool:
            raise _fail(f"empty N4 record pool for {hid}")
        pool.sort(key=lambda r: (r.history_id, r.file_id, r.patch_index))
    return pools


def run_preflight(args, bdoc: dict) -> dict:
    from representation.sprint18_harness import (
        FIT_HEALTHY_ROWS,
        deterministic_sampler_order,
        load_task6_binding,
        make_sprint18_arm_config,
    )

    worktree = Path(args.worktree)
    checks = self_check(args, bdoc)
    _check_contract_files(worktree)

    lock = load_task14_v3_lock(worktree)
    lock_binding = validate_task14_v3_lock(
        lock, worktree=worktree, observed_runner_sha256=checks["driver_sha256"]
    )
    legacy_b0 = load_legacy_b0_lock(worktree)
    binding = load_task6_binding(args.handoff)
    if binding.executable_binding_sha256 != EXPECTED_CONTRACT["executable_binding_sha256"]:
        raise _fail("executable binding digest changed")
    fit_histories = [h.actual_history_id for h in binding.role_histories("FIT")]
    if fit_histories != ["S18I-ITER-0005-FIT-01", "S18I-ITER-0005-FIT-02", "S18I-ITER-0005-FIT-03"]:
        raise _fail(f"Fit history roster changed: {fit_histories}")
    for seed in MODEL_SEEDS:
        ckpt_path = Path(bdoc["fixture"]["checkpoints"][str(seed)]["path"])
        if not ckpt_path.is_file():
            raise _fail(f"final checkpoint missing for seed {seed}")
        if _sha_file(ckpt_path) != EXPECTED_CONTRACT["checkpoints"][str(seed)]:
            raise _fail(f"final checkpoint hash mismatch for seed {seed}")
        cache_path = Path(bdoc["fixture"]["prediction_cache_paths"][str(seed)])
        if not cache_path.is_file():
            raise _fail(f"prediction cache missing for seed {seed}")
        if _sha_file(cache_path) != EXPECTED_CONTRACT["cache_files"][str(seed)]:
            raise _fail(f"prediction cache hash mismatch for seed {seed}")
    from representation.crn import (
        N4_GUARD_CONTRACT,
        N4_SCALE_CHECKPOINT_SCHEMA,
        N4_SCALE_PARAMETERIZATION,
        n4_canonical_fold_splits,
    )

    splits = n4_canonical_fold_splits(binding)
    if len(splits) != 3:
        raise _fail("canonical fold splits are not exactly three")
    sampler_ok = 0
    for seed in MODEL_SEEDS:
        arm_s = make_sprint18_arm_config(
            "N4-CRN", seed,
            {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None},
            selection_state="locked",
            coefficient_lock_sha256=lock["coefficient_lock_sha256"],
        )
        for training_ids, _heldout in splits:
            _, order = deterministic_sampler_order(
                binding, arm_s, workload="n4_scale", scope="experiment",
                role_history_ids=training_ids, epoch=1,
                eligible_source_ids=[f"preflight-placeholder-{i}" for i in range(8)],
            )
            if len(order) != 8:
                raise _fail("N4 sampler smoke permutation malformed")
            sampler_ok += 1
    preflight = {
        "schema": "s18-task13b-c01-preflight-v3",
        "task_id": "S18-T13B",
        "attempt_id": ATTEMPT,
        "source_commit": checks["head"],
        "driver_sha256": checks["driver_sha256"],
        "binding_sha256": checks["binding_sha256"],
        "handoff_sha256": _sha_file(Path(args.handoff)),
        "executable_binding_sha256": binding.executable_binding_sha256,
        "protocol_sha256": EXPECTED_CONTRACT["protocol_sha256"],
        "mechanism_spec_sha256": {"N4-CRN": EXPECTED_CONTRACT["mechanism_N4-CRN"]},
        "mechanism_spec_base_sha256": {"N4-CRN": EXPECTED_CONTRACT["mechanism_N4-CRN-v2-base"]},
        "mechanism_spec_path": MECHANISM_SPEC_PATH,
        "scale_parameterization": N4_SCALE_PARAMETERIZATION,
        "guard_contract": N4_GUARD_CONTRACT,
        "scale_checkpoint_schema": N4_SCALE_CHECKPOINT_SCHEMA,
        "task14_lock_binding": {**lock_binding, "status": "locked_executable"},
        "b0_legacy_lock_binding": dict(legacy_b0),
        "legacy_a03_disposition": dict(LEGACY_A03_DISPOSITION),
        "fit_history_ids": fit_histories,
        "fit_eligible_rows": dict(EXPECTED_CONTRACT["fit_rows"]),
        "fit_healthy_rows": FIT_HEALTHY_ROWS,
        "final_checkpoints_sha256": dict(EXPECTED_CONTRACT["checkpoints"]),
        "prediction_cache_sha256": dict(EXPECTED_CONTRACT["cache_files"]),
        "n4_sampler_smoke_ok": sampler_ok,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "PREFLIGHT_OK": True,
    }
    return preflight


def run_pilot(args, bdoc: dict) -> dict:
    from representation.config import V1Config
    from representation.crn import (
        N4_GUARD_CONTRACT,
        N4_SCALE_BATCH_SIZE,
        N4_SCALE_CHECKPOINT_SCHEMA,
        N4_SCALE_EPOCHS,
        N4_SCALE_PARAMETERIZATION,
        N4ScaleGuardError,
        fit_n4_scale_scope,
        init_n4_scale_model,
        load_n4_scale_checkpoint,
        n4_canonical_fold_splits,
        n4_residual_bounds,
        n4_scale_nll,
        n4_score_files,
        save_n4_scale_checkpoint,
        validate_n4_capacity,
    )
    from representation.model import V1RepresentationModel
    from representation.sprint18_checkpoint import restore_sprint18_checkpoint, state_sha256
    from representation.sprint18_harness import (
        arm_config_sha256,
        deterministic_mask_seed,
        load_certified_sample_subset,
        load_task6_binding,
        make_sprint18_arm_config,
    )
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    t0 = time.time()
    worktree = Path(args.worktree)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    checks = self_check(args, bdoc)
    _check_contract_files(worktree)

    # The digests are read early (as before) so the real fit path does not
    # depend on the preflight dict shape; run_preflight below fails closed
    # first. N4-CRN binds the prospective v3 digest; the B0 restore arms keep
    # the historical v1 digest their checkpoints were produced under.
    lock = load_task14_v3_lock(worktree)
    lock_digest = lock["coefficient_lock_sha256"]
    legacy_b0 = load_legacy_b0_lock(worktree)
    binding = load_task6_binding(args.handoff)
    preflight = run_preflight(args, bdoc)
    (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8")
    progress_path = out_dir / "progress.jsonl"
    if progress_path.exists():
        raise _fail("out/progress.jsonl already exists; refusing to overwrite a frozen run")

    def emit(obj: dict) -> None:
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(obj, sort_keys=True) + "\n")

    device = torch.device(args.device)
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True, pad_value=0.0))
    fit_histories = [h.actual_history_id for h in binding.role_histories("FIT")]
    splits = n4_canonical_fold_splits(binding)
    seeds = list(MODEL_SEEDS)
    all_fold_results: list[dict] = []
    per_seed_verdicts: dict[str, dict] = {}
    fold_checkpoints: list[dict] = []
    support_proof: dict[str, dict] = {}
    invariance_proof: dict[str, dict] = {}
    baseline_proof: dict[str, dict] = {}
    consumer_proof: dict[str, dict] = {}

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cache_path = Path(bdoc["fixture"]["prediction_cache_paths"][str(seed)])
        chead, cache_records = _load_cache_records(cache_path)
        mcfg = dict(chead["model_config"])
        config = V1Config(
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
        model = V1RepresentationModel(config, patchifier=patchifier)
        arm_b0 = make_sprint18_arm_config(
            "B0-FULL", seed, B0_FULL_COEFFICIENTS,
            selection_state="locked",
            coefficient_lock_sha256=legacy_b0["coefficient_lock_sha256"],
        )
        if arm_config_sha256(arm_b0) != legacy_b0["config_sha256"][str(seed)]:
            raise _fail(f"B0-FULL legacy lock binding differs for seed {seed}")
        ckpt_path = Path(bdoc["fixture"]["checkpoints"][str(seed)]["path"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
        restored = restore_sprint18_checkpoint(
            ckpt_path, model, optimizer, arm=arm_b0, binding=binding, allowed_scope="experiment",
        )
        if restored.checkpoint_sha256 != EXPECTED_CONTRACT["checkpoints"][str(seed)]:
            raise _fail(f"strict restore digest mismatch for seed {seed}")
        if state_sha256(model.state_dict()) != bdoc["fixture"]["live_model_state_sha256"][str(seed)]:
            raise _fail(f"restored online bytes differ from Task23 live digest for seed {seed}")
        if state_sha256(model.target_encoder.state_dict()) != bdoc["fixture"]["ema_state_sha256"][str(seed)]:
            raise _fail(f"restored EMA bytes differ from Task23 live digest for seed {seed}")
        model.to(device)
        model.eval()
        init_sha = state_sha256(model.state_dict())
        ema_before = state_sha256(model.target_encoder.state_dict())
        model_before = {k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v) for k, v in model.state_dict().items()}
        baseline_proof[str(seed)] = {
            "kind": "exact_final_B0-FULL_epoch100",
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": restored.checkpoint_sha256,
            "live_model_state_sha256": init_sha,
            "live_ema_state_sha256": ema_before,
            "cache_path": str(cache_path),
            "cache_file_sha256": _sha_file(cache_path),
            "cache_inner_sha256": str(chead.get("cache_sha256")),
            "cache_records": len(cache_records),
            "representation_epochs_completed": 100,
            "representation_optimizer_steps": 0,
            "coefficient_lock_sha256": arm_b0.coefficient_lock_sha256,
            "arm_config_sha256": arm_config_sha256(arm_b0),
        }
        arm = make_sprint18_arm_config(
            "N4-CRN", seed,
            {"lambda_info_max": None, "lambda_hcc_max": None, "alpha": None, "beta": None},
            selection_state="locked", coefficient_lock_sha256=lock_digest,
        )
        per_history_records = records_from_cache(cache_records, fit_histories)
        for hid in fit_histories:
            support_proof[f"{seed}|{hid}"] = {
                "cache_records": len(per_history_records[hid]),
                "masking_rule": "bound Task23 cache mask_seed per (file, history); no re-masking",
            }
        seed_fold_results: list[dict] = []
        seed_counts: list[int] = []
        for fold_index, (training_ids, heldout_id) in enumerate(splits):
            train_pool: list = []
            for hid in training_ids:
                train_pool.extend(per_history_records[hid])
            heldout_pool = list(per_history_records[heldout_id])
            train_residuals = np.array([r.residual for r in train_pool], dtype=np.float64)
            bounds = n4_residual_bounds(train_residuals)
            q05 = float(np.quantile(train_residuals, 0.05, method="linear"))
            q95 = float(np.quantile(train_residuals, 0.95, method="linear"))
            if bounds["q05"] != q05 or bounds["q95"] != q95:
                raise _fail("fold quantile recomputation differs from n4_residual_bounds")
            if bounds["v_floor"] != max(q05, 1e-6) or bounds["v_cap"] != max(q95, 10.0 * bounds["v_floor"]):
                raise _fail("fold bound edge rules differ from the frozen formulas")
            scale = init_n4_scale_model(128, seed)
            scale_before_nparams = sum(p.numel() for p in scale.parameters())
            if scale_before_nparams <= 0 or scale_before_nparams > 100_000:
                raise _fail("scale capacity outside the frozen envelope")
            scope = f"fold:{heldout_id}"
            try:
                result = fit_n4_scale_scope(
                    records=train_pool, training_history_ids=training_ids, scope=scope,
                    arm=arm, binding=binding, scale_model=scale,
                    v_floor=bounds["v_floor"], v_cap=bounds["v_cap"],
                    heldout_records=heldout_pool, device=device,
                )
            except N4ScaleGuardError as exc:
                failure = {
                    "schema": "s18-task13b-v3-guard-failure-v1",
                    "task_id": "S18-T13B",
                    "attempt_id": ATTEMPT,
                    "model_seed": seed,
                    "fold_index": fold_index,
                    "scope": scope,
                    "training_history_ids": list(training_ids),
                    "heldout_history_id": heldout_id,
                    "guard_id": exc.guard_id,
                    "guard_contract": N4_GUARD_CONTRACT,
                    "message": str(exc),
                    "diagnostics": exc.diagnostics,
                    "optimizer_steps_applied": 0,
                    "checkpoint_written": False,
                    "run_status": "FAILED",
                }
                (out_dir / "guard_failure.json").write_text(
                    json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
                )
                emit({"event": "scale_guard_failure", "seed": seed,
                      "fold": fold_index, "scope": scope, "guard_id": exc.guard_id})
                raise
            if result["record_count"] != len(train_pool) or result["heldout_count"] != len(heldout_pool):
                raise _fail("fold record counts differ from the cached pools")
            expected_steps = N4_SCALE_EPOCHS * math.ceil(len(train_pool) / N4_SCALE_BATCH_SIZE)
            if result["optimizer_steps"] != expected_steps:
                raise _fail("fold optimizer steps differ from 50*ceil(N/512)")
            if not math.isfinite(result["final_train_nll"]) or not math.isfinite(result["heldout_nll"]):
                raise _fail("fold NLL is nonfinite")
            ckpt_name = f"seed{seed}-fold{fold_index}-{heldout_id}.pt"
            ckpt_path = out_dir / "checkpoints" / ckpt_name
            checkpoint_meta = save_n4_scale_checkpoint(
                ckpt_path, scale, fit_scope=scope,
                training_history_ids=training_ids,
                v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
                initialization={
                    "init_anchor": dict(result["init_anchor"]),
                    "init_interiority": dict(result["init_interiority"]),
                    "first_step_gradients": dict(result["first_step_gradients"]),
                },
            )
            ckpt_sha = _sha_file(ckpt_path)
            live_sha = state_sha256(scale.state_dict())
            if live_sha != checkpoint_meta["state_dict_sha256"]:
                raise _fail("v3 scale checkpoint state digest differs from the trained scale")
            restored_scale = init_n4_scale_model(128, seed + 999999)
            restored_meta = load_n4_scale_checkpoint(
                ckpt_path, restored_scale, expected_fit_scope=scope,
                expected_training_history_ids=training_ids,
                expected_bounds={
                    "v_floor": float(bounds["v_floor"]),
                    "v_cap": float(bounds["v_cap"]),
                },
            )
            if restored_meta["state_dict_sha256"] != live_sha:
                raise _fail("strict v3 restore digest differs from the trained scale")
            if restored_meta["parameterization"] != N4_SCALE_PARAMETERIZATION:
                raise _fail("strict v3 restore parameterization differs from the frozen identity")
            restored_sha = state_sha256(restored_scale.state_dict())
            if live_sha != restored_sha:
                raise _fail("restored scale bytes differ from the trained scale")
            with torch.no_grad():
                ho_sorted = sorted(heldout_pool, key=lambda r: (r.history_id, r.file_id, r.patch_index))
                heldout_context_cpu = torch.stack(
                    [r.context.detach().float() for r in ho_sorted],
                    dim=0,
                )
                heldout_target_cpu = torch.tensor(
                    [float(r.residual) for r in ho_sorted],
                    dtype=torch.float32,
                )
                if str(device) != "cpu":
                    heldout_context = heldout_context_cpu.to(device)
                    heldout_target = heldout_target_cpu.to(device)
                    live_out = scale(heldout_context, v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]))
                    restored_on_device = restored_scale.to(device).eval()
                    restored_out = restored_on_device(heldout_context, v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]))
                    fwd_diff = float((live_out["log_v"].detach().cpu() - restored_out["log_v"].detach().cpu()).abs().max().item())
                    restored_scale.to("cpu").eval()
                else:
                    live_out = scale(heldout_context_cpu, v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]))
                    restored_out = restored_scale(heldout_context_cpu, v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]))
                    fwd_diff = float((live_out["log_v"] - restored_out["log_v"]).abs().max().item())
                if fwd_diff != 0.0:
                    raise _fail(f"restored forward parity diff {fwd_diff} != 0.0")
                live_nll = float(n4_scale_nll(heldout_target_cpu if str(device) == "cpu" else heldout_target.detach().cpu(), live_out["log_v"].detach().cpu()).item())
                restored_nll = float(n4_scale_nll(heldout_target_cpu if str(device) == "cpu" else heldout_target.detach().cpu(), restored_out["log_v"].detach().cpu()).item())
                if live_nll != restored_nll:
                    raise _fail("restored NLL differs from live NLL")
                # Real public-consumer proof through the permanent
                # ``run_fold_consumer_proof`` block below (file selection,
                # scoring, fail-closed bounds — never duplicated logic).
                consumer_scores = run_fold_consumer_proof(
                    model=model, scale=scale, heldout_pool=heldout_pool,
                    heldout_id=heldout_id, binding=binding, worktree=worktree,
                    seed=seed, patchifier=patchifier, bounds=bounds,
                )
            consumer_proof[f"{seed}|{fold_index}"] = {
                "files": list(consumer_scores),
                "scores": {k: dict(v) for k, v in consumer_scores.items()},
            }
            run_record = {
                "schema": "s18-task13b-c01-fold-run-v3",
                "model_seed": seed,
                "fold_index": fold_index,
                "scope": scope,
                "training_history_ids": list(training_ids),
                "heldout_history_id": heldout_id,
                "scale_parameterization": result["scale_parameterization"],
                "guard_contract": result["guard_contract"],
                "init_anchor": dict(result["init_anchor"]),
                "init_interiority": dict(result["init_interiority"]),
                "first_step_gradients": dict(result["first_step_gradients"]),
                "record_count": result["record_count"],
                "heldout_count": result["heldout_count"],
                "bounds": dict(result["bounds"]),
                "quantiles": {"q05": q05, "q95": q95},
                "optimizer_steps": result["optimizer_steps"],
                "per_epoch_steps": result["per_epoch_steps"],
                "per_epoch_nll": list(result["per_epoch_nll"]),
                "final_train_nll": result["final_train_nll"],
                "heldout_nll": result["heldout_nll"],
                "heldout_spearman": result["heldout_spearman"],
                "saturation_train": dict(result["saturation_train"]),
                "saturation_heldout": dict(result["saturation_heldout"]),
                "cap_gate_pass_train": bool(result["cap_gate_pass_train"]),
                "epoch_order_seeds": list(result["epoch_order_seeds"]),
                "order_seed_rule": str(result["order_seed_rule"]),
                "scale_parameters": int(result["scale_parameters"]),
                "scale_state_sha256": live_sha,
                "checkpoint_schema": checkpoint_meta["schema"],
                "checkpoint_parameterization": checkpoint_meta["parameterization"],
                "checkpoint_path": f"checkpoints/{ckpt_name}",
                "checkpoint_sha256": ckpt_sha,
                "restored_forward_max_abs_diff": fwd_diff,
                "restored_nll": restored_nll,
            }
            (out_dir / "runs" / f"seed{seed}-fold{fold_index}.json").write_text(
                json.dumps(run_record, indent=2, sort_keys=True), encoding="utf-8"
            )
            fold_checkpoints.append({
                "seed": seed, "fold": fold_index, "checkpoint_sha256": ckpt_sha,
                "schema": checkpoint_meta["schema"],
                "parameterization": checkpoint_meta["parameterization"],
                "state_dict_sha256": checkpoint_meta["state_dict_sha256"],
            })
            seed_fold_results.append(result)
            seed_counts.append(len(train_pool))
            all_fold_results.append(run_record)
            for epoch_index, nll in enumerate(result["per_epoch_nll"], start=1):
                emit({"event": "epoch", "seed": seed, "fold": fold_index, "epoch": epoch_index, "nll": nll})
            emit({"event": "fold_complete", "seed": seed, "fold": fold_index,
                  "record_count": len(train_pool), "optimizer_steps": result["optimizer_steps"],
                  "heldout_nll": result["heldout_nll"]})
        per_seed_updates = N4_SCALE_EPOCHS * sum(math.ceil(c / N4_SCALE_BATCH_SIZE) for c in seed_counts)
        verdict = validate_n4_capacity(seed_fold_results, record_counts=seed_counts, per_seed_updates=per_seed_updates)
        per_seed_verdicts[str(seed)] = {
            "verdict": verdict["verdict"],
            "reasons": list(verdict["reasons"]),
            "record_counts": list(seed_counts),
            "per_seed_updates": per_seed_updates,
        }
        emit({"event": "seed_complete", "seed": seed, "verdict": verdict["verdict"],
              "per_seed_updates": per_seed_updates})
        after = model.state_dict()
        rep_equal = set(after) == set(model_before) and all(
            (torch.equal(after[k].detach().cpu(), v) if isinstance(after[k], torch.Tensor) and isinstance(v, torch.Tensor) else after[k] == v)
            for k, v in model_before.items()
        )
        ema_after = state_sha256(model.target_encoder.state_dict())
        invariance_proof[str(seed)] = {
            "restored_online_sha256": init_sha,
            "model_state_equal_bitwise": bool(rep_equal),
            "ema_before_sha256": ema_before,
            "ema_after_sha256": ema_after,
            "ema_equal": bool(ema_before == ema_after),
        }
        if not rep_equal or ema_before != ema_after:
            raise _fail(f"representation invariance failed for seed {seed}")
        del model, optimizer, scale, restored_scale
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    wall = time.time() - t0
    cuda_info: dict[str, object] = {"cuda_available": bool(torch.cuda.is_available()), "device_used": str(device)}
    if torch.cuda.is_available():
        try:
            cuda_info["device_name"] = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            cuda_info["mem_free_bytes"] = int(free_b)
            cuda_info["mem_total_bytes"] = int(total_b)
        except Exception as exc:  # noqa: BLE001 - record, never fail the run
            cuda_info["mem_probe_error"] = str(exc)
    results = {
        "schema": "s18-task13b-c01-results-v3",
        "task_id": "S18-T13B",
        "attempt_id": ATTEMPT,
        "source_commit": checks["head"],
        "driver_sha256": checks["driver_sha256"],
        "binding_sha256": _sha_file(Path(args.binding)),
        "handoff_sha256": _sha_file(Path(args.handoff)),
        "executable_binding_sha256": binding.executable_binding_sha256,
        "protocol_sha256": EXPECTED_CONTRACT["protocol_sha256"],
        "mechanism_spec_sha256": {"N4-CRN": EXPECTED_CONTRACT["mechanism_N4-CRN"]},
        "mechanism_spec_base_sha256": {"N4-CRN": EXPECTED_CONTRACT["mechanism_N4-CRN-v2-base"]},
        "mechanism_spec_path": MECHANISM_SPEC_PATH,
        "scale_parameterization": N4_SCALE_PARAMETERIZATION,
        "guard_contract": N4_GUARD_CONTRACT,
        "scale_checkpoint_schema": N4_SCALE_CHECKPOINT_SCHEMA,
        "task14_lock_binding": dict(preflight["task14_lock_binding"]),
        "b0_legacy_lock_binding": dict(preflight["b0_legacy_lock_binding"]),
        "legacy_a03_disposition": dict(LEGACY_A03_DISPOSITION),
        "model_seeds": seeds,
        "fit_history_ids": fit_histories,
        "support": support_proof,
        "representation_baseline": baseline_proof,
        "folds": all_fold_results,
        "per_seed_verdicts": per_seed_verdicts,
        "representation_invariance": invariance_proof,
        "consumer_proof": consumer_proof,
        "checkpoints": fold_checkpoints,
        "device": str(device),
        "cuda": cuda_info,
        "wall_seconds": wall,
        "preflight": preflight,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    capacity = {
        "schema": "s18-task13-n4-capacity-v3",
        "task_id": "S18-T13",
        "attempt_id": ATTEMPT,
        "source_commit": checks["head"],
        "driver_sha256": checks["driver_sha256"],
        "executable_binding_sha256": binding.executable_binding_sha256,
        "task14_lock_binding": dict(preflight["task14_lock_binding"]),
        "b0_legacy_lock_binding": dict(preflight["b0_legacy_lock_binding"]),
        "mechanism_spec_sha256": {"N4-CRN": EXPECTED_CONTRACT["mechanism_N4-CRN"]},
        "mechanism_spec_base_sha256": {"N4-CRN": EXPECTED_CONTRACT["mechanism_N4-CRN-v2-base"]},
        "scale_parameterization": N4_SCALE_PARAMETERIZATION,
        "guard_contract": N4_GUARD_CONTRACT,
        "scale_checkpoint_schema": N4_SCALE_CHECKPOINT_SCHEMA,
        "legacy_a03_disposition": dict(LEGACY_A03_DISPOSITION),
        "model_seeds": seeds,
        "fit_history_ids": fit_histories,
        "folds": [
            {
                "model_seed": r["model_seed"],
                "fold_index": r["fold_index"],
                "scope": r["scope"],
                "training_history_ids": r["training_history_ids"],
                "heldout_history_id": r["heldout_history_id"],
                "scale_parameterization": r["scale_parameterization"],
                "guard_contract": r["guard_contract"],
                "init_anchor": dict(r["init_anchor"]),
                "init_interiority": dict(r["init_interiority"]),
                "first_step_gradients": dict(r["first_step_gradients"]),
                "record_count": r["record_count"],
                "heldout_count": r["heldout_count"],
                "bounds": r["bounds"],
                "optimizer_steps": r["optimizer_steps"],
                "per_epoch_steps": r["per_epoch_steps"],
                "final_train_nll": r["final_train_nll"],
                "heldout_nll": r["heldout_nll"],
                "heldout_spearman": r["heldout_spearman"],
                "saturation_train": r["saturation_train"],
                "saturation_heldout": r["saturation_heldout"],
                "scale_parameters": r["scale_parameters"],
                "scale_state_sha256": r["scale_state_sha256"],
                "checkpoint_sha256": r["checkpoint_sha256"],
                "checkpoint_schema": r["checkpoint_schema"],
            }
            for r in all_fold_results
        ],
        "per_seed_verdicts": per_seed_verdicts,
        "representation_invariance": invariance_proof,
        "representation_baseline": baseline_proof,
        "consumer_proof": consumer_proof,
        "device": str(device),
        "wall_seconds": wall,
    }
    (out_dir / "capacity.json").write_text(json.dumps(capacity, indent=2, sort_keys=True), encoding="utf-8")
    return {"results": results, "capacity": capacity, "PILOT_EXIT": "ok"}


def run_fold_consumer_proof(
    *,
    model,
    scale,
    heldout_pool: list,
    heldout_id: str,
    binding,
    worktree,
    seed: int,
    patchifier,
    bounds: dict,
) -> dict[str, dict[str, float]]:
    """Score the bounded public-consumer file triple on qualified heldout rows.

    Permanent runner consumer block (normative path): selects the first
    ``CONSUMER_FILES_PER_FOLD`` heldout files, loads the qualified
    heldout-history waveform rows, scores each file with its frozen per-file
    inference mask seed through the frozen B0 plus the fitted scale head, and
    enforces finite/non-negative outputs plus file-mean scale containment
    modulo the frozen per-side float32 rounding allowance. Local fixtures are
    exercised through this same function; the block is never duplicated.
    """
    from representation.sprint18_harness import deterministic_mask_seed
    from representation.crn import n4_score_files

    file_sorted = sorted(heldout_pool, key=lambda r: (r.history_id, r.file_id, r.patch_index))
    seen_files: list[str] = []
    for r in file_sorted:
        if r.file_id not in seen_files:
            seen_files.append(r.file_id)
    consumer_files = seen_files[:CONSUMER_FILES_PER_FOLD]
    consumer_rows = _consumer_samples(
        binding, worktree, heldout_id, consumer_files,
    )
    consumer_scores: dict[str, dict[str, float]] = {}
    for crow in consumer_rows:
        per_file_seed = deterministic_mask_seed(seed, heldout_id, crow.file_id)
        single = n4_score_files(
            model=model, samples=[crow.sample],
            scale_model=scale, v_floor=float(bounds["v_floor"]), v_cap=float(bounds["v_cap"]),
            patchifier=patchifier, masking_seed=per_file_seed,
        )
        consumer_scores.update(single)
    if set(consumer_scores) != set(consumer_files):
        raise _fail("public consumer file coverage differs from the requested files")
    for _fid, triple in consumer_scores.items():
        for _k in ("raw", "scale", "normalized"):
            if not math.isfinite(triple[_k]) or triple[_k] < 0.0:
                raise _fail("public consumer score is nonfinite/negative")
        if not consumer_scale_within_bounds(float(triple["scale"]), float(bounds["v_floor"]), float(bounds["v_cap"])):
            raise _fail("public consumer scale escapes the frozen bounds")
    return dict(consumer_scores)

def _consumer_samples(binding, worktree, heldout_id: str, file_ids: list[str]) -> list:
    """Load qualified heldout-history rows for the public-consumer proof.

    Uses the committed qualified loader (train_fit permission on the heldout
    Fit history) so the consumer path scores real waveform patches through
    the restored B0 plus the fitted scale head. Only the first
    CONSUMER_FILES_PER_FOLD heldout files per fold are requested (bounded
    proof, not resampling).
    """
    from representation.sprint18_harness import load_certified_sample_subset

    worktree = Path(worktree)
    history = binding.history(heldout_id)
    role_dir = worktree / history.relative_path
    manifest_path = role_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    # Shard map must cover EXACTLY the selected shards (loader fails closed
    # on over/under-coverage): resolve which shard holds each requested file.
    shard_of: dict[str, str] = {}
    for row in manifest["shards"]:
        for fid in row["file_ids"]:
            shard_of[str(fid)] = str(row["path"])
    selected = sorted({shard_of[fid] for fid in file_ids})
    shard_paths = {rel: str(role_dir / rel) for rel in selected}
    rows = load_certified_sample_subset(
        binding, heldout_id, manifest_path, shard_paths, list(file_ids),
        role_action="train_fit",
    )
    if [r.file_id for r in rows] != list(file_ids):
        raise _fail("consumer row identity differs from the requested heldout files")
    if not all(r.is_verified_healthy for r in rows):
        raise _fail("consumer rows fail the qualified healthy predicate")
    return list(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--mode", required=True, choices=("preflight", "pilot"))
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    bdoc = load_binding_doc(Path(args.binding))
    _execution_binding(bdoc, args)  # authorize execution identity before any preflight/pilot work
    if args.mode == "preflight":
        preflight = run_preflight(args, bdoc)
        print(json.dumps({"PREFLIGHT_OK": True, **{k: preflight[k] for k in ("source_commit", "driver_sha256", "fit_eligible_rows", "fit_healthy_rows", "n4_sampler_smoke_ok", "executable_binding_sha256", "task14_lock_binding", "b0_legacy_lock_binding")}}, sort_keys=True))
        print(json.dumps({"PILOT_EXIT": "preflight_ok"}))
        return 0
    outcome = run_pilot(args, bdoc)
    print(json.dumps({"PILOT_EXIT": outcome["PILOT_EXIT"], "seeds": outcome["results"]["model_seeds"],
                      "folds": len(outcome["results"]["folds"]),
                      "verdicts": outcome["results"]["per_seed_verdicts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
