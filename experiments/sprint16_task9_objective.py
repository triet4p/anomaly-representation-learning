"""Sprint 16 Task 9 (M9): objective/masking pressure diagnostics.

Frozen-checkpoint diagnostics plus the protocol-bounded C5 tiny-retrain
path, architecture fixed. Modes:

- frozen-diag: loads an accepted checkpoint (sha256 asserted), runs on a
  fixed Fit-healthy dev batch (first-N files whose TRUE conditioning
  indices lie in the checkpoint's fitted range — counted, never remapped;
  real regime ids via patch_regime_ids): per-term gradient-norm split
  (boundary/background/variance/covariance via per-term autograd.grad —
  weights never stepped, asserted unchanged), predicted-mean tracking
  (cosine between predicted-mean shift and actual corrupt shift, median +
  paired-bootstrap CI), severity response, and nuisance probes. No weight
  update in this mode.
- retrain: executes ONE v5 variant (control only, exactly 300 steps,
  one-shot output guard, accepted-weights init, one knob zeroed).
  Fail-closed gates: real-knob verification, control-only, no-replacement.
- eval-retrain: evaluates a given retrained checkpoint's local latents
  per-category on Confirmation (Task-7-style readout) for future use.
- self-test: synthetic plumbing checks, no checkpoint/data.

Budget (protocol hard caps, enforced in-code): at most 2 gradient runs,
each exactly 300 steps. Sealed roots are never
touched (only FIT paths in frozen-diag; CONFIRMATION only in eval-retrain).
Usage (server, verified repo root):
  .venv/bin/python experiments/sprint16_task9_objective.py --mode frozen-diag \\
    --checkpoint control --data-root <roots> --out <dir> --device cuda
Deterministic unless stated; eval mode throughout frozen-diag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

CHECKPOINTS = {
    "control": {
        "path": ("/tmp/sprint11-task30-corrected-control/full/"
                 "control-normal-only/default/v2_checkpoint.pt"),
        "sha256": "8bdb845b17a788ad39101e0f98c0654edb55727f0eec540ca44046b63dba2b7c",
    },
    "hybrid": {
        "path": ("/tmp/sprint11-task31-corrected-hybrid/full/"
                 "hybrid-boundary/default/v2_checkpoint.pt"),
        "sha256": "76be843b0a5fa94c8cdd646e2734c804498f9c38088ed71c3f16f9aa97e2cc94",
    },
}
FIT = (("H-FIT-28", 1604), ("H-FIT-29", 1605), ("H-FIT-30", 1606))
CONF = (("H-CONF-34", 1608), ("H-CONF-35", 1609), ("H-CONF-36", 1610),
        ("H-CONF-37", 1611))
CATEGORIES = ["P1", "P2", "W1", "W2", "A1", "A2", "P", "W", "A"]
SEVERITIES = (1.0, 2.0, 4.0)
CORRUPT_EVERY = 4  # deterministic: every 4th valid patch is corrupted
DEV_FILES = 8  # fixed first-N compatible Fit-healthy files per root
MAX_GRAD_RUNS = 2
MAX_GRAD_STEPS = 300
RETRAIN_STEPS = 300  # exact, no early stopping (avoids selection bias)
RETRAIN_LR = 1e-3  # accepted stack default (build_v2_training_stack)
RETRAIN_SEVERITY = 2.0  # frozen single severity for retrain batches
RETRAIN_GEN_SEED = 20260202

#: v5 C5 variants: exactly two FIT-healthy-only, architecture-fixed
#: diagnostic ablations of the REAL counterfactual objective. Both execute
#: on CONTROL only (v5 §2 budget accounting); any other checkpoint is
#: refused. Each executes at most once (existing output dir refuses).
C5_VARIANTS = {
    "nobackground": {"knob": "background_weight", "value": 0.0},
    "nocovariance": {"knob": "covariance_weight", "value": 0.0},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_c5_knob(variant: str) -> None:
    """Fail closed: the v5 C5 variant knob must exist to run.

    Returns silently when the knob is a real CounterfactualCriterion
    parameter (or V2Config field). Raises ValueError for anything else —
    including the superseded v3 mask/contrastive names, which are absent
    from the accepted stack. Never substitutes an unapproved variant.
    """
    import inspect

    from representation import v2_objectives as O
    from representation.v2_config import V2Config

    spec = C5_VARIANTS[variant]
    knob = spec["knob"]
    sig = str(inspect.signature(O.CounterfactualCriterion.__init__))
    fields = set(V2Config.model_fields)
    if knob in sig or knob in fields:
        return
    raise ValueError(
        f"C5 variant {variant!r} requires knob {knob!r}, absent from the "
        f"accepted V2 stack. Retrain refused without a protocol amendment.")


def audit_absent_objectives(model) -> dict[str, object]:
    """Document V1-objective absence on the loaded accepted model (v5 §1).

    Returns module/config/signature evidence that masked prediction, EMA
    target construction, masking schedules, and contrastive weighting are
    NOT PRESENT. Any presence finding raises (fail closed — the v5 premise
    would be false).
    """
    import inspect

    from representation import v2_objectives as O
    from representation.v2_config import V2Config

    modules = sorted(type(m).__name__ for m in model.modules())
    bad_modules = [m for m in modules
                   if "EMA" in m or "Masked" in m or "Contrastive" in m]
    if bad_modules:
        raise ValueError(f"unexpected objective modules: {bad_modules}")
    fields = set(V2Config.model_fields)
    bad_fields = [f for f in fields
                  if "mask" in f or "contrast" in f or "ema" in f.lower()]
    if bad_fields:
        raise ValueError(f"unexpected objective fields: {bad_fields}")
    sig = str(inspect.signature(O.CounterfactualCriterion.__init__))
    return {"ema_target_modules": [],
            "masked_prediction_modules": [],
            "mask_schedule_fields": [],
            "contrastive_terms": [],
            "criterion_signature": sig,
            "verdict": "NOT PRESENT (all four V1 objective families absent)"}


def tracking_cosine(pred_shift, true_shift) -> dict[str, float]:
    """Median cosine(pred_shift, true_shift) + paired-bootstrap CI.

    Frozen convention: default_rng(BOOTSTRAP_SEED), B=BOOTSTRAP_REPLICATES,
    2.5th/97.5th percentiles.
    """
    import numpy as np

    from representation import attribution_metrics as M

    a = np.asarray(pred_shift, dtype=np.float64)
    b = np.asarray(true_shift, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] == 0:
        raise ValueError("shifts must be non-empty [N,D] pairs")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("shifts must be finite")
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.where(denom > 0, (a * b).sum(axis=1) / np.maximum(denom, 1e-12), 0.0)
    rng = np.random.default_rng(M.BOOTSTRAP_SEED)
    draws = rng.integers(0, cos.size, size=(M.BOOTSTRAP_REPLICATES, cos.size))
    boots = np.median(cos[draws], axis=1)
    return {"median": float(np.median(cos)),
            "lcb": float(np.quantile(boots, 0.025)),
            "ucb": float(np.quantile(boots, 0.975)),
            "n": int(cos.size)}


def self_test() -> int:
    """Synthetic plumbing checks (no checkpoint/data)."""
    import numpy as np

    rng = np.random.default_rng(2)
    a = rng.normal(size=(200, 8))
    t = tracking_cosine(a, a)
    assert t["median"] == 1.0, t
    assert t["lcb"] <= 1.0 <= t["ucb"]
    b = rng.normal(size=(200, 8))
    t2 = tracking_cosine(a, b)
    assert -1.0 <= t2["median"] <= 1.0
    try:
        tracking_cosine(a[:10], a[:5])
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must raise")
    for variant in ("nobackground", "nocovariance"):
        check_c5_knob(variant)  # must NOT raise: real criterion knobs
    for variant in ("mask015", "nocontrast"):
        try:
            check_c5_knob(variant)
        except KeyError:
            pass  # superseded v3 names are not in C5_VARIANTS at all
        else:
            raise AssertionError(f"{variant} must not be a v5 variant")
    try:
        check_c5_knob("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown variant must raise KeyError")
    assert MAX_GRAD_RUNS == 2 and MAX_GRAD_STEPS == 300
    assert RETRAIN_STEPS == 300
    print(json.dumps({"self_test": "PASS"}))
    return 0


def _wrap(x):
    """Minimal patchify input (only .x is consumed)."""
    import numpy as _np

    return SimpleNamespace(x=_np.asarray(x, dtype=_np.float32))


def _auc_ci(pos: list[float], neg: list[float]) -> dict[str, float]:
    import numpy as _np

    from representation import attribution_metrics as _M

    if not pos or not neg:
        raise ValueError("auc requires non-empty pos and neg")
    scores = _np.array(list(pos) + list(neg), dtype=_np.float64)
    labels = _np.array([1.0] * len(pos) + [0.0] * len(neg))
    rng = _np.random.default_rng(_M.BOOTSTRAP_SEED)
    draws = rng.integers(0, scores.size,
                         size=(_M.BOOTSTRAP_REPLICATES, scores.size))
    aucs = _np.asarray([_M.tie_auc(scores[r], labels[r]) for r in draws])
    aucs = aucs[_np.isfinite(aucs)]
    if aucs.size == 0:
        raise ValueError("no finite bootstrap draws")
    return {"point": float(_M.tie_auc(scores, labels)),
            "lcb": float(_np.quantile(aucs, 0.025)),
            "ucb": float(_np.quantile(aucs, 0.975)),
            "n_pos": len(pos), "n_neg": len(neg)}


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("frozen-diag", "retrain", "eval-retrain"),
                    default="frozen-diag")
    ap.add_argument("--checkpoint", choices=("control", "hybrid"), default="control")
    ap.add_argument("--variant", choices=sorted(C5_VARIANTS), default="nobackground")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    import numpy as np
    import torch

    from representation import attribution_metrics as M
    from representation.v2_inference import V2InferencePipeline, patch_regime_ids
    from representation.v2_objectives import (
        CounterfactualCriterion, synthesize_corrupted_patches)
    from synth import balanced as B
    from synth.chronicle import load_chronological
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    if args.mode == "retrain":
        return _retrain(args)

    if args.mode == "eval-retrain":
        return _eval_retrain(args)

    # ---------------- frozen-diag ----------------
    if not args.data_root or not args.out:
        ap.error("--data-root and --out are required without --self-test")
    spec = CHECKPOINTS[args.checkpoint]
    ck_path = Path(spec["path"])
    if not ck_path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {ck_path}")
    digest = sha256_file(ck_path)
    if digest != spec["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {digest}")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but unavailable")
    outdir = Path(args.out) / args.checkpoint
    outdir.mkdir(parents=True, exist_ok=True)
    log = {"mode": "frozen-diag", "checkpoint": args.checkpoint,
           "sha256": digest, "device": device, "steps": []}
    pipe = V2InferencePipeline.load(str(ck_path), device=device)

    model = pipe.model.eval()
    absence = audit_absent_objectives(model)
    log["steps"].append({"absence_audit": absence})
    for p in model.parameters():
        p.requires_grad_(True)
    before = sum(p.detach().double().sum().item() for p in model.parameters())
    patchifier = Patchifier(PatchConfig())
    data_root = Path(args.data_root)

    def load_root(role: str, group: str):
        root = data_root / group / role
        manifest = json.loads((root / "manifest.json").read_text())
        samples, _ = load_chronological(root)
        return samples, manifest

    # Fixed Fit-healthy dev batch: deterministic file order, first N whose
    # TRUE conditioning indices lie in the fitted range (counted, never
    # remapped); real regime ids via patch_regime_ids.
    dev = []
    dev_skipped_idx = 0
    for role, seed in FIT:
        samples, manifest = load_root(role, "FIT")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, f"tag:{role}"
        by_id = {s.file_id: s for s in samples}
        taken = 0
        for row in manifest["files"]:
            if taken >= DEV_FILES:
                break
            if row["file_label"] != "normal" or row["is_quarantined"]:
                continue
            if (row["program_id"] == B.PROGRAM_RESERVE
                    or row["robot_id"] == B.ROBOT_RESERVE):
                continue
            sample = by_id[row["file_id"]]
            if sample.robot_idx > 2 or sample.program_idx > 2:
                dev_skipped_idx += 1
                continue
            x = np.asarray(sample.x, dtype=np.float64)
            batch = patchifier.patchify(_wrap(x))
            patches = np.asarray(batch.patches, dtype=np.float64)
            valid = np.asarray(batch.valid_len, dtype=int)
            pad = np.asarray(batch.pad_mask, dtype=bool)
            keep = np.flatnonzero(valid > 0)
            if keep.size == 0:
                continue
            starts = np.asarray(batch.starts, dtype=int)[keep]
            regs = patch_regime_ids([sample], starts.reshape(1, -1),
                                    int(keep.size))
            dev.append((patches[keep], pad[keep],
                        int(sample.robot_idx), int(sample.program_idx),
                        np.asarray(regs[0]), row["file_id"]))
            taken += 1
    log["steps"].append(
        {"dev_files": sum(1 for _ in dev),
         "dev_skipped_idx": dev_skipped_idx,
         "dev_patches": int(sum(d[0].shape[0] for d in dev))})

    cfg = pipe.config
    criterion = CounterfactualCriterion(
        boundary_margin=cfg.boundary_margin,
        background_weight=cfg.background_weight,
        variance_weight=cfg.variance_weight,
        covariance_weight=cfg.covariance_weight)
    gen = torch.Generator().manual_seed(20260202)
    grad_splits = []
    for severity in SEVERITIES:
        tvals, gnorms, track, satisf = [], {}, [], []
        for pw, pm, ri, pi, regs, fid in dev:
            K = pw.shape[0]
            cmask = np.zeros(K, dtype=bool)
            cmask[::CORRUPT_EVERY] = True
            ten = {
                "patches": torch.asarray(pw, dtype=torch.float32).unsqueeze(0),
                "patch_pad_mask": torch.asarray(pm, dtype=torch.bool).unsqueeze(0),
                "patch_valid_mask": torch.ones((1, K), dtype=torch.bool),
                "robot_idx": torch.tensor([ri], dtype=torch.long),
                "program_idx": torch.tensor([pi], dtype=torch.long),
                "regime_ids": torch.asarray(regs, dtype=torch.long).unsqueeze(0),
            }
            ten = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                   for k, v in ten.items()}
            cm = torch.asarray(cmask).unsqueeze(0)
            clean = model(**ten)
            corrupt_pw = synthesize_corrupted_patches(
                ten["patches"], ten["patch_pad_mask"], ten["patch_valid_mask"],
                cm.to(device), float(severity), generator=gen)
            corrupt = model(patches=corrupt_pw,
                            patch_pad_mask=ten["patch_pad_mask"],
                            patch_valid_mask=ten["patch_valid_mask"],
                            robot_idx=ten["robot_idx"],
                            program_idx=ten["program_idx"],
                            regime_ids=ten["regime_ids"])
            terms = criterion(
                clean["patch_latents"], corrupt["patch_latents"],
                clean["context_energy"], corrupt["context_energy"],
                ten["patch_valid_mask"], cm.to(device), step=0)
            tvals.append({k: float(v.detach().cpu().item())
                          for k, v in terms.items() if k != "loss"})
            model.zero_grad(set_to_none=True)
            params = [p for p in model.parameters() if p.requires_grad]
            for name in ("boundary_loss", "background_loss", "normal_loss",
                         "density_raw", "variance_raw", "covariance_raw"):
                if name not in terms:
                    continue
                g = torch.autograd.grad(
                    terms[name], params, retain_graph=True, allow_unused=True)
                vals = [float(x.detach().abs().sum().item())
                        for x in g if x is not None]
                gnorms.setdefault(name, []).append(float(sum(vals)))
            cl = clean["patch_latents"].detach().cpu().numpy()[0]
            kl = corrupt["patch_latents"].detach().cpu().numpy()[0]
            cm_np = clean["cond_mean"].detach().cpu().numpy()[0]
            km_np = corrupt["cond_mean"].detach().cpu().numpy()[0]
            m = cmask & np.ones(K, dtype=bool)
            track.append(tracking_cosine(km_np[m] - cm_np[m], kl[m] - cl[m]))
            e_clean = clean["context_energy"].detach().cpu().numpy()[0]
            e_corrupt = corrupt["context_energy"].detach().cpu().numpy()[0]
            satisf.append(float((e_corrupt[m] > e_clean[m]).mean()))
        grad_splits.append({
            "severity": severity,
            "term_means": {k: float(np.mean([t[k] for t in tvals]))
                           for k in tvals[0]},
            "grad_norm_means": {k: float(np.mean(v)) for k, v in gnorms.items()},
            "tracking_median": float(np.median([t["median"] for t in track])),
            "boundary_satisfaction": float(np.mean(satisf))})
    after = sum(p.detach().double().sum().item() for p in model.parameters())
    if after != before:
        raise ValueError("frozen-diag modified weights")
    log["steps"].append({"weights_unchanged": True, "grad_splits": grad_splits})

    # Nuisance invariance: same dev batch clean vs gain/offset transforms.
    nuisance = []
    for label, fn in (("clean", lambda a: a),
                      ("gain1.5", lambda a: 1.5 * a),
                      ("offset2.0", lambda a: a + 2.0)):
        vals = []
        for pw, pm, ri, pi, regs, fid in dev:
            ten = {
                "patches": torch.asarray(fn(pw), dtype=torch.float32).unsqueeze(0),
                "patch_pad_mask": torch.asarray(pm, dtype=torch.bool).unsqueeze(0),
                "patch_valid_mask": torch.ones((1, pw.shape[0]), dtype=torch.bool),
                "robot_idx": torch.tensor([ri], dtype=torch.long),
                "program_idx": torch.tensor([pi], dtype=torch.long),
                "regime_ids": torch.asarray(regs, dtype=torch.long).unsqueeze(0),
            }
            ten = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                   for k, v in ten.items()}
            with torch.no_grad():
                out = model(**ten)
            e = out["context_energy"].detach().cpu().numpy()[0]
            vals.append(float(np.median(np.abs(e))))
        nuisance.append({"transform": label,
                         "median_abs_energy": float(np.median(vals))})
    log["steps"].append({"nuisance": nuisance})

    metrics = {"mode": "frozen-diag", "checkpoint": args.checkpoint,
               "sha256": digest, "absence_audit": absence,
               "grad_splits": grad_splits, "nuisance": nuisance,
               "elapsed_s": round(time.time() - t0, 1)}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=1,
                                                    sort_keys=True))
    with open(outdir / "run.log", "w") as f:
        f.write(json.dumps(log, indent=1, sort_keys=True))
    print(json.dumps({"checkpoint": args.checkpoint,
                      "elapsed_s": metrics["elapsed_s"],
                      "severities": [g["severity"] for g in grad_splits]},
                     indent=1))
    return 0


def _retrain(args) -> int:
    """Execute one v5 C5 tiny retrain (control only, exactly 300 steps).

    Fail-closed gates (in order): variant must be a v5 knob (present);
    checkpoint must be control (v5 §2 budget accounting); output dir must
    not exist (one-shot, no replacement). Inits EXACTLY from the accepted
    checkpoint weights; AdamW lr 1e-3; criterion rebuilt from the checkpoint
    config with ONLY the variant knob zeroed; Fit-healthy compatible files
    in deterministic manifest order; corruption every 4th valid patch at
    severity 2.0, generator seed frozen. Asserts step count == RETRAIN_STEPS
    and loss finiteness every step.
    """
    import numpy as np
    import torch

    from representation.v2_inference import V2InferencePipeline, patch_regime_ids
    from representation.v2_objectives import (
        CounterfactualCriterion, synthesize_corrupted_patches)
    from representation.v2_config import V2Config
    from synth import balanced as B
    from synth.chronicle import load_chronological
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    check_c5_knob(args.variant)
    if args.checkpoint != "control":
        raise ValueError("v5 retrains execute on control only (budget cap)")
    if not args.data_root or not args.out:
        raise ValueError("--data-root and --out are required for retrain")
    outdir = Path(args.out) / f"retrain-{args.variant}"
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing to replace existing output: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    spec = CHECKPOINTS[args.checkpoint]
    ck_path = Path(spec["path"])
    if not ck_path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {ck_path}")
    digest = sha256_file(ck_path)
    if digest != spec["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {digest}")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but unavailable")
    pipe = V2InferencePipeline.load(str(ck_path), device=device)
    model = pipe.model.train()
    cfg = pipe.config
    assert isinstance(cfg, V2Config)
    knob = {"nobackground": "background_weight",
            "nocovariance": "covariance_weight"}[args.variant]
    criterion = CounterfactualCriterion(
        boundary_margin=cfg.boundary_margin,
        background_weight=0.0 if knob == "background_weight" else cfg.background_weight,
        variance_weight=cfg.variance_weight,
        covariance_weight=0.0 if knob == "covariance_weight" else cfg.covariance_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RETRAIN_LR)
    patchifier = Patchifier(PatchConfig())
    data_root = Path(args.data_root)
    gen = torch.Generator().manual_seed(RETRAIN_GEN_SEED)

    def file_batches():
        for role, seed in FIT:
            root = data_root / "FIT" / role
            manifest = json.loads((root / "manifest.json").read_text())
            assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
            samples, _ = load_chronological(root)
            by_id = {s.file_id: s for s in samples}
            for row in manifest["files"]:
                if row["file_label"] != "normal" or row["is_quarantined"]:
                    continue
                if (row["program_id"] == B.PROGRAM_RESERVE
                        or row["robot_id"] == B.ROBOT_RESERVE):
                    continue
                sample = by_id[row["file_id"]]
                if sample.robot_idx > 2 or sample.program_idx > 2:
                    continue
                x = np.asarray(sample.x, dtype=np.float64)
                batch = patchifier.patchify(_wrap(x))
                patches = np.asarray(batch.patches, dtype=np.float64)
                valid = np.asarray(batch.valid_len, dtype=int)
                pad = np.asarray(batch.pad_mask, dtype=bool)
                keep = np.flatnonzero(valid > 0)
                if keep.size == 0:
                    continue
                starts = np.asarray(batch.starts, dtype=int)[keep]
                regs = patch_regime_ids([sample], starts.reshape(1, -1),
                                        int(keep.size))
                cmask = np.zeros(keep.size, dtype=bool)
                cmask[::CORRUPT_EVERY] = True
                yield {
                    "patches": torch.asarray(patches[keep], dtype=torch.float32).unsqueeze(0),
                    "patch_pad_mask": torch.asarray(pad[keep], dtype=torch.bool).unsqueeze(0),
                    "patch_valid_mask": torch.ones((1, int(keep.size)), dtype=torch.bool),
                    "robot_idx": torch.tensor([int(sample.robot_idx)], dtype=torch.long),
                    "program_idx": torch.tensor([int(sample.program_idx)], dtype=torch.long),
                    "regime_ids": torch.asarray(np.asarray(regs[0]), dtype=torch.long).unsqueeze(0),
                    "corruption_mask": torch.asarray(cmask, dtype=torch.bool).unsqueeze(0),
                    "severity": float(RETRAIN_SEVERITY),
                }

    history = []
    steps = 0
    for batch in file_batches():
        if steps >= RETRAIN_STEPS:
            break
        ten = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
               for k, v in batch.items() if k not in ("corruption_mask", "severity")}
        cm = batch["corruption_mask"].to(device)
        clean = model(**ten)
        corrupt_pw = synthesize_corrupted_patches(
            ten["patches"], ten["patch_pad_mask"], ten["patch_valid_mask"],
            cm, float(batch["severity"]), generator=gen)
        corrupt = model(patches=corrupt_pw,
                        patch_pad_mask=ten["patch_pad_mask"],
                        patch_valid_mask=ten["patch_valid_mask"],
                        robot_idx=ten["robot_idx"],
                        program_idx=ten["program_idx"],
                        regime_ids=ten["regime_ids"])
        terms = criterion(
            clean["patch_latents"], corrupt["patch_latents"],
            clean["context_energy"], corrupt["context_energy"],
            ten["patch_valid_mask"], cm, step=steps)
        loss = terms["loss"]
        if not torch.isfinite(loss).all():
            raise ValueError(f"non-finite loss at step {steps}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({k: float(v.detach().cpu().item()) for k, v in terms.items()})
        steps += 1
    if steps != RETRAIN_STEPS:
        raise ValueError(f"ran {steps} steps, frozen count is {RETRAIN_STEPS}")
    torch.save({"variant": args.variant, "knob_zeroed": knob,
                "init_sha256": digest, "steps": steps,
                "config": pipe.config.model_dump(mode="json"),
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "history": history},
               outdir / "retrained.pt")
    print(json.dumps({"retrain": args.variant, "steps": steps,
                       "final_loss": history[-1]["loss"]}))
    return 0


def _eval_retrain(args) -> int:
    """Evaluate a retrained checkpoint's local latents per category."""
    import numpy as np
    import torch

    from representation import attribution_metrics as M
    from synth import balanced as B
    from synth import events as E
    from synth.chronicle import load_chronological
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    ck_path = Path(args.ckpt)
    if not ck_path.is_file():
        raise FileNotFoundError(f"retrained checkpoint missing: {ck_path}")
    digest = sha256_file(ck_path)
    payload = torch.load(str(ck_path), map_location="cpu", weights_only=False)
    if isinstance(payload.get("config"), dict):
        cfg = payload["config"]
        arch_provenance = "embedded"
    else:
        # First-round retrains predate the config-saving fix. Architecture
        # is exactly the accepted control config: retraining alters weights
        # only, never architecture — verified by loading the arch from the
        # hash-verified accepted control checkpoint (no new bytes invented).
        ctrl = CHECKPOINTS["control"]
        ctrl_path = Path(ctrl["path"])
        if sha256_file(ctrl_path) != ctrl["sha256"]:
            raise ValueError("accepted control checkpoint hash mismatch")
        cfg = torch.load(str(ctrl_path), map_location="cpu",
                         weights_only=False)["config"]
        arch_provenance = "accepted-control-checkpoint"
    from representation.v2_patch import ContextConditionedPatchEncoder

    _model = ContextConditionedPatchEncoder(
        n_channels=cfg["n_channels"], d_model=cfg["d_model"],
        n_robots=cfg["n_robots"], n_programs=cfg["n_programs"],
        n_regimes=cfg["n_regimes"], n_prototypes=cfg["n_prototypes"],
        sequence_layers=cfg["sequence_layers"],
        attention_heads=cfg["attention_heads"], dropout=0.0).eval()
    _model.load_state_dict(payload["model_state"])
    _model.to(args.device)
    local_enc = _model.local
    patchifier = Patchifier(PatchConfig())
    data_root = Path(args.data_root)

    def load_root(role: str, group: str):
        root = data_root / group / role
        manifest = json.loads((root / "manifest.json").read_text())
        samples, _ = load_chronological(root)
        return samples, manifest

    fit_lat = []
    for role, seed in FIT:
        samples, manifest = load_root(role, "FIT")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        by_id = {s.file_id: s for s in samples}
        for row in manifest["files"]:
            if row["file_label"] != "normal" or row["is_quarantined"]:
                continue
            if (row["program_id"] == B.PROGRAM_RESERVE
                    or row["robot_id"] == B.ROBOT_RESERVE):
                continue
            x = np.asarray(by_id[row["file_id"]].x, dtype=np.float64)
            batch = patchifier.patchify(_wrap(x))
            patches = np.asarray(batch.patches, dtype=np.float64)
            valid = np.asarray(batch.valid_len, dtype=int)
            pad = np.asarray(batch.pad_mask, dtype=bool)
            keep = valid > 0
            if not keep.any():
                continue
            with torch.no_grad():
                lat = local_enc(
                    torch.asarray(patches[keep], dtype=torch.float32).unsqueeze(0).to(args.device),
                    torch.asarray(pad[keep], dtype=torch.bool).unsqueeze(0).to(args.device))
                fit_lat.append(lat.detach().cpu().numpy()[0])
    fit_lat = np.vstack(fit_lat).astype(np.float64)
    st = M.FrozenStandardizer.fit(fit_lat, source="FIT-healthy-patches")
    cen = st.apply(fit_lat).mean(axis=0)
    per_history = []
    for role, seed in CONF:
        samples, manifest = load_root(role, "CONFIRMATION")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        by_id = {s.file_id: s for s in samples}
        rows = manifest["files"]
        wins = manifest["maintenance_windows"]
        ledger = E.failure_ledger(manifest)
        fscores = {}
        for row in rows:
            x = np.asarray(by_id[row["file_id"]].x, dtype=np.float64)
            batch = patchifier.patchify(_wrap(x))
            patches = np.asarray(batch.patches, dtype=np.float64)
            valid = np.asarray(batch.valid_len, dtype=int)
            pad = np.asarray(batch.pad_mask, dtype=bool)
            keep = np.flatnonzero(valid > 0)
            if keep.size == 0:
                raise ValueError(f"zero valid patches: {row['file_id']}")
            with torch.no_grad():
                lat = local_enc(
                    torch.asarray(patches[keep], dtype=torch.float32).unsqueeze(0).to(args.device),
                    torch.asarray(pad[keep], dtype=torch.bool).unsqueeze(0).to(args.device))
                z = st.apply(lat.detach().cpu().numpy()[0])
            fscores[row["file_id"]] = float(np.linalg.norm(z - cen, axis=1).max())
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)
        neg = [E.window_score([m["file_id"] for m in w["members"]], fscores)
               for w in controls]
        cats = {}
        for cat in CATEGORIES:
            cohort = cat[0] if len(cat) == 2 else cat
            subtype = cat if len(cat) == 2 else None

            def ev(failure: dict) -> float | None:
                if E.positive_window_intersects_reset(failure, wins):
                    return None
                cands = E.pos_files(rows, failure, wins)
                if not cands:
                    return None
                return E.window_score([c["file_id"] for c in cands], fscores)

            pos = [s for f in ledger
                   if f["cohort"] == cohort
                   and (subtype is None or f["subtype"] == subtype)
                   and (s := ev(f)) is not None]
            if not pos or not neg:
                cats[cat] = {"reason": "no support"}
            else:
                cats[cat] = {"auc": round(float(M.tie_auc(
                    np.array(pos + neg),
                    np.array([1.0] * len(pos) + [0.0] * len(neg)))), 4),
                    "n_pos": len(pos)}
        per_history.append({"role": role, "seed": seed, "categories": cats})
    evaldir = Path(args.out) / f"eval-{Path(args.ckpt).parent.name}"
    evaldir.mkdir(parents=True, exist_ok=True)
    (evaldir / "metrics.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({"eval_retrain": "done", "sha256": digest[:12]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
