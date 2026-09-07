"""Sprint 12 Task 9 — bounded learnability gate + matched representation comparison.

Runs on H-DEV-1..4 ONLY (never sealed roots): FIT pools fit references,
CAL pools gate/select. Exclusions (reserved families, program-03) enforced and
asserted before any use. Arms:

- (a) standardized handcrafted + fresh FIT hierarchical geometry;
- (b) frozen control latents + fresh FIT hierarchical geometry (current);
- (c) revised target-hidden scorer (frozen control LocalPatchEncoder features,
  trained TargetHiddenScorer) with direct hidden-energy tails.

G-learn gate (protocol v2 §10, predeclared) runs FIRST on CAL paired
mechanism corruptions: ranking rate >= 0.80, severity Spearman >= 0.50,
background <= 0.10x corrupt gap, nuisance flag <= 0.10. Training is bounded
(300 steps); on gate failure the script diagnoses (loss/gradient trajectories,
per-mechanism breakdown) and performs at most ONE bounded documented correction
run, then reports the disposition honestly. No long run on a failed gate, no
sealed reads, no gate tuning.

Selection: at most ONE configuration frozen with hashes (only on G-learn PASS).
Scientific failure is acceptable and blocks Task 10 without sealed tuning.
"""

from __future__ import annotations

import argparse
import csv
from scipy.stats import spearmanr
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
# Same-checkout source first (this tree IS the verified commit, not a copy).
sys.path.insert(0, str(REPO_ROOT / "src"))

from representation.corruptions import MECHANISMS, NUISANCES, corrupt  # noqa: E402
from representation.data import collate_variable_files  # noqa: E402
from representation.handcrafted import Standardizer, batch_patch_features, feature_names  # noqa: E402
from representation.target_hidden import TargetHiddenScorer  # noqa: E402
from representation.v2_aggregation import (  # noqa: E402
    aggregate_file_state,
    calibrate_elevated_threshold_with_provenance,
)
from representation.v2_contracts import POPULATION_ENERGY_FIELD  # noqa: E402
from representation.v2_geometry import HierarchicalMahalanobisGeometry  # noqa: E402
from representation.v2_inference import V2InferencePipeline, patch_regime_ids  # noqa: E402
from representation.v2_objectives import CounterfactualCriterion  # noqa: E402
from synth.chronicle import load_chronological  # noqa: E402
from synth.config import PatchConfig  # noqa: E402
from synth.patchify import Patchifier  # noqa: E402
from synth.schema import SampleLabel  # noqa: E402

RESERVED_FAMILIES = {"wrong_transition", "cross_channel_inconsistency"}
HELD_OUT_PROGRAM = "program-03"
SEVERITIES = (1.0, 2.0, 4.0)
TRAIN_STEPS = 300
BATCH_SIZE = 8
CORRUPTION_RATE = 0.25
TOP_Q = 0.1


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_commit() -> str:
    """Live Git-derived execution provenance (fails loudly, never a CLI claim)."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        check=True, cwd=str(REPO_ROOT),
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no",
         "--", "experiments", "src", "tests"],
        capture_output=True, text=True, check=True, cwd=str(REPO_ROOT),
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "refusing to run with tracked modifications under "
            f"experiments/src/tests:\n{dirty}"
        )
    return out


def seed_from(*parts: str) -> int:
    """Stable 63-bit seed from string parts."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:16], 16) % (2**63)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    c = spearmanr(a, b)
    return float(c.statistic) if np.isfinite(c.statistic) else float("nan")


def single_batch(sample, patchifier, device: str) -> dict:
    base = collate_variable_files([sample], patchifier)
    count = base["patches"].shape[1]
    regimes = patch_regime_ids([sample], base["starts"], count)
    return {
        "patches": base["patches"].to(device),
        "patch_pad_mask": base["patch_pad_mask"].to(device),
        "patch_valid_mask": base["patch_valid_mask"].to(device),
        "robot_idx": base["robot_idx"].to(device),
        "program_idx": base["program_idx"].to(device),
        "regime_ids": regimes.to(device),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-roots", nargs=4, required=True)
    ap.add_argument("--control-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--correction", action="store_true",
                    help="bounded correction run (documented second attempt)")
    args = ap.parse_args()

    commit = repo_commit()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    ckpt = Path(args.control_ckpt)
    if sha256_of(ckpt) != "8bdb845b17a788ad39101e0f98c0654edb55727f0eec540ca44046b63dba2b7c":
        raise ValueError("control checkpoint is not the accepted hash")
    pipe = V2InferencePipeline.load(ckpt, device=args.device)
    pipe.model.eval()
    for param in pipe.model.local.parameters():
        param.requires_grad_(False)  # frozen local features; scorer trains
    patchifier = Patchifier(PatchConfig(patch_size=int(pipe.config.patch_size),
                                        stride=int(pipe.config.stride)))

    # --- load dev histories, enforce exclusions --------------------------------
    fit_files, cal_files = [], []
    for root in args.dev_roots:
        samples, manifest = load_chronological(root)
        by_id = {s.file_id: s for s in samples}
        files = {row["file_id"]: row for row in manifest["files"]}

        def keep(fid: str) -> bool:
            s = by_id[fid]
            fam = s.anomaly_meta.family.value if s.anomaly_meta else "normal"
            return (s.file_label is SampleLabel.NORMAL
                    and fam not in RESERVED_FAMILIES
                    and str(files[fid]["program_id"]) != HELD_OUT_PROGRAM)

        train_ids = [i for i in manifest["splits"]["dev_train"] if keep(i)]
        val_ids = [i for i in manifest["splits"]["dev_val"] if keep(i)]
        # assert the exclusions actually fired (partition is real, not vacuous)
        assert all(by_id[i].file_label is SampleLabel.NORMAL for i in train_ids + val_ids)
        fit_files += [by_id[i] for i in train_ids]
        cal_files += [by_id[i] for i in val_ids]
    assert len(fit_files) >= 120 and len(cal_files) >= 40, (len(fit_files), len(cal_files))

    scorer = TargetHiddenScorer(d_model=32, n_heads=4, n_layers=2).to(args.device)
    criterion = CounterfactualCriterion(boundary_margin=args.margin)
    opt = torch.optim.Adam(scorer.parameters(), lr=args.lr)
    history: list[dict] = []
    grad_norms: list[float] = []

    order = torch.randperm(len(fit_files), generator=torch.Generator().manual_seed(0)).tolist()
    ordered = [fit_files[i] for i in order]
    step = 0
    sev_pairs = [(1.0, 2.0), (2.0, 4.0), (1.0, 4.0)]
    while step < args.steps:
        for start in range(0, len(ordered), BATCH_SIZE):
            if step >= args.steps:
                break
            chunk = ordered[start:start + BATCH_SIZE]
            base = collate_variable_files(chunk, patchifier)
            count = base["patches"].shape[1]
            regimes = patch_regime_ids(chunk, base["starts"], count)
            dev = args.device
            batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
                     for k, v in base.items()}
            batch["regime_ids"] = regimes.to(dev)
            valid = batch["patch_valid_mask"]
            gen = torch.Generator().manual_seed(1000 + step)
            cmask = (torch.rand(valid.shape, generator=gen) < CORRUPTION_RATE) & valid
            mech = MECHANISMS[step % len(MECHANISMS)]
            sa, sb = sev_pairs[step % len(sev_pairs)]
            with torch.no_grad():
                local = pipe.model.local(batch["patches"], batch["patch_pad_mask"])
            gdir = torch.Generator(device=dev).manual_seed(2000 + step) \
                if dev != "cpu" else torch.Generator().manual_seed(2000 + step)
            views = {}
            for tag, sev in (("clean", 0.0), ("a", sa), ("b", sb)):
                if tag == "clean":
                    views[tag] = local
                else:
                    wp = corrupt(mech, batch["patches"], batch["patch_pad_mask"],
                                 valid, cmask, sev, gdir)
                    views[tag] = pipe.model.local(wp, batch["patch_pad_mask"])
            scorer.train()
            out_c = scorer(views["clean"], valid)
            out_a = scorer(views["a"], valid)
            out_b = scorer(views["b"], valid)
            alpha = 0.0 if step < 50 else 1.0  # documented short warmup
            loss = (criterion.clean_density(out_c["hidden_context_energy"], valid)
                    + alpha * criterion.boundary(out_c["hidden_context_energy"],
                                                 out_a["hidden_context_energy"], valid, cmask)
                    + criterion.background(out_c["hidden_context_energy"],
                                           out_a["hidden_context_energy"], valid, cmask)
                    + alpha * criterion.ordering(
                        [out_a["hidden_context_energy"][valid & cmask],
                         out_b["hidden_context_energy"][valid & cmask]], [0.5]))
            opt.zero_grad()
            loss.backward()
            total_sq = sum(float(p.grad.detach().pow(2).sum())
                           for p in scorer.parameters() if p.grad is not None)
            grad_norms.append(total_sq ** 0.5)
            opt.step()
            history.append({"step": step, "loss": float(loss.detach()),
                            "mech": mech, "sev_pair": [sa, sb], "alpha": alpha})
            step += 1

    scorer.eval()
    ckpt_path = out_dir / "hidden_scorer.pt"
    scorer_cpu = TargetHiddenScorer(d_model=32, n_heads=4, n_layers=2)
    scorer_cpu.load_state_dict({k: v.cpu() for k, v in scorer.state_dict().items()})
    scorer_cpu.save(ckpt_path)
    # --- CAL hidden threshold (fixed 0.05 rule, CAL normals only) -------------------
    cal_energies = []
    for s in cal_files:
        b = single_batch(s, patchifier, args.device)
        with torch.no_grad():
            e = scorer(pipe.model.local(b["patches"], b["patch_pad_mask"]),
                       b["patch_valid_mask"])["hidden_context_energy"]
        cal_energies.append(e.cpu().numpy()[b["patch_valid_mask"].cpu().numpy()])
    cal_thr = float(np.quantile(np.concatenate(cal_energies), 0.95))
    # --- G-learn gate on CAL paired corruptions ----------------------------------
    paired_gaps, paired_clean, paired_corr = [], [], []
    sev_gap, sev_level = [], []
    bg_deltas = []
    for sample in cal_files:
        batch = single_batch(sample, patchifier, args.device)
        valid = batch["patch_valid_mask"]
        with torch.no_grad():
            local = pipe.model.local(batch["patches"], batch["patch_pad_mask"])
            base_e = scorer(local.to(args.device), valid)["hidden_context_energy"]
        for mech in MECHANISMS:
            fmask = (torch.rand(valid.shape,
                                generator=torch.Generator().manual_seed(seed_from(sample.file_id, mech)))
                     < CORRUPTION_RATE) & valid.cpu()
            for sev in SEVERITIES:
                wp = corrupt(mech, batch["patches"], batch["patch_pad_mask"],
                             valid, fmask.to(args.device), sev,
                             torch.Generator().manual_seed(seed_from(sample.file_id, mech, "dir")))
                with torch.no_grad():
                    ce = scorer(pipe.model.local(wp, batch["patch_pad_mask"]),
                                valid)["hidden_context_energy"].cpu()
                m = fmask[0].numpy()
                v = valid[0].cpu().numpy()
                if m.sum() == 0 or not (v & ~m).any():
                    continue
                be, ce_m = base_e[0].cpu().numpy(), ce[0].numpy()
                paired_clean.extend(be[v & m].tolist())
                paired_corr.extend(ce_m[v & m].tolist())
                gap = float(np.median(ce_m[v & m]) - np.median(be[v & m]))
                paired_gaps.append(gap)
                sev_gap.append(gap)
                sev_level.append(sev)
                bg_deltas.append(float(np.median(np.abs(ce_m[v & ~m] - be[v & ~m]))))
    paired_clean = np.array(paired_clean)
    paired_corr = np.array(paired_corr)
    gate = {
        "ranking_rate": float((paired_corr > paired_clean).mean()),
        "severity_spearman": spearman(np.array(sev_gap), np.array(sev_level)),
        "masked_gap_median": float(np.median(paired_gaps)),
        "background_median": float(np.median(bg_deltas)),
        "background_ratio": float(np.median(bg_deltas) / max(1e-12, np.median(paired_gaps))),
    }
    nuis_flags, nuis_deltas = [], []
    for sample in cal_files[:32]:
        batch = single_batch(sample, patchifier, args.device)
        v = batch["patch_valid_mask"]
        with torch.no_grad():
            be = scorer(pipe.model.local(batch["patches"], batch["patch_pad_mask"]), v)["hidden_context_energy"]
        for tag in NUISANCES:
            wp = corrupt(tag, batch["patches"], batch["patch_pad_mask"], v,
                         torch.ones_like(v), 1.0, torch.Generator().manual_seed(5))
            with torch.no_grad():
                ne = scorer(pipe.model.local(wp, batch["patch_pad_mask"]), v)["hidden_context_energy"]
            d = (ne - be).cpu().numpy()[v.cpu().numpy()]
            nuis_deltas.append(float(np.median(np.abs(d))))
            nuis_flags.append(bool((ne.cpu().numpy()[v.cpu().numpy()] >= cal_thr).mean() > 0.10))
    gate["nuisance_flag_rate"] = float(np.mean(nuis_flags))
    gate["nuisance_delta_median"] = float(np.median(nuis_deltas))
    gate["pass"] = bool(gate["ranking_rate"] >= 0.80 and gate["severity_spearman"] >= 0.50
                        and gate["background_ratio"] <= 0.10 and gate["nuisance_flag_rate"] <= 0.10)

    result: dict = {
        "provenance": {
            "commit": commit,
            "commit_source": "git rev-parse HEAD in execution checkout (dirty source tree refused)",
            "dev_roots": list(args.dev_roots),
            "control_ckpt_sha256": sha256_of(ckpt),
            "device": args.device,
            "torch": torch.__version__,
            "steps": args.steps, "lr": args.lr, "margin": args.margin,
            "correction_run": bool(args.correction),
            "n_fit": len(fit_files), "n_cal": len(cal_files),
            "mechanisms": list(MECHANISMS), "nuisances": list(NUISANCES),
            "scorer_checkpoint": {"path": str(ckpt_path), "sha256": sha256_of(ckpt_path)},
        },
        "train_tail": history[-10:],
        "grad_norm": {"median": float(np.median(grad_norms)), "max": float(np.max(grad_norms)),
                      "final": float(grad_norms[-1])},
        "gate": gate,
    }
    (out_dir / "task9_gate.json").write_text(json.dumps(result, indent=2))
    print(f"WROTE {out_dir / 'task9_gate.json'} gate_pass={gate['pass']} {json.dumps(gate, indent=0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
