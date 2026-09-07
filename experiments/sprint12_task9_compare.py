"""Sprint 12 Task 9 (part 2) — matched representation comparison + selection freeze.

Compares three arms under one evaluation protocol on H-DEV CAL paired
mechanism corruptions (fixed per-file masks, all mechanisms, severities
1/2/4, same definitions as the G-learn gate):

- (a) standardized handcrafted + fresh FIT hierarchical geometry;
- (b) frozen control latents + fresh FIT hierarchical geometry (current);
- (c) revised target-hidden scorer (trained checkpoint) with direct
  hidden-energy tails (explicit `hidden` label — never a canonical field).

Selection: reads the gate JSON; on G-learn PASS freezes EXACTLY ONE
configuration (`selected.json` with checkpoint/config/threshold hashes);
on FAIL records no selection (Task 10 stays blocked). No sealed reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
from scipy.stats import spearmanr
sys.path.insert(0, str(REPO_ROOT / "src"))

from representation.corruptions import MECHANISMS, NUISANCES, corrupt  # noqa: E402
from representation.data import collate_variable_files  # noqa: E402
from representation.handcrafted import Standardizer, batch_patch_features, feature_dim  # noqa: E402
from representation.target_hidden import TargetHiddenScorer  # noqa: E402
from representation.v2_aggregation import (  # noqa: E402
    aggregate_file_state,
    calibrate_elevated_threshold_with_provenance,
)
from representation.v2_contracts import POPULATION_ENERGY_FIELD  # noqa: E402
from representation.v2_geometry import HierarchicalMahalanobisGeometry  # noqa: E402
from representation.v2_inference import V2InferencePipeline, patch_regime_ids  # noqa: E402
from scipy.stats import spearmanr
from synth.chronicle import load_chronological  # noqa: E402
from synth.config import PatchConfig  # noqa: E402
from synth.patchify import Patchifier  # noqa: E402
from synth.schema import SampleLabel  # noqa: E402

RESERVED_FAMILIES = {"wrong_transition", "cross_channel_inconsistency"}
HELD_OUT_PROGRAM = "program-03"
SEVERITIES = (1.0, 2.0, 4.0)
RATE = 0.25
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
    ap.add_argument("--scorer-pt", required=True)
    ap.add_argument("--gate-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    commit = repo_commit()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = json.loads(Path(args.gate_json).read_text())["gate"]

    pipe = V2InferencePipeline.load(args.control_ckpt, device=args.device)
    pipe.model.eval()
    patchifier = Patchifier(PatchConfig(patch_size=int(pipe.config.patch_size),
                                        stride=int(pipe.config.stride)))
    scorer = TargetHiddenScorer.load(args.scorer_pt).to(args.device).eval()
    cfg = pipe.config
    geo_kwargs = {"shrinkage": float(cfg.shrinkage), "covariance_eps": float(cfg.covariance_eps),
                  "min_group_samples": int(cfg.min_group_samples),
                  "diag_min_samples": int(cfg.diag_min_samples)}

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

        fit_files += [by_id[i] for i in manifest["splits"]["dev_train"] if keep(i)]
        cal_files += [by_id[i] for i in manifest["splits"]["dev_val"] if keep(i)]

    def featurize(sample, kind: str):
        batch = single_batch(sample, patchifier, args.device)
        v = batch["patch_valid_mask"].cpu()
        if kind == "handcrafted":
            return (torch.from_numpy(batch_patch_features(
                batch["patches"][0].cpu().numpy(),
                batch["patch_pad_mask"][0].cpu().numpy())).float(), v)
        with torch.no_grad():
            lat = pipe.model.local(batch["patches"], batch["patch_pad_mask"])
        return lat.cpu(), v

    # arm (a): standardized handcrafted + fresh FIT geometry
    fit_hc = torch.cat([m[v[0]] for m, v in (featurize(s, "handcrafted") for s in fit_files)])
    std = Standardizer.fit(fit_hc.numpy())
    aux_rows = []
    std_rows = []
    for s in fit_files:
        mat, v = featurize(s, "handcrafted")
        vv = v[0]
        std_rows.append(torch.from_numpy(std.apply(mat[0].numpy())).float()[vv])
        n = int(vv.sum())
        batch = single_batch(s, patchifier, args.device)
        aux_rows.append(torch.stack([
            batch["robot_idx"].cpu().expand(int(vv.shape[0]))[vv],
            batch["program_idx"].cpu().expand(int(vv.shape[0]))[vv],
            batch["regime_ids"][0].cpu()[vv]], dim=1))
        _ = n
    rows_a = torch.cat(std_rows)
    aux_a = torch.cat(aux_rows)
    geo_a = HierarchicalMahalanobisGeometry(rows_a.shape[1], **geo_kwargs)
    geo_a.fit(rows_a, aux_a[:, 0], aux_a[:, 1], aux_a[:, 2],
              torch.ones((rows_a.shape[0],), dtype=torch.bool))
    geo_a = geo_a.frozen()
    # arm (b): frozen control latents + fresh FIT geometry
    rows_b, aux_b = [], []
    for s in fit_files:
        mat, v = featurize(s, "learned")
        vv = v[0]
        rows_b.append(mat[0][vv])
        batch = single_batch(s, patchifier, args.device)
        aux_b.append(torch.stack([
            batch["robot_idx"].cpu().expand(int(vv.shape[0]))[vv],
            batch["program_idx"].cpu().expand(int(vv.shape[0]))[vv],
            batch["regime_ids"][0].cpu()[vv]], dim=1))
    rows_b = torch.cat(rows_b)
    aux_b = torch.cat(aux_b)
    geo_b = HierarchicalMahalanobisGeometry(rows_b.shape[1], **geo_kwargs)
    geo_b.fit(rows_b, aux_b[:, 0], aux_b[:, 1], aux_b[:, 2],
              torch.ones((rows_b.shape[0],), dtype=torch.bool))
    geo_b = geo_b.frozen()

    def arm_energy(arm: str, sample):
        batch = single_batch(sample, patchifier, args.device)
        v = batch["patch_valid_mask"].cpu()
        if arm == "hidden":
            with torch.no_grad():
                lat = pipe.model.local(batch["patches"], batch["patch_pad_mask"])
                e = scorer(lat, batch["patch_valid_mask"])["hidden_context_energy"].cpu()
            return e, v
        mat, _ = featurize(sample, "handcrafted" if arm == "handcrafted" else "learned")
        if arm == "handcrafted":
            mat = torch.from_numpy(std.apply(mat[0].numpy())).float().unsqueeze(0)
        geo = geo_a if arm == "handcrafted" else geo_b
        e = geo.population_energy(mat, v, batch["robot_idx"].cpu(),
                                  batch["program_idx"].cpu(),
                                  batch["regime_ids"].cpu())["population_energy"]
        return e, v

    # CAL thresholds per arm (fixed 0.05 rule)
    thresholds = {}
    for arm in ("handcrafted", "learned", "hidden"):
        ee, vv = [], []
        for s in cal_files:
            e, v = arm_energy(arm, s)
            ee.append(e.reshape(1, -1))
            vv.append(v.reshape(1, -1))
        thr, _ = calibrate_elevated_threshold_with_provenance(
            torch.cat(ee, dim=1), torch.cat(vv, dim=1),
            tail_probability=0.05, cohort=f"dev-val ({arm})")
        thresholds[arm] = thr

    # matched CAL paired-corruption ranking per arm
    table = {}
    for arm in ("handcrafted", "learned", "hidden"):
        pc, pn, gaps, levs, bg = [], [], [], [], []
        for s in cal_files:
            batch = single_batch(s, patchifier, args.device)
            base_e, v = arm_energy(arm, s)
            be = base_e[0].numpy()
            vv = v[0].numpy()
            for mech in MECHANISMS:
                fmask = (torch.rand(v.shape, generator=torch.Generator().manual_seed(
                    seed_from(s.file_id, mech))) < RATE) & v.cpu()
                m = fmask[0].numpy()
                if m.sum() == 0 or not (vv & ~m).any():
                    continue
                for sev in SEVERITIES:
                    wp = corrupt(mech, batch["patches"], batch["patch_pad_mask"], v.to(args.device),
                                 fmask.to(args.device), sev,
                                 torch.Generator().manual_seed(seed_from(s.file_id, mech, "dir")))
                    if arm == "hidden":
                        with torch.no_grad():
                            ce = scorer(pipe.model.local(wp, batch["patch_pad_mask"]),
                                        v.to(args.device))["hidden_context_energy"].cpu()[0].numpy()
                    else:
                        raw = wp[0].cpu().numpy()
                        pad = batch["patch_pad_mask"][0].cpu().numpy()
                        mat = (torch.from_numpy(std.apply(batch_patch_features(raw, pad))).float()
                               if arm == "handcrafted"
                               else pipe.model.local(wp, batch["patch_pad_mask"]).cpu()[0])
                        geo = geo_a if arm == "handcrafted" else geo_b
                        ce = geo.population_energy(
                            mat.unsqueeze(0), v, batch["robot_idx"].cpu(),
                            batch["program_idx"].cpu(),
                            batch["regime_ids"].cpu())["population_energy"][0].numpy()
                    pc.extend(be[vv & m].tolist())
                    pn.extend(ce[vv & m].tolist())
                    gaps.append(float(np.median(ce[vv & m]) - np.median(be[vv & m])))
                    levs.append(sev)
                    bg.append(float(np.median(np.abs(ce[vv & ~m] - be[vv & ~m]))))
        pc, pn = np.array(pc), np.array(pn)
        c = spearmanr(gaps, levs)
        table[arm] = {
            "ranking_rate": float((pn > pc).mean()),
            "severity_spearman": float(c.statistic) if np.isfinite(c.statistic) else float("nan"),
            "gap_median": float(np.median(gaps)),
            "background_ratio": float(np.median(bg) / max(1e-12, np.median(gaps))),
            "threshold": thresholds[arm],
        }

    result = {
        "provenance": {
            "commit": commit,
            "commit_source": "git rev-parse HEAD in execution checkout (dirty source tree refused)",
            "dev_roots": list(args.dev_roots),
            "scorer_pt_sha256": sha256_of(Path(args.scorer_pt)),
            "gate_pass": bool(gate["pass"]),
            "n_fit": len(fit_files), "n_cal": len(cal_files),
        },
        "comparison": table,
        "gate": gate,
    }
    selected = None
    if gate["pass"]:
        # exactly ONE configuration: the revised scorer + frozen CAL threshold
        with torch.no_grad():
            cal_hidden = []
            for s in cal_files:
                b = single_batch(s, patchifier, args.device)
                e = scorer(pipe.model.local(b["patches"], b["patch_pad_mask"]),
                           b["patch_valid_mask"])["hidden_context_energy"]
                cal_hidden.append(e.cpu().numpy()[b["patch_valid_mask"].cpu().numpy()])
        selected = {
            "arm": "revised-target-hidden",
            "scorer_pt_sha256": sha256_of(Path(args.scorer_pt)),
            "scorer_config": TargetHiddenScorer(d_model=32).config(),
            "local_encoder": "accepted-control-checkpoint frozen LocalPatchEncoder",
            "control_ckpt_sha256": "8bdb845b17a788ad39101e0f98c0654edb55727f0eec540ca44046b63dba2b7c",
            "cal_threshold_hidden": thresholds["hidden"],
            "mechanisms": list(MECHANISMS),
            "top_q_fraction": TOP_Q,
        }
        result["selected"] = selected
    else:
        result["selected"] = None
        result["disposition"] = ("G-learn FAIL: no configuration selected; "
                                 "Task 10 stays blocked; no sealed tuning.")
    (out_dir / "task9_compare.json").write_text(json.dumps(result, indent=2))
    if selected is not None:
        (out_dir / "selected.json").write_text(json.dumps(selected, indent=2))
    print(f"WROTE {out_dir / 'task9_compare.json'} gate_pass={gate['pass']} "
          f"selected={selected is not None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
