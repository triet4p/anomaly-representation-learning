"""Sprint 12 Task 11 — observable precursor diagnostics by failure category (corrected).

DEV histories only (never sealed). For each failure episode, post-hoc
simulator state (episodes manifest) assigns ONE category:

- progressive: a degradation episode on the same robot overlapping
  [fail_start − 14d, fail_start] with duration ≥ 3d;
- abrupt: no degradation episode ending within 14d before fail_start
  (UNAVAILABLE under v2 abrupt_rate=0 — recorded, not sampled);
- weak: anything in between (likewise UNAVAILABLE here).

Two causal file-score probes (file end_time ≤ window end, strictly causal):
- amplitude: zero-fit max centered patch amplitude (W=32/S=16);
- geometry_tail: accepted standardized handcrafted features + frozen
  hierarchical conditional geometry fitted on H-DEV healthy FIT only,
  file score = mean of top-10% valid patch population energies.

Clean causal baseline per failure: same-robot files ending in
[fail−28d, fail−7d) that are NOT quarantined, do NOT overlap maintenance,
do NOT overlap any degradation episode, and do NOT intersect any other
failure's [other_start − 28d, other_start] window. Exclusions are counted
by reason (coverage/missing baselines quantified — a failure with zero kept
baseline files is excluded from gap stats and counted explicitly).

Labels/health/episodes enter ONLY post-hoc stratification — never any input.
Weak/abrupt categories are UNAVAILABLE under v2 abrupt_rate=0: conclusions
are scoped to progressive failures; no new data; no generalization.
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
# Same-checkout source first (this tree IS the verified commit, not a copy).
sys.path.insert(0, str(REPO_ROOT / "src"))

from representation.data import collate_variable_files  # noqa: E402
from representation.handcrafted import Standardizer, batch_patch_features  # noqa: E402
from representation.v2_geometry import HierarchicalMahalanobisGeometry  # noqa: E402
from representation.v2_inference import V2InferencePipeline, patch_regime_ids  # noqa: E402
from synth.chronicle import load_chronological  # noqa: E402
from synth.config import PatchConfig  # noqa: E402
from synth.patchify import Patchifier  # noqa: E402
from synth.schema import SampleLabel  # noqa: E402

DAY = 86400.0
LOOKBACK_S = 14 * DAY
PROG_MIN_DUR_S = 3 * DAY
WINDOWS = {"w1d": 1 * DAY, "w7d": 7 * DAY}
BASE_SPAN_D = 28
BASE_GAP_D = 7
TOP_Q = 0.1
RESERVED_FAMILIES = {"wrong_transition", "cross_channel_inconsistency"}
HELD_OUT_PROGRAM = "program-03"
CONTROL_CKPT_SHA256 = "8bdb845b17a788ad39101e0f98c0654edb55727f0eec540ca44046b63dba2b7c"


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


def describe(values: np.ndarray) -> dict:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
    }


def file_score(x: np.ndarray) -> float:
    """Zero-fit causal observable: max centered patch amplitude (W=32/S=16)."""
    med = np.median(x, axis=1, keepdims=True)
    dev = np.abs(x - med)
    _, t = dev.shape
    w, s = 32, 16
    best = 0.0
    for start in range(0, max(1, t - w + 1), s):
        best = max(best, float(dev[:, start:start + w].mean()))
    tail = dev[:, (max(0, t - w)):].mean() if t >= w else best
    return float(max(best, float(tail)))


def intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    """Half-open interval overlap [a0,a1) vs [b0,b1)."""
    return a0 < b1 and b0 < a1


def in_window_causal(file_end: float, fail_start: float, wlen: float) -> bool:
    """Causal probe-window membership: file must END at or before failure start."""
    return fail_start - wlen <= file_end <= fail_start


def baseline_eligible(
    f_start: float,
    f_end: float,
    *,
    fail_start: float,
    other_starts: list[float],
    degradations: list[tuple[float, float]],
    maint_list: list[tuple[float, float]],
    quarantined: bool,
) -> tuple[bool, str]:
    """Clean-baseline eligibility for one candidate file interval.

    Membership window [fail−28d, fail−7d) on end_time is checked by the caller;
    this judges contamination only. Returns (eligible, reason) with reason one
    of: ok, quarantined, maintenance, degradation, other_failure_window.
    """
    if quarantined:
        return False, "quarantined"
    for s, e in maint_list:
        if intervals_overlap(f_start, f_end, s, e):
            return False, "maintenance"
    for s, e in degradations:
        if intervals_overlap(f_start, f_end, s, e):
            return False, "degradation"
    for other in other_starts:
        if intervals_overlap(f_start, f_end, other - BASE_SPAN_D * DAY, other):
            return False, "other_failure_window"
    return True, "ok"


def tail_energy(energies: np.ndarray, valid: np.ndarray, top_q: float = TOP_Q) -> float:
    """Mean of top-q valid patch energies; nan when no valid patch."""
    e = np.asarray(energies, dtype=np.float64).ravel()
    v = np.asarray(valid, dtype=bool).ravel()
    if e.shape != v.shape or not bool(v.any()):
        return float("nan")
    vals = np.sort(e[v])
    k = max(1, int(round(top_q * vals.size)))
    return float(vals[-k:].mean())


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
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    commit = repo_commit()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if sha256_of(Path(args.control_ckpt)) != CONTROL_CKPT_SHA256:
        raise ValueError("control checkpoint is not the accepted Sprint 11 hash")

    pipe = V2InferencePipeline.load(args.control_ckpt, device=args.device)
    pipe.model.eval()
    patchifier = Patchifier(PatchConfig(patch_size=int(pipe.config.patch_size),
                                        stride=int(pipe.config.stride)))
    cfg = pipe.config
    geo_kwargs = {"shrinkage": float(cfg.shrinkage), "covariance_eps": float(cfg.covariance_eps),
                  "min_group_samples": int(cfg.min_group_samples),
                  "diag_min_samples": int(cfg.diag_min_samples)}

    # Phase 1: FIT reference on H-DEV healthy FIT only (exclusions enforced).
    fit_files = []
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

    def featurize(sample):
        batch = single_batch(sample, patchifier, args.device)
        v = batch["patch_valid_mask"].cpu()
        mat = torch.from_numpy(batch_patch_features(
            batch["patches"][0].cpu().numpy(),
            batch["patch_pad_mask"][0].cpu().numpy())).float()
        return mat, v, batch

    fit_mats = [m[v[0]] for m, v, _ in (featurize(s) for s in fit_files)]
    std = Standardizer.fit(torch.cat(fit_mats).numpy())
    std_rows, aux_rows = [], []
    for s in fit_files:
        mat, v, batch = featurize(s)
        vv = v[0]
        std_rows.append(torch.from_numpy(std.apply(mat.numpy())).float()[vv])
        aux_rows.append(torch.stack([
            batch["robot_idx"].cpu().expand(int(vv.shape[0]))[vv],
            batch["program_idx"].cpu().expand(int(vv.shape[0]))[vv],
            batch["regime_ids"][0].cpu()[vv]], dim=1))
    rows_a = torch.cat(std_rows)
    aux = torch.cat(aux_rows)
    geo = HierarchicalMahalanobisGeometry(rows_a.shape[1], **geo_kwargs)
    geo.fit(rows_a, aux[:, 0], aux[:, 1], aux[:, 2],
            torch.ones((rows_a.shape[0],), dtype=torch.bool))
    geo = geo.frozen()

    def geometry_tail(sample) -> float:
        batch = single_batch(sample, patchifier, args.device)
        v = batch["patch_valid_mask"].cpu()
        mat, _, _ = featurize(sample)
        mat = torch.from_numpy(std.apply(mat.numpy())).float().unsqueeze(0)
        with torch.no_grad():
            e = geo.population_energy(mat, v, batch["robot_idx"].cpu(),
                                      batch["program_idx"].cpu(),
                                      batch["regime_ids"].cpu())["population_energy"]
        return tail_energy(e.numpy(), v.numpy())

    # Phase 2: per-failure causal analysis (DEV only).
    per_failure: list[dict] = []
    for root in args.dev_roots:
        samples, manifest = load_chronological(root)
        by_id = {s.file_id: s for s in samples}
        files = {row["file_id"]: row for row in manifest["files"]}
        episodes = manifest.get("episodes", [])
        failures = [e for e in episodes if str(e.get("kind")) == "failure"]
        degs: dict[str, list] = {}
        for e in episodes:
            if str(e.get("kind")) == "degradation":
                degs.setdefault(str(e["robot_id"]), []).append(e)
        maint: dict[str, list] = {}
        for e in episodes:
            if str(e.get("kind")) == "maintenance":
                maint.setdefault(str(e["robot_id"]), []).append(
                    (float(e["start_time"]), float(e.get("end_time") or 0.0)))

        def in_maint(robot: str, start: float, end: float) -> bool:
            return any(s < end and start < e for s, e in maint.get(robot, []))

        by_robot: dict[str, list] = {}
        for s in samples:
            robot = str(files[s.file_id]["robot_id"])
            by_robot.setdefault(robot, []).append(s)
        for evs in by_robot.values():
            evs.sort(key=lambda s: float(files[s.file_id]["end_time"]))

        # file-level score cache per root (both probes)
        cache: dict[str, dict] = {}
        for s in samples:
            cache[s.file_id] = {"amp": file_score(s.x), "tail": geometry_tail(s)}

        for f in failures:
            robot = str(f["robot_id"])
            fstart = float(f["start_time"])
            recent = [e for e in degs.get(robot, [])
                      if float(e.get("end_time") or 0.0) >= fstart - LOOKBACK_S
                      and float(e["start_time"]) <= fstart]
            long_recent = [e for e in recent
                           if float(e.get("end_time") or 0.0) - float(e["start_time"]) >= PROG_MIN_DUR_S]
            if long_recent:
                cat = "progressive"
            elif not recent:
                cat = "abrupt"
            else:
                cat = "weak"
            other_starts = [float(e["start_time"]) for e in failures
                            if str(e["robot_id"]) == robot and str(e["episode_id"]) != str(f["episode_id"])]
            deg_spans = [(float(e["start_time"]), float(e.get("end_time") or 0.0))
                         for e in degs.get(robot, [])]
            rec: dict = {"root": str(root), "episode": f["episode_id"],
                         "robot": robot, "category": cat, "fail_start": fstart}
            for wname, wlen in WINDOWS.items():
                win = [s for s in by_robot.get(robot, [])
                       if in_window_causal(float(files[s.file_id]["end_time"]), fstart, wlen)
                       and not in_maint(robot, float(files[s.file_id]["start_time"]),
                                        float(files[s.file_id]["end_time"]))]
                rec[wname + "_n"] = len(win)
                rec[wname + "_amp"] = [cache[s.file_id]["amp"] for s in win]
                rec[wname + "_tail"] = [cache[s.file_id]["tail"] for s in win]
            # Clean causal baseline with per-reason exclusion accounting.
            bg_considered, bg_kept = 0, 0
            excluded: dict[str, int] = {}
            bg_amp, bg_tail = [], []
            for s in by_robot.get(robot, []):
                fend = float(files[s.file_id]["end_time"])
                if not (fstart - BASE_SPAN_D * DAY <= fend < fstart - BASE_GAP_D * DAY):
                    continue
                bg_considered += 1
                ok, reason = baseline_eligible(
                    float(files[s.file_id]["start_time"]), fend,
                    fail_start=fstart, other_starts=other_starts,
                    degradations=deg_spans,
                    maint_list=maint.get(robot, []),
                    quarantined=bool(files[s.file_id]["is_quarantined"]))
                if not ok:
                    excluded[reason] = excluded.get(reason, 0) + 1
                    continue
                bg_kept += 1
                bg_amp.append(cache[s.file_id]["amp"])
                bg_tail.append(cache[s.file_id]["tail"])
            rec["bg_considered"] = bg_considered
            rec["bg_kept"] = bg_kept
            rec["bg_excluded"] = excluded
            rec["bg_amp_median"] = float(np.median(bg_amp)) if bg_amp else float("nan")
            rec["bg_tail_median"] = float(np.median(bg_tail)) if bg_tail else float("nan")
            per_failure.append(rec)

    cats: dict[str, dict] = {}
    for cat in ("progressive", "weak", "abrupt"):
        rows = [r for r in per_failure if r["category"] == cat]
        entry: dict = {"n_failures": len(rows)}
        for probe in ("amp", "tail"):
            for wname in WINDOWS:
                gaps = np.array([
                    np.median(r[f"{wname}_{probe}"]) - r[f"bg_{probe}_median"]
                    for r in rows
                    if r[wname + "_n"] > 0 and r["bg_kept"] > 0
                    and np.isfinite(r[f"bg_{probe}_median"])])
                entry[f"{probe}_{wname}"] = describe(gaps)
        entry["window_n_median"] = {
            w: float(np.median([r[w + "_n"] for r in rows])) if rows else 0.0
            for w in WINDOWS}
        entry["bg_kept_median"] = float(
            np.median([r["bg_kept"] for r in rows])) if rows else 0.0
        entry["bg_empty_n"] = int(sum(1 for r in rows if r["bg_kept"] == 0))
        cats[cat] = entry
    result = {
        "provenance": {
            "commit": commit,
            "commit_source": "git rev-parse HEAD in execution checkout (dirty source tree refused)",
            "dev_roots": list(args.dev_roots),
            "control_ckpt_sha256": CONTROL_CKPT_SHA256,
            "geometry_hyperparams": geo_kwargs,
            "top_q": TOP_Q,
            "baseline_rule": "same-robot files ending in [fail-28d, fail-7d) excluding "
                             "quarantined / maintenance-overlapping / degradation-overlapping / "
                             "other-failure-window files; exclusions counted by reason",
            "score_rule": "causal only (file end_time <= window end); amplitude zero-fit; "
                          "geometry_tail = top-10% mean of frozen FIT-conditional population energies",
        },
        "fit": {"n_files": len(fit_files), "n_rows": int(rows_a.shape[0]),
                "feature_dim": int(rows_a.shape[1])},
        "n_failures": len(per_failure),
        "scope": ("progressive failures only; weak/abrupt categories are UNAVAILABLE "
                  "under v2 abrupt_rate=0 (no new data generated; no generalization)."),
        "categories": cats,
        "per_failure": [
            {"episode": r["episode"], "robot": r["robot"], "category": r["category"],
             "w1d_n": r["w1d_n"], "w7d_n": r["w7d_n"],
             "bg_considered": r["bg_considered"], "bg_kept": r["bg_kept"],
             "bg_excluded": r["bg_excluded"]}
            for r in per_failure],
    }
    (out_dir / "task11_diag.json").write_text(json.dumps(result, indent=2))
    print(f"WROTE {out_dir / 'task11_diag.json'} ({len(per_failure)} failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
