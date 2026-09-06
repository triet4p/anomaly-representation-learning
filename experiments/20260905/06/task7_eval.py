"""Sprint 9 Task 7: production mixed-sample evaluation (server, GPU).

Deterministic balanced sample from production test: first 400 files per
anomaly family (9 families -> <=3600 abnormal) + first 3600 normal files,
all in dataset order. Scores with the production checkpoint's restored
8192-row bank (no refit). Thresholds calibrated on production val (MAD x2.5,
same convention as Task 3). Reports confusion, per-family recall, margin
metrics, and localization quality against ground-truth anomaly masks.
"""
import json
import os
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

REPO = Path(os.environ.get("V1_REPO_ROOT", "/home/trietlm/anomaly-representation-learning"))
DATA_ROOT = REPO / "data" / "generated" / "production"
CKPT = Path(os.environ.get("V1_CHECKPOINT_PATH",
            REPO / "checkpoints" / "v1_representation_20260904_01.pt"))
OUTDIR = Path(os.environ.get("V1_OUT", "experiments/20260905/06"))
PER_FAMILY = int(os.environ.get("V1_PER_FAMILY", "400"))
BATCH = int(os.environ.get("V1_BATCH_SIZE", "64"))

import sys
sys.path.insert(0, str(REPO / "src"))

from representation import V1Config
from representation.checkpoint import load_checkpoint
from representation.data import FileDataset, collate_variable_files
from representation.inference import (NormalReferenceBank, RepresentationInference,
                                      mad_threshold, prepare_reference_bank)
from representation.model import V1RepresentationModel
from synth.config import PatchConfig
from synth.patchify import Patchifier
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
payload = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = V1Config(**payload.get("config", {}))
patchifier = Patchifier(PatchConfig(patch_size=cfg.patch_size, stride=cfg.stride, pad_end=True))

# ---- Phase A: scan production test order, deterministic per-class quotas ----
wanted_fam = PER_FAMILY
sel_normal, sel_abn = [], []
fam_counts, n_normal_seen, n_abn_seen = {}, 0, 0
for s in tqdm(FileDataset(DATA_ROOT, split="test"), total=manifest["counts"]["test"], desc="Scanning test"):
    abnormal = s.file_label.value.lower() == "abnormal"
    if abnormal:
        n_abn_seen += 1
        fam = s.anomaly_meta.family.value
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
        if fam_counts[fam] <= wanted_fam:
            sel_abn.append(s)
    else:
        n_normal_seen += 1
        if len(sel_normal) < len(sel_abn) or len(sel_normal) < wanted_fam * 9:
            sel_normal.append(s)
n_abn = len(sel_abn)
sel_normal = sel_normal[:n_abn]  # exact balance: pairs
print("test scan: normal seen", n_normal_seen, "abnormal seen", n_abn_seen)
print("family totals:", json.dumps(fam_counts, indent=1, sort_keys=True))
print("selected:", n_abn, "abnormal +", len(sel_normal), "normal")
assert n_abn > 0 and len(sel_normal) == n_abn, "balanced pair accounting failed"
assert all(v >= wanted_fam for v in fam_counts.values()), "a family fell short of quota"

# ---- Phase B: model + restored bank ----
model = V1RepresentationModel(cfg, patchifier=patchifier).to(device).eval()
bank = NormalReferenceBank(k=min(cfg.knn_k, manifest["counts"]["train"]))
meta = load_checkpoint(CKPT, model, reference_bank=bank)
ref_batch = collate_variable_files(
    list(islice(FileDataset(DATA_ROOT, split="train"), BATCH)),
    patchifier, masking_config=cfg, masking_seed=5)
with torch.no_grad():
    ref_out = model({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in ref_batch.items()})
bank_source = prepare_reference_bank(bank, ref_out["file_embedding"], refit=False)
print(f"bank rows {bank.embeddings.shape[0]} source {bank_source} step {meta['step']}")
assert bank.embeddings.shape[0] == 8192 and bank_source == "restored"

# ---- Phase C: thresholds on production val (normal reference operating point) ----
inference = RepresentationInference(model, bank, patchifier, masking_config=cfg)
val_pred, val_pop = [], []
val_samples = list(islice(FileDataset(DATA_ROOT, split="val"), 5000))
with torch.no_grad():
    for i in tqdm(range(0, len(val_samples), BATCH), desc="Calibrating on val"):
        b = collate_variable_files(val_samples[i:i + BATCH], patchifier, masking_config=cfg, masking_seed=6)
        s = inference.score_batch(b)
        val_pred.extend(s["S_pred"].cpu().tolist())
        val_pop.extend(s["S_pop"].cpu().tolist())
th_pred = mad_threshold(torch.tensor(val_pred), multiplier=2.5)
th_pop = mad_threshold(torch.tensor(val_pop), multiplier=2.5)
print(f"val(n={len(val_pred)}) S_pred mean={np.mean(val_pred):.4f} th={th_pred:.4f} | "
      f"S_pop mean={np.mean(val_pop):.4f} th={th_pop:.4f}")

# ---- Phase D: score mixed sample ----
all_samples = sel_normal + sel_abn
order = np.random.default_rng(0).permutation(len(all_samples))  # fixed interleave, labels kept
preds, pops, ys, fams, traces, masks = [], [], [], [], [], []
with torch.no_grad():
    for i in tqdm(range(0, len(all_samples), BATCH), desc="Scoring mixed"):
        chunk = [all_samples[j] for j in order[i:i + BATCH]]
        b = collate_variable_files(chunk, patchifier, masking_config=cfg, masking_seed=7)
        s = inference.score_batch(b)
        preds.extend(s["S_pred"].cpu().tolist())
        pops.extend(s["S_pop"].cpu().tolist())
        for smp, ts in zip(chunk, s["timestep_scores"]):
            ys.append(smp.file_label.value.lower() == "abnormal")
            fams.append(smp.anomaly_meta.family.value if smp.anomaly_meta is not None else "normal")
            traces.append(np.asarray(ts, dtype=np.float64))
            m = smp.anomaly_mask
            masks.append(np.asarray(m.any(axis=0), dtype=bool) if m is not None else None)
preds = np.asarray(preds); pops = np.asarray(pops); ys = np.asarray(ys, dtype=bool)
fams = np.asarray(fams)

det = ((preds > th_pred) | (pops > th_pop))
tp = int((det & ys).sum()); fp = int((det & ~ys).sum())
fn = int(((~det) & ys).sum()); tn = int(((~det) & ~ys).sum())
prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
f1 = 2 * prec * rec / max(1e-12, prec + rec)
print(f"mixed n={len(ys)} ({(~ys).sum()} normal, {ys.sum()} abnormal)")
print(f"MADx2.5 (th_pred={th_pred:.4f} th_pop={th_pop:.4f}): TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
print(f"AUROC S_pred={roc_auc_score(ys, preds):.4f} S_pop={roc_auc_score(ys, pops):.4f}")
a, b = preds[ys], preds[~ys]
d_pred = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
a, b = pops[ys], pops[~ys]
d_pop = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
print(f"margin: S_pred abn-mean={preds[ys].mean():.4f} nor-mean={preds[~ys].mean():.4f} d={d_pred:.4f} | "
      f"S_pop abn-mean={pops[ys].mean():.4f} nor-mean={pops[~ys].mean():.4f} d={d_pop:.4f}")

print("per-family recall (MAD operating point):")
fam_rows = {}
for fam in sorted(set(fams.tolist()) - {"normal"}):
    m = fams == fam
    r = float(det[m].mean())
    fam_rows[fam] = {"n": int(m.sum()), "recall": r}
    print(f"  {fam:28s} detected {int(det[m].sum())}/{int(m.sum())} recall={r:.4f}")

# ---- Phase E: localization vs ground-truth masks ----
peak_in, topmass, nondeg = [], [], 0
for t, mk, y in zip(traces, masks, ys):
    if not y or mk is None or len(mk) != len(t):
        continue
    if np.count_nonzero(t) > 1:
        nondeg += 1
    peak_in.append(bool(mk[int(np.argmax(t))]))
    k = max(1, len(t) // 10)
    top = np.argpartition(t, -k)[-k:]
    topmass.append(float(np.asarray(t)[top].sum() / max(1e-12, np.asarray(t).sum())))
print(f"localization on {len(peak_in)} abnormal files: argmax-in-mask={np.mean(peak_in):.4f} "
      f"top10%-mass-in-mask={np.mean(topmass):.4f} nondegenerate-traces={nondeg}/{len(peak_in)}")
print("trace stats: mean len", float(np.mean([len(t) for t in traces])),
      "mean nonzero frac", float(np.mean([np.count_nonzero(t) / len(t) for t in traces])))

OUTDIR.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUTDIR / "task7_mixed.npz", S_pred=preds, S_pop=pops, y=ys, fam=fams,
                    th_pred=np.float64(th_pred), th_pop=np.float64(th_pop))
(OUTDIR / "task7_summary.json").write_text(json.dumps({
    "n_normal": int((~ys).sum()), "n_abnormal": int(ys.sum()),
    "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    "precision": prec, "recall": rec, "f1": f1,
    "auroc_pred": float(roc_auc_score(ys, preds)), "auroc_pop": float(roc_auc_score(ys, pops)),
    "cohen_d_pred": float(d_pred), "cohen_d_pop": float(d_pop),
    "families": fam_rows,
    "argmax_in_mask": float(np.mean(peak_in)), "top10_mass": float(np.mean(topmass)),
    "bank_rows": int(bank.embeddings.shape[0]), "bank_source": bank_source,
}, indent=1))
print("wrote", OUTDIR / "task7_mixed.npz", "and task7_summary.json")
