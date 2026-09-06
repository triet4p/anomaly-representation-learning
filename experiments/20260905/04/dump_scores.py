"""Sprint 9 Tasks 4-5: dump per-file scores + frozen embeddings for offline analysis.

Scores medium val/test with the restored 8192-row bank (no refit) and saves
S_pred, S_pop, labels, families, and file embeddings to scores_medium.npz.
Deterministic seeds match the inference notebooks (ref 5 / val 6 / test 7).
"""
import json
import os
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

REPO = Path(os.environ.get("V1_REPO_ROOT", "/home/trietlm/anomaly-representation-learning"))
DATA_ROOT = Path(os.environ.get("V1_DATA_ROOT", REPO / "data" / "generated" / "medium"))
CKPT = Path(os.environ.get("V1_CHECKPOINT_PATH", REPO / "checkpoints" / "v1_representation_20260904_01.pt"))
OUT = Path(os.environ.get("V1_OUT", "experiments/20260905/04/scores_medium.npz"))
BATCH = int(os.environ.get("V1_BATCH_SIZE", "64"))

import sys
sys.path.insert(0, str(REPO / "src"))

from representation import V1Config
from representation.checkpoint import load_checkpoint
from representation.data import FileDataset, collate_variable_files
from representation.inference import NormalReferenceBank, RepresentationInference, prepare_reference_bank
from representation.model import V1RepresentationModel
from synth.config import PatchConfig
from synth.patchify import Patchifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
payload = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = V1Config(**payload.get("config", {}))
patchifier = Patchifier(PatchConfig(patch_size=cfg.patch_size, stride=cfg.stride, pad_end=True))

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

inference = RepresentationInference(model, bank, patchifier, masking_config=cfg)
out = {}
with torch.no_grad():
    for split, seed in (("val", 6), ("test", 7)):
        dataset = FileDataset(DATA_ROOT, split=split)
        total = manifest["counts"][split]
        preds, pops, labels, fams, embs = [], [], [], [], []
        chunk, bidx = [], [0]
        def flush():
            if not chunk:
                return
            b = collate_variable_files(chunk, patchifier, masking_config=cfg, masking_seed=seed + bidx[0])
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            s = inference.score_batch(b)
            preds.extend(s["S_pred"].cpu().tolist())
            pops.extend(s["S_pop"].cpu().tolist())
            mout = model(bd)
            embs.append(mout["file_embedding"].cpu().float())
            labels.extend([getattr(l, "value", str(l)).lower() == "abnormal" for l in b["file_labels"]])
            fams.extend([m.family.value if m is not None else "normal" for m in b["anomaly_meta"]])
            chunk.clear(); bidx[0] += 1
        for sample in tqdm(dataset, total=total, desc=f"Scoring {split}"):
            chunk.append(sample)
            if len(chunk) == BATCH:
                flush()
        flush()
        out[split] = {
            "S_pred": np.asarray(preds, dtype=np.float64),
            "S_pop": np.asarray(pops, dtype=np.float64),
            "is_anomalous": np.asarray(labels, dtype=bool),
            "family": np.asarray(fams),
            "emb": torch.cat(embs).numpy(),
        }
        print(split, "n=", len(preds), "abnormal=", int(np.asarray(labels).sum()),
              "S_pred mean=%.4f S_pop mean=%.4f" % (np.mean(preds), np.mean(pops)))

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT,
    val_pred=out["val"]["S_pred"], val_pop=out["val"]["S_pop"], val_y=out["val"]["is_anomalous"],
    val_fam=out["val"]["family"], val_emb=out["val"]["emb"],
    test_pred=out["test"]["S_pred"], test_pop=out["test"]["S_pop"], test_y=out["test"]["is_anomalous"],
    test_fam=out["test"]["family"], test_emb=out["test"]["emb"])
print("wrote", OUT, OUT.stat().st_size, "bytes")
