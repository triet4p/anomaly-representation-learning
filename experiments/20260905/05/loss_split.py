"""Sprint 9 Task 6: loss splits by normal/anomaly for a trained checkpoint.

Usage: V1_CKPT=<checkpoint> .venv/bin/python experiments/20260905/05/loss_split.py
Loads the checkpoint config (trainer unchanged), groups medium val into
normal-only and abnormal-only batches, and reports mean prediction and
contrastive loss per group (eval mode, view RNG reset per group for fairness).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

REPO = Path(os.environ.get("V1_REPO_ROOT", "/home/trietlm/anomaly-representation-learning"))
DATA_ROOT = REPO / "data" / "generated" / "medium"
CKPT = Path(os.environ["V1_CKPT"])
BATCH = 64

import sys
sys.path.insert(0, str(REPO / "src"))

from representation import V1Config
from representation.checkpoint import load_checkpoint
from representation.criterion import FileContrastiveCriterion, JointRepresentationCriterion, ProgressiveLambda
from representation.data import FileDataset, collate_variable_files
from representation.model import V1RepresentationModel
from synth.config import PatchConfig
from synth.patchify import Patchifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
payload = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = V1Config(**payload.get("config", {}))
patchifier = Patchifier(PatchConfig(patch_size=cfg.patch_size, stride=cfg.stride, pad_end=True))
model = V1RepresentationModel(cfg, patchifier=patchifier).to(device).eval()
load_checkpoint(CKPT, model)
crit = JointRepresentationCriterion(
    contrastive=FileContrastiveCriterion(temperature=cfg.contrastive_temperature),
    lambda_schedule=ProgressiveLambda(lambda_max=cfg.contrastive_weight_max,
                                      ramp_steps=cfg.contrastive_ramp_steps,
                                      warmup_steps=cfg.contrastive_warmup_steps))

normals, abnormals = [], []
for s in FileDataset(DATA_ROOT, split="val"):
    (abnormals if s.file_label.value.lower() == "abnormal" else normals).append(s)
print("val groups:", len(normals), "normal,", len(abnormals), "abnormal")

model.reset_view_rng(0)
with torch.no_grad():
    for name, group in (("normal", normals), ("abnormal", abnormals)):
        pl, cl, n = 0.0, 0.0, 0
        model.reset_view_rng(1234)
        for i in tqdm(range(0, len(group), BATCH), desc=name, leave=False):
            b = collate_variable_files(group[i:i + BATCH], patchifier,
                                       masking_config=cfg, masking_seed=11)
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            dg = crit(model(bd), step=10**9)
            m = len(group[i:i + BATCH])
            pl += float(dg["prediction_loss"]) * m
            cl += float(dg["contrastive_loss"]) * m
            n += m
        print(f"{name}: prediction_loss={pl / n:.6f} contrastive_loss={cl / n:.6f} (n={n})")
print("config: lambda_max", cfg.contrastive_weight_max, "tau", cfg.contrastive_temperature,
      "gain", cfg.contrastive_gain_std, "offset", cfg.contrastive_offset_std,
      "noise", cfg.contrastive_noise_std, "shift", cfg.contrastive_max_shift)
