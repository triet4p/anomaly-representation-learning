"""Sprint 9 Task 6: ablation training run (trainer unchanged, config-only knobs).

Mirrors notebooks/train_v1_representation.ipynb on the medium dataset.
Env knobs: V1_LAMBDA_MAX (0.1), V1_TEMPERATURE (0.2), V1_AUG_SCALE (1.0),
V1_EPOCHS (40), V1_BATCH_SIZE (128), V1_CHECKPOINT_OUT, V1_DATA_ROOT.
Writes checkpoint + history JSON next to the checkpoint.
"""
import json
import os
from itertools import islice
from pathlib import Path

import torch
from tqdm.auto import tqdm

REPO = Path(os.environ.get("V1_REPO_ROOT", "/home/trietlm/anomaly-representation-learning"))
DATA_ROOT = Path(os.environ.get("V1_DATA_ROOT", REPO / "data" / "generated" / "medium"))
CKPT_OUT = Path(os.environ.get("V1_CHECKPOINT_OUT", REPO / "experiments/20260905/05/ckpt_baseline.pt"))
EPOCHS = int(os.environ.get("V1_EPOCHS", "40"))
BATCH = int(os.environ.get("V1_BATCH_SIZE", "128"))
LAM = float(os.environ.get("V1_LAMBDA_MAX", "0.1"))
TAU = float(os.environ.get("V1_TEMPERATURE", "0.2"))
AUG = float(os.environ.get("V1_AUG_SCALE", "1.0"))
LR = float(os.environ.get("V1_LR", "1.5e-3"))

import sys
sys.path.insert(0, str(REPO / "src"))

from representation import V1Config
from representation.data import FileDataset, collate_variable_files
from representation.inference import NormalReferenceBank
from representation.model import V1RepresentationModel
from representation.trainer import RepresentationTrainer
from synth.config import PatchConfig
from synth.patchify import Patchifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
train_total, val_total = manifest["counts"]["train"], manifest["counts"]["val"]

cfg = V1Config(
    n_channels=6, patch_size=32, stride=16,
    contrastive_weight_max=LAM,
    contrastive_warmup_steps=196 * 5, contrastive_ramp_steps=196 * 5,
    contrastive_temperature=TAU,
    contrastive_gain_std=0.05 * AUG, contrastive_offset_std=0.03 * AUG,
    contrastive_noise_std=0.015 * AUG, contrastive_max_shift=int(round(4 * AUG)),
)
patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))


class StreamingBatchDataset:
    def __init__(self, split, b_size, base_seed):
        self.samples = list(FileDataset(DATA_ROOT, split=split))
        self.b_size = b_size
        self.base_seed = base_seed

    def __iter__(self):
        indices = torch.randperm(len(self.samples)).tolist()
        chunk, bidx = [], 0
        for idx in indices:
            chunk.append(self.samples[idx])
            if len(chunk) == self.b_size:
                yield collate_variable_files(chunk, patchifier, masking_config=cfg,
                                             masking_seed=self.base_seed + bidx)
                chunk, bidx = [], bidx + 1
        if chunk:
            yield collate_variable_files(chunk, patchifier, masking_config=cfg,
                                         masking_seed=self.base_seed + bidx)


train_batches = StreamingBatchDataset("train", BATCH, 3)
val_batches = StreamingBatchDataset("val", BATCH, 4)
num_b = (train_total + BATCH - 1) // BATCH
num_vb = (val_total + BATCH - 1) // BATCH
total_steps = EPOCHS * num_b

model = V1RepresentationModel(cfg, patchifier=patchifier)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
trainer = RepresentationTrainer(model, optimizer=optimizer, scheduler=scheduler,
                                device=device, seed=cfg.seed, step=0,
                                selection_metric="val_stationary_joint_loss",
                                max_grad_norm=1.0)
history = []
epoch_pbar = tqdm(range(1, EPOCHS + 1), desc="Training epochs")
for epoch in epoch_pbar:
    tm = trainer.train_epoch(tqdm(train_batches, total=num_b, desc=f"Epoch {epoch}/{EPOCHS}", leave=False))
    m = dict(tm)
    vm = trainer.validate(tqdm(val_batches, total=num_vb, desc="Validating", leave=False))
    m.update({f"val_{k}": v for k, v in vm.items()})
    trainer.record_eval(m, epoch=epoch)
    trainer.history[-1] = m
    history.append({k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in m.items()})
    epoch_pbar.set_postfix({"stat_joint": f"{m['stationary_joint_loss']:.4f}",
                            "val_stat": f"{m['val_stationary_joint_loss']:.4f}",
                            "pred": f"{m['prediction_loss']:.4f}",
                            "cont": f"{m['contrastive_loss']:.4f}",
                            "lambda": f"{m['effective_lambda']:.3f}"})
if trainer.best_state is not None:
    trainer.restore_best_state()
    print(f"[Selection] restored best epoch {trainer.best_epoch} score {trainer.best_loss:.4f}")

model.eval()
ref_loader = StreamingBatchDataset("train", BATCH, 99)
embs = []
with torch.no_grad():
    done = 0
    for batch in ref_loader:
        if done >= 8192:
            break
        dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        embs.append(model(dev)["file_embedding"].cpu())
        done += batch["signals"].shape[0]
ref = torch.cat(embs)[:8192]
bank = NormalReferenceBank(k=min(cfg.knn_k, ref.shape[0])).fit(ref)
CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
trainer.save_checkpoint(str(CKPT_OUT), reference_bank=bank)
(CKPT_OUT.parent / (CKPT_OUT.stem + "_history.json")).write_text(json.dumps(history, indent=1))
print("checkpoint", CKPT_OUT, "step", trainer.step, "bank", bank.embeddings.shape[0])
print("final", json.dumps(history[-1], indent=1))
