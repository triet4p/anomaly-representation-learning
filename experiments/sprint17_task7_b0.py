"""Sprint 17 Task 7 — train and evaluate unchanged baseline B0 on Development.

Retrains the unchanged V1 representation-to-score architecture on the frozen
Sprint 17 Fit histories for exactly 300 optimizer steps per frozen model seed,
refits the healthy reference bank on Fit only, calibrates per-branch q95
thresholds on Calibration only, and evaluates both independent score branches
(S_pred context mismatch, S_pop population mismatch) on Development.

Frozen contract: protocol ``sprint17-ablation-v4``, role binding
``experiments/sprint17-role-binding-v3.json``
(digest ``075868b22…8230ba``), model seeds ``[171701, 171702, 171703]``,
``optimizer_steps = checkpoint_step = 300``, calibration quantile 0.95,
qualification ``MEASURABLE_WITH_USER_WAIVER``.

Explicit non-goals: no Confirmation/Sealed access (refused at the loader),
no alternative arm, no Task 8 audit, no threshold tuning, no early stopping
or best-state selection (final step-300 state is the checkpoint), no extra
gradient runs, no pooled rescue.

Scoring-mask rule (frozen for reproducibility): every scored file uses
``masking_seed = sha256(file_id) mod 2**31`` with the frozen B0 masking
config, so file scores are batching- and order-independent. Training masks
use ``masking_seed = global step index`` over a deterministically shuffled
Fit-healthy pool (per-epoch shuffle RNG seeded by ``(model_seed, epoch)``).
"""

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PROTOCOL_ID = "sprint17-ablation-v4"
SCHEMA_ID = "sprint17-ablation-result-v4"
DATA_PROTOCOL = "sprint15-benchmark-protocol-v7"
BINDING_SHA256 = "075868b22c028a981cc63ffe37b8d29720cd324c3e2c7254ce688678758230ba"
MODEL_SEEDS = (171701, 171702, 171703)
OPTIMIZER_STEPS = 300
CHECKPOINT_STEP = 300
CALIBRATION_QUANTILE = 0.95
QUALIFICATION = "MEASURABLE_WITH_USER_WAIVER"
WAIVER_ID = "S17-V4-WAIVER-H-S17-V3-DESIGN-03-COHORT_MIX_15_60-P-91-149"
WAIVER_SCOPE = {
    "history_id": "H-S17-V3-DESIGN-03",
    "role": "DESIGN",
    "gate": "cohort_mix_15_60",
    "cohort": "P",
    "observed_numerator": 91,
    "observed_denominator": 149,
    "observed_share": 0.610738255033557,
    "original_upper_bound": 0.6,
}

FIT_ROLES = (("H-S17-V3-FIT-01", 2804), ("H-S17-V3-FIT-02", 2805),
             ("H-S17-V3-FIT-03", 2806))
CAL_ROLE = ("H-S17-V3-CAL-01", 2807)
DEV_ROLES = (("H-S17-V3-DEV-01", 2808), ("H-S17-V3-DEV-02", 2809),
             ("H-S17-V3-DEV-03", 2810), ("H-S17-V3-DEV-04", 2811))

# Task 3 v3 materialization manifest hashes for the 8 consumed roots
# (authoritative record: artifacts/sprint-17/task-3-v3.md section 3).
EXPECTED_MANIFEST_SHA256 = {
    "H-S17-V3-FIT-01": "46afe203c29a211a3ca9c8b21291fcaf1271d325c0ec7b6849b1d6fc5e11db13",
    "H-S17-V3-FIT-02": "136476984c0d1ee69e2aad01f6f1efc91750d8e98dd9d872b09c80d6a2a96669",
    "H-S17-V3-FIT-03": "07a223ca5714943a3973b01f4d851c639e54f9d2109cfdc8ceaa952c2637d305",
    "H-S17-V3-CAL-01": "e6bcdba617dabae52aba57905ba3f471768073c725df9afbc1c3f97919e0f101",
    "H-S17-V3-DEV-01": "83a3559794ac73bc673105e84c436267f699f2be1b3bfb684b7de952b9a3f438",
    "H-S17-V3-DEV-02": "75eb035f41387827ae39c5d2f82ed383e9aaa0d92b7caec57cedd118e2bd4073",
    "H-S17-V3-DEV-03": "f94e1130bfd88cc6548616d5eec5dbf216f14713b0145696a8aac868899744ed",
    "H-S17-V3-DEV-04": "ae2c87c21ce2d2f344538d21b5b5da06189cf8cb279afa332b6576c0e489ac8c",
}

#: Files whose bytes define the metric-code identity of this run.
METRIC_CODE_FILES = (
    "src/synth/probe15.py",
    "src/synth/events.py",
    "src/representation/attribution_metrics.py",
    "src/representation/sprint17_ablation.py",
    "experiments/sprint17_task7_b0.py",
)

#: Reference input length for the frozen FLOP count (near the Fit T median).
FLOP_REFERENCE_T = 512


# ---------------------------------------------------------------------------
# Pure helpers (focused-test surface)
# ---------------------------------------------------------------------------

def canonical_hash(obj: object) -> str:
    """sha256 of canonical JSON (sorted keys, compact separators, UTF-8)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_seed(file_id: str) -> int:
    """Deterministic per-file masking seed, batching- and order-independent."""
    digest = hashlib.sha256(file_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


def eligible_healthy_row(row: dict, windows: dict, program_reserve: str,
                         robot_reserve: str) -> bool:
    """Frozen Fit/Calibration row predicate (protocol v4 section 1.2).

    Excludes non-normal labels, quarantined, censored, reserve program/robot,
    and maintenance-overlapping rows. Labels/masks remain post-hoc only.
    """
    if row.get("file_label") != "normal":
        return False
    if row.get("is_quarantined") or row.get("is_censored"):
        return False
    if row.get("program_id") == program_reserve or row.get("robot_id") == robot_reserve:
        return False
    for start, end in windows.get(row.get("robot_id", ""), []):
        if start < row["end_time"] and row["start_time"] < end:
            return False
    return True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_jsonable(obj: object) -> object:
    """Convert numpy scalars/arrays (incl. NaN) into strict-JSON values."""
    import math
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    # Bool before int: Python bool is an int subclass.
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def build_score_map(roles: list[str], dev_out: dict[str, dict],
                    branch: str) -> dict[str, float]:
    """Cross-history file-score map; fail fast on file-id collision."""
    scores: dict[str, float] = {}
    for role in roles:
        for fid, value in zip(dev_out[role]["ids"], dev_out[role][branch]):
            if fid in scores:
                raise ValueError(f"file-id collision across histories: {fid}")
            scores[fid] = float(value)
    return scores


def check_binding_digest(binding_path: Path) -> dict:
    """Recompute the canonical v3 binding digest; fail fast on mismatch."""
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in binding.items() if k != "binding_sha256"}
    digest = canonical_hash(payload)
    if digest != BINDING_SHA256:
        raise ValueError(f"role-binding digest mismatch: {digest} != {BINDING_SHA256}")
    if binding.get("protocol_id") != "sprint17-ablation-v3":
        raise ValueError("role-binding protocol_id is not sprint17-ablation-v3")
    return binding


def load_verified_root(data_root: Path, group: str, role: str,
                       seed: int) -> tuple[list, dict]:
    """Load one chronological root with fail-fast role/manifest/score checks."""
    from synth import balanced as B
    from synth.chronicle import load_chronological

    if group in ("CONFIRMATION", "SEALED"):
        raise ValueError(f"refusing {group} access in Task 7 (Development only)")
    root = data_root / group / role
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing root manifest (no regeneration): {manifest_path}")
    manifest_digest = sha256_file(manifest_path)
    expected = EXPECTED_MANIFEST_SHA256.get(role)
    if expected is not None and manifest_digest != expected:
        raise ValueError(f"manifest hash mismatch for {role}: {manifest_digest}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("role") != role:
        raise ValueError(f"role mismatch in {role}: {manifest.get('role')}")
    if manifest.get("protocol") != DATA_PROTOCOL:
        raise ValueError(f"protocol tag mismatch in {role}")
    if manifest.get("seeds", {}).get("health") != seed:
        raise ValueError(f"seed mismatch in {role}")
    samples, reloaded = load_chronological(root)
    if reloaded.get("role") != role:
        raise ValueError(f"reload role mismatch in {role}")
    if not hasattr(B, "PROGRAM_RESERVE"):
        raise ValueError("frozen roster module unavailable")
    return samples, manifest


def b0_config_dict(model_seed: int) -> dict:
    """Frozen B0 concrete configuration (architecture unchanged; data-sized).

    Values follow the accepted production training recipe
    (notebooks/train_v1_representation.ipynb defaults) with two frozen,
    documented Task 7 bindings: ``n_robots=9``/``n_programs=8`` sized to the
    observed Sprint 17 data vocabulary (production 5/8 would fail closed on
    robot_idx 5..8 via ConditionalBatchNorm), and ``weight_decay=0.0`` as
    mandated by the frozen Task 5 TrainingParity contract. Contrastive
    warmup/ramp stay at production absolute values (980/980); no budget
    rescaling, no warm start, no early stopping.
    """
    return {
        "n_channels": 6,
        "patch_size": 32,
        "stride": 16,
        "pad_end": True,
        "d_model": 128,
        "sequence_layers": 4,
        "attention_heads": 4,
        "dropout": 0.1,
        "n_robots": 9,
        "n_programs": 8,
        "use_conditional_norm": True,
        "min_bucket_samples": 32,
        "total_mask_ratio": 0.40,
        "random_fraction": 0.34,
        "info_fraction": 0.33,
        "block_fraction": 0.33,
        "ema_decay": 0.996,
        "prediction_weight": 1.0,
        "contrastive_weight_max": 0.1,
        "contrastive_warmup_steps": 980,
        "contrastive_ramp_steps": 980,
        "contrastive_temperature": 0.2,
        "contrastive_gain_std": 0.05,
        "contrastive_offset_std": 0.03,
        "contrastive_noise_std": 0.015,
        "contrastive_max_shift": 4,
        "knn_k": 5,
        "optimizer": {"class": "AdamW", "lr": 1.5e-3, "weight_decay": 0.0},
        "schedule": {"class": "CosineAnnealingLR", "T_max": 300, "eta_min": 1e-5},
        "batch_size": 128,
        "batch_rule": "epoch-shuffled Fit-healthy pool cycling; per-epoch shuffle "
                      "RNG seeded (model_seed, epoch); masking_seed = global step",
        "scoring_mask_rule": "masking_seed = sha256(file_id) mod 2**31",
        "max_grad_norm": 1.0,
        "precision": "float32",
        "device_class": "cuda",
        "initialization": "default PyTorch init under torch.manual_seed(model_seed); "
                          "V1Config.seed = model_seed (contrastive-view RNG)",
        "optimizer_steps": OPTIMIZER_STEPS,
        "checkpoint_step": CHECKPOINT_STEP,
        "calibration_quantile": CALIBRATION_QUANTILE,
        "model_seed": model_seed,
    }


def count_parameters(model) -> int:
    """Total parameter count (norm-statistic/EMA buffers excluded; documented)."""
    return int(sum(p.numel() for p in model.parameters()))


def count_flops_reference(model, t: int = FLOP_REFERENCE_T) -> int:
    """Deterministic FLOP count of the scoring path at a frozen reference shape.

    Method: forward hooks. Linear/Conv1d count 2 FLOPs per MAC;
    MultiheadAttention counts qkv-projection (2·B·L·E·3E), QK^T + AV
    (2·B·H·L·S·D each) and softmax (5·B·H·L·S); LayerNorm counts 5/element;
    GELU counts 8/element. Dropout/pooling/masks are excluded (documented).
    The reference batch carries no ``file_samples`` so contrastive view
    augmentation is excluded: exactly the S_pred/S_pop scoring path at
    batch 1, C=6, T=``t``, all-valid.
    """
    import torch
    from torch import nn

    total = [0]

    def linear_hook(mod, args, _out):
        total[0] += 2 * int(args[0].numel()) * int(mod.out_features)

    def conv1d_hook(mod, args, out):
        total[0] += (2 * int(out.numel()) * int(mod.in_channels)
                     * int(mod.kernel_size[0]))

    def mha_hook(mod, args, _out):
        q = args[0]
        batch_first = bool(getattr(mod, "batch_first", False))
        if batch_first:
            b, length, width = (int(v) for v in q.shape[:3])
        else:
            length, b, width = (int(v) for v in q.shape[:3])
        key = args[1]
        src_len = int(key.shape[1 if batch_first else 0])
        heads = int(mod.num_heads)
        head_dim = width // heads
        total[0] += 2 * b * length * width * 3 * width
        total[0] += 2 * b * heads * length * src_len * head_dim
        total[0] += 5 * b * heads * length * src_len
        total[0] += 2 * b * heads * length * src_len * head_dim

    def layernorm_hook(_mod, args, _out):
        total[0] += 5 * int(args[0].numel())

    def gelu_hook(_mod, args, _out):
        total[0] += 8 * int(args[0].numel())

    handles = []
    for mod in model.modules():
        if isinstance(mod, nn.MultiheadAttention):
            handles.append(mod.register_forward_hook(mha_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(linear_hook))
        elif isinstance(mod, nn.Conv1d):
            handles.append(mod.register_forward_hook(conv1d_hook))
        elif isinstance(mod, nn.LayerNorm):
            handles.append(mod.register_forward_hook(layernorm_hook))
        elif isinstance(mod, nn.GELU):
            handles.append(mod.register_forward_hook(gelu_hook))
    try:
        import numpy as np

        from representation.data import collate_variable_files
        from synth.config import PatchConfig
        from synth.patchify import Patchifier
        from synth.schema import FileSample, SampleLabel

        rng = np.random.default_rng(7)
        sample = FileSample(
            x=rng.standard_normal((6, t)).astype(np.float32),
            file_id="FLOP-REFERENCE",
            file_label=SampleLabel.NORMAL,
            seed=7,
            generator_version="task7-flop-reference",
            config_hash="none",
            regime_sequence=[],
            robot_idx=0,
            program_idx=0,
        )
        patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
        batch = collate_variable_files([sample], patchifier, masking_config=None)
        batch = {k: v for k, v in batch.items() if k != "file_samples"}
        device = next(model.parameters()).device
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(batch)
        if was_training:
            model.train()
    finally:
        for handle in handles:
            handle.remove()
    return int(total[0])


def evaluable_support(rows: list[dict], ledger: list[dict], wins: dict) -> dict:
    """Mirror probe15.evaluate_histories event selection for support accounting.

    Returns evaluable (key, member-ids, cohort, severity, subtype) triples per
    cohort plus the skipped-failure count and control-window member lists.
    Uses the identical E.* predicates as the canonical metric path.
    """
    from synth import events as E

    pos: dict[str, list] = {"P": [], "W": [], "A": []}
    skipped = 0
    for failure in ledger:
        if E.positive_window_intersects_reset(failure, wins):
            skipped += 1
            continue
        cands = E.pos_files(rows, failure, wins)
        if not cands:
            skipped += 1
            continue
        key = (failure["cohort"], failure["robot_id"], float(failure["failure_time"]))
        pos[failure["cohort"]].append(
            {"key": list(key), "members": [c["file_id"] for c in cands],
             "severity": failure.get("severity"),
             "subtype": failure.get("subtype")})
    anchors = E.anchor_rows(rows, wins)
    controls = E.select_control_windows(anchors, ledger, wins)
    return {"pos": pos, "skipped": skipped,
            "controls": [[m["file_id"] for m in w["members"]] for w in controls]}


# ---------------------------------------------------------------------------
# Torch execution (per-seed train / refit / calibrate / score)
# ---------------------------------------------------------------------------

def build_model(cfg_dict: dict, device):
    import torch
    from representation.config import V1Config
    from representation.model import V1RepresentationModel
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    cfg = V1Config(
        n_channels=cfg_dict["n_channels"], patch_size=cfg_dict["patch_size"],
        stride=cfg_dict["stride"], d_model=cfg_dict["d_model"],
        sequence_layers=cfg_dict["sequence_layers"],
        attention_heads=cfg_dict["attention_heads"], dropout=cfg_dict["dropout"],
        n_robots=cfg_dict["n_robots"], n_programs=cfg_dict["n_programs"],
        use_conditional_norm=cfg_dict["use_conditional_norm"],
        min_bucket_samples=cfg_dict["min_bucket_samples"],
        total_mask_ratio=cfg_dict["total_mask_ratio"],
        random_fraction=cfg_dict["random_fraction"],
        info_fraction=cfg_dict["info_fraction"],
        block_fraction=cfg_dict["block_fraction"], ema_decay=cfg_dict["ema_decay"],
        prediction_weight=cfg_dict["prediction_weight"],
        contrastive_weight_max=cfg_dict["contrastive_weight_max"],
        contrastive_warmup_steps=cfg_dict["contrastive_warmup_steps"],
        contrastive_ramp_steps=cfg_dict["contrastive_ramp_steps"],
        contrastive_temperature=cfg_dict["contrastive_temperature"],
        contrastive_gain_std=cfg_dict["contrastive_gain_std"],
        contrastive_offset_std=cfg_dict["contrastive_offset_std"],
        contrastive_noise_std=cfg_dict["contrastive_noise_std"],
        contrastive_max_shift=cfg_dict["contrastive_max_shift"],
        knn_k=cfg_dict["knn_k"], seed=cfg_dict["model_seed"])
    patchifier = Patchifier(PatchConfig(patch_size=cfg.patch_size,
                                        stride=cfg.stride,
                                        pad_end=cfg_dict["pad_end"]))
    model = V1RepresentationModel(cfg, patchifier=patchifier)
    model.to(device)
    return model, cfg, patchifier


def train_seed(model_seed: int, fit_pool: list, device, log: list,
               steps: int, batch_size: int):
    """Train one B0 seed for exactly ``steps`` optimizer steps (no selection)."""
    import torch
    import numpy as np
    from representation.trainer import RepresentationTrainer

    torch.manual_seed(model_seed)
    np.random.seed(model_seed % (2 ** 32))
    random.seed(model_seed)
    cfg_dict = b0_config_dict(model_seed)
    cfg_dict["batch_size"] = batch_size
    model, cfg, patchifier = build_model(cfg_dict, device)
    opt_cfg = cfg_dict["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"],
                                  weight_decay=opt_cfg["weight_decay"])
    sched_cfg = cfg_dict["schedule"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sched_cfg["T_max"], eta_min=sched_cfg["eta_min"])
    trainer = RepresentationTrainer(model, optimizer=optimizer, scheduler=scheduler,
                                    device=device, seed=model_seed,
                                    max_grad_norm=cfg_dict["max_grad_norm"])
    from representation.data import collate_variable_files

    n_pool = len(fit_pool)
    step = 0
    epoch = 0
    first_loss = None
    last_metrics = None
    while step < steps:
        epoch += 1
        gen = torch.Generator().manual_seed((model_seed * 1000003 + epoch) % (2 ** 32))
        perm = torch.randperm(n_pool, generator=gen).tolist()
        batches = []
        for start in range(0, n_pool, batch_size):
            if step >= steps:
                break
            idx = perm[start:start + batch_size]
            batches.append(collate_variable_files(
                [fit_pool[i] for i in idx], patchifier,
                masking_config=cfg, masking_seed=step))
            step += 1
        last_metrics = trainer.train_epoch(iter(batches))
        if first_loss is None:
            first_loss = last_metrics["joint_loss"]
        log.append({"seed": model_seed, "epoch": epoch, "step": trainer.step,
                    "joint": last_metrics["joint_loss"],
                    "pred": last_metrics.get("prediction_loss"),
                    "cont": last_metrics.get("contrastive_loss"),
                    "lambda": last_metrics.get("effective_lambda"),
                    "lr": last_metrics.get("lr")})
    if steps == OPTIMIZER_STEPS:
        assert trainer.step == OPTIMIZER_STEPS == CHECKPOINT_STEP, trainer.step
    return model, trainer, cfg, patchifier, cfg_dict, first_loss, last_metrics


def score_items(model, bank, patchifier, cfg, items: list[tuple[str, object]],
                device):
    """Score (file_id, sample) items with per-file deterministic mask seeds."""
    import numpy as np
    import torch
    from representation.data import collate_variable_files
    from representation.inference import RepresentationInference

    inference = RepresentationInference(model, bank, patchifier, masking_config=cfg)
    pred, pop, emb, times = [], [], [], []
    with torch.no_grad():
        for fid, sample in items:
            batch = collate_variable_files([sample], patchifier,
                                           masking_config=cfg,
                                           masking_seed=file_seed(fid))
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            res = inference.score_batch(batch)
            pred.append(float(torch.as_tensor(res["S_pred"]).reshape(-1)[0].item()))
            pop.append(float(torch.as_tensor(res["S_pop"]).reshape(-1)[0].item()))
            emb.append(res["file_embedding"][0].detach().cpu())
            times.append(np.asarray(res["timestep_scores"][0], dtype=np.float64))
    return np.array(pred), np.array(pop), torch.stack(emb), times


def embed_items(model, patchifier, cfg, items: list[tuple[str, object]], device):
    """File embeddings for bank fitting (same deterministic masks as scoring)."""
    import torch
    from representation.data import collate_variable_files

    outs = []
    with torch.no_grad():
        for fid, sample in items:
            batch = collate_variable_files([sample], patchifier,
                                           masking_config=cfg,
                                           masking_seed=file_seed(fid))
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            outs.append(model(batch)["file_embedding"][0].detach().cpu())
    return torch.stack(outs)


# ---------------------------------------------------------------------------
# Metrics (canonical probe15 path + frozen extras)
# ---------------------------------------------------------------------------

def branch_metrics(branch: str, threshold: float, dev_file_scores: dict,
                   dev_supports: dict, dev_rows: dict, dev_ledgers: dict,
                   dev_wins: dict, dev_times: dict, fit_branch_scores,
                   roles: list[str]) -> dict:
    """Full per-branch Development metrics: canonical + frozen extras."""
    import numpy as np
    from synth import events as E
    from synth import probe15 as P
    from representation import attribution_metrics as M

    rows_list = [dev_rows[r] for r in roles]
    ledgers = [dev_ledgers[r] for r in roles]
    wins = [dev_wins[r] for r in roles]
    canonical = P.evaluate_histories(dev_file_scores, rows_list, ledgers, wins,
                                     threshold)
    per_history = []
    for h, role in enumerate(roles):
        entry = canonical["per_history"][h]
        support = dev_supports[role]
        pos = support["pos"]
        neg = [dev_file_scores[i] for w in support["controls"] for i in w]

        def auc(ids_scores):
            if not ids_scores or not neg:
                return None
            try:
                return E.roc_auc_tie_aware(ids_scores, neg)
            except E.UnavailableError:
                return None

        pw = [E.window_score(ev["members"], dev_file_scores)
              for ev in pos["P"] + pos["W"]]
        aucs = {"pw": entry["auc_pw"], "p": entry["auc_p"], "w": entry["auc_w"],
                "a": entry["auc_a"]}
        # Severity ordering within P and W (descriptive; cannot promote alone).
        sev = {}
        for cohort in ("P", "W"):
            pts = [(ev["severity"], E.window_score(ev["members"],
                                                   dev_file_scores))
                   for ev in pos[cohort] if ev["severity"] is not None]
            if len(pts) >= 3:
                rho = M.severity_spearman([s for _, s in pts], [v for v, _ in pts])
                sev[cohort] = float(rho) if np.isfinite(rho) else None
            else:
                sev[cohort] = None
        # Category slices on identical control negatives.
        cats = {}
        for cid, (cohort, subtype) in {
                "P1": ("P", "P1"), "P2": ("P", "P2"),
                "W1": ("W", "W1"), "W2": ("W", "W2"),
                "A1": ("A", "A1"), "A2": ("A", "A2")}.items():
            cell = [E.window_score(ev["members"], dev_file_scores)
                    for ev in pos[cohort] if ev["subtype"] == subtype]
            cats[cid] = auc(cell)
        # Unaffected-background stability vs Fit-healthy scores (frozen formula).
        bg = np.array(neg, dtype=np.float64)
        fit = np.asarray(fit_branch_scores, dtype=np.float64)
        stab = M.stability_ratio(bg, fit) if len(bg) and len(fit) else None
        stab_ci = M.stability_ci(bg, fit) if len(bg) and len(fit) else None
        # Localization: peak-timestep-in-horizon rate over evaluable P+W events.
        row_by_id = {r["file_id"]: r for r in dev_rows[role]}
        loc_hits, loc_total = 0, 0
        for ev in pos["P"] + pos["W"]:
            members = ev["members"]
            top = max(members, key=lambda i: dev_file_scores[i])
            t = np.asarray(dev_times[role][top], dtype=np.float64)
            if t.size == 0 or not np.isfinite(t).all() or t.max() == t.min():
                continue
            row = row_by_id[top]
            n_t = t.size
            dt = (row["end_time"] - row["start_time"]) / max(1, n_t)
            peak_time = row["start_time"] + int(np.argmax(t)) * dt
            match = [f for f in dev_ledgers[role]
                     if f["cohort"] == ev["key"][0]
                     and f["robot_id"] == ev["key"][1]
                     and float(f["failure_time"]) == ev["key"][2]]
            if len(match) != 1:
                raise ValueError(
                    f"support/ledger join failed for {ev['key']} in {role}")
            failure = match[0]
            loc_total += 1
            if failure["failure_time"] - E.HORIZON_S <= peak_time <= failure["failure_time"]:
                loc_hits += 1
        per_history.append({
            "history_id": role,
            "auc_pw": aucs["pw"], "auc_p": aucs["p"], "auc_w": aucs["w"],
            "auc_a": aucs["a"],
            "recall_p": entry["recall_p"], "recall_w": entry["recall_w"],
            "lead_p_median": entry["lead_p_median"],
            "lead_w_median": entry["lead_w_median"],
            "n_p": entry["n_p"], "n_w": entry["n_w"],
            "far": entry["far"], "robot_days": entry["robot_days"],
            "false_episodes": entry["false_episodes"],
            "severity_p": sev["P"], "severity_w": sev["W"],
            "categories": cats,
            "background_ratio": (stab["ratio"] if stab else None),
            "background_ci": ({"lcb": stab_ci["lcb"], "ucb": stab_ci["ucb"]}
                              if stab_ci else None),
            "localization_rate": (loc_hits / loc_total if loc_total else None),
            "n_events_pw": len(pw),
            "n_controls": len(support["controls"]),
        })
    p_vals = [h["auc_p"] for h in per_history]
    w_vals = [h["auc_w"] for h in per_history]
    pw_vals = [h["auc_pw"] for h in per_history]
    lcb_p = P.history_block_lcb(p_vals)
    lcb_w = P.history_block_lcb(w_vals)
    return {
        "branch": branch,
        "threshold": threshold,
        "canonical": canonical,
        "per_history": per_history,
        "macro_pw": canonical["macro_auc_pw"],
        "macro_p": canonical["macro_auc_p"],
        "macro_w": canonical["macro_auc_w"],
        "lcb_pw": canonical["lcb_pw"],
        "lcb_p": lcb_p["lcb"],
        "lcb_w": lcb_w["lcb"],
        "directional": canonical["directional_histories"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def git_provenance() -> dict:
    def run(*args):
        try:
            proc = subprocess.run(list(args), cwd=str(REPO_ROOT), capture_output=True,
                                  text=True, timeout=60)
            return proc.stdout.strip()
        except Exception:
            return ""
    return {"commit": run("git", "rev-parse", "HEAD"),
            "status": run("git", "status", "--porcelain"),
            "branch": run("git", "branch", "--show-current")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/generated/sprint17-ablation-v3")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(str(s) for s in MODEL_SEEDS))
    ap.add_argument("--smoke", action="store_true",
                    help="2-step CPU smoke on a tiny subset; outputs marked SMOKE, "
                         "never valid evidence")
    args = ap.parse_args()
    t0 = time.time()

    import numpy as np
    import torch

    from synth import balanced as B
    from synth import events as E
    from synth import probe15 as P
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if not args.smoke and tuple(seeds) != MODEL_SEEDS:
        raise ValueError("non-smoke runs must use exactly the frozen model seeds")
    if not args.out:
        raise ValueError("--out is required (server output directory)")
    outdir = Path(args.out)
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing to mix outputs into non-empty dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but unavailable (device_class parity)")
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    torch.backends.cudnn.benchmark = False

    data_root = Path(args.data_root)
    check_binding_digest(REPO_ROOT / "experiments" / "sprint17-role-binding-v3.json")
    metric_code_sha = canonical_hash([
        [f, sha256_file(REPO_ROOT / f)] for f in METRIC_CODE_FILES])

    run_log: list[dict] = []
    git = git_provenance()
    runtime = {"python": sys.version.split()[0], "torch": torch.__version__,
               "device": str(device), "device_name": device_name,
               "cuda_available": torch.cuda.is_available()}

    # ---- Fit pool (healthy-eligible rows only) ----
    fit_items: list[tuple[str, object]] = []
    fit_row_tags: list[str] = []
    fit_manifests: dict[str, dict] = {}
    for role, seed in FIT_ROLES:
        samples, manifest = load_verified_root(data_root, "FIT", role, seed)
        fit_manifests[role] = manifest
        by_id = {s.file_id: s for s in samples}
        for row in manifest["files"]:
            if not eligible_healthy_row(row, manifest["maintenance_windows"],
                                        B.PROGRAM_RESERVE, B.ROBOT_RESERVE):
                continue
            tag = f"{role}/{row['file_id']}"
            fit_row_tags.append(tag)
            fit_items.append((row["file_id"], by_id[row["file_id"]]))
    if not args.smoke and len(fit_items) < 200:
        raise ValueError(f"Fit healthy floor violated: {len(fit_items)}")
    fit_patches = 0
    for role, manifest in fit_manifests.items():
        idset = {t.split("/", 1)[1] for t in fit_row_tags
                 if t.startswith(role + "/")}
        fit_patches += sum(r["n_valid_patches"] for r in manifest["files"]
                           if r["file_id"] in idset)
    fit_pool = [s for _, s in fit_items]
    if args.smoke:
        fit_pool = fit_pool[:16]
        fit_items = fit_items[:16]
        fit_row_tags = fit_row_tags[:16]
        fit_patches = 0
        for role, manifest in fit_manifests.items():
            idset = {t.split("/", 1)[1] for t in fit_row_tags
                     if t.startswith(role + "/")}
            fit_patches += sum(r["n_valid_patches"] for r in manifest["files"]
                               if r["file_id"] in idset)

    # ---- Calibration rows (healthy-eligible only) ----
    cal_samples, manifest_cal = load_verified_root(data_root, "CALIBRATION", *CAL_ROLE)
    by_cal = {s.file_id: s for s in cal_samples}
    cal_items = [(r["file_id"], by_cal[r["file_id"]])
                 for r in manifest_cal["files"]
                 if eligible_healthy_row(r, manifest_cal["maintenance_windows"],
                                         B.PROGRAM_RESERVE, B.ROBOT_RESERVE)]
    if not args.smoke and len(cal_items) < 40:
        raise ValueError(f"Calibration healthy floor violated: {len(cal_items)}")
    if args.smoke:
        cal_items = cal_items[:32]

    # ---- Development files (all files; evaluation needs positives) ----
    dev_rows: dict[str, list] = {}
    dev_ledgers: dict[str, list] = {}
    dev_wins: dict[str, dict] = {}
    dev_supports: dict[str, dict] = {}
    dev_by_id: dict[str, dict] = {}
    dev_roles = [r for r, _ in DEV_ROLES]
    for role, seed in DEV_ROLES:
        samples, manifest = load_verified_root(data_root, "DEVELOPMENT", role, seed)
        by_id = {s.file_id: s for s in samples}
        dev_by_id[role] = by_id
        dev_rows[role] = manifest["files"]
        dev_ledgers[role] = E.failure_ledger(manifest)
        dev_wins[role] = manifest["maintenance_windows"]
        dev_supports[role] = evaluable_support(dev_rows[role], dev_ledgers[role],
                                               dev_wins[role])
    if args.smoke:
        for role in dev_roles:
            support = dev_supports[role]
            chosen: list[str] = []
            for ev in support["pos"]["P"][:1] + support["pos"]["W"][:1]:
                chosen += ev["members"][:6]
            if support["controls"]:
                chosen += support["controls"][0][:6]
            ops = [r["file_id"] for r in dev_rows[role]
                   if E.eligible_operational_row(r, dev_wins[role])]
            chosen += ops[::max(1, len(ops) // 12)][:12]
            keep = list(dict.fromkeys(chosen))[:32]
            keep_set = set(keep)
            dev_by_id[role] = {fid: dev_by_id[role][fid] for fid in keep}
            dev_rows[role] = [r for r in dev_rows[role]
                              if r["file_id"] in keep_set]
            dev_supports[role] = evaluable_support(
                dev_rows[role], dev_ledgers[role], dev_wins[role])
    # ---- Per-seed train / refit / calibrate / score ----
    from representation.checkpoint import save_checkpoint
    from representation.inference import NormalReferenceBank

    ckpt_dir = outdir / "checkpoints"
    cache_dir = outdir / "caches"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seed_records: dict[int, dict] = {}
    steps = 2 if args.smoke else OPTIMIZER_STEPS
    batch_size = 4 if args.smoke else 128
    for model_seed in seeds:
        st = time.time()
        model, trainer, cfg, patchifier, cfg_dict, first_loss, last = train_seed(
            model_seed, fit_pool, device, run_log, steps, batch_size)
        params = count_parameters(model)
        flops = count_flops_reference(model)
        model.eval()
        # Healthy bank refit on Fit only (identical rows every seed).
        fit_emb = embed_items(model, patchifier, cfg, fit_items, device)
        if not torch.isfinite(fit_emb).all():
            raise ValueError(f"non-finite bank embeddings: seed {model_seed}")
        bank = NormalReferenceBank(k=cfg.knn_k).fit(fit_emb)
        # Calibration thresholds (Calibration only, per branch).
        cal_pred, cal_pop, _, _ = score_items(model, bank, patchifier, cfg,
                                              cal_items, device)
        thr_pred = P.select_threshold(cal_pred)
        thr_pop = P.select_threshold(cal_pop)
        if not (np.isfinite(thr_pred) and np.isfinite(thr_pop)):
            raise ValueError(f"non-finite calibration threshold: seed {model_seed}")
        # Score Fit-healthy rows (stability denominator) + Development files.
        fit_pred, fit_pop, _, _ = score_items(model, bank, patchifier, cfg,
                                              fit_items, device)
        dev_out: dict[str, dict] = {}
        for role in dev_roles:
            items = [(r["file_id"], dev_by_id[role][r["file_id"]])
                     for r in dev_rows[role] if r["file_id"] in dev_by_id[role]]
            pred, pop, demb, ttime = score_items(model, bank, patchifier, cfg,
                                                 items, device)
            if not (np.isfinite(pred).all() and np.isfinite(pop).all()
                    and torch.isfinite(demb).all()):
                raise ValueError(f"non-finite Development outputs: {model_seed}/{role}")
            if len(set(items[i][0] for i in range(len(items)))) != len(items):
                raise ValueError(f"duplicate file ids scored: {model_seed}/{role}")
            dev_out[role] = {"ids": [fid for fid, _ in items], "S_pred": pred,
                             "S_pop": pop, "emb": demb, "t": ttime}
        # Score-branch separation (fail fast on aliased branches).
        corr = float(np.corrcoef(
            np.concatenate([dev_out[r]["S_pred"] for r in dev_roles]),
            np.concatenate([dev_out[r]["S_pop"] for r in dev_roles]))[0, 1])
        if not np.isfinite(corr) or corr >= 0.999:
            raise ValueError(f"score branches aliased: corr={corr}")
        # Persist checkpoint (step-300 final state; no selection) + caches.
        ckpt_path = ckpt_dir / f"b0_seed{model_seed}_step{trainer.step}.pt"
        save_checkpoint(ckpt_path, model, optimizer=trainer.optimizer,
                        scheduler=trainer.scheduler, config=cfg,
                        step=trainer.step, reference_bank=bank)
        ckpt_sha = sha256_file(ckpt_path)
        cache_shas: dict[str, str] = {}
        for role in dev_roles:
            d = dev_out[role]
            tmax = max((t.size for t in d["t"]), default=0)
            tpad = np.zeros((len(d["ids"]), tmax), dtype=np.float32)
            tlen = np.zeros(len(d["ids"]), dtype=np.int64)
            for i, t in enumerate(d["t"]):
                tpad[i, :t.size] = t.astype(np.float32)
                tlen[i] = t.size
            cache_path = cache_dir / f"b0_seed{model_seed}_{role}.npz"
            np.savez_compressed(
                cache_path,
                file_ids=np.array(d["ids"]),
                S_pred=d["S_pred"].astype(np.float64),
                S_pop=d["S_pop"].astype(np.float64),
                file_embedding=d["emb"].cpu().numpy().astype(np.float32),
                timestep_scores=tpad, timestep_lengths=tlen,
                provenance=np.array([f"B0 seed={model_seed} role={role} "
                                     f"step={trainer.step}"]),
            )
            cache_shas[role] = sha256_file(cache_path)
        seed_records[model_seed] = {
            "model_seed": model_seed, "steps": trainer.step,
            "first_joint": float(first_loss),
            "last_joint": float(last["joint_loss"]),
            "last_effective_lambda": float(last.get("effective_lambda", float("nan"))),
            "params": params, "flops_reference": flops,
            "flop_reference_t": FLOP_REFERENCE_T,
            "bank_rows": len(fit_items), "bank_k": cfg.knn_k,
            "bank_source": "Fit-only healthy-eligible rows "
                           f"({'+'.join(r for r, _ in FIT_ROLES)})",
            "thr_pred": float(thr_pred), "thr_pop": float(thr_pop),
            "cal_rows": len(cal_items), "fit_rows": len(fit_items),
            "fit_patches": fit_patches,
            "score_branch_corr": corr,
            "ckpt_path": str(ckpt_path), "ckpt_sha256": ckpt_sha,
            "cache_sha256": cache_shas,
            "elapsed_s": round(time.time() - st, 1),
            "dev_out": dev_out, "fit_pred": fit_pred, "fit_pop": fit_pop,
            "cfg_dict": cfg_dict,
        }

    branch_records: dict[int, dict[str, dict]] = {}
    # ---- Metrics per seed per branch ----
    for model_seed in seeds:
        rec = seed_records[model_seed]
        file_scores_pred = build_score_map(dev_roles, rec["dev_out"], "S_pred")
        file_scores_pop = build_score_map(dev_roles, rec["dev_out"], "S_pop")
        dev_times = {role: dict(zip(rec["dev_out"][role]["ids"],
                                    rec["dev_out"][role]["t"]))
                     for role in dev_roles}
        branch_records[model_seed] = {
            "S_pred": branch_metrics(
                "S_pred", rec["thr_pred"], file_scores_pred, dev_supports,
                dev_rows, dev_ledgers, dev_wins, dev_times, rec["fit_pred"],
                dev_roles),
            "S_pop": branch_metrics(
                "S_pop", rec["thr_pop"], file_scores_pop, dev_supports,
                dev_rows, dev_ledgers, dev_wins, dev_times, rec["fit_pop"],
                dev_roles),
        }

    (outdir / "run.log.json").write_text(json.dumps(to_jsonable({
        "smoke": args.smoke,
        "protocol": PROTOCOL_ID,
        "seeds": list(seeds),
        "steps": steps,
        "batch_size": batch_size,
        "git": git,
        "runtime": runtime,
        "metric_code_sha256": metric_code_sha,
        "metric_code_files": list(METRIC_CODE_FILES),
        "binding_sha256": BINDING_SHA256,
        "fit_rows": len(fit_items),
        "fit_patches": fit_patches,
        "cal_rows": len(cal_items),
        "train_trace": run_log,
        "elapsed_s": round(time.time() - t0, 1),
    }), indent=1, sort_keys=True))
    assemble_evidence(outdir, seeds, seed_records, branch_records, dev_supports,
                      dev_roles, metric_code_sha, git, runtime, t0,
                      len(fit_items), fit_patches, len(cal_items))
    if args.smoke:
        (outdir / "SMOKE.txt").write_text(
            "SMOKE RUN ONLY — 2 optimizer steps on a tiny subset; not evidence.\n")

    print(json.dumps(to_jsonable({
        "seeds": list(seed_records),
        "fit_rows": len(fit_items),
        "fit_patches": fit_patches,
        "cal_rows": len(cal_items),
        "thr": {str(s): {"S_pred": seed_records[s]["thr_pred"],
                         "S_pop": seed_records[s]["thr_pop"]} for s in seeds},
        "macro_pw_pred": {str(s): branch_records[s]["S_pred"]["macro_pw"]
                          for s in seeds},
        "macro_pw_pop": {str(s): branch_records[s]["S_pop"]["macro_pw"]
                         for s in seeds},
        "device": f"{device} ({device_name})",
        "elapsed_s": round(time.time() - t0, 1),
        "smoke": args.smoke,
    }), indent=1, sort_keys=True))
    return 0


def _mean_finite(vals: list) -> float | None:
    import numpy as np
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def assemble_evidence(outdir: Path, seeds: tuple, seed_records: dict,
                      branch_records: dict, dev_supports: dict, dev_roles: list,
                      metric_code_sha: str, git: dict, runtime: dict, t0: float,
                      fit_rows: int, fit_patches: int, cal_rows: int) -> None:
    """Write the v4-schema B0 Development document plus bounded sidecars."""
    import numpy as np
    import jsonschema

    from synth import probe15 as P
    from representation import sprint17_ablation as A

    # Task 5 harness guards over the frozen declarations.
    A.validate_arm(A.get_arm("B0"))
    A.validate_training_parity(A.TrainingParity())
    A.validate_eligibility({"qualification": QUALIFICATION, "waiver_id": WAIVER_ID,
                            "waiver_scope": WAIVER_SCOPE})
    for model_seed in seeds:
        rec = seed_records[model_seed]
        A.validate_compute_envelope(A.ComputeEnvelope(
            b0_params=rec["params"], params_total=rec["params"],
            b0_flops_per_reference_sample=rec["flops_reference"],
            flops_per_reference_sample=rec["flops_reference"]))
        A.validate_provenance({
            "data_protocol": DATA_PROTOCOL,
            "role_binding_sha256": BINDING_SHA256,
            "data_roles": [r for r, _ in FIT_ROLES] + [CAL_ROLE[0]] + dev_roles,
            "data_seeds": [2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811],
            "model_seeds": list(MODEL_SEEDS),
            "fit_root_sha256": canonical_hash(
                sorted(EXPECTED_MANIFEST_SHA256[r] for r, _ in FIT_ROLES)),
            "calibration_root_sha256": EXPECTED_MANIFEST_SHA256[CAL_ROLE[0]],
            "evaluation_root_sha256": canonical_hash(
                sorted(EXPECTED_MANIFEST_SHA256[r] for r in dev_roles)),
            "metric_code_sha256": metric_code_sha,
            "checkpoint_sha256": rec["ckpt_sha256"],
            "cache_sha256": canonical_hash(sorted(rec["cache_sha256"].values())),
            "parent_arm_ids": [],
        })

    # Aggregate across seeds (headline doc uses the S_pred branch readout;
    # the complete S_pop set is preserved in a labeled companion sidecar).
    agg: dict[str, dict] = {}
    for branch in ("S_pred", "S_pop"):
        per_history = []
        for h, role in enumerate(dev_roles):
            cells = [branch_records[s][branch]["per_history"][h] for s in seeds]
            per_history.append({
                "history_id": role,
                "auc_pw": _mean_finite([c["auc_pw"] for c in cells]),
                "auc_p": _mean_finite([c["auc_p"] for c in cells]),
                "auc_w": _mean_finite([c["auc_w"] for c in cells]),
                "auc_a": _mean_finite([c["auc_a"] for c in cells]),
                "recall_p": _mean_finite([c["recall_p"] for c in cells]),
                "recall_w": _mean_finite([c["recall_w"] for c in cells]),
                "lead_p_median": _mean_finite([c["lead_p_median"] for c in cells]),
                "lead_w_median": _mean_finite([c["lead_w_median"] for c in cells]),
                "far": _mean_finite([c["far"] for c in cells]),
                "robot_days": _mean_finite([c["robot_days"] for c in cells]),
                "severity_p": _mean_finite([c["severity_p"] for c in cells]),
                "severity_w": _mean_finite([c["severity_w"] for c in cells]),
                "background_ratio": _mean_finite(
                    [c["background_ratio"] for c in cells]),
                "localization_rate": _mean_finite(
                    [c["localization_rate"] for c in cells]),
                "n_events_pw": int(np.mean([c["n_events_pw"] for c in cells])),
                "n_controls": int(np.mean([c["n_controls"] for c in cells])),
            })
        pw = [h["auc_pw"] for h in per_history]
        pa = [h["auc_p"] for h in per_history]
        wa = [h["auc_w"] for h in per_history]
        agg[branch] = {
            "per_history": per_history,
            "macro_pw": float(np.mean(pw)),
            "macro_p": float(np.mean(pa)),
            "macro_w": float(np.mean(wa)),
            "lcb_pw": P.history_block_lcb(pw)["lcb"],
            "lcb_p": P.history_block_lcb(pa)["lcb"],
            "lcb_w": P.history_block_lcb(wa)["lcb"],
            "directional": sum(1 for v in pw if v > 0.55),
            "threshold_mean": float(np.mean(
                [seed_records[s][f"thr_{branch.split('_')[1]}"] for s in seeds])),
        }

    primary = agg["S_pred"]
    support_entries = []
    excluded_total = 0
    for h, role in enumerate(dev_roles):
        support = dev_supports[role]
        excluded_total += support["skipped"]
        support_entries.append({
            "history_id": role, "p": len(support["pos"]["P"]),
            "w": len(support["pos"]["W"]), "a": len(support["pos"]["A"]),
            "controls": len(support["controls"]),
            "robot_days": float(primary["per_history"][h]["robot_days"] or 0.0),
        })
    common_support_sha = canonical_hash({
        role: {"events": sorted(tuple(e["key"]) for e in dev_supports[role]["pos"]["P"]
                               + dev_supports[role]["pos"]["W"]
                               + dev_supports[role]["pos"]["A"]),
               "controls": sorted(tuple(sorted(w))
                                   for w in dev_supports[role]["controls"])}
        for role in dev_roles})

    rec0 = seed_records[seeds[0]]
    far_pass = all((h["far"] is not None and h["far"] <= 0.05)
                   for h in primary["per_history"])
    bg_pass = all((h["background_ratio"] is not None
                   and h["background_ratio"] <= 0.10)
                  for h in primary["per_history"])
    finite_ok = all(
        v is not None
        for h in primary["per_history"]
        for v in (h["auc_pw"], h["auc_p"], h["auc_w"]))

    doc = {
        "schema_id": SCHEMA_ID,
        "protocol_id": PROTOCOL_ID,
        "phase": "development",
        "arm_id": "B0",
        "arm_kind": "baseline",
        "component_set": [],
        "status": "VALID_NEGATIVE",
        "invalid_reasons": [],
        "provenance": {
            "data_protocol": DATA_PROTOCOL,
            "role_binding_sha256": BINDING_SHA256,
            "data_roles": ([r for r, _ in FIT_ROLES] + [CAL_ROLE[0]] + dev_roles),
            "data_seeds": [2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811],
            "model_seeds": list(MODEL_SEEDS),
            "fit_root_sha256": canonical_hash(
                sorted(EXPECTED_MANIFEST_SHA256[r] for r, _ in FIT_ROLES)),
            "calibration_root_sha256": EXPECTED_MANIFEST_SHA256[CAL_ROLE[0]],
            "evaluation_root_sha256": canonical_hash(
                sorted(EXPECTED_MANIFEST_SHA256[r] for r in dev_roles)),
            "metric_code_sha256": metric_code_sha,
            "checkpoint_sha256": canonical_hash(
                sorted(rec["ckpt_sha256"] for rec in seed_records.values())),
            "cache_sha256": canonical_hash(sorted(
                sha for rec in seed_records.values()
                for sha in rec["cache_sha256"].values())),
            "parent_arm_ids": [],
        },
        "config": {
            "one_principal_change": "none",
            "adapter_description": "no adapter; unchanged V1 architecture",
            "optimizer_steps": OPTIMIZER_STEPS,
            "checkpoint_step": CHECKPOINT_STEP,
            "calibration_quantile": CALIBRATION_QUANTILE,
            "score_branches": ["S_pred", "S_pop"],
        },
        "compute": {
            "b0_params": rec0["params"],
            "params_total": rec0["params"],
            "params_delta_fraction": 0.0,
            "b0_flops_per_reference_sample": rec0["flops_reference"],
            "flops_per_reference_sample": rec0["flops_reference"],
            "flops_delta_fraction": 0.0,
            "envelope_pass": True,
        },
        "support": {
            "common_support_sha256": common_support_sha,
            "per_history": support_entries,
            "excluded_rows": excluded_total,
        },
        "scores": {
            "S_pred": {"status": "PRESENT", "threshold": primary["threshold_mean"],
                       "provenance": canonical_hash(sorted(
                           rec["cache_sha256"][r] for rec in seed_records.values()
                           for r in dev_roles))},
            "S_pop": {"status": "PRESENT",
                      "threshold": agg["S_pop"]["threshold_mean"],
                      "provenance": canonical_hash(sorted(
                          rec["cache_sha256"][r] for rec in seed_records.values()
                          for r in dev_roles))},
        },
        "metrics": {
            "primary_PW": {
                "point": primary["macro_pw"],
                "lcb95": primary["lcb_pw"],
                "delta_vs_B0": 0.0,
                "directional_count": primary["directional"],
                "per_history": [h["auc_pw"] for h in primary["per_history"]],
            },
            "P": {
                "point": primary["macro_p"],
                "lcb95": primary["lcb_p"],
                "delta_vs_B0": 0.0,
                "recall_per_history": [h["recall_p"]
                                       for h in primary["per_history"]],
                "median_lead_days_per_history": [h["lead_p_median"]
                                                 for h in primary["per_history"]],
            },
            "W": {
                "point": primary["macro_w"],
                "lcb95": primary["lcb_w"],
                "delta_vs_B0": 0.0,
                "recall_per_history": [h["recall_w"]
                                       for h in primary["per_history"]],
                "median_lead_days_per_history": [h["lead_w_median"]
                                                 for h in primary["per_history"]],
            },
            "A_companion": {
                "point": _mean_finite([h["auc_a"]
                                       for h in primary["per_history"]]),
                "lcb95": (P.history_block_lcb(
                    [h["auc_a"] for h in primary["per_history"]
                     if h["auc_a"] is not None])["lcb"]
                    if any(h["auc_a"] is not None
                           for h in primary["per_history"]) else None),
                "per_history": [h["auc_a"] for h in primary["per_history"]],
            },
            "severity_ordering": {
                "P_spearman": [h["severity_p"] for h in primary["per_history"]],
                "W_spearman": [h["severity_w"] for h in primary["per_history"]],
            },
            "localization": {
                "present": True,
                "per_history": [h["localization_rate"]
                                for h in primary["per_history"]],
            },
            "background": {
                "stability_ratio_per_history": [h["background_ratio"]
                                                for h in primary["per_history"]],
                "far_per_robot_day": [h["far"] for h in primary["per_history"]],
                "delta_far_vs_B0": [0.0, 0.0, 0.0, 0.0],
            },
            "category_slices": {
                role: {
                    cid: _mean_finite([
                        branch_records[s]["S_pred"]["per_history"][h]
                        ["categories"][cid] for s in seeds])
                    for cid in ("P1", "P2", "W1", "W2", "A1", "A2")
                } for h, role in enumerate(dev_roles)
            },
            "bootstrap": {"replicates": 2000, "seed": 20260202,
                          "lcb_percentile": 2.5},
        },
        "gates": {
            "structural": True,
            "observable": True,
            "finite_and_support": bool(finite_ok),
            "score_separation": True,
            "compute_envelope": True,
            "minimum_effect": False,
            "P_noninferiority": True,
            "W_noninferiority": True,
            "nuisance": bool(far_pass),
            "background_stability": bool(bg_pass),
            "eligible": True,
            "recovery": False,
        },
        "eligibility": {
            "qualification": QUALIFICATION,
            "waiver_id": WAIVER_ID,
            "waiver_scope": WAIVER_SCOPE,
        },
    }
    proto_path = REPO_ROOT / "experiments" / "sprint17-ablation-protocol-v4.md"
    text = proto_path.read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(\{.*?\n\})\n```", text, re.DOTALL)
    schemas = [b for b in blocks if '"$schema"' in b]
    if len(schemas) != 1:
        raise ValueError(
            f"expected exactly one v4 JSON schema block, found {len(schemas)}")
    schema = json.loads(schemas[0])
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(to_jsonable(doc)),
                    key=lambda e: list(e.path))
    if errors:
        raise ValueError("B0 document schema errors: "
                         + "; ".join(f"{list(e.path)}: {e.message}"
                                     for e in errors[:8]))
    (outdir / "b0_development_v4.json").write_text(
        json.dumps(to_jsonable(doc), indent=1, sort_keys=True))
    # Companion sidecars (bounded; S_pop full set + per-seed records).
    slim_records = {}
    for s in seeds:
        rec = dict(seed_records[s])
        rec.pop("dev_out", None)
        rec.pop("fit_pred", None)
        rec.pop("fit_pop", None)
        slim_records[str(s)] = rec
    (outdir / "b0_spop_metrics.json").write_text(json.dumps(to_jsonable(
        {"branch": "S_pop", "aggregate": agg["S_pop"],
         "per_seed": {str(s): branch_records[s]["S_pop"] for s in seeds}}),
        indent=1, sort_keys=True))
    (outdir / "b0_per_seed_metrics.json").write_text(json.dumps(to_jsonable(
        {"S_pred_per_seed": {str(s): branch_records[s]["S_pred"] for s in seeds},
         "seed_records": slim_records}),
        indent=1, sort_keys=True))
    (outdir / "b0_run_manifest.json").write_text(json.dumps(to_jsonable({
        "protocol": PROTOCOL_ID,
        "arm": "B0",
        "seeds": list(seeds),
        "config_sha256": canonical_hash(
            {str(s): seed_records[s]["cfg_dict"] for s in seeds}),
        "git": git,
        "runtime": runtime,
        "metric_code_sha256": metric_code_sha,
        "common_support_sha256": common_support_sha,
        "fit_rows": fit_rows,
        "fit_patches": fit_patches,
        "cal_rows": cal_rows,
        "elapsed_s_total": round(time.time() - t0, 1),
    }), indent=1, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
