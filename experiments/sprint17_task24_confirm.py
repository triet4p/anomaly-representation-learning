"""Sprint 17 Task 24 - one-shot Confirmation scoring for B0 + 14 singles.

Scoring-only driver under the exact Task 23 lock (no training, no refit,
no recalibration, no selection, no pooling, no rerun). For each of the 15
registered arms it loads the frozen step-300 checkpoint, restores the
frozen Fit reference bank, reuses the frozen lock q95 thresholds, scores
the four untouched Confirmation histories with the frozen metric code on
common Confirmation support, and writes a schema-validated confirmation
document plus bounded sidecars. C7/C8/C9 re-materialize their Fit-only
reference/standardizer deterministically from frozen Fit forwards (pure
functions of frozen inputs; no Confirmation data, no tuning) and reuse
the bitwise-B0 branch exactly as on Development.

B0 runs first and alone: if its frozen reproducibility/eligibility
contract fails, no B0 confirmation summary exists and every single-arm
invocation (which requires --b0-confirm) refuses mechanically.

Refusals: K1-K9, DEVELOPMENT/CALIBRATION/SEALED groups, non-lock seeds,
recalibration, retraining. Labels and simulator state are post-hoc
diagnostics only; S_pred/S_pop stay independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

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
CONF_ROLES = (("H-S17-V3-CONF-01", 2812), ("H-S17-V3-CONF-02", 2813),
              ("H-S17-V3-CONF-03", 2814), ("H-S17-V3-CONF-04", 2815))
EXPECTED_MANIFEST_SHA256 = {
    "H-S17-V3-FIT-01": "46afe203c29a211a3ca9c8b21291fcaf1271d325c0ec7b6849b1d6fc5e11db13",
    "H-S17-V3-FIT-02": "136476984c0d1ee69e2aad01f6f1efc91750d8e98dd9d872b09c80d6a2a96669",
    "H-S17-V3-FIT-03": "07a223ca5714943a3973b01f4d851c639e54f9d2109cfdc8ceaa952c2637d305",
    "H-S17-V3-CONF-01": "141d79d3e30636fc7b04123f32a74111f1118ab879f2ac913cb7f5182db5b919",
    "H-S17-V3-CONF-02": "b8cc1f74dbf30b052849f1dad47a1a75c2506e070565ed000a3a18bd5d13da85",
    "H-S17-V3-CONF-03": "8826fb4d4b9a8b9d25bed56c4f8d818cc9e82edcb2ca52a910c24437d0bfa7e7",
    "H-S17-V3-CONF-04": "f5042b59c34997aa4effc9974eb92b30c79f4d9a0a5a8c9a84403ff437bfe452",
}
METRIC_CODE_FILES = (
    "src/synth/probe15.py",
    "src/synth/events.py",
    "src/representation/attribution_metrics.py",
    "src/representation/sprint17_ablation.py",
    "experiments/sprint17_task24_confirm.py",
)
B0_PARAMS = 1821698
B0_FLOPS_REFERENCE = 182016709
FIT_ROWS = 5040
FIT_PATCHES = 154129
CONFIRM_ARMS = ("B0", "C2-A", "C2-B", "C3-A", "C3-B", "C5-A", "C5-B",
                "C6-A", "C6-B", "C7-A", "C7-B", "C8-A", "C8-B", "C9-A", "C9-B")
TRAINABLE_ARMS = ("B0", "C2-A", "C2-B", "C3-A", "C3-B", "C5-A", "C5-B", "C6-A", "C6-B")
TRAIN_FREE_ARMS = ("C7-A", "C7-B", "C8-A", "C8-B", "C9-A", "C9-B")
ARM_FILE_PREFIX = {"B0": "b0", "C2-A": "c2a", "C2-B": "c2b", "C3-A": "c3a",
                   "C3-B": "c3b", "C5-A": "c5a", "C5-B": "c5b", "C6-A": "c6a",
                   "C6-B": "c6b", "C7-A": "c7a", "C7-B": "c7b", "C8-A": "c8a",
                   "C8-B": "c8b", "C9-A": "c9a", "C9-B": "c9b"}
ARM_COMPONENTS = {"B0": (), "C2-A": ("C2",), "C2-B": ("C2",), "C3-A": ("C3",),
                  "C3-B": ("C3",), "C5-A": ("C5",), "C5-B": ("C5",), "C6-A": ("C6",),
                  "C6-B": ("C6",), "C7-A": ("C7",), "C7-B": ("C7",), "C8-A": ("C8",),
                  "C8-B": ("C8",), "C9-A": ("C9",), "C9-B": ("C9",)}
TASK23_LOCK_PATH = "experiments/sprint17-task23-confirmation-lock.json"
TASK23_LOCK_SHA256 = "14fab303d78725af061aaa639a558f84b262c76d5b918042618a109d61afd039"
FIDELITY_ATOL = 1e-3
FLOP_REFERENCE_T = 512



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

    Method: forward hooks. Linear/Conv1d count 2 FLOPs per true MAC
    (grouped/depthwise convolutions divide by ``groups``, since one output
    element combines ``in_channels/groups`` times ``kernel_size`` products);
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
        groups = int(getattr(mod, "groups", 1))
        total[0] += (2 * int(out.numel()) * int(mod.in_channels)
                     * int(mod.kernel_size[0]) // groups)

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
        # C6 keeps B0 patchification and encoding: the reference batch uses the
        # B0 lattice, so the count isolates the file-pooling FLOP delta.
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


def score_items(model, bank, patchifier, cfg, items: list[tuple[str, object]],
                device, masking_policy=None):
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
                                           masking_seed=file_seed(fid),
                                           masking_policy=masking_policy)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            res = inference.score_batch(batch)
            pred.append(float(torch.as_tensor(res["S_pred"]).reshape(-1)[0].item()))
            pop.append(float(torch.as_tensor(res["S_pop"]).reshape(-1)[0].item()))
            emb.append(res["file_embedding"][0].detach().cpu())
            times.append(np.asarray(res["timestep_scores"][0], dtype=np.float64))
    return np.array(pred), np.array(pop), torch.stack(emb), times


def embed_items(model, patchifier, cfg, items: list[tuple[str, object]], device,
                masking_policy=None):
    """File embeddings for bank fitting (same deterministic masks as scoring)."""
    import torch
    from representation.data import collate_variable_files

    outs = []
    with torch.no_grad():
        for fid, sample in items:
            batch = collate_variable_files([sample], patchifier,
                                           masking_config=cfg,
                                           masking_seed=file_seed(fid),
                                           masking_policy=masking_policy)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            outs.append(model(batch)["file_embedding"][0].detach().cpu())
    return torch.stack(outs)


# ---------------------------------------------------------------------------
# Train-free helpers (K5/K6: hash-verified B0 reload + combined rescoring)
# ---------------------------------------------------------------------------


def condition_lists(items: list[tuple[str, object]]) -> tuple[list[int], list[int]]:
    """Model-input (robot_idx, program_idx) conditions for C7 references."""
    robots, programs = [], []
    for _, sample in items:
        robots.append(int(sample.robot_idx))
        programs.append(int(sample.program_idx))
    return robots, programs


def latent_items(model, patchifier, cfg, items: list[tuple[str, object]], device,
                 masking_policy=None):
    """Frozen forward latents per file (predicted/target/mask/embedding/map)."""
    import numpy as np
    import torch
    from representation.data import collate_variable_files

    model.eval()
    out = []
    with torch.no_grad():
        for fid, sample in items:
            batch = collate_variable_files([sample], patchifier,
                                           masking_config=cfg,
                                           masking_seed=file_seed(fid),
                                           masking_policy=masking_policy)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            res = model(batch)
            row = {
                "file_id": fid,
                "predicted": res["predicted_latents"][0].detach().cpu(),
                "target": res["target_latents"][0].detach().cpu(),
                "prediction_mask": res["prediction_mask"][0].detach().cpu(),
                "file_embedding": res["file_embedding"][0].detach().cpu(),
                "starts": batch["starts"][0].detach().cpu().numpy(),
                "valid_len": batch["valid_len"][0].detach().cpu().numpy(),
                "file_len": int(batch["file_valid_mask"][0].sum().item()),
            }
            if not (torch.isfinite(row["predicted"]).all()
                    and torch.isfinite(row["target"]).all()
                    and torch.isfinite(row["file_embedding"]).all()):
                raise ValueError(f"non-finite forward latents: {fid}")
            out.append(row)
    return out


def b0_file_scores(lat_list: list[dict]) -> tuple:
    """B0-MSE file scores + per-patch energies (fidelity oracle)."""
    import numpy as np
    import torch
    from representation.sprint17_c8 import b0_mse_energy, file_mean

    scores, patch = [], []
    for row in lat_list:
        pred = row["predicted"].unsqueeze(0)
        targ = row["target"].unsqueeze(0)
        mask = row["prediction_mask"].unsqueeze(0)
        energies = b0_mse_energy(pred, targ, mask)
        scores.append(float(file_mean(energies, mask)[0].item()))
        patch.append(energies[0].detach().cpu().numpy().astype(np.float64))
    return np.array(scores, dtype=np.float64), patch


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


def _far_delta_ok(d_pred: dict) -> bool:
    """Paired Delta FAR <= +0.02 on every history (False when unavailable)."""
    vals = d_pred.get("per_history_far") or []
    if len(vals) != 4 or any(v is None for v in vals):
        return False
    import numpy as np
    return bool(all(float(v) <= 0.02 for v in vals))


def _mean_finite(vals: list) -> float | None:
    import numpy as np
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None



def load_verified_root(data_root: Path, group: str, role: str,
                       seed: int) -> tuple[list, dict]:
    """Load one chronological root with fail-fast role/manifest checks.

    Task 24 allows exactly FIT (stability denominator + Fit-only
    reference re-materialization) and CONFIRMATION (one-shot scoring).
    DEVELOPMENT, CALIBRATION, and SEALED groups are refused: no
    Development reuse, no recalibration, no sealed contact.
    """
    from synth import balanced as B
    from synth.chronicle import load_chronological

    if group in ("DEVELOPMENT", "CALIBRATION", "SEALED"):
        raise ValueError(f"refusing {group} access in Task 24")
    if group not in ("FIT", "CONFIRMATION"):
        raise ValueError(f"unknown group in Task 24: {group}")
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


def check_task23_lock() -> dict:
    """Refuse scoring unless the committed Task 23 lock bytes are exact.

    The digest is over CRLF-normalized bytes so Windows (CRLF) and Linux
    (LF) checkouts of the identical blob gate identically. The loaded
    lock is fully validated (24 arms, members, thresholds, roots,
    waiver, gates, flags) before any score is computed.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from representation.sprint17_confirmation_lock import validate_lock

    data = (REPO_ROOT / TASK23_LOCK_PATH).read_bytes().replace(b"\r\n", b"\n")
    payload = json.loads(data.decode("utf-8"))
    digest = payload.pop("lock_sha256", None)
    if canonical_hash(payload) != TASK23_LOCK_SHA256 or digest != TASK23_LOCK_SHA256:
        raise ValueError("Task 23 lock mismatch (no outcome-driven change)")
    payload["lock_sha256"] = digest
    if validate_lock(payload) is not True:
        raise ValueError("Task 23 lock validation failed")
    if payload["confirmation_scoring_started"] is not False:
        raise ValueError("lock flags already consumed")
    return payload


def lock_thresholds(lock: dict, arm_id: str) -> dict[str, dict[str, float]]:
    """Frozen lock q95 thresholds per seed (never recomputed, no Calibration)."""
    if arm_id == "B0":
        src = lock["arms"]["B0"]["thresholds"]
        return {s: {"pred": v["pred"], "pop": v["pop"]} for s, v in src.items()}
    rec = lock["arms"][arm_id]["seeds"]
    if "reuses_b0_checkpoints" in rec:
        b0pop = lock["arms"]["B0"]["thresholds"]
        return {s: {"pred": lock["arms"][arm_id]["thresholds_pred"][s],
                    "pop": b0pop[s]["pop"]} for s in ("171701", "171702", "171703")}
    return {s: {"pred": v["thr_pred"], "pop": v["thr_pop"]} for s, v in rec.items()}


def masking_policy_for(arm_id: str):
    """Collate masking policy for an arm (None = frozen B0 composition)."""
    if arm_id == "C5-A":
        from representation.sprint17_c5 import c5a_block_policy
        return c5a_block_policy
    return None


def arm_config_dict(arm_id: str, model_seed: int) -> dict:
    """Frozen per-arm configuration: B0 values plus the declared change.

    Each branch repeats exactly the frozen delta its single-arm driver
    records (C2 stride/grids/weights; C3 encoder; C5 masking/criterion;
    C6 pooling; C7 reference; C8 scorer; C9 aggregation). No training,
    optimizer, schedule, or budget key differs by construction.
    """
    from representation.sprint17_c2 import ARM_ADAPTER_DESCRIPTION as C2_DESC
    from representation.sprint17_c2 import arm_grids
    from representation.sprint17_c3 import ARM_ADAPTER_DESCRIPTION as C3_DESC
    from representation.sprint17_c3 import ARM_LOCAL_ENCODER_PARAMS, arm_local_encoder
    from representation.sprint17_c5 import ARM_ADAPTER_DESCRIPTION as C5_DESC
    from representation.sprint17_c5 import C5A_HORIZONS
    from representation.sprint17_c6 import ARM_ADAPTER_DESCRIPTION as C6_DESC
    from representation.sprint17_c6 import ARM_POOLING_PARAMS, arm_pooling
    from representation.sprint17_c7 import ARM_ADAPTER_DESCRIPTION as C7_DESC
    from representation.sprint17_c7 import arm_reference
    from representation.sprint17_c8 import ARM_ADAPTER_DESCRIPTION as C8_DESC
    from representation.sprint17_c9 import ARM_ADAPTER_DESCRIPTION as C9_DESC
    if arm_id not in CONFIRM_ARMS:
        raise ValueError(f"Task 24 runs only {CONFIRM_ARMS}, got {arm_id!r}")
    cfg = b0_config_dict(model_seed)
    cfg["arm_id"] = arm_id
    if arm_id == "B0":
        cfg["adapter_description"] = "no adapter; unchanged V1 architecture"
        return cfg
    comp = arm_id.split("-")[0]
    if comp == "C2":
        cfg["stride"] = {"C2-A": 16, "C2-B": 32}[arm_id]
        cfg["c2_grids"] = [list(g) for g in arm_grids(arm_id)]
        cfg["support_weights"] = ("unit-mass two-grid support weights in file "
                                  "pooling" if arm_id == "C2-B" else "none "
                                  "(plain-mean pooling, B0 path unchanged)")
        cfg["adapter_description"] = C2_DESC[arm_id]
    elif comp == "C3":
        cfg["local_encoder"] = type(arm_local_encoder(
            arm_id, cfg["n_channels"], cfg["d_model"],
            cfg["dropout"])).__name__
        cfg["local_encoder_params"] = ARM_LOCAL_ENCODER_PARAMS[arm_id]
        cfg["adapter_description"] = C3_DESC[arm_id]
    elif comp == "C5":
        cfg["masking_policy"] = ("channel_time_block" if arm_id == "C5-A"
                                 else "b0-composition")
        cfg["criterion"] = ("multi-horizon-ema" if arm_id == "C5-A"
                            else "vicreg-file-level")
        cfg["prediction_horizons"] = list(C5A_HORIZONS) if arm_id == "C5-A" else [0]
        cfg["adapter_description"] = C5_DESC[arm_id]
    elif comp == "C6":
        cfg["pooling"] = type(arm_pooling(
            arm_id, cfg["d_model"])).__name__
        cfg["pooling_params"] = ARM_POOLING_PARAMS[arm_id]
        cfg["adapter_description"] = C6_DESC[arm_id]
    elif comp == "C7":
        cfg["reference"] = type(arm_reference(arm_id)).__name__
        cfg["adapter_description"] = C7_DESC[arm_id]
    elif comp == "C8":
        cfg["scorer"] = {"C8-A": "HuberStandardizer+huber-mean",
                         "C8-B": "cosine-distance-mean"}[arm_id]
        cfg["adapter_description"] = C8_DESC[arm_id]
    elif comp == "C9":
        cfg["aggregation"] = {"C9-A": "top-k-mean-fraction-0.25",
                              "C9-B": "contiguous-window-mean-duration-64"}[arm_id]
        cfg["adapter_description"] = C9_DESC[arm_id]
    else:
        raise ValueError(f"unknown component for {arm_id!r}")
    return cfg



def build_model(cfg_dict: dict, device):
    """Build the frozen arm graph (no training state; scoring only).

    C2-A/B arm patchifier (geometry-checked); C3-A/B residual/attention
    local encoder (parameter-checked); C6-A/B pooling model
    (parameter-checked); B0 graph otherwise (B0, C5-A/B, C7-A/B, C8-A/B,
    C9-A/B - the latter reuse frozen B0 states for scoring).
    """
    import torch
    from representation.config import V1Config
    from representation.model import V1RepresentationModel
    from representation.sprint17_c2 import arm_patchifier
    from representation.sprint17_c3 import (
        ARM_LOCAL_ENCODER_PARAMS, arm_local_encoder,
        count_local_encoder_params)
    from representation.sprint17_c6 import (
        ARM_POOLING_PARAMS, C6ArmModel, arm_pooling, count_pooling_params)
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    arm_id = cfg_dict["arm_id"]
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
    if arm_id in ("C2-A", "C2-B"):
        patchifier = arm_patchifier(arm_id)
        if (patchifier.cfg.patch_size != cfg.patch_size
                or patchifier.cfg.stride != cfg.stride):
            raise ValueError(f"{arm_id} patchifier geometry incompatible")
    else:
        patchifier = Patchifier(PatchConfig(patch_size=cfg.patch_size,
                                            stride=cfg.stride,
                                            pad_end=cfg_dict["pad_end"]))
    patch_encoder = None
    if arm_id in ("C3-A", "C3-B"):
        patch_encoder = arm_local_encoder(
            arm_id, cfg.n_channels, cfg.d_model, cfg.dropout)
        if count_local_encoder_params(patch_encoder) != ARM_LOCAL_ENCODER_PARAMS[arm_id]:
            raise ValueError(f"{arm_id} local-encoder parameter identity mismatch")
    if arm_id in ("C6-A", "C6-B"):
        pooling = arm_pooling(arm_id, cfg.d_model)
        if count_pooling_params(pooling) != ARM_POOLING_PARAMS[arm_id]:
            raise ValueError(f"{arm_id} pooling parameter identity mismatch")
        model = C6ArmModel(cfg, arm_id=arm_id, pooling=pooling,
                           patchifier=patchifier)
    else:
        if patch_encoder is None:
            model = V1RepresentationModel(cfg, patchifier=patchifier)
        else:
            model = V1RepresentationModel(cfg, patchifier=patchifier,
                                          patch_encoder=patch_encoder)
    model.to(device)
    return model, cfg, patchifier


def load_arm_checkpoint(lock: dict, arm_id: str, model_seed: int, ckpt_dir, device):
    """Load the frozen lock-verified step-300 state (zero optimizer steps).

    The checkpoint file hash must equal the Task 23 lock value for this
    arm and seed (trainable arms) or the B0 lock value (train-free arms).
    The restored bank must hold exactly the frozen Fit row count.
    """
    import torch
    from representation.checkpoint import load_checkpoint
    from representation.inference import NormalReferenceBank

    if arm_id == "B0":
        want = lock["arms"]["B0"]["checkpoints"][str(model_seed)]
        prefix = ARM_FILE_PREFIX[arm_id]
    elif arm_id in TRAINABLE_ARMS:
        want = lock["arms"][arm_id]["seeds"][str(model_seed)]["ckpt_sha256"]
        prefix = ARM_FILE_PREFIX[arm_id]
    else:
        want = lock["arms"]["B0"]["checkpoints"][str(model_seed)]
        prefix = "b0"
    cfg_dict = arm_config_dict(arm_id, model_seed)
    model, cfg, patchifier = build_model(cfg_dict, device)
    ckpt_path = Path(ckpt_dir) / f"{prefix}_seed{model_seed}_step300.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing frozen checkpoint (no retraining): {ckpt_path}")
    if sha256_file(ckpt_path) != want:
        raise ValueError(f"checkpoint hash mismatch for {arm_id} seed {model_seed} (no substitution)")
    bank = NormalReferenceBank(k=cfg.knn_k)
    info = load_checkpoint(ckpt_path, model, expected_config=cfg,
                           reference_bank=bank)
    if info["step"] != CHECKPOINT_STEP:
        raise ValueError(f"checkpoint step is {info['step']}, not 300")
    if not info["has_reference_bank"] or bank.embeddings is None:
        raise ValueError("checkpoint carries no reference bank")
    if bank.embeddings.shape[0] != FIT_ROWS:
        raise ValueError(f"restored bank rows {bank.embeddings.shape[0]} != frozen {FIT_ROWS}")
    model.eval()
    return model, bank, cfg, patchifier, cfg_dict, ckpt_path


def load_conf_cache(cache_dir, model_seed: int, role: str) -> dict:
    """Load one B0 Confirmation cache (bitwise branch oracle for C7/C8/C9).

    Caches are created by the B0 Confirmation run; no pre-pinned hash
    exists. Integrity is proven by coverage plus bitwise branch identity
    asserted by the caller (fail closed above FIDELITY_ATOL).
    """
    import numpy as np

    path = Path(cache_dir) / f"b0_seed{model_seed}_{role}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing B0 Confirmation cache (B0 must run first): {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def c8_file_scores(lat_list: list[dict], arm_id: str,
                   standardizer=None) -> tuple:
    """C8 file scores + per-patch energies via the frozen scorer module.

    C8-A standardizes residuals with the Fit-fitted standardizer; C8-B
    (cosine distance) fits nothing and takes standardizer=None.
    """
    import numpy as np
    import torch
    from representation.sprint17_c8 import arm_patch_energies, file_mean

    if arm_id not in ("C8-A", "C8-B"):
        raise ValueError(f"C8 scoring runs only C8-A/C8-B, got {arm_id!r}")
    scores, patch = [], []
    for row in lat_list:
        pred = row["predicted"].unsqueeze(0)
        targ = row["target"].unsqueeze(0)
        mask = row["prediction_mask"]
        if mask.dtype is not torch.bool:
            raise ValueError("prediction mask must be bool")
        mask = mask.unsqueeze(0)
        energies = arm_patch_energies(arm_id, pred, targ, mask, standardizer)
        scores.append(float(file_mean(energies, mask)[0].item()))
        patch.append(energies[0].detach().cpu().numpy().astype(np.float64))
    return np.array(scores, dtype=np.float64), patch


def c9_file_scores(lat_list: list[dict], arm_id: str) -> tuple:
    """C9 file scores over the identical valid B0 patch scores.

    Per-file B0 per-patch MSE energies come from the frozen forward
    latents; the arm aggregation (top-k / contiguous-window) maps them
    to file scores. Returns (scores, b0_patch_energies, provenance).
    """
    import numpy as np
    import torch
    from representation.sprint17_c9 import arm_file_scores, b0_patch_energies

    if arm_id not in ("C9-A", "C9-B"):
        raise ValueError(f"C9 scoring runs only C9-A/C9-B, got {arm_id!r}")
    scores, patch, prov = [], [], []
    for row in lat_list:
        pred = row["predicted"].unsqueeze(0)
        targ = row["target"].unsqueeze(0)
        mask = row["prediction_mask"]
        if mask.dtype is not torch.bool:
            raise ValueError("prediction mask must be bool")
        mask = mask.unsqueeze(0)
        energies = b0_patch_energies(pred, targ, mask)
        e = energies[0].detach().cpu().numpy().astype(np.float64)
        m = mask[0].detach().cpu().numpy().astype(bool)
        score, info = arm_file_scores(
            arm_id, e, m, row["starts"], row["valid_len"])
        if not np.isfinite(score):
            raise ValueError("non-finite C9 file score")
        scores.append(float(score))
        patch.append(e)
        prov.append(info)
    return np.array(scores, dtype=np.float64), patch, prov



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(CONFIRM_ARMS),
                    help="frozen arm to score once on Confirmation (one per invocation)")
    ap.add_argument("--ckpt-dir", default="",
                    help="dir holding the frozen step-300 checkpoints (own for trainable arms, B0 for C7/C8/C9)")
    ap.add_argument("--b0-confirm", default="",
                    help="B0 confirmation per-seed metrics JSON for paired deltas (required for singles, refused for B0)")
    ap.add_argument("--b0-caches", default="",
                    help="B0 Confirmation caches dir (required for C7/C8/C9 bitwise branch reuse)")
    ap.add_argument("--data-root", default="data/generated/sprint17-ablation-v3")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(str(s) for s in MODEL_SEEDS))
    ap.add_argument("--smoke", action="store_true",
                    help="CPU smoke on a tiny subset; outputs marked SMOKE, never valid evidence")
    args = ap.parse_args()
    t0 = time.time()

    import numpy as np
    import torch

    from synth import balanced as B
    from synth import events as E
    from synth import probe15 as P
    arm_id = args.arm
    prefix = ARM_FILE_PREFIX[arm_id]
    train_free = arm_id in TRAIN_FREE_ARMS
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if not args.smoke and tuple(seeds) != MODEL_SEEDS:
        raise ValueError("non-smoke runs must use exactly the frozen model seeds")
    if not args.out:
        raise ValueError("--out is required (server output directory)")
    outdir = Path(args.out)
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing to mix outputs into non-empty dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.ckpt_dir:
        raise ValueError("--ckpt-dir is required (frozen checkpoints, no retraining)")
    ckpt_dir_arg = Path(args.ckpt_dir)
    if arm_id == "B0" and args.b0_confirm:
        raise ValueError("B0 takes no --b0-confirm (it defines the baseline)")
    if arm_id != "B0" and args.b0_confirm == "" and not args.smoke:
        raise ValueError("--b0-confirm is required for non-smoke single runs (B0 gate)")
    if train_free and not args.b0_caches:
        raise ValueError("--b0-caches is required for C7/C8/C9 (bitwise branch reuse)")
    b0_cache_dir = Path(args.b0_caches) if args.b0_caches else None
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but unavailable (device_class parity)")
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    torch.backends.cudnn.benchmark = False

    data_root = Path(args.data_root)
    check_binding_digest(REPO_ROOT / "experiments" / "sprint17-role-binding-v3.json")
    lock = check_task23_lock()
    thresholds = lock_thresholds(lock, arm_id)
    for s in seeds:
        th = thresholds[str(s)]
        if not (np.isfinite(th["pred"]) and np.isfinite(th["pop"])):
            raise ValueError(f"non-finite lock threshold: seed {s}")
    metric_code_sha = canonical_hash([
        [f, sha256_file(REPO_ROOT / f)] for f in METRIC_CODE_FILES])

    b0_pred = None
    b0_pop = None
    b0_support = None
    if args.b0_confirm:
        given = Path(args.b0_confirm)
        pred_path = given / "b0_per_seed_metrics.json" if given.is_dir() else given
        pop_path = pred_path.parent / "b0_spop_metrics.json"
        b0_pred = json.loads(pred_path.read_text(encoding="utf-8"))
        b0_pop = json.loads(pop_path.read_text(encoding="utf-8"))
        if b0_pred.get("phase") != "confirmation" or b0_pred.get("arm_id") != "B0":
            raise ValueError("B0 gate file is not a B0 confirmation record (stop)")
        if b0_pred.get("status") not in ("VALID_NEGATIVE",):
            raise ValueError("B0 confirmation not eligible (stop alternatives)")
        for seed_int in MODEL_SEEDS:
            key = str(seed_int)
            for store_name, store in (("pred", b0_pred), ("pop", b0_pop)):
                cell = (store["S_pred_per_seed"] if store_name == "pred" else store["per_seed"])[key]
                if not np.isfinite(cell["macro_pw"]):
                    raise ValueError(f"B0 confirmation macro non-finite (stop): {key}")
        b0_support = b0_pred.get("confirmation_support_sha256")
        if not b0_support:
            raise ValueError("B0 confirmation support missing (stop)")

    run_log: list[dict] = []
    git = git_provenance()
    runtime = {"python": sys.version.split()[0], "torch": torch.__version__,
               "device": str(device), "device_name": device_name,
               "cuda_available": torch.cuda.is_available()}

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
    if not args.smoke and len(fit_items) != FIT_ROWS:
        raise ValueError(f"Fit healthy count drift: {len(fit_items)} != frozen {FIT_ROWS}")
    if len(fit_items) < 200:
        raise ValueError(f"Fit healthy floor violated: {len(fit_items)}")
    fit_patches = 0
    for role, manifest in fit_manifests.items():
        idset = {t.split("/", 1)[1] for t in fit_row_tags
                 if t.startswith(role + "/")}
        fit_patches += sum(r["n_valid_patches"] for r in manifest["files"]
                           if r["file_id"] in idset)
    if not args.smoke and fit_patches != FIT_PATCHES:
        raise ValueError("Fit patch count drift")
    if arm_id in ("C2-A", "C2-B"):
        from representation.sprint17_c2 import arm_patchifier as _arm_patchifier
        _patchifier = _arm_patchifier(arm_id)
    else:
        from synth.config import PatchConfig as _PatchConfig
        from synth.patchify import Patchifier as _Patchifier
        _patchifier = _Patchifier(_PatchConfig(patch_size=32, stride=16, pad_end=True))
    arm_fit_patches = sum(_patchifier.patchify(s).N for _, s in fit_items)
    if not args.smoke:
        want_patches = lock["arms"][arm_id]["arm_fit_patches"] if arm_id != "B0" else FIT_PATCHES
        if arm_fit_patches != want_patches:
            raise ValueError(f"{arm_id} lattice breach: {arm_fit_patches} != locked {want_patches}")
    if args.smoke:
        fit_items = fit_items[:16]
        fit_row_tags = fit_row_tags[:16]

    conf_rows: dict[str, list] = {}
    conf_ledgers: dict[str, list] = {}
    conf_wins: dict[str, dict] = {}
    conf_supports: dict[str, dict] = {}
    conf_by_id: dict[str, dict] = {}
    conf_roles = [r for r, _ in CONF_ROLES]
    for role, seed in CONF_ROLES:
        samples, manifest = load_verified_root(data_root, "CONFIRMATION", role, seed)
        by_id = {s.file_id: s for s in samples}
        conf_by_id[role] = by_id
        conf_rows[role] = manifest["files"]
        conf_ledgers[role] = E.failure_ledger(manifest)
        conf_wins[role] = manifest["maintenance_windows"]
        conf_supports[role] = evaluable_support(conf_rows[role], conf_ledgers[role], conf_wins[role])
    if args.smoke:
        for role in conf_roles:
            support = conf_supports[role]
            chosen: list[str] = []
            for ev in support["pos"]["P"][:1] + support["pos"]["W"][:1]:
                chosen += ev["members"][:6]
            if support["controls"]:
                chosen += support["controls"][0][:6]
            ops = [r["file_id"] for r in conf_rows[role]
                   if E.eligible_operational_row(r, conf_wins[role])]
            chosen += ops[::max(1, len(ops) // 12)][:12]
            keep = list(dict.fromkeys(chosen))[:32]
            keep_set = set(keep)
            conf_by_id[role] = {fid: conf_by_id[role][fid] for fid in keep}
            conf_rows[role] = [r for r in conf_rows[role] if r["file_id"] in keep_set]
            conf_supports[role] = evaluable_support(conf_rows[role], conf_ledgers[role], conf_wins[role])


    from representation.inference import NormalReferenceBank

    cache_dir = outdir / "caches"
    cache_dir.mkdir(parents=True, exist_ok=True)

    seed_records: dict[int, dict] = {}
    masking_policy = masking_policy_for(arm_id)
    for model_seed in seeds:
        st = time.time()
        skey = str(model_seed)
        thr_pred = float(thresholds[skey]["pred"])
        thr_pop = float(thresholds[skey]["pop"])
        model, bank, cfg, patchifier, cfg_dict, ckpt_path = load_arm_checkpoint(
            lock, arm_id, model_seed, ckpt_dir_arg, device)
        params = count_parameters(model)
        flops = count_flops_reference(model)
        if params <= 0 or flops <= 0:
            raise ValueError(f"non-positive compute counts: seed {model_seed}")
        model.eval()
        if arm_id in TRAINABLE_ARMS:
            fit_pred, fit_pop, _, _ = score_items(model, bank, patchifier, cfg,
                                                      fit_items, device, masking_policy)
            conf_out: dict[str, dict] = {}
            for role in conf_roles:
                items = [(r["file_id"], conf_by_id[role][r["file_id"]])
                         for r in conf_rows[role] if r["file_id"] in conf_by_id[role]]
                pred, pop, demb, ttime = score_items(model, bank, patchifier, cfg,
                                                         items, device, masking_policy)
                if not (np.isfinite(pred).all() and np.isfinite(pop).all()
                        and torch.isfinite(demb).all()):
                    raise ValueError(f"non-finite Confirmation outputs: {model_seed}/{role}")
                if len(set(items[i][0] for i in range(len(items)))) != len(items):
                    raise ValueError(f"duplicate file ids scored: {model_seed}/{role}")
                conf_out[role] = {"ids": [fid for fid, _ in items], "S_pred": pred,
                                  "S_pop": pop, "emb": demb, "t": ttime}
            corr = float(np.corrcoef(
                np.concatenate([conf_out[r]["S_pred"] for r in conf_roles]),
                np.concatenate([conf_out[r]["S_pop"] for r in conf_roles]))[0, 1])
            if not np.isfinite(corr) or corr >= 0.999:
                raise ValueError(f"score branches aliased: corr={corr}")
            cache_shas: dict[str, str] = {}
            for role in conf_roles:
                d = conf_out[role]
                tmax = max((t.size for t in d["t"]), default=0)
                tpad = np.zeros((len(d["ids"]), tmax), dtype=np.float32)
                tlen = np.zeros(len(d["ids"]), dtype=np.int64)
                for i, t in enumerate(d["t"]):
                    tpad[i, :t.size] = t.astype(np.float32)
                    tlen[i] = t.size
                cache_path = cache_dir / f"{prefix}_seed{model_seed}_{role}.npz"
                np.savez_compressed(cache_path, file_ids=np.array(d["ids"]),
                    S_pred=d["S_pred"].astype(np.float64), S_pop=d["S_pop"].astype(np.float64),
                    file_embedding=d["emb"].cpu().numpy().astype(np.float32),
                    timestep_scores=tpad, timestep_lengths=tlen,
                    provenance=np.array([f"{arm_id} seed={model_seed} role={role} step=300 confirmation"]))
                cache_shas[role] = sha256_file(cache_path)
            seed_records[model_seed] = {"model_seed": model_seed, "steps": CHECKPOINT_STEP,
                "optimizer_steps_executed": 0, "params": params, "flops_reference": flops,
                "flop_reference_t": FLOP_REFERENCE_T, "bank_rows": FIT_ROWS,
                "bank_source": "checkpoint-restored frozen Fit-only bank",
                "thr_pred": thr_pred, "thr_pop": thr_pop, "threshold_source": "task23-lock",
                "fit_rows": len(fit_items), "fit_patches": fit_patches,
                "score_branch_corr": corr, "ckpt_sha256": sha256_file(ckpt_path),
                "cache_sha256": cache_shas, "elapsed_s": round(time.time() - st, 1),
                "conf_out": conf_out, "fit_pred": fit_pred, "fit_pop": fit_pop,
                "cfg_dict": cfg_dict}
            continue
        fit_lat = latent_items(model, patchifier, cfg, fit_items, device, masking_policy)
        conf_lat: dict[str, list] = {}
        for role in conf_roles:
            items = [(r["file_id"], conf_by_id[role][r["file_id"]])
                     for r in conf_rows[role] if r["file_id"] in conf_by_id[role]]
            conf_lat[role] = latent_items(model, patchifier, cfg, items, device, masking_policy)
        fidelity: dict[str, float] = {}
        b0_conf: dict[str, dict] = {}
        for role in conf_roles:
            b0_conf[role] = load_conf_cache(b0_cache_dir, model_seed, role)
            z = b0_conf[role]
            pos = {str(fid): i for i, fid in enumerate(list(z["file_ids"]))}
            rows = conf_lat[role]
            want_ids = [r["file_id"] for r in rows]
            if not set(want_ids) <= set(pos):
                raise ValueError(f"B0 cache file-id mismatch: seed {model_seed}/{role}")
            if not args.smoke and set(want_ids) != set(pos):
                raise ValueError(f"B0 cache coverage mismatch: seed {model_seed}/{role}")
            idx = [pos[fid] for fid in want_ids]
            f_pred, _ = b0_file_scores(rows)
            f_emb = torch.stack([r["file_embedding"] for r in rows])
            f_pop = bank.score(f_emb).cpu().numpy().astype(np.float64)
            c_pred = np.asarray(z["S_pred"])[idx]
            c_pop = np.asarray(z["S_pop"])[idx]
            c_emb = torch.as_tensor(np.asarray(z["file_embedding"])[idx])
            d_pred = float(np.abs(f_pred - c_pred).max()) if len(c_pred) else 0.0
            d_pop = float(np.abs(f_pop - c_pop).max()) if len(c_pop) else 0.0
            d_emb = float((f_emb - c_emb).abs().max().item()) if len(idx) else 0.0
            fidelity[role] = max(d_pred, d_pop, d_emb)
            if fidelity[role] > FIDELITY_ATOL:
                raise ValueError(f"B0 fidelity breach: seed {model_seed}/{role} maxdiff={fidelity[role]}")

        if arm_id in ("C7-A", "C7-B"):
            from representation.sprint17_c7 import arm_reference
            fit_robots, fit_programs = condition_lists(fit_items)
            fit_labels = [sample.file_label for _, sample in fit_items]
            reference = arm_reference(arm_id).fit(
                torch.stack([r["file_embedding"] for r in fit_lat]),
                fit_robots, fit_programs, labels=fit_labels)
            fit_pop_c7, _ = reference.score(
                torch.stack([r["file_embedding"] for r in fit_lat]), fit_robots, fit_programs)
            fit_pop_c7 = np.asarray(fit_pop_c7, dtype=np.float64)
            fit_pred, _ = b0_file_scores(fit_lat)
            conf_out = {}
            for role in conf_roles:
                z = b0_conf[role]
                pos = {str(fid): i for i, fid in enumerate(list(z["file_ids"]))}
                ids = [r["file_id"] for r in conf_lat[role]]
                if not args.smoke and set(ids) != set(pos):
                    raise ValueError(f"B0 cache coverage mismatch: seed {model_seed}/{role}")
                if any(fid not in pos for fid in ids):
                    raise ValueError(f"file missing from B0 cache: seed {model_seed}/{role}")
                if len(set(ids)) != len(ids):
                    raise ValueError(f"duplicate file ids scored: seed {model_seed}/{role}")
                idx = [pos[fid] for fid in ids]
                pred = np.asarray(z["S_pred"])[idx].astype(np.float64)
                demb = torch.as_tensor(np.asarray(z["file_embedding"])[idx])
                tpad = np.asarray(z["timestep_scores"])[idx]
                tlen = np.asarray(z["timestep_lengths"])[idx]
                ttime = [np.asarray(tpad[i, :tlen[i]], dtype=np.float64) for i in range(len(ids))]
                d_robots = [int(conf_by_id[role][fid].robot_idx) for fid in ids]
                d_programs = [int(conf_by_id[role][fid].program_idx) for fid in ids]
                pop_c7, _ = reference.score(demb, d_robots, d_programs)
                pop = np.asarray(pop_c7, dtype=np.float64)
                if not (np.isfinite(pred).all() and np.isfinite(pop).all()
                        and torch.isfinite(demb).all()):
                    raise ValueError(f"non-finite Confirmation outputs: {model_seed}/{role}")
                conf_out[role] = {"ids": ids, "S_pred": pred, "S_pop": pop, "emb": demb, "t": ttime}
            corr = float(np.corrcoef(np.concatenate([conf_out[r]["S_pred"] for r in conf_roles]),
                np.concatenate([conf_out[r]["S_pop"] for r in conf_roles]))[0, 1])
            if not np.isfinite(corr) or corr >= 0.999:
                raise ValueError(f"score branches aliased: corr={corr}")
            cache_shas = persist_conf_caches(cache_dir, prefix, model_seed, conf_out)
            seed_records[model_seed] = train_free_record(lock, arm_id, model_seed, params, flops,
                thr_pred, thr_pop, fit_items, fit_patches, corr, ckpt_path, cache_shas,
                st, cfg_dict, conf_out, fit_pred, fit_pop_c7, fidelity)
            continue
        if arm_id in ("C8-A", "C8-B"):
            from representation.sprint17_c8 import HuberStandardizer
            standardizer = None
            scorer_prov: dict[str, object] = {"fitted": False}
            if arm_id == "C8-A":
                fit_res = torch.cat([(r["predicted"] - r["target"])[r["prediction_mask"]] for r in fit_lat], dim=0)
                fit_labels = []
                for r, (_, sample) in zip(fit_lat, fit_items):
                    n = int(r["prediction_mask"].sum().item())
                    fit_labels += [sample.file_label] * n
                standardizer = HuberStandardizer().fit(fit_res, labels=fit_labels)
                scorer_prov = standardizer.provenance()
            fit_pred_c8, _ = c8_file_scores(fit_lat, arm_id, standardizer)
            fit_emb = torch.stack([r["file_embedding"] for r in fit_lat])
            fit_pop_b0 = bank.score(fit_emb).cpu().numpy().astype(np.float64)
            conf_out = {}
            for role in conf_roles:
                z = b0_conf[role]
                pos = {str(fid): i for i, fid in enumerate(list(z["file_ids"]))}
                rows = conf_lat[role]
                ids = [r["file_id"] for r in rows]
                if not args.smoke and set(ids) != set(pos):
                    raise ValueError(f"B0 cache coverage mismatch: seed {model_seed}/{role}")
                if any(fid not in pos for fid in ids):
                    raise ValueError(f"file missing from B0 cache: seed {model_seed}/{role}")
                if len(set(ids)) != len(ids):
                    raise ValueError(f"duplicate file ids scored: seed {model_seed}/{role}")
                idx = [pos[fid] for fid in ids]
                pop = np.asarray(z["S_pop"])[idx].astype(np.float64)
                demb = torch.as_tensor(np.asarray(z["file_embedding"])[idx])
                pred_c8, patch_c8 = c8_file_scores(rows, arm_id, standardizer)
                ttime = [patchifier.patch_to_timestep_scores(patch_c8[i].astype(np.float32),
                    rows[i]["starts"], rows[i]["valid_len"], rows[i]["file_len"]) for i in range(len(ids))]
                ttime = [np.asarray(t, dtype=np.float64) for t in ttime]
                if not (np.isfinite(pred_c8).all() and np.isfinite(pop).all()
                        and torch.isfinite(demb).all()):
                    raise ValueError(f"non-finite Confirmation outputs: {model_seed}/{role}")
                conf_out[role] = {"ids": ids, "S_pred": pred_c8, "S_pop": pop, "emb": demb, "t": ttime}
            corr = float(np.corrcoef(np.concatenate([conf_out[r]["S_pred"] for r in conf_roles]),
                np.concatenate([conf_out[r]["S_pop"] for r in conf_roles]))[0, 1])
            if not np.isfinite(corr) or corr >= 0.999:
                raise ValueError(f"score branches aliased: corr={corr}")
            cache_shas = persist_conf_caches(cache_dir, prefix, model_seed, conf_out)
            seed_records[model_seed] = train_free_record(lock, arm_id, model_seed, params, flops,
                thr_pred, thr_pop, fit_items, fit_patches, corr, ckpt_path, cache_shas,
                st, cfg_dict, conf_out, fit_pred_c8, fit_pop_b0, fidelity,
                scorer_prov=scorer_prov)
            continue
        if arm_id in ("C9-A", "C9-B"):
            fit_pred_c9, _, _ = c9_file_scores(fit_lat, arm_id)
            fit_emb = torch.stack([r["file_embedding"] for r in fit_lat])
            fit_pop_b0 = bank.score(fit_emb).cpu().numpy().astype(np.float64)
            conf_out = {}
            for role in conf_roles:
                z = b0_conf[role]
                pos = {str(fid): i for i, fid in enumerate(list(z["file_ids"]))}
                rows = conf_lat[role]
                ids = [r["file_id"] for r in rows]
                if not args.smoke and set(ids) != set(pos):
                    raise ValueError(f"B0 cache coverage mismatch: seed {model_seed}/{role}")
                if any(fid not in pos for fid in ids):
                    raise ValueError(f"file missing from B0 cache: seed {model_seed}/{role}")
                if len(set(ids)) != len(ids):
                    raise ValueError(f"duplicate file ids scored: seed {model_seed}/{role}")
                idx = [pos[fid] for fid in ids]
                pop = np.asarray(z["S_pop"])[idx].astype(np.float64)
                demb = torch.as_tensor(np.asarray(z["file_embedding"])[idx])
                pred_c9, patch_c9, _ = c9_file_scores(rows, arm_id)
                ttime = [patchifier.patch_to_timestep_scores(patch_c9[i].astype(np.float32),
                    rows[i]["starts"], rows[i]["valid_len"], rows[i]["file_len"]) for i in range(len(ids))]
                ttime = [np.asarray(t, dtype=np.float64) for t in ttime]
                if not (np.isfinite(pred_c9).all() and np.isfinite(pop).all()
                        and torch.isfinite(demb).all()):
                    raise ValueError(f"non-finite Confirmation outputs: {model_seed}/{role}")
                conf_out[role] = {"ids": ids, "S_pred": pred_c9, "S_pop": pop, "emb": demb, "t": ttime}
            corr = float(np.corrcoef(np.concatenate([conf_out[r]["S_pred"] for r in conf_roles]),
                np.concatenate([conf_out[r]["S_pop"] for r in conf_roles]))[0, 1])
            if not np.isfinite(corr) or corr >= 0.999:
                raise ValueError(f"score branches aliased: corr={corr}")
            cache_shas = persist_conf_caches(cache_dir, prefix, model_seed, conf_out)
            seed_records[model_seed] = train_free_record(lock, arm_id, model_seed, params, flops,
                thr_pred, thr_pop, fit_items, fit_patches, corr, ckpt_path, cache_shas,
                st, cfg_dict, conf_out, fit_pred_c9, fit_pop_b0, fidelity)
            continue
        raise ValueError(f"unhandled arm in Task 24: {arm_id!r}")



def persist_conf_caches(cache_dir, prefix: str, model_seed: int, conf_out: dict) -> dict[str, str]:
    """Persist Confirmation caches (same npz contract as Development; server-side retained)."""
    import numpy as np

    cache_shas: dict[str, str] = {}
    for role, d in conf_out.items():
        tmax = max((t.size for t in d["t"]), default=0)
        tpad = np.zeros((len(d["ids"]), tmax), dtype=np.float32)
        tlen = np.zeros(len(d["ids"]), dtype=np.int64)
        for i, t in enumerate(d["t"]):
            tpad[i, :t.size] = t.astype(np.float32)
            tlen[i] = t.size
        cache_path = cache_dir / f"{prefix}_seed{model_seed}_{role}.npz"
        np.savez_compressed(cache_path, file_ids=np.array(d["ids"]),
            S_pred=d["S_pred"].astype(np.float64), S_pop=d["S_pop"].astype(np.float64),
            file_embedding=d["emb"].cpu().numpy().astype(np.float32),
            timestep_scores=tpad, timestep_lengths=tlen,
            provenance=np.array([f"{prefix} seed={model_seed} role={role} step=300 confirmation"]))
        cache_shas[role] = sha256_file(cache_path)
    return cache_shas


def train_free_record(lock: dict, arm_id: str, model_seed: int, params: int, flops: int,
                      thr_pred: float, thr_pop: float, fit_items: list, fit_patches: int,
                      corr: float, ckpt_path, cache_shas: dict, st: float, cfg_dict: dict,
                      conf_out: dict, fit_pred, fit_pop, fidelity: dict,
                      scorer_prov: dict | None = None) -> dict:
    """Seed record for train-free arms (frozen B0 states; zero optimizer steps)."""
    import torch

    rec = {"model_seed": model_seed, "steps": CHECKPOINT_STEP,
           "optimizer_steps_executed": 0, "params": params, "flops_reference": flops,
           "flop_reference_t": FLOP_REFERENCE_T, "bank_rows": FIT_ROWS,
           "bank_source": "B0 checkpoint-restored (fidelity-proven; branch reuse)",
           "thr_pred": thr_pred, "thr_pop": thr_pop, "threshold_source": "task23-lock",
           "fit_rows": len(fit_items), "fit_patches": fit_patches,
           "score_branch_corr": corr,
           "ckpt_sha256": lock["arms"]["B0"]["checkpoints"][str(model_seed)],
           "ckpt_path": str(ckpt_path), "cache_sha256": cache_shas,
           "fidelity_max_abs_diff": {r: float(v) for r, v in fidelity.items()},
           "elapsed_s": round(time.time() - st, 1),
           "conf_out": conf_out, "fit_pred": fit_pred, "fit_pop": fit_pop,
           "cfg_dict": cfg_dict}
    if scorer_prov is not None:
        rec["scorer_provenance"] = scorer_prov
    return rec


def _main_metrics_and_evidence(args, arm_id: str, prefix: str, seeds: tuple, seed_records: dict,
                               conf_supports: dict, conf_rows: dict, conf_ledgers: dict,
                               conf_wins: dict, conf_roles: list, metric_code_sha: str, git: dict,
                               runtime: dict, t0: float, fit_items: list, fit_patches: int,
                               arm_fit_patches: int, b0_pred, b0_pop, lock: dict,
                               outdir: Path) -> None:
    import numpy as np
    from synth import probe15 as P

    branch_records: dict[int, dict[str, dict]] = {}
    for model_seed in seeds:
        rec = seed_records[model_seed]
        file_scores_pred = build_score_map(conf_roles, rec["conf_out"], "S_pred")
        file_scores_pop = build_score_map(conf_roles, rec["conf_out"], "S_pop")
        conf_times = {role: dict(zip(rec["conf_out"][role]["ids"], rec["conf_out"][role]["t"]))
                      for role in conf_roles}
        branch_records[model_seed] = {
            "S_pred": branch_metrics("S_pred", rec["thr_pred"], file_scores_pred, conf_supports,
                conf_rows, conf_ledgers, conf_wins, conf_times, rec["fit_pred"], conf_roles),
            "S_pop": branch_metrics("S_pop", rec["thr_pop"], file_scores_pop, conf_supports,
                conf_rows, conf_ledgers, conf_wins, conf_times, rec["fit_pop"], conf_roles),
        }
    (outdir / "run.log.json").write_text(json.dumps(to_jsonable({
        "smoke": args.smoke, "protocol": PROTOCOL_ID, "phase": "confirmation",
        "arm": arm_id, "seeds": list(seeds), "training": "none (scoring-only)",
        "refit": "none", "recalibration": "none (thresholds task23-lock)",
        "git": git, "runtime": runtime, "metric_code_sha256": metric_code_sha,
        "metric_code_files": list(METRIC_CODE_FILES), "binding_sha256": BINDING_SHA256,
        "lock_sha256": TASK23_LOCK_SHA256, "fit_rows": len(fit_items),
        "fit_patches": fit_patches, "elapsed_s": round(time.time() - t0, 1)}),
        indent=1, sort_keys=True))
    assemble_confirmation_evidence(outdir, arm_id, seeds, seed_records, branch_records,
        conf_supports, conf_roles, metric_code_sha, git, runtime, t0,
        len(fit_items), fit_patches, arm_fit_patches, b0_pred, b0_pop, lock, args.smoke)
    if args.smoke:
        (outdir / "SMOKE.txt").write_text(
            "SMOKE RUN ONLY - scoring-only subset; not evidence.\n")
    print(json.dumps(to_jsonable({
        "arm": arm_id, "seeds": list(seed_records),
        "fit_rows": len(fit_items), "fit_patches": fit_patches,
        "thr": {str(s): {"S_pred": seed_records[s]["thr_pred"], "S_pop": seed_records[s]["thr_pop"]} for s in seeds},
        "macro_pw_pred": {str(s): branch_records[s]["S_pred"]["macro_pw"] for s in seeds},
        "macro_pw_pop": {str(s): branch_records[s]["S_pop"]["macro_pw"] for s in seeds},
        "device": f"{runtime['device']} ({runtime['device_name']})",
        "elapsed_s": round(time.time() - t0, 1), "smoke": args.smoke}), indent=1, sort_keys=True))
    return 0

    return _main_metrics_and_evidence(args, arm_id, prefix, seeds, seed_records,
        conf_supports, conf_rows, conf_ledgers, conf_wins, conf_roles,
        metric_code_sha, git, runtime, t0, fit_items, fit_patches,
        arm_fit_patches, b0_pred, b0_pop, lock, outdir)


def assemble_confirmation_evidence(outdir: Path, arm_id: str, seeds: tuple, seed_records: dict,
                      branch_records: dict, conf_supports: dict, conf_roles: list,
                      metric_code_sha: str, git: dict, runtime: dict, t0: float,
                      fit_rows: int, fit_patches: int, arm_fit_patches: int,
                      b0_pred, b0_pop, lock: dict, smoke: bool) -> None:
    """Write the v4-schema per-arm Confirmation document plus bounded sidecars.

    Mirrors the Development evidence shape with phase=confirmation,
    Confirmation roles/seeds/support, lock-sourced thresholds (no
    recalibration), and no training fields. Paired deltas use the B0
    Confirmation record; B0 carries no deltas by construction.
    """
    import numpy as np
    import jsonschema

    from synth import probe15 as P
    from representation import sprint17_ablation as A

    prefix = ARM_FILE_PREFIX[arm_id]
    components = list(ARM_COMPONENTS[arm_id])
    arm_kind = "baseline" if arm_id == "B0" else "single"
    if arm_id != "B0":
        A.validate_arm(A.get_arm(arm_id))
    A.validate_eligibility({"qualification": QUALIFICATION, "waiver_id": WAIVER_ID,
                            "waiver_scope": WAIVER_SCOPE})
    if not smoke and tuple(seeds) != MODEL_SEEDS:
        raise ValueError("non-smoke evidence requires the frozen model seeds")
    for model_seed in seeds:
        rec = seed_records[model_seed]
        if not smoke and rec["steps"] != OPTIMIZER_STEPS:
            raise ValueError(f"checkpoint-step breach: seed {model_seed}")
        want_thr = lock_thresholds(lock, arm_id)[str(model_seed)]
        if rec["thr_pred"] != want_thr["pred"] or rec["thr_pop"] != want_thr["pop"]:
            raise ValueError(f"threshold drift vs lock: seed {model_seed} (no recalibration)")
    params_set = {seed_records[s]["params"] for s in seeds}
    flops_set = {seed_records[s]["flops_reference"] for s in seeds}
    if len(params_set) != 1 or len(flops_set) != 1:
        raise ValueError("cross-seed parameter/FLOP counts must agree")
    params = params_set.pop()
    flops = flops_set.pop()
    A.validate_compute_envelope(A.ComputeEnvelope(
        b0_params=B0_PARAMS, params_total=params,
        b0_flops_per_reference_sample=B0_FLOPS_REFERENCE,
        flops_per_reference_sample=flops))
    for model_seed in seeds:
        rec = seed_records[model_seed]
        A.validate_provenance({
            "data_protocol": DATA_PROTOCOL,
            "role_binding_sha256": BINDING_SHA256,
            "data_roles": [r for r, _ in FIT_ROLES] + conf_roles,
            "data_seeds": [2804, 2805, 2806] + [2812, 2813, 2814, 2815],
            "model_seeds": list(MODEL_SEEDS),
            "fit_root_sha256": canonical_hash(
                sorted(EXPECTED_MANIFEST_SHA256[r] for r, _ in FIT_ROLES)),
            "calibration_root_sha256": lock["calibration_root_sha256"],
            "evaluation_root_sha256": canonical_hash(
                sorted(EXPECTED_MANIFEST_SHA256[r] for r in conf_roles)),
            "metric_code_sha256": metric_code_sha,
            "checkpoint_sha256": rec["ckpt_sha256"],
            "cache_sha256": canonical_hash(sorted(rec["cache_sha256"].values())),
            "parent_arm_ids": [] if arm_id == "B0" else ["B0"],
        })

    agg: dict[str, dict] = {}
    for branch in ("S_pred", "S_pop"):
        per_history = []
        for h, role in enumerate(conf_roles):
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
                "background_ratio": _mean_finite([c["background_ratio"] for c in cells]),
                "localization_rate": _mean_finite([c["localization_rate"] for c in cells]),
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
            "threshold_mean": float(np.mean([seed_records[s]["thr_" + branch.split("_")[1]] for s in seeds])),
        }

    primary = agg["S_pred"]
    support_entries = []
    excluded_total = 0
    for h, role in enumerate(conf_roles):
        support = conf_supports[role]
        excluded_total += support["skipped"]
        support_entries.append({
            "history_id": role, "p": len(support["pos"]["P"]),
            "w": len(support["pos"]["W"]), "a": len(support["pos"]["A"]),
            "controls": len(support["controls"]),
            "robot_days": float(primary["per_history"][h]["robot_days"] or 0.0),
        })
    common_support_sha = canonical_hash({
        role: {"events": sorted(tuple(e["key"]) for e in conf_supports[role]["pos"]["P"]
                               + conf_supports[role]["pos"]["W"]
                               + conf_supports[role]["pos"]["A"]),
               "controls": sorted(tuple(sorted(w)) for w in conf_supports[role]["controls"])}
        for role in conf_roles})
    if not smoke and arm_id != "B0":
        if b0_pred is None or b0_pop is None:
            raise ValueError("B0 confirmation record required for singles (B0 gate)")
        if common_support_sha != b0_pred.get("confirmation_support_sha256"):
            raise ValueError("common-support mismatch vs B0 Confirmation support")
    if not smoke and len(conf_roles) != 4:
        raise ValueError("Confirmation requires exactly four histories")
    for entry in support_entries:
        if not smoke and (entry["p"] == 0 or entry["w"] == 0 or entry["controls"] == 0):
            raise ValueError(f"empty Confirmation support cell: {entry}")

    deltas: dict[str, dict] = {}
    b0_hist: dict[str, list] = {}
    if b0_pred is not None and b0_pop is not None:
        stores = {"S_pred": (b0_pred, "S_pred_per_seed"), "S_pop": (b0_pop, "per_seed")}
        for branch, (store, key) in stores.items():
            per_seed_macro = {}
            hist_pw, hist_p, hist_w, hist_far = [], [], [], []
            for h, role in enumerate(conf_roles):
                bvals = []
                for s in seeds:
                    cell = store[key][str(s)]
                    if cell["per_history"][h]["history_id"] != role:
                        raise ValueError("B0 history alignment mismatch")
                    bvals.append(cell["per_history"][h])
                hist_pw.append(_mean_finite([c["auc_pw"] for c in bvals]))
                hist_p.append(_mean_finite([c["auc_p"] for c in bvals]))
                hist_w.append(_mean_finite([c["auc_w"] for c in bvals]))
                hist_far.append(_mean_finite([c["far"] for c in bvals]))
            b0_hist[branch] = hist_pw
            for s in seeds:
                cell = store[key][str(s)]
                per_seed_macro[s] = (cell["macro_pw"], cell["macro_p"], cell["macro_w"])
            arm_macros = {s: (branch_records[s][branch]["macro_pw"], branch_records[s][branch]["macro_p"], branch_records[s][branch]["macro_w"]) for s in seeds}
            d_pw = [arm_macros[s][0] - per_seed_macro[s][0] for s in seeds]
            d_p = [arm_macros[s][1] - per_seed_macro[s][1] for s in seeds]
            d_w = [arm_macros[s][2] - per_seed_macro[s][2] for s in seeds]
            arm_hist = agg[branch]["per_history"]
            deltas[branch] = {
                "macro_pw": float(np.mean(d_pw)), "macro_p": float(np.mean(d_p)), "macro_w": float(np.mean(d_w)),
                "per_seed_pw": {str(s): float(v) for s, v in zip(seeds, d_pw)},
                "per_seed_p": {str(s): float(v) for s, v in zip(seeds, d_p)},
                "per_seed_w": {str(s): float(v) for s, v in zip(seeds, d_w)},
                "per_history_pw": [(a["auc_pw"] - b if a["auc_pw"] is not None and b is not None else None) for a, b in zip(arm_hist, hist_pw)],
                "per_history_p": [(a["auc_p"] - b if a["auc_p"] is not None and b is not None else None) for a, b in zip(arm_hist, hist_p)],
                "per_history_w": [(a["auc_w"] - b if a["auc_w"] is not None and b is not None else None) for a, b in zip(arm_hist, hist_w)],
                "per_history_far": [(a["far"] - b if a["far"] is not None and b is not None else None) for a, b in zip(arm_hist, hist_far)],
            }
    else:
        for branch in ("S_pred", "S_pop"):
            deltas[branch] = {"macro_pw": None, "macro_p": None, "macro_w": None,
                "per_seed_pw": {}, "per_seed_p": {}, "per_seed_w": {},
                "per_history_pw": [None] * 4, "per_history_p": [None] * 4,
                "per_history_w": [None] * 4, "per_history_far": [None] * 4}
    far_pass = all((h["far"] is not None and h["far"] <= 0.05) for h in primary["per_history"])
    bg_pass = all((h["background_ratio"] is not None and h["background_ratio"] <= 0.10) for h in primary["per_history"])
    finite_ok = all(v is not None for h in primary["per_history"] for v in (h["auc_pw"], h["auc_p"], h["auc_w"]))
    if arm_id == "B0":
        if not smoke and not finite_ok:
            raise ValueError("B0 Confirmation eligibility failed: non-finite metrics (stop alternatives)")
        status, minimum_effect, p_noninf, w_noninf = "VALID_NEGATIVE", False, True, True
        d_pred = {"macro_pw": None, "macro_p": None, "macro_w": None, "per_history_far": [None] * 4}
    else:
        rec0 = seed_records[seeds[0]]
        d_pred = deltas["S_pred"]
        minimum_effect = (d_pred["macro_pw"] is not None and d_pred["macro_pw"] >= 0.05)
        p_noninf = (d_pred["macro_p"] is not None and d_pred["macro_p"] >= -0.02)
        w_noninf = (d_pred["macro_w"] is not None and d_pred["macro_w"] >= -0.02)
        status = "VALID_POSITIVE" if minimum_effect else "VALID_NEGATIVE"

    one_principal = "none" if arm_id == "B0" else arm_id.split("-")[0]
    cfg0 = seed_records[seeds[0]]["cfg_dict"]
    config = {"one_principal_change": one_principal, "adapter_description": cfg0["adapter_description"],
              "optimizer_steps": OPTIMIZER_STEPS, "checkpoint_step": CHECKPOINT_STEP,
              "calibration_quantile": CALIBRATION_QUANTILE,
              "score_branches": ["S_pred", "S_pop"]}
    rec0 = seed_records[seeds[0]]
    doc = {
        "schema_id": SCHEMA_ID, "protocol_id": PROTOCOL_ID, "phase": "confirmation",
        "arm_id": arm_id, "arm_kind": arm_kind, "component_set": components,
        "status": status, "invalid_reasons": [],
        "provenance": {
            "data_protocol": DATA_PROTOCOL, "role_binding_sha256": BINDING_SHA256,
            "data_roles": [r for r, _ in FIT_ROLES] + conf_roles,
            "data_seeds": [2804, 2805, 2806] + [2812, 2813, 2814, 2815],
            "model_seeds": list(MODEL_SEEDS),
            "fit_root_sha256": canonical_hash(sorted(EXPECTED_MANIFEST_SHA256[r] for r, _ in FIT_ROLES)),
            "calibration_root_sha256": lock["calibration_root_sha256"],
            "evaluation_root_sha256": canonical_hash(sorted(EXPECTED_MANIFEST_SHA256[r] for r in conf_roles)),
            "metric_code_sha256": metric_code_sha,
            "checkpoint_sha256": canonical_hash(sorted(rec["ckpt_sha256"] for rec in seed_records.values())),
            "cache_sha256": canonical_hash(sorted(sha for rec in seed_records.values() for sha in rec["cache_sha256"].values())),
            "parent_arm_ids": [] if arm_id == "B0" else ["B0"],
        },
        "config": config,
        "compute": {
            "b0_params": B0_PARAMS, "params_total": params,
            "params_delta_fraction": (params - B0_PARAMS) / B0_PARAMS,
            "b0_flops_per_reference_sample": B0_FLOPS_REFERENCE,
            "flops_per_reference_sample": flops,
            "flops_delta_fraction": (flops - B0_FLOPS_REFERENCE) / B0_FLOPS_REFERENCE,
            "envelope_pass": True,
        },
        "support": {"common_support_sha256": common_support_sha, "per_history": support_entries, "excluded_rows": excluded_total},
        "scores": {
            "S_pred": {"status": "PRESENT", "threshold": primary["threshold_mean"],
                       "provenance": canonical_hash(sorted(rec["cache_sha256"][r] for rec in seed_records.values() for r in conf_roles))},
            "S_pop": {"status": "PRESENT", "threshold": agg["S_pop"]["threshold_mean"],
                      "provenance": canonical_hash(sorted(rec["cache_sha256"][r] for rec in seed_records.values() for r in conf_roles))},
        },
        "metrics": {
            "primary_PW": {"point": primary["macro_pw"], "lcb95": primary["lcb_pw"], "delta_vs_B0": d_pred["macro_pw"],
                           "directional_count": primary["directional"],
                           "per_history": [h["auc_pw"] for h in primary["per_history"]]},
            "P": {"point": primary["macro_p"], "lcb95": primary["lcb_p"], "delta_vs_B0": d_pred["macro_p"],
                  "recall_per_history": [h["recall_p"] for h in primary["per_history"]],
                  "median_lead_days_per_history": [h["lead_p_median"] for h in primary["per_history"]]},
            "W": {"point": primary["macro_w"], "lcb95": primary["lcb_w"], "delta_vs_B0": d_pred["macro_w"],
                  "recall_per_history": [h["recall_w"] for h in primary["per_history"]],
                  "median_lead_days_per_history": [h["lead_w_median"] for h in primary["per_history"]]},
            "A_companion": {"point": _mean_finite([h["auc_a"] for h in primary["per_history"]]),
                "lcb95": (P.history_block_lcb([h["auc_a"] for h in primary["per_history"] if h["auc_a"] is not None])["lcb"]
                          if any(h["auc_a"] is not None for h in primary["per_history"]) else None),
                "per_history": [h["auc_a"] for h in primary["per_history"]]},
            "severity_ordering": {"P_spearman": [h["severity_p"] for h in primary["per_history"]],
                                  "W_spearman": [h["severity_w"] for h in primary["per_history"]]},
            "localization": {"present": True, "per_history": [h["localization_rate"] for h in primary["per_history"]]},
            "background": {"stability_ratio_per_history": [h["background_ratio"] for h in primary["per_history"]],
                           "far_per_robot_day": [h["far"] for h in primary["per_history"]],
                           "delta_far_vs_B0": d_pred["per_history_far"]},
            "category_slices": {role: {cid: _mean_finite([branch_records[s]["S_pred"]["per_history"][h]["categories"][cid] for s in seeds])
                                        for cid in ("P1", "P2", "W1", "W2", "A1", "A2")} for h, role in enumerate(conf_roles)},
            "bootstrap": {"replicates": 2000, "seed": 20260202, "lcb_percentile": 2.5},
        },
        "gates": {
            "structural": True, "observable": True, "finite_and_support": bool(finite_ok),
            "score_separation": True, "compute_envelope": True,
            "minimum_effect": bool(minimum_effect), "P_noninferiority": bool(p_noninf),
            "W_noninferiority": bool(w_noninf), "nuisance": bool(far_pass and _far_delta_ok(d_pred)),
            "background_stability": bool(bg_pass), "eligible": True, "recovery": False,
        },
        "eligibility": {"qualification": QUALIFICATION, "waiver_id": WAIVER_ID, "waiver_scope": WAIVER_SCOPE},
    }
    proto_path = REPO_ROOT / "experiments" / "sprint17-ablation-protocol-v4.md"
    text = proto_path.read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(\{.*?\n\})\n```", text, re.DOTALL)
    schemas = [b for b in blocks if '"$schema"' in b]
    if len(schemas) != 1:
        raise ValueError(f"expected exactly one v4 JSON schema block, found {len(schemas)}")
    schema = json.loads(schemas[0])
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(to_jsonable(doc)), key=lambda e: list(e.path))
    if errors:
        raise ValueError(f"{arm_id} document schema errors: " + "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:8]))
    (outdir / f"{prefix}_confirmation_v4.json").write_text(json.dumps(to_jsonable(doc), indent=1, sort_keys=True))
    slim_records = {}
    for s in seeds:
        rec = dict(seed_records[s])
        rec.pop("conf_out", None)
        rec.pop("fit_pred", None)
        rec.pop("fit_pop", None)
        slim_records[str(s)] = rec
    (outdir / f"{prefix}_spop_metrics.json").write_text(json.dumps(to_jsonable(
        {"branch": "S_pop", "aggregate": agg["S_pop"], "delta_vs_B0": deltas["S_pop"],
         "per_seed": {str(s): branch_records[s]["S_pop"] for s in seeds}}), indent=1, sort_keys=True))
    (outdir / f"{prefix}_per_seed_metrics.json").write_text(json.dumps(to_jsonable(
        {"phase": "confirmation", "arm_id": arm_id, "status": status,
         "S_pred_per_seed": {str(s): branch_records[s]["S_pred"] for s in seeds},
         "delta_vs_B0": deltas, "confirmation_support_sha256": common_support_sha,
         "seed_records": slim_records}), indent=1, sort_keys=True))
    (outdir / f"{prefix}_run_manifest.json").write_text(json.dumps(to_jsonable({
        "protocol": PROTOCOL_ID, "phase": "confirmation", "arm": arm_id, "seeds": list(seeds),
        "config_sha256": canonical_hash({str(s): seed_records[s]["cfg_dict"] for s in seeds}),
        "git": git, "runtime": runtime, "metric_code_sha256": metric_code_sha,
        "common_support_sha256": common_support_sha, "lock_sha256": TASK23_LOCK_SHA256,
        "training": "none (scoring-only)", "refit": "none",
        "recalibration": "none (thresholds task23-lock)",
        "fit_rows": fit_rows, "fit_patches": fit_patches, "arm_fit_patches": arm_fit_patches,
        "elapsed_s_total": round(time.time() - t0, 1)}), indent=1, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
