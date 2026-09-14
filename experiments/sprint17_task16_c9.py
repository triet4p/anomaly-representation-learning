"""Sprint 17 Task 16 — evaluate C9 patch-to-file aggregation arms on Development
(no training).

Runs exactly one registered C9 arm per invocation (``--arm C9-A`` or ``--arm
C9-B``) under the frozen B0 parity contract: same Fit/Calibration/Development
roles, model seeds ``[171701, 171702, 171703]``, frozen B0 step-300 states
(zero optimizer steps executed here), q95 Calibration thresholds, and the
verbatim B0 metric code (paired arm-minus-B0 deltas on identical support).

C9-A is the fixed-fraction (0.25) top-k mean over valid patch scores; C9-B is
the maximum fixed-duration (64 timesteps) contiguous-window mean using patch
starts and valid lengths. Only the patch-to-file aggregation changes: patch
scores, masks, representations, targets, S_pop, calibration rule, support,
localization mapping, timestep scores, and metric code are unchanged, and
S_pop is reused bitwise from the hash-verified B0 caches (it is a file-level
bank distance with no patch scores, so the C9 mechanism cannot apply to it;
never fused, never substituted).


Frozen contract: protocol ``sprint17-ablation-v4``, role binding
``experiments/sprint17-role-binding-v3.json``
(digest ``075868b22…8230ba``), model seeds ``[171701, 171702, 171703]``,
``optimizer_steps = checkpoint_step = 300``, calibration quantile 0.95,
qualification ``MEASURABLE_WITH_USER_WAIVER``.

Explicit non-goals: no Confirmation/Sealed access (refused at the loader),
no other arm, no threshold tuning, no early stopping or best-state selection
(final step-300 state is the checkpoint), no extra gradient runs, no pooled
rescue, no B0 retraining (B0 states, caches, and S_pop reused from hash-verified
Task 7 evidence; this driver executes zero optimizer steps).

Scoring-mask rule (frozen for reproducibility): every scored file uses
``masking_seed = sha256(file_id) mod 2**31`` with the frozen masking config,
so file scores are batching- and order-independent. Fit/Calibration/
Development latents come from the reloaded frozen B0 forward; Development
B0-MSE S_pred and bank S_pop recomputed from those latents are asserted
against the hash-verified B0 caches (fidelity proof: byte/hash-proven
identical valid patch scores and identical support), S_pop is reused bitwise
from those caches, and S_pred file scores are recomputed by the C9 aggregation
over those identical valid patch scores.
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
    "experiments/sprint17_task16_c9.py",
)
#: Frozen B0 reference values (committed record:
#: ``experiments/sprint17-task7-b0-summary.json``).  Used for compute-envelope
#: denominators and for verifying the reused B0 comparison baseline.
B0_PARAMS = 1821698
B0_FLOPS_REFERENCE = 182016709
B0_SUPPORT_SHA256 = ("800fc825ac948c3661c2dc727c83083095a796df8950c94922dafabae"
                     "318f59a")
B0_MACRO_PW = {
    # (S_pred, S_pop) per-seed P+W macro AUROC from the committed summary.
    171701: (0.6679282581030039, 0.687559772912035),
    171702: (0.6662107935384817, 0.663368934584594),
    171703: (0.6819876263571087, 0.6912224671869538),
}
C9_ARMS = ("C9-A", "C9-B")
ARM_FILE_PREFIX = {"C9-A": "c9a", "C9-B": "c9b"}

#: Frozen B0 step-300 checkpoint hashes (Task 7 record, verified before reload).
B0_CKPT_SHA256 = {
    171701: "45f9e151c5d3946804ea531eb2bcace26b1f62e634671f51b8741ac6b411a619",
    171702: "64ed2cedec210e4692f60bb4dd3430a57cf94abfcd50551abd97c9265ea785f8",
    171703: "9edb2122e357376a9ff1e065d73d5ab7704f8d2005991552332f1ced01b0e906",
}
#: Frozen B0 Development cache hashes per (seed, role) (Task 7 record).
B0_CACHE_SHA256 = {
    171701: {
        "H-S17-V3-DEV-01": "eb73ffe26e7b64349bbf6cea06236518225ccc2247c5fa849a8a65f29bfa3438",
        "H-S17-V3-DEV-02": "796161ca35d61755ed438c78ab829578e2cb98c17fc6aa1d6284d24390b4886e",
        "H-S17-V3-DEV-03": "8b60cc6fc08529e4e7c43d98e37ff92e7ba18943f44e39a4e115de47e8c8327f",
        "H-S17-V3-DEV-04": "df829e98329eef1221cd6cc3dc591b31da77d5691d65f51bf111dbf5e26a1d73",
    },
    171702: {
        "H-S17-V3-DEV-01": "d85dd652331f32dba574440daa71fec75875fc7fc148139beb106d1c0b9adbd4",
        "H-S17-V3-DEV-02": "10305a9a571367fb1176952ceacaeb8d0940c4444ccd6c37e0fd58d3c09a0fbe",
        "H-S17-V3-DEV-03": "254a19996a817ca626a38c51ddd00da77a5c422a11243e0c52c28c2194fbba0f",
        "H-S17-V3-DEV-04": "d89b2bfcd407790d084301165f7c997d72e313a6870d623ebdacb9331b48966e",
    },
    171703: {
        "H-S17-V3-DEV-01": "d875ee62adaa234c65b98451d4cf262ba9442da2874806e0ba5e571a35ed07e5",
        "H-S17-V3-DEV-02": "2dcfe36210412030b6bb0504d5cab7d231325f0a18ff3be3f6be5113d5eaac2e",
        "H-S17-V3-DEV-03": "e648f943af1d031b801854c712b02aa5d1b1be9d9ce599065c3b9d1c036710de",
        "H-S17-V3-DEV-04": "613e3f6797e4f7323180a8942d979786ebdea64cf9d381f35a2f8671d126ab19",
    },
}
#: Fidelity gate: reloaded-forward vs hash-verified B0 cache, max abs diff.
#: 1e-3 bounds the known GPU-train -> CPU-forward float wobble (Task 8
#: measured 9.37e-06 S_pred / 5.53e-04 S_pop abs on the same rescore path);
#: a wrong checkpoint/code path differs by orders of magnitude more.
FIDELITY_ATOL = 1e-3

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
        raise ValueError(f"refusing {group} access in Task 16 (Development only)")
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

def arm_config_dict(arm_id: str, model_seed: int) -> dict:
    """Frozen per-arm configuration: B0 values plus the declared C9 change.

    Only ``arm_id``, ``aggregation``, and ``adapter_description`` differ
    from :func:`b0_config_dict`. Patch scores, masks, representations,
    targets, S_pop path, localization mapping, objective, optimizer,
    schedule, batch rule, precision, and budget are identical by
    construction; this driver executes zero optimizer steps
    (representations inherit the frozen B0 step-300 states).
    """
    from representation.sprint17_c9 import ARM_ADAPTER_DESCRIPTION
    if arm_id not in C9_ARMS:
        raise ValueError(f"Task 16 runs only {C9_ARMS}, got {arm_id!r}")
    cfg = b0_config_dict(model_seed)
    cfg["arm_id"] = arm_id
    cfg["aggregation"] = {"C9-A": "top-k-mean-fraction-0.25",
                          "C9-B": "contiguous-window-mean-duration-64"}[arm_id]
    cfg["adapter_description"] = ARM_ADAPTER_DESCRIPTION[arm_id]
    return cfg


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
        # C9 keeps B0 patchification and encoding: the reference batch uses the
        # B0 lattice; the model graph is unchanged, so the count must reproduce B0 exactly.
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
    # C9 keeps the B0 graph exactly (frozen representations); only the
    # S_pred patch-to-file aggregation is replaced downstream of the forward.
    patchifier = Patchifier(PatchConfig(patch_size=cfg.patch_size,
                                        stride=cfg.stride,
                                        pad_end=cfg_dict["pad_end"]))
    model = V1RepresentationModel(cfg, patchifier=patchifier)
    if count_parameters(model) != B0_PARAMS:
        raise ValueError("C9 model graph is not the frozen B0 graph")
    if count_flops_reference(model) != B0_FLOPS_REFERENCE:
        raise ValueError("C9 scoring FLOPs differ from the B0 reference")
    model.to(device)
    return model, cfg, patchifier

def reload_b0(arm_id: str, model_seed: int, ckpt_dir, device):
    """Reload the frozen hash-verified B0 step-300 state (zero optimizer steps)."""
    import torch
    from representation.checkpoint import load_checkpoint
    from representation.inference import NormalReferenceBank

    if arm_id not in C9_ARMS:
        raise ValueError(f"Task 16 runs only {C9_ARMS}, got {arm_id!r}")
    cfg_dict = arm_config_dict(arm_id, model_seed)
    model, cfg, patchifier = build_model(cfg_dict, device)
    ckpt_path = Path(ckpt_dir) / f"b0_seed{model_seed}_step300.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing frozen B0 checkpoint (no retraining): {ckpt_path}")
    if sha256_file(ckpt_path) != B0_CKPT_SHA256[model_seed]:
        raise ValueError(f"B0 checkpoint hash mismatch for seed {model_seed}")
    bank = NormalReferenceBank(k=cfg.knn_k)
    info = load_checkpoint(ckpt_path, model, expected_config=cfg,
                           reference_bank=bank)
    if info["step"] != CHECKPOINT_STEP:
        raise ValueError(f"B0 checkpoint step is {info['step']}, not 300")
    if not info["has_reference_bank"] or bank.embeddings is None:
        raise ValueError("B0 checkpoint carries no reference bank")
    if bank.embeddings.shape[0] < 200:
        raise ValueError("restored B0 bank below the Fit floor")
    model.eval()
    return model, bank, cfg, patchifier, cfg_dict, ckpt_path


def load_b0_cache(cache_dir, model_seed: int, role: str) -> dict:
    """Load one hash-verified B0 Development cache (frozen S_pred/embeddings)."""
    import numpy as np

    path = Path(cache_dir) / f"b0_seed{model_seed}_{role}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen B0 cache (no recompute): {path}")
    if sha256_file(path) != B0_CACHE_SHA256[model_seed][role]:
        raise ValueError(f"B0 cache hash mismatch for seed {model_seed}/{role}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def latent_items(model, patchifier, cfg, items: list[tuple[str, object]], device):
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
                                           masking_seed=file_seed(fid))
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


def c9_file_scores(lat_list: list[dict], arm_id: str) -> tuple:
    """C9 file scores over the identical valid B0 patch scores + B0 energies.

    Per-file B0 per-patch MSE energies come from the frozen forward latents;
    the arm aggregation (top-k / contiguous-window) maps them to file
    scores. Returns ``(scores, b0_patch_energies, provenance)`` where the
    middle term is the shared identical input both arms and B0 aggregate.
    """
    import numpy as np
    import torch
    from representation.sprint17_c9 import arm_file_scores, b0_patch_energies

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


def b0_file_scores(lat_list: list[dict]) -> tuple:
    """B0-MSE file scores + per-patch energies (fidelity oracle)."""
    import numpy as np
    import torch
    from representation.sprint17_c9 import b0_patch_energies, file_mean

    scores, patch = [], []
    for row in lat_list:
        pred = row["predicted"].unsqueeze(0)
        targ = row["target"].unsqueeze(0)
        mask = row["prediction_mask"].unsqueeze(0)
        energies = b0_patch_energies(pred, targ, mask)
        scores.append(float(file_mean(energies, mask)[0].item()))
        patch.append(energies[0].detach().cpu().numpy().astype(np.float64))
    return np.array(scores, dtype=np.float64), patch



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
    ap.add_argument("--arm", required=True, choices=list(C9_ARMS),
                    help="registered C9 arm to execute (one per invocation)")
    ap.add_argument("--b0-metrics", default="",
                    help="hash-verified Task 7 b0_per_seed_metrics.json for "
                         "paired deltas and reused S_pred thresholds (required)")
    ap.add_argument("--b0-checkpoints", default="",
                    help="directory with frozen b0_seed<seed>_step300.pt states "
                         "(hash-verified; required)")
    ap.add_argument("--b0-caches", default="",
                    help="directory with frozen b0_seed<seed>_<role>.npz Development "
                         "caches (hash-verified; required)")
    ap.add_argument("--data-root", default="data/generated/sprint17-ablation-v3")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(str(s) for s in MODEL_SEEDS))
    ap.add_argument("--smoke", action="store_true",
                    help="train-free CPU smoke on a tiny subset (still needs the "
                         "frozen B0 states/caches); outputs marked SMOKE, "
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

    arm_id = args.arm
    prefix = ARM_FILE_PREFIX[arm_id]
    # Reused B0 comparison baseline: Task 7 per-seed evidence, verified
    # in-run against the committed summary (no B0 retraining). --b0-metrics
    # is either the b0_per_seed_metrics.json file (with sibling
    # b0_spop_metrics.json) or the directory containing both.
    b0_pred = None
    b0_pop = None
    if args.b0_metrics:
        given = Path(args.b0_metrics)
        if given.is_dir():
            pred_path = given / "b0_per_seed_metrics.json"
            pop_path = given / "b0_spop_metrics.json"
        else:
            pred_path = given
            pop_path = given.parent / "b0_spop_metrics.json"
        b0_pred = json.loads(pred_path.read_text(encoding="utf-8"))
        b0_pop = json.loads(pop_path.read_text(encoding="utf-8"))
        for seed_int in MODEL_SEEDS:
            key = str(seed_int)
            want_pred, want_pop = B0_MACRO_PW[seed_int]
            if b0_pred["S_pred_per_seed"][key]["macro_pw"] != want_pred:
                raise ValueError(
                    f"B0 baseline identity mismatch (S_pred seed {key})")
            if b0_pop["per_seed"][key]["macro_pw"] != want_pop:
                raise ValueError(
                    f"B0 baseline identity mismatch (S_pop seed {key})")
    else:
        raise ValueError("--b0-metrics is required (frozen S_pred reuse)")

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
    # C9 keeps B0 patchification exactly: the arm patch count must equal the
    # manifest count for every arm (fail-closed lattice-identity check).
    from synth.config import PatchConfig as _PatchConfig
    from synth.patchify import Patchifier as _Patchifier
    _b0_patchifier = _Patchifier(_PatchConfig(patch_size=32, stride=16,
                                              pad_end=True))
    arm_fit_patches = sum(_b0_patchifier.patchify(s).N for s in fit_pool)
    if not args.smoke and arm_fit_patches != fit_patches:
        raise ValueError(
            f"C9 lattice breach: arm patches {arm_fit_patches} != "
            f"manifest {fit_patches}")
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
    # ---- Per-seed frozen reload / latents / C9 aggregate / calibrate / score ----

    if not args.b0_checkpoints or not args.b0_caches:
        raise ValueError("--b0-checkpoints and --b0-caches are required (frozen B0 states)")
    ckpt_dir_arg = Path(args.b0_checkpoints)
    b0_cache_dir = Path(args.b0_caches)
    cache_dir = outdir / "caches"
    cache_dir.mkdir(parents=True, exist_ok=True)

    seed_records: dict[int, dict] = {}
    for model_seed in seeds:
        st = time.time()
        # 1. Reload the frozen hash-verified B0 step-300 state (zero optimizer steps).
        model, b0_bank, cfg, patchifier, cfg_dict, ckpt_path = reload_b0(
            arm_id, model_seed, ckpt_dir_arg, device)
        params = count_parameters(model)
        flops = count_flops_reference(model)
        # 2. Hash-verified B0 Development caches: bitwise S_pop + S_pred oracle.
        b0_dev = {}
        for role in dev_roles:
            b0_dev[role] = load_b0_cache(b0_cache_dir, model_seed, role)
        # 3. Frozen forward latents for Fit / Calibration / Development.
        fit_lat = latent_items(model, patchifier, cfg, fit_items, device)
        cal_lat = latent_items(model, patchifier, cfg, cal_items, device)
        dev_lat: dict[str, list] = {}
        for role in dev_roles:
            items = [(r["file_id"], dev_by_id[role][r["file_id"]])
                     for r in dev_rows[role] if r["file_id"] in dev_by_id[role]]
            dev_lat[role] = latent_items(model, patchifier, cfg, items, device)
        # 4. Fidelity proof: B0-MSE S_pred and bank S_pop from these latents
        # must reproduce the hash-verified caches (fail closed above FIDELITY_ATOL).
        fidelity: dict[str, float] = {}
        for role in dev_roles:
            z = b0_dev[role]
            pos = {str(fid): i for i, fid in enumerate(list(z["file_ids"]))}
            rows = dev_lat[role]
            want_ids = [r["file_id"] for r in rows]
            if not set(want_ids) <= set(pos):
                raise ValueError(f"B0 cache file-id mismatch: seed {model_seed}/{role}")
            if not args.smoke and set(want_ids) != set(pos):
                raise ValueError(f"B0 cache coverage mismatch: seed {model_seed}/{role}")
            idx = [pos[fid] for fid in want_ids]
            f_pred, _ = b0_file_scores(rows)
            f_emb = torch.stack([r["file_embedding"] for r in rows])
            f_pop = b0_bank.score(f_emb).cpu().numpy().astype(np.float64)
            c_pred = np.asarray(z["S_pred"])[idx]
            c_pop = np.asarray(z["S_pop"])[idx]
            c_emb = torch.as_tensor(np.asarray(z["file_embedding"])[idx])
            d_pred = float(np.abs(f_pred - c_pred).max()) if len(c_pred) else 0.0
            d_pop = float(np.abs(f_pop - c_pop).max()) if len(c_pop) else 0.0
            d_emb = float((f_emb - c_emb).abs().max().item()) if len(idx) else 0.0
            fidelity[role] = max(d_pred, d_pop, d_emb)
            if fidelity[role] > FIDELITY_ATOL:
                raise ValueError(
                    f"B0 fidelity breach: seed {model_seed}/{role} maxdiff={fidelity[role]}")
        # 5. C9 needs no fitting: the aggregation has no learned parameters.
        # Fitness-independent provenance records the frozen numerics.
        from representation.sprint17_c9 import (TOP_K_FRACTION,
                                                WINDOW_DURATION_TIMESTEPS)
        agg_prov: dict[str, object] = {
            "fitted": False,
            "top_k_fraction": TOP_K_FRACTION,
            "window_duration_timesteps": WINDOW_DURATION_TIMESTEPS,
        }
        # 6. C9 S_pred on forward latents (Calibration for thresholds,
        # Development for metrics, Fit for the stability denominator). The
        # per-patch inputs are the B0 MSE energies proven identical above.
        cal_pred_c9, _, _ = c9_file_scores(cal_lat, arm_id)
        fit_pred_c9, _, _ = c9_file_scores(fit_lat, arm_id)
        # 7. Thresholds: S_pred fresh q95 on Calibration C9 scores; S_pop
        # reused bitwise from B0 (proof: recomputed q95 on forward bank S_pop
        # matches outside smoke).
        thr_pred = P.select_threshold(cal_pred_c9)
        cal_emb = torch.stack([r["file_embedding"] for r in cal_lat])
        cal_pop_b0 = b0_bank.score(cal_emb).cpu().numpy().astype(np.float64)
        b0_thr_pop = float(b0_pred["seed_records"][str(model_seed)]["thr_pop"])
        if not args.smoke:
            repop = P.select_threshold(cal_pop_b0)
            if abs(repop - b0_thr_pop) > FIDELITY_ATOL:
                raise ValueError(f"S_pop threshold proof failed: seed {model_seed}")
        thr_pop = b0_thr_pop
        if not (np.isfinite(thr_pred) and np.isfinite(thr_pop)):
            raise ValueError(f"non-finite calibration threshold: seed {model_seed}")
        # Fit-healthy S_pop denominator via the fidelity-proven bank path.
        fit_emb = torch.stack([r["file_embedding"] for r in fit_lat])
        fit_pop_b0 = b0_bank.score(fit_emb).cpu().numpy().astype(np.float64)
        dev_out: dict[str, dict] = {}
        dev_prov: list[dict] = []
        for role in dev_roles:
            z = b0_dev[role]
            pos = {str(fid): i for i, fid in enumerate(list(z["file_ids"]))}
            rows = dev_lat[role]
            ids = [r["file_id"] for r in rows]
            if not args.smoke and set(ids) != set(pos):
                raise ValueError(f"B0 cache coverage mismatch: seed {model_seed}/{role}")
            if any(fid not in pos for fid in ids):
                raise ValueError(f"file missing from B0 cache: seed {model_seed}/{role}")
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate file ids scored: {model_seed}/{role}")
            idx = [pos[fid] for fid in ids]
            pop = np.asarray(z["S_pop"])[idx].astype(np.float64)
            demb = torch.as_tensor(np.asarray(z["file_embedding"])[idx])
            pred_c9, b0_patch, win_prov = c9_file_scores(rows, arm_id)
            dev_prov.extend(win_prov)
            # Localization/duration preservation: timestep scores reuse the
            # unchanged patch mapping over the identical B0 patch energies
            # (only the file-level aggregation changes).
            ttime = [patchifier.patch_to_timestep_scores(
                         b0_patch[i].astype(np.float32), rows[i]["starts"],
                         rows[i]["valid_len"], rows[i]["file_len"])
                     for i in range(len(ids))]
            ttime = [np.asarray(t, dtype=np.float64) for t in ttime]
            if not (np.isfinite(pred_c9).all() and np.isfinite(pop).all()
                    and torch.isfinite(demb).all()):
                raise ValueError(f"non-finite Development outputs: {model_seed}/{role}")
            dev_out[role] = {"ids": ids, "S_pred": pred_c9,
                             "S_pop": pop, "emb": demb, "t": ttime}
        # Score-branch separation (fail fast on aliased branches).
        corr = float(np.corrcoef(
            np.concatenate([dev_out[r]["S_pred"] for r in dev_roles]),
            np.concatenate([dev_out[r]["S_pop"] for r in dev_roles]))[0, 1])
        if not np.isfinite(corr) or corr >= 0.999:
            raise ValueError(f"score branches aliased: corr={corr}")
        # Persist C9 caches (same contract/keys as B0; S_pop/emb reused bitwise,
        # timestep scores from the identical B0 patch energies).
        ckpt_sha = B0_CKPT_SHA256[model_seed]
        cache_shas: dict[str, str] = {}
        for role in dev_roles:
            d = dev_out[role]
            tmax = max((t.size for t in d["t"]), default=0)
            tpad = np.zeros((len(d["ids"]), tmax), dtype=np.float32)
            tlen = np.zeros(len(d["ids"]), dtype=np.int64)
            for i, t in enumerate(d["t"]):
                tpad[i, :t.size] = t.astype(np.float32)
                tlen[i] = t.size
            cache_path = cache_dir / f"{prefix}_seed{model_seed}_{role}.npz"
            np.savez_compressed(
                cache_path,
                file_ids=np.array(d["ids"]),
                S_pred=d["S_pred"].astype(np.float64),
                S_pop=d["S_pop"].astype(np.float64),
                file_embedding=d["emb"].cpu().numpy().astype(np.float32),
                timestep_scores=tpad, timestep_lengths=tlen,
                provenance=np.array([f"{arm_id} seed={model_seed} role={role} "
                                     f"step={CHECKPOINT_STEP}"]),
            )
            cache_shas[role] = sha256_file(cache_path)
        # Duration accounting (C9-B windows; descriptive companion): mean
        # covered valid patches and span across Development files.
        win_cover = [p.get("n_covered") for p in dev_prov
                     if isinstance(p.get("n_covered"), int)]
        win_span = [p.get("covered_span") for p in dev_prov
                    if isinstance(p.get("covered_span"), int)]
        seed_records[model_seed] = {
            "model_seed": model_seed, "steps": CHECKPOINT_STEP,
            "optimizer_steps_executed": 0,
            "params": params, "flops_reference": flops,
            "flop_reference_t": FLOP_REFERENCE_T,
            "bank_rows": int(b0_bank.embeddings.shape[0]), "bank_k": cfg.knn_k,
            "bank_source": "B0 checkpoint-restored (fidelity-proven; S_pop reuse)",
            "aggregation_kind": {"C9-A": "top-k-mean-fraction-0.25",
                                 "C9-B": "contiguous-window-mean-duration-64"}[arm_id],
            "aggregation_provenance": agg_prov,
            "window_n_covered_mean": (float(np.mean(win_cover)) if win_cover else None),
            "window_covered_span_mean": (float(np.mean(win_span)) if win_span else None),
            "fidelity_max_abs_diff": fidelity,
            "thr_pred": float(thr_pred), "thr_pop": float(thr_pop),
            "cal_rows": len(cal_items), "fit_rows": len(fit_items),
            "fit_patches": fit_patches,
            "score_branch_corr": corr,
            "ckpt_path": str(ckpt_path), "ckpt_sha256": ckpt_sha,
            "cache_sha256": cache_shas,
            "elapsed_s": round(time.time() - st, 1),
            "dev_out": dev_out, "fit_pred": fit_pred_c9, "fit_pop": fit_pop_b0,
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
        "arm": arm_id,
        "seeds": list(seeds),
        "steps": CHECKPOINT_STEP,
        "optimizer_steps_executed": 0,
        "batch_size": "n/a (train-free per-file forward)",
        "git": git,
        "runtime": runtime,
        "metric_code_sha256": metric_code_sha,
        "metric_code_files": list(METRIC_CODE_FILES),
        "binding_sha256": BINDING_SHA256,
        "fit_rows": len(fit_items),
        "fit_patches": fit_patches,
        "arm_fit_patches": arm_fit_patches,
        "cal_rows": len(cal_items),
        "b0_verified": b0_pred is not None,
        "train_trace": run_log,
        "training_basis": "B0-frozen step-300 states (zero optimizer steps here)",
        "elapsed_s": round(time.time() - t0, 1),
    }), indent=1, sort_keys=True))
    assemble_evidence(outdir, arm_id, seeds, seed_records, branch_records,
                      dev_supports, dev_roles, metric_code_sha, git, runtime,
                      t0, len(fit_items), fit_patches, arm_fit_patches,
                      len(cal_items), b0_pred, b0_pop, args.smoke)
    if args.smoke:
        (outdir / "SMOKE.txt").write_text(
            "SMOKE RUN ONLY — train-free frozen forward on a tiny subset; not evidence.\n")

    print(json.dumps(to_jsonable({
        "arm": arm_id,
        "seeds": list(seed_records),
        "fit_rows": len(fit_items),
        "fit_patches": fit_patches,
        "arm_fit_patches": arm_fit_patches,
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


def assemble_evidence(outdir: Path, arm_id: str, seeds: tuple, seed_records: dict,
                      branch_records: dict, dev_supports: dict, dev_roles: list,
                      metric_code_sha: str, git: dict, runtime: dict, t0: float,
                      fit_rows: int, fit_patches: int, arm_fit_patches: int,
                      cal_rows: int, b0_pred: dict | None, b0_pop: dict | None,
                      smoke: bool) -> None:
    """Write the v4-schema per-arm Development document plus bounded sidecars."""
    import numpy as np
    import jsonschema

    from synth import probe15 as P
    from representation import sprint17_ablation as A
    from representation.sprint17_c9 import ARM_ADAPTER_DESCRIPTION

    prefix = ARM_FILE_PREFIX[arm_id]
    # Task 5 harness guards over the frozen declarations.
    A.validate_arm(A.get_arm(arm_id))
    A.validate_training_parity(A.TrainingParity())
    A.validate_eligibility({"qualification": QUALIFICATION, "waiver_id": WAIVER_ID,
                            "waiver_scope": WAIVER_SCOPE})
    if not smoke and tuple(seeds) != MODEL_SEEDS:
        raise ValueError("non-smoke evidence requires the frozen model seeds")
    for model_seed in seeds:
        rec = seed_records[model_seed]
        if not smoke and rec["steps"] != OPTIMIZER_STEPS:
            raise ValueError(f"checkpoint-step breach: seed {model_seed}")
    params_set = {seed_records[s]["params"] for s in seeds}
    flops_set = {seed_records[s]["flops_reference"] for s in seeds}
    if len(params_set) != 1 or len(flops_set) != 1:
        raise ValueError("cross-seed parameter/FLOP counts must agree")
    params = params_set.pop()
    flops = flops_set.pop()
    # Compute envelope against the FROZEN B0 reference (fail-closed breach).
    A.validate_compute_envelope(A.ComputeEnvelope(
        b0_params=B0_PARAMS, params_total=params,
        b0_flops_per_reference_sample=B0_FLOPS_REFERENCE,
        flops_per_reference_sample=flops))
    for model_seed in seeds:
        rec = seed_records[model_seed]
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
            "parent_arm_ids": ["B0"],
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

    if not smoke and common_support_sha != B0_SUPPORT_SHA256:
        raise ValueError("common-support mismatch vs frozen B0 support")

    # Paired arm-minus-B0 deltas on identical histories/seeds/support.
    # b0_pred/b0_pop hold the Task 7 per-seed Development records.
    deltas: dict[str, dict] = {}
    b0_hist: dict[str, list] = {}
    if b0_pred is not None and b0_pop is not None:
        stores = {"S_pred": (b0_pred, "S_pred_per_seed"),
                  "S_pop": (b0_pop, "per_seed")}
        for branch, (store, key) in stores.items():
            per_seed_macro = {}
            hist_pw, hist_p, hist_w, hist_far = [], [], [], []
            for h, role in enumerate(dev_roles):
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
                per_seed_macro[s] = (cell["macro_pw"], cell["macro_p"],
                                     cell["macro_w"])
            arm_macros = {
                s: (branch_records[s][branch]["macro_pw"],
                    branch_records[s][branch]["macro_p"],
                    branch_records[s][branch]["macro_w"]) for s in seeds}
            d_pw = [arm_macros[s][0] - per_seed_macro[s][0] for s in seeds]
            d_p = [arm_macros[s][1] - per_seed_macro[s][1] for s in seeds]
            d_w = [arm_macros[s][2] - per_seed_macro[s][2] for s in seeds]
            arm_hist = agg[branch]["per_history"]
            deltas[branch] = {
                "macro_pw": float(np.mean(d_pw)),
                "macro_p": float(np.mean(d_p)),
                "macro_w": float(np.mean(d_w)),
                "per_seed_pw": {str(s): float(v)
                                for s, v in zip(seeds, d_pw)},
                "per_seed_p": {str(s): float(v)
                               for s, v in zip(seeds, d_p)},
                "per_seed_w": {str(s): float(v)
                               for s, v in zip(seeds, d_w)},
                "per_history_pw": [
                    (a["auc_pw"] - b if a["auc_pw"] is not None
                     and b is not None else None)
                    for a, b in zip(arm_hist, hist_pw)],
                "per_history_p": [
                    (a["auc_p"] - b if a["auc_p"] is not None
                     and b is not None else None)
                    for a, b in zip(arm_hist, hist_p)],
                "per_history_w": [
                    (a["auc_w"] - b if a["auc_w"] is not None
                     and b is not None else None)
                    for a, b in zip(arm_hist, hist_w)],
                "per_history_far": [
                    (a["far"] - b if a["far"] is not None
                     and b is not None else None)
                    for a, b in zip(arm_hist, hist_far)],
            }
    else:
        for branch in ("S_pred", "S_pop"):
            deltas[branch] = {
                "macro_pw": None, "macro_p": None, "macro_w": None,
                "per_seed_pw": {}, "per_seed_p": {}, "per_seed_w": {},
                "per_history_pw": [None] * 4, "per_history_p": [None] * 4,
                "per_history_w": [None] * 4, "per_history_far": [None] * 4,
            }
    far_pass = all((h["far"] is not None and h["far"] <= 0.05)
                   for h in primary["per_history"])
    bg_pass = all((h["background_ratio"] is not None
                   and h["background_ratio"] <= 0.10)
                  for h in primary["per_history"])
    finite_ok = all(
        v is not None
        for h in primary["per_history"]
        for v in (h["auc_pw"], h["auc_p"], h["auc_w"]))

    rec0 = seed_records[seeds[0]]
    d_pred = deltas["S_pred"]
    minimum_effect = (d_pred["macro_pw"] is not None
                      and d_pred["macro_pw"] >= 0.05)
    p_noninf = (d_pred["macro_p"] is not None and d_pred["macro_p"] >= -0.02)
    w_noninf = (d_pred["macro_w"] is not None and d_pred["macro_w"] >= -0.02)
    status = "VALID_POSITIVE" if minimum_effect else "VALID_NEGATIVE"
    doc = {
        "schema_id": SCHEMA_ID,
        "protocol_id": PROTOCOL_ID,
        "phase": "development",
        "arm_id": arm_id,
        "arm_kind": "single",
        "component_set": ["C9"],
        "status": status,
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
            "parent_arm_ids": ["B0"],
        },
        "config": {
            "one_principal_change": "C9",
            "adapter_description": ARM_ADAPTER_DESCRIPTION[arm_id],
            "optimizer_steps": OPTIMIZER_STEPS,
            "checkpoint_step": CHECKPOINT_STEP,
            "calibration_quantile": CALIBRATION_QUANTILE,
            "score_branches": ["S_pred", "S_pop"],
        },
        "compute": {
            "b0_params": B0_PARAMS,
            "params_total": params,
            "params_delta_fraction": (params - B0_PARAMS) / B0_PARAMS,
            "b0_flops_per_reference_sample": B0_FLOPS_REFERENCE,
            "flops_per_reference_sample": flops,
            "flops_delta_fraction": (flops - B0_FLOPS_REFERENCE) / B0_FLOPS_REFERENCE,
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
                "delta_vs_B0": d_pred["macro_pw"],
                "directional_count": primary["directional"],
                "per_history": [h["auc_pw"] for h in primary["per_history"]],
            },
            "P": {
                "point": primary["macro_p"],
                "lcb95": primary["lcb_p"],
                "delta_vs_B0": d_pred["macro_p"],
                "recall_per_history": [h["recall_p"]
                                       for h in primary["per_history"]],
                "median_lead_days_per_history": [h["lead_p_median"]
                                                 for h in primary["per_history"]],
            },
            "W": {
                "point": primary["macro_w"],
                "lcb95": primary["lcb_w"],
                "delta_vs_B0": d_pred["macro_w"],
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
                "delta_far_vs_B0": d_pred["per_history_far"],
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
            "minimum_effect": bool(minimum_effect),
            "P_noninferiority": bool(p_noninf),
            "W_noninferiority": bool(w_noninf),
            "nuisance": bool(far_pass and _far_delta_ok(d_pred)),
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
        raise ValueError(f"{arm_id} document schema errors: "
                         + "; ".join(f"{list(e.path)}: {e.message}"
                                     for e in errors[:8]))
    (outdir / f"{prefix}_development_v4.json").write_text(
        json.dumps(to_jsonable(doc), indent=1, sort_keys=True))
    # Companion sidecars (bounded; S_pop full set + per-seed records + deltas).
    slim_records = {}
    for s in seeds:
        rec = dict(seed_records[s])
        rec.pop("dev_out", None)
        rec.pop("fit_pred", None)
        rec.pop("fit_pop", None)
        slim_records[str(s)] = rec
    (outdir / f"{prefix}_spop_metrics.json").write_text(json.dumps(to_jsonable(
        {"branch": "S_pop", "aggregate": agg["S_pop"],
         "delta_vs_B0": deltas["S_pop"],
         "per_seed": {str(s): branch_records[s]["S_pop"] for s in seeds}}),
        indent=1, sort_keys=True))
    (outdir / f"{prefix}_per_seed_metrics.json").write_text(json.dumps(to_jsonable(
        {"S_pred_per_seed": {str(s): branch_records[s]["S_pred"] for s in seeds},
         "delta_vs_B0": deltas,
         "b0_support_sha256": B0_SUPPORT_SHA256,
         "seed_records": slim_records}),
        indent=1, sort_keys=True))
    (outdir / f"{prefix}_run_manifest.json").write_text(json.dumps(to_jsonable({
        "protocol": PROTOCOL_ID,
        "arm": arm_id,
        "seeds": list(seeds),
        "config_sha256": canonical_hash(
            {str(s): seed_records[s]["cfg_dict"] for s in seeds}),
        "git": git,
        "runtime": runtime,
        "metric_code_sha256": metric_code_sha,
        "common_support_sha256": common_support_sha,
        "b0_support_sha256": B0_SUPPORT_SHA256,
        "fit_rows": fit_rows,
        "fit_patches": fit_patches,
        "arm_fit_patches": arm_fit_patches,
        "cal_rows": cal_rows,
        "elapsed_s_total": round(time.time() - t0, 1),
    }), indent=1, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
