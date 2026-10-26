#!/usr/bin/env python3
"""S24-GEO-TIME-v1 G2 diagnostic runner (tracked implementation, pre-review).

Tracked source path: experiments/sprint24_task7g2_diagnostic.py (this file).
Development lineage (ignored evidence, preserved): artifacts/sprint-24/evidence/s24-7-g2-a01/s24-g2-diagnostic.py.
Calibration-gate note (Main flag, binding on this runner): the §10 gate's
healthy-control check (post-hoc zero-annotation windows below the reference-only
q95) is a post-hoc validity assessment only. It never constructs a score or
threshold input: the MMD^2 q95 operating point is computed on reference
split-halves (seed 2407) before evaluation contact, with no evaluation-window,
label, or event input. When the gate fails the disposition is NO-CALIBRATION
with movement statistics standing and no event recall reported — never a
fallback threshold. Forward accounting: every frozen forward executed under
this runner (including the 8-file preflight smoke and the full 26,262-file
run) counts against the §6 compute cap; stored waveform shards are source
bytes, not embedding caches, and no hash-compatible embedding cache is reused
(T7 caches bind different window/reference support and a different identity).

Implements ONLY the locked S24-GEO-TIME-v1 contract
(docs/sprint24-geometry-time-diagnostic-v1.md, SHA-256
10c53edb30b0eaa257744bd5123f4fb6cbddb1b6e33f65b3afd8b8a005293f20):

- Preflight: exact source commit, accepted asset digests, frozen binding,
  Task 2 decile edges (c1/c2 per history), frozen state digests; manifest
  hashes must equal the accepted Task 7 lock values.
- Labels quarantined: window/reference manifests built from observable
  fields only via the frozen inherited eligibility predicate
  (is_eligible_for_training) + manifest maintenance windows. A provenance
  assertion fails closed if any annotation field is read during
  construction (checked by AST scan of eligible_observable + runtime guard).
- Capture: restore_s24_b0_bundle + capture_s24_file_features on the exact
  listed file IDs only (reference pool + eligible windows), frozen forward,
  per-file mask seeds under S24_MASK_RULE_ID. Missing-feature inference is
  bounded to re-running the frozen forward on listed IDs when no
  hash-compatible cache row exists; no imputation/interpolation/regeneration.
- Statistics: original 128-d space only. Per history x seed x window vs
  fixed reference: squared mean distance, standardized shift vector,
  log-trace ratio, median dim SD ratio, fixed-bandwidth Gaussian MMD^2
  (bandwidth = median reference pairwise distance, reference-only),
  KS distribution over 128 dims (median/IQR/max), k=10 NN reference
  fraction vs split-half null. Reference split-halves (seed 2407) give the
  null; MMD^2 q95 per history x seed is the prospective operating point.
- Controls: healthy windows (post-hoc zero abnormal + zero event
  membership), TV mix drift + last_reset recency covariates, HIGH-DRIFT
  top-decile subgroup, split-half null. Event mapping is post-hoc only:
  coverage rule (T-W,T]; detection delay clamped >= 0, non-recalled
  censored; false-alarm episodes = maximal crossing-window runs with no
  covered event, per 100 healthy-exposure days.
- Comparators: frozen B0 S_pred/S_pop + N4 S_pred file scores recomputed on
  identical window/file support (same coverage rule; frozen S18 q95
  thresholds only).
- Uncertainty: history-cluster bootstrap (B=2000, seed 2407) over the 4
  Development histories, descriptive given n=4.
- Compute cap: 4 histories x 3 seeds forward budget; abort INFEASIBLE if
  manifest support exceeds 2x the bound. Trainable params = 0.

Usage (remote, exact commit):
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
    python artifacts/sprint-24/evidence/s24-7-g2-a01/s24-g2-diagnostic.py \
    --out /home/trietlm/s24-7-g2-a01-runs --mode preflight|manifest|smoke|full

Modes:
  preflight: verify source/asset/binding/edges/frozen-state only, no capture.
  manifest:  preflight + build label-blind window/reference manifests.
  smoke:     manifest + bounded 8-file public-path capture + finite check.
  full:      manifest + all seeds/histories capture + statistics + controls +
             calibration + events + comparators + bootstrap + reports.

New implementation, not reviewed: never deploy remotely except through the
pre-deployment readiness review + Bronze operational checkpoint boundary.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import collections
import hashlib
import inspect
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from representation import sprint24_views as views  # noqa: E402
from representation.crn import ContextResidualScaleModel, load_n4_scale_checkpoint  # noqa: E402
from representation.sprint18_harness import (  # noqa: E402
    Sprint18ArmConfig,
    arm_config_sha256,
    load_task6_binding,
    validate_arm_config,
)
from representation.sprint24_features import (  # noqa: E402
    assert_s24_frozen_unchanged,
    capture_s24_file_features,
    restore_s24_b0_bundle,
    s24_device_identity,
    s24_feature_cache_identity,
    s24_live_state_digests,
    score_s24_n4_normalized,
)
from representation.sprint24_learners import enable_s24_deterministic_execution  # noqa: E402
from synth.dataset import load_sample_bytes  # noqa: E402

PROTOCOL = "S24-GEO-TIME-v1"
PROTOCOL_SHA256 = "10c53edb30b0eaa257744bd5123f4fb6cbddb1b6e33f65b3afd8b8a005293f20"
SOURCE_COMMIT = "f50f7c6de66181728153d4963c19db103b3372ff"
DEVICE = "cuda"
HISTORIES = tuple(f"S18I-ITER-0005-DEVELOPMENT-{i:02d}" for i in range(1, 5))
SEEDS = (181801, 181802, 181803)
BINS = {
    HISTORIES[0]: [0.0, 3225557.477, 6451114.954, 9676672.431, 12902229.909, 16127787.386, 19353344.863, 22578902.340, 25804459.817, 29030017.294, 32255574.772],
    HISTORIES[1]: [0.0, 3224340.604, 6448681.209, 9673021.813, 12897362.418, 16121703.022, 19346043.626, 22570384.231, 25794724.835, 29019065.440, 32243406.044],
    HISTORIES[2]: [0.0, 3225093.205, 6450186.410, 9675279.614, 12900372.819, 16125466.024, 19350559.229, 22575652.433, 25800745.638, 29025838.843, 32250932.048],
    HISTORIES[3]: [4641.356, 3228496.066, 6452350.777, 9676205.487, 12900060.198, 16123914.908, 19347769.618, 22571624.329, 25795479.039, 29019333.750, 32243188.460],
}
C1 = dict(zip(HISTORIES, [12902229.909, 12897362.418, 12900372.819, 12900060.198]))
C2 = dict(zip(HISTORIES, [22578902.340, 22570384.231, 22575652.433, 22571624.329]))
EXPECTED_CHECKPOINTS = {
    181801: "9292d7c59037cba7b7fd337dbb896ed492777abff4713307c98f86a3b5226a8f",
    181802: "8ae58da90818883ffa8b7dd7b0c73fadfcd5a24a0d3f2fc713c9a1d055b6621f",
    181803: "79edbdf37e4180368814e02b7201c7dd3f8a7b4c7bc47408bd9bbde5d059d408",
}
EXPECTED_BANK_FILE = {
    181801: "546e515d407f9bca3a0b9a15b78f281f9000231c27795585bee010f538d9387a",
    181802: "59b95f3c4897ce7ad35d75ce279c29e22d6338e10562627589b285be57a6b1bd",
    181803: "c234c0b637c46898f4edfad7802b3cf269ad0f7fc5487a28412c0772790fb235",
}
EXPECTED_BANK_TENSOR = {
    181801: "2d6e0a117feb7e332792d22e0ecd2d4ab64ef273290e9d9139f9e1efa2416750",
    181802: "d13f03945a80817c508dbe5654d785069eb8018a26a2b7dbd3f7671b9f08d40c",
    181803: "2a136229e19c98c9346c9217ebad71e54ef5e5b9e099a62d342270ac970fee01",
}
EXPECTED_N4 = {
    181801: "93a5524c312fd984132a7037bc9b7e54f145a0a012172ad79c278c88a5b8cd5a",
    181802: "09d8120ace481237a976cc3b1141acf45fe8f412de94559148205aabe09dcf8a",
    181803: "9626867ed8a454c6643cc96fe714d60b3a9e07da193f5eb2fb4544d706011c15",
}
EXPECTED_MANIFESTS = {
    HISTORIES[0]: "ce7c004b3ba7c6bca4187065df3299691c4d6247500eda480747e37b48fe1e08",
    HISTORIES[1]: "cee368b966874289e8b1b81980ebdf2db566501fe8e2864326651d73f6179310",
    HISTORIES[2]: "b1da957137936e01fda27b2af4194ea8f4be2cf611ce96315dba677145ae8f9e",
    HISTORIES[3]: "3af0811329031fc6f1ffc56ffa837d46a3b9b9095287128d08c46706b187f9af",
}
EXPECTED_BINDING = "383c62114ee6120d5fd19f866dfd585083d1d5899c9ffe91804761bd8a10a2d5"
EXPECTED_ARMS = {
    181801: "db83fed9ff129e0dc96a0c342aa400251f58d1febb2ced0ad10c849cee281973",
    181802: "d129c953181083709059ccead2d2e26a896ae88c0dd773d8c3c1f428bd69a4e5",
    181803: "67ae961a1db6ef3ba5187c2b1eecb3bd2f739b9362335e39718150addb5b197e",
}
FROZEN_S18_Q95 = {
    181801: {"S_pred": 0.13165724277496332, "S_pop": 5.98599042892456},
    181802: {"S_pred": 0.12028366327285757, "S_pop": 6.825963401794433},
    181803: {"S_pred": 0.10772727578878397, "S_pop": 6.811013221740723},
}
FROZEN_N4_Q95 = {181801: 1.7992383480072018, 181802: 1.7842277526855457, 181803: 1.6899769783020016}
WIDTH_S = 14 * 86400.0
STRIDE_S = 7 * 86400.0
MIN_N = 30
SPLIT_SEED = 2407
BOOT_B = 2000
K_NN = 10
FORBIDDEN_ROW_FIELDS = (
    "file_label", "anomaly_family", "anomaly_severity", "future_targets",
    "failure_id", "episode_id", "degradation_episode_id",
)
HANDOFF = REPO / "experiments/sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
LOCK = REPO / "experiments/sprint18-task14-config-lock.json"
DATA = Path("/home/trietlm/anomaly-representation-learning-s18t68-a02-worktree/data/generated/sprint18-iterative-v1/S18-ITER-0005-ASSESS-P-ALLOWANCE-V1")
ASSET = Path("/home/trietlm")
B0_ROOT = ASSET / "s18-t23-a04-runs/full"
BANK_ROOT = ASSET / "s18-t32-a02-runs/scores"
N4_ROOT = ASSET / "s18-t30-a14-runs/full"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha_obj(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def source_commit() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def eligible_observable(row, windows):
    """Frozen inherited eligibility on observable custody flags only.

    Reads ONLY quarantine/censor flags, robot/program reserve identity and
    manifest maintenance-window overlap. Never reads any annotation or
    episode/failure membership field (see module guard FORBIDDEN_ROW_FIELDS).
    """
    if row.get("is_quarantined") or row.get("is_censored"):
        return False
    if str(row.get("robot_id")) == "robot-08" or str(row.get("program_id")) == "program-03":
        return False
    if windows is not None and views._overlaps_maintenance(dict(row), dict(windows)):
        return False
    return True


def assert_label_blind_provenance() -> None:
    src = inspect.getsource(eligible_observable)
    body = "\n".join(
        line for line in src.splitlines()
        if line.strip() and not line.strip().startswith(("#", '"""', "'''", "@", "def "))
    )
    for field in FORBIDDEN_ROW_FIELDS:
        if field in body:
            raise RuntimeError(f"label-blind provenance violation: {field} read in eligibility body")
    tree = ast.parse(src)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for field in FORBIDDEN_ROW_FIELDS:
        if field in names:
            raise RuntimeError(f"label-blind provenance violation (AST): {field}")
    vsrc = inspect.getsource(views.is_eligible_for_training)
    if "file_label" in vsrc.replace("label check", ""):
        raise RuntimeError("inherited eligibility unexpectedly references file_label")


def device_provenance() -> dict:
    deterministic = enable_s24_deterministic_execution(DEVICE)
    if not deterministic["deterministic_algorithms"]:
        raise RuntimeError("deterministic CUDA algorithms were not enabled")
    if deterministic["cublas_workspace_config"] not in (":4096:8", ":16:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG is not the approved deterministic value")
    props = torch.cuda.get_device_properties(0)
    return {
        "requested": DEVICE, "identity": s24_device_identity(DEVICE),
        "deterministic": deterministic, "device_name": props.name,
        "cuda_build": torch.version.cuda, "parameter_dtype": "float32",
    }


def load_manifest_rows(hid: str):
    directory = DATA / "DEVELOPMENT" / hid
    mp = directory / "manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    return manifest, mp, directory


def check_history_manifest(hid: str):
    manifest, mp, directory = load_manifest_rows(hid)
    if sha(mp) != EXPECTED_MANIFESTS[hid]:
        raise RuntimeError(f"accepted Development manifest changed: {hid}")
    files = manifest["files"]
    if len({r["file_id"] for r in files}) != len(files):
        raise RuntimeError(f"duplicate file IDs in accepted manifest: {hid}")
    mw = manifest.get("maintenance_windows", {})
    ok = [r for r in files if eligible_observable(r, mw)]
    ref = [r for r in ok if float(r["end_time"]) <= C1[hid]]
    ev_spans = [float(r["end_time"]) for r in ok if float(r["end_time"]) > C1[hid]]
    tmax = max(float(r["end_time"]) for r in files)
    edges = BINS[hid]
    if len(edges) != 11 or edges[4] != C1[hid] or edges[7] != C2[hid]:
        raise RuntimeError(f"locked decile edges mismatch: {hid}")
    if len(ref) < 200:
        raise RuntimeError(f"reference pool below minimum: {hid} n={len(ref)}")
    return manifest, directory, files, mw, ok, ref, ev_spans, tmax


def build_window_manifest(hid: str, files, mw, ok, ref, ev_spans, tmax: float):
    ref_ends = sorted(float(r["end_time"]) for r in ref)
    by_id = {r["file_id"]: r for r in files}
    ref_record = {
        "history_id": hid, "n_ref": len(ref),
        "ref_end_max": max(ref_ends), "ref_end_min": min(ref_ends),
        "robot_mix": dict(sorted(collections.Counter(r["robot_id"] for r in ref).items())),
        "program_mix": dict(sorted(collections.Counter(str(r.get("program_id")) for r in ref).items())),
        "observable_inputs": ["start_time", "end_time", "robot_id", "program_id",
                              "last_reset_time", "is_quarantined", "is_censored",
                              "maintenance_windows", "reserve_identity_rule"],
        "annotations_excluded": list(FORBIDDEN_ROW_FIELDS),
        "support_sha256": sha_obj(sorted(r["file_id"] for r in ref)),
        "quarantine_excluded": sum(1 for r in files if r.get("is_quarantined")),
        "censor_excluded": sum(1 for r in files if r.get("is_censored")),
    }
    # Protocol §4 trailing windows (T-W,T] with W=14d>S=7d imply T1-W=c1-7d<c1,
    # so window 1 causally includes pre-c1 reference-pool files; the consistency
    # guard below therefore counts all eligible ends (not eval-span-only ends).
    # Overlap is observable-timestamp overlap only: reference construction keeps
    # zero evaluation contact and labels never enter construction (§3, §5).
    ok_sorted = sorted(float(r["end_time"]) for r in ok)
    windows = []
    unsupported = 0
    t = C1[hid] + STRIDE_S
    while t <= tmax:
        lo = t - WIDTH_S
        members = [r for r in ok if lo < float(r["end_time"]) <= t]
        n = len(members)
        assert n == bisect.bisect_right(ok_sorted, t) - bisect.bisect_right(ok_sorted, lo)
        status = "SUPPORTED" if n >= MIN_N else "UNSUPPORTED"
        unsupported += status == "UNSUPPORTED"
        windows.append({
            "history_id": hid, "window_end_T": t, "window_start": lo,
            "n_eligible": n,
            "n_ref_overlap": sum(1 for r in members if float(r["end_time"]) <= C1[hid]),
            "n_quarantine_excluded": sum(1 for r in files if r.get("is_quarantined") and lo < float(r["end_time"]) <= t),
            "n_overlap_excluded": 0,
            "robot_mix": dict(sorted(collections.Counter(r["robot_id"] for r in members).items())),
            "program_mix": dict(sorted(collections.Counter(str(r.get("program_id")) for r in members).items())),
            "status": status, "causal": True,
            "support_sha256": sha_obj(sorted(r["file_id"] for r in members)),
            "member_ids": sorted(r["file_id"] for r in members),
        })
        t += STRIDE_S
    ref_overlap_ids = sorted({fid for w in windows for fid in w["member_ids"]} & {r["file_id"] for r in ref})
    return {"protocol": PROTOCOL, "history_id": hid, "width_days": 14.0,
            "stride_days": 7.0, "min_n": MIN_N, "seed": SPLIT_SEED,
            "c1_s": C1[hid], "c2_s": C2[hid], "tmax_s": tmax,
            "device": DEVICE, "reference": ref_record, "windows": windows,
            "n_windows": len(windows), "n_unsupported": unsupported,
            "n_ref_overlap_any_window": len(ref_overlap_ids),
            "ref_overlap_file_ids": ref_overlap_ids,
            "label_blind_inputs_only": True}


def preflight_assets(expected_execution_commit: str) -> dict:
    actual = source_commit()
    if actual != expected_execution_commit:
        raise RuntimeError(f"execution commit mismatch: {actual} != {expected_execution_commit}")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain=v1"], text=True).strip():
        raise RuntimeError("exact source worktree is dirty")
    if sha(REPO / "docs/sprint24-geometry-time-diagnostic-v1.md") != PROTOCOL_SHA256:
        raise RuntimeError("locked diagnostic protocol bytes changed")
    assert_label_blind_provenance()
    device = device_provenance()
    binding = load_task6_binding(HANDOFF)
    if binding.executable_binding_sha256 != EXPECTED_BINDING:
        raise RuntimeError("accepted Sprint18 binding mismatch")
    lock_doc = json.loads(LOCK.read_text(encoding="utf-8"))
    arms = {}
    for seed in SEEDS:
        ck = B0_ROOT / f"seed-{seed}" / "checkpoints" / f"b0-seed{seed}-epoch100.pt"
        bk = BANK_ROOT / f"B0-FULL-{seed}.json.fit-bank" / "bank.pt"
        n4 = N4_ROOT / f"seed-{seed}" / "checkpoints" / f"n4-seed{seed}-final.pt"
        for path, digest in ((ck, EXPECTED_CHECKPOINTS[seed]), (bk, EXPECTED_BANK_FILE[seed]), (n4, EXPECTED_N4[seed])):
            if not path.is_file():
                raise RuntimeError(f"required accepted asset missing: {path}")
            if sha(path) != digest:
                raise RuntimeError(f"accepted asset digest mismatch: {path}")
        arm = validate_arm_config(Sprint18ArmConfig(**lock_doc["locked_configs"][f"B0-FULL/seed-{seed}"]), require_locked=True)
        if arm.model_seed != seed:
            raise RuntimeError(f"accepted B0 arm seed mismatch: {seed}")
        if arm_config_sha256(arm) != EXPECTED_ARMS[seed]:
            raise RuntimeError(f"accepted arm-config digest mismatch: {seed}")
        arms[seed] = arm
    return {"device": device, "binding": binding, "arms": arms}


def materialize(directory: Path, manifest, fids) -> dict:
    shard_of = {fid: sh["path"] for sh in manifest["shards"] for fid in sh["file_ids"]}
    payloads: dict[str, bytes] = {}
    for rel in sorted({shard_of[f] for f in fids}):
        with zipfile.ZipFile(directory / rel) as z:
            for f in fids:
                if shard_of[f] == rel:
                    payloads[f] = z.read(f + ".npz")
    if set(payloads) != set(fids):
        raise RuntimeError("waveform materialization lost support IDs")
    return {fid: load_sample_bytes(payloads[fid]) for fid in fids}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, choices=["preflight", "manifest", "smoke", "full"])
    ap.add_argument("--expected-execution-commit", required=True,
                    help="externally supplied expected execution commit (post-Bronze release); actual HEAD must match exactly")
    args = ap.parse_args()
    out = Path(args.out)
    expected = args.expected_execution_commit
    if not re.fullmatch(r"[0-9a-f]{40}", expected or ""):
        raise RuntimeError("expected execution commit must be a 40-char hex SHA")

    ctx = preflight_assets(expected)
    if SOURCE_COMMIT not in subprocess.check_output(["git", "-C", str(REPO), "log", "--format=%H", "--", "docs/sprint24-geometry-time-diagnostic-v1.md", "docs/sprint-plans/sprint-24.md"], text=True).split():
        raise RuntimeError("base provenance commit is not an ancestor of the execution checkout")
    per_history = {}
    for hid in HISTORIES:
        manifest, directory, files, mw, ok, ref, ev_spans, tmax = check_history_manifest(hid)
        per_history[hid] = {"manifest": manifest, "directory": directory, "files": files,
                            "mw": mw, "ok": ok, "ref": ref, "ev_spans": ev_spans, "tmax": tmax}
    n_ref_total = sum(len(v["ref"]) for v in per_history.values())
    n_win_files = sum(len(v["ev_spans"]) for v in per_history.values())
    forwards_per_seed = n_ref_total + n_win_files
    full_total = forwards_per_seed * 3
    smoke_total = 8 * len(HISTORIES) * len(SEEDS)
    locked_cap_total = 41000
    if full_total + smoke_total > locked_cap_total:
        raise RuntimeError(f"INFEASIBLE: planned forwards {full_total} + smoke {smoke_total} exceed locked cap {locked_cap_total}")
    preflight_record = {
        "protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": SOURCE_COMMIT, "runner": Path(__file__).name,
        "device": ctx["device"], "binding_sha256": EXPECTED_BINDING,
        "checkpoints": EXPECTED_CHECKPOINTS, "banks": EXPECTED_BANK_TENSOR,
        "bank_files": EXPECTED_BANK_FILE, "n4": EXPECTED_N4, "arms": EXPECTED_ARMS,
        "manifests": EXPECTED_MANIFESTS, "c1": C1, "c2": C2, "bins": BINS,
        "support": {hid: {"n_files": len(v["files"]), "n_eligible": len(v["ok"]),
                           "n_ref": len(v["ref"]), "n_eval": len(v["ev_spans"])}
                    for hid, v in per_history.items()},
        "forwards_per_seed": forwards_per_seed, "forwards_total_3seeds": full_total,
        "smoke_forwards_total": smoke_total, "grand_total_with_smoke": full_total + smoke_total,
        "compute_cap_per_seed": 13700, "compute_cap_total": locked_cap_total,
        "infeasible_guard": "INFEASIBLE when planned full + smoke forwards exceed the locked cap (no 2x expansion of the cap)",
    }
    if args.mode == "preflight":
        print(json.dumps({"status": "PREFLIGHT_OK", "forwards_per_seed": forwards_per_seed,
                           "total": full_total, "n_ref": n_ref_total,
                           "smoke_total": smoke_total, "grand_total": full_total + smoke_total}))
        out.mkdir(parents=True, exist_ok=True)
        (out / "preflight.json").write_text(json.dumps(preflight_record, sort_keys=True, indent=2), encoding="utf-8")
        return 0

    out.mkdir(parents=True, exist_ok=False)
    (out / "preflight.json").write_text(json.dumps(preflight_record, sort_keys=True, indent=2), encoding="utf-8")
    (out / "runner-sha256.txt").write_text(sha(Path(__file__)) + "\n", encoding="utf-8")
    (out / "state.json").write_text(json.dumps(
        {"state": "MANIFEST_STARTED", "protocol": PROTOCOL, "source_commit": SOURCE_COMMIT,
         "device": DEVICE, "mode": args.mode}, sort_keys=True, indent=2), encoding="utf-8")

    manifests = {}
    for hid in HISTORIES:
        v = per_history[hid]
        manifests[hid] = build_window_manifest(hid, v["files"], v["mw"], v["ok"], v["ref"], v["ev_spans"], v["tmax"])
    for hid, man in manifests.items():
        hdir = out / "manifests" / hid
        hdir.mkdir(parents=True)
        payload = canonical({k: val for k, val in man.items() if k != "windows"} | {"windows": man["windows"]}) + "\n"
        (hdir / "window-reference-manifest.json").write_text(payload, encoding="utf-8")
        (hdir / "manifest-sha256.txt").write_text(hashlib.sha256(payload.encode()).hexdigest() + "\n", encoding="utf-8")
        (hdir / "window-records.csv").write_text(
            "window_end_T,window_start,n_eligible,status,support_sha256\n" + "".join(
                f'{w["window_end_T"]},{w["window_start"]},{w["n_eligible"]},{w["status"]},{w["support_sha256"]}\n'
                for w in man["windows"]), encoding="utf-8")
    if args.mode == "manifest":
        print(json.dumps({"status": "MANIFEST_OK",
                           "windows": {h: m["n_windows"] for h, m in manifests.items()},
                           "unsupported": {h: m["n_unsupported"] for h, m in manifests.items()}}))
        return 0

    # ---- capture + statistics (smoke = bounded subset, full = everything) ----
    started = time.time()
    binding = ctx["binding"]
    arms = ctx["arms"]
    bundles = {}
    for seed in SEEDS:
        arm = arms[seed]
        bundles[seed] = restore_s24_b0_bundle(
            binding=binding, arm=arm,
            checkpoint_path=B0_ROOT / f"seed-{seed}" / "checkpoints" / f"b0-seed{seed}-epoch100.pt",
            bank_path=BANK_ROOT / f"B0-FULL-{seed}.json.fit-bank" / "bank.pt",
            expected_checkpoint_sha256=EXPECTED_CHECKPOINTS[seed],
            expected_bank_sha256=EXPECTED_BANK_TENSOR[seed], device=DEVICE)
    frozen_before = {s: s24_live_state_digests(b.model, b.bank) for s, b in bundles.items()}
    # support actually captured per history (full: ref + all SUPPORTED windows;
    # smoke: bounded 4-ref + 4-window subset of the first SUPPORTED window).
    capture_plan = {}
    for hid, man in manifests.items():
        ref_ids = [r["file_id"] for r in per_history[hid]["ref"]]
        first_supported = next((w for w in man["windows"] if w["status"] == "SUPPORTED"), None)
        win_ids = list(first_supported["member_ids"]) if first_supported else []
        ids = list(dict.fromkeys(ref_ids + [fid for w in man["windows"] if w["status"] == "SUPPORTED" for fid in w["member_ids"]]))
        if args.mode == "smoke":
            ids = (ref_ids[:4] + win_ids[:4]) or ids[:8]
        capture_plan[hid] = ids
    embeddings: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    file_scores: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    capture_walls = {}
    for hid in HISTORIES:
        v = per_history[hid]
        samples = materialize(v["directory"], v["manifest"], capture_plan[hid])
        by_id = {r["file_id"]: r for r in v["files"]}
        for seed in SEEDS:
            bundle = bundles[seed]
            t0 = time.perf_counter()
            cap = capture_s24_file_features(model=bundle.model, bank=bundle.bank,
                                            samples=[samples[f] for f in capture_plan[hid]],
                                            history_id=hid, model_seed=seed, device=DEVICE)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            if [c.file_id for c in cap] != capture_plan[hid]:
                raise RuntimeError("public feature capture changed input order")
            for c in cap:
                if not (torch.isfinite(c.embedding).all() and math.isfinite(c.s_pred) and math.isfinite(c.s_pop)):
                    raise RuntimeError(f"non-finite capture output: {hid}|{seed}|{c.file_id}")
            embeddings[(hid, seed)] = {c.file_id: c.embedding.detach().cpu().float().numpy() for c in cap}
            file_scores[(hid, seed)] = {c.file_id: {"S_pred": float(c.s_pred), "S_pop": float(c.s_pop)} for c in cap}
            # N4 comparator on identical support (frozen S18 q95 only)
            n4saved = torch.load(N4_ROOT / f"seed-{seed}" / "checkpoints" / f"n4-seed{seed}-final.pt", map_location="cpu", weights_only=True)
            scale = ContextResidualScaleModel(bundle.model.config.d_model)
            load_n4_scale_checkpoint(N4_ROOT / f"seed-{seed}" / "checkpoints" / f"n4-seed{seed}-final.pt",
                                     scale, expected_fit_scope="final",
                                     expected_training_history_ids=["S18I-ITER-0005-FIT-01", "S18I-ITER-0005-FIT-02", "S18I-ITER-0005-FIT-03"])
            scale = scale.to(device=torch.device(bundle.device), dtype=torch.float32)
            scale.eval()
            triples = score_s24_n4_normalized(model=bundle.model, scale_model=scale,
                                              samples=[samples[f] for f in capture_plan[hid]],
                                              history_id=hid, model_seed=seed,
                                              v_floor=float(n4saved["v_floor"]), v_cap=float(n4saved["v_cap"]),
                                              device=DEVICE)
            for fid, t in triples.items():
                file_scores[(hid, seed)][fid]["S_pred_N4"] = float(t["normalized"])
            capture_walls[f"{hid}|{seed}"] = wall
    for seed, bundle in bundles.items():
        assert_s24_frozen_unchanged(frozen_before[seed], s24_live_state_digests(bundle.model, bundle.bank))
    # frozen cache-identity bindings per (history, seed)
    cache_bindings = {}
    for hid in HISTORIES:
        for seed in SEEDS:
            bundle = bundles[seed]
            sup = sha_obj(sorted(capture_plan[hid]))
            cache_bindings[f"{hid}|{seed}"] = s24_feature_cache_identity(
                checkpoint_sha256=EXPECTED_CHECKPOINTS[seed], bank_sha256=EXPECTED_BANK_TENSOR[seed],
                support_sha256=sup, preprocessing_sha256=bundle.preprocessing_sha256,
                arm_config_sha256=EXPECTED_ARMS[seed], data_binding_sha256=EXPECTED_BINDING,
                model_seed=seed, scale_state_sha256=None, device=DEVICE)

    rng = np.random.default_rng(SPLIT_SEED)
    stats_rows, null_rows, control_rows, event_rows, comp_rows = [], [], [], [], []
    for hid in HISTORIES:
        man = manifests[hid]
        by_id = {r["file_id"]: r for r in per_history[hid]["files"]}
        ref_ids = [r["file_id"] for r in per_history[hid]["ref"] if r["file_id"] in embeddings[(hid, SEEDS[0])]]
        if args.mode == "full" and len(ref_ids) != len(per_history[hid]["ref"]):
            raise RuntimeError(f"full-mode capture missed reference files: {hid}")
        for seed in SEEDS:
            E = embeddings[(hid, seed)]
            R = np.stack([E[f] for f in ref_ids]).astype(np.float64)
            mu_r = R.mean(axis=0)
            sd_r = R.std(axis=0, ddof=0)
            tr_r = float(np.trace(np.cov(R, rowvar=False)))
            # fixed Gaussian bandwidth: median reference pairwise distance
            d2 = ((R[:400, None, :] - R[None, :400, :]) ** 2).sum(-1)
            iu = np.triu_indices(min(400, len(R)), k=1)
            bw = float(np.median(np.sqrt(d2[iu])))
            if not math.isfinite(bw) or bw <= 0:
                raise RuntimeError(f"non-finite kernel bandwidth: {hid}|{seed}")
            # split-half null (seed 2407): halves + resamples
            perm = rng.permutation(len(ref_ids))
            half = len(ref_ids) // 2
            A = R[perm[:half]]
            B = R[perm[half:2 * half]]
            null_mmds = [float(_mmd2(A, B, bw))]
            n_resample = 49 if args.mode == "smoke" else 200
            for _ in range(n_resample):
                p = rng.permutation(len(R))
                null_mmds.append(float(_mmd2(R[p[:half]], R[p[half:2 * half]], bw)))
            q95 = float(np.quantile(null_mmds, 0.95))
            K = min(K_NN, half - 1, len(ref_ids) - half)
            for w in man["windows"]:
                if args.mode == "smoke" and w["window_end_T"] != man["windows"][0]["window_end_T"]:
                    continue
                if w["status"] == "UNSUPPORTED" or not set(w["member_ids"]) <= set(E):
                    stats_rows.append({"history_id": hid, "encoder_seed": seed,
                                       "window_end_T": w["window_end_T"], "status": "UNSUPPORTED",
                                       "n_eligible": w["n_eligible"],
                                       "note": "below-minimum or uncaptured-in-smoke-mode"})
                    continue
                Wm = np.stack([E[f] for f in w["member_ids"]]).astype(np.float64)
                mu_w = Wm.mean(axis=0)
                loc = float(((mu_w - mu_r) ** 2).sum())
                with np.errstate(divide="ignore", invalid="ignore"):
                    shift = np.where(sd_r > 0, (mu_w - mu_r) / sd_r, np.nan)
                tr_w = float(np.trace(np.cov(Wm, rowvar=False))) if len(Wm) > 1 else float("nan")
                logtr = float(math.log(tr_w / tr_r)) if math.isfinite(tr_w) and tr_r > 0 and tr_w > 0 else float("nan")
                sd_w = Wm.std(axis=0, ddof=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    med_sdr = float(np.nanmedian(sd_w / np.where(sd_r > 0, sd_r, np.nan)))
                mmd = float(_mmd2(Wm, R, bw))
                ks = sorted(float(_ks_1d(Wm[:, d], R[:, d])) for d in range(R.shape[1]))
                nn = float(_nn_ref_fraction(Wm, R, K))
                halfA = float(_nn_ref_fraction(A, np.vstack([A, B]), K))
                stats_rows.append({
                    "history_id": hid, "encoder_seed": seed, "window_end_T": w["window_end_T"],
                    "status": "SUPPORTED", "n_eligible": w["n_eligible"], "n_ref": len(ref_ids),
                    "mean_sq_dist": loc, "shift_norm": float(np.sqrt(np.nansum(shift ** 2))),
                    "log_trace_ratio": logtr, "median_sd_ratio": med_sdr,
                    "mmd2": mmd, "kernel_bandwidth": bw,
                    "ks_median": float(np.median(ks)), "ks_iqr": float(np.percentile(ks, 75) - np.percentile(ks, 25)),
                    "ks_max": float(max(ks)), "nn_ref_fraction": nn,
                    "null_halfA_ref_fraction": halfA, "cache_identity": cache_bindings[f"{hid}|{seed}"],
                    "support_sha256": w["support_sha256"], "calibration_q95": q95,
                    "mmd_crosses_q95": bool(mmd >= q95),
                })
                # post-hoc controls only (labels/events read here, never in construction)
                ab = sum(1 for f in w["member_ids"] if by_id[f].get("file_label") == "abnormal")
                covered = _events_covered(man, per_history[hid]["manifest"], by_id, w)
                mix_r = _mix(ref_ids, by_id)
                mix_w = _mix(w["member_ids"], by_id)
                tv = 0.5 * sum(abs(mix_w.get(k, 0) - mix_r.get(k, 0)) for k in set(mix_r) | set(mix_w))
                resets = [w["window_end_T"] - float(by_id[f].get("last_reset_time", w["window_end_T"]))
                          for f in w["member_ids"] if by_id[f].get("last_reset_time") is not None]
                control_rows.append({
                    "history_id": hid, "encoder_seed": seed, "window_end_T": w["window_end_T"],
                    "n_abnormal_posthoc": ab, "n_covered_events_posthoc": len(covered),
                    "is_healthy_control": bool(ab == 0 and len(covered) == 0),
                    "robot_program_tv_vs_ref": tv,
                    "median_reset_recency_s": float(np.median(resets)) if resets else None,
                })
            null_rows.append({"history_id": hid, "encoder_seed": seed, "bandwidth": bw,
                              "q95": q95, "n_resamples": len(null_mmds),
                              "null_median": float(np.median(null_mmds)),
                              "finite_nondegenerate": bool(np.isfinite(null_mmds).all() and np.std(null_mmds) > 0)})
            # event mapping post-hoc: coverage rule (T-W,T] on failure_time, over evaluated windows only
            evaluated_ends = {r["window_end_T"] for r in stats_rows if r["history_id"] == hid and r["encoder_seed"] == seed and r.get("status") == "SUPPORTED"}
            for ev in per_history[hid]["manifest"].get("failure_events", []):
                ft = float(ev["failure_time"])
                covering = [w for w in man["windows"] if w["status"] == "SUPPORTED" and w["window_start"] < ft <= w["window_end_T"] and w["window_end_T"] in evaluated_ends]
                srows = [r for r in stats_rows if r["history_id"] == hid and r["encoder_seed"] == seed
                         and r.get("status") == "SUPPORTED"
                         and r["window_end_T"] in {w["window_end_T"] for w in covering}]
                crossed = [r for r in srows if r["mmd_crosses_q95"]]
                onset = ev.get("degradation_onset")
                start = float(onset) if onset is not None else ft
                if crossed:
                    delay = max(0.0, min(r["window_end_T"] for r in crossed) - start)
                    recalled = True
                else:
                    delay, recalled = None, False
                event_rows.append({
                    "history_id": hid, "encoder_seed": seed, "failure_id": ev.get("failure_id"),
                    "robot_id": ev.get("robot_id"), "failure_time": ft, "onset_s": start,
                    "n_covering_windows": len(covering), "recalled": recalled,
                    "detection_delay_s": delay,  # censored (excluded from delay) when not recalled
                })
            # comparator event summaries on identical support (frozen q95 only)
            for comp, key in (("M0a-S_pred", "S_pred"), ("M0b-S_pop", "S_pop"), ("M0c-N4-S_pred", "S_pred_N4")):
                thr = (FROZEN_S18_Q95 if key != "S_pred_N4" else FROZEN_N4_Q95)[seed] if key != "S_pop" else FROZEN_S18_Q95[seed]["S_pop"]
                sc = {f: file_scores[(hid, seed)][f][key] for f in capture_plan[hid]}
                rec, n, delays, alarms = _comparator_events(per_history[hid]["manifest"], by_id, man, sc, thr)
                comp_rows.append({"history_id": hid, "encoder_seed": seed, "method": comp,
                                  "threshold": thr, "event_recall": rec, "n_events": n,
                                  "delay_median_s": float(np.median(delays)) if delays else None,
                                  "delay_iqr_s": float(np.percentile(delays, 75) - np.percentile(delays, 25)) if delays else None,
                                  "false_alarm_episodes": alarms})
    # calibration gate: finite non-degenerate nulls + >= half healthy controls below q95
    healthy_below, healthy_total = 0, 0
    for c in control_rows:
        if c["is_healthy_control"]:
            s = next((r for r in stats_rows if r["history_id"] == c["history_id"]
                      and r["encoder_seed"] == c["encoder_seed"]
                      and r.get("window_end_T") == c["window_end_T"]
                      and r.get("status") == "SUPPORTED"), None)
            if s is not None:
                healthy_total += 1
                healthy_below += (not s["mmd_crosses_q95"])
    gate_ok = all(n["finite_nondegenerate"] for n in null_rows) and (healthy_total == 0 or healthy_below >= healthy_total / 2)
    disposition = "CALIBRATED" if gate_ok else "NO-CALIBRATION"
    # false-alarm episodes: maximal crossing-window runs with no covered event
    fa_rows = []
    for hid in HISTORIES:
        for seed in SEEDS:
            srows = sorted([r for r in stats_rows if r["history_id"] == hid and r["encoder_seed"] == seed
                            and r.get("status") == "SUPPORTED"], key=lambda r: r["window_end_T"])
            runs = 0
            prev_end = None
            for r in srows:
                c = next(cc for cc in control_rows if cc["history_id"] == hid and cc["encoder_seed"] == seed
                         and cc["window_end_T"] == r["window_end_T"])
                if r["mmd_crosses_q95"] and c["n_covered_events_posthoc"] == 0:
                    if prev_end is None or r["window_end_T"] - prev_end > STRIDE_S * 1.5:
                        runs += 1
                    prev_end = r["window_end_T"]
                else:
                    prev_end = None if not r["mmd_crosses_q95"] else prev_end
            fa_rows.append({"history_id": hid, "encoder_seed": seed, "false_alarm_episodes": runs,
                            "disposition": disposition})
    # history-cluster bootstrap (descriptive, n=4) over paired (history, seed) means
    boot = {}
    for seed in SEEDS:
        vals = []
        for hid in HISTORIES:
            v = [r["mmd2"] for r in stats_rows if r["history_id"] == hid and r["encoder_seed"] == seed
                 and r.get("status") == "SUPPORTED"]
            vals.append(float(np.mean(v)) if v else float("nan"))
        bvals = []
        hrng = np.random.default_rng(SPLIT_SEED)
        for _ in range(BOOT_B):
            idx = hrng.integers(0, 4, 4)
            bvals.append(float(np.nanmean([vals[i] for i in idx])))
        boot[str(seed)] = {"mean_mmd2": float(np.nanmean(vals)),
                            "ci_lo": float(np.percentile(bvals, 2.5)),
                            "ci_hi": float(np.percentile(bvals, 97.5)),
                            "n_histories": 4, "n_resamples": BOOT_B}

    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    (tables / "window-statistics.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in stats_rows), encoding="utf-8")
    (tables / "null-calibration.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in null_rows), encoding="utf-8")
    (tables / "controls.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in control_rows), encoding="utf-8")
    (tables / "event-mapping.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in event_rows), encoding="utf-8")
    (tables / "comparator-events.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in comp_rows), encoding="utf-8")
    (tables / "false-alarms.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in fa_rows), encoding="utf-8")
    (tables / "bootstrap.json").write_text(json.dumps(boot, sort_keys=True, indent=2), encoding="utf-8")
    actual_forwards = sum(len(ids) for ids in capture_plan.values()) * len(SEEDS)
    costs = {"trainable_parameters": 0, "forwards_per_seed": forwards_per_seed,
             "forwards_total": full_total, "forwards_actual_this_run": actual_forwards,
             "smoke_forwards_total": smoke_total, "cap_total": locked_cap_total,
             "capture_wall_s": capture_walls, "total_wall_s": time.time() - started,
             "device": DEVICE, "mode": args.mode}
    (out / "costs.json").write_text(json.dumps(costs, sort_keys=True, indent=2), encoding="utf-8")
    (out / "calibration.json").write_text(json.dumps(
        {"disposition": disposition, "gate_finite_nondegenerate": all(n["finite_nondegenerate"] for n in null_rows),
         "healthy_below": healthy_below, "healthy_total": healthy_total}, sort_keys=True, indent=2), encoding="utf-8")
    (out / "state.json").write_text(json.dumps(
        {"state": "COMPLETE", "protocol": PROTOCOL, "source_commit": SOURCE_COMMIT,
         "device": DEVICE, "mode": args.mode, "disposition": disposition,
         "runner_sha256": sha(Path(__file__))}, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "mode": args.mode, "disposition": disposition,
                       "n_stats": len(stats_rows), "n_events": len(event_rows),
                       "forwards_actual": actual_forwards, "forwards_full_planned": full_total,
                       "wall_s": round(time.time() - started, 1)}))
    return 0


def _mmd2(X: np.ndarray, Y: np.ndarray, bw: float) -> float:
    def k(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * bw * bw))
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def _ks_1d(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), x, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), x, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _nn_ref_fraction(W: np.ndarray, R: np.ndarray, k: int) -> float:
    P = np.vstack([W, R])
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    is_ref = np.zeros(len(P), dtype=bool)
    is_ref[len(W):] = True
    hits = is_ref[nn[:len(W)]].sum(axis=1)
    return float((hits / k).mean())


def _mix(ids, by_id) -> dict:
    from collections import Counter
    c = Counter((str(by_id[f]["robot_id"]), str(by_id[f].get("program_id"))) for f in ids)
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def _events_covered(man, manifest, by_id, w) -> list:
    lo, hi = w["window_start"], w["window_end_T"]
    return [ev for ev in manifest.get("failure_events", []) if lo < float(ev["failure_time"]) <= hi]


def _comparator_events(manifest, by_id, man, scores, thr: float):
    merged: dict[str, list] = collections.defaultdict(list)
    ep = {e["episode_id"]: e for e in manifest.get("episodes", [])}
    for ev in manifest.get("failure_events", []):
        onset = ev.get("degradation_onset")
        start = float(onset) if onset is not None else float(ev["failure_time"])
        m = ep.get(ev.get("maintenance_episode_id"))
        end = float(m["end_time"]) if m and m.get("end_time") is not None else float(ev["failure_time"])
        if end < start:
            end = float(ev["failure_time"])
        merged[str(ev["robot_id"])].append((start, end))
    for robot in merged:
        spans = sorted(merged[robot])
        out = []
        for s, e in spans:
            if out and s <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        merged[robot] = out
    sup_ids = [fid for w in man["windows"] if w["status"] == "SUPPORTED" for fid in w["member_ids"]]
    sup = [by_id[f] for f in sup_ids]
    dur = (max(float(r["end_time"]) for r in sup) - min(float(r["start_time"]) for r in sup)) / 86400.0
    rec, delays, n = 0, [], 0
    for ev in manifest.get("failure_events", []):
        onset = ev.get("degradation_onset")
        start = float(onset) if onset is not None else float(ev["failure_time"])
        m = ep.get(ev.get("maintenance_episode_id"))
        end = float(m["end_time"]) if m and m.get("end_time") is not None else float(ev["failure_time"])
        if end < start:
            end = float(ev["failure_time"])
        robot = str(ev["robot_id"])
        eids = [fid for fid in sup_ids if str(by_id[fid]["robot_id"]) == robot and start <= float(by_id[fid]["end_time"]) <= end]
        if not eids:
            continue
        n += 1
        hits = [(float(by_id[f]["end_time"]), f) for f in eids if scores.get(f, float("-inf")) >= thr]
        if hits:
            rec += 1
            delays.append(max(0.0, min(hits)[0] - start))
    alarms = [(str(by_id[f]["robot_id"]), float(by_id[f]["end_time"])) for f in sup_ids
              if scores.get(f, float("-inf")) >= thr
              and not any(lo <= float(by_id[f]["end_time"]) <= hi for lo, hi in merged.get(str(by_id[f]["robot_id"]), []))]
    runs, last = 0, {}
    for robot, t in sorted(alarms):
        if robot not in last or t - last[robot] >= 86400.0:
            runs += 1
        last[robot] = t
    return (rec / n if n else None), n, delays, runs


if __name__ == "__main__":
    raise SystemExit(main())
