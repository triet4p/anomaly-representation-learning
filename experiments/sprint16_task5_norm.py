"""Sprint 16 Task 5 (M1): conditional-normalization retention.

Compares pre-normalization vs post-production-normalization observable
probes on identical file support with the Task 3 contract reader
(``tie_auc`` + paired-bootstrap CI per history/category), holding
reader/support/aggregation fixed and isolating normalization only.

Accepted-path position (verified, not assumed): the accepted V2 production
pipeline contains NO external/conditional input-normalization stage —
both accepted checkpoint configs lack any norm flag and both state dicts
lack conditional-norm/BatchNorm statistics (hash-verified server-side;
raw inspection record in ``artifacts/sprint-16/task5-remote-inspection.md``).
The only normalization parameters anywhere in the accepted path are
parameter-only LayerNorm weights/biases INSIDE ``LocalPatchEncoder`` and
``SequenceContextEncoder`` (C3/C4 stages, measured by Tasks 7/8 — never
here). The driver asserts this structurally (fail closed) and routes
waveforms through the explicitly named verified-identity stage-1 operator.
R_pre and R_post are BOTH computed through the contract reader per
history/category; gaps are derived, never literal. A positive-control arm
(applied gain/offset on Fit-healthy files) proves the instrument would have
moved had any normalization existed. No causal BOTTLENECK claim (Task 16
owns verdicts).

Writes ``artifacts/sprint-16/task5-normalization.json`` and prints a summary
(recorded in ``artifacts/sprint-16/task-5.md``). Deterministic single run.
Sealed roots are never touched.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from representation import attribution_metrics as M

CONF = (("H-CONF-34", 1608), ("H-CONF-35", 1609), ("H-CONF-36", 1610),
        ("H-CONF-37", 1611))
FIT_SUBSET = ("H-FIT-28", 1604)
BASE = Path("data/generated/sprint15-v7")
OUT = Path("artifacts/sprint-16/task5-normalization.json")
CATEGORIES = ["P1", "P2", "W1", "W2", "A1", "A2", "P", "W", "A"]

FROZEN_BYTES = {
    "src/synth/probe15.py": "a08b3d5f",
    "src/synth/events.py": "85100f5e",
    "src/synth/chronicle.py": "b5992d6a",
}

#: Accepted-path files whose norm inventory is asserted below. LayerNorm is
#: permitted ONLY in the two encoder files (C3/C4); every other file must
#: contain no norm reference of any kind.
NORM_CHECK_FILES = [
    "src/representation/v2_inference.py",
    "src/representation/v2_config.py",
    "src/representation/v2_patch.py",
    "src/representation/layers/patch_encoder.py",
    "src/representation/layers/sequence_encoder.py",
    "src/synth/patchify.py",
]
ENCODER_FILES = {
    "src/representation/layers/patch_encoder.py",
    "src/representation/layers/sequence_encoder.py",
}


def check_static_norm_inventory() -> dict[str, object]:
    """Fail closed on any data-dependent or misplaced norm operator.

    Parses each accepted-path file and collects ``nn.XNorm`` uses plus any
    BatchNorm/ConditionalBatchNorm/running-stat reference. Requires: (a) zero
    ConditionalBatchNorm/BatchNorm/InstanceNorm/GroupNorm uses and zero
    running-stat buffers anywhere; (b) LayerNorm uses only (i) inside the two
    encoder files (C3/C4 stages), or (ii) inside
    ``ContextConditionedPatchEncoder.__init__`` in v2_patch.py — the
    parameter-only context-mixer/head norm over learned robot/program/regime
    embeddings (scorer/head territory, never waveform input). Every site is
    enumerated with file:line; frozen probe/events/chronicle digests must
    match Appendix A.
    """
    inventory: dict[str, list[str]] = {}
    for rel in NORM_CHECK_FILES:
        tree = ast.parse(Path(rel).read_text(encoding="utf-8"))
        found: list[str] = []
        stack: list[ast.AST] = [tree]
        parents: dict[int, ast.AST] = {}

        def visit(node: ast.AST, cls: str | None, func: str | None) -> None:
            for child in ast.iter_child_nodes(node):
                c, f = cls, func
                if isinstance(child, ast.ClassDef):
                    c = child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    f = child.name
                if isinstance(child, ast.Attribute) and child.attr.endswith("Norm"):
                    found.append(f"{rel}:{child.lineno}:{child.attr}@{c}.{f}")
                if isinstance(child, ast.Name) and child.id in (
                        "ConditionalBatchNorm", "BatchNorm1d", "BatchNorm2d"):
                    found.append(f"{rel}:{child.lineno}:{child.id}@{c}.{f}")
                visit(child, c, f)

        visit(tree, None, None)
        inventory[rel] = found
    allowed_v2_patch = "ContextConditionedPatchEncoder.__init__"
    for rel, found in inventory.items():
        for entry in found:
            site, where = entry.rsplit("@", 1)
            kind = site.rsplit(":", 1)[1]
            if kind != "LayerNorm":
                raise ValueError(f"non-LayerNorm operator: {entry}")
            if rel in ENCODER_FILES:
                continue
            if rel.endswith("v2_patch.py") and where == allowed_v2_patch:
                continue
            raise ValueError(f"misplaced norm operator: {entry}")
        text = Path(rel).read_text(encoding="utf-8")
        if "running_mean" in text or "num_batches_tracked" in text:
            raise ValueError(f"data-dependent norm state in {rel}")
    for rel, want in FROZEN_BYTES.items():
        got = hashlib.sha256(Path(rel).read_bytes()).hexdigest()[:8]
        if got != want:
            raise ValueError(f"frozen byte mismatch: {rel} {got} != {want}")
    layernorm_sites = [e for v in inventory.values() for e in v]
    if not layernorm_sites:
        raise ValueError("expected in-encoder LayerNorm sites, found none")
    return {"layernorm_sites": layernorm_sites,
            "conditional_or_batchnorm_sites": [],
            "frozen_bytes_ok": True}


def production_normalize(x: np.ndarray) -> np.ndarray:
    """Apply the accepted production stage-1 operator (verified identity).

    Justification: the accepted V2 pipeline performs no input normalization
    (see check_static_norm_inventory + remote inspection record). This
    function exists so the comparison routes through the named production
    stage explicitly; it MUST NOT gain parameters or branches without a
    protocol amendment.
    """
    return np.asarray(x, dtype=np.float64)


def load_root(role: str, group: str):
    """Load samples plus manifest for one materialized root (fail closed)."""
    from synth.chronicle import load_chronological

    root = BASE / group / role
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"missing root (refusing to proceed): {root}")
    samples, _ = load_chronological(root)
    manifest = json.loads((root / "manifest.json").read_text())
    return samples, manifest


def probe_features(sample_x: np.ndarray, row: dict) -> np.ndarray:
    """Frozen observable probe on one file waveform."""
    from synth import probe15 as P

    return P.extract_features(
        np.asarray(sample_x, dtype=np.float64),
        row["end_time"] - row["start_time"],
        row["end_time"] - row["last_reset_time"])


def family_index_sets(n_channels: int = 6) -> dict[str, dict]:
    """Map M1 probe families to frozen feature indices + coverage flags."""
    from synth import probe15 as P

    names = P.feature_names(n_channels)
    idx = {name: i for i, name in enumerate(names)}
    if len(names) != 59:
        raise ValueError(f"probe feature count changed: {len(names)}")
    ch = list(range(n_channels))
    bands = [n for n in names if "_band" in n]
    corr = [n for n in names if "_corr" in n]
    return {
        "level_gain": {"features": [f"ch{c}_rms" for c in ch]
                       + [f"ch{c}_std" for c in ch], "coverage": "full"},
        "slope_drift": {"features": [f"ch{c}_slope" for c in ch],
                        "coverage": "full"},
        "spectral": {"features": bands + [f"ch{c}_zcr" for c in ch],
                     "coverage": "full (zcr is a rate proxy, not a spectrum)"},
        "cross_channel_coupling": {"features": corr, "coverage": "full"},
        "phase_timing": {"features": ["duration_s", "time_since_reset_s"],
                         "coverage": "partial (timing metadata only; "
                                     "no direct phase feature exists)"},
        "localized_transients": {"features": [],
                                 "coverage": "none (no dedicated transient "
                                 "feature; zcr/bands respond indirectly)"},
        "persistent_degradation": {"features": [],
                                   "coverage": "none (no dedicated "
                                   "degradation feature; slope/rms respond "
                                   "indirectly)"},
        "benign_nuisance_controls": {"features": [],
                                     "coverage": "procedural (positive-"
                                     "control arm + shared control windows)"},
        "_index": idx,
    }


def auc_pair_ci(pos: list[float], neg: list[float]) -> dict[str, float]:
    """Contract-reader event AUROC with paired-bootstrap CI (frozen B/seed)."""
    if not pos or not neg:
        raise ValueError("auc_pair_ci requires non-empty pos and neg")
    scores = np.array(list(pos) + list(neg), dtype=np.float64)
    labels = np.array([1.0] * len(pos) + [0.0] * len(neg))
    if not np.isfinite(scores).all():
        raise ValueError("bootstrap inputs must be finite")
    rng = np.random.default_rng(M.BOOTSTRAP_SEED)
    draws = rng.integers(0, scores.size, size=(M.BOOTSTRAP_REPLICATES, scores.size))
    aucs = np.asarray([M.tie_auc(scores[row], labels[row]) for row in draws])
    aucs = aucs[np.isfinite(aucs)]
    if aucs.size == 0:
        raise ValueError("no finite bootstrap AUROC draws")
    return {
        "point": float(M.tie_auc(scores, labels)),
        "lcb": float(np.quantile(aucs, 0.025)),
        "ucb": float(np.quantile(aucs, 0.975)),
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def main() -> int:
    from synth import balanced as B
    from synth import events as E

    static_proof = check_static_norm_inventory()
    families = family_index_sets()
    index_of = families.pop("_index")

    # Frozen oracle reader (Task 4 pipeline): Fit-healthy standardization +
    # centroid, so R_pre/R_post use the identical reader as the ceiling.
    from synth import probe15 as P

    fit_feats = []
    for role, seed in (("H-FIT-28", 1604), ("H-FIT-29", 1605),
                       ("H-FIT-30", 1606)):
        samples, manifest = load_root(role, "FIT")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, f"tag:{role}"
        by_id = {s.file_id: s for s in samples}
        feats = []
        for row in manifest["files"]:
            if row["file_label"] != "normal" or row["is_quarantined"]:
                continue
            if (row["program_id"] == B.PROGRAM_RESERVE
                    or row["robot_id"] == B.ROBOT_RESERVE):
                continue
            feats.append(probe_features(
                np.asarray(by_id[row["file_id"]].x, dtype=np.float64), row))
        fit_feats.append(np.array(feats))
    fit_matrix = np.vstack(fit_feats)
    stats = P.standardize_fit(fit_matrix)
    centroid = P.fit_centroid(P.apply_standardization(fit_matrix, stats))

    def score_matrix(feats: np.ndarray) -> np.ndarray:
        return P.score_files(P.apply_standardization(feats, stats), centroid)

    per_history = []
    for role, seed in CONF:
        samples, manifest = load_root(role, "CONFIRMATION")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, f"tag:{role}"
        by_id = {s.file_id: s for s in samples}
        rows = manifest["files"]
        wins = manifest["maintenance_windows"]
        ledger = E.failure_ledger(manifest)
        pre_feats, post_feats, fids = [], [], []
        for row in rows:
            x = np.asarray(by_id[row["file_id"]].x, dtype=np.float64)
            pre = probe_features(x, row)
            post = probe_features(production_normalize(x), row)
            if pre.shape != post.shape:
                raise ValueError(f"probe shape changed: {role}")
            pre_feats.append(pre)
            post_feats.append(post)
            fids.append(row["file_id"])
        pre_feats = np.array(pre_feats)
        post_feats = np.array(post_feats)
        max_abs_diff = float(np.abs(pre_feats - post_feats).max())
        if max_abs_diff != 0.0:
            raise ValueError(f"identity violated: {role} {max_abs_diff}")
        fam_diffs = {}
        for fam, spec in families.items():
            cols = [index_of[n] for n in spec["features"]]
            fam_diffs[fam] = {
                "coverage": spec["coverage"],
                "n_features": len(cols),
                "max_abs_diff": (0.0 if not cols else float(
                    np.abs(pre_feats[:, cols] - post_feats[:, cols]).max())),
            }
        pre_scores = {f: float(v) for f, v in
                      zip(fids, score_matrix(pre_feats))}
        post_scores = {f: float(v) for f, v in
                       zip(fids, score_matrix(post_feats))}
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)

        def window_scores(score_map: dict[str, float]) -> tuple[dict, list]:
            neg = [E.window_score([m["file_id"] for m in w["members"]],
                                  score_map) for w in controls]

            def event_score(failure: dict) -> float | None:
                if E.positive_window_intersects_reset(failure, wins):
                    return None
                cands = E.pos_files(rows, failure, wins)
                if not cands:
                    return None
                return E.window_score([c["file_id"] for c in cands], score_map)
            return neg, event_score

        neg_pre, ev_pre = window_scores(pre_scores)
        neg_post, ev_post = window_scores(post_scores)
        cats: dict[str, dict] = {}
        for cat in CATEGORIES:
            cohort = cat[0] if len(cat) == 2 else cat
            subtype = cat if len(cat) == 2 else None
            pos_pre = [s for f in ledger
                       if f["cohort"] == cohort
                       and (subtype is None or f["subtype"] == subtype)
                       and (s := ev_pre(f)) is not None]
            pos_post = [s for f in ledger
                        if f["cohort"] == cohort
                        and (subtype is None or f["subtype"] == subtype)
                        and (s := ev_post(f)) is not None]
            if not pos_pre or not neg_pre:
                cats[cat] = {"reason": "no support",
                             "n_pos": len(pos_pre), "n_neg": len(neg_pre)}
                continue
            ci_pre = auc_pair_ci(pos_pre, neg_pre)
            ci_post = auc_pair_ci(pos_post, neg_post)
            cats[cat] = {
                "pre": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in ci_pre.items()},
                "post": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in ci_post.items()},
                "gap": round(ci_post["point"] - ci_pre["point"], 6),
            }
        per_history.append({
            "role": role, "seed": seed,
            "n_files_compared": len(rows),
            "max_abs_probe_difference": max_abs_diff,
            "bit_identical": True,
            "families": fam_diffs,
            "categories": cats,
        })

    samples, manifest = load_root(FIT_SUBSET[0], "FIT")
    assert manifest["seeds"]["health"] == FIT_SUBSET[1], "seed-match:fit"
    assert manifest.get("protocol") == B.S15_PROTOCOL_V7, "tag:fit"
    by_id = {s.file_id: s for s in samples}
    healthy_rows = [r for r in manifest["files"]
                    if r["file_label"] == "normal" and not r["is_quarantined"]]
    deltas = []
    for row in healthy_rows[:50]:
        x = np.asarray(by_id[row["file_id"]].x, dtype=np.float64)
        base = probe_features(x, row).mean()
        gained = probe_features(1.5 * x + 2.0, row).mean()
        deltas.append(abs(float(gained - base)))
    deltas = np.asarray(deltas)
    positive_control = {
        "n_files": int(deltas.size),
        "transform": "gain 1.5x + offset 2.0",
        "min_abs_mean_shift": float(deltas.min()),
        "median_abs_mean_shift": float(np.median(deltas)),
        "sensitive": bool((deltas > 0).all()),
    }
    if not positive_control["sensitive"]:
        raise ValueError("positive control failed: instrument insensitive")

    record = {
        "protocol": "experiments/sprint16-attribution-protocol-v3.md",
        "measurement": "M1",
        "static_proof": static_proof,
        "production_stage1_operator": "verified identity (no external/conditional input norm in accepted path)",
        "checkpoint_norm_inspection": "artifacts/sprint-16/task5-remote-inspection.md",
        "histories": per_history,
        "positive_control": positive_control,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True))
    print(json.dumps({
        "layernorm_sites": len(static_proof["layernorm_sites"]),
        "gaps": {h["role"]: {c: v.get("gap") for c, v in h["categories"].items()}
                 for h in per_history},
        "families_zero": all(
            f["max_abs_diff"] == 0.0
            for h in per_history for f in h["families"].values()),
        "positive_control": positive_control,
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
