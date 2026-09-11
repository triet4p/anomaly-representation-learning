"""Sprint 16 Task 4: raw/handcrafted oracle ceiling on candidate-7 roles.

Reconfirms observable signal per failure/anomaly category with the Task 3
shared evaluator on identical support: frozen Sprint 15 probe features
(Fit-healthy standardization + centroid, Calibration frozen threshold),
per-category event-window AUROC with paired-bootstrap CI, severity ordering,
unaffected-background stability, and background FPR description. Applies the
protocol G1 floor per category; absent/non-measurable signal is reported as
a data/physics limitation, never a representation failure. No production,
early-warning, risk, or bottleneck claim. Sealed roots are never touched.

Writes ``artifacts/sprint-16/oracle-ceiling.json`` and prints a summary
(recorded in ``artifacts/sprint-16/task-4.md``). Deterministic single run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from representation import attribution_metrics as M
FIT = (("H-FIT-28", 1604), ("H-FIT-29", 1605), ("H-FIT-30", 1606))
CAL = ("H-CAL-10", 1607)
CONF = (("H-CONF-34", 1608), ("H-CONF-35", 1609), ("H-CONF-36", 1610),
        ("H-CONF-37", 1611))
BASE = Path("data/generated/sprint15-v7")
OUT = Path("artifacts/sprint-16/oracle-ceiling.json")
CATEGORIES = ["P1", "P2", "W1", "W2", "A1", "A2", "P", "W", "A"]
FROZEN_THRESHOLD = M.FROZEN_THRESHOLD
#: Full-precision accepted value behind the frozen `138.03` display rounding
#: (`artifacts/sprint-15/observable-confirmation-cycle-7.json` calibration).
RECORDED_THRESHOLD = 138.0340508850525
#: Frozen input bytes (v3 Appendix A, 8-hex prefixes). Checked pre-run;
#: any mismatch aborts before any measurement.
FROZEN_BYTES = {
    "src/synth/probe15.py": "a08b3d5f",
    "src/synth/events.py": "85100f5e",
    "src/synth/balanced.py": "d62de342",
    "src/synth/chronicle.py": "b5992d6a",
    "src/synth/config.py": "55fed2ef",
    "src/synth/health.py": "e5854641",
}

def load_root(role: str, group: str):
    """Load samples plus manifest for one materialized root (fail closed)."""
    from synth.chronicle import load_chronological

    root = BASE / group / role
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"missing root (refusing to proceed): {root}")
    samples, _ = load_chronological(root)
    manifest = json.loads((root / "manifest.json").read_text())
    return samples, manifest


def healthy_features(samples, manifest):
    """Features of verified-healthy, non-reserve files (Fit/Cal inputs)."""
    from synth import balanced as B
    from synth import probe15 as P

    by_id = {s.file_id: s for s in samples}
    feats, ids = [], []
    for row in manifest["files"]:
        if row["file_label"] != "normal" or row["is_quarantined"]:
            continue
        if (row["program_id"] == B.PROGRAM_RESERVE
                or row["robot_id"] == B.ROBOT_RESERVE):
            continue
        sample = by_id[row["file_id"]]
        feats.append(P.extract_features(
            np.asarray(sample.x, dtype=np.float64),
            row["end_time"] - row["start_time"],
            row["end_time"] - row["last_reset_time"]))
        ids.append(row["file_id"])
    return np.array(feats), ids


def auc_bootstrap_ci(pos: list[float], neg: list[float]) -> dict[str, float]:
    """Paired-bootstrap CI for event AUROC (frozen B/seed convention).

    Resamples (score, label) pairs jointly with
    ``np.random.default_rng(M.BOOTSTRAP_SEED)``; B = M.BOOTSTRAP_REPLICATES.
    Same convention as the Task 3 contract, scoped to paired AUROC inputs.
    """
    if not pos or not neg:
        raise ValueError("auc_bootstrap_ci requires non-empty pos and neg")
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


def check_frozen_bytes() -> None:
    """Abort unless every frozen input matches its Appendix-A digest."""
    import hashlib

    for rel, want in FROZEN_BYTES.items():
        got = hashlib.sha256(Path(rel).read_bytes()).hexdigest()[:8]
        if got != want:
            raise ValueError(f"frozen byte mismatch: {rel} {got} != {want}")


def main() -> int:
    from synth import balanced as B
    from synth import probe15 as P

    check_frozen_bytes()
    fit_feats = []
    for role, seed in FIT:
        samples, manifest = load_root(role, "FIT")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, f"tag:{role}"
        feats, _ = healthy_features(samples, manifest)
        fit_feats.append(feats)
    fit_matrix = np.vstack(fit_feats)
    stats = P.standardize_fit(fit_matrix)
    # Bind Task 3 provenance: same bytes through the contract standardizer.
    contract_std = M.FrozenStandardizer.fit(fit_matrix, source="H-FIT-28..30")
    assert np.array_equal(
        P.apply_standardization(fit_matrix, stats),
        contract_std.apply(fit_matrix)), "standardizer-divergence"
    token = contract_std.token()
    centroid = P.fit_centroid(P.apply_standardization(fit_matrix, stats))
    fit_scores = P.score_files(P.apply_standardization(fit_matrix, stats), centroid)

    cal_role, cal_seed = CAL
    cal_samples, cal_manifest = load_root(cal_role, "CALIBRATION")
    assert cal_manifest["seeds"]["health"] == cal_seed, "seed-match:cal"
    assert cal_manifest.get("protocol") == B.S15_PROTOCOL_V7, "tag:cal"
    cal_feats, cal_ids = healthy_features(cal_samples, cal_manifest)
    n_fit, n_cal = int(fit_matrix.shape[0]), int(len(cal_ids))
    threshold = P.select_threshold(P.score_files(
        P.apply_standardization(cal_feats, stats), centroid))
    if abs(float(threshold) - RECORDED_THRESHOLD) > 1e-9:
        raise ValueError(
            f"threshold {threshold} != recorded {RECORDED_THRESHOLD}")

    from synth import events as E

    per_history = []
    for role, seed in CONF:
        samples, manifest = load_root(role, "CONFIRMATION")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, f"tag:{role}"
        by_id = {s.file_id: s for s in samples}
        file_scores: dict[str, float] = {}
        for row in manifest["files"]:
            sample = by_id[row["file_id"]]
            feat = P.extract_features(
                np.asarray(sample.x, dtype=np.float64),
                row["end_time"] - row["start_time"],
                row["end_time"] - row["last_reset_time"])
            file_scores[row["file_id"]] = float(
                P.score_files(P.apply_standardization(feat[None, :], stats),
                              centroid)[0])
        rows = manifest["files"]
        wins = manifest["maintenance_windows"]
        ledger = E.failure_ledger(manifest)
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)
        neg = [E.window_score([m["file_id"] for m in w["members"]], file_scores)
               for w in controls]

        def event_score(failure: dict) -> float | None:
            if E.positive_window_intersects_reset(failure, wins):
                return None
            cands = E.pos_files(rows, failure, wins)
            if not cands:
                return None
            return E.window_score([c["file_id"] for c in cands], file_scores)

        cats: dict[str, dict] = {}
        for cat in CATEGORIES:
            if len(cat) == 2:
                cohort, subtype = cat[0], cat
            else:
                cohort, subtype = cat, None
            pos = [s for f in ledger
                   if f["cohort"] == cohort
                   and (subtype is None or f["subtype"] == subtype)
                   and (s := event_score(f)) is not None]
            if not pos or not neg:
                cats[cat] = {"measurable": False,
                             "reason": "no support",
                             "n_pos": len(pos), "n_neg": len(neg)}
                continue
            ci = auc_bootstrap_ci(pos, neg)
            g1 = ci["point"] >= M.ORACLE_FLOOR and ci["lcb"] > M.ORACLE_LCB_FLOOR
            cats[cat] = {"measurable": bool(g1), **{k: round(v, 4) if isinstance(v, float) else v for k, v in ci.items()}}

        sev: dict[str, dict] = {}
        for cohort in ("P", "W"):
            pts = [(f["severity"], s) for f in ledger
                   if f["cohort"] == cohort
                   and (s := event_score(f)) is not None]
            if len(pts) < 3:
                sev[cohort] = {"rho": None, "reason": "no support"}
                continue
            lv = np.array([p[0] for p in pts])
            sc = np.array([p[1] for p in pts])
            rho = M.severity_spearman(sc, lv)
            rng = np.random.default_rng(M.BOOTSTRAP_SEED)
            draws = rng.integers(0, len(pts), size=(M.BOOTSTRAP_REPLICATES, len(pts)))
            rhos = np.asarray([M.severity_spearman(sc[r], lv[r]) for r in draws])
            rhos = rhos[np.isfinite(rhos)]
            sev[cohort] = {
                "rho": None if not np.isfinite(rho) else round(float(rho), 4),
                "lcb": None if rhos.size == 0 else round(float(np.quantile(rhos, 0.025)), 4),
                "ucb": None if rhos.size == 0 else round(float(np.quantile(rhos, 0.975)), 4),
                "n": len(pts),
            }

        # Stability background support (exact): verified-healthy file rows
        # (file_label normal, non-quarantined), all member views, no
        # episode grouping or reset-split applied here. The protocol
        # per-robot-day FAR below uses the frozen grouping/reset-split path.
        bg_ids = [row["file_id"] for row in rows
                  if row["file_label"] == "normal" and not row["is_quarantined"]]
        bg_scores = np.array([file_scores[i] for i in bg_ids])
        stab = M.stability_ci(bg_scores, np.asarray(fit_scores))
        # Rates use the recomputed `threshold`, asserted identical to the
        # accepted full-precision frozen value (`138.03` is display rounding).
        per_file_exceedance = M.exceedance_fraction(bg_scores, threshold)
        flagged_by_robot: dict = {}
        for row in rows:
            if not E.eligible_operational_row(row, wins):
                continue
            if file_scores[row["file_id"]] >= threshold:
                flagged_by_robot.setdefault(row["robot_id"], []).append(
                    row["end_time"])
        eval_days = {(r["robot_id"], int(r["end_time"] // 86400.0))
                     for r in rows if E.eligible_operational_row(r, wins)}
        false_episodes, far = E.false_alert_episodes(
            flagged_by_robot, ledger, float(len(eval_days)), wins)
        per_history.append({
            "role": role, "seed": seed,
            "n_files": len(rows), "n_background": len(bg_ids),
            "n_controls": len(neg),
            "categories": cats,
            "severity": sev,
            "stability": {k: round(v, 4) for k, v in stab.items()},
            "background_per_file_exceedance_at_frozen_threshold": round(
                float(per_file_exceedance), 4),
            "background_far_per_robot_day": round(float(far), 4),
            "background_false_episodes": int(false_episodes),
            "background_robot_days": float(len(eval_days)),
        })

    record = {
        "protocol": "experiments/sprint16-attribution-protocol-v3.md",
        "probe": P.PROBE_ID,
        "frozen_threshold": FROZEN_THRESHOLD,
        "threshold_recomputed": float(threshold),
        "standardization_provenance_token": token,
        "n_fit_healthy_files": n_fit,
        "n_cal_healthy_rows": n_cal,
        "frozen_bytes_checked": FROZEN_BYTES,
        "g1_rule": {"point_floor": M.ORACLE_FLOOR,
                    "lcb_floor": M.ORACLE_LCB_FLOOR},
        "schema_note": ("v2 record: background_fpr_at_frozen_threshold "
                        "(per-file exceedance, run hash 7636ccf6…) renamed to "
                        "background_per_file_exceedance_at_frozen_threshold; "
                        "background_far_per_robot_day (+ episodes/robot_days) "
                        "added via frozen E.false_alert_episodes."),
        "histories": per_history,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True))
    summary = {
        role: {c: (v.get("point"), v.get("lcb"), v.get("measurable"))
               for c, v in h["categories"].items()}
        for role, h in [(h["role"], h) for h in per_history]
    }
    print(json.dumps({
        "threshold": float(threshold),
        "provenance_token": token[:12],
        "g1": summary,
        "severity": {h["role"]: h["severity"] for h in per_history},
        "background_per_file_exceedance": {
            h["role"]: h["background_per_file_exceedance_at_frozen_threshold"]
            for h in per_history},
        "background_far_per_robot_day": {
            h["role"]: (h["background_far_per_robot_day"],
                        h["background_false_episodes"],
                        h["background_robot_days"]) for h in per_history},
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
