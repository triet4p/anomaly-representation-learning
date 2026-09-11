"""Sprint 16 Task 6 (M2): patchification retention without encoder weights.

Compares pre-patchify observables (file-waveform probe, R_file) against
post-patchify diagnostic readouts on IDENTICAL file support with the Task 3
contract reader: mean-of-valid-patch scores (R_patchmean — joint C2 + mean
reduction, non-causal) and max-of-valid-patch scores (R_patchmax — joint C2
+ within-file max aggregation (C6/C9-adjacent), non-causal; NOT a frozen
per-event-max analog). Production patchification only (Patchifier values
asserted from PatchConfig, fail-closed). No checkpoint/weights file is read,
no nn.Module is instantiated, no encoder forward executes (asserted by
driver self-check; transitive torch/model imports via the representation
package init are disclosed, not denied). Task 14 P-a/b/c variants are
untouched future work.

Reports per history/category event AUROC (+ paired-bootstrap CI) for all
three readers, dilution gaps (R_file − R_patchmean), max headroom
(R_patchmax − R_file), exact waveform coverage, duration/boundary slices,
padded-file involvement, control-window behavior, and excluded support. No
causal verdict (Task 16 owns verdicts).

Writes ``artifacts/sprint-16/task6-patchification.json`` and prints a
summary (recorded in ``artifacts/sprint-16/task-6.md``). Deterministic
single run. Sealed roots are never touched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from representation import attribution_metrics as M

CONF = (("H-CONF-34", 1608), ("H-CONF-35", 1609), ("H-CONF-36", 1610),
        ("H-CONF-37", 1611))
FIT = (("H-FIT-28", 1604), ("H-FIT-29", 1605), ("H-FIT-30", 1606))
BASE = Path("data/generated/sprint15-v7")
OUT = Path("artifacts/sprint-16/task6-patchification.json")
CATEGORIES = ["P1", "P2", "W1", "W2", "A1", "A2", "P", "W", "A"]
FROZEN_BYTES = {
    "src/synth/probe15.py": "a08b3d5f",
    "src/synth/events.py": "85100f5e",
    "src/synth/chronicle.py": "b5992d6a",
    "src/synth/config.py": "55fed2ef",
    "src/synth/balanced.py": "d62de342",
}


def check_production_patchify() -> dict[str, object]:
    """Fail closed on non-production patchification or driver weight use.

    Asserts PatchConfig VALUES are exactly (32, 16, True, 0.0) and emits
    them (never literals); scans the actual src/synth/patchify.py for
    filtering/normalization operators (slicing/padding only); asserts the
    DRIVER source contains no checkpoint/weights-loading or encoder-use
    calls; frozen digests match Appendix A. Transitive imports (torch and
    representation.model via the representation package init) are allowed
    and DISCLOSED as measured booleans — the proven invariant is narrower:
    no checkpoint/weights file read, no nn.Module instantiated, no encoder
    forward executed by this driver.
    """
    import sys

    from synth.config import PatchConfig

    cfg = PatchConfig()
    if not (cfg.patch_size, cfg.stride, cfg.pad_end, cfg.pad_value) == (32, 16, True, 0.0):
        raise ValueError(f"non-production patch config: {cfg!r}")
    import synth.patchify as mod

    lowered = Path(mod.__file__).read_text(encoding="utf-8").lower()
    for token in ("norm", "filter", "standard", "zscore", "whiten",
                  "fft", "smooth", "detrend"):
        if token in lowered:
            raise ValueError(f"patchify.py contains {token!r}")
    own = Path(__file__).read_text(encoding="utf-8")
    forbidden_tokens = ("torch.load", "load_checkpoint", "v2_checkpoint", "LocalPatchEncoder(", "SequenceContextEncoder(", "V1RepresentationModel(", ".forward(", "nn.Module")  # no-weights self-check
    for forbidden in forbidden_tokens:
        hits = [ln for ln, line in enumerate(own.splitlines(), 1)
                if forbidden in line and "forbidden_tokens" not in line
                and "no-weights" not in line and "instantiated" not in line]
        if hits:
            raise ValueError(f"weights/encoder use in driver {forbidden}: {hits}")
    for rel, want in FROZEN_BYTES.items():
        got = hashlib.sha256(Path(rel).read_bytes()).hexdigest()[:8]
        if got != want:
            raise ValueError(f"frozen byte mismatch: {rel} {got} != {want}")
    return {"patch_size": cfg.patch_size, "stride": cfg.stride,
            "pad_end": cfg.pad_end, "pad_value": cfg.pad_value,
            "driver_loads_no_weights": True,
            "torch_imported_transitively": bool("torch" in sys.modules),
            "model_imported_transitively": bool(
                "representation.model" in sys.modules),
            "frozen_bytes_ok": True}


def load_root(role: str, group: str):
    """Load samples plus manifest for one materialized root (fail closed)."""
    from synth.chronicle import load_chronological

    root = BASE / group / role
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"missing root (refusing to proceed): {root}")
    samples, _ = load_chronological(root)
    manifest = json.loads((root / "manifest.json").read_text())
    return samples, manifest


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
    from synth import probe15 as P
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    patch_proof = check_production_patchify()
    patchifier = Patchifier(PatchConfig())

    def score_vec(feats: np.ndarray) -> np.ndarray:
        return P.score_files(P.apply_standardization(feats, stats), centroid)
    fit_feats = []
    for role, seed in FIT:
        samples, manifest = load_root(role, "FIT")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, f"tag:{role}"
        by_id = {s.file_id: s for s in samples}
        for row in manifest["files"]:
            if row["file_label"] != "normal" or row["is_quarantined"]:
                continue
            if (row["program_id"] == B.PROGRAM_RESERVE
                    or row["robot_id"] == B.ROBOT_RESERVE):
                continue
            sample = by_id[row["file_id"]]
            fit_feats.append(P.extract_features(
                np.asarray(sample.x, dtype=np.float64),
                row["end_time"] - row["start_time"],
                row["end_time"] - row["last_reset_time"]))
    fit_matrix = np.vstack(fit_feats)
    stats = P.standardize_fit(fit_matrix)
    centroid = P.fit_centroid(P.apply_standardization(fit_matrix, stats))
    provenance_token = M.FrozenStandardizer.fit(
        fit_matrix, source="H-FIT-28..30").token()


    def file_probe(x: np.ndarray, row: dict) -> np.ndarray:
        return P.extract_features(
            x, row["end_time"] - row["start_time"],
            row["end_time"] - row["last_reset_time"])

    per_history = []
    for role, seed in CONF:
        samples, manifest = load_root(role, "CONFIRMATION")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, f"tag:{role}"
        by_id = {s.file_id: s for s in samples}
        rows = manifest["files"]
        wins = manifest["maintenance_windows"]
        ledger = E.failure_ledger(manifest)
        r_file, r_mean, r_max = {}, {}, {}
        coverage, padded, n_patches = [], 0, 0
        for row in rows:
            x = np.asarray(by_id[row["file_id"]].x, dtype=np.float64)
            C, T = x.shape
            dt = (row["end_time"] - row["start_time"]) / T
            since_reset = row["end_time"] - row["last_reset_time"]
            r_file[row["file_id"]] = float(score_vec(
                file_probe(x, row)[None, :])[0])
            batch = patchifier.patchify(SimpleNamespace(x=x.astype(np.float32)))
            patches = np.asarray(batch.patches, dtype=np.float64)
            valid_len = np.asarray(batch.valid_len, dtype=int)
            feats = []
            for p in range(patches.shape[0]):
                real = int(valid_len[p])
                if real <= 0:
                    continue
                feats.append(P.extract_features(
                    patches[p][:, :real], real * dt, since_reset))
            if not feats:
                raise ValueError(f"zero valid patches: {row['file_id']}")
            ps = score_vec(np.array(feats))
            n_patches += ps.size
            r_mean[row["file_id"]] = float(ps.mean())
            r_max[row["file_id"]] = float(ps.max())
            starts = np.asarray(batch.starts, dtype=int)
            covered = np.zeros(T, dtype=bool)
            for s, ln in zip(starts.tolist(), valid_len.tolist()):
                covered[s:s + ln] = True
            coverage.append(float(covered.mean()))
            if bool((valid_len < 32).any()):
                padded += 1
        coverage = np.asarray(coverage)
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)

        def window_reader(score_map: dict[str, float]):
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

        readers = {"file": r_file, "patchmean": r_mean, "patchmax": r_max}
        neg_map, ev_map = {}, {}
        for name, sm in readers.items():
            neg_map[name], ev_map[name] = window_reader(sm)

        def events_in(cohort: str | None, subtype: str | None = None,
                      extra=None) -> list[dict]:
            out = []
            for f in ledger:
                if cohort is not None and f["cohort"] != cohort:
                    continue
                if subtype is not None and f["subtype"] != subtype:
                    continue
                if extra is not None and not extra(f):
                    continue
                out.append(f)
            return out

        padded_id_set: set[str] = set()
        for row in rows:
            x = np.asarray(by_id[row["file_id"]].x, dtype=np.float64)
            batch = patchifier.patchify(SimpleNamespace(x=x.astype(np.float32)))
            if bool((np.asarray(batch.valid_len) < 32).any()):
                padded_id_set.add(row["file_id"])

        def padded_involved(failure: dict) -> bool:
            if E.positive_window_intersects_reset(failure, wins):
                return False
            cands = E.pos_files(rows, failure, wins)
            if not cands:
                return False
            return any(c["file_id"] in padded_id_set for c in cands)

        cats: dict[str, dict] = {}
        for cat in CATEGORIES:
            cohort = cat[0] if len(cat) == 2 else cat
            subtype = cat if len(cat) == 2 else None
            cell: dict[str, object] = {}
            ok = True
            for name in readers:
                pos = [s for f in events_in(cohort, subtype)
                       if (s := ev_map[name](f)) is not None]
                if not pos or not neg_map[name]:
                    cell[name] = {"reason": "no support",
                                  "n_pos": len(pos),
                                  "n_neg": len(neg_map[name])}
                    ok = False
                else:
                    ci = auc_pair_ci(pos, neg_map[name])
                    cell[name] = {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in ci.items()}
            if ok:
                cell["dilution_gap"] = round(
                    cell["file"]["point"] - cell["patchmean"]["point"], 4)
                cell["max_headroom"] = round(
                    cell["patchmax"]["point"] - cell["file"]["point"], 4)
            cats[cat] = cell

        slices: dict[str, dict] = {}
        slice_defs = [
            ("short_PW", lambda f: f["cohort"] in "PW" and f["duration_d"] < 1.0),
            ("long_PW", lambda f: f["cohort"] in "PW" and f["duration_d"] >= 1.0),
            ("padded_involved", lambda f: padded_involved(f)),
            ("interior_only", lambda f: not padded_involved(f)
             and E.positive_window_intersects_reset(f, wins) is False
             and bool(E.pos_files(rows, f, wins))),
        ]
        for label, pred in slice_defs:
            cell = {}
            ok = True
            for name in ("file", "patchmean"):
                pos = [s for f in events_in(None)
                       if pred(f) and (s := ev_map[name](f)) is not None]
                if len(pos) < 3 or not neg_map[name]:
                    cell[name] = {"reason": "no support", "n_pos": len(pos)}
                    ok = False
                else:
                    ci = auc_pair_ci(pos, neg_map[name])
                    cell[name] = {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in ci.items()}
            if ok:
                cell["dilution_gap"] = round(
                    cell["file"]["point"] - cell["patchmean"]["point"], 4)
            slices[label] = cell

        fid_order = list(r_file.keys())
        label_of = {r["file_id"]: (0.0 if r["file_label"] == "normal" else 1.0)
                    for r in rows}
        score_maps = {"file": r_file, "patchmean": r_mean, "patchmax": r_max}
        stage_sets = [
            M.DiagnosticSet(
                history_id=role, ids=np.asarray(fid_order),
                families={name: np.asarray([sm[f] for f in fid_order])},
                labels=np.asarray([label_of[f] for f in fid_order]),
                provenance=provenance_token)
            for name, sm in score_maps.items()
        ]
        checked_token = M.assert_same_provenance(*stage_sets)
        per_history.append({
            "role": role, "seed": seed,
            "n_files": len(rows),
            "n_patches_total": int(n_patches),
            "n_padded_files": int(padded),
            "coverage_min": round(float(coverage.min()), 4),
            "coverage_mean": round(float(coverage.mean()), 4),
            "standardization_provenance_token": checked_token,
            "categories": cats,
            "slices": slices,
            "control_medians": {
                name: round(float(np.median(neg_map[name])), 4)
                for name in readers},
        })

    record = {
        "protocol": "experiments/sprint16-attribution-protocol-v3.md",
        "measurement": "M2",
        "patchification": patch_proof,
        "histories": per_history,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True))
    print(json.dumps({
        "coverage": [(h["role"], h["coverage_min"], h["coverage_mean"])
                     for h in per_history],
        "padded": [(h["role"], h["n_padded_files"]) for h in per_history],
        "dilution": {h["role"]: {c: v.get("dilution_gap")
                                 for c, v in h["categories"].items()}
                     for h in per_history},
        "headroom": {h["role"]: {c: v.get("max_headroom")
                                 for c, v in h["categories"].items()}
                     for h in per_history},
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
