"""Sprint 17 Task 8 — audit B0 reproducibility/support from existing evidence.

Local mode (default): mechanical audit of Task 7 evidence against protocol v4
— config identity, three-seed completeness, finite outputs, support identity
(recomputed from local manifests), branch independence, params/FLOPs,
oracle ceiling (Task 4 probe record), provenance/checksum integrity, role
isolation (recounted from local manifests), gates/negatives. No retraining,
no GPU, no Confirmation/Sealed access. Writes
``experiments/sprint17-task8-audit.json``.

Checkpoint mode (``--checkpoint-check``): focused reload fidelity on the
execution host — restore one checkpoint via the public loader, assert step,
config, and bank identity, rescore a few cache files on CPU, and compare
against the stored cache. Strictly bounded; the only remote execution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

EVIDENCE = REPO_ROOT / "artifacts" / "sprint-17" / "task7-evidence"
AUDIT_OUT = REPO_ROOT / "experiments" / "sprint17-task8-audit.json"


def fail(checks: list, name: str, ok: bool, detail: object = None) -> None:
    checks.append({"check": name, "pass": bool(ok), "detail": detail})
    if not ok:
        raise ValueError(f"TASK8 AUDIT FAIL: {name}: {detail}")


def local_audit() -> dict:
    import numpy as np

    import sprint17_task7_b0 as D
    from synth import balanced as B
    from synth import events as E

    checks: list[dict] = []
    doc = json.loads((EVIDENCE / "b0_development_v4.json").read_text())
    spop = json.loads((EVIDENCE / "b0_spop_metrics.json").read_text())
    perseed = json.loads((EVIDENCE / "b0_per_seed_metrics.json").read_text())
    manifest = json.loads((EVIDENCE / "b0_run_manifest.json").read_text())
    runlog = json.loads((EVIDENCE / "run.log.json").read_text())
    probe = json.loads(
        (REPO_ROOT / "artifacts" / "sprint-17" / "task4-probe.json").read_text())

    # 1. Deterministic config identity.
    cfgs = [D.b0_config_dict(s) for s in D.MODEL_SEEDS]
    keys = [sorted(c.keys()) for c in cfgs]
    fail(checks, "config_keys_identical", all(k == keys[0] for k in keys))
    non_seed = [{k: v for k, v in c.items() if k != "model_seed"} for c in cfgs]
    fail(checks, "config_identical_except_seed",
         all(c == non_seed[0] for c in non_seed))
    fail(checks, "config_hash_matches_manifest",
         D.canonical_hash({str(s): perseed["seed_records"][str(s)]["cfg_dict"]
                           for s in D.MODEL_SEEDS}) == manifest["config_sha256"],
         manifest["config_sha256"][:16])
    fail(checks, "wd_zero_parity", cfgs[0]["optimizer"]["weight_decay"] == 0.0)
    fail(checks, "steps_checkpoint_300",
         all(c["optimizer_steps"] == 300 and c["checkpoint_step"] == 300
             for c in cfgs))

    # 2. Three-seed completeness.
    recs = perseed["seed_records"]
    fail(checks, "three_seeds_present",
         sorted(int(s) for s in recs) == [171701, 171702, 171703])
    fail(checks, "all_steps_300", all(r["steps"] == 300 for r in recs.values()))
    fail(checks, "distinct_checkpoints", len({r["ckpt_sha256"]
                                              for r in recs.values()}) == 3)
    fail(checks, "bank_rows_5040_each",
         all(r["bank_rows"] == 5040 for r in recs.values()))
    for s in ("171701", "171702", "171703"):
        for branch in ("S_pred", "S_pop"):
            src = (perseed["S_pred_per_seed"] if branch == "S_pred"
                   else spop["per_seed"])[s]
            fail(checks, f"metrics_present_{s}_{branch}",
                 src["macro_pw"] is not None and len(src["per_history"]) == 4)

    # 3. Finite outputs (numbers are finite; nulls only where schema allows).
    def scan(obj, path=""):
        bad = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                bad += scan(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                bad += scan(v, f"{path}[{i}]")
        elif isinstance(obj, float) and not np.isfinite(obj):
            bad.append(path)
        return bad
    for name, payload in (("doc", doc), ("spop", spop), ("perseed", {
            k: v for k, v in perseed.items() if k != "seed_records"})):
        fail(checks, f"finite_{name}", scan(payload) == [], scan(payload)[:3])

    # 4. Support identity, recomputed from local manifests (no samples).
    data_root = REPO_ROOT / "data" / "generated" / "sprint17-ablation-v3"
    support_recomputed = {}
    excluded = 0
    for role, seed in D.DEV_ROLES:
        man = json.loads((data_root / "DEVELOPMENT" / role
                          / "manifest.json").read_text())
        assert man["seeds"]["health"] == seed and man["role"] == role
        ledger = E.failure_ledger(man)
        sup = D.evaluable_support(man["files"], ledger,
                                  man["maintenance_windows"])
        support_recomputed[role] = sup
        excluded += sup["skipped"]
    for entry in doc["support"]["per_history"]:
        sup = support_recomputed[entry["history_id"]]
        fail(checks, f"support_counts_{entry['history_id']}",
             entry["p"] == len(sup["pos"]["P"])
             and entry["w"] == len(sup["pos"]["W"])
             and entry["a"] == len(sup["pos"]["A"])
             and entry["controls"] == len(sup["controls"]),
             entry)
    fail(checks, "excluded_total_131", doc["support"]["excluded_rows"] == 131
         and excluded == 131, excluded)
    fail(checks, "common_support_hash_reproduced",
         D.canonical_hash({
             role: {"events": sorted(tuple(e["key"])
                                     for e in support_recomputed[role]["pos"]["P"]
                                     + support_recomputed[role]["pos"]["W"]
                                     + support_recomputed[role]["pos"]["A"]),
                    "controls": sorted(tuple(sorted(w)) for w in
                                       support_recomputed[role]["controls"])}
             for role in [r for r, _ in D.DEV_ROLES]})
         == doc["support"]["common_support_sha256"])
    # Same evaluable support as the Task 4 observable probe (oracle path).
    probe_dev = {r: h for r, h in zip(
        probe["development"]["roots"],
        probe["development"]["metrics"]["per_history"])}
    for entry in doc["support"]["per_history"]:
        ph = probe_dev[entry["history_id"]]
        fail(checks, f"probe_support_parity_{entry['history_id']}",
             ph["n_p"] == entry["p"] and ph["n_w"] == entry["w"]
             and ph["robot_days"] == entry["robot_days"])

    # 5. Branch independence.
    corrs = {s: recs[s]["score_branch_corr"] for s in recs}
    fail(checks, "branches_not_aliased",
         all(abs(c) < 0.999 for c in corrs.values()), corrs)
    fail(checks, "thresholds_separated",
         all(abs(recs[s]["thr_pred"] - recs[s]["thr_pop"]) > 1.0
             for s in recs))

    # 6. Params/FLOPs.
    params = {s: recs[s]["params"] for s in recs}
    flops = {s: recs[s]["flops_reference"] for s in recs}
    fail(checks, "params_identical_1821698",
         set(params.values()) == {1821698}, params)
    fail(checks, "flops_identical_182016709",
         set(flops.values()) == {182016709}, flops)
    fail(checks, "envelope_zero_delta",
         doc["compute"]["params_delta_fraction"] == 0.0
         and doc["compute"]["flops_delta_fraction"] == 0.0
         and doc["compute"]["envelope_pass"] is True)

    # 7. Measurable oracle ceiling (Task 4 frozen probe, same histories).
    ceiling = probe["development"]["metrics"]["macro_auc_pw"]
    fail(checks, "oracle_ceiling_measurable",
         ceiling is not None and ceiling > 0.55, round(ceiling, 4))
    b0_primary = doc["metrics"]["primary_PW"]["point"]
    fail(checks, "b0_below_ceiling_gap_recorded",
         b0_primary < ceiling, {"b0": round(b0_primary, 4),
                                "ceiling": round(ceiling, 4),
                                "gap": round(ceiling - b0_primary, 4)})

    # 8. Provenance/checksum integrity.
    fail(checks, "checkpoint_aggregate_hash",
         D.canonical_hash(sorted(r["ckpt_sha256"] for r in recs.values()))
         == doc["provenance"]["checkpoint_sha256"])
    cache_all = sorted(sha for r in recs.values()
                       for sha in r["cache_sha256"].values())
    fail(checks, "twelve_caches_present", len(cache_all) == 12, len(cache_all))
    fail(checks, "cache_aggregate_hash",
         D.canonical_hash(cache_all) == doc["provenance"]["cache_sha256"])
    fail(checks, "binding_digest", runlog["binding_sha256"]
         == D.BINDING_SHA256 and doc["provenance"]["role_binding_sha256"]
         == D.BINDING_SHA256)
    fail(checks, "metric_code_hash",
         doc["provenance"]["metric_code_sha256"] == manifest["metric_code_sha256"])
    for role, _ in D.FIT_ROLES + (D.CAL_ROLE,) + D.DEV_ROLES:
        group = ("FIT" if "FIT" in role else
                 "CALIBRATION" if "CAL" in role else "DEVELOPMENT")
        digest = D.sha256_file(data_root / group / role / "manifest.json")
        fail(checks, f"manifest_bytes_{role}",
             digest == D.EXPECTED_MANIFEST_SHA256[role])
    fail(checks, "no_confirmation_roles",
         not any("CONF" in r for r in doc["provenance"]["data_roles"])
         and doc["phase"] == "development")
    fail(checks, "execution_commit_lineage",
         manifest["git"]["commit"] == "8c15f0204a3e495569b7f143dc109943e8b808de"
         and manifest["git"]["status"] == "")

    # 9. Bank/Calibration role isolation, recounted from local manifests.
    def eligible_count(group, role):
        man = json.loads((data_root / group / role / "manifest.json").read_text())
        return sum(1 for row in man["files"] if D.eligible_healthy_row(
            row, man["maintenance_windows"], B.PROGRAM_RESERVE, B.ROBOT_RESERVE))
    fit_total = sum(eligible_count("FIT", r) for r, _ in D.FIT_ROLES)
    cal_total = eligible_count("CALIBRATION", D.CAL_ROLE[0])
    fail(checks, "fit_pool_recount_5040", fit_total == 5040, fit_total)
    fail(checks, "cal_recount_1621", cal_total == 1621, cal_total)
    fail(checks, "runlog_fit_cal_match",
         runlog["fit_rows"] == 5040 and runlog["cal_rows"] == 1621)

    # 10. Gates and negative results, enumerated.
    gates = doc["gates"]
    fail(checks, "gates_true_set",
         all(gates[k] is True for k in ("structural", "observable",
                                        "finite_and_support",
                                        "score_separation",
                                        "compute_envelope", "P_noninferiority",
                                        "W_noninferiority", "nuisance",
                                        "background_stability", "eligible")),
         gates)
    fail(checks, "baseline_negative_by_construction",
         gates["minimum_effect"] is False and gates["recovery"] is False
         and doc["status"] == "VALID_NEGATIVE")
    recalls_p = doc["metrics"]["P"]["recall_per_history"]
    fail(checks, "dev04_p_recall_negative_recorded",
         recalls_p[3] is not None and recalls_p[3] < 0.50, recalls_p)

    # Doc/sidecar aggregation consistency.
    for h in range(4):
        vals = [perseed["S_pred_per_seed"][s]["per_history"][h]["auc_pw"]
                for s in ("171701", "171702", "171703")]
        fail(checks, f"headline_mean_seed_consistent_h{h}",
             abs(sum(vals) / 3 - doc["metrics"]["primary_PW"]
                 ["per_history"][h]) < 1e-12)

    result = {"protocol": D.PROTOCOL_ID, "arm": "B0", "phase": "development",
              "n_checks": len(checks), "n_pass": sum(1 for c in checks
                                                    if c["pass"]),
              "checks": checks,
              "oracle_ceiling_PW": ceiling,
              "b0_primary_PW": b0_primary}
    AUDIT_OUT.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"TASK8_AUDIT_LOCAL: {result['n_pass']}/{result['n_checks']} PASS")
    print(f"oracle_ceiling_PW={ceiling:.4f} b0_primary_PW={b0_primary:.4f}")
    return result


def checkpoint_check(ckpt: str, cache: str, data_root: str, files: str) -> int:
    """Reload one checkpoint; assert identity; rescore listed files on CPU."""
    import numpy as np
    import torch

    from representation.checkpoint import load_checkpoint
    from representation.data import collate_variable_files
    from representation.inference import NormalReferenceBank, RepresentationInference
    from synth.chronicle import load_chronological
    import sprint17_task7_b0 as D

    device = torch.device("cpu")
    # Rebuild the exact B0 architecture from the frozen seed config, then
    # restore weights/step/bank through the public loader (strict).
    model, _, patchifier = D.build_model(D.b0_config_dict(171701), device)
    bank = NormalReferenceBank(k=5)
    meta = load_checkpoint(ckpt, model, reference_bank=bank)
    step = int(meta["step"])
    assert step == 300, step
    assert meta["has_reference_bank"] is True
    assert model.config.n_robots == 9 and model.config.n_programs == 8
    assert bank.embeddings.shape[0] == 5040, bank.embeddings.shape
    assert bank.embeddings.shape[1] == 128
    model.eval()
    inference = RepresentationInference(model, bank, patchifier,
                                        masking_config=model.config)
    samples, _ = load_chronological(Path(data_root))
    by_id = {s.file_id: s for s in samples}
    stored = np.load(cache)
    ids = [str(v) for v in stored["file_ids"]]
    max_diff_pred = 0.0
    max_diff_pop = 0.0
    max_rel_pred = 0.0
    max_rel_pop = 0.0
    for fid in files.split(","):
        fid = fid.strip()
        assert fid in ids, f"file not in cache: {fid}"
        batch = collate_variable_files(
            [by_id[fid]], patchifier, masking_config=model.config,
            masking_seed=D.file_seed(fid))
        res = inference.score_batch(batch)
        got_pred = float(torch.as_tensor(res["S_pred"]).reshape(-1)[0])
        got_pop = float(torch.as_tensor(res["S_pop"]).reshape(-1)[0])
        idx = ids.index(fid)
        ref_pred = float(stored["S_pred"][idx])
        ref_pop = float(stored["S_pop"][idx])
        max_diff_pred = max(max_diff_pred, abs(got_pred - ref_pred))
        max_diff_pop = max(max_diff_pop, abs(got_pop - ref_pop))
        max_rel_pred = max(max_rel_pred,
                           abs(got_pred - ref_pred) / max(1e-12, abs(ref_pred)))
        max_rel_pop = max(max_rel_pop,
                          abs(got_pop - ref_pop) / max(1e-12, abs(ref_pop)))
    print(json.dumps({"checkpoint": ckpt, "step": step,
                      "bank_rows": bank.embeddings.shape[0],
                      "n_files": len(files.split(",")),
                      "max_abs_diff_pred": max_diff_pred,
                      "max_abs_diff_pop": max_diff_pop,
                      "max_rel_diff_pred": max_rel_pred,
                      "max_rel_diff_pop": max_rel_pop}))
    # Cross-device (GPU-train → CPU-rescore) float precision, not bitwise:
    # 1e-3 relative bounds device noise; missing state (norm/EMA/bank)
    # would move scores by orders of magnitude, not 1e-4.
    assert max_rel_pred < 1e-3 and max_rel_pop < 1e-3, (
        max_rel_pred, max_rel_pop)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-check", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--files", default="")
    args = ap.parse_args()
    if args.checkpoint_check:
        if not (args.ckpt and args.cache and args.data_root and args.files):
            raise ValueError("--ckpt/--cache/--data-root/--files required")
        return checkpoint_check(args.ckpt, args.cache, args.data_root,
                                args.files)
    local_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
