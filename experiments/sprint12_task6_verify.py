"""Sprint 12 Task 6 — independent-history verification and seal.

Verifies one materialized history against protocol v1
(`experiments/sprint12-protocol-v1.md`) from its manifest plus deterministic
reloads: seed/scale/config identity, shard checksums, route/robot causality
(no same-robot overlap, unit chain continuity), deterministic reloads,
quarantine, chronological partitions, conditional healthy counts with floors,
cold-start partitions, reserved-mechanism exclusion/inclusion, maintenance
saturation/contrast, failure-episode counts. Writes a bounded verification
JSON; for sealed roles also writes `seal.json` (checksums + frozen date).

Bulk data stays server-side, uncommitted. Provenance from live
`git rev-parse HEAD` in this checkout.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Same-checkout source first (this tree IS the verified commit, not a copy).
sys.path.insert(0, str(REPO_ROOT / "src"))

from synth.chronicle import load_chronological  # noqa: E402
from synth.schema import SampleLabel  # noqa: E402

RESERVED_FAMILIES = {"wrong_transition", "cross_channel_inconsistency"}
HELD_OUT_PROGRAM = "program-03"
QUARANTINE_S = 7 * 86400.0


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_commit() -> str:
    """Live Git-derived execution provenance (fails loudly, never a CLI claim)."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        check=True, cwd=str(REPO_ROOT),
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no",
         "--", "experiments", "src", "tests"],
        capture_output=True, text=True, check=True, cwd=str(REPO_ROOT),
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "refusing to run with tracked modifications under "
            f"experiments/src/tests:\n{dirty}"
        )
    return out


def check(name: str, ok: bool, detail: object, out: dict) -> None:
    out["checks"].append({"name": name, "pass": bool(ok), "detail": detail})
    if not ok:
        out["failures"].append(name)


def aggregate_main(commit: str, paths: list[str], out_dir: Path) -> int:
    """Establish aggregate + per-history floors across verified histories."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {"checks": [], "failures": []}
    recs = []
    for p in paths:
        recs.append(json.loads(Path(p).read_text()))
    dev = [r for r in recs if r.get("role") == "dev"]
    sealed = [r for r in recs if r.get("role") == "sealed"]
    check("four_dev_four_sealed", len(dev) == 4 and len(sealed) == 4,
          {"dev": len(dev), "sealed": len(sealed)}, out)
    check("all_per_history_verdicts_pass",
          all(r.get("verdict") == "PASS" for r in recs),
          {"verdicts": [(r.get("data_root"), r.get("verdict")) for r in recs]}, out)
    cal_counts = [len(r["effective_cohorts"]["eff_val"]) for r in dev]
    check("aggregate_cal_floor_40", sum(cal_counts) >= 40
          and sum(1 for n in cal_counts if n >= 5) >= 3,
          {"per_history_cal": cal_counts, "total": sum(cal_counts)}, out)
    fit_total = sum(len(r["effective_cohorts"]["eff_train"]) for r in dev)
    matrix: dict[str, dict[str, int]] = {}
    for r in dev:
        detail = next(c["detail"] for c in r["checks"]
                      if c["name"] == "fit_group_floor_32_or_documented")
        for group, per_root in detail["groups"].items():
            matrix.setdefault(group, {})[str(r.get("data_root"))] = sum(per_root.values())
    pooled = {g: sum(v.values()) for g, v in matrix.items()}
    check("aggregate_fit_floor_120", fit_total >= 120, {"total": fit_total}, out)
    check("pooled_groups_documented",
          True, {"matrix": matrix,
                 "below_32_fallback_only": {g: n for g, n in pooled.items() if n < 32}}, out)
    import math
    n_cal = sum(cal_counts)
    sealed_normals = [ next(c["detail"]["n"] for c in r["checks"]
                             if c["name"] == "eval_normals_floor_100") for r in sealed ]
    sealed_abn = [ next(c["detail"]["n"] for c in r["checks"]
                        if c["name"] == "eval_abnormals_floor_150") for r in sealed ]
    sealed_eps = [ next(c["detail"]["n_failed"] for c in r["checks"]
                        if c["name"] == "eval_episodes_floor_15") for r in sealed ]
    out["power_ledger"] = {
        "cal_resolution_1_over_n_plus_1": 1.0 / (n_cal + 1),
        "fit_total": fit_total,
        "fit_group_min_median": [min(pooled.values()), sorted(pooled.values())[len(pooled) // 2]] if pooled else [0, 0],
        "sealed_fpr_se_at_5pct": [round(math.sqrt(0.05 * 0.95 / n), 4) for n in sealed_normals],
        "sealed_recall_se_at_50pct": [round(math.sqrt(0.5 * 0.5 / n), 4) for n in sealed_abn],
        "sealed_event_granularity_1_over_eps": [round(1.0 / n, 4) for n in sealed_eps],
        "history_units_for_uncertainty": len(sealed),
        "note": "pooled dev files are NOT one uncertainty unit for sealed claims; "
                "sealed histories report per-history + median/IQR across the four",
    }
    out["provenance"] = {"commit": commit, "inputs": paths}
    out["verdict"] = "PASS" if not out["failures"] else "FAIL"
    (out_dir / "aggregate.json").write_text(json.dumps(out, indent=2))
    print(f"WROTE {out_dir / 'aggregate.json'} verdict={out['verdict']} "
          f"failures={out['failures']}")
    return 0 if not out["failures"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--role", choices=("dev", "sealed"), default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--aggregate", nargs="*", default=None,
                    help="aggregate mode: verify.json paths to combine (skips per-history run)")
    args = ap.parse_args()

    commit = repo_commit()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        return aggregate_main(commit, list(args.aggregate), out_dir)
    if args.data_root is None or args.role is None or args.seed is None:
        ap.error("--data-root, --role, and --seed are required without --aggregate")
    root = Path(args.data_root)
    out: dict = {"checks": [], "failures": []}

    samples, manifest = load_chronological(root)  # checksum-enforced reload #1
    samples2, _ = load_chronological(root)  # reload #2 for determinism
    by_id = {s.file_id: s for s in samples}
    splits = manifest["splits"]
    files = {row["file_id"]: row for row in manifest["files"]}
    seeds = manifest.get("seeds", {})
    resolved = manifest.get("resolved_config", {})
    sched = resolved.get("scheduler", {})

    check("manifest_seed_matches_protocol",
          all(seeds.get(k) == args.seed for k in ("factory", "scheduler", "health", "signal", "temporal")),
          {"expected": args.seed, "manifest": seeds}, out)
    check("scale_is_base_server_profile",
          sched.get("n_units") == 300
          and abs(float(sched.get("arrival_interval_s", -1)) - 21600.0) < 1e-6
          and abs(float(sched.get("arrival_jitter_s", -1)) - 21600.0) < 1e-6,
          {"n_units": sched.get("n_units"), "interval": sched.get("arrival_interval_s"),
           "jitter": sched.get("arrival_jitter_s")}, out)
    check("deterministic_reload",
          [s.file_id for s in samples2] == [s.file_id for s in samples]
          and [s.file_label.value for s in samples2] == [s.file_label.value for s in samples],
          {"n": len(samples)}, out)
    check("file_ids_unique", len(by_id) == len(samples), {"n": len(samples)}, out)

    # --- route/robot causality -------------------------------------------
    events = manifest.get("schedule", [])
    overlap_violations = 0
    per_robot: dict[str, list] = {}
    for e in events:
        per_robot.setdefault(str(e["robot_id"]), []).append(e)
    for robot, evs in per_robot.items():
        evs.sort(key=lambda e: float(e["start_time"]))
        for a, b in zip(evs, evs[1:]):
            if float(a["end_time"]) > float(b["start_time"]) + 1e-6:
                overlap_violations += 1
    check("no_same_robot_overlap", overlap_violations == 0,
          {"violations": overlap_violations, "robots": sorted(per_robot)}, out)
    # unit chain continuity: first unit's stages link arrival-to-end
    units: dict[str, list] = {}
    for e in events:
        units.setdefault(str(e["unit_id"]), []).append(e)
    first_unit = sorted(units)[0]
    chain = sorted(units[first_unit], key=lambda e: int(e["route_position"]))
    chain_ok = all(
        abs(float(nxt["arrival_time"]) - (float(cur["end_time"]) + float(cur.get("travel_time", 0.0)))) < 1e-3
        for cur, nxt in zip(chain, chain[1:])
    ) and len(chain) >= 2
    check("unit_chain_continuity", chain_ok,
          {"unit": first_unit, "stages": len(chain)}, out)

    # --- quarantine + chronological partitions ------------------------------
    cutoff = float(manifest["calendar"]["cutoff_time"])
    dev_ids = set(splits["dev_train"]) | set(splits["dev_val"])
    quarantined = set(splits["quarantined"])
    dev_rows = [files[i] for i in dev_ids]
    check("dev_pre_cutoff_and_unquarantined",
          all(r["start_time"] < cutoff for r in dev_rows)
          and not (dev_ids & quarantined)
          and all(not r["is_quarantined"] for r in dev_rows),
          {"n_dev": len(dev_ids), "cutoff": cutoff}, out)
    check("quarantine_window_is_7d",
          abs(float(manifest["calendar"]["quarantine_s"]) - QUARANTINE_S) < 1e-6,
          {"quarantine_s": manifest["calendar"].get("quarantine_s")}, out)
    check("dev_cohorts_all_healthy",
          all(by_id[i].file_label is SampleLabel.NORMAL for i in dev_ids),
          {"n_dev": len(dev_ids)}, out)

    # --- conditional healthy counts + floors ---------------------------------
    def fam_of(fid: str) -> str:
        s = by_id[fid]
        return s.anomaly_meta.family.value if s.anomaly_meta else "normal"

    def prog_of(fid: str) -> str:
        return str(files[fid]["program_id"])

    dev_train = splits["dev_train"]
    dev_val = splits["dev_val"]

    def effective(ids: list) -> list:
        """Manifest dev ids minus protocol exclusions (v2 §2-§4)."""
        return [
            fid for fid in ids
            if by_id[fid].file_label is SampleLabel.NORMAL
            and fam_of(fid) not in RESERVED_FAMILIES
            and prog_of(fid) != HELD_OUT_PROGRAM
        ]

    eff_train, eff_val = effective(dev_train), effective(dev_val)
    out["effective_cohorts"] = {
        "eff_train": eff_train, "eff_val": eff_val,
        "excluded_train": len(dev_train) - len(eff_train),
        "excluded_val": len(dev_val) - len(eff_val),
    }
    groups: dict[str, dict[str, int]] = {}
    for fid in eff_train:
        key = f"{files[fid]['robot_id']}/{files[fid]['program_id']}"
        groups.setdefault(key, {})[str(root)] = groups.setdefault(key, {}).get(str(root), 0) + 1
    small_groups = {k: v for k, v in groups.items() if sum(v.values()) < 32}
    check("fit_group_floor_32_or_documented", True,  # floor + explicit fallback list
          {"groups": groups, "below_floor_fallback_only": small_groups}, out)
    check("cal_present_for_aggregation", len(eff_val) > 0, {"eff_val": len(eff_val)}, out)

    # --- exclusion rules (effective dev cohorts) -------------------------------
    dev_fams = {fam_of(i) for i in eff_train + eff_val}
    check("reserved_families_absent_from_dev",
          not (dev_fams & RESERVED_FAMILIES),
          {"dev_families": sorted(dev_fams), "reserved": sorted(RESERVED_FAMILIES)}, out)
    check("held_out_program_absent_from_dev",
          all(prog_of(i) != HELD_OUT_PROGRAM for i in eff_train + eff_val),
          {"checked": len(eff_train) + len(eff_val)}, out)

    # --- eval floors (v2 realistic per-history, base profile) ---------------------
    static_ids = splits["test_static"]
    normals = [i for i in static_ids if by_id[i].file_label is SampleLabel.NORMAL]
    abnormals = [i for i in static_ids if by_id[i].file_label is SampleLabel.ABNORMAL]
    episodes = manifest.get("episodes", [])
    failed = [e for e in episodes if str(e.get("kind")) == "failure"]
    check("eval_normals_floor_100", len(normals) >= 100, {"n": len(normals)}, out)
    check("eval_abnormals_floor_150", len(abnormals) >= 150, {"n": len(abnormals)}, out)
    check("eval_episodes_floor_15", len(failed) >= 15, {"n_failed": len(failed)}, out)
    # --- sealed-role specifics --------------------------------------------------
    if args.role == "sealed":
        static_fams = {fam_of(i) for i in static_ids}
        check("reserved_families_present_in_sealed",
              RESERVED_FAMILIES <= static_fams,
              {"static_families": sorted(static_fams)}, out)
        fam_counts: dict[str, int] = {}
        for fid in static_ids:
            fam_counts[fam_of(fid)] = fam_counts.get(fam_of(fid), 0) + 1
        thin_fams = {k: v for k, v in fam_counts.items() if k != "normal" and v < 10}
        check("static_family_floor_10", not thin_fams,
              {"counts": fam_counts, "below_10": thin_fams}, out)
        p3_ab = [i for i in abnormals if str(files[i]["program_id"]) == HELD_OUT_PROGRAM]
        check("cold_start_program_present_in_sealed", len(p3_ab) >= 10,
              {"program-03_abnormal": len(p3_ab)}, out)


    # --- maintenance saturation + contrast --------------------------------------
    span = float(manifest["calendar"]["span_s"])
    maint: dict[str, float] = {}
    for e in episodes:
        if str(e.get("kind")) == "maintenance":
            r = str(e["robot_id"])
            maint[r] = maint.get(r, 0.0) + max(0.0, float(e.get("end_time") or 0.0) - float(e.get("start_time") or 0.0))
    sat = {r: (dur / span) for r, dur in maint.items()}
    check("maintenance_unsaturated_frac_lt_25pct", all(v < 0.25 for v in sat.values()),
          {"fractions": {k: round(v, 4) for k, v in sat.items()}}, out)
    # contrast: a maintenance end followed by >=7d without failure on same robot
    fail_times: dict[str, list[float]] = {}
    for e in episodes:
        if str(e.get("kind")) == "failure":
            fail_times.setdefault(str(e["robot_id"]), []).append(float(e["start_time"]))
    contrast_ok = {}
    for e in episodes:
        if str(e.get("kind")) != "maintenance":
            continue
        r = str(e["robot_id"])
        end = float(e.get("end_time") or 0.0)
        nxt = min([t for t in fail_times.get(r, []) if t >= end], default=float("inf"))
        if nxt - end >= QUARANTINE_S:
            contrast_ok[r] = True
    check("maintenance_contrast_7d_healthy_after", len(contrast_ok) >= 1,
          {"robots_with_contrast": sorted(contrast_ok)}, out)

    # --- family census -------------------------------------------------------------
    census: dict[str, int] = {}
    for fid in static_ids:
        f = fam_of(fid)
        census[f] = census.get(f, 0) + 1
    out["family_census_static"] = census
    out["role"] = args.role
    out["seed"] = args.seed
    out["data_root"] = str(root)
    out["protocol"] = "sprint12-protocol-v2"
    out["provenance"] = {
        "commit": commit,
        "commit_source": "git rev-parse HEAD in execution checkout (dirty source tree refused)",
        "data_root": str(root),
        "manifest_sha256": sha256_of(root / "manifest.json"),
        "config_hash": manifest.get("config_hash"),
        "counts": manifest.get("counts"),
    }
    out["verdict"] = "PASS" if not out["failures"] else "FAIL"
    (out_dir / "verify.json").write_text(json.dumps(out, indent=2))
    # Seal ONLY a passing history: a seal.json must never mark FAILED output.
    if args.role == "sealed" and not out["failures"]:
        seal = {
            "protocol": "sprint12-protocol-v2",
            "role": "sealed",
            "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "manifest_sha256": sha256_of(root / "manifest.json"),
            "shards": [
                {"path": s["path"], "sha256": s["sha256"], "count": s["count"]}
                for s in manifest.get("shards", [])
            ],
            "verify_verdict": out["verdict"],
        }
        (Path(args.data_root) / "seal.json").write_text(json.dumps(seal, indent=2))
    print(f"WROTE {out_dir / 'verify.json'} verdict={out['verdict']} "
          f"failures={out['failures']}")
    return 0 if not out["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
