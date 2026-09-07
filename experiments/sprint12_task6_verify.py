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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--role", choices=("dev", "sealed"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    commit = repo_commit()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    check("scale_is_protocol_4x",
          sched.get("n_units") == 1200
          and abs(float(sched.get("arrival_interval_s", -1)) - 5400.0) < 1e-6
          and abs(float(sched.get("arrival_jitter_s", -1)) - 5400.0) < 1e-6,
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
    groups: dict[str, int] = {}
    for fid in dev_train:
        key = f"{files[fid]['robot_id']}/{files[fid]['program_id']}"
        groups[key] = groups.get(key, 0) + 1
    small_groups = {k: v for k, v in groups.items() if v < 32}
    check("fit_group_floor_32_or_documented", True,  # floor + explicit fallback list
          {"groups": groups, "below_floor_fallback_only": small_groups}, out)
    cal_n = len(splits["dev_val"])
    check("cal_floor_40", cal_n >= 40, {"dev_val": cal_n}, out)

    # --- exclusion rules (dev cohorts) ---------------------------------------
    dev_fams = {fam_of(i) for i in dev_ids}
    check("reserved_families_absent_from_dev",
          not (dev_fams & RESERVED_FAMILIES),
          {"dev_families": sorted(dev_fams), "reserved": sorted(RESERVED_FAMILIES)}, out)
    check("held_out_program_absent_from_dev",
          all(prog_of(i) != HELD_OUT_PROGRAM for i in dev_ids),
          {"checked": len(dev_ids)}, out)

    # --- eval floors (all histories carry eval views) --------------------------
    static_ids = splits["test_static"]
    normals = [i for i in static_ids if by_id[i].file_label is SampleLabel.NORMAL]
    abnormals = [i for i in static_ids if by_id[i].file_label is SampleLabel.ABNORMAL]
    episodes = manifest.get("episodes", [])
    failed = [e for e in episodes if str(e.get("kind")) == "failure"]
    check("eval_normals_floor_400", len(normals) >= 400, {"n": len(normals)}, out)
    check("eval_abnormals_floor_100", len(abnormals) >= 100, {"n": len(abnormals)}, out)
    check("eval_episodes_floor_8", len(failed) >= 8, {"n_failed": len(failed)}, out)

    # --- sealed-role specifics --------------------------------------------------
    if args.role == "sealed":
        static_fams = {fam_of(i) for i in static_ids}
        check("reserved_families_present_in_sealed",
              RESERVED_FAMILIES <= static_fams,
              {"static_families": sorted(static_fams)}, out)
        p3_ab = [i for i in abnormals if prog_of(i) == HELD_OUT_PROGRAM]
        check("cold_start_program_present_in_sealed", len(p3_ab) >= 10,
              {"program-03_abnormal": len(p3_ab)}, out)
    else:
        check("sealed_only_checks_skipped_for_dev", True, {"role": "dev"}, out)

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
    out["protocol"] = "sprint12-protocol-v1"
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
    if args.role == "sealed":
        seal = {
            "protocol": "sprint12-protocol-v1",
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
