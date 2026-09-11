"""Sprint 15 Task 9 cycle 7: candidate-7 public-entry-point contract proof.

Runs the canonical local public entry point in fresh subprocesses for the two
disposable proof roots (seeds 1616/1617, role H-PROOF-13/14), then verifies
fresh-process determinism, chronological ordering, physical episode bounds,
anchor eligibility, exact quotas, bounded nuisances, the analytic observable
separation margin, subtype balance, no-overlap, no leakage, audit
computability, the §3b exact-CSP solver provenance (locked solver, proven
optimal status, deterministic persisted subset, floor concentration caps,
independent witness re-verification), and manifest/seal round trips. Writes
nothing outside ``artifacts/sprint-15/proof-candidate-7/`` and prints a
machine-readable summary (the worker records it in
``artifacts/sprint-15/task-9-cycle-7.md``).

Design/Fit/Calibration/Confirmation/Sealed roots are never touched here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROOF_DIR = Path("artifacts/sprint-15/proof-candidate-7")
ROSTER = (("H-PROOF-13", 1616), ("H-PROOF-14", 1617))
REPEAT_DIR = PROOF_DIR / "repeat-determinism"


def run_cli(role: str, seed: int, outdir: Path) -> dict:
    """Materialize one proof root in a fresh process via the public CLI."""
    cmd = [sys.executable, "-m", "synth.cli", "--chronological",
           "--profile", "sprint15-v7", "--seed", str(seed),
           "--role", role, "--protocol", "sprint15-benchmark-protocol-v7",
           "--output", str(outdir)]
    proc = subprocess.run(
        ["uv", "run"] + cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"public entry point failed for {role}:\n{proc.stderr[-4000:]}")
    manifest = json.loads((outdir / "manifest.json").read_text())
    return {"manifest": manifest, "stdout": proc.stdout.strip()}


def main() -> int:
    from synth import balanced as B
    from synth import events as E
    from synth.chronicle import load_chronological, verify_seal, write_seal

    results = []
    for role, seed in ROSTER:
        outdir = PROOF_DIR / role
        if outdir.exists():
            raise FileExistsError(
                f"proof root exists (refusing overwrite): {outdir}")
        first = run_cli(role, seed, outdir)
        manifest = first["manifest"]
        rows = manifest["files"]
        ledger = E.failure_ledger(manifest)
        wins = manifest["maintenance_windows"]
        errors = []

        block = manifest.get("sprint15") or {}
        if manifest.get("protocol") != B.S15_PROTOCOL_V7:
            errors.append("protocol-tag")
        if block.get("protocol") != B.S15_PROTOCOL_V7:
            errors.append("block-protocol")
        if block.get("profile") != B.S15_PROFILE_V7:
            errors.append("block-profile")
        if manifest.get("role") != role:
            errors.append("role-match")
        if manifest["seeds"]["health"] != seed:
            errors.append("seed-match")
        if manifest["resolved_config"]["health"].get(
                "min_duration_gate") is not True:
            errors.append("gate-flag-off")

        samples, _ = load_chronological(outdir)
        if [s.file_id for s in samples] != [r["file_id"] for r in rows]:
            errors.append("reload-order")
        for sample in samples:
            sample.validate()

        by_robot: dict = {}
        for row in rows:
            by_robot.setdefault(row["robot_id"], []).append(row)
        for robot, file_rows in by_robot.items():
            ordered = sorted(file_rows,
                             key=lambda r: (r["start_time"], r["end_time"]))
            for prev, cur in zip(ordered, ordered[1:]):
                if not cur["start_time"] >= prev["end_time"] - 1e-6:
                    errors.append(f"robot-serialization:{robot}")
                    break

        for record in ledger:
            if record["cohort"] == "A":
                if not (record["duration_d"] == 0.0
                        and record["degradation_onset"] is None
                        and record["subtype"] in ("A1", "A2")):
                    errors.append("a-shape")
                    break
        p_durs = sorted(r["duration_d"] for r in ledger
                        if r["cohort"] == "P")
        w_durs = sorted(r["duration_d"] for r in ledger
                        if r["cohort"] == "W")
        if p_durs and not (min(p_durs) >= 2.0 and max(p_durs) <= 15.0):
            errors.append("p-dist-shape")
        if w_durs and not (min(w_durs) >= 6.0 and max(w_durs) <= 28.0):
            errors.append("w-dist-shape")

        allocation = (block.get("allocation") or {})
        selected = allocation.get("selected", {})
        for cohort, subs in (("P", (("P1", 12), ("P2", 12))),
                             ("W", (("W1", 12), ("W2", 12))),
                             ("A", (("A1", 8), ("A2", 8)))):
            for subtype, need in subs:
                if len(selected.get(cohort, {}).get(subtype, [])) != need:
                    errors.append(f"quota:{cohort}/{subtype}")
        if len(allocation.get("controls", [])) < 48:
            errors.append("quota:controls")

        solver = allocation.get("solver") or {}
        if solver.get("solver") != "scipy.optimize.milp (HiGHS)":
            errors.append("solver-identity")
        if solver.get("scipy_version") != "1.18.1":
            errors.append("solver-version")
        if solver.get("solver_status") != 0:
            errors.append("solver-status")
        if set(solver) != set(B.SOLVER_MANIFEST_KEYS):
            errors.append("solver-persisted-keys")
        if "solve_s" in solver:
            errors.append("solver-nondeterministic-keys")

        el, rej, _ = B.eligible_anchors(rows, ledger, wins)
        audit = B.audit_sprint15(manifest, profile=B.S15_PROFILE_V7, protocol=B.S15_PROTOCOL_V7)
        if audit["design_state"] != "DESIGN-PASS":
            failed = [k for k, v in audit["promotion_checks"].items()
                      if not v]
            errors.append(f"audit:{failed}+{audit['task12_errors']}")

        if not (block.get("nuisance") or {}).get("pass", False):
            errors.append("nuisance-envelope")
        margin = (block.get("signal_margin") or {})
        if not margin.get("pass", False):
            errors.append("signal-margin")

        if any(set(r) - B.ALLOWED_ROW_KEYS for r in rows):
            errors.append("row-leakage")
        if any("subtype" in r or "cohort" in r for r in rows):
            errors.append("row-hidden-state")

        sel_ids = set(allocation.get("selected_ids", []))
        sel_moments = {}
        for failure in ledger:
            if failure["failure_id"] in sel_ids:
                sel_moments.setdefault(failure["robot_id"], []).append(
                    failure["failure_time"])
        for robot, moments in sel_moments.items():
            for prev, cur in zip(sorted(moments), sorted(moments)[1:]):
                if cur - prev < 14 * 86400.0 - 1e-6:
                    errors.append(f"no-overlap:{robot}")
                    break

        prog_by_op = {(r["robot_id"], r["end_time"]): r["program_id"]
                      for r in rows}
        per_robot_sel = Counter()
        per_program_sel: dict = {}
        for failure in ledger:
            if failure["failure_id"] not in sel_ids:
                continue
            per_robot_sel[failure["robot_id"]] += 1
            if failure["cohort"] in ("P", "W"):
                prog = prog_by_op.get((failure["robot_id"],
                                       failure["failure_time"]))
                if prog is not None:
                    per_program_sel.setdefault(
                        (failure["cohort"], prog), 0)
                    per_program_sel[(failure["cohort"], prog)] += 1
        if any(v > 22 for v in per_robot_sel.values()):
            errors.append("floor-cap:robot")
        if any(v > 14 for v in per_program_sel.values()):
            errors.append("floor-cap:program")

        program = B.build_allocation_program(el, rows, B.DEFAULT_QUOTA)
        by_id = {f["failure_id"]: f for f in el}
        breaches = B.validate_allocation_witness(
            sorted(sel_ids), program, by_id, B.DEFAULT_QUOTA.spacing_s)
        if breaches:
            errors.append(f"witness-reverify:{breaches}")

        seal = write_seal(outdir, role)
        roundtrip = verify_seal(outdir)
        if roundtrip["manifest_sha256"] != seal["manifest_sha256"]:
            errors.append("seal-roundtrip")

        digest = hashlib.sha256(
            (outdir / "manifest.json").read_bytes()).hexdigest()
        results.append({"role": role, "seed": seed,
                        "manifest_sha256": digest,
                        "allocated": {c: {s: len(v) for s, v in d.items()}
                                      for c, d in selected.items()},
                        "controls": len(allocation.get("controls", [])),
                        "margins": allocation.get("margins", {}),
                        "rejection": dict(rej),
                        "solver": solver,
                        "audit_state": audit["design_state"],
                        "errors": errors})

    rep = REPEAT_DIR / "H-PROOF-13"
    repeat = run_cli("H-PROOF-13", 1616, rep)
    repeat_digest = hashlib.sha256(
        (rep / "manifest.json").read_bytes()).hexdigest()
    first_digest = next(r["manifest_sha256"] for r in results
                        if r["role"] == "H-PROOF-13")
    determinism = {
        "byte_identical_manifest": repeat_digest == first_digest,
        "first": first_digest,
        "repeat": repeat_digest,
    }
    summary = {"histories": results, "determinism": determinism,
               "verdict": ("PROOF-PASS" if all(not r["errors"]
                           for r in results)
                           and determinism["byte_identical_manifest"]
                           else "PROOF-FAIL")}
    print(json.dumps(summary, indent=1, sort_keys=True))
    if summary["verdict"] != "PROOF-PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
