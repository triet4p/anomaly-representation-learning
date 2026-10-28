- Execution ID: `S24-CONTROLLED-INFERENCE-EXECUTION-v1`.
- Task / attempt / agent: `S24-7-M11` / `S24-7-M11-A02` / `S24M11GoldA02` (role Gold).
- Criterion: `docs/sprint-plans/sprint-24.md:211-215` (deployment review at `:215`). No new scientific success gate is introduced.
- Bound contract: `f4f92a60ee1782ccc0ec75587e9df5d1471e4045e201c43478893d294b5757f7` (`artifacts/sprint-24/evidence/s24-7-m10-a02/prospective-contract.json`).
- Status: **DEPLOYABLE CANDIDATE — S24-7-M11-DEPLOY-C1/R1 submitted.** A01 fail-closed integration owned and fixed by A02 (loader 18-group fix, per-run live-fit fix, unit-spacing exposure-geometry fix); remote prerequisites verified read-only (CUDA + bank 3/3 + checkpoint 3/3 present on `trietlm@192.168.99.214`); real execution still requires the deployment commit + corpus transfer + exact-commit run. No movement outcome on real data is reported. No final-execution gate PASS is claimed.
- Worker artifact: `artifacts/sprint-24/task-7-m11-a02.md` (deployable candidate; A01 record `artifacts/sprint-24/task-7-m11.md` and handoff `artifacts/sprint-24/handoffs/task-7-m11-a01.md` preserved).

## 1. What was built (and verified to the fail-closed boundary)

- `src/representation/sprint24_controlled_inference_execution.py` — contract-bound capture plan (byte-identical to the frozen identity-plan groups), identity-bound per-group cache with honest resume (skip-complete only, no duplicate forwards, no partial top-ups), admission through the accepted public path, identity-plan-driven group loading (committed loader keeps the old-study 12-group gate, so the candidate loads exactly the frozen 18-group plan after re-running the committed inspection/coverage/admission), frozen movement on admitted real embeddings (per-history/seed reference fit exactly once per group per run, unit-spacing exposure geometry, all retained pairs, C1 + history-cluster uncertainty + primary/replication splits, C2 ABSTAIN, no success gate).
- `experiments/sprint24_task7m11_controlled_inference_execution.py` — public `smoke` (24 forwards) / `full` (24,840 forwards) CLI. Verifies the canonical freeze, pinned corpus/audit/preflight digests and the contract binding gate BEFORE any computation; refuses non-empty `--out`, missing corpus, digest drift and identity mismatch; full mode enforces `--expected-execution-commit` == HEAD with per-group resume and GPU provenance.
- `tests/representation/test_sprint24_controlled_inference_execution.py` — 9 focused checks, all passing.

## 2. What was executed and validated (and what was not)

- Real public-CLI smoke through the executable path (A01, local host): exit 2, `REFUSED-PREREQUISITE`, 0 forwards, contract binding verified pre-execution, all four digests verified. The refusal names the exact missing bundle roots and CUDA requirement. Evidence: `artifacts/sprint-24/evidence/s24-7-m11-a01/run.json`.
- Remote prerequisite verification (A02, read-only SSH, no remote execution): server `trietlm@192.168.99.214` has CUDA (`torch 2.14.0+cu130`, RTX 4060 Ti), accepted bank bytes 3/3 digest-exact, accepted checkpoint bytes 3/3 digest-exact, runnable toolchain. Missing on server: M9 corpus (126 MB), audit/contract/preflight, deployment commit — transfer + exact-commit run still required. Evidence: `artifacts/sprint-24/evidence/s24-7-m11-a02/verification.md`.
- Full deployable-path validation on synthetic matrices through the REAL writer/admission/movement code (A02, local): ADMITTED 24,840/24,840 with 0 missing; 18/18 groups loaded; exactly 18 reference fits per run (once per history/seed, never per pair); 60/60 records with 180/180 COMPUTED endpoints and non-None `E1_weighted` on all 180 seed-cells (unit-spacing geometry); C2 `ABSTAIN-CORPUS-WIDE`; primary/replication splits estimable 6/6 cells each; two sequential full runs in one process isolated (no shared live fits). Matrices are plumbing-only, never corpus evidence. Evidence: `artifacts/sprint-24/evidence/s24-7-m11-a02/synthetic-pipeline.json`.
- NOT executed and NOT claimed: no forward, no model/bank load, no embedding, no reference fit on real data, no endpoint value on real data, no disposition, no per-file accuracy, no business-density claim, no training/refit, no Confirmation, no Phase C.

## 3. Deployment prerequisites (no placeholders)

See the worker artifact's prerequisite table. In short: the capable host holds CUDA + accepted bank/checkpoint bytes (verified read-only); the deployment commit (candidate file list in `artifacts/sprint-24/task-7-m11-a02.md`), the out-of-git corpus/evidence transfer with digest verify, and the exact-commit `--expected-execution-commit` run remain for Main + Bronze. Certified FIT support root is absent (rebuild closed, refit forbidden). Full mode refuses commit mismatch, non-empty `--out`, missing corpus, digest drift and identity mismatch before any forward.

## 4. Geometric fitting vs bank dependency

M10 permits per-new-history causal geometric reference fitting, and that fitting consumes embedding matrices only (proven COMPUTED with zero bank involvement). The bank is nevertheless unavoidable: accepted capture requires the restored `ProvenancedBank`, the bank digest is bound into the cache identity, and no model-only capture path exists. The blocker is the accepted bytes + bound device, not a legacy-loader artifact.

## 5. Reproduction

```bash
python experiments/sprint24_task7m11_controlled_inference_execution.py \
  --corpus data/generated/sprint24_controlled_v2 \
  --audit artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json \
  --contract artifacts/sprint-24/evidence/s24-7-m10-a02/prospective-contract.json \
  --cache-root <identity-bound cache dir> \
  --out artifacts/sprint-24/evidence/s24-7-m11-a01 \
  --mode smoke --b0-root <ckpt dir> --bank-root <bank dir>
python -m pytest tests/representation/test_sprint24_controlled_inference_execution.py -q
```

With the prerequisites satisfied on an authorized host, `smoke` captures 24 forwards with `gpu-forward-smoke` provenance, then `full --expected-execution-commit <HEAD>` captures 24,840 with `gpu-forward` provenance, admits, and runs the frozen movement.

## 6. Forward accounting, storage, runtime

- Historical observed forwards: 52,789 (unchanged). This attempt: 0. Planned on release: 24 + 24,840 = 24,864 new.
- New code: ~1,100 lines (module + CLI + tests). Evidence: single `run.json` (~1.5 kB) + test report.
- Smoke refusal wall time: ~5–13 s (digest verification + binding recomputation over 8,280 manifest files).

## 7. Limitations

Blocked execution only: no embedding, forward, reference fit, movement value, disposition or performance claim follows. Monthly horizons use the same capped-`n` rule as weekly (observed support accounting, not validated monthly inference). No positive result gate is authorized or applied.
