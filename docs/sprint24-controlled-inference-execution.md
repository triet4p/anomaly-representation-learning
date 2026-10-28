- Execution ID: `S24-CONTROLLED-INFERENCE-EXECUTION-v1`.
- Task / attempt / agent: `S24-7-M11` / `S24-7-M11-A04` / `S24M11GoldA04` (role Gold; correction candidate continuing A03's uncommitted correction).
- Criterion: `docs/sprint-plans/sprint-24.md:211-215` (M11 acceptance at `:211-213`, host release at `:214`, deployment dependency at `:215`). No new scientific success gate is introduced.
- Bound contract: `f4f92a60ee1782ccc0ec75587e9df5d1471e4045e201c43478893d294b5757f7` (`artifacts/sprint-24/evidence/s24-7-m10-a02/prospective-contract.json`).
- Status: **CORRECTED DEPLOYABLE CANDIDATE — S24-7-M11-DEPLOY-C2/R1 submitted; corrected real reanalysis pending.** Exact deployment commit `c8b30c434330fd86ce29e77059b1fdd76c39dfee` ran the real bounded smoke (24 forwards, `SMOKE-COMPLETE`) and the real full capture (24,840 forwards, `ADMITTED`, `EXECUTED-REAL-EMBEDDINGS`), and its movement output is preserved but SUPERSEDED (§3): review found equal-unit exposure geometry and week+month-pooled split summaries. The corrected analysis path (actual per-file overlap from manifest end times plus NPZ header durations in retained order, per-horizon splits, estimable-only bootstrap annotation) is verified locally and frozen for the deployment correction gate; §3A will report the corrected cache-only reanalysis of the same admitted cache (expected 0 new forwards). Descriptive/inconclusive result only; no business-validity or per-file-accuracy claim. No final-execution gate PASS is claimed.
- Worker artifact: `artifacts/sprint-24/task-7-m11-a04.md` (correction candidate; A03 record `artifacts/sprint-24/task-7-m11-a03.md` holds the real execution, A02 `artifacts/sprint-24/task-7-m11-a02.md`, A01 `artifacts/sprint-24/task-7-m11.md`, handoffs preserved). Evidence: `artifacts/sprint-24/evidence/s24-7-m11-a04/` (correction verification); real-run evidence remains `artifacts/sprint-24/evidence/s24-7-m11-a03/` (smoke-run.json, full-run.json, movement.json, resume-run.json, full-stdout.log, verification.md).

## 1. What was built (and verified to the fail-closed boundary)

- `src/representation/sprint24_controlled_inference_execution.py` — contract-bound capture plan (byte-identical to the frozen identity-plan groups), identity-bound per-group cache with honest resume (skip-complete only, no duplicate forwards, no partial top-ups), admission through the accepted public path, identity-plan-driven group loading (committed loader keeps the old-study 12-group gate, so the candidate loads exactly the frozen 18-group plan after re-running the committed inspection/coverage/admission), frozen movement on admitted real embeddings (per-history/seed reference fit exactly once per group per run, actual per-file overlap weights from manifest end times plus NPZ `x.npy` shape-header durations in retained order, all retained pairs, C1 + history-cluster uncertainty annotated with the estimable support it actually draws on, per-horizon primary/replication splits, explicit transition-vs-control/dependence/N1/Holm records, C2 ABSTAIN, no success gate).
- `experiments/sprint24_task7m11_controlled_inference_execution.py` — public `smoke` (24 forwards) / `full` (24,840 forwards) CLI. Verifies the canonical freeze, pinned corpus/audit/preflight digests and the contract binding gate BEFORE any computation; refuses non-empty `--out`, missing corpus, digest drift and identity mismatch; full mode enforces `--expected-execution-commit` == HEAD with per-group resume and GPU provenance.
- `tests/representation/test_sprint24_controlled_inference_execution.py` — 13 focused checks, all passing: contract/identity refusals, cache admission and resume, CLI refusal, the exact expected-overlap weighted-MMD fixture (real manifest end times plus NPZ header durations through the public M11 movement path, hand-derived weights and a closed-form expected value, not a `weighted != biased` inequality), its simultaneous-permutation invariance, per-horizon split keys, estimable-only bootstrap support annotation, and the C2/dependence/N1/Holm records.

## 2. What was executed on real data (exact commit `c8b30c4`, GPU server `trietlm@192.168.99.214`)

- Execution lineage: local HEAD `c8b30c434330fd86ce29e77059b1fdd76c39dfee` == `origin/master` (ordinary non-force push) == remote isolated checkout `/tmp/s24-7-m11-a03-run/repo` (detached, clean). The main server checkout stayed at `23642b4` with foreign untracked experiment dirs preserved; never reset/cleaned. Corpus/evidence transferred outside git with 5/5 sha256 match (manifest `9606648d…`, freeze `be6e1058…`, audit `5fddabc1…`, contract `34031e8c…`, preflight `26f9e45d…`); bank bytes 3/3 (`546e515d…`/`59b95f3c…`/`c234c0b6…`) and checkpoint bytes 3/3 (`9292d7c5…`/`8ae58da9…`/`79edbdf3…`) re-verified; CUDA `torch 2.14.0+cu130`, `cuda: True`, RTX 4060 Ti. All run records carry `source_commit=c8b30c4…` and `contract_binding_verified_pre_execution=true`.
- Real bounded smoke: `SMOKE-COMPLETE`, 24 forwards (8 files x 3 seeds, history `S24C-ITER-0001-CONTROLLED-01`), wall ~6 s. Evidence `artifacts/sprint-24/evidence/s24-7-m11-a03/smoke-run.json` (sha256 `8a384df8…`).
- Real full capture: `EXECUTED-REAL-EMBEDDINGS`, 24,840 forwards (8,280 files x 3 seeds), 18/18 groups captured with 0 resumed on first pass, admission `ADMITTED` 24,840/24,840 with 0 missing, wall ~1,476 s (~24.6 min). Evidence `artifacts/sprint-24/evidence/s24-7-m11-a03/full-run.json` (sha256 `0750acea…`).
- Resume idempotency (same cache, same commit): second full run `EXECUTED-REAL-EMBEDDINGS`, 0 new forwards, 0 captured / 18 resumed, wall ~5.3 s; all numeric endpoints identical (0 mismatches at 1e-9); C2/splits/`n_computed` identical. Evidence `artifacts/sprint-24/evidence/s24-7-m11-a03/resume-run.json` (sha256 `2c852f02…`).
- Identity/refusal boundaries on real inputs: contract binding OK with 18 groups via the CLI path; non-empty `--out` refused (`REFUSED-OVERWRITE`); commit mismatch refused (`REFUSED-IDENTITY`); foreign smoke-provenance cache refused admission. Pre-execution history (A01 local smoke refusal exit 2 with 0 forwards; A02 synthetic-only plumbing `ADMITTED` 24,840/24,840 with no real embeddings) is preserved in the A01/A02 worker records and is superseded by the real runs above.

## 3. Measured movement result — SUPERSEDED BY DEPLOY-C1/R3 CORRECTION (C1/R1 changes_requested)

The values in this section report the `c8b30c4` real-data run faithfully, but review found four analysis defects that change the reported sensitivity/summaries: (1) `E1_weighted` used fabricated unit-interval geometry (all weights equal, collapsing toward `E1_biased`) instead of the frozen actual-overlap weighting; (2) primary/replication split medians pooled week+month cells against the contract no-pooling-horizons term; (3) transition-vs-control/dependence/Holm/N1 reporting incomplete; (4) uncertainty note states n=4 while six histories executed. The `c8b30c4` movement.json above is preserved as the superseded analysis (counts/provenance intact); the corrected reanalysis runs cache-only (0 new forwards) after the R3 deployment commit + checkpoint. Corrected values follow in §3A after reanalysis; do not cite §3 values as the final result.

- 60 records (30 week + 30 month, all `calendar` mode); dispositions 48 FEASIBLE (`BASELINE-VS-ELEVATED`) + 12 LIMITED (`BASELINE-VS-BASELINE` controls). Raw arm exposure on transitions 0.439-0.731; controls 0.936-0.997. Retained matched-file exposure admits all 60 pairs. Capped-`n` per-period sizes 52-289 files. Burden contrast gap after matching -0.013 to 0.322 (median 0.211). Evidence `artifacts/sprint-24/evidence/s24-7-m11-a03/movement.json` (sha256 `2534b8d1…`, 120,520 bytes).
- Endpoints: 180/180 COMPUTED cells (60 pairs x 3 seeds), `n_expected=180`. Per cell `E1` (unbiased squared MMD, new-pool Gaussian bandwidth): -0.0061 to 0.0051, median -0.0010. `E1_weighted` finite 180/180: 0.0018-0.0169, median 0.0066. `E1_biased`/`E1_capped` recorded alongside. `E2_K8` (K=8 cluster occupancy): 0.031-0.297, median 0.123 (`E2_K4`/`E2_K16` sensitivities recorded). `E3`: 0.50-5.99, median 2.18. `E4`: 0.0006-0.0675, median 0.0159. Null controls `N2_current` (-0.0115 to 0.0295, median -0.0002) and `N2_previous` (-0.0125 to 0.0186, median -0.0006) with `N2_max` recorded; no crossing claim is made (crossing is necessary, never sufficient).
- Reference fits: exactly 18 (one per history/seed, never per pair), each `n_reference=220` causal files, bandwidth sigma 10.74-12.12 (median 11.75), with null median/q95 recorded per fit.
- C1 descriptive association (5-pair sets per history/horizon, control as low-burden pole): all 12 history/horizon cells estimable; seed-median rho values are mixed in sign (week: 0.3, 0.1, 0.0, -0.2, -0.1, -0.6; month: 0.1, -0.7, 0.1, -0.6, -0.1, 0.3). No disposition depends on rho; no success gate is applied.
- C2: `ABSTAIN-CORPUS-WIDE` (zero-variance regressor; never independent confirmation of C1).
- Primary/replication summaries: primary histories 01-03 median rho 0.1 (6/6 estimable); replication histories 04-06 median rho -0.15 (6/6 estimable). Descriptive replication only, no gate.
- History-level uncertainty (bootstrap B=2000, seed 2407, descriptive at small n): week point median -0.05, 95% interval [-0.40, 0.20]; month point median 0.0, 95% interval [-0.65, 0.20]. Intervals are descriptive, never precise standard errors.
- Five-pair rank reference `2/5! = 0.0167` recorded as arithmetic support only.
- Overall reading: movement is observed and fully computed under the frozen contract, but C1 signs are mixed across histories/horizons/seeds, null bands straddle zero, and uncertainty intervals cover zero — an inconclusive/descriptive outcome with no performance, detection-accuracy, business-density, or deployment claim.

### 3A. Corrected measured movement result (pending cache-only reanalysis)

No corrected real values exist yet. The corrected code is frozen for `S24-7-M11-DEPLOY-C2/R1`; after a matching PASS and a fresh Bronze checkpoint, the same admitted full cache (`/tmp/s24-7-m11-a03-run/cache-full`) is re-analysed into a FRESH output directory with `--expected-execution-commit <corrected-commit>` and no cache-root change (expected 0 new forwards; any cache-identity/custody mismatch fails closed and is reported rather than replaced by new forwards). This section will then record only that real output: actual-overlap `E1_weighted` beside `E1_biased` (180/180 finite), per-horizon primary/replication split medians, the estimable history count and interval support per horizon, transition-vs-control counts, the dependence declaration, the N1 reference-null identity and the vacuous Holm policy — with the scientific criteria unchanged. The §3 `c8b30c4` values remain the superseded outcome; the local synthetic-embedding fixture proves the geometry only and is never cited as corpus evidence.

## 4. Execution custody (measured, no placeholders)

All custody below was verified by code-executed digest checks on the exact commit, not planned. The capable host holds CUDA + accepted bank/checkpoint bytes (re-verified 3/3 + 3/3 at execution); the deployment commit `c8b30c4` was pushed and checked out isolated; the out-of-git corpus/evidence transfer verified 5/5 by sha256; the exact-commit `--expected-execution-commit c8b30c4` runs are recorded with `source_commit`. Certified FIT support root is absent (rebuild closed, refit forbidden). Full mode refuses commit mismatch, non-empty `--out`, missing corpus, digest drift and identity mismatch before any forward.

## 5. Geometric fitting vs bank dependency

M10 permits per-new-history causal geometric reference fitting, and that fitting consumes embedding matrices only (proven COMPUTED with zero bank involvement). The bank is nevertheless unavoidable: accepted capture requires the restored `ProvenancedBank`, the bank digest is bound into the cache identity, and no model-only capture path exists. The blocker is the accepted bytes + bound device, not a legacy-loader artifact.

## 6. Reproduction (exact measured commands; fresh outputs, resume vs new capture)

On the authorized host, from the exact-commit checkout (bulk cache stays server-side, uncommitted). Each CLI invocation below writes to a FRESH, previously nonexistent `--out` directory — the CLI refuses a non-empty `--out` (`REFUSED-OVERWRITE`), so reusing `out-full`/`out-smoke` for a second command would refuse, not overwrite. Resume (second `full` against the SAME `--cache-root`) adds 0 forwards; only a FRESH `--cache-root` (or deleted group files) triggers new forwards:

```bash
python experiments/sprint24_task7m11_controlled_inference_execution.py \
  --corpus /tmp/s24-7-m11-a03-run/corpus \
  --audit /tmp/s24-7-m11-a03-run/audit/corpus-audit.json \
  --contract /tmp/s24-7-m11-a03-run/contract/prospective-contract.json \
  --g2-preflight /tmp/s24-7-m11-a03-run/preflight/preflight.json \
  --cache-root /tmp/s24-7-m11-a03-run/cache-smoke \
  --out /tmp/s24-7-m11-a03-run/out-smoke-FRESH \
  --mode smoke --b0-root /home/trietlm/s18-t23-a04-runs/full --bank-root /home/trietlm/s18-t32-a02-runs/scores
python experiments/sprint24_task7m11_controlled_inference_execution.py \
  --corpus /tmp/s24-7-m11-a03-run/corpus \
  --audit /tmp/s24-7-m11-a03-run/audit/corpus-audit.json \
  --contract /tmp/s24-7-m11-a03-run/contract/prospective-contract.json \
  --g2-preflight /tmp/s24-7-m11-a03-run/preflight/preflight.json \
  --cache-root /tmp/s24-7-m11-a03-run/cache-full \
  --out /tmp/s24-7-m11-a03-run/out-full-FRESH \
  --mode full --expected-execution-commit <exact-commit> \
  --b0-root /home/trietlm/s18-t23-a04-runs/full --bank-root /home/trietlm/s18-t32-a02-runs/scores
python experiments/sprint24_task7m11_controlled_inference_execution.py \
  --corpus /tmp/s24-7-m11-a03-run/corpus \
  --audit /tmp/s24-7-m11-a03-run/audit/corpus-audit.json \
  --contract /tmp/s24-7-m11-a03-run/contract/prospective-contract.json \
  --g2-preflight /tmp/s24-7-m11-a03-run/preflight/preflight.json \
  --cache-root /tmp/s24-7-m11-a03-run/cache-full \
  --out /tmp/s24-7-m11-a03-run/out-resume-FRESH \
  --mode full --expected-execution-commit <exact-commit> \
  --b0-root /home/trietlm/s18-t23-a04-runs/full --bank-root /home/trietlm/s18-t32-a02-runs/scores
python -m pytest tests/representation/test_sprint24_controlled_inference_execution.py -q
```

With `data/generated/sprint24_controlled_v2` as `--corpus` and the in-repo audit/contract/preflight paths, the same CLI reproduces the gates locally up to the CUDA/bank requirement (A01 refusal path). Certified FIT support root is absent (rebuild closed, refit forbidden).

## 7. Forward accounting, storage, runtime

- Historical observed forwards: 52,789. This attempt: 24 smoke + 24,840 full + 0 failed/repeated = 24,864 new (actual counted, not planned); resume rerun added 0. Cumulative: 77,653.
- New code: ~1,100 lines (module + CLI + tests). Evidence: `smoke-run.json` (~1 kB) + `full-run.json` (~1.7 kB) + `movement.json` (120,520 bytes) + `resume-run.json` (~1.7 kB).
- Smoke wall ~6 s; full wall ~1,476 s; resume wall ~5.3 s (digest verification + binding recomputation over 8,280 manifest files included). Server-side cache: full 15 MB (36 files), smoke 40 KB; outputs ~242 KB total.

## 8. Limitations

- Real execution only: embeddings, forwards, reference fits, movement values and dispositions above are measured on real data under the frozen contract. Monthly horizons use the same capped-`n` rule as weekly (support accounting as executed; descriptive only). Uncertainty intervals are descriptive at small `n`. No positive result gate is authorized or applied; no performance, detection-accuracy, business-density, training/refit, Confirmation, or Phase C claim follows.

