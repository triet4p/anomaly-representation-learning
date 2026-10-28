- Execution ID: `S24-CONTROLLED-INFERENCE-EXECUTION-v1`.
- Task / attempt / agent: `S24-7-M11` / `S24-7-M11-A04` / `S24M11GoldA04` (role Gold; corrected result produced by A04 continuing A03's uncommitted correction; C2 reporting-only wording correction by `S24-7-M11-A05` / `S24M11GoldA05`, no data, code or cache change).
- Criterion: `docs/sprint-plans/sprint-24.md:211-215` (M11 acceptance at `:211-213`, host release at `:214`, deployment dependency at `:215`). No new scientific success gate is introduced.
- Bound contract: `f4f92a60ee1782ccc0ec75587e9df5d1471e4045e201c43478893d294b5757f7` (`artifacts/sprint-24/evidence/s24-7-m10-a02/prospective-contract.json`).
- Status: **CORRECTED RESULT — S24-7-M11-C2 reporting-only correction; C1/R3 FAIL preserved.** Deployment-correction commit `b7890d5bfb497f1ddcebdcf7e99dc57ae2d35165` passed `S24-7-M11-DEPLOY-C2/R1` and its Bronze checkpoint; the corrected analysis then re-used the SAME admitted cache (`/tmp/s24-7-m11-a03-run/cache-full`, original capture commit `c8b30c434330fd86ce29e77059b1fdd76c39dfee`) with **0 new forwards** (18/18 groups resumed on the frozen identity, admission `ADMITTED` 24,840/24,840) and produced the corrected result in §3A. R2's reporting items — the within-history/horizon transition-vs-control comparison, the fresh reproduction output path, and the raw-vs-retained exposure field disclosure — are addressed from the same real records without re-running inference. The superseded `c8b30c4` analysis is preserved in §3. Descriptive/inconclusive result only; no business-validity or per-file-accuracy claim; no success gate. C1 ended R1 `changes_requested` (`reviews/task-7-m11-evidence.md`), R2 `changes_requested` (`reviews/task-7-m11-final-r2-evidence.md`) and a Main-authorized R3 **FAIL** on the former unconditional no-separation sentence; those outcomes and all earlier DEPLOY/C1 records are preserved unchanged. The C2 correction (maximum two assessments, no automatic third) changes only that sentence's interpretation; no reported number, row, count or custody statement changed.
- Worker artifact: `artifacts/sprint-24/task-7-m11-a05.md` (C2 reporting correction; evidence `artifacts/sprint-24/evidence/s24-7-m11-a05/`); `artifacts/sprint-24/task-7-m11-a04.md` (corrected-result record; A03 record `artifacts/sprint-24/task-7-m11-a03.md` holds the real execution, A02 `artifacts/sprint-24/task-7-m11-a02.md`, A01 `artifacts/sprint-24/task-7-m11.md`, handoffs preserved). Evidence: `artifacts/sprint-24/evidence/s24-7-m11-a04/` (correction verification, `corrected-reanalysis.md`, corrected `movement.json`/`run.json`, `movement-summary.json`, `within-history-control-comparison.json`, `published-numbers-check.json`, logs); original capture evidence remains `artifacts/sprint-24/evidence/s24-7-m11-a03/` (smoke-run.json, full-run.json, movement.json, resume-run.json, full-stdout.log, verification.md).

## 1. What was built (and verified to the fail-closed boundary)

- `src/representation/sprint24_controlled_inference_execution.py` — contract-bound capture plan (byte-identical to the frozen identity-plan groups), identity-bound per-group cache with honest resume (skip-complete only, no duplicate forwards, no partial top-ups), admission through the accepted public path, identity-plan-driven group loading (committed loader keeps the old-study 12-group gate, so the candidate loads exactly the frozen 18-group plan after re-running the committed inspection/coverage/admission), frozen movement on admitted real embeddings (per-history/seed reference fit exactly once per group per run, actual per-file overlap weights from manifest end times plus NPZ `x.npy` shape-header durations in retained order, all retained pairs, C1 + history-cluster uncertainty annotated with the estimable support it actually draws on, per-horizon primary/replication splits, explicit transition-vs-control/dependence/N1/Holm records, C2 ABSTAIN, no success gate).
- `experiments/sprint24_task7m11_controlled_inference_execution.py` — public `smoke` (24 forwards) / `full` (24,840 forwards) CLI. Verifies the canonical freeze, pinned corpus/audit/preflight digests and the contract binding gate BEFORE any computation; refuses non-empty `--out`, missing corpus, digest drift and identity mismatch; full mode enforces `--expected-execution-commit` == HEAD with per-group resume and GPU provenance.
- `tests/representation/test_sprint24_controlled_inference_execution.py` — 13 focused checks, all passing: contract/identity refusals, cache admission and resume, CLI refusal, the exact expected-overlap weighted-MMD fixture (real manifest end times plus NPZ header durations through the public M11 movement path, hand-derived weights and a closed-form expected value, not a `weighted != biased` inequality), its simultaneous-permutation invariance, per-horizon split keys, estimable-only bootstrap support annotation, and the C2/dependence/N1/Holm records.

## 2. What was executed on real data (capture commit `c8b30c4`; corrected analysis code `b7890d5`, GPU server `trietlm@192.168.99.214`)

- Execution lineage: local HEAD `c8b30c434330fd86ce29e77059b1fdd76c39dfee` == `origin/master` (ordinary non-force push) == remote isolated checkout `/tmp/s24-7-m11-a03-run/repo` (detached, clean). The main server checkout stayed at `23642b4` with foreign untracked experiment dirs preserved; never reset/cleaned. Corpus/evidence transferred outside git with 5/5 sha256 match (manifest `9606648d…`, freeze `be6e1058…`, audit `5fddabc1…`, contract `34031e8c…`, preflight `26f9e45d…`); bank bytes 3/3 (`546e515d…`/`59b95f3c…`/`c234c0b6…`) and checkpoint bytes 3/3 (`9292d7c5…`/`8ae58da9…`/`79edbdf3…`) re-verified; CUDA `torch 2.14.0+cu130`, `cuda: True`, RTX 4060 Ti. All capture-run records carry `source_commit=c8b30c4…` and `contract_binding_verified_pre_execution=true`.
- Real bounded smoke: `SMOKE-COMPLETE`, 24 forwards (8 files x 3 seeds, history `S24C-ITER-0001-CONTROLLED-01`), wall ~6 s. Evidence `artifacts/sprint-24/evidence/s24-7-m11-a03/smoke-run.json` (sha256 `8a384df8…`).
- Real full capture: `EXECUTED-REAL-EMBEDDINGS`, 24,840 forwards (8,280 files x 3 seeds), 18/18 groups captured with 0 resumed on first pass, admission `ADMITTED` 24,840/24,840 with 0 missing, wall ~1,476 s (~24.6 min). Evidence `artifacts/sprint-24/evidence/s24-7-m11-a03/full-run.json` (sha256 `0750acea…`).
- Resume idempotency (same cache, same commit): second full run `EXECUTED-REAL-EMBEDDINGS`, 0 new forwards, 0 captured / 18 resumed, wall ~5.3 s; all numeric endpoints identical (0 mismatches at 1e-9); C2/splits/`n_computed` identical. Evidence `artifacts/sprint-24/evidence/s24-7-m11-a03/resume-run.json` (sha256 `2c852f02…`).
- Corrected cache-only reanalysis (analysis code commit `b7890d5bfb497f1ddcebdcf7e99dc57ae2d35165`, ordinary non-force push `c8b30c4..b7890d5`, isolated checkout detached at `b7890d5`, clean): the SAME admitted cache was reused — 18/18 groups complete on the frozen identity, admission `ADMITTED` 24,840/24,840 with 0 missing, **0 new forwards** (0 captured / 18 resumed), wall ~13 s, `source_commit=b7890d5…`. The 36 cache files were byte-identical before and after, and their sidecars retain the original `capture_commit=c8b30c4…`; no bank, checkpoint or encoder was refit or replaced. Evidence `artifacts/sprint-24/evidence/s24-7-m11-a04/movement.json` (sha256 `d8474744…`, 122,583 bytes), `run.json` (sha256 `87a71f8c…`), `corrected-reanalysis.md`, `movement-summary.json`.
- Identity/refusal boundaries on real inputs: contract binding OK with 18 groups via the CLI path; non-empty `--out` refused (`REFUSED-OVERWRITE`); commit mismatch refused (`REFUSED-IDENTITY`); foreign smoke-provenance cache refused admission. Pre-execution history (A01 local smoke refusal exit 2 with 0 forwards; A02 synthetic-only plumbing `ADMITTED` 24,840/24,840 with no real embeddings) is preserved in the A01/A02 worker records and is superseded by the real runs above.

## 3. Measured movement result — SUPERSEDED BY DEPLOY-C1/R3 CORRECTION (C1/R1 changes_requested)

The values in this section report the `c8b30c4` real-data run faithfully, but review found four analysis defects that change the reported sensitivity/summaries: (1) `E1_weighted` used fabricated unit-interval geometry (all weights equal, collapsing toward `E1_biased`) instead of the frozen actual-overlap weighting; (2) primary/replication split medians pooled week+month cells against the contract no-pooling-horizons term; (3) transition-vs-control/dependence/Holm/N1 reporting incomplete; (4) uncertainty note states n=4 while six histories executed. The `c8b30c4` movement.json above is preserved as the superseded analysis (counts/provenance intact); the corrected reanalysis ran cache-only (0 new forwards) under analysis code `b7890d5` after the deployment-correction PASS and Bronze checkpoint. Corrected values are in §3A; do not cite §3 values as the final result.

- 60 records (30 week + 30 month, all `calendar` mode); dispositions 48 FEASIBLE (`BASELINE-VS-ELEVATED`) + 12 LIMITED (`BASELINE-VS-BASELINE` controls). Raw arm exposure on transitions 0.439-0.731; controls 0.936-0.997. Retained matched-file exposure admits all 60 pairs. Capped-`n` per-period sizes 52-289 files. Burden contrast gap after matching -0.013 to 0.322 (median 0.211). Evidence `artifacts/sprint-24/evidence/s24-7-m11-a03/movement.json` (sha256 `2534b8d1…`, 120,520 bytes).
- Endpoints: 180/180 COMPUTED cells (60 pairs x 3 seeds), `n_expected=180`. Per cell `E1` (unbiased squared MMD, new-pool Gaussian bandwidth): -0.0061 to 0.0051, median -0.0010. `E1_weighted` finite 180/180: 0.0018-0.0169, median 0.0066. `E1_biased`/`E1_capped` recorded alongside. `E2_K8` (K=8 cluster occupancy): 0.031-0.297, median 0.123 (`E2_K4`/`E2_K16` sensitivities recorded). `E3`: 0.50-5.99, median 2.18. `E4`: 0.0006-0.0675, median 0.0159. Null controls `N2_current` (-0.0115 to 0.0295, median -0.0002) and `N2_previous` (-0.0125 to 0.0186, median -0.0006) with `N2_max` recorded; no crossing claim is made (crossing is necessary, never sufficient).
- Reference fits: exactly 18 (one per history/seed, never per pair), each `n_reference=220` causal files, bandwidth sigma 10.74-12.12 (median 11.75), with null median/q95 recorded per fit.
- C1 descriptive association (5-pair sets per history/horizon, control as low-burden pole): all 12 history/horizon cells estimable; seed-median rho values are mixed in sign (week: 0.3, 0.1, 0.0, -0.2, -0.1, -0.6; month: 0.1, -0.7, 0.1, -0.6, -0.1, 0.3). No disposition depends on rho; no success gate is applied.
- C2: `ABSTAIN-CORPUS-WIDE` (zero-variance regressor; never independent confirmation of C1).
- Primary/replication summaries: primary histories 01-03 median rho 0.1 (6/6 estimable); replication histories 04-06 median rho -0.15 (6/6 estimable). Descriptive replication only, no gate.
- History-level uncertainty (bootstrap B=2000, seed 2407, descriptive at small n): week point median -0.05, 95% interval [-0.40, 0.20]; month point median 0.0, 95% interval [-0.65, 0.20]. Intervals are descriptive, never precise standard errors.
- Five-pair rank reference `2/5! = 0.0167` recorded as arithmetic support only.
- Overall reading: movement is observed and fully computed under the frozen contract, but C1 signs are mixed across histories/horizons/seeds, null bands straddle zero, and uncertainty intervals cover zero — an inconclusive/descriptive outcome with no performance, detection-accuracy, business-density, or deployment claim.

### 3A. Corrected measured movement result (real cache-only reanalysis, analysis code `b7890d5`)

This is the corrected result, computed only from the real cache-only reanalysis output `artifacts/sprint-24/evidence/s24-7-m11-a04/movement.json` (sha256 `d8474744…`, analysis code `b7890d5…`, original capture commit `c8b30c4…`, 0 new forwards). The §3 values above remain the superseded outcome and are not cited as final; the local synthetic-embedding fixture proves the geometry only and is never corpus evidence.

- 60 records (30 week + 30 month, all `calendar` mode); dispositions 48 FEASIBLE (`BASELINE-VS-ELEVATED`) + 12 LIMITED (`BASELINE-VS-BASELINE` controls). Burden contrast gap after matching −0.013 to 0.322 (median 0.211); capped-`n` per-period sizes 52–289 files (identical admitted inputs to §3).
- Endpoints: 180/180 COMPUTED cells. `E1`: −0.006055 to 0.005086, median −0.000998. `E1_biased`: 0.001777 to 0.016897, median 0.006581. **`E1_weighted` (actual-overlap sensitivity): 0.001696 to 0.018111, median 0.008007; it differs from `E1_biased` on 180/180 cells (min |Δ| 1.5e-5, max |Δ| 5.3e-3), so the equal-unit collapse is gone.** `E1_capped`: −0.006055 to 0.005086, median −0.000998. `E2_K8`: 0.0311–0.2969, median 0.1234 (`E2_K4` 0.000–0.2545; `E2_K16` 0.0798–0.3750). `E3`: 0.4962–5.9859, median 2.1789. `E4`: 0.000600–0.06749, median 0.01592. Null bands `N2_current` −0.01155 to 0.02954 (median −0.000182), `N2_previous` median −0.000556, `N2_max` median 0.001691; no crossing claim is made.
- Reference fits: exactly 18 (one per history/seed, never per pair), each `n_reference=220` causal files, bandwidth sigma 10.7408–12.1219 (median 11.7546).
- C1 descriptive association: all 12 history/horizon cells estimable; seed-median rho week 0.3, 0.1, 0.0, −0.2, −0.1, −0.6 and month 0.1, −0.7, 0.1, −0.6, −0.1, 0.3. No disposition depends on rho.
- Record-field disclosure: the movement record's `exposure_ratio_after` is the **raw, pre-balancing arm ratio** copied from the M6 matched-comparison audit, which sets `exposure_ratio_after = exposure_ratio_before` (`src/representation/sprint24_weekly_matched_comparison.py:617-618`); in the frozen census its 48 transition values are 0.4393–0.7312 (all below the 0.8 floor) and its 12 control values are 0.9358–0.9971. It is **not** the retained exposure: the accepted retained-exposure census is 0.8520–0.9953 for transitions and 0.8852–0.9921 for controls (`docs/sprint24-controlled-inference-readiness.md:32-33`; frozen `eligibility-census.json`), admits all 60 pairs, and is computed separately on the retained matched files. No eligibility or analysis-set claim may be read from the raw field.
- Within-history/horizon movement versus baseline controls (derived from the same real records; descriptive only): per history × horizon, each of the 4 transition pairs and the single control pair is first reduced across the three encoder seeds to its seed-median endpoint (seeds are run replicates, never independent data), then the transition pairs are summarized by their median and min–max. Pairs sharing an arm remain dependent by construction; no p-value, threshold, gate or new estimand is applied.

| History | Horizon | E1 transitions: median [min, max] | E1 control | E1_weighted transitions: median | E1_weighted control | E2_K8 transitions: median | E2_K8 control | C1 rho |
|---|---|---|---|---|---|---|---|---|
| `01` | week | −0.001890 [−0.003233, −0.000730] | −0.001813 | 0.013201 | 0.009406 | 0.1766 | 0.1067 | 0.3 |
| `01` | month | 0.000056 [−0.001393, 0.001357] | 0.000014 | 0.005527 | 0.003168 | 0.1234 | 0.0801 | 0.1 |
| `02` | week | 0.000112 [−0.003216, 0.002435] | 0.001684 | 0.016914 | 0.012715 | 0.2364 | 0.1977 | 0.1 |
| `02` | month | −0.000242 [−0.001262, 0.001513] | 0.000592 | 0.005346 | 0.004072 | 0.1163 | 0.0796 | −0.7 |
| `03` | week | −0.003774 [−0.005863, 0.001287] | −0.000490 | 0.010201 | 0.009108 | 0.1687 | 0.1235 | 0.0 |
| `03` | month | −0.000559 [−0.001230, 0.000046] | −0.000970 | 0.005217 | 0.001901 | 0.1042 | 0.0852 | 0.1 |
| `04` | week | −0.000782 [−0.001669, 0.001927] | 0.002602 | 0.013739 | 0.012974 | 0.1847 | 0.1294 | −0.2 |
| `04` | month | −0.001508 [−0.001912, 0.001769] | 0.000332 | 0.003848 | 0.003865 | 0.1222 | 0.0971 | −0.6 |
| `05` | week | −0.005120 [−0.005556, −0.002954] | −0.000839 | 0.010016 | 0.012667 | 0.1464 | 0.1447 | −0.1 |
| `05` | month | −0.001456 [−0.002192, −0.001090] | −0.000298 | 0.004148 | 0.002492 | 0.0964 | 0.0915 | −0.1 |
| `06` | week | −0.002753 [−0.005770, 0.001488] | −0.000225 | 0.011419 | 0.012165 | 0.1536 | 0.1316 | −0.6 |
| `06` | month | −0.000651 [−0.002037, 0.000439] | −0.000481 | 0.003989 | 0.002383 | 0.0912 | 0.0842 | 0.3 |

Reading the table: the transition seed-median `E1` exceeds its control in only 2 of 12 history/horizon cells, and every history/horizon `E1` median (transitions and controls) lies within ±0.006, the same order of magnitude as the recorded N2 within-period null values (−0.0125 to 0.0295). Those are descriptive quantities, not a matched per-history/horizon noise-separation test: N2 is a deterministic within-period half-split band (`docs/sprint24-weekly-monthly-estimator-conventions-a01.json:95-110`), so setting its global range against these medians cannot establish that the unbiased-MMD contrast is separated from noise, or that it is not, anywhere. Three of the 180 individual `E1` cells do exceed their own fit's N1 q95 (`02|week|181803`, `04|week|181801`, `04|week control|181803`), but N1 crossing is necessary and never sufficient, so those isolated crossings neither confirm a promising reading nor justify a universal negative. No predeclared noise-separation criterion exists, and the existing descriptive summary establishes no consistent noise-separation either way; no disposition follows. `E1_weighted` exceeds the control in 9 of 12 cells (0.0038–0.0169 transitions vs 0.0019–0.0130 controls) and `E2_K8` occupancy-TV exceeds it in 12 of 12 (0.0912–0.2364 vs 0.0796–0.1977). Those are descriptive distribution contrasts across dependent shared-arm pairs, not independent samples, not a test, and not evidence of anomaly detection or business validity; they do not change the inconclusive reading.

- Per-horizon primary/replication summaries (no cross-horizon pooling): primary week 0.1, primary month 0.1, replication week −0.2, replication month −0.1; three cells and three estimable cells each.
- Transition-vs-control and dependence: 24 transitions / 6 controls per horizon (transition dispositions `FEASIBLE`, control dispositions `LIMITED`); comparisons sharing an arm are dependent by construction (`NONADJACENT-SHARED-SUPPORT`), so pair counts are support accounting, never independent samples. N1 is the reference split-half null recorded per fit; `C2 = ABSTAIN-CORPUS-WIDE`; the frozen Holm policy is recorded as `NOT-COMPUTED-NO-PVALUES` (no p-values exist, and no disposition depends on one).
- History-level uncertainty (bootstrap B=2000, seed 2407): 6 estimable histories of 6 with records, 0 non-estimable per horizon; week point median −0.05, 95% [−0.40, 0.20]; month point median 0.0, 95% [−0.65, 0.20]. Descriptive only, never a precise standard error.
- Five-pair rank reference `2/5! = 0.0167` recorded as arithmetic support only.
- Overall reading: unchanged — movement is fully computed under the frozen contract, but C1 signs are mixed, null bands straddle zero and the uncertainty intervals cover zero, so the outcome is inconclusive/descriptive with no performance, detection-accuracy, business-density, or deployment claim and no success gate.

## 4. Execution custody (measured, no placeholders)

All custody below was verified by code-executed digest checks on the exact commit, not planned. The capable host holds CUDA + accepted bank/checkpoint bytes (re-verified 3/3 + 3/3 at execution); the capture commit `c8b30c4` was pushed and checked out isolated; the out-of-git corpus/evidence transfer verified 5/5 by sha256; the exact-commit `--expected-execution-commit c8b30c4` runs are recorded with `source_commit`. Certified FIT support root is absent (rebuild closed, refit forbidden). Full mode refuses commit mismatch, non-empty `--out`, missing corpus, digest drift and identity mismatch before any forward.

The correction commit `b7890d5` was pushed with an ordinary non-force `c8b30c4..b7890d5` and checked out detached in the same isolated server checkout (clean; the main checkout at `23642b4` and its foreign untracked directories were never touched). Before any model load, the corrected code re-derived the frozen 18-group identity plan and found all 18 cache groups complete with full 1,380-row support; admission returned `ADMITTED` 24,840/24,840 with 0 missing. The 36 cache files were sha256-identical before and after the reanalysis, and their sidecars still record the original `capture_commit=c8b30c4…`; the new run record records `source_commit=b7890d5…`. No bank/checkpoint/encoder was fitted, refitted or replaced, and no new capture occurred.

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
  --mode full --expected-execution-commit <capture-or-correction-commit> \
  --b0-root /home/trietlm/s18-t23-a04-runs/full --bank-root /home/trietlm/s18-t32-a02-runs/scores
# corrected cache-only reanalysis: same admitted cache, fresh out path, 0 forwards
python experiments/sprint24_task7m11_controlled_inference_execution.py \
  --corpus /tmp/s24-7-m11-a03-run/corpus \
  --audit /tmp/s24-7-m11-a03-run/audit/corpus-audit.json \
  --contract /tmp/s24-7-m11-a03-run/contract/prospective-contract.json \
  --g2-preflight /tmp/s24-7-m11-a03-run/preflight/preflight.json \
  --cache-root /tmp/s24-7-m11-a03-run/cache-full \
  --out /tmp/s24-7-m11-a03-run/out-full-corrected-b7890d5-rerun \
  --mode full --expected-execution-commit b7890d5bfb497f1ddcebdcf7e99dc57ae2d35165 \
  --b0-root /home/trietlm/s18-t23-a04-runs/full --bank-root /home/trietlm/s18-t32-a02-runs/scores
python -m pytest tests/representation/test_sprint24_controlled_inference_execution.py -q
```

The measured A04 run wrote `/tmp/s24-7-m11-a03-run/out-full-corrected-b7890d5`; that directory is already populated, and the CLI refuses a non-empty `--out` (`REFUSED-OVERWRITE`), so the reproduction command above uses a distinct fresh `-rerun` path. On the exact `b7890d5` checkout it reuses all 18 complete groups and adds 0 forwards; the cache sidecars keep `capture_commit=c8b30c4…` while the new run record carries `source_commit=b7890d5…`.

With `data/generated/sprint24_controlled_v2` as `--corpus` and the in-repo audit/contract/preflight paths, the same CLI reproduces the gates locally up to the CUDA/bank requirement (A01 refusal path). Certified FIT support root is absent (rebuild closed, refit forbidden).

## 7. Forward accounting, storage, runtime

- Historical observed forwards before M11: 52,789. A03 capture: 24 smoke + 24,840 full + 0 failed/repeated = 24,864 new (actual counted). A04 corrected cache-only reanalysis: **0 new forwards** (0 captured / 18 resumed; 0 smoke). Cumulative remains **77,653**.
- New code: ~1,100 lines (module + CLI + tests). Evidence: `smoke-run.json` (~1 kB) + `full-run.json` (~1.7 kB) + `movement.json` (120,520 bytes) + `resume-run.json` (~1.7 kB); corrected `movement.json` 122,583 bytes + `run.json` (~1.7 kB) under `artifacts/sprint-24/evidence/s24-7-m11-a04/`.
- Smoke wall ~6 s; full capture wall ~1,476 s; resume wall ~5.3 s; corrected reanalysis wall ~13 s (run-record `elapsed_s` 10.73, including digest verification and binding recomputation over 8,280 manifest files). Server-side cache: full 15 MB (36 files), smoke 40 KB; A03 outputs ~242 KB, A04 corrected output ~124 KB.

## 8. Limitations

- Real execution only: embeddings, forwards, reference fits, movement values and dispositions above are measured on real data under the frozen contract. Monthly horizons use the same capped-`n` rule as weekly (support accounting as executed; descriptive only). Uncertainty intervals are descriptive at small `n`. No positive result gate is authorized or applied; no performance, detection-accuracy, business-density, training/refit, Confirmation, or Phase C claim follows.
- The §3A corrected analysis reuses the original `c8b30c4` capture bytes under corrected code `b7890d5`; it is not a new capture and does not establish reproducibility from a fresh forward pass. The §3 `c8b30c4` values remain preserved as the superseded outcome, and the §3A values come only from the corrected real cache-only output.

