# Sprint 16 Attribution Protocol v1 (prospective freeze)

**Status:** FROZEN pre-outcome. No comparative outcome was inspected to produce
this document. Any change requires `sprint16-attribution-protocol-v2.md` plus
fresh review; outcomes observed under v1 MUST NOT select v2 content.
**Scope:** Tasks 3–15 of `docs/sprint-plans/sprint-16.md`. Task 2 evidence:
`artifacts/sprint-16/task-2.md`.

## 1. Authority and amendment rule

1. This protocol freezes candidate components (§3), one-boundary interventions
   (§§6–7), budget (§8), eligible slices (§5), the common metric contract (§4),
   stopping rules (§9), and verdict rules with precedence (§10) BEFORE any new
   comparative outcome is inspected.
2. Tasks 3–15 execute exactly what is written here. Anything not listed here is
   forbidden unless a v2 protocol freezes it first.
3. Task 3 owns implementation of the diagnostic reader; it MUST satisfy the
   contract in §4 but has no freedom to change metrics, floors, or decision
   thresholds.
4. All Sprint 16 non-negotiables apply (see §11); conflicts resolve toward the
   more restrictive reading.

## 2. Accepted inputs (resolved by Task 1, re-verify before any run)

- Benchmark: Sprint 15 candidate 7, accepted `MEASURABLE`
  (`experiments/sprint15-benchmark-protocol-v7.md`, sha `a1fc09db…`).
- Roles (`artifacts/sprint-15/task-20.md` §2): Design H-DESIGN-37..40
  (1600–1603, structural reference/diagnostics only — NEVER evaluation);
  Fit H-FIT-28..30 (1604–1606, healthy rows only); Calibration H-CAL-10 (1607,
  frozen 95th-percentile rule, threshold `138.03` — selection only, no
  evaluation, no tuning); Confirmation H-CONF-34..37 (1608–1611, the single
  evaluation set); disposable proof H-PROOF-13/14 (evidence only).
- Sealed H-SEAL-37..40 (1612–1615): FORBIDDEN to open/score/probe. Seal
  manifests verify intact before any execution (§11.1).
- Checkpoints: accepted Sprint 12 control (`8bdb845b…`) and hybrid
  (`76be843b…`); hash-re-verify on the execution host before loading. Reference
  banks are checkpoint-embedded and restored as-is (no refit flag exists).
- Handcrafted oracle: `src/representation/handcrafted.py` (+ frozen
  dev-train-only `Standardizer`); accepted ceiling 0.663 file AUROC /
  0.948 paired ranking, FPR 0.669, diagnostic-only.
- Code bytes: config `55fed2ef…`, health `e5854641…`, chronicle `b5992d6a…`,
  balanced `d62de342…`, cli `3c264a33…`, probe15 `a08b3d5f…`,
  preflight15 `19f0300e…` (re-verify; mismatch voids the run per handoff §5).
- Runtime: repo root, locked `uv` env, python 3.13.3, torch 2.14.0+cpu, AMD64.

## 3. Candidate components and pipeline boundaries

Production pipeline under attribution (`src/representation/`, `V1Config`
defaults: patch 32 / stride 16, `d_model` 128, 4 layers × 4 heads, conditional
norm on, mask ratio 0.40 with random/info/block composition, contrastive
λmax 0.1, kNN k=5):

| ID | Component | Boundary (code) | Loss hypothesis |
|---|---|---|---|
| C1 | Conditional normalization | `layers/normalization.py::ConditionalBatchNorm` in/out | removes amplitude/gain/spectral/cross-channel info |
| C2 | Patchification | `collate_variable_files` + patchifier (32/16) | dilutes short/local/boundary anomalies |
| C3 | Local patch encoder | `layers/patch_encoder.py::LocalPatchEncoder` in/out | drops transient/phase/coupling/degradation signal |
| C4 | Context encoder + objective use | `layers/sequence_encoder.py` in/out (weights frozen) | mixing dilutes localized evidence |
| C5 | Objective / masking pressure | `masking.py`, `criterion.py`, `trainer.py` (training-time only) | interpolation/smoothness/nuisance shortcuts |
| C6 | File embedding / pooling | valid-patch → file embedding reduction | healthy-patch dominance erases local evidence |
| C7 | Conditional geometry | `inference.py::NormalReferenceBank`, `embedding_extraction.py` | wrong centroids/covariance/hierarchy/bank/scale |
| C8 | Scorers | `S_pred` (context mismatch) vs `S_pop` (population), kept SEPARATE | query-conditioning/energy terms unreadable |
| C9 | Patch-to-file aggregation | patch scores → file/event score reduction | tail behavior lost in reduction |

## 4. Common diagnostic contract (Task 3 MUST implement exactly this)

1. **Primary stagewise metric:** per-history P+W macro tie-aware event AUROC
   with history-block bootstrap B=2000, seed 20260202; report point estimate +
   95% LCB (2.5th percentile). Support: identical file/event set across the
   stages of one comparison (intersection of valid support); valid-patch masks
   preserved; history unit never pooled (per-history reporting; cross-history
   macros unweighted means only).
2. **Frozen operating metrics** (from handoff §4, unchanged): directional
   count (> 0.55); per-event max aggregation for REPORTING; P recall ≥ 0.50
   with median lead ≥ 1.0 d / W recall ≥ 0.25 with median lead ≥ 0.5 d at the
   frozen threshold `138.03`; FAR ≤ 0.05 per robot-day with ≤2-day grouping +
   reset-split; abrupt companion reported separately, never decisive.
3. **Reader properties:** same support, same standardization provenance
   (frozen Fit-healthy transform, `apply` only on eval rows), same tie
   handling, same aggregation for every stage except when the tested component
   IS aggregation (C9). `S_pred`-family and `S_pop`-family scores are never
   fused before the verdict stage.
4. **Candidate building blocks** (reusable, replaceable by Task 3 provided §4.1
   holds): `src/synth/probe15.py` (`feature_names`, `extract_features`,
   `standardize_fit`, `fit_centroid`, `score_files`, `history_block_lcb`),
   `src/representation/handcrafted.py` features.
5. Labels/masks/failure categories/simulator state: post-hoc diagnostics only,
   never representation inputs.

## 5. Eligible history and mechanism slices

- **Evaluation:** H-CONF-34..37 only, per-history. A claim needs ≥3 of 4
  histories (§10).
- **Fitting/standardization/banks:** Fit-healthy rows only
  (H-FIT-28..30 verified-healthy files).
- **Threshold:** frozen `138.03` from H-CAL-10; no re-thresholding, no tuning.
- **Structural reference:** Design roots for support/diagnostic sanity only.
- **Mechanism slices:** P1/P2, W1/W2, A1/A2, nuisance-only controls,
  unaffected background. P+W macro is primary; A1/A2 companion-only, never
  decisive alone. P and W cohort replication is individually required (§10).
- **Tiny gradient work (C5 only):** Fit-healthy + Design-structural rows only;
  never Confirmation, never Calibration, never Sealed.

## 6. Stagewise localization measurements (Tasks 5–12, no variants)

Each measurement compares adjacent stages with the §4 reader on identical
support and reports the upstream−downstream gap
`G = R_pre − R_post` (paired-ranking points):

- M1 (C1): pre-normalization vs post-normalization observable probes
  (level/gain, slope/drift, spectral bands, coupling, phase/timing, transients,
  degradation, nuisance controls).
- M2 (C2): pre-patchify vs post-patchify observables at production (32/16).
- M3 (C3): post-patchify observables vs frozen local latents, same patches.
- M4 (C4): frozen local vs frozen contextual latents at matched positions.
- M5 (C5): gradient/objective diagnostics on frozen checkpoints + ≤2 tiny
  dev-only runs per §8 (mask-ratio OR contrastive-weight factor, one factor).
- M6 (C6): valid-patch distribution vs mean/file embedding; duration/sparsity
  slices.
- M7 (C7): oracle features AND learned features through production conditional
  geometry vs the same regularized healthy-only diagnostic reader.
- M8 (C8/C9): production `S_pred` / `S_pop` (separately) vs predeclared simple
  diagnostic scorers/aggregations on fixed representations+geometry.

**Localization evidence** requires `G ≥ 0.10` with upstream LCB > 0.55
(upstream itself measurable). A gap is SUSPECT-class evidence, never causal.

## 7. Substitution matrix (Task 13 — exactly these 8 cells, no additions)

| Cell | Input | Reader/geometry | Question it answers |
|---|---|---|---|
| X1 | Handcrafted observables | Simple regularized reader | Observable ceiling? |
| X2 | Handcrafted observables | Production conditional geometry | Does geometry erase oracle signal? |
| X3 | Frozen local latents | Same simple reader | Local encoder retention? |
| X4 | Frozen contextual latents | Same simple reader | Contextual mixing effect? |
| X5 | Frozen learned latents | Production geometry | Current pipeline delivery? |
| X6 | Frozen learned latents | Predeclared alternative simple scorer | Signal present but unreadable? |
| X7 | Oracle/controlled features | Production pooling + aggregation | Does file reduction erase local signal? |
| X8 | Frozen encoder latents | Query-ablated (target-masked pre-mixing) scorer | Does self-conditioning suppress sensitivity? |

## 8. Bounded interventions (Task 14 — predeclared variants only)

One principal mechanism per intervention. Frozen values; no tuning, no search:

- **C1** (≤3): N-a global healthy z-score (frozen Fit mean/var, per-channel);
  N-b per-(robot,program) healthy z-score frozen from Fit (global fallback
  under 32 samples, mirroring `min_bucket_samples`); N-c identity (no norm).
- **C2** (≤3, diagnostic reader on patchified observables — frozen encoder
  weights are incompatible across patch sizes, so no encoder reuse):
  P-a (16/16), P-b (32/32), P-c (64/32). Production (32/16) is the reference.
- **C3**: NO bypass exists without redesign → capped at SUSPECT (see §10 P3).
- **C4** (≤2, frozen): B-a skip sequence encoder (local latents direct to
  pooling/geometry); B-b local + fixed uniform context average.
- **C5** (≤2 tiny retrains, §5 data only, ≤300 steps each): mask ratio
  0.40→0.15 (contrastive frozen); contrastive λmax 0.1→0.0 (masking frozen).
- **C6** (≤3, frozen latents): top-8 tail mean, median, max (k=8 frozen).
- **C7** (≤2): G-a diagnostic regularized healthy-only reader on learned
  features; G-b global unconditional healthy centroid, same distance form.
- **C8** (≤2, weight-free, `S_pred`/`S_pop` never fused): S-a query-ablated
  context score (target masked pre-mixing, frozen latents); S-b patch-MSE
  energy file score (mean, no tail).
- **C9** (≤3, diagnostic-only; reporting stays per-event max per §4.2):
  top-4 mean, event-window median, 90th percentile (k=4, q=0.9 frozen).

**Budget (hard):** ≤20 executed intervention configurations sprint-wide;
frozen-latent evaluations ≤30 min wall per history on CPU; gradient work ≤2
runs / ≤300 steps / ≤8 CPU-hours total. No architecture search, no
hyperparameter tuning, no large retraining. Exhaustion → §9.

## 9. Stopping rules

1. All predeclared variants of a component execute once each — no
   outcome-driven early stopping within a component (avoids selection bias).
2. A mechanism category whose Task 4 oracle fails the §10 G1 floor is labeled
   data/physics limitation and excluded from attribution (never charged to the
   learned pipeline).
3. If the primary metric is computable on <3 confirmation histories (support
   failure), affected components go UNRESOLVED, not SUSPECT-or-better.
4. Budget exhaustion stops new configurations; open components resolve to the
   best-supported verdict the existing evidence allows (SUSPECT or UNRESOLVED).
5. Never open Sealed histories to break ties or rescue an ambiguous call.

## 10. Verdict rules and decision precedence

**G1 oracle gate:** a category is attributable only if the Task 4 oracle
reaches paired metric ≥ 0.65 with LCB > 0.55 on the confirmation slice.
Otherwise: data/physics limitation.

**Per-component verdicts:**

- `HEALTHY`: upstream measurable, all gaps < 0.10, and no intervention moves
  the primary metric by ≥0.05 on ≥2 histories.
- `SUSPECT`: localization gap ≥ 0.10 with upstream measurable, but restoration
  / replication / nuisance conditions below are not all met.
- `BOTTLENECK`: ALL of (i) gap ≥ 0.10 upstream-measurable; (ii) restoration
  `R_int − R_post ≥ 0.5 × G` with post-intervention LCB > 0.55; (iii) nuisance
  veto passed — FAR ≤ 0.05/robot-day on all 4 histories AND ΔFAR ≤ +0.02 vs the
  production pipeline on identical support; (iv) replication on ≥3/4
  confirmation histories AND in P and W cohorts separately; (v) holds under
  BOTH frozen encoders (control + hybrid), except C1/C2 observable-level
  claims where no encoder is involved (stated per claim).

**Overall state:**

- `IDENTIFIED`: exactly one `BOTTLENECK`.
- `MULTIPLE BOTTLENECKS`: ≥2 `BOTTLENECK`s at independent stages (different
  pipeline stages with independent restoration — e.g. a C8 tail win and a C9
  tail win on the same signal count once unless each restores alone with the
  other held at production).
- `UNRESOLVED`: otherwise. Insufficient evidence defaults here, never to a
  guessed bottleneck.

**Precedence (numbered, conflicts resolve downward):**

- P0. Anything requiring Sealed data is rejected outright.
- P1. G1 eligibility gates every component claim on that category.
- P2. Nuisance/background regression vetoes `BOTTLENECK` (caps at `SUSPECT`)
  no matter how large the signal restoration.
- P3. Capability caps: C3 cannot exceed `SUSPECT` (no bypass exists without
  redesign); C5 `BOTTLENECK` additionally requires a replicated tiny-retrain
  win, else caps at `SUSPECT`.
- P4. Single-seed, pooled-only, or favorable-metric-only wins cannot establish
  `BOTTLENECK` (primary metric governs; secondaries are consistency only).
- P5. When two rules disagree on the overall state, the less favorable state
  wins (`UNRESOLVED` > `IDENTIFIED` ordering inverted: prefer `UNRESOLVED`).
- P6. Budget exhaustion resolves opens to `SUSPECT`/`UNRESOLVED` as evidence
  allows; extension needs a v2 protocol.

## 11. Non-negotiable operational restatements

1. Seals verify intact (round-trip) before any execution; seal manifests only,
   never waveform/score content. Fit/Calibration/Confirmation/Design role
   permissions per §2 (handoff §§2–4) govern every task.
2. Handcrafted/learned probes are diagnostic readers, never production
   performance; no probe metric is presented as detector success.
3. No calibrated-risk, probability-quality, production-threshold, or
   operational early-warning claim anywhere in Tasks 3–15.
4. No Sprint 12 BLOCKED-FINAL task reopened; no Sprint 14/15 byte modified;
   Sprint 15 sealed histories unopened for selection AND attribution.
5. One principal mechanism per intervention; no simultaneous
   normalization/encoder/objective/geometry/scorer redesign.

## 12. Traceability (Tasks 3–15 → sections)

Task 3 → §4 (implement reader). Task 4 → §10 G1 (oracle ceiling + floors).
Tasks 5–12 → §6 M1–M8. Task 13 → §7 X1–X8. Task 14 → §8 variants within §8
budget. Task 15 → §10 replication + §4 support rules. All tasks → §§2, 5, 9,
10 precedence, 11.

## 13. Reviewer static checklist (no outcomes needed)

- [ ] Every cited file/hash/role/seed in §2 exists byte-identically (Task 1 §6).
- [ ] §§6–8 contain no tunable values (all numbers frozen above).
- [ ] Variant counts per component ≤ caps; total ≤ 20.
- [ ] §10 precedence is total: every combination of (gap, restoration,
      nuisance, replication) maps to exactly one verdict.
- [ ] No task in §12 can consume Sealed, Calibration-eval, or Fit-beyond-
      healthy rows.
