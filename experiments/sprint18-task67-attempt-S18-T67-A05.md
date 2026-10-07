# Sprint 18 Task 67 — S18-T67-A05 v3 fixed-phase proof summary

**Task / attempt / agent:** `S18-T67` / `S18-T67-A05` / `S18Task67A05`
**Outcome:** COMPLETE worker proof (fresh review + Bronze checkpoint still required; no commit/ref change here).
**Role/tier:** explicit fresh `gold-task` (Gold), Main-selected continuous-Gold continuation of paused `S18-T67-A04`.
Received selector `opencode-go/muse-spark-1.3-contributor:xhigh` (blocking `false`; provider-applied effort/quota unknown).
**Base:** `HEAD == refs/heads/master == 820d724893341cee60f2a911b65881be2761b3e1`.
**Normative method:** `experiments/sprint18-iterative-data-contract-v3.md` SHA-256 `544de84bc201a07140559058d100c991cd148b698f93058dbd37d7fe4dc3c929` (frozen; untouched).
**Machine companion:** `experiments/sprint18-task67-attempt-S18-T67-A05.json` (same record, canonical JSON).

## Scope and exclusions

Implemented nothing new in A05: all nine tracked source/test edits and the A04 proof runner are inherited from paused A04 byte-identical
(verified before any A05 work; hashes in the JSON companion). A05's only work is the remaining explicit completion —
focused pytest, CHANGELOG + this durable summary, binding-collision/disjointness record, worker artifact append, and handoff.
`schema.py` remains prose-only (fields/types/validation/serialization/runtime frozen).

Proof hygiene: isolated attempt namespace `data/generated/.sprint18-iterative-contract-v3/task67/S18-T67-A04/`
(ignored raw roots; durable hashes below are the reviewable evidence). A04's complete generation/reload/fixture records were
preserved byte-identical and never regenerated. A05 executed no generator, preflight, diagnostic, training, bank, scorer,
Sealed, or candidate/retired-seed contact of any kind. The A03 v2 roots (including C02) stay immutable diagnostic evidence.
Fixture C is the frozen inherited in-memory observable input; no fresh role-root generation. Actual fresh healthy Fit/Cal
probe and DGP qualification stay mandatory later Task 69 work.

## Method (v3 §3, frozen)

- `phase[(seed, robot_id, cohort_id)] = SHA256("sprint18-failure-subtype-v3|seed|robot_id|cohort_id").digest()[0]`, once per group; ordinal excluded from the preimage.
- `fail_ordinal[(robot_id, cohort_id)] += 1` only when a P/W failure actually fires (0 → 1 on first firing).
- `subtype = labels[(fail_ordinal + phase) % len(labels)]`; zero shared-RNG draws; public `FailureEvent` label canonical; episodes retain legacy dual variant labels.
- Canonical frozen phases: `(94100, robot-02, P)` → 40; `(94101, robot-02, P)` → 234 (full 8-key table in JSON).

## Fixture A — actual public materialization and dual reloads (inherited complete, verified in A05)

- Config: `sprint18_iterative_v3_history_config(seed)`; profile `sprint18-iterative-v3`; protocol `sprint15-benchmark-protocol-v7`; 450-day span/cutoff; `n_units=2880`; no `sprint15` binding.
- Seed 94100 (`TASK67-FIXTURE-REAL-94100`, config hash `c617a40430e5`): total 6472 (normal 5089 / abnormal 1383), dev-train 1188, dev-val 298, static 3401, temporal 4039; failures 241; controls 70; selected P 12/12, W 12/12, A 8/8; per-group max-prefix imbalance 1; worst final group imbalance 1; dual-label failures 59; support 1659 robot-days, 9 positive / 9 negative robots.
- Seed 94101 (`TASK67-FIXTURE-REAL-94101`, config hash `34b32b360dd4`): total 6451 (normal 5011 / abnormal 1440), dev-train 1292, dev-val 324, static 3432, temporal 3978; failures 229; controls 67; selected P 12/12, W 12/12, A 8/8; per-group max-prefix imbalance 1; worst final group imbalance 1; dual-label failures 54; support 1665 robot-days, 9 positive / 9 negative robots.
- Both histories: two fresh-process public reloads each with distinct PIDs, `reload-comparison.identical=true` over all 22 compared fields; Task 5 structural + Design + hard-floor checks all true; timing/cohort bit-identical v3-vs-v1 excluding subtype labels; flag-off byte-identity preserved; precursor gains read from the failure record.

## Fixture B — allocator boundaries plus fixed-phase regression (inherited complete, verified in A05)

History seed 966660; `all_named_acceptance_passed=true`; `failure_ordinal_parity.pass=true`; `consumer_regressions.pass=true`
(phase determinism over 4 keys both parities; 130/130 prefix sweep n ∈ [0,64]; increment-once determinism; strict alternation;
per-cohort skew P 6 / W 0 ≤ 9; flag-off episode inheritance).

- Feasible: joint-good-49, controls-48, controls-49, mix-exact-15, mix-below-15 (allocation feasible; only its mix predicate fails), lead-76 (allocation feasible; only `P_lead_support_ge_80_percent` fails at 19/25 = 76%), lead-80 boundary (20/25 = 80% pass).
- Infeasible as frozen: controls-47 (`control-shortfall`, 47 < 48), joint-csp-conflict (solver status 2, proven infeasible), w1-11 (eligible W1 = 11 < 12), w2-11 (eligible W2 = 11 < 12).

## Fixture C — fixed-DGP observable path (inherited complete, verified in A05)

- Descriptor `a9f999fda657e4a9c4ff0af12975bfb21f958dee9d9c1802e163e43d9a239e88` match; 59 features; Fit std exactly `STD_FLOOR` (1e-6); centroid max-abs `4.440892098500626e-10`; 19 zero Calibration scores + outlier `6857808409.877925`; threshold `342890420.49390113`; Development and Confirmation macro AUC P/W/PW 1.0, history-block LCB 1.0, 4/4 directional histories each.
- Harness correction applied in A04 per R03 Finding 3: removed the unfounded all-non-P/W zero-score assertion; every frozen probe gate numerically unchanged. Provenance limits: in-memory synthetic path proof only, not real Fit/Calibration/Development statistics.

## Binding-collision / disjointness record

- Frozen disposable seeds 94100/94101 are arithmetically disjoint from prior fixture seeds (9000, 9001, 93058, 93061, 966660), the retired namespace 32000–32031, and the candidate-3 namespace 32032–32047.
- `experiments/sprint18-iterative-binding-v1.json` binds exactly seeds 32016–32031 (16/16 retired candidate-2 range); intersection with {94100, 94101} is empty.
- Contract-text whole-token matches for 94100/94101 occur only in the frozen v2/v3 contract documents (normative disposable-coordinate declarations), never as role bindings. Candidate 3 (S18-ITER-0003, 32032–32047) stays UNBOUND/UNREAD/UNCONTACTED; candidates 1–2 (32000–32031) retired, never replayed. No Task 68 binding edit was made here.

## Verification exercised in A05

- Focused pytest (only affected suites, no full suite/build/lint): `uv run --locked pytest -q tests/synth/test_chronicle.py tests/synth/test_cli.py tests/synth/test_preflight15.py` → **27 passed in 70.51s** (background job bg_5; wall 73.74s).
- Read-only reconciliation of every inherited proof hash above (reload JSONs, comparisons, fixture-B/C results), config hashes per seed, canonical 8-key phases, and ledger observables. No regeneration, no ground-truth failure rerun.
- Owned path scope (task-owned only): 9 modified tracked files (`src/synth/{__init__,chronicle,cli,health,preflight15,schema}.py`, `tests/synth/{test_chronicle,test_cli,test_preflight15}.py`) + new untracked `task67_a04_proof.py` + `CHANGELOG.md` implemented-v3 entry (this attempt) + this durable JSON/MD pair + worker artifact append. Historical `task67_a03_proof.py` preserved as failure evidence. Plans, mixed memory, user experiments, `sprint22_pilot_runner.py`, and unrelated untracked artifacts excluded.

## Limits and unresolved findings

- Passing disposable fixtures prove only the corrected construction path on disposable coordinates; they confer no design credit and predict no candidate outcome. Actual fresh healthy Fit/Cal fit and DGP qualification on the bound roster remain Task 69 acceptance work.
- Fresh independent evidence review and a separate Bronze checkpoint are still required; this worker claims no review verdict and no commit.
