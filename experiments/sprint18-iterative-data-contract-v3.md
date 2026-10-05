# Sprint 18 Iterative Data Recovery Contract v3 — fixed-phase failure-ordinal P/W subtype alternation

**Contract ID:** `sprint18-iterative-data-contract-v3`
**Status:** Task 66 prospective contract correction; **not implementation-reviewed, not role-bound, and not released for candidate data contact.**
**Supersedes:** nothing. `experiments/sprint18-iterative-data-contract-v1.md` (SHA-256 `056105f87b1097c45244287636bcef8c2fc790f3fd894775b9105fbdbfb59e9d`) and `experiments/sprint18-iterative-data-contract-v2.md` (SHA-256 `29164cc3e21a884469f77a1e9bc155399bf86ecc2633777801997890488b5264`) are preserved byte-identical as historical records. This v3 corrects exactly one normative defect in v2 §3 (variable-phase hash preimage) with its diagnosis, executable proof, Task 67 delta scope, and versioned identity; every other v1/v2 section is inherited unchanged by reference.
**Scope:** One fresh, common-method, success-only Sprint 18 data-recovery branch (4 Design, 3 Fit, 1 Calibration, 4 Development, 4 Confirmation whole-history roles). Candidate role contact stays behind Task 68 review and Main release. The unchanged Sprint 15 v7 fixed observable DGP sanity probe remains the sole outcome-scoring exception as specified in v1 §2. No eight-arm representation training, learned scorer, bank, Sealed access, or model study is authorized.
**Base:** commit `b2a8315864d999893c86ad1cb10ba1d4c9db8f89` on `refs/heads/master` (worktree `HEAD` and `refs/heads/master` agree; verified before writing).

## 1. Authority and immutable inputs

This version is prospective and subordinate to the current Sprint 18 plan and the user authorization recorded there. It does not replace or override any full-training protocol or inherited benchmark gate. The plan Tasks 66–71 own the data workflow and status. Task 66 does not edit the Main-owned plan, bind the next final roster, contact a final-role seed, or authorize runtime work. No source, schema, test, proof-script, binding, or generated-data file is changed by this documentary correction.

The v1 §§1–2, 4–5, 7–8 input tables, gates, and boundaries are inherited unchanged (v2 training package traceability, v7 protocol, C2/C3 records, v7 probe, exit gates, 450-day exposure leaves, quotas, caps, spacing, controls, mix, lead, probe). The following additional read-only identities bound this correction:

| Input | SHA-256 | Use here |
|---|---|---|
| `experiments/sprint18-iterative-data-contract-v1.md` | `056105f87b1097c45244287636bcef8c2fc790f3fd894775b9105fbdbfb59e9d` | Preserved historical v1; all unamended sections inherit from it. |
| `experiments/sprint18-iterative-data-contract-v2.md` | `29164cc3e21a884469f77a1e9bc155399bf86ecc2633777801997890488b5264` | Preserved defective v2; §3 formula under correction, §6 acceptance retained verbatim. |
| `artifacts/sprint-18/reviews/S18-T67-S18-T67-R03.md` | `82ca1045d38db5be2491bf4692247d03d9b6104ead8cd8f0074849f656e6ffb2` | Independent FAIL review (`advance=false`); sole gate authority for the defect below. |
| Stored counterexample `data/generated/.sprint18-iterative-contract-v2/task67/S18-T67-A03-C02/failure-ordinal-acceptance-counterexample.json` | `0ee95626928250820b849d3da7fd3675e7cd01fed39be9861b19453731c320b9` | Real observed `robot-02/P` P1=3/P2=5 imbalance-2 record; read, never rerun. |
| `src/synth/health.py` worktree bytes at freeze time | raw SHA-256 `654bc9f4a1aa616baf89ae46cf83949ce693659712e372f3e26fc74410b40df4` | Read-only source ground for counter/ordinal/increment semantics; untouched by Task 66. |
| `src/synth/health.py` base blob `a33e99ea60328a9e996ad16b1052ee116d584bf5` | (Git blob OID; base-commit bytes) | Distinguishes Task 67 A03 dirty implementation from the frozen base; Task 66 changes neither. |

Task 66 in this assignment performs no generator, preflight, qualification, probe, model, SSH, or raw-root execution, and contacts no candidate seed (including no candidate-3 seed `32032–32047`), no retired seed (`32000–32031`), and no disposable seed beyond read-only SHA recomputation of the frozen formulas. The §2 numbers below are read-only reconciliations of the stored authorized evidence above and the current source text; the §3 proof is a throwaway in-memory SHA/arithmetic check only (no test file added, no throwaway file retained).

**Tooling note (observed availability).** `xd://lsp` is unavailable in this runtime (`No such tool: xd://lsp. Mounted devices: ast_edit, debug.`). Per the prior A03-authorized fallback, this correction uses targeted source reads (`read` ranges) and literal-pattern search (`grep`) of `src/synth/health.py` instead of LSP navigation. All counter/ordinal/domain claims below cite exact line-anchored source text, not inference.

## 2. Defect diagnosis: v2's ordinal-dependent hash contradicts its own acceptance bound

### 2.1 What the stored evidence says

Fixture A seed `94100` (`TASK67-FIXTURE-REAL-94100`), generated through the real public CLI under the faithfully implemented v2 formula, reached the frozen §6 acceptance assertion on its first fresh-process public reload with a real counterexample: `robot-02/P` emitted P1=3, P2=5, absolute imbalance 2 against the frozen required maximum 1 (stored record verbatim, SHA above). The reload's manifest-equality check passed first — regenerated `FailureEvent` records exactly matched the public manifest `failure_events` — so the record is a genuine method counterexample, not a loader or comparison artifact. R03 Finding 1 classifies this as a normative conflict inside the v2 contract itself, with HIGH confidence; R03 Finding 3 separately confirms the public probe routines passed all macro/per-history observable gates (macro AUC 1.0, LCB 1.0, 4/4 directional) and isolates the Fixture C stop to a wrong ad-hoc harness assertion (addressed in §6, not here).

### 2.2 Source mechanism responsible (read-only, line-anchored)

Current `src/synth/health.py` (worktree bytes above; base blob differs only by the A03 v2 implementation, which this contract reads but does not modify):

- `_draw_pw_subtype` (episode-opening label, retained for diagnostics) increments `deg_sub_ordinal[(robot, cohort)]` and hashes the dedicated episode domain `sprint14-subtype` with the episode ordinal in the preimage.
- `_draw_pw_failure_subtype` (the v2 failure path) increments `fail_sub_ordinal[(robot, cohort)]` from 0 to 1 **before** hashing (`fail_sub_ordinal[key] = fail_sub_ordinal.get(key, 0) + 1`), then hashes the dedicated failure domain `sprint18-failure-subtype` with `str(fail_sub_ordinal[key])` **in the preimage**, and indexes `labels[(fail_ordinal + digest[0]) % len(labels)]`.
- The P/W failure call site invokes this helper exactly once per fired P/W failure (fired-path only), while the A branch uses a separate per-robot `abrupt_ordinal` with a **fixed** per-`(seed, robot)` offset from domain `sprint15-stratified-A` and **no ordinal in its hash preimage** — the proven alternation pattern this v3 adopts for P/W.
- The shared health RNG is never drawn inside either subtype helper (the A branch keeps a stream-preserving `rng.random()` no-op so failure timing/density stay invariant); the failed formula's defect is therefore purely the varying hash phase, not RNG disturbance.

### 2.3 Why the old offset varies: analytic recomputation without any generator rerun

Recomputing the frozen v2 formula's SHA inputs for the counterexample key `(seed 94100, robot-02, P)` over failure ordinals 1–8 gives first-digest-bytes `[158, 47, 75, 230, 81, 121, 58, 205]` — seven distinct offsets across eight firings, parities alternating irregularly. The resulting label sequence is `P2, P2, P1, P1, P1, P2, P2, P2`, i.e. P1=3/P2=5, imbalance 2, exactly reproducing the stored C02 counts:

| Prefix n | Old labels | Imbalance |
|---|---|---:|
| 1 | P2 | 1 |
| 2 | P2, P2 | **2 (bound already violated)** |
| 3 | P2, P2, P1 | 1 |
| 4 | P2, P2, P1, P1 | 0 |
| 5 | +P1 | 1 |
| 6 | +P2 | 0 |
| 7 | +P2 | 1 |
| 8 | +P2 | **2 (stored counterexample)** |

Because each firing re-hashes a new preimage, `digest[0]` parity is free to change at every step rather than anchoring a stable alternation phase; the parity sum `(ordinal + digest[0]) % 2` therefore behaves as ordinal-indexed pseudo-random trials, which no per-`(robot, cohort)` bound can hold. This refutes the v2 §3 guarantee sentence while leaving the v2 §6 acceptance bound (≤1) intact and unrelaxed.

### 2.4 What is not claimed

No iid/binomial/pass-rate/power/feasibility claim is inferred: the source text supports no stochastic assumption, the SHA phase is a fixed deterministic value per key (not a random draw), and none is needed. The diagnosis uses only stored counts, the frozen predicates, the exact hash formula, and the actual call-site code path. No independence or uniformity is assumed from hashing anywhere in this contract.

## 3. The correction: one deterministic group-specific phase; ordinal excluded from the hash preimage

**Frozen semantics.** Under the existing `stratified_subtype_emission=True` flag (same flag, extended meaning; flag-off behavior stays byte-identical to v6 and earlier), P/W subtype labels are assigned at **P/W failure-firing time** by strict alternation over a per-`(robot, cohort)` **failure** ordinal with one fixed group phase:

```text
phase[(seed, robot_id, cohort_id)] = SHA256(canonical(seed, robot_id, cohort_id)).digest()[0]  // once per group; ordinal NOT in preimage
fail_ordinal[(robot_id, cohort_id)] += 1   // only when a P/W failure actually fires; 0 -> 1 on first firing
subtype = labels[(fail_ordinal + phase) % len(labels)]
```

### 3.1 Canonical serialization (stable, typed, reviewable)

| Element | Frozen value |
|---|---|
| Hash domain | `sprint18-failure-subtype-v3` — distinct from the episode domain `sprint14-subtype`, the retired v2 failure domain `sprint18-failure-subtype`, and the A domain `sprint15-stratified-A`; also distinct from episode/A/shared-RNG streams. |
| Canonical preimage | `"sprint18-failure-subtype-v3" + "|" + str(seed) + "|" + robot_id + "|" + cohort_id`, encoded UTF-8, `|` separator (same join convention as the actual source helpers). |
| `seed` field | `HealthConfig.seed`, the integer history seed (e.g. `94100` rendered as `94100`, no padding, no hex). Source boundary: the health-stream seed, never the scheduler/signal seed and never a manifest hash. |
| `robot_id` / `cohort_id` | Exact runtime strings (`robot-01`…`robot-09`, `P`/`W`); no normalization, no lower-casing, no index remapping. |
| Phase extraction | `digest()[0]`, first byte, value 0–255; only its parity selects the alternation phase. No `digest[:8]` integer draw, no per-event re-hash, no per-event RNG draw. |
| Labels | The cohort's declared subtype tuple in source order (`P → (P1, P2)`, `W → (W1, W2)`). |
| Counter | Per-`(robot_id, cohort_id)` integer, initialized 0, incremented exactly once per fired P/W failure **before** indexing (first firing uses ordinal 1), matching the actual source increment timing. Never incremented on episode opening, hazard evaluation, non-firing steps, maintenance, or A events. |
| RNG discipline | The assignment performs zero shared-RNG draws (same invariance discipline as the A branch, including its stream-preserving no-op convention where applicable): failure timing, cohort assignment (`upcoming`), density, and all hazard draws are invariant by construction. No outcome-posthoc rebalancing, no relabeling after allocation or synthesis — labels are assigned once at event time, before allocation and waveform synthesis. |

### 3.2 Binding rules

- Episode-opening assignment is retained verbatim for diagnostic continuity (episode records keep their labels); the `FailureEvent` carries the failure-ordinal label. Precursor physics (`temporal._cohort_windows`) already reads the **failure record's** subtype, so the subtype→gain coupling (P1 1.0 / P2 0.55 / W1 0.3 / W2 0.16) is preserved with labels still assigned once at event time — never post-hoc relabeled.
- The A branch is unchanged (already fixed-phase failure-ordinal alternation). Flag-off behavior is unchanged (episode-subtype inheritance, byte-identical to v6 and earlier).
- Dual-label history is retained: episode history and `FailureEvent` physical/public subtype stay consistent through the existing failure-record read path. Schema fields, types, validation, serialization, and manifest format are unchanged; only a `FailureEvent` docstring clarification reflecting the corrected label origin is permitted in Task 67 (no field/type/behavior change).
- Eligibility remains subtype-blind and unchanged; the allocator, quotas, caps, spacing, controls, mix, lead, and all qualification predicates are unchanged. No scientific gate or process change, no blind horizon extension, no retired-history relabeling, no arbitrary guarantee.

### 3.3 Theorem (unbounded n) and finite-boundary executable proof

For fixed phase parity `φ ∈ {0, 1}` and labels of size 2, the sequence `s_k = (k + φ) mod 2`, `k = 1…n`, strictly alternates: consecutive terms always differ.

$$ |c_0(n) - c_1(n)| \le 1 \quad \text{for every prefix } n \ge 0, \text{ both } \varphi \in \{0,1\}. $$

Exact counts: for even `n`, each label appears `n/2` times; for odd `n`, the first label appears `ceil(n/2)` and the other `floor(n/2)`:

$$ \max(c_0, c_1) = \lceil n/2 \rceil, \qquad \min(c_0, c_1) = \lfloor n/2 \rfloor. $$

Executed proof (throwaway in-memory SHA/arithmetic check, no file retained): every prefix `n ∈ [0, 64]` was enumerated for both parities — all 130 prefixes satisfy imbalance ≤1, and the ceil/floor identities hold exactly on all 130. The declared finite boundary `[0, 64]` generously covers every plausible per-`(robot, cohort)` failure count (observed per-group counts are single digits); the theorem above extends the bound to unbounded `n` algebraically, so the boundary is a checked instance, not the guarantee itself.

Actual frozen-key phases under the canonical serialization (recomputed read-only; ordinal absent from every preimage):

| Key (seed, robot, cohort) | Canonical bytes | Phase | Parity | First 4 labels (P1/P2 order) | Prefixes 1–8 ≤1 |
|---|---|---:|---:|---|---|
| 94100, robot-02, P | `sprint18-failure-subtype-v3\|94100\|robot-02\|P` | 40 | even | P2, P1, P2, P1 | yes |
| 94100, robot-02, W | `sprint18-failure-subtype-v3\|94100\|robot-02\|W` | 237 | odd | W1, W2, W1, W2 | yes |
| 94100, robot-01, P | `sprint18-failure-subtype-v3\|94100\|robot-01\|P` | 187 | odd | P1, P2, P1, P2 | yes |
| 94100, robot-09, W | `sprint18-failure-subtype-v3\|94100\|robot-09\|W` | 93 | odd | W1, W2, W1, W2 | yes |
| 94101, robot-02, P | `sprint18-failure-subtype-v3\|94101\|robot-02\|P` | 234 | even | P2, P1, P2, P1 | yes |
| 94101, robot-02, W | `sprint18-failure-subtype-v3\|94101\|robot-02\|W` | 242 | even | W2, W1, W2, W1 | yes |
| 94101, robot-01, P | `sprint18-failure-subtype-v3\|94101\|robot-01\|P` | 10 | even | P2, P1, P2, P1 | yes |
| 94101, robot-09, W | `sprint18-failure-subtype-v3\|94101\|robot-09\|W` | 229 | odd | W1, W2, W1, W2 | yes |

Both parities occur across the frozen disposable keyspace (no first-label bias by construction); canonical bytes, hash phases, and the no-ordinal-in-preimage property were verified byte-for-byte in the same check. Counterfactual on the counterexample key: the corrected rule emits strict alternation `P2, P1, …` with every prefix ≤1, where the old rule emitted `P2, P2, P1, P1, P1, P2, P2, P2`. No permanent test was warranted for this documentary correction (per skill procedure, a test must protect an observable behavior against plausible regression; the executable check above is the documentary proof, and Task 67's frozen consumer regressions are the behavioral protection).

### 3.4 Honest aggregate limits (what per-group ≤1 does and does not imply)

Per-group ≤1 implies only a bounded aggregate imbalance: with 9 robots (`robot-01`…`robot-09`), each cohort's history-wide skew is bounded by 9 (one per group), and each history's total P/W skew by 18. At the observed W total 31 the adversarial-alignment floor is (31−9)/2 = 11; at the observed median W total (~46) it is 18 ≥ quota 12 — but adversarial alignment is possible, hash phases carry no independence/uniformity claim, and preflight remains the honest gate. Local balance does **not** imply global quota, supply, eligibility, mix, CSP feasibility, or any candidate PASS: a cohort failure total of 3, perfectly balanced (2/1), still fails quota 12 for lack of supply. Residual thin-total risk is retained exactly as in v2 §7. No iid, independence, uniformity, feasibility, or pass-rate claim is made from hashing.

**Considered and rejected (inherited from v2 §3, unchanged).** Blind span increase beyond 450 days; W rate / `upcoming_p` / cohort-share change; `upcoming` P/W stratification; quota/gate reduction (W1 12→11); seed-block switching; relabeling; outcome-based reassignment; within-candidate repair — all rejected as gate relaxation, seed shopping, physics change, or causality breaks.

**Unchanged physics (binding).** Per-time failure/noise rates, arrival cadence and jitter, program routes, fleet topology, maintenance/recommissioning, quarantine, reset/censoring rules, 7-day horizon, 14-day spacing, caps, 450-day span/cutoff, `n_units=2880`, and every gate/threshold stay exactly as in v1 §4. The complete resolved-config diff of the new factory versus C2 contains only the v1 three exposure leaves plus the corrected failure-ordinal mechanism flag semantics; any other diff fails review.

## 4. Versioned identity, affected boundaries, and invariants for Task 67

- **New profile:** `sprint18-iterative-v3` (source/CLI profile string; explicit clean cutover — no alias accepting `sprint18-iterative-v2` for the corrected method). **New factory:** `sprint18_iterative_v3_history_config(seed)`, additively composed from `sprint18_iterative_v1_history_config(seed)` with only the §3 corrected failure-ordinal semantics; the chronicle manifest `protocol` remains `sprint15-benchmark-protocol-v7`; the iterative contract ID recorded in binding/attempt sidecars becomes `sprint18-iterative-data-contract-v3`. The v2 profile/factory remain in source history untouched; the corrected proof uses only v3.
- **Exact affected source/config/API boundaries for the next Task 67 (nothing else may change):** `src/synth/health.py` (fixed-phase P/W failure assignment only — domain literal, preimage without ordinal, phase extraction, unchanged increment timing; hazard/upcoming/maintenance/severity/A-branch/flag-off paths untouched); `src/synth/config.py` (only if a mechanism flag/validator literal is required — prefer reusing the existing `stratified_subtype_emission` flag with extended documented semantics); `src/synth/chronicle.py` (new factory entry + `S18_ITERATIVE_V3_*` profile/contract constants only); `src/synth/cli.py` + `src/synth/__init__.py` + `src/synth/preflight15.py` (new profile branch/exports only); `src/synth/schema.py` (one `FailureEvent` docstring clarification only — fields/types/validation/serialization frozen); affected-profile consumer regression tests only. No parallel qualification pipeline, no mocks, no model restore, no representation/training/checkpoint-bank code.
- **Frozen by continuity:** `balanced.py` (quotas/caps/spacing/solver), `events.py`, `temporal.py`, `splits.py`, `scheduled.py`, `scheduler.py`, `dataset.py`, `probe15.py`, `experiments/sprint18_task5_measurability.py` surfaces, and all v1 §5 validator/bound semantics. Exact positive quotas P1/P2/W1/W2 = 12 each, A1/A2 = 8 each; construction controls ≥ 48 with 49+ retained; hard 25 / Design 32 unchanged; full-ELIGIBLE 15–60 mix, lead/caps, 14-day spacing, robot/program support, and causal probe surfaces unchanged. All accepted numeric gates from v2 remain numerically unchanged.
- **Per-history permissions:** v1 §2 role table inherited unchanged (Design diagnosis only on authorized roles; Fit/Calibration/Development/Confirmation permissions and Sealed prohibition intact). No prior history is promoted into a fresh role.
- **Role-safe common method:** the correction applies identically to every future history through the shared factory; no per-role/per-seed special-casing and no second pipeline or model pilot.

## 5. Prospective Task 68/69 consequences (no binding here)

The next candidate (N=3, seeds `32032–32047` per the frozen `32000+16×(N−1)` formula) is bound by a fresh Task 68 worker only after this correction's Task 67 implementation proof and independent reviews pass. No candidate-3 seed is inspected, contacted, prechecked, or generated by Task 66 or Task 67. Retired seeds `32000–32031` (including 32018) are never replayed under any method, even disposably: the mechanism regression below uses fresh disposable seeds only.

## 6. Frozen next-Task 67 disposable proof contract (delta from v2 §6)

All v1/v2 proof hygiene inherits: isolated attempt roots below `data/generated/.sprint18-iterative-contract-v3/task67/<attempt-id>/` (a **new namespace** — the A03 v2 roots including C02 are immutable diagnostic evidence under the bad formula, never overwritten, never reused as a corrected-method PASS or reusable prefix); never role roots, never promoted, append-only correction evidence; genuine calendar/signal waveforms via the public CLI; shards → dual fresh-process public reloads → existing qualification functions; no mocks, hand allocations, or compile-only proofs. The accepted v1 proof (6,508-sample Fixture A real materialization, allocator boundary suite, probe path) is the established base, not repeated.

### Fixture A — corrected public generation / reload / qualification (retained frozen disposable seeds)

- Seeds `94100` (`S18I-FIX-REAL-94100`, role `TASK67-FIXTURE-REAL-94100`) and `94101` (`S18I-FIX-REAL-94101`, role `TASK67-FIXTURE-REAL-94101`). Hygiene: the 94100-block has no exact seed binding anywhere in the repo (substring matches are floating-point metric tails such as `0.994100…`, not seed references); Task 67's binding collision check must verify disjointness against every existing history, research seed, model seed, prior candidate, and prior fixture seed (9000, 9001, 93058, 93061, 966660) plus the two frozen keys themselves before generation.
- Config `sprint18_iterative_v3_history_config(seed)`; invocation mirrors v1 §6 Fixture A with `--profile sprint18-iterative-v3`; manifest protocol `sprint15-benchmark-protocol-v7`, 450-day span/cutoff, `n_units=2880`, no `sprint15` block.
- Expected observables: chronological schedule/waveforms within the 450-day calendar; six finite channels; `FileSample.validate` green; shard/manifest/hash/order checks green on **both** public reloads of **both** histories; model-row allowlist with no leakage; exact allocator witness with all 64 quota members independently validated (caps/spacing/quotas); Task 5 structural + role checks green; **per-`(robot, cohort)` failure-subtype counts differ by ≤1 on each disposable history** (the §3 bound, asserted from the real ledger — now provable by construction); eligibility/mix/lead/support predicates unchanged.
- Timing/density/shared-RNG baseline: on each disposable seed, the failure-time and cohort sequences under the v3 factory equal those under the v1 factory bit-for-bit (same-seed ledger comparison **excluding subtype labels**); shared-RNG consumption is bit-identical (the assignment draws nothing); flag-off byte-identity is preserved for legacy profiles. What must match and why:

| Observable | Must match v1 baseline? | Why |
|---|---|---|
| Failure times, cohort sequence, per-group failure counts, density | Yes, bit-for-bit | The correction touches no hazard/RNG/cadence path; any drift here is a physics break, not the method. |
| Per-`(robot, cohort)` subtype balance ≤1 | Yes (newly satisfiable) | The §3 theorem; the observable the old method failed. |
| Waveform sample bytes where subtype-dependent gains apply | **No** — expected to differ exactly where the corrected label changes the precursor gain | Subtype→gain coupling is preserved by design; label changes legitimately change signal scale. Byte identity here would indicate relabeling-after-synthesis, which is forbidden. |
| Manifest schema/serialization, shard layout, counts of samples/controls | Yes | Unchanged code paths; method change alters labels, not containers. |

### Fixture B — allocator boundaries plus fixed-phase regression

- The v1 Fixture B suite (history seed 966660 in-memory builder) is retained in full for the unaffected allocator/qualification interactions, including the symmetric `w1-11` mirror (exactly 11 eligible W1 candidates, W clean-baseline support held at 20/24, controls 49, all other pools at/above quota → exact CSP rejects W1=11<12; passing W total/controls cannot rescue it). Expected results for all pre-existing cases are unchanged.
- Corrected `failure-ordinal-parity` regression (in-memory, same builder conventions): a synthetic eligible ledger in which episode-ordinal inheritance would concentrate one subtype (same-parity firing pattern) while fixed-phase failure-ordinal alternation balances to ≤1 per `(robot, cohort)`; assert the corrected health path emits the balanced assignment and the exact allocator places quotas with a valid witness. Serialize literal inputs and per-case selected IDs/witness validation as in v1.
- Additional concrete consumer regressions (in-memory, same conventions): (i) phase-extraction determinism — same canonical key recomputed twice yields the same phase, and both parities are exercised; (ii) prefix-bound sweep — both phases satisfy ≤1 on every prefix of the declared `[0, 64]` boundary; (iii) increment-once semantics — interleaved episodes, non-firing hazard evaluations, maintenance windows, and A events do not advance the P/W failure ordinal; (iv) multi-robot aggregation — a nine-group ledger with per-group ≤1 keeps per-cohort skew ≤9. These pin the exact consumer-visible boundary the v2 defect escaped.

### Fixture C — fixed-probe path (unchanged expectations, harness correction)

The v1 §6 Fixture C canonical descriptor (`a9f999fda657e4a9c4ff0af12975bfb21f958dee9d9c1802e163e43d9a239e88`), in-memory probe inputs, public routines (`healthy_features`, `probe15` feature/standardization/centroid/score/threshold/history-evaluation), and expected observables (Fit std exactly `STD_FLOOR`, zero healthy centroid; 19 zero Calibration scores + positive outlier with q95 = 0.05× outlier; macro `auc_pw/auc_p/auc_w = lcb_pw = 1.0`, 4/4 directional per block) are frozen and unaffected (probe code untouched). The next Task 67 worker re-runs it once through the corrected pipeline as a no-regression path proof with identical expected values, with exactly one harness correction: **remove the wrong extra assertion that treats abrupt-A files as score-zero negatives** (A files are anomaly positives with legitimately non-zero scores; R03 Finding 3). This is a harness bug fix, not an acceptance weakening — every frozen probe gate stays numerically unchanged. Provenance limits ride along unchanged: Fixture C is an in-memory synthetic path proof, not real Fit/Calibration/Development statistics; **no additional Fit/Cal/Dev role-root generation is authorized in Task 67** (actual fresh Fit/Cal/Dev statistics and the Task 69 probe remain mandatory later).

### Proof adequacy

Passing disposable fixtures prove only the corrected construction path on disposable coordinates; they confer no design credit and predict no candidate outcome. The complete 16-role candidate's structural and fixed-probe qualification — including EG4 on Development and Confirmation — must still pass on the bound roster. Signal-physics / schema / cutoff / eligible-denominator / real-49-control-preservation interactions are covered by Fixtures A+B through the real loader, real allocator, and real structural predicates. The next Task 67 worker additionally runs the focused affected-profile test scope and records its exact command/result; the full public-reload suite (both reloads × both seeds, cross-reload comparison, v1 invariance comparison) left unrun by A03 must be completed, not sampled.

## 7. Risks, residual uncertainty, and review boundary

1. Residual thin-total risk unchanged: histories with small cohort failure totals can still fail quotas even under exact failure-ordinal balance (worst-case floor (T−9)/2 per 9-robot aggregation). Preflight remains the honest gate; no guarantee is claimed.
2. Offset-phase clustering unchanged: hash-derived starting phases are fixed per key with both parities observed and no first-label bias by construction, but adversarial alignment across robots is possible; it has no systematic bias and no independence claim is made.
3. No surviving mechanism ambiguity for the two retired failures: 32018 (raw W1 11 < 12, zero W1 censored) and the 94100 v2 counterexample (analytic §2.3 reproduction) each have a standing code-grounded explanation; no competing eligibility-cause hypothesis remains for either.
4. The next Task 67 worker must not replay retired seeds `32000–32031` under any method, contact candidate-3 seeds, overwrite the A03 v2 roots, or promote any fixture root.
5. Acceptance of this Task 66 correction is documentary: a concrete bounded method decision + rationale + canonical formula + theorem with executed output + source/proof scope + permissions + risks + handoff, a reviewable complete artifact, and no source implementation or candidate contact. Implementation belongs to a separate Task 67 worker after a fresh review PASS and Bronze checkpoint of this contract; no self-checkpoint and no Task 67 code or next-candidate binding occurs here.

## 8. Handoff

- Review this v3 prospectively; on PASS, Bronze-checkpoint exactly the owned v3 + changelog paths, then dispatch a fresh Task 67 worker to implement §3 + §6 Fixtures A/B/C exactly as frozen above (including the Fixture C harness correction and the new `.sprint18-iterative-contract-v3` attempt namespace).
- Bind the next full candidate only in Task 68 after Task 67's proof passes its own review.
- No bounded question for Main: every material tradeoff in §3 was resolved from the existing source text and the stored counterexample (fixed-phase pattern mirrors the proven A branch; manifest shape preserved; no physics/gate change; rejected alternatives documented with code-grounded reasons; honest aggregate limits stated without stochastic claims).
