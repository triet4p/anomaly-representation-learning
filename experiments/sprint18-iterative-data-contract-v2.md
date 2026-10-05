# Sprint 18 Iterative Data Recovery Contract v2 — failure-ordinal P/W subtype stratification

**Contract ID:** `sprint18-iterative-data-contract-v2`
**Status:** Task 66 prospective contract amendment; **not implementation-reviewed, not role-bound, and not released for candidate data contact.**
**Supersedes:** nothing. `experiments/sprint18-iterative-data-contract-v1.md` (SHA-256 `056105f87b1097c45244287636bcef8c2fc790f3fd894775b9105fbdbfb59e9d`) is preserved byte-identical as the historical v1 record. This v2 adds exactly one construction-method amendment (§3) with its diagnosis (§2), Task 67 proof delta (§6), and versioned identity (§4); every other v1 section is inherited unchanged by reference.
**Scope:** One fresh, common-method, success-only Sprint 18 data-recovery branch (4 Design, 3 Fit, 1 Calibration, 4 Development, 4 Confirmation whole-history roles). Candidate role contact stays behind Task 68 review and Main release. The unchanged Sprint 15 v7 fixed observable DGP sanity probe remains the sole outcome-scoring exception as specified in v1 §2. No eight-arm representation training, learned scorer, bank, Sealed access, or model study is authorized.

## 1. Authority and immutable inputs

This version is prospective and subordinate to the current Sprint 18 plan and the user authorization recorded there. It does not replace or override any full-training protocol or inherited benchmark gate. The plan Tasks 66–71 own the data workflow and status. Task 66 does not edit the Main-owned plan, bind the next final roster, contact a final-role seed, or authorize runtime work.

The v1 §1 input table is inherited unchanged (v2 training package traceability, v7 protocol, C2/C3 records, v7 probe, exit gates). The following additional read-only identities bound this amendment:

| Input | SHA-256 | Use here |
|---|---|---|
| `experiments/sprint18-iterative-data-contract-v1.md` | `056105f87b1097c45244287636bcef8c2fc790f3fd894775b9105fbdbfb59e9d` | Preserved historical v1; all unamended sections inherit from it. |
| `experiments/sprint18-task69-attempt-S18-ITER-0002-A04-preflight-raw.json` | `2f9f79412c0f834bb352153ed6a6ca487ee1751f5e6da97d71d44395a36abd13` | Complete 16-seed preflight raw record (94,738 bytes); the verified failure boundary. |
| `experiments/sprint18-task69-attempt-S18-ITER-0002-A04.json` | `7bae04db6d55e3748bcb540d62a6f62745c9a61c1de99db0b46a4a20a0eb5955` | Bounded machine-readable outcome summary. |
| `experiments/sprint18-task69-attempt-S18-ITER-0002-A04.md` | `e4029a1bc0bcab37ba467b587756fdd2a767906ea62b3c26538a44a9c018c444` | Human-readable execution record. |
| `artifacts/sprint-18/reviews/S18-T69-S18-T69-R05.manifest.json` | manifest SHA-256 `42037c87a746082e85f9eb635d83e7b5abc45ec748273656a428f45e46e54c70` | Independent FAIL review (`advance=false`); sole gate authority for the diagnosis below. |
| Base commit | `860b31719bbb53060f9b0e46c3ae7dc166562bca` on `refs/heads/master` | Wholly-owned checkpointable base for this amendment. |

Task 66 in this assignment performs no generator, preflight, qualification, probe, model, SSH, or raw-root execution, and contacts no candidate seed (including no candidate-3 seed `32032–32047`). The §2 numbers below are read-only reconciliations of the stored authorized evidence above and the current source text; a throwaway arithmetic check only.

## 2. Observed-cause diagnosis: raw W1 supply shortfall, not eligibility filtering

### 2.1 What the stored evidence says

Design seed `32018` (`S18I-ITER-0002-DESIGN-03`) is the single infeasible coordinate of the retired candidate 2 (`PREFLIGHT-FAIL`, `15/16`):

- Eligible subtype pools: P1/P2/W1/W2/A1/A2 = 45/46/**11**/20/34/28. Only `eligible_W1_pool_ge_12` fails; the other five pool floors pass with margins ≥ +3 (smallest: W2 +8 at this seed; tightest anywhere: W2 +3 at seed 32017).
- Raw ledger (reviewer-verified, R05 §3 Finding 1): 32 W events = **11 W1 + 21 W2**, with **zero W1 events lost to reset-horizon censoring** (the single censored W event is W2: eligible W = 31 = 11 + 20). Raw W1 supply itself (11) is below quota (12).
- Every non-pool predicate at this seed passes: selected controls 50 ≥ 48, evaluable robot-days 1669 ≥ 240, 9 positive / 9 negative robots, full eligible cohort mix (A 33.7% / P 49.5% / W 16.8%), P and W lead support both 100% (91/91, 31/31), all structural checks true, all Design-promotion checks true.
- Exact CSP (SciPy 1.18.1 / HiGHS) reports proven infeasible, status 2, 184 variables / 94 constraints, no witness. With 12 binary picks required from a pool of 11, infeasibility is arithmetic, not a solver ambiguity: no unsat-core inference is needed or claimed.
- Rejection ledger: `reset-horizon: 67` is the sole rejection reason at this seed; `no-endpoints` and `lead-shortfall` record zero. Filtering is subtype-blind in realization here (it removed 0 W1).

Therefore the observed failure is a **raw subtype-supply shortfall of exactly one W1 event**, upstream of spacing, caps, controls, mix, lead, and solver behavior. Nothing downstream rescues it, and nothing downstream caused it.

### 2.2 Source mechanism responsible

`src/synth/health.py::_run_robot._draw_pw_subtype` (current source text, flag-on branch): with `stratified_subtype_emission=True` (inherited v7 → C2 → iterative-v1), P/W subtype labels are assigned at **degradation-episode opening** by per-`(robot, cohort)` **episode**-ordinal round-robin alternation `labels[(ordinal + digest[0]) % 2]` on the dedicated hash sub-stream `(health seed, robot, cohort, episode ordinal)`. When a P/W failure later fires, it **inherits** the episode's pre-assigned label (`subtype = open_degradation.subtype`); no subtype is drawn at failure time. The A cohort, by contrast, already assigns subtype at **failure time** by per-robot abrupt-failure-ordinal alternation.

Three code-grounded consequences follow without any probabilistic assumption:

1. **Episode population ≠ failure population.** Per-`(robot, cohort)` episode labels balance to ≤1, but failures are the hazard-gated firing subset of episodes (hazard + `min_duration_gate` + threshold draws on the shared RNG, which is subtype-blind but subset-selective). Non-firing episodes consume ordinals but never enter the ledger.
2. **Firing-parity skew.** Same-parity episode ordinals carry the same label (`(1+d)%2 == (3+d)%2`). Whenever the hazard selects same-parity episodes, all resulting failures share one subtype — the alternation bound is bypassed, not violated.
3. **Across-robot aggregation.** Per-robot episode balance does not imply global 50/50 when per-robot failure counts differ; hash-derived starting offsets aggregate across the nine robots.

At seed 32018 these produce raw 11 W1 / 21 W2 (eligible share 11/31 = 35.5%) with zero W1 censoring. The same thin-split signature recurs across methods (C2 W2 10<12 at 31829; research v7 W1 11<12; C3/C4 abrupt thin pools), while the failing subtype and seed differ each time — the signature of a structural emission-vs-realization gap, not a seed-specific accident.

### 2.3 Why switching seed blocks is not a robust repair

Candidate 2 itself proves the point: 15/16 coordinates pass with comfortable margins under the identical method, yet one coordinate fails by one event in a different subtype than any prior failure. The v7 stratification bounds episode emission; it does not bound failure-subset realization. Any fresh seed block drawn under the unchanged method re-rolls (i) cohort totals (observed W totals span 31–52), (ii) per-robot failure counts, (iii) firing parity, and (iv) offset aggregation — the exact four quantities whose joint realization starved W1 at 32018. Hoping the next block avoids all thin tails is seed shopping, forbidden by the frozen iteration policy, and leaves the next candidate exposed to the identical mechanism in another subtype. A robust repair must bound the **failure**-level subtype realization under the common method.

### 2.4 What is not claimed

No iid/binomial/horizon-probability model, no pass-rate or power estimate, and no feasibility guarantee is inferred: the source text supports no such stochastic assumption, and none is needed. The diagnosis above uses only stored counts, the frozen predicates, the solver status code, and the actual assignment-vs-inheritance code path.

## 3. The one amendment: failure-ordinal P/W subtype stratification (v7 §1a successor)

**Frozen semantics.** Under the existing `stratified_subtype_emission=True` flag (same flag, extended meaning; flag-off behavior stays byte-identical to v6 and earlier), P/W subtype labels are assigned at **P/W failure-firing time** by deterministic round-robin alternation over a per-`(robot, cohort)` **failure** ordinal:

```text
fail_ordinal[(robot_id, cohort_id)] += 1   // only when a P/W failure actually fires
digest = SHA256("sprint18-failure-subtype|<health.seed>|<robot_id>|<cohort_id>|<fail_ordinal>").digest()
subtype = labels[(fail_ordinal + digest[0]) % len(labels)]
```

**Binding rules.**

- The dedicated failure-ordinal digest domain `sprint18-failure-subtype` is distinct from the episode domain `sprint14-subtype`, so failure-alternation phases are independent of episode-alternation phases.
- The shared health RNG is never touched by this assignment (same invariance discipline as the accepted A branch, including its stream-preserving no-op convention where applicable): failure timing, cohort assignment (`upcoming`), density, and all hazard draws are invariant by construction.
- Episode-opening assignment is retained verbatim for diagnostic continuity (episode records keep their labels); the FailureEvent carries the failure-ordinal label. Precursor physics (`temporal._cohort_windows`) already reads the **failure record's** subtype, so the subtype→gain coupling (§5: P1 1.0 / P2 0.55 / W1 0.3 / W2 0.16) is preserved with labels still assigned once at event time, before allocation and before waveform synthesis — never post-hoc relabeled.
- The A branch is unchanged (already failure-ordinal alternation). After this amendment all three cohorts assign subtype at failure time: one unified construction pattern.
- Eligibility remains subtype-blind and unchanged; the allocator, quotas, caps, spacing, controls, mix, lead, and all qualification predicates are unchanged.

**Guarantee bound (honest, no probability).** Per-`(robot, cohort)` failure subtypes differ by ≤1 by construction, so the global split given a cohort failure total depends only on offset aggregation across nine robots (|skew| ≤ 9, typically ~2–3 with hash-balanced offsets). At the observed W total 31 the worst-case floor is (31−9)/2 = 11 — the residual adversarial-alignment risk is honestly retained and preflight still rejects honestly. At the observed median W total (~46) even the adversarial floor is 18 ≥ 12. The amendment removes the systematic parity-subset amplifier (the largest code-grounded skew source); it does not promise that every stochastic candidate passes.

**Considered and rejected.**

- Blind span increase beyond 450 days: rejected — capped by the config validator, changes evaluation denominators and control space, and does not target the split mechanism (v1 §4 feedback warning stands).
- W rate / `upcoming_p` / cohort-share change: rejected — alters per-time physics, density, and control space (v5/v6 precedents); the W total is adequate at median, the split is the defect.
- `upcoming` P/W stratification: rejected — `upcoming` drives wear/threshold dynamics during degradation; stratifying it changes health dynamics, a larger ripple than the failure-label change.
- Quota/gate reduction (W1 12→11), seed-block switching, relabeling, outcome-based reassignment, within-candidate repair: all rejected as gate relaxation, seed shopping, or causality breaks.

**Unchanged physics (binding).** Per-time failure/noise rates, arrival cadence and jitter, program routes, fleet topology, maintenance/recommissioning, quarantine, reset/censoring rules, 7-day horizon, 14-day spacing, caps, 450-day span/cutoff, `n_units=2880`, and every gate/threshold stay exactly as in v1 §4. The complete resolved-config diff of the new factory versus C2 contains only the v1 three exposure leaves plus the additive failure-ordinal mechanism flag semantics; any other diff fails review. No scientific tradeoff requiring a Main/user decision before implementation exists: no constraint is weakened, no physics changed.

## 4. Versioned identity, affected boundaries, and invariants for Task 67

- **New profile:** `sprint18-iterative-v2` (source/CLI profile string). **New factory:** `sprint18_iterative_v2_history_config(seed)`, additively composed from `sprint18_iterative_v1_history_config(seed)` with only the §3 failure-ordinal semantics; the chronicle manifest `protocol` remains `sprint15-benchmark-protocol-v7`; the iterative contract ID recorded in binding/attempt sidecars becomes `sprint18-iterative-data-contract-v2`.
- **Exact affected source/config/API boundaries for Task 67 (nothing else may change):** `src/synth/health.py` (failure-ordinal P/W assignment only; hazard/upcoming/maintenance/severity paths untouched); `src/synth/config.py` (only if a mechanism flag/validator literal is required — prefer reusing the existing `stratified_subtype_emission` flag with extended documented semantics); `src/synth/chronicle.py` (new factory entry + `S18_ITERATIVE_V2_*` profile/contract constants only); `src/synth/cli.py` + `src/synth/preflight15.py` (new profile branch registration only, mirroring the v1 branch); focused tests for the new semantics; `CHANGELOG.md` (concise planned-change entry only, added after proof per v1 practice).
- **Frozen by continuity:** `balanced.py` (quotas/caps/spacing/solver), `events.py`, `temporal.py`, `splits.py`, `scheduled.py`, `scheduler.py`, `dataset.py`, `schema.py`, `probe15.py`, `experiments/sprint18_task5_measurability.py` surfaces, and all v1 §5 validator/bound semantics. Exact positive quotas P1/P2/W1/W2 = 12 each, A1/A2 = 8 each; construction controls ≥ 48 with 49+ retained; hard 25 / Design 32 зоне unchanged; full-ELIGIBLE 15–60 mix, lead/caps, 14-day spacing, robot/program support, and causal chronology unchanged. Probe: 59 causal features, healthy Fit mean/std + centroid, Calibration q95, unchanged EG4 — all frozen.
- **Per-history permissions:** v1 §2 role table inherited unchanged (Design diagnosis only on authorized roles; Fit/Calibration/Development/Confirmation permissions and Sealed prohibition intact). No prior history is promoted into a fresh role.
- **Role-safe common method:** the amendment applies identically to every future history through the shared factory; no per-role/per-seed special-casing and no second pipeline or model pilot.

## 5. Prospective Task 68/69 consequences (no binding here)

The next candidate (N=3, seeds `32032–32047` per the frozen `32000+16×(N−1)` formula) is bound by a fresh Task 68 worker only after this amendment's Task 67 implementation proof and independent reviews pass. No candidate-3 seed is inspected, contacted, prechecked, or generated by Task 66 or Task 67. Retired seeds `32000–32031` (including 32018) are never replayed under any method, even disposably: the mechanism regression below uses fresh disposable seeds.

## 6. Frozen Task 67 disposable proof contract (delta from v1 §6)

All v1 proof hygiene inherits: isolated attempt roots below `data/generated/.sprint18-iterative-contract-v2/task67/<attempt-id>/`, never role roots, never promoted, append-only correction evidence; genuine calendar/signal waveforms via the public CLI; shards → dual fresh-process public reloads → existing qualification functions; no mocks, hand allocations, or compile-only proofs. The accepted v1 proof (6,508-sample Fixture A real materialization, allocator boundary suite, probe path) is the established baseline and is not repeated except where the amendment interacts.

### Fixture A — amended public generation / reload / qualification (fresh disposable seeds)
- Seeds `94100` (`S18I-FIX-REAL-94100`, role `TASK67-FIXTURE-REAL-94100`) and `94101` (`S18I-FIX-REAL-94101`, role `TASK67-FIXTURE-REAL-94101`). Hygiene: the 94100-block has no exact seed binding anywhere in the repo (substring matches are floating-point metric tails such as `0.994100…`, not seed references); Task 67's binding collision check must verify disjointness against every existing history, research seed, model seed, prior candidate, and prior fixture seed (9000, 9001, 93058, 93061, 966660) plus the reserved candidate-3 block `32032–32047` before contact, and refuse on any intersection.
- Config `sprint18_iterative_v2_history_config(seed)`; invocation mirrors v1 §6 Fixture A with `--profile sprint18-iterative-v2`; manifest protocol `sprint15-benchmark-protocol-v7`, 450-day span/cutoff, `n_units=2880`, no `sprint15` block.
- Expected observables: chronological schedule/waveforms within the 450-day calendar; six finite channels; `FileSample.validate` green; shard/manifest/hash/order checks green on both public reloads; model-row allowlist with no leakage; exact allocator witness with all 64 quota members independently validated (caps/spacing/quotas); Task 5 structural + role checks green; **per-`(robot, cohort)` failure-subtype counts differ by ≤1 on each disposable history** (the §3 bound, asserted from the real ledger); eligible-pool subtype margins recorded.
- Timing/density invariance: on each disposable seed, the failure-time and cohort sequences under the amended factory equal those under the v1 factory bit-for-bit (same-seed ledger comparison excluding subtype labels); flag-off byte-identity preserved for legacy profiles.

### Fixture B — allocator boundaries plus failure-ordinal regression

- The v1 Fixture B suite (history seed 966660 in-memory builder) is retained in full for the unaffected allocator/qualification interactions, with one symmetric addition: a `w1-11` mirror of the accepted `w2-11` case (exactly 11 eligible W1 candidates, W clean-baseline support held at 20/24, controls 49, all other pools at/above quota → exact CSP rejects W1=11<12; passing W total/controls cannot rescue it). Expected results for all pre-existing cases are unchanged.
- New `failure-ordinal-parity` regression (in-memory, same builder conventions): a synthetic eligible ledger in which episode-ordinal inheritance would concentrate one subtype (same-parity firing pattern) while failure-ordinal alternation balances to ≤1 per `(robot, cohort)`; assert the amended health path emits the balanced assignment and the exact allocator places quotas with a valid witness. Serialize literal inputs and per-case selected IDs/witness validation as in v1.

### Fixture C — fixed-probe path (unchanged expectations)

The v1 §6 Fixture C canonical descriptor, inputs, and expected observables are frozen and unaffected (probe code untouched). Task 67 re-runs it once through the amended pipeline as a no-regression path proof with identical expected values.

### Proof adequacy

Passing disposable fixtures prove only the amended construction path on disposable coordinates; they confer no design credit and predict no candidate outcome. The complete 16-role candidate's structural and fixed-probe qualification — including EG4 on Development and Confirmation — must still pass on the bound roster. Signal-physics / schema / cutoff / eligible-denominator / real-49-control-preservation interactions are covered by Fixtures A+B through the real loader, real allocator, and real structural predicates.

## 7. Risks, residual uncertainty, and review boundary

1. Residual thin-total risk: histories with small cohort failure totals can still fail quotas even under failure-ordinal balance (worst-case floor (T−9)/2 per 9-robot aggregation). Preflight remains the honest gate; no guarantee is claimed.
2. Offset-phase clustering: hash-derived starting offsets are pseudo-random per (seed, robot, cohort); adversarial alignment across robots is possible but has no systematic bias (no first-label bias by construction, same as v7).
3. No surviving mechanism ambiguity from the authorized evidence: reset-horizon censoring removed 0 W1 at the failing seed, so no competing eligibility-cause hypothesis remains standing for 32018.
4. Task 67 must not replay retired seeds `32000–32031` under any method, contact candidate-3 seeds, or promote any fixture root.
5. Acceptance of this Task 66 amendment is documentary: a concrete bounded method decision + rationale + proof contract + risks, a reviewable complete artifact, and no source implementation or candidate contact. Implementation belongs to a separate Task 67 worker after a fresh review PASS and Bronze checkpoint of this contract; no self-checkpoint and no Task 67 code or next-candidate binding occurs here.

## 8. Handoff

- Implement §3 + §6 Fixtures A/B/C exactly as frozen above after this contract's independent review.
- Bind the next full candidate only in Task 68 after Task 67's proof passes its own review.
- No bounded question for Main: every material tradeoff in §3 was resolved from the existing source text (dual-label variant preserves manifest shape; no physics/gate change; rejected alternatives documented with code-grounded reasons).
