# Sprint 18 Data Method C2 v1 — Prospective Cycle 2 Methodology Amendment

**Method ID:** `sprint18-data-method-c2-v1`
**Status:** PROSPECTIVE FREEZE FOR FRESH INDEPENDENT REVIEW — authorizes no implementation, binding, preflight, materialization, training, scoring, or Sealed access. Cycle 2 may proceed to Task 42 only after this document plus `artifacts/sprint-18/task-41.md` pass independent evidence review with zero actionable findings.
**Scope:** Exactly one causal, bounded generator-rate override on the accepted Sprint 15 Candidate 7 method, for Sprint 18 candidate Cycle 2 only. No model, probe, gate, floor, threshold, role-permission, or verdict-semantics change.

## 1. Normative composition and precedence

The Cycle 2 data contract is the exact composition of:

1. Accepted Sprint 18 full-training protocol v2 (`experiments/sprint18-full-training-protocol-v1.md` + v2 amendment), SHA-256 `e1dcd17914e920520ad0d8f50e185874b95414e6a4fe907903619daf145622d2`;
2. Inherited normative measurability gates (`docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md`), SHA-256 `d294762331ded4fd213f6870e563e32e17e12c632d7cb11a80373478ede34e1c`;
3. Accepted Sprint 15 Candidate 7 method (`experiments/sprint15-benchmark-protocol-v7.md`), SHA-256 `a1fc09db2e246ed79d0595aec953a7788fd1b47c4981fbe1d92017d944b8d7b6`, as executed through the frozen generator sources in §2;
4. This amendment, which overrides exactly one literal (§5) for profile `sprint18-c2` only.

On any conflict, the gate document (item 2) governs; this amendment never weakens, reinterprets, or replaces it. The v7 document bytes and all accepted Sprint 18 v2 protocol/spec bytes are never edited by this amendment. Cycle 1 records stand byte-identical and retired.

## 2. Accepted inputs and hash bindings

All digests below were re-verified read-only from disk during Task 41 diagnosis (exact `sha256sum` over each path; outputs matched the accepted Task 3/4/5 records):

| Input | Path | SHA-256 |
|---|---|---|
| Accepted Sprint 18 protocol v2 | `experiments/sprint18-full-training-protocol-v2.md` | `e1dcd17914e920520ad0d8f50e185874b95414e6a4fe907903619daf145622d2` |
| Inherited measurability gates v2 | `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` | `d294762331ded4fd213f6870e563e32e17e12c632d7cb11a80373478ede34e1c` |
| Candidate 7 protocol v7 | `experiments/sprint15-benchmark-protocol-v7.md` | `a1fc09db2e246ed79d0595aec953a7788fd1b47c4981fbe1d92017d944b8d7b6` |
| Frozen observable-probe spec v7 | `experiments/sprint15-observable-probe-v7.md` | `4d4c2b53b3879efdf92935e362956c87d16892feba8276367f1e5aa13aa4a51a` |
| Frozen probe implementation | `src/synth/probe15.py` | `a08b3d5fb83001e1f5c43f4c56ff536bae85e41d494db289304aeb33a339242b` |
| Generator: config | `src/synth/config.py` | `55fed2ef5b253d48c1b0c7f1dd5f7f966cdd03ad250cb86a988ac8de23d5c056` |
| Generator: health process | `src/synth/health.py` | `e5854641fb85eb9b88e05ee9d8e503c69a9432b13177d8c62ad38c3494e7fc6d` |
| Generator: event predicates | `src/synth/events.py` | `85100f5e0aca47dd2e8b01a08c58f39d32be4f1eab469cdc38e4a4b57768169d` |
| Generator: chronicle/materializer | `src/synth/chronicle.py` | `b5992d6af3c4387d18df746b43be17b4bfc1fac6938e8b44be35b6fb7c65dafd` |
| Generator: allocator/audit | `src/synth/balanced.py` | `d62de3422bd51f870d39d18b1a8bb942f9ca73fc0044d8f23cc2d0948bd35a43` |
| Generator: CLI | `src/synth/cli.py` | `3c264a33c7e74dd2b1ab2c9cfbb7a7e2e524f1b39a1228f734460edb0d900a37` |
| Generator: package init | `src/synth/__init__.py` | `5c979561a2f01581e2943f23e30061788d10dabc2ec1172cbd9f25275032b168` |
| Generator: preflight runner | `src/synth/preflight15.py` | `19f0300e97d55609c27f51403de9cc0f7a898684df7050424972fc4e4fb21dd1` |
| Cycle 1 corrected qualification | `artifacts/sprint-18/task5-measurability.json` | `dae1659eb538d940eace0c44a768f4ccc64df9751256e4367021a553a2f7e596` |
| Cycle 1 role binding | `experiments/sprint18-role-binding-v1.json` | `493c5d6a7a9b807f1fe915f00bb917be45a11183512581606eee34ee678bb228` (canonical digest `d2a398e223cbdfc14fd64cd582513aa71dc9c4e44737f6e66abf73954f270bdf`) |
| Cycle 1 raw preflight | `artifacts/sprint-18/task4-preflight.json` | `7a15d9d41852f1a9646e62ccb8be8230f92049dd8397614882d2c095001fae0d` (`PREFLIGHT-PASS`, 16/16, `no_write`) |
| Cycle 1 materialization | `artifacts/sprint-18/task4-materialization.json` | `3fb0059af0ea0326b92406ca426783c1ed5d3b1db6475f328a6d5f32fe901a11` (16 roots, 82,874 rows, 1,300 shards) |

Cycle 1 lineage: source commit `de3acd05dff6a9b9d3a06ab5dd5b8b782f8d268d1` (as recorded by Task 4); seeds 31800–31815 (consumed, retired, never reassigned).

## 3. Cycle 1 diagnosis from Design-only evidence

No Confirmation outcome sized any parameter below. CONF-03 (A = 19/140 = 13.57%, EG5 mix FAIL) is recorded here solely as a rejection fact; §5 is sized from Design roots only.

### 3.1 Raw-to-eligible A attrition (Design roots, read-only manifest ledgers vs qualification)

| Design history (seed) | Raw ledger P / W / A (total) | Eligible P / W / A (total) | Censored (all `reset-horizon`) | Eligible A share |
|---|---|---|---|---|
| H-S18-DES-01 (31800) | 61 / 54 / 55 (170) | 61 / 54 / 20 (135) | 35 | 20/135 = 14.81% FAIL |
| H-S18-DES-02 (31801) | 58 / 45 / 65 (168) | 56 / 45 / 33 (134) | 34 | 33/134 = 24.63% pass |
| H-S18-DES-03 (31802) | 49 / 63 / 50 (162) | 48 / 61 / 23 (132) | 30 | 23/132 = 17.42% pass |
| H-S18-DES-04 (31803) | 52 / 51 / 68 (171) | 52 / 50 / 32 (134) | 37 | 32/134 = 23.88% pass |

DES-01 detail: every censored event is abrupt-cohort (ledger A 55 → eligible A 20: A1 27→12, A2 28→8); P and W lose zero events (61→61, 54→54). Its only failed promotion check is `each_cohort_mix_15_to_60_percent`; all other EG2 checks pass (P≥13, W≥13, A≥10, total≥38, controls 66≥32, robot-days 1386≥188, caps, programs, lead-support 100%). DES-01 misses the 15% floor by 0.25 events — exactly one eligible A event. Per-event sensitivity at its denominator is `(135−20)/135² ≈ 0.63` percentage points, so any ±1–2 A censoring swing decides the gate: the history is fragile, not marginal on physics.

### 3.2 Causal mechanism: cohort-asymmetric reset-horizon censoring

The audit predicate (`eligible_anchors` in `src/synth/balanced.py`) declares an abrupt anchor eligible iff its 7-day causal window `[T−7d, T]` is reset-free with ≥1 POS-ELIGIBLE file, while P/W anchors additionally require degradation physics (onset + minimum duration gate + 3 endpoints). In the generator (`src/synth/health.py`), P/W failures can only fire after health reaches onset, a duration gate opens (P ≥ 2 d, W ≥ 6 d post-onset), and thresholds are crossed — dynamics that keep P/W out of post-maintenance shadows. Abrupt failures fire from a memoryless per-operation hazard (`abrupt_rate`) at any time, including inside the 7-day shadow of preventive (30-day cadence) and corrective (after every failure) maintenance. The v7 deterministic stratification (§1a) balances subtype *emission* to ≤1 per (robot, cohort), but eligibility censoring is cohort-asymmetric by construction: it removes ~53–64% of raw A (observed A yields 0.36/0.51/0.46/0.47) and ~0–4% of raw P/W. The v7 projection (worst A subtype pool 11 vs need 8) under-called this because it allowed one unit of filtering skew; observed DES-01 skew is 15 (A1) and 20 (A2) censored events. DES-01's A2 eligible pool is exactly 8 = quota need (zero slack): the same mechanism threatens both the mix gate and allocation headroom.

### 3.3 Denominator statement

The mix denominator is the full eligible set (135 for DES-01), not the allocated 64 (quotas 24/24/16 place exactly; A2 = 8/8 placed). With P+W ≈ 115 fixed by unaffected physics, the floor needs A ≥ 0.15 × T, i.e. A ≥ 20.29 → 21 for DES-01. Quota compliance (16 A placed) is therefore compatible with mix failure (20 eligible A): allocation feasibility does not imply mix compliance.

### 3.4 Preflight/qualification mismatch (predicate coverage, not nondeterminism)

The rejection-only preflight (`src/synth/preflight15.py`) executes calendar → eligibility → exact-CSP witness → subtype pools → controls → support margins, and records margins `{controls, robot_days, robots_pos, robots_neg, programs_P, programs_W}`. It never evaluates cohort mix on the eligible set, P/W 80% lead-support, eligible-set concentration, or positive-count floors. Hence 16/16 `PREFLIGHT-PASS` is consistent with Task 5 mix FAIL: preflight proves quota placeability, not gate compliance.

Determinism holds: preflight eligible subtype pools equal the qualification eligible subtype tallies exactly for all four Design seeds (e.g. seed 31800: 33/28/14/40/12/8 in both records; likewise 31801–31803). The gap is predicate coverage, not a materialization skew. Cycle 2 keeps the identical preflight machinery and adds the missing mix enforcement at the DesignQualification stage (§8: Task 45 retires the candidate on any Design mix miss before downstream roots exist — the structural lesson of Cycle 1, where all 16 roots were materialized before the two mix failures were found).

## 4. Causal physical invariants (preserved, non-negotiable)

The following are unchanged and constrain any implementation of §5:

- Calendar-first order: frozen calendar → eligibility → exact-CSP allocation → waveforms; waveform generation never moves, adds, or removes an anchor.
- Degradation physics: aging/wear/noise evolution, onset, P/W duration gates, threshold firing, recommissioning draws, preventive + corrective maintenance structure and durations.
- Hazard structure: P/W `base_rate × exp(αH + βU)` firing; memoryless abrupt branch; `upcoming_p` 0.52; subtype→gain severity coupling (P1 κ=1.0, P2 κ=0.55, W1 κ=0.30, W2 κ=0.16, cap 0.95); labels assigned at episode/failure time, never relabeled post-hoc.
- Stratified emission alternation bounds (≤1 per robot-cohort / per-robot A) and stream-preservation discipline.
- Event/metric definitions, eligibility predicates (horizon, reset-split, quarantine/censoring, lead-support), quarantine/split semantics, control-window selection, exact-CSP formulation (quotas 24/24/16; splits 12/12/12/12/8/8; caps 22/14; 14-day spacing; HiGHS exact path with independent witness validation).
- Balanced case-control interpretation (quotas control evaluation support; never fleet prevalence).

## 5. The amendment C2-M1: bounded abrupt-rate compensation (single change)

**Override:** for profile `sprint18-c2` only, the v7 §1a literal `A abrupt_rate 2.2e-5` is replaced by **`A abrupt_rate 3.3e-5` (factor κ = 1.5)**. This supersedes exactly that one literal as applied to Cycle 2; the v7 file bytes are untouched, and every other §1a literal (P `base_rate` 5.0e-10, W `base_rate` 3.0e-9, `upcoming_p` 0.52, wear/thresholds/gains, deterministic stratification mechanism and alternation bounds, timing/density documentary targets for P/W) remains in force.

**Rationale (Design-driven, not fraction-tuned):** κ = 1.5 restores approximately the protocol-nominal 25% A share at mean observed Design yield (mean raw A 59.5 → 89.25; mean yield 0.454 → eligible ≈ 40.5 against mean P+W 106.75 → ≈ 27.5%), while lifting the worst observed Design yield class (DES-01: yield 0.3636, P+W 115) from 14.81% to a projected 30/145 = 20.69% with +9.7 events of margin. κ was not solved to flip 20/135 to 21/135 (κ = 1.05 would do that); it targets nominal-mix restoration with worst-case margin.

**Why a rate, not a predicate or quota change:** floors, quotas, caps, windows, spacing, thresholds, and predicates are inherited acceptance criteria (§9) and cannot compensate; changing them would be gate-tuning. The deficit is a density-under-censoring problem, and cohort rates are explicitly documentary targets ("realized mix emerges from the rates", v7 §1a) with rate-revision precedent (v6→v7). Raising A (rather than cutting P/W) preserves P/W promotion strength and probe power, and is probe-neutral by construction (all probe gates are P/W-based; A is companion-reported with no predictability minimum).

**Bounded expected effects (yield-invariance projection ± sensitivity):**

- Primary: DES-01-class histories move from 20/135 = 14.81% to ≈ 30/145 = 20.69% if censoring yield holds; the fix tolerates up to ≈ 32% relative yield degradation (break-even yield 0.246 vs observed 0.364) before the floor binds again.
- Secondary: DES-01-class A2 pool moves from 8 (zero slack) to ≈ 12 (slack 4) under yield invariance, restoring allocation headroom.
- Cost bound: total ledger grows ≈ +18% (≈ +30 abrupt events/history, P/W-ledgers unchanged first-order); observed Design controls (58–84, minimum margin +10 over the 48 construction target) absorb this well below the v5 1.5×-total-volume regime that collapsed controls to 13/11. Control erosion is the principal adverse channel and is enforced, not assumed (§10).
- Second-order feedback (more A → more corrective shadows → slightly lower yields for all cohorts, altered health paths) is real, unsigned in detail, and bounded in sign expectation (A gains net; P/W currently at 96–100% yield with large floor margins). It is verified at Task 44 (preflight pools/margins) and Task 45 (per-history mix), which retire the candidate on any miss.

## 6. Rejected alternatives (considered, not taken)

- **Reset-shadow-aware abrupt gate (suppress A emission inside known 7-day reset shadows):** causally targeted but first-order effect on eligible counts is zero by construction (it removes only doomed events); its benefit is entirely second-order shadow reduction, unboundable from Design data prospectively. Rejected for unboundability.
- **Cutting P/W rates to shrink the denominator:** erodes P/W promotion margins and probe power to fix an A-density problem; rejected as weakening measured strengths.
- **Quota/cap/spacing/window/predicate/floor/threshold changes:** inherited acceptance criteria; changing them is gate-tuning, explicitly forbidden.
- **Maintenance-schedule or duration changes:** frozen causal physics; rejected.
- **Post-hoc subtype rebalancing or relabeling:** breaks subtype→gain coupling and waveform causality; forbidden by v7 §0.
- **Seed search, retry, replacement, resampling, pooling, history rescue, Confirmation-to-Design promotion:** forbidden one-shot/no-retry rules (§8); rejected.
- **Sizing κ from CONF-03 (19/140):** Confirmation-driven parameter tuning; explicitly not done (§3 preamble).

## 7. Cycle, seed, identity, and role namespace

- **Candidate cycles:** Cycle 1 (seeds 31800–31815) is consumed and retired. At most nine fresh cycles (Cycles 2–10) remain under the user-authorized maximum of 10 total. A cycle starts when its reviewed method/binding proceeds to a rejection-only preflight, whether that preflight or later qualification rejects or passes.
- **Seed-namespace rule:** Cycle k (2 ≤ k ≤ 10) uses exactly the disjoint block 31800 + 16×(k−1) through 31800 + 16×k − 1 in fixed role order Design×4, Fit×3, Calibration×1, Development×4, Confirmation×4. Cycle 2 is therefore 31816–31831. Every block is disjoint from Sprint 15 bands (1000–1617 + proof), Sprint 17 (2800–2815), Cycle 1 (31800–31815), and model seeds (181801–181803). A seed may be retired but never reassigned. Task 43 verifies disjointness mechanically before binding.
- **Identity rule:** Cycle 2 history IDs match `^H-S18C2-(DES|FIT|CAL|DEV|CONF)-[0-9]{2}$`, unique across all cycles; the exact 16 strings are frozen by Task 43, not here.
- **Directory rule:** Cycle 2 roots live under `data/generated/sprint18-full-training-c2-v1/<ROLE>/<history-id>`; the Cycle 1 directory is never written again and its roots are never reused, screened, pooled, or promoted.
- **Model seeds:** unchanged (`181801, 181802, 181803`); disjointness re-verified at Task 43.
- **The 16 role permissions (binding instances frozen at Task 43):**

  - C2 Design histories (4): structural/measurability qualification and healthy-only coefficient-pilot retention evaluation; Design-only diagnosis on failure. Never final fitting.
  - C2 Fit histories (3): verified-healthy representation fitting, EMA targets, Fit reference banks/standardizers, Fit-only auxiliary fitting and cross-fitting.
  - C2 Calibration history (1): verified-healthy per-arm/per-seed q95 thresholds only.
  - C2 Development histories (4): complete descriptive comparison and diagnostics after all configs are frozen. No coefficient, arm, mask, checkpoint, branch, or threshold selection.
  - C2 Confirmation histories (4): one-shot locked final comparison only, opened solely under Task 47 after all prior gates pass.

- **Universal prohibitions:** no Sprint 15 Sealed history/path opened, listed, hashed, probed, loaded, or touched; no Cycle 1 root reused for fitting, screening, pooling, replacement, or rescue; no failed Confirmation history ever becomes Design data; no representation execution before Task 47 `ELIGIBLE`.

## 8. Stage-wise promotion, retirement, verdict, and one-shot safeguards

Order is prospective and mechanical; each stage runs at most once per cycle:

1. **Task 42 (method proof, disposable fixtures only):** implement profile `sprint18-c2` as an additive factory (`sprint18_c2_history_config(seed)` = v7 factory with only `abrupt_rate` replaced) plus profile/protocol constants; prove deterministic byte-identity per seed, quota placeability on fixtures, causal order (calendar-first, anchors never moved), stratification alternation bounds, role-permission plumbing, and control-space behavior vs §5 cost bound. Keep all Cycle 1 roots immutable; create no candidate roots. Any proof miss stops Cycle 2 before binding.
2. **Task 43 (binding):** freeze the 16 Cycle 2 identities, disjoint seeds, paths, permissions (§7), canonical binding digest, and versioned generator/audit source digests (parent digests §2 plus the minimal Task 42 delta). No roots created.
3. **Task 44 (roster preflight + Design materialization once):** one rejection-only preflight over the full 16-seed roster (no writes); on unanimous pass, materialize exactly the four Design roots once (`overwrite=false`, attempt markers). Any preflight miss retires Cycle 2 in full with no retry, replacement, or tuning.
4. **Task 45 (Design qualification gate):** audit provenance, determinism, chronology, causal units, per-history floors/mix (immutable 15–60%), caps, lead-support, no-pooling, frozen fixtures for every Design history. Any miss retires Cycle 2; downstream roots are never materialized; Confirmation never opens.
5. **Task 46 (Fit/Calibration/Development):** only after Design PASS — materialize each prebound non-Confirmation root once, verify support/provenance, fit the frozen probe on Fit, calibrate on Calibration, audit Development. Any miss retires Cycle 2.
6. **Task 47 (Confirmation + eligibility verdict):** only after all prior gates PASS — materialize and audit each prebound Confirmation history once under frozen EG0–EG5 and the frozen probe; record each independent per-history gate and the mechanical `ELIGIBLE` / `REJECTED-CANDIDATE` verdict without model scoring. On PASS, Main unblocks Task 6 after evidence review; on FAIL, preserve the full cycle and create Cycle 3 tasks only if capacity remains (Cycles ≤ 10).

**One-shot/immutable safeguards:** `no_write` preflight; `overwrite=false` materialization; per-stage attempt markers; seeds never reassigned; failed cycles preserved byte-identical and retired in full; no second run of any stage within a cycle; methodology changes only via a new prospective versioned amendment (new tasks, new review) — never by editing this freeze or any accepted bytes.

## 9. Explicitly unchanged gates, floors, and thresholds

Immutable 15–60% per-cohort mix on every Design/Confirmation/Sealed history independently; hard floors P≥10/W≥10/A≥8/total≥30/controls≥25/robot-days≥150/robots≥6+6/programs≥2+2; concentration caps 35%/40%/60%; 80% P/W lead-support; Design promotion targets (P/W≥13, total≥38, controls≥32, robot-days≥188); Fit (≥200 files, ≥6,000 patches) and Calibration (≥40 rows) support floors; EG3 fixtures; observable-probe thresholds (macro PW≥0.65, block LCB95>0.50, ≥3/4 histories PW>0.55, macro P≥0.70, macro W≥0.60, P recall≥0.50/lead≥1d, W recall≥0.25/lead≥0.5d, FAR≤0.05); stability band [0.75, 1.25]; no-pooling; verdict precedence; eight-arm registry, training contract, coefficient grids, checkpoint rule, and all model/spec terms. None is waived, reinterpreted, or retuned by this amendment.

## 10. Failure conditions (non-exhaustive, each retires or stops Cycle 2 as stated)

- Task 42 fixture proof misses determinism, quota placeability, causal order, alternation bounds, or control behavior → stop before binding.
- Task 43 disjointness/digest/permission mismatch → stop before preflight.
- Task 44 preflight miss on any seed (quota shortfall, control/support shortfall, witness breach, replay anomaly) → retire Cycle 2.
- Task 45 any Design history mix/floor/cap/lead/chronology/provenance miss → retire Cycle 2; downstream roots never materialized.
- Task 46 Fit/Calibration support miss or Development fixed-probe miss → retire Cycle 2.
- Task 47 any Confirmation per-history gate miss or stability-band breach → `REJECTED-CANDIDATE`, preserve full cycle.
- Any Sealed contact, Cycle 1-root reuse, retry/replacement/pooling, or out-of-order execution → protocol breach; affected work invalid.

## 11. Review handoff

The reviewer should verify: (a) §3 arithmetic against `task5-measurability.json` (SHA `dae1659e…`) and read-only manifest ledgers; (b) preflight-pool/qual-tally equality for seeds 31800–31803; (c) κ sizing uses Design data only and targets nominal restoration, not the 20/135 or 19/140 fractions; (d) override minimality (one literal) and invariant preservation (§4, §9); (e) namespace/permission/stage rules (§7–§8) with exact binding deferred to Task 43; (f) all §2 digests reproduce via `sha256sum`; (g) no code, data, root, qualification, or plan-status write occurred under Task 41. Task 41 remains [~] until this review passes; no later task starts before it.
