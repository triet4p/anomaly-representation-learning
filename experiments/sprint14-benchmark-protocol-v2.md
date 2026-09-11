# Sprint 14 Benchmark Recovery Protocol v2 (FROZEN, AWAITING GATE B1 APPROVAL)

**Status:** Internally frozen by the Sprint 14 worker as the cycle-1 Task 5 amendment. NOT APPROVED — no implementation, materialization, audit, probe execution, or outcome generation under v2 may start before Evidence Gate B1 passes with zero actionable findings. Any further semantic change requires a new protocol version (`v3`, …), Main approval, and a fresh evidence review before materialization.

**Normative contract:** `docs/BENCHMARK_MEASURABILITY_EXIT_GATES.md`, file SHA256 `44d085b86ab7cf80aca15162a50aea74fa35c5bf7e7773514822a9e02e71efb1` (local-only locked-`uv` execution; evidence reviews between cycles; deep review only after provisional `MEASURABLE`). This protocol adds prospective freezing detail but never weakens, reinterprets, or replaces that contract. On any conflict, the exit-gate document governs and this protocol must be amended.

**Protocol identifier:** `sprint14-benchmark-protocol-v2`. This file's own SHA256 digest is recorded after freezing in `artifacts/sprint-14/task-5-cycle-1.md` and `artifacts/sprint-14/cycle-ledger.json` (self-embedding a digest would be circular).

**Diagnostic cycle:** cycle 1 of at most 3. A cycle ends at an evidence review. Failure after cycle 3 closes Sprint 14 as `NOT_MEASURABLE`.

## 0. Amendment record (v1 → v2; v1 preserved)

`experiments/sprint14-benchmark-protocol-v1.md` is preserved byte-identical (SHA256 `8279e2b5ec293a6517ec53ee38c1063313cf86a770ab9159b367d6c8b6697a1b`) and remains the pre-amendment record. This v2 is the single bounded cycle-1 DGP amendment driven by Task 4 (`artifacts/sprint-14/task-4-cycle-1.md`, Option A as directed by Main). Exact deltas, nothing else:

1. **Fleet/scheduler (§1):** robot vocabulary extended by one identity, `robot-09`; fleet size 8 → 9; route-B first-stage robot `robot-02` → `robot-09` (program-03 stays on route-B); total units (1152), arrival cadence (11200 s interval/jitter), route volumes, and unit-allocation rule unchanged — throughput-preserving by construction.
2. **P/W subtype emission (§1, §3, §6):** deterministic prospective partition draws `P1/P2` and `W1/W2` (probability one half each) on a dedicated history-seed-derived sub-stream at each non-abrupt episode open; recorded on episode and linked failure records; zero effect on wear/hazard/duration/severity/manifestation paths. Documented separately as holdout-enablement, not count tuning. Holdout-eligible Fit/Calibration pools additionally exclude P2/W2-attributed rows.
3. **Task 9 instrumentation (§12, new, non-semantic):** mandatory per-event lead-support endpoint tallies, per-robot evaluable-positive concentration, and per-candidate control-rejection reasons in the Design audit. No semantic change.
4. Everything else — cohort rates/physics, A1/A2 distribution, event/eligibility/metric semantics, maintenance/reset/quarantine/censoring rules, floors, margins, probe, seeds (700–733), roles, local-only runtime, verdict precedence, crosswalk rows — is carried forward byte-for-meaning identical from v1.

## 1. Cycle-1 amended DGP (frozen settings)

- Factory: 180-day span, day-120 development cutoff, 7-day quarantine; **9 robots on four routes** (routes A/B/C two-stage, route D three-stage); 1152 units, arrival interval/jitter 11200 s; op duration 600 s (φ ≈ 1.6 ops/robot-day).
- Frozen route table (the amendment changes exactly one cell: route-B stage 1 robot):

| Route | Stage 1 (robot, program) | Stage 2 (robot, program) | Stage 3 (robot, program) |
|---|---|---|---|
| route-A | robot-01, program-01 | robot-02, program-02 | — |
| route-B | **robot-09**, program-03 | robot-03, program-04 | — |
| route-C | robot-04, program-05 | robot-05, program-06 | — |
| route-D | robot-06, program-07 | robot-07, program-08 | robot-08, program-01 |

Every robot serves exactly one stage (the v4.1 robot-02 double duty is removed, not relocated: 9 slots over 9 robots). Unit allocation across routes, per-stage op volumes, and total throughput are unchanged, so total failure arrivals are preserved in expectation.

- Health/hazard (`HealthConfig`, seed = history seed): aging `1e-7`, pre-episode wear `3.0e-5`, noise `1e-3`, onset `0.3`, `alpha = 5.0`, severity scale `2.0`, `upcoming_p = 0.55`; corrective maintenance 2 d, preventive every 30 d + 1 d; cohorts `P(share 0.45, base 5.0e-10, wear 2.0e-4, H* 2.0)`, `W(share 0.30, base 3.0e-9, wear 5.0e-5, H* 1.35, κ 0.3)`, `A(share 0.25, abrupt_rate 1.1e-5 s⁻¹, subtypes A1/A2)` — all numerically identical to v4.1/v1. The ninth robot trajectory uses the same seeded construction as the existing eight (deterministic history-seed derivation; no shared-stream reordering — Task 6 implementation constraint, B1-verified).
- Causal cohort dynamics (v4.1 §§3–4, frozen): upcoming non-abrupt cohort drawn at history start and after every maintenance; pre-episode base wear; episode opens at H ≥ onset tagged with the upcoming cohort; base hazard fires only while an episode is open; deterministic fire at the cohort threshold; abrupt hazard always active and aborts open episodes (A records carry onset None, duration 0); maintenance closes interrupted episodes at recommissioning and redraws the upcoming cohort; trajectories never connect across maintenance.
- Manifestation (v4.1 §4, frozen): precursor severity = progressive_severity(G) × κ_cohort (κ_P 1.0, κ_W 0.3), capped 0.95; suppression within 7 d before abrupt failures; failure-file severity = min(0.95, 0.9·s/2), s ∈ {1,2,4} a label plus failure-magnitude scale only, never applied retrospectively to precursor windows.
- **P/W subtype partition (new, holdout-enablement only):** at each non-abrupt degradation-episode open, draw the episode subtype on a dedicated RNG sub-stream derived deterministically from (history seed, cohort id, episode ordinal): `P1`/`P2` with probability one half each, `W1`/`W2` with probability one half each. The subtype is recorded on the episode and on every linked failure-ledger record. Abrupt episodes keep the mechanism `A1`/`A2` draw untouched. The draw touches no wear, hazard, duration, severity, or manifestation code path: it is a deterministic accounting partition of existing P/W mechanisms with unchanged cohort rates and physics. Caveat (declared, for the B1 reviewer): these subtypes carry no distinct physics; they satisfy the presence-based holdout contract (reserved slice, eligible in Confirmation/Sealed, excluded from Fit/Calibration). Any future claim needing distinct P/W mechanism physics requires a new amendment.
- Frozen v1 gap resolved: the v4.1 `A1/A2`-only emission gap is closed by the partition rule above; holdout presence for P2/W2 is implementable and no longer blocks promotion to Confirmation on subtype grounds.

## 2. Prospective role topology and roster (cycle 1)

Whole histories are the independence unit. Roles are non-overlapping and roster-exact; every declared history is retained, including failures. No history is retried, replaced, omitted, or relabeled because its counts or probe outcomes are unfavorable.

| Role | Histories | Seeds (factory = health = scheduler = signal = temporal) | Permitted use |
|---|---|---|---|
| Design diagnostic | 4 | 700, 701, 702, 703 | Diagnose structural misses; estimate safety margins. Replaced only by a new protocol version. |
| Fit | 3 | 710, 711, 712 | Fit the fixed observable probe (§7); downstream healthy reference support. |
| Calibration | 1 | 713 | Select only the probe operating threshold; verify healthy support. |
| Confirmation | 4 | 720, 721, 722, 723 | One-shot observability/generalization gate after the DGP is frozen. Never reused as design data. |
| Sealed evaluation | 4 | 730, 731, 732, 733 | Structural audit and sealing only in Sprint 14; reserved for future independent evaluation. |

Freshness statement: none of these 16 seeds has ever been materialized or observed. Previously used or reserved seeds — Sprint 12 (100–103, 200–203), Sprint 13 v4 (300–308, 400–403), Sprint 13 v4.1/v4.1.1 (500–508, 600–603), analytic throwaways (920–927, 929, 930), simulation replicates (seed 13) — are all excluded (45 seeds total), and the 16 seeds above are pairwise distinct across roles.

## 3. Cohort definitions and subtype holdouts

Cohorts P (progressive, subtypes P1/P2 per §1 partition), W (weak precursor, subtypes W1/W2), A (abrupt/no-precursor, subtypes A1/A2 mechanism). Abrupt events stay in category-conditional and overall ledgers with no minimum predictability claim and no silent exclusion.

Reserved holdouts (outside Fit and Calibration; must occur eligible in Confirmation and Sealed):

- one complete robot identity: `robot-08` (untouched by the amendment; still closes route-D);
- one complete program identity: `program-03` (remains on route-B first stage, now operated by `robot-09`);
- one abrupt-mechanism subtype: `A2` (distribution and rate unchanged; held out of Fit/Calibration/Confirmation fitting and selection per v4.1 precedent; present eligible in Confirmation and Sealed for descriptive evaluation);
- one progressive-mechanism subtype `P2` and one weak-precursor subtype `W2`: held out of Fit/Calibration (any P2/W2-attributed row excluded from holdout-eligible pools and from probe fitting/threshold inputs), required present and eligible in Confirmation and Sealed.

## 4. Temporal views, horizons, censoring, maintenance/reset, quarantine, control predicates (frozen)

- Temporal-view membership (`test-temporal` in `member_views`) is required in POS-ELIGIBLE rows, ANCHOR-ELIGIBLE rows (hence control members), and E5 eligible-surveillance rows. Pre-cutoff DEV rows never enter evaluation controls or denominators.
- Causal seven-day pre-failure horizon `[T−7d, T]` per positive; symmetric 7-day clearance for control anchors (no failure start in `[e−7d, e+7d)`).
- Whole-window reset semantics, centralized in `window_intersects_reset` / `positive_window_intersects_reset`: a positive horizon intersecting any maintenance/recommission boundary is unevaluable (counted, never imputed); a control candidate intersecting one is rejected; alert episodes split at resets (no bridging). Row-level overlap checks remain for file eligibility; the span rule governs units and episodes.
- Eligible-operational-row predicate: a row counts iff NOT censored AND NOT full-interval maintenance-overlapping (`win.start < row.end AND row.start < win.end`), used identically for the E5 numerator, the E5 denominator days, and robot-day floors. A (robot, day) counts iff ≥1 eligible row ends that day.
- 7-day quarantine with per-reason tallies; censoring exclusions; maintenance/recommissioning resets split eligibility and alert episodes exactly once; strictly chronological file/event ordering; one robot performs at most one operation at a time.
- Negative controls: deterministic non-overlapping same-history/robot matched windows under the frozen eligibility predicate; greedy matching order is frozen in the audit implementation and never re-run to favor counts.

## 5. E1–E5 metric semantics (frozen)

Complete composite contract (v4.1.1 §2, carried forward): E1 tie-aware event AUC; E2 median first-alert lead + IQR with `SPARSE(n)` below 10 recalled events; E3 event recall plus exact Clopper–Pearson 95% interval; E4 persistence median + IQR with the same SPARSE rule; E5 false-alert episode count plus episodes-per-robot-day rate, with the zero-count rule-of-three bound. `computable` validates the full composite output, never AUC alone. Deterministic score fixtures must show: constant scores → AUROC exactly 0.5; strictly correct ordering → AUROC/G-rank exactly 1.0; reversed ordering → 0.0; onset-only alerts → lead exactly 0.0 d; the frozen advance-alert fixture → positive lead with reportable median/IQR at ≥10 recalled P and ≥10 recalled W; the false-alert fixture reproduces its analytically declared episode count and rate; E1–E5 outputs finite, schema-valid, deterministic; repeated execution byte-identical strict-JSON. Fixture success proves computability only and is never cited as anomaly-model or early-warning performance. A file ending at failure onset counts as detection but never as positive lead time. No file-AUROC anywhere, ever.

## 6. Hard floors, design margins, concentration, support, and stability (frozen verbatim from the contract)

Per-history floors (every Design, Confirmation, and Sealed history independently, after all §4 exclusions):

| Quantity | Hard floor | Design promotion target (floor × 1.25, rounded up) |
|---|---|---|
| Evaluable progressive events (P) | 10 | 13 |
| Evaluable weak-precursor events (W) | 10 | 13 |
| Evaluable abrupt events (A) | 8 | 10 |
| All evaluable positive events | 30 | 38 |
| Deterministic non-overlapping same-history/robot negative-control windows | 25 | 32 |
| Evaluable robot-days | 150 | 188 |
| Distinct robots contributing eligible positives | 6 | 6 |
| Distinct robots contributing negative controls | 6 | 6 |
| Programs represented in each P and W cohort | 2 | 2 |

Concentration limits per history: no robot >35% of eligible positives; no robot >40% of negative controls; no program >60% of either predictable cohort; each of P/W/A between 15% and 60% of all evaluable positives.

Lead-support predicates per history: ≥80% of P events and ≥80% of W events must have ≥3 eligible file endpoints inside the causal 7-day horizon, ≥1 eligible endpoint ending ≥24 h before failure, no maintenance/recommissioning reset inside the evaluated window, and a clean same-history/robot baseline window under the frozen eligibility predicate.

Downstream support floors: Fit ≥200 verified-healthy files and ≥6,000 valid patches in aggregate (every fallback/sparse conditioning group reported); Calibration ≥40 verified-healthy eligible rows; no sealed, confirmation, quarantined, censored, or maintenance-overlapping row may enter Fit or Calibration. Holdout-eligible pools exclude `program-03`, `robot-08`, and P2/W2-attributed rows; raw pools reported alongside.

Design promotion rule: every Design history must meet every design promotion target and concentration limit, and Fit/Calibration support floors must be met by the planned role configuration. A pooled, median, or three-of-four pass is a failure. Confirmation/Sealed stability: for each of P/W/A/negatives/robot-days, the Confirmation median must lie within [0.75, 1.25]× the frozen Design median (EG5), and the Sealed median within [0.75, 1.25]× the frozen Confirmation median (EG6). The v2 Design median is the new baseline: stability bands apply Design→Confirmation→Sealed within v2, never against v4.1 counts.

## 7. Fixed causal observable probe (frozen before any cycle-1 outcome)

One simple probe is allowed as a DGP sanity check; it is a benchmark observability check, not a production baseline.

- **Inference inputs (frozen):** causal observable telemetry only — per-file handcrafted patch descriptors already justified by Sprint 12 (`src/representation/handcrafted.py`: per-channel mean/std/rms/slope/min/max, pairwise cross-channel correlation and mean-difference, low/high spectral-band log1p power, max absolute step, valid fraction), mean-aggregated over valid patches in the causal 7-day pre-endpoint window, plus duration, usage, and time-since-maintenance counters. Forbidden as inputs: simulator health, future state, labels, cohort IDs, failure times, anomaly labels, diagnostic-only metadata.
- **Fitting (Fit only):** Standardizer center/scale fit on Fit eligible healthy rows; reader = L2-regularized logistic regression (C = 1.0, deterministic solver, fixed seed 0) trained on Fit eligible rows with frozen file-level precursor labels (1 iff the file ends inside a causal 7-day pre-failure horizon under §4 eligibility), excluding any P2/W2-attributed row. No refit on any other role, ever.
- **Threshold (Calibration only):** single operating threshold = 95th percentile of probe scores on Calibration verified-healthy eligible rows. No failure-time use in threshold selection.
- **Evaluation (once, on Confirmation only):** the EG4 thresholds restated in `artifacts/sprint-14/task-2.md` §5. Failing EG4 retires all four Confirmation histories; keeping favorable histories or retuning the probe is forbidden.

## 8. Amendment, cycle, and retirement rules (frozen)

- Every semantic change (physics, scheduling, maintenance, quarantine, windows, floors, roles, probes, verdict precedence) is prospective: new protocol version, Main approval, fresh evidence review before materialization. Post-generation semantic edits fail EG0 and retire the affected roots.
- One bounded DGP amendment per diagnostic cycle (this v2 is the cycle-1 amendment), driven only by design-role structural diagnostics and the frozen probe. Learned representation, anomaly, warning, and calibrated-risk outputs cannot select generator settings, histories, thresholds, or gates.
- Design miss → new protocol version, fresh Design roots, consumes one cycle. Confirmation failure (any hard floor, observable threshold, or generalization condition) retires the complete Confirmation set (and the candidate's Design/Fit/Calibration roots) and requires a new protocol version with fresh Design and Confirmation roots, consuming one cycle; failed Confirmation data never becomes design data. Sealed structural failure → final `NOT_MEASURABLE`; sealed outcomes never tune another candidate in this sprint.
- At most three diagnostic cycles. Failure after cycle three closes Sprint 14 as `NOT_MEASURABLE`. Operational PASS and scientific `MEASURABLE` are distinct.
- If fresh v2 Design histories miss any promotion target, Gate B2 evidence review occurs first; Main then chooses the next methodology from the new evidence. No Option B and no A+B cycle-2 pairing is pre-authorized by this protocol.

## 9. Canonical runtime, roots, and seals (frozen; local-only)

- Canonical runtime is the local repository/workstation: the locked `uv` environment (`uv.lock`, `.python-version` 3.13, `requires-python >=3.12`; observed toolchain `uv 0.9.7`, Python 3.13.3) invoked as `uv run python -m synth.cli …`. No remote host, no push/pull synchronization loop, no bulk-data transfer.
- Public entry point per history (local): `uv run python -m synth.cli --chronological --profile sprint14-v2 --seed <S> --role <R> --protocol sprint14-benchmark-protocol-v2 --output data/generated/sprint14-v2/<ROLE>/` (the `sprint14-v2` profile implements exactly the §1 settings; Task 6 owns the minimal implementation after Gate B1 approval — no implementation under v2 before Gate B1).
- Explicit local roots: `data/generated/sprint14-v2/<ROLE>/` under the repository, gitignored for bulk bytes. Bounded manifests committed per history (manifest SHA256, config hash, role, protocol tag, seeds, counts, timing, episode-subtype tallies); bulk bytes never committed. Seals via `write_seal`/`verify_seal` with the v2 protocol tag; manifest digests round-trip verified and digest-recorded in the cycle ledger. First-attempt status recorded; no retry/replacement.
- Canonical commit, runtime versions, output roots, and seal format are recorded in the cycle ledger at materialization time (Task 11 in later cycles; Task 8 for cycle-1 Design), not invented here.

## 10. Verdict precedence (frozen)

- `MEASURABLE`: every mandatory gate EG0–EG7 passes, every confirmation and sealed history passes its per-history structural floors, the fixed observable probe passes on confirmation histories, and the final evidence and deep-review gates have zero actionable findings.
- `NOT_MEASURABLE`: required roots exist and audits execute, but any mandatory structural, observability, replication, or generalization gate fails after the allowed diagnostic cycles.
- `UNAVAILABLE`: a required root cannot be generated or verified, provenance is incomplete, or the frozen metric implementation cannot produce a required estimand despite the structural units being present.
- `NOT_MEASURABLE` and `UNAVAILABLE` both block Sprint 15; no partial advancement. EG7 is evaluated if and only if EG0–EG6 mechanically yield provisional `MEASURABLE` (EG7 pass requires zero-findings evidence reviews, preserved failed/superseded records, zero-findings sprint-wide differential deep review, plan/manifest/seal/index agreement, mechanical derivation). Provisional `NOT_MEASURABLE`/`UNAVAILABLE` records EG7 `NOT_RUN` with no deep reviewer.

## 11. EG0–EG7 clause crosswalk (against exit-gate digest `44d085b8…2e71ef`)

| Gate clause | Exit-gate lines | Protocol section |
|---|---|---|
| EG0 protocol version + content digest | 105–118 | Header (identifier, digest in task-5-cycle-1.md/ledger); §8 (post-generation edits fail EG0) |
| EG0 exact generator configuration + causal physics | 105–118 | §1 (amended fleet/routes/subtypes; amendment record §0) |
| EG0 role names, history identities, seeds | 105–118 | §2 (16-history roster, fresh seeds 700–733, unchanged) |
| EG0 failure-cohort definitions + subtype holdouts | 105–118 | §3 (P1/P2, W1/W2, A1/A2; robot-08, program-03 holdouts intact) |
| EG0 temporal views, horizon, censoring, maintenance/reset, quarantine, control predicates | 105–118 | §4 |
| EG0 every hard floor, design target, concentration limit, probe metric, verdict-precedence rule | 105–118 | §6 (floors/margins/concentration/support/stability); §7 (probe); §10 (precedence) |
| EG0 canonical commit, runtime, output roots, seal format | 105–118 | §9 (commit/runtime/roots recorded at Task 8/11; format frozen here) |
| EG0 diagnostic-cycle number, maximum count, retirement ledger | 105–118 | Header (cycle 1 of ≤3); §8; cycle ledger |
| EG1 causality/determinism/role isolation (100%, zero exceptions) | 120–133 | §4 (ordering, one-op-per-robot, reset splitting, leakage ban, roster-exactness); §9 (reload/provenance/seals) |
| EG2 design structural margin (4 histories × promotion targets, no pooling) | 135–140 | §6 promotion rule; §2 design role; §8 design-miss path |
| EG3 metric-contract computability (fixtures 0.5/1.0/0.0/0.0 d, advance fixture, FAR fixture, byte-identical JSON) | 141–154 | §5 |
| EG4 observable-signal sanity (AUROC/recall/lead/FAR thresholds; ≥3/4 histories >0.55; A reported, no minimum) | 156–172 | §7 (frozen probe); thresholds restated in task-2.md §5 for audit use |
| EG5 confirmation structural generalization (floors, EG1+EG3, [0.75,1.25]× design medians, holdouts, no retry) | 174–184 | §6 (floors + stability); §2 (one-shot); §3 (holdouts); §8 (retire-all rule) |
| EG6 sealed structural generalization (floors, EG1, [0.75,1.25]× confirmation medians, seals, no scores, one failure → NOT_MEASURABLE) | 186–200 | §6; §2; §9 (sealing); §8/§10 (sealed-failure consequence) |
| EG7 evidence/closeout iff provisional MEASURABLE else NOT_RUN | 202–212 | §10; plan workflow (deep reviewer iff provisional MEASURABLE) |
| Boundaries 1–8 (history unit, no seed selection, no learned influence, one probe, confirmation discipline, sealed discipline, prospective-only, ≤3 cycles) | 21–31 | §§2, 7, 8, 10 |
| Role minimums 4/3/1/4/4 + robot/program/P/W-subtype reservations | 32–50 | §2 (counts exceed-or-meet: 4/3/1/4/4 exactly); §3 |
| Hard floors + concentration + 80% lead-support + Fit/Cal support | 52–88 | §6 + §4 |
| Threshold rationale (no gate weakened to fit counts) | 90–101 | §§6–7, thresholds (task-2.md §5); §8 (no post-hoc floor changes) |

No protocol clause weakens any crosswalked gate: floors, margins, thresholds, unanimity rules, and the EG7 conditional are transcribed verbatim from the exit-gate document.

## 12. Cycle-1 Task 9 instrumentation requirements (non-semantic)

The cycle-1 Design audit (Task 9) must additionally report, per history (and where stated, per robot), without changing any §4–§7 semantic:

1. per-event lead-support endpoint tallies: for every P/W evaluable positive, the count of eligible file endpoints inside `[T−7d, T]`, whether ≥1 ends ≥24 h before failure, reset presence in the evaluated window, and clean-baseline-window existence — the four 80%-predicate components separately;
2. per-robot evaluable-positive concentration: evaluable P/W/A/total shares by robot (the accepted v4.1 audit records raw-ledger cohorts per robot only);
3. per-candidate control-rejection reasons: for every rejected control candidate, exactly one reason — reset-span, clearance conflict, greedy-spacing skip, or anchor-ineligible (with the anchor excluder: quarantined/censored/maintenance-overlap/non-temporal-view).

These close the Task 4 §3.5 measurement gaps so the next diagnosis is fully instrumented.
