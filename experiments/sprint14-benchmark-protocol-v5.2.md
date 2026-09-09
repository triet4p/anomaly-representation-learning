# Sprint 14 Benchmark Recovery Protocol v5.2 (FROZEN — AWAITING FRESH EVIDENCE REVIEW)

**Status:** Internally frozen by the Sprint 14 worker as a tooling-only identity clarification. NOT APPROVED — nothing may execute under v5.2 (no re-evaluation rerun, no Task 8 materialization) before a fresh evidence review passes with zero actionable findings. Any semantic DGP change requires a new protocol version (`v5.3`, …), Main approval, and a fresh evidence review before materialization.

**Normative contract:** `docs/BENCHMARK_MEASURABILITY_EXIT_GATES.md`, file SHA256 `44d085b86ab7cf80aca15162a50aea74fa35c5bf7e7773514822a9e02e71efb1` (local-only locked-`uv` execution; evidence reviews between cycles; deep review only after provisional `MEASURABLE`). This protocol adds prospective freezing detail but never weakens, reinterprets, or replaces that contract. On any conflict, the exit-gate document governs and this protocol must be amended.

**Protocol identifier:** `sprint14-benchmark-protocol-v5.2`. This file's own SHA256 digest is recorded after freezing in the cycle ledger (self-embedding a digest would be circular).

**Diagnostic cycle:** cycle 3 of at most 3 (Cycles 1–2 consumed; no cycles remain after this one).

## 0. Amendment record (v5.1 → v5.2; identity-contract clarification, zero DGP change)

- `experiments/sprint14-benchmark-protocol-v1.md` (`8279e2b5…6c8b6697a1b`), `-v2.md` (`5b6eb212…7a7c67404`), `-v3.md` (`45b5ef69…fb85d1`), `-v4.md` (`88f2b426…bb3176`), `-v4.1.md` (`66572d72…03a5ebb4`), `-v5.md` (`159225d7…01d3016`), `-v5.1.md` (`988c34c8…090f46e`) preserved byte-identical. Cycle 1 roots/audits retired (`e018`); Cycle 2 roots preserved as diagnostic-only; quarantined Cycle 1 numbers are not cited anywhere in this file.
- Exact deltas, nothing else:
  1. **Identity contract made explicit (§9):** Protocol v5 (`159225d7…01d3016`) is the single immutable identifier for DGP physics, manifests, and seals. Protocol v5.2 is a proof/audit tooling revision bound alongside v5 — it versions the harness (re-evaluation mode, guards, output labels), never the science. No manifest or seal may carry a `v5.1` or `v5.2` tag; Task 8 uses profile `sprint14-v5` with manifest/seal tag `sprint14-benchmark-protocol-v5`; every Task 8/9 evidence record binds the v5 DGP digest, the v5.2 tooling digest, and the exact canonical commit.
  2. **Tooling label fix:** the re-evaluation output declaration is corrected from `"protocol": "sprint14-benchmark-protocol-v5.1"` to `"protocol": "sprint14-benchmark-protocol-v5"` plus `"tooling": "sprint14-benchmark-protocol-v5.2"`, so tooling outputs can never again be mistaken for a distinct DGP version. The already-produced re-evaluation output (which carries the old label) is preserved byte-identical as superseded evidence; it is not regenerated or mutated.
- The v5/v5.1 DGP (A+C settings, routes, physics, semantics, floors, probe, roster) is otherwise carried forward byte-for-meaning identical. The successful v5.1 re-evaluation stands on its v5 seal tag, which is correct under this dual-version contract; no rerun and no root mutation occurs under v5.2.

## 1. Cycle-3 DGP (identical to v5/v5.1; frozen settings restated)

- Factory: **240-day span**, **day-160 development cutoff**, 7-day quarantine; **9 robots on four routes** (routes A/B/C two-stage, route D three-stage); **1536 units**, arrival interval/jitter 11200 s (unit throughput 6.4/d preserved; per-robot op cadence φ ≈ 1.6 ops/robot-day preserved); op duration 600 s.
- Frozen route table (Option A):

| Route | Stage 1 (robot, program) | Stage 2 (robot, program) | Stage 3 (robot, program) |
|---|---|---|---|
| route-A | robot-01, program-01 | robot-02, program-02 | — |
| route-B | robot-09, program-03 | robot-03, program-04 | — |
| route-C | robot-04, program-05 | robot-05, program-06 | — |
| route-D | robot-06, program-07 | robot-07, program-08 | robot-08, program-01 |

Every robot serves exactly one stage. Unit allocation across routes, per-stage op volumes per unit time, and total throughput rate are unchanged.

- Health/hazard (`HealthConfig`, seed = history seed): aging `1e-7`, pre-episode wear `3.0e-5`, noise `1e-3`, onset `0.3`, `alpha = 5.0`, severity scale `2.0`, `upcoming_p = 0.55`; corrective maintenance **1 d**, preventive every **60 d + 1 d**; cohorts `P(share 0.45, base 5.0e-10, wear 2.0e-4, H* 2.0)`, `W(share 0.30, base 3.0e-9, wear 5.0e-5, H* 1.35, κ 0.3)`, `A(share 0.25, abrupt_rate 1.1e-5 s⁻¹, subtypes A1/A2)` — numerically identical to v5/v5.1.
- Causal cohort dynamics (v4.1 §§3–4, frozen): upcoming non-abrupt cohort drawn at history start and after every maintenance; pre-episode base wear; episode opens at H ≥ onset tagged with the upcoming cohort; base hazard fires only while an episode is open; deterministic fire at the cohort threshold; abrupt hazard always active and aborts open episodes (A records carry onset None, duration 0); maintenance closes interrupted episodes at recommissioning and redraws the upcoming cohort; trajectories never connect across maintenance.
- Manifestation base (v4.1 §4, frozen): precursor severity = progressive_severity(G) × κ with κ from the per-subtype gain table below; capped 0.95; suppression within 7 d before abrupt failures; failure-file severity = min(0.95, 0.9·s/2), s ∈ {1,2,4} a label plus failure-magnitude scale only, never applied retrospectively to precursor windows.
- **P/W subtype physical fork (identical to v5/v5.1):** at each non-abrupt degradation-episode open, the episode subtype is drawn on a dedicated RNG sub-stream derived deterministically from (history seed, cohort id, episode ordinal), `P1`/`P2` and `W1`/`W2` at probability one half each — causal in time, mirroring the abrupt-subtype draw precedent (`src/synth/health.py:412`). Dedicated sub-stream preserves failure-timing invariance. Stored on the degradation episode, propagated to linked failure-ledger records (diagnostic metadata only — never into model-visible file rows, whose key set is frozen).
- **Per-subtype manifestation gains (frozen, identical):**

| Subtype | Gain κ_sub | Derivation |
|---|---|---|
| P1 | 1.0 | legacy full amplitude |
| P2 | 0.55 | √(1.0·0.3): logarithmic midpoint of the generator's own two cohort gains |
| W1 | 0.3 | legacy `WEAK_AMPLITUDE_SCALE` |
| W2 | 0.16 | 0.3·√0.3: the same ratio step below W1 |

Wear, hazard, thresholds, onset, durations, shares, severity cap, quarantine, and the symptom-probability law are untouched: cohort rates, timing, density per unit time, duration bounds (P 2–15/med 5–10; W 6–28/med 12–24), and every symptom-blind eligibility predicate behave identically in expectation. Abrupt A1/A2 behavior is untouched. Subtypes differ in manifestation coupling, not in wear/hazard physics.

## 2. Prospective role topology, roster, and root identities (cycle 3, final)

Whole histories are the independence unit. Roles are non-overlapping and roster-exact; every declared history is retained, including failures. No history is retried, replaced, omitted, or relabeled because its counts or probe outcomes are unfavorable.

| Role | Root identities | Seeds (factory = health = scheduler = signal = temporal) | Permitted use |
|---|---|---|---|
| Design diagnostic | H-DESIGN-9, 10, 11, 12 | 780, 781, 782, 783 | Diagnose structural misses; estimate safety margins. Replaced only by a new protocol version. |
| Fit | H-FIT-7, 8, 9 | 784, 785, 786 | Fit the fixed observable probe (§7); downstream healthy reference support. |
| Calibration | H-CAL-3 | 787 | Select only the probe operating threshold; verify healthy support. |
| Confirmation | H-CONF-6, 7, 8, 9 | 788, 789, 790, 791 | One-shot observability/generalization gate after the DGP is frozen. Never reused as design data. |
| Sealed evaluation | H-SEAL-9, 10, 11, 12 | 792, 793, 794, 795 | Structural audit and sealing only in Sprint 14; reserved for future independent evaluation. |

Freshness statement (proven in Task 5 Cycle 3, not assumed): the 16 roster seeds plus Task 7 proof seeds 796/797 are pairwise distinct; their intersection with the v1–v3 band (700–733), all ledgered seeds to date, disposable proof seeds (740, 741), all 45 historical seeds, and test-generation seeds (910–914, 931–933) is strictly empty. Root identities continue past every prior H- family in both sprints. Cycle-3 Design materialization uses fresh roots under `data/generated/sprint14-v5/`; retired/quarantined roots are never touched.

## 3. Cohort definitions, subtype holdouts, and role allowlists

Cohorts P (progressive; physical subtypes P1/P2 per §1 fork), W (weak precursor; W1/W2), A (abrupt/no-precursor; mechanism subtypes A1/A2). Abrupt events stay in category-conditional and overall ledgers with no minimum predictability claim and no silent exclusion.

Reserved holdouts (outside Fit and Calibration; must occur eligible in Confirmation and Sealed):

- one complete robot identity: `robot-08` (closes route-D);
- one complete program identity: `program-03` (route-B first stage, operated by `robot-09`);
- one abrupt-mechanism subtype: `A2` (distribution and rate unchanged; held out of Fit/Calibration/Confirmation fitting and selection per v4.1 precedent; present eligible in Confirmation and Sealed for descriptive evaluation);
- one progressive-mechanism subtype `P2` and one weak-precursor subtype `W2`: held out of Fit/Calibration, required present and eligible in Confirmation and Sealed.

Role allowlist (frozen; enforced at pool-construction and probe-input time):

| Role | P1/W1/A1 rows | P2/W2-attributed rows | A2 rows | program-03 / robot-08 rows |
|---|---|---|---|---|
| Design 780–783 | included, per-subtype tallies required (§12) | included, per-subtype tallies required | included | included |
| Fit 784–786 | included | **excluded** from pools and probe-fitting inputs | excluded (precedent) | excluded (precedent) |
| Calibration 787 | included | **excluded** | excluded (precedent) | excluded (precedent) |
| Confirmation 788–791 | included | included; P2/W2 presence + eligibility required | present eligible (descriptive) | included |
| Sealed 792–795 | structural audit tallies per subtype | structural audit tallies per subtype | structural audit only | structural audit only |

Design measurability without replacement: with ~20–28 P and ~14–24 W evaluable events per history at 50/50 subtype draws, the probability of a missing subtype in any Design history is below 2^-14 by construction (structural assurance, not selection); floors are cohort-level, so even an absurd-tail absence cannot fail a floor. No history is replaced or reselected for subtype absence in any role — an absence is reported honestly.

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

False-alert normalization: EG4's false-alert bound (≤0.05 episodes per evaluated robot-day) is a rate and self-normalizes as evaluated robot-days grow with the longer calendar. No floor changes.

Design promotion rule: every Design history must meet every design promotion target and concentration limit, and Fit/Calibration support floors must be met by the planned role configuration. A pooled, median, or three-of-four pass is a failure. Confirmation/Sealed stability: for each of P/W/A/negatives/robot-days, the Confirmation median must lie within [0.75, 1.25]× the frozen Design median (EG5), and the Sealed median within [0.75, 1.25]× the frozen Confirmation median (EG6). The v5 Design median is the baseline: stability bands apply Design→Confirmation→Sealed within v5/v5.1/v5.2 (identical DGP), never against older counts.

## 7. Fixed causal observable probe (frozen before any cycle-3 outcome)

One simple probe is allowed as a DGP sanity check; it is a benchmark observability check, not a production baseline.

- **Inference inputs (frozen):** causal observable telemetry only — per-file handcrafted patch descriptors already justified by Sprint 12 (`src/representation/handcrafted.py`: per-channel mean/std/rms/slope/min/max, pairwise cross-channel correlation and mean-difference, low/high spectral-band log1p power, max absolute step, valid fraction), mean-aggregated over valid patches in the causal 7-day pre-endpoint window, plus duration, usage, and time-since-maintenance counters. Forbidden as inputs: simulator health, future state, labels, cohort IDs, failure times, anomaly labels, diagnostic-only metadata — and subtype IDs (diagnostic metadata only, never model-visible).
- **Fitting (Fit only):** Standardizer center/scale fit on Fit eligible healthy rows; reader = L2-regularized logistic regression (C = 1.0, deterministic solver, fixed seed 0) trained on Fit eligible rows with frozen file-level precursor labels (1 iff the file ends inside a causal 7-day pre-failure horizon under §4 eligibility), excluding any P2/W2-attributed row per the §3 allowlist. No refit on any other role, ever.
- **Threshold (Calibration only):** single operating threshold = 95th percentile of probe scores on Calibration verified-healthy eligible rows. No failure-time use in threshold selection.
- **Evaluation (once, on Confirmation only):** the EG4 thresholds restated in `artifacts/sprint-14/task-2.md` §5. Failing EG4 retires all four Confirmation histories; keeping favorable histories or retuning the probe is forbidden.
- **Disclosed composition effect:** P2/W2 muted precursors shift the Confirmation signal mix relative to pre-subtype baselines; macro EG4 thresholds are unchanged (contract) and whether the frozen probe still passes is empirical. This is accepted variance of a physical-subtype benchmark, not a tunable parameter.

## 8. Amendment, cycle, and retirement rules (frozen)

- Every semantic change (physics, scheduling, maintenance, quarantine, windows, floors, roles, probes, verdict precedence) is prospective: new protocol version, Main approval, fresh evidence review before materialization. Post-generation semantic edits fail EG0 and retire the affected roots.
- One bounded DGP amendment per diagnostic cycle (this v5/v5.1/v5.2 line is the cycle-3 amendment), driven only by design-role structural diagnostics and the frozen probe. Learned representation, anomaly, warning, and calibrated-risk outputs cannot select generator settings, histories, thresholds, or gates.
- Design miss → final `NOT_MEASURABLE` (no further cycles exist). Confirmation failure retires the complete Confirmation set with no further recourse inside this sprint. Sealed structural failure → final `NOT_MEASURABLE`; sealed outcomes never tune another candidate in this sprint.
- At most three diagnostic cycles (Cycles 1–2 consumed). Failure in this final cycle closes Sprint 14 as `NOT_MEASURABLE` after evidence review, with no deep reviewer (a negative verdict records EG7 `NOT_RUN` per the conditional rule). Operational PASS and scientific `MEASURABLE` are distinct.
- No Cycle 4 methodology is pre-authorized by this protocol; none can exist under the three-cycle cap.

## 9. Canonical runtime, roots, seals, provenance gate, tooling roots, and identity contract (frozen; local-only)

- Canonical runtime is the local repository/workstation: the locked `uv` environment (`uv.lock`, `.python-version` 3.13, `requires-python >=3.12`) invoked as `uv run python -m synth.cli …`. No remote host, no push/pull synchronization loop, no bulk-data transfer.
- Generation uses the `--profile sprint14-v5` code path with an explicit protocol tag and fresh roots: `uv run python -m synth.cli --chronological --profile sprint14-v5 --seed <S> --role <R> --protocol sprint14-benchmark-protocol-v5 --output data/generated/sprint14-v5/<ROLE>/`.
- Explicit local roots: `data/generated/sprint14-v5/<ROLE>/` under the repository, gitignored for bulk bytes. Bounded manifests committed per history (manifest SHA256, config hash, role, protocol tag, seeds, counts, timing, episode-subtype tallies); bulk bytes never committed. Seals via `write_seal`/`verify_seal` with the v5 protocol tag; manifest digests round-trip verified and digest-recorded in the cycle ledger. First-attempt status recorded; no retry/replacement.
- Canonical commit, runtime versions, output roots, and seal format are recorded in the cycle ledger at materialization time (Task 8), not invented here. The exact canonical implementation commit for v5 execution is established at Task 6 and independently evidence-reviewed with Task 7 before Task 8; reviewer approval of the v5.1 tooling correction governed the single deterministic re-evaluation already performed.
- **Provenance gate (binding pre-materialization check, R1–R4 formulation):** no Task 8 run may start unless (a) every runtime-affecting byte matches the canonical commit byte-for-byte (`git status` clean for those paths and blob hashes equal); (b) rejection triggers are clear — no changed or untracked importable code under `src/`, no invoked Task 7/9 script differing from its committed blob, no change to `pyproject.toml`, `uv.lock`, or the `.python-version` pin; (c) all other working-tree dirt is inventoried, digested, and proven non-importable and non-invokable. No globally clean tree is required; unrelated user files must not be modified, deleted, or committed. Any violation of (a) or (b) aborts generation before the first history. A base commit plus uncommitted runtime changes is not a canonical commit.
- **Tooling roots (collision-free):** Task 7 proof entry points write only to their frozen output roots (`server-task7-cycle-3/`, `server-task7-cycle-2-rerun/`) with disposable data roots under `data/generated/sprint14-task7-c3(-rerun)/`, and the v5.1 re-evaluation wrote only to `artifacts/sprint-14/server-task7-cycle-3-reeval/` (disposable seal-check copies alongside it) — never to cycle-1 paths, never over superseded evidence. Every entry point refuses to start when a target root exists and is non-empty; overwriting preserved evidence requires a new protocol version, not a flag.
- **Identity contract (binding):** Protocol v5 (`159225d7…01d3016`) is the single immutable identifier for DGP physics, manifests, and seals under this line: no manifest or seal may carry a `v5.1` or `v5.2` tag, and no Task 8/9 invocation may record one. Protocol v5.2 is a proof/audit tooling revision bound alongside v5 — it versions the harness (re-evaluation mode, guards, output identity labels), never the science. Every Task 8/9 evidence record binds three digests: the v5 DGP digest, the v5.2 tooling digest recorded below, and the exact canonical commit. The previously produced re-evaluation output (which declares the superseded v5.1 tooling label) is preserved byte-identical as superseded evidence; it is clarified, not regenerated or mutated, by this contract.

## 10. Verdict precedence (frozen)

- `MEASURABLE`: every mandatory gate EG0–EG7 passes, every confirmation and sealed history passes its per-history structural floors, the fixed observable probe passes on confirmation histories, and the final evidence and deep-review gates have zero actionable findings.
- `NOT_MEASURABLE`: required roots exist and audits execute, but any mandatory structural, observability, replication, or generalization gate fails after the allowed diagnostic cycles.
- `UNAVAILABLE`: a required root cannot be generated or verified, provenance is incomplete, or the frozen metric implementation cannot produce a required estimand despite the structural units being present.
- `NOT_MEASURABLE` and `UNAVAILABLE` both block Sprint 15; no partial advancement. EG7 is evaluated if and only if EG0–EG6 mechanically yield provisional `MEASURABLE` (EG7 pass requires zero-findings evidence reviews, preserved failed/superseded records, zero-findings sprint-wide differential deep review, plan/manifest/seal/index agreement, mechanical derivation). Provisional `NOT_MEASURABLE`/`UNAVAILABLE` records EG7 `NOT_RUN` with no deep reviewer.

## 11. EG0–EG7 clause crosswalk (against exit-gate digest `44d085b8…2e71ef`)

| Gate clause | Exit-gate lines | Protocol section |
|---|---|---|
| EG0 protocol version + content digest | 105–118 | Header (identifier, digest in ledger); §8 (post-generation edits fail EG0); §9 (provenance gate; identity contract) |
| EG0 exact generator configuration + causal physics | 105–118 | §1 (A+C settings; amendment record §0) |
| EG0 role names, history identities, seeds | 105–118 | §2 (16-history fresh roster with root identities; seeds 780–795; proof seeds 796/797) |
| EG0 failure-cohort definitions + subtype holdouts | 105–118 | §3 (P1/P2, W1/W2, A1/A2; robot-08, program-03 holdouts intact; role allowlists) |
| EG0 temporal views, horizon, censoring, maintenance/reset, quarantine, control predicates | 105–118 | §4 |
| EG0 every hard floor, design target, concentration limit, probe metric, verdict-precedence rule | 105–118 | §6 (floors/margins/concentration/support/stability); §7 (probe); §10 (precedence) |
| EG0 canonical commit, runtime, output roots, seal format | 105–118 | §9 (commit recorded at Task 8; format frozen here; provenance gate; tooling roots; identity contract) |
| EG0 diagnostic-cycle number, maximum count, retirement ledger | 105–118 | Header (cycle 3 of ≤3, final); §8; cycle ledger |
| EG1 causality/determinism/role isolation (100%, zero exceptions) | 120–133 | §4 (ordering, one-op-per-robot, reset splitting, leakage ban, roster-exactness); §9 (reload/provenance/seals); §13 (subtype checks) |
| EG2 design structural margin (4 histories × promotion targets, no pooling) | 135–140 | §6 promotion rule; §2 design role; §8 design-miss path (final) |
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

## 12. Cycle-3 Task 9 instrumentation requirements (non-semantic)

The cycle-3 Design audit (Task 9) must additionally report, per history (and where stated, per robot), without changing any §4–§7 semantic:

1. per-event lead-support endpoint tallies: for every P/W evaluable positive, the count of eligible file endpoints inside `[T−7d, T]`, whether ≥1 ends ≥24 h before failure, reset presence in the evaluated window, and clean-baseline-window existence — the four 80%-predicate components separately;
2. per-robot evaluable-positive concentration: evaluable P/W/A/total shares by robot;
3. per-candidate control-rejection reasons: for every rejected control candidate, exactly one reason — reset-span, clearance conflict, greedy-spacing skip, or anchor-ineligible (with the anchor excluder: quarantined/censored/maintenance-overlap/non-temporal-view);
4. per-subtype structural tallies: evaluable positives, eligibility rate, and concentration share for each of P1/P2/W1/W2/A1/A2 per history; subtype-stratified descriptive precursor-severity summaries (no gate attached).

These close the Task 4 measurement gaps so the final diagnosis — if needed at closeout — is fully instrumented.

## 13. Cycle-3 Task 7 checks, re-evaluation mode, and sample-share rule (as executed under v5.1; carried forward)

Task 7 proof and re-evaluation behavior, all read-only, no tuning:

1. subtype-draw determinism: two fresh-process materializations at the same seed yield byte-identical episode subtypes and failure-record subtype fields;
2. episode-open ordering: every non-abrupt episode's subtype draw precedes any manifestation file of that episode (causal selection, no retrospective labeling);
3. manifestation separation: mean manifested precursor severity orders P1 > P2 and W1 > W2 at matched manifested-health bands, while episode durations and failure times are statistically indistinguishable across subtypes within cohort (physics preserved);
4. metadata containment: subtype IDs appear in episode/failure/manifest diagnostic records only; the model-visible file-row key set is unchanged (the audit `no-score-keys` contract extended with an explicit subtype-absence assertion);
5. A1/A2 behavior unchanged: abrupt subtype draw, distribution, and suppression semantics identical to v4.1;
6. configured cohort probabilities: the resolved configuration's cohort shares are exactly P=.45/W=.30/A=.25; raw and evaluable realized shares are reported descriptively without gating.
- **No sample-share gating, anywhere:** no Task 7 check gates realized sample shares (raw or evaluable, disposable or candidate roots) against population parameters or against the EG2 15–60% concentration band, which applies solely to evaluable positives on Design/Confirmation/Sealed histories. A finite draw landing anywhere in or out of any band is reported, never failed.
- **Re-evaluation record:** the single Main-authorized deterministic re-evaluation was executed under v5.1 tooling against the preserved seed-796 roots with v5 seal tags; its outputs are preserved byte-identical and are clarified — not regenerated — by the identity contract above.
