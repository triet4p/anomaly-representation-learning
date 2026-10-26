# S24-7-G3 accepted decision record — hold Phase C; measurement research first (Option B)

- Task: `S24-7-G3` — decide the Phase C research direction (`docs/sprint-plans/sprint-24.md:143-144`).
- Attempts: `S24-7-G3-A01` (options brief + worker record) → `S24-7-G3-A02` (this accepted-decision finalization; replacement under the same criterion and cycle).
- Agent: `S24G3GoldA02` (role `gold-task`, owner-selected Gold, quota `unknown`, full route).
- Review cycle / round: `S24-7-G3-C1` R1 (assigned final review pending at finalization; candidate frozen in §5).
- Criterion reference: `docs/sprint-plans/sprint-24.md:143` — synthesize movement specificity, event usefulness, confounding and limitations; owner decision recorded; any future window/event-centered sequential protocol prospectively specifies endpoints, operating points, update target, retention probes and matched controls before runs; otherwise Phase C stays held with the reason recorded.
- Status: **ACCEPTED — the owner chose Option B.** Phase C stays held with the reason recorded below. No Phase C release, no new study execution, and no new geometry are authorized by this record.
- Scope: **SYNTHESIS / DESIGN ONLY.** No new analyses, runs, forwards, training, source generation, Confirmation access, or Phase C execution. Synthesis of existing verified outputs only. The metadata audit scoped in §4 is PROPOSED, not performed — metadata-only feasibility is the NEXT proposed study, not work done in G3.

## 1. Owner choice recorded

The owner explicitly selected **Option B** (`Giữ C, nghiên cứu phép đo trước`): hold Phase C and run bounded context-/time-resolution measurement research FIRST, in this order:

1. A metadata-only feasibility audit of ordinary-transition / healthy controls (**zero forwards** by construction);

2. then a SEPARATE prospective amendment (with separate execution authorization) before any new geometry, embedding outcomes, or inference.

Explicitly NOT authorized now: new study execution, encoder retraining, a new corpus, retrospective protocol changes, fresh Confirmation access or reuse as validation, and any Phase C release. Main owns the plan change reflecting this choice (`docs/sprint-plans/sprint-24.md:144`) and any future Phase C release — neither is granted by this worker record.

## 2. Evidence the choice rests on (existing verified outputs only; G3 computed nothing new)

- **NO-CALIBRATION** (owner-accepted negative outcome; final review `S24-7-G2-C1` R2 PASS, `advance: true`, 0 actionable findings): healthy-control support is exactly **0/369 SUPPORTED windows** (every window carries ≥ 1 abnormal file and covers ≥ 3 events), so protocol §10.3 condition (b) cannot hold. Raw **369/369 SUPPORTED window-seeds cross q95 (100%)** is preserved as descriptive movement only, not detections.
- Descriptive original-128-d movement (per history across 3 seeds): MMD² medians ≈ 0.006–0.008 (DEV-01 0.00653, DEV-02 0.00596, DEV-03 0.007951, DEV-04 0.005958) vs split-half null q95 ≈ 0.0022–0.0027 (null medians ≈ 0.0015–0.0018); NN-ref ≈ 0.91 vs ≈ 0.70 null. Consistent across seeds 181801/181802/181803 and all 4 histories.
- Consequently **unavailable per protocol**: calibrated MMD event recall (raw 567/950 per seed, identical across seeds by shared-coverage construction), detection delays (recalled-only medians ≈ 6.9–7.7 days, censored), and false-alarm rate (0 raw episodes over 35.3 healthy file-days reported for completeness only — UNDEFINED as a rate under NO-CALIBRATION).
- **Drift confound is association, not causal proof.** HIGH-DRIFT top-decile windows sit systematically higher on every history and seed (seed-181801 HI vs non-HI medians: DEV-01 0.010669 vs 0.006650; DEV-02 0.008608 vs 0.005751; DEV-03 0.014052 vs 0.007193; DEV-04 0.007939 vs 0.005823; 1.1x–2.05x across seeds). Reviewer-independent association: `r(MMD², TV drift) ≈ +0.78` vs `r(MMD², n_abnormal) ≈ −0.40`. This warns that ordinary-regime operational mix drift explains much of the movement, but with `n = 4` histories it cannot prove the encoder is causally driven only by drift.
- Therefore the diagnostic **cannot conclude the encoder is useless, and cannot conclude anomaly clusters do not shift.** The movement is real but unspecific; the anomaly-specific component is unidentified under the executed window design.
- **File-head Task 7 is unchanged and not re-decided:** `NOT_TRIGGERED` — mean `M1` file PR-AUC `0.24181` (< 0.70), mean `M3 − M1` `−0.03298` (not > 0.05); 108/108 reused fits, 828/828 score operations, 3,323,754 per-file rows; `M4`/T10 not executed. File-level PR-AUC is context only per protocol §0, not evidence of collective movement.
- **Comparators are retained but explicitly not identical-support:** MMD mapping covers all 950 manifest failure events (zero-cover 102/89/102/90 counted, censored from delay); `_comparator_events` additionally requires robot-matched in-span file support inside merged degradation-to-maintenance spans (35/35/34/33 events/history, 137 total). Retained frozen-S18-q95 recalls (seed 181801: M0a 0.176–0.485, M0b 0.171–0.364, M0c 0.286–0.455) are therefore not directly comparable to MMD coverage numbers; no rescoring was performed and none is invented here.
- Run bounds preserved: cumulative 26,431 forwards ≤ 41,000 cap; trainable params 0; 372 window rows (369 SUPPORTED + 3 rows for the single DEV-01 UNSUPPORTED window-1, `n = 28 < 30`) and 2,850 event rows (950 × 3 seeds) fully accounted.

## 3. Limitations carried forward (not false-proof)

- Healthy-control feasibility under any NEW observable grouping is **UNKNOWN** — no such audit was in G1/G2/G3 scope, and G3 performed none. The only support evidence on record is the G1 metadata-only fixture scenario (fleet-wide 14-day windows held support on the audit fixture while per-robot 7-day windows collapsed 1457/2007 below minimum) plus the accepted-data window projection (31 windows/history, UNSUPPORTED 1/0/0/0) — neither audits healthy-control density under an alternative grouping.
- Calibrated event utility remains unavailable under NO-CALIBRATION.
- Comparator identical-support mapping remains unavailable without rescoring (prohibited in R2).
- Post-run live model/bank digests remain unrecoverable (asserted in memory only, honestly disclosed, never retroactively recreated).
- Small-history uncertainty: history-cluster bootstrap is descriptive given `n = 4`; inherited temporal exposure means any future sequential read is label adaptation with prior unlabeled exposure, never strictly prospective end-to-end (protocol §2).
- Sprint 18 stays `UNRESOLVED`; original Confirmation stays evaluation-only secondary context with no adaptation, tuning, or checkpoint selection; no fresh generalization claim follows from this record.

## 4. Next safe conditions (all required before anything new)

1. **Metadata-only context/support audit FIRST** (the next proposed study): observable fields only — `start_time`/`end_time`, `robot_id`, `program_id`, `last_reset_time`, quarantine/censor flags, maintenance windows, eligibility predicate outcome; labels strictly post-hoc. Zero forwards by construction; no embedding inspection during the audit. Publish a control-support table (healthy-window density, HIGH-DRIFT overlap, reference depth) before any window/reference lock.
2. **Separate prospective amendment AFTER the audit** (separate owner authorization): it must prospectively specify endpoints (old/new/future performance, forgetting, alarms, cost), calibration-gated operating points (never ROC- or outcome-tuned), update target (frozen encoder; bounded head/replay adaptation only — the T7 trigger stays `NOT_TRIGGERED`), retention probes, matched healthy/ordinary-transition controls, identical-support event coverage, a forward/compute cap, `INFEASIBLE` / `UNSUPPORTED` / `NO-CALIBRATION` fail-closes retained verbatim, and an owner stop if movement remains drift-dominated without specificity.
3. **Separate execution authorization** before any new embedding outcomes or inference. A cheap negative (no supportable healthy-control design exists) is itself a valid answer — Phase C then stays held with the reason recorded, exactly as criterion `:143` contemplates.
4. **Forbidden regardless:** silent protocol / T7-trigger / record changes, retuned outcome-based window parameters, invented numerical gates, fake complete executable protocols, encoder retraining, a new corpus, fresh Confirmation access, and Phase C execution or release without the owner releasing the amended design.

## 5. Preserved records and candidate freeze

Preserved without change: locked protocol `docs/sprint24-geometry-time-diagnostic-v1.md` (SHA-256 `10c53edb30b0eaa257744bd5123f4fb6cbddb1b6e33f65b3afd8b8a005293f20`); G2 worker evidence `artifacts/sprint-24/task-7-g2.md`; derived report `artifacts/sprint-24/evidence/s24-7-g2/derived-no-calibration-report.md`; final review `artifacts/sprint-24/reviews/task-7-g2-r2-evidence.md` (PASS, 0 findings); final checkpoint `artifacts/sprint-24/checkpoints/task-7-g2-final-cp01.md` (checkpoint `1b53330bb74824085d6b4c40edf0a9bdc7fd829a`, base `24e8d6bd82e83e23f708514eb5b17539951efc0e`, frozen snapshot `04a1b193d14931c11942d85c5eb7200c12a20bb13338a5d6db56067bb35d246c`); Task 7 records with `NOT_TRIGGERED`; G1 contract and metadata-only scenario runlog; prior handoff `artifacts/sprint-24/handoffs/S24-7-G3-A01.md` (preserved, not rewritten).

Candidate freeze for `S24-7-G3-C1` R1 (frozen at `ready_for_review`; no further source edits pending assessment):

- This decision record: `docs/sprint24-geometry-time-decision-g3.md` (SHA-256 recorded in the worker record).
- Options brief: `docs/sprint24-geometry-time-research-direction.md` (SHA-256 recorded in the worker record; §7 now points here).
- Worker record: `artifacts/sprint-24/task-7-g3.md`.
- Base commit: `24e8d6bd82e83e23f708514eb5b17539951efc0e`; worktree HEAD at freeze: `1b53330bb74824085d6b4c40edf0a9bdc7fd829a`.
- No commit or push authorized or attempted for G3; Main owns plan changes and the assigned final review plus any separate Bronze checkpoint scope.
