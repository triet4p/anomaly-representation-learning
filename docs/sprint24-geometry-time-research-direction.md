# Sprint 24 geometry–time research direction — decision brief (G3 synthesis only; owner decision ACCEPTED — see §8)

- Scope: **SYNTHESIS / DESIGN ONLY**. No new analyses, runs, forwards, training, source generation, Confirmation access, or Phase C execution. Synthesis of existing verified outputs only.
- Task: `S24-7-G3`, attempts `S24-7-G3-A01` (options brief + worker record) → `S24-7-G3-A02` (accepted-decision finalization), worker chain `S24G3GoldA01` → `S24G3GoldA02` (owner-selected Gold, quota `unknown`, full route).
- Criterion: `docs/sprint-plans/sprint-24.md:143`.
- Status: **owner decision ACCEPTED — Option B (hold Phase C; bounded context-/time-resolution measurement research first). Accepted record: `docs/sprint24-geometry-time-decision-g3.md`.** Main owns the actual owner ask and any plan changes (`docs/sprint-plans/sprint-24.md:144` edit trail).
- Preserved without change: locked protocol `docs/sprint24-geometry-time-diagnostic-v1.md`, Task 7 records and `NOT_TRIGGERED` trigger disposition (`S18-UNRESOLVED`, Confirmation custody unchanged), G1/G2 worker evidence, R2 evidence PASS, and final checkpoint records. This brief designs no silent protocol edit, no retuned window parameter, and no new numerical gate.

## Sources (exact, no new inspection)

- Locked diagnostic protocol `docs/sprint24-geometry-time-diagnostic-v1.md` (SHA-256 `10c53edb30b0eaa257744bd5123f4fb6cbddb1b6e33f65b3afd8b8a005293f20`).
- Accepted G2 worker evidence `artifacts/sprint-24/task-7-g2.md` (R2-corrected disposition header).
- Derived existing-outputs-only report `artifacts/sprint-24/evidence/s24-7-g2/derived-no-calibration-report.md`.
- Final review `artifacts/sprint-24/reviews/task-7-g2-r2-evidence.md` (cycle `S24-7-G2-C1`, R2, verdict PASS, `advance: true`, 0 actionable findings).
- Final checkpoint `artifacts/sprint-24/checkpoints/task-7-g2-final-cp01.md` (checkpoint commit `1b53330bb74824085d6b4c40edf0a9bdc7fd829a`, parent/base `24e8d6bd82e83e23f708514eb5b17539951efc0e`, frozen snapshot `04a1b193d14931c11942d85c5eb7200c12a20bb13338a5d6db56067bb35d246c`).
- Completed file-head comparison `artifacts/sprint-24/task-7.md` (108/108 reused fits, 828/828 score operations, `NOT_TRIGGERED`).
- G1 contract record `artifacts/sprint-24/task-7-g1.md` and metadata-only scenario log `artifacts/sprint-24/evidence/s24-7-g1-a01/s24-g1-scenario-runlog.md`.
- Local reviewed table copies `artifacts/sprint-24/evidence/s24-7-g2/tables/` (hashes verified in R2: window-statistics `054ea5cb…`, controls `f7d17474…`, event-mapping `e789778d…`, null-calibration `caf1f180…`, comparator-events `eba2c4e4…`, false-alarms `4d5a75cc…`, bootstrap `0d7cd09d…`).

## 1. What the completed file-head evidence established (Task 7, unchanged)

Task 7 is complete and is **not re-decided** by geometry. Its descriptive file-level outcomes stand as the label-efficiency answer under the frozen budgets:

- 108/108 fitted cells reused and verified (no new fits); 828/828 planned score operations completed; 3,323,754 per-file score rows.
- Mean `M1` file PR-AUC `0.24181` (< 0.70) and mean `M3 − M1` `−0.03298` (not > 0.05), hence trigger disposition **`NOT_TRIGGERED`** — `M4`/T10 were not executed.
- Development B3 PR-AUC (illustrative, descriptive): `M0a` 0.2328, `M0b` 0.2436, `M0c` 0.2446, `M1` 0.2436, `M2` 0.2245, `M3` 0.2135; paired history-cluster contrasts descriptive with `n = 4` (e.g. `M1−M2` B3 `+0.0191 [+0.0019, +0.0303]`, `M1−M3` B3 `+0.0301 [+0.0191, +0.0410]`).
- File-level PR-AUC and the preserved above-threshold file-event summaries are **file-head** evidence. Per the locked protocol (§0), they are context only: they are not evidence of coherent collective distribution/cluster movement, and geometry does not retroactively change them.

## 2. What the geometry–time diagnostic actually showed (G2 EXECUTION3, R2-corrected)

All numbers below are existing verified outputs; G3 computed nothing new.

- Full run covered all 4 Development histories × 3 B0 seeds (`181801/181802/181803`); deterministic CUDA; cumulative 26,431 forwards ≤ 41,000 cap; trainable params 0; 372 window rows (369 SUPPORTED + 3 rows for the single DEV-01 UNSUPPORTED window-1, `n = 28 < 30`) and 2,850 event rows (950 × 3 seeds).
- Descriptive original-128-d movement (per history across 3 seeds, SUPPORTED windows):
  - DEV-01 (n=90): MMD² med 0.00653 / IQR 0.00308 / max 0.021909; loc med 1.173; logTR med +0.001; KS-med med 0.0983; NN-ref med 0.912 vs split-half null 0.704.
  - DEV-02 (n=93): MMD² med 0.00596 / IQR 0.00212 / max 0.030677; loc med 1.045; logTR med −0.010; KS-med med 0.0951; NN-ref med 0.915 vs 0.706.
  - DEV-03 (n=93): MMD² med 0.007951 / IQR 0.004389 / max 0.034184; loc med 1.450; logTR med 0.000; KS-med med 0.1046; NN-ref med 0.925 vs 0.704.
  - DEV-04 (n=93): MMD² med 0.005958 / IQR 0.002956 / max 0.013298; loc med 1.058; logTR med −0.006; KS-med med 0.0936; NN-ref med 0.906 vs 0.707.
- Nulls: all 12 split-half nulls finite + non-degenerate (medians ≈ 0.0015–0.0018; q95 ≈ 0.0022–0.0027). Raw crossings: **369/369 SUPPORTED window-seeds cross q95 (100%)** — preserved as **descriptive movement only, not detections**.
- Corrected disposition: **NO-CALIBRATION** (owner-accepted negative outcome). Healthy-control support is exactly **0/369 SUPPORTED windows** (verified from `tables/controls.jsonl`: every window carries ≥ 1 abnormal file and covers ≥ 3 events), so protocol §10.3 condition (b) cannot hold. The executed run's raw `CALIBRATED` label was vacuous (`healthy_total == 0` → or-pass); the R2 candidate runner requires `healthy_total > 0` and yields `NO-CALIBRATION`. Executed bytes were not overwritten; the derived report carries the correction with zero new forwards.
- Consequently **unavailable per protocol**: calibrated MMD event recall (raw 567/950 per seed, identical across seeds by shared-coverage construction), detection delays (recalled-only medians ≈ 6.9–7.7 days, censored), and false-alarm episodes (0 raw episodes) — none is a calibrated detection claim.
- Confounding (existing tables only): HIGH-DRIFT top-decile subgroup sits systematically higher on every history and seed (seed-181801 HI vs non-HI medians: DEV-01 0.010669 vs 0.006650; DEV-02 0.008608 vs 0.005751; DEV-03 0.014052 vs 0.007193; DEV-04 0.007939 vs 0.005823; replicated 1.1x–2.05x across seeds). Reviewer-independent association: `r(MMD², TV drift) = +0.78` vs `r(MMD², n_abnormal) = −0.40`. Movement tracks operational mix drift, not anomaly severity.
- Comparators (retained, explicitly **not** identical-support): MMD mapping covers all 950 manifest failure events (zero-cover 102/89/102/90 counted, censored from delay); `_comparator_events` additionally requires robot-matched in-span file support inside merged degradation-to-maintenance spans (35/35/34/33 events/history, 137 total). Retained frozen-S18-q95 comparator recalls (M0a 0.176–0.485, M0b 0.171–0.364, M0c 0.176–0.455) are therefore **not directly comparable** to MMD coverage numbers; no rescoring was performed and none is invented here.
- Healthy exposure from accepted manifests (existing metadata only): 8.5 / 9.4 / 7.8 / 9.6 file-days (total 35.3 over 1226/1353/1118/1379 eligible eval files). False-alarm rate per 100 healthy-exposure days is **UNDEFINED** under NO-CALIBRATION (0 raw episodes / 35.3 file-days reported for completeness only, not a rate).
- Built-in limitations carried forward: history-cluster bootstrap descriptive given `n = 4`; inherited temporal exposure (encoder/bank/thresholds saw Fit/Calibration-healthy rows plus Sprint 18 forward exposure) → sequential label adaptation with prior unlabeled exposure, never strictly prospective end-to-end (protocol §2); post-run live model/bank digests unrecoverable (asserted in memory only, honestly disclosed, never retroactively recreated).

## 3. What geometry does and does not show (inference cap)

- It **does** show coherent descriptive distribution movement: evaluation windows sit above their own past-era reference (MMD² medians ≈ 0.006–0.008 vs null q95 ≈ 0.0022–0.0027; NN-ref ≈ 0.91 vs ≈ 0.70 null; consistent across 3 seeds and 4 histories).
- It **does not** show event-detection utility: with zero healthy controls, there is no calibrated operating point, no calibrated recall/delay, and no calibrated false-alarm rate. Raw 100% crossing is movement above a past-era null, not anomaly specificity.
- The strong mix-drift association is **confounding evidence, not causal proof**. It warns that ordinary-regime operational change explains much of the movement, but with `n = 4` histories it cannot prove the encoder is causally driven only by drift.
- Therefore the diagnostic **cannot conclude the encoder is useless, and cannot conclude anomaly clusters do not shift**. Both directions remain open: the movement is real but unspecific, and the anomaly-specific component is unidentified under the executed window design.
- File-head and geometry answers stay separated: Task 7 says frozen-head file-level label efficiency is weak and the fine-tuning trigger did not fire; geometry says collective movement exists but is uncalibrated and drift-confounded. Neither overturns the other.

## 4. Could meaningful normal-transition / healthy controls be supported from existing metadata?

Question asked: without inspecting new embedding outcomes, does existing metadata already support a healthy-control design (normal transitions with zero abnormal content)?

- Under the **executed** design (fleet-wide 14-day trailing windows, 7-day stride, `n ≥ 30`, inherited eligibility predicate), the answer from verified outputs is **no**: 0/369 SUPPORTED windows are healthy. Every post-`c1` fleet window contains abnormal files and covers events. This is a structural property of that window design on Development, not a runner bug.
- Whether a **different** observable grouping could yield meaningful healthy controls (e.g. narrower widths, per-robot or per-program context matching, maintenance-bounded spans, pre-`c1`-adjacent normal transitions) is **UNKNOWN** from passed records. Feasibility is unverified because:
  - The only support evidence on record is the G1 metadata-only fixture scenario (fleet-wide 14-day windows held support on the audit fixture — run1 min/med/max 28/77/111 with 1 UNSUPPORTED, run3 31/31 SUPPORTED — while per-robot 7-day windows collapsed 1457/2007 below minimum on the audit fixture) plus the accepted-data window projection (31 windows/history, UNSUPPORTED 1/0/0/0) — neither audits healthy-control density under an alternative grouping, so the fleet-wide choice was support-forced on available evidence, not outcome-driven.
  - Accepted Development manifests and decile edges exist (Task 2 inventory, Task 7 lock) but have **not** been audited for healthy-control density under any alternative observable grouping. No such audit was in G1/G2 scope, and G3 performed none.
  - Any feasibility claim would require a **metadata-only context/support audit** (observable fields only — `start_time`/`end_time`, `robot_id`, `program_id`, `last_reset_time`, quarantine/censor flags, maintenance windows, eligibility predicate outcome; labels strictly post-hoc) before any window/reference lock. That audit has not been done; its outcome cannot be fabricated here.
- Hence: healthy-control feasibility under a new grouping is **UNKNOWN — pending metadata audit**. The options below sequence that audit before any lock, with no embedding inspection during the audit. The audit itself is the NEXT proposed study (zero forwards), not work performed in G3.

## 5. Bounded concrete direction alternatives

Both alternatives preserve the locked protocol, T7 records, and the T7 trigger; neither silently retunes the executed window parameter or invents a numerical gate. Exact scientific choices (widths, strides, groupings, thresholds, update targets) are **frozen only after the metadata audit via a versioned amendment** — they are deliberately not fixed in this brief (no fake complete executable protocol).

### Option A — Continue C with amendment (bounded sequential study)

- Prospective scope (to be locked post-audit, not run now): a versioned amendment (`S24-GEO-TIME-v2` or a Phase C sequential amendment) that prospectively specifies endpoints (old/new/future performance, forgetting, alarms, cost), operating points (calibration-gated per protocol §10, never ROC- or outcome-tuned), update target (frozen encoder; bounded head/replay adaptation only — no encoder fine-tuning beyond the T7 trigger, which stays `NOT_TRIGGERED`), retention probes, and matched healthy/ordinary-transition controls before any run.
- Preconditions before lock: observable context-grouping / causal-reference / control-support audit (metadata only, no embedding outcomes); common event-coverage rule reconciled prospectively so comparators share identical support; retention outcomes defined alongside detection outcomes.
- Smallest methods matrix (indicative, locked later): no-update vs bounded-replay vs cumulative-label references on inherited chronological views with frozen experience schedules — the minimal set that separates acquisition from forgetting. No architecture search, no budget/epoch reduction, no evaluation-driven expansion.
- Exposure limits and stop criteria (indicative, locked later): a prospectively locked forward/compute cap in the style of the G2 §6 cap; inherited-exposure disclosure carried forward; fail-closed `INFEASIBLE` on support shortfall or manifest/cache-identity mismatch; `NO-CALIBRATION` (not a fallback threshold) if healthy-control support fails again; owner stop if drift-confound dominance reappears without specificity.
- Cost/risk: fastest path to a sequential answer, but it bets a full sequential comparison on a measurement foundation whose specificity is currently unproven (zero healthy controls, drift confound). Risk of a second negative outcome at higher compute cost.

### Option B — Hold C; run bounded context-/time-resolution measurement research first (recommended)

- Prospective scope (to be locked post-audit, not run now): a measurement-first study that asks whether any observable grouping yields **specific, calibrated** movement before any sequential adaptation is attempted. Frozen encoder, descriptive statistics only, no model updates in this branch.
- Preconditions before lock (same audit as A, measurement-flavored): metadata-only audit of candidate observable groupings (fleet vs robot/program-matched; calendar vs maintenance-aware spans within the §3.2 boundary; candidate widths/strides) for healthy-control density and reference support; control-support table (healthy windows, HIGH-DRIFT overlap, reference depth) published before the window lock.
- Smallest methods matrix (indicative, locked later): frozen descriptive statistics (location/dispersion/MMD²/KS/neighborhood-stability per protocol §7) across the smallest set of groupings the audit supports — not a model-comparison matrix. Sequential-update arms belong to a later amendment only if measurement establishes specificity.
- Common outcomes: identical-support event coverage plus retention-probe definitions carried prospectively (so a later sequential study inherits them); per-100-healthy-exposure normalization defined only where the calibration gate passes.
- Exposure limits and stop criteria (indicative, locked later): metadata audit costs zero forwards by construction; any follow-on frozen-forward run gets its own prospectively locked cap; `INFEASIBLE` / `UNSUPPORTED` / `NO-CALIBRATION` dispositions retained verbatim; stop and return to the owner if no grouping yields healthy-control support or if movement remains drift-dominated.
- Cost/risk: slower path to a sequential answer, but bounds compute, avoids spending a sequential comparison on an uncalibrated signal, and directly attacks the diagnosed weakness (specificity/controls) rather than routing around it.

## 6. Evidence-based recommendation (conservative)

**Recommend Option B — hold Phase C and do bounded measurement research first.**

- The executed diagnostic answered its sensitivity question (movement exists) but failed its specificity question (zero healthy controls; drift association +0.78 vs anomaly association −0.40). A sequential study now would inherit an uncalibrated operating point and incomparable event denominators — the exact weaknesses R2 documented.
- The file-head record gives no counter-pressure to rush: `NOT_TRIGGERED` means no fine-tuning rescue is owed, and M1≈no-label-score performance means sequential adaptation has no demonstrated headroom to protect.
- Option B is the only alternative that sequences the missing prerequisite (metadata audit → control support → calibrated specificity) before spending sequential-comparison compute. If the audit finds no supportable healthy-control design, that negative result is itself a valid, cheap answer and Phase C stays held with the reason recorded — exactly what criterion `sprint-24.md:143` contemplates.
- This recommendation changes nothing already frozen: protocol v1, T7 trigger/records, G1/G2 evidence, review PASS, and checkpoint records stand regardless of which option the owner selects.

## 7. Owner decision requested — SUPERSEDED by §8 (accepted Option B)

- Original options (kept for lineage): **Option A** — continue C with amendment (bounded sequential study; preconditions and prospective lock as above). **Option B (recommended, now ACCEPTED)** — hold C; bounded context-/time-resolution measurement research first (metadata audit → control support → calibrated specificity; sequential arms only via later amendment).
- **Owner explicitly chose Option B.** The bounded accepted-decision record is `docs/sprint24-geometry-time-decision-g3.md`; this brief is now the options lineage for that decision. No Phase C release is authorized by this brief or the accepted record.

## 8. Accepted decision (current; see `docs/sprint24-geometry-time-decision-g3.md` for the full freeze)

- **Choice:** Option B — `Giữ C, nghiên cứu phép đo trước`. Phase C stays held with the reason recorded in `docs/sprint24-geometry-time-decision-g3.md` §§1–2; Task 7-G3 acceptance is decided there, not by the recommendation in §6 above.
- **Order:** metadata-only feasibility audit (NEXT proposed study, zero forwards) → SEPARATE prospective amendment → SEPARATE execution authorization before any new geometry. The audit was not performed in G3.
- **Not authorized:** new study execution, encoder retraining, a new corpus, retrospective protocol changes, fresh Confirmation access or reuse as validation, and any Phase C release. Main owns plan changes and any future release.
