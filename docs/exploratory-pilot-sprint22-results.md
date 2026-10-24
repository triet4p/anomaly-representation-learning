# Sprint 22 Exploratory Pilot Results: Stopped and Full-Roster Uncomputable

**Disposition: the one authorized invocation stopped at its first scientific support failure. The empirical H=32 score, variability, and dependence objective was not achieved.** The runner exited `0` because it successfully recorded `STOPPED_SCIENTIFIC_SUPPORT_FAIL`; this is not an empirical pass. The accepted disposable proof and accepted execution-evidence review validate their respective procedures and records, not the study's empirical objective.

This is a new Sprint 22 results report. The Sprint 21 report at [`exploratory-pilot-results.md`](exploratory-pilot-results.md) and all Sprint 21 evidence remain unchanged and immutable.

## Frozen study and provenance

The frozen protocol defined a research-only, simulator-conditional study of 32 independent whole histories: `H-S22-PILOT-01..32`, data seeds `922000..922031`. Scorer seeds `171701`, `171702`, and `171703` are repeated scorer variants, not additional histories. No contacted Sprint 21 coordinate was reused. The binding's freshness check covers its enumerated static provenance catalog only; no generated-history or Sealed-root scan was performed, so it is not a claim of global freshness beyond that catalog.

| Frozen item | Identity |
|---|---|
| Historical source | `8c15f0204a3e495569b7f143dc109943e8b808de` |
| Runner | `sprint22-pilot-runner-v2`, commit `41f3ec2e0e5e2d886f2ad01fae139d00d29ffbf2`; SHA-256 `eb097111296bc63fa7779cea518f9cfd5da84a5bf9531906ab15a5b66acf3444` |
| Protocol | `sprint22-exploratory-pilot-v1`; SHA-256 `168536e482ff8d926b2f094bec54222ac68b46394e999b6b1327d65542ae6dc5` |
| Binding | Commit `dec8597483fd871a2f68996757610198bdbba8fe`; canonical SHA-256 `dd947cc16189a6681dad716de32e8fdc2a8ca4165f53052747ac9ddeff14e9b4` |
| One-shot output root | `/tmp/sprint22-exploratory-pilot-v1` |
| Guard-observed runtime/device | Python `3.12.13`, Torch `2.14.0+cu130`, CUDA available, NVIDIA GeForce RTX 4060 Ti; 16,380 MiB device total and 15,947 MiB Torch-visible |

The final no-contact guard passed before the released invocation. It did not generate histories, run preflight, deserialize checkpoints, score, or touch the output root. It verified the frozen runner, protocol, binding, source closure, runtime/device identity, and original checkpoint hashes. The actual run then used the exact Main-released binding once. Its evidence and identities are recorded in the [Sprint 22 plan](sprint-plans/sprint-22.md), [frozen protocol](../experiments/sprint22-exploratory-pilot-v1.md), [binding](../experiments/sprint22-pilot-binding-v1.json), and [Task 3 evidence](../artifacts/sprint-22/task-3.md).

## One-shot outcome and roster accounting

The run started at `2026-10-01T06:51:17.528114+00:00`, ended at `2026-10-01T06:58:46Z`, and took 449.78 seconds by the terminal capture. The process exit code was `0`; runner stdout contains the binding digest, `history_count: 32`, and status `STOPPED_SCIENTIFIC_SUPPORT_FAIL`. Runner stderr is empty. The exit code means the runner recorded its stop state successfully, not that all 32 histories qualified or that the empirical analysis passed.

| Roster coordinates | Data seeds | Attempted / materialized | Hard support | Allocator controls returned | Scored |
|---|---:|---:|---|---:|---|
| `H-S22-PILOT-01..11` | `922000..922010` | 11 / 11 | PASS | 73, 57, 68, 59, 59, 55, 63, 64, 70, 53, 50 | 0 |
| `H-S22-PILOT-12` | `922011` | 1 / 1 | First failure; stop | 0 | 0 |
| `H-S22-PILOT-13..32` | `922012..922031` | 0 / 0 | NOT ATTEMPTED; not classified as failures | — | 0 |
| **Declared roster** | **`922000..922031`** | **12 / 32 attempted and materialized** | **11 PASS; 1 first failure; 20 unattempted** | **50–73 for the 11 successful allocations; no controls returned for H12** | **0 / 32** |

For the first 11 histories, the runner retained each complete qualifying allocator result; it did not truncate controls to 48. The shown control counts are descriptive observations for this stopped prefix, not estimates of the complete roster's support distribution or qualification rate.

`H-S22-PILOT-12` materialized 5,195 files. Its recorded pre-allocation support included 135 eligible positive windows (`P=61`, `W=52`, `A=22`), 63 eligible control windows, 1,406 eligible robot-days, and 36 anchor rejections with reason `reset-horizon`. The eligible control pool met the numeric 48-control reserve and the separate hard control floor was 25; all six diagnostic design-margin flags were true. Nevertheless, the exact positive-anchor allocation returned `InfeasibleCandidate` with reason `quota-shortfall`, returned zero controls, and the solver reported infeasibility for quotas `P/P1<12`, `P/P2<12`, `W/W1<12`, `W/W2<12`, `A/A1<8`, and `A/A2<8` (135 variables, 58 constraints). The runner's recorded hard-failure keys are `exact_positive_quota`, `allocator_seed_matches_history`, `allocator_result_minimum_48_controls_failed`, and `allocator_failure:quota-shortfall`.

This is an observed upstream allocation infeasibility, not evidence that the minimum-48 consumer rejected a successful 48-or-more-control allocation. The consumer was not reached; no checkpoint restore or scoring started. The bounded solver record establishes its reported infeasible status and problem size, but does not identify which individual constraints made the problem infeasible. In particular, the separate 50-day spacing used in the disposable fixture correction is not identified as a cause of this production history's allocation failure.

The append-only [pilot ledger](../artifacts/sprint-22/pilot-v1/pilot-ledger.jsonl) records the first stop at H12 and lists H13–H32 as unattempted. The [attempt record](../artifacts/sprint-22/pilot-v1/attempt.json) records the full bound roster and stop status; per-history support evidence for H12 is in [`H-S22-PILOT-12.json`](../artifacts/sprint-22/pilot-v1/history-evidence/H-S22-PILOT-12.json).

## Disposable contract proof versus actual study result

Task 2's independently reviewed `PASS_DISPOSABLE_ONLY` was a separate fixture proof: the genuine allocator rejected 47 eligible controls and retained all 48 or all 49 at the consumer boundary, and deterministic toy inputs passed the three frozen scorer paths and their declared checks. That proves the tested minimum-48 consumer behavior and fixture scorer behavior. It is not evidence that 32 generated whole histories qualify, and it contributes no whole-history AUROC or uncertainty estimate.

The actual H12 allocation stopped before the consumer boundary. Its `quota-shortfall` therefore does not demonstrate an exact-48 predicate mismatch, nor does it negate the separately observed 48/49 disposable proof. Conversely, the disposable result does not make H12 a qualification pass. The [accepted Task 2 evidence](../artifacts/sprint-22/task-2.md) and the [actual-run evidence](../artifacts/sprint-22/task-4.md) document these different scopes.

## Scores and uncertainty: all full-roster quantities remain null

No checkpoint was restored and no history was scored. The authoritative, valid [pilot `summary.json`](../artifacts/sprint-22/pilot-v1/summary.json) records each of the three scorer seeds as `UNCOMPUTABLE_FULL_ROSTER`. For every model seed, both primary branches (`S_pred` and `S_pop`) have null per-history values, full-roster macros, unbiased sample variances, and bootstrap LCBs. Secondary P/W/A branch macros are also null, and the specified 2,000-replicate paired whole-history bootstrap (seed `20260202`) has `computed: false`.

| Scorer seed | Full-roster status | `S_pred` and `S_pop` |
|---|---|---|
| `171701` | `UNCOMPUTABLE_FULL_ROSTER` | Per-history values, macro, sample variance, and bootstrap LCB are null |
| `171702` | `UNCOMPUTABLE_FULL_ROSTER` | Per-history values, macro, sample variance, and bootstrap LCB are null |
| `171703` | `UNCOMPUTABLE_FULL_ROSTER` | Per-history values, macro, sample variance, and bootstrap LCB are null |

No subset aggregate, pooled-seed estimate, failed-coordinate score, or imputed value was computed. No imputation or complete-case endpoint is permitted under the frozen protocol. The three scorer variants do not increase the independent-history count.

The protocol required all 32 declared histories to qualify before any model restore or scoring, and required every full-roster endpoint to be null if a history failed. The first-failure stop at H12 consequently prevented scoring even the 11 earlier histories that passed support. With no history scores, cross-history score variance, score/support dependence, branch differences, and score strata by support cannot be estimated. The stopped prefix also cannot estimate qualification frequency or failure rate: the first 11 successes were a sequential prefix, and the remaining 20 coordinates are unknown, not additional failures. In particular, the single observed failure among the 32 bound coordinates is not an unbiased 1/32 failure-frequency estimate.

H=32 was a fixed prospective sample size, not a miss allowance, precision guarantee, or license to continue after a failure. This report chooses no precision requirement, tolerance, miss allowance, positive acceptance gate, or revised endpoint. Sprint 20 remains `NOT_READY`; Sprint 18 remains blocked with its 48-control construction reserve and no-Cycle-4 status. Gate V2, hard/mix rules, and the prohibition on training, refitting, recalibration, Confirmation, or Sealed-root access are unchanged.

## Evidence custody and resource limits

The bounded evidence is under [`artifacts/sprint-22/pilot-v1/`](../artifacts/sprint-22/pilot-v1/). The transfer manifest SHA-256 is `6a733c9130a37ec5d2b0c2d399e13c5ff1ba8dbfda245f06319688aa776812ce`; the remote-backed evidence bytes and SHA-256 values were matched against their local copies. The transferred remote-backed pilot evidence totals 5,810,154 bytes. The local `pilot-terminal.txt` capture (477 bytes) and transfer manifest are separate custody records, not part of that byte total. Raw generated histories and checkpoints were not transferred; they remain on the remote roots recorded in Task 4.

The valid `summary.json`, attempt record, append-only ledger, per-history records, and clean runner stdout/empty stderr support the stop disposition. The derived `aggregate-endpoints.json` is preserved byte-for-byte but has a literal trailing backslash-n extraction defect; it is not parser authority and was not used to interpret endpoint values. Use `summary.json` for the endpoint disposition. The outer terminal-display wrapper emitted two `printf: --: invalid option` diagnostics after the runner completed because its labels began with `---`. Those warnings appear in the local terminal transcript, not in runner stderr, and do not change the runner's exit code or scientific stop. No post-contact correction or rerun occurred.

The only real-run resource measure is the 449.78-second wall time. The guard-time device total/visible memory above is hardware inventory, not measured utilization or peak VRAM. Real-run GPU memory, host RAM, and utilization were not measured. Task 2's 15.22 MiB peak allocated / 28.0 MiB peak reserved and 0.97256-second fixture measurement describe only a tiny disposable mathematical fixture, not a real-history resource envelope.

## Review record and disclosed process deviations

The accepted evidence gates are recorded in the [Sprint 22 plan](sprint-plans/sprint-22.md): Task 1 corrective review `S22Task1ReviewR2`, Task 2 review `S22Task2Review`, Task 3 corrected pre-contact review `S22Task3PrecontactR2`, and Task 4 evidence review `S22Task4Review` each passed with zero actionable findings. The Task 4 review accepts preserved execution evidence; it does not declare empirical success.

Two earlier review deviations are disclosed in the plan and evidence records and are excluded from accepted proof: the initial Task 1 reviewer performed disposable in-memory checks despite a static-only brief, and the initial Task 3 pre-contact reviewer reported graph status/presence despite the explicit no-graph rule. Corrected reviews were used for the accepted gates; the unauthorized Task 1 runtime claims and Task 3 graph-status observation are not evidence for this report. No graph action was taken for this task.

## Future evidence requires a separate decision

The consumed Sprint 22 authority does not permit replay, resume, finish, replacement, post-contact repair, or scoring for any H-S22-PILOT-01..32 coordinate, including the 20 unattempted histories. It grants no new run, seed, sample count, quota change, or acceptance rule. Sprint 21's full roster remains immutable as well.

If the research question is whether histories can be constructed, a separately authorized prospective study could define construction feasibility and its failure handling as its own estimand; an infeasible history must not be relabeled as an invalid history AUROC. If the question remains score variability or score/support dependence, a user must first make a separate decision on prospective handling of qualification failure and the intended estimand. Either path requires genuinely fresh, disjoint research roles, a separately versioned protocol/binding, independent review before contact, and separate explicit execution authority. This is a prerequisite description, not permission to run, a selected design, or a changed endpoint or gate.

## Evidence references

- [Sprint 22 plan, accepted gates, and frozen task boundaries](sprint-plans/sprint-22.md)
- [Sprint 22 frozen exploratory pilot protocol](../experiments/sprint22-exploratory-pilot-v1.md) and [binding](../experiments/sprint22-pilot-binding-v1.json)
- [Task 1 evidence and disclosed initial-review scope deviation](../artifacts/sprint-22/task-1.md)
- [Task 2 disposable-only proof evidence](../artifacts/sprint-22/task-2.md)
- [Task 3 provenance and no-contact evidence](../artifacts/sprint-22/task-3.md)
- [Task 4 one-shot execution evidence](../artifacts/sprint-22/task-4.md)
- [Attempt record](../artifacts/sprint-22/pilot-v1/attempt.json), [append-only ledger](../artifacts/sprint-22/pilot-v1/pilot-ledger.jsonl), [authoritative summary](../artifacts/sprint-22/pilot-v1/summary.json), [transfer manifest](../artifacts/sprint-22/pilot-v1/transfer-manifest.json), and [runner stdout](../artifacts/sprint-22/pilot-v1/stdout.log)
- [Sprint 21 results report](exploratory-pilot-results.md) (unchanged)
