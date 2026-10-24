# Sprint 21 Exploratory Pilot Results: Stopped and Uncomputable

**Disposition: incomplete / full-roster uncomputable.** One authorized, bounded invocation ran once and preserved the frozen terminal state `STOPPED_SCIENTIFIC_SUPPORT_FAIL`. Exit code 0 means the stop state was recorded successfully; it is not scientific success, pilot completion, or a pass. No empirical performance or tolerance conclusion is available.

The independent Task 5 evidence review (`agent://S21Task5Review`) returned `PASS` with no actionable findings. That review accepts the evidence and stop record; it does not turn the attempted history into a successful pilot or waive the frozen stop. The Task 5 evidence record and bounded transferred outputs are linked below.

## Observed execution and frozen provenance

The independent Task 4 pre-contact review passed and provided the required gate, but did not itself authorize contact (`summary.json` preserves `contact_authorized_by_task4: false`). Main separately authorized the single Task 5 invocation. A fresh pre-contact dry-run guard immediately before contact reported `DRY_RUN_PASS_NO_SEED_EVALUATION`, exit 0 in 1.74 seconds; it did not generate data, run preflight, deserialize checkpoints, score, or touch the output root. The actual run was executed once on `trietlm@192.168.30.244` from `/tmp/sprint21-proof-hist-v1`.

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/sprint21-proof-hist-v1/src /home/trietlm/anomaly-representation-learning/.venv/bin/python /tmp/sprint21-proof-hist-v1/experiments/sprint21_pilot_runner.py --run --review-passed --worktree-root /tmp/sprint21-proof-hist-v1 --checkpoint-root /tmp/sprint17-task7-out/checkpoints --binding /tmp/sprint21-pilot-binding-v1-final-a36f1c9.json --expected-binding-sha256 297022f2d5e296628ede18a3796c823279256bf561e161d191041298e30dd3e1 --output-dir /tmp/sprint21-exploratory-pilot-v1
```

| Frozen item | Observed identity |
|---|---|
| Historical base | `8c15f0204a3e495569b7f143dc109943e8b808de` |
| Task 4 runner source | Commit `dbacea75a5759974679267d450fd9b661c242a6d`; runner SHA-256 `23ca3fa1a6ddf0690405c953bbbcfe64e4ad72d54163913fd65729d4e8f26195` |
| Protocol | SHA-256 `09c27bf1863787f3420588e4604501d9044641a2b45ccb936316d8c0e4304391` |
| Binding | Commit `a36f1c92a779073671a81918135b439706e61e15`; raw SHA-256 `0521edd558799138048a7748585010b7e7dfe93b549d5da2c4272ac28b39b45c`; canonical SHA-256 `297022f2d5e296628ede18a3796c823279256bf561e161d191041298e30dd3e1` |
| Bound history roster | `H-S21-PILOT-01..32`, data seeds `921000..921031` (`H=32` is the frozen roster size only, not a miss allowance) |
| Scorer seeds | `171701`, `171702`, `171703`; repeated scorer variants, not additional histories |
| Runtime | Python `3.12.13`, PyTorch `2.14.0+cu130`, CUDA available, NVIDIA GeForce RTX 4060 Ti |
| Checkpoint root | `/tmp/sprint17-task7-out/checkpoints/` |
| Checkpoint SHA-256 | `171701`: `45f9e151c5d3946804ea531eb2bcace26b1f62e634671f51b8741ac6b411a619`; `171702`: `64ed2cedec210e4692f60bb4dd3430a57cf94abfcd50551abd97c9265ea785f8`; `171703`: `9edb2122e357376a9ff1e065d73d5ab7704f8d2005991552332f1ced01b0e906` |

The process ran from `2026-09-30T07:12:54.276518+00:00` to `2026-09-30T07:13:27.627083+00:00`, recorded `35.08s`, and exited `0`. `run.stdout` records `STOPPED_SCIENTIFIC_SUPPORT_FAIL`; `run.stderr` is empty. The exact invocation and provenance are also preserved in the [Task 5 evidence record](../artifacts/sprint-21/task-5.md#L37-L49).

## Evidence custody and attempt accounting

The remote output root was `/tmp/sprint21-exploratory-pilot-v1/`; the sole materialized raw history remains at `/tmp/sprint21-exploratory-pilot-v1/histories/H-S21-PILOT-01`. The 31 other bound history roots were not materialized. Checkpoints remain under `/tmp/sprint17-task7-out/checkpoints/`. No raw history files or checkpoints were transferred to the repository.

The bounded review evidence is under [`artifacts/sprint-21/pilot-v1/`](../artifacts/sprint-21/pilot-v1/): `attempt.json`, `summary.json`, `pilot-ledger.jsonl`, `history-evidence/H-S21-PILOT-01.json`, `logs/`, and `checksums.sha256`. Exactly 12 review-sized files totaling 990,108 bytes were transferred; the independent Task 5 review recomputed their SHA-256 values and found all matched the manifest. No scorer was restored, no score files or score directory were created, and no score/window-branch records exist.

| Roster state | Count and identity |
|---|---|
| Bound | 32 histories, `H-S21-PILOT-01..32` |
| Attempted and materialized | 1: `H-S21-PILOT-01`, data seed `921000` |
| Scored | 0 |
| Unattempted after the frozen stop | 31: `H-S21-PILOT-02..32`, data seeds `921001..921031` |

The append-only ledger records `run_started`, H01 materialization start and completion, then `scientific_support_stop`; it records no retry, resume, replacement, or scoring. H01 materialized 5,176 files with manifest SHA-256 `15c40f15e1b395ba4095b1e6a07bad6c391778b46de3b69491517ec52b986c0d`. The raw history report records hard status `FAIL` and the exact failure key `exact_existing_48_control_construction_quota`.

## Interpreting the sole hard failure

H01 had 61 eligible and selected control windows against the frozen runner's exact-48 predicate. Its observed support was `P=63`, `W=39`, `A=37`, 139 eligible positives, and 1,313 eligible robot-days. Every other reported hard flag passed for this history, and all six reported Design-margin flags passed (`A≥10`, `P≥13`, `W≥13`, positive total `≥38`, controls `≥32`, robot-days `≥188`). These are single-history support observations, not rates or cross-history evidence.

The exact failure key and `FAIL` status remain unchanged in the raw ledger and history evidence. Separately, the independent Task 5 review inspected the pinned static contracts: the allocator treats `quota.controls=48` as a minimum (it rejects only when fewer than 48 controls are available) and returns all qualifying controls, while the frozen runner requires exact equality to 48. Thus 61 controls satisfy the allocator's minimum but violate the runner's exact-equality predicate. This is a runner/allocator contract mismatch, not evidence that H01 lacked 48 controls or that its synthetic history had deficient control support. This interpretation does **not** relabel the frozen hard failure or stop as a pass; no truncation to 48, requalification, ledger rewrite, metric recomputation, or rescue was performed. See the reviewed source locations in [`balanced.py`](../src/synth/balanced.py#L718-L780) and [`sprint21_pilot_runner.py`](../experiments/sprint21_pilot_runner.py#L1038-L1054).

## Scientific results: uncomputable, not favorable or unfavorable

For each of the three scorer seeds, the frozen summary records `UNCOMPUTABLE_FULL_ROSTER`. For both independent branches (`S_pred` and `S_pop`), every per-history value is null; primary and secondary macros, sample variances, and bootstrap `LCB95` values are null. The bootstrap was not computed. No score was produced for H01 or any other history.

| Scorer seed | Full-roster status | `S_pred` and `S_pop` endpoints / uncertainty |
|---|---|---|
| `171701` | `UNCOMPUTABLE_FULL_ROSTER` | Per-history values, primary and secondary macros, sample variances, and bootstrap `LCB95` all null; bootstrap not computed |
| `171702` | `UNCOMPUTABLE_FULL_ROSTER` | Per-history values, primary and secondary macros, sample variances, and bootstrap `LCB95` all null; bootstrap not computed |
| `171703` | `UNCOMPUTABLE_FULL_ROSTER` | Per-history values, primary and secondary macros, sample variances, and bootstrap `LCB95` all null; bootstrap not computed |

No cross-history support-stratum comparison, paired-seed conclusion, or score-branch difference can be drawn. P-only/W-only/A endpoints are also uncomputable; there is no full-roster endpoint to interpret. One attempted history with no scores cannot estimate whole-history variance, score/support dependence, support-failure frequency, or a rate of construction qualification. Repeated scorer seeds do not increase the independent history count. Do not impute an endpoint, drop the failed coordinate, compute a computable-subset macro/bootstrap, or treat H=32 as an allowance.

## Decision boundary and what remains unresolved

Sprint 19 defines separate `S_pred` and `S_pop` whole-history estimands, common support, and full-roster handling; a structural failure or missing endpoint makes the declared primary result uncomputable rather than a zero or a complete-case estimate. Sprint 20 remains `NOT_READY` for any positive numerical support-margin allowance or tolerance-based acceptance rule. This stop supplies neither empirical success nor a passed soft gate and changes no prior decision: Sprint 18's 48-control construction reserve and stop before Cycle 4, and all Gate V2 predicates and verdicts, remain unchanged.

Yes—if an executable exploratory contract is expected to quantify history-level variance, score/support dependence, or margin-stratum sensitivity, it needs a new prospective independent sample: this attempt produced none of those empirical estimates. This is a scientific prerequisite statement, not authorization for another run.

The roster and any contacted roots/seeds are research-only forever; they must never be promoted to Evaluation or Confirmation. Do not replay, resume, finish, replace, or score the contacted frozen roster. Any future evidence-gathering effort would be a **different, prospectively versioned design**, not continuation of this pilot. It would require, at minimum:

1. Resolve the exact-48 runner predicate versus the allocator's minimum-48 contract against the authoritative static allocator source; prove the affected path with a disposable smoke and obtain independent review before any separately authorized contact.
2. Obtain a separate explicit user decision on the future sample-size and any tolerance-policy question. This report chooses no quota, `H`, miss allowance, seeds, or acceptance rule and grants no authority for another run.
3. Freeze genuinely fresh, disjoint research roles and a new roster under that design, then complete a new independent pre-contact review before any authorized execution.
4. Obtain the still-missing history-level variance and support/dependence evidence needed to assess precision and margin-stratum sensitivity. A scorer toy proof or this single unscored history does not provide it.

The prior audit limits remain: seed-collision evidence covered the static repository binding/evidence catalog only, not a data-root scan or Sealed-root inspection; raw data/checkpoints remain remote; peak GPU and host utilization were not measured. The detached launcher's optional PID echo was the literal string `$!`, not an active PID, but completion is independently evidenced by the recorded attempt, exit code, and wall-time log. These limits do not change the observed stopped status.

## Evidence references

- [Sprint 21 plan and stop boundaries](sprint-plans/sprint-21.md)
- [Sprint 19 estimands](exploratory-benchmark-estimands.md) and [non-executable protocol draft](exploratory-benchmark-protocol-draft-v1.md)
- [Sprint 20 `NOT_READY` decision](exploratory-tolerance-decision.md) and [sensitivity analysis](exploratory-tolerance-sensitivity.md)
- [Accepted Task 5 execution evidence](../artifacts/sprint-21/task-5.md) and independent review `agent://S21Task5Review` (`PASS`, zero actionable findings)
- Bounded records: [`attempt.json`](../artifacts/sprint-21/pilot-v1/attempt.json), [`summary.json`](../artifacts/sprint-21/pilot-v1/summary.json), [`pilot-ledger.jsonl`](../artifacts/sprint-21/pilot-v1/pilot-ledger.jsonl), [`H-S21-PILOT-01.json`](../artifacts/sprint-21/pilot-v1/history-evidence/H-S21-PILOT-01.json), and [`checksums.sha256`](../artifacts/sprint-21/pilot-v1/checksums.sha256)
