# Sprint 18 Task 69 — S18-ITER-0005 A08 guarded execution result (FAILED_ATTEMPT)

**Task / attempt / agent:** `S18-T69` / `S18-T69-A08` / `S18Task69A08`
**Outcome:** FAILED_ATTEMPT. Candidate `S18-ITER-0005` (seeds `32064–32079`) was contacted once under the Main-authorized release and rejected at the fixed whole-candidate gate: one-shot all-16 in-memory preflight returned `PREFLIGHT-FAIL` (`15/16`). The frozen ledger records `candidate_rejected` (`fixed_16_history_preflight_rejected`). No role was materialized, no qualification ran, no probe ran, no Confirmation waveform was written to a role root. This is a scientific feasibility failure of the whole candidate, not a metadata or environment issue. Task69 remains active and not data-done; no retry/resume/reseed under this candidate.

## Main authorization and fresh receipt (Main behalf, hash parity)

Main authorized this fresh worker to materialize the authorized receipt on Main's behalf and execute the first scientific candidate-5 attempt. Created only the new receipt `artifacts/sprint-18/S18-ITER-0005-main-task68-release-v1.json` (schema `sprint18-main-candidate-release-v2`, status `RELEASED_BY_MAIN`, scope `Task69-preflight-and-nonconfirmation`). It binds candidate `S18-ITER-0005`, canonical binding `10e8bbbfe91951296536a2e5032162d4dd3f97bdda121d5d89ca787df4f5f532`, raw binding file `c903f04bb3a64a30e66ee0cc02d91d02c23e432cb5a5d4ad79ea09d2b3a2086b`, closure `73f9125c3b28f9ff7127393f2a63d8c15baca53091ac7fe450788335ae1c10a6` (`git-tracked-utf8-text-lf-sha256-v1`), Task67 provenance `artifacts/sprint-18/task-67.md` / `e576debdd048c780ca6c25d91923014705bc8b3723e70566e8dd99c947139867`, base/parent `2b1bf310d15bb682e0278ae59ae64f62bb67589c`, checkpoint `bf49a81956e857563f5529022033795bd01e08bf`, review `agent://S18Task68R11` PASS zero findings. Local/remote receipt SHA-256 parity: `e2e4da5e45be341dd6cb2b5968b174a44175e9dfc4123b8b5fa34e1008ee603e` (1,053 bytes). Older receipts/candidates untouched. User `Ok tiếp` accepts disclosed R10/A07 metadata-only deviations as nonblocking; A07/R08 records preserved.

## Remote runtime and pre-contact validation

Execution worktree `/home/trietlm/anomaly-representation-learning-s18t68-a02-worktree`, branch `sprint18-task68-a02-runtime`, `HEAD=bf49a819…`, parent `2b1bf310…`, tree `206a97d9…`, `status --porcelain` 0 lines, stash 0, origin `https://github.com/triet4p/anomaly-representation-learning.git`, common dir `/home/trietlm/anomaly-representation-learning/.git`. Bound runtime: CPython `3.12.13`, NumPy `2.5.2`, SciPy `1.18.1`, Torch `2.14.0+cu130`, uv `0.12.21`/`7af826859`. Canonical checkout untouched. Normal `--stage validate` (exact bound env) exited `0` with `STATIC_BINDING_PASS`, matching binding/raw/closure digests, `contacted:false`.

## One-time all-16 preflight: PREFLIGHT-FAIL 15/16

The exact released `--stage preflight --release artifacts/sprint-18/S18-ITER-0005-main-task68-release-v1.json` ran once. Durable raw verdict `PREFLIGHT-FAIL`, `15/16`, protocol `sprint15-benchmark-protocol-v7`, seeds exactly `32064–32079` in fixed role order. In-memory waveforms generated; no role root/shard/manifest persisted.

Passing 15/15 non-failing coordinates (all `feasible=true`, zero failed checks, controls ≥55):

| Seed | Role | Pools P1/P2/W1/W2/A1/A2 | Controls |
| --- | --- | --- | --- |
| 32064 | DESIGN | 36/36/22/23/29/34 | 64 |
| 32065 | DESIGN | 33/33/25/26/31/21 | 73 |
| 32066 | DESIGN | 37/34/24/24/23/28 | 68 |
| 32067 | DESIGN | 36/32/26/21/28/29 | 65 |
| 32068 | FIT | 38/36/24/24/26/24 | 68 |
| 32069 | FIT | 35/36/24/24/28/28 | 70 |
| 32070 | FIT | 31/34/19/21/34/33 | 67 |
| 32071 | CALIBRATION | 41/42/21/19/26/21 | 65 |
| 32072 | DEVELOPMENT | 39/41/24/23/25/29 | 70 |
| 32073 | DEVELOPMENT | 34/39/23/23/23/29 | 74 |
| 32074 | DEVELOPMENT | 41/43/23/19/25/21 | 55 |
| 32075 | DEVELOPMENT | 31/29/28/29/19/29 | 83 |
| 32077 | CONFIRMATION | 32/33/27/22/28/26 | 69 |
| 32078 | CONFIRMATION | 37/35/19/25/23/26 | 69 |
| 32079 | CONFIRMATION | 33/35/27/26/26/22 | 75 |

Failing coordinate (the whole-candidate rejection cause):

- Seed `32076`, `S18I-ITER-0005-CONFIRMATION-01` (CONFIRMATION-01): `feasible=false`, `reason=qualification-shortfall`, `missing=['structural_core_integrity']`, elapsed `34.5s`.
- Eligible pools P1=28, P2=29, W1=26, W2=23, A1=30, A2=35 (all above quota P/W≥12, A≥8); allocated full quota 12/12/12/12/8/8; controls `76` (≥48); evaluable robot days `1734`; positive/negative robots `9/9`; cohort mix P=0.333/W=0.287/A=0.380 (all in 15–60%); lead P/W 80% pass; all 11 construction checks pass; all 19 Confirmation hard-floor/role-gate checks pass.
- Sole failed check: `structural_core_integrity=false`, driven by exactly one structural subcheck `cohort_subtype_physical_shapes_and_duration_bounds=false` (all other 15 structural subchecks pass). Allocation succeeded, so this is a physical-shape/duration-bound shortfall on the in-memory manifest, not a quota/control/allocator infeasibility.
- Support context: positives P=57/W=49/A=65 (total 171); eligibility rejections `reset-horizon=70` only.

Per strict policy this single-coordinate shortfall rejects the entire 16-history candidate. No per-role rescue, reseed, replacement, pooling, or passing-prefix promotion is permitted.

## Ledger/marker preservation and custody

Remote candidate root contains only `_attempt-001/` with `attempt.json`, `ledger.jsonl`, `preflight-raw.json`; zero role directories (`DESIGN`/`FIT`/`CALIBRATION`/`DEVELOPMENT`/`CONFIRMATION` absent — only the attempt directory observed). Marker start `2026-10-08T01:51:05.763199+00:00`, attempt 1. Ledger 4 events: `attempt_started` → `preflight_started` → `preflight_recorded` (`PREFLIGHT-FAIL`) → `candidate_rejected` (`fixed_16_history_preflight_rejected`). No `role_materialization_started`, no `role_materialized`, no `nonconfirmation_qualification_pass`, no Confirmation authorization event. Confirmation waveforms were generated only in memory inside the all-16 preflight (required fixed roster scope); no Confirmation role root/probe/selection ran. No Sealed/training/learned scorer contact. `record-nonconfirmation-pass` never ran (precondition unmet).

Hash parity (remote == local copy): marker `40e2cf3bb26ad4571c2028509968b2e8e02a0b3227a76ae29f5944672e1e4f3b` (3,288 bytes); ledger `1579d9b67be2ad0e703251f7ad10abbfeb2c30032d48cb5b7247b5a6098b1683` (5,280 bytes, 4 events); preflight-raw `9f28471eed1c3abf2795c45fd1193dfc4fcadcb7fe72a452ff7d6395368d3eef` (94,713 bytes). Local copies: `experiments/sprint18-task69-attempt-S18-ITER-0005-A08-preflight-raw.json`, `...-A08-attempt.json`, `...-A08-ledger.jsonl` (all non-ignored, `git check-ignore` exit 1). Bounded records: `experiments/sprint18-task69-attempt-S18-ITER-0005-A08.json` + this Markdown. Raw datasets stay remote/uncommitted. Old candidates/receipts/R08 FAIL preserved; nothing erased or relabeled. No source/env/ref mutation; remote worktree clean (0 status lines) at `bf49a819…`.

## Unreached criteria (by policy, not omitted work)

Preflight `15/16` → materialized `0/12`, qualified `0/12`; no Fit centroid/statistics, no Calibration q95, no Development probe evaluation, no public reloads (no roots to reload), no `record-nonconfirmation-pass`. No second preflight, no resume, no new candidate, no A13 action by this worker. Full SUCCESS requires a future reviewed correction + fresh whole candidate; review/checkpoint/Main status still pending. Latest user stop applies after recording + review; no further release by this attempt.
