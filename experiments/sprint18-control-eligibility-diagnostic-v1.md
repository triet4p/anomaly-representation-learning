# Sprint 18 Control and Eligibility Mechanism Diagnostic v1

**Diagnostic ID:** `sprint18-control-eligibility-mechanism-diagnostic-v1`  
**Status:** Prospective research-only diagnostic contract. Task 61 permits only a disjoint disposable instrumentation fixture; the study block is reserved for Task 62 and requires a fresh independent evidence-review PASS first.  
**Schema:** [`sprint18-control-eligibility-diagnostic-v1.schema.json`](sprint18-control-eligibility-diagnostic-v1.schema.json)  
**Runner:** [`sprint18_control_eligibility_diagnostic_v1.py`](sprint18_control_eligibility_diagnostic_v1.py)

## 1. Question and interpretation boundary

This diagnostic records where candidate failure controls and eligible positive anchors are lost while running the already-frozen `sprint15-v7` and `sprint18-c2` configurations. It explains mechanism-level eligibility, control-window, and exact-allocation counts under unchanged v7/C2 semantics. It is descriptive and research-only: it does not qualify data or a benchmark, establish causal effects or population pass probabilities, authorize training, alter a gate or physics, or implement Cycle4. Attribution labels describe identifiable generator reset operations only; they are not claims of physical cause or effects of a method.

Task 59 result SHA-256: `0850693cba63155d04243591b88323c819e2366b2590314983f5474f3612bb7b`. Task 60's interpretation artifact was reviewed PASS (`agent://S18Task60Review`). The observed historical failures remain context, not thresholds to optimize against: v7 had controls 46/48 at seed `918001` and W1 11/12 at `918003`; C2 had controls 45/48 at `918008` and 46/48 at `918010`. This diagnostic neither changes nor retries that previous block.

The frozen v1/v3 research contracts and Task 58 v1/v2 runner/contracts remain historical, immutable inputs. The v2 base source/config closure has 47 files and SHA-256 `2678d50a328fb4d34c941c38c8d04e91d85ad3c363f4793cd02e3aa0a5b0b834`. v1/v2 runner and contract SHA-256 values are included in the release binding below and checked by the runner. Task 59/60 interpretation does not modify those frozen inputs.

## 2. Fixed namespace and profile semantics

The sole prospective study block is exactly seeds `918100–918107`, ascending. For each seed, evaluate `sprint15-v7` and then `sprint18-c2`, producing at most 16 unique coordinates. Frozen reference-profile config hashes are `7de530a28d23` (v7) and `720e31e94681` (C2), with abrupt-A rates `2.2e-5` and `3.3e-5`; each coordinate row separately records its resolved config hash for that history seed. The active quotas are read from the frozen balanced allocator and bound in the release block, including the 48-control requirement and frozen P/W/A subtype quotas. No input profile, quota, seed, order, retry, replacement, resume, screening, or continuation behavior is user-selectable.

Seeds `918100–918107` MUST NOT be evaluated under either profile during Task 61. Frozen earlier-study seeds `918000–918015` MUST NOT be regenerated or reused. Namespace guards check recorded direct seeds and C3 ranges in the frozen bindings, the prior Task 59 block, and the disposable fixture seed. Any collision blocks the run; it does not authorize a substitute seed. The only permitted Task 61 evaluation is smoke fixture seed `93061`, which is not a study coordinate. Task 62 may execute the frozen block only after a fresh independent evidence-review PASS and an exact reviewed freeze digest are supplied.

## 3. Recorded denominators and trace semantics

Every summary carries its denominator or a named denominator field, with the raw event ledger and manifest row counts retained. The exact v7/C2 implementations remain the authorities: `balanced.eligible_anchors`, `events.anchor_rows`, `events.select_control_windows`, and the exact `balanced.allocate_quotas` path. The runner replays detailed predicates, reconciles membership/order against those canonical outputs, and stops rather than silently reporting a mismatching trace.

### 3.1 Raw events to eligible anchors

For each raw failure-ledger event, preserve its ledger index, failure ID, robot, program, failure time, cohort/subtype, disposition, reset-window intersection, positive-window file count, and the exact `balanced._lead_detail` record when evaluated. Program attribution joins the unique manifest operation row by same robot and exact operation-end/failure time; zero or multiple matches block the coordinate.

Eligibility uses the unchanged precedence: reset in the positive horizon; otherwise no positive endpoint files; otherwise P/W lead-support shortfall; otherwise eligible. The trace verifies event-by-event and in-order agreement with `eligible_anchors` and reconciles, by cohort and subtype, `raw = eligible + reset-horizon + no-endpoints + lead-shortfall`. Summaries include raw and eligible counts by cohort, subtype, robot, program, and robot/program; exclusion counts by reason/cohort/subtype; and eligible cohort mix with the eligible-pool count as denominator. Subtype counts are compared with the frozen P/W/A quota margins.

Reset attribution uses intersecting maintenance episode identities. A `pmaint-*` episode is labeled preventive based on the generator's preventive-block path. A `maint-*` episode is labeled corrective only when the same-robot, same-time failure linkage is unique. Multiple identifiable intersecting windows are labeled overlap; missing or ambiguous linkage is undetermined. The exclusion-class denominator is exactly events rejected for reset-horizon intersection. The report preserves each intersecting window and its source basis; an unrecognized or ambiguous source is never forced into a known single-cause class.

### 3.2 Manifest rows to control opportunities

The manifest denominator is every in-memory file row built for that profile/seed. The anchor-row stage records the row predicates for program/robot exclusions, temporal-view presence, censoring, quarantine, and maintenance overlap; it retains rejected rows and their first exclusion reason and verifies `input manifest rows = eligible anchor rows + rejected anchor rows`. With the frozen call's empty program/robot exclusion sets, those exclusion flags remain false.

Every anchor row retained by `anchor_rows` becomes one control-candidate opportunity, after stable end-time sorting. For each candidate, record robot/program, anchor and window bounds, independent flags for spacing from the last selected same-robot window, any same-robot failure in `[end-H, end+H)`, and any reset crossing `[end-H, end]`. Preserve all conflict IDs/windows and the canonical first-rejection precedence: prior-selected-window spacing, same-robot failure horizon, then maintenance reset. Record every selected window and every member file/program/end-time row. The trace must equal `select_control_windows` in order and membership and reconcile `candidate opportunities = selected windows + first-rejection count`.

Report the candidate denominator (eligible anchor rows), overlapping predicate-flag counts, first rejection counts by robot/program, selected windows by robot and anchor program, and member-program distributions both by membership occurrence and distinct file. Selected same-robot windows may touch at the spacing boundary; positive-duration overlap is an error. Member rows use the unchanged inclusive `[anchor_end-H, anchor_end]` range, so a boundary row may occur in more than one member listing and both the occurrence and unique-file denominators are explicit.

### 3.3 Exact allocator and margins

Record each eligible subtype's pool count, frozen quota, and pool-minus-quota margin, along with selected positive-event counts by subtype, robot, program, and robot/program whenever feasible. Record selected-control count, frozen control quota, and control margin before positive allocation. Preserve the allocator's unmodified exact result/margins when feasible or its unmodified infeasibility record and proof details when infeasible. `INFEASIBLE` is a valid, preserved coordinate result and does not imply a structural or qualification pass. `UNAVAILABLE`, a witness breach, a trace mismatch, or another infrastructure error is recorded as blocked and stops the block; no later coordinate is attempted.

## 4. No-write boundary, persistence, and freeze

The diagnostic uses `build_chronological` only in memory, extracts manifest rows, and runs the existing deterministic exact allocator. It MUST NOT call the chronological materializer or seal writer; write generated data/waveform roots; open Sealed inputs; touch benchmark paths or benchmark roles; train or validate a model; or qualify data. Each row records these boundaries and a canonical SHA-256 seal. The disposable smoke actively replaces materialization and seal-writing entry points with failures and persists only temporary JSON outside the project.

The fixed study output location is `artifacts/sprint-18/control-eligibility-diagnostic-v1/`, with `attempt.json` and `result.json`. The runner claims this directory once, writes the attempt marker before any coordinate begins, then atomically persists the result before the marker binding its result hash. Each in-progress coordinate and remaining budget are recorded. Any existing output directory blocks all execution. Interrupted or partial attempts are not resumable; do not retry or continue. All rows, including an infeasible or blocked row, are sealed and preserved before stopping whenever a coordinate evaluation has begun.

The release binding freezes the runner, schema, base and extended source closure, both profiles/rates, exact quota, fixed order/budget, namespace inputs, fixture seed, and historical runner/contracts. The freeze digest is SHA-256 over the bytes `SPRINT18-CONTROL-ELIGIBILITY-DIAGNOSTIC-FREEZE-v1\n`, the raw contract SHA-256 bytes, raw runner SHA-256 bytes, raw schema SHA-256 bytes, then canonical JSON containing the sorted extended source closure, checked namespace, profile hashes/rates, full coordinate order, fixture seed, and unique-coordinate budget. The attempt/result carry the digest and source-closure hashes; each row is sealed over canonical JSON with its seal field omitted.

## 5. Commands and authorization

Task 61 instrumentation proof (the only command authorized before independent review):

```bash
uv run --locked python experiments/sprint18_control_eligibility_diagnostic_v1.py --smoke
```

This command evaluates only fixture seed `93061` under v7 then C2, writing its two real rows only in a temporary directory. It verifies the attempt marker's result digest; source/freeze/namespace, coordinate-order, duplicate, row-integrity, no-write, and one-shot guards; and an impossible P1 quota on the already-built v7 fixture.
The failure probe compares its allocator inputs, default quota, and diagnostic row before and after. It does not create the fixed study output. No Task 61 invocation may use seeds `918100–918107`.

Only after a fresh independent Task 61 evidence-review PASS, Task 62's single permitted study invocation is:

```bash
uv run --locked python experiments/sprint18_control_eligibility_diagnostic_v1.py --study --review-passed --expected-freeze <freeze-sha256-from-reviewed-task-61-artifact>
```

The runner rejects missing review acknowledgement, missing/mismatched freeze digest, source/config drift, namespace collision, or an existing output directory. Do not run `--study` as part of Task 61. Main owns review and Task status; this contract does not itself grant authorization.

## 6. Release binding

The protocol, schema, and runner hashes are jointly checked before any smoke or study path. The base closure digest and historical v1/v2 hashes are fixed; the extended closure and current task-specific release values are recorded in the release block. The Task 61 artifact records the final observed release-binding values, freeze digest, disposable smoke output, and graph-update result.

<!-- release-binding -->
```json
{
  "runner_sha256": "881e29c638954b35849358615676655d0d269214e3fa7a5913ad19c4fcd5b961",
  "schema_sha256": "72612ba1f721390e7dbf76993469f134f64d6ebc622a06d6ac4e4b0769679d4e",
  "base_source_closure_sha256": "2678d50a328fb4d34c941c38c8d04e91d85ad3c363f4793cd02e3aa0a5b0b834",
  "source_closure_sha256": "dd8b2e0a3ac3b6bd6bc66dbbd7f14c58264da0f9ba90d164c0a7472888e9b383",
  "profile_config_hashes": {
    "sprint15-v7": "7de530a28d23",
    "sprint18-c2": "720e31e94681"
  },
  "profile_rates": {
    "sprint15-v7": 0.000022,
    "sprint18-c2": 0.000033
  },
  "quota": {
    "a": 16,
    "a1": 8,
    "a2": 8,
    "controls": 48,
    "p": 24,
    "p1": 12,
    "p2": 12,
    "program_cap": 0.6,
    "programs": 2,
    "robot_days": 240,
    "robot_neg_cap": 0.4,
    "robot_pos_cap": 0.35,
    "robots_neg": 6,
    "robots_pos": 6,
    "spacing_s": 1209600.0,
    "w": 24,
    "w1": 12,
    "w2": 12
  },
  "diagnostic_id": "sprint18-control-eligibility-mechanism-diagnostic-v1",
  "seed_order": [918100, 918101, 918102, 918103, 918104, 918105, 918106, 918107],
  "profile_order": ["sprint15-v7", "sprint18-c2"],
  "fixture_seed": 93061,
  "coordinate_budget": 16,
  "v1_runner_sha256": "d65f01ee703d2f89b46c5b0138b3690f811edce304878204d9b1cbf14fa83111",
  "v1_contract_sha256": "1252707032aabfd95d99d1944140e57f8eff3610f7c8d35810e649c65a276bd0",
  "v2_runner_sha256": "d5c4dfe3039bf82ec7eec05bf81f083966b81098ed54813b3e1c3d34c35f50b6",
  "v2_contract_sha256": "8b8e28e489c393cd0ccf33d5844bab9a1d5e562904d671f5b94c122fc4b5ec42"
}
```
