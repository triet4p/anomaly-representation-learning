# Sprint 18 Task 70 — S18-ITER-0005 assessment Confirmation A03 result (zero-output recovery completed; whole-16 certified)

**Task / attempt / agent:** `S18-T70` / `S18-T70-A03` / `S18Task70A03`
**Outcome:** `TASK70_ALL_OBSERVABLE_CRITERIA_PASS_SCIENTIFIC_REVIEW_PENDING`

The single user-authorized zero-output recovery of `S18-ITER-0005-ASSESS-P-ALLOWANCE-V1/_assessment-001`
was executed through the genuine **detached public path** (never an SSH-session-bound launcher, so the A02
operational mistake was not repeated), and all four unchanged Confirmation seeds were materialized exactly
once through the real public `synth.cli` children and reloaded through the public loader.

## What actually ran

Exactly **one** public invocation, detached before the SSH session could end:

```
setsid nohup sh /tmp/s18t70a03-run.sh > /home/trietlm/s18t70a03-recovery.log 2>&1 < /dev/null & echo LAUNCHED_BGPID=$!
```

with the public args `--assessment --binding experiments/sprint18-iterative-assessment-c5-allowance-v1.json
--release artifacts/sprint-18/S18-ITER-0005-ASSESS-main-assessment-release-v1.json
--stage materialize-confirmation --qualification-record experiments/sprint18-task69-assessment-S18-ITER-0005-A09-qualification-record.json
--qualification-sha256 c5a8e8ccbc6be35a58fe7363ee4e8af489d3fafb93aaf6411c3936e5d349dc4b
--task69-release artifacts/sprint-18/S18-ITER-0005-TASK69-main-assessment-release-v2.json
--execution-release artifacts/sprint-18/S18-ITER-0005-TASK70-corrected-execution-release-v3.json
--recovery-release artifacts/sprint-18/S18-ITER-0005-TASK70-interrupted-confirmation-recovery-v1.json`

- Launcher **pid 29880**, own session and process group (`sid = pgid = 29880`), log non-empty before disconnect.
- **After the SSH session closed**: pid 29880 with **PPID 1** (reparented to init), still alive — proof it is
  outside the SSH session lifetime.
- **Exit 0** (`RECOVERY_EXIT=0`), launched `2026-10-09T03:30:48Z` → done `2026-10-09T03:33:34Z` = **166 s**.
- No retry, no rerun, no parallel duplicate, no fallback.

## Ledger (append-only; the first 30 events are byte-identical to the A02 carrier)

- **38 events**, full sha256 `ac10a975ae9c161633e5dac8ef05b3d1bfa2e33487f0d84841a2439d6594f3a6`.
- **First-28 prefix immutable**: `625dcc1c6e14b43ac031afb590565a1af8461146579c8aff212d60bf3f061919`.
- Events 29/30 preserved verbatim (`confirmation_materialization_authorized` + CONFIRMATION-01 started).
- Event **31 = `confirmation_recovery_authorized`** (`a4ca695e…`, receipt `9a420327…`, interrupted
  `05cbc0fb…`, qualified `625dcc1c…`, resume `S18I-ITER-0005-CONFIRMATION-01`/32076).
- Census: 16 started / 16 materialized + 1 each attempt / preflight_started / preflight_recorded /
  nonconfirmation_qualification_pass / confirmation_materialization_authorized / confirmation_recovery_authorized.
  **Zero** fail / reject / retire / interrupt events.

## Confirmation 4/4 (seeds 32076–32079)

| History | Seed | Samples | Manifest sha256 |
|---|---|---|---|
| `S18I-ITER-0005-CONFIRMATION-01` | 32076 | 6,457 | `db2b50444cb54c8a4bce700771efcef691e147b96ceb68985997e1115d74bbd2` |
| `S18I-ITER-0005-CONFIRMATION-02` | 32077 | 6,468 | `54d787eb424066c3e8c18acb528f6ec19e2e9f4211b46e669390a3a330c8b762` |
| `S18I-ITER-0005-CONFIRMATION-03` | 32078 | 6,505 | `5c06a6613bcb7bc83b5646423f44f0440ecff8a276935b630c220ceff247ad3f` |
| `S18I-ITER-0005-CONFIRMATION-04` | 32079 | 6,456 | `50a506a4670e5da45dbfa6b29db79f74a84e947b93de6f2d47edd65613c8e840` |

All four reloaded through the public `synth.chronle.load_chronological` with role, protocol
`sprint15-benchmark-protocol-v7`, config hash, shard roster and sample order verified.

**Per-role nominal exception (hard predicate `2.0 <= duration_d < 16.0` holds for all 16 roles):** exactly one,
on seed **32076** only — count 1, duration **15.545084957564008 d**, allowance 1.0 d,
`structural_core_integrity = true`. Reported, not waived; no new metric introduced.

## Frozen statistics used (no refit, no new threshold)

Fit standardization + healthy centroid and Calibration q95 were recomputed **read-only from the frozen
Fit/Calibration roots before any Confirmation contact** and matched the immutable Task69 qualification record
byte-exactly: 5,446 healthy files / 167,419 patches, feature dim **59**, centroid norm
**2.290221527135249e-11**; Calibration 1,877 healthy files / 56,889 patches, q95 **195.7544682761532**.

The historical frozen-stats digest `181b83d18329d7ffe90b97ddffe63250c423c51c1f30e9760f1d19952ddff41e`
was **not** re-derived (A01's derivation input is not stored and its session log truncates the command at
512 chars); the underlying values were verified directly from the immutable arrays.

## Fixed observable probe on the four Confirmation histories — all EG4 gates PASS

Threshold `195.7544682761532`. Macro P+W **0.7782929988054331** (≥ 0.65), macro P **0.7950602711718822**
(≥ 0.70), macro W **0.7566116057485478** (≥ 0.60), history-block bootstrap LCB95 P+W **0.7555530669372365**
(> 0.50), directional **4/4** (≥ 3). Per history: P recall 0.667–0.719 with median lead 2.05–3.34 d,
W recall 0.591–0.717 with median lead 3.86–4.66 d, FAR 0.0219–0.0419 (≤ 0.05); abrupt companion AUROC
0.380–0.531 reported separately with no minimum. `support_sha256 da169c1f…`.

## Whole-16 certification — PASS (18/18 conditions, zero failures)

16-role roster exactly as bound, seeds 32064–32079 fixed order and disjoint, no replacement or search,
all 16 canonical `ROLE/history_id` paths distinct and non-nested, all 16 public-loader reloads, all 16
structural core-integrity checks, DESIGN block 4/4, CONFIRMATION hard floors 4/4, the 12 qualified manifests
byte-identical to the immutable Task69 record, the 4 fresh Confirmation manifests distinct, ledger
4-ordered-materialized with exactly one recovery event and an immutable first-28 prefix, frozen centroid
59 dims / norm exact and frozen q95 exact, and all 16 role permissions matching the binding.

## Custody

Remote worktree still at `8485344a2ee3143858db68fdbad3a6ed3def8890` with porcelain 0; no commit, ref write or
push by this worker; no orphan compute after exit; canonical `/home/trietlm/anomaly-representation-learning`
still `23642b47…` + 2 geometry dirs, untouched; all receipts and the qualification record unchanged; final
public bare `--assessment --stage validate` still exit 0 `STATIC_BINDING_PASS`.

**Not claimed:** no scientific retirement or reclassification, no Task71 / ML progress, and no independent
approval of this evidence — that gate is Main-owned and pending.

**Related records:** `artifacts/sprint-18/task-70.md#A03` (full durable handoff, including the artifact
restoration incident), `artifacts/sprint-18/s18-t70-a03-attempt-proof.json`,
`...-A03-whole16-certification.json`, `...-A03-role-reconciliation.json`,
`...-A03-frozen-statistics.json`, `...-A03-ledger.jsonl`, `...-A03-recovery-log.txt`.
