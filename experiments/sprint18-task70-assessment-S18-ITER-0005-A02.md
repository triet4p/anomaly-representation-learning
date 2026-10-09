# Sprint 18 — Task 70 A02 Confirmation attempt (runtime-guard defect, no scientific verdict)

**Task / attempt / agent:** `S18-T70` / `S18-T70-A02` / `S18Task70A02`
**Role / tier / selector:** explicit `gold-task` / Gold; received selector
`opencode-go/muse-spark-1.3-contributor:xhigh`, `blocking:false`;
provider-applied effort/quota unknown. `advisor_needed:false` throughout
(single frozen-guard path with direct ledger/exit evidence; the continuation
blocker below is a Main routing decision, not a diagnostic question).
**Outcome:** `RUNTIME_GUARD_DEFECT_AFTER_LEGITIMATE_APPEND_NO_SCIENTIFIC_VERDICT_CANDIDATE_NOT_RETIRED`
— not a scientific PASS/FAIL, not a retirement. Machine record:
`experiments/sprint18-task70-assessment-S18-ITER-0005-A02.json`.

## What happened (exact)

1. Pre-contact guards on the isolated Linux runtime at exact D
   `42df4249d3b4b13067dbef4e543bca82efe5c279` (parent C `9e9e2227…`,
   tree `f6323c0e…`, porcelain 0): public `--assessment --stage validate`
   exit `0` `STATIC_BINDING_PASS` (1,331 ms, stderr 0 B); `synth.cli --help`
   exit `0` (stderr 0 B). Qualified DATA custody verified byte-identical:
   ledger `625dcc1c…` (28 events), qualification `c5a8e8cc…`, attempt
   `875a3d03…`, preflight-raw `3d1d774f…`, `CONFIRMATION*` 0.
2. Exactly one public one-shot `--assessment --stage materialize-confirmation`
   with the BS02/BSR02-authorized receipts (assess-main v1 `45632676…`,
   Task69 v2 `217fb15f…`, execution v2 `314140ee…`, qualification `c5a8e8cc…`)
   launched detached (launcher pid 5994). It passed every release/execution
   guard and appended two legitimate ledger events — seq 29
   `confirmation_materialization_authorized` (`31dfa35d…`, Task69 checkpoint
   `7be574d9…`) and seq 30 `role_materialization_started`
   `S18I-ITER-0005-CONFIRMATION-01` seed 32076 (`87557f9d…`, hash-chained to
   event 29) — then the launcher was killed by the SSH session close (~4 min;
   pid gone, stdout/stderr 0 B, no child survived). No CONFIRMATION
   waveform/shard/manifest was written (`CONFIRMATION/` absent, 0/4).
3. A detached `setsid nohup … & < /dev/null` relaunch (launcher pid 7469,
   survives session close) refused exit `2`, stdout 0 B, stderr
   `sprint18 candidate guard refused: qualified assessment ledger event count differs from the reviewed state`.
   Root cause at source: every `materialize-confirmation` continuation
   re-enters through `_assert_qualified_assessment_execution_state` →
   `_verify_qualified_assessment_ledger_shape`
   (`experiments/sprint18_iterative_candidate_v1.py:972-996`), which pins
   `QUALIFIED_ASSESSMENT_LEDGER_EVENTS = 28` and the reviewed `625dcc1c…`
   bytes and rejects any ledger already carrying the authorized/interrupted
   Confirmation prefix. Retrying through the same entry would instead hit
   `interrupted_confirmation_no_resume` / `interrupted_materialization_no_resume`
   and record a candidate-retiring failure event, so no retry was attempted.
4. Post-state (verified, both claims grounded): ledger 30 events
   (`05cbc0fb…`; first-28 prefix still `625dcc1c…`); event census
   13 started / 12 materialized / 1 attempt_started / 1 preflight_started /
   1 preflight_recorded / 1 nonconfirmation_qualification_pass /
   1 confirmation_materialization_authorized; zero fail/reject/retire/interrupt
   events; no orphan compute (no `sprint18_iterative`/`synth.cli` process);
   receipts unchanged (`314140ee…`, `217fb15f…`, `45632676…`); only the ledger
   file is newer than run-start; all `/tmp/s18t70a02-*` scaffolds removed;
   local/origin/isolated HEAD still `42df4249…`, local status 64/0-staged.

## Why this is a guard defect, not a candidate verdict

- The two appended events are the runner's own legitimate authorization
   records (valid sequence + hash chain + Task69 checkpoint pin), not
   tampering: the first-28 prefix digest is unchanged and no failure event exists.
- The runner has no same-run resume path for a legitimately-appended
   authorized prefix: the pre-write custody gate demands the reviewed 28-event
   shape while the post-authorization state machine forbids replay/resume.
   A same-entry retry would convert this operational interrupt into a recorded
   scientific retirement, which the assignment forbids (Main decides).
- Scientific evidence stands at 0/4: no Confirmation history completed, no
   reload/probe/certification ran, frozen Fit centroid (dim 59,
   norm `2.290221527135249e-11`) and Calibration q95 (`195.7544682761532`)
   untouched, labels/durations/gates unchanged.

## Preserved / untouched

- Old A08 `_attempt-001` FAIL bytes; A09 six carriers + R10 PASS
   (`3458fba4…`, zero findings) + CP03/DEP03 (`7be574d9…`, parent `003b94bc…`);
   old C v1 receipt `4cf2ee68…` (stale, untouched); A01 GUARD-BLOCKED carriers;
   BS02/BSR02 observation receipts and reports; BSR02 snapshot `1f5b67f4…`.
- New bounded carriers (this attempt only): the JSON record above; this file;
   no new ledger/certification/result carrier was minted (nothing completed);
   `CHANGELOG.md` updated only if convention requires (Main-owned decision).

## Remaining gates (Main-owned, NOT done here)

- Fresh fix for the interrupted-authorized-prefix continuation (reviewed
   resume-aware continuation or ledger-restore-and-retry authorization), then
   the actual 4/4 Confirmation materialization + fixed-probe evaluation +
   whole-16 certification under a fresh review/authorization. Full Task70 stays
   incomplete; no DATA-complete, no Task71 claim.
- **Routing refs:** `history://S18Task70BS02`, `history://S18Task70BSR02`,
   `history://S18Task70C05`, `history://S18Task70CR05`, `history://S18Task70A01`,
   `history://S18Task69A09`, `history://S18Task69R10`, `history://S18Task69CP03`,
   `history://S18Task69DEP03`; plan `docs/sprint-plans/sprint-18.md:604`;
   BSR02 snapshot `1f5b67f4…`; CP03/DEP03 `7be574d9…` (parent `003b94bc…`).
