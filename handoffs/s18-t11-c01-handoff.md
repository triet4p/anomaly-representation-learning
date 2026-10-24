# Handoff: S18-T11-C01 (N2-MSP PR02 fresh correction + proof)

**Task:** `S18-T11` — Balance N2 coarse prediction (frozen `alpha` grid; fixed mask budget, scale support, leakage absence, gradient balance; no anomaly outcomes).
**Attempt:** `S18-T11-C01`.
**Runtime agent:** `S18T11C01`.
**Role/tier:** `gold-task` / Gold (user Gold authority for Tasks 10-12; fresh correction after PR02 FAIL of A03).
**Route:** `full` (prerequisite local proof -> independent evidence PASS -> separate Bronze source checkpoint before any new product remote execution; then actual frozen pilots -> final review PASS -> separate Bronze Task11 checkpoint before Task12).
**Quota/cost observation:** `unknown`.
**Handoff path:** `handoffs/s18-t11-c01-handoff.md`.
**Base:** `00e052d3a250f630fbeac02cf6b3a1516a8eefaf` on `refs/heads/master` (`HEAD` and `refs/heads/master` agree; no worker stage/commit/push).
**Outcome:** `prerequisite_fix_ready` (both PR02 findings corrected with focused proof complete; pilots 0/12 pending Main-owned review/checkpoint/rerun).
**Artifact path:** `artifacts/sprint-18/task-11-prerequisite-c01.md`.

## 1. Goal and acceptance (correction slice)

- Keep the verified A03 production fix byte-identical unless the consumer proof exposes a true bug (it did not): `src/representation/trainer.py` aligns the CPU-built N2 union to the collated batch device before the model call; CPU no-op with unchanged values/seeds/allocation/placement.
- HIGH F01: delete the invalid manual-copy test `test_msp_union_mask_aligns_to_batch_device_without_changing_values` (it called `msp_batch_masks` plus its own `.to(...)` and `validate_batch`, never the fixed `train_sprint18_msp_epoch`/`_run_msp_epoch` branch). Replace it with a meaningful CUDA-honest conditional consumer regression: tiny real dataset plus the real N2 trainer through the public `train_sprint18_msp_epoch` entry point with certified `fit_role_sha256`, so the actual bug branch runs; CPU-like loader inputs, CUDA trainer when available, real optimizer update, output/mask-budget/finite consumer assertions. No monkeypatched placement, copied alignment, mock echo, or manual repair. A CPU variant exercises the public path but does not prove CUDA; the honest skip is reported as a limit.
- MEDIUM F02: add one additive `CHANGELOG.md` `Fixed` entry for the N2-MSP union-mask batch-device contract; historical entries preserved.
- No scientific changes, no Task12 work, no plan edits, no worker Git mutation, no SSH, no remote rerun of the affected product before Main gates.

### Satisfied in this attempt

- Production fix preserved byte-exact: worktree `src/representation/trainer.py` SHA-256 `ba283d4e63f68b795e212d0e47a25bd99e84f456c3b8c36864cacfd719e137a4` equals the PR02-reviewed fix digest; `git diff` shows the same `+4/-1` alignment (`union.to(dtype=torch.bool, device=moved["mask"].device)` plus its 3-line comment).
- Invalid test deleted outright (no repinned wording or wiring): the old `test_msp_union_mask_aligns_to_batch_device_without_changing_values` body is gone.
- Replacement regression added: `test_msp_epoch_with_cpu_inputs_runs_on_cuda_batch_device` in `tests/representation/test_sprint18_msp.py`, guarded by `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for N2 union device-placement proof")`. It asserts CPU loader inputs, moves the trainer/model to CUDA, runs `train_sprint18_msp_epoch([batch], arm=arm, epoch=11, fit_role_sha256=fit_role)` with the certified role digest (reused `_canonical_sha256_for_test` helper already proven coherent in A03; no hardcoded fake checkpoint or validator bypass), and asserts a real optimizer step, changed model digest, finite positive coarse/joint outputs, epoch-11 balance support, and finite `r_aux`. On CUDA it traverses the fixed `_run_msp_epoch` branch and fails pre-fix with the server `RuntimeError` by construction.
- CHANGELOG updated additively: one `Fixed` bullet naming `RepresentationTrainer._run_msp_epoch` union-mask batch-device alignment and the CUDA end-to-end effect; no historical line touched.
- Focused proof only (per assignment): corrected public-path regression plus the real-CPU public-path trainer smoke; honest CUDA skip; no broad full suite.
- Prior reports/artifacts preserved: PR02 FAIL report plus manifest snapshot `1322f6da...`, A03 fix record, A03 handoff, Task10 driver hash `806ac120...`, failed server CUDA log `56b8ffc...` all untouched.

### Remaining (Main-owned gates + rerun)

- Fresh independent evidence PASS on `artifacts/sprint-18/task-11-prerequisite-c01.md` plus this handoff plus the 3 owned paths.
- Separate Bronze source checkpoint (exact snapshot of the corrected scope; no pilots bundled).
- Frozen 12-run N2 pilot rerun on the fixed source plus mechanical selection plus versioned selected-or-null provenance under `experiments/`.
- Actual target-CUDA confirmation happens there, not locally.
- Final evidence PASS plus separate Bronze Task11 checkpoint before Task12.

## 2. Root cause and fix (evidence-backed, unchanged)

- Server trace (`artifacts/sprint-18/s18-t11-a03-pilot-failed.log`, SHA-256 `56b8ffcb7265f125a990168c56db982b27ac50124b2fb4515b22a2bb634aaa20`): first N2 candidate `coef0.100-seed181801` epoch 1 → `train_sprint18_msp_epoch` → `_run_msp_epoch` → `model.forward` → `validate_batch` → `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`.
- Root cause: `msp_batch_masks` (`trainer.py:348`) builds `union = torch.zeros(...)` on CPU always; `_run_msp_epoch` moved the batch to `cuda:0` via `_move_batch` then assigned the CPU union to `moved["mask"]` without a device move. `validate_batch` combined CPU `mask` with CUDA `patch_valid_mask` and faulted.
- Fix (kept byte-identical): `moved["mask"] = union.to(dtype=torch.bool, device=moved["mask"].device)` with the CPU-no-op comment. Mask values, budget, seeds, allocation, and coarse group placement are untouched; B0/N1 arms unaffected.

## 3. Changes (wholly owned; no worker Git mutation)

| File | Change |
|---|---|
| `src/representation/trainer.py` | N2 union device alignment kept byte-identical to the reviewed fix (4 insertions incl. comment, 1 line changed) |
| `tests/representation/test_sprint18_msp.py` | Deleted invalid manual-copy test; added CUDA-honest public-path `test_msp_epoch_with_cpu_inputs_runs_on_cuda_batch_device` plus certified-digest `_canonical_sha256_for_test` helper (net +33 lines, no old-test wording retained) |
| `CHANGELOG.md` | One additive `Fixed` entry for the N2-MSP union-mask batch-device alignment (history preserved) |
| `artifacts/sprint-18/task-11-prerequisite-c01.md` | Correction record (this attempt) |
| `handoffs/s18-t11-c01-handoff.md` | Detailed acknowledgment handoff (this file) |

Owned diff stat: `CHANGELOG.md` +1; `src/representation/trainer.py` +4/-1; `tests/representation/test_sprint18_msp.py` +33. Foreign worktree state (other modified/untracked files owned by the user or siblings) preserved untouched; no stage/commit/push/SSH.

## 4. Verification (actual; focused only)

- `uv run --no-sync pytest tests/representation/test_sprint18_msp.py::test_msp_epoch_with_cpu_inputs_runs_on_cuda_batch_device tests/representation/test_sprint18_msp.py::test_n2_pilot_epoch_coarse_activates_after_warmup_with_infonce -q` — **1 passed, 1 skipped** in 25.79 s. The skip is the honest CUDA guard (`torch.cuda.is_available()` is `False` locally); the pass is the existing real-CPU public-path trainer test on the corrected source.
- `uv run --no-sync pytest tests/representation/test_sprint18_msp.py -q` — **8 passed, 1 skipped** in 87.52 s. All 7 pre-existing CPU tests pass; the new CUDA test skips honestly. No test was narrowed or edited to force passage.
- `src/representation/trainer.py` SHA-256 `ba283d4e63f68b795e212d0e47a25bd99e84f456c3b8c36864cacfd719e137a4` matches the PR02-reviewed fix bytes; the CPU no-op property means A03 local numerics carry over unchanged.
- CUDA availability checked directly: `torch.cuda.is_available()` → `False`. No CPU-proves-GPU claim is made; target-CUDA confirmation stays a Main-gated runtime step.
- Preserved-proof rehashes: `artifacts/sprint-18/s18-t10-a02-driver.py` → `806ac1205bebf732061574e65f02b2abfd4c716a5b1a90f903162c695df5724f`; `artifacts/sprint-18/s18-t11-a03-pilot-failed.log` → `56b8ffcb7265f125a990168c56db982b27ac50124b2fb4515b22a2bb634aaa20`.
- `git diff --stat` scope: only the 3 owned paths. `CHANGELOG.md` edit verified additive (single trailing `Fixed` bullet).

## 5. Tried / rejected

- Reverting or retuning the production fix: rejected — the consumer proof and PR02 audit both confirm the one-line alignment is correct, minimal, allocation-free when aligned, and scientifically neutral. Bytes kept identical.
- Repinning the old test's wording/wiring around the fixed path: rejected per assignment — the manual-copy body was deleted, not repaired.
- Monkeypatching batch placement, copying `.to(...)` in the test, or mocking the model echo: rejected per assignment — the replacement calls the real public epoch with a real optimizer and real consumer assertions.
- Adding a fake CPU proof that CUDA is fixed: rejected — CPU exercises the public path but cannot prove the CUDA branch; the replacement skips honestly without CUDA and documents the limit.
- Hardcoding a fake `fit_role_sha256` or bypassing checkpoint validators: rejected — the test reuses the existing coherent certified-binding helper and the frozen arm contract.
- Broad full-suite rerun: rejected per assignment — the existing 58-test semantics were established by A03/PR02 and the focused N2 file plus adjacent public-path smoke is the assigned proof.
- Any remote rerun of the affected product: not attempted — forbidden before Main gates.

## 6. Blocker

- `none` blocking forward fresh review, Bronze source checkpoint, and Main-gated rerun dispatch. Unobserved risk (separate from proof): actual CUDA device parity plus N2 coarse `r_aux`/retention/coarse-support behavior on certified Fit/Design stay unobserved until the rerun; no quality/capacity claim made.

## 7. Next safe steps (Main-owned)

1. Fresh independent evidence review of `artifacts/sprint-18/task-11-prerequisite-c01.md` plus this handoff plus the 3 owned paths.
2. Separate Bronze source checkpoint on the exact corrected snapshot (no pilots bundled).
3. Frozen 12-run N2 pilot rerun on the fixed source (`alpha` 0.10/0.25/0.50 plus matched B0-FULL/0.05 reference, seeds 181801–3, 30 epochs) through `train_sprint18_msp_epoch`, then `select_msp_coefficient` plus versioned selected-or-null provenance under `experiments/`, with actual target-CUDA confirmation.
4. Final review PASS plus separate Bronze Task11 checkpoint before Task12.

## 8. Yield summary (for Main dispatch)

- `task_id`: `S18-T11`, `attempt_id`: `S18-T11-C01`, `agent_id`: `S18T11C01`, role `gold-task`.
- Outcome: `prerequisite_fix_ready` (PR02 HIGH + MEDIUM closed; pilots pending rerun).
- Owned records: `artifacts/sprint-18/task-11-prerequisite-c01.md`; handoff `handoffs/s18-t11-c01-handoff.md`.
- Owned paths: `src/representation/trainer.py`, `tests/representation/test_sprint18_msp.py`, `CHANGELOG.md`.
- Base: `00e052d3a250f630fbeac02cf6b3a1516a8eefaf`; no worker commit/stage/push/SSH.
- Actual verification: corrected regression 1 passed + 1 honest CUDA skip; full N2 file 8 passed + 1 honest skip; fix bytes match reviewed digest; preserved hashes re-verified.
- Remaining gates: fresh evidence PASS → Bronze source checkpoint → pilot rerun with CUDA confirmation → final PASS → Bronze Task11 checkpoint.
- Pilots run: `0/12`; selected value: `null` (pending rerun).
