# Sprint 18 — Assets to retain before temporary-run cleanup

## Decision and verification boundary

**Do not delete the accepted data, model or evidence roots below as temporary runs.** Sprint 24 reuses the exact Sprint 18 baseline; a JSON digest or a Git commit is not a backup of the referenced binary.

This is a **repository-evidence dependency inventory**, prepared on 2026-10-25. Paths were extracted from the accepted binding, Confirmation lock, final publication and execution/sync records. No live server existence, current size, backup completeness or restore verification was performed for this note. No unlisted directory is certified safe to delete. No deletion or relocation was executed.

[Sprint 24 Task 1](sprint-plans/sprint-24.md) must verify the transitive dependency inventory and any destination archive before cleanup. Preserve source paths until a hash-verified relocation manifest and non-Confirmation restore smoke establish that the new package is usable. Do not edit immutable Sprint 18 locks to conceal relocation.

## 1. Required inherited data

Preserve this entire root, including hidden/custody files, manifests, shards, annotations and `_assessment-001`:

```text
/home/trietlm/anomaly-representation-learning-s18t68-a02-worktree/data/generated/sprint18-iterative-v1/S18-ITER-0005-ASSESS-P-ALLOWANCE-V1/
```

It is the accepted 16-history corpus, not an old rejected candidate. The historical Task 71 proof records approximately **1.8 GiB**, 16 manifests, 1,628 shards and 103,686 samples; these are historical observations, not a current disk audit.

Sources: [certified handoff](../experiments/sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json), [handoff proof](../artifacts/sprint-18/s18-t71-a01-handoff-proof.json), [binding](../experiments/sprint18-iterative-assessment-c5-allowance-v1.json).

**Do not remove the enclosing `anomaly-representation-learning-s18t68-a02-worktree` wholesale.** It contains the data and accepted support records. Retain its source identity, untracked/ignored evidence and any externally referenced files until archived. A clean tracked Git status does not establish that this directory has no unique files.

## 2. Model and downstream artifacts

All directories in the next table are under **`/home/trietlm/`**. Preserve whole roots for now; a future manifest may reduce them to the verified transitive file set.

| Directory to retain | Accepted contents / reason |
|---|---|
| `s18-t23-a04-runs/` | B0-FULL final epoch-100 checkpoints for seeds 181801/181802/181803; primary reusable encoder and N4's paired baseline |
| `s18-t30-a14-runs/` | N4-CRN coherent and final checkpoints for all three seeds; learned scale/scorer state, not a replacement encoder |
| `s18-t32-a02-runs/` | Locked per-arm input manifests, persisted Fit banks and bank provenance, Development scoring support |
| `s18-t24-a02-runs/` | Accepted K3-FULL epoch-100 checkpoints; preserve the Sprint 18 comparison even though not a primary Sprint 24 method |
| `s18-t25-a02-runs/` | Accepted K7-FULL epoch-100 checkpoints |
| `s18-t26-a02-runs/` | Accepted K9-FULL epoch-100 checkpoints **and `k9_downstream.pt`** per seed |
| `s18-t28-a04-runs/` | Accepted N2-MSP epoch-100 checkpoints |
| `s18-t29-a06-runs/` | Accepted N3-CCRP epoch-100 checkpoints |

Authoritative file names and hashes: [Task 33 lock](../experiments/sprint18-task33-confirmation-lock.json). Cross-check the complete [eight-arm publication](../artifacts/sprint-18/s18-t36-a01/final-eight-arm-matrix.json), accepted per-seed records and checkpoint-internal references; extracting top-level roots alone is not a transitive-dependency proof.

**Minimum for the planned B0/N4 reuse is not the same as minimum for preserving Sprint 18.** B0/N4 plus their banks/data enable the primary study, but deleting K/N alternatives would discard accepted scientific artifacts. Archive them rather than treating them as failed temporary attempts.

## 3. Confirmation results and execution evidence

Also retain these directories under **`/home/trietlm/`**:

| Directory to retain | Reason |
|---|---|
| `s18-t34-a02-runs/` | Accepted B0/K3/K7/K9 Confirmation cell documents, score/localization maps, ledger and metrics; preserve original INVALID B0 report alongside authoritative `metrics-fix01.json` |
| `s18-t35-a01-runs/` | Accepted N2/N3/N4 Confirmation results and ledger; executed by attempt A02 using the A01-named output root |
| `s18-t34-a02-runner/` | Runner/catalog provenance referenced by the accepted Task 35 prerequisite record; archive unique files before considering staging cleanup |
| `s18-t35-a01-runner/` | Prepared runner, review copy and pre-exposure `executor-provenance.json`; the latter explains A01/A02 identities |
| `s18-t39-final-evidence-sync01/` | Final immutable evidence bundle and administrative metadata bundle |

Sources: [Task 35 prerequisites](../artifacts/sprint-18/task-35.md), [Task 35 executed record](../artifacts/sprint-18/task-35-runtime-a02.md), [final sync](../artifacts/sprint-18/task-39-final-sync01.md), and the final eight-arm publication.

The final sync reports **29 immutable evidence files**. It is not evidence that the full dataset, all model binaries, banks and bulk scoring outputs were backed up there. Do not keep only that bundle and delete the run roots.

## 4. Repository-local records and reproducibility inputs

Preserve, including ignored files where present:

- `artifacts/sprint-18/`: accepted records, source extracts, support outputs, manifests, releases, historical failures, reviews, checkpoints and sync records. Do not assume all of this is tracked by Git.
- `experiments/sprint18*`: source/configuration/specification lineage, certified handoff and immutable locks; retain the source revisions and environment lockfile they reference.
- `docs/sprint-plans/sprint-18.md`, `docs/sprint18-baseline-and-continual-learning.md`, and the final verdict/publication referenced above.
- Any actual data/checkpoint/cache directory elsewhere that the transitive inventory resolves to a unique required file. This note does not infer that conventional local `checkpoints/` or `data/generated/` paths contain copies of server assets.

## 5. When a directory can be considered for cleanup

A run/staging/cache directory is only a cleanup candidate after all of the following hold:

1. Enumerate actual files, symlinks and checkpoint/cache dependencies; identify every unique accepted or historical evidence artifact. Resolve references outside the directory too.
2. Copy required bytes to durable storage with sufficient space; verify file counts, sizes and SHA-256 at destination, preserving the original hierarchy or an explicit old-path-to-new-path map. A symlink into a soon-to-be-deleted root is not a backup.
3. Preserve model states together with normalization, bank, scaler, configuration, source commit, environment lock and source-data identity. Preserve N4's B0 dependencies and K9 downstream state.
4. Run a real loader/restore/inference smoke against the archived B0/N4 assets using permitted non-Confirmation support, without tuning, rescoring old Confirmation or changing source artifacts. Audit other archived arms' required restore dependencies before releasing their roots too.
5. Record which original paths are now redundant and which remain unique. Obtain explicit deletion approval for the exact allowlist; do not use a blanket `s18-*` or `/tmp/s18*` deletion.

Failed runs, diagnostic caches and runner staging may be reducible **after** this audit, but this note grants no safe-delete list. Sprint 18 already records unrecovered historical A07 binary loss; do not repeat that loss by treating all run directories as disposable.
