# Sprint 10 Task 1 — Commit-or-ignore audit (experiments/20260905)

## Server read-only inventory (trietlm@192.168.30.244, commit d0af9e5 at audit time)

- `01/` — Sprint 6/8 geometry work (others'); `server-geometry-run-1-*/` — same family. Pre-existing
  unrelated; left untouched (tracked or untracked as found).
- `02/` — predecessor Task 3 partial (2 executed + 2 source notebooks + 2 small logs, run at 261f6f9).
  SUPERSEDED by `03/` reruns at exact commit d0af9e5; kept on server, ignored by explicit rule
  `experiments/20260905/02/` (never silently deleted; reason recorded here).
- `03/` — authoritative Task 3 reruns: `infer-{medium,production-head}.ipynb` (method),
  `infer-{medium,production-head}.executed.ipynb` (outputs), `medium_stdout.log` (464 B, exit 0),
  `production_head_stdout.log` (481 B, exit 0). COMMITTED (md5-verified identical both sides).
- `04/` — `dump_scores.py` COMMITTED; `dump_stdout.log` (31 KB tqdm noise) ignored by rule;
  `scores_medium.npz` (12.1 MB) ignored by pre-existing `*.npz` rule.
- `05/` — COMMITTED: `train_ablation.py`, `loss_split.py`, `run_grid.sh`, `eval_grid.sh`,
  `summarize_histories.py`, 6× `ckpt_*_history.json` (51–56 KB), `grid_master.log`, `eval_all.log`,
  `smoke_history.json`. Ignored: 6× `ckpt_*.pt` (~19.8 MB each, pre-existing `*.pt` rule),
  6× `scores_*.npz` (~12.1 MB each, `*.npz` rule), `train_*.log`/`dump_*.log`/`smoke.log`
  (31–740 KB tqdm noise, new explicit rules).
- `06/` — `task7_eval.py`, `task7_summary.json` (1032 B) COMMITTED; `task7.log` (62 KB) and
  `task7_mixed.npz` (70 KB) ignored by rule.
- Local `02-local-tmp/infer-medium.executed.ipynb` was a predecessor download, md5-identical to the
  superseded server `02/` copy (ead214b0…) and NOT the authoritative `03/` rerun (0d69f2e0…);
  removed locally (local-only, nothing of record lost — server `02/` retained).

## .gitignore additions (rules, not omissions)

- `experiments/20260905/02/` (superseded partial, reason above).
- `experiments/20260905/04/dump_stdout.log`, `experiments/20260905/05/train_*.log`,
  `experiments/20260905/05/dump_*.log`, `experiments/20260905/05/smoke.log`,
  `experiments/20260905/06/task7.log` (long raw logs; exit codes/timings live in
  artifacts/sprint-9/task-*.md + committed master logs). `*.npz`/`*.pt` were already ignored.

## Both-side proof

- Local commit `978370b`→`<task1-commit>` (this evidence + experiment record), pushed; server
  `git pull --ff-only` to the same hash. `git status --porcelain` on both sides shows no
  uncommitted/unignored file under `experiments/` or `artifacts/sprint-9/`; remaining entries are
  pre-existing unrelated user files (CHANGELOG.md, docs/PLAN.md, docs/sprint-plans/sprint-6.md,
  `artifacts/task_*_summary.md` deletions, `.agents/`, `artifacts/sprint-{2,4,6,8}/`,
  `experiments/20260904/`, `experiments/20260905/01` additions, `docs/sprint-plans/sprint-{3,4,5,8}.md`,
  server `01/`-geometry extras) — all untouched.
- Exact status outputs: <paste both `git status --porcelain` outputs at close>.
