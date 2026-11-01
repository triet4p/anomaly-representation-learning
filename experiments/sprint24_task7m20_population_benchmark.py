"""Public CLI for the Sprint 24 Task 7-M20 population benchmark.

Modes::

    # design manifest + resource forecast (no model)
    python experiments/sprint24_task7m20_population_benchmark.py --mode design --out <fresh dir>

    # frozen contract (written to the run dir and docs/sprint24-population-benchmark-contract-a01.json)
    python experiments/sprint24_task7m20_population_benchmark.py --mode contract --out <fresh dir>

    # real tiny smoke: accepted bundles -> all readouts -> disk -> fresh replay -> decision
    python experiments/sprint24_task7m20_population_benchmark.py --mode smoke --out <fresh dir> \
        --b0-root <accepted b0 dir> --bank-root <accepted bank dir> \
        --preflight <verified preflight.json> --device cpu --attempt-id S24-7-M20-A01

    # full frozen benchmark (refuses without an explicit matching release manifest)
    python experiments/sprint24_task7m20_population_benchmark.py --mode official --out <fresh dir> \
        --permit-official --release-manifest <released.json> \
        --b0-root <dir> --bank-root <dir> --preflight <json> --device cuda:0 --attempt-id <id>

    # fresh-process, encoder-free replay of any completed run directory
    python experiments/sprint24_task7m20_population_benchmark.py --mode replay --run <run dir> --out <dir>

Exit codes: 0 success, 1 refusal/failure (the message names the closed refusal
reason). The ``replay`` mode installs an import guard that forbids ``torch``
and ``representation`` before loading any benchmark module, so a replay can
never invoke an encoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

BLOCKED_MODULES = ("torch", "representation")


class _ReplayImportGuard:
    """Meta-path finder that refuses encoder-capable imports during replay."""

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        root = fullname.split(".")[0]
        if root in BLOCKED_MODULES:
            raise ImportError(f"replay forbids importing {fullname!r} (encoder-free custody)")
        return None


def _install_replay_guard() -> None:
    guard = _ReplayImportGuard()
    sys.meta_path.insert(0, guard)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=("design", "contract", "smoke", "official", "replay"))
    parser.add_argument("--out", default=None, help="explicit run/output directory (fresh unless --resume)")
    parser.add_argument("--run", default=None, help="existing run directory for --mode replay")
    parser.add_argument("--b0-root", default=None)
    parser.add_argument("--bank-root", default=None)
    parser.add_argument("--preflight", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--permit-official", dest="permit_official", action="store_true", default=False)
    parser.add_argument("--release-manifest", dest="release_manifest", default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    return parser


def _require_out(args: argparse.Namespace) -> Path:
    if not args.out:
        raise ValueError("REFUSED-PREREQUISITE: --out is required")
    return Path(args.out)


def _emit(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "replay":
            _install_replay_guard()
            from population_benchmark import runner

            if not args.run:
                raise ValueError("REFUSED-PREREQUISITE: --mode replay needs --run")
            out = Path(args.out) if args.out else None
            result = runner.replay_run(Path(args.run), out)
            _emit(result)
            return 0
        if args.mode == "design":
            from population_benchmark import runner, spec

            out = _require_out(args)
            if out.exists() and any(out.iterdir()):
                raise ValueError(f"REFUSED-OVERWRITE: {out} is non-empty")
            out.mkdir(parents=True, exist_ok=True)
            payload = runner.write_design(out, spec.official_plan())
            _emit(payload)
            return 0
        if args.mode == "contract":
            from population_benchmark import runner, spec, storage

            out = _require_out(args)
            if out.exists() and any(out.iterdir()):
                raise ValueError(f"REFUSED-OVERWRITE: {out} is non-empty")
            out.mkdir(parents=True, exist_ok=True)
            payload = runner.write_contract(out)
            verification = runner.verify_contract_file(Path(payload["path"]))
            _emit({
                **payload,
                "verification": verification,
                "design": spec.design_payload()["plan"],
                "contract_file": str(runner.DOCS_CONTRACT),
                "contract_file_sha256": storage.sha256_file(runner.DOCS_CONTRACT),
            })
            return 0
        if args.mode == "smoke":
            from population_benchmark import runner

            out = _require_out(args)
            result = runner.run_smoke(out, args)
            _emit({
                "status": result["status"],
                "forwards_executed_this_process": result["forwards_executed_this_process"],
                "model_loads": result["model_loads"],
                "decision": result["decision"]["action"],
                "replay": result["replay"]["status"],
                "behavior_pass": result["behavior"]["all_pass"],
                "smoke_record": str(out / "smoke.json"),
                "elapsed_s": result["elapsed_s"],
            })
            return 0
        if args.mode == "official":
            from population_benchmark import runner

            out = _require_out(args)
            result = runner.run_official(out, args)
            _emit({
                "status": result["status"],
                "decision": result["decision"],
                "forwards_executed_this_process": result["forwards_executed_this_process"],
                "replay_status": result["replay_status"],
                "run_record": str(out / "run.json"),
                "elapsed_s": result["elapsed_s"],
            })
            return 0
        raise ValueError(f"REFUSED-UNKNOWN-DESIGN: unknown mode {args.mode!r}")
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report the closed reason
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
