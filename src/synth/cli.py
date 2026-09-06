"""Command line entry point for bounded synthetic dataset generation."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path


_DISTRIBUTION_NAME = "anomaly-representation-learning"


def _load_generation_dependencies():
    from synth.config import SynthConfig
    from synth.dataset import DatasetBuilder

    return SynthConfig, DatasetBuilder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate coherent synthetic anomaly data")
    p.add_argument(
        "--version",
        action="version",
        version=version(_DISTRIBUTION_NAME),
        help="show the project version and exit",
    )
    p.add_argument("--output", type=Path, required=True, help="output directory")
    p.add_argument("--seed", type=int, default=None, help="base seed for disjoint splits")
    p.add_argument("--count", type=int, default=None, help="set all split counts")
    p.add_argument("--train-count", type=int, default=None)
    p.add_argument("--val-count", type=int, default=None)
    p.add_argument("--test-count", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=512)
    p.add_argument("--resume", action="store_true", help="resume verified completed shards")
    p.add_argument("--overwrite", action="store_true", help="replace an existing manifest")
    p.add_argument("--small", action="store_true", help="small smoke profile (12/8/12 samples)")
    p.add_argument("--chronological", action="store_true",
                   help="materialize a chronological factory dataset (Tasks 5-7 pipeline)")
    p.add_argument("--profile", choices=("client", "server"), default="client",
                   help="chronological scale profile (default: client)")
    p.add_argument("--channels", type=int, choices=(3, 6), default=6)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chronological:
        for flag in ("count", "train_count", "val_count", "test_count"):
            if getattr(args, flag) is not None:
                build_parser().error(
                    f"--{flag.replace('_', '-')} cannot be combined with --chronological"
                )
        if args.small:
            build_parser().error("--small cannot be combined with --chronological")
        from synth.chronicle import (
            client_config,
            materialize_chronological,
            server_config,
        )
        seed = 0 if args.seed is None else args.seed
        cfg = client_config(seed=seed) if args.profile == "client" else server_config(seed=seed)
        cfg.n_channels = args.channels
        manifest = materialize_chronological(
            cfg, args.output, shard_size=args.shard_size,
            overwrite=args.overwrite,
        )
        counts = manifest["counts"]
        print(f"wrote {args.output / 'manifest.json'}: "
              f"total={counts['total']} normal={counts['normal']} "
              f"abnormal={counts['abnormal']} dev_train={counts['dev_train']} "
              f"dev_val={counts['dev_val']} static={counts['test_static']} "
              f"temporal={counts['test_temporal']}")
        return 0
    SynthConfig, DatasetBuilder = _load_generation_dependencies()
    cfg = SynthConfig(n_channels=args.channels)
    if args.seed is not None:
        cfg.split.train_seed = args.seed
        cfg.split.val_seed = args.seed + 1_000_000
        cfg.split.test_seed = args.seed + 2_000_000
    if args.count is not None:
        cfg.split.n_train = cfg.split.n_val = cfg.split.n_test = args.count
    if args.train_count is not None:
        cfg.split.n_train = args.train_count
    if args.val_count is not None:
        cfg.split.n_val = args.val_count
    if args.test_count is not None:
        cfg.split.n_test = args.test_count
    manifest = DatasetBuilder(cfg).materialize_sharded(
        args.output, shard_size=args.shard_size, resume=args.resume,
        overwrite=args.overwrite, profile="small" if args.small else None,
    )
    counts = manifest["counts"]
    print(f"wrote {args.output / 'manifest.json'}: "
          f"train={counts['train']} val={counts['val']} test={counts['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
