"""Command line entry point for bounded synthetic dataset generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from synth.config import SynthConfig
from synth.dataset import DatasetBuilder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate coherent synthetic anomaly data")
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
    p.add_argument("--channels", type=int, choices=(3, 6), default=6)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
