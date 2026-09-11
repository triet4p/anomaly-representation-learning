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
    p.add_argument("--profile", choices=("client", "server", "sprint13",
                                         "sprint13-v41", "sprint14-v3",
                                         "sprint14-v5", "sprint15-v1",
                                         "sprint15-v2", "sprint15-v3",
                                         "sprint15-v4", "sprint15-v5",
                                         "sprint15-v6", "sprint15-v7"),
                   default="client",
                   help="chronological scale profile (default: client)")
    p.add_argument("--role", type=str, default=None,
                   help="whole-history role recorded in the manifest "
                   "(Sprint 13 Task 6 roster; default: unassigned)")
    p.add_argument("--protocol", type=str, default=None,
                   help="frozen benchmark version recorded in the manifest "
                   "(default: profile default)")
    p.add_argument("--channels", type=int, choices=(3, 6), default=6)
    p.add_argument("--units", type=int, default=None,
                   help="chronological unit count; arrival cadence scales inversely "
                   "to preserve the profile calendar span (default: profile value)")
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
            sprint13_history_config,
            sprint13_v41_history_config,
            sprint14_v3_history_config,
            sprint14_v5_history_config,
            sprint15_v1_history_config,
            sprint15_v2_history_config,
            sprint15_v3_history_config,
            sprint15_v4_history_config,
            sprint15_v5_history_config,
            sprint15_v6_history_config,
            sprint15_v7_history_config,
        )
        seed = 0 if args.seed is None else args.seed
        if args.profile == "client":
            cfg = client_config(seed=seed)
        elif args.profile == "server":
            cfg = server_config(seed=seed)
        elif args.profile == "sprint13-v41":
            cfg = sprint13_v41_history_config(seed=seed)
            if args.protocol is None:
                args.protocol = "sprint13-protocol-v4.1"
        elif args.profile == "sprint14-v3":
            cfg = sprint14_v3_history_config(seed=seed)
            if args.protocol is None:
                args.protocol = "sprint14-benchmark-protocol-v3"
        elif args.profile == "sprint14-v5":
            cfg = sprint14_v5_history_config(seed=seed)
            if args.protocol is None:
                args.protocol = "sprint14-benchmark-protocol-v5"
        elif args.profile == "sprint15-v1":
            cfg = sprint15_v1_history_config(seed=seed)
            if args.protocol is None:
                args.protocol = "sprint15-benchmark-protocol-v1"
        elif args.profile == "sprint15-v2":
            cfg = sprint15_v2_history_config(seed=seed)
            if args.protocol is None:
                args.protocol = "sprint15-benchmark-protocol-v2"
        elif args.profile == "sprint15-v3":
            cfg = sprint15_v3_history_config(seed=seed)
            if args.protocol is None:
                args.protocol = "sprint15-benchmark-protocol-v3"
        elif args.profile == "sprint15-v4":
            cfg = sprint15_v4_history_config(seed=seed)
            if args.protocol is None:
                args.protocol = "sprint15-benchmark-protocol-v4"
        elif args.profile == "sprint15-v5":
            cfg = sprint15_v5_history_config(seed=seed)
            if args.protocol is None:
                args.protocol = "sprint15-benchmark-protocol-v5"
        elif args.profile == "sprint15-v6":
            cfg = sprint15_v6_history_config(seed=seed)
            if args.protocol is None:
                args.protocol = "sprint15-benchmark-protocol-v6"
        elif args.profile == "sprint15-v7":
            cfg = sprint15_v7_history_config(seed=seed)
            if args.protocol is None:
                args.protocol = "sprint15-benchmark-protocol-v7"
        else:
            cfg = sprint13_history_config(seed=seed)
        cfg.n_channels = args.channels
        if args.units is not None:
            if args.units <= 0:
                build_parser().error("--units must be positive")
            base_units = cfg.scheduler.n_units
            base_interval = cfg.scheduler.arrival_interval_s
            cfg.scheduler.n_units = args.units
            scaled = base_interval * base_units / args.units
            cfg.scheduler.arrival_interval_s = scaled
            cfg.scheduler.arrival_jitter_s = scaled
        from synth.balanced import S15_PROFILE_V2, S15_PROTOCOL_V2
        from synth.balanced import S15_PROFILE_V3, S15_PROTOCOL_V3
        from synth.balanced import S15_PROFILE_V4, S15_PROTOCOL_V4
        from synth.balanced import S15_PROFILE_V5, S15_PROTOCOL_V5
        from synth.balanced import S15_PROFILE_V6, S15_PROTOCOL_V6
        from synth.balanced import S15_PROFILE_V7, S15_PROTOCOL_V7
        from synth.balanced import Sprint15Binding

        if args.profile == "sprint15-v7":
            binding = Sprint15Binding(profile=S15_PROFILE_V7,
                                      protocol=S15_PROTOCOL_V7,
                                      method="exact")
        elif args.profile == "sprint15-v6":
            binding = Sprint15Binding(profile=S15_PROFILE_V6,
                                      protocol=S15_PROTOCOL_V6,
                                      method="exact")
        elif args.profile == "sprint15-v5":
            binding = Sprint15Binding(profile=S15_PROFILE_V5,
                                      protocol=S15_PROTOCOL_V5,
                                      method="exact")
        elif args.profile == "sprint15-v4":
            binding = Sprint15Binding(profile=S15_PROFILE_V4,
                                      protocol=S15_PROTOCOL_V4,
                                      method="exact")
        elif args.profile == "sprint15-v3":
            binding = Sprint15Binding(profile=S15_PROFILE_V3,
                                      protocol=S15_PROTOCOL_V3)
        elif args.profile == "sprint15-v2":
            binding = Sprint15Binding(profile=S15_PROFILE_V2,
                                      protocol=S15_PROTOCOL_V2)
        else:
            binding = (Sprint15Binding()
                       if args.profile == "sprint15-v1" else None)
        manifest = materialize_chronological(
            cfg, args.output, shard_size=args.shard_size,
            overwrite=args.overwrite, role=args.role,
            protocol=args.protocol,
            sprint15=binding,
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
