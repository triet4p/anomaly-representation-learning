"""Automated generation, packaging, and metadata setup for Kaggle dataset upload."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package dataset and source code for Kaggle upload"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/kaggle-20260903-01"),
    )
    parser.add_argument(
        "--slug",
        type=str,
        default="trietp1253201581/anomaly-representation-20260903-01",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Anomaly Representation 20260903-01",
    )
    parser.add_argument("--train-count", type=int, default=25000)
    parser.add_argument("--val-count", type=int, default=5000)
    parser.add_argument("--test-count", type=int, default=20000)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip dataset generation if shards already exist",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = args.output if args.output.is_absolute() else repo_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    # Step 1: Generate dataset if needed
    if not args.skip_generate or not manifest_path.is_file():
        print(f"=== Step 1: Generating dataset into {output_dir} ===")
        cmd = [
            sys.executable,
            "-m",
            "synth.cli",
            "--output",
            str(output_dir),
            "--train-count",
            str(args.train_count),
            "--val-count",
            str(args.val_count),
            "--test-count",
            str(args.test_count),
            "--shard-size",
            str(args.shard_size),
            "--channels",
            str(args.channels),
            "--seed",
            str(args.seed),
            "--overwrite",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        print(f"Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=str(repo_root), env=env)
        if res.returncode != 0:
            print("Error: Dataset generation failed.")
            return res.returncode
    else:
        print(f"Skipping dataset generation (found existing {manifest_path}).")

    # Step 2: Copy src directory into dataset package
    print("=== Step 2: Bundling src/ modules into dataset ===")
    src_origin = repo_root / "src"
    src_dest = output_dir / "src"
    if src_dest.is_dir():
        shutil.rmtree(src_dest)
    shutil.copytree(src_origin, src_dest)
    print(f"Copied {src_origin} -> {src_dest}")

    # Step 3: Write dataset-metadata.json
    print("=== Step 3: Writing dataset-metadata.json ===")
    metadata = {
        "title": args.title,
        "id": args.slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = output_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata to {meta_path}")

    # Step 4: Ensure Kaggle kernel directory is synchronized
    print("=== Step 4: Synchronizing notebooks/kaggle/ ===")
    kaggle_kernel_dir = repo_root / "notebooks" / "kaggle"
    kaggle_kernel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        repo_root / "notebooks" / "train_v1_representation.ipynb",
        kaggle_kernel_dir / "train_v1_representation.ipynb",
    )
    kernel_meta = {
        "id": f"{args.slug.split('/')[0]}/anomaly-representation-training-gpu",
        "title": "Anomaly Representation Training GPU",
        "code_file": "train_v1_representation.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_gpu": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [args.slug],
        "kernel_sources": [],
    }
    (kaggle_kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel_meta, indent=2), encoding="utf-8"
    )
    print("Kaggle kernel files ready in", kaggle_kernel_dir)

    print("\nDataset package is ready for upload:")
    print(f"  Location: {output_dir}")
    print("  Upload command:")
    print(f"    cd {output_dir} && kaggle datasets create -p . -r zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
