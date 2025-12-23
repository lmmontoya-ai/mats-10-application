#!/usr/bin/env python3
"""
Download the generated dataset into the local data/ directory.

This script pulls the Task 1 dataset from the configured Hugging Face dataset
repo and writes it to:
  data/datasets/
  data/splits/
  data/qa/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "Luxel/needle-haystack-v1"
DEFAULT_REVISION = "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Task 1 dataset artifacts into data/.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo to download (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_REVISION,
        help="HF dataset repo revision (default: main)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    local_dir = Path("data")
    local_dir.mkdir(parents=True, exist_ok=True)

    existing = (local_dir / "datasets").exists()
    if existing and not args.force:
        print("data/datasets already exists. Use --force to re-download.")
        return 0

    print(f"Downloading dataset from {args.repo_id} ({args.revision})...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.git*"],
    )
    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
