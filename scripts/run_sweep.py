#!/usr/bin/env python3
"""
Run the full sweep pipeline (data -> probe training -> eval -> plots).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import gc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sweep pipeline end-to-end.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_sweep.yaml",
        help="Path to sweep configuration YAML file.",
    )
    parser.add_argument(
        "--doc-batch-size",
        type=int,
        default=1,
        help="Document batch size for train/eval extraction.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="device_map for model loading (e.g. 'auto').",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip dataset generation step.",
    )
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _cleanup_gpu_memory(stage: str) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as exc:
        logger.warning("GPU cleanup skipped after %s: %s", stage, exc)
    finally:
        gc.collect()
        logger.info("Memory cleanup complete after %s", stage)


def main() -> int:
    args = parse_args()
    config_path = args.config
    python = sys.executable

    if not args.skip_generate:
        _run([python, "scripts/generate_data.py", "--config", config_path])
        _cleanup_gpu_memory("data_generation")

    train_cmd = [
        python,
        "scripts/train_probe.py",
        "--config",
        config_path,
        "--doc-batch-size",
        str(args.doc_batch_size),
    ]
    if args.device_map:
        train_cmd.extend(["--device-map", args.device_map])
    _run(train_cmd)
    _cleanup_gpu_memory("probe_training")

    eval_cmd = [
        python,
        "scripts/run_eval.py",
        "--config",
        config_path,
        "--doc-batch-size",
        str(args.doc_batch_size),
    ]
    if args.device_map:
        eval_cmd.extend(["--device-map", args.device_map])
    _run(eval_cmd)
    _cleanup_gpu_memory("evaluation")

    _run([python, "scripts/make_plots.py", "--config", config_path])
    _cleanup_gpu_memory("plotting")

    logger.info("Sweep pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
