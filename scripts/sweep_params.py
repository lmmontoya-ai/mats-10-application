#!/usr/bin/env python3
"""
Parameter sensitivity sweep for aggregator hyperparameters.

This script evaluates different EMA alpha and window size values
to demonstrate robustness of the best aggregators.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.config import DatasetConfig, load_config
from src.dataset.io import DatasetReader
from src.dataset.schema import DatasetExample
from src.dataset.tokenizer_utils import TokenizerWrapper
from src.eval.aggregators import ema_max_pool, windowed_max_mean_pool
from src.eval.calibration import fit_platt
from src.eval.metrics import (
    compute_fpr_tpr,
    compute_intrinsic_fpr_at_tpr,
    threshold_at_tpr,
)
from src.probes.activation import ActivationExtractor
from src.probes.utils import (
    disable_tokenizers_parallelism,
    resolve_device,
    resolve_dtype,
    resolve_input_device,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep aggregator hyperparameters for sensitivity analysis.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_main.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--probe-path",
        type=str,
        required=True,
        help="Path to trained probe weights (.pt file).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index for the probe.",
    )
    parser.add_argument(
        "--length-bucket",
        type=int,
        default=None,
        help="Length bucket to evaluate (default: longest).",
    )
    parser.add_argument(
        "--distractor-level",
        type=int,
        default=None,
        help="Distractor level to evaluate (default: highest).",
    )
    parser.add_argument(
        "--ema-alphas",
        type=str,
        default="0.05,0.1,0.15,0.2,0.25,0.3",
        help="Comma-separated EMA alpha values to sweep.",
    )
    parser.add_argument(
        "--window-sizes",
        type=str,
        default="32,64,128,256,512",
        help="Comma-separated window sizes to sweep.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/analysis/param_sweep.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def _get_dataset_path(config: DatasetConfig) -> Path:
    return (
        Path(config.output.output_root)
        / "datasets"
        / f"{config.output.dataset_name}_{config.output.dataset_version}.jsonl"
    )


def _load_examples(config: DatasetConfig) -> list[DatasetExample]:
    dataset_path = _get_dataset_path(config)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return list(DatasetReader.read_jsonl(dataset_path))


def _filter_examples(
    examples: list[DatasetExample],
    split: str,
    length_bucket: int | None,
    distractor_level: int | None,
) -> list[DatasetExample]:
    filtered = [ex for ex in examples if ex.split == split]
    if length_bucket is not None:
        filtered = [ex for ex in filtered if ex.length_bucket == length_bucket]
    if distractor_level is not None:
        filtered = [ex for ex in filtered if ex.distractor_level == distractor_level]
    return filtered


def _build_linear_from_state(
    state: dict[str, torch.Tensor], device: torch.device
) -> torch.nn.Module:
    weight = state["weight"]
    in_dim = weight.shape[1]
    linear = torch.nn.Linear(in_dim, 1)
    linear.load_state_dict(state)
    linear.to(device)
    linear.eval()
    return linear


def _score_documents(
    examples: list[DatasetExample],
    extractor: ActivationExtractor,
    linear: torch.nn.Module,
    tokenizer: TokenizerWrapper,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get per-token logits for all documents."""
    all_logits = []
    all_masks = []
    labels = []

    for ex in tqdm(examples, desc="Scoring documents", unit="docs"):
        enc = tokenizer.tokenizer(
            ex.text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )
        input_ids = enc["input_ids"].to(extractor.device)
        attention_mask = enc["attention_mask"].to(extractor.device)

        with torch.inference_mode():
            hidden = extractor.extract_hidden_states(input_ids, attention_mask)
            if (
                linear.weight.dtype != hidden.dtype
                or linear.weight.device != hidden.device
            ):
                linear.to(device=hidden.device, dtype=hidden.dtype)
            logits = linear(hidden).squeeze(-1)  # [1, seq_len]

        all_logits.append(logits.cpu())
        all_masks.append(attention_mask.bool().cpu())
        labels.append(int(ex.needle_present))

    return all_logits, all_masks, torch.tensor(labels, dtype=torch.float32)


def main() -> int:
    args = parse_args()
    disable_tokenizers_parallelism()

    logger.info("Loading configuration from %s", args.config)
    config = load_config(args.config)

    ema_alphas = [float(x) for x in args.ema_alphas.split(",")]
    window_sizes = [int(x) for x in args.window_sizes.split(",")]

    length_bucket = args.length_bucket or max(config.grid.length_buckets_tokens)
    distractor_level = args.distractor_level
    if distractor_level is None:
        distractor_level = max(config.grid.distractor_levels)

    tokenizer = TokenizerWrapper(config.tokenizer)
    examples = _load_examples(config)

    val_examples = _filter_examples(examples, "val", length_bucket, distractor_level)
    test_examples = _filter_examples(examples, "test", length_bucket, distractor_level)

    if not val_examples or not test_examples:
        logger.error(
            "No examples found for length=%s, distractor=%s",
            length_bucket,
            distractor_level,
        )
        return 1

    device = resolve_device(config.probe.device)
    model_dtype = resolve_dtype(config.probe.model_dtype, device)

    logger.info("Loading model %s", config.tokenizer.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.tokenizer.model_id,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    input_device = resolve_input_device(model, fallback=device)

    probe_state = torch.load(args.probe_path, map_location="cpu")
    linear = _build_linear_from_state(probe_state, input_device)

    extractor = ActivationExtractor(
        model=model,
        layer_idx=args.layer,
        activation_site=config.probe.activation_site,
        device=input_device,
        feature_dtype=torch.float32,
    )

    logger.info("Scoring validation documents...")
    val_logits, val_masks, val_labels = _score_documents(
        val_examples, extractor, linear, tokenizer
    )

    logger.info("Scoring test documents...")
    test_logits, test_masks, test_labels = _score_documents(
        test_examples, extractor, linear, tokenizer
    )

    results = []
    target_tpr = config.eval.target_tprs[0]

    # Sweep EMA alpha
    logger.info("Sweeping EMA alpha values...")
    for alpha in ema_alphas:
        val_scores = []
        for logits, mask in zip(val_logits, val_masks):
            output = ema_max_pool(logits, mask, alpha)
            val_scores.append(output.scores.item())
        val_scores_t = torch.tensor(val_scores, dtype=torch.float32)

        test_scores = []
        for logits, mask in zip(test_logits, test_masks):
            output = ema_max_pool(logits, mask, alpha)
            test_scores.append(output.scores.item())
        test_scores_t = torch.tensor(test_scores, dtype=torch.float32)

        # Calibrate
        calibrator = fit_platt(val_scores_t, val_labels)
        val_probs = calibrator.predict_proba(val_scores_t)
        test_probs = calibrator.predict_proba(test_scores_t)

        threshold = threshold_at_tpr(val_probs, val_labels, target_tpr)
        fpr_fixed, tpr_fixed = compute_fpr_tpr(test_probs, test_labels, threshold)
        fpr_intrinsic = compute_intrinsic_fpr_at_tpr(
            test_scores_t, test_labels, target_tpr
        )

        results.append(
            {
                "aggregator": "ema_max",
                "param_name": "alpha",
                "param_value": alpha,
                "length_bucket": length_bucket,
                "distractor_level": distractor_level,
                "fpr_intrinsic": fpr_intrinsic,
                "fpr_fixed": fpr_fixed,
                "tpr_fixed": tpr_fixed,
                "n_test": len(test_examples),
            }
        )

    # Sweep window size
    logger.info("Sweeping window size values...")
    for window_size in window_sizes:
        val_scores = []
        for logits, mask in zip(val_logits, val_masks):
            output = windowed_max_mean_pool(logits, mask, window_size)
            val_scores.append(output.scores.item())
        val_scores_t = torch.tensor(val_scores, dtype=torch.float32)

        test_scores = []
        for logits, mask in zip(test_logits, test_masks):
            output = windowed_max_mean_pool(logits, mask, window_size)
            test_scores.append(output.scores.item())
        test_scores_t = torch.tensor(test_scores, dtype=torch.float32)

        # Calibrate
        calibrator = fit_platt(val_scores_t, val_labels)
        val_probs = calibrator.predict_proba(val_scores_t)
        test_probs = calibrator.predict_proba(test_scores_t)

        threshold = threshold_at_tpr(val_probs, val_labels, target_tpr)
        fpr_fixed, tpr_fixed = compute_fpr_tpr(test_probs, test_labels, threshold)
        fpr_intrinsic = compute_intrinsic_fpr_at_tpr(
            test_scores_t, test_labels, target_tpr
        )

        results.append(
            {
                "aggregator": "windowed_max_mean",
                "param_name": "window_size",
                "param_value": window_size,
                "length_bucket": length_bucket,
                "distractor_level": distractor_level,
                "fpr_intrinsic": fpr_intrinsic,
                "fpr_fixed": fpr_fixed,
                "tpr_fixed": tpr_fixed,
                "n_test": len(test_examples),
            }
        )

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info("Parameter sweep results written to %s", output_path)

    # Print summary
    print("\nParameter Sensitivity Summary:")
    print("=" * 60)
    print(f"Length bucket: {length_bucket}, Distractor level: {distractor_level}")
    print()
    print("EMA Alpha Sweep:")
    for r in results:
        if r["aggregator"] == "ema_max":
            print(f"  alpha={r['param_value']:.2f}: FPR={r['fpr_intrinsic']:.4f}")
    print()
    print("Window Size Sweep:")
    for r in results:
        if r["aggregator"] == "windowed_max_mean":
            print(f"  window={r['param_value']:4d}: FPR={r['fpr_intrinsic']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
