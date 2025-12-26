#!/usr/bin/env python3
"""
Evaluate document-level monitoring aggregators using trained probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.config import DatasetConfig, load_config
from src.dataset.io import DatasetReader
from src.dataset.schema import DatasetExample
from src.dataset.tokenizer_utils import TokenizerWrapper
from src.eval.aggregators import AggregatorOutput, build_aggregators
from src.eval.calibration import IdentityCalibrator, fit_platt
from src.eval.metrics import (
    compute_fpr_tpr,
    compute_full_metrics,
    compute_intrinsic_fpr_at_tpr,
    compute_metrics,
    threshold_at_tpr,
)
from src.probes.activation import ActivationExtractor
from src.probes.utils import (
    disable_tokenizers_parallelism,
    resolve_device,
    resolve_dtype,
)
from src.probes.utils import resolve_input_device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained probes and document-level aggregators.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_main.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--probe-dir",
        type=str,
        default=None,
        help="Path to probe directory (defaults to results/probes/{run_id}).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to evaluate.",
    )
    parser.add_argument(
        "--doc-batch-size",
        type=int,
        default=None,
        help="Override eval_doc_batch_size from config.",
    )
    parser.add_argument(
        "--max-eval-docs",
        type=int,
        default=None,
        help="Override max_eval_docs from config.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Hugging Face device_map for model loading (e.g. 'auto').",
    )
    return parser.parse_args()


def _get_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


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
    seed: int,
    max_docs: int | None,
) -> list[DatasetExample]:
    filtered = [ex for ex in examples if ex.split == split]
    if max_docs is not None and len(filtered) > max_docs:
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(filtered), generator=rng).tolist()
        filtered = [filtered[i] for i in perm[:max_docs]]
    return filtered


def _parse_layers(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def _load_probe_infos(
    probe_dir: Path, allowed_layers: list[int] | None
) -> list[dict[str, object]]:
    pattern = re.compile(r"probe_layer_(\d+)_seed_(\d+)\.pt")
    infos: list[dict[str, object]] = []
    for path in sorted(probe_dir.glob("probe_layer_*_seed_*.pt")):
        match = pattern.search(path.name)
        if not match:
            continue
        layer_idx = int(match.group(1))
        seed = int(match.group(2))
        if allowed_layers and layer_idx not in allowed_layers:
            continue
        meta_path = path.with_suffix(".json")
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        infos.append(
            {
                "layer": layer_idx,
                "seed": seed,
                "path": path,
                "metadata": metadata,
            }
        )
    if not infos:
        raise FileNotFoundError(f"No probe weights found in {probe_dir}")
    return infos


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


def _score_split(
    examples: list[DatasetExample],
    split: str,
    extractor: ActivationExtractor,
    linear: torch.nn.Module,
    tokenizer: TokenizerWrapper,
    aggregators: dict[str, callable],
    doc_batch_size: int,
    progress_desc: str,
    capture_max_index: bool = False,
    diagnostic_threshold: float | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    progress = tqdm(
        total=len(examples),
        desc=progress_desc,
        unit="docs",
        dynamic_ncols=True,
    )
    for start in range(0, len(examples), doc_batch_size):
        batch_examples = examples[start : start + doc_batch_size]
        texts = [ex.text for ex in batch_examples]
        enc = tokenizer.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
            padding=True,
        )
        input_ids = enc["input_ids"].to(extractor.device)
        attention_mask = enc["attention_mask"].to(extractor.device)
        mask = attention_mask.bool()

        with torch.inference_mode():
            hidden = extractor.extract_hidden_states(input_ids, attention_mask)
            if (
                linear.weight.dtype != hidden.dtype
                or linear.weight.device != hidden.device
            ):
                linear.to(device=hidden.device, dtype=hidden.dtype)
            logits = linear(hidden).squeeze(-1)

            agg_outputs: dict[str, AggregatorOutput] = {}
            for name, fn in aggregators.items():
                agg_outputs[name] = fn(logits, mask)

            diagnostic_max = None
            diagnostic_second = None
            diagnostic_counts = None
            diagnostic_lengths = None
            if diagnostic_threshold is not None:
                probs = torch.sigmoid(logits)
                masked_probs = probs.masked_fill(~mask, -1.0)
                if masked_probs.shape[1] < 2:
                    top1 = masked_probs.max(dim=1).values
                    top2 = torch.stack([top1, torch.full_like(top1, -1.0)], dim=1)
                else:
                    top2 = torch.topk(masked_probs, k=2, dim=1).values
                lengths = mask.sum(dim=1)
                second_max = torch.where(
                    lengths > 1, top2[:, 1], torch.zeros_like(top2[:, 1])
                )
                counts = ((probs >= diagnostic_threshold) & mask).sum(dim=1)
                diagnostic_max = top2[:, 0]
                diagnostic_second = second_max
                diagnostic_counts = counts
                diagnostic_lengths = lengths

            max_indices = None
            max_probs = None
            if capture_max_index and "max" in agg_outputs:
                max_indices = agg_outputs["max"].extras.get("max_token_index")
                max_probs = agg_outputs["max"].extras.get("max_token_prob")

        for idx, ex in enumerate(batch_examples):
            row: dict[str, object] = {
                "id": ex.id,
                "split": split,
                "length_bucket": ex.length_bucket,
                "distractor_level": ex.distractor_level,
                "needle_present": int(ex.needle_present),
                "needle_family": ex.needle_family,
            }
            for name, output in agg_outputs.items():
                row[f"{name}_score"] = float(output.scores[idx].item())
            if capture_max_index and max_indices is not None and max_probs is not None:
                row["max_token_index"] = int(max_indices[idx].item())
                row["max_token_prob"] = float(max_probs[idx].item())
            if diagnostic_threshold is not None and diagnostic_lengths is not None:
                row["seq_len"] = int(diagnostic_lengths[idx].item())
                row["max_token_prob"] = float(diagnostic_max[idx].item())
                row["second_max_token_prob"] = float(diagnostic_second[idx].item())
                row["num_tokens_above_threshold"] = int(diagnostic_counts[idx].item())
                row["diagnostic_threshold"] = float(diagnostic_threshold)
            rows.append(row)

        progress.update(len(batch_examples))
    progress.close()
    return rows


def _group_indices(
    rows: list[dict[str, object]],
) -> dict[tuple[object, object], list[int]]:
    groups: dict[tuple[object, object], list[int]] = {}
    for idx, row in enumerate(rows):
        key = (row["length_bucket"], row["distractor_level"])
        groups.setdefault(key, []).append(idx)
        groups.setdefault((row["length_bucket"], "all"), []).append(idx)
        groups.setdefault(("all", row["distractor_level"]), []).append(idx)
    groups[("all", "all")] = list(range(len(rows)))
    return groups


def _compute_metrics_for_group(
    rows: list[dict[str, object]],
    indices: list[int],
    scores: torch.Tensor,
    probs: torch.Tensor,
    labels: torch.Tensor,
    thresholds: dict[float, float],
    target_tprs: list[float],
) -> dict[str, object]:
    """Compute both intrinsic and fixed-threshold metrics for a group.

    Returns metrics dict with:
    - auroc, auprc, ece, brier: standard metrics
    - fpr_intrinsic_at_tpr_X.XX: FPR via ROC interpolation (threshold-free)
    - fpr_fixed_at_tpr_X.XX: FPR at fixed threshold from validation
    - tpr_fixed_at_tpr_X.XX: actual TPR at that fixed threshold
    - n_pos, n_neg: sample counts
    """
    if not indices:
        return {}
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    group_scores = scores[idx_tensor]
    group_probs = probs[idx_tensor]
    group_labels = labels[idx_tensor]

    # Legacy metrics (auroc, auprc, ece, brier)
    metrics = compute_metrics(group_probs, group_labels)

    # Intrinsic FPR@TPR (via ROC curve interpolation - threshold-free)
    for tpr in target_tprs:
        fpr_intrinsic = compute_intrinsic_fpr_at_tpr(group_scores, group_labels, tpr)
        metrics[f"fpr_intrinsic_at_tpr_{tpr:.2f}"] = fpr_intrinsic

    # Fixed-threshold FPR and TPR (using validation threshold)
    for tpr, threshold in thresholds.items():
        fpr_fixed, tpr_actual = compute_fpr_tpr(group_probs, group_labels, threshold)
        metrics[f"fpr_fixed_at_tpr_{tpr:.2f}"] = fpr_fixed
        metrics[f"tpr_fixed_at_tpr_{tpr:.2f}"] = tpr_actual

    metrics["n_pos"] = int(group_labels.sum().item())
    metrics["n_neg"] = int((group_labels == 0).sum().item())
    return metrics


def _write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_run_record(
    config: DatasetConfig,
    run_id: str,
    run_start: str,
    run_end: str,
    metrics_path: Path,
    runtime_info: dict[str, object],
) -> Path:
    repo_root = Path(__file__).parent.parent
    run_records_dir = repo_root / "results" / "run_records"
    run_records_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "stage": "evaluation",
        "run_id": run_id,
        "dataset_name": config.output.dataset_name,
        "dataset_version": config.output.dataset_version,
        "timestamps": {"start": run_start, "end": run_end},
        "model_id": config.tokenizer.model_id,
        "tokenizer_id": config.tokenizer.tokenizer_id,
        "eval_config": config.eval.__dict__,
        "metrics_path": str(metrics_path),
        "git_commit": _get_git_commit(repo_root),
        "runtime": runtime_info,
    }

    path = (
        run_records_dir
        / f"{config.output.dataset_name}_{config.output.dataset_version}_eval_run_record.json"
    )
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def _format_excerpt(text: str, start: int, end: int, window: int = 200) -> str:
    if start < 0 or end < 0 or start >= len(text):
        return text[: min(len(text), window)]
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right]
    marker_start = start - left
    marker_end = max(marker_start + 1, end - left)
    return (
        snippet[:marker_start]
        + "<<<"
        + snippet[marker_start:marker_end]
        + ">>>"
        + snippet[marker_end:]
    )


def _compute_top_triggering_tokens(
    examples: list[DatasetExample],
    rows: list[dict[str, object]],
    tokenizer: TokenizerWrapper,
    top_n: int = 20,
) -> list[dict[str, object]]:
    """Compute which token IDs most often attain the maximum score on negatives.

    This analysis helps identify token-identity leakage in the probe.
    """
    id_to_example = {ex.id: ex for ex in examples}
    token_id_counts: dict[int, int] = {}
    token_id_total_prob: dict[int, float] = {}

    for row in rows:
        if row["needle_present"]:
            continue  # Only analyze negatives
        ex = id_to_example.get(row["id"])
        if ex is None:
            continue

        max_idx = int(row.get("max_token_index", -1))
        max_prob = float(row.get("max_token_prob", 0.0))

        enc = tokenizer.tokenizer(
            ex.text,
            add_special_tokens=False,
            truncation=False,
        )
        input_ids = enc.get("input_ids", [])

        if 0 <= max_idx < len(input_ids):
            token_id = input_ids[max_idx]
            token_id_counts[token_id] = token_id_counts.get(token_id, 0) + 1
            token_id_total_prob[token_id] = (
                token_id_total_prob.get(token_id, 0.0) + max_prob
            )

    # Sort by count descending
    sorted_tokens = sorted(token_id_counts.items(), key=lambda x: -x[1])

    results = []
    for token_id, count in sorted_tokens[:top_n]:
        token_str = tokenizer.tokenizer.decode([token_id])
        avg_prob = token_id_total_prob[token_id] / count
        results.append(
            {
                "token_id": token_id,
                "token_str": repr(token_str),
                "count": count,
                "avg_prob": avg_prob,
            }
        )

    return results


def _write_top_triggering_analysis(
    output_path: Path,
    trigger_analysis: list[dict[str, object]],
    length_bucket: int | str,
    distractor_level: int | str,
) -> None:
    """Write top triggering token analysis to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"Top Triggering Tokens Analysis\n")
        f.write(
            f"Length bucket: {length_bucket}, Distractor level: {distractor_level}\n"
        )
        f.write("=" * 60 + "\n\n")
        f.write(
            "These are the token IDs that most frequently attain the maximum\n"
            "score on negative documents. High counts for common tokens (like\n"
            "'the', punctuation) indicate token-identity leakage.\n\n"
        )
        f.write(f"{'Rank':<6} {'Token ID':<12} {'Count':<8} {'Avg Prob':<10} Token\n")
        f.write("-" * 60 + "\n")
        for i, item in enumerate(trigger_analysis, 1):
            f.write(
                f"{i:<6} {item['token_id']:<12} {item['count']:<8} "
                f"{item['avg_prob']:<10.4f} {item['token_str']}\n"
            )


def _write_error_analysis(
    output_path: Path,
    examples: list[DatasetExample],
    rows: list[dict[str, object]],
    threshold: float,
    tokenizer: TokenizerWrapper,
    max_samples: int,
) -> None:
    id_to_example = {ex.id: ex for ex in examples}
    false_pos = []
    false_neg = []

    for row in rows:
        pred = row.get("max_token_prob", 0.0) >= threshold
        label = bool(row["needle_present"])
        if pred and not label:
            false_pos.append(row)
        elif (not pred) and label:
            false_neg.append(row)

    def _render_row(row: dict[str, object], label_name: str) -> str:
        ex = id_to_example.get(row["id"])
        if ex is None:
            return ""
        max_idx = int(row.get("max_token_index", -1))
        excerpt = ""
        if ex.needle_present and ex.needle_char_span is not None:
            start, end = ex.needle_char_span
            excerpt = _format_excerpt(ex.text, start, end)
        else:
            enc = tokenizer.tokenizer(
                ex.text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
            offsets = enc.get("offset_mapping", [])
            if 0 <= max_idx < len(offsets):
                start, end = offsets[max_idx]
                excerpt = _format_excerpt(ex.text, start, end)
            else:
                excerpt = ex.text[:200]
        return (
            f"[{label_name}] id={row['id']} length={row['length_bucket']} "
            f"distractors={row['distractor_level']} prob={row.get('max_token_prob', 0.0):.4f}\n"
            f"{excerpt}\n"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("False Positives (max aggregator, long length)\n")
        f.write("=" * 60 + "\n")
        for row in false_pos[:max_samples]:
            f.write(_render_row(row, "FP"))
            f.write("\n")

        f.write("\nFalse Negatives (max aggregator, long length)\n")
        f.write("=" * 60 + "\n")
        for row in false_neg[:max_samples]:
            f.write(_render_row(row, "FN"))
            f.write("\n")


def main() -> int:
    args = parse_args()
    disable_tokenizers_parallelism()

    logger.info("Loading configuration from %s", args.config)
    config = load_config(args.config)
    eval_cfg = config.eval

    run_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    doc_batch_size = (
        args.doc_batch_size
        if args.doc_batch_size is not None
        else eval_cfg.eval_doc_batch_size
    )
    max_eval_docs = (
        args.max_eval_docs if args.max_eval_docs is not None else eval_cfg.max_eval_docs
    )

    allowed_layers = _parse_layers(args.layers)
    probe_dir = (
        Path(args.probe_dir)
        if args.probe_dir
        else Path(config.probe.output_dir)
        / f"{config.output.dataset_name}_{config.output.dataset_version}"
    )
    probe_infos = _load_probe_infos(probe_dir, allowed_layers)

    tokenizer = TokenizerWrapper(config.tokenizer)
    examples = _load_examples(config)

    device = resolve_device(config.probe.device)
    model_dtype = resolve_dtype(config.probe.model_dtype, device)
    device_map = args.device_map

    logger.info("Loading model %s", config.tokenizer.model_id)
    model_kwargs: dict[str, object] = {
        "dtype": model_dtype,
        "trust_remote_code": True,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
        model_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(
        config.tokenizer.model_id, **model_kwargs
    )
    if not device_map:
        model.to(device)
    model.eval()

    input_device = resolve_input_device(model, fallback=device)

    aggregators = build_aggregators(
        names=eval_cfg.aggregators,
        temperature=eval_cfg.logsumexp_temperature,
        length_correction=eval_cfg.logsumexp_length_correction,
        topk_k=eval_cfg.topk_k,
        ema_alpha=eval_cfg.ema_alpha,
        window_size=eval_cfg.window_size,
    )

    metrics_rows: list[dict[str, object]] = []
    scores_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    for probe_info in probe_infos:
        layer_idx = int(probe_info["layer"])
        seed = int(probe_info["seed"])
        state = torch.load(probe_info["path"], map_location="cpu")
        linear = _build_linear_from_state(state, input_device)

        extractor = ActivationExtractor(
            model=model,
            layer_idx=layer_idx,
            activation_site=config.probe.activation_site,
            device=input_device,
            feature_dtype=torch.float32,
        )

        val_examples = _filter_examples(examples, "val", seed + 1, max_eval_docs)
        test_examples = _filter_examples(examples, "test", seed + 2, max_eval_docs)

        val_rows = _score_split(
            examples=val_examples,
            split="val",
            extractor=extractor,
            linear=linear,
            tokenizer=tokenizer,
            aggregators=aggregators,
            doc_batch_size=doc_batch_size,
            progress_desc=f"val scoring (L{layer_idx}, seed={seed})",
        )
        test_rows = _score_split(
            examples=test_examples,
            split="test",
            extractor=extractor,
            linear=linear,
            tokenizer=tokenizer,
            aggregators=aggregators,
            doc_batch_size=doc_batch_size,
            progress_desc=f"test scoring (L{layer_idx}, seed={seed})",
            capture_max_index=eval_cfg.error_analysis,
            diagnostic_threshold=eval_cfg.diagnostic_token_threshold,
        )

        if eval_cfg.save_scores:
            scores_rows.extend(val_rows + test_rows)

        for agg_name in eval_cfg.aggregators:
            val_scores = torch.tensor(
                [row[f"{agg_name}_score"] for row in val_rows],
                dtype=torch.float32,
            )
            val_labels = torch.tensor(
                [row["needle_present"] for row in val_rows],
                dtype=torch.float32,
            )
            test_scores = torch.tensor(
                [row[f"{agg_name}_score"] for row in test_rows],
                dtype=torch.float32,
            )
            test_labels = torch.tensor(
                [row["needle_present"] for row in test_rows],
                dtype=torch.float32,
            )

            if (
                not torch.isfinite(val_scores).all()
                or not torch.isfinite(test_scores).all()
            ):
                raise RuntimeError(
                    f"Non-finite scores for aggregator '{agg_name}'. "
                    "Check probe weights and feature extraction."
                )

            calibrators = {
                "uncalibrated": IdentityCalibrator(),
            }
            if eval_cfg.calibrator == "platt":
                calibrators["calibrated"] = fit_platt(val_scores, val_labels)

            for cal_name, calibrator in calibrators.items():
                val_probs = calibrator.predict_proba(val_scores)
                test_probs = calibrator.predict_proba(test_scores)
                thresholds = {
                    tpr: threshold_at_tpr(val_probs, val_labels, tpr)
                    for tpr in eval_cfg.target_tprs
                }

                groups = _group_indices(test_rows)
                for (length_bucket, distractor_level), indices in groups.items():
                    metrics = _compute_metrics_for_group(
                        rows=test_rows,
                        indices=indices,
                        scores=test_scores,
                        probs=test_probs,
                        labels=test_labels,
                        thresholds=thresholds,
                        target_tprs=eval_cfg.target_tprs,
                    )
                    if not metrics:
                        continue
                    row = {
                        "layer": layer_idx,
                        "seed": seed,
                        "aggregator": agg_name,
                        "calibrated": cal_name,
                        "split": "test",
                        "length_bucket": length_bucket,
                        "distractor_level": distractor_level,
                        "logsumexp_temperature": eval_cfg.logsumexp_temperature,
                        "logsumexp_length_correction": eval_cfg.logsumexp_length_correction,
                        "ema_alpha": eval_cfg.ema_alpha,
                        "window_size": eval_cfg.window_size,
                        "topk_k": eval_cfg.topk_k,
                        **metrics,
                    }
                    metrics_rows.append(row)

            if eval_cfg.calibrator == "platt":
                calib_dir = Path(eval_cfg.results_dir) / "calibration"
                calib_dir.mkdir(parents=True, exist_ok=True)
                calib_path = (
                    calib_dir / f"calib_layer_{layer_idx}_seed_{seed}_{agg_name}.json"
                )
                with open(calib_path, "w") as f:
                    json.dump(calibrators["calibrated"].state_dict(), f, indent=2)

        if eval_cfg.error_analysis:
            if "max" not in eval_cfg.aggregators:
                logger.warning(
                    "error_analysis requires 'max' aggregator; skipping for layer %s.",
                    layer_idx,
                )
                continue
            longest_bucket = (
                eval_cfg.error_length_bucket
                if eval_cfg.error_length_bucket is not None
                else max(config.grid.length_buckets_tokens)
            )
            val_long_rows = [
                row for row in val_rows if row["length_bucket"] == longest_bucket
            ]
            if not val_long_rows:
                val_long_rows = val_rows

            val_max_scores = torch.tensor(
                [row["max_score"] for row in val_long_rows], dtype=torch.float32
            )
            val_max_labels = torch.tensor(
                [row["needle_present"] for row in val_long_rows], dtype=torch.float32
            )
            val_max_probs = torch.sigmoid(val_max_scores)
            threshold = threshold_at_tpr(
                val_max_probs, val_max_labels, eval_cfg.target_tprs[0]
            )

            long_rows = [
                row for row in test_rows if row["length_bucket"] == longest_bucket
            ]
            if long_rows:
                error_path = (
                    Path(eval_cfg.results_dir)
                    / "analysis"
                    / f"{config.output.dataset_name}_{config.output.dataset_version}_errors_layer_{layer_idx}_seed_{seed}.txt"
                )
                long_examples = [
                    ex for ex in test_examples if ex.length_bucket == longest_bucket
                ]
                _write_error_analysis(
                    output_path=error_path,
                    examples=long_examples,
                    rows=long_rows,
                    threshold=threshold,
                    tokenizer=tokenizer,
                    max_samples=eval_cfg.error_samples,
                )

                # Top triggering token analysis (for each distractor level)
                for dist_level in [0] + config.grid.distractor_levels:
                    dist_rows = [
                        r
                        for r in long_rows
                        if r.get("distractor_level") == dist_level
                        or dist_level == "all"
                    ]
                    if len(dist_rows) < 10:
                        continue
                    dist_examples = [
                        ex
                        for ex in long_examples
                        if ex.distractor_level == dist_level or dist_level == "all"
                    ]
                    trigger_analysis = _compute_top_triggering_tokens(
                        examples=dist_examples,
                        rows=dist_rows,
                        tokenizer=tokenizer,
                        top_n=20,
                    )
                    if trigger_analysis:
                        trigger_path = (
                            Path(eval_cfg.results_dir)
                            / "analysis"
                            / f"{config.output.dataset_name}_{config.output.dataset_version}_triggers_layer_{layer_idx}_seed_{seed}_len_{longest_bucket}_dist_{dist_level}.txt"
                        )
                        _write_top_triggering_analysis(
                            output_path=trigger_path,
                            trigger_analysis=trigger_analysis,
                            length_bucket=longest_bucket,
                            distractor_level=dist_level,
                        )

        if eval_cfg.diagnostic_token_threshold is not None:
            for row in test_rows:
                if row.get("needle_present"):
                    continue
                if "second_max_token_prob" not in row:
                    continue
                diagnostic_rows.append(
                    {
                        "layer": layer_idx,
                        "seed": seed,
                        "id": row.get("id"),
                        "length_bucket": row.get("length_bucket"),
                        "distractor_level": row.get("distractor_level"),
                        "seq_len": row.get("seq_len"),
                        "max_token_prob": row.get("max_token_prob"),
                        "second_max_token_prob": row.get("second_max_token_prob"),
                        "num_tokens_above_threshold": row.get(
                            "num_tokens_above_threshold"
                        ),
                        "diagnostic_threshold": row.get("diagnostic_threshold"),
                    }
                )

    metrics_path = (
        Path(eval_cfg.results_dir)
        / "metrics"
        / f"{config.output.dataset_name}_{config.output.dataset_version}_eval_metrics.csv"
    )
    _write_metrics_csv(metrics_path, metrics_rows)

    if eval_cfg.save_scores:
        scores_path = (
            Path(eval_cfg.results_dir)
            / "metrics"
            / f"{config.output.dataset_name}_{config.output.dataset_version}_eval_scores.jsonl"
        )
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_path, "w") as f:
            for row in scores_rows:
                f.write(json.dumps(row) + "\n")

    if diagnostic_rows:
        diagnostic_path = (
            Path(eval_cfg.results_dir)
            / "analysis"
            / f"{config.output.dataset_name}_{config.output.dataset_version}_negative_token_stats.csv"
        )
        _write_metrics_csv(diagnostic_path, diagnostic_rows)

    run_end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_run_record(
        config=config,
        run_id=f"{config.output.dataset_name}_{config.output.dataset_version}",
        run_start=run_start,
        run_end=run_end,
        metrics_path=metrics_path,
        runtime_info={
            "device": str(device),
            "device_map": device_map,
            "doc_batch_size": doc_batch_size,
            "max_eval_docs": max_eval_docs,
        },
    )

    logger.info("Evaluation complete. Metrics written to %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
