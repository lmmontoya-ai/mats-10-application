#!/usr/bin/env python3
"""
Train token-level linear probes for prompt injection monitoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import random
import queue as queue_module
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.config import load_config, DatasetConfig
from src.dataset.io import DatasetReader
from src.dataset.schema import DatasetExample
from src.dataset.tokenizer_utils import TokenizerWrapper
from src.probes.activation import ActivationExtractor, resolve_transformer_layers
from src.probes.training import TrainConfig, train_linear_probe, evaluate_linear_probe
from src.probes.utils import (
    disable_tokenizers_parallelism,
    resolve_device,
    resolve_dtype,
    resolve_feature_dtype,
    resolve_input_device,
    set_seed,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train token-level probes from generated dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_main.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--doc-batch-size",
        type=int,
        default=1,
        help="Number of documents per forward pass.",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=1,
        help="Number of parallel extraction workers (CUDA only).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help=(
            "Hugging Face device_map setting for model sharding "
            "(e.g. 'auto', 'balanced', 'sequential')."
        ),
    )
    return parser.parse_args()


def _parse_device_map(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in {"none", "single", "disabled"}:
        return None
    return cleaned


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


def _load_examples_by_ids(dataset_path: Path, ids: list[str]) -> list[DatasetExample]:
    if not ids:
        return []
    id_set = set(ids)
    selected = [ex for ex in DatasetReader.read_jsonl(dataset_path) if ex.id in id_set]
    id_to_example = {ex.id: ex for ex in selected}
    return [id_to_example[ex_id] for ex_id in ids if ex_id in id_to_example]


def _filter_examples(
    examples: list[DatasetExample],
    split: str,
    seed: int,
    max_docs: int | None,
    length_buckets: list[int] | None,
) -> list[DatasetExample]:
    filtered = [ex for ex in examples if ex.split == split]
    if length_buckets:
        allowed = set(length_buckets)
        filtered = [ex for ex in filtered if ex.length_bucket in allowed]
    if max_docs is not None and len(filtered) > max_docs:
        rng = random.Random(seed)
        rng.shuffle(filtered)
        filtered = filtered[:max_docs]
    return filtered


def _sample_negatives(
    seq_len: int,
    positive_indices: set[int],
    n_negatives: int,
    rng: random.Random,
) -> list[int]:
    candidates = [i for i in range(seq_len) if i not in positive_indices]
    if not candidates:
        return []
    if n_negatives >= len(candidates):
        return [rng.choice(candidates) for _ in range(n_negatives)]
    return rng.sample(candidates, n_negatives)


def _build_feature_tensor_from_examples(
    split_examples: list[DatasetExample],
    extractor: ActivationExtractor,
    tokenizer: TokenizerWrapper,
    seed: int,
    negatives_per_doc: int,
    max_tokens: int | None,
    feature_dtype: torch.dtype,
    doc_batch_size: int,
    progress_desc: str | None = None,
    show_progress: bool = True,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    total_tokens = 0

    progress = None
    if show_progress:
        progress = tqdm(
            total=len(split_examples),
            desc=progress_desc or "extraction",
            unit="docs",
            dynamic_ncols=True,
        )

    for start in range(0, len(split_examples), doc_batch_size):
        batch_examples = split_examples[start : start + doc_batch_size]
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

        selected_indices_per_example: list[list[int]] = []
        labels_per_example: list[torch.Tensor] = []

        for batch_idx, ex in enumerate(batch_examples):
            seq_len = int(attention_mask[batch_idx].sum().item())
            if ex.length_tokens and abs(seq_len - ex.length_tokens) > 0:
                logger.debug(
                    "Token length mismatch for %s: dataset=%s, actual=%s",
                    ex.id,
                    ex.length_tokens,
                    seq_len,
                )

            positives = set(ex.needle_token_indices) if ex.needle_present else set()
            if positives and max(positives) >= seq_len:
                logger.warning(
                    "Needle indices out of bounds for %s (seq_len=%s); filtering",
                    ex.id,
                    seq_len,
                )
                positives = {i for i in positives if i < seq_len}

            neg_rng = random.Random(seed + ex.seed)
            negatives = _sample_negatives(
                seq_len, positives, negatives_per_doc, neg_rng
            )

            selected_indices = list(sorted(positives)) + negatives
            if (
                max_tokens is not None
                and total_tokens + len(selected_indices) > max_tokens
            ):
                remaining = max_tokens - total_tokens
                if remaining <= 0:
                    selected_indices_per_example.append([])
                    labels_per_example.append(torch.tensor([], dtype=torch.float32))
                    continue
                if len(positives) > remaining:
                    selected_indices = list(sorted(positives))[:remaining]
                else:
                    neg_keep = max(0, remaining - len(positives))
                    selected_indices = list(sorted(positives)) + negatives[:neg_keep]

            selected_pos = [idx for idx in selected_indices if idx in positives]
            selected_neg = [idx for idx in selected_indices if idx not in positives]
            labels = torch.tensor(
                [1] * len(selected_pos) + [0] * len(selected_neg),
                dtype=torch.float32,
            )
            selected_indices_per_example.append(selected_indices)
            labels_per_example.append(labels)
            total_tokens += labels.numel()

        if not any(indices for indices in selected_indices_per_example):
            if max_tokens is not None and total_tokens >= max_tokens:
                break
            continue

        selected_features_batch = extractor.extract_selected_tokens_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_indices_per_example=selected_indices_per_example,
        )

        for features, labels in zip(selected_features_batch, labels_per_example):
            if features.numel() == 0 or labels.numel() == 0:
                continue
            features_list.append(features.to(dtype=feature_dtype))
            labels_list.append(labels)

        if progress is not None:
            progress.update(len(batch_examples))
            progress.set_postfix(tokens=total_tokens)
        if progress_callback is not None:
            progress_callback(len(batch_examples))

        if max_tokens is not None and total_tokens >= max_tokens:
            break

    if progress is not None:
        progress.close()

    if not features_list:
        raise RuntimeError("No features extracted")

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    stats = {
        "docs": len(split_examples),
        "tokens": int(labels.numel()),
        "positives": int(labels.sum().item()),
        "negatives": int((labels == 0).sum().item()),
    }
    return features, labels, stats


def _build_feature_tensor(
    examples: list[DatasetExample],
    extractor: ActivationExtractor,
    tokenizer: TokenizerWrapper,
    split: str,
    seed: int,
    negatives_per_doc: int,
    max_docs: int | None,
    max_tokens: int | None,
    length_buckets: list[int] | None,
    feature_dtype: torch.dtype,
    doc_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    split_examples = _filter_examples(examples, split, seed, max_docs, length_buckets)
    return _build_feature_tensor_from_examples(
        split_examples=split_examples,
        extractor=extractor,
        tokenizer=tokenizer,
        seed=seed,
        negatives_per_doc=negatives_per_doc,
        max_tokens=max_tokens,
        feature_dtype=feature_dtype,
        doc_batch_size=doc_batch_size,
        progress_desc=f"{split} extraction",
        show_progress=True,
    )


def _partition_example_ids(
    examples: list[DatasetExample], num_workers: int
) -> list[list[str]]:
    shards: list[list[str]] = [[] for _ in range(num_workers)]
    for idx, ex in enumerate(examples):
        shards[idx % num_workers].append(ex.id)
    return shards


def _extract_worker(
    rank: int,
    config_path: str,
    dataset_path: str,
    example_ids: list[str],
    split: str,
    seed: int,
    layer_idx: int,
    doc_batch_size: int,
    negatives_per_doc: int,
    max_tokens: int | None,
    output_path: str,
    progress_queue: mp.Queue | None,
) -> None:
    disable_tokenizers_parallelism()
    torch.set_num_threads(1)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for parallel extraction")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    config = load_config(config_path)
    tokenizer = TokenizerWrapper(config.tokenizer)
    probe_config = config.probe

    model_dtype = resolve_dtype(probe_config.model_dtype, device)
    feature_dtype = resolve_feature_dtype(probe_config.feature_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        config.tokenizer.model_id,
        dtype=model_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    extractor = ActivationExtractor(
        model=model,
        layer_idx=layer_idx,
        activation_site=probe_config.activation_site,
        device=device,
        feature_dtype=feature_dtype,
    )

    if not example_ids:
        empty_stats = {"docs": 0, "tokens": 0, "positives": 0, "negatives": 0}
        torch.save(
            {
                "features": torch.empty((0, 0), dtype=feature_dtype),
                "labels": torch.empty((0,), dtype=torch.float32),
                "stats": empty_stats,
            },
            output_path,
        )
        return

    examples = _load_examples_by_ids(Path(dataset_path), example_ids)

    def _progress(count: int) -> None:
        if progress_queue is not None:
            progress_queue.put(count)

    features, labels, stats = _build_feature_tensor_from_examples(
        split_examples=examples,
        extractor=extractor,
        tokenizer=tokenizer,
        seed=seed,
        negatives_per_doc=negatives_per_doc,
        max_tokens=max_tokens,
        feature_dtype=feature_dtype,
        doc_batch_size=doc_batch_size,
        progress_desc=None,
        show_progress=False,
        progress_callback=_progress,
    )

    torch.save(
        {"features": features, "labels": labels, "stats": stats},
        output_path,
    )


def _build_feature_tensor_parallel(
    split_examples: list[DatasetExample],
    config_path: str,
    dataset_path: Path,
    split: str,
    seed: int,
    layer_idx: int,
    doc_batch_size: int,
    negatives_per_doc: int,
    max_tokens: int | None,
    cache_dir: Path,
    num_workers: int,
    progress_desc: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if not split_examples:
        raise RuntimeError(f"No examples found for split {split}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_ids = _partition_example_ids(split_examples, num_workers)

    per_worker_max = max_tokens
    if max_tokens is not None and num_workers > 1:
        per_worker_max = math.ceil(max_tokens / num_workers)
        logger.warning(
            "Parallel extraction splits max_tokens=%s across %s workers "
            "(per-worker cap=%s).",
            max_tokens,
            num_workers,
            per_worker_max,
        )

    ctx = mp.get_context("spawn")
    progress_queue: mp.Queue | None = ctx.Queue()
    processes: list[mp.Process] = []
    shard_paths: list[Path] = []

    for rank in range(num_workers):
        shard_path = cache_dir / f"{split}_layer_{layer_idx}_seed_{seed}_rank_{rank}.pt"
        shard_paths.append(shard_path)
        proc = ctx.Process(
            target=_extract_worker,
            args=(
                rank,
                config_path,
                str(dataset_path),
                shard_ids[rank],
                split,
                seed,
                layer_idx,
                doc_batch_size,
                negatives_per_doc,
                per_worker_max,
                str(shard_path),
                progress_queue,
            ),
        )
        proc.start()
        processes.append(proc)

    total_docs = len(split_examples)
    progress = tqdm(
        total=total_docs,
        desc=progress_desc,
        unit="docs",
        dynamic_ncols=True,
    )
    processed = 0

    while processed < total_docs:
        try:
            delta = progress_queue.get(timeout=0.5) if progress_queue else 0
            processed += delta
            progress.update(delta)
        except queue_module.Empty:
            if not any(proc.is_alive() for proc in processes):
                break

    progress.close()

    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(
                f"Extraction worker failed for split {split} (exitcode={proc.exitcode})"
            )

    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    stats = {"docs": 0, "tokens": 0, "positives": 0, "negatives": 0}

    for shard_path in shard_paths:
        data = torch.load(shard_path, map_location="cpu")
        shard_features = data.get("features")
        shard_labels = data.get("labels")
        shard_stats = data.get("stats", {})

        if isinstance(shard_stats, dict):
            stats["docs"] += int(shard_stats.get("docs", 0))
            stats["tokens"] += int(shard_stats.get("tokens", 0))
            stats["positives"] += int(shard_stats.get("positives", 0))
            stats["negatives"] += int(shard_stats.get("negatives", 0))

        if isinstance(shard_features, torch.Tensor) and shard_features.numel() > 0:
            features_list.append(shard_features)
        if isinstance(shard_labels, torch.Tensor) and shard_labels.numel() > 0:
            labels_list.append(shard_labels)

    if not features_list or not labels_list:
        raise RuntimeError(f"No features extracted for split {split}")

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels, stats


def _downsample_negatives(
    features: torch.Tensor, labels: torch.Tensor, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    positives = torch.where(labels == 1)[0]
    negatives = torch.where(labels == 0)[0]
    if positives.numel() == 0 or negatives.numel() == 0:
        return features, labels

    n_keep = min(positives.numel(), negatives.numel())
    rng = torch.Generator().manual_seed(seed)
    perm = negatives[torch.randperm(negatives.numel(), generator=rng)[:n_keep]]
    keep = torch.cat([positives, perm], dim=0)
    keep = keep[torch.randperm(keep.numel(), generator=rng)]
    return features[keep], labels[keep]


def _write_metrics_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _save_probe(
    output_dir: Path,
    layer_idx: int,
    seed: int,
    model: torch.nn.Module,
    metadata: dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_path = output_dir / f"probe_layer_{layer_idx}_seed_{seed}.pt"
    torch.save(model.state_dict(), probe_path)

    meta_path = output_dir / f"probe_layer_{layer_idx}_seed_{seed}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return probe_path


def _write_run_record(
    output_dir: Path,
    config: DatasetConfig,
    tokenizer: TokenizerWrapper,
    run_id: str,
    run_start: str,
    run_end: str,
    metrics_path: Path,
    selected_by_layer: dict[int, dict[str, float | int | str]],
    runtime_info: dict[str, object] | None = None,
) -> Path:
    repo_root = Path(__file__).parent.parent
    run_records_dir = repo_root / "results" / "run_records"
    run_records_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "stage": "probe_training",
        "run_id": run_id,
        "dataset_name": config.output.dataset_name,
        "dataset_version": config.output.dataset_version,
        "timestamps": {"start": run_start, "end": run_end},
        "model_id": config.tokenizer.model_id,
        "tokenizer": tokenizer.get_tokenizer_info(),
        "probe_config": config.probe.__dict__,
        "hyperparameters_searched": {"l2_grid": config.probe.l2_grid},
        "selected_hyperparameters": selected_by_layer,
        "metrics_path": str(metrics_path),
        "outputs_dir": str(output_dir),
        "git_commit": _get_git_commit(repo_root),
        "runtime": runtime_info or {},
    }

    path = (
        run_records_dir
        / f"{config.output.dataset_name}_{config.output.dataset_version}_probe_run_record.json"
    )
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    logger.info("Wrote run record to %s", path)
    return path


def main() -> int:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    disable_tokenizers_parallelism()

    if args.doc_batch_size < 1:
        raise ValueError("--doc-batch-size must be >= 1")
    if args.extract_workers < 1:
        raise ValueError("--extract-workers must be >= 1")

    logger.info("Loading configuration from %s", args.config)
    config = load_config(args.config)

    probe_config = config.probe
    if not probe_config.layers_to_probe:
        raise ValueError("probe.layers_to_probe must not be empty")
    if probe_config.class_balance not in {"weighted", "downsample"}:
        raise ValueError("probe.class_balance must be 'weighted' or 'downsample'")

    run_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    device_map = _parse_device_map(args.device_map)
    extract_workers = args.extract_workers
    if device_map is not None and extract_workers > 1:
        raise ValueError("Cannot combine --device-map with --extract-workers > 1")

    tokenizer = TokenizerWrapper(config.tokenizer)
    dataset_path = _get_dataset_path(config)
    examples = _load_examples(config)

    device = resolve_device(probe_config.device)
    model_dtype = resolve_dtype(probe_config.model_dtype, device)
    feature_dtype = resolve_feature_dtype(probe_config.feature_dtype)

    if device_map is not None:
        if device.type != "cuda":
            logger.warning(
                "device_map is set but resolved device is %s; multi-GPU sharding "
                "requires CUDA.",
                device.type,
            )
        logger.info(
            "Loading model %s with device_map=%s", config.tokenizer.model_id, device_map
        )
        if torch.cuda.device_count() < 2:
            logger.warning(
                "device_map is set but fewer than 2 CUDA devices are available."
            )
    else:
        logger.info("Loading model %s on %s", config.tokenizer.model_id, device)

    if extract_workers > 1:
        if device.type != "cuda":
            logger.warning(
                "Parallel extraction requires CUDA; falling back to 1 worker."
            )
            extract_workers = 1
        else:
            available = torch.cuda.device_count()
            if available < 2:
                logger.warning(
                    "Only one CUDA device available; falling back to 1 worker."
                )
                extract_workers = 1
            elif extract_workers > available:
                logger.warning(
                    "Requested %s extraction workers but only %s CUDA devices "
                    "available; using %s.",
                    extract_workers,
                    available,
                    available,
                )
                extract_workers = available

    model_kwargs: dict[str, object] = {
        "dtype": model_dtype,
        "trust_remote_code": True,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
        model_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(
        config.tokenizer.model_id, **model_kwargs
    )
    if device_map is None:
        model.to(device)
    model.eval()

    input_device = resolve_input_device(model, fallback=device)
    layers, layer_path = resolve_transformer_layers(model)
    logger.info("Resolved %d layers at %s", len(layers), layer_path)

    output_root = Path(probe_config.output_dir)
    run_id = f"{config.output.dataset_name}_{config.output.dataset_version}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "feature_cache"

    metrics_rows: list[dict[str, float | int | str]] = []
    selected_by_layer: dict[int, dict[int, dict[str, float | int | str]]] = {}

    for seed in probe_config.train_seeds:
        set_seed(seed)

        for layer_idx in probe_config.layers_to_probe:
            logger.info("Training probe for layer %s (seed=%s)", layer_idx, seed)

            if extract_workers > 1:
                train_examples = _filter_examples(
                    examples=examples,
                    split="train",
                    seed=seed,
                    max_docs=probe_config.max_train_docs,
                    length_buckets=probe_config.train_length_buckets,
                )
                val_examples = _filter_examples(
                    examples=examples,
                    split="val",
                    seed=seed + 1,
                    max_docs=probe_config.max_val_docs,
                    length_buckets=probe_config.val_length_buckets,
                )

                train_features, train_labels, train_stats = (
                    _build_feature_tensor_parallel(
                        split_examples=train_examples,
                        config_path=args.config,
                        dataset_path=dataset_path,
                        split="train",
                        seed=seed,
                        layer_idx=layer_idx,
                        doc_batch_size=args.doc_batch_size,
                        negatives_per_doc=probe_config.negatives_per_doc,
                        max_tokens=probe_config.max_train_tokens,
                        cache_dir=cache_dir,
                        num_workers=extract_workers,
                        progress_desc=f"train extraction (L{layer_idx})",
                    )
                )

                val_features, val_labels, val_stats = _build_feature_tensor_parallel(
                    split_examples=val_examples,
                    config_path=args.config,
                    dataset_path=dataset_path,
                    split="val",
                    seed=seed + 1,
                    layer_idx=layer_idx,
                    doc_batch_size=args.doc_batch_size,
                    negatives_per_doc=probe_config.negatives_per_doc,
                    max_tokens=probe_config.max_val_tokens,
                    cache_dir=cache_dir,
                    num_workers=extract_workers,
                    progress_desc=f"val extraction (L{layer_idx})",
                )
            else:
                extractor = ActivationExtractor(
                    model=model,
                    layer_idx=layer_idx,
                    activation_site=probe_config.activation_site,
                    device=input_device,
                    feature_dtype=feature_dtype,
                )

                train_features, train_labels, train_stats = _build_feature_tensor(
                    examples=examples,
                    extractor=extractor,
                    tokenizer=tokenizer,
                    split="train",
                    seed=seed,
                    negatives_per_doc=probe_config.negatives_per_doc,
                    max_docs=probe_config.max_train_docs,
                    max_tokens=probe_config.max_train_tokens,
                    length_buckets=probe_config.train_length_buckets,
                    feature_dtype=feature_dtype,
                    doc_batch_size=args.doc_batch_size,
                )

                val_features, val_labels, val_stats = _build_feature_tensor(
                    examples=examples,
                    extractor=extractor,
                    tokenizer=tokenizer,
                    split="val",
                    seed=seed + 1,
                    negatives_per_doc=probe_config.negatives_per_doc,
                    max_docs=probe_config.max_val_docs,
                    max_tokens=probe_config.max_val_tokens,
                    length_buckets=probe_config.val_length_buckets,
                    feature_dtype=feature_dtype,
                    doc_batch_size=args.doc_batch_size,
                )

            if probe_config.class_balance == "downsample":
                train_features, train_labels = _downsample_negatives(
                    train_features, train_labels, seed
                )

            train_features = train_features.to(dtype=torch.float32)
            train_labels = train_labels.to(dtype=torch.float32)
            val_features = val_features.to(dtype=torch.float32)
            val_labels = val_labels.to(dtype=torch.float32)

            train_cfg = TrainConfig(
                batch_size=probe_config.batch_size,
                num_epochs=probe_config.num_epochs,
                learning_rate=probe_config.learning_rate,
                class_balance=probe_config.class_balance,
            )

            best_l2 = None
            best_metrics = None
            best_model = None

            for l2_weight in probe_config.l2_grid:
                model_probe = train_linear_probe(
                    train_features=train_features,
                    train_labels=train_labels,
                    config=train_cfg,
                    l2_weight=l2_weight,
                    device=device,
                )

                metrics_train = evaluate_linear_probe(
                    model_probe,
                    train_features,
                    train_labels,
                    batch_size=probe_config.batch_size,
                    device=device,
                )
                metrics_val = evaluate_linear_probe(
                    model_probe,
                    val_features,
                    val_labels,
                    batch_size=probe_config.batch_size,
                    device=device,
                )

                row = {
                    "seed": seed,
                    "layer_idx": layer_idx,
                    "l2": l2_weight,
                    "split": "train",
                    **metrics_train,
                }
                metrics_rows.append(row)
                row = {
                    "seed": seed,
                    "layer_idx": layer_idx,
                    "l2": l2_weight,
                    "split": "val",
                    **metrics_val,
                }
                metrics_rows.append(row)

                if best_metrics is None or metrics_val["auroc"] > best_metrics["auroc"]:
                    best_metrics = metrics_val
                    best_l2 = l2_weight
                    best_model = model_probe

            if best_model is None or best_l2 is None or best_metrics is None:
                raise RuntimeError("Failed to select best probe")

            metadata = {
                "model_id": config.tokenizer.model_id,
                "tokenizer": tokenizer.get_tokenizer_info(),
                "layer_idx": layer_idx,
                "activation_site": probe_config.activation_site,
                "train_seed": seed,
                "selected_l2": best_l2,
                "train_stats": train_stats,
                "val_stats": val_stats,
                "val_metrics": best_metrics,
            }

            _save_probe(
                output_dir=run_dir,
                layer_idx=layer_idx,
                seed=seed,
                model=best_model,
                metadata=metadata,
            )

            selected_by_layer.setdefault(layer_idx, {})[seed] = {
                "seed": seed,
                "selected_l2": best_l2,
                **best_metrics,
            }

    metrics_path = run_dir / "probe_metrics.csv"
    _write_metrics_csv(metrics_path, metrics_rows)

    run_end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_run_record(
        output_dir=run_dir,
        config=config,
        tokenizer=tokenizer,
        run_id=run_id,
        run_start=run_start,
        run_end=run_end,
        metrics_path=metrics_path,
        selected_by_layer=selected_by_layer,
        runtime_info={
            "device": str(device),
            "device_map": device_map,
            "input_device": str(input_device),
            "doc_batch_size": args.doc_batch_size,
            "extract_workers": extract_workers,
        },
    )

    logger.info("Probe training complete. Metrics written to %s", metrics_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
