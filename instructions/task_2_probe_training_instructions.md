# Task 2 Probe Training Instructions

This document explains how to run Task 2 (token-level probe training), what it
does, and how to check the quality of the probe artifacts.

## Overview
- Goal: train a linear probe that predicts needle tokens from activations.
- Inputs: dataset JSONL from Task 1 + model/tokenizer defined in config.
- Outputs: probe weights, metadata, metrics CSV, and a run record.
- Deterministic: fixed seeds and sampling produce repeatable results.

## Prerequisites
- Task 1 dataset exists at:
  - `data/datasets/{dataset_name}_{dataset_version}.jsonl`
- Python environment with dependencies installed:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```
- Model access:
  - Default is `google/gemma-3-4b-it`.
  - Ensure HF auth and local cache if required.

Optional (silence tokenizer fork warning):
```bash
export TOKENIZERS_PARALLELISM=false
```

## Configuration
Task 2 reads the `probe:` section in `configs/dataset_main.yaml`. Key fields:
- `layers_to_probe`: list of layer indices to train.
- `activation_site`: currently `resid_post`.
- `negatives_per_doc`: number of negative tokens sampled per document.
- `max_train_docs` / `max_val_docs`: cap docs for quick tests.
- `max_train_tokens` / `max_val_tokens`: cap tokens for quick tests.
- `batch_size`, `num_epochs`, `learning_rate`
- `l2_grid`: L2 weights to search.
- `class_balance`: `weighted` or `downsample`
- `train_seeds`: list of seeds for training runs.
- `device`: `auto`, `cpu`, `cuda`, or `mps`
- `model_dtype`, `feature_dtype`
- `output_dir`: where probes and metrics are written.

## Run Task 2
```bash
python scripts/train_probe.py --config configs/dataset_main.yaml
```

For faster extraction, batch multiple documents per forward pass:
```bash
python scripts/train_probe.py --config configs/dataset_main.yaml --doc-batch-size 2
```

The script will:
- Load config and dataset.
- Load the model and hook the specified layer(s).
- Extract activations for all needle tokens + sampled negatives.
- Train a linear probe per layer and select L2 by validation AUROC.
- Save probe weights and metadata.
- Write metrics and a run record.
- Show a progress bar for document extraction (train/val).

## Outputs
- Probes:
  - `results/probes/{dataset_name}_{dataset_version}/probe_layer_{L}_seed_{S}.pt`
  - `results/probes/{dataset_name}_{dataset_version}/probe_layer_{L}_seed_{S}.json`
- Metrics:
  - `results/probes/{dataset_name}_{dataset_version}/probe_metrics.csv`
- Run record:
  - `results/run_records/{dataset_name}_{dataset_version}_probe_run_record.json`

## What the probe does
- The probe is a linear classifier over token activations.
- Training labels are token-level:
  - Positive: token indices inside the needle span.
  - Negative: randomly sampled non-needle tokens.
- Output is a logit per token; Task 3 will aggregate token logits into a
  document-level monitor.

## Quality checks
After training:
- Inspect `probe_metrics.csv`:
  - Validation AUROC should be above chance (0.5).
  - If all AUROC values are ~0.5, the probe is not learning.
- Check metadata JSON:
  - `selected_l2`, `train_stats`, `val_stats`, and `val_metrics` are present.
- Confirm run record exists and matches config.

## Troubleshooting
- Dataset not found:
  - Ensure Task 1 ran and wrote `data/datasets/...jsonl`.
- Model load failures:
  - Check HF auth, local cache, and network access.
- OOM / slow runs:
  - Reduce `max_train_docs`, `max_val_docs`, or `negatives_per_doc`.
  - Set `model_dtype: float16` or `bfloat16`.
- Poor metrics:
  - Try more negatives per doc.
  - Try different layers via `layers_to_probe`.
  - Verify data QA reports from Task 1.

## Quick smoke test
To validate the pipeline quickly:
- Set `max_train_docs: 50`, `max_val_docs: 20`.
- Set `layers_to_probe: [8]`.
- Run Task 2 and confirm outputs are created.
