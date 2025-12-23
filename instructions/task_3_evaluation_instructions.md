# Task 3 Evaluation and Plotting Instructions

This document explains how to run Task 3 (document-level evaluation and plots),
what it does, and how to check outputs.

## Overview
- Goal: evaluate document-level monitoring aggregators and produce plots.
- Inputs: Task 1 dataset + Task 2 probes.
- Outputs: metrics CSV, calibration params, error analysis samples, and plots.
- Deterministic: results are reproducible given the same config and seeds.

## Prerequisites
- Task 1 dataset exists at:
  - `data/datasets/{dataset_name}_{dataset_version}.jsonl`
- Task 2 probes exist at:
  - `results/probes/{dataset_name}_{dataset_version}/`
- Python environment ready:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Optional (silence tokenizer fork warning):
```bash
export TOKENIZERS_PARALLELISM=false
```

## Configuration
Task 3 reads the `eval:` section in `configs/dataset_main.yaml`. Key fields:
- `aggregators`: list of aggregators to compare.
  - `max`, `noisy_or`, `logsumexp`, `ema_max`, `windowed_max_mean`, `topk_mean`
- `logsumexp_temperature`, `logsumexp_length_correction`
- `ema_alpha`, `window_size`, `topk_k`
- `target_tprs`: e.g. `[0.95]` (thresholds selected on validation)
- `eval_doc_batch_size`, `max_eval_docs`
- `calibrator`: `platt` or `none`
- `error_analysis`, `error_samples`, `error_length_bucket`
- Plot controls: `plot_length_bucket`, `plot_distractor_level`,
  `plot_layer_aggregator`, `plot_layer_metric`

## Run Task 3
Evaluate aggregators and write metrics:
```bash
python scripts/run_eval.py --config configs/dataset_main.yaml
```

Optional overrides:
```bash
python scripts/run_eval.py \
  --config configs/dataset_main.yaml \
  --doc-batch-size 1 \
  --max-eval-docs 200
```

Generate plots from the metrics:
```bash
python scripts/make_plots.py --config configs/dataset_main.yaml
```

## Outputs
- Metrics table:
  - `results/metrics/{dataset_name}_{dataset_version}_eval_metrics.csv`
- Calibration params:
  - `results/calibration/calib_layer_{L}_seed_{S}_{aggregator}.json`
- Error analysis samples:
  - `results/analysis/{dataset_name}_{dataset_version}_errors_layer_{L}_seed_{S}.txt`
- Plots:
  - `results/plots/{dataset_name}_{dataset_version}_plot1_fpr_vs_length.png`
  - `results/plots/{dataset_name}_{dataset_version}_plot2_fpr_vs_distractor.png`
  - `results/plots/{dataset_name}_{dataset_version}_plot3_layer_sweep.png`
- Run record:
  - `results/run_records/{dataset_name}_{dataset_version}_eval_run_record.json`

## Aggregators implemented
- `max`: max of per-token probabilities.
- `noisy_or`: 1 − Π(1 − p_t).
- `logsumexp`: log-mean-exp of logits (length-corrected if enabled).
- `ema_max`: EMA over token probs, then max.
- `windowed_max_mean`: max mean over a sliding window.
- `topk_mean`: mean of top-k token probabilities (requires `topk_k`).

## Quality checks
- Plot 1: FPR@TPR vs length should show max pooling inflates with length.
- Plot 2: FPR@TPR vs distractor density should increase for max pooling.
- Plot 3: performance vs layer should show a clear band of stronger layers.
- Error analysis file should contain 20 FPs and 20 FNs for max pooling at long length.

## Troubleshooting
- Missing probes:
  - Ensure Task 2 finished and probe files exist.
- OOM / slow evaluation:
  - Set `eval_doc_batch_size: 1` and reduce `max_eval_docs`.
- Empty plots:
  - Ensure metrics file exists and `calibrator` matches plot filter.
  - If `calibrator: none`, plots will use uncalibrated rows.
