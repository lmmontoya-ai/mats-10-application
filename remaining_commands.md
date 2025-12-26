# Remaining Commands to Complete Research Project

This document contains the commands needed to run the improved research pipeline for the long-context probe monitoring study.

## Prerequisites

Make sure you have the environment set up:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Ensure dependencies are installed
uv pip install -e .
```

## Step 1: Data Generation (if not already done)

If you haven't generated the dataset yet:

```bash
python scripts/generate_data.py --config configs/dataset_main.yaml
```

This creates:
- `data/datasets/needle_haystack_v1.0.jsonl` - Full dataset
- `data/datasets/needle_haystack_v1.0_{train,val,test}.jsonl` - Split files
- `data/qa/` - Quality assurance reports

## Step 2: Train Probes with Improved Settings

The configuration has been updated with:
- **Token-matched negative sampling** (fixes token-identity leakage)
- **Layer sweep** (layers 4, 8, 12, 16, 20, 24)
- **Multiple seeds** (0, 1, 2 for confidence intervals)

Run probe training:

```bash
# Standard training (single GPU or CPU)
python scripts/train_probe.py --config configs/dataset_main.yaml --doc-batch-size 1

# For multi-GPU training (if available)
python scripts/train_probe.py --config configs/dataset_main.yaml --doc-batch-size 1 --device-map auto
```

**Note:** This will take longer than before due to:
- Training probes at 6 layers instead of 1
- Running with 3 seeds instead of 1
- Total: 18 probe configurations (6 layers × 3 seeds)

Outputs:
- `results/probes/needle_haystack_v1.0/probe_layer_{L}_seed_{S}.pt`
- `results/probes/needle_haystack_v1.0/probe_metrics.csv`
- `results/run_records/needle_haystack_v1.0_probe_run_record.json`

## Step 3: Run Evaluation

The evaluation now computes:
- **Intrinsic FPR@TPR** (via ROC interpolation, threshold-free)
- **Fixed-threshold FPR and TPR** (actual TPR at validation threshold)
- **Top triggering token analysis** (identifies token-identity leakage)

```bash
python scripts/run_eval.py --config configs/dataset_main.yaml --doc-batch-size 1
```

For faster evaluation on specific layers:

```bash
# Evaluate only the best layer from initial sweep
python scripts/run_eval.py --config configs/dataset_main.yaml --layers 16 --doc-batch-size 1
```

Outputs:
- `results/metrics/needle_haystack_v1.0_eval_metrics.csv` - Full metrics table
- `results/calibration/calib_layer_{L}_seed_{S}_{agg}.json` - Calibration params
- `results/analysis/` - Error analysis and triggering token reports
- `results/analysis/needle_haystack_v1.0_negative_token_stats.csv` - Token-score diagnostics (negatives only)

## Step 4: Generate Plots

The plotting script now generates:
- **Core plots** (3): FPR vs length, FPR vs distractor, layer sweep
- **Diagnostic plots** (2): Layer heatmap, calibration comparison
- **Per-distractor plots**: FPR vs length for each distractor level

```bash
python scripts/make_plots.py --config configs/dataset_main.yaml
```

Outputs in `results/plots/`:
- `needle_haystack_v1.0_plot1_fpr_vs_length.png`
- `needle_haystack_v1.0_plot2_fpr_vs_distractor.png`
- `needle_haystack_v1.0_plot3_layer_sweep.png` (now with error bars)
- `needle_haystack_v1.0_plot4_layer_heatmap.png`
- `needle_haystack_v1.0_plot5_calibration_comparison.png`
- `needle_haystack_v1.0_plot6_token_diagnostics.png` (max/second-max/count vs length)
- `needle_haystack_v1.0_fpr_vs_length_dist_{N}.png` (per distractor level)

## Step 5: Parameter Sensitivity Sweep (Optional but Recommended)

To demonstrate robustness of EMA and windowed aggregators:

```bash
# Find the best probe first (e.g., layer 16, seed 0)
PROBE_PATH="results/probes/needle_haystack_v1.0/probe_layer_16_seed_0.pt"

# Run parameter sweep
python scripts/sweep_params.py \
    --config configs/dataset_main.yaml \
    --probe-path $PROBE_PATH \
    --layer 16 \
    --ema-alphas "0.05,0.1,0.15,0.2,0.25,0.3" \
    --window-sizes "32,64,128,256,512" \
    --output results/analysis/param_sweep.csv
```

This shows sensitivity of best aggregators to hyperparameter choices.

## Quick Validation Run

If you want to verify the pipeline works before a full run:

1. Create a quick test config with smaller settings:

```bash
# Modify config temporarily for quick validation
# - Set layers_to_probe: [16]
# - Set train_seeds: [0]
# - Reduce n_train, n_val, n_test if needed
```

2. Run the pipeline:

```bash
python scripts/train_probe.py --config configs/dataset_main.yaml --doc-batch-size 1
python scripts/run_eval.py --config configs/dataset_main.yaml --doc-batch-size 1
python scripts/make_plots.py --config configs/dataset_main.yaml
```

## Expected Improvements from Code Changes

### 1. Proper Evaluation Metrics
- `fpr_intrinsic_at_tpr_0.95`: True FPR@TPR via ROC interpolation (threshold-free)
- `fpr_fixed_at_tpr_0.95`: FPR at fixed validation threshold
- `tpr_fixed_at_tpr_0.95`: Actual TPR at that threshold (should vary by condition)

### 2. Token-Matched Negative Sampling
The false positive triggers like "the", "NoSQL", "one" should be reduced because:
- Negatives now include tokens with same IDs as positives
- Forces probe to learn context, not token identity

Check `results/analysis/*_triggers_*.txt` to verify improvement.

### 3. Layer Sweep with Error Bars
- Plot 3 now shows performance across layers 4-24
- Error bars from 3 seeds show variance
- Helps identify optimal readout depth

### 4. Diagnostic Plots
- Layer heatmap shows all aggregators × layers at once
- Calibration comparison shows calibrated vs uncalibrated FPR

## Metrics to Report in Write-up

From `results/metrics/needle_haystack_v1.0_eval_metrics.csv`:

1. **Main claim**: `fpr_intrinsic_at_tpr_0.95` for max vs best aggregator at long length
2. **Length scaling**: `fpr_intrinsic_at_tpr_0.95` vs `length_bucket` (Plot 1)
3. **Distractor effect**: `fpr_intrinsic_at_tpr_0.95` vs `distractor_level` (Plot 2)
4. **Layer localization**: `auroc` or `fpr_intrinsic_at_tpr_0.95` vs `layer` (Plot 3)

## Troubleshooting

### Out of Memory
- Reduce `--doc-batch-size` to 1
- Use `--device-map auto` for model sharding
- Reduce `max_train_docs` / `max_eval_docs` in config

### Slow Training
- Reduce `layers_to_probe` to fewer layers initially
- Reduce `train_seeds` to [0] for quick iteration
- Use `--extract-workers N` for parallel extraction on multi-GPU

### Poor Results
- Check `results/analysis/*_triggers_*.txt` for token-identity leakage
- Verify `token_matched_negatives: true` in config
- Ensure template families are held out in test split
