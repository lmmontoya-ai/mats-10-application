# AGENTS.md

## Project overview

This repository implements a short, reproducible mechanistic-interpretability-style study of probe-based monitoring in long contexts. The core question is whether classic “scan all tokens then fire” probe monitors suffer predictable false-positive inflation as context length and distractor density increase, and whether a minimal change in aggregation and calibration can substantially stabilize false positives at a fixed recall level.

The project is intentionally designed to be configuration-driven and easy to re-run end-to-end. All results should be reproducible from a single config file, and the primary outputs are three headline plots plus a small set of tidy metrics tables.

## Success criteria

A run is considered “successful” when the pipeline produces (1) a dataset with template-family-held-out splits, (2) trained token-level probes saved with metadata, (3) calibrated document-level monitors for multiple aggregators, (4) a metrics table spanning the experiment matrix, and (5) the three core plots below.

The three core plots are:
1. False positive rate at fixed recall versus context length (lines = aggregators).
2. False positive rate at fixed recall versus distractor density at a fixed long length (lines = aggregators).
3. Performance versus layer (curve or heatmap), using the best aggregator and a fixed long length.

## Non-goals

This repo is not intended to build a production monitor, a large benchmark suite, or a full mechanistic circuit explanation. It also intentionally avoids SAE-heavy workflows. The focus is a pragmatic, disciplined experiment with strong baselines, robust controls, and clear plots.

## Model and scope defaults

The default target model is a long-context, open-weight transformer suitable for activation extraction. The recommended default is `google/gemma-3-4b-it` (or equivalent Gemma 3 4B instruction-tuned checkpoint). If a second model is used for replication, treat it as optional and do not allow it to threaten the core deliverables.

Do not use very small or outdated baselines (for example GPT-2). The project relies on a model that can meaningfully represent long-context structure.

## Repository conventions

The codebase should remain small and pipeline-oriented. Use a single “main” config and keep all stage outputs in `results/` and `data/` with run IDs.

Recommended layout:

| Path | Purpose |
| --- | --- |
| `configs/` | YAML/JSON configs defining the full run |
| `data_raw/` | Immutable filler text segments or downloaded text (do not mutate) |
| `data/` | Generated datasets (JSONL), split manifests, QA reports |
| `src/` | Importable modules for dataset, features, probes, eval, plotting |
| `scripts/` | Thin CLIs that load config and call `src/` |
| `results/` | Metrics tables, calibration objects, plots, run records |
| `logs/` | Console logs or experiment logs |

Every pipeline stage must write a `run_record.json` capturing model ID, tokenizer ID, seeds, dataset versioning metadata, hyperparameter searched, selected hyperparameter, and timestamps.

## Environment setup

This project is intended to run on macOS or Linux, with GPU strongly preferred for long-context activation extraction. Keep the environment instructions simple and deterministic.

Create an isolated environment and install dependencies using `uv`:

```bash
uv venv
source .venv/bin/activate

# Install dependencies (choose one supported option).
uv pip install -r requirements.txt
# or
uv pip install -e .
