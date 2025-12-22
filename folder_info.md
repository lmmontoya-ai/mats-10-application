# Folder Structure Guide

This document describes the folder structure and conventions for this project.

## Directory Structure

### `configs/`
Contains configuration files for the main run and optional overrides.
- **Format**: YAML or JSON
- **Purpose**: Centralized configuration management
- **Files**:
  - Main config file (e.g., `config.yaml` or `config.json`)
  - Optional override files for different experiments or environments

### `data_raw/`
Contains raw, immutable data sources.
- **Content**: Filler-text corpus or downloaded text segments
- **Immutable**: Files in this directory should not be modified after initial creation
- **Purpose**: Source data that feeds into dataset generation pipelines

### `data/`
Contains processed datasets and data splits.
- **Format**: JSONL (JSON Lines) for datasets
- **Content**:
  - Generated datasets
  - Split manifests (train/val/test splits)
- **Purpose**: Processed data ready for model training and evaluation

### `src/`
Contains importable Python modules organized by functionality.
- **Structure**: Modular code organized by purpose
- **Modules**:
  - `dataset/` - Dataset loading and processing utilities
  - `features/` - Feature extraction and engineering
  - `probes/` - Probing models and analysis tools
  - `eval/` - Evaluation metrics and procedures
  - `plots/` - Plotting and visualization utilities
- **Purpose**: Reusable, importable code components

### `scripts/`
Contains thin CLI wrapper scripts.
- **Purpose**: Command-line entry points that load configuration and call functions from `src/`
- **Pattern**: Each script should be minimal, loading config and delegating to `src/` modules
- **Usage**: Scripts are the primary way to run experiments and analyses

### `results/`
Contains outputs from experiments and analyses.
- **Content**:
  - Metrics tables (CSV, JSON, or other formats)
  - Calibration objects (pickled models or serialized data)
  - Plots and visualizations (PNG, PDF, SVG, etc.)
- **Purpose**: All experimental outputs and results

### `logs/`
Contains run metadata and logging information.
- **Content**:
  - Git hash/commit information
  - Environment details (Python version, package versions)
  - Random seed values
  - Timestamps
  - Run logs and execution traces
- **Purpose**: Reproducibility and debugging information

## Conventions

- **Immutability**: `data_raw/` should be treated as read-only after initial population
- **Modularity**: Keep business logic in `src/`, keep scripts thin
- **Reproducibility**: Always log git hash, environment, and seed in `logs/`
- **Configuration**: Use `configs/` for all experiment parameters
- **Outputs**: All generated outputs go to `results/` or `logs/`
