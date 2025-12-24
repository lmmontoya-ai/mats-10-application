#!/usr/bin/env python3
"""
Generate plots from evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.config import load_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make plots from eval metrics.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_main.yaml",
        help="Path to configuration YAML file.",
    )
    return parser.parse_args()


def _load_metrics(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = None
                    continue
                if value in {"", "None", "null"}:
                    parsed[key] = None
                    continue
                if key in {"length_bucket", "distractor_level"}:
                    try:
                        parsed[key] = int(value)
                    except ValueError:
                        parsed[key] = value
                else:
                    try:
                        if "." in value or "e" in value or "E" in value:
                            parsed[key] = float(value)
                        else:
                            parsed[key] = int(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


def _filter_rows(
    rows: list[dict[str, object]],
    split: str,
    calibrated: str,
    aggregator: str | None = None,
    length_bucket: object | None = None,
    distractor_level: object | None = None,
) -> list[dict[str, object]]:
    filtered = [
        r for r in rows if r.get("split") == split and r.get("calibrated") == calibrated
    ]
    if aggregator is not None:
        filtered = [r for r in filtered if r.get("aggregator") == aggregator]
    if length_bucket is not None:
        filtered = [r for r in filtered if r.get("length_bucket") == length_bucket]
    if distractor_level is not None:
        filtered = [
            r for r in filtered if r.get("distractor_level") == distractor_level
        ]
    return filtered


def _unique_sorted(values: list[object]) -> list[object]:
    numeric = []
    non_numeric = []
    for v in values:
        if isinstance(v, (int, float)):
            numeric.append(v)
        else:
            non_numeric.append(v)
    return sorted(numeric) + sorted(non_numeric)


def plot_fpr_vs_length(
    rows: list[dict[str, object]],
    aggregators: list[str],
    distractor_level: object,
    metric_key: str,
    calibrated_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    for agg in aggregators:
        subset = _filter_rows(
            rows,
            split="test",
            calibrated=calibrated_label,
            aggregator=agg,
            distractor_level=distractor_level,
        )
        subset = [r for r in subset if r.get("length_bucket") not in {"all", None}]
        buckets = _unique_sorted([r["length_bucket"] for r in subset])
        values = []
        for bucket in buckets:
            match_rows = [r for r in subset if r["length_bucket"] == bucket]
            if not match_rows:
                values.append(None)
                continue
            vals = [float(r.get(metric_key, 0.0)) for r in match_rows]
            values.append(sum(vals) / len(vals))
        plt.plot(buckets, values, marker="o", label=agg)

    plt.xlabel("Context length (tokens)")
    plt.ylabel(metric_key.replace("_", " ").upper())
    plt.title(f"FPR vs length (distractors={distractor_level})")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fpr_vs_distractor(
    rows: list[dict[str, object]],
    aggregators: list[str],
    length_bucket: object,
    metric_key: str,
    calibrated_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    for agg in aggregators:
        subset = _filter_rows(
            rows,
            split="test",
            calibrated=calibrated_label,
            aggregator=agg,
            length_bucket=length_bucket,
        )
        subset = [r for r in subset if r.get("distractor_level") not in {"all", None}]
        levels = _unique_sorted([r["distractor_level"] for r in subset])
        values = []
        for level in levels:
            match_rows = [r for r in subset if r["distractor_level"] == level]
            if not match_rows:
                values.append(None)
                continue
            vals = [float(r.get(metric_key, 0.0)) for r in match_rows]
            values.append(sum(vals) / len(vals))
        plt.plot(levels, values, marker="o", label=agg)

    plt.xlabel("Distractor level")
    plt.ylabel(metric_key.replace("_", " ").upper())
    plt.title(f"FPR vs distractors (length={length_bucket})")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_performance_vs_layer(
    rows: list[dict[str, object]],
    aggregator: str,
    length_bucket: object,
    metric_key: str,
    calibrated_label: str,
    output_path: Path,
) -> None:
    subset = _filter_rows(
        rows,
        split="test",
        calibrated=calibrated_label,
        aggregator=aggregator,
        length_bucket=length_bucket,
        distractor_level="all",
    )
    layer_to_vals: dict[int, list[float]] = {}
    for row in subset:
        layer = int(row["layer"])
        layer_to_vals.setdefault(layer, []).append(float(row.get(metric_key, 0.0)))

    layers = sorted(layer_to_vals.keys())
    values = [sum(layer_to_vals[l]) / len(layer_to_vals[l]) for l in layers]

    plt.figure(figsize=(6, 4))
    plt.plot(layers, values, marker="o")
    plt.xlabel("Layer")
    plt.ylabel(metric_key.replace("_", " ").upper())
    plt.title(f"Performance vs layer ({aggregator})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    eval_cfg = config.eval

    run_id = f"{config.output.dataset_name}_{config.output.dataset_version}"
    metrics_path = Path(eval_cfg.results_dir) / "metrics" / f"{run_id}_eval_metrics.csv"
    rows = _load_metrics(metrics_path)

    metric_key = eval_cfg.plot_layer_metric
    aggregators = eval_cfg.aggregators

    calibrated_label = "calibrated"
    if not any(r.get("calibrated") == calibrated_label for r in rows):
        calibrated_label = "uncalibrated"

    length_buckets = sorted(
        {r["length_bucket"] for r in rows if isinstance(r.get("length_bucket"), int)}
    )
    distractor_levels = sorted(
        {
            r["distractor_level"]
            for r in rows
            if isinstance(r.get("distractor_level"), int)
        }
    )
    if not length_buckets or not distractor_levels:
        raise RuntimeError("No length buckets or distractor levels found in metrics.")

    plot_distractor = (
        eval_cfg.plot_distractor_level
        if eval_cfg.plot_distractor_level is not None
        else min(distractor_levels)
    )
    plot_length = (
        eval_cfg.plot_length_bucket
        if eval_cfg.plot_length_bucket is not None
        else max(length_buckets)
    )

    if eval_cfg.plot_layer_aggregator is None:
        overall = _filter_rows(
            rows,
            split="test",
            calibrated=calibrated_label,
            length_bucket="all",
            distractor_level="all",
        )
        agg_scores: dict[str, float] = {}
        for row in overall:
            agg = row.get("aggregator")
            if agg not in aggregators:
                continue
            agg_scores.setdefault(agg, []).append(float(row.get(metric_key, 0.0)))
        if agg_scores:
            if metric_key.startswith("fpr_") or metric_key.startswith("brier"):
                eval_cfg.plot_layer_aggregator = min(
                    agg_scores.items(),
                    key=lambda item: sum(item[1]) / len(item[1]),
                )[0]
            else:
                eval_cfg.plot_layer_aggregator = max(
                    agg_scores.items(),
                    key=lambda item: sum(item[1]) / len(item[1]),
                )[0]
        else:
            eval_cfg.plot_layer_aggregator = aggregators[0]

    plots_dir = Path(eval_cfg.results_dir) / "plots"
    plot_fpr_vs_length(
        rows=rows,
        aggregators=aggregators,
        distractor_level=plot_distractor,
        metric_key=metric_key,
        calibrated_label=calibrated_label,
        output_path=plots_dir / f"{run_id}_plot1_fpr_vs_length.png",
    )
    plot_fpr_vs_distractor(
        rows=rows,
        aggregators=aggregators,
        length_bucket=plot_length,
        metric_key=metric_key,
        calibrated_label=calibrated_label,
        output_path=plots_dir / f"{run_id}_plot2_fpr_vs_distractor.png",
    )
    plot_performance_vs_layer(
        rows=rows,
        aggregator=eval_cfg.plot_layer_aggregator,
        length_bucket=plot_length,
        metric_key=metric_key,
        calibrated_label=calibrated_label,
        output_path=plots_dir / f"{run_id}_plot3_layer_sweep.png",
    )

    logger.info("Plots written to %s", plots_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
