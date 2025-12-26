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


def _select_best_layer(
    rows: list[dict[str, object]],
    aggregator: str,
    metric_key: str,
    calibrated_label: str,
    length_bucket: object,
) -> int | None:
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
    if not layer_to_vals:
        return None
    if metric_key.startswith("fpr_") or metric_key.startswith("brier"):
        return min(layer_to_vals.items(), key=lambda item: sum(item[1]) / len(item[1]))[
            0
        ]
    return max(layer_to_vals.items(), key=lambda item: sum(item[1]) / len(item[1]))[0]


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
    means = [sum(layer_to_vals[l]) / len(layer_to_vals[l]) for l in layers]

    # Compute std for error bars if multiple seeds
    stds = []
    for l in layers:
        vals = layer_to_vals[l]
        if len(vals) > 1:
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            stds.append(variance**0.5)
        else:
            stds.append(0.0)

    plt.figure(figsize=(6, 4))
    if any(s > 0 for s in stds):
        plt.errorbar(layers, means, yerr=stds, marker="o", capsize=3)
    else:
        plt.plot(layers, means, marker="o")
    plt.xlabel("Layer")
    plt.ylabel(metric_key.replace("_", " ").upper())
    plt.title(f"Performance vs layer ({aggregator})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_layer_sweep_heatmap(
    rows: list[dict[str, object]],
    aggregators: list[str],
    length_bucket: object,
    metric_key: str,
    calibrated_label: str,
    output_path: Path,
) -> None:
    """Plot layer sweep as heatmap with aggregators on y-axis."""
    import numpy as np

    # Collect data
    layer_set = set()
    data: dict[str, dict[int, float]] = {}

    for agg in aggregators:
        subset = _filter_rows(
            rows,
            split="test",
            calibrated=calibrated_label,
            aggregator=agg,
            length_bucket=length_bucket,
            distractor_level="all",
        )
        layer_to_vals: dict[int, list[float]] = {}
        for row in subset:
            layer = int(row["layer"])
            layer_set.add(layer)
            layer_to_vals.setdefault(layer, []).append(float(row.get(metric_key, 0.0)))

        data[agg] = {l: sum(vs) / len(vs) for l, vs in layer_to_vals.items()}

    if not layer_set or not data:
        return

    layers = sorted(layer_set)
    matrix = np.zeros((len(aggregators), len(layers)))
    for i, agg in enumerate(aggregators):
        for j, layer in enumerate(layers):
            matrix[i, j] = data.get(agg, {}).get(layer, 0.0)

    plt.figure(figsize=(8, 4))
    plt.imshow(
        matrix, aspect="auto", cmap="RdYlGn_r" if "fpr" in metric_key else "RdYlGn"
    )
    plt.colorbar(label=metric_key.replace("_", " "))
    plt.xticks(range(len(layers)), layers)
    plt.yticks(range(len(aggregators)), aggregators)
    plt.xlabel("Layer")
    plt.ylabel("Aggregator")
    plt.title(f"Layer sweep heatmap (length={length_bucket})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fpr_comparison_calibrated_vs_uncalibrated(
    rows: list[dict[str, object]],
    aggregators: list[str],
    length_bucket: object,
    metric_key: str,
    output_path: Path,
) -> None:
    """Bar chart comparing calibrated vs uncalibrated FPR for each aggregator."""
    import numpy as np

    uncal_data = {}
    cal_data = {}

    for agg in aggregators:
        uncal_subset = _filter_rows(
            rows,
            split="test",
            calibrated="uncalibrated",
            aggregator=agg,
            length_bucket=length_bucket,
            distractor_level="all",
        )
        cal_subset = _filter_rows(
            rows,
            split="test",
            calibrated="calibrated",
            aggregator=agg,
            length_bucket=length_bucket,
            distractor_level="all",
        )

        if uncal_subset:
            vals = [float(r.get(metric_key, 0.0)) for r in uncal_subset]
            uncal_data[agg] = sum(vals) / len(vals)
        if cal_subset:
            vals = [float(r.get(metric_key, 0.0)) for r in cal_subset]
            cal_data[agg] = sum(vals) / len(vals)

    if not uncal_data and not cal_data:
        return

    x = np.arange(len(aggregators))
    width = 0.35

    plt.figure(figsize=(8, 4))
    if uncal_data:
        uncal_vals = [uncal_data.get(a, 0.0) for a in aggregators]
        plt.bar(x - width / 2, uncal_vals, width, label="Uncalibrated")
    if cal_data:
        cal_vals = [cal_data.get(a, 0.0) for a in aggregators]
        plt.bar(x + width / 2, cal_vals, width, label="Calibrated")

    plt.xlabel("Aggregator")
    plt.ylabel(metric_key.replace("_", " ").upper())
    plt.title(f"Calibrated vs Uncalibrated (length={length_bucket})")
    plt.xticks(x, aggregators, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_token_score_diagnostics(
    rows: list[dict[str, object]],
    length_buckets: list[int],
    diagnostic_threshold: float,
    layer: int | None,
    output_path: Path,
) -> None:
    """Plot negative-document token-score diagnostics vs length."""
    import numpy as np

    if not rows or not length_buckets:
        return

    metrics = [
        ("max_token_prob", "Max token prob"),
        ("second_max_token_prob", "Second-max token prob"),
        ("num_tokens_above_threshold", f"Tokens >= {diagnostic_threshold:.2f}"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 8), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, metrics):
        medians = []
        p10 = []
        p90 = []
        for bucket in length_buckets:
            vals = [
                float(r[key])
                for r in rows
                if r.get("length_bucket") == bucket and r.get(key) is not None
            ]
            if not vals:
                medians.append(0.0)
                p10.append(0.0)
                p90.append(0.0)
                continue
            medians.append(float(np.percentile(vals, 50)))
            p10.append(float(np.percentile(vals, 10)))
            p90.append(float(np.percentile(vals, 90)))
        yerr = [
            [m - lo for m, lo in zip(medians, p10)],
            [hi - m for m, hi in zip(medians, p90)],
        ]
        ax.errorbar(length_buckets, medians, yerr=yerr, marker="o", capsize=3)
        ax.set_ylabel(label)

    axes[-1].set_xlabel("Context length (tokens)")
    title = "Negative token-score diagnostics"
    if layer is not None:
        title += f" (layer={layer})"
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    eval_cfg = config.eval

    run_id = f"{config.output.dataset_name}_{config.output.dataset_version}"
    metrics_path = Path(eval_cfg.results_dir) / "metrics" / f"{run_id}_eval_metrics.csv"
    rows = _load_metrics(metrics_path)

    # Use intrinsic FPR@TPR by default (threshold-free, via ROC interpolation)
    # Falls back to legacy metric name if new columns not present
    metric_key = eval_cfg.plot_layer_metric
    if metric_key.startswith("fpr_at_tpr_"):
        # Check if we have the new intrinsic metric
        tpr_suffix = metric_key.replace("fpr_at_tpr_", "")
        intrinsic_key = f"fpr_intrinsic_at_tpr_{tpr_suffix}"
        if rows and intrinsic_key in rows[0]:
            metric_key = intrinsic_key
            logger.info("Using intrinsic FPR metric: %s", metric_key)

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

    # Core Plot 1: FPR vs context length
    plot_fpr_vs_length(
        rows=rows,
        aggregators=aggregators,
        distractor_level=plot_distractor,
        metric_key=metric_key,
        calibrated_label=calibrated_label,
        output_path=plots_dir / f"{run_id}_plot1_fpr_vs_length.png",
    )

    # Core Plot 2: FPR vs distractor density
    plot_fpr_vs_distractor(
        rows=rows,
        aggregators=aggregators,
        length_bucket=plot_length,
        metric_key=metric_key,
        calibrated_label=calibrated_label,
        output_path=plots_dir / f"{run_id}_plot2_fpr_vs_distractor.png",
    )

    # Core Plot 3: Performance vs layer (with error bars if multiple seeds)
    plot_performance_vs_layer(
        rows=rows,
        aggregator=eval_cfg.plot_layer_aggregator,
        length_bucket=plot_length,
        metric_key=metric_key,
        calibrated_label=calibrated_label,
        output_path=plots_dir / f"{run_id}_plot3_layer_sweep.png",
    )

    # Diagnostic Plot 4: Layer sweep heatmap (all aggregators)
    plot_layer_sweep_heatmap(
        rows=rows,
        aggregators=aggregators,
        length_bucket=plot_length,
        metric_key=metric_key,
        calibrated_label=calibrated_label,
        output_path=plots_dir / f"{run_id}_plot4_layer_heatmap.png",
    )

    # Diagnostic Plot 5: Calibrated vs uncalibrated comparison
    plot_fpr_comparison_calibrated_vs_uncalibrated(
        rows=rows,
        aggregators=aggregators,
        length_bucket=plot_length,
        metric_key=metric_key,
        output_path=plots_dir / f"{run_id}_plot5_calibration_comparison.png",
    )

    # Plot for each distractor level at longest length
    for dist_level in distractor_levels:
        plot_fpr_vs_length(
            rows=rows,
            aggregators=aggregators,
            distractor_level=dist_level,
            metric_key=metric_key,
            calibrated_label=calibrated_label,
            output_path=plots_dir / f"{run_id}_fpr_vs_length_dist_{dist_level}.png",
        )

    diagnostic_path = (
        Path(eval_cfg.results_dir) / "analysis" / f"{run_id}_negative_token_stats.csv"
    )
    if diagnostic_path.exists():
        diagnostic_rows = _load_metrics(diagnostic_path)
        best_layer = _select_best_layer(
            rows=rows,
            aggregator=eval_cfg.plot_layer_aggregator,
            metric_key=metric_key,
            calibrated_label=calibrated_label,
            length_bucket=plot_length,
        )
        if best_layer is not None:
            diagnostic_rows = [
                r for r in diagnostic_rows if r.get("layer") == best_layer
            ]
        diagnostic_threshold = None
        for row in diagnostic_rows:
            if row.get("diagnostic_threshold") is not None:
                diagnostic_threshold = float(row["diagnostic_threshold"])
                break
        if diagnostic_threshold is None:
            diagnostic_threshold = eval_cfg.diagnostic_token_threshold or 0.9
        diag_length_buckets = sorted(
            {
                r["length_bucket"]
                for r in diagnostic_rows
                if isinstance(r.get("length_bucket"), int)
            }
        )
        plot_token_score_diagnostics(
            rows=diagnostic_rows,
            length_buckets=diag_length_buckets,
            diagnostic_threshold=diagnostic_threshold,
            layer=best_layer,
            output_path=plots_dir / f"{run_id}_plot6_token_diagnostics.png",
        )

    logger.info("Plots written to %s", plots_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
