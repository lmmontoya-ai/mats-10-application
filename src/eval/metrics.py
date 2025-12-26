"""
Metrics for document-level evaluation.

This module provides two distinct metric regimes:
1. Intrinsic separability: FPR@TPR computed via ROC curve interpolation (threshold-free)
2. Fixed-threshold monitoring: FPR and TPR at a pre-selected threshold per condition
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.probes.metrics import compute_auroc


@dataclass
class ROCCurve:
    """ROC curve data for a single evaluation condition."""

    fprs: torch.Tensor
    tprs: torch.Tensor
    thresholds: torch.Tensor


def compute_roc_curve(scores: torch.Tensor, labels: torch.Tensor) -> ROCCurve:
    """Compute ROC curve from scores and binary labels.

    Returns (fprs, tprs, thresholds) sorted by decreasing threshold.
    """
    scores = scores.detach().flatten()
    labels = labels.detach().flatten().to(dtype=torch.float32)

    if scores.numel() == 0:
        return ROCCurve(
            fprs=torch.tensor([0.0, 1.0]),
            tprs=torch.tensor([0.0, 1.0]),
            thresholds=torch.tensor([1.0, 0.0]),
        )

    n_pos = int(labels.sum().item())
    n_neg = labels.numel() - n_pos

    if n_pos == 0 or n_neg == 0:
        return ROCCurve(
            fprs=torch.tensor([0.0, 1.0]),
            tprs=torch.tensor([0.0, 1.0]),
            thresholds=torch.tensor([1.0, 0.0]),
        )

    # Sort by decreasing score
    sorted_idx = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_idx]
    sorted_scores = scores[sorted_idx]

    # Cumulative TP/FP counts
    tp_cumsum = torch.cumsum(sorted_labels, dim=0)
    fp_cumsum = torch.cumsum(1.0 - sorted_labels, dim=0)

    # TPR and FPR at each threshold
    tprs = tp_cumsum / n_pos
    fprs = fp_cumsum / n_neg

    # Prepend (0, 0) for threshold = infinity
    tprs = torch.cat([torch.tensor([0.0]), tprs])
    fprs = torch.cat([torch.tensor([0.0]), fprs])
    thresholds = torch.cat([torch.tensor([float("inf")]), sorted_scores])

    return ROCCurve(fprs=fprs, tprs=tprs, thresholds=thresholds)


def fpr_at_tpr_interpolated(roc: ROCCurve, target_tpr: float) -> float:
    """Compute FPR at target TPR via linear interpolation on ROC curve.

    This is a threshold-free metric that measures intrinsic separability.
    """
    tprs = roc.tprs.detach().cpu().numpy()
    fprs = roc.fprs.detach().cpu().numpy()

    # Find interpolation point
    # We want the smallest FPR such that TPR >= target_tpr
    for i in range(len(tprs)):
        if tprs[i] >= target_tpr:
            if i == 0:
                return float(fprs[i])
            # Linear interpolation between points i-1 and i
            t0, t1 = tprs[i - 1], tprs[i]
            f0, f1 = fprs[i - 1], fprs[i]
            if t1 == t0:
                return float(f0)
            alpha = (target_tpr - t0) / (t1 - t0)
            return float(f0 + alpha * (f1 - f0))

    return float(fprs[-1])


def compute_auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().flatten()
    labels = labels.detach().flatten().to(dtype=torch.float32)
    if scores.numel() == 0:
        return 0.0
    n_pos = int(labels.sum().item())
    if n_pos == 0:
        return 0.0

    sorted_idx = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_idx]
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1.0 - sorted_labels, dim=0)
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / n_pos

    recall_prev = torch.cat([torch.tensor([0.0]), recall[:-1]])
    auprc = torch.sum((recall - recall_prev) * precision)
    return float(auprc.item())


def threshold_at_tpr(
    probs: torch.Tensor, labels: torch.Tensor, target_tpr: float
) -> float:
    """Find threshold that achieves target TPR on the given data.

    This is used for fixed-threshold monitoring evaluation.
    """
    labels = labels.detach().flatten()
    probs = probs.detach().flatten()
    pos_scores = probs[labels == 1]
    if pos_scores.numel() == 0:
        return 1.0
    sorted_scores, _ = torch.sort(pos_scores)
    n_pos = sorted_scores.numel()
    cutoff = max(0, int(math.ceil((1.0 - target_tpr) * n_pos)) - 1)
    cutoff = min(cutoff, n_pos - 1)
    return float(sorted_scores[cutoff].item())


def compute_fpr_tpr(
    probs: torch.Tensor, labels: torch.Tensor, threshold: float
) -> tuple[float, float]:
    """Compute FPR and TPR at a fixed threshold.

    Used for fixed-threshold monitoring evaluation where a single
    threshold is applied across all conditions.
    """
    labels = labels.detach().flatten().to(dtype=torch.int64)
    preds = (probs >= threshold).to(dtype=torch.int64)
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return fpr, tpr


def compute_intrinsic_fpr_at_tpr(
    scores: torch.Tensor, labels: torch.Tensor, target_tpr: float
) -> float:
    """Compute FPR@TPR via ROC curve interpolation (intrinsic metric).

    This is a threshold-free metric that measures the best achievable
    FPR at a given TPR, using all the data to construct the ROC curve.
    """
    scores = scores.detach().flatten()
    labels = labels.detach().flatten()
    roc = compute_roc_curve(scores, labels)
    return fpr_at_tpr_interpolated(roc, target_tpr)


@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""

    # Intrinsic separability (threshold-free)
    auroc: float
    auprc: float
    fpr_at_tpr_intrinsic: dict[float, float]  # target_tpr -> fpr

    # Fixed-threshold monitoring (from validation threshold)
    fpr_fixed: dict[float, float]  # target_tpr -> fpr at that threshold
    tpr_fixed: dict[float, float]  # target_tpr -> actual tpr at that threshold

    # Calibration
    ece: float
    brier: float

    # Sample counts
    n_pos: int
    n_neg: int


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    probs = probs.detach().flatten()
    labels = labels.detach().flatten()
    if probs.numel() == 0:
        return 0.0

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = torch.tensor(0.0)
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if not mask.any():
            continue
        bin_probs = probs[mask]
        bin_labels = labels[mask]
        acc = bin_labels.float().mean()
        conf = bin_probs.mean()
        ece += mask.float().mean() * (conf - acc).abs()
    return float(ece.item())


def compute_brier(probs: torch.Tensor, labels: torch.Tensor) -> float:
    probs = probs.detach().flatten()
    labels = labels.detach().flatten()
    if probs.numel() == 0:
        return 0.0
    return float(torch.mean((probs - labels) ** 2).item())


def compute_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Legacy metrics function for backwards compatibility."""
    scores = torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))
    return {
        "auroc": compute_auroc(scores, labels),
        "auprc": compute_auprc(scores, labels),
        "ece": compute_ece(probs, labels),
        "brier": compute_brier(probs, labels),
    }


def compute_full_metrics(
    scores: torch.Tensor,
    probs: torch.Tensor,
    labels: torch.Tensor,
    target_tprs: list[float],
    thresholds: dict[float, float],
) -> EvalMetrics:
    """Compute full evaluation metrics including intrinsic and fixed-threshold.

    Args:
        scores: Raw aggregator scores (logits)
        probs: Calibrated probabilities
        labels: Binary labels
        target_tprs: Target TPR values to compute FPR at
        thresholds: Pre-computed thresholds from validation (target_tpr -> threshold)

    Returns:
        EvalMetrics with both intrinsic and fixed-threshold metrics
    """
    labels = labels.detach().flatten()
    scores = scores.detach().flatten()
    probs = probs.detach().flatten()

    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())

    # Intrinsic metrics (threshold-free)
    auroc = compute_auroc(scores, labels)
    auprc = compute_auprc(scores, labels)

    fpr_at_tpr_intrinsic = {}
    for target_tpr in target_tprs:
        fpr_at_tpr_intrinsic[target_tpr] = compute_intrinsic_fpr_at_tpr(
            scores, labels, target_tpr
        )

    # Fixed-threshold metrics
    fpr_fixed = {}
    tpr_fixed = {}
    for target_tpr, threshold in thresholds.items():
        fpr, tpr = compute_fpr_tpr(probs, labels, threshold)
        fpr_fixed[target_tpr] = fpr
        tpr_fixed[target_tpr] = tpr

    # Calibration metrics
    ece = compute_ece(probs, labels)
    brier = compute_brier(probs, labels)

    return EvalMetrics(
        auroc=auroc,
        auprc=auprc,
        fpr_at_tpr_intrinsic=fpr_at_tpr_intrinsic,
        fpr_fixed=fpr_fixed,
        tpr_fixed=tpr_fixed,
        ece=ece,
        brier=brier,
        n_pos=n_pos,
        n_neg=n_neg,
    )
