"""
Metrics for document-level evaluation.
"""

from __future__ import annotations

import math

import torch

from src.probes.metrics import compute_auroc


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
    labels = labels.detach().flatten().to(dtype=torch.int64)
    preds = (probs >= threshold).to(dtype=torch.int64)
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return fpr, tpr


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
    scores = torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))
    return {
        "auroc": compute_auroc(scores, labels),
        "auprc": compute_auprc(scores, labels),
        "ece": compute_ece(probs, labels),
        "brier": compute_brier(probs, labels),
    }
