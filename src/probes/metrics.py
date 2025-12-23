"""
Metrics for probe training and evaluation.
"""

from __future__ import annotations

import torch


def compute_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute AUROC using rank statistics (assumes few exact ties)."""
    scores = scores.detach().flatten()
    labels = labels.detach().flatten().to(dtype=torch.float32)
    if scores.numel() == 0:
        return 0.0

    n_pos = int(labels.sum().item())
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Rank scores (1-based). Ties are ignored; logits are typically continuous.
    ranks = torch.argsort(scores).argsort().to(dtype=torch.float32) + 1.0
    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc.item())


def compute_binary_metrics(
    scores: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    """Compute accuracy/precision/recall/F1/AUROC at threshold 0.5."""
    probs = torch.sigmoid(scores)
    preds = (probs >= 0.5).to(dtype=torch.int64)
    labels_int = labels.to(dtype=torch.int64)

    tp = int(((preds == 1) & (labels_int == 1)).sum().item())
    fp = int(((preds == 1) & (labels_int == 0)).sum().item())
    tn = int(((preds == 0) & (labels_int == 0)).sum().item())
    fn = int(((preds == 0) & (labels_int == 1)).sum().item())

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )

    auroc = compute_auroc(scores, labels)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
    }
