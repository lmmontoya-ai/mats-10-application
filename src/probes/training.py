"""
Training utilities for linear probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import compute_binary_metrics


@dataclass
class TrainConfig:
    """Training hyperparameters for a linear probe."""

    batch_size: int
    num_epochs: int
    learning_rate: float
    class_balance: str


def _build_dataloader(
    features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    config: TrainConfig,
    l2_weight: float,
    device: torch.device,
) -> nn.Module:
    """Train a linear probe with BCEWithLogitsLoss."""
    if train_features.numel() == 0:
        raise ValueError("Training features are empty")

    model = nn.Linear(train_features.shape[1], 1).to(device)

    if config.class_balance == "weighted":
        pos_weight = (
            train_labels.numel() - train_labels.sum()
        ) / train_labels.sum().clamp_min(1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=l2_weight,
    )

    loader = _build_dataloader(
        train_features, train_labels, config.batch_size, shuffle=True
    )

    model.train()
    for _ in range(config.num_epochs):
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features).squeeze(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    return model


def evaluate_linear_probe(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a linear probe and return metrics."""
    if features.numel() == 0:
        raise ValueError("Evaluation features are empty")

    loader = _build_dataloader(features, labels, batch_size, shuffle=False)
    model.eval()
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            logits = model(batch_features).squeeze(-1).cpu()
            all_scores.append(logits)
            all_labels.append(batch_labels.cpu())

    scores = torch.cat(all_scores, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    return compute_binary_metrics(scores, labels_all)
