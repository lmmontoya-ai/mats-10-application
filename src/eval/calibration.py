"""
Calibration helpers for document-level scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PlattCalibrator:
    """Platt scaling calibrator: sigmoid(a * score + b)."""

    a: float
    b: float

    def predict_proba(self, scores: torch.Tensor) -> torch.Tensor:
        logits = self.a * scores + self.b
        return torch.sigmoid(logits)

    def state_dict(self) -> dict[str, float]:
        return {"a": float(self.a), "b": float(self.b)}


@dataclass
class IdentityCalibrator:
    """No calibration; pass scores through sigmoid."""

    def predict_proba(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(scores)

    def state_dict(self) -> dict[str, float]:
        return {}


def fit_platt(
    scores: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.1,
) -> PlattCalibrator:
    scores = scores.detach().flatten()
    labels = labels.detach().flatten()
    if scores.numel() == 0:
        raise ValueError("No scores provided for calibration")

    a = torch.tensor(1.0, device=scores.device, requires_grad=True)
    b = torch.tensor(0.0, device=scores.device, requires_grad=True)

    optimizer = torch.optim.Adam([a, b], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        logits = a * scores + b
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    return PlattCalibrator(a=float(a.item()), b=float(b.item()))
