"""
Document-level aggregation functions for token scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


EPS = 1e-6


@dataclass
class AggregatorOutput:
    """Container for aggregation outputs."""

    scores: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


def _safe_logit(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(EPS, 1.0 - EPS)
    return torch.log(probs / (1.0 - probs))


def max_pool(logits: torch.Tensor, mask: torch.Tensor) -> AggregatorOutput:
    probs = torch.sigmoid(logits)
    masked_probs = probs.masked_fill(~mask, 0.0)
    max_probs, max_indices = masked_probs.max(dim=1)
    scores = _safe_logit(max_probs)
    return AggregatorOutput(
        scores=scores,
        extras={"max_token_index": max_indices, "max_token_prob": max_probs},
    )


def noisy_or(logits: torch.Tensor, mask: torch.Tensor) -> AggregatorOutput:
    probs = torch.sigmoid(logits).clamp(EPS, 1.0 - EPS)
    log1m = torch.log1p(-probs)
    log1m = torch.where(mask, log1m, torch.zeros_like(log1m))
    sum_log1m = log1m.sum(dim=1)
    doc_prob = 1.0 - torch.exp(sum_log1m)
    scores = _safe_logit(doc_prob)
    return AggregatorOutput(scores=scores)


def logsumexp_pool(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    length_correction: bool,
) -> AggregatorOutput:
    scaled = logits / temperature
    mask_value = torch.finfo(scaled.dtype).min
    scaled = scaled.masked_fill(~mask, mask_value)
    lse = torch.logsumexp(scaled, dim=1)
    if length_correction:
        lengths = mask.sum(dim=1).clamp_min(1).to(dtype=logits.dtype)
        lse = lse - torch.log(lengths)
    return AggregatorOutput(scores=lse)


def topk_mean_pool(
    logits: torch.Tensor, mask: torch.Tensor, k: int
) -> AggregatorOutput:
    probs = torch.sigmoid(logits)
    scores = []
    for idx in range(probs.shape[0]):
        seq_len = int(mask[idx].sum().item())
        if seq_len <= 0:
            scores.append(torch.tensor(0.0, device=probs.device))
            continue
        k_eff = min(k, seq_len)
        topk_vals = torch.topk(probs[idx, :seq_len], k_eff).values
        doc_prob = topk_vals.mean()
        scores.append(_safe_logit(doc_prob))
    return AggregatorOutput(scores=torch.stack(scores, dim=0))


def ema_max_pool(
    logits: torch.Tensor, mask: torch.Tensor, alpha: float
) -> AggregatorOutput:
    probs = torch.sigmoid(logits)
    scores = []
    for idx in range(probs.shape[0]):
        seq_len = int(mask[idx].sum().item())
        if seq_len <= 0:
            scores.append(torch.tensor(0.0, device=probs.device))
            continue
        values = probs[idx, :seq_len]
        ema = torch.tensor(0.0, device=probs.device)
        max_ema = torch.tensor(0.0, device=probs.device)
        for token_prob in values:
            ema = alpha * token_prob + (1.0 - alpha) * ema
            max_ema = torch.maximum(max_ema, ema)
        scores.append(_safe_logit(max_ema))
    return AggregatorOutput(scores=torch.stack(scores, dim=0))


def windowed_max_mean_pool(
    logits: torch.Tensor, mask: torch.Tensor, window_size: int
) -> AggregatorOutput:
    probs = torch.sigmoid(logits)
    scores = []
    for idx in range(probs.shape[0]):
        seq_len = int(mask[idx].sum().item())
        if seq_len <= 0:
            scores.append(torch.tensor(0.0, device=probs.device))
            continue
        values = probs[idx, :seq_len]
        if seq_len <= window_size:
            doc_prob = values.mean()
            scores.append(_safe_logit(doc_prob))
            continue
        pooled = F.avg_pool1d(values.view(1, 1, -1), kernel_size=window_size, stride=1)
        doc_prob = pooled.max().squeeze(0).squeeze(0)
        scores.append(_safe_logit(doc_prob))
    return AggregatorOutput(scores=torch.stack(scores, dim=0))


def build_aggregators(
    names: list[str],
    temperature: float,
    length_correction: bool,
    topk_k: int | None,
    ema_alpha: float,
    window_size: int,
) -> dict[str, callable]:
    registry: dict[str, callable] = {}
    for name in names:
        if name == "max":
            registry[name] = max_pool
        elif name == "noisy_or":
            registry[name] = noisy_or
        elif name == "logsumexp":
            registry[name] = lambda logits, mask, t=temperature, lc=length_correction: (
                logsumexp_pool(logits, mask, t, lc)
            )
        elif name == "topk_mean":
            if topk_k is None:
                raise ValueError("topk_k must be set when using topk_mean")
            registry[name] = lambda logits, mask, k=topk_k: topk_mean_pool(
                logits, mask, k
            )
        elif name == "ema_max":
            registry[name] = lambda logits, mask, a=ema_alpha: ema_max_pool(
                logits, mask, a
            )
        elif name == "windowed_max_mean":
            registry[name] = lambda logits, mask, w=window_size: (
                windowed_max_mean_pool(logits, mask, w)
            )
        else:
            raise ValueError(f"Unknown aggregator: {name}")
    return registry
