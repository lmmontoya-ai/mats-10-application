"""
Utility helpers for probe training.
"""

from __future__ import annotations

import os
import random
from typing import Literal

import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """Resolve runtime device from config string."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    """Resolve model dtype from config string."""
    if dtype_name == "auto":
        if device.type in {"cuda", "mps"}:
            return torch.float16
        return torch.float32
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_name}")


def resolve_feature_dtype(dtype_name: str) -> torch.dtype:
    """Resolve feature storage dtype."""
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    raise ValueError(f"Unknown feature dtype: {dtype_name}")


def disable_tokenizers_parallelism() -> None:
    """Disable tokenizers parallelism to avoid fork warnings."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
