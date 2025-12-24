"""
Activation extraction utilities for token-level probes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from transformers import PreTrainedModel


def _named_modulelist_candidates(
    model: PreTrainedModel,
    target_len: int | None,
) -> list[tuple[str, nn.ModuleList]]:
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList):
            if target_len is None or len(module) == target_len:
                candidates.append((name, module))
    return candidates


def _score_modulelist(name: str) -> tuple[int, int, int, str]:
    if name.endswith("layers"):
        rank = 0
    elif name.endswith("blocks"):
        rank = 1
    elif name.endswith("h"):
        rank = 2
    else:
        rank = 3
    depth = name.count(".")
    return (rank, depth, len(name), name)


def resolve_transformer_layers(model: PreTrainedModel) -> tuple[list[nn.Module], str]:
    """Resolve the transformer block list for common decoder-only architectures."""
    attribute_paths = [
        ("model.layers", ["model", "layers"]),
        ("model.model.layers", ["model", "model", "layers"]),
        ("model.decoder.layers", ["model", "decoder", "layers"]),
        ("model.model.decoder.layers", ["model", "model", "decoder", "layers"]),
        ("decoder.layers", ["decoder", "layers"]),
        ("transformer.h", ["transformer", "h"]),
        ("transformer.layers", ["transformer", "layers"]),
        ("gpt_neox.layers", ["gpt_neox", "layers"]),
        ("base_model.layers", ["base_model", "layers"]),
        ("base_model.model.layers", ["base_model", "model", "layers"]),
        ("language_model.layers", ["language_model", "layers"]),
    ]

    for path_name, attrs in attribute_paths:
        current = model
        found = True
        for attr in attrs:
            if not hasattr(current, attr):
                found = False
                break
            current = getattr(current, attr)
        if found and isinstance(current, nn.ModuleList):
            return list(current), path_name

    target_len = getattr(model.config, "num_hidden_layers", None)
    candidates = _named_modulelist_candidates(model, target_len)
    if candidates:
        candidates.sort(key=lambda item: _score_modulelist(item[0]))
        name, modulelist = candidates[0]
        return list(modulelist), name

    # Fallback: choose the longest ModuleList available.
    candidates = _named_modulelist_candidates(model, None)
    if candidates:
        name, modulelist = max(candidates, key=lambda item: len(item[1]))
        return list(modulelist), name

    raise ValueError("Unable to locate transformer layers on the model")


@dataclass
class ActivationExtractor:
    """Extracts activations from a specified transformer layer."""

    model: PreTrainedModel
    layer_idx: int
    activation_site: str
    device: torch.device
    feature_dtype: torch.dtype

    def __post_init__(self) -> None:
        layers, layer_path = resolve_transformer_layers(self.model)
        if not (0 <= self.layer_idx < len(layers)):
            raise ValueError(
                f"layer_idx {self.layer_idx} out of range for {layer_path} "
                f"(num_layers={len(layers)})"
            )
        self.layer = layers[self.layer_idx]
        self.layer_path = layer_path
        if self.activation_site != "resid_post":
            raise ValueError(f"Unsupported activation_site: {self.activation_site}")

    def _capture_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        captured: dict[str, torch.Tensor | None] = {"value": None}

        def _hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: object) -> None:
            if isinstance(output, tuple):
                captured["value"] = output[0]
            else:
                captured["value"] = output  # type: ignore[assignment]

        handle = self.layer.register_forward_hook(_hook)
        try:
            with torch.inference_mode():
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            handle.remove()

        hidden = captured["value"]
        if hidden is None:
            raise RuntimeError("Activation hook did not capture any output")
        return hidden

    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model and return hidden states for all tokens."""
        hidden = self._capture_hidden(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        if hidden.dim() != 3:
            raise RuntimeError(f"Unexpected hidden state shape: {hidden.shape}")
        # Convert to feature_dtype to match training (avoids bfloat16 precision issues)
        return hidden.to(dtype=self.feature_dtype)

    def extract_selected_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_indices: Iterable[int],
    ) -> torch.Tensor:
        """Run the model and return activations for selected token indices."""
        indices = list(token_indices)
        if not indices:
            return torch.empty((0, 0), dtype=self.feature_dtype)

        hidden = self._capture_hidden(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if hidden.dim() == 3:
            hidden = hidden[0]
        elif hidden.dim() != 2:
            raise RuntimeError(f"Unexpected hidden state shape: {hidden.shape}")

        index_tensor = torch.tensor(indices, device=hidden.device)
        selected = hidden.index_select(0, index_tensor)
        return selected.detach().to("cpu", dtype=self.feature_dtype)

    def extract_selected_tokens_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_indices_per_example: list[list[int]],
    ) -> list[torch.Tensor]:
        """Run the model on a batch and return per-example activations."""
        if not token_indices_per_example:
            return []

        hidden = self._capture_hidden(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        if hidden.dim() != 3:
            raise RuntimeError(f"Unexpected hidden state shape: {hidden.shape}")

        features: list[torch.Tensor] = []
        for batch_idx, indices in enumerate(token_indices_per_example):
            if not indices:
                features.append(
                    torch.empty((0, hidden.shape[-1]), dtype=self.feature_dtype)
                )
                continue
            idx_tensor = torch.tensor(indices, device=hidden.device)
            selected = hidden[batch_idx].index_select(0, idx_tensor)
            features.append(selected.detach().to("cpu", dtype=self.feature_dtype))
        return features
