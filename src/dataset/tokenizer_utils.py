"""
Tokenizer utilities for length enforcement and span mapping.

This module provides robust character-to-token mapping and length
enforcement for dataset generation.
"""

from dataclasses import dataclass
import logging
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import TokenizerConfig


logger = logging.getLogger(__name__)


@dataclass
class TokenSpan:
    """Result of character span to token span mapping."""

    token_indices: list[int]
    char_start: int
    char_end: int
    used_fallback: bool = False
    warning: str | None = None


class TokenizerWrapper:
    """
    Wrapper around HuggingFace tokenizer with span mapping utilities.

    Provides robust character-to-token index mapping using offset mappings
    when available, with fallback for tokenizers that don't support them.
    """

    def __init__(self, config: TokenizerConfig):
        """
        Initialize tokenizer wrapper.

        Args:
            config: Tokenizer configuration.
        """
        self.config = config
        self.model_id = config.model_id
        self.tokenizer_id = config.tokenizer_id or config.model_id
        self.tokenizer_revision = config.tokenizer_revision

        # Load tokenizer
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            revision=self.tokenizer_revision,
            trust_remote_code=True,
        )

        # Check if tokenizer supports offset mapping
        self._supports_offsets = self._check_offset_support()
        if not self._supports_offsets:
            logger.warning(
                f"Tokenizer {self.tokenizer_id} does not support offset mapping. "
                f"Using fallback character search method."
            )

    def _check_offset_support(self) -> bool:
        """Check if tokenizer supports return_offsets_mapping."""
        try:
            result = self.tokenizer(
                "test string",
                return_offsets_mapping=True,
                return_tensors=None,
            )
            return "offset_mapping" in result
        except Exception:
            return False

    def tokenize(self, text: str) -> dict[str, Any]:
        """
        Tokenize text and return tokens with optional offset mapping.

        Args:
            text: Input text to tokenize.

        Returns:
            Dictionary with 'input_ids', 'tokens', and optionally 'offset_mapping'.
        """
        kwargs: dict[str, Any] = {
            "return_tensors": None,
            "add_special_tokens": False,  # Don't add special tokens for length counting
        }

        if self._supports_offsets:
            kwargs["return_offsets_mapping"] = True

        result = self.tokenizer(text, **kwargs)

        output = {
            "input_ids": result["input_ids"],
            "tokens": self.tokenizer.convert_ids_to_tokens(result["input_ids"]),
        }

        if self._supports_offsets and "offset_mapping" in result:
            output["offset_mapping"] = result["offset_mapping"]

        return output

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text.

        Returns:
            Number of tokens.
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def char_span_to_token_indices(
        self,
        text: str,
        char_start: int,
        char_end: int,
    ) -> TokenSpan:
        """
        Map character span to token indices.

        Uses offset mapping when available, falls back to search-based
        method otherwise.

        Args:
            text: Full document text.
            char_start: Start character index (inclusive).
            char_end: End character index (exclusive).

        Returns:
            TokenSpan with token indices and metadata.
        """
        if self._supports_offsets:
            return self._map_with_offsets(text, char_start, char_end)
        else:
            return self._map_with_fallback(text, char_start, char_end)

    def _map_with_offsets(
        self,
        text: str,
        char_start: int,
        char_end: int,
    ) -> TokenSpan:
        """Map using tokenizer offset mapping."""
        result = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors=None,
            add_special_tokens=False,
        )

        offset_mapping = result["offset_mapping"]
        token_indices = []

        for idx, (token_start, token_end) in enumerate(offset_mapping):
            # Check if token overlaps with target span
            if token_start < char_end and token_end > char_start:
                token_indices.append(idx)

        return TokenSpan(
            token_indices=token_indices,
            char_start=char_start,
            char_end=char_end,
            used_fallback=False,
        )

    def _map_with_fallback(
        self,
        text: str,
        char_start: int,
        char_end: int,
    ) -> TokenSpan:
        """
        Map using fallback search method.

        This tokenizes the full text and the needle substring separately,
        then searches for the needle tokens within the full token sequence.
        """
        # Get the needle text
        needle_text = text[char_start:char_end]

        # Tokenize full text and needle
        full_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        needle_tokens = self.tokenizer.encode(needle_text, add_special_tokens=False)

        if not needle_tokens:
            return TokenSpan(
                token_indices=[],
                char_start=char_start,
                char_end=char_end,
                used_fallback=True,
                warning="Needle produced no tokens",
            )

        # Search for needle tokens in full sequence
        # This is approximate - we look for the best matching window
        best_match_start = None
        best_match_score = 0

        for i in range(len(full_tokens) - len(needle_tokens) + 1):
            window = full_tokens[i : i + len(needle_tokens)]
            if window == needle_tokens:
                # Exact match found
                best_match_start = i
                best_match_score = len(needle_tokens)
                break
            # Track partial matches
            score = sum(1 for a, b in zip(window, needle_tokens) if a == b)
            if score > best_match_score:
                best_match_score = score
                best_match_start = i

        if best_match_start is None:
            return TokenSpan(
                token_indices=[],
                char_start=char_start,
                char_end=char_end,
                used_fallback=True,
                warning="Could not locate needle tokens in full sequence",
            )

        token_indices = list(
            range(best_match_start, best_match_start + len(needle_tokens))
        )

        warning = None
        if best_match_score < len(needle_tokens):
            warning = (
                f"Fallback mapping found partial match "
                f"({best_match_score}/{len(needle_tokens)} tokens)"
            )
            logger.warning(warning)

        return TokenSpan(
            token_indices=token_indices,
            char_start=char_start,
            char_end=char_end,
            used_fallback=True,
            warning=warning,
        )

    def is_within_tolerance(
        self,
        actual_tokens: int,
        target_tokens: int,
    ) -> bool:
        """
        Check if token count is within tolerance of target.

        Args:
            actual_tokens: Actual token count.
            target_tokens: Target token count.

        Returns:
            True if within tolerance.
        """
        tolerance = int(target_tokens * self.config.token_tolerance_ratio)
        return abs(actual_tokens - target_tokens) <= tolerance

    def get_tolerance_range(self, target_tokens: int) -> tuple[int, int]:
        """
        Get acceptable token count range for target.

        Args:
            target_tokens: Target token count.

        Returns:
            Tuple of (min_tokens, max_tokens).
        """
        tolerance = int(target_tokens * self.config.token_tolerance_ratio)
        return (target_tokens - tolerance, target_tokens + tolerance)

    @property
    def supports_offsets(self) -> bool:
        """Whether the tokenizer supports offset mappings."""
        return self._supports_offsets

    def get_tokenizer_info(self) -> dict[str, Any]:
        """Return tokenizer metadata for run records."""
        return {
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "is_fast": getattr(self.tokenizer, "is_fast", False),
            "supports_offsets": self._supports_offsets,
        }


def truncate_text_to_tokens(
    tokenizer: TokenizerWrapper,
    text: str,
    max_tokens: int,
) -> str:
    """
    Truncate text to approximately max_tokens.

    Args:
        tokenizer: Tokenizer wrapper.
        text: Text to truncate.
        max_tokens: Maximum token count.

    Returns:
        Truncated text.
    """
    tokens = tokenizer.tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return tokenizer.tokenizer.decode(truncated_tokens)
