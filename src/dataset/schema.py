"""
Dataset schema definitions.

This module defines the schema for dataset examples and split manifests,
ensuring consistent structure across the pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ExampleMetadata:
    """Metadata for a dataset example."""

    filler_segment_ids: list[str] = field(default_factory=list)
    needle_position_ratio: float | None = None  # Position of needle as fraction of doc
    generation_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetExample:
    """
    A single dataset example for prompt injection monitoring.

    This schema captures all information needed for downstream training
    and evaluation, including document-level and token-level labels.
    """

    # Identification
    id: str
    split: str  # train/val/test
    length_bucket: int  # Target token count
    length_tokens: int  # Actual token count
    distractor_level: int  # Number of distractors

    # Labels
    needle_present: bool
    needle_family: str | None  # Family ID if needle present
    distractor_families: list[str]  # List of distractor family IDs
    needle_text: str | None  # Exact needle text if present

    # Content
    text: str  # Full document text

    # Span labels (sparse)
    needle_char_span: tuple[int, int] | None  # (start, end) character indices
    needle_token_indices: list[int]  # Token indices overlapping needle

    # Reproducibility
    seed: int  # Seed used for this example

    # Additional metadata
    meta: ExampleMetadata

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert tuple to list for JSON
        if d["needle_char_span"] is not None:
            d["needle_char_span"] = list(d["needle_char_span"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetExample":
        """Create from dictionary."""
        # Convert list back to tuple for needle_char_span
        if d.get("needle_char_span") is not None:
            d["needle_char_span"] = tuple(d["needle_char_span"])
        if "needle_text" not in d:
            d["needle_text"] = None
        # Convert meta dict to ExampleMetadata
        if isinstance(d.get("meta"), dict):
            d["meta"] = ExampleMetadata(**d["meta"])
        return cls(**d)

    def validate(self) -> list[str]:
        """
        Validate the example and return list of errors.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Check split
        if self.split not in ("train", "val", "test"):
            errors.append(f"Invalid split: {self.split}")

        # Check length consistency
        if self.length_tokens <= 0:
            errors.append(f"Invalid length_tokens: {self.length_tokens}")

        # Check label consistency
        if self.needle_present:
            if self.needle_family is None:
                errors.append("needle_present=True but needle_family is None")
            if self.needle_text is None:
                errors.append("needle_present=True but needle_text is None")
            if self.needle_char_span is None:
                errors.append("needle_present=True but needle_char_span is None")
            if not self.needle_token_indices:
                errors.append("needle_present=True but needle_token_indices is empty")
            # Validate span bounds
            if self.needle_char_span is not None:
                start, end = self.needle_char_span
                if start < 0 or end < start or end > len(self.text):
                    errors.append(
                        f"Invalid needle_char_span: ({start}, {end}) for text of length {len(self.text)}"
                    )
                elif self.needle_text is not None:
                    span_text = self.text[start:end]
                    if span_text != self.needle_text:
                        errors.append("needle_text does not match needle_char_span")
            # Validate token indices
            if self.needle_token_indices:
                if min(self.needle_token_indices) < 0:
                    errors.append("needle_token_indices contains negative indices")
                if max(self.needle_token_indices) >= self.length_tokens:
                    errors.append("needle_token_indices out of bounds")
        else:
            if self.needle_family is not None:
                errors.append("needle_present=False but needle_family is not None")
            if self.needle_text is not None:
                errors.append("needle_present=False but needle_text is not None")
            if self.needle_char_span is not None:
                errors.append("needle_present=False but needle_char_span is not None")
            if self.needle_token_indices:
                errors.append(
                    "needle_present=False but needle_token_indices is not empty"
                )

        # Check distractor level consistency
        if len(self.distractor_families) != self.distractor_level:
            errors.append(
                f"distractor_level={self.distractor_level} but "
                f"{len(self.distractor_families)} distractor families listed"
            )

        return errors


@dataclass
class SplitManifestEntry:
    """Entry in the split manifest CSV."""

    id: str
    split: str
    length_bucket: int
    distractor_level: int
    needle_present: bool
    needle_family: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for CSV writing."""
        return {
            "id": self.id,
            "split": self.split,
            "length_bucket": self.length_bucket,
            "distractor_level": self.distractor_level,
            "needle_present": self.needle_present,
            "needle_family": self.needle_family if self.needle_family else "",
        }

    @classmethod
    def from_example(cls, example: DatasetExample) -> "SplitManifestEntry":
        """Create manifest entry from dataset example."""
        return cls(
            id=example.id,
            split=example.split,
            length_bucket=example.length_bucket,
            distractor_level=example.distractor_level,
            needle_present=example.needle_present,
            needle_family=example.needle_family,
        )
