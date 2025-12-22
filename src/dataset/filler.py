"""
Filler text sourcing and management.

This module handles loading and sampling filler text segments used to
pad documents to target token lengths.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import random

from .config import FillerConfig


@dataclass
class FillerSegment:
    """A filler text segment."""

    id: str
    text: str
    source: str = "unknown"
    estimated_tokens: int | None = None


class FillerManager:
    """
    Manages filler text segments for document assembly.

    Loads segments from a JSONL file and provides deterministic sampling.
    """

    def __init__(
        self,
        config: FillerConfig,
        seed: int,
    ):
        """
        Initialize filler manager.

        Args:
            config: Filler configuration.
            seed: Random seed for deterministic sampling.

        Raises:
            FileNotFoundError: If filler file doesn't exist and LLM fallback is disabled.
        """
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)

        self.segments: list[FillerSegment] = []
        self.used_fallback = False
        self.fallback_info: dict[str, str | None] = {}
        self._load_segments()

    def _load_segments(self) -> None:
        """Load filler segments from file."""
        filler_path = Path(self.config.filler_path)

        if not filler_path.exists():
            if self.config.allow_llm_fallback:
                self._generate_fallback_segments()
            else:
                raise FileNotFoundError(
                    f"Filler segments file not found: {filler_path}\n\n"
                    f"To create this file, you have several options:\n"
                    f"1. Create {filler_path} manually with JSONL format:\n"
                    f'   {{"id": "seg_001", "text": "Your paragraph text...", "source": "wikipedia"}}\n'
                    f'   {{"id": "seg_002", "text": "Another paragraph...", "source": "wikipedia"}}\n\n'
                    f"2. Enable LLM fallback in config (not recommended for reproducibility):\n"
                    f"   filler:\n"
                    f"     allow_llm_fallback: true\n\n"
                    f"The filler file should contain diverse paragraphs of general text\n"
                    f"(e.g., from Wikipedia, news articles, or technical documentation)."
                )

        with open(filler_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    segment = FillerSegment(
                        id=data.get("id", f"seg_{line_num:06d}"),
                        text=data["text"],
                        source=data.get("source", "unknown"),
                        estimated_tokens=data.get("estimated_tokens"),
                    )
                    self.segments.append(segment)
                except (json.JSONDecodeError, KeyError) as e:
                    raise ValueError(
                        f"Error parsing filler segment at line {line_num}: {e}"
                    )

        if not self.segments:
            raise ValueError(f"No filler segments found in {filler_path}")

    def _generate_fallback_segments(self) -> None:
        """Generate fallback filler segments (for testing only)."""
        self.used_fallback = True
        self.fallback_info = {
            "type": "synthetic_fallback",
            "prompt": self.config.llm_fallback_prompt,
        }
        # This creates deterministic synthetic paragraphs
        # In production, this should be replaced with real text
        topics = [
            "machine learning",
            "software engineering",
            "data science",
            "natural language processing",
            "computer vision",
            "robotics",
            "cloud computing",
            "cybersecurity",
            "database systems",
            "web development",
            "mobile applications",
            "algorithms",
        ]

        templates = [
            "The field of {topic} has seen remarkable advances in recent years. "
            "Researchers have developed new techniques that improve performance "
            "across a wide range of applications. These developments have important "
            "implications for both industry and academia.",
            "{topic} is an area of active research with many open problems. "
            "Current approaches typically involve sophisticated mathematical "
            "frameworks and extensive computational resources. Despite these "
            "challenges, progress continues at a rapid pace.",
            "Understanding {topic} requires familiarity with several foundational "
            "concepts. Students often begin by studying the theoretical background "
            "before moving on to practical applications. This progression helps "
            "build intuition for more advanced topics.",
            "Industry adoption of {topic} has accelerated significantly. "
            "Companies are investing heavily in infrastructure and talent to "
            "leverage these capabilities. The economic impact is projected to "
            "grow substantially over the coming decade.",
            "The history of {topic} spans several decades of innovation. "
            "Early pioneers laid the groundwork for techniques that are now "
            "standard practice. Modern approaches build on this foundation while "
            "introducing novel improvements.",
        ]

        rng = random.Random(self.seed + 9999)  # Separate seed for fallback

        for i, topic in enumerate(topics):
            for j, template in enumerate(templates):
                segment_id = f"fallback_{i:03d}_{j:03d}"
                text = template.format(topic=topic)
                self.segments.append(
                    FillerSegment(
                        id=segment_id,
                        text=text,
                        source="synthetic_fallback",
                    )
                )

        # Shuffle deterministically
        rng.shuffle(self.segments)

    def get_filler_metadata(self) -> dict[str, str | bool | None]:
        """Return metadata about the filler source for run records."""
        return {
            "filler_path": str(self.config.filler_path),
            "used_fallback": self.used_fallback,
            "fallback_type": self.fallback_info.get("type"),
            "fallback_prompt": self.fallback_info.get("prompt"),
        }

    def sample_segments(
        self,
        n: int,
        rng: random.Random | None = None,
    ) -> list[FillerSegment]:
        """
        Sample filler segments.

        Args:
            n: Number of segments to sample.
            rng: Random generator to use (None = use internal).

        Returns:
            List of sampled FillerSegments.
        """
        rng = rng or self.rng

        if n <= 0:
            return []

        if n >= len(self.segments):
            # Sample with replacement if needed
            return [rng.choice(self.segments) for _ in range(n)]

        # Sample without replacement for smaller n
        return rng.sample(self.segments, n)

    def get_segments_for_tokens(
        self,
        target_tokens: int,
        tokens_per_segment_estimate: int = 100,
        rng: random.Random | None = None,
    ) -> list[FillerSegment]:
        """
        Get enough filler segments to approximately reach target token count.

        Args:
            target_tokens: Target number of tokens.
            tokens_per_segment_estimate: Estimated tokens per segment.
            rng: Random generator.

        Returns:
            List of filler segments.
        """
        n_segments = max(1, target_tokens // tokens_per_segment_estimate)
        # Add extra buffer for adjustment
        n_segments = int(n_segments * 1.2)
        return self.sample_segments(n_segments, rng)

    def get_segment_by_id(self, segment_id: str) -> FillerSegment | None:
        """Get a specific segment by ID."""
        for segment in self.segments:
            if segment.id == segment_id:
                return segment
        return None

    def __len__(self) -> int:
        """Return number of available segments."""
        return len(self.segments)
