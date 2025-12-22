"""
Dataset example generator with token-length enforcement.

This module handles the core generation logic, assembling documents from
filler segments, needles, and distractors while enforcing token length
constraints.
"""

from dataclasses import dataclass
import logging
import random
from typing import Iterator

from .config import DatasetConfig
from .schema import DatasetExample, ExampleMetadata
from .templates import TemplateManager, NeedleTemplate, DistractorTemplate
from .filler import FillerManager, FillerSegment
from .tokenizer_utils import TokenizerWrapper


logger = logging.getLogger(__name__)


@dataclass
class AssembledDocument:
    """Intermediate representation of an assembled document."""

    text: str
    segments: list[str]  # Segment IDs used
    needle: NeedleTemplate | None
    needle_char_span: tuple[int, int] | None
    distractors: list[DistractorTemplate]
    token_count: int
    iterations: int = 0
    trimmed_segments: int = 0
    assembly_attempts: int = 0


class DatasetGenerator:
    """
    Generates dataset examples with token-length enforcement.

    Uses iterative assembly to ensure documents match target token lengths
    within tolerance.
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: TokenizerWrapper,
        templates: TemplateManager,
        filler: FillerManager,
    ):
        """
        Initialize generator.

        Args:
            config: Dataset configuration.
            tokenizer: Tokenizer wrapper for length enforcement.
            templates: Template manager for needles and distractors.
            filler: Filler manager for padding text.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.templates = templates
        self.filler = filler

        self.segment_delimiter = "\n\n"
        self.max_iterations = 50  # Maximum adjustment iterations
        self.max_assembly_attempts = 5  # Maximum resampling attempts
        self.trim_buffer_tokens = 50  # Safety buffer when trimming

    def generate_example(
        self,
        example_id: str,
        split: str,
        length_bucket: int,
        distractor_level: int,
        needle_present: bool,
        allowed_needle_families: list[str] | None,
        allowed_distractor_families: list[str] | None,
        seed: int,
    ) -> DatasetExample:
        """
        Generate a single dataset example.

        Args:
            example_id: Unique identifier for this example.
            split: Data split (train/val/test).
            length_bucket: Target token count.
            distractor_level: Number of distractors to include.
            needle_present: Whether to include a needle.
            allowed_needle_families: Allowed needle family IDs for this split.
            allowed_distractor_families: Allowed distractor family IDs.
            seed: Random seed for this example.

        Returns:
            Generated DatasetExample.

        Raises:
            RuntimeError: If unable to generate example within constraints.
        """
        rng = random.Random(seed)

        # Sample needle if needed
        needle = None
        if needle_present:
            needle = self.templates.sample_needle(
                allowed_families=allowed_needle_families,
                rng=rng,
            )

        # Sample distractors
        distractors = self.templates.sample_distractors(
            n=distractor_level,
            allowed_families=allowed_distractor_families,
            rng=rng,
        )

        # Assemble document with length enforcement
        doc = self._assemble_document(
            length_bucket=length_bucket,
            needle=needle,
            distractors=distractors,
            rng=rng,
        )

        if not self.tokenizer.is_within_tolerance(doc.token_count, length_bucket):
            raise RuntimeError(
                f"Example {example_id} out of tolerance: "
                f"{doc.token_count} tokens for bucket {length_bucket}"
            )

        # Map character span to token indices
        needle_token_indices: list[int] = []
        span_used_fallback = False
        span_warning = None
        if doc.needle_char_span is not None:
            span_result = self.tokenizer.char_span_to_token_indices(
                doc.text,
                doc.needle_char_span[0],
                doc.needle_char_span[1],
            )
            needle_token_indices = span_result.token_indices
            span_used_fallback = span_result.used_fallback
            span_warning = span_result.warning

            if span_result.warning:
                logger.warning(f"Example {example_id}: {span_result.warning}")
            elif span_used_fallback:
                logger.warning(f"Example {example_id}: used fallback span mapping")
            if not needle_token_indices:
                raise RuntimeError(
                    f"Example {example_id} failed to map needle span to tokens"
                )

        if needle_present and doc.needle_char_span is None:
            raise RuntimeError(
                f"Example {example_id} missing needle_char_span for positive example"
            )

        # Compute needle position ratio
        position_ratio = None
        if doc.needle_char_span is not None:
            needle_center = (doc.needle_char_span[0] + doc.needle_char_span[1]) / 2
            position_ratio = needle_center / len(doc.text) if doc.text else 0

        # Build metadata
        meta = ExampleMetadata(
            filler_segment_ids=doc.segments,
            needle_position_ratio=position_ratio,
            generation_params={
                "seed": seed,
                "delimiter": self.segment_delimiter,
                "assembly_attempts": doc.assembly_attempts,
                "assembly_iterations": doc.iterations,
                "trimmed_segments": doc.trimmed_segments,
                "span_mapping_used_fallback": span_used_fallback,
                "span_mapping_warning": span_warning,
            },
        )

        return DatasetExample(
            id=example_id,
            split=split,
            length_bucket=length_bucket,
            length_tokens=doc.token_count,
            distractor_level=distractor_level,
            needle_present=needle_present,
            needle_family=needle.family_id if needle else None,
            needle_text=needle.text if needle else None,
            distractor_families=[d.family_id for d in distractors],
            text=doc.text,
            needle_char_span=doc.needle_char_span,
            needle_token_indices=needle_token_indices,
            seed=seed,
            meta=meta,
        )

    def _assemble_document(
        self,
        length_bucket: int,
        needle: NeedleTemplate | None,
        distractors: list[DistractorTemplate],
        rng: random.Random,
    ) -> AssembledDocument:
        """
        Assemble a document to target length.

        Uses iterative adjustment to hit the target token count within tolerance.

        Args:
            length_bucket: Target token count.
            needle: Needle to embed (or None).
            distractors: Distractors to include.
            rng: Random generator.

        Returns:
            AssembledDocument with text and metadata.
        """
        min_tokens, max_tokens = self.tokenizer.get_tolerance_range(length_bucket)

        # Calculate approximate tokens needed for needle and distractors
        special_tokens = 0
        if needle:
            special_tokens += self.tokenizer.count_tokens(needle.text)
        for d in distractors:
            special_tokens += self.tokenizer.count_tokens(d.text)

        if special_tokens > max_tokens:
            raise RuntimeError(
                f"Special tokens ({special_tokens}) exceed max length ({max_tokens}) "
                f"for bucket {length_bucket}"
            )

        for attempt in range(1, self.max_assembly_attempts + 1):
            trimmed_segments = 0

            # Start with estimated filler segments
            filler_target = length_bucket - special_tokens
            if filler_target <= 0:
                segments: list[FillerSegment] = []
            else:
                segments = self.filler.get_segments_for_tokens(
                    target_tokens=max(100, filler_target),
                    tokens_per_segment_estimate=100,
                    rng=rng,
                )

            # Iteratively adjust to hit target
            for iteration in range(1, self.max_iterations + 1):
                # Build candidate document
                doc = self._build_candidate(segments, needle, distractors, rng)

                if min_tokens <= doc.token_count <= max_tokens:
                    return AssembledDocument(
                        text=doc.text,
                        segments=doc.segments,
                        needle=doc.needle,
                        needle_char_span=doc.needle_char_span,
                        distractors=doc.distractors,
                        token_count=doc.token_count,
                        iterations=iteration,
                        trimmed_segments=trimmed_segments,
                        assembly_attempts=attempt,
                    )

                if doc.token_count < min_tokens:
                    # Too short - add more segments
                    additional = self.filler.sample_segments(
                        n=max(1, (min_tokens - doc.token_count) // 100),
                        rng=rng,
                    )
                    segments.extend(additional)
                    continue

                if doc.token_count > max_tokens:
                    # Too long - remove or trim segments
                    if not segments:
                        break

                    excess_tokens = doc.token_count - max_tokens
                    last_seg_tokens = self.tokenizer.count_tokens(segments[-1].text)

                    if last_seg_tokens <= excess_tokens + self.trim_buffer_tokens:
                        segments.pop()
                    else:
                        target_tokens = max(1, last_seg_tokens - excess_tokens)
                        segments[-1] = self._trim_segment(
                            segments[-1],
                            max_tokens=target_tokens,
                        )
                        trimmed_segments += 1
                        logger.warning(
                            "Trimmed filler segment to reach target length "
                            f"(attempt {attempt}, iteration {iteration})"
                        )

            logger.warning(
                f"Assembly attempt {attempt} failed to reach target length "
                f"for bucket {length_bucket}"
            )

        raise RuntimeError(
            f"Unable to assemble document within tolerance for bucket {length_bucket}"
        )

    def _build_candidate(
        self,
        segments: list[FillerSegment],
        needle: NeedleTemplate | None,
        distractors: list[DistractorTemplate],
        rng: random.Random,
    ) -> AssembledDocument:
        """
        Build a candidate document from components.

        Randomly positions needle and distractors among filler segments.

        Args:
            segments: Filler segments.
            needle: Needle to embed.
            distractors: Distractors to include.
            rng: Random generator.

        Returns:
            AssembledDocument.
        """
        # Create list of all content pieces with type markers
        pieces: list[tuple[str, str, str | None]] = []  # (type, text, id)

        for seg in segments:
            pieces.append(("filler", seg.text, seg.id))

        for d in distractors:
            pieces.append(("distractor", d.text, d.family_id))

        if needle:
            pieces.append(("needle", needle.text, needle.family_id))

        # Shuffle to randomize positions
        rng.shuffle(pieces)

        # Build document text and track needle position
        text_parts = []
        segment_ids = []
        needle_char_span = None
        current_pos = 0

        for piece_type, text, piece_id in pieces:
            if piece_type == "filler":
                segment_ids.append(piece_id)

            if piece_type == "needle":
                needle_char_span = (current_pos, current_pos + len(text))

            text_parts.append(text)
            current_pos += len(text) + len(self.segment_delimiter)

        full_text = self.segment_delimiter.join(text_parts)
        token_count = self.tokenizer.count_tokens(full_text)

        return AssembledDocument(
            text=full_text,
            segments=segment_ids,
            needle=needle,
            needle_char_span=needle_char_span,
            distractors=distractors,
            token_count=token_count,
        )

    def _trim_segment(
        self,
        segment: FillerSegment,
        max_tokens: int,
    ) -> FillerSegment:
        """Trim a segment to approximately max_tokens."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 when trimming a segment")
        tokens = self.tokenizer.tokenizer.encode(segment.text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return segment

        trimmed_tokens = tokens[:max_tokens]
        trimmed_text = self.tokenizer.tokenizer.decode(trimmed_tokens)

        return FillerSegment(
            id=f"{segment.id}_trimmed",
            text=trimmed_text,
            source=segment.source,
        )

    def _force_truncate(
        self,
        doc: AssembledDocument,
        max_tokens: int,
    ) -> AssembledDocument:
        """Force truncate document to max_tokens."""
        # If needle present, ensure it's preserved
        if doc.needle_char_span is not None:
            needle_start, needle_end = doc.needle_char_span
            # Tokenize text before needle to estimate position
            pre_needle_tokens = self.tokenizer.count_tokens(doc.text[:needle_start])
            needle_tokens = self.tokenizer.count_tokens(
                doc.text[needle_start:needle_end]
            )

            if pre_needle_tokens + needle_tokens > max_tokens:
                # Needle too far in - truncate from start
                logger.warning("Needle position too far in document for truncation")
                # Just truncate from end
                tokens = self.tokenizer.tokenizer.encode(
                    doc.text, add_special_tokens=False
                )
                truncated = self.tokenizer.tokenizer.decode(tokens[:max_tokens])
                return AssembledDocument(
                    text=truncated,
                    segments=doc.segments,
                    needle=doc.needle,
                    needle_char_span=None,  # Lost due to truncation
                    distractors=doc.distractors,
                    token_count=max_tokens,
                )

        # Simple truncation from end
        tokens = self.tokenizer.tokenizer.encode(doc.text, add_special_tokens=False)
        truncated = self.tokenizer.tokenizer.decode(tokens[:max_tokens])

        # Recalculate needle span if present
        new_needle_span = None
        if doc.needle_char_span is not None and doc.needle is not None:
            # Check if needle still in truncated text
            if doc.needle.text in truncated:
                start = truncated.find(doc.needle.text)
                new_needle_span = (start, start + len(doc.needle.text))

        return AssembledDocument(
            text=truncated,
            segments=doc.segments,
            needle=doc.needle,
            needle_char_span=new_needle_span,
            distractors=doc.distractors,
            token_count=self.tokenizer.count_tokens(truncated),
        )

    def generate_split(
        self,
        split: str,
        length_buckets: list[int],
        distractor_levels: list[int],
        n_per_condition: int,
        positive_fraction: float,
        allowed_needle_families: list[str],
        allowed_distractor_families: list[str] | None,
        base_seed: int,
    ) -> Iterator[DatasetExample]:
        """
        Generate all examples for a split.

        Args:
            split: Split name (train/val/test).
            length_buckets: Token length buckets.
            distractor_levels: Distractor counts.
            n_per_condition: Examples per (bucket, distractor) condition.
            positive_fraction: Fraction of positive (needle present) examples.
            allowed_needle_families: Allowed needle families for this split.
            allowed_distractor_families: Allowed distractor families.
            base_seed: Base seed for this split.

        Yields:
            Generated DatasetExamples.
        """
        example_counter = 0

        for bucket in length_buckets:
            for distractor_level in distractor_levels:
                n_positive = int(n_per_condition * positive_fraction)
                n_negative = n_per_condition - n_positive

                # Generate positives
                for i in range(n_positive):
                    example_id = (
                        f"{split}_{bucket}_{distractor_level}_{example_counter:06d}"
                    )
                    example_seed = base_seed + example_counter

                    yield self.generate_example(
                        example_id=example_id,
                        split=split,
                        length_bucket=bucket,
                        distractor_level=distractor_level,
                        needle_present=True,
                        allowed_needle_families=allowed_needle_families,
                        allowed_distractor_families=allowed_distractor_families,
                        seed=example_seed,
                    )
                    example_counter += 1

                # Generate negatives
                for i in range(n_negative):
                    example_id = (
                        f"{split}_{bucket}_{distractor_level}_{example_counter:06d}"
                    )
                    example_seed = base_seed + example_counter

                    yield self.generate_example(
                        example_id=example_id,
                        split=split,
                        length_bucket=bucket,
                        distractor_level=distractor_level,
                        needle_present=False,
                        allowed_needle_families=None,  # Not used for negatives
                        allowed_distractor_families=allowed_distractor_families,
                        seed=example_seed,
                    )
                    example_counter += 1
