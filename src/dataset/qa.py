"""
Quality assurance and validation suite for generated datasets.

This module performs comprehensive validation checks and produces QA reports
to ensure dataset integrity and detect potential issues.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import random
import re
from typing import Any

from .schema import DatasetExample
from .splits import SplitManager
from .templates import TemplateManager
from .tokenizer_utils import TokenizerWrapper


logger = logging.getLogger(__name__)


@dataclass
class LengthStats:
    """Statistics for token lengths."""

    bucket: int
    count: int
    min_tokens: int
    max_tokens: int
    mean_tokens: float
    within_tolerance: int
    outside_tolerance: int


@dataclass
class BaselineResult:
    """Result of trivial baseline evaluation."""

    method: str
    description: str
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc_estimate: float  # Approximate AUROC


@dataclass
class QAReport:
    """Complete QA report for a generated dataset."""

    # Metadata
    dataset_name: str
    dataset_version: str
    total_examples: int
    timestamp: str

    # Split statistics
    split_counts: dict[str, int]
    examples_per_condition: dict[str, int]

    # Family leakage checks
    family_leakage_passed: bool
    family_assignment: dict[str, Any]

    # Length checks
    length_stats: list[LengthStats]
    length_check_passed: bool
    length_violations: list[str]

    # Span integrity checks
    span_check_passed: bool
    span_violations: list[str]

    # Trivial baseline results
    baseline_results: list[BaselineResult]
    baseline_warning: str | None

    # Overall status
    all_checks_passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "total_examples": self.total_examples,
            "timestamp": self.timestamp,
            "split_counts": self.split_counts,
            "examples_per_condition": self.examples_per_condition,
            "family_leakage_passed": self.family_leakage_passed,
            "family_assignment": self.family_assignment,
            "length_stats": [
                {
                    "bucket": s.bucket,
                    "count": s.count,
                    "min_tokens": s.min_tokens,
                    "max_tokens": s.max_tokens,
                    "mean_tokens": s.mean_tokens,
                    "within_tolerance": s.within_tolerance,
                    "outside_tolerance": s.outside_tolerance,
                }
                for s in self.length_stats
            ],
            "length_check_passed": self.length_check_passed,
            "length_violations": self.length_violations,
            "span_check_passed": self.span_check_passed,
            "span_violations": self.span_violations,
            "baseline_results": [
                {
                    "method": b.method,
                    "description": b.description,
                    "true_positives": b.true_positives,
                    "false_positives": b.false_positives,
                    "true_negatives": b.true_negatives,
                    "false_negatives": b.false_negatives,
                    "accuracy": b.accuracy,
                    "precision": b.precision,
                    "recall": b.recall,
                    "f1": b.f1,
                    "auroc_estimate": b.auroc_estimate,
                }
                for b in self.baseline_results
            ],
            "baseline_warning": self.baseline_warning,
            "all_checks_passed": self.all_checks_passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def to_summary_text(self) -> str:
        """Generate human-readable summary text."""
        lines = [
            "=" * 60,
            "DATASET QA REPORT",
            "=" * 60,
            f"Dataset: {self.dataset_name} v{self.dataset_version}",
            f"Total examples: {self.total_examples}",
            f"Generated: {self.timestamp}",
            "",
            "SPLIT DISTRIBUTION",
            "-" * 40,
        ]

        for split, count in sorted(self.split_counts.items()):
            lines.append(f"  {split}: {count}")

        lines.extend(
            [
                "",
                "FAMILY LEAKAGE CHECK",
                "-" * 40,
                f"  Status: {'PASSED' if self.family_leakage_passed else 'FAILED'}",
            ]
        )

        if self.family_assignment:
            for split, families in self.family_assignment.get(
                "needle_families", {}
            ).items():
                lines.append(f"  {split} needle families: {families}")

        lines.extend(
            [
                "",
                "LENGTH VALIDATION",
                "-" * 40,
                f"  Status: {'PASSED' if self.length_check_passed else 'FAILED'}",
            ]
        )

        for stat in self.length_stats:
            lines.append(
                f"  Bucket {stat.bucket}: {stat.count} examples, "
                f"range [{stat.min_tokens}-{stat.max_tokens}], "
                f"mean {stat.mean_tokens:.1f}, "
                f"{stat.outside_tolerance} outside tolerance"
            )

        if self.length_violations:
            lines.append(f"  Violations: {len(self.length_violations)}")
            for v in self.length_violations[:5]:
                lines.append(f"    - {v}")
            if len(self.length_violations) > 5:
                lines.append(f"    ... and {len(self.length_violations) - 5} more")

        lines.extend(
            [
                "",
                "SPAN INTEGRITY CHECK",
                "-" * 40,
                f"  Status: {'PASSED' if self.span_check_passed else 'FAILED'}",
            ]
        )

        if self.span_violations:
            lines.append(f"  Violations: {len(self.span_violations)}")
            for v in self.span_violations[:5]:
                lines.append(f"    - {v}")

        lines.extend(
            [
                "",
                "TRIVIAL BASELINE EVALUATION",
                "-" * 40,
            ]
        )

        for baseline in self.baseline_results:
            lines.extend(
                [
                    f"  {baseline.method}:",
                    f"    Accuracy: {baseline.accuracy:.3f}",
                    f"    Precision: {baseline.precision:.3f}",
                    f"    Recall: {baseline.recall:.3f}",
                    f"    F1: {baseline.f1:.3f}",
                    f"    AUROC (est): {baseline.auroc_estimate:.3f}",
                ]
            )

        if self.baseline_warning:
            lines.extend(
                [
                    "",
                    "  WARNING: " + self.baseline_warning,
                ]
            )

        lines.extend(
            [
                "",
                "=" * 60,
                f"OVERALL STATUS: {'ALL CHECKS PASSED' if self.all_checks_passed else 'CHECKS FAILED'}",
                "=" * 60,
            ]
        )

        if self.errors:
            lines.append("\nERRORS:")
            for e in self.errors:
                lines.append(f"  - {e}")

        if self.warnings:
            lines.append("\nWARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)


class QARunner:
    """
    Runs QA checks on generated datasets.

    Performs family leakage validation, length checks, span integrity
    verification, and trivial baseline evaluation.
    """

    def __init__(
        self,
        templates: TemplateManager,
        splits: SplitManager,
        tolerance_ratio: float = 0.02,
        tokenizer: TokenizerWrapper | None = None,
    ):
        """
        Initialize QA runner.

        Args:
            templates: Template manager.
            splits: Split manager with family assignments.
            tolerance_ratio: Token length tolerance ratio.
        """
        self.templates = templates
        self.splits = splits
        self.tolerance_ratio = tolerance_ratio
        self.tokenizer = tokenizer

        # Keywords for baseline detection
        self.injection_keywords = self.templates.get_needle_keywords()

    def run_qa(
        self,
        examples: list[DatasetExample],
        dataset_name: str,
        dataset_version: str,
        timestamp: str,
    ) -> QAReport:
        """
        Run complete QA suite on generated examples.

        Args:
            examples: List of generated examples.
            dataset_name: Name of dataset.
            dataset_version: Version string.
            timestamp: Generation timestamp.

        Returns:
            Complete QAReport.
        """
        errors = []
        warnings = []

        # Family leakage check
        family_check_passed, family_errors = self._check_family_leakage(examples)
        errors.extend(family_errors)

        # Length checks
        length_stats, length_passed, length_violations = self._check_lengths(examples)
        if length_violations:
            warnings.extend(length_violations[:10])

        # Span integrity
        span_passed, span_violations = self._check_span_integrity(examples)
        errors.extend(span_violations[:10])

        if self.tokenizer is not None and not self.tokenizer.supports_offsets:
            warnings.append(
                "Tokenizer does not support offset mapping; span labels use fallback mapping."
            )

        fallback_count = sum(
            1
            for ex in examples
            if ex.meta.generation_params.get("span_mapping_used_fallback")
        )
        if fallback_count:
            warnings.append(f"Span mapping fallback used in {fallback_count} examples.")

        # Trivial baselines
        baseline_results, baseline_warning = self._run_trivial_baselines(examples)
        if baseline_warning:
            warnings.append(baseline_warning)

        # Compute statistics
        split_counts = defaultdict(int)
        condition_counts = defaultdict(int)
        for ex in examples:
            split_counts[ex.split] += 1
            key = f"{ex.split}_{ex.length_bucket}_{ex.distractor_level}"
            condition_counts[key] += 1

        all_passed = family_check_passed and length_passed and span_passed

        return QAReport(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            total_examples=len(examples),
            timestamp=timestamp,
            split_counts=dict(split_counts),
            examples_per_condition=dict(condition_counts),
            family_leakage_passed=family_check_passed,
            family_assignment=self.splits.get_split_info(),
            length_stats=length_stats,
            length_check_passed=length_passed,
            length_violations=length_violations,
            span_check_passed=span_passed,
            span_violations=span_violations,
            baseline_results=baseline_results,
            baseline_warning=baseline_warning,
            all_checks_passed=all_passed,
            errors=errors,
            warnings=warnings,
        )

    def _check_family_leakage(
        self,
        examples: list[DatasetExample],
    ) -> tuple[bool, list[str]]:
        """Check for needle family leakage across splits."""
        errors = []

        # Get assigned families per split
        train_families = set(self.splits.assignment.train_needle_families)
        val_families = set(self.splits.assignment.val_needle_families)
        test_families = set(self.splits.assignment.test_needle_families)

        # Check each example
        for ex in examples:
            if not ex.needle_present or not ex.needle_family:
                continue

            if ex.split == "train":
                if ex.needle_family not in train_families:
                    errors.append(
                        f"Example {ex.id}: needle family '{ex.needle_family}' "
                        f"not allowed in train split"
                    )
            elif ex.split == "val":
                if ex.needle_family not in val_families:
                    errors.append(
                        f"Example {ex.id}: needle family '{ex.needle_family}' "
                        f"not allowed in val split"
                    )
            elif ex.split == "test":
                if ex.needle_family not in test_families:
                    errors.append(
                        f"Example {ex.id}: needle family '{ex.needle_family}' "
                        f"not allowed in test split"
                    )

        passed = len(errors) == 0
        return passed, errors

    def _check_lengths(
        self,
        examples: list[DatasetExample],
    ) -> tuple[list[LengthStats], bool, list[str]]:
        """Check token length distribution and tolerance."""
        violations = []
        bucket_data: dict[int, list[int]] = defaultdict(list)

        for ex in examples:
            bucket_data[ex.length_bucket].append(ex.length_tokens)

            # Check tolerance
            tolerance = int(ex.length_bucket * self.tolerance_ratio)
            if abs(ex.length_tokens - ex.length_bucket) > tolerance:
                violations.append(
                    f"Example {ex.id}: {ex.length_tokens} tokens, "
                    f"target {ex.length_bucket} (tolerance ±{tolerance})"
                )

        stats = []
        for bucket, lengths in sorted(bucket_data.items()):
            tolerance = int(bucket * self.tolerance_ratio)
            within = sum(1 for l in lengths if abs(l - bucket) <= tolerance)
            stats.append(
                LengthStats(
                    bucket=bucket,
                    count=len(lengths),
                    min_tokens=min(lengths),
                    max_tokens=max(lengths),
                    mean_tokens=sum(lengths) / len(lengths),
                    within_tolerance=within,
                    outside_tolerance=len(lengths) - within,
                )
            )

        passed = len(violations) == 0
        return stats, passed, violations

    def _check_span_integrity(
        self,
        examples: list[DatasetExample],
    ) -> tuple[bool, list[str]]:
        """Check span label integrity."""
        violations = []

        for ex in examples:
            ex_errors = ex.validate()
            if ex_errors:
                for e in ex_errors:
                    violations.append(f"Example {ex.id}: {e}")

            # Additional checks
            if ex.needle_present and ex.needle_char_span:
                start, end = ex.needle_char_span
                needle_text = ex.text[start:end]

                # Check that needle text looks like an injection
                has_keyword = any(
                    kw in needle_text.lower() for kw in self.injection_keywords
                )
                if not has_keyword:
                    # This might indicate span misalignment
                    logger.debug(
                        f"Example {ex.id}: needle span may be misaligned "
                        f"(no keywords found in span text)"
                    )

            if ex.needle_token_indices:
                if max(ex.needle_token_indices) >= ex.length_tokens:
                    violations.append(
                        f"Example {ex.id}: needle_token_indices out of bounds"
                    )

            if (
                self.tokenizer is not None
                and ex.needle_char_span
                and ex.needle_token_indices
            ):
                start, end = ex.needle_char_span
                span_result = self.tokenizer.char_span_to_token_indices(
                    ex.text,
                    start,
                    end,
                )
                expected = set(span_result.token_indices)
                actual = set(ex.needle_token_indices)
                if not (expected & actual):
                    violations.append(
                        f"Example {ex.id}: needle_token_indices do not overlap char span"
                    )

        passed = len(violations) == 0
        return passed, violations

    def _run_trivial_baselines(
        self,
        examples: list[DatasetExample],
    ) -> tuple[list[BaselineResult], str | None]:
        """Run trivial baseline classifiers."""
        results = []
        warning = None

        # Keyword baseline
        kw_result = self._keyword_baseline(examples)
        results.append(kw_result)

        # Regex pattern baseline
        regex_result = self._regex_baseline(examples)
        results.append(regex_result)

        # Check if any baseline is too good
        max_auroc = max(r.auroc_estimate for r in results)
        if max_auroc > 0.95:
            warning = (
                f"Trivial baseline achieves AUROC {max_auroc:.3f}. "
                f"Dataset may be too easy - consider revising templates "
                f"to reduce lexical overlap."
            )
        elif max_auroc > 0.85:
            warning = (
                f"Trivial baseline achieves AUROC {max_auroc:.3f}. "
                f"Some lexical patterns may be exploitable."
            )

        return results, warning

    def _keyword_baseline(
        self,
        examples: list[DatasetExample],
    ) -> BaselineResult:
        """Simple keyword-matching baseline."""
        tp = fp = tn = fn = 0

        for ex in examples:
            text_lower = ex.text.lower()
            predicted_positive = any(kw in text_lower for kw in self.injection_keywords)

            if ex.needle_present:
                if predicted_positive:
                    tp += 1
                else:
                    fn += 1
            else:
                if predicted_positive:
                    fp += 1
                else:
                    tn += 1

        return self._compute_baseline_metrics(
            "keyword_matching",
            "Predicts positive if any injection keyword present",
            tp,
            fp,
            tn,
            fn,
        )

    def _regex_baseline(
        self,
        examples: list[DatasetExample],
    ) -> BaselineResult:
        """Regex pattern baseline."""
        patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions?",
            r"disregard\s+(everything|all)",
            r"override\s+(active|protocol)",
            r"bypass\s+(enabled|safety|filters)",
            r"jailbreak",
            r"developer\s+mode\s+(enabled|activated)",
            r"dan\s+mode",
            r"\[system\]",
            r"<<system",
            r"admin_mode",
            r"root\s+access",
        ]

        combined_pattern = "|".join(f"({p})" for p in patterns)
        regex = re.compile(combined_pattern, re.IGNORECASE)

        tp = fp = tn = fn = 0

        for ex in examples:
            predicted_positive = bool(regex.search(ex.text))

            if ex.needle_present:
                if predicted_positive:
                    tp += 1
                else:
                    fn += 1
            else:
                if predicted_positive:
                    fp += 1
                else:
                    tn += 1

        return self._compute_baseline_metrics(
            "regex_patterns",
            "Predicts positive if injection-like regex patterns match",
            tp,
            fp,
            tn,
            fn,
        )

    def _compute_baseline_metrics(
        self,
        method: str,
        description: str,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
    ) -> BaselineResult:
        """Compute metrics for a baseline."""
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Approximate AUROC using TPR and FPR
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        # Simple approximation: area under single point + triangles
        auroc_estimate = 0.5 * (1 + tpr - fpr)

        return BaselineResult(
            method=method,
            description=description,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auroc_estimate=auroc_estimate,
        )

    def generate_spot_check_file(
        self,
        examples: list[DatasetExample],
        n_positive: int = 10,
        n_negative: int = 10,
        seed: int = 42,
    ) -> str:
        """
        Generate spot-check file content for manual inspection.

        Args:
            examples: List of examples.
            n_positive: Number of positive examples to include.
            n_negative: Number of negative examples to include.
            seed: Random seed for sampling.

        Returns:
            Formatted text content for spot-check file.
        """
        rng = random.Random(seed)

        positives = [ex for ex in examples if ex.needle_present]
        negatives = [ex for ex in examples if not ex.needle_present]

        sampled_positives = rng.sample(positives, min(n_positive, len(positives)))
        sampled_negatives = rng.sample(negatives, min(n_negative, len(negatives)))

        lines = [
            "=" * 80,
            "DATASET SPOT-CHECK FILE",
            "=" * 80,
            "",
            "This file contains sample examples for manual inspection.",
            "Positive examples (needle present) show excerpt around the needle.",
            "Negative examples show a random window.",
            "",
        ]

        # Positive examples
        lines.extend(
            [
                "=" * 80,
                "POSITIVE EXAMPLES (needle present)",
                "=" * 80,
                "",
            ]
        )

        for i, ex in enumerate(sampled_positives, 1):
            lines.extend(self._format_example_for_spot_check(ex, i, "POSITIVE", rng))

        # Negative examples
        lines.extend(
            [
                "",
                "=" * 80,
                "NEGATIVE EXAMPLES (no needle)",
                "=" * 80,
                "",
            ]
        )

        for i, ex in enumerate(sampled_negatives, 1):
            lines.extend(self._format_example_for_spot_check(ex, i, "NEGATIVE", rng))

        return "\n".join(lines)

    def _format_example_for_spot_check(
        self,
        ex: DatasetExample,
        index: int,
        label: str,
        rng: random.Random,
    ) -> list[str]:
        """Format a single example for the spot-check file."""
        lines = [
            "-" * 60,
            f"Example {index} ({label})",
            "-" * 60,
            f"ID: {ex.id}",
            f"Split: {ex.split}",
            f"Length bucket: {ex.length_bucket}",
            f"Actual tokens: {ex.length_tokens}",
            f"Distractor level: {ex.distractor_level}",
            f"Distractor families: {ex.distractor_families}",
        ]

        if ex.needle_present:
            lines.extend(
                [
                    f"Needle family: {ex.needle_family}",
                    f"Needle char span: {ex.needle_char_span}",
                    f"Needle token indices: {ex.needle_token_indices[:20]}{'...' if len(ex.needle_token_indices) > 20 else ''}",
                ]
            )

        lines.append("")
        lines.append("EXCERPT:")
        lines.append("-" * 40)

        # Show excerpt around needle (or random window for negatives)
        if ex.needle_present and ex.needle_char_span:
            start, end = ex.needle_char_span
            context_before = 200
            context_after = 200
            excerpt_start = max(0, start - context_before)
            excerpt_end = min(len(ex.text), end + context_after)

            excerpt = ex.text[excerpt_start:excerpt_end]
            # Mark the needle position
            needle_start_in_excerpt = start - excerpt_start
            needle_end_in_excerpt = end - excerpt_start

            lines.append(f"... {excerpt[:needle_start_in_excerpt]}")
            lines.append(">>> NEEDLE START >>>")
            lines.append(excerpt[needle_start_in_excerpt:needle_end_in_excerpt])
            lines.append("<<< NEEDLE END <<<")
            lines.append(f"{excerpt[needle_end_in_excerpt:]} ...")
        else:
            # Random window for negatives
            window_size = 500
            max_start = max(0, len(ex.text) - window_size)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            lines.append(f"... {ex.text[start:start + window_size]} ...")

        lines.append("")
        return lines
