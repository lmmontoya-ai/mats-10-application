"""
Train/val/test split management with family-based isolation.

This module ensures that needle template families are strictly partitioned
across splits to prevent lexical leakage and ensure valid evaluation.
"""

from dataclasses import dataclass
import hashlib
import logging
from typing import Any

from .config import SplitsConfig
from .templates import TemplateManager


logger = logging.getLogger(__name__)


@dataclass
class SplitAssignment:
    """Assignment of families to splits."""

    train_needle_families: list[str]
    val_needle_families: list[str]
    test_needle_families: list[str]

    train_distractor_families: list[str]
    val_distractor_families: list[str]
    test_distractor_families: list[str]

    def validate(self) -> list[str]:
        """Validate that split assignments have no leakage."""
        errors = []

        # Check needle family disjointness
        train_set = set(self.train_needle_families)
        val_set = set(self.val_needle_families)
        test_set = set(self.test_needle_families)

        train_val_overlap = train_set & val_set
        if train_val_overlap:
            errors.append(
                f"Needle family overlap between train and val: {train_val_overlap}"
            )

        train_test_overlap = train_set & test_set
        if train_test_overlap:
            errors.append(
                f"Needle family overlap between train and test: {train_test_overlap}"
            )

        val_test_overlap = val_set & test_set
        if val_test_overlap:
            errors.append(
                f"Needle family overlap between val and test: {val_test_overlap}"
            )

        # Warn if any split has no needle families (but don't error)
        if not self.train_needle_families:
            logger.warning("No needle families assigned to train split")
        if not self.val_needle_families:
            logger.warning("No needle families assigned to val split")
        if not self.test_needle_families:
            logger.warning("No needle families assigned to test split")

        return errors

    def get_split_info(self) -> dict[str, Any]:
        """Get summary information about split assignments."""
        return {
            "needle_families": {
                "train": self.train_needle_families,
                "val": self.val_needle_families,
                "test": self.test_needle_families,
            },
            "distractor_families": {
                "train": self.train_distractor_families,
                "val": self.val_distractor_families,
                "test": self.test_distractor_families,
            },
            "counts": {
                "train_needle": len(self.train_needle_families),
                "val_needle": len(self.val_needle_families),
                "test_needle": len(self.test_needle_families),
                "train_distractor": len(self.train_distractor_families),
                "val_distractor": len(self.val_distractor_families),
                "test_distractor": len(self.test_distractor_families),
            },
        }


class SplitManager:
    """
    Manages family-based train/val/test splits.

    Ensures strict disjointness of needle families between splits to
    prevent trivial lexical shortcuts during evaluation.
    """

    def __init__(
        self,
        config: SplitsConfig,
        templates: TemplateManager,
        seed: int,
    ):
        """
        Initialize split manager.

        Args:
            config: Splits configuration.
            templates: Template manager with available families.
            seed: Random seed for deterministic assignment.
        """
        self.config = config
        self.templates = templates
        self.seed = seed

        self.assignment = self._compute_assignment()

        # Validate immediately
        errors = self.assignment.validate()
        if errors:
            raise ValueError(
                "Split assignment validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def _compute_assignment(self) -> SplitAssignment:
        """Compute family assignments to splits."""
        all_needle_families = self.templates.get_needle_family_ids()
        all_distractor_families = self.templates.get_distractor_family_ids()

        # Assign needle families
        if self.config.test_needle_families or self.config.val_needle_families:
            # Explicit assignment
            test_needles = [
                f for f in self.config.test_needle_families if f in all_needle_families
            ]
            val_needles = [
                f for f in self.config.val_needle_families if f in all_needle_families
            ]
            unknown_test = set(self.config.test_needle_families) - set(
                all_needle_families
            )
            unknown_val = set(self.config.val_needle_families) - set(
                all_needle_families
            )
            if unknown_test:
                logger.warning(
                    f"Unknown test needle families ignored: {sorted(unknown_test)}"
                )
            if unknown_val:
                logger.warning(
                    f"Unknown val needle families ignored: {sorted(unknown_val)}"
                )
            # Train gets everything else
            train_needles = [
                f
                for f in all_needle_families
                if f not in test_needles and f not in val_needles
            ]
        else:
            # Hash-based partitioning
            test_needles, val_needles, train_needles = self._hash_partition(
                all_needle_families,
                self.config.test_fraction,
                self.config.val_fraction,
            )

        # Assign distractor families
        if self.config.holdout_distractor_families:
            # Also partition distractors
            test_distractors, val_distractors, train_distractors = self._hash_partition(
                all_distractor_families,
                self.config.test_fraction,
                self.config.val_fraction,
            )
        else:
            # All distractors available to all splits
            train_distractors = all_distractor_families
            val_distractors = all_distractor_families
            test_distractors = all_distractor_families

        return SplitAssignment(
            train_needle_families=train_needles,
            val_needle_families=val_needles,
            test_needle_families=test_needles,
            train_distractor_families=train_distractors,
            val_distractor_families=val_distractors,
            test_distractor_families=test_distractors,
        )

    def _hash_partition(
        self,
        families: list[str],
        test_fraction: float,
        val_fraction: float,
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Partition families using deterministic hashing.

        Args:
            families: List of family IDs.
            test_fraction: Fraction for test.
            val_fraction: Fraction for val.

        Returns:
            Tuple of (test_families, val_families, train_families).
        """
        test_families = []
        val_families = []
        train_families = []

        for family_id in sorted(families):  # Sort for determinism
            # Hash family ID with seed
            hash_input = f"{self.seed}:{family_id}".encode()
            hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
            # Normalize to [0, 1)
            normalized = (hash_value % 10000) / 10000.0

            if normalized < test_fraction:
                test_families.append(family_id)
            elif normalized < test_fraction + val_fraction:
                val_families.append(family_id)
            else:
                train_families.append(family_id)

        return test_families, val_families, train_families

    def get_allowed_families(self, split: str) -> tuple[list[str], list[str]]:
        """
        Get allowed needle and distractor families for a split.

        Args:
            split: Split name (train/val/test).

        Returns:
            Tuple of (allowed_needle_families, allowed_distractor_families).

        Raises:
            ValueError: If split name is invalid.
        """
        if split == "train":
            return (
                self.assignment.train_needle_families,
                self.assignment.train_distractor_families,
            )
        elif split == "val":
            return (
                self.assignment.val_needle_families,
                self.assignment.val_distractor_families,
            )
        elif split == "test":
            return (
                self.assignment.test_needle_families,
                self.assignment.test_distractor_families,
            )
        else:
            raise ValueError(f"Invalid split name: {split}")

    def get_split_info(self) -> dict[str, Any]:
        """Get summary information about splits."""
        return self.assignment.get_split_info()

    def check_family_in_split(self, family_id: str, split: str) -> bool:
        """
        Check if a needle family is allowed in a split.

        Args:
            family_id: Needle family ID.
            split: Split name.

        Returns:
            True if family is allowed in split.
        """
        needle_families, _ = self.get_allowed_families(split)
        return family_id in needle_families
