"""
Dataset I/O utilities for JSONL and manifest files.

This module handles reading and writing dataset files in a consistent,
reproducible format.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Iterator

from .schema import DatasetExample, SplitManifestEntry
from .qa import QAReport


logger = logging.getLogger(__name__)


class DatasetWriter:
    """
    Writes dataset files in standardized formats.

    Handles JSONL dataset files, CSV split manifests, and QA reports.
    """

    def __init__(self, output_root: str):
        """
        Initialize dataset writer.

        Args:
            output_root: Root directory for output files.
        """
        self.output_root = Path(output_root)
        self.datasets_dir = self.output_root / "datasets"
        self.splits_dir = self.output_root / "splits"
        self.qa_dir = self.output_root / "qa"

        # Create directories
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.qa_dir.mkdir(parents=True, exist_ok=True)

    def write_dataset(
        self,
        examples: list[DatasetExample],
        dataset_name: str,
        dataset_version: str,
    ) -> Path:
        """
        Write dataset examples to JSONL file.

        Args:
            examples: List of examples to write.
            dataset_name: Name of the dataset.
            dataset_version: Version string.

        Returns:
            Path to written file.
        """
        filename = f"{dataset_name}_{dataset_version}.jsonl"
        filepath = self.datasets_dir / filename

        with open(filepath, "w") as f:
            for example in examples:
                json_line = json.dumps(example.to_dict(), ensure_ascii=False)
                f.write(json_line + "\n")

        logger.info(f"Wrote {len(examples)} examples to {filepath}")
        return filepath

    def write_split_datasets(
        self,
        examples: list[DatasetExample],
        dataset_name: str,
        dataset_version: str,
    ) -> dict[str, Path]:
        """
        Write separate JSONL files per split.

        Args:
            examples: List of all examples.
            dataset_name: Name of the dataset.
            dataset_version: Version string.

        Returns:
            Dictionary mapping split names to file paths.
        """
        split_examples: dict[str, list[DatasetExample]] = {}
        for ex in examples:
            if ex.split not in split_examples:
                split_examples[ex.split] = []
            split_examples[ex.split].append(ex)

        paths = {}
        for split, split_exs in split_examples.items():
            filename = f"{dataset_name}_{dataset_version}_{split}.jsonl"
            filepath = self.datasets_dir / filename

            with open(filepath, "w") as f:
                for example in split_exs:
                    json_line = json.dumps(example.to_dict(), ensure_ascii=False)
                    f.write(json_line + "\n")

            logger.info(f"Wrote {len(split_exs)} {split} examples to {filepath}")
            paths[split] = filepath

        return paths

    def write_split_manifest(
        self,
        examples: list[DatasetExample],
        dataset_name: str,
        dataset_version: str,
    ) -> Path:
        """
        Write split manifest CSV.

        Args:
            examples: List of all examples.
            dataset_name: Name of the dataset.
            dataset_version: Version string.

        Returns:
            Path to written manifest file.
        """
        filename = f"{dataset_name}_{dataset_version}_manifest.csv"
        filepath = self.splits_dir / filename

        fieldnames = [
            "id",
            "split",
            "length_bucket",
            "distractor_level",
            "needle_present",
            "needle_family",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for example in examples:
                entry = SplitManifestEntry.from_example(example)
                writer.writerow(entry.to_dict())

        logger.info(f"Wrote manifest with {len(examples)} entries to {filepath}")
        return filepath

    def write_qa_report(
        self,
        report: QAReport,
        dataset_name: str,
        dataset_version: str,
    ) -> tuple[Path, Path]:
        """
        Write QA report in JSON and text formats.

        Args:
            report: QAReport to write.
            dataset_name: Name of the dataset.
            dataset_version: Version string.

        Returns:
            Tuple of (JSON path, text path).
        """
        # JSON report
        json_filename = f"{dataset_name}_{dataset_version}_qa.json"
        json_path = self.qa_dir / json_filename

        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Wrote QA JSON report to {json_path}")

        # Text summary
        txt_filename = f"{dataset_name}_{dataset_version}_qa.txt"
        txt_path = self.qa_dir / txt_filename

        with open(txt_path, "w") as f:
            f.write(report.to_summary_text())

        logger.info(f"Wrote QA text summary to {txt_path}")

        return json_path, txt_path

    def write_spot_check(
        self,
        content: str,
        dataset_name: str,
        dataset_version: str,
    ) -> Path:
        """
        Write spot-check file for manual inspection.

        Args:
            content: Spot-check file content.
            dataset_name: Name of the dataset.
            dataset_version: Version string.

        Returns:
            Path to written file.
        """
        filename = f"{dataset_name}_{dataset_version}_spot_check.txt"
        filepath = self.qa_dir / filename

        with open(filepath, "w") as f:
            f.write(content)

        logger.info(f"Wrote spot-check file to {filepath}")
        return filepath


class DatasetReader:
    """
    Reads dataset files.

    Provides utilities for loading JSONL datasets and manifests.
    """

    @staticmethod
    def read_jsonl(filepath: str | Path) -> Iterator[DatasetExample]:
        """
        Read examples from JSONL file.

        Args:
            filepath: Path to JSONL file.

        Yields:
            DatasetExample instances.
        """
        filepath = Path(filepath)

        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield DatasetExample.from_dict(data)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue

    @staticmethod
    def read_manifest(filepath: str | Path) -> list[SplitManifestEntry]:
        """
        Read split manifest from CSV.

        Args:
            filepath: Path to manifest CSV.

        Returns:
            List of SplitManifestEntry instances.
        """
        filepath = Path(filepath)
        entries = []

        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = SplitManifestEntry(
                    id=row["id"],
                    split=row["split"],
                    length_bucket=int(row["length_bucket"]),
                    distractor_level=int(row["distractor_level"]),
                    needle_present=row["needle_present"].lower() == "true",
                    needle_family=(
                        row["needle_family"] if row["needle_family"] else None
                    ),
                )
                entries.append(entry)

        return entries

    @staticmethod
    def load_dataset(filepath: str | Path) -> list[DatasetExample]:
        """
        Load all examples from JSONL file into memory.

        Args:
            filepath: Path to JSONL file.

        Returns:
            List of DatasetExample instances.
        """
        return list(DatasetReader.read_jsonl(filepath))
