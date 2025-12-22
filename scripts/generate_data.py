#!/usr/bin/env python3
"""
Dataset Generation Script for Prompt Injection Monitoring

This script generates a controlled long-context "needle-in-a-haystack" dataset
for training and evaluating prompt injection detectors.

Usage:
    python scripts/generate_data.py [--config CONFIG_PATH]

Example:
    python scripts/generate_data.py --config configs/dataset_main.yaml

The script will:
1. Load and validate configuration
2. Initialize tokenizer and template managers
3. Generate examples for all conditions and splits
4. Write dataset JSONL files and split manifest
5. Run QA validation and produce reports
6. Output summary to console
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.config import load_config, DatasetConfig
from src.dataset.templates import TemplateManager
from src.dataset.filler import FillerManager
from src.dataset.tokenizer_utils import TokenizerWrapper
from src.dataset.generator import DatasetGenerator
from src.dataset.splits import SplitManager
from src.dataset.qa import QARunner
from src.dataset.io import DatasetWriter
from src.dataset.schema import DatasetExample


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate needle-in-a-haystack dataset for prompt injection monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_main.yaml",
        help="Path to configuration YAML file (default: configs/dataset_main.yaml)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def generate_dataset(
    config: DatasetConfig,
) -> tuple[
    list[DatasetExample],
    str,
    TokenizerWrapper,
    TemplateManager,
    FillerManager,
    SplitManager,
]:
    """
    Generate complete dataset from configuration.

    Args:
        config: Validated dataset configuration.

    Returns:
        Tuple of (examples, timestamp, tokenizer, templates, filler, splits).
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Initialize components
    logger.info(
        f"Loading tokenizer: {config.tokenizer.tokenizer_id} "
        f"(model_id={config.tokenizer.model_id})"
    )
    tokenizer = TokenizerWrapper(config.tokenizer)

    logger.info("Initializing template manager")
    templates = TemplateManager(config.templates, config.randomness.template_seed)
    logger.info(
        f"Loaded {len(templates.needle_families)} needle families, "
        f"{len(templates.distractor_families)} distractor families"
    )

    logger.info("Initializing filler manager")
    filler = FillerManager(config.filler, config.randomness.filler_seed)
    logger.info(f"Loaded {len(filler)} filler segments")

    logger.info("Computing split assignments")
    splits = SplitManager(config.splits, templates, config.randomness.dataset_seed)
    split_info = splits.get_split_info()
    logger.info(f"Split assignment: {split_info['counts']}")

    logger.info("Initializing generator")
    generator = DatasetGenerator(config, tokenizer, templates, filler)

    # Generate examples for each split
    all_examples: list[DatasetExample] = []

    for split_name, n_examples in [
        ("train", config.scale.n_train),
        ("val", config.scale.n_val),
        ("test", config.scale.n_test),
    ]:
        logger.info(f"Generating {split_name} split...")

        needle_families, distractor_families = splits.get_allowed_families(split_name)
        logger.info(f"  Allowed needle families: {needle_families}")

        # Compute base seed for this split
        split_seed_offset = {"train": 0, "val": 100000, "test": 200000}
        base_seed = config.randomness.dataset_seed + split_seed_offset[split_name]

        # Generate examples
        examples = list(
            generator.generate_split(
                split=split_name,
                length_buckets=config.grid.length_buckets_tokens,
                distractor_levels=config.grid.distractor_levels,
                n_per_condition=n_examples,
                positive_fraction=config.scale.positive_fraction,
                allowed_needle_families=needle_families,
                allowed_distractor_families=distractor_families,
                base_seed=base_seed,
            )
        )

        logger.info(f"  Generated {len(examples)} examples")
        all_examples.extend(examples)

    return all_examples, timestamp, tokenizer, templates, filler, splits


def run_qa_and_write_outputs(
    examples: list[DatasetExample],
    config: DatasetConfig,
    timestamp: str,
    tokenizer: TokenizerWrapper,
    templates: TemplateManager,
    splits: SplitManager,
) -> tuple["QAReport", dict[str, str]]:
    """
    Run QA validation and write all output files.

    Args:
        examples: Generated examples.
        config: Dataset configuration.
        timestamp: Generation timestamp.
        tokenizer: Tokenizer wrapper used for generation.
        templates: Template manager used for generation.
        splits: Split manager used for generation.

    Returns:
        Tuple of (QAReport, output_paths).
    """
    # Initialize writer
    writer = DatasetWriter(config.output.output_root)

    # Write dataset files
    logger.info("Writing dataset files...")
    dataset_path = writer.write_dataset(
        examples,
        config.output.dataset_name,
        config.output.dataset_version,
    )

    # Also write per-split files
    split_paths = writer.write_split_datasets(
        examples,
        config.output.dataset_name,
        config.output.dataset_version,
    )

    # Write manifest
    manifest_path = writer.write_split_manifest(
        examples,
        config.output.dataset_name,
        config.output.dataset_version,
    )

    # Run QA
    logger.info("Running QA validation...")
    qa_runner = QARunner(
        templates=templates,
        splits=splits,
        tolerance_ratio=config.tokenizer.token_tolerance_ratio,
        tokenizer=tokenizer,
    )

    report = qa_runner.run_qa(
        examples=examples,
        dataset_name=config.output.dataset_name,
        dataset_version=config.output.dataset_version,
        timestamp=timestamp,
    )

    # Write QA reports
    qa_json_path, qa_txt_path = writer.write_qa_report(
        report,
        config.output.dataset_name,
        config.output.dataset_version,
    )

    # Generate and write spot-check file
    spot_check_content = qa_runner.generate_spot_check_file(
        examples,
        n_positive=10,
        n_negative=10,
        seed=config.randomness.dataset_seed,
    )
    spot_check_path = writer.write_spot_check(
        spot_check_content,
        config.output.dataset_name,
        config.output.dataset_version,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal examples generated: {len(examples)}")
    print(f"  Train: {sum(1 for e in examples if e.split == 'train')}")
    print(f"  Val: {sum(1 for e in examples if e.split == 'val')}")
    print(f"  Test: {sum(1 for e in examples if e.split == 'test')}")

    print("\nOutput files:")
    print(f"  Dataset: {dataset_path}")
    for split, path in split_paths.items():
        print(f"  {split.capitalize()} split: {path}")
    print(f"  Manifest: {manifest_path}")
    print(f"  QA Report (JSON): {qa_json_path}")
    print(f"  QA Report (TXT): {qa_txt_path}")
    print(f"  Spot Check: {spot_check_path}")

    print("\nQA Summary:")
    print(
        f"  Family leakage check: {'PASSED' if report.family_leakage_passed else 'FAILED'}"
    )
    print(f"  Length check: {'PASSED' if report.length_check_passed else 'FAILED'}")
    print(
        f"  Span integrity check: {'PASSED' if report.span_check_passed else 'FAILED'}"
    )

    if report.baseline_results:
        print("\nTrivial baseline results:")
        for baseline in report.baseline_results:
            print(
                f"  {baseline.method}: AUROC={baseline.auroc_estimate:.3f}, F1={baseline.f1:.3f}"
            )

    if report.baseline_warning:
        print(f"\n  WARNING: {report.baseline_warning}")

    if report.all_checks_passed:
        print("\n" + "=" * 60)
        print("ALL QA CHECKS PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("QA CHECKS FAILED - See report for details")
        print("=" * 60)
        if report.errors:
            print("\nErrors:")
            for e in report.errors[:5]:
                print(f"  - {e}")
            if len(report.errors) > 5:
                print(f"  ... and {len(report.errors) - 5} more errors")

    print("\n" + "-" * 60)
    print("HOW TO USE THIS DATASET")
    print("-" * 60)
    print(
        """
1. Load the dataset:
   from src.dataset.io import DatasetReader
   examples = DatasetReader.load_dataset('data/datasets/needle_haystack_v1.0.jsonl')

2. Access example fields:
   for ex in examples:
       print(ex.id, ex.needle_present, ex.length_bucket)
       if ex.needle_present:
           print(f"Needle at tokens {ex.needle_token_indices}")

3. Filter by split/condition:
   train_examples = [e for e in examples if e.split == 'train']
   long_examples = [e for e in examples if e.length_bucket >= 4096]

4. For probe training, use needle_token_indices for token-level labels.
"""
    )

    output_paths = {
        "dataset": str(dataset_path),
        "manifest": str(manifest_path),
        "qa_json": str(qa_json_path),
        "qa_txt": str(qa_txt_path),
        "spot_check": str(spot_check_path),
    }
    output_paths.update({f"{k}_split": str(v) for k, v in split_paths.items()})

    return report, output_paths


def _get_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def write_run_record(
    *,
    config: DatasetConfig,
    tokenizer: TokenizerWrapper,
    templates: TemplateManager,
    filler: FillerManager,
    splits: SplitManager,
    output_paths: dict[str, str],
    qa_report: "QAReport",
    qa_summary_paths: dict[str, str],
    run_start: str,
    run_end: str,
    config_path: str,
) -> Path:
    """Write run_record.json for dataset generation."""
    repo_root = Path(__file__).parent.parent
    run_records_dir = repo_root / "results" / "run_records"
    run_records_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "stage": "dataset_generation",
        "dataset_name": config.output.dataset_name,
        "dataset_version": config.output.dataset_version,
        "timestamps": {"start": run_start, "end": run_end},
        "model_id": config.tokenizer.model_id,
        "tokenizer": tokenizer.get_tokenizer_info(),
        "seeds": {
            "dataset_seed": config.randomness.dataset_seed,
            "template_seed": config.randomness.template_seed,
            "filler_seed": config.randomness.filler_seed,
        },
        "grid": {
            "length_buckets_tokens": config.grid.length_buckets_tokens,
            "distractor_levels": config.grid.distractor_levels,
        },
        "scale": {
            "n_train": config.scale.n_train,
            "n_val": config.scale.n_val,
            "n_test": config.scale.n_test,
            "positive_fraction": config.scale.positive_fraction,
        },
        "splits": config.splits.__dict__,
        "template_families": {
            "needle": templates.get_needle_family_ids(),
            "distractor": templates.get_distractor_family_ids(),
        },
        "split_assignment": splits.get_split_info(),
        "filler": filler.get_filler_metadata(),
        "outputs": output_paths,
        "qa": {
            "all_checks_passed": qa_report.all_checks_passed,
            "errors": qa_report.errors,
            "warnings": qa_report.warnings,
            "baseline_warning": qa_report.baseline_warning,
        },
        "qa_summary_paths": qa_summary_paths,
        "hyperparameters_searched": {},
        "selected_hyperparameters": {},
        "config_path": config_path,
        "git_commit": _get_git_commit(repo_root),
    }

    path = (
        run_records_dir
        / f"{config.output.dataset_name}_{config.output.dataset_version}_run_record.json"
    )
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"Wrote run record to {path}")
    return path


def main() -> int:
    """Main entry point."""
    args = parse_args()

    run_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1

    logger.info(
        f"Configuration loaded: {config.output.dataset_name} v{config.output.dataset_version}"
    )

    # Generate dataset
    try:
        examples, timestamp, tokenizer, templates, filler, splits = generate_dataset(
            config
        )
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}", exc_info=True)
        return 1

    # Run QA and write outputs
    try:
        report, output_paths = run_qa_and_write_outputs(
            examples,
            config,
            timestamp,
            tokenizer,
            templates,
            splits,
        )
    except Exception as e:
        logger.error(f"QA or output writing failed: {e}", exc_info=True)
        return 1

    run_end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    qa_passed = report.all_checks_passed

    write_run_record(
        config=config,
        tokenizer=tokenizer,
        templates=templates,
        filler=filler,
        splits=splits,
        output_paths=output_paths,
        qa_report=report,
        qa_summary_paths={
            "qa_json_path": output_paths["qa_json"],
            "qa_txt_path": output_paths["qa_txt"],
        },
        run_start=run_start,
        run_end=run_end,
        config_path=args.config,
    )

    if not qa_passed:
        raise RuntimeError("QA checks failed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
