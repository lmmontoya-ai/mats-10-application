"""
Configuration parsing and validation for dataset generation.

This module defines the configuration schema and provides loading/validation
functions to ensure reproducible dataset generation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer settings."""

    model_id: str
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    max_length_bucket: int = 16384
    token_tolerance_ratio: float = 0.02  # +/- 2% tolerance

    def __post_init__(self) -> None:
        if self.tokenizer_id is None:
            self.tokenizer_id = self.model_id


@dataclass
class DatasetScaleConfig:
    """Configuration for dataset scale per condition."""

    n_train: int = 100
    n_val: int = 50
    n_test: int = 50
    positive_fraction: float = 0.5


@dataclass
class ExperimentGridConfig:
    """Configuration for experimental conditions."""

    length_buckets_tokens: list[int] = field(
        default_factory=lambda: [2048, 8192, 16384]
    )
    distractor_levels: list[int] = field(default_factory=lambda: [0, 1, 4, 8])


@dataclass
class RandomnessConfig:
    """Configuration for random seeds."""

    dataset_seed: int = 42
    template_seed: int | None = None  # Derived from dataset_seed if None
    filler_seed: int | None = None  # Derived from dataset_seed if None


@dataclass
class TemplateFamily:
    """A family of related templates."""

    family_id: str
    description: str
    variants: list[str]


@dataclass
class TemplatesConfig:
    """Configuration for needle and distractor templates."""

    needle_families: list[TemplateFamily] = field(default_factory=list)
    distractor_families: list[TemplateFamily] = field(default_factory=list)
    templates_path: str | None = None  # Optional path to external templates file


@dataclass
class SplitsConfig:
    """Configuration for train/val/test splits."""

    test_needle_families: list[str] = field(default_factory=list)
    val_needle_families: list[str] = field(default_factory=list)
    # If families not specified, use hash-based partitioning
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    holdout_distractor_families: bool = False  # Whether to holdout distractors too


@dataclass
class OutputConfig:
    """Configuration for output paths."""

    dataset_name: str = "needle_haystack"
    dataset_version: str = "v1.0"
    output_root: str = "data"


@dataclass
class FillerConfig:
    """Configuration for filler text sourcing."""

    filler_path: str = "data_raw/filler_segments.jsonl"
    allow_llm_fallback: bool = False
    llm_fallback_prompt: str | None = None
    min_segment_tokens: int = 50
    max_segment_tokens: int = 500


@dataclass
class ProbeConfig:
    """Configuration for token-level probe training."""

    layers_to_probe: list[int] = field(default_factory=lambda: [16])
    activation_site: str = "resid_post"
    negatives_per_doc: int = 128
    train_length_buckets: list[int] | None = None
    val_length_buckets: list[int] | None = None
    max_train_docs: int | None = None
    max_val_docs: int | None = None
    max_train_tokens: int | None = None
    max_val_tokens: int | None = None
    batch_size: int = 256
    num_epochs: int = 3
    learning_rate: float = 1e-3
    l2_grid: list[float] = field(default_factory=lambda: [0.0, 1e-4, 1e-3, 1e-2])
    class_balance: str = "weighted"  # weighted | downsample
    train_seeds: list[int] = field(default_factory=lambda: [0])
    device: str = "auto"  # auto | cpu | cuda | mps
    model_dtype: str = "auto"  # auto | float32 | float16 | bfloat16
    feature_dtype: str = "float32"  # float32 | float16
    save_features: bool = False
    output_dir: str = "results/probes"

    def __post_init__(self) -> None:
        def _coerce_int(value: Any, name: str) -> int | None:
            if value is None:
                return None
            if isinstance(value, bool):
                raise ValueError(f"{name} must be an integer, not bool")
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            return int(str(value))

        def _coerce_float(value: Any, name: str) -> float:
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a float, not bool")
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value))

        if self.layers_to_probe is not None:
            self.layers_to_probe = [
                _coerce_int(v, "probe.layers_to_probe") for v in self.layers_to_probe
            ]
        self.negatives_per_doc = _coerce_int(
            self.negatives_per_doc, "probe.negatives_per_doc"
        )
        self.batch_size = _coerce_int(self.batch_size, "probe.batch_size")
        self.num_epochs = _coerce_int(self.num_epochs, "probe.num_epochs")
        self.learning_rate = _coerce_float(self.learning_rate, "probe.learning_rate")
        if self.l2_grid is not None:
            self.l2_grid = [_coerce_float(v, "probe.l2_grid") for v in self.l2_grid]
        if self.train_seeds is not None:
            self.train_seeds = [
                _coerce_int(v, "probe.train_seeds") for v in self.train_seeds
            ]
        self.max_train_docs = _coerce_int(self.max_train_docs, "probe.max_train_docs")
        self.max_val_docs = _coerce_int(self.max_val_docs, "probe.max_val_docs")
        self.max_train_tokens = _coerce_int(
            self.max_train_tokens, "probe.max_train_tokens"
        )
        self.max_val_tokens = _coerce_int(self.max_val_tokens, "probe.max_val_tokens")


@dataclass
class DatasetConfig:
    """
    Complete configuration for dataset generation.

    All parameters required for reproducible dataset generation are captured here.
    """

    tokenizer: TokenizerConfig
    scale: DatasetScaleConfig
    grid: ExperimentGridConfig
    randomness: RandomnessConfig
    templates: TemplatesConfig
    splits: SplitsConfig
    output: OutputConfig
    filler: FillerConfig
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    def __post_init__(self) -> None:
        """Derive dependent seeds if not specified."""
        if self.randomness.template_seed is None:
            self.randomness.template_seed = self.randomness.dataset_seed + 1000
        if self.randomness.filler_seed is None:
            self.randomness.filler_seed = self.randomness.dataset_seed + 2000

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Validate tokenizer config
        if not self.tokenizer.model_id:
            errors.append("tokenizer.model_id must be specified")
        if not self.tokenizer.tokenizer_id:
            errors.append("tokenizer.tokenizer_id must be specified")
        if (
            self.tokenizer.token_tolerance_ratio <= 0
            or self.tokenizer.token_tolerance_ratio > 0.5
        ):
            errors.append("tokenizer.token_tolerance_ratio must be in (0, 0.5]")

        # Validate scale config
        if self.scale.n_train <= 0 or self.scale.n_val <= 0 or self.scale.n_test <= 0:
            errors.append("scale.n_train/n_val/n_test must all be positive")
        if not 0 < self.scale.positive_fraction < 1:
            errors.append("scale.positive_fraction must be in (0, 1)")

        # Validate grid config
        if not self.grid.length_buckets_tokens:
            errors.append("grid.length_buckets_tokens must not be empty")
        if any(b <= 0 for b in self.grid.length_buckets_tokens):
            errors.append("All length buckets must be positive")
        if self.grid.length_buckets_tokens:
            max_bucket = max(self.grid.length_buckets_tokens)
            if max_bucket > self.tokenizer.max_length_bucket:
                errors.append(
                    "grid.length_buckets_tokens exceeds tokenizer.max_length_bucket"
                )
        if not self.grid.distractor_levels:
            errors.append("grid.distractor_levels must not be empty")
        if any(d < 0 for d in self.grid.distractor_levels):
            errors.append("All distractor levels must be non-negative")

        # Validate splits config
        if self.splits.test_fraction < 0 or self.splits.test_fraction > 0.5:
            errors.append("splits.test_fraction must be in [0, 0.5]")
        if self.splits.val_fraction < 0 or self.splits.val_fraction > 0.5:
            errors.append("splits.val_fraction must be in [0, 0.5]")
        if self.splits.test_fraction + self.splits.val_fraction >= 1:
            errors.append("splits.test_fraction + val_fraction must be < 1")

        # Validate output config
        if not self.output.dataset_name:
            errors.append("output.dataset_name must be specified")
        if not self.output.dataset_version:
            errors.append("output.dataset_version must be specified")

        return errors


def _parse_template_family(data: dict[str, Any]) -> TemplateFamily:
    """Parse a template family from dictionary."""
    return TemplateFamily(
        family_id=data["family_id"],
        description=data.get("description", ""),
        variants=data.get("variants", []),
    )


def _parse_config_section(
    data: dict[str, Any], section: str, dataclass_type: type
) -> Any:
    """Parse a config section into its dataclass type."""
    section_data = data.get(section, {})
    if dataclass_type == TemplatesConfig:
        # Special handling for templates with nested families
        needle_families = [
            _parse_template_family(f) for f in section_data.get("needle_families", [])
        ]
        distractor_families = [
            _parse_template_family(f)
            for f in section_data.get("distractor_families", [])
        ]
        return TemplatesConfig(
            needle_families=needle_families,
            distractor_families=distractor_families,
            templates_path=section_data.get("templates_path"),
        )
    return dataclass_type(**section_data)


def load_config(config_path: str | Path) -> DatasetConfig:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Validated DatasetConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If configuration is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Parse each section
    tokenizer = _parse_config_section(data, "tokenizer", TokenizerConfig)
    scale = _parse_config_section(data, "scale", DatasetScaleConfig)
    grid = _parse_config_section(data, "grid", ExperimentGridConfig)
    randomness = _parse_config_section(data, "randomness", RandomnessConfig)
    templates = _parse_config_section(data, "templates", TemplatesConfig)
    splits = _parse_config_section(data, "splits", SplitsConfig)
    output = _parse_config_section(data, "output", OutputConfig)
    filler = _parse_config_section(data, "filler", FillerConfig)
    probe = _parse_config_section(data, "probe", ProbeConfig)

    config = DatasetConfig(
        tokenizer=tokenizer,
        scale=scale,
        grid=grid,
        randomness=randomness,
        templates=templates,
        splits=splits,
        output=output,
        filler=filler,
        probe=probe,
    )

    # Validate
    errors = config.validate()
    if errors:
        raise ValueError(
            f"Configuration validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return config
