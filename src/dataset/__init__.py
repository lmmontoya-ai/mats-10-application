"""
Dataset generation pipeline for prompt injection monitoring.

This module provides a configuration-driven, deterministic pipeline for generating
long-context "needle-in-a-haystack" datasets for prompt injection detection.
"""

from .config import DatasetConfig, load_config
from .schema import DatasetExample, SplitManifestEntry
from .templates import TemplateManager, NeedleTemplate, DistractorTemplate
from .filler import FillerManager
from .tokenizer_utils import TokenizerWrapper
from .generator import DatasetGenerator
from .splits import SplitManager
from .qa import QARunner
from .io import DatasetWriter

__all__ = [
    "DatasetConfig",
    "load_config",
    "DatasetExample",
    "SplitManifestEntry",
    "TemplateManager",
    "NeedleTemplate",
    "DistractorTemplate",
    "FillerManager",
    "TokenizerWrapper",
    "DatasetGenerator",
    "SplitManager",
    "QARunner",
    "DatasetWriter",
]
