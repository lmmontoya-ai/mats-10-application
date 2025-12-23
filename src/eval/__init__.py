"""
Evaluation utilities for document-level monitoring.
"""

from .aggregators import AggregatorOutput, build_aggregators
from .calibration import IdentityCalibrator, PlattCalibrator
from .metrics import (
    compute_auprc,
    compute_ece,
    compute_fpr_tpr,
    compute_metrics,
    threshold_at_tpr,
)

__all__ = [
    "AggregatorOutput",
    "build_aggregators",
    "IdentityCalibrator",
    "PlattCalibrator",
    "compute_auprc",
    "compute_ece",
    "compute_fpr_tpr",
    "compute_metrics",
    "threshold_at_tpr",
]
