"""Utilities for watermark-based user attribution."""

from .metrics import (
    bitwise_accuracy,
    false_detection_rate,
    match_watermarks,
    maximum_pairwise_accuracy,
    true_detection_and_attribution_rates,
)
from .selection import (
    generate_absta,
    generate_nrg,
    generate_random,
    select_watermarks,
)
from .theory import (
    fdr_upper_bound_independent,
    fdr_upper_bound_random,
    paper_table_5,
    tar_lower_bound,
    tdr_lower_bound,
)

__all__ = [
    "bitwise_accuracy",
    "false_detection_rate",
    "fdr_upper_bound_independent",
    "fdr_upper_bound_random",
    "generate_absta",
    "generate_nrg",
    "generate_random",
    "match_watermarks",
    "maximum_pairwise_accuracy",
    "paper_table_5",
    "select_watermarks",
    "tar_lower_bound",
    "tdr_lower_bound",
    "true_detection_and_attribution_rates",
]
