"""Theoretical bounds from *Watermark-based Attribution of AI-Generated Content*."""

from __future__ import annotations

import math

from scipy.stats import binom


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _ceil_count(value: float) -> int:
    """Ceil a count while tolerating floating-point noise at integer boundaries."""
    return math.ceil(value - 1e-12)


def _floor_count(value: float) -> int:
    """Floor a count while tolerating floating-point noise at integer boundaries."""
    return math.floor(value + 1e-12)


def tdr_lower_bound(
    n: int,
    beta: float,
    tau: float,
    alpha_min: float,
) -> float:
    """Return the user-specific TDR lower bound from Theorem 1.

    ``alpha_min`` is the minimum pairwise bitwise accuracy between the user's
    watermark and any other watermark.
    """
    n = _positive_int("n", n)
    beta = _probability("beta", beta)
    tau = _probability("tau", tau)
    alpha_min = _probability("alpha_min", alpha_min)
    if not 0.5 < tau < beta:
        raise ValueError("Theorem 1 assumes 0.5 < tau < beta")

    distribution = binom(n, beta)
    detection_threshold = _ceil_count(tau * n)
    direct_detection = distribution.sf(detection_threshold - 1)

    second_cutoff = _floor_count(n * (1.0 - tau - alpha_min))
    indirect_detection = distribution.cdf(second_cutoff)
    return float(min(1.0, direct_detection + indirect_detection))


def fdr_upper_bound_random(
    n: int,
    tau: float,
    alpha_max: float,
) -> float:
    """Return the FDR upper bound from Theorem 2.

    This theorem assumes the reference watermark is selected uniformly at
    random. ``alpha_max`` is its maximum pairwise bitwise accuracy with the
    other watermarks.
    """
    n = _positive_int("n", n)
    tau = _probability("tau", tau)
    alpha_max = _probability("alpha_max", alpha_max)
    if tau <= 0.5:
        raise ValueError("The detection threshold tau must be greater than 0.5")

    distribution = binom(n, 0.5)
    detection_threshold = _ceil_count(tau * n)
    upper_tail = distribution.sf(detection_threshold - 1)
    lower_cutoff = _floor_count(n * (1.0 - tau + alpha_max))
    lower_tail = distribution.cdf(lower_cutoff)
    return float(min(1.0, upper_tail + lower_tail))


def fdr_upper_bound_independent(
    s: int,
    n: int,
    tau: float,
    gamma: float,
) -> float:
    """Return the FDR upper bound from Theorem 3.

    The bound assumes ``s`` independently selected watermarks and a
    gamma-random decoder on non-AI-generated content.
    """
    s = _positive_int("s", s)
    n = _positive_int("n", n)
    tau = _probability("tau", tau)
    gamma = _probability("gamma", gamma)
    if tau <= 0.5:
        raise ValueError("The detection threshold tau must be greater than 0.5")
    if gamma > 0.5:
        raise ValueError("gamma must be in [0, 0.5]")

    detection_threshold = _ceil_count(tau * n)
    below_threshold = float(
        binom.cdf(detection_threshold - 1, n, 0.5 + gamma)
    )
    if below_threshold <= 0.0:
        return 1.0
    if below_threshold >= 1.0:
        return 0.0

    # 1 - p**s, evaluated stably when p is close to one.
    return float(-math.expm1(s * math.log(below_threshold)))


def tar_lower_bound(
    n: int,
    beta: float,
    tau: float,
    alpha_max: float,
) -> float:
    """Return the user-specific TAR lower bound from Theorem 4.

    ``alpha_max`` is the maximum pairwise bitwise accuracy between the user's
    watermark and any other watermark.
    """
    n = _positive_int("n", n)
    beta = _probability("beta", beta)
    tau = _probability("tau", tau)
    alpha_max = _probability("alpha_max", alpha_max)
    if tau <= 0.5:
        raise ValueError("The detection threshold tau must be greater than 0.5")

    attribution_threshold = _floor_count((1.0 + alpha_max) * n / 2.0) + 1
    detection_threshold = _ceil_count(tau * n)
    threshold = max(attribution_threshold, detection_threshold)
    return float(binom.sf(threshold - 1, n, beta))


def paper_table_5() -> dict[str, float]:
    """Compute the three bounds reported in Table 5 of the paper."""
    return {
        "TDR lower bound": tdr_lower_bound(
            n=64,
            beta=0.99,
            tau=0.9,
            alpha_min=0.2,
        ),
        "FDR upper bound": fdr_upper_bound_independent(
            s=100_000_000,
            n=64,
            tau=0.9,
            gamma=0.05,
        ),
        "TAR lower bound": tar_lower_bound(
            n=64,
            beta=0.99,
            tau=0.9,
            alpha_max=0.8,
        ),
    }
