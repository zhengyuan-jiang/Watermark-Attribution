"""Metrics for watermark-based detection and user attribution."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


BitArray = NDArray[np.uint8]


def _bits(name: str, values: ArrayLike, dimensions: tuple[int, ...]) -> BitArray:
    array = np.asarray(values)
    if array.ndim not in dimensions:
        expected = " or ".join(str(value) for value in dimensions)
        raise ValueError(f"{name} must have {expected} dimensions")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only 0 and 1")
    return array.astype(np.uint8, copy=False)


def _pool(watermarks: ArrayLike) -> BitArray:
    pool = _bits("watermarks", watermarks, (2,))
    if pool.shape[0] == 0 or pool.shape[1] == 0:
        raise ValueError("watermarks must be a non-empty matrix")
    return pool


def bitwise_accuracy(left: ArrayLike, right: ArrayLike) -> float:
    """Return the fraction of equal bits in two equally shaped arrays."""
    left_bits = _bits("left", left, (1, 2))
    right_bits = _bits("right", right, (left_bits.ndim,))
    if left_bits.shape != right_bits.shape:
        raise ValueError("left and right must have the same shape")
    if left_bits.size == 0:
        raise ValueError("left and right must not be empty")
    return float(np.mean(left_bits == right_bits))


def maximum_pairwise_accuracy(watermarks: ArrayLike) -> float:
    """Return the largest bitwise accuracy between distinct watermarks."""
    pool = _pool(watermarks)
    if pool.shape[0] < 2:
        raise ValueError("at least two watermarks are required")

    maximum = 0.0
    for index in range(pool.shape[0] - 1):
        accuracies = np.mean(pool[index + 1 :] == pool[index], axis=1)
        maximum = max(maximum, float(np.max(accuracies)))
    return maximum


def match_watermarks(
    decoded_watermarks: ArrayLike,
    watermark_pool: ArrayLike,
    tau: float,
) -> tuple[NDArray[np.bool_], NDArray[np.int64], NDArray[np.float64]]:
    """Detect and attribute decoded watermarks.

    Returns three one-dimensional arrays: whether each watermark is detected,
    the index of its closest user watermark, and the corresponding bitwise
    accuracy.
    """
    decoded = _bits("decoded_watermarks", decoded_watermarks, (1, 2))
    if decoded.ndim == 1:
        decoded = decoded.reshape(1, -1)
    pool = _pool(watermark_pool)
    if decoded.shape[1] != pool.shape[1]:
        raise ValueError("decoded watermarks and pool must have the same length")
    tau = float(tau)
    if not 0.5 < tau <= 1.0:
        raise ValueError("tau must be in (0.5, 1]")

    predicted = np.empty(decoded.shape[0], dtype=np.int64)
    scores = np.empty(decoded.shape[0], dtype=np.float64)
    for index, watermark in enumerate(decoded):
        accuracies = np.mean(pool == watermark, axis=1)
        predicted[index] = int(np.argmax(accuracies))
        scores[index] = float(accuracies[predicted[index]])
    return scores >= tau, predicted, scores


def true_detection_and_attribution_rates(
    decoded_watermarks: ArrayLike,
    target_users: ArrayLike,
    watermark_pool: ArrayLike,
    tau: float,
) -> tuple[float, float]:
    """Return empirical true detection and true attribution rates."""
    detected, predicted, _ = match_watermarks(
        decoded_watermarks,
        watermark_pool,
        tau,
    )
    targets = np.asarray(target_users)
    if targets.ndim != 1 or targets.shape[0] != detected.shape[0]:
        raise ValueError("target_users must contain one index per decoded watermark")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("target_users must contain integer indices")
    pool_size = np.asarray(watermark_pool).shape[0]
    if np.any((targets < 0) | (targets >= pool_size)):
        raise ValueError("target_users contains an index outside the watermark pool")

    correct = predicted == targets
    return float(np.mean(detected)), float(np.mean(detected & correct))


def false_detection_rate(
    decoded_watermarks: ArrayLike,
    watermark_pool: ArrayLike,
    tau: float,
) -> float:
    """Return the fraction of decoded non-AI watermarks that are detected."""
    detected, _, _ = match_watermarks(decoded_watermarks, watermark_pool, tau)
    return float(np.mean(detected))
