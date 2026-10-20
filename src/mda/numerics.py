"""Small numerical primitives shared by the inference layers."""

from __future__ import annotations

import numpy as np


def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return -np.inf
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    normalizer = logsumexp(log_weights)
    if not np.isfinite(normalizer):
        raise ValueError("all particle weights are zero or non-finite")
    return np.exp(np.asarray(log_weights, dtype=float) - normalizer), normalizer


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        return 0.0
    normalized = weights / total
    return 1.0 / float(np.square(normalized).sum())


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return unbiased systematic-resampling ancestor indices."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.minimum(np.searchsorted(cumulative, positions, side="right"), len(weights) - 1)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return np.tensordot(weights, values, axes=(0, 0))


def weighted_covariance(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Population cross-covariance for flattened vector observations."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    left_centered = left - weighted_mean(left, weights)
    right_centered = right - weighted_mean(right, weights)
    return (left_centered * weights[:, None]).T @ right_centered
