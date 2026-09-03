"""Target-space residual checks used to trigger M-open expansion (Algorithm 8)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .posterior import BeliefState, ModelFit
from .types import Datum


TargetFn = Callable[[np.ndarray, Any], np.ndarray]


def identity_target(value: np.ndarray, action: Any) -> np.ndarray:
    del action
    return np.asarray(value, dtype=float)


def relative_residual(
    observed: np.ndarray,
    predicted: np.ndarray,
    floor: float | np.ndarray,
) -> float:
    observed = np.asarray(observed, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted targets have different shapes")
    denominator = np.maximum(np.abs(observed), np.asarray(floor, dtype=float))
    if np.any(denominator <= 0) or np.any(~np.isfinite(denominator)):
        raise ValueError("predictive-check floors must be finite and positive")
    return float(np.median(np.abs(observed - predicted) / denominator))


def fit_residual(
    fit: ModelFit,
    datum: Datum,
    target: TargetFn,
    floor: float | np.ndarray,
) -> float:
    theta = fit.parameters.map_parameters
    observed = target(datum.observation, datum.action)
    predicted = fit.model.target_prediction(theta, datum.action)
    return relative_residual(observed, predicted, floor)


def predictive_check(
    belief: BeliefState,
    fresh: Datum,
    history: Sequence[Datum],
    *,
    target: TargetFn = identity_target,
    floor: float | np.ndarray = 1e-6,
    backtest: bool = True,
) -> float:
    """Score a fresh datum with the pre-observation MAP fit, then optional history."""
    fit = belief.map_fit
    errors = [fit_residual(fit, fresh, target, floor)]
    if backtest:
        errors.extend(fit_residual(fit, datum, target, floor) for datum in history)
    return float(np.median(errors))


def in_sample_check(
    belief: BeliefState,
    dataset: Sequence[Datum],
    *,
    target: TargetFn = identity_target,
    floor: float | np.ndarray = 1e-6,
) -> float:
    errors = [fit_residual(belief.map_fit, datum, target, floor) for datum in dataset]
    return float(np.median(errors)) if errors else 0.0


def residual_report(
    fit: ModelFit,
    dataset: Sequence[Datum],
    *,
    target: TargetFn = identity_target,
    floor: float | np.ndarray = 1e-6,
) -> dict[str, Any]:
    errors = [fit_residual(fit, datum, target, floor) for datum in dataset]
    return {
        "name": fit.model.name,
        "structure": fit.model.structure,
        "median_relative_residual": float(np.median(errors)) if errors else 0.0,
        "per_experiment": errors,
    }
