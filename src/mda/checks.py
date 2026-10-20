"""Target-space residual checks used to trigger M-open expansion (``predictive-check``)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .posterior import BeliefState, ModelFit
from .types import CandidateModel, Datum, NonFinitePrediction, PredictFn


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


def observed_target_prediction(model: CandidateModel) -> PredictFn:
    """The prediction compared against ``target(observation)`` in residual checks.

    By default this is ``model.target_prediction``. A model whose evaluation
    target cannot be recovered from its observations (the stochastic neuron
    features, which count spikes the ~4 ms observation grid misses) supplies
    ``metadata["observed_target_prediction"]``: the same functional evaluated
    on the observation grid, so checks and residual reports compare like with
    like while held-out evaluation keeps the full-resolution target.
    """
    callback = model.metadata.get("observed_target_prediction")
    if callback is None:
        return model.target_prediction

    def predict(theta, action):
        value = np.asarray(callback(np.asarray(theta, dtype=float), action), dtype=float)
        if np.any(~np.isfinite(value)):
            raise NonFinitePrediction(f"model {model.name!r} returned a non-finite target prediction")
        return value

    return predict


def fit_residual(
    fit: ModelFit,
    datum: Datum,
    target: TargetFn,
    floor: float | np.ndarray,
) -> float:
    theta = fit.parameters.map_parameters
    observed = target(datum.observation, datum.action)
    predicted = observed_target_prediction(fit.model)(theta, datum.action)
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
    errors = []
    signed_errors = []
    theta = fit.parameters.map_parameters
    for datum in dataset:
        observed = np.asarray(target(datum.observation, datum.action), dtype=float).reshape(-1)
        predicted = np.asarray(
            observed_target_prediction(fit.model)(theta, datum.action), dtype=float
        ).reshape(-1)
        if observed.shape != predicted.shape:
            raise ValueError("observed and predicted targets have different shapes")
        denominator = np.maximum(np.abs(observed), np.asarray(floor, dtype=float))
        if np.any(denominator <= 0) or np.any(~np.isfinite(denominator)):
            raise ValueError("predictive-check floors must be finite and positive")
        relative = (observed - predicted) / denominator
        errors.append(float(np.median(np.abs(relative))))
        signed_errors.append(float(np.median(relative)))
    return {
        "name": fit.model.name,
        "structure": fit.model.structure,
        "median_relative_residual": float(np.median(errors)) if errors else 0.0,
        "per_experiment": errors,
        "signed_per_experiment": signed_errors,
    }
