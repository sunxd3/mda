"""Gaussian model-EIG, joint-EIG, and task-VoI surrogates (Algorithm 9)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .numerics import weighted_covariance, weighted_mean
from .posterior import BeliefState


def _flat(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _covariance_matrix(covariance: float | np.ndarray, dimension: int) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 0:
        matrix = np.eye(dimension) * float(covariance)
    elif covariance.ndim == 1:
        if covariance.shape != (dimension,):
            raise ValueError("diagonal covariance has the wrong dimension")
        matrix = np.diag(covariance)
    elif covariance.shape == (dimension, dimension):
        matrix = covariance
    else:
        raise ValueError("covariance must be scalar, diagonal, or square")
    matrix = (matrix + matrix.T) / 2
    if np.any(~np.isfinite(matrix)) or np.min(np.linalg.eigvalsh(matrix)) <= 0:
        raise ValueError("covariance must be finite and positive definite")
    return matrix


def _quadratic(value: np.ndarray, covariance: np.ndarray) -> float:
    return float(value @ np.linalg.solve(covariance, value))


def model_eig(
    action: Any,
    belief: BeliefState,
    observation_covariance: float | np.ndarray,
) -> float:
    """Paper Eq. 34: between-structure predictive disagreement.

    This is the trace surrogate used by the experiments, not the log-determinant
    moment-matched MI of Eq. 31. The two are equivalent only for scalar outcomes.
    """
    model_means = []
    for fit in belief.fits:
        predictions = np.stack(
            [_flat(fit.model.observation_prediction(theta, action)) for theta in fit.parameters.particles]
        )
        model_means.append(weighted_mean(predictions, fit.parameters.weights))
    means = np.stack(model_means)
    overall = weighted_mean(means, belief.model_weights)
    covariance = _covariance_matrix(observation_covariance, len(overall))
    return sum(
        float(weight) * _quadratic(mean - overall, covariance)
        for weight, mean in zip(belief.model_weights, means, strict=True)
    )


def joint_eig(
    action: Any,
    belief: BeliefState,
    observation_covariance: float | np.ndarray,
) -> float:
    """Paper Eq. 38: model plus within-structure parameter disagreement."""
    predictions = []
    weights = []
    for model, theta, weight in belief.iter_parameter_particles():
        predictions.append(_flat(model.observation_prediction(theta, action)))
        weights.append(weight)
    values = np.stack(predictions)
    weights_array = np.asarray(weights)
    mean = weighted_mean(values, weights_array)
    covariance = _covariance_matrix(observation_covariance, len(mean))
    return sum(
        float(weight) * _quadratic(value - mean, covariance)
        for weight, value in zip(weights_array, values, strict=True)
    )


def task_value_of_information(
    action: Any,
    query_actions: Sequence[Any],
    belief: BeliefState,
    observation_covariance: float | np.ndarray,
    target_covariance: float | np.ndarray | None = None,
) -> float:
    """Squared-loss task VoI (paper Eq. 43), generalized to vector targets.

    For each query this computes the linear-Gaussian expected reduction in target
    variance, tr(C_qy V_y^-1 C_yq). A supplied target covariance whitens target
    coordinates, matching the paper's component-standardized NeuronBench score.
    """
    particles = list(belief.iter_parameter_particles())
    weights = np.asarray([weight for _, _, weight in particles], dtype=float)
    observed = np.stack(
        [_flat(model.observation_prediction(theta, action)) for model, theta, _ in particles]
    )
    observed_covariance = weighted_covariance(observed, observed, weights)
    noise = _covariance_matrix(observation_covariance, observed.shape[1])
    predictive_covariance = observed_covariance + noise

    values = []
    for query in query_actions:
        targets = np.stack(
            [_flat(model.target_prediction(theta, query)) for model, theta, _ in particles]
        )
        cross = weighted_covariance(targets, observed, weights)
        reduction = cross @ np.linalg.solve(predictive_covariance, cross.T)
        if target_covariance is None:
            values.append(float(np.trace(reduction)))
        else:
            scale = _covariance_matrix(target_covariance, reduction.shape[0])
            values.append(float(np.trace(np.linalg.solve(scale, reduction))))
    return float(np.mean(values)) if values else 0.0


def choose_design(
    actions: Sequence[Any],
    belief: BeliefState,
    *,
    flavour: str,
    observation_covariance: float | np.ndarray,
    query_actions: Sequence[Any] = (),
    target_covariance: float | np.ndarray | None = None,
) -> tuple[Any, np.ndarray]:
    if not actions:
        raise ValueError("the design space is empty")
    scorers = {
        "model": lambda action: model_eig(action, belief, observation_covariance),
        "joint": lambda action: joint_eig(action, belief, observation_covariance),
        "task": lambda action: task_value_of_information(
            action,
            query_actions,
            belief,
            observation_covariance,
            target_covariance,
        ),
    }
    if flavour not in scorers:
        raise ValueError(f"unknown design flavour {flavour!r}")
    scores = np.asarray([scorers[flavour](action) for action in actions], dtype=float)
    scores[~np.isfinite(scores)] = -np.inf
    if not np.any(np.isfinite(scores)):
        raise ValueError("every candidate design has a non-finite VoI")
    index = int(np.argmax(scores))
    return actions[index], scores
