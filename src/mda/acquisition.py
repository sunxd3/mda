"""Model-EIG and joint-EIG design surrogates (v5 ``VoI``) plus task-VoI extensions.

v5 uses the model-disagreement surrogate (Eq. voi) by default and the joint
variant (Eq. voiFull) as an alternative. It leaves task-aware VoI to future
work; the two task flavours here implement v4's coordinatewise objective and
a Gaussian alternative, and are not part of the v5 method.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .numerics import weighted_covariance, weighted_mean
from .posterior import BeliefState
from .types import NonFinitePrediction


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


def _diagonal_variances(covariance: float | np.ndarray, dimension: int) -> np.ndarray:
    values = np.asarray(covariance, dtype=float)
    if values.ndim == 2:
        if values.shape != (dimension, dimension) or np.any(values != np.diag(np.diag(values))):
            raise ValueError("coordinatewise task VoI requires diagonal covariances")
        values = np.diag(values)
    if values.ndim == 0:
        values = np.full(dimension, float(values))
    if values.shape != (dimension,) or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("variances must be finite, positive, and match the prediction dimension")
    return values


def _weighted_disagreement(values, weights, covariance) -> float:
    centered = values - weighted_mean(values, weights)
    if np.asarray(covariance).ndim < 2:
        # Do not allocate a dense time-by-time matrix for independent trace noise.
        variances = _diagonal_variances(covariance, values.shape[1])
        squared = np.sum(centered**2 / variances, axis=1)
    else:
        matrix = _covariance_matrix(covariance, values.shape[1])
        squared = np.sum(centered * np.linalg.solve(matrix, centered.T).T, axis=1)
    return float(np.asarray(weights) @ squared)


def model_eig(
    action: Any,
    belief: BeliefState,
    observation_covariance: float | np.ndarray,
) -> float:
    """v5 Eq. voi: weighted between-structure predictive disagreement.

    This is the trace surrogate used by the experiments, not the log-determinant
    moment-matched MI of Eq. migauss. Their scalar rankings agree, not their values.
    """
    model_means = []
    model_weights = []
    for weight, fit in zip(belief.model_weights, belief.fits, strict=True):
        if weight == 0.0:
            continue
        predictions = np.stack(
            [_flat(fit.model.observation_prediction(theta, action)) for theta in fit.parameters.particles]
        )
        model_means.append(weighted_mean(predictions, fit.parameters.weights))
        model_weights.append(weight)
    means = np.stack(model_means)
    return _weighted_disagreement(means, np.asarray(model_weights), observation_covariance)


def joint_eig(
    action: Any,
    belief: BeliefState,
    observation_covariance: float | np.ndarray,
) -> float:
    """v5 Eq. voiFull: model plus within-structure parameter disagreement."""
    predictions = []
    weights = []
    for model, theta, weight in belief.iter_parameter_particles():
        predictions.append(_flat(model.observation_prediction(theta, action)))
        weights.append(weight)
    values = np.stack(predictions)
    weights_array = np.asarray(weights)
    return _weighted_disagreement(values, weights_array, observation_covariance)


def task_value_of_information(
    action: Any,
    query_actions: Sequence[Any],
    belief: BeliefState,
    observation_covariance: float | np.ndarray,
    target_covariance: float | np.ndarray | None = None,
) -> float:
    """v4 Eq. 43: target-to-target covariance, summed coordinatewise after whitening.

    Here ``observation_covariance`` is noise in *target* space, not raw trace
    space. It and the optional loss scale ``target_covariance`` must be scalar
    or diagonal. By default the noise variances also whiten the loss, as in the
    prose after v4 Eq. 45. Query targets must have the same coordinates as the
    target at the proposed action. No cross-coordinate conditioning is used.
    """
    particles = list(belief.iter_parameter_particles())
    weights = np.asarray([weight for _, _, weight in particles])
    observed = np.stack(
        [_flat(model.target_prediction(theta, action)) for model, theta, _ in particles]
    )
    noise = _diagonal_variances(observation_covariance, observed.shape[1])
    scale = noise if target_covariance is None else _diagonal_variances(
        target_covariance, observed.shape[1]
    )
    centered = observed - weighted_mean(observed, weights)
    variance = weighted_mean(centered**2, weights)
    values = []
    for query in query_actions:
        targets = np.stack(
            [_flat(model.target_prediction(theta, query)) for model, theta, _ in particles]
        )
        if targets.shape != observed.shape:
            raise ValueError("query and design targets must have the same coordinates")
        cross = weighted_mean((targets - weighted_mean(targets, weights)) * centered, weights)
        values.append(float(np.sum(cross**2 / (variance + noise) / scale)))
    return float(np.mean(values)) if values else 0.0


def full_observation_task_value_of_information(
    action: Any,
    query_actions: Sequence[Any],
    belief: BeliefState,
    observation_covariance: float | np.ndarray,
    target_covariance: float | np.ndarray | None = None,
) -> float:
    """Alternative Gaussian conditioning on raw observations, not literal Eq. 43.

    For each query this computes the linear-Gaussian expected reduction in target
    variance, tr(C_qy V_y^-1 C_yq). A supplied target covariance whitens target
    coordinates. This dense calculation includes cross-coordinate information.
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


def value_of_information(
    action: Any,
    belief: BeliefState,
    *,
    flavour: str,
    observation_covariance: float | np.ndarray,
    query_actions: Sequence[Any] = (),
    target_covariance: float | np.ndarray | None = None,
) -> float:
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
        "task_full_observation": lambda action: full_observation_task_value_of_information(
            action, query_actions, belief, observation_covariance, target_covariance,
        ),
    }
    if flavour not in scorers:
        raise ValueError(f"unknown design flavour {flavour!r}")
    try:
        return scorers[flavour](action)
    except (NonFinitePrediction, ArithmeticError):
        # A weighted model that cannot predict this action makes it unusable.
        return -np.inf


def choose_design(
    actions: Sequence[Any],
    belief: BeliefState,
    **score_options,
) -> tuple[Any, np.ndarray]:
    if len(actions) == 0:
        raise ValueError("the design space is empty")
    scores = np.asarray([
        value_of_information(action, belief, **score_options) for action in actions
    ], dtype=float)
    scores[~np.isfinite(scores)] = -np.inf
    if not np.any(np.isfinite(scores)):
        raise ValueError("every candidate design has a non-finite VoI")
    index = int(np.argmax(scores))
    return actions[index], scores
