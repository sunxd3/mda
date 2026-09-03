"""Algorithm 3 and 4 likelihood implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .numerics import logsumexp, systematic_resample
from .types import CandidateModel, Datum


NoiseFn = Callable[[Datum, np.ndarray], float | np.ndarray]


def _normal_logpdf(observed: np.ndarray, mean: np.ndarray, std: float | np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if observed.shape != mean.shape:
        raise ValueError(f"observation shape {observed.shape} != prediction shape {mean.shape}")
    if np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("observation standard deviations must be finite and positive")
    residual = (observed - mean) / std
    return -0.5 * float(np.sum(residual * residual + 2 * np.log(std) + np.log(2 * np.pi)))


@dataclass(frozen=True)
class DeterministicGaussianLikelihood:
    """Exact likelihood for deterministic latent dynamics (paper Algorithm 3)."""

    observation_std: float | np.ndarray | NoiseFn

    def __call__(
        self,
        dataset: Sequence[Datum],
        model: CandidateModel,
        theta: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        del rng
        total = 0.0
        try:
            for datum in dataset:
                prediction = model.observation_prediction(theta, datum.action)
                std = (
                    self.observation_std(datum, prediction)
                    if callable(self.observation_std)
                    else self.observation_std
                )
                total += _normal_logpdf(datum.observation, prediction, std)
        except (FloatingPointError, OverflowError, ValueError):
            return -np.inf
        return total if np.isfinite(total) else -np.inf


@dataclass(frozen=True)
class BootstrapParticleFilter:
    """Bootstrap PF estimate of an intractable state-space likelihood (Algorithm 4).

    The returned *likelihood* estimate is unbiased under ordinary bootstrap
    resampling. Its logarithm is not. When nested inside likelihood-tempered SMC,
    fractional powers are therefore approximate pseudo-marginal targets, exactly
    as acknowledged in footnote 10 of the paper.
    """

    latent_particles: int
    observation_std: float | np.ndarray

    def __post_init__(self) -> None:
        if self.latent_particles < 1:
            raise ValueError("latent_particles must be positive")

    def __call__(
        self,
        dataset: Sequence[Datum],
        model: CandidateModel,
        theta: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        if model.state_space is None:
            raise ValueError(f"model {model.name!r} has no state-space callbacks")
        std = np.asarray(self.observation_std, dtype=float)
        if np.any(~np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("observation standard deviations must be finite and positive")

        total = 0.0
        state_model = model.state_space
        for datum in dataset:
            observations = np.asarray(datum.observation, dtype=float)
            if observations.ndim == 1:
                observations = observations[:, None]
            if observations.ndim != 2:
                raise ValueError("particle-filter observations must have shape (time, features)")
            states = np.asarray(
                state_model.initial_state(theta, datum.action, self.latent_particles, rng),
                dtype=float,
            )
            if states.shape[0] != self.latent_particles:
                raise ValueError("initial_state returned the wrong number of particles")

            for time_index, observed in enumerate(observations):
                states = np.asarray(
                    state_model.transition(states, theta, datum.action, time_index, rng),
                    dtype=float,
                )
                means = np.asarray(
                    state_model.observation_mean(states, theta, datum.action, time_index),
                    dtype=float,
                )
                if means.ndim == 1:
                    means = means[:, None]
                if means.shape != (self.latent_particles, observations.shape[1]):
                    raise ValueError(
                        "observation_mean must return (latent_particles, observation_features)"
                    )
                residual = (observed[None, :] - means) / std
                log_weights = -0.5 * np.sum(
                    residual * residual + 2 * np.log(std) + np.log(2 * np.pi), axis=1
                )
                increment = logsumexp(log_weights) - np.log(self.latent_particles)
                if not np.isfinite(increment):
                    return -np.inf
                total += increment
                normalized = np.exp(log_weights - logsumexp(log_weights))
                states = states[systematic_resample(normalized, rng)]
        return total if np.isfinite(total) else -np.inf
