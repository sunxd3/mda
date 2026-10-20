"""Trajectory likelihoods: deterministic Gaussian, bootstrap PF, and BSL."""

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
    """Exact likelihood for deterministic latent dynamics (``loglik-det``)."""

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
        except (ArithmeticError, ValueError):
            return -np.inf
        return total if np.isfinite(total) else -np.inf


def _simulate_traces(model, theta, datum, count, std, rng) -> np.ndarray:
    """Draw ``count`` noisy trajectories with the datum's shape from the SSM."""
    observations = np.asarray(datum.observation, dtype=float)
    if observations.ndim > 2:
        raise ValueError("synthetic-likelihood observations must have shape (time, features)")
    shape = observations.shape if observations.ndim == 2 else (observations.size, 1)
    state_model = model.state_space
    states = np.asarray(state_model.initial_state(theta, datum.action, count, rng), dtype=float)
    traces = np.empty((count, *shape))
    for time_index in range(shape[0]):
        states = np.asarray(
            state_model.transition(states, theta, datum.action, time_index, rng), dtype=float
        )
        means = np.asarray(
            state_model.observation_mean(states, theta, datum.action, time_index), dtype=float
        ).reshape(count, shape[1])
        traces[:, time_index] = means + rng.normal(0.0, 1.0, means.shape) * std
    return traces if observations.ndim == 2 else traces.reshape(count, *observations.shape)


@dataclass(frozen=True)
class BayesianSyntheticLikelihood:
    """Gaussian synthetic log-likelihood of experiment summaries (v5 ``loglik-bsl``).

    For each experiment, ``simulations`` noisy trajectories are drawn from the
    model's state-space callbacks, reduced by ``summary(trace, action)``, and
    the observed summary is scored under a Gaussian with the simulated mean and
    covariance ``(1-shrinkage) S + shrinkage diag(S) + ridge I``. This is a
    density of summaries, not of raw trajectories.
    """

    summary: Callable[[np.ndarray, object], np.ndarray]
    simulations: int
    observation_std: float | np.ndarray
    shrinkage: float = 0.0
    ridge: float = 1e-6

    def __post_init__(self) -> None:
        if self.simulations < 2:
            raise ValueError("BSL needs at least two simulations per experiment")
        if not 0 <= self.shrinkage <= 1 or self.ridge <= 0:
            raise ValueError("shrinkage must be in [0, 1] and ridge positive")

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
        # Shape problems are configuration errors and must not become -inf below.
        if any(np.asarray(datum.observation, dtype=float).ndim > 2 for datum in dataset):
            raise ValueError("synthetic-likelihood observations must have shape (time, features)")
        total = 0.0
        try:
            for datum in dataset:
                traces = _simulate_traces(model, theta, datum, self.simulations, std, rng)
                summaries = np.stack([
                    np.asarray(self.summary(trace, datum.action), dtype=float).reshape(-1)
                    for trace in traces
                ])
                observed = np.asarray(self.summary(datum.observation, datum.action), dtype=float)
                observed = observed.reshape(-1)
                mean = summaries.mean(axis=0)
                sample = np.atleast_2d(np.cov(summaries, rowvar=False, ddof=1))
                covariance = (
                    (1 - self.shrinkage) * sample
                    + self.shrinkage * np.diag(np.diag(sample))
                    + self.ridge * np.eye(len(mean))
                )
                cholesky = np.linalg.cholesky(covariance)
                whitened = np.linalg.solve(cholesky, observed - mean)
                total += -0.5 * float(
                    whitened @ whitened
                    + 2 * np.sum(np.log(np.diag(cholesky)))
                    + len(mean) * np.log(2 * np.pi)
                )
        except (ArithmeticError, ValueError):  # LinAlgError is a ValueError
            return -np.inf
        return total if np.isfinite(total) else -np.inf


@dataclass(frozen=True)
class BootstrapParticleFilter:
    """Bootstrap PF estimate of an intractable state-space likelihood (``loglik-pf``).

    The returned *likelihood* estimate is unbiased under ordinary bootstrap
    resampling. Its logarithm is not. When nested inside likelihood-tempered SMC,
    fractional powers are therefore approximate pseudo-marginal targets, as the
    paper acknowledges in its tempered-SMC appendix.
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
            if observations.ndim == 0:  # a single scalar observation (T = 1)
                observations = observations.reshape(1, 1)
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
                # A diverged particle (NaN or infinite state) gets zero weight
                # instead of voiding the estimate; resampling then drops it.
                log_weights = np.where(np.isfinite(log_weights), log_weights, -np.inf)
                increment = logsumexp(log_weights) - np.log(self.latent_particles)
                if not np.isfinite(increment):
                    return -np.inf
                total += increment
                normalized = np.exp(log_weights - logsumexp(log_weights))
                states = states[systematic_resample(normalized, rng)]
        return total if np.isfinite(total) else -np.inf
