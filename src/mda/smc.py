"""Adaptive likelihood-tempered SMC over model parameters (Algorithm 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .numerics import effective_sample_size, logsumexp, normalize_log_weights, systematic_resample
from .types import CandidateModel, Datum


Likelihood = Callable[
    [Sequence[Datum], CandidateModel, np.ndarray, np.random.Generator],
    float,
]


@dataclass(frozen=True)
class SMCConfig:
    parameter_particles: int = 100
    rejuvenation_steps: int = 3
    target_ess_fraction: float = 0.6
    max_temperatures: int = 80
    proposal_scale: float = 0.5
    minimum_proposal_scale: float = 1e-6

    def __post_init__(self) -> None:
        if self.parameter_particles < 1:
            raise ValueError("parameter_particles must be positive")
        if self.rejuvenation_steps < 0:
            raise ValueError("rejuvenation_steps cannot be negative")
        if not 0 < self.target_ess_fraction <= 1:
            raise ValueError("target_ess_fraction must be in (0, 1]")
        if self.max_temperatures < 1:
            raise ValueError("max_temperatures must be positive")
        if self.proposal_scale <= 0 or self.minimum_proposal_scale <= 0:
            raise ValueError("proposal scales must be positive")


@dataclass(frozen=True)
class ParameterPosterior:
    particles: np.ndarray
    weights: np.ndarray
    log_likelihoods: np.ndarray
    log_evidence: float
    temperatures: tuple[float, ...]
    acceptance_rate: float
    ess_before_resampling: tuple[float, ...] = ()
    last_rung_forced_below_target: bool = False

    @property
    def map_index(self) -> int:
        return int(np.argmax(self.log_likelihoods))

    @property
    def map_parameters(self) -> np.ndarray:
        return self.particles[self.map_index].copy()


def _evaluate_likelihoods(
    dataset: Sequence[Datum],
    model: CandidateModel,
    particles: np.ndarray,
    likelihood: Likelihood,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.empty(len(particles), dtype=float)
    for index, theta in enumerate(particles):
        try:
            value = float(likelihood(dataset, model, theta, rng))
        except (FloatingPointError, OverflowError, ValueError):
            value = -np.inf
        values[index] = value if np.isfinite(value) else -np.inf
    return values


def _ess_after_step(log_weights: np.ndarray, log_likelihoods: np.ndarray, step: float) -> float:
    scores = log_weights + step * log_likelihoods
    finite = np.isfinite(scores)
    if not np.any(finite):
        return 0.0
    normalized, _ = normalize_log_weights(scores)
    return effective_sample_size(normalized)


def _temperature_step(
    beta: float,
    log_weights: np.ndarray,
    log_likelihoods: np.ndarray,
    target_ess: float,
) -> float:
    remaining = 1.0 - beta
    if _ess_after_step(log_weights, log_likelihoods, remaining) >= target_ess:
        return remaining
    low, high = 0.0, remaining
    for _ in range(48):
        midpoint = (low + high) / 2
        if _ess_after_step(log_weights, log_likelihoods, midpoint) < target_ess:
            high = midpoint
        else:
            low = midpoint
    return max(low, np.finfo(float).eps)


def parameter_posterior(
    dataset: Sequence[Datum],
    model: CandidateModel,
    likelihood: Likelihood,
    config: SMCConfig,
    rng: np.random.Generator,
) -> ParameterPosterior:
    """Fit one structure and estimate its marginal likelihood.

    Proposals outside the uniform-prior support are rejected rather than clipped;
    this preserves the symmetric random-walk acceptance ratio printed in Algorithm 5.
    If ``max_temperatures`` is reached, the final rung always advances fully to
    beta=1 rather than returning a silently partial posterior.  The diagnostics
    report when that forced step falls below the requested ESS target.
    """
    count = config.parameter_particles
    particles = model.prior.sample(rng, count)
    log_likelihoods = _evaluate_likelihoods(dataset, model, particles, likelihood, rng)
    if not np.any(np.isfinite(log_likelihoods)):
        return ParameterPosterior(
            particles=particles,
            weights=np.full(count, 1 / count),
            log_likelihoods=log_likelihoods,
            log_evidence=-np.inf,
            temperatures=(0.0,),
            acceptance_rate=0.0,
        )

    log_weights = np.full(count, -np.log(count))
    log_evidence = 0.0
    beta = 0.0
    temperatures = [beta]
    ess_before_resampling = []
    last_rung_forced_below_target = False
    accepted = attempted = 0

    for rung in range(config.max_temperatures):
        if beta >= 1.0 - 1e-12:
            break
        step = (
            1.0 - beta
            if rung == config.max_temperatures - 1
            else _temperature_step(
                beta,
                log_weights,
                log_likelihoods,
                config.target_ess_fraction * count,
            )
        )
        unnormalized = log_weights + step * log_likelihoods
        increment = logsumexp(unnormalized)
        if not np.isfinite(increment):
            log_evidence = -np.inf
            break
        log_evidence += increment
        weights, _ = normalize_log_weights(unnormalized)
        rung_ess = effective_sample_size(weights)
        ess_before_resampling.append(rung_ess)
        if rung == config.max_temperatures - 1:
            last_rung_forced_below_target = (
                rung_ess < config.target_ess_fraction * count
            )
        beta = min(1.0, beta + step)
        temperatures.append(beta)

        ancestors = systematic_resample(weights, rng)
        particles = particles[ancestors].copy()
        log_likelihoods = log_likelihoods[ancestors].copy()
        log_weights.fill(-np.log(count))

        if model.parameter_count == 0 or config.rejuvenation_steps == 0:
            continue
        spread = np.std(particles, axis=0)
        prior_width = model.prior.upper - model.prior.lower
        proposal_std = np.maximum(
            config.proposal_scale * spread,
            np.maximum(config.minimum_proposal_scale, prior_width * 1e-6),
        )
        for _ in range(config.rejuvenation_steps):
            for index in range(count):
                attempted += 1
                proposal = particles[index] + rng.normal(0.0, proposal_std)
                proposal_prior = model.prior.logpdf(proposal)
                if not np.isfinite(proposal_prior):
                    continue
                proposal_likelihood = _evaluate_likelihoods(
                    dataset, model, proposal[None, :], likelihood, rng
                )[0]
                if not np.isfinite(proposal_likelihood):
                    continue
                current_prior = model.prior.logpdf(particles[index])
                log_acceptance = (
                    beta * (proposal_likelihood - log_likelihoods[index])
                    + proposal_prior
                    - current_prior
                )
                if np.log(rng.random()) < min(0.0, log_acceptance):
                    particles[index] = proposal
                    log_likelihoods[index] = proposal_likelihood
                    accepted += 1

    return ParameterPosterior(
        particles=particles,
        weights=np.full(count, 1 / count),
        log_likelihoods=log_likelihoods,
        log_evidence=log_evidence,
        temperatures=tuple(temperatures),
        acceptance_rate=accepted / attempted if attempted else 0.0,
        ess_before_resampling=tuple(ess_before_resampling),
        last_rung_forced_below_target=last_rung_forced_below_target,
    )
