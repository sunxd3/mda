"""Per-structure evidence backends: tempered SMC, recursive updates, and BIC.

Each backend maps ``(dataset, model, likelihood, rng)`` to a
``ParameterPosterior`` whose ``log_evidence`` feeds the model posterior. v5's
Algorithm 2 selects one of ``marglik-smc`` (the default), ``marglik-pf``
(recursive updates) or ``marglik-bic`` per domain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .numerics import effective_sample_size, logsumexp, systematic_resample
from .smc import (
    Likelihood, ParameterPosterior, SMCConfig, _evaluate_likelihoods, parameter_posterior,
)
from .types import CandidateModel, Datum


EvidenceBackend = Callable[
    [Sequence[Datum], CandidateModel, Likelihood, np.random.Generator],
    ParameterPosterior,
]


@dataclass(frozen=True)
class TemperedSMCEvidence:
    """Full-history adaptive-tempering SMC (``marglik-smc``), the paper default."""

    config: SMCConfig = field(default_factory=SMCConfig)

    def __call__(self, dataset, model, likelihood, rng) -> ParameterPosterior:
        return parameter_posterior(dataset, model, likelihood, self.config, rng)


def _cache_key(model: CandidateModel) -> tuple:
    """Structure plus prior support: one key per structure would let a law
    re-proposed with different bounds inherit a posterior outside them."""
    prior = model.prior
    bounds = tuple(
        np.asarray(getattr(prior, name), dtype=float).tobytes()
        for name in ("lower", "upper") if hasattr(prior, name)
    )
    return (model.key, *bounds)


def _is_prefix(old: Sequence[Datum], new: Sequence[Datum]) -> bool:
    return len(old) <= len(new) and all(a is b for a, b in zip(old, new))


class RecursiveEvidence:
    """Recursive parameter-weight and evidence updates (``marglik-pf``).

    A structure without a valid cache is fitted on all data by ``initial``
    (tempered SMC by default). Afterwards each newly appended experiment only
    reweights the retained parameter particles by its own likelihood, adding
    log sum_i W_i L(theta_i) to the log evidence. Particles are systematically
    resampled when the ESS falls below ``resample_fraction`` times the particle
    count; there is no rejuvenation, so parameter diversity can collapse.

    The cache is keyed by ``model.key`` and is valid only while the stored
    datasets are identity-prefixes of the new one. Calling again with the same
    data returns the cached posterior, so a re-judge after expansion only fits
    the newly proposed structures.
    """

    def __init__(
        self,
        initial: EvidenceBackend | None = None,
        *,
        resample_fraction: float = 0.5,
    ) -> None:
        if not 0 < resample_fraction <= 1:
            raise ValueError("resample_fraction must be in (0, 1]")
        self.initial = initial or TemperedSMCEvidence()
        self.resample_fraction = resample_fraction
        self._states: dict[str, tuple[tuple[Datum, ...], ParameterPosterior]] = {}

    def __call__(self, dataset, model, likelihood, rng) -> ParameterPosterior:
        dataset = tuple(dataset)
        key = _cache_key(model)
        cached = self._states.get(key)
        if cached is None or not _is_prefix(cached[0], dataset):
            posterior = self.initial(dataset, model, likelihood, rng)
        else:
            posterior = cached[1]
            for datum in dataset[len(cached[0]):]:
                posterior = self._assimilate(posterior, datum, model, likelihood, rng)
        self._states[key] = (dataset, posterior)
        return posterior

    def _assimilate(self, posterior, datum, model, likelihood, rng) -> ParameterPosterior:
        count = len(posterior.particles)
        increments = _evaluate_likelihoods([datum], model, posterior.particles, likelihood, rng)
        with np.errstate(divide="ignore"):  # zero-weight particles stay at -inf
            log_weights = np.log(np.asarray(posterior.weights, dtype=float)) + increments
        increment = logsumexp(log_weights)
        log_likelihoods = posterior.log_likelihoods + increments
        ess = list(posterior.ess_before_resampling)
        if not np.isfinite(increment) or not np.isfinite(posterior.log_evidence):
            return ParameterPosterior(
                particles=posterior.particles, weights=np.full(count, 1 / count),
                log_likelihoods=log_likelihoods, log_evidence=-np.inf,
                temperatures=posterior.temperatures, acceptance_rate=posterior.acceptance_rate,
                ess_before_resampling=tuple(ess),
                last_rung_forced_below_target=posterior.last_rung_forced_below_target,
                log_priors=posterior.log_priors, map_log_priors=posterior.map_log_priors,
            )
        weights = np.exp(log_weights - increment)
        particles = posterior.particles
        log_priors, map_log_priors = posterior.log_priors, posterior.map_log_priors
        ess.append(effective_sample_size(weights))
        if ess[-1] < self.resample_fraction * count:
            ancestors = systematic_resample(weights, rng)
            particles = particles[ancestors].copy()
            log_likelihoods = log_likelihoods[ancestors]
            log_priors = None if log_priors is None else log_priors[ancestors]
            map_log_priors = None if map_log_priors is None else map_log_priors[ancestors]
            weights = np.full(count, 1 / count)
        return ParameterPosterior(
            particles=particles, weights=weights, log_likelihoods=log_likelihoods,
            log_evidence=posterior.log_evidence + increment,
            temperatures=posterior.temperatures, acceptance_rate=posterior.acceptance_rate,
            ess_before_resampling=tuple(ess),
            last_rung_forced_below_target=posterior.last_rung_forced_below_target,
            log_priors=log_priors, map_log_priors=map_log_priors,
        )


def _bounds(model: CandidateModel) -> tuple[np.ndarray, np.ndarray]:
    lower = getattr(model.prior, "lower", None)
    upper = getattr(model.prior, "upper", None)
    if lower is None or upper is None:
        raise ValueError(f"BIC fitting of {model.name!r} requires a bounded (lower/upper) prior")
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


@dataclass(frozen=True)
class BICEvidence:
    """Bounded multistart optimization with a BIC evidence score (``marglik-bic``).

    ``residual_scale`` selects the v5 Appendix BIC variant:

    * ``"profiled"``: ``N log(RSS/N) + k log N`` with a fitted common residual
      multiplier (Eq. BIC), the ChemBench default;
    * ``"fixed"``: ``RSS + k log N`` with fixed whitening scales (Eq. BICfixed);
    * ``"likelihood"``: ``-2 loglik(theta_hat) + k log N`` using the agent's
      likelihood (deterministic, PF, or BSL), maximized under common random
      numbers so a noisy estimator gives a deterministic objective.

    RSS is whitened by ``sigma_n = max(relative_noise*|y_n|, noise_floor)``
    computed from the observations, so the scales are shared by every
    candidate. An optional ``transform`` (e.g. ``np.log1p`` for ChemBench's
    log-rate residuals) is applied to observations and predictions first,
    and ``y_n`` then denotes the transformed observation. ``N`` is the number of scalar observations unless
    ``observation_count`` is supplied. The returned posterior is a point mass
    at the best optimum and ``log_evidence = -BIC/2`` up to a constant common
    to all candidates on the same data.
    """

    residual_scale: str = "profiled"
    relative_noise: float = 0.03
    noise_floor: float = 0.01
    starts: int = 8
    max_evaluations: int = 2000
    likelihood_seed: int = 0
    observation_count: Callable[[Sequence[Datum]], int] | None = None
    transform: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.residual_scale not in {"profiled", "fixed", "likelihood"}:
            raise ValueError("residual_scale must be profiled, fixed, or likelihood")
        if self.relative_noise < 0 or self.noise_floor <= 0:
            raise ValueError("noise scales must satisfy relative_noise >= 0 and noise_floor > 0")
        if self.starts < 1 or self.max_evaluations < 1:
            raise ValueError("starts and max_evaluations must be positive")

    def _count(self, dataset: Sequence[Datum]) -> int:
        if self.observation_count is not None:
            return int(self.observation_count(dataset))
        return int(sum(datum.observation.size for datum in dataset))

    def _starts(self, model, rng) -> np.ndarray:
        lower, upper = _bounds(model)
        starts = np.asarray(model.prior.sample(rng, self.starts), dtype=float)
        return np.clip(starts, lower, upper)

    def __call__(self, dataset, model, likelihood, rng) -> ParameterPosterior:
        try:
            from scipy.optimize import least_squares, minimize
        except ImportError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("install model-discovery-agent[bic] for BIC evidence") from error
        dataset = list(dataset)
        count = self._count(dataset)
        if count < 1:
            raise ValueError("BIC requires at least one observation")
        k = model.parameter_count
        if self.residual_scale == "likelihood":
            def negative_log_likelihood(theta):
                rng_eval = np.random.default_rng(self.likelihood_seed)
                try:
                    value = float(likelihood(dataset, model, np.asarray(theta), rng_eval))
                except (ArithmeticError, ValueError):
                    return np.inf
                return -value if np.isfinite(value) else np.inf

            best_theta, best_value = self._optimize_scalar(
                negative_log_likelihood, model, rng, minimize,
            )
            log_likelihood = -best_value
            bic = -2 * log_likelihood + k * np.log(count)
        else:
            transform = self.transform or (lambda values: values)
            with np.errstate(all="ignore"):
                observed = transform(
                    np.concatenate([datum.observation.reshape(-1) for datum in dataset])
                )
            if not np.all(np.isfinite(observed)):
                raise ValueError("transformed observations must be finite")
            scales = np.maximum(self.relative_noise * np.abs(observed), self.noise_floor)

            def residuals(theta):
                try:
                    predicted = np.concatenate([
                        model.observation_prediction(theta, datum.action).reshape(-1)
                        for datum in dataset
                    ])
                except (ArithmeticError, ValueError):
                    return None
                if predicted.shape != observed.shape:
                    raise ValueError("model predictions do not match the observation shapes")
                with np.errstate(all="ignore"):
                    predicted = transform(predicted)
                if not np.all(np.isfinite(predicted)):
                    return None
                return (observed - predicted) / scales

            best_theta, rss = self._optimize_residuals(residuals, model, rng, least_squares)
            normalizer = float(np.sum(np.log(2 * np.pi * scales**2)))
            if self.residual_scale == "profiled":
                rss = max(rss, np.finfo(float).tiny)
                bic = count * np.log(rss / count) + k * np.log(count)
                log_likelihood = -0.5 * (count * np.log(rss / count) + count + normalizer)
            else:
                bic = rss + k * np.log(count)
                log_likelihood = -0.5 * (rss + normalizer)
        log_evidence = -0.5 * bic if np.isfinite(bic) else -np.inf
        log_prior = model.prior.logpdf(best_theta)
        map_prior = model.metadata.get("map_log_prior", model.prior.logpdf)(best_theta)
        return ParameterPosterior(
            particles=best_theta[None, :], weights=np.ones(1),
            log_likelihoods=np.asarray([log_likelihood if np.isfinite(bic) else -np.inf]),
            log_evidence=float(log_evidence), temperatures=(1.0,), acceptance_rate=0.0,
            log_priors=np.asarray([log_prior]), map_log_priors=np.asarray([map_prior]),
        )

    def _optimize_residuals(self, residuals, model, rng, least_squares):
        if model.parameter_count == 0:
            values = residuals(np.zeros(0))
            return np.zeros(0), (np.inf if values is None else float(values @ values))
        lower, upper = _bounds(model)
        best_theta, best_rss = None, np.inf
        for start in self._starts(model, rng):
            initial = residuals(start)
            if initial is None or not np.all(np.isfinite(initial)):
                continue
            penalty = np.full_like(initial, 1e8)

            def safe(theta):
                values = residuals(theta)
                return penalty if values is None or not np.all(np.isfinite(values)) else values

            try:
                fit = least_squares(safe, start, bounds=(lower, upper),
                                    max_nfev=self.max_evaluations, x_scale="jac")
            except (ArithmeticError, ValueError):
                continue
            values = residuals(fit.x)
            if values is None or not np.all(np.isfinite(values)):
                continue
            rss = float(values @ values)
            if rss < best_rss:
                best_theta, best_rss = np.clip(fit.x, lower, upper), rss
        if best_theta is None:
            return self._starts(model, rng)[0], np.inf
        return best_theta, best_rss

    def _optimize_scalar(self, objective, model, rng, minimize):
        if model.parameter_count == 0:
            return np.zeros(0), objective(np.zeros(0))
        lower, upper = _bounds(model)
        best_theta, best_value = None, np.inf
        for start in self._starts(model, rng):
            if not np.isfinite(objective(start)):
                continue
            bounded = lambda theta: min(objective(np.clip(theta, lower, upper)), 1e300)
            fit = minimize(bounded, start, method="Powell", bounds=list(zip(lower, upper)),
                           options={"maxfev": self.max_evaluations, "xtol": 1e-6, "ftol": 1e-9})
            theta = np.clip(fit.x, lower, upper)
            value = objective(theta)
            if value < best_value:
                best_theta, best_value = theta, value
        if best_theta is None:
            return self._starts(model, rng)[0], np.inf
        return best_theta, best_value
