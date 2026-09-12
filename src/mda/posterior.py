"""Model-level evidence weighting and support pruning (Algorithm 6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from .numerics import logsumexp, normalize_log_weights
from .smc import Likelihood, ParameterPosterior, SMCConfig, parameter_posterior
from .types import CandidateModel, Datum


@dataclass(frozen=True)
class ModelFit:
    model: CandidateModel
    parameters: ParameterPosterior
    log_score: float


@dataclass(frozen=True)
class BeliefState:
    fits: tuple[ModelFit, ...]
    model_weights: np.ndarray
    discarded_mass: float = 0.0
    log_mixture_evidence: float = 0.0

    def __post_init__(self) -> None:
        weights = np.asarray(self.model_weights, dtype=float)
        if (
            weights.shape != (len(self.fits),)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.isclose(weights.sum(), 1.0)
        ):
            raise ValueError("model weights must be normalized and align with fits")
        positive = weights > 0.0
        object.__setattr__(
            self,
            "fits",
            tuple(fit for fit, keep in zip(self.fits, positive, strict=True) if keep),
        )
        object.__setattr__(self, "model_weights", weights[positive])

    @property
    def map_index(self) -> int:
        return int(np.argmax(self.model_weights))

    @property
    def map_fit(self) -> ModelFit:
        return self.fits[self.map_index]

    def iter_parameter_particles(self) -> Iterator[tuple[CandidateModel, np.ndarray, float]]:
        for model_weight, fit in zip(self.model_weights, self.fits, strict=True):
            for theta, parameter_weight in zip(
                fit.parameters.particles, fit.parameters.weights, strict=True
            ):
                yield fit.model, theta, float(model_weight * parameter_weight)


def _deduplicate(models: Sequence[CandidateModel]) -> list[CandidateModel]:
    unique: dict[str, CandidateModel] = {}
    for model in models:
        unique.setdefault(model.key, model)
    return list(unique.values())


def fit_model_posterior(
    dataset: Sequence[Datum],
    models: Sequence[CandidateModel],
    likelihood: Likelihood,
    smc_config: SMCConfig,
    *,
    model_cap: int,
    occam_penalty: float,
    rng: np.random.Generator,
) -> BeliefState:
    """Full-batch evidence refit followed by deterministic top-N support control."""
    if model_cap < 1:
        raise ValueError("model_cap must be positive")
    if occam_penalty < 0:
        raise ValueError("occam_penalty cannot be negative")
    candidates = _deduplicate(models)
    if not candidates:
        raise ValueError("at least one candidate model is required")

    fits = []
    for model in candidates:
        posterior = parameter_posterior(dataset, model, likelihood, smc_config, rng)
        score = posterior.log_evidence - occam_penalty * model.parameter_count
        fits.append(ModelFit(model=model, parameters=posterior, log_score=score))
    scores = np.array([fit.log_score for fit in fits])
    if not np.any(np.isfinite(scores)):
        raise ValueError("every candidate model has zero or non-finite evidence")

    # Evidence under the normalized parameter-count prior over the complete,
    # deduplicated input mixture.  Compute this before support pruning.
    log_model_prior_scores = np.array(
        [-occam_penalty * model.parameter_count for model in candidates], dtype=float
    )
    log_mixture_evidence = logsumexp(scores) - logsumexp(log_model_prior_scores)

    all_weights, _ = normalize_log_weights(scores)
    order = np.argsort(-scores, kind="stable")
    keep = order[: min(model_cap, len(order))]
    discarded_mass = float(1.0 - all_weights[keep].sum())
    kept_scores = scores[keep]
    kept_weights, _ = normalize_log_weights(kept_scores)
    return BeliefState(
        fits=tuple(fits[index] for index in keep),
        model_weights=kept_weights,
        discarded_mass=max(0.0, discarded_mass),
        log_mixture_evidence=log_mixture_evidence,
    )
