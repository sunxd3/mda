from __future__ import annotations

from math import erf, log, pi, sqrt

import numpy as np

from mda.likelihoods import BootstrapParticleFilter, DeterministicGaussianLikelihood
from mda.posterior import fit_model_posterior
from mda.smc import SMCConfig, parameter_posterior
from mda.types import CandidateModel, Datum, StateSpaceModel, UniformPrior


def model(name="constant", dimensions=1, prediction=None):
    prediction = prediction or (lambda theta, action: np.asarray(theta[0]))
    return CandidateModel(
        name=name,
        structure=name,
        prior=UniformPrior(
            tuple(f"p{i}" for i in range(dimensions)),
            np.full(dimensions, -2.0),
            np.full(dimensions, 2.0),
        ),
        predict=prediction,
    )


def test_constant_likelihood_has_exact_evidence():
    posterior = parameter_posterior(
        [],
        model(),
        lambda dataset, candidate, theta, rng: -2.5,
        SMCConfig(parameter_particles=64),
        np.random.default_rng(1),
    )
    assert posterior.log_evidence == -2.5
    assert posterior.temperatures[-1] == 1.0
    assert np.isclose(posterior.weights.sum(), 1.0)


def test_uniform_gaussian_evidence_matches_analytic_integral():
    datum = Datum(0, np.asarray(0.5))
    sigma = 0.5
    posterior = parameter_posterior(
        [datum],
        model(prediction=lambda theta, action: np.asarray(theta[0])),
        DeterministicGaussianLikelihood(sigma),
        SMCConfig(
            parameter_particles=20_000,
            rejuvenation_steps=0,
            target_ess_fraction=0.05,
        ),
        np.random.default_rng(2),
    )
    cdf = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    analytic = log((cdf((2 - 0.5) / sigma) - cdf((-2 - 0.5) / sigma)) / 4)
    assert abs(posterior.log_evidence - analytic) < 0.03


def test_occam_prior_penalizes_extra_parameters_and_pruning_renormalizes():
    simple = model("simple", dimensions=1)
    complex_model = model("complex", dimensions=2)
    belief = fit_model_posterior(
        [],
        [complex_model, simple],
        lambda dataset, candidate, theta, rng: 0.0,
        SMCConfig(parameter_particles=16),
        model_cap=1,
        occam_penalty=2.0,
        rng=np.random.default_rng(3),
    )
    assert belief.map_fit.model.name == "simple"
    assert belief.model_weights.tolist() == [1.0]
    assert belief.discarded_mass > 0


def test_particle_filter_matches_one_step_gaussian_marginal():
    process_std = 0.7
    observation_std = 0.4

    def initial(theta, action, count, rng):
        return np.zeros((count, 1))

    def transition(states, theta, action, time_index, rng):
        return states + rng.normal(0, process_std, states.shape)

    state_space = StateSpaceModel(
        initial_state=initial,
        transition=transition,
        observation_mean=lambda states, theta, action, time_index: states,
    )
    candidate = CandidateModel(
        "random walk",
        UniformPrior(("unused",), np.asarray([-1.0]), np.asarray([1.0])),
        predict=lambda theta, action: np.asarray([0.0]),
        state_space=state_space,
    )
    y = 0.3
    estimate = BootstrapParticleFilter(50_000, observation_std)(
        [Datum(None, np.asarray([[y]]))], candidate, np.asarray([0.0]), np.random.default_rng(4)
    )
    total_variance = process_std**2 + observation_std**2
    analytic = -0.5 * (y**2 / total_variance + log(2 * pi * total_variance))
    assert abs(estimate - analytic) < 0.03


def test_all_impossible_parameter_particles_return_negative_infinite_evidence():
    posterior = parameter_posterior(
        [],
        model(),
        lambda dataset, candidate, theta, rng: -np.inf,
        SMCConfig(parameter_particles=8),
        np.random.default_rng(5),
    )
    assert posterior.log_evidence == -np.inf
