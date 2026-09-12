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


def test_temperature_cap_forces_beta_one_and_reports_below_target_ess():
    likelihood = lambda dataset, candidate, theta, rng: -500.0 * theta[0] ** 2
    forced = parameter_posterior(
        [],
        model(),
        likelihood,
        SMCConfig(parameter_particles=256, rejuvenation_steps=0, max_temperatures=1),
        np.random.default_rng(11),
    )
    adaptive = parameter_posterior(
        [],
        model(),
        likelihood,
        SMCConfig(parameter_particles=256, rejuvenation_steps=0, max_temperatures=20),
        np.random.default_rng(11),
    )
    target = 0.6 * 256
    assert forced.temperatures[-1] == 1.0
    assert forced.ess_before_resampling[-1] < target
    assert forced.last_rung_forced_below_target
    assert adaptive.temperatures[-1] == 1.0
    assert adaptive.ess_before_resampling[-1] >= target
    assert not adaptive.last_rung_forced_below_target


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


def test_log_mixture_evidence_uses_normalized_prior_after_dedup_before_pruning():
    zero_dimensional = model(
        "zero", dimensions=0, prediction=lambda theta, action: np.asarray(0.0)
    )
    zero_posterior = parameter_posterior(
        [],
        zero_dimensional,
        lambda dataset, candidate, theta, rng: -3.0,
        SMCConfig(parameter_particles=16),
        np.random.default_rng(12),
    )
    assert np.isclose(zero_posterior.log_evidence, -3.0)

    parameterized = model("parameterized", dimensions=2)
    duplicate = model("duplicate", dimensions=1)
    duplicate = CandidateModel(
        duplicate.name, duplicate.prior, duplicate.predict, structure=parameterized.key
    )
    belief = fit_model_posterior(
        [],
        [zero_dimensional, parameterized, duplicate],
        lambda dataset, candidate, theta, rng: -3.0 if candidate.name == "zero" else -0.5,
        SMCConfig(parameter_particles=16),
        model_cap=1,
        occam_penalty=0.7,
        rng=np.random.default_rng(12),
    )
    # Unequal model evidences distinguish normalized prior weighting from a
    # uniform average, a score sum, or evidence recomputed only after pruning.
    expected = log((np.exp(-3.0) + np.exp(-1.4) * np.exp(-0.5)) / (1 + np.exp(-1.4)))
    assert np.isclose(belief.log_mixture_evidence, expected)
    assert belief.map_fit.model.name == "parameterized"
    assert len(belief.fits) == 1
    assert belief.discarded_mass > 0.0


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


def test_nested_particle_filter_smc_model_posterior_matches_gaussian_evidence():
    process_std = 0.35
    observation_std = 0.45
    observations = np.asarray([0.3, 0.25, 0.8, 0.65])
    prior_lower, prior_upper = -0.8, 0.8
    time = np.arange(1, len(observations) + 1, dtype=float)

    def candidate(name, transition):
        return CandidateModel(
            name,
            UniformPrior(("location",), np.asarray([prior_lower]), np.asarray([prior_upper])),
            predict=lambda theta, action: np.asarray(theta[0]),
            state_space=StateSpaceModel(
                initial_state=lambda theta, action, count, rng: np.zeros((count, 1)),
                transition=transition,
                observation_mean=lambda states, theta, action, time_index: states,
            ),
        )

    random_walk = candidate(
        "random-walk drift",
        lambda states, theta, action, time_index, rng: states
        + theta[0]
        + rng.normal(0.0, process_std, states.shape),
    )
    independent_mean = candidate(
        "independent latent mean",
        lambda states, theta, action, time_index, rng: theta[0]
        + rng.normal(0.0, process_std, states.shape),
    )

    # Independently integrate the exact multivariate Gaussian likelihood over
    # each bounded prior.  The random walk has Cov(z_i, z_j) = q^2 min(i, j),
    # while the competing structure has independent latent states.
    def integrated_log_evidence(design, covariance):
        grid = np.linspace(prior_lower, prior_upper, 20_001)
        residuals = observations[:, None] - design[:, None] * grid
        precision = np.linalg.inv(covariance)
        log_likelihood = -0.5 * (
            np.einsum("it,ij,jt->t", residuals, precision, residuals)
            + np.linalg.slogdet(covariance)[1]
            + len(observations) * log(2 * pi)
        )
        peak = log_likelihood.max()
        density = np.exp(log_likelihood - peak)
        return peak + log(
            np.sum((density[:-1] + density[1:]) * np.diff(grid) / 2)
            / (prior_upper - prior_lower)
        )

    random_walk_covariance = (
        process_std**2 * np.minimum.outer(time, time)
        + observation_std**2 * np.eye(len(time))
    )
    independent_covariance = (process_std**2 + observation_std**2) * np.eye(len(time))
    expected_log_evidence = np.asarray(
        [
            integrated_log_evidence(time, random_walk_covariance),
            integrated_log_evidence(np.ones_like(time), independent_covariance),
        ]
    )
    expected_weights = np.exp(expected_log_evidence - expected_log_evidence.max())
    expected_weights /= expected_weights.sum()

    belief = fit_model_posterior(
        [Datum(None, observations[:, None])],
        [random_walk, independent_mean],
        BootstrapParticleFilter(800, observation_std),
        SMCConfig(
            parameter_particles=1_200,
            rejuvenation_steps=1,
            target_ess_fraction=0.6,
        ),
        model_cap=2,
        occam_penalty=0.0,
        rng=np.random.default_rng(2026),
    )

    actual_by_name = {fit.model.name: fit.log_score for fit in belief.fits}
    actual_log_evidence = np.asarray(
        [actual_by_name[random_walk.name], actual_by_name[independent_mean.name]]
    )
    weights_by_name = {
        fit.model.name: weight
        for fit, weight in zip(belief.fits, belief.model_weights, strict=True)
    }
    actual_weights = np.asarray(
        [weights_by_name[random_walk.name], weights_by_name[independent_mean.name]]
    )
    assert all(len(fit.parameters.temperatures) > 2 for fit in belief.fits)
    assert all(0 < fit.parameters.acceptance_rate < 1 for fit in belief.fits)
    assert np.allclose(actual_log_evidence, expected_log_evidence, atol=0.08, rtol=0.0)
    assert np.allclose(actual_weights, expected_weights, atol=0.025, rtol=0.0)


def test_all_impossible_parameter_particles_return_negative_infinite_evidence():
    posterior = parameter_posterior(
        [],
        model(),
        lambda dataset, candidate, theta, rng: -np.inf,
        SMCConfig(parameter_particles=8),
        np.random.default_rng(5),
    )
    assert posterior.log_evidence == -np.inf
