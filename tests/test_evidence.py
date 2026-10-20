from __future__ import annotations

import numpy as np
import pytest

from mda.agent import AgentConfig, ModelDiscoveryAgent
from mda.evidence import BICEvidence, RecursiveEvidence, TemperedSMCEvidence
from mda.likelihoods import BayesianSyntheticLikelihood, DeterministicGaussianLikelihood
from mda.smc import SMCConfig
from mda.types import CandidateModel, Datum, StateSpaceModel, UniformPrior

pytest.importorskip("scipy")


def line_model():
    return CandidateModel(
        "line", UniformPrior(("a", "b"), np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
        lambda theta, x: np.asarray(theta[0] + theta[1] * x), structure="a+b*x",
    )


def constant_model(low=-5.0, high=5.0):
    return CandidateModel(
        "constant", UniformPrior(("c",), np.array([low]), np.array([high])),
        lambda theta, x: np.asarray(theta[0]), structure="c",
    )


def line_data(rng, count=12, noise=0.1):
    xs = np.linspace(-1, 1, count)
    return [Datum(float(x), np.asarray(1.0 + 2.0 * x + noise * rng.normal())) for x in xs]


def test_profiled_bic_matches_weighted_least_squares():
    data = line_data(np.random.default_rng(0))
    backend = BICEvidence(relative_noise=0.0, noise_floor=0.1)
    posterior = backend(data, line_model(), None, np.random.default_rng(1))
    xs = np.array([d.action for d in data])
    ys = np.array([float(d.observation) for d in data])
    coefficients, *_ = np.linalg.lstsq(np.c_[np.ones_like(xs), xs], ys, rcond=None)
    assert posterior.map_parameters == pytest.approx(coefficients, abs=1e-5)
    rss = float(np.sum(((ys - coefficients[0] - coefficients[1] * xs) / 0.1) ** 2))
    n = len(ys)
    assert posterior.log_evidence == pytest.approx(-0.5 * (n * np.log(rss / n) + 2 * np.log(n)))
    assert posterior.weights.tolist() == [1.0]


def test_fixed_and_likelihood_bic_variants_agree_for_gaussian_noise():
    data = line_data(np.random.default_rng(2))
    rng = np.random.default_rng(3)
    fixed = BICEvidence("fixed", relative_noise=0.0, noise_floor=0.1)(data, line_model(), None, rng)
    direct = BICEvidence("likelihood")(
        data, line_model(), DeterministicGaussianLikelihood(0.1), rng,
    )
    normalizer = len(data) * np.log(2 * np.pi * 0.1**2)
    # -2 log L = RSS + sum log(2 pi sigma^2); the constant is common to all candidates.
    assert direct.log_evidence == pytest.approx(fixed.log_evidence - 0.5 * normalizer, abs=1e-4)
    assert direct.map_parameters == pytest.approx(fixed.map_parameters, abs=1e-3)


def test_bic_penalizes_unneeded_parameters():
    rng = np.random.default_rng(4)
    data = [Datum(float(x), np.asarray(0.7 + 0.01 * rng.normal())) for x in np.linspace(-1, 1, 20)]
    backend = BICEvidence(relative_noise=0.0, noise_floor=0.01)
    constant = backend(data, constant_model(), None, rng)
    line = backend(data, line_model(), None, rng)
    assert constant.log_evidence > line.log_evidence


def test_bic_heteroscedastic_scales_use_relative_noise_and_floor():
    data = [Datum(0.0, np.asarray(10.0)), Datum(1.0, np.asarray(0.001))]
    model = CandidateModel("fixed", UniformPrior((), np.zeros(0), np.zeros(0)),
                           lambda theta, x: np.asarray(9.0 if x == 0.0 else 0.0))
    posterior = BICEvidence("fixed", relative_noise=0.03, noise_floor=0.01)(
        data, model, None, np.random.default_rng(0))
    rss = (1.0 / 0.3) ** 2 + (0.001 / 0.01) ** 2
    assert posterior.log_evidence == pytest.approx(-0.5 * rss)


def _log_evidence_on_grid(data, sigma):
    grid = np.linspace(-0.01, 0.01, 200_001)
    ys = np.array([float(d.observation) for d in data])
    log_likelihood = -0.5 * np.sum(((ys[None, :] - grid[:, None]) / sigma) ** 2
                                   + np.log(2 * np.pi * sigma**2), axis=1)
    peak = log_likelihood.max()
    return peak + np.log(np.mean(np.exp(log_likelihood - peak)))


def test_recursive_evidence_tracks_the_full_batch_evidence():
    rng = np.random.default_rng(5)
    model = constant_model(-0.01, 0.01)
    likelihood = DeterministicGaussianLikelihood(0.01)
    data = [Datum(float(i), np.asarray(0.003 + 0.01 * rng.normal())) for i in range(6)]
    backend = RecursiveEvidence(TemperedSMCEvidence(SMCConfig(parameter_particles=4000)))
    for size in range(1, len(data) + 1):
        posterior = backend(data[:size], model, likelihood, rng)
    assert posterior.log_evidence == pytest.approx(_log_evidence_on_grid(data, 0.01), abs=0.05)
    assert len(posterior.ess_before_resampling) >= 5  # one entry per assimilated datum


def test_recursive_evidence_reuses_and_invalidates_its_cache():
    calls = []

    def initial(dataset, model, likelihood, rng):
        calls.append(len(dataset))
        return TemperedSMCEvidence(SMCConfig(parameter_particles=16))(dataset, model, likelihood, rng)

    rng = np.random.default_rng(6)
    data = [Datum(float(i), np.asarray(0.0)) for i in range(3)]
    backend = RecursiveEvidence(initial)
    likelihood = DeterministicGaussianLikelihood(1.0)
    first = backend(data[:2], constant_model(), likelihood, rng)
    assert backend(data[:2], constant_model(), likelihood, rng) is first
    backend(data, constant_model(), likelihood, rng)
    assert calls == [2]
    backend([data[0], Datum(9.0, np.asarray(1.0))], constant_model(), likelihood, rng)
    assert calls == [2, 2]  # not an identity prefix: full refit


def test_bsl_matches_the_gaussian_summary_density():
    theta = np.array([0.4])
    model = CandidateModel(
        "level", UniformPrior(("mu",), np.array([-1.0]), np.array([1.0])),
        lambda theta, x: np.full(8, theta[0]),
        state_space=StateSpaceModel(
            initial_state=lambda theta, action, n, rng: np.zeros((n, 1)),
            transition=lambda states, theta, action, t, rng: np.full_like(states, theta[0]),
            observation_mean=lambda states, theta, action, t: states,
        ),
    )
    observed = np.full(8, 0.5)
    likelihood = BayesianSyntheticLikelihood(
        lambda trace, action: np.asarray([np.mean(trace)]), simulations=20_000,
        observation_std=0.2, ridge=1e-12,
    )
    value = likelihood([Datum(0, observed)], model, theta, np.random.default_rng(7))
    variance = 0.2**2 / 8
    expected = -0.5 * ((0.5 - 0.4) ** 2 / variance + np.log(2 * np.pi * variance))
    assert value == pytest.approx(expected, abs=0.03)


def test_bsl_requires_state_space_and_two_simulations():
    with pytest.raises(ValueError, match="two simulations"):
        BayesianSyntheticLikelihood(lambda y, a: y, simulations=1, observation_std=1.0)
    likelihood = BayesianSyntheticLikelihood(lambda y, a: y, simulations=4, observation_std=1.0)
    with pytest.raises(ValueError, match="state-space"):
        likelihood([Datum(0, np.zeros(2))], constant_model(), np.zeros(1), np.random.default_rng(0))


class RecordingProposer:
    def __init__(self):
        self.reports = []

    def propose(self, *, residuals, **kwargs):
        self.reports.append(residuals)
        return [line_model()]


class CountingEvidence:
    def __init__(self, backend):
        self.backend = backend
        self.fitted = []

    def __call__(self, dataset, model, likelihood, rng):
        self.fitted.append((len(dataset), model.key))
        return self.backend(dataset, model, likelihood, rng)


def test_v5_update_refits_before_expansion_and_rejudges_only_new_models():
    proposer = RecordingProposer()
    evidence = CountingEvidence(BICEvidence("fixed", relative_noise=0.0, noise_floor=0.1))
    agent = ModelDiscoveryAgent(
        context="toy", initial_data=[Datum(0.0, np.asarray(0.0))],
        initial_models=[constant_model()], proposer=proposer,
        likelihood=DeterministicGaussianLikelihood(0.1), actions=[1.0, 2.0],
        observation_covariance=0.01, evidence=evidence,
        config=AgentConfig(model_cap=4, model_search_rounds=1, error_threshold=0.1),
    )
    agent.update_state(1.0, np.asarray(2.0))
    # The constant is refitted on both points (c=1) before the proposer sees it.
    last_residual = proposer.reports[0][0]["per_experiment"][-1]
    assert last_residual == pytest.approx(0.5, abs=1e-6)
    assert evidence.fitted == [(1, "c"), (2, "c"), (2, "a+b*x")]
    assert agent.get_model().name == "line"


def test_bic_transform_scores_residuals_in_log_space():
    data = [Datum(0.0, np.asarray(np.expm1(1.0))), Datum(1.0, np.asarray(np.expm1(2.0)))]
    model = CandidateModel("fixed", UniformPrior((), np.zeros(0), np.zeros(0)),
                           lambda theta, x: np.asarray(np.expm1(1.1 + x)))
    posterior = BICEvidence("fixed", relative_noise=0.0, noise_floor=0.1, transform=np.log1p)(
        data, model, None, np.random.default_rng(0))
    assert posterior.log_evidence == pytest.approx(-0.5 * 2 * (0.1 / 0.1) ** 2)
    invalid = CandidateModel("negative", UniformPrior((), np.zeros(0), np.zeros(0)),
                             lambda theta, x: np.asarray(-2.0))
    rejected = BICEvidence(transform=np.log1p)(data, invalid, None, np.random.default_rng(0))
    assert rejected.log_evidence == -np.inf


def test_recursive_evidence_refits_when_the_same_structure_returns_with_new_bounds():
    calls = []

    def initial(dataset, model, likelihood, rng):
        calls.append(float(model.prior.upper[0]))
        return TemperedSMCEvidence(SMCConfig(parameter_particles=16))(dataset, model, likelihood, rng)

    rng = np.random.default_rng(6)
    data = [Datum(float(i), np.asarray(0.0)) for i in range(2)]
    backend = RecursiveEvidence(initial)
    likelihood = DeterministicGaussianLikelihood(1.0)
    backend(data, constant_model(-5.0, 5.0), likelihood, rng)
    wider = backend(data, constant_model(-5.0, 50.0), likelihood, rng)
    assert calls == [5.0, 50.0]
    assert np.all(wider.particles <= 50.0)


def test_bsl_rejects_observations_the_particle_filter_rejects():
    model = CandidateModel(
        "level", UniformPrior(("mu",), np.array([-1.0]), np.array([1.0])),
        lambda theta, x: np.full(8, theta[0]),
        state_space=StateSpaceModel(
            initial_state=lambda theta, action, count, rng: np.zeros((count, 1)),
            transition=lambda states, theta, action, index, rng: states,
            observation_mean=lambda states, theta, action, index: states[:, 0],
        ),
    )
    likelihood = BayesianSyntheticLikelihood(lambda y, a: y, simulations=4, observation_std=1.0)
    with pytest.raises(ValueError, match="shape"):
        likelihood([Datum(0.0, np.zeros((3, 2, 2)))], model, np.array([0.0]), np.random.default_rng(0))
