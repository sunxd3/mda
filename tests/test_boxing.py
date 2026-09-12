from math import lgamma

import numpy as np
import pytest

from mda.benchmarks.boxing import (
    BoxingEnvironment, BoxingLikelihood, DOMAINS, _NumPyroProposer, make_agent,
)
from mda.types import CandidateModel, Datum, UniformPrior


def _constant(value):
    return CandidateModel("c", UniformPrior(("x",), np.array([-1.]), np.array([1.])),
                          lambda theta, action: np.asarray(value))


@pytest.mark.parametrize("family,y,mean,expected", [
    ("gaussian", 2., 1., -.5 * (1 + np.log(2*np.pi))),
    ("bernoulli", 1., .25, np.log(.25)),
    ("poisson", 3., 2., 3*np.log(2)-2-lgamma(4)),
])
def test_exact_likelihoods(family, y, mean, expected):
    got = BoxingLikelihood(family)([Datum(0, np.asarray(y))], _constant(mean),
                                   np.array([0.]), np.random.default_rng(0))
    assert got == pytest.approx(expected)


def test_binomial_prediction_is_observable_mean():
    likelihood = BoxingLikelihood("binomial", trials=50)
    got = likelihood([Datum(0, np.asarray(10))], _constant(10), np.array([0.]),
                     np.random.default_rng(0))
    assert got == pytest.approx(lgamma(51)-lgamma(11)-lgamma(41)+10*np.log(.2)+40*np.log(.8))


@pytest.mark.parametrize("family,y,mean", [
    ("bernoulli", 0, 0), ("bernoulli", 1, 1),
    ("binomial", 0, 0), ("binomial", 50, 50), ("poisson", 0, 0),
])
def test_likelihood_exact_endpoints(family, y, mean):
    likelihood = BoxingLikelihood(family, trials=50)
    assert likelihood([Datum(0, np.asarray(y))], _constant(mean), np.array([0.]),
                      np.random.default_rng(0)) == pytest.approx(0)


@pytest.mark.parametrize("domain", DOMAINS)
def test_seeded_world_streams_and_clean_truth(domain):
    a = BoxingEnvironment(domain, seed=7, candidate_count=12, query_count=4)
    b = BoxingEnvironment(domain, seed=7, candidate_count=12, query_count=4)
    assert a.actions == b.actions and a.query_actions == b.query_actions
    assert np.allclose([x.observation for x in a.initial_data],
                       [x.observation for x in b.initial_data])
    assert not set(map(repr, a.actions)).intersection(map(repr, a.query_actions))
    # A perfect conditional-mean predictor gets exactly zero except for the
    # explicitly fixed-outcome, Brier-scored classification tasks.
    score = a.evaluate_predictor(a.conditional_mean)
    assert score >= 0
    if domain not in {"irt", "survival"}: assert score == pytest.approx(0)


def test_indexed_latent_domains_include_query_feature_rows():
    for domain in ("irt", "location", "survival"):
        env = BoxingEnvironment(domain, candidate_count=20, query_count=8)
        expected_rows = {"irt": 36, "location": 28, "survival": 100}[domain]
        assert env.action_features.shape == (expected_rows, 2)
        assert max(env.query_actions) < len(env.action_features)


def test_multivariable_symbolic_actions_use_named_mapping_api():
    env = BoxingEnvironment("hyperbolic", candidate_count=4, query_count=2)
    assert set(env.actions[0]) == {"immediate", "delayed", "days"}


@pytest.mark.parametrize("domain", ["irt", "survival"])
def test_fixed_binary_outcomes_and_brier_score(domain):
    env = BoxingEnvironment(domain, seed=11, candidate_count=20, query_count=8)
    action = env.actions[0]
    assert env.step(action) == env.step(action)
    expected = np.mean([(0.25 - env.step(a)) ** 2 for a in env.query_actions])
    assert env.evaluate_predictor(lambda action: .25) == pytest.approx(expected)
    # Brier is against outcomes, independently of clean probability variance.
    assert env.evaluate_predictor(env.conditional_mean) == pytest.approx(
        np.mean([(env.conditional_mean(a) - env.step(a)) ** 2 for a in env.query_actions]))


def test_numpyro_proposer_uses_schema_pool_residuals_and_full_feature_table():
    pytest.importorskip("numpyro")
    env = BoxingEnvironment("irt", candidate_count=4, query_count=2)

    class Client:
        def ask_json(self, system, prompt, **kwargs):
            assert kwargs["schema"]["additionalProperties"] is False
            assert "old structure" in prompt and "signed_per_experiment" in prompt
            assert "feature row" in prompt and str(env.action_features[0].tolist()) in prompt
            assert "already in scope" in prompt
            return {"models": [{"name": "array", "source": """
def model(ctx):
    rows = ctx['features']
    latent = numpyro.sample('latent', dist.Normal(jnp.zeros(len(rows)), 1))
    numpyro.deterministic('mu', latent)
"""}]}

    old = CandidateModel("old", UniformPrior(("x",), np.array([-1.]), np.array([1.])),
                         lambda theta, action: np.asarray(0), structure="old structure")
    models = _NumPyroProposer(Client(), env, True).propose(
        context="task", dataset=env.initial_data, current_models=[old],
        residuals=[{"signed_per_experiment": [1, -1]}], count=1)
    theta = models[0].prior.sample(np.random.default_rng(0), 1)[0]
    assert np.isfinite(models[0].predict(theta, env.query_actions[-1]))
    assert "def model(ctx)" in models[0].structure


def test_survival_routes_to_numpyro_and_requires_opt_in(monkeypatch):
    env = BoxingEnvironment("survival", candidate_count=4, query_count=2)
    with pytest.raises(PermissionError):
        make_agent(env, object())
    monkeypatch.setattr(_NumPyroProposer, "propose",
                        lambda self, **kwargs: [_constant(.5)])
    agent = make_agent(env, object(), allow_code_execution=True)
    assert isinstance(agent.proposer, _NumPyroProposer)


def test_location_has_three_initial_observations():
    assert len(BoxingEnvironment("location", candidate_count=4, query_count=2).initial_data) == 3


def test_smoke_agent_and_unsafe_boundary():
    env = BoxingEnvironment("dugongs", candidate_count=4, query_count=2)
    agent = make_agent(env, None, smoke=True, experiment_budget=1)
    assert agent.config.model_cap == 16
    with pytest.raises(PermissionError):
        make_agent(BoxingEnvironment("irt", candidate_count=4, query_count=2), object())
