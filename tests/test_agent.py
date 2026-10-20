from __future__ import annotations

import numpy as np
import pytest

from mda.agent import AgentConfig, ModelDiscoveryAgent
from mda.likelihoods import DeterministicGaussianLikelihood
from mda.smc import SMCConfig
from mda.types import CandidateModel, Datum, UniformPrior


def fixed_model(name, predict):
    return CandidateModel(
        name,
        UniformPrior((), np.asarray([]), np.asarray([])),
        lambda theta, x: np.asarray(predict(x)),
        structure=name,
    )


def constant_model():
    return CandidateModel(
        "zero",
        UniformPrior(("offset",), np.asarray([-0.01]), np.asarray([0.01])),
        lambda theta, x: np.asarray(theta[0]),
        structure="offset",
    )


class LinearProposer:
    def __init__(self):
        self.calls = 0

    def propose(self, **kwargs):
        self.calls += 1
        return [
            CandidateModel(
                "linear",
                UniformPrior(("slope",), np.asarray([1.0]), np.asarray([3.0])),
                lambda theta, x: np.asarray(theta[0] * x),
                structure="slope*x",
            )
        ]


def test_failed_prequential_check_expands_then_refits_full_batch():
    proposer = LinearProposer()
    agent = ModelDiscoveryAgent(
        context="linear toy",
        initial_data=[Datum(0.0, np.asarray(0.0))],
        initial_models=[constant_model()],
        proposer=proposer,
        likelihood=DeterministicGaussianLikelihood(0.05),
        actions=[0.0, 1.0, 2.0],
        observation_covariance=0.05**2,
        config=AgentConfig(
            model_cap=2,
            new_models_per_expansion=1,
            model_search_rounds=1,
            expansion_cap=1,
            error_threshold=0.2,
            occam_penalty=0.0,
            smc=SMCConfig(parameter_particles=200, rejuvenation_steps=3),
        ),
        seed=10,
    )
    error = agent.update_state(1.0, np.asarray(2.0))
    assert error > 0.2
    assert proposer.calls == 1
    assert agent.get_model().name == "linear"
    assert abs(float(agent.predict(2.0)) - 4.0) < 0.2
    assert len(agent.dataset) == 2


def test_designs_are_not_repeated_by_default():
    agent = ModelDiscoveryAgent(
        context="toy",
        initial_data=[Datum(0.0, np.asarray(0.0))],
        initial_models=[constant_model()],
        likelihood=DeterministicGaussianLikelihood(1.0),
        actions=[0.0, 1.0],
        observation_covariance=1.0,
        config=AgentConfig(smc=SMCConfig(parameter_particles=8, rejuvenation_steps=0)),
    )
    assert agent.design() == 1.0


def test_duplicate_proposals_count_toward_the_expansion_cap():
    class DuplicateProposer:
        calls = 0

        def propose(self, **kwargs):
            DuplicateProposer.calls += 1
            return [constant_model()]

    agent = ModelDiscoveryAgent(
        context="toy",
        initial_data=[Datum(0.0, np.asarray(0.0))],
        initial_models=[constant_model()],
        proposer=DuplicateProposer(),
        likelihood=DeterministicGaussianLikelihood(0.05),
        actions=[0.0, 1.0],
        observation_covariance=0.05**2,
        config=AgentConfig(
            model_search_rounds=2,
            expansion_cap=1,
            error_threshold=0.1,
            smc=SMCConfig(parameter_particles=16, rejuvenation_steps=0),
        ),
    )
    agent.update_state(1.0, np.asarray(2.0))
    agent.update_state(0.5, np.asarray(3.0))
    # v5 Algorithm 2: every call increments n_e, so N_e^max bounds LLM calls.
    assert agent.expansion_count == DuplicateProposer.calls == 1
    assert len(agent.models) == 1


def test_zero_evidence_model_is_excluded_from_design_and_bma():
    agent = ModelDiscoveryAgent(
        context="toy",
        initial_data=[Datum(0.0, np.asarray(1.0))],
        initial_models=[fixed_model("good", lambda x: 1.0), fixed_model("bad", lambda x: np.nan)],
        likelihood=DeterministicGaussianLikelihood(1.0),
        actions=[0.0, 1.0],
        observation_covariance=1.0,
        config=AgentConfig(smc=SMCConfig(parameter_particles=1, rejuvenation_steps=0)),
    )

    assert agent.belief.model_weights.tolist() == [1.0]
    assert [model.name for model in agent.models] == ["good"]
    assert agent.design() == 1.0
    assert agent.last_design_scores.tolist() == [0.0]
    assert float(agent.predict(1.0, "bma")) == 1.0


def test_post_expansion_check_respects_backtest_flag():
    class ConstantProposer:
        def __init__(self):
            self.calls = 0

        def propose(self, **kwargs):
            self.calls += 1
            return [fixed_model(f"proposal-{self.calls}", lambda x: 1.0)]

    def run(backtest):
        proposer = ConstantProposer()
        agent = ModelDiscoveryAgent(
            context="toy",
            initial_data=[Datum(x, np.asarray(1.0)) for x in range(3)],
            initial_models=[fixed_model("current", lambda x: 2.0 if x in range(3) else 1.0)],
            proposer=proposer,
            likelihood=DeterministicGaussianLikelihood(1.0),
            actions=[0, 1, 2, 3],
            observation_covariance=1.0,
            config=AgentConfig(
                model_cap=4,
                new_models_per_expansion=1,
                model_search_rounds=2,
                expansion_cap=4,
                error_threshold=0.1,
                occam_penalty=0.0,
                backtest=backtest,
                smc=SMCConfig(parameter_particles=1, rejuvenation_steps=0),
            ),
        )
        error = agent.update_state(3, np.asarray(2.0))
        return agent, proposer, error

    fresh_only, fresh_proposer, fresh_error = run(False)
    historical, historical_proposer, historical_error = run(True)

    assert fresh_error == 0.5
    assert historical_error == 1.0
    assert fresh_proposer.calls == fresh_only.expansion_count == 2
    assert historical_proposer.calls == historical.expansion_count == 1
    assert abs(2.0 - float(fresh_only.predict(3))) / 2.0 == 0.5


def test_optimizer_uses_agent_score_and_budget_context():
    class Proposer:
        def __init__(self):
            self.contexts = []

        def propose(self, **kwargs):
            self.contexts.append(kwargs["context"])
            if len(self.contexts) == 1:
                return [fixed_model("left", lambda x: -x), fixed_model("right", lambda x: x)]
            return [fixed_model("new", lambda x: 3.0)]

    def optimize(score, rng):
        assert score(0.0) == -np.inf  # seed experiment excluded
        assert score(1.25) == pytest.approx(1.25**2)
        assert score(2.75) == pytest.approx(2.75**2)
        return 2.75

    proposer = Proposer()
    agent = ModelDiscoveryAgent(
        context="toy", initial_data=[Datum(0.0, np.asarray(0.0))], initial_models=[],
        proposer=proposer, likelihood=DeterministicGaussianLikelihood(1.0),
        actions=[], optimizer=optimize, experiment_budget=1, observation_covariance=1.0,
        config=AgentConfig(backtest=False, model_search_rounds=1,
                           smc=SMCConfig(parameter_particles=1, rejuvenation_steps=0)),
    )
    assert agent.design() == 2.75
    assert len(agent.last_design_scores) == 3  # returned action is not evaluated twice
    agent.update_state(2.75, np.asarray(3.0))
    assert "budget remaining: 1" in proposer.contexts[0]
    assert "budget remaining: 0" in proposer.contexts[1]
    assert "Current phase: explore" in proposer.contexts[1]
    with pytest.raises(ValueError, match="budget is exhausted"):
        agent.design()


def test_continuous_cmaes_is_wired_to_agent():
    from mda.optimizers import CMAESOptimizer

    bump = lambda x: np.exp(-((x[0] - 0.7)**2 + 2 * (x[1] + 0.4)**2))
    agent = ModelDiscoveryAgent(
        context="continuous toy", initial_data=[],
        initial_models=[fixed_model("zero", lambda x: 0.0), fixed_model("bump", bump)],
        likelihood=DeterministicGaussianLikelihood(1.0), actions=[], observation_covariance=1.0,
        optimizer=CMAESOptimizer([-2, -3], [3, 1], max_evaluations=120),
        config=AgentConfig(smc=SMCConfig(parameter_particles=1, rejuvenation_steps=0)),
    )
    assert np.allclose(agent.design(), [0.7, -0.4], atol=0.1)
    assert len(agent.last_design_scores) <= 120


def test_adaptive_pool_requires_both_gates_and_reopens_on_fresh_failure():
    class Proposer:
        def propose(self, **kwargs):
            return [fixed_model("new", lambda x: 3.0)]

    def make(initial, confidence=0.9):
        return ModelDiscoveryAgent(
            context="pool", initial_data=[Datum(0.0, np.asarray(initial))],
            initial_models=[fixed_model("one", lambda x: 1.0), fixed_model("two", lambda x: 2.0),
                            fixed_model("four", lambda x: 4.0)],
            proposer=Proposer(), likelihood=DeterministicGaussianLikelihood(0.4),
            actions=[1.0], observation_covariance=0.16,
            config=AgentConfig(model_cap=3, minimum_model_cap=2, shrink_confidence=confidence,
                               backtest=False, model_search_rounds=1, error_threshold=0.1,
                               smc=SMCConfig(parameter_particles=1, rejuvenation_steps=0)),
        )

    def updated(agent, observation=1.0):
        agent.update_state(1.0, np.asarray(observation))
        return agent

    unadapted = make(1.0)
    assert unadapted.active_model_cap == 3  # no shrinkage at initialization
    confident = updated(make(1.0))
    assert confident.active_model_cap == 2
    assert [model.name for model in confident.models] == ["one", "two"]
    # Confident but poor fit must not shrink; nor may good fit without confidence.
    assert updated(make(10.0), 10.0).active_model_cap == 3
    assert updated(make(1.0, confidence=0.999)).active_model_cap == 3  # MAP weight ~0.998
    confident.update_state(2.0, np.asarray(3.0))
    assert confident.active_model_cap == 3
    assert len(confident.models) == 3  # expansion used reopened cap before the refit


def test_task_design_has_separate_target_noise_units():
    from dataclasses import replace

    models = [
        replace(fixed_model(str(sign), lambda x: np.zeros(20)),
                predict_target=lambda theta, x, sign=sign: np.asarray(sign * x))
        for sign in (-1, 1)
    ]
    agent = ModelDiscoveryAgent(
        context="features differ from trace", initial_data=[], initial_models=models,
        likelihood=DeterministicGaussianLikelihood(1.0), actions=[1.0, 2.0],
        observation_covariance=np.ones(20), task_noise_covariance=2.0, query_actions=[3.0],
        config=AgentConfig(design_flavour="task", smc=SMCConfig(parameter_particles=1)),
    )
    assert agent.design() == 2.0
    assert agent.last_design_scores.tolist() == pytest.approx([1.5, 3.0])


def domain_limited_model():
    def predict(theta, x):
        if x < 0:
            raise ValueError("log of a negative input")
        return np.asarray(np.log(x) if x > 0 else 0.0)

    return fixed_model("log", lambda x: predict(None, x))


def test_an_experiment_the_map_cannot_evaluate_fails_the_check_instead_of_aborting():
    agent = ModelDiscoveryAgent(
        context="", initial_data=[Datum(1.0, np.asarray(0.0)), Datum(2.0, np.asarray(np.log(2.0)))],
        initial_models=[domain_limited_model(), constant_model()],
        likelihood=DeterministicGaussianLikelihood(0.05), actions=[-1.0, 3.0],
        observation_covariance=0.05**2,
        config=AgentConfig(expansion_cap=0, smc=SMCConfig(parameter_particles=16, max_temperatures=4)),
    )
    assert agent.belief.map_fit.model.name == "log"
    error = agent.update_state(-1.0, np.asarray(0.0))
    assert error == np.inf
    assert len(agent.dataset) == 3 and agent.experiments_run == 1
    assert agent.belief.map_fit.model.name == "zero"  # the only model that explains every datum


def test_a_failed_refit_leaves_the_agent_as_it_was():
    agent = ModelDiscoveryAgent(
        context="", initial_data=[Datum(1.0, np.asarray(0.0))],
        initial_models=[domain_limited_model()],
        likelihood=DeterministicGaussianLikelihood(0.05), actions=[-1.0, 3.0],
        observation_covariance=0.05**2,
        config=AgentConfig(expansion_cap=0, smc=SMCConfig(parameter_particles=16, max_temperatures=4)),
    )
    before = (len(agent.dataset), agent.experiments_run, set(agent.used_action_keys), agent.belief)
    with pytest.raises(ValueError, match="non-finite evidence"):
        agent.update_state(-1.0, np.asarray(0.0))
    assert (len(agent.dataset), agent.experiments_run, agent.used_action_keys, agent.belief) == before


def test_task_flavours_require_query_actions_and_iterators_still_mark_initial_actions():
    make = lambda **kwargs: ModelDiscoveryAgent(
        context="", initial_models=[constant_model()],
        likelihood=DeterministicGaussianLikelihood(0.05), actions=[0.0, 1.0],
        observation_covariance=0.05**2, **kwargs,
    )
    with pytest.raises(ValueError, match="query_actions"):
        make(initial_data=[Datum(0.0, np.asarray(0.0))], config=AgentConfig(design_flavour="task"))
    agent = make(initial_data=iter([Datum(0.0, np.asarray(0.0))]))
    assert agent.used_action_keys == {"0.0"}
    assert agent.available_actions() == [1.0]


def test_default_action_key_is_exact_for_arrays():
    from mda.agent import default_action_key

    first, second = np.asarray([0.1, 0.2]), np.asarray([0.1, 0.2 + 1e-12])
    assert repr(first) == repr(second)
    assert default_action_key(first) != default_action_key(second)
    assert default_action_key({"b": first, "a": 1}) == default_action_key({"a": 1, "b": first.copy()})
    assert default_action_key(1.0) == "1.0"
