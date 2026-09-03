from __future__ import annotations

import numpy as np

from mda.agent import AgentConfig, ModelDiscoveryAgent
from mda.likelihoods import DeterministicGaussianLikelihood
from mda.smc import SMCConfig
from mda.types import CandidateModel, Datum, UniformPrior


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


def test_duplicate_proposal_does_not_consume_expansion_budget():
    class DuplicateProposer:
        def propose(self, **kwargs):
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
            model_search_rounds=1,
            expansion_cap=1,
            error_threshold=0.1,
            smc=SMCConfig(parameter_particles=16, rejuvenation_steps=0),
        ),
    )
    agent.update_state(1.0, np.asarray(2.0))
    assert agent.expansion_count == 0
    assert len(agent.models) == 1
