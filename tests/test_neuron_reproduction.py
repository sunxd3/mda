import numpy as np
import pytest

from mda.benchmarks.neuron import _StochasticNeuronProposer, make_agent
from mda.neuron_stochastic import StochasticNeuronBenchEnvironment
from mda.neuronbench_adapter import NeuronBenchEnvironment, NeuronHypothesis, neuron_candidate

nb = pytest.importorskip("neuronbench")


def test_deterministic_smoke_agent_uses_paper_configuration_and_plain_baseline():
    environment = NeuronBenchEnvironment("ca_rebound", seed=2)
    agent = make_agent(environment, None, smoke=True)
    assert agent.models[0].name == "Na+K (plain)"
    assert agent.config.model_cap == 5
    assert agent.config.new_models_per_expansion == 1
    assert agent.config.model_search_rounds == 2
    assert agent.config.expansion_cap == 3
    assert agent.config.error_threshold == pytest.approx(0.18)
    assert agent.config.occam_penalty == pytest.approx(2.0)
    assert agent.config.smc.target_ess_fraction == pytest.approx(0.5)
    assert len(agent.query_actions) == 1


def test_stochastic_d0_is_fixed_and_smoke_is_cheap():
    environment = StochasticNeuronBenchEnvironment("ca_rebound", seed=3)
    assert [p[0] for p in environment.initial_protocols] == [
        "brief step (12 uA, 40 ms)",
        "long step (10 uA, 300 ms)",
    ]
    agent = make_agent(environment, None, smoke=True)
    model = agent.models[0]
    assert model.state_space is not None
    assert model.metadata["prediction_rollouts"] == 2
    assert agent.likelihood.latent_particles == 3
    assert agent.config.smc.parameter_particles == 2


def test_stochastic_proposer_preserves_hypothesis_metadata():
    spec = NeuronHypothesis(
        "test channel", "channel", -77.0, "depol", True, 8.0, 2,
        -60.0, 5.0, 100.0, (0.2, 3.0), (-50.0, -20.0), (2.0, 20.0),
        tau_gate="inactivation", activation_tau_ms=3.0,
    )

    class Stub:
        def propose(self, **kwargs):
            return [neuron_candidate(spec)]

    model = _StochasticNeuronProposer(
        Stub(), channel_count=17, prediction_rollouts=4, seed=9
    ).propose()
    assert len(model) == 1
    assert model[0].metadata["neuron_hypothesis"] == spec
    assert model[0].metadata["channel_count"] == 17
    assert model[0].metadata["prediction_rollouts"] == 4


def test_stochastic_evaluation_uses_component_query_variance(monkeypatch):
    environment = StochasticNeuronBenchEnvironment("ca_rebound", seed=1)
    truth = {
        protocol[0]: (np.arange(6, dtype=float) + index, 1, 0)
        for index, protocol in enumerate(environment.query_actions)
    }
    monkeypatch.setattr(nb.evaluator, "held_out_feature_targets", lambda *a, **k: truth)
    score = environment.evaluate_predictor(
        lambda protocol: truth[protocol[0]][0] + np.arange(1, 7)
    )
    expected = np.mean(np.arange(1, 7, dtype=float) ** 2 / np.var(np.arange(6)))
    assert score == pytest.approx(expected)
