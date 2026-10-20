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
        "weak step (5 uA, 120 ms)",
    ]
    agent = make_agent(environment, None, smoke=True)
    model = agent.models[0]
    assert model.state_space is not None
    assert model.metadata["prediction_rollouts"] == 2
    assert agent.likelihood.latent_particles == 3
    assert agent.config.smc.parameter_particles == 2
    from mda.evidence import RecursiveEvidence
    assert isinstance(agent.evidence, RecursiveEvidence)


def test_stochastic_archive_is_frozen_unless_online_expansion_is_requested():
    environment = StochasticNeuronBenchEnvironment("ca_rebound", seed=3)
    assert make_agent(environment, None, smoke=True).config.expansion_cap == 0
    assert make_agent(environment, None, smoke=True, online_expansion=True).config.expansion_cap == 3
    with pytest.raises(ValueError, match="stochastic"):
        make_agent(NeuronBenchEnvironment("ca_rebound", seed=3), None, smoke=True, online_expansion=True)


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


def test_stochastic_evaluation_uses_v5_fixed_feature_scales(monkeypatch):
    environment = StochasticNeuronBenchEnvironment("ca_rebound", seed=1)
    truth = {
        protocol[0]: (np.arange(6, dtype=float) + index, 1, 0)
        for index, protocol in enumerate(environment.query_actions)
    }
    monkeypatch.setattr(nb.evaluator, "held_out_feature_targets", lambda *a, **k: truth)
    score = environment.evaluate_predictor(
        lambda protocol: truth[protocol[0]][0] + np.arange(1, 7)
    )
    scales = np.array([3.0, 3.0, 3.0, 3.0, 12.0, 8.0])
    assert score == pytest.approx(np.mean((np.arange(1, 7) / scales) ** 2))
    from mda.neuronbench_adapter import FEATURE_SCALES
    assert np.array_equal(FEATURE_SCALES, nb.evaluator._EVAL_FEAT_SCALES)


def test_structural_recovery_uses_channel_properties_not_names():
    from dataclasses import replace
    from mda.neuronbench_adapter import evaluate_neuron_recovery
    spec = NeuronHypothesis(
        "deliberately misleading h_sag name", "channel", 120., "depol", True,
        6., 2, -87., 4., 22., (0., 8.), (-65., -45.), (12., 180.),
    )
    model = neuron_candidate(spec)
    theta = np.array([.3, 3., -54., 22.])
    assert evaluate_neuron_recovery("ca_rebound", model, theta)["recovered"]
    assert evaluate_neuron_recovery("z_rebound", model, theta)["recovered"]
    assert not evaluate_neuron_recovery("h_sag", model, theta)["recovered"]
    theta[1] = 0
    assert evaluate_neuron_recovery("ca_rebound", model, theta)["class"] == "unknown"
    theta[1] = 3
    outward = neuron_candidate(replace(spec, reversal_mv=-77.))
    assert evaluate_neuron_recovery("d_type", outward, theta)["recovered"]
    degenerate = neuron_candidate(replace(spec, reversal_mv=-65.))
    assert evaluate_neuron_recovery("d_type", degenerate, theta)["class"] == "unknown"


def test_recovery_runner_passes_fitted_neuron_parameters():
    from mda.benchmarks.reproduce import run
    environment = NeuronBenchEnvironment("ca_rebound", seed=2)
    agent = make_agent(environment, None, smoke=True, experiment_budget=0)
    result = run(environment, agent, budget=0, smoke=True, evaluate=False, recovery=True)
    recovery = result["history"][0]["recovery"]
    assert recovery["class"] == "unknown"  # Plain baseline has no extra channel.
    assert not recovery["recovered"]


def test_stochastic_menu_allows_repeats_only_once_exhausted():
    environment = StochasticNeuronBenchEnvironment("ca_rebound", seed=3)
    agent = make_agent(environment, None, smoke=True, experiment_budget=8)
    unused = agent.available_actions()
    assert len(unused) == 7 and agent.config.repeat_when_exhausted
    agent.used_action_keys.update(agent.action_key(action) for action in unused)
    assert len(agent.available_actions()) == len(environment.actions)
