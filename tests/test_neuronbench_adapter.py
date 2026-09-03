import numpy as np
import pytest

from mda.neuronbench_adapter import (
    NeuronBenchEnvironment,
    NeuronHypothesis,
    neuron_candidate,
    reference_neuron_candidate,
)


pytest.importorskip("neuronbench")


def test_environment_supplies_paper_seed_set_without_truth():
    environment = NeuronBenchEnvironment("h_sag", seed=0)
    assert [datum.action[0] for datum in environment.initial_data] == [
        "brief step (12 uA, 40 ms)",
        "long step (10 uA, 300 ms)",
    ]
    assert len(environment.actions) == 7
    assert "h_sag" not in environment.context


def test_reference_candidate_reproduces_released_plain_simulator():
    import neuronbench as nb

    world = nb.load_world("h_sag", stochastic=False, seed=0)
    protocol = world.protocol_pool[0]
    candidate = reference_neuron_candidate()
    predicted = candidate.observation_prediction(np.asarray([0.3]), protocol)
    _, segments = protocol
    current, start = nb.worlds.build_I(segments)
    _, expected = nb.worlds.simulate(
        current, start, extra=(), slow_na=False, gL=0.3, trace=True
    )
    np.testing.assert_allclose(predicted, expected[::10])
    assert candidate.target_prediction(np.asarray([0.3]), protocol).shape == (6,)


def test_neuron_candidate_key_ignores_display_name():
    values = dict(
        mechanism_type="channel",
        reversal_mv=-30.0,
        opens_on="hyperpol",
        inactivates=False,
        activation_slope_mv=8.0,
        activation_power=1,
        inactivation_half_mv=-50.0,
        inactivation_slope_mv=5.0,
        inactivation_tau_ms=100.0,
        g_bounds=(0.1, 2.0),
        half_mv_bounds=(-100.0, -40.0),
        tau_ms_bounds=(10.0, 1000.0),
    )
    assert neuron_candidate(NeuronHypothesis(name="h current", **values)).key == neuron_candidate(
        NeuronHypothesis(name="sag channel", **values)
    ).key
