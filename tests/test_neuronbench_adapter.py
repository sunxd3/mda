import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mda.neuronbench_adapter import (
    NeuronBenchEnvironment,
    NeuronBenchProposer,
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


def test_sampled_tau_is_routed_to_selected_chan_gate(monkeypatch):
    captured = []

    class Chan:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    stub = SimpleNamespace(
        Chan=Chan,
        worlds=SimpleNamespace(
            build_I=lambda segments: (np.zeros(20), 0),
            simulate=lambda *args, **kwargs: (0, np.zeros(20)),
        ),
    )
    monkeypatch.setattr("mda.neuronbench_adapter._neuronbench", lambda: stub)
    base = dict(
        name="anonymous", mechanism_type="channel", reversal_mv=120.0,
        opens_on="depol", inactivates=True, activation_slope_mv=6.0,
        activation_power=2, inactivation_half_mv=-87.0,
        inactivation_slope_mv=4.0, inactivation_tau_ms=130.0,
        g_bounds=(0.1, 4.0), half_mv_bounds=(-70.0, -40.0),
        tau_ms_bounds=(10.0, 300.0),
    )
    protocol = ("tiny", [(1.0, 0.0)])
    neuron_candidate(NeuronHypothesis(**base)).observation_prediction(
        np.asarray([0.3, 2.0, -54.0, 22.0]), protocol
    )
    neuron_candidate(NeuronHypothesis(
        **base, tau_gate="inactivation", activation_tau_ms=2.0
    )).observation_prediction(np.asarray([0.3, 2.0, -54.0, 22.0]), protocol)

    assert (captured[0]["mtau"], captured[0]["htau"]) == (22.0, 130.0)
    assert (captured[1]["mtau"], captured[1]["htau"]) == (2.0, 22.0)


def test_effective_tau_kinetics_distinguish_structures_and_validate_gate():
    values = dict(
        name="Ca", mechanism_type="channel", reversal_mv=120.0,
        opens_on="depol", inactivates=True, activation_slope_mv=6.0,
        activation_power=2, inactivation_half_mv=-87.0,
        inactivation_slope_mv=4.0, inactivation_tau_ms=999.0,
        g_bounds=(0.1, 4.0), half_mv_bounds=(-70.0, -40.0),
        tau_ms_bounds=(10.0, 300.0), tau_gate="inactivation",
    )
    ca2 = neuron_candidate(NeuronHypothesis(**values, activation_tau_ms=2.0))
    ca3 = neuron_candidate(NeuronHypothesis(**values, activation_tau_ms=3.0))
    assert ca2.key != ca3.key
    # A fixed inactivation tau is ineffective when that gate's tau is sampled.
    assert ca2.key == neuron_candidate(NeuronHypothesis(
        **{**values, "inactivation_tau_ms": 50.0}, activation_tau_ms=2.0
    )).key
    with pytest.raises(ValueError, match="requires an inactivating"):
        NeuronHypothesis(**{**values, "inactivates": False}, activation_tau_ms=2.0)


def test_asymmetric_inactivation_tau_candidate_produces_real_trace():
    environment = NeuronBenchEnvironment("ca_rebound", seed=0)
    candidate = neuron_candidate(NeuronHypothesis(
        name="Ca", mechanism_type="channel", reversal_mv=120.0,
        opens_on="depol", inactivates=True, activation_slope_mv=6.0,
        activation_power=2, inactivation_half_mv=-87.0,
        inactivation_slope_mv=4.0, inactivation_tau_ms=999.0,
        g_bounds=(0.1, 4.0), half_mv_bounds=(-70.0, -40.0),
        tau_ms_bounds=(10.0, 300.0), tau_gate="inactivation",
        activation_tau_ms=2.0,
    ))
    trace = candidate.observation_prediction(
        np.asarray([0.3, 3.2, -54.0, 22.0]), environment.actions[0]
    )
    assert trace.ndim == 1 and trace.size > 100
    assert np.isfinite(trace).all()


def _slow_na_spec():
    return NeuronHypothesis(
        name="slow use dependence", mechanism_type="slow_na", reversal_mv=50.0,
        opens_on="depol", inactivates=True, activation_slope_mv=4.0,
        activation_power=3, inactivation_half_mv=-45.0,
        inactivation_slope_mv=4.0, inactivation_tau_ms=380.0,
        g_bounds=(0.1, 1.0), half_mv_bounds=(-60.0, -30.0),
        tau_ms_bounds=(1.0, 2.0),
    )


def test_slow_na_tau_380_reproduces_entire_upstream_trace():
    import neuronbench as nb

    protocol = (
        "depolarization and recovery",
        [(300.0, 12.0), (250.0, 0.0), (300.0, 12.0)],
    )
    current, start = nb.worlds.build_I(protocol[1])
    _, expected = nb.worlds.simulate(
        current, start, extra=(), slow_na=True, gL=0.3, trace=True
    )
    candidate = neuron_candidate(_slow_na_spec())
    actual = candidate.observation_prediction(np.asarray([0.3, 380.0]), protocol)

    assert candidate.prior.names == ("gL", "tau_s_ms")
    np.testing.assert_array_equal(candidate.prior.lower, [0.2, 200.0])
    np.testing.assert_array_equal(candidate.prior.upper, [0.45, 600.0])
    assert "recovery_tau_s_ms=1500" in candidate.structure
    np.testing.assert_allclose(actual, expected[::10], rtol=1e-13, atol=1e-13)


def test_slow_na_tau_changes_paired_pulse_features():
    environment = NeuronBenchEnvironment("na_fatigue", seed=0)
    paired = next(p for p in environment.actions if p[0].startswith("paired long pulses"))
    candidate = neuron_candidate(_slow_na_spec())
    fast = candidate.target_prediction(np.asarray([0.3, 200.0]), paired)
    slow = candidate.target_prediction(np.asarray([0.3, 600.0]), paired)
    # Paired-pulse run-down/adaptation, not merely trace finiteness, responds to tau.
    assert not np.allclose(fast[2:4], slow[2:4])


def test_observation_std_converts_table14_floor_to_raw_mv():
    prediction = np.asarray([-0.1, 0.1, -25.0, 25.0, -70.0])
    actual = NeuronBenchEnvironment.observation_std(None, prediction)
    np.testing.assert_allclose(actual, [3.0, 3.0, 3.0, 3.0, 8.4])


def test_evaluate_predictor_uses_component_query_variances_without_training_mutation(
    monkeypatch,
):
    actions = ["q1", "q2", "q3"]
    truth = {
        "q1": np.asarray([1.0, 10.0]),
        "q2": np.asarray([3.0, 14.0]),
        "q3": np.asarray([5.0, 10.0]),
    }
    forecasts = {
        "q1": np.asarray([2.0, 8.0]),
        "q2": np.asarray([3.0, 16.0]),
        "q3": np.asarray([8.0, 10.0]),
    }

    class World:
        def __init__(self):
            self.calls = []

        def run(self, action):
            self.calls.append(action)
            return SimpleNamespace(voltage=truth[action])

    environment = object.__new__(NeuronBenchEnvironment)
    environment.world = World()
    environment.query_actions = actions
    environment.initial_data = ["training sentinel"]
    monkeypatch.setattr(
        "mda.neuronbench_adapter.trace_features", lambda trace, protocol: np.asarray(trace)
    )
    before = list(environment.initial_data)
    score = environment.evaluate_predictor(lambda action: forecasts[action])

    # Component MSEs 10/3 and 8/3, query variances 8/3 and 32/9:
    # mean(5/4, 3/4) = 1, unlike a ratio of pooled sums or fixed-scale MSE.
    assert score == pytest.approx(1.0)
    assert environment.world.calls == actions
    assert environment.initial_data == before


def test_evaluate_predictor_rejects_zero_query_variance(monkeypatch):
    environment = object.__new__(NeuronBenchEnvironment)
    environment.query_actions = ["a", "b"]
    environment.world = SimpleNamespace(
        run=lambda action: SimpleNamespace(voltage=np.asarray([1.0, 2.0 if action == "a" else 4.0]))
    )
    monkeypatch.setattr(
        "mda.neuronbench_adapter.trace_features", lambda trace, protocol: np.asarray(trace)
    )
    with pytest.raises(ValueError, match="zero query variance"):
        environment.evaluate_predictor(lambda action: np.asarray([1.0, 3.0]))


def test_evaluate_predictor_real_queries_smoke():
    from mda.neuronbench_adapter import trace_features

    environment = NeuronBenchEnvironment("h_sag", seed=0)
    score = environment.evaluate_predictor(
        lambda action: trace_features(environment.world.run(action).voltage, action)
    )
    assert score == pytest.approx(0.0)


def test_neuron_proposer_receives_structures_priors_and_residuals():
    class Client:
        def ask_json(self, system, prompt, **kwargs):
            self.prompt = prompt
            return {"hypotheses": []}

    spec = NeuronHypothesis(
        name="anonymous", mechanism_type="channel", reversal_mv=-30.0,
        opens_on="hyperpol", inactivates=True, activation_slope_mv=8.0,
        activation_power=2, inactivation_half_mv=-75.0,
        inactivation_slope_mv=4.0, inactivation_tau_ms=130.0,
        g_bounds=(0.1, 2.0), half_mv_bounds=(-100.0, -40.0),
        tau_ms_bounds=(10.0, 1000.0),
    )
    # Identical names must not hide different structures. Include the plain
    # reference, which has no neuron_hypothesis metadata, in the same pool.
    models = [
        neuron_candidate(spec),
        neuron_candidate(replace(spec, reversal_mv=120.0, opens_on="depol")),
        reference_neuron_candidate(),
    ]
    reports = [
        {"name": model.name, "structure": model.key,
         "median_relative_residual": 0.4, "per_experiment": errors}
        for model, errors in zip(models, ([0.1, 0.7], [0.7, 0.1], [0.2, 0.6]), strict=True)
    ]
    environment = NeuronBenchEnvironment("h_sag")
    client = Client()
    result = NeuronBenchProposer(client).propose(
        context="Unknown neuron", dataset=environment.initial_data,
        current_models=models, residuals=reports, count=1,
    )

    assert result == []
    structures, sent_reports = [
        json.loads(line) for line in client.prompt.splitlines() if line.startswith("[{")
    ]
    assert len(structures) == 3
    assert [item["structure"] for item in structures] == [model.key for model in models]
    first = structures[0]["mechanism"]
    assert first["reversal_mv"] == -30.0
    assert first["opens_on"] == "hyperpol"
    assert first["activation_power"] == 2
    assert first["inactivation_tau_ms"] == 130.0
    assert first["tau_gate"] == "activation"
    assert first["activation_tau_ms"] == 1.0
    assert structures[1]["mechanism"]["reversal_mv"] == 120.0
    assert structures[1]["mechanism"]["opens_on"] == "depol"
    assert structures[0]["uniform_parameter_priors"] == {
        "gL": [0.2, 0.45], "g": [0.1, 2.0],
        "half_mv": [-100.0, -40.0], "tau_ms": [10.0, 1000.0],
    }
    assert structures[2]["mechanism"] is None
    assert structures[2]["uniform_parameter_priors"] == {"gL": [0.2, 0.45]}
    assert sent_reports == reports
    assert "per_experiment follows the experiment order above" in client.prompt
    assert "tau_gate identifies" in client.prompt
    assert "fixed bounds gL in [0.2, 0.45]" in client.prompt
    assert "tau_s_ms in\n[200, 600]" in client.prompt
    assert "recovery time constant is fixed at 1500 ms" in client.prompt
    assert "Ih/M" not in client.prompt
    assert "Ca/D" not in client.prompt
