import numpy as np
import pytest

from mda.likelihoods import BootstrapParticleFilter
from mda.neuron_stochastic import stochastic_neuron_candidate, stochastic_trace_features
from mda.types import Datum

nb = pytest.importorskip("neuronbench")


def test_observed_features_use_stochastic_sampling_grid():
    protocol = ("condition then test", [(40., -10.), (40., 12.)])
    _, indices, start = nb.stochastic.make_protocol(protocol[1])
    boundary = int(np.searchsorted(indices, start))
    trace = np.full(len(indices), -65.)
    trace[boundary + 2] = 20.
    features = stochastic_trace_features(trace, protocol)
    assert features[0] == 1  # The test spike must not be classified as pre-pulse.
    assert features[1] == 0


def test_reference_one_rollout_matches_pinned_upstream():
    protocol = ("tiny", [(0.1, 2.0), (0.1, 0.0)])
    candidate = stochastic_neuron_candidate(None, prediction_rollouts=1, seed=9)
    theta = np.array([0.3])
    got = candidate.predict(theta, protocol)
    I, idx, _ = nb.stochastic.make_protocol(protocol[1])
    rng = np.random.default_rng(
        np.random.SeedSequence([9, __import__("zlib").crc32(b"tiny")])
    )
    expected = nb.stochastic.run_particles(
        1, I, {"extra": [], "slow_na": False}, 100, rng
    )[0, idx]
    np.testing.assert_allclose(got, expected)


def test_pf_state_is_carried_and_likelihood_finite():
    protocol = ("tiny", [(4.1, 0.0)])
    model = stochastic_neuron_candidate(None, prediction_rollouts=2, seed=3)
    theta = np.array([0.3])
    observation = model.predict(theta, protocol)
    state = model.state_space.initial_state(
        theta, protocol, 4, np.random.default_rng(1)
    )
    assert state.shape == (4, 4)
    moved = model.state_space.transition(
        state, theta, protocol, 0, np.random.default_rng(2)
    )
    assert moved.shape == state.shape and not np.array_equal(moved, state)
    value = BootstrapParticleFilter(8, 2.0)(
        [Datum(protocol, observation)], model, theta, np.random.default_rng(4)
    )
    assert np.isfinite(value)


def test_target_is_mean_of_nonlinear_rollout_features():
    protocol = ("tiny", [(20.0, 12.0), (8.0, 0.0)])
    model = stochastic_neuron_candidate(
        None, channel_count=1, prediction_rollouts=8, seed=7
    )
    theta = np.array([0.3])
    expected_features = model.predict_target(theta, protocol)
    coarse_mean = model.predict(theta, protocol)
    _, idx, start = nb.stochastic.make_protocol(protocol[1])
    feature_of_mean = nb.evaluator.eval_features(
        coarse_mean, np.searchsorted(idx, start)
    )
    assert expected_features.shape == (6,)
    assert not np.allclose(expected_features, feature_of_mean)


def test_predictors_share_one_cached_rollout_bit_identically(monkeypatch):
    import zlib

    import mda.neuron_stochastic as ns

    protocol = ("tiny", [(20.0, 12.0), (8.0, 0.0)])
    theta = np.array([0.3])
    I, idx, start = nb.stochastic.make_protocol(protocol[1])
    # Uncached reference: the rollout the predictors each repeated before.
    rng = np.random.default_rng(np.random.SeedSequence([7, zlib.crc32(b"tiny")]))
    state, traces = ns._initial(theta, None, 4), np.empty((4, len(I)))
    for index, applied in enumerate(I):
        state = ns._step(state, theta, None, applied, rng, 1, 0.025)
        traces[:, index] = state[:, 0]

    calls = []
    step = ns._step

    def counted(*args):
        calls.append(None)
        return step(*args)

    monkeypatch.setattr(ns, "_step", counted)
    model = stochastic_neuron_candidate(None, channel_count=1, prediction_rollouts=4, seed=7)
    mean = model.predict(theta, protocol)
    features = model.predict_target(theta, protocol)
    observed = model.metadata["observed_target_prediction"](theta, protocol)
    assert len(calls) == len(I)
    np.testing.assert_array_equal(mean, traces[:, idx].mean(axis=0))
    np.testing.assert_array_equal(
        features, np.mean([nb.evaluator.eval_features(t, start) for t in traces], axis=0)
    )
    np.testing.assert_array_equal(
        observed,
        np.mean([stochastic_trace_features(t, protocol) for t in traces[:, idx]], axis=0),
    )
    mean[:] = 0.0  # Callers get copies, so the cached summary is unchanged.
    np.testing.assert_array_equal(model.predict(theta, protocol), traces[:, idx].mean(axis=0))
    assert len(calls) == len(I)
    model.predict(np.array([0.31]), protocol)
    assert len(calls) == 2 * len(I)


def test_true_ca_rebound_passes_check_only_on_the_observation_grid():
    from mda.checks import observed_target_prediction, relative_residual
    from mda.neuron_stochastic import StochasticNeuronBenchEnvironment
    from mda.neuronbench_adapter import NeuronHypothesis

    # The statistic at the truth still varies with the world's realisation (a
    # single realisation's integer counts against their expectation leave
    # ~0.1-0.15 per count feature); world seed 1 is one the truth passes.
    environment = StochasticNeuronBenchEnvironment("ca_rebound", seed=1)
    truth = NeuronHypothesis(
        "T", "channel", 120.0, "depol", True, 6.0, 2, -87.0, 4.0, 22.0,
        (0.1, 4.0), (-70.0, -40.0), (10.0, 300.0),
        tau_gate="inactivation", activation_tau_ms=2.0,
    )
    model = stochastic_neuron_candidate(truth, prediction_rollouts=32, seed=0)
    theta = np.array([0.3, 3.2, -54.0, 22.0])

    def statistic(predict):
        return np.median([
            relative_residual(
                stochastic_trace_features(datum.observation, datum.action),
                predict(theta, datum.action),
                2.0,
            )
            for datum in environment.initial_data
        ])

    assert statistic(observed_target_prediction(model)) < 0.15
    # Fine-grid rollouts count the spikes the ~4 ms samples miss.
    assert statistic(model.target_prediction) > 0.3
