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
