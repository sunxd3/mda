"""Fox--Lu stochastic NeuronBench candidate and environment.

The one-step equations below intentionally mirror NeuronBench commit
``c354622``.  Unlike ``neuronbench.stochastic.run_particles``, they expose and
carry voltage and every gate so that a bootstrap filter can resample the latent
cell between observations.  The only numerical extension is the candidate's
sampled slow-Na depolarized tau; recovery remains the benchmark's 1500 ms.

The environment follows the v5 paper ("HHbench") where the pinned package
allows it: the brief/weak-step D0 and the fixed-scale feature metric. v5's
90-protocol menu, 20 stratified voltage samples per experiment, and shared
24-query set are not released, so the pinned package's protocol pool, ~4 ms
observation grid, and per-world held-out protocols are used instead.
"""

from __future__ import annotations

import zlib

import numpy as np

from .neuronbench_adapter import (
    NeuronHypothesis,
    _neuronbench,
    evaluate_neuron_recovery,
    neuron_candidate,
    reference_neuron_candidate,
)
from .types import CandidateModel, Datum, StateSpaceModel


# v5 Eq. hhFixedScaleMSE scales for [n_test, n_pre, run_down, adaptation, V_min, V_end].
FEATURE_SCALES = np.array([3.0, 3.0, 3.0, 3.0, 12.0, 8.0])


def _protocol(protocol):
    nb = _neuronbench()
    return nb.stochastic.make_protocol(protocol[1])


def stochastic_trace_features(trace, protocol):
    """Summarize the observed ~4 ms grid, not the deterministic ~0.1 ms grid.

    Coarse observations cannot recover spikes missed between samples. Forecasts
    and held-out targets instead average features of full simulated traces.
    """
    _, indices, start = _protocol(protocol)
    return _neuronbench().evaluator.eval_features(
        np.asarray(trace), int(np.searchsorted(indices, start))
    )


def _mechanism(spec: NeuronHypothesis | None, theta: np.ndarray):
    nb = _neuronbench()
    if spec is None:
        return [], False
    if spec.mechanism_type == "slow_na":
        return [], True
    channel = nb.Chan(
        name=spec.name,
        g=float(theta[1]),
        E=spec.reversal_mv,
        mvh=float(theta[2]),
        mk=-spec.activation_slope_mv
        if spec.opens_on == "hyperpol"
        else spec.activation_slope_mv,
        mtau=float(theta[3])
        if spec.tau_gate == "activation"
        else spec.activation_tau_ms,
        mpow=spec.activation_power,
        hvh=spec.inactivation_half_mv if spec.inactivates else None,
        hk=spec.inactivation_slope_mv,
        htau=float(theta[3])
        if spec.tau_gate == "inactivation"
        else spec.inactivation_tau_ms,
    )
    return [nb.stochastic._chan_to_dict(channel)], False


def _initial(theta, spec, particles):
    nb = _neuronbench()
    fx = nb.stochastic.fx
    v = np.full(particles, -65.0)
    am, bm, ah, bh, an, bn = fx.ab(v)
    cols = [v, am / (am + bm), ah / (ah + bh), an / (an + bn)]
    extra, slow = _mechanism(spec, theta)
    if slow:
        cols.append(np.full(particles, nb.stochastic._slow_na_init(-65.0)))
    for entry in fx.extra_init(v, extra):
        cols.append(entry["x"])
        if entry["h"] is not None:
            cols.append(entry["h"])
    return np.column_stack(cols)


def _step(states, theta, spec, current, rng, channel_count, dt):
    """Advance one integration step, with state layout V,m,h,n[,s][,x,h]."""
    nb = _neuronbench()
    fx = nb.stochastic.fx
    out = states.copy()
    v, m, h, n = (out[:, i] for i in range(4))
    am, bm, ah, bh, an, bn = fx.ab(v)
    m[:] = fx.gate_step(m, am, bm, dt, channel_count, rng)
    h[:] = fx.gate_step(h, ah, bh, dt, channel_count, rng)
    n[:] = fx.gate_step(n, an, bn, dt, channel_count, rng)
    extra, slow = _mechanism(spec, theta)
    col = 4
    gna = 120.0 * m**3 * h
    if slow:
        s = out[:, col]
        col += 1
        sinf = 1.0 / (1.0 + np.exp((v + 45.0) / 4.0))
        dep_tau = float(theta[1])
        s[:] = fx.tau_gate_step(
            s, sinf, np.where(v > -55.0, dep_tau, 1500.0), dt, channel_count, rng
        )
        gna *= s
    extra_current = 0.0
    for c in extra:
        x = out[:, col]
        col += 1
        x[:] = fx.tau_gate_step(
            x, fx.minf(v, c["mvh"], c["mk"]), c["mtau"], dt, channel_count, rng
        )
        gh = 1.0
        if c["hvh"] is not None:
            gate = out[:, col]
            col += 1
            gate[:] = fx.tau_gate_step(
                gate, fx.hinf(v, c["hvh"], c["hk"]), c["htau"], dt, channel_count, rng
            )
            gh = gate ** c.get("hpow", 1)
        extra_current += c["g"] * x ** c["mpow"] * gh * (v - c["E"])
    ionic = gna * (v - 50.0) + 36.0 * n**4 * (v + 77.0) + float(theta[0]) * (v + 54.4)
    v[:] = np.clip(v + dt * (current - ionic - extra_current), -95.0, 60.0)
    return out


def stochastic_neuron_candidate(
    spec: NeuronHypothesis | None,
    channel_count: int = 100,
    prediction_rollouts: int = 200,
    seed: int = 0,
) -> CandidateModel:
    """Build a stochastic candidate with PF callbacks and stable CRN forecasts."""
    if channel_count <= 0 or prediction_rollouts <= 0:
        raise ValueError("channel_count and prediction_rollouts must be positive")
    base = reference_neuron_candidate() if spec is None else neuron_candidate(spec)

    def rollout(theta, action, count):
        I, obs_idx, _ = _protocol(action)
        key = zlib.crc32(str(action[0]).encode())
        rng = np.random.default_rng(np.random.SeedSequence([seed, key]))
        state = _initial(theta, spec, count)
        traces = np.empty((count, len(I)))
        for index, applied in enumerate(I):
            state = _step(state, theta, spec, applied, rng, channel_count, 0.025)
            traces[:, index] = state[:, 0]
        return traces, obs_idx

    def predict(theta, action):
        traces, obs_idx = rollout(theta, action, prediction_rollouts)
        return traces[:, obs_idx].mean(axis=0)

    def predict_target(theta, action):
        nb = _neuronbench()
        traces, _ = rollout(theta, action, prediction_rollouts)
        _, _, test_start = _protocol(action)
        return np.mean(
            [nb.evaluator.eval_features(trace, test_start) for trace in traces], axis=0
        )

    def initial(theta, action, particles, rng):
        del action, rng
        return _initial(theta, spec, particles)

    def transition(states, theta, action, time_index, rng):
        I, obs_idx, _ = _protocol(action)
        begin = 0 if time_index == 0 else int(obs_idx[time_index - 1]) + 1
        for index in range(begin, int(obs_idx[time_index]) + 1):
            states = _step(states, theta, spec, I[index], rng, channel_count, 0.025)
        return states

    state_space = StateSpaceModel(
        initial_state=initial,
        transition=transition,
        observation_mean=lambda states, theta, action, index: states[:, 0],
    )
    return CandidateModel(
        name=base.name,
        structure=base.structure,
        prior=base.prior,
        predict=predict,
        predict_target=predict_target,
        state_space=state_space,
        metadata={
            **base.metadata,
            "channel_count": channel_count,
            "prediction_rollouts": prediction_rollouts,
            "observation_std_mv": 2.0,
        },
    )


class StochasticNeuronBenchEnvironment:
    """Public stochastic-world facade with evaluation RNG isolated from training."""

    def __init__(
        self, world_name: str, *, seed: int = 0, channel_count: int = 100
    ) -> None:
        nb = _neuronbench()
        self.world = nb.load_world(
            world_name, stochastic=True, n_channels=channel_count, seed=seed
        )
        self.context = (
            "A noisy single-compartment Hodgkin-Huxley neuron may contain a novel "
            "membrane mechanism. " + self.world.text_prior
        )
        # v5 fixes D0 by protocol identity (a brief 12 uA/40 ms and a weak
        # 5 uA/120 ms step), not by the ordering of the upstream design menu.
        wanted = {
            "brief step (12 uA, 40 ms)",
            "weak step (5 uA, 120 ms)",
        }
        by_label = {protocol[0]: protocol for protocol in self.world.protocol_pool}
        missing = wanted.difference(by_label)
        if missing:
            raise RuntimeError(f"NeuronBench is missing the paper D0 protocols: {sorted(missing)}")
        self.initial_protocols = [by_label[label] for label in sorted(wanted)]
        self.initial_data = [
            Datum(p, self.world.run(p).voltage) for p in self.initial_protocols
        ]
        labels = {p[0] for p in self.initial_protocols}
        self.actions = [p for p in self.world.protocol_pool if p[0] not in labels]
        self.query_actions = list(self.world.test_protocols)
        self.world_name, self.seed, self.channel_count = world_name, seed, channel_count

    @staticmethod
    def action_key(protocol) -> str:
        return protocol[0]

    @staticmethod
    def observation_std(datum: Datum, prediction: np.ndarray) -> float:
        del datum, prediction
        return 2.0

    def step(self, protocol) -> np.ndarray:
        return self.world.run(protocol).voltage

    def evaluate_model(self, model: CandidateModel, parameters: np.ndarray) -> dict:
        return evaluate_neuron_recovery(self.world_name, model, parameters)

    def evaluate_predictor(self, predictor) -> float:
        """v5 Eq. hhFixedScaleMSE: mean squared feature error over fixed scales.

        Scales are s = (3, 3, 3, 3, 12, 8) for the six features; references are
        expected features from 200 independent oracle rollouts (v5 gives no
        count; 200 is the pinned evaluator's default).
        """
        nb = _neuronbench()
        targets = nb.evaluator.held_out_feature_targets(
            self.world_name,
            stochastic=True,
            N=self.channel_count,
            seed=self.seed,
            reps=200,
        )
        truth = np.stack([targets[p[0]][0] for p in self.query_actions])
        forecast = np.stack(
            [np.asarray(predictor(p), float) for p in self.query_actions]
        )
        if forecast.shape != truth.shape:
            raise ValueError("predictor features must match held-out feature shape")
        return float(np.mean(((forecast - truth) / FEATURE_SCALES) ** 2))
