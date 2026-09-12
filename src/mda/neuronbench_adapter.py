"""Deterministic NeuronBench adapter using the paper's data and proposal contracts.

The released benchmark remains the source of truth for protocols and HH simulation.
This module supplies the missing agent/environment glue without reading its private
truth fields. Stochastic model fitting uses the generic ``BootstrapParticleFilter``
once a domain model supplies ``StateSpaceModel`` callbacks; the released benchmark
does not expose one-step transitions, so this adapter deliberately does not fake them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .openai_client import OpenAIJSONClient
from .types import CandidateModel, Datum, UniformPrior


def _neuronbench():
    try:
        import neuronbench as nb
    except ImportError as error:
        raise RuntimeError("install model-discovery-agent[neuron]") from error
    return nb


def _test_start(protocol) -> int:
    nb = _neuronbench()
    _, segments = protocol
    current, start = nb.worlds.build_I(segments)
    indices = np.arange(0, len(current), 10)
    return int(np.searchsorted(indices, start))


def trace_features(trace: np.ndarray, protocol) -> np.ndarray:
    return _neuronbench().evaluator.eval_features(np.asarray(trace), _test_start(protocol))


def _slow_na_trace(current: np.ndarray, g_leak: float, depolarized_tau_ms: float) -> np.ndarray:
    """Roll out pinned slow-Na dynamics with recovery fixed at 1500 ms.

    The candidate's sampled tau is assumed to replace only the upstream 380 ms
    depolarized (v > -55 mV) time constant; recovery remains fixed.
    """
    # Keep the numerical backbone at the pinned dependency's source of truth.
    worlds = _neuronbench().worlds
    rates, clip = worlds._ab, worlds._clip
    dt = 0.01
    v = -65.0
    am, bm, ah, bh, an, bn = rates(v)
    m, h, n = am / (am + bm), ah / (ah + bh), an / (an + bn)
    s = 1.0
    voltage = np.empty(len(current), dtype=float)
    for index, applied in enumerate(current):
        am, bm, ah, bh, an, bn = rates(v)
        m += dt * (am * (1 - m) - bm * m)
        h += dt * (ah * (1 - h) - bh * h)
        n += dt * (an * (1 - n) - bn * n)
        m = clip(m, 0, 1)
        h = clip(h, 0, 1)
        n = clip(n, 0, 1)
        s_inf = 1 / (1 + np.exp((v + 45) / 4))
        tau = depolarized_tau_ms if v > -55 else 1500.0
        s += dt * (s_inf - s) / tau
        s = clip(s, 0, 1)
        ionic = (
            120 * m**3 * h * s * (v - 50)
            + 36 * n**4 * (v + 77)
            + g_leak * (v + 54.4)
        )
        v += dt * (float(applied) - ionic)
        v = clip(v, -95, 80)
        voltage[index] = v
    return voltage


@dataclass(frozen=True)
class NeuronHypothesis:
    name: str
    mechanism_type: str
    reversal_mv: float
    opens_on: str
    inactivates: bool
    activation_slope_mv: float
    activation_power: int
    inactivation_half_mv: float
    inactivation_slope_mv: float
    inactivation_tau_ms: float
    g_bounds: tuple[float, float]
    half_mv_bounds: tuple[float, float]
    tau_ms_bounds: tuple[float, float]
    # Table 14 samples the activation time constant for Ih/M, but the
    # inactivation time constant for Ca/D.  Defaults preserve the original
    # generic-channel contract.
    tau_gate: str = "activation"
    activation_tau_ms: float = 1.0

    def __post_init__(self) -> None:
        if self.mechanism_type not in {"channel", "slow_na"}:
            raise ValueError("mechanism_type must be channel or slow_na")
        if self.opens_on not in {"depol", "hyperpol"}:
            raise ValueError("opens_on must be depol or hyperpol")
        if self.tau_gate not in {"activation", "inactivation"}:
            raise ValueError("tau_gate must be activation or inactivation")
        if self.tau_gate == "inactivation" and not self.inactivates:
            raise ValueError("sampling inactivation tau requires an inactivating channel")
        for bounds in (self.g_bounds, self.half_mv_bounds, self.tau_ms_bounds):
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError("all neuron parameter bounds must be finite and ordered")
        if self.activation_power < 1 or self.activation_slope_mv <= 0:
            raise ValueError("activation power and slope magnitude must be positive")
        if not np.isfinite(self.activation_tau_ms) or self.activation_tau_ms <= 0:
            raise ValueError("fixed activation tau must be positive and finite")
        if self.inactivates and (
            self.inactivation_slope_mv <= 0 or self.inactivation_tau_ms <= 0
        ):
            raise ValueError("inactivation kinetics must be positive")


def neuron_candidate(spec: NeuronHypothesis) -> CandidateModel:
    """Compile an LLM-proposed mechanism into the released deterministic HH simulator."""
    nb = _neuronbench()
    if spec.mechanism_type == "slow_na":
        names = ("gL", "tau_s_ms")
        lower = np.asarray([0.2, 200.0])
        upper = np.asarray([0.45, 600.0])
    else:
        names = ("gL", "g", "half_mv", "tau_ms")
        lower = np.asarray([0.2, spec.g_bounds[0], spec.half_mv_bounds[0], spec.tau_ms_bounds[0]])
        upper = np.asarray([0.45, spec.g_bounds[1], spec.half_mv_bounds[1], spec.tau_ms_bounds[1]])
    prior = UniformPrior(names, lower, upper)

    def mechanism(theta: np.ndarray):
        channel = nb.Chan(
            name=spec.name,
            g=float(theta[1]),
            E=spec.reversal_mv,
            mvh=float(theta[2]),
            mk=(-spec.activation_slope_mv if spec.opens_on == "hyperpol" else spec.activation_slope_mv),
            mtau=(float(theta[3]) if spec.tau_gate == "activation" else spec.activation_tau_ms),
            mpow=spec.activation_power,
            hvh=(spec.inactivation_half_mv if spec.inactivates else None),
            hk=spec.inactivation_slope_mv,
            htau=(float(theta[3]) if spec.tau_gate == "inactivation" else spec.inactivation_tau_ms),
        )
        return {"extra": [channel], "slow_na": False}

    def predict(theta: np.ndarray, protocol) -> np.ndarray:
        _, segments = protocol
        current, start = nb.worlds.build_I(segments)
        if spec.mechanism_type == "slow_na":
            return _slow_na_trace(current, float(theta[0]), float(theta[1]))[::10]
        _, voltage = nb.worlds.simulate(
            current,
            start,
            **mechanism(theta),
            gL=float(theta[0]),
            trace=True,
        )
        return voltage[::10]

    return CandidateModel(
        name=spec.name,
        structure=repr(
            (spec.mechanism_type, "depolarized_tau_s_ms", "recovery_tau_s_ms=1500")
            if spec.mechanism_type == "slow_na"
            else (
                spec.mechanism_type,
                spec.reversal_mv,
                spec.opens_on,
                spec.inactivates,
                spec.activation_slope_mv,
                spec.activation_power,
                spec.inactivation_half_mv if spec.inactivates else None,
                spec.inactivation_slope_mv if spec.inactivates else None,
                spec.tau_gate,
                # Include only the fixed tau alongside the identity of the
                # sampled gate; the sampled tau's support appears below.
                spec.activation_tau_ms if spec.tau_gate == "inactivation" else None,
                spec.inactivation_tau_ms
                if spec.inactivates and spec.tau_gate == "activation" else None,
                spec.g_bounds,
                spec.half_mv_bounds,
                spec.tau_ms_bounds,
            )
        ),
        prior=prior,
        predict=predict,
        predict_target=lambda theta, protocol: trace_features(predict(theta, protocol), protocol),
        metadata={"neuron_hypothesis": spec},
    )


def reference_neuron_candidate() -> CandidateModel:
    """The non-leaking plain Na/K/leak reference model supplied by the benchmark."""
    nb = _neuronbench()
    prior = UniformPrior(("gL",), np.asarray([0.2]), np.asarray([0.45]))

    def predict(theta: np.ndarray, protocol) -> np.ndarray:
        _, segments = protocol
        current, start = nb.worlds.build_I(segments)
        _, voltage = nb.worlds.simulate(
            current,
            start,
            extra=(),
            slow_na=False,
            gL=float(theta[0]),
            trace=True,
        )
        return voltage[::10]

    return CandidateModel(
        name="Na+K (plain)",
        structure="plain Na/K/leak",
        prior=prior,
        predict=predict,
        predict_target=lambda theta, protocol: trace_features(predict(theta, protocol), protocol),
    )


class NeuronBenchEnvironment:
    """Algorithm 1 environment facade with the paper's fixed two-protocol D0."""

    def __init__(self, world_name: str, *, seed: int = 0) -> None:
        nb = _neuronbench()
        self.world = nb.load_world(world_name, stochastic=False, seed=seed)
        self.context = (
            "A single-compartment conductance-based Hodgkin-Huxley neuron has standard Na+/K+ "
            "spiking currents and may contain a novel membrane mechanism. " + self.world.text_prior
        )
        self.initial_protocols = self.world.protocol_pool[:2]
        self.initial_data = [
            Datum(protocol, self.world.run(protocol).voltage) for protocol in self.initial_protocols
        ]
        initial_labels = {protocol[0] for protocol in self.initial_protocols}
        self.actions = [p for p in self.world.protocol_pool if p[0] not in initial_labels]
        self.query_actions = list(self.world.test_protocols)

    @staticmethod
    def target(trace: np.ndarray, protocol) -> np.ndarray:
        return trace_features(trace, protocol)

    @staticmethod
    def action_key(protocol) -> str:
        return protocol[0]

    @staticmethod
    def observation_std(datum: Datum, prediction: np.ndarray) -> np.ndarray:
        del datum
        # Table 14 uses standardized voltage with 1 unit = 3 mV.  Traces here
        # remain in raw mV, so its one-unit floor is 3 mV.
        return np.maximum(0.12 * np.abs(prediction), 3.0)

    def evaluate_predictor(self, predictor) -> float:
        """Evaluate Eq. 1 on held-out queries, separately from training data."""
        truth = np.stack([
            trace_features(self.world.run(action).voltage, action)
            for action in self.query_actions
        ])
        forecast = np.stack([np.asarray(predictor(action), dtype=float) for action in self.query_actions])
        if forecast.shape != truth.shape:
            raise ValueError("predictor features must match held-out feature shape")
        query_variance = np.var(truth, axis=0)
        if np.any(query_variance == 0):
            raise ValueError("Eq. 1 is undefined for zero query variance")
        component_mse = np.mean((forecast - truth) ** 2, axis=0)
        return float(np.mean(component_mse / query_variance))

    def step(self, protocol) -> np.ndarray:
        return self.world.run(protocol).voltage


class NeuronBenchProposer:
    """Electrophysiology structure proposer based on Appendix H.4.1."""

    def __init__(self, client: OpenAIJSONClient, *, feature_fn=trace_features) -> None:
        self.client = client
        self.feature_fn = feature_fn

    def propose(
        self,
        *,
        context: str,
        dataset: Sequence[Datum],
        current_models: Sequence[CandidateModel],
        residuals: Sequence[dict[str, Any]],
        count: int,
    ) -> list[CandidateModel]:
        experiments = "\n".join(
            f"  - {datum.action[0]} -> {self.feature_fn(datum.observation, datum.action).tolist()}"
            for datum in dataset
        )
        tried = json.dumps([
            {
                "name": model.name,
                "structure": model.key,
                "mechanism": (
                    asdict(model.metadata["neuron_hypothesis"])
                    if "neuron_hypothesis" in model.metadata else None
                ),
                "uniform_parameter_priors": {
                    name: [float(low), float(high)]
                    for name, low, high in zip(
                        model.prior.names, model.prior.lower, model.prior.upper, strict=True
                    )
                },
            }
            for model in current_models
        ])
        prompt = f"""{context}

Each experiment reports [n_test, n_pre, run_down, adaptation, V_min, V_end].
Experiments:
{experiments}

Mechanisms already tried (do not repeat these structures, even under new names):
{tried}

Residual reports (matched by structure; per_experiment follows the experiment order above):
{json.dumps(list(residuals))}

Propose {count} distinct candidate mechanisms beyond the standard Na+/K+ currents.
A mechanism may instead be slow use-dependent Na inactivation. Reversal potential,
activation direction, powers, slopes, and inactivation kinetics define structure;
g, activation half-voltage, and one explicitly selected gate tau have uniform-prior bounds.
For generic channels, tau_gate identifies whether tau_ms is the activation or
inactivation time constant; activation_tau_ms is fixed when inactivation tau is
sampled, and inactivation sampling requires inactivates=true. For slow use-dependent
Na inactivation, use the fixed bounds gL in [0.2, 0.45] and depolarized tau_s_ms in
[200, 600] ms; its recovery time constant is fixed at 1500 ms."""
        number_pair = {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        }
        properties = {
            "name": {"type": "string"},
            "mechanism_type": {"type": "string", "enum": ["channel", "slow_na"]},
            "reversal_mv": {"type": "number"},
            "opens_on": {"type": "string", "enum": ["depol", "hyperpol"]},
            "inactivates": {"type": "boolean"},
            "activation_slope_mv": {"type": "number"},
            "activation_power": {"type": "integer", "minimum": 1, "maximum": 4},
            "inactivation_half_mv": {"type": "number"},
            "inactivation_slope_mv": {"type": "number"},
            "inactivation_tau_ms": {"type": "number"},
            "tau_gate": {"type": "string", "enum": ["activation", "inactivation"]},
            "activation_tau_ms": {"type": "number", "exclusiveMinimum": 0},
            "g_bounds": number_pair,
            "half_mv_bounds": number_pair,
            "tau_ms_bounds": number_pair,
        }
        schema = {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties),
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["hypotheses"],
            "additionalProperties": False,
        }
        payload = self.client.ask_json(
            "You are an electrophysiologist proposing ion-channel mechanisms. Reply JSON only.",
            prompt,
            schema=schema,
            name="neuron_hypotheses",
            max_tokens=2400,
        )
        existing = {model.key for model in current_models}
        models = []
        for item in payload["hypotheses"]:
            try:
                spec = NeuronHypothesis(
                    **{
                        **item,
                        "g_bounds": tuple(item["g_bounds"]),
                        "half_mv_bounds": tuple(item["half_mv_bounds"]),
                        "tau_ms_bounds": tuple(item["tau_ms_bounds"]),
                    }
                )
                candidate = neuron_candidate(spec)
                if candidate.key not in existing:
                    models.append(candidate)
                    existing.add(candidate.key)
            except (TypeError, ValueError):
                continue
        return models[:count]
