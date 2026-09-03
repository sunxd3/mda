"""Deterministic NeuronBench adapter using the paper's data and proposal contracts.

The released benchmark remains the source of truth for protocols and HH simulation.
This module supplies the missing agent/environment glue without reading its private
truth fields. Stochastic model fitting uses the generic ``BootstrapParticleFilter``
once a domain model supplies ``StateSpaceModel`` callbacks; the released benchmark
does not expose one-step transitions, so this adapter deliberately does not fake them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if self.mechanism_type not in {"channel", "slow_na"}:
            raise ValueError("mechanism_type must be channel or slow_na")
        if self.opens_on not in {"depol", "hyperpol"}:
            raise ValueError("opens_on must be depol or hyperpol")
        for bounds in (self.g_bounds, self.half_mv_bounds, self.tau_ms_bounds):
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError("all neuron parameter bounds must be finite and ordered")
        if self.activation_power < 1 or self.activation_slope_mv <= 0:
            raise ValueError("activation power and slope magnitude must be positive")
        if self.inactivates and (
            self.inactivation_slope_mv <= 0 or self.inactivation_tau_ms <= 0
        ):
            raise ValueError("inactivation kinetics must be positive")


def neuron_candidate(spec: NeuronHypothesis) -> CandidateModel:
    """Compile an LLM-proposed mechanism into the released deterministic HH simulator."""
    nb = _neuronbench()
    if spec.mechanism_type == "slow_na":
        names = ("gL",)
        lower = np.asarray([0.2])
        upper = np.asarray([0.45])
    else:
        names = ("gL", "g", "half_mv", "tau_ms")
        lower = np.asarray([0.2, spec.g_bounds[0], spec.half_mv_bounds[0], spec.tau_ms_bounds[0]])
        upper = np.asarray([0.45, spec.g_bounds[1], spec.half_mv_bounds[1], spec.tau_ms_bounds[1]])
    prior = UniformPrior(names, lower, upper)

    def mechanism(theta: np.ndarray):
        if spec.mechanism_type == "slow_na":
            return {"extra": [], "slow_na": True}
        channel = nb.Chan(
            name=spec.name,
            g=float(theta[1]),
            E=spec.reversal_mv,
            mvh=float(theta[2]),
            mk=(-spec.activation_slope_mv if spec.opens_on == "hyperpol" else spec.activation_slope_mv),
            mtau=float(theta[3]),
            mpow=spec.activation_power,
            hvh=(spec.inactivation_half_mv if spec.inactivates else None),
            hk=spec.inactivation_slope_mv,
            htau=spec.inactivation_tau_ms,
        )
        return {"extra": [channel], "slow_na": False}

    def predict(theta: np.ndarray, protocol) -> np.ndarray:
        _, segments = protocol
        current, start = nb.worlds.build_I(segments)
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
            (spec.mechanism_type,)
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
                spec.inactivation_tau_ms if spec.inactivates else None,
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
        # Table 14's robust per-sample Gaussian kernel.
        return np.maximum(0.12 * np.abs(prediction), 1.0)

    def step(self, protocol) -> np.ndarray:
        return self.world.run(protocol).voltage


class NeuronBenchProposer:
    """Electrophysiology structure proposer based on Appendix H.4.1."""

    def __init__(self, client: OpenAIJSONClient) -> None:
        self.client = client

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
            f"  - {datum.action[0]} -> {trace_features(datum.observation, datum.action).tolist()}"
            for datum in dataset
        )
        tried = "\n".join(
            f"  - {report['name']}: residual={report['median_relative_residual']:.4g}"
            for report in residuals
        ) or "  (none)"
        prompt = f"""{context}

Each experiment reports [n_test, n_pre, run_down, adaptation, V_min, V_end].
Experiments:
{experiments}

Mechanisms already tried (do not repeat):
{tried}

Propose {count} distinct candidate mechanisms beyond the standard Na+/K+ currents.
A mechanism may instead be slow use-dependent Na inactivation. Reversal potential,
activation direction, powers, slopes, and inactivation kinetics define structure;
g, activation half-voltage, and activation tau have uniform-prior bounds."""
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
