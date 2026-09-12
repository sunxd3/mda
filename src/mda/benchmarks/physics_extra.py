"""Public-API MDA adapters for ForceBench's circle, ether, and Hubble worlds.

The launch menu below is a local, reproducible six-design menu: the paper's
exact launch values were not released.  Hidden world descriptions and
executor attributes are deliberately never read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from mda.agent import AgentConfig, ModelDiscoveryAgent
from mda.expressions import compile_expression
from mda.likelihoods import DeterministicGaussianLikelihood
from mda.smc import SMCConfig
from mda.types import CandidateModel, Datum, UniformPrior

WORLDS = ("circle", "ether", "hubble")
TIMES = [0.5, 1.0, 2.0, 3.0, 4.0]


def _scienceagent():
    try:
        from scienceagent import evaluator, worlds
    except ImportError as error:
        raise RuntimeError("physics extras require DiscoverPhysics commit "
                           "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5") from error
    return worlds, evaluator


def _circle_action(radius: float, speed: float) -> dict[str, Any]:
    return {"ring_radius": radius, "initial_tangential_velocity": speed,
            "measurement_times": list(TIMES)}


def _probe_action(radius: float, speed: float = 0.0,
                  masses: Sequence[float] | None = None) -> dict[str, Any]:
    positions = [[radius, 0], [0, radius], [-radius, 0], [0, -radius],
                 [radius / np.sqrt(2), radius / np.sqrt(2)]]
    velocities = [[0, speed], [-speed, 0], [0, -speed], [speed, 0],
                  [-speed / np.sqrt(2), speed / np.sqrt(2)]]
    result: dict[str, Any] = {"probe_positions": positions,
                             "probe_velocities": velocities,
                             "measurement_times": list(TIMES)}
    if masses is not None:
        result["probe_masses"] = list(masses)
    return result


def paper_actions(world_name: str) -> list[dict[str, Any]]:
    """Six local design templates (exact paper launch values are unreleased)."""
    if world_name == "circle":
        return [_circle_action(r, v) for r, v in
                ((3, 0), (5, 0), (8, 0), (3, .3), (5, .5), (8, .8))]
    mixed = [1., 2., 4., 1., 2.]
    return [_probe_action(r, v, mixed if i % 2 else None)
            for i, (r, v) in enumerate(((3, 0), (6, 0), (10, 0), (15, 0),
                                         (6, 1.5), (10, 2.0)))]


def action_key(action: Mapping[str, Any]) -> str:
    # Public reconstruction metadata is intentionally excluded from identity.
    return json.dumps({k: action[k] for k in ("ring_radius",
        "initial_tangential_velocity", "probe_positions", "probe_velocities",
        "probe_masses", "measurement_times") if k in action}, sort_keys=True)


def _initial_state(action: Mapping[str, Any], world: str):
    if world == "circle":
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        r, v = float(action["ring_radius"]), float(action["initial_tangential_velocity"])
        pos = np.vstack(([0., 0.], np.column_stack((r*np.cos(angles), r*np.sin(angles)))))
        vel = np.vstack(([0., 0.], np.column_stack((-v*np.sin(angles), v*np.cos(angles)))))
        masses = np.ones(11)
    else:
        pos = np.vstack((action["background_initial_positions"], action["probe_positions"]))
        vel = np.vstack((action["background_initial_velocities"], action["probe_velocities"]))
        masses = np.asarray(action["particle_masses"], dtype=float)
    return np.asarray(pos, float), np.asarray(vel, float), masses


def force_candidate(name: str, expression: str, parameter_names: Sequence[str],
                    lower: Sequence[float], upper: Sequence[float], *,
                    world: str, dt: float = .005) -> CandidateModel:
    """Compile f(r,qi,qj,t) and integrate asymmetric all-pair forces.

    ``qi`` is response weight and ``qj`` source weight; inertia remains a
    separate mass. Ether/Hubble append a fitted external-acceleration nuisance
    coefficient (local broad bounds [-1,1] when the proposer did not specify
    one), rather than embedding the world's true value.
    """
    names, lo, hi = list(parameter_names), list(lower), list(upper)
    external = "alpha" if world == "ether" else "H" if world == "hubble" else None
    if external and external not in names:
        names.append(external); lo.append(-1.0); hi.append(1.0)
    force_names = tuple(n for n in names if n != external)
    force_indices = [names.index(n) for n in force_names]
    force = compile_expression(expression, ("r", "qi", "qj", "t"), force_names)
    prior = UniformPrior(tuple(names), np.asarray(lo), np.asarray(hi))

    def predict(theta: np.ndarray, action: Mapping[str, Any]) -> np.ndarray:
        pos, vel, masses = _initial_state(action, world)
        times = np.asarray(sorted(action["measurement_times"]), float)
        if np.any(times < 0): raise ValueError("measurement times must be non-negative")
        n = len(pos)
        if world == "circle":
            source = response = np.ones(n); fixed = np.zeros(n, bool)
        else:
            # Public roles: particle 0 is the only source; all other particles
            # are source-free test bodies whose response weight equals mass.
            source = np.zeros(n); source[0] = 1.
            response = masses.copy(); response[0] = 0.
            # In the public Ether executor the external field accelerates even
            # the anchor. Hubble's anchor remains at zero without a special mask.
            fixed = np.zeros(n, bool)
        ftheta = np.asarray(theta)[force_indices]
        ext = float(theta[names.index(external)]) if external else 0.

        def acceleration(x, clock):
            delta = x[:, None, :] - x[None, :, :]       # i - j
            radius = np.linalg.norm(delta, axis=2)
            safe = np.where(np.eye(n, dtype=bool), 1., radius)
            mag = np.asarray(force(ftheta, {"r": safe,
                "qi": response[:, None], "qj": source[None, :], "t": clock}), float)
            mag = np.broadcast_to(mag, (n, n)).copy(); np.fill_diagonal(mag, 0.)
            mag[:, source == 0] = 0.
            mag[response == 0, :] = 0.
            acc = -np.sum(mag[..., None] * delta / safe[..., None], axis=1) / masses[:, None]
            if world == "ether": acc[:, 1] += ext
            elif world == "hubble": acc += ext * x
            acc[fixed] = 0.
            return acc

        output, clock = [], 0.
        for target in times:
            while clock < target - 1e-12:
                h = min(dt, target-clock)
                vel += .5*h*acceleration(pos, clock); vel[fixed] = 0.
                pos += h*vel; clock += h
                vel += .5*h*acceleration(pos, clock); vel[fixed] = 0.
            output.append(pos.copy())
        trajectory = np.asarray(output)
        return trajectory if world == "circle" else trajectory[:, 21:26]

    return CandidateModel(name, prior, predict, structure=expression,
                          metadata={"world": world})


class ForceProposer:
    def __init__(self, client, world: str): self.client, self.world = client, world

    def propose(self, *, context: str, dataset: Sequence[Datum],
                current_models: Sequence[CandidateModel], residuals, count: int):
        observations = [{"action": d.action, "positions": d.observation.tolist()} for d in dataset]
        models_with_priors = [{"structure": m.structure, "priors": dict(zip(m.prior.names,
            zip(m.prior.lower.tolist(), m.prior.upper.tolist())))} for m in current_models]
        prompt = f"""{context}\nPropose {count} parsimonious central all-pair force
magnitudes f(r,qi,qj,t). Positive means attraction. qi is response weight and
qj source weight. Return finite LLM-declared uniform bounds for every unknown.
The external acceleration coefficient is fitted separately, not part of the central expression.
Observed experiments: {json.dumps(observations)}
Existing structures and priors: {json.dumps(models_with_priors)}; residuals: {list(residuals)}"""
        parameter = {"type":"object", "properties":{"name":{"type":"string"},
            "low":{"type":"number"}, "high":{"type":"number"}},
            "required":["name","low","high"], "additionalProperties":False}
        item = {"type":"object", "properties":{"name":{"type":"string"},
            "expr":{"type":"string"}, "params":{"type":"array","items":parameter}},
            "required":["name","expr","params"], "additionalProperties":False}
        schema = {"type":"object", "properties":{"hypotheses":{"type":"array","items":item}},
                  "required":["hypotheses"], "additionalProperties":False}
        payload = self.client.ask_json("Propose physical laws. JSON only.", prompt,
            schema=schema, name="force_hypotheses", max_tokens=max(1600, 400*count))
        models, existing = [], {m.key for m in current_models}
        for spec in payload.get("hypotheses", []):
            try:
                p = spec["params"]
                model = force_candidate(spec["name"], spec["expr"], [x["name"] for x in p],
                    [x["low"] for x in p], [x["high"] for x in p], world=self.world)
                if model.key not in existing:
                    model.observation_prediction(model.prior.sample(np.random.default_rng(0),1)[0], dataset[0].action)
                    models.append(model); existing.add(model.key)
            except (KeyError, TypeError, ValueError, FloatingPointError): pass
        return models[:count]


class PhysicsExtraEnvironment:
    def __init__(self, world_name: str, seed: int = 0):
        if world_name not in WORLDS: raise ValueError(f"world must be one of {WORLDS}")
        worlds, evaluators = _scienceagent()
        public = worlds.get_world(world_name, engine="nbody", noise_seed=seed, noise_std=.03)
        self.executor, self.world_name, self.context = public["executor"], world_name, public["mission"]
        self.context += {
            "circle": " Eleven equal particles interact through the same unknown central pair force.",
            "ether": " One anchor sources an unknown central force; source-free probes have response weight equal to mass. A uniform +y acceleration alpha acts on every body, including the anchor.",
            "hubble": " One anchor sources an unknown central force; source-free probes have response weight equal to mass. An additional acceleration H*position acts on every body.",
        }[world_name]
        menu = paper_actions(world_name)
        first = self._run_public(menu[0])
        self.initial_data = [Datum(first[0], first[1])]
        self.actions = [self._attach_background(a, first[0]) for a in menu[1:]]
        evaluator_cls = {"circle": evaluators.CircleEvaluator,
                         "ether": evaluators.EtherEvaluator,
                         "hubble": evaluators.HubbleEvaluator}[world_name]
        defaults = evaluator_cls(self.executor).test_cases
        observed = {action_key(d.action) for d in self.initial_data}
        self.query_actions = [self._attach_background(deepcopy(a), first[0]) for a in defaults
                              if action_key(a) not in observed]

    def _attach_background(self, action, reference):
        action = deepcopy(action)
        for key in ("background_initial_positions", "background_initial_velocities"):
            if key in reference: action[key] = deepcopy(reference[key])
        if self.world_name != "circle":
            bgm = list(reference["particle_masses"][:21])
            action["particle_masses"] = bgm + list(action.get("probe_masses", [1.]*5))
        return action

    def _run_public(self, action):
        result = self.executor.run([action])[0]
        enriched = self._attach_background(action, result)
        observation = np.asarray(result["positions"], float)
        if self.world_name != "circle": observation = observation[:, 21:26]
        return enriched, observation

    def step(self, action): return self._run_public(action)[1]

    def evaluate_predictor(self, predictor):
        with self.executor.noise_disabled():
            truths = [self._run_public(a)[1] for a in self.query_actions]
        predictions = [np.asarray(predictor(a), float) for a in self.query_actions]
        if any(p.shape != t.shape for p, t in zip(predictions, truths)):
            raise ValueError("predictor trajectories must match scored trajectory shape")
        truth = np.concatenate([t.ravel() for t in truths])
        forecast = np.concatenate([p.ravel() for p in predictions])
        variance = float(np.var(truth))
        if variance <= 0:
            raise ValueError("normalized trajectory MSE requires positive query variance")
        return float(np.mean((forecast-truth)**2) / variance)


def _generic_models(world):
    # Same law-agnostic smoke seeds in all worlds; external terms are nuisances.
    return [force_candidate("inverse", "a*qi*qj/r", ("a",), (.001,), (20.,), world=world),
            force_candidate("power", "a*qi*qj/r**b", ("a","b"), (.001,.1),
                            (20.,4.), world=world)]


def make_agent(environment: PhysicsExtraEnvironment, client, *, seed: int = 0,
               smoke: bool = False, experiment_budget: int = 8) -> ModelDiscoveryAgent:
    if client is None and not smoke: raise ValueError("client=None requires smoke=True")
    models = _generic_models(environment.world_name) if smoke else []
    config = AgentConfig(model_cap=24, new_models_per_expansion=6, model_search_rounds=2,
        expansion_cap=8, error_threshold=1.4, occam_penalty=2.5,
        predictive_floor=.03, design_flavour="model",
        smc=SMCConfig(parameter_particles=3 if smoke else 60,
            rejuvenation_steps=0 if smoke else 3, target_ess_fraction=.6,
            max_temperatures=3 if smoke else 80))
    proposer = ForceProposer(client, environment.world_name) if client else None
    return ModelDiscoveryAgent(context=environment.context, initial_data=environment.initial_data,
        initial_models=models, proposer=proposer,
        likelihood=DeterministicGaussianLikelihood(.03), actions=environment.actions,
        observation_covariance=.03**2, config=config, query_actions=environment.query_actions,
        experiment_budget=experiment_budget, action_key=action_key, seed=seed)
