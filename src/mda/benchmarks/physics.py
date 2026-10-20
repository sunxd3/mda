"""Public-API reconstruction of the six two-particle ForceBench worlds.

The adapter deliberately consumes only ``get_world``'s executor and mission;
the hidden law, title, rubric, and optimal explanation are never retained.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from mda.agent import AgentConfig, ModelDiscoveryAgent
from mda.expressions import canonical_expression, compile_expression, screen_candidate
from mda.likelihoods import DeterministicGaussianLikelihood
from mda.smc import SMCConfig
from mda.types import CandidateModel, Datum, UniformPrior

MEASUREMENT_TIMES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
WORLD_ALIASES = {
    "gravity": "gravity", "yukawa": "yukawa", "coulomb": "coulomb_easy",
    "coulomb_easy": "coulomb_easy", "oscillator": "oscillator",
    "fractional": "fractional", "extra-dim": "extra_dimensions",
    "extra_dim": "extra_dimensions", "extra_dimensions": "extra_dimensions",
}


def _scienceagent():
    try:
        from scienceagent import evaluator, worlds
    except ImportError as error:
        raise RuntimeError(
            "ForceBench requires ScienceAgent/PhysicsSchool from DiscoverPhysics "
            "commit 33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
        ) from error
    return worlds, evaluator


def _action(radius: float, *, p1: float = 1, p2: float = 1,
            velocity=(0.0, 0.0)) -> dict[str, Any]:
    return {"p1": float(p1), "p2": float(p2), "pos2": [float(radius), 0.0],
            "velocity2": list(map(float, velocity)),
            "measurement_times": list(MEASUREMENT_TIMES)}


def paper_actions() -> list[dict[str, Any]]:
    """The 13 interventions in the paper menu (Table 6)."""
    actions = [_action(r) for r in (1.5, 2, 3, 4, 5, 6, 8, 10)]
    actions += [_action(r, velocity=(0, .4)) for r in (2, 4)]
    actions += [_action(4, p1=2), _action(3, p2=2), _action(4, p1=2, p2=2)]
    return actions


def action_key(action: Mapping[str, Any]) -> str:
    fields = ("p1", "p2", "pos2", "velocity2", "measurement_times", "start_time")
    return json.dumps({key: action[key] for key in fields if key in action}, sort_keys=True)


def force_candidate(name: str, expression: str, parameter_names: Sequence[str],
                    lower: Sequence[float], upper: Sequence[float], *,
                    coulomb_roles: bool = False, dt: float = .005) -> CandidateModel:
    """Compile ``fmag(r, qi, qj, t; theta)`` and integrate with leapfrog.

    Positive force magnitude points from particle 2 toward the pinned particle
    1.  This is the documented ``-Fmag*rhat/m`` convention.
    """
    names = tuple(parameter_names)
    force = compile_expression(expression, ("r", "qi", "qj", "t"), names)
    prior = UniformPrior(names, np.asarray(lower), np.asarray(upper))

    def predict(theta: np.ndarray, action: Mapping[str, Any]) -> np.ndarray:
        times = np.asarray(sorted(action["measurement_times"]), dtype=float)
        if times.size == 0 or np.any(times < 0):
            raise ValueError("measurement_times must be non-negative")
        position = np.asarray(action["pos2"], dtype=float).copy()
        velocity = np.asarray(action["velocity2"], dtype=float).copy()
        if position.shape != (2,) or velocity.shape != (2,):
            raise ValueError("pos2 and velocity2 must be two-vectors")
        qi = float(action["p2"]) if coulomb_roles else 1.0
        qj = float(action["p1"])
        mass = 1.0 if coulomb_roles else float(action["p2"])
        start = float(action.get("start_time", 0.0))

        def acceleration(x, clock):
            radius = float(np.linalg.norm(x))
            if radius <= 0:
                raise ValueError("force is singular at zero separation")
            magnitude = float(force(theta, {"r": radius, "qi": qi, "qj": qj, "t": clock}))
            return -magnitude * x / (radius * mass)

        result, clock = [], 0.0
        for target in times:
            while clock < target - 1e-12:
                h = min(dt, target - clock)
                # Kick-drift-kick (velocity Verlet), a second-order symplectic method.
                velocity += .5 * h * acceleration(position, start + clock)
                position += h * velocity
                clock += h
                velocity += .5 * h * acceleration(position, start + clock)
            result.append(position.copy())
        return np.asarray(result)

    return CandidateModel(name=name, structure=canonical_expression(expression), prior=prior,
                          predict=predict)


class ForceProposer:
    """Structured JSON proposer for central-force magnitudes."""
    def __init__(self, client, *, coulomb_roles: bool = False):
        self.client, self.coulomb_roles = client, coulomb_roles

    def propose(self, *, context: str, dataset: Sequence[Datum],
                current_models: Sequence[CandidateModel],
                residuals: Sequence[dict[str, Any]], count: int) -> list[CandidateModel]:
        structures = [{"name": m.name, "expression": m.structure,
            "priors": dict(zip(m.prior.names,
                zip(m.prior.lower.tolist(), m.prior.upper.tolist()), strict=True))}
            for m in current_models]
        observations = [{"action": d.action, "positions2": d.observation.tolist()} for d in dataset]
        prompt = f"""{context}
Infer a central force magnitude fmag(r, qi, qj, t; theta). Positive magnitude is
attraction and acceleration is -fmag*rhat/m. qi is the probe charge and qj the
source coupling. Propose {count} parsimonious forms.
Approved functions are abs, cos, exp, gamma, k0, k1, log, log10, maximum,
minimum, sign, sin, sqrt, tan, tanh. Unknown constants need finite
uniform priors. Existing structures and priors: {json.dumps(structures)}
Residual reports: {json.dumps(list(residuals))}
Trajectory observations: {json.dumps(observations)}"""
        param = {"type": "object", "properties": {"name": {"type": "string"},
            "low": {"type": "number"}, "high": {"type": "number"}},
            "required": ["name", "low", "high"], "additionalProperties": False}
        item = {"type": "object", "properties": {"name": {"type": "string"},
            "expr": {"type": "string"}, "params": {"type": "array", "items": param}},
            "required": ["name", "expr", "params"], "additionalProperties": False}
        schema = {"type": "object", "properties": {"hypotheses": {
            "type": "array", "items": item}}, "required": ["hypotheses"],
            "additionalProperties": False}
        payload = self.client.ask_json("Propose physical force laws. JSON only.", prompt,
                                       schema=schema, name="force_hypotheses",
                                       max_tokens=max(1600, 400 * count))
        existing, models = {m.key for m in current_models}, []
        for spec in payload.get("hypotheses", []):
            try:
                params = spec["params"]
                model = force_candidate(spec["name"].strip(), spec["expr"].strip(),
                    [p["name"] for p in params], [p["low"] for p in params],
                    [p["high"] for p in params], coulomb_roles=self.coulomb_roles)
                # Compile and execute against real action structure before admitting it.
                if model.key not in existing and screen_candidate(model, dataset, actions=1):
                    models.append(model); existing.add(model.key)
            except (KeyError, TypeError, ValueError, RuntimeError, ArithmeticError):
                continue
        return models[:count]


class PhysicsEnvironment:
    def __init__(self, world_name: str, seed: int = 0):
        canonical = WORLD_ALIASES.get(world_name)
        if canonical is None:
            raise ValueError(f"world must be one of {sorted(WORLD_ALIASES)}")
        worlds, evaluator_module = _scienceagent()
        public = worlds.get_world(canonical, engine="nbody", noise_seed=seed, noise_std=.05)
        self.executor = public["executor"]
        self.world_name = canonical
        self.context = public["mission"]
        initial = _action(3)
        self.initial_data = [Datum(initial, np.asarray(self.executor.run([initial])[0]["pos2"]))]
        initial_key = action_key(initial)
        self.actions = [a for a in paper_actions() if action_key(a) != initial_key]
        # Use the release's Evaluator defaults when exposed, and remove any observed IC.
        defaults = evaluator_module.Evaluator(self.executor).test_cases
        observed = {action_key(d.action) for d in self.initial_data}
        self.query_actions = [deepcopy(a) for a in defaults if action_key(a) not in observed]

    def step(self, action) -> np.ndarray:
        return np.asarray(self.executor.run([action])[0]["pos2"], dtype=float)

    def evaluate_predictor(self, predictor) -> float:
        with self.executor.noise_disabled():
            truth = np.asarray([self.executor.run([a])[0]["pos2"] for a in self.query_actions])
        forecast = np.asarray([predictor(a) for a in self.query_actions], dtype=float)
        if forecast.shape != truth.shape:
            raise ValueError("predictor trajectories must match query trajectory shape")
        variance = float(np.var(truth))
        if variance <= 0:
            raise ValueError("normalized trajectory MSE is undefined for zero variance")
        return float(np.mean((forecast - truth) ** 2) / variance)

    def evaluate_model(self, model: CandidateModel, parameters=None) -> dict[str, Any]:
        """A2 radial-force recovery, measured only through clean public launches.

        The paper does not release its grid or finite-difference step. We use
        interleaved log-radius grids over [1.5, 10] and a .01 time-unit drop.
        Unit source/probe knobs and t=0 test the radial profile, not full
        charge/time dependence. Signed log1p preserves repulsive predictions.
        """
        del parameters  # This diagnostic refits the free constants.
        from .recovery import canonical_symbolic_candidate, evaluate_recovery

        canonical = canonical_symbolic_candidate(
            model.structure, ("r", "qi", "qj", "t"), model.prior)
        radii = np.geomspace(1.5, 10., 128)
        h = .01
        launches = [_action(radius) for radius in radii]
        for action in launches:
            action["measurement_times"] = [h]
        with self.executor.noise_disabled():
            results = self.executor.run(launches)
        # x(h) = r - F(r) h²/2 + O(h⁴), for a unit probe at rest.
        force = np.asarray([2 * (radius - result["pos2"][0][0]) / h**2
                            for radius, result in zip(radii, results, strict=True)])
        grids = [{"r": float(r), "qi": 1., "qj": 1., "t": 0.} for r in radii]
        parameters, rmsle, recovered = evaluate_recovery(
            canonical, (grids[::2], grids[1::2]), (force[::2], force[1::2]),
            tolerance=.05, signed_log1p=True)
        return {"parameters": parameters, "rmsle": rmsle, "recovered": recovered,
                "canonical_structure": canonical.structure,
                "scope": "unit-coupling radial profile at t=0",
                "transform": "signed_log1p", "probe_duration": h,
                "radius_range": [1.5, 10.], "grid_points": len(radii)}


def _generic_models(coulomb_roles: bool) -> list[CandidateModel]:
    # Identical hypothesis menu for every world: no hidden-law dispatch.
    specs = [("inverse", "a*qi*qj/r", ("a",), (.01,), (2.,)),
             ("power", "a*qi*qj/r**b", ("a", "b"), (.01, .2), (2., 3.5)),
             ("screened", "a*qi*qj*exp(-r/l)/r", ("a", "l"), (.01, .2), (2., 8.))]
    return [force_candidate(n, e, p, lo, hi, coulomb_roles=coulomb_roles)
            for n, e, p, lo, hi in specs]


def make_agent(environment: PhysicsEnvironment, client, *, seed: int = 0,
               smoke: bool = False, experiment_budget: int = 8) -> ModelDiscoveryAgent:
    if client is None and not smoke:
        raise ValueError("client=None is supported only for smoke=True")
    coulomb = environment.world_name == "coulomb_easy"
    proposer = ForceProposer(client, coulomb_roles=coulomb) if client is not None else None
    models = _generic_models(coulomb) if smoke else []
    config = AgentConfig(model_cap=24, new_models_per_expansion=6,
        model_search_rounds=2, expansion_cap=8, error_threshold=1.4,
        occam_penalty=2.5, predictive_floor=.05, design_flavour="model",
        smc=SMCConfig(parameter_particles=3 if smoke else 60,
                      rejuvenation_steps=0 if smoke else 3,
                      target_ess_fraction=.6, max_temperatures=3 if smoke else 80))
    return ModelDiscoveryAgent(context=environment.context,
        initial_data=environment.initial_data, initial_models=models, proposer=proposer,
        likelihood=DeterministicGaussianLikelihood(.05), actions=environment.actions,
        observation_covariance=.05**2, config=config,
        query_actions=environment.query_actions, experiment_budget=experiment_budget,
        action_key=action_key, seed=seed)
