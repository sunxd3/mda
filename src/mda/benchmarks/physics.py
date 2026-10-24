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
from mda.expressions import (
    canonical_expression, compile_expression, compile_expressions, screen_candidate,
)
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
    # Validates the expression and its names; the batched evaluator below runs it.
    compile_expression(expression, ("r", "qi", "qj", "t"), names)
    evaluate = compile_expressions([expression], ("r", "qi", "qj", "t", *names))
    prior = UniformPrior(names, np.asarray(lower), np.asarray(upper))

    def integrate(thetas: np.ndarray, actions: Sequence[Mapping[str, Any]],
                  times: np.ndarray, start: float) -> np.ndarray:
        """Every (action, parameter row) pair in one array: shape (A, N, T, 2).

        The actions share sorted measurement times and start time. Rows that
        blow up become NaN.
        """
        count, launches = len(thetas), len(actions)
        position = np.repeat(np.asarray([a["pos2"] for a in actions], dtype=float), count, axis=0)
        velocity = np.repeat(np.asarray([a["velocity2"] for a in actions], dtype=float), count, axis=0)
        if position.shape != (launches * count, 2) or velocity.shape != position.shape:
            raise ValueError("pos2 and velocity2 must be two-vectors")
        p1 = np.repeat(np.asarray([float(a["p1"]) for a in actions]), count)
        p2 = np.repeat(np.asarray([float(a["p2"]) for a in actions]), count)
        qi, qj = (p2 if coulomb_roles else 1.0), p1
        mass = 1.0 if coulomb_roles else p2
        parameters = dict(zip(names, np.tile(thetas, (launches, 1)).T, strict=True))
        rows = launches * count

        def acceleration(x, clock):
            radius = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
            magnitude = np.broadcast_to(
                evaluate({**parameters, "r": radius, "qi": qi, "qj": qj, "t": clock})[0], (rows,))
            return -(magnitude / (radius * mass))[:, None] * x

        result, clock = [], 0.0
        with np.errstate(all="ignore"):  # a singular or overflowing row becomes NaN
            for target in times:
                while clock < target - 1e-12:
                    h = min(dt, target - clock)
                    # Kick-drift-kick (velocity Verlet), a second-order symplectic method.
                    velocity += .5 * h * acceleration(position, start + clock)
                    position += h * velocity
                    clock += h
                    velocity += .5 * h * acceleration(position, start + clock)
                result.append(position.copy())
        return np.stack(result, axis=1).reshape(launches, count, len(times), 2)

    def predict_grid(thetas: np.ndarray, actions: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
        """Trajectories of every parameter row under every action, one (N, T, 2) array per action."""
        thetas = np.asarray(thetas, dtype=float).reshape(-1, len(names))
        groups: dict[tuple, list[int]] = {}
        for index, action in enumerate(actions):
            times = tuple(sorted(float(t) for t in action["measurement_times"]))
            if not times or times[0] < 0:
                raise ValueError("measurement_times must be non-negative")
            groups.setdefault((times, float(action.get("start_time", 0.0))), []).append(index)
        results: list[Any] = [None] * len(actions)
        for (times, start), indices in groups.items():
            block = integrate(thetas, [actions[i] for i in indices], np.asarray(times), start)
            for i, trajectories in zip(indices, block, strict=True):
                results[i] = trajectories
        return results

    def predict_many(thetas: np.ndarray, action: Mapping[str, Any]) -> np.ndarray:
        return predict_grid(thetas, [action])[0]

    def predict(theta: np.ndarray, action: Mapping[str, Any]) -> np.ndarray:
        return predict_many(np.asarray(theta, dtype=float)[None, :], action)[0]

    return CandidateModel(name=name, structure=canonical_expression(expression), prior=prior,
                          predict=predict, metadata={"predict_many": predict_many, "predict_grid": predict_grid})


class ForceProposer:
    """Structured JSON proposer for central-force magnitudes."""
    def __init__(self, client, *, coulomb_roles: bool = False, guided: bool = False,
                 max_parameters: int | None = None):
        """``guided`` adds the current best fit and where it fails to expansion
        prompts (``self.agent`` must then be set); ``max_parameters`` drops laws
        with more free parameters."""
        self.client, self.coulomb_roles = client, coulomb_roles
        self.guided, self.max_parameters = guided, max_parameters
        self.agent = None

    def _guidance(self, dataset: Sequence[Datum]) -> str:
        """The MAP law, its per-experiment misfit, and the latest trajectory against it."""
        fit = self.agent.belief.map_fit
        theta = fit.parameters.map_parameters
        lines = [f"Current best law: F_mag = {fit.model.structure} with "
                 + ", ".join(f"{n}={v:.4g}" for n, v in zip(fit.model.prior.names, theta, strict=True)) + "."]
        lines.append("Its position error in units of the measurement noise (sigma=0.05), per experiment in order:")
        misfits = []
        for datum in dataset:
            try:
                predicted = fit.model.observation_prediction(theta, datum.action)
                misfits.append(f"{np.sqrt(np.mean(((datum.observation - predicted) / .05) ** 2)):.3g}")
            except (ValueError, ArithmeticError):
                misfits.append("diverges")
        lines.append(", ".join(misfits))
        latest = dataset[-1]
        a = latest.action
        try:
            predicted = fit.model.observation_prediction(theta, a)
            predicted_text = str([[round(float(v), 3) for v in row] for row in predicted])
        except (ValueError, ArithmeticError):
            predicted_text = "diverges"
        lines.append(f"Latest experiment (pos2={a['pos2']} velocity2={a['velocity2']} p1={a['p1']} p2={a['p2']}), "
                     f"probe positions at t={list(sorted(a['measurement_times']))}:")
        lines.append(f"observed  {[[round(float(v), 3) for v in row] for row in latest.observation]}")
        lines.append(f"predicted {predicted_text}")
        lines.append("A misfit far above 1 means the current laws are wrong where that experiment probes. "
                     "Propose structurally different laws that explain ALL experiments, not small "
                     "corrections to the current best law.")
        return "\n".join(lines) + "\n"

    def propose(self, *, context: str, dataset: Sequence[Datum],
                current_models: Sequence[CandidateModel],
                residuals: Sequence[dict[str, Any]], count: int) -> list[CandidateModel]:
        del context, residuals  # v4's template states the world itself and sends no residuals.
        knobs = ("p1 is the source charge (qj) and p2 the probe charge (qi); the probe has unit inertia."
                 if self.coulomb_roles else
                 "p1 is the source coupling (qj) and p2 the probe inertia (mass); the probe charge qi is 1.")
        runs = []
        for datum in dataset:
            a = datum.action
            radius = np.linalg.norm(np.asarray(datum.observation, dtype=float), axis=-1)
            runs.append(f"launch pos2={a['pos2']} velocity2={a['velocity2']} p1={a['p1']} p2={a['p2']}: "
                        f"t={list(sorted(a['measurement_times']))} "
                        f"r(t)={[round(float(r), 4) for r in radius]}")
        tried = "\n".join(f"{m.name}: F_mag = {m.structure}" for m in current_models)
        if self.guided and current_models and self.agent is not None:
            tried += "\n\n" + self._guidance(dataset)
        limit = (f" Use at most {self.max_parameters} free parameters per law."
                 if self.max_parameters is not None and current_models else "")
        # v4's published DiscoverPhysics proposer prompt (dp_proposer.txt); the
        # forms already in the pool fill its "(or refinements of those tried)".
        prompt = f"""A test probe moves in an unknown central force sourced by a fixed body at the origin. The force MAY be static or MAY vary with time t (e.g. a time-modulated coupling). Each experiment launches the probe from a position with a velocity, and sets two knobs p1, p2. {knobs} F_mag is the pairwise force magnitude between the probe (charge qi) and source (charge qj) at separation r and time t.

Observed data:
{chr(10).join(runs)}
{f"{chr(10)}Forms already tried:{chr(10)}{tried}{chr(10)}" if tried else ""}
Propose {count} distinct plausible force laws (or refinements of those tried).{limit}

Express each F_mag as a Python expression in the symbols r, qi, qj, t and your OWN named free parameters ONLY. The source coupling is carried by qj (the probe is qi); do NOT reference p1 or p2 in the expression -- introduce named parameters (e.g. k, lam, s) for coupling constants and length scales. Allowed functions: exp, log, sqrt, sin, cos, tanh, k0, k1, gamma, pi, np. Return JSON {{"hypotheses": [{{"name": str, "fmag": str, "operator": str, "params": [{{"name": str, "low": float, "high": float}}], "rationale": str}}]}}."""
        param = {"type": "object", "properties": {"name": {"type": "string"},
            "low": {"type": "number"}, "high": {"type": "number"}},
            "required": ["name", "low", "high"], "additionalProperties": False}
        item = {"type": "object", "properties": {"name": {"type": "string"},
            "fmag": {"type": "string"}, "operator": {"type": "string"},
            "params": {"type": "array", "items": param}, "rationale": {"type": "string"}},
            "required": ["name", "fmag", "operator", "params", "rationale"],
            "additionalProperties": False}
        schema = {"type": "object", "properties": {"hypotheses": {
            "type": "array", "items": item}}, "required": ["hypotheses"],
            "additionalProperties": False}
        payload = self.client.ask_json(
            "You are a physicist proposing candidate pairwise force laws to explain probe-orbit data. "
            "Reply JSON only.", prompt, schema=schema, name="force_hypotheses",
            max_tokens=max(1600, 400 * count))
        existing, models = {m.key for m in current_models}, []
        for spec in payload.get("hypotheses", []):
            try:
                params = spec["params"]
                if self.max_parameters is not None and len(params) > self.max_parameters:
                    continue
                # The template allows ``np``; the expression grammar takes bare names.
                expression = spec["fmag"].strip().replace("np.", "")
                model = force_candidate(spec["name"].strip(), expression,
                    [p["name"] for p in params], [p["low"] for p in params],
                    [p["high"] for p in params], coulomb_roles=self.coulomb_roles)
                # Compile and execute against real action structure before admitting it.
                if model.key not in existing and screen_candidate(model, dataset, actions=1):
                    models.append(model); existing.add(model.key)
            except (KeyError, TypeError, ValueError, RuntimeError, ArithmeticError):
                continue
        return models[:count]


class PhysicsEnvironment:
    # The physics writeup scores a failed or non-finite prediction as nMSE 1e3.
    failed_score = 1e3

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


# The continuous design box of the v5-era physics writeup (its §A.2), which is
# DiscoverPhysics' documented control range: position, velocity, then p1, p2.
CONTINUOUS_LOWER = (-10., -10., -5., -5., .1, .1)
CONTINUOUS_UPPER = (10., 10., 5., 5., 10., 10.)
MINIMUM_LAUNCH_RADIUS = .5


def continuous_action(vector: Sequence[float]) -> dict[str, Any]:
    """Decode a box point into a launch, moving radii below 0.5 out to 0.5."""
    x, y, vx, vy, p1, p2 = map(float, vector)
    radius = float(np.hypot(x, y))
    if radius < MINIMUM_LAUNCH_RADIUS:
        x, y = ((MINIMUM_LAUNCH_RADIUS, 0.) if radius == 0 else
                (x * MINIMUM_LAUNCH_RADIUS / radius, y * MINIMUM_LAUNCH_RADIUS / radius))
    return {"p1": p1, "p2": p2, "pos2": [x, y], "velocity2": [vx, vy],
            "measurement_times": list(MEASUREMENT_TIMES)}


def continuous_optimizer(max_evaluations: int = 144):
    """Differential evolution over the box; p1 and p2 are searched on log scales."""
    from mda.optimizers import DifferentialEvolutionOptimizer
    return DifferentialEvolutionOptimizer(
        CONTINUOUS_LOWER, CONTINUOUS_UPPER, log_axes=np.array([False] * 4 + [True] * 2),
        decode=continuous_action, max_evaluations=max_evaluations)


def fresh_chi2_check(belief, fresh: Datum, history: Sequence[Datum], *,
                     sigma: float = .05, level: float = .99) -> float:
    """Noise-calibrated adequacy of one fresh trajectory; above 1 means expand.

    The MAP fit's whitened squared error on the fresh datum, divided by the
    ``level`` quantile of chi-squared with one degree of freedom per
    coordinate. Noise alone exceeds 1 with probability ``1 - level``. The
    history is ignored, so a single surprising experiment can trigger expansion.
    """
    del history
    from scipy.stats import chi2

    fit = belief.map_fit
    try:
        predicted = fit.model.observation_prediction(fit.parameters.map_parameters, fresh.action)
    except (ValueError, ArithmeticError):
        return np.inf
    residual = (np.asarray(fresh.observation, dtype=float) - predicted) / sigma
    return float(np.sum(residual**2) / chi2.ppf(level, residual.size))


def make_agent(environment: PhysicsEnvironment, client, *, seed: int = 0,
               smoke: bool = False, experiment_budget: int = 8,
               design_space: str = "menu", check: str = "v4",
               max_parameters: int | None = None) -> ModelDiscoveryAgent:
    """v4's ForceBench agent on the 13-action menu or the continuous box.

    ``check="v4"`` is v4's predictive check: expand when the median relative
    residual over the fresh and stored trajectories exceeds 1.4, gating the
    first expansion on cached residuals as v5 says the executed DP agent did.
    ``check="chi2"`` expands when the fresh trajectory alone fails
    :func:`fresh_chi2_check`. ``check="guided"`` uses the same test but spends
    at most one expansion per experiment (up to 13), and shows the proposer
    the current best law and where it fails. ``max_parameters`` drops
    proposed laws with more free parameters (v4 states 1-3 per candidate).
    """
    if design_space not in {"menu", "continuous"}:
        raise ValueError("design_space must be menu or continuous")
    if check not in {"v4", "chi2", "guided"}:
        raise ValueError("check must be v4, chi2 or guided")
    if client is None and not smoke:
        raise ValueError("client=None is supported only for smoke=True")
    coulomb = environment.world_name == "coulomb_easy"
    proposer = (ForceProposer(client, coulomb_roles=coulomb, guided=check == "guided",
                              max_parameters=max_parameters) if client is not None else None)
    models = _generic_models(coulomb) if smoke else []
    config = AgentConfig(model_cap=24, new_models_per_expansion=6,
        model_search_rounds=1 if check == "guided" else 2,
        expansion_cap=13 if check == "guided" else 8, error_threshold=1.4 if check == "v4" else 1.0,
        occam_penalty=2.5, predictive_floor=.05, design_flavour="model",
        expansion_gate="cached" if check == "v4" else "prequential",
        smc=SMCConfig(parameter_particles=3 if smoke else 60,
                      rejuvenation_steps=0 if smoke else 3,
                      target_ess_fraction=.6, max_temperatures=3 if smoke else 80))
    agent = ModelDiscoveryAgent(context=environment.context,
        initial_data=environment.initial_data, initial_models=models, proposer=proposer,
        likelihood=DeterministicGaussianLikelihood(.05),
        actions=environment.actions if design_space == "menu" else [],
        optimizer=continuous_optimizer() if design_space == "continuous" else None,
        observation_covariance=.05**2, config=config,
        query_actions=environment.query_actions, experiment_budget=experiment_budget,
        action_key=action_key, check=None if check == "v4" else fresh_chi2_check, seed=seed)
    if proposer is not None:
        proposer.agent = agent
    return agent
