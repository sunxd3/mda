"""GlucoseBench adapter and the paper's v5 glucose MDA configuration (App. C).

The public ``glucosebench`` package (pinned at ``6bb0ff6``; install the
``glucose`` extra) owns the simulator, the 48-template menu, the passive D0,
the sealed 12-query test set, prefix payloads, and the training-variance nMSE.
Everything on the learner side is implemented here:

- reduced compartmental ODEs in a restricted expression language (states,
  right-hand sides, initial states, CGM and insulin outputs), restarted at
  minute 45 and integrated with one-minute RK4 steps under the delivered
  per-minute meal (g/min) and bolus (U/min above basal) inputs;
- a working likelihood with AR(1) CGM residuals, whose innovation variance is
  profiled out, and independent insulin assays (SD 10 pmol/L), fitted by
  multistart L-BFGS-B, with BIC counting the ODE parameters plus the AR(1)
  correlation and scale and using the number of observed targets as N;
- the noise-standardized model-disagreement design score (Eq.
  glucose-model-disagreement) over unused menu templates, ties broken randomly;
- EKF conditioning on the query prefix and a BIC-weighted mixture forecast
  (Eq. glucose_full_predictive at point parameters);
- the v5 proposer prompt, at most two models per call, and the archive's
  expansion gate.

v5 leaves several choices open, marked "local choice" below: the equations and
bounds of three of its four seed models, multistart counts, the AR(1)
correlation bounds, the EKF initial covariance, finite-difference Jacobians,
and the nominal delivery schedule used to score not-yet-run designs. The
expansion gate follows the statistic in the paper's source (Eq.
glucose_expansion_gate), which the rendered paper leaves out.
"""

from __future__ import annotations

import json
import keyword
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mda.agent import AgentConfig, ModelDiscoveryAgent
from mda.expressions import _FUNCTIONS, canonical_expression, compile_expressions
from mda.posterior import BeliefState
from mda.smc import ParameterPosterior
from mda.types import CandidateModel, Datum, UniformPrior

CGM_TIMES = np.arange(0, 361, 5)        # 73 readings, including t=0
INSULIN_TIMES = np.arange(0, 361, 15)   # 25 assays
PREFIX_MIN = 45
ASSAY_SD = 10.0
BUDGETS = (2, 3, 4, 6)
_CGM_FIT = CGM_TIMES >= PREFIX_MIN       # 64 readings from minute 45
_INSULIN_FIT = INSULIN_TIMES >= PREFIX_MIN  # 22 assays from minute 45
_CGM_PREFIX = CGM_TIMES <= PREFIX_MIN
_INSULIN_PREFIX = INSULIN_TIMES <= PREFIX_MIN
GLUCOSE_FUNCTIONS = ("exp", "log", "sqrt", "tanh", "abs", "minimum", "maximum")
CONSTANT_NAMES = ("gb", "ib")
INPUT_NAMES = ("meal", "insulin")
MAX_STATES, MAX_PARAMETERS = 6, 12
RHO_BOUNDS = (0.0, 0.99)  # local choice
_DIVERGENCE = 1e7  # rollouts beyond this magnitude are invalid
# Errors a parseable expression can still raise when evaluated (e.g. 1/0, 1e200**2).
_EVALUATION_ERRORS = (ArithmeticError, TypeError, ValueError, KeyError, RecursionError)
_PENALTY = 1e20

GLUCOSE_PROPOSER_SYSTEM = (
    "Infer a reduced continuous-time compartmental ODE from noisy CGM and intermittent "
    "plasma-insulin observations and known meal/insulin inputs. You have no simulator access. "
    'Return ONLY JSON {"models":[...]} with at most two new models. Each model has name, states '
    "(1-6 unique identifiers), rhs (one expression per state), initial_states, outputs (two "
    "observation expressions: CGM mg/dL and plasma insulin pmol/L), bounds (positive [lo,hi] per "
    "parameter), initial (parameter values). Parameters p0,p1,...,p11. Independent constants: gb "
    "(mean CGM prefix), ib (mean observed insulin prefix), meal (g/min), insulin (U/min above "
    "background). Time in minutes. Allowed arithmetic +,-,*,/,**, exp,log,sqrt,tanh,abs,minimum,"
    "maximum; no arbitrary Python, indexing, conditionals, or imports. Initial state expressions "
    "may use only gb, ib and parameters. Models restart at minute45; effective meal/insulin "
    "deviation states start at zero. Insulin assays have independent Gaussian SD 10 pmol/L and "
    "missing entries are null, never zero. All parameters and CGM AR1 observation-noise "
    "parameters will be refitted by multistart optimization. Propose mechanistic edits addressing "
    "training residuals, not polynomial time interpolation. No patient identity or ground-truth "
    "model is known. Existing models are examples of the exact schema. For EACH proposed model add "
    "parent_ids (a list of model_id strings from existing_models), origin (\"edit\", "
    '"recombination", or "de_novo"), and edit_summary (a nonempty explanation). An edit has '
    "exactly one parent; recombination has at least two distinct parents; de_novo has none. Name "
    "actual source models, not simply the best-scoring model. All models remain visible as "
    "context, regardless of which parents you declare."
)
GLUCOSE_DELIVERY = (
    "Meals consumed at 5g/min; bolus spread uniformly over 5min. "
    "Constant basal is absorbed into equilibrium."
)
GLUCOSE_ICL_SYSTEM = (
    "Predict future CGM (mg/dL) and plasma insulin (pmol/L) directly from observed examples. Do "
    "not propose a mechanistic model or run code. You have complete training episodes, only the "
    "query observations through the forecast cut, and the complete known query input sequence. "
    "Insulin nulls are missing assays, not zeros. CGM noise is correlated; insulin assay SD is 10 "
    "pmol/L. Predict conditional means, not random noise. Return ONLY JSON with cgm_mg_dl and "
    "insulin_pmol_l arrays, in exactly the requested time order. No extra keys. No future "
    "measurements are available."
)


def _glucosebench():
    try:
        import glucosebench
    except ImportError as error:
        raise RuntimeError("install model-discovery-agent[glucose] for GlucoseBench") from error
    return glucosebench


# ---------------------------------------------------------------------------
# Episodes and inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class GlucoseInputs:
    """Per-minute meal (g/min) and bolus (U/min above basal) inputs of one episode."""

    key: str
    meal: np.ndarray
    insulin: np.ndarray
    protocol: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("meal", "insulin"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (360,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must hold 360 finite per-minute rates")
            object.__setattr__(self, name, values)

    def __repr__(self) -> str:
        return f"GlucoseInputs({self.key!r})"


def nominal_inputs(protocol: Mapping[str, float], key: str) -> GlucoseInputs:
    """Delivery schedule of a menu template, before running it.

    Meals are eaten at up to 5 g/min and boluses spread uniformly over five
    minutes, as the proposer prompt states; the oracle's pump quantization of
    bolus rates is not applied (local choice). Fits use delivered inputs.
    """
    meal, insulin = np.zeros(360), np.zeros(360)
    remaining, minute = float(protocol["meal_g"]), int(protocol["meal_minute"])
    while remaining > 0 and minute < 360:
        meal[minute] = min(5.0, remaining)
        remaining -= meal[minute]
        minute += 1
    start = int(protocol["bolus_minute"])
    insulin[start:start + 5] = float(protocol["bolus_U"]) / 5
    return GlucoseInputs(key, meal, insulin, dict(protocol))


def episode_inputs(row: Mapping[str, Any], key: str) -> GlucoseInputs:
    columns = np.asarray(row["minute_inputs"], dtype=float)
    return GlucoseInputs(key, columns[:, 1], columns[:, 3], dict(row.get("protocol", {})))


def episode_observation(row: Mapping[str, Any], key: str = "episode") -> np.ndarray:
    """CGM at the 73 sample times followed by the 25 scheduled insulin assays.

    A missing or non-finite scheduled value raises, naming ``key`` and the
    minute, rather than giving every model an infinite likelihood.
    """
    cgm = np.asarray(row["cgm_mg_dl"], dtype=float)
    mask = np.asarray(row["insulin_mask"], dtype=bool)
    times = np.asarray(row["times_min"], dtype=float)
    if cgm.shape != (73,) or not np.array_equal(times[mask], INSULIN_TIMES):
        raise ValueError("expected 73 CGM readings and assays every 15 minutes")
    insulin = np.asarray([row["insulin_pmol_l"][i] for i in np.flatnonzero(mask)], dtype=float)
    for label, values, minutes in (("CGM reading", cgm, CGM_TIMES), ("insulin assay", insulin, INSULIN_TIMES)):
        missing = np.flatnonzero(~np.isfinite(values))
        if missing.size:
            raise ValueError(f"{key}: the {label} at minute {int(minutes[missing[0]])} is missing or non-finite")
    return np.concatenate([cgm, insulin])


def _split(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observation = np.asarray(observation, dtype=float)
    return observation[..., :73], observation[..., 73:]


@dataclass
class GlucoseConstants:
    """Training-prefix baselines g_b and I_b shared by models and inference.

    v5 averages CGM and observed insulin over the 45-minute prefixes of all
    training episodes. The evidence backend refreshes them on every refit.
    """

    gb: float = 120.0
    ib: float = 100.0

    def update(self, dataset: Sequence[Datum]) -> None:
        cgm, insulin = _split(np.stack([datum.observation for datum in dataset]))
        self.gb = float(np.mean(cgm[:, _CGM_PREFIX]))
        self.ib = float(np.mean(insulin[:, _INSULIN_PREFIX]))

    @property
    def values(self) -> tuple[float, float]:
        return self.gb, self.ib


# ---------------------------------------------------------------------------
# Model specifications and vectorized integration
# ---------------------------------------------------------------------------

def _tuple_of_strings(value, name) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{name} must be a list of strings")
    return tuple(v.strip() for v in value)


@dataclass(frozen=True)
class GlucoseModelSpec:
    """One reduced ODE in the proposer's JSON schema."""

    name: str
    states: tuple[str, ...]
    rhs: tuple[str, ...]
    initial_states: tuple[str, ...]
    outputs: tuple[str, str]
    bounds: tuple[tuple[float, float], ...]
    initial: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a model needs a name")
        states = _tuple_of_strings(self.states, "states")
        reserved = set(CONSTANT_NAMES) | set(INPUT_NAMES) | set(_FUNCTIONS) | {"pi", "e"}
        if not 1 <= len(states) <= MAX_STATES or len(set(states)) != len(states):
            raise ValueError(f"models need 1-{MAX_STATES} unique states")
        if any(not s.isidentifier() or keyword.iskeyword(s) or s in reserved
               or (s.startswith("p") and s[1:].isdigit()) for s in states):
            raise ValueError("state names must be identifiers distinct from constants and parameters")
        rhs = _tuple_of_strings(self.rhs, "rhs")
        initial_states = _tuple_of_strings(self.initial_states, "initial_states")
        outputs = _tuple_of_strings(self.outputs, "outputs")
        if len(rhs) != len(states) or len(initial_states) != len(states) or len(outputs) != 2:
            raise ValueError("need one rhs and initial state per state and two outputs")
        try:
            bounds = tuple((float(lo), float(hi)) for lo, hi in self.bounds)
            initial = tuple(float(v) for v in self.initial)
        except (TypeError, ValueError) as error:
            raise ValueError("bounds must be [lo, hi] pairs and initial a list of numbers") from error
        if len(bounds) > MAX_PARAMETERS or len(initial) != len(bounds):
            raise ValueError(f"at most {MAX_PARAMETERS} parameters, each with bounds and a value")
        if any(not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi) for lo, hi in bounds):
            raise ValueError("parameter bounds must be finite, positive, and ordered")
        if any(not lo <= v <= hi for v, (lo, hi) in zip(initial, bounds)):
            raise ValueError("initial parameter values must lie within their bounds")
        object.__setattr__(self, "name", self.name.strip())
        for key, value in (("states", states), ("rhs", rhs), ("initial_states", initial_states),
                           ("outputs", outputs), ("bounds", bounds), ("initial", initial)):
            object.__setattr__(self, key, value)
        GlucoseODE(self)  # validates every expression

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"p{i}" for i in range(len(self.bounds)))

    @property
    def key(self) -> str:
        """Expression identity, up to whitespace and redundant parentheses, used to deduplicate proposals."""
        expressions = [[canonical_expression(e) for e in group]
                       for group in (self.rhs, self.initial_states, self.outputs)]
        return json.dumps([self.states, *expressions])

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "states": list(self.states), "rhs": list(self.rhs),
                "initial_states": list(self.initial_states), "outputs": list(self.outputs),
                "bounds": [list(b) for b in self.bounds], "initial": list(self.initial)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlucoseModelSpec":
        try:
            return cls(payload["name"], payload["states"], payload["rhs"], payload["initial_states"],
                       payload["outputs"], payload["bounds"], payload["initial"])
        except (KeyError, TypeError) as error:
            raise ValueError("incomplete model specification") from error


class GlucoseODE:
    """Vectorized integration of one specification.

    States have shape ``(n_states, B, E)``: ``B`` parameter vectors (or state
    perturbations) by ``E`` episodes. Parameters enter with shape ``(B, 1)``.
    Before minute 45 the model sits at its initial state; afterwards one RK4
    step per minute uses that minute's inputs.
    """

    def __init__(self, spec: GlucoseModelSpec) -> None:
        self.spec = spec
        self.parameters = spec.parameter_names
        self.n = len(spec.states)
        self.k = len(self.parameters)
        self._rhs = compile_expressions(
            spec.rhs, (*spec.states, *CONSTANT_NAMES, *INPUT_NAMES, *self.parameters), GLUCOSE_FUNCTIONS)
        self._initial = compile_expressions(
            spec.initial_states, (*CONSTANT_NAMES, *self.parameters), GLUCOSE_FUNCTIONS)
        self._outputs = compile_expressions(
            spec.outputs, (*spec.states, *CONSTANT_NAMES, *self.parameters), GLUCOSE_FUNCTIONS)

    def values(self, theta: np.ndarray, constants: tuple[float, float]) -> dict[str, Any]:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))[:, :self.k]
        values = {name: theta[:, j, None] for j, name in enumerate(self.parameters)}
        values.update(gb=constants[0], ib=constants[1])
        return values

    def initial(self, values: Mapping[str, Any], shape: tuple[int, ...]) -> np.ndarray:
        return np.stack([np.broadcast_to(v, shape) for v in self._initial(values)]).astype(float)

    def derivative(self, z, meal, insulin, values) -> np.ndarray:
        environment = dict(values)
        environment.update(zip(self.spec.states, z))
        environment.update(meal=meal, insulin=insulin)
        return np.stack([np.broadcast_to(v, z.shape[1:]) for v in self._rhs(environment)])

    def outputs(self, z, values) -> tuple[np.ndarray, np.ndarray]:
        environment = dict(values)
        environment.update(zip(self.spec.states, z))
        cgm, insulin = self._outputs(environment)
        return np.broadcast_to(cgm, z.shape[1:]), np.broadcast_to(insulin, z.shape[1:])

    def advance(self, z, meal_rows, insulin_rows, values) -> np.ndarray:
        """RK4 through consecutive minutes; rows are ``(minutes, E)`` input rates."""
        with np.errstate(all="ignore"):
            for meal, insulin in zip(meal_rows, insulin_rows):
                k1 = self.derivative(z, meal, insulin, values)
                k2 = self.derivative(z + 0.5 * k1, meal, insulin, values)
                k3 = self.derivative(z + 0.5 * k2, meal, insulin, values)
                k4 = self.derivative(z + k3, meal, insulin, values)
                z = z + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return z

    def simulate(self, theta, inputs: Sequence[GlucoseInputs], constants) -> tuple[np.ndarray, np.ndarray]:
        """Mean CGM ``(B, E, 73)`` and insulin ``(B, E, 25)``; NaN marks invalid rollouts."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        meal = np.stack([i.meal for i in inputs], axis=1)
        insulin = np.stack([i.insulin for i in inputs], axis=1)
        values = self.values(theta, constants)
        shape = (theta.shape[0], len(inputs))
        z = self.initial(values, shape)
        cgm, assay = np.empty((*shape, 73)), np.empty((*shape, 25))
        c, a = self.outputs(z, values)
        cgm[..., _CGM_PREFIX] = c[..., None]
        assay[..., _INSULIN_PREFIX] = a[..., None]
        for index in np.flatnonzero(~_CGM_PREFIX):
            t = CGM_TIMES[index]
            z = self.advance(z, meal[t - 5:t], insulin[t - 5:t], values)
            z = np.where(np.isfinite(z) & (np.abs(z) < _DIVERGENCE), z, np.nan)
            c, a = self.outputs(z, values)
            cgm[..., index] = c
            if t % 15 == 0:
                assay[..., t // 15] = a
        bad = ~np.isfinite(cgm).all(axis=-1) | ~np.isfinite(assay).all(axis=-1)
        cgm[bad], assay[bad] = np.nan, np.nan
        return cgm, assay


# ---------------------------------------------------------------------------
# Likelihood, BIC evidence, and the expansion gate
# ---------------------------------------------------------------------------

def _negative_log_likelihood(cgm_residual, insulin_residual, rho, sigma=None):
    """AR(1) CGM residuals with profiled innovation variance plus Gaussian assays.

    Residual arrays are ``(B, E, T)`` from minute 45; ``rho`` is ``(B,)``.
    Returns the minimized negative log likelihood and the profiled sigma_eta,
    or, given ``sigma`` ``(B,)``, the value at that innovation SD and ``sigma``.
    """
    rho = np.asarray(rho, dtype=float)
    innovations = np.empty_like(cgm_residual)
    innovations[..., 0] = np.sqrt(1 - rho**2)[:, None] * cgm_residual[..., 0]
    innovations[..., 1:] = cgm_residual[..., 1:] - rho[:, None, None] * cgm_residual[..., :-1]
    episodes, n_cgm = cgm_residual.shape[1], cgm_residual.shape[1] * cgm_residual.shape[2]
    n_insulin = insulin_residual.shape[1] * insulin_residual.shape[2]
    squares = np.sum(innovations**2, axis=(1, 2))
    if sigma is None:
        variance = np.maximum(squares / n_cgm, np.finfo(float).tiny)
        cgm_term = 0.5 * n_cgm * (np.log(2 * np.pi * variance) + 1)
    else:
        variance = np.asarray(sigma, dtype=float) ** 2
        cgm_term = 0.5 * n_cgm * np.log(2 * np.pi * variance) + 0.5 * squares / variance
    nll = (cgm_term - 0.5 * episodes * np.log(1 - rho**2)
           + 0.5 * np.sum(insulin_residual**2, axis=(1, 2)) / ASSAY_SD**2
           + 0.5 * n_insulin * np.log(2 * np.pi * ASSAY_SD**2))
    return np.where(np.isfinite(nll), nll, np.inf), np.sqrt(variance)


def _residuals(ode, theta, dataset, constants):
    cgm, insulin = ode.simulate(theta, [d.action for d in dataset], constants)
    observed_cgm, observed_insulin = _split(np.stack([d.observation for d in dataset]))
    return (observed_cgm[None, :, _CGM_FIT] - cgm[..., _CGM_FIT],
            observed_insulin[None, :, _INSULIN_FIT] - insulin[..., _INSULIN_FIT])


@dataclass(frozen=True)
class GlucoseLikelihood:
    """Working log likelihood at ``theta = (ODE parameters, rho, sigma_eta)``.

    Only used outside the BIC backend, e.g. if an SMC backend is selected;
    ``ModelDiscoveryAgent`` requires a likelihood. Expressions that raise when
    evaluated give a log likelihood of -inf.
    """

    constants: GlucoseConstants

    def __call__(self, dataset, model, theta, rng) -> float:
        del rng
        ode = model.metadata["ode"]
        theta = np.asarray(theta, dtype=float)
        try:
            cgm, insulin = _residuals(ode, theta[None, :ode.k], dataset, self.constants.values)
        except _EVALUATION_ERRORS:
            return -np.inf
        nll, _ = _negative_log_likelihood(cgm, insulin, theta[None, ode.k], theta[None, ode.k + 1])
        return -float(nll[0])


@dataclass(frozen=True)
class GlucoseFitSettings:
    random_starts: int = 3  # local choice; v5 says "multistart" without a count
    max_iterations: int = 200
    step: float = 1e-6


class GlucoseEvidence:
    """``marglik-bic`` for glucose ODEs: multistart L-BFGS-B and BIC.

    Parameters are optimized in log space (bounds are positive) with rho in
    ``RHO_BOUNDS``; gradients are batched forward differences. Starts are the
    model's previous optimum (warm start), its declared initial values, and
    ``random_starts`` log-uniform draws. BIC = 2 NLL + (k + 2) log N with N
    the number of observed targets from minute 45. A model whose every start
    fails, including one whose expressions raise when evaluated, gets zero
    evidence and leaves the pool. The returned point mass is
    ``(theta_hat, rho_hat, sigma_eta_hat)``.
    """

    def __init__(self, constants: GlucoseConstants, settings: GlucoseFitSettings = GlucoseFitSettings()):
        self.constants = constants
        self.settings = settings
        self._warm: dict[str, np.ndarray] = {}

    def __call__(self, dataset, model, likelihood, rng) -> ParameterPosterior:
        from scipy.optimize import minimize

        del likelihood
        dataset = list(dataset)
        self.constants.update(dataset)
        constants = self.constants.values
        ode: GlucoseODE = model.metadata["ode"]
        k = ode.k
        lower = np.log([b[0] for b in ode.spec.bounds])
        upper = np.log([b[1] for b in ode.spec.bounds])
        bounds = [*zip(lower, upper), RHO_BOUNDS]
        x_upper = np.r_[upper, RHO_BOUNDS[1]]

        def batch(x_rows):
            x_rows = np.atleast_2d(x_rows)
            try:
                cgm, insulin = _residuals(ode, np.exp(x_rows[:, :k]), dataset, constants)
            except _EVALUATION_ERRORS:  # an expression that raises is an invalid rollout
                return np.full(len(x_rows), np.inf), np.full(len(x_rows), np.nan)
            return _negative_log_likelihood(cgm, insulin, x_rows[:, k])

        def objective(x):
            steps = self.settings.step * np.maximum(1.0, np.abs(x))
            steps = np.where(x + steps > x_upper, -steps, steps)
            rows = np.vstack([x, x + np.diag(steps)])
            values, _ = batch(rows)
            if not np.isfinite(values[0]):
                return _PENALTY, np.zeros_like(x)
            gradient = np.where(np.isfinite(values[1:]), (values[1:] - values[0]) / steps, 0.0)
            return float(values[0]), gradient

        starts = [] if model.key not in self._warm else [self._warm[model.key]]
        starts.append(np.r_[np.log(ode.spec.initial), 0.5])
        for _ in range(self.settings.random_starts):
            starts.append(np.r_[rng.uniform(lower, upper), rng.uniform(*RHO_BOUNDS)])
        best_x, best_value = None, np.inf
        for start in starts:
            start = np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds])
            result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": self.settings.max_iterations})
            value = float(batch(result.x)[0][0])
            if value < best_value:
                best_x, best_value = result.x, value
        count = len(dataset) * int(_CGM_FIT.sum() + _INSULIN_FIT.sum())
        if best_x is None or not np.isfinite(best_value):
            theta = np.r_[np.exp(starts[-1][:k]), starts[-1][k], 1.0]
            return _point_posterior(theta, -np.inf, -np.inf)
        self._warm[model.key] = best_x
        sigma = float(batch(best_x)[1][0])
        bic = 2 * best_value + (k + 2) * np.log(count)
        return _point_posterior(np.r_[np.exp(best_x[:k]), best_x[k], sigma], -best_value, -0.5 * bic)


def _point_posterior(theta, log_likelihood, log_evidence) -> ParameterPosterior:
    return ParameterPosterior(
        particles=np.asarray(theta, dtype=float)[None, :], weights=np.ones(1),
        log_likelihoods=np.asarray([log_likelihood]), log_evidence=float(log_evidence),
        temperatures=(1.0,), acceptance_rate=0.0, log_priors=np.zeros(1), map_log_priors=np.zeros(1),
    )


def glucose_candidate(spec: GlucoseModelSpec, constants: GlucoseConstants,
                      genealogy: Mapping[str, Any] | None = None) -> CandidateModel:
    """A pool member; its parameters are the ODE constants plus rho and sigma_eta."""
    ode = GlucoseODE(spec)
    lower = [b[0] for b in spec.bounds] + [RHO_BOUNDS[0], 1e-9]
    upper = [b[1] for b in spec.bounds] + [RHO_BOUNDS[1], 1e9]

    def predict(theta, action):
        cgm, insulin = ode.simulate(np.asarray(theta)[None, :ode.k], [action], constants.values)
        return np.concatenate([cgm[0, 0], insulin[0, 0]])

    return CandidateModel(
        name=spec.name, structure=spec.key,
        prior=UniformPrior((*spec.parameter_names, "rho", "sigma_eta"), np.asarray(lower), np.asarray(upper)),
        predict=predict, metadata={"spec": spec, "ode": ode, "genealogy": dict(genealogy or {})},
    )


def gate_statistic(model: CandidateModel, theta: np.ndarray, datum: Datum,
                   constants: tuple[float, float]) -> float:
    """A_m of Eq. glucose_expansion_gate for one episode from minute 45.

    CGM residuals are whitened by the fitted AR(1) correlation and innovation
    scale; insulin residuals use the declared assay SD.
    """
    ode = model.metadata["ode"]
    theta = np.asarray(theta, dtype=float)
    rho, sigma = theta[ode.k], theta[ode.k + 1]
    cgm, insulin = _residuals(ode, theta[None, :ode.k], [datum], constants)
    r = cgm[0, 0]
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(insulin)):
        return np.inf
    innovations = np.r_[np.sqrt(1 - rho**2) * r[0], r[1:] - rho * r[:-1]]
    n = r.size + insulin[0, 0].size
    return float((np.sum(innovations**2) / sigma**2 + np.sum(insulin**2) / ASSAY_SD**2) / n)


def gate_threshold(quantile: float = 0.995) -> float:
    from scipy.stats import chi2

    n = int(_CGM_FIT.sum() + _INSULIN_FIT.sum())
    return float(chi2.ppf(quantile, n) / n)


def make_gate(constants: GlucoseConstants, quantile: float = 0.995):
    """Pre-refit gate: min over the pool of A_m on the new episode, over the chi^2 cutoff.

    Values above one (the agent's ``error_threshold``) trigger expansion.
    """
    threshold = gate_threshold(quantile)

    def check(belief: BeliefState, fresh: Datum, history) -> float:
        del history
        return min(gate_statistic(fit.model, fit.parameters.map_parameters, fresh, constants.values)
                   for fit in belief.fits) / threshold

    return check


# ---------------------------------------------------------------------------
# Seed library (v5: four reduced compartmental ODEs; only the two-stage meal
# model's equations are printed, the others are local reconstructions)
# ---------------------------------------------------------------------------

_TAU = (1.0, 500.0)
_KG, _C0, _ALPHA, _BETA, _GAIN = (1e-4, 1.0), (0.5, 2.0), (1e-4, 10.0), (1e-4, 50.0), (1.0, 5000.0)
SEED_SPECS = (
    GlucoseModelSpec(
        "one_stage_linear_action", ("q", "i", "g"),
        ("meal - q/p0", "insulin - i/p1", "p2*(gb*p3 - g) + p4*q - p5*i"),
        ("0", "0", "gb"), ("g", "ib + p6*i"),
        (_TAU, _TAU, _KG, _C0, _ALPHA, _BETA, _GAIN), (30.0, 60.0, 0.01, 1.0, 0.05, 0.5, 100.0)),
    GlucoseModelSpec(
        "one_stage_glucose_dependent_action", ("q", "i", "g"),
        ("meal - q/p0", "insulin - i/p1", "p2*(gb*p3 - g) + p4*q - p5*i*g/gb"),
        ("0", "0", "gb"), ("g", "ib + p6*i"),
        (_TAU, _TAU, _KG, _C0, _ALPHA, _BETA, _GAIN), (30.0, 60.0, 0.01, 1.0, 0.05, 0.5, 100.0)),
    GlucoseModelSpec(  # Eq. glucose_reduced_ode
        "two_stage_meal", ("q1", "q2", "i", "g"),
        ("meal - q1/p0", "(q1 - q2)/p1", "insulin - i/p2", "p3*(gb*p4 - g) + p5*q2 - p6*i*g/gb"),
        ("0", "0", "0", "gb"), ("g", "ib + p7*i"),
        (_TAU, _TAU, _TAU, _KG, _C0, _ALPHA, _BETA, _GAIN),
        (20.0, 40.0, 60.0, 0.01, 1.0, 0.05, 0.5, 100.0)),
    GlucoseModelSpec(
        "delayed_cgm_readout", ("q", "i", "g", "c"),
        ("meal - q/p0", "insulin - i/p1", "p2*(gb*p3 - g) + p4*q - p5*i*g/gb", "(g - c)/p7"),
        ("0", "0", "gb", "gb"), ("c", "ib + p6*i"),
        (_TAU, _TAU, _KG, _C0, _ALPHA, _BETA, _GAIN, (1.0, 60.0)),
        (30.0, 60.0, 0.01, 1.0, 0.05, 0.5, 100.0, 10.0)),
)


# ---------------------------------------------------------------------------
# Design and forecasting
# ---------------------------------------------------------------------------

def _components(belief: BeliefState):
    for weight, fit in zip(belief.model_weights, belief.fits, strict=True):
        if weight == 0.0:
            continue
        ode = fit.model.metadata["ode"]
        theta = fit.parameters.map_parameters
        yield ode, theta[:ode.k], float(theta[ode.k]), float(theta[ode.k + 1]), float(weight)


def design_scores(belief: BeliefState, candidates: Sequence[GlucoseInputs],
                  constants: tuple[float, float]) -> np.ndarray:
    """Eq. glucose-model-disagreement for each candidate input schedule."""
    components = list(_components(belief))
    weights = np.asarray([c[4] for c in components])
    cgm_means, insulin_means, stationary = [], [], []
    for ode, theta, rho, sigma, _ in components:
        cgm, insulin = ode.simulate(theta[None, :], candidates, constants)
        cgm_means.append(cgm[0][:, _CGM_FIT])
        insulin_means.append(insulin[0][:, _INSULIN_FIT])
        stationary.append(sigma**2 / (1 - rho**2))
    scale_cgm = max(1.0, float(weights @ np.asarray(stationary)))
    cgm_stack, insulin_stack = np.stack(cgm_means), np.stack(insulin_means)  # (models, candidates, times)
    # A model whose rollout fails on a candidate drops out of that candidate's
    # mixture, so one invalid low-weight model cannot void the whole score.
    finite = np.all(np.isfinite(cgm_stack), axis=-1) & np.all(np.isfinite(insulin_stack), axis=-1)
    mixture = weights[:, None] * finite
    total = mixture.sum(axis=0)
    mixture = np.divide(mixture, total, out=np.zeros_like(mixture), where=total > 0)

    def disagreement(means):
        means = np.where(finite[..., None], means, 0.0)
        centered = means - np.einsum("mc,mct->ct", mixture, means)
        return np.einsum("mc,mct->c", mixture, centered**2) / means.shape[-1]

    scores = 0.5 * (disagreement(cgm_stack) / scale_cgm + disagreement(insulin_stack) / ASSAY_SD**2)
    return np.where(np.isfinite(scores) & (total > 0), scores, -np.inf)


def _ekf_forecast(ode, theta, rho, sigma, constants, payload) -> tuple[np.ndarray, np.ndarray]:
    """One component's conditional mean after filtering the query prefix.

    The augmented state is the ODE state plus the AR(1) sensor error e. Before
    minute 45 the ODE state is held at its restart value; e follows the AR(1)
    on the five-minute grid. Jacobians are finite differences of the five-
    minute RK4 flow (local choice), the initial ODE-state SD is 5% of its
    magnitude with a unit floor (local choice), and CGM updates use a 1e-6
    jitter because the sensor error is already a state. As in ``simulate``, a
    state that becomes non-finite or reaches ``_DIVERGENCE`` voids the whole
    forecast (all NaN), so the mixture drops the component.
    """
    query, cut = payload["query"], int(payload["forecast_cut_min"])
    inputs = episode_inputs(query, "query")
    values = ode.values(theta[None, :], constants)
    n = ode.n
    z = ode.initial(values, (1, 1))[:, 0, 0]
    e = 0.0
    covariance = np.diag(np.r_[(0.05 * np.maximum(np.abs(z), 1.0))**2, sigma**2 / (1 - rho**2)])
    cgm_seen = list(query["cgm_mg_dl"])
    insulin_seen, insulin_mask = list(query["insulin_pmol_l"]), list(query["insulin_mask"])
    cgm_pred, insulin_pred = np.empty(73), np.empty(25)
    failed = np.full(73, np.nan), np.full(25, np.nan)

    def diverged():
        state = np.r_[z, e]
        return not (np.all(np.isfinite(state)) and np.all(np.abs(state) < _DIVERGENCE))

    def outputs(states):
        c, i = ode.outputs(states[:, :, None], values)
        return c[:, 0], i[:, 0]

    def jacobian_outputs(z):
        steps = 1e-4 * np.maximum(1.0, np.abs(z))
        states = np.column_stack([z, z[:, None] + np.diag(steps)])
        c, i = outputs(states)
        return c[0], i[0], (c[1:] - c[0]) / steps, (i[1:] - i[0]) / steps

    def update(y, prediction, h, noise):
        nonlocal z, e, covariance
        h = np.asarray(h, dtype=float)
        innovation_variance = float(h @ covariance @ h) + noise
        gain = covariance @ h / innovation_variance
        state = np.r_[z, e] + gain * (y - prediction)
        z, e = state[:n], state[n]
        identity = np.eye(n + 1) - np.outer(gain, h)
        covariance = identity @ covariance @ identity.T + noise * np.outer(gain, gain)

    for index, t in enumerate(CGM_TIMES):
        filtering = 0 < cut and t <= cut
        if index > 0:
            transition = np.eye(n + 1)
            if t > PREFIX_MIN:
                window = slice(t - 5, t)
                if filtering:
                    steps = 1e-4 * np.maximum(1.0, np.abs(z))
                    states = np.column_stack([z, z[:, None] + np.diag(steps)])[:, :, None]
                    moved = ode.advance(states, inputs.meal[window, None], inputs.insulin[window, None], values)
                    z_next = moved[:, 0, 0]
                    transition[:n, :n] = (moved[:, 1:, 0] - z_next[:, None]) / steps
                    z = z_next
                else:
                    z = ode.advance(z[:, None, None], inputs.meal[window, None],
                                    inputs.insulin[window, None], values)[:, 0, 0]
            e *= rho
            if filtering:
                transition[n, n] = rho
                covariance = transition @ covariance @ transition.T
                covariance[n, n] += sigma**2
        if diverged():
            return failed
        if filtering:
            c, i, dc, di = jacobian_outputs(z)
            update(float(cgm_seen[index]), c + e, np.r_[dc, 1.0], 1e-6)
            if insulin_mask[index] and insulin_seen[index] is not None:
                c, i, dc, di = jacobian_outputs(z)
                update(float(insulin_seen[index]), i, np.r_[di, 0.0], ASSAY_SD**2)
            covariance = (covariance + covariance.T) / 2
            if diverged():
                return failed
        c, i = outputs(z[:, None])
        cgm_pred[index] = c[0] + e
        if t % 15 == 0:
            insulin_pred[t // 15] = i[0]
    return cgm_pred, insulin_pred


@dataclass(frozen=True)
class GlucoseForecaster:
    """Frozen snapshot predictor for the glucosebench payload contract.

    Each component is (ODE, theta_hat, rho, sigma_eta, weight); the forecast
    is the weight-averaged EKF conditional mean (``bma``) or the MAP
    component's (``map``).
    """

    components: tuple[tuple[GlucoseODE, np.ndarray, float, float, float], ...]
    constants: tuple[float, float]

    @classmethod
    def from_agent(cls, agent: "GlucoseAgent", flavour: str = "bma") -> "GlucoseForecaster":
        components = list(_components(agent.belief))
        if flavour == "map":
            best = max(components, key=lambda c: c[4])
            components = [(*best[:4], 1.0)]
        elif flavour != "bma":
            raise ValueError("glucose prediction flavour must be bma or map")
        return cls(tuple(components), agent.constants.values)

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, list[float]]:
        cut = int(payload["forecast_cut_min"])
        cgm_times = np.asarray(payload["requested_cgm_times_min"], dtype=float)
        insulin_times = np.asarray(payload["requested_insulin_times_min"], dtype=float)
        cgm, insulin, total = np.zeros(73), np.zeros(25), 0.0
        for ode, theta, rho, sigma, weight in self.components:
            c, i = _ekf_forecast(ode, theta, rho, sigma, self.constants, payload)
            if np.all(np.isfinite(c)) and np.all(np.isfinite(i)):
                cgm, insulin, total = cgm + weight * c, insulin + weight * i, total + weight
        if total <= 0:
            raise ValueError("every mixture component failed to forecast")
        cgm, insulin = cgm / total, insulin / total
        if not np.array_equal(cgm_times, CGM_TIMES[CGM_TIMES > cut]):
            raise ValueError("unexpected requested CGM times")
        return {"cgm_mg_dl": cgm[CGM_TIMES > cut].tolist(),
                "insulin_pmol_l": insulin[np.searchsorted(INSULIN_TIMES, insulin_times)].tolist()}


# ---------------------------------------------------------------------------
# LLM proposer, ICL baseline, agent, environment, and runner
# ---------------------------------------------------------------------------

def _public_episode(datum: Datum) -> dict[str, Any]:
    cgm, insulin = _split(datum.observation)
    assays = {int(t): float(v) for t, v in zip(INSULIN_TIMES, insulin)}
    return {"times_min": CGM_TIMES.tolist(), "cgm_mg_dl": np.round(cgm, 3).tolist(),
            "insulin_pmol_l": [round(assays[int(t)], 3) if int(t) in assays else None for t in CGM_TIMES],
            "insulin_mask": [int(t) % 15 == 0 for t in CGM_TIMES],
            "insulin_noise_sd_pmol_l": ASSAY_SD, "protocol": dict(datum.action.protocol),
            "prefix_end_min": PREFIX_MIN}


_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "states": {"type": "array", "items": {"type": "string"}},
        "rhs": {"type": "array", "items": {"type": "string"}},
        "initial_states": {"type": "array", "items": {"type": "string"}},
        "outputs": {"type": "array", "items": {"type": "string"}},
        "bounds": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        "initial": {"type": "array", "items": {"type": "number"}},
        "parent_ids": {"type": "array", "items": {"type": "string"}},
        "origin": {"type": "string", "enum": ["edit", "recombination", "de_novo"]},
        "edit_summary": {"type": "string"},
    },
    "required": ["name", "states", "rhs", "initial_states", "outputs", "bounds", "initial",
                 "parent_ids", "origin", "edit_summary"],
    "additionalProperties": False,
}


class GlucoseProposer:
    """v5 glucose proposer: the published system prompt and JSON model schema.

    The payload holds the training observations, the delivery description,
    and each existing model's specification, fitted parameters, BIC, and
    rounded training predictions. Invalid specifications, inconsistent
    genealogy (an edit needs one existing parent, a recombination two or more
    distinct ones, de novo none), and models whose expressions raise when
    simulated once at their declared parameter values are dropped; a rollout
    that merely diverges is kept, since fitting treats it as invalid.
    """

    def __init__(self, client, constants: GlucoseConstants) -> None:
        self.client = client
        self.constants = constants
        self.belief_source: Callable[[], BeliefState] | None = None

    def propose(self, *, context, dataset, current_models, residuals, count) -> list[CandidateModel]:
        del context, residuals
        belief = self.belief_source() if self.belief_source is not None else None
        existing = []
        for fit in (belief.fits if belief is not None else ()):
            ode = fit.model.metadata["ode"]
            theta = fit.parameters.map_parameters
            predictions = [np.round(fit.model.observation_prediction(theta, d.action), 3) for d in dataset]
            existing.append({
                "model_id": fit.model.name, "specification": ode.spec.to_dict(),
                "theta": np.round(theta[:ode.k], 6).tolist(),
                "ar1_rho": round(float(theta[ode.k]), 4), "ar1_sigma": round(float(theta[ode.k + 1]), 4),
                "bic": round(-2 * fit.parameters.log_evidence, 3),
                "training_predictions": [{"cgm_mg_dl": p[:73].tolist(), "insulin_assays_pmol_l": p[73:].tolist()}
                                         for p in predictions],
            })
        payload = {"observations": [_public_episode(d) for d in dataset],
                   "delivery": GLUCOSE_DELIVERY, "existing_models": existing}
        schema = {"type": "object", "properties": {"models": {"type": "array", "items": _SPEC_SCHEMA}},
                  "required": ["models"], "additionalProperties": False}
        answer = self.client.ask_json(GLUCOSE_PROPOSER_SYSTEM, json.dumps(payload), schema=schema,
                                      name="glucose_models", max_tokens=6000)
        names = {model.name for model in current_models}
        keys = {model.key for model in current_models}
        # Screening inputs: the first training episode, or a nominal schedule before any data.
        screen = [dataset[0].action] if dataset else [
            nominal_inputs(dict(meal_g=50, meal_minute=60, bolus_U=1, bolus_minute=60), "screen")]
        models = []
        for item in answer.get("models", [])[:min(2, count)]:
            try:
                spec = GlucoseModelSpec.from_dict(item)
                parents = list(dict.fromkeys(item["parent_ids"]))
                origin = item["origin"]
                if any(parent not in names for parent in parents) or not str(item["edit_summary"]).strip():
                    raise ValueError("unknown parent or empty edit summary")
                if (origin, len(parents)) not in {("edit", 1), ("de_novo", 0)} and not (
                        origin == "recombination" and len(parents) >= 2):
                    raise ValueError("genealogy inconsistent with origin")
                if spec.key in keys:
                    continue
                GlucoseODE(spec).simulate(np.asarray(spec.initial)[None], screen, self.constants.values)
            except _EVALUATION_ERRORS:
                continue
            name, suffix = spec.name, 2
            while name in names:
                name, suffix = f"{spec.name}_{suffix}", suffix + 1
            spec = GlucoseModelSpec.from_dict({**spec.to_dict(), "name": name})
            models.append(glucose_candidate(spec, self.constants, {
                "parent_ids": parents, "origin": origin, "edit_summary": item["edit_summary"]}))
            names.add(name)
            keys.add(spec.key)
        return models


def make_glucose_icl_forecaster(client):
    """ICL baseline: the v5 forecast prompt with one schema-only retry."""
    schema = {"type": "object", "properties": {
        "cgm_mg_dl": {"type": "array", "items": {"type": "number"}},
        "insulin_pmol_l": {"type": "array", "items": {"type": "number"}}},
        "required": ["cgm_mg_dl", "insulin_pmol_l"], "additionalProperties": False}

    def predictor(payload):
        prompt, error = json.dumps(payload), None
        for _ in range(2):
            user = prompt if error is None else (
                f"{prompt}\nPrevious output failed schema validation: {error}. Return complete corrected arrays.")
            answer = client.ask_json(GLUCOSE_ICL_SYSTEM, user, schema=schema, name="glucose_forecast",
                                     max_tokens=4000)
            for channel, times in (("cgm_mg_dl", "requested_cgm_times_min"),
                                   ("insulin_pmol_l", "requested_insulin_times_min")):
                values = np.asarray(answer.get(channel, []), dtype=float)
                if values.shape != (len(payload[times]),) or not np.all(np.isfinite(values)):
                    error = f"{channel} needs {len(payload[times])} finite values"
                    break
            else:
                return {"cgm_mg_dl": answer["cgm_mg_dl"], "insulin_pmol_l": answer["insulin_pmol_l"]}
        raise ValueError(error)

    return predictor


class GlucoseAgent(ModelDiscoveryAgent):
    """MDA on one GlucoseBench patient.

    After fitting D0, the initialization gate (min over models of the maximum
    A_m across D0) may trigger one expansion. Designs enumerate unused menu
    templates by Eq. glucose-model-disagreement with random tie-breaking.
    """

    def __init__(self, *, menu: Mapping[int, Mapping[str, float]], constants: GlucoseConstants,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.menu = dict(menu)
        self.constants = constants
        if isinstance(self.proposer, GlucoseProposer):
            self.proposer.belief_source = lambda: self.belief
        self.initial_gate_error = None
        if self.proposer is not None:
            threshold = gate_threshold()
            self.initial_gate_error = min(
                max(gate_statistic(fit.model, fit.parameters.map_parameters, datum, constants.values)
                    for datum in self.dataset)
                for fit in self.belief.fits) / threshold
            if self.initial_gate_error > self.config.error_threshold:
                self._expand()

    def candidate_actions(self) -> list[GlucoseInputs]:
        return [nominal_inputs(protocol, f"menu:{index}") for index, protocol in sorted(self.menu.items())
                if f"menu:{index}" not in self.used_action_keys]

    def design(self) -> GlucoseInputs:
        if self.experiment_budget is not None and self.experiments_run >= self.experiment_budget:
            raise ValueError("the experiment budget is exhausted")
        candidates = self.candidate_actions()
        if not candidates:
            raise ValueError("the design menu is exhausted")
        scores = design_scores(self.belief, candidates, self.constants.values)
        if not np.any(np.isfinite(scores)):
            raise ValueError("every candidate design has a non-finite score")
        best = np.flatnonzero(scores == scores.max())
        self.last_design_scores = scores
        return candidates[int(self.rng.choice(best))]


class GlucoseBenchEnvironment:
    """One patient's acquisition phase; evaluation starts only after ``freeze``."""

    def __init__(self, patient: str, *, replicate: int = 0) -> None:
        self.benchmark = _glucosebench().Benchmark(patient, replicate)
        self.patient, self.replicate = patient, replicate
        self.initial_data = [
            Datum(episode_inputs(row, f"passive:{index}"), episode_observation(row, f"passive:{index}"))
            for index, row in enumerate(self.benchmark.training)]
        self.menu = dict(self.benchmark.actions)
        self.context = ("Forecast one simulated type-1 diabetes patient's CGM and plasma insulin "
                        "responses to meal and bolus interventions.")

    def step(self, action: GlucoseInputs) -> tuple[GlucoseInputs, np.ndarray]:
        """Run a menu template; return its delivered inputs and observation vector."""
        row = self.benchmark.observe(int(action.key.split(":", 1)[1]))
        return episode_inputs(row, action.key), episode_observation(row, action.key)

    def freeze(self):
        return self.benchmark.freeze()


def make_agent(environment: GlucoseBenchEnvironment, client: Any, *, seed: int = 0, smoke: bool = False,
               experiment_budget: int = 4, no_m_open: bool = False) -> GlucoseAgent:
    """The v5 glucose arm: four seed ODEs, BIC weights, at most two proposals
    per call, an archive cap of 14, and one search round per update.

    ``no_m_open`` builds the agent without a proposer, so not even the
    initialization gate can add LLM models."""
    if client is None and not smoke:
        raise ValueError("client=None is permitted only when smoke=True")
    if not 0 <= experiment_budget <= 4:
        raise ValueError("GlucoseBench allows at most four selected experiments")
    constants = GlucoseConstants()
    settings = GlucoseFitSettings(random_starts=0, max_iterations=15) if smoke else GlucoseFitSettings()
    proposer = None if client is None or no_m_open else GlucoseProposer(client, constants)
    return GlucoseAgent(
        menu=environment.menu, constants=constants, context=environment.context,
        initial_data=environment.initial_data,
        initial_models=[glucose_candidate(spec, constants) for spec in SEED_SPECS],
        likelihood=GlucoseLikelihood(constants), actions=[], observation_covariance=1.0,
        config=AgentConfig(model_cap=14, new_models_per_expansion=2, model_search_rounds=1,
                           expansion_cap=5, error_threshold=1.0, occam_penalty=0.0, backtest=False),
        proposer=proposer, experiment_budget=experiment_budget, action_key=lambda action: action.key,
        evidence=GlucoseEvidence(constants, settings), check=make_gate(constants), seed=seed,
    )


def _record(agent: GlucoseAgent, action: GlucoseInputs | None = None) -> dict[str, Any]:
    return {
        "episodes": len(agent.dataset),
        "action": None if action is None else action.key,
        "protocol": None if action is None else dict(action.protocol),
        "model_names": [fit.model.name for fit in agent.belief.fits],
        "model_weights": agent.belief.model_weights,
        "bic": [-2 * fit.parameters.log_evidence for fit in agent.belief.fits],
        "map_model": agent.get_model().name,
        "map_parameters": agent.belief.map_fit.parameters.map_parameters,
        "gate_error": agent.initial_gate_error if action is None else agent.last_predictive_error,
        "expansions": agent.expansion_count,
        "genealogy": {fit.model.name: fit.model.metadata.get("genealogy", {}) for fit in agent.belief.fits},
    }


def run_glucose(environment: GlucoseBenchEnvironment, agent: GlucoseAgent, *, budget: int,
                cuts: Sequence[int] = (45,), design: str = "model", prediction: str = "bma",
                no_m_open: bool = False, icl_client=None, evaluate: bool = True) -> dict[str, Any]:
    """Acquire, snapshot forecasters at budgets 2/3/4/6, then freeze and score.

    Scores are glucosebench's per-channel training-variance nMSE, averaged
    arithmetically over the 12 queries at each budget and cut. Persistence is
    always scored; ICL (+MDA design) is scored when ``icl_client`` is given.
    """
    if design not in {"model", "random"}:
        raise ValueError("glucose design must be model or random")
    if no_m_open:
        agent.proposer = None
    snapshots = {len(agent.dataset): GlucoseForecaster.from_agent(agent, prediction)}
    history = [_record(agent)]
    design_rng = agent.random_design_rng()
    for _ in range(budget):
        if design == "random":
            candidates = agent.candidate_actions()
            action = candidates[int(design_rng.integers(len(candidates)))]
        else:
            action = agent.design()
        delivered, observation = environment.step(action)
        agent.update_state(delivered, observation)
        history.append(_record(agent, action))
        if len(agent.dataset) in BUDGETS:
            snapshots[len(agent.dataset)] = GlucoseForecaster.from_agent(agent, prediction)
    result = {"patient": environment.patient, "replicate": environment.replicate, "design": design,
              "prediction": prediction, "no_m_open": no_m_open, "cuts": list(cuts),
              "history": history, "evaluation_performed": evaluate}
    if not evaluate:
        return result
    gb = _glucosebench()
    evaluator = environment.freeze()
    predictors = {"mda": snapshots, "persistence": {b: gb.persistence for b in snapshots}}
    if icl_client is not None:
        icl = make_glucose_icl_forecaster(icl_client)
        predictors["icl_mda_design"] = {b: icl for b in snapshots}
    result["scores"] = {}
    for method, by_budget in predictors.items():
        rows = []
        for episodes, predictor in sorted(by_budget.items()):
            rows.extend(evaluator.evaluate(predictor, budget=episodes, cuts=tuple(cuts)))
        result["scores"][method] = {"summary": gb.summarize(rows), "rows": rows}
    return result
