"""Public adapter for the pinned LLM-AutoSciLab ChemBench oracle (paper v5 App. B).

The upstream project is not packaged.  Either put its checkout on ``PYTHONPATH``
or set ``AUTOSCILAB_SOURCE_ROOT`` to a checkout of commit
``9a37255ff5432a7155b6cde8985e9654f2f8f272``.  Only its NumPy oracle path is
used; PySR and the broader AutoSciLab dependencies are not required.

v5 evaluates all 57 standard worlds (nine canonical plus 48 compound
mechanisms; upstream ``CHEM_BENCHMARK_DOMAINS``) at medium difficulty with 1%
multiplicative noise. Its primary arm starts from the nine canonical laws plus
one 12-proposal LLM batch, fits each law by bounded multistart least squares in
log-rate space, weights laws by profiled BIC (no parameter-count prior), and
predicts with the highest-weight law. The SMC control keeps 100 particles and
``p(m) ~ exp(-2 k)``. Model predictions, predictive checks, and observations
exposed to the agent remain raw rates.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mda.agent import AgentConfig, ModelDiscoveryAgent
from mda.evidence import BICEvidence
from mda.expressions import SymbolicRegressionProposer, canonical_expression, compile_expression
from mda.likelihoods import DeterministicGaussianLikelihood
from mda.optimizers import CMAESOptimizer
from mda.smc import SMCConfig
from mda.types import CandidateModel, Datum, UniformPrior

VARIABLE_NAMES = ("C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH")
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "C_A": (0.01, 100.0),
    "C_I": (0.0, 50.0),
    "C_B": (0.01, 100.0),
    "C_P": (0.0, 20.0),
    "Enz": (0.01, 10.0),
    "T": (278.0, 368.0),
    "pH": (4.0, 10.0),
}
# v5 log-maps "concentration and enzyme axes"; C_I and C_P have a zero lower
# bound, so like upstream sampling they stay linear.
LOG_AXES = frozenset(("C_A", "C_B", "Enz"))
_INITIAL_C_A = (0.03, 0.3, 3.0)
# v5 Appendix B: D0 holds (C_I, C_B, C_P, Enz, T, pH) = (0, 1, 0, 1, 310, 7).
INITIAL_DEFAULTS = {"C_I": 0.0, "C_B": 1.0, "C_P": 0.0, "Enz": 1.0, "T": 310.0, "pH": 7.0}
PASS_RMSLE = 0.05

# The nine canonical mechanisms of the 57-world benchmark (upstream c0-c9
# without the excluded pH law c4). v5 seeds the pool with them but does not
# print their forms or bounds; these follow the upstream rate functions, and
# the positive bounds are local choices covering every difficulty's values.
_K, _KM = ("k", 0.01, 100.0), ("Km", 0.001, 100.0)
CANONICAL_LAWS: tuple[tuple[str, str, tuple[tuple[str, float, float], ...]], ...] = (
    ("michaelis_menten", "k*Enz*C_A/(Km + C_A)", (_K, _KM)),
    ("competitive_inhibition", "k*Enz*C_A/(Km*(1 + C_I/Ki) + C_A)",
     (_K, _KM, ("Ki", 0.001, 100.0))),
    ("product_inhibition", "k*Enz*C_A/(Km*(1 + C_P/Kp) + C_A)",
     (_K, _KM, ("Kp", 0.001, 100.0))),
    ("arrhenius_temperature", "k*exp(-Ea/8.314*(1/T - 1/310))*Enz*C_A/(Km + C_A)",
     (_K, ("Ea", 1000.0, 200000.0), _KM)),
    ("pingpong_bisubstrate", "k*Enz*C_A*C_B/(KmA*C_B + KmB*C_A + C_A*C_B)",
     (_K, ("KmA", 0.001, 100.0), ("KmB", 0.001, 100.0))),
    ("uncompetitive_inhibition", "k*Enz*C_A/(Km + C_A*(1 + C_I/Ki))",
     (_K, _KM, ("Ki", 0.001, 100.0))),
    ("substrate_inhibition", "k*Enz*C_A/(Km + C_A + C_A**2/Ki)",
     (_K, _KM, ("Ki", 0.001, 1000.0))),
    ("hill_cooperativity", "k*Enz*C_A**n/(K_half**n + C_A**n)",
     (_K, ("K_half", 0.001, 100.0), ("n", 0.1, 8.0))),
    ("noncompetitive_inhibition", "k*Enz*C_A/((1 + C_I/Ki)*(Km + C_A))",
     (_K, _KM, ("Ki", 0.001, 100.0))),
)


def _oracle_class() -> type:
    source_root = os.environ.get("AUTOSCILAB_SOURCE_ROOT")
    if source_root:
        root = str(Path(source_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        from autoscilab.oracle.chembench import ChemBenchOracle
    except ImportError as error:
        raise RuntimeError(
            "ChemBench requires the pinned LLM-AutoSciLab source checkout; set "
            "AUTOSCILAB_SOURCE_ROOT or add that checkout to PYTHONPATH"
        ) from error
    return ChemBenchOracle


def benchmark_worlds() -> list[str]:
    """The 57 standard world IDs evaluated by v5 (requires the pinned checkout)."""
    _oracle_class()
    from autoscilab.oracle.chembench import CHEM_BENCHMARK_DOMAINS
    return list(CHEM_BENCHMARK_DOMAINS)


def canonical_models() -> list[CandidateModel]:
    """Compile the nine canonical seed laws as bounded expression models."""
    models = []
    for name, expression, params in CANONICAL_LAWS:
        names = tuple(param[0] for param in params)
        models.append(CandidateModel(
            name=name, structure=canonical_expression(expression),
            prior=UniformPrior(names, np.array([p[1] for p in params]), np.array([p[2] for p in params])),
            predict=compile_expression(expression, VARIABLE_NAMES, names),
        ))
    return models


def _sample_actions(count: int, seed: int | Sequence[int]) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    columns: dict[str, np.ndarray] = {}
    for name, (low, high) in PARAM_RANGES.items():
        columns[name] = (
            np.exp(rng.uniform(np.log(low), np.log(high), count))
            if name in LOG_AXES
            else rng.uniform(low, high, count)
        )
    return [{name: float(columns[name][i]) for name in VARIABLE_NAMES} for i in range(count)]


class ChemBenchEnvironment:
    """Scalar-rate environment with noisy experiments and a clean held-out set.

    Held-out references are the clean primary law, as in upstream
    ``evaluate_law`` (no noise and no small secondary T/pH effects). Query
    conditions depend on ``seed``, as v5 states; the rendered paper gives no
    query count, so the default keeps upstream's 1000.
    """

    def __init__(self, domain_id: str, difficulty: str = "medium", seed: int = 0, query_count: int = 1000):
        if query_count < 1:
            raise ValueError("query_count must be positive")
        oracle_type = _oracle_class()
        self.oracle = oracle_type(domain_id, difficulty, noise_level=0.01, law_version="v0", grammar_mode="none")
        self.evaluation_oracle = oracle_type(
            domain_id, difficulty, noise_level=0.0, law_version="v0", grammar_mode="none"
        )
        self.domain_id, self.difficulty, self.seed = domain_id, difficulty, seed
        self.context = (
            "Discover a scalar rate law from seven measured inputs: C_A, C_I, C_B, C_P, "
            "Enz, T, and pH. Inputs and ranges are public; the mechanism and law parameters "
            "are unknown. Return a generic algebraic predictive model of the raw rate."
        )
        initial_actions = [dict(INITIAL_DEFAULTS, C_A=value) for value in _INITIAL_C_A]
        self.initial_data = [Datum(action, np.asarray(self.oracle.run(action).measurement)) for action in initial_actions]
        self.actions: list[dict[str, float]] = []  # designs are continuous
        # Upstream evaluates at seed 42; v5 draws query conditions per seed.
        # Continuous draws are disjoint from the three fixed training points
        # except for a measure-zero coincidence.
        self.query_actions = _sample_actions(query_count, (42, seed))

    def step(self, action: Mapping[str, float]) -> np.ndarray:
        normalized = self._validate_action(action)
        return np.asarray(self.oracle.run(normalized).measurement)

    @staticmethod
    def _validate_action(action: Mapping[str, float]) -> dict[str, float]:
        if set(action) != set(VARIABLE_NAMES):
            raise ValueError(f"action must contain exactly {VARIABLE_NAMES}")
        result = {name: float(action[name]) for name in VARIABLE_NAMES}
        if any(not np.isfinite(value) or not PARAM_RANGES[name][0] <= value <= PARAM_RANGES[name][1]
               for name, value in result.items()):
            raise ValueError("action values must be finite and within PARAM_RANGES")
        return result

    def _reference(self, action: Mapping[str, float]) -> float:
        oracle = self.evaluation_oracle
        rate = getattr(oracle, "_rate_fn", None)
        if rate is None:  # oracles without the upstream internals
            return float(oracle.run(action).measurement)
        return float(rate(oracle._law_params, *(action[name] for name in VARIABLE_NAMES)))

    def evaluation_metrics(self, predictor: Callable[[Mapping[str, float]], Any]) -> dict[str, Any]:
        """nMSLE (Eq. chemnmse), RMSLE, and the benchmark pass flag RMSLE < 0.05.

        A law that cannot be evaluated over the test range counts as a failure
        (infinite errors). Negative rate predictions are clipped to zero.
        """
        truth = np.asarray([self._reference(action) for action in self.query_actions])
        try:
            with np.errstate(all="ignore"):
                predicted = np.asarray([float(np.asarray(predictor(action))) for action in self.query_actions])
        except (ArithmeticError, TypeError, ValueError):
            predicted = np.full(len(truth), np.nan)
        if not np.all(np.isfinite(predicted)):
            return {"nmsle": np.inf, "rmsle": np.inf, "passed": False}
        predicted = np.maximum(predicted, 0.0)
        log_error = np.log1p(predicted) - np.log1p(truth)
        variance = float(np.var(np.log1p(truth)))
        msle = float(np.mean(log_error**2))
        rmsle = float(np.sqrt(msle))
        return {"nmsle": msle / max(variance, np.finfo(float).eps), "rmsle": rmsle,
                "passed": bool(rmsle < PASS_RMSLE)}

    def evaluate_predictor(self, predictor: Callable[[Mapping[str, float]], Any]) -> float:
        """Return held-out log1p MSE normalized by clean log-rate variance."""
        return self.evaluation_metrics(predictor)["nmsle"]

    def evaluate_model(self, model: CandidateModel, parameters=None) -> dict[str, Any]:
        """Constant-refit recovery on fixed grids, isolated from agent fitting.

        A v4 diagnostic: v5 reports predictive pass rates, not symbolic recovery.
        """
        del parameters  # Recovery refits constants rather than scoring their learned values.
        from .recovery import canonical_symbolic_candidate, evaluate_recovery
        canonical = canonical_symbolic_candidate(model.structure, VARIABLE_NAMES, model.prior)
        # The clean primary law, as for the held-out metrics: the oracle's
        # measurement carries noise and secondary T/pH effects that put even
        # the true law near the recovery tolerance.
        truth = lambda action: np.asarray(self._reference(action))
        parameters, rmsle, recovered = evaluate_recovery(
            canonical, (_sample_actions(128, 43), _sample_actions(128, 44)),
            (truth, truth), tolerance=.02,
        )
        return {"parameters": parameters, "rmsle": rmsle, "recovered": recovered,
                "canonical_structure": canonical.structure}


def make_agent(
    environment: ChemBenchEnvironment,
    client: Any,
    *,
    seed: int = 0,
    smoke: bool = False,
    experiment_budget: int = 22,
    inference: str = "bic",
    llm_initialization: bool = True,
) -> ModelDiscoveryAgent:
    """Construct the v5 ChemBench agent (without running it).

    ``inference="bic"`` is the primary arm; ``"smc"`` is the SMC control with
    100 particles and a 2-nat-per-parameter prior. Both fit in log-rate space.
    The pool starts from the nine canonical laws plus, when a client is given
    and ``llm_initialization`` holds, one 12-proposal batch; expansions request
    six laws, the cap is 24, and a confident (weight > 0.9) adequate (median
    relative residual <= 0.05) fit shrinks it to six. The first expansion of
    an update is gated on the preceding fit's cached residuals. Without a
    client (smoke, or the fixed nine-law control) no LLM call is made.
    ``model_search_rounds=2`` and ``expansion_cap=4`` keep v4's values, which
    v5 does not restate.
    """
    if client is None and not smoke and llm_initialization:
        raise ValueError("client=None is permitted only for smoke or llm_initialization=False")
    if inference not in {"bic", "smc"}:
        raise ValueError("inference must be bic or smc")
    proposer = None
    if client is not None and llm_initialization:
        proposer = SymbolicRegressionProposer(client, variable_names=VARIABLE_NAMES)
    lower = [PARAM_RANGES[name][0] for name in VARIABLE_NAMES]
    upper = [PARAM_RANGES[name][1] for name in VARIABLE_NAMES]
    optimizer = CMAESOptimizer(
        lower, upper, log_axes=[name in LOG_AXES for name in VARIABLE_NAMES],
        decode=lambda values: {name: float(values[i]) for i, name in enumerate(VARIABLE_NAMES)},
        max_evaluations=20 if smoke else 48,
    )

    # Compare log1p(raw prediction) to log1p(raw observation).  Keeping the
    # transform here means predictive checks and target predictions stay raw.
    gaussian = DeterministicGaussianLikelihood(0.01)
    def log1p_likelihood(dataset, model, theta, rng):
        transformed = []
        for datum in dataset:
            prediction = float(np.asarray(model.observation_prediction(theta, datum.action)))
            if prediction <= -1.0:
                return -np.inf
            transformed.append(Datum(datum.action, np.asarray(np.log1p(float(datum.observation)))))
        proxy = CandidateModel(model.name, model.prior, lambda t, a: np.log1p(model.observation_prediction(t, a)), model.structure)
        return gaussian(transformed, proxy, theta, rng)

    evidence = None
    if inference == "bic":
        # Profiled BIC with sigma_n = max(0.03 |y_n|, 0.01) on log1p(rate).
        evidence = BICEvidence("profiled", relative_noise=0.03, noise_floor=0.01,
                               starts=2 if smoke else 8, transform=np.log1p)
    return ModelDiscoveryAgent(
        context=environment.context,
        initial_data=environment.initial_data,
        initial_models=canonical_models(),
        likelihood=log1p_likelihood,
        actions=environment.actions,
        # V(a) is unwhitened between-model variance; a common scalar scale
        # leaves its argmax unchanged.
        observation_covariance=0.01**2,
        config=AgentConfig(
            model_cap=24, new_models_per_expansion=6, model_search_rounds=2,
            expansion_cap=4, error_threshold=0.05,
            occam_penalty=0.0 if inference == "bic" else 2.0,
            minimum_model_cap=6, shrink_confidence=0.9, initial_proposals=12,
            expansion_gate="cached",
            smc=SMCConfig(parameter_particles=6 if smoke else 100, target_ess_fraction=0.6, rejuvenation_steps=3),
        ),
        proposer=proposer, query_actions=environment.query_actions,
        optimizer=optimizer, experiment_budget=experiment_budget, evidence=evidence, seed=seed,
    )
