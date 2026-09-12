"""Public adapter for the pinned LLM-AutoSciLab ChemBench oracle.

The upstream project is not packaged.  Either put its checkout on ``PYTHONPATH``
or set ``AUTOSCILAB_SOURCE_ROOT`` to a checkout of commit
``9a37255ff5432a7155b6cde8985e9654f2f8f272``.  Only its NumPy oracle path is
used; PySR and the broader AutoSciLab dependencies are not required.

The paper/release mixes raw- and log-space likelihood descriptions.  This
adapter explicitly uses a Gaussian likelihood for ``log1p(rate)`` (a practical
approximation to the oracle's 1% multiplicative noise), while model predictions,
predictive checks, and observations exposed to the agent remain raw rates.

No authoritative Exact36 task manifest exists in the pinned source.  Paper-level
aggregation must therefore be given a user-supplied manifest; this module does
not infer a 36-task list from mechanism names or the larger upstream registry.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mda.agent import AgentConfig, ModelDiscoveryAgent
from mda.expressions import SymbolicRegressionProposer
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
LOG_AXES = frozenset(("C_A", "C_B", "Enz"))
_INITIAL_C_A = (0.03, 0.3, 3.0)
# The release does not prescribe companions for its three C_A seeds.  These are
# deliberately documented wrapper defaults, not ChemBench ground-truth fields.
INITIAL_DEFAULTS = {"C_I": 0.0, "C_B": 1.0, "C_P": 0.0, "Enz": 1.0, "T": 310.0, "pH": 7.0}


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


def _sample_actions(count: int, seed: int) -> list[dict[str, float]]:
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
    """Scalar-rate environment with noisy experiments and a clean held-out set."""

    def __init__(self, domain_id: str, difficulty: str = "easy", seed: int = 0, query_count: int = 1000):
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
        # Seed 42 matches upstream evaluation. Continuous draws are disjoint from
        # the three fixed training points except for a measure-zero coincidence.
        self.query_actions = _sample_actions(query_count, 42)

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

    def evaluation_metrics(self, predictor: Callable[[Mapping[str, float]], Any]) -> dict[str, float]:
        truth = np.asarray([self.evaluation_oracle.run(action).measurement for action in self.query_actions])
        predicted = np.asarray([float(np.asarray(predictor(action))) for action in self.query_actions])
        predicted = np.maximum(predicted, 0.0)
        log_error = np.log1p(predicted) - np.log1p(truth)
        variance = float(np.var(np.log1p(truth)))
        msle = float(np.mean(log_error**2))
        return {"nmsle": msle / max(variance, np.finfo(float).eps), "rmsle": float(np.sqrt(msle))}

    def evaluate_predictor(self, predictor: Callable[[Mapping[str, float]], Any]) -> float:
        """Return held-out log1p MSE normalized by clean log-rate variance."""
        return self.evaluation_metrics(predictor)["nmsle"]

    def evaluate_model(self, model: CandidateModel, parameters=None) -> dict[str, Any]:
        """Constant-refit recovery on fixed grids, isolated from agent fitting."""
        del parameters  # Recovery refits constants rather than scoring their learned values.
        from .recovery import canonical_symbolic_candidate, evaluate_recovery
        canonical = canonical_symbolic_candidate(model.structure, VARIABLE_NAMES, model.prior)
        truth = lambda action: np.asarray(self.evaluation_oracle.run(action).measurement)
        parameters, rmsle, recovered = evaluate_recovery(
            canonical, (_sample_actions(128, 43), _sample_actions(128, 44)),
            (truth, truth), tolerance=.02,
        )
        return {"parameters": parameters, "rmsle": rmsle, "recovered": recovered,
                "canonical_structure": canonical.structure}


def _generic_models() -> list[CandidateModel]:
    constant = CandidateModel(
        "constant", UniformPrior(("a",), np.array([0.0]), np.array([100.0])),
        lambda theta, action: np.asarray(theta[0]), structure="a",
    )
    names = ("a",) + tuple(f"b_{name}" for name in VARIABLE_NAMES)
    linear = CandidateModel(
        "linear", UniformPrior(names, np.full(8, -10.0), np.full(8, 10.0)),
        lambda theta, action: np.asarray(theta[0] + sum(theta[i + 1] * float(action[name]) for i, name in enumerate(VARIABLE_NAMES))),
        structure="a + " + " + ".join(f"b_{name}*{name}" for name in VARIABLE_NAMES),
    )
    return [constant, linear]


def make_agent(
    environment: ChemBenchEnvironment,
    client: Any,
    *,
    seed: int = 0,
    smoke: bool = False,
    experiment_budget: int = 60,
) -> ModelDiscoveryAgent:
    """Construct the paper-configured discovery agent (without running it)."""
    if client is None and not smoke:
        raise ValueError("client=None is permitted only when smoke=True")
    proposer = None if client is None else SymbolicRegressionProposer(client, variable_names=VARIABLE_NAMES)
    lower = [PARAM_RANGES[name][0] for name in VARIABLE_NAMES]
    upper = [PARAM_RANGES[name][1] for name in VARIABLE_NAMES]
    optimizer = CMAESOptimizer(
        lower, upper, log_axes=[name in LOG_AXES for name in VARIABLE_NAMES],
        decode=lambda values: {name: float(values[i]) for i, name in enumerate(VARIABLE_NAMES)},
        max_evaluations=20 if smoke else 200,
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

    return ModelDiscoveryAgent(
        context=environment.context,
        initial_data=environment.initial_data,
        initial_models=_generic_models() if smoke else [],
        likelihood=log1p_likelihood,
        actions=environment.actions,
        observation_covariance=0.01**2,
        config=AgentConfig(
            model_cap=16, new_models_per_expansion=4, model_search_rounds=2,
            expansion_cap=4, occam_penalty=2.0, minimum_model_cap=6,
            smc=SMCConfig(parameter_particles=6 if smoke else 100, target_ess_fraction=0.6, rejuvenation_steps=3),
        ),
        proposer=proposer, query_actions=environment.query_actions,
        optimizer=optimizer, experiment_budget=experiment_budget, seed=seed,
    )
