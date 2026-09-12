"""Boxing Apprentice baseline from paper Appendix H.3.2.

Unlike MDA, the apprentice asks the language model to select an experiment and
keeps exactly one model.  Parameter inference and model evidence remain the
same full-batch SMC computation used by :class:`ModelDiscoveryAgent`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mda.agent import AgentConfig, ModelDiscoveryAgent
from mda.checks import predictive_check, residual_report
from mda.expressions import SymbolicRegressionProposer
from mda.smc import SMCConfig
from mda.types import Datum

from .boxing import BoxingEnvironment, _NumPyroProposer, _generic_model


class BoxingApprentice(ModelDiscoveryAgent):
    """Single-model Boxing baseline with LLM experiment selection/refinement."""

    def __init__(self, *args, design_client=None, environment: BoxingEnvironment, **kwargs):
        self.design_client = design_client
        self.environment = environment
        super().__init__(*args, **kwargs)

    def _proposal_context(self) -> str:
        phase = "explore" if self.experiments_run == 0 else "refine"
        return (f"{self.context}\nCurrent phase: {phase}. Refine the single current model "
                "using its residuals; a refinement is requested every round.\n"
                f"Experiment budget remaining: {self.experiment_budget - self.experiments_run}.")

    def _features(self, action: Any) -> Any:
        if self.environment.domain in {"irt", "location", "survival"}:
            return self.environment.action_features[int(action)].tolist()
        return action

    def design(self) -> Any:
        if self.experiment_budget is not None and self.experiments_run >= self.experiment_budget:
            raise ValueError("the experiment budget is exhausted")
        available = self.available_actions()
        if not available:
            raise ValueError("no unrepeated experiment designs remain")
        # Smoke is deliberately provider-free.  Its seeded random choice merely
        # exercises benchmark plumbing and is not reported as an LLM baseline.
        if self.design_client is None:
            return available[int(self.rng.integers(len(available)))]

        rows = [{"choice": i, "action": action, "features": self._features(action)}
                for i, action in enumerate(available)]
        observed = [{"action": d.action, "features": self._features(d.action),
                     "observation": np.asarray(d.observation).tolist()}
                    for d in self.dataset]
        prompt = (
            f"{self.context}\nCurrent model source/structure:\n{self.models[0].structure}\n"
            f"Observed data (only completed experiments): {observed}\n"
            f"Available unrepeated feature rows: {rows}\n"
            "Choose exactly one next experiment. Return its integer choice field, not an "
            "action or feature-row index. Do not invent a choice."
        )
        schema = {"type": "object", "properties": {"choice": {"type": "integer"}},
                  "required": ["choice"], "additionalProperties": False}
        answer = self.design_client.ask_json(
            "Select one informative experiment from the supplied menu; JSON only.", prompt,
            schema=schema, name="apprentice_design", max_tokens=200,
        )
        choice = answer.get("choice") if isinstance(answer, dict) else None
        if isinstance(choice, bool) or not isinstance(choice, (int, np.integer)):
            raise ValueError("apprentice returned a non-integer experiment choice")
        if not 0 <= int(choice) < len(available):
            raise ValueError("apprentice returned an unavailable experiment choice")
        action = available[int(choice)]
        if self.action_key(action) in self.used_action_keys:
            raise ValueError("apprentice returned a repeated experiment design")
        return action

    def update_state(self, action: Any, observation: np.ndarray) -> float:
        """Append the result, then request exactly one refinement every round."""
        if self.action_key(action) not in {self.action_key(a) for a in self.available_actions()}:
            raise ValueError("update action is unavailable or repeated")
        fresh = Datum(action, np.asarray(observation))
        error = predictive_check(
            self.belief, fresh, self.dataset, target=self.target,
            floor=self.config.predictive_floor, backtest=self.config.backtest,
        )
        self.last_predictive_error = error
        self.dataset.append(fresh)
        self.used_action_keys.add(self.action_key(action))
        self.experiments_run += 1

        old_models = self.models
        reports = [residual_report(
            fit, self.dataset, target=self.target, floor=self.config.predictive_floor,
        ) for fit in self.belief.fits]
        proposals = [] if self.proposer is None else self.proposer.propose(
            context=self._proposal_context(), dataset=self.dataset,
            current_models=old_models, residuals=reports, count=1,
        )
        novel = next((model for model in proposals[:1]
                      if model.key not in {old.key for old in old_models}), None)
        if novel is not None:
            self.belief = self._fit([novel])
            self.expansion_count += 1
        else:
            self.belief = self._fit(old_models)
        return error


def make_agent(environment: BoxingEnvironment, client, *, seed: int = 0,
               smoke: bool = False, experiment_budget: int = 10,
               allow_code_execution: bool = False):
    """Construct the Appendix H.3.2 apprentice with a one-model pool."""
    if not isinstance(environment, BoxingEnvironment):
        raise TypeError("environment must be a BoxingEnvironment")
    if smoke and client is not None:
        raise ValueError("smoke mode does not use a client")
    if not smoke and client is None:
        raise ValueError("a client is required unless smoke=True")
    latent = environment.domain in {"irt", "location", "survival"}
    if not smoke and latent and not allow_code_execution:
        raise PermissionError("IRT/location/survival NumPyro proposals require allow_code_execution=True")
    proposer = None
    if not smoke:
        proposer = (_NumPyroProposer(client, environment, allow_code_execution) if latent else
                    SymbolicRegressionProposer(client, variable_names={
                        "hyperbolic": ("immediate", "delayed", "days"),
                    }.get(environment.domain, ("x",))))
    particles = 30 if smoke else 1500
    config = AgentConfig(
        model_cap=1, new_models_per_expansion=1, model_search_rounds=1,
        expansion_cap=max(1, experiment_budget), error_threshold=1,
        occam_penalty=1, smc=SMCConfig(parameter_particles=particles,
            rejuvenation_steps=4, target_ess_fraction=.5, max_temperatures=60),
    )
    return BoxingApprentice(
        environment=environment, design_client=None if smoke else client,
        context=environment.context, initial_data=environment.initial_data,
        initial_models=[_generic_model(environment)] if smoke else [],
        proposer=proposer, likelihood=environment.likelihood,
        actions=environment.actions, query_actions=environment.query_actions,
        observation_covariance={"dugongs": .25**2, "location": .5**2}.get(
            environment.domain, .25),
        config=config, experiment_budget=experiment_budget, seed=seed,
    )
