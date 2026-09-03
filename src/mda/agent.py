"""The stateful Model Discovery Agent loop (paper Algorithm 2)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .acquisition import choose_design
from .checks import TargetFn, identity_target, in_sample_check, predictive_check, residual_report
from .posterior import BeliefState, fit_model_posterior
from .proposer import ModelProposer
from .smc import Likelihood, SMCConfig
from .types import CandidateModel, Datum


@dataclass(frozen=True)
class AgentConfig:
    model_cap: int = 16
    new_models_per_expansion: int = 4
    model_search_rounds: int = 2
    expansion_cap: int = 4
    error_threshold: float = 0.05
    occam_penalty: float = 2.0
    predictive_floor: float | np.ndarray = 1e-6
    backtest: bool = True
    design_flavour: str = "model"
    repeat_designs: bool = False
    smc: SMCConfig = field(default_factory=SMCConfig)

    def __post_init__(self) -> None:
        if self.model_cap < 1 or self.new_models_per_expansion < 1:
            raise ValueError("model pool sizes must be positive")
        if self.model_search_rounds < 0 or self.expansion_cap < 0:
            raise ValueError("search and expansion limits cannot be negative")
        if self.error_threshold < 0 or self.occam_penalty < 0:
            raise ValueError("error threshold and Occam penalty cannot be negative")
        if self.design_flavour not in {"model", "joint", "task"}:
            raise ValueError("design_flavour must be model, joint, or task")


class ModelDiscoveryAgent:
    """Full-batch M-open SMC model discovery with value-of-information design.

    The LLM is used only by ``proposer``. Every weight, evidence, design score,
    and prediction is numerical, preserving the paper's core division of labor.
    """

    def __init__(
        self,
        *,
        context: str,
        initial_data: Sequence[Datum],
        initial_models: Sequence[CandidateModel],
        likelihood: Likelihood,
        actions: Sequence[Any],
        observation_covariance: float | np.ndarray,
        config: AgentConfig = AgentConfig(),
        proposer: ModelProposer | None = None,
        target: TargetFn = identity_target,
        query_actions: Sequence[Any] = (),
        target_covariance: float | np.ndarray | None = None,
        action_key: Callable[[Any], str] = repr,
        seed: int = 0,
    ) -> None:
        self.context = context
        self.dataset = list(initial_data)
        self.likelihood = likelihood
        self.actions = list(actions)
        self.observation_covariance = observation_covariance
        self.config = config
        self.proposer = proposer
        self.target = target
        self.query_actions = list(query_actions)
        self.target_covariance = target_covariance
        self.action_key = action_key
        self.rng = np.random.default_rng(seed)
        self.expansion_count = 0
        self.used_action_keys = {action_key(datum.action) for datum in initial_data}
        self.last_design_scores: np.ndarray | None = None
        self.last_predictive_error: float | None = None

        models = list(initial_models)
        if not models:
            if proposer is None:
                raise ValueError("initial_models cannot be empty without a proposer")
            models = proposer.propose(
                context=context,
                dataset=self.dataset,
                current_models=(),
                residuals=(),
                count=config.model_cap,
            )
        if not models:
            raise ValueError("the proposer returned no valid initial models")
        self.belief = self._fit(models)

    @property
    def models(self) -> tuple[CandidateModel, ...]:
        return tuple(fit.model for fit in self.belief.fits)

    def _fit(self, models: Sequence[CandidateModel]) -> BeliefState:
        return fit_model_posterior(
            self.dataset,
            models,
            self.likelihood,
            self.config.smc,
            model_cap=self.config.model_cap,
            occam_penalty=self.config.occam_penalty,
            rng=self.rng,
        )

    def available_actions(self) -> list[Any]:
        if self.config.repeat_designs:
            return list(self.actions)
        return [action for action in self.actions if self.action_key(action) not in self.used_action_keys]

    def design(self) -> Any:
        action, scores = choose_design(
            self.available_actions(),
            self.belief,
            flavour=self.config.design_flavour,
            observation_covariance=self.observation_covariance,
            query_actions=self.query_actions,
            target_covariance=self.target_covariance,
        )
        self.last_design_scores = scores
        return action

    def update_state(self, action: Any, observation: np.ndarray) -> float:
        """Absorb one result, expanding only after a failed prequential check.

        Algorithm 2 prints the append before Algorithm 8's pre-observation check.
        We evaluate and cache that check first, which is the ordering required by
        the surrounding text and prevents leakage from the fresh outcome.
        """
        fresh = Datum(action, observation)
        error = predictive_check(
            self.belief,
            fresh,
            self.dataset,
            target=self.target,
            floor=self.config.predictive_floor,
            backtest=self.config.backtest,
        )
        self.last_predictive_error = error
        self.dataset.append(fresh)
        self.used_action_keys.add(self.action_key(action))

        expanded = False
        check_error = error
        for _ in range(self.config.model_search_rounds):
            if check_error <= self.config.error_threshold:
                break
            if self.expansion_count >= self.config.expansion_cap or self.proposer is None:
                break
            reports = [
                residual_report(
                    fit,
                    self.dataset,
                    target=self.target,
                    floor=self.config.predictive_floor,
                )
                for fit in self.belief.fits
            ]
            proposals = self.proposer.propose(
                context=self.context,
                dataset=self.dataset,
                current_models=self.models,
                residuals=reports,
                count=self.config.new_models_per_expansion,
            )
            existing_keys = {model.key for model in self.models}
            novel = []
            for proposal in proposals:
                if proposal.key not in existing_keys:
                    novel.append(proposal)
                    existing_keys.add(proposal.key)
            if not novel:
                break
            self.belief = self._fit([*self.models, *novel])
            self.expansion_count += 1
            expanded = True
            check_error = in_sample_check(
                self.belief,
                self.dataset,
                target=self.target,
                floor=self.config.predictive_floor,
            )

        if not expanded:
            self.belief = self._fit(self.models)
        return error

    def get_model(self) -> CandidateModel:
        return self.belief.map_fit.model

    def predict(self, action: Any, flavour: str = "map") -> np.ndarray:
        if flavour == "map":
            fit = self.belief.map_fit
            return fit.model.target_prediction(fit.parameters.map_parameters, action)
        if flavour != "bma":
            raise ValueError("prediction flavour must be map or bma")
        predictions = []
        weights = []
        for model, theta, weight in self.belief.iter_parameter_particles():
            predictions.append(model.target_prediction(theta, action))
            weights.append(weight)
        return np.tensordot(np.asarray(weights), np.stack(predictions), axes=(0, 0))

    def get_predictor(self, flavour: str = "map") -> Callable[[Any], np.ndarray]:
        return lambda action: self.predict(action, flavour)
