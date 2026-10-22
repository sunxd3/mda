"""The stateful Model Discovery Agent loop (paper Algorithm 2, ``alg:mda`` in v5)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .acquisition import choose_design, value_of_information
from .checks import TargetFn, identity_target, in_sample_check, predictive_check, residual_report
from .evidence import EvidenceBackend
from .posterior import BeliefState, fit_model_posterior
from .programs import (
    EpisodeSpec, FixedProgram, Program, ProgramError, ProgramScore, belief_summary,
    disagreement_report, imagined_runs, trajectory_information,
)
from .proposer import ModelProposer, ProgramProposer
from .smc import Likelihood, SMCConfig
from .types import CandidateModel, Datum


def default_action_key(action: Any) -> str:
    """Exact text key for an experiment.

    ``repr`` of an array keeps eight significant digits, elides long arrays
    and follows the global print options, so distinct designs could collide
    in ``used_action_keys`` and in the optimizer's evaluation table.
    """
    if isinstance(action, np.ndarray):
        return f"ndarray{action.shape}{action.dtype}:{action.tobytes().hex()}"
    if isinstance(action, Mapping):
        items = sorted(action.items(), key=lambda item: repr(item[0]))
        return "{" + ", ".join(f"{key!r}: {default_action_key(value)}" for key, value in items) + "}"
    if isinstance(action, (list, tuple)):
        return f"{type(action).__name__}({', '.join(default_action_key(value) for value in action)})"
    return repr(action)


@dataclass(frozen=True)
class AgentConfig:
    model_cap: int = 16
    new_models_per_expansion: int = 4
    model_search_rounds: int = 2
    expansion_cap: int = 4
    error_threshold: float = 0.05
    occam_penalty: float = 0.0
    predictive_floor: float | np.ndarray = 1e-6
    backtest: bool = True
    design_flavour: str = "model"
    repeat_designs: bool = False
    smc: SMCConfig = field(default_factory=SMCConfig)
    minimum_model_cap: int | None = None
    shrink_confidence: float = 0.9
    initial_proposals: int = 0
    expansion_gate: str = "prequential"
    repeat_when_exhausted: bool = False
    # Experiment programs (design_flavour="program"; beyond v5, see mda.programs).
    program_cap: int = 8
    programs_per_proposal: int = 2
    program_proposal_cap: int = 4
    program_draws: int = 128

    def __post_init__(self) -> None:
        if self.model_cap < 1 or self.new_models_per_expansion < 1:
            raise ValueError("model pool sizes must be positive")
        if self.model_search_rounds < 0 or self.expansion_cap < 0:
            raise ValueError("search and expansion limits cannot be negative")
        if self.error_threshold < 0 or self.occam_penalty < 0:
            raise ValueError("error threshold and Occam penalty cannot be negative")
        if self.design_flavour not in {"model", "joint", "task", "task_full_observation", "program"}:
            raise ValueError("unknown design_flavour")
        if min(self.program_cap, self.programs_per_proposal, self.program_draws) < 1 or self.program_proposal_cap < 0:
            raise ValueError("program pool sizes and draws must be positive")
        if self.minimum_model_cap is not None and not 1 <= self.minimum_model_cap <= self.model_cap:
            raise ValueError("minimum_model_cap must be between 1 and model_cap")
        if not 0 < self.shrink_confidence <= 1:
            raise ValueError("shrink_confidence must be in (0, 1]")
        if self.initial_proposals < 0:
            raise ValueError("initial_proposals cannot be negative")
        if self.expansion_gate not in {"prequential", "cached"}:
            raise ValueError("expansion_gate must be prequential or cached")


class ModelDiscoveryAgent:
    """M-open model discovery with value-of-information design.

    The LLM is used only by ``proposer``. Every weight, evidence, design score,
    and prediction is numerical, preserving the paper's core division of labor.
    ``evidence`` selects the per-structure evidence backend: full-history
    tempered SMC with ``config.smc`` by default, or e.g. ``BICEvidence`` or
    ``RecursiveEvidence`` from :mod:`mda.evidence`.
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
        task_noise_covariance: float | np.ndarray | None = None,
        optimizer: Callable[[Callable[[Any], float], np.random.Generator], Any] | None = None,
        experiment_budget: int | None = None,
        action_key: Callable[[Any], str] = default_action_key,
        evidence: EvidenceBackend | None = None,
        check: Callable[[BeliefState, Datum, Sequence[Datum]], float] | None = None,
        seed: int = 0,
        episode: EpisodeSpec | None = None,
        programs: Sequence[Program] = (),
        program_proposer: ProgramProposer | None = None,
    ) -> None:
        self.context = context
        self.evidence = evidence
        self.check = check
        self.dataset = list(initial_data)
        self.likelihood = likelihood
        self.actions = list(actions)
        self.observation_covariance = observation_covariance
        self.config = config
        self.proposer = proposer
        self.target = target
        self.query_actions = list(query_actions)
        self.target_covariance = target_covariance
        self.task_noise_covariance = task_noise_covariance
        self.optimizer = optimizer
        if experiment_budget is not None and experiment_budget < 0:
            raise ValueError("experiment_budget cannot be negative")
        self.experiment_budget = experiment_budget
        self.experiments_run = 0
        self.active_model_cap = config.model_cap
        self.action_key = action_key
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.expansion_count = 0
        self.used_action_keys = {action_key(datum.action) for datum in self.dataset}
        if config.design_flavour in {"task", "task_full_observation"} and not self.query_actions:
            raise ValueError("task value of information needs query_actions")
        self.last_design_scores: np.ndarray | None = None
        self.last_predictive_error: float | None = None
        if config.design_flavour == "program" and (
            episode is None or not (programs or program_proposer is not None)
        ):
            raise ValueError("program design needs an episode and seed programs or a program proposer")
        self.episode = episode
        self.programs: list[Program] = list(programs)
        self.program_proposer = program_proposer
        self.program_proposals = 0
        self._program_proposal_models: frozenset[str] | None = None
        self.last_program_scores: list[tuple[str, float]] = []

        # init-hyp-space: a configured seed library plus, optionally, one LLM
        # batch conditioned only on the context and D0, merged without duplicates.
        models = list(initial_models)
        if not models and proposer is None:
            raise ValueError("initial_models cannot be empty without a proposer")
        if proposer is not None and (not models or config.initial_proposals):
            keys = {model.key for model in models}
            for proposal in proposer.propose(
                context=self._proposal_context(),
                dataset=self.dataset,
                current_models=(),
                residuals=(),
                count=config.initial_proposals or config.model_cap,
            ):
                if proposal.key not in keys:
                    models.append(proposal)
                    keys.add(proposal.key)
        if not models:
            raise ValueError("the proposer returned no valid initial models")
        # v5 Algorithm 2 applies adapt-and-prune only at the end of updates.
        self.belief = self._fit(models)

    @property
    def models(self) -> tuple[CandidateModel, ...]:
        return tuple(fit.model for fit in self.belief.fits)

    def _fit(self, models: Sequence[CandidateModel], *, reuse: bool = False) -> BeliefState:
        """Model posterior on the current data.

        With ``reuse`` the current fits are kept for their structures; this is
        valid only when they were computed on exactly ``self.dataset``.
        """
        return fit_model_posterior(
            self.dataset,
            models,
            self.likelihood,
            self.config.smc,
            model_cap=self.active_model_cap,
            occam_penalty=self.config.occam_penalty,
            rng=self.rng,
            evidence=self.evidence,
            reuse={fit.model.key: fit.parameters for fit in self.belief.fits} if reuse else None,
        )

    def _proposal_context(self) -> str:
        status = "Current phase: explore (the pool is being initialized or has failed its check)."
        if self.experiment_budget is not None:
            status += f"\nExperiment budget remaining: {max(0, self.experiment_budget - self.experiments_run)}."
        return f"{self.context}\n\n{status}"

    def _adapt_pool(self) -> None:
        """Optional ``adapt-and-prune``: shrink only if MAP weight and fit both pass.

        v5 ChemBench shrinks to 6 when the largest weight exceeds 0.9 and the
        median relative fit residual is at most 0.05; otherwise the cap is the
        full ``model_cap``. Raising the cap does not restore pruned models.
        """
        if self.config.minimum_model_cap is None:
            return
        error = in_sample_check(
            self.belief, self.dataset, target=self.target, floor=self.config.predictive_floor,
        )
        confident = self.belief.model_weights[self.belief.map_index] > self.config.shrink_confidence
        self.active_model_cap = (
            self.config.minimum_model_cap
            if confident and error <= self.config.error_threshold else self.config.model_cap
        )
        if len(self.belief.fits) > self.active_model_cap:
            order = np.argsort(-self.belief.model_weights, kind="stable")[:self.active_model_cap]
            weights = self.belief.model_weights[order]
            retained = float(weights.sum())
            self.belief = BeliefState(
                tuple(self.belief.fits[index] for index in order), weights / retained,
                1 - (1 - self.belief.discarded_mass) * retained,
                self.belief.log_mixture_evidence,
            )

    def available_actions(self) -> list[Any]:
        if self.config.repeat_designs:
            return list(self.actions)
        unused = [action for action in self.actions if self.action_key(action) not in self.used_action_keys]
        if not unused and self.config.repeat_when_exhausted:
            return list(self.actions)
        return unused

    def random_design_rng(self) -> np.random.Generator:
        """A generator that depends only on the seed, not on inference.

        Random-design controls drawn from it share their design history across
        inference arms, as v5's ChemBench controls require.
        """
        return np.random.default_rng(np.random.SeedSequence([self.seed, 0xD5]))

    def design(self) -> Any:
        if self.experiment_budget is not None and self.experiments_run >= self.experiment_budget:
            raise ValueError("the experiment budget is exhausted")
        if self.config.design_flavour in {"task", "task_full_observation"} and not self.query_actions:
            raise ValueError("task value of information needs query_actions")
        options = dict(
            flavour=self.config.design_flavour,
            observation_covariance=(
                self.task_noise_covariance
                if self.config.design_flavour == "task" and self.task_noise_covariance is not None
                else self.observation_covariance
            ),
            query_actions=self.query_actions,
            target_covariance=self.target_covariance,
        )
        if self.config.design_flavour == "program":
            return self._design_program()
        if self.optimizer is not None:
            scores = []
            evaluated = {}

            def score(action):
                key = self.action_key(action)
                value = (
                    -np.inf if not self.config.repeat_designs and key in self.used_action_keys
                    else value_of_information(action, self.belief, **options)
                )
                scores.append(value)
                evaluated[key] = value
                return value

            action = self.optimizer(score, self.rng)
            if not np.isfinite(evaluated.get(self.action_key(action), -np.inf)):
                raise ValueError("optimizer must return an evaluated, finite, non-repeated design")
            scores = np.asarray(scores)
        else:
            action, scores = choose_design(self.available_actions(), self.belief, **options)
        self.last_design_scores = scores
        return action

    def _design_program(self) -> Program:
        """Choose the program whose closed-loop trajectory is most informative about M.

        Every candidate is scored numerically under one fresh seed (common
        random numbers); the LLM only writes candidates. The program proposer
        is called when the model pool differs from the one it last saw (so at
        the first program design and after expansion or pruning), at most
        ``program_proposal_cap`` times. The pool keeps the ``program_cap`` best
        programs by the current score; failing programs and fixed programs
        whose inputs were already run (unless ``repeat_designs``) leave it.
        """
        missing = [model.name for model in self.models if model.sample_step is None]
        if missing:
            raise ValueError(f"program design needs sample_step on every model: {missing}")
        episode = self.episode
        seed = int(self.rng.integers(2**63 - 1))
        summary = belief_summary(self.belief)

        def score(program: Program) -> ProgramScore | None:
            try:
                return trajectory_information(
                    program, self.belief, self.likelihood, episode,
                    draws=self.config.program_draws, seed=seed, summary=summary,
                )
            except (ProgramError, ValueError, ArithmeticError):
                return None

        def used(program: Program) -> bool:
            return (
                not self.config.repeat_designs and isinstance(program, FixedProgram)
                and self.action_key(np.clip(program.inputs, episode.lower, episode.upper))
                in self.used_action_keys
            )

        scored: dict[str, tuple[Program, ProgramScore | None]] = {}
        for program in self.programs:
            if program.key not in scored and not used(program):
                scored[program.key] = (program, score(program))
        model_keys = frozenset(model.key for model in self.models)
        if (
            self.program_proposer is not None
            and model_keys != self._program_proposal_models
            and self.program_proposals < self.config.program_proposal_cap
        ):
            valid = [(program, result) for program, result in scored.values() if result is not None]
            best = max(valid, key=lambda item: item[1].bits)[1] if valid else None
            proposals = self.program_proposer.propose(
                context=self._proposal_context(),
                dataset=self.dataset,
                current_models=self.models,
                disagreements={
                    "predictions": disagreement_report(self.belief, episode),
                    "imagined_runs_of_best_program": (
                        imagined_runs(best, self.belief) if best is not None else {}
                    ),
                },
                current_programs=[(program, result.bits) for program, result in valid],
                count=self.config.programs_per_proposal,
            )
            self.program_proposals += 1
            self._program_proposal_models = model_keys
            for program in proposals:
                if program.key not in scored and not used(program):
                    scored[program.key] = (program, score(program))
        ranked = sorted(
            ((program, result) for program, result in scored.values() if result is not None),
            key=lambda item: -item[1].bits,
        )
        self.programs = [program for program, _ in ranked[: self.config.program_cap]]
        self.last_program_scores = [(program.label, result.bits) for program, result in ranked]
        self.last_design_scores = np.asarray([result.bits for _, result in ranked])
        if not ranked:
            raise ValueError("no candidate program could be scored")
        return ranked[0][0]

    def update_state(self, action: Any, observation: np.ndarray) -> float:
        """Absorb one result, expanding only after a failed prequential check.

        Follows v5 Algorithm 2: score the fresh datum against the pre-update
        belief, append it, refit the current pool on all data, and only then
        run up to ``model_search_rounds`` expansions. Proposals therefore see
        residuals of the refitted pool. Each re-judge refits only the new
        structures; the others keep their fits on this same dataset.

        ``expansion_gate="cached"`` instead gates the first expansion on the
        in-sample residuals of the preceding fit, as v5's executed ChemBench
        agent does, so a surprising datum can trigger expansion one update
        later. The returned value is always the prequential error.
        """
        fresh = Datum(action, observation)
        error = self._check(fresh)
        self.last_predictive_error = error
        gate_error = error if self.config.expansion_gate == "prequential" else in_sample_check(
            self.belief, self.dataset, target=self.target, floor=self.config.predictive_floor,
        )
        key = self.action_key(action)
        key_was_used = key in self.used_action_keys
        previous_cap = self.active_model_cap
        self.dataset.append(fresh)
        self.used_action_keys.add(key)
        self.experiments_run += 1
        if gate_error > self.config.error_threshold:
            self.active_model_cap = self.config.model_cap
        try:
            belief = self._fit(self.models)
        except Exception:
            # Leave the agent exactly as it was, so that the failure can be
            # reported and a retry does not absorb the datum twice.
            self.dataset.pop()
            if not key_was_used:
                self.used_action_keys.discard(key)
            self.experiments_run -= 1
            self.active_model_cap = previous_cap
            raise
        self.belief = belief

        check_error = gate_error
        try:
            for _ in range(self.config.model_search_rounds):
                if check_error <= self.config.error_threshold or not self._expand():
                    break
                check_error = (
                    in_sample_check(
                        self.belief,
                        self.dataset if self.config.backtest else [fresh],
                        target=self.target,
                        floor=self.config.predictive_floor,
                    )
                    if self.check is None else self.check(self.belief, fresh, self.dataset[:-1])
                )
        finally:
            # The refitted pool is consistent with the data even if a proposer
            # call failed part-way; keep the cap bookkeeping in step with it.
            self._adapt_pool()
        return error

    def _check(self, fresh: Datum) -> float:
        """Pre-update adequacy of the current belief on a fresh datum.

        The default is the paper's relative-residual ``predictive-check``; a
        domain may supply ``check(belief, fresh, history)`` instead, compared
        against the same ``error_threshold``.
        """
        try:
            if self.check is not None:
                return float(self.check(self.belief, fresh, self.dataset))
            return predictive_check(
                self.belief,
                fresh,
                self.dataset,
                target=self.target,
                floor=self.config.predictive_floor,
                backtest=self.config.backtest,
            )
        except (ValueError, ArithmeticError):
            # The current best model cannot even evaluate this experiment
            # (a domain error at the chosen design): the check fails rather
            # than aborting after the experiment has already been run.
            return np.inf

    def _expand(self) -> bool:
        """One ``expand-hyp-space`` call and re-judge; False if no call was allowed.

        As in v5 Algorithm 2, every call counts toward ``expansion_cap``
        (N_e^max), including one whose proposals are all invalid or duplicates;
        the search then continues with the next round.
        """
        if self.expansion_count >= self.config.expansion_cap or self.proposer is None:
            return False
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
            context=self._proposal_context(),
            dataset=self.dataset,
            current_models=self.models,
            residuals=reports,
            count=self.config.new_models_per_expansion,
        )
        self.expansion_count += 1
        existing_keys = {model.key for model in self.models}
        novel = []
        for proposal in proposals:
            if proposal.key not in existing_keys:
                novel.append(proposal)
                existing_keys.add(proposal.key)
        if novel:
            self.belief = self._fit([*self.models, *novel], reuse=True)
        return True

    def get_model(self) -> CandidateModel:
        """Current-data MAP structure, as in v5 Algorithm 2.

        Never a winner selected using held-out truth: best-so-far held-out
        curves belong in the evaluator, not in this selection step.
        """
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
