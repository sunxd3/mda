"""Experiment programs: adaptive multi-step designs (an extension beyond v5).

A *program* chooses each input of an episode from what the episode has shown
so far: ``step(history, belief) -> next input``. ``design_flavour="program"``
makes ``ModelDiscoveryAgent.design()`` return one. The completed episode is an
ordinary datum, ``Datum(action=realised_inputs, observation=trajectory)``, so
fitting, checks and expansion are unchanged; an ordinary whole-trajectory
action is the special case ``FixedProgram(inputs)``.

Programs are scored by the information the whole trajectory carries about the
model index, estimated by nested Monte Carlo with each imagined episode run
closed loop against one belief particle::

    J(pi) = I(M; Y_1:T | D, pi)
          ~ mean_i [ log p(y_i | m_i) - log sum_m w_m p(y_i | m) ] / log 2,
    p(y_i | m) = sum_{j in m, j != i} w_j p(y_i | theta_j) / sum_{j in m, j != i} w_j,

where ``p(y_i | theta_j)`` is mda's own likelihood of the datum
``(u_i, y_i)``. The estimator is ported from the program-design study: common
random numbers (one seed per decision, shared by every candidate) and
self-exclusion by *value*, so SMC duplicates of the outer particle do not
inflate ``p(y_i | m_i)``. A hypothesis with a single distinct value keeps it.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .posterior import BeliefState
from .types import CandidateModel, Datum, PredictFn, StepSampler

LOG2 = math.log(2.0)
StepBatch = Callable[[Sequence["History"], Mapping[str, Any]], Sequence[float]]


class ProgramError(RuntimeError):
    """A program failed: it raised, returned a non-number, or broke a sandbox limit."""


@dataclass
class History:
    """What ``step`` sees: this episode's inputs and observations so far."""

    inputs: list[float] = field(default_factory=list)
    obs: list[float] = field(default_factory=list)
    horizon: int = 0

    @property
    def t(self) -> int:
        return len(self.inputs)

    def obs_at(self, index: int) -> float | None:
        return self.obs[index] if 0 <= index < len(self.obs) else None

    def last_obs(self) -> float | None:
        return self.obs[-1] if self.obs else None


@dataclass(frozen=True)
class EpisodeSpec:
    """A fixed-length episode of scalar inputs in ``[lower, upper]``.

    Early stopping is not supported: the estimator assumes a fixed horizon.
    """

    steps: int
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.steps < 1 or not (np.isfinite(self.lower) and np.isfinite(self.upper)
                                  and self.lower < self.upper):
            raise ValueError("an episode needs at least one step and finite bounds lower < upper")


class EpisodicEnvironment(Protocol):
    """An environment that can be observed one input at a time within an episode."""

    def start_episode(self) -> None: ...

    def observe(self, value: float) -> float: ...


class Program:
    """A deterministic policy over one episode.

    Subclass and implement ``step`` for a trusted, hand-written program; LLM
    source goes through ``mda.program_sandbox.compile_program`` instead.
    ``session()`` returns a context manager yielding a batch step function;
    the scorer calls it once per episode step with every imagined history.
    """

    name = "program"

    def step(self, history: History, belief: Mapping[str, Any]) -> float:
        raise NotImplementedError

    @property
    def key(self) -> str:
        return f"native:{type(self).__qualname__}:{self.name}"

    @property
    def label(self) -> str:
        return self.name

    def session(self) -> contextlib.AbstractContextManager[StepBatch]:
        return contextlib.nullcontext(
            lambda histories, belief: [self.step(history, belief) for history in histories]
        )

    def __repr__(self) -> str:
        return self.label


class FixedProgram(Program):
    """Issue ``inputs`` regardless of feedback: an ordinary action as a program."""

    name = "fixed"

    def __init__(self, inputs: Sequence[float]) -> None:
        self.inputs = np.asarray(inputs, dtype=float).reshape(-1).copy()
        self.inputs.setflags(write=False)

    def step(self, history: History, belief: Mapping[str, Any]) -> float:
        return float(self.inputs[history.t])

    @property
    def key(self) -> str:
        return f"fixed:{self.inputs.tobytes().hex()}"

    @property
    def label(self) -> str:
        return f"fixed({', '.join(f'{value:.4g}' for value in self.inputs)})"


def gaussian_step(predict: PredictFn, std: float) -> StepSampler:
    """``sample_step`` for a deterministic model with Gaussian observation noise.

    The mean of y[t] is the last entry of ``predict`` on the inputs so far, so
    ``predict`` must accept any prefix of an episode's inputs and its mean at
    step t must not depend on later inputs.
    """
    if not std > 0:
        raise ValueError("std must be positive")

    def sample(theta, inputs, observations, rng):
        del observations
        mean = np.asarray(predict(np.asarray(theta, dtype=float), inputs), dtype=float).reshape(-1)
        return float(mean[-1] + std * rng.standard_normal())

    return sample


def _next_inputs(step: StepBatch, histories: Sequence[History], belief, episode: EpisodeSpec) -> np.ndarray:
    """One validated, clipped input per history; shared by scoring and execution."""
    try:
        values = np.asarray(step(histories, belief), dtype=float).reshape(-1)
    except ProgramError:
        raise
    except Exception as error:
        raise ProgramError(f"{type(error).__name__}: {error}") from error
    if values.shape != (len(histories),) or np.any(~np.isfinite(values)):
        raise ProgramError("a program must return one finite number per step")
    return np.clip(values, episode.lower, episode.upper)


def run_episode(
    program: Program,
    environment: EpisodicEnvironment,
    episode: EpisodeSpec,
    belief: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Execute ``program`` against ``environment``; return realised inputs and trajectory."""
    environment.start_episode()
    history = History(horizon=episode.steps)
    with program.session() as step:
        for _ in range(episode.steps):
            value = float(_next_inputs(step, [history], belief, episode)[0])
            observation = float(environment.observe(value))
            history.inputs.append(value)
            history.obs.append(observation)
    return np.asarray(history.inputs), np.asarray(history.obs)


# --------------------------------------------------------------------------- belief

def model_names(belief: BeliefState) -> dict[str, str]:
    """Unique display names by model key, in pool order."""
    names: dict[str, str] = {}
    for fit in belief.fits:
        name, suffix = fit.model.name, 2
        while name in names.values():
            name, suffix = f"{fit.model.name}#{suffix}", suffix + 1
        names[fit.model.key] = name
    return names


def _moments(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    weights = weights / weights.sum()
    mean = float(weights @ values)
    return {"mean": mean, "sd": float(np.sqrt(max(0.0, weights @ (values - mean) ** 2)))}


def belief_summary(belief: BeliefState) -> dict[str, Any]:
    """The JSON belief a program sees: weights and parameter moments.

    ``weights[model]``; ``parameters[model][name]`` with ``mean``, ``sd`` and,
    for uniform priors, ``low``/``high``; ``pooled[name]``, the moments of a
    parameter name over every weighted model that has it.
    """
    names = model_names(belief)
    weights, parameters = {}, {}
    pooled: dict[str, tuple[list, list]] = {}
    for weight, fit in zip(belief.model_weights, belief.fits, strict=True):
        name = names[fit.model.key]
        weights[name] = float(weight)
        particles = np.asarray(fit.parameters.particles, dtype=float)
        particle_weights = np.asarray(fit.parameters.weights, dtype=float)
        prior = fit.model.prior
        entry = {}
        for index, parameter in enumerate(prior.names):
            entry[parameter] = _moments(particles[:, index], particle_weights)
            if hasattr(prior, "lower") and hasattr(prior, "upper"):
                entry[parameter].update(low=float(prior.lower[index]), high=float(prior.upper[index]))
            if weight > 0:
                values, masses = pooled.setdefault(parameter, ([], []))
                values.append(particles[:, index])
                masses.append(weight * particle_weights)
        parameters[name] = entry
    return {
        "weights": weights,
        "parameters": parameters,
        "pooled": {
            parameter: _moments(np.concatenate(values), np.concatenate(masses))
            for parameter, (values, masses) in pooled.items()
        },
    }


@dataclass(frozen=True)
class _Particles:
    """Distinct weighted ``(model, theta)`` values of the weighted part of a belief."""

    models: tuple[CandidateModel, ...]
    model_weights: np.ndarray
    hypothesis: np.ndarray
    thetas: tuple[np.ndarray, ...]
    weight: np.ndarray

    @classmethod
    def from_belief(cls, belief: BeliefState) -> "_Particles":
        # Merging equal values is the self-exclusion key and also removes
        # repeated likelihood evaluations of SMC duplicates.
        models, positions, values, hypothesis, weight = [], {}, {}, [], []
        for model, theta, mass in belief.iter_parameter_particles():
            h = positions.setdefault(model.key, len(models))
            if h == len(models):
                models.append(model)
            theta = np.asarray(theta, dtype=float)
            key = (h, theta.tobytes())
            if key in values:
                weight[values[key]] += mass
            else:
                values[key] = len(weight)
                hypothesis.append(h)
                weight.append(mass)
        thetas = [None] * len(weight)
        for (h, raw), index in values.items():
            thetas[index] = np.frombuffer(raw, dtype=float).copy()
        hypothesis = np.asarray(hypothesis, dtype=int)
        weight = np.asarray(weight, dtype=float)
        weight /= weight.sum()
        model_weights = np.bincount(hypothesis, weights=weight, minlength=len(models))
        return cls(tuple(models), model_weights, hypothesis, tuple(thetas), weight)


def _row_logsumexp(values: np.ndarray) -> np.ndarray:
    top = values.max(axis=1, keepdims=True)
    safe = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore"):
        return safe[:, 0] + np.log(np.exp(values - safe).sum(axis=1))


@dataclass(frozen=True)
class ProgramScore:
    """A trajectory-information estimate and the imagined episodes behind it."""

    bits: float
    inputs: np.ndarray
    observations: np.ndarray
    models: tuple[CandidateModel, ...]
    hypothesis: np.ndarray


def trajectory_information(
    program: Program,
    belief: BeliefState,
    likelihood: Callable[..., float],
    episode: EpisodeSpec,
    *,
    draws: int,
    seed: int,
    summary: Mapping[str, Any] | None = None,
) -> ProgramScore:
    """Nested-MC ``I(M; Y_1:T | D, program)`` in bits (see the module docstring).

    The same ``seed`` gives every candidate the same outer particles, step
    noise streams (one generator per outer draw) and likelihood generator.
    Every weighted model must have ``sample_step``. Raises ``ProgramError`` if
    the program fails and ``ValueError`` if a model cannot simulate an input.
    """
    if draws < 1:
        raise ValueError("draws must be positive")
    particles = _Particles.from_belief(belief)
    missing = [model.name for model in particles.models if model.sample_step is None]
    if missing:
        raise ValueError(f"program design needs sample_step on every weighted model: {missing}")
    summary = belief_summary(belief) if summary is None else summary
    streams = np.random.SeedSequence(seed).spawn(draws + 2)
    outer = np.random.default_rng(streams[0]).choice(particles.weight.size, size=draws, p=particles.weight)
    likelihood_rng = np.random.default_rng(streams[1])
    step_rngs = [np.random.default_rng(stream) for stream in streams[2:]]

    histories = [History(horizon=episode.steps) for _ in range(draws)]
    with program.session() as step:
        for _ in range(episode.steps):
            values = _next_inputs(step, histories, summary, episode)
            for history, value, particle, rng in zip(histories, values, outer, step_rngs, strict=True):
                history.inputs.append(float(value))
                model = particles.models[particles.hypothesis[particle]]
                y = float(model.sample_step(particles.thetas[particle], np.asarray(history.inputs),
                                            np.asarray(history.obs), rng))
                if not np.isfinite(y):
                    raise ValueError(f"model {model.name!r} simulated a non-finite observation")
                history.obs.append(y)
    inputs = np.asarray([history.inputs for history in histories])
    observations = np.asarray([history.obs for history in histories])

    log_likelihood = np.full((draws, particles.weight.size), -np.inf)
    for value, theta in enumerate(particles.thetas):
        model = particles.models[particles.hypothesis[value]]
        for i in range(draws):
            try:
                log_likelihood[i, value] = float(
                    likelihood([Datum(inputs[i], observations[i])], model, theta, likelihood_rng)
                )
            except (ValueError, ArithmeticError):
                pass
    log_likelihood[~np.isfinite(log_likelihood)] = -np.inf

    log_weight = np.log(particles.weight)
    log_p = np.full((draws, len(particles.models)), -np.inf)
    for h in range(len(particles.models)):
        members = np.flatnonzero(particles.hypothesis == h)
        keep = members[None, :] != outer[:, None]
        keep[~keep.any(axis=1)] = True
        member_log_weight = np.where(keep, log_weight[members][None, :], -np.inf)
        log_p[:, h] = (_row_logsumexp(log_likelihood[:, members] + member_log_weight)
                       - _row_logsumexp(member_log_weight))
    own = log_p[np.arange(draws), particles.hypothesis[outer]]
    mixture = _row_logsumexp(log_p + np.log(particles.model_weights)[None, :])
    good = np.isfinite(own) & np.isfinite(mixture)
    bits = float(np.mean(own[good] - mixture[good]) / LOG2) if good.any() else 0.0
    return ProgramScore(bits, inputs, observations, particles.models, particles.hypothesis[outer])


# --------------------------------------------------------------------------- proposer context

def disagreement_report(belief: BeliefState, episode: EpisodeSpec, points: int = 9) -> list[dict[str, Any]]:
    """Each weighted model's predictive mean and SD of a one-step episode on an input grid."""
    names = model_names(belief)
    particles = _Particles.from_belief(belief)
    report = []
    for value in np.linspace(episode.lower, episode.upper, points):
        row = {}
        for h, model in enumerate(particles.models):
            members = np.flatnonzero(particles.hypothesis == h)
            try:
                means = np.asarray([
                    float(np.asarray(model.observation_prediction(particles.thetas[j], np.asarray([value]))
                                     ).reshape(-1)[-1])
                    for j in members
                ])
            except (ValueError, ArithmeticError):
                continue
            row[names[model.key]] = {k: round(v, 6) for k, v in
                                     _moments(means, particles.weight[members]).items()}
        report.append({"input": round(float(value), 6), "models": row})
    return report


def imagined_runs(score: ProgramScore, belief: BeliefState, per_model: int = 3) -> dict[str, list[dict]]:
    """A few of the scorer's imagined episodes under each model."""
    names = model_names(belief)
    runs: dict[str, list[dict]] = {}
    for h, model in enumerate(score.models):
        rows = np.flatnonzero(score.hypothesis == h)[:per_model]
        runs[names[model.key]] = [
            {"inputs": np.round(score.inputs[i], 6).tolist(), "obs": np.round(score.observations[i], 6).tolist()}
            for i in rows
        ]
    return runs


__all__ = [
    "EpisodeSpec", "EpisodicEnvironment", "FixedProgram", "History", "Program", "ProgramError",
    "ProgramScore", "belief_summary", "disagreement_report", "gaussian_step", "imagined_runs",
    "model_names", "run_episode", "trajectory_information",
]
