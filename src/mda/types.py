"""Public data structures and model contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np


Array = np.ndarray
PredictFn = Callable[[Array, Any], Array]


class NonFinitePrediction(ValueError):
    """A model returned NaN or infinity at some parameter and action."""


@runtime_checkable
class Prior(Protocol):
    """Distribution over a model's flat parameter vector.

    ``proposal_scale_hint`` gives a per-coordinate physical (or sampler-coordinate)
    random-walk scale.  It is a scale hint, not a support bound.
    """

    names: tuple[str, ...]

    @property
    def dimension(self) -> int: ...

    @property
    def proposal_scale_hint(self) -> Array: ...

    def sample(self, rng: np.random.Generator, count: int) -> Array: ...

    def logpdf(self, theta: Array) -> float: ...


@dataclass(frozen=True, eq=False)
class Datum:
    """One intervention/design and its observed scalar, vector, or trajectory."""

    action: Any
    observation: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation", np.asarray(self.observation, dtype=float))


@dataclass(frozen=True)
class UniformPrior:
    """Independent uniform coefficient priors, as used in the paper's symbolic domains."""

    names: tuple[str, ...]
    lower: Array
    upper: Array

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.shape != upper.shape or lower.ndim != 1:
            raise ValueError("prior bounds must be one-dimensional arrays of equal shape")
        if len(self.names) != len(lower):
            raise ValueError("one parameter name is required for each bound")
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower >= upper):
            raise ValueError("every prior bound must be finite with lower < upper")
        if len(set(self.names)) != len(self.names):
            raise ValueError("parameter names must be unique")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dimension(self) -> int:
        return len(self.names)

    @property
    def proposal_scale_hint(self) -> Array:
        return self.upper - self.lower

    def sample(self, rng: np.random.Generator, count: int) -> Array:
        return rng.uniform(self.lower, self.upper, size=(count, self.dimension))

    def logpdf(self, theta: Array) -> float:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.dimension,) or np.any(theta < self.lower) or np.any(theta > self.upper):
            return -np.inf
        return -float(np.log(self.upper - self.lower).sum())


@dataclass(frozen=True)
class StateSpaceModel:
    """Callbacks needed by the bootstrap particle-filter likelihood.

    ``transition`` advances all particles from z[t-1] to z[t]. ``observation_mean``
    maps the resulting states to the mean of y[t]. Both receive the experiment and
    observation index, so controls and interventions stay in the domain adapter.
    """

    initial_state: Callable[[Array, Any, int, np.random.Generator], Array]
    transition: Callable[[Array, Array, Any, int, np.random.Generator], Array]
    observation_mean: Callable[[Array, Array, Any, int], Array]


@dataclass(frozen=True, eq=False)
class CandidateModel:
    """One model-structure particle and its parameter prior."""

    name: str
    prior: Prior
    predict: PredictFn
    structure: str | None = None
    predict_target: PredictFn | None = None
    state_space: StateSpaceModel | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.structure or self.name

    @property
    def parameter_count(self) -> int:
        return self.prior.dimension

    def observation_prediction(self, theta: Array, action: Any) -> Array:
        value = np.asarray(self.predict(np.asarray(theta, dtype=float), action), dtype=float)
        if np.any(~np.isfinite(value)):
            raise NonFinitePrediction(f"model {self.name!r} returned a non-finite prediction")
        return value

    def target_prediction(self, theta: Array, action: Any) -> Array:
        callback = self.predict_target or self.predict
        value = np.asarray(callback(np.asarray(theta, dtype=float), action), dtype=float)
        if np.any(~np.isfinite(value)):
            raise NonFinitePrediction(f"model {self.name!r} returned a non-finite target prediction")
        return value
