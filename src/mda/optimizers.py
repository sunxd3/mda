"""Optional optimizers for continuous experimental designs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class CMAESOptimizer:
    """Maximize a score over a bounded vector using :mod:`cma`.

    CMA-ES operates in a normalized unit box.  Coordinates selected by
    ``log_axes`` are decoded by geometric rather than linear interpolation.
    """

    def __init__(
        self,
        lower: Any,
        upper: Any,
        *,
        log_axes: Any | None = None,
        decode: Callable[[np.ndarray], Any] | None = None,
        max_evaluations: int = 200,
    ) -> None:
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if (
            self.lower.ndim != 1
            or self.upper.shape != self.lower.shape
            or self.lower.size == 0
            or not np.all(np.isfinite(self.lower))
            or not np.all(np.isfinite(self.upper))
            or np.any(self.lower >= self.upper)
        ):
            raise ValueError("lower and upper must be nonempty finite vectors with lower < upper")

        if log_axes is None:
            self.log_axes = np.zeros(self.lower.size, dtype=bool)
        else:
            raw_log_axes = np.asarray(log_axes)
            if raw_log_axes.shape != self.lower.shape or raw_log_axes.dtype.kind != "b":
                raise ValueError("log_axes must be a boolean vector matching the bounds")
            self.log_axes = raw_log_axes.astype(bool, copy=True)
        if np.any(self.lower[self.log_axes] <= 0):
            raise ValueError("bounds for logarithmic axes must be positive")
        if not isinstance(max_evaluations, int) or isinstance(max_evaluations, bool) or max_evaluations < 1:
            raise ValueError("max_evaluations must be a positive integer")

        self.decode = decode if decode is not None else lambda value: value
        if not callable(self.decode):
            raise TypeError("decode must be callable")
        self.max_evaluations = max_evaluations

    def _domain_vector(self, normalized: Any) -> np.ndarray:
        unit = np.clip(np.asarray(normalized, dtype=float), 0.0, 1.0)
        value = self.lower + unit * (self.upper - self.lower)
        if np.any(self.log_axes):
            value[self.log_axes] = np.exp(
                np.log(self.lower[self.log_axes])
                + unit[self.log_axes]
                * (np.log(self.upper[self.log_axes]) - np.log(self.lower[self.log_axes]))
            )
        return value

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw a uniform design in the normalized (log-mapped) box."""
        return self.decode(self._domain_vector(rng.random(self.lower.size)))

    def __call__(self, score: Callable[[Any], float], rng: np.random.Generator) -> Any:
        """Return the best finite-scoring action actually evaluated."""
        try:
            import cma
        except ImportError as error:  # pragma: no cover - exercised without the optional extra
            raise ImportError("CMAESOptimizer requires the 'design' extra") from error

        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        local_rng = np.random.default_rng(seed)

        def randn(*shape: int) -> np.ndarray:
            return local_rng.standard_normal(shape)

        dimension = self.lower.size
        population_size = max(2, min(self.max_evaluations, 4 + int(3 * np.log(dimension))))
        # The evaluation budget is the only intended stopping rule: a nearly
        # flat score (a concentrated model posterior) would otherwise trip
        # cma's tolerance tests after a single generation.
        options = {
            "bounds": [0.0, 1.0],
            "popsize": population_size,
            "randn": randn,
            "seed": np.nan,
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
            "tolfun": 0.0,
            "tolfunhist": 0.0,
            "tolfunrel": 0.0,
            "tolx": 0.0,
            "tolflatfitness": self.max_evaluations,
            "tolstagnation": self.max_evaluations,
        }
        strategy = cma.CMAEvolutionStrategy(np.full(dimension, 0.5), 0.25, options)

        best_action: Any = None
        best_score = -np.inf
        evaluated = 0
        while evaluated < self.max_evaluations:
            if strategy.stop():
                # Any remaining stop (e.g. numerical degeneracy) restarts from a
                # random point so the full budget is still spent.
                strategy = cma.CMAEvolutionStrategy(local_rng.random(dimension), 0.25, options)
            count = min(population_size, self.max_evaluations - evaluated)
            candidates = strategy.ask(number=count)
            losses: list[float] = []
            for candidate in candidates:
                vector = self._domain_vector(candidate)
                action = self.decode(vector.copy())
                value = float(score(action))
                evaluated += 1
                if np.isfinite(value):
                    if value > best_score:
                        best_score = value
                        best_action = action
                    losses.append(-value)
                else:
                    losses.append(np.inf)
            # A partial final generation cannot reliably update a strategy whose
            # recombination weights were configured for the full population.
            if count == population_size:
                strategy.tell(candidates, losses)

        if best_action is None:
            raise ValueError("CMA-ES did not observe a finite score")
        return best_action


class DifferentialEvolutionOptimizer(CMAESOptimizer):
    """Maximize a score over the same box with SciPy's differential evolution.

    The score is *batched*: it takes a list of actions and returns their
    scores, so each generation of ``population_size * dimension`` candidates
    is scored in one call (SciPy's vectorized, deferred-update DE). Generations
    run until ``max_evaluations`` is spent; there is no polishing step. Bounds,
    log axes, ``decode`` and ``sample`` behave as in :class:`CMAESOptimizer`.
    """

    batched = True

    def __init__(self, lower: Any, upper: Any, *, population_size: int = 4, **options: Any) -> None:
        super().__init__(lower, upper, **options)
        if population_size < 1:
            raise ValueError("population_size must be positive")
        self.population_size = population_size

    def __call__(self, score: Callable[[list[Any]], Any], rng: np.random.Generator) -> Any:
        from scipy.optimize import differential_evolution

        dimension = self.lower.size
        generation = self.population_size * dimension
        if self.max_evaluations < generation:
            raise ValueError("max_evaluations must cover at least one generation")
        best: list[Any] = [None, -np.inf]

        def loss(units: np.ndarray) -> np.ndarray:
            actions = [self.decode(self._domain_vector(unit)) for unit in np.atleast_2d(units.T)]
            values = np.asarray(score(actions), dtype=float)
            for action, value in zip(actions, values, strict=True):
                if np.isfinite(value) and value > best[1]:
                    best[:] = [action, float(value)]
            # A finite penalty keeps SciPy's convergence statistics defined.
            return np.where(np.isfinite(values), -values, 1e300)

        differential_evolution(
            loss, [(0.0, 1.0)] * dimension, popsize=self.population_size,
            maxiter=self.max_evaluations // generation - 1, polish=False, tol=0.0, atol=0.0,
            init="latinhypercube", vectorized=True, updating="deferred",
            rng=np.random.default_rng(int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))),
        )
        if best[0] is None:
            raise ValueError("differential evolution did not observe a finite score")
        return best[0]
