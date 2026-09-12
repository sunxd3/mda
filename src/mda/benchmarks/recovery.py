"""Finite-grid numerical structural-recovery check for paper A2/B1.

This module refits only the numerical constants of an already trusted
``CandidateModel``.  Recovery means that the refitted model has held-out RMSLE
at most ``tolerance``; it is a finite-grid numerical approximation, not proof
of exact symbolic equality.  In particular, no SymPy parsing or
canonicalization is performed.  Symbolic candidates should be constructed by
the project's safe :func:`mda.expressions.compile_expression`; fitting calls
the resulting trusted ``candidate.predict`` function and never evaluates the
raw ``candidate.structure`` string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mda.types import CandidateModel, UniformPrior


def _transform(values: np.ndarray, signed_log1p: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("truth and predictions must be finite")
    if signed_log1p:
        return np.sign(values) * np.log1p(np.abs(values))
    if np.any(values < 0.0):
        raise ValueError(
            "ordinary RMSLE requires nonnegative truth and predictions; "
            "pass signed_log1p=True for signed quantities"
        )
    return np.log1p(values)


def _as_actions(actions: Any) -> list[Any]:
    if isinstance(actions, np.ndarray) and actions.ndim == 0:
        return [actions.item()]
    try:
        result = list(actions)
    except TypeError:
        result = [actions]
    if not result:
        raise ValueError("action grids must not be empty")
    return result


def _values(truth: Any, actions: Sequence[Any]) -> list[np.ndarray]:
    if callable(truth):
        return [np.asarray(truth(action), dtype=float) for action in actions]
    values = np.asarray(truth, dtype=float)
    if values.ndim == 0:
        if len(actions) != 1:
            raise ValueError("truth must have one leading entry per action")
        return [values]
    if values.shape[0] != len(actions):
        raise ValueError("truth must have one leading entry per action")
    return [np.asarray(value, dtype=float) for value in values]


def _explicit_grids(actions: Any, truth: Any) -> tuple[Any, Any, Any, Any] | None:
    """Decode explicitly separate grids without mistaking an action dict for one."""
    if isinstance(actions, Mapping) and set(actions) == {"fit", "holdout"}:
        if not isinstance(truth, Mapping) or set(truth) != {"fit", "holdout"}:
            raise ValueError("split actions require split truth with fit and holdout keys")
        return actions["fit"], actions["holdout"], truth["fit"], truth["holdout"]
    # A tuple is reserved for the explicit (fit, holdout) form.  Use a list for
    # a single grid of two actions.
    if isinstance(actions, tuple) and len(actions) == 2:
        if not isinstance(truth, tuple) or len(truth) != 2:
            raise ValueError("split actions require split truth")
        return actions[0], actions[1], truth[0], truth[1]
    return None


def evaluate_recovery(
    candidate: CandidateModel,
    actions: Any,
    truth: Any,
    tolerance: float = 0.02,
    seed: int = 0,
    max_evaluations: int = 2000,
    *,
    signed_log1p: bool = False,
) -> tuple[np.ndarray, float, bool]:
    """Refit constants and evaluate finite-grid held-out structural recovery.

    ``actions`` and ``truth`` preferably contain separate fit and holdout grids,
    either as ``{"fit": ..., "holdout": ...}`` mappings or two-tuples.  If a
    single grid is supplied, a seeded 80/20 split is made. ``truth`` may instead
    contain callables in either form.  The candidate must use an independent,
    bounded :class:`UniformPrior`; NumPyro-backed and general priors are not
    supported.

    The return value is ``(refitted_parameters, holdout_rmsle, recovered)``.
    Neither the candidate nor any posterior object is modified.
    """
    if not isinstance(candidate.prior, UniformPrior):
        raise TypeError("recovery requires an independent bounded UniformPrior")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")

    split = _explicit_grids(actions, truth)
    if split is None:
        all_actions = _as_actions(actions)
        all_truth = _values(truth, all_actions)
        if len(all_actions) < 2:
            raise ValueError("a single grid needs at least two actions for a holdout")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(all_actions))
        holdout_count = max(1, int(np.ceil(0.2 * len(all_actions))))
        holdout_indices, fit_indices = order[:holdout_count], order[holdout_count:]
        fit_actions = [all_actions[i] for i in fit_indices]
        holdout_actions = [all_actions[i] for i in holdout_indices]
        fit_truth = [all_truth[i] for i in fit_indices]
        holdout_truth = [all_truth[i] for i in holdout_indices]
    else:
        fit_raw, holdout_raw, fit_truth_raw, holdout_truth_raw = split
        fit_actions, holdout_actions = _as_actions(fit_raw), _as_actions(holdout_raw)
        fit_truth = _values(fit_truth_raw, fit_actions)
        holdout_truth = _values(holdout_truth_raw, holdout_actions)
        rng = np.random.default_rng(seed)

    def residuals(theta: np.ndarray, grid: Sequence[Any], expected: Sequence[np.ndarray]) -> np.ndarray:
        chunks = []
        for action, target in zip(grid, expected, strict=True):
            prediction = np.asarray(candidate.predict(theta, action), dtype=float)
            if prediction.shape != target.shape:
                raise ValueError("candidate prediction and truth shapes differ")
            chunks.append((_transform(prediction, signed_log1p) - _transform(target, signed_log1p)).ravel())
        return np.concatenate(chunks)

    try:
        from scipy.optimize import least_squares
    except ImportError as error:  # pragma: no cover - depends on installation extras
        raise RuntimeError("structural recovery requires scipy") from error

    prior = candidate.prior
    if prior.dimension == 0:
        parameters = np.empty(0, dtype=float)
        # A parameter-free structure still has a meaningful recovery score.
        heldout = residuals(parameters, holdout_actions, holdout_truth)
        rmsle = float(np.sqrt(np.mean(heldout**2)))
        return parameters, rmsle, bool(rmsle <= tolerance)
    starts = min(8, max_evaluations)
    initial = [0.5 * (prior.lower + prior.upper)]
    if starts > 1:
        initial.extend(prior.sample(rng, starts - 1))
    budget, remainder = divmod(max_evaluations, starts)
    best = None
    for index, start in enumerate(initial):
        result = least_squares(
            residuals, np.asarray(start), args=(fit_actions, fit_truth),
            bounds=(prior.lower, prior.upper), max_nfev=budget + (index < remainder),
        )
        score = float(np.mean(residuals(result.x, fit_actions, fit_truth) ** 2))
        if best is None or score < best[0]:
            best = (score, np.array(result.x, copy=True))
    assert best is not None
    parameters = best[1]
    heldout = residuals(parameters, holdout_actions, holdout_truth)
    rmsle = float(np.sqrt(np.mean(heldout**2)))
    return parameters, rmsle, bool(rmsle <= tolerance)
