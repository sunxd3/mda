"""Finite-grid numerical and opt-in symbolic structural recovery (A2/B1)."""

from __future__ import annotations

import ast
import keyword
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mda.types import CandidateModel, UniformPrior


_SYMBOLIC_FUNCTIONS = {
    "abs", "cos", "exp", "gamma", "k0", "k1", "log", "log10",
    "maximum", "minimum", "sign", "sin", "sqrt", "tan", "tanh", "where",
}


def canonical_symbolic_candidate(
    expression: str,
    input_names: Sequence[str],
    prior: UniformPrior,
    *,
    name: str | None = None,
) -> CandidateModel:
    """Safely parse, canonicalize, and compile an explicitly symbolic candidate.

    Python's AST is translated node-by-node; neither ``sympify`` nor evaluation
    of the supplied text is used.  Inputs and free parameters must be declared.
    Approved functions may be bare (``exp(x)``) or NumPy-qualified
    (``np.exp(x)``).  SymPy and SciPy are provided by the ``recovery`` extra.
    """
    if not isinstance(prior, UniformPrior):
        raise TypeError("symbolic recovery requires a UniformPrior")
    input_names = tuple(input_names)
    declared = (*input_names, *prior.names)
    reserved = _SYMBOLIC_FUNCTIONS | {"pi", "e", "np"}
    if any(not item.isidentifier() or keyword.iskeyword(item) for item in declared):
        raise ValueError("input and parameter names must be valid Python identifiers")
    if len(set(input_names)) != len(input_names) or len(set(prior.names)) != len(prior.names):
        raise ValueError("input and parameter names must be unique")
    if set(input_names) & set(prior.names):
        raise ValueError("input and parameter names must be disjoint")
    if set(declared) & reserved:
        raise ValueError("input and parameter names cannot shadow functions or constants")
    try:
        import sympy as sp
        from scipy import special
    except ImportError as error:  # pragma: no cover - depends on installation extras
        raise RuntimeError("symbolic recovery requires sympy and scipy") from error

    symbols = {item: sp.Symbol(item, real=True) for item in declared}
    functions = {
        "abs": sp.Abs, "cos": sp.cos, "exp": sp.exp, "gamma": sp.gamma,
        "k0": lambda value: sp.besselk(0, value),
        "k1": lambda value: sp.besselk(1, value),
        "log": sp.log, "log10": lambda value: sp.log(value, 10),
        "maximum": sp.Max, "minimum": sp.Min, "sign": sp.sign,
        "sin": sp.sin, "sqrt": sp.sqrt, "tan": sp.tan, "tanh": sp.tanh,
        "where": lambda condition, yes, no: sp.Piecewise((yes, condition), (no, True)),
    }
    binary = {
        ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b, ast.Mod: sp.Mod,
    }

    def translate(node: ast.AST):
        if isinstance(node, ast.Expression):
            return translate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return sp.Integer(node.value) if isinstance(node.value, int) else sp.Float(node.value)
        if isinstance(node, ast.Name):
            if node.id in symbols:
                return symbols[node.id]
            if node.id == "pi":
                return sp.pi
            if node.id == "e":
                return sp.E
            raise ValueError(f"undeclared name {node.id!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](translate(node.left), translate(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = translate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Call) and not node.keywords:
            function_name = None
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == "np"):
                function_name = node.func.attr
            if function_name in functions:
                try:
                    return functions[function_name](*(translate(arg) for arg in node.args))
                except TypeError as error:
                    raise ValueError(f"invalid arguments for function {function_name!r}") from error
            raise ValueError("only approved numerical functions may be called")
        raise ValueError(f"unsupported expression syntax: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid symbolic expression syntax") from error
    canonical = sp.simplify(translate(tree))
    ordered_symbols = [symbols[item] for item in declared]
    numerical = sp.lambdify(
        ordered_symbols, canonical,
        modules=[{"besselk": special.kv, "gamma": special.gamma}, "numpy"],
        cse=True,
    )

    def predict(theta: np.ndarray, action: Any) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (prior.dimension,):
            raise ValueError("parameter vector has the wrong shape")
        if isinstance(action, Mapping):
            inputs = [action[item] for item in input_names]
        elif len(input_names) == 1:
            inputs = [action]
        else:
            raise ValueError("multi-input expressions require a mapping action")
        with np.errstate(all="raise"):
            return np.asarray(numerical(*inputs, *theta), dtype=float)

    structure = sp.sstr(canonical)
    return CandidateModel(name or structure, prior, predict, structure=structure)


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
        return parameters, rmsle, bool(rmsle < tolerance)
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
    return parameters, rmsle, bool(rmsle < tolerance)
