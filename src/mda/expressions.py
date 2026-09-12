"""Validated expression models for scalar scientific-law discovery."""

from __future__ import annotations

import ast
import json
import keyword
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .openai_client import OpenAIJSONClient
from .types import CandidateModel, Datum, UniformPrior


def _scipy_special(name: str):
    """Return a lazy SciPy special function (SciPy is an optional dependency)."""
    def call(value):
        try:
            from scipy import special
        except ImportError as error:
            raise RuntimeError(f"{name} expressions require scipy") from error
        return getattr(special, name)(value)

    return call


_FUNCTIONS = {
    "abs": np.abs,
    "cos": np.cos,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "sign": np.sign,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "tan": np.tan,
    "tanh": np.tanh,
    "where": np.where,
    "k0": _scipy_special("k0"),
    "k1": _scipy_special("k1"),
    "gamma": _scipy_special("gamma"),
}
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.UAdd,
    ast.USub,
)


def compile_expression(
    expression: str,
    variable_names: Sequence[str],
    parameter_names: Sequence[str],
):
    variable_names = tuple(variable_names)
    parameter_names = tuple(parameter_names)
    reserved = set(_FUNCTIONS) | {"pi", "e"}
    declared = (*variable_names, *parameter_names)
    if any(not name.isidentifier() or keyword.iskeyword(name) for name in declared):
        raise ValueError("variable and parameter names must be valid Python identifiers")
    if len(set(variable_names)) != len(variable_names):
        raise ValueError("variable names must be unique")
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter names must be unique")
    if set(variable_names) & set(parameter_names):
        raise ValueError("variable and parameter names must be disjoint")
    if set(declared) & reserved:
        raise ValueError("variable and parameter names cannot shadow functions or constants")

    tree = ast.parse(expression, mode="eval")
    allowed_names = set(declared) | reserved
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"unsupported expression syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"undeclared name {node.id!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise ValueError("only approved numerical functions may be called")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed in expressions")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("only numerical constants are allowed")
    code = compile(tree, "<mda-expression>", "eval")

    def evaluate(theta: np.ndarray, action: Any) -> np.ndarray:
        if isinstance(action, Mapping):
            variables = {name: action[name] for name in variable_names}
        elif len(variable_names) == 1:
            variables = {variable_names[0]: action}
        else:
            raise ValueError("multi-input expressions require a mapping action")
        environment = {
            **_FUNCTIONS,
            "pi": np.pi,
            "e": np.e,
            **variables,
            **dict(zip(parameter_names, np.asarray(theta, dtype=float), strict=True)),
        }
        with np.errstate(all="raise"):
            return np.asarray(eval(code, {"__builtins__": {}}, environment), dtype=float)

    return evaluate


class SymbolicRegressionProposer:
    """Paper-style JSON proposer for algebraic scalar models."""

    def __init__(
        self,
        client: OpenAIJSONClient,
        *,
        variable_names: Sequence[str] = ("x",),
    ) -> None:
        self.client = client
        self.variable_names = tuple(variable_names)

    def propose(
        self,
        *,
        context: str,
        dataset: Sequence[Datum],
        current_models: Sequence[CandidateModel],
        residuals: Sequence[dict[str, Any]],
        count: int,
    ) -> list[CandidateModel]:
        observations = "\n".join(
            f"  {datum.action!r} -> {np.asarray(datum.observation).tolist()}" for datum in dataset
        ) or "  (none)"
        tried = json.dumps(list(residuals))
        structures = json.dumps([
            {
                "name": model.name,
                "structure": model.structure,
                "uniform_parameter_priors": {
                    name: [float(low), float(high)]
                    for name, low, high in zip(
                        model.prior.names, model.prior.lower, model.prior.upper, strict=True
                    )
                },
            }
            for model in current_models
        ])
        correlations = []
        for report in residuals:
            signed = report.get("signed_per_experiment")
            if not isinstance(signed, Sequence) or isinstance(signed, (str, bytes)):
                continue
            try:
                residual_vector = np.asarray(signed, dtype=float)
            except (TypeError, ValueError):
                continue
            if residual_vector.shape != (len(dataset),) or len(dataset) < 2:
                continue
            by_input = {}
            for variable in self.variable_names:
                values = []
                try:
                    for datum in dataset:
                        if isinstance(datum.action, Mapping):
                            value = datum.action[variable]
                        elif len(self.variable_names) == 1:
                            value = datum.action
                        else:
                            raise ValueError
                        scalar = np.asarray(value, dtype=float)
                        if scalar.ndim != 0:
                            raise ValueError
                        values.append(float(scalar))
                except (KeyError, TypeError, ValueError):
                    continue
                input_vector = np.asarray(values)
                if (
                    np.all(np.isfinite(input_vector))
                    and np.all(np.isfinite(residual_vector))
                    and np.ptp(input_vector) > 0
                    and np.ptp(residual_vector) > 0
                ):
                    correlation = float(np.corrcoef(input_vector, residual_vector)[0, 1])
                    if np.isfinite(correlation):
                        by_input[variable] = correlation
            if by_input:
                correlations.append({
                    "structure": report.get("structure"),
                    "input_signed_residual_correlations": by_input,
                })
        correlation_context = json.dumps(correlations)
        variables = ", ".join(self.variable_names)
        prompt = f"""{context}

Observed experiments (inputs -> output):
{observations}

Current model structures and uniform parameter priors:
{structures}

Forms already tried and their residuals (do not repeat them):
{tried}
The per_experiment residuals follow the experiment order above. Use their variation
across inputs to identify where each structure fails.

Pearson correlations between numeric scalar inputs and signed residuals, linked to
the corresponding model structure (undefined correlations are omitted):
{correlation_context}

Propose {count} distinct, parsimonious closed-form hypotheses for the mean output.
Each expr must be a Python expression using only variables {variables}, declared
parameters, arithmetic, and these functions: {', '.join(sorted(_FUNCTIONS))}.
Every unknown coefficient or exponent must be a named parameter with finite,
scientifically plausible uniform-prior bounds. Return JSON only."""
        schema = {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "expr": {"type": "string"},
                            "params": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "low": {"type": "number"},
                                        "high": {"type": "number"},
                                    },
                                    "required": ["name", "low", "high"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["name", "expr", "params"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["hypotheses"],
            "additionalProperties": False,
        }
        payload = self.client.ask_json(
            "You are a scientific model-discovery assistant. Reply with JSON only.",
            prompt,
            schema=schema,
            name="model_hypotheses",
            max_tokens=max(1600, 400 * count),
        )
        models = []
        existing = {model.key for model in current_models}
        for item in payload["hypotheses"]:
            try:
                names = tuple(parameter["name"] for parameter in item["params"])
                prior = UniformPrior(
                    names,
                    np.asarray([parameter["low"] for parameter in item["params"]]),
                    np.asarray([parameter["high"] for parameter in item["params"]]),
                )
                expression = item["expr"].strip()
                if expression in existing:
                    continue
                predict = compile_expression(expression, self.variable_names, names)
                model = CandidateModel(
                    name=item["name"].strip(),
                    structure=expression,
                    prior=prior,
                    predict=predict,
                )
                for datum in dataset[:3]:
                    model.observation_prediction(prior.sample(np.random.default_rng(0), 1)[0], datum.action)
                models.append(model)
                existing.add(expression)
            except (KeyError, TypeError, ValueError):
                continue
        return models[:count]
