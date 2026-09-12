"""Pure in-context-learning forecasting baseline (paper H.3.3/H.4.2)."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def _identity(value: Any) -> Any:
    return value


def _jsonable(value: Any) -> Any:
    """Convert benchmark values to JSON without walking arbitrary object state."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    raise TypeError(
        f"ICL prompt value of type {type(value).__name__} is not JSON serializable; "
        "supply action_features"
    )


def _extract_target(fn: Callable[..., Any], observation: Any, action: Any) -> Any:
    try:
        parameters = inspect.signature(fn).parameters.values()
        accepts_action = any(p.kind == p.VAR_POSITIONAL for p in parameters) or sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters
        ) >= 2
    except (TypeError, ValueError):
        accepts_action = False
    return fn(observation, action) if accepts_action else fn(observation)


def _value_schema(shape: tuple[int, ...]) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number"}
    for size in reversed(shape):
        schema = {
            "type": "array", "minItems": size, "maxItems": size, "items": schema
        }
    return schema


def make_icl_predictor(
    client: Any,
    context: str,
    dataset: Sequence[Any],
    query_actions: Sequence[Any],
    *,
    target: Callable[..., Any] = _identity,
    action_features: Callable[[Any], Any] = _identity,
    action_key: Callable[[Any], Any] = repr,
    target_shape: int | Sequence[int] | tuple[()] = (),
) -> Callable[[Any], np.ndarray]:
    """Forecast all queries in one LLM call and return a key-based predictor.

    The request contains only the domain context, observed action/target pairs,
    and query action features.  In particular, no held-out outcomes or MDA model
    state is accepted by or included in this boundary.
    """
    shape = (target_shape,) if isinstance(target_shape, int) else tuple(target_shape)
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape):
        raise ValueError("target_shape must contain non-negative integers")
    queries = list(query_actions)
    keys = [action_key(action) for action in queries]
    if len(set(keys)) != len(keys):
        raise ValueError("query action_key values must be unique")

    observations = []
    for datum in dataset:
        try:
            action, observation = datum.action, datum.observation
        except AttributeError:
            try:
                action, observation = datum
            except (TypeError, ValueError) as error:
                raise TypeError("dataset entries must be Datum-like or (action, observation)") from error
        observations.append({
            "action": _jsonable(action_features(action)),
            "target": _jsonable(_extract_target(target, observation, action)),
        })
    query_rows = [
        {"index": index, "action": _jsonable(action_features(action))}
        for index, action in enumerate(queries)
    ]
    system = (
        "Forecast outcomes from in-context observations only. Do not infer or request "
        "held-out labels, and do not use predictions or models from another agent. "
        "Return one prediction for every query, in the supplied index order."
    )
    prompt = json.dumps(
        {"context": context, "observations": observations, "queries": query_rows},
        ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    )
    value_schema = _value_schema(shape)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["predictions"],
        "properties": {"predictions": {
            "type": "array", "minItems": len(queries), "maxItems": len(queries),
            "items": value_schema,
        }},
    }
    payload = client.ask_json(
        system, prompt, schema=schema, name="icl_forecast", temperature=0.3,
        max_tokens=max(800, 80 * max(1, len(queries)) * max(1, int(np.prod(shape)))),
    )
    predictions = payload.get("predictions") if isinstance(payload, dict) else None
    if not isinstance(predictions, list) or len(predictions) != len(queries):
        raise ValueError("ICL response must contain one ordered prediction per query")
    checked: list[np.ndarray] = []
    for index, prediction in enumerate(predictions):
        try:
            array = np.asarray(prediction, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"ICL prediction {index} is not numeric") from error
        if array.shape != shape:
            raise ValueError(
                f"ICL prediction {index} has shape {array.shape}, expected {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"ICL prediction {index} is not finite")
        checked.append(array.copy())
    by_key = dict(zip(keys, checked, strict=True))

    def predict(action: Any) -> np.ndarray:
        key = action_key(action)
        if key not in by_key:
            raise KeyError("action was not in the ICL query batch")
        return by_key[key].copy()

    return predict
