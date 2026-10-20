import inspect
import json
import sys
from types import SimpleNamespace

import pytest

from mda.anthropic_client import AnthropicJSONClient


def _install_stub(monkeypatch, messages):
    calls = []
    queued = iter(messages)

    class Stream:
        def __init__(self, message):
            self.message = message

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def get_final_message(self):
            return self.message

    class Messages:
        def stream(self, **request):
            calls.append(request)
            return Stream(next(queued))

    class Anthropic:
        def __init__(self, *, api_key):
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Anthropic))
    return calls


def _message(content, stop_reason="tool_use"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def _tool(value, name="answer", stop_reason="tool_use"):
    return _message([SimpleNamespace(type="tool_use", name=name, input=value)], stop_reason)


def test_cache_replay_needs_no_sdk_or_credentials(monkeypatch, tmp_path):
    calls = _install_stub(monkeypatch, [_tool({"value": 4})])
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    writer = AnthropicJSONClient(model="test", api_key="stub", cache_dir=tmp_path)
    assert writer.ask_json("system", "user", schema=schema, name="answer") == {"value": 4}

    monkeypatch.delitem(sys.modules, "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    replay = AnthropicJSONClient(model="test", cache_dir=tmp_path, replay_only=True)
    assert replay.ask_json("system", "user", schema=schema, name="answer") == {"value": 4}
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["prompt", "schema", "tokens", "model"])
def test_effective_request_mismatches_are_replay_misses(monkeypatch, tmp_path, change):
    schema = {"type": "object"}
    calls = _install_stub(monkeypatch, [_tool({"ok": True})])
    AnthropicJSONClient(model="a", api_key="stub", cache_dir=tmp_path).ask_json(
        "s", "p", schema=schema, name="answer"
    )
    kwargs = dict(schema=schema, name="answer", temperature=0.3, max_tokens=1600)
    system, prompt, model = "s", "p", "a"
    if change == "prompt": prompt = "different"
    if change == "schema": kwargs["schema"] = {"type": "object", "required": ["ok"]}
    if change == "tokens": kwargs["max_tokens"] = 1599
    if change == "model": model = "b"
    with pytest.raises(RuntimeError, match="cache miss"):
        AnthropicJSONClient(model=model, cache_dir=tmp_path, replay_only=True).ask_json(
            system, prompt, **kwargs
        )
    assert len(calls) == 1


def test_temperature_is_not_sent_and_not_part_of_cache_key(monkeypatch, tmp_path):
    calls = _install_stub(monkeypatch, [_tool({"ok": True})])
    schema = {"type": "object"}
    AnthropicJSONClient(api_key="stub", cache_dir=tmp_path).ask_json(
        "s", "p", schema=schema, name="answer", temperature=0.3
    )
    assert "temperature" not in calls[0]
    replay = AnthropicJSONClient(cache_dir=tmp_path, replay_only=True)
    assert replay.ask_json("s", "p", schema=schema, name="answer", temperature=0.9) == {
        "ok": True
    }
    assert len(calls) == 1


def test_forced_tool_request_and_malformed_response(monkeypatch):
    calls = _install_stub(monkeypatch, [_tool({"ok": True}), _message([])])
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    client = AnthropicJSONClient(api_key="stub")
    assert client.ask_json("rules", "question", schema=schema, name="answer") == {"ok": True}
    request = calls[0]
    assert request["tool_choice"] == {"type": "tool", "name": "answer"}
    assert request["tools"][0]["input_schema"] == schema
    assert request["tools"][0]["strict"] is True
    with pytest.raises(RuntimeError, match="forced tool output"):
        client.ask_json("rules", "another", schema=schema, name="answer")


def test_request_kwargs_are_accepted_by_installed_sdk(monkeypatch):
    pytest.importorskip("anthropic")
    from anthropic.resources.messages import Messages

    accepted = {}
    for method in (Messages.create, Messages.stream):
        parameters = inspect.signature(method).parameters
        if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
            pytest.skip(f"{method.__qualname__} accepts arbitrary keyword arguments")
        accepted[method.__qualname__] = set(parameters) - {"self"}

    calls = _install_stub(monkeypatch, [_tool({"ok": True}), _tool({}, name="response")])
    client = AnthropicJSONClient(api_key="stub")
    client.ask_json("s", "p", schema={"type": "object"}, name="answer", temperature=0.3)
    client.ask("s", "p", temperature=0.7, max_tokens=80_000)
    for request in calls:
        for name, parameters in accepted.items():
            assert set(request) <= parameters, (name, set(request) - parameters)


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal", "end_turn", "pause_turn", None])
def test_incomplete_stop_reason_is_not_cached(monkeypatch, tmp_path, stop_reason):
    truncated = _tool({"ok": True}, stop_reason=stop_reason)
    calls = _install_stub(monkeypatch, [truncated, _tool({"ok": False})])
    schema = {"type": "object"}
    with pytest.raises(RuntimeError, match=f"did not complete: {stop_reason}"):
        AnthropicJSONClient(api_key="stub", cache_dir=tmp_path).ask_json(
            "s", "p", schema=schema, name="answer"
        )
    assert list(tmp_path.iterdir()) == []

    live = AnthropicJSONClient(api_key="stub", cache_dir=tmp_path)
    assert live.ask_json("s", "p", schema=schema, name="answer") == {"ok": False}
    replay = AnthropicJSONClient(cache_dir=tmp_path, replay_only=True)
    assert replay.ask_json("s", "p", schema=schema, name="answer") == {"ok": False}
    assert len(calls) == 2


def _sse(events):
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode("utf-8")


def test_large_budget_streams_through_installed_sdk(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    import httpx2

    bodies = []
    arguments = '{"predictions": [1.5, 2.5]}'
    events = [
        {"type": "message_start", "message": {
            "id": "msg_test", "type": "message", "role": "assistant",
            "model": "claude-opus-4-7", "content": [], "stop_reason": None,
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
        }},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "toolu_test", "name": "icl_forecast", "input": {},
        }},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": arguments[:17]}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": arguments[17:]}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
         "usage": {"output_tokens": 12}},
        {"type": "message_stop"},
    ]

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse(events)
        )

    real = anthropic.Anthropic

    def offline(*, api_key):
        return real(
            api_key=api_key, base_url="http://anthropic.test", max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        )

    monkeypatch.setattr(anthropic, "Anthropic", offline)
    schema = {"type": "object", "properties": {
        "predictions": {"type": "array", "items": {"type": "number"}}}}
    client = AnthropicJSONClient(api_key="stub")
    assert client.ask_json(
        "s", "p", schema=schema, name="icl_forecast", temperature=0.3, max_tokens=80_000
    ) == {"predictions": [1.5, 2.5]}
    assert len(bodies) == 1
    assert bodies[0]["stream"] is True
    assert bodies[0]["max_tokens"] == 80_000
    assert not {"temperature", "top_p", "top_k"} & set(bodies[0])
