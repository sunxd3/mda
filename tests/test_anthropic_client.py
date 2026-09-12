import sys
from types import SimpleNamespace

import pytest

from mda.anthropic_client import AnthropicJSONClient


def _install_stub(monkeypatch, contents):
    calls = []
    queued = iter(contents)

    class Messages:
        def create(self, **request):
            calls.append(request)
            return SimpleNamespace(content=next(queued))

    class Anthropic:
        def __init__(self, *, api_key):
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Anthropic))
    return calls


def _tool(value, name="answer"):
    return [SimpleNamespace(type="tool_use", name=name, input=value)]


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


@pytest.mark.parametrize("change", ["prompt", "schema", "temperature", "tokens", "model"])
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
    if change == "temperature": kwargs["temperature"] = 0.4
    if change == "tokens": kwargs["max_tokens"] = 1599
    if change == "model": model = "b"
    with pytest.raises(RuntimeError, match="cache miss"):
        AnthropicJSONClient(model=model, cache_dir=tmp_path, replay_only=True).ask_json(
            system, prompt, **kwargs
        )
    assert len(calls) == 1


def test_forced_tool_request_and_malformed_response(monkeypatch):
    calls = _install_stub(monkeypatch, [_tool({"ok": True}), []])
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    client = AnthropicJSONClient(api_key="stub")
    assert client.ask_json("rules", "question", schema=schema, name="answer") == {"ok": True}
    request = calls[0]
    assert request["tool_choice"] == {"type": "tool", "name": "answer"}
    assert request["tools"][0]["input_schema"] == schema
    assert request["tools"][0]["strict"] is True
    with pytest.raises(RuntimeError, match="forced tool output"):
        client.ask_json("rules", "another", schema=schema, name="answer")
