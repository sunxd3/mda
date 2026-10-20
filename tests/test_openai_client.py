import inspect
import json
import sys
from types import SimpleNamespace

import pytest

from mda.openai_client import OpenAIJSONClient


def _install_openai_stub(monkeypatch, responses):
    calls = []
    queued = iter(responses)

    class Responses:
        def create(self, **request):
            calls.append(request)
            return next(queued)

    class OpenAI:
        def __init__(self, *, api_key):
            self.responses = Responses()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=OpenAI))
    return calls


def _completed(text):
    return SimpleNamespace(status="completed", output_text=text)


def test_cache_hits_across_instances_and_effective_request_changes_miss(
    monkeypatch, tmp_path
):
    calls = _install_openai_stub(
        monkeypatch,
        [
            _completed('{"call": 1}'),
            _completed('{"call": 2}'),
            _completed('{"call": 3}'),
            _completed('{"call": 4}'),
        ],
    )
    schema = {"type": "object", "properties": {"call": {"type": "integer"}}}

    first = OpenAIJSONClient(model="test-model", api_key="not-a-secret", cache_dir=tmp_path)
    assert first.ask_json("system", "prompt", schema=schema) == {"call": 1}

    second = OpenAIJSONClient(model="test-model", api_key="different-key", cache_dir=tmp_path)
    assert second.ask_json("system", "prompt", schema=schema) == {"call": 1}
    assert len(calls) == 1

    changed_schema = {**schema, "required": ["call"]}
    assert second.ask_json("system", "prompt", schema=changed_schema) == {"call": 2}
    different_model = OpenAIJSONClient(
        model="other-model", api_key="not-a-secret", cache_dir=tmp_path
    )
    assert different_model.ask_json("system", "prompt", schema=schema) == {"call": 3}
    assert second.ask_json("system", "different prompt", schema=schema) == {"call": 4}
    assert len(calls) == 4


def test_incomplete_response_is_not_cached(monkeypatch, tmp_path):
    incomplete = SimpleNamespace(
        status="incomplete", incomplete_details=SimpleNamespace(reason="limit")
    )
    calls = _install_openai_stub(monkeypatch, [incomplete, _completed('{"ok": 1}')])

    client = OpenAIJSONClient(api_key="stub", cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="limit"):
        client.ask("system", "prompt")

    new_client = OpenAIJSONClient(api_key="stub", cache_dir=tmp_path)
    assert new_client.ask("system", "prompt") == '{"ok": 1}'
    assert len(calls) == 2

    assert OpenAIJSONClient(api_key="stub", cache_dir=tmp_path).ask(
        "system", "prompt"
    ) == '{"ok": 1}'
    assert len(calls) == 2


def test_luna_defaults_to_medium_effort_and_replay_never_loads_sdk(monkeypatch, tmp_path):
    calls = _install_openai_stub(monkeypatch, [_completed('{"call": 1}')])
    live = OpenAIJSONClient(model="gpt-5.6-luna", api_key="stub", cache_dir=tmp_path)
    assert live.ask("system", "prompt") == '{"call": 1}'
    assert calls[0]["reasoning"] == {"effort": "medium"}
    assert "temperature" not in calls[0]

    monkeypatch.delitem(sys.modules, "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    replay = OpenAIJSONClient(model="gpt-5.6-luna", cache_dir=tmp_path, replay_only=True)
    assert replay.ask("system", "prompt") == '{"call": 1}'
    with pytest.raises(RuntimeError, match="replay cache miss"):
        replay.ask("system", "other prompt")


@pytest.mark.parametrize("bad", ["not json", '{"call": ', ""])
def test_malformed_json_is_not_cached(monkeypatch, tmp_path, bad):
    calls = _install_openai_stub(monkeypatch, [_completed(bad), _completed('{"call": 2}')])
    schema = {"type": "object", "properties": {"call": {"type": "integer"}}}
    with pytest.raises(json.JSONDecodeError):
        OpenAIJSONClient(api_key="stub", cache_dir=tmp_path).ask_json(
            "system", "prompt", schema=schema
        )
    assert list(tmp_path.iterdir()) == []

    live = OpenAIJSONClient(api_key="stub", cache_dir=tmp_path)
    assert live.ask_json("system", "prompt", schema=schema) == {"call": 2}
    replay = OpenAIJSONClient(cache_dir=tmp_path, replay_only=True)
    assert replay.ask_json("system", "prompt", schema=schema) == {"call": 2}
    assert len(calls) == 2


def test_luna_default_request_and_cache_key_are_pinned(monkeypatch, tmp_path):
    calls = _install_openai_stub(monkeypatch, [_completed('{"ok": true}')])
    client = OpenAIJSONClient(api_key="stub", cache_dir=tmp_path)
    assert client.ask_json("system", "prompt", schema={"type": "object"}, name="answer") == {
        "ok": True
    }
    assert calls == [{
        "model": "gpt-5.6-luna",
        "instructions": "system",
        "input": "prompt",
        "max_output_tokens": 4096,
        "text": {"format": {
            "type": "json_schema", "name": "answer", "strict": True,
            "schema": {"type": "object"},
        }},
        "reasoning": {"effort": "medium"},
    }]
    assert [path.name for path in tmp_path.iterdir()] == [
        "fc54e5487c27e2105ec6466134080ba4fa5550e8ee688e3a34c0524c82010d15.txt"
    ]


@pytest.mark.parametrize("model, reasoning", [
    ("gpt-5.6-luna", True), ("gpt-5.6", True), ("gpt-5", True), ("gpt-5-mini", True),
    ("o1", True), ("o3", True), ("o3-mini", True), ("o4-mini", True),
    ("gpt-4.1", False), ("gpt-4o", False),
])
def test_reasoning_model_family_detection(monkeypatch, model, reasoning):
    calls = _install_openai_stub(monkeypatch, [_completed("{}")])
    OpenAIJSONClient(model=model, api_key="stub").ask(
        "system", "prompt", temperature=0.2, max_tokens=800
    )
    request = calls[0]
    if reasoning:
        assert request["reasoning"] == {"effort": "medium"}
        assert "temperature" not in request
        assert request["max_output_tokens"] == 4096
    else:
        assert request["temperature"] == 0.2
        assert "reasoning" not in request
        assert request["max_output_tokens"] == 800


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-4.1"])
def test_request_kwargs_are_accepted_by_installed_sdk(monkeypatch, model):
    pytest.importorskip("openai")
    from openai.resources.responses import Responses

    parameters = inspect.signature(Responses.create).parameters
    if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        pytest.skip("Responses.create accepts arbitrary keyword arguments")
    accepted = set(parameters) - {"self"}

    calls = _install_openai_stub(monkeypatch, [_completed("{}")])
    OpenAIJSONClient(model=model, api_key="stub").ask_json(
        "system", "prompt", schema={"type": "object"}
    )
    assert set(calls[0]) <= accepted, set(calls[0]) - accepted
