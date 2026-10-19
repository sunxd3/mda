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
    calls = _install_openai_stub(monkeypatch, [incomplete, _completed("ok")])

    client = OpenAIJSONClient(api_key="stub", cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="limit"):
        client.ask("system", "prompt")

    new_client = OpenAIJSONClient(api_key="stub", cache_dir=tmp_path)
    assert new_client.ask("system", "prompt") == "ok"
    assert len(calls) == 2

    assert OpenAIJSONClient(api_key="stub", cache_dir=tmp_path).ask(
        "system", "prompt"
    ) == "ok"
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
