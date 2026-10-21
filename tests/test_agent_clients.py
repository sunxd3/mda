import json
import sys
from types import SimpleNamespace

import pytest

from mda.agent_clients import ClaudeCodeJSONClient, CodexJSONClient

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["value"],
    "properties": {"value": {"type": "integer"}},
}


def _stub_claude(monkeypatch, results):
    calls = []
    queued = iter(results)

    class ResultMessage(SimpleNamespace):
        pass

    def ClaudeAgentOptions(**options):
        return SimpleNamespace(**options)

    async def query(*, prompt, options):
        calls.append({"prompt": prompt, **vars(options)})
        yield SimpleNamespace(type="assistant")
        yield ResultMessage(**next(queued))

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", SimpleNamespace(
        ClaudeAgentOptions=ClaudeAgentOptions, ResultMessage=ResultMessage, query=query))
    return calls


def _claude_result(value, subtype="success", is_error=False):
    return dict(subtype=subtype, is_error=is_error, structured_output=value, errors=None)


def _stub_codex(monkeypatch, results):
    calls = []
    queued = iter(results)

    class Thread:
        def __init__(self, started):
            self.started = started

        def run(self, prompt, **turn):
            calls.append({"prompt": prompt, "thread": self.started, "turn": turn})
            return next(queued)

    class Codex:
        def __init__(self, config):
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def thread_start(self, **started):
            return Thread({**started, "launch": self.config})

    catalog = {"models": [
        {"slug": "gpt-5.6-luna", "tool_mode": "code_mode_only", "apply_patch_tool_type": "freeform",
         "supports_search_tool": True, "multi_agent_version": "v1", "default_verbosity": "low"},
        {"slug": "other", "tool_mode": "code_mode_only"},
    ]}
    monkeypatch.setitem(sys.modules, "codex_cli_bin", SimpleNamespace(bundled_codex_path=lambda: "codex"))
    monkeypatch.setattr("mda.agent_clients.subprocess.run",
                        lambda *a, **k: SimpleNamespace(stdout=json.dumps(catalog)))
    monkeypatch.setitem(sys.modules, "openai_codex", SimpleNamespace(
        ApprovalMode=SimpleNamespace(deny_all="deny_all"), Codex=Codex,
        CodexConfig=lambda **config: SimpleNamespace(**config),
        Sandbox=SimpleNamespace(read_only="read-only")))
    return calls


def _codex_result(value, status="completed"):
    text = None if value is None else json.dumps(value)
    return SimpleNamespace(status=SimpleNamespace(value=status), final_response=text, error=None)


def test_claude_code_runs_tool_free_with_the_schema(monkeypatch, tmp_path):
    calls = _stub_claude(monkeypatch, [_claude_result({"value": 3})])
    client = ClaudeCodeJSONClient(model="claude-x", effort="medium")
    assert client.ask_json("sys", "user", schema=SCHEMA) == {"value": 3}
    (call,) = calls
    assert call["prompt"] == "user" and call["system_prompt"] == "sys"
    assert call["model"] == "claude-x" and call["effort"] == "medium"
    assert call["tools"] == [] and call["setting_sources"] == [] and call["strict_mcp_config"]
    assert call["output_format"] == {"type": "json_schema", "schema": SCHEMA}


def test_codex_runs_read_only_with_the_schema(monkeypatch):
    calls = _stub_codex(monkeypatch, [_codex_result({"value": 3})])
    assert CodexJSONClient().ask_json("sys", "user", schema=SCHEMA) == {"value": 3}
    (call,) = calls
    assert call["thread"]["model"] == "gpt-5.6-luna" and call["thread"]["base_instructions"] == "sys"
    assert call["thread"]["sandbox"] == "read-only" and call["thread"]["approval_mode"] == "deny_all"
    assert call["thread"]["ephemeral"] is True
    assert call["turn"] == {"effort": "medium", "output_schema": SCHEMA}


def test_codex_launches_with_its_own_context_and_tools_switched_off(monkeypatch):
    calls = _stub_codex(monkeypatch, [_codex_result({"value": 3})])
    client = CodexJSONClient()
    client.ask_json("sys", "user", schema=SCHEMA)
    overrides = calls[0]["thread"]["launch"].config_overrides
    for setting in ("include_permissions_instructions=false", "include_environment_context=false",
                    "skills.include_instructions=false", "features.shell_tool=false"):
        assert setting in overrides
    catalog_path = json.loads(overrides[0].removeprefix("model_catalog_json="))
    luna, other = json.load(open(catalog_path))["models"]
    assert luna == {"slug": "gpt-5.6-luna", "tool_mode": None, "apply_patch_tool_type": None,
                    "supports_search_tool": False, "multi_agent_version": None,
                    "default_verbosity": "medium"}
    assert other["tool_mode"] == "code_mode_only"


@pytest.mark.parametrize("make, stub, ok, module", [
    (ClaudeCodeJSONClient, _stub_claude, _claude_result, "claude_agent_sdk"),
    (CodexJSONClient, _stub_codex, _codex_result, "openai_codex"),
])
def test_cache_replays_draws_in_order_without_the_sdk(monkeypatch, tmp_path, make, stub, ok, module):
    calls = stub(monkeypatch, [ok({"value": 1}), ok({"value": 2})])
    live = make(cache_dir=tmp_path, sample="chemistry/w/seed=0")
    first = [live.ask_json("sys", "user", schema=SCHEMA) for _ in range(2)]
    assert first == [{"value": 1}, {"value": 2}]

    monkeypatch.delitem(sys.modules, module)
    replay = make(cache_dir=tmp_path, replay_only=True, sample="chemistry/w/seed=0")
    assert [replay.ask_json("sys", "user", schema=SCHEMA) for _ in range(2)] == first
    with pytest.raises(RuntimeError, match="replay cache miss"):
        replay.ask_json("sys", "user", schema=SCHEMA)
    with pytest.raises(RuntimeError, match="replay cache miss"):
        make(cache_dir=tmp_path, replay_only=True, sample="chemistry/w/seed=1").ask_json(
            "sys", "user", schema=SCHEMA)
    assert len(calls) == 2


def test_harnesses_do_not_share_cache_entries(monkeypatch, tmp_path):
    _stub_claude(monkeypatch, [_claude_result({"value": 1})])
    ClaudeCodeJSONClient(model="m", effort="medium", cache_dir=tmp_path).ask_json("sys", "user", schema=SCHEMA)
    with pytest.raises(RuntimeError, match="replay cache miss"):
        CodexJSONClient(model="m", cache_dir=tmp_path, replay_only=True).ask_json("sys", "user", schema=SCHEMA)


@pytest.mark.parametrize("make, stub, failed", [
    (ClaudeCodeJSONClient, _stub_claude, _claude_result(None, subtype="error_max_turns", is_error=True)),
    (ClaudeCodeJSONClient, _stub_claude, _claude_result(None, subtype="success", is_error=True)),
    (CodexJSONClient, _stub_codex, _codex_result({"value": 1}, status="failed")),
    (CodexJSONClient, _stub_codex, _codex_result(None)),
])
def test_incomplete_runs_raise_and_are_not_cached(monkeypatch, tmp_path, make, stub, failed):
    stub(monkeypatch, [failed])
    with pytest.raises(RuntimeError, match="did not complete"):
        make(cache_dir=tmp_path).ask_json("sys", "user", schema=SCHEMA)
    assert not list(tmp_path.iterdir())
