"""Coding-agent SDK adapters: Claude Code and Codex as structured-output proposers.

Both clients implement ``ask_json`` like ``OpenAIJSONClient`` and share its
cache layout and replay rules. Each call runs one agent turn with no tools in an
empty temporary directory, so the harness only answers the prompt. The harness
still adds its own framing, so its responses are not interchangeable with the
direct API clients'; the cache key records the harness name to keep them apart.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from .openai_client import OpenAIJSONClient


class _AgentJSONClient:
    harness = ""

    def __init__(
        self,
        model: str,
        *,
        effort: str | None,
        cache_dir: Path | str | None = None,
        replay_only: bool = False,
        sample: str | None = None,
    ) -> None:
        if replay_only and cache_dir is None:
            raise ValueError("replay_only requires cache_dir")
        self.model = model
        self.effort = effort
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.replay_only = replay_only
        self.sample = sample
        self._request_counts: dict[str, int] = {}

    _read_cache = staticmethod(OpenAIJSONClient._read_cache)
    _write_cache = staticmethod(OpenAIJSONClient._write_cache)

    def _cache_path(self, request: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        # Same keying as the API clients: ``sample`` separates independent
        # runs, and the n-th repeat of an identical request is a fresh draw.
        base = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        repeat = self._request_counts.get(base, 0)
        self._request_counts[base] = repeat + 1
        sample = self.sample if repeat == 0 else f"{self.sample}/repeat={repeat}"
        keyed = request if sample is None else {"request": request, "sample": sample}
        canonical = json.dumps(
            keyed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(canonical).hexdigest()}.json"

    def _run(self, system: str, prompt: str, schema: dict[str, Any]) -> Any:
        raise NotImplementedError

    def ask_json(
        self,
        system: str,
        prompt: str,
        *,
        schema: dict[str, Any],
        name: str = "response",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return the harness's schema-validated answer.

        ``name``, ``temperature`` and ``max_tokens`` are accepted for parity with
        the API clients but neither sent nor keyed: the harnesses expose none.
        """
        request = {
            "harness": self.harness,
            "model": self.model,
            "effort": self.effort,
            "system": system,
            "user": prompt,
            "schema": schema,
        }
        cache_path = self._cache_path(request)
        if cache_path is not None:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return json.loads(cached)
        if self.replay_only:
            raise RuntimeError(f"{self.harness} replay cache miss")
        value = self._run(system, prompt, schema)
        if not isinstance(value, dict):
            raise ValueError("the model did not return a JSON object")
        if cache_path is not None:
            self._write_cache(cache_path, json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        return value


class ClaudeCodeJSONClient(_AgentJSONClient):
    """Claude Code through ``claude-agent-sdk``, with its JSON-schema output.

    Authentication is Claude Code's own: ``ANTHROPIC_API_KEY`` if set,
    otherwise a Claude subscription login or ``CLAUDE_CODE_OAUTH_TOKEN``.
    ``effort=None`` leaves Claude Code's default effort.
    """

    harness = "claude-code"

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        *,
        effort: str | None = None,
        cache_dir: Path | str | None = None,
        replay_only: bool = False,
        sample: str | None = None,
    ) -> None:
        super().__init__(model, effort=effort, cache_dir=cache_dir, replay_only=replay_only, sample=sample)

    def _run(self, system: str, prompt: str, schema: dict[str, Any]) -> Any:
        try:
            from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
        except ImportError as error:
            raise RuntimeError("install model-discovery-agent[agents] to use Claude Code") from error

        async def run():
            with tempfile.TemporaryDirectory() as cwd:
                options = ClaudeAgentOptions(
                    model=self.model,
                    effort=self.effort,
                    system_prompt=system,
                    tools=[],
                    strict_mcp_config=True,
                    setting_sources=[],
                    output_format={"type": "json_schema", "schema": schema},
                    cwd=cwd,
                    extra_args={"no-session-persistence": None},
                )
                result = None
                async for message in query(prompt=prompt, options=options):
                    if isinstance(message, ResultMessage):
                        result = message
                return result

        result = asyncio.run(run())
        if result is None or result.is_error or result.subtype != "success":
            reason = None if result is None else (result.errors or result.subtype)
            raise RuntimeError(f"Claude Code run did not complete: {reason}")
        return result.structured_output


class CodexJSONClient(_AgentJSONClient):
    """Codex through the official ``openai-codex`` SDK, with ``output_schema``.

    Authentication is Codex's own login (``codex login``, with ChatGPT or an
    API key). The thread is ephemeral, read-only, denies every approval, and
    has web search off. ``system`` replaces Codex's base instructions.
    """

    harness = "codex"

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        *,
        effort: str | None = "medium",
        cache_dir: Path | str | None = None,
        replay_only: bool = False,
        sample: str | None = None,
    ) -> None:
        super().__init__(model, effort=effort, cache_dir=cache_dir, replay_only=replay_only, sample=sample)

    def _run(self, system: str, prompt: str, schema: dict[str, Any]) -> Any:
        try:
            from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
        except ImportError as error:
            raise RuntimeError("install model-discovery-agent[agents] to use Codex") from error
        with tempfile.TemporaryDirectory() as cwd, Codex(CodexConfig(cwd=cwd)) as codex:
            thread = codex.thread_start(
                model=self.model,
                base_instructions=system,
                config={"web_search": "disabled"},
                cwd=cwd,
                ephemeral=True,
                sandbox=Sandbox.read_only,
                approval_mode=ApprovalMode.deny_all,
            )
            result = thread.run(prompt, effort=self.effort, output_schema=schema)
        status = getattr(result.status, "value", result.status)
        if status != "completed" or not result.final_response:
            raise RuntimeError(f"Codex turn did not complete: {status} {result.error}")
        return json.loads(result.final_response)
