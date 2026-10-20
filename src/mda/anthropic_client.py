"""Anthropic Messages API adapter with replayable, request-keyed caching."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class AnthropicJSONClient:
    """Anthropic client implementing the ``OpenAIJSONClient`` ask interface.

    SDK loading and credential validation are deferred until a cache miss, so
    cached paper runs can be replayed in an environment without Anthropic.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        api_key: str | None = None,
        *,
        cache_dir: Path | str | None = None,
        replay_only: bool = False,
    ) -> None:
        if replay_only and cache_dir is None:
            raise ValueError("replay_only requires cache_dir")
        self.model = model
        self.api_key = api_key
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.replay_only = replay_only
        self._client: Any | None = None

    def _cache_path(self, request: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        canonical = json.dumps(
            request, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(canonical).hexdigest()}.json"

    @staticmethod
    def _read_cache(path: Path) -> str | None:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None
        return text if text.strip() else None

    @staticmethod
    def _write_cache(path: Path, text: str) -> None:
        temporary: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", delete=False,
            ) as output:
                temporary = output.name
                output.write(text)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            temporary = None
        except OSError:
            pass
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _sdk_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as error:
                raise RuntimeError(
                    "install the Anthropic SDK to use AnthropicJSONClient"
                ) from error
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = Anthropic(api_key=key)
        return self._client

    def ask(
        self,
        system: str,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 800,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Return the forced tool call's input as compact JSON text.

        ``temperature`` is accepted for parity with ``OpenAIJSONClient`` but is
        not used: it is neither sent nor part of the cache key, because Opus 4.7
        rejects sampling parameters. Requests are streamed so that any
        ``max_tokens`` is accepted; the SDK refuses large non-streaming budgets.
        Only a response that stopped at the forced tool call is cached.
        """
        output_format = response_format or {"type": "json_object"}
        schema = output_format.get("schema")
        if schema is None:
            schema = {"type": "object", "additionalProperties": True}
        tool_name = str(output_format.get("name", "response"))

        # This representation is deliberately API-independent and includes
        # every effective input which can alter a response.
        effective_request = {
            "model": self.model,
            "system": system,
            "user": prompt,
            "schema": schema,
            "tool_name": tool_name,
            "token_limit": max_tokens,
        }
        cache_path = self._cache_path(effective_request)
        if cache_path is not None:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached
        if self.replay_only:
            raise RuntimeError("Anthropic replay cache miss")

        tool = {
            "name": tool_name,
            "description": "Return the requested structured response.",
            "input_schema": schema,
        }
        if output_format.get("type") == "json_schema":
            tool["strict"] = True
        with self._sdk_client().messages.stream(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
        ) as stream:
            response = stream.get_final_message()
        # A tool input cut off by max_tokens or a refusal can still parse.
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason != "tool_use":
            raise RuntimeError(f"Anthropic response did not complete: {stop_reason}")
        tool_inputs = [
            block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
            for block in getattr(response, "content", ())
            if (block.get("type") if isinstance(block, dict) else getattr(block, "type", None))
            == "tool_use"
            and (block.get("name") if isinstance(block, dict) else getattr(block, "name", None))
            == tool_name
        ]
        if len(tool_inputs) != 1 or not isinstance(tool_inputs[0], dict):
            raise RuntimeError("Anthropic response did not contain the forced tool output")
        text = json.dumps(tool_inputs[0], ensure_ascii=False, separators=(",", ":"))
        if cache_path is not None:
            self._write_cache(cache_path, text)
        return text

    def ask_json(
        self,
        system: str,
        prompt: str,
        *,
        schema: dict[str, Any],
        name: str = "response",
        temperature: float = 0.3,
        max_tokens: int = 1600,
    ) -> dict[str, Any]:
        """Return the strict-schema tool input; ``temperature`` is unused (see ``ask``)."""
        text = self.ask(
            system, prompt, temperature=temperature, max_tokens=max_tokens,
            response_format={
                "type": "json_schema", "name": name, "strict": True, "schema": schema
            },
        )
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("the model did not return a JSON object")
        return value
