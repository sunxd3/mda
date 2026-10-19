"""OpenAI Responses API adapter used by proposers and reference baselines."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class OpenAIJSONClient:
    """Responses API client with optional local caching.

    When ``cache_dir`` is set, response text is persisted under request hashes
    in that directory. Callers are responsible for the privacy and
    lifecycle of that data. GPT-5.6 models default to medium reasoning effort,
    the setting of the v5 paper's main-text runs. ``replay_only`` serves the
    cache and fails on a miss without loading the SDK or credentials.
    """

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        api_key: str | None = None,
        *,
        cache_dir: Path | str | None = None,
        reasoning_effort: str = "medium",
        replay_only: bool = False,
    ) -> None:
        if replay_only and cache_dir is None:
            raise ValueError("replay_only requires cache_dir")
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.reasoning_effort = reasoning_effort
        self.replay_only = replay_only
        self.api_key = api_key
        self._client = None
        if not replay_only:
            self._sdk_client()

    def _sdk_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError("install model-discovery-agent[llm] to use OpenAI") from error
            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = OpenAI(api_key=key)
        return self._client

    def _cache_path(self, request: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        canonical = json.dumps(
            request, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(canonical).hexdigest()}.txt"

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
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as output:
                temporary = output.name
                output.write(text)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            temporary = None
        except OSError:
            # Caching is an optimization and must not make a successful API
            # response fail.
            pass
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def ask(
        self,
        system: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        output_format = response_format or {"type": "json_object"}
        is_5_6 = self.model.startswith("gpt-5.6-")
        request = dict(
            model=self.model,
            instructions=system,
            input=prompt,
            max_output_tokens=max(max_tokens, 4096) if is_5_6 else max_tokens,
            text={"format": output_format},
        )
        # The GPT-5.6 reasoning variants expose no sampling-temperature knob.
        if is_5_6:
            request["reasoning"] = {"effort": self.reasoning_effort}
        else:
            request["temperature"] = temperature
        cache_path = self._cache_path(request)
        if cache_path is not None:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached
        if self.replay_only:
            raise RuntimeError("OpenAI replay cache miss")
        response = self._sdk_client().responses.create(**request)
        if response.status != "completed":
            reason = getattr(response.incomplete_details, "reason", "unknown")
            raise RuntimeError(f"OpenAI response did not complete: {reason}")
        text = response.output_text
        if cache_path is not None and isinstance(text, str) and text.strip():
            self._write_cache(cache_path, text)
        return text

    def ask_json(
        self,
        system: str,
        prompt: str,
        *,
        schema: dict[str, Any],
        name: str = "response",
        temperature: float = 0.2,
        max_tokens: int = 1600,
    ) -> dict[str, Any]:
        text = self.ask(
            system,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "name": name,
                "strict": True,
                "schema": schema,
            },
        )
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("the model did not return a JSON object")
        return value
