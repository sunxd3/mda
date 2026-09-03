"""OpenAI Responses API adapter used by proposers and reference baselines."""

from __future__ import annotations

import json
import os
from typing import Any


class OpenAIJSONClient:
    def __init__(self, model: str = "gpt-5.6-luna", api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("install model-discovery-agent[llm] to use OpenAI") from error
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model
        self._client = OpenAI(api_key=key)

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
            request["reasoning"] = {"effort": "low"}
        else:
            request["temperature"] = temperature
        response = self._client.responses.create(**request)
        if response.status != "completed":
            reason = getattr(response.incomplete_details, "reason", "unknown")
            raise RuntimeError(f"OpenAI response did not complete: {reason}")
        return response.output_text

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
