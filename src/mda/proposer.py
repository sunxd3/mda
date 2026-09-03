"""The sole LLM boundary: structure birth proposals (Algorithm 7)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .types import CandidateModel, Datum


class ModelProposer(Protocol):
    def propose(
        self,
        *,
        context: str,
        dataset: Sequence[Datum],
        current_models: Sequence[CandidateModel],
        residuals: Sequence[dict[str, Any]],
        count: int,
    ) -> list[CandidateModel]: ...
