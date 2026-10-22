"""LLM boundaries: structure proposals (``init-hyp-space``/``expand-hyp-space``) and, beyond v5, experiment programs."""

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


class ProgramProposer(Protocol):
    """LLM boundary for experiment programs (an extension beyond v5).

    Mirrors ``ModelProposer``: it writes candidates and never scores them.
    ``disagreements`` holds where the current models' predictions differ and
    imagined episodes of the best current program under each model;
    ``current_programs`` pairs each pooled program with its latest score in bits.
    """

    def propose(
        self,
        *,
        context: str,
        dataset: Sequence[Datum],
        current_models: Sequence[CandidateModel],
        disagreements: dict[str, Any],
        current_programs: Sequence[tuple[Any, float]],
        count: int,
    ) -> list[Any]: ...
