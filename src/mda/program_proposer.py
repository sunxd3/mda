"""LLM-written experiment programs (``ProgramProposer``; an extension beyond v5).

The LLM sees the context, the data, the current models with their priors and
where their predictions disagree, imagined episodes of the best current
program under each model, and the current programs with their scores. It
returns program source, which is admitted only through the sandbox
(``mda.program_sandbox``). It never scores anything: the agent does that by
closed-loop simulation. Any ``ask_json`` client works (OpenAI, Anthropic,
Claude Code, Codex), with their caching and replay.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from .program_sandbox import SandboxError, SandboxedProgram, compile_program
from .programs import EpisodeSpec
from .types import CandidateModel, Datum


PROGRAM_API = """Write each program as the source of ONE Python class:

class Name(Program):
    def step(self, history, belief):
        ...
        return next_input

step() is called once per step of a {steps}-step episode and returns the next input,
a number (it is clipped to [{lower}, {upper}]). Programs must be deterministic and
stateless: recompute everything from the arguments on every call.
- history.inputs, history.obs: lists of the inputs issued and observations seen so far
  in this episode; history.t is the step index (len(history.inputs)), history.horizon
  the number of steps; history.obs_at(i) and history.last_obs() return None if absent.
- belief: the current posterior as nested dicts. belief["weights"][model] is each
  model's posterior probability; belief["parameters"][model][parameter] has "mean",
  "sd" and, for uniform priors, "low" and "high"; belief["pooled"][parameter] has the
  "mean" and "sd" of a parameter over every weighted model that has it.
Allowed: arithmetic, comparisons, and/or/not, x if c else y, if/elif/else, for i in
range(...), assignment, return, subscripts, tuples and lists, the functions min, max,
abs, len, range, float, int, round, any, all, sum, and math.*. Not allowed: imports,
while, other functions or methods, lambdas, comprehensions, try, with, dict or set
literals, f-strings, keyword arguments, names starting with _, self, and any other
attribute or call."""


class LLMProgramProposer:
    """Paper-style JSON proposer of sandboxed experiment programs."""

    system = (
        "You design adaptive experiments for Bayesian model discrimination. "
        "Write programs that choose each input from the observations so far so that "
        "the whole episode best tells the candidate models apart. Reply with JSON only."
    )

    def __init__(self, client: Any, episode: EpisodeSpec, **sandbox_limits) -> None:
        self.client = client
        self.episode = episode
        self.sandbox_limits = sandbox_limits

    def build_prompt(
        self,
        *,
        context: str,
        dataset: Sequence[Datum],
        current_models: Sequence[CandidateModel],
        disagreements: dict[str, Any],
        current_programs: Sequence[tuple[Any, float]],
        count: int,
    ) -> str:
        episode = self.episode
        observations = "\n".join(
            f"  inputs {np.round(np.asarray(datum.action, dtype=float), 6).tolist()} -> "
            f"observations {np.round(datum.observation, 6).tolist()}"
            for datum in dataset
        ) or "  (none)"
        models = json.dumps([
            {
                "name": model.name,
                "structure": model.structure,
                "parameters": list(model.prior.names),
            }
            for model in current_models
        ])
        programs = "\n\n".join(
            f"# {program.label}: {bits:.3f} bits\n{getattr(program, 'source', '(built-in program)')}"
            for program, bits in current_programs
        ) or "(none)"
        return f"""{context}

Each experiment is an episode of {episode.steps} steps. At each step you choose a scalar
input in [{episode.lower}, {episode.upper}] and then observe a noisy scalar response.

Completed episodes:
{observations}

Candidate models of the response:
{models}

Where the models' predictions of a one-step episode differ (posterior mean and SD of
the response at each input, by model):
{json.dumps(disagreements.get("predictions", []))}

Imagined episodes of the best current program, simulated under each model:
{json.dumps(disagreements.get("imagined_runs_of_best_program", {}))}

Current programs and their expected information about which model is true:
{programs}

{PROGRAM_API.format(steps=episode.steps, lower=episode.lower, upper=episode.upper)}

Propose {count} distinct programs that should carry more information than the
current ones. Return JSON only."""

    def propose(
        self,
        *,
        context: str,
        dataset: Sequence[Datum],
        current_models: Sequence[CandidateModel],
        disagreements: dict[str, Any],
        current_programs: Sequence[tuple[Any, float]],
        count: int,
    ) -> list[SandboxedProgram]:
        prompt = self.build_prompt(
            context=context, dataset=dataset, current_models=current_models,
            disagreements=disagreements, current_programs=current_programs, count=count,
        )
        schema = {
            "type": "object",
            "properties": {
                "programs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "rationale": {"type": "string"},
                            "source": {"type": "string"},
                        },
                        "required": ["name", "rationale", "source"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["programs"],
            "additionalProperties": False,
        }
        payload = self.client.ask_json(
            self.system, prompt, schema=schema, name="experiment_programs",
            max_tokens=max(1600, 800 * count),
        )
        existing = {program.key for program, _ in current_programs}
        programs = []
        for item in payload["programs"]:
            try:
                program = compile_program(item["source"], str(item["name"]).strip() or "program",
                                          **self.sandbox_limits)
            except (KeyError, TypeError, SandboxError):
                # Unsafe or broken source is dropped, as invalid laws are.
                continue
            if program.key not in existing:
                programs.append(program)
                existing.add(program.key)
        return programs[:count]
