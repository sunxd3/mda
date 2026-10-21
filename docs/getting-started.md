# Getting started

## Requirements

Python 3.11 or newer. The core library depends only on NumPy. Everything else is an optional extra.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

`.[dev]` installs pytest plus the `bic`, `design`, `glucose`, `llm`, `neuron`, `numpyro`, and `recovery` extras. The repository's `.agents/setup` script does the same.

### Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `llm` | `openai`, `anthropic` | LLM proposers and the ICL baseline |
| `agents` | `claude-agent-sdk`, `openai-codex` | Claude Code and Codex as LLM providers |
| `numpyro` | `numpyro`, `jax` | `NumPyroPrior` and generated NumPyro models (Boxing) |
| `design` | `cma` | `CMAESOptimizer` continuous design (ChemBench) |
| `bic` | `scipy` | `BICEvidence` |
| `recovery` | `scipy`, `sympy` | Structural recovery diagnostics (`--recovery`) |
| `neuron` | [neuronbench](https://github.com/murphyk/neuronbench) at `c354622` | NeuronBench domain |
| `glucose` | [glucosebench](https://github.com/murphyk/glucosebench) at `6bb0ff6`, `scipy` | GlucoseBench domain |
| `physics` | DiscoverPhysics `PhysicsSchool` and `ScienceAgent` at `33b7fa9` | ForceBench (v4 legacy) |
| `baselines` | `neurobench` | Independent NeuroBench metric check (separate environment) |

ChemBench and ForceBench also need source checkouts. See [Reproducing the paper](reproduce.md#extra-setup-for-chembench-and-forcebench).

## Run the tests

```bash
# Full offline suite, including the authors' released NeuronBench contract.
.venv/bin/pytest -m 'not live'
```

Tests marked `live` call a paid API and are skipped unless `RUN_LIVE_OPENAI=1` is set.

The official [NeuroBench](https://github.com/NeuroBench/neurobench) suite is used only to confirm the NeuronBench MSE definition. Both projects install an import named `neurobench`, so run that check in a separate environment:

```bash
python -m venv .venv-neurobench
.venv-neurobench/bin/pip install -e '.[baselines]' pytest
.venv-neurobench/bin/pytest -q tests/test_neurobench_reference.py
```

## Command-line tools

| Command | Purpose |
|---|---|
| `mda-toy` | End-to-end scalar law discovery on a known quadratic; prints the discovered law, its parameters, the selected experiments, and held-out nMSE |
| `mda-reproduce` | Run one paper benchmark configuration and write a JSON result. See [Reproducing the paper](reproduce.md) |
| `mda-neuron-baseline` | The NeuronBench authors' pure-LLM baseline through the same OpenAI client |

### Offline demo

```bash
.venv/bin/mda-toy
```

The demo starts from four seed laws (constant, linear, quadratic, exponential), runs three designed experiments, and reports the MAP law. It makes no network calls.

### Live LLM calls

These commands call a paid API.

```bash
export OPENAI_API_KEY=...

# Live proposal check.
RUN_LIVE_OPENAI=1 .venv/bin/pytest -q tests/test_openai_live.py

# End-to-end discovery with LLM-proposed laws instead of the seed library.
.venv/bin/mda-toy --live --model gpt-5.6-luna

# The authors' pure-LLM NeuronBench baseline.
.venv/bin/mda-neuron-baseline --world ca_rebound --budget 2 --model gpt-5.6-luna
```

Anthropic runs need `ANTHROPIC_API_KEY`. `--provider claude-code` and `--provider codex` use those agents' own logins instead; see [coding-agent clients](library.md#coding-agent-clients). Responses can be cached and replayed without credentials; see [LLM clients](library.md#llm-clients-caching-and-replay).

## Next steps

- [Concepts](concepts.md) explains the agent loop.
- [Library guide](library.md) shows how to build an agent for your own system.
- [Reproducing the paper](reproduce.md) runs the paper's benchmarks.
