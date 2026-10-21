# Model Discovery Agent

A reimplementation of **Model Discovery Agent: LLM-assisted Bayesian experiment design for data-efficient discovery of mechanistic world models** ([arXiv:2608.09696v5](https://arxiv.org/abs/2608.09696v5), 27 Sep 2026).

The Model Discovery Agent (MDA) learns a mechanistic model of a system from very few experiments. A large language model proposes candidate model structures. Everything else is numerical: Bayesian inference weighs the candidates against the data, value-of-information design picks the next experiment, and predictive checks decide when the pool needs new structures.

```text
          ┌──────────────────────────────────────────────────────┐
          │                                                      │
          ▼                                                      │
┌───────────────────┐   ┌─────────────────┐   ┌──────────────┐   │
│ design()          │──▶│ run experiment  │──▶│ update_state │───┘
│ VoI over actions  │   │ (environment)   │   │ check, refit │
└───────────────────┘   └─────────────────┘   └──────┬───────┘
                                                     │ check failed
                                                     ▼
                                            ┌──────────────────┐
                                            │ LLM proposer     │
                                            │ new structures   │
                                            └──────────────────┘
```

## Features

- **M-open model discovery**: prequential predictive checks and residual-guided structure expansion, following v5's Algorithm 2.
- **Three evidence backends**: likelihood-tempered SMC (the default), recursive per-experiment updates, and multistart optimization with BIC.
- **Likelihoods**: exact Gaussian for deterministic simulators, a bootstrap particle filter for stochastic state-space models, and Bayesian synthetic likelihood.
- **Experiment design**: model EIG and joint model/parameter EIG over a finite menu or a continuous box searched by CMA-ES.
- **LLM proposers**: AST-validated symbolic laws and NumPyro programs, with OpenAI and Anthropic API clients or the Claude Code and Codex agents, response caching, and credential-free replay.
- **Benchmark runners** for the paper's domains: ChemBench, GlucoseBench, the stochastic neuron benchmark (NeuronBench), BoxingGym, and the v4-only ForceBench.

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'

.venv/bin/pytest -m 'not live'   # offline test suite
.venv/bin/mda-toy                # end-to-end offline law discovery
```

Use the agent as a library:

```python
import numpy as np

from mda import (
    AgentConfig, CandidateModel, Datum, DeterministicGaussianLikelihood,
    ModelDiscoveryAgent, SMCConfig, UniformPrior,
)

linear = CandidateModel(
    name="linear",
    structure="a + b*x",
    prior=UniformPrior(("a", "b"), np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
    predict=lambda theta, x: np.asarray(theta[0] + theta[1] * x),
)
agent = ModelDiscoveryAgent(
    context="A scalar input-output system.",
    initial_data=[Datum(0.0, np.asarray(1.0))],
    initial_models=[linear],
    likelihood=DeterministicGaussianLikelihood(0.1),
    actions=[-2.0, -1.0, 1.0, 2.0],
    observation_covariance=0.1**2,
    config=AgentConfig(smc=SMCConfig(parameter_particles=200)),
)
x = agent.design()                                  # choose the next experiment
agent.update_state(x, np.asarray(1.0 + 2.0 * x))    # absorb its result
forecast = agent.predict(3.0, flavour="bma")        # model-averaged prediction
```

Run one benchmark experiment without any LLM calls:

```bash
.venv/bin/mda-reproduce chemistry --world c0_michaelis_menten --smoke --budget 1 --output /tmp/chemistry.json
```

ChemBench and ForceBench need extra source checkouts. See [Reproducing the paper](docs/reproduce.md).

## Documentation

| Page | Contents |
|---|---|
| [Getting started](docs/getting-started.md) | Installation, optional extras, tests, demos, live LLM calls |
| [Concepts](docs/concepts.md) | The agent loop, its parts, and how they map to the paper |
| [Library guide](docs/library.md) | Building agents: models, priors, likelihoods, evidence, design, proposers, LLM clients, configuration |
| [Reproducing the paper](docs/reproduce.md) | The `mda-reproduce` driver: modes, flags, output, and safety |
| Benchmarks | [ChemBench](docs/benchmarks/chembench.md) · [GlucoseBench](docs/benchmarks/glucosebench.md) · [NeuronBench](docs/benchmarks/neuronbench.md) · [BoxingGym](docs/benchmarks/boxing.md) · [ForceBench (v4)](docs/benchmarks/forcebench.md) |
| [Replication](docs/replication.md) | Full ChemBench and GlucoseBench sweeps against the paper: results, protocol, the bugs that closed the gap, limits |
| [Paper fidelity](docs/paper-fidelity.md) | Algorithm-to-code map, v5 changes, corrections, open choices, and what still differs from the paper |

## Status

This is an independent reimplementation, not the authors' code. A full ChemBench sweep (171 cells, live GPT-5.6 Luna calls) passes 78.9% at 25 observations, against the paper's 78.4%, and a 30-patient GlucoseBench sweep reaches CGM nMSE 0.25 after six episodes, against the paper's 0.316; see [Replication](docs/replication.md). NeuronBench and BoxingGym have not been swept, and the paper's per-call LLM responses are not public, so its exact runs cannot be replayed. [Paper fidelity](docs/paper-fidelity.md#what-still-prevents-an-exact-paper-reproduction) lists every known difference.

## License

MIT. The separately installed benchmarks retain their own licenses.
