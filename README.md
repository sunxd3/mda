# Model Discovery Agent

A clean-room, executable implementation of **Model Discovery Agent: LLM-assisted
Bayesian experiment design for data-efficient discovery of mechanistic world
models** ([arXiv:2608.09696](https://arxiv.org/abs/2608.09696)). The paper does
not publish the MDA implementation. This project reconstructs Algorithms 2–9
from the paper and uses the authors' released
[NeuronBench](https://github.com/murphyk/neuronbench) only as an environment and
reference baseline.

The LLM proposes model structures. Numerical code—not the LLM—fits parameters,
estimates marginal evidence, maintains the model posterior, checks residuals,
chooses experiments, and predicts outcomes.

```text
LLM structure proposals ──▶ model pool ──▶ full-batch evidence SMC
                                ▲                    │
                                │ failed check       ▼
experiment ◀── maximize VoI ◀── posterior ──▶ MAP / BMA prediction
    │                                                ▲
    └────────────── observation ─────────────────────┘
```

## What is implemented

- M-open model discovery with prequential predictive checks and residual-guided
  structure expansion.
- Adaptive likelihood-tempered parameter SMC, marginal-likelihood model weights,
  an explicit parameter-count prior, and deterministic top-N pruning.
- Exact Gaussian likelihoods for deterministic simulators and a bootstrap
  particle-filter likelihood for user-supplied stochastic state-space models.
- Model EIG, joint model/parameter EIG, and vector-valued task VoI.
- MAP and Bayesian-model-averaged prediction.
- Safe, AST-validated symbolic models and a structured OpenAI Responses API
  proposer, tested with `gpt-5.6-luna`.
- A deterministic NeuronBench adapter restoring the paper's fixed two-experiment
  seed set and six-dimensional electrophysiology feature prompt.

## Install and run

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'

# Full offline suite, including the authors' released NeuronBench contract.
.venv/bin/pytest -m 'not live'

# End-to-end offline law discovery.
.venv/bin/mda-toy

# Live LLM proposal check and end-to-end discovery. These call a paid API.
export OPENAI_API_KEY=...
RUN_LIVE_OPENAI=1 .venv/bin/pytest -q tests/test_openai_live.py
.venv/bin/mda-toy --live --model gpt-5.6-luna

# Authors' pure-LLM NeuronBench baseline through the same Luna client.
.venv/bin/mda-neuron-baseline \
  --world ca_rebound --budget 2 --model gpt-5.6-luna
```

`gpt-5.6-luna` does not accept `temperature`; the client omits it for GPT-5.6
models, requests low reasoning effort, and reserves enough output tokens for
strict-schema responses.

### Minimal API example

```python
import numpy as np

from mda import (
    AgentConfig,
    CandidateModel,
    Datum,
    DeterministicGaussianLikelihood,
    ModelDiscoveryAgent,
    SMCConfig,
    UniformPrior,
)

linear = CandidateModel(
    name="linear",
    structure="a + b*x",
    prior=UniformPrior(
        ("a", "b"), np.array([-5.0, -5.0]), np.array([5.0, 5.0])
    ),
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
next_experiment = agent.design()
agent.update_state(next_experiment, np.asarray(1.0 + 2.0 * next_experiment))
forecast = agent.predict(3.0, flavour="bma")
```

Pass a `ModelProposer` implementation to make the pool M-open. See
`SymbolicRegressionProposer` and `NeuronBenchProposer` for concrete structured
LLM boundaries. A stochastic model supplies `StateSpaceModel` callbacks and uses
`BootstrapParticleFilter` as its likelihood.

## Paper fidelity map

| Paper | Implementation |
|---|---|
| Algorithm 2, stateful MDA loop | `mda.agent.ModelDiscoveryAgent` |
| Algorithm 3, deterministic log-likelihood | `mda.likelihoods.DeterministicGaussianLikelihood` |
| Algorithm 4, bootstrap PF | `mda.likelihoods.BootstrapParticleFilter` |
| Algorithm 5, parameter/evidence SMC | `mda.smc.parameter_posterior` |
| Algorithm 6, model posterior and pruning | `mda.posterior.fit_model_posterior` |
| Algorithm 7, residual-conditioned expansion | `mda.proposer` plus the proposer implementations |
| Algorithm 8, predictive check | `mda.checks` |
| Algorithm 9, VoI variants | `mda.acquisition` |

Defaults match the paper's general setting where it specifies one: 16 live
models, four proposals per expansion, two search rounds, 100 parameter particles,
three rejuvenation moves, ESS target 0.6, 80 tempering rungs, a 0.05 error
threshold, and a 2-nat penalty per parameter. Domain experiments can override
these values, as the paper does.

### Necessary corrections to the pseudocode

The reconstruction makes four explicit corrections rather than reproducing
internally inconsistent pseudocode:

1. Algorithm 2 prints the data append before a check described as
   *pre-observation*. The implementation scores the fresh observation against the
   old belief first, then appends it.
2. Algorithm 6 prints pruning without returning the pruned support or
   renormalizing it. The implementation does both and records discarded mass.
3. The NeuronBench appendix describes clipped random-walk proposals, while
   Algorithm 5 uses a symmetric-proposal acceptance ratio. Out-of-prior proposals
   are rejected, preserving that ratio.
4. Equation 34 is implemented as the paper's covariance-trace disagreement
   surrogate. It is not labeled as exact multivariate mutual information.

The bootstrap PF gives an unbiased **likelihood** estimate. Its logarithm is not
unbiased, and fractional likelihood tempering is only approximately
pseudo-marginal, as the paper itself notes in footnote 10.

## NeuronBench and NeuroBench checks

The `neuron` extra pins the authors' NeuronBench benchmark at commit
`c354622458c460b419cab821d482c879f0578377`. The adapter never reads its private
truth mechanism. It uses the released simulator for candidate rollouts, restores
the paper's fixed brief/long-step `D0`, and gives the proposer the Appendix H
feature vector `[n_test, n_pre, run_down, adaptation, V_min, V_end]`.

The unrelated official [NeuroBench](https://github.com/NeuroBench/neurobench)
suite has no Hodgkin–Huxley mechanism-discovery or active-design baseline. It is
used here only to independently confirm the MSE definition. Both projects install
an import named `neurobench`, so run this check in a separate environment:

```bash
python -m venv .venv-neurobench
.venv-neurobench/bin/pip install -e '.[baselines]' pytest
.venv-neurobench/bin/pytest -q tests/test_neurobench_reference.py
```

### Released-benchmark limitations

The public NeuronBench repository is not the exact experimental harness described
by the paper:

- it ships no MDA, SMC, likelihood, or VoI implementation;
- its pure-LLM baseline starts without the paper's fixed `D0` and receives only
  spike counts rather than the paper's six features;
- its headline score is floored spike-count MSE and its secondary feature score
  uses fixed scales, rather than the paper's per-component query-variance nMSE;
- some released held-out protocols repeat brief/long protocols from the design
  menu despite the paper calling the sets disjoint; and
- it exposes stochastic rollouts, but not the one-step latent transition contract
  needed to fit arbitrary proposed mechanisms with a bootstrap PF. Its slow-Na
  time constant is also fixed rather than parameterized.

Therefore this repository does not pretend that the released deterministic
adapter reproduces the paper's stochastic NeuronBench results. The generic SMC³
path is complete and analytically tested; reproducing that domain result requires
the missing parameterized Fox–Lu candidate transition model.

## License

MIT. The separately installed benchmarks retain their own licenses.
