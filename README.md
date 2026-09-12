# Model Discovery Agent

A clean-room, executable implementation of **Model Discovery Agent: LLM-assisted
Bayesian experiment design for data-efficient discovery of mechanistic world
models** ([arXiv:2608.09696v4](https://arxiv.org/html/2608.09696v4)). The paper does
not publish the MDA implementation. This project reconstructs Algorithms 2–9
from the paper and uses the authors' released
[NeuronBench](https://github.com/murphyk/neuronbench) for protocols, deterministic
HH simulation, and a reference baseline.

This is a reconstruction of the core algorithms, **not a reproduction of the
paper's reported benchmark results**. The supported parameter prior is a bounded
independent uniform prior; the NumPyro latent-program backend is not implemented.

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
- Model EIG, joint model/parameter EIG, and coordinatewise task VoI, with a
  separately named full-observation Gaussian alternative.
- Finite-menu design or bounded CMA-ES optimization, including logarithmic axes.
- MAP and Bayesian-model-averaged prediction.
- AST-validated symbolic models and a structured OpenAI Responses API proposer,
  with optional persistent response caching. AST validation restricts syntax;
  it is not a resource sandbox for arbitrary untrusted computation.
- A deterministic NeuronBench adapter restoring the fixed two-experiment seed
  set, six-feature prompt, selectable activation/inactivation kinetics, inferred
  slow-Na time constant, and component-normalized query MSE evaluation.

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

### Continuous design and proposal replay

Install `.[design]` for CMA-ES (included in `.[dev]`). Supply `actions=[]` and an
optimizer to the agent, for example:

```python
from mda import CMAESOptimizer

optimizer = CMAESOptimizer(
    [0.001, 280.0], [10.0, 330.0],
    log_axes=[True, False],
    decode=lambda x: {"concentration": float(x[0]), "temperature": float(x[1])},
    max_evaluations=200,
)
# ModelDiscoveryAgent(..., actions=[], optimizer=optimizer)
```

An optimizer receives `(score, rng)`, maximizes `score(action)`, and must return
an evaluated action. The agent excludes used `action_key`s unless
`repeat_designs=True`. CMA-ES returns its best finite evaluated design and honors
its evaluation limit. For continuous actions, use a stable, collision-free
`action_key`; the default `repr` is suitable only for small actions.

`OpenAIJSONClient(cache_dir=".cache/mda-responses")` caches completed, nonempty
responses by the complete effective request, including the model and schema.
The key excludes credentials; cached response text may still contain sensitive
data. Keep the directory private and out of version control. The default is no
cache. This enables replay of captured calls, not reproduction of the paper's
unreleased Opus-4.7 responses. Luna remains the default provider/model choice.

Pass `experiment_budget=B` to include the remaining budget in proposal context
and prevent `design()` after B updates. Proposals are labeled `explore`: the core
loop invokes them only at initialization or after a failed adequacy check, not
to tune an already adequate pool. Symbolic proposals receive signed residual/input
correlations; both proposers receive the entire model pool, priors, and per-example
residual reports.

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

Defaults mostly follow the paper's **ChemBench** settings: 16 live
models, four proposals per expansion, two search rounds, 100 parameter particles,
three rejuvenation moves, ESS target 0.6, 80 tempering rungs, a 0.05 error
threshold, and a 2-nat penalty per parameter. Domain experiments can override
these values, as the paper does.

Optional Appendix B.3 shrinking is enabled with `minimum_model_cap=6` in
`AgentConfig`. It requires both a well-fit MAP model and posterior confidence
at least `shrink_confidence` (default 0.95, a local choice because the paper gives
no numerical confidence cutoff). A failed check or reduced confidence restores
the larger cap; evicted models are not automatically resurrected. Shrinking is
disabled by default, matching the base Algorithm 2.

`BeliefState.log_mixture_evidence` is computed before pruning over the deduplicated
input pool, with its parameter-count model prior normalized over that pool.
`discarded_mass` includes support truncation. Each parameter posterior records
`ess_before_resampling` and `last_rung_forced_below_target`: reaching the rung
limit forces the full likelihood (`beta=1`), possibly at the cost of low ESS.
Increase the rung/particle budget if these diagnostics show collapse.

### Task-VoI has explicit target-space units

`design_flavour="task"` uses Eq. 43's target predictions at **both** the proposed
and query actions, with each target coordinate scored independently. It averages
across query actions and sums coordinate contributions, whitening the loss by
the target-space noise variances. `task_noise_covariance` supplies those variances
to the agent; when omitted it falls back to `observation_covariance`. Use a
separate target-space value for features derived from raw traces. `target_covariance`
overrides the loss whitening (set it to 1 for unwhitened squared loss). Both must
be scalar or diagonal for this flavour.

The old dense matrix conditional-variance calculation remains available as
`design_flavour="task_full_observation"` or
`full_observation_task_value_of_information`. It uses raw observations at the
proposed action and all cross-coordinate information; optional `target_covariance`
whitens its loss. This is a coherent Gaussian alternative inspired by Eq. 42,
but it is **not** the coordinatewise implementation described after Eq. 45.
Neither objective is exact information gain for a general non-Gaussian mixture.

### Necessary corrections to the pseudocode

The reconstruction makes explicit corrections rather than reproducing
internally inconsistent pseudocode:

1. Algorithm 2 prints the data append before a check described as
   *pre-observation*. The implementation scores the fresh observation against the
   old belief first, then appends it.
2. Algorithm 6's top-N support truncation is implemented with explicit
   renormalization of the retained weights and a discarded-mass diagnostic.
3. The NeuronBench appendix describes clipped random-walk proposals, while
   Algorithm 5 uses a symmetric-proposal acceptance ratio. Out-of-prior proposals
   are rejected, preserving that ratio.
4. Equation 34 is implemented as the paper's covariance-trace disagreement
   surrogate. It is not labeled as exact multivariate mutual information.
5. Later search rounds use an in-sample check after refitting, honoring the
   `backtest` flag. Only the first check can be genuinely pre-observation.
6. `get_model()` selects the current-data Occam-MAP. Algorithm 2's “over any
   round” rule does not define a comparable historical selection score.
   Historical posterior masses are not compared, and held-out truth is never
   used to select the agent's model. Best-so-far evaluation curves are a separate
   reporting convention, not this agent's selection policy.

The bootstrap PF gives an unbiased **likelihood** estimate. Its logarithm is not
unbiased, and fractional likelihood tempering is only approximately
pseudo-marginal, as the paper itself notes in footnote 10.

## NeuronBench and NeuroBench checks

The `neuron` extra pins the authors' NeuronBench benchmark at
[`c354622`](https://github.com/murphyk/neuronbench/commit/c354622458c460b419cab821d482c879f0578377).
The adapter never reads its private
truth mechanism. It uses the released simulator for candidate rollouts, restores
the paper's fixed brief/long-step `D0`, and gives the proposer the Appendix H
feature vector `[n_test, n_pre, run_down, adaptation, V_min, V_end]`.

`NeuronHypothesis.tau_gate` selects activation (Ih/M) or inactivation (Ca/D),
with `activation_tau_ms` fixed in the latter case. Anonymous channel proposals
remain supported; their bounds and fixed gating forms come from the proposer,
not a hidden truth-based archetype lookup. Table 14 and H.4.1 disagree about
which gate's tau is inferred; this API makes that choice explicit.

Slow-Na candidates infer `gL ~ Uniform(0.2,0.45)` and
`tau_s_ms ~ Uniform(200,600)`. A local deterministic rollout extends the pinned
simulator's equations without changing the dependency: the sampled tau replaces
the depolarized branch only, with the recovery branch fixed at 1500 ms. At
380 ms it reproduces the released rollout. This interpretation of Table 14's
single tau is explicit rather than claiming the paper specifies both branches.

Voltage traces remain in raw mV. Interpreting Table 14's “1 unit = 3 mV” as
`V/3`, `observation_std` returns `max(0.12*abs(V), 3)` mV. The table mixes feature
and trace terminology, so this is a documented unit interpretation, not a
verified reconstruction of the authors' unavailable likelihood code.

`environment.evaluate_predictor(agent.get_predictor())` computes Eq. 1's mean
per-component query-variance-normalized MSE without adding queries to training
data. It raises for zero query variance rather than introducing an undocumented
floor. It inherits the released query set, including overlaps described below.

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
  needed to fit arbitrary proposed mechanisms with a bootstrap PF.

Therefore this repository does not pretend that the released deterministic
adapter reproduces the paper's stochastic NeuronBench results. The generic nested
PF/parameter-SMC/model-posterior path is tested against integrated multi-step
Gaussian evidence and model probabilities. That controlled test does not
establish calibration for nonlinear neuron dynamics. A parameterized Fox–Lu
candidate transition model and stochastic expected-feature predictor are still
needed for that domain reproduction.

There are no full ForceBench, ChemBench, BoxingGym, or stochastic NeuronBench
comparison runners here, and no claim of matched paper scores, prompts, seeds,
or budgets. The neuron CLI is the released pure-LLM baseline, not an MDA benchmark
battery. Learned summary statistics are future work in v4, not a missing core
algorithm requirement.

## License

MIT. The separately installed benchmarks retain their own licenses.
