# Model Discovery Agent

A clean-room, executable implementation of **Model Discovery Agent: LLM-assisted
Bayesian experiment design for data-efficient discovery of mechanistic world
models** ([arXiv:2608.09696v4](https://arxiv.org/html/2608.09696v4)). The paper does
not publish the MDA implementation. This project reconstructs Algorithms 2–9
from the paper and uses the authors' released
[NeuronBench](https://github.com/murphyk/neuronbench) for protocols, deterministic
HH simulation, and a reference baseline.

This is an executable reconstruction, **not a reproduction of the paper's
reported benchmark results**. It includes single-run drivers for all four
benchmark families, including stochastic NeuronBench, and a NumPyro
latent-program backend. Missing public experimental inputs and remaining
comparison-harness limits are listed below; smoke runs are not benchmark scores.

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
- A Fox–Lu stochastic candidate with latent-state PF callbacks, 200-rollout
  expected-feature forecasts, and independent held-out targets.
- NumPyro continuous scalar/array priors with support transforms and Jacobians;
  sampler-coordinate inference and physical-coordinate MAP ranking. Discrete,
  observed, and dimension-changing (e.g. simplex) sites are explicitly rejected.
- Cached Anthropic Opus proposals, credential-free offline replay, scalar
  GLM likelihoods, an ICL forecaster, and finite-grid constant-refit recovery tests.

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

### Reproduction driver: one explicit run at a time

```bash
.venv/bin/pip install -e '.[dev,numpyro,physics]'

# PhysicsSchool's pinned wheel omits namespace subpackages. It needs an
# editable checkout in addition to the physics extra above.
git clone https://github.com/SampsonML/DiscoverPhysics.git /tmp/DiscoverPhysics
git -C /tmp/DiscoverPhysics checkout 33b7fa9df96de9c35744efd181ca7e5a8dd60ad5
.venv/bin/pip install -e /tmp/DiscoverPhysics/PhysicsSchool

# ChemBench is an unpackaged source dependency (NumPy oracle only; no PySR).
git clone https://github.com/scientific-discovery/LLM-AutoSciLab.git /tmp/LLM-AutoSciLab
git -C /tmp/LLM-AutoSciLab checkout 9a37255ff5432a7155b6cde8985e9654f2f8f272
export AUTOSCILAB_SOURCE_ROOT=/tmp/LLM-AutoSciLab

.venv/bin/mda-reproduce boxing --world death --smoke --budget 1 --output /tmp/boxing.json
.venv/bin/mda-reproduce boxing --world death --agent apprentice --smoke --budget 1 --output /tmp/apprentice.json
.venv/bin/mda-reproduce physics --world gravity --smoke --budget 1 --output /tmp/physics.json
.venv/bin/mda-reproduce chemistry --world c0_michaelis_menten --smoke --budget 1 --output /tmp/chemistry.json
.venv/bin/mda-reproduce neuron --world h_sag --stochastic --smoke --budget 1 --output /tmp/neuron.json
```

The driver requires exactly one of `--smoke`, `--replay`, or `--live`. Smoke
uses small numerical budgets and generic, non-truth baselines; it makes no LLM
calls. Neuron smoke checks feature forecasts but omits expensive held-out
200-rollout scoring. Replay reads previously captured Anthropic responses and
fails on cache misses without contacting a provider. Live explicitly authorizes
paid calls; it defaults to `claude-opus-4-7`, temperature 0.3. `--model` overrides
the provider model ID. No model availability or live score claim is made here.

Normal runs use Table 2 settings. Specify `--budget` explicitly (the conservative
CLI default is one experiment, not a paper sweep). `--design` selects model,
joint, task, or finite-menu random design; `--no-m-open` disables expansion after
the initial pool. Boxing defaults to BMA prediction; the other domains use MAP.
Neuron task design requires an explicit `--task-noise-variance` (one or six
positive feature-space variances), since the paper does not release that
calibration. It must not silently use raw voltage noise for six different features.

`--agent apprentice` implements the Boxing H.3.2 comparison: one evolving
model, LLM-selected experiments, and a residual-conditioned refinement on every
round, fitted by the same SMC. `--icl` adds the data-only forecasting baseline
after every round (replay/live only). `--recovery` adds ChemBench or central
two-particle ForceBench constant-refit recovery scoring (install `.[recovery]`);
recovery parameters never replace the agent's fitted parameters.
The physics driver also accepts `circle`, `ether`, and `hubble`. These extensions
use documented local launch menus because the exact paper menus are unreleased.

IRT, location, and survival generate NumPyro Python. Non-smoke runs require
`--allow-code-execution`: **this executes arbitrary generated Python with process
privileges, not in a security sandbox**. Use only a suitably isolated environment
and trusted/reviewed proposals. The AST symbolic backend does not require this flag.

JSON results contain the request, package versions, effective numerical settings,
per-round actions, observations, current model/parameters, model weights,
log-evidences, forced-tempering diagnostics, scores, and a separately labeled
best-so-far curve. Evaluation does not select models or feed query labels into
training. Output files are never intentionally overwritten.

`mda.benchmarks.icl.make_icl_predictor` provides a model-free baseline using only
observed action-target pairs and query inputs. `mda.benchmarks.recovery.evaluate_recovery`
refits a trusted candidate's constants on a separate noise-free grid and tests
held-out RMSLE. The benchmark evaluators first compile the symbolic law using
`canonical_symbolic_candidate`: a whitelist AST translation into SymPy, followed
by simplification and numerical compilation, without parsing arbitrary Python
through `sympify`. This is a numerical approximation, not proof of symbolic equality.
ForceBench measures the true radial force through clean public release-at-rest
experiments, never private force fields. Its local grid has 128 interleaved
log-spaced radii in [1.5, 10], with a .01 time-unit finite difference; it uses
signed `log1p` and tolerance .05. This tests the radial profile at unit coupling
and time zero, not the complete charge/time dependence. Chemistry uses separate
128-point fit and holdout grids and ordinary `log1p` with tolerance .02. Neither
grid is claimed to be the authors' unreleased evaluation grid.

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
unreleased Opus-4.7 responses. The toy CLI uses Luna; the reproduction driver uses
Anthropic by default. `AnthropicJSONClient(..., replay_only=True)` fails before
SDK loading or credential checks on a missing cached response.

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

The stochastic adapter now supplies the missing transition contract and is tested
against pinned upstream rollouts. PF integration is separately checked against
analytic multi-step Gaussian evidence. Neither test establishes nonlinear-neuron
calibration or reproduces the paper's scores. Its observed-feature summaries use
the stochastic sampling grid, but cannot recover spikes missed between ~4 ms
samples; expected targets use full simulated traces. Table 2's 200/.5 particle/ESS
settings are used rather than Table 14's conflicting deterministic 48/.6 settings.

### What still prevents an exact paper reproduction

- The paper's complete cached Opus outputs, exact prompts/configuration files,
  validation choices, and ChemBench 36-task manifest are not publicly supplied.
  The pinned ChemBench registry differs from the paper's stated universe. The
  driver requires an explicit domain ID; it does not invent a matching subset.
- ChemBench follows Appendix B.2's Gaussian fit in `log1p(rate)` space with SD
  .01, whereas the main text describes raw-rate multiplicative noise. Seed-point
  companion values are documented local defaults. The upstream oracle's noise
  is input-hashed, not a new independent draw for each run seed.
- ForceBench covers the six central two-particle worlds plus Circle, Ether,
  and Hubble. Dark-matter/three-species inference, the larger MIRT/curriculum
  example, and exact external DP-Agent/AutoSciLab comparison harnesses are not implemented.
  Thus the main-domain MDA runner is not a complete all-figures reproduction.
  Ether follows the public simulator's common drift of the anchor and probes,
  rather than interpreting the appendix's anchor as immovable. Extension
  background-coefficient bounds [-1,1] and launch templates are local choices.
- Boxing uses reconstructed equations with private RNGs, not the original PyMC
  execution or identical RNG draws. IRT and survival use fixed outcomes and
  Brier loss; the remaining tasks use clean-mean nMSE. Dugongs follows Table 8's
  Gaussian observation, not upstream's folded output. Generic 24-design/12-query
  pools are configurable local defaults, not claimed paper evaluation grids.
- Recovery covers chemistry and the six central-force worlds, not the many-body
  extensions or neuron channel classification. Large benchmark runs, paid
  proposal calls, statistical score matching, and full comparison plots remain
  unexecuted. Learned summaries are future work in v4, not a missing algorithm.

## License

MIT. The separately installed benchmarks retain their own licenses.
