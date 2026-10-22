# Library guide

This page shows how to build a `ModelDiscoveryAgent` for your own system. Everything listed here can be imported from the top-level `mda` package unless a module path is given. [Concepts](concepts.md) explains the loop these pieces plug into.

- [A minimal agent](#a-minimal-agent)
- [Candidate models and priors](#candidate-models-and-priors)
- [Likelihoods](#likelihoods)
- [Evidence backends](#evidence-backends)
- [Experiment design](#experiment-design)
- [Proposers: making the pool M-open](#proposers-making-the-pool-m-open)
- [LLM clients, caching, and replay](#llm-clients-caching-and-replay)
- [Prediction](#prediction)
- [Configuration reference](#configuration-reference)
- [Diagnostics](#diagnostics)

## A minimal agent

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
    context="A scalar input-output system.",        # text shown to the proposer
    initial_data=[Datum(0.0, np.asarray(1.0))],      # D0
    initial_models=[linear],                         # seed library
    likelihood=DeterministicGaussianLikelihood(0.1),
    actions=[-2.0, -1.0, 1.0, 2.0],                  # finite design menu
    observation_covariance=0.1**2,                   # noise used to whiten VoI
    config=AgentConfig(smc=SMCConfig(parameter_particles=200)),
)

for _ in range(3):
    x = agent.design()
    y = np.asarray(1.0 + 2.0 * x)                    # run the real experiment here
    agent.update_state(x, y)

print(agent.get_model().structure)
forecast = agent.predict(3.0, flavour="bma")
```

[`mda.benchmarks.toy`](../src/mda/benchmarks/toy.py) is a complete runnable version with four seed laws (`mda-toy`).

## Candidate models and priors

A `CandidateModel` is one model structure:

| Field | Meaning |
|---|---|
| `name` | Display name |
| `prior` | Distribution over the flat parameter vector (any object with the `Prior` protocol) |
| `predict(theta, action)` | Mean observation for one experiment; may be a scalar, vector, or trajectory |
| `structure` | Canonical structure text; used as the deduplication key (falls back to `name`) |
| `predict_target(theta, action)` | Optional prediction in target space when the task target differs from the raw observation |
| `state_space` | `StateSpaceModel` callbacks for stochastic models (see [Likelihoods](#likelihoods)) |
| `metadata` | Free-form dictionary |

A `Datum(action, observation)` is one experiment. Actions can be any Python object: floats, arrays, or mappings such as `{"concentration": 0.1, "temperature": 300.0}`.

A prediction containing NaN or infinity raises `NonFinitePrediction`, which makes the law invalid at that parameter value.

### Priors

- `UniformPrior(names, lower, upper)`: independent uniform priors with finite bounds, as in the paper's symbolic domains.
- `mda.numpyro_backend.NumPyroPrior` and `compile_numpyro_model(model, action_features)`: a NumPyro program as a prior and predictor (install `.[numpyro]`). Continuous scalar and array sites are supported, with support transforms and Jacobians. Inference runs in sampler coordinates and MAP ranking in physical coordinates. Discrete, observed, and dimension-changing sites (for example a simplex) are rejected. The program must define `numpyro.deterministic("mu", ...)` with one value per feature row.

### Symbolic expressions

`compile_expression(expression, variable_names, parameter_names)` turns a formula string into a `predict` function:

```python
from mda import compile_expression
from mda.expressions import canonical_expression

expr = "a * x / (b + x)"
model = CandidateModel(
    name="michaelis-menten",
    structure=canonical_expression(expr),
    prior=UniformPrior(("a", "b"), np.array([0.0, 0.01]), np.array([10.0, 5.0])),
    predict=compile_expression(expr, ("x",), ("a", "b")),
)
```

Expressions are parsed and validated as an AST. Only approved functions with fixed argument counts are allowed; numeric constants are evaluated as floats; attributes, keywords, and comparisons are rejected. Overflow, division by zero, and invalid operations make a law invalid at that parameter value; underflow does not. With several input variables the action must be a mapping from variable name to value.

> AST validation restricts syntax. It is **not** a resource sandbox for arbitrary untrusted computation.

## Likelihoods

A likelihood is a callable `(dataset, model, theta, rng) → log-likelihood`.

| Class | Paper | Use for |
|---|---|---|
| `DeterministicGaussianLikelihood(observation_std)` | `loglik-det` | Deterministic simulators with Gaussian noise. `observation_std` may be a scalar, an array, or a function `(datum, prediction) → std` |
| `BootstrapParticleFilter(latent_particles, observation_std)` | `loglik-pf` | Stochastic state-space models |
| `BayesianSyntheticLikelihood(summary, simulations, observation_std, shrinkage=0, ridge=1e-6)` | `loglik-bsl` | Scoring trajectory summaries `summary(trace, action)` under a simulated Gaussian |

### Stochastic models

A stochastic model supplies `StateSpaceModel` callbacks on its `CandidateModel`:

```python
from mda import StateSpaceModel

StateSpaceModel(
    initial_state=lambda theta, action, particles, rng: ...,           # (particles, state_dim)
    transition=lambda states, theta, action, t, rng: ...,             # z[t-1] -> z[t]
    observation_mean=lambda states, theta, action, t: ...,            # mean of y[t]
)
```

Both the particle filter and BSL simulate through these callbacks with Gaussian emissions, so BSL also needs `state_space` and a fixed observation SD rather than an arbitrary simulator. A particle whose state diverges (NaN or infinite) gets zero weight.

The bootstrap PF gives an unbiased estimate of the **likelihood**. Its logarithm is not unbiased, so fractional likelihood tempering is only approximately pseudo-marginal, as the paper notes in its tempered-SMC appendix.

## Evidence backends

Pass `evidence=` to the agent to choose how each structure's parameter posterior and log-evidence are computed.

| Backend | Paper | Behavior |
|---|---|---|
| `TemperedSMCEvidence(SMCConfig(...))` | `marglik-smc` | Adaptive likelihood-tempered SMC on the full history. The default, configured by `AgentConfig.smc` |
| `RecursiveEvidence(initial=None, resample_fraction=0.5)` | `marglik-pf` | Fits a new structure with `initial` (tempered SMC by default), then reweights its particles by each new experiment's likelihood. Resamples when ESS falls below the fraction; no rejuvenation |
| `BICEvidence(residual_scale="profiled", ...)` | `marglik-bic` | Bounded multistart optimization (install `.[bic]`). Returns a point mass at the best optimum with `log_evidence = −BIC/2` |

```python
from mda import BICEvidence, RecursiveEvidence, SMCConfig, TemperedSMCEvidence

# Profiled BIC, sigma_n = max(0.03|y_n|, 0.01), fitted in log1p space (ChemBench).
bic = BICEvidence("profiled", relative_noise=0.03, noise_floor=0.01, transform=np.log1p)

# Tempered SMC on first sight of a structure, then one reweighting per new experiment.
recursive = RecursiveEvidence(TemperedSMCEvidence(SMCConfig(parameter_particles=128, max_temperatures=16)))

# ModelDiscoveryAgent(..., evidence=bic)
```

`BICEvidence.residual_scale` selects the variant:

- `"profiled"`: `N log(RSS/N) + k log N` with a fitted common residual scale (ChemBench's default).
- `"fixed"`: `RSS + k log N` with fixed whitening scales.
- `"likelihood"`: `−2 loglik(θ̂) + k log N` using the agent's likelihood, maximized under common random numbers so a noisy estimator gives a deterministic objective.

Other fields: `starts` (8), `max_evaluations` (2000), `transform`, and `observation_count`. Because BIC returns a point mass, `predict(flavour="bma")` is the BIC-weighted average of plug-in predictions.

`RecursiveEvidence` caches each structure by `model.key` and keeps its particles while the data only grow. A pruned structure that is proposed again resumes from its cache.

## Experiment design

### Value of information

`AgentConfig.design_flavour` selects the design objective:

| Flavour | Objective |
|---|---|
| `"model"` (default) | Model EIG: weighted disagreement between structures' mean predictions (v5 Eq. voi) |
| `"joint"` | Joint model/parameter EIG over all parameter particles (v5 Eq. voiFull) |
| `"task"` | v4's coordinatewise task VoI (an extension; needs `query_actions`) |
| `"task_full_observation"` | A dense full-observation Gaussian task VoI (an extension; needs `query_actions`) |

Both task flavours are v4 extensions; v5 leaves task-aware design to future work. See [Paper fidelity](paper-fidelity.md#task-voi-a-v4-extension) for their exact definitions and units. The functions `model_eig`, `joint_eig`, `task_value_of_information`, and `full_observation_task_value_of_information` are also exported directly.

### Finite menus

Pass the candidate experiments as `actions`. Used designs are excluded (by `action_key`) unless `repeat_designs=True`, or `repeat_when_exhausted=True` to allow repeats only after the menu is used up. Ties are broken by menu order. A design at which a weighted model cannot predict scores −∞ instead of aborting.

### Continuous design with CMA-ES

Install `.[design]`. Supply `actions=[]` and an optimizer:

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

Any optimizer is a callable `(score, rng) → action` that maximizes `score(action)` and returns an action it evaluated. CMA-ES works in a normalized unit box, decodes `log_axes` geometrically, returns its best finite evaluated design, and always spends its full evaluation budget (cma's tolerance stops are disabled, and any remaining stop restarts from a random point).

### Action keys

The agent identifies used designs by `action_key(action)`. The default is exact for arrays and mappings of arrays; `repr` would round to eight digits and elide long arrays. Supply your own `action_key` for other action types that `repr` does not distinguish.

### Experiment budget

Pass `experiment_budget=B` to include the remaining budget in the proposal context and to make `design()` raise after B updates.

### Experiment programs (beyond v5)

An experiment can be an *episode* of `T` scalar inputs in which each input is chosen after seeing the observations so far. `design_flavour="program"` makes `design()` return a **program**, `step(history, belief) -> next input`, instead of an action. This is an extension beyond v5; every default is unchanged, and nothing here runs unless you select it.

```python
from mda import AgentConfig, EpisodeSpec, LLMProgramProposer, ModelDiscoveryAgent, gaussian_step, run_episode
from mda.programs import belief_summary

episode = EpisodeSpec(steps=5, lower=0.0, upper=1.0)
# Each model needs sample_step; for a deterministic model with Gaussian noise:
#   CandidateModel(..., predict=predict, sample_step=gaussian_step(predict, std))
agent = ModelDiscoveryAgent(
    ..., actions=[], episode=episode,
    config=AgentConfig(design_flavour="program"),
    programs=seed_programs,                                    # like initial_models
    program_proposer=LLMProgramProposer(client, episode),      # optional
)
program = agent.design()
inputs, trajectory = run_episode(program, environment, episode, belief_summary(agent.belief))
agent.update_state(inputs, trajectory)                         # an ordinary datum
```

**Contract.**

- *Models.* `CandidateModel.sample_step(theta, inputs, observations, rng) -> y` draws the next observation given the episode so far (`inputs` includes the new one). `gaussian_step(predict, std)` builds it when `predict(theta, inputs)` returns the mean trajectory for any prefix of inputs. `predict` alone is a whole-action mean and cannot simulate a closed loop in general.
- *Environments.* `start_episode()` and `observe(u) -> y`. `run_episode` executes the program and returns the realised inputs and trajectory, which go to `update_state` unchanged as `Datum(action=inputs, observation=trajectory)`. Fitting, checks, expansion and `action_key` see an ordinary array action. `FixedProgram(inputs)` is an ordinary action as a program. `mda.benchmarks.reproduce.run(..., design="program")` does this for you.
- *Programs.* The source of one class with one method, `step(self, history, belief)`. `history` has `inputs`, `obs`, `t`, `horizon`, `obs_at(i)` and `last_obs()`. `belief` is `programs.belief_summary(agent.belief)`: model weights, parameter means and SDs per model, and pooled over models. Programs are stateless and deterministic, and inputs are clipped to the episode bounds. Episodes have a fixed length.

**Scoring.** Each candidate is scored by `trajectory_information`: a nested-Monte-Carlo estimate of the information, in bits, that the whole trajectory carries about the model index. Each imagined episode runs closed loop against one belief particle with that model's `sample_step`. The inner term is the agent's own `likelihood` of the datum `(inputs, trajectory)`. Every candidate in one decision shares one fresh seed (common random numbers). The outer particle is excluded from its own model's inner sum *by value*, so SMC duplicates do not inflate the estimate. The LLM never scores anything. A program that raises, returns a non-number, or breaks a sandbox limit is dropped from the pool.

**Program pool.** Like the model pool, it is capped (`program_cap`) and pruned to the best current scores at each decision. Fixed programs whose inputs were already run are dropped unless `repeat_designs`. The program proposer is called at a decision whenever the set of models differs from the one it last saw: at the first program design and after expansion or pruning. It is called at most `program_proposal_cap` times, for `programs_per_proposal` programs each.

**Proposers.** A `ProgramProposer` mirrors `ModelProposer`. `propose(context, dataset, current_models, disagreements, current_programs, count)` returns programs. `disagreements` holds each model's one-step predictive mean and SD on an input grid, and imagined episodes of the best current program under each model. `LLMProgramProposer(client, episode)` works with every client below, including caching and replay.

**Sandbox.** `mda.program_sandbox.compile_program(source)` admits LLM source in two layers. First, an AST whitelist: arithmetic, `if`, `for` over `range`, assignments, a few builtins, `math.*`, and the `history` attributes. It rejects imports, `while`, other functions, lambdas, comprehensions, `try`, underscore names, `self`, and any other attribute or call. Second, the source runs only in a separate `python -I -S` worker with CPU and memory limits, dangerous modules removed, and whitelisted builtins. The scorer sends the worker all imagined histories of one step at once, so scoring costs one round trip per episode step. The sandbox reduces risk; for untrusted proposals, still run in an isolated environment.

`mda.benchmarks.episodic` is a small test domain: a device whose two mechanisms differ only just above an unknown threshold, so a program that brackets the threshold within the episode beats a fixed protocol.

## Proposers: making the pool M-open

Without a proposer the pool is fixed. Pass `proposer=` to let the agent add structures. A proposer implements the `ModelProposer` protocol:

```python
class MyProposer:
    def propose(self, *, context, dataset, current_models, residuals, count):
        """Return up to `count` new CandidateModel objects."""
```

`residuals` holds one report per current model: its name, structure, median relative residual, and per-experiment absolute and signed residuals.

The proposer is called:

- once at construction, if `AgentConfig(initial_proposals=N)` is set; its N models are merged with the seed library without duplicates (as v5's ChemBench does), and
- after a failed predictive check, up to `model_search_rounds` times per update and `expansion_cap` times in total.

Proposals are labeled `explore` in the context: the loop never calls the proposer to tune a pool that already passes its check.

Concrete proposers:

| Proposer | Domain |
|---|---|
| `SymbolicRegressionProposer(client, variable_names=("x",))` | Algebraic scalar laws. Also receives signed residual–input correlations. Subclass it and override `system`, `build_prompt`, and `check_hypothesis` for a domain prompt and grammar |
| `mda.benchmarks.chemistry.ChemBenchProposer(client)` | ChemBench rate laws with v5's published prompt and restricted grammar |
| `NeuronBenchProposer(client)` | Hodgkin–Huxley channel hypotheses |
| `mda.benchmarks.glucose.GlucoseProposer` | Compartmental glucose ODEs |
| NumPyro proposer in `mda.benchmarks.boxing` | Generated NumPyro programs |

A proposed symbolic law is screened at a few fixed prior draws and kept if any draw evaluates on the observed experiments. Structures are stored in canonical `ast.unparse` form, so spacing or parenthesis variants of one law are duplicates.

### Generated code

`mda.numpyro_backend.compile_numpyro_source(source, action_features, allow_code_execution=False)` turns LLM-written NumPyro source into a model. It refuses by default. With `allow_code_execution=True` it **executes arbitrary Python with the process's privileges; it is not a security sandbox**. Use it only in an isolated environment with trusted or reviewed proposals. The AST symbolic backend does not execute code and needs no such flag.

## LLM clients, caching, and replay

| Client | Default model | Notes |
|---|---|---|
| `OpenAIJSONClient(model="gpt-5.6-luna", cache_dir=None, reasoning_effort="medium", replay_only=False)` | GPT-5.6 Luna | Responses API with structured output. Reads `OPENAI_API_KEY` |
| `mda.anthropic_client.AnthropicJSONClient(model="claude-opus-4-7", cache_dir=None, replay_only=False)` | Claude Opus 4.7 | Forced tool call for structured output. Reads `ANTHROPIC_API_KEY` |

Install `.[llm]`. Both clients stream requests, so any output budget is accepted.

**Reasoning models.** `gpt-5.6-luna` does not accept `temperature`. The OpenAI client omits it for GPT-5-family and o-series reasoning models (`gpt-5*`, `o1*`, `o3*`, `o4*`), requests medium reasoning effort (configurable with `reasoning_effort`), and sends no output-token cap, because hidden reasoning shares that budget and a cap sized for the answer can truncate it; the model's own limit applies. Cache keys still record the former 4096-token floor, so responses cached by earlier versions replay unchanged. The Anthropic client sends no temperature either: Opus 4.7 rejects sampling parameters.

**Caching.** With `cache_dir=".cache/mda-responses"`:

- `OpenAIJSONClient` caches completed responses that parse as JSON, keyed by the complete effective request, including the model, reasoning effort, and schema.
- `AnthropicJSONClient` caches only responses that stopped at the forced tool call (`stop_reason` `tool_use`). Its key omits temperature, which it does not send.

Pass `sample=` (any string) to either client to keep independent draws apart: clients with different labels never share an entry, even for identical requests. `mda-reproduce` labels each run with its domain, world, and seed, so seeds that start from the same D0 still get their own LLM proposals.

Within one client, a repeat of an identical request is a new draw, as it would be without a cache: the *n*-th repeat is keyed with the label `<sample>/repeat=<n>`. Replay serves the draws in the same order.

The cache key excludes credentials, but cached response text may still contain sensitive data. Keep the directory private and out of version control (it is in `.gitignore`). The default is no cache.

**Replay.** `replay_only=True` serves only cached responses and fails on a miss before loading the SDK or checking credentials. This replays calls you captured; it does not reproduce the paper's unreleased responses.

### Coding-agent clients

`mda.agent_clients` runs each request as one tool-free turn of a coding agent, through its official Python SDK. Install `.[agents]`. Both clients have the same `ask_json` interface, `cache_dir`, `sample`, and `replay_only` behaviour as the API clients.

| Client | Default model | Notes |
|---|---|---|
| `ClaudeCodeJSONClient(model="claude-opus-4-7", effort=None)` | Claude Opus 4.7 | `claude-agent-sdk` with JSON-schema output. Uses `ANTHROPIC_API_KEY` if set, otherwise the Claude subscription login or `CLAUDE_CODE_OAUTH_TOKEN`. `effort=None` keeps Claude Code's default |
| `CodexJSONClient(model="gpt-5.6-luna", effort="medium")` | GPT-5.6 Luna | `openai-codex` with `output_schema`. Uses Codex's login (`codex login`, ChatGPT or API key) |

Each call runs in an empty temporary directory with no tools, MCP servers, user settings, web search, or saved session. The system prompt replaces the agent's own. Neither agent exposes temperature or an output-token cap, so both are ignored.

The responses do not count as the paper's API configuration. Cache keys include the agent's name, so they never replay as API responses or as each other's. Claude Code returns its answer through an extra tool turn, so it costs more per call than the direct Anthropic client.

By default Codex adds about 8,000 tokens to every call: a JavaScript `exec` tool that wraps shell and patch tools, sub-agent tools, a skills list, sandbox and environment text (working directory, date, timezone), and low verbosity. The shell tool could read any file on disk, including a benchmark package's hidden parameters. `CodexJSONClient` switches all of this off at launch, with config overrides and a copy of Codex's model catalog whose entry for the model has no tool mode, patch tool, tool search, or sub-agents, and medium verbosity. A call then sends only the system prompt (as a developer message), the user prompt, and the output schema: a trivial prompt costs 58 input tokens instead of 7,962.

Codex sets up its state directory (`CODEX_HOME`, default `~/.codex`) on its first run. Make one Codex call before you start parallel runs on a fresh home, because concurrent first runs can fail to set it up.

## Prediction

- `agent.predict(action, flavour="map")`: the current-data MAP structure at its MAP parameters.
- `agent.predict(action, flavour="bma")`: the Bayesian model average over all structures and parameter particles.
- `agent.get_predictor(flavour)`: the same as a function of the action, for evaluators.
- `agent.get_model()`: the MAP structure. It is always selected on the current data, never by held-out truth.

## Configuration reference

### `AgentConfig`

| Field | Default | Meaning |
|---|---|---|
| `model_cap` | 16 | Maximum pool size (top-N pruning) |
| `new_models_per_expansion` | 4 | Structures requested per proposer call |
| `model_search_rounds` | 2 | Expansion rounds allowed per update |
| `expansion_cap` | 4 | Total proposer calls allowed (N_e^max) |
| `error_threshold` | 0.05 | Predictive-check threshold on the median relative residual |
| `occam_penalty` | 0.0 | λ in the model prior `p(m) ∝ exp(−λk)`; 0 is uniform |
| `predictive_floor` | 1e-6 | Denominator floor for relative residuals |
| `backtest` | True | Include earlier data in the predictive check |
| `design_flavour` | `"model"` | `model`, `joint`, `task`, `task_full_observation`, or `program` (beyond v5) |
| `repeat_designs` | False | Allow repeated designs |
| `repeat_when_exhausted` | False | Allow repeats only once the menu is used up |
| `smc` | `SMCConfig()` | Settings for the default tempered-SMC evidence |
| `minimum_model_cap` | None | Enables `adapt-and-prune` shrinking to this cap |
| `shrink_confidence` | 0.9 | MAP weight required before shrinking |
| `initial_proposals` | 0 | Size of the LLM batch at initialization |
| `expansion_gate` | `"prequential"` | `"cached"` gates the first expansion on the previous fit's in-sample residuals, as v5's executed ChemBench agent does |
| `program_cap` | 8 | Maximum program-pool size (program design only) |
| `programs_per_proposal` | 2 | Programs requested per program-proposer call |
| `program_proposal_cap` | 4 | Total program-proposer calls allowed |
| `program_draws` | 128 | Imagined episodes per program score |

Shrinking requires both a well-fit MAP model (median relative residual at most `error_threshold`) and a MAP weight above `shrink_confidence`; otherwise the cap is `model_cap`. It never runs at initialization. A failed check restores the larger cap, but evicted models are not resurrected.

### `SMCConfig`

| Field | Default | Meaning |
|---|---|---|
| `parameter_particles` | 100 | Particles per structure |
| `rejuvenation_steps` | 3 | Random-walk Metropolis moves per rung |
| `target_ess_fraction` | 0.6 | Adaptive tempering ESS target |
| `max_temperatures` | 80 | Rung limit; the last rung is forced to β = 1 |
| `proposal_scale` | 0.5 | Random-walk scale relative to the particle spread |
| `minimum_proposal_scale` | 1e-6 | Lower bound on the random-walk scale |

These defaults are v4 Table 2's ChemBench settings, since v5 gives no values. Out-of-bounds random-walk proposals are rejected, not clipped.

### Other agent arguments

| Argument | Meaning |
|---|---|
| `target(observation, action)` | Maps raw observations to the task target used by checks and predictions (identity by default) |
| `query_actions` | Held-out actions for task VoI |
| `target_covariance`, `task_noise_covariance` | Task-VoI loss whitening and target-space noise |
| `check(belief, fresh, history)` | Replaces the default predictive check; compared against `error_threshold` |
| `seed` | Seeds all agent randomness |
| `episode`, `programs`, `program_proposer` | Program design: the episode shape, the seed program library, and the optional program proposer |

## Diagnostics

After each fit, `agent.belief` is a `BeliefState`:

- `fits`, `model_weights`, `map_fit`, `map_index`: the pool and its posterior.
- `log_mixture_evidence`: computed before pruning over the deduplicated pool. It is a diagnostic only.
- `discarded_mass`: weight lost to pruning and support truncation.

Each structure's `fit.parameters` is a `ParameterPosterior` with `particles`, `weights`, `log_evidence`, `temperatures`, `acceptance_rate`, `ess_before_resampling`, and `last_rung_forced_below_target`. Reaching the rung limit forces the full likelihood (β = 1), possibly at the cost of a low ESS. Increase the rung or particle budget if these diagnostics show collapse.

`agent.last_design_scores` holds the VoI scores from the last `design()` call and `agent.last_predictive_error` the last check value.
