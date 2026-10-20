# Reimplementation of **Model Discovery Agent: LLM-assisted Bayesian experiment design for data-efficient discovery of mechanistic world models** ([arXiv:2608.09696v5](https://arxiv.org/abs/2608.09696v5), 27 Sep 2026).

v5 restructured the paper: its main domains are now ChemBench, the new GlucoseBench, and the stochastic neuron benchmark (NeuronBench; its LaTeX macro is `\HHbench`); BoxingGym moved to an appendix, and the ForceBench/DiscoverPhysics domain was dropped. [What changed in v5](#what-changed-in-v5) lists how this repository follows it.

## What is implemented

- M-open model discovery with prequential predictive checks and residual-guided structure expansion, following v5's Algorithm 2 ordering.
- Three per-structure evidence backends: full-history adaptive likelihood-tempered SMC (the default), recursive parameter-weight/evidence updates (`marglik-pf`), and bounded multistart optimization with BIC (`marglik-bic`, profiled or fixed residual scale, or a direct likelihood). Model weights use an optional parameter-count prior and deterministic top-N pruning.
- Exact Gaussian likelihoods for deterministic simulators, a bootstrap particle-filter likelihood for user-supplied stochastic state-space models, and a Bayesian synthetic likelihood of trajectory summaries.
- Model EIG and joint model/parameter EIG surrogates. v4's coordinatewise task VoI and a full-observation Gaussian alternative are kept as extensions; v5 leaves task-aware design to future work.
- Finite-menu design or bounded CMA-ES optimization, including logarithmic axes.
- MAP and Bayesian-model-averaged prediction.
- AST-validated symbolic models and a structured OpenAI Responses API proposer, with optional persistent response caching. AST validation restricts syntax (approved functions with fixed argument counts, numeric constants evaluated as floats, no attributes, keywords or comparisons); it is not a resource sandbox for arbitrary untrusted computation. Overflow, division by zero and invalid operations make a law invalid at that parameter value; underflow does not.
- A v5 ChemBench runner: the nine canonical seed laws plus one 12-law LLM batch, profiled BIC in log-rate space, adaptive 6/24 pool caps, 48-evaluation CMA-ES, the benchmark pass flag, and an SMC control arm.
- A v5 GlucoseBench runner on the public `glucosebench` package: LLM-editable reduced compartmental ODEs, an AR(1)-sensor working likelihood fitted by multistart L-BFGS-B with BIC weights, a whitened-residual χ² expansion gate, noise-normalized disagreement design over the 48-template menu, and EKF prefix conditioning with a BIC-weighted mixture forecast.
- A Fox–Lu stochastic neuron candidate with latent-state PF callbacks, recursive SMC evidence, v5's brief/weak-step seed set, and its fixed-scale six-feature metric. The v4 deterministic NeuronBench adapter is kept as legacy.
- NumPyro continuous scalar/array priors with support transforms and Jacobians; sampler-coordinate inference and physical-coordinate MAP ranking. Discrete, observed, and dimension-changing (e.g. simplex) sites are explicitly rejected.
- Cached OpenAI (GPT-5.6 Luna, medium reasoning effort) and Anthropic (Opus 4.7) proposals with credential-free offline replay for both, scalar GLM likelihoods, an ICL forecaster, and finite-grid constant-refit recovery tests.

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
.venv/bin/mda-reproduce physics --world gravity --smoke --budget 1 --output /tmp/physics.json  # v4 only
.venv/bin/mda-reproduce chemistry --world c0_michaelis_menten --smoke --budget 1 --output /tmp/chemistry.json
.venv/bin/mda-reproduce neuron --world h_sag --stochastic --smoke --budget 1 --output /tmp/neuron.json
.venv/bin/mda-reproduce glucose --world 'adult#001' --smoke --budget 1 --output /tmp/glucose.json
```

The driver requires exactly one of `--smoke`, `--replay`, or `--live`. Smoke uses small numerical budgets and generic, non-truth baselines (for ChemBench, the nine canonical laws); it makes no LLM calls. Neuron smoke checks feature forecasts but omits expensive held-out 200-rollout scoring. Replay reads previously captured responses and fails on cache misses without contacting a provider. Live explicitly authorizes paid calls. As in v5, the default provider is OpenAI `gpt-5.6-luna` at medium reasoning effort, except Boxing, whose appendix results use `claude-opus-4-7` with reasoning off. v5 states no temperature, and none is sent: Opus 4.7 rejects sampling parameters, and GPT-5.6 Luna has none. Requests are streamed, so any output budget is accepted, and only a response that stopped at the forced tool call is cached. `--provider` and `--model` override these. No model availability or live score claim is made here.

Normal runs use the per-domain settings of v5's main-settings table and domain appendices (Boxing keeps v4's Table 2 column). Specify `--budget` explicitly (the conservative CLI default is one experiment, not a paper sweep; v5 uses 3+22 ChemBench, 2+4 GlucoseBench, and 2+8 neuron experiments). The released neuron pool has only seven unused protocols after D0, so the stochastic runner repeats protocols once they are exhausted. A finite-menu run whose budget exceeds the unused designs fails before it starts. `--difficulty` defaults to ChemBench's v5 `medium`. Flags that would have no effect in the chosen configuration (`--difficulty` outside ChemBench, `--cuts`/`--replicate` outside GlucoseBench, `--allow-code-execution` outside Boxing, `--task-noise-variance` without `--design task`, and `--model`/`--provider`/`--cache-dir` under `--smoke`) are rejected rather than recorded in the result as if they applied. A run that fails part-way still writes every completed snapshot, with an `error` field, and exits non-zero; a failure of the `--icl` side baseline is recorded on that snapshot (`icl_score: null`, `icl_error`) without ending the run. `--design` selects model, joint, task, or random design; random design samples a finite menu, or ChemBench's log-mapped box uniformly, from a generator that depends only on the seed, so random arms with different inference share their design history, as v5's ChemBench controls do. `--no-m-open` disables expansion after the initial pool (for GlucoseBench, including the initialization gate); for ChemBench it also skips LLM initialization, giving v5's fixed nine-law control. `--inference smc` selects ChemBench's SMC control instead of the profiled-BIC primary arm. Boxing and GlucoseBench default to BMA prediction; the other domains use MAP. Neuron task design requires an explicit `--task-noise-variance` (one or six positive feature-space variances), since no paper version releases that calibration. It must not silently use raw voltage noise for six different features.

`--agent apprentice` implements the Boxing Apprentice comparison: one evolving model, LLM-selected experiments, and a residual-conditioned refinement on every round, fitted by the same SMC. `--icl` adds the data-only forecasting baseline after every round (replay/live only); this is v5's "ICL (+MDA design)". For the neuron it sees the six summary features with a generic prompt, not v5's 20 raw voltage samples and HH prompt. `--recovery` adds ChemBench or central two-particle ForceBench constant-refit recovery scoring (install `.[recovery]`). These are v4 diagnostics: v5 reports the ChemBench pass rate (`metrics.passed`) and moved structural recovery out of the paper. Recovery parameters never replace the agent's fitted parameters. For NeuronBench, the same flag scores the coarse inward/outward, depolarisation/hyperpolarisation, transient/persistent channel class (or slow Na), not exact kinetics. The evaluator uses the proposed channel properties, not its name; zero conductance or reversal exactly at the local -65 mV reference yields "unknown". Calcium and z-rebound intentionally share a class. The physics driver also accepts `circle`, `ether`, and `hubble`. These extensions use documented local launch menus because the exact v4 menus are unreleased.

GlucoseBench runs one patient (`--world 'adult#001'`, any of the 30 profiles) for at most four selected experiments (v5 uses all four). Following the package's protocol, the agent's forecaster is snapshotted at 2, 3, 4 and 6 training episodes, acquisition is then frozen, and each snapshot is scored on the 12 sealed queries at the requested `--cuts` (default 45, the headline). Scores are the package's per-channel training-variance nMSE, averaged over queries; failed cells are counted, not dropped, whereas v5 scores only the 273 of 360 patient–query pairs valid for every method, budget and cut. The package's persistence baseline, which v5 does not report, is always scored. `--icl` adds ICL (+MDA design) with v5's census forecast prompt at every cut. v5's separate multi-cut prompt (an explicit empty prefix at cut 0, up to two schema retries) is not implemented, and v5 does not say which prompt produced its cut-45 headline. `--prediction map` uses only the MAP model instead of the BIC-weighted mixture. Smoke mode fits with one start and 15 iterations; it simulates the 12 test episodes, so it takes about a minute.

IRT, location, and survival generate NumPyro Python. Non-smoke runs require `--allow-code-execution`: **this executes arbitrary generated Python with process privileges, not in a security sandbox**. Use only a suitably isolated environment and trusted/reviewed proposals. The AST symbolic backend does not require this flag.

JSON results contain the request, package versions, effective numerical settings, per-round actions, observations, current model/parameters, model weights, log-evidences, forced-tempering diagnostics, scores (with ChemBench nMSLE, RMSLE, and pass flag), and a separately labeled best-so-far curve. Evaluation does not select models or feed query labels into training. Output files are never intentionally overwritten.

`mda.benchmarks.icl.make_icl_predictor` provides a model-free baseline using only observed action-target pairs and query inputs. `mda.benchmarks.recovery.evaluate_recovery` refits a trusted candidate's constants on a separate noise-free grid and tests held-out RMSLE. The benchmark evaluators first compile the symbolic law using `canonical_symbolic_candidate`: a whitelist AST translation into SymPy, followed by simplification and numerical compilation, without parsing arbitrary Python through `sympify`. This is a numerical approximation, not proof of symbolic equality. ForceBench measures the true radial force through clean public release-at-rest experiments, never private force fields. Its local grid has 128 interleaved log-spaced radii in [1.5, 10], with a .01 time-unit finite difference; it uses signed `log1p` and tolerance .05. This tests the radial profile at unit coupling and time zero, not the complete charge/time dependence. Chemistry uses separate 128-point fit and holdout grids, the clean primary law as truth, and ordinary `log1p` with tolerance .02; a refit start that leaves the law undefined is skipped, and a law undefined at every start is recorded as not recovered. Neither grid is claimed to be the authors' unreleased evaluation grid.

`gpt-5.6-luna` does not accept `temperature`; the client omits it for GPT-5-family and o-series reasoning models (`gpt-5*`, `o1*`, `o3*`, `o4*`), requests medium reasoning effort (configurable with `reasoning_effort`), and reserves at least 4096 output tokens, which also cover reasoning. The ICL request budgets 32 output tokens per predicted value with an 800-token floor.

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

Pass a `ModelProposer` implementation to make the pool M-open. See `SymbolicRegressionProposer` and `NeuronBenchProposer` for concrete structured LLM boundaries. With `AgentConfig(initial_proposals=N)` a seed library is combined with one N-model LLM batch, as v5's ChemBench does. A stochastic model supplies `StateSpaceModel` callbacks and uses `BootstrapParticleFilter` (or `BayesianSyntheticLikelihood`) as its likelihood.

Pass `evidence=` to choose the evidence backend:

```python
from mda import BICEvidence, RecursiveEvidence, TemperedSMCEvidence

# Profiled BIC, sigma_n = max(0.03|y_n|, 0.01), fitted in log1p space (ChemBench).
bic = BICEvidence("profiled", relative_noise=0.03, noise_floor=0.01, transform=np.log1p)
# Tempered SMC on first sight of a structure, then one reweighting per new experiment.
recursive = RecursiveEvidence(TemperedSMCEvidence(SMCConfig(parameter_particles=128, max_temperatures=16)))
# ModelDiscoveryAgent(..., evidence=bic)
```

BIC returns a point mass at the best optimum, so `predict(flavour="bma")` is the BIC-weighted average of plug-in predictions. Install `.[bic]` (SciPy) for it; it is included in `.[dev]`.

### Continuous design and proposal replay

Install `.[design]` for CMA-ES (included in `.[dev]`). Supply `actions=[]` and an optimizer to the agent, for example:

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

An optimizer receives `(score, rng)`, maximizes `score(action)`, and must return an evaluated action. The agent excludes used `action_key`s unless `repeat_designs=True`. CMA-ES returns its best finite evaluated design and honors its evaluation limit. The default `action_key` is exact for arrays and mappings of arrays (`repr` would round to eight digits and elide long arrays); supply your own for other action types that `repr` does not distinguish.

`OpenAIJSONClient(cache_dir=".cache/mda-responses")` caches completed responses that parse as JSON, keyed by the complete effective request, including the model, reasoning effort, and schema. `AnthropicJSONClient` caches only responses that stopped at the forced tool call (`stop_reason` `tool_use`); its key omits temperature, which it does not send. The key excludes credentials; cached response text may still contain sensitive data. Keep the directory private and out of version control. The default is no cache. This enables replay of captured calls, not reproduction of the paper's unreleased responses. `replay_only=True`, on either `OpenAIJSONClient` or `AnthropicJSONClient`, fails on a missing cached response before SDK loading or credential checks.

Pass `experiment_budget=B` to include the remaining budget in proposal context and prevent `design()` after B updates. Proposals are labeled `explore`: the core loop invokes them only at initialization or after a failed adequacy check, not to tune an already adequate pool. Symbolic proposals receive signed residual/input correlations; both proposers receive the entire model pool, priors, and per-example residual reports.

## Paper fidelity map

v5 names its algorithms; the numbers are its Appendix A order.

| Paper (v5) | Implementation |
|---|---|
| Algorithm 1, agent/environment interface | `run` in `mda.benchmarks.reproduce` |
| Algorithm 2, MDA agent (`init`, `design`, `update-state`, `get-model`, `get-predictor`) | `mda.agent.ModelDiscoveryAgent` |
| `loglik-det`, `loglik-pf`, `loglik-bsl` | `mda.likelihoods.DeterministicGaussianLikelihood`, `BootstrapParticleFilter`, `BayesianSyntheticLikelihood` |
| `marglik-smc`, `marglik-pf`, `marglik-bic` | `mda.evidence.TemperedSMCEvidence` (`mda.smc.parameter_posterior`), `RecursiveEvidence`, `BICEvidence` |
| `model-posterior` | `mda.posterior.fit_model_posterior` |
| `adapt-and-prune` | `ModelDiscoveryAgent._adapt_pool` (`minimum_model_cap`, `shrink_confidence`) |
| `init-hyp-space`, `expand-hyp-space` | `mda.proposer` plus the proposer implementations (`initial_proposals`) |
| `predictive-check` | `mda.checks` |
| `VoI` (Eq. voi model, Eq. voiFull joint) | `mda.acquisition` |

Generic defaults use v5's default evidence method, tempered SMC, with v4 Table 2's ChemBench settings (100 particles, three rejuvenation moves, ESS target 0.6, 80 rungs), since v5 gives no values. They also use v5's default uniform model prior (`occam_penalty=0`), with v4's 16-model pool, four proposals per expansion, two search rounds, and 0.05 error threshold. Domain runners set their own values, as the paper does: ChemBench uses BIC, a 24-model cap, six proposals per expansion and no parameter-count prior; the stochastic neuron runner uses recursive SMC with `p(m) ~ exp(-2k)`.

Optional `adapt-and-prune` shrinking is enabled with `minimum_model_cap` in `AgentConfig`. After an update it requires both a well-fit MAP model (median relative residual at most `error_threshold`) and a MAP weight above `shrink_confidence` (default 0.9, v5's ChemBench value); otherwise the cap is `model_cap`. It never runs at initialization. A failed check or reduced confidence restores the larger cap; evicted models are not automatically resurrected. Two details differ from Algorithm 2: a failed check restores the cap before that update's expansion rounds rather than at the end, and shrinking renormalizes the retained weights instead of refitting (the same for deterministic evidence). Shrinking is disabled by default, matching the base Algorithm 2.

`expansion_gate="cached"` reproduces v5's executed ChemBench agent, which gates its first expansion on the in-sample residuals of the preceding fit rather than on the prequential check; later rounds recheck the refitted pool in both modes.

Model revision follows v5's narrower account. The LLM is used only by the proposer: once at initialization (`initial_proposals`) and in `expand-hyp-space`. Proposals are conditioned on per-model residual reports, which v5 treats as optional context "inspired by" SMC-S. v5 calls top-N pruning deterministic hypothesis search rather than a posterior-invariant resampling kernel, so model weights are a posterior over the proposed pool only and assign nothing to models never proposed. Every proposer call counts toward `expansion_cap` (N_e^max), even one that returns only invalid or duplicate models. A proposed law is screened at a few fixed prior draws and kept if any draw evaluates on the observed experiments; symbolic structures are stored in canonical `ast.unparse` form, so spacing or parenthesis variants of one law are duplicates. A model whose weight underflows to zero stays in the pool and can regain weight; only structures whose fit fails entirely leave it. A design that the current best model cannot evaluate at all (a domain error at the chosen input) counts as a failed predictive check rather than aborting after the experiment has run, and if the refit on the new datum fails, the agent is left exactly as it was. The domain runners differ as in v5: ChemBench adds up to 12 LLM laws at initialization and requests six per expansion; GlucoseBench adds at most two per call and shows the proposer the fitted equations and their predictions; the stochastic neuron runner builds its archive once and then freezes it.

`BeliefState.log_mixture_evidence` is computed before pruning over the deduplicated input pool, with its parameter-count model prior normalized over that pool. v5 stores this normalizer after pruning; here it is a diagnostic only. `discarded_mass` includes support truncation. Each parameter posterior records `ess_before_resampling` and `last_rung_forced_below_target`: reaching the rung limit forces the full likelihood (`beta=1`), possibly at the cost of low ESS. Increase the rung/particle budget if these diagnostics show collapse.

### Task-VoI (a v4 extension) has explicit target-space units

v5 keeps only the general decision-theoretic task VoI (Eq. taskvoigeneral) and leaves it to future work. The two task flavours below implement v4's Eq. 42–45 and are not part of the v5 method.

`design_flavour="task"` uses v4 Eq. 43's target predictions at **both** the proposed and query actions, with each target coordinate scored independently. It averages across query actions and sums coordinate contributions, whitening the loss by the target-space noise variances. `task_noise_covariance` supplies those variances to the agent; when omitted it falls back to `observation_covariance`. Use a separate target-space value for features derived from raw traces. `target_covariance` overrides the loss whitening (set it to 1 for unwhitened squared loss). Both must be scalar or diagonal for this flavour.

The old dense matrix conditional-variance calculation remains available as `design_flavour="task_full_observation"` or `full_observation_task_value_of_information`. It uses raw observations at the proposed action and all cross-coordinate information; optional `target_covariance` whitens its loss. This is a coherent Gaussian alternative inspired by v4 Eq. 42, but it is **not** the coordinatewise implementation described after v4 Eq. 45. Neither objective is exact information gain for a general non-Gaussian mixture.

### Pseudocode corrections and implementation choices

This repository corrected six inconsistencies in v4's pseudocode. v5 now states each of them itself: the pre-observation check runs before the append; top-N pruning renormalizes the retained weights; out-of-bounds random-walk proposals are rejected (v5 notes that the authors' NumPy fitter clipped instead, which can bias its legacy results); the trace surrogate is not exact mutual information; checks after expansion are in-sample; and `get_model()` returns the current-data MAP rather than a maximum over rounds. Held-out truth is never used to select the agent's model; best-so-far curves are a separate reporting convention.

Remaining choices that v5 leaves open:

- After appending a datum the pool is refitted, so proposals see residuals of the refitted pool (v5 Algorithm 2). The re-judge after an expansion fits only the new structures and reuses the others' fits on the identical dataset, rather than rerunning their SMC.
- Profiled BIC needs more observations than parameters, and v5 does not say how it handles an exact fit. When a model interpolates the data (e.g. three-parameter laws on ChemBench's three-point D0), its RSS is floored at the smallest positive float, so it takes essentially all the weight until more data arrive. The other models stay in the pool with zero weight and recover once the fit is no longer exact.
- v5 does not give the number of BIC multistarts (8 here) or its CMA-ES population size and initial step (cma's defaults here). It states a proposal-scale floor of 0.5 × max(std, 1e-3) only for its legacy NumPy fitter; this one uses a relative floor.
- CMA-ES always spends its full evaluation budget: cma's tolerance stops are disabled, so a nearly flat score (a concentrated posterior) cannot end the search after one generation, and any remaining stop restarts from a random point.
- Finite-menu design breaks ties by menu order (GlucoseBench breaks them randomly, as v5 specifies there). Used designs are excluded unless `repeat_designs` or `repeat_when_exhausted` is set. A design at which a weighted model cannot predict scores −∞ rather than aborting.
- `BayesianSyntheticLikelihood` simulates through the same Gaussian-emission state-space callbacks as the PF, so it needs `state_space` and a fixed observation SD, not an arbitrary simulator. Its defaults of no shrinkage and ridge 1e-6 are local choices.
- `RecursiveEvidence` keeps each structure's particles while the data only grow, including after the archive changes; v5's HH implementation refits when the archive changes. A pruned structure that is proposed again resumes from its cache.
- A PF particle whose state diverges (NaN or infinite) gets zero weight rather than voiding the likelihood estimate.

The bootstrap PF gives an unbiased **likelihood** estimate. Its logarithm is not unbiased, and fractional likelihood tempering is only approximately pseudo-marginal, as the paper notes in its tempered-SMC appendix.

## GlucoseBench

The `glucose` extra pins the public [glucosebench](https://github.com/murphyk/glucosebench) package at `6bb0ff6`. It is a small simglucose adaptation with 30 virtual patients, CGM every 5 minutes, and noisy plasma-insulin assays every 15 minutes over six hours. It owns the simulator, the two passive D0 episodes, the 48-template meal/bolus menu, the 12 sealed test interventions, the prefix payloads, and the metric. Only that package's learner-facing payloads reach the agent; it never sees patient identity, parameters, or hidden states.

`mda.benchmarks.glucose` implements the learner side of v5's Appendix C:

- **Models.** Each candidate is a JSON specification of 1–6 states, one right-hand side per state, initial states, and CGM and insulin outputs, with up to 12 positive-bounded parameters, written in the paper's restricted expression grammar. Models restart at minute 45 from their initial states (holding them before then) and are integrated with one-minute RK4 steps under the delivered meal (g/min) and bolus (U/min above basal) inputs. `gb` and `ib` are the CGM and insulin means over the training prefixes.
- **Fitting and weights.** The working likelihood has AR(1) CGM residuals, with the innovation variance profiled out, and independent insulin assays with SD 10 pmol/L. It is fitted from minute 45 by multistart L-BFGS-B in log-parameter space with batched finite-difference gradients. BIC counts the ODE parameters plus the AR(1) correlation and scale, with N the number of observed targets, and weights are proportional to exp(−BIC/2) with no extra prior.
- **Library.** The pool starts from four seed models: one-stage absorption with linear or glucose-dependent insulin action, two-stage meal absorption (Eq. glucose_reduced_ode), and a delayed CGM readout. The LLM, using v5's published proposer prompt and schema, adds at most two models per call, with genealogy checks and deduplication up to whitespace and redundant parentheses; a malformed proposal, or one whose equations raise when simulated once at its declared parameter values, is dropped without discarding the other. A rollout that only diverges is kept, and a model whose expressions raise during fitting gets zero evidence and leaves the pool. Missing or non-finite scheduled CGM readings or insulin assays raise an error naming the episode and minute. The archive is capped at 14 with one search round per update. The payload also gives each model's fitted AR(1) ρ and σ, which the prompt does not list, and the expression parser also accepts `%`, `pi` and `e`, which the prompt does not offer.
- **Expansion gate.** Expansion happens when even the best archived model fails a whitened-residual check on the new episode before refitting: the minimum over models of A_m exceeds χ²₀.₉₉₅(86)/86 ≈ 1.436. At initialization the gate uses each model's worst D0 episode. This statistic is printed only in the paper's source, not its rendered text. Residuals are anchored at the pooled training-prefix means; that source anchors each checked episode at its own 45-minute prefix.
- **Design.** Each unused template is scored by the BIC-weighted between-model variance of point predictions, normalized by the model-weighted stationary AR(1) variance for CGM and the assay variance for insulin, averaged over observation times, with ties broken randomly.
- **Forecasting.** An EKF over the ODE state and the AR(1) sensor error conditions on the query prefix, then propagates without further updates. The forecast is the BIC-weighted mixture of component means at point parameters; a component whose forecast fails, including one whose EKF state becomes non-finite or reaches 1e7 in magnitude (the divergence rule the fits use), drops out and the rest are renormalized. Forecasts are means only: v5's 90% mixture-quantile intervals are not computed.

Local choices, where v5 is silent: the equations and bounds of the three seed models whose equations it does not print; three random starts plus the declared and warm-start values; AR(1) correlation in [0, 0.99]; designs scored on the nominal delivery schedule (meals at 5 g/min, bolus over 5 minutes) without the pump's quantization, which fits do include; the EKF's initial state SD (5% of magnitude, floor 1) and finite-difference Jacobians; and holding the restart state through the prefix, so that EKF updates can use every prefix observation. v5 mentions an EKF positivity safeguard without specifying it; none is applied here beyond the divergence guard above.

## NeuronBench and NeuroBench checks

The `neuron` extra pins the authors' NeuronBench benchmark at [`c354622`](https://github.com/murphyk/neuronbench/commit/c354622458c460b419cab821d482c879f0578377), still its latest commit. v5 points to this repository for its NeuronBench (LaTeX macro `\HHbench`): the six worlds and their kinetics are unchanged, but the pinned commit lacks v5's 90-action, 20-sample, 24-query harness. The adapter never reads its private truth mechanism. It uses the released simulator for candidate rollouts and gives the proposer the feature vector `[n_test, n_pre, run_down, adaptation, V_min, V_end]`.

The stochastic environment follows v5 where the pinned package allows it. D0 is the brief 12 µA/40 ms and weak 5 µA/120 ms steps. `evaluate_predictor` computes v5's fixed-scale feature MSE, Eq. hhFixedScaleMSE with s = (3, 3, 3, 3, 12, 8), against 200-rollout expected features (v5 gives no rollout count). The runner uses a 128-particle PF and 128-particle tempered SMC with at most 16 adaptive initialization rungs (the last forced to β = 1), then recursive updates without rejuvenation. The initialization's ESS target 0.5 and three rejuvenation moves are v4's HH settings; v5 gives neither. The archive is the plain Na/K/leak cell plus three LLM proposals, and the model prior is `exp(-2k)`. As in v5's headline cohort, later model births are disabled, so the archive stays fixed after initialization. `--online-expansion` (`online_expansion=True`) re-enables v4's residual-triggered expansion, which is not v5's stochastic rank-test gate. That gate appears only in the paper's source, in an audit where 1 of 36 runs triggered it and the three added models ended with 0.067% of the weight. v5 selects eight experiments from 90; with seven unused released protocols, the eighth repeats one.

The channel parameterization is not v5's. In v5's headline priors the LLM gives bounds for g, V½ and the activation τ (or only the slow-Na τ), and the backend adds gL and, for transient currents, τ_h ~ U(20,300) ms, fixing slopes at 8/−6 mV and the inactivation V½ 22 mV below activation. Here transient channels infer one τ (four parameters rather than five, which also changes the `exp(-2k)` prior), the LLM fixes the other gating constants, and slow-Na uses U(200,600) ms rather than LLM bounds. The proposer and ICL context include the package's `text_prior`; for `textbook_M` it names the M current, whereas v5 gives the agent a prior that does not name the novel current.

The deterministic `NeuronBenchEnvironment`, with v4's Table 14 noise model and brief/long-step D0, is v4 legacy: v5 has no deterministic benchmark. It is scored with the same fixed-scale metric as the stochastic environment.

`NeuronHypothesis.tau_gate` selects activation (Ih/M) or inactivation (Ca/D), with `activation_tau_ms` fixed in the latter case. Anonymous channel proposals remain supported; their bounds and fixed gating forms come from the proposer, not a hidden truth-based archetype lookup. v4's Table 14 and H.4.1 disagreed about which gate's tau is inferred; v5 settles it (see above), and this API keeps the choice explicit.

Slow-Na candidates infer `gL ~ Uniform(0.2,0.45)` and `tau_s_ms ~ Uniform(200,600)`. A local deterministic rollout extends the pinned simulator's equations without changing the dependency: the sampled tau replaces the depolarized branch only, with the recovery branch fixed at 1500 ms. At 380 ms it reproduces the released rollout. This interpretation of Table 14's single tau is explicit rather than claiming the paper specifies both branches.

Voltage traces remain in raw mV. Interpreting Table 14's “1 unit = 3 mV” as `V/3`, `observation_std` returns `max(0.12*abs(V), 3)` mV. The table mixes feature and trace terminology, so this is a documented unit interpretation, not a verified reconstruction of the authors' unavailable likelihood code.

For the deterministic environment, `environment.evaluate_predictor(agent.get_predictor())` computes the same v5 fixed-scale feature MSE (Eq. hhFixedScaleMSE, s = (3, 3, 3, 3, 12, 8)) against upstream `held_out_feature_targets(stochastic=False)`, the features of each query's full-resolution true trace, without adding queries to training data. v4's Eq. 1 is not used: `n_pre` has zero query variance on four of the six released test sets, so it is undefined there. Both environments inherit the released query set, including overlaps described below.

The unrelated official [NeuroBench](https://github.com/NeuroBench/neurobench) suite has no Hodgkin–Huxley mechanism-discovery or active-design baseline. It is used here only to independently confirm the MSE definition. Both projects install an import named `neurobench`, so run this check in a separate environment:

```bash
python -m venv .venv-neurobench
.venv-neurobench/bin/pip install -e '.[baselines]' pytest
.venv-neurobench/bin/pytest -q tests/test_neurobench_reference.py
```

### Released-benchmark limitations (NeuronBench)

The public NeuronBench repository is not the exact experimental harness described by the paper:

- it ships no MDA, SMC, likelihood, or VoI implementation;
- its pure-LLM baseline starts without the paper's fixed `D0` and receives only spike counts rather than v5's 20 sparse voltage samples (v4 used six features);
- its headline score is floored spike-count MSE; its secondary feature score uses the same fixed scales as v5's metric (v4 used per-component query variance);
- some released held-out protocols repeat design-menu protocols or D0 itself (the brief 12 µA/40 ms step is a query in h_sag, ca_rebound and d_type), whereas v5 says its queries do not overlap the design menu; and
- it exposes stochastic rollouts, but not the one-step latent transition contract needed to fit arbitrary proposed mechanisms with a bootstrap PF.

The stochastic adapter now supplies the missing transition contract and is tested against pinned upstream rollouts. PF integration is separately checked against analytic multi-step Gaussian evidence. Neither test establishes nonlinear-neuron calibration or reproduces the paper's scores. Its observed-feature summaries use the stochastic sampling grid, but cannot recover spikes missed between ~4 ms samples. Held-out targets and forecasts therefore use full simulated traces, while predictive checks and residual reports compare observed features with the same features averaged over rollouts sampled on the ~4 ms grid (`metadata["observed_target_prediction"]`); comparing the two grids directly overstated the residual four- to five-fold. Even so, at the true ca_rebound parameters this statistic sits near the 0.18 threshold (median 0.20 over world seeds), because one noisy run is compared with an expectation; this affects only `--online-expansion` and the residual reports. Rollouts are cached per (parameters, protocol), so the three predictors share one simulation.

v5's NeuronBench harness is not in the pinned release, so these remain different from the paper:

- v5 uses a 90-protocol menu (single pulses, conditioning-test, and paired pulses), 20 stratified voltage samples per experiment, and 24 shared held-out queries. Here the agent uses the package's 9-protocol pool, its ~4 ms observation grid, and per-world test protocols. v5 does not give the remaining pulse durations, the sampling rule, or the query schedules.
- v5 integrates its filter at 0.05 ms; this one uses the oracle's 0.025 ms.
- v5 acquisition scores deterministic feature means; this one scores 200-rollout mean voltages. v5 predicts from two 128-draw mixture samples; this one uses MAP by default or a weighted average over all particles (`--prediction bma`).
- v5 conflicts on some settings. Its main-settings table and recursive-evidence appendix give 64 parameter particles, while its main-text figure and neuron appendix give 128 (used here). Its headline cohort disables births after initialization, while its table lists online expansion (the headline setting is the default here; `--online-expansion` restores births). Its prompt and proposer schema changed to sparse voltages and fixed slopes, which is not adopted. Its ODE+BIC, SDE+BIC, ODE+recursive-SMC and BSL arms, its stochastic expansion statistic, and ICL (+LLM design) are not wired into the runner, although `BICEvidence` and `BayesianSyntheticLikelihood` implement the generic algorithms. v5's ICL baseline scores a missing forecast as a zero vector; here a malformed response raises.

### What still prevents an exact paper reproduction

- **GlucoseBench.** The MDA arm and ICL (+MDA design) are implemented, plus the package's persistence baseline, which v5 does not report (see [GlucoseBench](#glucosebench)). The ARX, VARX, and SINDYc baselines and ICL (+LLM design) are not. The paper's per-call responses and selected histories are not public, so its scores cannot be replayed.
- **ChemBench.** v5's universe is exactly the pinned registry's 57 standard worlds (`chemistry.benchmark_worlds()`), at medium difficulty with three seeds each. Its exact per-call responses and configuration files are not public. Its prompts are published, but this proposer uses a generic prompt. It lacks the enzyme-kinetics framing, the input units, the restricted grammar (`+ - * / **` with exp, log and sqrt; 15 functions are allowed here), the requirement of positive bounds (not enforced here), and the "structurally different" and explore/refine instructions. In exchange it sends every pool structure's priors, per-experiment residuals and residual–input correlations. The canonical laws' bounds, the number of multistarts, and v4's τ_e/R_m/N_e^max values (kept) are local choices. Rendered v5 gives no query count and says the seed changes the queries; its source uses the benchmark's fixed 1000-condition grid for the pass rate and a separate 60-condition draw for nMSLE, while this runner uses one seed-dependent 1000-point set for both. The expansion and shrink checks use raw-rate relative residuals rather than v5's log(1+y) target. CMA-ES maps C_I and C_P linearly because their lower bound is 0, although v5 says concentration axes are logarithmic. Negative predictions are clipped to 0, whereas upstream scores |ŷ|. The fixed-scale BIC control is available as `BICEvidence("fixed")` but not as a CLI arm. A law that cannot be evaluated over the queries is recorded as a failure (`evaluable: false`, `passed: false`). Held-out references are the clean primary law, as in upstream `evaluate_law`. The upstream oracle's noise is input-hashed, not a new independent draw for each run seed. ICL (+LLM design) and the SciLab/PySR baseline are not implemented.
- **ForceBench (v4 only).** v5 drops the DiscoverPhysics domain; its runners are kept as v4 legacy. They cover the six central two-particle worlds plus Circle, Ether, and Hubble. Dark-matter/three-species inference and the exact external DP-Agent comparison harness are not implemented. Ether follows the public simulator's common drift of the anchor and probes, rather than interpreting the appendix's anchor as immovable. Extension background-coefficient bounds [-1,1] and launch templates are local choices.
- **Boxing.** v5 moves it to an appendix, runs Opus 4.7 with reasoning off, drops best-so-far curves and the curriculum example, and redescribes the MIRT illustration (a fixed five-model ladder, Brier-risk design, scoring over all 192 cells). The runner here keeps v4's Boxing setup. Boxing uses reconstructed equations with private RNGs, not the original PyMC execution or identical RNG draws. IRT and survival use fixed outcomes and Brier loss; the remaining tasks use clean-mean nMSE. Dugongs follows the paper's Gaussian observation, not upstream's folded output. Generic 24-design/12-query pools are configurable local defaults, not claimed paper evaluation grids. v5's MIRT illustration is not implemented.
- Recovery covers chemistry, the six central-force worlds, and coarse neuron channel classes, not the many-body extensions. Large benchmark runs, paid proposal calls, statistical score matching, and full comparison plots remain unexecuted. Learned summaries, amortized SBI, and neural ratio estimation are future work in v5, not missing algorithms.

## What changed in v5

| v5 change | Status here |
|---|---|
| Algorithm 2 checks before the append, refits before searching, returns the current-data MAP, and has an optional `adapt-and-prune` step | Implemented; shrinkage no longer runs at initialization |
| New evidence backends: recursive updates (`marglik-pf`) and BIC (`marglik-bic`); BSL, described in v4, becomes the plug-in `loglik-bsl` | Implemented (`mda.evidence`, `BayesianSyntheticLikelihood`) |
| Default model prior is uniform (λ = 0) | `AgentConfig.occam_penalty` now defaults to 0 |
| ChemBench: BIC primary arm, nine canonical seeds plus 12 LLM laws, caps 24/6 at weight > 0.9, six proposals per expansion, 48 CMA-ES evaluations, 3+22 budget, medium difficulty, all 57 worlds, pass rate | Implemented; the SMC control (`--inference smc`) and fixed nine-law control (`--no-m-open`) are included |
| LLM: GPT-5.6 Luna at medium effort for main domains | Default provider and effort updated; replay works for both providers |
| Stochastic neuron: brief/weak D0, fixed-scale metric, recursive SMC with N_p = N_z = 128, archive frozen after LLM initialization, 2+8 budget | Implemented; `--online-expansion` restores later births; protocols repeat once the released pool is exhausted; v5's 90-protocol harness is not in the pinned release (see above) |
| Model revision: a separate `init-hyp-space`; pool residuals are optional proposer context; top-N pruning is deterministic search over the proposed pool, not an SMC kernel | Implemented (`initial_proposals`, per-domain proposers); see the fidelity map |
| GlucoseBench domain | MDA arm and ICL (+MDA design) implemented, plus the package's persistence baseline; ARX, VARX, SINDYc, and ICL (+LLM design) are not |
| ForceBench removed; task VoI and structural recovery moved to future work or removed | Kept and labeled as v4 legacy |

## License

MIT. The separately installed benchmarks retain their own licenses.
