# Reimplementation of **Model Discovery Agent: LLM-assisted Bayesian experiment design for data-efficient discovery of mechanistic world models** ([arXiv:2608.09696v5](https://arxiv.org/abs/2608.09696v5), 27 Sep 2026).

v5 restructured the paper: its main domains are now ChemBench, the new GlucoseBench, and the stochastic neuron benchmark (typeset "HHbench", still NeuronBench); BoxingGym moved to an appendix, and the ForceBench/DiscoverPhysics domain was dropped. [What changed in v5](#what-changed-in-v5) lists how this repository follows it.

## What is implemented

- M-open model discovery with prequential predictive checks and residual-guided structure expansion, following v5's Algorithm 2 ordering.
- Three per-structure evidence backends: full-history adaptive likelihood-tempered SMC (the default), recursive parameter-weight/evidence updates (`marglik-pf`), and bounded multistart optimization with BIC (`marglik-bic`, profiled or fixed residual scale, or a direct likelihood). Model weights use an optional parameter-count prior and deterministic top-N pruning.
- Exact Gaussian likelihoods for deterministic simulators, a bootstrap particle-filter likelihood for user-supplied stochastic state-space models, and a Bayesian synthetic likelihood of trajectory summaries.
- Model EIG and joint model/parameter EIG surrogates. v4's coordinatewise task VoI and a full-observation Gaussian alternative are kept as extensions; v5 leaves task-aware design to future work.
- Finite-menu design or bounded CMA-ES optimization, including logarithmic axes.
- MAP and Bayesian-model-averaged prediction.
- AST-validated symbolic models and a structured OpenAI Responses API proposer, with optional persistent response caching. AST validation restricts syntax; it is not a resource sandbox for arbitrary untrusted computation.
- A v5 ChemBench runner: the nine canonical seed laws plus one 12-law LLM batch, profiled BIC in log-rate space, adaptive 6/24 pool caps, 48-evaluation CMA-ES, the benchmark pass flag, and an SMC control arm.
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
```

The driver requires exactly one of `--smoke`, `--replay`, or `--live`. Smoke uses small numerical budgets and generic, non-truth baselines (for ChemBench, the nine canonical laws); it makes no LLM calls. Neuron smoke checks feature forecasts but omits expensive held-out 200-rollout scoring. Replay reads previously captured responses and fails on cache misses without contacting a provider. Live explicitly authorizes paid calls. As in v5, the default provider is OpenAI `gpt-5.6-luna` at medium reasoning effort, except Boxing, whose appendix results use `claude-opus-4-7` (temperature 0.3). `--provider` and `--model` override these. No model availability or live score claim is made here.

Normal runs use the per-domain settings of v5's main-settings table and domain appendices (Boxing keeps v4's Table 2 column). Specify `--budget` explicitly (the conservative CLI default is one experiment, not a paper sweep; v5 uses 3+22 ChemBench and 2+8 neuron experiments). `--difficulty` defaults to ChemBench's v5 `medium`. `--design` selects model, joint, task, or random design; random design samples a finite menu, or ChemBench's log-mapped box uniformly. `--no-m-open` disables expansion after the initial pool; for ChemBench it also skips LLM initialization, giving v5's fixed nine-law control. `--inference smc` selects ChemBench's SMC control instead of the profiled-BIC primary arm. Boxing defaults to BMA prediction; the other domains use MAP. Neuron task design requires an explicit `--task-noise-variance` (one or six positive feature-space variances), since no paper version releases that calibration. It must not silently use raw voltage noise for six different features.

`--agent apprentice` implements the Boxing Apprentice comparison: one evolving model, LLM-selected experiments, and a residual-conditioned refinement on every round, fitted by the same SMC. `--icl` adds the data-only forecasting baseline after every round (replay/live only); this is v5's "ICL (+MDA design)". `--recovery` adds ChemBench or central two-particle ForceBench constant-refit recovery scoring (install `.[recovery]`). These are v4 diagnostics: v5 reports the ChemBench pass rate (`metrics.passed`) and moved structural recovery out of the paper. Recovery parameters never replace the agent's fitted parameters. For NeuronBench, the same flag scores the coarse inward/outward, depolarisation/hyperpolarisation, transient/persistent channel class (or slow Na), not exact kinetics. The evaluator uses the proposed channel properties, not its name; zero conductance or reversal exactly at the local -65 mV reference yields "unknown". Calcium and z-rebound intentionally share a class. The physics driver also accepts `circle`, `ether`, and `hubble`. These extensions use documented local launch menus because the exact v4 menus are unreleased.

IRT, location, and survival generate NumPyro Python. Non-smoke runs require `--allow-code-execution`: **this executes arbitrary generated Python with process privileges, not in a security sandbox**. Use only a suitably isolated environment and trusted/reviewed proposals. The AST symbolic backend does not require this flag.

JSON results contain the request, package versions, effective numerical settings, per-round actions, observations, current model/parameters, model weights, log-evidences, forced-tempering diagnostics, scores (with ChemBench nMSLE, RMSLE, and pass flag), and a separately labeled best-so-far curve. Evaluation does not select models or feed query labels into training. Output files are never intentionally overwritten.

`mda.benchmarks.icl.make_icl_predictor` provides a model-free baseline using only observed action-target pairs and query inputs. `mda.benchmarks.recovery.evaluate_recovery` refits a trusted candidate's constants on a separate noise-free grid and tests held-out RMSLE. The benchmark evaluators first compile the symbolic law using `canonical_symbolic_candidate`: a whitelist AST translation into SymPy, followed by simplification and numerical compilation, without parsing arbitrary Python through `sympify`. This is a numerical approximation, not proof of symbolic equality. ForceBench measures the true radial force through clean public release-at-rest experiments, never private force fields. Its local grid has 128 interleaved log-spaced radii in [1.5, 10], with a .01 time-unit finite difference; it uses signed `log1p` and tolerance .05. This tests the radial profile at unit coupling and time zero, not the complete charge/time dependence. Chemistry uses separate 128-point fit and holdout grids and ordinary `log1p` with tolerance .02. Neither grid is claimed to be the authors' unreleased evaluation grid.

`gpt-5.6-luna` does not accept `temperature`; the client omits it for GPT-5.6 models, requests medium reasoning effort (configurable with `reasoning_effort`), and reserves enough output tokens for strict-schema responses.

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

An optimizer receives `(score, rng)`, maximizes `score(action)`, and must return an evaluated action. The agent excludes used `action_key`s unless `repeat_designs=True`. CMA-ES returns its best finite evaluated design and honors its evaluation limit. For continuous actions, use a stable, collision-free `action_key`; the default `repr` is suitable only for small actions.

`OpenAIJSONClient(cache_dir=".cache/mda-responses")` caches completed, nonempty responses by the complete effective request, including the model, reasoning effort, and schema. The key excludes credentials; cached response text may still contain sensitive data. Keep the directory private and out of version control. The default is no cache. This enables replay of captured calls, not reproduction of the paper's unreleased responses. `replay_only=True`, on either `OpenAIJSONClient` or `AnthropicJSONClient`, fails on a missing cached response before SDK loading or credential checks.

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

Generic defaults use v5's default evidence (tempered SMC with 100 particles, three rejuvenation moves, ESS target 0.6, 80 rungs) and its default uniform model prior (`occam_penalty=0`), with v4's 16-model pool, four proposals per expansion, two search rounds, and 0.05 error threshold. Domain runners set their own values, as the paper does: ChemBench uses BIC, a 24-model cap, six proposals per expansion and no parameter-count prior; the stochastic neuron runner uses recursive SMC with `p(m) ~ exp(-2k)`.

Optional `adapt-and-prune` shrinking is enabled with `minimum_model_cap` in `AgentConfig`. After an update it requires both a well-fit MAP model (median relative residual at most `error_threshold`) and a MAP weight above `shrink_confidence` (default 0.9, v5's ChemBench value); otherwise the cap is `model_cap`. It never runs at initialization. A failed check or reduced confidence restores the larger cap; evicted models are not automatically resurrected. Shrinking is disabled by default, matching the base Algorithm 2.

`expansion_gate="cached"` reproduces v5's executed ChemBench agent, which gates its first expansion on the in-sample residuals of the preceding fit rather than on the prequential check; later rounds recheck the refitted pool in both modes.

`BeliefState.log_mixture_evidence` is computed before pruning over the deduplicated input pool, with its parameter-count model prior normalized over that pool. `discarded_mass` includes support truncation. Each parameter posterior records `ess_before_resampling` and `last_rung_forced_below_target`: reaching the rung limit forces the full likelihood (`beta=1`), possibly at the cost of low ESS. Increase the rung/particle budget if these diagnostics show collapse.

### Task-VoI (a v4 extension) has explicit target-space units

v5 keeps only the general decision-theoretic task VoI (Eq. taskvoigeneral) and leaves it to future work. The two task flavours below implement v4's Eq. 42–45 and are not part of the v5 method.

`design_flavour="task"` uses v4 Eq. 43's target predictions at **both** the proposed and query actions, with each target coordinate scored independently. It averages across query actions and sums coordinate contributions, whitening the loss by the target-space noise variances. `task_noise_covariance` supplies those variances to the agent; when omitted it falls back to `observation_covariance`. Use a separate target-space value for features derived from raw traces. `target_covariance` overrides the loss whitening (set it to 1 for unwhitened squared loss). Both must be scalar or diagonal for this flavour.

The old dense matrix conditional-variance calculation remains available as `design_flavour="task_full_observation"` or `full_observation_task_value_of_information`. It uses raw observations at the proposed action and all cross-coordinate information; optional `target_covariance` whitens its loss. This is a coherent Gaussian alternative inspired by v4 Eq. 42, but it is **not** the coordinatewise implementation described after v4 Eq. 45. Neither objective is exact information gain for a general non-Gaussian mixture.

### Pseudocode corrections and implementation choices

This repository corrected six inconsistencies in v4's pseudocode. v5 now states each of them itself: the pre-observation check runs before the append; top-N pruning renormalizes the retained weights; out-of-bounds random-walk proposals are rejected (v5 notes that the authors' NumPy fitter clipped instead, biasing its legacy results); the trace surrogate is not exact mutual information; checks after expansion are in-sample; and `get_model()` returns the current-data MAP rather than a maximum over rounds. Held-out truth is never used to select the agent's model; best-so-far curves are a separate reporting convention.

Remaining choices that v5 leaves open:

- After appending a datum the pool is refitted, so proposals see residuals of the refitted pool (v5 Algorithm 2). The re-judge after an expansion fits only the new structures and reuses the others' fits on the identical dataset, rather than rerunning their SMC.
- Profiled BIC needs more observations than parameters. When a model interpolates the data exactly (e.g. three-parameter laws on ChemBench's three-point D0), its RSS is floored at the smallest positive float, so such fits dominate until more data arrive.
- v5 does not give the number of BIC multistarts (8 here), its CMA-ES population size or initial step (unchanged), or its proposal-scale floor for SMC rejuvenation (v5 states 0.5 × max(std, 1e-3) for its fitter; this one uses a relative floor).

The bootstrap PF gives an unbiased **likelihood** estimate. Its logarithm is not unbiased, and fractional likelihood tempering is only approximately pseudo-marginal, as the paper notes in its tempered-SMC appendix.

## NeuronBench and NeuroBench checks

The `neuron` extra pins the authors' NeuronBench benchmark at [`c354622`](https://github.com/murphyk/neuronbench/commit/c354622458c460b419cab821d482c879f0578377), still its latest commit. v5's "HHbench" is this benchmark: the six worlds and their kinetics are unchanged, but v5 uses a new, unreleased harness. The adapter never reads its private truth mechanism. It uses the released simulator for candidate rollouts and gives the proposer the feature vector `[n_test, n_pre, run_down, adaptation, V_min, V_end]`.

The stochastic environment follows v5 where the pinned package allows it. D0 is the brief 12 µA/40 ms and weak 5 µA/120 ms steps. `evaluate_predictor` computes v5's fixed-scale feature MSE, Eq. hhFixedScaleMSE with s = (3, 3, 3, 3, 12, 8), against 200-rollout expected features (v5 gives no rollout count). The runner uses a 128-particle PF and 128-particle tempered SMC with 16 initialization rungs, then recursive updates without rejuvenation. The archive is the plain Na/K/leak cell plus three LLM proposals, and the model prior is `exp(-2k)`.

The deterministic `NeuronBenchEnvironment`, with v4's Table 14 noise model, Eq. 1 query-variance metric, and brief/long-step D0, is v4 legacy: v5 has no deterministic benchmark.

`NeuronHypothesis.tau_gate` selects activation (Ih/M) or inactivation (Ca/D), with `activation_tau_ms` fixed in the latter case. Anonymous channel proposals remain supported; their bounds and fixed gating forms come from the proposer, not a hidden truth-based archetype lookup. Table 14 and H.4.1 disagree about which gate's tau is inferred; this API makes that choice explicit.

Slow-Na candidates infer `gL ~ Uniform(0.2,0.45)` and `tau_s_ms ~ Uniform(200,600)`. A local deterministic rollout extends the pinned simulator's equations without changing the dependency: the sampled tau replaces the depolarized branch only, with the recovery branch fixed at 1500 ms. At 380 ms it reproduces the released rollout. This interpretation of Table 14's single tau is explicit rather than claiming the paper specifies both branches.

Voltage traces remain in raw mV. Interpreting Table 14's “1 unit = 3 mV” as `V/3`, `observation_std` returns `max(0.12*abs(V), 3)` mV. The table mixes feature and trace terminology, so this is a documented unit interpretation, not a verified reconstruction of the authors' unavailable likelihood code.

For the deterministic environment, `environment.evaluate_predictor(agent.get_predictor())` computes v4 Eq. 1's mean per-component query-variance-normalized MSE without adding queries to training data. It raises for zero query variance rather than introducing an undocumented floor. Both environments inherit the released query set, including overlaps described below.

The unrelated official [NeuroBench](https://github.com/NeuroBench/neurobench) suite has no Hodgkin–Huxley mechanism-discovery or active-design baseline. It is used here only to independently confirm the MSE definition. Both projects install an import named `neurobench`, so run this check in a separate environment:

```bash
python -m venv .venv-neurobench
.venv-neurobench/bin/pip install -e '.[baselines]' pytest
.venv-neurobench/bin/pytest -q tests/test_neurobench_reference.py
```

### Released-benchmark limitations

The public NeuronBench repository is not the exact experimental harness described by the paper:

- it ships no MDA, SMC, likelihood, or VoI implementation;
- its pure-LLM baseline starts without the paper's fixed `D0` and receives only spike counts rather than the paper's six features;
- its headline score is floored spike-count MSE; its secondary feature score uses the same fixed scales as v5's metric (v4 used per-component query variance);
- some released held-out protocols repeat brief/long protocols from the design menu despite the paper calling the sets disjoint; and
- it exposes stochastic rollouts, but not the one-step latent transition contract needed to fit arbitrary proposed mechanisms with a bootstrap PF.

The stochastic adapter now supplies the missing transition contract and is tested against pinned upstream rollouts. PF integration is separately checked against analytic multi-step Gaussian evidence. Neither test establishes nonlinear-neuron calibration or reproduces the paper's scores. Its observed-feature summaries use the stochastic sampling grid, but cannot recover spikes missed between ~4 ms samples; expected targets use full simulated traces.

v5's HHbench harness is not released, so these remain different from the paper:

- v5 uses a 90-protocol menu (single pulses, conditioning-test, and paired pulses), 20 stratified voltage samples per experiment, and 24 shared held-out queries. Here the agent uses the package's 9-protocol pool, its ~4 ms observation grid, and per-world test protocols. v5 does not give the remaining pulse durations, the sampling rule, or the query schedules.
- v5 integrates its filter at 0.05 ms; this one uses the oracle's 0.025 ms.
- v5 acquisition scores deterministic feature means; this one scores 200-rollout mean voltages. v5 predicts from two 128-draw mixture samples; this one uses MAP by default or a weighted average over all particles (`--prediction bma`).
- v5 conflicts on some settings. Its main-settings table and recursive-evidence appendix give 64 parameter particles, while its main-text figure and neuron appendix give 128 (used here). Its headline cohort disables births after initialization, while its table lists online expansion (kept; use `--no-m-open` to freeze the archive). Its prompt and proposer schema changed to sparse voltages and fixed slopes, which is not adopted. Its ODE+BIC, SDE+BIC and BSL arms and its stochastic expansion statistic are not wired into the runner, although `BICEvidence` and `BayesianSyntheticLikelihood` implement the generic algorithms.

### What still prevents an exact paper reproduction

- **GlucoseBench is not implemented.** v5's new partially observed ODE domain uses the public [glucosebench](https://github.com/murphyk/glucosebench) package (a modified simglucose with 30 patients, 48 meal/bolus templates, and 12 held-out interventions scored after a 45-minute prefix). The agent side needs an ODE-model DSL, AR(1)-noise likelihood fitting with BIC, EKF conditioning on the query prefix, and ARX/SINDYc/ICL baselines. v5 leaves several of these choices open (three of its four seed models' equations, multistart and EKF settings, and the expansion gate).
- **ChemBench.** v5's universe is exactly the pinned registry's 57 standard worlds (`chemistry.benchmark_worlds()`), at medium difficulty with three seeds each. Its exact per-call responses and configuration files are not public. Its prompts are published, but this proposer still uses a generic prompt rather than the paper's enzyme-kinetics prompt. The canonical laws' bounds, the number of multistarts, the rendered query count (1000 here, seed-dependent as v5 states), and v4's τ_e/R_m/N_e^max values (kept) are local choices. Held-out references are the clean primary law, as in upstream `evaluate_law`. The upstream oracle's noise is input-hashed, not a new independent draw for each run seed. ICL (+LLM design) and the SciLab/PySR baseline are not implemented.
- **ForceBench (v4 only).** v5 drops the DiscoverPhysics domain; its runners are kept as v4 legacy. They cover the six central two-particle worlds plus Circle, Ether, and Hubble. Dark-matter/three-species inference and the exact external DP-Agent comparison harness are not implemented. Ether follows the public simulator's common drift of the anchor and probes, rather than interpreting the appendix's anchor as immovable. Extension background-coefficient bounds [-1,1] and launch templates are local choices.
- **Boxing.** It is unchanged between v4 and v5 apart from wording, and still uses Opus 4.7. Boxing uses reconstructed equations with private RNGs, not the original PyMC execution or identical RNG draws. IRT and survival use fixed outcomes and Brier loss; the remaining tasks use clean-mean nMSE. Dugongs follows the paper's Gaussian observation, not upstream's folded output. Generic 24-design/12-query pools are configurable local defaults, not claimed paper evaluation grids. v5 removes v4's MIRT curriculum example; its higher-signal MIRT illustration is not implemented.
- Recovery covers chemistry, the six central-force worlds, and coarse neuron channel classes, not the many-body extensions. Large benchmark runs, paid proposal calls, statistical score matching, and full comparison plots remain unexecuted. Learned summaries, amortized SBI, and neural ratio estimation are future work in v5, not missing algorithms.

## What changed in v5

| v5 change | Status here |
|---|---|
| Algorithm 2 checks before the append, refits before searching, returns the current-data MAP, and has an optional `adapt-and-prune` step | Implemented; shrinkage no longer runs at initialization |
| New evidence backends: recursive updates (`marglik-pf`) and BIC (`marglik-bic`); new BSL likelihood | Implemented (`mda.evidence`, `BayesianSyntheticLikelihood`) |
| Default model prior is uniform (λ = 0) | `AgentConfig.occam_penalty` now defaults to 0 |
| ChemBench: BIC primary arm, nine canonical seeds plus 12 LLM laws, caps 24/6 at weight > 0.9, six proposals per expansion, 48 CMA-ES evaluations, 3+22 budget, medium difficulty, all 57 worlds, pass rate | Implemented; the SMC control (`--inference smc`) and fixed nine-law control (`--no-m-open`) are included |
| LLM: GPT-5.6 Luna at medium effort for main domains | Default provider and effort updated; replay works for both providers |
| Stochastic neuron: brief/weak D0, fixed-scale metric, recursive SMC with N_p = N_z = 128 | Implemented; v5's 90-protocol harness is unreleased (see above) |
| GlucoseBench domain | Not implemented (see above) |
| ForceBench removed; task VoI and structural recovery moved to future work or removed | Kept and labeled as v4 legacy |

## License

MIT. The separately installed benchmarks retain their own licenses.
