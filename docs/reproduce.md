# Reproducing the paper

`mda-reproduce` runs **one** explicit benchmark configuration and writes a JSON result. It never runs an implicit sweep and never makes paid calls unless you pass `--live`.

```bash
mda-reproduce DOMAIN --world WORLD (--smoke | --replay | --live) --budget N --output result.json [options]
```

Domains: `chemistry`, `glucose`, `neuron`, `boxing`, and the v4-only `physics`. See each [benchmark page](#benchmarks) for its worlds and settings.

## Install

```bash
.venv/bin/pip install -e '.[dev,numpyro,physics]'
```

### Extra setup for ChemBench and ForceBench

```bash
# ChemBench is an unpackaged source dependency (NumPy oracle only; no PySR).
git clone https://github.com/scientific-discovery/LLM-AutoSciLab.git /tmp/LLM-AutoSciLab
git -C /tmp/LLM-AutoSciLab checkout 9a37255ff5432a7155b6cde8985e9654f2f8f272
export AUTOSCILAB_SOURCE_ROOT=/tmp/LLM-AutoSciLab

# ForceBench (v4 only): PhysicsSchool's pinned wheel omits namespace subpackages,
# so it needs an editable checkout in addition to the physics extra above.
git clone https://github.com/SampsonML/DiscoverPhysics.git /tmp/DiscoverPhysics
git -C /tmp/DiscoverPhysics checkout 33b7fa9df96de9c35744efd181ca7e5a8dd60ad5
.venv/bin/pip install -e /tmp/DiscoverPhysics/PhysicsSchool
```

GlucoseBench and NeuronBench are installed by the `glucose` and `neuron` extras.

## Smoke runs

One offline experiment per domain:

```bash
.venv/bin/mda-reproduce chemistry --world c0_michaelis_menten --smoke --budget 1 --output /tmp/chemistry.json
.venv/bin/mda-reproduce glucose --world 'adult#001' --smoke --budget 1 --output /tmp/glucose.json
.venv/bin/mda-reproduce neuron --world h_sag --stochastic --smoke --budget 1 --output /tmp/neuron.json
.venv/bin/mda-reproduce boxing --world death --smoke --budget 1 --output /tmp/boxing.json
.venv/bin/mda-reproduce boxing --world death --agent apprentice --smoke --budget 1 --output /tmp/apprentice.json
.venv/bin/mda-reproduce physics --world gravity --smoke --budget 1 --output /tmp/physics.json  # v4 only
```

## Run modes

Exactly one mode is required.

| Mode | LLM calls | Use |
|---|---|---|
| `--smoke` | None | Small numerical budgets and generic, non-truth baselines (for ChemBench, the nine canonical laws). Neuron smoke checks feature forecasts but skips the expensive held-out 200-rollout scoring |
| `--replay` | Cache only | Reads previously captured responses and fails on a cache miss without contacting a provider |
| `--live` | Paid | Explicitly authorizes paid calls; responses are cached per domain, world, and seed |

As in v5, the default provider is OpenAI `gpt-5.6-luna` at medium reasoning effort, except Boxing, whose appendix results use Anthropic `claude-opus-4-7` with reasoning off. v5 states no temperature and none is sent. Requests are streamed, so any output budget is accepted, and only a response that stopped at the forced tool call is cached. `--provider` and `--model` override these. `--provider claude-code` or `codex` sends the same prompts through those coding agents (install `.[agents]`); see [coding-agent clients](library.md#coding-agent-clients). No model availability or live score claim is made here.

## Settings and budgets

Normal (non-smoke) runs use the per-domain settings of v5's main-settings table and domain appendices; Boxing keeps v4's Table 2 column.

Always pass `--budget`. The CLI default is one experiment, not a paper sweep. The paper's budgets (initial + designed experiments):

| Domain | v5 budget | `--budget` |
|---|---|---|
| ChemBench | 3 + 22 | 22 |
| GlucoseBench | 2 + 4 | 4 |
| Stochastic neuron | 2 + 8 | 8 |

A finite-menu run whose budget exceeds the unused designs fails before it starts. The released neuron pool has only seven unused protocols after D0, so the stochastic neuron runner repeats protocols once they are exhausted.

## Options

| Option | Applies to | Effect |
|---|---|---|
| `--world` | all | World, patient, or task name (required) |
| `--seed` | all | Run seed (default 0) |
| `--budget` | all | Number of designed experiments (default 1) |
| `--design {model,joint,task,random}` | all | Design objective. `random` samples a finite menu, or ChemBench's log-mapped box uniformly, from a generator that depends only on the seed, so random arms with different inference share their design history, as v5's ChemBench controls do |
| `--prediction {map,bma}` | all | Default `bma` for Boxing and GlucoseBench, `map` otherwise |
| `--no-m-open` | all | Disable expansion after the initial pool. For ChemBench also skips LLM initialization (v5's fixed nine-law control); for GlucoseBench also disables the initialization gate |
| `--icl` | replay/live | Add the data-only ICL forecaster after every round (v5's "ICL (+MDA design)") |
| `--recovery` | chemistry, physics, neuron | Structural recovery diagnostic (install `.[recovery]`) |
| `--provider`, `--model`, `--cache-dir` | replay/live | Provider (`openai`, `anthropic`, `claude-code`, `codex`), model, and response cache (default `.cache/mda-responses`) |
| `--difficulty` | chemistry | Default `medium` (v5's setting) |
| `--inference {bic,smc}` | chemistry | Profiled-BIC primary arm (default) or the SMC control |
| `--cuts`, `--replicate` | glucose | Forecast cuts in minutes (0, 45, 120, 180; default 45) and replicate (default 0) |
| `--design-space {menu,continuous}`, `--check {v4,chi2,guided}`, `--max-params` | central two-particle physics | The 13-action menu (default) or the continuous launch box; v4's expansion check (default), a χ² test on the fresh trajectory, or that test with residual-guided expansions; and a cap on free parameters per law; see [ForceBench](benchmarks/forcebench.md#continuous-design-and-the-expansion-check) |
| `--stochastic` | neuron | Use v5's stochastic neuron environment (the deterministic one is v4 legacy) |
| `--online-expansion` | stochastic neuron | Re-enable model births after initialization |
| `--task-noise-variance` | `--design task` | Target-space noise. Required for neuron task design: one or six positive feature-space variances |
| `--agent {mda,apprentice}` | boxing | `apprentice` runs the Boxing Apprentice comparison |
| `--allow-code-execution` | boxing | **Unsafe**: execute generated NumPyro Python. See [BoxingGym](benchmarks/boxing.md#generated-code) |

Flags that would have no effect in the chosen configuration are rejected rather than recorded as if they applied.

Neuron task design requires an explicit `--task-noise-variance` because no paper version releases that calibration; it must not silently use raw voltage noise for six different features.

## Output

The JSON result contains:

- the request, package versions, effective numerical settings, and model name;
- for each round: the action, observation, current MAP model and parameters, model weights, log-evidences, forced-tempering diagnostics, predictive error, expansion count, and score (with ChemBench nMSLE, RMSLE, and pass flag under `metrics`);
- a separately labeled best-so-far curve (`best_score_so_far`), which is a reporting convention only.

Evaluation never selects models or feeds query labels into training. Output files are never overwritten; choose a new path for each run.

A run that fails part-way still writes every completed snapshot, with an `error` field, and exits non-zero. A failure of the `--icl` side baseline is recorded on that snapshot (`icl_score: null`, `icl_error`) without ending the run.

## Baselines and diagnostics

### ICL forecaster

`mda.benchmarks.icl.make_icl_predictor` is a model-free baseline that sees only observed action–target pairs and the query inputs. The request budgets 32 output tokens per predicted value with an 800-token floor. For the neuron it sees the six summary features with a generic prompt, not v5's 20 raw voltage samples and HH prompt. GlucoseBench uses its own ICL forecaster; see [GlucoseBench](benchmarks/glucosebench.md).

### Boxing Apprentice

`--agent apprentice` keeps one evolving model, lets the LLM select experiments, and refines the model with residual conditioning every round, fitted by the same SMC.

### Structural recovery (v4 diagnostic)

`--recovery` adds constant-refit recovery scoring for ChemBench, the central two-particle ForceBench worlds, and coarse neuron channel classes. These are v4 diagnostics: v5 reports the ChemBench pass rate (`metrics.passed`) and moved structural recovery out of the paper. Recovery parameters never replace the agent's fitted parameters.

`mda.benchmarks.recovery.evaluate_recovery` refits a trusted candidate's constants on a separate noise-free grid and tests held-out RMSLE. The evaluators first compile the symbolic law with `canonical_symbolic_candidate`: a whitelist AST translation into SymPy, followed by simplification and numerical compilation, without parsing arbitrary Python through `sympify`. This is a numerical approximation, not a proof of symbolic equality. Neither grid below is claimed to be the authors' unreleased evaluation grid.

- **Chemistry**: separate 128-point fit and holdout grids, the clean primary law as truth, and ordinary `log1p` with tolerance 0.02. A refit start that leaves the law undefined is skipped; a law undefined at every start is recorded as not recovered.
- **ForceBench**: the true radial force is measured through clean public release-at-rest experiments, never private force fields. The grid has 128 interleaved log-spaced radii in [1.5, 10] with a 0.01 time-unit finite difference, signed `log1p`, and tolerance 0.05. This tests the radial profile at unit coupling and time zero, not the complete charge and time dependence.
- **Neuron**: scores the coarse inward/outward, depolarisation/hyperpolarisation, and transient/persistent channel class (or slow Na), not exact kinetics. The evaluator uses the proposed channel's properties, not its name. Zero conductance, or a reversal potential exactly at the local −65 mV reference, yields "unknown". Calcium and z-rebound intentionally share a class.

## Benchmarks

| Benchmark | Paper status | Page |
|---|---|---|
| ChemBench | v5 main domain | [chembench.md](benchmarks/chembench.md) |
| GlucoseBench | v5 main domain (new) | [glucosebench.md](benchmarks/glucosebench.md) |
| NeuronBench (stochastic neuron) | v5 main domain | [neuronbench.md](benchmarks/neuronbench.md) |
| BoxingGym | v5 appendix | [boxing.md](benchmarks/boxing.md) |
| ForceBench / DiscoverPhysics | v4 only; dropped in v5 | [forcebench.md](benchmarks/forcebench.md) |
