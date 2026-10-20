# Concepts

The Model Discovery Agent works in an **M-open** setting: the true model is not assumed to be in the initial candidate set. It keeps a weighted pool of candidate model structures, chooses experiments that best separate them, and asks an LLM for new structures only when the pool stops predicting well.

## Division of labor

The LLM is used in one place only: the **proposer**, which writes new model structures. Every weight, evidence value, design score, and prediction is numerical. This is the paper's core design and the implementation keeps it.

| Role | Done by | Code |
|---|---|---|
| Propose structures | LLM | `ModelProposer` implementations |
| Fit parameters and weigh structures | Bayesian inference | `mda.evidence`, `mda.smc`, `mda.posterior` |
| Choose the next experiment | Value of information | `mda.acquisition`, `mda.optimizers` |
| Decide whether to expand the pool | Predictive check | `mda.checks` |
| Predict | MAP or Bayesian model averaging | `ModelDiscoveryAgent.predict` |

## The agent loop

`ModelDiscoveryAgent` follows v5's Algorithm 2. Its public methods are the paper's agent interface:

| Paper | Method | What it does |
|---|---|---|
| `init` | `ModelDiscoveryAgent(...)` | Builds the pool from a seed library plus, optionally, one LLM batch (`initial_proposals`), and fits it to the initial data D0 |
| `design` | `agent.design()` | Returns the unused action with the highest value of information |
| `update-state` | `agent.update_state(action, y)` | Absorbs one experiment's result (below) |
| `get-model` | `agent.get_model()` | The current-data MAP structure |
| `get-predictor` | `agent.get_predictor(flavour)` | A function `action → prediction`, using `"map"` or `"bma"` |

Each call to `update_state` runs these steps in order:

```text
┌────────────────────────────┐
│ 1. Predictive check        │  score the new datum against the belief
│    (before the append)     │  held before it arrived
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ 2. Append and refit        │  refit every structure in the pool
│                            │  on all data so far
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐  only if the check error exceeds
│ 3. Expand (up to R rounds) │  error_threshold: ask the proposer
│                            │  for new structures, fit them, recheck
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ 4. Adapt and prune         │  optional: shrink the pool cap
│                            │  when one model clearly dominates
└────────────────────────────┘
```

1. **Predictive check.** The median relative residual of the pre-update MAP fit on the fresh datum and, with `backtest=True` (the default), on the earlier data. Because the fresh datum is scored before it is added, this is a prequential, out-of-sample test. A design the current best model cannot evaluate counts as a failed check. Domains can supply their own `check`.
2. **Refit.** The datum is appended and the pool is refitted, so proposals see residuals of the refitted pool. If the refit fails, the agent is left exactly as it was before the call.
3. **Expand.** If the check failed, the proposer receives the context, data, every model with its priors, and per-model residual reports, and returns new structures. Only the new structures are fitted; the others reuse their fits on the same data. Checks after an expansion are in-sample. Every proposer call counts toward `expansion_cap`, even one that returns only invalid or duplicate models.
4. **Adapt and prune.** Optional (`minimum_model_cap`). The pool is cut to the top-N structures by weight and the retained weights are renormalized.

## Model weights

Each structure's weight is proportional to its prior times its marginal likelihood (evidence). The model prior is uniform by default, or `exp(−λk)` for a structure with `k` parameters (`occam_penalty=λ`). Weights form a posterior over the **proposed pool only**: top-N pruning is a deterministic search step, not a posterior-invariant kernel, and assigns nothing to models never proposed.

A model whose weight underflows to zero stays in the pool and can regain weight later. Only structures whose fit fails entirely leave it.

## Building blocks

- **Candidate model** (`CandidateModel`): a structure, a prior over its parameter vector, and a `predict(theta, action)` function. Stochastic models also supply `StateSpaceModel` callbacks.
- **Likelihood**: scores data under one parameter value. Exact Gaussian for deterministic simulators (`loglik-det`), a bootstrap particle filter for stochastic state-space models (`loglik-pf`), or Bayesian synthetic likelihood of trajectory summaries (`loglik-bsl`).
- **Evidence backend**: turns a likelihood into a parameter posterior and a log-evidence per structure. Tempered SMC on the full history (`marglik-smc`, the default), recursive per-experiment reweighting (`marglik-pf`), or multistart optimization with BIC (`marglik-bic`).
- **Value of information**: model EIG is the weighted, noise-whitened disagreement between the structures' mean predictions at an action; joint EIG measures disagreement across all parameter particles of all structures. These are the trace surrogates the paper's experiments use, not exact mutual information.
- **Design search**: a finite action menu, or a continuous box searched by CMA-ES.
- **Proposer**: the LLM boundary. Symbolic laws are AST-validated; generated NumPyro programs need explicit permission to execute.

The [Library guide](library.md) shows how to use each of these. [Paper fidelity](paper-fidelity.md) maps each paper algorithm to its code.
