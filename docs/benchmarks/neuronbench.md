# NeuronBench (stochastic neuron)

NeuronBench asks the agent to identify the missing ion channel in a Hodgkin–Huxley-style neuron from current-clamp experiments. v5 calls it the stochastic neuron benchmark (its LaTeX macro is `\HHbench`) and makes it one of its three main domains. The runner is [`mda.benchmarks.neuron`](../../src/mda/benchmarks/neuron.py), with the environments in [`mda.neuron_stochastic`](../../src/mda/neuron_stochastic.py) and [`mda.neuronbench_adapter`](../../src/mda/neuronbench_adapter.py).

## The benchmark package

The `neuron` extra pins the authors' [NeuronBench](https://github.com/murphyk/neuronbench) at [`c354622`](https://github.com/murphyk/neuronbench/commit/c354622458c460b419cab821d482c879f0578377), still its latest commit. v5 points to this repository: the six worlds and their kinetics are unchanged, but the pinned commit lacks v5's 90-action, 20-sample, 24-query harness.

Worlds: `h_sag`, `ca_rebound`, `d_type`, `z_rebound`, `na_fatigue`, `textbook_M`.

The adapter never reads the private truth mechanism. It uses the released simulator for candidate rollouts and gives the proposer the feature vector `[n_test, n_pre, run_down, adaptation, V_min, V_end]`.

## Run

```bash
# Offline: checks feature forecasts but skips held-out 200-rollout scoring.
mda-reproduce neuron --world h_sag --stochastic --smoke --budget 1 --output /tmp/neuron.json

# v5 setting (paid): 2 + 8 experiments, archive frozen after initialization.
mda-reproduce neuron --world ca_rebound --stochastic --live --budget 8 --output /tmp/neuron-live.json

# The authors' pure-LLM baseline.
mda-neuron-baseline --world ca_rebound --budget 2 --model gpt-5.6-luna
```

Task design (`--design task`) requires `--task-noise-variance` with one or six positive feature-space variances.

## Stochastic environment (v5)

The environment follows v5 where the pinned package allows it.

| Setting | Value |
|---|---|
| D0 | The brief 12 µA/40 ms and weak 5 µA/120 ms steps |
| Design | The package's 9-protocol pool; seven are unused after D0, so the eighth experiment repeats one |
| Likelihood | 128-particle bootstrap PF |
| Evidence | 128-particle tempered SMC with at most 16 adaptive initialization rungs (the last forced to β = 1), then recursive updates without rejuvenation |
| Initialization SMC | ESS target 0.5 and three rejuvenation moves (v4's HH settings; v5 gives neither) |
| Archive | The plain Na/K/leak cell plus three LLM proposals, frozen after initialization as in v5's headline cohort |
| Model prior | `exp(−2k)` |
| Prediction | MAP by default, or a weighted average over all particles (`--prediction bma`) |
| Metric | v5's fixed-scale feature MSE (Eq. hhFixedScaleMSE) with s = (3, 3, 3, 3, 12, 8), against 200-rollout expected features (v5 gives no rollout count) |

`--online-expansion` (`online_expansion=True`) re-enables v4's residual-triggered expansion. That is not v5's stochastic rank-test gate, which appears only in the paper's source, in an audit where 1 of 36 runs triggered it and the three added models ended with 0.067% of the weight.

### Channel parameterization

This is not v5's parameterization. In v5's headline priors the LLM gives bounds for g, V½, and the activation τ (or only the slow-Na τ), and the backend adds gL and, for transient currents, τ_h ~ U(20, 300) ms, fixing slopes at 8/−6 mV and the inactivation V½ 22 mV below activation. Here:

- Transient channels infer one τ (four parameters rather than five, which also changes the `exp(−2k)` prior), and the LLM fixes the other gating constants.
- Slow-Na uses U(200, 600) ms rather than LLM bounds.
- The proposer and ICL context include the package's `text_prior`. For `textbook_M` it names the M current, whereas v5 gives the agent a prior that does not name the novel current.

### Observation grid

Observed-feature summaries use the stochastic sampling grid, but cannot recover spikes missed between ~4 ms samples. Held-out targets and forecasts therefore use full simulated traces, while predictive checks and residual reports compare observed features with the same features averaged over rollouts sampled on the ~4 ms grid (`metadata["observed_target_prediction"]`). Comparing the two grids directly overstated the residual four- to five-fold.

Even so, at the true ca_rebound parameters this statistic sits near the 0.18 threshold (median 0.20 over world seeds), because one noisy run is compared with an expectation. This affects only `--online-expansion` and the residual reports. Rollouts are cached per (parameters, protocol), so the three predictors share one simulation.

### Validation

The stochastic adapter supplies the one-step latent transition contract the release lacks and is tested against pinned upstream rollouts. PF integration is separately checked against analytic multi-step Gaussian evidence. Neither test establishes nonlinear-neuron calibration or reproduces the paper's scores.

## Deterministic environment (v4 legacy)

`NeuronBenchEnvironment` (without `--stochastic`) keeps v4's Table 14 noise model, brief/long-step D0, and online expansion; v5 has no deterministic benchmark. It is scored with the same fixed-scale metric: `environment.evaluate_predictor(agent.get_predictor())` compares against upstream `held_out_feature_targets(stochastic=False)`, the features of each query's full-resolution true trace, without adding queries to the training data. v4's Eq. 1 is not used: `n_pre` has zero query variance on four of the six released test sets, so it is undefined there.

Voltage traces remain in raw mV. Interpreting Table 14's "1 unit = 3 mV" as `V/3`, `observation_std` returns `max(0.12·|V|, 3)` mV. The table mixes feature and trace terminology, so this is a documented unit interpretation, not a verified reconstruction of the authors' unavailable likelihood code.

## Channel hypotheses

`NeuronHypothesis.tau_gate` selects whether the inferred τ belongs to activation (Ih/M) or inactivation (Ca/D), with `activation_tau_ms` fixed in the latter case. v4's Table 14 and H.4.1 disagreed about which gate's τ is inferred; v5 settles it (see above), and this API keeps the choice explicit. Anonymous channel proposals are supported; their bounds and fixed gating forms come from the proposer, not a hidden truth-based archetype lookup.

Slow-Na candidates infer `gL ~ Uniform(0.2, 0.45)` and `tau_s_ms ~ Uniform(200, 600)`. A local deterministic rollout extends the pinned simulator's equations without changing the dependency: the sampled τ replaces the depolarized branch only, with the recovery branch fixed at 1500 ms. At 380 ms it reproduces the released rollout. This interpretation of Table 14's single τ is explicit rather than claiming the paper specifies both branches.

## Limitations of the released benchmark

The public NeuronBench repository is not the exact experimental harness described by the paper:

- it ships no MDA, SMC, likelihood, or VoI implementation;
- its pure-LLM baseline starts without the paper's fixed D0 and receives only spike counts rather than v5's 20 sparse voltage samples (v4 used six features);
- its headline score is floored spike-count MSE; its secondary feature score uses the same fixed scales as v5's metric (v4 used per-component query variance);
- some released held-out protocols repeat design-menu protocols or D0 itself (the brief 12 µA/40 ms step is a query in h_sag, ca_rebound, and d_type), whereas v5 says its queries do not overlap the design menu; both environments here inherit the released query set;
- it exposes stochastic rollouts, but not the one-step latent transition contract needed to fit arbitrary proposed mechanisms with a bootstrap PF.

## Differences from v5

v5's harness is not in the pinned release, so these remain different from the paper:

- v5 uses a 90-protocol menu (single pulses, conditioning-test, and paired pulses), 20 stratified voltage samples per experiment, and 24 shared held-out queries. Here the agent uses the package's 9-protocol pool, its ~4 ms observation grid, and per-world test protocols. v5 does not give the remaining pulse durations, the sampling rule, or the query schedules.
- v5 integrates its filter at 0.05 ms; this one uses the oracle's 0.025 ms.
- v5's acquisition scores deterministic feature means; this one scores 200-rollout mean voltages. v5 predicts from two 128-draw mixture samples; this one uses MAP by default or a weighted average over all particles.
- v5 conflicts on some settings. Its main-settings table and recursive-evidence appendix give 64 parameter particles, while its main-text figure and neuron appendix give 128 (used here). Its headline cohort disables births after initialization, while its table lists online expansion (the headline setting is the default here). Its prompt and proposer schema changed to sparse voltages and fixed slopes, which is not adopted.
- Its ODE+BIC, SDE+BIC, ODE+recursive-SMC, and BSL arms, its stochastic expansion statistic, and ICL (+LLM design) are not wired into the runner, although `BICEvidence` and `BayesianSyntheticLikelihood` implement the generic algorithms.
- `RecursiveEvidence` keeps each structure's particles while the data only grow, including after the archive changes; v5's HH implementation refits when the archive changes.
- v5's ICL baseline scores a missing forecast as a zero vector; here a malformed response raises. The ICL baseline sees the six summary features with a generic prompt, not v5's 20 raw voltage samples and HH prompt.
- `--recovery` scores the coarse channel class only; see [Reproducing the paper](../reproduce.md#structural-recovery-v4-diagnostic).

## NeuroBench metric check

The unrelated official [NeuroBench](https://github.com/NeuroBench/neurobench) suite has no Hodgkin–Huxley mechanism-discovery or active-design baseline. It is used only to independently confirm the MSE definition. Both projects install an import named `neurobench`, so run the check in a separate environment:

```bash
python -m venv .venv-neurobench
.venv-neurobench/bin/pip install -e '.[baselines]' pytest
.venv-neurobench/bin/pytest -q tests/test_neurobench_reference.py
```
