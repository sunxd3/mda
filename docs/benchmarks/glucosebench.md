# GlucoseBench

GlucoseBench asks the agent to learn a compartmental model of a virtual type-1 diabetes patient's glucose–insulin response and forecast new meal and bolus interventions. It is new in v5 and one of its three main domains. The runner is [`mda.benchmarks.glucose`](../../src/mda/benchmarks/glucose.py), which implements the learner side of v5's Appendix C.

## The benchmark package

The `glucose` extra pins the public [glucosebench](https://github.com/murphyk/glucosebench) package at `6bb0ff6`. It is a small simglucose adaptation with 30 virtual patients (`child#001`–`#010`, `adolescent#001`–`#010`, `adult#001`–`#010`), CGM every 5 minutes, and noisy plasma-insulin assays every 15 minutes over six hours.

The package owns the simulator, the two passive D0 episodes, the 48-template meal/bolus menu, the 12 sealed test interventions, the prefix payloads, and the metric. Only the package's learner-facing payloads reach the agent; it never sees patient identity, parameters, or hidden states.

## Run

```bash
# Offline: fits with one start and 15 iterations; takes about a minute.
mda-reproduce glucose --world 'adult#001' --smoke --budget 1 --output /tmp/glucose.json

# v5 setting (paid): all four selected experiments, scored at the cut-45 headline.
mda-reproduce glucose --world 'adult#001' --live --budget 4 --output /tmp/glucose-live.json

# Also score other forecast cuts and the ICL (+MDA design) baseline.
mda-reproduce glucose --world 'adult#001' --live --budget 4 --cuts 0 45 120 180 --icl --output /tmp/glucose-all.json
```

A run covers one patient for at most four selected experiments (v5 uses all four). Following the package's protocol, the agent's forecaster is snapshotted at 2, 3, 4, and 6 training episodes; acquisition is then frozen, and each snapshot is scored on the 12 sealed queries at the requested `--cuts` (default 45, the headline).

Scores are the package's per-channel training-variance nMSE, averaged over queries. Failed cells are counted, not dropped, whereas v5 scores only the 273 of 360 patient–query pairs valid for every method, budget, and cut. The package's persistence baseline, which v5 does not report, is always scored.

`--design` accepts `model` or `random`. `--prediction map` uses only the MAP model instead of the BIC-weighted mixture. `--no-m-open` also disables the initialization gate.

## Agent

**Models.** Each candidate is a JSON specification of 1–6 states, one right-hand side per state, initial states, and CGM and insulin outputs, with up to 12 positive-bounded parameters, written in the paper's restricted expression grammar. Models restart at minute 45 from their initial states (holding them before then) and are integrated with one-minute RK4 steps under the delivered meal (g/min) and bolus (U/min above basal) inputs. `gb` and `ib` are the CGM and insulin means over the training prefixes.

**Fitting and weights.** The working likelihood has AR(1) CGM residuals, with the innovation variance profiled out, and independent insulin assays with SD 10 pmol/L. It is fitted from minute 45 by multistart L-BFGS-B in log-parameter space with batched finite-difference gradients. BIC counts the ODE parameters plus the AR(1) correlation and scale, with N the number of observed targets, and weights are proportional to exp(−BIC/2) with no extra prior.

**Library.** The pool starts from four seed models: one-stage absorption with linear or glucose-dependent insulin action, two-stage meal absorption (Eq. glucose_reduced_ode), and a delayed CGM readout. The LLM, using v5's published proposer prompt and schema, adds at most two models per call, with genealogy checks and deduplication up to whitespace and redundant parentheses. The proposer is shown the fitted equations and their predictions.

- A malformed proposal, or one whose equations raise when simulated once at its declared parameter values, is dropped without discarding the others.
- A rollout that only diverges is kept. A model whose expressions raise during fitting gets zero evidence and leaves the pool.
- Missing or non-finite scheduled CGM readings or insulin assays raise an error naming the episode and minute.
- The archive is capped at 14 with one search round per update.
- The payload also gives each model's fitted AR(1) ρ and σ, which the prompt does not list. The expression parser also accepts `%`, `pi`, and `e`, which the prompt does not offer.

**Expansion gate.** Expansion happens when even the best archived model fails a whitened-residual check on the new episode before refitting: the minimum over models of A_m exceeds χ²₀.₉₉₅(86)/86 ≈ 1.436. At initialization the gate uses each model's worst D0 episode. This statistic is printed only in the paper's source, not its rendered text. Residuals are anchored at the pooled training-prefix means; that source anchors each checked episode at its own 45-minute prefix.

**Design.** Each unused template is scored by the BIC-weighted between-model variance of point predictions, normalized by the model-weighted stationary AR(1) variance for CGM and the assay variance for insulin, averaged over observation times, with ties broken randomly.

**Forecasting.** An EKF over the ODE state and the AR(1) sensor error conditions on the query prefix, then propagates without further updates. The forecast is the BIC-weighted mixture of component means at point parameters. A component whose forecast fails, including one whose EKF state becomes non-finite or reaches 1e7 in magnitude (the divergence rule the fits use), drops out and the rest are renormalized. Forecasts are means only: v5's 90% mixture-quantile intervals are not computed.

**ICL (+MDA design).** `--icl` adds the data-only forecaster with v5's census forecast prompt at every cut.

## Local choices where v5 is silent

- The equations and bounds of the three seed models whose equations v5 does not print.
- Three random starts plus the declared and warm-start values.
- AR(1) correlation in [0, 0.99].
- Designs scored on the nominal delivery schedule (meals at 5 g/min, bolus over 5 minutes) without the pump's quantization, which fits do include.
- The EKF's initial state SD (5% of magnitude, floor 1) and finite-difference Jacobians.
- Holding the restart state through the prefix, so that EKF updates can use every prefix observation.
- v5 mentions an EKF positivity safeguard without specifying it; none is applied beyond the divergence guard above.

## Not implemented

- The ARX, VARX, and SINDYc baselines, and ICL (+LLM design).
- v5's separate multi-cut ICL prompt (an explicit empty prefix at cut 0, up to two schema retries). v5 does not say which prompt produced its cut-45 headline.
- The paper's per-call responses and selected histories are not public, so its scores cannot be replayed.
