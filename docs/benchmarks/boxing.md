# BoxingGym

BoxingGym is a suite of seven small scientific-modeling tasks. v5 moved it to an appendix; the runner here keeps v4's Boxing setup. The runner is [`mda.benchmarks.boxing`](../../src/mda/benchmarks/boxing.py), with the Apprentice comparison in [`mda.benchmarks.apprentice`](../../src/mda/benchmarks/apprentice.py).

Worlds: `dugongs`, `death`, `peregrines`, `hyperbolic`, `survival`, `irt`, `location`.

## Run

```bash
# Offline, generic non-truth baseline model.
mda-reproduce boxing --world death --smoke --budget 1 --output /tmp/boxing.json
mda-reproduce boxing --world death --agent apprentice --smoke --budget 1 --output /tmp/apprentice.json

# Paid; Boxing defaults to Anthropic claude-opus-4-7 with reasoning off.
mda-reproduce boxing --world death --live --budget 10 --output /tmp/boxing-live.json
```

Boxing defaults to BMA prediction.

## Environment

The equations and priors are transcribed (not copied) from BoxingGym commit `b43e38c` and reconstructed with private, seeded RNGs, so this is not the original PyMC execution and does not reproduce identical RNG draws.

- IRT and survival use fixed outcomes, sampled once, scored with Brier loss. The remaining tasks use clean-mean nMSE.
- Dugongs follows the paper's Gaussian observation, not upstream's folded output.
- Generic pools of 24 designs and 12 queries are configurable local defaults, not claimed paper evaluation grids.

## Agent

v4 Table 2's Boxing column: a 16-model cap, four proposals per expansion, two search rounds, at most four expansions, error threshold 1, `occam_penalty=1`, and tempered SMC with 1500 particles, four rejuvenation moves, ESS target 0.5, and 60 rungs. Normal runs start from LLM proposals only; smoke runs use one generic model and no client.

Symbolic tasks use `SymbolicRegressionProposer`. IRT, location, and survival need latent structure, so their proposer generates NumPyro Python.

### Generated code

IRT, location, and survival require `--allow-code-execution` in non-smoke runs. **This executes arbitrary generated Python with process privileges, not in a security sandbox.** Use it only in a suitably isolated environment with trusted or reviewed proposals. The AST symbolic backend does not require this flag.

## Apprentice comparison

`--agent apprentice` keeps one evolving model, lets the LLM select experiments, and runs a residual-conditioned refinement on every round, fitted by the same SMC as the MDA. It controls its own design and refinement policy, so it rejects `--design` other than `model` and `--no-m-open`.

## Differences from v5

v5 runs Opus 4.7 with reasoning off, drops best-so-far curves and the curriculum example, and redescribes the MIRT illustration (a fixed five-model ladder, Brier-risk design, scoring over all 192 cells). The runner keeps v4's setup, and v5's MIRT illustration is not implemented.
