# ChemBench

ChemBench asks the agent to discover an enzyme-kinetics rate law from a handful of experiments. It is one of v5's three main domains. The runner is [`mda.benchmarks.chemistry`](../../src/mda/benchmarks/chemistry.py).

## Setup

ChemBench's oracle comes from an unpackaged source checkout (NumPy oracle only; no PySR):

```bash
git clone https://github.com/scientific-discovery/LLM-AutoSciLab.git /tmp/LLM-AutoSciLab
git -C /tmp/LLM-AutoSciLab checkout 9a37255ff5432a7155b6cde8985e9654f2f8f272
export AUTOSCILAB_SOURCE_ROOT=/tmp/LLM-AutoSciLab
```

## Run

```bash
# Offline, the nine canonical laws only.
mda-reproduce chemistry --world c0_michaelis_menten --smoke --budget 1 --output /tmp/chem.json

# v5's primary arm (paid): BIC, nine seeds + 12 LLM laws, 22 designed experiments.
mda-reproduce chemistry --world c0_michaelis_menten --live --budget 22 --output /tmp/chem-live.json

# v5's controls.
mda-reproduce chemistry --world c0_michaelis_menten --live --budget 22 --inference smc --output /tmp/chem-smc.json
mda-reproduce chemistry --world c0_michaelis_menten --live --budget 22 --no-m-open --output /tmp/chem-fixed.json
mda-reproduce chemistry --world c0_michaelis_menten --live --budget 22 --design random --output /tmp/chem-random.json
```

v5's universe is exactly the pinned registry's 57 standard worlds (`chemistry.benchmark_worlds()`), at medium difficulty (`--difficulty`, default `medium`) with three seeds each.

## Environment

- **Inputs**: seven experimental variables, `C_A`, `C_I`, `C_B`, `C_P`, `Enz`, `T`, and `pH`, within the upstream ranges.
- **D0**: three experiments at `C_A ∈ {0.03, 0.3, 3}` with the other variables at `(C_I, C_B, C_P, Enz, T, pH) = (0, 1, 0, 1, 310, 7)` (v5 Appendix B).
- **Design**: continuous; CMA-ES with 48 evaluations per experiment over the full box. `C_A`, `C_B`, and `Enz` are log-mapped.
- **Scoring**: nMSLE and RMSLE against the clean primary law over a seed-dependent set of 1000 query conditions, and the benchmark pass flag (`metrics.passed`, RMSLE ≤ 0.05).

## Agent

| Setting | Value |
|---|---|
| Initial pool | The nine canonical seed laws (upstream c0–c9 without the excluded pH law c4) plus one 12-law LLM batch |
| Evidence | Profiled BIC in log-rate space, `σ_n = max(0.03·\|y_n\|, 0.01)` on `log1p(rate)`, 8 multistarts |
| Model prior | Uniform (no parameter-count prior) |
| Pool cap | 24, shrinking to 6 when the MAP weight exceeds 0.9 and its median relative residual is at most 0.05 |
| Expansion | Six proposals per call, two search rounds per update, at most four calls (v4's values, which v5 keeps implicitly) |
| Expansion gate | `cached`: the first expansion of an update is gated on the preceding fit's in-sample residuals, as v5's executed agent does |
| Prediction | MAP |
| SMC control (`--inference smc`) | Tempered SMC with 100 particles, three rejuvenation moves, ESS target 0.6, and a 2-nat-per-parameter prior |

The LLM proposer is `ChemBenchProposer`, which sends v5's published ChemBench prompt (`prompts/chem_proposer.txt` in the arXiv source): the enzyme-kinetics framing with input units, the experiments, and, when refining, the forms already tried with their median residuals and the inputs the best form's residual correlates with. It drops laws that use functions other than exp, log, and sqrt, or `%`. It repairs two common slips instead of dropping the law: a variable spelled `CA`, `CB`, `CI`, `CP`, `E`, or `Enzyme` (when that name is not a declared parameter) is renamed, and a lower bound of 0 or below is raised to 10⁻⁶ of a positive upper bound. Laws with a non-positive upper bound are still dropped. Smoke mode uses 20 CMA-ES evaluations, two BIC starts, and 12 queries.

## Differences from the paper

- v5's exact per-call responses and configuration files are not public.
- The prompt template leaves some details open. Here the "current best form" whose residual correlations are reported is the lowest-residual form, because the proposer does not see model weights, and the two inputs with the largest absolute correlation are named. The budget/phase line is the agent's own status text.
- The canonical laws' forms follow the upstream rate functions; their bounds, the number of multistarts, and v4's τ_e, R_m, and N_e^max values are local choices.
- Rendered v5 gives no query count and says the seed changes the queries. Its source uses the benchmark's fixed 1000-condition grid for the pass rate and a separate 60-condition draw for nMSLE; this runner uses one seed-dependent 1000-point set for both.
- The expansion and shrink checks use raw-rate relative residuals rather than v5's `log(1+y)` target.
- CMA-ES maps `C_I` and `C_P` linearly because their lower bound is 0, although v5 says concentration axes are logarithmic.
- Negative predictions are clipped to 0, whereas upstream scores `|ŷ|`.
- The fixed-scale BIC control is available as `BICEvidence("fixed")` but not as a CLI arm.
- A law that cannot be evaluated over the queries is recorded as a failure (`evaluable: false`, `passed: false`).
- Held-out references are the clean primary law, as in upstream `evaluate_law`. The upstream oracle's noise is input-hashed, not a new independent draw for each run seed.
- When a model interpolates the data (for example a three-parameter law on the three-point D0), profiled BIC floors its RSS at the smallest positive float, so it takes essentially all the weight until more data arrive. The other models stay in the pool with zero weight and recover once the fit is no longer exact. v5 does not say how it handles an exact fit.
- v5 does not say how it handles a proposed bound of 0 (common for activation energies). Repairing it is a local choice; dropping such laws instead discarded most expansion proposals in the `*_arrhenius` worlds.
- ICL (+LLM design) and the SciLab/PySR baseline are not implemented.
- `--recovery` adds v4's constant-refit structural recovery; see [Reproducing the paper](../reproduce.md#structural-recovery-v4-diagnostic).
