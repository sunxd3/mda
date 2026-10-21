# Replication

We have run two of the paper's main benchmarks end to end with live LLM calls. Both match the paper. NeuronBench and BoxingGym have not been swept.

- [ChemBench](#chembench): 78.9% pass rate at 25 observations (paper 78.4%).
- [GlucoseBench](#glucosebench): CGM nMSE 0.25 after six episodes (paper 0.316), with Codex as the LLM.

## ChemBench

We ran the paper's main ChemBench experiment end to end: all 57 worlds × 3 seeds (171 cells), 22 adaptive experiments each, GPT-5.6 Luna at medium reasoning effort. After fixing two bugs in this repository, the result matches the paper.

![ChemBench before and after the fixes, with the paper's values](images/chembench-replication.png)

### Result

| Observations | Before fixes | After fixes | Paper |
|---|---|---|---|
| 10 | 37.4% | 41.5% | — |
| 15 | 52.6% | 64.3% | — |
| 20 | 53.8% | **77.2%** | 74.9% |
| 25 | 55.6% | **78.9%** (135/171) | 78.4% |
| nMSLE at 25 | 0.0073 | **0.0016** | ≈0.0010 |

- Pass rate at 25 observations has a 95% interval of 70.8–86.5% (world bootstrap). The paper's 78.4% is inside it.
- In-library worlds pass 27/27, as in the paper. Out-of-library worlds pass 108/144.
- Paired against the run before the fixes, the pass rate rose 23.4 points (95% CI +14.6 to +31.6) and nMSLE fell to 0.31× (0.19–0.49).
- nMSLE is still about 1.6× the paper's value read off its figure.

### Protocol

- **Code:** `main` at [`017f23b`](https://github.com/sunxd3/mda/commit/017f23beac114093ba95b8c65a0455b7831d59bd); upstream LLM-AutoSciLab at `9a37255`.
- **Command, per cell:** `mda-reproduce chemistry --world WORLD --seed SEED --live --budget 22`, with an empty response cache.
- **Pass:** RMSLE < 0.05 on upstream `evaluate_law`'s fixed 1000-point grid, as in v5. Every scored snapshot was re-evaluated with upstream code and checked against the run's stored RMSLE (855 of 855 matched).
- **nMSLE:** averaged over seeds within a world, then geometric mean over worlds.
- **Runs:** all 171 completed 22 of 22. Four first attempts failed with transient OpenAI 503 errors and passed on retry.
- **Cost:** about 2 cents of API calls per cell, so a few dollars for the sweep. Each cell takes 1–50 minutes of CPU, mostly BIC fitting.

### What closed the gap

The first full sweeps passed 55–60%. Swapping in the paper's published prompt did not help (−4.7 points, not significant). A fixed nine-law control with no LLM passed 31.6% at 15 observations, matching the paper's fixed arm (about 32%), so scoring, fitting, and experiment design were not the problem. Refitting the true law to each failing run's own data passed in 138 of 140 failures: the data were good enough, but the right law was never proposed.

Two bugs starved the LLM search ([PR #8](https://github.com/sunxd3/mda/pull/8)):

1. **Valid laws were thrown away.** Any law with a lower parameter bound of 0 was dropped. The LLM routinely writes activation energies as `Ea ∈ [0, …]`, so 65% of expansion proposals were discarded, mostly in the `*_arrhenius` worlds. Such bounds are now raised to 10⁻⁶ of the upper bound.
2. **Repeated prompts got the same answer.** When an expansion added nothing, the next prompt was identical, and the response cache returned the same reply. The search stalled after a few rounds. A repeated request is now a fresh draw.

A third bug, an output-token cap that truncated long reasoning and aborted runs, was fixed earlier ([PR #7](https://github.com/sunxd3/mda/pull/7)).

### What still fails

36 cells fail at 25 observations:

- **Allosteric activation** (`c67`, `c78`, `c92`): 0 of 9 pass, with RMSLE 0.3–1.2. The search never finds this form.
- **Near misses:** 16 cells with RMSLE between 0.05 and 0.1, spread across families; sinh worlds are the largest group (5).
- **Wrong laws:** the other 11, such as `c61_sinh_product_arrhenius` (all three seeds) and two seeds of `c39_hill_competitive_arrhenius`.

### Limits

- One sweep, one seed of LLM randomness per cell. A rerun will differ by a few cells.
- The paper does not say how it handled 0 bounds; repairing them is our choice.
- Other known differences are listed under [ChemBench](benchmarks/chembench.md#differences-from-the-paper).

## GlucoseBench

We ran GlucoseBench for all 30 virtual patients, one run each as in v5: two initial episodes, four selected experiments, and forecasts of the 12 sealed interventions after a 45-minute prefix. The LLM was GPT-5.6 Luna at medium effort through Codex (`--provider codex`), not the direct API. The result matches the paper's MDA curve.

![GlucoseBench CGM error against the paper](images/glucosebench-replication.png)

### Result

Patient-level geometric means of training-variance-normalized MSE, pooled over observations after the cut, with 95% patient-bootstrap intervals:

| Training episodes | CGM (ours) | CGM (v5 MDA) | Insulin (ours) | Insulin (v5 MDA) |
|---|---|---|---|---|
| 2 | 1.12 [0.71, 1.80] | ≈1.15 | 0.20 [0.16, 0.25] | — |
| 3 | 0.33 [0.24, 0.45] | ≈0.35 | 0.19 [0.15, 0.24] | — |
| 4 | 0.29 [0.21, 0.40] | ≈0.34 | 0.19 [0.15, 0.24] | — |
| 6 | **0.25** [0.19, 0.33] | **0.316** | **0.19** [0.15, 0.23] | **0.216** |

- The v5 values at six episodes are stated in the paper's source; the others are read off its figure.
- Every v5 value lies inside our interval. Ours run slightly lower, which fits scoring all 360 patient–query pairs: v5 reports 0.304 on the 359 pairs valid at this cut and 0.316 on the 273 valid for every method and cut.
- MDA's CGM error is about 5× lower than v5's best baseline (SINDYc, 1.40) and 24× lower than the persistence baseline (5.97). We did not run SINDYc, ARX, VARX or ICL, so the paper's finding that SINDYc predicts insulin better (0.151) is untested here.
- No forecast failed. 27 of 30 runs added LLM-proposed models; children needed the most (3.2 expansions on average, against 1.5 for adolescents and 1.3 for adults).

### Protocol

- **Code:** `main` at [`c6334e4`](https://github.com/sunxd3/mda/commit/c6334e4c13823b68847c1b34d3c40f361a7628c9); `glucosebench` at `6bb0ff6`.
- **Command, per patient:** `mda-reproduce glucose --world PATIENT --live --budget 4 --provider codex`, with an empty response cache.
- **Codex:** the client switches off Codex's own instructions, tools and environment text, so each call sends only the benchmark's system prompt, the data, and the output schema ([coding-agent clients](library.md#coding-agent-clients)).
- **Runs:** all 30 completed 4 of 4 experiments on the first attempt. Runs take 6–40 minutes of CPU, mostly fitting; Codex is called 0–5 times per patient.

### Limits

- One run per patient, so the interval reflects variation across patients, not LLM randomness.
- Codex wraps the request differently from the direct API (the system prompt is a developer message), so this is not the paper's exact client.
- Only the 45-minute cut was compared; v5 also reports cuts at 0, 120 and 180 minutes. Other differences are listed under [GlucoseBench](benchmarks/glucosebench.md).
