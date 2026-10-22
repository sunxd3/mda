# Paper fidelity

This page records how the implementation follows [arXiv:2608.09696v5](https://arxiv.org/abs/2608.09696v5), where it makes choices the paper leaves open, and what still differs. Domain-specific differences are on each [benchmark page](reproduce.md#benchmarks).

- [Algorithm map](#algorithm-map)
- [Defaults](#defaults)
- [Agent loop details](#agent-loop-details)
- [Task VoI (a v4 extension)](#task-voi-a-v4-extension)
- [Experiment programs (beyond v5)](#experiment-programs-beyond-v5)
- [Pseudocode corrections](#pseudocode-corrections)
- [Choices v5 leaves open](#choices-v5-leaves-open)
- [What changed in v5](#what-changed-in-v5)
- [What still prevents an exact paper reproduction](#what-still-prevents-an-exact-paper-reproduction)

## Algorithm map

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
| Beyond v5: experiment programs | `mda.programs`, `mda.program_sandbox`, `mda.program_proposer` |

## Defaults

Generic defaults use v5's default evidence method, tempered SMC, with v4 Table 2's ChemBench settings (100 particles, three rejuvenation moves, ESS target 0.6, 80 rungs), since v5 gives no values. They also use v5's default uniform model prior (`occam_penalty=0`), with v4's 16-model pool, four proposals per expansion, two search rounds, and 0.05 error threshold.

Domain runners set their own values, as the paper does:

| Domain | Evidence | Pool | Model prior |
|---|---|---|---|
| ChemBench | Profiled BIC | Cap 24 (shrinks to 6), six proposals per expansion, 12 LLM laws at initialization | Uniform |
| GlucoseBench | AR(1) working likelihood with BIC | Cap 14, at most two proposals per call, fitted equations and predictions shown to the proposer | Uniform |
| Stochastic neuron | Recursive SMC | Archive built once and then frozen | `exp(−2k)` |
| Boxing (v4) | Tempered SMC | Cap 16, four proposals per expansion | `exp(−k)` |

## Agent loop details

**Model revision** follows v5's narrower account. The LLM is used only by the proposer: once at initialization (`initial_proposals`) and in `expand-hyp-space`. Proposals are conditioned on per-model residual reports, which v5 treats as optional context "inspired by" SMC-S. v5 calls top-N pruning deterministic hypothesis search rather than a posterior-invariant resampling kernel, so model weights are a posterior over the proposed pool only and assign nothing to models never proposed.

- Every proposer call counts toward `expansion_cap` (N_e^max), even one that returns only invalid or duplicate models.
- A proposed law is screened at a few fixed prior draws and kept if any draw evaluates on the observed experiments. Symbolic structures are stored in canonical `ast.unparse` form, so spacing or parenthesis variants of one law are duplicates.
- A model whose weight underflows to zero stays in the pool and can regain weight; only structures whose fit fails entirely leave it.
- A design that the current best model cannot evaluate at all (a domain error at the chosen input) counts as a failed predictive check rather than aborting after the experiment has run. If the refit on the new datum fails, the agent is left exactly as it was.

**Adapt and prune.** Optional shrinking is enabled with `minimum_model_cap`. After an update it requires both a well-fit MAP model (median relative residual at most `error_threshold`) and a MAP weight above `shrink_confidence` (default 0.9, v5's ChemBench value); otherwise the cap is `model_cap`. It never runs at initialization. A failed check or reduced confidence restores the larger cap; evicted models are not automatically resurrected. Shrinking is disabled by default, matching the base Algorithm 2. Two details differ from Algorithm 2:

- a failed check restores the cap before that update's expansion rounds rather than at the end, and
- shrinking renormalizes the retained weights instead of refitting (the same for deterministic evidence).

**Expansion gate.** `expansion_gate="cached"` reproduces v5's executed ChemBench agent, which gates its first expansion on the in-sample residuals of the preceding fit rather than on the prequential check. Later rounds recheck the refitted pool in both modes.

**Diagnostics.** `BeliefState.log_mixture_evidence` is computed before pruning over the deduplicated input pool, with its parameter-count model prior normalized over that pool. v5 stores this normalizer after pruning; here it is a diagnostic only. `discarded_mass` includes support truncation. Each parameter posterior records `ess_before_resampling` and `last_rung_forced_below_target`: reaching the rung limit forces the full likelihood (β = 1), possibly at the cost of low ESS. Increase the rung or particle budget if these diagnostics show collapse.

## Task VoI (a v4 extension)

v5 keeps only the general decision-theoretic task VoI (Eq. taskvoigeneral) and leaves it to future work. The two task flavours below implement v4's Eq. 42–45 and are not part of the v5 method. Both need `query_actions`, and both have explicit target-space units.

`design_flavour="task"` uses v4 Eq. 43's target predictions at **both** the proposed and query actions, with each target coordinate scored independently. It averages across query actions and sums coordinate contributions, whitening the loss by the target-space noise variances.

- `task_noise_covariance` supplies those variances to the agent; when omitted it falls back to `observation_covariance`. Use a separate target-space value for features derived from raw traces.
- `target_covariance` overrides the loss whitening (set it to 1 for unwhitened squared loss).
- Both must be scalar or diagonal for this flavour.

`design_flavour="task_full_observation"` (or `full_observation_task_value_of_information`) is the older dense-matrix conditional-variance calculation. It uses raw observations at the proposed action and all cross-coordinate information; optional `target_covariance` whitens its loss. This is a coherent Gaussian alternative inspired by v4 Eq. 42, but it is **not** the coordinatewise implementation described after v4 Eq. 45.

Neither objective is exact information gain for a general non-Gaussian mixture.

## Experiment programs (beyond v5)

v5 designs one action at a time. `design_flavour="program"` instead designs a whole episode whose inputs adapt to the observations within it. An LLM writes the programs (`ProgramProposer`), but they are scored numerically. The score is a nested-Monte-Carlo estimate of I(M; Y₁:T | D, program), with closed-loop simulation against a kernel-smoothed version of the particle belief and mda's own likelihood as the inner term. See [Library: experiment programs](library.md#experiment-programs-beyond-v5). The idea is from `expprog` (sunxd3/model-agent).

The completed episode is stored as `Datum(realised_inputs, trajectory)`, so the model side of Algorithm 2 is untouched. The trajectory likelihood is ignorable for a deterministic program, or a program whose randomness does not depend on the model: the inputs carry no information about the model beyond the observations they were chosen from. The predictive check still compares the trajectory with each model's mean trajectory at the realised inputs. That is valid without per-episode latent variables, which this extension does not support.

Known limits:

- The estimand is the information under the *smoothed* belief, which is wider than the particle belief by a Silverman-rule bandwidth. Smoothing is truncated to the prior's support but does not reweight by the likelihood, so it can put mass where the data rule a model out.
- Outer and inner draws are independent, so, in expectation, the estimate is a Barber–Agakov lower bound. The defensive weight ε = 10⁻³ (the inner model posterior mixed with the model weights) caps the loss from an episode that no inner draw explains at log₂(1/ε) ≈ 10 bits. The remaining gap shrinks as `program_inner_particles` grows. In `mda.benchmarks.episodic`, two indistinguishable models score between −0.03 and +0.01 bits (SD up to 0.07, over 10 scorer seeds) at 40, 100 and 200 SMC particles per model. Narrow likelihoods in higher-dimensional parameter spaces need more inner draws.
- An earlier version used the SMC particles themselves as the inner set and excluded the outer particle by value. With sparse particles, that left no own-model value explaining a sharp trajectory, and single draws scored hundreds of negative bits.

## Pseudocode corrections

This repository corrected six inconsistencies in v4's pseudocode. v5 now states each of them itself:

1. the pre-observation check runs before the append;
2. top-N pruning renormalizes the retained weights;
3. out-of-bounds random-walk proposals are rejected (v5 notes that the authors' NumPy fitter clipped instead, which can bias its legacy results);
4. the trace surrogate is not exact mutual information;
5. checks after expansion are in-sample; and
6. `get_model()` returns the current-data MAP rather than a maximum over rounds.

Held-out truth is never used to select the agent's model; best-so-far curves are a separate reporting convention.

## Choices v5 leaves open

- After appending a datum the pool is refitted, so proposals see residuals of the refitted pool (v5 Algorithm 2). The re-judge after an expansion fits only the new structures and reuses the others' fits on the identical dataset, rather than rerunning their SMC.
- Profiled BIC needs more observations than parameters, and v5 does not say how it handles an exact fit. When a model interpolates the data (for example three-parameter laws on ChemBench's three-point D0), its RSS is floored at the smallest positive float, so it takes essentially all the weight until more data arrive. The other models stay in the pool with zero weight and recover once the fit is no longer exact.
- v5 does not give the number of BIC multistarts (8 here) or its CMA-ES population size and initial step (cma's defaults here). It states a proposal-scale floor of 0.5 × max(std, 1e-3) only for its legacy NumPy fitter; this one uses a relative floor.
- CMA-ES always spends its full evaluation budget: cma's tolerance stops are disabled, so a nearly flat score (a concentrated posterior) cannot end the search after one generation, and any remaining stop restarts from a random point.
- Finite-menu design breaks ties by menu order (GlucoseBench breaks them randomly, as v5 specifies there). Used designs are excluded unless `repeat_designs` or `repeat_when_exhausted` is set. A design at which a weighted model cannot predict scores −∞ rather than aborting.
- `BayesianSyntheticLikelihood` simulates through the same Gaussian-emission state-space callbacks as the PF, so it needs `state_space` and a fixed observation SD, not an arbitrary simulator. Its defaults of no shrinkage and ridge 1e-6 are local choices.
- `RecursiveEvidence` keeps each structure's particles while the data only grow, including after the archive changes; v5's HH implementation refits when the archive changes. A pruned structure that is proposed again resumes from its cache.
- A PF particle whose state diverges (NaN or infinite) gets zero weight rather than voiding the likelihood estimate.

The bootstrap PF gives an unbiased **likelihood** estimate. Its logarithm is not unbiased, and fractional likelihood tempering is only approximately pseudo-marginal, as the paper notes in its tempered-SMC appendix.

## What changed in v5

v5 restructured the paper: its main domains are now ChemBench, the new GlucoseBench, and the stochastic neuron benchmark (NeuronBench; its LaTeX macro is `\HHbench`). BoxingGym moved to an appendix, and the ForceBench/DiscoverPhysics domain was dropped.

| v5 change | Status here |
|---|---|
| Algorithm 2 checks before the append, refits before searching, returns the current-data MAP, and has an optional `adapt-and-prune` step | Implemented; shrinkage no longer runs at initialization |
| New evidence backends: recursive updates (`marglik-pf`) and BIC (`marglik-bic`); BSL, described in v4, becomes the plug-in `loglik-bsl` | Implemented (`mda.evidence`, `BayesianSyntheticLikelihood`) |
| Default model prior is uniform (λ = 0) | `AgentConfig.occam_penalty` now defaults to 0 |
| ChemBench: BIC primary arm, nine canonical seeds plus 12 LLM laws, caps 24/6 at weight > 0.9, six proposals per expansion, 48 CMA-ES evaluations, 3+22 budget, medium difficulty, all 57 worlds, pass rate | Implemented; the SMC control (`--inference smc`) and fixed nine-law control (`--no-m-open`) are included |
| LLM: GPT-5.6 Luna at medium effort for main domains | Default provider and effort updated; replay works for both providers |
| Stochastic neuron: brief/weak D0, fixed-scale metric, recursive SMC with N_p = N_z = 128, archive frozen after LLM initialization, 2+8 budget | Implemented; `--online-expansion` restores later births; protocols repeat once the released pool is exhausted; v5's 90-protocol harness is not in the pinned release |
| Model revision: a separate `init-hyp-space`; pool residuals are optional proposer context; top-N pruning is deterministic search over the proposed pool, not an SMC kernel | Implemented (`initial_proposals`, per-domain proposers) |
| GlucoseBench domain | MDA arm and ICL (+MDA design) implemented, plus the package's persistence baseline; ARX, VARX, SINDYc, and ICL (+LLM design) are not |
| ForceBench removed; task VoI and structural recovery moved to future work or removed | Kept and labeled as v4 legacy |

## What still prevents an exact paper reproduction

- **No released responses.** The paper's per-call LLM responses, selected histories, and configuration files are not public, so its scores cannot be replayed. Caching enables replay of calls captured here, not of the paper's.
- **Unexecuted runs.** ChemBench and GlucoseBench have been swept with live LLM calls and match the paper ([Replication](replication.md)). NeuronBench and BoxingGym have no live sweep or score claim.
- **ChemBench.** A different query set for the pass rate and nMSLE, raw-rate check residuals, and linear `C_I`/`C_P` axes, among others. ICL (+LLM design) and the SciLab/PySR baseline are not implemented. See [ChemBench](benchmarks/chembench.md#differences-from-the-paper).
- **GlucoseBench.** The ARX, VARX, and SINDYc baselines and ICL (+LLM design) are not implemented; failed cells are counted rather than restricted to v5's valid subset. See [GlucoseBench](benchmarks/glucosebench.md#not-implemented).
- **NeuronBench.** v5's 90-protocol, 20-sample, 24-query harness is not in the pinned release, the channel parameterization differs, and v5's other inference arms are not wired into the runner. See [NeuronBench](benchmarks/neuronbench.md#differences-from-v5).
- **Boxing.** Reconstructed equations with private RNGs instead of the original PyMC execution, v4's setup instead of v5's appendix changes, and no MIRT illustration. See [BoxingGym](benchmarks/boxing.md).
- **ForceBench (v4 only).** Dark-matter and three-species inference and the DP-Agent comparison harness are not implemented. See [ForceBench](benchmarks/forcebench.md).
- **Recovery** covers chemistry, the six central-force worlds, and coarse neuron channel classes, not the many-body extensions.
- Learned summaries, amortized SBI, and neural ratio estimation are future work in v5, not missing algorithms.
