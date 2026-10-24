# ForceBench (v4 legacy)

ForceBench, from the DiscoverPhysics project, asks the agent to discover a force law from particle trajectories. v5 dropped this domain; its runners are kept here as v4 legacy. They are [`mda.benchmarks.physics`](../../src/mda/benchmarks/physics.py) and [`mda.benchmarks.physics_extra`](../../src/mda/benchmarks/physics_extra.py).

Worlds: the six central two-particle worlds `gravity`, `yukawa`, `coulomb`, `oscillator`, `fractional`, and `extra-dim`, plus the extensions `circle`, `ether`, and `hubble`.

## Setup

```bash
.venv/bin/pip install -e '.[physics]'

# PhysicsSchool's pinned wheel omits namespace subpackages, so it also needs an editable checkout.
git clone https://github.com/SampsonML/DiscoverPhysics.git /tmp/DiscoverPhysics
git -C /tmp/DiscoverPhysics checkout 33b7fa9df96de9c35744efd181ca7e5a8dd60ad5
.venv/bin/pip install -e /tmp/DiscoverPhysics/PhysicsSchool
```

## Run

```bash
mda-reproduce physics --world gravity --smoke --budget 1 --output /tmp/physics.json
```

### Continuous design and the expansion check

`--design-space continuous` replaces the 13-action menu with the continuous launch box of the v5-era physics writeup (its §A.2), which is DiscoverPhysics' documented control range: probe position in [−10, 10]², velocity in [−5, 5]², and p1, p2 in [0.1, 10] (searched on log scales). Launches closer than 0.5 to the source are moved out to radius 0.5 along their direction. VoI is maximized by differential evolution with 144 evaluations (SciPy, vectorized with deferred updates, no polishing). Observation times stay at {0.5, 1, 1.5, 2, 3, 4}.

`--check` selects the M-open expansion trigger:

- `v4` (default): v4's predictive check. Expansion needs the median relative residual over the fresh and all stored trajectories to exceed τₑ = 1.4, with the first expansion gated on the residuals of the preceding fit, as v5 says the executed DP agent did. After a few experiments one surprising trajectory cannot move this median, so the pool rarely or never expands.
- `chi2`: expand when the fresh trajectory's whitened squared error under the pre-update MAP fit exceeds the 99% quantile of χ² with one degree of freedom per coordinate (σ = 0.05). Noise alone triggers it 1% of the time; after each expansion the same statistic re-judges the refitted pool on that trajectory.
- `guided`: the `chi2` test, but at most one expansion per experiment (up to 13), so the budget lasts the whole run. Expansion prompts also show the current best law with its fitted constants, its misfit on every experiment in units of σ, and the latest trajectory observed and predicted, and they ask for structurally different laws instead of corrections.

`--max-params N` drops proposed laws with more than N free parameters, including from the initial pool. v4 states that ForceBench candidates have 1–3.

The proposer uses v4's published DiscoverPhysics prompt (`dp_proposer.txt`): the world's knob roles, each run's launch and probe radius r(t), and the forms already in the pool. A failed or non-finite prediction scores nMSE 10³, as in the writeup, instead of aborting the run.

```bash
mda-reproduce physics --world yukawa --live --provider codex --budget 13 \
  --design-space continuous --check chi2 --output /tmp/yukawa.json
```

## Notes

- The extensions use documented local launch menus because the exact v4 menus are unreleased. Their background-coefficient bounds [−1, 1] and launch templates are local choices.
- Ether follows the public simulator's common drift of the anchor and probes, rather than interpreting the appendix's anchor as immovable.
- Dark-matter and three-species inference and the exact external DP-Agent comparison harness are not implemented.
- `--recovery` supports the six central two-particle worlds, not the extensions; see [Reproducing the paper](../reproduce.md#structural-recovery-v4-diagnostic).
