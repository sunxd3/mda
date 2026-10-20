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

## Notes

- The extensions use documented local launch menus because the exact v4 menus are unreleased. Their background-coefficient bounds [−1, 1] and launch templates are local choices.
- Ether follows the public simulator's common drift of the anchor and probes, rather than interpreting the appendix's anchor as immovable.
- Dark-matter and three-species inference and the exact external DP-Agent comparison harness are not implemented.
- `--recovery` supports the six central two-particle worlds, not the extensions; see [Reproducing the paper](../reproduce.md#structural-recovery-v4-diagnostic).
