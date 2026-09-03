"""Run the released NeuronBench pure-LLM baseline through OpenAI."""

from __future__ import annotations

import argparse
import json

from mda.openai_client import OpenAIJSONClient


def run_baseline(
    *,
    world_name: str,
    model: str = "gpt-5.6-luna",
    stochastic: bool = False,
    budget: int = 3,
    seed: int = 0,
) -> dict:
    try:
        import neuronbench as nb
    except ImportError as error:
        raise RuntimeError("install model-discovery-agent[neuron]") from error
    world = nb.load_world(world_name, stochastic=stochastic, n_channels=100, seed=seed)
    return nb.agent.run_baseline(world, OpenAIJSONClient(model=model), budget=budget, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="ca_rebound")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_baseline(
        world_name=args.world,
        model=args.model,
        stochastic=args.stochastic,
        budget=args.budget,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
