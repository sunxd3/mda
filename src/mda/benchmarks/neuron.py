"""Run the released NeuronBench pure-LLM baseline through OpenAI."""

from __future__ import annotations

import argparse
import json
from typing import Any

from mda.agent import AgentConfig, ModelDiscoveryAgent
from mda.evidence import RecursiveEvidence, TemperedSMCEvidence
from mda.likelihoods import BootstrapParticleFilter, DeterministicGaussianLikelihood
from mda.neuron_stochastic import (
    StochasticNeuronBenchEnvironment,
    stochastic_neuron_candidate,
    stochastic_trace_features,
)
from mda.neuronbench_adapter import (
    NeuronBenchEnvironment,
    NeuronBenchProposer,
    reference_neuron_candidate,
    trace_features,
)
from mda.openai_client import OpenAIJSONClient
from mda.smc import SMCConfig


class _StochasticNeuronProposer:
    """Compile the structured hypotheses emitted by the shared proposer as PF models."""

    def __init__(self, proposer, *, channel_count: int, prediction_rollouts: int, seed: int):
        self.proposer = proposer
        self.channel_count = channel_count
        self.prediction_rollouts = prediction_rollouts
        self.seed = seed

    def propose(self, **kwargs):
        deterministic = self.proposer.propose(**kwargs)
        return [
            stochastic_neuron_candidate(
                model.metadata.get("neuron_hypothesis"),
                channel_count=self.channel_count,
                prediction_rollouts=self.prediction_rollouts,
                seed=self.seed,
            )
            for model in deterministic
        ]


def make_agent(
    environment: NeuronBenchEnvironment | StochasticNeuronBenchEnvironment,
    client: Any,
    *,
    seed: int = 0,
    smoke: bool = False,
    experiment_budget: int = 6,
    online_expansion: bool = False,
) -> ModelDiscoveryAgent:
    """Construct the NeuronBench reproduction agent.

    The stochastic environment follows v5's SDE+SMC arm: a 128-particle
    bootstrap PF, a 128-particle tempered-SMC fit with 16 initialization rungs,
    then recursive weight/evidence updates without rejuvenation, an archive of
    the plain Na/K/leak cell plus three LLM proposals, and ``p(m) ~ exp(-2k)``.
    As in v5's headline cohort, later model births are disabled, so the LLM
    only builds the initial archive. ``online_expansion=True`` re-enables
    v4's residual-triggered expansion; it is not v5's stochastic rank-test gate.
    The deterministic environment keeps the v4 (Appendix H.4) configuration,
    including online expansion.
    Smoke mode preserves the same model and data contracts while sharply
    reducing numerical work; it is not suitable for benchmark reporting.
    """
    if client is None and not smoke:
        raise ValueError("client=None is permitted only when smoke=True")
    stochastic = isinstance(environment, StochasticNeuronBenchEnvironment)
    if online_expansion and not stochastic:
        raise ValueError("online_expansion applies only to the stochastic environment")
    feature_fn = stochastic_trace_features if stochastic else trace_features
    proposal = None if client is None else NeuronBenchProposer(client, feature_fn=feature_fn)
    prediction_rollouts = 2 if smoke else 200
    if stochastic:
        proposer = (
            None
            if proposal is None
            else _StochasticNeuronProposer(
                proposal,
                channel_count=environment.channel_count,
                prediction_rollouts=prediction_rollouts,
                seed=seed,
            )
        )
        baseline = stochastic_neuron_candidate(
            None,
            channel_count=environment.channel_count,
            prediction_rollouts=prediction_rollouts,
            seed=seed,
        )
        likelihood = BootstrapParticleFilter(3 if smoke else 128, 2.0)
        observation_covariance = 2.0**2
        initial_proposals = 3
    else:
        proposer = proposal
        baseline = reference_neuron_candidate()
        likelihood = DeterministicGaussianLikelihood(environment.observation_std)
        observation_covariance = 3.0**2
        initial_proposals = 4

    config = AgentConfig(
        model_cap=5,
        new_models_per_expansion=1,
        model_search_rounds=2,
        expansion_cap=0 if stochastic and not online_expansion else 3,
        # v5 runs 2+8 experiments on a 90-action menu; the released pool has
        # only seven unused protocols, so repeats begin once it is exhausted.
        repeat_when_exhausted=stochastic,
        error_threshold=0.18,
        occam_penalty=2.0,
        predictive_floor=2.0 if stochastic else 3.0,
        smc=SMCConfig(
            parameter_particles=2 if smoke else (128 if stochastic else 200),
            rejuvenation_steps=0 if smoke else 3,
            target_ess_fraction=0.5,
            max_temperatures=3 if smoke else (16 if stochastic else 80),
        ),
    )
    evidence = RecursiveEvidence(TemperedSMCEvidence(config.smc)) if stochastic else None
    queries = environment.query_actions[:1] if smoke else environment.query_actions
    models = [baseline]
    if proposer is not None:
        models.extend(proposer.propose(
            context=f"{environment.context}\nExperiment budget remaining: {experiment_budget}.\nCurrent phase: explore.",
            dataset=environment.initial_data, current_models=models, residuals=(),
            count=initial_proposals,
        ))
    return ModelDiscoveryAgent(
        context=environment.context,
        initial_data=environment.initial_data,
        initial_models=models,
        proposer=proposer,
        likelihood=likelihood,
        actions=environment.actions,
        observation_covariance=observation_covariance,
        config=config,
        target=feature_fn,
        query_actions=queries,
        experiment_budget=experiment_budget,
        action_key=environment.action_key,
        evidence=evidence,
        seed=seed,
    )


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
