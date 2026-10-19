"""Single-run reproduction driver; no implicit benchmark sweep or paid calls."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np


def build(domain, world, client, *, seed, smoke, budget, stochastic=False,
          difficulty="medium", allow_code_execution=False, method="mda",
          inference=None, llm_initialization=True):
    if method not in {"mda", "apprentice"} or (method == "apprentice" and domain != "boxing"):
        raise ValueError("apprentice is supported only for Boxing")
    if domain == "boxing":
        from .boxing import BoxingEnvironment, make_agent
        if method == "apprentice":
            from .apprentice import make_agent
        environment = BoxingEnvironment(world, seed=seed)
        extra = {"allow_code_execution": allow_code_execution}
    elif domain == "chemistry":
        from .chemistry import ChemBenchEnvironment, make_agent
        environment = ChemBenchEnvironment(world, difficulty, seed, query_count=12 if smoke else 1000)
        extra = {"inference": inference or "bic", "llm_initialization": llm_initialization}
    elif domain == "physics":
        from .physics import PhysicsEnvironment, make_agent
        if world in {"circle", "ether", "hubble"}:
            from .physics_extra import PhysicsExtraEnvironment, make_agent
            environment = PhysicsExtraEnvironment(world, seed)
        else:
            environment = PhysicsEnvironment(world, seed)
        extra = {}
    elif domain == "neuron":
        from .neuron import make_agent
        from mda.neuronbench_adapter import NeuronBenchEnvironment
        from mda.neuron_stochastic import StochasticNeuronBenchEnvironment
        factory = StochasticNeuronBenchEnvironment if stochastic else NeuronBenchEnvironment
        environment = factory(world, seed=seed)
        extra = {}
    else:
        raise ValueError(f"unknown domain {domain!r}")
    if inference is not None and domain != "chemistry":
        raise ValueError("--inference currently selects only the ChemBench arm")
    agent = make_agent(environment, client, seed=seed, smoke=smoke,
                       experiment_budget=budget, **extra)
    return environment, agent


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def run(environment, agent, *, budget, smoke=False, prediction="map",
        design="model", evaluate=True, no_m_open=False, icl_client=None, recovery=False):
    """Report every current-data fit; best-so-far is reporting only.

    Evaluation never participates in updates or selects a historical model.
    Smoke may omit expensive neuron ground-truth rollout aggregation while
    still checking the expected-feature predictor for finite outputs.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if agent.experiment_budget is not None and budget > agent.experiment_budget - agent.experiments_run:
        raise ValueError("run exceeds the agent's remaining experiment budget")
    agent.config = replace(agent.config, design_flavour="model" if design == "random" else design)
    if no_m_open:
        agent.proposer = None
    history = []
    best = float("inf")

    def snapshot(action=None, observation=None):
        nonlocal best
        predictor = agent.get_predictor(prediction)
        predictions = [np.asarray(predictor(a)) for a in agent.query_actions]
        if any(np.any(~np.isfinite(p)) for p in predictions):
            raise ValueError("non-finite query prediction")
        metrics = None
        if evaluate and hasattr(environment, "evaluation_metrics"):
            metrics = environment.evaluation_metrics(predictor)
            score = float(metrics["nmsle"])
        else:
            score = float(environment.evaluate_predictor(predictor)) if evaluate else None
        if score is not None:
            if not np.isfinite(score):
                raise ValueError("non-finite evaluation score")
            best = min(best, score)
        fit = agent.belief.map_fit
        history.append({
            "experiments": agent.experiments_run,
            "action": action, "observation": observation,
            "score": score, "best_score_so_far": best if score is not None else None,
            "map_model": fit.model.name, "structure": fit.model.structure,
            "map_parameters": fit.parameters.map_parameters,
            "model_weights": agent.belief.model_weights,
            "model_names": [f.model.name for f in agent.belief.fits],
            "log_evidences": [f.parameters.log_evidence for f in agent.belief.fits],
            "forced_tempering": [f.parameters.last_rung_forced_below_target for f in agent.belief.fits],
            "predictive_error": agent.last_predictive_error,
            "expansions": agent.expansion_count,
        })
        if metrics is not None:
            history[-1]["metrics"] = metrics
        if recovery:
            try:
                history[-1]["recovery"] = environment.evaluate_model(fit.model, fit.parameters.map_parameters)
            except ValueError as error:
                history[-1]["recovery"] = {"error": str(error), "recovered": False}
        if icl_client is not None:
            from .icl import make_icl_predictor
            features = (lambda a: environment.action_features[int(a)]) if getattr(
                environment, "domain", None) in {"irt", "location", "survival"} else (lambda a: a)
            # Shape, not numerical predictions, is the sole model-derived input.
            icl = make_icl_predictor(icl_client, environment.context, agent.dataset,
                environment.query_actions, target=agent.target, action_features=features,
                action_key=agent.action_key, target_shape=predictions[0].shape)
            history[-1]["icl_score"] = float(environment.evaluate_predictor(icl))

    snapshot()
    for _ in range(budget):
        if design == "random" and hasattr(agent.optimizer, "sample"):
            action = agent.optimizer.sample(agent.rng)  # uniform in the design box
        elif design == "random":
            choices = agent.available_actions()
            if not choices or agent.optimizer is not None:
                raise ValueError("random design requires a finite menu or a sampling optimizer")
            action = choices[int(agent.rng.integers(len(choices)))]
        else:
            action = agent.design()
        observation = environment.step(action)
        agent.update_state(action, observation)
        snapshot(action, observation)
    return {"smoke": smoke, "prediction": prediction, "config": asdict(agent.config),
            "agent_class": type(agent).__name__,
            "design": design, "no_m_open": no_m_open,
            "evaluation_performed": evaluate, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=["boxing", "chemistry", "physics", "neuron", "glucose"])
    parser.add_argument("--world", required=True)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--agent", choices=["mda", "apprentice"], default="mda")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="reduced numerics; no LLM")
    mode.add_argument("--replay", action="store_true", help="cached provider calls only")
    mode.add_argument("--live", action="store_true", help="authorize paid provider calls")
    parser.add_argument("--provider", choices=["anthropic", "openai"],
                        help="default: anthropic (Opus 4.7) for Boxing, openai (GPT-5.6 Luna) otherwise")
    parser.add_argument("--inference", choices=["bic", "smc"],
                        help="ChemBench arm: profiled-BIC primary (default) or the SMC control")
    parser.add_argument("--model")
    parser.add_argument("--cache-dir", default=".cache/mda-responses")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--allow-code-execution", action="store_true",
                        help="UNSAFE: execute generated NumPyro Python with process privileges")
    parser.add_argument("--design", choices=["model", "joint", "task", "random"], default="model")
    parser.add_argument("--prediction", choices=["map", "bma"],
                        help="default: bma for Boxing, map for other domains")
    parser.add_argument("--task-noise-variance", type=float, nargs="+")
    parser.add_argument("--no-m-open", action="store_true",
                        help="no expansion; for ChemBench also no LLM initialization (fixed nine laws)")
    parser.add_argument("--icl", action="store_true", help="also forecast using only the collected data")
    parser.add_argument("--recovery", action="store_true", help="symbolic-law or coarse neuron-class recovery diagnostic")
    parser.add_argument("--cuts", type=int, nargs="+", default=[45],
                        help="GlucoseBench forecast cuts in minutes (0, 45, 120, 180)")
    parser.add_argument("--replicate", type=int, default=0, help="GlucoseBench replicate (paper: 0)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("budget must be non-negative")
    if args.agent == "apprentice" and (args.domain != "boxing" or args.design != "model" or args.no_m_open):
        parser.error("Apprentice is Boxing-only and controls its own design/refinement policy")
    if args.output.exists():
        parser.error("output already exists; choose another path")
    if args.stochastic and args.domain != "neuron":
        parser.error("--stochastic applies only to neuron")
    if args.domain == "neuron" and args.design == "task" and args.task_noise_variance is None:
        parser.error("neuron task VoI requires --task-noise-variance in feature-space units")
    if args.task_noise_variance is not None and (
        any(not np.isfinite(v) or v <= 0 for v in args.task_noise_variance)
        or (args.domain == "neuron" and len(args.task_noise_variance) not in (1, 6))
    ):
        parser.error("task noise variances must be positive; neuron needs one or six values")
    if args.inference is not None and args.domain != "chemistry":
        parser.error("--inference applies only to chemistry")
    if args.domain == "glucose" and (
        args.budget > 4 or args.design not in {"model", "random"} or args.agent != "mda"
        or args.recovery or args.stochastic or args.prediction not in {None, "map", "bma"}
        or any(cut not in (0, 45, 120, 180) for cut in args.cuts)
    ):
        parser.error("glucose takes at most four experiments, model/random design, MDA, and cuts 0/45/120/180")
    if args.icl and args.smoke:
        parser.error("ICL needs a replay or explicitly authorized live client")
    if args.recovery and (args.domain not in {"chemistry", "physics", "neuron"}
                          or args.world in {"circle", "ether", "hubble"}):
        parser.error("recovery supports chemistry, central two-particle physics, and neuron")
    client = None
    provider = args.provider or ("anthropic" if args.domain == "boxing" else "openai")
    if not args.smoke:
        if provider == "anthropic":
            from mda.anthropic_client import AnthropicJSONClient
            client = AnthropicJSONClient(model=args.model or "claude-opus-4-7",
                cache_dir=args.cache_dir, replay_only=args.replay)
        else:
            from mda.openai_client import OpenAIJSONClient
            client = OpenAIJSONClient(model=args.model or "gpt-5.6-luna",
                cache_dir=args.cache_dir, replay_only=args.replay)
    if args.domain == "glucose":
        from .glucose import GlucoseBenchEnvironment, make_agent as make_glucose_agent, run_glucose
        environment = GlucoseBenchEnvironment(args.world, replicate=args.replicate)
        agent = make_glucose_agent(environment, client, seed=args.seed, smoke=args.smoke,
                                   experiment_budget=args.budget)
        result = run_glucose(environment, agent, budget=args.budget, cuts=tuple(dict.fromkeys(args.cuts)),
                             design=args.design, prediction=args.prediction or "bma",
                             no_m_open=args.no_m_open, icl_client=client if args.icl else None)
        _write_result(args, result, client)
        return
    fixed_library = args.domain == "chemistry" and args.no_m_open
    environment, agent = build(args.domain, args.world, client, seed=args.seed,
        smoke=args.smoke, budget=args.budget, stochastic=args.stochastic,
        difficulty=args.difficulty, allow_code_execution=args.allow_code_execution, method=args.agent,
        inference=args.inference, llm_initialization=not fixed_library)
    if args.task_noise_variance is not None:
        agent.task_noise_covariance = (args.task_noise_variance[0] if len(args.task_noise_variance) == 1
                                       else np.asarray(args.task_noise_variance))
    result = run(environment, agent, budget=args.budget, smoke=args.smoke,
        prediction=args.prediction or ("bma" if args.domain == "boxing" else "map"),
        design=args.design, no_m_open=args.no_m_open,
        icl_client=client if args.icl else None, recovery=args.recovery,
        evaluate=not (args.smoke and args.domain == "neuron"))
    if args.agent == "apprentice":
        result["design"] = "smoke-random" if args.smoke else "llm"
    _write_result(args, result, client)


def _write_result(args, result, client):
    result["request"] = {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()}
    result["model"] = None if client is None else client.model
    result["versions"] = {}
    for package in ("model-discovery-agent", "numpy", "scipy", "numpyro", "jax", "neuronbench",
                    "glucosebench", "anthropic", "openai"):
        try:
            result["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, default=_json_default, allow_nan=False, indent=2) + "\n")
    print(f"Completed {args.budget} experiments; saved {args.output}")


if __name__ == "__main__":
    main()
