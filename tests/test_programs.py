from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mda.agent import AgentConfig, ModelDiscoveryAgent, default_action_key
from mda.benchmarks import episodic
from mda.benchmarks.reproduce import run
from mda.likelihoods import DeterministicGaussianLikelihood
from mda.openai_client import OpenAIJSONClient
from mda.posterior import BeliefState, ModelFit
from mda.program_proposer import LLMProgramProposer
from mda.program_sandbox import SandboxError, SandboxedProgram, compile_program
from mda.programs import (
    FixedProgram, History, ProgramError, belief_summary, run_episode, trajectory_information,
)
from mda.smc import ParameterPosterior, SMCConfig
from mda.types import Datum

LIKELIHOOD = DeterministicGaussianLikelihood(episodic.STD)
SMC = SMCConfig(parameter_particles=40, rejuvenation_steps=2)


def _initial_data():
    environment = episodic.ThresholdEpisodes(seed=99)
    return [Datum(np.asarray([0.1, 0.9]), environment.step([0.1, 0.9]))]


def _agent(**overrides):
    options = dict(
        context=episodic.CONTEXT, initial_data=_initial_data(), initial_models=episodic.models(),
        likelihood=LIKELIHOOD, actions=[], observation_covariance=episodic.STD**2,
        config=AgentConfig(design_flavour="program", smc=SMC, program_draws=60),
        episode=episodic.EPISODE, programs=episodic.seed_programs(), seed=3,
    )
    options.update(overrides)
    return ModelDiscoveryAgent(**options)


def test_a_fixed_program_is_an_ordinary_action():
    inputs = [0.3, 0.5, 0.58, 0.7, 0.9]
    ordinary = episodic.ThresholdEpisodes(seed=5).step(inputs)
    action, observation = run_episode(
        FixedProgram(inputs), episodic.ThresholdEpisodes(seed=5), episodic.EPISODE, {})
    np.testing.assert_array_equal(action, inputs)
    np.testing.assert_array_equal(observation, ordinary)
    # Repeat detection keys the realised inputs exactly like the ordinary action.
    assert default_action_key(action) == default_action_key(np.asarray(inputs))

    agent = _agent(programs=[FixedProgram(inputs), *episodic.seed_programs()])
    agent.update_state(action, observation)
    agent.design()
    assert all(not isinstance(p, FixedProgram) or not np.array_equal(p.inputs, inputs)
               for p in agent.programs)


def test_feedback_reaches_a_sandboxed_program():
    bracket = episodic.seed_programs()[1]
    summary = {"pooled": {"threshold": {"mean": 0.5}}}
    runs = {
        (mechanism, threshold): run_episode(
            bracket, episodic.ThresholdEpisodes(mechanism, threshold, seed=1), episodic.EPISODE, summary)
        for mechanism in ("rise", "fall") for threshold in (0.3, 0.7)
    }
    first_inputs = {inputs[0] for inputs, _ in runs.values()}
    assert first_inputs == {0.55}
    # Different observations lead to different later inputs: the program
    # brackets downwards after saturation and upwards after a zero response.
    low, high = runs["rise", 0.3][0], runs["rise", 0.7][0]
    assert low[1] < 0.55 < high[1]
    for (mechanism, threshold), (inputs, observations) in runs.items():
        assert threshold < inputs[-1] <= threshold + episodic.WINDOW
        assert np.sign(observations[-1]) == (1 if mechanism == "rise" else -1)


def test_fitting_is_unchanged_on_a_realised_episode():
    program_agent = _agent()
    ordinary_agent = _agent(config=AgentConfig(smc=SMC))
    program = program_agent.design()
    inputs, trajectory = run_episode(program, episodic.ThresholdEpisodes(seed=2), episodic.EPISODE,
                                     belief_summary(program_agent.belief))
    ordinary_agent.rng.bit_generator.state = copy.deepcopy(program_agent.rng.bit_generator.state)
    program_agent.update_state(inputs, trajectory)
    ordinary_agent.update_state(inputs, trajectory)
    assert program_agent.dataset[-1].action is inputs
    for left, right in zip(program_agent.belief.fits, ordinary_agent.belief.fits, strict=True):
        assert left.parameters.log_evidence == right.parameters.log_evidence
        np.testing.assert_array_equal(left.parameters.particles, right.parameters.particles)
    assert default_action_key(inputs) in program_agent.used_action_keys


def _particle_belief(models, thetas, copies=1):
    """A belief of equally weighted models whose particles are ``thetas`` repeated ``copies`` times."""
    fits = []
    for model in models:
        particles = np.repeat(np.asarray(thetas, dtype=float).reshape(-1, 1), copies, axis=0)
        weights = np.full(len(particles), 1.0 / len(particles))
        posterior = ParameterPosterior(particles, weights, np.zeros(len(particles)), 0.0, (1.0,), 1.0)
        fits.append(ModelFit(model, posterior, 0.0))
    return BeliefState(tuple(fits), np.full(len(models), 1.0 / len(models)))


@pytest.mark.slow
def test_scorer_is_zero_for_indistinguishable_models_and_bounded_otherwise():
    thetas = np.random.default_rng(0).uniform(0.2, 0.8, 300)
    bracket = episodic.seed_programs()[1]
    twin = replace(episodic.make_model("rise", 1.0), structure="twin")
    twins = _particle_belief([episodic.make_model("rise", 1.0), twin], thetas)
    zero = trajectory_information(bracket, twins, LIKELIHOOD, episodic.EPISODE, draws=150, seed=4)
    assert abs(zero.bits) < 0.05

    belief = _particle_belief(episodic.models(), thetas)
    adaptive = trajectory_information(bracket, belief, LIKELIHOOD, episodic.EPISODE, draws=150, seed=4)
    spread = trajectory_information(FixedProgram([0.3, 0.45, 0.55, 0.65, 0.75]), belief, LIKELIHOOD,
                                    episodic.EPISODE, draws=150, seed=4)
    below = trajectory_information(FixedProgram([0.05] * 5), belief, LIKELIHOOD, episodic.EPISODE,
                                   draws=150, seed=4)
    # One bit is all there is; bracketing finds the window, a fixed spread
    # only sometimes, and inputs below every threshold never.
    assert 0.9 < adaptive.bits <= 1.0 + 1e-9
    assert 0.2 < spread.bits < adaptive.bits
    assert abs(below.bits) < 0.05


def test_scorer_uses_common_random_numbers_and_ignores_duplicate_particles():
    thetas = np.random.default_rng(1).uniform(0.2, 0.8, 30)
    bracket = episodic.seed_programs()[1]
    single = _particle_belief(episodic.models(), thetas)
    tripled = _particle_belief(episodic.models(), thetas, copies=3)
    score = lambda belief, seed: trajectory_information(  # noqa: E731
        bracket, belief, LIKELIHOOD, episodic.EPISODE, draws=100, seed=seed).bits
    assert score(single, 5) == score(single, 5)
    # Self-exclusion is by value: SMC copies of the outer particle never stay
    # in its own inner sum, so duplicating every particle changes nothing.
    assert score(tripled, 5) == score(single, 5)


@pytest.mark.parametrize("source", [
    "import os",
    "class P(Program):\n    def step(self, history, belief):\n        import os\n        return 0.5",
    "class P(Program):\n    def step(self, history, belief):\n        return __import__('os')",
    "class P(Program):\n    def step(self, history, belief):\n        return open('/etc/passwd')",
    "class P(Program):\n    def step(self, history, belief):\n        return history.__class__",
    "class P(Program):\n    def step(self, history, belief):\n        return (1).__class__",
    "class P(Program):\n    def step(self, history, belief):\n        history.inputs.append(1)\n        return 0.5",
    "class P(Program):\n    def step(self, history, belief):\n        while True:\n            pass",
    "class P(Program):\n    def step(self, history, belief):\n        return (lambda: 0.5)()",
    "class P(Program):\n    def step(self, history, belief):\n        return getattr(history, 't')",
    "class P(Program):\n    def step(self, history, belief):\n        self.x = 1\n        return 0.5",
    "class P(Program):\n    def helper(self):\n        return 1\n    def step(self, history, belief):\n        return 0.5",
    "class P(Program):\n    def step(self, history, belief):\n        return ''.join([])",
    "class P(object):\n    def step(self, history, belief):\n        return 0.5",
])
def test_sandbox_rejects_unsafe_source(source):
    with pytest.raises(SandboxError):
        compile_program(source)


def test_sandbox_contains_runaway_and_failing_programs():
    burn = SandboxedProgram(
        "class P(Program):\n    def step(self, history, belief):\n        return 10 ** 10 ** 9",
        cpu_seconds=1, memory_mb=256, timeout=20)
    with burn.session() as step, pytest.raises(ProgramError):
        step([episodic_history()], {})
    broken = compile_program("class P(Program):\n    def step(self, history, belief):\n        return history.obs[3]")
    with pytest.raises(ProgramError):
        run_episode(broken, episodic.ThresholdEpisodes(), episodic.EPISODE, {})
    # The agent drops a program that fails instead of aborting the design.
    agent = _agent(programs=[broken, *episodic.seed_programs()])
    agent.design()
    assert broken.key not in {program.key for program in agent.programs}


def episodic_history():
    return History(horizon=5)


VALID = '''
class Midpoint(Program):
    def step(self, history, belief):
        return belief["pooled"]["threshold"]["mean"] + 0.05
'''


class FakeClient:
    model = "fake"

    def __init__(self, programs):
        self.programs = programs
        self.calls = []

    def ask_json(self, system, prompt, *, schema, name, max_tokens):
        self.calls.append(prompt)
        return {"programs": self.programs}


def _payload():
    return [
        {"name": "Midpoint", "rationale": "probe the window", "source": VALID},
        {"name": "Unsafe", "rationale": "", "source": "import os"},
        {"name": "Bracket", "rationale": "duplicate", "source": episodic.BRACKET_SOURCE},
    ]


def test_program_proposer_is_called_when_the_pool_changes_and_programs_are_capped():
    client = FakeClient(_payload())
    proposer = LLMProgramProposer(client, episodic.EPISODE)
    agent = _agent(
        programs=episodic.seed_programs(), program_proposer=proposer,
        config=AgentConfig(design_flavour="program", smc=SMC, program_draws=60, program_cap=2),
    )
    agent.design()
    assert len(client.calls) == 1 and agent.program_proposals == 1
    prompt = client.calls[0]
    assert "rise" in prompt and "fall" in prompt and "Bracket" in prompt and "bits" in prompt
    scored = [label for label, _ in agent.last_program_scores]
    assert len(scored) == 3 and any(label.startswith("Midpoint") for label in scored)
    assert len(agent.programs) == 2
    # Same models: no new call.
    agent.design()
    assert len(client.calls) == 1


def test_program_proposer_replays_from_the_cache_offline(monkeypatch, tmp_path):
    calls = []

    class Responses:
        def create(self, **request):
            calls.append(request)
            return SimpleNamespace(status="completed", output_text=json.dumps({"programs": _payload()}))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(
        OpenAI=lambda api_key: SimpleNamespace(responses=Responses())))
    proposals = {}
    for replay in (False, True):
        client = OpenAIJSONClient(model="gpt-5.6-luna", api_key="not-a-secret", cache_dir=tmp_path,
                                  replay_only=replay, sample="test")
        agent = _agent(program_proposer=LLMProgramProposer(client, episodic.EPISODE))
        agent.design()
        proposals[replay] = sorted(label for label, _ in agent.last_program_scores)
    assert len(calls) == 1
    assert proposals[False] == proposals[True]


def test_run_executes_programs_within_episodes():
    agent = _agent()
    result = run(episodic.ThresholdEpisodes(seed=7), agent, budget=2, design="program", evaluate=False)
    assert len(result["history"]) == 3
    for datum in agent.dataset[1:]:
        assert np.asarray(datum.action).shape == (episodic.EPISODE.steps,)
        assert datum.observation.shape == (episodic.EPISODE.steps,)
    assert agent.belief.model_weights[0] > 0.99  # the truth is "rise"


def test_program_design_requires_step_samplers_and_an_episode():
    with pytest.raises(ValueError):
        _agent(episode=None)
    models = [replace(model, sample_step=None) for model in episodic.models()]
    with pytest.raises(ValueError, match="sample_step"):
        _agent(initial_models=models).design()
