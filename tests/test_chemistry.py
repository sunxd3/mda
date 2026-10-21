import numpy as np
import pytest

from mda.benchmarks.chemistry import (
    CANONICAL_LAWS, PARAM_RANGES, VARIABLE_NAMES, ChemBenchEnvironment, ChemBenchProposer,
    canonical_models, make_agent,
)
from mda.evidence import BICEvidence


class FakeResult:
    def __init__(self, measurement): self.measurement = measurement


class FakeOracle:
    def __init__(self, domain_id, difficulty="easy", noise_level=.01, law_version="v0", grammar_mode="none"):
        self.noise_level = noise_level

    def run(self, action):
        return FakeResult(action["C_A"] + 2 * action["Enz"])


@pytest.fixture
def environment(monkeypatch):
    monkeypatch.setattr("mda.benchmarks.chemistry._oracle_class", lambda: FakeOracle)
    return ChemBenchEnvironment("public-domain", seed=7, query_count=12)


def test_environment_uses_public_inputs_and_clean_seeded_queries(environment):
    assert [datum.action["C_A"] for datum in environment.initial_data] == [.03, .3, 3.]
    assert environment.actions == []
    assert len(environment.query_actions) == 12
    assert all(set(action) == set(VARIABLE_NAMES) for action in environment.query_actions)
    assert all(PARAM_RANGES[k][0] <= action[k] <= PARAM_RANGES[k][1]
               for action in environment.query_actions for k in VARIABLE_NAMES)
    assert "public-domain" not in environment.context


def test_metrics_are_zero_for_exact_raw_predictor(environment):
    metrics = environment.evaluation_metrics(lambda a: a["C_A"] + 2 * a["Enz"])
    assert metrics == {"nmsle": 0.0, "rmsle": 0.0, "passed": True}


def test_pass_threshold_and_unevaluable_law_is_a_failure(environment):
    slightly_off = environment.evaluation_metrics(lambda a: 1.2 * (a["C_A"] + 2 * a["Enz"]))
    assert slightly_off["rmsle"] > 0.05 and not slightly_off["passed"]
    broken = environment.evaluation_metrics(lambda a: np.nan)
    assert broken == {"nmsle": np.inf, "rmsle": np.inf, "passed": False}


def test_query_conditions_depend_on_the_seed(monkeypatch):
    monkeypatch.setattr("mda.benchmarks.chemistry._oracle_class", lambda: FakeOracle)
    first, second = (ChemBenchEnvironment("w", seed=s, query_count=4) for s in (1, 2))
    assert first.query_actions != second.query_actions
    assert first.difficulty == "medium"


def test_canonical_seed_laws_have_positive_bounds_and_mm_form():
    models = canonical_models()
    assert len(models) == len(CANONICAL_LAWS) == 9
    assert all(np.all(m.prior.lower > 0) for m in models)
    mm = models[0]
    action = {"C_A": 1.0, "C_I": 0.0, "C_B": 1.0, "C_P": 0.0, "Enz": 2.0, "T": 310.0, "pH": 7.0}
    assert float(mm.observation_prediction(np.array([5.0, 1.0]), action)) == pytest.approx(5.0)
    arrhenius = models[3]
    assert float(arrhenius.observation_prediction(np.array([5.0, 65000.0, 1.0]), action)) == pytest.approx(5.0)


def test_smoke_agent_has_v5_configuration(environment):
    agent = make_agent(environment, None, smoke=True, experiment_budget=5)
    assert len(agent.models) <= 9
    assert {model.name for model in agent.models} <= {law[0] for law in CANONICAL_LAWS}
    config = agent.config
    assert (config.model_cap, config.minimum_model_cap, config.new_models_per_expansion) == (24, 6, 6)
    assert config.shrink_confidence == pytest.approx(0.9)
    assert config.initial_proposals == 12 and config.expansion_gate == "cached"
    assert config.occam_penalty == 0.0
    assert isinstance(agent.evidence, BICEvidence) and agent.evidence.transform is np.log1p
    assert agent.experiment_budget == 5
    assert agent.optimizer.log_axes.tolist() == [True, False, True, False, True, False, False]


def test_smc_control_and_fixed_library_arms(environment):
    smc = make_agent(environment, None, smoke=True, inference="smc")
    assert smc.evidence is None and smc.config.occam_penalty == 2.0
    fixed = make_agent(environment, None, llm_initialization=False)
    assert fixed.proposer is None and fixed.experiment_budget == 22
    assert fixed.optimizer.max_evaluations == 48


def test_random_design_samples_the_log_mapped_box(environment):
    agent = make_agent(environment, None, smoke=True)
    actions = [agent.optimizer.sample(np.random.default_rng(i)) for i in range(50)]
    for name in VARIABLE_NAMES:
        values = [a[name] for a in actions]
        assert PARAM_RANGES[name][0] <= min(values) and max(values) <= PARAM_RANGES[name][1]


def test_client_is_required_outside_smoke(environment):
    with pytest.raises(ValueError, match="only for smoke"):
        make_agent(environment, None)


def test_driver_records_pass_metrics_and_samples_random_designs(monkeypatch):
    from mda.benchmarks.reproduce import build, run
    monkeypatch.setattr("mda.benchmarks.chemistry._oracle_class", lambda: FakeOracle)
    environment, agent = build("chemistry", "world", None, seed=3, smoke=True, budget=1)
    assert environment.difficulty == "medium"
    result = run(environment, agent, budget=1, smoke=True, design="random")
    last = result["history"][-1]
    assert set(last["metrics"]) == {"nmsle", "rmsle", "passed"}
    assert last["score"] == last["metrics"]["nmsle"]
    assert set(last["action"]) == set(VARIABLE_NAMES)
    with pytest.raises(ValueError, match="selects only the ChemBench arm"):
        build("boxing", "death", None, seed=0, smoke=True, budget=1, inference="smc")


def test_proposer_uses_v5_prompt_repairs_bounds_and_drops_laws_outside_its_grammar(environment):
    class Client:
        def ask_json(self, system, prompt, **kwargs):
            self.system, self.prompt = system, prompt
            positive = [{"name": "k", "low": .01, "high": 100}, {"name": "Km", "low": .001, "high": 100}]
            return {"hypotheses": [
                {"name": "mm", "expr": "k*Enz*C_A/(Km + C_A)", "params": positive},
                {"name": "tanh", "expr": "k*Enz*tanh(C_A/Km)", "params": positive},
                {"name": "offset", "expr": "k*CA + c", "params": [positive[0], {"name": "c", "low": 0, "high": 2}]},
                {"name": "negative", "expr": "k*C_A + c", "params": [positive[0], {"name": "c", "low": -2, "high": -1}]},
            ]}

    client = Client()
    models = ChemBenchProposer(client).propose(
        context=environment.context + "\n\nCurrent phase: explore.", dataset=environment.initial_data,
        current_models=(), residuals=(), count=3)
    assert [model.name for model in models] == ["mm", "offset"]
    assert models[1].structure == "k * C_A + c"
    assert models[1].prior.lower.tolist() == [.01, 2e-6]
    assert client.system.startswith("You are an enzyme kineticist")
    assert "C_A=0.03, C_I=0, C_B=1, C_P=0, Enz=1, T=310, pH=7 -> r0=" in client.prompt
    assert "Current phase: explore." in client.prompt and "ALREADY TRIED" not in client.prompt
