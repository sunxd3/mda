import numpy as np
import pytest

from mda.benchmarks.chemistry import PARAM_RANGES, VARIABLE_NAMES, ChemBenchEnvironment, make_agent


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
    assert metrics == {"nmsle": 0.0, "rmsle": 0.0}


def test_smoke_agent_has_paper_configuration(environment):
    agent = make_agent(environment, None, smoke=True, experiment_budget=5)
    assert {model.name for model in agent.models} <= {"constant", "linear"}
    assert agent.config.model_cap == 16
    assert agent.config.minimum_model_cap == 6
    assert agent.config.smc.parameter_particles == 6
    assert agent.experiment_budget == 5
    assert agent.optimizer.log_axes.tolist() == [True, False, True, False, True, False, False]


def test_client_is_required_outside_smoke(environment):
    with pytest.raises(ValueError, match="only when smoke=True"):
        make_agent(environment, None)
