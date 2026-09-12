import numpy as np
import pytest

from mda.benchmarks.apprentice import BoxingApprentice, make_agent
from mda.benchmarks.boxing import BoxingEnvironment


def test_smoke_is_runner_compatible_and_keeps_one_model():
    env = BoxingEnvironment("dugongs", seed=2, candidate_count=5, query_count=2)
    agent = make_agent(env, None, smoke=True, experiment_budget=1)
    assert isinstance(agent, BoxingApprentice)
    assert agent.config.model_cap == 1
    action = agent.design()
    agent.update_state(action, env.step(action))
    assert len(agent.models) == 1
    assert agent.experiments_run == 1
    assert np.isfinite(agent.get_predictor()(env.query_actions[0]))


def test_design_uses_available_feature_rows_and_rejects_bad_choice():
    env = BoxingEnvironment("dugongs", seed=3, candidate_count=5, query_count=1)
    agent = make_agent(env, None, smoke=True)

    class Client:
        def ask_json(self, system, prompt, **kwargs):
            assert "Current model source/structure" in prompt
            assert "Available unrepeated feature rows" in prompt
            assert "_truth" not in prompt
            return {"choice": 999}

    agent.design_client = Client()
    with pytest.raises(ValueError, match="unavailable"):
        agent.design()


def test_refines_every_result_with_one_proposal_and_residuals():
    env = BoxingEnvironment("dugongs", seed=4, candidate_count=5, query_count=1)
    agent = make_agent(env, None, smoke=True)
    old = agent.models[0]

    class Proposer:
        def propose(self, **kwargs):
            assert kwargs["count"] == 1
            assert kwargs["residuals"] and "signed_per_experiment" in kwargs["residuals"][0]
            return []

    agent.proposer = Proposer()
    action = agent.available_actions()[0]
    agent.update_state(action, env.step(action))
    assert agent.models[0].key == old.key
    assert agent.expansion_count == 0


def test_indexed_prompt_maps_action_to_full_feature_row():
    env = BoxingEnvironment("survival", seed=1, candidate_count=5, query_count=2)
    agent = make_agent(env, None, smoke=True)

    class Client:
        def ask_json(self, system, prompt, **kwargs):
            assert str(env.action_features[agent.available_actions()[0]].tolist()) in prompt
            return {"choice": 0}

    agent.design_client = Client()
    assert agent.design() == agent.available_actions()[0]
