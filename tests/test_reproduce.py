import json
from dataclasses import replace

import numpy as np
import pytest

from mda.benchmarks.reproduce import _json_default, build, main, run


def test_run_records_updates_and_keeps_evaluation_out_of_training():
    env, agent = build("boxing", "death", None, seed=4, smoke=True, budget=2)
    initial_count = len(agent.dataset)
    results = run(env, agent, budget=2, smoke=True, design="joint")
    assert len(agent.dataset) == initial_count + 2
    assert len(results["history"]) == 3
    assert [r["experiments"] for r in results["history"]] == [0, 1, 2]
    query_keys = {repr(a) for a in env.query_actions}
    assert not query_keys.intersection(repr(d.action) for d in agent.dataset)
    scores = [r["score"] for r in results["history"]]
    np.testing.assert_allclose([r["best_score_so_far"] for r in results["history"]],
                               np.minimum.accumulate(scores))
    json.dumps(results, default=_json_default, allow_nan=False)


def test_evaluation_does_not_change_designs_or_fits():
    runs = []
    for evaluate in (False, True):
        env, agent = build("boxing", "dugongs", None, seed=7, smoke=True, budget=1)
        runs.append(run(env, agent, budget=1, evaluate=evaluate)["history"])
    assert runs[0][-1]["action"] == runs[1][-1]["action"]
    np.testing.assert_array_equal(runs[0][-1]["map_parameters"], runs[1][-1]["map_parameters"])
    assert runs[0][-1]["score"] is None


def test_cli_requires_explicit_execution_mode(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["mda-reproduce", "boxing", "--world", "death",
                                    "--output", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize("extra", [[], ["--stochastic", "--no-m-open"]])
def test_cli_rejects_online_expansion_outside_open_stochastic_neuron(monkeypatch, tmp_path, extra):
    monkeypatch.setattr("sys.argv", ["mda-reproduce", "neuron", "--world", "ca_rebound", "--smoke",
                                     "--online-expansion", *extra,
                                     "--output", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_smoke_cli_writes_result(monkeypatch, tmp_path):
    path = tmp_path / "result.json"
    monkeypatch.setattr("sys.argv", ["mda-reproduce", "boxing", "--world", "hyperbolic",
                                    "--smoke", "--output", str(path)])
    main()
    result = json.loads(path.read_text())
    assert result["model"] is None
    assert result["prediction"] == "bma"
    assert result["history"][-1]["experiments"] == 1
    assert result["request"]["seed"] == 0


def test_random_no_m_open_and_icl_keep_model_inference_independent():
    env, agent = build("boxing", "irt", None, seed=3, smoke=True, budget=1)

    class ForbiddenProposer:
        def propose(self, **kwargs):
            raise AssertionError("no-M-open must disable expansion")

    class Client:
        def ask_json(self, system, prompt, **kwargs):
            payload = json.loads(prompt)
            assert "model" not in payload and "truth" not in payload
            assert len(payload["observations"]) in (2, 3)
            assert all(len(q["action"]) == 2 for q in payload["queries"])
            return {"predictions": [.25] * len(payload["queries"])}

    agent.proposer = ForbiddenProposer()
    result = run(env, agent, budget=1, design="random", no_m_open=True, icl_client=Client())
    assert result["history"][-1]["expansions"] == 0
    assert result["history"][-1]["action"] not in [d.action for d in env.initial_data]
    assert result["history"][-1]["icl_score"] == pytest.approx(
        np.mean([(.25 - env.step(a))**2 for a in env.query_actions]))


def test_random_runner_honors_remaining_budget():
    env, agent = build("boxing", "death", None, seed=1, smoke=True, budget=0)
    with pytest.raises(ValueError, match="remaining experiment budget"):
        run(env, agent, budget=1, design="random")


def test_unevaluable_law_is_recorded_as_a_failure_not_an_abort():
    env, agent = build("boxing", "death", None, seed=1, smoke=True, budget=0)

    class Unevaluable:
        query_actions = env.query_actions

        def evaluation_metrics(self, predictor):
            return {"nmsle": np.inf, "rmsle": np.inf, "passed": False}

    agent.predict = lambda action, flavour="map": np.asarray(np.inf)
    result = run(Unevaluable(), agent, budget=0)
    row = result["history"][0]
    assert row["evaluable"] is False and row["score"] is None
    assert row["metrics"] == {"nmsle": None, "rmsle": None, "passed": False}
    json.dumps(result, default=_json_default, allow_nan=False)


def test_random_designs_do_not_depend_on_inference_randomness():
    actions = []
    for particles in (8, 24):
        env, agent = build("boxing", "irt", None, seed=5, smoke=True, budget=3)
        agent.config = replace(agent.config, smc=replace(agent.config.smc, parameter_particles=particles))
        history = run(env, agent, budget=3, smoke=True, design="random", evaluate=False)["history"]
        actions.append([repr(row["action"]) for row in history[1:]])
    assert actions[0] == actions[1]


def test_budget_beyond_the_unused_menu_fails_before_running():
    env, agent = build("boxing", "death", None, seed=1, smoke=True, budget=1000)
    with pytest.raises(ValueError, match="unused designs"):
        run(env, agent, budget=len(agent.available_actions()) + 1, smoke=True)


def test_an_aborted_run_hands_back_its_completed_snapshots():
    from mda.benchmarks.reproduce import RunAborted

    env, agent = build("boxing", "death", None, seed=4, smoke=True, budget=3)
    steps = []
    original = env.step

    def failing_step(action):
        if len(steps) == 1:
            raise RuntimeError("instrument offline")
        steps.append(action)
        return original(action)

    env.step = failing_step
    with pytest.raises(RunAborted, match="instrument offline") as aborted:
        run(env, agent, budget=3, smoke=True)
    history = aborted.value.result["history"]
    assert [r["experiments"] for r in history] == [0, 1]
    json.dumps(aborted.value.result, default=_json_default, allow_nan=False)


def test_icl_failure_is_recorded_beside_the_run_not_fatal():
    env, agent = build("boxing", "death", None, seed=4, smoke=True, budget=1)

    class WrongLength:
        def ask_json(self, system, prompt, **kwargs):
            return {"predictions": [0.5]}

    result = run(env, agent, budget=1, smoke=True, icl_client=WrongLength())
    assert len(result["history"]) == 2
    assert all(r["icl_score"] is None and "icl_error" in r for r in result["history"])


@pytest.mark.parametrize("argv", [
    ["boxing", "--world", "death", "--smoke", "--difficulty", "hard"],
    ["boxing", "--world", "death", "--smoke", "--cuts", "45"],
    ["chemistry", "--world", "c0", "--smoke", "--allow-code-execution"],
    ["boxing", "--world", "death", "--smoke", "--task-noise-variance", "0.1"],
    ["boxing", "--world", "death", "--smoke", "--model", "gpt-5.6"],
    ["boxing", "--world", "death", "--smoke", "--cache-dir", "/tmp/x"],
])
def test_cli_rejects_flags_that_would_have_no_effect(monkeypatch, tmp_path, argv):
    monkeypatch.setattr("sys.argv", ["mda-reproduce", *argv, "--output", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
