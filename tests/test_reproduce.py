import json

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
