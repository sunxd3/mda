import numpy as np

from mda.benchmarks.physics import MEASUREMENT_TIMES, action_key, force_candidate, paper_actions


def test_environment_explicitly_enables_training_noise(monkeypatch):
    from types import SimpleNamespace
    from mda.benchmarks.physics import PhysicsEnvironment

    class Worlds:
        def get_world(self, name, **kwargs):
            assert kwargs["noise_std"] == .05
            return {"mission": "public", "executor": SimpleNamespace(
                run=lambda actions: [{"pos2": np.zeros((6, 2)).tolist()}])}

    monkeypatch.setattr("mda.benchmarks.physics._scienceagent", lambda: (
        Worlds(), SimpleNamespace(Evaluator=lambda executor: SimpleNamespace(test_cases=[]))))
    PhysicsEnvironment("gravity")


def test_paper_menu_and_seed():
    actions = paper_actions()
    assert len(actions) == 13
    assert len({action_key(a) for a in actions}) == 13
    assert all(a["measurement_times"] == MEASUREMENT_TIMES for a in actions)


def test_constant_central_force_symplectic_trajectory():
    model = force_candidate("constant", "a*qi*qj", ("a",), (0.5,), (1.5,))
    action = {"p1": 1, "p2": 2, "pos2": [3, 0], "velocity2": [0, 0],
              "measurement_times": [.5, 1.]}
    trajectory = model.observation_prediction(np.array([1.]), action)
    np.testing.assert_allclose(trajectory[:, 0], [3 - .25 * .5**2, 3 - .25 * 1**2], atol=1e-10)
    np.testing.assert_allclose(trajectory[:, 1], 0)


def test_coulomb_roles_use_both_charges_not_p2_as_mass():
    model = force_candidate("charge", "a*qi*qj", ("a",), (.5,), (1.5,),
                            coulomb_roles=True)
    action = {"p1": 2, "p2": 3, "pos2": [3, 0], "velocity2": [0, 0],
              "measurement_times": [.5]}
    np.testing.assert_allclose(model.observation_prediction(np.array([1.]), action)[0], [2.25, 0], atol=1e-9)


def test_qj_is_source_and_qi_is_probe_for_asymmetric_law():
    model = force_candidate("source only", "a*qj", ("a",), (.5,), (1.5,),
                            coulomb_roles=True)
    action = {"p1": 2, "p2": 3, "pos2": [3, 0], "velocity2": [0, 0],
              "measurement_times": [.5]}
    np.testing.assert_allclose(model.observation_prediction(np.array([1.]), action)[0],
                               [2.75, 0], atol=1e-9)


def test_recovery_measures_force_without_training_or_hidden_fields():
    from contextlib import contextmanager
    from mda.benchmarks.physics import PhysicsEnvironment

    class PublicExecutor:
        clean = False

        @contextmanager
        def noise_disabled(self):
            self.clean = True
            try:
                yield
            finally:
                self.clean = False

        def run(self, actions):
            assert self.clean
            outputs = []
            for action in actions:
                assert action["p1"] == action["p2"] == 1
                assert action["velocity2"] == [0., 0.]
                r, h = action["pos2"][0], action["measurement_times"][0]
                # Independent analytic initial-acceleration fixture, F=3/r.
                outputs.append({"pos2": [[r - .5 * (3/r) * h*h, 0.]]})
            return outputs

    env = object.__new__(PhysicsEnvironment)
    env.executor = PublicExecutor()
    env.initial_data = []
    right = force_candidate("inverse", "a/r", ("a",), (.1,), (10.,))
    wrong = force_candidate("square", "a/r**2", ("a",), (.1,), (10.,))
    result = env.evaluate_model(right)
    np.testing.assert_allclose(result["parameters"], [3.], atol=1e-6)
    assert result["recovered"] and result["rmsle"] < 1e-7
    assert not env.evaluate_model(wrong)["recovered"]
    assert env.initial_data == [] and not env.executor.clean


def test_force_proposer_drops_invalid_laws_but_keeps_their_valid_siblings():
    from mda.benchmarks.physics import ForceProposer
    from mda.types import Datum

    hypotheses = [
        {"name": "divides", "fmag": "a*qi*qj/r + 1/0", "params": [{"name": "a", "low": 0.1, "high": 2.0}]},
        {"name": "overflows", "fmag": "a*qi*qj/r*10.0**400", "params": [{"name": "a", "low": 0.1, "high": 2.0}]},
        {"name": "fine", "fmag": "a*qi*qj/r", "params": [{"name": "a", "low": 0.1, "high": 2.0}]},
    ]

    class Client:
        def ask_json(self, system, prompt, **kwargs):
            return {"hypotheses": hypotheses}

    action = {"p1": 1, "p2": 2, "pos2": [3, 0], "velocity2": [0, 0], "measurement_times": [.5, 1.]}
    models = ForceProposer(Client()).propose(
        context="", dataset=[Datum(action, np.zeros((2, 2)))], current_models=[], residuals=[], count=3)
    assert [m.name for m in models] == ["fine"]


def test_continuous_launches_keep_direction_but_respect_the_minimum_radius():
    from mda.benchmarks.physics import continuous_action

    action = continuous_action([.1, -.2, 1, 2, 3, 4])
    np.testing.assert_allclose(np.hypot(*action["pos2"]), .5)
    np.testing.assert_allclose(np.arctan2(action["pos2"][1], action["pos2"][0]), np.arctan2(-.2, .1))
    assert (action["velocity2"], action["p1"], action["p2"]) == ([1., 2.], 3., 4.)
    assert continuous_action([4, 0, 0, 0, 1, 1])["pos2"] == [4., 0.]


def test_chi2_check_fails_a_charge_blind_law_on_one_charge_probe():
    from mda.benchmarks.physics import _action, fresh_chi2_check
    from mda.likelihoods import DeterministicGaussianLikelihood
    from mda.posterior import fit_model_posterior
    from mda.smc import SMCConfig
    from mda.types import Datum

    true = force_candidate("true", "k*qi*qj/r**2", ("k",), (.01,), (5.,))
    blind = force_candidate("blind", "k/r**2", ("k",), (.01,), (5.,))
    seed_drop = [Datum(_action(3), true.predict(np.array([1.]), _action(3)))]
    probe = _action(4, p1=3)
    fresh = Datum(probe, true.predict(np.array([1.]), probe))
    checks = []
    for model in (true, blind):
        belief = fit_model_posterior(seed_drop, [model], DeterministicGaussianLikelihood(.05),
                                     SMCConfig(parameter_particles=60), model_cap=1,
                                     occam_penalty=0., rng=np.random.default_rng(0))
        checks.append(fresh_chi2_check(belief, fresh, seed_drop))
    # Both laws fit the unit-charge drop; only a charge intervention separates them.
    assert checks[0] < 1 < checks[1]


def test_guided_proposer_shows_where_the_best_law_fails_and_caps_parameters():
    from types import SimpleNamespace
    from mda.benchmarks.physics import ForceProposer, _action
    from mda.types import Datum

    law = force_candidate("blind", "k/r**2", ("k",), (.01,), (5.,))
    fit = SimpleNamespace(model=law, parameters=SimpleNamespace(map_parameters=np.array([1.])))
    probe = _action(4, p1=3)
    true = force_candidate("true", "k*qi*qj/r**2", ("k",), (.01,), (5.,))
    data = [Datum(probe, true.predict(np.array([1.]), probe))]
    prompts = []

    class Client:
        def ask_json(self, system, prompt, **kwargs):
            prompts.append(prompt)
            return {"hypotheses": [
                {"name": "wide", "fmag": "a*qi*qj/r**b + c*exp(-r/d)", "operator": "", "rationale": "",
                 "params": [{"name": n, "low": .1, "high": 2.} for n in "abcd"]},
                {"name": "ok", "fmag": "a*qi*qj/r**2", "operator": "", "rationale": "",
                 "params": [{"name": "a", "low": .1, "high": 2.}]}]}

    proposer = ForceProposer(Client(), guided=True, max_parameters=3)
    proposer.agent = SimpleNamespace(belief=SimpleNamespace(map_fit=fit))
    models = proposer.propose(context="", dataset=data, current_models=[law], residuals=[], count=6)
    assert [m.name for m in models] == ["ok"]
    assert "Current best law: F_mag = k / r ** 2 with k=1." in prompts[0]
    assert "at most 3 free parameters" in prompts[0]
    # The blind law misses the charge-3 probe by far more than the noise.
    misfit = float(prompts[0].split("per experiment in order:\n")[1].split("\n")[0])
    assert misfit > 5
