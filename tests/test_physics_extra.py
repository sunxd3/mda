import numpy as np

from mda.benchmarks.physics_extra import (
    PhysicsExtraEnvironment, ForceProposer, _circle_action, force_candidate, paper_actions,
)
from mda.types import Datum


def test_all_pair_solver_preserves_symmetric_circle_centre():
    model = force_candidate("inverse", "a*qi*qj/r", ("a",), (0.1,), (2.0,),
                            world="circle", dt=.02)
    trajectory = model.predict(np.array([1.0]), _circle_action(3, 0))
    assert trajectory.shape == (5, 11, 2)
    np.testing.assert_allclose(trajectory[:, 0], 0, atol=1e-12)


def test_asymmetric_source_weights_and_external_kick():
    action = paper_actions("ether")[0]
    background = np.column_stack((np.arange(21, dtype=float) + 100., np.zeros(21)))
    background[0] = 0.
    action.update(background_initial_positions=background.tolist(),
                  background_initial_velocities=np.zeros((21, 2)).tolist(),
                  particle_masses=([1e15] + [1.] * 25), measurement_times=[.2])
    # Zero pair law isolates the fitted, mass-independent ether acceleration.
    model = force_candidate("external", "z*qi*qj", ("z",), (-1,), (1,),
                            world="ether", dt=.01)
    result = model.predict(np.array([0., .4]), action)
    initial_y = np.asarray(action["probe_positions"])[:, 1]
    np.testing.assert_allclose(result[0, :, 1] - initial_y,
                               .5 * .4 * .2**2, atol=1e-10)
    # The returned tensor contains only probes, as required by the evaluator.
    assert result.shape == (1, 5, 2)


def test_local_designs_contain_only_public_action_fields():
    allowed = {"probe_positions", "probe_velocities", "probe_masses",
               "measurement_times", "ring_radius", "initial_tangential_velocity"}
    for world in ("circle", "ether", "hubble"):
        actions = paper_actions(world)
        assert len(actions) == 6
        assert all(set(action) <= allowed for action in actions)


def test_extra_evaluator_normalizes_by_variance():
    from contextlib import nullcontext
    from types import SimpleNamespace
    env = object.__new__(PhysicsExtraEnvironment)
    env.executor = SimpleNamespace(noise_disabled=nullcontext)
    env.query_actions = [0, 1]
    truths = [np.array([[[1., 3.]]]), np.array([[[5., 7.]]])]
    env._run_public = lambda a: (a, truths[a])
    assert env.evaluate_predictor(lambda a: truths[a] + 2) == 4/5


def test_extra_proposer_receives_observations_and_complete_priors():
    action = _circle_action(3., 0.)
    model = force_candidate("candidate", "a*qi*qj/r", ("a",), (.1,), (1.7,), world="circle")

    class Client:
        def ask_json(self, system, prompt, **kwargs):
            assert "Observed experiments" in prompt and "123.25" in prompt
            assert '"a": [0.1, 1.7]' in prompt
            return {"hypotheses": []}

    ForceProposer(Client(), "circle").propose(context="test", dataset=[Datum(action, np.full((5,11,2),123.25))],
        current_models=[model], residuals=[], count=1)
