import pytest


nb = pytest.importorskip("neuronbench")


def test_released_neuronbench_contract_and_metric():
    assert nb.list_worlds() == [
        "z_rebound",
        "h_sag",
        "na_fatigue",
        "ca_rebound",
        "d_type",
        "textbook_M",
    ]
    world = nb.load_world("ca_rebound", stochastic=False, seed=0)
    assert len(world.protocol_pool) == 9
    observation = world.run(world.protocol_pool[0])
    assert observation.voltage.ndim == 1
    targets = nb.evaluator.held_out_targets("ca_rebound", stochastic=False, seed=0)
    assert world.spike_forecast_mse(targets) == 0.0
