"""Run in a separate environment because NeuroBench and NeuronBench share a package name."""

import numpy as np
import pytest


workload = pytest.importorskip("neurobench.metrics.workload")
torch = pytest.importorskip("torch")


def test_neurobench_mse_matches_the_standard_definition():
    predictions = torch.tensor([1.0, 3.0, -1.0])
    targets = torch.tensor([0.0, 1.0, 2.0])
    reference = workload.MSE()(None, predictions, (None, targets))
    expected = float(np.mean((predictions.numpy() - targets.numpy()) ** 2))
    assert reference == expected
