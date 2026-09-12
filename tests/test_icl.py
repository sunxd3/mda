import json

import numpy as np
import pytest

from mda.benchmarks.icl import make_icl_predictor
from mda.types import Datum


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def ask_json(self, system, prompt, **kwargs):
        self.calls.append((system, prompt, kwargs))
        return self.response


def test_icl_uses_only_observed_targets_and_decoded_actions():
    client = FakeClient({"predictions": [[3, 4], [5, 6]]})
    data = [Datum(0, np.array([10.0, 999.0]))]
    predictor = make_icl_predictor(
        client, "system description", data, [1, 2],
        target=lambda y: y[:1], action_features=lambda i: [i, i + 0.5],
        action_key=str, target_shape=(2,),
    )
    assert len(client.calls) == 1
    body = json.loads(client.calls[0][1])
    assert body == {
        "context": "system description",
        "observations": [{"action": [0, 0.5], "target": [10.0]}],
        "queries": [{"index": 0, "action": [1, 1.5]}, {"index": 1, "action": [2, 2.5]}],
    }
    assert "999" not in client.calls[0][1]
    np.testing.assert_array_equal(predictor(2), [5, 6])
    with pytest.raises(KeyError):
        predictor(3)


@pytest.mark.parametrize("response", [
    {"predictions": [1.0]},
    {"predictions": [[1.0], [2.0, 3.0]]},
    {"predictions": [[1.0], [float("nan")]]},
    {"wrong": []},
])
def test_icl_rejects_malformed_predictions(response):
    with pytest.raises(ValueError):
        make_icl_predictor(FakeClient(response), "c", [], ["a", "b"], target_shape=(1,))


def test_scalar_predictions_and_two_argument_neuron_target():
    client = FakeClient({"predictions": [1.5]})
    predictor = make_icl_predictor(
        client, "neuron", [("protocol", [1, 2])], ["query"],
        target=lambda trace, protocol: [len(trace), len(protocol)],
    )
    assert float(predictor("query")) == 1.5
    assert json.loads(client.calls[0][1])["observations"][0]["target"] == [2, 8]
