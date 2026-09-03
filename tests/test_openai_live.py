import os

import numpy as np
import pytest

from mda.expressions import SymbolicRegressionProposer
from mda.openai_client import OpenAIJSONClient
from mda.types import Datum


pytestmark = pytest.mark.live


@pytest.mark.skipif(os.environ.get("RUN_LIVE_OPENAI") != "1", reason="set RUN_LIVE_OPENAI=1")
def test_gpt_5_6_luna_returns_executable_scientific_models():
    proposer = SymbolicRegressionProposer(OpenAIJSONClient(model="gpt-5.6-luna"))
    data = [Datum(x, np.asarray(0.25 + 1.5 * x**2)) for x in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    models = proposer.propose(
        context="A noise-free scalar response follows a smooth low-order algebraic law y=f(x).",
        dataset=data,
        current_models=[],
        residuals=[],
        count=4,
    )
    assert len(models) >= 2
    for model in models:
        theta = model.prior.sample(np.random.default_rng(0), 1)[0]
        assert np.isfinite(model.observation_prediction(theta, 0.5))
