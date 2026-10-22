"""A small episodic domain where within-episode feedback matters (program-design tests).

A device trips at an unknown threshold. Below it the response is 0; for
inputs at most ``window`` above it, mechanism ``rise`` responds +1 and
mechanism ``fall`` responds -1; further above, both saturate at 2. Noise is
Gaussian with SD ``std``. The threshold is an ordinary model parameter, not a
per-episode latent. Only inputs in the narrow window tell the mechanisms
apart, so a program that brackets the threshold from what it has seen in the
episode can land there when a fixed protocol, chosen before seeing anything,
mostly cannot.
"""

from __future__ import annotations

import numpy as np

from mda.program_sandbox import compile_program
from mda.programs import EpisodeSpec, FixedProgram, Program, gaussian_step
from mda.types import CandidateModel, UniformPrior

CONTEXT = (
    "A device is driven with a scalar input in [0, 1]. It responds about 0 below an unknown "
    "threshold and saturates at about 2 well above it. Just above the threshold the response "
    "is either about +1 or about -1, depending on the mechanism; noise SD 0.1."
)
EPISODE = EpisodeSpec(steps=5, lower=0.0, upper=1.0)
STD = 0.1
WINDOW = 0.1


def mean_response(threshold: float, inputs, sign: float, window: float = WINDOW) -> np.ndarray:
    inputs = np.asarray(inputs, dtype=float)
    above = inputs - threshold
    return np.where(above <= 0, 0.0, np.where(above <= window, sign, 2.0))


def make_model(name: str, sign: float, *, std: float = STD) -> CandidateModel:
    def predict(theta, inputs):
        return mean_response(theta[0], inputs, sign)

    return CandidateModel(
        name=name,
        structure=f"threshold-{name}",
        prior=UniformPrior(("threshold",), np.asarray([0.2]), np.asarray([0.8])),
        predict=predict,
        sample_step=gaussian_step(predict, std),
    )


def models() -> list[CandidateModel]:
    return [make_model("rise", +1.0), make_model("fall", -1.0)]


class ThresholdEpisodes:
    """The true device, observed one input at a time (``EpisodicEnvironment``).

    ``step(inputs)`` runs a whole fixed protocol, the ordinary mda contract.
    """

    def __init__(self, mechanism: str = "rise", threshold: float = 0.55, *, seed: int = 0) -> None:
        self.sign = {"rise": 1.0, "fall": -1.0}[mechanism]
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)
        self.inputs: list[float] = []

    def start_episode(self) -> None:
        self.inputs = []

    def observe(self, value: float) -> float:
        self.inputs.append(float(value))
        mean = float(mean_response(self.threshold, value, self.sign))
        return mean + STD * float(self.rng.standard_normal())

    def step(self, action) -> np.ndarray:
        self.start_episode()
        return np.asarray([self.observe(value) for value in np.asarray(action, dtype=float)])


BRACKET_SOURCE = '''
class Bracket(Program):
    def step(self, history, belief):
        low = 0.0
        high = 1.0
        for i in range(history.t):
            y = history.obs[i]
            if abs(y) < 0.5:
                low = max(low, history.inputs[i])
            elif y > 1.5:
                high = min(high, history.inputs[i])
            else:
                return history.inputs[i]
        if history.t == 0:
            return belief["pooled"]["threshold"]["mean"] + 0.05
        return (low + high) / 2
'''


def seed_programs(episode: EpisodeSpec = EPISODE) -> list[Program]:
    """The seed library: a fixed spread and one sandboxed bracketing program."""
    return [
        FixedProgram(np.linspace(0.25, 0.85, episode.steps)),
        compile_program(BRACKET_SOURCE, "Bracket"),
    ]
