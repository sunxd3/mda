from __future__ import annotations

import numpy as np

from mda.acquisition import joint_eig, model_eig, task_value_of_information
from mda.posterior import BeliefState, ModelFit
from mda.smc import ParameterPosterior
from mda.types import CandidateModel, UniformPrior


def fit(name, particles, predict):
    particles = np.asarray(particles, dtype=float)
    prior = UniformPrior(
        tuple(f"p{i}" for i in range(particles.shape[1])),
        np.full(particles.shape[1], -10.0),
        np.full(particles.shape[1], 10.0),
    )
    candidate = CandidateModel(name, prior, predict, structure=name)
    posterior = ParameterPosterior(
        particles=particles,
        weights=np.full(len(particles), 1 / len(particles)),
        log_likelihoods=np.zeros(len(particles)),
        log_evidence=0.0,
        temperatures=(0.0, 1.0),
        acceptance_rate=1.0,
    )
    return ModelFit(candidate, posterior, 0.0)


def test_model_eig_prefers_where_structures_disagree():
    left = fit("left", [[0.0]], lambda theta, x: np.asarray(-x))
    right = fit("right", [[0.0]], lambda theta, x: np.asarray(x))
    belief = BeliefState((left, right), np.asarray([0.5, 0.5]))
    assert model_eig(3.0, belief, 1.0) > model_eig(0.5, belief, 1.0)


def test_joint_eig_retains_parameter_uncertainty():
    uncertain = fit("slope", [[-1.0], [1.0]], lambda theta, x: np.asarray(theta[0] * x))
    belief = BeliefState((uncertain,), np.asarray([1.0]))
    assert model_eig(2.0, belief, 1.0) == 0.0
    assert joint_eig(2.0, belief, 1.0) == 4.0


def test_task_voi_is_larger_for_query_correlated_design():
    uncertain = fit("slope", [[-1.0], [1.0]], lambda theta, x: np.asarray(theta[0] * x))
    belief = BeliefState((uncertain,), np.asarray([1.0]))
    near = task_value_of_information(2.0, [2.0], belief, 0.1)
    null = task_value_of_information(0.0, [2.0], belief, 0.1)
    assert near > null
    assert null == 0.0
