from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mda.acquisition import (
    choose_design, full_observation_task_value_of_information,
    joint_eig, model_eig, task_value_of_information,
)
from mda.posterior import BeliefState, ModelFit, fit_model_posterior
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


def test_zero_mass_fit_stays_in_pool_but_not_in_design_scores():
    good = fit("good", [[0.0]], lambda theta, x: np.asarray(1.0))
    invalid = fit("invalid", [[0.0]], lambda theta, x: np.asarray(np.nan))
    belief = BeliefState((good, invalid), np.asarray([1.0, 0.0]))

    assert belief.fits == (good, invalid)
    assert belief.model_weights.tolist() == [1.0, 0.0]
    assert model_eig(0.0, belief, 1.0) == 0.0
    assert joint_eig(0.0, belief, 1.0) == 0.0


def test_model_posterior_keeps_underflowed_models_and_drops_failed_fits():
    from mda.smc import SMCConfig

    def evidence_for(scores):
        def evidence(dataset, model, likelihood, rng):
            return replace(fit(model.name, [[0.0]], model.predict).parameters,
                           log_evidence=scores[model.name])
        return evidence

    models = [fit(name, [[0.0]], lambda theta, x: np.asarray(0.0)).model for name in ("best", "far", "failed")]
    belief = fit_model_posterior(
        [], models, None, SMCConfig(), model_cap=16, occam_penalty=0.0, rng=np.random.default_rng(0),
        evidence=evidence_for({"best": 0.0, "far": -2000.0, "failed": -np.inf}),
    )
    assert [f.model.name for f in belief.fits] == ["best", "far"]
    assert belief.model_weights.tolist() == [1.0, 0.0]


def test_design_skips_actions_a_weighted_model_cannot_predict():
    left = fit("left", [[0.0]], lambda theta, x: np.asarray(np.inf if x == "bad" else 0.0))
    right = fit("right", [[0.0]], lambda theta, x: np.asarray(1.0))
    belief = BeliefState((left, right), np.asarray([0.5, 0.5]))
    action, scores = choose_design(["bad", "good"], belief, flavour="model", observation_covariance=1.0)
    assert action == "good"
    assert scores[0] == -np.inf


def test_task_uses_targets_at_design_not_raw_observations():
    uncertain = fit("separate targets", [[-1.0], [1.0]], lambda theta, x: theta)
    uncertain = replace(uncertain, model=replace(
        uncertain.model, predict_target=lambda theta, x: theta if x == "query" else np.zeros(1),
    ))
    belief = BeliefState((uncertain,), np.asarray([1.0]))
    assert task_value_of_information("design", ["query"], belief, 1.0) == 0.0
    assert full_observation_task_value_of_information("design", ["query"], belief, 1.0) == 0.5


def test_weighted_vector_acquisitions_distinguish_coordinatewise_and_full_conditioning():
    def prediction(theta, action):
        return theta if action == "design" else np.asarray([2 * theta[0] - theta[1], -theta[0] + 3 * theta[1]])

    left = fit("left", [[0, 1], [2, -1]], prediction)
    right = fit("right", [[1, 4], [3, 0]], prediction)
    left = replace(left, parameters=replace(left.parameters, weights=np.asarray([0.8, 0.2])))
    right = replace(right, parameters=replace(right.parameters, weights=np.asarray([0.4, 0.6])))
    belief = BeliefState((left, right), np.asarray([0.25, 0.75]))
    noise = np.asarray([[2, 1], [1, 3]])
    # Independent rational arithmetic: pooled weights (1/5,1/20,3/10,9/20),
    # mean (7/4,27/20), covariance ((119/80,-101/80),(-101/80,1291/400)).
    assert model_eig("design", belief, noise) == pytest.approx(609 / 2000)
    assert joint_eig("design", belief, noise) == pytest.approx(5377 / 2000)
    score = task_value_of_information("design", ["query"], belief, np.asarray([2, 3]), np.asarray([5, 7]))
    assert score == pytest.approx(816826457 / 216218800)
    full = full_observation_task_value_of_information("design", ["query"], belief, np.asarray([2, 3]), np.asarray([5, 7]))
    assert not np.isclose(score, full)
    # Default loss whitening is by noise variance; explicit unit scale gives
    # unwhitened Eq.43. A second identical query must not double the objective.
    assert task_value_of_information("design", ["query", "query"], belief, 2.0) == pytest.approx(
        task_value_of_information("design", ["query"], belief, 2.0, 1.0) / 2
    )
    with pytest.raises(ValueError, match="diagonal"):
        task_value_of_information("design", ["query"], belief, noise)


def test_independent_trace_noise_never_uses_dense_linear_algebra(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("dense solve on independent trace noise")

    monkeypatch.setattr(np.linalg, "solve", fail)
    left = fit("left", [[0]], lambda theta, action: np.zeros(10_000))
    right = fit("right", [[0]], lambda theta, action: np.full(10_000, 2.0))
    belief = BeliefState((left, right), np.asarray([0.25, 0.75]))
    assert model_eig(None, belief, 2.0) == pytest.approx(3750.0)
    assert joint_eig(None, belief, np.full(10_000, 2.0)) == pytest.approx(3750.0)
