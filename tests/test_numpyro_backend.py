from math import lgamma

import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro")
import numpyro.distributions as dist

from mda.numpyro_backend import compile_numpyro_model, compile_numpyro_source


def test_beta_density_in_unconstrained_coordinates_and_prediction():
    def model(ctx):
        coefficient = numpyro.sample("coefficient", dist.Beta(2.0, 3.0))
        numpyro.deterministic("mu", coefficient * ctx["features"][:, 0])

    candidate = compile_numpyro_model(model, np.ones((3, 1)))
    z = np.asarray([0.4])
    x = 1 / (1 + np.exp(-z[0]))
    beta_logpdf = lgamma(5) - lgamma(2) - lgamma(3) + np.log(x) + 2 * np.log(1 - x)
    assert np.isclose(candidate.prior.logpdf(z), beta_logpdf + np.log(x * (1 - x)), atol=1e-6)
    assert np.isclose(candidate.prior.physical_logpdf(z), beta_logpdf, atol=1e-6)
    assert candidate.predict(z, np.arange(4.0)[:, None]).shape == (4,)


def test_plated_latent_is_flattened_and_mu_covers_all_rows():
    def model(ctx):
        with numpyro.plate("features", 2):
            weights = numpyro.sample("weights", dist.Normal(0, 1))
        numpyro.deterministic("mu", ctx["features"] @ weights)

    candidate = compile_numpyro_model(model, np.ones((5, 2)))
    assert candidate.prior.names == ("weights[0]", "weights[1]")
    theta = candidate.prior.sample(np.random.default_rng(2), 7)
    assert theta.shape == (7, 2)
    assert np.allclose(candidate.predict(theta[0], np.eye(2)), theta[0])


def test_rejects_observations_discrete_sites_bad_mu_and_source_by_default():
    def observed(ctx):
        numpyro.sample("y", dist.Normal(), obs=np.zeros(len(ctx)))
        numpyro.deterministic("mu", np.zeros(len(ctx)))

    def discrete(ctx):
        numpyro.sample("k", dist.Bernoulli(0.5))
        numpyro.deterministic("mu", np.zeros(len(ctx)))

    def bad_mu(ctx):
        numpyro.sample("x", dist.Normal())
        numpyro.deterministic("mu", 0.0)

    for model in (observed, discrete, bad_mu):
        with pytest.raises(ValueError):
            compile_numpyro_model(model, np.ones((2, 1)))
    with pytest.raises(PermissionError):
        compile_numpyro_source("def model(ctx): pass", np.ones((2, 1)))


def test_dependent_support_uses_substituted_parent_and_simplex_is_rejected():
    def dependent(ctx):
        upper = numpyro.sample("upper", dist.LogNormal(0, 1))
        child = numpyro.sample("child", dist.Uniform(0, upper))
        numpyro.deterministic("mu", child * np.ones(len(ctx["features"])))

    candidate = compile_numpyro_model(dependent, np.ones((2, 1)))
    values = candidate.prior.physical_values(np.array([3.0, 0.0]))
    assert values["upper"] == pytest.approx(np.exp(3.0))
    assert values["child"] == pytest.approx(np.exp(3.0) / 2)

    def simplex(ctx):
        weights = numpyro.sample("weights", dist.Dirichlet(np.ones(3)))
        numpyro.deterministic("mu", weights[0] * np.ones(len(ctx["features"])))

    with pytest.raises(ValueError, match="changes dimension"):
        compile_numpyro_model(simplex, np.ones((2, 1)))


def test_non_finite_mu_is_a_non_finite_prediction_and_mu_must_be_one_row_per_feature_row():
    from mda.types import NonFinitePrediction

    def ratio(ctx):
        scale = numpyro.sample("scale", dist.Normal())
        numpyro.deterministic("mu", scale / ctx["features"][:, 0])

    candidate = compile_numpyro_model(ratio, np.ones((2, 1)))
    with pytest.raises(NonFinitePrediction):
        candidate.observation_prediction(np.asarray([1.0]), np.zeros((2, 1)))

    def matrix(ctx):
        scale = numpyro.sample("scale", dist.Normal())
        numpyro.deterministic("mu", scale * ctx["features"])

    with pytest.raises(ValueError, match="shape"):
        compile_numpyro_model(matrix, np.ones((2, 2)))
