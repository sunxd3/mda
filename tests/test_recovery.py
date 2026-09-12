import numpy as np
import pytest

from mda.benchmarks.recovery import evaluate_recovery
from mda.expressions import compile_expression
from mda.types import CandidateModel, UniformPrior


def _model(expression, names, lower, upper):
    return CandidateModel(
        expression,
        UniformPrior(names, np.asarray(lower), np.asarray(upper)),
        compile_expression(expression, ("x",), names),
        structure=expression,
    )


def test_wrong_constants_of_same_nonlinear_form_are_recovered_without_mutation():
    model = _model("a*x**2 + b", ("a", "b"), (0.1, 0.0), (8.0, 6.0))
    fit = np.linspace(0.1, 2.5, 31)
    holdout = np.linspace(0.15, 2.45, 29)
    truth = lambda x: 2.75 * x**2 + 1.4
    lower_before, upper_before = model.prior.lower.copy(), model.prior.upper.copy()

    parameters, rmsle, recovered = evaluate_recovery(
        model, (fit, holdout), (truth, truth), seed=4
    )

    assert parameters == pytest.approx([2.75, 1.4], rel=1e-6)
    assert rmsle < 1e-7
    assert recovered
    assert np.array_equal(model.prior.lower, lower_before)
    assert np.array_equal(model.prior.upper, upper_before)
    assert model.structure == "a*x**2 + b"


def test_incompatible_saturating_form_does_not_recover_quadratic():
    model = _model("a*x/(b + x)", ("a", "b"), (0.01, 0.01), (10.0, 10.0))
    fit = np.linspace(0.1, 4.0, 50)
    holdout = np.linspace(4.1, 8.0, 50)
    truth = lambda x: 0.5 + 0.8 * x**2

    _, rmsle, recovered = evaluate_recovery(
        model, {"fit": fit, "holdout": holdout},
        {"fit": truth, "holdout": truth}, tolerance=0.02,
    )

    assert rmsle > 0.02
    assert not recovered


def test_signed_data_requires_explicit_transform():
    model = _model("a*x", ("a",), (-2.0,), (2.0,))
    grids = ([-2.0, -1.0, 1.0], [-1.5, 1.5])
    truth = (lambda x: -x, lambda x: -x)

    with pytest.raises(ValueError, match="signed_log1p"):
        evaluate_recovery(model, grids, truth)
    parameters, rmsle, recovered = evaluate_recovery(
        model, grids, truth, signed_log1p=True
    )
    assert parameters == pytest.approx([-1.0], abs=1e-7)
    assert rmsle < 1e-7
    assert recovered
