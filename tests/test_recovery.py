import numpy as np
import pytest

from mda.benchmarks.recovery import canonical_symbolic_candidate, evaluate_recovery
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


def test_canonical_form_drives_refitting_and_discriminates_bad_form():
    prior = UniformPrior(("a", "b"), np.array([0.1, 0.0]), np.array([5.0, 4.0]))
    equivalent = canonical_symbolic_candidate("b + x*a + a*x", ("x",), prior)
    wrong = canonical_symbolic_candidate("a*x**2 + b", ("x",), prior)
    fit, holdout = np.linspace(.1, 2, 30), np.linspace(2.1, 4, 30)
    truth = lambda x: 3*x + 1.25

    parameters, rmsle, recovered = evaluate_recovery(
        equivalent, (fit, holdout), (truth, truth), tolerance=1e-6
    )
    assert equivalent.structure == "2*a*x + b"
    assert parameters == pytest.approx([1.5, 1.25], rel=1e-6)
    assert rmsle < 1e-7 and recovered
    assert not evaluate_recovery(wrong, (fit, holdout), (truth, truth), tolerance=.01)[2]


def test_symbolic_parser_rejects_malicious_syntax_without_execution(tmp_path):
    marker = tmp_path / "executed"
    expression = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    prior = UniformPrior((), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="approved|unsupported|undeclared"):
        canonical_symbolic_candidate(expression, ("x",), prior)
    assert not marker.exists()


def test_symbolic_special_functions_and_numpy_aliases_match_scipy():
    scipy_special = pytest.importorskip("scipy.special")
    prior = UniformPrior(("a",), np.array([.1]), np.array([2.0]))
    model = canonical_symbolic_candidate(
        "a*(np.k0(x) + k1(x) + gamma(x))", ("x",), prior
    )
    x = np.array([.5, 1.25, 2.0])
    expected = .7 * (scipy_special.k0(x) + scipy_special.k1(x) + scipy_special.gamma(x))
    assert model.predict(np.array([.7]), x) == pytest.approx(expected)


def test_recovery_threshold_is_strict():
    prior = UniformPrior((), np.array([]), np.array([]))
    model = canonical_symbolic_candidate("0", ("x",), prior)
    _, rmsle, recovered = evaluate_recovery(
        model, ([0.0], [1.0]), ([0.0], [np.e - 1.0]), tolerance=1.0
    )
    assert rmsle == pytest.approx(1.0)
    assert not recovered


def test_refit_skips_starts_where_the_law_is_undefined():
    # sqrt(x - c) is undefined for c above the smallest input; the optimizer
    # may visit such c on its way, which must not abort the recovery.
    model = _model("a*sqrt(x - c)", ("a", "c"), (0.1, -1.0), (5.0, 2.5), )
    fit = np.linspace(1.0, 4.0, 25)
    holdout = np.linspace(1.1, 3.9, 20)
    truth = lambda x: 2.0 * np.sqrt(x - 0.5)
    parameters, rmsle, recovered = evaluate_recovery(model, (fit, holdout), (truth, truth), seed=1)
    assert recovered and parameters == pytest.approx([2.0, 0.5], rel=1e-4)
    undefined = _model("a*sqrt(x - c)", ("a", "c"), (0.1, 10.0), (5.0, 20.0))
    with pytest.raises(ValueError, match="finite residuals"):
        evaluate_recovery(undefined, (fit, holdout), (truth, truth), seed=1)
