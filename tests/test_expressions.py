import json
from types import SimpleNamespace

import numpy as np
import pytest

from mda.checks import residual_report
from mda.expressions import SymbolicRegressionProposer, compile_expression
from mda.posterior import ModelFit
from mda.types import CandidateModel, Datum, UniformPrior


def test_compiled_expression_uses_declared_variables_and_parameters():
    fn = compile_expression("a + b*x**2", ("x",), ("a", "b"))
    assert fn(np.asarray([1.0, 2.0]), 3.0) == 19.0


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('id')", "np.exp(x)", "x.__class__", "open('/tmp/x')"],
)
def test_compiler_rejects_code_execution(expression):
    with pytest.raises((SyntaxError, ValueError)):
        compile_expression(expression, ("x",), ())


@pytest.mark.parametrize(
    ("variables", "parameters"),
    [
        (("not valid",), ()),
        (("for",), ()),
        (("x",), ("x",)),
        (("exp",), ()),
        (("x",), ("pi",)),
        (("x", "x"), ()),
    ],
)
def test_compiler_rejects_ambiguous_declared_names(variables, parameters):
    with pytest.raises(ValueError):
        compile_expression("0", variables, parameters)


def test_residual_report_adds_signed_coordinate_median_without_losing_absolute_json():
    calls = 0

    def predict(theta, action):
        nonlocal calls
        calls += 1
        return np.asarray(action)

    model = CandidateModel(
        "vector", UniformPrior((), np.asarray([]), np.asarray([])), predict,
        structure="vector structure",
    )
    fit = ModelFit(
        model, SimpleNamespace(map_parameters=np.asarray([])), log_score=0.0
    )
    report = residual_report(
        fit, [Datum(np.asarray([1.0, 8.0]), np.asarray([2.0, 4.0]))], floor=0.1
    )

    assert report == {
        "name": "vector", "structure": "vector structure",
        "median_relative_residual": 0.75, "per_experiment": [0.75],
        "signed_per_experiment": [-0.25],
    }
    assert calls == 1


def test_symbolic_proposer_preserves_per_experiment_residuals():
    class Client:
        def ask_json(self, system, prompt, **kwargs):
            self.prompt = prompt
            return {"hypotheses": []}

    client = Client()
    reports = [
        {"name": "candidate", "structure": "a*x", "median_relative_residual": 0.4,
         "per_experiment": [0.1, 0.7]},
        {"name": "candidate", "structure": "b*x**2", "median_relative_residual": 0.4,
         "per_experiment": [0.7, 0.1]},
    ]
    result = SymbolicRegressionProposer(client).propose(
        context="Unknown response",
        dataset=[Datum(2.0, np.asarray(3.0)), Datum(8.0, np.asarray(12.0))],
        current_models=(), residuals=reports, count=2,
    )

    assert result == []
    sent = json.loads(next(line for line in client.prompt.splitlines() if line.startswith("[{")))
    assert sent == reports
    assert "2.0 -> 3.0" in client.prompt
    assert "8.0 -> 12.0" in client.prompt
    assert "per_experiment residuals follow the experiment order above" in client.prompt


def test_symbolic_proposer_reports_signed_input_correlations_and_full_model_pool():
    class Client:
        def ask_json(self, system, prompt, **kwargs):
            self.prompt = prompt
            return {"hypotheses": []}

    prior = UniformPrior(("a",), np.asarray([-2.0]), np.asarray([2.0]))
    models = [
        CandidateModel("duplicate", prior, lambda theta, action: 0.0, structure="a*x"),
        CandidateModel("duplicate", prior, lambda theta, action: 0.0, structure="a/x"),
    ]
    reports = [{
        "name": "duplicate", "structure": "a*x", "median_relative_residual": 2.0,
        "per_experiment": [3.0, 2.0, 1.0],
        "signed_per_experiment": [3.0, 2.0, 1.0],
    }]
    dataset = [
        Datum({"x": x, "constant": 7.0}, np.asarray(0.0))
        for x in (1.0, 2.0, 3.0)
    ]
    client = Client()
    SymbolicRegressionProposer(client, variable_names=("x", "constant")).propose(
        context="Unknown response", dataset=dataset, current_models=models,
        residuals=reports, count=1,
    )

    json_lines = [json.loads(line) for line in client.prompt.splitlines() if line.startswith("[{")]
    structures, sent_reports, correlations = json_lines
    assert [item["structure"] for item in structures] == ["a*x", "a/x"]
    assert structures[0]["uniform_parameter_priors"] == {"a": [-2.0, 2.0]}
    assert sent_reports == reports
    assert correlations == [{
        "structure": "a*x", "input_signed_residual_correlations": {"x": -1.0}
    }]


def test_symbolic_proposer_drops_malformed_and_overflowing_laws():
    hypotheses = [
        {"name": "syntax", "expr": "a*(x - ", "params": [{"name": "a", "low": 0.1, "high": 2.0}]},
        {"name": "log zero", "expr": "a*log(y)", "params": [{"name": "a", "low": 0.1, "high": 2.0}]},
        {"name": "valid", "expr": "a*x", "params": [{"name": "a", "low": 0.1, "high": 2.0}]},
    ]

    class Client:
        def ask_json(self, system, prompt, **kwargs):
            return {"hypotheses": hypotheses}

    dataset = [Datum({"x": x, "y": 0.0}, np.asarray(x)) for x in (1.0, 2.0, 3.0)]
    models = SymbolicRegressionProposer(Client(), variable_names=("x", "y")).propose(
        context="", dataset=dataset, current_models=(), residuals=(), count=3,
    )
    assert [model.name for model in models] == ["valid"]


def test_underflow_is_a_tiny_value_not_an_invalid_law():
    fn = compile_expression("a*exp(-b*x)", ("x",), ("a", "b"))
    value = fn(np.asarray([1.0, 740.0]), 1.0)
    assert 0.0 <= value < 1e-300


@pytest.mark.parametrize("expression", ["exp(r, r)", "maximum(r)", "sqrt()", "where(r)"])
def test_calls_with_the_wrong_argument_count_are_rejected(expression):
    # A surplus positional argument would reach a ufunc's ``out`` slot and
    # overwrite the caller's array; ``where`` cannot be used without comparisons.
    with pytest.raises(ValueError):
        compile_expression(expression, ("r",), ())


def test_integer_power_towers_overflow_immediately_instead_of_stalling():
    import time

    fn = compile_expression("9**9**9 + x", ("x",), ())
    started = time.perf_counter()
    with pytest.raises(OverflowError):
        fn(np.zeros(0), 1.0)
    assert time.perf_counter() - started < 1.0


def test_canonical_expression_normalizes_spacing_and_parentheses():
    from mda.expressions import canonical_expression

    assert canonical_expression("a*x") == canonical_expression(" a * x ") == canonical_expression("(a*x)")
    with pytest.raises(ValueError):
        canonical_expression("a*(x")


def test_screening_keeps_laws_valid_on_part_of_their_prior_and_dedupes_spacing():
    data = [Datum({"x": x}, np.asarray(1.0)) for x in (310.0, 350.0, 390.0)]
    hypotheses = [
        # Valid only where x0 <= 310: one unlucky prior draw must not reject it.
        {"name": "sqrt", "expr": "k*sqrt(x - x0)",
         "params": [{"name": "k", "low": 0.1, "high": 2.0}, {"name": "x0", "low": 300.0, "high": 340.0}]},
        {"name": "lin", "expr": "a*x", "params": [{"name": "a", "low": 0.0, "high": 1.0}]},
        {"name": "lin again", "expr": "a * x", "params": [{"name": "a", "low": 0.0, "high": 1.0}]},
        # Invalid everywhere on its prior.
        {"name": "never", "expr": "sqrt(-x)*k", "params": [{"name": "k", "low": 0.1, "high": 2.0}]},
    ]

    class Client:
        def ask_json(self, system, prompt, **kwargs):
            return {"hypotheses": hypotheses}

    models = SymbolicRegressionProposer(Client(), variable_names=("x",)).propose(
        context="", dataset=data, current_models=[], residuals=[], count=4,
    )
    assert [m.name for m in models] == ["sqrt", "lin"]
    assert models[1].structure == "a * x"
