import numpy as np
import pytest

from mda.expressions import compile_expression


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
