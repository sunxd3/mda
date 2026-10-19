from __future__ import annotations

import numpy as np
import pytest

from mda.optimizers import CMAESOptimizer


def test_cma_es_converges_on_asymmetric_quadratic():
    optimizer = CMAESOptimizer([-4.0, -20.0], [8.0, 5.0], max_evaluations=180)

    result = optimizer(lambda x: -((x[0] - 2.25) ** 2 + 0.08 * (x[1] + 7.0) ** 2), np.random.default_rng(3))

    np.testing.assert_allclose(result, [2.25, -7.0], atol=0.2)


def test_log_axis_and_decode_are_applied_before_scoring():
    seen: list[dict[str, float]] = []
    optimizer = CMAESOptimizer(
        [1e-3, 0.0],
        [1e3, 2.0],
        log_axes=[True, False],
        decode=lambda x: {"rate": float(x[0]), "level": float(x[1])},
        max_evaluations=160,
    )

    def objective(action: dict[str, float]) -> float:
        seen.append(action)
        return -(np.log10(action["rate"]) - 2.0) ** 2 - (action["level"] - 0.4) ** 2

    result = optimizer(objective, np.random.default_rng(9))

    assert len(seen) == 160
    assert 1e-3 <= result["rate"] <= 1e3
    assert result["rate"] == pytest.approx(100.0, rel=0.08)
    assert result["level"] == pytest.approx(0.4, abs=0.06)


def test_cma_es_is_deterministic_for_equal_rng_seeds():
    optimizer = CMAESOptimizer([-2.0, -2.0], [2.0, 2.0], max_evaluations=47)
    score = lambda x: -float(np.sum((x - np.array([0.3, -0.8])) ** 2))

    first = optimizer(score, np.random.default_rng(1234))
    second = optimizer(score, np.random.default_rng(1234))

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("lower", "upper", "kwargs"),
    [
        ([0.0], [0.0], {}),
        ([0.0], [np.inf], {}),
        ([0.0, 1.0], [1.0], {}),
        ([-1.0], [1.0], {"log_axes": [True]}),
        ([1.0], [2.0], {"log_axes": [True, False]}),
        ([0.0], [1.0], {"max_evaluations": 0}),
    ],
)
def test_invalid_bounds_and_options_raise(lower, upper, kwargs):
    with pytest.raises(ValueError):
        CMAESOptimizer(lower, upper, **kwargs)


def test_no_finite_score_raises_and_respects_budget():
    calls = 0

    def score(_):
        nonlocal calls
        calls += 1
        return np.nan

    with pytest.raises(ValueError, match="finite score"):
        CMAESOptimizer([0.0], [1.0], max_evaluations=3)(score, np.random.default_rng(0))
    assert calls == 3


def test_flat_score_still_spends_the_full_budget():
    calls = 0

    def score(_):
        nonlocal calls
        calls += 1
        return 0.0

    CMAESOptimizer([0.0] * 7, [1.0] * 7, max_evaluations=48)(score, np.random.default_rng(0))
    assert calls == 48
