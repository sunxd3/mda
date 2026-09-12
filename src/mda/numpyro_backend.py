"""NumPyro model adapter for Appendix D.4.

The callable passed here is trusted Python and is invoked as ``model(ctx)``.  The
adapter does not move the observation likelihood into NumPyro: a model contains
only latent prior sample sites and a deterministic site named ``mu``.

SMC operates in flattened unconstrained coordinates.  :class:`NumPyroPrior`'s
``logpdf`` consequently includes the transform Jacobian.  Candidate MAP ranking
uses its separate physical-space density, because modes are not invariant under
reparameterisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .types import CandidateModel


def _imports():
    try:
        import jax.numpy as jnp
        import jax.random as random
        import numpyro.handlers as handlers
        from numpyro.distributions import biject_to
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("the NumPyro backend requires numpyro and jax") from exc
    return jnp, random, handlers, biject_to


@dataclass(frozen=True)
class _Site:
    name: str
    shape: tuple[int, ...]
    size: int


class NumPyroPrior:
    """Joint NumPyro prior represented in unconstrained sampler coordinates."""

    def __init__(self, model: Callable[[Any], Any], context: Any):
        self.model, self.context = model, context
        trace = self._trace(seed=0)
        sites: list[_Site] = []
        for name, site in trace.items():
            if site["type"] != "sample":
                continue
            if site.get("is_observed", False):
                raise ValueError(f"observed NumPyro site {name!r} is not allowed")
            if getattr(site["fn"].support, "is_discrete", False):
                raise ValueError(f"discrete latent site {name!r} is not supported")
            value = np.asarray(site["value"])
            _, _, _, biject_to = _imports()
            unconstrained = np.asarray(biject_to(site["fn"].support).inv(site["value"]))
            if unconstrained.shape != value.shape:
                raise ValueError(
                    f"latent site {name!r} changes dimension under its support transform "
                    "(for example, simplex latents); this backend does not support it"
                )
            sites.append(_Site(name, value.shape, value.size))
        self._sites = tuple(sites)
        flat_names = []
        for site in sites:
            flat_names.extend(
                site.name if site.size == 1 else f"{site.name}[{i}]"
                for i in range(site.size)
            )
        self.names = tuple(flat_names)

    @property
    def dimension(self) -> int:
        return sum(site.size for site in self._sites)

    @property
    def proposal_scale_hint(self) -> np.ndarray:
        return np.ones(self.dimension)

    def _trace(self, seed: int, values: Mapping[str, Any] | None = None):
        _, random, handlers, _ = _imports()
        fn = self.model
        if values is not None:
            fn = handlers.substitute(fn, data=values)
        return handlers.trace(handlers.seed(fn, random.PRNGKey(seed))).get_trace(self.context)

    def _physical(self, theta: np.ndarray) -> dict[str, Any]:
        jnp, _, _, biject_to = _imports()
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.dimension,):
            raise ValueError(f"theta must have shape ({self.dimension},)")
        values, offset = {}, 0
        for spec in self._sites:
            unconstrained = jnp.asarray(theta[offset : offset + spec.size].reshape(spec.shape))
            # Re-run with all preceding physical values substituted. Supports may
            # depend on earlier sites, so a seed-0 prototype is not authoritative.
            trace = self._trace(0, values)
            values[spec.name] = biject_to(trace[spec.name]["fn"].support)(unconstrained)
            offset += spec.size
        return values

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        _, _, _, biject_to = _imports()
        result = np.empty((count, self.dimension))
        for row in range(count):
            trace = self._trace(int(rng.integers(0, 2**31 - 1)))
            offset = 0
            for spec in self._sites:
                transform = biject_to(trace[spec.name]["fn"].support)
                value = np.asarray(transform.inv(trace[spec.name]["value"]), dtype=float).reshape(-1)
                result[row, offset : offset + spec.size] = value
                offset += spec.size
        return result

    def _densities(self, theta: np.ndarray) -> tuple[float, float]:
        _, _, _, biject_to = _imports()
        values = self._physical(theta)
        trace = self._trace(0, values)
        physical = jacobian = 0.0
        offset = 0
        for spec in self._sites:
            site = trace[spec.name]
            lp = np.asarray(site["fn"].log_prob(site["value"]), dtype=float).sum()
            z = np.asarray(theta[offset : offset + spec.size]).reshape(spec.shape)
            ladj = biject_to(site["fn"].support).log_abs_det_jacobian(z, site["value"])
            physical += float(lp)
            jacobian += float(np.asarray(ladj).sum())
            offset += spec.size
        return physical + jacobian, physical

    def logpdf(self, theta: np.ndarray) -> float:
        try:
            return self._densities(theta)[0]
        except (ValueError, FloatingPointError):
            return -np.inf

    def physical_logpdf(self, theta: np.ndarray) -> float:
        """Joint density of the represented physical latent values (no Jacobian)."""
        try:
            return self._densities(theta)[1]
        except (ValueError, FloatingPointError):
            return -np.inf

    def physical_values(self, theta: np.ndarray) -> Mapping[str, np.ndarray]:
        return {key: np.asarray(value) for key, value in self._physical(theta).items()}


def compile_numpyro_model(
    model: Callable[[Any], Any], action_features: Any, *, name: str = "numpyro"
) -> CandidateModel:
    """Compile a trusted NumPyro callable into a :class:`CandidateModel`.

    ``action_features`` is used to validate that ``mu`` predicts every one of its
    C rows. Calls to the resulting predictor may use a different C-row context.
    """
    if not callable(model):
        raise TypeError("model must be a trusted callable; source strings are not accepted")
    features = np.asarray(action_features)
    if features.ndim == 0:
        raise ValueError("action_features must have a leading C dimension")
    model_context = {"features": features}
    prior = NumPyroPrior(model, model_context)

    def predict(theta: np.ndarray, context: Any) -> np.ndarray:
        feature_rows = np.asarray(context["features"] if isinstance(context, Mapping) else context)
        run_context = {"features": feature_rows}
        values = prior._physical(theta)
        trace = prior._trace(0, values) if run_context["features"] is features else None
        if trace is None:
            _, random, handlers, _ = _imports()
            fn = handlers.substitute(model, data=values)
            trace = handlers.trace(handlers.seed(fn, random.PRNGKey(0))).get_trace(run_context)
        if "mu" not in trace or trace["mu"]["type"] != "deterministic":
            raise ValueError("NumPyro model must define numpyro.deterministic('mu', value)")
        mu = np.asarray(trace["mu"]["value"], dtype=float)
        rows = feature_rows.shape[0]
        if mu.ndim == 0 or mu.shape[0] != rows or np.any(~np.isfinite(mu)):
            raise ValueError(f"mu must be finite with leading shape ({rows},), got {mu.shape}")
        return mu

    # Validate eagerly, including missing/invalid mu.
    probe = prior.sample(np.random.default_rng(0), 1)[0]
    predict(probe, action_features)
    return CandidateModel(
        name=name,
        prior=prior,
        predict=predict,
        structure=name,
        metadata={"backend": "numpyro", "map_log_prior": prior.physical_logpdf},
    )


def compile_numpyro_source(
    source: str,
    action_features: Any,
    *,
    allow_code_execution: bool = False,
    function_name: str = "model",
    namespace: dict[str, Any] | None = None,
    name: str = "numpyro",
) -> CandidateModel:
    """Explicit opt-in bridge for structured proposer output.

    This executes arbitrary Python with the process's full privileges; it is **not
    sandboxed**. The default is therefore rejection. Callers must establish their
    own trust boundary before setting ``allow_code_execution=True``.
    """
    if not allow_code_execution:
        raise PermissionError("source execution is disabled; pass allow_code_execution=True only for trusted code")
    scope = {} if namespace is None else dict(namespace)
    exec(compile(source, "<numpyro-proposal>", "exec"), scope)  # noqa: S102
    model = scope.get(function_name)
    if not callable(model):
        raise ValueError(f"source must define callable {function_name!r}")
    return compile_numpyro_model(model, action_features, name=name)
