"""Reproduction adapter for the seven public BoxingGym scientific domains.

The equations and priors are transcribed (not copied) from BoxingGym commit
``b43e38cb03d09c13efa9cf4d9bae740d51157bfd``, files
``src/boxing_gym/envs/{dugongs,death_process,peregrines,
hyperbolic_temporal_discount,survival_analysis,irt,location_finding}.py``.
BoxingGym is MIT licensed (Copyright its contributors).

This small reconstruction deliberately fixes benchmark plumbing problems:
random state is private and seeded, while the IRT response cells and survival
patients are sampled once and scored as fixed outcomes with Brier loss.  The
other domains use clean conditional means.  Table 8 specifies a Gaussian
dugongs observation (rather than upstream's absolute-error implementation), so
the Gaussian likelihood here is a deliberate paper-over-upstream choice.

The paper does not specify finite-pool sizes for generic random-input domains;
24 design candidates and 12 queries are local, configurable defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, lgamma
from typing import Any

import numpy as np

from mda.agent import AgentConfig, ModelDiscoveryAgent
from mda.expressions import SymbolicRegressionProposer
from mda.numpyro_backend import compile_numpyro_source
from mda.smc import SMCConfig
from mda.types import CandidateModel, Datum, UniformPrior

DOMAINS = ("dugongs", "death", "peregrines", "hyperbolic", "survival", "irt", "location")


def _scalar_prediction(model: CandidateModel, theta: np.ndarray, action: Any) -> float:
    value = np.asarray(model.observation_prediction(theta, action), dtype=float)
    if value.size != 1:
        raise ValueError("Boxing likelihoods require a scalar prediction")
    return float(value.reshape(-1)[0])


@dataclass(frozen=True)
class BoxingLikelihood:
    """Exact scalar Gaussian, Bernoulli, Binomial, or Poisson log likelihood."""

    family: str
    scale: float = 1.0
    trials: int = 1

    def __call__(self, dataset, model, theta, rng) -> float:
        del rng
        total = 0.0
        try:
            for datum in dataset:
                y = float(np.asarray(datum.observation))
                mu = _scalar_prediction(model, theta, datum.action)
                if self.family == "gaussian":
                    if self.scale <= 0:
                        raise ValueError("scale must be positive")
                    z = (y - mu) / self.scale
                    term = -.5 * (z * z + np.log(2 * np.pi * self.scale**2))
                elif self.family == "bernoulli":
                    if y not in (0, 1) or not 0 <= mu <= 1:
                        raise ValueError
                    term = (np.log(mu) if y else np.log1p(-mu))
                elif self.family == "binomial":
                    if y != int(y) or not 0 <= y <= self.trials or not 0 <= mu <= self.trials:
                        raise ValueError
                    probability = mu / self.trials
                    term = (lgamma(self.trials + 1) - lgamma(y + 1)
                            - lgamma(self.trials - y + 1)
                            + (y * np.log(probability) if y else 0.0)
                            + ((self.trials - y) * np.log1p(-probability)
                               if y < self.trials else 0.0))
                elif self.family == "poisson":
                    if y != int(y) or y < 0 or mu < 0:
                        raise ValueError
                    term = (-mu - lgamma(y + 1) + (y * np.log(mu) if y else 0.0))
                else:
                    raise ValueError(f"unknown observation family {self.family!r}")
                total += float(term)
        except (ValueError, FloatingPointError, OverflowError):
            return -np.inf
        return total if np.isfinite(total) else -np.inf


class BoxingEnvironment:
    """One sampled latent world, with disjoint design and held-out sets."""

    def __init__(self, domain: str, seed: int = 0, candidate_count: int = 24,
                 query_count: int = 12):
        if domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}")
        seed_count = 3 if domain == "location" else 2
        if candidate_count < seed_count or query_count < 1:
            raise ValueError(f"candidate_count >= {seed_count} and query_count >= 1 are required")
        self.domain, self.seed = domain, seed
        latent_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        action_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self._train_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        self._eval_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
        self._truth = self._sample_truth(latent_rng)
        requested = candidate_count + query_count
        if domain == "irt":
            all_actions = self._sample_actions(action_rng, 36)
            if requested > 36: raise ValueError("IRT has only 36 distinct actions")
        elif domain == "survival":
            # H.3.1 uses one fixed cohort of 100 patients. Experiments address
            # rows in this table rather than generating a new patient per call.
            all_actions = self._sample_actions(action_rng, 100)
            if requested > 100: raise ValueError("survival has a fixed cohort of 100 patients")
        else:
            all_actions = self._sample_actions(action_rng, requested)
        if domain in {"irt", "location", "survival"}:
            # NumPyro models receive this complete table and use the action as a
            # row index; held-out rows must be present to define predictions.
            self.action_features = np.asarray(all_actions, dtype=float)
            all_actions = list(range(len(all_actions)))
        else:
            self.action_features = np.asarray(all_actions, dtype=object)
        seed_actions = all_actions[:seed_count]
        self.actions = list(all_actions[:candidate_count])
        self.query_actions = list(all_actions[candidate_count:requested])
        self._fixed_outcomes = None
        if domain in {"irt", "survival"}:
            probabilities = [self.conditional_mean(i) for i in range(len(self.action_features))]
            self._fixed_outcomes = self._eval_rng.binomial(1, probabilities)
        self.initial_data = [Datum(a, np.asarray(self.step(a))) for a in seed_actions]
        self.context = self._context()

    def _sample_truth(self, r):
        if self.domain == "dugongs":
            return (r.normal(2, .2), r.normal(1.5, .5), r.normal(.4, .5))
        if self.domain == "death":
            # scipy truncnorm(0,2; location 1, scale 1)
            while True:
                x = r.normal(1, 1)
                if 0 <= x <= 2: return (x,)
        if self.domain == "peregrines":
            return tuple(r.normal([4.5, 1.2, .07, -.24], [.1, .1, .01, .05]))
        if self.domain == "hyperbolic":
            return (np.exp(r.normal(-4.25, .5)), abs(r.normal(0, 2)))
        if self.domain == "survival":
            return (r.gamma(.1, 10), abs(r.normal(0, 10)))
        if self.domain == "irt":
            return (r.normal(size=6), r.normal(size=6), r.normal(size=6))
        return (r.normal(size=(3, 2)),)

    def _sample_actions(self, r, n):
        if self.domain in {"dugongs", "peregrines"}: return r.uniform(0, 5, n).tolist()
        if self.domain == "death": return r.uniform(np.nextafter(0., 1.), 2, n).tolist()
        if self.domain == "hyperbolic":
            ir = r.integers(1, 200, n); dr = np.array([r.integers(x + 1, 300) for x in ir])
            return [{"immediate": int(a), "delayed": int(b), "days": int(d)}
                    for a, b, d in zip(ir, dr, r.integers(1, 365, n))]
        if self.domain == "survival":
            return [(int(m), float(t))
                    for m, t in zip(r.integers(0, 2, n), r.uniform(0, 10, n))]
        if self.domain == "irt":
            pairs = [(s, q) for s in range(6) for q in range(6)]
            if n > len(pairs): raise ValueError("IRT has only 36 distinct actions")
            return [pairs[i] for i in r.choice(len(pairs), n, replace=False)]
        return [tuple(x) for x in r.uniform(-2, 2, (n, 2))]

    def _context(self):
        ranges = {
            "dugongs": "age x in [0,5]; Gaussian length",
            "death": "time x in (0,2); Binomial(50, p) infection count",
            "peregrines": "time x in [0,5]; Poisson count",
            "hyperbolic": "(immediate, delayed, days), delayed>immediate; Bernoulli choice",
            "survival": "(metastasized in {0,1}, time in [0,10]); Bernoulli death",
            "irt": "(student,question), each 0..5; Bernoulli correctness (2PL IRT)",
            "location": "2-vector in [-2,2]^2; Gaussian superposed source intensity",
        }
        return "Discover the conditional observation law. Inputs: " + ranges[self.domain]

    @property
    def likelihood(self):
        return {"dugongs": BoxingLikelihood("gaussian", .25),
                "death": BoxingLikelihood("binomial", trials=50),
                "peregrines": BoxingLikelihood("poisson"),
                "hyperbolic": BoxingLikelihood("bernoulli"),
                "survival": BoxingLikelihood("bernoulli"),
                "irt": BoxingLikelihood("bernoulli"),
                "location": BoxingLikelihood("gaussian", .5)}[self.domain]

    def conditional_mean(self, action):
        features = (self.action_features[int(action)]
                    if self.domain in {"irt", "location", "survival"} else action)
        if self.domain == "dugongs":
            a, b, l = self._truth; return a - b * abs(l) ** float(features)
        if self.domain == "death": return 50 * (1 - np.exp(-self._truth[0] * float(features)))
        if self.domain == "peregrines":
            a, b1, b2, b3 = self._truth; x = float(features)
            return np.exp(a + b1*x + b2*x*x + b3*x*x*x)
        if self.domain == "hyperbolic":
            ir, dr, days = (features[k] for k in ("immediate", "delayed", "days")); k, alpha = self._truth
            z = (dr / (1 + k*days) - ir) / alpha
            return .01 + .98 * .5 * (1 + erf(z / np.sqrt(2)))
        if self.domain == "survival":
            m, t = features; lam, beta = self._truth
            z = t * np.exp(beta*m) * lam
            return 1 / (1 + np.exp(-z))
        if self.domain == "irt":
            s, q = np.asarray(features, dtype=int); ability, difficulty, discrimination = self._truth
            z = discrimination[q] * (ability[s] - difficulty[q])
            return 1 / (1 + np.exp(-z))
        sources, = self._truth; x = np.asarray(features)
        return .1 + sum(1 / (.0001 + np.sum((source-x)**2)) for source in sources)

    def step(self, action):
        """Draw a training observation; latent truth is never returned."""
        if self.domain in {"irt", "survival"}:
            return int(self._fixed_outcomes[int(action)])
        mean, r = self.conditional_mean(action), self._train_rng
        if self.domain == "death": return int(r.binomial(50, mean / 50))
        if self.domain == "peregrines": return int(r.poisson(mean))
        if self.domain == "hyperbolic": return int(r.binomial(1, mean))
        return float(r.normal(mean, .25 if self.domain == "dugongs" else .5))

    def evaluate_predictor(self, predictor):
        """Paper nMSE, except fixed-outcome IRT/survival Brier loss."""
        pred = np.asarray([float(np.asarray(predictor(a))) for a in self.query_actions])
        means = np.asarray([self.conditional_mean(a) for a in self.query_actions])
        if self.domain in {"irt", "survival"}:
            outcomes = self._fixed_outcomes[np.asarray(self.query_actions, dtype=int)]
            return float(np.mean((pred - outcomes) ** 2))
        variance = float(np.var(means))
        return float(np.mean((pred-means)**2) / max(variance, 1e-12))


def _generic_model(env: BoxingEnvironment) -> CandidateModel:
    family = env.likelihood.family
    def predict(theta, action):
        del action
        if family == "bernoulli": return np.asarray(1/(1+np.exp(-theta[0])))
        if family == "binomial": return np.asarray(env.likelihood.trials/(1+np.exp(-theta[0])))
        if family == "poisson": return np.asarray(np.exp(theta[0]))
        return np.asarray(theta[0])
    return CandidateModel("constant", UniformPrior(("c",), np.array([-5.]), np.array([5.])),
                          predict, structure="constant baseline")


class _NumPyroProposer:
    """Strict-JSON Appendix H.3.1 proposer for latent-domain models."""
    def __init__(self, client, env, allow_code_execution):
        self.client, self.env, self.allow = client, env, allow_code_execution

    def propose(self, *, context, dataset, current_models, residuals, count):
        rows = [(d.action, self.env.action_features[int(d.action)].tolist(),
                 float(d.observation)) for d in dataset]
        prompt = (f"{context}\nData (row index, feature row, observation): {rows}\n"
                  f"Current models: {[(m.name, m.structure) for m in current_models]}\n"
                  f"Residual reports: {list(residuals)}\n"
                  f"Return JSON {{\"models\":[{{\"name\":string,\"source\":string}}]}} with at most {count} "
                  "NumPyro models. numpyro, dist, jax, jnp, and np are already in scope; do not "
                  "import them. Define model(ctx); read the complete feature "
                  "table from ctx['features']; sample continuous priors; and set "
                  "numpyro.deterministic('mu', a finite vector with one value per feature row).")
        item = {"type": "object", "properties": {"name": {"type": "string"},
                "source": {"type": "string"}}, "required": ["name", "source"],
                "additionalProperties": False}
        schema = {"type": "object", "properties": {"models": {"type": "array", "items": item}},
                  "required": ["models"], "additionalProperties": False}
        entries = self.client.ask_json("Propose executable probabilistic models; JSON only.", prompt,
                                       schema=schema, name="numpyro_models",
                                       max_tokens=max(2400, 1200 * count))["models"]
        result = []
        for entry in entries[:count]:
            import jax
            import jax.numpy as jnp
            import numpyro
            import numpyro.distributions as dist
            try:
                compiled = compile_numpyro_source(entry["source"], self.env.action_features,
                                                  allow_code_execution=self.allow,
                                                  namespace={"numpyro": numpyro, "dist": dist,
                                                             "jax": jax, "jnp": jnp, "np": np},
                                                  name=str(entry.get("name", "numpyro")))
            except (KeyError, TypeError, ValueError, SyntaxError, ArithmeticError, RuntimeError):
                # One malformed program is dropped; its valid siblings stay.
                continue
            # The backend consumes feature matrices; the benchmark action is a
            # row index so query features cannot accidentally be omitted.
            def predict(theta, index, candidate=compiled, table=self.env.action_features):
                return np.asarray(candidate.predict(theta, table))[int(index)]
            result.append(CandidateModel(compiled.name, compiled.prior, predict,
                                         structure=entry["source"], metadata=compiled.metadata))
        return result


def make_agent(env: BoxingEnvironment, client, *, seed: int = 0, smoke: bool = False,
               experiment_budget: int = 10, allow_code_execution: bool = False):
    """Build the paper-configured MDA. NumPyro source execution is explicit opt-in.

    IRT/location/survival require latent domains and hence generated NumPyro code in a
    normal run.  Refusing these without the unsafe opt-in makes the trust boundary
    visible; smoke mode always uses the generic, non-truth baseline and no client.
    """
    if not isinstance(env, BoxingEnvironment): raise TypeError("env must be a BoxingEnvironment")
    if smoke and client is not None: raise ValueError("smoke mode does not use a client")
    if not smoke and client is None: raise ValueError("a client is required unless smoke=True")
    if not smoke and env.domain in {"irt", "location", "survival"} and not allow_code_execution:
        raise PermissionError("IRT/location/survival NumPyro proposals require allow_code_execution=True")
    proposer = None
    if not smoke and env.domain in {"irt", "location", "survival"}:
        proposer = _NumPyroProposer(client, env, allow_code_execution)
    elif not smoke:
        proposer = SymbolicRegressionProposer(
        client, variable_names={"hyperbolic": ("immediate", "delayed", "days"),
                               "survival": ("metastasized", "time"),
                               }.get(env.domain, ("x",)))
    particles = 30 if smoke else 1500
    return ModelDiscoveryAgent(
        context=env.context, initial_data=env.initial_data,
        initial_models=[_generic_model(env)] if smoke else [], proposer=proposer, likelihood=env.likelihood,
        actions=env.actions, query_actions=env.query_actions,
        observation_covariance={"dugongs": .25**2, "location": .5**2}.get(env.domain, .25),
        config=AgentConfig(model_cap=16, new_models_per_expansion=4,
                           model_search_rounds=2, expansion_cap=4, error_threshold=1,
                           occam_penalty=1, smc=SMCConfig(parameter_particles=particles,
                           rejuvenation_steps=4, target_ess_fraction=.5, max_temperatures=60)),
        experiment_budget=experiment_budget, seed=seed)
