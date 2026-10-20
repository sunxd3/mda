from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from mda.benchmarks import glucose as g  # noqa: E402
from mda.types import Datum  # noqa: E402

TRUTH = np.array([15.0, 35.0, 50.0, 0.02, 1.0, 0.08, 0.8, 150.0])
CONSTANTS = (130.0, 95.0)
PROTOCOLS = [dict(meal_g=30, meal_minute=60, bolus_U=1, bolus_minute=75),
             dict(meal_g=45, meal_minute=90, bolus_U=1.5, bolus_minute=90),
             dict(meal_g=60, meal_minute=120, bolus_U=0.5, bolus_minute=150)]
# Glucose right-hand sides of SEED_SPECS[0] that parse but raise when evaluated.
RAISING_RHS = ["p2*(gb*p3 - log(g, 10)) + p4*q - p5*i", "p2*(gb*p3 - minimum(g)) + p4*q - p5*i",
               "p2*(gb*p3 - g) + p4*q - p5*i + 0*(1/0)", "p2*(gb*p3 - g) + p4*q - p5*i + 0*g*1e200**2",
               "p2*(gb*p3 - g) + p4*q - p5*i + 0*g*10**400"]


def synthetic_episode(protocol, key, rng, offset=0.0):
    inputs = g.nominal_inputs(protocol, key)
    cgm, insulin = g.GlucoseODE(g.SEED_SPECS[2]).simulate(TRUTH[None], [inputs], CONSTANTS)
    noise = np.zeros(73)
    noise[0] = rng.normal(0, 3 / np.sqrt(1 - 0.7**2))
    for t in range(1, 73):
        noise[t] = 0.7 * noise[t - 1] + rng.normal(0, 3)
    return Datum(inputs, np.r_[cgm[0, 0] + noise + offset, insulin[0, 0] + rng.normal(0, 10, 25)])


def payload(datum, cut):
    """A glucosebench-style prefix payload for a synthetic query."""
    stop = 0 if cut == 0 else cut // 5 + 1
    cgm, insulin = datum.observation[:73], datum.observation[73:]
    assays = dict(zip(g.INSULIN_TIMES.tolist(), insulin.tolist()))
    full_insulin = [assays.get(int(t)) for t in g.CGM_TIMES]
    minute_inputs = np.column_stack([np.arange(360), datum.action.meal, np.full(360, 0.02),
                                     datum.action.insulin]).tolist()
    query = {"times_min": g.CGM_TIMES[:stop].tolist(), "cgm_mg_dl": cgm[:stop].tolist(),
             "insulin_pmol_l": full_insulin[:stop],
             "insulin_mask": [int(t) % 15 == 0 for t in g.CGM_TIMES[:stop]],
             "minute_inputs": minute_inputs, "protocol": dict(datum.action.protocol)}
    return {"training_episodes": [], "query": query, "forecast_cut_min": cut,
            "requested_cgm_times_min": g.CGM_TIMES[g.CGM_TIMES > cut].tolist(),
            "requested_insulin_times_min": g.INSULIN_TIMES[g.INSULIN_TIMES > cut].tolist()}


def test_nominal_delivery_and_observation_layout():
    inputs = g.nominal_inputs(dict(meal_g=12, meal_minute=60, bolus_U=1, bolus_minute=70), "x")
    assert inputs.meal[60:63].tolist() == [5.0, 5.0, 2.0] and inputs.meal.sum() == 12
    assert np.allclose(inputs.insulin[70:75], 0.2) and inputs.insulin.sum() == pytest.approx(1.0)
    row = {"cgm_mg_dl": list(range(73)), "times_min": g.CGM_TIMES.tolist(),
           "insulin_mask": [int(t) % 15 == 0 for t in g.CGM_TIMES],
           "insulin_pmol_l": [float(t) if int(t) % 15 == 0 else None for t in g.CGM_TIMES]}
    observation = g.episode_observation(row)
    assert observation.shape == (98,) and observation[73:].tolist() == g.INSULIN_TIMES.tolist()


@pytest.mark.parametrize("value", [None, float("nan")])
def test_episode_observation_rejects_missing_scheduled_assays(value):
    row = {"cgm_mg_dl": [120.0] * 73, "times_min": g.CGM_TIMES.tolist(),
           "insulin_mask": [int(t) % 15 == 0 for t in g.CGM_TIMES],
           "insulin_pmol_l": [(value if int(t) == 90 else 100.0) if int(t) % 15 == 0 else None
                              for t in g.CGM_TIMES]}
    with pytest.raises(ValueError, match="menu:3.*insulin assay at minute 90"):
        g.episode_observation(row, "menu:3")


def test_specifications_are_validated():
    good = g.SEED_SPECS[2].to_dict()
    assert g.GlucoseModelSpec.from_dict(good).key == g.SEED_SPECS[2].key
    bad = [
        {**good, "states": ["a", "b", "c", "d", "e", "f", "h"], "rhs": ["0"] * 7, "initial_states": ["0"] * 7},
        {**good, "rhs": [*good["rhs"][:3], "__import__('os')"]},
        {**good, "rhs": [*good["rhs"][:3], "p12*g"]},
        {**good, "initial_states": ["meal", "0", "0", "gb"]},  # initial states see only gb, ib, params
        {**good, "bounds": [[-1, 2]] + good["bounds"][1:]},
        {**good, "initial": [999.0] + good["initial"][1:]},
        {**good, "states": ["p0", "q2", "i", "g"]},
    ]
    for spec in bad:
        with pytest.raises(ValueError):
            g.GlucoseModelSpec.from_dict(spec)


def test_ar1_bic_fit_recovers_noise_and_scores_bic():
    rng = np.random.default_rng(0)
    data = [synthetic_episode(p, f"e{i}", rng) for i, p in enumerate(PROTOCOLS)]
    constants = g.GlucoseConstants()
    model = g.glucose_candidate(g.SEED_SPECS[2], constants)
    posterior = g.GlucoseEvidence(constants, g.GlucoseFitSettings(random_starts=0, max_iterations=60))(
        data, model, None, rng)
    theta = posterior.map_parameters
    assert theta[8] == pytest.approx(0.7, abs=0.12) and theta[9] == pytest.approx(3.0, rel=0.25)
    assert theta[7] == pytest.approx(150, rel=0.1)  # insulin output gain
    n = 3 * 86
    assert posterior.log_evidence == pytest.approx(posterior.log_likelihoods[0] - 0.5 * 10 * np.log(n))
    assert constants.gb == pytest.approx(130, abs=4)
    # The fitted, correctly specified model passes the chi^2 gate on a fresh episode.
    fresh = synthetic_episode(dict(meal_g=40, meal_minute=60, bolus_U=1, bolus_minute=90), "f", rng)
    assert g.gate_statistic(model, theta, fresh, constants.values) < g.gate_threshold()
    wrong = theta.copy()
    wrong[5] *= 4  # four times the meal effect
    assert g.gate_statistic(model, wrong, fresh, constants.values) > g.gate_threshold()


def test_ekf_conditions_on_the_prefix_and_returns_requested_arrays():
    rng = np.random.default_rng(1)
    query = synthetic_episode(PROTOCOLS[1], "q", rng, offset=25.0)
    ode = g.GlucoseODE(g.SEED_SPECS[2])
    forecaster = g.GlucoseForecaster(((ode, TRUTH, 0.95, 3.0, 1.0),), CONSTANTS)
    for cut, (n_cgm, n_insulin) in {0: (72, 24), 45: (63, 21), 120: (48, 16), 180: (36, 12)}.items():
        forecast = forecaster(payload(query, cut))
        assert len(forecast["cgm_mg_dl"]) == n_cgm and len(forecast["insulin_pmol_l"]) == n_insulin
        assert np.all(np.isfinite(forecast["cgm_mg_dl"] + forecast["insulin_pmol_l"]))
    truth = query.observation[:73][g.CGM_TIMES > 180]
    open_loop = np.asarray(forecaster(payload(query, 0))["cgm_mg_dl"])[-36:]
    filtered = np.asarray(forecaster(payload(query, 180))["cgm_mg_dl"])
    assert np.mean((filtered - truth) ** 2) < 0.5 * np.mean((open_loop - truth) ** 2)


def test_design_prefers_templates_that_separate_the_models():
    constants = g.GlucoseConstants(*CONSTANTS)
    weak, strong = TRUTH.copy(), TRUTH.copy()
    strong[5] *= 2
    models = [g.glucose_candidate(g.SEED_SPECS[2], constants) for _ in range(2)]
    from mda.posterior import BeliefState, ModelFit
    belief = BeliefState(tuple(
        ModelFit(model, g._point_posterior(np.r_[theta, 0.5, 3.0], 0.0, 0.0), 0.0)
        for model, theta in zip(models, (weak, strong))), np.array([0.5, 0.5]))
    candidates = [g.nominal_inputs(dict(meal_g=m, meal_minute=60, bolus_U=0, bolus_minute=60), f"m{m}")
                  for m in (20, 40, 60)]
    scores = g.design_scores(belief, candidates, constants.values)
    assert np.argmax(scores) == 2 and np.all(np.diff(scores) > 0)


class FakeEnvironment:
    def __init__(self, rng):
        self.rng = rng
        self.initial_data = [synthetic_episode(p, f"passive:{i}", rng) for i, p in enumerate(PROTOCOLS[:2])]
        self.menu = {i: dict(meal_g=m, meal_minute=60, bolus_U=b, bolus_minute=60)
                     for i, (m, b) in enumerate([(20, 0), (20, 0), (40, 1), (60, 1.5)])}
        self.context = "synthetic"
        self.patient, self.replicate = "synthetic", 0

    def step(self, action):
        datum = synthetic_episode(action.protocol, action.key, self.rng)
        return datum.action, datum.observation


class FakeClient:
    def __init__(self, models):
        self.models = models
        self.prompts = []

    def ask_json(self, system, prompt, **kwargs):
        self.prompts.append((system, json.loads(prompt)))
        return {"models": self.models}


def test_agent_gate_failure_calls_proposer_and_rejects_invalid_births():
    rng = np.random.default_rng(2)
    environment = FakeEnvironment(rng)
    sensor_lag = {**g.SEED_SPECS[3].to_dict(), "name": "lagged", "parent_ids": ["delayed_cgm_readout"],
                  "origin": "edit", "edit_summary": "add a sensor lag"}
    orphan = {**g.SEED_SPECS[0].to_dict(), "name": "orphan", "rhs": ["meal - q/p0", "insulin - i/p1", "0*g"],
              "parent_ids": ["missing"], "origin": "edit", "edit_summary": "x"}
    client = FakeClient([sensor_lag, orphan])
    agent = g.make_agent(environment, None, smoke=True, experiment_budget=2)
    agent.proposer = g.GlucoseProposer(client, agent.constants)
    agent.proposer.belief_source = lambda: agent.belief
    agent.check = lambda belief, fresh, history: 2.0  # force the gate to fail
    first = agent.design()
    delivered, observation = environment.step(first)
    agent.update_state(delivered, observation)
    system, sent = client.prompts[0]
    assert system == g.GLUCOSE_PROPOSER_SYSTEM
    assert {m["model_id"] for m in sent["existing_models"]} >= {"two_stage_meal"}
    assert sent["observations"][0]["insulin_pmol_l"][1] is None
    # An exact-expression duplicate and an unknown parent are both rejected,
    # but the call still counts toward the expansion cap.
    assert agent.expansion_count == 1
    assert {m.name for m in agent.models} <= {spec.name for spec in g.SEED_SPECS}
    second = agent.design()
    assert second.key != first.key
    assert np.isfinite(agent.last_design_scores).all()


def test_proposer_accepts_new_structures_with_valid_genealogy():
    constants = g.GlucoseConstants(*CONSTANTS)
    existing = [g.glucose_candidate(spec, constants) for spec in g.SEED_SPECS]
    new = {**g.SEED_SPECS[2].to_dict(), "name": "two_stage_meal",
           "states": ["q1", "q2", "i", "j", "g"],
           "rhs": ["meal - q1/p0", "(q1 - q2)/p1", "insulin - i/p2", "p8*(i - j)",
                   "p3*(gb*p4 - g) + p5*q2 - p6*j*g/gb"],
           "initial_states": ["0", "0", "0", "0", "gb"], "outputs": ["g", "ib + p7*j"],
           "bounds": g.SEED_SPECS[2].to_dict()["bounds"] + [[0.001, 1.0]],
           "initial": list(g.SEED_SPECS[2].initial) + [0.05],
           "parent_ids": ["two_stage_meal"], "origin": "edit", "edit_summary": "insulin distribution"}
    recombined = {**new, "rhs": [*new["rhs"][:4], "p3*(gb*p4 - g) + p5*q2 - p6*i*g/gb"],
                  "parent_ids": ["two_stage_meal"], "origin": "recombination"}
    proposer = g.GlucoseProposer(FakeClient([new, recombined]), constants)
    models = proposer.propose(context="", dataset=[], current_models=existing, residuals=(), count=2)
    assert [m.name for m in models] == ["two_stage_meal_2"]  # recombination needs two parents
    assert models[0].metadata["genealogy"]["origin"] == "edit"
    assert models[0].parameter_count == 9 + 2


def test_proposer_drops_malformed_syntax_but_keeps_valid_siblings():
    constants = g.GlucoseConstants(*CONSTANTS)
    existing = [g.glucose_candidate(spec, constants) for spec in g.SEED_SPECS]
    base = {**g.SEED_SPECS[0].to_dict(), "parent_ids": ["one_stage_linear_action"], "origin": "edit"}
    broken = {**base, "name": "broken", "rhs": ["meal - q/p0", "insulin - i/p1", "p2*(gb*p3 - g"],
              "edit_summary": "unbalanced parenthesis"}
    valid = {**base, "name": "valid", "rhs": ["meal - q/p0", "insulin - i/p1", "p2*(gb - g) + p3*q - p4*i"],
             "edit_summary": "drop the equilibrium multiplier"}
    proposer = g.GlucoseProposer(FakeClient([broken, valid]), constants)
    models = proposer.propose(context="", dataset=[], current_models=existing, residuals=(), count=2)
    assert [m.name for m in models] == ["valid"]


def test_proposer_drops_models_that_raise_when_simulated(monkeypatch):
    rng = np.random.default_rng(0)
    constants = g.GlucoseConstants(*CONSTANTS)
    existing = [g.glucose_candidate(spec, constants) for spec in g.SEED_SPECS]
    dataset = [synthetic_episode(PROTOCOLS[0], "e0", rng)]
    base = {**g.SEED_SPECS[0].to_dict(), "parent_ids": ["one_stage_linear_action"], "origin": "edit",
            "edit_summary": "x"}
    valid = {**base, "name": "valid", "rhs": [*base["rhs"][:2], "p2*(gb - g) + p3*q - p4*i"]}
    for rhs in RAISING_RHS:
        raising = {**base, "name": "raising", "rhs": [*base["rhs"][:2], rhs]}
        for data in (dataset, []):  # screened on the first training episode, or a nominal schedule
            proposer = g.GlucoseProposer(FakeClient([raising, valid]), constants)
            models = proposer.propose(context="", dataset=data, current_models=existing, residuals=(), count=2)
            assert [m.name for m in models] == ["valid"]
    # A rollout that only diverges is not an error (fitting marks it invalid),
    # and a RecursionError is dropped like any other evaluation error.
    diverging = {**base, "name": "diverging", "rhs": [*base["rhs"][:2], "p2*g*g"]}
    deep = {**valid, "name": "deep"}
    simulate = g.GlucoseODE.simulate

    def deep_raises(ode, *args):
        if ode.spec.name == "deep":
            raise RecursionError("maximum recursion depth exceeded")
        return simulate(ode, *args)

    monkeypatch.setattr(g.GlucoseODE, "simulate", deep_raises)
    proposer = g.GlucoseProposer(FakeClient([deep, diverging]), constants)
    models = proposer.propose(context="", dataset=dataset, current_models=existing, residuals=(), count=2)
    assert [m.name for m in models] == ["diverging"]
    assert np.isnan(models[0].predict(np.r_[g.SEED_SPECS[0].initial, 0.5, 3.0], dataset[0].action)).all()


def test_evidence_fails_models_that_raise_without_aborting_the_posterior():
    from mda.posterior import fit_model_posterior
    rng = np.random.default_rng(0)
    data = [synthetic_episode(p, f"e{i}", rng) for i, p in enumerate(PROTOCOLS[:2])]
    constants = g.GlucoseConstants()
    evidence = g.GlucoseEvidence(constants, g.GlucoseFitSettings(random_starts=1, max_iterations=5))
    base = g.SEED_SPECS[0].to_dict()
    raising = []
    for rhs in RAISING_RHS:
        try:
            spec = g.GlucoseModelSpec.from_dict({**base, "name": f"raising{len(raising)}",
                                                 "rhs": [*base["rhs"][:2], rhs]})
        except ValueError:
            continue  # rejected when compiled, e.g. by an arity check
        raising.append(g.glucose_candidate(spec, constants))
        assert evidence(data, raising[-1], None, rng).log_evidence == -np.inf
    assert len(raising) >= 3  # the arithmetic errors parse and compile
    seed = g.glucose_candidate(g.SEED_SPECS[0], constants)
    belief = fit_model_posterior(data, [*raising, seed], None, None, model_cap=14, occam_penalty=0.0,
                                 rng=rng, evidence=evidence)
    assert [fit.model.name for fit in belief.fits] == [seed.name]


def test_proposer_rejects_whitespace_and_parenthesis_variants_of_existing_models():
    constants = g.GlucoseConstants(*CONSTANTS)
    existing = [g.glucose_candidate(spec, constants) for spec in g.SEED_SPECS]
    base = g.SEED_SPECS[2].to_dict()
    variant = {**base, "name": "respaced", "rhs": [*base["rhs"][:3], "p3 * (gb * p4 - g) + (p5*q2) - p6*i*g/gb"],
               "outputs": ["(g)", "ib + p7 * i"], "parent_ids": ["two_stage_meal"], "origin": "edit",
               "edit_summary": "same model"}
    assert g.GlucoseModelSpec.from_dict(variant).key == g.SEED_SPECS[2].key
    proposer = g.GlucoseProposer(FakeClient([variant]), constants)
    assert proposer.propose(context="", dataset=[], current_models=existing, residuals=(), count=2) == []


def test_working_likelihood_reads_sigma_from_theta():
    rng = np.random.default_rng(0)
    data = [synthetic_episode(p, f"e{i}", rng) for i, p in enumerate(PROTOCOLS)]
    constants = g.GlucoseConstants(*CONSTANTS)
    model = g.glucose_candidate(g.SEED_SPECS[2], constants)
    likelihood = g.GlucoseLikelihood(constants)
    cgm, insulin = g._residuals(model.metadata["ode"], TRUTH[None], data, constants.values)
    profiled, sigma_hat = g._negative_log_likelihood(cgm, insulin, np.array([0.7]))
    values = {s: likelihood(data, model, np.r_[TRUTH, 0.7, s], None) for s in (1.0, sigma_hat[0], 1000.0)}
    assert values[sigma_hat[0]] == pytest.approx(-profiled[0])  # the profiled sigma maximizes it
    assert values[sigma_hat[0]] > values[1.0] > values[1000.0]


def test_forecaster_drops_components_whose_ekf_state_diverges():
    grow = g.GlucoseModelSpec("grow", ("g",), ("p0*meal*g",), ("gb",), ("g", "ib + 0*g"), ((1e-4, 1.0),), (0.2,))
    ode, theta = g.GlucoseODE(grow), np.array([0.2])
    datum = synthetic_episode(dict(meal_g=60, meal_minute=60, bolus_U=0, bolus_minute=60), "q",
                              np.random.default_rng(0))
    assert np.isnan(ode.simulate(theta[None], [datum.action], CONSTANTS)[0]).all()
    cgm, insulin = g._ekf_forecast(ode, theta, 0.7, 3.0, CONSTANTS, payload(datum, 45))
    assert np.isnan(cgm).all() and np.isnan(insulin).all()
    good = (g.GlucoseODE(g.SEED_SPECS[2]), TRUTH, 0.7, 3.0, 0.99)
    alone = g.GlucoseForecaster((good,), CONSTANTS)(payload(datum, 45))
    mixed = g.GlucoseForecaster((good, (ode, theta, 0.7, 3.0, 0.01)), CONSTANTS)(payload(datum, 45))
    np.testing.assert_allclose(mixed["cgm_mg_dl"], alone["cgm_mg_dl"])
    np.testing.assert_allclose(mixed["insulin_pmol_l"], alone["insulin_pmol_l"])


def test_forecaster_ignores_components_that_fail():
    rng = np.random.default_rng(0)
    datum = synthetic_episode(PROTOCOLS[0], "q", rng)
    ode = g.GlucoseODE(g.SEED_SPECS[2])
    good = (ode, TRUTH, 0.7, 3.0, 0.5)
    failing = (ode, np.full(TRUTH.shape, np.nan), 0.7, 3.0, 0.5)
    alone = g.GlucoseForecaster((good,), CONSTANTS)(payload(datum, 45))
    mixed = g.GlucoseForecaster((good, failing), CONSTANTS)(payload(datum, 45))
    np.testing.assert_allclose(mixed["cgm_mg_dl"], alone["cgm_mg_dl"])


def test_icl_forecaster_retries_once_on_schema_errors():
    calls = []

    class Client:
        def ask_json(self, system, prompt, **kwargs):
            calls.append(prompt)
            cgm_count = 62 if len(calls) == 1 else 63  # first reply is one value short
            return {"cgm_mg_dl": [100.0] * cgm_count, "insulin_pmol_l": [90.0] * 21}

    query = payload(synthetic_episode(PROTOCOLS[0], "q", np.random.default_rng(3)), 45)
    forecast = g.make_glucose_icl_forecaster(Client())(query)
    assert len(calls) == 2 and "failed schema validation" in calls[1]
    assert len(forecast["cgm_mg_dl"]) == 63


@pytest.mark.slow
def test_glucosebench_smoke_run_scores_every_query():
    pytest.importorskip("glucosebench")
    environment = g.GlucoseBenchEnvironment("adult#001")
    agent = g.make_agent(environment, None, smoke=True, experiment_budget=1)
    result = g.run_glucose(environment, agent, budget=1, cuts=(45,))
    mda = result["scores"]["mda"]["summary"]
    assert [row["budget"] for row in mda] == [2, 3]
    assert all(row["failed"] == 0 and np.isfinite(list(row["mean_nmse"].values())).all() for row in mda)
    assert result["history"][-1]["action"].startswith("menu:")
    json.dumps(result, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v))


@pytest.mark.parametrize("extra", [["--budget", "5"], ["--cuts", "30"], ["--design", "joint"]])
def test_cli_rejects_invalid_glucose_requests(monkeypatch, tmp_path, extra):
    from mda.benchmarks.reproduce import main
    monkeypatch.setattr("sys.argv", ["mda-reproduce", "glucose", "--world", "adult#001", "--smoke",
                                     "--output", str(tmp_path / "result.json"), *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_no_m_open_builds_the_agent_without_a_proposer():
    environment = FakeEnvironment(np.random.default_rng(3))
    client = FakeClient([])
    agent = g.make_agent(environment, client, smoke=True, experiment_budget=2, no_m_open=True)
    assert agent.proposer is None and agent.initial_gate_error is None
    assert client.prompts == []
