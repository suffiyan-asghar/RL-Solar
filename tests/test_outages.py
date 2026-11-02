"""Load-shedding: published schedules (CSV), semi-Markov outages, log replay and fitting."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from solar_rl import load_config
from solar_rl.sim import Dispatch, Simulator
from solar_rl.sim.outages import (OutageModel, SemiMarkovOutages, fit_outage_model, load_outage_log,
                                  outage_events, read_schedule_csv)
from tests.conftest import ROOT, synth_overrides

EXAMPLE = f"{ROOT}/configs/examples/schedule_example.csv"


def test_read_schedule_csv():
    s = read_schedule_csv(EXAMPLE)
    assert s["low_loss"] == set()
    assert s["medium_loss"] == {7, 8, 19, 20}
    assert s["high_loss"] == {4, 5, 6, 12, 13, 14, 20, 21, 22}


def test_schedule_csv_wraps_midnight(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("feeder_category,start,end\nx,22:00,01:00\ny,10:30,12:30\n")
    s = read_schedule_csv(p)
    assert s["x"] == {22, 23, 0}
    assert s["y"] == {11, 12}


def test_csv_schedule_in_simulator(cfg_factory):
    cfg = cfg_factory(extra={"outages": {"schedule": {"source": "csv", "csv_path": EXAMPLE,
                                                      "category_by_level": {"heavy": "high_loss"}},
                                         "intensity": 0.0}})
    sim = Simulator(cfg)
    sim.reset(seed=0, level="heavy")
    while not sim.done:
        h = sim.hours[sim.t]
        f = sim.step(Dispatch())
        assert f["grid_on"] == (h not in {4, 5, 6, 12, 13, 14, 20, 21, 22})


def test_semi_markov_durations_follow_distribution():
    cfg = load_config("default", {"outages": {"semi_markov": {"duration": {
        "distribution": "weibull", "params": {"c": 1.5, "scale": 3.0}, "max_hours": 24}}}})
    m = SemiMarkovOutages(cfg, "heavy", np.random.default_rng(0))
    d = np.array([m.sample_duration() for _ in range(20000)])
    expected = np.mean(np.ceil(stats.weibull_min(c=1.5, scale=3.0).rvs(200000, random_state=1)).clip(1, 24))
    assert d.mean() == pytest.approx(expected, rel=0.03)
    assert d.min() >= 1 and d.max() <= 24


def test_semi_markov_start_probability_factors():
    hf = [1.0] * 24
    hf[19] = 3.0
    mf = [1.0] * 12
    mf[5] = 2.0
    cfg = load_config("default", {"outages": {"semi_markov": {"hour_factor": hf, "month_factor": mf,
                                                              "heat_factor_per_c": 0.1}}})
    m = SemiMarkovOutages(cfg, "medium", np.random.default_rng(0))
    base = cfg.outages.semi_markov.start_prob["medium"]
    assert m.start_prob(10, 1, 30.0) == pytest.approx(base)
    assert m.start_prob(19, 6, 30.0) == pytest.approx(base * 6)
    assert m.start_prob(10, 1, 37.0) == pytest.approx(base * 1.5)


def test_semi_markov_outage_lasts_its_duration():
    cfg = load_config("default", {"outages": {"semi_markov": {"start_prob": {"light": 1.0, "medium": 1.0, "heavy": 1.0},
                                                              "duration": {"distribution": "exponential",
                                                                           "params": {"scale": 1e-9},
                                                                           "max_hours": 12}}}})
    m = SemiMarkovOutages(cfg, "heavy", np.random.default_rng(0))
    m.sample_duration = lambda: 4
    seq = [m.step(t, 10, 1, 25.0) for t in range(8)]
    assert seq == [True] * 8           # back-to-back 4 h outages (start prob 1)
    m2 = SemiMarkovOutages(cfg, "heavy", np.random.default_rng(0))
    m2.sample_duration = lambda: 3
    m2.s.start_prob["heavy"] = 0.0
    m2.remaining = 2
    assert [m2.step(t, 10, 1, 25.0) for t in range(4)] == [True, True, False, False]


def make_log(tmp_path, grid_on, start="2021-03-01"):
    idx = pd.date_range(start, periods=len(grid_on), freq="h")
    p = tmp_path / "outages.csv"
    pd.DataFrame({"timestamp": idx, "grid_on": grid_on}).to_csv(p, index=False)
    return p


def test_load_outage_log_state_changes(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text("timestamp,grid_on\n2023-01-01 00:00,1\n2023-01-01 03:20,0\n2023-01-01 05:10,1\n2023-01-01 08:00,1\n")
    g = load_outage_log(p, off_threshold=0.5)
    assert list(g.loc["2023-01-01 02:00":"2023-01-01 06:00"]) == [1.0, 0.0, 0.0, 1.0, 1.0]


def test_replay_outage_log(tmp_path, cfg_factory):
    rng = np.random.default_rng(5)
    g = (rng.random(24 * 60) > 0.2).astype(int)
    p = make_log(tmp_path, g)
    sim = Simulator(cfg_factory(extra={"outages": {"model": "replay", "replay": {"csv_path": str(p)}}}))
    assert sim.index[sim.valid_starts].min() >= pd.Timestamp("2021-03-01")
    sim.reset(seed=0)
    i0 = sim.t
    while not sim.done:
        t = sim.t
        f = sim.step(Dispatch())
        assert f["grid_on"] == bool(g[t - (i0 - (sim.index[i0] - pd.Timestamp("2021-03-01")) // pd.Timedelta("1h"))])
    assert sim.level == "replay"


def test_fit_recovers_parameters(tmp_path):
    """Generate a log from known semi-Markov parameters + a schedule, then fit it back."""
    rng = np.random.default_rng(11)
    dist = stats.lognorm(s=0.5, scale=2.5)
    n = 24 * 365 * 2
    hours = np.arange(n) % 24
    on = np.ones(n)
    t = 0
    p0 = 0.02
    while t < n:
        if hours[t] in (8, 9):                    # scheduled block 08:00-09:59 every day
            on[t] = 0
            t += 1
            continue
        p = p0 * (2.0 if hours[t] >= 18 else 1.0)
        if rng.random() < p:
            d = int(np.clip(np.ceil(dist.rvs(random_state=rng)), 1, 24))
            on[t:t + d] = 0
            t += d
        else:
            t += 1
    g = pd.Series(on, index=pd.date_range("2021-01-01", periods=n, freq="h"))
    fit = fit_outage_model(g)
    assert fit["scheduled_hours"] == [8, 9]
    sm = fit["semi_markov"]
    assert sm["start_prob_per_hour"] == pytest.approx(p0 * (18 + 2 * 6) / 24 * 24 / 22 / 1.1, rel=0.35)
    evening = np.mean(sm["hour_factor"][18:24])
    day = np.mean([sm["hour_factor"][h] for h in range(11, 18)])
    assert evening / day == pytest.approx(2.0, rel=0.3)
    assert sm["duration"]["distribution"] in ("lognormal", "gamma", "weibull")
    durs = np.array(fit["durations"])
    assert durs.mean() == pytest.approx(np.mean(np.ceil(dist.rvs(100000, random_state=0)).clip(1, 24)), rel=0.15)


def test_outage_events():
    g = pd.Series([1, 0, 0, 1, 0, 1, 0, 0, 0], index=pd.date_range("2021-01-01", periods=9, freq="h"))
    assert [d for _, d in outage_events(g)] == [2, 1, 3]


def test_intensity_zero_means_no_unscheduled_outages(cfg_factory):
    sim = Simulator(cfg_factory(extra={"outages": {"level": "light", "intensity": 0.0}}))
    sim.reset(seed=0)
    while not sim.done:
        assert sim.step(Dispatch())["grid_on"]
