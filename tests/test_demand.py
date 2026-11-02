"""Demand: appliance profiles, weekend/Ramadan variants, measured logs, room thermal model."""
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from solar_rl import load_config
from solar_rl.sim import Dispatch, Simulator
from solar_rl.sim.demand import DemandModel, day_types, load_usage_log
from solar_rl.sim.thermal import Room
from tests.conftest import synth_overrides


def week(start="2023-06-05", hours=24 * 14, temp=30.0):
    idx = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame({"ghi": 0.0, "temp_c": temp, "wind_ms": 2.0, "precip_mm": 0.0}, index=idx)


def test_profiles_load_and_validate():
    for name in ("household", "shop", "household_v1"):
        cfg = load_config("default", {"demand": {"profile": name}})
        assert cfg.demand.profile.name == name
        assert len(cfg.demand.profile.appliances) > 0


def test_weekend_and_ramadan_days():
    d = load_config("default").demand
    idx = pd.date_range("2024-03-08", "2024-04-12", freq="D")
    weekend, ramadan = day_types(idx, d)
    assert weekend[idx.day_of_week == 6].all() and not weekend[idx.day_of_week != 6].any()
    assert ramadan[idx == "2024-03-12"].all() and ramadan[idx == "2024-04-09"].all()
    assert not ramadan[idx == "2024-03-11"].any() and not ramadan[idx == "2024-04-10"].any()


def test_variant_overrides():
    p = load_config("default").demand.profile
    base = {a.name: a for a in p.appliances}
    wk = {a.name: a for a in p.variant(weekend=True, ramadan=False).appliances}
    rm = {a.name: a for a in p.variant(weekend=False, ramadan=True).appliances}
    assert 14 in wk["tv"].hours and 14 not in base["tv"].hours
    assert rm["lights_night"].kw > base["lights_night"].kw            # suhoor lighting
    assert 10 in p.variant(True, False).cooling.occupied_hours


def test_probability_of_use():
    cfg = load_config("default", {"demand": {"noise": 0.0, "profile_overrides": {
        "appliances": [{"name": "tv", "probability": 0.25}]}}})
    w = week(hours=24 * 400)
    dm = DemandModel(cfg, w)
    rng = np.random.default_rng(0)
    on = []
    for t in range(len(w)):
        if w.index[t].hour == 21 and not dm.weekend[t]:
            _, oth, _ = dm.sample(t, rng)
            # other at 21:00 = tv (0.12, p) + charging (0.08, 0.6) + dispenser (0.08, 0.5)
            on.append(oth)
    on = np.array(on)
    mean_expected = 0.12 * 0.25 + 0.08 * 0.6 + 0.08 * 0.5
    assert on.mean() == pytest.approx(mean_expected, rel=0.08)


def test_temperature_scaling_of_fans():
    cfg = load_config("default", {"demand": {"noise": 0.0, "profile": "household_v1"}})
    rng = np.random.default_rng(0)
    cool = DemandModel(cfg, week(temp=20.0))
    hot = DemandModel(cfg, week(temp=40.0))
    t = 12
    e_cool, _, _ = cool.sample(t, rng)
    e_hot, _, _ = hot.sample(t, rng)
    assert e_hot - e_cool == pytest.approx(0.40)


def test_load_usage_log_resamples_and_converts(tmp_path):
    idx = pd.date_range("2023-06-01", periods=12 * 24 * 2, freq="5min")
    df = pd.DataFrame({"Timestamp": idx, "load_w": 1000.0 + 500.0 * (idx.hour >= 18)})
    df = df.drop(index=range(12 * 30, 12 * 30 + 10))          # most of one hour missing
    p = tmp_path / "log.csv"
    df.to_csv(p, index=False)
    h = load_usage_log(p)
    assert h.loc["2023-06-01 10:00"] == pytest.approx(1.0)
    assert h.loc["2023-06-01 19:00"] == pytest.approx(1.5)
    assert np.isnan(h.loc["2023-06-02 06:00"])


def test_measured_demand_replaces_profile(tmp_path):
    idx = pd.date_range("2021-03-01", "2021-04-30 23:00", freq="h")
    pd.DataFrame({"timestamp": idx, "load_kw": 0.5 + 0.1 * idx.hour}).to_csv(tmp_path / "m.csv", index=False)
    cfg = load_config("default", synth_overrides({"demand": {"model": "measured", "measured_log": str(tmp_path / "m.csv")}}))
    sim = Simulator(cfg)
    assert sim.index[sim.valid_starts].min() >= pd.Timestamp("2021-03-01")
    sim.reset(seed=0, level="light")
    rows = [sim.step(Dispatch(run_flex=True)) for _ in range(48)]
    for r in rows:
        assert r["essential"] == pytest.approx(0.5 + 0.1 * r["hour"])
        assert r["cooling"] == 0 and r["flex"] == 0


# ---------------------------------------------------------------- thermal
def cc_default():
    return load_config("default").demand.profile.cooling


def test_room_analytic_solution():
    cc = cc_default()
    room = Room(cc, initial_temp=26.0)
    mean, end = room.step(36.0, 0.0, 0.0)
    t_eq = 36.0 + cc.r_c_per_kw * cc.internal_gain_kw
    tau = cc.r_c_per_kw * cc.c_kwh_per_c
    assert end == pytest.approx(t_eq + (26.0 - t_eq) * math.exp(-1 / tau))
    assert 26.0 < mean < end


def test_ac_reaches_setpoint_and_room_heats_without_it():
    cc = cc_default()
    room = Room(cc, initial_temp=27.0)
    q = room.cooling_needed(35.0, 800.0, cc.setpoint_c)
    assert 0 < q < cc.ac_cooling_kw
    _, end = room.step(35.0, 800.0, q)
    assert end == pytest.approx(cc.setpoint_c, abs=1e-9)
    temps = [room.step(35.0, 800.0, 0.0)[1] for _ in range(6)]
    assert all(b > a for a, b in zip(temps, temps[1:]))            # outage: room heats up
    assert Room(cc, initial_temp=22.0).cooling_needed(20.0, 0.0, 26.0) == 0.0   # never heats


def test_capacity_limit():
    cc = cc_default()
    room = Room(cc, initial_temp=40.0)
    assert room.cooling_needed(45.0, 1000.0, 18.0) == pytest.approx(cc.ac_cooling_kw)


def test_precooling_reduces_comfort_loss_during_outage(cfg_factory):
    """Pre-cooling before a known outage keeps the room cooler while the AC is off."""
    over = {"outages": {"level": "light", "intensity": 0.0}, "battery": {"soc_cutoff_backup": 0.05},
            "inverter": {"backup_max_kw": 0.3}}           # backup too small to run the AC
    res = {}
    for mode in ("normal", "precool"):
        cfg = cfg_factory(extra=over)
        sim = Simulator(cfg)
        start = int(sim.valid_starts[np.searchsorted(sim.index[sim.valid_starts], pd.Timestamp("2021-06-14"))])
        sim.reset(seed=0, start_index=start, level="light", soc=0.9)
        comfort = 0.0
        for k in range(24):
            h = sim.hours[sim.t]
            if 13 <= h <= 16:                               # force an outage 13:00-16:59
                sim.grid_on = False
            ac = mode if h in (10, 11, 12) else "normal"
            f = sim.step(Dispatch(battery="hold", ac=ac))
            if 13 <= h <= 16:
                assert not f["grid_on"]
                comfort += f["comfort_degree_hours"]
        res[mode] = comfort
    assert res["precool"] < res["normal"]


def test_comfort_only_counted_when_occupied(cfg_factory):
    sim = Simulator(cfg_factory(extra={"outages": {"level": "light", "intensity": 0.0}}))
    sim.reset(seed=0)
    while not sim.done:
        f = sim.step(Dispatch(ac="off"))
        if not sim.demand.profile_at(sim.t - 1).cooling.occupied_hours.__contains__(f["hour"]):
            assert f["comfort_degree_hours"] == 0.0
