"""Simulator invariants, checked every hour for every scenario preset and battery mode:
energy balance, battery accounting, SoC and power limits, no grid flows during outages.
Run with:  python -m pytest -q   (or  python tests/test_env.py)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import synth_overrides  # noqa: E402

from solar_rl import SolarHomeEnv, load_config  # noqa: E402
from solar_rl.config import available_scenarios  # noqa: E402
from solar_rl.sim import BATTERY_MODES, Dispatch, Simulator  # noqa: E402

SCENARIOS = [s for s in available_scenarios() if s != "default"] + ["default"]


def random_dispatch(rng, sim):
    mode = BATTERY_MODES[int(rng.integers(0, len(BATTERY_MODES)))]
    return Dispatch(battery=mode, run_flex=bool(rng.integers(0, 2)), reserve=float(rng.choice([0.1, 0.3, 0.5, 0.7])),
                    ac=["normal", "precool", "off"][int(rng.integers(0, 3))],
                    charge_kw=float(rng.uniform(0, 6)), discharge_kw=float(rng.uniform(0, 6)))


def check_hour(f, sim, soh_before):
    tol = 1e-6
    for k in ("solar", "import_kwh", "export_kwh", "charged", "discharged", "curtailed", "standby",
              "unserved_essential", "unserved_other", "unmet_cooling", "flex", "cooling"):
        assert f[k] >= -1e-12, (k, f[k])
    supply = f["solar"] + f["import_kwh"] + f["discharged"]
    use = ((f["essential"] - f["unserved_essential"]) + (f["other"] - f["unserved_other"])
           + (f["cooling"] - f["unmet_cooling"]) + f["flex"] + f["export_kwh"] + f["charged"]
           + f["curtailed"] + f["standby"])
    assert abs(supply - use) < tol, (supply, use, f)
    b = sim.batt
    assert f["charged_dc"] <= f["charged"] + 1e-12 and f["discharged_dc"] >= f["discharged"] - 1e-12
    assert b.floor_backup - 1e-9 <= f["soc"] <= b.soc_max + 1e-9
    if not f["grid_on"]:
        assert f["import_kwh"] == 0 and f["export_kwh"] == 0
    else:
        assert f["unserved_essential"] == 0 and f["unserved_other"] == 0
    return soh_before


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_energy_balance_and_limits(scenario):
    cfg = load_config(scenario, synth_overrides())
    sim = Simulator(cfg)
    for level in ["light", "medium", "heavy"]:
        for seed in range(2):
            sim.reset(seed=seed, level=level)
            rng = np.random.default_rng(seed)
            soc_prev, cap_prev = sim.batt.soc, sim.batt.capacity
            while not sim.done:
                d = random_dispatch(rng, sim)
                lim_c, lim_d = None, None
                f = sim.step(d)
                check_hour(f, sim, None)
                # battery: SoC change equals cell energy in minus out
                d_soc = (f["charged_dc"] - f["discharged_dc"]) / cap_prev
                assert abs((f["soc"] - soc_prev) - d_soc) < 1e-9
                soc_prev, cap_prev = f["soc"], sim.batt.capacity
            assert len(sim.log) == cfg.episode.hours


def test_power_limits_lfp(cfg_factory):
    cfg = cfg_factory()
    bc = cfg.battery
    sim = Simulator(cfg)
    sim.reset(seed=3, level="heavy")
    rng = np.random.default_rng(3)
    while not sim.done:
        f = sim.step(random_dispatch(rng, sim))
        assert f["charged"] <= bc.max_charge_c * bc.capacity_kwh + 1e-9
        assert f["discharged"] <= bc.max_discharge_c * bc.capacity_kwh + 1e-9


def test_no_grid_flows_during_outages_with_grid_charge(cfg_factory):
    sim = Simulator(cfg_factory(extra={"outages": {"level": "heavy"}}))
    sim.reset(seed=0)
    off = 0
    while not sim.done:
        f = sim.step(Dispatch(battery="grid_charge", run_flex=True, ac="precool"))
        if not f["grid_on"]:
            off += 1
            assert f["import_kwh"] == 0 and f["export_kwh"] == 0 and f["override"]
    assert off > 0


def test_backup_limit_respected(cfg_factory):
    sim = Simulator(cfg_factory(extra={"inverter": {"backup_max_kw": 0.5}, "outages": {"level": "heavy"}}))
    sim.reset(seed=1, soc=0.9)
    while not sim.done:
        f = sim.step(Dispatch(battery="discharge", run_flex=True, ac="precool"))
        if not f["grid_on"]:
            served = (f["essential"] - f["unserved_essential"] + f["other"] - f["unserved_other"]
                      + f["cooling"] - f["unmet_cooling"] + f["flex"])
            assert served <= 0.5 + 1e-9


def test_env_episode_runs_both_action_spaces(cfg_factory):
    for space, obs in [("reserve", "extended"), ("legacy", "legacy"), ("reserve", "legacy")]:
        env = SolarHomeEnv(cfg_factory(extra={"rl": {"action_space": space, "observation": obs}}))
        o, _ = env.reset(seed=0)
        assert env.observation_space.contains(o)
        n = 0
        while True:
            o, r, term, trunc, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(o) and np.isfinite(r)
            n += 1
            if trunc:
                break
        assert n == env.cfg.episode.hours
        assert "episode_stats" in info


def test_log_saves_csv(cfg_factory, tmp_path):
    sim = Simulator(cfg_factory())
    sim.reset(seed=0)
    while not sim.done:
        sim.step(Dispatch(battery="reserve", reserve=0.3))
    p = tmp_path / "week.csv"
    sim.save_log(p)
    import pandas as pd
    df = pd.read_csv(p)
    assert len(df) == 168 and {"solar", "import_kwh", "soc", "room_temp", "soh"} <= set(df.columns)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
