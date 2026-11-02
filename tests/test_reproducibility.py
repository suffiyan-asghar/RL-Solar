"""Reproducibility from one seed, independent random streams, and identical results for the
same setup whichever way the config is given."""
import numpy as np
import pandas as pd
import pytest

from solar_rl import SolarHomeEnv, load_config
from solar_rl.sim import Dispatch, Simulator
from tests.conftest import synth_overrides

EXOGENOUS = ["grid_on", "essential", "other", "solar", "temp_out"]


def run(cfg, seed, policy="reserve", level=None, weather=None):
    sim = Simulator(cfg, weather)
    sim.reset(seed=seed, level=level)
    rng = np.random.default_rng(99)
    while not sim.done:
        if policy == "reserve":
            d = Dispatch(battery="reserve", reserve=0.3, run_flex=sim.hours[sim.t] >= 9)
        else:
            d = Dispatch(battery=["grid_charge", "solar_charge", "hold", "discharge"][rng.integers(0, 4)],
                         run_flex=bool(rng.integers(0, 2)), ac=["normal", "precool"][rng.integers(0, 2)])
        sim.step(d)
    return sim.log_df()


def test_same_seed_same_episode(cfg_factory):
    cfg = cfg_factory(extra={"outages": {"level": "heavy"}})
    a, b = run(cfg, 7), run(cfg, 7)
    pd.testing.assert_frame_equal(a, b)
    c = run(cfg, 8)
    assert not a[EXOGENOUS].equals(c[EXOGENOUS])


def test_env_reproducible_across_resets(cfg_factory):
    def rollout():
        env = SolarHomeEnv(cfg_factory())
        env.reset(seed=123)
        out = []
        for _ in range(3):                       # later resets without a seed are still determined by it
            obs, _ = env.reset()
            total = 0.0
            for k in range(168):
                obs, r, *_ = env.step(k % env.action_space.n)
                total += r
            out.append((float(obs.sum()), total))
        return out
    assert rollout() == rollout()


def test_actions_do_not_change_exogenous_randomness(cfg_factory):
    """Each random component has its own RNG: outages and demand draws are identical whatever
    the agent does (so every policy faces the same conditions)."""
    cfg = cfg_factory(extra={"outages": {"level": "medium"}})
    a = run(cfg, 3, policy="reserve")
    b = run(cfg, 3, policy="random")
    pd.testing.assert_frame_equal(a[EXOGENOUS], b[EXOGENOUS])


def test_forecast_noise_does_not_change_simulation(cfg_factory):
    a = run(cfg_factory(), 5)
    b = run(cfg_factory(extra={"episode": {"solar_forecast_noise": 0.5}}), 5)
    pd.testing.assert_frame_equal(a, b)


def test_same_setup_any_config_route(tmp_path):
    """large_home.yaml == default.yaml + the same overrides == the saved resolved config."""
    preset = load_config("large_home", synth_overrides())
    overrides = synth_overrides({
        "pv": {"kwp": 10.0}, "battery": {"capacity_kwh": 20.0}, "inverter": {"rated_kw": 10.0, "backup_max_kw": 10.0},
        "demand": {"profile_overrides": {
            "cooling": {"r_c_per_kw": 3.0, "c_kwh_per_c": 2.0, "solar_aperture_m2": 1.0, "internal_gain_kw": 0.40,
                        "ac_cooling_kw": 10.6},
            "appliances": [{"name": "fridge", "kw": 0.20}, {"name": "lights_evening", "kw": 0.50},
                           {"name": "fans", "kw": 0.60}],
            "flexible": {"daily_kwh": 3.0, "max_kw": 1.5}}}})
    manual = load_config("default", overrides)
    path = tmp_path / "resolved.yaml"
    preset.save(path)
    saved = load_config(str(path))
    assert preset.to_dict() == manual.to_dict() == saved.to_dict()
    ref = run(preset, 4, level="heavy")
    for cfg in (manual, saved, preset.copy()):
        pd.testing.assert_frame_equal(run(cfg, 4, level="heavy"), ref)


@pytest.mark.parametrize("preset,override", [
    ("heavy_loadshedding", {"outages": {"level": "heavy"}}),
    ("legacy_net_metering", {"tariff": "legacy_net_metering_placeholder"}),
    ("shop", {"demand": {"profile": "shop"}, "tariff": "commercial_placeholder"}),
    ("small_home", {"pv": {"kwp": 3.0}, "battery": {"capacity_kwh": 5.0},
                    "inverter": {"rated_kw": 4.0, "backup_max_kw": 4.0}}),
])
def test_presets_equal_default_plus_overrides(preset, override):
    a = load_config(preset, synth_overrides())
    b = load_config("default", synth_overrides(override))
    assert a.to_dict() == b.to_dict()
    ea, eb = SolarHomeEnv(a), SolarHomeEnv(b)
    for env in (ea, eb):
        env.reset(seed=11)
    for k in range(168):
        oa, ra, *_ = ea.step(k % 16)
        ob, rb, *_ = eb.step(k % 16)
        assert ra == rb and np.array_equal(oa, ob)
