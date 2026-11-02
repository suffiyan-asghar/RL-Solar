"""The refactored simulator reproduces the ORIGINAL one (commit a1cc63d) exactly.

tests/data/golden_v1.json was recorded with the original code: 32 episodes (4 outage levels
x 2 seeds x 4 fixed action sequences) on a 3-week slice of real 2024 Karachi weather
(tests/data/weather_slice_2024.csv). configs/legacy_v1.yaml selects the original models
(simple PV and battery, rule-based AC, Markov outages, one shared RNG, legacy actions and
observation). Every per-step reward, the observations and all episode stats must match.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from solar_rl import SolarHomeEnv, load_config

HERE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")


def _weather():
    w = pd.read_csv(os.path.join(HERE, "data", "weather_slice_2024.csv"), parse_dates=["time"]).set_index("time")
    w.attrs["source"] = "slice"
    return w


GOLDEN = json.load(open(os.path.join(HERE, "data", "golden_v1.json")))["cases"]


@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: f"{c['level']}-s{c['seed']}-a{c['actions'][0]}")
def test_matches_original(case):
    cfg = load_config("legacy_v1", {"outages": {"level": case["level"]}})
    env = SolarHomeEnv(cfg, weather=_weather())
    obs, info = env.reset(seed=case["seed"], options=case["options"])
    assert info["start_time"] == case["start_time"]
    obs_sum = np.array(obs, dtype=float)
    rewards = []
    for a in case["actions"]:
        obs, r, term, trunc, i = env.step(a)
        rewards.append(r)
        obs_sum += obs
    assert trunc
    np.testing.assert_allclose(rewards, case["rewards"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(obs_sum, case["obs_sum"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(obs, case["final_obs"], rtol=1e-6, atol=1e-6)
    stats = i["episode_stats"]
    for k, v in case["stats"].items():
        if isinstance(v, str):
            assert str(stats[k]) == v, k
        else:
            assert stats[k] == pytest.approx(v, rel=1e-9, abs=1e-9), k
