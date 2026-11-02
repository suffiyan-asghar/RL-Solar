"""Sanity tests: energy is conserved every hour and the battery stays within limits.
Run with:  python -m pytest -q   (or  python tests/test_env.py)
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from solar_rl import SimConfig, SolarHomeEnv  # noqa: E402


def run_random(level, seed):
    cfg = SimConfig(outage_level=level)
    env = SolarHomeEnv(cfg)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(cfg.episode_hours):
        soc_before = env.batt.soc
        obs, r, term, trunc, i = env.step(int(rng.integers(0, 8)))
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        supply = i["solar"] + i["import_kwh"] + i["discharged"]
        use = ((i["essential"] - i["unserved_essential"]) + (i["cooling"] - i["unmet_cooling"])
               + i["flex"] + i["export_kwh"] + i["charged"] + i["curtailed"])
        assert abs(supply - use) < 1e-6, (supply, use, i)
        d_soc = (i["charged"] * cfg.eta_charge - i["discharged"] / cfg.eta_discharge) / cfg.batt_kwh
        assert abs((i["soc"] - soc_before) - d_soc) < 1e-9
        assert cfg.soc_min - 1e-9 <= i["soc"] <= cfg.soc_max + 1e-9
        assert i["charged"] <= cfg.batt_max_kw + 1e-9 and i["discharged"] <= cfg.batt_max_kw + 1e-9
        if not i["grid_on"]:
            assert i["import_kwh"] == 0 and i["export_kwh"] == 0
        if term or trunc:
            break
    assert trunc


def test_energy_balance_all_levels():
    for level in ["light", "medium", "heavy"]:
        for seed in range(3):
            run_random(level, seed)


if __name__ == "__main__":
    test_energy_balance_all_levels()
    print("all tests passed")
