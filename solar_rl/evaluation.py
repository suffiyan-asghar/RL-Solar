"""Evaluate any policy on a FIXED set of test episodes so every policy faces
exactly the same weather, demand noise and outages (same seeds, same start dates).
"""
import numpy as np
import pandas as pd

from .env import SolarHomeEnv

METRICS = ["total_cost_incl_wear", "cost", "wear_cost", "unserved_essential_kwh", "unserved_hours",
           "unmet_cooling_kwh", "flex_missed_kwh", "battery_cycles", "self_consumption",
           "outage_hours", "reward"]


class SB3Policy:
    """Wrap a trained Stable-Baselines3 model so it looks like a baseline."""

    def __init__(self, model, name):
        self.model, self.name = model, name

    def act(self, env, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def episode_plan(env: SolarHomeEnv, n_episodes, levels, base_seed=12345):
    """Evenly spaced start dates across the data, crossed with outage levels."""
    starts = env.valid_starts[np.linspace(0, len(env.valid_starts) - 1, n_episodes).astype(int)]
    plan = []
    for i, s in enumerate(starts):
        for level in levels:
            plan.append(dict(start_index=int(s), level=level, soc=0.5, seed=base_seed + i))
    return plan


def run_episode(env, policy, ep, record=False):
    obs, info = env.reset(seed=ep["seed"], options=ep)
    trace = []
    while True:
        action = policy.act(env, obs)
        obs, r, term, trunc, info = env.step(action)
        if record:
            trace.append(info)
        if term or trunc:
            break
    stats = info["episode_stats"]
    stats["start_time"] = str(env.weather.index[ep["start_index"]])
    stats["month"] = env.weather.index[ep["start_index"]].month
    return stats, (pd.DataFrame(trace) if record else None)


def evaluate(env, policies, plan):
    rows = []
    for p in policies:
        for ep in plan:
            stats, _ = run_episode(env, p, ep)
            stats["policy"] = p.name
            rows.append(stats)
    return pd.DataFrame(rows)


def season(month):
    if month in (6, 7, 8, 9):
        return "monsoon/summer"
    if month in (11, 12, 1, 2):
        return "winter"
    return "spring/autumn"


def summarise(df, by=("policy",)):
    agg = df.groupby(list(by))[METRICS].agg(["mean", "std"])
    return agg
