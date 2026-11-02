"""Evaluate policies on a FIXED set of test episodes, so every policy faces exactly the same
weather, demand and outages (same seeds, same start dates).

Seeds: each evaluation seed s uses its own set of episode seeds (base_seed = 10000 * (s + 1)),
so different seeds see different random outages and demand. A trained agent can be given per
seed (e.g. models/ppo_seed{seed}/best_model.zip): seed s is then evaluated with model s. Every
metric is reported as the mean over episodes per seed, then mean +/- std across seeds.
"""
import json
import os

import numpy as np
import pandas as pd

from .config import load_config
from .env import SolarHomeEnv
from .sim import Dispatch

METRICS = ["total_cost_incl_wear", "cost", "wear_cost", "wear_cycle_cost", "unserved_essential_kwh",
           "unserved_hours", "unserved_other_kwh", "unmet_cooling_kwh", "comfort_degree_hours", "flex_missed_kwh",
           "battery_cycles", "self_consumption", "import_kwh", "export_kwh", "outage_hours", "reward"]


class SB3Policy:
    """A trained Stable-Baselines3 model that looks like a baseline. `rl` holds the action and
    observation settings the model was trained with (from env_config.yaml next to it)."""

    def __init__(self, model, name, rl=None):
        self.model, self.name, self.rl = model, name, rl or {"action_space": "legacy", "observation": "legacy"}

    def act(self, env, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def load_model(path, name=None):
    """Load a PPO/DQN model; read its rl settings from env_config.yaml in the same folder."""
    from stable_baselines3 import DQN, PPO
    run = os.path.dirname(path)
    meta = os.path.join(run, "train_config.json")
    algo = None
    if os.path.exists(meta):
        algo = json.load(open(meta)).get("algo")
    algo = algo or ("dqn" if "dqn" in os.path.basename(run).lower() else "ppo")
    cls = DQN if algo == "dqn" else PPO
    rl = None
    env_cfg = os.path.join(run, "env_config.yaml")
    if os.path.exists(env_cfg):
        rl = load_config(env_cfg).to_dict()["rl"]
    return SB3Policy(cls.load(path, device="cpu"), name or os.path.basename(run), rl)


def episode_plan(env, n_episodes, levels, base_seed=12345):
    """Evenly spaced start dates across the data, crossed with outage levels."""
    starts = env.valid_starts[np.linspace(0, len(env.valid_starts) - 1, n_episodes).astype(int)]
    plan = []
    for i, s in enumerate(starts):
        for level in levels:
            plan.append(dict(start_index=int(s), level=level, soc=0.5, seed=base_seed + i))
    return plan


def env_for(policy, base_env):
    """An env sharing base_env's simulator but with the policy's own action/observation settings."""
    rl = getattr(policy, "rl", None)
    if not rl:
        return base_env
    cur = base_env.cfg.to_dict()["rl"]
    if rl.get("action_space") == cur["action_space"] and rl.get("observation") == cur["observation"] \
            and rl.get("reserve_levels", cur["reserve_levels"]) == cur["reserve_levels"]:
        return base_env
    cache = getattr(policy, "_env_cache", None)
    if cache is not None and cache[0] is base_env:
        return cache[1]
    cfg = base_env.cfg.copy({"rl": rl})
    env = SolarHomeEnv(cfg, sim=base_env.sim)
    env.tariff = base_env.tariff
    policy._env_cache = (base_env, env)
    return env


def run_episode(env, policy, ep, record=False):
    env = env_for(policy, env)
    obs, info = env.reset(seed=ep["seed"], options=ep)
    trace = []
    while True:
        a = policy.act(env, obs)
        obs, r, term, trunc, info = env.step_dispatch(a) if isinstance(a, Dispatch) else env.step(a)
        if record:
            trace.append(info)
        if term or trunc:
            break
    stats = info["episode_stats"]
    stats["start_time"] = str(env.weather.index[ep["start_index"]])
    stats["month"] = int(env.weather.index[ep["start_index"]].month)
    if getattr(policy, "lp_objective", None) is not None:
        stats["lp_objective_rs"] = policy.lp_objective
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


def summarise(df, by=("policy",), metrics=METRICS):
    """Mean over episodes per seed, then mean and std across seeds."""
    metrics = [m for m in metrics if m in df]
    if "eval_seed" not in df:
        return df.groupby(list(by))[metrics].agg(["mean", "std"])
    per_seed = df.groupby(list(by) + ["eval_seed"])[metrics].mean()
    return per_seed.groupby(level=list(range(len(by))))[metrics].agg(["mean", "std"])


def mean_pm_std(summary, metric):
    m, s = summary[(metric, "mean")], summary[(metric, "std")].fillna(0.0)
    return m.map(lambda x: f"{x:,.2f}") + " ± " + s.map(lambda x: f"{x:,.2f}")


def scatter_plot(summary, path, title="Cost vs. unserved essential load"):
    """Weekly cost vs. unserved essential load; the reserve sweep is drawn as a line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = summary[("unserved_essential_kwh", "mean")]
    y = summary[("total_cost_incl_wear", "mean")]
    xe = summary[("unserved_essential_kwh", "std")].fillna(0)
    ye = summary[("total_cost_incl_wear", "std")].fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sweep = sorted([p for p in summary.index if str(p).startswith("battery_first_r")],
                   key=lambda p: int(str(p).rsplit("r", 1)[1]))
    if sweep:
        ax.plot(x[sweep], y[sweep], "-o", color="#7f7f7f", ms=4, lw=1.2, label="battery_first reserve sweep")
        for p in sweep:
            ax.annotate(str(p).rsplit("_", 1)[1].replace("r", "") + "%", (x[p], y[p]), fontsize=7,
                        xytext=(4, 2), textcoords="offset points", color="#555555")
    others = [p for p in summary.index if p not in sweep]
    for p in others:
        marker = "*" if p == "oracle" else ("D" if any(k in str(p).lower() for k in ("ppo", "dqn")) else "o")
        ax.errorbar(x[p], y[p], xerr=xe[p], yerr=ye[p], fmt=marker, ms=10 if marker == "*" else 6, capsize=3,
                    label=str(p))
    ax.set_xlabel("unserved essential load (kWh/week)")
    ax.set_ylabel("cost incl. battery wear (Rs/week)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
