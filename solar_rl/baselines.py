"""Rule-based baselines. Each policy has act(env, obs) -> Dispatch (or an action index).

They read the simulator's current state (hour, SoC, grid status, published schedule), the same
information a real inverter has. None of them see the future, except the oracle
(solar_rl/oracle.py), which is an upper bound.
"""
import numpy as np

from .sim import LEGACY_MODES, Dispatch


def naive_flex(env):
    """What most households do: run the motor/washing in the morning from 8:00."""
    s = env.sim
    return bool(s.flex_remaining > 1e-6 and s.hours[s.t] >= 8)


def peak_hours(env):
    s = env.sim
    rates = env.tariff.rates[s.months[s.t] - 1]
    return set(np.where(rates > rates.min())[0].tolist())


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs):
        a = int(self.rng.integers(0, 8))
        return Dispatch(battery=LEGACY_MODES[a // 2], run_flex=bool(a % 2))


class SolarFirst:
    """Solar -> load, surplus -> battery, grid covers deficit. Battery only used in outages."""
    name = "solar_first"

    def act(self, env, obs):
        return Dispatch(battery="solar_charge", run_flex=naive_flex(env))


class BatteryFirst:
    """Solar -> load, then battery, then grid (SBU mode). Saves grid units, risks an empty battery."""
    name = "battery_first"

    def act(self, env, obs):
        return Dispatch(battery="discharge", run_flex=naive_flex(env))


class BatteryFirstReserve:
    """battery_first, but never discharges below a fixed reserve (charging from the grid if below it)."""

    def __init__(self, reserve=0.3):
        self.reserve = reserve
        self.name = f"battery_first_r{int(round(reserve * 100))}"

    def act(self, env, obs):
        return Dispatch(battery="reserve", reserve=self.reserve, run_flex=naive_flex(env))


class GridFirst:
    """Grid serves load; battery kept full as backup (charges from grid if needed)."""
    name = "grid_first"

    def act(self, env, obs):
        b = env.sim.batt
        mode = "grid_charge" if b.soc < b.soc_max - 0.02 else "hold"
        return Dispatch(battery=mode, run_flex=naive_flex(env))


class HandTunedRule:
    """Strongest heuristic: keep a reserve before risky hours, use battery at peak price,
    run flexible loads when solar is strong.
    """
    name = "hand_tuned"

    def __init__(self, reserve_risky=0.6, reserve_normal=0.3, risky_hours=range(16, 24)):
        self.reserve_risky = reserve_risky
        self.reserve_normal = reserve_normal
        self.risky_hours = set(risky_hours)

    def act(self, env, obs):
        s = env.sim
        h = int(s.hours[s.t])
        soc = s.batt.soc
        peak = peak_hours(env)
        heavy = s.level == "heavy"
        reserve = self.reserve_risky if (h in self.risky_hours or heavy) else self.reserve_normal
        sched_soon = s.outages.hours_to_next_scheduled(h) <= 3

        if soc < reserve and (sched_soon or h in self.risky_hours) and h not in peak:
            mode = "grid_charge"
        elif soc > reserve and h in peak:
            mode = "discharge"
        else:
            mode = "solar_charge"

        fc = s.demand.flex_cfg(s.t)
        solar_now = s.pv[s.t]
        hours_left = fc.deadline_hour - h
        need_hours = int(np.ceil(s.flex_remaining / fc.max_kw)) if s.flex_remaining > 1e-6 else 0
        strong_sun = solar_now >= 0.3 * env.cfg.pv.kwp
        run = need_hours > 0 and (strong_sun or hours_left <= need_hours)
        return Dispatch(battery=mode, run_flex=bool(run))


RESERVE_SWEEP = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
ALL_BASELINES = {p.name: p for p in [RandomPolicy, SolarFirst, BatteryFirst, GridFirst, HandTunedRule]}


def make_baseline(name):
    """Baseline by name: one of ALL_BASELINES, battery_first_r<NN>, or oracle."""
    if name in ALL_BASELINES:
        return ALL_BASELINES[name]()
    if name.startswith("battery_first_r"):
        return BatteryFirstReserve(int(name.rsplit("r", 1)[1]) / 100.0)
    if name == "oracle":
        from .oracle import OraclePolicy
        return OraclePolicy()
    raise KeyError(f"unknown baseline {name}")


def default_baselines(include_oracle=True):
    names = list(ALL_BASELINES) + [f"battery_first_r{int(r * 100)}" for r in RESERVE_SWEEP]
    return names + (["oracle"] if include_oracle else [])
