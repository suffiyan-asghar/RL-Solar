"""Rule-based baselines. Each policy has act(env, obs) -> action.

They read the environment's true current state (hour, SoC, grid status),
which is the same information a real inverter has. None of them see the future.
"""
import numpy as np

from .env import DISCHARGE, GRID_CHARGE, HOLD, SOLAR_CHARGE, encode_action


def naive_flex(env):
    """What most households do: run the motor/washing in the morning from 8:00."""
    h = env.hours[env.t]
    return int(env.flex_remaining > 1e-6 and h >= 8)


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs):
        return int(self.rng.integers(0, 8))


class SolarFirst:
    """Solar -> load, surplus -> battery, grid covers deficit. Battery only used in outages."""
    name = "solar_first"

    def act(self, env, obs):
        return encode_action(SOLAR_CHARGE, naive_flex(env))


class BatteryFirst:
    """Solar -> load, then battery, then grid (SBU mode). Saves grid units, risks empty battery."""
    name = "battery_first"

    def act(self, env, obs):
        return encode_action(DISCHARGE, naive_flex(env))


class GridFirst:
    """Grid serves load; battery kept full as backup (charges from grid if needed)."""
    name = "grid_first"

    def act(self, env, obs):
        mode = GRID_CHARGE if env.batt.soc < env.cfg.soc_max - 0.02 else HOLD
        return encode_action(mode, naive_flex(env))


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
        c = env.cfg
        h = env.hours[env.t]
        soc = env.batt.soc
        heavy = env.level == "heavy"
        reserve = self.reserve_risky if (h in self.risky_hours or heavy) else self.reserve_normal
        sched_soon = env.outages.hours_to_next_scheduled(h) <= 3

        if soc < reserve and (sched_soon or h in self.risky_hours) and h not in c.peak_hours:
            mode = GRID_CHARGE
        elif soc > reserve and h in c.peak_hours:
            mode = DISCHARGE
        else:
            mode = SOLAR_CHARGE

        solar_now = env.pv[env.t]
        hours_left = c.flex_deadline_hour - h
        need_hours = int(np.ceil(env.flex_remaining / c.flex_max_kw)) if env.flex_remaining > 1e-6 else 0
        run = int(need_hours > 0 and (solar_now >= 1.5 or hours_left <= need_hours))
        return encode_action(mode, run)


ALL_BASELINES = {p.name: p for p in [RandomPolicy, SolarFirst, BatteryFirst, GridFirst, HandTunedRule]}
