"""Gymnasium wrapper around the simulator: observation, action mapping and reward only.

Action spaces (rl.action_space)
    legacy   Discrete(8) = battery mode {grid_charge, solar_charge, hold, discharge} x flexible {delay, run}
    reserve  Discrete(R x 2 x 2) = target battery reserve {10, 30, 50, 70 %} x flexible {delay, run}
             x AC {normal setpoint, pre-cool}. The dispatch controller (Dispatch battery="reserve")
             discharges above the reserve, charges from solar first and from the grid only while
             SoC is below the reserve.

Observation (rl.observation)
    legacy    30 values: hour and day-of-year (sin/cos), SoC, solar now, demand now, AC demand,
              flexible energy left, hours to deadline, grid on/off, hours since last outage,
              outage level, price, hours to next scheduled outage, scheduled now, temperature,
              episode progress, noisy 6-hour forecasts of solar and temperature
    extended  legacy + room temperature, occupied now, hours into the current outage,
              weekend, Ramadan, state of health (36 values)
    The agent is never told when unscheduled outages will happen.

Reward (Rs x reward.scale)
    -(import cost - export credit) - battery wear
    - penalty_unserved_essential x kWh - penalty_unserved_other x kWh - penalty_unmet_cooling x kWh
    - penalty_comfort_degree_hour x degree-hours - penalty_flex_missed x kWh
    Battery wear comes from the degradation model (cycle + calendar damage x replacement cost),
    or a flat Rs/kWh discharged (reward.wear: flat). Fixed monthly charges are reported in the
    episode stats but left out of the reward because no action changes them.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import LEVELS, SimConfig, load_config
from .sim import LEGACY_MODES, Dispatch, Simulator
from .tariff import Tariff

MODES = LEGACY_MODES
GRID_CHARGE, SOLAR_CHARGE, HOLD, DISCHARGE = range(4)
LEVEL_CODE = {"light": 0.0, "medium": 0.5, "heavy": 1.0, "replay": 0.5}
STAT_KEYS = ["cost", "import_kwh", "export_kwh", "import_cost", "export_credit", "wear_cost", "wear_cycle_cost",
             "wear_calendar_cost", "fixed_charges", "solar_kwh", "curtailed_kwh", "batt_discharge_kwh",
             "batt_charge_kwh", "battery_losses_kwh", "standby_kwh", "unserved_essential_kwh", "unserved_hours",
             "unserved_other_kwh", "unmet_cooling_kwh", "comfort_degree_hours", "ac_kwh", "flex_missed_kwh",
             "outage_hours", "transfer_events", "overrides", "reward"]


def decode_action(a):
    return int(a) // 2, int(a) % 2


def encode_action(mode, run_flex):
    return int(mode) * 2 + int(run_flex)


class SolarHomeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None, weather=None, sim=None):
        super().__init__()
        if cfg is None or isinstance(cfg, str):
            cfg = load_config(cfg or "default")
        self.cfg: SimConfig = cfg
        self.sim = sim if sim is not None else Simulator(cfg, weather)
        self.weather = self.sim.weather
        self.tariff = Tariff(cfg.tariff)
        rl = cfg.rl
        self.reserves = list(rl.reserve_levels)
        n_actions = 8 if rl.action_space == "legacy" else len(self.reserves) * 4
        self.action_space = spaces.Discrete(n_actions)
        H = cfg.episode.forecast_hours
        self.obs_dim = 18 + 2 * H + (6 if rl.observation == "extended" else 0)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.obs_dim,), dtype=np.float32)
        wc = cfg.battery
        self.replacement_cost = wc.replacement_cost_rs_per_kwh * wc.capacity_kwh

    # ------------------------------------------------------------ convenience views
    @property
    def valid_starts(self):
        return self.sim.valid_starts

    @property
    def level(self):
        return self.sim.level

    @property
    def t(self):
        return self.sim.t

    @property
    def hours(self):
        return self.sim.hours

    @property
    def pv(self):
        return self.sim.pv

    def action_to_dispatch(self, action):
        a = int(action)
        if self.cfg.rl.action_space == "legacy":
            mode, run_flex = decode_action(a)
            return Dispatch(battery=MODES[mode], run_flex=bool(run_flex))
        ac = a % 2
        run_flex = (a // 2) % 2
        r = a // 4
        return Dispatch(battery="reserve", reserve=self.reserves[r], run_flex=bool(run_flex),
                        ac="precool" if ac else "normal")

    def dispatch_to_action(self, d):
        """Nearest action for a Dispatch (used to show baselines in the agent's action terms)."""
        if self.cfg.rl.action_space == "legacy":
            return encode_action(MODES.index(d.battery) if d.battery in MODES else SOLAR_CHARGE, d.run_flex)
        r = int(np.argmin([abs(x - d.reserve) for x in self.reserves]))
        return (r * 2 + int(d.run_flex)) * 2 + int(d.ac == "precool")

    # ------------------------------------------------------------ observation
    def _observe(self):
        c, s = self.cfg, self.sim
        t = s.t
        h = s.hours[t]
        d = s.doy[t]
        fc = s.demand.flex_cfg(t)
        cc = s.cooling_cfg(t)
        ac_norm = (cc.kw if cc.model == "rule" else cc.ac_max_kw_elec) if cc is not None else 1.0
        hours_to_deadline = max(fc.deadline_hour - h, 0)
        obs = [
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365),
            s.batt.soc,
            s.pv[t] / c.pv.kwp,
            (s.cur_essential + s.cur_other) / 1.0,
            s.ac_demand_kw("normal") / max(ac_norm, 1e-6),
            s.flex_remaining / max(fc.daily_kwh, 1e-6),
            hours_to_deadline / 24.0,
            float(s.grid_on),
            min(s.hours_since_outage, 24) / 24.0,
            LEVEL_CODE.get(str(s.level), 0.5),
            self.tariff.import_rate(s.months[t], h) / 60.0,
            s.outages.hours_to_next_scheduled(h) / 12.0,
            float(s.outages.scheduled(h)),
            (s.temp[t] - 25.0) / 10.0,
            s.t_in_episode / c.episode.hours,
        ]
        H = c.episode.forecast_hours
        rng = s.rngs["forecast"]
        for k in range(1, H + 1):
            sigma = c.episode.solar_forecast_noise * (0.5 + k / H)
            obs.append(s.pv[t + k] / c.pv.kwp * max(0.0, 1 + rng.normal(0, sigma)))
        for k in range(1, H + 1):
            obs.append((s.temp[t + k] + rng.normal(0, c.episode.temp_forecast_noise_c * k / H) - 25.0) / 10.0)
        if c.rl.observation == "extended":
            room = s.room.temp if s.room is not None else 25.0
            obs += [(room - 25.0) / 10.0, float(s.occupied()), min(s.hours_in_outage, 12) / 12.0,
                    float(s.weekend[t]), float(s.ramadan[t]), (s.batt.soh - 0.7) / 0.3]
        return np.clip(np.array(obs, dtype=np.float32), -5, 5)

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if self.cfg.rng.streams == "shared":
            self.sim.reset(shared_rng=self.np_random, start_index=options.get("start_index"),
                           level=options.get("level"), soc=options.get("soc"))
        else:
            sim_seed = int(self.np_random.integers(0, 2 ** 63 - 1))
            self.sim.reset(seed=sim_seed, start_index=options.get("start_index"),
                           level=options.get("level"), soc=options.get("soc"))
        self.stats = {k: 0.0 for k in STAT_KEYS}
        return self._observe(), {"level": self.sim.level, "start_time": str(self.weather.index[self.sim.t])}

    def step(self, action):
        return self.step_dispatch(self.action_to_dispatch(action))

    def step_dispatch(self, d: Dispatch):
        c, s = self.cfg, self.sim
        t = s.t
        f = s.step(d)
        rw = c.reward
        month, h = int(s.months[t]), f["hour"]
        import_cost, export_credit, fixed, price = self.tariff.hour_cost(
            f["import_kwh"], f["export_kwh"], month, h, int(s.index[t].year))
        if rw.wear == "flat":
            wear = f["discharged"] * rw.flat_wear_rs_per_kwh
            wear_cyc, wear_cal = wear, 0.0
        else:
            wear_cyc = f["cycle_damage"] * self.replacement_cost
            wear_cal = f["calendar_damage"] * self.replacement_cost if rw.include_calendar_wear else 0.0
            wear = wear_cyc + wear_cal
        penalty = (rw.penalty_unserved_essential * f["unserved_essential"]
                   + rw.penalty_unserved_other * f["unserved_other"]
                   + rw.penalty_unmet_cooling * f["unmet_cooling"]
                   + rw.penalty_flex_missed * f["flex_missed"]
                   + rw.penalty_comfort_degree_hour * f["comfort_degree_hours"])
        rupees = -(import_cost - export_credit) - wear - penalty
        reward = rupees * rw.scale

        info = dict(f)
        info["time"] = str(f["time"])
        info["mode"] = f["mode"]
        info.update(price=price, import_cost=import_cost, export_credit=export_credit, wear_cost=wear,
                    fixed_charge=fixed, penalty=penalty, reward=reward)

        st = self.stats
        st["import_kwh"] += f["import_kwh"]; st["export_kwh"] += f["export_kwh"]
        st["import_cost"] += import_cost; st["export_credit"] += export_credit
        st["cost"] += import_cost - export_credit; st["wear_cost"] += wear
        st["wear_cycle_cost"] += wear_cyc; st["wear_calendar_cost"] += wear_cal
        st["fixed_charges"] += fixed
        st["solar_kwh"] += f["solar"]; st["curtailed_kwh"] += f["curtailed"]
        st["batt_discharge_kwh"] += f["discharged"]; st["batt_charge_kwh"] += f["charged"]
        st["battery_losses_kwh"] += (f["charged"] - f["charged_dc"]) + (f["discharged_dc"] - f["discharged"])
        st["standby_kwh"] += f["standby"]
        st["unserved_essential_kwh"] += f["unserved_essential"]
        st["unserved_hours"] += float(f["unserved_essential"] > 1e-3)   # ignore switchover blips
        st["unserved_other_kwh"] += f["unserved_other"]
        st["unmet_cooling_kwh"] += f["unmet_cooling"]; st["flex_missed_kwh"] += f["flex_missed"]
        st["comfort_degree_hours"] += f["comfort_degree_hours"]
        st["ac_kwh"] += f["cooling"] - f["unmet_cooling"]
        st["outage_hours"] += float(not f["grid_on"]); st["overrides"] += float(f["override"])
        st["transfer_events"] += float(f["transfer_event"])
        st["reward"] += reward

        truncated = s.done
        if truncated:
            info["episode_stats"] = self.episode_stats()
        return self._observe(), float(reward), False, truncated, info

    def episode_stats(self):
        st = dict(self.stats)
        st["battery_cycles"] = st["batt_discharge_kwh"] / self.cfg.battery.capacity_kwh
        used = st["solar_kwh"] - st["export_kwh"] - st["curtailed_kwh"]
        st["self_consumption"] = used / st["solar_kwh"] if st["solar_kwh"] > 0 else 0.0
        st["total_cost_incl_wear"] = st["cost"] + st["wear_cost"]
        st["bill_incl_fixed"] = st["cost"] + st["fixed_charges"]
        st["soh_end"] = self.sim.batt.soh
        st["level"] = self.sim.level
        return st
