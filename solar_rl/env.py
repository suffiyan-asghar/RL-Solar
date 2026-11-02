"""Gymnasium environment: hourly control of rooftop solar + battery under load-shedding.

Actions (Discrete 8) = battery mode (4) x flexible load (2)
    battery mode: 0 grid_charge  - surplus solar AND grid charge the battery
                  1 solar_charge - surplus solar charges battery, grid covers deficit
                  2 hold         - battery idle, surplus exported, grid covers deficit
                  3 discharge    - battery covers deficit first, surplus charges battery
    flexible:     0 delay, 1 run (up to flex_max_kw this hour)

During an outage the inverter islands automatically: solar + battery serve
essential -> cooling -> flexible load, whatever the chosen mode. Grid charging
during an outage is impossible and is logged as an override.

Reward (Rs, scaled by cfg.reward_scale):
    -(import cost - export credit) - battery wear
    - lambda1 * unserved essential kWh - lambda3 * unmet cooling kWh
    - flex penalty * flexible kWh not done by the deadline
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .components import Battery, OutageModel, cooling_kw, essential_kw, pv_power_kw
from .config import LEVELS, SimConfig
from .weather import load_years

MODES = ["grid_charge", "solar_charge", "hold", "discharge"]
GRID_CHARGE, SOLAR_CHARGE, HOLD, DISCHARGE = range(4)


def decode_action(a):
    return int(a) // 2, int(a) % 2


def encode_action(mode, run_flex):
    return int(mode) * 2 + int(run_flex)


class SolarHomeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: SimConfig = None, weather=None):
        super().__init__()
        self.cfg = cfg or SimConfig()
        c = self.cfg
        self.weather = weather if weather is not None else load_years(c.years, c)
        self.ghi = self.weather["ghi"].to_numpy()
        self.temp = self.weather["temp_c"].to_numpy()
        self.pv = pv_power_kw(self.ghi, self.temp, c)
        idx = self.weather.index
        self.hours = idx.hour.to_numpy()
        self.doy = idx.dayofyear.to_numpy()
        last = len(self.pv) - c.episode_hours - c.forecast_hours - 1
        self.valid_starts = np.where(self.hours[:last] == 0)[0]
        if len(self.valid_starts) == 0:
            raise ValueError("Not enough weather data for one episode.")

        self.action_space = spaces.Discrete(8)
        self.obs_dim = 18 + 2 * c.forecast_hours
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------ helpers
    def _price(self, hour):
        return self.cfg.import_peak if hour in self.cfg.peak_hours else self.cfg.import_offpeak

    def _sample_demand(self):
        """Draw this hour's demand (with noise) for the current time index."""
        c, t = self.cfg, self.t
        h, T = self.hours[t], self.temp[t]
        n1, n2 = self.np_random.lognormal(0, c.load_noise, 2)
        self.cur_essential = essential_kw(h, T) * n1
        self.cur_cooling = cooling_kw(h, T, c) * n2

    def _observe(self):
        c, t = self.cfg, self.t
        h = self.hours[t]
        d = self.doy[t]
        hours_to_deadline = max(c.flex_deadline_hour - h, 0)
        obs = [
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365),
            self.batt.soc,
            self.pv[t] / c.pv_kwp,
            self.cur_essential / 1.0,
            self.cur_cooling / max(c.ac_kw, 1e-6),
            self.flex_remaining / max(c.flex_daily_kwh, 1e-6),
            hours_to_deadline / 24.0,
            float(self.grid_on),
            min(self.hours_since_outage, 24) / 24.0,
            LEVELS.index(self.level) / 2.0,
            self._price(h) / 60.0,
            self.outages.hours_to_next_scheduled(h) / 12.0,
            float(self.outages.scheduled(h)),
            (self.temp[t] - 25.0) / 10.0,
            self.t_in_episode / c.episode_hours,
        ]
        # noisy forecasts: error grows with lead time
        for k in range(1, c.forecast_hours + 1):
            sigma = c.forecast_noise * (0.5 + k / c.forecast_hours)
            obs.append(self.pv[t + k] / c.pv_kwp * max(0.0, 1 + self.np_random.normal(0, sigma)))
        for k in range(1, c.forecast_hours + 1):
            obs.append((self.temp[t + k] + self.np_random.normal(0, 1.5 * k / c.forecast_hours) - 25.0) / 10.0)
        return np.clip(np.array(obs, dtype=np.float32), -5, 5)

    def _advance_grid(self):
        self.grid_on = self.outages.step(self.hours[self.t], self.temp[self.t])
        self.hours_since_outage = 0 if not self.grid_on else self.hours_since_outage + 1

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        c = self.cfg
        options = options or {}
        self.start = int(options.get("start_index", self.np_random.choice(self.valid_starts)))
        self.t = self.start
        self.t_in_episode = 0
        lvl = options.get("level", c.outage_level)
        self.level = self.np_random.choice(LEVELS) if lvl == "random" else lvl
        self.batt = Battery(c, options.get("soc", self.np_random.uniform(0.2, 0.9)))
        self.outages = OutageModel(c, self.level, self.np_random)
        self.flex_remaining = c.flex_daily_kwh
        self.hours_since_outage = 24
        self._advance_grid()
        self._sample_demand()
        self.stats = {k: 0.0 for k in [
            "cost", "import_kwh", "export_kwh", "import_cost", "export_credit", "wear_cost",
            "solar_kwh", "curtailed_kwh", "batt_discharge_kwh", "batt_charge_kwh",
            "unserved_essential_kwh", "unserved_hours", "unmet_cooling_kwh", "flex_missed_kwh",
            "outage_hours", "overrides", "reward"]}
        return self._observe(), {"level": self.level, "start_time": str(self.weather.index[self.t])}

    def step(self, action):
        c, t = self.cfg, self.t
        mode, run_flex = decode_action(action)
        h = self.hours[t]
        S, E, C = self.pv[t], self.cur_essential, self.cur_cooling
        F_req = min(self.flex_remaining, c.flex_max_kw) if run_flex else 0.0
        b = self.batt
        b_in_budget = c.batt_max_kw          # per-hour power limits shared across uses
        b_out_budget = c.batt_max_kw

        imp = exp = curtailed = charged = discharged = 0.0
        unserved_E = unmet_C = 0.0
        flex_done = 0.0
        override = False

        if self.grid_on:
            load = E + C + F_req
            net = S - load
            if net >= 0:
                if mode != HOLD:
                    charged = b.charge(min(net, b_in_budget))
                exp = net - charged
            else:
                deficit = -net
                if mode == DISCHARGE:
                    discharged = b.discharge(min(deficit, b_out_budget))
                    deficit -= discharged
                imp = deficit
            if mode == GRID_CHARGE:
                extra = b.charge(max(0.0, b_in_budget - charged))
                charged += extra
                imp += extra
            flex_done = F_req
        else:
            if mode == GRID_CHARGE:
                override = True
            avail = S + min(b.max_discharge_output(), b_out_budget)
            served_E = min(E, avail); avail -= served_E
            served_C = min(C, avail); avail -= served_C
            served_F = min(F_req, avail)
            unserved_E, unmet_C, flex_done = E - served_E, C - served_C, served_F
            served = served_E + served_C + served_F
            if served > S:
                discharged = b.discharge(served - S)
            else:
                surplus = S - served
                charged = b.charge(min(surplus, b_in_budget))
                curtailed = surplus - charged

        self.flex_remaining -= flex_done
        flex_missed = 0.0
        if h == c.flex_deadline_hour:
            flex_missed = self.flex_remaining
            self.flex_remaining = 0.0

        price = self._price(h)
        import_cost, export_credit = imp * price, exp * c.export_rate
        wear = discharged * c.wear_cost_per_kwh
        penalty = (c.penalty_unserved_essential * unserved_E + c.penalty_unmet_cooling * unmet_C
                   + c.penalty_flex_missed * flex_missed)
        rupees = -(import_cost - export_credit) - wear - penalty
        reward = rupees * c.reward_scale

        info = dict(time=str(self.weather.index[t]), hour=int(h), mode=MODES[mode], run_flex=run_flex,
                    grid_on=bool(self.grid_on), solar=S, essential=E, cooling=C, flex=flex_done,
                    import_kwh=imp, export_kwh=exp, charged=charged, discharged=discharged,
                    curtailed=curtailed, unserved_essential=unserved_E, unmet_cooling=unmet_C,
                    flex_missed=flex_missed, soc=b.soc, price=price, override=override,
                    level=self.level)

        s = self.stats
        s["import_kwh"] += imp; s["export_kwh"] += exp
        s["import_cost"] += import_cost; s["export_credit"] += export_credit
        s["cost"] += import_cost - export_credit; s["wear_cost"] += wear
        s["solar_kwh"] += S; s["curtailed_kwh"] += curtailed
        s["batt_discharge_kwh"] += discharged; s["batt_charge_kwh"] += charged
        s["unserved_essential_kwh"] += unserved_E; s["unserved_hours"] += float(unserved_E > 1e-6)
        s["unmet_cooling_kwh"] += unmet_C; s["flex_missed_kwh"] += flex_missed
        s["outage_hours"] += float(not self.grid_on); s["overrides"] += float(override)
        s["reward"] += reward

        # advance time
        self.t += 1
        self.t_in_episode += 1
        if self.hours[self.t] == 0:
            self.flex_remaining = c.flex_daily_kwh
        self._advance_grid()
        self._sample_demand()

        terminated = False
        truncated = self.t_in_episode >= c.episode_hours
        if truncated:
            info["episode_stats"] = self.episode_stats()
        return self._observe(), float(reward), terminated, truncated, info

    def episode_stats(self):
        s = dict(self.stats)
        s["battery_cycles"] = s["batt_discharge_kwh"] / self.cfg.batt_kwh
        used = s["solar_kwh"] - s["export_kwh"] - s["curtailed_kwh"]
        s["self_consumption"] = used / s["solar_kwh"] if s["solar_kwh"] > 0 else 0.0
        s["total_cost_incl_wear"] = s["cost"] + s["wear_cost"]
        s["level"] = self.level
        return s
