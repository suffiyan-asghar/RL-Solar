"""The physical simulator: one home (or shop) with rooftop PV, a battery behind a hybrid
inverter, the grid with load-shedding, and household demand including one air-conditioned
room. It knows nothing about reinforcement learning, prices or rewards.

    sim = Simulator(load_config("default"))
    sim.reset(seed=0, level="heavy")
    while not sim.done:
        flows = sim.step(Dispatch(battery="reserve", reserve=0.3, run_flex=True))
    sim.log_df().to_csv("week.csv")

Each hour the caller gives a Dispatch (battery mode, flexible load on/off, AC mode) and gets
back the energy flows of that hour (kWh): solar, load served/unserved, battery charge and
discharge (AC side and cell side), grid import/export, curtailment, inverter standby,
plus room temperature, comfort and battery ageing.

Order inside one hour
  1. demand and grid status for the hour were drawn at the end of the previous step
  2. AC demand from the thermal model and the chosen AC mode
  3. energy dispatch (grid on: battery mode decides; grid off: inverter islands and serves
     standby -> essential -> AC -> other -> flexible from solar + battery)
  4. room temperature, battery ageing, flexible-load deadline, log row
  5. advance time, draw next hour's grid status and demand
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import LEVELS
from .battery import make_battery
from .demand import DemandModel
from .outages import OutageModel, load_outage_log
from .pv import pv_power
from .thermal import Room
from .weather import as_weather, load_years

BATTERY_MODES = ["grid_charge", "solar_charge", "hold", "discharge", "reserve", "setpoint"]
LEGACY_MODES = BATTERY_MODES[:4]


@dataclass
class Dispatch:
    """One hour's decision.

    battery   grid_charge   surplus solar and the grid charge the battery
              solar_charge  surplus solar charges the battery, grid covers any deficit
              hold          battery idle, surplus exported, grid covers any deficit
              discharge     battery covers the deficit (down to the grid-on minimum), surplus charges
              reserve       discharge above `reserve` SoC, charge from solar first, charge from
                            the grid only while SoC is below `reserve`
              setpoint      explicit charge_kw / discharge_kw (used by the LP oracle); charging
                            uses surplus solar first, then the grid
    run_flex  run the flexible load this hour (up to its per-hour limit), or flex_kw if given
    ac        normal (setpoint during occupied hours) | precool (lower setpoint, any hour) | off
    ac_kw     explicit AC electric power (overrides `ac`; used by the oracle)
    During an outage the battery mode is ignored: the inverter serves loads automatically.
    """
    battery: str = "solar_charge"
    run_flex: bool = False
    reserve: float = 0.3
    ac: str = "normal"
    charge_kw: float = 0.0
    discharge_kw: float = 0.0
    flex_kw: Optional[float] = None
    ac_kw: Optional[float] = None


def make_rngs(seed, streams="independent", shared=None):
    names = ["episode", "outages", "demand", "forecast"]
    if streams == "shared":
        g = shared if shared is not None else np.random.default_rng(seed)
        return {n: g for n in names}
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {n: np.random.default_rng(s) for n, s in zip(names, children)}


class Simulator:
    def __init__(self, cfg, weather=None):
        self.cfg = cfg
        c = cfg
        self.weather = as_weather(weather) if weather is not None else load_years(c.data.years, c)
        w = self.weather
        self.ghi = w["ghi"].to_numpy()
        self.temp = w["temp_c"].to_numpy()
        self.pv = pv_power(w, c)
        idx = w.index
        self.index = idx
        self.hours = idx.hour.to_numpy()
        self.doy = idx.day_of_year.to_numpy()
        self.months = idx.month.to_numpy()
        self.demand = DemandModel(c, w)
        self.weekend, self.ramadan = self.demand.weekend, self.demand.ramadan
        self.replay = None
        if c.outages.model == "replay":
            log = load_outage_log(c.outages.replay.csv_path, c.outages.replay.off_threshold, c.site.timezone)
            self.replay = log.reindex(idx).to_numpy()
        self.valid_starts = self._valid_starts()
        if len(self.valid_starts) == 0:
            raise ValueError("Not enough weather (or log) data for one episode.")
        self.log = []
        self.done = True

    def _valid_starts(self):
        c = self.cfg
        n = len(self.pv)
        last = n - c.episode.hours - c.episode.forecast_hours - 1
        starts = np.where(self.hours[:last] == 0)[0]
        ok = self.demand.valid_mask()
        if self.replay is not None:
            ok = ok & ~np.isnan(self.replay)
        if not ok.all():
            span = c.episode.hours + 1
            bad = np.convolve(~ok, np.ones(span, dtype=int))[span - 1:]
            starts = starts[bad[starts] == 0]
        return starts

    def fresh_copy(self):
        """A new simulator sharing this one's precomputed weather, PV and demand tables
        (cheap; use one per parallel environment)."""
        import copy
        other = copy.copy(self)
        other.log, other.done = [], True
        return other

    # ------------------------------------------------------------------ episode
    def reset(self, seed=None, start_index=None, level=None, soc=None, shared_rng=None, record=True):
        c = self.cfg
        self.rngs = make_rngs(seed, c.rng.streams, shared_rng)
        r = self.rngs["episode"]
        start = r.choice(self.valid_starts)          # always drawn, so the stream is the same
        self.start = int(start if start_index is None else start_index)
        lvl = c.outages.level if level is None else level
        if c.outages.model == "replay":
            self.level = "replay"
        else:
            self.level = r.choice(LEVELS) if lvl == "random" else lvl
        lo, hi = c.episode.initial_soc_range
        s0 = r.uniform(lo, hi)
        self.batt = make_battery(c.battery, s0 if soc is None else soc)
        self.outages = OutageModel(c, self.level, self.rngs["outages"], self.replay)
        cc = self.demand.cooling_cfg(self.start)
        self.room = Room(cc) if cc is not None and cc.model == "thermal" else None
        self.t = self.start
        self.t_in_episode = 0
        self.flex_remaining = self.demand.flex_daily_kwh(self.t)
        self.hours_since_outage = 24
        self.hours_in_outage = 0
        self.grid_on = True
        self.record = record
        self.log = []
        self.done = False
        self._advance()
        return self

    def _advance(self):
        t = self.t
        was_on = self.grid_on
        self.grid_on = self.outages.step(t, int(self.hours[t]), int(self.months[t]), float(self.temp[t]))
        self.hours_since_outage = 0 if not self.grid_on else self.hours_since_outage + 1
        self.hours_in_outage = 0 if self.grid_on else self.hours_in_outage + 1
        self.transfer = was_on and not self.grid_on
        self.cur_essential, self.cur_other, self.cur_rule_cooling = self.demand.sample(t, self.rngs["demand"])

    # ------------------------------------------------------------------ helpers
    def cooling_cfg(self, t=None):
        return self.demand.cooling_cfg(self.t if t is None else t)

    def occupied(self, t=None):
        t = self.t if t is None else t
        cc = self.demand.cooling_cfg(t)
        return cc is not None and int(self.hours[t]) in cc.occupied_hours

    def ac_demand_kw(self, mode="normal", t=None):
        """AC electric power the thermostat would draw this hour in `mode` (no side effects)."""
        t = self.t if t is None else t
        cc = self.demand.cooling_cfg(t)
        if cc is None:
            return 0.0
        if cc.model == "rule":
            return self.cur_rule_cooling
        if self.room is None or mode == "off":
            return 0.0
        if mode == "precool":
            target = cc.precool_setpoint_c
        elif int(self.hours[t]) in cc.occupied_hours:
            target = cc.setpoint_c
        else:
            return 0.0
        q = self.room.cooling_needed(self.temp[t], self.ghi[t], target, cc)
        return q / cc.ac_cop

    @property
    def soc(self):
        return self.batt.soc

    def flex_max_kw(self, t=None):
        return self.demand.flex_cfg(self.t if t is None else t).max_kw

    def flex_deadline(self, t=None):
        return self.demand.flex_cfg(self.t if t is None else t).deadline_hour

    # ------------------------------------------------------------------ one hour
    def step(self, d: Dispatch):
        if self.done:
            raise RuntimeError("episode finished: call reset()")
        c, t = self.cfg, self.t
        h = int(self.hours[t])
        b = self.batt
        b.begin_hour()
        cc = self.demand.cooling_cfg(t)
        S, E, O = self.pv[t], self.cur_essential, self.cur_other

        # AC demand (kW electric)
        if cc is not None and cc.model == "thermal" and self.room is not None and d.ac_kw is not None:
            C = min(max(d.ac_kw, 0.0), cc.ac_max_kw_elec)
        else:
            C = self.ac_demand_kw(d.ac, t) if cc is not None else 0.0
        fc = self.demand.flex_cfg(t)
        if d.flex_kw is not None:
            F_req = min(self.flex_remaining, max(d.flex_kw, 0.0), fc.max_kw)
        else:
            F_req = min(self.flex_remaining, fc.max_kw) if d.run_flex else 0.0
        standby = c.inverter.standby_w / 1000.0
        b_in_budget = b.charge_limit()
        imp = exp = curtailed = charged = discharged = 0.0
        unserved_E = unserved_O = unmet_C = 0.0
        flex_done = 0.0
        override = False
        mode = d.battery

        if self.grid_on:
            load = E + O + C + F_req + standby
            net = S - load
            if mode in LEGACY_MODES:
                b_out_budget = b.discharge_limit(b.floor_grid)
                if net >= 0:
                    if mode != "hold":
                        charged = b.charge(min(net, b_in_budget))
                    exp = net - charged
                else:
                    deficit = -net
                    if mode == "discharge":
                        discharged = b.discharge(min(deficit, b_out_budget), b.floor_grid)
                        deficit -= discharged
                    imp = deficit
                if mode == "grid_charge":
                    extra = b.charge(max(0.0, b_in_budget - charged))
                    charged += extra
                    imp += extra
            elif mode == "reserve":
                floor = max(d.reserve, b.floor_grid)
                if net >= 0:
                    charged = b.charge(min(net, b_in_budget))
                    exp = net - charged
                else:
                    deficit = -net
                    if b.soc > floor:
                        discharged = b.discharge(min(deficit, b.discharge_limit(floor)), floor)
                        deficit -= discharged
                    imp = deficit
                if b.soc < d.reserve:
                    need = b.ac_to_reach(d.reserve)
                    extra = b.charge(min(need, max(0.0, b_in_budget - charged)))
                    charged += extra
                    imp += extra
            elif mode == "setpoint":
                discharged = b.discharge(min(max(d.discharge_kw, 0.0), b.discharge_limit(b.floor_grid)), b.floor_grid)
                charged = b.charge(min(max(d.charge_kw, 0.0), b_in_budget)) if discharged == 0 else 0.0
                bal = S + discharged - load - charged
                exp, imp = (bal, 0.0) if bal >= 0 else (0.0, -bal)
            else:
                raise ValueError(f"unknown battery mode {mode}")
            flex_done = F_req
            standby_served = standby
        else:
            if mode == "grid_charge":
                override = True
            floor = b.floor_backup
            avail = S + min(b.max_discharge_output(floor), b.discharge_limit(floor))
            cap = c.inverter.backup_limit_kw
            standby_served = min(standby, avail); avail -= standby_served
            served_E = min(E, avail, cap); avail -= served_E; cap -= served_E
            served_C = min(C, avail, cap); avail -= served_C; cap -= served_C
            served_O = min(O, avail, cap); avail -= served_O; cap -= served_O
            served_F = min(F_req, avail, cap)
            if self.transfer and c.inverter.transfer_time_ms > 0:
                served_E -= served_E * c.inverter.transfer_time_ms / 3.6e6   # loads dark during switchover
            unserved_E, unmet_C, unserved_O, flex_done = E - served_E, C - served_C, O - served_O, served_F
            served = standby_served + served_E + served_C + served_O + served_F
            if served > S:
                discharged = b.discharge(served - S, floor)
            else:
                surplus = S - served
                charged = b.charge(min(surplus, b_in_budget))
                curtailed = surplus - charged

        # room temperature and comfort
        room_temp = room_end = np.nan
        comfort = 0.0
        q_cool = 0.0
        if self.room is not None and cc is not None:
            frac = 1.0 if C <= 0 else (C - unmet_C) / C
            q_cool = C * cc.ac_cop * frac
            room_temp, room_end = self.room.step(self.temp[t], self.ghi[t], q_cool, cc)
            if h in cc.occupied_hours:
                comfort = max(room_temp - cc.setpoint_c, 0.0)

        cyc, cal = b.end_hour(float(self.temp[t]))

        self.flex_remaining -= flex_done
        flex_missed = 0.0
        if h == fc.deadline_hour:
            flex_missed = self.flex_remaining
            self.flex_remaining = 0.0

        row = dict(
            time=self.index[t], hour=h, grid_on=bool(self.grid_on), level=self.level,
            mode=mode, reserve=d.reserve if mode == "reserve" else np.nan, ac_mode=d.ac,
            run_flex=int(d.run_flex), override=override,
            temp_out=float(self.temp[t]), ghi=float(self.ghi[t]),
            solar=S, essential=E, other=O, cooling=C, flex=flex_done, standby=standby_served,
            import_kwh=imp, export_kwh=exp, charged=charged, discharged=discharged,
            charged_dc=b.dc_in, discharged_dc=b.dc_out, curtailed=curtailed,
            unserved_essential=unserved_E, unserved_other=unserved_O, unmet_cooling=unmet_C,
            flex_missed=flex_missed, soc=b.soc, soh=b.soh, room_temp=room_temp, ac_cooling_kw=q_cool,
            comfort_degree_hours=comfort, cycle_damage=cyc, calendar_damage=cal,
            transfer_event=bool(self.transfer), weekend=bool(self.weekend[t]), ramadan=bool(self.ramadan[t]),
        )
        if self.record:
            self.log.append(row)

        # advance time
        self.t += 1
        self.t_in_episode += 1
        if self.hours[self.t] == 0:
            self.flex_remaining = self.demand.flex_daily_kwh(self.t)
        self._advance()
        self.done = self.t_in_episode >= c.episode.hours
        return row

    # ------------------------------------------------------------------ output
    def log_df(self):
        import pandas as pd
        return pd.DataFrame(self.log)

    def save_log(self, path):
        self.log_df().to_csv(path, index=False)
