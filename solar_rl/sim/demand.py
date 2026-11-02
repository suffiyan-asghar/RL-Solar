"""Household / shop electricity demand.

appliances  Each appliance in the YAML profile has a power (kW), the hours it may run and
            the probability it is on in each of those hours (one independent draw per
            appliance-hour). Optional temperature scaling (fans) and per-hour factors.
            Weekend and Ramadan variants override appliances and AC occupancy. The hourly
            total is multiplied by lognormal noise. Air conditioning is NOT an appliance:
            it comes from the room thermal model (thermal.py) or, in the original
            model, from a temperature rule.
measured    Replays a real inverter-app log (timestamp, load_kw): the logged load becomes
            the base demand. The log already contains the AC and the flexible loads, so
            the AC model and the flexible load are switched off.
"""
import datetime as dt

import numpy as np
import pandas as pd


def load_usage_log(path, timezone=None):
    """Read an inverter-app usage log and return hourly mean load (kW), hour-beginning labels.

    Columns: a timestamp column (timestamp/time/datetime) and load_kw (or load_w).
    Timestamps may be at any resolution (e.g. 5-minute samples); hours with less than half
    of their samples present are left as NaN.
    """
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = next((cols[c] for c in ("timestamp", "time", "datetime", "date_time") if c in cols), None)
    if tcol is None:
        raise ValueError(f"{path}: need a timestamp column (timestamp/time/datetime)")
    if "load_kw" in cols:
        load = df[cols["load_kw"]].astype(float)
    elif "load_w" in cols:
        load = df[cols["load_w"]].astype(float) / 1000.0
    else:
        raise ValueError(f"{path}: need a load_kw (or load_w) column")
    t = pd.to_datetime(df[tcol])
    if t.dt.tz is not None:
        t = t.dt.tz_convert(timezone).dt.tz_localize(None) if timezone else t.dt.tz_localize(None)
    s = pd.Series(load.to_numpy(), index=t).sort_index()
    s = s[~s.index.duplicated()]
    step = s.index.to_series().diff().median()
    per_hour = max(int(round(pd.Timedelta(hours=1) / step)), 1) if pd.notna(step) else 1
    g = s.resample("h")
    hourly = g.mean()
    hourly[g.count() < 0.5 * per_hour] = np.nan
    hourly.name = "load_kw"
    return hourly


def day_types(index, demand_cfg):
    """Boolean arrays (weekend, ramadan) for each timestamp."""
    weekend = np.isin(index.day_of_week, demand_cfg.weekend_days)
    ramadan = np.zeros(len(index), dtype=bool)
    dates = index.normalize()
    for first, last in demand_cfg.ramadan_dates.values():
        ramadan |= (dates >= pd.Timestamp(first)) & (dates <= pd.Timestamp(last))
    return weekend, ramadan


class DemandModel:
    def __init__(self, cfg, weather):
        self.cfg = cfg
        d = cfg.demand
        self.noise = d.noise
        self.index = weather.index
        self.temp = weather["temp_c"].to_numpy()
        self.hours = weather.index.hour.to_numpy()
        self.weekend, self.ramadan = day_types(weather.index, d)
        base = d.profile
        self.profiles = {(w, r): base.variant(w, r) if (w or r) else base
                         for w in (False, True) for r in (False, True)}
        self.measured = None
        if d.model == "measured":
            log = load_usage_log(d.measured_log, cfg.site.timezone)
            self.measured = log.reindex(weather.index).to_numpy()
            self.measured_priority = d.measured_priority

    def profile_at(self, t):
        return self.profiles[(bool(self.weekend[t]), bool(self.ramadan[t]))]

    def cooling_cfg(self, t):
        if self.measured is not None:
            return None
        c = self.profile_at(t).cooling
        return None if c.model == "none" else c

    def flex_cfg(self, t):
        return self.profile_at(t).flexible

    def flex_daily_kwh(self, t):
        return 0.0 if self.measured is not None else self.profile_at(t).flexible.daily_kwh

    def valid_mask(self):
        """Hours where demand is defined (all, unless replaying a log with gaps)."""
        if self.measured is None:
            return np.ones(len(self.index), dtype=bool)
        return ~np.isnan(self.measured)

    def sample(self, t, rng):
        """Draw this hour's demand. Returns (essential_kw, other_kw, rule_cooling_kw)."""
        h, T = self.hours[t], self.temp[t]
        if self.measured is not None:
            v = float(self.measured[t])
            return (v, 0.0, 0.0) if self.measured_priority == "essential" else (0.0, v, 0.0)
        prof = self.profile_at(t)
        ess = 0.0
        oth = 0.0
        for a in prof.appliances:
            if h not in a.hours:
                continue
            if 0.0 < a.probability < 1.0:
                if rng.random() >= a.probability:
                    continue
            elif a.probability == 0.0:
                continue
            p = a.kw
            if a.temp_scale is not None:
                lo, hi = a.temp_scale
                p = p * np.clip((T - lo) / (hi - lo), 0.0, 1.0)
            if a.hour_factor:
                p = p * a.hour_factor.get(int(h), 1.0)
            if a.priority == "essential":
                ess += p
            else:
                oth += p
        n1, n2 = rng.lognormal(0, self.noise, 2)
        cool = 0.0
        c = prof.cooling
        if c.model == "rule" and h in c.hours:
            cool = c.kw * float(np.clip((T - c.temp_low_c) / (c.temp_high_c - c.temp_low_c), 0.0, 1.0))
        return ess * n1, oth * n1, cool * n2

    def expected_profile(self, t0, n_hours, rng, n_samples=20):
        """Mean simulated base demand (essential + other) for hours t0..t0+n-1 (for validation)."""
        out = np.zeros(n_hours)
        for _ in range(n_samples):
            for k in range(n_hours):
                e, o, _ = self.sample(t0 + k, rng)
                out[k] += e + o
        return out / n_samples
