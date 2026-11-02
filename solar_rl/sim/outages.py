"""Load-shedding: is the grid on in each hour?

Grid off = scheduled block OR unscheduled outage.

Scheduled blocks
    synthetic  n blocks of k hours a day, evenly spaced with a random offset (per episode)
    csv        a published schedule: rows (feeder_category, start HH:MM, end HH:MM); the
               category used for each outage level is set in schedule.category_by_level
Unscheduled outages
    markov       (original) 2-state Markov chain: start probability per hour (x heat, x
                 evening), constant end probability -> geometric durations
    semi_markov  start probability = base(level) x hour_factor x month_factor x heat factor;
                 when an outage starts, its duration is drawn from a fitted distribution
    replay       a recorded outage log (timestamp, grid_on) replaces both
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

from ..config import LEVELS

DISTRIBUTIONS = {"lognormal": stats.lognorm, "weibull": stats.weibull_min, "gamma": stats.gamma,
                 "exponential": stats.expon}


# ----------------------------------------------------------------------------- schedules
def read_schedule_csv(path):
    """Published schedule -> {category: set of hours}. Times are local, end exclusive."""
    df = pd.read_csv(path, dtype=str).fillna("")
    need = {"feeder_category", "start", "end"}
    if not need <= set(df.columns):
        raise ValueError(f"{path}: columns must include {sorted(need)}")
    out = {}
    for _, r in df.iterrows():
        hours = out.setdefault(r["feeder_category"].strip(), set())
        if not r["start"].strip():
            continue
        h0 = _parse_hhmm(r["start"])
        h1 = _parse_hhmm(r["end"])
        n = (h1 - h0) % 24 or 24
        hours.update((h0 + k) % 24 for k in range(n))
    return out


def _parse_hhmm(s):
    hh, mm = s.strip().split(":")
    h = int(hh) % 24
    if int(mm) >= 30:       # hourly model: a block starting at hh:30 or later starts next hour
        h = (h + 1) % 24
    return h


class Schedule:
    def __init__(self, cfg, level, rng):
        sc = cfg.outages.schedule
        self.hours = set()
        if sc.source == "synthetic" and level in LEVELS:
            n = sc.blocks_per_day[level]
            length = sc.block_hours[level]
            if n > 0:
                offset = int(rng.integers(0, 24 // n))
                for b in range(n):
                    start = (offset + b * (24 // n)) % 24
                    for k in range(length):
                        self.hours.add((start + k) % 24)
        elif sc.source == "csv":
            cat = sc.category_by_level.get(level)
            if cat:
                table = read_schedule_csv(sc.csv_path)
                if cat not in table:
                    raise ValueError(f"schedule category '{cat}' not in {sc.csv_path} ({sorted(table)})")
                self.hours = table[cat]

    def scheduled(self, hour):
        return hour in self.hours

    def hours_to_next(self, hour, cap=12):
        for k in range(1, cap + 1):
            if (hour + k) % 24 in self.hours:
                return k
        return cap


# ----------------------------------------------------------------------------- unscheduled
class MarkovOutages:
    """Original model. One random draw per hour."""

    def __init__(self, cfg, level, rng):
        self.m = cfg.outages.markov
        self.intensity = cfg.outages.intensity
        self.level, self.rng = level, rng
        self.on = False

    def step(self, t, hour, month, temp_c):
        m = self.m
        p_start = m.start_prob[self.level]
        p_start *= 1.0 + m.heat_factor_per_c * max(temp_c - m.heat_threshold_c, 0.0)
        if hour in m.evening_hours:
            p_start *= m.evening_factor
        if self.intensity != 1.0:
            p_start *= self.intensity
        if self.on:
            if self.rng.random() < m.end_prob:
                self.on = False
        elif self.rng.random() < p_start:
            self.on = True
        return self.on


class SemiMarkovOutages:
    def __init__(self, cfg, level, rng):
        self.s = cfg.outages.semi_markov
        self.intensity = cfg.outages.intensity
        self.level, self.rng = level, rng
        d = self.s.duration
        self.dist = DISTRIBUTIONS[d.distribution](**d.params)
        self.max_hours = d.max_hours
        self.remaining = 0

    def start_prob(self, hour, month, temp_c):
        s = self.s
        p = s.start_prob[self.level] * s.hour_factor[hour] * s.month_factor[month - 1]
        p *= 1.0 + s.heat_factor_per_c * max(temp_c - s.heat_threshold_c, 0.0)
        return min(p * self.intensity, 1.0)

    def sample_duration(self):
        x = float(self.dist.rvs(random_state=self.rng))
        return int(min(max(math.ceil(x), 1), self.max_hours))

    def step(self, t, hour, month, temp_c):
        if self.remaining > 0:
            self.remaining -= 1
            return True
        if self.rng.random() < self.start_prob(hour, month, temp_c):
            self.remaining = self.sample_duration() - 1
            return True
        return False


class ReplayOutages:
    def __init__(self, grid_on):
        self.grid_on = grid_on          # array aligned with the weather index (NaN = unknown)

    def step(self, t, hour, month, temp_c):
        return not bool(self.grid_on[t])


class OutageModel:
    """Combines the schedule and the unscheduled model. step() -> True if the grid is ON."""

    def __init__(self, cfg, level, rng, replay=None):
        self.level = level
        self.schedule = Schedule(cfg, level, rng)
        model = cfg.outages.model
        if model == "replay":
            self.unsched = ReplayOutages(replay)
            self.schedule.hours = set()
        elif model == "markov":
            self.unsched = MarkovOutages(cfg, level, rng)
        else:
            self.unsched = SemiMarkovOutages(cfg, level, rng)

    def scheduled(self, hour):
        return self.schedule.scheduled(hour)

    def hours_to_next_scheduled(self, hour, cap=12):
        return self.schedule.hours_to_next(hour, cap)

    def step(self, t, hour, month, temp_c):
        off = self.unsched.step(t, hour, month, temp_c)
        return not (off or self.schedule.scheduled(hour))


# ----------------------------------------------------------------------------- logs & fitting
def load_outage_log(path, off_threshold=0.5, timezone=None):
    """Recorded outage log -> hourly grid availability (1.0 on, 0.0 off, NaN unknown).

    CSV columns: timestamp and grid_on (1/0, true/false, on/off). Rows can be regular
    samples or state changes; the state is held until the next row. An hour counts as
    OFF if the grid was off for at least `off_threshold` of it.
    """
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = next((cols[c] for c in ("timestamp", "time", "datetime") if c in cols), None)
    if tcol is None or "grid_on" not in cols:
        raise ValueError(f"{path}: need columns timestamp and grid_on")
    raw = df[cols["grid_on"]].astype(str).str.strip().str.lower()
    val = raw.map({"1": 1.0, "0": 0.0, "true": 1.0, "false": 0.0, "on": 1.0, "off": 0.0, "1.0": 1.0, "0.0": 0.0})
    if val.isna().any():
        raise ValueError(f"{path}: grid_on must be 1/0, true/false or on/off")
    t = pd.to_datetime(df[tcol])
    if t.dt.tz is not None:
        t = t.dt.tz_convert(timezone).dt.tz_localize(None) if timezone else t.dt.tz_localize(None)
    s = pd.Series(val.to_numpy(), index=t).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    minute = s.resample("min").ffill()
    frac_on = minute.resample("h").mean()
    hourly = (1.0 - frac_on < off_threshold).astype(float)
    hourly[frac_on.isna()] = np.nan
    hourly.name = "grid_on"
    return hourly


def outage_events(grid_on):
    """List of (start timestamp, duration hours) of contiguous off periods."""
    off = (grid_on == 0).astype(int)
    events = []
    start = None
    for ts, v in off.items():
        if v and start is None:
            start, n = ts, 0
        if v:
            n += 1
        elif start is not None:
            events.append((start, n))
            start = None
    if start is not None:
        events.append((start, n))
    return events


def fit_outage_model(grid_on, temp_c=None, scheduled_share=0.6, heat_threshold_c=32.0):
    """Fit semi-Markov parameters from an hourly grid_on series (1/0, NaN = unknown).

    1. Hours of the day that are off on more than `scheduled_share` of days are treated as
       a recurring schedule and excluded from the unscheduled fit.
    2. Durations of the remaining outages: lognormal, Weibull, gamma and exponential are
       fitted by maximum likelihood (to duration - 0.5 h, the middle of the hourly bucket);
       the lowest AIC wins.
    3. Start probability per on-hour, then hour-of-day and month factors relative to it, and
       a linear heat factor above `heat_threshold_c` (least squares), if temperatures are given.
    """
    g = grid_on.dropna()
    off_by_hour = 1 - g.groupby(g.index.hour).mean().reindex(range(24), fill_value=1.0)
    sched_hours = sorted(int(h) for h in off_by_hour.index if off_by_hour[h] > scheduled_share)
    unsched = g.copy()
    unsched[np.isin(unsched.index.hour, sched_hours)] = np.nan
    ev = [(s, d) for s, d in outage_events(unsched.dropna()) if s.hour not in sched_hours]
    durations = np.array([d for _, d in ev], dtype=float)

    fits = {}
    if len(durations) >= 5:
        x = np.clip(durations - 0.5, 0.05, None)
        for name, dist in DISTRIBUTIONS.items():
            try:
                params = dist.fit(x, floc=0)
                ll = np.sum(dist.logpdf(x, *params))
                k = len(params) - 1
                fits[name] = {"aic": float(2 * k - 2 * ll), "params": _named_params(name, params)}
            except Exception:  # noqa: BLE001 - a failed fit just drops out of the comparison
                continue
    best = min(fits, key=lambda n: fits[n]["aic"]) if fits else None

    # start probability: P(outage starts at hour t | grid on at t-1)
    prev_on = unsched.shift(1) == 1
    starts = prev_on & (unsched == 0)
    at_risk = prev_on & unsched.notna()
    base = starts.sum() / max(at_risk.sum(), 1)

    def factor(key, n, offset=0):
        out = []
        for k in range(n):
            m = key == k + offset
            r = starts[m].sum() / max(at_risk[m].sum(), 1)
            out.append(round(float(r / base), 3) if base > 0 and at_risk[m].sum() > 0 else 1.0)
        return out

    hour_factor = factor(unsched.index.hour, 24)
    month_factor = factor(unsched.index.month, 12, offset=1)
    heat = 0.0
    if temp_c is not None and base > 0:
        T = temp_c.reindex(unsched.index)
        hot = at_risk & (T > heat_threshold_c)
        if hot.sum() > 20:
            excess = (T[hot] - heat_threshold_c).to_numpy()
            y = starts[hot].astype(float).to_numpy() / base - 1.0
            heat = float(max(np.sum(excess * y) / max(np.sum(excess ** 2), 1e-9), 0.0))
    return {
        "n_hours": int(len(g)), "n_outages": int(len(ev)), "share_off": float(1 - g.mean()),
        "scheduled_hours": sched_hours,
        "duration_fits": fits,
        "semi_markov": {
            "start_prob_per_hour": round(float(base), 5),
            "hour_factor": hour_factor,
            "month_factor": month_factor,
            "heat_threshold_c": heat_threshold_c,
            "heat_factor_per_c": round(heat, 4),
            "duration": {"distribution": best, "params": fits[best]["params"] if best else {},
                         "max_hours": int(max(durations.max(), 1)) if len(durations) else 12},
        },
        "durations": durations.tolist(),
    }


def _named_params(name, params):
    if name == "lognormal":
        return {"s": float(params[0]), "scale": float(params[2])}
    if name in ("weibull", "gamma"):
        key = "c" if name == "weibull" else "a"
        return {key: float(params[0]), "scale": float(params[2])}
    return {"scale": float(params[1])}
