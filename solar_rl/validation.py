"""Helpers that compare the simulator with itself and with real data (used by
scripts/validate_sim.py, scripts/compare_demand.py and the tests)."""
import numpy as np
import pandas as pd

from .config import LEVELS
from .sim import Dispatch, Simulator
from .sim.outages import outage_events


def simulated_demand(cfg, start=None, end=None, seed=0, sim=None):
    """Hourly total demand (kW) the simulator produces between start and end, with the grid
    always on and the AC on its normal thermostat (flexible load from 08:00). Columns:
    essential, other, ac, flex, total."""
    over = {"outages": {"level": "light", "intensity": 0.0, "model": "semi_markov"},
            "episode": {"forecast_hours": 1}}
    if sim is None:
        c = cfg.copy(over)
        sim = Simulator(c)
    idx = sim.index
    start = pd.Timestamp(start) if start is not None else idx[0]
    end = pd.Timestamp(end) if end is not None else idx[-1]
    i0 = int(np.searchsorted(idx, start.normalize()))
    i1 = int(np.searchsorted(idx, end, side="right"))
    n = max(min(i1, len(idx) - 2) - i0, 1)
    sim.cfg = sim.cfg.copy({"episode": {"hours": n}})
    sim.valid_starts = np.array([i0])
    sim.reset(seed=seed, start_index=i0, level="light", soc=0.5)
    rows = []
    while not sim.done:
        h = sim.hours[sim.t]
        f = sim.step(Dispatch(battery="hold", run_flex=bool(h >= 8)))
        rows.append((f["time"], f["essential"], f["other"], f["cooling"], f["flex"]))
    df = pd.DataFrame(rows, columns=["time", "essential", "other", "ac", "flex"]).set_index("time")
    df["total"] = df[["essential", "other", "ac", "flex"]].sum(axis=1)
    return df


def compare_series(sim_s, real_s):
    """Error metrics between two aligned series (NaNs dropped)."""
    d = pd.concat([sim_s.rename("sim"), real_s.rename("real")], axis=1).dropna()
    if d.empty:
        return {}
    err = d["sim"] - d["real"]
    return {"n": int(len(d)), "mean_real": float(d["real"].mean()), "mean_sim": float(d["sim"].mean()),
            "bias": float(err.mean()), "mae": float(err.abs().mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "mape_pct": float((err.abs() / d["real"].clip(lower=1e-6)).mean() * 100),
            "corr": float(d["sim"].corr(d["real"])) if len(d) > 2 else float("nan")}


def demand_comparison(sim_df, real_hourly):
    """Daily-energy and mean-hourly-profile comparison of simulated vs. logged demand."""
    real = real_hourly.reindex(sim_df.index)
    daily_sim = sim_df["total"].resample("D").sum(min_count=24)
    daily_real = real.resample("D").sum(min_count=24)
    prof_sim = sim_df["total"].groupby(sim_df.index.hour).mean()
    prof_real = real.groupby(real.index.hour).mean()
    return {"hourly": compare_series(sim_df["total"], real), "daily_kwh": compare_series(daily_sim, daily_real),
            "profile": compare_series(prof_sim, prof_real)}, (daily_sim, daily_real, prof_sim, prof_real)


def simulated_outages(cfg, level, n_weeks=52, seed=0, sim=None):
    """Grid availability over n_weeks of simulated weeks (1 on / 0 off), concatenated."""
    sim = sim or Simulator(cfg)
    starts = sim.valid_starts[np.linspace(0, len(sim.valid_starts) - 1, n_weeks).astype(int)]
    out = []
    for i, s in enumerate(starts):
        sim.reset(seed=seed * 1000 + i, start_index=int(s), level=level, soc=0.5, record=False)
        g = []
        while not sim.done:
            g.append(float(sim.grid_on))
            sim.step(Dispatch(battery="hold"))
        out.append(pd.Series(g, index=sim.index[int(s): int(s) + len(g)]))
    return out


def durations(series_list):
    d = []
    for s in series_list:
        d += [n for _, n in outage_events(s)]
    return np.array(d, dtype=float)


def outage_summary(series_list):
    d = durations(series_list)
    total = sum(len(s.dropna()) for s in series_list)
    off = sum(float((s == 0).sum()) for s in series_list)
    return {"hours": int(total), "share_off": off / max(total, 1), "events_per_week": len(d) / max(total / 168, 1e-9),
            "mean_duration_h": float(d.mean()) if len(d) else 0.0,
            "p90_duration_h": float(np.percentile(d, 90)) if len(d) else 0.0}


def ks_distance(a, b):
    """Kolmogorov-Smirnov distance between two samples of durations."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.union1d(a, b)
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


__all__ = ["simulated_demand", "compare_series", "demand_comparison", "simulated_outages", "durations",
           "outage_summary", "ks_distance", "LEVELS"]
