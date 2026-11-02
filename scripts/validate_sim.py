"""Validation report for the simulator: plots in results/validation/ and report.md.

    python scripts/validate_sim.py
    python scripts/validate_sim.py --inverter-log data/inverter.csv --outage-log data/outages.csv
    python scripts/validate_sim.py --quick          # fewer weeks, for a fast check

Sections
  1. monthly solar yield (pvlib with/without soiling, original model), kWh/kWp
  2. daily demand profile: weekday, weekend, Ramadan (+ real inverter log: error metrics)
  3. outage durations per level (+ real outage log: share off, events/week, KS distance)
  4. battery SoC behaviour under each baseline: mean SoC by hour, SoC histogram, limit checks
  5. annual cost of every baseline in the test year (52 weekly episodes per outage level)
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.plot_monthly_energy import monthly_yield  # noqa: E402
from solar_rl import SolarHomeEnv, Simulator, load_config  # noqa: E402
from solar_rl.baselines import make_baseline  # noqa: E402
from solar_rl.config import LEVELS  # noqa: E402
from solar_rl.evaluation import episode_plan, evaluate, run_episode  # noqa: E402
from solar_rl.sim.demand import load_usage_log  # noqa: E402
from solar_rl.sim.outages import load_outage_log  # noqa: E402
from solar_rl.validation import (demand_comparison, durations, ks_distance, outage_summary,  # noqa: E402
                                 simulated_demand, simulated_outages)

BASELINES = ["random", "solar_first", "battery_first", "grid_first", "hand_tuned", "battery_first_r30", "oracle"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=120)
    plt.close(fig)
    return name


def section_solar(cfg, years, out, md):
    per_year, hourly, source = monthly_yield(cfg, years)
    monthly = per_year.groupby(level="month").mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    m = np.arange(1, 13)
    ax.bar(m - 0.27, monthly["pvlib"], 0.27, label="pvlib (soiling)", color="#f1a340")
    ax.bar(m, monthly["pvlib_no_soiling"], 0.27, label="pvlib (clean)", color="#fdd49e")
    ax.bar(m + 0.27, monthly["simple"], 0.27, label="original model", color="#999999")
    ax.set_xticks(m)
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("kWh/kWp")
    ax.set_title(f"Monthly solar yield ({min(years)}-{max(years)}, {source})")
    ax.legend()
    img = save(fig, out, "1_solar_monthly.png")
    annual = per_year.groupby(level="year").sum()
    md += ["## 1. Solar yield\n", f"![]({img})\n", "Annual kWh/kWp per year:\n", annual.round(0).to_markdown(),
           f"\nMean soiling loss {hourly['soiling'].mean():.1%}. Compare with Global Solar Atlas / PVGIS "
           f"for Karachi at tilt {cfg.pv.tilt:g}°, azimuth {cfg.pv.azimuth:g}°.\n"]
    return {"annual_kwh_per_kwp": annual["pvlib"].round(1).to_dict()}


def section_demand(cfg, out, md, log):
    sim_df = simulated_demand(cfg, seed=0)
    w = sim_df.index.day_of_week.isin(cfg.demand.weekend_days)
    sim = Simulator(cfg.copy({"episode": {"forecast_hours": 1}}))
    ram = pd.Series(sim.ramadan, index=sim.index).reindex(sim_df.index).fillna(False).to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, mask, col in (("weekday", ~w & ~ram, "#2c7bb6"), ("weekend", w & ~ram, "#d7191c"),
                             ("Ramadan", ram, "#1a9641")):
        if mask.any():
            p = sim_df["total"][mask].groupby(sim_df.index[mask].hour).mean()
            ax.plot(p.index, p.values, "-o", ms=3, color=col, label=f"simulated {label}")
    p_ac = sim_df["ac"].groupby(sim_df.index.hour).mean()
    ax.plot(p_ac.index, p_ac.values, ":", color="#1f78b4", label="of which AC (all days)")
    res = {"daily_kwh_sim": float(sim_df["total"].resample("D").sum().mean())}
    if log:
        real = load_usage_log(log, cfg.site.timezone)
        known = real.dropna()
        cfg_l = cfg.copy({"data": {"years": sorted(set(known.index.year))}})
        sdf = simulated_demand(cfg_l, known.index[0], known.index[-1], seed=0)
        metrics, (_, _, ps, pr) = demand_comparison(sdf, real)
        ax.plot(pr.index, pr.values, "k-o", ms=3, label="real log (mean)")
        res["vs_log"] = metrics
    ax.set_xlabel("hour of day")
    ax.set_ylabel("kW")
    ax.set_title(f"Daily demand profile ({cfg.demand.profile.name})")
    ax.legend(fontsize=8)
    img = save(fig, out, "2_demand_profile.png")
    md += ["## 2. Demand\n", f"![]({img})\n", f"Mean simulated demand {res['daily_kwh_sim']:.1f} kWh/day "
           f"({cfg.demand.profile.name} profile, grid always on, AC on its normal thermostat).\n"]
    if "vs_log" in res:
        md += ["Simulated vs. real inverter log:\n",
               pd.DataFrame(res["vs_log"]).T.round(3).to_markdown(), ""]
    return res


def section_outages(cfg, out, md, log, weeks):
    sim = Simulator(cfg)
    fig, ax = plt.subplots(figsize=(9, 4))
    rows = {}
    sims = {}
    for i, level in enumerate(LEVELS):
        s = simulated_outages(cfg, level, weeks, seed=1, sim=sim)
        d = durations(s)
        sims[level] = d
        rows[level] = outage_summary(s)
        if len(d):
            ax.hist(d + (i - 1) * 0.25, bins=np.arange(0.5, 14.5, 1) + (i - 1) * 0.25, alpha=0.5, label=level,
                    density=True, width=0.25)
    res = {"simulated": rows}
    if log:
        g = load_outage_log(log, cfg.outages.replay.off_threshold, cfg.site.timezone)
        real = [g.dropna()]
        dr = durations(real)
        rows["real log"] = outage_summary(real)
        ax.hist(dr, bins=np.arange(0.5, 14.5, 1), histtype="step", color="k", lw=1.5, density=True,
                label="real log")
        res["ks_distance"] = {lvl: ks_distance(sims[lvl], dr) for lvl in LEVELS}
    ax.set_xlabel("outage duration (hours, scheduled blocks included)")
    ax.set_ylabel("share of outages")
    ax.set_title(f"Outage durations ({cfg.outages.model}, {weeks} weeks per level)")
    ax.legend()
    img = save(fig, out, "3_outage_durations.png")
    md += ["## 3. Load-shedding\n", f"![]({img})\n", pd.DataFrame(rows).T.round(3).to_markdown(), ""]
    if "ks_distance" in res:
        md += ["KS distance between simulated and logged durations: " +
               ", ".join(f"{k} {v:.3f}" for k, v in res["ks_distance"].items()) + "\n"]
    return res


def section_battery(cfg, out, md, episodes):
    env = SolarHomeEnv(cfg)
    plan = episode_plan(env, episodes, ["medium"], base_seed=777)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    res = {}
    for name in BASELINES[1:]:
        pol = make_baseline(name)
        socs = []
        for ep in plan:
            _, tr = run_episode(env, pol, ep, record=True)
            socs.append(tr[["hour", "soc", "grid_on"]])
        d = pd.concat(socs)
        prof = d.groupby("hour")["soc"].mean() * 100
        axes[0].plot(prof.index, prof.values, "-", label=name)
        axes[1].hist(d["soc"] * 100, bins=np.arange(0, 102, 2), histtype="step", lw=1.3, density=True, label=name)
        res[name] = {"min_soc": float(d["soc"].min()), "max_soc": float(d["soc"].max()),
                     "min_soc_grid_on": float(d.loc[d["grid_on"], "soc"].min())}
    b = env.sim.batt
    axes[0].axhline(b.floor_grid * 100, color="k", ls=":", lw=0.8)
    axes[0].axhline(b.floor_backup * 100, color="r", ls=":", lw=0.8)
    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("mean SoC (%)")
    axes[0].set_title("Battery SoC by hour (medium load-shedding)")
    axes[1].set_xlabel("SoC (%)")
    axes[1].set_title("SoC distribution")
    axes[0].legend(fontsize=7)
    img = save(fig, out, "4_battery_soc.png")
    lim = {"floor_grid": b.floor_grid, "floor_backup": b.floor_backup, "soc_max": b.soc_max}
    ok = all(v["min_soc"] >= lim["floor_backup"] - 1e-9 and v["max_soc"] <= lim["soc_max"] + 1e-9 for v in res.values())
    md += ["## 4. Battery\n", f"![]({img})\n",
           f"Limits: {lim}. All SoC values within limits: **{ok}**.\n",
           pd.DataFrame(res).T.round(3).to_markdown(), ""]
    return {"limits_ok": ok, **res}


def section_cost(cfg, out, md, weeks):
    env = SolarHomeEnv(cfg)
    plan = episode_plan(env, weeks, LEVELS, base_seed=2024)
    df = evaluate(env, [make_baseline(n) for n in BASELINES], plan)
    scale = 52.0 / weeks
    annual = df.groupby(["policy", "level"])[["total_cost_incl_wear", "bill_incl_fixed", "unserved_essential_kwh",
                                               "comfort_degree_hours"]].sum() * scale
    cost = annual["total_cost_incl_wear"].unstack("level")[LEVELS]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cost.sort_values("medium").plot.bar(ax=ax, rot=20)
    ax.set_ylabel("Rs per year (energy + battery wear)")
    ax.set_title(f"Annual cost per baseline, {cfg.data.years[0]} ({weeks} weeks per level, scaled to 52)")
    img = save(fig, out, "5_annual_cost.png")
    md += ["## 5. Annual cost\n", f"![]({img})\n",
           "Rs per year (energy + battery wear):\n", cost.round(0).to_markdown(),
           "\nUnserved essential load (kWh/year):\n",
           annual["unserved_essential_kwh"].unstack("level")[LEVELS].round(1).to_markdown(), ""]
    return {"annual_cost": cost.round(0).to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="default")
    ap.add_argument("--test-year", type=int, default=2024)
    ap.add_argument("--solar-years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024])
    ap.add_argument("--inverter-log", default=None, help="CSV timestamp,load_kw")
    ap.add_argument("--outage-log", default=None, help="CSV timestamp,grid_on")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/validation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.scenario, {"data": {"years": [args.test_year]}})
    weeks = 6 if args.quick else 52
    md = [f"# Simulator validation: {args.scenario}\n",
          f"Test year {args.test_year}. Generated by scripts/validate_sim.py.\n"]
    res = {}
    res["solar"] = section_solar(cfg, args.solar_years if not args.quick else [args.test_year], args.out, md)
    print("solar done")
    res["demand"] = section_demand(cfg, args.out, md, args.inverter_log)
    print("demand done")
    res["outages"] = section_outages(cfg, args.out, md, args.outage_log, weeks)
    print("outages done")
    res["battery"] = section_battery(cfg, args.out, md, 4 if args.quick else 12)
    print("battery done")
    res["cost"] = section_cost(cfg, args.out, md, 4 if args.quick else 52)
    print("cost done")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"Saved {args.out}/report.md, results.json and plots")


if __name__ == "__main__":
    main()
