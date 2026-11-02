"""Plot one test week for two policies side by side (the demo figure).

    python scripts/plot_episode.py --policies hand_tuned battery_first_r30 --level heavy --date 2024-06-03
    python scripts/plot_episode.py --policies ppo=models/ppo_seed0/best_model.zip oracle --csv results/week
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from solar_rl import SolarHomeEnv, load_config  # noqa: E402
from solar_rl.baselines import make_baseline  # noqa: E402
from solar_rl.evaluation import load_model, run_episode  # noqa: E402

MODE_COLORS = {"grid_charge": "#7b3294", "solar_charge": "#f1a340", "hold": "#999999", "discharge": "#2c7bb6",
               "reserve": "#4daf4a", "setpoint": "#e7298a"}


def make_policy(spec):
    if "=" in spec:
        name, path = spec.split("=", 1)
        return load_model(path, name)
    return make_baseline(spec)


def plot_traces(traces, title, out, days=3):
    n = len(traces)
    fig, axes = plt.subplots(4, n, figsize=(7 * n, 11), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [3, 2, 2, 1.4]})
    for j, (name, stats, tr) in enumerate(traces):
        tr = tr.iloc[: 24 * days]
        x = pd.to_datetime(tr["time"])
        demand = tr["essential"] + tr["other"] + tr["cooling"] + tr["flex"]
        off = ~tr["grid_on"].astype(bool)
        ax = axes[0, j]
        ax.fill_between(x, tr["solar"], step="post", alpha=0.35, color="#f1a340", label="solar")
        ax.step(x, demand, where="post", color="k", lw=1.2, label="demand")
        ax.step(x, tr["cooling"], where="post", color="#1f78b4", lw=0.9, ls="--", label="AC")
        ax.step(x, tr["import_kwh"], where="post", color="#d7191c", lw=1, label="grid import")
        ax.set_ylabel("kW")
        ax.set_title(f"{name}  |  week cost Rs {stats['total_cost_incl_wear']:,.0f}  |  "
                     f"unserved {stats['unserved_essential_kwh']:.1f} kWh  |  "
                     f"{stats['comfort_degree_hours']:.0f} °C·h")
        ax.legend(loc="upper left", fontsize=8)
        ax = axes[1, j]
        ax.step(x, tr["soc"] * 100, where="post", color="#2c7bb6", lw=1.6)
        ax.bar(x[tr["unserved_essential"] > 1e-3], 100, width=1 / 24, color="#d7191c", alpha=0.5,
               align="edge", label="essential load unserved")
        ax.set_ylabel("battery %")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left", fontsize=8)
        ax = axes[2, j]
        if tr["room_temp"].notna().any():
            ax.step(x, tr["room_temp"], where="post", color="#e66101", lw=1.4, label="room")
        ax.step(x, tr["temp_out"], where="post", color="#999999", lw=1, label="outdoor")
        ax.set_ylabel("°C")
        ax.legend(loc="upper left", fontsize=8)
        ax = axes[3, j]
        ax.bar(x, 1, width=1 / 24, color=[MODE_COLORS.get(m, "#000") for m in tr["mode"]], align="edge")
        ax.bar(x[tr["run_flex"] == 1], 0.3, width=1 / 24, bottom=1.05, color="#1a9641", align="edge")
        if "ac_mode" in tr:
            ax.bar(x[tr["ac_mode"] == "precool"], 0.3, width=1 / 24, bottom=1.4, color="#1f78b4", align="edge")
        ax.set_yticks([0.5, 1.2, 1.55])
        ax.set_yticklabels(["battery mode", "flex run", "pre-cool"])
        ax.set_ylim(0, 1.75)
        used = [m for m in MODE_COLORS if m in set(tr["mode"])]
        handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS[m]) for m in used]
        ax.legend(handles, used, loc="upper center", ncol=len(used), fontsize=7, bbox_to_anchor=(0.5, -0.35))
        for a in axes[:, j]:
            for t0, is_off in zip(x, off):
                if is_off:
                    a.axvspan(t0, t0 + pd.Timedelta(hours=1), color="#444444", alpha=0.12, lw=0)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["hand_tuned", "battery_first_r30"])
    ap.add_argument("--scenario", default="default")
    ap.add_argument("--years", type=int, nargs="+", default=[2024])
    ap.add_argument("--date", default=None, help="YYYY-MM-DD start (midnight)")
    ap.add_argument("--level", default="heavy")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--days", type=int, default=3, help="days shown (of the 7-day episode)")
    ap.add_argument("--out", default="results/episode.png")
    ap.add_argument("--csv", default=None, help="also save each policy's hourly log to <csv>_<policy>.csv")
    args = ap.parse_args()

    env = SolarHomeEnv(load_config(args.scenario, {"data": {"years": args.years}}))
    if args.date:
        start = int(np.searchsorted(env.weather.index, pd.Timestamp(args.date)))
    else:
        start = int(env.valid_starts[len(env.valid_starts) // 2])
    ep = dict(start_index=start, level=args.level, soc=0.5, seed=args.seed)
    traces = []
    for spec in args.policies:
        pol = make_policy(spec)
        stats, tr = run_episode(env, pol, ep, record=True)
        traces.append((pol.name, stats, tr))
        if args.csv:
            env.sim.save_log(f"{args.csv}_{pol.name}.csv")
    plot_traces(traces, f"{args.scenario}: test episode from {env.weather.index[start].date()} | "
                        f"load-shedding: {args.level} (grey = grid off)", args.out, args.days)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
