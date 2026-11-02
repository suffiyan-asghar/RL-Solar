"""Plot one test week for two policies side by side (the demo figure).

Example
    python scripts/plot_episode.py --policies hand_tuned battery_first --level heavy --date 2024-06-03
    python scripts/plot_episode.py --policies ppo=models/ppo_seed0/best_model.zip solar_first
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

from solar_rl import MODES, SimConfig, SolarHomeEnv  # noqa: E402
from solar_rl.baselines import ALL_BASELINES  # noqa: E402
from solar_rl.evaluation import run_episode  # noqa: E402

MODE_COLORS = {"grid_charge": "#7b3294", "solar_charge": "#f1a340", "hold": "#999999", "discharge": "#2c7bb6"}


def make_policy(spec):
    if "=" in spec:
        from scripts.evaluate import load_model
        return load_model(spec)
    return ALL_BASELINES[spec]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["hand_tuned", "battery_first"])
    ap.add_argument("--years", type=int, nargs="+", default=[2024])
    ap.add_argument("--date", default=None, help="YYYY-MM-DD start (midnight)")
    ap.add_argument("--level", default="heavy")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--days", type=int, default=3, help="days shown (of the 7-day episode)")
    ap.add_argument("--out", default="results/episode.png")
    args = ap.parse_args()

    env = SolarHomeEnv(SimConfig(years=args.years))
    if args.date:
        start = int(np.searchsorted(env.weather.index, pd.Timestamp(args.date)))
    else:
        start = int(env.valid_starts[len(env.valid_starts) // 2])
    ep = dict(start_index=start, level=args.level, soc=0.5, seed=args.seed)

    fig, axes = plt.subplots(3, len(args.policies), figsize=(7 * len(args.policies), 9),
                             sharex=True, squeeze=False)
    for j, spec in enumerate(args.policies):
        pol = make_policy(spec)
        stats, tr = run_episode(env, pol, ep, record=True)
        tr = tr.iloc[: 24 * args.days]
        x = pd.to_datetime(tr["time"])
        demand = tr["essential"] + tr["cooling"] + tr["flex"]
        off = ~tr["grid_on"]

        ax = axes[0, j]
        ax.fill_between(x, tr["solar"], step="post", alpha=0.35, color="#f1a340", label="solar")
        ax.step(x, demand, where="post", color="k", lw=1.2, label="demand")
        ax.step(x, tr["import_kwh"], where="post", color="#d7191c", lw=1, label="grid import")
        ax.set_ylabel("kW")
        ax.set_title(f"{pol.name}  |  week cost Rs {stats['total_cost_incl_wear']:,.0f}  |  "
                     f"unserved {stats['unserved_essential_kwh']:.1f} kWh")
        ax.legend(loc="upper left", fontsize=8)

        ax = axes[1, j]
        ax.step(x, tr["soc"] * 100, where="post", color="#2c7bb6", lw=1.6)
        ax.bar(x[tr["unserved_essential"] > 0], 100, width=1 / 24, color="#d7191c", alpha=0.5,
               align="edge", label="essential load unserved")
        ax.set_ylabel("battery %")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left", fontsize=8)

        ax = axes[2, j]
        cols = [MODE_COLORS[m] for m in tr["mode"]]
        ax.bar(x, 1, width=1 / 24, color=cols, align="edge")
        ax.bar(x[tr["run_flex"] == 1], 0.3, width=1 / 24, bottom=1.05, color="#1a9641", align="edge")
        ax.set_yticks([0.5, 1.2]); ax.set_yticklabels(["battery mode", "flex run"])
        ax.set_ylim(0, 1.4)
        handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS[m]) for m in MODES]
        ax.legend(handles, MODES, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.25))

        for a in axes[:, j]:
            for t0, is_off in zip(x, off):
                if is_off:
                    a.axvspan(t0, t0 + pd.Timedelta(hours=1), color="#444444", alpha=0.12, lw=0)

    fig.suptitle(f"Test episode from {env.weather.index[start].date()} | load-shedding: {args.level} "
                 f"(grey = grid off)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
