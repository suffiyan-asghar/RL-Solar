"""Compare simulated daily demand with a real inverter-app usage log.

    python scripts/compare_demand.py --log data/my_inverter_log.csv
    python scripts/compare_demand.py --log data/my_inverter_log.csv --scenario shop

The log is a CSV with a timestamp column and load_kw (or load_w); any sampling interval.
The simulator is run over the same dates (weather for those years is needed in data/, or
synthetic weather is used) with the grid always on and the AC on its normal thermostat.
Reports error metrics for hourly load, daily energy and the mean hourly profile, and saves
plots to --out. Use the result to calibrate configs/demand/<profile>.yaml.
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from solar_rl import load_config  # noqa: E402
from solar_rl.sim.demand import load_usage_log  # noqa: E402
from solar_rl.validation import demand_comparison, simulated_demand  # noqa: E402


def compare(log_path, scenario="default", seed=0):
    cfg0 = load_config(scenario)
    real = load_usage_log(log_path, cfg0.site.timezone)
    known = real.dropna()
    years = sorted(set(known.index.year))
    cfg = load_config(scenario, {"data": {"years": years}})
    sim_df = simulated_demand(cfg, known.index[0], known.index[-1], seed=seed)
    metrics, curves = demand_comparison(sim_df, real)
    return metrics, sim_df, real, curves


def plot(out, scenario, curves, metrics):
    daily_sim, daily_real, prof_sim, prof_real = curves
    os.makedirs(out, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    ax = axes[0]
    ax.plot(prof_real.index, prof_real.values, "k-o", ms=3, label="real log")
    ax.plot(prof_sim.index, prof_sim.values, "-o", color="#e66101", ms=3, label=f"simulated ({scenario})")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean load (kW)")
    ax.set_title("Mean daily profile (MAE %.2f kW)" % metrics["profile"].get("mae", float("nan")))
    ax.legend()
    ax = axes[1]
    ax.plot(daily_real.index, daily_real.values, "k-", lw=1, label="real")
    ax.plot(daily_sim.index, daily_sim.values, "-", color="#e66101", lw=1, label="simulated")
    ax.set_ylabel("kWh/day")
    ax.set_title("Daily energy (bias %+.1f kWh/day)" % metrics["daily_kwh"].get("bias", float("nan")))
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    ax = axes[2]
    d = daily_real.to_frame("real").join(daily_sim.rename("sim")).dropna()
    ax.scatter(d["real"], d["sim"], s=12)
    top = max(float(d.max().max()) * 1.05, 1.0) if len(d) else 1.0
    ax.plot([0, top], [0, top], "k--", lw=0.8)
    ax.set_xlabel("real kWh/day")
    ax.set_ylabel("simulated kWh/day")
    ax.set_title("Daily energy, r = %.2f" % metrics["daily_kwh"].get("corr", float("nan")))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "demand_vs_log.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="CSV with timestamp and load_kw (or load_w)")
    ap.add_argument("--scenario", default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/demand_compare")
    args = ap.parse_args()
    metrics, sim_df, real, curves = compare(args.log, args.scenario, args.seed)
    plot(args.out, args.scenario, curves, metrics)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    for k, m in metrics.items():
        print(f"{k:10s} " + "  ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in m.items()))
    print(f"Saved {args.out}/demand_vs_log.png and metrics.json")


if __name__ == "__main__":
    main()
