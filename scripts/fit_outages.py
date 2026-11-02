"""Fit load-shedding model parameters from a recorded outage log.

    python scripts/fit_outages.py --log data/outage_log.csv
    python scripts/fit_outages.py --log data/outage_log.csv --level heavy --out results/outage_fit

The log is a CSV with timestamp and grid_on (1/0); rows can be regular samples or state
changes. Prints and saves (fit.yaml) a configs-ready `outages` block with the fitted
semi-Markov parameters and the recurring (scheduled) hours, and plots the duration histogram
with every fitted distribution. Paste the block into a scenario YAML.
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
import yaml  # noqa: E402

from solar_rl import load_config  # noqa: E402
from solar_rl.sim.outages import DISTRIBUTIONS, fit_outage_model, load_outage_log  # noqa: E402
from solar_rl.sim.weather import load_years  # noqa: E402


def fit_log(log, level="heavy", scheduled_share=0.6, off_threshold=0.5):
    cfg = load_config("default")
    g = load_outage_log(log, off_threshold, cfg.site.timezone)
    years = sorted(set(g.dropna().index.year))
    temp = None
    try:
        temp = load_years(years, load_config("default", {"data": {"years": years, "allow_synthetic": False}}))["temp_c"]
    except FileNotFoundError:
        print("(no weather for the log's years: heat factor not fitted)")
    fit = fit_outage_model(g, temp, scheduled_share, cfg.outages.semi_markov.heat_threshold_c)
    sm = fit["semi_markov"]
    start = dict(cfg.outages.semi_markov.start_prob)
    start[level] = sm["start_prob_per_hour"]
    block = {"outages": {
        "model": "semi_markov",
        "semi_markov": {"start_prob": start, "hour_factor": sm["hour_factor"], "month_factor": sm["month_factor"],
                        "heat_threshold_c": sm["heat_threshold_c"], "heat_factor_per_c": sm["heat_factor_per_c"],
                        "duration": sm["duration"]}}}
    return fit, block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--level", default="heavy", help="which level the fitted start probability is for")
    ap.add_argument("--scheduled-share", type=float, default=0.6,
                    help="an hour of day that is off on more than this share of days counts as scheduled")
    ap.add_argument("--off-threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results/outage_fit")
    args = ap.parse_args()
    fit, block = fit_log(args.log, args.level, args.scheduled_share, args.off_threshold)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "fit.yaml")
    with open(path, "w") as f:
        f.write(f"# fitted from {args.log}: {fit['n_hours']} h, {fit['n_outages']} unscheduled outages, "
                f"grid off {fit['share_off']:.1%} of the time\n")
        f.write(f"# recurring (scheduled) hours of day: {fit['scheduled_hours']} -> describe them in a schedule CSV\n")
        f.write("# duration fits (AIC, lower is better): " +
                ", ".join(f"{k} {v['aic']:.1f}" for k, v in fit["duration_fits"].items()) + "\n")
        yaml.safe_dump(block, f, sort_keys=False)
    print(open(path).read())

    d = np.array(fit["durations"])
    if len(d):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(d, bins=np.arange(0.5, d.max() + 1.5), density=True, alpha=0.5, color="#999999",
                label=f"log ({len(d)} outages)")
        k = np.arange(1, int(d.max()) + 1)
        for name, f_ in fit["duration_fits"].items():
            dist = DISTRIBUTIONS[name](**f_["params"])
            ax.plot(k, dist.cdf(k) - dist.cdf(k - 1), "-o", ms=3, label=f"{name} (AIC {f_['aic']:.0f})")
        ax.set_xlabel("outage duration (hours)")
        ax.set_ylabel("share of outages")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "durations.png"), dpi=130)
        print(f"Saved {path} and {args.out}/durations.png")


if __name__ == "__main__":
    main()
