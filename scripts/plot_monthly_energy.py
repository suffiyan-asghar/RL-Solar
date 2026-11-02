"""Monthly PV energy (kWh/kWp) to sanity-check the solar model against published Karachi yields.

    python scripts/plot_monthly_energy.py --years 2020 2021 2022 2023 2024
    python scripts/plot_monthly_energy.py --reference my_reference.csv   # columns: month, kwh_per_kwp

Plots the pvlib model (with and without soiling) and the original simple model per month,
averaged over the chosen years, and saves a CSV. For a published comparison, take the
monthly PVOUT for Karachi from the Global Solar Atlas (https://globalsolaratlas.info, use the
same tilt and azimuth) or PVGIS and pass it with --reference.
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

from solar_rl import load_config  # noqa: E402
from solar_rl.sim.pv import pv_power_simple, pvlib_components  # noqa: E402
from solar_rl.sim.weather import load_years  # noqa: E402


def monthly_yield(cfg, years):
    w = load_years(years, cfg)
    comp = pvlib_components(w, cfg)
    clean_cfg = cfg.copy({"pv": {"soiling_model": "none"}})
    clean = pvlib_components(w, clean_cfg)["pac_kw"]
    simple = pv_power_simple(w["ghi"].to_numpy(), w["temp_c"].to_numpy(), cfg.pv)
    df = pd.DataFrame({"pvlib": comp["pac_kw"], "pvlib_no_soiling": clean, "simple": simple,
                       "soiling": comp["soiling"]}, index=w.index)
    per_year = df[["pvlib", "pvlib_no_soiling", "simple"]].groupby([df.index.year, df.index.month]).sum() / cfg.pv.kwp
    per_year.index.names = ["year", "month"]
    return per_year, df, w.attrs.get("source")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="default")
    ap.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024])
    ap.add_argument("--reference", default=None, help="CSV with month (1-12) and kwh_per_kwp")
    ap.add_argument("--out", default="results/pv_monthly")
    args = ap.parse_args()
    cfg = load_config(args.scenario)
    per_year, hourly, source = monthly_yield(cfg, args.years)
    monthly = per_year.groupby(level="month").mean()
    spread = per_year.groupby(level="month")["pvlib"].agg(["min", "max"])
    os.makedirs(args.out, exist_ok=True)
    per_year.round(2).to_csv(os.path.join(args.out, "monthly_kwh_per_kwp_by_year.csv"))
    monthly.round(2).to_csv(os.path.join(args.out, "monthly_kwh_per_kwp_mean.csv"))

    fig, ax = plt.subplots(figsize=(10, 5))
    m = np.arange(1, 13)
    w = 0.27
    ax.bar(m - w, monthly["pvlib"], w, label="pvlib (with soiling)", color="#f1a340",
           yerr=[monthly["pvlib"] - spread["min"], spread["max"] - monthly["pvlib"]], capsize=2)
    ax.bar(m, monthly["pvlib_no_soiling"], w, label="pvlib (clean panels)", color="#fdd49e")
    ax.bar(m + w, monthly["simple"], w, label="original simple model", color="#999999")
    if args.reference:
        ref = pd.read_csv(args.reference)
        ax.plot(ref["month"], ref["kwh_per_kwp"], "k-o", label="reference")
    p = cfg.pv
    annual = monthly.sum()
    ax.set_title(f"Monthly PV yield, {cfg.site.name}, tilt {p.tilt:g}°, azimuth {p.azimuth:g}° "
                 f"({min(args.years)}-{max(args.years)}, {source})\n"
                 f"annual kWh/kWp: pvlib {annual['pvlib']:.0f}, clean {annual['pvlib_no_soiling']:.0f}, "
                 f"simple {annual['simple']:.0f}")
    ax.set_xticks(m)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("kWh per kWp")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "monthly_yield.png"), dpi=130)
    print(monthly.round(1).to_string())
    print(f"\nAnnual kWh/kWp (mean of {len(args.years)} years): " +
          ", ".join(f"{k} {v:.0f}" for k, v in annual.items()))
    print(f"Mean soiling loss {hourly['soiling'].mean():.1%}. Saved {args.out}/monthly_yield.png and CSVs")


if __name__ == "__main__":
    main()
