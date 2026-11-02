"""Hourly weather for Karachi: global horizontal irradiance (W/m2) and air temperature (C).

Real data: run scripts/download_weather.py on your own computer. It saves
data/karachi_<year>.csv (columns: time, ghi, temp_c) from NASA POWER, and this
module loads those files automatically.

Synthetic fallback: if a year's CSV is missing, a realistic synthetic year is
generated (pvlib clear-sky sun x random cloudiness with a monsoon season, plus
monthly Karachi temperatures). Results for the report should use REAL data;
the synthetic generator is for development and for robustness tests.
"""
import os
import warnings

import numpy as np
import pandas as pd

# Approximate monthly mean air temperature in Karachi (C), Jan..Dec
KARACHI_MONTHLY_TEMP = [19, 21, 25, 29, 31, 32, 31, 30, 29, 28, 24, 20]
# Mean "cloud loss" per month (fraction of clear-sky irradiance lost), Jan..Dec.
# Higher in Jul-Sep (monsoon) and slightly in winter (haze).
KARACHI_MONTHLY_CLOUD = [0.12, 0.10, 0.08, 0.07, 0.08, 0.15, 0.35, 0.38, 0.25, 0.08, 0.09, 0.12]


def load_year(year, cfg):
    path = os.path.join(cfg.data_dir, f"karachi_{year}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
        df = df[["ghi", "temp_c"]].astype(float)
        df["ghi"] = df["ghi"].clip(lower=0).interpolate().fillna(0)
        df["temp_c"] = df["temp_c"].interpolate().bfill().ffill()
        df.attrs["source"] = "nasa_power"
        return df
    if not cfg.allow_synthetic_weather:
        raise FileNotFoundError(f"{path} not found. Run scripts/download_weather.py first.")
    warnings.warn(f"{path} not found: using SYNTHETIC weather for {year}.", stacklevel=2)
    return synthetic_year(year, cfg)


def synthetic_year(year, cfg):
    """Synthetic but plausible hourly weather. Seeded by year so each year differs."""
    import pvlib

    rng = np.random.default_rng(1000 + year)
    times = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h", tz=cfg.timezone)
    loc = pvlib.location.Location(cfg.latitude, cfg.longitude, tz=cfg.timezone)
    clear_ghi = loc.get_clearsky(times, model="ineichen")["ghi"].to_numpy()

    n_days = len(times) // 24
    months = times.month.to_numpy()[::24] - 1

    # Daily cloud loss: AR(1) around the monthly mean, so cloudy spells last several days
    cloud = np.zeros(n_days)
    x = 0.0
    for d in range(n_days):
        x = 0.6 * x + rng.normal(0, 0.12)
        cloud[d] = np.clip(KARACHI_MONTHLY_CLOUD[months[d]] + x, 0.0, 0.85)
    hourly_cloud = np.repeat(cloud, 24)
    hourly_cloud = np.clip(hourly_cloud + rng.normal(0, 0.05, len(times)), 0, 0.95)
    ghi = clear_ghi * (1 - hourly_cloud)

    # Temperature: monthly mean + daily anomaly (AR1) + diurnal cycle peaking ~15:00
    daily_anom = np.zeros(n_days)
    a = 0.0
    for d in range(n_days):
        a = 0.7 * a + rng.normal(0, 1.2)
        daily_anom[d] = a
    hours = times.hour.to_numpy()
    month_mean = np.array(KARACHI_MONTHLY_TEMP)[times.month.to_numpy() - 1]
    diurnal = 4.5 * np.cos(2 * np.pi * (hours - 15) / 24)
    temp = month_mean + np.repeat(daily_anom, 24) + diurnal - 1.5 * hourly_cloud

    df = pd.DataFrame({"ghi": ghi, "temp_c": temp}, index=times.tz_localize(None))
    df.index.name = "time"
    df.attrs["source"] = "synthetic"
    return df


def load_years(years, cfg):
    frames = [load_year(y, cfg) for y in years]
    df = pd.concat(frames)
    df = df[~df.index.duplicated()].sort_index()
    df.attrs["source"] = ",".join(sorted({f.attrs.get("source", "?") for f in frames}))
    return df
