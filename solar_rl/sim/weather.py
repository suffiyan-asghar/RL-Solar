"""Hourly weather: irradiance (GHI, DNI, DHI in W/m2), air temperature (C), wind speed (m/s)
and precipitation (mm/h). Index = local time (site timezone), hour-beginning labels: the row
labelled 13:00 is the average over 13:00-14:00.

Real data: scripts/download_weather.py saves data/karachi_<year>.csv from NASA POWER.
Synthetic fallback: if a year's CSV is missing, a plausible synthetic year is generated
(pvlib clear-sky sun x random cloudiness with a monsoon season, monthly temperatures,
random monsoon rain). Use real data for results; synthetic is for development and tests.

Files from the original downloader (columns time, ghi, temp_c only) still load, but they
were labelled in NASA POWER "LST" (UTC+4 at Karachi), one hour behind Pakistan time.
Re-download them for the pvlib solar model.
"""
import os
import warnings

import numpy as np
import pandas as pd

COLUMNS = ["ghi", "dni", "dhi", "temp_c", "wind_ms", "precip_mm"]
DEFAULT_WIND_MS = 1.0

# Approximate monthly mean air temperature in Karachi (C), Jan..Dec (synthetic weather only)
KARACHI_MONTHLY_TEMP = [19, 21, 25, 29, 31, 32, 31, 30, 29, 28, 24, 20]
# Mean fraction of clear-sky irradiance lost to clouds, Jan..Dec (synthetic weather only)
KARACHI_MONTHLY_CLOUD = [0.12, 0.10, 0.08, 0.07, 0.08, 0.15, 0.35, 0.38, 0.25, 0.08, 0.09, 0.12]
# Chance of a rainy day, Jan..Dec (synthetic weather only)
KARACHI_MONTHLY_RAIN_DAY = [0.02, 0.02, 0.01, 0.0, 0.0, 0.02, 0.12, 0.15, 0.05, 0.0, 0.0, 0.02]


def load_year(year, cfg):
    path = os.path.join(cfg.data.dir, f"karachi_{year}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
        legacy = "dni" not in df.columns
        df = df.reindex(columns=[c for c in COLUMNS if c in df.columns]).astype(float)
        for c in ("ghi", "dni", "dhi", "precip_mm"):
            if c in df:
                df[c] = df[c].clip(lower=0).interpolate().fillna(0)
        for c in ("temp_c", "wind_ms"):
            if c in df:
                df[c] = df[c].interpolate().bfill().ffill()
        if "wind_ms" not in df:
            df["wind_ms"] = DEFAULT_WIND_MS
        if "precip_mm" not in df:
            df["precip_mm"] = 0.0
        df.attrs["source"] = "nasa_power_lst_v1" if legacy else "nasa_power"
        return df
    if not cfg.data.allow_synthetic:
        raise FileNotFoundError(f"{path} not found. Run scripts/download_weather.py first.")
    warnings.warn(f"{path} not found: using SYNTHETIC weather for {year}.", stacklevel=2)
    return synthetic_year(year, cfg)


def synthetic_year(year, cfg):
    """Synthetic but plausible hourly weather. Seeded by year so each year differs."""
    import pvlib

    rng = np.random.default_rng(1000 + year)
    tz = cfg.site.timezone
    times = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h", tz=tz)
    loc = pvlib.location.Location(cfg.site.latitude, cfg.site.longitude, tz=tz)
    mid = times + pd.Timedelta(minutes=30)
    cs = loc.get_clearsky(mid, model="ineichen")
    n_days = len(times) // 24
    months = times.month.to_numpy()[::24] - 1

    # Daily cloud loss: AR(1) around the monthly mean, so cloudy spells last several days
    cloud = np.zeros(n_days)
    x = 0.0
    for d in range(n_days):
        x = 0.6 * x + rng.normal(0, 0.12)
        cloud[d] = np.clip(KARACHI_MONTHLY_CLOUD[months[d]] + x, 0.0, 0.85)
    hourly_cloud = np.clip(np.repeat(cloud, 24) + rng.normal(0, 0.05, len(times)), 0, 0.95)
    ghi = cs["ghi"].to_numpy() * (1 - hourly_cloud)
    dni = cs["dni"].to_numpy() * (1 - hourly_cloud) ** 1.5
    zen = loc.get_solarposition(mid)["zenith"].to_numpy()
    dhi = np.clip(ghi - dni * np.clip(np.cos(np.radians(zen)), 0, None), 0, None)

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

    # Sea breeze: stronger in the afternoon
    wind = np.clip(4.0 + 1.5 * np.cos(2 * np.pi * (hours - 16) / 24) + rng.normal(0, 0.8, len(times)), 0.2, None)
    # Rain: a few monsoon days, spread over 4 afternoon hours
    rain = np.zeros(len(times))
    for d in range(n_days):
        if rng.random() < KARACHI_MONTHLY_RAIN_DAY[months[d]]:
            rain[d * 24 + 14: d * 24 + 18] = rng.exponential(10.0) / 4
    df = pd.DataFrame({"ghi": ghi, "dni": dni, "dhi": dhi, "temp_c": temp, "wind_ms": wind, "precip_mm": rain},
                      index=times.tz_localize(None))
    df.index.name = "time"
    df.attrs["source"] = "synthetic"
    return df


def load_years(years, cfg):
    frames = [load_year(y, cfg) for y in years]
    df = pd.concat(frames)
    df = df[~df.index.duplicated()].sort_index()
    df.attrs["source"] = ",".join(sorted({f.attrs.get("source", "?") for f in frames}))
    return df


def as_weather(df):
    """Accept any DataFrame with at least ghi and temp_c; fill optional columns."""
    df = df.copy()
    if "wind_ms" not in df:
        df["wind_ms"] = DEFAULT_WIND_MS
    if "precip_mm" not in df:
        df["precip_mm"] = 0.0
    df.attrs["source"] = df.attrs.get("source", "custom")
    return df
