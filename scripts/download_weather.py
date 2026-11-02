"""Download real hourly weather for Karachi from NASA POWER (free, no API key).

    python scripts/download_weather.py --years 2020 2021 2022 2023 2024

Saves data/karachi_<year>.csv with columns
    time       local time (Asia/Karachi, PKT = UTC+5), hour-beginning: 13:00 = average over 13:00-14:00
    ghi        global horizontal irradiance (W/m2)        ALLSKY_SFC_SW_DWN
    dni        direct normal irradiance (W/m2)            ALLSKY_SFC_SW_DNI
    dhi        diffuse horizontal irradiance (W/m2)       ALLSKY_SFC_SW_DIFF
    temp_c     air temperature at 2 m (C)                 T2M
    wind_ms    wind speed at 2 m (m/s)                    WS2M
    precip_mm  precipitation (mm/h)                       PRECTOTCORR

Data are requested in UTC and converted to local time here. (The previous version asked
for NASA POWER "LST", which at Karachi's longitude is UTC+4, one hour behind PKT.)
"""
import argparse
import json
import os
import urllib.request

import pandas as pd

PARAMS = {"ALLSKY_SFC_SW_DWN": "ghi", "ALLSKY_SFC_SW_DNI": "dni", "ALLSKY_SFC_SW_DIFF": "dhi",
          "T2M": "temp_c", "WS2M": "wind_ms", "PRECTOTCORR": "precip_mm"}
URL = ("https://power.larc.nasa.gov/api/temporal/hourly/point"
       "?parameters={params}&community=RE&longitude={lon}&latitude={lat}"
       "&start={start}&end={end}&format=JSON&time-standard=UTC")


def download(year, lat, lon, out_dir, tz="Asia/Karachi"):
    # UTC request covering the whole local year (PKT is ahead of UTC)
    url = URL.format(params=",".join(PARAMS), lat=lat, lon=lon, start=f"{year - 1}1231", end=f"{year}1231")
    print(f"Downloading {year} ...")
    with urllib.request.urlopen(url, timeout=180) as r:
        data = json.load(r)
    p = data["properties"]["parameter"]
    df = pd.DataFrame({col: pd.Series(p[key], dtype=float) for key, col in PARAMS.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H").tz_localize("UTC").tz_convert(tz).tz_localize(None)
    df = df.replace(-999.0, float("nan")).interpolate().bfill().ffill()
    for c in ("ghi", "dni", "dhi", "precip_mm"):
        df[c] = df[c].clip(lower=0)
    df = df.loc[f"{year}-01-01 00:00":f"{year}-12-31 23:00"]
    df.index.name = "time"
    path = os.path.join(out_dir, f"karachi_{year}.csv")
    df.to_csv(path, float_format="%.3f")
    print(f"  saved {path} ({len(df)} hours, mean GHI {df['ghi'].mean():.0f} W/m2, "
          f"mean temp {df['temp_c'].mean():.1f} C, rain {df['precip_mm'].sum():.0f} mm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024])
    ap.add_argument("--lat", type=float, default=None, help="default: site.latitude from the config")
    ap.add_argument("--lon", type=float, default=None, help="default: site.longitude from the config")
    ap.add_argument("--config", default="default")
    ap.add_argument("--out", default=None, help="default: data.dir from the config")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from solar_rl.config import load_config
    cfg = load_config(args.config)
    lat = args.lat if args.lat is not None else cfg.site.latitude
    lon = args.lon if args.lon is not None else cfg.site.longitude
    out = args.out or cfg.data.dir
    os.makedirs(out, exist_ok=True)
    for y in args.years:
        download(y, lat, lon, out, cfg.site.timezone)


if __name__ == "__main__":
    main()
