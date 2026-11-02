"""Download real hourly weather for Karachi from NASA POWER (free, no API key).

Run this on your own computer (needs internet):
    python scripts/download_weather.py --years 2020 2021 2022 2023 2024

Saves data/karachi_<year>.csv with columns: time, ghi (W/m2), temp_c (C), in local time.
Once these files exist, the simulator uses them instead of synthetic weather.
"""
import argparse
import json
import os
import urllib.request

import pandas as pd

URL = ("https://power.larc.nasa.gov/api/temporal/hourly/point"
       "?parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE&longitude={lon}&latitude={lat}"
       "&start={y}0101&end={y}1231&format=JSON&time-standard=LST")


def download(year, lat, lon, out_dir):
    url = URL.format(lat=lat, lon=lon, y=year)
    print(f"Downloading {year} ...")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = json.load(r)
    p = data["properties"]["parameter"]
    ghi = pd.Series(p["ALLSKY_SFC_SW_DWN"], dtype=float)
    temp = pd.Series(p["T2M"], dtype=float)
    df = pd.DataFrame({"ghi": ghi, "temp_c": temp})
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df = df.replace(-999.0, float("nan")).interpolate().bfill().ffill()
    df["ghi"] = df["ghi"].clip(lower=0)
    df.index.name = "time"
    path = os.path.join(out_dir, f"karachi_{year}.csv")
    df.to_csv(path)
    print(f"  saved {path} ({len(df)} hours, mean GHI {df['ghi'].mean():.0f} W/m2, "
          f"mean temp {df['temp_c'].mean():.1f} C)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024])
    ap.add_argument("--lat", type=float, default=24.86)
    ap.add_argument("--lon", type=float, default=67.01)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for y in args.years:
        download(y, args.lat, args.lon, args.out)


if __name__ == "__main__":
    main()
