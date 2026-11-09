"""Download hourly ERA5 data for Patna in yearly chunks (Open-Meteo archive API).

Raw yearly JSON goes to RAW_DIR untouched; a combined CSV goes to INTERIM_DIR.
Safe to re-run: years already downloaded are skipped.
Run from project root: python -m src.data.download_era5
"""
import json
import time
import requests
import pandas as pd
from src.config import RAW_DIR, INTERIM_DIR

LAT, LON = 25.594, 85.138
YEARS = range(2011, 2026)          # 2011-2025 inclusive
VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "surface_pressure", "pressure_msl", "wind_speed_10m",
        "wind_direction_10m", "cloud_cover", "precipitation"]
URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_year(year):
    params = {"latitude": LAT, "longitude": LON,
              "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
              "hourly": ",".join(VARS), "timezone": "GMT", "models": "era5"}
    for attempt in range(5):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 429:               # rate limited: wait and retry
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Gave up on {year} after repeated rate limiting")


frames = []
for year in YEARS:
    path = RAW_DIR / f"era5_patna_{year}.json"
    if path.exists():
        raw = json.loads(path.read_text())
        print(f"{year}: already downloaded")
    else:
        raw = fetch_year(year)
        path.write_text(json.dumps(raw))
        print(f"{year}: downloaded")
        time.sleep(3)

    df = pd.DataFrame(raw["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    expected = len(pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h"))
    nulls = int(df[VARS].isna().sum().sum())
    flag = "" if (len(df) == expected and nulls == 0) else "  <-- CHECK"
    print(f"   rows {len(df)}/{expected}, nulls {nulls}{flag}")
    frames.append(df)

full = pd.concat(frames).drop_duplicates("time").sort_values("time")
full.to_csv(INTERIM_DIR / "era5_patna_2011_2025.csv", index=False)
print(f"\nCombined: {len(full)} rows, {full['time'].min()} to {full['time'].max()}")
print(f"Grid point used: {raw['latitude']}, {raw['longitude']} (model request: era5)")