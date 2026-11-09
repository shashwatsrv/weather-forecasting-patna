"""ERA5 data sanity check (Open-Meteo archive API) for Patna.

Pulls a small window of hourly data, saves the raw response untouched,
and prints a report: variables, units, timezone, gaps, nulls, precipitation.
Run from project root: python -m src.data.era5_data_check
"""
import json
import requests
import pandas as pd
from src.config import RAW_DIR

LAT, LON = 25.594, 85.138          # Patna (approx.)
START, END = "2023-06-01", "2023-08-31"   # includes monsoon; adjust freely
VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "surface_pressure", "pressure_msl", "wind_speed_10m",
        "wind_direction_10m", "cloud_cover", "precipitation", "rain"]

URL = "https://archive-api.open-meteo.com/v1/archive"
params = {"latitude": LAT, "longitude": LON, "start_date": START,
          "end_date": END, "hourly": ",".join(VARS), "timezone": "GMT"}

resp = requests.get(URL, params=params, timeout=60)
resp.raise_for_status()
raw = resp.json()

with open(RAW_DIR / "raw_patna_sample.json", "w") as f:   # raw pull, as downloaded
    json.dump(raw, f)

df = pd.DataFrame(raw["hourly"])
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")

print("== Response metadata ==")
for k in ["latitude", "longitude", "elevation", "timezone",
          "utc_offset_seconds", "generationtime_ms"]:
    print(f"{k}: {raw.get(k)}")

print("\n== Units ==")
for k, v in raw["hourly_units"].items():
    print(f"{k}: {v}")

print("\n== Time coverage ==")
expected = pd.date_range(df.index.min(), df.index.max(), freq="h")
missing = expected.difference(df.index)
print(f"rows: {len(df)}, expected: {len(expected)}, missing timestamps: {len(missing)}")
print(f"duplicates: {df.index.duplicated().sum()}")

print("\n== Nulls per column ==")
print(df.isna().sum().to_string())

print("\n== Summary stats ==")
print(df.describe().T[["min", "mean", "max"]].to_string())

print("\n== Precipitation ==")
p = df["precipitation"]
print(f"dry hours (==0): {(p == 0).mean():.1%}")
print(f"max hourly: {p.max()}  | hours > 10: {(p > 10).sum()}")
print(f"precipitation vs rain identical: {(df['precipitation'] == df['rain']).all()}")
print("Check docs: is precipitation an accumulation over the PRECEDING hour?")

print("\nAlso check: license/attribution terms, and whether the variables above "
      "match ERA5 vs a blended/downscaled source for your location.")
