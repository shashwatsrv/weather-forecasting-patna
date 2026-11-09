"""Build lagged features, shifted targets and chronological splits.

Run from project root: python -m src.features.build_features
Output: PROCESSED_DIR/features_h{1,6,24}.csv
"""
import numpy as np
import pandas as pd
from src.config import (INTERIM_DIR, PROCESSED_DIR, RAIN_THRESHOLD_MM,
                        HEAVY_DAY_MM, HORIZONS, TRAIN_END, VAL_END)

IST = "Asia/Kolkata"

df = pd.read_csv(INTERIM_DIR / "era5_patna_2011_2025.csv", parse_dates=["time"])
df["time"] = df["time"].dt.tz_localize("UTC")
df = df.set_index("time").sort_index()
assert (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all(), "gaps in hourly index"

T, rh, p = df["temperature_2m"], df["relative_humidity_2m"], df["precipitation"]
ist = df.index.tz_convert(IST)

# ---------- features (use only data available at time t) ----------
F = pd.DataFrame(index=df.index)
for c in ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure",
          "wind_speed_10m", "cloud_cover", "precipitation"]:
    F[c] = df[c]
F["dewpoint_depression"] = T - df["dew_point_2m"]
wd = np.deg2rad(df["wind_direction_10m"])
F["wind_dir_sin"], F["wind_dir_cos"] = np.sin(wd), np.cos(wd)

for k in [1, 3, 6, 12, 24, 48]:
    F[f"temp_lag{k}"] = T.shift(k)
    F[f"rh_lag{k}"] = rh.shift(k)
for k in [3, 6, 24]:
    F[f"pres_chg{k}"] = df["surface_pressure"] - df["surface_pressure"].shift(k)
for k in [1, 2, 3, 6, 12, 24]:
    F[f"precip_lag{k}"] = p.shift(k)
for w in [3, 6, 24]:
    F[f"precip_sum{w}"] = p.rolling(w).sum()      # window ends at t: past data only

F["hour_sin"] = np.sin(2 * np.pi * np.asarray(ist.hour) / 24)
F["hour_cos"] = np.cos(2 * np.pi * np.asarray(ist.hour) / 24)
F["doy_sin"] = np.sin(2 * np.pi * np.asarray(ist.dayofyear) / 365.25)
F["doy_cos"] = np.cos(2 * np.pi * np.asarray(ist.dayofyear) / 365.25)

# ---------- heavy-rain days (IST days, threshold fixed from training years) ----------
daily = p.groupby(ist.date).sum()
heavy_hour = pd.Series(ist.date, index=df.index).map(daily >= HEAVY_DAY_MM).astype(float)

# ---------- targets, baselines helpers, splits ----------
train_end = pd.Timestamp(TRAIN_END, tz="UTC") + pd.Timedelta(days=1)   # exclusive
val_end = pd.Timestamp(VAL_END, tz="UTC") + pd.Timedelta(days=1)       # exclusive

for h in HORIZONS:
    out = F.copy()
    fut = p.shift(-h)
    out["y_temp"] = T.shift(-h)
    out["y_amount"] = fut
    out["y_rain"] = (fut >= RAIN_THRESHOLD_MM).astype(float).where(fut.notna())
    out["base_yday"] = T.shift(24 - h)               # same hour as target, previous day
    out["eval_heavy"] = heavy_hour.shift(-h)         # is the target's IST day a heavy-rain day?
    tgt = out.index + pd.Timedelta(hours=h)
    out["eval_monsoon"] = np.isin(np.asarray(tgt.tz_convert(IST).month), [6, 7, 8, 9]).astype(float)

    # split by issue time; drop rows whose target reaches into the next split
    out["split"] = np.select(
        [tgt < train_end, (out.index >= train_end) & (tgt < val_end), out.index >= val_end],
        ["train", "val", "test"], default="gap")

    # leakage spot checks
    i = out.index[2000]
    assert out.loc[i, "y_temp"] == T.loc[i + pd.Timedelta(hours=h)]
    assert out.loc[i, "precip_lag1"] == p.loc[i - pd.Timedelta(hours=1)]

    out = out.dropna()
    out = out[out["split"] != "gap"]
    out.to_csv(PROCESSED_DIR / f"features_h{h}.csv")

    print(f"\n=== horizon {h}h: {out.shape[1]} columns ===")
    print(out.groupby("split").agg(rows=("y_temp", "size"),
                                   wet_share=("y_rain", "mean"),
                                   heavy_share=("eval_heavy", "mean")).round(3))