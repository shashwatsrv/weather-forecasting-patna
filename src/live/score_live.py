"""Score the live forecast log against ERA5 once ERA5 has caught up (~5 days behind).

Run from project root: python -m src.live.score_live
Writes <DATA_DIR>/live/live_scores.csv and prints a table per horizon.

Fairness notes, report them with any result:
  - Truth is ERA5, the same as in training. ECMWF IFS comes from the same model family as
    ERA5, which favours IFS in this comparison.
  - Weather-service rain is a single amount, not a probability, so its Brier score uses
    1/0 for "forecast at least 0.1 mm". That's standard, but it penalises confident misses.
  - Only forecasts whose valid hour ERA5 already covers are scored; a few months are needed
    before differences mean much, and monsoon results need a monsoon season.
"""
import numpy as np
import pandas as pd
import requests
from src.config import DATA_DIR, RAIN_THRESHOLD_MM

LIVE = DATA_DIR / "live"
LAT, LON = 25.594, 85.138
NWP = ["ecmwf_ifs025", "gfs_seamless"]


def era5(start, end):
    r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                     params={"latitude": LAT, "longitude": LON, "start_date": start, "end_date": end,
                             "hourly": "temperature_2m,precipitation", "timezone": "GMT",
                             "models": "era5"}, timeout=120)
    r.raise_for_status()
    t = pd.DataFrame(r.json()["hourly"])
    t["valid_utc"] = pd.to_datetime(t["time"]).dt.tz_localize("UTC")
    return t.drop(columns="time").rename(columns={"temperature_2m": "y_temp",
                                                  "precipitation": "y_amount"}).dropna()


def main():
    log = pd.read_csv(LIVE / "forecast_log.csv")
    log["valid_utc"] = pd.to_datetime(log["valid_utc"], utc=True)
    start = log["valid_utc"].min().normalize()
    end = min(log["valid_utc"].max(), pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).normalize()
    truth = era5(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}") if end >= start else pd.DataFrame(
        columns=["valid_utc", "y_temp", "y_amount"])
    d = log.merge(truth, on="valid_utc", how="inner")
    if d.empty:
        print(f"Nothing to score yet: {len(log)} logged rows, ERA5 doesn't cover them yet.")
        return
    d["y_rain"] = (d["y_amount"] >= RAIN_THRESHOLD_MM).astype(float)

    rows = []
    for h, g in d.groupby("h"):
        temp = {"LightGBM": g["lgbm_temp"], "best baseline": g["base_temp"]}
        rain = {"LightGBM": g["lgbm_rain_prob"], "best baseline": g["base_rain"]}
        amt = {"LightGBM": g["lgbm_amount"], "best baseline": g["base_amount"]}
        for m in NWP:
            temp[m], amt[m] = g[f"{m}_temp"], g[f"{m}_precip"]
            rain[m] = (g[f"{m}_precip"] >= RAIN_THRESHOLD_MM).astype(float)
        for name in temp:
            rows.append({"h": h, "forecast": name, "n_hours": len(g),
                         "n_days": g["valid_utc"].dt.date.nunique(),
                         "temp_MAE": (temp[name] - g["y_temp"]).abs().mean(),
                         "rain_Brier": ((rain[name] - g["y_rain"]) ** 2).mean(),
                         "amount_RMSE": np.sqrt(((amt[name] - g["y_amount"]) ** 2).mean()),
                         "wet_share": g["y_rain"].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(LIVE / "live_scores.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"Scored {len(d)} of {len(log)} logged rows "
          f"({d['valid_utc'].min():%Y-%m-%d} to {d['valid_utc'].max():%Y-%m-%d}).\n")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
