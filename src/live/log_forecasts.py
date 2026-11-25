"""Log one live forecast per horizon, next to the baselines and two weather-service models.

Run once an hour (GitHub Actions schedule, or Windows Task Scheduler):  python -m src.live.log_forecasts
Appends to <DATA_DIR>/live/forecast_log.csv (one row per issue hour and horizon).
Score the log later with:  python -m src.live.score_live

How it works:
  - Inputs: the last ~3 days of hourly weather for the same 0.25 deg cell (25.5N, 85.25E) from
    Open-Meteo's ECMWF IFS feed. ERA5 itself arrives ~5 days late, so a live system can't use it.
    IFS is the same model family as ERA5, but it isn't identical (and its recent hours are
    short-range forecasts), so expect the live skill to be lower than the backtest.
  - Features: computed with the same make_features() used for training.
  - Models: the frozen final models saved by src.models.save_final_models.
  - Comparison: ECMWF IFS and GFS forecasts for the same valid times, as issued at that moment.
Only coordinates are sent to Open-Meteo. Free, non-commercial tier.
"""
import json
import sys
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
from src.config import DATA_DIR, HORIZONS, RAIN_THRESHOLD_MM
from src.features.build_features import make_features
from src.models.frozen_config import IST, BEST_BASELINE

LAT, LON = 25.594, 85.138                       # same request point as the ERA5 download
URL = "https://api.open-meteo.com/v1/forecast"
VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure",
        "wind_speed_10m", "wind_direction_10m", "cloud_cover", "precipitation"]
NWP = ["ecmwf_ifs025", "gfs_seamless"]
MODELS, LIVE = DATA_DIR / "models", DATA_DIR / "live"
LOG, STATUS = LIVE / "forecast_log.csv", LIVE / "logger.log"


def get(params):
    for attempt in range(4):                                     # retry rate limits / server errors
        r = requests.get(URL, params={"latitude": LAT, "longitude": LON, "timezone": "GMT", **params},
                         timeout=60)
        if r.status_code not in (429, 500, 502, 503, 504):
            break
        time.sleep(20 * (attempt + 1))
    r.raise_for_status()
    j = r.json()
    h = pd.DataFrame(j["hourly"])
    h["time"] = pd.to_datetime(h["time"]).dt.tz_localize("UTC")
    return j, h.set_index("time")


def key(ts):
    t = ts.tz_convert(IST)
    return t.month * 100 + t.hour


def main():
    LIVE.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp(datetime.now(timezone.utc))
    t = now.floor("h")                                            # issue time

    # inputs: past hours from the IFS feed, same grid cell as ERA5
    j, x = get({"hourly": ",".join(VARS), "models": "ecmwf_ifs025", "past_days": 3,
                "forecast_days": 1})
    assert (j["latitude"], j["longitude"]) == (25.5, 85.25), f"unexpected cell {j['latitude']}, {j['longitude']}"
    x = x[x.index <= t]
    assert x.index[-1] == t and len(x) >= 49, "not enough input history"
    F = make_features(x).loc[[t]]
    assert not F.isna().any().any(), "missing input values"

    # weather-service forecasts for the same valid times
    _, nwp = get({"hourly": "temperature_2m,precipitation", "models": ",".join(NWP),
                  "forecast_days": 3})

    meta = json.loads((MODELS / "meta.json").read_text())
    clim = pd.read_csv(MODELS / "climatology.csv", index_col=0)
    T, p = x["temperature_2m"], x["precipitation"]
    rows = []
    for h in HORIZONS:
        v = t + pd.Timedelta(hours=h)
        f = F[meta["features"][str(h)]]
        temp = T[t] + lgb.Booster(model_file=str(MODELS / f"temp_h{h}.txt")).predict(f)[0]
        prob = lgb.Booster(model_file=str(MODELS / f"rain_h{h}.txt")).predict(f)[0]
        amt_wet = max(np.exp(lgb.Booster(model_file=str(MODELS / f"amount_h{h}.txt")).predict(f)[0])
                      * meta["smear"][str(h)] - 1, 0)
        ck, nk = key(v), key(t)
        tb = {"persistence": T[t], "same_hour_yesterday": T.get(v - pd.Timedelta(hours=24), np.nan),
              "clim+anomaly": clim.at[ck, "temp"] + T[t] - clim.at[nk, "temp"]}
        rb = {"rain_now": float(p[t] >= RAIN_THRESHOLD_MM), "seasonal_prob": clim.at[ck, "rain"]}
        ab = {"rain_now": p[t], "climatology_mean": clim.at[ck, "amount"]}
        row = {"logged_at_utc": now.isoformat(), "issue_utc": t.isoformat(), "h": h,
               "valid_utc": v.isoformat(), "valid_ist": v.tz_convert(IST).strftime("%Y-%m-%d %H:%M"),
               "lgbm_temp": temp, "lgbm_rain_prob": prob, "lgbm_amount": prob * amt_wet,
               "base_temp": tb[BEST_BASELINE["temp"][h]], "base_rain": rb[BEST_BASELINE["rain"][h]],
               "base_amount": ab[BEST_BASELINE["amount"][h]], "persistence_temp": T[t]}
        for m in NWP:
            row[f"{m}_temp"] = nwp[f"temperature_2m_{m}"].get(v, np.nan)
            row[f"{m}_precip"] = nwp[f"precipitation_{m}"].get(v, np.nan)
        rows.append(row)

    new = pd.DataFrame(rows)
    if LOG.exists():
        old = pd.read_csv(LOG)
        new = new[~new.set_index(["issue_utc", "h"]).index.isin(old.set_index(["issue_utc", "h"]).index)]
    new.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    msg = f"{now:%Y-%m-%d %H:%M}Z issue {t:%Y-%m-%d %H}Z: logged {len(new)} rows"
    print(msg)
    return msg


if __name__ == "__main__":
    LIVE.mkdir(parents=True, exist_ok=True)
    ok = True
    try:
        line = main()
    except Exception:
        ok = False
        line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z FAILED\n{traceback.format_exc()}"
        print(line)
    with open(STATUS, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    sys.exit(0 if ok else 1)                                     # a failed run shows up as failed
