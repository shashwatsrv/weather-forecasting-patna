"""Baselines evaluated on the validation split only.
Run from project root: python -m src.evaluation.baselines
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from src.config import PROCESSED_DIR, HORIZONS, RAIN_THRESHOLD_MM

IST = "Asia/Kolkata"


def load(h):
    d = pd.read_csv(PROCESSED_DIR / f"features_h{h}.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    now, tgt = d.index.tz_convert(IST), d.index.tz_convert(IST) + pd.Timedelta(hours=h)
    d["now_key"] = np.asarray(now.month) * 100 + np.asarray(now.hour)
    d["tgt_key"] = np.asarray(tgt.month) * 100 + np.asarray(tgt.hour)
    return d


for h in HORIZONS:
    d = load(h)
    tr, va = d[d["split"] == "train"], d[d["split"] == "val"]
    slices = {"all": va.index == va.index,
              "monsoon": (va["eval_monsoon"] == 1).values,
              "heavy_day": (va["eval_heavy"] == 1).values}

    # ---- temperature ----
    clim = tr.groupby("tgt_key")["y_temp"].mean()          # mean by (month, hour) from train
    clim_tgt, clim_now = va["tgt_key"].map(clim), va["now_key"].map(clim)
    temp_preds = {
        "persistence": va["temperature_2m"],
        "same_hour_yesterday": va["base_yday"],
        "climatology": clim_tgt,
        "clim + current anomaly": clim_tgt + (va["temperature_2m"] - clim_now),
    }
    rows = []
    for sname, m in slices.items():
        y = va.loc[m, "y_temp"]
        base = (temp_preds["persistence"][m] - y).abs().mean()
        for name, p in temp_preds.items():
            e = p[m] - y
            rows.append({"slice": sname, "baseline": name, "MAE": e.abs().mean(),
                         "RMSE": np.sqrt((e ** 2).mean()),
                         "skill_vs_persist": 1 - e.abs().mean() / base})
    print(f"\n===== {h}h TEMPERATURE (validation) =====")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---- rain occurrence ----
    seasonal_p = tr.groupby("tgt_key")["y_rain"].mean()
    rain_preds = {
        "never_rain": pd.Series(0.0, index=va.index),
        "rain_now_persistence": (va["precipitation"] >= RAIN_THRESHOLD_MM).astype(float),
        "seasonal_prob": va["tgt_key"].map(seasonal_p),
    }
    rows = []
    for sname, m in slices.items():
        y = va.loc[m, "y_rain"]
        for name, s in rain_preds.items():
            pr = s[m]
            rows.append({"slice": sname, "baseline": name, "wet_share": y.mean(),
                         "Brier": brier_score_loss(y, pr),
                         "PR_AUC": average_precision_score(y, pr),
                         "F1@0.5": f1_score(y, pr >= 0.5, zero_division=0)})
    print(f"\n===== {h}h RAIN OCCURRENCE (validation) =====")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---- rain amount ----
    amt_clim = tr.groupby("tgt_key")["y_amount"].mean()
    amt_preds = {
        "zero": pd.Series(0.0, index=va.index),
        "rain_now_persistence": va["precipitation"],
        "climatology_mean": va["tgt_key"].map(amt_clim),
    }
    rows = []
    for sname, m in slices.items():
        y, wet = va.loc[m, "y_amount"], (va.loc[m, "y_rain"] == 1)
        for name, s in amt_preds.items():
            e = (s[m] - y).abs()
            rows.append({"slice": sname, "baseline": name,
                         "MAE_all_hours": e.mean(), "MAE_wet_hours": e[wet].mean()})
    print(f"\n===== {h}h RAIN AMOUNT, mm (validation) =====")
    print(pd.DataFrame(rows).round(3).to_string(index=False))