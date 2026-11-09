"""Is the rain skill weather signal, or just finer seasonality?
Compares four rain-occurrence models on validation only:
  month_hour_baseline : lookup of rain frequency by (month, hour)
  doy_only            : LightGBM using only day-of-year + hour (smooth seasonal baseline)
  full_minus_doy      : all features except day-of-year
  full                : all features
Run from project root: python -m src.evaluation.doy_ablation
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, average_precision_score
from src.config import PROCESSED_DIR, HORIZONS

IST = "Asia/Kolkata"
NOT_FEATURES = ["y_temp", "y_amount", "y_rain", "base_yday",
                "eval_heavy", "eval_monsoon", "split", "now_key", "tgt_key"]
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
         random_state=42, verbose=-1)

for h in HORIZONS:
    d = pd.read_csv(PROCESSED_DIR / f"features_h{h}.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    tgt = d.index.tz_convert(IST) + pd.Timedelta(hours=h)
    d["tgt_key"] = np.asarray(tgt.month) * 100 + np.asarray(tgt.hour)

    tr, va = d[d["split"] == "train"], d[d["split"] == "val"]
    feats = [c for c in d.columns if c not in NOT_FEATURES]
    sets = {"doy_only": ["doy_sin", "doy_cos", "hour_sin", "hour_cos"],
            "full_minus_doy": [c for c in feats if not c.startswith("doy_")],
            "full": feats}

    probs = {"month_hour_baseline": va["tgt_key"].map(tr.groupby("tgt_key")["y_rain"].mean())}
    for name, cols in sets.items():
        m = lgb.LGBMClassifier(**P).fit(tr[cols], tr["y_rain"])
        probs[name] = pd.Series(m.predict_proba(va[cols])[:, 1], index=va.index)

    monsoon = (va["eval_monsoon"] == 1).values
    slices = {"all": np.ones(len(va), bool), "monsoon": monsoon, "non_monsoon": ~monsoon}
    rows = []
    for s, mask in slices.items():
        y = va["y_rain"][mask]
        b_mh = brier_score_loss(y, probs["month_hour_baseline"][mask])
        b_doy = brier_score_loss(y, probs["doy_only"][mask])
        for name, p in probs.items():
            br = brier_score_loss(y, p[mask])
            rows.append({"slice": s, "model": name, "Brier": br,
                         "BSS_vs_month_hour": 1 - br / b_mh,
                         "BSS_vs_doy_only": 1 - br / b_doy,
                         "PR_AUC": average_precision_score(y, p[mask])})
    print(f"\n===== {h}h RAIN OCCURRENCE ABLATION (validation) =====")
    print(pd.DataFrame(rows).round(3).to_string(index=False))