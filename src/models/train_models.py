"""Train ridge + LightGBM, evaluate on validation only vs the best baseline.
Run from project root: python -m src.models.train_models
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, average_precision_score
from src.config import PROCESSED_DIR, HORIZONS, RAIN_THRESHOLD_MM

IST = "Asia/Kolkata"
NOT_FEATURES = ["y_temp", "y_amount", "y_rain", "base_yday",
                "eval_heavy", "eval_monsoon", "split", "now_key", "tgt_key"]
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
         random_state=42, verbose=-1)


def load(h):
    d = pd.read_csv(PROCESSED_DIR / f"features_h{h}.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    now = d.index.tz_convert(IST)
    tgt = now + pd.Timedelta(hours=h)
    d["now_key"] = np.asarray(now.month) * 100 + np.asarray(now.hour)
    d["tgt_key"] = np.asarray(tgt.month) * 100 + np.asarray(tgt.hour)
    return d


for h in HORIZONS:
    d = load(h)
    tr, va = d[d["split"] == "train"], d[d["split"] == "val"]
    feats = [c for c in d.columns if c not in NOT_FEATURES]
    monsoon = (va["eval_monsoon"] == 1).values
    slices = {"all": np.ones(len(va), bool), "monsoon": monsoon,
              "non_monsoon": ~monsoon, "heavy_day": (va["eval_heavy"] == 1).values}
    print(f"\n################ HORIZON {h}h  ({len(feats)} features) ################")

    # ---------------- temperature (models predict change from current temp) ----------------
    ytr = tr["y_temp"] - tr["temperature_2m"]
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(tr[feats], ytr)
    gbm = lgb.LGBMRegressor(**P).fit(tr[feats], ytr)
    clim = tr.groupby("tgt_key")["y_temp"].mean()
    c_tgt, c_now = va["tgt_key"].map(clim), va["now_key"].map(clim)
    base = {"persistence": va["temperature_2m"], "same_hour_yesterday": va["base_yday"],
            "climatology": c_tgt, "clim+anomaly": c_tgt + (va["temperature_2m"] - c_now)}
    best_name = min(base, key=lambda k: (base[k] - va["y_temp"]).abs().mean())
    preds = {**base,
             "ridge": va["temperature_2m"] + ridge.predict(va[feats]),
             "lightgbm": va["temperature_2m"] + gbm.predict(va[feats])}
    rows = []
    for s, m in slices.items():
        y = va["y_temp"][m]
        best_mae = (base[best_name][m] - y).abs().mean()
        for name in [best_name, "ridge", "lightgbm"]:
            e = preds[name][m] - y
            rows.append({"slice": s, "model": name, "MAE": e.abs().mean(),
                         "RMSE": np.sqrt((e ** 2).mean()),
                         "skill_vs_best_baseline": 1 - e.abs().mean() / best_mae})
    print(f"\n--- TEMPERATURE (best baseline: {best_name}) ---")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---------------- rain occurrence ----------------
    clf = lgb.LGBMClassifier(**P).fit(tr[feats], tr["y_rain"])
    prob = pd.Series(clf.predict_proba(va[feats])[:, 1], index=va.index)
    seas = va["tgt_key"].map(tr.groupby("tgt_key")["y_rain"].mean())
    rbase = {"seasonal_prob": seas,
             "rain_now": (va["precipitation"] >= RAIN_THRESHOLD_MM).astype(float)}
    rbest = min(rbase, key=lambda k: brier_score_loss(va["y_rain"], rbase[k]))
    rows = []
    for s, m in slices.items():
        if s == "heavy_day":
            continue                      # conditioned on the outcome: not meaningful here
        y = va["y_rain"][m]
        b = brier_score_loss(y, rbase[rbest][m])
        for name, p in [(rbest, rbase[rbest]), ("lightgbm", prob)]:
            br = brier_score_loss(y, p[m])
            rows.append({"slice": s, "model": name, "Brier": br,
                         "BSS_vs_best_baseline": 1 - br / b,
                         "PR_AUC": average_precision_score(y, p[m]), "wet_share": y.mean()})
    print(f"\n--- RAIN OCCURRENCE (best baseline: {rbest}) ---")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---------------- rain amount (trained on wet hours, log target) ----------------
    wet = tr[tr["y_rain"] == 1]
    reg = lgb.LGBMRegressor(**P).fit(wet[feats], np.log1p(wet["y_amount"]))
    amt_wet = pd.Series(np.expm1(reg.predict(va[feats])), index=va.index)  # amount IF it rains
    expected = prob * amt_wet                                              # P(rain) x amount
    clim_amt = va["tgt_key"].map(tr.groupby("tgt_key")["y_amount"].mean())
    amt = {"zero": pd.Series(0.0, index=va.index), "climatology_mean": clim_amt,
           "lgbm_expected (P x amt)": expected}
    rows = []
    for s, m in slices.items():
        y, wetm = va["y_amount"][m], (va["y_rain"][m] == 1)
        for name, p in amt.items():
            e = p[m] - y
            rows.append({"slice": s, "model": name, "MAE_wet_hours": e[wetm].abs().mean(),
                         "RMSE_all_hours": np.sqrt((e ** 2).mean()), "bias_mm_per_hr": e.mean()})
        e = amt_wet[m] - y
        rows.append({"slice": s, "model": "lgbm_amount_given_wet",
                     "MAE_wet_hours": e[wetm].abs().mean(),
                     "RMSE_all_hours": np.nan, "bias_mm_per_hr": np.nan})
    print("\n--- RAIN AMOUNT, mm ---")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    imp = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nTop rain-classifier features:", ", ".join(imp.index[:8]))

    out = pd.DataFrame({"y_temp": va["y_temp"], "temp_ridge": preds["ridge"],
                        "temp_lgbm": preds["lightgbm"], "y_rain": va["y_rain"],
                        "rain_prob": prob, "y_amount": va["y_amount"], "amt_expected": expected})
    out.to_csv(PROCESSED_DIR / f"val_preds_h{h}.csv")