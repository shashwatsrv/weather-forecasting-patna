"""Re-fit the FROZEN final models (train+val, 2011-2024) and save them for live use.

Same data, parameters and seed as `evaluate_test --final`, so the saved models must
reproduce the 2025 predictions in test_eval/preds_h{h}.csv; this is checked before saving.
Also saves the climatology tables the baselines need.

Run from project root: python -m src.models.save_final_models
Output: <DATA_DIR>/models/
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.config import DATA_DIR, PROCESSED_DIR, HORIZONS
from src.models.frozen_config import IST, NOT_FEATURES, P, SMEAR, params

OUT = DATA_DIR / "models"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"smear": SMEAR, "features": {}}
    for h in HORIZONS:
        d = pd.read_csv(PROCESSED_DIR / f"features_h{h}.csv", index_col=0)
        d.index = pd.to_datetime(d.index, utc=True)
        tgt = d.index.tz_convert(IST) + pd.Timedelta(hours=h)
        d["tgt_key"] = np.asarray(tgt.month) * 100 + np.asarray(tgt.hour)
        fit, test = d[d["split"].isin(["train", "val"])], d[d["split"] == "test"]
        feats = [c for c in d.columns if c not in NOT_FEATURES]
        meta["features"][h] = feats

        temp = lgb.LGBMRegressor(**params("temp", h)).fit(fit[feats], fit["y_temp"] - fit["temperature_2m"])
        rain = lgb.LGBMClassifier(**params("rain", h)).fit(fit[feats], fit["y_rain"])
        wet = fit[fit["y_rain"] == 1]
        amt = lgb.LGBMRegressor(**P).fit(wet[feats], np.log1p(wet["y_amount"]))

        # must reproduce the frozen test predictions
        ref = pd.read_csv(PROCESSED_DIR / "test_eval" / f"preds_h{h}.csv", index_col=0)
        prob = rain.predict_proba(test[feats])[:, 1]
        chk = {"temp_lgbm": test["temperature_2m"] + temp.predict(test[feats]),
               "rain_prob": prob,
               "amt_expected": prob * np.maximum(np.exp(amt.predict(test[feats])) * SMEAR[h] - 1, 0)}
        for k, v in chk.items():
            diff = np.abs(np.asarray(v) - ref[k].values).max()
            print(f"{h}h {k}: max abs diff vs frozen test preds = {diff:.2e}")
            assert diff < 1e-6, f"{h}h {k} does not reproduce the frozen predictions"

        for name, m in [("temp", temp), ("rain", rain), ("amount", amt)]:
            m.booster_.save_model(str(OUT / f"{name}_h{h}.txt"))

    # climatology tables (fit on 2011-2024) keyed by IST month*100+hour, for the baselines
    d = pd.read_csv(PROCESSED_DIR / "features_h1.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    d = d[d["split"].isin(["train", "val"])]
    tgt = d.index.tz_convert(IST) + pd.Timedelta(hours=1)
    key = np.asarray(tgt.month) * 100 + np.asarray(tgt.hour)
    clim = pd.DataFrame({"tgt_key": key, "temp": d["y_temp"].values, "rain": d["y_rain"].values,
                         "amount": d["y_amount"].values}).groupby("tgt_key").mean()
    clim.to_csv(OUT / "climatology.csv")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved models and climatology to {OUT}")


if __name__ == "__main__":
    main()
