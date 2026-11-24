"""Does a back-transform / scale correction fix the rain-amount underestimate?

The amount model predicts log1p(mm) on wet hours. expm1(prediction) is closer to a
conditional MEDIAN than a mean, so P(rain) x expm1(pred) is biased low.
Corrections compared (each fitted on an inner held-out year, never on the scored year):
  raw     : P x expm1(pred)                               (current train_models.py)
  duan    : P x (exp(pred) * S - 1),  S = mean(exp(log1p residuals)) on wet hours
  ratio   : a x P x expm1(pred),       a = sum(y) / sum(raw expected) over all hours

Walk-forward inside 2011-2022 only. For scored year Y:
  inner : fit models on years < Y-1, estimate S and a on year Y-1
  outer : refit models on years < Y, apply S and a, score on year Y
Expected-value forecasts are mean forecasts, so the matching scores are bias and RMSE.
Wet-hour MAE rewards the median and is expected to get WORSE when bias is fixed.

Run from project root: python -m src.models.rain_amount_correction
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.config import PROCESSED_DIR, HORIZONS

FOLD_YEARS = [2019, 2020, 2021, 2022]
# Copied from train_models.py (importing it would run training).
NOT_FEATURES = ["y_temp", "y_amount", "y_rain", "base_yday",
                "eval_heavy", "eval_monsoon", "split", "now_key", "tgt_key"]
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
         random_state=42, verbose=-1)
RAIN_TUNED = {1: (31, 200, 0.05, 218), 6: (127, 200, 0.05, 95), 24: (15, 200, 0.05, 160)}


def year_start(y):
    return pd.Timestamp(f"{y}-01-01", tz="UTC")


def fit_predict(train, score, feats, h):
    """Train classifier + wet-hour log amount model on `train`; predict on `score`."""
    nl, mcs, lr, n = RAIN_TUNED[h]
    clf = lgb.LGBMClassifier(**{**P, "num_leaves": nl, "min_child_samples": mcs,
                                "learning_rate": lr, "n_estimators": n})
    clf.fit(train[feats], train["y_rain"])
    wet = train[train["y_rain"] == 1]
    reg = lgb.LGBMRegressor(**P).fit(wet[feats], np.log1p(wet["y_amount"]))
    return clf.predict_proba(score[feats])[:, 1], reg.predict(score[feats])


def before(d, y, h):
    """Rows whose target time is before the start of year y."""
    return d[d.index + pd.Timedelta(hours=h) < year_start(y)]


def in_year(d, y):
    return d[(d.index >= year_start(y)) & (d.index < year_start(y + 1))]


def scores(y, pred, wet, heavy, monsoon):
    e = pred - y
    out = {}
    for s, m in {"all": np.ones(len(y), bool), "monsoon": monsoon,
                 "non_monsoon": ~monsoon, "heavy_day": heavy}.items():
        out[f"{s}_bias"] = e[m].mean()
        out[f"{s}_rmse"] = np.sqrt((e[m] ** 2).mean())
    out["wet_mae"] = np.abs(e[wet]).mean()
    out["total_ratio"] = pred.sum() / y.sum()          # 1.0 = right total rainfall
    return out


def main():
    rows, factors = [], []
    for h in HORIZONS:
        d = pd.read_csv(PROCESSED_DIR / f"features_h{h}.csv", index_col=0)
        d.index = pd.to_datetime(d.index, utc=True)
        d = d[d["split"] == "train"]
        assert d.index.max() < year_start(2023), "non-train rows loaded"
        feats = [c for c in d.columns if c not in NOT_FEATURES]
        for Y in FOLD_YEARS:
            # inner: estimate corrections on year Y-1 from models trained before it
            inner = in_year(d, Y - 1)
            p_i, lp_i = fit_predict(before(d, Y - 1, h), inner, feats, h)
            w = inner["y_rain"].values == 1
            S = np.mean(np.exp(np.log1p(inner["y_amount"].values[w]) - lp_i[w]))
            a = inner["y_amount"].sum() / (p_i * np.expm1(lp_i)).sum()
            factors.append({"h": h, "fold": Y, "duan_S": S, "ratio_a": a})
            # outer: refit on everything before Y, score on Y
            outer = in_year(d, Y)
            p, lp = fit_predict(before(d, Y, h), outer, feats, h)
            preds = {"raw": p * np.expm1(lp),
                     "duan": p * np.maximum(np.exp(lp) * S - 1, 0),
                     "ratio": a * p * np.expm1(lp)}
            y = outer["y_amount"].values
            wet = outer["y_rain"].values == 1
            heavy = outer["eval_heavy"].values == 1
            mon = outer["eval_monsoon"].values == 1
            for name, pr in preds.items():
                rows.append({"h": h, "fold": Y, "method": name,
                             **scores(y, pr, wet, heavy, mon)})
            print(f"h={h} fold {Y} done (S={S:.3f}, a={a:.3f})", flush=True)

    res = pd.DataFrame(rows)
    out_dir = PROCESSED_DIR / "tuning"
    out_dir.mkdir(exist_ok=True)
    res.to_csv(out_dir / "rain_amount_correction_cv.csv", index=False)
    pd.DataFrame(factors).to_csv(out_dir / "rain_amount_factors.csv", index=False)

    cols = ["all_bias", "all_rmse", "monsoon_bias", "monsoon_rmse", "heavy_day_bias",
            "heavy_day_rmse", "wet_mae", "total_ratio"]
    print("\nFactors per fold:\n" + pd.DataFrame(factors).round(3).to_string(index=False))
    for h in HORIZONS:
        r = res[res["h"] == h]
        print(f"\n######## h={h}h  (mean over folds 2019-2022) ########")
        print(r.groupby("method")[cols].mean().round(3).to_string())
        # paired: does each correction lower all-hours RMSE in every fold?
        piv = r.pivot_table(index="fold", columns="method", values="all_rmse")
        for m in ["duan", "ratio"]:
            dlt = piv[m] - piv["raw"]
            print(f"  {m}: all-hours RMSE change per fold {dlt.round(4).tolist()}  "
                  f"(better in {(dlt < 0).sum()}/4)")


if __name__ == "__main__":
    main()
