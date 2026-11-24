"""Final evaluation with the FROZEN configuration (src/models/frozen_config.py).

Two modes:
  python -m src.evaluation.evaluate_test            CHECK: train on 2011-2022, score 2023-2024.
                                                    Test rows are dropped right after loading.
  python -m src.evaluation.evaluate_test --final    FINAL: train on 2011-2024, score 2025 ONCE.
                                                    Refuses to run if a previous final run exists.

Headline skill is against the best baseline per horizon frozen in BEST_BASELINE, with 95%
day-block bootstrap intervals (resampling whole IST target days, 2000 draws).
Outputs go to <PROCESSED_DIR>/test_eval/ (final) or <PROCESSED_DIR>/val_check/ (check).
"""
import sys
import json
import subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, average_precision_score
from src.config import PROCESSED_DIR, HORIZONS, RAIN_THRESHOLD_MM
from src.models.frozen_config import IST, NOT_FEATURES, P, SMEAR, BEST_BASELINE, params

FINAL = "--final" in sys.argv
OUT = PROCESSED_DIR / ("test_eval" if FINAL else "val_check")
MARKER = PROCESSED_DIR / "test_eval" / "FINAL_RUN_DONE.json"
N_BOOT, SEED = 2000, 0


def load(h):
    d = pd.read_csv(PROCESSED_DIR / f"features_h{h}.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    if FINAL:
        fit, score = d[d["split"].isin(["train", "val"])], d[d["split"] == "test"]
    else:
        d = d[d["split"] != "test"]                       # never keep 2025 in check mode
        assert d.index.max() < pd.Timestamp("2025-01-01", tz="UTC")
        fit, score = d[d["split"] == "train"], d[d["split"] == "val"]
    assert fit.index.max() < score.index.min()
    out = []
    for x in (fit, score):
        x = x.copy()
        now, tgt = x.index.tz_convert(IST), x.index.tz_convert(IST) + pd.Timedelta(hours=h)
        x["now_key"] = np.asarray(now.month) * 100 + np.asarray(now.hour)
        x["tgt_key"] = np.asarray(tgt.month) * 100 + np.asarray(tgt.hour)
        # rain stamped t covers (t-1h, t]: the target hour belongs to the IST day of tgt - 1h
        x["day"] = (tgt - pd.Timedelta(hours=1)).date
        out.append(x)
    return out


def boot_skill(day, num, den, rng, sqrt=False):
    """95% CI of 1 - sum(num)/sum(den) (or 1 - sqrt(...)) resampling whole days."""
    g = pd.DataFrame({"day": day, "n": num, "d": den}).groupby("day")[["n", "d"]].sum()
    w = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=N_BOOT)
    r = (w @ g["n"].values) / (w @ g["d"].values)
    s = 1 - (np.sqrt(r) if sqrt else r)
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def slices(sc, heavy=True):
    m = (sc["eval_monsoon"] == 1).values
    s = {"all": np.ones(len(sc), bool), "monsoon": m, "non_monsoon": ~m}
    if heavy:
        s["heavy_day"] = (sc["eval_heavy"] == 1).values
    return s


def main():
    if FINAL and MARKER.exists():
        sys.exit(f"Final evaluation already done: {MARKER.read_text()}\nRefusing to run again.")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tables, headline = [], []

    for h in HORIZONS:
        fit, sc = load(h)
        feats = [c for c in fit.columns if c not in NOT_FEATURES + ["day"]]
        print(f"\n######## {h}h: fit {fit.index.min():%Y-%m}..{fit.index.max():%Y-%m} "
              f"({len(fit)} rows), score {sc.index.min():%Y-%m}..{sc.index.max():%Y-%m} "
              f"({len(sc)} rows) ########", flush=True)

        # ---------- temperature ----------
        ytr = fit["y_temp"] - fit["temperature_2m"]
        gbm = lgb.LGBMRegressor(**params("temp", h)).fit(fit[feats], ytr)
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(fit[feats], ytr)
        clim = fit.groupby("tgt_key")["y_temp"].mean()
        c_t, c_n = sc["tgt_key"].map(clim), sc["now_key"].map(clim)
        tp = {"persistence": sc["temperature_2m"], "same_hour_yesterday": sc["base_yday"],
              "climatology": c_t, "clim+anomaly": c_t + (sc["temperature_2m"] - c_n),
              "ridge": sc["temperature_2m"] + ridge.predict(sc[feats]),
              "lightgbm": sc["temperature_2m"] + gbm.predict(sc[feats])}
        best = BEST_BASELINE["temp"][h]
        for s, m in slices(sc).items():
            y = sc["y_temp"][m]
            bmae = (tp[best][m] - y).abs().mean()
            for name, p in tp.items():
                e = p[m] - y
                tables.append({"h": h, "target": "temp", "slice": s, "model": name,
                               "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                               "skill_vs_best": 1 - e.abs().mean() / bmae})
            em, eb = (tp["lightgbm"][m] - y).abs(), (tp[best][m] - y).abs()
            lo, hi = boot_skill(sc["day"][m].values, em.values, eb.values, rng)
            headline.append({"h": h, "target": "temp MAE", "slice": s, "baseline": best,
                             "baseline_score": eb.mean(), "model_score": em.mean(),
                             "skill": 1 - em.mean() / eb.mean(), "ci_lo": lo, "ci_hi": hi})

        # ---------- rain occurrence ----------
        clf = lgb.LGBMClassifier(**params("rain", h)).fit(fit[feats], fit["y_rain"])
        prob = pd.Series(clf.predict_proba(sc[feats])[:, 1], index=sc.index)
        rp = {"seasonal_prob": sc["tgt_key"].map(fit.groupby("tgt_key")["y_rain"].mean()),
              "rain_now": (sc["precipitation"] >= RAIN_THRESHOLD_MM).astype(float),
              "lightgbm": prob}
        best = BEST_BASELINE["rain"][h]
        for s, m in slices(sc, heavy=False).items():     # heavy-day slice: outcome-conditioned
            y = sc["y_rain"][m]
            b = brier_score_loss(y, rp[best][m])
            for name, p in rp.items():
                br = brier_score_loss(y, p[m])
                tables.append({"h": h, "target": "rain", "slice": s, "model": name,
                               "Brier": br, "BSS_vs_best": 1 - br / b,
                               "PR_AUC": average_precision_score(y, p[m]), "wet_share": y.mean()})
            sm, sb = (prob[m] - y) ** 2, (rp[best][m] - y) ** 2
            lo, hi = boot_skill(sc["day"][m].values, sm.values, sb.values, rng)
            headline.append({"h": h, "target": "rain Brier", "slice": s, "baseline": best,
                             "baseline_score": sb.mean(), "model_score": sm.mean(),
                             "skill": 1 - sm.mean() / sb.mean(), "ci_lo": lo, "ci_hi": hi})

        # ---------- rain amount (P x smeared amount-if-wet) ----------
        wet = fit[fit["y_rain"] == 1]
        reg = lgb.LGBMRegressor(**P).fit(wet[feats], np.log1p(wet["y_amount"]))
        amt_wet = np.maximum(np.exp(reg.predict(sc[feats])) * SMEAR[h] - 1, 0)
        expected = prob * amt_wet
        ap = {"zero": pd.Series(0.0, index=sc.index),
              "climatology_mean": sc["tgt_key"].map(fit.groupby("tgt_key")["y_amount"].mean()),
              "rain_now": sc["precipitation"], "lightgbm_expected": expected}
        best = BEST_BASELINE["amount"][h]
        for s, m in slices(sc).items():
            y, w = sc["y_amount"][m], (sc["y_rain"][m] == 1)
            brmse = np.sqrt(((ap[best][m] - y) ** 2).mean())
            for name, p in ap.items():
                e = p[m] - y
                rmse = np.sqrt((e ** 2).mean())
                tables.append({"h": h, "target": "amount", "slice": s, "model": name,
                               "RMSE_all_hours": rmse, "bias": e.mean(),
                               "MAE_wet_hours": e[w].abs().mean(),
                               "total_ratio": p[m].sum() / y.sum() if y.sum() > 0 else np.nan,
                               "RMSE_skill_vs_best": 1 - rmse / brmse})
            sm, sb = (expected[m] - y) ** 2, (ap[best][m] - y) ** 2
            lo, hi = boot_skill(sc["day"][m].values, sm.values, sb.values, rng, sqrt=True)
            headline.append({"h": h, "target": "amount RMSE", "slice": s, "baseline": best,
                             "baseline_score": np.sqrt(sb.mean()),
                             "model_score": np.sqrt(sm.mean()),
                             "skill": 1 - np.sqrt(sm.mean() / sb.mean()), "ci_lo": lo, "ci_hi": hi})

        pd.DataFrame({"y_temp": sc["y_temp"], "temp_lgbm": tp["lightgbm"],
                      "temp_best_baseline": tp[BEST_BASELINE["temp"][h]],
                      "y_rain": sc["y_rain"], "rain_prob": prob,
                      "rain_best_baseline": rp[BEST_BASELINE["rain"][h]],
                      "y_amount": sc["y_amount"], "amt_expected": expected,
                      "amt_best_baseline": ap[BEST_BASELINE["amount"][h]],
                      "eval_monsoon": sc["eval_monsoon"], "eval_heavy": sc["eval_heavy"]}
                     ).to_csv(OUT / f"preds_h{h}.csv")

    tab, head = pd.DataFrame(tables), pd.DataFrame(headline)
    tab.to_csv(OUT / "all_metrics.csv", index=False)
    head.to_csv(OUT / "headline.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n================ HEADLINE: LightGBM vs frozen best baseline ================")
    print(head.round(3).to_string(index=False))

    if FINAL:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        MARKER.write_text(json.dumps({"run_at_utc": datetime.now(timezone.utc).isoformat(),
                                      "git_commit": commit}, indent=2))
        print(f"\nFinal run recorded in {MARKER}")


if __name__ == "__main__":
    main()
