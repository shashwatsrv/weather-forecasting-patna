"""Rolling-origin evaluation of the FROZEN recipe (reporting only, nothing is tuned here).

For each year Y in 2019-2025: fit on every row whose target is before Y, score year Y.
Shows how much skill varies from year to year, instead of relying on one test year.

Caveat, reported with the results: only 2025 is fully out-of-sample. Hyperparameters and
smearing factors were chosen by walk-forward CV on 2019-2022, and the best baselines were
picked on 2023-2024, so those years are somewhat optimistic for the model.

Run from project root: python -m src.evaluation.rolling_origin
Output: <PROCESSED_DIR>/rolling_origin/by_year.csv and summary.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss
from src.config import PROCESSED_DIR, HORIZONS, RAIN_THRESHOLD_MM
from src.models.frozen_config import IST, NOT_FEATURES, P, SMEAR, BEST_BASELINE, params
from src.evaluation.evaluate_test import boot_skill

YEARS = range(2019, 2026)
OUT = PROCESSED_DIR / "rolling_origin"
USED_FOR = {**{y: "CV fold (tuning)" for y in range(2019, 2023)},
            2023: "validation (baseline choice)", 2024: "validation (baseline choice)",
            2025: "test (clean)"}


def load(h):
    d = pd.read_csv(PROCESSED_DIR / f"features_h{h}.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    now, tgt = d.index.tz_convert(IST), d.index.tz_convert(IST) + pd.Timedelta(hours=h)
    d["now_key"] = np.asarray(now.month) * 100 + np.asarray(now.hour)
    d["tgt_key"] = np.asarray(tgt.month) * 100 + np.asarray(tgt.hour)
    d["day"] = (tgt - pd.Timedelta(hours=1)).date
    d["tgt_utc"] = d.index + pd.Timedelta(hours=h)
    return d


def score_year(d, h, y, feats, rng):
    start, end = pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
    fit = d[d["tgt_utc"] < start]                                   # targets strictly before Y
    sc = d[(d.index >= start) & (d["tgt_utc"] < end)]
    assert fit["tgt_utc"].max() < sc.index.min() + pd.Timedelta(hours=h)
    rows = []

    def add(target, num, den, sqrt=False):
        for s, m in [("all", np.ones(len(sc), bool)), ("monsoon", (sc["eval_monsoon"] == 1).values)]:
            n, dd = num[m], den[m]
            r = n.mean() / dd.mean()
            lo, hi = boot_skill(sc["day"][m].values, n.values, dd.values, rng, sqrt=sqrt)
            rows.append({"h": h, "year": y, "used_for": USED_FOR[y], "target": target, "slice": s,
                         "skill": 1 - (np.sqrt(r) if sqrt else r), "ci_lo": lo, "ci_hi": hi})

    # temperature: change from current temperature
    gbm = lgb.LGBMRegressor(**params("temp", h)).fit(fit[feats], fit["y_temp"] - fit["temperature_2m"])
    clim = fit.groupby("tgt_key")["y_temp"].mean()
    c_t, c_n = sc["tgt_key"].map(clim), sc["now_key"].map(clim)
    tb = {"persistence": sc["temperature_2m"], "same_hour_yesterday": sc["base_yday"],
          "clim+anomaly": c_t + (sc["temperature_2m"] - c_n)}[BEST_BASELINE["temp"][h]]
    tm = sc["temperature_2m"] + gbm.predict(sc[feats])
    add("temp MAE", (tm - sc["y_temp"]).abs(), (tb - sc["y_temp"]).abs())

    # rain occurrence
    clf = lgb.LGBMClassifier(**params("rain", h)).fit(fit[feats], fit["y_rain"])
    prob = pd.Series(clf.predict_proba(sc[feats])[:, 1], index=sc.index)
    rb = {"seasonal_prob": sc["tgt_key"].map(fit.groupby("tgt_key")["y_rain"].mean()),
          "rain_now": (sc["precipitation"] >= RAIN_THRESHOLD_MM).astype(float)}[BEST_BASELINE["rain"][h]]
    add("rain Brier", (prob - sc["y_rain"]) ** 2, (rb - sc["y_rain"]) ** 2)
    assert np.isclose(((prob - sc["y_rain"]) ** 2).mean(), brier_score_loss(sc["y_rain"], prob))

    # rain amount: P(rain) x smeared amount-if-wet
    wet = fit[fit["y_rain"] == 1]
    reg = lgb.LGBMRegressor(**P).fit(wet[feats], np.log1p(wet["y_amount"]))
    expected = prob * np.maximum(np.exp(reg.predict(sc[feats])) * SMEAR[h] - 1, 0)
    ab = {"climatology_mean": sc["tgt_key"].map(fit.groupby("tgt_key")["y_amount"].mean()),
          "rain_now": sc["precipitation"]}[BEST_BASELINE["amount"][h]]
    add("amount RMSE", (expected - sc["y_amount"]) ** 2, (ab - sc["y_amount"]) ** 2, sqrt=True)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for h in HORIZONS:
        d = load(h)
        feats = [c for c in d.columns if c not in NOT_FEATURES + ["day", "tgt_utc"]]
        for y in YEARS:
            rows += score_year(d, h, y, feats, rng)
            print(f"{h}h {y} done", flush=True)
    by = pd.DataFrame(rows)
    by.to_csv(OUT / "by_year.csv", index=False)

    a = by[by["slice"] == "all"]
    summ = (a.groupby(["target", "h"])["skill"]
             .agg(mean="mean", sd="std", min="min", max="max", years="size").reset_index())
    clean = a[a["year"] == 2025].set_index(["target", "h"])["skill"].rename("skill_2025")
    summ = summ.join(clean, on=["target", "h"])
    summ.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + a.pivot_table(index=["target", "h"], columns="year", values="skill").round(3).to_string())
    print("\n" + summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
