"""Walk-forward CV tuning of LightGBM, using the TRAIN split (2011-2022) only.

Folds (expanding window, boundaries in UTC like build_features.py):
    train 2011..Y-1  ->  validate year Y,   for Y in 2019, 2020, 2021, 2022
Training rows whose target (t + h) falls inside the validation year are dropped,
same rule as the main split. 2023-2024 (val) and 2025 (test) are never loaded here.

Stage 1: grid over num_leaves x min_child_samples x (with / without doy features)
         at learning_rate 0.05; n_estimators chosen by early stopping per fold.
Stage 2: learning-rate check (0.02 / 0.05 / 0.1) on the stage-1 winner.

Every config is compared to the current default from train_models.py
(31 leaves, min_child_samples 50, 400 trees, doy on) with a PAIRED per-fold
difference, because year-to-year difficulty dominates the raw spread.

Run from project root: python -m src.models.tune_cv
"""
import json
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss
from src.config import PROCESSED_DIR, HORIZONS

# Same list as train_models.py (copied: importing that module would run training).
NOT_FEATURES = ["y_temp", "y_amount", "y_rain", "base_yday",
                "eval_heavy", "eval_monsoon", "split", "now_key", "tgt_key"]
FOLD_YEARS = [2019, 2020, 2021, 2022]
BASE = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=-1)
DEFAULT = dict(num_leaves=31, min_child_samples=50, learning_rate=0.05, doy=True)
GRID = dict(num_leaves=[15, 31, 63, 127], min_child_samples=[20, 50, 200], doy=[True, False])
LR_CHECK = [0.02, 0.05, 0.1]
MAX_TREES, ES_ROUNDS = 3000, 100
OUT = PROCESSED_DIR / "tuning"


def load_train(h):
    d = pd.read_csv(PROCESSED_DIR / f"features_h{h}.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    d = d[d["split"] == "train"]
    assert d.index.max() < pd.Timestamp("2023-01-01", tz="UTC"), "non-train rows loaded"
    return d


def folds(d, h):
    for y in FOLD_YEARS:
        start = pd.Timestamp(f"{y}-01-01", tz="UTC")
        end = pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
        tr = d[d.index + pd.Timedelta(hours=h) < start]          # target before fold year
        va = d[(d.index >= start) & (d.index < end)]
        assert (tr.index + pd.Timedelta(hours=h)).max() < va.index.min()
        yield y, tr, va


def fit_eval(task, cfg, tr, va, feats, fixed_trees=None):
    """Return (score on validation fold, trees used). Lower score is better."""
    p = {**BASE, "num_leaves": cfg["num_leaves"],
         "min_child_samples": cfg["min_child_samples"], "learning_rate": cfg["learning_rate"],
         "n_estimators": fixed_trees or MAX_TREES}
    cb = [] if fixed_trees else [lgb.early_stopping(ES_ROUNDS, verbose=False)]
    if task == "temp":
        ytr, yva = tr["y_temp"] - tr["temperature_2m"], va["y_temp"] - va["temperature_2m"]
        m = lgb.LGBMRegressor(**p).fit(tr[feats], ytr, eval_set=[(va[feats], yva)],
                                       eval_metric="l1", callbacks=cb)
        score = np.abs(m.predict(va[feats]) - yva).mean()                  # MAE, deg C
    else:
        m = lgb.LGBMClassifier(**p).fit(tr[feats], tr["y_rain"],
                                        eval_set=[(va[feats], va["y_rain"])],
                                        eval_metric="binary_logloss", callbacks=cb)
        score = brier_score_loss(va["y_rain"], m.predict_proba(va[feats])[:, 1])
    return score, (m.best_iteration_ or p["n_estimators"])


def feature_list(d, doy):
    feats = [c for c in d.columns if c not in NOT_FEATURES]
    return feats if doy else [c for c in feats if not c.startswith("doy_")]


def run_configs(task, cfgs, fold_data, d):
    rows = []
    for cfg in cfgs:
        feats = feature_list(d, cfg["doy"])
        for y, tr, va in fold_data:
            fixed = 400 if cfg.get("is_default") else None
            s, n = fit_eval(task, cfg, tr, va, feats, fixed_trees=fixed)
            rows.append({**{k: v for k, v in cfg.items() if k != "is_default"},
                         "config": cfg_name(cfg), "fold": y, "score": s, "trees": n})
    return pd.DataFrame(rows)


def cfg_name(cfg):
    if cfg.get("is_default"):
        return "DEFAULT(400 trees)"
    return (f"lr{cfg['learning_rate']}_nl{cfg['num_leaves']}_mcs{cfg['min_child_samples']}"
            f"_{'doy' if cfg['doy'] else 'nodoy'}")


def summarize(res):
    """Mean score, and paired difference vs DEFAULT per fold (negative = better)."""
    piv = res.pivot_table(index="config", columns="fold", values="score")
    diff = piv.sub(piv.loc["DEFAULT(400 trees)"], axis=1)
    s = pd.DataFrame({"mean_score": piv.mean(axis=1),
                      "mean_diff_vs_default": diff.mean(axis=1),
                      "folds_better": (diff < 0).sum(axis=1),
                      "worst_fold_diff": diff.max(axis=1),
                      "mean_trees": res.groupby("config")["trees"].mean().round()})
    s["rel_gain_%"] = -100 * s["mean_diff_vs_default"] / piv.loc["DEFAULT(400 trees)"].mean()
    return s.sort_values("mean_score")


def main():
    OUT.mkdir(exist_ok=True)
    chosen = {}
    grid = [dict(zip(GRID, v), learning_rate=0.05) for v in itertools.product(*GRID.values())]
    for h in HORIZONS:
        d = load_train(h)
        fold_data = list(folds(d, h))
        for task in ["temp", "rain"]:
            label = "MAE degC" if task == "temp" else "Brier"
            print(f"\n######## h={h}h  {task}  ({label}, lower is better) ########", flush=True)
            res = run_configs(task, [{**DEFAULT, "is_default": True}] + grid, fold_data, d)
            s1 = summarize(res)
            best = res[res["config"] == s1.drop("DEFAULT(400 trees)").index[0]].iloc[0]
            lr_cfgs = [dict(num_leaves=int(best["num_leaves"]),
                            min_child_samples=int(best["min_child_samples"]),
                            doy=bool(best["doy"]), learning_rate=lr)
                       for lr in LR_CHECK if lr != 0.05]
            res = pd.concat([res, run_configs(task, lr_cfgs, fold_data, d)])
            res.to_csv(OUT / f"cv_{task}_h{h}.csv", index=False)
            s = summarize(res)
            print(s.head(8).round(4).to_string())
            print("DEFAULT row:\n" + s.loc[["DEFAULT(400 trees)"]].round(4).to_string())
            # doy question: best with vs best without, paired over folds
            piv = res.pivot_table(index="config", columns="fold", values="score")
            bd = s[s.index.str.endswith("_doy")].index[0]
            bn = s[s.index.str.endswith("_nodoy")].index[0]
            dd = piv.loc[bn] - piv.loc[bd]
            print(f"doy check: best nodoy - best doy per fold = {dd.round(4).tolist()} "
                  f"(mean {dd.mean():+.4f}; positive = doy helps)")
            win = s.drop("DEFAULT(400 trees)").index[0]
            r = res[res["config"] == win].iloc[0]
            chosen[f"{task}_h{h}"] = dict(config=win, num_leaves=int(r["num_leaves"]),
                                          min_child_samples=int(r["min_child_samples"]),
                                          learning_rate=float(r["learning_rate"]),
                                          doy=bool(r["doy"]),
                                          n_estimators=int(s.loc[win, "mean_trees"]),
                                          cv_score=float(s.loc[win, "mean_score"]),
                                          default_cv_score=float(s.loc["DEFAULT(400 trees)",
                                                                       "mean_score"]))
    (OUT / "cv_best_params.json").write_text(json.dumps(chosen, indent=2))
    print("\nSaved", OUT / "cv_best_params.json")


if __name__ == "__main__":
    main()
