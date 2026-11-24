"""Error analysis of the frozen models on the 2025 test year (analysis only, no tuning).

Reads <PROCESSED_DIR>/test_eval/preds_h{h}.csv written by evaluate_test.py --final.
Writes figures to reports/figures/ and tables to <PROCESSED_DIR>/test_eval/error_analysis/.
  1. Temperature MAE by IST hour of the target and by month (model vs best baseline)
  2. Rain-probability calibration (reliability) and sharpness, monsoon vs non-monsoon
  3. Heavy-rain days: forecast vs observed daily totals
  4. Worst temperature errors and the largest missed rain hours
Run from project root: python -m src.evaluation.error_analysis
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import PROCESSED_DIR, HORIZONS, HEAVY_DAY_MM
from src.models.frozen_config import IST, BEST_BASELINE

SRC = PROCESSED_DIR / "test_eval"
OUT = SRC / "error_analysis"
FIG = Path("reports/figures")
COL = {1: "#1f77b4", 6: "#ff7f0e", 24: "#2ca02c"}


def load(h):
    d = pd.read_csv(SRC / f"preds_h{h}.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    tgt = d.index.tz_convert(IST) + pd.Timedelta(hours=h)      # valid time of the forecast
    d["tgt_hour"], d["tgt_month"] = tgt.hour, tgt.month
    d["day"] = (tgt - pd.Timedelta(hours=1)).date               # rain hour (t-1h, t] -> IST day
    d["e_model"] = d["temp_lgbm"] - d["y_temp"]
    d["e_base"] = d["temp_best_baseline"] - d["y_temp"]
    return d


def temp_by(d, key):
    g = d.groupby(key)
    return pd.DataFrame({"model_MAE": g["e_model"].apply(lambda e: e.abs().mean()),
                         "baseline_MAE": g["e_base"].apply(lambda e: e.abs().mean()),
                         "model_bias": g["e_model"].mean()})


def reliability(p, y, bins=10):
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    t = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(
        mean_p=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    return t[t["n"] >= 20]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    D = {h: load(h) for h in HORIZONS}

    # ---------------- 1. temperature by hour and month ----------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for h, d in D.items():
        bh, bm = temp_by(d, "tgt_hour"), temp_by(d, "tgt_month")
        bh.to_csv(OUT / f"temp_by_hour_h{h}.csv"); bm.to_csv(OUT / f"temp_by_month_h{h}.csv")
        for a, t in [(ax[0], bh), (ax[1], bm)]:
            a.plot(t.index, t["model_MAE"], color=COL[h], label=f"{h}h LightGBM")
            a.plot(t.index, t["baseline_MAE"], color=COL[h], ls="--", alpha=.6,
                   label=f"{h}h {BEST_BASELINE['temp'][h]}")
        print(f"\n[temp {h}h] worst target hours (IST) by model MAE:\n"
              + bh.sort_values("model_MAE", ascending=False).head(4).round(3).to_string())
        print(f"[temp {h}h] by month:\n" + bm.round(3).T.to_string())
    ax[0].set(xlabel="Target hour (IST)", ylabel="MAE (°C)", title="Temperature MAE by hour, 2025")
    ax[1].set(xlabel="Target month", ylabel="MAE (°C)", title="Temperature MAE by month, 2025")
    ax[1].set_xticks(range(1, 13))
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "temp_mae_by_hour_month.png", dpi=120); plt.close(fig)

    # ---------------- 2. rain-probability calibration ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, (name, mon) in enumerate([("monsoon", 1.0), ("non-monsoon", 0.0)]):
        ax[i].plot([0, 1], [0, 1], color="grey", lw=.8)
        for h, d in D.items():
            s = d[d["eval_monsoon"] == mon]
            r = reliability(s["rain_prob"].values, s["y_rain"].values)
            r.to_csv(OUT / f"reliability_{name}_h{h}.csv")
            ax[i].plot(r["mean_p"], r["obs"], "o-", color=COL[h], label=f"{h}h")
            # one-number summaries: mean forecast vs observed frequency
            print(f"[rain {h}h {name}] mean P {s['rain_prob'].mean():.3f} vs observed "
                  f"{s['y_rain'].mean():.3f}; max calibration gap "
                  f"{(r['obs'] - r['mean_p']).abs().max():.3f}")
        ax[i].set(xlabel="Forecast probability", ylabel="Observed frequency",
                  title=f"Rain reliability, {name} 2025", xlim=(0, 1), ylim=(0, 1))
        ax[i].legend()
    fig.tight_layout(); fig.savefig(FIG / "rain_reliability.png", dpi=120); plt.close(fig)

    # ---------------- 3. heavy-rain days: daily totals ----------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    rows = []
    for i, (h, d) in enumerate(D.items()):
        day = d.groupby("day").agg(obs=("y_amount", "sum"), model=("amt_expected", "sum"),
                                   base=("amt_best_baseline", "sum"), hours=("y_amount", "size"))
        day = day[day["hours"] >= 20]                     # skip partial edge days
        heavy = day[day["obs"] >= HEAVY_DAY_MM]
        rows.append({"h": h, "heavy_days": len(heavy),
                     "obs_mean_mm": heavy["obs"].mean(), "model_mean_mm": heavy["model"].mean(),
                     "baseline_mean_mm": heavy["base"].mean(),
                     "model_ratio": heavy["model"].sum() / heavy["obs"].sum(),
                     "corr_daily_all": day[["obs", "model"]].corr().iloc[0, 1],
                     "heavy_days_forecast_over_10mm": int((heavy["model"] >= 10).sum())})
        ax[i].scatter(day["obs"], day["model"], s=8, alpha=.5, color=COL[h])
        ax[i].scatter(heavy["obs"], heavy["model"], s=14, color="red", label="heavy day")
        m = max(day["obs"].max(), day["model"].max())
        ax[i].plot([0, m], [0, m], color="grey", lw=.8)
        ax[i].axvline(HEAVY_DAY_MM, color="red", lw=.6, ls=":")
        ax[i].set(title=f"{h}h: daily total, 2025", xlabel="Observed (mm)")
    ax[0].set_ylabel("Sum of hourly expected forecasts (mm)"); ax[0].legend()
    fig.tight_layout(); fig.savefig(FIG / "heavy_day_totals.png", dpi=120); plt.close(fig)
    hv = pd.DataFrame(rows); hv.to_csv(OUT / "heavy_days.csv", index=False)
    print("\n[heavy-rain days, 2025]\n" + hv.round(3).to_string(index=False))

    # ---------------- 4. worst cases ----------------
    for h, d in D.items():
        w = d.reindex(d["e_model"].abs().sort_values(ascending=False).index).head(8)
        w = w.assign(valid_ist=(w.index.tz_convert(IST) + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M"))
        w[["valid_ist", "y_temp", "temp_lgbm", "temp_best_baseline", "e_model",
           "eval_monsoon"]].to_csv(OUT / f"worst_temp_h{h}.csv", index=False)
        print(f"\n[worst temp errors {h}h]\n" + w[["valid_ist", "y_temp", "temp_lgbm",
              "temp_best_baseline", "e_model"]].round(2).to_string(index=False))
        miss = d[d["y_rain"] == 1].sort_values("y_amount", ascending=False).head(8)
        miss = miss.assign(valid_ist=(miss.index.tz_convert(IST) + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M"))
        miss[["valid_ist", "y_amount", "rain_prob", "amt_expected"]].to_csv(
            OUT / f"wettest_hours_h{h}.csv", index=False)
        print(f"[wettest hours {h}h: what was forecast]\n" + miss[["valid_ist", "y_amount",
              "rain_prob", "amt_expected"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
