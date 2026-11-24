"""FROZEN configuration (2026-11-24). Do not change after the 2025 test evaluation.

Everything here was chosen without looking at 2025:
  - hyperparameters by walk-forward CV inside 2011-2022 (src/models/tune_cv.py)
  - smearing factors by walk-forward CV inside 2011-2022 (src/models/rain_amount_correction.py)
  - best baselines per horizon from the 2023-2024 validation split
Thresholds live in src/config.py (RAIN_THRESHOLD_MM = 0.1, HEAVY_DAY_MM = 17.5).
"""
IST = "Asia/Kolkata"

# Columns that are never model inputs.
NOT_FEATURES = ["y_temp", "y_amount", "y_rain", "base_yday",
                "eval_heavy", "eval_monsoon", "split", "now_key", "tgt_key"]

# Base LightGBM settings (the rain-amount model uses these as-is; it was not tuned).
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
         random_state=42, verbose=-1)

# Tuned (num_leaves, min_child_samples, learning_rate, n_estimators); all use doy features.
TUNED = {("temp", 1): (127, 20, 0.02, 880), ("temp", 6): (127, 200, 0.02, 934),
         ("temp", 24): (127, 200, 0.02, 356), ("rain", 1): (31, 200, 0.05, 218),
         ("rain", 6): (127, 200, 0.05, 95), ("rain", 24): (15, 200, 0.05, 160)}

# Duan smearing factor for the log1p rain-amount model: expm1(pred) alone underestimates totals.
SMEAR = {1: 1.089, 6: 1.206, 24: 1.230}

# Best baseline per horizon, chosen on validation 2023-2024 (headline metric in brackets).
BEST_BASELINE = {
    "temp": {1: "clim+anomaly", 6: "same_hour_yesterday", 24: "persistence"},   # [MAE]
    "rain": {1: "rain_now", 6: "seasonal_prob", 24: "seasonal_prob"},           # [Brier]
    "amount": {1: "rain_now", 6: "climatology_mean", 24: "climatology_mean"},   # [RMSE, all hours]
}


def params(task, h):
    nl, mcs, lr, n = TUNED[(task, h)]
    return {**P, "num_leaves": nl, "min_child_samples": mcs, "learning_rate": lr,
            "n_estimators": n}
