import os
from pathlib import Path



DATA_DIR = Path(os.environ.get("WEATHER_DATA_DIR", r"D:\datasets\weather-patna"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"


# Locked after EDA (thresholds computed from 2011-2022 only)
RAIN_THRESHOLD_MM = 0.1      # wet hour = precipitation >= 0.1 mm (data is quantized at 0.1)
HEAVY_DAY_MM = 17.5          # training-set 95th percentile of daily total (IST days)
# Was 17.8. Changed 2026-11-24: a value stamped t covers the hour ending at t (ERA5 docs),
# so hours are now assigned to the IST day of (t - 1h); the 00:00 value belongs to the day before.
HORIZONS = [1, 6, 24]
TRAIN_END = "2022-12-31"     # train 2011-2022
VAL_END = "2024-12-31"       # validate 2023-2024; 2025 is the untouched test year