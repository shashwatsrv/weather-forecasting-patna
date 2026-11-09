import os
from pathlib import Path



DATA_DIR = Path(os.environ.get("WEATHER_DATA_DIR", r"D:\datasets\weather-patna"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"


# Locked after EDA (thresholds computed from 2011-2022 only)
RAIN_THRESHOLD_MM = 0.1      # wet hour = precipitation >= 0.1 mm (data is quantized at 0.1)
HEAVY_DAY_MM = 17.8          # training-set 95th percentile of daily total (IST days)
HORIZONS = [1, 6, 24]
TRAIN_END = "2022-12-31"     # train 2011-2022
VAL_END = "2024-12-31"       # validate 2023-2024; 2025 is the untouched test year