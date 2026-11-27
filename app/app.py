"""Backtesting dashboard for the frozen 2025 test-year predictions.

Run from the project root:  streamlit run app/app.py
Reads <WEATHER_DATA_DIR>/processed/test_eval/ written by src/evaluation/evaluate_test.py --final.
If that folder isn't there (e.g. on Streamlit Community Cloud), it reads byte-identical copies
published in the live-log repo, so this repo stays code-only.
Shows precomputed results only; it never calls a weather API.
"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DIR, HEAVY_DAY_MM, RAIN_THRESHOLD_MM  # noqa: E402
from src.models.frozen_config import IST, BEST_BASELINE  # noqa: E402

LOCAL = PROCESSED_DIR / "test_eval"
REMOTE = "https://raw.githubusercontent.com/shashwatsrv/weather-patna-live-log/main/data/backtest/"
SRC = str(LOCAL) + "/" if (LOCAL / "headline.csv").exists() else REMOTE
alt.data_transformers.disable_max_rows()
st.set_page_config(page_title="Patna ERA5 forecast backtest", layout="wide")


@st.cache_data
def load_preds(h):
    d = pd.read_csv(SRC + f"preds_h{h}.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    valid = d.index.tz_convert(IST) + pd.Timedelta(hours=h)
    d["valid"] = valid.tz_localize(None)                        # IST wall-clock time
    d["day"] = (valid - pd.Timedelta(hours=1)).date            # rain (t-1h, t] -> IST day
    return d.reset_index(drop=True)


@st.cache_data
def load_headline():
    return pd.read_csv(SRC + "headline.csv")


try:
    load_headline()
except Exception as e:
    st.error(f"Couldn't load the test-year results from {SRC} ({e}). Locally, run "
             "`python -m src.evaluation.evaluate_test --final` first.")
    st.stop()

# ---------------- header ----------------
st.title("Patna ERA5 forecast backtest, 2025")
st.warning("**Backtest, not a live forecast.** Every number on this page comes from forecasts "
           "made for the held-out year 2025 using historical ERA5 reanalysis, with a "
           "configuration frozen before 2025 was looked at.")
st.caption("Location: the ERA5 0.25° grid cell at 25.5°N, 85.25°E, about 15 km from central "
           "Patna. Times are IST. Models: LightGBM, one per horizon, trained on 2011-2024.")

with st.sidebar:
    st.header("Options")
    h = st.radio("Forecast horizon", [1, 6, 24], format_func=lambda x: f"{x} hour(s) ahead")
    d = load_preds(h)
    start_default = pd.Timestamp("2025-07-25")
    rng = st.date_input("Window (IST)", value=(start_default.date(),
                        (start_default + pd.Timedelta(days=13)).date()),
                        min_value=d["valid"].min().date(), max_value=d["valid"].max().date())
    st.markdown("---")
    st.markdown(f"**Best baselines ({h}h)**  \n"
                f"Temperature: `{BEST_BASELINE['temp'][h]}`  \n"
                f"Rain occurrence: `{BEST_BASELINE['rain'][h]}`  \n"
                f"Rain amount: `{BEST_BASELINE['amount'][h]}`")
    st.caption("Baselines were chosen on the 2023-2024 validation years.")

lo, hi = (rng if isinstance(rng, tuple) and len(rng) == 2 else (rng, rng))
w = d[(d["valid"] >= pd.Timestamp(lo)) & (d["valid"] < pd.Timestamp(hi) + pd.Timedelta(days=1))]

# ---------------- headline ----------------
st.subheader(f"Skill vs the best baseline, {h}h, whole year 2025")
hd = load_headline()
hd = hd[hd["h"] == h].copy()
hd["skill (95% CI)"] = hd.apply(lambda r: f"{r.skill:+.1%}  [{r.ci_lo:+.1%}, {r.ci_hi:+.1%}]",
                                axis=1)
show = hd[["target", "slice", "baseline", "baseline_score", "model_score", "skill (95% CI)"]]
st.dataframe(show.rename(columns={"baseline_score": "baseline", "model_score": "LightGBM",
                                  "baseline": "baseline name"}).round(3),
             hide_index=True, use_container_width=True)
st.caption("Skill = 1 - model error / baseline error (MAE for temperature, Brier score for "
           "rain occurrence, RMSE for rain amount). Intervals come from resampling whole days. "
           "Rain occurrence isn't scored on heavy-rain days, because that slice is defined by "
           "the outcome.")

# ---------------- temperature ----------------
st.subheader("Temperature")
tl = w.melt(id_vars="valid", value_vars=["y_temp", "temp_lgbm", "temp_best_baseline"],
            var_name="series", value_name="°C")
names = {"y_temp": "ERA5 (truth)", "temp_lgbm": "LightGBM",
         "temp_best_baseline": f"baseline: {BEST_BASELINE['temp'][h]}"}
tl["series"] = tl["series"].map(names)
st.altair_chart(alt.Chart(tl).mark_line(strokeWidth=1.5).encode(
    x=alt.X("valid:T", title="Valid time (IST)"), y=alt.Y("°C:Q", scale=alt.Scale(zero=False)),
    color=alt.Color("series:N", scale=alt.Scale(domain=list(names.values()),
                    range=["#9467bd", "#1f77b4", "#ff7f0e"])),
    tooltip=["valid:T", "series:N", alt.Tooltip("°C:Q", format=".1f")]
).properties(height=300), use_container_width=True)
c1, c2, c3 = st.columns(3)
if len(w):
    em, eb = (w["temp_lgbm"] - w["y_temp"]).abs().mean(), (w["temp_best_baseline"] - w["y_temp"]).abs().mean()
    c1.metric("LightGBM MAE in window", f"{em:.2f} °C")
    c2.metric("Baseline MAE in window", f"{eb:.2f} °C")
    c3.metric("Skill in window", f"{1 - em / eb:+.1%}" if eb > 0 else "n/a")
st.caption("A short window is noisy; use the whole-year table above for conclusions.")

# ---------------- rain ----------------
st.subheader("Rain")
base = alt.Chart(w).encode(x=alt.X("valid:T", title="Valid time (IST)"))
prob = base.mark_line(color="#1f77b4").encode(
    y=alt.Y("rain_prob:Q", title="P(rain ≥ 0.1 mm)", scale=alt.Scale(domain=[0, 1])),
    tooltip=["valid:T", alt.Tooltip("rain_prob:Q", format=".2f")])
wet = base.transform_filter(alt.datum.y_rain == 1).mark_tick(color="#2ca02c", thickness=2).encode(
    y=alt.value(4))
st.altair_chart((prob + wet).properties(height=200, title="Rain probability (green ticks = "
                "it actually rained that hour)"), use_container_width=True)

al = w.melt(id_vars="valid", value_vars=["y_amount", "amt_expected", "amt_best_baseline"],
            var_name="series", value_name="mm")
anames = {"y_amount": "ERA5 (truth)", "amt_expected": "LightGBM expected",
          "amt_best_baseline": f"baseline: {BEST_BASELINE['amount'][h]}"}
al["series"] = al["series"].map(anames)
st.altair_chart(alt.Chart(al).mark_line(strokeWidth=1.2).encode(
    x=alt.X("valid:T", title="Valid time (IST)"), y=alt.Y("mm:Q", title="Rain (mm/h)"),
    color=alt.Color("series:N", scale=alt.Scale(domain=list(anames.values()),
                    range=["#9467bd", "#1f77b4", "#ff7f0e"])),
    tooltip=["valid:T", "series:N", alt.Tooltip("mm:Q", format=".2f")]
).properties(height=250, title="Hourly rain amount"), use_container_width=True)
st.caption("The LightGBM amount is an expected value: P(rain) × amount if it rains. It spreads "
           "uncertain peaks out, so it rarely reaches the heaviest hourly totals.")

# ---------------- whole-year diagnostics ----------------
st.subheader(f"Whole-year diagnostics, {h}h")
c1, c2 = st.columns(2)
bins = np.clip((d["rain_prob"] * 10).astype(int), 0, 9)
rel = (d.assign(b=bins, season=np.where(d["eval_monsoon"] == 1, "monsoon", "non-monsoon"))
       .groupby(["season", "b"]).agg(forecast=("rain_prob", "mean"), observed=("y_rain", "mean"),
                                     n=("y_rain", "size")).reset_index())
rel = rel[rel["n"] >= 20]
diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="grey").encode(x="x", y="y")
c1.altair_chart((diag + alt.Chart(rel).mark_line(point=True).encode(
    x=alt.X("forecast:Q", title="Forecast probability", scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("observed:Q", title="Observed frequency", scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("season:N", scale=alt.Scale(domain=["monsoon", "non-monsoon"],
                    range=["#1f77b4", "#ff7f0e"])), tooltip=["season", alt.Tooltip("forecast:Q", format=".2f"),
                               alt.Tooltip("observed:Q", format=".2f"), "n"]))
    .properties(height=300, title="Rain probability calibration"), use_container_width=True)

day = d.groupby("day").agg(observed=("y_amount", "sum"), forecast=("amt_expected", "sum"),
                           hours=("y_amount", "size")).reset_index()
day = day[day["hours"] >= 20]
day["heavy day"] = day["observed"] >= HEAVY_DAY_MM
m = float(max(day["observed"].max(), day["forecast"].max()))
line = alt.Chart(pd.DataFrame({"x": [0, m], "y": [0, m]})).mark_line(color="grey").encode(x="x", y="y")
c2.altair_chart((line + alt.Chart(day).mark_circle(size=30).encode(
    x=alt.X("observed:Q", title="Observed daily total (mm)"),
    y=alt.Y("forecast:Q", title="Forecast daily total (mm)"),
    color=alt.Color("heavy day:N", scale=alt.Scale(domain=[False, True], range=["#9ecae1", "#d62728"])),
    tooltip=["day:T", alt.Tooltip("observed:Q", format=".1f"), alt.Tooltip("forecast:Q", format=".1f")]))
    .properties(height=300, title=f"Daily rain totals (heavy day ≥ {HEAVY_DAY_MM} mm)"),
    use_container_width=True)
st.caption(f"Wet hour = at least {RAIN_THRESHOLD_MM} mm. Points above the diagonal on the left "
           "mean the model was under-confident: it rained more often than it said.")

# ---------------- footer / attribution ----------------
st.markdown("---")
st.markdown(
    "[Weather data by Open-Meteo.com](https://open-meteo.com/) "
    "([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). "
    "Contains modified Copernicus Climate Change Service information 2026. Neither the "
    "European Commission nor ECMWF is responsible for any use that may be made of the "
    "Copernicus information or data it contains. ERA5 dataset: "
    "[doi:10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47). "
    "This page shows a historical backtest, not a weather forecast.")
