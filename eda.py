"""
Exploratory data analysis: prints the key data findings and saves two plots to outputs/.

Usage:  python eda.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA, OUT = Path("data"), Path("outputs")
OUT.mkdir(exist_ok=True)
train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
valid = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])

print("=== 1. Shape and time coverage ===")
print(f"train: {len(train):,} loads, {train.date.min():%Y-%m-%d} .. {train.date.max():%Y-%m-%d}")
print(f"valid: {len(valid):,} loads, {valid.date.min():%Y-%m-%d} .. {valid.date.max():%Y-%m-%d}")
print("-> validation is entirely AFTER training: we must forecast forward, so validate on later months.\n")

print("=== 2. Data quality ===")
for name, d in [("train", train), ("valid", valid)]:
    print(f"{name}: missing weight={d.weight.isna().sum()}, negative weight={(d.weight < 0).sum()}, "
          f"weight capped at 47,500={(d.weight == 47500).sum()}, missing market_index={d.market_index.isna().sum()}")
seen_cities = set(train.pickup) | set(train.delivery)
new_cities = sorted((set(valid.pickup) | set(valid.delivery)) - seen_cities)
print(f"cities in validation never seen in training ({len(new_cities)}): {new_cities}")
print("-> use coordinates (lat/lon) instead of city names so the model works for new cities.")

# Corrupted labels: rate per mile far away from what is normal for that distance/equipment.
X = pd.get_dummies(train.equipment).astype(float).assign(log_dist=np.log(train.distance))
y = np.log(train.posted_rate)
resid = y - LinearRegression().fit(X, y).predict(X)
bad = np.abs(resid - np.median(resid)) > 0.5
print(f"corrupted posted_rate labels: {bad.sum()} ({bad.mean():.1%}), typically x{np.exp(resid[bad & (resid > 0)]).median():.1f} "
      f"too high or x{np.exp(resid[bad & (resid < 0)]).median():.2f} too low -> removed from training.\n")

print("=== 3. What drives the rate level over time? ===")
clean = train[~bad].copy()
clean["resid"] = resid[~bad]  # log rate after removing distance + equipment effects
daily = clean.groupby("date").agg(resid=("resid", "mean"), mi=("market_index", "mean"))
daily["log_mi"] = np.log(daily.mi)
daily["t"] = (daily.index - pd.Timestamp("2025-01-01")).days / 365
daily["qramp"] = (daily.index.month % 3 == 0) * (daily.index.day - 1) / (daily.index.days_in_month - 1)
for cols in [["log_mi"], ["log_mi", "t"], ["log_mi", "t", "qramp"]]:
    r = LinearRegression().fit(daily[cols], daily.resid)
    print(f"daily level ~ {' + '.join(cols):22s} R2={r.score(daily[cols], daily.resid):.3f}  coefs={np.round(r.coef_, 3)}")
print("-> market index explains part of it; a steady ~7%/year upward drift and a ~4% ramp through the last")
print("   month of each quarter (Mar, Jun, Sep) explain almost all the rest. December is a quarter-end month.\n")

# Plot: daily price level with quarter-end months shaded.
fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
ax.plot(daily.index, 100 * daily.resid, color="#064A56", lw=1.5, label="daily avg rate level (after distance & equipment)")
fit = LinearRegression().fit(daily[["log_mi", "t", "qramp"]], daily.resid)
ax.plot(daily.index, 100 * fit.predict(daily[["log_mi", "t", "qramp"]]), color="#E07A1F", lw=1.2,
        label="fit: market index + time trend + quarter-end ramp")
for m in [3, 6, 9]:
    ax.axvspan(pd.Timestamp(2025, m, 1), pd.Timestamp(2025, m, 1) + pd.offsets.MonthEnd(1), color="grey", alpha=0.15)
ax.set_ylabel("% vs average")
ax.set_title("Daily rate level: weekly market cycle + upward drift + quarter-end ramps (shaded)", loc="left")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "eda_daily_level.png")

# Plot: market index over the whole year (train + validation) - the input we have for Nov/Dec.
mi_all = pd.concat([train, valid]).groupby("date").market_index.mean()
fig, ax = plt.subplots(figsize=(11, 3), dpi=150)
ax.plot(mi_all.index, mi_all.values, color="#064A56", lw=1)
ax.axvline(pd.Timestamp("2025-11-01"), color="red", ls="--", lw=1)
ax.text(pd.Timestamp("2025-11-03"), mi_all.max(), "validation period", color="red", va="top", fontsize=8)
ax.set_title("Daily market_index (known for every day, including Nov-Dec)", loc="left")
fig.tight_layout()
fig.savefig(OUT / "eda_market_index.png")
print("Saved outputs/eda_daily_level.png and outputs/eda_market_index.png")
