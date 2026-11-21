"""
Freight rate model: train, backtest, and predict.

Model = (1) a small linear model for market/time effects  +  (2) XGBoost for load/lane effects.

  log(rate) = Linear(equipment, log distance, market index, time trend, quarter-end ramp)
            + XGBoost(equipment, distance, coordinates, weight, quote signal)

Why two parts?  The validation loads are in Nov-Dec, AFTER all training data (Jan-Oct).
Tree models cannot extrapolate a trend over time, so everything that depends on the
date (market level, upward drift, quarter-end ramp) goes into the linear part, which
extrapolates naturally. The trees only learn things that do not change with time.

Usage:  python train.py [--device cuda|cpu]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge

DATA = Path("data")
OUT = Path("outputs")
START = pd.Timestamp("2025-01-01")
BROKEN_PRICE_LOG_GAP = 0.5  # price > e^0.5 = 1.65x away from the simple distance formula = corrupted label

# Linear part: market + time effects (these must extrapolate into Nov/Dec).
LINEAR_FEATURES = [
    "is_reefer", "is_flatbed", "log_distance",
    "log_mi_day",           # market index of the day (daily average)
    "log_mi_dev",           # this load's market index vs the daily average
    "t",                    # years since 2025-01-01 -> steady upward drift
    "qramp",                # 0 -> 1 across the last month of each quarter (Mar, Jun, Sep, Dec)
    "qramp_reefer", "qramp_flatbed",  # the quarter-end ramp is stronger for these
]
# Tree part: load / lane effects only (no date or market features -> no extrapolation issues).
TREE_FEATURES = [
    "is_dryvan", "is_reefer", "is_flatbed", "distance", "log_distance", "straight_line_miles",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "weight", "quote_signal",
]
XGB_PARAMS = dict(n_estimators=3000, learning_rate=0.02, max_depth=7, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, tree_method="hist")


# ----------------------------------------------------------------------------- features
def daily_market_index(*frames):
    """Average market_index per day, using every file that has it (inputs only, no labels)."""
    allrows = pd.concat([f[["date", "market_index"]] for f in frames])
    return allrows.groupby(pd.to_datetime(allrows["date"]))["market_index"].mean()


def add_features(df, mi_by_day, city_coords, default_quote_signal):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])

    # --- cleaning ---
    d["weight"] = d["weight"].abs()  # negative weights are sign errors; NaN is handled by XGBoost
    d["mi_day"] = d["date"].map(mi_by_day)
    if "market_index" not in d:  # December chart rows: use the day's market level
        d["market_index"] = np.nan
    d["market_index"] = d["market_index"].fillna(d["mi_day"])
    if "quote_signal" not in d:
        d["quote_signal"] = default_quote_signal
    for side in ["pickup", "delivery"]:  # December chart rows have no coordinates: look up by city
        if f"{side}_lat" not in d:
            d[f"{side}_lat"] = d[side].map(city_coords["lat"])
            d[f"{side}_lon"] = d[side].map(city_coords["lon"])

    # --- load features ---
    d["is_dryvan"] = (d["equipment"] == "Dry Van").astype(float)
    d["is_reefer"] = (d["equipment"] == "Reefer").astype(float)
    d["is_flatbed"] = (d["equipment"] == "Flatbed").astype(float)
    d["log_distance"] = np.log(d["distance"])
    d["straight_line_miles"] = haversine_miles(d)

    # --- market / time features ---
    d["log_mi_day"] = np.log(d["mi_day"])
    d["log_mi_dev"] = np.log(d["market_index"]) - d["log_mi_day"]
    d["t"] = (d["date"] - START).dt.days / 365
    is_quarter_end_month = d["date"].dt.month % 3 == 0
    d["qramp"] = is_quarter_end_month * (d["date"].dt.day - 1) / (d["date"].dt.days_in_month - 1)
    d["qramp_reefer"] = d["qramp"] * d["is_reefer"]
    d["qramp_flatbed"] = d["qramp"] * d["is_flatbed"]
    return d


def flag_broken_prices(d):
    """
    ~1.3% of prices are data errors (about x4 too high or x0.3 too low). Compare each price with a
    simple formula (truck type + distance); anything more than 1.65x away is flagged.
    A simple formula is used on purpose: a powerful model (XGBoost) would partly memorise the
    broken prices and then fail to flag them.
    """
    X = d[["is_reefer", "is_flatbed", "log_distance"]]
    y = np.log(d["posted_rate"])
    residual = y - LinearRegression().fit(X, y).predict(X)
    return (np.abs(residual - np.median(residual)) > BROKEN_PRICE_LOG_GAP).values


def haversine_miles(d):
    lat1, lon1, lat2, lon2 = (np.radians(d[c]) for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


def city_coordinates(*frames):
    rows = []
    for f in frames:
        for side in ["pickup", "delivery"]:
            rows.append(f[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(rows).groupby("city").first()


# ----------------------------------------------------------------------------- model
class FreightModel:
    def __init__(self, device):
        self.device = device

    def fit(self, d):
        d = d[~d["broken_price"]]  # never learn from corrupted labels
        y = np.log(d["posted_rate"])
        # Step 1: linear model learns the market/time effects.
        self.linear = Ridge(alpha=1e-3).fit(d[LINEAR_FEATURES], y)
        # Step 2: XGBoost learns what the linear model got wrong, from load features only.
        residual = y - self.linear.predict(d[LINEAR_FEATURES])
        self.trees = xgb.XGBRegressor(**XGB_PARAMS, device=self.device).fit(d[TREE_FEATURES], residual)
        return self

    def predict_log(self, d):
        return self.linear.predict(d[LINEAR_FEATURES]) + self.trees.predict(d[TREE_FEATURES])

    def predict(self, d):
        return np.exp(self.predict_log(d))


def metrics(actual, predicted, broken):
    a, p = actual[~broken], predicted[~broken]  # score on real prices only (broken ones cannot be predicted)
    return {
        "MAE": np.mean(np.abs(p - a)),
        "MAPE_%": 100 * np.mean(np.abs(p / a - 1)),
        "RMSE": np.sqrt(np.mean((p - a) ** 2)),
        "bias_%": 100 * np.mean(np.log(p / a)),
        "MAE_incl_bad_labels": np.mean(np.abs(predicted - actual)),
        "n_test": len(actual),
        "n_bad_labels": int(broken.sum()),
    }


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()
    if args.device == "cuda":
        import warnings
        warnings.filterwarnings("error", message=".*No visible GPU.*")  # fail loudly instead of silently using CPU
        warnings.filterwarnings("ignore", message=".*mismatched devices.*")  # harmless: pandas input lives on CPU
    OUT.mkdir(exist_ok=True)

    train_raw = pd.read_csv(DATA / "train_test.csv")
    valid_raw = pd.read_csv(DATA / "validation.csv")
    december_raw = pd.read_csv(DATA / "december_chart_inputs.csv")

    mi_by_day = daily_market_index(train_raw, valid_raw)
    coords = city_coordinates(train_raw, valid_raw)
    default_qs = train_raw["quote_signal"].median()
    feats = lambda df: add_features(df, mi_by_day, coords, default_qs)
    train, valid, december = feats(train_raw), feats(valid_raw), feats(december_raw)
    train["broken_price"] = flag_broken_prices(train)

    # 1) Backtest: always train on the past, test on the next 2 months (same as Jan-Oct -> Nov-Dec).
    folds = [("2025-05-01", "2025-06-30"), ("2025-07-01", "2025-08-31"), ("2025-09-01", "2025-10-31")]
    rows, fold_predictions = [], []
    for start, end in folds:
        past = train[train["date"] < start]
        future = train[(train["date"] >= start) & (train["date"] <= end)]
        model = FreightModel(args.device).fit(past)
        predicted = model.predict(future)
        m = metrics(future["posted_rate"].values, predicted, future["broken_price"].values)
        rows.append({"train": f"Jan 1 - {pd.Timestamp(start) - pd.Timedelta(days=1):%b %d}",
                     "test": f"{start} .. {end}", **m})
        keep = ["load_id", "date", "equipment", "distance", "posted_rate", "broken_price"]
        fold_predictions.append(future[keep].assign(fold=f"{start}..{end}", predicted_rate=predicted))
    pd.concat(fold_predictions).to_csv(OUT / "backtest_predictions.csv", index=False)  # used by make_report.py
    backtest = pd.DataFrame(rows)
    average = {"train": "AVERAGE", "test": "", **backtest.mean(numeric_only=True)}
    backtest = pd.concat([backtest, pd.DataFrame([average])], ignore_index=True)
    backtest.to_csv(OUT / "backtest_results.csv", index=False)
    print("\nTime-based backtest (errors exclude corrupted labels):")
    print(backtest.round(2).to_string(index=False))

    # 2) Final model on all labelled data (Jan-Oct).
    model = FreightModel(args.device).fit(train)
    print(f"\nFinal model: removed {train['broken_price'].sum()} corrupted labels out of {len(train)}")
    coefs = pd.Series(model.linear.coef_, index=LINEAR_FEATURES).round(4)
    print("Linear (market/time) coefficients on log(rate):\n" + coefs.to_string())
    coefs.to_csv(OUT / "linear_coefficients.csv")

    # 3) Predictions for the 12,000 validation loads.
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    pred = pd.Series(model.predict(valid), index=valid["load_id"])
    template["predicted_rate"] = template["load_id"].map(pred).round(2)
    assert template["predicted_rate"].notna().all() and len(template) == 12_000
    template[["load_id", "predicted_rate"]].to_csv("validation_predictions.csv", index=False)

    # 4) December chart inputs (fill predicted_rate, keep the original columns).
    december_raw["predicted_rate"] = model.predict(december).round(2)
    december_raw.to_csv(DATA / "december_chart_inputs.csv", index=False)

    print("\nSaved validation_predictions.csv and data/december_chart_inputs.csv")
    print(december_raw[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
