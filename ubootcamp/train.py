import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from .data import download_prices, load_from_csv, add_features, make_supervised
from .model import build_models

def train(outdir: Path,
          horizon: int = 1,
          ticker: str | None = None,
          start: str = "2018-01-01",
          csv_path: str | None = None,
          date_col: str | None = None,
          close_col: str = "Close"):
    outdir.mkdir(parents=True, exist_ok=True)

    # Data source selection
    if csv_path:
        raw = load_from_csv(csv_path, date_col=date_col, close_col=close_col)
        src_name = f"csv:{csv_path}"
        ticker = ticker or "CSV"
    else:
        if not ticker:
            raise ValueError("Either provide --csv PATH or --ticker for yfinance.")
        raw = download_prices(ticker, start=start, end=None)
        src_name = f"yfinance:{ticker}"

    feat_df = add_features(raw)
    X, y, features, data = make_supervised(feat_df, horizon=horizon)

    # Split
    split = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test,  y_test  = X.iloc[split:], y.iloc[split:]

    # Baseline (naive)
    yhat_naive = X_test["lag1"].values
    mae_naive = mean_absolute_error(y_test, yhat_naive)

    # Models
    models = build_models()

    # Linear Regression
    models.lr.fit(X_train, y_train)
    yhat_lr = models.lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, yhat_lr)
    r2_lr  = r2_score(y_test, yhat_lr)

    # Random Forest
    models.rf.fit(X_train, y_train)
    yhat_rf = models.rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, yhat_rf)
    r2_rf  = r2_score(y_test, yhat_rf)

    # Choose best
    metrics = [
        {"model":"naive", "MAE": float(mae_naive), "R2": None},
        {"model":"LinearRegression", "MAE": float(mae_lr), "R2": float(r2_lr)},
        {"model":"RandomForest",    "MAE": float(mae_rf), "R2": float(r2_rf)},
    ]
    best = min([m for m in metrics if m["model"] != "naive"], key=lambda m: m["MAE"])

    # Save metrics with context
    out = {
        "source": src_name,
        "horizon": horizon,
        "metrics": metrics
    }
    (outdir / "metrics.json").write_text(json.dumps(out, indent=2))

    # Save best model
    best_model_name = best["model"]
    best_model = models.rf if best_model_name == "RandomForest" else models.lr
    joblib.dump(best_model, outdir / "best_model.joblib")

    # Plot last 200
    plot_df = pd.DataFrame({
        "y_true": y_test,
        "yhat_lr": yhat_lr,
        "yhat_rf": yhat_rf
    }, index=y_test.index).tail(200)

    plt.figure(figsize=(12,4))
    plot_df[["y_true","yhat_lr","yhat_rf"]].plot(figsize=(12,4))
    plt.title(f"{ticker or 'CSV'} – Son 200 gün: Gerçek vs Tahmin")
    plt.xlabel("Tarih"); plt.ylabel("Fiyat")
    plt.tight_layout()
    plt.savefig(outdir / "plot.png", dpi=150)
    plt.close()

    return best_model_name, outdir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--horizon", type=int, default=1)

    # yfinance option
    ap.add_argument("--ticker", type=str, default=None)
    ap.add_argument("--start", type=str, default="2018-01-01")

    # csv option
    ap.add_argument("--csv", dest="csv_path", type=str, default=None)
    ap.add_argument("--date_col", type=str, default=None)
    ap.add_argument("--close_col", type=str, default="Close")

    args = ap.parse_args()

    best_name, outdir = train(
        outdir=Path(args.outdir),
        horizon=args.horizon,
        ticker=args.ticker,
        start=args.start,
        csv_path=args.csv_path,
        date_col=args.date_col,
        close_col=args.close_col,
    )
    print("En iyi model:", best_name)

if __name__ == "__main__":
    main()
