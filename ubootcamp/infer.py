import argparse, json
from pathlib import Path
import joblib
import pandas as pd

from .data import download_prices, load_from_csv, add_features, make_supervised

def infer(model_path: Path,
          outdir: Path,
          horizon: int = 1,
          ticker: str | None = None,
          start: str = "2018-01-01",
          csv_path: str | None = None,
          date_col: str | None = None,
          close_col: str = "Close"):
    outdir.mkdir(parents=True, exist_ok=True)

    if csv_path:
        raw = load_from_csv(csv_path, date_col=date_col, close_col=close_col)
        src_name = f"csv:{csv_path}"
    else:
        if not ticker:
            raise ValueError("Either provide --csv PATH or --ticker for yfinance.")
        raw = download_prices(ticker, start=start, end=None)
        src_name = f"yfinance:{ticker}"

    feat_df = add_features(raw)
    X, y, features, data = make_supervised(feat_df, horizon=horizon)

    last = data.iloc[-1]
    x_pred = pd.DataFrame([{f: last[f] for f in features}])

    model = joblib.load(model_path)
    next_pred = float(model.predict(x_pred)[0])
    out = {
        "source": src_name,
        "last_close": float(last["Close"]),
        "next_day_prediction": next_pred,
        "horizon": horizon
    }
    (outdir / "next_day_prediction.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="outputs/best_model.joblib")
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

    infer(
        model_path=Path(args.model),
        outdir=Path(args.outdir),
        horizon=args.horizon,
        ticker=args.ticker,
        start=args.start,
        csv_path=args.csv_path,
        date_col=args.date_col,
        close_col=args.close_col,
    )

if __name__ == "__main__":
    main()
