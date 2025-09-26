#!/usr/bin/env bash
set -e

TICKER=${1:-AAPL}
START=${2:-2018-01-01}
HORIZON=${3:-1}
OUTDIR=${4:-outputs}

python -m src.train --ticker "$TICKER" --start "$START" --horizon "$HORIZON" --outdir "$OUTDIR"
python -m src.infer --model "$OUTDIR/best_model.joblib" --ticker "$TICKER" --start "$START" --horizon "$HORIZON" --outdir "$OUTDIR"
