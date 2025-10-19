import os
import glob
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
import numpy as np

def evaluate_latest_week():
    # 🔍 find latest prediction file
    files = sorted(glob.glob("data/predictions_*_week.csv"))
    if not files:
        print("⚠️ No prediction files found.")
        return

    latest_file = files[-1]
    df_pred = pd.read_csv(latest_file)
    print(f"\n📅 Evaluating predictions from: {os.path.basename(latest_file)}")

    # Extract date from filename
    date_str = os.path.basename(latest_file).split('_')[1]
    prediction_date = datetime.strptime(date_str, "%Y-%m-%d")
    end_date = prediction_date + timedelta(days=7)

    actual_returns = []
    valid_stocks = []

    for _, row in df_pred.iterrows():
        sym = row["Stock"]
        try:
            data = yf.download(sym, start=prediction_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
            if len(data) < 2:
                continue
            start_price = data["Close"].iloc[0]
            end_price = data["Close"].iloc[-1]
            actual_return = (end_price - start_price) / start_price
            actual_returns.append(actual_return)
            valid_stocks.append(sym)
        except Exception:
            continue

    # Only keep valid stocks that we got data for
    df_actual = pd.DataFrame({
        "Stock": valid_stocks,
        "Actual_1W": actual_returns
    })

    # Merge with predictions
    df_merged = pd.merge(df_pred, df_actual, on="Stock", how="inner")

    # Compute error metrics
    mae = mean_absolute_error(df_merged["Actual_1W"], df_merged["Pred_1W"])
    corr = np.corrcoef(df_merged["Actual_1W"], df_merged["Pred_1W"])[0, 1]

    print(f"\n📊 Evaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Correlation (Pred vs Actual): {corr:.3f}")
    print(f"Compared {len(df_merged)} stocks.")

    # Top & worst performers
    df_merged["Diff"] = df_merged["Actual_1W"] - df_merged["Pred_1W"]
    print("\n✅ Top 5 Most Accurate Predictions:")
    print(df_merged.reindex(df_merged["Diff"].abs().sort_values().index).head(5))

    print("\n❌ 5 Largest Misses:")
    print(df_merged.reindex(df_merged["Diff"].abs().sort_values(ascending=False).index).head(5))

    df_merged.to_csv("data/evaluation_latest_week.csv", index=False)
    print("\n📁 Saved detailed evaluation to data/evaluation_latest_week.csv")
