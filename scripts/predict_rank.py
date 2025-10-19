import os
import pandas as pd
import numpy as np
import joblib
from datetime import date

def predict_and_rank(top_n=10):
    print("\n📈 Running weekly predictions...")

    model_path = "models/model_week.pkl"
    scaler_path = "models/scaler_week.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Train first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    results = []
    features = [
        'Return', 'LogReturn', 'MA_10', 'MA_50', 'MA_Ratio',
        'Volatility_10', 'Volatility_50', 'Vol_Ratio',
        'Volume_Change', 'Volume_Surge', 'GapUp', 'IntraVol',
        'Momentum_5', 'RSI_14', 'MACD', 'MACD_Signal',
        'EMA_10', 'EMA_50', 'ATR_14', 'OBV', 'BB_Position'
    ]

    feature_dir = "data/features"
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith(".csv")]

    for f in feature_files:
        sym = f.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(feature_dir, f))
            if len(df) < 30:
                print(f"⚠️ Not enough data for {sym}, skipping.")
                continue

            latest = df[features].tail(1).values
            scaled = scaler.transform(latest)
            pred = model.predict(scaled)[0]

            # --- Nonlinear stretch to emphasize breakout predictions ---
            pred_adjusted = np.tanh(pred * 1.8)  # expand magnitude but keep stability
            last_close = float(df["Close"].iloc[-1]) if "Close" in df.columns else np.nan
            predicted_price = last_close * np.exp(pred_adjusted / 2.0)

            results.append({
                "Stock": sym,
                "Last_Close": last_close,
                "Pred_1W": pred_adjusted,
                "Predicted_Price_1W": predicted_price
            })

        except Exception as e:
            print(f"❌ Error processing {sym}: {e}")

    if not results:
        print("⚠️ No predictions generated.")
        return

    df_results = pd.DataFrame(results).dropna(subset=["Pred_1W"])
    df_results = df_results.sort_values(by="Pred_1W", ascending=False).reset_index(drop=True)

    print("\n🏆 Top stocks by predicted 1-Week Return:")
    print(df_results.head(top_n).to_string(index=False))

    out_path = f"data/predictions_{date.today()}_week.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\n📁 Saved ranked predictions to {out_path}\n")
