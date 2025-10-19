import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


def train_weekly_model():
    print("\n🚀 Loading weekly feature data...")

    all_files = [f"data/features/{f}" for f in os.listdir("data/features") if f.endswith(".csv")]
    if not all_files:
        raise Exception("No feature files found.")

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            if "Target_1W" in df.columns:
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Could not read {f}: {e}")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"✅ Using {len(df_all)} total feature rows after cleaning.\n")

    features = [
        'Return', 'LogReturn', 'MA_10', 'MA_50', 'MA_Ratio',
        'Volatility_10', 'Volatility_50', 'Vol_Ratio',
        'Volume_Change', 'Volume_Surge', 'GapUp', 'IntraVol',
        'Momentum_5', 'RSI_14', 'MACD', 'MACD_Signal',
        'EMA_10', 'EMA_50', 'ATR_14', 'OBV', 'BB_Position'
    ]

    X = df_all[features].values
    y = df_all["Target_1W"].values

    # --- Split and scale ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Weight large returns more heavily ---
    weights = np.clip(np.abs(y_train) * 10, 1.0, 10.0)

    print("🚀 Training XGBoost model (volatility-aware weekly)...")

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        objective="reg:pseudohubererror",
        eval_metric="rmse",
    )

    model.fit(
        X_train_scaled, y_train,
        sample_weight=weights,
        eval_set=[(X_test_scaled, y_test)],
        verbose=50
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_week.pkl")
    joblib.dump(scaler, "models/scaler_week.pkl")

    print(f"✅ Trained 1-Week XGBoost model on {len(X_train)} samples.\n")
