import os
import pandas as pd
import numpy as np
import ta  # Technical Analysis library


def build_feature_set(symbols):
    os.makedirs('data/features', exist_ok=True)
    total_rows = 0
    processed_count = 0
    skipped_count = 0

    for sym in symbols:
        file_path = f"data/historical/{sym}.csv"
        if not os.path.exists(file_path):
            print(f"⚠️ No data for {sym}, skipping.")
            skipped_count += 1
            continue

        try:
            df = pd.read_csv(file_path)

            # --- Ensure numeric and clean data ---
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['Close', 'Volume']).copy()

            # --- Base Returns ---
            df['Return'] = df['Close'].pct_change(fill_method=None)
            df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))

            # --- Moving Averages & Ratios ---
            df['MA_10'] = df['Close'].rolling(10).mean()
            df['MA_50'] = df['Close'].rolling(50).mean()
            df['MA_Ratio'] = df['MA_10'] / df['MA_50']

            # --- Volatility Features ---
            df['Volatility_10'] = df['Return'].rolling(10).std()
            df['Volatility_50'] = df['Return'].rolling(50).std()
            df['Vol_Ratio'] = df['Volatility_10'] / df['Volatility_50']

            # --- Volume Behavior ---
            df['Volume_Change'] = df['Volume'].pct_change(fill_method=None)
            df['Volume_Surge'] = df['Volume'] / df['Volume'].rolling(20).mean()

            # --- Price Action Indicators ---
            df['GapUp'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            df['IntraVol'] = (df['High'] - df['Low']) / df['Close']
            df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1

            # --- RSI & MACD ---
            df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()

            # --- Exponential MAs ---
            df['EMA_10'] = ta.trend.EMAIndicator(df['Close'], window=10).ema_indicator()
            df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()

            # --- Average True Range (volatility intensity) ---
            df['ATR_14'] = ta.volatility.AverageTrueRange(
                df['High'], df['Low'], df['Close'], window=14
            ).average_true_range()

            # --- On-Balance Volume (cumulative momentum) ---
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

            # --- Bollinger Band Position (breakout signal) ---
            bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
            df['BB_Position'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])

            # --- Targets (log returns) ---
            df['Target_1D'] = np.log(df['Close'].shift(-1) / df['Close'])
            df['Target_1W'] = np.log(df['Close'].shift(-5) / df['Close']) * 2.0  # rescaled for large moves
            df['Target_1M'] = np.log(df['Close'].shift(-21) / df['Close'])

            df = df.dropna().copy()

            # --- Feature columns ---
            features = [
                'Return', 'LogReturn', 'MA_10', 'MA_50', 'MA_Ratio',
                'Volatility_10', 'Volatility_50', 'Vol_Ratio',
                'Volume_Change', 'Volume_Surge', 'GapUp', 'IntraVol',
                'Momentum_5', 'RSI_14', 'MACD', 'MACD_Signal',
                'EMA_10', 'EMA_50', 'ATR_14', 'OBV', 'BB_Position'
            ]

            if len(df) < 100:
                print(f"⚠️ Not enough data for {sym}, skipping.")
                skipped_count += 1
                continue

            # --- Fix for missing Date column ---
            if 'Date' not in df.columns:
                df.reset_index(inplace=True)
                if 'index' in df.columns:
                    df.rename(columns={'index': 'Date'}, inplace=True)

            # --- Save feature file safely ---
            feature_path = f"data/features/{sym}.csv"
            cols_to_save = ['Date'] + features + ['Target_1D', 'Target_1W', 'Target_1M']
            df_to_save = (
                df[cols_to_save]
                if all(col in df.columns for col in cols_to_save)
                else df[features + ['Target_1D', 'Target_1W', 'Target_1M']]
            )
            df_to_save.to_csv(feature_path, index=False)

            total_rows += len(df)
            processed_count += 1
            print(f"✅ Processed {sym}, {len(df)} rows.")

        except Exception as e:
            print(f"❌ Error processing {sym}: {e}")
            skipped_count += 1

    print(f"\n✅ Finished feature generation.")
    print(f"   • Processed: {processed_count} symbols")
    print(f"   • Skipped:   {skipped_count} symbols")
    print(f"   • Total rows: {total_rows}\n")
