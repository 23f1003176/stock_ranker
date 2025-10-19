import yfinance as yf
import pandas as pd
import os

def fetch_all_data(symbols, start='2020-01-01', end='2025-01-01'):
    os.makedirs('data/historical', exist_ok=True)
    data_dict = {}

    for sym in symbols:
        print(f"Fetching {sym}...")
        try:
            df = yf.download(sym, start=start, end=end)
            if df.empty:
                print(f"⚠️ No data for {sym}, skipping.")
                continue
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.to_csv(f'data/historical/{sym}.csv')
            data_dict[sym] = df
        except Exception as e:
            print(f"❌ Error fetching {sym}: {e}")
            continue

    return data_dict
