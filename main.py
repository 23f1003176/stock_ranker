import pandas as pd
from scripts.fetch_data import fetch_all_data
from scripts.feature_engineering import build_feature_set
from scripts.train_models import train_weekly_model
from scripts.predict_rank import predict_and_rank
from scripts.evaluate_accuracy import evaluate_latest_week

def main():
    print("\n--- 🧠 STOCK RANKER MENU ---")
    print("1️⃣  Train and predict this week's returns")
    print("2️⃣  Evaluate last week's predictions")
    choice = input("\nEnter your choice (1 or 2): ").strip()

    if choice == "1":
        # --- Step 1: Load Nifty MidSmall 400 stock list ---
        df_symbols = pd.read_csv('data/niftymidsml400.csv')

        # Handle various column names
        if 'Symbol' in df_symbols.columns:
            symbols = [s + '.NS' for s in df_symbols['Symbol'].tolist()]
        elif 'Ticker' in df_symbols.columns:
            symbols = [s + '.NS' for s in df_symbols['Ticker'].tolist()]
        else:
            symbols = [s.strip() + '.NS' for s in df_symbols.iloc[:, 0].tolist()]

        print(f"\n✅ Loaded {len(symbols)} tickers from index file.")

        # --- Step 2: Fetch historical stock data ---
        fetch_all_data(symbols)

        # --- Step 3: Generate features ---
        build_feature_set(symbols)

        # --- Step 4: Train 1-week prediction model ---
        train_weekly_model()

        # --- Step 5: Predict & rank top stocks ---
        predict_and_rank(top_n=10)

    elif choice == "2":
        print("\n📈 Evaluating model performance for last week's predictions...\n")
        evaluate_latest_week()

    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
