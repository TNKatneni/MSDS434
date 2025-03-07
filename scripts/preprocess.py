"""
scripts/preprocess.py
---------------------
Usage:
    python preprocess.py

Ensure you have 'pandas' and 'scikit-learn' installed.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

INPUT_CSV = "data/raw_housing_data.csv"  # Input dataset
OUTPUT_CSV = "data/sampled_data.csv"     # Processed dataset
NUM_ROWS = 10000

def main():
    print("📂 Reading large CSV file...")
    df = pd.read_csv(INPUT_CSV)

    # Select only relevant features
    print("🔍 Selecting relevant columns...")
    df = df[["price", "bed", "bath", "acre_lot", "house_size"]]

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Remove outliers in price (top 1% of most expensive houses)
    upper_limit = df["price"].quantile(0.99)
    df = df[df["price"] < upper_limit]

    # Log transform the price column
    print("🛠️ Applying log transformation to price...")
    df["price"] = np.log1p(df["price"])  # log(1 + price) transformation

    # Normalize acre_lot and house_size
    print("📏 Normalizing acre_lot and house_size...")
    scaler = MinMaxScaler()
    df[["acre_lot", "house_size"]] = scaler.fit_transform(df[["acre_lot", "house_size"]])

    # Randomly sample data
    print(f"📊 Sampling {NUM_ROWS} rows out of {len(df)} total...")
    df_sampled = df.sample(n=min(NUM_ROWS, len(df)), random_state=42)

    # Save the cleaned dataset
    df_sampled.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Processed data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
