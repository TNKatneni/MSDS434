import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

INPUT_CSV = "data/raw_housing_data.csv"  # Input dataset
OUTPUT_CSV = "data/sampled_data.csv"     # Processed dataset
NUM_ROWS = 10000

def main():
    print("📂 Reading large CSV file...")
    df = pd.read_csv(INPUT_CSV)

    print("🔍 Selecting relevant columns...")
    df = df[["price", "bed", "bath", "acre_lot", "house_size"]]

    df.dropna(inplace=True)

    # Remove outliers in price (top 1% of most expensive houses)
    upper_limit = df["price"].quantile(0.99)
    df = df[df["price"] < upper_limit]

    print("🛠️ Applying log transformation to price...")
    df["price"] = np.log1p(df["price"]) 

    # Remove outliers in acre_lot and house_size (top 1%)
    df = df[df["acre_lot"] < df["acre_lot"].quantile(0.99)]
    df = df[df["house_size"] < df["house_size"].quantile(0.99)]

    df["acre_lot"] = np.log1p(df["acre_lot"])

    # Normalize house_size
    print("📏 Normalizing house_size...")
    scaler = MinMaxScaler()
    df[["house_size"]] = scaler.fit_transform(df[["house_size"]])
    print(f"House Size after scaling: min={df['house_size'].min()}, max={df['house_size'].max()}")

    # Convert bed & bath to categorical bins
    print("🏠 Binning bed and bath features...")
    df["bed"] = pd.cut(df["bed"], bins=[0, 1, 2, 3, 4, 5, 10, 20], labels=[1, 2, 3, 4, 5, 6, 7])
    df["bath"] = pd.cut(df["bath"], bins=[0, 1, 2, 3, 4, 5, 10, 20], labels=[1, 2, 3, 4, 5, 6, 7])

    # Convert categorical bins to numeric values
    df["bed"] = df["bed"].astype(float)
    df["bath"] = df["bath"].astype(float)

    # Print unique values to verify bins
    print(f"Unique bed values: {df['bed'].unique()}")
    print(f"Unique bath values: {df['bath'].unique()}")

    # Randomly sample data
    print(f"📊 Sampling {NUM_ROWS} rows out of {len(df)} total...")
    df_sampled = df.sample(n=min(NUM_ROWS, len(df)), random_state=42)

    # Save cleaned dataset
    df_sampled.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Processed data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
