"""
split_data.py

Splits a customer churn CSV into train (70%), validation (15%), and test (15%)
sets, stratified on the 'Exited' column so all three files keep the same
churn/stay ratio. Also label-encodes Gender so the output is ready to feed
straight into XGBoost or FT-Transformer.

USAGE (Colab or local):
    python split_data.py --input Customer-Churn-Records.csv --output_dir ./splits

Then you'll have:
    ./splits/train.csv
    ./splits/val.csv
    ./splits/test.csv
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# The columns you decided to keep. Add "NumOfProducts" and/or "Complain"
# to this list if you choose to include them later.
FEATURE_COLUMNS = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
]
TARGET_COLUMN = "Exited"
RANDOM_STATE = 42


def main(input_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"These required columns are missing from the file: {missing}")

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    # Encode Gender: Female=0, Male=1
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
    if df["Gender"].isnull().any():
        raise ValueError("Found a Gender value that wasn't 'Female' or 'Male' — check the raw data.")

    print(f"Loaded {len(df)} rows.")
    print("Class balance (Exited):")
    print(df[TARGET_COLUMN].value_counts(normalize=True).rename("proportion"))

    # Step 1: split off 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    # Step 2: split the remaining 30% into 15% val, 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSplit sizes:")
    print(f"  train: {len(train_df)} rows -> {train_path}")
    print(f"  val:   {len(val_df)} rows -> {val_path}")
    print(f"  test:  {len(test_df)} rows -> {test_path}")

    print("\nChurn ratio check (should all be close to each other):")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rate = d[TARGET_COLUMN].mean()
        print(f"  {name}: {rate:.3f} churn rate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the full Customer-Churn-Records.csv")
    parser.add_argument("--output_dir", default="./splits", help="Where to write train/val/test csv files")
    args = parser.parse_args()
    main(args.input, args.output_dir)