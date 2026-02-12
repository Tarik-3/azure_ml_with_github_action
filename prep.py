"""
Data Preparation Step
- Loads raw CSV data
- Handles missing values
- Splits into train/test sets
- Saves to output directory
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {input_file}")
    
    # Load CSV data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Handle missing values (simple strategy: drop rows with any NaN)
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing values")
    
    # Assume last column is target
    if len(df.columns) > 1:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        raise ValueError(f"Expected at least 2 columns, got {len(df.columns)}")
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Save processed data
    train_file = output_dir / "X_train.csv"
    test_file = output_dir / "X_test.csv"
    train_target_file = output_dir / "y_train.csv"
    test_target_file = output_dir / "y_test.csv"
    
    X_train.to_csv(train_file, index=False)
    X_test.to_csv(test_file, index=False)
    y_train.to_csv(train_target_file, index=False, header=False)
    y_test.to_csv(test_target_file, index=False, header=False)
    
    print(f"\n[SUCCESS] Prepared data saved to: {output_dir}")
    print(f"   - {train_file.name}")
    print(f"   - {test_file.name}")
    print(f"   - {train_target_file.name}")
    print(f"   - {test_target_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()
    main(args.input, args.output)