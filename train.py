"""
Model Training Step
- Loads prepared training data
- Trains a linear regression model
- Saves trained model to output directory
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def main(input_path: str, output_path: str) -> None:
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading training data from: {input_dir}")
    
    # Load training data from prep step
    X_train = pd.read_csv(input_dir / "X_train.csv")
    y_train = pd.read_csv(input_dir / "y_train.csv", header=None).values.ravel()
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # Train linear regression model
    print("\nTraining LinearRegression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Get model info
    train_score = model.score(X_train, y_train)
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Model coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_}")
    
    # Save model
    model_file = output_dir / "model.pkl"
    joblib.dump(model, model_file)
    print(f"\n[SUCCESS] Model saved to: {model_file}")
    
    # Save metadata
    metadata = {
        "model_type": "LinearRegression",
        "train_score": float(train_score),
        "n_features": len(model.coef_),
        "n_training_samples": len(X_train)
    }
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--input", type=str, required=True, help="Path to prepared data directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()
    main(args.input, args.output)