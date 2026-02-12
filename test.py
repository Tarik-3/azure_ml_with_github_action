"""
Model Testing Step
- Loads test data from prep step
- Loads trained model from train step
- Evaluates model performance on test set
- Prints metrics (MSE, R², MAE)
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


def main(input_path: str, model_path: str) -> None:
    input_dir = Path(input_path)
    model_dir = Path(model_path)
    
    print(f"Loading test data from: {input_dir}")
    print(f"Loading model from: {model_dir}")
    
    # Load test data
    X_test = pd.read_csv(input_dir / "X_test.csv")
    y_test = pd.read_csv(input_dir / "y_test.csv", header=None).values.ravel()
    
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Load trained model
    model_file = model_dir / "model.pkl"
    model = joblib.load(model_file)
    print(f"Model loaded from: {model_file}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print("\n" + "=" * 60)
    print("Model Evaluation Results (on Test Set)")
    print("=" * 60)
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("=" * 60)
    
    # Save metrics to file
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2),
        "n_test_samples": len(y_test)
    }
    metrics_file = Path("outputs/metrics.json")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SUCCESS] Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test/evaluate ML model")
    parser.add_argument("--input", type=str, required=True, help="Path to prepared test data directory")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model directory")
    args = parser.parse_args()
    main(args.input, args.model)