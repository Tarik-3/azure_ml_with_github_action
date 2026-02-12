"""
Prediction/Inference Step
- Loads a trained model
- Loads new data to make predictions on
- Generates predictions and saves them
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import joblib


def main(model_path: str, input_data_path: str, output_path: str) -> None:
    model_dir = Path(model_path)
    input_file = Path(input_data_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from: {model_dir}")
    model_file = model_dir / "model.pkl"
    model = joblib.load(model_file)
    print(f"Model loaded successfully")
    
    print(f"\nLoading new data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Make predictions
    print(f"\nGenerating predictions...")
    predictions = model.predict(df)
    print(f"Generated {len(predictions)} predictions")
    
    # Add predictions to dataframe
    df['prediction'] = predictions
    
    # Save predictions
    predictions_file = output_dir / "predictions.csv"
    df.to_csv(predictions_file, index=False)
    print(f"\n[SUCCESS] Predictions saved to: {predictions_file}")
    
    # Save summary statistics
    summary = {
        "n_predictions": len(predictions),
        "mean_prediction": float(predictions.mean()),
        "min_prediction": float(predictions.min()),
        "max_prediction": float(predictions.max()),
        "std_prediction": float(predictions.std())
    }
    
    summary_file = output_dir / "prediction_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    
    # Display sample predictions
    print(f"\n{'='*60}")
    print("Sample Predictions (first 5 rows):")
    print(f"{'='*60}")
    print(df.head().to_string())
    print(f"\n{'='*60}")
    print("Prediction Statistics:")
    print(f"{'='*60}")
    print(f"Count: {len(predictions)}")
    print(f"Mean:  {predictions.mean():.4f}")
    print(f"Std:   {predictions.std():.4f}")
    print(f"Min:   {predictions.min():.4f}")
    print(f"Max:   {predictions.max():.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--input", type=str, required=True, help="Path to new data CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()
    main(args.model, args.input, args.output)
