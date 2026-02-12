import argparse
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def main(input_path, model_path):
    # Load data and model
    X_train, X_test, y_train, y_test = joblib.load(input_path)
    model = joblib.load(model_path)

    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Test MSE: {mse}")
    print(f"Test R2: {r2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args.input, args.model_path)