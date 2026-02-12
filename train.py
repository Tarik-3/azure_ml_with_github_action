import argparse
import joblib
from sklearn.linear_model import LinearRegression

def main(input_path, model_output):
    # Load prepared data
    X_train, X_test, y_train, y_test = joblib.load(input_path)

    # Train simple linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()
    main(args.input, args.model_output)