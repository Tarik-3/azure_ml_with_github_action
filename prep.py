import argparse
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def main(output_path):
    # Load dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save prepared data
    joblib.dump((X_train, X_test, y_train, y_test), output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args.output)