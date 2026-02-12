import argparse
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--output")
args = parser.parse_args()

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(X, y)

joblib.dump(model, f"{args.output}/model.pkl")