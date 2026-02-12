import argparse
from pathlib import Path

import joblib

def _resolve_input_path(path_value: str, filename: str) -> Path:
    # AzureML pipeline inputs can be a folder or a file
    path = Path(path_value)
    if path.suffix:
        return path
    return path / filename


def main(input_path, model_path):
    # Dummy test step for pipeline validation
    data_file = _resolve_input_path(input_path, "prepared_data.pkl")
    model_file = _resolve_input_path(model_path, "model.pkl")

    prep_payload = joblib.load(data_file)
    model_payload = joblib.load(model_file)

    print(f"Test step received prep payload: {prep_payload}")
    print(f"Test step received model payload: {model_payload}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args.input, args.model_path)