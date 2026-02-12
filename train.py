import argparse
from pathlib import Path

import joblib

def _resolve_input_path(path_value: str, filename: str) -> Path:
    # AzureML pipeline inputs can be a folder or a file
    path = Path(path_value)
    if path.suffix:
        return path
    return path / filename


def _resolve_output_path(path_value: str, filename: str) -> Path:
    # AzureML pipeline outputs are directories; pick a file name inside
    path = Path(path_value)
    if path.suffix:
        return path
    return path / filename


def main(input_path, model_output):
    # Dummy train step for pipeline validation
    print(f"Train step received input: {input_path}")

    # Save a placeholder model for the test step
    output_file = _resolve_output_path(model_output, "model.pkl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"status": "ok", "message": "train complete"}, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()
    main(args.input, args.model_output)