import argparse
from pathlib import Path

import joblib

def _resolve_output_path(path_value: str, filename: str) -> Path:
    # AzureML pipeline outputs are directories; pick a file name inside
    path = Path(path_value)
    if path.suffix:
        return path
    return path / filename


def main(output_path, input_path=None):
    # Dummy prep step for pipeline validation
    print(f"Prep step received input: {input_path}")

    # Save a placeholder output for downstream steps
    output_file = _resolve_output_path(output_path, "prepared_data.pkl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"status": "ok", "message": "prep complete"}, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--input", type=str, required=False)
    args = parser.parse_args()
    main(args.output, args.input)