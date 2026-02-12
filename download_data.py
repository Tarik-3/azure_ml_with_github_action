import argparse
import os
import shutil
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


def _resolve_output_path(path_value: str, filename: str) -> Path:
    # AzureML pipeline outputs are directories; pick a file name inside
    path = Path(path_value)
    if path.suffix:
        return path
    return path / filename


def _build_raw_url(repo_url: str, ref: str, file_path: str) -> str:
    # Convert repo URL and file path into a raw.githubusercontent.com URL
    cleaned = repo_url.rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    if cleaned.startswith("https://github.com/"):
        cleaned = cleaned[len("https://github.com/"):]
    if cleaned.startswith("http://github.com/"):
        cleaned = cleaned[len("http://github.com/"):]
    encoded_path = quote(file_path, safe="/")
    return f"https://raw.githubusercontent.com/{cleaned}/{ref}/{encoded_path}"


def main(repo_url: str, ref: str, file_path: str, output_path: str) -> None:
    output_file = _resolve_output_path(output_path, Path(file_path).name)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if repo_url.lower() == "local":
        # Copy a local file into the pipeline output directory
        source_file = Path(__file__).resolve().parent / file_path
        if not source_file.exists():
            raise FileNotFoundError(f"Local source file not found: {source_file}")
        shutil.copyfile(source_file, output_file)
        print(f"Copied local file {source_file} to {output_file}")
        return

    raw_url = _build_raw_url(repo_url, ref, file_path)
    request = Request(raw_url)

    # Optional auth for private repos
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"token {token}")

    with urlopen(request) as response:
        payload = response.read()
        if response.status >= 400:
            raise RuntimeError(f"Download failed with status {response.status}")
        with output_file.open("wb") as handle:
            handle.write(payload)

    print(f"Downloaded {raw_url} to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.repo, args.ref, args.path, args.output)
