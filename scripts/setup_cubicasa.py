"""One-time setup for the learned parser: fetch CubiCasa5K's model code and pretrained weights.

    uv run --group ml python scripts/setup_cubicasa.py            # code + weights
    uv run --group ml python scripts/setup_cubicasa.py --dataset  # also the 5.5 GB dataset (training only)

Everything lands in git-ignored folders: vendor/ (their code, only needed to export the model),
models/ (weights) and datasets/. The code, weights and dataset are licensed CC BY-NC(-SA) 4.0 by
CubiCasa: fine for non-commercial use, never committed here. This script is the only place that
downloads anything; the parser itself runs offline.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "vendor" / "cubicasa5k"
MODELS_DIR = ROOT / "models"
DATASETS_DIR = ROOT / "datasets"

REPO_URL = "https://github.com/CubiCasa/CubiCasa5k.git"
REPO_COMMIT = "c34440266665a11f4484eb06cd2e4b7d72ad76c1"

WEIGHTS_FILE = MODELS_DIR / "cubicasa5k_weights.pkl"
WEIGHTS_DRIVE_ID = "1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK"
WEIGHTS_SHA256: str | None = "dd20b4e1bf1d670f2125107b079df06958b1ccd36e49a464ab739aeb00b8e7a2"
"""Recorded after the first download, so a changed or corrupted file is noticed."""

DATASET_FILE = DATASETS_DIR / "cubicasa5k.zip"
DATASET_URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_code() -> None:
    if not (VENDOR_DIR / ".git").is_dir():
        VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", REPO_URL, str(VENDOR_DIR)], check=True)
    subprocess.run(["git", "-C", str(VENDOR_DIR), "checkout", "--quiet", REPO_COMMIT], check=True)
    print(f"CubiCasa5k code at {REPO_COMMIT[:10]} in {VENDOR_DIR.relative_to(ROOT)}")


def fetch_weights() -> None:
    if not WEIGHTS_FILE.is_file():
        import gdown  # ml group only

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if gdown.download(id=WEIGHTS_DRIVE_ID, output=str(WEIGHTS_FILE), quiet=False) is None:
            sys.exit(
                "Google Drive refused the download (quota or confirmation page). Download it in a browser:\n"
                f"  https://drive.google.com/file/d/{WEIGHTS_DRIVE_ID}/view\n"
                f"and save it as {WEIGHTS_FILE}"
            )
    actual = sha256(WEIGHTS_FILE)
    if WEIGHTS_SHA256 is not None and actual != WEIGHTS_SHA256:
        sys.exit(f"{WEIGHTS_FILE.name} has SHA-256 {actual}, expected {WEIGHTS_SHA256}. Delete it and rerun.")
    print(f"Weights in {WEIGHTS_FILE.relative_to(ROOT)} (SHA-256 {actual})")


def fetch_dataset() -> None:
    if DATASET_FILE.is_file():
        print(f"Dataset already in {DATASET_FILE.relative_to(ROOT)}")
        return
    import requests  # ml group only (via gdown)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    partial = DATASET_FILE.with_suffix(".zip.part")
    with requests.get(DATASET_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1 << 20):
                file.write(chunk)
    partial.rename(DATASET_FILE)
    print(f"Dataset in {DATASET_FILE.relative_to(ROOT)}")


def main() -> None:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)
    parser.add_argument("--dataset", action="store_true", help="also fetch the 5.5 GB training dataset")
    args = parser.parse_args()
    fetch_code()
    fetch_weights()
    if args.dataset:
        fetch_dataset()


if __name__ == "__main__":
    main()
