import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loading import download_dataset
from src.models.pretrained.loading import download_model
from src.utils.config import load_config


def main():
    config = load_config(PROJECT_ROOT / "configs/download.yaml")

    model_path = download_model(config["model"])
    dataset_path = download_dataset(config["dataset"])

    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")


if __name__ == "__main__":
    main()
