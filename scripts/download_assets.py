# scripts/download_assets.py
from src.data.loading import download_dataset
from src.models.pretrained.loading import download_model
from src.utils.config import load_config


def main():
    config = load_config("configs/download.yaml")

    model_path = download_model(config["model"])
    dataset_path = download_dataset(config["dataset"])

    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")


if __name__ == "__main__":
    main()