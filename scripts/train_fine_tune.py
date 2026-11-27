import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pretrained.train import train
from src.utils.config import load_config


def main():
    config = load_config(PROJECT_ROOT / "configs/fine_tuning.yaml")
    train(config)


if __name__ == "__main__":
    main()
