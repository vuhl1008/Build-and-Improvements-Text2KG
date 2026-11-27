import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare import load_raw_splits, prepare_webnlg_dataset, save_prepared_dataset
from src.utils.config import load_config

def main():
    config = load_config(PROJECT_ROOT/"configs/fine_tuning.yaml")
    raw_dataset = load_raw_splits(config["dataset"]["raw_dir"])
    cleaned_dataset = prepare_webnlg_dataset(raw_dataset)
    save_prepared_dataset(cleaned_dataset, config["dataset"]["processed_dir"])

if __name__ == "__main__":
    main()