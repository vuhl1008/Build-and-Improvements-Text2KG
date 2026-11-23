import json
from pathlib import Path
from datasets import load_dataset as hf_load_dataset, Dataset, DatasetDict

#Tải dataset từ HF về local
def download_dataset(config):
    dataset_id = config['id']
    dataset_cache = config['cache_dir']
    revision  = config['revision']
    subset = config['subset']

    base = f"hf://datasets/{dataset_id}@{revision}/{subset}"

    data_files = {
        split: f"{base}/{split}/*.{config['format']}"
        for split in config['splits']
    }

    source_files = hf_load_dataset(
        config['format'],
        data_files = data_files,
        cache_dir = dataset_cache
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, dataset in source_files.items():
        path = output_dir / f"{split}.jsonl"
        dataset.to_json(str(path))
        print(f"Saved {len(dataset)} rows to {path}")

    return source_files
