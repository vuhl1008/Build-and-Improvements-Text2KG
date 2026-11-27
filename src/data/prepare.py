import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from src.data.cleaning import clean_text


def load_raw_splits(raw_dir, splits=("train", "validation", "test")):
    """Load only fields used for preparation, ignoring inconsistent metadata."""
    raw_dir = Path(raw_dir)
    datasets = {}
    for split in splits:
        path = raw_dir / f"{split}.jsonl"
        rows = []
        with path.open(encoding="utf-8") as raw_file:
            for line_number, line in enumerate(raw_file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = {"input", "target"}.difference(row)
                if missing:
                    raise ValueError(
                        f"{path}:{line_number} is missing required fields: {sorted(missing)}"
                    )
                rows.append({"input": row["input"], "target": row["target"]})
        datasets[split] = Dataset.from_list(rows)
    return DatasetDict(datasets)


def prepare_webnlg_dataset(source):
    """Normalize WebNLG input text and reference triples for shared use."""
    prepared = DatasetDict()
    for split, dataset in source.items():

        def convert(example):
            triples = example["input"]
            if not isinstance(triples, list):
                triples = [triples]
            normalized_triples = [clean_text(triple) for triple in triples]
            return {
                "input": clean_text(example["target"]),
                "target": " ; ".join(triple for triple in normalized_triples if triple),
            }

        prepared[split] = dataset.map(convert, remove_columns=dataset.column_names)
    return prepared


def save_prepared_dataset(dataset, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in dataset.items():
        rows.to_json(str(output_dir / f"{split}.jsonl"), force_ascii=False)
