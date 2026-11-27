import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pretrained.loading import download_model
from src.models.pretrained.model import generate_answer, load_adapter, load_base_model
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Generate WebNLG triples from text")
    parser.add_argument("text", help="Input text to convert into knowledge graph triples")
    parser.add_argument("--adapter", required=True, help="Path to a saved LoRA adapter")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "configs/fine_tuning.yaml")
    model_path = download_model(config["model"])
    tokenizer, base_model = load_base_model(model_path, config["model"].get("torch_dtype", "float16"))
    model = load_adapter(base_model, args.adapter)
    answer = generate_answer(model, tokenizer, args.text, **config.get("generation", {}))
    print(answer)


if __name__ == "__main__":
    main()
