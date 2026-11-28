import argparse
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PAD_ID = 0
BOS_ID = SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


@dataclass
class TrainConfig:
    raw_dir: Path = PROJECT_ROOT / "data/raw"
    output_dir: Path = PROJECT_ROOT / "data/transformer/runs"
    vocab_size: int = 4096
    max_len: int = 512
    d_model: int = 128
    num_heads: int = 4
    d_ff: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 3e-4
    seed: int = 42
    eval_samples: int = 10

    def __post_init__(self) -> None:
        if self.vocab_size < 260:
            raise ValueError("vocab_size must fit 256 bytes and four special tokens")
        if not 3 <= self.max_len <= 512:
            raise ValueError("max_len must be between 3 and 512 for the positional embeddings")
        if self.num_heads < 1 or self.d_model < 1 or self.d_model % self.num_heads:
            raise ValueError("d_model must be positive and divisible by a positive num_heads")
        if min(self.d_ff, self.num_layers, self.epochs, self.batch_size) < 1:
            raise ValueError("d_ff, num_layers, epochs and batch_size must be positive")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0:
            raise ValueError("dropout must be in [0, 1) and learning_rate must be positive")
        if self.eval_samples < 0:
            raise ValueError("eval_samples must be nonnegative")


def config_from_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train a text-to-KG Transformer on raw WebNLG JSONL.")
    parser.add_argument("--raw-dir", type=Path, default=defaults.raw_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    for name in ("vocab_size", "max_len", "d_model", "num_heads", "d_ff", "num_layers",
                 "batch_size", "epochs", "seed", "eval_samples"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    return TrainConfig(**vars(parser.parse_args()))
