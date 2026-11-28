import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.prepare import load_raw_splits, prepare_webnlg_dataset
from src.models.transformer.config import BOS_ID, EOS_ID, PAD_ID, UNK_ID, TrainConfig


def load_examples(raw_dir: Path):
    """Read WebNLG JSONL as text inputs paired with serialized target triples."""
    return prepare_webnlg_dataset(load_raw_splits(raw_dir))


def train_tokenizer(train_examples, vocab_size: int) -> Tokenizer:
    """Fit one reversible byte-level BPE vocabulary on both sides of the training pairs."""
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    if vocab_size < len(alphabet) + len(special_tokens):
        raise ValueError("vocab_size must be at least 260 for byte-level BPE")
    if not len(train_examples):
        raise ValueError("The training split is empty")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.encode_special_tokens = True  # Treat literal marker strings in the corpus as text.

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, initial_alphabet=alphabet, special_tokens=special_tokens,
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        ((row["input"], row["target"]) for row in train_examples), trainer=trainer,
        length=len(train_examples),
    )

    if [tokenizer.token_to_id(s) for s in special_tokens] != [PAD_ID, BOS_ID, EOS_ID, UNK_ID]:
        raise ValueError("Special token IDs differ from the training configuration")
    return tokenizer


class Text2KGDataset(Dataset):
    """Encode each pair once, keeping the original examples available for evaluation."""

    def __init__(self, examples, tokenizer: Tokenizer, max_len: int):
        # tokenizers does not persist this option in tokenizer.json.
        tokenizer.encode_special_tokens = True
        self.examples = examples
        self.pairs = []
        for i, row in enumerate(examples):
            if not row["input"] or not row["target"]:
                raise ValueError(f"Example {i} needs both source text and target triples")
            
            source = tokenizer.encode(row["input"]).ids
            graph = tokenizer.encode(row["target"]).ids
            needed = max(len(source) + 2, len(graph) + 1)
            if needed > max_len:
                raise ValueError(f"Example {i} needs {needed} positions, exceeding max_len={max_len}")
            
            self.pairs.append((
                torch.tensor([BOS_ID, *source, EOS_ID]),
                torch.tensor([BOS_ID, *graph, EOS_ID]),
            ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, graph = self.pairs[index]
        return {"src": source, "tgt": graph[:-1], "labels": graph[1:]}


def collate_batch(batch):
    """Pad to the batch maxima. True masks a padding key or a future target position."""
    result = {
        key: pad_sequence([row[key] for row in batch], batch_first=True, padding_value=PAD_ID)
        for key in ("src", "tgt", "labels")
    }
    result["src_padding_mask"] = result["src"].eq(PAD_ID)
    result["tgt_padding_mask"] = result["tgt"].eq(PAD_ID)
    target_length = result["tgt"].size(1)
    result["causal_mask"] = torch.ones(target_length, target_length, dtype=torch.bool).triu(1)
    return result

if __name__ == "__main__":
    config = TrainConfig()
    examples = load_examples(config.raw_dir)
    tokenizer = train_tokenizer(examples["train"], config.vocab_size)
    sample = examples["train"].take(3)
    dataset = Text2KGDataset(sample, tokenizer, config.max_len)
    batch = collate_batch(list(dataset))
    print("batch keys:", list(batch.keys()))
    print("src:", batch["src"])
    print("tgt:", batch["tgt"])
    print("labels:", batch["labels"])
    print("src_padding_mask:", batch["src_padding_mask"])
    print("tgt_padding_mask:", batch["tgt_padding_mask"])
    print("causal_mask:", batch["causal_mask"])
