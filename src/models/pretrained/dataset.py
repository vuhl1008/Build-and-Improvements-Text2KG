from src.models.pretrained.prompt import SYSTEM_PROMPT


def format_for_sft(dataset):
    """Add the chat representation expected by TRL to shared text/triples."""
    def convert(example):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["target"]},
            ]
        }

    return dataset.map(convert, remove_columns=dataset.column_names)


