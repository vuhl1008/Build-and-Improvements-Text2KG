import re
import unicodedata


def clean_text(text):
    """Normalize a WebNLG text field while retaining its meaning."""
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
