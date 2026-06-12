"""Text cleaning and privacy protection helpers."""

import re
from typing import Iterable


def mask_personal_data(text: str) -> str:
    """Mask personal data before analysis or explanation.

    This reduces privacy risk by removing common identifiers such as email
    addresses, phone numbers, URLs, and simple address patterns.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", " [EMAIL] ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
    text = re.sub(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b", " [PHONE] ", text)
    text = re.sub(
        r"\b\d{1,5}\s+[A-Za-z0-9\s.,'-]{2,40}\s+"
        r"(street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd)\b",
        " [ADDRESS] ",
        text,
        flags=re.IGNORECASE,
    )
    return text


def clean_text(text: str) -> str:
    """Return normalized, privacy-masked text for machine learning."""
    if not isinstance(text, str):
        return ""

    text = mask_personal_data(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def combine_text_columns(row, columns: Iterable[str]) -> str:
    """Combine available dataset text columns into one training document."""
    parts = []
    for column in columns:
        value = row.get(column, "")
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts)
