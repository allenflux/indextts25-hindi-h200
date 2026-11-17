"""Conservative Hindi text policy, deliberately without automatic verbalization."""
from __future__ import annotations

import re
import unicodedata


class DataError(ValueError):
    """An input cannot safely satisfy the declared data contract."""


def normalize_hindi(text: str, *, allow_as_spoken_numbers: bool = False,
                    numbers_as_spoken: bool = False, min_chars: int = 2,
                    max_chars: int = 400) -> str:
    """NFC + whitespace only. Preserve nukta, virama, ZWJ, and ZWNJ.

    Numbers and currency require both a CLI acknowledgment and per-row assertion
    that the exact written form matches the recording. No Hindi number guesser.
    """
    if not isinstance(text, str):
        raise DataError("text must be a string")
    if "<|" in text or "|>" in text:
        raise DataError("text contains a reserved model-token delimiter")
    for char in text:
        category = unicodedata.category(char)
        if category in {"Cs", "Cn"} or (category == "Cc" and char not in "\t\r\n"):
            raise DataError("text contains an unsupported or control character")
        if category == "Cf" and char not in "\u200c\u200d":
            raise DataError("text contains a bidi/format control other than ZWJ/ZWNJ")
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
    if not min_chars <= len(normalized) <= max_chars:
        raise DataError(f"normalized text length must be {min_chars}..{max_chars} characters")
    if not any("\u0900" <= c <= "\u097f" and unicodedata.category(c).startswith("L")
               for c in normalized):
        raise DataError("Hindi rows must contain a Devanagari letter")
    if any(unicodedata.category(c) in {"Nd", "Nl", "No", "Sc"} for c in normalized):
        if not (allow_as_spoken_numbers and numbers_as_spoken is True):
            raise DataError("digits/currency require manual spoken-text review and both acknowledgments")
    return normalized
