"""Load English / Urdu alphabet label maps (static vs dynamic flags)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.console import configure_utf8_stdout
from src.paths import ENGLISH_ALPHABET_PATH, URDU_ALPHABET_PATH

Alphabet = dict[str, Any]


def load_alphabet(language: str) -> Alphabet:
    language = language.lower().strip()
    path = {
        "english": ENGLISH_ALPHABET_PATH,
        "en": ENGLISH_ALPHABET_PATH,
        "asl": ENGLISH_ALPHABET_PATH,
        "urdu": URDU_ALPHABET_PATH,
        "ur": URDU_ALPHABET_PATH,
        "psl": URDU_ALPHABET_PATH,
    }.get(language)
    if path is None:
        raise ValueError(f"Unknown language '{language}'. Use 'english' or 'urdu'.")
    return load_alphabet_file(path)


def load_alphabet_file(path: Path) -> Alphabet:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    letters = data.get("letters") or []
    if not letters:
        raise ValueError(f"No letters found in {path}")
    ids = [letter["id"] for letter in letters]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate letter ids in {path}")
    return data


def letters_of_kind(alphabet: Alphabet, kind: str) -> list[dict[str, Any]]:
    return [letter for letter in alphabet["letters"] if letter["kind"] == kind]


def summarize(alphabet: Alphabet) -> str:
    static = letters_of_kind(alphabet, "static")
    dynamic = letters_of_kind(alphabet, "dynamic")
    lang = alphabet.get("language", "?")
    return (
        f"{lang}: {len(alphabet['letters'])} letters "
        f"({len(static)} static, {len(dynamic)} dynamic)"
    )


def main() -> None:
    configure_utf8_stdout()
    for language in ("english", "urdu"):
        alphabet = load_alphabet(language)
        print(summarize(alphabet))
        dynamic = letters_of_kind(alphabet, "dynamic")
        if dynamic:
            names = ", ".join(
                f"{letter.get('char', letter['id'])} ({letter.get('transliteration', letter['id'])})"
                for letter in dynamic
            )
            print(f"  dynamic: {names}")


if __name__ == "__main__":
    main()
