"""Assemble confirmed letters into a word / sentence."""

from __future__ import annotations


class TextBuffer:
    def __init__(self) -> None:
        self.chars: list[str] = []

    def append(self, letter: str) -> None:
        if letter:
            self.chars.append(letter)

    def space(self) -> None:
        if self.chars and self.chars[-1] != " ":
            self.chars.append(" ")

    def backspace(self) -> None:
        if self.chars:
            self.chars.pop()

    def clear(self) -> None:
        self.chars.clear()

    @property
    def text(self) -> str:
        return "".join(self.chars)

    def __str__(self) -> str:
        return self.text
