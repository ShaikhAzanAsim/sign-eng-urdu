"""Merge per-letter landmark CSVs into matrices for training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.alphabet import letters_of_kind, load_alphabet
from src.paths import LANDMARKS_DIR


def letter_csv(language: str, folder: str) -> Path:
    return LANDMARKS_DIR / language / f"{folder}.csv"


def load_static_dataframe(language: str) -> pd.DataFrame:
    alphabet = load_alphabet(language)
    frames: list[pd.DataFrame] = []
    for letter in letters_of_kind(alphabet, "static"):
        path = letter_csv(language, letter["folder"])
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["letter_id"] = letter["id"]
        df["char"] = letter["char"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"letter", "letter_id", "char", "kind", "handedness", "mirrored"}
    return [c for c in df.columns if c not in skip]


def matrices(language: str, include_angles: bool | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = load_static_dataframe(language)
    if df.empty:
        return np.zeros((0, 1)), np.zeros((0,), dtype=int), []
    if include_angles is None:
        include_angles = language.lower() == "urdu"
    cols = [c for c in feature_columns(df) if include_angles or not str(c).startswith("ang")]
    labels = sorted(df["letter_id"].unique())
    label_to_idx = {name: i for i, name in enumerate(labels)}
    x = df[cols].to_numpy(dtype=np.float32)
    y = np.array([label_to_idx[v] for v in df["letter_id"]], dtype=np.int32)
    return x, y, labels
