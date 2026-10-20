"""Project-root path helpers. All modules should import paths from here."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
ENGLISH_MODELS_DIR = MODELS_DIR / "english"
URDU_MODELS_DIR = MODELS_DIR / "urdu"
HAND_LANDMARKER_TASK = MODELS_DIR / "hand_landmarker.task"

LABELS_DIR = PROJECT_ROOT / "labels"
ENGLISH_ALPHABET_PATH = LABELS_DIR / "english_alphabet.json"
URDU_ALPHABET_PATH = LABELS_DIR / "urdu_alphabet.json"

DATA_DIR = PROJECT_ROOT / "data"
RAW_CAPTURES_DIR = DATA_DIR / "raw_captures"
LANDMARKS_DIR = DATA_DIR / "landmarks"
REFERENCES_DIR = DATA_DIR / "references"

FONTS_DIR = PROJECT_ROOT / "fonts"

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
