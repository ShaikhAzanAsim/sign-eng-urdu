"""Download Google's Hand Landmarker .task bundle into models/ if missing."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from src.paths import HAND_LANDMARKER_TASK, HAND_LANDMARKER_URL, MODELS_DIR


def ensure_hand_landmarker_model(dest: Path | None = None) -> Path:
    path = dest or HAND_LANDMARKER_TASK
    if path.exists() and path.stat().st_size > 1_000_000:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand_landmarker.task to {path} ...")
    tmp = path.with_suffix(".task.part")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, tmp)
    tmp.replace(path)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved {path.name} ({size_mb:.1f} MB)")
    return path


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_hand_landmarker_model()


if __name__ == "__main__":
    sys.exit(main() or 0)
