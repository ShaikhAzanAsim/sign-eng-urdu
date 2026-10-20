"""Webcam tool: toggle record to append normalized landmark rows for the selected letter."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from src.alphabet import load_alphabet
from src.camera import open_camera
from src.console import configure_utf8_stdout
from src.landmark_extraction import HandLandmarkerEngine, draw_landmarks
from src.normalization import feature_names, normalize_landmarks
from src.paths import LANDMARKS_DIR, PROJECT_ROOT
from src.ui import OverlayState, draw_hud


def _csv_path(language: str, folder: str) -> Path:
    path = LANDMARKS_DIR / language / f"{folder}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def _append_row(path: Path, row: dict[str, object], columns: list[str]) -> None:
    df = pd.DataFrame([row], columns=columns)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def run_collection(language: str, start_id: str | None, camera: int) -> None:
    alphabet = load_alphabet(language)
    letters = alphabet["letters"]
    include_angles = language == "urdu"
    feat = feature_names(include_angles)
    columns = ["letter", "kind", "handedness", "mirrored", *feat]

    index = 0
    if start_id:
        wanted = start_id.lower()
        for i, letter in enumerate(letters):
            haystack = {
                letter["id"].lower(),
                letter["char"].lower(),
                letter["folder"].lower(),
                str(letter.get("transliteration", "")).lower(),
            }
            if wanted in haystack:
                index = i
                break

    cap = open_camera(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {camera}")

    print("r = toggle recording   n/p = next/prev letter   q = quit")
    recording = False
    last_save = 0.0
    interval = 1.0 / 12.0

    with HandLandmarkerEngine(running_mode="VIDEO", num_hands=2) as engine:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            letter = letters[index]
            path = _csv_path(language, letter["folder"])
            count = _row_count(path)

            hands = engine.detect_bgr(frame)
            annotated = draw_landmarks(frame, hands)

            if recording and hands:
                now = time.perf_counter()
                if now - last_save >= interval:
                    hand = hands[0]
                    norm = normalize_landmarks(hand.landmarks_norm, hand.handedness)
                    feats = norm.features(include_angles)
                    row: dict[str, object] = {
                        "letter": letter["id"],
                        "kind": letter["kind"],
                        "handedness": hand.handedness,
                        "mirrored": int(norm.mirrored),
                    }
                    row.update({name: float(value) for name, value in zip(feat, feats)})
                    _append_row(path, row, columns)
                    last_save = now
                    count += 1

            if recording and not hands:
                status = "RECORDING — no hand detected"
            elif recording:
                status = f"RECORDING  {count} samples"
            else:
                status = f"Idle  {count} samples  (target 150-300 static)"

            label = f"{letter['char']} {letter.get('transliteration', letter['id'])} [{letter['kind']}]"
            state = OverlayState(
                mode=language,
                preview=letter["char"],
                sentence=label,
                hands=len(hands),
                status=status,
                hints=["r record", "n next", "p prev", "q quit"],
            )
            draw_hud(annotated, state)
            cv2.imshow("Data collection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("n"):
                index = (index + 1) % len(letters)
                recording = False
            if key == ord("p"):
                index = (index - 1) % len(letters)
                recording = False
            if key == ord("r"):
                recording = not recording

    cap.release()
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record normalized landmarks per letter.")
    parser.add_argument("--lang", choices=("english", "urdu"), default="english")
    parser.add_argument("--letter", default=None, help="Start at this letter id/char/folder.")
    parser.add_argument("--camera", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdout()
    args = build_parser().parse_args(argv)
    print(f"Project root: {PROJECT_ROOT}")
    run_collection(args.lang, args.letter, args.camera)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
