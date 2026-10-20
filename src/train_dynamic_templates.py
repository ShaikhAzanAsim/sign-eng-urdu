"""Record a handful of DTW reference trajectories for dynamic letters."""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from src.alphabet import letters_of_kind, load_alphabet
from src.camera import open_camera
from src.console import configure_utf8_stdout
from src.dtw_match import TrajectoryBuffer
from src.landmark_extraction import HandLandmarkerEngine, draw_landmarks
from src.normalization import normalize_landmarks
from src.paths import ENGLISH_MODELS_DIR, PROJECT_ROOT, URDU_MODELS_DIR
from src.ui import OverlayState, draw_hud


def _templates_dir(language: str):
    root = ENGLISH_MODELS_DIR if language == "english" else URDU_MODELS_DIR
    path = root / "dynamic_templates"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_recorder(language: str, camera: int) -> None:
    dynamic = letters_of_kind(load_alphabet(language), "dynamic")
    if not dynamic:
        raise SystemExit(f"No dynamic letters listed for {language}")
    index = 0
    recording = False
    buffer = TrajectoryBuffer(maxlen=36)
    saved_counts: dict[str, int] = {letter["id"]: 0 for letter in dynamic}
    out_dir = _templates_dir(language)

    cap = open_camera(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {camera}")

    print("Sign the motion, press r to start, r again to save the trajectory.")
    with HandLandmarkerEngine(running_mode="VIDEO", num_hands=2) as engine:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            letter = dynamic[index]
            hands = engine.detect_bgr(frame)
            annotated = draw_landmarks(frame, hands)
            if recording and hands:
                norm = normalize_landmarks(hands[0].landmarks_norm, hands[0].handedness)
                buffer.push(norm.fingertip_xy())

            status = (
                f"RECORDING {len(buffer)} frames"
                if recording
                else f"Idle  saved={saved_counts[letter['id']]}  (3-8 templates is enough)"
            )
            state = OverlayState(
                mode=language,
                preview=letter["char"],
                sentence=f"{letter['char']} {letter.get('transliteration', letter['id'])}",
                hands=len(hands),
                status=status,
                hints=["r start/save", "n next", "p prev", "q quit"],
            )
            draw_hud(annotated, state)
            cv2.imshow("Dynamic templates", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("n"):
                index = (index + 1) % len(dynamic)
                recording = False
                buffer.clear()
            if key == ord("p"):
                index = (index - 1) % len(dynamic)
                recording = False
                buffer.clear()
            if key == ord("r"):
                if not recording:
                    buffer.clear()
                    recording = True
                else:
                    traj = buffer.as_array()
                    recording = False
                    if len(traj) < 10:
                        print("Too short — hold the motion longer.")
                        continue
                    stamp = int(time.time())
                    path = out_dir / f"{letter['id']}__{stamp}.npy"
                    np.save(path, traj)
                    saved_counts[letter["id"]] += 1
                    print(f"saved {path}  shape={traj.shape}")
                    buffer.clear()

    index_path = out_dir / "index.json"
    index_path.write_text(
        json.dumps({"language": language, "counts": saved_counts}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cap.release()
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record DTW templates for dynamic letters.")
    parser.add_argument("--lang", choices=("english", "urdu"), default="english")
    parser.add_argument("--camera", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdout()
    args = build_parser().parse_args(argv)
    print(f"Project root: {PROJECT_ROOT}")
    run_recorder(args.lang, args.camera)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
