"""Real-time bilingual fingerspelling loop (OpenCV window, keypress mode toggle)."""

from __future__ import annotations

import argparse
import json
import time

import cv2
import joblib
import numpy as np

from src.alphabet import load_alphabet
from src.camera import open_camera
from src.console import configure_utf8_stdout
from src.dtw_match import DynamicMatcher, TrajectoryBuffer
from src.landmark_extraction import HandLandmarkerEngine, draw_landmarks
from src.normalization import normalize_landmarks
from src.paths import ENGLISH_MODELS_DIR, PROJECT_ROOT, URDU_MODELS_DIR
from src.smoothing import PredictionSmoother
from src.text_buffer import TextBuffer
from src.ui import OverlayState, draw_hud
from src.urdu_text_render import ensure_urdu_font


class LanguageBundle:
    def __init__(self, language: str) -> None:
        self.language = language
        self.alphabet = load_alphabet(language)
        self.include_angles = language == "urdu"
        self.static_ids = {
            letter["id"]: letter for letter in self.alphabet["letters"] if letter["kind"] == "static"
        }
        self.dynamic_ids = {
            letter["id"]: letter for letter in self.alphabet["letters"] if letter["kind"] == "dynamic"
        }
        self.id_to_char = {letter["id"]: letter["char"] for letter in self.alphabet["letters"]}
        root = ENGLISH_MODELS_DIR if language == "english" else URDU_MODELS_DIR
        self.classifier = None
        self.labels: list[str] = []
        clf_path = root / "static_classifier.pkl"
        labels_path = root / "static_labels.json"
        if clf_path.exists() and labels_path.exists():
            self.classifier = joblib.load(clf_path)
            payload = json.loads(labels_path.read_text(encoding="utf-8"))
            self.labels = list(payload["labels"])
            self.include_angles = bool(payload.get("include_angles", self.include_angles))
        self.matcher = DynamicMatcher(root / "dynamic_templates")

    def display(self, letter_id: str | None) -> str:
        if not letter_id:
            return ""
        return self.id_to_char.get(letter_id, letter_id)

    def predict_static(self, features: np.ndarray) -> tuple[str | None, float]:
        if self.classifier is None:
            return None, 0.0
        feats = features.reshape(1, -1)
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(feats)[0]
            idx = int(np.argmax(proba))
            return self.labels[idx], float(proba[idx])
        idx = int(self.classifier.predict(feats)[0])
        return self.labels[idx], 1.0


def run_inference(start_mode: str, camera: int) -> None:
    ensure_urdu_font()
    bundles = {
        "english": LanguageBundle("english"),
        "urdu": LanguageBundle("urdu"),
    }
    mode = start_mode
    smoother = PredictionSmoother()
    buffer = TextBuffer()
    traj = TrajectoryBuffer(maxlen=28)

    cap = open_camera(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {camera}")

    print("Live inference. m = switch English/Urdu. q = quit.")
    prev = time.perf_counter()
    with HandLandmarkerEngine(running_mode="VIDEO", num_hands=2) as engine:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            bundle = bundles[mode]
            hands = engine.detect_bgr(frame)
            annotated = draw_landmarks(frame, hands)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            preview_id: str | None = None
            confidence = 0.0
            dynamic_guess = ""
            status = ""

            if not hands:
                traj.clear()
                emitted = smoother.observe(None, now)
                status = "No hand detected"
            else:
                if bundle.classifier is None:
                    status = f"No {mode} static model — collect data, then train_static"
                norm = normalize_landmarks(hands[0].landmarks_norm, hands[0].handedness)
                traj.push(norm.fingertip_xy())
                feats = norm.features(bundle.include_angles)
                static_id, confidence = bundle.predict_static(feats)
                dyn_id, dyn_dist = bundle.matcher.match(traj.as_array())
                if dyn_id:
                    preview_id = dyn_id
                    dynamic_guess = f"{dyn_id} d={dyn_dist:.3f}"
                    confidence = max(confidence, 0.99)
                else:
                    preview_id = static_id
                emitted = smoother.observe(preview_id, now)

            if emitted:
                buffer.append(bundle.display(emitted))

            state = OverlayState(
                mode=mode,
                preview=bundle.display(smoother.preview or preview_id),
                confidence=confidence,
                sentence=buffer.text,
                hands=len(hands),
                fps=fps,
                status=status,
                dynamic_guess=dynamic_guess,
                hints=["m mode", "space", "backspace", "c clear", "q quit"],
            )
            draw_hud(annotated, state)
            cv2.imshow("Sign alphabet", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mode = "urdu" if mode == "english" else "english"
                smoother.reset()
                traj.clear()
            if key in (ord(" "),):
                buffer.space()
            if key in (8, 127):
                buffer.backspace()
            if key == ord("c"):
                buffer.clear()

    cap.release()
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live bilingual sign-alphabet recognizer.")
    parser.add_argument("--lang", choices=("english", "urdu"), default="english")
    parser.add_argument("--camera", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdout()
    args = build_parser().parse_args(argv)
    print(f"Project root: {PROJECT_ROOT}")
    run_inference(args.lang, args.camera)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
