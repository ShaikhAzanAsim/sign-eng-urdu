"""OpenCV HUD: mode, prediction, sentence, key hints."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.urdu_text_render import contains_arabic, render_urdu_text


@dataclass
class OverlayState:
    mode: str = "english"
    preview: str = ""
    confirmed: str = ""
    confidence: float = 0.0
    sentence: str = ""
    status: str = ""
    fps: float = 0.0
    hands: int = 0
    dynamic_guess: str = ""
    hints: list[str] = field(default_factory=list)


def _box(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.55) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_hud(frame: np.ndarray, state: OverlayState) -> np.ndarray:
    h, w = frame.shape[:2]
    _box(frame, 8, 8, w - 8, 86)
    _box(frame, 8, h - 78, w - 8, h - 8)

    mode = state.mode.upper()
    cv2.putText(
        frame,
        f"{mode}   FPS {state.fps:.0f}   hands {state.hands}",
        (22, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 220, 255) if state.mode == "english" else (180, 160, 255),
        2,
        cv2.LINE_AA,
    )
    status = state.status or ("No hand detected" if state.hands == 0 else "")
    if status:
        cv2.putText(
            frame,
            status,
            (22, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 180, 255),
            1,
            cv2.LINE_AA,
        )

    preview = state.preview or "—"
    if contains_arabic(preview):
        render_urdu_text(frame, preview, (w - 220, 16), font_size=48, color=(255, 255, 255))
    else:
        cv2.putText(
            frame,
            preview,
            (w - 220, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

    sentence = state.sentence if state.sentence else "(empty sentence)"
    if contains_arabic(sentence) or (state.mode == "urdu" and state.sentence):
        render_urdu_text(frame, sentence, (22, h - 70), font_size=32, color=(240, 240, 240))
    else:
        cv2.putText(
            frame,
            sentence[-80:],
            (22, h - 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

    hints = state.hints or [
        "m mode   space word-break   backspace   c clear   q quit",
    ]
    cv2.putText(
        frame,
        "  |  ".join(hints),
        (22, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    if state.dynamic_guess:
        cv2.putText(
            frame,
            f"DTW {state.dynamic_guess}  conf {state.confidence:.2f}",
            (w // 2 - 80, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 255, 160),
            1,
            cv2.LINE_AA,
        )
    return frame
