"""MediaPipe Tasks HandLandmarker wrapper — shared by English and Urdu modes.

Uses ``mediapipe.tasks.python.vision.HandLandmarker`` (current Tasks API).
Does **not** import the deprecated ``mediapipe.solutions.hands`` module.
Landmarks are drawn with OpenCV so we don't depend on removed drawing utilities.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.alphabet import load_alphabet, summarize
from src.camera import open_camera
from src.console import configure_utf8_stdout
from src.download_model import ensure_hand_landmarker_model
from src.paths import HAND_LANDMARKER_TASK, PROJECT_ROOT

# MediaPipe 21-point hand topology (wrist + 5 fingers). Same indices as the
# official Hands model; we keep the connection list here so drawing never
# touches mediapipe.solutions.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

FINGERTIP_INDICES = (4, 8, 12, 16, 20)
WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9

_LANDMARK_BGR = (0, 255, 0)
_FINGERTIP_BGR = (0, 220, 255)
_WRIST_BGR = (255, 80, 80)
_CONNECTION_BGR = (80, 180, 255)


@dataclass(frozen=True)
class DetectedHand:
    """One detected hand in a single frame."""

    landmarks_norm: np.ndarray  # (21, 3) image-normalized x, y, z
    world_landmarks: np.ndarray  # (21, 3) meters, origin at hand centroid
    handedness: str  # "Left" or "Right"
    handedness_score: float

    def as_pixel_xy(self, width: int, height: int) -> np.ndarray:
        xy = self.landmarks_norm[:, :2].copy()
        xy[:, 0] *= width
        xy[:, 1] *= height
        return xy


class HandLandmarkerEngine:
    """Thin wrapper around MediaPipe Tasks HandLandmarker.

    Webcam / video callers should use ``running_mode="VIDEO"`` and pass a
    monotonically increasing ``timestamp_ms`` (tracking stays on). Still
    images can use ``running_mode="IMAGE"``.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        num_hands: int = 2,
        running_mode: str = "VIDEO",
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp

        path = Path(model_path) if model_path else HAND_LANDMARKER_TASK
        if not path.exists():
            path = ensure_hand_landmarker_model(path)

        mode_name = running_mode.upper()
        try:
            mode = mp.tasks.vision.RunningMode[mode_name]
        except KeyError as exc:
            raise ValueError(
                f"running_mode must be IMAGE, VIDEO, or LIVE_STREAM, got {running_mode!r}"
            ) from exc

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path.resolve())),
            running_mode=mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp = mp
        self._mode = mode
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self._t0 = time.perf_counter()

    def _next_timestamp_ms(self) -> int:
        elapsed = int((time.perf_counter() - self._t0) * 1000)
        if elapsed <= self._timestamp_ms:
            elapsed = self._timestamp_ms + 1
        self._timestamp_ms = elapsed
        return elapsed

    def detect_bgr(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> list[DetectedHand]:
        """Run detection on a BGR OpenCV frame. Returns [] if no hand is present."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        if self._mode.name == "IMAGE":
            result = self._landmarker.detect(mp_image)
        else:
            ts = timestamp_ms if timestamp_ms is not None else self._next_timestamp_ms()
            result = self._landmarker.detect_for_video(mp_image, ts)

        return parse_hand_landmarker_result(result)

    def close(self) -> None:
        closer = getattr(self._landmarker, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "HandLandmarkerEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def parse_hand_landmarker_result(result: object) -> list[DetectedHand]:
    hands: list[DetectedHand] = []
    landmark_lists = getattr(result, "hand_landmarks", None) or []
    world_lists = getattr(result, "hand_world_landmarks", None) or []
    handedness_lists = getattr(result, "handedness", None) or []

    for i, landmarks in enumerate(landmark_lists):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        if coords.shape != (21, 3):
            continue

        if i < len(world_lists) and world_lists[i]:
            world = np.array(
                [[lm.x, lm.y, lm.z] for lm in world_lists[i]], dtype=np.float32
            )
        else:
            world = np.zeros((21, 3), dtype=np.float32)

        handedness = "Unknown"
        score = 0.0
        if i < len(handedness_lists) and handedness_lists[i]:
            category = handedness_lists[i][0]
            handedness = getattr(category, "category_name", None) or getattr(
                category, "display_name", "Unknown"
            )
            score = float(getattr(category, "score", 0.0))

        hands.append(
            DetectedHand(
                landmarks_norm=coords,
                world_landmarks=world,
                handedness=handedness,
                handedness_score=score,
            )
        )
    return hands


def draw_landmarks(
    frame_bgr: np.ndarray,
    hands: list[DetectedHand],
    draw_indices: bool = False,
) -> np.ndarray:
    """Draw 21 points + connections. Safe to call with an empty hand list."""
    annotated = frame_bgr
    height, width = annotated.shape[:2]

    if not hands:
        _put_banner(annotated, "No hand detected", (40, 180, 255))
        return annotated

    for hand in hands:
        pixels = hand.as_pixel_xy(width, height)
        for a, b in HAND_CONNECTIONS:
            pa = (int(pixels[a, 0]), int(pixels[a, 1]))
            pb = (int(pixels[b, 0]), int(pixels[b, 1]))
            cv2.line(annotated, pa, pb, _CONNECTION_BGR, 2, cv2.LINE_AA)

        for idx, (x, y) in enumerate(pixels):
            center = (int(x), int(y))
            if idx == WRIST_INDEX:
                color, radius = _WRIST_BGR, 7
            elif idx in FINGERTIP_INDICES:
                color, radius = _FINGERTIP_BGR, 6
            else:
                color, radius = _LANDMARK_BGR, 4
            cv2.circle(annotated, center, radius, color, -1, cv2.LINE_AA)
            if draw_indices:
                cv2.putText(
                    annotated,
                    str(idx),
                    (center[0] + 6, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        label = f"{hand.handedness} ({hand.handedness_score:.2f})"
        wrist = (int(pixels[0, 0]), int(pixels[0, 1]) - 16)
        cv2.putText(
            annotated,
            label,
            wrist,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    _put_banner(annotated, f"Hands: {len(hands)}", (80, 220, 80))
    return annotated


def _put_banner(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (8, 8), (420, 48), (20, 20, 20), -1)
    cv2.putText(
        frame,
        text,
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def run_webcam(camera: int, draw_indices: bool) -> None:
    ensure_hand_landmarker_model()
    cap = open_camera(camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {camera}. Close other apps using the camera and retry."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = "Hand landmarks (q quit, i toggle indices)"
    print("Live overlay running. Press q to quit, i to toggle landmark indices.")
    show_indices = draw_indices
    prev = time.perf_counter()

    with HandLandmarkerEngine(running_mode="VIDEO", num_hands=2) as engine:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            hands = engine.detect_bgr(frame)
            annotated = draw_landmarks(frame, hands, draw_indices=show_indices)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(
                annotated,
                f"{fps:.0f} FPS",
                (annotated.shape[1] - 140, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                "Phase 1 overlay  |  q quit  i indices",
                (18, annotated.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("i"):
                show_indices = not show_indices

    cap.release()
    cv2.destroyAllWindows()


def run_image(
    image_path: Path,
    draw_indices: bool,
    save_path: Path | None,
    show_window: bool = True,
) -> None:
    ensure_hand_landmarker_model()
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    with HandLandmarkerEngine(running_mode="IMAGE", num_hands=4) as engine:
        hands = engine.detect_bgr(frame)
        annotated = draw_landmarks(frame, hands, draw_indices=draw_indices)

    print(f"{image_path.name}: {len(hands)} hand(s) detected")
    for i, hand in enumerate(hands):
        print(f"  #{i}: {hand.handedness} score={hand.handedness_score:.3f}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)
        print(f"Wrote overlay to {save_path}")

    if not show_window:
        return

    window = "Hand landmarks (q to close)"
    cv2.imshow(window, annotated)
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live MediaPipe Tasks hand-landmark overlay (Phase 1)."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Run on a still image instead of the webcam.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="When using --image, also write the annotated frame here.",
    )
    parser.add_argument(
        "--indices",
        action="store_true",
        help="Draw landmark index numbers (0-20).",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Skip OpenCV windows (use with --image --save for a headless smoke test).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdout()
    args = build_parser().parse_args(argv)
    print(f"Project root: {PROJECT_ROOT}")
    print(summarize(load_alphabet("english")))
    print(summarize(load_alphabet("urdu")))

    if args.image is not None:
        run_image(args.image, args.indices, args.save, show_window=not args.no_window)
    else:
        if args.no_window:
            raise SystemExit("--no-window only applies with --image")
        run_webcam(args.camera, args.indices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
