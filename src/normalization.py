"""Landmark normalization shared by English and Urdu modes.

Order (hard requirement):
1. Translate — wrist (landmark 0) becomes the origin
2. Scale — divide by wrist → middle-finger MCP (landmark 9)
3. Mirror left hands (flip x) onto right-hand convention
4. Flatten to 63-d (21 × x,y,z)
5. Optionally concatenate inter-joint angles (Urdu) — never replace the 63-d vector
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
EPS = 1e-8

# (previous, joint, next) along each finger, plus wrist→MCP for each digit.
ANGLE_TRIPLETS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
)

FINGERTIP_XY_INDICES = (4, 8, 12, 16, 20)


@dataclass(frozen=True)
class NormalizedHand:
    coords: np.ndarray  # (21, 3) wrist-origin, scale-normalized, right-hand frame
    vector_63: np.ndarray
    angles: np.ndarray  # (15,) radians
    handedness: str
    mirrored: bool

    def features(self, include_angles: bool) -> np.ndarray:
        if include_angles:
            return np.concatenate([self.vector_63, self.angles.astype(np.float32)])
        return self.vector_63

    def fingertip_xy(self) -> np.ndarray:
        """Five fingertips as a (10,) xy vector — used for DTW trajectories."""
        return self.coords[list(FINGERTIP_XY_INDICES), :2].reshape(-1).astype(np.float32)


def _as_points(landmarks: np.ndarray) -> np.ndarray:
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {pts.shape}")
    return pts.copy()


def translate_to_wrist(landmarks: np.ndarray) -> np.ndarray:
    pts = _as_points(landmarks)
    pts -= pts[WRIST_INDEX]
    return pts


def scale_by_wrist_mcp(landmarks: np.ndarray) -> np.ndarray:
    pts = _as_points(landmarks)
    ref = float(np.linalg.norm(pts[MIDDLE_MCP_INDEX] - pts[WRIST_INDEX]))
    if ref < EPS:
        dists = np.linalg.norm(pts - pts[WRIST_INDEX], axis=1)
        ref = float(np.max(dists))
    if ref < EPS:
        return pts
    return pts / ref


def mirror_to_right_hand(landmarks: np.ndarray, handedness: str) -> tuple[np.ndarray, bool]:
    pts = _as_points(landmarks)
    if str(handedness).lower().startswith("left"):
        pts[:, 0] *= -1.0
        return pts, True
    return pts, False


def joint_angles(landmarks: np.ndarray) -> np.ndarray:
    pts = _as_points(landmarks)
    angles = np.zeros(len(ANGLE_TRIPLETS), dtype=np.float32)
    for i, (a, b, c) in enumerate(ANGLE_TRIPLETS):
        v1 = pts[a] - pts[b]
        v2 = pts[c] - pts[b]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < EPS or n2 < EPS:
            continue
        cos = float(np.dot(v1, v2) / (n1 * n2))
        angles[i] = np.arccos(np.clip(cos, -1.0, 1.0))
    return angles


def normalize_landmarks(
    landmarks: np.ndarray,
    handedness: str = "Right",
) -> NormalizedHand:
    """Full shared pipeline. Language modes only choose whether to append angles."""
    pts = translate_to_wrist(landmarks)
    pts = scale_by_wrist_mcp(pts)
    pts, mirrored = mirror_to_right_hand(pts, handedness)
    angles = joint_angles(pts)
    flat = pts.reshape(-1).astype(np.float32)
    return NormalizedHand(
        coords=pts.astype(np.float32),
        vector_63=flat,
        angles=angles,
        handedness=handedness,
        mirrored=mirrored,
    )


def feature_names(include_angles: bool) -> list[str]:
    names = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    if include_angles:
        names.extend(f"ang{i}" for i in range(len(ANGLE_TRIPLETS)))
    return names


def augment_coords(
    coords: np.ndarray,
    rng: np.random.Generator,
    max_deg: float = 12.0,
    scale_jitter: float = 0.08,
    noise_std: float = 0.012,
) -> np.ndarray:
    """In-plane rotation + scale + Gaussian noise on already-normalized coords."""
    pts = _as_points(coords)
    theta = np.deg2rad(rng.uniform(-max_deg, max_deg))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    pts = pts @ rot.T
    pts *= rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
    pts += rng.normal(0.0, noise_std, size=pts.shape).astype(np.float32)
    return pts


def _self_test() -> None:
    rng = np.random.default_rng(0)
    canonical = rng.normal(size=(21, 3)).astype(np.float32)
    canonical[0] = 0
    near = canonical * 0.35 + np.array([0.22, 0.41, 0.0], dtype=np.float32)
    far = canonical * 0.90 + np.array([0.61, 0.07, 0.0], dtype=np.float32)
    n1 = normalize_landmarks(near, "Right")
    n2 = normalize_landmarks(far, "Right")
    diff = float(np.max(np.abs(n1.vector_63 - n2.vector_63)))
    print(f"translate+scale invariance max |delta| = {diff:.6g} (expect ~0)")
    if diff > 1e-5:
        raise SystemExit("Normalization self-test failed: scale/translation not removed")

    left = near.copy()
    left[:, 0] = 1.0 - left[:, 0]
    n_left = normalize_landmarks(left, "Left")
    mirror_diff = float(np.max(np.abs(n1.vector_63 - n_left.vector_63)))
    print(f"left->right mirror max |delta| = {mirror_diff:.6g} (expect small)")
    print(f"63-d + {len(n1.angles)} angle features ready")
    print("normalization self-test passed")


if __name__ == "__main__":
    _self_test()
