"""DTW matching of buffered fingertip trajectories against stored templates."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

try:
    from dtaidistance.dtw_ndim import distance as dtw_ndim_distance
except ImportError:  # pragma: no cover
    dtw_ndim_distance = None


def _fallback_dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Simple multivariate DTW if dtaidistance is unavailable."""
    n, m = len(a), len(b)
    inf = 1e9
    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if dtw_ndim_distance is not None:
        return float(dtw_ndim_distance(a, b))
    return _fallback_dtw(a, b)


def path_travel(traj: np.ndarray) -> float:
    if len(traj) < 2:
        return 0.0
    diffs = np.diff(np.asarray(traj, dtype=np.float64), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


class TrajectoryBuffer:
    def __init__(self, maxlen: int = 28) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=maxlen)

    def push(self, fingertip_xy: np.ndarray) -> None:
        self._frames.append(np.asarray(fingertip_xy, dtype=np.float32).reshape(-1))

    def clear(self) -> None:
        self._frames.clear()

    def as_array(self) -> np.ndarray:
        if not self._frames:
            return np.zeros((0, 10), dtype=np.float32)
        return np.stack(self._frames, axis=0)

    def __len__(self) -> int:
        return len(self._frames)


class DynamicMatcher:
    def __init__(self, templates_dir: Path, threshold: float = 0.045) -> None:
        self.templates_dir = Path(templates_dir)
        self.threshold = threshold
        self.templates: dict[str, list[np.ndarray]] = {}
        self.reload()

    def reload(self) -> None:
        self.templates = {}
        if not self.templates_dir.exists():
            return
        for npy in sorted(self.templates_dir.glob("*.npy")):
            letter = npy.stem.split("__")[0]
            traj = np.load(npy)
            if traj.ndim == 1:
                continue
            self.templates.setdefault(letter, []).append(traj.astype(np.float32))
        meta = self.templates_dir / "index.json"
        if meta.exists():
            # optional; filenames already encode the letter id
            _ = json.loads(meta.read_text(encoding="utf-8"))

    def match(self, trajectory: np.ndarray, min_frames: int = 12) -> tuple[str | None, float]:
        if trajectory is None or len(trajectory) < min_frames:
            return None, float("inf")
        if path_travel(trajectory.reshape(len(trajectory), -1)) < 0.35:
            return None, float("inf")

        best_letter: str | None = None
        best_dist = float("inf")
        series = trajectory.reshape(len(trajectory), -1)
        for letter, refs in self.templates.items():
            for ref in refs:
                dist = dtw_distance(series, ref.reshape(len(ref), -1))
                norm = dist / max(len(series), len(ref), 1)
                if norm < best_dist:
                    best_dist = norm
                    best_letter = letter
        if best_letter is None or best_dist > self.threshold:
            return None, best_dist
        return best_letter, best_dist

    @property
    def ready(self) -> bool:
        return any(self.templates.values())
