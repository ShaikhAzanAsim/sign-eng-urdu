"""Rolling majority vote + hold-to-confirm so the sentence buffer doesn't stutter."""

from __future__ import annotations

from collections import Counter, deque


class PredictionSmoother:
    def __init__(
        self,
        window: int = 12,
        majority_frac: float = 0.7,
        hold_seconds: float = 0.7,
        cooldown_seconds: float = 0.45,
    ) -> None:
        self.window = window
        self.majority_frac = majority_frac
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self._votes: deque[str | None] = deque(maxlen=window)
        self._candidate: str | None = None
        self._candidate_since: float | None = None
        self._last_emitted: str | None = None
        self._last_emit_time: float = -1e9
        self._gap_since_emit = True

    def reset(self) -> None:
        self._votes.clear()
        self._candidate = None
        self._candidate_since = None

    def observe(self, label: str | None, now: float) -> str | None:
        """Feed one frame. Returns a newly confirmed letter, else None."""
        self._votes.append(label)

        if label is None:
            self._votes.clear()
            self._candidate = None
            self._candidate_since = None
            self._gap_since_emit = True
            return None

        confirmed = self.majority_label()
        if confirmed is None:
            self._candidate = None
            self._candidate_since = None
            return None

        if confirmed != self._candidate:
            self._candidate = confirmed
            self._candidate_since = now
            return None

        held = now - (self._candidate_since if self._candidate_since is not None else now)
        if held < self.hold_seconds:
            return None

        same_as_last = confirmed == self._last_emitted and not self._gap_since_emit
        if same_as_last:
            return None
        if now - self._last_emit_time < self.cooldown_seconds:
            return None

        self._last_emitted = confirmed
        self._last_emit_time = now
        self._gap_since_emit = False
        self._candidate_since = now
        return confirmed

    def majority_label(self) -> str | None:
        labels = [v for v in self._votes if v is not None]
        if not labels:
            return None
        label, count = Counter(labels).most_common(1)[0]
        if count / max(len(labels), 1) < self.majority_frac:
            return None
        return label

    @property
    def preview(self) -> str | None:
        return self.majority_label()
