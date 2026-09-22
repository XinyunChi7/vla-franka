"""Pure-Python gripper command hysteresis for the live policy bridge."""
from __future__ import annotations

import math


class GripperDebouncer:
    """Turn noisy continuous model output into deliberate gripper edges.

    Closing uses a shorter confirmation than opening so a grasp is responsive
    while accidental releases need stronger evidence.  There is deliberately
    no reverse lockout: a release/re-grasp can start confirming immediately.
    The model's original sign convention is preserved: non-positive means
    close and positive means open.  Debouncing changes only when an edge is
    executed, not how a model output is classified.
    """

    def __init__(
        self,
        *,
        close_confirm_ticks: int = 3,
        open_confirm_ticks: int = 6,
    ) -> None:
        if min(close_confirm_ticks, open_confirm_ticks) < 1:
            raise ValueError("confirmation ticks must be positive")
        self.close_confirm_ticks = int(close_confirm_ticks)
        self.open_confirm_ticks = int(open_confirm_ticks)
        self.stable_should_close = False
        self.pending_should_close: bool | None = None
        self.pending_ticks = 0

    def reset(self, *, is_closed: bool) -> None:
        self.stable_should_close = bool(is_closed)
        self.pending_should_close = None
        self.pending_ticks = 0

    def classify(self, value: float) -> bool | None:
        if not math.isfinite(value):
            raise ValueError(f"non-finite gripper model output: {value}")
        if value <= 0.0:
            return True
        return False

    def update(self, value: float) -> bool | None:
        """Return a new stable state only when a real edge is confirmed."""
        requested = self.classify(float(value))
        if requested == self.stable_should_close:
            self.pending_should_close = None
            self.pending_ticks = 0
            return None
        if requested != self.pending_should_close:
            self.pending_should_close = requested
            self.pending_ticks = 1
        else:
            self.pending_ticks += 1
        required = self.close_confirm_ticks if requested else self.open_confirm_ticks
        if self.pending_ticks < required:
            return None
        self.stable_should_close = requested
        self.pending_should_close = None
        self.pending_ticks = 0
        return requested
