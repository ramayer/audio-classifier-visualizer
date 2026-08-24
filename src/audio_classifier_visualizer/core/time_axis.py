"""Time handling.

Design rule: everything internal is seconds relative to the start of the
in-memory buffer. Absolute/UTC time is a *view* derived from a single
optional anchor (``absolute_start``), never a second parallel state that
has to be kept in sync by hand. This replaces the old pattern of bouncing
between "0 at start of clip" and "UTC clock time" throughout the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True, slots=True)
class TimeAxis:
    """Maps relative seconds (0 == start of the loaded buffer) to wall-clock time.

    ``absolute_start`` is optional. If it is None, this audio has no known
    real-world anchor (e.g. synthetic test data) and absolute-time queries
    raise rather than silently returning nonsense.
    """

    absolute_start: datetime | None = None

    def to_absolute(self, relative_seconds: float) -> datetime:
        if self.absolute_start is None:
            msg = "This TimeAxis has no absolute_start anchor; cannot convert to wall-clock time."
            raise ValueError(msg)
        return self.absolute_start + timedelta(seconds=relative_seconds)

    def from_absolute(self, when: datetime) -> float:
        if self.absolute_start is None:
            msg = "This TimeAxis has no absolute_start anchor; cannot convert from wall-clock time."
            raise ValueError(msg)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        anchor = self.absolute_start
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        return (when - anchor).total_seconds()

    @property
    def has_absolute_time(self) -> bool:
        return self.absolute_start is not None

    def format_relative(self, relative_seconds: float, *, always_show_hours: bool = False) -> str:
        """Human ``H:MM:SS.sss`` / ``MM:SS.sss`` formatting for axis ticks and titles."""
        sign = "-" if relative_seconds < 0 else ""
        x = abs(relative_seconds)
        hours = int(x // 3600)
        minutes = int((x // 60) % 60)
        seconds = x % 60
        if seconds == int(seconds):
            sec_str = f"{int(seconds):02d}"
        else:
            sec_str = f"{seconds:06.3f}"
        if hours or always_show_hours:
            return f"{sign}{hours}:{minutes:02d}:{sec_str}"
        return f"{sign}{minutes:02d}:{sec_str}"
