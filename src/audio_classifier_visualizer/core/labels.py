"""Time/frequency-boxed annotations (e.g. Raven selection table rows)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LabelBox:
    """A single annotation: a rectangle in (time, frequency) space.

    Replaces the old ``dataclasses.astuple(row)`` positional-unpacking
    pattern (``bt, et, lf, hf, dur, fn, tags, notes, tag1, tag2, score,
    raven_file``) with named, self-documenting fields.
    """

    start_time: float
    end_time: float
    low_freq: float
    high_freq: float
    text: str = ""
    score: float | None = None
    tags: list[str] = field(default_factory=list)
    source_file: str | None = None

    def overlaps(self, start_time: float, end_time: float) -> bool:
        return not (self.end_time < start_time or self.start_time > end_time)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
