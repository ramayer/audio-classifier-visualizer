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


@dataclass(slots=True)
class PointLabel:
    """A single marked point on the waveform: a (time, amplitude) pair, drawn as a
    dot directly on the WAVEFORM track. Replaces the original's bare ``(t, h)``
    tuple (``point_labels: list[tuple[float, float]]``) with named fields, same
    reasoning as ``LabelBox`` replacing positional tuples.

    This was present in the pre-1.0 code (as far back as v0.0.7) but was dropped
    in the initial 1.0 rewrite -- not a deliberate cut, just missed.
    """

    time: float
    amplitude: float
    text: str = ""
    color: str = "red"

    def in_range(self, start_time: float, end_time: float) -> bool:
        return start_time <= self.time <= end_time