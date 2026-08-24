"""The audio buffer itself, decoupled from how it was loaded or how it will be drawn."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from audio_classifier_visualizer.core.time_axis import TimeAxis


@dataclass(slots=True)
class AudioSignal:
    """In-memory audio buffer.

    ``samples`` is always ``(n_channels, n_samples)``, even for mono, so that
    surround-sound audio isn't a bolted-on special case later -- it's just
    ``n_channels > 1``.
    """

    samples: np.ndarray  # shape: (n_channels, n_samples)
    sr: float
    time_axis: TimeAxis = field(default_factory=TimeAxis)
    source_path: str | None = None

    def __post_init__(self) -> None:
        if self.samples.ndim == 1:
            self.samples = self.samples[np.newaxis, :]
        if self.samples.ndim != 2:
            msg = f"AudioSignal.samples must be 1-D or 2-D, got shape {self.samples.shape}"
            raise ValueError(msg)

    @property
    def n_channels(self) -> int:
        return self.samples.shape[0]

    @property
    def n_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sr

    def channel(self, index: int) -> np.ndarray:
        return self.samples[index]

    def as_mono(self) -> np.ndarray:
        """Average across channels. Convenience for renderers/features that don't care about channels."""
        return self.samples.mean(axis=0)

    def slice_time(self, start_time: float, end_time: float | None = None) -> AudioSignal:
        """Return a new AudioSignal covering [start_time, end_time) of *this* buffer.

        The returned signal's own time axis is still relative to its own start
        (0 == start_time of the slice); if this signal has an absolute_start,
        the slice's absolute_start is shifted accordingly so downstream code
        never has to remember an offset.
        """
        end_time = self.duration if end_time is None else end_time
        start_idx = max(0, round(start_time * self.sr))
        end_idx = min(self.n_samples, round(end_time * self.sr))
        new_axis = self.time_axis
        if self.time_axis.has_absolute_time:
            from audio_classifier_visualizer.core.time_axis import TimeAxis as _TA

            new_axis = _TA(absolute_start=self.time_axis.to_absolute(start_time))
        return AudioSignal(
            samples=self.samples[:, start_idx:end_idx].copy(),
            sr=self.sr,
            time_axis=new_axis,
            source_path=self.source_path,
        )
