"""Per-window classifier output (binary or multiclass), decoupled from audio/plotting.

Notably: no torch dependency. The one thing the old code used torch for was
1-D linear interpolation to stretch classifier-rate scores up to spectrogram
resolution; ``numpy.interp`` does that in a couple of lines without dragging
in a multi-hundred-MB dependency (and without torchaudio/torchcodec churn).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ClassifierOutput:
    """Windowed classifier probabilities.

    ``probabilities`` has shape ``(n_windows, n_classes)``. For a binary
    classifier this is still shape ``(n_windows, 2)`` -- there's no separate
    "binary mode"; a 2-class multiclass output *is* the binary case.

    ``feature_rate`` is windows per second (e.g. sr/320 for AVES/HuBERT-style
    embeddings, but callers should measure their own model rather than assume
    a default -- getting this wrong silently misaligns everything downstream).
    """

    probabilities: np.ndarray  # (n_windows, n_classes)
    feature_rate: float
    class_labels: list[str]
    time_offset: float = 0.0
    """Absolute time (seconds, same coordinate frame as the audio it's paired with)
    that window index 0 actually starts at. Not necessarily what you sliced at:
    slicing can only happen at whole-window boundaries, so ``slice_time`` rounds
    the requested start to the nearest one and records the *actual* rounded time
    here -- callers that instead assumed the requested start_time exactly would
    accumulate up to half a window of drift on every slice.
    """

    def __post_init__(self) -> None:
        if self.probabilities.ndim != 2:
            msg = f"probabilities must be 2-D (n_windows, n_classes), got shape {self.probabilities.shape}"
            raise ValueError(msg)
        if self.probabilities.shape[1] != len(self.class_labels):
            msg = (
                f"probabilities has {self.probabilities.shape[1]} classes "
                f"but {len(self.class_labels)} class_labels were given"
            )
            raise ValueError(msg)

    @property
    def n_windows(self) -> int:
        return self.probabilities.shape[0]

    @property
    def n_classes(self) -> int:
        return self.probabilities.shape[1]

    def time_to_index(self, t: float) -> int:
        return round((t - self.time_offset) * self.feature_rate)

    def index_to_time(self, i: int) -> float:
        return self.time_offset + i / self.feature_rate

    def window_centers(self) -> np.ndarray:
        """Absolute seconds (this object's ``time_offset`` + each window's own
        center), in the same coordinate frame as the audio it's paired with --
        directly comparable to a ``VisualizationSpec``'s ``start_time``/``end_time``,
        no separate offset needs to be added by the caller.
        """
        return self.time_offset + np.arange(self.n_windows) / self.feature_rate + 0.5 / self.feature_rate

    def class_index(self, label_or_index: str | int) -> int:
        if isinstance(label_or_index, int):
            return label_or_index
        return self.class_labels.index(label_or_index)

    def slice_time(self, start_time: float, end_time: float) -> ClassifierOutput:
        start_idx = max(0, self.time_to_index(start_time))
        end_idx = min(self.n_windows, self.time_to_index(end_time))
        return ClassifierOutput(
            probabilities=self.probabilities[start_idx:end_idx].copy(),
            feature_rate=self.feature_rate,
            class_labels=self.class_labels,
            time_offset=self.index_to_time(start_idx),
        )

    def resample_class_to(
        self,
        class_index: int,
        target_length: int,
        duration: float | None = None,
        start_time: float = 0.0,
    ) -> np.ndarray:
        """Stretch one class's per-window scores to ``target_length`` samples (e.g. spectrogram width).

        Interpolates using each window's *center* time (``window_centers()``, same
        convention as the similarity-line/probability-stack tracks) against each
        destination sample's center time, so the colorized spectrogram lines up with
        those tracks instead of appearing shifted -- the old version placed window
        ``i`` at ``i/(n-1)`` of the span (ignoring feature_rate and the half-window
        center offset entirely), a different, incompatible time axis from
        ``window_centers()``.

        ``start_time``/``duration`` describe the real-world span the destination
        (``target_length``) axis covers, in the same coordinate frame as
        ``window_centers()`` -- e.g. a displayed audio slice's ``start_time`` and
        duration. Getting ``start_time`` right matters as much as getting
        ``window_centers()`` right: passing 0.0 for a slice that doesn't itself
        start at absolute time 0 reintroduces the same kind of misalignment this
        method exists to avoid.
        """
        scores = self.probabilities[:, class_index]
        if len(scores) == 0:
            return np.zeros(target_length)
        if duration is None:
            duration = self.n_windows / self.feature_rate
        src_t = self.window_centers()
        dst_t = start_time + (np.arange(target_length) + 0.5) / target_length * duration
        return np.interp(dst_t, src_t, scores)
