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
        return round(t * self.feature_rate)

    def index_to_time(self, i: int) -> float:
        return i / self.feature_rate

    def window_centers(self) -> np.ndarray:
        """Seconds relative to *this object's own* window 0 -- callers displaying a
        slice alongside audio that has its own display offset (see
        ``VisualizationSpec.display_offset``) must add that offset themselves.
        """
        return np.arange(self.n_windows) / self.feature_rate + 0.5 / self.feature_rate

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
        )

    def resample_class_to(self, class_index: int, target_length: int) -> np.ndarray:
        """Stretch one class's per-window scores to ``target_length`` samples (e.g. spectrogram width)."""
        scores = self.probabilities[:, class_index]
        if len(scores) == 0:
            return np.zeros(target_length)
        if len(scores) == target_length:
            return scores.astype(float)
        src_x = np.linspace(0.0, 1.0, num=len(scores))
        dst_x = np.linspace(0.0, 1.0, num=target_length)
        return np.interp(dst_x, src_x, scores)
