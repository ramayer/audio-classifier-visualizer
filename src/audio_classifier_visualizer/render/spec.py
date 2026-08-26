"""Declarative description of what to draw, independent of how it gets drawn.

Any renderer (matplotlib today; a Bokeh/HoloViews-based interactive one
later) implements ``Renderer.render(spec)`` against the same ``VisualizationSpec``.
Adding a second renderer should require zero changes here or in
core/features -- that's the point of pulling this apart from the old
single matplotlib-entangled class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from audio_classifier_visualizer.core.audio_signal import AudioSignal
from audio_classifier_visualizer.core.classifier_output import ClassifierOutput
from audio_classifier_visualizer.core.labels import LabelBox, PointLabel
from audio_classifier_visualizer.features.stft import STFTFeatureExtractor
from audio_classifier_visualizer.features.wavelet import WaveletFeatureExtractor


class Track(Enum):
    WAVEFORM = "waveform"
    STFT_SPECTROGRAM = "stft_spectrogram"
    WAVELET_SPECTROGRAM = "wavelet_spectrogram"
    CLASS_PROBABILITIES = "class_probabilities"
    CLASS_HEATMAP = "class_heatmap"

    # Deprecated: kept so existing code keeps working. SIMILARITY_LINES ->
    # CLASS_PROBABILITIES(classes=[target_class]) (drops the redundant inverse
    # line -- define two classes that sum to 1 if you actually want that back).
    # CLASS_PROBABILITY_STACK -> CLASS_PROBABILITIES(all classes), unstacked --
    # stacking made it hard to read any one class's own trend independent of
    # whatever was stacked below it. See MatplotlibRenderer._draw_track.
    SIMILARITY_LINES = "similarity_lines"
    CLASS_PROBABILITY_STACK = "class_probability_stack"


DEFAULT_TRACKS = (Track.WAVEFORM, Track.STFT_SPECTROGRAM)


@dataclass(slots=True)
class VisualizationSpec:
    """``audio`` is always relative to its own start (sample 0 == relative time 0),
    since that's what slicing/loading naturally produces. ``display_offset`` is the
    real-world-within-the-file time that sample 0 corresponds to, so that zooming
    into e.g. [3600, 3720) of a 24-hour file still shows an axis reading ~1:00:00,
    not 0:00:00 -- the display range is a property of *where this slice sits in the
    larger recording*, not of the slice's own buffer, which is why it's tracked here
    rather than folded into AudioSignal.
    """

    audio: AudioSignal
    tracks: tuple[Track, ...] = DEFAULT_TRACKS
    title: str = ""
    display_offset: float = 0.0

    # Wider-than-``audio`` buffer used only for spectrogram computation (STFT/CWT),
    # so that analysis windows near the display boundary have real neighboring
    # samples to work with instead of the feature extractors' own internal
    # zero-padding -- see AudioVisualization._load_context_audio for how this gets
    # built (real audio where available, reflect/edge-padded at true file
    # boundaries). None means "no context available" (e.g. a hand-built spec) --
    # renderers must fall back to computing directly on ``audio`` in that case.
    context_audio: AudioSignal | None = None
    context_seconds: float = 0.0

    labels: list[LabelBox] = field(default_factory=list)
    point_labels: list[PointLabel] = field(default_factory=list)
    classifier_output: ClassifierOutput | None = None
    target_class: str | int | None = None
    classes: list[str] | None = None
    """Explicit class selection for CLASS_PROBABILITIES/CLASS_HEATMAP. If both this
    and top_k are None, all classes in classifier_output are shown (uncapped) --
    that's what the deprecated CLASS_PROBABILITY_STACK alias relies on to preserve
    old behavior; direct use of the new tracks should usually set top_k instead.
    """
    top_k: int | None = None
    """Show only the top_k classes by peak probability within the *currently
    displayed* window (recomputed on zoom, not fixed at load time) -- ignored if
    ``classes`` is given explicitly."""

    stft: STFTFeatureExtractor = field(default_factory=STFTFeatureExtractor)
    wavelet: WaveletFeatureExtractor = field(default_factory=WaveletFeatureExtractor)

    colorize_style: str = "bright"
    per_channel_normalize: bool = True
    clip_outliers: bool = True

    @property
    def start_time(self) -> float:
        return self.display_offset

    @property
    def end_time(self) -> float:
        return self.display_offset + self.audio.duration


class Renderer(Protocol):
    def render(
        self,
        spec: VisualizationSpec,
        *,
        width: float = 19.2,
        height: float = 12.8,
        save_file: str | None = None,
    ) -> object:
        """Draw the spec. Returns a renderer-specific handle (e.g. a matplotlib Figure)."""
        ...
