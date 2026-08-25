"""The notebook-facing entry point.

    viz = AudioVisualization(audio_file="day.wav", classifier_output=out, labels=boxes)
    viz.show(start_time=3600, end_time=3720, tracks=[Track.WAVEFORM, Track.WAVELET_SPECTROGRAM])

For a 24-hour file, pass ``audio_file`` and only the requested [start_time,
end_time) slice is ever read off disk and turned into features -- matching
the original design goal of being able to show an overview of an entire day
without loading the whole day into memory, or zoom into a minute of it
without recomputing anything for the rest of the day.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from audio_classifier_visualizer.core.audio_signal import AudioSignal
from audio_classifier_visualizer.core.classifier_output import ClassifierOutput
from audio_classifier_visualizer.core.labels import LabelBox, PointLabel
from audio_classifier_visualizer.features.stft import STFTFeatureExtractor
from audio_classifier_visualizer.features.wavelet import WaveletFeatureExtractor
from audio_classifier_visualizer.io.audio_loader import load_audio
from audio_classifier_visualizer.render.matplotlib_renderer import MatplotlibRenderer
from audio_classifier_visualizer.render.spec import DEFAULT_TRACKS, Renderer, Track, VisualizationSpec

if TYPE_CHECKING:
    pass

_CACHE_SIZE = 8


class AudioVisualization:
    def __init__(
        self,
        audio_file: str | None = None,
        y: np.ndarray | None = None,
        sr: float | None = None,
        *,
        absolute_start: datetime | None = None,
        classifier_output: ClassifierOutput | None = None,
        labels: list[LabelBox] | None = None,
        point_labels: list[PointLabel] | None = None,
        stft: STFTFeatureExtractor | None = None,
        wavelet: WaveletFeatureExtractor | None = None,
        renderer: Renderer | None = None,
        target_sr: float | None = None,
    ) -> None:
        if audio_file is None and (y is None or sr is None):
            msg = "Provide either audio_file, or both y and sr."
            raise ValueError(msg)
        self._audio_file = audio_file
        self._target_sr = target_sr
        self._preloaded_signal = None if audio_file else AudioSignal(samples=y, sr=sr, source_path=None)
        if target_sr is not None and self._preloaded_signal is not None and target_sr != sr:
            import librosa

            resampled = librosa.resample(self._preloaded_signal.samples, orig_sr=sr, target_sr=target_sr)
            self._preloaded_signal = AudioSignal(samples=resampled, sr=target_sr, source_path=None)
        if absolute_start is not None and self._preloaded_signal is not None:
            from audio_classifier_visualizer.core.time_axis import TimeAxis

            self._preloaded_signal.time_axis = TimeAxis(absolute_start=absolute_start)
        self._absolute_start = absolute_start
        self.classifier_output = classifier_output
        self.labels = labels or []
        self.point_labels = point_labels or []
        self.stft = stft or STFTFeatureExtractor()
        self.wavelet = wavelet or WaveletFeatureExtractor()
        self.renderer = renderer or MatplotlibRenderer()
        self._slice_cache: dict[tuple, AudioSignal] = {}
        self._duration: float | None = None

    @property
    def duration(self) -> float:
        """Full-file duration in seconds, without loading samples for a file on disk."""
        if self._duration is None:
            if self._preloaded_signal is not None:
                self._duration = self._preloaded_signal.duration
            else:
                import soundfile as sf

                info = sf.info(self._audio_file)
                self._duration = info.frames / info.samplerate
        return self._duration

    def _load_slice(self, start_time: float, end_time: float) -> AudioSignal:
        key = (self._audio_file, round(start_time, 6), round(end_time, 6))
        if key in self._slice_cache:
            return self._slice_cache[key]
        if self._audio_file is not None:
            signal = load_audio(
                self._audio_file,
                start_time=start_time,
                end_time=end_time,
                absolute_start=self._absolute_start,
                target_sr=self._target_sr,
            )
        else:
            signal = self._preloaded_signal.slice_time(start_time, end_time)
        if len(self._slice_cache) >= _CACHE_SIZE:
            self._slice_cache.pop(next(iter(self._slice_cache)))
        self._slice_cache[key] = signal
        return signal

    def _effective_sr(self) -> float:
        """The sample rate audio actually gets processed at -- after any target_sr
        resampling -- needed to convert a context padding requirement (in samples)
        into seconds before we've necessarily loaded anything yet."""
        if self._target_sr is not None:
            return self._target_sr
        if self._preloaded_signal is not None:
            return self._preloaded_signal.sr
        import soundfile as sf

        return float(sf.info(self._audio_file).samplerate)

    def _context_seconds(self) -> float:
        """How much real (or, failing that, reflect/edge-padded) audio to fetch on
        each side of the requested display range before computing STFT/CWT, so
        that analysis windows near the display boundary have genuine neighboring
        samples instead of the feature extractors' own internal zero-padding.

        Sized to the larger of what STFT (half its FFT window) and the wavelet
        extractor (its own chunk-overlap) need -- both are already-meaningful
        quantities the user may have tuned, not new knobs to configure separately.
        """
        sr = self._effective_sr()
        stft_context = (self.stft.n_fft / 2) / sr
        wavelet_context = self.wavelet.overlap / sr
        return max(stft_context, wavelet_context)

    def _load_context_audio(self, start_time: float, end_time: float, context_seconds: float) -> AudioSignal:
        """Load [start_time - context_seconds, end_time + context_seconds), using real
        neighboring samples where the source has them. Where it doesn't (the
        request runs past the true start/end of the file or array), pad with
        reflected audio instead of the zero-silence a naive slice would otherwise
        imply -- a hard edge into silence is itself a spectrogram artifact, just a
        different one than the zero-padding this whole mechanism exists to avoid.
        Falls back from reflect to edge-repeat if there isn't even enough real data
        to reflect (only possible for extremely short clips).
        """
        want_left = start_time - context_seconds
        want_right = end_time + context_seconds
        avail_left = max(0.0, want_left)
        avail_right = min(self.duration, want_right)

        signal = self._load_slice(avail_left, avail_right)

        pad_left = round(max(0.0, avail_left - want_left) * signal.sr)
        pad_right = round(max(0.0, want_right - avail_right) * signal.sr)
        if pad_left == 0 and pad_right == 0:
            return signal

        samples = signal.samples
        try:
            padded = np.pad(samples, ((0, 0), (pad_left, pad_right)), mode="reflect")
        except ValueError:
            # reflect requires each pad width < the axis length; only possible for
            # a clip shorter than the context window itself.
            padded = np.pad(samples, ((0, 0), (pad_left, pad_right)), mode="edge")
        return AudioSignal(samples=padded, sr=signal.sr, source_path=signal.source_path)

    def build_spec(
        self,
        *,
        start_time: float = 0.0,
        end_time: float | None = None,
        tracks: tuple[Track, ...] = DEFAULT_TRACKS,
        target_class: str | int | None = None,
        colorize_style: str = "bright",
        per_channel_normalize: bool = True,
        clip_outliers: bool = True,
        title: str = "",
    ) -> VisualizationSpec:
        if end_time is None:
            end_time = self.duration
        signal = self._load_slice(start_time, end_time)

        classifier_slice = None
        if self.classifier_output is not None:
            classifier_slice = self.classifier_output.slice_time_with_margin(start_time, end_time)

        labels_in_range = [box for box in self.labels if box.overlaps(start_time, end_time)]
        point_labels_in_range = [p for p in self.point_labels if p.in_range(start_time, end_time)]

        context_seconds = self._context_seconds()
        context_audio = self._load_context_audio(start_time, end_time, context_seconds)

        return VisualizationSpec(
            audio=signal,
            tracks=tuple(tracks),
            title=title,
            display_offset=start_time,
            context_audio=context_audio,
            context_seconds=context_seconds,
            labels=labels_in_range,
            point_labels=point_labels_in_range,
            classifier_output=classifier_slice,
            target_class=target_class,
            stft=self.stft,
            wavelet=self.wavelet,
            colorize_style=colorize_style,
            per_channel_normalize=per_channel_normalize,
            clip_outliers=clip_outliers,
        )

    def show(
        self,
        *,
        start_time: float = 0.0,
        end_time: float | None = None,
        tracks: tuple[Track, ...] = DEFAULT_TRACKS,
        target_class: str | int | None = None,
        colorize_style: str = "bright",
        per_channel_normalize: bool = True,
        clip_outliers: bool = True,
        title: str = "",
        width: float = 19.2,
        height: float = 12.8,
        save_file: str | None = None,
    ):
        spec = self.build_spec(
            start_time=start_time,
            end_time=end_time,
            tracks=tracks,
            target_class=target_class,
            colorize_style=colorize_style,
            per_channel_normalize=per_channel_normalize,
            clip_outliers=clip_outliers,
            title=title,
        )
        return self.renderer.render(spec, width=width, height=height, save_file=save_file)
