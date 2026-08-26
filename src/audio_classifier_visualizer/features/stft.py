"""Standard STFT spectrogram computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class STFTResult:
    complex_spec: np.ndarray  # (n_freqs, n_frames)
    power: np.ndarray  # (n_freqs, n_frames)
    freqs: np.ndarray  # (n_freqs,)


class STFTFeatureExtractor:
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int | None = None,
        freq_range_of_interest: tuple[float, float] | None = None,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.freq_range_of_interest = freq_range_of_interest

    def with_overrides(self, **kwargs) -> STFTFeatureExtractor:
        """A shallow copy with the given fields changed -- for a one-off .show()
        call without mutating the shared extractor (which persists across calls
        and is what direct attribute assignment, e.g. ``viz.stft.n_fft = 512``,
        is for). Raises AttributeError on an unknown field name rather than
        silently creating a new, unused attribute (most likely a typo)."""
        import copy

        new = copy.copy(self)
        for key, value in kwargs.items():
            if not hasattr(new, key):
                msg = f"{type(self).__name__} has no attribute {key!r} to override"
                raise AttributeError(msg)
            setattr(new, key, value)
        return new

    def compute(self, y: np.ndarray, sr: float) -> STFTResult:
        import librosa

        spec = librosa.stft(y, n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        power = np.abs(spec) ** 2
        if self.freq_range_of_interest:
            lo, hi = self.freq_range_of_interest
            keep = (freqs >= lo) & (freqs <= hi)
            freqs, spec, power = freqs[keep], spec[keep], power[keep]
        return STFTResult(complex_spec=spec, power=power, freqs=freqs)
