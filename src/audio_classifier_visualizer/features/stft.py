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
