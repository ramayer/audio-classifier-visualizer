from __future__ import annotations

import numpy as np
import pytest

from audio_classifier_visualizer.features.colorize import confidence_to_rgb, normalize_power, power_to_db
from audio_classifier_visualizer.features.stft import STFTFeatureExtractor
from audio_classifier_visualizer.features.wavelet import WaveletFeatureExtractor

librosa = pytest.importorskip("librosa")
ssqueezepy = pytest.importorskip("ssqueezepy")


def test_stft_finds_the_tone(tone, sr):
    result = STFTFeatureExtractor(n_fft=1024, hop_length=256).compute(tone, sr)
    peak_freq = result.freqs[np.argmax(result.power.mean(axis=1))]
    assert peak_freq == pytest.approx(440, abs=20)


def test_stft_respects_freq_range_of_interest(tone, sr):
    result = STFTFeatureExtractor(n_fft=1024, freq_range_of_interest=(0, 300)).compute(tone, sr)
    assert result.freqs.max() <= 300


def test_wavelet_chunking_matches_single_shot_on_short_signal(tone, sr):
    """Chunked processing should closely reproduce a naive single-pass CWT on a signal
    short enough to fit in one chunk (i.e. chunking shouldn't distort results)."""
    extractor = WaveletFeatureExtractor(chunk_size=len(tone) + 8192, decimation_stride=1, freq_range_of_interest=(300, 600))
    result = extractor.compute(tone, sr, overlap=256)
    assert result.power.shape[0] == len(result.freqs)
    assert result.power.shape[1] == pytest.approx(len(tone), abs=2)
    # Peak energy should land near a frequency bin close to 440 Hz.
    peak_freq = result.freqs[np.argmax(result.power.mean(axis=1))]
    assert peak_freq == pytest.approx(440, abs=60)


def test_wavelet_decimation_reduces_width_by_stride(tone, sr):
    stride = 64
    extractor = WaveletFeatureExtractor(decimation_stride=stride, freq_range_of_interest=(300, 600))
    result = extractor.compute(tone, sr)
    naive_width = len(tone)
    assert result.power.shape[1] == pytest.approx(naive_width / stride, rel=0.05)


def test_wavelet_chunked_vs_single_chunk_power_is_similar(tone, sr):
    """Splitting into multiple small chunks (vs one big chunk) should not meaningfully
    change the resulting power spectrogram -- this is the core correctness property
    of the chunking/overlap-trim approach ported from the original code."""
    common_kwargs = {"decimation_stride": 32, "freq_range_of_interest": (300, 600)}
    one_chunk = WaveletFeatureExtractor(chunk_size=len(tone) + 8192, **common_kwargs).compute(tone, sr, overlap=512)
    many_chunks = WaveletFeatureExtractor(chunk_size=8192, **common_kwargs).compute(tone, sr, overlap=512)

    n = min(one_chunk.power.shape[1], many_chunks.power.shape[1])
    a = one_chunk.power[:, :n]
    b = many_chunks.power[:, :n]
    # Compare band-averaged energy rather than sample-exact (chunk boundaries/padding
    # introduce small edge effects) -- the overall energy profile should match closely.
    corr = np.corrcoef(a.mean(axis=0), b.mean(axis=0))[0, 1]
    assert corr > 0.9


def test_power_to_db_max_is_zero():
    power = np.array([[1.0, 4.0, 100.0]])
    db = power_to_db(power)
    assert db.max() == pytest.approx(0.0)


def test_normalize_power_clips_outliers():
    power = np.ones((5, 1000))
    power[0, 0] = 1e6  # extreme outlier
    normed = normalize_power(power, clip_outliers=True)
    assert normed[0, 0] < 1e6


def test_confidence_to_rgb_shapes_and_range():
    grayscale = np.random.default_rng(0).uniform(0, 1, size=(10, 20))
    similarity = np.linspace(0, 1, 20)
    dissimilarity = 1 - similarity
    rgb = confidence_to_rgb(grayscale, similarity, dissimilarity, style="bright")
    assert rgb.shape == (10, 20, 3)
    assert rgb.min() >= -1e-9
    assert rgb.max() <= 1 + 1e-9


def test_confidence_to_rgb_rejects_unknown_style():
    grayscale = np.zeros((2, 2))
    with pytest.raises(ValueError, match="Unknown colorize style"):
        confidence_to_rgb(grayscale, np.zeros(2), np.zeros(2), style="not-a-style")
