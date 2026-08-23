from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("librosa")
pytest.importorskip("ssqueezepy")
soundfile = pytest.importorskip("soundfile")

from audio_classifier_visualizer.core.classifier_output import ClassifierOutput
from audio_classifier_visualizer.core.labels import LabelBox
from audio_classifier_visualizer.render.spec import Track
from audio_classifier_visualizer.visualization import AudioVisualization


@pytest.fixture
def wav_path(tmp_path, tone, sr):
    path = tmp_path / "mono.wav"
    soundfile.write(path, tone, int(sr))
    return str(path)


def test_show_from_array_returns_figure(tone, sr):
    viz = AudioVisualization(y=tone, sr=sr)
    fig = viz.show(end_time=2.0, tracks=(Track.WAVEFORM, Track.STFT_SPECTROGRAM))
    assert fig is not None
    assert len(fig.axes) == 2


def test_show_from_file_only_loads_requested_slice(wav_path, sr):
    viz = AudioVisualization(audio_file=wav_path)
    fig = viz.show(start_time=1.0, end_time=2.0, tracks=(Track.WAVEFORM,))
    ax = fig.axes[0]
    assert ax.get_xlim() == pytest.approx((1.0, 2.0), abs=1e-2)


def test_show_with_classifier_output_and_labels(tone, sr):
    n_windows = 100
    probs = np.zeros((n_windows, 2))
    probs[:, 1] = np.linspace(0, 1, n_windows)
    probs[:, 0] = 1 - probs[:, 1]
    co = ClassifierOutput(probabilities=probs, feature_rate=n_windows / 5.0, class_labels=["background", "target"])
    labels = [LabelBox(start_time=1.0, end_time=1.5, low_freq=300, high_freq=600, text="test-box")]

    viz = AudioVisualization(y=tone, sr=sr, classifier_output=co, labels=labels)
    fig = viz.show(
        end_time=5.0,
        tracks=(Track.WAVEFORM, Track.STFT_SPECTROGRAM, Track.SIMILARITY_LINES, Track.CLASS_PROBABILITY_STACK),
        target_class="target",
    )
    assert len(fig.axes) == 4


def test_show_drops_classifier_tracks_when_no_classifier_output(tone, sr):
    viz = AudioVisualization(y=tone, sr=sr)
    fig = viz.show(end_time=2.0, tracks=(Track.WAVEFORM, Track.SIMILARITY_LINES, Track.CLASS_PROBABILITY_STACK))
    assert len(fig.axes) == 1


def test_slice_cache_reuses_loaded_signal(wav_path):
    viz = AudioVisualization(audio_file=wav_path)
    sig1 = viz._load_slice(0.0, 1.0)
    sig2 = viz._load_slice(0.0, 1.0)
    assert sig1 is sig2


def test_duration_does_not_require_full_load(wav_path, tone, sr):
    viz = AudioVisualization(audio_file=wav_path)
    assert viz.duration == pytest.approx(len(tone) / sr, abs=1e-3)
    assert viz._slice_cache == {}  # duration lookup must not have loaded any audio
