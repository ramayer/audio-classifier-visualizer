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


def test_target_sr_resamples_file_audio(wav_path, sr):
    viz = AudioVisualization(audio_file=wav_path, target_sr=sr / 2)
    sig = viz._load_slice(0.0, 1.0)
    assert sig.sr == pytest.approx(sr / 2)


def test_target_sr_resamples_preloaded_array(tone, sr):
    viz = AudioVisualization(y=tone, sr=sr, target_sr=sr / 2)
    fig = viz.show(end_time=1.0, tracks=(Track.WAVEFORM,))
    assert fig is not None


def test_duration_does_not_require_full_load(wav_path, tone, sr):
    viz = AudioVisualization(audio_file=wav_path)
    assert viz.duration == pytest.approx(len(tone) / sr, abs=1e-3)
    assert viz._slice_cache == {}  # duration lookup must not have loaded any audio


def test_wavelet_yaxis_labeled_high_to_low(tone, sr):
    """High frequency must be on top (small index), with actual Hz labels -- not the
    raw row-index integers imshow would otherwise show by default."""
    viz = AudioVisualization(y=tone, sr=sr)
    fig = viz.show(end_time=1.0, tracks=(Track.WAVELET_SPECTROGRAM,))
    ax = fig.axes[0]
    ticklabels = [t.get_text() for t in ax.get_yticklabels()]
    values = [float(t) for t in ticklabels if t]
    assert len(values) >= 2
    assert values == sorted(values, reverse=True)  # highest Hz label first (top)


def test_wavelet_label_box_lands_within_axis_range(tone, sr):
    from audio_classifier_visualizer.core.labels import LabelBox
    from audio_classifier_visualizer.features.wavelet import WaveletFeatureExtractor
    from matplotlib.patches import Rectangle

    extractor = WaveletFeatureExtractor(freq_range_of_interest=(300, 600))
    box = LabelBox(start_time=0.1, end_time=0.3, low_freq=400, high_freq=500, text="tone")
    viz = AudioVisualization(y=tone, sr=sr, labels=[box], wavelet=extractor)
    fig = viz.show(end_time=1.0, tracks=(Track.WAVELET_SPECTROGRAM,))
    ax = fig.axes[0]
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert rects
    # index-based axis: y-origin should be a small row-index number, not a raw Hz value.
    for r in rects:
        assert 0 <= r.get_y() <= 10_000  # generous bound; specifically NOT ~400-500 landing far off an index axis
        assert r.get_y() < 1000  # sanity: wavelet extractor here has well under 1000 rows


def test_per_channel_normalize_flag_reaches_spec(tone, sr):
    viz = AudioVisualization(y=tone, sr=sr)
    spec_on = viz.build_spec(end_time=1.0, per_channel_normalize=True)
    spec_off = viz.build_spec(end_time=1.0, per_channel_normalize=False)
    assert spec_on.per_channel_normalize is True
    assert spec_off.per_channel_normalize is False


def test_per_channel_normalize_changes_rendered_power(tone, sr):
    """Not just plumbing -- confirm the flag actually changes what gets drawn."""
    viz = AudioVisualization(y=tone, sr=sr)
    fig_on = viz.show(end_time=1.0, tracks=(Track.STFT_SPECTROGRAM,), per_channel_normalize=True)
    fig_off = viz.show(end_time=1.0, tracks=(Track.STFT_SPECTROGRAM,), per_channel_normalize=False)
    img_on = fig_on.axes[0].get_images()[0].get_array()
    img_off = fig_off.axes[0].get_images()[0].get_array()
    assert not np.allclose(np.asarray(img_on), np.asarray(img_off))


def test_clip_outliers_flag_reaches_spec(tone, sr):
    viz = AudioVisualization(y=tone, sr=sr)
    spec_on = viz.build_spec(end_time=1.0, clip_outliers=True)
    spec_off = viz.build_spec(end_time=1.0, clip_outliers=False)
    assert spec_on.clip_outliers is True
    assert spec_off.clip_outliers is False

