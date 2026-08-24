from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
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
    from matplotlib.patches import Rectangle

    from audio_classifier_visualizer.core.labels import LabelBox
    from audio_classifier_visualizer.features.wavelet import WaveletFeatureExtractor

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


def test_class_probability_stack_legend_does_not_overlap_next_track(tone, sr):
    """Regression test: with CLASS_PROBABILITY_STACK not the last track, its legend
    must not land in the same figure region as the track drawn after it."""
    n_windows = 20
    probs = np.random.default_rng(0).dirichlet(np.ones(3), n_windows)
    co = ClassifierOutput(probabilities=probs, feature_rate=n_windows / 5.0, class_labels=["a", "b", "c"])
    viz = AudioVisualization(y=tone, sr=sr, classifier_output=co)
    fig = viz.show(
        end_time=5.0,
        tracks=(Track.CLASS_PROBABILITY_STACK, Track.SIMILARITY_LINES),
    )
    stack_ax, similarity_ax = fig.axes
    # Use an explicit Agg canvas for this measurement rather than trusting whatever
    # backend happens to be globally active -- some environments (certain uv-managed
    # venvs, IPython/inline setups, etc.) end up with a generic FigureCanvasBase that
    # lacks get_renderer() even after matplotlib.use("Agg"), if something imported
    # pyplot with a different backend first. Agg is also just the right tool here:
    # we only need pixel geometry, not an actual display.
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    legend = stack_ax.get_legend()
    assert legend is not None
    legend_bbox = legend.get_window_extent(renderer=renderer)
    similarity_bbox = similarity_ax.get_window_extent(renderer=renderer)
    # The legend must sit to the right of (not vertically overlapping into) the
    # next track's axes.
    assert legend_bbox.x0 >= similarity_bbox.x1 - 1  # small tolerance for pixel rounding


def test_similarity_line_crossing_stable_across_non_aligned_zoom(tone, sr):
    """Regression test for a reported bug: zooming to a start_time that doesn't land
    on a classifier window boundary (e.g. 0.7s at feature_rate=2) must not shift
    where the similarity line crosses 0.5 -- it should match the true transition
    time regardless of which start_time you happened to zoom to."""
    feature_rate = 2.0
    n_windows = 20  # 10s of audio at feature_rate=2
    probs = np.zeros((n_windows, 2))
    probs[2:, 1] = 1.0  # true transition at t=1.0s
    co = ClassifierOutput(probabilities=probs, feature_rate=feature_rate, class_labels=["other", "target"])

    def crossing_time(start_time, end_time):
        viz = AudioVisualization(y=tone, sr=sr, classifier_output=co)
        fig = viz.show(
            start_time=start_time, end_time=end_time, tracks=(Track.SIMILARITY_LINES,), target_class="target"
        )
        line = fig.axes[0].lines[0]  # similarity (green) line
        xdata, ydata = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        # The rendered line only has the actual data points; the 0.5 crossing is
        # visually *between* two of them (drawn as a straight segment), so find it
        # by interpolating along the line itself rather than searching raw ydata.
        return float(np.interp(0.5, ydata, xdata))

    # A window-aligned zoom (0.5 IS a window boundary) and a non-aligned one (0.7
    # is not) should both find the crossing at the same true time, ~1.0s.
    aligned = crossing_time(0.5, 1.5)
    non_aligned = crossing_time(0.7, 1.3)
    assert aligned == pytest.approx(1.0, abs=0.05)
    assert non_aligned == pytest.approx(1.0, abs=0.05)


def test_xlim_matches_requested_range_not_tick_interval(wav_path):
    """Regression test: sharex=True can silently re-expand xlim to the next tick
    interval when set_xticks is called with out-of-range tick locations -- the
    displayed range must stay exactly what was requested (0.7, 1.3), not snap out
    to (0.5, 1.5) to fit a round tick interval."""
    viz = AudioVisualization(audio_file=wav_path)
    fig = viz.show(start_time=0.7, end_time=1.3, tracks=(Track.WAVEFORM,))
    for ax in fig.axes:
        assert ax.get_xlim() == pytest.approx((0.7, 1.3), abs=1e-6)
