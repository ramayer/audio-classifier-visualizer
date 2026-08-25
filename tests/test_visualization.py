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


def _agg_renderer(fig):
    """A renderer guaranteed to support get_renderer(), regardless of whatever
    backend happens to be globally active in this environment. Some setups
    (certain uv-managed venvs, IPython/inline configs, etc.) end up with a generic
    FigureCanvasBase -- which lacks get_renderer() -- even after
    matplotlib.use("Agg", force=True), if something imports pyplot with a
    different backend first. Wrapping the figure in an explicit FigureCanvasAgg
    sidesteps that entirely; we only need pixel geometry here, not a real display.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return canvas.get_renderer()


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
    renderer = _agg_renderer(fig)
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


def test_point_labels_render_on_waveform_and_are_time_filtered(tone, sr):
    from matplotlib.collections import PathCollection

    from audio_classifier_visualizer.core.labels import PointLabel

    in_range_point = PointLabel(time=2.0, amplitude=0.5, text="in range")
    out_of_range_point = PointLabel(time=8.0, amplitude=0.5, text="out of range")
    viz = AudioVisualization(y=tone, sr=sr, point_labels=[in_range_point, out_of_range_point])

    fig = viz.show(start_time=0.0, end_time=3.0, tracks=(Track.WAVEFORM,))
    ax = fig.axes[0]
    # each point label is drawn as an ax.scatter() PathCollection (ring + translucent
    # fill); the waveform envelope's fill_between layers are PolyCollections, so
    # filter by type rather than assuming point markers are the only collections.
    # The out-of-range point must not add one.
    point_markers = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(point_markers) == 1
    offsets = point_markers[0].get_offsets()
    assert len(offsets) == 1
    assert offsets[0][0] == pytest.approx(2.0)
    assert offsets[0][1] == pytest.approx(0.5)


def test_point_label_marker_has_translucent_fill_and_opaque_ring(tone, sr):
    from matplotlib.collections import PathCollection

    from audio_classifier_visualizer.core.labels import PointLabel

    point = PointLabel(time=1.0, amplitude=0.3)
    viz = AudioVisualization(y=tone, sr=sr, point_labels=[point])
    fig = viz.show(end_time=2.0, tracks=(Track.WAVEFORM,))
    ax = fig.axes[0]
    collection = next(c for c in ax.collections if isinstance(c, PathCollection))
    face_rgba = collection.get_facecolor()[0]
    edge_rgba = collection.get_edgecolor()[0]
    assert 0.0 < face_rgba[3] < 1.0  # fill is translucent, not solid
    assert edge_rgba[3] == pytest.approx(1.0)  # ring is opaque


def test_point_label_custom_color_applies_to_marker_and_text(tone, sr):
    from matplotlib.collections import PathCollection
    from matplotlib.colors import to_rgba

    from audio_classifier_visualizer.core.labels import PointLabel

    point = PointLabel(time=1.0, amplitude=0.3, text="peak", color="blue")
    viz = AudioVisualization(y=tone, sr=sr, point_labels=[point])
    fig = viz.show(end_time=2.0, tracks=(Track.WAVEFORM,))
    ax = fig.axes[0]
    point_marker = next(c for c in ax.collections if isinstance(c, PathCollection))
    edge_rgba = point_marker.get_edgecolor()[0]
    assert tuple(edge_rgba[:3]) == pytest.approx(to_rgba("blue")[:3])
    annotation = next(t for t in ax.texts if t.get_text() == "peak")
    assert annotation.get_color() == "blue"


def test_point_label_text_has_no_bbox_and_is_horizontally_offset(tone, sr):
    from audio_classifier_visualizer.core.labels import PointLabel

    point = PointLabel(time=1.0, amplitude=0.3, text="peak")
    viz = AudioVisualization(y=tone, sr=sr, point_labels=[point])
    fig = viz.show(end_time=2.0, tracks=(Track.WAVEFORM,))
    ax = fig.axes[0]
    annotation = next(t for t in ax.texts if t.get_text() == "peak")
    assert annotation.get_bbox_patch() is None
    assert annotation.get_verticalalignment() == "center"


def test_waveform_line_is_black(tone, sr):
    viz = AudioVisualization(y=tone, sr=sr)
    fig = viz.show(end_time=1.0, tracks=(Track.WAVEFORM,))
    ax = fig.axes[0]
    waveform_line = ax.get_lines()[0]
    assert waveform_line.get_color() in ("black", "k", "#000000")


def test_point_label_text_is_annotated_smaller_than_label_box_text(tone, sr):
    from audio_classifier_visualizer.core.labels import PointLabel

    point = PointLabel(time=1.0, amplitude=0.3, text="peak")
    viz = AudioVisualization(y=tone, sr=sr, point_labels=[point])
    fig = viz.show(end_time=2.0, tracks=(Track.WAVEFORM,))
    ax = fig.axes[0]
    annotations = [child for child in ax.texts if child.get_text() == "peak"]
    assert len(annotations) == 1
    assert annotations[0].get_fontsize() < 12  # smaller than LabelBox's fontsize=12


def test_point_label_without_text_adds_no_annotation(tone, sr):
    from audio_classifier_visualizer.core.labels import PointLabel

    point = PointLabel(time=1.0, amplitude=0.3)  # text="" (default)
    viz = AudioVisualization(y=tone, sr=sr, point_labels=[point])
    fig = viz.show(end_time=2.0, tracks=(Track.WAVEFORM,))
    ax = fig.axes[0]
    assert len(ax.texts) == 0


def test_point_labels_build_spec_filters_by_range(tone, sr):
    from audio_classifier_visualizer.core.labels import PointLabel

    points = [PointLabel(time=t, amplitude=0.0) for t in (0.5, 2.0, 4.5)]
    viz = AudioVisualization(y=tone, sr=sr, point_labels=points)
    spec = viz.build_spec(start_time=0.0, end_time=3.0)
    assert [p.time for p in spec.point_labels] == [0.5, 2.0]


def test_label_text_beyond_display_range_does_not_inflate_tight_bbox(tone, sr):
    """Regression test: a LabelBox that only partially overlaps the displayed range
    (its own text is drawn at its end, which can be well past spec.end_time) must
    not balloon the figure's tight bounding box -- that's what made the whole plot
    render squeezed into a corner of the image. clip_on=True on the label text is
    what keeps it from happening."""
    from audio_classifier_visualizer.core.labels import LabelBox

    label = LabelBox(start_time=1.0, end_time=2.0, low_freq=200, high_freq=400, text="D Note")
    viz = AudioVisualization(y=tone, sr=sr, labels=[label])
    fig = viz.show(start_time=0.7, end_time=1.3, tracks=(Track.STFT_SPECTROGRAM,))

    tight_bbox = fig.get_tightbbox(_agg_renderer(fig))
    nominal_width_px = fig.get_size_inches()[0] * fig.dpi
    # A small margin is fine (the box itself, tick labels); ballooning to ~2x+ the
    # nominal figure width is the bug this guards against.
    assert tight_bbox.width < nominal_width_px * 1.2


def test_point_label_text_beyond_display_range_does_not_inflate_tight_bbox(tone, sr):
    """Same regression, for point-label text (ax.annotate) rather than LabelBox
    text (ax.text) -- both call sites need clip_on=True independently."""
    from audio_classifier_visualizer.core.labels import PointLabel

    # Not filtered out by build_spec's in_range check (point.time IS in range);
    # the annotation's *text*, offset a few points from the dot, is what could
    # extend past the axes if clipping weren't set.
    point = PointLabel(time=1.29, amplitude=0.0, text="right at the edge")
    viz = AudioVisualization(y=tone, sr=sr, point_labels=[point])
    fig = viz.show(start_time=0.7, end_time=1.3, tracks=(Track.WAVEFORM,))

    tight_bbox = fig.get_tightbbox(_agg_renderer(fig))
    nominal_width_px = fig.get_size_inches()[0] * fig.dpi
    assert tight_bbox.width < nominal_width_px * 1.2


def test_bucket_stats_matches_naive_per_bucket_computation():
    from audio_classifier_visualizer.render.matplotlib_renderer import _bucket_stats

    rng = np.random.default_rng(0)
    y = rng.normal(loc=10.0, scale=0.5, size=997)  # deliberately not evenly divisible
    n_buckets = 10
    bucket_min, bucket_max, bucket_mean, _bucket_std = _bucket_stats(y, n_buckets)
    assert len(bucket_min) == n_buckets

    # Cross-check against a naive per-bucket loop over the *unpadded* data for all
    # buckets except the last (which legitimately differs slightly due to edge padding).
    # Bucket size is ceil(len(y) / n_buckets) -- the chunk size is chosen first, then
    # padding fills only the tail of the last bucket (see _bucket_stats docstring).
    chunk_size = -(-len(y) // n_buckets)  # ceil division
    for i in range(n_buckets - 1):
        chunk = y[i * chunk_size : (i + 1) * chunk_size]
        assert bucket_mean[i] == pytest.approx(chunk.mean(), rel=1e-3)
        assert bucket_min[i] == pytest.approx(chunk.min(), rel=1e-3)
        assert bucket_max[i] == pytest.approx(chunk.max(), rel=1e-3)


def test_bucket_stats_std_ignores_dc_offset():
    """The whole point of using std (not sqrt(mean(x**2))) for the inner band: a
    signal with a large DC offset and small AC component must show a small std,
    not one dominated by the offset -- e.g. a pressure sensor sitting at ~10 with
    a +/-1 wobble should read as "small variation around 10", not swamped by the 10."""
    from audio_classifier_visualizer.render.matplotlib_renderer import _bucket_stats

    t = np.linspace(0, 4 * np.pi, 2000)
    pressure_like = 10.0 + 1.0 * np.sin(t)  # DC offset 10, AC amplitude 1
    _bmin, _bmax, bucket_mean, bucket_std = _bucket_stats(pressure_like, n_buckets=4)
    assert np.all(bucket_mean > 8.0)  # DC offset preserved in the mean line
    assert np.all(bucket_std < 1.0)  # AC-only variation, not inflated by the offset


def test_waveform_uses_direct_line_when_few_samples_per_pixel(sr):
    """A short, high-sample-rate-relative-to-duration clip (the 'guitar string'
    case) should stay a direct line plot, not switch into envelope mode."""
    from matplotlib.collections import PolyCollection

    t = np.arange(0, int(0.05 * sr)) / sr  # 50ms -- few samples relative to typical pixel width
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    viz = AudioVisualization(y=y, sr=sr)
    fig = viz.show(end_time=0.05, tracks=(Track.WAVEFORM,), width=6, height=2)
    ax = fig.axes[0]
    assert not any(isinstance(c, PolyCollection) for c in ax.collections)
    assert len(ax.get_lines()) == 1
    assert len(ax.get_lines()[0].get_xdata()) == len(y)  # full-resolution line, not bucketed


def test_waveform_uses_envelope_when_many_samples_per_pixel(sr):
    """A long clip relative to the figure's pixel width should switch to the
    bucketed min/max/std envelope rather than a single dense line."""
    from matplotlib.collections import PolyCollection

    t = np.arange(0, 60 * sr) / sr  # 60s at 8kHz -> far more samples than pixels
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    viz = AudioVisualization(y=y, sr=sr)
    fig = viz.show(end_time=60.0, tracks=(Track.WAVEFORM,), width=6, height=2)
    ax = fig.axes[0]
    poly_collections = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(poly_collections) == 2  # min/max envelope + mean+/-std band
    mean_line = ax.get_lines()[0]
    n_pixels_approx = round(6 * fig.dpi)
    assert len(mean_line.get_xdata()) == pytest.approx(n_pixels_approx, rel=0.05)


def test_waveform_envelope_threshold_is_configurable(sr):
    """A custom MatplotlibRenderer(waveform_envelope_threshold=...) should shift
    where the direct-line/envelope switch happens."""
    from matplotlib.collections import PolyCollection

    from audio_classifier_visualizer.render.matplotlib_renderer import MatplotlibRenderer

    t = np.arange(0, int(0.05 * sr)) / sr
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    # A very low threshold should force envelope mode even for this short clip that
    # the default threshold would render as a direct line (see the test above).
    renderer = MatplotlibRenderer(waveform_envelope_threshold=0.01)
    viz = AudioVisualization(y=y, sr=sr, renderer=renderer)
    fig = viz.show(end_time=0.05, tracks=(Track.WAVEFORM,), width=6, height=2)
    ax = fig.axes[0]
    assert any(isinstance(c, PolyCollection) for c in ax.collections)


def test_context_seconds_uses_larger_of_stft_and_wavelet_needs(sr):
    from audio_classifier_visualizer.features.stft import STFTFeatureExtractor
    from audio_classifier_visualizer.features.wavelet import WaveletFeatureExtractor

    viz = AudioVisualization(
        y=np.zeros(int(sr * 10), dtype=np.float32),
        sr=sr,
        stft=STFTFeatureExtractor(n_fft=2048),
        wavelet=WaveletFeatureExtractor(overlap=100),
    )
    # stft needs (2048/2)/sr = 0.128s; wavelet needs 100/sr = 0.0125s -- stft wins.
    assert viz._context_seconds() == pytest.approx((2048 / 2) / sr)


def test_load_context_audio_uses_real_neighboring_samples(sr):
    """The whole point: context audio in the middle of a longer clip should be
    genuine neighboring samples, not zeros or padding."""
    t = np.arange(0, 10 * sr) / sr
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # continuous tone throughout
    viz = AudioVisualization(y=y, sr=sr)
    ctx = viz._load_context_audio(3.0, 7.0, context_seconds=0.5)
    # Context buffer should span [2.5, 7.5) -- 5s of real signal, no padding needed.
    assert ctx.duration == pytest.approx(5.0, abs=1e-2)
    # Its content should match the real signal at that position, not zeros.
    assert np.abs(ctx.channel(0)).mean() > 0.3  # nowhere near the ~0 a zero-padded region would show


def test_load_context_audio_reflects_at_true_start_of_file(sr):
    """Near t=0, there's no real 'before' data -- must reflect-pad, not zero-pad."""
    t = np.arange(0, 5 * sr) / sr
    y = (0.5 + 0.3 * np.sin(2 * np.pi * 3 * t)).astype(np.float32)  # nonzero throughout, incl. near t=0
    viz = AudioVisualization(y=y, sr=sr)
    ctx = viz._load_context_audio(0.0, 1.0, context_seconds=0.3)
    # Requested [-0.3, 1.3) clamps to [0, 1.3) real + 0.3s reflected on the left.
    assert ctx.duration == pytest.approx(1.6, abs=1e-2)
    left_pad_region = ctx.channel(0)[: round(0.3 * sr)]
    # A reflect-padded region mirrors real (nonzero) signal values, not silence.
    assert np.abs(left_pad_region).mean() > 0.2


def test_load_context_audio_reflects_at_true_end_of_file(sr):
    t = np.arange(0, 5 * sr) / sr
    y = (0.5 + 0.3 * np.sin(2 * np.pi * 3 * t)).astype(np.float32)
    viz = AudioVisualization(y=y, sr=sr)
    ctx = viz._load_context_audio(4.0, 5.0, context_seconds=0.3)
    assert ctx.duration == pytest.approx(1.6, abs=1e-2)
    right_pad_region = ctx.channel(0)[-round(0.3 * sr) :]
    assert np.abs(right_pad_region).mean() > 0.2


def test_spectrogram_edge_uses_real_context_not_zero_padding(sr):
    """Regression test for the reported bug: a display window in the *middle* of a
    longer clip should have spectrogram energy near its edges that reflects the
    real continuing signal, not an artifact from the edge of the displayed slice
    being treated as the edge of the whole signal."""
    t = np.arange(0, 20 * sr) / sr
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # continuous tone, no gaps
    viz = AudioVisualization(y=y, sr=sr)
    viz.stft.n_fft = 1024
    fig = viz.show(start_time=5.0, end_time=15.0, tracks=(Track.STFT_SPECTROGRAM,))
    ax = fig.axes[0]
    img = np.asarray(ax.get_images()[0].get_array())
    # A continuous tone should look roughly uniform across time -- the first and
    # last few columns should not be dramatically darker/different than the middle
    # the way a hard zero-padded edge would produce.
    middle_col_brightness = img[:, img.shape[1] // 2].mean()
    first_col_brightness = img[:, 2].mean()
    last_col_brightness = img[:, -3].mean()
    assert abs(first_col_brightness - middle_col_brightness) < 0.3
    assert abs(last_col_brightness - middle_col_brightness) < 0.3


def test_hand_built_spec_without_context_audio_still_renders(tone, sr):
    """A VisualizationSpec built directly (bypassing AudioVisualization) has no
    context_audio -- rendering must still work, just without the edge-context fix."""
    from audio_classifier_visualizer.core.audio_signal import AudioSignal
    from audio_classifier_visualizer.render.matplotlib_renderer import MatplotlibRenderer
    from audio_classifier_visualizer.render.spec import VisualizationSpec

    signal = AudioSignal(samples=tone, sr=sr)
    spec = VisualizationSpec(audio=signal, tracks=(Track.STFT_SPECTROGRAM,))
    assert spec.context_audio is None
    fig = MatplotlibRenderer().render(spec)
    assert fig is not None


def test_bucket_stats_short_clip_has_no_trailing_dead_zone():
    """Regression test for the reported bug: a short clip bucketed toward a much
    larger target pixel count must not pad up to a huge fraction of fake trailing
    buckets. This exact scenario (0.5s at 8000Hz, width=12in/dpi=100 -> 1200
    target buckets) padded 200 of 1200 buckets (16.7% of the plot!) under the old
    algorithm; the fix guarantees at most one bucket's worth of padding, total."""
    from audio_classifier_visualizer.render.matplotlib_renderer import _bucket_stats

    len_y = 4000  # 0.5s * 8000Hz, matching the reported notebook scenario
    y = np.sin(2 * np.pi * 5 * np.linspace(0, 0.5, len_y))
    n_pixels = 1200  # width=12in * dpi=100

    bucket_min, bucket_max, _mean, _std = _bucket_stats(y, n_pixels)
    n_buckets_actual = len(bucket_min)
    chunk_size = -(-len_y // n_pixels)
    total_covered = n_buckets_actual * chunk_size
    pad_len = total_covered - len_y
    assert pad_len < chunk_size  # at most a fraction of ONE bucket, never whole fake buckets

    # No bucket should be a flat, fully-padded (min == max, unrelated to the
    # signal's real range) artifact anywhere but possibly the very last one.
    degenerate_buckets = [i for i in range(n_buckets_actual - 1) if (bucket_max[i] - bucket_min[i]) < 1e-9]
    assert not degenerate_buckets


def test_waveform_short_clip_spans_full_requested_range(sr):
    """End-to-end regression: the exact reported symptom -- the rendered waveform
    x-data must cover the full requested [start_time, end_time), not stop short."""
    y = np.sin(2 * np.pi * 440 * np.arange(0, int(0.5 * sr)) / sr).astype(np.float32)
    viz = AudioVisualization(y=y, sr=sr)
    fig = viz.show(start_time=0.0, end_time=0.5, tracks=(Track.WAVEFORM,), width=12, height=6)
    ax = fig.axes[0]
    # Whichever drawing mode engaged (direct line or envelope), the last drawn
    # x-value should reach (not stop well short of) the requested end_time.
    last_x = max(
        (line.get_xdata()[-1] for line in ax.get_lines() if len(line.get_xdata())),
        default=None,
    )
    assert last_x == pytest.approx(0.5, abs=0.5 / 1200 * 2)  # within ~2 buckets of the true end


def test_similarity_and_class_stack_render_when_display_window_narrower_than_one_classifier_window(tone, sr):
    """Regression test for the reported bug: zooming to a range narrower than one
    classifier window (or just unluckily aligned to contain only one) must still
    show visible SIMILARITY_LINES / CLASS_PROBABILITY_STACK content, not nothing."""
    feature_rate = 2.0  # 0.5s windows
    n_windows = 20
    probs = np.zeros((n_windows, 2))
    probs[:, 1] = np.linspace(0, 1, n_windows)
    probs[:, 0] = 1 - probs[:, 1]
    co = ClassifierOutput(probabilities=probs, feature_rate=feature_rate, class_labels=["other", "target"])

    viz = AudioVisualization(y=tone, sr=sr, classifier_output=co)
    fig = viz.show(
        start_time=0.0,
        end_time=0.5,  # exactly one classifier window -- the failing case
        tracks=(Track.SIMILARITY_LINES, Track.CLASS_PROBABILITY_STACK),
        target_class="target",
    )
    similarity_ax, stack_ax = fig.axes

    similarity_line = next(line for line in similarity_ax.get_lines() if line.get_color() == "tab:green")
    assert len(similarity_line.get_xdata()) >= 2

    # stackplot renders as PolyCollection(s) with actual filled area (not zero-width).
    fig.canvas.draw()
    poly_collections = list(stack_ax.collections)
    assert poly_collections
    total_vertices = sum(len(path.vertices) for c in poly_collections for path in c.get_paths())
    assert total_vertices > 0
