from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from audio_classifier_visualizer.core.audio_signal import AudioSignal
from audio_classifier_visualizer.core.classifier_output import ClassifierOutput
from audio_classifier_visualizer.core.labels import LabelBox
from audio_classifier_visualizer.core.time_axis import TimeAxis


def test_audio_signal_promotes_mono_to_2d(tone, sr):
    sig = AudioSignal(samples=tone, sr=sr)
    assert sig.samples.ndim == 2
    assert sig.n_channels == 1
    assert sig.n_samples == len(tone)


def test_audio_signal_rejects_bad_ndim(sr):
    with pytest.raises(ValueError, match="1-D or 2-D"):
        AudioSignal(samples=np.zeros((2, 3, 4)), sr=sr)


def test_audio_signal_duration(tone, sr):
    sig = AudioSignal(samples=tone, sr=sr)
    assert sig.duration == pytest.approx(5.0)


def test_slice_time_keeps_relative_zero(tone, sr):
    sig = AudioSignal(samples=tone, sr=sr)
    sliced = sig.slice_time(1.0, 2.0)
    assert sliced.duration == pytest.approx(1.0, abs=1e-3)


def test_slice_time_shifts_absolute_anchor(tone, sr):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    sig = AudioSignal(samples=tone, sr=sr, time_axis=TimeAxis(absolute_start=start))
    sliced = sig.slice_time(2.0, 3.0)
    assert sliced.time_axis.to_absolute(0.0) == start + timedelta(seconds=2.0)


def test_as_mono_averages_channels(stereo_tone, sr):
    sig = AudioSignal(samples=stereo_tone, sr=sr)
    mono = sig.as_mono()
    assert mono.shape == (stereo_tone.shape[1],)
    np.testing.assert_allclose(mono, stereo_tone.mean(axis=0))


def test_time_axis_round_trips(sr):
    start = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    axis = TimeAxis(absolute_start=start)
    when = axis.to_absolute(90.0)
    assert axis.from_absolute(when) == pytest.approx(90.0)


def test_time_axis_without_anchor_raises():
    axis = TimeAxis()
    with pytest.raises(ValueError, match="no absolute_start"):
        axis.to_absolute(10.0)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(5, "00:05"), (65, "01:05"), (3661, "1:01:01"), (1.5, "00:01.500")],
)
def test_time_axis_format_relative(seconds, expected):
    axis = TimeAxis()
    assert axis.format_relative(seconds) == expected


def test_label_box_overlaps():
    box = LabelBox(start_time=10, end_time=20, low_freq=100, high_freq=200)
    assert box.overlaps(15, 25)
    assert box.overlaps(0, 10.5)
    assert not box.overlaps(21, 30)


def test_classifier_output_shape_validation():
    with pytest.raises(ValueError, match="classes"):
        ClassifierOutput(probabilities=np.zeros((10, 3)), feature_rate=25.0, class_labels=["a", "b"])


def test_classifier_output_time_index_round_trip():
    co = ClassifierOutput(probabilities=np.zeros((100, 2)), feature_rate=25.0, class_labels=["neg", "pos"])
    idx = co.time_to_index(4.0)
    assert idx == 100
    assert co.index_to_time(idx) == pytest.approx(4.0)


def test_classifier_output_resample_matches_endpoints():
    probs = np.zeros((10, 2))
    probs[:, 1] = np.linspace(0, 1, 10)
    co = ClassifierOutput(probabilities=probs, feature_rate=1.0, class_labels=["neg", "pos"])
    stretched = co.resample_class_to(1, target_length=100)
    assert stretched.shape == (100,)
    assert stretched[0] == pytest.approx(0.0, abs=1e-6)
    assert stretched[-1] == pytest.approx(1.0, abs=1e-6)


def test_classifier_output_resample_aligns_with_window_centers():
    """Regression test: a step at the boundary between window i and i+1 should cross
    0.5 at the *midpoint between their centers*, matching window_centers()/the
    similarity-line track -- not at the boundary's own raw time (the old, buggy
    behavior, which put the 0.5 crossing half a window too early)."""
    feature_rate = 2.0  # 0.5s windows, matching the notebook scenario that surfaced this
    probs = np.zeros((4, 2))
    probs[2:, 1] = 1.0  # step up at window index 2 (raw window-start time == 1.0s)
    co = ClassifierOutput(probabilities=probs, feature_rate=feature_rate, class_labels=["other", "target"])

    duration = 2.0
    target_length = 2000  # fine enough to localize the crossing precisely
    stretched = co.resample_class_to(1, target_length=target_length, duration=duration)
    dst_t = (np.arange(target_length) + 0.5) / target_length * duration

    # window 1 center = 0.75s (value 0), window 2 center = 1.25s (value 1) ->
    # linear crossing of 0.5 lands exactly at t=1.0s, not at t=0.75s (a naive
    # "index/(n-1)-of-span" mapping) and not smeared to some other point.
    crossing_idx = np.argmin(np.abs(stretched - 0.5))
    assert dst_t[crossing_idx] == pytest.approx(1.0, abs=0.01)


def test_classifier_output_slice_time():
    co = ClassifierOutput(probabilities=np.arange(20).reshape(10, 2), feature_rate=2.0, class_labels=["a", "b"])
    sliced = co.slice_time(1.0, 3.0)  # indices 2..6
    assert sliced.n_windows == 4


def test_classifier_output_slice_time_tracks_actual_rounded_start():
    """slice_time can only cut at whole-window boundaries. Regression test for a real
    bug: a zoom start_time that doesn't land on one (e.g. 0.7s at feature_rate=2, which
    rounds to window index 1 -> actual start 0.5s) must have its slice's window_centers()
    reflect that *actual* rounded start, not the originally-requested 0.7s."""
    co = ClassifierOutput(probabilities=np.zeros((10, 1)), feature_rate=2.0, class_labels=["x"])
    sliced = co.slice_time(0.7, 1.3)
    assert sliced.time_offset == pytest.approx(0.5)  # round(0.7*2)/2, not 0.7
    assert sliced.window_centers()[0] == pytest.approx(0.75)  # 0.5 + 0.5*(1/2), absolute


def test_window_centers_after_slice_matches_true_original_window_centers():
    """The window that becomes local index 0 after slicing must report the same
    center time it had *before* slicing -- i.e. slicing must not shift any window's
    reported time at all, only select a subrange of them."""
    feature_rate = 2.0
    co = ClassifierOutput(probabilities=np.zeros((10, 1)), feature_rate=feature_rate, class_labels=["x"])
    original_centers = co.window_centers()
    sliced = co.slice_time(0.7, 10.0)  # rounds to starting at window index 1
    np.testing.assert_allclose(sliced.window_centers(), original_centers[1:])


def test_resample_class_to_with_start_time_matches_narrow_zoom():
    """End-to-end regression for the reported bug: at a narrow, non-window-aligned
    zoom range, the resampled color curve's 0.5 crossing must land at the same
    absolute time as window_centers()' own crossing -- not shifted by the zoom's
    rounding error."""
    feature_rate = 2.0
    probs = np.zeros((4, 2))
    probs[2:, 1] = 1.0  # true transition at t=1.0s (between window 1 center=.75 and window 2 center=1.25)
    co = ClassifierOutput(probabilities=probs, feature_rate=feature_rate, class_labels=["other", "target"])

    start_time, end_time = 0.7, 1.3
    sliced = co.slice_time(start_time, end_time)
    target_length = 6000
    duration = end_time - start_time
    stretched = sliced.resample_class_to(1, target_length=target_length, duration=duration, start_time=start_time)
    dst_t = start_time + (np.arange(target_length) + 0.5) / target_length * duration

    crossing_idx = np.argmin(np.abs(stretched - 0.5))
    assert dst_t[crossing_idx] == pytest.approx(1.0, abs=0.01)
