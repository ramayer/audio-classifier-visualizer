from __future__ import annotations

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

from audio_classifier_visualizer.io.audio_loader import load_audio
from audio_classifier_visualizer.io.labels_io import load_raven_selection_table


@pytest.fixture
def wav_path(tmp_path, tone, sr):
    path = tmp_path / "mono.wav"
    soundfile.write(path, tone, int(sr))
    return str(path)


@pytest.fixture
def surround_wav_path(tmp_path, sr):
    path = tmp_path / "surround.wav"
    data = np.random.default_rng(0).uniform(-0.1, 0.1, size=(int(sr), 7)).astype("float32")  # 6+1 channels
    soundfile.write(path, data, int(sr))
    return str(path), data


def test_load_audio_round_trips_mono(wav_path, tone, sr):
    sig = load_audio(wav_path)
    assert sig.n_channels == 1
    assert sig.sr == sr
    np.testing.assert_allclose(sig.channel(0), tone, atol=1e-4)


def test_load_audio_time_range(wav_path, sr):
    sig = load_audio(wav_path, start_time=1.0, end_time=2.0)
    assert sig.duration == pytest.approx(1.0, abs=1e-3)


def test_load_audio_preserves_channel_count_for_surround(surround_wav_path):
    path, data = surround_wav_path
    sig = load_audio(path)
    assert sig.n_channels == 7
    np.testing.assert_allclose(sig.samples, data.T, atol=1e-4)


def test_load_raven_selection_table(tmp_path):
    table = tmp_path / "selections.txt"
    table.write_text(
        "Selection\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tAnnotation\n"
        "1\t10.0\t12.5\t100\t400\trumble\n"
        "2\t50.0\t51.0\t50\t200\ttrumpet\n"
    )
    boxes = load_raven_selection_table(str(table))
    assert len(boxes) == 2
    assert boxes[0].text == "rumble"
    assert boxes[0].start_time == pytest.approx(10.0)
    assert boxes[1].high_freq == pytest.approx(200)


def test_load_audio_shifts_absolute_start_by_start_time(tmp_path, tone, sr):
    """Regression test: absolute_start is the real time of the *whole file's*
    sample 0, not of whatever start_time-offset slice is being loaded -- loading
    a 2-second-in slice must anchor its own relative time 0 at absolute_start+2s,
    not at absolute_start unshifted (which was silently correct only at
    start_time=0, subtly wrong everywhere else -- the same shape of edge-alignment
    bug this library has hit several times before)."""
    from datetime import datetime, timedelta, timezone

    from audio_classifier_visualizer.io.audio_loader import load_audio

    path = tmp_path / "shift_test.wav"
    soundfile.write(path, tone, int(sr))

    file_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    signal = load_audio(str(path), start_time=2.0, end_time=3.0, absolute_start=file_start)
    assert signal.time_axis.to_absolute(0.0) == file_start + timedelta(seconds=2.0)
