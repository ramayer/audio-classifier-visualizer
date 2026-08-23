"""Load audio files into AudioSignal objects.

Deliberately not entangled with visualization (the old code loaded audio
directly inside the plotting class's __init__). Uses ``soundfile`` as the
primary backend: it handles arbitrary channel counts (including 6+1
surround) natively via libsndfile, and isn't caught in the
torchaudio -> torchcodec migration churn. ``librosa.load`` remains available
as a fallback for formats libsndfile doesn't handle (e.g. some compressed
formats), but is not required for the common WAV/FLAC case.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from audio_classifier_visualizer.core.audio_signal import AudioSignal
from audio_classifier_visualizer.core.time_axis import TimeAxis


def load_audio(
    path: str,
    *,
    start_time: float = 0.0,
    end_time: float | None = None,
    target_sr: float | None = None,
    absolute_start: datetime | None = None,
) -> AudioSignal:
    """Load a (possibly multichannel) audio file, optionally just a time range of it.

    Raises the underlying soundfile error if the file can't be opened;
    callers wanting the librosa fallback path can catch that and call
    ``load_audio_via_librosa`` explicitly (kept separate, not a silent
    try/except cascade, so failures are visible).
    """
    import soundfile as sf

    with sf.SoundFile(path) as f:
        sr = float(f.samplerate)
        start_frame = max(0, round(start_time * sr))
        n_frames = f.frames - start_frame if end_time is None else round((end_time - start_time) * sr)
        f.seek(start_frame)
        data = f.read(frames=n_frames, dtype="float32", always_2d=True)  # (n_frames, n_channels)

    samples = data.T  # -> (n_channels, n_frames)

    if target_sr is not None and target_sr != sr:
        samples, sr = _resample(samples, sr, target_sr)

    axis = TimeAxis(absolute_start=absolute_start)
    return AudioSignal(samples=samples, sr=sr, time_axis=axis, source_path=path)


def load_audio_via_librosa(
    path: str,
    *,
    start_time: float = 0.0,
    end_time: float | None = None,
    target_sr: float | None = None,
    absolute_start: datetime | None = None,
) -> AudioSignal:
    """Fallback loader for formats libsndfile can't read. Mixes down to mono like the old default."""
    import librosa

    y, sr = librosa.load(
        path,
        sr=target_sr,
        offset=start_time,
        duration=None if end_time is None else end_time - start_time,
        mono=True,
    )
    axis = TimeAxis(absolute_start=absolute_start)
    return AudioSignal(samples=y[np.newaxis, :], sr=sr, time_axis=axis, source_path=path)


def _resample(samples: np.ndarray, orig_sr: float, target_sr: float) -> tuple[np.ndarray, float]:
    import librosa

    resampled = np.stack([librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr) for ch in samples])
    return resampled, target_sr
