from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sr() -> float:
    return 8000.0


@pytest.fixture
def tone(sr) -> np.ndarray:
    """5 seconds of a 440 Hz tone -- enough for STFT/CWT sanity checks without being slow."""
    t = np.arange(0, 5 * sr) / sr
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def stereo_tone(tone) -> np.ndarray:
    return np.stack([tone, tone * 0.5])
