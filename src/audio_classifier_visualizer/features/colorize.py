"""Turn (spectral power, classifier confidence) into an RGB image.

Pulled out of the old ``_SpectrogramComponent`` god-class so it's testable in
isolation and reusable regardless of which renderer eventually draws it.
"""

from __future__ import annotations

import numpy as np


def power_to_db(power: np.ndarray, ref: float | None = None) -> np.ndarray:
    ref = np.max(power) if ref is None else ref
    ref = max(ref, np.finfo(power.dtype if power.dtype.kind == "f" else float).eps)
    return 10.0 * np.log10(np.maximum(power, 1e-12) / ref)


def normalize_power(
    power: np.ndarray,
    *,
    per_channel_normalize: bool = False,
    clip_outliers: bool = True,
) -> np.ndarray:
    power = power.copy()
    if per_channel_normalize:
        median_per_band = np.median(power, axis=1)
        median_per_band[median_per_band == 0] = 1
        power = power / median_per_band[:, None]
    if clip_outliers:
        lo = np.percentile(power, 0.1)
        hi = np.percentile(power, 99.9)
        power = np.clip(power, lo, hi)
    return power


def confidence_to_rgb(
    grayscale: np.ndarray,
    similarity: np.ndarray,
    dissimilarity: np.ndarray,
    *,
    style: str = "bright",
) -> np.ndarray:
    """Tint a grayscale (0..1 normalized) spectrogram by classifier confidence.

    ``similarity``/``dissimilarity`` must already be resampled to
    ``grayscale.shape[1]`` (use ``ClassifierOutput.resample_class_to``).

    style:
      "bright" -- red (dissimilar) <-> yellow (uncertain) <-> green (similar), smooth.
      "clean"  -- similar transition, normalized by max absolute difference.
      "raw"    -- similarity -> green channel, dissimilarity -> red channel, directly.
    """
    rgb = np.stack([grayscale, grayscale, grayscale], axis=-1)

    if style in ("bright", "clean"):
        diff = similarity - dissimilarity
        denom = np.max(np.abs(diff)) or 1.0
        diff = diff / denom
        if style == "bright":
            redness = np.clip(-diff + 1, 0, 1)
            greenness = np.clip(diff + 1, 0, 1)
        else:  # clean
            redness = np.clip(-diff * 8 + 1, 0, 1)
            greenness = np.clip(diff * 8 + 1, 0, 1)
    elif style == "raw":
        redness = dissimilarity.copy()
        greenness = similarity.copy()
        for arr in (redness, greenness):
            span = arr.max() - arr.min()
            if span > 0:
                arr -= arr.min()
                arr /= span
    else:
        msg = f"Unknown colorize style: {style!r}"
        raise ValueError(msg)

    blueness = np.clip(1 - (redness + greenness), 0, None)
    rgb[:, :, 0] *= redness
    rgb[:, :, 1] *= greenness
    rgb[:, :, 2] *= blueness
    return rgb
