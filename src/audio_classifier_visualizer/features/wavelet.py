"""Chunked, decimated CWT / synchrosqueezed-CWT computation for very long audio.

This is a direct, deliberately-conservative port of the chunking/decimation
approach from the original ``_WaveletComponent`` -- that part of the old code
was good: it makes wavelet transforms of 24-hour recordings roughly as
memory-efficient as an STFT with a large hop length, by processing in
overlapping chunks and decimating (mean-pooling power, subsampling complex
amplitude) as it goes.

What changed vs. the original:
  * GPU use is an explicit constructor arg (``use_gpu=True``) instead of
    reading the ``SSQ_GPU`` environment variable as a side effect.
  * No dependency on ``torch`` for anything -- ssqueezepy may internally use
    torch/cupy tensors, but we convert to numpy at the boundary, same as before.
  * Returns a small ``WaveletResult`` dataclass instead of a bare tuple.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import einx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WaveletResult:
    amplitude: np.ndarray  # complex, shape (n_freqs, n_decimated_samples)
    power: np.ndarray  # real, shape (n_freqs, n_decimated_samples), decimated by mean-pooling
    freqs: np.ndarray  # shape (n_freqs,)


class WaveletFeatureExtractor:
    def __init__(
        self,
        chunk_size: int = 65536,
        decimation_stride: int = 1024,
        freq_range_of_interest: tuple[float, float] | None = (100, 1200),
        *,
        use_gpu: bool = False,
        synchrosqueeze: bool = False,
    ) -> None:
        self.decimation_stride = decimation_stride
        self.chunk_size = ((chunk_size + decimation_stride) // decimation_stride) * decimation_stride
        self.freq_range_of_interest = freq_range_of_interest
        self.use_gpu = use_gpu
        self.synchrosqueeze = synchrosqueeze
        self._wavelet = None  # lazily constructed; ssqueezepy import deferred to first use

    def _ensure_ssqueezepy(self):
        try:
            import ssqueezepy as sqz
        except ImportError as e:
            msg = (
                "Wavelet/CWT features require the optional 'ssqueezepy' dependency. "
                "Install with: pip install 'audio-classifier-visualizer[wavelet]'"
            )
            raise ImportError(msg) from e
        if self._wavelet is None:
            self._wavelet = sqz.Wavelet()
        return sqz

    def _get_scales_and_freqs(self, sqz, n_samples: int, sr: float):
        from ssqueezepy import utils as ssq_utils
        from ssqueezepy.experimental import scale_to_freq

        scale_size = min(self.chunk_size, n_samples, round(sr * 10))
        bounds = ssq_utils.cwt_scalebounds(self._wavelet, scale_size)
        scales = ssq_utils.make_scales(scale_size, bounds[0], bounds[1], scaletype="log-piecewise", wavelet=self._wavelet)
        freqs = scale_to_freq(scales, self._wavelet, scale_size, fs=sr)
        if self.freq_range_of_interest:
            lo, hi = self.freq_range_of_interest
            keep = (freqs >= lo) & (freqs <= hi)
            freqs, scales = freqs[keep], scales[keep]
        return scales, freqs

    def decimate(self, spec_amp: np.ndarray, spec_pwr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean-pool power (meaningful average), subsample amplitude (can't mean-pool complex phase).

        See module docstring -- this is the memory-efficiency trick that makes
        24-hour-scale wavelet spectrograms tractable.
        """
        stride = self.decimation_stride
        if spec_pwr.shape[1] % stride != 0:
            pad_len = (stride - spec_pwr.shape[1] % stride) % stride
            mean_value = np.mean(spec_pwr)
            spec_pwr = np.pad(spec_pwr, ((0, 0), (0, pad_len)), mode="constant", constant_values=mean_value)
        decimated_amp = spec_amp[:, ::stride].copy()
        decimated_pwr = einx.mean("a (b c) -> a b", spec_pwr, c=stride)
        return decimated_amp, decimated_pwr

    def compute(self, y: np.ndarray, sr: float, overlap: int = 512) -> WaveletResult:
        """Run the (possibly synchrosqueezed) CWT over ``y`` in overlapping chunks."""
        sqz = self._ensure_ssqueezepy()
        prev_gpu_env = os.environ.get("SSQ_GPU")
        if self.use_gpu:
            os.environ["SSQ_GPU"] = "1"
        try:
            scales, freqs = self._get_scales_and_freqs(sqz, len(y), sr)
            padded = np.pad(y, (overlap, overlap), mode="constant")
            amp_chunks: list[np.ndarray] = []
            pwr_chunks: list[np.ndarray] = []
            for start in range(0, len(padded) - 2 * overlap, self.chunk_size):
                chunk = padded[start : start + self.chunk_size + 2 * overlap]
                if self.synchrosqueeze:
                    tx, _wx, _ssq_freqs, _scales = sqz.ssq_cwt(chunk, self._wavelet, scales=scales)
                    result = tx
                else:
                    result, _scales = sqz.cwt(chunk, self._wavelet, scales=scales)
                if hasattr(result, "detach"):  # torch/cupy tensor -> numpy, at the boundary only
                    result = result.detach().cpu().numpy() if hasattr(result, "cpu") else result.detach().numpy()
                trimmed = result[:, overlap:-overlap]
                power = np.abs(trimmed) ** 2
                dec_amp, dec_pwr = self.decimate(trimmed, power)
                amp_chunks.append(dec_amp)
                pwr_chunks.append(dec_pwr)
            amplitude = np.concatenate(amp_chunks, axis=1)
            power = np.concatenate(pwr_chunks, axis=1)
            return WaveletResult(amplitude=amplitude, power=power, freqs=freqs)
        finally:
            if self.use_gpu:
                if prev_gpu_env is None:
                    os.environ.pop("SSQ_GPU", None)
                else:
                    os.environ["SSQ_GPU"] = prev_gpu_env
