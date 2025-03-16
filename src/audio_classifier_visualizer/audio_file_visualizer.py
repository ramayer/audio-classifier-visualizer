from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import einx
import librosa
import matplotlib.pyplot as plt
import numpy as np
import ssqueezepy as sqz
import torch
from matplotlib import patches
from ssqueezepy import utils as ssq_utils

# from ssqueezepy import ssq_cwt, ssq_stft
from ssqueezepy.experimental import scale_to_freq

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from enum import Enum


class Subplot(Enum):
    WAVEFORM = "waveform"
    STFT_SPECTROGRAM = "stft_spectrogram"
    WAVELET_SPECTROGRAM = "wavelet_spectrogram"
    SIMILARITIES = "similarities"
    CLASS_PROBABILITIES = "class_probabilities"
    CLASS_PROBABILITY_LINES = "class_probability_lines"


class _STFTComponent:
    def __init__(self, n_fft: int = 2048, hop_length: int | None = None, freq_range_of_interest=(100, 1200)):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_range_of_interest = freq_range_of_interest

    def get_spectrogram(self, y: np.ndarray, sr: int):
        spec = librosa.stft(y, n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        spectral_power = np.abs(spec) ** 2  # type: ignore

        if self.freq_range_of_interest:
            lo_freq = self.freq_range_of_interest[0]
            hi_freq = self.freq_range_of_interest[1]
            freq_filter = (freqs >= lo_freq) & (freqs <= hi_freq)
            freqs = freqs[freq_filter]
            spec = spec[freq_filter]
            spectral_power = spectral_power[freq_filter]

        return spec, freqs, spectral_power


class _WaveletComponent:
    def __init__(
        self,
        chunk_size: int = 65536,
        decimation_stride: int = 1024,
        freq_range_of_interest: tuple[float, float] = (100, 1200),
    ):
        self.wavelet = sqz.Wavelet()
        self.logger = logging.getLogger(__name__)
        self.decimation_stride = decimation_stride
        self.chunk_size = ((chunk_size + decimation_stride) // decimation_stride) * decimation_stride
        self.freq_range_of_interest = freq_range_of_interest

    def get_bounds_scales_and_freqs(self, y, sr, wavelet=None):
        # scale_size = min([self.chunk_size, len(y), sr * 4])
        scale_size = min([self.chunk_size, len(y), sr * 10])
        wavelet = wavelet or self.wavelet
        bounds = ssq_utils.cwt_scalebounds(wavelet, scale_size)
        scales = ssq_utils.make_scales(
            scale_size, bounds[0], bounds[1], scaletype="log-piecewise", wavelet=wavelet
        )  # ,nv=32)
        freqs = scale_to_freq(scales, self.wavelet, scale_size, fs=sr)

        if self.freq_range_of_interest:
            lo_freq = self.freq_range_of_interest[0]
            hi_freq = self.freq_range_of_interest[1]
            freq_filter = (freqs >= lo_freq) & (freqs <= hi_freq)
            freqs = freqs[freq_filter]
            scales = scales[freq_filter]

        return bounds, scales, freqs

    def cwt_in_chunks(self, y: np.ndarray, sr: int, wavelet=None, overlap=512):
        wavelet = wavelet or self.wavelet
        bounds, scales, freqs = self.get_bounds_scales_and_freqs(y, sr, wavelet)

        padded_audio = np.pad(y, (overlap, overlap), mode="constant")
        results = []
        for start in range(0, len(padded_audio) - 2 * overlap, self.chunk_size):
            end = start + self.chunk_size + 2 * overlap
            chunk = padded_audio[start:end]
            wx, scales = sqz.cwt(chunk, wavelet, scales=scales)
            if isinstance(wx, torch.Tensor):
                wx = wx.detach().numpy(force=True)
            spec_for_this_chunk = wx[:, overlap:-overlap]

            results.append(spec_for_this_chunk)
        spectral_amplitude = np.concatenate(results, axis=1)
        spectral_power = np.abs(spectral_amplitude) ** 2

        return spectral_amplitude, freqs, spectral_power

    def ssq_cwt_in_chunks(self, y: np.ndarray, sr: int, wavelet=None, overlap=4096):
        wavelet = wavelet or self.wavelet
        bounds, scales, freqs = self.get_bounds_scales_and_freqs(y, sr, wavelet)

        padded_audio = np.pad(y, (overlap, overlap), mode="constant")
        results_amp = []
        results_pwr = []
        self.logger.debug("... processing chunks")
        for start in range(0, len(padded_audio) - 2 * overlap, self.chunk_size):
            end = start + self.chunk_size + 2 * overlap
            chunk = padded_audio[start:end]
            tx, _wx, _ssq_freqs, _scales = sqz.ssq_cwt(
                chunk,
                self.wavelet,
                scales=scales,  # 'log-piecewise',
                # cache_wavelet=True,
                # fs=44100,
            )
            # self.logger.debug(f"chunk shape {chunk.shape}, tx shape {tx.shape}")
            if isinstance(tx, torch.Tensor):
                tx = tx.detach().numpy(force=True)

            spec_for_this_chunk = tx[:, overlap:-overlap]
            spectral_power = np.abs(spec_for_this_chunk) ** 2
            # decimate results similar to stft hop_length to avoid OOM on 24-hour audio recordings
            decimation_stride = self.decimation_stride
            if spectral_power.shape[1] % decimation_stride != 0:
                mean_value = np.mean(spectral_power)  # TODO: consider row means
                padding_length = (decimation_stride - spectral_power.shape[1] % decimation_stride) % decimation_stride
                padded_spectral_power = np.pad(
                    spectral_power, ((0, 0), (0, padding_length)), mode="constant", constant_values=mean_value
                )
                spectral_power = padded_spectral_power
            # Decimating spectral power using mean() makes better visualizations than
            # just subsampling like stft hop lengths.  Doesn't work on amplitudes, though,
            # because they're complex values; so we just subsample those.
            # spectral_power = spectral_power[:, ::decimation_stride]
            decimated_tx = spec_for_this_chunk[:, ::decimation_stride].copy()  # helps garbage collection?
            decimated_spectral_power = einx.mean("a (b c) -> a b", spectral_power, c=decimation_stride)
            results_amp.append(decimated_tx)
            results_pwr.append(decimated_spectral_power)
            # logging.getLogger(__name__).debug("... Done decimating cwt")

        # spectral_amplitude = np.concatenate(results, axis=1)
        # spectral_power = np.abs(spectral_amplitude) ** 2
        spectral_amplitude = np.concatenate(results_amp, axis=1)
        spectral_power = np.concatenate(results_pwr, axis=1)
        self.logger.debug(f"... concatenated chunks {spectral_power.shape}, {spectral_amplitude.shape}")

        return spectral_amplitude, freqs, spectral_power


class _SpectrogramComponent:
    def __init__(
        self,
        y: np.ndarray,
        sr: int,
        class_probabilities: torch.Tensor | None = None,
        feature_rate: float | None = None,
        target_class: int | None = None,
        n_fft: int = 2048,
        hop_length: int | None = None,
        colormap: str = "bright",
        try_per_channel_normalization: bool = False,  # noqa: FBT001 FBT002
        clip_outliers: bool = True,  # noqa: FBT001 FBT002
        label_boxes: list[Any] | None = None,
        freq_range_of_interest=None,
    ):
        self.audio = y
        self.sr = sr

        self.stft_component = _STFTComponent(n_fft, hop_length, freq_range_of_interest=freq_range_of_interest)
        self.wavelet_component = _WaveletComponent(
            freq_range_of_interest=freq_range_of_interest, decimation_stride=hop_length
        )

        self.class_probabilities = class_probabilities
        self.feature_rate = feature_rate if feature_rate is not None else sr // 320
        self.target_class = target_class
        self.colormap = colormap
        self.try_per_channel_normalization = try_per_channel_normalization
        self.clip_outliers = clip_outliers
        self.label_boxes = label_boxes

        if class_probabilities is not None:
            self.similarity_scores = class_probabilities.cpu()[:, target_class or 1]
            self.dissimilarity_scores = 1 - self.similarity_scores  # class_probabilities.cpu()[:, 0]
        else:
            self.similarity_scores = None
            self.dissimilarity_scores = None

        self.duration = self.audio.shape[0] / self.sr

        # This component shouldn't care much about absolute times, except for adding labels
        self.start_time = 0
        self.end_time = self.audio.shape[0] / self.sr

    def time_to_score_index(self, t: float) -> int:
        return t * self.feature_rate

    def score_index_to_time(self, s: int) -> float:
        return s / self.feature_rate

    def interpolate_1d_tensor(self, input_tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        z = input_tensor[None, None, :]
        return torch.nn.functional.interpolate(z, target_length, mode="linear")[0][0]

    def add_annotation_boxes(
        self,
        labels: list[Any],
        patch_start: float,
        patch_end: float,
        axarr: Axes,
        offset: float = 0.2,
        color: tuple[float, float, float] = (0.0, 1.0, 1.0),
        vert_offset: int = 0,
    ) -> None:
        """Add annotation boxes to the visualization.
            Cool
        Args:
            labels: List of annotation labels
            patch_start: Start time of the patch
            patch_end: End time of the patch
            axarr: Matplotlib axes to draw on
            offset: Offset for box drawing (default: 0.2)
            color: RGB color tuple for the boxes
        """
        for row in labels:
            # bt, et, lf, hf, dur, fn, tags, notes, tag1, tag2, score, raven_file = dataclasses.astuple(row)
            # logging.getLogger(__name__).debug(f"{row}, {patch_start} {patch_end}") # labels can be drawn outside of display
            if row.et < patch_start or row.bt > patch_end:
                continue
            xy = (row.bt - patch_start - offset, row.lf - 5)
            width = row.et - row.bt + offset * 2
            height = row.hf - row.lf + 10

            for linewidth, edgecolor in [(3, (0, 0, 0)), (1, color)]:
                rect = patches.Rectangle(
                    xy,
                    width,
                    height,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    facecolor="none",
                )
                axarr.add_patch(rect)

                # Add label below the box with a dark background
                # TODO: the offset of the label should depend on the timescale being shown
                label_x, label_y = (xy[0] + width, xy[1] + height + vert_offset)  # Adjust the y position as needed
                bbox_props = {"boxstyle": "round,pad=0.3", "edgecolor": edgecolor, "facecolor": "black", "alpha": 0.7}
                axarr.text(
                    label_x, label_y, row.notes, ha="right", va="bottom", fontsize=12, color="white", bbox=bbox_props
                )

    def _normalize_spectral_power(
        self,
        np_spectral_power: np.ndarray,
        try_per_channel_normalization: bool,  # noqa: FBT001
        clip_outliers: bool,  # noqa: FBT001
    ) -> np.ndarray:  # FBT001
        if try_per_channel_normalization:
            median_pwr_per_spectral_band = np.median(np_spectral_power, axis=1)
            median_pwr_per_spectral_band[median_pwr_per_spectral_band == 0] = 1
            np_spectral_power = np_spectral_power / median_pwr_per_spectral_band[:, None]

        if clip_outliers:
            noise_floor = np.percentile(np_spectral_power, 0.1)
            clip_level = np.percentile(np_spectral_power, 99.9)
            db_normalized = np_spectral_power
            db_normalized[np_spectral_power < noise_floor] = noise_floor
            db_normalized[np_spectral_power > clip_level] = clip_level
            np_spectral_power = db_normalized

        return np_spectral_power

    def _compute_color_channels(
        self, stretched_similarity: torch.Tensor, stretched_dissimilarity: torch.Tensor, colormap: str = "bright"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if colormap == "clean":
            nearness = stretched_similarity
            farness = stretched_dissimilarity

            sim = nearness - farness
            sim /= sim.abs().max()
            sim = sim.numpy()

            redness = -sim * 8 + 1
            redness[redness > 1] = 1
            redness[redness < 0] = 0

            greenness = sim * 8 + 1
            greenness[greenness > 1] = 1
            greenness[greenness < 0] = 0

        elif colormap == "bright":
            nearness = stretched_similarity
            farness = stretched_dissimilarity

            sim = nearness - farness
            sim /= sim.abs().max()
            sim = sim.numpy()

            # Create a smooth transition from red to yellow to green
            # When sim is -1, we want red (1,0,0)
            # When sim is 0, we want yellow (1,1,0)
            # When sim is 1, we want green (0,1,0)

            redness = np.clip(-sim + 1, 0, 1)
            greenness = np.clip(sim + 1, 0, 1)

        else:
            redness = stretched_dissimilarity.numpy()
            greenness = stretched_similarity.numpy()
            if redness.max() > 1 or redness.min() < 0 or greenness.max() > 1 or greenness.min() < 0:
                redness -= redness.min()
                redness /= redness.max()
                greenness -= greenness.min()
                greenness /= greenness.max()

        blueness = 1 - (redness + greenness)
        blueness[blueness < 0] = 0
        return redness, greenness, blueness

    def _ticks(self, xticks, yticks, ax):
        # from ssqueezepy
        def fmt(ticks):
            if all(isinstance(h, str) for h in ticks):
                return "%s"
            return "%.d" if all(float(h).is_integer() for h in ticks) else "%.4g"

        if yticks is not None:
            if not hasattr(yticks, "__len__") and not yticks:
                ax.set_yticks([])
            else:
                idxs = np.linspace(0, len(yticks) - 1, 8).astype("int32")
                yt = [fmt(yticks) % h for h in np.asarray(yticks)[idxs]]
                ax.set_yticks(idxs)
                ax.set_yticklabels(yt)
        if xticks is not None:
            if not hasattr(xticks, "__len__") and not xticks:
                ax.set_xticks([])
            else:
                idxs = np.linspace(0, len(xticks) - 1, 8).astype("int32")
                xt = [fmt(xticks) % h for h in np.asarray(xticks)[idxs]]
                ax.set_xticks(idxs)
                ax.set_xticklabels(xt)

    def plot_spectrogram(self, ax: Axes, method=Subplot.STFT_SPECTROGRAM) -> None:
        if method == Subplot.STFT_SPECTROGRAM:
            spec, freqs, spectral_power = self.stft_component.get_spectrogram(self.audio, self.sr)

        elif method == Subplot.WAVELET_SPECTROGRAM:
            spec, freqs, spectral_power = self.wavelet_component.ssq_cwt_in_chunks(self.audio, self.sr)
            # spec, freqs, spectral_power = self.wavelet_component.cwt_in_chunks(self.audio, self.sr)
            logging.getLogger(__name__).debug("... Done processing cwt")
            # decimation_stride = self.stft_component.hop_length
            # mean_value = np.mean(spectral_power)  # TODO: consider row means
            # padding_length = (decimation_stride - spectral_power.shape[1] % decimation_stride) % decimation_stride
            # padded_spectral_power = np.pad(
            #     spectral_power, ((0, 0), (0, padding_length)), mode="constant", constant_values=mean_value
            # )
            # spectral_power = einx.mean("a (b c) -> a b", padded_spectral_power, c=decimation_stride)
            # # spectral_power = spectral_power[:, ::decimation_stride]
            # spec = spec[:, ::decimation_stride]
            # logging.getLogger(__name__).debug("... Done decimating cwt")

            t = np.linspace(0, self.duration, len(spectral_power))  # TODO: is this needed?
            self._ticks(t, freqs, ax)

        try_pcn = (method == Subplot.STFT_SPECTROGRAM) and self.try_per_channel_normalization
        np_spectral_power = self._normalize_spectral_power(spectral_power, try_pcn, self.clip_outliers)
        s_db = librosa.power_to_db(np_spectral_power, ref=np.max)

        if self.similarity_scores is not None and self.dissimilarity_scores is not None:
            start_index = int(self.time_to_score_index(self.start_time))
            end_index = int(self.time_to_score_index(self.end_time))

            similarity = self.similarity_scores[start_index:end_index].clone()
            dissimilarity = self.dissimilarity_scores[start_index:end_index].clone()

            if similarity.shape[0] < end_index - start_index:
                padding_needed = end_index - start_index - similarity.shape[0]
                similarity = np.pad(similarity, (0, padding_needed), mode="constant", constant_values=0)
                dissimilarity = np.pad(dissimilarity, (0, padding_needed), mode="constant", constant_values=0)
                similarity = torch.tensor(similarity)
                dissimilarity = torch.tensor(dissimilarity)

            mx, mn = np.max(s_db), np.min(s_db)
            normed = (s_db - mn) / (mx - mn)
            s_db_rgb = np.stack((normed, normed, normed), axis=-1)

            spec.shape[1]

            # This says how many samples it will create, but not the start or end.
            stretched_similarity = self.interpolate_1d_tensor(similarity, spec.shape[1])
            stretched_dissimilarity = self.interpolate_1d_tensor(dissimilarity, spec.shape[1])

            redness, greenness, blueness = self._compute_color_channels(
                stretched_similarity, stretched_dissimilarity, self.colormap
            )

            s_db_rgb[:, :, 0] = s_db_rgb[:, :, 0] * redness
            s_db_rgb[:, :, 1] = s_db_rgb[:, :, 1] * greenness
            s_db_rgb[:, :, 2] = s_db_rgb[:, :, 2] * blueness
        else:
            mx, mn = np.max(s_db), np.min(s_db)
            normed = (s_db - mn) / (mx - mn)
            s_db_rgb = np.stack((normed, normed, normed), axis=-1)
            s_db_rgb = normed

        if method == Subplot.WAVELET_SPECTROGRAM:
            ax.set_ylabel("Wavelet Hz")
            self._ticks(t, freqs, ax)
            # ax.imshow(s_db, aspect="auto",cmap='magma',extent=[self.start_time, self.end_time, len(freqs),0],)
            ax.imshow(
                s_db_rgb,
                aspect="auto",
                extent=[self.start_time, self.end_time, len(freqs), 0],
            )

        if method == Subplot.STFT_SPECTROGRAM:
            ax.imshow(
                s_db_rgb,  # .transpose(0, 1, 2),
                aspect="auto",
                origin="lower",
                extent=[self.start_time, self.end_time, freqs[0], freqs[-1]],
            )
            ax.set_ylabel("STFT Hz")

        # else:
        #     librosa.display.specshow(
        #         s_db,
        #         sr=self.sr,
        #         n_fft=self.n_fft,
        #         hop_length=self.hop_length,
        #         x_axis="time",
        #         y_axis="linear",
        #         y_coords=self.freqs,
        #         ax=ax,
        #     )

        if self.label_boxes and method == Subplot.STFT_SPECTROGRAM:
            # TODO: find the write positions for the labels in the wavelet image
            vert_offset = (freqs[-1] - freqs[0]) // 25
            self.add_annotation_boxes(
                self.label_boxes,
                self.start_time,
                self.end_time,
                ax,
                offset=0.1,
                color=(0, 1, 1),
                vert_offset=vert_offset,
            )


class AudioFileVisualizer:
    def __init__(
        self,
        # alternative audio setup
        audio_file: str | None = None,
        # prefered audio setup
        y: np.ndarray | None = None,
        sr: int | None = None,
        *,
        # alternative audio setup
        start_time: float = 0,
        end_time: float | None = None,
        # stft setup
        n_fft: int = 2048,
        freq_range_of_interest: tuple[float, float] | None = None,
        # clasifier/label setup
        feature_rate: int | None = None,
        class_probabilities: torch.Tensor | None = None,
        class_labels: list[str] | None = None,
        label_boxes: list[Any] | None = None,
        target_class: int | None = None,
        #
        resample_to_sr: int | None = None,
        display_time_offset: int = 0,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Intialzed logging")
        if audio_file:
            self.audio, self.sr = librosa.load(
                audio_file, sr=sr, offset=start_time, duration=None if end_time is None else end_time - start_time
            )
        elif y is not None and sr:
            self.audio, self.sr = y, sr

        if resample_to_sr is not None:
            self.audio = librosa.resample(self.audio, orig_sr=self.sr, target_sr=resample_to_sr)
            self.sr = resample_to_sr

        self.audio_duration = self.audio.shape[0] / self.sr
        self.display_time_offset = display_time_offset

        # Default for AVES/HuBERT embeddings
        # from https://arxiv.org/pdf/2106.07447
        self.feature_rate = self.sr // 320 if feature_rate is None else feature_rate
        self.class_probabilities = class_probabilities
        self.class_labels = class_labels
        self.target_class = target_class

        # Compute spectral features
        self.logger.debug("constructing spectrogram")

        self.spectrogram_component = _SpectrogramComponent(
            y=self.audio,
            sr=self.sr,
            class_probabilities=class_probabilities,
            feature_rate=feature_rate,
            target_class=target_class,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            label_boxes=label_boxes,
            freq_range_of_interest=freq_range_of_interest,
            try_per_channel_normalization=True,
        )

        self.enabled_subplots = {
            Subplot.WAVEFORM,
            Subplot.STFT_SPECTROGRAM,
            Subplot.WAVELET_SPECTROGRAM,
            Subplot.SIMILARITIES,
            Subplot.CLASS_PROBABILITIES,
            # Subplot.CLASS_PROBABILITY_LINES,
        }
        self.logger.debug("constructed")

    def time_to_score_index(self, t: float) -> int:
        return t * self.feature_rate

    def score_index_to_time(self, s: int) -> float:
        return s / self.feature_rate

    def interpolate_1d_tensor(self, input_tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        z = input_tensor[None, None, :]
        return torch.nn.functional.interpolate(z, target_length, mode="linear")[0][0]

    def _plot_waveform(self, ax: Axes) -> None:
        times = np.linspace(0, self.audio_duration, len(self.audio))
        ax.plot(times, self.audio, linewidth=0.5)
        ax.set_ylabel("Amplitude")
        # ax.set_xlim(self.start_time, self.duration)
        # ax.set_xticks(np.arange(0, self.duration + 1, 30))

    def _plot_similarities(self, ax: Axes) -> None:
        similarity = self.class_probabilities[:, self.target_class or 1]
        dissimilarity = 1 - similarity
        time_axis = np.arange(similarity.shape[0]) / self.feature_rate + 1 / self.feature_rate / 2
        ax.plot(time_axis, similarity, color="tab:green")
        ax.plot(time_axis, dissimilarity, color="tab:red")
        ax.set_ylabel(self.class_labels[self.target_class or 1])
        ### For efficiency we may want to limit how many samples we give matplotlib later
        # start_index = int(self.time_to_score_index(self.start_time))
        # end_index = int(self.time_to_score_index(self.end_time))
        # fairseq_time = [i * self.feature_rate for i in range(similarity.shape[0])]
        # ax.plot(fairseq_time[start_index:end_index], similarity[start_index:end_index], color="tab:green")
        # ax.plot(fairseq_time[start_index:end_index], dissimilarity[start_index:end_index], color="tab:red")
        # ax.set_xlim(0, self.duration)
        # ax.set_xticks(np.arange(0, self.duration + 1, 30))

    def _plot_class_probabilities(self, ax: Axes) -> None:
        probs = self.class_probabilities
        labels = self.class_labels

        time_axis = np.arange(probs.shape[0]) / self.feature_rate + 1 / self.feature_rate / 2
        ax.stackplot(time_axis, probs.T, labels=labels)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, prop={"size": 8})
        plt.subplots_adjust(bottom=0.5)
        # ax.set_xlim(0, self.duration)
        # ax.set_ylim(0, 1)
        ax.set_ylabel("Cls Prob")
        # ax.set_xticks(np.arange(0, self.duration + 1, 30))

    def _plot_class_probability_lines(self, ax: Axes) -> None:
        probs = self.class_probabilities
        labels = self.class_labels

        time_axis = np.arange(probs.shape[0]) / self.feature_rate + 1 / self.feature_rate / 2
        # ax.stackplot(time_axis, probs.T, labels=labels)
        for i in range(probs.shape[1]):
            ax.plot(time_axis, probs[:, i], label=labels[i])

        ax.legend(loc="upper right")
        # ax.set_xlim(0, self.duration)
        # ax.set_ylim(0, 1)
        ax.set_ylabel("Target Cls")
        # ax.set_xticks(np.arange(0, self.duration + 1, 30))

    def _get_tick_interval(self, duration: float) -> float:
        """Calculate appropriate human-friendly tick interval based on duration."""
        intervals = {1: 0.25, 3: 0.5, 10: 1, 60: 5, 300: 30, 600: 60, 3600: 300, 7200: 600, 86400: 3600}
        return next((v for k, v in intervals.items() if duration <= k), 7200)

    def generate_plot(
        self, height: float = 1280 / 100, width: float = 1920 / 100, save_file: str | None = None
    ) -> None:
        plt.ioff()
        font_size = width * 10 / (1920 / 100)  # font size 10 looks good on that sized plot
        plt.rc("font", size=font_size)

        enabled_subplots = self.enabled_subplots

        if self.class_probabilities is None:
            enabled_subplots.discard(Subplot.SIMILARITIES)
            enabled_subplots.discard(Subplot.CLASS_PROBABILITIES)
            enabled_subplots.discard(Subplot.CLASS_PROBABILITY_LINES)

        n_subplots = len(enabled_subplots)
        height_ratios = [
            1 if Subplot.WAVEFORM in enabled_subplots else 0,
            3 if Subplot.STFT_SPECTROGRAM in enabled_subplots else 0,
            3 if Subplot.WAVELET_SPECTROGRAM in enabled_subplots else 0,
            1 if Subplot.SIMILARITIES in enabled_subplots else 0,
            2 if Subplot.CLASS_PROBABILITIES in enabled_subplots else 0,
            1 if Subplot.CLASS_PROBABILITY_LINES in enabled_subplots else 0,
        ]
        height_ratios = [h for h in height_ratios if h > 0]  # Remove zero heights

        gs = {"height_ratios": height_ratios}
        self.fig, self.axes = plt.subplots(len(height_ratios), 1, sharex=True, figsize=(width, height), gridspec_kw=gs)
        for i in range(len(height_ratios)):
            self.axes[i].set_xlim(60, 120)

        subplot_index = 0
        if Subplot.WAVEFORM in enabled_subplots:
            self.logger.debug("WAVEFORM")
            self._plot_waveform(self.axes[subplot_index])
            subplot_index += 1
        if Subplot.STFT_SPECTROGRAM in enabled_subplots:
            self.logger.debug("STFT_SPECTROGRAM")
            self.spectrogram_component.plot_spectrogram(self.axes[subplot_index])
            subplot_index += 1
        if Subplot.WAVELET_SPECTROGRAM in enabled_subplots:
            self.logger.debug("WAVELET_SPECTROGRAM")
            self.spectrogram_component.plot_spectrogram(self.axes[subplot_index], method=Subplot.WAVELET_SPECTROGRAM)
            subplot_index += 1
        if Subplot.SIMILARITIES in enabled_subplots:
            self.logger.debug("SIMILARITIES")
            self._plot_similarities(self.axes[subplot_index])
            subplot_index += 1
        if Subplot.CLASS_PROBABILITIES in enabled_subplots:
            self.logger.debug("CLASS_PROBABILITIES")
            self._plot_class_probabilities(self.axes[subplot_index])
            subplot_index += 1
        if Subplot.CLASS_PROBABILITY_LINES in enabled_subplots:
            self.logger.debug("CLASS_PROBABILITY_LINES")
            self._plot_class_probability_lines(self.axes[subplot_index])

        for i in range(n_subplots):
            self.axes[i].set_xlim(60, 120)

        for ax in self.axes:
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            # ax.set_xlim(self.start_time, self.end_time)

        self.logger.debug("axes formatting")
        display_time_offset = self.display_time_offset

        def format_time(x, _pos):
            x += display_time_offset  # needs to stay out of the loop because of Ruff:B023
            hours = int(x // 60 // 60)
            minutes = int(x // 60 % 60)
            seconds = x % 60
            if seconds == int(seconds):
                return f"{minutes:02}:{int(seconds):02}" if hours < 1 else f"{hours}:{minutes:02}:{int(seconds):02}"
            else:  # noqa: RET505 , easier to read with the else
                return f"{minutes:02}:{seconds:06.3f}" if hours < 1 else f"{hours}:{minutes:02}:{seconds:06.3f}"

        for ax in self.axes[-1:]:
            ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
            ax.set_xlabel("Time")
            tick_interval = self._get_tick_interval(self.end_time - self.start_time)
            ax.set_xticks(np.arange(self.start_time % tick_interval, self.end_time + 1, tick_interval))
            from matplotlib.ticker import FuncFormatter

            ax.xaxis.set_major_formatter(FuncFormatter(format_time))

        hour, minute, second = (
            int(self.start_time // 60 // 60),
            int(self.start_time // 60) % 60,
            int(self.start_time) % 60,
        )
        displaytime = f"{hour:02}:{minute:02}:{second:02}" if hour > 0 else f"{minute:02}:{second:02}"
        # plt.gca().set_xticks(np.arange(88, 3*60, 30))
        plt.gca().set_xlim(self.start_time, self.end_time)
        plt.subplots_adjust(top=0.93, left=0)
        self.fig.suptitle(f"{self.title}", fontsize=16, ha="left", x=0)
        self.logger.debug("  saving %s", displaytime)

        if save_file:
            plt.savefig(save_file, bbox_inches="tight", pad_inches=0.02)
            plt.close()
            plt.close("all")
        self.logger.info("  visualizations saved to %s", save_file)
        return plt

    def setup_plot(self, title: str, start_time: float, end_time: float) -> None:
        self.title = title
        # self.save_file = save_file
        self.start_time = start_time
        self.end_time = end_time
        # self.width = width
        # self.height = height
        self.duration = end_time - start_time
        self.audio_duration = self.audio.shape[0] / self.sr

    def visualize_audio_file_fragment(
        self,
        title: str,
        *,
        save_file: str | None = None,
        start_time: float = 0,
        end_time: float | None = None,
        height: float = 1280 / 100,
        width: float = 1920 / 100,
    ) -> None:
        # For backwards compatibility
        if end_time is None:
            end_time = self.audio_duration
        self.setup_plot(title, start_time, end_time)
        return self.generate_plot(width=width, height=height, save_file=save_file)
