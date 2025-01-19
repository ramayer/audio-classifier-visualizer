from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class AudioFileVisualizer:
    def __init__(self) -> None:
        """Initialize the AudioFileVisualizer."""
        print("AAAAAAAAAAAAAAAAAAAAB")
        self.logger = logging.getLogger(__name__)

    def interpolate_1d_tensor(self, input_tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        z = input_tensor[None, None, :]
        return torch.nn.functional.interpolate(z, target_length)[0][0]

    def add_annotation_boxes(
        self,
        labels: list[Any],
        patch_start: float,
        patch_end: float,
        axarr: Axes,
        offset: float = 0.2,
        only: float | None = None,
        color: tuple[float, float, float] = (0.0, 1.0, 1.0),
    ) -> None:
        """Add annotation boxes to the visualization.

        Args:
            labels: List of annotation labels
            patch_start: Start time of the patch
            patch_end: End time of the patch
            axarr: Matplotlib axes to draw on
            offset: Offset for box drawing (default: 0.2)
            only: If set, only draw box for this time point
            color: RGB color tuple for the boxes
        """
        for row in labels:
            bt, et, lf, hf, dur, fn, tags, notes, tag1, tag2, score, raven_file = dataclasses.astuple(row)
            if et < patch_start:
                continue
            if bt > patch_end:
                continue
            if only is not None and only != bt:
                continue
            rect = patches.Rectangle(
                (bt - patch_start - offset, lf - 5),
                (et - bt + offset * 2),
                (hf - lf + 10),
                linewidth=3,
                edgecolor=(0, 0, 0),
                facecolor="none",
            )
            axarr.add_patch(rect)
            rect = patches.Rectangle(
                (bt - patch_start - offset, lf - 5),
                (et - bt + offset * 2),
                (hf - lf + 10),
                linewidth=1,
                edgecolor=color,
                facecolor="none",
            )
            axarr.add_patch(rect)
    def _compute_spectral_power(self, audio: np.ndarray, sr: int, n_fft: int, hop_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        spec = librosa.stft(audio, n_fft=n_fft, win_length=n_fft, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        np_spectral_power = np.abs(spec) ** 2  # type: ignore
        return spec, freqs, np_spectral_power

    def _normalize_spectral_power(self, np_spectral_power: np.ndarray, try_per_channel_normalization: bool, clip_outliers: bool) -> np.ndarray:
        if try_per_channel_normalization:
            median_pwr_per_spectral_band = np.median(np_spectral_power, axis=1)
            np_spectral_power = np_spectral_power / median_pwr_per_spectral_band[:, None]

        if clip_outliers:
            noise_floor = np.percentile(np_spectral_power, 0.1)
            clip_level = np.percentile(np_spectral_power, 99.9)
            db_normalized = np_spectral_power
            db_normalized[np_spectral_power < noise_floor] = noise_floor
            db_normalized[np_spectral_power > clip_level] = clip_level
            np_spectral_power = db_normalized

        return np_spectral_power

    def _compute_color_channels(self, stretched_similarity: torch.Tensor, stretched_dissimilarity: torch.Tensor, colormap: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _create_plot(self, s_db_rgb: np.ndarray, sr: int, n_fft: int, hop_length: int, freqs: np.ndarray, 
                    duration: float, actual_duration: float, similarity: torch.Tensor,
                    similarity_scoresz: torch.Tensor, dissimilarity_scoresz: torch.Tensor,
                    start_index: int, end_index: int, labels: list, negative_labels: list,
                    start_time: float, end_time: float, width: float, height: float,
                    title: str, save_file: str) -> None:
        plt.ioff()
        fs = (width, height)
        gs = {"height_ratios": [3, 1]}
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=fs, gridspec_kw=gs)  # type: ignore

        librosa.display.specshow(
            s_db_rgb[:, :],
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            x_axis="time",
            y_axis="log",
            y_coords=freqs,
            ax=ax1,
        )
        plt.gca().set_xticks(np.arange(0, duration, 30))

        self.add_annotation_boxes(labels, start_time, end_time, ax1, offset=0.5, color=(0, 1, 1))
        self.add_annotation_boxes(negative_labels, start_time, end_time, ax1, offset=0.5, color=(0, 0, 1))

        fairseq_time = [i * actual_duration / similarity.shape[0] for i in range(similarity.shape[0])]
        ax2.plot(fairseq_time, similarity_scoresz[start_index:end_index], color="tab:green")
        ax2.plot(fairseq_time, dissimilarity_scoresz[start_index:end_index], color="tab:red")
        ax1.set_xlim(0, duration)
        ax2.set_xlim(0, duration)

        hour, minute, _second = int(start_time // 60 // 60), int(start_time // 60) % 60, start_time % 60
        displaytime = f"{hour:02}:{minute:02}"

        plt.subplots_adjust(top=0.93, left=0)
        fig.suptitle(f"{title}", fontsize=16, ha="left", x=0)
        self.logger.debug("  saving %s", displaytime)

        plt.savefig(save_file, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        plt.close("all")
        self.logger.info("  visualizations saved to %s", save_file)

    def visualize_audio_file_fragment(
        self,
        title,
        save_file,
        audio_file,
        similarity_scoresz,
        dissimilarity_scoresz,
        audio_file_processor,  # : afp.AudioFileProcessor,
        start_time=0,
        end_time=60 * 6,
        height=1280 / 100,
        width=1920 / 100,
        colormap="raw",
        labels=None,
        negative_labels=None,
        try_per_channel_normalization_on_power: bool = True,
        clip_outliers: bool = True,
    ):
        import time

        if negative_labels is None:
            negative_labels = []
        if labels is None:
            labels = []
        t0 = time.time()
        start_index = audio_file_processor.time_to_score_index(start_time)
        end_index = audio_file_processor.time_to_score_index(end_time)
        similarity = similarity_scoresz[start_index:end_index].clone()
        dissimilarity = dissimilarity_scoresz[start_index:end_index].clone()

        n_fft = 2048
        hop_length = n_fft // 4
        duration = end_time - start_time
        audio, sr = librosa.load(audio_file, sr=2000, offset=start_time, duration=end_time - start_time)
        actual_duration = audio.shape[0] / sr
        self.logger.debug("  loaded audio in %s", time.time() - t0)
        self.logger.debug("  duration intended=%s actual=%s", duration, actual_duration)

        spec, freqs, np_spectral_power = self._compute_spectral_power(audio, sr, n_fft, hop_length)
        self.logger.debug("  did stft in %s", time.time() - t0)

        np_spectral_power = self._normalize_spectral_power(np_spectral_power, try_per_channel_normalization_on_power, clip_outliers)
        s_db = librosa.power_to_db(np_spectral_power, ref=np.max)

        mx = np.max(s_db)
        mn = np.min(s_db)
        normed = (s_db - mn) / (mx - mn)
        s_db_rgb = np.stack((normed, normed, normed), axis=-1)
        self.logger.debug("  coloring at %s", time.time() - t0)

        stretched_similarity = self.interpolate_1d_tensor(similarity, spec.shape[1])
        stretched_dissimilarity = self.interpolate_1d_tensor(dissimilarity, spec.shape[1])

        redness, greenness, blueness = self._compute_color_channels(stretched_similarity, stretched_dissimilarity, colormap)

        s_db_rgb[:, :, 0] = s_db_rgb[:, :, 0] * redness
        s_db_rgb[:, :, 1] = s_db_rgb[:, :, 1] * greenness
        s_db_rgb[:, :, 2] = s_db_rgb[:, :, 2] * blueness

        self.logger.debug("  plotting at %s", time.time() - t0)

        self._create_plot(s_db_rgb, sr, n_fft, hop_length, freqs, duration, actual_duration,
                         similarity, similarity_scoresz, dissimilarity_scoresz, start_index, end_index,
                         labels, negative_labels, start_time, end_time, width, height, title, save_file)


# AudioFileVisualizer().visualize_audio_file_fragment(
#     f"{audio_file} and scores",
#     '/home/ron/proj/elephantlistening/tmp/aves/test.png',
#     audio_file,
#     scores[:,1],
#     scores[:,0],
#     afp,
#     start_time=0,
#     end_time=60*60
# )
