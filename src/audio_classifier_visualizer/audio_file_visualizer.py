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
    def __init__(self, audio_file: str, 
                 start_time: float = 0, end_time: float | None = None, 
                 n_fft: int = 2048,
                 sr: int = 2000, feature_rate: int|None = None) -> None:
        """Initialize the AudioFileVisualizer.
        
        Args:
            audio_file: Path to audio file to visualize
            start_time: Start time in seconds to load from
            end_time: End time in seconds to load until (default: entire file)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load audio
        self.audio, self.sr = librosa.load(audio_file, sr=sr, offset=start_time, duration=None if end_time is None else end_time - start_time)
        self.duration = self.audio.shape[0] / self.sr
        
        # Compute spectral features
        self.n_fft = n_fft
        self.hop_length = self.n_fft // 4

        if feature_rate is None:
            # Default for AVES/HuBERT embeddings
            # from https://arxiv.org/pdf/2106.07447
            default_feature_rate = 320
            self.feature_rate = self.sr // default_feature_rate
        else:
            self.feature_rate = feature_rate
        
        # Compute STFT and spectral power
        self.spec = librosa.stft(self.audio, n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length)
        #self.spec = librosa.stft(self.audio)
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.spectral_power = np.abs(self.spec) ** 2  # type: ignore

        # Initialize plot components
        self.plot_components = []
        self.fig = None
        self.axes = []
        self.title = ""
        self.save_file = ""
        self.duration = 0
        self.actual_duration = 0
        self.start_time = 0
        self.end_time = 0
        self.width = 0
        self.height = 0

    def time_to_score_index(self, t: float) -> int:
        return t * self.feature_rate

    def score_index_to_time(self, s: int) -> float:
        return s / self.feature_rate

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
        color: tuple[float, float, float] = (0.0, 1.0, 1.0),
    ) -> None:
        """Add annotation boxes to the visualization.

        Args:
            labels: List of annotation labels
            patch_start: Start time of the patch
            patch_end: End time of the patch
            axarr: Matplotlib axes to draw on
            offset: Offset for box drawing (default: 0.2)
            color: RGB color tuple for the boxes
        """
        for row in labels:
            bt, et, lf, hf, dur, fn, tags, notes, tag1, tag2, score, raven_file = dataclasses.astuple(row)
            if et < patch_start:
                continue
            if bt > patch_end:
                continue
            xy = (bt - patch_start - offset, lf - 5)
            width = et - bt + offset * 2
            height = hf - lf + 10
            
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

    def setup_plot(self, title: str, save_file: str, start_time: float, end_time: float, width: float, height: float) -> None:
        self.title = title
        self.save_file = save_file
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        self.duration = end_time - start_time
        self.actual_duration = self.audio.shape[0] / self.sr

    def add_waveform(self) -> None:
        self.plot_components.append("waveform")

    def add_spectrogram(self, labels: list | None = None, negative_labels: list | None = None,
                       try_per_channel_normalization: bool = True, clip_outliers: bool = True,
                       similarity_scoresz: torch.Tensor | None = None,
                       dissimilarity_scoresz: torch.Tensor | None = None,
                       colormap: str = "raw",
                       log_scale: bool = False) -> None:
        self.plot_components.append({
            "type": "spectrogram",
            "labels": labels or [],
            "negative_labels": negative_labels or [],
            "normalize": try_per_channel_normalization,
            "clip_outliers": clip_outliers,
            "similarity": similarity_scoresz,
            "dissimilarity": dissimilarity_scoresz,
            "colormap": colormap,
            "log_scale": log_scale
        })

    def add_similarities(self, similarity_scoresz: torch.Tensor, dissimilarity_scoresz: torch.Tensor) -> None:
        self.plot_components.append({
            "type": "similarities",
            "similarity": similarity_scoresz,
            "dissimilarity": dissimilarity_scoresz
        })

    def _plot_waveform(self, ax: Axes) -> None:
        times = np.linspace(0, self.actual_duration, len(self.audio))
        ax.plot(times, self.audio, linewidth=0.5)
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, self.duration)
        ax.set_xticks(np.arange(0, self.duration + 1, 30))

    def _plot_spectrogram(self, ax: Axes, component: dict) -> None:
        np_spectral_power = self._normalize_spectral_power(
            self.spectral_power, 
            component["normalize"],
            component["clip_outliers"]
        )
        s_db = librosa.power_to_db(np_spectral_power, ref=np.max)
        
        if component["similarity"] is not None and component["dissimilarity"] is not None:
            start_index = self.time_to_score_index(self.start_time)
            end_index = self.time_to_score_index(self.end_time)
            similarity = component["similarity"][start_index:end_index].clone()
            dissimilarity = component["dissimilarity"][start_index:end_index].clone()
            
            mx, mn = np.max(s_db), np.min(s_db)
            normed = (s_db - mn) / (mx - mn)
            s_db_rgb = np.stack((normed, normed, normed), axis=-1)
            
            stretched_similarity = self.interpolate_1d_tensor(similarity, self.spec.shape[1])
            stretched_dissimilarity = self.interpolate_1d_tensor(dissimilarity, self.spec.shape[1])
            
            redness, greenness, blueness = self._compute_color_channels(
                stretched_similarity, stretched_dissimilarity, component["colormap"]
            )
            
            s_db_rgb[:, :, 0] = s_db_rgb[:, :, 0] * redness
            s_db_rgb[:, :, 1] = s_db_rgb[:, :, 1] * greenness
            s_db_rgb[:, :, 2] = s_db_rgb[:, :, 2] * blueness
            
            librosa.display.specshow(
                s_db_rgb[:, :],
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                x_axis="time",
                y_axis="log" if component["log_scale"] else "linear",
                y_coords=self.freqs,
                ax=ax,
            )
        else:
            librosa.display.specshow(
                s_db,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                x_axis="time",
                y_axis="log" if component["log_scale"] else "linear",
                y_coords=self.freqs,
                ax=ax,
            )

        ax.set_xticks(np.arange(0, self.duration + 1, 30))
        
        if component["labels"]:
            self.add_annotation_boxes(component["labels"], self.start_time, self.end_time, ax, offset=0.5, color=(0, 1, 1))
        if component["negative_labels"]:
            self.add_annotation_boxes(component["negative_labels"], self.start_time, self.end_time, ax, offset=0.5, color=(0, 0, 1))

    def _plot_similarities(self, ax: Axes, component: dict) -> None:
        start_index = self.time_to_score_index(self.start_time)
        end_index = self.time_to_score_index(self.end_time)
        fairseq_time = [i * self.actual_duration / component["similarity"].shape[0] for i in range(component["similarity"].shape[0])]
        print(f"end_time = {self.end_time}, end_index = {end_index}, similarity.shape = {component['similarity'].shape}, len(fairseq_time) = {len(fairseq_time)}")
        end_index = len(fairseq_time)
        ax.plot(fairseq_time, component["similarity"][start_index:end_index], color="tab:green")
        ax.plot(fairseq_time, component["dissimilarity"][start_index:end_index], color="tab:red")
        ax.set_xlim(0, self.duration)
        ax.set_xticks(np.arange(0, self.duration + 1, 30))

    def generate_plot(self) -> None:
        plt.ioff()
        n_subplots = len(self.plot_components)
        if n_subplots == 0:
            raise ValueError("No plot components added")

        height_ratios = [1 if comp == "waveform" or comp.get("type") == "similarities" else 3 
                        for comp in self.plot_components]
        gs = {"height_ratios": height_ratios}
        self.fig, self.axes = plt.subplots(n_subplots, 1, sharex=True, 
                                         figsize=(self.width, self.height), 
                                         gridspec_kw=gs)
        if n_subplots == 1:
            self.axes = [self.axes]

        for ax, component in zip(self.axes, self.plot_components):
            if component == "waveform":
                self._plot_waveform(ax)
            elif component.get("type") == "spectrogram":
                self._plot_spectrogram(ax, component)
            elif component.get("type") == "similarities":
                self._plot_similarities(ax, component)

        # Enable x-axis ticks and labels for all subplots
        for ax in self.axes:
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
            ax.set_xlabel('Time')
            print(f"duration = {self.duration}")
            # Set x-ticks based on duration
            if self.duration <= 10:  # For very short clips
                tick_interval = 1  # One tick per second
            elif self.duration <= 60:  # Up to 1 minute
                tick_interval = 5  # Tick every 5 seconds
            elif self.duration <= 300:  # Up to 5 minutes
                tick_interval = 30  # Tick every 30 seconds
            elif self.duration <= 3600:  # Up to 1 hour
                tick_interval = 300  # Tick every 5 minutes
            elif self.duration <= 7200:  # Up to 2 hours
                tick_interval = 600  # Tick every 10 minutes
            elif self.duration <= 86400:  # Up to 24 hours
                tick_interval = 3600  # Tick every hour
            else:  # More than 24 hours
                tick_interval = 7200  # Tick every 2 hours
                
            ax.set_xticks(np.arange(0, self.duration + 1, tick_interval))

        hour, minute = int(self.start_time // 60 // 60), int(self.start_time // 60) % 60
        displaytime = f"{hour:02}:{minute:02}"

        plt.subplots_adjust(top=0.93, left=0)
        self.fig.suptitle(f"{self.title}", fontsize=16, ha="left", x=0)
        self.logger.debug("  saving %s", displaytime)

        plt.savefig(self.save_file, bbox_inches="tight", pad_inches=0.02)
        plt.close()
        plt.close("all")
        self.logger.info("  visualizations saved to %s", self.save_file)



    def visualize_audio_file_fragment(
        self,
        title: str,
        save_file: str,
        similarity_scoresz: torch.Tensor,
        dissimilarity_scoresz: torch.Tensor,
        audio_file_processor,  # : afp.AudioFileProcessor,
        start_time: float = 0,
        end_time: float = 60 * 6,
        height: float = 1280 / 100,
        width: float = 1920 / 100,
        colormap: str = "raw",
        labels: list | None = None,
        negative_labels: list | None = None,
        try_per_channel_normalization_on_power: bool = True,
        clip_outliers: bool = True,
        log_scale: bool = False,
    ) -> None:
        # For backwards compatibility
        self.setup_plot(title, save_file, start_time, end_time, width, height)
        self.add_waveform()
        self.add_spectrogram(labels, negative_labels, try_per_channel_normalization_on_power,
                           clip_outliers, similarity_scoresz, dissimilarity_scoresz, colormap, log_scale)
        self.add_similarities(similarity_scoresz, dissimilarity_scoresz)
        self.generate_plot()
