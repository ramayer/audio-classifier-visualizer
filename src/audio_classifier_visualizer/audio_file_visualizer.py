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
        log_scale: bool = False,  # noqa: FBT001 FBT002
        try_per_channel_normalization: bool = True,  # noqa: FBT001 FBT002
        clip_outliers: bool = True,  # noqa: FBT001 FBT002
        labels: list[Any] | None = None,
        negative_labels: list[Any] | None = None,
    ):
        self.audio = y
        self.sr = sr
        self.class_probabilities = class_probabilities
        self.feature_rate = feature_rate if feature_rate is not None else sr // 320
        self.target_class = target_class
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.colormap = colormap
        self.log_scale = log_scale
        self.try_per_channel_normalization = try_per_channel_normalization
        self.clip_outliers = clip_outliers

        # TODO: single array of labels of many classes
        self.labels = labels
        self.negative_labels = negative_labels

        self.similarity_scores = class_probabilities.cpu()[:, 1]
        self.dissimilarity_scores = class_probabilities.cpu()[:, 0]

        self.duration = self.audio.shape[0] / self.sr
        self.spec = librosa.stft(self.audio, n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length)
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.spectral_power = np.abs(self.spec) ** 2  # type: ignore
        # This component shouldn't care much about absolute times, except for adding labels
        self.start_time = 0
        self.end_time = self.audio.shape[0] / self.sr

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

    def _normalize_spectral_power(
        self,
        np_spectral_power: np.ndarray,
        try_per_channel_normalization: bool,  # noqa: FBT001
        clip_outliers: bool,  # noqa: FBT001
    ) -> np.ndarray:  # FBT001
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

    def plot_spectrogram(self, ax: Axes) -> None:
        np_spectral_power = self._normalize_spectral_power(
            self.spectral_power, self.try_per_channel_normalization, self.clip_outliers
        )
        s_db = librosa.power_to_db(np_spectral_power, ref=np.max)
        # component["similarity"] = self.similarity_scores
        # component["dissimilarity"] = self.dissimilarity_scores

        if self.similarity_scores is not None and self.dissimilarity_scores is not None:
            start_index = int(self.time_to_score_index(self.start_time))
            end_index = int(self.time_to_score_index(self.end_time))
            similarity = self.similarity_scores[start_index:end_index].clone()
            dissimilarity = self.dissimilarity_scores[start_index:end_index].clone()

            mx, mn = np.max(s_db), np.min(s_db)
            normed = (s_db - mn) / (mx - mn)
            s_db_rgb = np.stack((normed, normed, normed), axis=-1)

            stretched_similarity = self.interpolate_1d_tensor(similarity, self.spec.shape[1])
            stretched_dissimilarity = self.interpolate_1d_tensor(dissimilarity, self.spec.shape[1])

            redness, greenness, blueness = self._compute_color_channels(
                stretched_similarity, stretched_dissimilarity, self.colormap
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
                y_axis="log" if self.log_scale else "linear",
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
                y_axis="log" if self.log_scale else "linear",
                y_coords=self.freqs,
                ax=ax,
            )

        ax.set_xticks(np.arange(0, self.duration + 1, 30))

        if self.labels:
            self.add_annotation_boxes(self.labels, self.start_time, self.end_time, ax, offset=0.5, color=(0, 1, 1))
        if self.negative_labels:
            self.add_annotation_boxes(
                self.negative_labels, self.start_time, self.end_time, ax, offset=0.5, color=(0, 0, 1)
            )


class AudioFileVisualizer:
    def __init__(
        self,
        audio_file: str,
        start_time: float = 0,
        end_time: float | None = None,
        n_fft: int = 2048,
        sr: int = 2000,
        feature_rate: int | None = None,
        class_probabilities: torch.Tensor | None = None,
        class_labels: list[str] | None = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        # Load audio
        self.audio, self.sr = librosa.load(
            audio_file, sr=sr, offset=start_time, duration=None if end_time is None else end_time - start_time
        )
        self.duration = self.audio.shape[0] / self.sr
        self.audio_duration = self.audio.shape[0] / self.sr

        # Default for AVES/HuBERT embeddings
        # from https://arxiv.org/pdf/2106.07447
        self.feature_rate = self.sr // 320 if feature_rate is None else feature_rate
        self.class_probabilities = class_probabilities
        self.class_labels = class_labels

        # Compute spectral features
        self.spectrogram_component = _SpectrogramComponent(
            y=self.audio,
            sr=self.sr,
            class_probabilities=class_probabilities,
            feature_rate=feature_rate,
            target_class=0,
            n_fft=n_fft,
            hop_length=n_fft // 4,
        )

        # Initialize plot components
        self.plot_components = []
        self.fig = None
        self.axes = []
        self.title = ""
        self.save_file = ""
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

    def setup_plot(
        self, title: str, save_file: str, start_time: float, end_time: float, width: float, height: float
    ) -> None:
        self.title = title
        self.save_file = save_file
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        self.duration = end_time - start_time
        self.audio_duration = self.audio.shape[0] / self.sr

    def add_waveform(self) -> None:
        self.plot_components.append("waveform")

    def add_spectrogram(self) -> None:  # FBT001
        self.plot_components.append(
            {
                "type": "spectrogram",
            }
        )

    def add_similarities(self, similarity_scoresz: torch.Tensor, dissimilarity_scoresz: torch.Tensor) -> None:
        self.plot_components.append(
            {"type": "similarities", "similarity": similarity_scoresz, "dissimilarity": dissimilarity_scoresz}
        )

    def add_class_probabilities(self) -> None:
        if self.class_probabilities is not None:
            self.plot_components.append(
                {"type": "class_probabilities", "probabilities": self.class_probabilities, "labels": self.class_labels}
            )

    def _plot_waveform(self, ax: Axes) -> None:
        times = np.linspace(0, self.audio_duration, len(self.audio))
        ax.plot(times, self.audio, linewidth=0.5)
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, self.duration)
        ax.set_xticks(np.arange(0, self.duration + 1, 30))

    def _plot_similarities(self, ax: Axes, component: dict) -> None:
        start_index = int(self.time_to_score_index(self.start_time))
        end_index = int(self.time_to_score_index(self.end_time))
        fairseq_time = [
            i * self.audio_duration / component["similarity"].shape[0] for i in range(component["similarity"].shape[0])
        ]
        end_index = len(fairseq_time)
        ax.plot(fairseq_time, component["similarity"][start_index:end_index], color="tab:green")
        ax.plot(fairseq_time, component["dissimilarity"][start_index:end_index], color="tab:red")
        ax.set_xlim(0, self.duration)
        ax.set_xticks(np.arange(0, self.duration + 1, 30))

    def _plot_class_probabilities(self, ax: Axes) -> None:
        probs = self.class_probabilities
        labels = self.class_labels

        time_axis = np.arange(probs.shape[0]) / self.feature_rate
        ax.stackplot(time_axis, probs.T, labels=labels)
        ax.legend(loc="upper right")
        ax.set_xlim(0, self.duration)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Class Probabilities")
        ax.set_xticks(np.arange(0, self.duration + 1, 30))

    def _plot_class_probability_lines(self, ax: Axes) -> None:
        probs = self.class_probabilities
        labels = self.class_labels

        time_axis = np.arange(probs.shape[0]) / self.feature_rate
        # ax.stackplot(time_axis, probs.T, labels=labels)
        for i in range(probs.shape[1]):
            ax.plot(time_axis, probs[:, i], label=labels[i])

        ax.legend(loc="upper right")
        ax.set_xlim(0, self.duration)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Class Probabilities")
        ax.set_xticks(np.arange(0, self.duration + 1, 30))

    def _get_tick_interval(self, duration: float) -> float:
        """Calculate appropriate human-friendly tick interval based on duration."""
        intervals = {10: 1, 60: 5, 300: 30, 3600: 300, 7200: 600, 86400: 3600}
        return next((v for k, v in intervals.items() if duration <= k), 7200)

    def generate_plot(self) -> None:
        plt.ioff()
        n_subplots = len(self.plot_components)
        if n_subplots == 0:
            msg = "No plot components added"
            raise ValueError(msg)

        height_ratios = [
            1 if comp == "waveform" or comp.get("type") in ["similarities", "class_probabilities"] else 3
            for comp in self.plot_components
        ]
        gs = {"height_ratios": height_ratios}
        self.fig, self.axes = plt.subplots(
            n_subplots, 1, sharex=True, figsize=(self.width, self.height), gridspec_kw=gs
        )
        if n_subplots == 1:
            self.axes = [self.axes]

        for ax, component in zip(self.axes, self.plot_components):
            if component == "waveform":
                self._plot_waveform(ax)
            elif component.get("type") == "spectrogram":
                self.spectrogram_component.plot_spectrogram(ax)
            elif component.get("type") == "similarities":
                self._plot_similarities(ax, component)
            elif component.get("type") == "class_probabilities":
                self._plot_class_probabilities(ax)
                # self._plot_class_probability_lines(ax)

        for ax in self.axes[-1:-1]:
            ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
            ax.set_xlabel("Time")

            tick_interval = self._get_tick_interval(self.duration)
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
        _audio_file_processor,  # : afp.AudioFileProcessor,  # deprecate then delete
        start_time: float = 0,
        end_time: float = 60 * 6,
        height: float = 1280 / 100,
        width: float = 1920 / 100,
        _colormap: str = "raw",
        _labels: list | None = None,
        _negative_labels: list | None = None,
        _try_per_channel_normalization_on_power: bool = True,  # noqa: FBT001 FBT002
        _clip_outliers: bool = True,  # noqa: FBT001 FBT002
        _log_scale: bool = False,  # noqa: FBT001 FBT002
    ) -> None:
        # For backwards compatibility
        if end_time is None:
            end_time = self.duration
        self.setup_plot(title, save_file, start_time, end_time, width, height)
        self.add_waveform()
        self.add_spectrogram()
        self.add_similarities(similarity_scoresz, dissimilarity_scoresz)
        if self.class_probabilities is not None:
            self.add_class_probabilities()
        self.generate_plot()


if __name__ == "__main__":
    import numpy as np
    import torch

    import audio_classifier_visualizer as acv

    audio_path = "/tmp/test.wav"  # noqa: S108

    y, sr = librosa.load(audio_path, sr=None, duration=60 * 5)
    expected_embedding_length = y.shape[0] // 320

    scores_yes = torch.linspace(0, 1, expected_embedding_length)
    scores_no = torch.linspace(1, 0, expected_embedding_length)

    probs = torch.stack([scores_yes, scores_no], dim=0).T

    afv = acv.AudioFileVisualizer(
        audio_path,
        n_fft=512,
        start_time=0,
        end_time=60 * 5,
        sr=2000,
        class_probabilities=probs,
        class_labels=["yes", "no"],
    )  # Added class probabilities here

    afv.visualize_audio_file_fragment(
        "demo",
        "/tmp/1.png",  # noqa: S108
        # audio_path,
        scores_yes,
        scores_no,
        None,
        start_time=0,
        end_time=60 * 5,
        colormap="raw",
        width=10,
        height=5,
        labels=None,  # Add the demo labels
    )
    import IPython.display as ipd
    from PIL import Image

    ipd.display(Image.open("/tmp/1.png"))  # noqa: S108
