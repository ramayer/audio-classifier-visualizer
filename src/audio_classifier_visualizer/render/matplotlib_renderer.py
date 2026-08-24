"""Static-image renderer. First (and simplest) implementation of the Renderer protocol.

Good fit for: Jupyter notebooks, batch report generation, thumbnails of a
whole day. Not interactive -- for drag-zoom/pan/hover, a future renderer
(e.g. Bokeh/HoloViews+Datashader) implements the same protocol and slots in
without touching this file or anything in core/features.
"""

from __future__ import annotations

import logging

import numpy as np

from audio_classifier_visualizer.features.colorize import confidence_to_rgb, normalize_power, power_to_db
from audio_classifier_visualizer.render.spec import Track, VisualizationSpec

logger = logging.getLogger(__name__)

_TICK_INTERVALS = ((1, 0.25), (3, 0.5), (10, 1), (60, 5), (300, 30), (600, 60), (3600, 300), (7200, 600))


def _tick_interval(duration: float) -> float:
    return next((v for limit, v in _TICK_INTERVALS if duration <= limit), 3600)


class MatplotlibRenderer:
    def render(
        self,
        spec: VisualizationSpec,
        *,
        width: float = 19.2,
        height: float = 12.8,
        save_file: str | None = None,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        plt.ioff()
        height_per_track = {
            Track.WAVEFORM: 2,
            Track.STFT_SPECTROGRAM: 3,
            Track.WAVELET_SPECTROGRAM: 3,
            Track.SIMILARITY_LINES: 1,
            Track.CLASS_PROBABILITY_STACK: 2,
        }
        tracks = [t for t in spec.tracks if t != Track.SIMILARITY_LINES or spec.classifier_output is not None]
        tracks = [t for t in tracks if t != Track.CLASS_PROBABILITY_STACK or spec.classifier_output is not None]
        ratios = [height_per_track[t] for t in tracks]

        fig, axes = plt.subplots(
            len(tracks), 1, sharex=True, figsize=(width, height), gridspec_kw={"height_ratios": ratios}
        )
        axes = list(np.atleast_1d(axes))

        font_size = width * 10 / 19.2
        plt.rc("font", size=font_size)

        for ax, track in zip(axes, tracks, strict=True):
            self._draw_track(ax, track, spec)

        for ax in axes:
            ax.set_xlim(spec.start_time, spec.end_time)
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        last_ax = axes[-1]
        last_ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
        last_ax.set_xlabel("Time")
        interval = _tick_interval(spec.end_time - spec.start_time)
        last_ax.set_xticks(
            np.arange(spec.start_time - (spec.start_time % interval), spec.end_time + interval, interval)
        )
        last_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: spec.audio.time_axis.format_relative(x)))
        # set_xticks on a sharex=True group re-expands the shared xlim to include any
        # tick location outside the current view (confirmed against matplotlib
        # directly) -- undoing the set_xlim() calls above. Re-assert it last, after
        # ticks are placed, so the displayed range matches what was actually
        # requested rather than snapping out to the next tick interval.
        for ax in axes:
            ax.set_xlim(spec.start_time, spec.end_time)

        fig.suptitle(spec.title, fontsize=16, ha="left", x=0)
        right_margin = 0.85 if Track.CLASS_PROBABILITY_STACK in tracks else 0.98
        plt.subplots_adjust(top=0.93, left=0.06, right=right_margin)

        if save_file:
            fig.savefig(save_file, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            logger.info("saved visualization to %s", save_file)
            return None
        # Close from pyplot's global figure registry so Jupyter's inline-backend
        # auto-display hook doesn't render this figure a second time in addition
        # to the return-value repr -- the Figure object itself still renders fine
        # (its canvas isn't touched by plt.close), so returning it for display
        # still works.
        plt.close(fig)
        return fig

    def _draw_track(self, ax, track: Track, spec: VisualizationSpec) -> None:
        if track == Track.WAVEFORM:
            self._draw_waveform(ax, spec)
        elif track == Track.STFT_SPECTROGRAM:
            self._draw_spectrogram(ax, spec, wavelet=False)
        elif track == Track.WAVELET_SPECTROGRAM:
            self._draw_spectrogram(ax, spec, wavelet=True)
        elif track == Track.SIMILARITY_LINES:
            self._draw_similarity_lines(ax, spec)
        elif track == Track.CLASS_PROBABILITY_STACK:
            self._draw_class_probability_stack(ax, spec)
        else:
            msg = f"Unhandled track type: {track}"
            raise ValueError(msg)

    def _draw_waveform(self, ax, spec: VisualizationSpec) -> None:
        from matplotlib.colors import to_rgba

        y = spec.audio.as_mono() if spec.audio.n_channels > 1 else spec.audio.channel(0)
        times = np.linspace(spec.start_time, spec.end_time, len(y))
        ax.plot(times, y, linewidth=0.5, color="black")
        for point in spec.point_labels:
            # Thin opaque ring + translucent fill (not a solid dot) so the waveform
            # underneath the marker stays visible -- a solid marker at the peak
            # amplitude (the common case) would otherwise sit right on top of the
            # very feature it's meant to point out.
            ax.scatter(
                [point.time],
                [point.amplitude],
                s=120,
                facecolors=[to_rgba(point.color, alpha=0.3)],
                edgecolors=point.color,
                linewidths=1.5,
                zorder=3,
            )
            if point.text:
                # Horizontal offset only (va="center"): a vertical offset would land
                # above or below depending on which side of the trace the point is
                # on, and at a local peak the label can end up covering the very
                # dot it's meant to label. No background box, and text color
                # matches the marker -- this is meant to read as "this dot's name",
                # not a separate boxed callout like LabelBox's spectrogram text.
                ax.annotate(
                    point.text,
                    (point.time, point.amplitude),
                    xytext=(8, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color=point.color,
                    clip_on=True,
                )
        ax.set_ylabel("Amplitude")

    def _resampled_confidence(self, spec: VisualizationSpec, target_length: int):
        co = spec.classifier_output
        if co is None:
            return None, None
        target_class = co.class_index(spec.target_class if spec.target_class is not None else 1)
        similarity = co.resample_class_to(
            target_class, target_length, duration=spec.audio.duration, start_time=spec.display_offset
        )
        dissimilarity = 1.0 - similarity
        return similarity, dissimilarity

    def _draw_spectrogram(self, ax, spec: VisualizationSpec, *, wavelet: bool) -> None:
        y = spec.audio.as_mono() if spec.audio.n_channels > 1 else spec.audio.channel(0)
        sr = spec.audio.sr
        if wavelet:
            result = spec.wavelet.compute(y, sr)
            power, freqs = result.power, result.freqs
            ax.set_ylabel("Wavelet Hz")
        else:
            result = spec.stft.compute(y, sr)
            power, freqs = result.power, result.freqs
            ax.set_ylabel("STFT Hz")

        power = normalize_power(
            power, per_channel_normalize=spec.per_channel_normalize, clip_outliers=spec.clip_outliers
        )
        db = power_to_db(power)
        normed = (db - db.min()) / max(db.max() - db.min(), 1e-9)

        similarity, dissimilarity = self._resampled_confidence(spec, normed.shape[1])
        if similarity is not None:
            rgb = confidence_to_rgb(normed, similarity, dissimilarity, style=spec.colorize_style)
        else:
            rgb = np.stack([normed, normed, normed], axis=-1)

        origin = "lower" if not wavelet else "upper"
        extent = (
            (spec.start_time, spec.end_time, freqs[0], freqs[-1])
            if not wavelet
            else (
                spec.start_time,
                spec.end_time,
                len(freqs),
                0,
            )
        )
        ax.imshow(rgb, aspect="auto", origin=origin, extent=extent)
        if wavelet:
            self._label_log_freq_axis(ax, freqs)

        if spec.labels:
            self._draw_label_boxes(ax, spec, freqs, wavelet=wavelet)

    def _label_log_freq_axis(self, ax, freqs: np.ndarray, n_ticks: int = 8) -> None:
        """Wavelet rows are evenly spaced by index, not by Hz (the whole point of a
        log-piecewise CWT scale is that frequency resolution is finer at low
        frequencies) -- so unlike the STFT axis (a plain linear Hz extent), we pick a
        handful of index positions and label them with their *actual* (non-uniformly
        spaced) Hz values. ``freqs`` is expected high-to-low (index 0 == top row ==
        highest frequency), matching the imshow extent above.
        """
        idxs = np.linspace(0, len(freqs) - 1, n_ticks).astype(int)
        labels = [f"{freqs[i]:.0f}" for i in idxs]
        ax.set_yticks(idxs)
        ax.set_yticklabels(labels)

    def _draw_label_boxes(self, ax, spec: VisualizationSpec, freqs, *, wavelet: bool) -> None:
        from matplotlib import patches

        vert_offset = (freqs[-1] - freqs[0]) / 25 if not wavelet else (len(freqs) / 25)
        for box in spec.labels:
            if not box.overlaps(spec.start_time, spec.end_time):
                continue
            if wavelet:
                # The wavelet axis is index-based (see _label_log_freq_axis), so Hz
                # values from the label need mapping to a (fractional) row index
                # before they mean anything as a y-coordinate here.
                low_y = self._freq_to_index(box.low_freq, freqs)
                high_y = self._freq_to_index(box.high_freq, freqs)
                xy = (box.start_time, min(low_y, high_y))
                height_ = abs(high_y - low_y)
            else:
                xy = (box.start_time, box.low_freq)
                height_ = box.high_freq - box.low_freq
            width_ = box.duration
            for linewidth, edgecolor in ((3, (0, 0, 0)), (1, (0, 1, 1))):
                ax.add_patch(
                    patches.Rectangle(xy, width_, height_, linewidth=linewidth, edgecolor=edgecolor, facecolor="none")
                )
            if box.text:
                ax.text(
                    xy[0] + width_,
                    xy[1] + height_ + vert_offset,
                    box.text,
                    ha="right",
                    va="bottom",
                    fontsize=12,
                    color="white",
                    clip_on=True,
                    bbox={"boxstyle": "round,pad=0.3", "edgecolor": "cyan", "facecolor": "black", "alpha": 0.7},
                )

    @staticmethod
    def _freq_to_index(freq_hz: float, freqs: np.ndarray) -> float:
        """Map a Hz value to its (fractional) row index in a high-to-low ``freqs`` array."""
        ascending_freqs = freqs[::-1]
        ascending_idxs = np.arange(len(freqs) - 1, -1, -1)
        return float(np.interp(freq_hz, ascending_freqs, ascending_idxs))

    def _draw_similarity_lines(self, ax, spec: VisualizationSpec) -> None:
        co = spec.classifier_output
        target_class = co.class_index(spec.target_class if spec.target_class is not None else 1)
        similarity = co.probabilities[:, target_class]
        dissimilarity = 1 - similarity
        # window_centers() is already absolute (includes co.time_offset, which is the
        # *actual* rounded-to-window-boundary start of this slice) -- do not add
        # spec.display_offset here too, that would double-count it and reintroduce
        # the quantization-drift bug this was fixed for.
        t = co.window_centers()
        ax.plot(t, similarity, color="tab:green")
        ax.plot(t, dissimilarity, color="tab:red")
        ax.set_ylabel(co.class_labels[target_class])

    def _draw_class_probability_stack(self, ax, spec: VisualizationSpec) -> None:
        co = spec.classifier_output
        t = co.window_centers()  # already absolute -- see _draw_similarity_lines
        ax.stackplot(t, co.probabilities.T, labels=co.class_labels)
        # Right-of-axes (not below): a legend anchored below its own axes only has
        # room when that axes happens to be the bottommost thing on the figure --
        # with any other track after it (e.g. SIMILARITY_LINES), that track's own
        # axes gets drawn right over the legend. Right-side placement uses space
        # reserved once, in `render()`, regardless of track order.
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), prop={"size": 8})
        ax.set_ylabel("Cls Prob")
