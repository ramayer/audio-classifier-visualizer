"""Static-image renderer. First (and simplest) implementation of the Renderer protocol.

Good fit for: Jupyter notebooks, batch report generation, thumbnails of a
whole day. Not interactive -- for drag-zoom/pan/hover, a future renderer
(e.g. Bokeh/HoloViews+Datashader) implements the same protocol and slots in
without touching this file or anything in core/features.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings

import numpy as np

from audio_classifier_visualizer.features.colorize import confidence_to_rgb, normalize_power, power_to_db
from audio_classifier_visualizer.render.spec import Track, VisualizationSpec

logger = logging.getLogger(__name__)

_TICK_INTERVALS = ((1, 0.25), (3, 0.5), (10, 1), (60, 5), (300, 30), (600, 60), (3600, 300), (7200, 600))

# Tracks that only make sense with a classifier_output attached -- both the new
# CLASS_PROBABILITIES/CLASS_HEATMAP and the two deprecated aliases they replace.
_CLASSIFIER_TRACKS = frozenset(
    {Track.CLASS_PROBABILITIES, Track.CLASS_HEATMAP, Track.SIMILARITY_LINES, Track.CLASS_PROBABILITY_STACK}
)


def _tick_interval(duration: float) -> float:
    return next((v for limit, v in _TICK_INTERVALS if duration <= limit), 3600)


def _bucket_stats(y: np.ndarray, n_buckets: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ``y`` into up to ``n_buckets`` equal-size chunks and return
    (min, max, mean, std) per chunk, fully vectorized (no per-bucket Python loop)
    via pad-then-reshape.

    The chunk size is chosen first (ceil(len(y) / n_buckets)), and the actual
    bucket count derived from *that* -- not the other way around. That guarantees
    at most one chunk's worth of padding total, confined to (at most) the tail of
    the very last bucket. The naive approach of padding up to the next multiple of
    n_buckets can instead need up to n_buckets-1 samples of padding, which -- for a
    short clip where each bucket only holds a handful of real samples -- can mean
    dozens or hundreds of *entirely fake* trailing buckets (all edge-repeated
    padding, no real data at all), rendering as the plot going dead/flat well
    before the requested end_time. Confirmed as the root cause of exactly that
    symptom: a 0.5s/8000Hz clip bucketed toward 1200 target pixels padded 800 of
    4000 samples (200 of 1200 buckets, 16.7% of the plot width) under the naive
    approach; this version pads 0.

    Padding (on the rare occasion any is still needed) repeats the last real
    sample rather than filling with e.g. the global mean -- that only affects
    the tail of the last bucket, and repeating an already-present value doesn't
    distort its min/max/mean/std the way injecting an unrelated fill value could.

    The returned arrays may have slightly fewer than ``n_buckets`` entries (never
    more) -- callers should size their x-axis off the actual returned length, not
    the requested ``n_buckets``.
    """
    n_buckets = max(1, n_buckets)
    chunk_size = max(1, -(-len(y) // n_buckets))  # ceil(len(y) / n_buckets) via negated floor division
    n_buckets_actual = -(-len(y) // chunk_size)  # ceil(len(y) / chunk_size); guaranteed <= n_buckets
    pad_len = n_buckets_actual * chunk_size - len(y)
    padded = np.pad(y, (0, pad_len), mode="edge") if pad_len else y
    reshaped = padded.reshape(n_buckets_actual, chunk_size)
    return reshaped.min(axis=1), reshaped.max(axis=1), reshaped.mean(axis=1), reshaped.std(axis=1)


class MatplotlibRenderer:
    def __init__(self, waveform_envelope_threshold: float = 2.0) -> None:
        """
        Args:
            waveform_envelope_threshold: when samples-per-pixel in the WAVEFORM
                track exceeds this, switch from a direct line plot to a bucketed
                min/max/mean/std envelope (see _draw_waveform). Lower values switch
                to the envelope sooner (more conservative about plot cost/clarity
                at the expense of raw detail); higher values keep the direct line
                plot longer.
        """
        self.waveform_envelope_threshold = waveform_envelope_threshold
        # Stable class -> color mapping across separate .show() calls (and across
        # CLASS_PROBABILITIES vs. any other line-based track): a class gets the
        # same color the first time it's ever drawn by this renderer instance and
        # keeps it, rather than colors being reassigned per-figure based on
        # whatever happens to be selected that call.
        self._class_colors: dict[str, tuple] = {}

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
            Track.CLASS_PROBABILITIES: 2,
            Track.CLASS_HEATMAP: 2.5,
            Track.SIMILARITY_LINES: 1,
            Track.CLASS_PROBABILITY_STACK: 2,
        }
        tracks = [t for t in spec.tracks if t not in _CLASSIFIER_TRACKS or spec.classifier_output is not None]
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
        last_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: self._format_tick(spec, x)))
        # set_xticks on a sharex=True group re-expands the shared xlim to include any
        # tick location outside the current view (confirmed against matplotlib
        # directly) -- undoing the set_xlim() calls above. Re-assert it last, after
        # ticks are placed, so the displayed range matches what was actually
        # requested rather than snapping out to the next tick interval.
        for ax in axes:
            ax.set_xlim(spec.start_time, spec.end_time)

        fig.suptitle(self._title_with_date(spec), fontsize=16, ha="left", x=0)
        right_margin = 0.85 if any(t in _CLASSIFIER_TRACKS for t in tracks) else 0.98
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
        elif track == Track.CLASS_PROBABILITIES:
            self._draw_class_probabilities(ax, spec)
        elif track == Track.CLASS_HEATMAP:
            self._draw_class_heatmap(ax, spec)
        elif track == Track.SIMILARITY_LINES:
            warnings.warn(
                "Track.SIMILARITY_LINES is deprecated and will be removed in a future "
                "release; use Track.CLASS_PROBABILITIES(classes=[target_class]) instead "
                "(drops the redundant inverse line -- define two classes that sum to 1 "
                "if you want that look back).",
                DeprecationWarning,
                stacklevel=2,
            )
            effective = dataclasses.replace(spec, classes=[self._target_class_name(spec)], top_k=None)
            self._draw_class_probabilities(ax, effective)
        elif track == Track.CLASS_PROBABILITY_STACK:
            warnings.warn(
                "Track.CLASS_PROBABILITY_STACK is deprecated and will be removed in a "
                "future release; use Track.CLASS_PROBABILITIES instead (unstacked -- "
                "stacking made any one class's own trend hard to read independent of "
                "whatever was stacked below it).",
                DeprecationWarning,
                stacklevel=2,
            )
            effective = dataclasses.replace(spec, classes=list(spec.classifier_output.class_labels), top_k=None)
            self._draw_class_probabilities(ax, effective)
        else:
            msg = f"Unhandled track type: {track}"
            raise ValueError(msg)

    def _draw_waveform(self, ax, spec: VisualizationSpec) -> None:
        from matplotlib.colors import to_rgba

        y = spec.audio.as_mono() if spec.audio.n_channels > 1 else spec.audio.channel(0)
        n_pixels = max(1, round(ax.figure.get_size_inches()[0] * ax.figure.dpi))
        samples_per_pixel = len(y) / n_pixels

        if samples_per_pixel <= self.waveform_envelope_threshold:
            # Few enough samples per pixel that a direct line plot is both cheap and
            # actually shows the waveform's shape (a guitar string's not-quite-sine
            # wave, etc.) -- exactly what an envelope would obscure.
            times = np.linspace(spec.start_time, spec.end_time, len(y))
            ax.plot(times, y, linewidth=0.5, color="black")
        else:
            # Too many samples per pixel for a direct line plot to be meaningful (it
            # becomes a solid black blob) or fast. Bucket to ~one point per pixel and
            # draw three nested layers, all from the same per-bucket statistics:
            #   - min/max envelope (light grey): the full excursion in each bucket.
            #   - mean +/- std band (medium grey): despite the name "RMS envelope",
            #     this is std of the *detrended* signal, not sqrt(mean(x**2)) --
            #     plain RMS is dominated by any DC offset (e.g. a pressure sensor
            #     sitting at ~10 with a small AC component would render as a flat
            #     line at ~10 either way), telling you nothing about the AC
            #     variation. std around the local mean stays meaningful regardless
            #     of DC offset.
            #   - mean (black line): ~0 for zero-centered audio (harmless, sits
            #     inside the std band) but becomes the visible DC-trend line for a
            #     signal like a pressure sensor -- exactly the case this needs to
            #     support alongside ordinary zero-centered audio.
            bucket_min, bucket_max, bucket_mean, bucket_std = _bucket_stats(y, n_pixels)
            bucket_times = np.linspace(spec.start_time, spec.end_time, len(bucket_min))
            ax.fill_between(bucket_times, bucket_min, bucket_max, color="0.75", linewidth=0, zorder=1)
            ax.fill_between(
                bucket_times, bucket_mean - bucket_std, bucket_mean + bucket_std, color="0.45", linewidth=0, zorder=2
            )
            ax.plot(bucket_times, bucket_mean, color="black", linewidth=0.8, zorder=2.5)

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
        # Compute over context_audio (real neighboring samples, or reflect/edge
        # padding at a true file boundary -- see AudioVisualization._load_context_audio)
        # when available, so analysis windows near the display edge aren't relying on
        # the feature extractors' own internal zero-padding. Falls back to computing
        # directly on spec.audio (old behavior, edge artifacts and all) for a spec
        # built by hand without going through AudioVisualization.
        use_context = spec.context_audio is not None
        source = spec.context_audio if use_context else spec.audio
        y = source.as_mono() if source.n_channels > 1 else source.channel(0)
        sr = source.sr
        if wavelet:
            result = spec.wavelet.compute(y, sr)
            power, freqs = result.power, result.freqs
            ax.set_ylabel("Wavelet Hz")
            columns_per_second = sr / spec.wavelet.decimation_stride
        else:
            result = spec.stft.compute(y, sr)
            power, freqs = result.power, result.freqs
            ax.set_ylabel("STFT Hz")
            columns_per_second = sr / spec.stft.hop_length

        if use_context:
            # Crop the context-widened spectrogram back down to just the requested
            # display range -- the extra columns on each side existed only to give
            # the edge analysis windows real data to work with, not to be shown.
            start_col = round(spec.context_seconds * columns_per_second)
            end_col = min(power.shape[1], start_col + round(spec.audio.duration * columns_per_second))
            start_col = min(start_col, end_col)
            power = power[:, start_col:end_col]

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

    def _format_tick(self, spec: VisualizationSpec, relative_seconds: float) -> str:
        """Clock-time ticks (HH:MM:SS, in whatever display_timezone is set) when the
        audio has a known real-world anchor; otherwise the existing relative
        H:MM:SS.sss-from-start-of-clip formatting."""
        axis = spec.audio.time_axis
        if axis.has_absolute_time:
            return axis.format_absolute(relative_seconds)
        return axis.format_relative(relative_seconds)

    def _title_with_date(self, spec: VisualizationSpec) -> str:
        """Append the date to the title when displaying clock-time ticks -- once,
        in the title, rather than repeating it on every tick label."""
        axis = spec.audio.time_axis
        if not axis.has_absolute_time:
            return spec.title
        date_str = axis.display_date(spec.start_time)
        if date_str in spec.title:
            return spec.title  # caller already put a/the date in the title themselves
        return f"{spec.title} — {date_str}" if spec.title else date_str

    def _target_class_name(self, spec: VisualizationSpec) -> str:
        co = spec.classifier_output
        idx = co.class_index(spec.target_class if spec.target_class is not None else 1)
        return co.class_labels[idx]

    def _select_classes(self, spec: VisualizationSpec) -> list[str]:
        """Resolve which classes CLASS_PROBABILITIES/CLASS_HEATMAP should show.

        Explicit ``classes`` wins if given. Otherwise ``top_k`` selects by peak
        probability within the *currently displayed* window (spec.classifier_output
        is already sliced -- with a little margin -- to the display range by
        AudioVisualization, so this naturally recomputes on every zoom rather than
        being fixed at load time). With neither given, all classes are shown,
        uncapped -- what the deprecated CLASS_PROBABILITY_STACK alias relies on to
        preserve its old behavior; direct use of the new tracks should usually set
        top_k for anything with more than a handful of classes.
        """
        co = spec.classifier_output
        if spec.classes is not None:
            return list(spec.classes)
        if spec.top_k is not None:
            peak = co.probabilities.max(axis=0)
            order = np.argsort(peak)[::-1][: spec.top_k]
            return [co.class_labels[i] for i in order]
        return list(co.class_labels)

    def _class_color(self, name: str):
        if name not in self._class_colors:
            import matplotlib.pyplot as plt

            palette = plt.get_cmap("tab20").colors
            self._class_colors[name] = palette[len(self._class_colors) % len(palette)]
        return self._class_colors[name]

    def _draw_class_probabilities(self, ax, spec: VisualizationSpec) -> None:
        """Overlaid (not stacked) per-class probability lines. Any number of
        classes can be near 1.0 at once -- e.g. a hierarchical "DOG" and "MAMMAL"
        both reading ~100% -- since nothing here assumes they sum to 1."""
        co = spec.classifier_output
        selected = self._select_classes(spec)
        t = co.window_centers()  # already absolute -- see the wavelet/STFT crop comment
        for name in selected:
            idx = co.class_index(name)
            ax.plot(t, co.probabilities[:, idx], color=self._class_color(name), label=name, linewidth=1.2)
        ax.set_ylim(0, 1)
        if len(selected) == 1:
            # Matches the old SIMILARITY_LINES look for the common single-class
            # case: the y-label already says what this is, a one-entry legend
            # would just take up space for no benefit.
            ax.set_ylabel(selected[0])
        else:
            ax.set_ylabel("Cls Prob")
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), prop={"size": 8})

    def _draw_class_heatmap(self, ax, spec: VisualizationSpec) -> None:
        """Class-on-y-axis, time-on-x, color = probability. Scales to far more
        classes than an overlaid line chart can stay readable with; rows are
        sorted by peak probability within the displayed window (most locally
        relevant classes near the top), not by class_labels' original order."""
        co = spec.classifier_output
        selected = self._select_classes(spec)
        idxs = [co.class_index(name) for name in selected]
        peaks = co.probabilities[:, idxs].max(axis=0)
        order = np.argsort(peaks)[::-1]
        names_sorted = [selected[i] for i in order]
        data = co.probabilities[:, [idxs[i] for i in order]].T  # (n_classes, n_windows)

        centers = co.window_centers()
        im = ax.imshow(
            data,
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
            origin="upper",
            extent=(centers[0], centers[-1], len(names_sorted), 0),
        )
        ax.set_yticks(np.arange(len(names_sorted)) + 0.5)
        ax.set_yticklabels(names_sorted, fontsize=8)
        cbar = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
        cbar.ax.tick_params(labelsize=8)
