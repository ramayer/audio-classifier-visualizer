"""Fetch AudioSet Strong clips + labels for test/example data.

Ported near-verbatim from the original ``audioset_helper.py`` (moved under
``helpers/`` as a clearly-optional, clearly-not-core-to-the-library tool).
The one substantive change: ``create_label_arrays`` now returns a plain
numpy array instead of a ``torch.Tensor`` -- torch was never doing anything
here except holding an array so ``einx.mean`` had something to reduce, and
einx works directly on numpy.

Requires the optional ``audioset`` extra: ``pip install
audio-classifier-visualizer[audioset]`` (duckdb, yt-dlp, pandas, librosa,
einx). Also requires the ``yt-dlp`` *command-line tool* to be on PATH for
``download_audio``/``get_audio`` (downloading the underlying YouTube clips),
separate from the pip package of the same name.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np


class AudioSetHelper:
    def __init__(
        self,
        output_dir: str,
        audio_format: str = "opus",
        audio_quality: str = "5",
        sample_rate: int = 16000,
    ) -> None:
        """
        Args:
            output_dir: Directory to cache downloaded audio + the duckdb catalog in.
            audio_format: e.g. 'opus', 'mp3', 'wav'. Opus is the best size/quality
                tradeoff; mp3/aac/wav are worse or much bigger.
            audio_quality: e.g. '5' for Opus, '64K' for mp3.
            sample_rate: Hz.
        """
        self.output_dir = output_dir
        self.audio_format = audio_format
        self.audio_quality = audio_quality
        self.sample_rate = sample_rate
        self._ddb = None
        os.makedirs(self.output_dir, exist_ok=True)

    def _parse_clip_id(self, clip_id: str):
        yt_id, start_time_ms = clip_id.split("_")
        start_time_seconds = int(start_time_ms) / 1000
        cache_file = os.path.join(self.output_dir, f"{clip_id}.{self.audio_format}")
        return yt_id, start_time_seconds, cache_file

    def download_audio(self, clip_id: str):
        """Download the 10-second AudioSet clip via yt-dlp.

        Fetches a little extra past the 10s boundary since youtube/yt-dlp/ffmpeg/
        librosa each tend to lose a few samples off the end otherwise.
        """
        yt_id, st, cache_file = self._parse_clip_id(clip_id)
        et = st + 10 + 1.1
        command = [
            "yt-dlp",
            "-x",
            "--audio-format",
            self.audio_format,
            "--audio-quality",
            self.audio_quality,
            "--postprocessor-args",
            f"-ss {st} -to {et} -ar {self.sample_rate}",
            "-o",
            cache_file,
            f"https://www.youtube.com/watch?v={yt_id}",
        ]
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode:
            with open(cache_file + ".errors", "w") as f:
                f.write(process.stderr)
        return process.returncode, process.stderr

    def get_audio(self, clip_id: str):
        import librosa

        _yt_id, _start, cache_file = self._parse_clip_id(clip_id)
        if os.path.exists(cache_file + ".errors"):
            return None, None
        if not os.path.exists(cache_file):
            rc, _stderr = self.download_audio(clip_id)
            if rc:
                return None, None
        y, sr = librosa.load(cache_file, mono=True, sr=self.sample_rate)
        return y, sr

    def get_youtube_url(self, clip_id: str) -> str:
        yt_id, start_time_seconds, _cache_file = self._parse_clip_id(clip_id)
        return f"https://www.youtube.com/watch?v={yt_id}&t={start_time_seconds}s"

    def get_labels_for_a_clip(self, clip_id: str):
        """Returns a pandas DataFrame of (clip_id, st, et, mid, displayname) rows."""
        result = self.ddb.execute(
            """
            select clip_id, st, et, mid, displayname as lbl
            from source_audioset_train_strong
            join mid_to_display_name using (mid)
            where clip_id = ?
            and displayname != 'Background noise'
            order by st
            """,
            (clip_id,),
        )
        return result.df()

    def create_label_arrays(self, clip_id: str, bin_ms: int = 100) -> tuple[list[str], np.ndarray]:
        """Turn labeled time intervals into a dense (n_bins, n_labels) binary array.

        AudioSet Strong labels are given as (start_ms, end_ms) intervals; this
        rasterizes them to ``bin_ms``-wide bins (default matches AudioSet's own
        ~960ms label spacing when combined with the 10x oversampling below).
        """
        clip_duration_ms = 10 * 1000
        df = self.get_labels_for_a_clip(clip_id)
        label_arrays: dict[str, np.ndarray] = {}
        for _, row in df.iterrows():
            lbl = row["lbl"]
            start_ms, end_ms = int(row["st"] * 1000), int(row["et"] * 1000)
            if lbl not in label_arrays:
                label_arrays[lbl] = np.zeros(clip_duration_ms, dtype=float)
            label_arrays[lbl][start_ms:end_ms] = 1

        label_list = list(label_arrays.keys())
        if not label_list:
            return label_list, np.zeros((clip_duration_ms // bin_ms, 0))
        v = np.stack(list(label_arrays.values()))
        import einx

        v = einx.mean("a (b c) -> a b", v, c=bin_ms).T
        return label_list, v

    @property
    def ddb(self):
        if self._ddb:
            return self._ddb
        import duckdb

        ddb = duckdb.connect(self.output_dir + "/audioset.ddb")
        base_uri = "http://storage.googleapis.com/us_audioset/youtube_corpus/strong/"
        ddb.read_csv(base_uri + "mid_to_display_name.tsv", header=False, names=["mid", "displayname"]).create_view(
            "source_mid_to_display_name"
        )
        ddb.read_csv(
            base_uri + "audioset_train_strong.tsv", header=True, names=["clip_id", "st", "et", "mid"]
        ).create_view("source_audioset_train_strong")
        ddb.read_csv(
            "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
        ).create_view("class_labels_indices")
        ddb.sql("create table if not exists mid_to_display_name as select * from source_mid_to_display_name")
        ddb.sql("create table if not exists audioset_train_strong as select * from source_audioset_train_strong")
        self._ddb = ddb
        return ddb

    def _refresh_cookies(self) -> None:
        if not os.path.exists(f"{self.output_dir}/cookies.txt"):
            subprocess.run(
                f"yt-dlp --cookies-from-browser "
                f"chromium:~/snap/chromium/common/chromium/Default "
                f"--cookies {self.output_dir}/cookies.txt 0",
                shell=True,
                check=False,
            )
