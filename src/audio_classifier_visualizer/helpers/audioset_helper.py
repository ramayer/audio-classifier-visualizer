import os
import subprocess

import duckdb
import einx
import librosa
import numpy as np
import torch


class AudioSetHelper:
    def __init__(self, output_dir, audio_format="opus", audio_quality="5", sample_rate=16000):
        """
        Initializes the AudioSetHelper with common parameters.

        Args:
            output_dir (str): Directory to save the audio files.
            audio_format (str): Audio format (e.g., 'opus', 'mp3', 'wav').
            audio_quality (str): Audio quality (e.g., '5' for Opus, '64K' for MP3).
            sample_rate (int): Sample rate in Hz (e.g., 16000).
        Note:
            Opus seems the best tradeoff.  mpa and aac are slow. mp3 is worse.  wav is huge
        """
        self.output_dir = output_dir
        self.audio_format = audio_format
        self.audio_quality = audio_quality
        self.sample_rate = sample_rate

        self._ddb = None
        os.makedirs(self.output_dir, exist_ok=True)

    def _parse_clip_id(self, clip_id):
        yt_id, start_time_ms = clip_id.split("_")
        start_time_seconds = int(start_time_ms) / 1000  # Convert milliseconds to seconds
        cache_file = os.path.join(self.output_dir, f"{clip_id}.{self.audio_format}")
        return yt_id, start_time_seconds, cache_file

    def download_audio(self, clip_id):
        """
        Downloads the entire 10-second audio clip specified by clip_id.

        Args:
            clip_id (str): The clip ID in the format 'ytid_starttimems'.

        Returns:
            tuple: (return_code, stderr) where return_code is the exit status of yt-dlp,
                   and stderr is the error output (if any).

        Note:
            * yt-dlp often ignores -ar
            * between youtube's API, yt-dlp, ffmpeg and librosa, the
            end of the audio clips lose samples, so fetch a little extra
        """
        yt_id, st, cache_file = self._parse_clip_id(clip_id)
        et = st + 10 + 1.1

        _consider_download_sections = """
            yt-dlp -x --audio-format {self.format}
            --audio-quality {self.quality}
            --output "{os.path.join(self.root_path, first_display_label, ytid)}_{start_seconds}-{end_seconds}.%(ext)s"
            --download-sections *{start_seconds}-{end_seconds}
            --force-keyframes-at-cuts --cookies {self.cookies} https://www.youtube.com/watch?v={ytid}
        """
        command = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format",
            self.audio_format,  # Set audio format
            "--audio-quality",
            self.audio_quality,  # Set audio quality
            "--postprocessor-args",
            f"-ss {st} -to {et}  -ar {self.sample_rate}",  # Trim and resample
            "-o",
            cache_file,  # Output file path
            f"https://www.youtube.com/watch?v={yt_id}",  # YouTube URL
        ]
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode:
            with open(cache_file + ".errors", "w") as f:
                f.write(process.stderr)
        return process.returncode, process.stderr

    def get_audio(self, clip_id):
        _yt_id, _start_time_seconds, cache_file = self._parse_clip_id(clip_id)
        if os.path.exists(cache_file + ".errors"):
            return None, None
        if not os.path.exists(cache_file):
            rc, se = self.download_audio(clip_id)
            if rc:
                return None, None
        y, sr = librosa.load(cache_file, mono=True, sr=16000)
        return y, sr

    def get_youtube_url(self, clip_id):
        yt_id, start_time_seconds, cache_file = self._parse_clip_id(clip_id)
        return f"https://www.youtube.com/watch?v={yt_id}&t={start_time_seconds}s"

    def get_labels_for_a_clip(self, clip_id):
        """
        Returns all the audioset labels for a clip as a pandas dataframe like this
        ┌────────────────────┬────────┬────────┬────────────┬───────────────────────┐
        │      clip_id       │   st   │   et   │    mid     │      displayname      │
        │      varchar       │ double │ double │  varchar   │        varchar        │
        ├────────────────────┼────────┼────────┼────────────┼───────────────────────┤
        │ kuoQtqGpsD0_120000 │    0.0 │  0.156 │ /m/0bt9lr  │ Dog                   │
        │ kuoQtqGpsD0_120000 │  0.488 │  2.846 │ /m/01yrx   │ Cat                   │
        │ kuoQtqGpsD0_120000 │  0.737 │  1.062 │ /t/dd00141 │ Pant (dog)            │
        └───────────────────────────────────────────────────────────────────────────┘
        """
        labels_for_a_clip = self.ddb.execute(
            """
            select clip_id,st,et,mid,displayname as lbl
            from source_audioset_train_strong
            join mid_to_display_name using (mid)
            where clip_id = ?
            and displayname != 'Background noise'
            order by st
        """,
            (clip_id,),
        )
        return labels_for_a_clip.df()

    def create_label_arrays(self, clip_id):
        """
        Converts labeled time intervals into binary numpy arrays for each label.

        Args:
            df (pd.DataFrame): The DataFrame containing the labeled time intervals.
            clip_duration_ms (int): The duration of the clip in milliseconds (default: 10,000 ms).

        Returns:
            dict: A dictionary where keys are labels and values are binary numpy arrays.
        """
        clip_duration_ms = 10 * 1000
        df = self.get_labels_for_a_clip(clip_id)
        # Initialize a dictionary to store the arrays
        label_arrays = {}

        # Iterate through each row in the DataFrame
        for _, row in df.iterrows():
            lbl = row["lbl"]
            start_time = row["st"]
            end_time = row["et"]

            # Convert start and end times to milliseconds
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)

            # Ensure the label array exists in the dictionary
            if lbl not in label_arrays:
                label_arrays[lbl] = np.zeros(clip_duration_ms, dtype=int)

            # Set the corresponding elements to 1
            label_arrays[lbl][start_ms:end_ms] = 1

        # return label_arrays

        label_list = list(label_arrays.keys())
        v = np.stack(list(label_arrays.values())) * 1.0
        v = torch.tensor(v)
        v = einx.mean("a (b c) -> a b", v, c=100).T
        return label_list, v

    @property
    def ddb(self):
        if self._ddb:
            return self._ddb
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

        # ddb.read_csv('http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv',skiprows=2) # v1

        ddb.sql("create table if not exists mid_to_display_name as select * from source_mid_to_display_name")
        ddb.sql("create table if not exists audioset_train_strong as select * from source_audioset_train_strong")
        self._ddb = ddb
        return ddb

    def _refresh_cookies(self):
        if not os.path.exists(f"{self.output_dir}/cookies.txt"):
            subprocess.run(
                f"""
                yt-dlp --cookies-from-browser chromium:~/snap/chromium/common/chromium/Default --cookies {self.output_dir}/cookies.txt 0
                """,
                shell=True,  # noqa: S602
                check=False,
            )
