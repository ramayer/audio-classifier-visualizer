from __future__ import annotations

import csv
import glob
import logging
import os
import re
from dataclasses import dataclass
from string import Template
from typing import Any

import duckdb
import torch


# shorter names for important columns from the raven file
@dataclass
class RavenLabel:
    bt: float
    et: float
    lf: float
    hf: float
    duration: float | None = None
    audio_file: str | None = None
    t1: str | None = None
    t2: str | None = None
    t3: str | None = None
    notes: str | None = None
    score: float | None = None
    ravenfile: str | None = None


class RavenFileHelper:
    def __init__(self, root_path: str | None = None) -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.ddb: duckdb.DuckDBPyConnection = duckdb.connect()
        if root_path:
            self.root_path: str = root_path
            self.raven_files: list[str] = self.find_candidate_raven_files(root_path)
            self.all_raven_data: duckdb.DuckDBPyRelation = self.all_raven_files_as_one_table(self.raven_files)
            self.identify_useful_files()

    def find_continuous_segments(self, boolean_tensor: torch.Tensor) -> list[tuple[int, int]]:
        if boolean_tensor.shape[0] == 0:
            return []
        sign_changes = torch.cat(
            [
                torch.tensor([True]),
                boolean_tensor[1:] != boolean_tensor[:-1],
                torch.tensor([True]),
            ]
        )
        change_indices = torch.where(sign_changes)[0]
        segments = []
        for start, end in zip(change_indices[:-1], change_indices[1:]):
            if boolean_tensor[start]:
                segments.append((start.item(), end.item() - 1))
        return segments

    def find_long_enough_segments(self, segments: list[tuple[int, int]], n: int = 3) -> list[tuple[int, int]]:
        return [(a, b) for a, b in segments if b - a >= n]

    def save_segments_to_raven_file(
        self, raven_labels: list[RavenLabel], filename: str, _audio_file_name: str, _audio_file_processor: Any
    ) -> None:
        self.write_raven_file(raven_labels, filename)

    def find_candidate_raven_files(self, root_path: str) -> list[str]:
        pattern = root_path + "/**/*.txt"
        txtfiles = glob.glob(pattern, recursive=True)
        raven_files = []
        for txtfile in txtfiles:
            tbl = self.ddb.read_csv(txtfile, delimiter="\t", header=True)
            if "Selection" in tbl.columns:
                raven_files.append(txtfile)
        return raven_files

    def load_one_raven_file(self, raven_file: str) -> list[RavenLabel]:
        """usage
        rfh.load_one_raven_file('/tmp/rf.raven')
        """

        self.ddb.sql("""
                    create or replace temp view empty_table_all_columns as
                     select * from all_raven_files limit 0
                     """)
        # Create view for reading CSV file
        self.ddb.execute(
            "CREATE OR REPLACE VIEW raven_file_data AS "
            "SELECT * FROM read_csv(?, auto_type_candidates = ['BIGINT', 'DOUBLE','VARCHAR'])",
            [raven_file],
        )

        # Union with empty table to get all columns
        self.ddb.sql(
            "CREATE OR REPLACE VIEW one_raven_file AS "
            "SELECT * FROM raven_file_data "
            "UNION ALL BY NAME "
            "(SELECT * FROM empty_table_all_columns)"
        )
        useful_cols = self.ddb.sql(
            """
            SELECT
                 "File Offset (s)"::double as start_time,
                 "File Offset (s)"::double - "Begin Time (s)"::double + "End Time (s)"::double as end_time,
                 "Low Freq (Hz)" as low_freq,
                 "High Freq (Hz)" as high_freq,
                 "End Time (s)"::double - "Begin Time (s)"::double as duration,
                 REPLACE("Begin File",'dzan','dz') as audio_file,
                 "Tag 1" as tag1,
                 "Tag 2" as tag2,
                 "Tags" as tags,
                 "Notes" as notes,
                 "Score" as score,
                 raven_filename as raven_file
               FROM one_raven_file
               WHERE end_time is not null
            """
        )
        return [RavenLabel(*row) for row in useful_cols.fetchall()]

    def all_raven_files_as_one_table(self, raven_files: list[str]) -> duckdb.DuckDBPyRelation:
        for idx, f in enumerate(raven_files):
            sql_template = Template("""
                CREATE OR REPLACE VIEW raven_file_$idx as
                SELECT *,'$filename' as raven_filename
                FROM read_csv('$filename', auto_type_candidates = ['BIGINT', 'DOUBLE','VARCHAR']);
                """)
            self.ddb.sql(sql_template.substitute(idx=idx, filename=f))
        tables_to_union = [
            Template("select * from raven_file_$idx").substitute(idx=idx) for idx in range(len(raven_files))
        ]
        union_sql = Template("""
            CREATE OR REPLACE TEMPORARY TABLE all_raven_files as
            $union_tables
            """).substitute(union_tables=" UNION ALL BY NAME ".join(tables_to_union))
        self.ddb.sql(union_sql)

        self.ddb.sql(
            """
            CREATE OR REPLACE TEMP TABLE all_raven_labels AS
              SELECT
                 "File Offset (s)"::double as start_time,
                 "File Offset (s)"::double - "Begin Time (s)"::double + "End Time (s)"::double as end_time,
                 "Low Freq (Hz)" as low_freq,
                 "High Freq (Hz)" as high_freq,
                 "End Time (s)"::double - "Begin Time (s)"::double as duration,
                 REPLACE("Begin File",'dzan','dz') as audio_file,
                 "Tag 1" as tag1,
                 "Tag 2" as tag2,
                 "Tags" as tags,
                 "Notes" as notes,
                 "Score" as score,
                 raven_filename as raven_file
               FROM all_raven_files
               WHERE end_time is not null
            """
        )
        return self.ddb.sql("select * from all_raven_files")

    def write_raven_file(self, labels: list[RavenLabel], output_filename: str) -> None:
        raven_file_columns = [
            "Selection",
            "View",
            "Channel",
            "Begin Time (s)",
            "End Time (s)",
            "Low Freq (Hz)",
            "High Freq (Hz)",
            "Begin Date",
            "Begin Hour",
            "Begin Path",
            "Begin File",
            "File Offset (s)",
            "Date Check",
            "Time Check",
            "Score",
            "Tags",
            "Notes",
            "column17",
            "raven_filename",
            "Analyst",
            "Verification_2",
            "OLD-Selection",
            "Site",
            "Channel-OLD",
            "hour",
            "file date",
            "date(raven)",
            "Tag 1",
            "Tag 2",
            "fileDate",
            "Begin Path - old",
        ]

        with open(output_filename, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(raven_file_columns)
            for idx, row in enumerate(labels):
                # row = RavenLabel(*row)
                data = [
                    idx + 1,
                    "Spectrogram 1",  # "View",
                    "01",  # "Channel",
                    row.bt,  # "Begin Time (s)",
                    row.et,  # "End Time (s)",
                    row.lf,  # "Low Freq (Hz)",
                    row.hf,  # "High Freq (Hz)",
                    "01/01/1970",  # "Begin Date",
                    "00",  # "Begin Hour",
                    row.audio_file,  # "Begin Path",
                    row.audio_file,  # "Begin File",
                    row.bt,  # "File Offset (s)",
                    "01/01/1970",  # "Date Check",
                    "00:00:00.000",  # "Time Check",
                    row.score,  # "Score",
                    "Maybe Rumble",  # tags
                    "Maybe Rumble",  # Notes
                    "Maybe Rumble",  # column 17
                    "possible_rumbles.txt",
                    "",  # analyst
                    "",  # 'Verification_2',
                    "",  # 'OLD-Selection',
                    "",  # site
                    "",  # channel-old
                    "",  # hour
                    "",  # fileDate
                    "",  # date(raven)
                    "Maybe Rumble",  # tag1
                    "Maybe Rumble",  # tag2
                    "",  # "fileDate",
                    "",  # "Begin Path - old",
                ]
                writer.writerow(data)

    def get_all_labels_for_wav_file(self, wav_file: str) -> list[RavenLabel]:
        """
        Usage:
             lbls = rfh.get_all_labels_for_wav_file('CEB1_20111010_000000.wav')
             for idx,row in enumerate(lbls):
                 print(row)
                 if idx>3:
                     break

        """
        wav_file_without_path = os.path.basename(wav_file)
        wav_file_pattern = re.sub(r"dz(an)?", "dz%", wav_file_without_path)
        rs = self.ddb.execute(
            """
            SELECT * From all_raven_labels
            WHERE audio_file ilike ?
            ORDER BY start_time
        """,
            ["%" + wav_file_pattern + "%"],
        )
        return [RavenLabel(*row) for row in rs.fetchall()]

    ################################################################################
    # Training tools (not needed for inferrence)
    ################################################################################

    def get_files_from_test_folder(self) -> list[str]:
        ifs = self.get_interesting_files()
        return [f for f in ifs if "Test" in self.audio_filename_to_path[f]]

    def get_files_from_train_folder(self) -> list[str]:
        ifs = self.get_interesting_files()
        return [f for f in ifs if "Train" in self.audio_filename_to_path[f]]

    def find_candidate_audio_files(self, root_path: str) -> list[str]:
        pattern = root_path + "/**/*.wav"
        return glob.glob(pattern, recursive=True)

    def identify_useful_files(self) -> set[str]:
        audio_files = self.find_candidate_audio_files(self.root_path)
        self.files_from_raven: list[tuple[str]] = self.ddb.sql(
            'select distinct "Begin File" from all_raven_files'
        ).fetchall()
        files_from_raven_with_fixed_names = {re.sub("dzan", "dz", row[0]) for row in self.files_from_raven}
        self.audio_filename_to_path: dict[str, str] = {re.sub(".*/", "", f): f for f in audio_files}
        audio_files_without_path = {re.sub(".*/", "", f) for f in audio_files}
        return files_from_raven_with_fixed_names & audio_files_without_path

    def get_interesting_files(self) -> set[str]:
        return self.identify_useful_files()

    ## Helper functions to quickly get 1khz samples from files

    def load_entire_wav_file(self, wav_file_path: str, new_sr: int | None = None) -> torch.Tensor:
        import torchaudio.io as tai

        if not new_sr:
            msg = "load_entire_wav_file now requires a sr"

            class SampleRateError(Exception):
                """Error raised when sample rate is not provided."""

            raise SampleRateError(msg)
        streamer = tai.StreamReader(wav_file_path)
        int(streamer.get_src_stream_info(0).sample_rate)
        streamer.add_basic_audio_stream(
            stream_index=0,
            sample_rate=new_sr,
            frames_per_chunk=new_sr * 60 * 60 * 24,
        )
        results = []
        for idx, (chunk,) in enumerate(streamer.stream()):
            self.logger.debug("%d %s", idx, chunk.shape)
            results.append(chunk)
        return torch.cat(results)

    def get_cached_path(self, audio_filename: str, new_sr: int, prefix: str = "/tmp/downsampled_audio") -> str:  # noqa: S108
        os.makedirs(prefix, exist_ok=True)
        return f"{prefix}/{audio_filename}.{new_sr}.pt"

    def precompute_downsampled_pytorch_tensor(self, audio_filename: str, new_sr: int) -> None:
        cached_path = self.get_cached_path(audio_filename, new_sr)
        self.logger.info("resampling %s to %d at %s", audio_filename, new_sr, cached_path)
        source_path = self.audio_filename_to_path[audio_filename]
        audio_samples = self.load_entire_wav_file(source_path, new_sr=new_sr)
        audio_samples = audio_samples.flatten().to(torch.float16)
        torch.save(audio_samples, cached_path)

    def get_downsampled_tensor(self, audio_filename: str, start: float, duration: float, new_sr: int) -> torch.Tensor:
        cached_path = self.get_cached_path(audio_filename, new_sr)
        if not os.path.exists(cached_path):
            self.precompute_downsampled_pytorch_tensor(audio_filename, new_sr)
        y = torch.load(cached_path, mmap=True)
        return y[int(start * new_sr) : int((start + duration) * new_sr + 1)].clone().detach()

    ## End of Helper functions to quickly get 1khz samples from files

    ## NEGATIVE LABELS
    def get_negative_labels(self, positive_labels: list[RavenLabel]) -> list[RavenLabel]:
        """
        get a fragment with no lables half-way between the positive labels
        """
        pl = sorted(positive_labels, key=lambda x: x.bt)
        if len(pl) == 0:
            return []

        negative_labels = []

        negative_label = RavenLabel(
            0,
            min(pl[0].bt - 5, 60),
            25,
            100,
            min(pl[0].bt - 5, 60),
            pl[0].audio_file,
            "not a rumble",
            "not a rumble",
            "not a rumble",
            "not a rumble",
            -1,
            pl[0].ravenfile,
        )
        negative_labels.append(negative_label)

        # The constants 20, 130, etc are because many sections between close rumbles seem to also have rumbles
        for l1, l2 in zip(pl, pl[1:]):
            if l2.bt - l1.et < 20:  # noqa PLR2004
                continue
            if l2.bt - l1.et < 130:  # noqa PLR2004
                negative_label = RavenLabel(
                    l1.et + 5,
                    l2.bt - 5,
                    (l1.lf + l2.lf) / 2,
                    (l1.hf + l2.hf) / 2,
                    (l2.bt - 5) - (l1.et + 5),
                    l1.audio_file,
                    "not a rumble",
                    "not a rumble",
                    "not a rumble",
                    "not a rumble",
                    -1,
                    l1.ravenfile,
                )
                negative_labels.append(negative_label)
                continue
            if l2.bt - l1.et > 130:  # noqa PLR2004
                negative_label = RavenLabel(
                    l1.et + 5,
                    l1.et + 65,
                    (l1.lf + l2.lf) / 2,
                    (l1.hf + l2.hf) / 2,
                    60,
                    l1.audio_file,
                    "not a rumble",
                    "not a rumble",
                    "not a rumble",
                    "not a rumble",
                    -1,
                    l1.ravenfile,
                )
                negative_labels.append(negative_label)
                negative_label = RavenLabel(
                    l2.bt - 65,
                    l2.bt - 5,
                    (l1.lf + l2.lf) / 2,
                    (l1.hf + l2.hf) / 2,
                    60,
                    l1.audio_file,
                    "not a rumble",
                    "not a rumble",
                    "not a rumble",
                    "not a rumble",
                    -1,
                    l1.ravenfile,
                )
                negative_labels.append(negative_label)

            if l2.bt - l1.et > 130 * 2:
                negative_label = RavenLabel(
                    (l1.et + l2.bt) / 2 - 60,
                    (l1.et + l2.bt) / 2 + 60,
                    (l1.lf + l2.lf) / 2,
                    (l1.hf + l2.hf) / 2,
                    120,
                    l1.audio_file,
                    "not a rumble",
                    "not a rumble",
                    "not a rumble",
                    "not a rumble",
                    -1,
                    l1.ravenfile,
                )
                negative_labels.append(negative_label)

        return negative_labels


if this_should_be_a_unit_test := False:
    rfh = RavenFileHelper(root_path="/home/ron/proj/elephantlistening/data/Rumble")
    candidate_files = rfh.find_candidate_raven_files("/home/ron/proj/elephantlistening/data/Rumble")
    all_raven_files = rfh.all_raven_files_as_one_table(candidate_files)
    rfh.logger.debug(all_raven_files.columns)
    lbls = rfh.get_all_labels_for_wav_file("CEB1_20111010_000000.wav")
    for idx, row in enumerate(lbls):
        rfh.logger.debug(row)
        if idx > 3:  # noqa PLR2004
            break

    labels = [
        RavenLabel(
            bt=27.34,
            et=30.443,
            lf=22.8,
            hf=79.0,
            duration=3.1030000000000015,
            audio_file="CEB1_20111010_000000.wav",
            t1=None,
            t2=None,
            t3=None,
            notes=None,
            score=None,
            ravenfile="/home/ron/proj/elephantlistening/data/Rumble/Training/Clearings/rumble_clearing_00-24hr_56days.txt",
        ),
    ]
    rfh.write_raven_file(labels, "/tmp/rf.raven")  # noqa: S108
    rfh.load_one_raven_file("/tmp/rf.raven")  # noqa: S108
