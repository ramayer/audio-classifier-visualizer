"""Parsers for label files -> list[LabelBox]. Pure functions, no plotting/audio coupling."""

from __future__ import annotations

import csv

from audio_classifier_visualizer.core.labels import LabelBox

# Common Raven Pro selection-table column name variants.
_START_COLS = ("Begin Time (s)", "begin_time", "start_time")
_END_COLS = ("End Time (s)", "end_time", "stop_time")
_LOW_FREQ_COLS = ("Low Freq (Hz)", "low_freq", "freq_lo")
_HIGH_FREQ_COLS = ("High Freq (Hz)", "high_freq", "freq_hi")
_TEXT_COLS = ("Annotation", "notes", "label", "tags")


def _first_present(row: dict, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def load_raven_selection_table(path: str) -> list[LabelBox]:
    """Parse a Raven Pro (or similarly-shaped tab-separated) selection table into LabelBoxes."""
    boxes: list[LabelBox] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            start = _first_present(row, _START_COLS)
            end = _first_present(row, _END_COLS)
            low = _first_present(row, _LOW_FREQ_COLS)
            high = _first_present(row, _HIGH_FREQ_COLS)
            if start is None or end is None:
                continue
            text = _first_present(row, _TEXT_COLS) or ""
            score_raw = row.get("Score") or row.get("score")
            boxes.append(
                LabelBox(
                    start_time=float(start),
                    end_time=float(end),
                    low_freq=float(low) if low is not None else 0.0,
                    high_freq=float(high) if high is not None else 0.0,
                    text=text,
                    score=float(score_raw) if score_raw else None,
                    source_file=path,
                )
            )
    return boxes
