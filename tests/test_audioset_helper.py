from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("einx")

from audio_classifier_visualizer.helpers.audioset_helper import AudioSetHelper


def test_create_label_arrays_rasterizes_intervals(tmp_path, monkeypatch):
    helper = AudioSetHelper(output_dir=str(tmp_path))
    fake_df = pd.DataFrame(
        [
            {"clip_id": "abc_0", "st": 0.0, "et": 1.0, "mid": "/m/x", "lbl": "Dog"},
            {"clip_id": "abc_0", "st": 5.0, "et": 6.0, "mid": "/m/y", "lbl": "Cat"},
        ]
    )
    monkeypatch.setattr(helper, "get_labels_for_a_clip", lambda clip_id: fake_df)

    labels, arr = helper.create_label_arrays("abc_0", bin_ms=100)

    assert labels == ["Dog", "Cat"]
    assert arr.shape == (100, 2)  # 10_000ms / 100ms bins x 2 labels
    assert arr[0, 0] == pytest.approx(1.0)  # Dog active in bin 0
    assert arr[50, 1] == pytest.approx(1.0)  # Cat active at t=5s -> bin 50
    assert arr[0, 1] == pytest.approx(0.0)


def test_create_label_arrays_handles_no_labels(tmp_path, monkeypatch):
    helper = AudioSetHelper(output_dir=str(tmp_path))
    monkeypatch.setattr(helper, "get_labels_for_a_clip", lambda clip_id: pd.DataFrame(columns=["st", "et", "lbl"]))
    labels, arr = helper.create_label_arrays("abc_0")
    assert labels == []
