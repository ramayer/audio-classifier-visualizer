from audio_classifier_visualizer.audio_file_visualizer import AudioFileVisualizer

import pytest

@pytest.mark.xfail(reason="Test file not yet added to repository")
#@pytest.mark.parametrize("audio_file", ["tests/data/test.wav"])
def test_can_create_audio_file_visualizer(audio_file: str = "tests/data/test.wav"):
    visualizer = AudioFileVisualizer(audio_file)
    assert visualizer is not None
