from audio_classifier_visualizer.audio_file_visualizer import AudioFileVisualizer


def test_can_create_audio_file_visualizer():
    visualizer = AudioFileVisualizer()
    assert visualizer is not None
