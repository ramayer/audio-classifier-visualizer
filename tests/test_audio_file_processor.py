from audio_classifier_visualizer.audio_file_processor import AudioFileProcessor

def test_can_create_audio_file_processor():
    processor = AudioFileProcessor()
    assert processor is not None 