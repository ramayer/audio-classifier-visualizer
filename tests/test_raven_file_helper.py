from audio_classifier_visualizer.raven_file_helper import RavenFileHelper


def test_can_create_raven_file_helper():
    helper = RavenFileHelper()
    assert helper is not None
