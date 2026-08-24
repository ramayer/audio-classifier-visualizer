"""audio-classifier-visualizer: time-aligned visualization of long-audio classifier output.

Public API surface -- everything else under this package is an implementation
detail and may change between minor versions.
"""

from audio_classifier_visualizer.core.audio_signal import AudioSignal
from audio_classifier_visualizer.core.classifier_output import ClassifierOutput
from audio_classifier_visualizer.core.labels import LabelBox
from audio_classifier_visualizer.core.time_axis import TimeAxis
from audio_classifier_visualizer.features.stft import STFTFeatureExtractor
from audio_classifier_visualizer.features.wavelet import WaveletFeatureExtractor
from audio_classifier_visualizer.io.audio_loader import load_audio, load_audio_via_librosa
from audio_classifier_visualizer.io.labels_io import load_raven_selection_table
from audio_classifier_visualizer.render.matplotlib_renderer import MatplotlibRenderer
from audio_classifier_visualizer.render.spec import Track, VisualizationSpec
from audio_classifier_visualizer.visualization import AudioVisualization

try:
    # Optional: needs the 'audioset' extra (duckdb, pandas, einx, librosa).
    from audio_classifier_visualizer.helpers.audioset_helper import AudioSetHelper
except ImportError:
    AudioSetHelper = None

__version__ = "1.0.0"

__all__ = [
    "AudioSetHelper",
    "AudioSignal",
    "AudioVisualization",
    "ClassifierOutput",
    "LabelBox",
    "MatplotlibRenderer",
    "STFTFeatureExtractor",
    "TimeAxis",
    "Track",
    "VisualizationSpec",
    "WaveletFeatureExtractor",
    "load_audio",
    "load_audio_via_librosa",
    "load_raven_selection_table",
]
