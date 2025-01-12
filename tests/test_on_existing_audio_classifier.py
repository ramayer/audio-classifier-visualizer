
import torch
import elephant_rumble_inference as eri

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

aves_hubert_model = eri.AvesTorchaudioWrapper().to(DEVICE)
elephant_rumble_classifier = eri.ElephantRumbleClassifier().to('cpu')
aves_hubert_model.eval() # only train our layers - give up on fine-tuning AVES

#raven_file_helper     = eri.RavenFileHelper("./data/Rumble")
audio_file_visualizer = eri.AudioFileVisualizer()
audio_file_processor  = eri.AudioFileProcessor(aves_hubert_model,elephant_rumble_classifier)

def test_can_construct_classifier():
    assert aves_hubert_model is not None 
    assert elephant_rumble_classifier is not None 
    assert audio_file_visualizer is not None 
    assert audio_file_processor is not None 
    #assert raven_file_helper is not None 
