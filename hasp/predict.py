import joblib
from pathlib import Path

DEFAULT_MODEL_PATH = Path(__file__).parent / 'model' / 'model.joblib'


class SoundClassifier():
    '''Binary classifier for audio files'''
    def __init__(self, model_path=None):
        if not model_path:
            model_path = DEFAULT_MODEL_PATH
        else:
            model_path = Path(model_path)
        self.model = joblib.load(model_path)

    def classify(self, audio_data):
        '''
        Classify a chunk of audio data as safe or dangerous
        parameters:
            audio_data: np.array of pcm float32 sample points
        returns:
           int {0, 1}
        '''
        pred = self.model.predict([audio_data])
        return pred[0]
