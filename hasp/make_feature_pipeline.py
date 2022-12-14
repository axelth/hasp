from sklearn import set_config

set_config(display="diagram")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from librosa.feature import mfcc
import numpy as np


def samples_to_mean_mfcc(
    examples, sr=16000, n_fft=512, hop_length=128, fmin=0.0, fmax=8000, **kwargs
):
    """The librosa MFCC function can take either an array of audio samples using the keyword argument y=, or an array of spectrograms using the keyword argument S= as its input.
    However, the pipeline has no way of specifying which argument we are using, so we must define a wrapper function that passes the input in the right way"""
    # to prevent trying to take the lo
    return np.array(
        [
            mfcc(
                y=sample,
                sr=sr,
                n_fft=n_fft,
                n_mels=100,
                hop_length=128,
                fmin=0.0,
                fmax=8000,
                **kwargs
            ).mean(axis=1)
            for sample in examples
        ],
        dtype=np.float32,
    )


def make_feature_pipeline():
    """Make feature pipeline"""
    mean_mfcc_feat = FunctionTransformer(
        samples_to_mean_mfcc,
        kw_args={
            "sr": 16000,
            "n_mfcc": 20,
            "n_fft": 512,
            "hop_length": 128,
            "fmin": 0.0,
            "fmax": None,
        },
    )
    pipe = Pipeline([("mean_mfcc", mean_mfcc_feat), ("scaler", StandardScaler())])

    return pipe
