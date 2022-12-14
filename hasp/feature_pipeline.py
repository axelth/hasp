import numpy as np
from imblearn.pipeline import Pipeline as ImPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from librosa.feature import mfcc
from librosa.feature import delta
from sklearn import set_config

set_config(display="diagram")
from hasp.augmenting_oversampler import AugmentingRandomOversampler as OverSampler
from hasp.util import np_pad_wrapper


def samples_to_mean_mfcc(
        examples, sr=16000, n_fft=512, hop_length=128,
        fmin=0.0, fmax=8000, n_mels=100, delta1=False, delta2=False,
        **kwargs
) -> np.ndarray:
    """
    Motivation: The librosa MFCC function can take either an array of audio samples
    using the keyword argument y=, or an array of spectrograms using
    the keyword argument S= as its input.
    However, the pipeline has no way of specifying which argument we
    are using, so we must define a wrapper function that passes
    the input in the right way

    """
    features = []
    for sample in examples:
        mfcc_arr = mfcc(
                y=sample[sample > -2],
                sr=sr,
                n_fft=n_fft,
                n_mels=n_mels,
                hop_length=128,
                fmin=0.0,
                fmax=8000,
                **kwargs
            )
        feature = [mfcc_arr]

        if delta1:
            feature.append(delta(mfcc_arr, width=5, order=1))
        if delta2:
            feature.append(delta(mfcc_arr, width=5, order=2))

        features.append(np.hstack(
            [f.mean(axis=1) for f in feature]
        ))

    return np.vstack(features)


def samples_to_mfcc(
        examples, sr=16000, n_fft=512, hop_length=128,
        fmin=0.0, fmax=8000, n_mels=100, delta1=False, delta2=False,
        **kwargs
) -> np.ndarray:
    """
    compute the mfcc and optionally the delta and delta^2

    """
    features = []
    for sample in examples:
        mfcc_arr = mfcc(
                y=sample[sample > -2],
                sr=sr,
                n_fft=n_fft,
                n_mels=n_mels,
                hop_length=128,
                fmin=0.0,
                fmax=8000,
                **kwargs
            )
        feature = [mfcc_arr]

        if delta1:
            feature.append(delta(mfcc_arr, width=5, order=1))
        if delta2:
            feature.append(delta(mfcc_arr, width=5, order=2))

        features.append(np.hstack([feature]))

    return np.vstack(features)


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
            "n_mels": 100,
            "delta1": False,
            "delta2": False,
        },
    )
    pipe = Pipeline([("mean_mfcc", mean_mfcc_feat), ("scaler", StandardScaler())])

    return pipe


def make_oversampled_feature_pipeline():
    '''
    Return and imblearn.pipeline.Pipeline with empty slots for oversampler and
    estimator.
    After creating the pipeline, use
       pipe.set_params(over_sampler=.., estimator=..)
    to finish it.
    '''
    padder = FunctionTransformer(
        np_pad_wrapper,
        kw_args={'max_len': 64000}
    )

    mean_mfcc_feat = FunctionTransformer(
        samples_to_mean_mfcc,
        kw_args={
            "sr": 16000,
            "n_mfcc": 20,
            "n_fft": 512,
            "hop_length": 128,
            "fmin": 0.0,
            "fmax": None,
            "n_mels": 100,
            "delta1": False,
            "delta2": False,
        },
    )

    pipe = ImPipeline(
        [
            ('pad', padder),
            ("over_sampler", None),
            ("mean_mfcc", mean_mfcc_feat),
            ("scaler", StandardScaler()),
            ('estimator', None)
        ]
    )
    print('Use pipe.set_params(over_sampler=.., estimator=..) to finish the pipeline')
    return pipe
